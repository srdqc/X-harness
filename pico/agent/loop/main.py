"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from pico.agent.context import ContextBuilder
from pico.agent.loop.recovery import (
    POST_TOOL_NUDGE,
    RecoveryAction,
    RecoveryLimits,
    classify_empty_response,
)
from pico.agent.subagent import SubagentManager
from pico.agent.tools.ask_user import AskUserTool
from pico.agent.tools.base import ToolResult
from pico.agent.tools.execution import ToolExecution, ToolExecutionContext, ToolInvocation
from pico.agent.tools.file_search import FindTool, GrepTool
from pico.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from pico.agent.tools.message import MessageTool
from pico.agent.tools.registry import ToolRegistry
from pico.agent.tools.shell import ExecTool
from pico.agent.tools.skill import SkillReadTool
from pico.agent.tools.spawn import SpawnTool
from pico.agent.tools.web import WebFetchTool, WebSearchTool
from pico.call_efficiency.pricing import resolve_context_window
from pico.memory_engine.base import TokenBudget
from pico.memory_engine.consolidate.consolidator import MemoryConsolidator, MemoryStore
from pico.providers.base import ErrorClassification, LLMProvider, LLMResponse, ToolCallRequest
from pico.sandbox import SandboxConfig, SandboxExecutor, SandboxInitError, build_executor
from pico.session.manager import Session, SessionManager
from pico.spine.message import Media
from pico.spine.turn import Origin
from pico.tracing import semconv, trace
from pico.utils.helpers import estimate_prompt_tokens
from pico.utils.persisted_payload import sanitize_persisted_payload

# 刻意在 ``__init__`` 和 ``_assemble_context_messages`` 内延迟导入 ``pico.context_engine``，以打破运行时
# 循环导入：``pico.agent.__init__`` 急切加载 AgentLoop，而 ``pico.context_engine.curator``
# 又从 ``pico.agent.context`` 导入 ``ContextBuilder``。如果在此模块顶层导入，会重新进入
# 只初始化了一部分的包，并在 ``TurnContext`` 上抛出 ImportError。

if TYPE_CHECKING:
    from pico.agent.hook import CompositeHook
    from pico.agent.tools.base import Tool
    from pico.call_efficiency import CallEfficiency
    from pico.config.pico import (
        ContextConfig,
        MemoryConfig,
        RuntimeConfig,
        SkillForgeRouterConfig,
    )
    from pico.config.schema import ChannelsConfig, ExecToolConfig
    from pico.context_engine import ContextEngine
    from pico.context_engine.factory import ContextEngineFactory
    from pico.memory_engine.backend import MemoryBackend
    from pico.proactive_engine.schedulers.cron.service import CronService
    from pico.routing.router import ModelRouter
    from pico.sandbox.debug_server import SandboxDebugServer
    from pico.spine.runner import Drain, Emit, TurnOutcome
    from pico.spine.turn import TurnRequest
    from pico.token_wise.base import UsageSnapshot
    from pico.token_wise.registry import StrategyRegistry


@dataclass
class TurnOutcome:
    """Result of one ``_run_agent_loop`` turn beyond its text reply.

    ``status`` distinguishes a normal completion from a max-iteration
    interruption or an LLM error — so the caller never mistakes "ran out of
    budget" for "done" (Bug2 / decision B). ``checkpoint_id`` and
    ``edited_files`` carry the shadow-git snapshot info used to build the
    next turn's recovery prompt.
    """

    status: str = "completed"  # 可选 "completed" | "interrupted" | "error"
    checkpoint_id: str | None = None
    edited_files: list[str] = field(default_factory=list)
    error_category: str | None = None


class ProviderTurnError(RuntimeError):
    """Safe terminal error raised when a Provider cannot complete a Turn."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"provider_error:{category}")


# 迭代预算用尽后，让模型尽力收尾。本次调用不提供工具，因此提示词不能诱导
# 模型再次调用工具或提问，因为不会再有下一轮回答。
_MAX_ITER_SYNTHESIS_PROMPT = (
    "You've used up the tool-calling budget for this turn, so no tools are "
    "available now. Using only what you've already gathered, give your best "
    "final answer: summarize what you accomplished, deliver any partial "
    "results, and briefly note what's left undone. Do not ask questions — "
    "there is no further turn to answer them. Reply in the same language as "
    "the user's request (this instruction is in English, but it is not the "
    "conversation language)."
)

# 只在合成调用自身失败时返回，绝不让 Turn 没有响应。
_MAX_ITER_STATIC_FALLBACK = (
    "I reached the maximum number of tool call iterations ({n}) without "
    "completing the task. You can try breaking the task into smaller steps."
)

# 这些来源的内容并非用户输入，所以对应 Turn 跳过用户入站 Hook。
_SKIP_USER_INBOUND_ORIGINS = frozenset({Origin.CRON, Origin.SUBAGENT})

# 普通重试很可能清除这些失败标记，因此不得计入工具失败连续记录；
# 对会自愈的 429 发出提示只会制造噪声。
_TRANSIENT_FAILURE_MARKERS = (
    "429",
    "rate limit",
    "timed out",
    "timeout",
    "no healthy upstream",
    "502",
    "503",
)
# 成功但为空的结果表示工具正常运行，只是没找到内容。重复空搜索是合法探索，
# 而不是卡死的无效调用，因此不得计入失败连续记录。
_EMPTY_SUCCESS_MARKERS = ("no matches found", "no files found")


def _is_tool_failure(result: object) -> bool:
    if isinstance(result, ToolResult):
        return result.failed

    s = str(result).strip()
    match = re.search(r"(?:^|\n)Exit code:\s*(-?\d+)(?:\s|$)", s)
    if match:
        return match.group(1) != "0"
    if s.startswith("{"):
        try:
            payload = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(payload, dict) and payload.get("error"):
                return True
    low = s.lower()
    return (
        low.startswith("error:")
        or low.startswith("error ")
        or low.startswith("proxy error:")
        or low.startswith("(mcp tool call failed:")
        or low.startswith("(mcp tool call timed out ")
        or low.startswith("(mcp tool call was cancelled)")
    )


def _is_hard_tool_failure(result: object) -> bool:
    """True for a deterministic tool failure (recurs on an identical retry).

    False for success or a transient/retryable error. Used to decide whether a
    repeated identical tool call is a stuck loop worth breaking.
    """
    s = str(result)
    low = s.lower()
    if any(m in low for m in _TRANSIENT_FAILURE_MARKERS):
        return False
    if s.strip().rstrip(".").lower() in _EMPTY_SUCCESS_MARKERS:
        return False
    return _is_tool_failure(result)


def _loop_break_nudge(tool: str, n: int) -> str:
    """Injected when the same tool fails deterministically N times running, so
    the model stops repeating a dead approach instead of adapting."""
    return (
        f"[loop] `{tool}` has failed {n} times in a row with the same kind of error. "
        "Stop repeating it. If it is an external dependency (network/API/search), "
        "complete what you can offline from local data and report what stayed blocked. "
        "If it is a file or path error, re-examine the EXACT path before any retry — "
        "do not call it again unchanged. Otherwise change approach: a different tool, "
        "command, or strategy."
    )


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the spine
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    # 每个 Turn 在上下文溢出成为致命错误前，最多执行的紧急收缩次数。
    _MAX_COMPRESS_RETRIES = 2
    # 紧急收缩时保持完整的最新工具结果数；更旧结果被省略，
    # 因为它们的正文是 Turn 中途上下文增长的主体。
    _SHRINK_KEEP_RECENT_TOOL_RESULTS = 3
    # 工具失败循环打断：同一工具连续确定性失败达到此次数后发出提示；
    # 同时限制每个 Turn 的提示次数，避免提示本身形成循环。
    _LOOP_BREAK_THRESHOLD = 2
    _LOOP_BREAK_MAX = 2

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        sandbox_config: SandboxConfig | None = None,
        channels_config: ChannelsConfig | None = None,
        router: "ModelRouter | None" = None,
        strategies: "StrategyRegistry | None" = None,
        call_efficiency: "CallEfficiency | None" = None,
        skill_forge_config: Any = None,
        hooks: "CompositeHook | None" = None,
        now_fn: Callable | None = None,
        context_config: "ContextConfig | None" = None,
        runtime_config: "RuntimeConfig | None" = None,
        interactive: bool = True,
        jina_api_key: str | None = None,
        max_concurrent_subagents: int = 4,
        max_subagent_spawns_per_hour: int = 30,
        disabled_tools: list[str] | None = None,
        tool_search_config: Any = None,
        # 可选的插件 MemoryBackend。None 禁用后端回忆和存储，但保留 Session 和本地 Skill。
        backend: "MemoryBackend | None" = None,
        # 转发给 ``build_context_engine``，供 Memory 通道和本地 Skill 路由器使用。
        memory_config: "MemoryConfig | None" = None,
        skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
        # 已激活插件贡献的工具，由 CLI 通过 ``build_plugin_tools`` 构建，并在
        # ``_register_default_tools`` 中与内置工具一起注册。None 或空列表表示无插件工具，
        # 默认行为不变。
        plugin_tools: "list[Tool] | None" = None,
        empty_recovery: RecoveryLimits | None = None,
        context_engine_factory: "ContextEngineFactory | None" = None,
        state: Path | None = None,
    ):
        from pico.agent.hook import CompositeHook
        from pico.call_efficiency import CallEfficiency
        from pico.config.schema import ExecToolConfig
        from pico.token_wise.registry import StrategyRegistry

        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.state = state or workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        # 空响应恢复预算。None 表示使用已启用的默认值。
        self._recovery_limits = empty_recovery if empty_recovery is not None else RecoveryLimits()
        self.context_window_tokens = context_window_tokens
        self.brave_api_key = brave_api_key
        self.jina_api_key = jina_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        # 旧版注册表继续作为工具列表和基准扩展边界。
        self.strategies = strategies if strategies is not None else StrategyRegistry([])
        self.call_efficiency = call_efficiency or CallEfficiency.disabled()
        # 基准和模拟框架的伪时钟注入点。默认使用墙上时钟，不影响网关和 REPL 等生产路径。
        # 它同时用于会话项时间戳并传入 ContextBuilder，使 LLM 提示词中的 ``Current Time:``
        # 与持久化消息记录的时间保持同步。
        self._now_fn = now_fn or datetime.now

        self.backend: "MemoryBackend | None" = backend
        self.memory_enabled = backend is not None

        # 已激活插件贡献的工具，由 ``_register_default_tools`` 注册到 ToolRegistry。
        self.plugin_tools: "list[Tool]" = list(plugin_tools or [])

        self.context = ContextBuilder(
            workspace,
            state=self.state,
            skill_forge_config=skill_forge_config,
            llm_provider=self.provider,
            now_fn=now_fn,
        )
        self.sessions = session_manager or SessionManager(self.state)
        # 从注册表中排除的工具名称。在默认工具注册和 MCP 连接后均应用，因此可同时限制两组。
        # 供 BCP 等需要严格工具子集的评测框架使用。
        self._disabled_tools = set(disabled_tools or [])
        self._tool_search_config = tool_search_config
        self.tools = ToolRegistry()

        # Context Engine 是唯一的 ContextAssembler。在 self.tools 之后于此构建，使工厂能将
        # ``self.tools.get_definitions`` 捕获为延迟可调用对象；真正的工具注册表内容
        # 稍后由同一构造函数中的 ``_register_default_tools`` 填充。
        #
        # 延迟导入 ``pico.context_engine`` 的原因见模块顶部关于 ``pico.agent.__init__`` 循环导入的说明。
        if context_config is None:
            from pico.config.pico import ContextConfig

            context_config = ContextConfig()
        if context_engine_factory is None:
            from pico.context_engine import build_context_engine

            context_engine_factory = build_context_engine
        self.context_config = context_config

        self.context_engine: "ContextEngine" = context_engine_factory(
            workspace=workspace,
            config=context_config,
            builder=self.context,
            provider=self.provider,
            model=self.model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
            # 工厂使用这些参数组装统一的 Memory 和本地 Skill 通道。
            backend=backend,
            memory_config=memory_config,
            skill_forge_router_config=skill_forge_router_config,
            skill_forge_config=skill_forge_config,
        )

        # 运行时约束（第五支柱）。检查点受 policy 和 interactive 联合门控，见 ``_checkpoint_active``。
        # 门禁关闭时，Agent Loop 与基线字节级一致。
        if runtime_config is None:
            from pico.config.pico import RuntimeConfig

            runtime_config = RuntimeConfig()
        self.runtime_config = runtime_config
        self.interactive = interactive
        self._checkpoint = None
        if self._checkpoint_active(runtime_config.checkpoint.policy, interactive):
            from pico.agent.loop.checkpoint import CheckpointService

            try:
                self._checkpoint = CheckpointService(
                    workspace,
                    shadow_dir=runtime_config.checkpoint.shadow_dir,
                    state=self.state if self.state != workspace else None,
                )
            except ValueError as exc:
                # shadow_dir 无效（如 ``../escape`` 或绝对路径）时 CheckpointService 拒绝构建。
                # 不因配置笔误让整个 Agent 崩溃；记录日志并禁用安全网，使 Turn 仍可运行。
                logger.warning("runtime.checkpoint disabled — {}", exc)
        # Turn 因迭代上限中断时，保存 session_key -> {"checkpoint_id", "files"}；
        # 下一个 Turn 的恢复提示词会消费它。
        self._pending_recovery: dict[str, dict] = {}

        self._sandbox_config = sandbox_config
        self._owned_ids: set[str] = set()
        self.subagents = SubagentManager(
            provider=self.provider,
            workspace=workspace,
            state=self.state,
            model=self.model,
            brave_api_key=brave_api_key,
            jina_api_key=jina_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            sandbox_config=sandbox_config,
            owned_ids=self._owned_ids,
            max_concurrent=max_concurrent_subagents,
            max_spawns_per_hour=max_subagent_spawns_per_hour,
        )

        # 执行器此处只同步构建，虚拟机在 _start_executor() 中启动。
        self._executor: SandboxExecutor = build_executor(sandbox_config, workspace, self._owned_ids)
        self._executor_stack: AsyncExitStack | None = None
        self._executor_started: bool = False
        self._executor_start_lock = asyncio.Lock()
        self._debug_server: SandboxDebugServer | None = None

        self.router = router
        self.enable_personalization = False  # 通过 configure_personalization() 设置
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._processing_lock = asyncio.Lock()
        # 每个已分派 Turn 结束后触发，无论成功、错误还是取消。回调必须低成本且不得抛错。
        self.on_turn_complete: list[Callable[[], None]] = []
        self.memory_consolidator = MemoryConsolidator(
            workspace=self.state,
            provider=self.provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
        )

        self._personalization_tasks: set[asyncio.Task[Any]] = set()
        self._personalization_closed = False
        self._close_lock = asyncio.Lock()
        self._closed = False

        # B-3 阶段已移除 L4 外观（``DefaultMemoryEngine`` / ``MemoryEngine`` 抽象基类）。
        # AgentLoop 现在直接持有底层子系统：
        #
        # - ``self.memory_consolidator`` 负责 Markdown 压缩策略，并拥有它构建的 ``MemoryStore``；
        #   需要时通过 ``self.memory_consolidator.store`` 访问。
        # - ``self.context.skills`` 是 :class:`LocalSkillCatalog`，负责常驻 Skill 和 ``# Skills`` 渲染路径。
        #   由 ``context_engine.factory`` 组装的 SkillForgeRouter 栈拥有检索职责。

        # AgentHook 生命周期链。评测 Hook 和调用方 Hook 共享同一个有序接口，
        # 无需按功能定制回调适配器。
        self.hooks: "CompositeHook" = CompositeHook()
        if hooks is not None:
            self.hooks.extend(hooks)

        self._register_default_tools()
        self._apply_disabled_tools()

    def _apply_disabled_tools(self) -> None:
        """Unregister tools whose names appear in ``tools.disabled_tools``.

        Run after :meth:`_register_default_tools` (here) and after MCP connect
        (see :meth:`_connect_mcp`) so the blacklist can cover either group.
        Silent on misses — eval configs commonly carry an over-broad list
        that's a no-op for tools that weren't registered in this build.
        """
        if not self._disabled_tools:
            return
        for name in self._disabled_tools:
            if self.tools.has(name):
                self.tools.unregister(name)

    def configure_personalization(self, enable: bool) -> None:
        """Global switch for the 4-step personalization flow (PAHF-inspired).

        When enabled, each message goes through:
          Step 1 - classify:          classify() — does this request need a preference question?
          Step 2 - pre-action interaction: ask one question if needed, extract and store the answer
          Step 3 - execute:           normal agent loop (unchanged)
          Step 4 - post-action learn: post_learn() runs in background after every response

        Disabled by default. Enable via config: agents.defaults.enable_personalization: true
        """
        self.enable_personalization = bool(enable and self.memory_enabled)
        logger.info("Personalization flow: {}", "enabled" if self.enable_personalization else "disabled")

    def _start_personalization_task(self, factory: Callable[[], Awaitable[Any]]) -> None:
        if self._personalization_closed:
            return
        task = asyncio.create_task(factory())
        self._personalization_tasks.add(task)
        task.add_done_callback(self._personalization_tasks.discard)

    def begin_close(self) -> None:
        """Seal and cancel background personalization without yielding."""
        if self._personalization_closed:
            return
        self._personalization_closed = True
        for task in tuple(self._personalization_tasks):
            task.cancel()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SkillReadTool(self.context.skills))
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
                executor=self._executor,
            )
        )
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(api_key=self.jina_api_key, proxy=self.web_proxy))
        self.tools.register(MessageTool())
        self.tools.register(SpawnTool(manager=self.subagents))
        # QuestionBroker 按传输层为单例；当传输层（TUI RPC 服务器或网关 hub）存在后，
        # 通过 set_broker 延迟绑定。
        self.tools.register(AskUserTool())
        if self.cron_service:
            # 延迟导入：CronTool 所在模块会导入 pico.agent.tools.base，触发 pico.agent.__init__，
            # 后者又导入当前循环模块。在函数作用域导入可打破循环，因为执行
            # _register_default_tools 时 loop.py 已完全加载。
            from pico.proactive_engine.schedulers.cron.tool import CronTool

            self.tools.register(CronTool(self.cron_service))

        # 插件贡献的工具最后注册，使插件在有意提供同名工具时能覆盖内置实现。
        # 随后仍会运行 ``_apply_disabled_tools``，因此任何工具都可被移除。
        for tool in self.plugin_tools:
            self.tools.register(tool, replace=True)

        # 渐进式工具披露最后注册，使其搜索目录覆盖上方全部内置和插件工具。
        # MCP 工具稍后在 ``_connect_mcp`` 中加入；策略每个 Turn 都重读注册表，因此能自动获取。
        cfg = self._tool_search_config
        if cfg is not None and cfg.enabled:
            from pico.agent.tools.tool_search import (
                DEFAULT_ALWAYS_VISIBLE,
                ToolCallTool,
                ToolSearchController,
                ToolSearchStrategy,
                ToolSearchTool,
            )

            always = set(DEFAULT_ALWAYS_VISIBLE) | set(cfg.always_visible)
            self.tool_search_controller = ToolSearchController(
                self.tools,
                always_visible=always,
                search_result_limit=cfg.search_result_limit,
            )
            self.tools.register(ToolSearchTool(self.tool_search_controller))
            self.tools.register(ToolCallTool(self.tool_search_controller))
            # ``first=True`` 表示在 CacheOptimizer 用 ``cache_control`` 标记最后一个工具前先过滤列表；
            # 否则已标记的工具可能被过滤，导致缓存断点丢失。
            self.strategies.register(
                ToolSearchStrategy(
                    self.tool_search_controller,
                    compaction_threshold=cfg.compaction_threshold,
                ),
                first=True,
            )

    # ── 上下文引擎辅助方法 ─────────────────────────────────────────────

    def _context_messages_for_session(self, session: Session) -> list[dict[str, Any]]:
        """Return the candidate message view owned by the active context engine.

        Curator (``owns_compaction=True``) wants the full append-only log so
        it can decide what to archive itself; Legacy wants the post-consolidation
        slice to match the pre-Curator behavior exactly.
        """
        if self.context_engine.owns_compaction:
            return list(session.messages)
        return session.get_history(max_messages=0)

    def _make_token_budget(self, selected_skills: list[Any] | None = None) -> TokenBudget:
        """Compute a conservative per-turn prompt budget for the active engine."""
        reserved_output = int(getattr(getattr(self.provider, "generation", None), "max_tokens", 4096) or 4096)
        tool_tokens = estimate_prompt_tokens([], self.tools.get_definitions())
        system_prompt = self.context.build_system_prompt(
            selected_skills,
            include_memory=self.memory_enabled,
        )
        system_tokens = estimate_prompt_tokens([{"role": "system", "content": system_prompt}])
        available_history = max(
            0,
            self.context_window_tokens - reserved_output - tool_tokens - system_tokens,
        )
        return TokenBudget(
            context_length=self.context_window_tokens,
            reserved_output=reserved_output,
            reserved_tools=tool_tokens,
            reserved_system=system_tokens,
            available_history=available_history,
        )

    async def _select_skills_for_turn(
        self,
        current_message: str,
        history: list[dict],
    ) -> list[Any] | None:
        """No host-side pre-selection — the engine's SkillForgeRouter owns it.

        The unified engine selects + renders skills internally and
        surfaces ``injected_skill_ids`` via per-Turn assembly metadata.
        No SkillMeta list flows through this path.
        """
        return None

    async def _assemble_context_messages(
        self,
        *,
        session: Session,
        session_key: str,
        current_message: str,
        media: list[str | Media] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        selected_skills: list[Any] | None = None,
        metadata_sink: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the active context engine for the main-agent message window."""
        from pico.context_engine import TurnContext  # 延迟导入，原因见模块说明

        session_messages = self._context_messages_for_session(session)
        assembled = await self.context_engine.assemble(
            session_key,
            session_messages,
            self._make_token_budget(selected_skills),
            turn=TurnContext(
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
                selected_skills=selected_skills,
            ),
        )
        if metadata_sink is not None:
            metadata_sink.update(assembled.metadata or {})
        messages = assembled.messages
        self._inject_recovery_block(session_key, messages)
        return messages

    @staticmethod
    def _checkpoint_active(policy: str, interactive: bool) -> bool:
        """Resolve ``runtime.checkpoint.policy`` against the call-site's
        ``interactive`` signal. ``"interactive"`` (the default) skips the
        snapshot for one-shot ``-m`` invocations — those have no "next turn"
        to inject recovery into, so paying the snapshot cost there is just
        deadweight. ``"always"`` opts in regardless; ``"never"`` opts out
        regardless."""
        if policy == "never":
            return False
        if policy == "always":
            return True
        return interactive  # policy 为 interactive

    def _stash_recovery(self, session_key: str, outcome: "TurnOutcome") -> None:
        """Remember an interrupted turn's snapshot so the next turn in this
        session gets a recovery prompt. No-op unless checkpoint is enabled
        and the turn was actually interrupted with something to recover.

        Status filter is intentional: only ``"interrupted"`` triggers a
        recovery prompt. ``"error"`` turns still get a per-turn shadow
        commit (useful for audit), but they don't usually have a partial-
        edits trajectory to resume (provider 400 etc.) and surfacing
        "Files modified last turn" for them would be misleading.
        """
        if self._checkpoint is None or outcome.status != "interrupted":
            return
        if outcome.edited_files or outcome.checkpoint_id:
            self._pending_recovery[session_key] = {
                "checkpoint_id": outcome.checkpoint_id,
                "files": outcome.edited_files,
            }

    def _inject_recovery_block(self, session_key: str, messages: list[dict]) -> None:
        """Prepend a recovery notice to the current user message when the
        previous turn for this session was interrupted. Consumed once on
        successful injection; if the current message's content has an
        unexpected shape (None / dict / etc.) the pending entry is kept so
        a later assembly with a normal content can still inject it."""
        recovery = self._pending_recovery.get(session_key)
        if not recovery or not messages:
            return
        last = messages[-1]
        if last.get("role") != "user":
            # 最后一条消息不是用户 Turn 时，保持恢复待处理，使下次以用户消息结尾的组装注入它。
            return
        content = last.get("content")
        files = recovery.get("files") or []
        cid = recovery.get("checkpoint_id")
        lines = ["[Recovery — the previous turn was interrupted before finishing]"]
        if files:
            lines.append("Files modified last turn: " + ", ".join(files))
        if cid:
            lines.append(f"Checkpoint: {cid}")
        lines.append("Verify the current state of these files before continuing.")
        block = "\n".join(lines)
        # 先修改、后弹出，从调用方视角看是原子操作。如果无法安全写入形状未知的 ``content``，
        # 恢复保持待处理，而不是被静默丢弃。
        if isinstance(content, str):
            last["content"] = f"{block}\n\n{content}"
        elif isinstance(content, list):
            last["content"] = [{"type": "text", "text": block}] + content
        else:
            return  # 内容形状异常时保持待恢复状态
        self._pending_recovery.pop(session_key, None)

    @trace.instrument("memory.store", extract=semconv.memory_store)
    async def _dispatch_backend_store(
        self,
        session_key: str,
        messages_slice: list[dict],
    ) -> None:
        """AG-1: forward a turn's messages to the plugin :class:`MemoryBackend`.

        Third peer step in the after-turn pipeline alongside
        ``context_engine.after_turn`` (engine-side bookkeeping) and
        ``memory.maybe_consolidate`` (Pico compaction). When no
        backend was wired (``self.backend is None``), this is a no-op so
        legacy callsites that never registered a plugin behave
        identically to pre-AG-1.

        Backend failures propagate after the append-only Session save, so the
        Turn cannot be reported as successful when indexing failed.
        """
        if self.backend is None or not messages_slice:
            return
        await self.backend.store(
            session_key,
            sanitize_persisted_payload(messages_slice),
        )

    def _collect_injected_skill_ids(
        self,
        selected: list[Any] | None,
    ) -> list[str]:
        """Combine selector top-K + always-skills into a deduplicated id list.

        ``selected`` is the :class:`SkillMeta` list returned by the
        retrieval selector for this turn (or ``None`` when the selector
        is disabled / returned empty). always-skills are pulled from
        :class:`LocalSkillCatalog` since they are unconditionally rendered
        regardless of the selector's output.

        Returns ids canonicalized to ``{source}/{stable_key}`` form.
        Different SkillMeta producers populate ``meta.id`` inconsistently,
        so this function normalizes them to a single shape for Turn evidence.
        """
        skills_svc = getattr(self.context, "skills", None)
        if skills_svc is None:
            return []

        seen: set[str] = set()
        ids: list[str] = []

        def _add(meta: Any) -> None:
            src = getattr(meta, "source", None)
            mid = getattr(meta, "id", None)
            if not src or not mid:
                return
            canonical = mid if "/" in mid else f"{src}/{mid}"
            if canonical not in seen:
                seen.add(canonical)
                ids.append(canonical)

        for meta in selected or []:
            _add(meta)
        try:
            always = skills_svc.get_always_skills()
        except Exception:
            always = []
        for meta in always:
            _add(meta)
        return ids

    async def _start_executor(self) -> None:
        """Idempotent: start the sandbox executor once before first use."""
        async with self._executor_start_lock:
            if self._executor_started:
                return
            stack = AsyncExitStack()
            try:
                await stack.__aenter__()
                await stack.enter_async_context(self._executor)
            except Exception:
                await stack.aclose()
                raise
            self._executor_stack = stack
            self._executor_started = True

    async def _start_debug_server(self) -> None:
        """Start the sandbox debug socket server if debug mode is enabled."""
        cfg = self._sandbox_config
        if cfg is None or not cfg.debug.enabled:
            return
        if cfg.backend == "none":
            logger.warning(
                "sandbox.debug.enabled=true is ignored because backend='none' (no boxlite runtime is active)"
            )
            return
        try:
            from pico.config.paths import get_data_dir
            from pico.sandbox.debug_server import SandboxDebugServer

            socket_path = SandboxDebugServer.resolve_socket_path(cfg.debug.socket, get_data_dir())
            server = SandboxDebugServer(
                socket_path=socket_path,
                owned_ids=self._owned_ids,
                max_message_bytes=cfg.debug.max_message_bytes,
            )
            await server.start()
            self._debug_server = server
        except Exception as exc:
            # 用户已明确选择调试模式；此处静默失败会让用户在稍后看到 `pico sandbox`
            # 报“套接字不存在”时困惑。需明确记录原因。
            logger.error("Failed to start sandbox debug server: %s", exc)

    async def close_executor(self) -> None:
        """Tear down the sandbox executor."""
        if self._debug_server is not None:
            try:
                await self._debug_server.stop()
            except Exception as exc:
                logger.warning("Error stopping sandbox debug server: %s", exc)
            self._debug_server = None
        if self._executor_stack:
            try:
                await self._executor_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._executor_stack = None
        self._executor_started = False

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        # 在第一个 await 之前同步设置标志。asyncio 单线程执行，此处不会发生上下文切换，
        # 该互斥模式无需锁。
        self._mcp_connecting = True
        try:
            await self._start_executor()  # MCP 服务器连接前，执行器必须已启动
            from pico.agent.tools.mcp import connect_mcp_servers

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(
                self._mcp_servers,
                self.tools,
                self._mcp_stack,
                executor=self._executor,
            )
            # 重新应用黑名单：MCP 服务器可能注册也出现在 ``disabled_tools`` 中的工具名，
            # 例如 ``mcp_<server>_search``。
            self._apply_disabled_tools()
            self._mcp_connected = True
            self._mcp_connecting = False
        except Exception:
            # 重置进行中标志，使后续调用可重试。
            self._mcp_connecting = False
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
            raise

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None, session_key: str | None = None
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if not hasattr(tool, "set_context"):
                    continue
                if name == "message":
                    tool.set_context(channel, chat_id, message_id)
                elif name == "spawn":
                    tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")
                else:
                    tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _build_usage_snapshot(response, model: str, session_key: str) -> "UsageSnapshot":
        """Build the historical TokenWise view through canonical normalization."""
        from pico.call_efficiency import CallEfficiency

        return (
            CallEfficiency.disabled()
            .record(
                response,
                requested_model=model,
                session_key=session_key,
            )
            .to_legacy_snapshot()
        )

    @trace.instrument("llm.call", extract=semconv.llm_call_stream)
    async def _llm_call_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream LLM response via ``provider.chat_stream`` + accumulate to LLMResponse.

        Per design.md §D3: when a turn caller wires ``on_token_delta``, AgentLoop
        diverts to this helper instead of ``chat_with_retry``. Each non-empty
        content chunk fires the callback; tool_call fragments are merged
        positionally; the final response object is shape-compatible with what
        ``chat()`` would have returned.

        v0.1 first-cut tool-call merge: assumes one tool call per position,
        fragments arrive in order, ``id`` / ``function.name`` appear in the
        first fragment, ``function.arguments`` is the concatenation of
        per-fragment arguments strings. Multi-tool / out-of-order merging is
        a v0.2 ask.

        No retry on transient errors in v0.1 stream mode — adding retry to
        a partially-streamed call requires either restarting from scratch
        (wasteful) or resume-from-offset (provider-specific). Deferred.
        """
        content_buf: list[str] = []
        reasoning_buf: list[str] = []
        tool_call_slots: list[dict[str, Any]] = []
        final_usage: dict[str, Any] | None = None
        final_finish_reason: str | None = None
        final_error_classification: ErrorClassification | None = None
        actual_model: str | None = None
        cache_policy: str | None = None
        call_record = None

        async for delta in self.provider.chat_stream(
            messages=messages,
            tools=tools,
            model=model,
        ):
            delta_finish_reason = getattr(delta, "finish_reason", None)
            if delta_finish_reason is not None:
                final_finish_reason = delta_finish_reason
            delta_error_classification = getattr(delta, "error_classification", None)
            if delta_error_classification is not None:
                final_error_classification = delta_error_classification
            delta_model = getattr(delta, "model", None)
            if delta_model:
                actual_model = delta_model
            delta_cache_policy = getattr(delta, "cache_policy", None)
            if delta_cache_policy:
                cache_policy = delta_cache_policy
            delta_call_record = getattr(delta, "call_record", None)
            if delta_call_record is not None:
                call_record = delta_call_record
            is_error_delta = delta_finish_reason == "error"
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_buf.append(reasoning_delta)
                if on_reasoning_delta is not None and not is_error_delta:
                    await on_reasoning_delta(reasoning_delta)
            if delta.content:
                content_buf.append(delta.content)
                if on_token_delta is not None and not is_error_delta:
                    await on_token_delta(delta.content)
            if delta.tool_call_delta:
                _merge_tool_call_fragments(
                    tool_call_slots,
                    delta.tool_call_delta,
                )
            if delta.usage is not None:
                final_usage = delta.usage

        tool_calls = _finalize_tool_calls(tool_call_slots)
        finish_reason = final_finish_reason or ("tool_calls" if tool_calls else "stop")

        return LLMResponse(
            content="".join(content_buf),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage or {},
            reasoning_content="".join(reasoning_buf) or None,
            error_classification=final_error_classification,
            model=actual_model or model,
            cache_policy=cache_policy,
            call_record=call_record,
        )

    @classmethod
    def _emergency_shrink(cls, messages: list[dict]) -> tuple[list[dict], int]:
        """Elide the bodies of older tool-result messages to fit a tighter window.

        Mid-turn context overflow is almost always accumulated tool output, so
        replacing the content of all but the most recent few ``role="tool"``
        messages with a short placeholder frees the most tokens while keeping
        system / user / assistant reasoning intact. Deterministic, no extra LLM
        call. Returns ``(new_messages, num_elided)``; ``num_elided == 0`` means
        there was nothing worth eliding (caller should not bother retrying).
        """
        placeholder = "[earlier tool output elided to fit the context window]"
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_idxs) <= cls._SHRINK_KEEP_RECENT_TOOL_RESULTS:
            return messages, 0
        elide = set(tool_idxs[: -cls._SHRINK_KEEP_RECENT_TOOL_RESULTS])
        shrunk: list[dict] = []
        elided = 0
        for i, m in enumerate(messages):
            if i in elide and m.get("content") and m.get("content") != placeholder:
                clean = dict(m)
                clean["content"] = placeholder
                shrunk.append(clean)
                elided += 1
            else:
                shrunk.append(m)
        return shrunk, elided

    async def _synthesize_final_on_exhaustion(
        self,
        messages: list[dict],
        model: str | None,
        fallback_models: list[str] | None,
        session_key: str = "",
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """One tools-disabled LLM call to wrap up after the iteration budget runs out.

        Instead of returning a canned apology, ask the model to summarize what
        it accomplished and deliver its best partial answer. Tools are withheld
        (``tools=None``) so it cannot start another tool call — or an
        ``ask_user`` — at the cliff edge. Falls back to a static message if the
        call errors or comes back empty, so the turn is never left silent.

        When the turn caller wired streaming callbacks, this synthesized reply
        must stream too — otherwise it never reaches a streaming outlet: the
        run_turn boundary only emits a closing ``Text`` when nothing streamed,
        so a non-streamed wrap-up after an already-streamed turn gets dropped.
        """
        synth_messages = messages + [{"role": "user", "content": _MAX_ITER_SYNTHESIS_PROMPT}]
        try:
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=synth_messages,
                    tools=None,
                    model=model,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=synth_messages,
                    tools=None,
                    model=model,
                    fallback_models=fallback_models,
                )
            if response.call_record is None:
                response.call_record = self.call_efficiency.record(
                    response,
                    requested_model=model or self.model,
                    session_key=session_key,
                    cache_policy=response.cache_policy,
                )
            text = self._strip_think(response.content)
            if response.finish_reason != "error" and text:
                return text
            logger.warning(
                "Max-iter synthesis returned no usable content (finish_reason={})",
                response.finish_reason,
            )
        except Exception as exc:
            logger.warning("Max-iter synthesis call failed: {}", exc)
        fallback = _MAX_ITER_STATIC_FALLBACK.format(n=self.max_iterations)
        # 流式成功路径已通过 ``on_token_delta`` 交付文本，此回退路径则没有。因此也要将它推入流；
        # 否则一旦已有内容流出，run_turn 边界会抑制结尾 ``Text``，导致流式出口丢失回退文本。
        if on_token_delta is not None:
            await on_token_delta(fallback)
        return fallback

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session_key: str | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        injected_skill_ids: list[str] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        usage_sink: dict[str, Any] | None = None,
        drain: Drain | None = None,
        origin: Origin | None = None,
    ) -> tuple[str | None, list[str], list[dict], TurnOutcome]:
        """Run the agent iteration loop.

        ``drain``, when wired, is called at the top of each iteration to pull
        any user messages injected mid-turn (BusyPolicy.INJECT) and merge them
        as user turns before the next LLM call.

        ``session_key`` attributes usage and Checkpoint evidence to the
        conversation when the loop is called through a host.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        session_key = session_key or ""
        effective_model = model or self.model

        # 记录 Turn 是正常退出还是因迭代上限中断。下游只读取 ``status``，
        # 用于标记影子 Git 提交并写入 ``TurnOutcome``。
        status = "completed"
        error_category: str | None = None

        # 上下文溢出恢复：限制紧急收缩次数，避免省略后仍溢出的 Turn 永久循环。
        compress_retries = 0
        # 工具失败循环打断：跨迭代跟踪同一工具的连续硬失败；
        # 每个新连续段提示一次，并按 Turn 限制总数。
        loop_fail_tool: str | None = None
        loop_fail_streak = 0
        loop_nudges = 0
        # 空响应恢复状态属于当前 Turn。AgentLoop 是跨会话共享的长生命周期单例，
        # 实例级计数器会跨 Turn 泄漏；此处重置可为每个 Turn 提供干净预算。
        prev_had_tool_calls = False
        post_tool_nudges = 0
        prefill_retries = 0
        empty_retries = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(
                "Iteration {}/{} model={}",
                iteration,
                self.max_iterations,
                effective_model,
            )

            # 在本次迭代的 LLM 调用前合并所有通过 BusyPolicy.INJECT 注入的用户消息。
            # 携带媒体的注入会在文本中保留文件路径，避免内容静默丢失。
            if drain is not None:
                for inj in drain():
                    inj_text = inj.text or ""
                    inj_paths = [m.path for m in inj.media]
                    if inj_paths:
                        prefix = inj_text + "\n" if inj_text else ""
                        inj_text = f"{prefix}[injected message; attached files: {', '.join(inj_paths)}]"
                    if inj_text:
                        messages.append({"role": "user", "content": inj_text})
                        logger.info("inject: merged a mid-turn user message")

            tool_defs = self.tools.get_definitions()

            # 先运行历史策略，使工具过滤先于缓存规划。
            call_messages, call_tools, call_model = await self.strategies.before_llm_call(
                messages,
                tool_defs,
                effective_model,
            )
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    fallback_models=fallback_models,
                )
            call_record = response.call_record
            if call_record is None:
                call_record = self.call_efficiency.record(
                    response,
                    requested_model=call_model,
                    session_key=session_key,
                    cache_policy=response.cache_policy,
                )
                response.call_record = call_record
            usage_snapshot = call_record.to_legacy_snapshot()
            await self.strategies.after_llm_call(
                {
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
                usage_snapshot,
            )
            # tui-chat L2-A 线路：流式调用方（turn.* 处理器）可能需要用最后一次迭代用量
            # 按 CAP-CHAT-1 形状填充 `message.complete.payload.usage`。应使用线上合约 UsageSnapshot
            # 的 prompt_tokens、completion_tokens 和 total_tokens，而非带模型、缓存和成本字段的
            # Agent 内部快照。
            if usage_sink is not None and response.usage:
                prompt_tokens = int(response.usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(response.usage.get("completion_tokens", 0) or 0)
                # LiteLLM 信息滞后时，从模型提供商表获取真实窗口；否则使用配置默认值。
                context_max = (
                    resolve_context_window(
                        call_record.accounting_model,
                        allow_litellm_import=False,
                    )
                    or self.context_window_tokens
                )
                context_used = prompt_tokens + completion_tokens
                usage_sink.clear()
                usage_sink["prompt_tokens"] = prompt_tokens
                usage_sink["completion_tokens"] = completion_tokens
                usage_sink["total_tokens"] = int(response.usage.get("total_tokens", 0) or 0)
                if usage_snapshot.estimated_cost_usd is not None:
                    usage_sink["cost_usd"] = usage_snapshot.estimated_cost_usd
                else:
                    usage_sink.pop("cost_usd", None)
                usage_sink["context_max"] = context_max
                usage_sink["context_used"] = context_used
                usage_sink["context_percent"] = round(100 * context_used / context_max) if context_max else 0

                # 上下文窗口溢出恢复：结构化分类器标记 should_compress。更小窗口无济于事，
                # 但省略大量累积工具输出有效。就地收缩并重试当前迭代，而不暴露为致命错误；
                # 重试次数受限。
            cls_ = response.error_classification
            if (
                response.finish_reason == "error"
                and cls_ is not None
                and cls_.should_compress
                and compress_retries < self._MAX_COMPRESS_RETRIES
            ):
                shrunk, elided = self._emergency_shrink(messages)
                if elided > 0:
                    messages = shrunk
                    compress_retries += 1
                    iteration -= 1  # 溢出的调用没有执行工作，不计入迭代次数
                    logger.warning(
                        "Context overflow; elided {} old tool result(s), retrying ({}/{})",
                        elided,
                        compress_retries,
                        self._MAX_COMPRESS_RETRIES,
                    )
                    continue

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                invocations: list[ToolInvocation] = []
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    invocations.append(
                        ToolInvocation(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            context=ToolExecutionContext(
                                call_id=tool_call.id,
                                session_key=session_key,
                                iteration=iteration,
                                origin=origin.value if origin is not None else None,
                            ),
                        )
                    )

                async def _tool_started(invocation: ToolInvocation) -> None:
                    if on_tool_event is not None:
                        nested = invocation.context.parent_call_id is not None
                        await on_tool_event(
                            "start",
                            {
                                "tool_call_id": (
                                    invocation.context.parent_call_id if nested else invocation.context.call_id
                                ),
                                "name": "tool_call" if nested else invocation.name,
                                "arguments": (
                                    {"name": invocation.name, "arguments": invocation.arguments}
                                    if nested
                                    else invocation.arguments
                                ),
                                "target_call_id": invocation.context.call_id if nested else None,
                                "target_name": invocation.name if nested else None,
                                "target_arguments": invocation.arguments if nested else None,
                            },
                        )

                async def _tool_completed(execution: ToolExecution) -> None:
                    invocation = execution.invocation
                    result = execution.result
                    result_str = str(result)
                    preview = result_str.replace("\n", " ")[:200]
                    logger.info(
                        "Tool result: {} duration={}ms result={}",
                        invocation.name,
                        int(execution.duration_ms),
                        preview,
                    )
                    if on_tool_event is not None:
                        nested = invocation.context.parent_call_id is not None
                        await on_tool_event(
                            "complete",
                            {
                                "tool_call_id": (
                                    invocation.context.parent_call_id if nested else invocation.context.call_id
                                ),
                                "name": "tool_call" if nested else invocation.name,
                                "result_preview": preview,
                                "truncated": len(result_str) > 200,
                                "failed": _is_tool_failure(result),
                                "target_call_id": invocation.context.call_id if nested else None,
                                "target_name": invocation.name if nested else None,
                                "target_arguments": invocation.arguments if nested else None,
                                "duration_ms": execution.duration_ms,
                            },
                        )

                executions = await self.tools.execute_many(
                    invocations,
                    on_start=_tool_started,
                    on_complete=_tool_completed,
                )

                for tool_call, execution in zip(response.tool_calls, executions, strict=True):
                    result = execution.result
                    messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
                    # 跟踪同一工具的连续确定性失败；排除可通过重试清除的短暂错误。
                    if _is_hard_tool_failure(result):
                        if tool_call.name == loop_fail_tool:
                            loop_fail_streak += 1
                        else:
                            loop_fail_tool, loop_fail_streak = tool_call.name, 1
                    else:
                        loop_fail_tool, loop_fail_streak = None, 0

                # 失败循环打断：同一工具连续确定性失败 `threshold` 次后，
                # 向最后一个工具结果附加改变方法的提示，让模型停止重复无效调用。
                if (
                    loop_fail_streak >= self._LOOP_BREAK_THRESHOLD
                    and loop_nudges < self._LOOP_BREAK_MAX
                    and messages
                    and messages[-1].get("role") == "tool"
                ):
                    loop_nudges += 1
                    messages[-1]["content"] = (
                        str(messages[-1].get("content", ""))
                        + "\n\n"
                        + _loop_break_nudge(loop_fail_tool, loop_fail_streak)
                    )
                    loop_fail_streak = 0  # 每轮新的连续失败只触发一次
                prev_had_tool_calls = True
            else:
                clean = self._strip_think(response.content)
                # 不将错误响应持久化到会话历史，因为它们可能污染上下文，导致永久 400 循环。
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    status = "error"
                    classification = response.error_classification
                    if classification is None:
                        classifier = getattr(self.provider, "classify_error", None)
                        classification = classifier(content=clean) if classifier is not None else None
                    error_category = classification.category if classification is not None else "unknown"
                    break

                # 空响应恢复：如果不处理，空的 Assistant Turn 会在此退出，暴露“无回复可给”的无效结果。
                # 放弃前先尝试恢复。合成脚手架用 ``_recovery_synthetic`` 标记，并在持久化和提取前移除，
                # 避免污染未来上下文。
                action = classify_empty_response(
                    response,
                    clean,
                    prev_had_tool_calls=prev_had_tool_calls,
                    nudges_done=post_tool_nudges,
                    prefill_retries=prefill_retries,
                    empty_retries=empty_retries,
                    limits=self._recovery_limits,
                )
                if action is RecoveryAction.PREFILL:
                    prefill_retries += 1
                    logger.warning(
                        "empty-recovery: thinking-only prefill {}/{}",
                        prefill_retries,
                        self._recovery_limits.thinking_prefill_max_retries,
                    )
                    # 将模型自身未删减的推理回填，使其继续生成正文。消息标记为合成，
                    # 在持久化和提取前丢弃；提供商的键白名单会从线上请求中移除推理字段。
                    messages = self.context.add_assistant_message(
                        messages,
                        response.content,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    messages[-1]["_recovery_synthetic"] = True
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.NUDGE:
                    post_tool_nudges += 1
                    logger.warning("empty-recovery: post-tool empty nudge")
                    # 空 Assistant 消息必须位于工具结果和提示之间，因为多数 API 会对裸的
                    # tool → user 序列返回 400。
                    messages = self.context.add_assistant_message(messages, "(empty)")
                    messages[-1]["_recovery_synthetic"] = True
                    messages.append({"role": "user", "content": POST_TOOL_NUDGE, "_recovery_synthetic": True})
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.RETRY:
                    empty_retries += 1
                    logger.warning(
                        "empty-recovery: plain empty retry {}/{}",
                        empty_retries,
                        self._recovery_limits.empty_content_max_retries,
                    )
                    prev_had_tool_calls = False
                    continue

                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached; synthesizing final answer", self.max_iterations)
            # 耗尽包含两个彼此独立的事实，并非二选一：
            #   1. 本轮尚未完成——将其标记为 ``interrupted``，让影子 Git 检查点提交带上
            #      对应标签，并让下一轮的恢复提示展示提交 SHA 和已编辑文件以便续作。
            #   2. 用户现在仍应得到有用的回复——因此无论是否存在检查点，都尽力生成收尾
            #      （调用一次禁用工具的模型，总结已完成和待完成事项），而不是返回固定道歉。
            #      如果生成失败，辅助方法内部会回退到静态消息，确保本轮不会静默结束。
            status = "interrupted"
            final_content = await self._synthesize_final_on_exhaustion(
                messages,
                effective_model,
                fallback_models,
                session_key,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
            # 像普通最终回复一样，把收尾内容持久化到历史中。下游持久化只读取返回的
            # ``messages`` 列表；若不这样做，生成的答案虽会经由流到达用户，却不会进入
            # 对话，下一轮（尤其是中断恢复轮）便看不到本次总结。生成提示本身只保留在
            # 辅助方法内部，因此这里只落下回复。
            if final_content:
                messages = self.context.add_assistant_message(messages, final_content)

        # 持久化和返回前移除临时的空响应恢复脚手架。Pico 会用
        # ``_recovery_synthetic`` 标记合成的推动/预填消息，必须移除以免持久化。
        if any(m.get("_recovery_synthetic") for m in messages):
            messages = [m for m in messages if not m.get("_recovery_synthetic")]

        outcome = TurnOutcome(
            status=status,
            error_category=error_category,
        )
        if self._checkpoint is not None:
            # 每轮快照：一次提交覆盖本轮全部编辑，正常退出和中断退出均如此
            # （与 Claude Code/Cursor 的粒度一致）。这里只尽力而为，commit_turn 从不抛错。
            label = f"turn {session_key or 'anon'} [{status}]"
            cid, changed = await self._checkpoint.commit_turn(label)
            outcome.checkpoint_id = cid
            if status == "interrupted":
                outcome.edited_files = changed

        return final_content, tools_used, messages, outcome

    async def run(self) -> None:
        """Bring the agent runtime up and stay alive.

        Turns arrive through the spine (``run_turn``); this coroutine no longer
        drains an inbound bus. It starts the executor / debug server / MCP, then
        idles on ``self._running`` so the gateway can gather it as a long-lived
        task and tear it down via ``stop()`` on shutdown.
        """
        self._running = True
        try:
            await self._start_executor()
            await self._start_debug_server()
            await self._connect_mcp()
        except SandboxInitError as exc:
            logger.error("Sandbox failed to start: {}", exc)
            await self.close_executor()
            self._running = False
            return
        except Exception:
            await self.close_executor()
            raise
        logger.info("Agent loop started")

        while self._running:
            await asyncio.sleep(1.0)

    @property
    def is_processing(self) -> bool:
        """True while a turn is being dispatched under the global lock."""
        return self._processing_lock.locked()

    def _notify_turn_complete(self) -> None:
        for callback in self.on_turn_complete:
            try:
                callback()
            except Exception:
                logger.exception("on_turn_complete callback failed")

    async def close_mcp(self) -> None:
        """Close MCP connections and the sandbox executor."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK 的取消作用域清理会产生噪声，但无害
            self._mcp_stack = None
        self._mcp_connected = False  # 重置后 _connect_mcp() 才能在关闭后重连
        self._mcp_connecting = False  # 重置以免并发调用方永久阻塞
        await self.close_executor()  # 即使未配置 MCP 服务器也始终执行

    async def close(self) -> None:
        """Stop background personalization before closing runtime resources."""
        async with self._close_lock:
            if self._closed:
                return
            self.begin_close()
            tasks = tuple(self._personalization_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.close_mcp()
            self._closed = True

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def replace_provider(self, provider: LLMProvider, *, model: str | None = None) -> None:
        from pico.call_efficiency.provider import CallEfficiencyProvider

        if isinstance(self.provider, CallEfficiencyProvider):
            self.provider.replace(provider)
        else:
            self.provider = provider
        if model is None:
            return
        self.model = model
        self.subagents.model = model
        self.memory_consolidator.model = model
        replace_model = getattr(self.context_engine, "replace_model", None)
        if callable(replace_model):
            replace_model(model)

    @trace.instrument("session.turn", seed=semconv.turn_seed, on_open=semconv.turn_open, extract=semconv.turn)
    async def _process_message(
        self,
        req: TurnRequest,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        usage_sink: dict[str, Any] | None = None,
        origin: Origin | None = None,
        drain: Drain | None = None,
        context_metadata_sink: dict[str, Any] | None = None,
    ) -> tuple[str | None, list[str]] | None:
        """Process a single turn request and return its reply.

        Returns ``(reply_content, media_paths)`` for a turn that produced an
        outbound reply, or ``None`` for a silent turn (the message tool already
        sent, or a hook short-circuit chose to return None). ``origin`` is the
        spine TurnRequest's origin.
        """
        from pico.agent.hook import AgentHookContext

        channel = req.source.channel
        sender_id = req.source.sender_id
        chat_id = req.source.chat_id
        content = req.text
        metadata = dict(req.source.extras)
        turn_media = list(req.media)
        msg_session_key = req.conversation or f"{channel}:{chat_id}"

        # AgentHook 的 ``before_user_inbound`` 链。
        #
        # 观察型钩子和短路型钩子共用同一条有序链。来源不是用户输入时跳过本阶段。
        skip_user_inbound = origin in _SKIP_USER_INBOUND_ORIGINS
        if len(self.hooks) > 0 and not skip_user_inbound:
            _hook_ctx = AgentHookContext(
                session_key=msg_session_key,
                turn_request=req,
            )
            _decision = await self.hooks.before_user_inbound(_hook_ctx)
            if _decision.short_circuit_result is not None:
                return _decision.short_circuit_result

        preview = content[:80] + "..." if len(content) > 80 else content
        logger.info("Processing message from {}:{}: {}", channel, sender_id, preview)

        key = session_key or msg_session_key
        session = self.sessions.get_or_create(key)

        # 斜杠命令
        cmd = content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return (
                        "Memory archival failed, session not cleared. Please try again.",
                        [],
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return (
                    "Memory archival failed, session not cleared. Please try again.",
                    [],
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return ("New session started.", [])
        if cmd == "/help":
            lines = [
                "✦ Pico commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return ("\n".join(lines), [])
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        # ── 个性化流程（全局开关：self.enable_personalization）────────────────
        # 子智能体结果回注时跳过：其内容是系统生成的通知而非用户输入；个性化处理会污染
        # 用户画像，或针对通知触发澄清。此处仅 SUBAGENT 跳过（不是更宽泛的用户输入集合）：
        # Cron 轮次目前仍会进入并保留该流程。
        if self.enable_personalization and origin is not Origin.SUBAGENT:
            from datetime import datetime as _dt

            from pico.agent.personalizer import Personalizer

            _personalizer = Personalizer(MemoryStore(self.state), self.provider, self.model)

            # ── 步骤 2 完成阶段：用户正在回答待处理的澄清问题 ──
            if session.pending_clarification:
                _pending = session.pending_clarification

                # 判断用户是在回答上一个问题，还是发起新请求。新请求通常包含动作动词，
                # 且与原请求无关；重新分类以作判断：若仍需澄清，就按新请求处理。
                _recent = session.get_history(max_messages=4)
                _recheck = await _personalizer.classify(content, history=_recent)
                _is_new_request = _recheck.get("needs_clarification", False)

                if _is_new_request:
                    # 用户发起了新请求；丢弃旧的待处理状态并重新分类。
                    session.pending_clarification = None
                    self.sessions.save(session)
                    logger.info("Personalization: new request detected, discarding old pending_clarification")

                    _question = await _personalizer.generate_question(
                        content,
                        _recheck.get("domain", ""),
                    )
                    if _question:
                        _ts = _dt.now().isoformat()
                        session.record({"role": "user", "content": content, "timestamp": _ts})
                        session.record({"role": "assistant", "content": _question, "timestamp": _ts})
                        session.pending_clarification = {
                            "original_message": content,
                            "question": _question,
                            "domain": _recheck.get("domain", ""),
                        }
                        self.sessions.save(session)
                        logger.info(
                            "Personalization: asked clarification for new request, session {}",
                            session.key,
                        )
                        return (_question, [])
                        # 问题生成失败时清除待处理状态，并按正常流程继续。
                    session.pending_clarification = None

                else:
                    # 用户正在回答上一个问题；提取偏好并恢复原任务。
                    session.pending_clarification = None

                    # 后台提取偏好并写入 MEMORY.md，不阻塞响应。
                    async def _extract():
                        await _personalizer.extract_and_store_preference(
                            original_message=_pending["original_message"],
                            question=_pending["question"],
                            answer=content,
                        )

                    self._start_personalization_task(_extract)
                    # 正常继续：LLM 通过对话历史理解任务。

            else:
                # ── 步骤 1：分类请求——判断是否需要澄清 ──
                _recent = session.get_history(max_messages=4)
                _classification = await _personalizer.classify(content, history=_recent)

                if _classification.get("needs_clarification"):
                    # ── 步骤 2：行动前交互——生成并返回澄清问题 ──
                    _question = await _personalizer.generate_question(
                        content,
                        _classification.get("domain", ""),
                    )

                    if _question:
                        # 将原请求和澄清问题写入历史，保持对话连贯。
                        _ts = _dt.now().isoformat()
                        session.record({"role": "user", "content": content, "timestamp": _ts})
                        session.record({"role": "assistant", "content": _question, "timestamp": _ts})

                        # 保存待处理状态，让下一条消息可以恢复流程。
                        session.pending_clarification = {
                            "original_message": content,
                            "question": _question,
                            "domain": _classification.get("domain", ""),
                        }
                        self.sessions.save(session)

                        logger.info("Personalization: asked clarification for session {}", session.key)
                        return (_question, [])
                        # generate_question 失败：静默跳过并继续
            # ── 个性化流程结束 ────────────────────────────────────────────────

        self._set_tool_context(channel, chat_id, metadata.get("message_id"), session_key=key)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
            # ask_user 使用真实 conversation_id（即通道/门控键）作为键；该标识感知话题
            # （req.conversation），而非仅使用 channel:chat_id。
        if (ask_tool := self.tools.get("ask_user")) and isinstance(ask_tool, AskUserTool):
            ask_tool.set_context(key)

        context_messages = self._context_messages_for_session(session)
        # SkillForge：选择器选出前 K 个。参见上方系统消息分支中的说明——返回空列表时
        # 回退到完整目录。B-3 阶段统一经由 ``_select_skills_for_turn`` 路由，使新的
        # ``default`` 引擎可在此直接结束选择。
        selected_skills = await self._select_skills_for_turn(
            content,
            context_messages,
        )
        context_metadata = context_metadata_sink if context_metadata_sink is not None else {}
        initial_messages = await self._assemble_context_messages(
            session=session,
            session_key=key,
            current_message=content,
            media=turn_media if turn_media else None,
            channel=channel,
            chat_id=chat_id,
            selected_skills=selected_skills or None,
            metadata_sink=context_metadata,
        )
        injected_skill_ids = list(
            context_metadata.get("injected_skill_ids") or self._collect_injected_skill_ids(selected_skills)
        )

        # ── 模型路由（EcoClaw 风格）──────────────────────────────────────────
        routed_model: str | None = None
        fallback_models: list[str] = []
        if self.router is not None:
            routed_model, fallback_models = await self.router.select_model_chain(content)
            if routed_model and routed_model != self.model:
                logger.info("Router: {} → {}", self.model, routed_model)
            if fallback_models:
                logger.info("Router fallback chain: {}", fallback_models)

        turn_start_idx = len(initial_messages) - 1
        final_content, _, all_msgs, outcome = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress,
            session_key=key,
            model=routed_model,
            fallback_models=fallback_models,
            injected_skill_ids=injected_skill_ids,
            on_token_delta=on_token_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_event=on_tool_event,
            usage_sink=usage_sink,
            drain=drain,
            origin=origin,
        )
        self._stash_recovery(key, outcome)
        if outcome.status == "error":
            raise ProviderTurnError(outcome.error_category or "unknown")

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

            # AgentHook 的 ``after_send`` 是通用出站阶段，适用于所有来源。
        if len(self.hooks) > 0:
            from pico.agent.hook import AgentHookContext

            _send_ctx = AgentHookContext(
                session_key=key,
                outbound_content=final_content,
            )
            _send_decision = await self.hooks.after_send(_send_ctx)
            if _send_decision.modified_content is not None:
                final_content = _send_decision.modified_content
                for message in reversed(all_msgs[turn_start_idx:]):
                    if message.get("role") == "assistant" and not message.get("tool_calls"):
                        message["content"] = final_content
                        break

        turn_artifact_messages = self._save_turn(
            session,
            all_msgs,
            turn_start_idx,
            origin,
        )
        self.sessions.save(session)
        await self.context_engine.after_turn(
            key,
            {
                "final_content": final_content,
                "messages": turn_artifact_messages,
                "context": context_metadata,
            },
        )
        await self._dispatch_backend_store(
            key,
            turn_artifact_messages,
        )
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

            # ── 步骤 4：行动后学习（后台、非阻塞）──────────────────────────────
            # 子智能体结果回注时跳过（参见上方轮次前流程）：其内容是系统生成的通知，
            # 不是可供学习的用户输入。
        if self.enable_personalization and origin is not Origin.SUBAGENT:
            from pico.agent.personalizer import Personalizer

            _p4 = Personalizer(MemoryStore(self.state), self.provider, self.model)

            async def _post_learn():
                await _p4.post_learn(content, final_content)

            self._start_personalization_task(_post_learn)
            # ── 步骤 4 结束 ────────────────────────────────────────────────────

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt.sent_in_turn:
            # 防御性指纹。过去智能体通过 message 工具回复并静默返回 None 时不会留下
            # 痕迹，导致随机出现的无效轮次无法通过 grep 发现。记录原本要返回的响应，
            # 为后续调查留下与下方 "Response to ..." 并行的线索。
            if final_content:
                preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
                logger.info(
                    "MessageTool sent in turn for {}:{}: {}",
                    channel,
                    sender_id,
                    preview,
                )
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", channel, sender_id, preview)
        return (final_content, [])

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        origin: Origin | None = None,
    ) -> list[dict]:
        """Save new-turn messages into session, truncating large tool results."""
        persisted: list[dict] = []
        for index, m in enumerate(messages[skip:]):
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if origin is Origin.SUBAGENT and index == 0 and role == "user":
                continue
            if entry.get("_recovery_synthetic"):
                continue  # #1a 合成恢复推动消息——绝不持久化脚手架
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # 跳过空助手消息——它们会污染会话上下文
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # 去除运行时上下文前缀，只保留用户文本。
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if (
                            c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                        ):
                            continue  # 从多模态消息中去除运行时上下文
                        if c.get("type") == "image_url" and c.get("image_url", {}).get("url", "").startswith(
                            "data:image/"
                        ):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry = sanitize_persisted_payload(entry)
            entry.setdefault("timestamp", self._now_fn().isoformat())
            session.record(entry)
            persisted.append(entry)
        session.updated_at = self._now_fn()
        return persisted

    async def run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
        usage_sink: dict[str, Any] | None = None,
        text_sink: dict[str, Any] | None = None,
    ) -> TurnOutcome:
        """Spine-native turn entry: consume a TurnRequest, fan the agent's output
        onto the single ``emit``, return a TurnOutcome. Collapses the legacy
        output paths (a str return + the five callbacks) onto one boundary.

        Named ``run_turn`` rather than ``run``: ``run`` is the runtime keep-alive
        (executor / debug server / MCP up, then idle). A spine runner wraps this
        method to satisfy the TurnRunner protocol.

        ``stream`` is the canon Q2-D assembly switch: a streaming outlet (TUI)
        wires it True so the reply goes out as StreamDelta and dissolves (b2 — no
        trailing Text); a non-streaming outlet (REPL) wires it False so the reply
        is one Text. It gates both LLM callbacks (the loop streams when either is
        wired) and the message-tool routing, so the whole reply travels one way.

        Exceptions propagate so the lane turns them into TurnFailed — run_turn
        does not catch sandbox-init to return an error string (the legacy direct
        path did; the spine surfaces it as a TurnFailed event instead).

        ``usage_sink`` lets a caller observe the turn's full token accounting
        (cost / context, richer than the three-field TurnOutcome.usage): pass a
        dict and it is filled. The TUI passes one to attach the rich usage to
        message.complete; the REPL omits it and uses TurnOutcome.usage.

        ``text_sink`` is its sibling for the reply text: pass a dict and the
        final reply lands in text_sink["text"] (the reply still goes out via
        emit — this is an observation copy, not a second delivery). Cron hosts
        use it for broadcast or TUI delivery after the turn completes.

        ``drain`` pulls user messages injected mid-turn (BusyPolicy.INJECT); it
        is threaded into the agent loop and consumed at the top of each iteration.
        """
        from pico.proactive_engine.schedulers.cron.tool import CronTool
        from pico.spine.events import (
            MediaOut,
            Notice,
            NoticeKind,
            Reasoning,
            StreamDelta,
            Text,
            ToolEvent,
            ToolPhase,
            Usage,
        )
        from pico.spine.message import Media
        from pico.spine.runner import TurnOutcome

        cid = req.conversation or f"{req.source.channel}:{req.source.chat_id}"

        streamed = False
        tool_calls = 0
        tool_failures = 0
        context_metadata: dict[str, Any] = {}

        async def on_token(text: str) -> None:
            nonlocal streamed
            if not text:
                return
            streamed = True
            await emit(StreamDelta(delta=text))

        async def on_reasoning(text: str) -> None:
            if text:
                await emit(Reasoning(content=text))

        async def on_tool(phase: str, info: dict[str, Any]) -> None:
            nonlocal tool_calls, tool_failures
            if phase == "start":
                tool_calls += 1
                if info["name"] != "message":
                    await emit(
                        ToolEvent(
                            phase=ToolPhase.START,
                            tool_call_id=info["tool_call_id"],
                            name=info["name"],
                            arguments=info["arguments"],
                            target_call_id=info.get("target_call_id"),
                            target_name=info.get("target_name"),
                            target_arguments=info.get("target_arguments"),
                        )
                    )
            else:
                if info["failed"]:
                    tool_failures += 1
                if info["name"] != "message" or info["failed"]:
                    await emit(
                        ToolEvent(
                            phase=ToolPhase.COMPLETE,
                            tool_call_id=info["tool_call_id"],
                            name=info["name"],
                            result_preview=info["result_preview"],
                            truncated=info["truncated"],
                            failed=info["failed"],
                            target_call_id=info.get("target_call_id"),
                            target_name=info.get("target_name"),
                            target_arguments=info.get("target_arguments"),
                            duration_ms=info.get("duration_ms"),
                        )
                    )

        async def on_progress(text: str, tool_hint: bool = False) -> None:
            # 保留进度与工具提示的区别，让出口像总线路径一样，分别通过配置开关
            # （send_progress 与 send_tool_hints）控制：工具提示文本使用
            # NoticeKind.TOOL_HINT，进度使用 PROGRESS。不渲染两者的出口仍会吞掉这两类通知。
            if text:
                await emit(
                    Notice(
                        kind=NoticeKind.TOOL_HINT if tool_hint else NoticeKind.PROGRESS,
                        detail=text,
                    )
                )

        async def _emit_media(paths: list[str]) -> None:
            await emit(
                MediaOut(media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in paths))
            )

        # 将 message 工具的回复路由到令牌流，使工具驱动的回复像主响应一样流式输出；
        # 此时 _process_message 返回 None，因此下方边界不会再次发出内容。回调属于当前轮次
        # （MessageTool 中的 ContextVar），并发轮次无法覆盖本轮路由，无需保存和恢复。
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):

            async def _route_to_stream(content: str, media: list[str]) -> None:
                # message 工具的回复可以附带媒体；需独立发出以免丢失（工具回复会让
                # _process_message 返回 None，下方边界看不到它）。内容与主回复遵循同一流式开关：
                # 流式时使用 StreamDelta，否则使用单个 Text；不然非流式出口会吞掉增量。
                if media:
                    await _emit_media(media)
                if text_sink is not None and content:
                    text_sink["text"] = content
                if stream:
                    await on_token(content)
                elif content:
                    await emit(Text(content=content))

            message_tool.set_send_callback(_route_to_stream)

        # CRON 轮次不得让智能体在运行途中调度新的 cron 任务。CronTool 通过 ContextVar
        # 防护；需在此处、即实际运行本轮的通道任务中设置，才能传播到工具。cron 回调在
        # 另一个任务中设置该值，无法传到本任务。
        cron_tool = self.tools.get("cron")
        cron_token = None
        if req.origin is Origin.CRON and isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)

        if usage_sink is None:
            usage_sink = {}
        try:
            await self._start_executor()
            await self._connect_mcp()
            out = await self._process_message(
                req,
                session_key=cid,
                on_progress=on_progress,
                on_token_delta=on_token if stream else None,
                on_reasoning_delta=on_reasoning if stream else None,
                on_tool_event=on_tool,
                usage_sink=usage_sink,
                origin=req.origin,
                drain=drain,
                context_metadata_sink=context_metadata,
            )
        except Exception:
            await self.close_executor()
            raise
        finally:
            if cron_token is not None and isinstance(cron_tool, CronTool):
                cron_tool.reset_cron_context(cron_token)

        # 单一的返回到发出边界（N-UNIFORM）。MediaOut 独立于流，并先于 Text
        # （G-MEDIA-2(a)：当前顺序为媒体优先）。
        if out is not None:
            reply_content, reply_media = out
            if reply_media:
                await _emit_media(reply_media)
            if not streamed and reply_content:
                await emit(Text(content=reply_content))
            if text_sink is not None and reply_content:
                text_sink["text"] = reply_content

        usage = Usage(
            prompt_tokens=int(usage_sink.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_sink.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_sink.get("total_tokens", 0) or 0),
        )
        # message 工具回复虽会让 _process_message 返回 None，但确实已回复，
        # 因此也计作显式回复。
        replied_via_tool = isinstance(message_tool, MessageTool) and message_tool.sent_in_turn
        return TurnOutcome(
            usage=usage,
            explicit_reply=out is not None or replied_via_tool,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
            memory_hits=int(context_metadata.get("memory_hits", 0) or 0),
            injected_skill_ids=tuple(context_metadata.get("injected_skill_ids") or ()),
            context_path=context_metadata.get("path"),
            context_fallback_reason=context_metadata.get("fallback_reason"),
            skill_source_failures=tuple(context_metadata.get("skill_source_failures") or ()),
        )


def _merge_tool_call_fragments(
    slots: list[dict[str, Any]],
    delta: dict[str, Any],
) -> None:
    """Merge a single chat_stream tool_call_delta into accumulator slots.

    Each slot follows the shape ``{id, function: {name, arguments_buf: [str]}}``.
    Per provider chunk semantics (OpenAI/LiteLLM): each tool call fragment
    carries an ``index`` field; ``id`` / ``function.name`` typically appear in
    the first fragment for that index, ``function.arguments`` is a JSON string
    streamed in pieces.

    Respects the ``index`` field so parallel multi-tool streams do not
    collapse into ``slots[0]``. Fragments without an ``index`` default to 0
    (single-tool case, backward-compatible).
    """
    incoming = delta.get("tool_calls") or []
    if not incoming:
        return
    for tc in incoming:
        idx = int(tc.get("index", 0) or 0)
        while len(slots) <= idx:
            slots.append({"id": None, "function": {"name": None, "arguments_buf": []}})
        slot = slots[idx]
        if tc.get("id") and not slot["id"]:
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not slot["function"]["name"]:
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments_buf"].append(fn["arguments"])


def _finalize_tool_calls(slots: list[dict[str, Any]]) -> list[ToolCallRequest]:
    """Convert accumulator slots into final ToolCallRequest list."""
    result: list[ToolCallRequest] = []
    for slot in slots:
        name = slot["function"]["name"]
        if not name:
            continue
        args_text = "".join(slot["function"]["arguments_buf"])
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": args_text}
        result.append(
            ToolCallRequest(
                id=slot["id"] or "",
                name=name,
                arguments=args,
            )
        )
    return result
