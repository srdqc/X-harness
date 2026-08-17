"""Subagent manager for background task execution."""

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

from pico.agent.tools.base import ToolResult
from pico.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from pico.agent.tools.registry import ToolRegistry
from pico.agent.tools.shell import ExecTool
from pico.agent.tools.web import WebFetchTool, WebSearchTool
from pico.config.schema import ExecToolConfig
from pico.providers.base import LLMProvider
from pico.sandbox import SandboxConfig, build_executor
from pico.security.trust import wrap_untrusted
from pico.tracing import semconv, trace
from pico.utils.helpers import build_assistant_message

# 使用一小时窗口：失控的重新注入循环会快速触发上限，而合法启动分散在时间上，
# 会在受限前过期淘汰。
_SPAWN_WINDOW_SECONDS = 3600


class SubagentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    EXHAUSTED = "exhausted"

    @property
    def failed(self) -> bool:
        return self is not SubagentStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class SubagentOutcome:
    status: SubagentStatus
    result: str

    @property
    def failed(self) -> bool:
        return self.status.failed


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        sandbox_config: "SandboxConfig | None" = None,
        owned_ids: set[str] | None = None,
        jina_api_key: str | None = None,
        max_concurrent: int = 4,
        max_spawns_per_hour: int = 30,
        state: Path | None = None,
    ):
        from pico.config.schema import ExecToolConfig

        self.provider = provider
        self.workspace = workspace
        self.state = state or workspace
        # Spine submit 延迟绑定。调度器在构建时绑定所属事件循环，并于各入口的运行循环内构建；
        # 而本管理器在 AgentLoop.__init__ 的同步前导阶段构建。任何通知前通过 set_submit 连接，
        # 结果重新注入时提交一个来源为 SUBAGENT 的 Turn。
        self._submit = None
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.jina_api_key = jina_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._sandbox_config = sandbox_config
        self._owned_ids = owned_ids
        self._running_tasks: dict[str, asyncio.Task[SubagentOutcome]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # 会话键 -> {任务 ID, ...}
        self._gate = asyncio.Semaphore(max_concurrent)
        self._max_spawns_per_hour = max_spawns_per_hour
        # 按会话保存单调的启动时间戳，而非按进程，避免一个繁忙会话限流其他会话。
        # 每次访问都将双端队列剪裁到滚动窗口内，因此容量自动有界。
        self._session_spawn_times: dict[str, deque[float]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        quota_key = session_key or "default"
        now = time.monotonic()
        window = self._session_spawn_times.setdefault(quota_key, deque())
        cutoff = now - _SPAWN_WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._max_spawns_per_hour:
            logger.warning(
                "Spawn refused: session {!r} hit spawn rate limit ({}/hour)",
                quota_key,
                self._max_spawns_per_hour,
            )
            return ToolResult(
                f"Spawn refused: this session hit its subagent spawn rate limit "
                f"({self._max_spawns_per_hour} per hour). It recovers automatically "
                f"as earlier spawns age out — if this is unexpected, the task may "
                f"be looping; reconsider the approach instead of spawning again.",
                failed=True,
            )
        window.append(now)
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id, "session_key": quota_key}

        bg_task = asyncio.create_task(self._run_subagent(task_id, task, display_label, origin))
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task[SubagentOutcome]) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    @trace.instrument("subagent.run", extract=semconv.subagent)
    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> SubagentOutcome:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            # 每个子 Agent 运行独立沙箱虚拟机，需限制数量，避免大规模扇出耗尽宿主机资源。
            async with self._gate:
                executor = build_executor(self._sandbox_config, self.workspace, self._owned_ids)
                async with executor:
                    outcome = await self._run_subagent_inner(task_id, task, label, origin, executor)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Subagent [{}] failed: {}", task_id, e)
            outcome = SubagentOutcome(SubagentStatus.FAILED, f"Error: {str(e)}")

        await self._announce_result(task_id, label, task, outcome.result, origin, outcome.status)
        return outcome

    async def _run_subagent_inner(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        executor: Any,
    ) -> SubagentOutcome:
        try:
            # 构建子 Agent 工具，不包含 message 和 spawn
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                    executor=executor,
                )
            )
            tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
            tools.register(WebFetchTool(api_key=self.jina_api_key, proxy=self.web_proxy))

            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # 在有限迭代次数内运行 Agent Loop
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                    messages.append(
                        build_assistant_message(
                            response.content or "",
                            tool_calls=tool_call_dicts,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        )
                    )

                    # 执行工具
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug(
                            "Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str
                        )
                        result = await tools.execute(tool_call.name, tool_call.arguments, tool_call.id)
                        # 子 Agent 循环同样是不可信数据路径，需像主循环的 add_tool_result 一样
                        # 对工具输出设置边界。
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": wrap_untrusted(result, source=tool_call.name),
                            }
                        )
                else:
                    final_result = response.content
                    break
            else:
                final_result = f"Iteration budget exhausted after {max_iterations} iterations without a final response."
                logger.warning("Subagent [{}] exhausted its iteration budget", task_id)
                return SubagentOutcome(SubagentStatus.EXHAUSTED, final_result)

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info("Subagent [{}] completed successfully", task_id)
            return SubagentOutcome(SubagentStatus.COMPLETED, final_result)

        except Exception as e:
            logger.error("Subagent [{}] failed: {}", task_id, e)
            return SubagentOutcome(SubagentStatus.FAILED, f"Error: {str(e)}")

    def set_submit(self, submit) -> None:
        self._submit = submit

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: SubagentStatus,
    ) -> None:
        """Announce the subagent result to the main agent via the spine.

        This inbound system message triggers a main-agent turn directly.
        """
        status_text = {
            SubagentStatus.COMPLETED: "completed successfully",
            SubagentStatus.FAILED: "failed",
            SubagentStatus.EXHAUSTED: "failed: iteration budget exhausted",
        }[status]

        # 子 Agent 结果可受攻击者影响，因为它可能抓取了网页或读取了文件。
        # 在重新进入主 Agent 上下文前，需将其作为不可信数据隔离。
        fenced_result = wrap_untrusted(result, source="subagent")
        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{fenced_result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # 重新注入以在原始会话中触发主 Agent Turn。Spine 路径以原始会话为 conversation
        # 路由，并设 origin=SUBAGENT；每个宿主运行器通过其保留的输出界面交付最终回复。
        # 通知内容已固定，因此可即发即忘。
        assert self._submit is not None
        from pico.spine import ChatType, Origin, Source, TurnRequest

        self._submit(
            TurnRequest(
                origin=Origin.SUBAGENT,
                source=Source(
                    channel=origin["channel"],
                    chat_id=origin["chat_id"],
                    sender_id="subagent",
                    chat_type=ChatType.DM,
                ),
                text=announce_content,
                conversation=origin["session_key"],
            )
        )
        logger.debug("Subagent [{}] announced result to {}", task_id, origin["session_key"])

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from pico.agent.context import ContextBuilder
        from pico.memory_engine.skill_forge import LocalSkillCatalog

        # 使用临时 ContextBuilder 访问运行时上下文构建器，因为 SubagentManager 没有自己的实例。
        time_ctx = ContextBuilder(
            self.workspace,
            state=self.state,
            start_watcher=False,
        )._build_runtime_context(None, None)
        parts = [
            f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""
        ]

        skills_summary = LocalSkillCatalog(
            self.state,
            start_watcher=False,
        ).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        return "\n\n".join(parts)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # 销毁时删除当前会话的限流项。剪裁只会清空双端队列，不会删除键；
        # 如果不在此处清理，字典会在整个进程生命周期内为每个会话保留一项。
        self._session_spawn_times.pop(session_key, None)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
