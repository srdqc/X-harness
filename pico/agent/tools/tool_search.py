"""Progressive tool disclosure for large catalogs.

When built-ins + plugins + MCP push the tool count past a threshold, injecting
every schema into each request burns context that scales with tool count. This
module withholds most schemas and exposes two meta-tools instead:

  - ``tool_search`` — BM25 keyword search over the hidden catalog; each hit
                      carries name + description + parameter schema, enough to
                      call the tool without a second lookup.
  - ``tool_call``   — invoke a cataloged tool by name; forwards through the
                      registry, which validates arguments and returns a
                      correctable error when they don't fit the schema.

The tool list sent to the model never changes turn-to-turn (always the core
set + these two meta-tools), so the prompt cache stays stable across the
whole session — tools sit ahead of system+messages in the cached prefix, so a
changing tool list would invalidate everything after it. The cost is that
cataloged tools are invoked through ``tool_call`` rather than native
function-calling.

Two visibility tiers per turn (see :class:`ToolSearchStrategy`):
  - always-visible: a core set + the meta-tools (full schema every turn);
  - cataloged:      everything else — searchable, schema withheld until asked.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pico.agent.tools.base import Tool
from pico.agent.tools.execution import ToolExecutionContext, ToolInvocation
from pico.agent.tools.tool_index import ToolIndex
from pico.token_wise.base import TokenStrategy

if TYPE_CHECKING:
    from pico.agent.tools.registry import ToolRegistry

# 核心工具在每个 Turn 都保持可见，否则 Agent 还得搜索它们，基本能力会受限。除文件、
# 搜索和执行原语外，``message``、``ask_user`` 和 ``spawn`` 是任何 Turn 都可能需要的
# 交互和编排原语：回复、通过提问解除阻塞、委托子 Agent。隐藏后模型可能根本想不到搜索它们。
# ``tools.tool_search.always_visible`` 配置可扩展此集合。
DEFAULT_ALWAYS_VISIBLE: tuple[str, ...] = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "find",
    "exec",
    "message",
    "ask_user",
    "spawn",
)

TOOL_CALL_NAME: str = "tool_call"
# 元工具在功能开启时始终注册，但绝不进入目录。
META_TOOL_NAMES: frozenset[str] = frozenset({"tool_search", TOOL_CALL_NAME})


class ToolSearchController:
    """Shared state between the meta-tools and the strategy.

    Holds the live registry (the catalog source of truth) and the BM25 index.
    The visible set is constant (core + meta-tools), which keeps the per-turn
    tool list — and thus the prompt cache — stable.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        *,
        always_visible: set[str],
        search_result_limit: int = 10,
    ) -> None:
        self._registry = registry
        self.always_visible = set(always_visible) | META_TOOL_NAMES
        self.search_result_limit = search_result_limit
        self._index = ToolIndex()

    def _catalog_tools(self) -> list[Tool]:
        """All registered tools except the meta-tools (never self-searchable)."""
        out = []
        for name in self._registry.tool_names:
            if name in META_TOOL_NAMES:
                continue
            tool = self._registry.get(name)
            if tool is not None:
                out.append(tool)
        return out

    def refresh(self) -> None:
        """Sync the BM25 index with the current registry (no-op if unchanged)."""
        self._index.ensure(self._catalog_tools())

    def visible_names(self) -> set[str]:
        return self.always_visible

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Hits carry name + description + parameter schema, so the model can go
        straight to tool_call without a separate describe round-trip."""
        names = self._index.search(query, limit or self.search_result_limit)
        hits = []
        for name in names:
            tool = self._registry.get(name)
            if tool is None:
                continue
            hits.append(
                {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return hits

    def resolve_invocation(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        context: ToolExecutionContext,
    ) -> ToolInvocation | None:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, (dict, type(None))):
            return None
        if name in META_TOOL_NAMES or not self._registry.has(name):
            return None
        child_call_id = f"{context.call_id}:{name}" if context.call_id else None
        return ToolInvocation(
            name=name,
            arguments=arguments or {},
            context=context.child(child_call_id),
        )

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        context: ToolExecutionContext | None = None,
    ) -> str:
        """Invoke a cataloged tool: forward to the registry (validates args).

        Models sometimes emit the nested ``arguments`` as a JSON string rather
        than an object; parse that case so the call still goes through.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "Error: 'arguments' must be a JSON object."
        if name in META_TOOL_NAMES:
            return f"Error: '{name}' cannot be invoked via tool_call."
        if not self._registry.has(name):
            return f"Error: tool '{name}' not found. Use tool_search to find it."
        invocation = self.resolve_invocation(name, arguments, context or ToolExecutionContext())
        if invocation is None:
            return "Error: 'arguments' must be a JSON object."
        execution = await self._registry.execute_invocation(invocation)
        return execution.result


class ToolSearchTool(Tool):
    """Keyword search over tools whose schemas are not currently loaded."""

    def __init__(self, controller: ToolSearchController) -> None:
        self._ctrl = controller

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "Search the catalog of additional tools that are available but not "
            "currently loaded. Returns matching tools with their description and "
            "parameter schema, ready to invoke with tool_call. Query with task "
            "keywords, e.g. 'create github issue' or '生成图片'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Task keywords describing the capability you need.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int | None = None) -> str:
        hits = self._ctrl.search(query, limit)
        if not hits:
            return f"No tools matched '{query}'. Try broader or different keywords."
        return json.dumps(hits, ensure_ascii=False)


class ToolCallTool(Tool):
    """Invoke a cataloged tool by name. Arguments are validated by the registry."""

    def __init__(self, controller: ToolSearchController) -> None:
        self._ctrl = controller

    @property
    def name(self) -> str:
        return TOOL_CALL_NAME

    @property
    def description(self) -> str:
        return (
            "Invoke a tool found via tool_search by name, passing its arguments. "
            "If the arguments don't fit the tool's schema the registry returns a "
            "validation error describing the fix; adjust and call again."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact tool name from a tool_search result.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments object for the target tool.",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        return await self._ctrl.call(name, arguments)

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        return await self._ctrl.call(name, arguments, context)

    def resolve_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        name = invocation.arguments.get("name")
        if not isinstance(name, str):
            return invocation
        resolved = self._ctrl.resolve_invocation(
            name,
            invocation.arguments.get("arguments"),
            invocation.context,
        )
        return resolved or invocation


class ToolSearchStrategy(TokenStrategy):
    """``before_llm_call`` hook that compacts the tool list for large catalogs.

    At or below ``compaction_threshold`` tools it passes through unchanged (and
    drops the meta-tools, so small setups are byte-for-byte as before). Above
    it, only the always-visible core + meta-tools keep their schema in the
    request; the rest stay reachable via ``tool_search`` / ``tool_call``.
    """

    def __init__(self, controller: ToolSearchController, *, compaction_threshold: int = 50) -> None:
        self._ctrl = controller
        self._compaction_threshold = compaction_threshold

    @property
    def name(self) -> str:
        return "tool_search"

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        if not tools:
            return messages, tools, model
        self._ctrl.refresh()
        catalog_size = sum(1 for t in tools if t["function"]["name"] not in META_TOOL_NAMES)
        if catalog_size <= self._compaction_threshold:
            out = [t for t in tools if t["function"]["name"] not in META_TOOL_NAMES]
            return messages, out, model
        present = {t["function"]["name"] for t in tools}
        if not META_TOOL_NAMES.issubset(present):
            # 元工具不可用时，例如被 disabled_tools 移除，直接暴露全部工具；
            # 不要将已编目的工具困在模型无法调用的搜索之后。
            return messages, tools, model
        visible = self._ctrl.visible_names()
        out = [t for t in tools if t["function"]["name"] in visible]
        return messages, out, model
