"""Tool registry for dynamic tool management."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from typing import Any

from pico.agent.tools.base import Tool, ToolResult
from pico.agent.tools.execution import (
    ToolEffect,
    ToolExecution,
    ToolExecutionContext,
    ToolInvocation,
)
from pico.tracing import semconv, trace

ToolStartCallback = Callable[[ToolInvocation], Awaitable[None]]
ToolCompleteCallback = Callable[[ToolExecution], Awaitable[None]]


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    # 为未设置自身 ``timeout_seconds`` 的工具提供兜底上限。刻意设得宽裕：它用于打断
    # 没有内部超时且永不返回的无限挂起，而不是强制严格的单工具 SLA。
    DEFAULT_TOOL_TIMEOUT_S = 300.0
    DEFAULT_MAX_PARALLEL = 4

    def __init__(self, *, max_parallel: int = DEFAULT_MAX_PARALLEL):
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        self._tools: dict[str, Tool] = {}
        self._max_parallel = max_parallel

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a tool."""
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    @trace.instrument("tool.call", extract=semconv.tool_call)
    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        call_id: str | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Execute a tool by name with given parameters.

        ``call_id`` is the model's own tool-call id when the caller has one; it
        is not used for dispatch, only recorded as ``tool.call_id`` so the span
        joins the ToolEvent the same call emits.
        """
        _hint = "\n\n[Analyze the error above and try a different approach.]"

        invocation = ToolInvocation(
            name=name,
            arguments=params,
            context=context or ToolExecutionContext(call_id=call_id),
        )
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}",
                failed=True,
            )

        try:
            # 尝试转换参数以匹配模式类型
            params = tool.cast_params(params)

            # 校验参数
            errors = tool.validate_params(params)
            if errors:
                return ToolResult(
                    f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _hint,
                    failed=True,
                )

            ceiling = tool.timeout_seconds or self.DEFAULT_TOOL_TIMEOUT_S
            if tool.blocking_interaction:
                # 该工具有意等待人类，不得被超时计时器终止。
                result = await tool.execute_with_context(invocation.context, **params)
            else:
                result = await asyncio.wait_for(
                    tool.execute_with_context(invocation.context, **params),
                    timeout=ceiling,
                )

            if isinstance(result, ToolResult):
                return result
            if isinstance(result, str) and result.startswith("Error"):
                return ToolResult(result + _hint, failed=True)
            return ToolResult(result)
        except asyncio.TimeoutError:
            return ToolResult(
                f"Error: Tool '{name}' timed out after {ceiling:.0f}s." + _hint,
                failed=True,
            )
        except Exception as e:
            return ToolResult(f"Error executing {name}: {str(e)}" + _hint, failed=True)

    async def execute_invocation(self, invocation: ToolInvocation) -> ToolExecution:
        started = time.perf_counter_ns()
        result = await self.execute(
            invocation.name,
            invocation.arguments,
            invocation.context.call_id,
            invocation.context,
        )
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        return ToolExecution(invocation=invocation, result=result, duration_ms=duration_ms)

    async def execute_many(
        self,
        invocations: Iterable[ToolInvocation],
        *,
        parallel_safe: bool = True,
        on_start: ToolStartCallback | None = None,
        on_complete: ToolCompleteCallback | None = None,
    ) -> list[ToolExecution]:
        pending = list(invocations)
        executions: list[ToolExecution] = []
        index = 0
        while index < len(pending):
            if parallel_safe and self._is_concurrency_safe(pending[index]):
                end = index + 1
                while end < len(pending) and self._is_concurrency_safe(pending[end]):
                    end += 1
                for start in range(index, end, self._max_parallel):
                    executions.extend(
                        await self._execute_parallel(
                            pending[start : min(start + self._max_parallel, end)],
                            on_start,
                            on_complete,
                        )
                    )
                index = end
                continue
            executions.append(await self._execute_observed(pending[index], on_start, on_complete))
            index += 1
        return executions

    def _is_concurrency_safe(self, invocation: ToolInvocation) -> bool:
        resolved = self._resolve_invocation(invocation)
        tool = self._tools.get(resolved.name)
        return bool(tool is not None and tool.capability.effect is ToolEffect.READ and tool.capability.concurrency_safe)

    def _resolve_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        tool = self._tools.get(invocation.name)
        if tool is None:
            return invocation
        return tool.resolve_invocation(invocation)

    async def _execute_parallel(
        self,
        invocations: list[ToolInvocation],
        on_start: ToolStartCallback | None,
        on_complete: ToolCompleteCallback | None,
    ) -> list[ToolExecution]:
        tasks = [asyncio.create_task(self._execute_observed(item, on_start, on_complete)) for item in invocations]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute_observed(
        self,
        invocation: ToolInvocation,
        on_start: ToolStartCallback | None,
        on_complete: ToolCompleteCallback | None,
    ) -> ToolExecution:
        observed = self._resolve_invocation(invocation)
        if on_start is not None:
            await on_start(observed)
        execution = await self.execute_invocation(invocation)
        if execution.invocation is not observed:
            execution = replace(execution, invocation=observed)
        if on_complete is not None:
            await on_complete(execution)
        return execution

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
