"""The behavioural seam between spine and the agent: how one turn runs.

spine defines the ``TurnRunner`` protocol and the agent loop implements it;
spine never imports the agent (dependency inversion). ``emit`` is narrowed to
``RunnerEvent`` so a runner cannot emit lifecycle events — the worker owns
those. With no static checker in this repo that narrowing is intent only; the
enforcing guard lives at the scheduler's emit boundary.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pico.spine.events import RunnerEvent, Usage
from pico.spine.turn import TurnRequest

Emit = Callable[[RunnerEvent], Awaitable[None]]
# 读取并移除当前 lane 待处理的 inject，以便在工具循环间隙合并。runner 可以
# 忽略它；最小合法实现从不 drain，因此每个 inject 都回退为 APPEND Turn。
# 该操作同步执行，因为 drain 只是读取 deque。
Drain = Callable[[], list[TurnRequest]]


@dataclass(frozen=True)
class TurnOutcome:
    """What a finished run hands back; the worker fills TurnEnded from it."""

    usage: Usage
    explicit_reply: bool
    tool_calls: int = 0
    tool_failures: int = 0
    memory_hits: int = 0
    injected_skill_ids: tuple[str, ...] = ()
    context_path: str | None = None
    context_fallback_reason: str | None = None
    skill_source_failures: tuple[str, ...] = ()


@runtime_checkable
class TurnRunner(Protocol):
    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome: ...
