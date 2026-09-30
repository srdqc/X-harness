"""定义 Spine 与 Agent 之间“一个 Turn 如何运行”的行为接缝。

Spine 只声明 ``TurnRunner`` protocol，Agent Loop 提供实现；Spine 从不反向 import Agent，
因此调度、取消和终态所有权不依赖具体模型引擎。Runner 接收 `TurnRequest`、一个事件
``emit`` 和一个中途消息 ``drain``，完成后只返回结构化 `TurnOutcome`。

``emit`` 的类型被收窄为 ``RunnerEvent``，表示 Runner 只能发文本、媒体、Tool、Reasoning
和 Notice，不能自行发 TurnStarted/TurnEnded/TurnFailed lifecycle event；这些终态由 Worker
唯一拥有。仓库没有静态检查器能完全执行该限制，所以 Scheduler 的 emit boundary 还有
运行时类型守卫。``drain`` 可以被最小实现忽略，此时所有 INJECT 最终回退为 APPEND Turn。
"""

import contextlib
import contextvars
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pico.spine.events import RunnerEvent, Usage
from pico.spine.turn import TurnRequest

Emit = Callable[[RunnerEvent], Awaitable[None]]
# 读取并移除当前 lane 待处理的 inject，以便在工具循环间隙合并。runner 可以
# 忽略它；最小合法实现从不 drain，因此每个 inject 都回退为 APPEND Turn。
# 该操作同步执行，因为 drain 只是读取 deque。
Drain = Callable[[], list[TurnRequest]]

_CURRENT_TURN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pico_runtime_turn_id",
    default=None,
)


def current_turn_id() -> str | None:
    """Return the Runtime identity of the independent Turn currently executing."""

    return _CURRENT_TURN_ID.get()


@contextlib.contextmanager
def turn_identity_scope(turn_id: str) -> Iterator[None]:
    """Propagate Runtime Turn identity without attaching it to TurnRequest."""

    token = _CURRENT_TURN_ID.set(turn_id)
    try:
        yield
    finally:
        _CURRENT_TURN_ID.reset(token)


@dataclass(frozen=True)
class TurnOutcome:
    """记录 Runner 完成一轮后交回 Worker 的可核算结果。

    `usage` 保存 Token 账目，``explicit_reply`` 表示本轮是否明确产生回复；Tool 调用与失败、
    Memory 命中、注入 Skill、Context 路径和回退原因则提供运行证据。Worker 使用这些字段
    填充 TurnEnded 和 tracing，而不是从已发送事件反推执行情况。

    该对象不携带回复正文：正文已经沿 ``emit`` 单一路径交付。`skill_source_failures` 等 tuple
    使用不可变形状，整个数据类也冻结，确保 Worker 观察到的是 Runner 返回时的终态快照。
    """

    usage: Usage
    explicit_reply: bool
    tool_calls: int = 0
    tool_failures: int = 0
    memory_hits: int = 0
    injected_skill_ids: tuple[str, ...] = ()
    context_path: str | None = None
    context_fallback_reason: str | None = None
    skill_source_failures: tuple[str, ...] = ()
    tool_disclosure_fallback_used: bool = False
    tool_disclosure_fallback_reason: str | None = None
    tool_disclosure_fallback_iteration: int | None = None
    tool_disclosure_zero_hits_before_fallback: int = 0
    tool_disclosure_provider_calls_before_fallback: int = 0
    tool_disclosure_provider_calls_after_fallback: int = 0
    tool_disclosure_recovery_succeeded: bool | None = None


@runtime_checkable
class TurnRunner(Protocol):
    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome: ...
