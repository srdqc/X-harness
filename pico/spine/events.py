"""The single output vocabulary: everything a turn can emit."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pico.spine.message import Media, Source


@dataclass(frozen=True)
class Usage:
    """Token accounting for one turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NoticeKind(StrEnum):
    """Out-of-band signals a turn surfaces to the user."""

    PROGRESS = "progress"
    TOOL_HINT = "tool_hint"
    INJECTED = "injected"
    DELIVERY_FAILED = "delivery_failed"


class ToolPhase(StrEnum):
    """When a tool event fires; outlets render the two phases differently."""

    START = "start"
    COMPLETE = "complete"


# lifecycle event 由 worker 发出，runner 不得发出。


@dataclass(frozen=True)
class TurnStarted:
    """Marker that a turn began."""

    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnFailed:
    error: str
    cancelled: bool
    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnEnded:
    usage: Usage
    latency_ms: float
    explicit_reply: bool
    conversation_id: str | None = None
    tool_calls: int = 0
    tool_failures: int = 0


# 可投递 event 由 runner 发出，并路由到 Outlet。


@dataclass(frozen=True)
class ToolEvent:
    phase: ToolPhase
    tool_call_id: str
    name: str = ""
    arguments: dict[str, Any] | None = None
    result_preview: str = ""
    truncated: bool = False
    failed: bool = False
    target_call_id: str | None = None
    target_name: str | None = None
    target_arguments: dict[str, Any] | None = None
    duration_ms: float | None = None
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Text:
    content: str
    source: Source | None = None
    reply_to: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class MediaOut:
    media: tuple[Media, ...]
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class StreamDelta:
    delta: str
    stream_id: str | None = None
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Reasoning:
    content: str
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    source: Source | None = None
    detail: str | None = None
    conversation_id: str | None = None


RunnerEvent = ToolEvent | Text | MediaOut | StreamDelta | Reasoning | Notice
# 同一个 union，按投递职责命名：hub 路由、Outlet 渲染的内容。
Deliverable = RunnerEvent
TurnEvent = TurnStarted | TurnFailed | TurnEnded | RunnerEvent
