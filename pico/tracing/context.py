"""Per-turn trace context, propagated via contextvars.

contextvars survive ``await`` and are snapshotted when ``asyncio.create_task``
forks a child task — so a subagent spawned mid-turn (P1) inherits the turn's
span as parent automatically, and nested LLM/tool calls hang off the right node.
"""

from __future__ import annotations

import contextlib
import contextvars
import secrets
import time
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True)
class TraceCtx:
    trace_id: str
    session_key: str | None = None
    channel: str | None = None
    chat_id: str | None = None
    parent_span_id: str | None = None
    turn_span_id: str | None = None
    # 最近一层非模型 span 的名称，表示模型调用服务的目的
    # （turn / memory.extract / skill.gate / ...）。模型类 span 继承该来源，
    # 而不是把自身作为来源，因此嵌套调用
    # ``llm.call`` 无需遍历树即可自行标记；通用跨度没有采用方。
    # 的名称无需硬编码在此处。
    source: str | None = None


_CTX: contextvars.ContextVar[TraceCtx | None] = contextvars.ContextVar("pico_tracing_ctx", default=None)


def current() -> TraceCtx | None:
    return _CTX.get()


def new_trace_id() -> str:
    return f"trace-{int(time.time() * 1000):x}-{secrets.token_hex(4)}"


def new_span_id() -> str:
    return f"span-{int(time.time() * 1000):x}-{secrets.token_hex(3)}"


@contextlib.contextmanager
def turn_scope(
    *,
    session_key: str | None,
    channel: str | None,
    chat_id: str | None,
    root_span_id: str,
) -> Iterator[TraceCtx]:
    """Open a fresh trace for one turn; child spans parent onto ``root_span_id``."""
    ctx = TraceCtx(
        trace_id=new_trace_id(),
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        parent_span_id=root_span_id,
    )
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def push(
    *,
    trace_id: str,
    span_id: str,
    name: str | None = None,
    kind: str | None = None,
    session_key: str | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
):
    """Set the active ctx so descendants parent onto ``span_id``; returns a reset token.

    Used by the ``trace.span`` facade for manual instrumentation — it controls
    enter/exit explicitly rather than via a ``with`` block. Pair with :func:`reset`.

    ``name``/``kind`` propagate the enclosing ``source`` (see :class:`TraceCtx`):
    a non-model span becomes the source for its descendants; a model span inherits
    its parent's source (so it is never its own invocation source).
    """
    cur = _CTX.get()
    parent_source = cur.source if cur else None
    source = parent_source if (kind == "model" or not name) else name
    return _CTX.set(
        TraceCtx(
            trace_id=trace_id,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            parent_span_id=span_id,
            turn_span_id=span_id if name == "session.turn" else (cur.turn_span_id if cur else None),
            source=source,
        )
    )


def reset(token) -> None:
    _CTX.reset(token)


@contextlib.contextmanager
def child_scope(span_id: str) -> Iterator[TraceCtx]:
    """Re-parent descendants onto ``span_id`` (used by the subagent probe, P1)."""
    cur = _CTX.get() or TraceCtx(trace_id=new_trace_id())
    token = _CTX.set(replace(cur, parent_span_id=span_id))
    try:
        yield _CTX.get()  # type: ignore[misc]
    finally:
        _CTX.reset(token)
