"""``turn.*`` real handlers.

* ``turn.send`` submits the turn onto the spine (one per ``session_key``) and
  returns ``{turn_id, accepted: True}`` synchronously; streaming output flows
  out via the build_tui runner/hub/sink as ``SubscriptionEmitter``
  notifications.
* ``turn.subscribe`` wraps ``SubscriptionEmitter.register``.
* ``turn.unsubscribe`` wraps ``SubscriptionEmitter.unregister`` (idempotent).
* ``turn.cancel`` cancels the in-flight turn handle and emits the one
  ``error`` event with ``reason="cancelled_by_client"`` (the sink stays silent
  on a cancelled TurnFailed to avoid a double error).

The handlers are exposed at the module level so tests can patch the
``_resolve_model`` seam. ``register_turn_methods`` closes the ``emitter`` and
the build_tui bundle (``scheduler`` / ``turn_ids`` / ``submission_ids`` /
``build_error``) into single-argument dispatcher handlers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from pico.spine import ChatType, Origin, Source, TurnHandle, TurnRequest
from pico.spine.scheduler import Scheduler, SchedulerDrainingError
from pico.tui_rpc.errors import RpcError, TurnInProgressError
from pico.tui_rpc.methods.image import consume_pending_images, pending_images
from pico.tui_rpc.models import (
    TurnCancelParams,
    TurnSendParams,
    TurnSubscribeParams,
    TurnUnsubscribeParams,
)
from pico.tui_rpc.subscriptions import SubscriptionEmitter

if TYPE_CHECKING:
    from pico.tui_rpc.dispatcher import Dispatcher

_TURN_FAILED_CODE = -32099
SchedulerFactory = Callable[[], Awaitable[Scheduler]]

# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------

# 以 session_key 为键保存正在执行的 Turn 句柄。每个会话同一时刻只能有一个 Turn：
# 存在句柄时 ``turn.send`` 以 -32003 拒绝，``turn.cancel`` 则取消该句柄。build_tui sink
# 会在每个 Turn 结束时通过 ``clear_active`` 回调清空槽位，因此此处存在记录就表示正在执行。
_active_turns: dict[str, TurnHandle] = {}
_active_request_keys: dict[str, int] = {}


def is_turn_active(session_key: str) -> bool:
    """True if a turn is in flight for this session (the sink drops the slot on
    turn end, so presence is liveness)."""
    return session_key in _active_turns


def has_active_turns() -> bool:
    return bool(_active_turns)


def clear_active(session_key: str, request_key: int | None = None) -> None:
    """Drop the matching session active-turn slot.

    ``build_tui`` supplies the completed request's identity, so a preceding
    system turn on the same Spine lane cannot clear a queued user turn. A caller
    may omit it for unconditional teardown between independent lifecycles.
    """
    if request_key is not None and _active_request_keys.get(session_key) != request_key:
        return
    _active_turns.pop(session_key, None)
    _active_request_keys.pop(session_key, None)


# ---------------------------------------------------------------------------
# 可模拟边界
# ---------------------------------------------------------------------------


def _resolve_model(parsed: TurnSendParams) -> str:
    """Resolve the model id for a turn before spawning AgentLoop.

    Raises ``ModelNotAvailableError`` (-32008) if no provider/model is
    routable. The default impl is a no-op pass-through — AgentLoop owns the
    real model selection. Tests patch this seam to assert -32008 path.
    """
    return "default"


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------


async def _emit_start_then_error(
    emitter: SubscriptionEmitter,
    session_key: str,
    submission_id: str,
    turn_id: str,
    code: int,
    message: str,
    *,
    attachments_discarded: bool = False,
) -> None:
    # 先发 message.start，让前端有可清理的 Turn；随后错误会通过 onError 重置 turnId，
    # 与旧的单 Turn 任务保持相同形状。
    await emitter.emit(
        session_key,
        {
            "type": "message.start",
            "payload": {"submission_id": submission_id, "turn_id": turn_id},
        },
    )
    await emitter.emit(
        session_key,
        {
            "type": "error",
            "payload": {
                "attachments_discarded": attachments_discarded,
                "code": code,
                "message": message,
                "reason": "internal",
                "submission_id": submission_id,
                "turn_id": turn_id,
            },
        },
    )


async def turn_send(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
    scheduler: Scheduler | None = None,
    scheduler_factory: SchedulerFactory | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
    build_error: RpcError | None = None,
) -> dict[str, Any]:
    """``turn.send`` — submit a turn onto the spine, return ``{turn_id, accepted}``.

    The turn streams out via the build_tui runner/hub/sink. The runner emits
    message.start only when this exact request begins, followed by token deltas;
    the sink emits message.complete / error.

    Errors:
      -32003 (TurnInProgressError) — session already has an active TUI turn or
      its Scheduler Lane owns pending/running work.
      -32008 (ModelNotAvailableError) — no provider/model routable.
    """
    try:
        parsed = TurnSendParams.model_validate(params)
    except ValidationError as exc:
        # 原样重新抛出；分派器会捕获并发出 -32603 internal_error。
        raise exc

    if scheduler is None and scheduler_factory is not None:
        try:
            scheduler = await scheduler_factory()
        except RpcError as exc:
            build_error = exc

    # 快速失败：在占用活动 Turn 槽位之前检查模型可用性，避免 -32008 拒绝后
    # 会话无法继续发送请求。
    _resolve_model(parsed)

    turn_id = uuid4().hex
    submission_id = parsed.submission_id or uuid4().hex

    if scheduler is None:
        # 未连接 Agent Loop（构建失败或无提供商）。对当前 Turn 优先暴露构建错误自身的错误码，
        # 否则使用 -32008；不执行 Turn。
        attachments_discarded = bool(pending_images(parsed.session_key))
        consume_pending_images(parsed.session_key)
        if emitter is not None:
            if build_error is not None:
                await _emit_start_then_error(
                    emitter,
                    parsed.session_key,
                    submission_id,
                    turn_id,
                    build_error.code,
                    build_error.public_message,
                    attachments_discarded=attachments_discarded,
                )
            else:
                await _emit_start_then_error(
                    emitter,
                    parsed.session_key,
                    submission_id,
                    turn_id,
                    -32008,
                    "model_not_available",
                    attachments_discarded=attachments_discarded,
                )
        return {"turn_id": turn_id, "accepted": True}

    if is_turn_active(parsed.session_key) or scheduler.has_pending_or_running(parsed.session_key):
        raise TurnInProgressError(
            f"session {parsed.session_key!r} already has pending or running work",
        )

    req = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel=parsed.channel or "tui",
            chat_id=parsed.chat_id or "default",
            sender_id=parsed.sender_id or "user",
            chat_type=ChatType.DM,
        ),
        text=parsed.content,
        media=pending_images(parsed.session_key),
        # conversation 等于前端订阅键，使运行器的流和 sink 的 message.complete
        # 到达正确订阅。
        conversation=parsed.session_key,
    )
    request_key = id(req)
    try:
        handle = scheduler.submit(req)
    except SchedulerDrainingError:
        # 服务器正在关闭：发出 turn_failed 让前端清理槽位；此时未绑定任何内容，不会泄漏。
        attachments_discarded = bool(pending_images(parsed.session_key))
        consume_pending_images(parsed.session_key)
        if emitter is not None:
            await _emit_start_then_error(
                emitter,
                parsed.session_key,
                submission_id,
                turn_id,
                _TURN_FAILED_CODE,
                "turn_failed",
                attachments_discarded=attachments_discarded,
            )
        return {"turn_id": turn_id, "accepted": True}
    consume_pending_images(parsed.session_key)

    # submit 后立即绑定，中间不出现 await。worker 虽已排程，但在这些按请求存储的记录
    # 建立前无法运行。以请求身份为键，可防止同一会话通道中的先前系统 Turn
    # 消费当前用户 Turn 的关联信息。
    if turn_ids is not None:
        turn_ids[request_key] = turn_id
    if submission_ids is not None:
        submission_ids[request_key] = submission_id
    _active_turns[parsed.session_key] = handle
    _active_request_keys[parsed.session_key] = request_key

    return {"turn_id": turn_id, "accepted": True}


async def turn_subscribe(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
) -> dict[str, Any]:
    """``turn.subscribe`` — open a subscription, return ``{subscription_id}``."""
    parsed = TurnSubscribeParams.model_validate(params)
    if emitter is None:
        raise RuntimeError(
            "turn.subscribe requires a SubscriptionEmitter; register_turn_methods must be called with emitter=...",
        )
    sub_id = await emitter.register(parsed.session_key)
    return {"subscription_id": sub_id}


async def turn_unsubscribe(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
) -> dict[str, Any]:
    """``turn.unsubscribe`` — close a subscription (idempotent)."""
    parsed = TurnUnsubscribeParams.model_validate(params)
    if emitter is None:
        raise RuntimeError(
            "turn.unsubscribe requires a SubscriptionEmitter; register_turn_methods must be called with emitter=...",
        )
    unsubscribed = await emitter.unregister(parsed.subscription_id)
    return {"unsubscribed": unsubscribed}


async def turn_cancel(
    params: dict[str, Any],
    *,
    emitter: SubscriptionEmitter | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
) -> dict[str, Any]:
    """``turn.cancel`` — cancel the in-flight turn + notify subscribers.

    Sequence:
      1. Look up the active turn handle; if absent → ``{cancelled: False}``.
      2. ``handle.cancel()``.
      3. ``emitter.emit(session_key, error(reason="cancelled_by_client"))`` — the
         client resets its UI off this event. This is the ONLY cancelled-turn
         error; the sink stays silent on a cancelled TurnFailed (avoiding a
         double error), so this emit is the one signal that clears the front-end
         turn slot — it must always fire.
      4. Await the handle so the turn is provably unwound (the sink's TurnFailed
         handler drops the active-turn slot) before returning, so the next
         ``turn.send`` cannot race a half-unwound turn into a phantom -32003.
      5. Return ``{cancelled: True}``.

    The subscription is SESSION-scoped, not turn-scoped: a per-turn cancel ends
    only the turn and MUST leave the session's subscriptions open so the next
    turn's events still reach the client.
    """
    parsed = TurnCancelParams.model_validate(params)

    handle = _active_turns.get(parsed.session_key)
    if handle is None:
        return {"cancelled": False}
    request_key = _active_request_keys.get(parsed.session_key)

    handle.cancel()

    if emitter is not None:
        turn_id = turn_ids.get(request_key) if turn_ids is not None else None
        submission_id = submission_ids.get(request_key) if submission_ids is not None else None
        payload: dict[str, Any] = {
            "code": _TURN_FAILED_CODE,
            "message": "turn_cancelled",
            "reason": "cancelled_by_client",
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
        if submission_id is not None:
            payload["submission_id"] = submission_id
        await emitter.emit(
            parsed.session_key,
            {
                "type": "error",
                "payload": payload,
            },
        )

    # 返回前排空，确保 sink 已释放活动 Turn 槽位。
    # handle.result() 在取消时返回 None，不抛异常。
    await handle.result()
    clear_active(parsed.session_key, request_key)
    if request_key is not None:
        if turn_ids is not None:
            turn_ids.pop(request_key, None)
        if submission_ids is not None:
            submission_ids.pop(request_key, None)

    return {"cancelled": True}


# ---------------------------------------------------------------------------
# 分派器注册
# ---------------------------------------------------------------------------


def register_turn_methods(
    dispatcher: "Dispatcher",
    *,
    emitter: SubscriptionEmitter | None = None,
    scheduler: Scheduler | None = None,
    scheduler_factory: SchedulerFactory | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
    build_error: RpcError | None = None,
) -> None:
    """Register ``turn.{send,subscribe,unsubscribe,cancel}`` on a dispatcher.

    Wraps the four module-level handlers in single-argument closures that
    pre-bind the ``emitter`` and the build_tui spine bundle (``scheduler`` /
    ``turn_ids``) plus the latched ``build_error``, per the dispatcher's
    single-argument handler contract.
    """

    async def _send(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_send(
            params,
            emitter=emitter,
            scheduler=scheduler,
            scheduler_factory=scheduler_factory,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
            build_error=build_error,
        )

    async def _subscribe(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_subscribe(params, emitter=emitter)

    async def _unsubscribe(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_unsubscribe(params, emitter=emitter)

    async def _cancel(params: dict[str, Any]) -> dict[str, Any]:
        return await turn_cancel(
            params,
            emitter=emitter,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )

    dispatcher.register("turn.send", _send)
    dispatcher.register("turn.subscribe", _subscribe)
    dispatcher.register("turn.unsubscribe", _unsubscribe)
    dispatcher.register("turn.cancel", _cancel)


__all__ = [
    "has_active_turns",
    "register_turn_methods",
    "turn_send",
    "turn_subscribe",
    "turn_unsubscribe",
    "turn_cancel",
]
