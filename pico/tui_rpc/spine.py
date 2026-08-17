"""Spine wiring for the TUI RPC turn path: the runner (a TuiTurnRunner driving
the agent loop's native run_turn), the outlet that maps each
spine event to its wire event (token.delta / thinking.delta / tool.*), and the
sink that fires ``message.complete`` / ``error`` after the render barrier.

The TUI runs turns through spine (submit -> lane -> run_turn -> hub -> outlet).
All of token/reasoning/tool/Text flow through the hub to the TuiOutlet, so they
share one per-outlet FIFO. spine never imports tui_rpc; tui_rpc imports spine.

Why ``message.complete`` is fired from the sink (not from a stream-close): it is
an unconditional per-turn signal — the front-end clears its turn slot on it, so a
turn that streams nothing (empty reply, tool-only) must still emit it or the UI
wedges. The sink awaits ``wait_idle`` first so it lands after the turn's last
``token.delta``; an empty turn never built a queue, so the barrier returns at
once. This is the REPL's ``result() -> wait_idle`` render barrier moved into the
sink.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pico.agent.spine_runner import AgentTurnRunner
from pico.agent.tools.message import MessageTool
from pico.spine import (
    Deliverable,
    Origin,
    OriginPools,
    Reasoning,
    RunnerEvent,
    Scheduler,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnStarted,
)
from pico.spine.delivery import Capabilities, DeliveryHub
from pico.spine.events import TurnEvent
from pico.spine.runner import Drain, Emit
from pico.spine.teardown import teardown_spine
from pico.tui_rpc.errors import RpcError
from pico.tui_rpc.subscriptions import SubscriptionEmitter

_TURN_FAILED_CODE = -32099


def _conversation_id(req: TurnRequest) -> str:
    return req.conversation or f"{req.source.channel}:{req.source.chat_id}"


class TuiTurnRunner(AgentTurnRunner):
    """Runs a TUI turn through the agent loop's native run_turn. User turns
    stream token/reasoning/tool/Text through the hub to the TuiOutlet (one
    per-outlet FIFO — no dual path). Three TUI-specific bits the generic runner
    does not carry:

    - it emits ``message.start`` when the exact bound TUI request starts and
      suppresses deliverables from unbound system requests on the same lane;
    - it passes its own ``usage_sink`` so the sink can attach the full usage
      (cost / context, richer than the three-field TurnOutcome.usage) to
      ``message.complete``; the rich usage stays TUI-internal, off the wire;
    - it fires the synthetic tool.complete when the turn replied via the
      message tool (the loop's general tool path skips the message tool), so the
      UI records that the agent acted.
    """

    def __init__(
        self,
        agent_loop: Any,
        emitter: SubscriptionEmitter,
        usages: dict[int, dict[str, Any]],
        turn_ids: dict[int, str],
        readback_texts: dict[str, str],
        running_requests: dict[str, int] | None = None,
        submission_ids: dict[int, str] | None = None,
        rpc_errors: dict[int, RpcError] | None = None,
        await_runtime_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(agent_loop, stream=True)
        self._emitter = emitter
        self._usages = usages
        self._turn_ids = turn_ids
        self._running_requests = running_requests if running_requests is not None else {}
        self._submission_ids = submission_ids if submission_ids is not None else {}
        self._rpc_errors = rpc_errors if rpc_errors is not None else {}
        self._await_runtime_ready = await_runtime_ready
        self._readback_texts = readback_texts

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        cid = _conversation_id(req)
        request_key = id(req)
        self._running_requests[cid] = request_key
        turn_id = self._turn_ids.get(request_key)
        submission_id = self._submission_ids.get(request_key)
        bound = req.origin is Origin.USER and turn_id is not None and submission_id is not None

        async def emit_bound(event: RunnerEvent) -> None:
            if bound:
                await emit(event)

        if bound:
            await self._emitter.emit(
                cid,
                {
                    "type": "message.start",
                    "payload": {
                        "submission_id": submission_id,
                        "turn_id": turn_id,
                    },
                },
            )

        if self._await_runtime_ready is not None:
            try:
                await self._await_runtime_ready()
            except RpcError as exc:
                self._rpc_errors[request_key] = exc
                raise

        # CRON Turn 不是用户 Turn：它以非流式方式运行（一次回复，而非令牌流），
        # 并捕获回复文本供 cron 扇出，后者向每个会话交付 cron.delivered 事件。
        # 未绑定的可交付项在此处被抑制，不进入用户聊天流。该路径与网关的
        # GatewayTurnRunner 读回路径一致。
        if req.origin is Origin.CRON:
            text_sink: dict[str, str] = {}
            outcome = await self._loop.run_turn(req, emit_bound, drain, stream=False, text_sink=text_sink)
            if req.conversation is not None and (text := text_sink.get("text")) is not None:
                self._readback_texts[req.conversation] = text
            return outcome
        if req.origin is Origin.SUBAGENT:
            text_sink = {}
            outcome = await self._loop.run_turn(req, emit_bound, drain, stream=False, text_sink=text_sink)
            if text := text_sink.get("text"):
                await self._emitter.emit(
                    cid,
                    {
                        "type": "subagent.delivered",
                        "payload": {"text": text},
                    },
                )
            return outcome
        usage_sink: dict[str, Any] = {}
        outcome = await self._loop.run_turn(req, emit_bound, drain, stream=True, usage_sink=usage_sink)

        # 消息工具触发时生成一个合成 tool.complete（Agent Loop 在通用工具路径中会跳过它），
        # 让 UI 记录 Agent 已采取行动；回复已作为令牌增量流式输出。返回前发出，
        # 确保它在 Turn 事件流中位于 TurnEnded 之前。
        message_tool = self._loop.tools.get("message")
        if bound and isinstance(message_tool, MessageTool) and message_tool.sent_in_turn:
            await emit(
                ToolEvent(
                    phase=ToolPhase.COMPLETE,
                    tool_call_id=f"msg-{turn_id}",
                    result_preview="(message sent via tool)",
                )
            )

        self._usages[request_key] = dict(usage_sink)
        return outcome


class TuiOutlet:
    """The TUI's send surface. Maps each spine event to its wire event on the
    conversation's subscription: streamed token content via ``send_stream_chunk``
    (-> token.delta), and the discrete deliverables via ``deliver`` (Reasoning ->
    thinking.delta, ToolEvent -> tool.start / tool.complete, a non-streamed Text
    -> a token.delta). The turn's completion (``message.complete``) and failure
    (``error``) are emitted by the sink after the render barrier. Notice and
    MediaOut are eaten — the wire protocol has no event for them and the TUI shows
    no per-turn progress or tool media today (a known gap, deferred)."""

    def __init__(self, channel: str, emitter: SubscriptionEmitter) -> None:
        self.name = channel
        self.capabilities = Capabilities(streaming=True)
        self._emitter = emitter

    async def deliver(self, out: Deliverable) -> None:
        cid = out.conversation_id
        if isinstance(out, Reasoning):
            if out.content:
                await self._emitter.emit(cid, {"type": "thinking.delta", "payload": {"text": out.content}})
        elif isinstance(out, ToolEvent):
            if out.phase is ToolPhase.START:
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.start",
                        "payload": {
                            "tool_call_id": out.tool_call_id,
                            "name": out.name,
                            "arguments": out.arguments or {},
                        },
                    },
                )
            else:
                payload = {
                    "tool_call_id": out.tool_call_id,
                    "result_preview": out.result_preview,
                    "truncated": out.truncated,
                }
                if out.failed:
                    payload["failed"] = True
                await self._emitter.emit(
                    cid,
                    {
                        "type": "tool.complete",
                        "payload": payload,
                    },
                )
        elif isinstance(out, Text):
            # 非流式回复（澄清、Hook 短路或空回退）通过一个 token.delta 进入流式回复使用的
            # 同一缓冲区，使 message.complete 能像处理其他文本一样完成它。
            if out.content:
                await self._emitter.emit(cid, {"type": "token.delta", "payload": {"text": out.content}})
            # Notice 和 MediaOut 当前没有线上事件，因此被消耗。

    async def send_stream_chunk(self, chat_id: str, stream_id: str, delta: str, *, done: bool = False) -> None:
        if done:
            # 前端没有 stream-done 事件；Turn 由 sink 发出的 message.complete 完结。
            # done=True 只让 hub 关闭其流状态。
            return
        if not delta:
            return
        await self._emitter.emit(stream_id, {"type": "token.delta", "payload": {"text": delta}})

    async def emit_complete(
        self,
        conversation_id: str,
        submission_id: str,
        turn_id: str,
        usage: dict[str, Any],
    ) -> None:
        await self._emitter.emit(
            conversation_id,
            {
                "type": "message.complete",
                "payload": {
                    "submission_id": submission_id,
                    "turn_id": turn_id,
                    "usage": usage,
                },
            },
        )

    async def emit_error(
        self,
        conversation_id: str,
        submission_id: str,
        turn_id: str,
        code: int,
        message: str,
        reason: str,
    ) -> None:
        await self._emitter.emit(
            conversation_id,
            {
                "type": "error",
                "payload": {
                    "code": code,
                    "message": message,
                    "reason": reason,
                    "submission_id": submission_id,
                    "turn_id": turn_id,
                },
            },
        )


def _make_tui_sink(
    hub: DeliveryHub,
    outlet: TuiOutlet,
    channel: str,
    turn_ids: dict[int, str],
    submission_ids: dict[int, str],
    usages: dict[int, dict[str, Any]],
    running_requests: dict[str, int],
    rpc_errors: dict[int, RpcError],
    on_turn_end: Callable[[str, int], None] | None,
) -> Callable[[TurnEvent], Awaitable[None]]:
    """Adapt the hub into the scheduler's EventSink for the TUI. Deliverables
    route through the hub; a turn's end fires message.complete / error after the
    render barrier (so they land after the last token.delta). ``on_turn_end`` is
    called at each turn exit (before message.complete) so turn.send's active-turn
    slot is cleared before the front-end is told it may submit the next turn.
    It receives the exact request identity, so unrelated turns sharing the lane
    cannot clear that slot.
    This sink is build_tui's alone — the CLI keeps its own lifecycle-dropping
    sink."""

    async def _finish(conversation_id: str) -> None:
        # close_stream 清除 hub 按流保存的状态，使该会话的下一个 Turn 能干净重开；
        # 随后 wait_idle 阻塞，直到所有排队的 token.delta 都已交付。空 Turn 从未创建队列，
        # 因此会立即返回。
        await hub.close_stream(conversation_id)
        await hub.wait_idle(channel)

    def _correlation(conversation_id: str) -> tuple[int | None, str | None, str | None]:
        request_key = running_requests.get(conversation_id)
        if request_key is None:
            return None, None, None
        return request_key, turn_ids.get(request_key), submission_ids.get(request_key)

    def _drop(conversation_id: str, request_key: int | None) -> None:
        if request_key is None:
            return
        if running_requests.get(conversation_id) == request_key:
            running_requests.pop(conversation_id, None)
        turn_ids.pop(request_key, None)
        submission_ids.pop(request_key, None)
        usages.pop(request_key, None)
        rpc_errors.pop(request_key, None)
        if on_turn_end is not None:
            on_turn_end(conversation_id, request_key)

    async def sink(event: TurnEvent) -> None:
        if isinstance(event, TurnEnded):
            await _finish(event.conversation_id)
            request_key, turn_id, submission_id = _correlation(event.conversation_id)
            bound = turn_id is not None and submission_id is not None
            usage = usages.get(request_key) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            _drop(event.conversation_id, request_key)
            if not bound:
                return
            await outlet.emit_complete(event.conversation_id, submission_id, turn_id, usage)
            return
        if isinstance(event, TurnFailed):
            await _finish(event.conversation_id)
            request_key, turn_id, submission_id = _correlation(event.conversation_id)
            bound = turn_id is not None and submission_id is not None
            rpc_error = rpc_errors.get(request_key) if request_key is not None else None
            _drop(event.conversation_id, request_key)
            # 已取消 Turn 的错误由 turn.cancel 发出，此处不再发送，避免重复错误事件。
            if bound and not event.cancelled:
                code = rpc_error.code if rpc_error is not None else _TURN_FAILED_CODE
                message = rpc_error.public_message if rpc_error is not None else "turn_failed"
                await outlet.emit_error(
                    event.conversation_id,
                    submission_id,
                    turn_id,
                    code,
                    message,
                    "internal",
                )
            return
        if isinstance(event, TurnStarted):
            # TuiTurnRunner 将精确请求与对应 turn.send 绑定关联后，再发出 message.start。
            return
        await hub.dispatch(event)

    return sink


def build_tui(
    agent_loop: Any,
    emitter: SubscriptionEmitter,
    *,
    channel: str = "tui",
    on_turn_end: Callable[[str, int], None] | None = None,
    readback_texts: dict[str, str] | None = None,
    user_pool: int = 1,
    system_pool: int = 1,
    await_runtime_ready: Callable[[], Awaitable[None]] | None = None,
    turn_ids: dict[int, str] | None = None,
    submission_ids: dict[int, str] | None = None,
) -> tuple[Scheduler, DeliveryHub, dict[int, str], dict[int, str], Callable[[], Awaitable[None]]]:
    """Wire the spine pieces a TUI turn flows through: a hub with the channel's
    TuiOutlet, and a Scheduler whose runner streams the agent loop and whose sink
    fires message.complete / error after the render barrier. Returns those plus
    the per-request ``turn_ids`` and ``submission_ids`` maps used to correlate
    terminal events, and a ``teardown`` the caller awaits on exit.
    ``on_turn_end`` lets turn.send drop only the matching active-turn slot.

    ``readback_texts`` is the cron read-back map (conversation -> reply text): the
    runner stores a CRON turn's reply there so the cron fan-out can deliver it as a
    cron.delivered event. Pass the same dict the cron callback reads; defaults to a
    private map when cron is not wired (e.g. tests)."""
    hub = DeliveryHub()
    outlet = TuiOutlet(channel, emitter)
    hub.register(outlet)
    turn_ids = turn_ids if turn_ids is not None else {}
    submission_ids = submission_ids if submission_ids is not None else {}
    usages: dict[int, dict[str, Any]] = {}
    running_requests: dict[str, int] = {}
    rpc_errors: dict[int, RpcError] = {}
    if readback_texts is None:
        readback_texts = {}
    scheduler = Scheduler(
        TuiTurnRunner(
            agent_loop,
            emitter,
            usages,
            turn_ids,
            readback_texts,
            running_requests,
            submission_ids,
            rpc_errors,
            await_runtime_ready,
        ),
        OriginPools(user=user_pool, system=system_pool),
        _make_tui_sink(
            hub,
            outlet,
            channel,
            turn_ids,
            submission_ids,
            usages,
            running_requests,
            rpc_errors,
            on_turn_end,
        ),
    )

    async def teardown() -> None:
        await teardown_spine(scheduler, hub, grace=0.0)

    return scheduler, hub, turn_ids, submission_ids, teardown
