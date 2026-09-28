import asyncio

import pytest

from pico.cli._gateway_spine import build_gateway
from pico.spine import (
    ChatType,
    Origin,
    Source,
    Text,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from pico.spine.delivery import DeliveryResult, TerminalDeliveryError
from pico.tracing import semconv


def _request(text: str, *, channel: str = "control", conversation: str = "control:chat") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id="chat", sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation=conversation,
    )


def _id_factory(*values: str):
    ids = iter(values)
    return lambda: next(ids)


class RecordingChannel:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls = 0
        self.sent: list[tuple[str, str]] = []

    async def send(self, chat_id: str, content: str, media=None) -> None:
        self.calls += 1
        if self.fail:
            raise TerminalDeliveryError("transport rejected notification")
        self.sent.append((chat_id, content))


class ScriptedAgent:
    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self.fail = fail
        self.block = block
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.notify_count = 0

    def _notify_turn_complete(self) -> None:
        self.notify_count += 1

    async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None) -> TurnOutcome:
        self.calls.append(req.text)
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("provider failed")
        await emit(Text(content=f"result:{req.text}"))
        return TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)


def _collectors():
    terminals: list[TurnEnded | TurnFailed] = []
    deliveries: list[DeliveryResult] = []

    async def terminal_sink(event: TurnEnded | TurnFailed) -> None:
        terminals.append(event)

    async def delivery_sink(result: DeliveryResult) -> None:
        deliveries.append(result)

    return terminals, deliveries, terminal_sink, delivery_sink


@pytest.mark.parametrize("channel_name", ["telegram", "feishu"])
async def test_success_preserves_turn_id_across_terminal_and_delivery_evidence(channel_name: str):
    agent = ScriptedAgent()
    channel = RecordingChannel(channel_name)
    terminals, deliveries, terminal_sink, delivery_sink = _collectors()
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {channel_name: channel},
        turn_id_factory=_id_factory("turn-success"),
        on_terminal_event=terminal_sink,
        on_delivery_result=delivery_sink,
    )
    try:
        handle = scheduler.submit(_request("success", channel=channel_name, conversation=f"{channel_name}:chat"))
        await handle.result()
        await hub.wait_idle(channel_name)

        assert handle.turn_id == "turn-success"
        assert [(type(event), event.turn_id) for event in terminals] == [(TurnEnded, "turn-success")]
        assert [(result.turn_id, result.outcome) for result in deliveries] == [
            ("turn-success", semconv.CHANNEL_DELIVERED)
        ]
        assert deliveries[0].conversation_id == f"{channel_name}:chat"
        assert scheduler.conversation_status(f"{channel_name}:chat").is_idle
    finally:
        await teardown()


async def test_runner_failure_and_successful_failure_notification_keep_separate_outcomes():
    agent = ScriptedAgent(fail=True)
    channel = RecordingChannel("control")
    terminals, deliveries, terminal_sink, delivery_sink = _collectors()
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {"control": channel},
        turn_id_factory=_id_factory("turn-failed"),
        on_terminal_event=terminal_sink,
        on_delivery_result=delivery_sink,
    )
    try:
        handle = scheduler.submit(_request("failure"))
        assert await handle.result() is None
        await hub.wait_idle("control")

        assert len(terminals) == 1
        assert isinstance(terminals[0], TurnFailed)
        assert terminals[0].turn_id == "turn-failed"
        assert terminals[0].cancelled is False
        assert [(result.turn_id, result.outcome) for result in deliveries] == [
            ("turn-failed", semconv.CHANNEL_DELIVERED)
        ]
        assert channel.sent == [("chat", "Sorry, I encountered an error.")]
    finally:
        await teardown()


async def test_cancellation_terminal_evidence_keeps_the_running_turn_id():
    agent = ScriptedAgent(block=True)
    channel = RecordingChannel("control")
    terminals, deliveries, terminal_sink, delivery_sink = _collectors()
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {"control": channel},
        turn_id_factory=_id_factory("turn-cancelled"),
        on_terminal_event=terminal_sink,
        on_delivery_result=delivery_sink,
    )
    try:
        handle = scheduler.submit(_request("cancel"))
        await agent.started.wait()
        handle.cancel()
        assert await handle.result() is None

        assert len(terminals) == 1
        assert isinstance(terminals[0], TurnFailed)
        assert terminals[0].turn_id == "turn-cancelled"
        assert terminals[0].cancelled is True
        assert deliveries == []
        assert channel.sent == []
    finally:
        await teardown()


async def test_two_independent_turns_keep_terminal_and_delivery_results_separate():
    agent = ScriptedAgent()
    channel = RecordingChannel("control")
    terminals, deliveries, terminal_sink, delivery_sink = _collectors()
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {"control": channel},
        turn_id_factory=_id_factory("turn-one", "turn-two"),
        on_terminal_event=terminal_sink,
        on_delivery_result=delivery_sink,
    )
    try:
        first = scheduler.submit(_request("one"))
        second = scheduler.submit(_request("two"))
        await asyncio.gather(first.result(), second.result())
        await hub.wait_idle("control")

        assert [event.turn_id for event in terminals] == ["turn-one", "turn-two"]
        assert [result.turn_id for result in deliveries] == ["turn-one", "turn-two"]
        assert channel.sent == [("chat", "result:one"), ("chat", "result:two")]
    finally:
        await teardown()


async def test_successful_turn_with_failed_delivery_stays_successful_and_is_not_rerun():
    agent = ScriptedAgent()
    channel = RecordingChannel("control", fail=True)
    terminals, deliveries, terminal_sink, delivery_sink = _collectors()
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {"control": channel},
        turn_id_factory=_id_factory("turn-delivery-failed"),
        on_terminal_event=terminal_sink,
        on_delivery_result=delivery_sink,
    )
    try:
        handle = scheduler.submit(_request("delivery failure"))
        outcome = await handle.result()
        await hub.wait_idle("control")

        assert isinstance(outcome, TurnOutcome)
        assert len(terminals) == 1 and isinstance(terminals[0], TurnEnded)
        assert terminals[0].turn_id == "turn-delivery-failed"
        assert len(deliveries) == 1
        assert deliveries[0].turn_id == "turn-delivery-failed"
        assert deliveries[0].outcome == semconv.CHANNEL_DROPPED
        assert deliveries[0].attempts == 1
        assert deliveries[0].error == "TerminalDeliveryError"
        assert agent.calls == ["delivery failure"]
        assert channel.calls == 1
    finally:
        await teardown()
