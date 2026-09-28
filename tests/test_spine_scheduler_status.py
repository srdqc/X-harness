import asyncio
from dataclasses import FrozenInstanceError

import pytest

from pico.spine import (
    BusyPolicy,
    ChatType,
    ConversationStatus,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnOutcome,
    TurnRequest,
    Usage,
)


def _request(text: str, *, busy: BusyPolicy = BusyPolicy.APPEND) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="test", chat_id="chat", sender_id="user", chat_type=ChatType.DM),
        text=text,
        busy=busy,
    )


def _id_factory(*values: str):
    ids = iter(values)
    return lambda: next(ids)


async def _sink(_event) -> None:
    pass


class ControlledRunner:
    def __init__(self, *, drain_injects: bool = False):
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.drained = asyncio.Event()
        self.ran: list[str] = []
        self._drain_injects = drain_injects

    async def run(self, req, emit, drain) -> TurnOutcome:
        started = self.started.setdefault(req.text, asyncio.Event())
        release = self.release.setdefault(req.text, asyncio.Event())
        self.ran.append(req.text)
        started.set()
        if self._drain_injects:
            await release.wait()
            drain()
            self.drained.set()
        else:
            await release.wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    async def wait_started(self, text: str) -> None:
        event = self.started.setdefault(text, asyncio.Event())
        await event.wait()

    def finish(self, text: str) -> None:
        self.release.setdefault(text, asyncio.Event()).set()


def _scheduler(runner: ControlledRunner, *turn_ids: str) -> Scheduler:
    return Scheduler(
        runner,
        OriginPools(user=1, system=1),
        _sink,
        turn_id_factory=_id_factory(*turn_ids),
    )


async def test_idle_conversation_has_an_immutable_empty_snapshot_without_creating_a_lane():
    sched = _scheduler(ControlledRunner())

    status = sched.conversation_status("test:missing")

    assert status == ConversationStatus("test:missing", None, (), False)
    assert status.is_idle is True
    assert status.has_scheduled_work is False
    assert "test:missing" not in sched._lanes


async def test_running_turn_is_projected_by_runtime_turn_id():
    runner = ControlledRunner()
    sched = _scheduler(runner, "turn-running")
    handle = sched.submit(_request("running"))
    await runner.wait_started("running")

    status = sched.conversation_status("test:chat")

    assert status.running_turn_id == "turn-running"
    assert status.queued_turn_ids == ()
    assert status.has_pending_inject is False
    assert status.has_scheduled_work is True
    runner.finish("running")
    await handle.result()


async def test_running_and_queued_turns_are_projected_in_execution_order():
    runner = ControlledRunner()
    sched = _scheduler(runner, "turn-running", "turn-queued-1", "turn-queued-2")
    running = sched.submit(_request("running"))
    await runner.wait_started("running")
    queued_1 = sched.submit(_request("queued-1"))
    queued_2 = sched.submit(_request("queued-2"))

    status = sched.conversation_status("test:chat")

    assert status.running_turn_id == "turn-running"
    assert status.queued_turn_ids == ("turn-queued-1", "turn-queued-2")
    runner.finish("running")
    await running.result()
    await runner.wait_started("queued-1")
    runner.finish("queued-1")
    await queued_1.result()
    await runner.wait_started("queued-2")
    runner.finish("queued-2")
    await queued_2.result()
    assert runner.ran == ["running", "queued-1", "queued-2"]


async def test_pending_and_merged_inject_never_appears_as_an_independent_turn():
    runner = ControlledRunner(drain_injects=True)
    sched = _scheduler(runner, "turn-host")
    host = sched.submit(_request("host"))
    await runner.wait_started("host")
    inject = sched.submit(_request("inject", busy=BusyPolicy.INJECT))

    pending = sched.conversation_status("test:chat")
    assert pending.running_turn_id == "turn-host"
    assert pending.queued_turn_ids == ()
    assert pending.has_pending_inject is True

    runner.finish("host")
    await runner.drained.wait()
    merged = sched.conversation_status("test:chat")
    assert merged.running_turn_id == "turn-host"
    assert merged.queued_turn_ids == ()
    assert merged.has_pending_inject is False
    await host.result()
    await inject.result()


async def test_undrained_inject_is_projected_with_its_fallback_turn_id():
    runner = ControlledRunner()
    sched = _scheduler(runner, "turn-host", "turn-fallback")
    host = sched.submit(_request("host"))
    await runner.wait_started("host")
    inject = sched.submit(_request("inject", busy=BusyPolicy.INJECT))
    runner.finish("host")
    await host.result()
    await runner.wait_started("inject")

    status = sched.conversation_status("test:chat")

    assert inject.turn_id == "turn-fallback"
    assert status.running_turn_id == "turn-fallback"
    assert status.queued_turn_ids == ()
    assert status.has_pending_inject is False
    runner.finish("inject")
    await inject.result()


async def test_cancel_cleanup_leaves_an_idle_snapshot():
    runner = ControlledRunner()
    sched = _scheduler(runner, "turn-running", "turn-queued")
    running = sched.submit(_request("running"))
    await runner.wait_started("running")
    queued = sched.submit(_request("queued"))
    inject = sched.submit(_request("inject", busy=BusyPolicy.INJECT))

    assert sched.cancel_conversation("test:chat") == 3
    await running.result()
    await queued.result()
    await inject.result()

    status = sched.conversation_status("test:chat")
    assert status.is_idle is True
    assert status.running_turn_id is None
    assert status.queued_turn_ids == ()
    assert status.has_pending_inject is False


async def test_status_queries_are_side_effect_free_and_cannot_mutate_lane_state():
    runner = ControlledRunner()
    sched = _scheduler(runner, "turn-running", "turn-queued")
    running = sched.submit(_request("running"))
    await runner.wait_started("running")
    queued = sched.submit(_request("queued"))

    first = sched.conversation_status("test:chat")
    second = sched.conversation_status("test:chat")

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.running_turn_id = "changed"
    with pytest.raises(FrozenInstanceError):
        first.queued_turn_ids += ("changed",)
    assert sched.conversation_status("test:chat") == second

    runner.finish("running")
    await running.result()
    await runner.wait_started("queued")
    runner.finish("queued")
    await queued.result()
    assert runner.ran == ["running", "queued"]
