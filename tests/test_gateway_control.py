import asyncio

import pytest

from pico.channels.control import (
    GatewayControlAdapter,
    GatewayControlOperation,
)
from pico.spine import (
    BusyPolicy,
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from pico.tui_rpc.question_broker import QuestionBroker


def _request(
    text: str,
    *,
    channel: str = "gateway-test",
    chat_id: str = "chat",
    conversation: str | None = None,
    busy: BusyPolicy = BusyPolicy.APPEND,
) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat_id, sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation=conversation,
        busy=busy,
    )


def _id_factory(*values: str):
    ids = iter(values)
    return lambda: next(ids)


async def _sink(_event) -> None:
    pass


class BlockingRunner:
    def __init__(self) -> None:
        self.requests: list[TurnRequest] = []
        self.started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.requests.append(req)
        self.started.set()
        await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class NoQuestions:
    def pending_req(self, conversation_id: str) -> str | None:
        return None

    def reply(self, key: str, answer: str) -> bool:
        raise AssertionError("no question is pending")


def _scheduler(runner: BlockingRunner, *turn_ids: str) -> Scheduler:
    return Scheduler(
        runner,
        OriginPools(user=1, system=1),
        _sink,
        turn_id_factory=_id_factory(*turn_ids),
    )


async def _stop_and_shutdown(adapter: GatewayControlAdapter, scheduler: Scheduler) -> None:
    await adapter.dispatch(_request("/stop"))
    await scheduler.shutdown(grace=0.0)


async def test_status_reports_idle_conversation_without_scheduling_work():
    runner = BlockingRunner()
    scheduler = _scheduler(runner)
    adapter = GatewayControlAdapter(scheduler, NoQuestions())

    result = await adapter.dispatch(_request("/status"))

    assert result.operation is GatewayControlOperation.STATUS
    assert result.status is not None and result.status.is_idle
    assert runner.requests == []


async def test_status_reports_one_running_turn():
    runner = BlockingRunner()
    scheduler = _scheduler(runner, "turn-running")
    adapter = GatewayControlAdapter(scheduler, NoQuestions())
    running = scheduler.submit(_request("work"))
    await runner.started.wait()

    result = await adapter.dispatch(_request("/status"))

    assert result.status is not None
    assert result.status.running_turn_id == "turn-running"
    assert result.status.queued_turn_ids == ()
    await _stop_and_shutdown(adapter, scheduler)
    assert await running.result() is None


async def test_status_reports_running_and_queued_turns_in_order():
    runner = BlockingRunner()
    scheduler = _scheduler(runner, "turn-running", "turn-queued-1", "turn-queued-2")
    adapter = GatewayControlAdapter(scheduler, NoQuestions())
    running = scheduler.submit(_request("running"))
    await runner.started.wait()
    queued_1 = scheduler.submit(_request("queued-1"))
    queued_2 = scheduler.submit(_request("queued-2"))

    result = await adapter.dispatch(_request("/status"))

    assert result.status is not None
    assert result.status.running_turn_id == "turn-running"
    assert result.status.queued_turn_ids == ("turn-queued-1", "turn-queued-2")
    await _stop_and_shutdown(adapter, scheduler)
    assert await asyncio.gather(running.result(), queued_1.result(), queued_2.result()) == [None, None, None]


async def test_active_ordinary_message_uses_inject_and_status_reports_it_pending():
    runner = BlockingRunner()
    scheduler = _scheduler(runner, "turn-running")
    adapter = GatewayControlAdapter(scheduler, NoQuestions())
    running = scheduler.submit(_request("running"))
    await runner.started.wait()

    acknowledgement = await adapter.dispatch(_request("late constraint"))
    status = await adapter.dispatch(_request("/status"))

    assert acknowledgement.operation is GatewayControlOperation.INJECT
    assert acknowledgement.correlated_turn_id == "turn-running"
    assert acknowledgement.status is not None
    assert acknowledgement.status.queued_turn_ids == ()
    assert acknowledgement.status.has_pending_inject is True
    assert status.status is not None and status.status.has_pending_inject is True
    await _stop_and_shutdown(adapter, scheduler)
    assert await running.result() is None


async def test_idle_ordinary_message_enters_the_normal_scheduler_path():
    runner = BlockingRunner()
    scheduler = _scheduler(runner, "turn-new")
    adapter = GatewayControlAdapter(scheduler, NoQuestions())

    result = await adapter.dispatch(_request("new work"))
    await runner.started.wait()

    assert result.operation is GatewayControlOperation.SUBMIT
    assert [request.text for request in runner.requests] == ["new work"]
    assert runner.requests[0].busy is BusyPolicy.APPEND
    await _stop_and_shutdown(adapter, scheduler)


async def test_stop_cancels_running_queued_inject_and_related_conversation_work():
    runner = BlockingRunner()
    scheduler = _scheduler(runner, "turn-running", "turn-queued")
    cancelled_related: list[str] = []

    async def cancel_related(conversation_id: str) -> int:
        cancelled_related.append(conversation_id)
        return 2

    adapter = GatewayControlAdapter(scheduler, NoQuestions(), cancel_related=cancel_related)
    running = scheduler.submit(_request("running"))
    await runner.started.wait()
    queued = scheduler.submit(_request("queued"))
    inject = scheduler.submit(_request("inject", busy=BusyPolicy.INJECT))

    result = await adapter.dispatch(_request("/stop"))

    assert result.operation is GatewayControlOperation.STOP
    assert result.stopped_count == 5
    assert cancelled_related == ["gateway-test:chat"]
    assert await asyncio.gather(running.result(), queued.result(), inject.result()) == [None, None, None]
    assert scheduler.conversation_status("gateway-test:chat").is_idle
    await scheduler.shutdown(grace=0.0)


async def test_pending_ask_user_answer_bypasses_agent_submission():
    frames: list[dict] = []
    question_sent = asyncio.Event()

    async def send_frame(frame: dict) -> None:
        frames.append(frame)
        question_sent.set()

    runner = BlockingRunner()
    scheduler = _scheduler(runner)
    broker = QuestionBroker(send_frame)
    adapter = GatewayControlAdapter(scheduler, broker)
    waiting = asyncio.create_task(
        broker.await_question(
            "gateway-test:chat",
            prompt="Which option?",
            timeout_s=1.0,
        )
    )
    await question_sent.wait()

    result = await adapter.dispatch(_request("option two"))

    assert result.operation is GatewayControlOperation.QUESTION_REPLY
    assert await waiting == "option two"
    assert frames[0]["params"]["conversation_id"] == "gateway-test:chat"
    assert runner.requests == []
    assert scheduler.conversation_status("gateway-test:chat").is_idle


async def test_status_and_stop_do_not_invoke_the_runner_directly():
    runner = BlockingRunner()
    scheduler = _scheduler(runner)
    adapter = GatewayControlAdapter(scheduler, NoQuestions())

    await adapter.dispatch(_request("/status"))
    await adapter.dispatch(_request("/stop"))

    assert runner.requests == []


@pytest.mark.parametrize("channel", ["feishu", "telegram"])
async def test_control_semantics_are_independent_of_channel_transport(channel: str):
    runner = BlockingRunner()
    scheduler = _scheduler(runner)
    adapter = GatewayControlAdapter(scheduler, NoQuestions())

    result = await adapter.dispatch(_request("/status", channel=channel))

    assert result.operation is GatewayControlOperation.STATUS
    assert result.conversation_id == f"{channel}:chat"
    assert result.status is not None and result.status.is_idle
