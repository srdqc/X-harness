from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pico.agent.loop.main import ProviderTurnError
from pico.spine import ChatType, Origin, Source, Text, TurnOutcome, TurnRequest, Usage
from pico.spine.delivery import Capabilities, DeliveryHub, TerminalDeliveryError
from pico.spine.scheduler import Lane, OriginPools
from pico.tracing import evidence
from pico.tracing import spans as tracing_spans
from pico.tracing.store import TraceStore


@pytest.fixture
def evidence_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "traces"
    monkeypatch.setenv("PICO_TRACING", "1")
    monkeypatch.setenv("PICO_TRACING_DIR", str(state_dir))
    tracing_spans._store = None
    yield state_dir
    tracing_spans._store = None


def _request(text: str = "hello") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="test", chat_id="chat", sender_id="user", chat_type=ChatType.DM),
        text=text,
    )


class _Runner:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error

    async def run(self, req, emit, drain) -> TurnOutcome:
        if self.error is not None:
            raise self.error
        return TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)


async def _run_lane(runner, turn_id: str = "turn-1"):
    events = []

    async def sink(event):
        events.append(event)

    lane = Lane(
        runner=runner,
        pools=OriginPools(user=1, system=1),
        sink=sink,
        conversation_id="test:chat",
        turn_id_factory=lambda: turn_id,
    )
    result = await lane.submit(_request())
    return result, events


async def test_success_turn_is_versioned_complete_and_keyed_by_turn_id(evidence_dir):
    outcome, _ = await _run_lane(_Runner(), "turn-success")
    result = evidence.read_turn_evidence(evidence_dir, "turn-success")

    assert outcome is not None
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE
    assert [event.event_type for event in result.events] == [evidence.TURN_STARTED, evidence.TURN_TERMINAL]
    assert {event.schema_version for event in result.events} == {1}
    assert {event.turn_id for event in result.events} == {"turn-success"}
    assert result.events[-1].metadata["outcome"] == "completed"


async def test_trace_identity_is_correlated_but_remains_separate(evidence_dir):
    await _run_lane(_Runner(), "turn-runtime")
    result = evidence.read_turn_evidence(evidence_dir, "turn-runtime")

    trace_ids = {event.trace_id for event in result.events}
    assert len(trace_ids) == 1
    assert None not in trace_ids
    assert trace_ids != {"turn-runtime"}


async def test_independent_turns_have_independent_monotonic_sequences(evidence_dir):
    await _run_lane(_Runner(), "turn-a")
    await _run_lane(_Runner(), "turn-b")

    assert [event.sequence for event in evidence.read_turn_evidence(evidence_dir, "turn-a").events] == [1, 2]
    assert [event.sequence for event in evidence.read_turn_evidence(evidence_dir, "turn-b").events] == [1, 2]


async def test_provider_and_generic_failures_have_durable_terminal_evidence(evidence_dir):
    await _run_lane(_Runner(error=ProviderTurnError("rate_limit")), "turn-provider")
    await _run_lane(_Runner(error=ValueError("secret failure detail")), "turn-error")

    provider = evidence.read_turn_evidence(evidence_dir, "turn-provider")
    generic = evidence.read_turn_evidence(evidence_dir, "turn-error")
    assert provider.completeness is evidence.EvidenceCompleteness.COMPLETE
    assert provider.events[-1].metadata == {
        "outcome": "provider_failed",
        "lifecycle_event": "TurnFailed",
        "error_class": "ProviderTurnError",
        "provider_error_category": "rate_limit",
    }
    assert generic.completeness is evidence.EvidenceCompleteness.COMPLETE
    assert generic.events[-1].metadata["outcome"] == "error"
    assert "secret failure detail" not in json.dumps(generic.events[-1].to_record())


async def test_cancellation_has_durable_terminal_evidence(evidence_dir):
    class HangingRunner:
        def __init__(self):
            self.started = asyncio.Event()

        async def run(self, req, emit, drain):
            self.started.set()
            await asyncio.Event().wait()

    runner = HangingRunner()
    lane = Lane(
        runner=runner,
        pools=OriginPools(user=1, system=1),
        sink=lambda event: asyncio.sleep(0),
        conversation_id="test:chat",
        turn_id_factory=lambda: "turn-cancel",
    )
    future = lane.submit(_request())
    await runner.started.wait()
    assert lane.cancel_running() == 1
    assert await future is None

    result = evidence.read_turn_evidence(evidence_dir, "turn-cancel")
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE
    assert result.events[-1].metadata["outcome"] == "cancelled"


def _record(turn_id: str, sequence: int, event_type: str, *, timestamp: str = "later") -> dict:
    metadata = {"outcome": "completed", "lifecycle_event": "TurnEnded"} if event_type == evidence.TURN_TERMINAL else {}
    return evidence.TurnEvidenceEvent(
        turn_id=turn_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=timestamp,
        trace_id=f"trace-{turn_id}",
        metadata=metadata,
    ).to_record()


def test_restart_readback_orders_by_sequence_not_timestamp_or_append_order(evidence_dir):
    store = TraceStore(evidence_dir)
    assert store.append_event(_record("turn-order", 2, evidence.TURN_TERMINAL, timestamp="earlier"))
    assert store.append_event(_record("turn-order", 1, evidence.TURN_STARTED, timestamp="later"))

    first = evidence.read_turn_evidence(evidence_dir, "turn-order")
    second = evidence.read_turn_evidence(Path(str(evidence_dir)), "turn-order")
    assert [event.sequence for event in first.events] == [1, 2]
    assert first == second
    assert first.completeness is evidence.EvidenceCompleteness.COMPLETE


@pytest.mark.parametrize(
    ("sequences", "expected", "finding"),
    [
        ((1, 1, 2), evidence.EvidenceCompleteness.CORRUPT, "duplicate_sequence"),
        ((1, 3), evidence.EvidenceCompleteness.PARTIAL, "missing_sequence"),
    ],
)
def test_duplicate_and_missing_sequences_are_detected(evidence_dir, sequences, expected, finding):
    store = TraceStore(evidence_dir)
    for index, sequence in enumerate(sequences):
        event_type = evidence.TURN_STARTED if index == 0 else evidence.TURN_TERMINAL
        assert store.append_event(_record("turn-gap", sequence, event_type))

    result = evidence.read_turn_evidence(evidence_dir, "turn-gap")
    assert result.completeness is expected
    assert finding in result.findings


def test_malformed_and_truncated_records_are_classified(evidence_dir):
    log = evidence_dir / "logs" / "audit-events.log"
    log.parent.mkdir(parents=True)
    log.write_text("{bad}\n", encoding="utf-8")
    malformed = evidence.read_turn_evidence(evidence_dir, "turn-x")
    assert malformed.completeness is evidence.EvidenceCompleteness.CORRUPT
    assert "malformed_record" in malformed.findings

    log.write_text('{"schema":"pico.turn.evidence.event.v1"', encoding="utf-8")
    truncated = evidence.read_turn_evidence(evidence_dir, "turn-x")
    assert truncated.completeness is evidence.EvidenceCompleteness.PARTIAL
    assert "truncated_final_record" in truncated.findings


async def test_write_failure_is_nonfatal_and_cannot_read_as_complete(evidence_dir, monkeypatch):
    real_emit = tracing_spans.emit_event
    attempts = 0

    def fail_first(record):
        nonlocal attempts
        attempts += 1
        return False if attempts == 1 else real_emit(record)

    monkeypatch.setattr(tracing_spans, "emit_event", fail_first)
    outcome, _ = await _run_lane(_Runner(), "turn-degraded")
    result = evidence.read_turn_evidence(evidence_dir, "turn-degraded")

    assert outcome is not None
    assert result.completeness is evidence.EvidenceCompleteness.PARTIAL
    assert {"missing_sequence", "write_degradation", "missing_start"} <= set(result.findings)


def test_sequence_lock_does_not_cover_writer_io():
    entered = 0
    all_entered = __import__("threading").Event()
    release = __import__("threading").Event()
    guard = __import__("threading").Lock()

    def writer(_record):
        nonlocal entered
        with guard:
            entered += 1
            if entered == 2:
                all_entered.set()
        assert all_entered.wait(1)
        assert release.wait(1)
        return True

    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-concurrent",
        conversation_id="test:chat",
        trace_id="trace-1",
        root_span_id="span-1",
        writer=writer,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(recorder.emit, "probe") for _ in range(2)]
        assert all_entered.wait(1)
        release.set()
        events = [future.result() for future in futures]
    assert sorted(event.sequence for event in events) == [1, 2]


def test_event_is_immutable_and_contains_no_large_payload_fields():
    event = evidence.TurnEvidenceEvent(
        turn_id="turn-privacy",
        sequence=1,
        event_type=evidence.TURN_STARTED,
        timestamp="now",
        metadata={"origin": "user"},
    )
    with pytest.raises(TypeError):
        event.metadata["prompt"] = "secret"
    record = event.to_record()
    assert not ({"prompt", "messages", "tool_arguments", "tool_output", "content"} & set(record))


class _DroppingOutlet:
    name = "test"
    capabilities = Capabilities()

    async def deliver(self, out):
        raise TerminalDeliveryError("remote detail")


class _SuccessfulOutlet:
    name = "test"
    capabilities = Capabilities()

    async def deliver(self, out):
        return None


async def test_runtime_and_delivery_outcomes_remain_separate(evidence_dir):
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-delivery",
        conversation_id="test:chat",
        trace_id="trace-1",
        root_span_id="span-1",
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )
    hub = DeliveryHub(send_max_retries=0)
    hub.register(_DroppingOutlet())
    try:
        with evidence.turn_scope(recorder):
            await hub.dispatch(
                Text(content="reply", source=_request().source, conversation_id="test:chat", turn_id="turn-delivery")
            )
        await hub.wait_idle("test")
    finally:
        await hub.aclose()

    result = evidence.read_turn_evidence(evidence_dir, "turn-delivery")
    terminal = next(event for event in result.events if event.event_type == evidence.TURN_TERMINAL)
    delivery = next(event for event in result.events if event.event_type == evidence.DELIVERY_OUTCOME)
    assert terminal.metadata["outcome"] == "completed"
    assert delivery.metadata["outcome"] == "dropped"
    assert delivery.metadata["attempts"] == 1
    assert "remote detail" not in json.dumps(delivery.to_record())
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE


async def test_failed_runtime_and_successful_failure_notification_remain_separate(evidence_dir):
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-failed-delivered",
        conversation_id="test:chat",
        trace_id="trace-1",
        root_span_id="span-1",
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "error", "lifecycle_event": "TurnFailed", "error_class": "ValueError"},
    )
    hub = DeliveryHub()
    hub.register(_SuccessfulOutlet())
    try:
        with evidence.turn_scope(recorder):
            await hub.dispatch(
                Text(
                    content="failure notification",
                    source=_request().source,
                    conversation_id="test:chat",
                    turn_id="turn-failed-delivered",
                )
            )
        await hub.wait_idle("test")
    finally:
        await hub.aclose()

    result = evidence.read_turn_evidence(evidence_dir, "turn-failed-delivered")
    outcomes = {event.event_type: event.metadata["outcome"] for event in result.events if "outcome" in event.metadata}
    assert outcomes[evidence.TURN_TERMINAL] == "error"
    assert outcomes[evidence.DELIVERY_OUTCOME] == "delivered"
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE


async def test_nonterminal_delivery_does_not_fabricate_delivery_evidence(evidence_dir):
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-no-delivery",
        conversation_id="test:chat",
        trace_id="trace-1",
        root_span_id="span-1",
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "cancelled", "lifecycle_event": "TurnFailed"},
    )
    result = evidence.read_turn_evidence(evidence_dir, "turn-no-delivery")
    assert all(event.event_type != evidence.DELIVERY_OUTCOME for event in result.events)
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE
