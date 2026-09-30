from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pico.tracing import evidence, replay
from pico.tracing.store import TraceStore


def _recorder(state_dir: Path, turn_id: str = "turn-replay") -> evidence.TurnEvidenceRecorder:
    store = TraceStore(state_dir)
    return evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="test:chat",
        trace_id="trace-replay",
        root_span_id="span-turn",
        writer=store.append_event,
    )


def _start(recorder: evidence.TurnEvidenceRecorder) -> None:
    recorder.emit(evidence.TURN_STARTED)


def _terminal(
    recorder: evidence.TurnEvidenceRecorder,
    outcome: str = "completed",
    **metadata,
) -> None:
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={
            "outcome": outcome,
            "lifecycle_event": "TurnEnded" if outcome.startswith("completed") else "TurnFailed",
            **metadata,
        },
    )


def _provider(
    recorder: evidence.TurnEvidenceRecorder,
    logical_id: str,
    ordinal: int,
    *,
    outcome: str | None = "success",
    attempted_model: str = "model-a",
    usage: dict[str, int] | None = None,
) -> None:
    attempt_id = f"{logical_id}:attempt:{ordinal}"
    correlations = {
        "logical_call_id": logical_id,
        "attempt_id": attempt_id,
        "attempt_ordinal": ordinal,
    }
    recorder.emit(
        evidence.PROVIDER_ATTEMPT_STARTED,
        correlations=correlations,
        metadata={
            "receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA,
            "requested_model": "model-a",
            "attempted_model": attempted_model,
            "provider": "fixture",
            "request_digest": f"request-{ordinal}",
        },
    )
    if outcome is not None:
        recorder.emit(
            evidence.PROVIDER_ATTEMPT_COMPLETED,
            correlations=correlations,
            metadata={
                "receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA,
                "actual_model": attempted_model,
                "outcome": outcome,
                "finish_reason": "stop" if outcome == "success" else "error",
                "error_category": None if outcome == "success" else "network",
                "response_digest": f"response-{ordinal}",
                "usage_available": usage is not None,
                "usage": usage or {},
                "duration_ms": ordinal,
            },
        )


def _tool(
    recorder: evidence.TurnEvidenceRecorder,
    receipt_id: str,
    *,
    requested_name: str,
    resolved_name: str | None,
    model_call_id: str,
    parent_call_id: str | None = None,
    routed_via: str | None = None,
    effect: str = "read",
    outcome: str | None = "success",
    failure_stage: str | None = None,
    failure_category: str | None = None,
) -> None:
    correlations = {
        "receipt_id": receipt_id,
        "model_call_id": model_call_id,
        "parent_call_id": parent_call_id,
    }
    recorder.emit(
        evidence.TOOL_EXECUTION_STARTED,
        correlations=correlations,
        metadata={
            "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
            "requested_name": requested_name,
            "resolved_name": resolved_name,
            "routed_via": routed_via,
            "effect": effect,
            "argument_digest": f"arguments-{receipt_id}",
        },
    )
    if outcome is not None:
        recorder.emit(
            evidence.TOOL_EXECUTION_COMPLETED,
            correlations=correlations,
            metadata={
                "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
                "outcome": outcome,
                "failure_stage": failure_stage,
                "failure_category": failure_category,
                "result_digest": f"result-{receipt_id}",
                "result_size": 3,
                "duration_ms": 2,
            },
        )


@pytest.mark.parametrize(
    ("runtime_status", "delivery_status"),
    [
        ("completed", None),
        ("error", "delivered"),
        ("cancelled", None),
        ("completed", "dropped"),
    ],
)
def test_runtime_and_delivery_outcomes_remain_independent(
    tmp_path: Path,
    runtime_status: str,
    delivery_status: str | None,
) -> None:
    recorder = _recorder(tmp_path, f"turn-{runtime_status}-{delivery_status}")
    _start(recorder)
    _terminal(recorder, runtime_status)
    if delivery_status is not None:
        recorder.emit(
            evidence.DELIVERY_OUTCOME,
            correlations={"delivery_trace_id": "delivery-trace"},
            metadata={
                "channel": "test",
                "event": "Text",
                "outcome": delivery_status,
                "attempts": 1,
            },
        )

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert result.runtime_outcome is not None
    assert result.runtime_outcome.outcome == runtime_status
    assert [item.outcome for item in result.delivery_outcomes] == (
        [delivery_status] if delivery_status is not None else []
    )
    assert result.evidence_status is evidence.EvidenceCompleteness.COMPLETE


def test_provider_attempts_are_grouped_without_collapsing_retries_or_fallback(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _provider(recorder, "call-1", 1, outcome="error", usage={"prompt_tokens": 2})
    _provider(
        recorder,
        "call-1",
        2,
        attempted_model="model-b",
        usage={"prompt_tokens": 3, "completion_tokens": 1},
    )
    _provider(recorder, "call-2", 1, usage={"prompt_tokens": 5})
    _terminal(recorder)

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert [item.logical_call_id for item in result.provider_calls] == ["call-1", "call-2"]
    first = result.provider_calls[0]
    assert [item.outcome for item in first.attempts] == ["error", "success"]
    assert first.final_known_outcome == "success"
    assert first.retry_count == first.fallback_count == 1
    assert first.usage.totals == (("completion_tokens", 1), ("prompt_tokens", 5))
    assert first.usage.complete is True


def test_tool_projection_preserves_direct_meta_unresolved_and_failures(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _tool(
        recorder,
        "direct",
        requested_name="read_file",
        resolved_name="read_file",
        model_call_id="model-direct",
    )
    _tool(
        recorder,
        "meta",
        requested_name="write_file",
        resolved_name="write_file",
        model_call_id="model-meta:write_file",
        parent_call_id="model-meta",
        routed_via="tool_call",
        effect="write",
    )
    _tool(
        recorder,
        "unresolved",
        requested_name="stale_tool",
        resolved_name=None,
        model_call_id="model-stale",
        routed_via="tool_call",
        outcome="failure",
        failure_stage="resolution",
        failure_category="not_found",
    )
    _tool(
        recorder,
        "schema",
        requested_name="known",
        resolved_name="known",
        model_call_id="model-schema",
        outcome="failure",
        failure_stage="schema_validation",
        failure_category="invalid_arguments",
    )
    _tool(
        recorder,
        "timeout",
        requested_name="remote",
        resolved_name="remote",
        model_call_id="model-timeout",
        effect="external",
        outcome="failure",
        failure_stage="timeout",
        failure_category="timeout",
    )
    _terminal(recorder, tool_calls=4, tool_failures=3)

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    direct, meta, unresolved, schema, timeout = result.tool_executions
    assert (direct.requested_name, direct.resolved_name, direct.routed_via) == (
        "read_file",
        "read_file",
        None,
    )
    assert (meta.parent_call_id, meta.routed_via, meta.effect) == ("model-meta", "tool_call", "write")
    assert unresolved.resolved_name is None
    assert (unresolved.failure_stage, unresolved.failure_category) == ("resolution", "not_found")
    assert schema.failure_stage == "schema_validation"
    assert timeout.failure_stage == timeout.failure_category == "timeout"
    assert all(item.argument_digest and item.result_digest for item in result.tool_executions)


def test_incomplete_provider_and_effectful_tool_are_partial_and_unknown(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _provider(recorder, "call-incomplete", 1, outcome=None)
    _tool(
        recorder,
        "external-incomplete",
        requested_name="remote_write",
        resolved_name="remote_write",
        model_call_id="model-write",
        effect="external",
        outcome=None,
    )
    _terminal(recorder)

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert result.evidence_status is evidence.EvidenceCompleteness.PARTIAL
    assert result.provider_calls[0].attempts[0].outcome is None
    assert result.provider_calls[0].attempts[0].completion_status == "unknown"
    assert result.tool_executions[0].outcome is None
    assert result.tool_executions[0].completion_status == "unknown"
    assert "incomplete_provider_attempt" in result.warnings
    assert "incomplete_tool_execution" in result.warnings
    assert "unsafe_tool_completion_unknown:external-incomplete" in result.warnings


def test_duplicate_receipt_and_conflicting_terminal_are_corrupt(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _tool(
        recorder,
        "duplicate",
        requested_name="read",
        resolved_name="read",
        model_call_id="model-read",
    )
    recorder.emit(
        evidence.TOOL_EXECUTION_COMPLETED,
        correlations={"receipt_id": "duplicate"},
        metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "outcome": "failure"},
    )
    _terminal(recorder, "completed")
    _terminal(recorder, "error")

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert result.evidence_status is evidence.EvidenceCompleteness.CORRUPT
    assert result.runtime_outcome is None
    assert "duplicate_tool_execution_marker" in result.inconsistencies
    assert "conflicting_terminal_outcomes" in result.inconsistencies


def _raw_event(
    turn_id: str,
    sequence: int,
    event_type: str,
    *,
    timestamp: str,
    metadata: dict | None = None,
) -> dict:
    return evidence.TurnEvidenceEvent(
        turn_id=turn_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=timestamp,
        conversation_id="test:chat",
        trace_id="trace-raw",
        span_id="span-raw",
        metadata=metadata or {},
    ).to_record()


def test_sequence_orders_replay_not_append_order_or_timestamp(tmp_path: Path) -> None:
    turn_id = "turn-order"
    store = TraceStore(tmp_path)
    assert store.append_event(
        _raw_event(
            turn_id,
            3,
            evidence.TURN_TERMINAL,
            timestamp="same",
            metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
        )
    )
    assert store.append_event(_raw_event(turn_id, 1, evidence.TURN_STARTED, timestamp="same"))
    assert store.append_event(_raw_event(turn_id, 2, evidence.AGENT_ENTERED, timestamp="same"))

    result = replay.replay_turn(tmp_path, turn_id)

    assert [item.sequence for item in result.ordered_timeline] == [1, 2, 3]
    assert result.evidence_status is evidence.EvidenceCompleteness.COMPLETE


def test_sequence_gap_is_partial(tmp_path: Path) -> None:
    turn_id = "turn-gap"
    store = TraceStore(tmp_path)
    assert store.append_event(_raw_event(turn_id, 1, evidence.TURN_STARTED, timestamp="later"))
    assert store.append_event(
        _raw_event(
            turn_id,
            3,
            evidence.TURN_TERMINAL,
            timestamp="earlier",
            metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
        )
    )

    result = replay.replay_turn(tmp_path, turn_id)

    assert result.evidence_status is evidence.EvidenceCompleteness.PARTIAL
    assert "missing_sequence" in result.warnings


def test_session_and_checkpoint_references_join_read_only(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.emit(
        evidence.SESSION_BOUNDARY,
        correlations={"session_id": "test:chat", "boundary_id": "boundary-1"},
        metadata={"boundary_version": 1, "message_count": 4, "history_digest": "history"},
    )
    recorder.emit(
        evidence.CHECKPOINT_REFERENCE,
        correlations={
            "session_id": "test:chat",
            "boundary_id": "boundary-1",
            "checkpoint_record_id": "record-1",
            "checkpoint_id": "tree-1",
        },
        metadata={
            "checkpoint_status": "created",
            "checkpoint_revision": 2,
            "checkpoint_schema_version": 1,
        },
    )
    _terminal(recorder)
    calls: list[tuple] = []

    def boundary_lookup(session_id: str, boundary_id: str):
        calls.append(("session", session_id, boundary_id))
        return SimpleNamespace(
            boundary_id="boundary-1",
            version=1,
            message_count=4,
            history_digest="history",
            turn_id=recorder.turn_id,
        )

    def checkpoint_lookup(record_id: str):
        calls.append(("checkpoint", record_id))
        return SimpleNamespace(
            record_id="record-1",
            checkpoint_id="tree-1",
            revision=2,
            status="created",
            schema_version=1,
            session_id="test:chat",
            boundary_id="boundary-1",
            turn_id=recorder.turn_id,
        )

    result = replay.replay_turn(
        tmp_path,
        recorder.turn_id,
        session_boundary_lookup=boundary_lookup,
        checkpoint_record_lookup=checkpoint_lookup,
    )

    assert calls == [("session", "test:chat", "boundary-1"), ("checkpoint", "record-1")]
    assert result.session_boundary_ref is not None and result.session_boundary_ref.resolved is True
    assert result.checkpoint_refs[0].resolved is True
    assert result.session_id == "test:chat"
    assert not result.warnings
    assert not result.inconsistencies


def test_unresolved_references_are_preserved_with_warning(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.emit(
        evidence.SESSION_BOUNDARY,
        correlations={"session_id": "test:chat", "boundary_id": "missing-boundary"},
        metadata={"boundary_version": 1, "message_count": 2, "history_digest": "digest"},
    )
    recorder.emit(
        evidence.CHECKPOINT_REFERENCE,
        correlations={
            "session_id": "test:chat",
            "boundary_id": "missing-boundary",
            "checkpoint_record_id": "missing-record",
            "checkpoint_id": "missing-tree",
        },
        metadata={
            "checkpoint_status": "created",
            "checkpoint_revision": 1,
            "checkpoint_schema_version": 1,
        },
    )
    _terminal(recorder)

    result = replay.replay_turn(
        tmp_path,
        recorder.turn_id,
        session_boundary_lookup=lambda _session, _boundary: None,
        checkpoint_record_lookup=lambda _record: None,
    )

    assert result.session_boundary_ref is not None
    assert result.session_boundary_ref.boundary_id == "missing-boundary"
    assert result.session_boundary_ref.resolved is False
    assert result.checkpoint_refs[0].record_id == "missing-record"
    assert result.checkpoint_refs[0].resolved is False
    assert set(result.warnings) == {
        "unresolved_session_boundary_reference",
        "unresolved_checkpoint_reference",
    }


def test_invalid_durable_reference_correlation_is_corrupt(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.emit(
        evidence.SESSION_BOUNDARY,
        correlations={"session_id": "test:chat", "boundary_id": "boundary-1"},
        metadata={"boundary_version": 1, "message_count": 4, "history_digest": "expected"},
    )
    _terminal(recorder)

    result = replay.replay_turn(
        tmp_path,
        recorder.turn_id,
        session_boundary_lookup=lambda _session, _boundary: SimpleNamespace(
            boundary_id="boundary-1",
            version=1,
            message_count=4,
            history_digest="different",
            turn_id=recorder.turn_id,
        ),
    )

    assert result.evidence_status is evidence.EvidenceCompleteness.CORRUPT
    assert "invalid_session_boundary_correlation" in result.inconsistencies


def test_replay_digest_and_provenance_are_stable_across_restart(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _provider(recorder, "call", 1, usage={"prompt_tokens": 1})
    _terminal(recorder)

    first = replay.replay_turn(tmp_path, recorder.turn_id)
    second = replay.replay_turn(Path(str(tmp_path)), recorder.turn_id)

    assert first.replay_digest == second.replay_digest
    assert first.source_evidence_references == second.source_evidence_references
    assert [item.evidence_id for item in first.source_evidence_references] == [
        item.evidence_id for item in first.ordered_timeline
    ]
    with pytest.raises(FrozenInstanceError):
        first.ordered_timeline[0].sequence = 99  # type: ignore[misc]


def test_timestamps_do_not_change_structural_digest(tmp_path: Path) -> None:
    def write(root: Path, timestamp: str) -> None:
        store = TraceStore(root)
        assert store.append_event(_raw_event("turn-time", 1, evidence.TURN_STARTED, timestamp=timestamp))
        assert store.append_event(
            _raw_event(
                "turn-time",
                2,
                evidence.TURN_TERMINAL,
                timestamp=timestamp,
                metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
            )
        )

    write(tmp_path / "one", "2020-01-01")
    write(tmp_path / "two", "2030-01-01")

    assert replay.replay_turn(tmp_path / "one", "turn-time").replay_digest == replay.replay_turn(
        tmp_path / "two", "turn-time"
    ).replay_digest


def test_legacy_untracked_tool_execution_is_explicit_partial(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _terminal(recorder, tool_calls=1, tool_failures=0)

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert result.evidence_status is evidence.EvidenceCompleteness.PARTIAL
    assert result.tool_executions == ()
    assert "legacy_or_untracked_tool_execution" in result.warnings


def test_conflicting_trace_identity_is_not_silently_selected(tmp_path: Path) -> None:
    turn_id = "turn-conflict"
    first = _raw_event(turn_id, 1, evidence.TURN_STARTED, timestamp="same")
    second = _raw_event(
        turn_id,
        2,
        evidence.TURN_TERMINAL,
        timestamp="same",
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )
    second["trace_id"] = "different-trace"
    store = TraceStore(tmp_path)
    assert store.append_event(first)
    assert store.append_event(second)

    result = replay.replay_turn(tmp_path, turn_id)

    assert result.evidence_status is evidence.EvidenceCompleteness.CORRUPT
    assert result.trace_id is None
    assert "conflicting_trace_id" in result.inconsistencies


def test_replay_is_projection_only_and_does_not_require_live_services(tmp_path: Path, monkeypatch) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    _terminal(recorder)
    reads = 0
    original = evidence.read_turn_evidence

    def counted_read(state_dir, turn_id):
        nonlocal reads
        reads += 1
        return original(state_dir, turn_id)

    monkeypatch.setattr(evidence, "read_turn_evidence", counted_read)

    result = replay.replay_turn(tmp_path, recorder.turn_id)

    assert reads == 1
    assert result.runtime_outcome is not None
    assert result.runtime_outcome.outcome == "completed"
    assert not hasattr(replay, "Provider")
    assert not hasattr(replay, "ToolRegistry")
    assert not hasattr(replay, "DeliveryHub")
    assert not hasattr(replay, "WorkspaceRestoreService")
    assert replace(result, replay_digest=result.replay_digest) == result
