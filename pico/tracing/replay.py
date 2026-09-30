"""Deterministic, read-only projection of one Runtime Turn's evidence.

Replay in this module means reconstruction only.  It intentionally depends on
the durable evidence reader and optional read-only reference lookups, never on
Provider, Tool, Delivery, Session mutation, or Workspace recovery interfaces.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from . import evidence

SCHEMA = "pico.trace-replay.v1"
SCHEMA_VERSION = 1

SessionBoundaryLookup = Callable[[str, str], object | None]
CheckpointRecordLookup = Callable[[str], object | None]


@dataclass(frozen=True)
class SourceEvidenceReference:
    sequence: int
    event_type: str
    evidence_id: str
    structural_digest: str


@dataclass(frozen=True)
class ReplayTimelineEntry:
    sequence: int
    event_type: str
    evidence_id: str
    trace_id: str | None
    span_id: str | None


@dataclass(frozen=True)
class ProviderUsageSummary:
    totals: tuple[tuple[str, int | float], ...]
    complete: bool
    attempts_with_usage: int


@dataclass(frozen=True)
class ProviderLogicalCall:
    logical_call_id: str
    attempts: tuple[evidence.ProviderAttemptEvidence, ...]
    final_known_outcome: str | None
    retry_count: int
    fallback_count: int
    usage: ProviderUsageSummary
    first_sequence: int
    last_sequence: int


@dataclass(frozen=True)
class RuntimeOutcome:
    sequence: int
    outcome: str
    lifecycle_event: str | None
    error_class: str | None
    provider_error_category: str | None
    tool_calls: int | None
    tool_failures: int | None


@dataclass(frozen=True)
class DeliveryOutcome:
    sequence: int
    channel: str | None
    event: str | None
    outcome: str | None
    attempts: int | None
    error_class: str | None
    delivery_trace_id: str | None


@dataclass(frozen=True)
class SessionBoundaryReference:
    session_id: str | None
    boundary_id: str | None
    turn_id: str
    version: int | None
    message_count: int | None
    history_digest: str | None
    resolved: bool | None


@dataclass(frozen=True)
class CheckpointReference:
    record_id: str | None
    checkpoint_id: str | None
    revision: int | None
    status: str | None
    schema_version: int | None
    session_id: str | None
    boundary_id: str | None
    turn_id: str
    resolved: bool | None


@dataclass(frozen=True)
class TraceReplayResult:
    schema: str
    schema_version: int
    turn_id: str
    trace_id: str | None
    conversation_id: str | None
    session_id: str | None
    evidence_status: evidence.EvidenceCompleteness
    ordered_timeline: tuple[ReplayTimelineEntry, ...]
    provider_calls: tuple[ProviderLogicalCall, ...]
    tool_executions: tuple[evidence.ToolExecutionEvidence, ...]
    runtime_outcome: RuntimeOutcome | None
    delivery_outcomes: tuple[DeliveryOutcome, ...]
    session_boundary_ref: SessionBoundaryReference | None
    checkpoint_refs: tuple[CheckpointReference, ...]
    warnings: tuple[str, ...]
    inconsistencies: tuple[str, ...]
    source_evidence_references: tuple[SourceEvidenceReference, ...]
    replay_digest: str


def _value(obj: object, name: str) -> Any:
    return getattr(obj, name, None)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _integer(value: Any, *, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _event_structure(event: evidence.TurnEvidenceEvent) -> dict[str, Any]:
    return {
        "schema": event.schema,
        "schema_version": event.schema_version,
        "turn_id": event.turn_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "conversation_id": event.conversation_id,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "correlations": _canonical(event.correlations),
        "metadata": _canonical(event.metadata),
        "write_failures_before": event.write_failures_before,
    }


def _source_reference(event: evidence.TurnEvidenceEvent) -> SourceEvidenceReference:
    evidence_id = f"{event.turn_id}:{event.sequence}:{event.event_type}"
    return SourceEvidenceReference(
        sequence=event.sequence,
        event_type=event.event_type,
        evidence_id=evidence_id,
        structural_digest=evidence.canonical_digest(_event_structure(event)),
    )


def _provider_calls(
    attempts: tuple[evidence.ProviderAttemptEvidence, ...],
) -> tuple[ProviderLogicalCall, ...]:
    grouped: dict[str, list[evidence.ProviderAttemptEvidence]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.logical_call_id].append(attempt)

    calls: list[ProviderLogicalCall] = []
    for logical_call_id, values in grouped.items():
        ordered = tuple(
            sorted(
                values,
                key=lambda item: (
                    item.attempt_ordinal,
                    item.started_sequence if item.started_sequence is not None else item.completed_sequence or 0,
                ),
            )
        )
        sequences = [
            sequence
            for item in ordered
            for sequence in (item.started_sequence, item.completed_sequence)
            if sequence is not None
        ]
        totals: dict[str, int | float] = defaultdict(int)
        attempts_with_usage = 0
        for item in ordered:
            if item.usage_available is True:
                attempts_with_usage += 1
            for key, value in item.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[str(key)] += value
        complete_usage = bool(ordered) and all(
            item.complete and item.usage_available is True for item in ordered
        )
        attempted_models = [item.attempted_model for item in ordered]
        fallback_count = sum(
            previous is not None and current is not None and previous != current
            for previous, current in zip(attempted_models, attempted_models[1:], strict=False)
        )
        completed = [item for item in ordered if item.completed_sequence is not None]
        calls.append(
            ProviderLogicalCall(
                logical_call_id=logical_call_id,
                attempts=ordered,
                final_known_outcome=completed[-1].outcome if completed else None,
                retry_count=max(0, len(ordered) - 1),
                fallback_count=fallback_count,
                usage=ProviderUsageSummary(
                    totals=tuple(sorted(totals.items())),
                    complete=complete_usage,
                    attempts_with_usage=attempts_with_usage,
                ),
                first_sequence=min(sequences),
                last_sequence=max(sequences),
            )
        )
    return tuple(sorted(calls, key=lambda item: (item.first_sequence, item.logical_call_id)))


def _runtime_outcome(events: tuple[evidence.TurnEvidenceEvent, ...]) -> RuntimeOutcome | None:
    terminals = [event for event in events if event.event_type == evidence.TURN_TERMINAL]
    if len(terminals) != 1:
        return None
    event = terminals[0]
    outcome = event.metadata.get("outcome")
    if not isinstance(outcome, str):
        return None
    return RuntimeOutcome(
        sequence=event.sequence,
        outcome=outcome,
        lifecycle_event=event.metadata.get("lifecycle_event"),
        error_class=event.metadata.get("error_class"),
        provider_error_category=event.metadata.get("provider_error_category"),
        tool_calls=event.metadata.get("tool_calls"),
        tool_failures=event.metadata.get("tool_failures"),
    )


def _delivery_outcomes(events: tuple[evidence.TurnEvidenceEvent, ...]) -> tuple[DeliveryOutcome, ...]:
    return tuple(
        DeliveryOutcome(
            sequence=event.sequence,
            channel=event.metadata.get("channel"),
            event=event.metadata.get("event"),
            outcome=event.metadata.get("outcome"),
            attempts=event.metadata.get("attempts"),
            error_class=event.metadata.get("error_class"),
            delivery_trace_id=event.correlations.get("delivery_trace_id"),
        )
        for event in events
        if event.event_type == evidence.DELIVERY_OUTCOME
    )


def _session_reference(
    event: evidence.TurnEvidenceEvent,
    lookup: SessionBoundaryLookup | None,
    warnings: list[str],
    inconsistencies: list[str],
) -> SessionBoundaryReference:
    session_id = event.correlations.get("session_id")
    boundary_id = event.correlations.get("boundary_id")
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(boundary_id, str)
        or not boundary_id
        or not _integer(event.metadata.get("boundary_version"), minimum=1)
        or not _integer(event.metadata.get("message_count"), minimum=0)
        or not isinstance(event.metadata.get("history_digest"), str)
        or not event.metadata.get("history_digest")
    ):
        inconsistencies.append("invalid_session_boundary_reference")
    resolved: bool | None = None
    boundary = None
    if lookup is not None and isinstance(session_id, str) and isinstance(boundary_id, str):
        try:
            boundary = lookup(session_id, boundary_id)
        except Exception as exc:  # noqa: BLE001 -- lookup failure is replay evidence
            warnings.append(f"session_boundary_lookup_failed:{type(exc).__name__}")
        else:
            resolved = boundary is not None
            if boundary is None:
                warnings.append("unresolved_session_boundary_reference")

    reference = SessionBoundaryReference(
        session_id=session_id if isinstance(session_id, str) else None,
        boundary_id=boundary_id if isinstance(boundary_id, str) else None,
        turn_id=event.turn_id,
        version=_value(boundary, "version") if boundary is not None else event.metadata.get("boundary_version"),
        message_count=(
            _value(boundary, "message_count") if boundary is not None else event.metadata.get("message_count")
        ),
        history_digest=(
            _value(boundary, "history_digest") if boundary is not None else event.metadata.get("history_digest")
        ),
        resolved=resolved,
    )
    if boundary is not None:
        expected = (
            event.metadata.get("boundary_version"),
            event.metadata.get("message_count"),
            event.metadata.get("history_digest"),
            event.turn_id,
        )
        actual = (
            _value(boundary, "version"),
            _value(boundary, "message_count"),
            _value(boundary, "history_digest"),
            _value(boundary, "turn_id"),
        )
        if expected != actual or _value(boundary, "boundary_id") != boundary_id:
            inconsistencies.append("invalid_session_boundary_correlation")
    return reference


def _checkpoint_reference(
    event: evidence.TurnEvidenceEvent,
    lookup: CheckpointRecordLookup | None,
    warnings: list[str],
    inconsistencies: list[str],
) -> CheckpointReference:
    record_id = event.correlations.get("checkpoint_record_id")
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(event.correlations.get("session_id"), str)
        or not isinstance(event.correlations.get("boundary_id"), str)
        or not _integer(event.metadata.get("checkpoint_revision"), minimum=1)
        or not _integer(event.metadata.get("checkpoint_schema_version"), minimum=1)
        or not isinstance(event.metadata.get("checkpoint_status"), str)
    ):
        inconsistencies.append("invalid_checkpoint_reference")
    record = None
    resolved: bool | None = None
    if lookup is not None and isinstance(record_id, str):
        try:
            record = lookup(record_id)
        except Exception as exc:  # noqa: BLE001 -- lookup failure is replay evidence
            warnings.append(f"checkpoint_lookup_failed:{type(exc).__name__}")
        else:
            resolved = record is not None
            if record is None:
                warnings.append("unresolved_checkpoint_reference")

    status = _enum_value(_value(record, "status")) if record is not None else event.metadata.get("checkpoint_status")
    reference = CheckpointReference(
        record_id=record_id if isinstance(record_id, str) else None,
        checkpoint_id=(
            _value(record, "checkpoint_id") if record is not None else event.correlations.get("checkpoint_id")
        ),
        revision=_value(record, "revision") if record is not None else event.metadata.get("checkpoint_revision"),
        status=status,
        schema_version=(
            _value(record, "schema_version")
            if record is not None
            else event.metadata.get("checkpoint_schema_version")
        ),
        session_id=(
            _value(record, "session_id") if record is not None else event.correlations.get("session_id")
        ),
        boundary_id=(
            _value(record, "boundary_id") if record is not None else event.correlations.get("boundary_id")
        ),
        turn_id=event.turn_id,
        resolved=resolved,
    )
    if record is not None:
        expected = (
            record_id,
            event.correlations.get("checkpoint_id"),
            event.metadata.get("checkpoint_revision"),
            event.metadata.get("checkpoint_status"),
            event.metadata.get("checkpoint_schema_version"),
            event.correlations.get("session_id"),
            event.correlations.get("boundary_id"),
            event.turn_id,
        )
        actual = (
            _value(record, "record_id"),
            _value(record, "checkpoint_id"),
            _value(record, "revision"),
            _enum_value(_value(record, "status")),
            _value(record, "schema_version"),
            _value(record, "session_id"),
            _value(record, "boundary_id"),
            _value(record, "turn_id"),
        )
        if expected != actual:
            inconsistencies.append("invalid_checkpoint_correlation")
    return reference


def _status(
    original: evidence.EvidenceCompleteness,
    *,
    derived_partial: bool,
    derived_corrupt: bool,
) -> evidence.EvidenceCompleteness:
    if original is evidence.EvidenceCompleteness.CORRUPT or derived_corrupt:
        return evidence.EvidenceCompleteness.CORRUPT
    if original is evidence.EvidenceCompleteness.PARTIAL or derived_partial:
        return evidence.EvidenceCompleteness.PARTIAL
    return evidence.EvidenceCompleteness.COMPLETE


def replay_turn(
    state_dir: str | Path,
    turn_id: str,
    *,
    session_boundary_lookup: SessionBoundaryLookup | None = None,
    checkpoint_record_lookup: CheckpointRecordLookup | None = None,
) -> TraceReplayResult:
    """Reconstruct one Turn without executing or mutating any Runtime domain."""

    source = evidence.read_turn_evidence(state_dir, turn_id)
    # Sequence is the only semantic clock.  Canonical structural tie-breaking
    # keeps an already-corrupt duplicate sequence deterministic without
    # pretending that either duplicate happened first.
    events = tuple(
        sorted(
            source.events,
            key=lambda event: (
                event.sequence,
                event.event_type,
                evidence.canonical_digest(_event_structure(event)),
            ),
        )
    )
    projection_source = evidence.TurnEvidenceReadResult(
        turn_id=source.turn_id,
        completeness=source.completeness,
        events=events,
        findings=source.findings,
    )
    warnings = list(source.findings if source.completeness is evidence.EvidenceCompleteness.PARTIAL else ())
    inconsistencies = list(
        source.findings if source.completeness is evidence.EvidenceCompleteness.CORRUPT else ()
    )

    references = tuple(_source_reference(event) for event in events)
    timeline = tuple(
        ReplayTimelineEntry(
            sequence=event.sequence,
            event_type=event.event_type,
            evidence_id=reference.evidence_id,
            trace_id=event.trace_id,
            span_id=event.span_id,
        )
        for event, reference in zip(events, references, strict=True)
    )
    trace_ids = {event.trace_id for event in events if event.trace_id}
    conversation_ids = {event.conversation_id for event in events if event.conversation_id}
    trace_id = next(iter(trace_ids)) if len(trace_ids) == 1 else None
    conversation_id = next(iter(conversation_ids)) if len(conversation_ids) == 1 else None

    boundary_events = tuple(event for event in events if event.event_type == evidence.SESSION_BOUNDARY)
    session_ref = None
    if len(boundary_events) == 1:
        session_ref = _session_reference(
            boundary_events[0], session_boundary_lookup, warnings, inconsistencies
        )
    elif len(boundary_events) > 1:
        inconsistencies.append("conflicting_session_boundary_reference")

    checkpoint_refs = tuple(
        _checkpoint_reference(event, checkpoint_record_lookup, warnings, inconsistencies)
        for event in events
        if event.event_type == evidence.CHECKPOINT_REFERENCE
    )
    session_ids = {
        item
        for item in [session_ref.session_id if session_ref is not None else None]
        + [reference.session_id for reference in checkpoint_refs]
        if item
    }
    session_id = next(iter(session_ids)) if len(session_ids) == 1 else None
    if len(session_ids) > 1:
        inconsistencies.append("conflicting_session_id")

    provider_calls = _provider_calls(projection_source.provider_attempts)
    tool_executions = projection_source.tool_executions
    runtime_outcome = _runtime_outcome(events)
    if sum(event.event_type == evidence.TURN_TERMINAL for event in events) > 1:
        inconsistencies.append("conflicting_terminal_outcomes")

    derived_partial = False
    if runtime_outcome is not None and isinstance(runtime_outcome.tool_calls, int):
        tracked_top_level = sum(item.parent_call_id is None for item in tool_executions)
        if runtime_outcome.tool_calls > tracked_top_level:
            warnings.append("legacy_or_untracked_tool_execution")
            derived_partial = True
    for item in tool_executions:
        if not item.complete and item.effect in {"write", "execute", "network", "external"}:
            warnings.append(f"unsafe_tool_completion_unknown:{item.receipt_id}")

    warnings = list(dict.fromkeys(warnings))
    inconsistencies = list(dict.fromkeys(inconsistencies))
    status = _status(
        source.completeness,
        derived_partial=derived_partial,
        derived_corrupt=bool(
            {
                "invalid_session_boundary_correlation",
                "invalid_session_boundary_reference",
                "invalid_checkpoint_correlation",
                "invalid_checkpoint_reference",
                "conflicting_session_boundary_reference",
                "conflicting_session_id",
                "conflicting_terminal_outcomes",
            }
            & set(inconsistencies)
        ),
    )
    deliveries = _delivery_outcomes(events)

    structural = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "turn_id": turn_id,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "session_id": session_id,
        "evidence_status": status.value,
        "ordered_timeline": _canonical(timeline),
        "provider_calls": _canonical(provider_calls),
        "tool_executions": _canonical(tool_executions),
        "runtime_outcome": _canonical(runtime_outcome),
        "delivery_outcomes": _canonical(deliveries),
        "session_boundary_ref": _canonical(session_ref),
        "checkpoint_refs": _canonical(checkpoint_refs),
        "warnings": warnings,
        "inconsistencies": inconsistencies,
        "source_evidence_references": _canonical(references),
    }
    return TraceReplayResult(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        turn_id=turn_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
        session_id=session_id,
        evidence_status=status,
        ordered_timeline=timeline,
        provider_calls=provider_calls,
        tool_executions=tool_executions,
        runtime_outcome=runtime_outcome,
        delivery_outcomes=deliveries,
        session_boundary_ref=session_ref,
        checkpoint_refs=checkpoint_refs,
        warnings=tuple(warnings),
        inconsistencies=tuple(inconsistencies),
        source_evidence_references=references,
        replay_digest=evidence.canonical_digest(structural),
    )


__all__ = [
    "CheckpointRecordLookup",
    "CheckpointReference",
    "DeliveryOutcome",
    "ProviderLogicalCall",
    "ProviderUsageSummary",
    "ReplayTimelineEntry",
    "RuntimeOutcome",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SessionBoundaryLookup",
    "SessionBoundaryReference",
    "SourceEvidenceReference",
    "TraceReplayResult",
    "replay_turn",
]
