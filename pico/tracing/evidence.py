"""Durable, structural evidence for one Runtime Turn.

The envelope is deliberately smaller than the span model.  It records ordering,
identity and durable references, but never Provider prompts, Tool arguments or
outputs.  Recording is observational: every public write helper is no-throw.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import spans

SCHEMA = "pico.turn.evidence.event.v1"
SCHEMA_VERSION = 1

TURN_STARTED = "turn.started"
AGENT_ENTERED = "agent.entered"
SESSION_BOUNDARY = "session.boundary"
CHECKPOINT_REFERENCE = "checkpoint.reference"
TURN_TERMINAL = "turn.terminal"
DELIVERY_OUTCOME = "delivery.outcome"
WRITE_DEGRADED = "evidence.write_degraded"
PROVIDER_ATTEMPT_STARTED = "provider.attempt.started"
PROVIDER_ATTEMPT_COMPLETED = "provider.attempt.completed"
TOOL_EXECUTION_STARTED = "tool.execution.started"
TOOL_EXECUTION_COMPLETED = "tool.execution.completed"
DECISION_RECEIPT = "decision.completed"
KNOWLEDGE_USAGE = "knowledge.usage"
AGENT_ITERATION_BUDGET_EXHAUSTED = "agent.iteration_budget_exhausted"
PROVIDER_RECEIPT_SCHEMA = "pico.provider-attempt.v1"
TOOL_RECEIPT_SCHEMA = "pico.resolved-tool-execution.v1"

PROVIDER_TRANSPORT = "provider_transport"
PROVIDER_SERVER = "provider_server"
PROVIDER_MALFORMED_RESPONSE = "provider_malformed_response"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
PROVIDER_AUTH = "provider_auth"
PROVIDER_MODEL_ERROR = "provider_model_error"
PROVIDER_UNKNOWN = "provider_unknown"

_TERMINAL_OUTCOMES = {"completed", "completed_with_tool_failure", "provider_failed", "error", "cancelled"}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _freeze_value(values or {})


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest without persisting the payload."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: f"<{type(item).__module__}.{type(item).__qualname__}>",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_provider_failure(
    category: str | None,
    *,
    exception_type: str | None = None,
    message: str | None = None,
) -> str:
    """Return a bounded privacy-safe Provider failure category.

    Message text is inspected in memory only and is never returned or persisted.
    Empty/non-JSON normalization requires the specific parse-failure signature;
    a generic error remains unknown.
    """

    lowered = (message or "").casefold()
    if (
        "unable to get json response" in lowered
        and "expecting value" in lowered
        and "original response:" in lowered
    ):
        return PROVIDER_MALFORMED_RESPONSE
    value = (category or "").casefold()
    exc = (exception_type or "").casefold()
    if value in {"network", "transport", "connection", "timeout"} or any(
        item in exc for item in ("connection", "timeout", "transport")
    ):
        return PROVIDER_TRANSPORT
    if value in {"server", "service_unavailable"}:
        return PROVIDER_SERVER
    if value in {"rate_limit", "ratelimit"}:
        return PROVIDER_RATE_LIMIT
    if value in {"auth", "authentication", "permission"}:
        return PROVIDER_AUTH
    if value in {"model", "invalid_request", "content_policy"}:
        return PROVIDER_MODEL_ERROR
    return PROVIDER_UNKNOWN


@dataclass(frozen=True)
class TurnEvidenceEvent:
    """One immutable, versioned structural fact for a Runtime Turn."""

    turn_id: str
    sequence: int
    event_type: str
    timestamp: str
    conversation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    correlations: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    write_failures_before: int = 0
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlations", _freeze(self.correlations))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
        }
        if self.conversation_id is not None:
            record["conversation_id"] = self.conversation_id
        if self.trace_id is not None:
            record["trace_id"] = self.trace_id
        if self.span_id is not None:
            record["span_id"] = self.span_id
        if self.correlations:
            record["correlations"] = _thaw(self.correlations)
        if self.metadata:
            record["metadata"] = _thaw(self.metadata)
        if self.write_failures_before:
            record["write_failures_before"] = self.write_failures_before
        return record


class EvidenceCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class TurnEvidenceReadResult:
    turn_id: str
    completeness: EvidenceCompleteness
    events: tuple[TurnEvidenceEvent, ...]
    findings: tuple[str, ...] = ()

    @property
    def provider_attempts(self) -> tuple["ProviderAttemptEvidence", ...]:
        return _provider_attempts(self.events)

    @property
    def tool_executions(self) -> tuple["ToolExecutionEvidence", ...]:
        return _tool_executions(self.events)


@dataclass(frozen=True)
class ProviderAttemptEvidence:
    logical_call_id: str
    attempt_id: str
    attempt_ordinal: int
    started_sequence: int | None
    completed_sequence: int | None
    requested_model: str | None
    attempted_model: str | None
    actual_model: str | None
    provider: str | None
    outcome: str | None
    finish_reason: str | None
    error_category: str | None
    request_digest: str | None
    response_digest: str | None
    usage_available: bool | None
    usage: Mapping[str, Any]
    duration_ms: int | None
    trace_id: str | None
    span_id: str | None
    normalized_error_category: str | None = None
    call_role: str = "main_agent"

    @property
    def complete(self) -> bool:
        return self.started_sequence is not None and self.completed_sequence is not None

    @property
    def completion_status(self) -> str:
        """Explicit readback state; absence of completion is never failure or success."""

        return "complete" if self.complete else "unknown"


@dataclass(frozen=True)
class ToolExecutionEvidence:
    receipt_id: str
    model_call_id: str | None
    parent_call_id: str | None
    started_sequence: int | None
    completed_sequence: int | None
    requested_name: str
    resolved_name: str | None
    routed_via: str | None
    effect: str | None
    outcome: str | None
    failure_stage: str | None
    failure_category: str | None
    argument_digest: str | None
    result_digest: str | None
    result_size: int | None
    duration_ms: int | None
    trace_id: str | None
    span_id: str | None
    repository_read_path: str | None = None

    @property
    def complete(self) -> bool:
        return self.started_sequence is not None and self.completed_sequence is not None

    @property
    def completion_status(self) -> str:
        """Explicit readback state, including uncertain effectful execution."""

        return "complete" if self.complete else "unknown"


EvidenceWriter = Callable[[dict[str, Any]], bool]


class TurnEvidenceRecorder:
    """Turn-local sequence allocator and best-effort durable writer.

    The lock protects only sequence/failure counters.  Store I/O happens after
    releasing it, so evidence numbering cannot serialize Provider or Tool work.
    """

    def __init__(
        self,
        *,
        turn_id: str,
        conversation_id: str | None,
        trace_id: str | None,
        root_span_id: str | None,
        writer: EvidenceWriter | None = None,
    ) -> None:
        self.turn_id = turn_id
        self.conversation_id = conversation_id
        self.trace_id = trace_id
        self.root_span_id = root_span_id
        self._writer = writer or spans.emit_event
        self._lock = threading.Lock()
        self._sequence = 0
        self._write_failures = 0
        self._identity_counters: dict[str, int] = {}

    @property
    def write_failures(self) -> int:
        with self._lock:
            return self._write_failures

    def next_identity(self, kind: str) -> str:
        """Allocate a deterministic Turn-local correlation identity."""

        with self._lock:
            ordinal = self._identity_counters.get(kind, 0) + 1
            self._identity_counters[kind] = ordinal
        return f"{self.turn_id}:{kind}:{ordinal}"

    def emit(
        self,
        event_type: str,
        *,
        span_id: str | None = None,
        correlations: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnEvidenceEvent:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            failures = self._write_failures
        event = TurnEvidenceEvent(
            turn_id=self.turn_id,
            sequence=sequence,
            event_type=event_type,
            timestamp=spans.now_iso(),
            conversation_id=self.conversation_id,
            trace_id=self.trace_id,
            span_id=span_id or self.root_span_id,
            correlations=correlations or {},
            metadata=metadata or {},
            write_failures_before=failures,
        )
        try:
            written = self._writer(event.to_record())
        except Exception:  # noqa: BLE001 -- evidence cannot affect Runtime
            written = False
        if not written:
            with self._lock:
                self._write_failures += 1
                self._sequence += 1
                marker_sequence = self._sequence
                failures = self._write_failures
            marker = TurnEvidenceEvent(
                turn_id=self.turn_id,
                sequence=marker_sequence,
                event_type=WRITE_DEGRADED,
                timestamp=spans.now_iso(),
                conversation_id=self.conversation_id,
                trace_id=self.trace_id,
                span_id=self.root_span_id,
                correlations={"failed_sequence": sequence},
                metadata={"failed_event_type": event_type},
                write_failures_before=failures,
            )
            try:
                marker_written = self._writer(marker.to_record())
            except Exception:  # noqa: BLE001 -- evidence cannot affect Runtime
                marker_written = False
            if not marker_written:
                with self._lock:
                    self._write_failures += 1
        return event


class ProviderCallEvidenceContext:
    """Ephemeral identity shared by retry and model-fallback attempts."""

    def __init__(
        self,
        recorder: TurnEvidenceRecorder,
        requested_model: str | None,
        call_role: str = "main_agent",
    ) -> None:
        self.recorder = recorder
        self.logical_call_id = recorder.next_identity("provider-call")
        self.requested_model = requested_model
        self.call_role = call_role
        self._lock = threading.Lock()
        self._attempt_ordinal = 0

    def next_attempt(self) -> tuple[str, int]:
        with self._lock:
            self._attempt_ordinal += 1
            ordinal = self._attempt_ordinal
        return f"{self.logical_call_id}:attempt:{ordinal}", ordinal


_CURRENT: contextvars.ContextVar[TurnEvidenceRecorder | None] = contextvars.ContextVar(
    "pico_turn_evidence_recorder", default=None
)
_PROVIDER_CALL: contextvars.ContextVar[ProviderCallEvidenceContext | None] = contextvars.ContextVar(
    "pico_provider_call_evidence", default=None
)


def current() -> TurnEvidenceRecorder | None:
    return _CURRENT.get()


@contextlib.contextmanager
def turn_scope(recorder: TurnEvidenceRecorder) -> Iterator[TurnEvidenceRecorder]:
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def provider_call_scope(
    requested_model: str | None,
    call_role: str = "main_agent",
) -> Iterator[ProviderCallEvidenceContext | None]:
    existing = _PROVIDER_CALL.get()
    if existing is not None:
        yield existing
        return
    recorder = current()
    if recorder is None:
        yield None
        return
    value = ProviderCallEvidenceContext(recorder, requested_model, call_role)
    token = _PROVIDER_CALL.set(value)
    try:
        yield value
    finally:
        _PROVIDER_CALL.reset(token)


def emit_current(
    event_type: str,
    *,
    span_id: str | None = None,
    correlations: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TurnEvidenceEvent | None:
    recorder = current()
    if recorder is None:
        return None
    return recorder.emit(event_type, span_id=span_id, correlations=correlations, metadata=metadata)


def _event_paths(state_dir: Path) -> list[Path]:
    logs = state_dir.expanduser() / "logs"
    archived = sorted((logs / "archive").glob("**/audit-events-*.log"))
    active = logs / "audit-events.log"
    return [*archived, *([active] if active.exists() else [])]


def _decode_event(record: dict[str, Any]) -> TurnEvidenceEvent:
    required = ("turn_id", "sequence", "event_type", "timestamp")
    if record.get("schema") != SCHEMA or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported_schema")
    if any(key not in record for key in required):
        raise ValueError("missing_required_field")
    if not isinstance(record["turn_id"], str) or not record["turn_id"]:
        raise ValueError("invalid_turn_id")
    if not isinstance(record["sequence"], int) or isinstance(record["sequence"], bool) or record["sequence"] < 1:
        raise ValueError("invalid_sequence")
    if not isinstance(record["event_type"], str) or not isinstance(record["timestamp"], str):
        raise ValueError("invalid_event_field")
    correlations = record.get("correlations", {})
    metadata = record.get("metadata", {})
    if not isinstance(correlations, dict) or not isinstance(metadata, dict):
        raise ValueError("invalid_metadata")
    failures = record.get("write_failures_before", 0)
    if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
        raise ValueError("invalid_write_failure_count")
    return TurnEvidenceEvent(
        turn_id=record["turn_id"],
        sequence=record["sequence"],
        event_type=record["event_type"],
        timestamp=record["timestamp"],
        conversation_id=record.get("conversation_id"),
        trace_id=record.get("trace_id"),
        span_id=record.get("span_id"),
        correlations=correlations,
        metadata=metadata,
        write_failures_before=failures,
    )


def _events_by_correlation(
    events: tuple[TurnEvidenceEvent, ...],
    *,
    event_types: tuple[str, str],
    correlation_key: str,
) -> tuple[tuple[str, TurnEvidenceEvent | None, TurnEvidenceEvent | None], ...]:
    paired: dict[str, list[TurnEvidenceEvent | None]] = {}
    for event in events:
        if event.event_type not in event_types:
            continue
        identity = event.correlations.get(correlation_key)
        if not isinstance(identity, str) or not identity:
            continue
        slots = paired.setdefault(identity, [None, None])
        slots[0 if event.event_type == event_types[0] else 1] = event
    return tuple(
        (identity, slots[0], slots[1])
        for identity, slots in sorted(
            paired.items(),
            key=lambda item: min(event.sequence for event in item[1] if event is not None),
        )
    )


def _provider_attempts(events: tuple[TurnEvidenceEvent, ...]) -> tuple[ProviderAttemptEvidence, ...]:
    attempts: list[ProviderAttemptEvidence] = []
    for attempt_id, started, completed in _events_by_correlation(
        events,
        event_types=(PROVIDER_ATTEMPT_STARTED, PROVIDER_ATTEMPT_COMPLETED),
        correlation_key="attempt_id",
    ):
        source = started or completed
        if source is None:
            continue
        start_meta = started.metadata if started is not None else {}
        end_meta = completed.metadata if completed is not None else {}
        logical_call_id = source.correlations.get("logical_call_id")
        ordinal = source.correlations.get("attempt_ordinal")
        attempts.append(
            ProviderAttemptEvidence(
                logical_call_id=logical_call_id if isinstance(logical_call_id, str) else "",
                attempt_id=attempt_id,
                attempt_ordinal=ordinal if isinstance(ordinal, int) else 0,
                started_sequence=started.sequence if started else None,
                completed_sequence=completed.sequence if completed else None,
                requested_model=start_meta.get("requested_model"),
                attempted_model=start_meta.get("attempted_model"),
                actual_model=end_meta.get("actual_model"),
                provider=start_meta.get("provider"),
                outcome=end_meta.get("outcome"),
                finish_reason=end_meta.get("finish_reason"),
                error_category=end_meta.get("error_category"),
                normalized_error_category=end_meta.get("normalized_error_category"),
                call_role=(
                    start_meta.get("call_role")
                    if isinstance(start_meta.get("call_role"), str)
                    else "main_agent"
                ),
                request_digest=start_meta.get("request_digest"),
                response_digest=end_meta.get("response_digest"),
                usage_available=end_meta.get("usage_available"),
                usage=_freeze(end_meta.get("usage") if isinstance(end_meta.get("usage"), Mapping) else {}),
                duration_ms=end_meta.get("duration_ms"),
                trace_id=source.trace_id,
                span_id=source.span_id,
            )
        )
    return tuple(attempts)


def _tool_executions(events: tuple[TurnEvidenceEvent, ...]) -> tuple[ToolExecutionEvidence, ...]:
    executions: list[ToolExecutionEvidence] = []
    for receipt_id, started, completed in _events_by_correlation(
        events,
        event_types=(TOOL_EXECUTION_STARTED, TOOL_EXECUTION_COMPLETED),
        correlation_key="receipt_id",
    ):
        source = started or completed
        if source is None:
            continue
        start_meta = started.metadata if started is not None else {}
        end_meta = completed.metadata if completed is not None else {}
        executions.append(
            ToolExecutionEvidence(
                receipt_id=receipt_id,
                model_call_id=source.correlations.get("model_call_id"),
                parent_call_id=source.correlations.get("parent_call_id"),
                started_sequence=started.sequence if started else None,
                completed_sequence=completed.sequence if completed else None,
                requested_name=start_meta.get("requested_name", ""),
                resolved_name=start_meta.get("resolved_name"),
                routed_via=start_meta.get("routed_via"),
                effect=start_meta.get("effect"),
                outcome=end_meta.get("outcome"),
                failure_stage=end_meta.get("failure_stage"),
                failure_category=end_meta.get("failure_category"),
                argument_digest=start_meta.get("argument_digest"),
                result_digest=end_meta.get("result_digest"),
                result_size=end_meta.get("result_size"),
                duration_ms=end_meta.get("duration_ms"),
                trace_id=source.trace_id,
                span_id=source.span_id,
                repository_read_path=start_meta.get("repository_read_path"),
            )
        )
    return tuple(executions)


def _receipt_integrity(events: list[TurnEvidenceEvent]) -> tuple[bool, bool, list[str]]:
    corrupt = False
    partial = False
    findings: list[str] = []
    specs = (
        (
            (PROVIDER_ATTEMPT_STARTED, PROVIDER_ATTEMPT_COMPLETED),
            "attempt_id",
            "provider_attempt",
            PROVIDER_RECEIPT_SCHEMA,
        ),
        (
            (TOOL_EXECUTION_STARTED, TOOL_EXECUTION_COMPLETED),
            "receipt_id",
            "tool_execution",
            TOOL_RECEIPT_SCHEMA,
        ),
    )
    for event_types, correlation_key, label, receipt_schema in specs:
        counts: dict[str, list[int]] = {}
        for event in events:
            if event.event_type not in event_types:
                continue
            if event.metadata.get("receipt_schema") != receipt_schema:
                corrupt = True
                findings.append(f"unsupported_{label}_schema")
            identity = event.correlations.get(correlation_key)
            if not isinstance(identity, str) or not identity:
                corrupt = True
                findings.append(f"invalid_{label}_identity")
                continue
            slots = counts.setdefault(identity, [0, 0])
            slots[0 if event.event_type == event_types[0] else 1] += 1
        for starts, completions in counts.values():
            if starts > 1 or completions > 1:
                corrupt = True
                findings.append(f"duplicate_{label}_marker")
            if starts == 0 or completions == 0:
                partial = True
                findings.append(f"incomplete_{label}")

    ordinals: dict[str, list[int]] = {}
    for event in events:
        if event.event_type != PROVIDER_ATTEMPT_STARTED:
            continue
        logical_call_id = event.correlations.get("logical_call_id")
        ordinal = event.correlations.get("attempt_ordinal")
        if not isinstance(logical_call_id, str) or not logical_call_id or not isinstance(ordinal, int) or ordinal < 1:
            corrupt = True
            findings.append("invalid_provider_attempt_order")
            continue
        ordinals.setdefault(logical_call_id, []).append(ordinal)
    for values in ordinals.values():
        if len(values) != len(set(values)):
            corrupt = True
            findings.append("duplicate_provider_attempt_ordinal")
        elif sorted(values) != list(range(1, max(values) + 1)):
            partial = True
            findings.append("missing_provider_attempt_ordinal")
    return corrupt, partial, findings


def read_turn_evidence(state_dir: str | Path, turn_id: str) -> TurnEvidenceReadResult:
    """Read and structurally classify one Turn's durable evidence after restart."""

    events: list[TurnEvidenceEvent] = []
    findings: list[str] = []
    corrupt = False
    partial = False
    for path in _event_paths(Path(state_dir)):
        try:
            payload = path.read_bytes()
        except OSError:
            partial = True
            findings.append("event_log_unreadable")
            continue
        lines = payload.splitlines(keepends=True)
        for index, raw in enumerate(lines):
            if not raw.strip():
                continue
            has_newline = raw.endswith((b"\n", b"\r"))
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index == len(lines) - 1 and not has_newline:
                    partial = True
                    findings.append("truncated_final_record")
                else:
                    corrupt = True
                    findings.append("malformed_record")
                continue
            if not isinstance(record, dict):
                continue
            if record.get("schema") != SCHEMA:
                if record.get("turn_id") == turn_id:
                    corrupt = True
                    findings.append("unsupported_schema")
                continue
            if record.get("turn_id") != turn_id:
                continue
            try:
                events.append(_decode_event(record))
            except ValueError as exc:
                corrupt = True
                findings.append(str(exc))

    events.sort(key=lambda event: event.sequence)
    sequences = [event.sequence for event in events]
    if len(sequences) != len(set(sequences)):
        corrupt = True
        findings.append("duplicate_sequence")
    if sequences and sequences != list(range(1, max(sequences) + 1)):
        partial = True
        findings.append("missing_sequence")
    if any(event.write_failures_before for event in events):
        partial = True
        findings.append("write_degradation")

    starts = [event for event in events if event.event_type == TURN_STARTED]
    terminals = [event for event in events if event.event_type == TURN_TERMINAL]
    if len(starts) > 1 or len(terminals) > 1:
        corrupt = True
        findings.append("duplicate_lifecycle_marker")
    if not starts:
        partial = True
        findings.append("missing_start")
    if not terminals:
        partial = True
        findings.append("missing_terminal")
    if starts and terminals and terminals[0].sequence <= starts[0].sequence:
        corrupt = True
        findings.append("terminal_before_start")
    for terminal in terminals:
        if terminal.metadata.get("outcome") not in _TERMINAL_OUTCOMES:
            corrupt = True
            findings.append("invalid_terminal_outcome")

    trace_ids = {event.trace_id for event in events if event.trace_id}
    conversation_ids = {event.conversation_id for event in events if event.conversation_id}
    if len(trace_ids) > 1:
        corrupt = True
        findings.append("conflicting_trace_id")
    if len(conversation_ids) > 1:
        corrupt = True
        findings.append("conflicting_conversation_id")

    receipt_corrupt, receipt_partial, receipt_findings = _receipt_integrity(events)
    corrupt = corrupt or receipt_corrupt
    partial = partial or receipt_partial
    findings.extend(receipt_findings)

    completeness = (
        EvidenceCompleteness.CORRUPT
        if corrupt
        else EvidenceCompleteness.PARTIAL
        if partial
        else EvidenceCompleteness.COMPLETE
    )
    return TurnEvidenceReadResult(turn_id, completeness, tuple(events), tuple(dict.fromkeys(findings)))


__all__ = [
    "AGENT_ENTERED",
    "CHECKPOINT_REFERENCE",
    "DECISION_RECEIPT",
    "DELIVERY_OUTCOME",
    "EvidenceCompleteness",
    "KNOWLEDGE_USAGE",
    "PROVIDER_ATTEMPT_COMPLETED",
    "PROVIDER_ATTEMPT_STARTED",
    "PROVIDER_RECEIPT_SCHEMA",
    "ProviderAttemptEvidence",
    "ProviderCallEvidenceContext",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SESSION_BOUNDARY",
    "TURN_STARTED",
    "TURN_TERMINAL",
    "TOOL_EXECUTION_COMPLETED",
    "TOOL_EXECUTION_STARTED",
    "TOOL_RECEIPT_SCHEMA",
    "ToolExecutionEvidence",
    "WRITE_DEGRADED",
    "TurnEvidenceEvent",
    "TurnEvidenceReadResult",
    "TurnEvidenceRecorder",
    "current",
    "canonical_digest",
    "emit_current",
    "provider_call_scope",
    "read_turn_evidence",
    "turn_scope",
]
