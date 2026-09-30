"""Durable, structural evidence for one Runtime Turn.

The envelope is deliberately smaller than the span model.  It records ordering,
identity and durable references, but never Provider prompts, Tool arguments or
outputs.  Recording is observational: every public write helper is no-throw.
"""

from __future__ import annotations

import contextlib
import contextvars
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

    @property
    def write_failures(self) -> int:
        with self._lock:
            return self._write_failures

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


_CURRENT: contextvars.ContextVar[TurnEvidenceRecorder | None] = contextvars.ContextVar(
    "pico_turn_evidence_recorder", default=None
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
    "DELIVERY_OUTCOME",
    "EvidenceCompleteness",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SESSION_BOUNDARY",
    "TURN_STARTED",
    "TURN_TERMINAL",
    "WRITE_DEGRADED",
    "TurnEvidenceEvent",
    "TurnEvidenceReadResult",
    "TurnEvidenceRecorder",
    "current",
    "emit_current",
    "read_turn_evidence",
    "turn_scope",
]
