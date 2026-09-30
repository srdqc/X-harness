"""Independent structural verification for immutable Turn replay projections.

The verifier proves only facts supported by :class:`TraceReplayResult`.  It
does not inspect Agent prose and has no live execution or mutation dependency.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

from . import evidence
from .replay import SCHEMA as REPLAY_SCHEMA
from .replay import SCHEMA_VERSION as REPLAY_SCHEMA_VERSION
from .replay import TraceReplayResult, compute_replay_digest

SCHEMA = "pico.trace-verification.v1"
SCHEMA_VERSION = 1
CHECK_DEFINITIONS_VERSION = 1

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_OUTCOMES = {
    "completed",
    "completed_with_tool_failure",
    "provider_failed",
    "error",
    "cancelled",
}
_DELIVERY_OUTCOMES = {"delivered", "dropped", "no_outlet"}
_TOOL_EFFECTS = {"unknown", "read", "write", "execute", "external"}


class VerificationStatus(str, Enum):
    PASS = "pass"  # noqa: S105 -- verification outcome, not a credential
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class TraceVerificationProfile:
    """External deterministic contract; never populated from Agent output."""

    profile_id: str = "core"
    version: int = 1
    verification_receipt_ids: tuple[str, ...] = ()
    require_verification_after_mutation: bool = False


@dataclass(frozen=True)
class VerificationCheck:
    check_id: str
    category: str
    status: VerificationStatus
    required: bool
    evidence_references: tuple[str, ...]
    message: str
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class TraceDerivedFacts:
    logical_provider_call_count: int
    provider_attempt_count: int
    provider_retry_count: int
    provider_fallback_count: int
    provider_recovered_after_failure: bool
    tool_execution_count: int
    direct_tool_execution_count: int
    meta_routed_tool_execution_count: int
    unresolved_tool_count: int
    incomplete_provider_attempt_count: int
    incomplete_tool_execution_count: int
    tool_success_count: int
    tool_failure_count: int
    tool_unknown_count: int
    validation_failure_count: int
    execution_failure_count: int
    timeout_count: int
    duplicate_model_call_ids: tuple[str, ...]
    repeated_resolved_executions: tuple[str, ...]
    read_observed: bool
    mutation_observed: bool
    execute_observed: bool
    external_effect_observed: bool
    external_state_current: bool | None
    verification_after_last_mutation: bool | None
    runtime_success: bool
    runtime_failure: bool
    runtime_cancelled: bool
    delivery_success: bool
    delivery_failed: bool
    delivery_dropped: bool
    delivery_unknown: bool
    task_success_proven: bool


@dataclass(frozen=True)
class TraceVerificationResult:
    schema: str
    schema_version: int
    check_definitions_version: int
    profile_id: str
    profile_version: int
    profile_digest: str
    turn_id: str
    replay_digest: str
    overall_status: VerificationStatus
    checks: tuple[VerificationCheck, ...]
    derived_facts: TraceDerivedFacts
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    inconclusive_reasons: tuple[str, ...]
    source_evidence_references: tuple[str, ...]
    verification_digest: str


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


def _refs(replay: TraceReplayResult, sequences: set[int] | None = None) -> tuple[str, ...]:
    return tuple(
        item.evidence_id
        for item in replay.source_evidence_references
        if sequences is None or item.sequence in sequences
    )


def _check(
    check_id: str,
    category: str,
    status: VerificationStatus,
    message: str,
    *,
    required: bool = True,
    refs: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> VerificationCheck:
    return VerificationCheck(
        check_id=check_id,
        category=category,
        status=status,
        required=required,
        evidence_references=refs,
        message=message,
        metadata=tuple(sorted((metadata or {}).items())),
    )


def _schema_and_digest_check(replay: TraceReplayResult) -> VerificationCheck:
    if replay.schema != REPLAY_SCHEMA or replay.schema_version != REPLAY_SCHEMA_VERSION:
        return _check("core.replay_integrity", "integrity", VerificationStatus.FAIL, "unsupported replay schema")
    if not _DIGEST.fullmatch(replay.replay_digest):
        return _check("core.replay_integrity", "integrity", VerificationStatus.FAIL, "invalid replay digest")
    try:
        computed_digest = compute_replay_digest(replay)
    except (AttributeError, TypeError, ValueError):
        return _check("core.replay_integrity", "integrity", VerificationStatus.FAIL, "malformed replay structure")
    if computed_digest != replay.replay_digest:
        return _check("core.replay_integrity", "integrity", VerificationStatus.FAIL, "replay digest mismatch")
    return _check("core.replay_integrity", "integrity", VerificationStatus.PASS, "replay structure is digest-bound")


def _evidence_status_check(replay: TraceReplayResult) -> VerificationCheck:
    if replay.evidence_status is evidence.EvidenceCompleteness.CORRUPT or replay.inconsistencies:
        return _check(
            "core.evidence_status",
            "integrity",
            VerificationStatus.FAIL,
            "replay evidence is corrupt or contradictory",
            refs=_refs(replay),
        )
    if replay.evidence_status is evidence.EvidenceCompleteness.PARTIAL or replay.warnings:
        return _check(
            "core.evidence_status",
            "integrity",
            VerificationStatus.INCONCLUSIVE,
            "replay evidence is incomplete or degraded",
            refs=_refs(replay),
        )
    return _check(
        "core.evidence_status",
        "integrity",
        VerificationStatus.PASS,
        "replay evidence is structurally complete",
        refs=_refs(replay),
    )


def _identity_check(replay: TraceReplayResult) -> VerificationCheck:
    if not replay.turn_id:
        return _check("core.identity", "identity", VerificationStatus.FAIL, "turn identity is missing")
    timeline_by_id = {item.evidence_id: item for item in replay.ordered_timeline}
    source_ids = [item.evidence_id for item in replay.source_evidence_references]
    if len(source_ids) != len(set(source_ids)):
        return _check("core.identity", "identity", VerificationStatus.FAIL, "duplicate evidence identity")
    if set(source_ids) != set(timeline_by_id):
        return _check("core.identity", "identity", VerificationStatus.FAIL, "timeline provenance mismatch")
    if any(not identity.startswith(f"{replay.turn_id}:") for identity in source_ids):
        return _check("core.identity", "identity", VerificationStatus.FAIL, "foreign Turn evidence")
    timeline_trace_ids = {item.trace_id for item in replay.ordered_timeline if item.trace_id}
    if len(timeline_trace_ids) > 1 or (
        replay.trace_id is not None and timeline_trace_ids and timeline_trace_ids != {replay.trace_id}
    ):
        return _check("core.identity", "identity", VerificationStatus.FAIL, "conflicting trace correlation")
    if replay.trace_id is None:
        return _check(
            "core.identity",
            "identity",
            VerificationStatus.INCONCLUSIVE,
            "trace correlation is unavailable",
            refs=_refs(replay),
        )
    return _check(
        "core.identity", "identity", VerificationStatus.PASS, "Turn and trace identities correlate", refs=_refs(replay)
    )


def _ordering_check(replay: TraceReplayResult) -> VerificationCheck:
    sequences = [item.sequence for item in replay.ordered_timeline]
    if len(sequences) != len(set(sequences)):
        return _check("core.sequence", "ordering", VerificationStatus.FAIL, "duplicate Turn sequence")
    if sequences != sorted(sequences):
        return _check("core.sequence", "ordering", VerificationStatus.FAIL, "timeline is not sequence ordered")
    if sequences and sequences != list(range(1, max(sequences) + 1)):
        return _check("core.sequence", "ordering", VerificationStatus.INCONCLUSIVE, "Turn sequence has a gap")
    starts = [item.sequence for item in replay.ordered_timeline if item.event_type == evidence.TURN_STARTED]
    terminals = [item.sequence for item in replay.ordered_timeline if item.event_type == evidence.TURN_TERMINAL]
    if len(starts) > 1 or len(terminals) > 1:
        return _check("core.sequence", "ordering", VerificationStatus.FAIL, "conflicting lifecycle markers")
    if starts and terminals and terminals[0] <= starts[0]:
        return _check("core.sequence", "ordering", VerificationStatus.FAIL, "terminal precedes Turn start")
    if not starts or not terminals:
        return _check("core.sequence", "ordering", VerificationStatus.INCONCLUSIVE, "lifecycle boundary is incomplete")
    return _check(
        "core.sequence", "ordering", VerificationStatus.PASS, "durable Turn sequence is valid", refs=_refs(replay)
    )


def _provider_check(replay: TraceReplayResult) -> VerificationCheck:
    if not replay.provider_calls:
        return _check(
            "core.provider_receipts",
            "provider",
            VerificationStatus.INCONCLUSIVE,
            "no Provider receipt is available",
            required=False,
        )
    logical_ids: set[str] = set()
    attempt_ids: set[str] = set()
    incomplete = False
    sequences: set[int] = set()
    for call in replay.provider_calls:
        if not call.logical_call_id or call.logical_call_id in logical_ids:
            return _check("core.provider_receipts", "provider", VerificationStatus.FAIL, "invalid logical call identity")
        logical_ids.add(call.logical_call_id)
        ordinals = [attempt.attempt_ordinal for attempt in call.attempts]
        if ordinals != list(range(1, len(ordinals) + 1)):
            return _check("core.provider_receipts", "provider", VerificationStatus.FAIL, "invalid attempt ordinal chain")
        for attempt in call.attempts:
            if attempt.logical_call_id != call.logical_call_id or not attempt.attempt_id:
                return _check("core.provider_receipts", "provider", VerificationStatus.FAIL, "attempt identity mismatch")
            if attempt.attempt_id in attempt_ids:
                return _check("core.provider_receipts", "provider", VerificationStatus.FAIL, "duplicate attempt identity")
            attempt_ids.add(attempt.attempt_id)
            if attempt.started_sequence is not None:
                sequences.add(attempt.started_sequence)
            if attempt.completed_sequence is not None:
                sequences.add(attempt.completed_sequence)
            if (
                attempt.started_sequence is not None
                and attempt.completed_sequence is not None
                and attempt.completed_sequence <= attempt.started_sequence
            ):
                return _check("core.provider_receipts", "provider", VerificationStatus.FAIL, "attempt completion ordering is invalid")
            incomplete = incomplete or not attempt.complete
    status = VerificationStatus.INCONCLUSIVE if incomplete else VerificationStatus.PASS
    message = "Provider attempt completion is unknown" if incomplete else "Provider receipt chains are valid"
    return _check("core.provider_receipts", "provider", status, message, refs=_refs(replay, sequences))


def _tool_check(replay: TraceReplayResult) -> VerificationCheck:
    receipt_ids: set[str] = set()
    by_model_call: dict[str, list[Any]] = {}
    incomplete = False
    sequences: set[int] = set()
    for item in replay.tool_executions:
        if not item.receipt_id or item.receipt_id in receipt_ids:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "duplicate Tool receipt identity")
        receipt_ids.add(item.receipt_id)
        if item.model_call_id:
            by_model_call.setdefault(item.model_call_id, []).append(item)
        if not item.requested_name:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool request identity is missing")
        if item.resolved_name is not None and item.resolved_name != item.requested_name:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "requested/resolved Tool mismatch")
        if item.resolved_name is None and item.outcome == "success":
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "unresolved Tool reports success")
        if item.resolved_name is None and item.complete and item.failure_stage != "resolution":
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "unresolved Tool lacks resolution failure")
        if item.effect is not None and item.effect not in _TOOL_EFFECTS:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "unsupported Tool effect")
        if item.complete and item.outcome not in {"success", "failure", "cancelled"}:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "unsupported Tool outcome")
        if item.outcome == "failure" and not item.failure_stage:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool failure stage is missing")
        if item.started_sequence is not None:
            sequences.add(item.started_sequence)
        if item.completed_sequence is not None:
            sequences.add(item.completed_sequence)
        if (
            item.started_sequence is not None
            and item.completed_sequence is not None
            and item.completed_sequence <= item.started_sequence
        ):
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool completion ordering is invalid")
        incomplete = incomplete or not item.complete

    for child in replay.tool_executions:
        if child.parent_call_id is None:
            continue
        parents = by_model_call.get(child.parent_call_id, [])
        if len(parents) != 1:
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool parent is missing or ambiguous")
        parent = parents[0]
        if (
            parent.started_sequence is not None
            and child.started_sequence is not None
            and parent.started_sequence >= child.started_sequence
        ):
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool parent does not precede child")
        if (
            parent.completed_sequence is not None
            and child.completed_sequence is not None
            and parent.completed_sequence <= child.completed_sequence
        ):
            return _check("core.tool_receipts", "tool", VerificationStatus.FAIL, "Tool parent completion does not enclose child")

    duplicate_model_calls = [key for key, values in by_model_call.items() if len(values) > 1]
    if duplicate_model_calls:
        return _check(
            "core.tool_receipts",
            "tool",
            VerificationStatus.INCONCLUSIVE,
            "one model call identity has multiple execution receipts",
            refs=_refs(replay, sequences),
            metadata={"duplicate_model_call_count": len(duplicate_model_calls)},
        )
    if incomplete:
        return _check(
            "core.tool_receipts",
            "tool",
            VerificationStatus.INCONCLUSIVE,
            "Tool completion is unknown",
            refs=_refs(replay, sequences),
        )
    return _check(
        "core.tool_receipts", "tool", VerificationStatus.PASS, "Tool receipts are structurally valid", refs=_refs(replay, sequences)
    )


def _terminal_check(replay: TraceReplayResult) -> VerificationCheck:
    terminal = replay.runtime_outcome
    if terminal is None:
        status = VerificationStatus.FAIL if replay.evidence_status is evidence.EvidenceCompleteness.CORRUPT else VerificationStatus.INCONCLUSIVE
        return _check("core.terminal", "runtime", status, "Runtime terminal outcome is unavailable")
    if terminal.outcome not in _TERMINAL_OUTCOMES:
        return _check("core.terminal", "runtime", VerificationStatus.FAIL, "unsupported Runtime outcome")
    expected_event = "TurnEnded" if terminal.outcome.startswith("completed") else "TurnFailed"
    if terminal.lifecycle_event != expected_event:
        return _check("core.terminal", "runtime", VerificationStatus.FAIL, "terminal lifecycle contradicts outcome")
    return _check(
        "core.terminal",
        "runtime",
        VerificationStatus.PASS,
        "Runtime terminal evidence is consistent",
        refs=_refs(replay, {terminal.sequence}),
    )


def _session_reference_check(replay: TraceReplayResult) -> VerificationCheck:
    boundary = replay.session_boundary_ref
    if boundary is None:
        return _check(
            "optional.session_reference",
            "correlation",
            VerificationStatus.INCONCLUSIVE,
            "Session boundary was not observed",
            required=False,
        )
    if boundary.turn_id != replay.turn_id:
        return _check("core.session_reference", "correlation", VerificationStatus.FAIL, "Session boundary Turn mismatch")
    if boundary.resolved is not True:
        return _check("core.session_reference", "correlation", VerificationStatus.INCONCLUSIVE, "Session boundary is not resolved")
    return _check(
        "core.session_reference",
        "correlation",
        VerificationStatus.PASS,
        "Session boundary correlates",
        refs=_refs(replay),
    )


def _checkpoint_reference_check(replay: TraceReplayResult) -> VerificationCheck:
    boundary = replay.session_boundary_ref
    if not replay.checkpoint_refs:
        return _check(
            "optional.checkpoint_references",
            "correlation",
            VerificationStatus.INCONCLUSIVE,
            "Checkpoint was not observed",
            required=False,
        )
    for checkpoint in replay.checkpoint_refs:
        if checkpoint.turn_id != replay.turn_id:
            return _check("core.checkpoint_references", "correlation", VerificationStatus.FAIL, "Checkpoint Turn mismatch")
        if boundary is not None and (
            checkpoint.session_id != boundary.session_id or checkpoint.boundary_id != boundary.boundary_id
        ):
            return _check("core.checkpoint_references", "correlation", VerificationStatus.FAIL, "Checkpoint Session boundary mismatch")
        if checkpoint.resolved is not True:
            return _check("core.checkpoint_references", "correlation", VerificationStatus.INCONCLUSIVE, "Checkpoint is not resolved")
    return _check(
        "core.checkpoint_references",
        "correlation",
        VerificationStatus.PASS,
        "Checkpoint references correlate",
        refs=_refs(replay),
    )


def _delivery_check(replay: TraceReplayResult) -> VerificationCheck:
    if not replay.delivery_outcomes:
        return _check(
            "optional.delivery",
            "delivery",
            VerificationStatus.INCONCLUSIVE,
            "Delivery was not observed",
            required=False,
        )
    timeline = {(item.sequence, item.event_type) for item in replay.ordered_timeline}
    for delivery in replay.delivery_outcomes:
        if delivery.outcome not in _DELIVERY_OUTCOMES:
            return _check("optional.delivery", "delivery", VerificationStatus.FAIL, "unsupported Delivery outcome")
        if (delivery.sequence, evidence.DELIVERY_OUTCOME) not in timeline:
            return _check("optional.delivery", "delivery", VerificationStatus.FAIL, "Delivery provenance mismatch")
    return _check(
        "optional.delivery",
        "delivery",
        VerificationStatus.PASS,
        "Delivery evidence is structurally valid",
        required=False,
        refs=_refs(replay, {item.sequence for item in replay.delivery_outcomes}),
    )


def _provenance_check(replay: TraceReplayResult) -> VerificationCheck:
    for item in replay.source_evidence_references:
        if not _DIGEST.fullmatch(item.structural_digest):
            return _check("core.provenance", "integrity", VerificationStatus.FAIL, "invalid source evidence digest")
    timeline = [(item.sequence, item.event_type, item.evidence_id) for item in replay.ordered_timeline]
    sources = [(item.sequence, item.event_type, item.evidence_id) for item in replay.source_evidence_references]
    if timeline != sources:
        return _check("core.provenance", "integrity", VerificationStatus.FAIL, "source provenance is not aligned")
    return _check(
        "core.provenance", "integrity", VerificationStatus.PASS, "source provenance is structurally valid", refs=_refs(replay)
    )


def _facts(replay: TraceReplayResult, profile: TraceVerificationProfile) -> TraceDerivedFacts:
    attempts = tuple(attempt for call in replay.provider_calls for attempt in call.attempts)
    tools = replay.tool_executions
    successful = tuple(item for item in tools if item.complete and item.outcome == "success")
    model_counts = Counter(item.model_call_id for item in tools if item.model_call_id)
    resolved_counts = Counter(item.resolved_name for item in tools if item.resolved_name)
    terminal = replay.runtime_outcome
    delivery_outcomes = {item.outcome for item in replay.delivery_outcomes}

    mutations = tuple(item for item in successful if item.effect == "write")
    verification_after: bool | None = None
    if mutations and profile.verification_receipt_ids:
        last_mutation = max(item.completed_sequence or -1 for item in mutations)
        verification_tools = [item for item in successful if item.receipt_id in profile.verification_receipt_ids]
        if verification_tools:
            verification_after = any((item.started_sequence or -1) > last_mutation for item in verification_tools)

    return TraceDerivedFacts(
        logical_provider_call_count=len(replay.provider_calls),
        provider_attempt_count=len(attempts),
        provider_retry_count=sum(call.retry_count for call in replay.provider_calls),
        provider_fallback_count=sum(call.fallback_count for call in replay.provider_calls),
        provider_recovered_after_failure=any(
            any(attempt.outcome != "success" for attempt in call.attempts[:-1])
            and call.final_known_outcome == "success"
            for call in replay.provider_calls
        ),
        tool_execution_count=len(tools),
        direct_tool_execution_count=sum(item.routed_via is None for item in tools),
        meta_routed_tool_execution_count=sum(item.routed_via is not None for item in tools),
        unresolved_tool_count=sum(item.resolved_name is None for item in tools),
        incomplete_provider_attempt_count=sum(not item.complete for item in attempts),
        incomplete_tool_execution_count=sum(not item.complete for item in tools),
        tool_success_count=sum(item.complete and item.outcome == "success" for item in tools),
        tool_failure_count=sum(item.complete and item.outcome == "failure" for item in tools),
        tool_unknown_count=sum(not item.complete for item in tools),
        validation_failure_count=sum(item.failure_stage == "schema_validation" for item in tools),
        execution_failure_count=sum(item.failure_stage == "execution" for item in tools),
        timeout_count=sum(item.failure_stage == "timeout" for item in tools),
        duplicate_model_call_ids=tuple(sorted(key for key, count in model_counts.items() if count > 1)),
        repeated_resolved_executions=tuple(sorted(key for key, count in resolved_counts.items() if count > 1)),
        read_observed=any(item.effect == "read" for item in successful),
        mutation_observed=bool(mutations),
        execute_observed=any(item.effect == "execute" for item in successful),
        external_effect_observed=any(item.effect == "external" for item in successful),
        external_state_current=None,
        verification_after_last_mutation=verification_after,
        runtime_success=terminal is not None and terminal.outcome.startswith("completed"),
        runtime_failure=terminal is not None and terminal.outcome in {"provider_failed", "error"},
        runtime_cancelled=terminal is not None and terminal.outcome == "cancelled",
        delivery_success="delivered" in delivery_outcomes,
        delivery_failed=bool(delivery_outcomes - {"delivered", "dropped"}),
        delivery_dropped="dropped" in delivery_outcomes,
        delivery_unknown=not replay.delivery_outcomes,
        task_success_proven=False,
    )


def _verification_after_mutation_check(
    facts: TraceDerivedFacts,
    profile: TraceVerificationProfile,
) -> VerificationCheck:
    required = profile.require_verification_after_mutation
    if not facts.mutation_observed:
        return _check(
            "optional.verification_after_mutation",
            "derived_fact",
            VerificationStatus.INCONCLUSIVE,
            "no completed mutation is evidenced",
            required=required,
        )
    if facts.verification_after_last_mutation is True:
        return _check(
            "optional.verification_after_mutation",
            "derived_fact",
            VerificationStatus.PASS,
            "explicit verification receipt follows the last mutation",
            required=required,
        )
    if facts.verification_after_last_mutation is False:
        return _check(
            "optional.verification_after_mutation",
            "derived_fact",
            VerificationStatus.FAIL if required else VerificationStatus.INCONCLUSIVE,
            "explicit verification does not follow the last mutation",
            required=required,
        )
    return _check(
        "optional.verification_after_mutation",
        "derived_fact",
        VerificationStatus.INCONCLUSIVE,
        "no explicit verification-role evidence is configured",
        required=required,
    )


def _overall(checks: tuple[VerificationCheck, ...]) -> VerificationStatus:
    if any(item.status is VerificationStatus.FAIL for item in checks):
        return VerificationStatus.FAIL
    if any(item.required and item.status is VerificationStatus.INCONCLUSIVE for item in checks):
        return VerificationStatus.INCONCLUSIVE
    return VerificationStatus.PASS


def verification_structural_payload(result: TraceVerificationResult) -> dict[str, Any]:
    return {
        "schema": result.schema,
        "schema_version": result.schema_version,
        "check_definitions_version": result.check_definitions_version,
        "profile_id": result.profile_id,
        "profile_version": result.profile_version,
        "profile_digest": result.profile_digest,
        "turn_id": result.turn_id,
        "replay_digest": result.replay_digest,
        "overall_status": result.overall_status.value,
        "checks": _canonical(result.checks),
        "derived_facts": _canonical(result.derived_facts),
        "warnings": list(result.warnings),
        "failures": list(result.failures),
        "inconclusive_reasons": list(result.inconclusive_reasons),
        "source_evidence_references": list(result.source_evidence_references),
    }


def compute_verification_digest(result: TraceVerificationResult) -> str:
    return evidence.canonical_digest(verification_structural_payload(result))


def verify_replay(
    replay: TraceReplayResult,
    profile: TraceVerificationProfile | None = None,
) -> TraceVerificationResult:
    """Verify one immutable replay without consulting any live Runtime service."""

    selected = profile or TraceVerificationProfile()
    checks_list = [
        _schema_and_digest_check(replay),
        _evidence_status_check(replay),
        _identity_check(replay),
        _ordering_check(replay),
        _provider_check(replay),
        _tool_check(replay),
        _terminal_check(replay),
        _session_reference_check(replay),
        _checkpoint_reference_check(replay),
        _delivery_check(replay),
        _provenance_check(replay),
    ]
    facts = _facts(replay, selected)
    checks_list.append(_verification_after_mutation_check(facts, selected))
    checks = tuple(checks_list)
    failures = tuple(item.message for item in checks if item.status is VerificationStatus.FAIL)
    inconclusive = tuple(item.message for item in checks if item.status is VerificationStatus.INCONCLUSIVE)
    warnings = tuple(dict.fromkeys((*replay.warnings, *(item.message for item in checks if not item.required and item.status is VerificationStatus.INCONCLUSIVE))))
    result = TraceVerificationResult(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        check_definitions_version=CHECK_DEFINITIONS_VERSION,
        profile_id=selected.profile_id,
        profile_version=selected.version,
        profile_digest=evidence.canonical_digest(_canonical(selected)),
        turn_id=replay.turn_id,
        replay_digest=replay.replay_digest,
        overall_status=_overall(checks),
        checks=checks,
        derived_facts=facts,
        warnings=warnings,
        failures=failures,
        inconclusive_reasons=inconclusive,
        source_evidence_references=tuple(item.evidence_id for item in replay.source_evidence_references),
        verification_digest="",
    )
    return replace(result, verification_digest=compute_verification_digest(result))


__all__ = [
    "CHECK_DEFINITIONS_VERSION",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TraceDerivedFacts",
    "TraceVerificationProfile",
    "TraceVerificationResult",
    "VerificationCheck",
    "VerificationStatus",
    "compute_verification_digest",
    "verification_structural_payload",
    "verify_replay",
]
