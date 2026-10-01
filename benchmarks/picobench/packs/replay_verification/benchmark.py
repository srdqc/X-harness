"""Frozen, offline P1C replay/verification acceptance benchmark.

Fixtures contain identities, outcomes, sequence, and digests only.  They never
invoke the historical Provider, Tool, Delivery, Session, or Workspace action.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_bytes, canonical_digest, to_primitive
from benchmarks.picobench.schema import ExperimentRef
from pico.tracing import evidence, replay, verifier

BENCHMARK_SCHEMA = "pico.picobench.replay-verification.v1"
BENCHMARK_VERSION = 1
BENCHMARK_ID = "p1c-replay-verification-v1"
_TIMESTAMP = "2026-01-01T00:00:00+00:00"
_TRACE_ID = "trace-frozen"
_CONVERSATION_ID = "picobench:replay-verification"
_SENSITIVE_MARKERS = (
    "tests passed secret final answer",
    "agent claims failure secret final answer",
    "raw prompt secret",
    "raw tool argument secret",
    "raw tool result secret",
)

ScenarioClass = Literal["valid", "partial", "corrupt"]


@dataclass(frozen=True)
class ReplayVerificationScenarioResult:
    scenario_id: str
    scenario_class: ScenarioClass
    replay_status: str
    verification_status: str
    expected_replay_status: str
    expected_verification_status: str
    expectation_met: bool
    replay_digest: str
    verification_digest: str
    event_count: int
    persisted_evidence_bytes: int
    replay_result_bytes: int
    verification_result_bytes: int
    replay_latency_ns: int
    verification_latency_ns: int
    agent_claim: str | None
    task_success_proven: bool
    mutation_observed: bool
    verification_after_last_mutation: bool | None
    external_state_current: bool | None
    provider_attempt_count: int
    provider_retry_count: int
    provider_fallback_count: int
    tool_execution_count: int
    direct_tool_execution_count: int
    meta_routed_tool_execution_count: int
    tool_effects: tuple[str, ...]
    validation_failure_count: int
    execution_failure_count: int
    timeout_count: int
    runtime_success: bool
    runtime_failure: bool
    runtime_cancelled: bool
    delivery_success: bool
    delivery_dropped: bool
    delivery_unknown: bool


@dataclass(frozen=True)
class ReplayVerificationMetrics:
    valid_scenario_count: int
    valid_replay_pass_count: int
    valid_replay_pass_rate: float
    partial_scenario_count: int
    correct_inconclusive_count: int
    correct_inconclusive_rate: float
    corruption_scenario_count: int
    corruption_detected_count: int
    corruption_detection_rate: float
    false_positive_corruption_count: int
    false_negative_corruption_count: int
    total_evidence_events: int
    total_persisted_evidence_bytes: int
    total_replay_result_bytes: int
    total_verification_result_bytes: int
    total_replay_latency_ns: int
    total_verification_latency_ns: int
    privacy_safe: bool
    side_effect_calls: int


@dataclass(frozen=True)
class ReplayVerificationBenchmarkResult:
    schema: str
    version: int
    benchmark_id: str
    passed: bool
    metrics: ReplayVerificationMetrics
    scenarios: tuple[ReplayVerificationScenarioResult, ...]
    semantic_digest: str
    report_path: Path


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    scenario_class: ScenarioClass
    recipe: str = "basic"
    runtime_outcome: str = "completed"
    delivery_outcome: str | None = None
    expected_replay_status: str = "complete"
    expected_verification_status: str = "pass"
    mutation: str | None = None
    join_mode: str | None = None
    agent_final_text: str | None = None
    agent_claim: str | None = None
    require_verification: bool = False
    verification_receipt_ids: tuple[str, ...] = ()
    reverse_append_order: bool = False


_SCENARIOS = (
    _Scenario("valid-success", "valid"),
    _Scenario("valid-runtime-failure", "valid", runtime_outcome="error"),
    _Scenario("valid-cancelled", "valid", runtime_outcome="cancelled"),
    _Scenario("valid-provider-retry-fallback", "valid", recipe="provider-retry"),
    _Scenario("valid-direct-read", "valid", recipe="direct-read"),
    _Scenario("valid-meta-write", "valid", recipe="meta-write"),
    _Scenario("valid-effect-matrix", "valid", recipe="effect-matrix"),
    _Scenario("valid-runtime-success-delivery-dropped", "valid", delivery_outcome="dropped"),
    _Scenario(
        "valid-runtime-failure-delivery-success",
        "valid",
        runtime_outcome="error",
        delivery_outcome="delivered",
    ),
    _Scenario("valid-session-checkpoint", "valid", recipe="joined", join_mode="valid"),
    _Scenario(
        "valid-explicit-verification-after-mutation",
        "valid",
        recipe="write-verify",
        require_verification=True,
        verification_receipt_ids=("receipt-verify",),
    ),
    _Scenario(
        "valid-agent-failure-claim-ignored",
        "valid",
        agent_final_text="agent claims failure secret final answer",
        agent_claim="failure",
        reverse_append_order=True,
    ),
    _Scenario(
        "partial-provider-incomplete",
        "partial",
        recipe="provider-incomplete",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-tool-incomplete",
        "partial",
        recipe="tool-incomplete",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-write-incomplete",
        "partial",
        recipe="write-incomplete",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-external-incomplete",
        "partial",
        recipe="external-incomplete",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-missing-terminal",
        "partial",
        recipe="missing-terminal",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-write-degraded",
        "partial",
        recipe="write-degraded",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-unresolved-reference",
        "partial",
        recipe="joined",
        join_mode="unresolved",
        expected_replay_status="complete",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-sequence-gap",
        "partial",
        recipe="sequence-gap",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-legacy-untracked-tool",
        "partial",
        recipe="legacy-untracked",
        expected_replay_status="partial",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "partial-agent-success-claim-unproven",
        "partial",
        recipe="write-only",
        expected_replay_status="complete",
        expected_verification_status="inconclusive",
        agent_final_text="tests passed secret final answer",
        agent_claim="success",
        require_verification=True,
    ),
    _Scenario(
        "partial-duplicate-model-call-ambiguous",
        "partial",
        recipe="duplicate-model-call",
        expected_replay_status="complete",
        expected_verification_status="inconclusive",
    ),
    _Scenario(
        "corrupt-duplicate-sequence",
        "corrupt",
        recipe="duplicate-sequence",
        expected_replay_status="corrupt",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-conflicting-terminal",
        "corrupt",
        recipe="conflicting-terminal",
        expected_replay_status="corrupt",
        expected_verification_status="fail",
    ),
    _Scenario("corrupt-wrong-turn", "corrupt", mutation="wrong-turn", expected_verification_status="fail"),
    _Scenario(
        "corrupt-provider-ownership",
        "corrupt",
        mutation="provider-ownership",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-provider-ordinal",
        "corrupt",
        mutation="provider-ordinal",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-duplicate-receipt",
        "corrupt",
        recipe="duplicate-receipt",
        expected_replay_status="corrupt",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-impossible-parent",
        "corrupt",
        recipe="direct-read",
        mutation="parent",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-requested-resolved",
        "corrupt",
        recipe="direct-read",
        mutation="requested-resolved",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-session-boundary",
        "corrupt",
        recipe="joined",
        join_mode="session-mismatch",
        expected_replay_status="corrupt",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-checkpoint",
        "corrupt",
        recipe="joined",
        join_mode="checkpoint-mismatch",
        expected_replay_status="corrupt",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-delivery-wrong-turn",
        "corrupt",
        delivery_outcome="delivered",
        mutation="delivery-wrong-turn",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-altered-replay-digest",
        "corrupt",
        mutation="replay-digest",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-altered-source-digest",
        "corrupt",
        mutation="source-digest",
        expected_verification_status="fail",
    ),
    _Scenario(
        "corrupt-unsupported-schema",
        "corrupt",
        mutation="unsupported-schema",
        expected_verification_status="fail",
    ),
)


class _EvidenceBuilder:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.sequence = 0
        self.records: list[dict[str, Any]] = []

    def emit(
        self,
        event_type: str,
        *,
        correlations: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        sequence: int | None = None,
        failures: int = 0,
    ) -> int:
        if sequence is None:
            self.sequence += 1
            sequence = self.sequence
        else:
            self.sequence = max(self.sequence, sequence)
        self.records.append(
            evidence.TurnEvidenceEvent(
                turn_id=self.turn_id,
                sequence=sequence,
                event_type=event_type,
                timestamp=_TIMESTAMP,
                conversation_id=_CONVERSATION_ID,
                trace_id=_TRACE_ID,
                span_id=f"span-{sequence}",
                correlations=correlations or {},
                metadata=metadata or {},
                write_failures_before=failures,
            ).to_record()
        )
        return sequence

    def provider_attempt(
        self,
        logical_call_id: str,
        ordinal: int,
        *,
        outcome: str | None = "success",
        model: str = "model-a",
    ) -> None:
        attempt_id = f"{logical_call_id}:attempt:{ordinal}"
        correlation = {
            "logical_call_id": logical_call_id,
            "attempt_id": attempt_id,
            "attempt_ordinal": ordinal,
        }
        self.emit(
            evidence.PROVIDER_ATTEMPT_STARTED,
            correlations=correlation,
            metadata={
                "receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA,
                "requested_model": "model-a",
                "attempted_model": model,
                "provider": "frozen",
                "request_digest": evidence.canonical_digest({"fixture": logical_call_id, "ordinal": ordinal}),
            },
        )
        if outcome is not None:
            self.emit(
                evidence.PROVIDER_ATTEMPT_COMPLETED,
                correlations=correlation,
                metadata={
                    "receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA,
                    "actual_model": model,
                    "outcome": outcome,
                    "finish_reason": "stop" if outcome == "success" else "error",
                    "error_category": None if outcome == "success" else "frozen_failure",
                    "response_digest": evidence.canonical_digest({"fixture": "response", "ordinal": ordinal}),
                    "usage_available": True,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "duration_ms": 1,
                },
            )

    def tool_start(
        self,
        receipt_id: str,
        name: str,
        *,
        model_call_id: str,
        parent_call_id: str | None = None,
        routed_via: str | None = None,
        effect: str | None = "read",
        resolved_name: str | None | object = ...,
    ) -> dict[str, Any]:
        resolved = name if resolved_name is ... else resolved_name
        correlation = {
            "receipt_id": receipt_id,
            "model_call_id": model_call_id,
            "parent_call_id": parent_call_id,
        }
        self.emit(
            evidence.TOOL_EXECUTION_STARTED,
            correlations=correlation,
            metadata={
                "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
                "requested_name": name,
                "resolved_name": resolved,
                "routed_via": routed_via,
                "effect": effect,
                "argument_digest": evidence.canonical_digest({"fixture": receipt_id}),
            },
        )
        return correlation

    def tool_complete(
        self,
        correlation: dict[str, Any],
        *,
        outcome: str = "success",
        failure_stage: str | None = None,
        failure_category: str | None = None,
    ) -> None:
        self.emit(
            evidence.TOOL_EXECUTION_COMPLETED,
            correlations=correlation,
            metadata={
                "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
                "outcome": outcome,
                "failure_stage": failure_stage,
                "failure_category": failure_category,
                "result_digest": evidence.canonical_digest({"fixture": correlation["receipt_id"], "result": True}),
                "result_size": 8,
                "duration_ms": 1,
            },
        )


def _build_records(spec: _Scenario) -> tuple[str, tuple[dict[str, Any], ...]]:
    turn_id = f"benchmark-{spec.scenario_id}"
    builder = _EvidenceBuilder(turn_id)
    builder.emit(evidence.TURN_STARTED, metadata={"origin": "user", "channel": "picobench"})
    builder.emit(evidence.AGENT_ENTERED)

    if spec.recipe == "provider-retry":
        builder.provider_attempt("provider-call-1", 1, outcome="error")
        builder.provider_attempt("provider-call-1", 2, model="model-b")
    elif spec.recipe == "provider-incomplete":
        builder.provider_attempt("provider-call-1", 1, outcome=None)
    else:
        builder.provider_attempt("provider-call-1", 1)

    if spec.recipe == "direct-read":
        call = builder.tool_start("receipt-read", "read_file", model_call_id="model-read")
        builder.tool_complete(call)
    elif spec.recipe == "meta-write":
        outer = builder.tool_start("receipt-outer", "tool_call", model_call_id="model-outer")
        child = builder.tool_start(
            "receipt-write",
            "write_file",
            model_call_id="model-child",
            parent_call_id="model-outer",
            routed_via="tool_call",
            effect="write",
        )
        builder.tool_complete(child)
        builder.tool_complete(outer)
    elif spec.recipe == "effect-matrix":
        for effect in ("read", "write", "execute", "external"):
            call = builder.tool_start(
                f"receipt-{effect}",
                f"fixture_{effect}",
                model_call_id=f"model-{effect}",
                effect=effect,
            )
            builder.tool_complete(call)
        schema = builder.tool_start("receipt-schema", "fixture_write", model_call_id="model-schema", effect="write")
        builder.tool_complete(
            schema,
            outcome="failure",
            failure_stage="schema_validation",
            failure_category="invalid_arguments",
        )
        timeout = builder.tool_start("receipt-timeout", "fixture_external", model_call_id="model-timeout", effect="external")
        builder.tool_complete(timeout, outcome="failure", failure_stage="timeout", failure_category="timeout")
        execution = builder.tool_start("receipt-execution", "fixture_external", model_call_id="model-execution", effect="external")
        builder.tool_complete(execution, outcome="failure", failure_stage="execution", failure_category="remote")
        unresolved = builder.tool_start(
            "receipt-unresolved",
            "missing_tool",
            model_call_id="model-unresolved",
            routed_via="tool_call",
            effect=None,
            resolved_name=None,
        )
        builder.tool_complete(
            unresolved,
            outcome="failure",
            failure_stage="resolution",
            failure_category="not_found",
        )
    elif spec.recipe == "duplicate-model-call":
        duplicate_a = builder.tool_start(
            "receipt-repeat-a",
            "repeat_read",
            model_call_id="model-repeat",
            effect="read",
        )
        builder.tool_complete(duplicate_a)
        duplicate_b = builder.tool_start(
            "receipt-repeat-b",
            "repeat_read",
            model_call_id="model-repeat",
            effect="read",
        )
        builder.tool_complete(duplicate_b)
    elif spec.recipe in {"tool-incomplete", "write-incomplete", "external-incomplete"}:
        effect = {"tool-incomplete": "read", "write-incomplete": "write", "external-incomplete": "external"}[spec.recipe]
        builder.tool_start("receipt-incomplete", f"fixture_{effect}", model_call_id="model-incomplete", effect=effect)
    elif spec.recipe in {"write-only", "write-verify"}:
        mutation = builder.tool_start("receipt-write", "fixture_write", model_call_id="model-write", effect="write")
        builder.tool_complete(mutation)
        if spec.recipe == "write-verify":
            check = builder.tool_start("receipt-verify", "frozen_check", model_call_id="model-verify", effect="execute")
            builder.tool_complete(check)
    elif spec.recipe == "duplicate-receipt":
        call = builder.tool_start("receipt-duplicate", "read_file", model_call_id="model-read")
        builder.tool_complete(call)
        builder.tool_complete(call, outcome="failure", failure_stage="execution", failure_category="duplicate")

    if spec.recipe == "joined":
        builder.emit(
            evidence.SESSION_BOUNDARY,
            correlations={"session_id": _CONVERSATION_ID, "boundary_id": "boundary-1"},
            metadata={"boundary_version": 1, "message_count": 4, "history_digest": "history-digest"},
        )
        builder.emit(
            evidence.CHECKPOINT_REFERENCE,
            correlations={
                "session_id": _CONVERSATION_ID,
                "boundary_id": "boundary-1",
                "checkpoint_record_id": "record-1",
                "checkpoint_id": "checkpoint-1",
            },
            metadata={
                "checkpoint_status": "created",
                "checkpoint_revision": 1,
                "checkpoint_schema_version": 1,
            },
        )

    if spec.recipe == "write-degraded":
        builder.emit(
            evidence.WRITE_DEGRADED,
            correlations={"failed_sequence": builder.sequence},
            metadata={"failed_event_type": "fixture.event"},
            failures=1,
        )

    if spec.recipe == "sequence-gap":
        builder.sequence += 1

    tool_count = sum(
        record.get("event_type") == evidence.TOOL_EXECUTION_STARTED
        and not (record.get("correlations") or {}).get("parent_call_id")
        for record in builder.records
    )
    if spec.recipe == "legacy-untracked":
        tool_count += 1

    if spec.recipe != "missing-terminal":
        builder.emit(
            evidence.TURN_TERMINAL,
            metadata={
                "outcome": spec.runtime_outcome,
                "lifecycle_event": "TurnEnded" if spec.runtime_outcome.startswith("completed") else "TurnFailed",
                "tool_calls": tool_count,
                "tool_failures": sum(
                    record.get("event_type") == evidence.TOOL_EXECUTION_COMPLETED
                    and (record.get("metadata") or {}).get("outcome") == "failure"
                    for record in builder.records
                ),
            },
        )

    if spec.delivery_outcome is not None:
        builder.emit(
            evidence.DELIVERY_OUTCOME,
            correlations={"delivery_trace_id": "delivery-trace"},
            metadata={
                "channel": "picobench",
                "event": "Text",
                "outcome": spec.delivery_outcome,
                "attempts": 1,
                "error_class": None,
            },
        )

    if spec.recipe == "duplicate-sequence":
        builder.emit(evidence.AGENT_ENTERED, sequence=2, metadata={"conflict": True})
    elif spec.recipe == "conflicting-terminal":
        builder.emit(
            evidence.TURN_TERMINAL,
            metadata={"outcome": "error", "lifecycle_event": "TurnFailed"},
        )

    records = tuple(reversed(builder.records)) if spec.reverse_append_order else tuple(builder.records)
    return turn_id, records


def _lookups(spec: _Scenario, turn_id: str):
    if spec.join_mode is None:
        return None, None
    if spec.join_mode == "unresolved":
        return (lambda _session, _boundary: None), (lambda _record: None)

    boundary = SimpleNamespace(
        boundary_id="boundary-1",
        version=1,
        message_count=4,
        history_digest="wrong-history" if spec.join_mode == "session-mismatch" else "history-digest",
        turn_id=turn_id,
    )
    checkpoint = SimpleNamespace(
        record_id="record-1",
        checkpoint_id="checkpoint-1",
        revision=1,
        status="created",
        schema_version=1,
        session_id=_CONVERSATION_ID,
        boundary_id="wrong-boundary" if spec.join_mode == "checkpoint-mismatch" else "boundary-1",
        turn_id=turn_id,
    )
    return (lambda _session, _boundary: boundary), (lambda _record: checkpoint)


def _redigest(value: replay.TraceReplayResult) -> replay.TraceReplayResult:
    candidate = replace(value, replay_digest="")
    return replace(candidate, replay_digest=replay.compute_replay_digest(candidate))


def _mutate(value: replay.TraceReplayResult, mutation: str | None) -> replay.TraceReplayResult:
    if mutation is None:
        return value
    if mutation == "wrong-turn":
        timeline = (replace(value.ordered_timeline[0], evidence_id="foreign-turn:1:turn.started"), *value.ordered_timeline[1:])
        refs = (replace(value.source_evidence_references[0], evidence_id="foreign-turn:1:turn.started"), *value.source_evidence_references[1:])
        return _redigest(replace(value, ordered_timeline=timeline, source_evidence_references=refs))
    if mutation in {"provider-ownership", "provider-ordinal"}:
        call = value.provider_calls[0]
        attempt = call.attempts[0]
        changed = replace(
            attempt,
            logical_call_id="foreign-call" if mutation == "provider-ownership" else attempt.logical_call_id,
            attempt_ordinal=2 if mutation == "provider-ordinal" else attempt.attempt_ordinal,
        )
        return _redigest(replace(value, provider_calls=(replace(call, attempts=(changed,)), *value.provider_calls[1:])))
    if mutation in {"parent", "requested-resolved"}:
        tool = value.tool_executions[0]
        changed = replace(
            tool,
            parent_call_id="missing-parent" if mutation == "parent" else tool.parent_call_id,
            resolved_name="different_tool" if mutation == "requested-resolved" else tool.resolved_name,
        )
        return _redigest(replace(value, tool_executions=(changed, *value.tool_executions[1:])))
    if mutation == "delivery-wrong-turn":
        delivery_index = next(
            index for index, item in enumerate(value.ordered_timeline) if item.event_type == evidence.DELIVERY_OUTCOME
        )
        changed_timeline = list(value.ordered_timeline)
        changed_refs = list(value.source_evidence_references)
        changed_timeline[delivery_index] = replace(
            changed_timeline[delivery_index], evidence_id="foreign-turn:delivery"
        )
        changed_refs[delivery_index] = replace(changed_refs[delivery_index], evidence_id="foreign-turn:delivery")
        return _redigest(
            replace(
                value,
                ordered_timeline=tuple(changed_timeline),
                source_evidence_references=tuple(changed_refs),
            )
        )
    if mutation == "replay-digest":
        return replace(value, replay_digest="0" * 64)
    if mutation == "source-digest":
        changed = replace(value.source_evidence_references[0], structural_digest="invalid")
        return _redigest(replace(value, source_evidence_references=(changed, *value.source_evidence_references[1:])))
    if mutation == "unsupported-schema":
        return _redigest(replace(value, schema_version=999))
    raise ValueError(f"unknown replay mutation: {mutation}")


def _write_fixture(path: Path, records: tuple[dict[str, Any], ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return len(payload)


def _run_scenario(output_root: Path, spec: _Scenario) -> ReplayVerificationScenarioResult:
    turn_id, records = _build_records(spec)
    state_dir = output_root / "fixtures" / spec.scenario_id
    evidence_bytes = _write_fixture(state_dir / "logs" / "audit-events.log", records)
    session_lookup, checkpoint_lookup = _lookups(spec, turn_id)

    replay_started = time.perf_counter_ns()
    replay_result = replay.replay_turn(
        state_dir,
        turn_id,
        session_boundary_lookup=session_lookup,
        checkpoint_record_lookup=checkpoint_lookup,
    )
    replay_latency = time.perf_counter_ns() - replay_started
    replay_result = _mutate(replay_result, spec.mutation)

    profile = verifier.TraceVerificationProfile(
        profile_id=f"benchmark-{spec.scenario_id}",
        verification_receipt_ids=spec.verification_receipt_ids,
        require_verification_after_mutation=spec.require_verification,
    )
    verification_started = time.perf_counter_ns()
    verification = verifier.verify_replay(replay_result, profile)
    verification_latency = time.perf_counter_ns() - verification_started

    replay_status = replay_result.evidence_status.value
    verification_status = verification.overall_status.value
    expectation_met = (
        replay_status == spec.expected_replay_status
        and verification_status == spec.expected_verification_status
    )
    return ReplayVerificationScenarioResult(
        scenario_id=spec.scenario_id,
        scenario_class=spec.scenario_class,
        replay_status=replay_status,
        verification_status=verification_status,
        expected_replay_status=spec.expected_replay_status,
        expected_verification_status=spec.expected_verification_status,
        expectation_met=expectation_met,
        replay_digest=replay_result.replay_digest,
        verification_digest=verification.verification_digest,
        event_count=len(records),
        persisted_evidence_bytes=evidence_bytes,
        replay_result_bytes=len(canonical_bytes(replay_result)),
        verification_result_bytes=len(canonical_bytes(verification)),
        replay_latency_ns=replay_latency,
        verification_latency_ns=verification_latency,
        agent_claim=spec.agent_claim,
        task_success_proven=verification.derived_facts.task_success_proven,
        mutation_observed=verification.derived_facts.mutation_observed,
        verification_after_last_mutation=verification.derived_facts.verification_after_last_mutation,
        external_state_current=verification.derived_facts.external_state_current,
        provider_attempt_count=verification.derived_facts.provider_attempt_count,
        provider_retry_count=verification.derived_facts.provider_retry_count,
        provider_fallback_count=verification.derived_facts.provider_fallback_count,
        tool_execution_count=verification.derived_facts.tool_execution_count,
        direct_tool_execution_count=verification.derived_facts.direct_tool_execution_count,
        meta_routed_tool_execution_count=verification.derived_facts.meta_routed_tool_execution_count,
        tool_effects=tuple(sorted({item.effect for item in replay_result.tool_executions if item.effect})),
        validation_failure_count=verification.derived_facts.validation_failure_count,
        execution_failure_count=verification.derived_facts.execution_failure_count,
        timeout_count=verification.derived_facts.timeout_count,
        runtime_success=verification.derived_facts.runtime_success,
        runtime_failure=verification.derived_facts.runtime_failure,
        runtime_cancelled=verification.derived_facts.runtime_cancelled,
        delivery_success=verification.derived_facts.delivery_success,
        delivery_dropped=verification.derived_facts.delivery_dropped,
        delivery_unknown=verification.derived_facts.delivery_unknown,
    )


def _metrics(results: tuple[ReplayVerificationScenarioResult, ...], privacy_safe: bool) -> ReplayVerificationMetrics:
    valid = tuple(item for item in results if item.scenario_class == "valid")
    partial = tuple(item for item in results if item.scenario_class == "partial")
    corrupt = tuple(item for item in results if item.scenario_class == "corrupt")
    valid_pass = sum(item.verification_status == "pass" and item.expectation_met for item in valid)
    partial_inconclusive = sum(
        item.verification_status == "inconclusive" and item.expectation_met for item in partial
    )
    corruption_detected = sum(item.verification_status == "fail" for item in corrupt)
    return ReplayVerificationMetrics(
        valid_scenario_count=len(valid),
        valid_replay_pass_count=valid_pass,
        valid_replay_pass_rate=valid_pass / len(valid),
        partial_scenario_count=len(partial),
        correct_inconclusive_count=partial_inconclusive,
        correct_inconclusive_rate=partial_inconclusive / len(partial),
        corruption_scenario_count=len(corrupt),
        corruption_detected_count=corruption_detected,
        corruption_detection_rate=corruption_detected / len(corrupt),
        false_positive_corruption_count=sum(item.verification_status == "fail" for item in (*valid, *partial)),
        false_negative_corruption_count=sum(item.verification_status != "fail" for item in corrupt),
        total_evidence_events=sum(item.event_count for item in results),
        total_persisted_evidence_bytes=sum(item.persisted_evidence_bytes for item in results),
        total_replay_result_bytes=sum(item.replay_result_bytes for item in results),
        total_verification_result_bytes=sum(item.verification_result_bytes for item in results),
        total_replay_latency_ns=sum(item.replay_latency_ns for item in results),
        total_verification_latency_ns=sum(item.verification_latency_ns for item in results),
        privacy_safe=privacy_safe,
        side_effect_calls=0,
    )


def _semantic_projection(results: tuple[ReplayVerificationScenarioResult, ...]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": item.scenario_id,
            "scenario_class": item.scenario_class,
            "replay_status": item.replay_status,
            "verification_status": item.verification_status,
            "expected_replay_status": item.expected_replay_status,
            "expected_verification_status": item.expected_verification_status,
            "expectation_met": item.expectation_met,
            "replay_digest": item.replay_digest,
            "verification_digest": item.verification_digest,
            "event_count": item.event_count,
            "persisted_evidence_bytes": item.persisted_evidence_bytes,
            "replay_result_bytes": item.replay_result_bytes,
            "verification_result_bytes": item.verification_result_bytes,
            "agent_claim": item.agent_claim,
            "task_success_proven": item.task_success_proven,
            "mutation_observed": item.mutation_observed,
            "verification_after_last_mutation": item.verification_after_last_mutation,
            "external_state_current": item.external_state_current,
            "provider_attempt_count": item.provider_attempt_count,
            "provider_retry_count": item.provider_retry_count,
            "provider_fallback_count": item.provider_fallback_count,
            "tool_execution_count": item.tool_execution_count,
            "direct_tool_execution_count": item.direct_tool_execution_count,
            "meta_routed_tool_execution_count": item.meta_routed_tool_execution_count,
            "tool_effects": item.tool_effects,
            "validation_failure_count": item.validation_failure_count,
            "execution_failure_count": item.execution_failure_count,
            "timeout_count": item.timeout_count,
            "runtime_success": item.runtime_success,
            "runtime_failure": item.runtime_failure,
            "runtime_cancelled": item.runtime_cancelled,
            "delivery_success": item.delivery_success,
            "delivery_dropped": item.delivery_dropped,
            "delivery_unknown": item.delivery_unknown,
        }
        for item in results
    ]


def run_replay_verification_benchmark(output_root: Path) -> ReplayVerificationBenchmarkResult:
    """Run the frozen P1C benchmark entirely from local structural evidence."""

    root = Path(output_root)
    results = tuple(_run_scenario(root, spec) for spec in _SCENARIOS)
    serialized = canonical_bytes(_semantic_projection(results))
    privacy_safe = all(marker.encode("utf-8") not in serialized for marker in _SENSITIVE_MARKERS)
    metrics = _metrics(results, privacy_safe)
    passed = (
        all(item.expectation_met for item in results)
        and metrics.valid_replay_pass_rate == 1.0
        and metrics.correct_inconclusive_rate == 1.0
        and metrics.corruption_detection_rate == 1.0
        and metrics.false_positive_corruption_count == 0
        and metrics.false_negative_corruption_count == 0
        and metrics.side_effect_calls == 0
        and metrics.privacy_safe
        and all(not item.task_success_proven for item in results)
    )
    semantic_digest = canonical_digest(
        {
            "schema": BENCHMARK_SCHEMA,
            "version": BENCHMARK_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "scenarios": _semantic_projection(results),
        }
    )
    report_path = root / "report.json"
    result = ReplayVerificationBenchmarkResult(
        schema=BENCHMARK_SCHEMA,
        version=BENCHMARK_VERSION,
        benchmark_id=BENCHMARK_ID,
        passed=passed,
        metrics=metrics,
        scenarios=results,
        semantic_digest=semantic_digest,
        report_path=report_path,
    )
    ArtifactStore(ExperimentRef(BENCHMARK_ID, root)).write_summary(report_path, to_primitive(result))
    return result


def read_benchmark_report(path: Path) -> dict[str, Any]:
    """Read a retained benchmark report for deterministic test/audit use."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("invalid replay verification benchmark report")
    return value


__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_SCHEMA",
    "BENCHMARK_VERSION",
    "ReplayVerificationBenchmarkResult",
    "ReplayVerificationMetrics",
    "ReplayVerificationScenarioResult",
    "read_benchmark_report",
    "run_replay_verification_benchmark",
]
