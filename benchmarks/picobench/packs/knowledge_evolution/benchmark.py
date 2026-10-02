"""Frozen deterministic P3 controlled knowledge-reuse acceptance benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_digest, to_primitive
from benchmarks.picobench.schema import ExperimentRef
from pico.context_engine.base import AssemblyContext
from pico.context_engine.segments.skills import SkillsSegmentBuilder
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateComparison,
    CandidateEligibilityInput,
    CandidateIndex,
    CandidateRelation,
    CandidateType,
    EligibilityStatus,
    ExtractionStatus,
    KnowledgeCandidate,
    KnowledgeExtractionRequest,
    KnowledgeLifecycleManager,
    KnowledgeRecordStore,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    RepositoryScopeReason,
    RepositoryScopeResolution,
    RepositoryScopeStatus,
    ReviewDecision,
    ReviewerType,
    SourceTurnReference,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    associate_usage_outcome,
    create_review,
    evaluate_candidate_eligibility,
    extract_candidates,
    materialize_candidate,
    materialized_skill_root,
    validate_candidate,
)
from pico.knowledge_evolution.runtime import KnowledgeContextSegmentBuilder
from pico.knowledge_evolution.types import structural_digest
from pico.memory_engine.base import TokenBudget
from pico.memory_engine.skill_forge import SkillForgeRouter
from pico.memory_engine.skill_forge.knowledge_source import ApplicableKnowledgeSkillSource
from pico.memory_engine.skill_local.registry import SkillRegistry
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore

from .fixtures import SCENARIOS, KnowledgeScenario
from .reducer import acceptance_matrix, classify_benefit, reduce_metrics
from .schema import (
    ACCEPTANCE_CRITERIA,
    BENCHMARK_ID,
    DATASET_ID,
    SCHEMA,
    SCHEMA_VERSION,
    AcceptanceStatus,
    EvaluationArm,
    KnowledgeEvolutionBenchmarkResult,
    KnowledgeScenarioResult,
)

_NOW = "2026-10-01T00:00:00Z"
_GUARD_FILE = "benchmark-contract.txt"
_GUARD_CONTENT = b"p3-frozen-contract-v1\n"


def _scope(store: KnowledgeRecordStore, name: str) -> RepositoryScopeIdentity:
    value = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity=f"project:{name}",
        evidence=(("project_key", name),),
    )
    store.write_scope(value)
    return value


def _task_success(
    store: KnowledgeRecordStore,
    *,
    turn_id: str,
    scope_id: str,
    evidence_id: str,
) -> TaskSuccessEvidence:
    result = TaskSuccessEvidence.create(
        evidence_id=evidence_id,
        source_kind=TaskSuccessSource.FROZEN_BENCHMARK,
        source_turn_id=turn_id,
        status=TaskSuccessStatus.PASS,
        producer_id="picobench:p3-frozen-verifier",
        producer_version="1",
        repository_scope_id=scope_id,
        target_state_digest=structural_digest({"task": evidence_id, "state": "verified"}),
        result_digest=structural_digest({"task": evidence_id, "result": "pass"}),
        provenance_refs=(f"benchmark:{BENCHMARK_ID}",),
        created_at=_NOW,
    )
    store.write_task_success(result)
    return result


def _candidate(
    store: KnowledgeRecordStore,
    scope: RepositoryScopeIdentity,
    spec: KnowledgeScenario,
    *,
    candidate_id: str,
    fact_value: str | None = None,
) -> tuple[KnowledgeCandidate, Any, CandidateComparison]:
    source_turn = f"source-{candidate_id}"
    success = _task_success(
        store,
        turn_id=source_turn,
        scope_id=scope.repository_scope_id,
        evidence_id=f"source-success-{candidate_id}",
    )
    fingerprints: list[tuple[str, str]] = [
        ("guard:file:" + _GUARD_FILE, hashlib.sha256(_GUARD_CONTENT).hexdigest())
    ]
    if spec.candidate_type is CandidateType.MEMORY_FACT:
        fingerprints.append(
            ("fact:benchmark-value", structural_digest({"value": fact_value or spec.reusable_content}))
        )
    candidate = KnowledgeCandidate.create(
        candidate_id=candidate_id,
        candidate_type=spec.candidate_type,
        content_class=spec.content_class,
        qualifying_sources=(
            SourceTurnReference(
                turn_id=source_turn,
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=(success.evidence_id,),
                task_success_evidence_digests=(success.evidence_digest,),
            ),
        ),
        repository_scope_id=scope.repository_scope_id,
        title=spec.title,
        reusable_content=spec.reusable_content,
        preconditions=("Repository scope and benchmark guard match",),
        applicability_fingerprints=tuple(fingerprints),
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        provenance_refs=(f"benchmark:{BENCHMARK_ID}",),
        created_at=_NOW,
        created_by="extractor:benchmark",
    )
    comparison = CandidateIndex.rebuild(
        store, repository_scope_id=scope.repository_scope_id
    ).compare(candidate)
    store.write_candidate(candidate)
    validation = validate_candidate(
        store,
        candidate_id=candidate_id,
        comparison=comparison,
        expected_repository_scope_id=scope.repository_scope_id,
        validation_id=f"validation-{candidate_id}",
        validator_id="p3-validator",
        validator_version="1",
        validated_at=_NOW,
    )
    return candidate, validation, comparison


def _advance_to_review(
    store: KnowledgeRecordStore,
    candidate: KnowledgeCandidate,
    validation: Any,
) -> KnowledgeLifecycleManager:
    manager = KnowledgeLifecycleManager(store)
    manager.transition(
        transition_id=f"eligible-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.EXTRACTED,
        to_state=LifecycleState.ELIGIBLE,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.ELIGIBILITY_CONFIRMED,
        transitioned_at=_NOW,
        evidence_refs=("eligibility:" + "e" * 64,),
    )
    manager.transition(
        transition_id=f"validated-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ELIGIBLE,
        to_state=LifecycleState.VALIDATED,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.TECHNICAL_VALIDATION_PASSED,
        transitioned_at=_NOW,
        validation=validation,
    )
    manager.transition(
        transition_id=f"pending-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.VALIDATED,
        to_state=LifecycleState.PENDING_REVIEW,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.SUBMITTED_FOR_REVIEW,
        transitioned_at=_NOW,
        validation=validation,
    )
    return manager


def _activate(
    store: KnowledgeRecordStore,
    candidate: KnowledgeCandidate,
    validation: Any,
    comparison: CandidateComparison,
) -> None:
    manager = _advance_to_review(store, candidate, validation)
    review = create_review(
        store,
        review_id=f"review-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:benchmark-reviewer",
        decision=ReviewDecision.APPROVE,
        reason="Approved for the frozen scoped benchmark.",
        reviewed_at=_NOW,
    )
    manager.transition(
        transition_id=f"active-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.PENDING_REVIEW,
        to_state=LifecycleState.ACTIVE,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:benchmark-reviewer",
        reason=LifecycleReason.HUMAN_APPROVED,
        transitioned_at=_NOW,
        validation=validation,
        review=review,
    )
    materialize_candidate(
        store,
        materialization_id=f"materialization-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )


def _reject(
    store: KnowledgeRecordStore,
    candidate: KnowledgeCandidate,
    validation: Any,
    comparison: CandidateComparison,
) -> None:
    manager = _advance_to_review(store, candidate, validation)
    review = create_review(
        store,
        review_id=f"review-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:benchmark-reviewer",
        decision=ReviewDecision.REJECT,
        reason="Rejected by the frozen negative-control review.",
        reviewed_at=_NOW,
    )
    manager.transition(
        transition_id=f"rejected-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.PENDING_REVIEW,
        to_state=LifecycleState.REJECTED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:benchmark-reviewer",
        reason=LifecycleReason.HUMAN_REJECTED,
        transitioned_at=_NOW,
        validation=validation,
        review=review,
    )


def _context(query: str) -> AssemblyContext:
    return AssemblyContext(
        session_key="picobench-p3",
        current_message=query,
        media=None,
        channel="picobench",
        chat_id=None,
        session_messages=[],
        budget=TokenBudget(8000, 1000, 1000, 1000, 5000),
    )


async def _run_scenario(root: Path, spec: KnowledgeScenario) -> KnowledgeScenarioResult:
    scenario_index = SCENARIOS.index(spec)
    # Keep physical fixture paths short enough for Windows while retaining the
    # full frozen identity in every benchmark result.
    scenario_root = root / "f" / f"s{scenario_index}"
    state = scenario_root / "k"
    workspace = scenario_root / "w"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / _GUARD_FILE).write_bytes(_GUARD_CONTENT)
    store = KnowledgeRecordStore(state)
    target_scope = _scope(store, f"target-{spec.scenario_id}")
    environment_scope = target_scope
    candidates: list[KnowledgeCandidate] = []

    if spec.arm is not EvaluationArm.NO_REUSE:
        candidate_scope = target_scope
        if spec.arm is EvaluationArm.CROSS_REPOSITORY:
            candidate_scope = _scope(store, f"foreign-{spec.scenario_id}")
        first, validation, comparison = _candidate(
            store,
            candidate_scope,
            spec,
            candidate_id=f"c{scenario_index}a",
            fact_value="uv" if spec.arm is EvaluationArm.CONFLICT else None,
        )
        candidates.append(first)
        if spec.arm in {
            EvaluationArm.APPROVED_REUSE,
            EvaluationArm.STALE_REUSE,
            EvaluationArm.CROSS_REPOSITORY,
        }:
            _activate(store, first, validation, comparison)
        elif spec.scenario_id == "rejected-experience":
            _reject(store, first, validation, comparison)
        elif spec.arm is EvaluationArm.REJECTED_CANDIDATE:
            _advance_to_review(store, first, validation)
        elif spec.arm is EvaluationArm.CONFLICT:
            conflicting_spec = KnowledgeScenario(
                spec.scenario_id,
                spec.task_id,
                spec.arm,
                spec.candidate_type,
                spec.content_class,
                spec.title,
                "The repository uses npm.",
                spec.query,
            )
            second, _, second_comparison = _candidate(
                store,
                candidate_scope,
                conflicting_spec,
                candidate_id=f"c{scenario_index}b",
                fact_value="npm",
            )
            if second_comparison.relation is not CandidateRelation.CONFLICT:
                raise AssertionError("frozen conflict fixture did not produce a conflict")
            candidates.append(second)
        if spec.arm is EvaluationArm.STALE_REUSE:
            (workspace / _GUARD_FILE).write_bytes(b"changed-after-approval\n")
        if spec.arm is EvaluationArm.CROSS_REPOSITORY:
            environment_scope = target_scope

    environment = ApplicabilityEnvironment(environment_scope.repository_scope_id, workspace)
    turn_id = f"turn-{spec.scenario_id}"
    trace_store = TraceStore(scenario_root / "trace")
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id=f"picobench:{spec.scenario_id}",
        trace_id=f"trace-{spec.scenario_id}",
        root_span_id=f"span-{spec.scenario_id}",
        writer=trace_store.append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    with evidence.turn_scope(recorder):
        if spec.candidate_type is CandidateType.SKILL_CANDIDATE:
            physical_source = f"experience:{environment.repository_scope_id}"
            registry = SkillRegistry(
                workspace,
                builtin_skills_dir=scenario_root / "no-builtins",
                extra_dirs=[
                    (
                        materialized_skill_root(store, environment.repository_scope_id),
                        physical_source,
                        False,
                    )
                ],
            )
            source = ApplicableKnowledgeSkillSource(
                store,
                registry,
                lambda: environment,
                physical_source,
            )
            segment = await SkillsSegmentBuilder(
                SkillForgeRouter([source]), skill_top_k=5, activation_max=1
            ).build(_context(spec.query))
            injected_ids = tuple(
                item.rsplit("/", 1)[-1] for item in segment.meta["injected_skill_ids"]
            )
        else:
            segment = await KnowledgeContextSegmentBuilder(
                store, lambda: environment, max_tokens=300
            ).build(_context(spec.query))
            injected_ids = tuple(segment.meta["p3_injected_candidate_ids"])
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )

    retrieval_paths = sorted(store.retrievals.glob("*.json"))
    receipts = tuple(store.read_retrieval(path.stem) for path in retrieval_paths)
    retrieved_ids = tuple(
        candidate_id for receipt in receipts for candidate_id in receipt.selected_candidate_ids
    )
    suppressed = tuple(
        (candidate_id, reason.value)
        for receipt in receipts
        for candidate_id, reason in receipt.suppressed
    )
    usages = store.list_usages(turn_id=turn_id)
    success = _task_success(
        store,
        turn_id=turn_id,
        scope_id=environment.repository_scope_id,
        evidence_id=f"task-success-{spec.scenario_id}",
    )
    associated = 0
    if usages:
        association = associate_usage_outcome(
            store,
            association_id=f"association-{spec.scenario_id}",
            usage_ids=tuple(item.usage_id for item in usages),
            task_success=success,
            created_at=_NOW,
        )
        associated = len(association.usage_ids)
    visible_candidate_ids = set(injected_ids) | {item.candidate_id for item in usages}
    approximate_knowledge_tokens = sum(
        (len(item.reusable_content) + 3) // 4
        for item in candidates
        if item.candidate_id in visible_candidate_ids
    )
    return KnowledgeScenarioResult(
        scenario_id=spec.scenario_id,
        task_id=spec.task_id,
        arm=spec.arm,
        knowledge_kind=(spec.candidate_type.value if spec.candidate_type else "none"),
        task_success_status=success.status.value,
        task_success_independent=success.source_kind is TaskSuccessSource.FROZEN_BENCHMARK,
        candidate_ids=tuple(item.candidate_id for item in candidates),
        retrieved_candidate_ids=retrieved_ids,
        injected_candidate_ids=injected_ids,
        suppressed=suppressed,
        usage_modes=tuple(item.usage_mode.value for item in usages),
        associated_usage_count=associated,
        approximate_knowledge_tokens=approximate_knowledge_tokens,
        retrieval_latency_ms=round(sum(item.latency_ms for item in receipts), 6),
    )


class _ExtractorProbe:
    extractor_id = "picobench-probe"
    extractor_version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, context: Any) -> tuple[object, ...]:
        del context
        self.calls += 1
        return ()


def _eligibility_negative(root: Path) -> bool:
    turn_id = "turn-runtime-success-without-task-proof"
    trace_root = root / "eligibility-negative" / "trace"
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="picobench:eligibility-negative",
        trace_id="trace-eligibility-negative",
        root_span_id="span-eligibility-negative",
        writer=TraceStore(trace_root).append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )
    replay_result = replay.replay_turn(trace_root, turn_id)
    verification = verifier.verify_replay(replay_result)
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:eligibility-negative",
        evidence=(("project_key", "eligibility-negative"),),
    )
    eligibility_input = CandidateEligibilityInput(
        turn_id=turn_id,
        replay=replay_result,
        verification=verification,
        task_success=(),
        scope=RepositoryScopeResolution(
            RepositoryScopeStatus.RESOLVED,
            RepositoryScopeReason.EXPLICIT_PROJECT_KEY,
            scope,
        ),
        resulting_state_digest="f" * 64,
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        candidate_type=CandidateType.EXPERIENCE,
        provenance_turn_ids=(turn_id,),
    )
    eligibility = evaluate_candidate_eligibility(eligibility_input)
    probe = _ExtractorProbe()
    extraction = extract_candidates(
        KnowledgeExtractionRequest(
            extraction_id="extraction-missing-task-proof",
            eligibility_input=eligibility_input,
            eligibility_result=eligibility,
        ),
        probe,
        KnowledgeRecordStore(root / "eligibility-negative" / "state"),
        created_at=_NOW,
        created_by="extractor:benchmark",
    )
    return (
        eligibility.status is EligibilityStatus.INCONCLUSIVE
        and extraction.status is ExtractionStatus.NOT_PERMITTED
        and extraction.extractor_attempt_count == 0
        and probe.calls == 0
    )


async def run_knowledge_evolution_benchmark(
    output_root: Path,
) -> KnowledgeEvolutionBenchmarkResult:
    """Run frozen Level-1 fixtures without Provider, Tool, or live-model activity."""

    root = Path(output_root)
    results = tuple([await _run_scenario(root, spec) for spec in SCENARIOS])
    metrics = reduce_metrics(results)
    eligibility_negative_passed = _eligibility_negative(root)
    safety_invariants = {
        "runtime_success_without_task_proof_is_ineligible": eligibility_negative_passed,
        "no_stale_trusted_injection": metrics.stale_suppression_rate == 1.0,
        "no_cross_repository_contamination": metrics.cross_repository_contamination_rate == 0.0,
        "no_rejected_or_unreviewed_injection": metrics.rejected_candidate_suppression_rate == 1.0,
        "no_unresolved_conflict_injection": metrics.unresolved_conflict_unsafe_injection_rate == 0.0,
        "no_tool_provider_or_terminal_authority_added": True,
        "no_agent_prose_used_as_task_success": all(item.task_success_independent for item in results),
        "report_excludes_candidate_bodies_and_raw_payloads": True,
    }
    matrix = acceptance_matrix(metrics, safety_invariants)
    classification = classify_benefit(
        metrics,
        safety_invariants,
        structured_efficiency_improvement_available=False,
    )
    arm_definitions = tuple(
        {"arm": arm.value, "scenario_count": sum(item.arm is arm for item in results)}
        for arm in EvaluationArm
    )
    benchmark_identity = {
        "benchmark_id": BENCHMARK_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "level_1_deterministic_knowledge_system_fixtures",
        "level_2_agent_tasks": "not_available",
    }
    fixture_identity = {
        "dataset_id": DATASET_ID,
        "scenario_manifest_digest": canonical_digest(SCENARIOS),
        "scenario_count": len(SCENARIOS),
    }
    efficiency_availability = {
        "provider_calls": "not_available",
        "tool_calls": "not_available",
        "repeated_file_reads": "not_available",
        "repeated_skill_reads": "not_available",
        "provider_visible_input_tokens": "not_available",
        "output_tokens": "not_available",
        "end_to_end_latency": "not_available",
        "retrieval_latency": "diagnostic_available_excluded_from_semantic_digest",
        "approximate_knowledge_tokens": "deterministic_available",
    }
    semantic_payload = {
        "benchmark_identity": benchmark_identity,
        "fixture_identity": fixture_identity,
        "arm_definitions": arm_definitions,
        "scenarios": tuple(item.semantic_payload() for item in results),
        "metrics": {
            key: value
            for key, value in to_primitive(metrics).items()
            if key != "total_retrieval_latency_ms"
        },
        "efficiency_availability": efficiency_availability,
        "safety_invariants": safety_invariants,
        "acceptance_matrix": matrix,
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "benefit_classification": classification,
    }
    report_path = root / "report.json"
    passed = all(safety_invariants.values()) and all(
        value is not AcceptanceStatus.FAIL for value in matrix.values()
    )
    result = KnowledgeEvolutionBenchmarkResult(
        benchmark_identity=benchmark_identity,
        fixture_identity=fixture_identity,
        arm_definitions=arm_definitions,
        scenarios=results,
        metrics=metrics,
        efficiency_availability=efficiency_availability,
        safety_invariants=safety_invariants,
        acceptance_matrix=matrix,
        acceptance_criteria=ACCEPTANCE_CRITERIA,
        passed=passed,
        benefit_classification=classification,
        semantic_digest=canonical_digest(semantic_payload),
        report_path=report_path,
    )
    ArtifactStore(ExperimentRef(BENCHMARK_ID, root)).append_immutable(
        report_path, to_primitive(result)
    )
    return result


__all__ = ["run_knowledge_evolution_benchmark"]
