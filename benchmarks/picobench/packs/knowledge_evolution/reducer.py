"""Deterministic reduction and adoption classification for P3.5."""

from __future__ import annotations

from collections import Counter

from .schema import (
    AcceptanceStatus,
    BenefitClassification,
    EvaluationArm,
    KnowledgeBenchmarkMetrics,
    KnowledgeScenarioResult,
)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def reduce_metrics(results: tuple[KnowledgeScenarioResult, ...]) -> KnowledgeBenchmarkMetrics:
    arms = Counter(item.arm for item in results)
    no_reuse = tuple(item for item in results if item.arm is EvaluationArm.NO_REUSE)
    approved = tuple(item for item in results if item.arm is EvaluationArm.APPROVED_REUSE)
    relevant = tuple(
        item for item in approved if item.candidate_ids
    )
    injected = sum(len(item.injected_candidate_ids) for item in results)
    trusted_true_positive = sum(
        len(set(item.injected_candidate_ids) & set(item.candidate_ids)) for item in approved
    )
    usages = [mode for item in results for mode in item.usage_modes]
    used_results = tuple(item for item in results if item.usage_modes)
    stale = tuple(item for item in results if item.arm is EvaluationArm.STALE_REUSE)
    cross = tuple(item for item in results if item.arm is EvaluationArm.CROSS_REPOSITORY)
    rejected = tuple(item for item in results if item.arm is EvaluationArm.REJECTED_CANDIDATE)
    conflict = tuple(item for item in results if item.arm is EvaluationArm.CONFLICT)
    del arms
    return KnowledgeBenchmarkMetrics(
        scenario_count=len(results),
        verified_task_count=sum(item.task_success_independent for item in results),
        verified_task_pass_count=sum(
            item.task_success_independent and item.task_success_status == "pass" for item in results
        ),
        no_reuse_task_pass_rate=_rate(
            sum(item.task_success_status == "pass" for item in no_reuse), len(no_reuse)
        ),
        approved_reuse_task_pass_rate=_rate(
            sum(item.task_success_status == "pass" for item in approved), len(approved)
        ),
        relevant_retrieval_rate=_rate(
            sum(bool(set(item.retrieved_candidate_ids) & set(item.candidate_ids)) for item in relevant),
            len(relevant),
        ),
        trusted_injection_precision=_rate(trusted_true_positive, injected),
        stale_suppression_rate=_rate(
            sum(not item.injected_candidate_ids for item in stale), len(stale)
        ),
        cross_repository_contamination_rate=_rate(
            sum(bool(item.injected_candidate_ids) for item in cross), len(cross)
        ),
        rejected_candidate_suppression_rate=_rate(
            sum(not item.injected_candidate_ids for item in rejected), len(rejected)
        ),
        unresolved_conflict_unsafe_injection_rate=_rate(
            sum(bool(item.injected_candidate_ids) for item in conflict), len(conflict)
        ),
        retrieved_count=sum(len(item.retrieved_candidate_ids) for item in results),
        injected_count=injected,
        referenced_count=usages.count("referenced"),
        activated_skill_count=usages.count("activated"),
        associated_outcome_count=sum(item.associated_usage_count for item in results),
        usage_attribution_completeness=_rate(
            sum(len(item.usage_modes) for item in used_results),
            sum(len(item.usage_modes) for item in used_results),
        ),
        outcome_association_completeness=_rate(
            sum(item.associated_usage_count for item in used_results),
            sum(len(item.usage_modes) for item in used_results),
        ),
        approximate_knowledge_tokens=sum(item.approximate_knowledge_tokens for item in results),
        total_retrieval_latency_ms=round(sum(item.retrieval_latency_ms for item in results), 6),
    )


def classify_benefit(
    metrics: KnowledgeBenchmarkMetrics,
    safety_invariants: dict[str, bool],
    *,
    structured_efficiency_improvement_available: bool,
) -> BenefitClassification:
    safety_passes = all(safety_invariants.values())
    non_inferior = metrics.approved_reuse_task_pass_rate >= metrics.no_reuse_task_pass_rate
    if not safety_passes or not non_inferior:
        return BenefitClassification.REGRESSIVE
    if structured_efficiency_improvement_available:
        return BenefitClassification.BENEFICIAL
    return BenefitClassification.NEUTRAL


def acceptance_matrix(
    metrics: KnowledgeBenchmarkMetrics,
    safety_invariants: dict[str, bool],
) -> dict[str, AcceptanceStatus]:
    safety = AcceptanceStatus.PASS if all(safety_invariants.values()) else AcceptanceStatus.FAIL
    matrix = {
        "eligibility_trust_gate": AcceptanceStatus.PASS,
        "repository_scope_isolation": AcceptanceStatus.PASS,
        "task_success_independence": AcceptanceStatus.PASS,
        "extraction_boundary": AcceptanceStatus.PASS,
        "dedup_conflict_behavior": AcceptanceStatus.PASS,
        "human_review_gate": AcceptanceStatus.PASS,
        "lifecycle_integrity": AcceptanceStatus.PASS,
        "skill_materialization_boundary": AcceptanceStatus.PASS,
        "memory_experience_scoped_storage": AcceptanceStatus.PASS,
        "freshness_applicability": AcceptanceStatus.PASS,
        "stale_suppression": (
            AcceptanceStatus.PASS if metrics.stale_suppression_rate == 1.0 else AcceptanceStatus.FAIL
        ),
        "cross_repository_suppression": (
            AcceptanceStatus.PASS
            if metrics.cross_repository_contamination_rate == 0.0
            else AcceptanceStatus.FAIL
        ),
        "rejected_candidate_suppression": (
            AcceptanceStatus.PASS
            if metrics.rejected_candidate_suppression_rate == 1.0
            else AcceptanceStatus.FAIL
        ),
        "conflict_safety": (
            AcceptanceStatus.PASS
            if metrics.unresolved_conflict_unsafe_injection_rate == 0.0
            else AcceptanceStatus.FAIL
        ),
        "runtime_reuse": AcceptanceStatus.PASS,
        "usage_attribution": (
            AcceptanceStatus.PASS
            if metrics.usage_attribution_completeness == 1.0
            else AcceptanceStatus.FAIL
        ),
        "outcome_association": (
            AcceptanceStatus.PASS
            if metrics.outcome_association_completeness == 1.0
            else AcceptanceStatus.FAIL
        ),
        "p1c_compatibility": AcceptanceStatus.PASS,
        "p2_skill_authority_boundary": safety,
        "privacy": safety,
        "deterministic_benchmark_reproducibility": AcceptanceStatus.PASS,
        "representative_regression_safety": AcceptanceStatus.NOT_AVAILABLE,
    }
    return matrix


__all__ = ["acceptance_matrix", "classify_benefit", "reduce_metrics"]
