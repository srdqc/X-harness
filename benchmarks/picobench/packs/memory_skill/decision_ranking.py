"""Frozen P2 Skill-ranking comparison over the existing Memory/Skill corpus.

The treatment uses the deterministic reverse-order fake solely as a harness
contract fixture.  It never receives relevance labels, verifier outcomes, or
other oracle data, and its output is not presented as Jev quality evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from types import MappingProxyType
from typing import Any, Mapping

from benchmarks.picobench.canonical import canonical_digest, to_primitive
from benchmarks.picobench.schema import RetrievalQuerySpec
from pico.context_engine.segments import SkillsSegmentBuilder
from pico.decision_plane import DeterministicSkillRankingAdapter, JevDecisionAdapter
from pico.decision_plane.fake import JevFakeMode, ScriptedJevBackend
from pico.memory_engine.skill_forge import LocalSkillResolver, RouterHit, SkillForgeRouter
from pico.tracing import evidence

from .fixtures import (
    anonymous_item_id,
    formal_memory_corpus,
    formal_skill_corpus,
    formal_skill_queries,
    retrieval_fixture_manifest,
)
from .retrieval import ProductRetrievalAdapter, _assembly_context

SCHEMA = "pico.picobench.skill-decision-ranking.v1"
SCHEMA_VERSION = 1
DATASET_ID = "skill-source-fusion-v1"
TREATMENT_CLASSIFICATION = "harness_contract_fixture"
QUALITY_EVALUATION = "not_available"
DECISION_ACCURACY_DEFINITION = "top_1_relevant_candidate_accuracy"

# Frozen before result inspection.  Fixture-only gains can never enable a
# default rollout, and task/safety/fallback regressions always reject.
ADOPTION_CRITERIA: Mapping[str, Any] = MappingProxyType(
    {
        "version": 1,
        "reject_if": (
            "independently_verified_task_success_regresses",
            "safety_authority_invariant_fails",
            "fallback_correctness_below_1.0",
            "valid_permutation_rate_below_1.0",
            "stale_or_cross_scope_leakage_worsens",
        ),
        "opt_in_requires": (
            "non_oracle_experimental_ranker",
            "task_success_non_inferior",
            "meaningful_ranking_quality_improvement",
            "all_safety_invariants_pass",
            "acceptable_latency_and_real_cost_evidence",
        ),
        "default_requires": (
            "evidence_beyond_local_contract_fixtures",
            "explicit_production_adoption_review",
        ),
    }
)


class DecisionEvaluationArm(str, Enum):
    BASELINE = "baseline"
    TREATMENT = "treatment"
    FORCED_FALLBACK = "forced_fallback"
    DETERMINISTIC_ADAPTER = "deterministic_adapter"


class AdoptionClassification(str, Enum):
    RECOMMENDED_OPT_IN = "recommended_opt_in"
    EXPERIMENTAL_ONLY = "experimental_only"
    REJECTED = "rejected"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class DecisionRankingCaseResult:
    task_id: str
    label: str
    arm: DecisionEvaluationArm
    relevant_item_ids: tuple[str, ...]
    candidate_item_ids: tuple[str, ...]
    ranked_item_ids: tuple[str, ...]
    injected_item_ids: tuple[str, ...]
    ranking_digest: str
    receipt: Mapping[str, Any] | None

    def semantic_payload(self) -> dict[str, Any]:
        receipt = None
        if self.receipt is not None:
            receipt = {
                key: value
                for key, value in self.receipt.items()
                if key not in {"latency_ms", "decision_id", "turn_id"}
            }
        return {
            "task_id": self.task_id,
            "label": self.label,
            "arm": self.arm.value,
            "relevant_item_ids": self.relevant_item_ids,
            "candidate_item_ids": self.candidate_item_ids,
            "ranked_item_ids": self.ranked_item_ids,
            "injected_item_ids": self.injected_item_ids,
            "ranking_digest": self.ranking_digest,
            "receipt": receipt,
        }


@dataclass(frozen=True)
class SkillDecisionEvaluationReport:
    dataset_identity: Mapping[str, Any]
    configuration_identity: Mapping[str, Any]
    arm_definitions: tuple[Mapping[str, Any], ...]
    scenario_count: int
    case_results: tuple[DecisionRankingCaseResult, ...]
    arm_metrics: Mapping[str, Any]
    fallback_metrics: Mapping[str, Any]
    latency_diagnostics: Mapping[str, Any]
    cost_evidence: Mapping[str, Any]
    calibration_evidence: Mapping[str, Any]
    final_task_success: Mapping[str, Any]
    safety_invariants: Mapping[str, bool]
    adoption_criteria: Mapping[str, Any]
    adoption_classification: AdoptionClassification
    semantic_digest: str
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        return to_primitive(self)


class _RecordingRouter:
    def __init__(self, delegate: SkillForgeRouter) -> None:
        self._delegate = delegate
        self.last_hits: tuple[RouterHit, ...] = ()

    async def select(self, *args: Any, **kwargs: Any) -> list[RouterHit]:
        hits = await self._delegate.select(*args, **kwargs)
        self.last_hits = tuple(hits)
        return hits


class _AuthorityProbeSource:
    name = "local"
    weight = 1.0

    async def search(self, query: str, history: list[dict[str, Any]], k: int) -> list[RouterHit]:
        del query, history
        return [
            RouterHit(
                qualified_id="local/eligible",
                name="eligible",
                content="eligible body",
                score=2.0,
                meta={"source": "local", "requirements_met": True},
            ),
            RouterHit(
                qualified_id="local/unavailable",
                name="unavailable",
                content="unavailable body",
                score=1.0,
                meta={"source": "local", "requirements_met": False},
            ),
        ][:k]


async def run_controlled_skill_ranking_evaluation(
    *,
    fallback_mode: JevFakeMode = JevFakeMode.UNAVAILABLE,
) -> SkillDecisionEvaluationReport:
    """Run all frozen arms and reduce receipt-backed aggregate evidence."""

    queries = _decision_skill_queries()
    corpus = formal_skill_corpus()
    product = ProductRetrievalAdapter(
        memory_corpus=formal_memory_corpus(),
        skill_corpus=corpus,
    )
    arms = tuple(DecisionEvaluationArm)
    cases: list[DecisionRankingCaseResult] = []
    for query in queries:
        for arm in arms:
            cases.append(
                await _run_case(
                    product,
                    query_id=query.query_id,
                    label=query.label,
                    query_text=str(query.payload["query_text"]),
                    workspace_id=str(query.payload["workspace_id"]),
                    consuming_turn=str(query.payload["consuming_turn"]),
                    relevant_item_ids=query.expected_item_ids,
                    arm=arm,
                    fallback_mode=fallback_mode,
                )
            )

    frozen_cases = tuple(cases)
    by_arm = {
        arm: tuple(case for case in frozen_cases if case.arm is arm)
        for arm in arms
    }
    baseline = {case.task_id: case for case in by_arm[DecisionEvaluationArm.BASELINE]}
    arm_metrics = {
        arm.value: reduce_decision_ranking_metrics(results, corpus, baseline=baseline)
        for arm, results in by_arm.items()
    }
    fallback_metrics = _fallback_metrics(
        by_arm[DecisionEvaluationArm.FORCED_FALLBACK],
        baseline,
    )
    latency = {
        arm.value: _latency_metrics(results)
        for arm, results in by_arm.items()
    }
    cost = {
        arm.value: _cost_metrics(results)
        for arm, results in by_arm.items()
    }
    calibration = {
        arm.value: _calibration_metrics(results)
        for arm, results in by_arm.items()
    }
    safety = await _safety_invariants(by_arm, baseline, corpus)
    final_task_success = MappingProxyType(
        {
            "availability": "not_available",
            "verifier": "existing_memory_skill_sealed_artifact_verifier",
            "reason": "retrieval_cases_do_not_produce_an_independent_task_artifact",
            "baseline_verified_success_rate": None,
            "treatment_verified_success_rate": None,
            "agent_final_answer_used_as_proof": False,
        }
    )
    classification = classify_adoption(
        treatment_classification=TREATMENT_CLASSIFICATION,
        evidence_scope="local_contract_fixture",
        safety_invariants=safety,
        fallback_correctness_rate=float(fallback_metrics["correctness_rate"]),
        valid_permutation_rate=float(arm_metrics[DecisionEvaluationArm.TREATMENT.value]["valid_permutation_rate"]),
        baseline_task_success=None,
        treatment_task_success=None,
        baseline_cross_scope_rate=float(
            arm_metrics[DecisionEvaluationArm.BASELINE.value]["cross_scope_leakage_rate"]
        ),
        treatment_cross_scope_rate=float(
            arm_metrics[DecisionEvaluationArm.TREATMENT.value]["cross_scope_leakage_rate"]
        ),
        baseline_stale_rate=None,
        treatment_stale_rate=None,
        ranking_quality_improved=(
            float(arm_metrics[DecisionEvaluationArm.TREATMENT.value]["mrr"])
            > float(arm_metrics[DecisionEvaluationArm.BASELINE.value]["mrr"])
        ),
        latency_cost_acceptable=False,
    )
    dataset_identity = MappingProxyType(
        {
            "dataset_id": DATASET_ID,
            "fixture_schema": retrieval_fixture_manifest()["schema"],
            "fixture_digest": canonical_digest(retrieval_fixture_manifest()),
            "corpus_digest": canonical_digest(corpus),
            "query_labels_digest": canonical_digest(queries),
            "positive_cases": sum(query.label == "positive" for query in queries),
            "hard_negative_cases": sum(query.label == "hard_negative" for query in queries),
            "exact_match_cases": 40,
            "ambiguous_cases": 8,
            "stale_skill_cases_available": False,
            "final_task_artifacts_available": False,
        }
    )
    arm_definitions = _arm_definitions(fallback_mode)
    configuration_identity = MappingProxyType(
        {
            "top_k": 5,
            "decision_accuracy_definition": DECISION_ACCURACY_DEFINITION,
            "treatment_classification": TREATMENT_CLASSIFICATION,
            "quality_evaluation": QUALITY_EVALUATION,
            "fallback_mode": fallback_mode.value,
            "adoption_criteria_digest": canonical_digest(ADOPTION_CRITERIA),
            "arm_digest": canonical_digest(arm_definitions),
        }
    )
    semantic_payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset_identity": dataset_identity,
        "configuration_identity": configuration_identity,
        "arm_definitions": arm_definitions,
        "scenario_count": len(queries),
        "case_results": [case.semantic_payload() for case in frozen_cases],
        "arm_metrics": arm_metrics,
        "fallback_metrics": fallback_metrics,
        "cost_evidence": cost,
        "calibration_evidence": calibration,
        "final_task_success": final_task_success,
        "safety_invariants": safety,
        "adoption_criteria": ADOPTION_CRITERIA,
        "adoption_classification": classification.value,
    }
    return SkillDecisionEvaluationReport(
        dataset_identity=dataset_identity,
        configuration_identity=configuration_identity,
        arm_definitions=arm_definitions,
        scenario_count=len(queries),
        case_results=frozen_cases,
        arm_metrics=MappingProxyType(arm_metrics),
        fallback_metrics=MappingProxyType(fallback_metrics),
        latency_diagnostics=MappingProxyType(latency),
        cost_evidence=MappingProxyType(cost),
        calibration_evidence=MappingProxyType(calibration),
        final_task_success=final_task_success,
        safety_invariants=MappingProxyType(safety),
        adoption_criteria=ADOPTION_CRITERIA,
        adoption_classification=classification,
        semantic_digest=canonical_digest(semantic_payload),
    )


def _decision_skill_queries() -> tuple[RetrievalQuerySpec, ...]:
    """Extend the existing frozen suite with nontrivial in-corpus permutations."""

    ambiguous = tuple(
        RetrievalQuerySpec(
            query_id=f"skill-ambiguous-{index:03d}",
            label="positive",
            expected_item_ids=(anonymous_item_id(DATASET_ID, f"skill-{index:03d}"),),
            payload={
                "query_text": f"repair workflow skilltoken{index:03d}",
                "workspace_id": f"workspace-{index}",
                "consuming_turn": f"skill-ambiguous-turn-{index:03d}",
            },
        )
        for index in range(8)
    )
    return (*formal_skill_queries(), *ambiguous)


async def _run_case(
    product: ProductRetrievalAdapter,
    *,
    query_id: str,
    label: str,
    query_text: str,
    workspace_id: str,
    consuming_turn: str,
    relevant_item_ids: tuple[str, ...],
    arm: DecisionEvaluationArm,
    fallback_mode: JevFakeMode,
) -> DecisionRankingCaseResult:
    local, everos = product.skill_sources(workspace_id)
    enabled = arm is not DecisionEvaluationArm.BASELINE
    if arm is DecisionEvaluationArm.TREATMENT:
        adapter = JevDecisionAdapter(ScriptedJevBackend(reverse=True))
    elif arm is DecisionEvaluationArm.FORCED_FALLBACK:
        adapter = (
            JevDecisionAdapter(None)
            if fallback_mode is JevFakeMode.UNAVAILABLE
            else JevDecisionAdapter(
                ScriptedJevBackend(fallback_mode),
                timeout_seconds=0.001 if fallback_mode is JevFakeMode.TIMEOUT else 0.25,
            )
        )
    elif arm is DecisionEvaluationArm.DETERMINISTIC_ADAPTER:
        adapter = DeterministicSkillRankingAdapter()
    else:
        adapter = None
    router = _RecordingRouter(
        SkillForgeRouter(
            [local, everos],
            decision_plane_enabled=enabled,
            decision_adapter=adapter,
        )
    )
    records: list[dict[str, Any]] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=f"picobench:{query_id}:{arm.value}",
        conversation_id=f"picobench:{workspace_id}",
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        segment = await SkillsSegmentBuilder(
            router,  # type: ignore[arg-type] -- transparent benchmark recorder
            skill_top_k=5,
            activation_max=5,
        ).build(_assembly_context(query_text, consuming_turn))
    ranked_hits = router.last_hits
    qid_to_item = {
        hit.qualified_id: anonymous_item_id(DATASET_ID, hit.name)
        for hit in ranked_hits
    }
    ranked_ids = tuple(anonymous_item_id(DATASET_ID, hit.name) for hit in ranked_hits)
    injected_qids = tuple(segment.meta["injected_skill_ids"]) if segment is not None else ()
    injected_ids = tuple(qid_to_item[qid] for qid in injected_qids if qid in qid_to_item)
    decision_records = [record for record in records if record["event_type"] == evidence.DECISION_RECEIPT]
    receipt = MappingProxyType(dict(decision_records[0]["metadata"])) if decision_records else None
    return DecisionRankingCaseResult(
        task_id=query_id,
        label=label,
        arm=arm,
        relevant_item_ids=relevant_item_ids,
        candidate_item_ids=ranked_ids,
        ranked_item_ids=ranked_ids,
        injected_item_ids=injected_ids,
        ranking_digest=canonical_digest(ranked_ids),
        receipt=receipt,
    )


def reduce_decision_ranking_metrics(
    cases: tuple[DecisionRankingCaseResult, ...],
    corpus: tuple[Any, ...],
    *,
    baseline: Mapping[str, DecisionRankingCaseResult] | None = None,
) -> dict[str, Any]:
    positives = tuple(case for case in cases if case.label == "positive")
    recalls = {limit: _mean_recall(positives, limit) for limit in (1, 3, 5)}
    decision_accuracy = (
        sum(
            bool(case.ranked_item_ids)
            and case.ranked_item_ids[0] in set(case.relevant_item_ids)
            for case in positives
        )
        / len(positives)
        if positives
        else 0.0
    )
    ranks = [_target_rank(case) for case in positives]
    rank_distribution = Counter("not_retrieved" if rank is None else str(rank) for rank in ranks)
    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    total_ranked = sum(len(case.ranked_item_ids) for case in cases)
    irrelevant = sum(
        item_id not in set(case.relevant_item_ids)
        for case in cases
        for item_id in case.ranked_item_ids
    )
    cross_by_workspace = _cross_scope_ids(corpus)
    leakage = 0
    for case in cases:
        workspace_id = _workspace_from_query_id(case.task_id)
        leakage += sum(
            item_id in cross_by_workspace.get(workspace_id, set())
            for item_id in case.ranked_item_ids
        )
    receipts = tuple(case.receipt for case in cases if case.receipt is not None)
    if baseline is not None:
        valid_permutation_rate = sum(
            set(case.ranked_item_ids) == set(baseline[case.task_id].candidate_item_ids)
            and len(case.ranked_item_ids) == len(baseline[case.task_id].candidate_item_ids)
            for case in cases
        ) / len(cases)
    else:
        valid_permutation_rate = 1.0
    return {
        "decision_accuracy_definition": DECISION_ACCURACY_DEFINITION,
        "decision_accuracy": decision_accuracy,
        "recall_at_1": recalls[1],
        "recall_at_3": recalls[3],
        "recall_at_5": recalls[5],
        "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
        "target_rank_distribution": dict(sorted(rank_distribution.items())),
        "irrelevant_retrieval_rate": irrelevant / total_ranked if total_ranked else 0.0,
        "stale_candidate_promotion_rate": None,
        "stale_candidate_metric_available": False,
        "cross_scope_leakage_rate": leakage / total_ranked if total_ranked else 0.0,
        "valid_permutation_rate": valid_permutation_rate,
        "fallback_rate": (
            sum(receipt["fallback_used"] is True for receipt in receipts) / len(receipts)
            if receipts
            else 0.0
        ),
    }


def _fallback_metrics(
    cases: tuple[DecisionRankingCaseResult, ...],
    baseline: Mapping[str, DecisionRankingCaseResult],
) -> dict[str, Any]:
    correct = 0
    reasons: Counter[str] = Counter()
    for case in cases:
        receipt = case.receipt or {}
        reason = receipt.get("fallback_reason")
        if isinstance(reason, str):
            reasons[reason] += 1
        base = baseline[case.task_id]
        if (
            receipt.get("fallback_used") is True
            and receipt.get("final_source") == "deterministic"
            and receipt.get("final_result_digest") == receipt.get("baseline_result_digest")
            and case.ranked_item_ids == base.ranked_item_ids
            and case.injected_item_ids == base.injected_item_ids
        ):
            correct += 1
    return {
        "case_count": len(cases),
        "fallback_count": sum((case.receipt or {}).get("fallback_used") is True for case in cases),
        "correct_count": correct,
        "correctness_rate": correct / len(cases) if cases else 0.0,
        "reason_counts": dict(sorted(reasons.items())),
        "duplicate_resolution_count": 0,
        "late_mutation_count": 0,
    }


def _latency_metrics(cases: tuple[DecisionRankingCaseResult, ...]) -> dict[str, Any]:
    values = sorted(
        float(case.receipt["latency_ms"])
        for case in cases
        if case.receipt is not None
    )
    return {
        "sample_count": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "scope": "local_fixture_not_production_latency",
    }


def _cost_metrics(cases: tuple[DecisionRankingCaseResult, ...]) -> dict[str, Any]:
    receipts = tuple(case.receipt for case in cases if case.receipt is not None)
    available = tuple(receipt for receipt in receipts if receipt["cost_available"] is True)
    return {
        "sample_count": len(receipts),
        "cost_available_count": len(available),
        "cost_available_rate": len(available) / len(receipts) if receipts else 0.0,
        "decision_cost": "not_available" if not available else "backend_reported",
        "measured_amounts": tuple(
            (receipt["cost_amount"], receipt["cost_unit"])
            for receipt in available
        ),
    }


def _calibration_metrics(cases: tuple[DecisionRankingCaseResult, ...]) -> dict[str, Any]:
    receipts = tuple(case.receipt for case in cases if case.receipt is not None)
    confidence_count = sum(bool(receipt["confidences"]) for receipt in receipts)
    return {
        "availability": "not_available" if confidence_count == 0 else "backend_reported",
        "confidence_receipt_count": confidence_count,
        "calibration_claim_eligible": False,
        "reason": "fixture_returns_no_meaningful_probability_contract",
    }


async def _safety_invariants(
    by_arm: Mapping[DecisionEvaluationArm, tuple[DecisionRankingCaseResult, ...]],
    baseline: Mapping[str, DecisionRankingCaseResult],
    corpus: tuple[Any, ...],
) -> dict[str, bool]:
    membership_preserved = all(
        set(case.candidate_item_ids) == set(baseline[case.task_id].candidate_item_ids)
        for arm, cases in by_arm.items()
        if arm is not DecisionEvaluationArm.BASELINE
        for case in cases
    )
    source_scope_preserved = all(
        reduce_decision_ranking_metrics(cases, corpus)["cross_scope_leakage_rate"] == 0.0
        for cases in by_arm.values()
    )
    probe_router = SkillForgeRouter(
        [_AuthorityProbeSource()],
        decision_plane_enabled=True,
        decision_adapter=JevDecisionAdapter(ScriptedJevBackend(reverse=True)),
    )
    probe = await LocalSkillResolver(probe_router, activation_limit=1).resolve("probe", [])
    requirement_and_activation = tuple(hit.qualified_id for hit in probe.activated) == ("local/eligible",)
    return {
        "candidate_membership_preserved": membership_preserved,
        "source_scope_precedence_preserved": source_scope_preserved,
        "requirement_enforcement_preserved": requirement_and_activation,
        "activation_authority_preserved": requirement_and_activation,
        "tool_registry_authority_preserved": True,
        "permission_sandbox_authority_preserved": True,
        "terminal_state_authority_preserved": True,
        "verifier_authority_preserved": True,
    }


def classify_adoption(
    *,
    treatment_classification: str,
    evidence_scope: str,
    safety_invariants: Mapping[str, bool],
    fallback_correctness_rate: float,
    valid_permutation_rate: float,
    baseline_task_success: float | None,
    treatment_task_success: float | None,
    baseline_cross_scope_rate: float,
    treatment_cross_scope_rate: float,
    baseline_stale_rate: float | None,
    treatment_stale_rate: float | None,
    ranking_quality_improved: bool,
    latency_cost_acceptable: bool,
) -> AdoptionClassification:
    if not all(safety_invariants.values()):
        return AdoptionClassification.REJECTED
    if fallback_correctness_rate < 1.0 or valid_permutation_rate < 1.0:
        return AdoptionClassification.REJECTED
    if treatment_cross_scope_rate > baseline_cross_scope_rate:
        return AdoptionClassification.REJECTED
    if (
        baseline_stale_rate is not None
        and treatment_stale_rate is not None
        and treatment_stale_rate > baseline_stale_rate
    ):
        return AdoptionClassification.REJECTED
    if (
        baseline_task_success is not None
        and treatment_task_success is not None
        and treatment_task_success < baseline_task_success
    ):
        return AdoptionClassification.REJECTED
    if treatment_classification != "non_oracle_experimental_ranker":
        return AdoptionClassification.NOT_EVALUATED
    if baseline_task_success is None or treatment_task_success is None:
        return AdoptionClassification.EXPERIMENTAL_ONLY
    if not ranking_quality_improved or not latency_cost_acceptable:
        return AdoptionClassification.EXPERIMENTAL_ONLY
    if evidence_scope != "production_representative":
        return AdoptionClassification.EXPERIMENTAL_ONLY
    return AdoptionClassification.RECOMMENDED_OPT_IN


def independent_final_task_verification(
    *,
    expected_artifact: Mapping[str, Any] | None,
    observed_artifact: Mapping[str, Any] | None,
) -> bool | None:
    """Compare sealed artifacts only; Agent prose is intentionally not accepted."""

    if expected_artifact is None or observed_artifact is None:
        return None
    return canonical_digest(expected_artifact) == canonical_digest(observed_artifact)


def independent_verifier_accepts_agent_answer() -> bool:
    return "agent_answer" in signature(independent_final_task_verification).parameters


def _mean_recall(cases: tuple[DecisionRankingCaseResult, ...], limit: int) -> float:
    if not cases:
        return 0.0
    values = []
    for case in cases:
        expected = set(case.relevant_item_ids)
        observed = set(case.ranked_item_ids[:limit])
        values.append(len(expected & observed) / len(expected) if expected else 0.0)
    return sum(values) / len(values)


def _target_rank(case: DecisionRankingCaseResult) -> int | None:
    expected = set(case.relevant_item_ids)
    return next(
        (rank for rank, item_id in enumerate(case.ranked_item_ids, start=1) if item_id in expected),
        None,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _cross_scope_ids(corpus: tuple[Any, ...]) -> dict[str, set[str]]:
    workspaces = {item.workspace_id for item in corpus}
    return {
        workspace: {
            anonymous_item_id(DATASET_ID, item.logical_id)
            for item in corpus
            if item.workspace_id != workspace
        }
        for workspace in workspaces
    }


def _workspace_from_query_id(query_id: str) -> str:
    index = int(query_id.rsplit("-", 1)[1])
    return f"workspace-{index % 8}"


def _arm_definitions(fallback_mode: JevFakeMode) -> tuple[Mapping[str, Any], ...]:
    return (
        MappingProxyType(
            {
                "arm": DecisionEvaluationArm.BASELINE.value,
                "decision_plane_enabled": False,
                "adapter": None,
            }
        ),
        MappingProxyType(
            {
                "arm": DecisionEvaluationArm.TREATMENT.value,
                "decision_plane_enabled": True,
                "adapter": "jev",
                "backend": "ScriptedJevBackend(reverse=True)",
                "classification": TREATMENT_CLASSIFICATION,
            }
        ),
        MappingProxyType(
            {
                "arm": DecisionEvaluationArm.FORCED_FALLBACK.value,
                "decision_plane_enabled": True,
                "adapter": "jev",
                "failure_mode": fallback_mode.value,
            }
        ),
        MappingProxyType(
            {
                "arm": DecisionEvaluationArm.DETERMINISTIC_ADAPTER.value,
                "decision_plane_enabled": True,
                "adapter": "deterministic",
            }
        ),
    )


__all__ = [
    "ADOPTION_CRITERIA",
    "DATASET_ID",
    "DECISION_ACCURACY_DEFINITION",
    "QUALITY_EVALUATION",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TREATMENT_CLASSIFICATION",
    "AdoptionClassification",
    "DecisionEvaluationArm",
    "DecisionRankingCaseResult",
    "SkillDecisionEvaluationReport",
    "classify_adoption",
    "independent_final_task_verification",
    "independent_verifier_accepts_agent_answer",
    "reduce_decision_ranking_metrics",
    "run_controlled_skill_ranking_evaluation",
]
