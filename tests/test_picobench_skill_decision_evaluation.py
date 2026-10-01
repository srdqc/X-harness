from __future__ import annotations

import asyncio
import json
from dataclasses import fields

import pytest

from benchmarks.picobench.packs.memory_skill import (
    AdoptionClassification,
    DecisionEvaluationArm,
    classify_adoption,
    independent_final_task_verification,
    independent_verifier_accepts_agent_answer,
    reduce_decision_ranking_metrics,
    run_controlled_skill_ranking_evaluation,
)
from benchmarks.picobench.packs.memory_skill.decision_ranking import (
    DECISION_ACCURACY_DEFINITION,
    QUALITY_EVALUATION,
    TREATMENT_CLASSIFICATION,
    DecisionRankingCaseResult,
)
from pico.decision_plane import DecisionCandidate, DecisionRequest, DecisionType
from pico.decision_plane.fake import JevFakeMode, ScriptedJevBackend


@pytest.fixture(scope="module")
def report():
    return asyncio.run(run_controlled_skill_ranking_evaluation())


def _case(
    task_id: str,
    relevant: tuple[str, ...],
    ranked: tuple[str, ...],
) -> DecisionRankingCaseResult:
    return DecisionRankingCaseResult(
        task_id=task_id,
        label="positive",
        arm=DecisionEvaluationArm.BASELINE,
        relevant_item_ids=relevant,
        candidate_item_ids=ranked,
        ranked_item_ids=ranked,
        injected_item_ids=ranked,
        ranking_digest="fixture-digest",
        receipt=None,
    )


def _classification(**overrides: object) -> AdoptionClassification:
    values: dict[str, object] = {
        "treatment_classification": "non_oracle_experimental_ranker",
        "evidence_scope": "production_representative",
        "safety_invariants": {"authority": True},
        "fallback_correctness_rate": 1.0,
        "valid_permutation_rate": 1.0,
        "baseline_task_success": 1.0,
        "treatment_task_success": 1.0,
        "baseline_cross_scope_rate": 0.0,
        "treatment_cross_scope_rate": 0.0,
        "baseline_stale_rate": 0.0,
        "treatment_stale_rate": 0.0,
        "ranking_quality_improved": True,
        "latency_cost_acceptable": True,
    }
    values.update(overrides)
    return classify_adoption(**values)  # type: ignore[arg-type]


def test_frozen_dataset_arms_and_privacy_bounded_artifact(report) -> None:
    assert report.schema == "pico.picobench.skill-decision-ranking.v1"
    assert report.schema_version == 1
    assert report.scenario_count == 68
    assert report.dataset_identity["positive_cases"] == 48
    assert report.dataset_identity["hard_negative_cases"] == 20
    assert report.dataset_identity["exact_match_cases"] == 40
    assert report.dataset_identity["ambiguous_cases"] == 8
    assert report.dataset_identity["stale_skill_cases_available"] is False
    assert {definition["arm"] for definition in report.arm_definitions} == {
        arm.value for arm in DecisionEvaluationArm
    }
    counts = {
        arm: sum(case.arm.value == arm for case in report.case_results)
        for arm in (item.value for item in DecisionEvaluationArm)
    }
    assert counts == {arm.value: 68 for arm in DecisionEvaluationArm}
    persisted = json.dumps(report.to_record(), sort_keys=True)
    assert "ambertoken" not in persisted
    assert "skilltoken" not in persisted
    assert "query_text" not in persisted


def test_explicit_ranking_metrics_are_reduced_correctly() -> None:
    metrics = reduce_decision_ranking_metrics(
        (
            _case("skill-positive-000", ("a",), ("b", "a", "c")),
            _case("skill-positive-001", ("d", "e"), ("d", "x", "e")),
        ),
        (),
    )
    assert metrics["decision_accuracy_definition"] == DECISION_ACCURACY_DEFINITION
    assert metrics["decision_accuracy"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.25)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert metrics["recall_at_5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["target_rank_distribution"] == {"1": 1, "2": 1}
    assert metrics["irrelevant_retrieval_rate"] == pytest.approx(0.5)
    assert metrics["stale_candidate_promotion_rate"] is None
    assert metrics["stale_candidate_metric_available"] is False
    assert metrics["cross_scope_leakage_rate"] == 0.0


def test_baseline_treatment_and_deterministic_adapter_contracts(report) -> None:
    by_arm = {
        arm: {
            case.task_id: case
            for case in report.case_results
            if case.arm is arm
        }
        for arm in DecisionEvaluationArm
    }
    baseline = by_arm[DecisionEvaluationArm.BASELINE]
    treatment = by_arm[DecisionEvaluationArm.TREATMENT]
    deterministic = by_arm[DecisionEvaluationArm.DETERMINISTIC_ADAPTER]
    assert all(case.receipt is None for case in baseline.values())
    assert all(case.receipt and not case.receipt["fallback_used"] for case in treatment.values())
    assert all(
        set(treatment[task_id].ranked_item_ids) == set(base.ranked_item_ids)
        for task_id, base in baseline.items()
    )
    assert any(
        treatment[task_id].ranked_item_ids != base.ranked_item_ids
        for task_id, base in baseline.items()
    )
    assert all(
        deterministic[task_id].ranked_item_ids == base.ranked_item_ids
        and deterministic[task_id].injected_item_ids == base.injected_item_ids
        for task_id, base in baseline.items()
    )
    assert report.arm_metrics["treatment"]["valid_permutation_rate"] == 1.0
    assert report.arm_metrics["baseline"]["cross_scope_leakage_rate"] == 0.0


def test_treatment_is_non_oracle_contract_fixture(report) -> None:
    request_fields = {field.name for field in fields(DecisionRequest)}
    assert not request_fields.intersection(
        {"expected_item_ids", "gold_label", "verifier_outcome", "task_success"}
    )
    request = DecisionRequest(
        decision_type=DecisionType.SKILL_RANKING,
        query="visible query",
        candidates=(
            DecisionCandidate("local/a", "a", "visible a", "local", 1, 1.0),
            DecisionCandidate("local/b", "b", "visible b", "local", 2, 0.5),
        ),
        decision_id="decision:oracle-isolation",
    )
    result = asyncio.run(ScriptedJevBackend(reverse=True).rank_skill_candidates(request))
    assert [item.candidate_id for item in result.ranking] == ["local/b", "local/a"]
    assert TREATMENT_CLASSIFICATION == "harness_contract_fixture"
    assert QUALITY_EVALUATION == "not_available"
    assert report.configuration_identity["quality_evaluation"] == "not_available"


def test_receipt_backed_fallback_is_exact_and_nonduplicating(report) -> None:
    baseline = {
        case.task_id: case
        for case in report.case_results
        if case.arm is DecisionEvaluationArm.BASELINE
    }
    fallback = tuple(
        case
        for case in report.case_results
        if case.arm is DecisionEvaluationArm.FORCED_FALLBACK
    )
    assert report.fallback_metrics == {
        "case_count": 68,
        "fallback_count": 68,
        "correct_count": 68,
        "correctness_rate": 1.0,
        "reason_counts": {"backend_unavailable": 68},
        "duplicate_resolution_count": 0,
        "late_mutation_count": 0,
    }
    for case in fallback:
        receipt = case.receipt
        assert receipt is not None
        assert receipt["fallback_used"] is True
        assert receipt["final_source"] == "deterministic"
        assert receipt["final_result_digest"] == receipt["baseline_result_digest"]
        assert case.ranked_item_ids == baseline[case.task_id].ranked_item_ids
        assert case.injected_item_ids == baseline[case.task_id].injected_item_ids


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        (JevFakeMode.TIMEOUT, "backend_timeout"),
        (JevFakeMode.MALFORMED, "malformed_response"),
    ],
)
def test_timeout_and_malformed_results_fall_back_exactly(mode, reason) -> None:
    report = asyncio.run(run_controlled_skill_ranking_evaluation(fallback_mode=mode))
    assert report.fallback_metrics["correctness_rate"] == 1.0
    assert report.fallback_metrics["reason_counts"] == {reason: 68}
    assert report.fallback_metrics["duplicate_resolution_count"] == 0
    assert report.fallback_metrics["late_mutation_count"] == 0


def test_latency_cost_calibration_and_final_task_contract(report) -> None:
    assert report.latency_diagnostics["baseline"]["sample_count"] == 0
    for arm in ("treatment", "forced_fallback", "deterministic_adapter"):
        latency = report.latency_diagnostics[arm]
        assert latency["sample_count"] == 68
        assert latency["p50_ms"] is not None
        assert latency["p95_ms"] >= latency["p50_ms"]
        assert latency["scope"] == "local_fixture_not_production_latency"
        assert report.cost_evidence[arm]["decision_cost"] == "not_available"
        assert report.cost_evidence[arm]["cost_available_rate"] == 0.0
        assert report.calibration_evidence[arm]["availability"] == "not_available"
    assert report.final_task_success["availability"] == "not_available"
    assert report.final_task_success["agent_final_answer_used_as_proof"] is False
    assert independent_final_task_verification(
        expected_artifact={"sealed": [1, 2]},
        observed_artifact={"sealed": [1, 2]},
    ) is True
    assert independent_final_task_verification(
        expected_artifact={"sealed": [1, 2]},
        observed_artifact={"sealed": [2, 1]},
    ) is False
    assert independent_final_task_verification(
        expected_artifact=None,
        observed_artifact={"sealed": [1, 2]},
    ) is None
    assert independent_verifier_accepts_agent_answer() is False


def test_safety_and_adoption_gates_are_conservative(report) -> None:
    assert all(report.safety_invariants.values())
    assert report.adoption_classification is AdoptionClassification.NOT_EVALUATED
    assert _classification(safety_invariants={"authority": False}) is AdoptionClassification.REJECTED
    assert _classification(treatment_task_success=0.9) is AdoptionClassification.REJECTED
    assert _classification(fallback_correctness_rate=0.99) is AdoptionClassification.REJECTED
    assert _classification(
        treatment_classification="harness_contract_fixture"
    ) is AdoptionClassification.NOT_EVALUATED
    assert _classification(evidence_scope="local_contract_fixture") is AdoptionClassification.EXPERIMENTAL_ONLY


def test_semantic_digest_and_structural_results_repeat(report) -> None:
    repeated = asyncio.run(run_controlled_skill_ranking_evaluation())
    assert repeated.semantic_digest == report.semantic_digest
    assert repeated.arm_metrics == report.arm_metrics
    assert repeated.fallback_metrics == report.fallback_metrics
    assert repeated.cost_evidence == report.cost_evidence
    assert repeated.calibration_evidence == report.calibration_evidence
    assert repeated.final_task_success == report.final_task_success
    assert repeated.safety_invariants == report.safety_invariants
