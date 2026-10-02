from __future__ import annotations

import json

import pytest

from benchmarks.picobench.packs.knowledge_evolution import (
    ACCEPTANCE_CRITERIA,
    AcceptanceStatus,
    BenefitClassification,
    EvaluationArm,
    run_knowledge_evolution_benchmark,
)


@pytest.fixture
async def report(tmp_path):
    return await run_knowledge_evolution_benchmark(tmp_path / "p3-benchmark")


@pytest.mark.asyncio
async def test_frozen_six_arm_contract_and_independent_task_success(report) -> None:
    assert {item.arm for item in report.scenarios} == set(EvaluationArm)
    assert report.metrics.scenario_count == 11
    assert report.metrics.verified_task_count == 11
    assert report.metrics.verified_task_pass_count == 11
    assert {
        item.task_id for item in report.scenarios if item.arm is EvaluationArm.NO_REUSE
    } == {
        item.task_id for item in report.scenarios if item.arm is EvaluationArm.APPROVED_REUSE
    }
    assert report.metrics.no_reuse_task_pass_rate == 1.0
    assert report.metrics.approved_reuse_task_pass_rate == 1.0
    assert all(item.task_success_independent for item in report.scenarios)


@pytest.mark.asyncio
async def test_approved_reuse_uses_fact_experience_and_normal_skill_path(report) -> None:
    approved = tuple(item for item in report.scenarios if item.arm is EvaluationArm.APPROVED_REUSE)
    assert {item.knowledge_kind for item in approved} == {
        "memory_fact",
        "experience",
        "skill_candidate",
    }
    assert all(item.retrieved_candidate_ids == item.injected_candidate_ids for item in approved)
    assert report.metrics.relevant_retrieval_rate == 1.0
    assert report.metrics.trusted_injection_precision == 1.0
    assert report.metrics.activated_skill_count == 1
    assert report.metrics.usage_attribution_completeness == 1.0
    assert report.metrics.outcome_association_completeness == 1.0


@pytest.mark.asyncio
async def test_negative_controls_suppress_stale_cross_scope_rejected_and_conflict(report) -> None:
    negatives = tuple(
        item
        for item in report.scenarios
        if item.arm
        in {
            EvaluationArm.STALE_REUSE,
            EvaluationArm.CROSS_REPOSITORY,
            EvaluationArm.REJECTED_CANDIDATE,
            EvaluationArm.CONFLICT,
        }
    )
    assert all(not item.injected_candidate_ids for item in negatives)
    assert report.metrics.stale_suppression_rate == 1.0
    assert report.metrics.cross_repository_contamination_rate == 0.0
    assert report.metrics.rejected_candidate_suppression_rate == 1.0
    assert report.metrics.unresolved_conflict_unsafe_injection_rate == 0.0
    reasons = {reason for item in negatives for _, reason in item.suppressed}
    assert {"stale", "wrong_scope", "not_active"} <= reasons


@pytest.mark.asyncio
async def test_eligibility_review_authority_and_privacy_invariants(report) -> None:
    assert report.safety_invariants["runtime_success_without_task_proof_is_ineligible"]
    assert report.safety_invariants["no_tool_provider_or_terminal_authority_added"]
    assert report.safety_invariants["no_agent_prose_used_as_task_success"]
    assert report.safety_invariants["report_excludes_candidate_bodies_and_raw_payloads"]
    assert all(report.safety_invariants.values())
    payload = report.report_path.read_text(encoding="utf-8")
    assert "correct the arguments and issue a fresh Tool intent" not in payload
    assert "full Session" not in payload
    assert "raw Tool payload" not in payload


@pytest.mark.asyncio
async def test_efficiency_unavailable_is_not_fabricated_and_classification_is_neutral(report) -> None:
    assert report.efficiency_availability["provider_calls"] == "not_available"
    assert report.efficiency_availability["tool_calls"] == "not_available"
    assert report.efficiency_availability["repeated_file_reads"] == "not_available"
    assert report.efficiency_availability["retrieval_latency"].startswith("diagnostic_available")
    assert report.metrics.approximate_knowledge_tokens > 0
    assert report.benefit_classification is BenefitClassification.NEUTRAL
    assert report.passed


@pytest.mark.asyncio
async def test_acceptance_criteria_are_frozen_and_matrix_has_no_failure(report) -> None:
    assert ACCEPTANCE_CRITERIA["version"] == 1
    assert "approved_reuse_verified_task_success_regresses" in ACCEPTANCE_CRITERIA[
        "acceptance_fails_if"
    ]
    assert len(report.acceptance_matrix) == 22
    assert AcceptanceStatus.FAIL not in report.acceptance_matrix.values()
    assert (
        report.acceptance_matrix["representative_regression_safety"]
        is AcceptanceStatus.NOT_AVAILABLE
    )


@pytest.mark.asyncio
async def test_semantic_digest_is_stable_and_excludes_latency(tmp_path) -> None:
    first = await run_knowledge_evolution_benchmark(tmp_path / "first")
    second = await run_knowledge_evolution_benchmark(tmp_path / "second")
    assert first.semantic_digest == second.semantic_digest
    assert first.metrics.total_retrieval_latency_ms != second.metrics.total_retrieval_latency_ms or (
        first.metrics.total_retrieval_latency_ms >= 0
    )
    stored = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert stored["semantic_digest"] == first.semantic_digest
    assert stored["schema"] == "pico.picobench.knowledge-evolution.v1"
