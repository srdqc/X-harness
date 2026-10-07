from __future__ import annotations

import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live.artifacts import (
    completed_run_ids,
    freeze_manifest,
    load_manifest,
    load_run,
    write_run,
)
from benchmarks.picobench.packs.knowledge_evolution_live.knowledge import (
    CORPUS,
    corpus_digest,
    prepare_approved_corpus,
)
from benchmarks.picobench.packs.knowledge_evolution_live.metrics import (
    available,
    extract_run_metrics,
    repeated_repository_reads,
    unavailable,
)
from benchmarks.picobench.packs.knowledge_evolution_live.prepare import (
    _assert_replication_frozen,
    preflight,
)
from benchmarks.picobench.packs.knowledge_evolution_live.protocol import (
    REPLICATION_QUESTIONS,
    arm_configuration_digest,
    arm_configuration_payload,
    arm_order,
    assert_fair_plan,
    create_manifest,
    fairness_digest,
)
from benchmarks.picobench.packs.knowledge_evolution_live.reducer import classify, reduce_campaign
from benchmarks.picobench.packs.knowledge_evolution_live.runner import (
    execute_one,
    missing_runs,
    run_campaign,
    run_replacement,
    schedule_replacement,
)
from benchmarks.picobench.packs.knowledge_evolution_live.schema import (
    Arm,
    Availability,
    BenefitClassification,
    CampaignMode,
    CampaignPaths,
    InfraInvalidReason,
    MetricValue,
    RunMetrics,
    RunRecord,
    RuntimeBudget,
    RunValidity,
    TokenAccountingStatus,
)
from benchmarks.picobench.packs.knowledge_evolution_live.solvability import (
    SolvabilityAudit,
    audit_pilot_solvability,
)
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import PILOT_TASK_IDS, TASKS
from benchmarks.picobench.packs.knowledge_evolution_live.validity import classify_run_validity
from benchmarks.picobench.packs.knowledge_evolution_live.verifiers import (
    VERIFIERS,
    HiddenVerifierSpec,
    changed_path_findings,
    integration_proof_findings,
    verify_workspace,
)
from pico.agent.tools.registry import normalize_repository_read_path
from pico.knowledge_evolution import (
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeRelevanceSelection,
    KnowledgeRetrievalReceipt,
    KnowledgeSelectionMode,
    KnowledgeUsageMode,
    KnowledgeUsageReceipt,
    RelevanceDecision,
    RelevanceReason,
)
from pico.tracing import evidence
from pico.tracing.store import TraceStore


def _manifest(
    mode: CampaignMode = CampaignMode.PILOT,
    *,
    base_sha: str = "a" * 40,
    pilot_repetition: int = 1,
):
    return create_manifest(
        mode=mode,
        base_commit_sha=base_sha,
        seed=17,
        provider_id="fixture",
        actual_model_id="fixture/model",
        provider_model_config_digest="b" * 64,
        tool_config_digest="c" * 64,
        runtime_config_digest="d" * 64,
        created_at="2026-10-02T00:00:00Z",
        pilot_repetition=pilot_repetition,
    )


def _metrics(value: int, *, available_reads: bool = True) -> RunMetrics:
    read = available(value) if available_reads else unavailable()
    return RunMetrics(
        provider_logical_calls=available(value),
        provider_attempts=available(value),
        tool_calls_total=available(value),
        unique_repo_files_read=read,
        total_repo_file_reads=read,
        repeated_repo_file_reads=read,
        skill_candidates_retrieved=available(0),
        skills_referenced=available(0),
        skills_activated=available(0),
        repeated_skill_reads=unavailable(),
        input_tokens=available(value * 100),
        output_tokens=available(value * 10),
        cached_tokens=unavailable(),
        turn_latency_ms=available(value * 1000),
        provider_retries=available(0),
        failed_tool_attempts=available(0),
        validation_failures=available(0),
        p3_approximate_context_tokens=unavailable(),
    )


def _record(manifest, planned, value: int, outcome: str = "pass", **overrides) -> RunRecord:
    values = dict(
        campaign_id=manifest.campaign_id,
        task_id=planned.task_id,
        arm=planned.arm,
        repetition=planned.repetition,
        run_id=planned.run_id,
        workspace_identity=f"workspace:{planned.run_id}",
        turn_id=f"turn:{planned.run_id}",
        provider_id=manifest.provider_id,
        actual_model_id=manifest.actual_model_id,
        started_at="2026-10-02T00:00:00Z",
        terminal_at="2026-10-02T00:00:01Z",
        runtime_outcome="completed",
        verifier_outcome=outcome,
        safety_outcome="pass",
        safety_findings=(),
        task_success_evidence_ref=f"success:{planned.run_id}",
        knowledge_retrieval_refs=(),
        knowledge_usage_refs=(),
        outcome_association_refs=(),
        retrieved_candidate_ids=(),
        injected_candidate_ids=(),
        referenced_candidate_ids=(),
        activated_candidate_ids=(),
        metrics=_metrics(value),
        patch_digest="e" * 64,
        patch_artifact_ref=f"patches/{planned.run_id}.json",
        trace_evidence_refs=(f"trace:{planned.run_id}",),
        fairness_digest=fairness_digest(manifest, planned.task_id, planned.repetition),
    )
    values.update(overrides)
    return RunRecord.create(**values)


def test_frozen_task_manifest_has_exact_category_balance_and_no_oracle_labels() -> None:
    assert len(TASKS) == len({item.task_id for item in TASKS}) == 12
    categories = {category: 0 for category in {item.category for item in TASKS}}
    for task in TASKS:
        categories[task.category] += 1
        assert task.verifier_id in VERIFIERS
        assert task.verifier_digest == VERIFIERS[task.verifier_id].digest
        assert task.task_id not in task.prompt
        assert "NO_REUSE" not in task.prompt and "APPROVED_REUSE" not in task.prompt
    assert sorted(categories.values()) == [3, 3, 3, 3]
    assert len(PILOT_TASK_IDS) == 3


@pytest.mark.parametrize(
    ("mode", "runs"),
    [(CampaignMode.PILOT, 6), (CampaignMode.OFFICIAL_SINGLE, 24), (CampaignMode.OFFICIAL_REPEAT2, 48)],
)
def test_campaign_manifest_is_frozen_balanced_and_round_trips(tmp_path: Path, mode, runs) -> None:
    manifest = _manifest(mode)
    assert len(manifest.planned_runs) == runs
    assert_fair_plan(manifest)
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)
    assert load_manifest(paths.manifest) == manifest
    freeze_manifest(paths, manifest)
    with pytest.raises(Exception):
        freeze_manifest(paths, replace(manifest, campaign_seed=18))


def test_arm_order_is_deterministic_balanced_and_reversed_for_repeat() -> None:
    first = [arm_order(task_id=item.task_id, repetition=1, seed=9) for item in TASKS]
    assert first == [arm_order(task_id=item.task_id, repetition=1, seed=9) for item in TASKS]
    assert {order[0] for order in first} == {Arm.NO_REUSE, Arm.APPROVED_REUSE}
    for task, order in zip(TASKS, first, strict=True):
        assert arm_order(task_id=task.task_id, repetition=2, seed=9) == tuple(reversed(order))


def test_pilot_repetition_two_reverses_order_and_freezes_questions() -> None:
    first = _manifest(pilot_repetition=1)
    second = _manifest(pilot_repetition=2)
    assert second.campaign_id != first.campaign_id
    assert second.replication_questions == REPLICATION_QUESTIONS
    assert {item.repetition for item in second.planned_runs} == {2}
    for task_id in second.task_ids:
        first_order = tuple(item.arm for item in first.planned_runs if item.task_id == task_id)
        second_order = tuple(item.arm for item in second.planned_runs if item.task_id == task_id)
        assert second_order == tuple(reversed(first_order))


def test_p3r3_three_arm_manifest_is_deterministic_and_isolated() -> None:
    first = _manifest(CampaignMode.P3R3_EXPLORATORY)
    second = _manifest(CampaignMode.P3R3_EXPLORATORY)
    assert first == second
    assert len(first.planned_runs) == 9
    expected = {
        Arm.NO_REUSE,
        Arm.APPROVED_REUSE_LEGACY,
        Arm.APPROVED_REUSE_SELECTIVE,
    }
    for task_id in first.task_ids:
        planned = tuple(item for item in first.planned_runs if item.task_id == task_id)
        assert {item.arm for item in planned} == expected
        assert len({arm_configuration_digest(first, item) for item in planned}) == 3
        common = []
        for item in planned:
            payload = arm_configuration_payload(first, item)
            payload.pop("p3_reuse_available")
            payload.pop("knowledge_selection_mode")
            common.append(payload)
        assert common[0] == common[1] == common[2]
    assert first.replication_questions != REPLICATION_QUESTIONS
    assert_fair_plan(first)


def test_p3r3_reducer_reports_three_arm_relevance_funnel_without_reinterpreting_pairs() -> None:
    manifest = _manifest(CampaignMode.P3R3_EXPLORATORY)
    all_candidates = tuple(f"candidate-{index}" for index in range(6))
    selective_by_task = {
        "p3r-nav-01": ("candidate-0",),
        "p3r-debug-01": ("candidate-4",),
        "p3r-int-01": ("candidate-2", "candidate-1"),
    }
    records = []
    for planned in manifest.planned_runs:
        selected = (
            ()
            if planned.arm is Arm.NO_REUSE
            else (
                all_candidates
                if planned.arm is Arm.APPROVED_REUSE_LEGACY
                else selective_by_task[planned.task_id]
            )
        )
        abstained = tuple(item for item in all_candidates if item not in selected)
        records.append(
            _record(
                manifest,
                planned,
                1,
                retrieved_candidate_ids=selected,
                injected_candidate_ids=selected,
                relevance_selected_candidate_ids=selected,
                relevance_abstained_candidate_ids=abstained,
                relevance_abstention_reason_counts=(
                    (("abstain_no_meaningful_overlap", len(abstained)),)
                    if abstained
                    else ()
                ),
            )
        )
    result = reduce_campaign(manifest, tuple(records))
    assert result["schema"] == "pico.picobench.p3r3-exploratory-reduction.v1"
    assert result["claim_eligible"] is False
    assert result["fairness_valid"] is True
    assert len(result["comparisons"]) == 3
    selective = result["candidate_set_overlap"][Arm.APPROVED_REUSE_SELECTIVE.value]
    assert selective["selected"]["distinct_set_count"] == 3
    assert result["abstention_reason_counts"]["abstain_no_meaningful_overlap"] > 0


def test_replication_lock_preserves_pilot_inputs_except_verifier_version() -> None:
    first = _manifest(pilot_repetition=1)
    second = _manifest(pilot_repetition=2)
    _assert_replication_frozen(first, second)
    with pytest.raises(ValueError, match="tool_config_digest"):
        _assert_replication_frozen(first, replace(second, tool_config_digest="f" * 64))
    assert second.knowledge_corpus_digest == first.knowledge_corpus_digest
    assert second.task_prompt_digests == first.task_prompt_digests


def test_fairness_digest_excludes_only_arm_and_detects_material_drift() -> None:
    manifest = _manifest()
    planned = manifest.planned_runs[0]
    assert fairness_digest(manifest, planned.task_id, planned.repetition) == fairness_digest(
        manifest, planned.task_id, planned.repetition
    )
    changed = replace(manifest, tool_config_digest="f" * 64)
    assert fairness_digest(changed, planned.task_id, planned.repetition) != fairness_digest(
        manifest, planned.task_id, planned.repetition
    )


def test_preflight_creates_and_removes_isolated_worktree_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "p3r@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "P3R"], cwd=repository, check=True)
    (repository / "marker.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    monkeypatch.setattr(
        "benchmarks.picobench.packs.knowledge_evolution_live.prepare._provider_config_present",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "benchmarks.picobench.packs.knowledge_evolution_live.prepare.prepare_approved_corpus",
        lambda **_kwargs: ("candidate",),
    )
    monkeypatch.setattr(
        "benchmarks.picobench.packs.knowledge_evolution_live.prepare.audit_pilot_solvability",
        lambda _repository: (
            SolvabilityAudit("p3r-debug-01", True, ()),
            SolvabilityAudit("p3r-int-01", True, ()),
        ),
    )
    result = preflight(
        repository=repository,
        base_commit_sha=sha,
        reviewer_id="human:test",
        config=object(),
        manifest=_manifest(base_sha=sha),
    )
    assert result.worktree_supported is True
    assert result.corpus_candidate_count == 1
    assert subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=repository, text=True).count("worktree ") == 1


def test_frozen_corpus_uses_public_reviewed_p3_path(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "p3r@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "P3R"], cwd=repository, check=True)
    (repository / "marker.txt").write_text("scope", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "scope"], cwd=repository, check=True)
    short_parent = Path.cwd() / ".tmp"
    short_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p3rt-", dir=short_parent) as state_text:
        state = Path(state_text)
        candidate_ids = prepare_approved_corpus(
            state_root=state,
            workspace=repository,
            reviewer_id="human:p3r-test",
        )
        store = KnowledgeRecordStore(state)
        assert len(candidate_ids) == len(CORPUS) == 6
        assert all(store.read_candidate(candidate_id) is not None for candidate_id in candidate_ids)
        assert all(
            len(store.list_materialization_results(candidate_id)) == 1
            for candidate_id in candidate_ids
        )


def test_repeated_read_calculation_uses_normalized_repository_paths() -> None:
    assert repeated_repository_reads(("pico/a.py", "pico\\a.py", "PICO/a.py", "pico/b.py")) == (2, 4, 2)


def test_structured_metric_extraction_counts_calls_attempts_tools_reads_usage(tmp_path: Path) -> None:
    turn_id = "turn-p3r"
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="p3r:test",
        trace_id="trace-p3r",
        root_span_id="span-p3r",
        writer=TraceStore(tmp_path).append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    for ordinal in (1, 2):
        correlations = {"logical_call_id": "call-1", "attempt_id": f"attempt-{ordinal}", "attempt_ordinal": ordinal}
        recorder.emit(evidence.PROVIDER_ATTEMPT_STARTED, correlations=correlations, metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA, "requested_model": "m", "attempted_model": "m", "provider": "fixture", "request_digest": str(ordinal)})
        recorder.emit(evidence.PROVIDER_ATTEMPT_COMPLETED, correlations=correlations, metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA, "actual_model": "m", "outcome": "success", "response_digest": str(ordinal), "usage_available": True, "usage": {"prompt_tokens": 4, "completion_tokens": 2}, "duration_ms": 1})
    for index, path in enumerate(("pico/a.py", "pico/a.py", "pico/b.py"), start=1):
        correlations = {"receipt_id": f"tool-{index}", "model_call_id": "call-1", "parent_call_id": None}
        recorder.emit(evidence.TOOL_EXECUTION_STARTED, correlations=correlations, metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "requested_name": "read_file", "resolved_name": "read_file", "effect": "read", "argument_digest": str(index), "repository_read_path": path})
        recorder.emit(evidence.TOOL_EXECUTION_COMPLETED, correlations=correlations, metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "outcome": "success", "result_digest": str(index), "result_size": 1, "duration_ms": 1})
    recorder.emit(evidence.TURN_TERMINAL, metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"})
    store = KnowledgeRecordStore(tmp_path)
    usage = KnowledgeUsageReceipt.create(
        usage_id="usage-1", turn_id=turn_id, repository_scope_id="a" * 64,
        candidate_id="candidate-1", candidate_manifest_digest="b" * 64,
        materialization_digest="c" * 64, knowledge_type=CandidateType.EXPERIENCE,
        lifecycle_state="active", applicability_id="app-1", applicability_digest="d" * 64,
        retrieval_id="retrieval-missing", retrieval_rank=1, retrieval_score=1.0,
        usage_mode=KnowledgeUsageMode.REFERENCED, context_identity="context:1",
        created_at="2026-10-02T00:00:00Z",
    )
    store.write_usage(usage)
    metrics, refs = extract_run_metrics(trace_root=tmp_path, turn_id=turn_id, knowledge_state_root=tmp_path)
    assert metrics.provider_logical_calls.value == 1
    assert metrics.provider_attempts.value == 2
    assert metrics.provider_retries.value == 1
    assert metrics.tool_calls_total.value == 3
    assert (metrics.unique_repo_files_read.value, metrics.total_repo_file_reads.value, metrics.repeated_repo_file_reads.value) == (2, 3, 1)
    assert (metrics.input_tokens.value, metrics.output_tokens.value) == (8, 4)
    assert refs["usage_refs"] == ("usage-1",)
    assert refs["referenced_ids"] == ("candidate-1",)
    assert refs["repository_read_paths"] == ("pico/a.py", "pico/a.py", "pico/b.py")


def test_selected_candidate_metrics_preserve_retrieval_order(tmp_path: Path) -> None:
    turn_id = "turn-selection-order"
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="p3r:test",
        trace_id="trace-selection-order",
        root_span_id="span-selection-order",
        writer=TraceStore(tmp_path).append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(evidence.TURN_TERMINAL, metadata={"outcome": "completed"})
    store = KnowledgeRecordStore(tmp_path)
    scope_digest = "a" * 64
    query_digest = "b" * 64
    selector_digest = "c" * 64
    candidates = (
        ("candidate-z", "retrieval-first", "2026-10-07T00:00:00.000001Z"),
        ("candidate-a", "retrieval-second", "2026-10-07T00:00:00.000002Z"),
    )
    for rank, (candidate_id, retrieval_id, created_at) in enumerate(candidates, start=1):
        store.write_retrieval(
            KnowledgeRetrievalReceipt.create(
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                repository_scope_id=scope_digest,
                query_digest=query_digest,
                candidate_set_digest="d" * 64,
                ranked_candidate_ids=(candidate_id,),
                selected_candidate_ids=(candidate_id,),
                suppressed=(),
                applicable_count=1,
                suppressed_count=0,
                latency_ms=1.0,
                created_at=created_at,
                retrieval_policy="fixture",
                retrieval_version=1,
            )
        )
        store.write_relevance_selection(
            KnowledgeRelevanceSelection.create(
                selection_id=f"selection-{rank}",
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                repository_scope_id=scope_digest,
                selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
                query_digest=query_digest,
                candidate_id=candidate_id,
                candidate_type=CandidateType.EXPERIENCE,
                rank=1,
                relevance_score=1.0,
                meaningful_overlap=("validation",),
                identifier_overlap=(),
                decision=RelevanceDecision.SELECT,
                reason=RelevanceReason.SELECT_RELEVANT,
                selector_version=1,
                selector_config_digest=selector_digest,
            )
        )

    _, refs = extract_run_metrics(
        trace_root=tmp_path,
        turn_id=turn_id,
        knowledge_state_root=tmp_path,
    )

    assert refs["relevance_selected_candidate_ids"] == ("candidate-z", "candidate-a")


def test_first_edit_iteration_is_derived_from_durable_tool_receipt(tmp_path: Path) -> None:
    turn_id = "turn-edit"
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="p3r:test",
        trace_id="trace-edit",
        root_span_id="span-edit",
        writer=TraceStore(tmp_path).append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    correlations = {"receipt_id": "edit-1", "model_call_id": "call-1"}
    recorder.emit(
        evidence.TOOL_EXECUTION_STARTED,
        correlations=correlations,
        metadata={
            "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
            "requested_name": "edit_file",
            "resolved_name": "edit_file",
            "effect": "write",
            "argument_digest": "a" * 64,
            "agent_iteration": 3,
        },
    )
    recorder.emit(
        evidence.TOOL_EXECUTION_COMPLETED,
        correlations=correlations,
        metadata={
            "receipt_schema": evidence.TOOL_RECEIPT_SCHEMA,
            "outcome": "success",
            "result_digest": "b" * 64,
        },
    )
    recorder.emit(evidence.TURN_TERMINAL, metadata={"outcome": "completed"})
    metrics, _ = extract_run_metrics(
        trace_root=tmp_path,
        turn_id=turn_id,
        knowledge_state_root=tmp_path,
    )
    assert metrics.first_edit_iteration == available(3)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"valid": True, "control_success_rate": 1.0, "treatment_success_rate": 1.0, "baseline_pass_treatment_fail": 0, "median_relative_deltas": {"tool_calls_total": -0.15}, "safety_passed": True}, BenefitClassification.BENEFICIAL),
        ({"valid": True, "control_success_rate": 1.0, "treatment_success_rate": 1.0, "baseline_pass_treatment_fail": 0, "median_relative_deltas": {"tool_calls_total": -0.10}, "safety_passed": True}, BenefitClassification.NEUTRAL),
        ({"valid": True, "control_success_rate": 1.0, "treatment_success_rate": 0.5, "baseline_pass_treatment_fail": 1, "median_relative_deltas": {}, "safety_passed": True}, BenefitClassification.REGRESSIVE),
        ({"valid": False, "control_success_rate": 1.0, "treatment_success_rate": 1.0, "baseline_pass_treatment_fail": 0, "median_relative_deltas": {}, "safety_passed": True}, BenefitClassification.INVALID),
        ({"valid": True, "control_success_rate": 1.0, "treatment_success_rate": 1.0, "baseline_pass_treatment_fail": 0, "median_relative_deltas": {}, "safety_passed": False}, BenefitClassification.REGRESSIVE),
    ],
)
def test_predeclared_classification(kwargs, expected) -> None:
    assert classify(**kwargs) is expected


def test_repeat2_reducer_uses_per_task_arm_medians_before_pairing() -> None:
    manifest = _manifest(CampaignMode.OFFICIAL_REPEAT2)
    records = []
    for planned in manifest.planned_runs:
        value = 10 if planned.arm is Arm.NO_REUSE else (4 if planned.repetition == 1 else 12)
        records.append(_record(manifest, planned, value))
    result = reduce_campaign(manifest, tuple(records))
    first = result["per_task_arm_median_pairs"][0]["arm_median_deltas"]["tool_calls_total"]
    assert first["absolute"] == -2.0
    assert first["relative"] == -0.2
    assert result["classification"] == BenefitClassification.BENEFICIAL.value


def test_pilot_reducer_is_not_eligible_for_benefit_claim() -> None:
    manifest = _manifest(CampaignMode.PILOT)
    records = tuple(
        _record(manifest, planned, 10 if planned.arm is Arm.NO_REUSE else 5)
        for planned in manifest.planned_runs
    )
    result = reduce_campaign(manifest, records)
    assert result["campaign_label"] == "PILOT / NOT FOR BENEFIT CLAIM"
    assert result["claim_eligible"] is False
    assert result["classification"] == BenefitClassification.NOT_EVALUATED.value
    assert result["diagnostic_classification"] == BenefitClassification.BENEFICIAL.value


def test_reducer_reports_candidate_overlap_and_replication_evidence() -> None:
    manifest = _manifest(pilot_repetition=2)
    records = []
    candidate_ids = tuple(f"candidate-{index}" for index in range(6))
    injected_ids = candidate_ids[:5]
    for planned in manifest.planned_runs:
        treatment = planned.arm is Arm.APPROVED_REUSE
        metrics = replace(
            _metrics(2 if treatment else 1),
            agent_iterations=available(4 if treatment else 2),
            first_edit_iteration=available(3 if treatment else 1),
            skills_referenced=available(1 if treatment else 0),
        )
        records.append(
            _record(
                manifest,
                planned,
                2 if treatment else 1,
                metrics=metrics,
                retrieved_candidate_ids=candidate_ids if treatment else (),
                injected_candidate_ids=injected_ids if treatment else (),
                referenced_candidate_ids=("candidate-0",) if treatment else (),
                repository_read_paths=("pico/common.py", "pico/treatment.py") if treatment else ("pico/common.py",),
                changed_paths=("pico/change.py",),
            )
        )
    result = reduce_campaign(manifest, tuple(records))
    assert result["candidate_set_overlap"]["retrieved"]["identical_pair_count"] == 3
    assert result["candidate_set_overlap"]["injected"]["identical_pair_count"] == 3
    assert all(
        item["jaccard"] == 1.0
        for item in result["candidate_set_overlap"]["retrieved"]["comparisons"]
    )
    nav = result["nav_cost_replication"][0]
    assert nav["deltas"]["first_edit_iteration"]["absolute"] == 2.0
    assert nav["treatment_only_explored_path_count"] == 1
    assert nav["arms"][Arm.APPROVED_REUSE.value]["referenced_skill_ids"] == ["candidate-0"]


def test_resume_uses_only_missing_immutable_run_ids(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)
    first = _record(manifest, manifest.planned_runs[0], 2)
    write_run(paths, first)
    assert completed_run_ids(paths) == {first.run_id}
    assert [item.run_id for item in missing_runs(paths, manifest)] == [
        item.run_id for item in manifest.planned_runs[1:]
    ]
    with pytest.raises(Exception):
        write_run(paths, replace(first, patch_digest="f" * 64))


def test_execute_live_gate_prevents_subprocess_or_provider_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)
    invoked = False

    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("live child process must not start")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(PermissionError, match="--execute-live"):
        run_campaign(
            repository=tmp_path,
            campaign_root=paths.root,
            reviewer_id="human:test",
            execute_live=False,
        )
    assert invoked is False


def test_child_host_process_failure_persists_infra_invalid_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)

    def fail_child(*_args, **_kwargs):
        raise subprocess.CalledProcessError(9, "p3r-child")

    monkeypatch.setattr(subprocess, "run", fail_child)
    with pytest.raises(subprocess.CalledProcessError):
        run_campaign(
            repository=tmp_path,
            campaign_root=paths.root,
            reviewer_id="human:test",
            execute_live=True,
        )
    planned = manifest.planned_runs[0]
    record = load_run(paths.runs / f"{planned.run_id}.json")
    assert record.run_validity is RunValidity.INFRA_INVALID
    assert record.infra_invalid_reason is InfraInvalidReason.BENCHMARK_HOST_FAILURE


def test_canonicalization_preserves_machine_checkable_guard_prefix() -> None:
    from pico.knowledge_evolution.canonicalize import KnowledgeProposal, validate_and_canonicalize
    from pico.knowledge_evolution.types import CandidateType, ContentClass

    proposal = KnowledgeProposal(
        CandidateType.EXPERIENCE.value,
        ContentClass.STRATEGY.value,
        "Use repository reader",
        "Inspect the repository through the canonical read tool.",
        ("When repository evidence is required",),
        ("tool:read_file", "arbitrary human predicate"),
    )
    result = validate_and_canonicalize(proposal)
    assert result.proposal is not None
    assert result.proposal.applicability_fingerprints[0][0] == "guard:0"
    assert result.proposal.applicability_fingerprints[1][0] == "guard:tool:read_file"


@pytest.mark.parametrize(
    ("runtime_outcome", "category", "expected", "reason"),
    [
        ("completed", "provider_server", RunValidity.VALID, None),
        ("completed_with_tool_failure", "provider_transport", RunValidity.VALID, None),
        ("timeout", "", RunValidity.VALID, None),
        ("provider_failed", "provider_transport", RunValidity.INFRA_INVALID, InfraInvalidReason.PROVIDER_TRANSPORT),
        ("provider_failed", "provider_server", RunValidity.INFRA_INVALID, InfraInvalidReason.PROVIDER_SERVER),
        ("provider_failed", "provider_malformed_response", RunValidity.INFRA_INVALID, InfraInvalidReason.PROVIDER_MALFORMED_RESPONSE),
        ("provider_failed", "provider_unknown", RunValidity.VALID, None),
        ("provider_failed", "provider_model_error", RunValidity.VALID, None),
    ],
)
def test_run_validity_separates_task_failure_from_objective_infrastructure(
    runtime_outcome, category, expected, reason
) -> None:
    result = classify_run_validity(
        runtime_outcome=runtime_outcome,
        normalized_provider_failures=((category,) if category else ()),
    )
    assert (result.validity, result.reason) == (expected, reason)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (InfraInvalidReason.BENCHMARK_HOST_FAILURE, RunValidity.INFRA_INVALID),
        (InfraInvalidReason.WORKTREE_SETUP_FAILURE, RunValidity.INFRA_INVALID),
        (InfraInvalidReason.VERIFIER_HOST_FAILURE, RunValidity.INFRA_INVALID),
    ],
)
def test_explicit_harness_failures_are_infra_invalid(reason, expected) -> None:
    result = classify_run_validity(runtime_outcome="error", infrastructure_reason=reason)
    assert result.validity is expected
    assert result.reason is reason


def test_worktree_setup_failure_persists_immutable_infra_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)

    def fail_worktree(*_args, **_kwargs):
        raise RuntimeError("bounded setup failure")

    monkeypatch.setattr(
        "benchmarks.picobench.packs.knowledge_evolution_live.runner._git",
        fail_worktree,
    )
    planned = manifest.planned_runs[0]
    record = execute_one(
        repository=tmp_path,
        campaign_root=paths.root,
        run_id=planned.run_id,
        reviewer_id="human:test",
        execute_live=True,
    )
    assert record.run_validity is RunValidity.INFRA_INVALID
    assert record.infra_invalid_reason is InfraInvalidReason.WORKTREE_SETUP_FAILURE
    assert load_run(paths.runs / f"{planned.run_id}.json") == record


def test_provider_failure_normalization_is_bounded_and_privacy_safe() -> None:
    raw = (
        "DeepseekException - Unable to get json response - Expecting value: line 1 "
        "column 1 - Original Response: secret-body"
    )
    normalized = evidence.normalize_provider_failure("server", message=raw)
    assert normalized == evidence.PROVIDER_MALFORMED_RESPONSE
    assert "secret" not in normalized
    assert evidence.normalize_provider_failure("network") == evidence.PROVIDER_TRANSPORT
    assert evidence.normalize_provider_failure("server") == evidence.PROVIDER_SERVER
    assert evidence.normalize_provider_failure("rate_limit") == evidence.PROVIDER_RATE_LIMIT
    assert evidence.normalize_provider_failure("auth") == evidence.PROVIDER_AUTH
    assert evidence.normalize_provider_failure("invalid_request") == evidence.PROVIDER_MODEL_ERROR
    assert evidence.normalize_provider_failure("unknown") == evidence.PROVIDER_UNKNOWN


def test_replacement_is_fresh_lineaged_bounded_and_never_automatic(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = CampaignPaths.at(tmp_path / manifest.campaign_id)
    freeze_manifest(paths, manifest)
    original = manifest.planned_runs[0]
    invalid = _record(
        manifest,
        original,
        2,
        run_validity=RunValidity.INFRA_INVALID,
        infra_invalid_reason=InfraInvalidReason.PROVIDER_TRANSPORT,
    )
    write_run(paths, invalid)
    with pytest.raises(PermissionError, match="--execute-live"):
        run_replacement(
            repository=tmp_path,
            campaign_root=paths.root,
            invalid_run_id=original.run_id,
            reviewer_id="human:test",
            execute_live=False,
        )
    assert not paths.replacements.exists()
    replacement = schedule_replacement(paths, manifest, invalid_run_id=original.run_id)
    assert replacement.run_id != original.run_id
    assert replacement.replacement_for_run_id == original.run_id
    assert (replacement.task_id, replacement.arm, replacement.repetition) == (
        original.task_id,
        original.arm,
        original.repetition,
    )
    with pytest.raises(ValueError, match="at most one"):
        schedule_replacement(paths, manifest, invalid_run_id=original.run_id)


def test_reducer_excludes_invalid_efficacy_but_retains_reliability() -> None:
    manifest = _manifest()
    records = []
    invalid_planned = manifest.planned_runs[0]
    for planned in manifest.planned_runs:
        if planned == invalid_planned:
            records.append(
                _record(
                    manifest,
                    planned,
                    3,
                    outcome="fail",
                    run_validity=RunValidity.INFRA_INVALID,
                    infra_invalid_reason=InfraInvalidReason.PROVIDER_SERVER,
                    terminal_provider_failure=True,
                )
            )
        else:
            records.append(_record(manifest, planned, 2))
    result = reduce_campaign(manifest, tuple(records))
    assert result["infra_invalid_runs"] == 1
    assert result["provider_reliability"]["terminal_failures"] == 1
    assert result["valid_complete_pairs"] == 2
    assert result["incomplete_pairs"] == [
        {"task_id": invalid_planned.task_id, "repetition": invalid_planned.repetition}
    ]


def test_integration_verifier_accepts_fix_or_exact_proof_and_rejects_scratch() -> None:
    spec = VERIFIERS["p3r-v-integration-skill"]
    assert changed_path_findings(
        spec,
        ["pico/memory_engine/skill_forge/knowledge_source.py", "tests/test_x.py"],
    ) == ()
    assert changed_path_findings(spec, ["tests/test_knowledge_runtime_integration.py"]) == ()
    assert changed_path_findings(spec, ["tests/test_scratch_two_source.py"]) == (
        "missing_required_path_class",
    )
    assert spec.version == 2


def test_integration_proof_only_requires_actual_two_source_router(tmp_path: Path) -> None:
    spec = VERIFIERS["p3r-v-integration-skill"]
    proof = tmp_path / "tests/test_knowledge_runtime_integration.py"
    proof.parent.mkdir(parents=True)
    proof.write_text("router = SkillForgeRouter([p3_source])", encoding="utf-8")
    changed = ["tests/test_knowledge_runtime_integration.py"]
    assert integration_proof_findings(spec, tmp_path, changed) == (
        "regression_test_missing",
    )
    proof.write_text(
        "router = SkillForgeRouter([local_source, p3_source])",
        encoding="utf-8",
    )
    assert integration_proof_findings(spec, tmp_path, changed) == ()
    assert integration_proof_findings(
        spec,
        tmp_path,
        ["pico/memory_engine/skill_forge/router.py", "tests/test_x.py"],
    ) == ()


def test_repaired_verifier_digest_does_not_replace_historical_digest() -> None:
    historical_digest = "e21f4f4753c1b2c7330d667481d6b0d58221d809ab952764a26f47f6cfde3bab"
    spec = VERIFIERS["p3r-v-integration-skill"]
    manifest = _manifest()
    assert spec.digest != historical_digest
    assert dict(manifest.verifier_digests)["p3r-int-01"] == spec.digest
    assert manifest.benchmark_version.endswith("-v4")


def test_debug_verifier_is_semantic_version_three_and_preserves_old_digest() -> None:
    historical = HiddenVerifierSpec(
        "p3r-v-debug-guard",
        ("pico/knowledge_evolution/canonicalize.py", "tests/*"),
        (("pico/knowledge_evolution/canonicalize.py", "guard:"),),
        ("tests/test_knowledge_canonicalization.py", "tests/test_knowledge_applicability.py"),
        version=2,
    )
    assert historical.digest == "7fe01a075a597289156261d8eeb619104a308bae8f60ca49dc4cc8d848fb57b6"
    assert VERIFIERS["p3r-v-debug-guard"].version == 3
    assert VERIFIERS["p3r-v-debug-guard"].digest != historical.digest
    assert corpus_digest() == "39b04fca4f6aa9e54568594b1c18dfe9f71c2292c643469197a96c8977368fbf"


def test_debug_verifier_requires_regression_coverage() -> None:
    spec = VERIFIERS["p3r-v-debug-guard"]
    assert changed_path_findings(
        spec,
        ["pico/knowledge_evolution/canonicalize.py"],
    ) == ("regression_test_missing",)


def test_debug_verifier_preserves_targeted_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.picobench.packs.knowledge_evolution_live.verifiers as module

    monkeypatch.setattr(
        module,
        "_git",
        lambda _workspace, command, *_args: (
            "pico/knowledge_evolution/canonicalize.py\ntests/test_guard.py"
            if command == "diff"
            else ""
        ),
    )
    monkeypatch.setattr(module, "debug_semantic_findings", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )
    result = verify_workspace(
        "p3r-v-debug-guard",
        tmp_path,
        python_executable="python",
    )
    assert result.passed is False
    assert result.findings == ("targeted_test_failed",)


def test_hidden_verifier_findings_round_trip_but_are_not_turn_input(tmp_path: Path) -> None:
    manifest = _manifest()
    record = _record(
        manifest,
        manifest.planned_runs[0],
        1,
        outcome="fail",
        verifier_findings=("targeted_test_failed", "regression_test_missing"),
    )
    path = tmp_path / "record.json"
    from benchmarks.picobench.artifacts import ArtifactStore
    from benchmarks.picobench.canonical import to_primitive
    from benchmarks.picobench.schema import ExperimentRef

    ArtifactStore(ExperimentRef("test", tmp_path)).append_immutable(path, to_primitive(record))
    loaded = load_run(path)
    assert loaded.verifier_findings == record.verifier_findings
    assert all(finding not in loaded.trace_evidence_refs for finding in loaded.verifier_findings)


def test_pilot_tasks_have_mechanical_non_live_solvability_fixtures(tmp_path: Path) -> None:
    debug = tmp_path / "pico/knowledge_evolution/canonicalize.py"
    integration = tmp_path / "pico/memory_engine/skill_forge/knowledge_source.py"
    debug.parent.mkdir(parents=True)
    integration.parent.mkdir(parents=True)
    debug.write_text(
        'fingerprints.append((f"guard:{index}", structural_digest({"guard": guard})))',
        encoding="utf-8",
    )
    integration.write_text("class ApplicableKnowledgeSkillSource: pass", encoding="utf-8")
    for relative in {
        *VERIFIERS["p3r-v-debug-guard"].targeted_tests,
        *VERIFIERS["p3r-v-integration-skill"].targeted_tests,
    }:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# deterministic fixture", encoding="utf-8")
    results = {
        item.task_id: item
        for item in audit_pilot_solvability(tmp_path, execute_references=False)
    }
    assert results["p3r-debug-01"].passed, results["p3r-debug-01"].findings
    assert results["p3r-int-01"].passed, results["p3r-int-01"].findings


def test_frozen_pilot_base_reference_fixtures_pass_sealed_verifiers(tmp_path: Path) -> None:
    repository = Path.cwd()
    base = tmp_path / "pilot-base"
    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(base),
            "f5ac937b091905786a88cfc9abfbe0c2fa9bb22d",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        results = {item.task_id: item for item in audit_pilot_solvability(base)}
        assert results["p3r-debug-01"].passed, results["p3r-debug-01"].findings
        assert results["p3r-int-01"].passed, results["p3r-int-01"].findings
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(base)],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )


def test_repository_read_projection_is_relative_normalized_and_contained(tmp_path: Path) -> None:
    workspace = tmp_path / "WorkTree"
    nested = workspace / "pico" / "a.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")
    assert normalize_repository_read_path(str(nested), workspace) == "pico/a.py"
    assert normalize_repository_read_path(str(nested).swapcase(), workspace).casefold() == "pico/a.py"
    assert normalize_repository_read_path("pico\\a.py", workspace) == "pico/a.py"
    assert normalize_repository_read_path("../outside.py", workspace) is None
    assert normalize_repository_read_path(str(tmp_path / "outside.py"), workspace) is None
    projected = normalize_repository_read_path(str(nested), workspace)
    assert str(workspace) not in projected


def _token_metrics(tmp_path: Path, usages: tuple[dict | None, ...], *, exhausted: bool = False):
    turn_id = "turn-token"
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="p3r:test",
        trace_id="trace-token",
        root_span_id="span-token",
        writer=TraceStore(tmp_path).append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    for index, usage in enumerate(usages, start=1):
        correlations = {
            "logical_call_id": f"call-{index}",
            "attempt_id": f"attempt-{index}",
            "attempt_ordinal": 1,
        }
        recorder.emit(
            evidence.PROVIDER_ATTEMPT_STARTED,
            correlations=correlations,
            metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA},
        )
        recorder.emit(
            evidence.PROVIDER_ATTEMPT_COMPLETED,
            correlations=correlations,
            metadata={
                "receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA,
                "outcome": "success" if usage is not None else "error",
                "usage_available": usage is not None,
                "usage": usage or {},
            },
        )
    if exhausted:
        recorder.emit(
            evidence.AGENT_ITERATION_BUDGET_EXHAUSTED,
            metadata={"agent_iterations": len(usages), "final_synthesis_call_expected": True},
        )
        correlations = {
            "logical_call_id": "call-synthesis",
            "attempt_id": "attempt-synthesis",
            "attempt_ordinal": 1,
        }
        recorder.emit(evidence.PROVIDER_ATTEMPT_STARTED, correlations=correlations, metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA})
        recorder.emit(evidence.PROVIDER_ATTEMPT_COMPLETED, correlations=correlations, metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA, "outcome": "success", "usage_available": True, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}})
    recorder.emit(evidence.TURN_TERMINAL, metadata={"outcome": "completed"})
    return extract_run_metrics(trace_root=tmp_path, turn_id=turn_id, knowledge_state_root=tmp_path)[0]


def test_token_accounting_complete_partial_and_not_available(tmp_path: Path) -> None:
    complete = _token_metrics(
        tmp_path / "complete",
        ({"prompt_tokens": 4, "completion_tokens": 2, "cached_tokens": 1},),
    )
    partial_metrics = _token_metrics(
        tmp_path / "partial",
        ({"prompt_tokens": 4, "completion_tokens": 2, "cached_tokens": 1}, None),
    )
    missing = _token_metrics(tmp_path / "missing", (None,))
    assert complete.token_accounting_status is TokenAccountingStatus.COMPLETE
    assert complete.known_input_tokens_sum == available(4)
    assert partial_metrics.token_accounting_status is TokenAccountingStatus.PARTIAL
    assert partial_metrics.known_input_tokens_sum.value == 4
    assert partial_metrics.known_input_tokens_sum.availability is Availability.PARTIAL
    assert partial_metrics.attempts_with_usage.value == 1
    assert partial_metrics.attempts_without_usage.value == 1
    assert missing.token_accounting_status is TokenAccountingStatus.NOT_AVAILABLE
    assert missing.known_input_tokens_sum.value is None


def test_partial_token_pair_is_diagnostic_only() -> None:
    manifest = _manifest()
    records = []
    target = manifest.planned_runs[0].task_id
    for planned in manifest.planned_runs:
        metrics = replace(
            _metrics(2),
            token_accounting_status=TokenAccountingStatus.COMPLETE,
            known_input_tokens_sum=available(200),
        )
        if planned.task_id == target and planned.arm is Arm.APPROVED_REUSE:
            metrics = replace(
                metrics,
                input_tokens=MetricValue(200, Availability.PARTIAL),
                known_input_tokens_sum=MetricValue(200, Availability.PARTIAL),
                token_accounting_status=TokenAccountingStatus.PARTIAL,
            )
        records.append(_record(manifest, planned, 2, metrics=metrics))
    reduced = reduce_campaign(manifest, tuple(records))
    target_pair = next(item for item in reduced["pairs"] if item["task_id"] == target)
    assert target_pair["deltas"]["input_tokens"] == {"absolute": None, "relative": None}


def test_iteration_exhaustion_and_synthesis_are_separate_evidence(tmp_path: Path) -> None:
    metrics = _token_metrics(
        tmp_path,
        ({"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0},),
        exhausted=True,
    )
    assert metrics.agent_iterations.value == 1
    assert metrics.iteration_exhausted is True
    assert metrics.final_synthesis_call_present is True
    assert metrics.provider_logical_calls.value == 2


def test_budget_contract_labels_only_enforced_controls_as_limits() -> None:
    budget = RuntimeBudget(
        max_agent_iterations=40,
        provider_logical_calls_observational=40,
        tool_calls_observational=40,
    )
    assert budget.max_agent_iterations == 40
    assert budget.final_synthesis_call_allowed is True
    assert not hasattr(budget, "max_provider_logical_calls")
    assert not hasattr(budget, "max_tool_calls")
