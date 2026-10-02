from __future__ import annotations

import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live.artifacts import (
    completed_run_ids,
    freeze_manifest,
    load_manifest,
    write_run,
)
from benchmarks.picobench.packs.knowledge_evolution_live.knowledge import (
    CORPUS,
    prepare_approved_corpus,
)
from benchmarks.picobench.packs.knowledge_evolution_live.metrics import (
    available,
    extract_run_metrics,
    repeated_repository_reads,
    unavailable,
)
from benchmarks.picobench.packs.knowledge_evolution_live.prepare import preflight
from benchmarks.picobench.packs.knowledge_evolution_live.protocol import (
    arm_order,
    assert_fair_plan,
    create_manifest,
    fairness_digest,
)
from benchmarks.picobench.packs.knowledge_evolution_live.reducer import classify, reduce_campaign
from benchmarks.picobench.packs.knowledge_evolution_live.runner import missing_runs, run_campaign
from benchmarks.picobench.packs.knowledge_evolution_live.schema import (
    Arm,
    BenefitClassification,
    CampaignMode,
    CampaignPaths,
    RunMetrics,
    RunRecord,
)
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import PILOT_TASK_IDS, TASKS
from benchmarks.picobench.packs.knowledge_evolution_live.verifiers import VERIFIERS
from pico.knowledge_evolution import (
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeUsageMode,
    KnowledgeUsageReceipt,
)
from pico.tracing import evidence
from pico.tracing.store import TraceStore


def _manifest(mode: CampaignMode = CampaignMode.PILOT, *, base_sha: str = "a" * 40):
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


def _record(manifest, planned, value: int, outcome: str = "pass") -> RunRecord:
    return RunRecord.create(
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
