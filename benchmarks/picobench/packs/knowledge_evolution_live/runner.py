"""Human-gated live runner. Importing this module never constructs a Provider."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.host import RecordingOutlet, RuntimeTrialHost
from pico.config.paths import RuntimePaths
from pico.knowledge_evolution import (
    KnowledgeRecordStore,
    RepositoryScopeResolver,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    associate_usage_outcome,
)
from pico.spine.message import ChatType, Source
from pico.spine.turn import Origin, TurnRequest
from pico.tracing import evidence

from .artifacts import (
    completed_run_ids,
    load_manifest,
    load_replacements,
    load_runs,
    store_for,
    write_replacement,
    write_run,
)
from .knowledge import prepare_approved_corpus
from .metrics import extract_run_metrics
from .prepare import configuration_identity
from .protocol import assert_fair_plan, fairness_digest
from .schema import (
    Arm,
    Availability,
    CampaignPaths,
    InfraInvalidReason,
    MetricValue,
    PlannedRun,
    RunMetrics,
    RunRecord,
    RunValidity,
    TokenAccountingStatus,
)
from .tasks import task_by_id
from .validity import classify_run_validity
from .verifiers import verify_workspace


def campaign_summary(manifest) -> str:
    repetitions = max(item.repetition for item in manifest.planned_runs)
    return (
        f"campaign={manifest.campaign_id} live_runs={len(manifest.planned_runs)} "
        f"tasks={len(manifest.task_ids)} arms=2 repetitions={repetitions} "
        f"provider={manifest.provider_id} model={manifest.actual_model_id} "
        f"max_agent_iterations={manifest.budget.max_agent_iterations} "
        f"final_synthesis_call_allowed={manifest.budget.final_synthesis_call_allowed} "
        f"provider_calls_observational={manifest.budget.provider_logical_calls_observational} "
        f"tool_calls_observational={manifest.budget.tool_calls_observational}"
    )


def missing_runs(paths: CampaignPaths, manifest) -> tuple[PlannedRun, ...]:
    completed = completed_run_ids(paths)
    planned = (*manifest.planned_runs, *load_replacements(paths))
    return tuple(item for item in planned if item.run_id not in completed)


def schedule_replacement(
    paths: CampaignPaths,
    manifest,
    *,
    invalid_run_id: str,
) -> PlannedRun:
    invalid = next((item for item in load_runs(paths) if item.run_id == invalid_run_id), None)
    if invalid is None or invalid.run_validity is not RunValidity.INFRA_INVALID:
        raise ValueError("replacement requires an immutable INFRA_INVALID run")
    existing = tuple(
        item for item in load_replacements(paths) if item.replacement_for_run_id == invalid_run_id
    )
    if existing:
        raise ValueError("at most one replacement is allowed for an INFRA_INVALID run")
    original = next((item for item in manifest.planned_runs if item.run_id == invalid_run_id), None)
    if original is None:
        raise ValueError("invalid run is not in the frozen campaign")
    planned = PlannedRun(
        run_id=f"{invalid_run_id}-replacement-1",
        task_id=original.task_id,
        arm=original.arm,
        repetition=original.repetition,
        order=max((item.order for item in (*manifest.planned_runs, *load_replacements(paths))), default=0)
        + 1,
        replacement_for_run_id=invalid_run_id,
        replacement_ordinal=1,
    )
    write_replacement(paths, planned)
    return planned


def run_replacement(
    *,
    repository: Path,
    campaign_root: Path,
    invalid_run_id: str,
    reviewer_id: str,
    execute_live: bool,
) -> RunRecord:
    if not execute_live:
        raise PermissionError("replacement live execution requires explicit --execute-live")
    paths = CampaignPaths.at(campaign_root)
    manifest = load_manifest(paths.manifest)
    planned = schedule_replacement(paths, manifest, invalid_run_id=invalid_run_id)
    return execute_one(
        repository=repository,
        campaign_root=campaign_root,
        run_id=planned.run_id,
        reviewer_id=reviewer_id,
        execute_live=True,
    )


def run_campaign(
    *,
    repository: Path,
    campaign_root: Path,
    reviewer_id: str,
    execute_live: bool,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    paths = CampaignPaths.at(campaign_root)
    manifest = load_manifest(paths.manifest)
    assert_fair_plan(manifest)
    print(campaign_summary(manifest))
    if not execute_live:
        raise PermissionError("live execution requires explicit --execute-live")
    missing = missing_runs(paths, manifest)
    for planned in missing:
        try:
            subprocess.run(
                [
                    python_executable,
                    "-m",
                    "benchmarks.picobench.packs.knowledge_evolution_live",
                    "_run-one",
                    "--repository",
                    str(repository),
                    "--campaign-root",
                    str(campaign_root),
                    "--run-id",
                    planned.run_id,
                    "--reviewer-id",
                    reviewer_id,
                    "--execute-live",
                ],
                cwd=repository,
                check=True,
            )
        except subprocess.CalledProcessError:
            if planned.run_id not in completed_run_ids(paths):
                write_run(
                    paths,
                    _infrastructure_failure_record(
                        manifest=manifest,
                        planned=planned,
                        reason=InfraInvalidReason.BENCHMARK_HOST_FAILURE,
                    ),
                )
            raise
    return tuple(item.run_id for item in missing)


def execute_one(
    *,
    repository: Path,
    campaign_root: Path,
    run_id: str,
    reviewer_id: str,
    execute_live: bool,
) -> RunRecord:
    if not execute_live:
        raise PermissionError("live execution requires explicit --execute-live")
    paths = CampaignPaths.at(campaign_root)
    manifest = load_manifest(paths.manifest)
    planned = next(
        (
            item
            for item in (*manifest.planned_runs, *load_replacements(paths))
            if item.run_id == run_id
        ),
        None,
    )
    if planned is None:
        raise ValueError(f"run is not in frozen campaign: {run_id}")
    if run_id in completed_run_ids(paths):
        raise FileExistsError(f"immutable run already completed: {run_id}")
    worktree = paths.worktrees / _short_run_name(planned)
    state = paths.states / _short_run_name(planned)
    if worktree.exists() or state.exists():
        raise FileExistsError(f"run isolation path already exists: {run_id}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(repository, "worktree", "add", "--detach", str(worktree), manifest.base_commit_sha)
    except (OSError, RuntimeError):
        record = _infrastructure_failure_record(
            manifest=manifest,
            planned=planned,
            reason=InfraInvalidReason.WORKTREE_SETUP_FAILURE,
        )
        write_run(paths, record)
        return record
    try:
        if _git(worktree, "rev-parse", "HEAD") != manifest.base_commit_sha:
            record = _infrastructure_failure_record(
                manifest=manifest,
                planned=planned,
                reason=InfraInvalidReason.WORKTREE_SETUP_FAILURE,
            )
        else:
            try:
                record = asyncio.run(
                    _execute_turn(
                        paths=paths,
                        manifest=manifest,
                        planned=planned,
                        worktree=worktree,
                        state=state,
                        reviewer_id=reviewer_id,
                    )
                )
            except Exception:  # noqa: BLE001 -- frozen host-failure evidence boundary
                record = _infrastructure_failure_record(
                    manifest=manifest,
                    planned=planned,
                    reason=InfraInvalidReason.BENCHMARK_HOST_FAILURE,
                )
        write_run(paths, record)
        return record
    finally:
        if worktree.exists():
            _git(repository, "worktree", "remove", "--force", str(worktree))


async def _execute_turn(*, paths, manifest, planned, worktree, state, reviewer_id) -> RunRecord:
    from pico.cli._helpers import make_provider
    from pico.config.loader import load_config
    from pico.config.pico import PicoConfig, load_pico_config
    from pico.proactive_engine.schedulers.cron.service import CronService

    config = load_config()
    pico_config = load_pico_config()
    identity = configuration_identity(config, manifest.budget)
    expected = (
        identity["provider_id"],
        identity["model"],
        identity["provider_digest"],
        identity["tool_digest"],
        identity["runtime_digest"],
    )
    actual = (
        manifest.provider_id,
        manifest.actual_model_id,
        manifest.provider_model_config_digest,
        manifest.tool_config_digest,
        manifest.runtime_config_digest,
    )
    if expected != actual:
        raise RuntimeError("configured Runtime/Provider differs from frozen campaign")
    # Both arms use the same isolated no-shared-memory configuration. P3 state
    # remains available only in the treatment arm's dedicated state root.
    pico_config = PicoConfig.model_validate(
        {**pico_config.model_dump(mode="python"), "memory": {"backend": None}}
    )
    if planned.arm is Arm.APPROVED_REUSE:
        prepare_approved_corpus(state_root=state, workspace=worktree, reviewer_id=reviewer_id)

    old_trace_root = os.environ.get("PICO_TRACING_DIR")
    os.environ["PICO_TRACING_DIR"] = str(state)
    started_at = datetime.now(timezone.utc).isoformat()
    outlet = RecordingOutlet("p3r-live")
    provider = make_provider(config)
    cron = CronService(state / "cron" / "jobs.json", allowed_channels={"p3r-live"})
    host = await RuntimeTrialHost.build(
        config=config,
        pico_config=pico_config,
        provider=provider,
        cron_service=cron,
        outlet=outlet,
        paths=RuntimePaths(workspace=worktree, state=state),
        turn_id_factory=lambda: f"{planned.run_id}-turn",
    )
    turn_id = f"{planned.run_id}-turn"
    runtime_outcome = "timeout"
    try:
        request = TurnRequest(
            origin=Origin.USER,
            source=Source("p3r-live", planned.run_id, "p3r-human", ChatType.DM),
            text=task_by_id(planned.task_id).prompt,
            conversation=f"p3r:{planned.run_id}",
        )
        try:
            observation = await asyncio.wait_for(
                host.run(request), timeout=manifest.budget.wall_clock_timeout_seconds
            )
        except TimeoutError:
            # wait_for cancellation flows through Scheduler terminal handling;
            # the one evaluated attempt is recorded and is never retried.
            pass
        else:
            turn_id = observation.turn_id
            runtime_outcome = observation.runtime_state.value
    finally:
        await host.close()
        if old_trace_root is None:
            os.environ.pop("PICO_TRACING_DIR", None)
        else:
            os.environ["PICO_TRACING_DIR"] = old_trace_root
    terminal_at = datetime.now(timezone.utc).isoformat()

    verifier_result = verify_workspace(
        task_by_id(planned.task_id).verifier_id,
        worktree,
        python_executable=sys.executable,
    )
    patch = _workspace_patch(worktree)
    patch_digest = canonical_digest(patch)
    patch_path = paths.patches / f"{planned.run_id}.json"
    store_for(paths).append_immutable(
        patch_path,
        {"run_id": planned.run_id, "patch_digest": patch_digest, **patch},
    )
    store = KnowledgeRecordStore(state)
    scope = RepositoryScopeResolver(worktree, state).resolve()
    if not scope.resolved or scope.identity is None:
        raise RuntimeError("run repository scope is unresolved")
    success = TaskSuccessEvidence.create(
        evidence_id=f"{planned.run_id}-task-success",
        source_kind=TaskSuccessSource.FROZEN_BENCHMARK,
        source_turn_id=turn_id,
        status=TaskSuccessStatus.PASS if verifier_result.passed else TaskSuccessStatus.FAIL,
        producer_id=verifier_result.verifier_id,
        producer_version="1",
        repository_scope_id=scope.identity.repository_scope_id,
        target_state_digest=patch_digest,
        result_digest=canonical_digest(
            {"passed": verifier_result.passed, "findings": verifier_result.findings}
        ),
        provenance_refs=(f"verifier:{verifier_result.verifier_digest}",),
        created_at=terminal_at,
    )
    store.write_task_success(success)
    usages = store.list_usages(turn_id=turn_id)
    associations: tuple[str, ...] = ()
    if usages:
        association = associate_usage_outcome(
            store,
            association_id=f"{planned.run_id}-outcome",
            usage_ids=tuple(item.usage_id for item in usages),
            task_success=success,
            created_at=terminal_at,
        )
        associations = (association.association_id,)
    metrics, refs = extract_run_metrics(
        trace_root=state,
        turn_id=turn_id,
        knowledge_state_root=state,
    )
    readback = evidence.read_turn_evidence(state, turn_id)
    normalized_provider_failures = tuple(refs["normalized_provider_failure_categories"])
    validity = classify_run_validity(
        runtime_outcome=runtime_outcome,
        normalized_provider_failures=normalized_provider_failures,
        infrastructure_reason=(
            InfraInvalidReason.VERIFIER_HOST_FAILURE
            if verifier_result.infrastructure_failure
            else None
        ),
        mandatory_evidence_complete=readback.completeness is evidence.EvidenceCompleteness.COMPLETE,
    )
    attempts_by_call: dict[str, list[evidence.ProviderAttemptEvidence]] = {}
    for attempt in readback.provider_attempts:
        attempts_by_call.setdefault(attempt.logical_call_id, []).append(attempt)
    recovered_provider_failures = sum(
        any(item.outcome == "error" for item in attempts)
        and attempts[-1].outcome == "success"
        for attempts in attempts_by_call.values()
    )
    return RunRecord.create(
        campaign_id=manifest.campaign_id,
        task_id=planned.task_id,
        arm=planned.arm,
        repetition=planned.repetition,
        run_id=planned.run_id,
        workspace_identity=f"git:{manifest.base_commit_sha}:{planned.run_id}",
        turn_id=turn_id,
        provider_id=manifest.provider_id,
        actual_model_id=manifest.actual_model_id,
        started_at=started_at,
        terminal_at=terminal_at,
        runtime_outcome=runtime_outcome,
        verifier_outcome="pass" if verifier_result.passed else "fail",
        safety_outcome="pass" if verifier_result.safety_passed else "fail",
        safety_findings=tuple(
            item for item in verifier_result.findings if item == "prohibited_change_detected"
        ),
        task_success_evidence_ref=success.evidence_id,
        knowledge_retrieval_refs=refs["retrieval_refs"],
        knowledge_usage_refs=refs["usage_refs"],
        outcome_association_refs=associations,
        retrieved_candidate_ids=refs["retrieved_candidate_ids"],
        injected_candidate_ids=refs["injected_ids"],
        referenced_candidate_ids=refs["referenced_ids"],
        activated_candidate_ids=refs["activated_ids"],
        metrics=metrics,
        patch_digest=patch_digest,
        patch_artifact_ref=patch_path.relative_to(paths.root).as_posix(),
        trace_evidence_refs=tuple(
            f"{item.turn_id}:{item.sequence}:{item.event_type}" for item in readback.events
        ),
        fairness_digest=fairness_digest(manifest, planned.task_id, planned.repetition),
        run_validity=validity.validity,
        infra_invalid_reason=validity.reason,
        normalized_provider_failure_categories=normalized_provider_failures,
        recovered_provider_failure_count=recovered_provider_failures,
        terminal_provider_failure=runtime_outcome == "provider_failed",
        verifier_findings=verifier_result.findings,
        replacement_for_run_id=planned.replacement_for_run_id,
        repository_read_paths=refs["repository_read_paths"],
        changed_paths=_changed_paths(worktree),
    )


def _short_run_name(planned: PlannedRun) -> str:
    arm = "a" if planned.arm is Arm.NO_REUSE else "b"
    return f"{planned.task_id.removeprefix('p3r-')}-r{planned.repetition}-{arm}"


def _infrastructure_failure_record(*, manifest, planned, reason: InfraInvalidReason) -> RunRecord:
    """Persist a bounded invalid outcome when no trustworthy Turn record can be completed."""

    now = datetime.now(timezone.utc).isoformat()
    unavailable = MetricValue(None, Availability.NOT_AVAILABLE)
    metrics = RunMetrics(
        provider_logical_calls=unavailable,
        provider_attempts=unavailable,
        tool_calls_total=unavailable,
        unique_repo_files_read=unavailable,
        total_repo_file_reads=unavailable,
        repeated_repo_file_reads=unavailable,
        skill_candidates_retrieved=unavailable,
        skills_referenced=unavailable,
        skills_activated=unavailable,
        repeated_skill_reads=unavailable,
        input_tokens=unavailable,
        output_tokens=unavailable,
        cached_tokens=unavailable,
        turn_latency_ms=unavailable,
        provider_retries=unavailable,
        failed_tool_attempts=unavailable,
        validation_failures=unavailable,
        p3_approximate_context_tokens=unavailable,
        provider_failed_attempts=unavailable,
        token_accounting_status=TokenAccountingStatus.NOT_AVAILABLE,
    )
    return RunRecord.create(
        campaign_id=manifest.campaign_id,
        task_id=planned.task_id,
        arm=planned.arm,
        repetition=planned.repetition,
        run_id=planned.run_id,
        workspace_identity=f"git:{manifest.base_commit_sha}:{planned.run_id}",
        turn_id="",
        provider_id=manifest.provider_id,
        actual_model_id=manifest.actual_model_id,
        started_at=now,
        terminal_at=now,
        runtime_outcome="benchmark_host_failed",
        verifier_outcome="not_run",
        safety_outcome="inconclusive",
        safety_findings=(),
        task_success_evidence_ref=None,
        knowledge_retrieval_refs=(),
        knowledge_usage_refs=(),
        outcome_association_refs=(),
        retrieved_candidate_ids=(),
        injected_candidate_ids=(),
        referenced_candidate_ids=(),
        activated_candidate_ids=(),
        metrics=metrics,
        patch_digest=canonical_digest({"patch": "unavailable", "reason": reason.value}),
        patch_artifact_ref="",
        trace_evidence_refs=(),
        fairness_digest=fairness_digest(manifest, planned.task_id, planned.repetition),
        run_validity=RunValidity.INFRA_INVALID,
        infra_invalid_reason=reason,
        verifier_findings=(),
        replacement_for_run_id=planned.replacement_for_run_id,
        repository_read_paths=(),
        changed_paths=(),
    )


def _workspace_patch(worktree: Path) -> dict[str, object]:
    untracked = _git(worktree, "ls-files", "--others", "--exclude-standard").splitlines()
    files = []
    for relative in sorted(untracked):
        path = worktree / relative
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return {
        "tracked_binary_diff": _git(worktree, "diff", "--binary", "HEAD"),
        "untracked_files": files,
    }


def _changed_paths(worktree: Path) -> tuple[str, ...]:
    tracked = _git(worktree, "diff", "--name-only", "HEAD").splitlines()
    untracked = _git(worktree, "ls-files", "--others", "--exclude-standard").splitlines()
    return tuple(sorted({path.replace("\\", "/") for path in (*tracked, *untracked)}))


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=repository, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = ["campaign_summary", "execute_one", "missing_runs", "run_campaign"]
