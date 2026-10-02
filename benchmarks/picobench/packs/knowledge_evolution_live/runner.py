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

from .artifacts import completed_run_ids, load_manifest, store_for, write_run
from .knowledge import prepare_approved_corpus
from .metrics import extract_run_metrics
from .prepare import configuration_identity
from .protocol import assert_fair_plan, fairness_digest
from .schema import Arm, CampaignPaths, PlannedRun, RunRecord
from .tasks import task_by_id
from .verifiers import verify_workspace


def campaign_summary(manifest) -> str:
    repetitions = max(item.repetition for item in manifest.planned_runs)
    return (
        f"campaign={manifest.campaign_id} live_runs={len(manifest.planned_runs)} "
        f"tasks={len(manifest.task_ids)} arms=2 repetitions={repetitions} "
        f"provider={manifest.provider_id} model={manifest.actual_model_id} "
        f"max_steps={manifest.budget.max_agent_steps} "
        f"max_provider_calls={manifest.budget.max_provider_logical_calls} "
        f"max_tool_calls={manifest.budget.max_tool_calls}"
    )


def missing_runs(paths: CampaignPaths, manifest) -> tuple[PlannedRun, ...]:
    completed = completed_run_ids(paths)
    return tuple(item for item in manifest.planned_runs if item.run_id not in completed)


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
    planned = next((item for item in manifest.planned_runs if item.run_id == run_id), None)
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
    _git(repository, "worktree", "add", "--detach", str(worktree), manifest.base_commit_sha)
    try:
        if _git(worktree, "rev-parse", "HEAD") != manifest.base_commit_sha:
            raise RuntimeError("live run worktree base SHA mismatch")
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
            item for item in verifier_result.findings if item.startswith("safety:")
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
    )


def _short_run_name(planned: PlannedRun) -> str:
    arm = "a" if planned.arm is Arm.NO_REUSE else "b"
    return f"{planned.task_id.removeprefix('p3r-')}-r{planned.repetition}-{arm}"


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


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=repository, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = ["campaign_summary", "execute_one", "missing_runs", "run_campaign"]
