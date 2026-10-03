"""Frozen planning and fairness rules for the P3R paired campaign."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmarks.picobench.canonical import canonical_digest

from .knowledge import corpus_digest
from .schema import (
    BENCHMARK_VERSION,
    Arm,
    CampaignManifest,
    CampaignMode,
    PlannedRun,
    RuntimeBudget,
)
from .tasks import PILOT_TASK_IDS, TASKS

EXPERIMENT_QUESTION = (
    "Under identical real repository tasks, model/provider configuration, Runtime budget, "
    "repository state, Tool availability, and verifier contract, does approved P3 reusable "
    "knowledge reduce Agent discovery/execution cost without reducing independently verified "
    "final-task success?"
)
BENEFICIAL_MEDIAN_IMPROVEMENT = -0.15
MATERIAL_MEDIAN_REGRESSION = 0.20
REPEATED_BASELINE_PASS_TREATMENT_FAIL = 2
REPLICATION_QUESTIONS = (
    "Does nav-01 treatment again explore materially longer than baseline?",
    "Does nav-01 again receive all six candidates?",
    "Does treatment again delay first edit compared with baseline?",
    "Does debug treatment still take a broader cross-module path after the semantic verifier repair?",
    "Does int-01 treatment again reduce repository reads/tokens relative to baseline?",
    "Do all treatment tasks still show identical 6-retrieved / 5-injected / 1-referenced behavior?",
)


def selected_task_ids(mode: CampaignMode) -> tuple[str, ...]:
    return PILOT_TASK_IDS if mode is CampaignMode.PILOT else tuple(item.task_id for item in TASKS)


def repetitions(mode: CampaignMode) -> int:
    return 2 if mode is CampaignMode.OFFICIAL_REPEAT2 else 1


def arm_order(*, task_id: str, repetition: int, seed: int) -> tuple[Arm, Arm]:
    """Pre-freeze balanced order; the second repetition reverses the first."""

    parity = int(canonical_digest({"seed": seed, "task_id": task_id})[:8], 16) % 2
    if repetition % 2 == 0:
        parity ^= 1
    return (Arm.NO_REUSE, Arm.APPROVED_REUSE) if parity == 0 else (Arm.APPROVED_REUSE, Arm.NO_REUSE)


def make_plan(
    mode: CampaignMode,
    *,
    seed: int,
    pilot_repetition: int = 1,
) -> tuple[PlannedRun, ...]:
    runs: list[PlannedRun] = []
    order = 0
    repetition_values = (
        (pilot_repetition,)
        if mode is CampaignMode.PILOT
        else tuple(range(1, repetitions(mode) + 1))
    )
    for repetition in repetition_values:
        for task_id in selected_task_ids(mode):
            for arm in arm_order(task_id=task_id, repetition=repetition, seed=seed):
                order += 1
                runs.append(
                    PlannedRun(
                        run_id=f"{task_id}-r{repetition}-{arm.value}",
                        task_id=task_id,
                        arm=arm,
                        repetition=repetition,
                        order=order,
                    )
                )
    return tuple(runs)


def fairness_payload(manifest: CampaignManifest, task_id: str, repetition: int) -> dict[str, Any]:
    """Fields that MUST match across arms; knowledge availability is excluded."""

    prompts = dict(manifest.task_prompt_digests)
    verifiers = dict(manifest.verifier_digests)
    return {
        "task_id": task_id,
        "repetition": repetition,
        "base_commit_sha": manifest.base_commit_sha,
        "task_prompt_digest": prompts[task_id],
        "provider_id": manifest.provider_id,
        "actual_model_id": manifest.actual_model_id,
        "provider_model_config_digest": manifest.provider_model_config_digest,
        "tool_config_digest": manifest.tool_config_digest,
        "runtime_config_digest": manifest.runtime_config_digest,
        "runtime_budget": manifest.budget,
        "verifier_digest": verifiers[task_id],
        "context_budget": manifest.budget.context_window_tokens,
        "retry_policy": manifest.budget.retry_policy,
    }


def fairness_digest(manifest: CampaignManifest, task_id: str, repetition: int) -> str:
    return canonical_digest(fairness_payload(manifest, task_id, repetition))


def assert_fair_plan(manifest: CampaignManifest) -> None:
    repetition_values = (
        (manifest.pilot_repetition,)
        if manifest.mode is CampaignMode.PILOT
        else tuple(range(1, repetitions(manifest.mode) + 1))
    )
    expected = {
        (task_id, repetition, arm)
        for task_id in manifest.task_ids
        for repetition in repetition_values
        for arm in Arm
    }
    actual = {(item.task_id, item.repetition, item.arm) for item in manifest.planned_runs}
    if expected != actual:
        raise ValueError("P3R manifest does not contain exactly one run per paired arm")


def create_manifest(
    *,
    mode: CampaignMode,
    base_commit_sha: str,
    seed: int,
    provider_id: str,
    actual_model_id: str,
    provider_model_config_digest: str,
    tool_config_digest: str,
    runtime_config_digest: str,
    budget: RuntimeBudget | None = None,
    created_at: str | None = None,
    pilot_repetition: int = 1,
) -> CampaignManifest:
    task_ids = selected_task_ids(mode)
    tasks = tuple(item for item in TASKS if item.task_id in task_ids)
    identity = {
        "benchmark_version": BENCHMARK_VERSION,
        "validity_policy_version": 1,
        "task_contract_version": 2,
        "mode": mode.value,
        "base_commit_sha": base_commit_sha,
        "seed": seed,
        "task_ids": task_ids,
        "task_prompt_digests": tuple((item.task_id, item.prompt_digest) for item in tasks),
        "verifier_digests": tuple((item.task_id, item.verifier_digest) for item in tasks),
        "provider_model_config_digest": provider_model_config_digest,
        "tool_config_digest": tool_config_digest,
        "runtime_config_digest": runtime_config_digest,
        "knowledge_corpus_digest": corpus_digest(),
        "pilot_repetition": pilot_repetition,
        "replication_questions": REPLICATION_QUESTIONS,
    }
    manifest = CampaignManifest.create(
        campaign_id=f"p3r-{canonical_digest(identity)[:16]}",
        mode=mode,
        base_commit_sha=base_commit_sha,
        campaign_seed=seed,
        provider_id=provider_id,
        actual_model_id=actual_model_id,
        provider_model_config_digest=provider_model_config_digest,
        tool_config_digest=tool_config_digest,
        runtime_config_digest=runtime_config_digest,
        budget=budget or RuntimeBudget(),
        task_ids=task_ids,
        task_prompt_digests=tuple((item.task_id, item.prompt_digest) for item in tasks),
        verifier_digests=tuple((item.task_id, item.verifier_digest) for item in tasks),
        knowledge_corpus_digest=corpus_digest(),
        planned_runs=make_plan(mode, seed=seed, pilot_repetition=pilot_repetition),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        pilot_repetition=pilot_repetition,
        replication_questions=REPLICATION_QUESTIONS,
    )
    manifest.validate()
    assert_fair_plan(manifest)
    return manifest


__all__ = [
    "BENEFICIAL_MEDIAN_IMPROVEMENT",
    "EXPERIMENT_QUESTION",
    "MATERIAL_MEDIAN_REGRESSION",
    "REPEATED_BASELINE_PASS_TREATMENT_FAIL",
    "REPLICATION_QUESTIONS",
    "arm_order",
    "assert_fair_plan",
    "create_manifest",
    "fairness_digest",
    "fairness_payload",
    "make_plan",
    "repetitions",
    "selected_task_ids",
]
