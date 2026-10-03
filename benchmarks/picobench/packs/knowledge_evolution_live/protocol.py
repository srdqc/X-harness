"""Frozen planning and fairness rules for the P3R paired campaign."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmarks.picobench.canonical import canonical_digest

from .knowledge import corpus_digest
from .official_fixtures import OFFICIAL_SUITE_VERSION, official_reference_digests
from .schema import (
    BENCHMARK_VERSION,
    Arm,
    CampaignManifest,
    CampaignMode,
    PlannedRun,
    RuntimeBudget,
)
from .tasks import PILOT_TASK_IDS, TASKS, task_by_id

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
P3R3_EXPLORATORY_QUESTIONS = (
    "Does selective mode eliminate identical all-six exposure across heterogeneous tasks?",
    "Does nav selective reduce discovery cost relative to legacy without reducing verified success?",
    "Does integration selective retain genuinely relevant Skill-flow knowledge?",
    "Does debug selective retain fail-closed/recovery knowledge without unrelated knowledge?",
    "Can selective mode return zero candidates for an unrelated or low-signal task?",
    "Does selective mode preserve scope, lifecycle, applicability, authority, and safety?",
)


def selected_task_ids(mode: CampaignMode) -> tuple[str, ...]:
    return (
        PILOT_TASK_IDS
        if mode in {CampaignMode.PILOT, CampaignMode.P3R3_EXPLORATORY}
        else tuple(item.task_id for item in TASKS)
    )


def repetitions(mode: CampaignMode) -> int:
    return 2 if mode is CampaignMode.OFFICIAL_REPEAT2 else 1


def arm_order(*, task_id: str, repetition: int, seed: int, treatment: Arm = Arm.APPROVED_REUSE) -> tuple[Arm, Arm]:
    """Pre-freeze balanced order; the second repetition reverses the first."""

    parity = int(canonical_digest({"seed": seed, "task_id": task_id})[:8], 16) % 2
    if repetition % 2 == 0:
        parity ^= 1
    return (Arm.NO_REUSE, treatment) if parity == 0 else (treatment, Arm.NO_REUSE)


def arms_for_mode(mode: CampaignMode) -> tuple[Arm, ...]:
    if mode is CampaignMode.P3R3_EXPLORATORY:
        return (
            Arm.NO_REUSE,
            Arm.APPROVED_REUSE_LEGACY,
            Arm.APPROVED_REUSE_SELECTIVE,
        )
    if mode is CampaignMode.PILOT:
        return (Arm.NO_REUSE, Arm.APPROVED_REUSE)
    return (Arm.NO_REUSE, Arm.APPROVED_REUSE_SELECTIVE)


def experimental_arm_order(*, task_id: str, repetition: int, seed: int) -> tuple[Arm, ...]:
    arms = arms_for_mode(CampaignMode.P3R3_EXPLORATORY)
    offset = int(canonical_digest({"seed": seed, "task_id": task_id, "repetition": repetition})[:8], 16) % len(arms)
    return arms[offset:] + arms[:offset]


def make_plan(
    mode: CampaignMode,
    *,
    seed: int,
    pilot_repetition: int = 1,
) -> tuple[PlannedRun, ...]:
    runs: list[PlannedRun] = []
    order = 0
    repetition_values = (pilot_repetition,) if mode is CampaignMode.PILOT else tuple(range(1, repetitions(mode) + 1))
    for repetition in repetition_values:
        for task_id in selected_task_ids(mode):
            ordered_arms = (
                experimental_arm_order(task_id=task_id, repetition=repetition, seed=seed)
                if mode is CampaignMode.P3R3_EXPLORATORY
                else arm_order(
                    task_id=task_id,
                    repetition=repetition,
                    seed=seed,
                    treatment=arms_for_mode(mode)[1],
                )
            )
            for arm in ordered_arms:
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


def arm_configuration_payload(manifest: CampaignManifest, planned: PlannedRun) -> dict[str, Any]:
    payload = fairness_payload(manifest, planned.task_id, planned.repetition)
    reuse = planned.arm is not Arm.NO_REUSE
    selection_mode = {
        Arm.NO_REUSE: None,
        Arm.APPROVED_REUSE: "legacy_applicable",
        Arm.APPROVED_REUSE_LEGACY: "legacy_applicable",
        Arm.APPROVED_REUSE_SELECTIVE: "task_relevance_v1",
    }[planned.arm]
    return {
        **payload,
        "p3_reuse_available": reuse,
        "knowledge_selection_mode": selection_mode,
    }


def arm_configuration_digest(manifest: CampaignManifest, planned: PlannedRun) -> str:
    return canonical_digest(arm_configuration_payload(manifest, planned))


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
        for arm in arms_for_mode(manifest.mode)
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
    tasks = tuple(task_by_id(task_id) for task_id in task_ids)
    official = mode in {CampaignMode.OFFICIAL_SINGLE, CampaignMode.OFFICIAL_REPEAT2}
    campaign_benchmark_version = BENCHMARK_VERSION if official else "p3r-live-experience-reuse-v4"
    selector_mode = "task_relevance_v1" if official else ""
    selector_version = 1 if official else 0
    selector_config_digest = (
        canonical_digest({"selection_mode": selector_mode, "selector_version": selector_version}) if official else ""
    )
    identity = {
        "benchmark_version": campaign_benchmark_version,
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
        "replication_questions": (
            P3R3_EXPLORATORY_QUESTIONS if mode is CampaignMode.P3R3_EXPLORATORY else REPLICATION_QUESTIONS
        ),
        "suite_version": OFFICIAL_SUITE_VERSION if official else "",
        "mechanical_solvability_digests": official_reference_digests() if official else (),
        "selector_mode": selector_mode,
        "selector_version": selector_version,
        "selector_config_digest": selector_config_digest,
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
        replication_questions=(
            P3R3_EXPLORATORY_QUESTIONS if mode is CampaignMode.P3R3_EXPLORATORY else REPLICATION_QUESTIONS
        ),
        suite_version=OFFICIAL_SUITE_VERSION if official else "",
        mechanical_solvability_digests=official_reference_digests() if official else (),
        selector_mode=selector_mode,
        selector_version=selector_version,
        selector_config_digest=selector_config_digest,
        benchmark_version=campaign_benchmark_version,
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
    "P3R3_EXPLORATORY_QUESTIONS",
    "arm_order",
    "arm_configuration_digest",
    "arm_configuration_payload",
    "arms_for_mode",
    "assert_fair_plan",
    "create_manifest",
    "fairness_digest",
    "fairness_payload",
    "experimental_arm_order",
    "make_plan",
    "repetitions",
    "selected_task_ids",
]
