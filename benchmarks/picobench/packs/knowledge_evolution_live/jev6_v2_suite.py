"""Frozen offline design contract for the JEV.6 held-out v2 benchmark."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from pico.knowledge_evolution.identity import CANDIDATE_EVIDENCE_IDENTITY_SCHEMA

from .jev6_suite import (
    AGENT_BUDGET,
    BASE_COMMIT,
    BENEFIT_CRITERIA,
    CONTRACT_STATUS,
    EARLY_STOP_POLICY,
    MODEL_ID,
    NO_OUTCOME_LEAKAGE,
    PLANNED_LIVE_RUNS,
    PRIMARY_CORRECTNESS,
    PRIMARY_EFFICIENCY,
    PROVIDER_ID,
    PROVIDER_MODEL_DIGEST,
    REASONING_ACCOUNTING,
    REPETITIONS,
    RUN_ISOLATION,
    SECONDARY_METRICS,
    SELECTOR_CONFIG,
    SELECTOR_CONFIG_DIGEST,
    UTILITY_ACCOUNTING,
    UTILITY_POLICY,
    UTILITY_POLICY_DIGEST,
    Arm,
    Category,
    ContractAudit,
    PlannedRun,
    verifier_digest,
)
from .jev6_suite import (
    TASKS as V1_TASKS,
)
from .schema import LiveTask

SUITE_NAME = "JEV.6 held-out Provider Utility benchmark v2"
SUITE_VERSION = "jev6-held-out-v2"
PREDECESSOR_SUITE = "jev6-held-out-v1"
REPLACEMENT_REASON = "nav-02 live Agent exposure during invalid JEV.6B2"
RUN_ORDER_SEED = 62_207
EXPOSURE_SNAPSHOT_SCHEMA = "pico.jev6-heldout-exposure-snapshot.v2"

RETAINED_TASK_IDS = (
    "jev6-nav-01-trace-event-reader",
    "jev6-impl-01-safe-segment-bound",
    "jev6-impl-02-delivery-result-flags",
    "jev6-debug-01-bm25-parameters",
    "jev6-debug-02-decision-receipt-identity",
    "jev6-int-01-public-turn-events",
    "jev6-int-02-resolve-available-skill",
)
RETIRED_TASK_ID = "jev6-nav-02-skill-variants"
NEW_TASK_ID = "jev6v2-nav-02-plugin-contributions"

_V1_BY_ID = {task.task_id: task for task in V1_TASKS}
NEW_TASK = LiveTask(
    NEW_TASK_ID,
    Category.NAVIGATION_RETRIEVAL.value,
    "Add a public PluginRegistry.contributions_for(plugin_id) method that returns every activated contribution owned by that plugin as (kind, name, factory_reference) tuples, ordered by kind then name. Include both memory_backend and tool contributions, return an empty tuple for an unknown or skipped plugin, inspect only activation records without importing factories, and add focused tests.",
    "jev6v2-v-plugin-contributions",
    verifier_digest("jev6v2-v-plugin-contributions"),
    ("memory_fact",),
    "Requires locating activation-record ownership and retrieving contributions without crossing the lazy factory-import boundary.",
)
TASKS = (
    _V1_BY_ID["jev6-nav-01-trace-event-reader"],
    NEW_TASK,
    *(_V1_BY_ID[task_id] for task_id in RETAINED_TASK_IDS[1:]),
)

CONTRACT_AUDIT = tuple(
    ContractAudit(
        task.task_id,
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
        CONTRACT_STATUS,
    )
    for task in TASKS
)


def make_run_order(seed: int = RUN_ORDER_SEED) -> tuple[PlannedRun, ...]:
    """Create a new deterministic paired and counterbalanced v2 order."""

    rows: list[PlannedRun] = []
    arms = tuple(Arm)
    offset = seed % len(TASKS)
    for repetition in range(1, REPETITIONS + 1):
        task_order = TASKS[offset:] + TASKS[:offset]
        for task_index, task in enumerate(task_order):
            first = (task_index + repetition + seed) % 2
            pair = arms[first:] + arms[:first]
            for arm in pair:
                rows.append(
                    PlannedRun(
                        f"{task.task_id}-{arm.value}-rep{repetition}",
                        task.task_id,
                        arm.value,
                        repetition,
                        len(rows) + 1,
                    )
                )
        offset = (offset + 3) % len(TASKS)
    return tuple(rows)


RUN_ORDER = make_run_order()
RUN_ORDER_DIGEST = canonical_digest(RUN_ORDER)


def suite_payload(
    *,
    selector_audit: dict[str, Any],
    mechanical_audit: dict[str, Any],
    exposure_snapshot: dict[str, Any],
) -> dict[str, Any]:
    from .jev6_v2_fixtures import reference_digests
    from .jev6_v2_verifiers import verifier_set_digest

    references = reference_digests()
    canonical_sequences = {
        task_id: tuple(
            (item["candidate_id"], item["semantic_digest"])
            for item in row["ordered_selected_identities"]
        )
        for task_id, row in selector_audit["tasks"].items()
    }
    payload = {
        "schema": "pico.jev6-heldout-suite.v2",
        "schema_version": 2,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "predecessor_suite": PREDECESSOR_SUITE,
        "replacement_reason": REPLACEMENT_REASON,
        "candidate_identity_schema": CANDIDATE_EVIDENCE_IDENTITY_SCHEMA,
        "base_commit": BASE_COMMIT,
        "tasks": tuple(asdict(task) | {"prompt_digest": task.prompt_digest} for task in TASKS),
        "task_set_digest": canonical_digest(TASKS),
        "verifier_set_digest": verifier_set_digest(),
        "reference_fixtures": references,
        "fixture_set_digest": canonical_digest(references),
        "mechanical_audit": mechanical_audit,
        "mechanical_audit_digest": canonical_digest(mechanical_audit),
        "contract_audit": tuple(asdict(row) for row in CONTRACT_AUDIT),
        "contract_audit_digest": canonical_digest(CONTRACT_AUDIT),
        "selector_audit": selector_audit,
        "selector_audit_digest": canonical_digest(selector_audit),
        "canonical_selector_identities": canonical_sequences,
        "canonical_selector_identities_digest": canonical_digest(canonical_sequences),
        "exposure_snapshot": exposure_snapshot,
        "exposure_snapshot_digest": canonical_digest(exposure_snapshot),
        "corpus_digest": selector_audit["historical_corpus_digest"],
        "selector_config": SELECTOR_CONFIG,
        "selector_config_digest": SELECTOR_CONFIG_DIGEST,
        "provider_id": PROVIDER_ID,
        "model_id": MODEL_ID,
        "provider_model_digest": PROVIDER_MODEL_DIGEST,
        "agent_budget": asdict(AGENT_BUDGET),
        "utility_prompt_digest": __import__(
            "pico.decision_plane.provider_utility", fromlist=["PROVIDER_UTILITY_PROMPT_DIGEST"]
        ).PROVIDER_UTILITY_PROMPT_DIGEST,
        "utility_inference_policy": UTILITY_POLICY,
        "utility_inference_policy_digest": UTILITY_POLICY_DIGEST,
        "arms": tuple(arm.value for arm in Arm),
        "repetitions": REPETITIONS,
        "planned_live_runs": PLANNED_LIVE_RUNS,
        "run_order": tuple(asdict(row) for row in RUN_ORDER),
        "run_order_digest": RUN_ORDER_DIGEST,
        "primary_correctness": PRIMARY_CORRECTNESS,
        "primary_efficiency": PRIMARY_EFFICIENCY,
        "secondary_metrics": SECONDARY_METRICS,
        "reasoning_accounting": REASONING_ACCOUNTING,
        "utility_accounting": UTILITY_ACCOUNTING,
        "run_isolation": RUN_ISOLATION,
        "early_stop_policy": EARLY_STOP_POLICY,
        "no_outcome_leakage": NO_OUTCOME_LEAKAGE,
        "benefit_criteria": asdict(BENEFIT_CRITERIA),
        "benefit_criteria_digest": canonical_digest(BENEFIT_CRITERIA),
        "provider_calls_in_freeze": 0,
        "agent_turns_in_freeze": 0,
    }
    payload["treatment_config_digest"] = canonical_digest(
        {
            "arms": payload["arms"],
            "provider_model_digest": PROVIDER_MODEL_DIGEST,
            "selector_config_digest": SELECTOR_CONFIG_DIGEST,
            "utility_inference_policy_digest": UTILITY_POLICY_DIGEST,
            "agent_budget": payload["agent_budget"],
            "run_isolation": RUN_ISOLATION,
        }
    )
    payload["semantic_digest"] = canonical_digest(payload)
    return payload


__all__ = [
    "CONTRACT_AUDIT",
    "EXPOSURE_SNAPSHOT_SCHEMA",
    "NEW_TASK",
    "NEW_TASK_ID",
    "PREDECESSOR_SUITE",
    "REPLACEMENT_REASON",
    "RETAINED_TASK_IDS",
    "RETIRED_TASK_ID",
    "RUN_ORDER",
    "RUN_ORDER_DIGEST",
    "SUITE_NAME",
    "SUITE_VERSION",
    "TASKS",
    "make_run_order",
    "suite_payload",
]
