"""Frozen offline design contract for the JEV.6 held-out v3 benchmark."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from pico.knowledge_evolution.identity import CANDIDATE_EVIDENCE_IDENTITY_SCHEMA

from .jev6_suite import (
    AGENT_BUDGET,
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
from .jev6_v2_suite import TASKS as V2_TASKS
from .schema import LiveTask

SUITE_NAME = "JEV.6 held-out Provider Utility benchmark v3"
SUITE_VERSION = "jev6-held-out-v3"
PREDECESSOR_SUITE = "jev6-held-out-v2"
REPLACEMENT_REASON = "int-02 live Agent exposure during invalid JEV.6V2B"
BASE_COMMIT = "49243eb838448091cdb5edd2b2d1cda82475a831"
RUN_ORDER_SEED = 63_211
EXPOSURE_LEDGER_SCHEMA = "pico.jev6-heldout-exposure-ledger.v2"
EXPOSURE_SNAPSHOT_SCHEMA = "pico.jev6-heldout-exposure-snapshot.v3"
SANDBOX_REQUIREMENT_SCHEMA = "pico.jev6-sandbox-requirement.v1"
REAL_SANDBOX_EVIDENCE_DIGEST = (
    "c6d806ee997da76079552349470c8e52fba89f427746198ad11af07d6fee4e16"
)

RETAINED_TASK_IDS = (
    "jev6-nav-01-trace-event-reader",
    "jev6v2-nav-02-plugin-contributions",
    "jev6-impl-01-safe-segment-bound",
    "jev6-impl-02-delivery-result-flags",
    "jev6-debug-01-bm25-parameters",
    "jev6-debug-02-decision-receipt-identity",
    "jev6-int-01-public-turn-events",
)
RETIRED_TASK_ID = "jev6-int-02-resolve-available-skill"
NEW_TASK_ID = "jev6v3-int-02-provider-resolution"

_V2_BY_ID = {task.task_id: task for task in V2_TASKS}
NEW_TASK = LiveTask(
    NEW_TASK_ID,
    Category.INTEGRATION_API_COMPOSITION.value,
    "In pico/providers/registry.py, add a public resolve_provider_spec function that composes the existing registry lookups. It must prefer an exact provider_name match, then gateway or local detection from provider_name, api_key, or api_base, then model matching, and return None when nothing resolves. It must not construct a Provider or mutate the registry. Add focused tests covering every precedence tier and the unresolved case.",
    "jev6v3-v-provider-resolution",
    verifier_digest("jev6v3-v-provider-resolution"),
    (),
    "Composes stable provider registry lookup boundaries without constructing providers.",
)
TASKS = tuple(_V2_BY_ID[task_id] for task_id in RETAINED_TASK_IDS) + (NEW_TASK,)

CONTRACT_AUDIT = tuple(
    ContractAudit(
        task.task_id,
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
        CONTRACT_STATUS,
    )
    for task in TASKS
)

SANDBOX_REQUIREMENT = {
    "schema": SANDBOX_REQUIREMENT_SCHEMA,
    "schema_version": 1,
    "backend": "boxlite",
    "allow_net": False,
    "extra_volumes": (),
    "rejected_backends": ("none", "auto"),
    "required_smoke_result": "PASS_REAL_BOXLITE",
    "real_smoke_evidence_schema": "pico.jev6-benchmark-sandbox-smoke.v1",
    "real_smoke_evidence_digest": REAL_SANDBOX_EVIDENCE_DIGEST,
    "registry_is_bootstrap_plumbing": True,
}
SANDBOX_REQUIREMENT_DIGEST = canonical_digest(SANDBOX_REQUIREMENT)


def validate_sandbox_requirement(
    *,
    backend: str,
    allow_net: bool | list[str],
    extra_volumes: list[list[str]] | tuple[object, ...],
    smoke_result: str,
) -> None:
    """Fail closed unless future execution matches the frozen real sandbox gate."""

    if backend != "boxlite":
        raise ValueError("JEV.6 V3 requires explicit backend='boxlite'")
    if allow_net is not False:
        raise ValueError("JEV.6 V3 requires allow_net=false")
    if extra_volumes:
        raise ValueError("JEV.6 V3 forbids extra sandbox volumes")
    if smoke_result != "PASS_REAL_BOXLITE":
        raise ValueError("JEV.6 V3 requires PASS_REAL_BOXLITE evidence")


def make_run_order(seed: int = RUN_ORDER_SEED) -> tuple[PlannedRun, ...]:
    """Create a deterministic paired, interleaved v3 order."""

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
        offset = (offset + 5) % len(TASKS)
    return tuple(rows)


RUN_ORDER = make_run_order()
RUN_ORDER_DIGEST = canonical_digest(RUN_ORDER)


def suite_payload(
    *,
    selector_audit: dict[str, Any],
    mechanical_audit: dict[str, Any],
    exposure_snapshot: dict[str, Any],
) -> dict[str, Any]:
    from .jev6_v3_fixtures import reference_digests
    from .jev6_v3_verifiers import verifier_set_digest

    references = reference_digests()
    canonical_sequences = {
        task_id: tuple(
            (item["candidate_id"], item["semantic_digest"])
            for item in row["ordered_selected_identities"]
        )
        for task_id, row in selector_audit["tasks"].items()
    }
    payload = {
        "schema": "pico.jev6-heldout-suite.v3",
        "schema_version": 3,
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
            "pico.decision_plane.provider_utility",
            fromlist=["PROVIDER_UTILITY_PROMPT_DIGEST"],
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
        "sandbox_requirement": SANDBOX_REQUIREMENT,
        "sandbox_requirement_digest": SANDBOX_REQUIREMENT_DIGEST,
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
            "sandbox_requirement_digest": SANDBOX_REQUIREMENT_DIGEST,
        }
    )
    payload["semantic_digest"] = canonical_digest(payload)
    return payload


__all__ = [
    "BASE_COMMIT",
    "CONTRACT_AUDIT",
    "EXPOSURE_LEDGER_SCHEMA",
    "EXPOSURE_SNAPSHOT_SCHEMA",
    "NEW_TASK",
    "NEW_TASK_ID",
    "PREDECESSOR_SUITE",
    "REAL_SANDBOX_EVIDENCE_DIGEST",
    "REPLACEMENT_REASON",
    "RETAINED_TASK_IDS",
    "RETIRED_TASK_ID",
    "RUN_ORDER",
    "RUN_ORDER_DIGEST",
    "SANDBOX_REQUIREMENT",
    "SANDBOX_REQUIREMENT_DIGEST",
    "SUITE_NAME",
    "SUITE_VERSION",
    "TASKS",
    "make_run_order",
    "suite_payload",
    "validate_sandbox_requirement",
]
