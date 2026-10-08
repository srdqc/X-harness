"""Frozen offline design contract for the final JEV.6 held-out v4 suite."""

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
from .jev6_v3_suite import TASKS as V3_TASKS
from .schema import LiveTask

SUITE_NAME = "JEV.6 final successor held-out Provider Utility benchmark v4"
SUITE_VERSION = "jev6-held-out-v4"
SUITE_SCHEMA = "pico.jev6-heldout-suite.v4"
PREDECESSOR_SUITE = "jev6-held-out-v3"
REPLACEMENT_REASON = "three V3 tasks received live Agent exposure before ordering-drift invalidation"
BASE_COMMIT = "473304a788acbce8be02445893435c757dd3455c"
RUN_ORDER_SEED = 64_409
EXPOSURE_LEDGER_SCHEMA = "pico.jev6-heldout-exposure-ledger.v3"
EXPOSURE_SNAPSHOT_SCHEMA = "pico.jev6-heldout-exposure-snapshot.v4"
SANDBOX_REQUIREMENT_SCHEMA = "pico.jev6-sandbox-requirement.v2"
REAL_SANDBOX_EVIDENCE_DIGEST = (
    "c6d806ee997da76079552349470c8e52fba89f427746198ad11af07d6fee4e16"
)
PROVIDER_CONFIG_SOURCE = "~/.pico/config.json"
SANDBOX_CONFIG_SOURCE = "frozen_jev6_v4_infrastructure"

RETAINED_TASK_IDS = (
    "jev6-nav-01-trace-event-reader",
    "jev6v2-nav-02-plugin-contributions",
    "jev6-impl-01-safe-segment-bound",
    "jev6-int-01-public-turn-events",
    "jev6v3-int-02-provider-resolution",
)
RETIRED_EXPOSED_TASK_IDS = (
    "jev6-impl-02-delivery-result-flags",
    "jev6-debug-01-bm25-parameters",
    "jev6-debug-02-decision-receipt-identity",
)

NEW_TASKS = (
    LiveTask(
        "jev6v4-impl-02-replace-provider-models",
        Category.IMPLEMENTATION.value,
        "Add a public replace_provider_models function in pico/config/update_providers.py. It must replace one provider's curated models list atomically, trim model names, reject non-string or empty names with ValueError, remove duplicates while preserving first occurrence order, validate the complete provider section before writing, return the normalized list, preserve unrelated provider fields, raise the existing unknown-provider KeyError, and add focused tests.",
        "jev6v4-v-replace-provider-models",
        verifier_digest("jev6v4-v-replace-provider-models"),
        (),
        "Adds a bounded atomic Provider configuration operation without network activity.",
    ),
    LiveTask(
        "jev6v4-debug-01-capability-token-payload",
        Category.DEBUGGING_VALIDATION.value,
        "Harden pico/auth/capability_token.py so verify_token fails closed by returning None for a correctly signed JSON payload that is not an object and for malformed or non-ASCII encoded token payloads. Preserve valid token verification, expiry handling, and constant-time signature comparison, and add focused regression tests proving these inputs never escape as parser exceptions.",
        "jev6v4-v-capability-token-payload",
        verifier_digest("jev6v4-v-capability-token-payload"),
        (),
        "Repairs a deterministic fail-closed parsing boundary unrelated to prior receipt validation.",
    ),
    LiveTask(
        "jev6v4-debug-02-portable-media-name",
        Category.DEBUGGING_VALIDATION.value,
        "Harden pico/channels/media.py safe_name so both forward-slash and backslash path components are stripped on every host platform. Empty names and terminal dot or dot-dot components must return 'file'; ordinary basenames must remain unchanged. Preserve save_media_bytes content-hash naming and add focused cross-platform path regression tests.",
        "jev6v4-v-portable-media-name",
        verifier_digest("jev6v4-v-portable-media-name"),
        (),
        "Repairs platform-independent filename sanitation without changing media persistence semantics.",
    ),
)
NEW_TASK_IDS = tuple(task.task_id for task in NEW_TASKS)

_V3_BY_ID = {task.task_id: task for task in V3_TASKS}
TASKS = (
    _V3_BY_ID[RETAINED_TASK_IDS[0]],
    _V3_BY_ID[RETAINED_TASK_IDS[1]],
    _V3_BY_ID[RETAINED_TASK_IDS[2]],
    NEW_TASKS[0],
    NEW_TASKS[1],
    NEW_TASKS[2],
    _V3_BY_ID[RETAINED_TASK_IDS[3]],
    _V3_BY_ID[RETAINED_TASK_IDS[4]],
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

SANDBOX_REQUIREMENT = {
    "schema": SANDBOX_REQUIREMENT_SCHEMA,
    "schema_version": 2,
    "source": SANDBOX_CONFIG_SOURCE,
    "backend": "boxlite",
    "boxlite_version": "0.9.5",
    "allow_net": False,
    "extra_volumes": (),
    "image_search_registry": "docker.m.daocloud.io",
    "native_runtime_home": "/tmp/pico-jev6v4-boxlite",
    "runtime_home_filesystem": "native_linux",
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
    boxlite_version: str,
    allow_net: bool | list[str],
    extra_volumes: list[list[str]] | tuple[object, ...],
    image_search_registry: str,
    native_runtime_home: str,
    smoke_result: str,
) -> None:
    """Fail closed unless execution matches the frozen V4 sandbox identity."""

    expected = SANDBOX_REQUIREMENT
    actual = {
        "backend": backend,
        "boxlite_version": boxlite_version,
        "allow_net": allow_net,
        "extra_volumes": tuple(extra_volumes),
        "image_search_registry": image_search_registry,
        "native_runtime_home": native_runtime_home,
        "smoke_result": smoke_result,
    }
    required = {
        "backend": expected["backend"],
        "boxlite_version": expected["boxlite_version"],
        "allow_net": expected["allow_net"],
        "extra_volumes": expected["extra_volumes"],
        "image_search_registry": expected["image_search_registry"],
        "native_runtime_home": expected["native_runtime_home"],
        "smoke_result": expected["required_smoke_result"],
    }
    if actual != required:
        raise ValueError("JEV.6 V4 sandbox identity mismatch")


def make_run_order(seed: int = RUN_ORDER_SEED) -> tuple[PlannedRun, ...]:
    """Create the new deterministic paired, counterbalanced V4 order."""

    rows: list[PlannedRun] = []
    arms = tuple(Arm)
    offset = seed % len(TASKS)
    for repetition in range(1, REPETITIONS + 1):
        task_order = TASKS[offset:] + TASKS[:offset]
        for task_index, task in enumerate(task_order):
            first = (task_index + repetition + seed) % 2
            for arm in arms[first:] + arms[:first]:
                rows.append(
                    PlannedRun(
                        f"{task.task_id}-{arm.value}-rep{repetition}",
                        task.task_id,
                        arm.value,
                        repetition,
                        len(rows) + 1,
                    )
                )
        offset = (offset + 7) % len(TASKS)
    return tuple(rows)


RUN_ORDER = make_run_order()
RUN_ORDER_DIGEST = canonical_digest(RUN_ORDER)


def suite_payload(
    *,
    selector_audit: dict[str, Any],
    mechanical_audit: dict[str, Any],
    exposure_snapshot: dict[str, Any],
) -> dict[str, Any]:
    from .jev6_v4_fixtures import reference_digests
    from .jev6_v4_verifiers import verifier_set_digest

    references = reference_digests()
    canonical_sequences = {
        task_id: tuple(
            (item["candidate_id"], item["semantic_digest"])
            for item in row["ordered_selected_identities"]
        )
        for task_id, row in selector_audit["tasks"].items()
    }
    payload = {
        "schema": SUITE_SCHEMA,
        "schema_version": 4,
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
        "canonical_corpus_identity_digest": selector_audit["canonical_corpus_identity_digest"],
        "selector_config": SELECTOR_CONFIG,
        "selector_config_digest": SELECTOR_CONFIG_DIGEST,
        "provider_id": PROVIDER_ID,
        "model_id": MODEL_ID,
        "provider_model_digest": PROVIDER_MODEL_DIGEST,
        "provider_config_source": PROVIDER_CONFIG_SOURCE,
        "sandbox_config_source": SANDBOX_CONFIG_SOURCE,
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
    "NEW_TASKS",
    "NEW_TASK_IDS",
    "PREDECESSOR_SUITE",
    "PROVIDER_CONFIG_SOURCE",
    "REAL_SANDBOX_EVIDENCE_DIGEST",
    "REPLACEMENT_REASON",
    "RETAINED_TASK_IDS",
    "RETIRED_EXPOSED_TASK_IDS",
    "RUN_ORDER",
    "RUN_ORDER_DIGEST",
    "SANDBOX_CONFIG_SOURCE",
    "SANDBOX_REQUIREMENT",
    "SANDBOX_REQUIREMENT_DIGEST",
    "SUITE_NAME",
    "SUITE_SCHEMA",
    "SUITE_VERSION",
    "TASKS",
    "make_run_order",
    "suite_payload",
    "validate_sandbox_requirement",
]
