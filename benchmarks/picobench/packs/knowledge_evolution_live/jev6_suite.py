"""Frozen offline design contract for the JEV.6 held-out utility benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_PROMPT_DIGEST,
    UtilityInferencePolicy,
)
from pico.decision_plane.utility import JEV_UTILITY_RESPONSE_SCHEMA, JEV_UTILITY_SCHEMA_VERSION
from pico.knowledge_evolution.relevance import SELECTOR_VERSION

from .knowledge import corpus_digest
from .schema import LiveTask, RuntimeBudget

SUITE_NAME = "JEV.6 held-out Provider Utility benchmark"
SUITE_VERSION = "jev6-held-out-v1"
BASE_COMMIT = "8adee66f691d183695734a4373c62da1b514235a"
PROVIDER_ID = "deepseek"
MODEL_ID = "deepseek/deepseek-v4-flash"
UTILITY_TIMEOUT_SECONDS = 15.0
UTILITY_MAX_OUTPUT_TOKENS = 1024
REPETITIONS = 3
PLANNED_LIVE_RUNS = 48
RUN_ORDER_SEED = 61_203
CONTRACT_STATUS = "EXPLICITLY_SUPPORTED"


class Arm(StrEnum):
    TASK_RELEVANCE_V1 = "task_relevance_v1"
    TASK_RELEVANCE_V1_PROVIDER_UTILITY = "task_relevance_v1_provider_utility"


class Category(StrEnum):
    NAVIGATION_RETRIEVAL = "navigation_retrieval"
    IMPLEMENTATION = "implementation"
    DEBUGGING_VALIDATION = "debugging_validation"
    INTEGRATION_API_COMPOSITION = "integration_api_composition"


@dataclass(frozen=True)
class ContractAudit:
    task_id: str
    visible_contract: tuple[str, ...]
    verifier_contract: tuple[str, ...]
    status: str = CONTRACT_STATUS


@dataclass(frozen=True)
class BenefitCriteria:
    beneficial: tuple[str, ...]
    neutral: tuple[str, ...]
    regressive: tuple[str, ...]
    invalid: tuple[str, ...]


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    task_id: str
    arm: str
    repetition: int
    order: int


def verifier_digest(verifier_id: str) -> str:
    return canonical_digest({"verifier_id": verifier_id, "version": 1})


def _task(
    task_id: str,
    category: Category,
    prompt: str,
    verifier_id: str,
    expected: tuple[str, ...],
    rationale: str,
) -> LiveTask:
    return LiveTask(
        task_id,
        category.value,
        prompt,
        verifier_id,
        verifier_digest(verifier_id),
        expected,
        rationale,
    )


TASKS = (
    _task(
        "jev6-nav-01-trace-event-reader",
        Category.NAVIGATION_RETRIEVAL,
        "Add a public TraceStore.read_events method that reads the active audit event JSONL in file order and accepts optional turn_id and event_type filters that compose. Return an empty tuple when the log is absent or no records match. Reject malformed JSON lines or non-object records with ValueError instead of silently skipping them, and add focused tests.",
        "jev6-v-trace-event-reader",
        ("memory_fact", "experience"),
        "Requires locating the tracing persistence owner and its event identity fields.",
    ),
    _task(
        "jev6-nav-02-skill-variants",
        Category.NAVIGATION_RETRIEVAL,
        "Add a public SkillRegistry.list_variants(name) method that returns every cached physical SkillMeta with that display name, ordered by source then stable ID. It must preserve shadowed cross-source variants, return an empty tuple for no match, avoid rescanning when the cache is clean, and add focused tests.",
        "jev6-v-skill-variants",
        ("memory_fact",),
        "Requires following compound skill identity and cache ownership rather than name-only lookup.",
    ),
    _task(
        "jev6-impl-01-safe-segment-bound",
        Category.IMPLEMENTATION,
        "Extend pico.tracing.store.safe_segment with a keyword-only max_length parameter. Keep the existing default length of 80, require a positive integer, apply the bound after normalization and fallback selection, and add focused compatibility and invalid-input tests.",
        "jev6-v-safe-segment-bound",
        ("experience",),
        "A bounded path-segment extension with a compatibility requirement.",
    ),
    _task(
        "jev6-impl-02-delivery-result-flags",
        Category.IMPLEMENTATION,
        "Add derived DeliveryResult.succeeded and DeliveryResult.retry_exhausted boolean properties. succeeded is true only for outcome 'delivered'; retry_exhausted is true only for outcome 'dropped' with attempts greater than one. Do not change dataclass fields or delivery routing, and add focused tests covering both truth tables.",
        "jev6-v-delivery-result-flags",
        (),
        "A small runtime-domain extension intentionally unrelated to the frozen knowledge corpus.",
    ),
    _task(
        "jev6-debug-01-bm25-parameters",
        Category.DEBUGGING_VALIDATION,
        "Harden BM25Okapi construction so k1 must be finite and greater than zero and b must be finite and within [0, 1]. Reject booleans, NaN, infinities, and out-of-range values with ValueError while preserving valid scoring behavior. Add focused positive and negative tests.",
        "jev6-v-bm25-parameters",
        ("experience",),
        "Exercises deterministic validation in a retrieval utility without changing ranking semantics.",
    ),
    _task(
        "jev6-debug-02-decision-receipt-identity",
        Category.DEBUGGING_VALIDATION,
        "Harden DecisionReceipt validation so its five digest fields are exactly 64 lowercase hexadecimal characters and confidence candidate IDs are unique. Preserve existing valid receipts and metadata, reject malformed values with ValueError, and add focused tests.",
        "jev6-v-decision-receipt-identity",
        ("memory_fact", "experience"),
        "Targets fail-closed integrity at the durable decision-evidence boundary.",
    ),
    _task(
        "jev6-int-01-public-turn-events",
        Category.INTEGRATION_API_COMPOSITION,
        "Expose a public read_turn_events(store, turn_id) helper from pico.tracing. It must delegate to TraceStore.read_events with the supplied turn_id, preserve event file order, return an empty tuple for no matches, and be exported through pico.tracing.__all__. Add focused integration tests; malformed event records must still fail closed through the store reader.",
        "jev6-v-public-turn-events",
        ("memory_fact", "experience"),
        "Composes tracing storage, filtering, and the public package boundary.",
    ),
    _task(
        "jev6-int-02-resolve-available-skill",
        Category.INTEGRATION_API_COMPOSITION,
        "Add SkillRegistry.resolve_available(name, source=None), composing existing exact/priority lookup with requirement checking. Return the resolved SkillMeta only when it exists and all declared requirements are available; otherwise return None. Preserve source-specific lookup and cache behavior, and add focused tests.",
        "jev6-v-resolve-available-skill",
        ("memory_fact",),
        "Composes existing Skill registry lookup and availability ownership without bypassing either.",
    ),
)


CONTRACT_AUDIT = tuple(
    ContractAudit(
        task.task_id,
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
        tuple(part.strip() for part in task.prompt.split(".") if part.strip()),
    )
    for task in TASKS
)


BENEFIT_CRITERIA = BenefitCriteria(
    beneficial=(
        "infrastructure/fairness/safety valid",
        "C verified successes are at least B verified successes",
        "no repeated clear correctness regression attributable to C",
        "at least one primary combined efficiency metric has >=15% median paired task-level improvement",
        "no unexplained >20% median regression across multiple other primary metrics",
        "utility contract and fallback behavior acceptable",
    ),
    neutral=(
        "correctness noninferior",
        "infrastructure valid",
        "no primary combined efficiency metric reaches >=15% improvement",
    ),
    regressive=(
        "C verified successes lower than B",
        "repeated B-success/C-failure pattern or clear utility correctness harm",
        "multiple primary combined metrics materially regress without compensating benefit",
    ),
    invalid=("fairness, infrastructure, safety, or utility contract invalidates comparison",),
)


PRIMARY_CORRECTNESS = "independent verifier pass rate: B successes/24 and C successes/24; per task 0..3"
PRIMARY_EFFICIENCY = (
    "combined_provider_logical_calls",
    "combined_input_tokens",
    "combined_output_tokens",
    "tool_calls",
    "repeated_repository_reads",
    "turn_latency_ms",
)
SECONDARY_METRICS = (
    "agent_iterations",
    "first_edit_iteration",
    "unique_repository_reads",
    "failed_tool_attempts",
    "utility_calls",
    "utility_attempts",
    "utility_latency_ms",
    "utility_reasoning_tokens",
    "utility_raw_choices",
    "candidate_retention_ratio",
    "fallback_rate",
)
RUN_ISOLATION = {
    "fresh_process": True,
    "fresh_worktree": True,
    "fresh_session": True,
    "fresh_turn": True,
    "fresh_state": True,
    "fresh_trace": True,
    "fresh_temp": True,
    "fresh_pip_environment": True,
}
UTILITY_ACCOUNTING = {
    "combined_cost_includes_utility": True,
    "fallback_target": "task_relevance_v1_selected_set",
    "rerun_utility_on_failure": False,
    "replace_agent_run_on_failure": False,
    "persist_failure_category": True,
    "all_explicit_abstain": "valid complete explicit ABSTAIN decisions with no fallback",
    "utility_decision_unavailable": "selected candidates exist, fallback used, zero valid decisions",
    "zero_selection_control": "B selected=0; C selected=0; C utility calls=0",
}
REASONING_ACCOUNTING = (
    "reasoning_mode_requested",
    "reasoning_tokens",
    "output_tokens",
    "visible_output_tokens",
    "reasoning_output_ratio",
    "finish_reason",
    "payload_outcome",
)
EARLY_STOP_POLICY = {
    "outcome_based_stopping": False,
    "allowed": (
        "infrastructure_failure",
        "shared_environment_drift",
        "systematic_contract_corruption",
        "safety_or_fairness_invalidity",
    ),
}
NO_OUTCOME_LEAKAGE = (
    "task_verifier",
    "expected_patch",
    "historical_candidate_outcome",
    "previous_repetition_results",
    "arm_results",
    "success_failure_labels",
)


def make_run_order(seed: int = RUN_ORDER_SEED) -> tuple[PlannedRun, ...]:
    """Interleave paired arms while counterbalancing first arm by task/repetition."""
    rows: list[PlannedRun] = []
    arms = tuple(Arm)
    offset = seed % 2
    for repetition in range(1, REPETITIONS + 1):
        task_order = TASKS[offset:] + TASKS[:offset]
        for task_index, task in enumerate(task_order):
            first = (task_index + repetition + offset) % 2
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
    return tuple(rows)


RUN_ORDER = make_run_order()
RUN_ORDER_DIGEST = canonical_digest(RUN_ORDER)
SELECTOR_CONFIG = {"selection_mode": "task_relevance_v1", "selector_version": SELECTOR_VERSION}
SELECTOR_CONFIG_DIGEST = canonical_digest(SELECTOR_CONFIG)
UTILITY_POLICY = {
    "typed_choices": ("KEEP", "ABSTAIN", "UNCERTAIN"),
    "effective_policy": {"KEEP": "KEEP", "ABSTAIN": "ABSTAIN", "UNCERTAIN": "KEEP"},
    "reasoning_effort": UtilityInferencePolicy().reasoning_effort,
    "structured_output": UtilityInferencePolicy().response_format,
    "strict_local_validation": True,
    "response_schema": JEV_UTILITY_RESPONSE_SCHEMA,
    "response_schema_version": JEV_UTILITY_SCHEMA_VERSION,
    "max_output_tokens": UTILITY_MAX_OUTPUT_TOKENS,
    "timeout_seconds": UTILITY_TIMEOUT_SECONDS,
}
UTILITY_POLICY_DIGEST = canonical_digest(UTILITY_POLICY)
PROVIDER_MODEL_DIGEST = canonical_digest(
    {"provider": PROVIDER_ID, "model": MODEL_ID, "main_agent": True, "utility": True}
)
AGENT_BUDGET = RuntimeBudget()


def suite_payload(selector_audit: dict[str, Any]) -> dict[str, Any]:
    references = __import__(
        "benchmarks.picobench.packs.knowledge_evolution_live.jev6_fixtures",
        fromlist=["reference_digests"],
    ).reference_digests()
    verifier_set = tuple((task.verifier_id, task.verifier_digest) for task in TASKS)
    payload = {
        "schema": "pico.jev6-heldout-suite.v1",
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "base_commit": BASE_COMMIT,
        "tasks": tuple(asdict(task) | {"prompt_digest": task.prompt_digest} for task in TASKS),
        "task_set_digest": canonical_digest(TASKS),
        "verifier_set": verifier_set,
        "verifier_set_digest": canonical_digest(verifier_set),
        "reference_fixtures": references,
        "fixture_set_digest": canonical_digest(references),
        "contract_audit": tuple(asdict(row) for row in CONTRACT_AUDIT),
        "contract_audit_digest": canonical_digest(CONTRACT_AUDIT),
        "selector_audit": selector_audit,
        "selector_audit_digest": canonical_digest(selector_audit),
        "corpus_digest": corpus_digest(),
        "selector_config": SELECTOR_CONFIG,
        "selector_config_digest": SELECTOR_CONFIG_DIGEST,
        "provider_id": PROVIDER_ID,
        "model_id": MODEL_ID,
        "provider_model_digest": PROVIDER_MODEL_DIGEST,
        "agent_budget": asdict(AGENT_BUDGET),
        "utility_prompt_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
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
        "per_task_reduction": "three-repetition medians per task/arm; success counts remain integer",
        "suite_reduction": "paired task-level C median minus B median across eight tasks",
        "benefit_criteria": asdict(BENEFIT_CRITERIA),
        "benefit_criteria_digest": canonical_digest(BENEFIT_CRITERIA),
        "no_adaptive_stopping": True,
        "replacement_runs": False,
        "provider_calls_in_freeze": 0,
    }
    anti_tuning = {
        key: payload[key]
        for key in (
            "task_set_digest",
            "verifier_set_digest",
            "fixture_set_digest",
            "contract_audit_digest",
            "selector_audit_digest",
            "corpus_digest",
            "selector_config_digest",
            "utility_prompt_digest",
            "utility_inference_policy_digest",
            "provider_model_digest",
            "run_order_digest",
            "benefit_criteria_digest",
        )
    }
    payload["anti_tuning_digests"] = anti_tuning
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
    "AGENT_BUDGET",
    "Arm",
    "BASE_COMMIT",
    "BENEFIT_CRITERIA",
    "Category",
    "CONTRACT_AUDIT",
    "MODEL_ID",
    "PLANNED_LIVE_RUNS",
    "PROVIDER_ID",
    "REPETITIONS",
    "REASONING_ACCOUNTING",
    "RUN_ORDER",
    "RUN_ORDER_DIGEST",
    "RUN_ISOLATION",
    "SUITE_NAME",
    "SUITE_VERSION",
    "TASKS",
    "UTILITY_MAX_OUTPUT_TOKENS",
    "UTILITY_POLICY",
    "UTILITY_POLICY_DIGEST",
    "UTILITY_ACCOUNTING",
    "UTILITY_TIMEOUT_SECONDS",
    "make_run_order",
    "suite_payload",
    "verifier_digest",
]
