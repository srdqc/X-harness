from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live import jev4_pilot, jev5a_probe
from benchmarks.picobench.packs.knowledge_evolution_live.jev3b_pack import (
    PACK_CASES as JEV3B_CASES,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_fixtures import reference_digests
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_freeze import (
    audit_mechanical_solvability,
    evaluate_selector_audit,
    historical_campaign_classifications,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    BENEFIT_CRITERIA,
    CONTRACT_AUDIT,
    PLANNED_LIVE_RUNS,
    PRIMARY_EFFICIENCY,
    REASONING_ACCOUNTING,
    REPETITIONS,
    RUN_ISOLATION,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    SUITE_VERSION,
    TASKS,
    UTILITY_ACCOUNTING,
    UTILITY_POLICY,
    Arm,
    Category,
    suite_payload,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_verifiers import (
    SPECS,
    verifier_set_digest,
)
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import (
    EXPLORATORY_TASKS,
    OFFICIAL_TASKS_V1,
    OFFICIAL_TASKS_V2,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = (
    ROOT
    / "benchmarks/picobench/packs/knowledge_evolution_live/jev6_frozen_manifest.json"
)


@pytest.fixture(scope="module")
def selector_audit() -> dict[str, object]:
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="j6s-", dir=ROOT / ".tmp") as state:
        yield evaluate_selector_audit(state_root=Path(state), workspace=ROOT)


def test_suite_has_exactly_eight_balanced_new_tasks() -> None:
    assert SUITE_VERSION == "jev6-held-out-v1"
    assert len(TASKS) == len({task.task_id for task in TASKS}) == 8
    assert Counter(task.category for task in TASKS) == {
        category.value: 2 for category in Category
    }
    historical_tasks = (*EXPLORATORY_TASKS, *OFFICIAL_TASKS_V1, *OFFICIAL_TASKS_V2)
    historical_ids = {task.task_id for task in historical_tasks} | {
        task.task_id for task in jev4_pilot.TASKS
    }
    historical_prompts = {task.prompt for task in historical_tasks} | {
        task.prompt for task in jev4_pilot.TASKS
    }
    assert not historical_ids.intersection(task.task_id for task in TASKS)
    assert not historical_prompts.intersection(task.prompt for task in TASKS)
    old_case_text = json.dumps(JEV3B_CASES, default=str) + json.dumps(
        jev5a_probe.CASES, default=str
    )
    assert all(task.prompt not in old_case_text for task in TASKS)


def test_prompts_are_blind_to_verifiers_candidates_and_utility_expectations() -> None:
    corpus_only_terms = {
        "immutable-record-conventions",
        "phase-test-registration",
        "scoped-skill-flow",
        "structured-evidence-first",
        "fail-closed-recovery",
        "portable-artifact-paths",
        "KEEP",
        "ABSTAIN",
        "UNCERTAIN",
        "expected_direction",
        "verifier_id",
    }
    for task in TASKS:
        assert all(term not in task.prompt for term in corpus_only_terms)
        spec = next(spec for spec in SPECS if spec.verifier_id == task.verifier_id)
        assert spec.semantic_probe not in task.prompt


def test_contract_audit_and_independent_verifiers_cover_every_task() -> None:
    assert len(CONTRACT_AUDIT) == 8
    assert {row.task_id for row in CONTRACT_AUDIT} == {task.task_id for task in TASKS}
    assert all(row.status == "EXPLICITLY_SUPPORTED" for row in CONTRACT_AUDIT)
    assert {spec.verifier_id for spec in SPECS} == {task.verifier_id for task in TASKS}
    assert all(spec.run_after_terminal for spec in SPECS)
    assert all(task.verifier_digest == next(
        spec.digest for spec in SPECS if spec.verifier_id == task.verifier_id
    ) for task in TASKS)


def test_mechanical_references_pass_eight_of_eight() -> None:
    results = audit_mechanical_solvability(ROOT)
    assert len(results) == 8
    assert all(result.passed and not result.findings for result in results)
    assert tuple((item.task_id, item.reference_digest) for item in results) == reference_digests()


def test_selector_audit_is_deterministic_and_reports_actual_distribution(
    selector_audit: dict[str, object],
) -> None:
    with tempfile.TemporaryDirectory(prefix="j6r-", dir=ROOT / ".tmp") as state:
        repeated = evaluate_selector_audit(state_root=Path(state), workspace=ROOT)
    assert repeated == selector_audit
    counts = selector_audit["task_selected_counts"]
    assert counts == {
        "jev6-nav-01-trace-event-reader": 1,
        "jev6-nav-02-skill-variants": 2,
        "jev6-impl-01-safe-segment-bound": 0,
        "jev6-impl-02-delivery-result-flags": 0,
        "jev6-debug-01-bm25-parameters": 0,
        "jev6-debug-02-decision-receipt-identity": 2,
        "jev6-int-01-public-turn-events": 1,
        "jev6-int-02-resolve-available-skill": 1,
    }
    assert selector_audit["zero_selection_task_count"] == 3
    assert selector_audit["one_selection_task_count"] == 3
    assert selector_audit["many_selection_task_count"] == 2
    assert selector_audit["selection_distribution_policy"] == (
        "reported_as_observed_without_prompt_rewriting"
    )


def test_two_arms_three_repetitions_and_counterbalanced_48_run_order() -> None:
    assert tuple(Arm) == (
        Arm.TASK_RELEVANCE_V1,
        Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY,
    )
    assert REPETITIONS == 3
    assert PLANNED_LIVE_RUNS == len(RUN_ORDER) == 48
    identities = {(row.task_id, row.arm, row.repetition) for row in RUN_ORDER}
    assert len(identities) == 48
    assert [row.order for row in RUN_ORDER] == list(range(1, 49))
    first_arms = Counter(RUN_ORDER[index].arm for index in range(0, 48, 2))
    assert first_arms == {arm.value: 12 for arm in Arm}
    for index in range(0, 48, 2):
        left, right = RUN_ORDER[index : index + 2]
        assert (left.task_id, left.repetition) == (right.task_id, right.repetition)
        assert {left.arm, right.arm} == {arm.value for arm in Arm}
    assert RUN_ORDER_DIGEST == canonical_digest(RUN_ORDER)


def test_cost_reasoning_fallback_and_classification_policies_are_frozen() -> None:
    assert PRIMARY_EFFICIENCY == (
        "combined_provider_logical_calls",
        "combined_input_tokens",
        "combined_output_tokens",
        "tool_calls",
        "repeated_repository_reads",
        "turn_latency_ms",
    )
    assert UTILITY_POLICY["reasoning_effort"] == "none"
    assert UTILITY_POLICY["max_output_tokens"] == 1024
    assert UTILITY_POLICY["timeout_seconds"] == 15.0
    assert UTILITY_POLICY["effective_policy"] == {
        "KEEP": "KEEP",
        "ABSTAIN": "ABSTAIN",
        "UNCERTAIN": "KEEP",
    }
    assert "reasoning_tokens" in REASONING_ACCOUNTING
    assert "reasoning_output_ratio" in REASONING_ACCOUNTING
    assert UTILITY_ACCOUNTING["combined_cost_includes_utility"] is True
    assert UTILITY_ACCOUNTING["rerun_utility_on_failure"] is False
    assert UTILITY_ACCOUNTING["replace_agent_run_on_failure"] is False
    assert "fallback used, zero valid decisions" in UTILITY_ACCOUNTING[
        "utility_decision_unavailable"
    ]
    assert all(RUN_ISOLATION.values())
    assert ">=15%" in " ".join(BENEFIT_CRITERIA.beneficial)
    assert "C verified successes lower than B" in BENEFIT_CRITERIA.regressive


def test_frozen_manifest_matches_all_anti_tuning_digests(
    selector_audit: dict[str, object],
) -> None:
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    payload = suite_payload(selector_audit)
    digests = frozen["anti_tuning_digests"]
    assert digests["suite"] == payload["semantic_digest"]
    assert digests["task_set"] == payload["task_set_digest"]
    assert digests["verifier_set"] == verifier_set_digest() == payload["verifier_set_digest"]
    assert digests["fixtures"] == payload["fixture_set_digest"]
    assert digests["contract_audit"] == payload["contract_audit_digest"]
    assert digests["selector_audit"] == payload["selector_audit_digest"]
    assert digests["run_order"] == payload["run_order_digest"]
    assert digests["classification_policy"] == payload["benefit_criteria_digest"]
    assert frozen["provider_calls"] == 0
    assert frozen["live_outcomes_present"] is False


def test_historical_campaigns_remain_unchanged() -> None:
    paths = tuple(
        ROOT / ".p3r" / campaign / "reduced.json"
        for campaign in (
            "jev4-9168e909e0c47dfb",
            "jev4r-3f7c246afe462f5e",
            "jev4r2-868dd3997095b407",
            "jev4r3-384e7aefbe9cca17",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    assert historical_campaign_classifications(ROOT) == {
        "jev4-9168e909e0c47dfb": "HOLD_AND_FORENSIC",
        "jev4r-3f7c246afe462f5e": "HOLD_INFRA",
        "jev4r2-868dd3997095b407": "HOLD_INFRA",
        "jev4r3-384e7aefbe9cca17": "INFRA_PASS_BEHAVIOR_MIXED",
    }
    after = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    assert before == after
