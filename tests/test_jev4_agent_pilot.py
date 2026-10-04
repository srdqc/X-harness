from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live import jev4_pilot as pilot
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import (
    EXPLORATORY_TASKS,
    OFFICIAL_TASKS_V1,
    OFFICIAL_TASKS_V2,
)


@pytest.fixture(scope="module")
def offline_preflight() -> dict[str, object]:
    repository = Path.cwd()
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    return pilot.preflight(repository, base_sha, "human:test")


def test_frozen_plan_has_three_tasks_three_arms_and_nine_interleaved_runs() -> None:
    plan = pilot.make_plan()
    assert len(pilot.TASKS) == 3
    assert len(tuple(pilot.Arm)) == 3
    assert len(plan) == 9
    assert Counter(item["task_id"] for item in plan) == Counter(
        {task.task_id: 3 for task in pilot.TASKS}
    )
    assert Counter(item["arm"] for item in plan) == Counter(
        {arm.value: 3 for arm in pilot.Arm}
    )
    assert tuple(item["arm"] for item in plan[:3]) != tuple(
        item["arm"] for item in plan[3:6]
    )


def test_primary_comparison_is_b_vs_c_and_a_is_reference_only() -> None:
    assert pilot.PRIMARY_COMPARISON == (
        pilot.Arm.TASK_RELEVANCE_V1.value,
        pilot.Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value,
    )
    assert pilot.Arm.NO_REUSE.value not in pilot.PRIMARY_COMPARISON


def test_tasks_do_not_reuse_old_p3r_ids_or_prompts() -> None:
    old = (*EXPLORATORY_TASKS, *OFFICIAL_TASKS_V1, *OFFICIAL_TASKS_V2)
    assert not {item.task_id for item in old} & {item.task_id for item in pilot.TASKS}
    assert not {item.prompt for item in old} & {item.prompt for item in pilot.TASKS}


def test_visible_prompts_do_not_leak_hidden_or_expected_decisions() -> None:
    forbidden = (
        "semantic_probe",
        "verifier",
        "reference fixture",
        "candidate_id",
        "expected utility",
        "arm winner",
        "keep / abstain / uncertain",
    )
    for task in pilot.TASKS:
        assert all(value not in task.prompt.casefold() for value in forbidden)


def test_prompt_verifier_contract_audit_and_mechanical_solvability(
    offline_preflight: dict[str, object],
) -> None:
    assert len(pilot.CONTRACT_AUDIT) == 3
    assert offline_preflight["contract_audit"] == "PASS"
    assert offline_preflight["mechanical_solvability"] == "3/3 PASS"


def test_frozen_selection_conditions_include_zero_control(
    offline_preflight: dict[str, object],
) -> None:
    selected = offline_preflight["relevance_selected_candidate_ids"]
    assert selected[pilot.TASKS[0].task_id]
    assert selected[pilot.TASKS[1].task_id]
    assert selected[pilot.TASKS[2].task_id] == ()
    assert offline_preflight["zero_selection_utility_calls_expected"] == 0


def test_combined_cost_includes_utility_without_adding_nested_latency() -> None:
    main = {
        "provider_logical_calls": {"value": 3, "availability": "available"},
        "provider_attempts": {"value": 4, "availability": "available"},
        "input_tokens": {"value": 100, "availability": "available"},
        "output_tokens": {"value": 20, "availability": "available"},
    }
    utility = {
        "utility_logical_calls": 1,
        "utility_provider_attempts": 2,
        "input_tokens": 30,
        "output_tokens": 5,
    }
    combined = pilot._combined_cost(main, utility)
    assert combined["provider_logical_calls"] == 4
    assert combined["provider_attempts"] == 6
    assert combined["input_tokens"] == 130
    assert combined["output_tokens"] == 25
    assert "not added" in combined["latency_semantics"]


def test_utility_projection_is_separate_and_preserves_raw_and_effective_choices() -> None:
    refs = {
        "relevance_selected_candidate_ids": ("c1", "c2"),
        "utility_decisions": (
            {"candidate_id": "c1", "decision": "uncertain", "effective_decision": "keep"},
            {"candidate_id": "c2", "decision": "abstain", "effective_decision": "abstain"},
        ),
        "utility_invoked_count": 1,
        "utility_provider_calls": 1,
        "utility_provider_attempts": 1,
        "utility_kept_candidate_ids": ("c1",),
        "utility_abstained_candidate_ids": ("c2",),
        "utility_fallback_count": 0,
        "utility_fallback_reasons": (),
        "utility_input_tokens": 10,
        "utility_output_tokens": 4,
        "utility_provider_latency_ms": 12.0,
        "utility_latency_ms": 13.0,
    }
    result = pilot._utility_projection(refs)
    assert result["raw_choice_counts"] == {"KEEP": 0, "ABSTAIN": 1, "UNCERTAIN": 1}
    assert result["effective_choice_counts"] == {"KEEP": 1, "ABSTAIN": 1}
    assert result["utility_logical_calls"] == 1


def test_no_result_based_replacement_surface_exists() -> None:
    parser = pilot.build_parser()
    help_text = parser.format_help()
    assert "replace" not in help_text
    assert "replacement_policy" in pilot._semantic_payload_keys()


def test_campaign_semantic_inputs_and_plan_are_deterministic(
    offline_preflight: dict[str, object],
) -> None:
    selected = offline_preflight["relevance_selected_candidate_ids"]
    first = {
        "tasks": tuple((item.task_id, item.prompt_digest, item.verifier_digest) for item in pilot.TASKS),
        "plan": pilot.make_plan(),
        "selected": selected,
        "seed": pilot.CAMPAIGN_SEED,
    }
    second = {
        "tasks": tuple((item.task_id, item.prompt_digest, item.verifier_digest) for item in pilot.TASKS),
        "plan": pilot.make_plan(),
        "selected": selected,
        "seed": pilot.CAMPAIGN_SEED,
    }
    assert canonical_digest(first) == canonical_digest(second)

