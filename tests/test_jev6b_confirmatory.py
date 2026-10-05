from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_benchmark as benchmark
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    PLANNED_LIVE_RUNS,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    Arm,
)
from benchmarks.picobench.packs.knowledge_evolution_live.run_isolation import (
    canonical_environment_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "benchmarks/picobench/packs/knowledge_evolution_live/jev6_frozen_manifest.json"


def _metric(value: int | float) -> dict[str, object]:
    return {"value": value, "availability": "available"}


def _synthetic_runs(selected: dict[str, tuple[str, ...]]) -> tuple[dict[str, object], ...]:
    rows = []
    for index, planned in enumerate(RUN_ORDER):
        is_c = planned.arm == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value
        candidates = selected[planned.task_id]
        utility_call = int(is_c and bool(candidates))
        utility = {
            "relevance_selected_count": len(candidates),
            "utility_logical_calls": utility_call,
            "utility_provider_attempts": utility_call,
            "raw_choice_counts": {"KEEP": len(candidates) if is_c else 0, "ABSTAIN": 0, "UNCERTAIN": 0},
            "effective_choice_counts": {"KEEP": len(candidates) if is_c else 0, "ABSTAIN": 0},
            "abstained_candidate_ids": (),
            "fallback_count": 0,
            "utility_decision_unavailable": False,
            "all_abstain": False,
            "generation_outcomes": ("stop",) if utility_call else (),
            "payload_outcomes": ("valid_typed_json",) if utility_call else (),
            "reasoning_tokens": (10,) if utility_call else (),
            "visible_output_tokens": (5,) if utility_call else (),
            "output_tokens": 15 if utility_call else 0,
            "provider_latency_ms": 20.0 if utility_call else 0.0,
            "finish_reasons": ("stop",) if utility_call else (),
        }
        main = {
            "provider_logical_calls": _metric(2),
            "provider_attempts": _metric(2),
            "input_tokens": _metric(100),
            "output_tokens": _metric(50),
            "tool_calls_total": _metric(4),
            "repeated_repo_file_reads": _metric(1),
            "turn_latency_ms": _metric(1000),
        }
        rows.append(
            {
                "run_id": planned.run_id,
                "task_id": planned.task_id,
                    "arm": planned.arm,
                "repetition": planned.repetition,
                "order": planned.order,
                "run_validity": "valid",
                "verified_success": True,
                "relevance_selected_candidate_ids": candidates,
                "utility_metrics": utility,
                "main_agent_metrics": main,
                "combined_provider_cost": {
                    "provider_logical_calls": 2,
                    "provider_attempts": 2,
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
                "child_process_id": index + 100,
                "trace_canary": {"passed": True},
                "mandatory_evidence": {"complete": True},
                "environment_drift_detected": False,
                "cross_run_isolation": {"passed": True},
                "config_source_unchanged": True,
                "provider_id": "deepseek",
                "model_id": "deepseek/deepseek-v4-flash",
                "typesafe_enabled": False,
            }
        )
    return tuple(rows)


def test_freeze_recomputes_exactly_and_historical_tasks_are_blind(tmp_path: Path) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    result = benchmark.verify_frozen_suite(ROOT)
    assert result["passed"] is True
    assert result["digests"] == frozen["anti_tuning_digests"]
    (tmp_path / ".p3r").mkdir()
    assert benchmark.verify_historical_task_blindness(tmp_path)["prior_live_agent_exposures"] == ()


def test_plan_is_exactly_frozen_two_arm_three_rep_48_run_order() -> None:
    assert len(RUN_ORDER) == PLANNED_LIVE_RUNS == 48
    assert canonical_digest(RUN_ORDER) == RUN_ORDER_DIGEST
    assert {item.arm for item in RUN_ORDER} == {arm.value for arm in Arm}
    assert all(
        sum(item.task_id == task_id and item.arm == arm.value for item in RUN_ORDER) == 3
        for task_id in {item.task_id for item in RUN_ORDER}
        for arm in Arm
    )


def test_live_gate_and_one_shot_guard(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="--execute-live"):
        benchmark.run_campaign(tmp_path, tmp_path / "missing", "human:test", execute_live=False)


def test_environment_identity_survives_json_round_trip() -> None:
    fingerprint = {
        "python_executable_digest": "a" * 64,
        "python_version": (3, 12, 14),
        "sys_path_digest": "c" * 64,
        "distribution_digest": "b" * 64,
        "distribution_count": 121,
        "user_site_enabled": False,
    }
    identity = canonical_environment_fingerprint(fingerprint)
    assert json.loads(json.dumps(identity)) == identity


def test_invalid_predecessor_pre_turn_record_is_not_heldout_exposure(
    tmp_path: Path,
) -> None:
    predecessor = ROOT / ".p3r" / benchmark.INVALID_PREDECESSOR_ID
    copied = tmp_path / ".p3r" / benchmark.INVALID_PREDECESSOR_ID
    shutil.copytree(predecessor / "runs", copied / "runs")
    audit = benchmark.verify_historical_task_blindness(tmp_path)
    predecessor_records = tuple(
        row
        for row in audit["observed_pre_turn_records"]
        if row["campaign_id"] == benchmark.INVALID_PREDECESSOR_ID
    )
    assert audit["live_task_exposure_count"] == 0
    assert predecessor_records
    assert all(row["agent_turn_started"] is False for row in predecessor_records)
    assert all(row["provider_calls"] == 0 for row in predecessor_records)
    assert all(row["live_task_exposed"] is False for row in predecessor_records)


def test_campaign_generation_freezes_lineage_and_infrastructure_only_delta() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    semantic = benchmark._campaign_semantic(
        {
            "digests": frozen["anti_tuning_digests"],
            "frozen_manifest_file_digest": "d" * 64,
            "selector_audit": {
                "task_selected_candidate_ids": frozen["selector_audit"][
                    "selected_candidate_ids"
                ]
            },
        },
        {
            "config_identity_digest": "a" * 64,
            "config_source_digest": "b" * 64,
            "provider_model_config_digest": "c" * 64,
            "offline_rehearsal_digest": "e" * 64,
        },
    )
    assert benchmark.CAMPAIGN_PREFIX == "jev6b2"
    assert semantic["campaign_generation"] == "JEV.6B2"
    assert semantic["invalid_predecessor_campaign_id"] == benchmark.INVALID_PREDECESSOR_ID
    assert semantic["claim_eligible"] is True
    assert semantic["typesafe_enabled"] is False


def test_utility_decision_semantics_do_not_conflate_fallback_and_abstain() -> None:
    refs = {
        "relevance_selected_candidate_ids": ("a",),
        "utility_invoked_count": 1,
        "utility_provider_calls": 1,
        "utility_provider_attempts": 1,
        "utility_candidate_count": 1,
        "utility_decisions": (),
        "utility_kept_candidate_ids": ("a",),
        "utility_abstained_candidate_ids": (),
        "utility_fallback_count": 1,
        "utility_fallback_reasons": (("malformed_response", 1),),
        "utility_input_tokens": 10,
        "utility_output_tokens": 5,
        "utility_provider_latency_ms": 1.0,
        "utility_latency_ms": 1.0,
    }
    value = benchmark._utility_projection(refs)
    assert value["utility_decision_unavailable"] is True
    assert value["all_abstain"] is False


def test_reducer_uses_per_task_medians_and_frozen_classification(monkeypatch, tmp_path: Path) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    selected = {key: tuple(value) for key, value in frozen["selector_audit"]["selected_candidate_ids"].items()}
    runs = _synthetic_runs(selected)
    manifest = {
        "campaign_id": "jev6b-synthetic",
        "planned_runs": tuple({"run_id": item.run_id} for item in RUN_ORDER),
        "frozen_selected_candidate_ids": selected,
        "historical_artifact_digests_before": {"same": "x"},
        "frozen_source_digests_before": {"same": "y"},
    }
    monkeypatch.setattr(benchmark, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(benchmark, "load_runs", lambda _root: runs)
    monkeypatch.setattr(benchmark, "historical_artifact_digests", lambda _repo: {"same": "x"})
    monkeypatch.setattr(benchmark, "frozen_source_digests", lambda: {"same": "y"})
    summary = benchmark.reduce_campaign(tmp_path, ROOT)
    assert summary["B_success"] == summary["C_success"] == 24
    assert summary["utility_invocation_count"] == 15
    assert summary["classification"] == "NEUTRAL"
    assert summary["rerun_count"] == summary["replacement_count"] == 0
    assert summary["typesafe_enabled"] is False
    assert set(summary["suite_level_median_paired_relative_deltas"]) == set(benchmark.PRIMARY_METRICS)
