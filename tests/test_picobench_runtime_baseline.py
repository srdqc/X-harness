from __future__ import annotations

import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.picobench.packs.runtime.baseline import (
    RUNTIME_BASELINE_SCHEMA,
    RUNTIME_BASELINE_TASK_IDS,
    RuntimeBaselineVerifier,
    run_runtime_baseline,
)
from benchmarks.picobench.records import VerificationState


def _records(root: Path) -> dict[str, dict]:
    records = {}
    for path in root.glob("trials/**/trial-record.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["key"]["task_id"]] = record
    return records


async def test_frozen_runtime_baseline_runs_real_path_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def reject_network(*_args, **_kwargs):
        raise AssertionError("runtime baseline must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    ref = await run_runtime_baseline(tmp_path / "baseline")
    records = _records(ref.root)

    assert set(records) == set(RUNTIME_BASELINE_TASK_IDS)

    no_tool = records["success-no-tool"]
    assert no_tool["status"] == "passed"
    assert no_tool["runtime_state"] == "completed"
    assert no_tool["delivery_state"] == "delivered"
    assert no_tool["verification"]["state"] == "passed"
    assert no_tool["metrics"]["provider_attempts"] == 1
    assert no_tool["metrics"]["tool_executions"] == 0

    tool = records["success-tool"]
    assert tool["status"] == "passed"
    assert tool["verification"]["state"] == "passed"
    assert tool["metrics"]["provider_attempts"] == 2
    assert tool["metrics"]["tool_executions"] == 1
    assert tool["metrics"]["tool_execution_evidence"][0]["result_preview"] == (
        "runtime-baseline-tool-result:frozen"
    )

    failure = records["provider-failure"]
    assert failure["status"] == "provider_failure"
    assert failure["runtime_state"] == "provider_failed"
    assert failure["delivery_state"] is None
    assert failure["verification"]["state"] == "failed"
    assert failure["verification"]["metrics"]["evidence_contract_valid"] is True
    assert failure["metrics"]["provider_attempts"] == 1
    assert failure["metrics"]["delivery_results"] == []

    dropped = records["success-delivery-dropped"]
    assert dropped["status"] == "passed"
    assert dropped["runtime_state"] == "completed"
    assert dropped["delivery_state"] == "dropped"
    assert dropped["verification"]["state"] == "passed"
    assert dropped["metrics"]["delivery_results"][0]["outcome"] == "dropped"

    for task_id, record in records.items():
        metrics = record["metrics"]
        turn_id = f"runtime-baseline-turn-{task_id}"
        assert metrics["schema"] == RUNTIME_BASELINE_SCHEMA
        assert metrics["turn_id"] == metrics["terminal_turn_id"] == metrics["trace_turn_id"] == turn_id
        assert metrics["trace_id"]
        assert metrics["latency_ms"] >= 0
        assert record["artifact_refs"]
        if metrics["delivery_results"]:
            assert metrics["delivery_results"][0]["turn_id"] == turn_id
            assert metrics["delivery_trace_ids"] == [metrics["trace_id"]]


async def test_repeated_runtime_baselines_have_identical_semantic_outcomes(tmp_path: Path):
    first_ref = await run_runtime_baseline(tmp_path / "first")
    second_ref = await run_runtime_baseline(tmp_path / "second")
    first = _semantic_projection(_records(first_ref.root))
    second = _semantic_projection(_records(second_ref.root))

    assert first_ref.experiment_id == second_ref.experiment_id
    assert first == second


@pytest.mark.parametrize(
    ("task_id", "corrupt"),
    [
        ("success-no-tool", lambda evidence: evidence.pop("trace_id")),
        (
            "success-no-tool",
            lambda evidence: evidence.__setitem__("terminal_turn_id", "wrong-turn"),
        ),
        (
            "success-no-tool",
            lambda evidence: evidence.__setitem__("trace_outcome", "corrupted"),
        ),
        (
            "success-tool",
            lambda evidence: evidence["tool_execution_evidence"][0].__setitem__(
                "turn_id",
                "wrong-turn",
            ),
        ),
    ],
    ids=("missing", "wrong-turn", "corrupted", "wrong-tool-turn"),
)
async def test_runtime_baseline_verifier_rejects_missing_mismatched_or_corrupted_evidence(
    tmp_path: Path,
    task_id: str,
    corrupt,
):
    ref = await run_runtime_baseline(tmp_path / "baseline")
    evidence = deepcopy(_records(ref.root)[task_id]["metrics"])
    verifier = RuntimeBaselineVerifier()

    assert verifier.verify(task_id, evidence).state is VerificationState.PASSED
    corrupt(evidence)
    result = verifier.verify(task_id, evidence)

    assert result.state is VerificationState.FAILED
    assert result.metrics["evidence_contract_valid"] is False


def _semantic_projection(records: dict[str, dict]) -> dict[str, dict]:
    return {
        task_id: {
            "status": record["status"],
            "runtime_state": record["runtime_state"],
            "delivery_state": record["delivery_state"],
            "verification_state": record["verification"]["state"],
            "verification_valid": record["verification"]["metrics"]["evidence_contract_valid"],
            "conversation_id": record["metrics"]["conversation_id"],
            "turn_id": record["metrics"]["turn_id"],
            "terminal_event": record["metrics"]["terminal_event"],
            "terminal_outcome": record["metrics"]["terminal_outcome"],
            "provider_attempts": record["metrics"]["provider_attempts"],
            "tool_executions": record["metrics"]["tool_executions"],
            "delivery_outcome": record["metrics"]["delivery_outcome"],
            "trace_correlated": (
                record["metrics"]["turn_id"]
                == record["metrics"]["terminal_turn_id"]
                == record["metrics"]["trace_turn_id"]
            ),
        }
        for task_id, record in records.items()
    }
