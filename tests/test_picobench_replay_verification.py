from __future__ import annotations

import json
from pathlib import Path

from benchmarks.picobench.packs.replay_verification import (
    BENCHMARK_SCHEMA,
    run_replay_verification_benchmark,
)


def _scenarios(result):
    return {item.scenario_id: item for item in result.scenarios}


def test_frozen_replay_verification_benchmark_meets_acceptance_contract(tmp_path: Path) -> None:
    result = run_replay_verification_benchmark(tmp_path / "benchmark")
    metrics = result.metrics

    assert result.schema == BENCHMARK_SCHEMA
    assert result.passed is True
    assert metrics.valid_scenario_count == 12
    assert metrics.valid_replay_pass_count == 12
    assert metrics.valid_replay_pass_rate == 1.0
    assert metrics.partial_scenario_count == 11
    assert metrics.correct_inconclusive_count == 11
    assert metrics.correct_inconclusive_rate == 1.0
    assert metrics.corruption_scenario_count == 14
    assert metrics.corruption_detected_count == 14
    assert metrics.corruption_detection_rate == 1.0
    assert metrics.false_positive_corruption_count == 0
    assert metrics.false_negative_corruption_count == 0
    assert metrics.side_effect_calls == 0
    assert metrics.privacy_safe is True
    assert result.report_path.is_file()


def test_provider_tool_effect_and_mutation_scenarios_preserve_structural_facts(tmp_path: Path) -> None:
    scenarios = _scenarios(run_replay_verification_benchmark(tmp_path / "benchmark"))

    retry = scenarios["valid-provider-retry-fallback"]
    assert retry.provider_attempt_count == 2
    assert retry.provider_retry_count == 1
    assert retry.provider_fallback_count == 1

    direct = scenarios["valid-direct-read"]
    assert direct.direct_tool_execution_count == 1
    assert direct.meta_routed_tool_execution_count == 0
    assert direct.tool_effects == ("read",)

    meta = scenarios["valid-meta-write"]
    assert meta.meta_routed_tool_execution_count == 1
    assert meta.mutation_observed is True
    assert set(meta.tool_effects) == {"read", "write"}

    matrix = scenarios["valid-effect-matrix"]
    assert set(matrix.tool_effects) == {"execute", "external", "read", "write"}
    assert matrix.validation_failure_count == 1
    assert matrix.execution_failure_count == 1
    assert matrix.timeout_count == 1
    assert matrix.external_state_current is None

    explicit = scenarios["valid-explicit-verification-after-mutation"]
    assert explicit.mutation_observed is True
    assert explicit.verification_after_last_mutation is True

    absent = scenarios["partial-agent-success-claim-unproven"]
    assert absent.mutation_observed is True
    assert absent.verification_after_last_mutation is None
    assert absent.verification_status == "inconclusive"


def test_runtime_delivery_and_agent_text_remain_non_authoritative(tmp_path: Path) -> None:
    result = run_replay_verification_benchmark(tmp_path / "benchmark")
    scenarios = _scenarios(result)

    dropped = scenarios["valid-runtime-success-delivery-dropped"]
    assert dropped.runtime_success is True
    assert dropped.delivery_dropped is True
    assert dropped.delivery_success is False

    notified = scenarios["valid-runtime-failure-delivery-success"]
    assert notified.runtime_failure is True
    assert notified.delivery_success is True
    assert notified.runtime_success is False

    no_delivery = scenarios["valid-success"]
    assert no_delivery.runtime_success is True
    assert no_delivery.delivery_unknown is True
    assert no_delivery.verification_status == "pass"

    success_claim = scenarios["partial-agent-success-claim-unproven"]
    failure_claim = scenarios["valid-agent-failure-claim-ignored"]
    assert success_claim.agent_claim == "success"
    assert success_claim.task_success_proven is False
    assert failure_claim.agent_claim == "failure"
    assert failure_claim.verification_status == "pass"
    assert failure_claim.runtime_success is True
    assert all(item.task_success_proven is False for item in result.scenarios)


def test_session_checkpoint_and_corruption_matrix_are_frozen(tmp_path: Path) -> None:
    scenarios = _scenarios(run_replay_verification_benchmark(tmp_path / "benchmark"))

    assert scenarios["valid-session-checkpoint"].verification_status == "pass"
    assert scenarios["partial-unresolved-reference"].verification_status == "inconclusive"
    assert scenarios["corrupt-session-boundary"].verification_status == "fail"
    assert scenarios["corrupt-checkpoint"].verification_status == "fail"

    corrupt_ids = {
        scenario_id
        for scenario_id, item in scenarios.items()
        if item.scenario_class == "corrupt"
    }
    assert corrupt_ids == {
        "corrupt-altered-replay-digest",
        "corrupt-altered-source-digest",
        "corrupt-checkpoint",
        "corrupt-conflicting-terminal",
        "corrupt-delivery-wrong-turn",
        "corrupt-duplicate-receipt",
        "corrupt-duplicate-sequence",
        "corrupt-impossible-parent",
        "corrupt-provider-ordinal",
        "corrupt-provider-ownership",
        "corrupt-requested-resolved",
        "corrupt-session-boundary",
        "corrupt-unsupported-schema",
        "corrupt-wrong-turn",
    }
    assert all(scenarios[scenario_id].verification_status == "fail" for scenario_id in corrupt_ids)


def test_repeat_runs_have_stable_semantics_and_local_diagnostics(tmp_path: Path) -> None:
    first = run_replay_verification_benchmark(tmp_path / "first")
    second = run_replay_verification_benchmark(tmp_path / "second")

    assert first.semantic_digest == second.semantic_digest
    assert [item.replay_digest for item in first.scenarios] == [
        item.replay_digest for item in second.scenarios
    ]
    assert [item.verification_digest for item in first.scenarios] == [
        item.verification_digest for item in second.scenarios
    ]
    assert first.metrics.total_evidence_events == second.metrics.total_evidence_events
    assert first.metrics.total_persisted_evidence_bytes == second.metrics.total_persisted_evidence_bytes
    assert first.metrics.total_replay_result_bytes == second.metrics.total_replay_result_bytes
    assert first.metrics.total_verification_result_bytes == second.metrics.total_verification_result_bytes
    assert first.metrics.total_replay_latency_ns >= 0
    assert first.metrics.total_verification_latency_ns >= 0


def test_report_contains_no_full_agent_or_tool_content(tmp_path: Path) -> None:
    result = run_replay_verification_benchmark(tmp_path / "benchmark")
    report = result.report_path.read_text(encoding="utf-8")
    parsed = json.loads(report)

    assert parsed["schema"] == BENCHMARK_SCHEMA
    assert "tests passed secret final answer" not in report
    assert "agent claims failure secret final answer" not in report
    assert "raw prompt secret" not in report
    assert "raw tool argument secret" not in report
    assert "raw tool result secret" not in report
    assert '"prompt"' not in report
    assert '"reasoning"' not in report
    assert '"arguments"' not in report
    assert '"result_body"' not in report


def test_benchmark_invokes_no_live_execution_or_recovery_service(tmp_path: Path, monkeypatch) -> None:
    from pico.agent.loop.rewind import SelectiveRewindCoordinator
    from pico.agent.loop.workspace_restore import WorkspaceRestoreService
    from pico.agent.tools.base import Tool
    from pico.agent.tools.registry import ToolRegistry
    from pico.providers.base import LLMProvider
    from pico.session.manager import Session, SessionManager
    from pico.spine.delivery import DeliveryHub

    calls: list[str] = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"offline benchmark invoked {name}")

        return fail

    monkeypatch.setattr(LLMProvider, "chat", forbidden("provider"))
    monkeypatch.setattr(ToolRegistry, "execute", forbidden("tool_registry_execute"))
    monkeypatch.setattr(ToolRegistry, "execute_resolved", forbidden("tool_registry_execute_resolved"))
    monkeypatch.setattr(Tool, "execute", forbidden("concrete_tool"))
    monkeypatch.setattr(DeliveryHub, "dispatch", forbidden("delivery"))
    monkeypatch.setattr(SessionManager, "fork", forbidden("session_fork"))
    monkeypatch.setattr(SessionManager, "get_or_create", forbidden("session_create"))
    monkeypatch.setattr(Session, "clear", forbidden("session_clear"))
    monkeypatch.setattr(WorkspaceRestoreService, "restore", forbidden("workspace_restore"))
    monkeypatch.setattr(SelectiveRewindCoordinator, "rewind", forbidden("selective_rewind"))

    result = run_replay_verification_benchmark(tmp_path / "benchmark")

    assert result.passed is True
    assert result.metrics.side_effect_calls == 0
    assert calls == []
