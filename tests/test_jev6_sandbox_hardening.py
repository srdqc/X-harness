from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_benchmark
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_sandbox import (
    assess_benchmark_sandbox,
    run_benchmark_sandbox_smoke,
)
from benchmarks.picobench.packs.knowledge_evolution_live.run_isolation import AgentRunRoots
from pico.sandbox import ExecResult, SandboxConfig


def test_none_backend_is_not_a_benchmark_sandbox(tmp_path: Path) -> None:
    evidence = assess_benchmark_sandbox(
        SandboxConfig(backend="none"),
        workspace=tmp_path,
        platform_name="linux",
        dependency_available=True,
        kvm_available=True,
    )

    assert evidence["passed"] is False
    assert evidence["infra_invalid_reason"] == "sandbox_backend_none"
    assert evidence["host_execution_allowed"] is True
    assert evidence["provider_calls"] == evidence["agent_turns"] == 0


def test_required_boxlite_contract_is_bounded_and_secret_free(tmp_path: Path) -> None:
    evidence = assess_benchmark_sandbox(
        SandboxConfig(backend="boxlite", allow_net=False),
        workspace=tmp_path,
        platform_name="linux",
        dependency_available=True,
        kvm_available=True,
    )

    assert evidence["passed"] is True
    assert evidence["host_execution_allowed"] is False
    assert evidence["checks"]["network_isolation_configured"] is True
    assert evidence["workspace_root_identity_digest"] == canonical_digest(
        str(tmp_path.resolve())
    )
    assert str(tmp_path.resolve()) not in json.dumps(evidence)


def test_windows_fails_closed_without_weakening_requirement(tmp_path: Path) -> None:
    evidence = assess_benchmark_sandbox(
        SandboxConfig(backend="boxlite", allow_net=False),
        workspace=tmp_path,
        platform_name="win32",
        dependency_available=True,
    )

    assert evidence["passed"] is False
    assert evidence["infra_invalid_reason"] == "unsupported_platform:win32"
    assert evidence["checks"]["platform_supported"] is False


def test_offline_smoke_records_bounded_host_isolation_evidence(tmp_path: Path) -> None:
    config = SandboxConfig(backend="boxlite", allow_net=False)
    capability = assess_benchmark_sandbox(
        config,
        workspace=tmp_path,
        platform_name="linux",
        dependency_available=True,
        kvm_available=True,
    )

    class FakeIsolatedExecutor:
        is_sandboxed = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def exec(self, command: str, **_kwargs) -> ExecResult:
            if command.startswith("printf 'jev6s'"):
                (tmp_path / ".jev6s-sandbox-smoke").write_text(
                    "jev6s", encoding="utf-8"
                )
                return ExecResult("", "", 0)
            if command.startswith("test \"$(cat"):
                return ExecResult("pass", "", 0)
            if command.startswith("awk "):
                return ExecResult("", "", 0)
            return ExecResult("", "not mounted", 1)

    result = asyncio.run(
        run_benchmark_sandbox_smoke(
            config,
            tmp_path,
            executor_factory=lambda _config, _workspace: FakeIsolatedExecutor(),
            capability_evidence=capability,
            require_real_boxlite=False,
        )
    )

    assert result["passed"] is True
    assert result["inside_workspace_write"] is True
    assert result["worktree_read_write"] is True
    assert result["outside_workspace_host_write_rejected"] is True
    assert result["host_user_site_mutation_rejected"] is True
    assert result["network_isolation_supported"] is True
    assert result["host_fingerprint_unchanged"] is True
    assert result["validation_result"] == "PASS_DETERMINISTIC"
    assert result["provider_calls"] == result["agent_turns"] == 0


def test_v2_execute_one_rejects_none_before_agent_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    planned = {
        "run_id": "jev6-test-task_relevance_v1-rep1",
        "task_id": "jev6-test",
        "arm": "task_relevance_v1",
        "repetition": 1,
        "order": 1,
    }
    manifest = {
        "campaign_id": "jev6v2b-test",
        "config_identity_digest": "config",
    }
    bootstrap = {
        "canonical_environment_fingerprint": {
            "schema": "pico.environment-fingerprint.v1",
            "schema_version": 1,
            "python_executable_digest": "a" * 64,
            "python_version": [3, 12, 0],
            "sys_path_digest": "b" * 64,
            "distribution_digest": "c" * 64,
            "distribution_count": 1,
            "user_site_enabled": False,
            "fingerprint_digest": "unused",
        }
    }
    bootstrap["integrity_digest"] = canonical_digest(bootstrap)
    (root / "config-bootstrap.json").write_text(json.dumps(bootstrap), encoding="utf-8")
    fingerprint = {
        "python_executable_digest": "a" * 64,
        "python_version": (3, 12, 0),
        "sys_path_digest": "b" * 64,
        "distribution_digest": "c" * 64,
        "distribution_count": 1,
        "user_site_enabled": False,
    }
    roots = AgentRunRoots.create(root, planned["run_id"], 1)
    roots.prepare_non_worktree_roots()
    monkeypatch.setenv("PICO_HOME", str(roots.state / "pico-home"))
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(
        jev6_benchmark,
        "_config_identity",
        lambda _path: {
            "config_identity_digest": "config",
            "provider_id": "deepseek",
            "model_id": "deepseek/deepseek-v4-flash",
        },
    )
    monkeypatch.setattr(jev6_benchmark, "environment_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(
        jev6_benchmark,
        "verify_trace_canary",
        lambda *_args, **_kwargs: {"passed": True, "event_count": 1},
    )
    monkeypatch.setattr(
        jev6_benchmark,
        "PRE_RUN_SANDBOX_AUDIT",
        lambda **_kwargs: assess_benchmark_sandbox(
            SandboxConfig(backend="none"),
            platform_name="linux",
            dependency_available=True,
            kvm_available=True,
        ),
    )
    agent_started = False

    async def forbidden_agent(*_args, **_kwargs):
        nonlocal agent_started
        agent_started = True
        raise AssertionError("Agent Turn must not start")

    monkeypatch.setattr(jev6_benchmark, "_execute_turn", forbidden_agent)
    record = jev6_benchmark.execute_one(
        tmp_path,
        root,
        manifest,
        planned,
        "human:test",
        config_path=tmp_path / "config.json",
    )

    assert record["run_validity"] == "infra_invalid"
    assert record["infra_invalid_reason"] == "sandbox_backend_none"
    assert record["provider_calls"] == record["agent_turns"] == 0
    assert record["live_agent_exposed"] is False
    assert agent_started is False
