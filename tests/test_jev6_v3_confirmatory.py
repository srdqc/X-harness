from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live import jev6_benchmark
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_suite import (
    RUN_ORDER,
)
from benchmarks.picobench.packs.knowledge_evolution_live.run_isolation import (
    AgentRunRoots,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = "benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_benchmark"


def test_v3_runner_profile_isolated_in_child_process() -> None:
    script = """
import json
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_v3_benchmark
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_benchmark as runner
print(json.dumps({
    "module": runner.MODULE_NAME,
    "prefix": runner.CAMPAIGN_PREFIX,
    "suite": runner.SUITE_VERSION,
    "runs": len(runner.RUN_ORDER),
    "sandbox": runner.PRE_RUN_SANDBOX_SMOKE is not None,
    "sandbox_backend": runner.RUNTIME_SANDBOX_CONFIG.backend,
    "sandbox_allow_net": runner.RUNTIME_SANDBOX_CONFIG.allow_net,
    "sandbox_extra_volumes": runner.RUNTIME_SANDBOX_CONFIG.extra_volumes,
    "sandbox_registry": runner.RUNTIME_SANDBOX_CONFIG.image_search_registry,
    "sandbox_runtime_home": runner.RUNTIME_SANDBOX_CONFIG.runtime_home.as_posix(),
    "sandbox_source": runner.SANDBOX_SOURCE_IDENTITY["source"],
    "resume_safe": runner.RESUME_SAFE_CAMPAIGN,
    "exposure": runner.PERSIST_LIVE_EXPOSURE,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "module": MODULE,
        "prefix": "jev6v3",
        "suite": "jev6-held-out-v3",
        "runs": 48,
        "sandbox": True,
        "sandbox_backend": "boxlite",
        "sandbox_allow_net": False,
        "sandbox_extra_volumes": [],
        "sandbox_registry": "docker.m.daocloud.io",
        "sandbox_runtime_home": "/tmp/pico-jev6v3-boxlite",
        "sandbox_source": "frozen_jev6_v3_infrastructure",
        "resume_safe": True,
        "exposure": True,
    }


def test_v3_runtime_overlays_only_typed_frozen_sandbox_in_child_process() -> None:
    script = """
import json
from pico.config.loader import get_config_path, load_config
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_v3_benchmark
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_benchmark as runner
source = load_config(get_config_path())
runtime = runner._with_runtime_sandbox(source)
print(json.dumps({
    "same_object": source is runtime,
    "source_backend": source.tools.sandbox.backend,
    "runtime_backend": runtime.tools.sandbox.backend,
    "runtime_allow_net": runtime.tools.sandbox.allow_net,
    "runtime_extra_volumes": runtime.tools.sandbox.extra_volumes,
    "runtime_registry": runtime.tools.sandbox.image_search_registry,
    "runtime_home": runtime.tools.sandbox.runtime_home.as_posix(),
    "provider_blocks_unchanged": source.providers == runtime.providers,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    assert value == {
        "same_object": False,
        "source_backend": "none",
        "runtime_backend": "boxlite",
        "runtime_allow_net": False,
        "runtime_extra_volumes": [],
        "runtime_registry": "docker.m.daocloud.io",
        "runtime_home": "/tmp/pico-jev6v3-boxlite",
        "provider_blocks_unchanged": True,
    }


def test_v3_profile_recomputes_every_frozen_digest() -> None:
    script = """
import json
from pathlib import Path
from benchmarks.picobench.packs.knowledge_evolution_live import jev6_v3_benchmark as profile
result = profile.verify_frozen_suite(Path.cwd())
print(json.dumps({
    "passed": result["passed"],
    "checks": all(result["checks"].values()),
    "digest_count": len(result["digests"]),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "passed": True,
        "checks": True,
        "digest_count": 18,
    }


def test_v3_cli_refuses_live_path_without_explicit_flag() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "run",
            "--repository",
            str(ROOT),
            "--campaign-root",
            str(ROOT / ".p3r/missing-jev6v3"),
            "--reviewer-id",
            "human:test",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--execute-live" in completed.stderr


def _manifest() -> dict[str, object]:
    return {
        "campaign_id": "jev6v3-test",
        "suite_version": "jev6-held-out-v3",
        "planned_runs": [asdict(item) for item in RUN_ORDER],
    }


def test_exposure_is_atomic_and_resume_accepts_only_completed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jev6_benchmark, "PERSIST_LIVE_EXPOSURE", True)
    manifest = _manifest()
    first = manifest["planned_runs"][0]
    jev6_benchmark._mark_live_exposure(tmp_path, manifest, first)
    record = {
        "schema": "test",
        "campaign_id": manifest["campaign_id"],
        **first,
        "run_validity": "valid",
    }
    record["integrity_digest"] = jev6_benchmark.canonical_digest(record)
    jev6_benchmark._write_run(tmp_path, record)

    assert jev6_benchmark._resume_completed_prefix(tmp_path, manifest) == (
        first["run_id"],
    )
    exposure = jev6_benchmark._load_live_exposure(tmp_path)
    assert exposure is not None
    assert exposure["tasks"][first["task_id"]]["live_agent_exposed"] is True


def test_resume_fails_closed_after_exposure_without_completed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jev6_benchmark, "PERSIST_LIVE_EXPOSURE", True)
    manifest = _manifest()
    jev6_benchmark._mark_live_exposure(
        tmp_path, manifest, manifest["planned_runs"][0]
    )
    with pytest.raises(RuntimeError, match="rerun is forbidden"):
        jev6_benchmark._resume_completed_prefix(tmp_path, manifest)


def test_child_failure_forensics_distinguish_capability_stage_and_stay_zero_live() -> None:
    completed = SimpleNamespace(
        returncode=2,
        stderr="Authorization: Bearer forbidden-secret\ncapability failed",
        stdout="api_key=forbidden-secret\n",
    )
    record = {
        "infra_invalid_reason": "unsupported_platform:win32",
        "sandbox_capability": {
            "configured_backend": "boxlite",
            "platform_family": "win32",
        },
        "child_environment_identity": {
            "sandbox_backend": "boxlite",
            "allow_net": False,
            "extra_volumes": [],
        },
        "provider_calls": 0,
        "agent_turns": 0,
        "live_agent_exposed": False,
    }
    evidence = jev6_benchmark._rehearsal_failure_evidence(
        campaign_id="jev6v3-test",
        completed=completed,
        record=record,
        python_executable=sys.executable,
        cwd=ROOT,
    )

    assert evidence["last_completed_startup_stage"] == "sandbox_config_resolution"
    assert evidence["failing_startup_stage"] == "sandbox_capability_validation"
    assert evidence["boxlite_runtime_construction_started"] is False
    assert evidence["microvm_startup_started"] is False
    assert evidence["workspace_mount_started"] is False
    assert evidence["provider_calls"] == evidence["agent_turns"] == 0
    assert evidence["live_agent_exposed"] is False
    assert "forbidden-secret" not in evidence["stderr_tail"]
    assert "forbidden-secret" not in evidence["stdout_tail"]


def test_child_environment_identity_is_allowlisted_and_propagates_sandbox_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pico.config.loader import get_config_path, load_config

    roots = AgentRunRoots.create(tmp_path, "forensic", 1)
    roots.prepare_non_worktree_roots()
    config_path = get_config_path().resolve()
    expected = load_config(config_path).tools.sandbox
    monkeypatch.setattr(
        jev6_benchmark,
        "_config_source_digest",
        lambda path: "d" * 64,
    )
    identity = jev6_benchmark._bounded_child_environment_identity(
        config_path, roots
    )

    assert identity["sandbox_backend"] == expected.backend
    assert identity["allow_net"] == expected.allow_net
    assert identity["extra_volumes"] == expected.extra_volumes
    assert identity["provider_source_identity"] == {
        "config_path": str(config_path),
        "config_source_digest": "d" * 64,
    }
    assert "environment" not in identity
    assert "api_key" not in identity


def test_forensic_failure_artifact_is_atomic_and_integrity_checked(tmp_path: Path) -> None:
    evidence = {
        "schema": "pico.jev6-production-child-forensic.v1",
        "provider_calls": 0,
        "agent_turns": 0,
        "live_agent_exposed": False,
    }
    evidence["integrity_digest"] = jev6_benchmark.canonical_digest(evidence)
    path = tmp_path / ".p3r" / "failure.json"
    path.parent.mkdir()
    jev6_benchmark._write_forensic_artifact(path, evidence)

    stored = json.loads(path.read_text(encoding="utf-8"))
    digest = stored.pop("integrity_digest")
    assert digest == jev6_benchmark.canonical_digest(stored)
