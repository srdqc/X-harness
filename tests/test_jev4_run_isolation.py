from __future__ import annotations

import json
import shutil
import site
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live import jev4_pilot
from benchmarks.picobench.packs.knowledge_evolution_live.run_isolation import (
    RUNNER_VERSION,
    AgentRunRoots,
    environment_fingerprint,
)
from benchmarks.picobench.packs.knowledge_evolution_live.validity import classify_run_validity
from pico.sandbox.direct_executor import _baseline_env
from pico.tracing import evidence

_CHILD = r"""
import json, os
from pathlib import Path
from benchmarks.picobench.packs.knowledge_evolution_live.run_isolation import verify_trace_canary
result = verify_trace_canary(Path(os.environ['PICO_TRACING_DIR']), canary_id=os.environ['CANARY_ID'], emit_terminal=os.environ.get('SUPPRESS_TERMINAL') != '1')
print(json.dumps({'pid': os.getpid(), 'result': result}, sort_keys=True))
"""


def _spawn_canary(roots: AgentRunRoots, canary_id: str, *, suppress_terminal: bool = False):
    roots.prepare_non_worktree_roots()
    env = roots.child_environment()
    # The CI test interpreter in this repository may obtain test-only
    # dependencies from user-site. Production JEV.4R uses the repository
    # environment unchanged and keeps PYTHONNOUSERSITE=1.
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("PYTHONUSERBASE", None)
    env["CANARY_ID"] = canary_id
    if suppress_terminal:
        env["SUPPRESS_TERMINAL"] = "1"
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD], cwd=Path.cwd(), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_two_children_keep_trace_roots_turns_and_processes_isolated(tmp_path: Path) -> None:
    first = AgentRunRoots.create(tmp_path, "first", 1)
    second = AgentRunRoots.create(tmp_path, "second", 2)
    processes = (_spawn_canary(first, "turn-a"), _spawn_canary(second, "turn-b"))
    values = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        values.append(json.loads(stdout))
    assert values[0]["pid"] != values[1]["pid"]
    assert values[0]["result"]["passed"] is True
    assert values[1]["result"]["passed"] is True
    assert evidence.read_turn_evidence(first.trace, "turn-a").completeness is evidence.EvidenceCompleteness.COMPLETE
    assert not evidence.read_turn_evidence(first.trace, "turn-b").events
    assert evidence.read_turn_evidence(second.trace, "turn-b").completeness is evidence.EvidenceCompleteness.COMPLETE
    assert not evidence.read_turn_evidence(second.trace, "turn-a").events


def test_canary_and_mandatory_evidence_fail_closed_when_terminal_is_missing(tmp_path: Path) -> None:
    good = AgentRunRoots.create(tmp_path, "good", 1)
    bad = AgentRunRoots.create(tmp_path, "bad", 2)
    good_process = _spawn_canary(good, "complete-turn")
    bad_process = _spawn_canary(bad, "partial-turn", suppress_terminal=True)
    good_value = json.loads(good_process.communicate(timeout=30)[0])
    bad_value = json.loads(bad_process.communicate(timeout=30)[0])
    assert good_value["result"]["passed"] is True
    assert bad_value["result"]["passed"] is False
    assert bad_value["result"]["evidence_completeness"] == "partial"
    assert "missing_terminal" in bad_value["result"]["findings"]
    validity = classify_run_validity(
        runtime_outcome="completed",
        normalized_provider_failures=(),
        infrastructure_reason=None,
        mandatory_evidence_complete=False,
    )
    assert validity.validity.value == "infra_invalid"
    assert validity.reason.value == "mandatory_evidence_failure"


def test_run_local_python_and_pip_environment_does_not_target_shared_sites(tmp_path: Path) -> None:
    roots = AgentRunRoots.create(tmp_path, "pip", 1)
    roots.prepare_non_worktree_roots()
    env = roots.child_environment()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert Path(env["PYTHONUSERBASE"]) == roots.python_user_base
    assert Path(env["PIP_TARGET"]) == roots.pip_target
    assert Path(env["PIP_CACHE_DIR"]) == roots.pip_cache
    assert str(roots.pip_target) in env["PYTHONPATH"]
    assert Path(site.getusersitepackages()).resolve() not in {
        roots.python_user_base.resolve(), roots.pip_target.resolve()
    }


def test_direct_executor_propagates_run_local_python_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONUSERBASE": "run-userbase",
        "PIP_TARGET": "run-target",
        "PIP_CACHE_DIR": "run-cache",
        "PYTHONPYCACHEPREFIX": "run-pycache",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    baseline = _baseline_env()
    assert {key: baseline[key] for key in expected} == expected


def test_offline_local_pip_install_is_confined_to_run_target(tmp_path: Path) -> None:
    roots = AgentRunRoots.create(tmp_path, "pip-install", 1)
    roots.prepare_non_worktree_roots()
    wheel = tmp_path / "jev4_local_fixture-0.0.1-py3-none-any.whl"
    dist_info = "jev4_local_fixture-0.0.1.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("jev4_local_fixture.py", "VALUE = 1\n")
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: jev4-local-fixture\nVersion: 0.0.1\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: jev4-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    user_site = Path(site.getusersitepackages())
    shared_marker = user_site / "jev4_local_fixture.py"
    assert not shared_marker.exists()
    completed = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--disable-pip-version-check", "--no-deps", str(wheel),
        ],
        cwd=tmp_path, env=roots.child_environment(), check=False, capture_output=True, text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert (roots.pip_target / "jev4_local_fixture.py").exists()
    assert not shared_marker.exists()


def test_environment_fingerprint_detects_bounded_sys_path_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    before = environment_fingerprint()
    monkeypatch.setattr(sys, "path", [*sys.path, "synthetic-drift-entry"])
    after = environment_fingerprint()
    assert before["fingerprint_digest"] != after["fingerprint_digest"]
    assert before["distribution_digest"] == after["distribution_digest"]


def test_historical_campaign_loads_and_reduces_from_copy(tmp_path: Path) -> None:
    source = Path(".p3r/jev4-9168e909e0c47dfb")
    copied = tmp_path / source.name
    copied.mkdir()
    shutil.copy2(source / "manifest.json", copied / "manifest.json")
    shutil.copytree(source / "runs", copied / "runs")
    manifest = jev4_pilot.load_manifest(copied)
    assert manifest["schema_version"] == 1
    summary = jev4_pilot.reduce_campaign(copied)
    assert summary["classification"] == "HOLD_AND_FORENSIC"
    assert summary["actual_agent_runs"] == 2


def test_jev4r_delta_rejects_behavioral_changes_and_requires_new_runner() -> None:
    source = {key: f"frozen-{key}" for key in jev4_pilot._BEHAVIORAL_MANIFEST_KEYS}
    candidate = {**source, "runner_version": RUNNER_VERSION}
    jev4_pilot.assert_infrastructure_only_rerun(source, candidate)
    candidate["utility_prompt_digest"] = "changed"
    with pytest.raises(ValueError, match="behavioral inputs changed"):
        jev4_pilot.assert_infrastructure_only_rerun(source, candidate)


def test_original_campaign_remains_immutable_and_hold() -> None:
    root = Path(".p3r/jev4-9168e909e0c47dfb")
    manifest_before = (root / "manifest.json").read_bytes()
    reduced_before = (root / "reduced.json").read_bytes()
    assert json.loads(reduced_before)["classification"] == "HOLD_AND_FORENSIC"
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert (root / "reduced.json").read_bytes() == reduced_before
