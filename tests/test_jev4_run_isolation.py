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


def test_explicit_config_path_survives_run_local_pico_home(tmp_path: Path) -> None:
    from pico.config.schema import Config

    fake_secret = "jev4r2-fake-secret-never-persist"
    config = Config()
    config.agents.defaults.model = "deepseek/frozen-test-model"
    config.providers.deepseek.api_key = fake_secret
    config_path = tmp_path / "source-config.json"
    config_path.write_text(config.model_dump_json(by_alias=True), encoding="utf-8")
    roots = AgentRunRoots.create(tmp_path / "campaign", "config", 1)
    roots.prepare_non_worktree_roots()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from pico.config.loader import load_config, set_config_path; "
                "set_config_path(Path(sys.argv[1])); "
                "print(load_config().agents.defaults.model)"
            ),
            str(config_path),
        ],
        cwd=Path.cwd(), env=roots.child_environment(), check=False,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "deepseek/frozen-test-model"

    source = jev4_pilot.load_manifest(Path(".p3r/jev4-9168e909e0c47dfb"))
    budget = jev4_pilot.RuntimeBudget(**source["budget"])
    identity = jev4_pilot._sanitized_config_identity(config_path, budget)
    assert identity["provider_id"] == "deepseek"
    assert identity["model_id"] == "deepseek/frozen-test-model"
    assert identity["credential_available"] is True
    assert fake_secret not in json.dumps(identity, sort_keys=True)
    assert jev4_pilot._config_source_digest(config_path) == identity["config_source_digest"]


def test_missing_explicit_config_path_fails_identity_guard_with_isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pico.config.schema import Config

    source = jev4_pilot.load_manifest(Path(".p3r/jev4-9168e909e0c47dfb"))
    budget = jev4_pilot.RuntimeBudget(**source["budget"])
    isolated_home = tmp_path / "isolated-pico-home"
    monkeypatch.setenv("PICO_HOME", str(isolated_home))
    isolated_home.mkdir()
    default_config = Config()
    default_config.providers.anthropic.api_key = "fake-anthropic-secret"
    default_path = isolated_home / "config.json"
    default_path.write_text(default_config.model_dump_json(by_alias=True), encoding="utf-8")
    with pytest.raises(ValueError, match="DeepSeek"):
        jev4_pilot._sanitized_config_identity(default_path, budget)


def test_verified_bootstrap_artifact_is_revalidated_without_second_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "jev4r2-test"
    root.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    source_digest = jev4_pilot._config_source_digest(config_path)
    manifest = {
        "actual_model_id": "deepseek/deepseek-v4-flash",
        "config_identity_digest": "identity",
        "config_source_digest": source_digest,
    }
    record = {
        "passed": True,
        "lifecycle_state": "VERIFIED",
        "secret_leak_free": True,
        "config_source_unchanged": True,
        "config_identity_digest": "identity",
    }
    record["integrity_digest"] = jev4_pilot.canonical_digest(record)
    jev4_pilot._store(root).write_summary(root / "config-bootstrap.json", record)
    monkeypatch.setattr(jev4_pilot, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(
        jev4_pilot,
        "_validate_child_config",
        lambda _manifest, _path: {"passed": True, "config_identity_digest": "identity"},
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("verified bootstrap must not spawn a second child"),
    )
    result = jev4_pilot.run_config_bootstrap(root, config_path)
    assert result["passed"] is True
    assert result["revalidated_without_child_rerun"] is True


def test_duplicate_bootstrap_lifecycle_is_rejected_before_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "config-bootstrap.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="cannot return to RUNNING"):
        jev4_pilot._execute_config_bootstrap(
            root, {"campaign_id": "campaign"}, tmp_path / "unused-config.json"
        )
    assert not (root / "s" / "r00" / "logs" / "audit-events.log").exists()


def test_full_pre_live_rehearsal_spawns_one_bootstrap_then_stops(tmp_path: Path) -> None:
    from pico.config.schema import Config

    config = Config()
    config.agents.defaults.model = "deepseek/deepseek-v4-flash"
    config.providers.deepseek.api_key = "jev4r2a-fake-secret"
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(by_alias=True), encoding="utf-8")
    result = jev4_pilot.rehearse_pre_live(
        Path.cwd(),
        Path(".p3r/jev4-9168e909e0c47dfb").resolve(),
        Path(".p3r/jev4r-3f7c246afe462f5e").resolve(),
        Path(".p3r/jev4r2-868dd3997095b407").resolve(),
        "human:test",
        config_path=config_path,
    )
    assert result["bootstrap_child_count"] == 1
    assert len(result["bootstrap_child_process_ids"]) == 1
    assert result["bootstrap_lifecycle_transitions"] == [
        "NOT_STARTED", "RUNNING", "VERIFIED",
    ]
    assert result["bootstrap_artifact_digest_before"] == result["bootstrap_artifact_digest_after"]
    assert result["parent_revalidation_count"] == 3
    assert result["additional_trace_event_count"] == 0
    assert result["live_ready"] is True
    assert result["next_action"]["run_id"] == jev4_pilot.make_plan()[0]["run_id"]
    assert result["next_action"]["arm"] == jev4_pilot.Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value
    assert result["main_provider_calls"] == 0
    assert result["utility_provider_calls"] == 0
    assert result["network_calls"] == 0
    assert result["agent_turns"] == 0


def test_all_historical_jev4_campaigns_keep_frozen_classifications() -> None:
    expected = {
        "jev4-9168e909e0c47dfb": "HOLD_AND_FORENSIC",
        "jev4r-3f7c246afe462f5e": "HOLD_INFRA",
        "jev4r2-868dd3997095b407": "HOLD_INFRA",
    }
    for campaign_id, classification in expected.items():
        root = Path(".p3r") / campaign_id
        jev4_pilot.load_manifest(root)
        assert json.loads((root / "reduced.json").read_text(encoding="utf-8"))[
            "classification"
        ] == classification


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
    source = jev4_pilot.load_manifest(Path(".p3r/jev4-9168e909e0c47dfb"))
    candidate = {
        **source,
        "schema": jev4_pilot.SCHEMA,
        "schema_version": jev4_pilot.SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "source_campaign_id": source["campaign_id"],
        "source_campaign_semantic_digest": source["campaign_semantic_digest"],
        "utility_max_tokens": 1024,
        "behavioral_input_digest": jev4_pilot.canonical_digest(
            jev4_pilot._behavioral_input_payload(source)
        ),
    }
    jev4_pilot.assert_infrastructure_only_rerun(source, candidate)
    candidate["base_commit_sha"] = "0" * 40
    with pytest.raises(ValueError, match="behavioral inputs changed"):
        jev4_pilot.assert_infrastructure_only_rerun(source, candidate)


def test_zero_selection_contract_fails_closed_for_b_and_c() -> None:
    common = {"task_id": jev4_pilot.TASKS[2].task_id}
    assert jev4_pilot._zero_selection_contract_failure({
        **common,
        "arm": jev4_pilot.Arm.TASK_RELEVANCE_V1.value,
        "relevance_selected_candidate_ids": ("unexpected",),
    }) == "zero_selection_contract_failure"
    assert jev4_pilot._zero_selection_contract_failure({
        **common,
        "arm": jev4_pilot.Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value,
        "relevance_selected_candidate_ids": (),
        "utility_metrics": {"utility_logical_calls": 1},
    }) == "zero_selection_utility_invoked"
    assert jev4_pilot._zero_selection_contract_failure({
        **common,
        "arm": jev4_pilot.Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value,
        "relevance_selected_candidate_ids": (),
        "utility_metrics": {"utility_logical_calls": 0},
    }) is None


def test_original_campaign_remains_immutable_and_hold() -> None:
    root = Path(".p3r/jev4-9168e909e0c47dfb")
    manifest_before = (root / "manifest.json").read_bytes()
    reduced_before = (root / "reduced.json").read_bytes()
    assert json.loads(reduced_before)["classification"] == "HOLD_AND_FORENSIC"
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert (root / "reduced.json").read_bytes() == reduced_before
