from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_tests as test_cli
from scripts.testing import runner


def _matrix() -> dict:
    return runner.load_json(runner.MATRIX_PATH)


def _issues() -> dict:
    return runner.load_json(runner.ISSUES_PATH)


def test_matrix_and_issue_registry_load_and_validate() -> None:
    matrix = _matrix()
    issues = _issues()
    runner.validate_matrix(matrix)
    runner.validate_issues(issues)
    assert matrix["schema"] == "x-harness.test-matrix.v1"
    assert issues["schema"] == "x-harness.known-environment-issues.v1"


def test_duplicate_targets_are_removed_without_reordering() -> None:
    unique, removed = runner.deduplicate(["a", "b", "a", "c", "b"])
    assert unique == ["a", "b", "c"]
    assert removed == 2


def test_phase_resolution_is_recursive_and_deduplicated() -> None:
    targets, requested, removed, suites = runner.resolve_targets(_matrix(), tier="phase", phase="p1b")
    assert suites == ["phase_p1b"]
    assert "tests/test_agent_loop_tool_search.py" in targets
    assert "tests/test_spine_scheduler.py" in targets
    assert requested > len(targets)
    assert removed == requested - len(targets)


def test_global_resolution_uses_single_tree_target() -> None:
    targets, requested, removed, suites = runner.resolve_targets(_matrix(), tier="global")
    assert targets == ["tests"]
    assert (requested, removed, suites) == (1, 0, ["global"])


@pytest.mark.parametrize(
    ("module", "issue_id"),
    [
        ("websockets", "ENV-CHANNEL-WEBSOCKETS-MISSING"),
        ("lark_oapi", "ENV-CHANNEL-LARK-MISSING"),
        ("botpy", "ENV-CHANNEL-BOTPY-MISSING"),
        ("wecom_aibot_sdk", "ENV-CHANNEL-WECOM-MISSING"),
    ],
)
def test_windows_optional_dependency_blocker_detection(
    monkeypatch: pytest.MonkeyPatch, module: str, issue_id: str
) -> None:
    original = runner.importlib.util.find_spec
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: None if name == module else original(name),
    )
    issue = next(item for item in _issues()["issues"] if item["id"] == issue_id)
    assert runner.issue_is_active(issue, system="Windows") is True


def test_windows_preflight_selects_writable_short_temp(tmp_path: Path) -> None:
    items, temp_root = runner.run_preflight(_issues(), root=tmp_path, system="Windows")
    assert temp_root == tmp_path / ".tmp" / "pytest"
    assert temp_root.is_dir()
    path_item = next(item for item in items if item.issue_id == "ENV-WINDOWS-PYTEST-PATH-LIMIT")
    assert path_item.state == "AVAILABLE"


def test_project_python_prefers_repository_environment(tmp_path: Path) -> None:
    executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert runner.project_python(tmp_path) == executable


def test_module_availability_override_uses_test_environment() -> None:
    issue = next(
        item for item in _issues()["issues"] if item["id"] == "ENV-CHANNEL-WEBSOCKETS-MISSING"
    )
    assert runner.issue_is_active(issue, module_availability={"websockets": True}) is False
    assert runner.issue_is_active(issue, module_availability={"websockets": False}) is True


def test_blocked_and_fail_are_distinct_with_fail_precedence() -> None:
    assert runner.classify_outcome(command_failed=False, blocked_issue_ids=["ENV-X"]) == "BLOCKED"
    assert runner.classify_outcome(command_failed=True, blocked_issue_ids=[]) == "FAIL"
    assert runner.classify_outcome(command_failed=True, blocked_issue_ids=["ENV-X"]) == "FAIL"


def test_retry_policy_only_allows_registered_transient_external() -> None:
    correctness = {"classification": "KNOWN_TEST_DEBT", "retry_allowed": True}
    transient = {"classification": "TRANSIENT_EXTERNAL", "retry_allowed": True, "max_retries": 9}
    assert runner.retry_limit_for(None) == 0
    assert runner.retry_limit_for(correctness) == 0
    assert runner.retry_limit_for(transient) == 2


def test_unknown_failure_is_not_auto_classified() -> None:
    assert runner.classify_outcome(command_failed=True, blocked_issue_ids=[]) == "FAIL"


def test_stale_test_debt_is_not_an_environment_blocker() -> None:
    debt = next(
        issue for issue in _issues()["issues"] if issue["id"] == "DEBT-P0-TURN-ID-FIELD-ASSERTIONS"
    )
    assert debt["classification"] == "KNOWN_TEST_DEBT"
    assert debt["blocks_tiers"] == []
    assert debt["classification"] not in runner.ENVIRONMENT_CLASSIFICATIONS
    assert runner.pytest_exclusions([debt]) == [
        "--deselect=tests/test_spine_events.py::test_turn_failed_has_no_usage",
        "--deselect=tests/test_spine_events.py::test_turn_ended_carries_usage_latency_and_explicit_reply",
    ]


def test_registry_json_is_stably_serializable() -> None:
    assert json.loads(json.dumps(_issues(), sort_keys=True)) == _issues()


def test_interrupted_runner_reports_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(test_cli, "execute", lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))
    args = SimpleNamespace(
        tier="fast",
        suite=["test_infrastructure"],
        phase=None,
        target=[],
        dry_run=False,
        preflight_only=False,
        skip_ruff=False,
        skip_diff=False,
    )
    monkeypatch.setattr(test_cli.argparse.ArgumentParser, "parse_args", lambda _self: args)
    assert test_cli.main() == 1
    output = capsys.readouterr().out
    assert '"final":"FAIL"' in output
    assert '"unknown_failures":["interrupted"]' in output
