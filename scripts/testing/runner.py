"""Small, dependency-free test-tier resolver and executor."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = Path(__file__).with_name("test_matrix.json")
ISSUES_PATH = Path(__file__).with_name("known_environment_issues.json")
ENVIRONMENT_CLASSIFICATIONS = {
    "ENV_MISSING_OPTIONAL_DEPENDENCY",
    "PLATFORM_INCOMPATIBLE",
    "TEST_INFRA_PATH_LIMIT",
    "TRANSIENT_EXTERNAL",
}


class ConfigurationError(ValueError):
    """Raised for an invalid matrix or issue registry."""


@dataclass(frozen=True)
class PreflightItem:
    issue_id: str
    state: str
    classification: str
    detail: str
    active: bool


@dataclass(frozen=True)
class CommandResult:
    name: str
    returncode: int
    elapsed_seconds: float
    output: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return value


def validate_matrix(matrix: dict[str, Any], *, root: Path = REPO_ROOT) -> None:
    if matrix.get("schema") != "x-harness.test-matrix.v1":
        raise ConfigurationError("unsupported test matrix schema")
    suites = matrix.get("suites")
    tiers = matrix.get("tiers")
    if not isinstance(suites, dict) or not isinstance(tiers, dict):
        raise ConfigurationError("matrix requires suites and tiers objects")
    for name, suite in suites.items():
        if not isinstance(suite, dict):
            raise ConfigurationError(f"suite {name!r} must be an object")
        for reference in suite.get("suites", []):
            if reference not in suites:
                raise ConfigurationError(f"suite {name!r} references unknown suite {reference!r}")
        for target in suite.get("targets", []):
            path_text = str(target).split("::", 1)[0]
            if not (root / path_text).exists():
                raise ConfigurationError(f"suite {name!r} references missing target {target!r}")
    for phase, names in tiers.get("phase", {}).get("phases", {}).items():
        for name in names:
            if name not in suites:
                raise ConfigurationError(f"phase {phase!r} references unknown suite {name!r}")


def validate_issues(registry: dict[str, Any], *, root: Path = REPO_ROOT) -> None:
    if registry.get("schema") != "x-harness.known-environment-issues.v1":
        raise ConfigurationError("unsupported environment issue schema")
    issues = registry.get("issues")
    if not isinstance(issues, list):
        raise ConfigurationError("issue registry requires an issues list")
    seen: set[str] = set()
    required = {
        "id",
        "title",
        "platforms",
        "condition",
        "signature",
        "affected_targets",
        "classification",
        "handling",
        "blocks_tiers",
        "retry_allowed",
        "workaround",
        "first_known_evidence",
        "status",
    }
    for issue in issues:
        missing = required - set(issue)
        if missing:
            raise ConfigurationError(f"issue is missing fields: {sorted(missing)}")
        issue_id = issue["id"]
        if issue_id in seen:
            raise ConfigurationError(f"duplicate issue id {issue_id!r}")
        seen.add(issue_id)
        for target in issue["affected_targets"]:
            path_text = str(target).split("::", 1)[0]
            if not (root / path_text).exists():
                raise ConfigurationError(f"issue {issue_id!r} references missing target {target!r}")


def deduplicate(values: Iterable[str]) -> tuple[list[str], int]:
    unique: list[str] = []
    seen: set[str] = set()
    requested = 0
    for value in values:
        requested += 1
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique, requested - len(unique)


def _expand_suite(name: str, suites: dict[str, Any], stack: tuple[str, ...] = ()) -> list[str]:
    if name in stack:
        raise ConfigurationError(f"cyclic suite reference: {' -> '.join((*stack, name))}")
    try:
        suite = suites[name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown suite {name!r}") from exc
    targets = list(suite.get("targets", []))
    for reference in suite.get("suites", []):
        targets.extend(_expand_suite(reference, suites, (*stack, name)))
    return targets


def resolve_targets(
    matrix: dict[str, Any],
    *,
    tier: str,
    suites: Sequence[str] = (),
    phase: str | None = None,
    extra_targets: Sequence[str] = (),
) -> tuple[list[str], int, int, list[str]]:
    all_suites = matrix["suites"]
    if suites:
        selected = list(suites)
    elif tier == "phase":
        phases = matrix["tiers"]["phase"]["phases"]
        if phase not in phases:
            raise ConfigurationError(f"unknown or missing phase {phase!r}")
        selected = list(phases[phase])
    else:
        selected = list(matrix["tiers"][tier]["default_suites"])
    requested: list[str] = []
    for name in selected:
        requested.extend(_expand_suite(name, all_suites))
    requested.extend(extra_targets)
    unique, removed = deduplicate(requested)
    return unique, len(requested), removed, selected


def issue_is_active(
    issue: dict[str, Any],
    *,
    system: str | None = None,
    module_availability: dict[str, bool] | None = None,
) -> bool:
    current = system or platform.system()
    if "*" not in issue["platforms"] and current not in issue["platforms"]:
        return False
    condition = issue["condition"]
    if condition.get("always") is True:
        return True
    if expected := condition.get("platform"):
        return current == expected
    if module := condition.get("missing_module"):
        if module_availability is not None:
            return not module_availability.get(module, False)
        return importlib.util.find_spec(module) is None
    return False


def run_preflight(
    registry: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
    system: str | None = None,
    module_availability: dict[str, bool] | None = None,
) -> tuple[list[PreflightItem], Path]:
    current = system or platform.system()
    temp_root = root / ".tmp" / "pytest"
    temp_root.mkdir(parents=True, exist_ok=True)
    probe = temp_root / ".write-probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    items: list[PreflightItem] = []
    for issue in registry["issues"]:
        applicable = "*" in issue["platforms"] or current in issue["platforms"]
        active = applicable and issue_is_active(
            issue, system=current, module_availability=module_availability
        )
        classification = issue["classification"]
        if not applicable or classification == "KNOWN_TEST_DEBT":
            state = "NOT_APPLICABLE"
        elif active and issue["handling"] == "exclude_when_active":
            state = "BLOCKED"
        else:
            state = "AVAILABLE"
        items.append(
            PreflightItem(
                issue_id=issue["id"],
                state=state,
                classification=classification,
                detail=issue["workaround"],
                active=active,
            )
        )
    return items, temp_root


def _base_target(target: str) -> str:
    return target.split("::", 1)[0].replace("\\", "/").rstrip("/")


def target_selects(affected: str, selected: Sequence[str]) -> bool:
    affected_base = _base_target(affected)
    for target in selected:
        selected_base = _base_target(target)
        if selected_base == "tests" or selected_base == affected_base:
            return True
        if affected.startswith(f"{target}::") or target.startswith(f"{affected}::"):
            return True
    return False


def applicable_issues(
    registry: dict[str, Any],
    selected: Sequence[str],
    *,
    system: str | None = None,
    module_availability: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    return [
        issue
        for issue in registry["issues"]
        if issue_is_active(issue, system=system, module_availability=module_availability)
        and any(target_selects(target, selected) for target in issue["affected_targets"])
    ]


def pytest_exclusions(issues: Sequence[dict[str, Any]]) -> list[str]:
    args: list[str] = []
    for issue in issues:
        if issue["handling"] == "exclude_when_active":
            args.extend(f"--ignore={_base_target(target)}" for target in issue["affected_targets"])
        elif issue["handling"] == "deselect_exact_nodes":
            args.extend(f"--deselect={target}" for target in issue["affected_targets"])
    return deduplicate(args)[0]


def retry_limit_for(issue: dict[str, Any] | None) -> int:
    if not issue or issue["classification"] != "TRANSIENT_EXTERNAL":
        return 0
    if not issue.get("retry_allowed", False):
        return 0
    return max(0, min(int(issue.get("max_retries", 1)), 2))


def classify_outcome(*, command_failed: bool, blocked_issue_ids: Sequence[str]) -> str:
    if command_failed:
        return "FAIL"
    if blocked_issue_ids:
        return "BLOCKED"
    return "PASS"


def _run(name: str, command: Sequence[str], *, root: Path) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout or ""
    print(output, end="" if output.endswith("\n") or not output else "\n")
    return CommandResult(name, completed.returncode, time.perf_counter() - started, output)


def project_python(root: Path = REPO_ROOT) -> Path:
    """Prefer the repository environment while keeping the launcher stdlib-only."""
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), Path(sys.executable))


def probe_module_availability(
    registry: dict[str, Any], python: Path, *, root: Path = REPO_ROOT
) -> dict[str, bool]:
    modules = sorted(
        {
            issue["condition"]["missing_module"]
            for issue in registry["issues"]
            if "missing_module" in issue["condition"]
        }
    )
    script = (
        "import importlib.util,json,sys;"
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in sys.argv[1:]}))"
    )
    completed = subprocess.run(
        [str(python), "-c", script, *modules],
        cwd=root,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ConfigurationError(
            f"cannot probe test interpreter {python}: {(completed.stderr or completed.stdout).strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"test interpreter returned invalid preflight data: {completed.stdout!r}") from exc
    return {str(name): bool(available) for name, available in result.items()}


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {name: 0 for name in ("passed", "failed", "skipped", "errors")}
    for name in counts:
        matches = re.findall(rf"(\d+) {name[:-1] if name == 'errors' else name}", output)
        if matches:
            counts[name] = int(matches[-1])
    return counts


def _changed_python_files(root: Path) -> list[str]:
    commands = (
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    paths: list[str] = []
    for command in commands:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            paths.extend(line.strip() for line in completed.stdout.splitlines() if line.strip().endswith(".py"))
    return [path for path in deduplicate(paths)[0] if (root / path).exists()]


def execute(
    *,
    tier: str,
    suites: Sequence[str] = (),
    phase: str | None = None,
    extra_targets: Sequence[str] = (),
    dry_run: bool = False,
    preflight_only: bool = False,
    skip_ruff: bool = False,
    skip_diff: bool = False,
    root: Path = REPO_ROOT,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    matrix = load_json(MATRIX_PATH)
    registry = load_json(ISSUES_PATH)
    validate_matrix(matrix, root=root)
    validate_issues(registry, root=root)
    targets, requested_count, duplicates_removed, selected_suites = resolve_targets(
        matrix, tier=tier, suites=suites, phase=phase, extra_targets=extra_targets
    )
    test_python = project_python(root)
    module_availability = probe_module_availability(registry, test_python, root=root)
    preflight, temp_parent = run_preflight(
        registry, root=root, module_availability=module_availability
    )
    active = applicable_issues(
        registry, targets, module_availability=module_availability
    )
    blocking = [
        issue
        for issue in active
        if issue["classification"] in ENVIRONMENT_CLASSIFICATIONS
        and tier in issue["blocks_tiers"]
        and issue["handling"] == "exclude_when_active"
    ]
    exclusions = pytest_exclusions(active)
    basetemp = temp_parent / f"{tier}-{os.getpid()}"
    pytest_command = [
        str(test_python),
        "-m",
        "pytest",
        "-q",
        f"--basetemp={basetemp}",
        *exclusions,
        *targets,
    ]
    print(
        f"PREFLIGHT platform={platform.system()} launcher_python={platform.python_version()} "
        f"test_python={test_python} root={root}"
    )
    print(f"PREFLIGHT temp_root={basetemp}")
    for item in preflight:
        print(f"{item.state} {item.issue_id}: {item.detail}")
    print(
        f"PLAN tier={tier} suites={','.join(selected_suites)} "
        f"requested_targets={requested_count} unique_targets={len(targets)} "
        f"duplicates_removed={duplicates_removed} "
        f"pytest_processes={0 if dry_run or preflight_only else 1}"
    )
    results: list[CommandResult] = []
    if not dry_run and not preflight_only:
        results.append(_run("pytest", pytest_command, root=root))
        if not skip_ruff:
            changed_python = _changed_python_files(root)
            if changed_python:
                results.append(_run("ruff", [str(test_python), "-m", "ruff", "check", *changed_python], root=root))
        if not skip_diff:
            results.append(_run("git_diff_check", ["git", "diff", "--check"], root=root))
    command_failed = any(result.returncode != 0 for result in results)
    outcome = classify_outcome(
        command_failed=command_failed,
        blocked_issue_ids=[issue["id"] for issue in blocking],
    )
    counts = _pytest_counts(next((r.output for r in results if r.name == "pytest"), ""))
    summary = {
        "schema": "x-harness.test-run-summary.v1",
        "tier": tier,
        "phase_or_suites": phase or list(selected_suites),
        "platform": platform.system(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "requested_test_targets": requested_count,
        "unique_pytest_targets": len(targets),
        "duplicate_targets_removed": duplicates_removed,
        "pytest_process_launches": 0 if dry_run or preflight_only else 1,
        "tests": counts,
        "blocked_suites": deduplicate(
            target for issue in blocking for target in issue["affected_targets"]
        )[0],
        "known_environment_issue_ids": [issue["id"] for issue in blocking],
        "known_test_debt_ids": [
            issue["id"] for issue in active if issue["classification"] == "KNOWN_TEST_DEBT"
        ],
        "unknown_failures": command_failed,
        "retries_performed": 0,
        "ruff": next(("PASS" if r.returncode == 0 else "FAIL" for r in results if r.name == "ruff"), "NOT_RUN"),
        "git_diff_check": next(
            ("PASS" if r.returncode == 0 else "FAIL" for r in results if r.name == "git_diff_check"),
            "NOT_RUN",
        ),
        "final": outcome,
    }
    print("TEST_SUMMARY " + json.dumps(summary, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[outcome], summary
