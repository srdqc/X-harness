"""Independent semantic verifiers for the JEV.6 held-out v3 suite."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from .jev6_v2_verifiers import SPECS as V2_SPECS
from .jev6_v3_suite import RETAINED_TASK_IDS, TASKS
from .jev6_verifiers import VerifierResult, VerifierSpec

_GIT_EXECUTABLE = shutil.which("git")
if _GIT_EXECUTABLE is None:
    raise RuntimeError("git executable is required for JEV.6V3 verification")

_PROVIDER_RESOLUTION_PROBE = r'''
from pico.providers.registry import PROVIDERS, resolve_provider_spec
before = tuple(PROVIDERS)
assert resolve_provider_spec(provider_name="deepseek", model="claude-3") is next(s for s in PROVIDERS if s.name == "deepseek")
assert resolve_provider_spec(provider_name="openrouter", model="deepseek-chat") is next(s for s in PROVIDERS if s.name == "openrouter")
assert resolve_provider_spec(api_key="sk-or-example", model="claude-3") is next(s for s in PROVIDERS if s.name == "openrouter")
assert resolve_provider_spec(api_base="http://localhost:11434/v1", model="deepseek-chat") is next(s for s in PROVIDERS if s.name == "ollama")
assert resolve_provider_spec(model="deepseek-chat") is next(s for s in PROVIDERS if s.name == "deepseek")
assert resolve_provider_spec(provider_name="missing", model="unknown-model") is None
assert tuple(PROVIDERS) == before
'''

_V2_BY_TASK = {
    task.task_id: next(spec for spec in V2_SPECS if spec.verifier_id == task.verifier_id)
    for task in __import__(
        "benchmarks.picobench.packs.knowledge_evolution_live.jev6_v2_suite",
        fromlist=["TASKS"],
    ).TASKS
}
SPECS = tuple(_V2_BY_TASK[task_id] for task_id in RETAINED_TASK_IDS) + (
    VerifierSpec(
        "jev6v3-v-provider-resolution",
        ("pico/providers/registry.py",),
        "tests/test_provider_catalog.py",
        _PROVIDER_RESOLUTION_PROBE,
    ),
)


def verifier_set_digest() -> str:
    ordered = tuple(
        (
            task.verifier_id,
            next(spec.digest for spec in SPECS if spec.verifier_id == task.verifier_id),
        )
        for task in TASKS
    )
    return canonical_digest(ordered)


def spec_by_id(verifier_id: str) -> VerifierSpec:
    return next(spec for spec in SPECS if spec.verifier_id == verifier_id)


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        [_GIT_EXECUTABLE, "status", "--porcelain"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(sorted(line[3:].replace("\\", "/") for line in completed.stdout.splitlines()))


def verify_task(
    task_id: str,
    workspace: Path,
    *,
    python_executable: str | None = None,
) -> VerifierResult:
    task = next(task for task in TASKS if task.task_id == task_id)
    spec = spec_by_id(task.verifier_id)
    findings: list[str] = []
    changed = _changed_paths(workspace)
    allowed = (*spec.production_paths, spec.test_path)
    if any(not any(fnmatch.fnmatch(path, pattern) for pattern in allowed) for path in changed):
        findings.append("prohibited_change_detected")
    if any(
        not any(fnmatch.fnmatch(path, pattern) for path in changed)
        for pattern in spec.production_paths
    ):
        findings.append("required_production_change_missing")
    if spec.test_path not in changed:
        findings.append("required_test_change_missing")
    executable = python_executable or sys.executable
    probe = subprocess.run(
        [executable, "-c", spec.semantic_probe],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if probe.returncode != 0:
        findings.append("semantic_probe_failed")
    tests = subprocess.run(
        [executable, "-m", "pytest", spec.test_path, "-q"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if tests.returncode != 0:
        findings.append("targeted_test_failed")
    return VerifierResult(not findings, tuple(findings), spec.verifier_id, spec.digest)


__all__ = ["SPECS", "spec_by_id", "verifier_set_digest", "verify_task"]
