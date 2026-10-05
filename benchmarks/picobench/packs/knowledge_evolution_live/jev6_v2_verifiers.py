"""Independent semantic verifiers for the JEV.6 held-out v2 suite."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest

from .jev6_v2_suite import RETAINED_TASK_IDS, TASKS
from .jev6_verifiers import SPECS as V1_SPECS
from .jev6_verifiers import VerifierResult, VerifierSpec

_PLUGIN_CONTRIBUTIONS_PROBE = r'''
import sys
from pico.plugin.discover import DiscoveredPlugin, Source
from pico.plugin.manifest import Contributes, MemoryBackendContribution, PluginManifest, ToolContribution
from pico.plugin.registry import PluginRegistry
manifest=PluginManifest(id="owner",version="1",enabled_by_default=True,contributes=Contributes(
    memory_backends=[MemoryBackendContribution(name="zeta",factory="jev6_missing_backend:make")],
    tools=[ToolContribution(name="beta",factory="jev6_missing_tool:make"),ToolContribution(name="alpha",factory="jev6_missing_alpha:make")],
))
registry=PluginRegistry(); registry.activate([DiscoveredPlugin(manifest=manifest,source=Source.USER,location=None)])
assert registry.contributions_for("owner") == (
    ("memory_backend","zeta","jev6_missing_backend:make"),
    ("tool","alpha","jev6_missing_alpha:make"),
    ("tool","beta","jev6_missing_tool:make"),
)
assert registry.contributions_for("missing") == ()
assert registry._resolved_factories == {}
assert not any(name in sys.modules for name in ("jev6_missing_backend","jev6_missing_tool","jev6_missing_alpha"))
'''

_V1_BY_TASK = {
    task.task_id: next(spec for spec in V1_SPECS if spec.verifier_id == task.verifier_id)
    for task in __import__(
        "benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite", fromlist=["TASKS"]
    ).TASKS
}
SPECS = tuple(_V1_BY_TASK[task_id] for task_id in RETAINED_TASK_IDS) + (
    VerifierSpec(
        "jev6v2-v-plugin-contributions",
        ("pico/plugin/registry.py",),
        "tests/test_plugin_registry.py",
        _PLUGIN_CONTRIBUTIONS_PROBE,
    ),
)


def verifier_set_digest() -> str:
    ordered = tuple(
        (task.verifier_id, next(spec.digest for spec in SPECS if spec.verifier_id == task.verifier_id))
        for task in TASKS
    )
    return canonical_digest(ordered)


def spec_by_id(verifier_id: str) -> VerifierSpec:
    return next(spec for spec in SPECS if spec.verifier_id == verifier_id)


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607 -- repository-native executable
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(sorted(line[3:].replace("\\", "/") for line in completed.stdout.splitlines()))


def verify_task(task_id: str, workspace: Path, *, python_executable: str | None = None) -> VerifierResult:
    task = next(task for task in TASKS if task.task_id == task_id)
    spec = spec_by_id(task.verifier_id)
    findings: list[str] = []
    changed = _changed_paths(workspace)
    allowed = (*spec.production_paths, spec.test_path)
    if any(not any(fnmatch.fnmatch(path, pattern) for pattern in allowed) for path in changed):
        findings.append("prohibited_change_detected")
    if any(not any(fnmatch.fnmatch(path, pattern) for path in changed) for pattern in spec.production_paths):
        findings.append("required_production_change_missing")
    if spec.test_path not in changed:
        findings.append("required_test_change_missing")
    executable = python_executable or sys.executable
    probe = subprocess.run(
        [executable, "-c", spec.semantic_probe], cwd=workspace, check=False,
        capture_output=True, text=True, timeout=90,
    )
    if probe.returncode != 0:
        findings.append("semantic_probe_failed")
    tests = subprocess.run(
        [executable, "-m", "pytest", spec.test_path, "-q"], cwd=workspace,
        check=False, capture_output=True, text=True, timeout=300,
    )
    if tests.returncode != 0:
        findings.append("targeted_test_failed")
    return VerifierResult(not findings, tuple(findings), spec.verifier_id, spec.digest)


__all__ = ["SPECS", "spec_by_id", "verifier_set_digest", "verify_task"]
