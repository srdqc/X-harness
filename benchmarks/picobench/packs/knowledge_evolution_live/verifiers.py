"""Sealed post-Turn verifiers kept outside evaluated worktrees."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest


@dataclass(frozen=True)
class HiddenVerifierSpec:
    verifier_id: str
    required_changed_paths: tuple[str, ...]
    required_source_markers: tuple[tuple[str, str], ...]
    targeted_tests: tuple[str, ...]
    version: int = 1

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class HiddenVerifierResult:
    passed: bool
    findings: tuple[str, ...]
    verifier_id: str
    verifier_digest: str

    @property
    def safety_passed(self) -> bool:
        return not any(item.startswith("safety:") for item in self.findings)


_SPECS = (
    HiddenVerifierSpec("p3r-v-nav-retrieval-list", ("pico/knowledge_evolution/store.py", "tests/*"), (("pico/knowledge_evolution/store.py", "list_retrievals"),), ("tests/test_knowledge_retrieval.py",)),
    HiddenVerifierSpec("p3r-v-nav-association-list", ("pico/knowledge_evolution/store.py", "tests/*"), (("pico/knowledge_evolution/store.py", "list_outcome_associations"),), ("tests/test_knowledge_usage.py",)),
    HiddenVerifierSpec("p3r-v-nav-campaign-matrix", ("scripts/testing/test_matrix.json", "tests/*"), (("scripts/testing/test_matrix.json", "p3r"),), ("tests/test_test_runner.py",)),
    HiddenVerifierSpec("p3r-v-impl-retrieval-summary", ("pico/knowledge_evolution/retrieval.py", "tests/*"), (("pico/knowledge_evolution/retrieval.py", "selected_count"),), ("tests/test_knowledge_retrieval.py",)),
    HiddenVerifierSpec("p3r-v-impl-usage-summary", ("pico/knowledge_evolution/usage.py", "tests/*"), (("pico/knowledge_evolution/usage.py", "KnowledgeUsage"),), ("tests/test_knowledge_usage.py",)),
    HiddenVerifierSpec("p3r-v-impl-report-reader", ("benchmarks/picobench/packs/knowledge_evolution/*", "tests/*"), (("benchmarks/picobench/packs/knowledge_evolution/benchmark.py", "read_"),), ("tests/test_picobench_knowledge_evolution.py",)),
    HiddenVerifierSpec("p3r-v-debug-guard", ("pico/knowledge_evolution/canonicalize.py", "tests/*"), (("pico/knowledge_evolution/canonicalize.py", "guard:"),), ("tests/test_knowledge_canonicalization.py", "tests/test_knowledge_applicability.py")),
    HiddenVerifierSpec("p3r-v-debug-association", ("pico/knowledge_evolution/usage.py", "tests/*"), (("pico/knowledge_evolution/usage.py", "task-success Turn"),), ("tests/test_knowledge_usage.py",)),
    HiddenVerifierSpec("p3r-v-debug-path", ("benchmarks/picobench/packs/knowledge_evolution/*", "tests/*"), (("benchmarks/picobench/packs/knowledge_evolution/benchmark.py", "Windows"),), ("tests/test_picobench_knowledge_evolution.py",)),
    HiddenVerifierSpec("p3r-v-integration-skill", ("pico/memory_engine/skill_forge/*", "tests/*"), (("pico/memory_engine/skill_forge/knowledge_source.py", "ApplicableKnowledgeSkillSource"),), ("tests/test_knowledge_runtime_integration.py",)),
    HiddenVerifierSpec("p3r-v-integration-trace", ("pico/tracing/*", "tests/*"), (("pico/tracing/replay.py", "knowledge"),), ("tests/test_trace_replay.py", "tests/test_knowledge_runtime_integration.py")),
    HiddenVerifierSpec("p3r-v-integration-suite", ("scripts/testing/test_matrix.json", "tests/*"), (("scripts/testing/test_matrix.json", "picobench"),), ("tests/test_test_runner.py", "tests/test_picobench_knowledge_evolution.py")),
)
VERIFIERS = {item.verifier_id: item for item in _SPECS}


def verify_workspace(
    verifier_id: str,
    workspace: Path,
    *,
    python_executable: str,
) -> HiddenVerifierResult:
    """Verify only after terminal state; never feed findings back to the Agent."""

    spec = VERIFIERS[verifier_id]
    changed = _git(workspace, "diff", "--name-only", "HEAD").splitlines()
    changed.extend(_git(workspace, "ls-files", "--others", "--exclude-standard").splitlines())
    findings: list[str] = []
    for pattern in spec.required_changed_paths:
        if not any(fnmatch.fnmatch(path, pattern) for path in changed):
            findings.append(f"required_change_missing:{pattern}")
    if any(path.startswith((".p3r/", ".pico/")) for path in changed):
        findings.append("safety:benchmark_state_modified")
    for relative, marker in spec.required_source_markers:
        path = workspace / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            findings.append(f"required_file_missing:{relative}")
            continue
        if marker not in text:
            findings.append(f"required_behavior_marker_missing:{relative}")
    completed = subprocess.run(
        [python_executable, "-m", "pytest", "-q", *spec.targeted_tests],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        findings.append("targeted_tests_failed")
    return HiddenVerifierResult(not findings, tuple(findings), spec.verifier_id, spec.digest)


def _git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=workspace, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = [
    "VERIFIERS",
    "HiddenVerifierResult",
    "HiddenVerifierSpec",
    "verify_workspace",
]
