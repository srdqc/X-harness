"""Offline-only audits used to freeze the final JEV.6 held-out v4 suite."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest

from .jev6_v4_fixtures import apply_reference, reference_digest
from .jev6_v4_suite import (
    BASE_COMMIT,
    EXPOSURE_LEDGER_SCHEMA,
    EXPOSURE_SNAPSHOT_SCHEMA,
    RETIRED_EXPOSED_TASK_IDS,
    TASKS,
)
from .jev6_v4_verifiers import verify_task
from .selectivity import evaluate_candidate_identity_audit

_GIT_EXECUTABLE = shutil.which("git")
if _GIT_EXECUTABLE is None:
    raise RuntimeError("git executable is required for JEV.6V4 freeze audits")


@dataclass(frozen=True)
class SolvabilityResult:
    task_id: str
    passed: bool
    findings: tuple[str, ...]
    verifier_digest: str
    reference_digest: str


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        [_GIT_EXECUTABLE, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def audit_mechanical_solvability(
    repository: Path,
    *,
    python_executable: str | None = None,
) -> tuple[SolvabilityResult, ...]:
    executable = python_executable or sys.executable
    results: list[SolvabilityResult] = []
    temp_parent = repository / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jev6-v4-solvability-", dir=temp_parent) as root:
        for index, task in enumerate(TASKS):
            workspace = Path(root) / f"t{index}"
            _git(repository, "worktree", "add", "--detach", str(workspace), BASE_COMMIT)
            try:
                apply_reference(task.task_id, workspace)
                verified = verify_task(
                    task.task_id,
                    workspace,
                    python_executable=executable,
                )
                results.append(
                    SolvabilityResult(
                        task.task_id,
                        verified.passed,
                        verified.findings,
                        verified.verifier_digest,
                        reference_digest(task.task_id),
                    )
                )
            finally:
                _git(repository, "worktree", "remove", "--force", str(workspace))
    return tuple(results)


def solvability_payload(results: tuple[SolvabilityResult, ...]) -> dict[str, Any]:
    payload = {
        "schema": "pico.jev6-mechanical-solvability.v4",
        "schema_version": 4,
        "base_commit": BASE_COMMIT,
        "results": tuple(asdict(result) for result in results),
        "passed_count": sum(result.passed for result in results),
        "total_count": len(results),
    }
    return {**payload, "semantic_digest": canonical_digest(payload)}


def evaluate_selector_identity_audit(*, state_root: Path, workspace: Path) -> dict[str, Any]:
    return evaluate_candidate_identity_audit(
        state_root=state_root,
        workspace=workspace,
        tasks=TASKS,
        reviewer_id="human:jev6-v4-offline",
    )


def load_exposure_snapshot(repository: Path) -> dict[str, Any]:
    base = repository / "benchmarks/picobench/packs/knowledge_evolution_live"
    ledger = json.loads((base / "jev6_v4_exposure_ledger.json").read_text(encoding="utf-8"))
    snapshot = json.loads((base / "jev6_v4_exposure_snapshot.json").read_text(encoding="utf-8"))
    if ledger["schema"] != EXPOSURE_LEDGER_SCHEMA:
        raise ValueError("unexpected JEV.6 v4 exposure ledger schema")
    if snapshot["schema"] != EXPOSURE_SNAPSHOT_SCHEMA:
        raise ValueError("unexpected JEV.6 v4 exposure snapshot schema")
    if snapshot["source_ledger_digest"] != canonical_digest(ledger):
        raise ValueError("JEV.6 v4 exposure history drift")
    expected = {task.task_id for task in TASKS}
    if set(snapshot["tasks"]) != expected:
        raise ValueError("JEV.6 v4 exposure snapshot task mismatch")
    if any(row["live_agent_exposed"] for row in snapshot["tasks"].values()):
        raise ValueError("JEV.6 v4 contains an exposed task")
    retired = snapshot["retired_history"]
    if not all(retired[task_id]["live_agent_exposed"] for task_id in RETIRED_EXPOSED_TASK_IDS):
        raise ValueError("JEV.6 V3 exposure history was reset")
    if sum(row["live_run_count"] for row in retired.values()) != 7:
        raise ValueError("JEV.6 historical exposure count drift")
    return snapshot


__all__ = [
    "SolvabilityResult",
    "audit_mechanical_solvability",
    "evaluate_selector_identity_audit",
    "load_exposure_snapshot",
    "solvability_payload",
]
