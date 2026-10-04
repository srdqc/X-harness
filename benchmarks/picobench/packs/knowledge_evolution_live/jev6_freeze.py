"""Offline-only audits used to freeze the JEV.6 held-out suite."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest

from .jev6_fixtures import apply_reference, reference_digest
from .jev6_suite import BASE_COMMIT, TASKS
from .jev6_verifiers import verify_task
from .selectivity import evaluate_selectivity


@dataclass(frozen=True)
class SolvabilityResult:
    task_id: str
    passed: bool
    findings: tuple[str, ...]
    verifier_digest: str
    reference_digest: str


def evaluate_selector_audit(*, state_root: Path, workspace: Path) -> dict[str, Any]:
    raw = evaluate_selectivity(
        state_root=state_root,
        workspace=workspace,
        reviewer_id="human:jev6-offline",
        task_ids=tuple(task.task_id for task in TASKS),
        tasks=TASKS,
    )
    tasks = raw["modes"]["task_relevance_v1"]["tasks"]
    counts = {task_id: int(row["selected_count"]) for task_id, row in tasks.items()}
    selected_sets = {
        task_id: tuple(row["selected_candidate_ids"]) for task_id, row in tasks.items()
    }
    result = {
        "schema": "pico.jev6-selector-audit.v1",
        "corpus_digest": raw["corpus_digest"],
        "selector_version": raw["selector_version"],
        "task_selected_counts": counts,
        "task_selected_candidate_ids": selected_sets,
        "zero_selection_task_count": sum(value == 0 for value in counts.values()),
        "one_selection_task_count": sum(value == 1 for value in counts.values()),
        "many_selection_task_count": sum(value > 1 for value in counts.values()),
        "selected_set_jaccard": raw["modes"]["task_relevance_v1"]["selected_set_jaccard"],
        "selection_distribution_policy": "reported_as_observed_without_prompt_rewriting",
    }
    return {**result, "semantic_digest": canonical_digest(result)}


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
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
    with tempfile.TemporaryDirectory(prefix="jev6-solvability-", dir=temp_parent) as root:
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


def historical_campaign_classifications(repository: Path) -> dict[str, str]:
    expected = {
        "jev4-9168e909e0c47dfb": "HOLD_AND_FORENSIC",
        "jev4r-3f7c246afe462f5e": "HOLD_INFRA",
        "jev4r2-868dd3997095b407": "HOLD_INFRA",
        "jev4r3-384e7aefbe9cca17": "INFRA_PASS_BEHAVIOR_MIXED",
    }
    actual: dict[str, str] = {}
    for campaign_id in expected:
        path = repository / ".p3r" / campaign_id / "reduced.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        actual[campaign_id] = str(value["classification"])
    if actual != expected:
        raise ValueError("historical JEV campaign classification drift")
    return actual


def solvability_payload(results: tuple[SolvabilityResult, ...]) -> dict[str, Any]:
    payload = {
        "schema": "pico.jev6-mechanical-solvability.v1",
        "base_commit": BASE_COMMIT,
        "results": tuple(asdict(result) for result in results),
        "passed_count": sum(result.passed for result in results),
    }
    return {**payload, "semantic_digest": canonical_digest(payload)}


__all__ = [
    "SolvabilityResult",
    "audit_mechanical_solvability",
    "evaluate_selector_audit",
    "historical_campaign_classifications",
    "solvability_payload",
]
