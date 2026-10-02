"""Non-live mechanical task/verifier contract audits for the P3R Pilot."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .verifiers import (
    VERIFIERS,
    changed_path_findings,
    has_two_source_router_proof,
    verify_workspace,
)


@dataclass(frozen=True)
class SolvabilityAudit:
    task_id: str
    passed: bool
    findings: tuple[str, ...]


def audit_pilot_solvability(
    repository: Path,
    *,
    execute_references: bool = True,
    python_executable: str = sys.executable,
) -> tuple[SolvabilityAudit, ...]:
    """Prove bounded reference paths exist without exposing them to evaluated worktrees."""

    debug_path = repository / "pico/knowledge_evolution/canonicalize.py"
    try:
        debug_source = debug_path.read_text(encoding="utf-8")
    except OSError:
        debug_source = ""
    debug_findings = list(
        changed_path_findings(
            VERIFIERS["p3r-v-debug-guard"],
            ("pico/knowledge_evolution/canonicalize.py", "tests/test_knowledge_canonicalization.py"),
        )
    )
    old_guard = 'fingerprints.append((f"guard:{index}", structural_digest({"guard": guard})))'
    reference_guard = 'f"guard:{guard}"'
    if old_guard not in debug_source:
        debug_findings.append("reference_patch_anchor_missing")
    else:
        reference_source = debug_source.replace(
            old_guard,
            (
                'key = f"guard:{guard}" if guard.startswith('
                '("tool:", "binary:", "file:", "dependency:")) else f"guard:{index}"\n'
                '        fingerprints.append((key, structural_digest({"guard": guard})))'
            ),
            1,
        )
        if reference_guard not in reference_source or old_guard in reference_source:
            debug_findings.append("reference_fixture_failed")
    if not all((repository / path).is_file() for path in VERIFIERS["p3r-v-debug-guard"].targeted_tests):
        debug_findings.append("targeted_test_missing")

    integration_spec = VERIFIERS["p3r-v-integration-skill"]
    integration_source = repository / "pico/memory_engine/skill_forge/knowledge_source.py"
    production_findings = changed_path_findings(
        integration_spec,
        ("pico/memory_engine/skill_forge/knowledge_source.py", "tests/test_knowledge_runtime_integration.py"),
    )
    proof_findings = changed_path_findings(
        integration_spec,
        ("tests/test_knowledge_runtime_integration.py",),
    )
    incomplete_findings = changed_path_findings(
        integration_spec,
        ("tests/test_scratch_two_source.py",),
    )
    integration_findings = [*production_findings, *proof_findings]
    try:
        source_text = integration_source.read_text(encoding="utf-8")
    except OSError:
        source_text = ""
    if "ApplicableKnowledgeSkillSource" not in source_text:
        integration_findings.append("reference_fixture_failed")
    proof_path = repository / "tests/test_knowledge_runtime_integration.py"
    try:
        proof_source = proof_path.read_text(encoding="utf-8")
    except OSError:
        proof_source = ""
    reference_proof = proof_source + "\nproof_router = SkillForgeRouter([local_source, p3_source])\n"
    if not has_two_source_router_proof(reference_proof):
        integration_findings.append("reference_fixture_failed")
    if not incomplete_findings:
        integration_findings.append("incomplete_proof_was_accepted")
    if not all((repository / path).is_file() for path in integration_spec.targeted_tests):
        integration_findings.append("targeted_test_missing")

    if execute_references and not debug_findings and not integration_findings:
        reference_findings = _execute_reference_verifiers(repository, python_executable)
        debug_findings.extend(reference_findings["p3r-debug-01"])
        integration_findings.extend(reference_findings["p3r-int-01"])

    return (
        SolvabilityAudit("p3r-debug-01", not debug_findings, tuple(debug_findings)),
        SolvabilityAudit(
            "p3r-int-01",
            not integration_findings,
            tuple(integration_findings),
        ),
    )


def _execute_reference_verifiers(
    repository: Path,
    python_executable: str,
) -> dict[str, tuple[str, ...]]:
    """Run sealed verifiers in disposable sibling worktrees, never the evaluated checkout."""

    results: dict[str, tuple[str, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="p3r-solvability-") as temporary:
        root = Path(temporary)
        for task_id, verifier_id, apply_fixture in (
            ("p3r-debug-01", "p3r-v-debug-guard", _apply_debug_reference),
            ("p3r-int-01", "p3r-v-integration-skill", _apply_integration_reference),
        ):
            worktree = root / task_id
            try:
                _git(repository, "worktree", "add", "--detach", str(worktree), "HEAD")
                apply_fixture(worktree)
                result = verify_workspace(
                    verifier_id,
                    worktree,
                    python_executable=python_executable,
                )
                results[task_id] = () if result.passed else result.findings
            except (OSError, RuntimeError):
                results[task_id] = ("reference_fixture_failed",)
            finally:
                if worktree.exists():
                    try:
                        _git(repository, "worktree", "remove", "--force", str(worktree))
                    except (OSError, RuntimeError):
                        results[task_id] = ("reference_fixture_failed",)
    return results


def _apply_debug_reference(worktree: Path) -> None:
    source_path = worktree / "pico/knowledge_evolution/canonicalize.py"
    source = source_path.read_text(encoding="utf-8")
    old = 'fingerprints.append((f"guard:{index}", structural_digest({"guard": guard})))'
    new = (
        'key = f"guard:{guard}" if guard.startswith('
        '("tool:", "binary:", "file:", "dependency:")) else f"guard:{index}"\n'
        '        fingerprints.append((key, structural_digest({"guard": guard})))'
    )
    if old not in source:
        raise RuntimeError("debug reference anchor missing")
    source_path.write_text(source.replace(old, new, 1), encoding="utf-8")
    test_path = worktree / "tests/test_knowledge_canonicalization.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + "\n\ndef test_machine_guard_reference_fixture():\n"
        + '    result = validate_and_canonicalize(_proposal(applicability=("file:pyproject.toml",)))\n'
        + '    assert result.proposal.applicability_fingerprints[0][0] == "guard:file:pyproject.toml"\n',
        encoding="utf-8",
    )


def _apply_integration_reference(worktree: Path) -> None:
    test_path = worktree / "tests/test_knowledge_runtime_integration.py"
    source = test_path.read_text(encoding="utf-8")
    source = source.replace(
        "from pico.memory_engine.skill_forge import SkillForgeRouter",
        "from pico.memory_engine.skill_forge import LocalSkillSource, SkillForgeRouter",
        1,
    ).replace(
        "from pico.memory_engine.skill_local.registry import SkillRegistry",
        (
            "from pico.memory_engine.skill_local.local_pool import LocalPool\n"
            "from pico.memory_engine.skill_local.registry import SkillRegistry"
        ),
        1,
    )
    old = "builder = SkillsSegmentBuilder(SkillForgeRouter([source]), skill_top_k=5, activation_max=1)"
    new = (
        'ordinary = workspace / "skills" / "ordinary"\n'
        "    ordinary.mkdir(parents=True)\n"
        "    (ordinary / \"SKILL.md\").write_text(\n"
        '        "---\\nname: ordinary\\ndescription: ordinary local skill\\n---\\nordinary",\n'
        '        encoding="utf-8",\n'
        "    )\n"
        "    local_registry = SkillRegistry(workspace, builtin_skills_dir=tmp_path / \"none-local\")\n"
        "    local_source = LocalSkillSource(LocalPool(local_registry), local_registry)\n"
        "    builder = SkillsSegmentBuilder(\n"
        "        SkillForgeRouter([local_source, source]), skill_top_k=5, activation_max=1\n"
        "    )"
    )
    if old not in source:
        raise RuntimeError("integration reference anchor missing")
    test_path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = ["SolvabilityAudit", "audit_pilot_solvability"]
