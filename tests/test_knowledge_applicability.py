from __future__ import annotations

from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    ApplicabilityReason,
    ApplicabilityStatus,
    CandidateType,
    KnowledgeRecordStore,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    RepositoryScopeIdentity,
    evaluate_applicability,
)
from pico.knowledge_evolution.types import structural_digest
from tests._knowledge_runtime_helpers import NOW, active_materialized_candidate, file_guard, make_scope


def _evaluate(store, candidate_id, scope, workspace, suffix="1", **kwargs):
    return evaluate_applicability(
        store,
        candidate_id=candidate_id,
        environment=ApplicabilityEnvironment(
            repository_scope_id=scope.repository_scope_id,
            workspace=workspace,
            **kwargs,
        ),
        applicability_id=f"app-{candidate_id}-{suffix}",
        evaluated_at=NOW,
    )


def test_active_matching_file_guard_is_applicable_and_digest_deterministic(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    guard = file_guard(workspace, "pyproject.toml", b"[project]\nname='demo'\n")
    candidate, _, _ = active_materialized_candidate(
        store,
        scope,
        "fact-package",
        CandidateType.MEMORY_FACT,
        title="Package metadata",
        content="The package metadata is declared in pyproject.toml.",
        fingerprints=(guard,),
    )
    first = _evaluate(store, candidate.candidate_id, scope, workspace)
    loaded = store.read_applicability(first.applicability_id)
    assert first.status is ApplicabilityStatus.APPLICABLE
    assert loaded == first
    assert loaded.applicability_digest == first.applicability_digest


def test_changed_guard_is_stale_and_restoration_does_not_mutate_lifecycle(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    guard = file_guard(workspace, "config.ini", b"mode=safe\n")
    candidate, _, manager = active_materialized_candidate(
        store,
        scope,
        "experience-config",
        CandidateType.EXPERIENCE,
        title="Safe configuration",
        content="Check safe configuration before deployment.",
        fingerprints=(guard,),
    )
    (workspace / "config.ini").write_bytes(b"mode=changed\n")
    stale = _evaluate(store, candidate.candidate_id, scope, workspace, "stale")
    assert stale.status is ApplicabilityStatus.STALE
    assert ApplicabilityReason.FINGERPRINT_CHANGED in stale.reasons
    assert manager.rebuild(candidate.candidate_id).state is LifecycleState.ACTIVE
    (workspace / "config.ini").write_bytes(b"mode=safe\n")
    restored = _evaluate(store, candidate.candidate_id, scope, workspace, "restored")
    assert restored.status is ApplicabilityStatus.APPLICABLE


@pytest.mark.parametrize(
    ("fingerprint", "environment", "status", "reason"),
    [
        (("guard:tool:missing_tool", "0" * 64), {}, ApplicabilityStatus.NOT_APPLICABLE, ApplicabilityReason.MISSING_TOOL),
        (("guard:binary:definitely-not-a-real-binary", "0" * 64), {}, ApplicabilityStatus.NOT_APPLICABLE, ApplicabilityReason.MISSING_BINARY),
        (("guard:unknown", "0" * 64), {}, ApplicabilityStatus.INCONCLUSIVE, ApplicabilityReason.MISSING_GUARD_EVIDENCE),
        (
            ("guard:dependency:demo", structural_digest({"version": "1"})),
            {"dependency_versions": (("demo", "2"),)},
            ApplicabilityStatus.STALE,
            ApplicabilityReason.DEPENDENCY_MISMATCH,
        ),
    ],
)
def test_declared_guard_failures_are_fail_closed(tmp_path, fingerprint, environment, status, reason) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    candidate, _, _ = active_materialized_candidate(
        store,
        scope,
        "skill-guard",
        CandidateType.SKILL_CANDIDATE,
        title="Guarded procedure",
        content="Follow the guarded procedure.",
        fingerprints=(fingerprint,),
    )
    result = _evaluate(store, candidate.candidate_id, scope, tmp_path, "guard", **environment)
    assert result.status is status
    assert reason in result.reasons


def test_wrong_scope_deprecated_and_tampered_materialization_are_suppressed(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    candidate, materialization, manager = active_materialized_candidate(
        store,
        scope,
        "experience-state",
        CandidateType.EXPERIENCE,
        title="State handling",
        content="Verify state before continuing.",
        fingerprints=(("guard:explicit", structural_digest({"value": "current"})),),
    )
    other = RepositoryScopeIdentity.create(
        method=scope.method,
        canonical_identity="project:other",
        evidence=(("project_key", "other"),),
    )
    wrong = _evaluate(
        store,
        candidate.candidate_id,
        other,
        tmp_path,
        "wrong",
        fingerprints=(("explicit", structural_digest({"value": "current"})),),
    )
    assert wrong.status is ApplicabilityStatus.NOT_APPLICABLE
    artifact = store.materialized / materialization.artifact_relative_path
    artifact.write_text("tampered", encoding="utf-8")
    corrupt = _evaluate(
        store,
        candidate.candidate_id,
        scope,
        tmp_path,
        "corrupt",
        fingerprints=(("explicit", structural_digest({"value": "current"})),),
    )
    assert corrupt.status is ApplicabilityStatus.INCONCLUSIVE
    manager.transition(
        transition_id="deprecated",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.DEPRECATED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_DEPRECATED,
        transitioned_at=NOW,
    )
    deprecated = _evaluate(store, candidate.candidate_id, scope, tmp_path, "deprecated")
    assert deprecated.status is ApplicabilityStatus.NOT_APPLICABLE

