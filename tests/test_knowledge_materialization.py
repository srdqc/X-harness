from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    CandidateComparison,
    CandidateRelation,
    CandidateType,
    ContentClass,
    KnowledgeCandidate,
    KnowledgeLifecycleManager,
    KnowledgeRecordStore,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    MaterializationStatus,
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    ReviewDecision,
    ReviewerType,
    SourceTurnReference,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    create_review,
    materialize_candidate,
    materialized_skill_root,
    validate_candidate,
)
from pico.knowledge_evolution.types import structural_digest
from pico.memory_engine.skill_local.registry import SkillRegistry

_NOW = "2026-10-01T00:00:00Z"


def _active_case(tmp_path: Path, candidate_type: CandidateType, *, activate: bool = True):
    store = KnowledgeRecordStore(tmp_path)
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity=f"project:materialize-{candidate_type.value}",
        evidence=(("project_key", f"materialize-{candidate_type.value}"),),
    )
    store.write_scope(scope)
    success = TaskSuccessEvidence.create(
        evidence_id="success-1",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id="turn-1",
        status=TaskSuccessStatus.PASS,
        producer_id="sealed",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest="a" * 64,
        result_digest="b" * 64,
        provenance_refs=("artifact:1",),
        created_at=_NOW,
    )
    store.write_task_success(success)
    details = {
        CandidateType.MEMORY_FACT: (
            ContentClass.FACT,
            "Package manager",
            "The repository uses uv.",
            ("Repository scope matches",),
            (
                ("fact:package manager", structural_digest({"value": "uv"})),
                ("guard:0", structural_digest({"guard": "scope"})),
            ),
        ),
        CandidateType.EXPERIENCE: (
            ContentClass.WARNING,
            "Revalidate remote state",
            "Revalidate remote state before relying on historical evidence.",
            ("Remote state may change",),
            (("guard:0", structural_digest({"guard": "remote"})),),
        ),
        CandidateType.SKILL_CANDIDATE: (
            ContentClass.PROCEDURE,
            "Validate repository changes",
            "Run deterministic checks.\n\nValidation expectations:\n- Checks pass",
            ("Repository checkout is available",),
            (("guard:0", structural_digest({"guard": "checkout"})),),
        ),
    }[candidate_type]
    candidate = KnowledgeCandidate.create(
        candidate_id=f"candidate-{candidate_type.value}",
        candidate_type=candidate_type,
        content_class=details[0],
        qualifying_sources=(
            SourceTurnReference(
                turn_id="turn-1",
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=(success.evidence_id,),
                task_success_evidence_digests=(success.evidence_digest,),
            ),
        ),
        repository_scope_id=scope.repository_scope_id,
        title=details[1],
        reusable_content=details[2],
        preconditions=details[3],
        applicability_fingerprints=details[4],
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        provenance_refs=("context:bounded",),
        created_at=_NOW,
        created_by="extractor:agent",
    )
    store.write_candidate(candidate)
    comparison = CandidateComparison(CandidateRelation.NEW, candidate.candidate_id, (), 0)
    validation = validate_candidate(
        store,
        candidate_id=candidate.candidate_id,
        comparison=comparison,
        expected_repository_scope_id=scope.repository_scope_id,
        validation_id="validation-1",
        validator_id="p3-validator",
        validator_version="1",
        validated_at=_NOW,
    )
    manager = KnowledgeLifecycleManager(store)
    manager.transition(
        transition_id="eligible",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.EXTRACTED,
        to_state=LifecycleState.ELIGIBLE,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.ELIGIBILITY_CONFIRMED,
        transitioned_at=_NOW,
        evidence_refs=("eligibility:" + "e" * 64,),
    )
    manager.transition(
        transition_id="validated",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ELIGIBLE,
        to_state=LifecycleState.VALIDATED,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.TECHNICAL_VALIDATION_PASSED,
        transitioned_at=_NOW,
        validation=validation,
    )
    manager.transition(
        transition_id="pending",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.VALIDATED,
        to_state=LifecycleState.PENDING_REVIEW,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.SUBMITTED_FOR_REVIEW,
        transitioned_at=_NOW,
        validation=validation,
    )
    review = create_review(
        store,
        review_id="review-1",
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:alice",
        decision=ReviewDecision.APPROVE,
        reason="Approved for scoped reusable knowledge.",
        reviewed_at=_NOW,
    )
    if activate:
        manager.transition(
            transition_id="active",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.PENDING_REVIEW,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.HUMAN,
            actor_id="human:alice",
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
            validation=validation,
            review=review,
        )
    return store, candidate, validation, review, manager


def test_active_skill_materializes_with_provenance_and_normal_registry_discovery(
    tmp_path: Path,
) -> None:
    store, candidate, validation, review, _ = _active_case(
        tmp_path / "state", CandidateType.SKILL_CANDIDATE
    )
    result = materialize_candidate(
        store,
        materialization_id="materialization-1",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    assert result.status is MaterializationStatus.CREATED
    path = store.materialized / result.artifact_relative_path
    text = path.read_text(encoding="utf-8")
    assert candidate.candidate_id in text
    assert candidate.manifest_digest in text
    assert review.review_digest in text
    assert result.content_digest == hashlib.sha256(path.read_bytes()).hexdigest()

    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    workspace.mkdir()
    builtin.mkdir()
    assert SkillRegistry(workspace, builtin_skills_dir=builtin).get(candidate.candidate_id) is None
    source = f"experience:{candidate.repository_scope_id}"
    registry = SkillRegistry(
        workspace,
        builtin_skills_dir=builtin,
        extra_dirs=[(materialized_skill_root(store, candidate.repository_scope_id), source, False)],
    )
    discovered = registry.get(candidate.candidate_id, source=source)
    assert discovered is not None
    assert discovered.always is False


def test_nonactive_candidate_cannot_materialize(tmp_path: Path) -> None:
    store, candidate, validation, review, _ = _active_case(
        tmp_path, CandidateType.EXPERIENCE, activate=False
    )
    result = materialize_candidate(
        store,
        materialization_id="nonactive",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    assert result.status is MaterializationStatus.NOT_APPLICABLE
    assert result.artifact_relative_path is None


def test_materialization_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    store, candidate, validation, review, _ = _active_case(
        tmp_path, CandidateType.SKILL_CANDIDATE
    )
    first = materialize_candidate(
        store,
        materialization_id="first",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    second = materialize_candidate(
        store,
        materialization_id="second",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    assert first.status is MaterializationStatus.CREATED
    assert second.status is MaterializationStatus.UNCHANGED
    assert KnowledgeRecordStore(tmp_path).read_materialization_result(second.materialization_id) == second

    path = store.materialized / first.artifact_relative_path
    path.write_text("tampered", encoding="utf-8")
    conflict = materialize_candidate(
        store,
        materialization_id="conflict",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    assert conflict.status is MaterializationStatus.FAILED
    assert path.read_text(encoding="utf-8") == "tampered"


@pytest.mark.parametrize("candidate_type", [CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE])
def test_fact_and_experience_use_scoped_p3_storage(tmp_path: Path, candidate_type) -> None:
    store, candidate, validation, review, _ = _active_case(tmp_path, candidate_type)
    result = materialize_candidate(
        store,
        materialization_id=f"materialize-{candidate_type.value}",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    assert result.status is MaterializationStatus.CREATED
    assert result.artifact_relative_path.startswith(candidate.repository_scope_id + "/")
    expected_directory = "facts/" if candidate_type is CandidateType.MEMORY_FACT else "guidance/"
    assert expected_directory in result.artifact_relative_path
    assert not hasattr(result, "tool_name")


def test_deprecation_retains_artifact_but_is_logically_inactive(tmp_path: Path) -> None:
    store, candidate, validation, review, manager = _active_case(
        tmp_path, CandidateType.EXPERIENCE
    )
    result = materialize_candidate(
        store,
        materialization_id="materialize",
        candidate_id=candidate.candidate_id,
        validation=validation,
        review=review,
    )
    path = store.materialized / result.artifact_relative_path
    manager.transition(
        transition_id="deprecated",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.DEPRECATED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_DEPRECATED,
        transitioned_at=_NOW,
    )
    assert path.exists()
    assert manager.rebuild(candidate.candidate_id).is_active is False


def test_materialized_path_traversal_and_runtime_authority_are_absent(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    with pytest.raises(ValueError):
        store.write_materialized_bytes(("..", "escape"), b"bad")
    source = (Path(__file__).parents[1] / "pico" / "knowledge_evolution" / "materialize.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "MemoryBackend",
        "ToolRegistry",
        "pico.providers",
        "pico.delivery",
        "sandbox",
        "terminal_state",
        "inject_context",
    )
    assert all(item not in source for item in forbidden)
