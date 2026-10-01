from __future__ import annotations

import json
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
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    ReviewDecision,
    ReviewerType,
    SourceTurnReference,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    create_review,
    validate_candidate,
)
from pico.knowledge_evolution.store import KnowledgeStoreError
from pico.knowledge_evolution.types import structural_digest

_NOW = "2026-10-01T00:00:00Z"


def _prepared(tmp_path: Path, candidate_id: str = "candidate-1"):
    store = KnowledgeRecordStore(tmp_path)
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:lifecycle",
        evidence=(("project_key", "lifecycle"),),
    )
    store.write_scope(scope)
    success = TaskSuccessEvidence.create(
        evidence_id=f"success-{candidate_id}",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id=f"turn-{candidate_id}",
        status=TaskSuccessStatus.PASS,
        producer_id="sealed-verifier",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest="a" * 64,
        result_digest="b" * 64,
        provenance_refs=("artifact:verified",),
        created_at=_NOW,
    )
    store.write_task_success(success)
    candidate = KnowledgeCandidate.create(
        candidate_id=candidate_id,
        candidate_type=CandidateType.EXPERIENCE,
        content_class=ContentClass.STRATEGY,
        qualifying_sources=(
            SourceTurnReference(
                turn_id=success.source_turn_id,
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=(success.evidence_id,),
                task_success_evidence_digests=(success.evidence_digest,),
            ),
        ),
        repository_scope_id=scope.repository_scope_id,
        title="Verify before reuse",
        reusable_content="Use independent verification before accepting reusable guidance.",
        preconditions=("Repository scope matches",),
        applicability_fingerprints=(("guard:0", structural_digest({"guard": "scope"})),),
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
        validation_id=f"validation-{candidate_id}",
        validator_id="p3-technical-validator",
        validator_version="1",
        validated_at=_NOW,
    )
    return store, candidate, comparison, validation


def _pending(store, candidate, validation):
    manager = KnowledgeLifecycleManager(store)
    manager.transition(
        transition_id="transition-eligible",
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
        transition_id="transition-validated",
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
        transition_id="transition-review",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.VALIDATED,
        to_state=LifecycleState.PENDING_REVIEW,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.SUBMITTED_FOR_REVIEW,
        transitioned_at=_NOW,
        validation=validation,
    )
    return manager


def _approve(store, candidate, comparison, validation, review_id="review-1"):
    return create_review(
        store,
        review_id=review_id,
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:alice",
        decision=ReviewDecision.APPROVE,
        reason="Approved for repository-scoped reuse.",
        reviewed_at=_NOW,
    )


def _activate(store, candidate, comparison, validation):
    manager = _pending(store, candidate, validation)
    review = _approve(store, candidate, comparison, validation)
    active = manager.transition(
        transition_id="transition-active",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.PENDING_REVIEW,
        to_state=LifecycleState.ACTIVE,
        actor_type=LifecycleActorType.HUMAN,
        actor_id=review.reviewer_id,
        reason=LifecycleReason.HUMAN_APPROVED,
        transitioned_at=_NOW,
        validation=validation,
        review=review,
    )
    return manager, review, active


def test_candidate_starts_extracted_and_legal_chain_rebuilds_after_restart(tmp_path: Path) -> None:
    store, candidate, comparison, validation = _prepared(tmp_path)
    manager = KnowledgeLifecycleManager(store)
    assert manager.rebuild(candidate.candidate_id).state is LifecycleState.EXTRACTED
    manager, _, active = _activate(store, candidate, comparison, validation)
    projection = KnowledgeLifecycleManager(KnowledgeRecordStore(tmp_path)).rebuild(
        candidate.candidate_id
    )
    assert projection.state is LifecycleState.ACTIVE
    assert projection.head_transition_digest == active.transition_digest
    assert len(projection.transitions) == 4


def test_illegal_transition_and_from_state_mismatch_fail_closed(tmp_path: Path) -> None:
    store, candidate, _, _ = _prepared(tmp_path)
    manager = KnowledgeLifecycleManager(store)
    with pytest.raises(ValueError, match="illegal"):
        manager.transition(
            transition_id="bad-active",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.EXTRACTED,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.SYSTEM,
            actor_id="p3",
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
        )
    with pytest.raises(ValueError, match="from_state"):
        manager.transition(
            transition_id="bad-from",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.ELIGIBLE,
            to_state=LifecycleState.VALIDATED,
            actor_type=LifecycleActorType.SYSTEM,
            actor_id="p3",
            reason=LifecycleReason.TECHNICAL_VALIDATION_PASSED,
            transitioned_at=_NOW,
        )


def test_pending_review_cannot_activate_without_matching_human_review(tmp_path: Path) -> None:
    store, candidate, _, validation = _prepared(tmp_path)
    manager = _pending(store, candidate, validation)
    with pytest.raises(ValueError, match="human approval"):
        manager.transition(
            transition_id="missing-review-active",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.PENDING_REVIEW,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.HUMAN,
            actor_id="human:alice",
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
            validation=validation,
        )


def test_exact_duplicate_is_blocked_and_conflict_requires_explicit_resolution(tmp_path: Path) -> None:
    exact_store, candidate, _, _ = _prepared(tmp_path / "exact")
    exact = CandidateComparison(
        CandidateRelation.EXACT_DUPLICATE,
        candidate.candidate_id,
        ("candidate-existing",),
        1,
    )
    exact_validation = validate_candidate(
        exact_store,
        candidate_id=candidate.candidate_id,
        comparison=exact,
        expected_repository_scope_id=candidate.repository_scope_id,
        validation_id="validation-exact",
        validator_id="p3-validator",
        validator_version="1",
        validated_at=_NOW,
    )
    exact_manager = _pending(exact_store, candidate, exact_validation)
    exact_review = _approve(exact_store, candidate, exact, exact_validation, "review-exact")
    with pytest.raises(ValueError, match="exact duplicate"):
        exact_manager.transition(
            transition_id="exact-active",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.PENDING_REVIEW,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.HUMAN,
            actor_id="human:alice",
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
            validation=exact_validation,
            review=exact_review,
        )

    conflict_store, conflict_candidate, _, _ = _prepared(tmp_path / "conflict")
    conflict = CandidateComparison(
        CandidateRelation.CONFLICT,
        conflict_candidate.candidate_id,
        ("candidate-conflicting",),
        1,
    )
    conflict_validation = validate_candidate(
        conflict_store,
        candidate_id=conflict_candidate.candidate_id,
        comparison=conflict,
        expected_repository_scope_id=conflict_candidate.repository_scope_id,
        validation_id="validation-conflict",
        validator_id="p3-validator",
        validator_version="1",
        validated_at=_NOW,
    )
    conflict_manager = _pending(conflict_store, conflict_candidate, conflict_validation)
    unresolved = _approve(
        conflict_store,
        conflict_candidate,
        conflict,
        conflict_validation,
        "review-conflict",
    )
    with pytest.raises(ValueError, match="conflict remains"):
        conflict_manager.transition(
            transition_id="conflict-active",
            candidate_id=conflict_candidate.candidate_id,
            from_state=LifecycleState.PENDING_REVIEW,
            to_state=LifecycleState.ACTIVE,
            actor_type=LifecycleActorType.HUMAN,
            actor_id="human:alice",
            reason=LifecycleReason.HUMAN_APPROVED,
            transitioned_at=_NOW,
            validation=conflict_validation,
            review=unresolved,
        )
def test_digest_chain_tampering_and_malformed_line_are_detected(tmp_path: Path) -> None:
    store, candidate, _, validation = _prepared(tmp_path)
    manager = _pending(store, candidate, validation)
    path = store.lifecycle / candidate.repository_scope_id / f"{candidate.candidate_id}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["previous_transition_digest"] = "0" * 64
    lines[-1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(KnowledgeStoreError):
        manager.rebuild(candidate.candidate_id)
    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(KnowledgeStoreError):
        manager.rebuild(candidate.candidate_id)


def test_active_can_be_deprecated_without_deleting_history(tmp_path: Path) -> None:
    store, candidate, comparison, validation = _prepared(tmp_path)
    manager, _, _ = _activate(store, candidate, comparison, validation)
    manager.transition(
        transition_id="transition-deprecated",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.DEPRECATED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_DEPRECATED,
        transitioned_at=_NOW,
    )
    projection = manager.rebuild(candidate.candidate_id)
    assert projection.state is LifecycleState.DEPRECATED
    assert store.read_candidate(candidate.candidate_id) == candidate


def test_supersession_requires_explicit_reviewed_same_scope_replacement(tmp_path: Path) -> None:
    store, candidate, comparison, validation = _prepared(tmp_path)
    manager, review, _ = _activate(store, candidate, comparison, validation)
    _, replacement, _, _ = _prepared(tmp_path, "candidate-2")
    with pytest.raises(ValueError):
        manager.transition(
            transition_id="missing-replacement",
            candidate_id=candidate.candidate_id,
            from_state=LifecycleState.ACTIVE,
            to_state=LifecycleState.SUPERSEDED,
            actor_type=LifecycleActorType.HUMAN,
            actor_id="human:alice",
            reason=LifecycleReason.HUMAN_SUPERSEDED,
            transitioned_at=_NOW,
            review=review,
        )
    manager.transition(
        transition_id="transition-superseded",
        candidate_id=candidate.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.SUPERSEDED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_SUPERSEDED,
        transitioned_at=_NOW,
        review=review,
        related_candidate_id=replacement.candidate_id,
    )
    assert manager.rebuild(candidate.candidate_id).state is LifecycleState.SUPERSEDED
