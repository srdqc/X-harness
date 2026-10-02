from __future__ import annotations

from dataclasses import replace
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


def _prepared(tmp_path: Path):
    store = KnowledgeRecordStore(tmp_path)
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:review",
        evidence=(("project_key", "review"),),
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
    candidate = KnowledgeCandidate.create(
        candidate_id="candidate-1",
        candidate_type=CandidateType.EXPERIENCE,
        content_class=ContentClass.WARNING,
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
        title="Validate external state",
        reusable_content="Revalidate external state before depending on historical evidence.",
        preconditions=("External state may have changed",),
        applicability_fingerprints=(("guard:0", structural_digest({"guard": "external"})),),
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
    return store, candidate, comparison, validation, manager


@pytest.mark.parametrize("decision", tuple(ReviewDecision))
def test_human_review_decisions_are_immutable_and_survive_restart(tmp_path: Path, decision) -> None:
    store, candidate, comparison, validation, manager = _prepared(tmp_path)
    receipt = create_review(
        store,
        review_id=f"review-{decision.value}",
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:alice",
        decision=decision,
        reason=f"Human decision: {decision.value}",
        reviewed_at=_NOW,
    )
    assert KnowledgeRecordStore(tmp_path).read_review(receipt.review_id) == receipt
    assert manager.rebuild(candidate.candidate_id).state is LifecycleState.PENDING_REVIEW


def test_generating_agent_and_nonhuman_interactions_cannot_review(tmp_path: Path) -> None:
    store, candidate, comparison, validation, _ = _prepared(tmp_path)
    with pytest.raises(ValueError, match="generating actor"):
        create_review(
            store,
            review_id="self-review",
            candidate_id=candidate.candidate_id,
            validation=validation,
            comparison=comparison,
            reviewer_type=ReviewerType.HUMAN,
            reviewer_id=candidate.created_by,
            decision=ReviewDecision.APPROVE,
            reason="Self approval",
            reviewed_at=_NOW,
        )
    for nonhuman in ("ask_user", "privileged_approval", "provider", "delivery_recipient"):
        with pytest.raises(ValueError):
            create_review(
                store,
                review_id=f"review-{nonhuman}",
                candidate_id=candidate.candidate_id,
                validation=validation,
                comparison=comparison,
                reviewer_type=nonhuman,
                reviewer_id="human:alice",
                decision=ReviewDecision.APPROVE,
                reason="Not a knowledge review authority",
                reviewed_at=_NOW,
            )


def test_missing_identity_and_mismatched_validation_are_rejected(tmp_path: Path) -> None:
    store, candidate, comparison, validation, _ = _prepared(tmp_path)
    with pytest.raises(ValueError):
        create_review(
            store,
            review_id="missing-reviewer",
            candidate_id=candidate.candidate_id,
            validation=validation,
            comparison=comparison,
            reviewer_type=ReviewerType.HUMAN,
            reviewer_id=" ",
            decision=ReviewDecision.APPROVE,
            reason="Missing identity",
            reviewed_at=_NOW,
        )
    with pytest.raises(ValueError, match="validation evidence"):
        create_review(
            store,
            review_id="bad-validation",
            candidate_id=candidate.candidate_id,
            validation=replace(validation, validation_digest="0" * 64),
            comparison=comparison,
            reviewer_type=ReviewerType.HUMAN,
            reviewer_id="human:alice",
            decision=ReviewDecision.APPROVE,
            reason="Mismatched validation",
            reviewed_at=_NOW,
        )


def test_review_receipt_digest_tampering_fails_closed(tmp_path: Path) -> None:
    store, candidate, comparison, validation, _ = _prepared(tmp_path)
    receipt = create_review(
        store,
        review_id="review-1",
        candidate_id=candidate.candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:alice",
        decision=ReviewDecision.APPROVE,
        reason="Approved",
        reviewed_at=_NOW,
    )
    path = store.reviews / f"{receipt.review_id}.json"
    payload = path.read_text(encoding="utf-8").replace(receipt.review_digest, "0" * 64)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(KnowledgeStoreError, match="validation|digest"):
        store.read_review(receipt.review_id)
