from __future__ import annotations

from pathlib import Path

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
    materialize_candidate,
    validate_candidate,
)
from pico.knowledge_evolution.types import structural_digest

NOW = "2026-10-01T00:00:00Z"


def make_scope(store: KnowledgeRecordStore, name: str = "runtime") -> RepositoryScopeIdentity:
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity=f"project:{name}",
        evidence=(("project_key", name),),
    )
    store.write_scope(scope)
    return scope


def active_materialized_candidate(
    store: KnowledgeRecordStore,
    scope: RepositoryScopeIdentity,
    candidate_id: str,
    candidate_type: CandidateType,
    *,
    title: str,
    content: str,
    fingerprints: tuple[tuple[str, str], ...] = (),
):
    turn_id = f"source-{candidate_id}"
    success = TaskSuccessEvidence.create(
        evidence_id=f"success-{candidate_id}",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id=turn_id,
        status=TaskSuccessStatus.PASS,
        producer_id="sealed",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest="a" * 64,
        result_digest="b" * 64,
        provenance_refs=("artifact:verified",),
        created_at=NOW,
    )
    store.write_task_success(success)
    content_class = {
        CandidateType.MEMORY_FACT: ContentClass.FACT,
        CandidateType.EXPERIENCE: ContentClass.STRATEGY,
        CandidateType.SKILL_CANDIDATE: ContentClass.PROCEDURE,
    }[candidate_type]
    required = list(fingerprints)
    if candidate_type is CandidateType.MEMORY_FACT:
        required.append(("fact:value", structural_digest({"value": content})))
    if not any(key.startswith("guard:") for key, _ in required):
        required.append(("guard:explicit", structural_digest({"value": "current"})))
    if candidate_type is CandidateType.SKILL_CANDIDATE and "Validation expectations:" not in content:
        content += "\n\nValidation expectations:\n- Deterministic checks pass"
    candidate = KnowledgeCandidate.create(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        content_class=content_class,
        qualifying_sources=(
            SourceTurnReference(
                turn_id=turn_id,
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=(success.evidence_id,),
                task_success_evidence_digests=(success.evidence_digest,),
            ),
        ),
        repository_scope_id=scope.repository_scope_id,
        title=title,
        reusable_content=content,
        preconditions=("Repository scope and declared guards match",),
        applicability_fingerprints=tuple(required),
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        provenance_refs=("context:bounded",),
        created_at=NOW,
        created_by="extractor:agent",
    )
    store.write_candidate(candidate)
    comparison = CandidateComparison(CandidateRelation.NEW, candidate_id, (), 0)
    validation = validate_candidate(
        store,
        candidate_id=candidate_id,
        comparison=comparison,
        expected_repository_scope_id=scope.repository_scope_id,
        validation_id=f"validation-{candidate_id}",
        validator_id="p3-validator",
        validator_version="1",
        validated_at=NOW,
    )
    manager = KnowledgeLifecycleManager(store)
    manager.transition(
        transition_id=f"eligible-{candidate_id}",
        candidate_id=candidate_id,
        from_state=LifecycleState.EXTRACTED,
        to_state=LifecycleState.ELIGIBLE,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.ELIGIBILITY_CONFIRMED,
        transitioned_at=NOW,
        evidence_refs=("eligibility:" + "e" * 64,),
    )
    manager.transition(
        transition_id=f"validated-{candidate_id}",
        candidate_id=candidate_id,
        from_state=LifecycleState.ELIGIBLE,
        to_state=LifecycleState.VALIDATED,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.TECHNICAL_VALIDATION_PASSED,
        transitioned_at=NOW,
        validation=validation,
    )
    manager.transition(
        transition_id=f"pending-{candidate_id}",
        candidate_id=candidate_id,
        from_state=LifecycleState.VALIDATED,
        to_state=LifecycleState.PENDING_REVIEW,
        actor_type=LifecycleActorType.SYSTEM,
        actor_id="p3",
        reason=LifecycleReason.SUBMITTED_FOR_REVIEW,
        transitioned_at=NOW,
        validation=validation,
    )
    review = create_review(
        store,
        review_id=f"review-{candidate_id}",
        candidate_id=candidate_id,
        validation=validation,
        comparison=comparison,
        reviewer_type=ReviewerType.HUMAN,
        reviewer_id="human:alice",
        decision=ReviewDecision.APPROVE,
        reason="Approved for scoped reuse.",
        reviewed_at=NOW,
    )
    manager.transition(
        transition_id=f"active-{candidate_id}",
        candidate_id=candidate_id,
        from_state=LifecycleState.PENDING_REVIEW,
        to_state=LifecycleState.ACTIVE,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_APPROVED,
        transitioned_at=NOW,
        validation=validation,
        review=review,
    )
    materialization = materialize_candidate(
        store,
        materialization_id=f"materialization-{candidate_id}",
        candidate_id=candidate_id,
        validation=validation,
        review=review,
    )
    return candidate, materialization, manager


def file_guard(workspace: Path, relative: str, content: bytes) -> tuple[str, str]:
    import hashlib

    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return f"guard:file:{relative}", hashlib.sha256(content).hexdigest()
