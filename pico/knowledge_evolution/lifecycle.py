"""Append-only lifecycle state reconstructed from digest-chained transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .canonicalize import inspect_unsafe_text, normalize_inline
from .index import CandidateRelation
from .review import KnowledgeReviewReceipt, ReviewDecision, ReviewerType
from .store import ImmutableWriteStatus, KnowledgeRecordStore
from .types import require_digest, structural_digest
from .validation import CandidateValidationResult, ValidationStatus

LIFECYCLE_SCHEMA = "pico.knowledge-lifecycle-transition.v1"
SCHEMA_VERSION = 1


class LifecycleState(str, Enum):
    EXTRACTED = "extracted"
    ELIGIBLE = "eligible"
    INCONCLUSIVE = "inconclusive"
    VALIDATED = "validated"
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class LifecycleActorType(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"


class LifecycleReason(str, Enum):
    ELIGIBILITY_CONFIRMED = "eligibility_confirmed"
    EVIDENCE_INCONCLUSIVE = "evidence_inconclusive"
    TECHNICAL_VALIDATION_PASSED = "technical_validation_passed"
    TECHNICAL_VALIDATION_FAILED = "technical_validation_failed"
    SUBMITTED_FOR_REVIEW = "submitted_for_review"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_DEPRECATED = "human_deprecated"
    HUMAN_SUPERSEDED = "human_superseded"


LEGAL_TRANSITIONS = {
    LifecycleState.EXTRACTED: {
        LifecycleState.ELIGIBLE,
        LifecycleState.INCONCLUSIVE,
        LifecycleState.REJECTED,
    },
    LifecycleState.INCONCLUSIVE: {LifecycleState.ELIGIBLE, LifecycleState.REJECTED},
    LifecycleState.ELIGIBLE: {
        LifecycleState.VALIDATED,
        LifecycleState.INCONCLUSIVE,
        LifecycleState.REJECTED,
    },
    LifecycleState.VALIDATED: {
        LifecycleState.PENDING_REVIEW,
        LifecycleState.INCONCLUSIVE,
        LifecycleState.REJECTED,
    },
    LifecycleState.PENDING_REVIEW: {
        LifecycleState.ACTIVE,
        LifecycleState.REJECTED,
        LifecycleState.SUPERSEDED,
    },
    LifecycleState.ACTIVE: {LifecycleState.DEPRECATED, LifecycleState.SUPERSEDED},
}


@dataclass(frozen=True)
class KnowledgeLifecycleTransition:
    transition_id: str
    sequence: int
    candidate_id: str
    candidate_manifest_digest: str
    repository_scope_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    actor_type: LifecycleActorType
    actor_id: str
    reason: LifecycleReason
    comment: str | None
    evidence_refs: tuple[str, ...]
    related_candidate_id: str | None
    previous_transition_digest: str | None
    transitioned_at: str
    transition_digest: str
    schema: str = LIFECYCLE_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != LIFECYCLE_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle schema")
        if self.sequence < 1:
            raise ValueError("lifecycle sequence must be positive")
        require_digest(self.candidate_manifest_digest, "candidate_manifest_digest")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.transition_digest, "transition_digest")
        if self.previous_transition_digest is not None:
            require_digest(self.previous_transition_digest, "previous_transition_digest")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "sequence": self.sequence,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "repository_scope_id": self.repository_scope_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "reason": self.reason.value,
            "comment": self.comment,
            "evidence_refs": list(self.evidence_refs),
            "related_candidate_id": self.related_candidate_id,
            "previous_transition_digest": self.previous_transition_digest,
            "transitioned_at": self.transitioned_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "transition_digest": self.transition_digest}

    @classmethod
    def create(cls, **values: Any) -> KnowledgeLifecycleTransition:
        base = cls(**values, transition_digest="0" * 64)
        return cls(**{**base.__dict__, "transition_digest": structural_digest(base._payload())})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeLifecycleTransition:
        transition = cls(
            transition_id=str(value["transition_id"]),
            sequence=int(value["sequence"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            from_state=LifecycleState(value["from_state"]),
            to_state=LifecycleState(value["to_state"]),
            actor_type=LifecycleActorType(value["actor_type"]),
            actor_id=str(value["actor_id"]),
            reason=LifecycleReason(value["reason"]),
            comment=(str(value["comment"]) if value.get("comment") is not None else None),
            evidence_refs=tuple(str(item) for item in value["evidence_refs"]),
            related_candidate_id=(
                str(value["related_candidate_id"])
                if value.get("related_candidate_id") is not None
                else None
            ),
            previous_transition_digest=(
                str(value["previous_transition_digest"])
                if value.get("previous_transition_digest") is not None
                else None
            ),
            transitioned_at=str(value["transitioned_at"]),
            transition_digest=str(value["transition_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if transition.transition_digest != structural_digest(transition._payload()):
            raise ValueError("lifecycle transition digest mismatch")
        return transition


@dataclass(frozen=True)
class LifecycleProjection:
    candidate_id: str
    repository_scope_id: str
    state: LifecycleState
    head_transition_digest: str | None
    transitions: tuple[KnowledgeLifecycleTransition, ...]

    @property
    def is_active(self) -> bool:
        return self.state is LifecycleState.ACTIVE


class KnowledgeLifecycleManager:
    def __init__(self, store: KnowledgeRecordStore) -> None:
        self.store = store

    def rebuild(self, candidate_id: str) -> LifecycleProjection:
        candidate = self.store.read_candidate(candidate_id)
        if candidate is None:
            raise ValueError("candidate does not exist")
        transitions = self.store.read_lifecycle(candidate.repository_scope_id, candidate_id)
        state = LifecycleState.EXTRACTED
        previous: str | None = None
        for sequence, item in enumerate(transitions, start=1):
            if (
                item.sequence != sequence
                or item.candidate_id != candidate.candidate_id
                or item.candidate_manifest_digest != candidate.manifest_digest
                or item.repository_scope_id != candidate.repository_scope_id
                or item.previous_transition_digest != previous
                or item.from_state is not state
                or item.to_state not in LEGAL_TRANSITIONS.get(state, set())
            ):
                raise ValueError("invalid lifecycle chain")
            if item.to_state is LifecycleState.ACTIVE and item.actor_type is not LifecycleActorType.HUMAN:
                raise ValueError("ACTIVE transition lacks human authority")
            state = item.to_state
            previous = item.transition_digest
        return LifecycleProjection(
            candidate.candidate_id,
            candidate.repository_scope_id,
            state,
            previous,
            transitions,
        )

    def transition(
        self,
        *,
        transition_id: str,
        candidate_id: str,
        from_state: LifecycleState,
        to_state: LifecycleState,
        actor_type: LifecycleActorType,
        actor_id: str,
        reason: LifecycleReason,
        transitioned_at: str,
        evidence_refs: tuple[str, ...] = (),
        comment: str | None = None,
        validation: CandidateValidationResult | None = None,
        review: KnowledgeReviewReceipt | None = None,
        related_candidate_id: str | None = None,
    ) -> KnowledgeLifecycleTransition:
        candidate = self.store.read_candidate(candidate_id)
        if candidate is None:
            raise ValueError("candidate does not exist")
        projection = self.rebuild(candidate_id)
        from_state = LifecycleState(from_state)
        to_state = LifecycleState(to_state)
        if projection.state is not from_state:
            raise ValueError("lifecycle from_state does not match current state")
        if to_state not in LEGAL_TRANSITIONS.get(from_state, set()):
            raise ValueError("illegal lifecycle transition")
        actor_type = LifecycleActorType(actor_type)
        actor_id = normalize_inline(actor_id, maximum=256)
        reason = LifecycleReason(reason)
        if comment is not None:
            comment = normalize_inline(comment, maximum=1_024)
            inspect_unsafe_text(comment)

        refs = set(evidence_refs)
        if to_state is LifecycleState.ELIGIBLE and not any(
            item.startswith("eligibility:") for item in refs
        ):
            raise ValueError("ELIGIBLE transition requires eligibility evidence")
        if validation is not None:
            persisted_validation = self.store.read_validation(validation.validation_id)
            if persisted_validation != validation or (
                validation.candidate_id != candidate.candidate_id
                or validation.candidate_manifest_digest != candidate.manifest_digest
                or validation.repository_scope_id != candidate.repository_scope_id
            ):
                raise ValueError("validation evidence does not bind candidate")
            refs.add(f"validation:{validation.validation_digest}")
        if to_state in {LifecycleState.VALIDATED, LifecycleState.PENDING_REVIEW, LifecycleState.ACTIVE}:
            if validation is None or validation.status is not ValidationStatus.PASS:
                raise ValueError("transition requires passing technical validation")

        if review is not None:
            persisted_review = self.store.read_review(review.review_id)
            if persisted_review != review or (
                review.candidate_id != candidate.candidate_id
                or review.candidate_manifest_digest != candidate.manifest_digest
                or review.repository_scope_id != candidate.repository_scope_id
            ):
                raise ValueError("review evidence does not bind candidate")
            refs.add(f"review:{review.review_digest}")

        if to_state is LifecycleState.ACTIVE:
            if (
                actor_type is not LifecycleActorType.HUMAN
                or review is None
                or review.reviewer_type is not ReviewerType.HUMAN
                or review.reviewer_id != actor_id
                or review.decision is not ReviewDecision.APPROVE
                or review.validation_digest != validation.validation_digest
            ):
                raise ValueError("ACTIVE requires matching human approval")
            if review.comparison_relation is CandidateRelation.EXACT_DUPLICATE:
                raise ValueError("exact duplicate cannot become a second ACTIVE candidate")
            if review.comparison_relation is CandidateRelation.CONFLICT and set(
                review.resolved_conflict_candidate_ids
            ) != set(review.comparison_candidate_ids):
                raise ValueError("blocking conflict remains unresolved")

        if to_state is LifecycleState.REJECTED and from_state is LifecycleState.PENDING_REVIEW:
            if (
                actor_type is not LifecycleActorType.HUMAN
                or review is None
                or review.reviewer_id != actor_id
                or review.decision is not ReviewDecision.REJECT
            ):
                raise ValueError("review-stage rejection requires human REJECT receipt")
        if to_state in {LifecycleState.DEPRECATED, LifecycleState.SUPERSEDED}:
            if actor_type is not LifecycleActorType.HUMAN:
                raise ValueError("terminal knowledge governance transition requires human actor")
        if to_state is LifecycleState.SUPERSEDED:
            if related_candidate_id is None:
                raise ValueError("supersession requires replacement candidate")
            replacement = self.store.read_candidate(related_candidate_id)
            if (
                replacement is None
                or replacement.candidate_id == candidate.candidate_id
                or replacement.repository_scope_id != candidate.repository_scope_id
                or review is None
                or review.decision is not ReviewDecision.APPROVE
            ):
                raise ValueError("supersession requires reviewed same-scope replacement")
            refs.add(f"replacement:{replacement.manifest_digest}")
        elif related_candidate_id is not None:
            raise ValueError("related candidate is only valid for supersession")

        transition = KnowledgeLifecycleTransition.create(
            transition_id=transition_id,
            sequence=len(projection.transitions) + 1,
            candidate_id=candidate.candidate_id,
            candidate_manifest_digest=candidate.manifest_digest,
            repository_scope_id=candidate.repository_scope_id,
            from_state=from_state,
            to_state=to_state,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
            comment=comment,
            evidence_refs=tuple(sorted(refs)),
            related_candidate_id=related_candidate_id,
            previous_transition_digest=projection.head_transition_digest,
            transitioned_at=transitioned_at,
        )
        status = self.store.append_lifecycle_transition(transition)
        if status is ImmutableWriteStatus.CONFLICT:
            raise ValueError("lifecycle head changed during append")
        return transition


__all__ = [
    "LEGAL_TRANSITIONS",
    "KnowledgeLifecycleManager",
    "KnowledgeLifecycleTransition",
    "LifecycleActorType",
    "LifecycleProjection",
    "LifecycleReason",
    "LifecycleState",
]
