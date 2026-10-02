"""Immutable human knowledge-review receipts, separate from Runtime approval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .canonicalize import inspect_unsafe_text, normalize_inline
from .index import CandidateComparison, CandidateRelation
from .store import ImmutableWriteStatus, KnowledgeRecordStore
from .types import require_digest, structural_digest
from .validation import CandidateValidationResult, ValidationStatus

REVIEW_SCHEMA = "pico.knowledge-review.v1"
SCHEMA_VERSION = 1


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_EDIT = "request_edit"


class ReviewerType(str, Enum):
    HUMAN = "human"


@dataclass(frozen=True)
class KnowledgeReviewReceipt:
    review_id: str
    candidate_id: str
    candidate_manifest_digest: str
    repository_scope_id: str
    validation_id: str
    validation_digest: str
    comparison_relation: CandidateRelation
    comparison_candidate_ids: tuple[str, ...]
    resolved_conflict_candidate_ids: tuple[str, ...]
    reviewer_type: ReviewerType
    reviewer_id: str
    decision: ReviewDecision
    reason: str
    reviewed_at: str
    review_digest: str
    schema: str = REVIEW_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != REVIEW_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported review schema")
        require_digest(self.candidate_manifest_digest, "candidate_manifest_digest")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.validation_digest, "validation_digest")
        require_digest(self.review_digest, "review_digest")
        if not self.reviewer_id.strip():
            raise ValueError("human reviewer identity is required")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "repository_scope_id": self.repository_scope_id,
            "validation_id": self.validation_id,
            "validation_digest": self.validation_digest,
            "comparison_relation": self.comparison_relation.value,
            "comparison_candidate_ids": list(self.comparison_candidate_ids),
            "resolved_conflict_candidate_ids": list(self.resolved_conflict_candidate_ids),
            "reviewer_type": self.reviewer_type.value,
            "reviewer_id": self.reviewer_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "reviewed_at": self.reviewed_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "review_digest": self.review_digest}

    @classmethod
    def create(cls, **values: Any) -> KnowledgeReviewReceipt:
        base = cls(**values, review_digest="0" * 64)
        return cls(**{**base.__dict__, "review_digest": structural_digest(base._payload())})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeReviewReceipt:
        receipt = cls(
            review_id=str(value["review_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            validation_id=str(value["validation_id"]),
            validation_digest=str(value["validation_digest"]),
            comparison_relation=CandidateRelation(value["comparison_relation"]),
            comparison_candidate_ids=tuple(str(item) for item in value["comparison_candidate_ids"]),
            resolved_conflict_candidate_ids=tuple(
                str(item) for item in value["resolved_conflict_candidate_ids"]
            ),
            reviewer_type=ReviewerType(value["reviewer_type"]),
            reviewer_id=str(value["reviewer_id"]),
            decision=ReviewDecision(value["decision"]),
            reason=str(value["reason"]),
            reviewed_at=str(value["reviewed_at"]),
            review_digest=str(value["review_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if receipt.review_digest != structural_digest(receipt._payload()):
            raise ValueError("review digest mismatch")
        return receipt


def create_review(
    store: KnowledgeRecordStore,
    *,
    review_id: str,
    candidate_id: str,
    validation: CandidateValidationResult,
    comparison: CandidateComparison,
    reviewer_type: ReviewerType,
    reviewer_id: str,
    decision: ReviewDecision,
    reason: str,
    reviewed_at: str,
    resolved_conflict_candidate_ids: tuple[str, ...] = (),
) -> KnowledgeReviewReceipt:
    candidate = store.read_candidate(candidate_id)
    if candidate is None:
        raise ValueError("candidate does not exist")
    from .lifecycle import KnowledgeLifecycleManager, LifecycleState

    if KnowledgeLifecycleManager(store).rebuild(candidate_id).state is not LifecycleState.PENDING_REVIEW:
        raise ValueError("knowledge review requires PENDING_REVIEW state")
    persisted_validation = store.read_validation(validation.validation_id)
    if persisted_validation != validation:
        raise ValueError("validation evidence is missing or mismatched")
    if (
        validation.candidate_id != candidate.candidate_id
        or validation.candidate_manifest_digest != candidate.manifest_digest
        or validation.repository_scope_id != candidate.repository_scope_id
    ):
        raise ValueError("validation does not bind the reviewed candidate")
    if comparison.candidate_id != candidate.candidate_id:
        raise ValueError("comparison does not bind the reviewed candidate")
    reviewer_type = ReviewerType(reviewer_type)
    reviewer_id = normalize_inline(reviewer_id, maximum=256)
    if reviewer_id == candidate.created_by:
        raise ValueError("the generating actor cannot review its own candidate")
    reason = normalize_inline(reason, maximum=1_024)
    inspect_unsafe_text(reason)
    decision = ReviewDecision(decision)
    if decision is ReviewDecision.APPROVE and validation.status is not ValidationStatus.PASS:
        raise ValueError("technical validation must pass before approval")
    resolved = tuple(sorted(set(resolved_conflict_candidate_ids)))
    if comparison.relation is CandidateRelation.CONFLICT:
        unknown = set(resolved) - set(comparison.existing_candidate_ids)
        if unknown:
            raise ValueError("review resolves an unrelated conflict")
    elif resolved:
        raise ValueError("conflict resolutions require a conflict comparison")
    receipt = KnowledgeReviewReceipt.create(
        review_id=review_id,
        candidate_id=candidate.candidate_id,
        candidate_manifest_digest=candidate.manifest_digest,
        repository_scope_id=candidate.repository_scope_id,
        validation_id=validation.validation_id,
        validation_digest=validation.validation_digest,
        comparison_relation=comparison.relation,
        comparison_candidate_ids=comparison.existing_candidate_ids,
        resolved_conflict_candidate_ids=resolved,
        reviewer_type=reviewer_type,
        reviewer_id=reviewer_id,
        decision=decision,
        reason=reason,
        reviewed_at=reviewed_at,
    )
    if store.write_review(receipt) is ImmutableWriteStatus.CONFLICT:
        raise ValueError("immutable review receipt conflict")
    return receipt


__all__ = [
    "KnowledgeReviewReceipt",
    "ReviewDecision",
    "ReviewerType",
    "create_review",
]
