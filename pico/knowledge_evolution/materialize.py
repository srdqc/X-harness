"""Controlled repository-scoped materialization of reviewed ACTIVE knowledge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .lifecycle import KnowledgeLifecycleManager, LifecycleState
from .review import KnowledgeReviewReceipt, ReviewDecision
from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .types import CandidateType, canonical_json, require_digest, structural_digest
from .validation import CandidateValidationResult, ValidationStatus

MATERIALIZATION_SCHEMA = "pico.knowledge-materialization-result.v1"
SCHEMA_VERSION = 1


class MaterializationStatus(str, Enum):
    CREATED = "created"
    UNCHANGED = "unchanged"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class KnowledgeMaterializationResult:
    materialization_id: str
    candidate_id: str
    candidate_manifest_digest: str
    repository_scope_id: str
    lifecycle_state: LifecycleState
    lifecycle_transition_digest: str | None
    validation_digest: str
    review_digest: str
    status: MaterializationStatus
    artifact_kind: str
    artifact_relative_path: str | None
    content_digest: str | None
    provenance_refs: tuple[str, ...]
    failure_reason: str | None
    result_digest: str
    schema: str = MATERIALIZATION_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != MATERIALIZATION_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported materialization schema")
        require_digest(self.candidate_manifest_digest, "candidate_manifest_digest")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.validation_digest, "validation_digest")
        require_digest(self.review_digest, "review_digest")
        require_digest(self.result_digest, "result_digest")
        if self.lifecycle_transition_digest is not None:
            require_digest(self.lifecycle_transition_digest, "lifecycle_transition_digest")
        if self.content_digest is not None:
            require_digest(self.content_digest, "content_digest")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "materialization_id": self.materialization_id,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "repository_scope_id": self.repository_scope_id,
            "lifecycle_state": self.lifecycle_state.value,
            "lifecycle_transition_digest": self.lifecycle_transition_digest,
            "validation_digest": self.validation_digest,
            "review_digest": self.review_digest,
            "status": self.status.value,
            "artifact_kind": self.artifact_kind,
            "artifact_relative_path": self.artifact_relative_path,
            "content_digest": self.content_digest,
            "provenance_refs": list(self.provenance_refs),
            "failure_reason": self.failure_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "result_digest": self.result_digest}

    @classmethod
    def create(cls, **values: Any) -> KnowledgeMaterializationResult:
        base = cls(**values, result_digest="0" * 64)
        return cls(**{**base.__dict__, "result_digest": structural_digest(base._payload())})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeMaterializationResult:
        result = cls(
            materialization_id=str(value["materialization_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            lifecycle_state=LifecycleState(value["lifecycle_state"]),
            lifecycle_transition_digest=(
                str(value["lifecycle_transition_digest"])
                if value.get("lifecycle_transition_digest") is not None
                else None
            ),
            validation_digest=str(value["validation_digest"]),
            review_digest=str(value["review_digest"]),
            status=MaterializationStatus(value["status"]),
            artifact_kind=str(value["artifact_kind"]),
            artifact_relative_path=(
                str(value["artifact_relative_path"])
                if value.get("artifact_relative_path") is not None
                else None
            ),
            content_digest=(
                str(value["content_digest"]) if value.get("content_digest") is not None else None
            ),
            provenance_refs=tuple(str(item) for item in value["provenance_refs"]),
            failure_reason=(
                str(value["failure_reason"]) if value.get("failure_reason") is not None else None
            ),
            result_digest=str(value["result_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if result.result_digest != structural_digest(result._payload()):
            raise ValueError("materialization result digest mismatch")
        return result


def materialized_skill_root(store: KnowledgeRecordStore, repository_scope_id: str) -> Path:
    return store.materialized / repository_scope_id / "skills"


def _artifact(candidate: Any, validation: Any, review: Any, lifecycle_digest: str) -> tuple[str, tuple[str, ...], bytes]:
    common = {
        "schema": "pico.materialized-knowledge.v1",
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_digest": candidate.manifest_digest,
        "repository_scope_id": candidate.repository_scope_id,
        "validation_digest": validation.validation_digest,
        "review_digest": review.review_digest,
        "lifecycle_transition_digest": lifecycle_digest,
        "content_fingerprint": candidate.content_fingerprint,
    }
    if candidate.candidate_type is CandidateType.SKILL_CANDIDATE:
        metadata = canonical_json({"p3": common})
        body = (
            "---\n"
            f"name: {candidate.candidate_id}\n"
            f"description: {candidate.title}\n"
            "always: false\n"
            f"metadata: {metadata}\n"
            "---\n\n"
            f"# {candidate.title}\n\n"
            f"{candidate.reusable_content}\n\n"
            "## Preconditions\n\n"
            + "\n".join(f"- {item}" for item in candidate.preconditions)
            + "\n"
        )
        return "skill", (
            candidate.repository_scope_id,
            "skills",
            candidate.candidate_id,
            "SKILL.md",
        ), body.encode("utf-8")
    payload = {
        **common,
        "candidate_type": candidate.candidate_type.value,
        "content_class": candidate.content_class.value,
        "title": candidate.title,
        "reusable_content": candidate.reusable_content,
        "preconditions": list(candidate.preconditions),
        "applicability_fingerprints": [list(item) for item in candidate.applicability_fingerprints],
        "provenance_refs": list(candidate.provenance_refs),
    }
    kind = "fact" if candidate.candidate_type is CandidateType.MEMORY_FACT else "guidance"
    directory = "facts" if kind == "fact" else "guidance"
    return kind, (
        candidate.repository_scope_id,
        directory,
        f"{candidate.candidate_id}.json",
    ), (canonical_json(payload) + "\n").encode("utf-8")


def materialize_candidate(
    store: KnowledgeRecordStore,
    *,
    materialization_id: str,
    candidate_id: str,
    validation: CandidateValidationResult,
    review: KnowledgeReviewReceipt,
) -> KnowledgeMaterializationResult:
    candidate = store.read_candidate(candidate_id)
    if candidate is None:
        raise ValueError("candidate does not exist")
    projection = KnowledgeLifecycleManager(store).rebuild(candidate_id)
    persisted_validation = store.read_validation(validation.validation_id)
    persisted_review = store.read_review(review.review_id)
    bindings_valid = (
        persisted_validation == validation
        and persisted_review == review
        and validation.status is ValidationStatus.PASS
        and review.decision is ReviewDecision.APPROVE
        and validation.candidate_manifest_digest == candidate.manifest_digest
        and review.candidate_manifest_digest == candidate.manifest_digest
        and review.validation_digest == validation.validation_digest
        and projection.head_transition_digest is not None
        and f"validation:{validation.validation_digest}" in projection.transitions[-1].evidence_refs
        and f"review:{review.review_digest}" in projection.transitions[-1].evidence_refs
    )
    artifact_kind = {
        CandidateType.SKILL_CANDIDATE: "skill",
        CandidateType.MEMORY_FACT: "fact",
        CandidateType.EXPERIENCE: "guidance",
    }[candidate.candidate_type]
    if projection.state is not LifecycleState.ACTIVE or not bindings_valid:
        result = KnowledgeMaterializationResult.create(
            materialization_id=materialization_id,
            candidate_id=candidate.candidate_id,
            candidate_manifest_digest=candidate.manifest_digest,
            repository_scope_id=candidate.repository_scope_id,
            lifecycle_state=projection.state,
            lifecycle_transition_digest=projection.head_transition_digest,
            validation_digest=validation.validation_digest,
            review_digest=review.review_digest,
            status=MaterializationStatus.NOT_APPLICABLE,
            artifact_kind=artifact_kind,
            artifact_relative_path=None,
            content_digest=None,
            provenance_refs=tuple(sorted(candidate.provenance_refs)),
            failure_reason="candidate_is_not_reviewed_active_knowledge",
        )
        if store.write_materialization_result(result) is ImmutableWriteStatus.CONFLICT:
            raise KnowledgeStoreError("immutable materialization result conflict")
        return result

    kind, parts, content = _artifact(
        candidate, validation, review, projection.head_transition_digest
    )
    content_digest = hashlib.sha256(content).hexdigest()
    write_status, path = store.write_materialized_bytes(parts, content)
    status = {
        ImmutableWriteStatus.CREATED: MaterializationStatus.CREATED,
        ImmutableWriteStatus.UNCHANGED: MaterializationStatus.UNCHANGED,
        ImmutableWriteStatus.CONFLICT: MaterializationStatus.FAILED,
    }[write_status]
    relative_path = path.relative_to(store.materialized).as_posix()
    result = KnowledgeMaterializationResult.create(
        materialization_id=materialization_id,
        candidate_id=candidate.candidate_id,
        candidate_manifest_digest=candidate.manifest_digest,
        repository_scope_id=candidate.repository_scope_id,
        lifecycle_state=projection.state,
        lifecycle_transition_digest=projection.head_transition_digest,
        validation_digest=validation.validation_digest,
        review_digest=review.review_digest,
        status=status,
        artifact_kind=kind,
        artifact_relative_path=relative_path,
        content_digest=content_digest,
        provenance_refs=tuple(sorted(candidate.provenance_refs)),
        failure_reason=("materialized_artifact_conflict" if status is MaterializationStatus.FAILED else None),
    )
    result_status = store.write_materialization_result(result)
    if result_status is ImmutableWriteStatus.CONFLICT:
        raise KnowledgeStoreError("immutable materialization result conflict")
    return result


__all__ = [
    "KnowledgeMaterializationResult",
    "MaterializationStatus",
    "materialize_candidate",
    "materialized_skill_root",
]
