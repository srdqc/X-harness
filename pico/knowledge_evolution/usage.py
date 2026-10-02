"""Immutable Turn-correlated knowledge disposition and outcome evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from pico.tracing import evidence, spans

from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .task_success import TaskSuccessEvidence, TaskSuccessStatus
from .types import CandidateType, require_digest, structural_digest

USAGE_SCHEMA = "pico.knowledge-usage.v1"
OUTCOME_ASSOCIATION_SCHEMA = "pico.knowledge-usage-outcome.v1"
SCHEMA_VERSION = 1


class KnowledgeUsageMode(str, Enum):
    RETRIEVED = "retrieved"
    INJECTED = "injected"
    REFERENCED = "referenced"
    ACTIVATED = "activated"


@dataclass(frozen=True)
class KnowledgeUsageReceipt:
    usage_id: str
    turn_id: str
    repository_scope_id: str
    candidate_id: str
    candidate_manifest_digest: str
    materialization_digest: str
    knowledge_type: CandidateType
    lifecycle_state: str
    applicability_id: str
    applicability_digest: str
    retrieval_id: str
    retrieval_rank: int
    retrieval_score: float
    usage_mode: KnowledgeUsageMode
    context_identity: str | None
    created_at: str
    usage_digest: str
    schema: str = USAGE_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> "KnowledgeUsageReceipt":
        base = cls(**values, usage_digest="0" * 64)
        return cls(**{**base.__dict__, "usage_digest": structural_digest(base._payload())})

    def __post_init__(self) -> None:
        if self.schema != USAGE_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported knowledge usage schema")
        if not self.turn_id.strip():
            raise ValueError("knowledge usage requires Runtime turn_id")
        for value, name in (
            (self.repository_scope_id, "repository_scope_id"),
            (self.candidate_manifest_digest, "candidate_manifest_digest"),
            (self.materialization_digest, "materialization_digest"),
            (self.applicability_digest, "applicability_digest"),
            (self.usage_digest, "usage_digest"),
        ):
            require_digest(value, name)
        if self.retrieval_rank < 1:
            raise ValueError("retrieval_rank must be positive")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "usage_id": self.usage_id,
            "turn_id": self.turn_id,
            "repository_scope_id": self.repository_scope_id,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "materialization_digest": self.materialization_digest,
            "knowledge_type": self.knowledge_type.value,
            "lifecycle_state": self.lifecycle_state,
            "applicability_id": self.applicability_id,
            "applicability_digest": self.applicability_digest,
            "retrieval_id": self.retrieval_id,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "usage_mode": self.usage_mode.value,
            "context_identity": self.context_identity,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "usage_digest": self.usage_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeUsageReceipt":
        receipt = cls(
            usage_id=str(value["usage_id"]),
            turn_id=str(value["turn_id"]),
            repository_scope_id=str(value["repository_scope_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            materialization_digest=str(value["materialization_digest"]),
            knowledge_type=CandidateType(value["knowledge_type"]),
            lifecycle_state=str(value["lifecycle_state"]),
            applicability_id=str(value["applicability_id"]),
            applicability_digest=str(value["applicability_digest"]),
            retrieval_id=str(value["retrieval_id"]),
            retrieval_rank=int(value["retrieval_rank"]),
            retrieval_score=float(value["retrieval_score"]),
            usage_mode=KnowledgeUsageMode(value["usage_mode"]),
            context_identity=(str(value["context_identity"]) if value.get("context_identity") else None),
            created_at=str(value["created_at"]),
            usage_digest=str(value["usage_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if receipt.usage_digest != structural_digest(receipt._payload()):
            raise ValueError("knowledge usage digest mismatch")
        return receipt


@dataclass(frozen=True)
class KnowledgeUsageOutcomeAssociation:
    association_id: str
    turn_id: str
    usage_ids: tuple[str, ...]
    usage_digests: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    task_success_evidence_id: str
    task_success_evidence_digest: str
    task_success_status: TaskSuccessStatus
    association_policy: str
    association_version: int
    created_at: str
    association_digest: str
    schema: str = OUTCOME_ASSOCIATION_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> "KnowledgeUsageOutcomeAssociation":
        base = cls(**values, association_digest="0" * 64)
        return cls(**{**base.__dict__, "association_digest": structural_digest(base._payload())})

    def __post_init__(self) -> None:
        if self.schema != OUTCOME_ASSOCIATION_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported outcome association schema")
        if not self.usage_ids or len(self.usage_ids) != len(self.usage_digests):
            raise ValueError("usage IDs and digests must be non-empty and aligned")
        for digest in (*self.usage_digests, self.task_success_evidence_digest, self.association_digest):
            require_digest(digest, "association digest")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "association_id": self.association_id,
            "turn_id": self.turn_id,
            "usage_ids": list(self.usage_ids),
            "usage_digests": list(self.usage_digests),
            "candidate_ids": list(self.candidate_ids),
            "task_success_evidence_id": self.task_success_evidence_id,
            "task_success_evidence_digest": self.task_success_evidence_digest,
            "task_success_status": self.task_success_status.value,
            "association_policy": self.association_policy,
            "association_version": self.association_version,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "association_digest": self.association_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeUsageOutcomeAssociation":
        receipt = cls(
            association_id=str(value["association_id"]),
            turn_id=str(value["turn_id"]),
            usage_ids=tuple(str(item) for item in value["usage_ids"]),
            usage_digests=tuple(str(item) for item in value["usage_digests"]),
            candidate_ids=tuple(str(item) for item in value["candidate_ids"]),
            task_success_evidence_id=str(value["task_success_evidence_id"]),
            task_success_evidence_digest=str(value["task_success_evidence_digest"]),
            task_success_status=TaskSuccessStatus(value["task_success_status"]),
            association_policy=str(value["association_policy"]),
            association_version=int(value["association_version"]),
            created_at=str(value["created_at"]),
            association_digest=str(value["association_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if receipt.association_digest != structural_digest(receipt._payload()):
            raise ValueError("outcome association digest mismatch")
        return receipt


def persist_usage(store: KnowledgeRecordStore, receipt: KnowledgeUsageReceipt) -> None:
    if store.write_usage(receipt) is ImmutableWriteStatus.CONFLICT:
        raise KnowledgeStoreError("immutable usage receipt conflict")
    recorder = evidence.current()
    if recorder is not None and recorder.turn_id == receipt.turn_id:
        recorder.emit(
            evidence.KNOWLEDGE_USAGE,
            correlations={
                "usage_id": receipt.usage_id,
                "candidate_id": receipt.candidate_id,
                "applicability_id": receipt.applicability_id,
            },
            metadata={
                "receipt_schema": receipt.schema,
                "receipt_digest": receipt.usage_digest,
                "usage_mode": receipt.usage_mode.value,
                "knowledge_type": receipt.knowledge_type.value,
            },
        )


def associate_usage_outcome(
    store: KnowledgeRecordStore,
    *,
    association_id: str,
    usage_ids: tuple[str, ...],
    task_success: TaskSuccessEvidence,
    created_at: str | None = None,
) -> KnowledgeUsageOutcomeAssociation:
    usages = tuple(store.read_usage(item) for item in usage_ids)
    if any(item is None for item in usages):
        raise ValueError("usage receipt does not exist")
    typed = tuple(item for item in usages if item is not None)
    turn_ids = {item.turn_id for item in typed}
    if len(turn_ids) != 1 or task_success.source_turn_id not in turn_ids:
        raise ValueError("task-success Turn does not match usage Turn")
    persisted_success = store.read_task_success(task_success.evidence_id)
    if persisted_success != task_success:
        raise ValueError("task-success evidence is not persisted or mismatched")
    receipt = KnowledgeUsageOutcomeAssociation.create(
        association_id=association_id,
        turn_id=task_success.source_turn_id,
        usage_ids=tuple(item.usage_id for item in typed),
        usage_digests=tuple(item.usage_digest for item in typed),
        candidate_ids=tuple(dict.fromkeys(item.candidate_id for item in typed)),
        task_success_evidence_id=task_success.evidence_id,
        task_success_evidence_digest=task_success.evidence_digest,
        task_success_status=task_success.status,
        association_policy="p3.independent-task-success-association",
        association_version=1,
        created_at=created_at or spans.now_iso(),
    )
    if store.write_outcome_association(receipt) is ImmutableWriteStatus.CONFLICT:
        raise KnowledgeStoreError("immutable outcome association conflict")
    return receipt


__all__ = [
    "KnowledgeUsageMode",
    "KnowledgeUsageOutcomeAssociation",
    "KnowledgeUsageReceipt",
    "associate_usage_outcome",
    "persist_usage",
]
