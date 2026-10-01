"""Independent task-success evidence consumed by P3 eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .types import require_digest, structural_digest

TASK_SUCCESS_SCHEMA = "pico.task-success-evidence.v1"
SCHEMA_VERSION = 1


class TaskSuccessSource(str, Enum):
    SEALED_VERIFIER = "sealed_verifier"
    WORKSPACE_TEST = "workspace_test"
    EXPECTED_ARTIFACT = "expected_artifact"
    HUMAN_VALIDATION = "human_validation"
    FROZEN_BENCHMARK = "frozen_benchmark"


class TaskSuccessStatus(str, Enum):
    PASS = "pass"  # noqa: S105 -- verification status, not a credential
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class TaskSuccessEvidence:
    evidence_id: str
    source_kind: TaskSuccessSource
    source_turn_id: str
    status: TaskSuccessStatus
    producer_id: str
    producer_version: str
    repository_scope_id: str
    target_state_digest: str | None
    result_digest: str
    provenance_refs: tuple[str, ...]
    created_at: str
    human_actor_id: str | None = None
    comment: str | None = None
    evidence_digest: str = ""
    schema: str = TASK_SUCCESS_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **kwargs: Any) -> "TaskSuccessEvidence":
        base = cls(**kwargs, evidence_digest="0" * 64)
        return cls(**{**base.__dict__, "evidence_digest": structural_digest(base._digest_payload())})

    def __post_init__(self) -> None:
        if self.schema != TASK_SUCCESS_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported task-success evidence schema")
        for name in (
            "evidence_id",
            "source_turn_id",
            "producer_id",
            "producer_version",
            "created_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
                raise ValueError(f"{name} must be a bounded non-empty string")
        object.__setattr__(self, "source_kind", TaskSuccessSource(self.source_kind))
        object.__setattr__(self, "status", TaskSuccessStatus(self.status))
        require_digest(self.repository_scope_id, "repository_scope_id")
        if self.target_state_digest is not None:
            require_digest(self.target_state_digest, "target_state_digest")
        require_digest(self.result_digest, "result_digest")
        require_digest(self.evidence_digest, "evidence_digest")
        if len(self.provenance_refs) != len(set(self.provenance_refs)):
            raise ValueError("provenance_refs must be unique")
        if self.source_kind is TaskSuccessSource.HUMAN_VALIDATION:
            if not self.human_actor_id or len(self.human_actor_id) > 256:
                raise ValueError("human validation requires human_actor_id")
        elif self.human_actor_id is not None:
            raise ValueError("human_actor_id is only valid for human validation")
        if self.comment is not None and len(self.comment) > 1024:
            raise ValueError("comment exceeds 1024 characters")

    @property
    def has_state_binding(self) -> bool:
        return self.target_state_digest is not None

    def _digest_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_digest")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "source_kind": self.source_kind.value,
            "source_turn_id": self.source_turn_id,
            "status": self.status.value,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "repository_scope_id": self.repository_scope_id,
            "target_state_digest": self.target_state_digest,
            "result_digest": self.result_digest,
            "provenance_refs": list(self.provenance_refs),
            "created_at": self.created_at,
            "human_actor_id": self.human_actor_id,
            "comment": self.comment,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSuccessEvidence":
        evidence = cls(
            evidence_id=str(value["evidence_id"]),
            source_kind=TaskSuccessSource(value["source_kind"]),
            source_turn_id=str(value["source_turn_id"]),
            status=TaskSuccessStatus(value["status"]),
            producer_id=str(value["producer_id"]),
            producer_version=str(value["producer_version"]),
            repository_scope_id=str(value["repository_scope_id"]),
            target_state_digest=(
                str(value["target_state_digest"])
                if value.get("target_state_digest") is not None
                else None
            ),
            result_digest=str(value["result_digest"]),
            provenance_refs=tuple(str(item) for item in value["provenance_refs"]),
            created_at=str(value["created_at"]),
            human_actor_id=(str(value["human_actor_id"]) if value.get("human_actor_id") else None),
            comment=(str(value["comment"]) if value.get("comment") is not None else None),
            evidence_digest=str(value["evidence_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if evidence.evidence_digest != structural_digest(evidence._digest_payload()):
            raise ValueError("task-success evidence digest mismatch")
        return evidence


__all__ = [
    "SCHEMA_VERSION",
    "TASK_SUCCESS_SCHEMA",
    "TaskSuccessEvidence",
    "TaskSuccessSource",
    "TaskSuccessStatus",
]
