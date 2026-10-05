"""Versioned candidate identity used by deterministic evaluation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .lifecycle import LifecycleState
from .types import KnowledgeCandidate, require_digest, structural_digest

CANDIDATE_EVIDENCE_IDENTITY_SCHEMA = "pico.knowledge-candidate-identity.v1"
CANDIDATE_EVIDENCE_IDENTITY_VERSION = 1


def _digest_strings(values: Sequence[str]) -> str:
    return structural_digest(list(values))


@dataclass(frozen=True)
class CandidateEvidenceIdentity:
    """Separate durable ID, semantic identity, and optional display label."""

    candidate_id: str
    semantic_digest: str
    payload_digest: str
    source_identity_digest: str
    repository_scope_id: str
    candidate_type: str
    content_class: str
    title_digest: str
    reusable_content_digest: str
    preconditions_digest: str
    applicability_fingerprints_digest: str
    lifecycle_state: str
    portable_label: str | None = None
    schema: str = CANDIDATE_EVIDENCE_IDENTITY_SCHEMA
    schema_version: int = CANDIDATE_EVIDENCE_IDENTITY_VERSION

    @classmethod
    def from_candidate(
        cls,
        candidate: KnowledgeCandidate,
        *,
        lifecycle_state: LifecycleState | str,
        portable_label: str | None = None,
    ) -> "CandidateEvidenceIdentity":
        state = LifecycleState(lifecycle_state).value
        title_digest = structural_digest(candidate.title)
        content_digest = structural_digest(candidate.reusable_content)
        preconditions_digest = _digest_strings(candidate.preconditions)
        applicability_digest = structural_digest(
            [list(item) for item in candidate.applicability_fingerprints]
        )
        payload_digest = structural_digest(
            {
                "candidate_type": candidate.candidate_type.value,
                "content_class": candidate.content_class.value,
                "repository_scope_id": candidate.repository_scope_id,
                "title": candidate.title,
                "reusable_content": candidate.reusable_content,
                "preconditions": list(candidate.preconditions),
                "applicability_fingerprints": [
                    list(item) for item in candidate.applicability_fingerprints
                ],
            }
        )
        source_identity_digest = structural_digest(
            {
                "qualifying_sources": [
                    source.to_dict() for source in candidate.qualifying_sources
                ],
                "supporting_source_turn_ids": list(
                    candidate.supporting_source_turn_ids
                ),
                "extraction_policy": candidate.extraction_policy,
                "extraction_policy_version": candidate.extraction_policy_version,
                "provenance_refs": list(candidate.provenance_refs),
                "created_by": candidate.created_by,
            }
        )
        values = {
            "candidate_id": candidate.candidate_id,
            "payload_digest": payload_digest,
            "source_identity_digest": source_identity_digest,
            "repository_scope_id": candidate.repository_scope_id,
            "candidate_type": candidate.candidate_type.value,
            "content_class": candidate.content_class.value,
            "title_digest": title_digest,
            "reusable_content_digest": content_digest,
            "preconditions_digest": preconditions_digest,
            "applicability_fingerprints_digest": applicability_digest,
            "lifecycle_state": state,
        }
        return cls(
            **values,
            semantic_digest=structural_digest(cls._semantic_payload(values)),
            portable_label=portable_label,
        )

    @staticmethod
    def _semantic_payload(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_EVIDENCE_IDENTITY_SCHEMA,
            "schema_version": CANDIDATE_EVIDENCE_IDENTITY_VERSION,
            **{
                name: values[name]
                for name in (
                    "candidate_id",
                    "payload_digest",
                    "source_identity_digest",
                    "repository_scope_id",
                    "candidate_type",
                    "content_class",
                    "title_digest",
                    "reusable_content_digest",
                    "preconditions_digest",
                    "applicability_fingerprints_digest",
                    "lifecycle_state",
                )
            },
        }

    def __post_init__(self) -> None:
        if (
            self.schema != CANDIDATE_EVIDENCE_IDENTITY_SCHEMA
            or self.schema_version != CANDIDATE_EVIDENCE_IDENTITY_VERSION
        ):
            raise ValueError("unsupported candidate evidence identity schema")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        for name in (
            "semantic_digest",
            "payload_digest",
            "source_identity_digest",
            "repository_scope_id",
            "title_digest",
            "reusable_content_digest",
            "preconditions_digest",
            "applicability_fingerprints_digest",
        ):
            require_digest(str(getattr(self, name)), name)
        LifecycleState(self.lifecycle_state)
        expected = structural_digest(self._semantic_payload(self.__dict__))
        if self.semantic_digest != expected:
            raise ValueError("candidate semantic identity digest mismatch")
        if self.portable_label is not None and not self.portable_label:
            raise ValueError("portable_label must be non-empty when present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "semantic_digest": self.semantic_digest,
            "payload_digest": self.payload_digest,
            "source_identity_digest": self.source_identity_digest,
            "repository_scope_id": self.repository_scope_id,
            "candidate_type": self.candidate_type,
            "content_class": self.content_class,
            "title_digest": self.title_digest,
            "reusable_content_digest": self.reusable_content_digest,
            "preconditions_digest": self.preconditions_digest,
            "applicability_fingerprints_digest": self.applicability_fingerprints_digest,
            "lifecycle_state": self.lifecycle_state,
            "portable_label": self.portable_label,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEvidenceIdentity":
        return cls(
            candidate_id=str(value["candidate_id"]),
            semantic_digest=str(value["semantic_digest"]),
            payload_digest=str(value["payload_digest"]),
            source_identity_digest=str(value["source_identity_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            candidate_type=str(value["candidate_type"]),
            content_class=str(value["content_class"]),
            title_digest=str(value["title_digest"]),
            reusable_content_digest=str(value["reusable_content_digest"]),
            preconditions_digest=str(value["preconditions_digest"]),
            applicability_fingerprints_digest=str(
                value["applicability_fingerprints_digest"]
            ),
            lifecycle_state=str(value["lifecycle_state"]),
            portable_label=(
                str(value["portable_label"])
                if value.get("portable_label") is not None
                else None
            ),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )


def ordered_candidate_evidence_equal(
    expected: Sequence[CandidateEvidenceIdentity],
    actual: Sequence[CandidateEvidenceIdentity],
) -> bool:
    """Fail closed on ambiguity and compare ordered durable+semantic identities."""

    for values in (expected, actual):
        durable = tuple(item.candidate_id for item in values)
        semantic = tuple(item.semantic_digest for item in values)
        if len(set(durable)) != len(durable) or len(set(semantic)) != len(semantic):
            raise ValueError("ambiguous candidate evidence identity")
    return tuple(
        (item.candidate_id, item.semantic_digest) for item in expected
    ) == tuple((item.candidate_id, item.semantic_digest) for item in actual)


__all__ = [
    "CANDIDATE_EVIDENCE_IDENTITY_SCHEMA",
    "CANDIDATE_EVIDENCE_IDENTITY_VERSION",
    "CandidateEvidenceIdentity",
    "ordered_candidate_evidence_equal",
]
