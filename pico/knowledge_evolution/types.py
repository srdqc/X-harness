"""Immutable P3 knowledge-candidate contracts and canonical serialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

KNOWLEDGE_CANDIDATE_SCHEMA = "pico.knowledge-candidate.v1"
SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def structural_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_digest(value: str, field_name: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _text(value: str, field_name: str, *, maximum: int, multiline: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if multiline:
        normalized = "\n".join(line.rstrip() for line in normalized.strip().splitlines())
    else:
        normalized = " ".join(normalized.split())
    if not normalized or len(normalized) > maximum or "\x00" in normalized:
        raise ValueError(f"{field_name} must contain 1-{maximum} safe characters")
    return normalized


def _strings(values: tuple[str, ...], field_name: str, *, maximum: int = 512) -> tuple[str, ...]:
    normalized = tuple(_text(value, field_name, maximum=maximum) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


class CandidateType(str, Enum):
    MEMORY_FACT = "memory_fact"
    EXPERIENCE = "experience"
    SKILL_CANDIDATE = "skill_candidate"


class ContentClass(str, Enum):
    FACT = "fact"
    STRATEGY = "strategy"
    RECOVERY = "recovery"
    WARNING = "warning"
    ANTI_PATTERN = "anti_pattern"
    PROCEDURE = "procedure"


@dataclass(frozen=True)
class SourceTurnReference:
    turn_id: str
    replay_digest: str
    verification_digest: str
    task_success_evidence_ids: tuple[str, ...]
    task_success_evidence_digests: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _text(self.turn_id, "turn_id", maximum=256))
        require_digest(self.replay_digest, "replay_digest")
        require_digest(self.verification_digest, "verification_digest")
        ids = _strings(self.task_success_evidence_ids, "task_success_evidence_ids")
        digests = tuple(self.task_success_evidence_digests)
        if len(ids) != len(digests) or not ids:
            raise ValueError("task success evidence IDs and digests must be non-empty and aligned")
        for digest in digests:
            require_digest(digest, "task_success_evidence_digest")
        object.__setattr__(self, "task_success_evidence_ids", ids)
        object.__setattr__(self, "task_success_evidence_digests", digests)
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs, "evidence_refs"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "replay_digest": self.replay_digest,
            "verification_digest": self.verification_digest,
            "task_success_evidence_ids": list(self.task_success_evidence_ids),
            "task_success_evidence_digests": list(self.task_success_evidence_digests),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceTurnReference":
        return cls(
            turn_id=str(value["turn_id"]),
            replay_digest=str(value["replay_digest"]),
            verification_digest=str(value["verification_digest"]),
            task_success_evidence_ids=tuple(str(item) for item in value["task_success_evidence_ids"]),
            task_success_evidence_digests=tuple(
                str(item) for item in value["task_success_evidence_digests"]
            ),
            evidence_refs=tuple(str(item) for item in value.get("evidence_refs", ())),
        )


@dataclass(frozen=True)
class KnowledgeCandidate:
    candidate_id: str
    candidate_type: CandidateType
    content_class: ContentClass
    qualifying_sources: tuple[SourceTurnReference, ...]
    supporting_source_turn_ids: tuple[str, ...]
    repository_scope_id: str
    title: str
    reusable_content: str
    preconditions: tuple[str, ...]
    applicability_fingerprints: tuple[tuple[str, str], ...]
    extraction_policy: str
    extraction_policy_version: int
    provenance_refs: tuple[str, ...]
    created_at: str
    created_by: str
    content_fingerprint: str
    manifest_digest: str
    schema: str = KNOWLEDGE_CANDIDATE_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        candidate_type: CandidateType,
        content_class: ContentClass,
        qualifying_sources: tuple[SourceTurnReference, ...],
        supporting_source_turn_ids: tuple[str, ...] = (),
        repository_scope_id: str,
        title: str,
        reusable_content: str,
        preconditions: tuple[str, ...] = (),
        applicability_fingerprints: tuple[tuple[str, str], ...] = (),
        extraction_policy: str,
        extraction_policy_version: int,
        provenance_refs: tuple[str, ...],
        created_at: str,
        created_by: str,
    ) -> "KnowledgeCandidate":
        normalized_title = _text(title, "title", maximum=256)
        normalized_content = _text(
            reusable_content, "reusable_content", maximum=32_768, multiline=True
        )
        normalized_preconditions = _strings(preconditions, "preconditions", maximum=1024)
        content_fingerprint = structural_digest(
            {
                "candidate_type": CandidateType(candidate_type).value,
                "content_class": ContentClass(content_class).value,
                "repository_scope_id": repository_scope_id,
                "title": normalized_title,
                "reusable_content": normalized_content,
                "preconditions": list(normalized_preconditions),
            }
        )
        base = cls(
            candidate_id=candidate_id,
            candidate_type=CandidateType(candidate_type),
            content_class=ContentClass(content_class),
            qualifying_sources=qualifying_sources,
            supporting_source_turn_ids=supporting_source_turn_ids,
            repository_scope_id=repository_scope_id,
            title=normalized_title,
            reusable_content=normalized_content,
            preconditions=normalized_preconditions,
            applicability_fingerprints=applicability_fingerprints,
            extraction_policy=extraction_policy,
            extraction_policy_version=extraction_policy_version,
            provenance_refs=provenance_refs,
            created_at=created_at,
            created_by=created_by,
            content_fingerprint=content_fingerprint,
            manifest_digest="0" * 64,
        )
        return cls(**{**base.__dict__, "manifest_digest": structural_digest(base._manifest_payload())})

    def __post_init__(self) -> None:
        if self.schema != KNOWLEDGE_CANDIDATE_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported knowledge candidate schema")
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id", maximum=256))
        object.__setattr__(self, "candidate_type", CandidateType(self.candidate_type))
        object.__setattr__(self, "content_class", ContentClass(self.content_class))
        if not self.qualifying_sources:
            raise ValueError("at least one qualifying source is required")
        turns = tuple(source.turn_id for source in self.qualifying_sources)
        if len(turns) != len(set(turns)):
            raise ValueError("qualifying source Turns must be unique")
        object.__setattr__(
            self,
            "supporting_source_turn_ids",
            _strings(self.supporting_source_turn_ids, "supporting_source_turn_ids"),
        )
        require_digest(self.repository_scope_id, "repository_scope_id")
        object.__setattr__(self, "title", _text(self.title, "title", maximum=256))
        object.__setattr__(
            self,
            "reusable_content",
            _text(self.reusable_content, "reusable_content", maximum=32_768, multiline=True),
        )
        object.__setattr__(self, "preconditions", _strings(self.preconditions, "preconditions", maximum=1024))
        fingerprints = tuple(sorted((str(key), str(value)) for key, value in self.applicability_fingerprints))
        if len({key for key, _ in fingerprints}) != len(fingerprints):
            raise ValueError("applicability fingerprint keys must be unique")
        for key, digest in fingerprints:
            _text(key, "applicability fingerprint key", maximum=128)
            require_digest(digest, "applicability fingerprint")
        object.__setattr__(self, "applicability_fingerprints", fingerprints)
        object.__setattr__(
            self, "extraction_policy", _text(self.extraction_policy, "extraction_policy", maximum=128)
        )
        if self.extraction_policy_version < 1:
            raise ValueError("extraction_policy_version must be positive")
        object.__setattr__(self, "provenance_refs", _strings(self.provenance_refs, "provenance_refs"))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at", maximum=64))
        object.__setattr__(self, "created_by", _text(self.created_by, "created_by", maximum=256))
        require_digest(self.content_fingerprint, "content_fingerprint")
        require_digest(self.manifest_digest, "manifest_digest")

    def _manifest_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("manifest_digest")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type.value,
            "content_class": self.content_class.value,
            "qualifying_sources": [source.to_dict() for source in self.qualifying_sources],
            "supporting_source_turn_ids": list(self.supporting_source_turn_ids),
            "repository_scope_id": self.repository_scope_id,
            "title": self.title,
            "reusable_content": self.reusable_content,
            "preconditions": list(self.preconditions),
            "applicability_fingerprints": [list(item) for item in self.applicability_fingerprints],
            "extraction_policy": self.extraction_policy,
            "extraction_policy_version": self.extraction_policy_version,
            "provenance_refs": list(self.provenance_refs),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "content_fingerprint": self.content_fingerprint,
            "manifest_digest": self.manifest_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeCandidate":
        candidate = cls(
            candidate_id=str(value["candidate_id"]),
            candidate_type=CandidateType(value["candidate_type"]),
            content_class=ContentClass(value["content_class"]),
            qualifying_sources=tuple(
                SourceTurnReference.from_dict(item) for item in value["qualifying_sources"]
            ),
            supporting_source_turn_ids=tuple(str(item) for item in value["supporting_source_turn_ids"]),
            repository_scope_id=str(value["repository_scope_id"]),
            title=str(value["title"]),
            reusable_content=str(value["reusable_content"]),
            preconditions=tuple(str(item) for item in value["preconditions"]),
            applicability_fingerprints=tuple(
                (str(item[0]), str(item[1])) for item in value["applicability_fingerprints"]
            ),
            extraction_policy=str(value["extraction_policy"]),
            extraction_policy_version=int(value["extraction_policy_version"]),
            provenance_refs=tuple(str(item) for item in value["provenance_refs"]),
            created_at=str(value["created_at"]),
            created_by=str(value["created_by"]),
            content_fingerprint=str(value["content_fingerprint"]),
            manifest_digest=str(value["manifest_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        expected_content = structural_digest(
            {
                "candidate_type": candidate.candidate_type.value,
                "content_class": candidate.content_class.value,
                "repository_scope_id": candidate.repository_scope_id,
                "title": candidate.title,
                "reusable_content": candidate.reusable_content,
                "preconditions": list(candidate.preconditions),
            }
        )
        if candidate.content_fingerprint != expected_content:
            raise ValueError("knowledge candidate content fingerprint mismatch")
        if candidate.manifest_digest != structural_digest(candidate._manifest_payload()):
            raise ValueError("knowledge candidate manifest digest mismatch")
        return candidate


__all__ = [
    "KNOWLEDGE_CANDIDATE_SCHEMA",
    "SCHEMA_VERSION",
    "CandidateType",
    "ContentClass",
    "KnowledgeCandidate",
    "SourceTurnReference",
    "canonical_json",
    "require_digest",
    "structural_digest",
]
