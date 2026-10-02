"""Deterministic technical validation for immutable knowledge candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .index import CandidateComparison, CandidateRelation
from .store import ImmutableWriteStatus, KnowledgeRecordStore
from .task_success import TaskSuccessStatus
from .types import CandidateType, ContentClass, KnowledgeCandidate, require_digest, structural_digest

VALIDATION_SCHEMA = "pico.knowledge-validation.v1"
SCHEMA_VERSION = 1

_AUTHORITY_LANGUAGE = re.compile(
    r"\b(?:bypass (?:approval|sandbox)|grant (?:permission|access)|"
    r"execute tools? without|ignore (?:tool policy|permissions?))\b",
    re.IGNORECASE,
)


class ValidationStatus(str, Enum):
    PASS = "pass"  # noqa: S105 -- validation outcome
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ValidationReason(str, Enum):
    CANDIDATE_INTEGRITY_INVALID = "candidate_integrity_invalid"
    REPOSITORY_SCOPE_MISMATCH = "repository_scope_mismatch"
    REPOSITORY_SCOPE_MISSING = "repository_scope_missing"
    PROVENANCE_MISSING = "provenance_missing"
    TASK_SUCCESS_MISSING = "task_success_missing"
    TASK_SUCCESS_INVALID = "task_success_invalid"
    UNSUPPORTED_EXTRACTION_POLICY = "unsupported_extraction_policy"
    COMPARISON_MISMATCH = "comparison_mismatch"
    MEMORY_FACT_STRUCTURE_INVALID = "memory_fact_structure_invalid"
    EXPERIENCE_STRUCTURE_INVALID = "experience_structure_invalid"
    SKILL_STRUCTURE_INVALID = "skill_structure_invalid"
    EXECUTION_AUTHORITY_LANGUAGE = "execution_authority_language"


@dataclass(frozen=True)
class CandidateValidationResult:
    validation_id: str
    candidate_id: str
    candidate_manifest_digest: str
    repository_scope_id: str
    status: ValidationStatus
    reasons: tuple[ValidationReason, ...]
    comparison_relation: CandidateRelation
    comparison_candidate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    validator_id: str
    validator_version: str
    validated_at: str
    validation_digest: str
    schema: str = VALIDATION_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != VALIDATION_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported validation schema")
        require_digest(self.candidate_manifest_digest, "candidate_manifest_digest")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.validation_digest, "validation_digest")
        if self.status is ValidationStatus.PASS and self.reasons:
            raise ValueError("passing validation cannot contain failure reasons")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "repository_scope_id": self.repository_scope_id,
            "status": self.status.value,
            "reasons": [item.value for item in self.reasons],
            "comparison_relation": self.comparison_relation.value,
            "comparison_candidate_ids": list(self.comparison_candidate_ids),
            "evidence_refs": list(self.evidence_refs),
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "validated_at": self.validated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "validation_digest": self.validation_digest}

    @classmethod
    def create(cls, **values: Any) -> CandidateValidationResult:
        base = cls(**values, validation_digest="0" * 64)
        return cls(**{**base.__dict__, "validation_digest": structural_digest(base._payload())})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateValidationResult:
        result = cls(
            validation_id=str(value["validation_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            status=ValidationStatus(value["status"]),
            reasons=tuple(ValidationReason(item) for item in value["reasons"]),
            comparison_relation=CandidateRelation(value["comparison_relation"]),
            comparison_candidate_ids=tuple(str(item) for item in value["comparison_candidate_ids"]),
            evidence_refs=tuple(str(item) for item in value["evidence_refs"]),
            validator_id=str(value["validator_id"]),
            validator_version=str(value["validator_version"]),
            validated_at=str(value["validated_at"]),
            validation_digest=str(value["validation_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if result.validation_digest != structural_digest(result._payload()):
            raise ValueError("validation digest mismatch")
        return result


def _candidate_structure_reasons(candidate: KnowledgeCandidate) -> set[ValidationReason]:
    reasons: set[ValidationReason] = set()
    has_guard = any(key.startswith("guard:") for key, _ in candidate.applicability_fingerprints)
    if candidate.candidate_type is CandidateType.MEMORY_FACT:
        has_fact = any(key.startswith("fact:") for key, _ in candidate.applicability_fingerprints)
        if candidate.content_class is not ContentClass.FACT or not has_fact or not has_guard:
            reasons.add(ValidationReason.MEMORY_FACT_STRUCTURE_INVALID)
    elif candidate.candidate_type is CandidateType.EXPERIENCE:
        if candidate.content_class not in {
            ContentClass.STRATEGY,
            ContentClass.RECOVERY,
            ContentClass.WARNING,
            ContentClass.ANTI_PATTERN,
        } or not candidate.preconditions or not has_guard:
            reasons.add(ValidationReason.EXPERIENCE_STRUCTURE_INVALID)
    elif candidate.candidate_type is CandidateType.SKILL_CANDIDATE:
        if (
            candidate.content_class is not ContentClass.PROCEDURE
            or not candidate.preconditions
            or "Validation expectations:" not in candidate.reusable_content
            or not has_guard
        ):
            reasons.add(ValidationReason.SKILL_STRUCTURE_INVALID)
    if _AUTHORITY_LANGUAGE.search(
        "\n".join((candidate.title, candidate.reusable_content, *candidate.preconditions))
    ):
        reasons.add(ValidationReason.EXECUTION_AUTHORITY_LANGUAGE)
    return reasons


def validate_candidate(
    store: KnowledgeRecordStore,
    *,
    candidate_id: str,
    comparison: CandidateComparison,
    expected_repository_scope_id: str,
    validation_id: str,
    validator_id: str,
    validator_version: str,
    validated_at: str,
) -> CandidateValidationResult:
    candidate = store.read_candidate(candidate_id)
    if candidate is None:
        raise ValueError("candidate does not exist")
    fail: set[ValidationReason] = set()
    inconclusive: set[ValidationReason] = set()
    try:
        KnowledgeCandidate.from_dict(candidate.to_dict())
    except (KeyError, TypeError, ValueError):
        fail.add(ValidationReason.CANDIDATE_INTEGRITY_INVALID)
    if candidate.repository_scope_id != expected_repository_scope_id:
        fail.add(ValidationReason.REPOSITORY_SCOPE_MISMATCH)
    scope = store.read_scope(candidate.repository_scope_id)
    if scope is None:
        inconclusive.add(ValidationReason.REPOSITORY_SCOPE_MISSING)
    if comparison.candidate_id != candidate.candidate_id:
        fail.add(ValidationReason.COMPARISON_MISMATCH)
    if not candidate.provenance_refs or not candidate.qualifying_sources:
        inconclusive.add(ValidationReason.PROVENANCE_MISSING)
    if candidate.extraction_policy != "p3.manual-candidate" or candidate.extraction_policy_version != 1:
        fail.add(ValidationReason.UNSUPPORTED_EXTRACTION_POLICY)

    evidence_refs: set[str] = set(candidate.provenance_refs)
    found_success = False
    for source in candidate.qualifying_sources:
        for evidence_id, evidence_digest in zip(
            source.task_success_evidence_ids,
            source.task_success_evidence_digests,
            strict=True,
        ):
            task_success = store.read_task_success(evidence_id)
            if task_success is None:
                inconclusive.add(ValidationReason.TASK_SUCCESS_MISSING)
                continue
            if (
                task_success.evidence_digest != evidence_digest
                or task_success.source_turn_id != source.turn_id
                or task_success.repository_scope_id != candidate.repository_scope_id
            ):
                fail.add(ValidationReason.TASK_SUCCESS_INVALID)
                continue
            evidence_refs.add(f"task-success:{task_success.evidence_digest}")
            if task_success.status is TaskSuccessStatus.PASS:
                found_success = True
            elif task_success.status is TaskSuccessStatus.FAIL:
                fail.add(ValidationReason.TASK_SUCCESS_INVALID)
    if not found_success:
        inconclusive.add(ValidationReason.TASK_SUCCESS_MISSING)
    fail.update(_candidate_structure_reasons(candidate))

    status = (
        ValidationStatus.FAIL
        if fail
        else ValidationStatus.INCONCLUSIVE
        if inconclusive
        else ValidationStatus.PASS
    )
    reasons = tuple(sorted(fail | inconclusive, key=lambda item: item.value))
    result = CandidateValidationResult.create(
        validation_id=validation_id,
        candidate_id=candidate.candidate_id,
        candidate_manifest_digest=candidate.manifest_digest,
        repository_scope_id=candidate.repository_scope_id,
        status=status,
        reasons=reasons,
        comparison_relation=comparison.relation,
        comparison_candidate_ids=comparison.existing_candidate_ids,
        evidence_refs=tuple(sorted(evidence_refs)),
        validator_id=validator_id,
        validator_version=validator_version,
        validated_at=validated_at,
    )
    if store.write_validation(result) is ImmutableWriteStatus.CONFLICT:
        raise ValueError("immutable validation result conflict")
    return result


__all__ = [
    "CandidateValidationResult",
    "ValidationReason",
    "ValidationStatus",
    "validate_candidate",
]
