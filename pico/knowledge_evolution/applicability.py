"""Deterministic, fail-closed applicability checks for reviewed knowledge."""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .lifecycle import KnowledgeLifecycleManager, LifecycleState
from .materialize import MaterializationStatus
from .review import ReviewDecision
from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .types import KnowledgeCandidate, require_digest, structural_digest
from .validation import ValidationStatus

APPLICABILITY_SCHEMA = "pico.knowledge-applicability.v1"
SCHEMA_VERSION = 1


class ApplicabilityStatus(str, Enum):
    APPLICABLE = "applicable"
    STALE = "stale"
    NOT_APPLICABLE = "not_applicable"
    INCONCLUSIVE = "inconclusive"


class ApplicabilityReason(str, Enum):
    APPLICABLE = "applicable"
    NOT_ACTIVE = "not_active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    WRONG_SCOPE = "wrong_scope"
    DIGEST_MISMATCH = "digest_mismatch"
    MISSING_MATERIALIZATION = "missing_materialization"
    MISSING_GUARD_EVIDENCE = "missing_guard_evidence"
    FINGERPRINT_CHANGED = "fingerprint_changed"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    MISSING_TOOL = "missing_tool"
    MISSING_BINARY = "missing_binary"


@dataclass(frozen=True)
class ApplicabilityEnvironment:
    repository_scope_id: str
    workspace: Path
    available_tools: tuple[str, ...] = ()
    fingerprints: tuple[tuple[str, str], ...] = ()
    dependency_versions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class KnowledgeApplicabilityResult:
    applicability_id: str
    candidate_id: str
    candidate_manifest_digest: str
    repository_scope_id: str
    lifecycle_state: LifecycleState
    lifecycle_transition_digest: str | None
    materialization_id: str | None
    materialization_digest: str | None
    status: ApplicabilityStatus
    reasons: tuple[ApplicabilityReason, ...]
    guard_evidence: tuple[tuple[str, str], ...]
    evaluated_at: str
    evaluator_policy: str
    evaluator_version: int
    applicability_digest: str
    schema: str = APPLICABILITY_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> "KnowledgeApplicabilityResult":
        base = cls(**values, applicability_digest="0" * 64)
        return cls(**{**base.__dict__, "applicability_digest": structural_digest(base._payload())})

    def __post_init__(self) -> None:
        if self.schema != APPLICABILITY_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported applicability schema")
        require_digest(self.candidate_manifest_digest, "candidate_manifest_digest")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.applicability_digest, "applicability_digest")
        if self.lifecycle_transition_digest is not None:
            require_digest(self.lifecycle_transition_digest, "lifecycle_transition_digest")
        if self.materialization_digest is not None:
            require_digest(self.materialization_digest, "materialization_digest")
        if not self.reasons:
            raise ValueError("applicability reasons cannot be empty")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "applicability_id": self.applicability_id,
            "candidate_id": self.candidate_id,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "repository_scope_id": self.repository_scope_id,
            "lifecycle_state": self.lifecycle_state.value,
            "lifecycle_transition_digest": self.lifecycle_transition_digest,
            "materialization_id": self.materialization_id,
            "materialization_digest": self.materialization_digest,
            "status": self.status.value,
            "reasons": [item.value for item in self.reasons],
            "guard_evidence": [list(item) for item in self.guard_evidence],
            "evaluated_at": self.evaluated_at,
            "evaluator_policy": self.evaluator_policy,
            "evaluator_version": self.evaluator_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "applicability_digest": self.applicability_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeApplicabilityResult":
        result = cls(
            applicability_id=str(value["applicability_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_manifest_digest=str(value["candidate_manifest_digest"]),
            repository_scope_id=str(value["repository_scope_id"]),
            lifecycle_state=LifecycleState(value["lifecycle_state"]),
            lifecycle_transition_digest=(str(value["lifecycle_transition_digest"]) if value.get("lifecycle_transition_digest") else None),
            materialization_id=(str(value["materialization_id"]) if value.get("materialization_id") else None),
            materialization_digest=(str(value["materialization_digest"]) if value.get("materialization_digest") else None),
            status=ApplicabilityStatus(value["status"]),
            reasons=tuple(ApplicabilityReason(item) for item in value["reasons"]),
            guard_evidence=tuple((str(item[0]), str(item[1])) for item in value["guard_evidence"]),
            evaluated_at=str(value["evaluated_at"]),
            evaluator_policy=str(value["evaluator_policy"]),
            evaluator_version=int(value["evaluator_version"]),
            applicability_digest=str(value["applicability_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if result.applicability_digest != structural_digest(result._payload()):
            raise ValueError("applicability result digest mismatch")
        return result


def _artifact_integrity(
    store: KnowledgeRecordStore,
    candidate: KnowledgeCandidate,
    lifecycle_transition_digest: str | None,
) -> tuple[Any | None, bool]:
    try:
        results = [
            item
            for item in store.list_materialization_results(candidate.candidate_id)
            if item.status in {MaterializationStatus.CREATED, MaterializationStatus.UNCHANGED}
        ]
    except KnowledgeStoreError:
        return None, False
    if not results:
        return None, False
    result = sorted(results, key=lambda item: item.materialization_id)[-1]
    if (
        result.candidate_manifest_digest != candidate.manifest_digest
        or result.repository_scope_id != candidate.repository_scope_id
        or result.lifecycle_transition_digest != lifecycle_transition_digest
        or not result.artifact_relative_path
        or not result.content_digest
    ):
        return result, False
    try:
        validations = store.list_validations(candidate_id=candidate.candidate_id)
        reviews = store.list_reviews(candidate_id=candidate.candidate_id)
    except KnowledgeStoreError:
        return result, False
    validation_ok = any(
        item.validation_digest == result.validation_digest
        and item.status is ValidationStatus.PASS
        and item.candidate_manifest_digest == candidate.manifest_digest
        for item in validations
    )
    review_ok = any(
        item.review_digest == result.review_digest
        and item.decision is ReviewDecision.APPROVE
        and item.validation_digest == result.validation_digest
        and item.candidate_manifest_digest == candidate.manifest_digest
        for item in reviews
    )
    if not validation_ok or not review_ok:
        return result, False
    path = store.materialized / result.artifact_relative_path
    try:
        content = path.read_bytes()
    except OSError:
        return result, False
    return result, hashlib.sha256(content).hexdigest() == result.content_digest


def _guard_value(key: str, environment: ApplicabilityEnvironment) -> tuple[str | None, ApplicabilityReason | None]:
    supplied = dict(environment.fingerprints)
    if key in supplied:
        return supplied[key], None
    key = key.removeprefix("guard:")
    if key in supplied:
        return supplied[key], None
    if key.startswith("file:"):
        relative = Path(key.removeprefix("file:"))
        if relative.is_absolute() or ".." in relative.parts:
            return None, ApplicabilityReason.MISSING_GUARD_EVIDENCE
        try:
            return hashlib.sha256((environment.workspace / relative).read_bytes()).hexdigest(), None
        except OSError:
            return None, ApplicabilityReason.MISSING_GUARD_EVIDENCE
    if key.startswith("tool:"):
        name = key.removeprefix("tool:")
        return ("present" if name in environment.available_tools else None), (
            None if name in environment.available_tools else ApplicabilityReason.MISSING_TOOL
        )
    if key.startswith("binary:"):
        name = key.removeprefix("binary:")
        return ("present" if shutil.which(name) else None), (
            None if shutil.which(name) else ApplicabilityReason.MISSING_BINARY
        )
    if key.startswith("dependency:"):
        name = key.removeprefix("dependency:")
        versions = dict(environment.dependency_versions)
        if name in versions:
            return structural_digest({"version": versions[name]}), None
        try:
            return structural_digest({"version": importlib.metadata.version(name)}), None
        except importlib.metadata.PackageNotFoundError:
            return None, ApplicabilityReason.DEPENDENCY_MISMATCH
    return None, ApplicabilityReason.MISSING_GUARD_EVIDENCE


def evaluate_applicability(
    store: KnowledgeRecordStore,
    *,
    candidate_id: str,
    environment: ApplicabilityEnvironment,
    applicability_id: str,
    evaluated_at: str,
) -> KnowledgeApplicabilityResult:
    candidate = store.read_candidate(candidate_id)
    if candidate is None:
        raise ValueError("candidate does not exist")
    projection = KnowledgeLifecycleManager(store).rebuild(candidate_id)
    reasons: list[ApplicabilityReason] = []
    guard_evidence: list[tuple[str, str]] = []
    materialization = None
    artifact_valid = False

    if candidate.repository_scope_id != environment.repository_scope_id:
        status = ApplicabilityStatus.NOT_APPLICABLE
        reasons.append(ApplicabilityReason.WRONG_SCOPE)
    elif projection.state is not LifecycleState.ACTIVE:
        status = ApplicabilityStatus.NOT_APPLICABLE
        reasons.append(
            ApplicabilityReason.DEPRECATED
            if projection.state is LifecycleState.DEPRECATED
            else ApplicabilityReason.SUPERSEDED
            if projection.state is LifecycleState.SUPERSEDED
            else ApplicabilityReason.NOT_ACTIVE
        )
    else:
        materialization, artifact_valid = _artifact_integrity(
            store,
            candidate,
            projection.head_transition_digest,
        )
        if materialization is None:
            status = ApplicabilityStatus.INCONCLUSIVE
            reasons.append(ApplicabilityReason.MISSING_MATERIALIZATION)
        elif not artifact_valid:
            status = ApplicabilityStatus.INCONCLUSIVE
            reasons.append(ApplicabilityReason.DIGEST_MISMATCH)
        else:
            status = ApplicabilityStatus.APPLICABLE
        for key, expected in candidate.applicability_fingerprints:
            if status is not ApplicabilityStatus.APPLICABLE:
                break
            if not key.startswith("guard:"):
                continue
            current, failure = _guard_value(key, environment)
            guard_evidence.append((key, current or "unavailable"))
            if failure is not None:
                reasons.append(failure)
            elif key.startswith(("guard:tool:", "guard:binary:")):
                continue
            elif current != expected:
                reasons.append(
                    ApplicabilityReason.DEPENDENCY_MISMATCH
                    if key.startswith("guard:dependency:")
                    else ApplicabilityReason.FINGERPRINT_CHANGED
                )
        if status is not ApplicabilityStatus.APPLICABLE:
            pass
        elif any(item in {ApplicabilityReason.FINGERPRINT_CHANGED, ApplicabilityReason.DEPENDENCY_MISMATCH} for item in reasons):
            status = ApplicabilityStatus.STALE
        elif reasons:
            status = (
                ApplicabilityStatus.NOT_APPLICABLE
                if any(item in {ApplicabilityReason.MISSING_TOOL, ApplicabilityReason.MISSING_BINARY} for item in reasons)
                else ApplicabilityStatus.INCONCLUSIVE
            )
        else:
            status = ApplicabilityStatus.APPLICABLE
            reasons.append(ApplicabilityReason.APPLICABLE)

    result = KnowledgeApplicabilityResult.create(
        applicability_id=applicability_id,
        candidate_id=candidate.candidate_id,
        candidate_manifest_digest=candidate.manifest_digest,
        repository_scope_id=candidate.repository_scope_id,
        lifecycle_state=projection.state,
        lifecycle_transition_digest=projection.head_transition_digest,
        materialization_id=(materialization.materialization_id if materialization else None),
        materialization_digest=(materialization.result_digest if materialization else None),
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        guard_evidence=tuple(guard_evidence),
        evaluated_at=evaluated_at,
        evaluator_policy="p3.runtime-applicability",
        evaluator_version=1,
    )
    if store.write_applicability(result) is ImmutableWriteStatus.CONFLICT:
        raise KnowledgeStoreError("immutable applicability result conflict")
    return result


__all__ = [
    "ApplicabilityEnvironment",
    "ApplicabilityReason",
    "ApplicabilityStatus",
    "KnowledgeApplicabilityResult",
    "evaluate_applicability",
]
