"""Deterministic P3.1 eligibility gate over public P1C evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pico.tracing.evidence import EvidenceCompleteness
from pico.tracing.replay import TraceReplayResult, compute_replay_digest
from pico.tracing.verifier import (
    TraceVerificationResult,
    VerificationStatus,
    compute_verification_digest,
)

from .scope import RepositoryScopeResolution, RepositoryScopeStatus
from .task_success import TaskSuccessEvidence, TaskSuccessStatus
from .types import CandidateType, structural_digest

ELIGIBILITY_SCHEMA = "pico.candidate-eligibility.v1"
SCHEMA_VERSION = 1
SUPPORTED_POLICY = ("p3.manual-candidate", 1)


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    INCONCLUSIVE = "inconclusive"


class EligibilityReason(str, Enum):
    MISSING_TURN_ID = "missing_turn_id"
    REPLAY_DIGEST_INVALID = "replay_digest_invalid"
    VERIFICATION_DIGEST_INVALID = "verification_digest_invalid"
    STRUCTURAL_VERIFICATION_FAILED = "structural_verification_failed"
    STRUCTURAL_VERIFICATION_INCONCLUSIVE = "structural_verification_inconclusive"
    SOURCE_EVIDENCE_INCOMPLETE = "source_evidence_incomplete"
    SOURCE_EVIDENCE_CORRUPT = "source_evidence_corrupt"
    TASK_SUCCESS_MISSING = "task_success_missing"
    TASK_SUCCESS_FAILED = "task_success_failed"
    TASK_SUCCESS_INCONCLUSIVE = "task_success_inconclusive"
    TASK_SUCCESS_DIGEST_INVALID = "task_success_digest_invalid"
    TASK_SUCCESS_TURN_MISMATCH = "task_success_turn_mismatch"
    TASK_SUCCESS_STATE_MISMATCH = "task_success_state_mismatch"
    REPOSITORY_SCOPE_MISSING = "repository_scope_missing"
    REPOSITORY_SCOPE_INVALID = "repository_scope_invalid"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    UNSUPPORTED_POLICY_VERSION = "unsupported_policy_version"


@dataclass(frozen=True)
class CandidateEligibilityInput:
    turn_id: str
    replay: TraceReplayResult
    verification: TraceVerificationResult
    task_success: tuple[TaskSuccessEvidence, ...]
    scope: RepositoryScopeResolution
    resulting_state_digest: str | None
    extraction_policy: str
    extraction_policy_version: int
    candidate_type: CandidateType
    provenance_turn_ids: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class CandidateEligibilityResult:
    status: EligibilityStatus
    reasons: tuple[EligibilityReason, ...]
    turn_id: str
    repository_scope_id: str | None
    checked_evidence_digests: tuple[str, ...]
    policy: str
    policy_version: int
    result_digest: str
    schema: str = ELIGIBILITY_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "turn_id": self.turn_id,
            "repository_scope_id": self.repository_scope_id,
            "checked_evidence_digests": list(self.checked_evidence_digests),
            "policy": self.policy,
            "policy_version": self.policy_version,
            "result_digest": self.result_digest,
        }


def evaluate_candidate_eligibility(value: CandidateEligibilityInput) -> CandidateEligibilityResult:
    ineligible: set[EligibilityReason] = set()
    inconclusive: set[EligibilityReason] = set()
    checked: set[str] = set()

    if value.schema_version != SCHEMA_VERSION:
        ineligible.add(EligibilityReason.UNSUPPORTED_SCHEMA_VERSION)
    if (value.extraction_policy, value.extraction_policy_version) != SUPPORTED_POLICY:
        ineligible.add(EligibilityReason.UNSUPPORTED_POLICY_VERSION)
    if not value.turn_id.strip():
        inconclusive.add(EligibilityReason.MISSING_TURN_ID)
    if value.turn_id not in value.provenance_turn_ids:
        ineligible.add(EligibilityReason.PROVENANCE_MISMATCH)

    try:
        replay_digest = compute_replay_digest(value.replay)
    except Exception:
        replay_digest = ""
    if replay_digest != value.replay.replay_digest:
        ineligible.add(EligibilityReason.REPLAY_DIGEST_INVALID)
    else:
        checked.add(value.replay.replay_digest)
    if value.replay.turn_id != value.turn_id:
        ineligible.add(EligibilityReason.PROVENANCE_MISMATCH)
    if value.replay.evidence_status is EvidenceCompleteness.CORRUPT:
        ineligible.add(EligibilityReason.SOURCE_EVIDENCE_CORRUPT)
    elif value.replay.evidence_status is not EvidenceCompleteness.COMPLETE:
        inconclusive.add(EligibilityReason.SOURCE_EVIDENCE_INCOMPLETE)

    try:
        verification_digest = compute_verification_digest(value.verification)
    except Exception:
        verification_digest = ""
    if verification_digest != value.verification.verification_digest:
        ineligible.add(EligibilityReason.VERIFICATION_DIGEST_INVALID)
    else:
        checked.add(value.verification.verification_digest)
    if (
        value.verification.turn_id != value.turn_id
        or value.verification.replay_digest != value.replay.replay_digest
    ):
        ineligible.add(EligibilityReason.PROVENANCE_MISMATCH)
    if value.verification.overall_status is VerificationStatus.FAIL:
        ineligible.add(EligibilityReason.STRUCTURAL_VERIFICATION_FAILED)
    elif value.verification.overall_status is VerificationStatus.INCONCLUSIVE:
        inconclusive.add(EligibilityReason.STRUCTURAL_VERIFICATION_INCONCLUSIVE)

    scope_id = value.scope.identity.repository_scope_id if value.scope.resolved else None
    if value.scope.status in {RepositoryScopeStatus.INVALID, RepositoryScopeStatus.AMBIGUOUS}:
        ineligible.add(EligibilityReason.REPOSITORY_SCOPE_INVALID)
    elif not value.scope.resolved:
        inconclusive.add(EligibilityReason.REPOSITORY_SCOPE_MISSING)

    if not value.task_success:
        inconclusive.add(EligibilityReason.TASK_SUCCESS_MISSING)
    matching: list[TaskSuccessEvidence] = []
    for evidence in value.task_success:
        try:
            decoded = TaskSuccessEvidence.from_dict(evidence.to_dict())
        except (KeyError, TypeError, ValueError):
            ineligible.add(EligibilityReason.TASK_SUCCESS_DIGEST_INVALID)
            continue
        checked.add(decoded.evidence_digest)
        if decoded.source_turn_id != value.turn_id:
            ineligible.add(EligibilityReason.TASK_SUCCESS_TURN_MISMATCH)
            continue
        if scope_id is not None and decoded.repository_scope_id != scope_id:
            ineligible.add(EligibilityReason.PROVENANCE_MISMATCH)
            continue
        if decoded.target_state_digest is None:
            inconclusive.add(EligibilityReason.TASK_SUCCESS_STATE_MISMATCH)
            continue
        if value.resulting_state_digest is None:
            inconclusive.add(EligibilityReason.TASK_SUCCESS_STATE_MISMATCH)
            continue
        if decoded.target_state_digest != value.resulting_state_digest:
            ineligible.add(EligibilityReason.TASK_SUCCESS_STATE_MISMATCH)
            continue
        matching.append(decoded)

    if any(item.status is TaskSuccessStatus.FAIL for item in matching):
        ineligible.add(EligibilityReason.TASK_SUCCESS_FAILED)
    elif any(item.status is TaskSuccessStatus.PASS for item in matching):
        pass
    elif matching:
        inconclusive.add(EligibilityReason.TASK_SUCCESS_INCONCLUSIVE)
    elif value.task_success and not ineligible:
        inconclusive.add(EligibilityReason.TASK_SUCCESS_INCONCLUSIVE)

    reasons = tuple(sorted(ineligible | inconclusive, key=lambda item: item.value))
    status = (
        EligibilityStatus.INELIGIBLE
        if ineligible
        else EligibilityStatus.INCONCLUSIVE
        if inconclusive
        else EligibilityStatus.ELIGIBLE
    )
    payload = {
        "schema": ELIGIBILITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status.value,
        "reasons": [reason.value for reason in reasons],
        "turn_id": value.turn_id,
        "repository_scope_id": scope_id,
        "checked_evidence_digests": sorted(checked),
        "policy": value.extraction_policy,
        "policy_version": value.extraction_policy_version,
    }
    return CandidateEligibilityResult(
        status=status,
        reasons=reasons,
        turn_id=value.turn_id,
        repository_scope_id=scope_id,
        checked_evidence_digests=tuple(sorted(checked)),
        policy=value.extraction_policy,
        policy_version=value.extraction_policy_version,
        result_digest=structural_digest(payload),
    )


__all__ = [
    "ELIGIBILITY_SCHEMA",
    "SUPPORTED_POLICY",
    "CandidateEligibilityInput",
    "CandidateEligibilityResult",
    "EligibilityReason",
    "EligibilityStatus",
    "evaluate_candidate_eligibility",
]
