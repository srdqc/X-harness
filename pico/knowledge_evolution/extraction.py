"""Bounded P3.2 extraction orchestration over trusted P3.1 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from .canonicalize import (
    CanonicalProposal,
    KnowledgeProposal,
    ProposalRejectionReason,
    UnsafeKnowledgeContentError,
    inspect_unsafe_text,
    normalize_inline,
    validate_and_canonicalize,
)
from .eligibility import (
    CandidateEligibilityInput,
    CandidateEligibilityResult,
    EligibilityStatus,
    evaluate_candidate_eligibility,
)
from .index import CandidateComparison, CandidateIndex, CandidateRelation
from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .task_success import TaskSuccessStatus
from .types import (
    CandidateType,
    KnowledgeCandidate,
    SourceTurnReference,
    require_digest,
    structural_digest,
)

EXTRACTION_CONTEXT_SCHEMA = "pico.knowledge-extraction-context.v1"
EXTRACTION_RESULT_SCHEMA = "pico.knowledge-extraction-result.v1"
SCHEMA_VERSION = 1
MAX_TOOL_SUMMARIES = 32
MAX_RECOVERY_FACTS = 16
MAX_PROPOSALS = 16


class KnowledgeExtractor(Protocol):
    extractor_id: str
    extractor_version: str

    def extract(self, context: KnowledgeExtractionContext) -> Sequence[object]: ...


@dataclass(frozen=True)
class ToolExecutionSummary:
    receipt_id: str
    requested_name: str
    resolved_name: str | None
    effect: str | None
    outcome: str | None
    failure_stage: str | None
    failure_category: str | None


@dataclass(frozen=True)
class TaskSuccessReference:
    evidence_id: str
    evidence_digest: str
    source_kind: str
    status: str
    target_state_digest: str | None


@dataclass(frozen=True)
class KnowledgeExtractionContext:
    extraction_id: str
    source_turn_id: str
    repository_scope_id: str
    eligibility_digest: str
    replay_digest: str
    verification_digest: str
    task_goal: str | None
    task_success_refs: tuple[TaskSuccessReference, ...]
    execution_facts: tuple[tuple[str, int | bool], ...]
    tool_summaries: tuple[ToolExecutionSummary, ...]
    total_tool_execution_count: int
    state_refs: tuple[str, ...]
    recovery_facts: tuple[str, ...]
    extraction_policy: str
    extraction_policy_version: int
    structural_digest: str
    schema: str = EXTRACTION_CONTEXT_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "extraction_id": self.extraction_id,
            "source_turn_id": self.source_turn_id,
            "repository_scope_id": self.repository_scope_id,
            "eligibility_digest": self.eligibility_digest,
            "replay_digest": self.replay_digest,
            "verification_digest": self.verification_digest,
            "task_goal": self.task_goal,
            "task_success_refs": [item.__dict__ for item in self.task_success_refs],
            "execution_facts": [list(item) for item in self.execution_facts],
            "tool_summaries": [item.__dict__ for item in self.tool_summaries],
            "total_tool_execution_count": self.total_tool_execution_count,
            "state_refs": list(self.state_refs),
            "recovery_facts": list(self.recovery_facts),
            "extraction_policy": self.extraction_policy,
            "extraction_policy_version": self.extraction_policy_version,
            "structural_digest": self.structural_digest,
        }


class ExtractionStatus(str, Enum):
    COMPLETED = "completed"
    NOT_PERMITTED = "not_permitted"
    FAILED = "failed"


class ExtractionFailureReason(str, Enum):
    ELIGIBILITY_NOT_ELIGIBLE = "eligibility_not_eligible"
    ELIGIBILITY_MISMATCH = "eligibility_mismatch"
    CONTEXT_UNSAFE = "context_unsafe"
    EXTRACTOR_ERROR = "extractor_error"
    MALFORMED_RESPONSE = "malformed_response"
    NO_PROPOSALS = "no_proposals"
    NO_ACCEPTED_PROPOSALS = "no_accepted_proposals"
    STORE_CONFLICT = "store_conflict"


@dataclass(frozen=True)
class RejectedProposal:
    proposal_index: int
    reason: ProposalRejectionReason
    sanitization_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateExtractionOutcome:
    proposal_index: int
    candidate_id: str
    relation: CandidateRelation
    existing_candidate_ids: tuple[str, ...]
    write_status: str
    compared_candidate_count: int
    portable_hint: bool


@dataclass(frozen=True)
class KnowledgeExtractionResult:
    extraction_id: str
    source_turn_id: str
    repository_scope_id: str | None
    eligibility_digest: str
    context_digest: str | None
    extraction_policy: str
    extraction_policy_version: int
    extractor_id: str
    extractor_version: str
    extractor_attempt_count: int
    status: ExtractionStatus
    failure_reason: ExtractionFailureReason | None
    proposal_count: int
    proposal_types: tuple[str, ...]
    accepted_proposal_count: int
    rejected_proposals: tuple[RejectedProposal, ...]
    candidate_outcomes: tuple[CandidateExtractionOutcome, ...]
    sanitization_findings: tuple[str, ...]
    result_digest: str
    schema: str = EXTRACTION_RESULT_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != EXTRACTION_RESULT_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported extraction result schema")
        if self.extractor_attempt_count not in {0, 1}:
            raise ValueError("extractor_attempt_count must be zero or one")
        if min(self.proposal_count, self.accepted_proposal_count) < 0:
            raise ValueError("proposal counts must be non-negative")
        if self.accepted_proposal_count != len(self.candidate_outcomes):
            raise ValueError("accepted proposal count does not match outcomes")
        if self.accepted_proposal_count + len(self.rejected_proposals) > self.proposal_count:
            raise ValueError("proposal accounting exceeds proposal count")
        if self.status is ExtractionStatus.COMPLETED and self.failure_reason is not None:
            raise ValueError("completed extraction cannot have a failure reason")
        if self.status is not ExtractionStatus.COMPLETED and self.failure_reason is None:
            raise ValueError("non-completed extraction requires a failure reason")
        require_digest(self.eligibility_digest, "eligibility_digest")
        require_digest(self.result_digest, "result_digest")
        if self.repository_scope_id is not None:
            require_digest(self.repository_scope_id, "repository_scope_id")
        if self.context_digest is not None:
            require_digest(self.context_digest, "context_digest")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "extraction_id": self.extraction_id,
            "source_turn_id": self.source_turn_id,
            "repository_scope_id": self.repository_scope_id,
            "eligibility_digest": self.eligibility_digest,
            "context_digest": self.context_digest,
            "extraction_policy": self.extraction_policy,
            "extraction_policy_version": self.extraction_policy_version,
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "extractor_attempt_count": self.extractor_attempt_count,
            "status": self.status.value,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "proposal_count": self.proposal_count,
            "proposal_types": list(self.proposal_types),
            "accepted_proposal_count": self.accepted_proposal_count,
            "rejected_proposals": [
                {
                    "proposal_index": item.proposal_index,
                    "reason": item.reason.value,
                    "sanitization_findings": list(item.sanitization_findings),
                }
                for item in self.rejected_proposals
            ],
            "candidate_outcomes": [
                {
                    "proposal_index": item.proposal_index,
                    "candidate_id": item.candidate_id,
                    "relation": item.relation.value,
                    "existing_candidate_ids": list(item.existing_candidate_ids),
                    "write_status": item.write_status,
                    "compared_candidate_count": item.compared_candidate_count,
                    "portable_hint": item.portable_hint,
                }
                for item in self.candidate_outcomes
            ],
            "sanitization_findings": list(self.sanitization_findings),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "result_digest": self.result_digest}

    @classmethod
    def create(cls, **values: object) -> KnowledgeExtractionResult:
        result = cls(**values, result_digest="0" * 64)
        return cls(**{**result.__dict__, "result_digest": structural_digest(result._payload())})

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> KnowledgeExtractionResult:
        result = cls(
            extraction_id=str(value["extraction_id"]),
            source_turn_id=str(value["source_turn_id"]),
            repository_scope_id=(
                str(value["repository_scope_id"])
                if value.get("repository_scope_id") is not None
                else None
            ),
            eligibility_digest=str(value["eligibility_digest"]),
            context_digest=(
                str(value["context_digest"]) if value.get("context_digest") is not None else None
            ),
            extraction_policy=str(value["extraction_policy"]),
            extraction_policy_version=int(value["extraction_policy_version"]),
            extractor_id=str(value["extractor_id"]),
            extractor_version=str(value["extractor_version"]),
            extractor_attempt_count=int(value["extractor_attempt_count"]),
            status=ExtractionStatus(value["status"]),
            failure_reason=(
                ExtractionFailureReason(value["failure_reason"])
                if value.get("failure_reason") is not None
                else None
            ),
            proposal_count=int(value["proposal_count"]),
            proposal_types=tuple(str(item) for item in value["proposal_types"]),
            accepted_proposal_count=int(value["accepted_proposal_count"]),
            rejected_proposals=tuple(
                RejectedProposal(
                    proposal_index=int(item["proposal_index"]),
                    reason=ProposalRejectionReason(item["reason"]),
                    sanitization_findings=tuple(item["sanitization_findings"]),
                )
                for item in value["rejected_proposals"]
            ),
            candidate_outcomes=tuple(
                CandidateExtractionOutcome(
                    proposal_index=int(item["proposal_index"]),
                    candidate_id=str(item["candidate_id"]),
                    relation=CandidateRelation(item["relation"]),
                    existing_candidate_ids=tuple(item["existing_candidate_ids"]),
                    write_status=str(item["write_status"]),
                    compared_candidate_count=int(item["compared_candidate_count"]),
                    portable_hint=bool(item["portable_hint"]),
                )
                for item in value["candidate_outcomes"]
            ),
            sanitization_findings=tuple(value["sanitization_findings"]),
            result_digest=str(value["result_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if result.result_digest != structural_digest(result._payload()):
            raise ValueError("extraction result digest mismatch")
        return result


@dataclass(frozen=True)
class KnowledgeExtractionRequest:
    extraction_id: str
    eligibility_input: CandidateEligibilityInput
    eligibility_result: CandidateEligibilityResult
    task_goal: str | None = None
    recovery_facts: tuple[str, ...] = ()


def _context_payload(context: KnowledgeExtractionContext) -> dict[str, object]:
    payload = context.to_dict()
    payload.pop("structural_digest")
    return payload


def build_extraction_context(request: KnowledgeExtractionRequest) -> KnowledgeExtractionContext:
    value = request.eligibility_input
    result = request.eligibility_result
    scope_id = result.repository_scope_id
    if scope_id is None:
        raise ValueError("eligible extraction requires repository scope")
    task_goal = None
    if request.task_goal is not None:
        task_goal = normalize_inline(request.task_goal, maximum=1_024)
        inspect_unsafe_text(task_goal)
    if len(request.recovery_facts) > MAX_RECOVERY_FACTS:
        raise ValueError("too many recovery facts")
    recovery_facts = tuple(
        sorted(
            {normalize_inline(item, maximum=512) for item in request.recovery_facts},
            key=lambda item: (item.casefold(), item),
        )
    )
    for item in recovery_facts:
        inspect_unsafe_text(item)

    task_refs = tuple(
        TaskSuccessReference(
            evidence_id=item.evidence_id,
            evidence_digest=item.evidence_digest,
            source_kind=item.source_kind.value,
            status=item.status.value,
            target_state_digest=item.target_state_digest,
        )
        for item in sorted(value.task_success, key=lambda evidence: evidence.evidence_id)
    )
    derived = value.verification.derived_facts
    execution_facts = tuple(
        sorted(
            {
                "provider_attempt_count": derived.provider_attempt_count,
                "provider_recovered_after_failure": derived.provider_recovered_after_failure,
                "tool_execution_count": derived.tool_execution_count,
                "tool_failure_count": derived.tool_failure_count,
                "validation_failure_count": derived.validation_failure_count,
                "execution_failure_count": derived.execution_failure_count,
                "timeout_count": derived.timeout_count,
                "runtime_success": derived.runtime_success,
                "runtime_failure": derived.runtime_failure,
                "runtime_cancelled": derived.runtime_cancelled,
            }.items()
        )
    )
    tools = sorted(
        value.replay.tool_executions,
        key=lambda item: (
            item.started_sequence if item.started_sequence is not None else 2**31,
            item.receipt_id,
        ),
    )
    tool_summaries = tuple(
        ToolExecutionSummary(
            receipt_id=item.receipt_id,
            requested_name=item.requested_name,
            resolved_name=item.resolved_name,
            effect=item.effect,
            outcome=item.outcome,
            failure_stage=item.failure_stage,
            failure_category=item.failure_category,
        )
        for item in tools[:MAX_TOOL_SUMMARIES]
    )
    state_refs = tuple(
        sorted(
            {
                item.target_state_digest
                for item in value.task_success
                if item.target_state_digest is not None
            }
        )
    )
    base = KnowledgeExtractionContext(
        extraction_id=normalize_inline(request.extraction_id, maximum=256),
        source_turn_id=value.turn_id,
        repository_scope_id=scope_id,
        eligibility_digest=result.result_digest,
        replay_digest=value.replay.replay_digest,
        verification_digest=value.verification.verification_digest,
        task_goal=task_goal,
        task_success_refs=task_refs,
        execution_facts=execution_facts,
        tool_summaries=tool_summaries,
        total_tool_execution_count=len(tools),
        state_refs=state_refs,
        recovery_facts=recovery_facts,
        extraction_policy=value.extraction_policy,
        extraction_policy_version=value.extraction_policy_version,
        structural_digest="0" * 64,
    )
    return KnowledgeExtractionContext(
        **{**base.__dict__, "structural_digest": structural_digest(_context_payload(base))}
    )


def _candidate_id(
    context: KnowledgeExtractionContext,
    proposal: CanonicalProposal,
) -> str:
    digest = structural_digest(
        {
            "repository_scope_id": context.repository_scope_id,
            "source_turn_id": context.source_turn_id,
            "policy": context.extraction_policy,
            "policy_version": context.extraction_policy_version,
            "candidate_type": proposal.candidate_type.value,
            "content_class": proposal.content_class.value,
            "title": proposal.title,
            "content": proposal.reusable_content,
            "preconditions": list(proposal.preconditions),
            "applicability_fingerprints": [list(item) for item in proposal.applicability_fingerprints],
        }
    )
    return f"candidate-{digest}"


def _construct_candidate(
    context: KnowledgeExtractionContext,
    eligibility_input: CandidateEligibilityInput,
    proposal: CanonicalProposal,
    *,
    created_at: str,
    created_by: str,
) -> KnowledgeCandidate:
    qualifying = tuple(
        sorted(
            (
                item
                for item in eligibility_input.task_success
                if item.status is TaskSuccessStatus.PASS
                and item.source_turn_id == context.source_turn_id
                and item.repository_scope_id == context.repository_scope_id
            ),
            key=lambda item: item.evidence_id,
        )
    )
    source = SourceTurnReference(
        turn_id=context.source_turn_id,
        replay_digest=context.replay_digest,
        verification_digest=context.verification_digest,
        task_success_evidence_ids=tuple(item.evidence_id for item in qualifying),
        task_success_evidence_digests=tuple(item.evidence_digest for item in qualifying),
        evidence_refs=(f"eligibility:{context.eligibility_digest}",),
    )
    provenance = tuple(
        sorted(
            {
                f"context:{context.structural_digest}",
                f"replay:{context.replay_digest}",
                f"verification:{context.verification_digest}",
                *(f"task-success:{item.evidence_digest}" for item in qualifying),
                *(
                    item
                    for item in proposal.evidence_refs
                    if item
                    in {
                        *(reference.evidence_id for reference in context.task_success_refs),
                        *(summary.receipt_id for summary in context.tool_summaries),
                        *context.state_refs,
                    }
                ),
            }
        )
    )
    return KnowledgeCandidate.create(
        candidate_id=_candidate_id(context, proposal),
        candidate_type=proposal.candidate_type,
        content_class=proposal.content_class,
        qualifying_sources=(source,),
        repository_scope_id=context.repository_scope_id,
        title=proposal.title,
        reusable_content=proposal.reusable_content,
        preconditions=proposal.preconditions,
        applicability_fingerprints=proposal.applicability_fingerprints,
        extraction_policy=context.extraction_policy,
        extraction_policy_version=context.extraction_policy_version,
        provenance_refs=provenance,
        created_at=created_at,
        created_by=created_by,
    )


def _result(
    request: KnowledgeExtractionRequest,
    extractor: KnowledgeExtractor,
    *,
    status: ExtractionStatus,
    failure_reason: ExtractionFailureReason | None,
    proposal_count: int = 0,
    proposal_types: tuple[str, ...] = (),
    rejected: tuple[RejectedProposal, ...] = (),
    outcomes: tuple[CandidateExtractionOutcome, ...] = (),
    findings: tuple[str, ...] = (),
    extractor_attempt_count: int = 0,
    context_digest: str | None = None,
) -> KnowledgeExtractionResult:
    combined_findings = tuple(
        sorted(
            {
                *findings,
                *(finding for item in rejected for finding in item.sanitization_findings),
            }
        )
    )
    return KnowledgeExtractionResult.create(
        extraction_id=request.extraction_id,
        source_turn_id=request.eligibility_input.turn_id,
        repository_scope_id=request.eligibility_result.repository_scope_id,
        eligibility_digest=request.eligibility_result.result_digest,
        context_digest=context_digest,
        extraction_policy=request.eligibility_input.extraction_policy,
        extraction_policy_version=request.eligibility_input.extraction_policy_version,
        extractor_id=normalize_inline(extractor.extractor_id, maximum=256),
        extractor_version=normalize_inline(extractor.extractor_version, maximum=128),
        extractor_attempt_count=extractor_attempt_count,
        status=status,
        failure_reason=failure_reason,
        proposal_count=proposal_count,
        proposal_types=proposal_types,
        accepted_proposal_count=len(outcomes),
        rejected_proposals=rejected,
        candidate_outcomes=outcomes,
        sanitization_findings=combined_findings,
    )


def _persist_result(
    store: KnowledgeRecordStore, result: KnowledgeExtractionResult
) -> KnowledgeExtractionResult:
    status = store.write_extraction_result(result)
    if status is ImmutableWriteStatus.CONFLICT:
        raise KnowledgeStoreError("immutable extraction result conflict")
    return result


def extract_candidates(
    request: KnowledgeExtractionRequest,
    extractor: KnowledgeExtractor,
    store: KnowledgeRecordStore,
    *,
    created_at: str,
    created_by: str,
    top_k: int = 5,
) -> KnowledgeExtractionResult:
    recomputed = evaluate_candidate_eligibility(request.eligibility_input)
    if recomputed != request.eligibility_result:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.NOT_PERMITTED,
                failure_reason=ExtractionFailureReason.ELIGIBILITY_MISMATCH,
            ),
        )
    if recomputed.status is not EligibilityStatus.ELIGIBLE:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.NOT_PERMITTED,
                failure_reason=ExtractionFailureReason.ELIGIBILITY_NOT_ELIGIBLE,
            ),
        )
    try:
        context = build_extraction_context(request)
    except UnsafeKnowledgeContentError as exc:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.CONTEXT_UNSAFE,
                findings=(exc.finding,),
            ),
        )
    except (TypeError, ValueError):
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.CONTEXT_UNSAFE,
            ),
        )
    try:
        raw = extractor.extract(context)
    except Exception:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.EXTRACTOR_ERROR,
                extractor_attempt_count=1,
                context_digest=context.structural_digest,
            ),
        )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) > MAX_PROPOSALS:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.MALFORMED_RESPONSE,
                extractor_attempt_count=1,
                context_digest=context.structural_digest,
            ),
        )
    if not raw:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.NO_PROPOSALS,
                extractor_attempt_count=1,
                context_digest=context.structural_digest,
            ),
        )
    if any(not isinstance(item, KnowledgeProposal) for item in raw):
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.MALFORMED_RESPONSE,
                proposal_count=len(raw),
                extractor_attempt_count=1,
                context_digest=context.structural_digest,
            ),
        )

    proposal_types = tuple(
        item.candidate_type
        if item.candidate_type in {candidate_type.value for candidate_type in CandidateType}
        else "unsupported"
        for item in raw
    )

    rejected: list[RejectedProposal] = []
    canonical: list[tuple[int, CanonicalProposal]] = []
    for index, proposal in enumerate(raw):
        validation = validate_and_canonicalize(proposal)
        if validation.proposal is None:
            rejected.append(
                RejectedProposal(
                    index,
                    validation.reason or ProposalRejectionReason.MALFORMED_PROPOSAL,
                    validation.sanitization_findings,
                )
            )
        else:
            canonical.append((index, validation.proposal))
    if not canonical:
        return _persist_result(
            store,
            _result(
                request,
                extractor,
                status=ExtractionStatus.FAILED,
                failure_reason=ExtractionFailureReason.NO_ACCEPTED_PROPOSALS,
                proposal_count=len(raw),
                proposal_types=proposal_types,
                rejected=tuple(rejected),
                extractor_attempt_count=1,
                context_digest=context.structural_digest,
            ),
        )

    existing = list(store.list_candidates(repository_scope_id=context.repository_scope_id))
    pending: list[tuple[int, KnowledgeCandidate, CandidateComparison, bool]] = []
    outcomes: list[CandidateExtractionOutcome] = []
    for proposal_index, proposal in canonical:
        candidate = _construct_candidate(
            context,
            request.eligibility_input,
            proposal,
            created_at=created_at,
            created_by=created_by,
        )
        comparison = CandidateIndex(tuple(existing), top_k=top_k).compare(candidate)
        if comparison.relation is CandidateRelation.EXACT_DUPLICATE:
            outcomes.append(
                CandidateExtractionOutcome(
                    proposal_index,
                    comparison.existing_candidate_ids[0],
                    comparison.relation,
                    comparison.existing_candidate_ids,
                    "reused",
                    comparison.compared_candidate_count,
                    proposal.portable_hint,
                )
            )
            continue
        persisted = store.read_candidate(candidate.candidate_id)
        if persisted is not None and persisted != candidate:
            return _persist_result(
                store,
                _result(
                    request,
                    extractor,
                    status=ExtractionStatus.FAILED,
                    failure_reason=ExtractionFailureReason.STORE_CONFLICT,
                    proposal_count=len(raw),
                    proposal_types=proposal_types,
                    rejected=tuple(rejected),
                    extractor_attempt_count=1,
                    context_digest=context.structural_digest,
                ),
            )
        pending.append((proposal_index, candidate, comparison, proposal.portable_hint))
        existing.append(candidate)

    for proposal_index, candidate, comparison, portable_hint in pending:
        write_status = store.write_candidate(candidate)
        if write_status is ImmutableWriteStatus.CONFLICT:
            raise KnowledgeStoreError("candidate changed during immutable extraction write")
        outcomes.append(
            CandidateExtractionOutcome(
                proposal_index,
                candidate.candidate_id,
                comparison.relation,
                comparison.existing_candidate_ids,
                write_status.value,
                comparison.compared_candidate_count,
                portable_hint,
            )
        )
    outcomes.sort(key=lambda item: item.proposal_index)
    return _persist_result(
        store,
        _result(
            request,
            extractor,
            status=ExtractionStatus.COMPLETED,
            failure_reason=None,
            proposal_count=len(raw),
            proposal_types=proposal_types,
            rejected=tuple(rejected),
            outcomes=tuple(outcomes),
            extractor_attempt_count=1,
            context_digest=context.structural_digest,
        ),
    )


__all__ = [
    "CandidateExtractionOutcome",
    "ExtractionFailureReason",
    "ExtractionStatus",
    "KnowledgeExtractionContext",
    "KnowledgeExtractionRequest",
    "KnowledgeExtractionResult",
    "KnowledgeExtractor",
    "RejectedProposal",
    "TaskSuccessReference",
    "ToolExecutionSummary",
    "build_extraction_context",
    "extract_candidates",
]
