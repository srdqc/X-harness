from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    SUPPORTED_POLICY,
    CandidateEligibilityInput,
    CandidateType,
    ContentClass,
    EligibilityStatus,
    ExtractionFailureReason,
    ExtractionStatus,
    KnowledgeExtractionRequest,
    KnowledgeProposal,
    KnowledgeRecordStore,
    ProposalRejectionReason,
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    RepositoryScopeReason,
    RepositoryScopeResolution,
    RepositoryScopeStatus,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    build_extraction_context,
    evaluate_candidate_eligibility,
    extract_candidates,
)
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore

_STATE = "e" * 64


class FakeExtractor:
    extractor_id = "deterministic-fake"
    extractor_version = "1"

    def __init__(self, proposals=(), *, error: Exception | None = None) -> None:
        self.proposals = proposals
        self.error = error
        self.calls = []

    def extract(self, context):
        self.calls.append(context)
        if self.error:
            raise self.error
        return self.proposals


def _proposal(candidate_type=CandidateType.EXPERIENCE, **overrides) -> KnowledgeProposal:
    values = {
        "candidate_type": candidate_type.value if isinstance(candidate_type, CandidateType) else candidate_type,
        "content_class": ContentClass.STRATEGY.value,
        "title": "Prefer deterministic verification",
        "reusable_content": "Verify the resulting state independently before accepting success.",
        "preconditions": ("Independent verifier is available",),
        "applicability": ("Repository scope matches",),
        "evidence_refs": ("verification:core",),
    }
    values.update(overrides)
    return KnowledgeProposal(**values)


def _eligible(tmp_path: Path, *, status: TaskSuccessStatus = TaskSuccessStatus.PASS):
    turn_id = "turn-extract"
    store = TraceStore(tmp_path / "trace")
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=store.append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )
    replay_result = replay.replay_turn(tmp_path / "trace", turn_id)
    verification = verifier.verify_replay(replay_result)
    scope = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:extraction-test",
        evidence=(("project_key", "extraction-test"),),
    )
    success = TaskSuccessEvidence.create(
        evidence_id="success-1",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id=turn_id,
        status=status,
        producer_id="sealed-verifier",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest=_STATE,
        result_digest="f" * 64,
        provenance_refs=("artifact:verified",),
        created_at="2026-10-01T00:00:00Z",
    )
    eligibility_input = CandidateEligibilityInput(
        turn_id=turn_id,
        replay=replay_result,
        verification=verification,
        task_success=(success,),
        scope=RepositoryScopeResolution(
            RepositoryScopeStatus.RESOLVED,
            RepositoryScopeReason.EXPLICIT_PROJECT_KEY,
            scope,
        ),
        resulting_state_digest=_STATE,
        extraction_policy=SUPPORTED_POLICY[0],
        extraction_policy_version=SUPPORTED_POLICY[1],
        candidate_type=CandidateType.EXPERIENCE,
        provenance_turn_ids=(turn_id,),
    )
    return eligibility_input, evaluate_candidate_eligibility(eligibility_input)


def _request(tmp_path: Path, extraction_id: str = "extraction-1", **kwargs):
    eligibility_input, result = _eligible(tmp_path)
    return KnowledgeExtractionRequest(
        extraction_id,
        eligibility_input,
        result,
        **kwargs,
    )


def _run(request, extractor, store):
    return extract_candidates(
        request,
        extractor,
        store,
        created_at="2026-10-01T00:00:00Z",
        created_by="extractor:test",
    )


def test_eligible_invokes_extractor_and_binds_trusted_scope(tmp_path: Path) -> None:
    request = _request(tmp_path)
    extractor = FakeExtractor((_proposal(),))
    store = KnowledgeRecordStore(tmp_path / "state")
    result = _run(request, extractor, store)
    assert result.status is ExtractionStatus.COMPLETED
    assert len(extractor.calls) == 1
    assert result.extractor_attempt_count == 1
    assert result.context_digest == extractor.calls[0].structural_digest
    assert result.repository_scope_id == request.eligibility_result.repository_scope_id
    candidate = store.read_candidate(result.candidate_outcomes[0].candidate_id)
    assert candidate.repository_scope_id == request.eligibility_result.repository_scope_id
    assert candidate.qualifying_sources[0].turn_id == request.eligibility_input.turn_id
    assert "verification:core" not in candidate.provenance_refs
    assert not hasattr(extractor.proposals[0], "candidate_id")
    assert not hasattr(extractor.proposals[0], "repository_scope_id")
    assert not hasattr(extractor.proposals[0], "active")


@pytest.mark.parametrize("status", [TaskSuccessStatus.FAIL, TaskSuccessStatus.INCONCLUSIVE])
def test_noneligible_input_never_invokes_extractor(tmp_path: Path, status) -> None:
    eligibility_input, eligibility_result = _eligible(tmp_path, status=status)
    extractor = FakeExtractor((_proposal(),))
    result = _run(
        KnowledgeExtractionRequest(f"extraction-{status.value}", eligibility_input, eligibility_result),
        extractor,
        KnowledgeRecordStore(tmp_path / "state"),
    )
    assert eligibility_result.status is not EligibilityStatus.ELIGIBLE
    assert result.status is ExtractionStatus.NOT_PERMITTED
    assert result.failure_reason is ExtractionFailureReason.ELIGIBILITY_NOT_ELIGIBLE
    assert result.extractor_attempt_count == 0
    assert extractor.calls == []


def test_extractor_cannot_override_supplied_eligibility(tmp_path: Path) -> None:
    request = _request(tmp_path)
    forged = replace(request.eligibility_result, status=EligibilityStatus.ELIGIBLE)
    ineligible_input = replace(request.eligibility_input, task_success=())
    extractor = FakeExtractor((_proposal(),))
    result = _run(
        replace(request, eligibility_input=ineligible_input, eligibility_result=forged),
        extractor,
        KnowledgeRecordStore(tmp_path / "state"),
    )
    assert result.failure_reason is ExtractionFailureReason.ELIGIBILITY_MISMATCH
    assert extractor.calls == []


def test_projection_is_bounded_and_excludes_raw_private_domains(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        task_goal="Create independently verified reusable guidance",
        recovery_facts=("Invalid arguments were corrected before success",),
    )
    context = build_extraction_context(request)
    payload = context.to_dict()
    assert context.repository_scope_id == request.eligibility_result.repository_scope_id
    assert context.task_success_refs[0].evidence_id == "success-1"
    assert len(context.tool_summaries) <= 32
    assert set(payload).isdisjoint(
        {"session_transcript", "provider_reasoning", "tool_arguments", "tool_results", "environment"}
    )


def test_unsafe_context_fails_closed_before_extractor(tmp_path: Path) -> None:
    request = _request(tmp_path, task_goal="Use api_key=never-persist-this")
    extractor = FakeExtractor((_proposal(),))
    result = _run(request, extractor, KnowledgeRecordStore(tmp_path / "state"))
    assert result.failure_reason is ExtractionFailureReason.CONTEXT_UNSAFE
    assert result.sanitization_findings == ("suspected_secret",)
    assert extractor.calls == []


def test_all_supported_proposal_types_construct_candidates(tmp_path: Path) -> None:
    proposals = (
        _proposal(
            CandidateType.MEMORY_FACT,
            content_class=ContentClass.FACT.value,
            title="Supported configuration format",
            reusable_content="The repository uses TOML configuration.",
            fact_subject="configuration format",
            fact_value="TOML",
        ),
        _proposal(),
        _proposal(
            CandidateType.SKILL_CANDIDATE,
            content_class=ContentClass.PROCEDURE.value,
            title="Verify a repository change",
            reusable_content="Run the scoped verifier and inspect its sealed result.",
            validation_expectations=("Verifier reports PASS",),
        ),
    )
    result = _run(
        _request(tmp_path),
        FakeExtractor(proposals),
        KnowledgeRecordStore(tmp_path / "state"),
    )
    assert result.status is ExtractionStatus.COMPLETED
    assert result.accepted_proposal_count == 3
    assert result.proposal_types == ("memory_fact", "experience", "skill_candidate")


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        (_proposal("tool"), ProposalRejectionReason.UNSUPPORTED_CANDIDATE_TYPE),
        (_proposal(reusable_content="x" * 32_769), ProposalRejectionReason.OVERSIZED_CONTENT),
        (
            _proposal(reusable_content="Authorization: Bearer abcdefghijklmnop"),
            ProposalRejectionReason.UNSAFE_CONTENT,
        ),
    ],
)
def test_invalid_or_unsafe_proposals_are_rejected(tmp_path: Path, proposal, reason) -> None:
    result = _run(
        _request(tmp_path),
        FakeExtractor((proposal,)),
        KnowledgeRecordStore(tmp_path / "state"),
    )
    assert result.status is ExtractionStatus.FAILED
    assert result.rejected_proposals[0].reason is reason
    assert result.candidate_outcomes == ()


def test_malformed_response_and_extractor_failure_create_no_candidate(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    existing = _run(
        _request(tmp_path / "valid-evidence", "valid"),
        FakeExtractor((_proposal(),)),
        store,
    )
    before = store.list_candidates()
    malformed = _run(
        _request(tmp_path / "malformed-evidence", "malformed"),
        FakeExtractor(({"title": "bad"},)),
        store,
    )
    failed = _run(
        _request(tmp_path / "failed-evidence", "failed"),
        FakeExtractor(error=TimeoutError("provider timeout")),
        store,
    )
    assert malformed.failure_reason is ExtractionFailureReason.MALFORMED_RESPONSE
    assert failed.failure_reason is ExtractionFailureReason.EXTRACTOR_ERROR
    assert store.list_candidates() == before
    assert existing.candidate_outcomes[0].candidate_id == before[0].candidate_id


def test_no_proposals_is_structured_failure(tmp_path: Path) -> None:
    result = _run(
        _request(tmp_path),
        FakeExtractor(()),
        KnowledgeRecordStore(tmp_path / "state"),
    )
    assert result.failure_reason is ExtractionFailureReason.NO_PROPOSALS
    assert result.candidate_outcomes == ()


def test_recovered_trajectory_can_yield_recovery_experience(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        recovery_facts=("Schema validation failed before a corrected call succeeded",),
    )
    proposal = _proposal(
        content_class=ContentClass.RECOVERY.value,
        title="Recover from schema validation",
        reusable_content="Correct arguments against the schema, then issue a fresh call.",
    )
    result = _run(request, FakeExtractor((proposal,)), KnowledgeRecordStore(tmp_path / "state"))
    assert result.status is ExtractionStatus.COMPLETED
    assert result.accepted_proposal_count == 1


def test_portable_hint_remains_non_authoritative_and_repository_scoped(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = KnowledgeRecordStore(tmp_path / "state")
    result = _run(request, FakeExtractor((_proposal(portable_hint=True),)), store)
    outcome = result.candidate_outcomes[0]
    candidate = store.read_candidate(outcome.candidate_id)
    assert outcome.portable_hint is True
    assert candidate.repository_scope_id == request.eligibility_result.repository_scope_id
    assert not hasattr(candidate, "portable")


def test_result_digest_and_persisted_round_trip_are_deterministic(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = KnowledgeRecordStore(tmp_path / "state")
    result = _run(request, FakeExtractor((_proposal(),)), store)
    assert store.read_extraction_result(result.extraction_id) == result
    assert store.read_candidate(result.candidate_outcomes[0].candidate_id) is not None
    assert result.result_digest == type(result).from_dict(result.to_dict()).result_digest


def test_repeated_source_extraction_reuses_exact_candidate(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    first_request = _request(tmp_path, "first")
    first = _run(first_request, FakeExtractor((_proposal(),)), store)
    second_request = replace(first_request, extraction_id="second")
    second = _run(second_request, FakeExtractor((_proposal(),)), store)
    assert first.candidate_outcomes[0].candidate_id == second.candidate_outcomes[0].candidate_id
    assert second.candidate_outcomes[0].relation.value == "exact_duplicate"
    assert second.candidate_outcomes[0].write_status == "reused"
    assert len(store.list_candidates()) == 1
