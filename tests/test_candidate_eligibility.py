from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    SUPPORTED_POLICY,
    CandidateEligibilityInput,
    CandidateType,
    EligibilityReason,
    EligibilityStatus,
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    RepositoryScopeReason,
    RepositoryScopeResolution,
    RepositoryScopeStatus,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    evaluate_candidate_eligibility,
)
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore

_STATE = "e" * 64


def _p1c(tmp_path: Path, turn_id: str = "turn-eligible"):
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
    return replay_result, verifier.verify_replay(replay_result)


def _scope() -> RepositoryScopeResolution:
    identity = RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:test",
        evidence=(("project_key", "test"),),
    )
    return RepositoryScopeResolution(
        RepositoryScopeStatus.RESOLVED,
        RepositoryScopeReason.EXPLICIT_PROJECT_KEY,
        identity,
    )


def _success(scope_id: str, **overrides) -> TaskSuccessEvidence:
    values = {
        "evidence_id": "success-1",
        "source_kind": TaskSuccessSource.SEALED_VERIFIER,
        "source_turn_id": "turn-eligible",
        "status": TaskSuccessStatus.PASS,
        "producer_id": "sealed",
        "producer_version": "1",
        "repository_scope_id": scope_id,
        "target_state_digest": _STATE,
        "result_digest": "f" * 64,
        "provenance_refs": ("result:1",),
        "created_at": "2026-10-01T00:00:00Z",
    }
    values.update(overrides)
    return TaskSuccessEvidence.create(**values)


def _input(tmp_path: Path, **overrides) -> CandidateEligibilityInput:
    replay_result, verification = _p1c(tmp_path)
    scope = _scope()
    values = {
        "turn_id": "turn-eligible",
        "replay": replay_result,
        "verification": verification,
        "task_success": (_success(scope.identity.repository_scope_id),),
        "scope": scope,
        "resulting_state_digest": _STATE,
        "extraction_policy": SUPPORTED_POLICY[0],
        "extraction_policy_version": SUPPORTED_POLICY[1],
        "candidate_type": CandidateType.EXPERIENCE,
        "provenance_turn_ids": ("turn-eligible",),
    }
    values.update(overrides)
    return CandidateEligibilityInput(**values)


def test_complete_independently_proven_evidence_is_eligible_and_deterministic(tmp_path: Path) -> None:
    value = _input(tmp_path)
    first = evaluate_candidate_eligibility(value)
    second = evaluate_candidate_eligibility(value)
    assert first.status is EligibilityStatus.ELIGIBLE
    assert first.reasons == ()
    assert first.result_digest == second.result_digest


@pytest.mark.parametrize(
    ("status", "expected_status", "reason"),
    [
        (TaskSuccessStatus.FAIL, EligibilityStatus.INELIGIBLE, EligibilityReason.TASK_SUCCESS_FAILED),
        (
            TaskSuccessStatus.INCONCLUSIVE,
            EligibilityStatus.INCONCLUSIVE,
            EligibilityReason.TASK_SUCCESS_INCONCLUSIVE,
        ),
    ],
)
def test_task_success_status_aggregates_conservatively(tmp_path: Path, status, expected_status, reason) -> None:
    value = _input(tmp_path)
    changed = replace(
        value,
        task_success=(_success(value.scope.identity.repository_scope_id, status=status),),
    )
    result = evaluate_candidate_eligibility(changed)
    assert result.status is expected_status
    assert reason in result.reasons


def test_missing_task_success_or_scope_is_inconclusive(tmp_path: Path) -> None:
    value = _input(tmp_path)
    missing_success = evaluate_candidate_eligibility(replace(value, task_success=()))
    missing_scope = evaluate_candidate_eligibility(
        replace(
            value,
            scope=RepositoryScopeResolution(
                RepositoryScopeStatus.UNAVAILABLE,
                RepositoryScopeReason.NON_GIT_REQUIRES_PROJECT_KEY,
            ),
        )
    )
    assert EligibilityReason.TASK_SUCCESS_MISSING in missing_success.reasons
    assert missing_success.status is EligibilityStatus.INCONCLUSIVE
    assert EligibilityReason.REPOSITORY_SCOPE_MISSING in missing_scope.reasons
    assert missing_scope.status is EligibilityStatus.INCONCLUSIVE


def test_missing_state_binding_and_turn_id_are_inconclusive(tmp_path: Path) -> None:
    value = _input(tmp_path)
    scope_id = value.scope.identity.repository_scope_id
    missing_binding = evaluate_candidate_eligibility(
        replace(
            value,
            task_success=(_success(scope_id, target_state_digest=None),),
        )
    )
    missing_turn = evaluate_candidate_eligibility(
        replace(value, turn_id="", provenance_turn_ids=("",))
    )
    assert missing_binding.status is EligibilityStatus.INCONCLUSIVE
    assert EligibilityReason.TASK_SUCCESS_STATE_MISMATCH in missing_binding.reasons
    assert missing_turn.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.MISSING_TURN_ID in missing_turn.reasons
    assert EligibilityReason.PROVENANCE_MISMATCH in missing_turn.reasons


def test_replay_and_verification_digest_mismatch_are_ineligible(tmp_path: Path) -> None:
    value = _input(tmp_path)
    bad_replay = evaluate_candidate_eligibility(
        replace(value, replay=replace(value.replay, replay_digest="0" * 64))
    )
    bad_verification = evaluate_candidate_eligibility(
        replace(value, verification=replace(value.verification, verification_digest="0" * 64))
    )
    assert bad_replay.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.REPLAY_DIGEST_INVALID in bad_replay.reasons
    assert bad_verification.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.VERIFICATION_DIGEST_INVALID in bad_verification.reasons


def test_corrupt_task_success_digest_is_ineligible(tmp_path: Path) -> None:
    value = _input(tmp_path)
    corrupt = replace(value.task_success[0], evidence_digest="0" * 64)
    result = evaluate_candidate_eligibility(replace(value, task_success=(corrupt,)))
    assert result.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.TASK_SUCCESS_DIGEST_INVALID in result.reasons


def test_structural_fail_inconclusive_and_partial_evidence(tmp_path: Path) -> None:
    value = _input(tmp_path)
    failed = replace(value.verification, overall_status=verifier.VerificationStatus.FAIL)
    failed = replace(failed, verification_digest=verifier.compute_verification_digest(failed))
    fail_result = evaluate_candidate_eligibility(replace(value, verification=failed))

    partial_replay = replace(
        value.replay,
        evidence_status=evidence.EvidenceCompleteness.PARTIAL,
        replay_digest="",
    )
    partial_replay = replace(partial_replay, replay_digest=replay.compute_replay_digest(partial_replay))
    inconclusive_verification = verifier.verify_replay(partial_replay)
    partial_result = evaluate_candidate_eligibility(
        replace(value, replay=partial_replay, verification=inconclusive_verification)
    )
    assert fail_result.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.STRUCTURAL_VERIFICATION_FAILED in fail_result.reasons
    assert partial_result.status is EligibilityStatus.INCONCLUSIVE
    assert EligibilityReason.SOURCE_EVIDENCE_INCOMPLETE in partial_result.reasons


def test_turn_state_scope_and_provenance_mismatch_are_ineligible(tmp_path: Path) -> None:
    value = _input(tmp_path)
    scope_id = value.scope.identity.repository_scope_id
    cases = (
        replace(value, task_success=(_success(scope_id, source_turn_id="other"),)),
        replace(value, task_success=(_success(scope_id, target_state_digest="1" * 64),)),
        replace(value, provenance_turn_ids=("other",)),
        replace(
            value,
            scope=RepositoryScopeResolution(
                RepositoryScopeStatus.INVALID,
                RepositoryScopeReason.LOCAL_BINDING_INVALID,
            ),
        ),
    )
    expected = (
        EligibilityReason.TASK_SUCCESS_TURN_MISMATCH,
        EligibilityReason.TASK_SUCCESS_STATE_MISMATCH,
        EligibilityReason.PROVENANCE_MISMATCH,
        EligibilityReason.REPOSITORY_SCOPE_INVALID,
    )
    for item, reason in zip(cases, expected, strict=True):
        result = evaluate_candidate_eligibility(item)
        assert result.status is EligibilityStatus.INELIGIBLE
        assert reason in result.reasons


def test_unsupported_schema_policy_and_conflicting_success_fail_are_ineligible(tmp_path: Path) -> None:
    value = _input(tmp_path)
    scope_id = value.scope.identity.repository_scope_id
    conflict = replace(
        value,
        task_success=(
            _success(scope_id),
            _success(scope_id, evidence_id="failure", status=TaskSuccessStatus.FAIL),
        ),
    )
    assert evaluate_candidate_eligibility(replace(value, schema_version=99)).status is EligibilityStatus.INELIGIBLE
    assert evaluate_candidate_eligibility(replace(value, extraction_policy_version=99)).status is EligibilityStatus.INELIGIBLE
    result = evaluate_candidate_eligibility(conflict)
    assert result.status is EligibilityStatus.INELIGIBLE
    assert EligibilityReason.TASK_SUCCESS_FAILED in result.reasons


def test_agent_or_delivery_text_cannot_change_eligibility_input(tmp_path: Path) -> None:
    value = _input(tmp_path)
    assert not hasattr(value, "agent_answer")
    assert not hasattr(value, "delivery_success")
    assert not hasattr(value, "runtime_success")
