from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pico.knowledge_evolution import (
    CandidateType,
    ContentClass,
    KnowledgeCandidate,
    SourceTurnReference,
)

_SCOPE = "a" * 64


def _source() -> SourceTurnReference:
    return SourceTurnReference(
        turn_id="turn-1",
        replay_digest="b" * 64,
        verification_digest="c" * 64,
        task_success_evidence_ids=("success-1",),
        task_success_evidence_digests=("d" * 64,),
        evidence_refs=("evidence:1",),
    )


def _candidate(**overrides) -> KnowledgeCandidate:
    values = {
        "candidate_id": "candidate-1",
        "candidate_type": CandidateType.EXPERIENCE,
        "content_class": ContentClass.STRATEGY,
        "qualifying_sources": (_source(),),
        "supporting_source_turn_ids": ("turn-failed",),
        "repository_scope_id": _SCOPE,
        "title": "  Validate before promotion ",
        "reusable_content": "Check evidence.  \r\nThen promote only after validation.  ",
        "preconditions": ("Repository scope matches",),
        "applicability_fingerprints": (("config", "e" * 64),),
        "extraction_policy": "p3.manual-candidate",
        "extraction_policy_version": 1,
        "provenance_refs": ("turn:turn-1",),
        "created_at": "2026-10-01T00:00:00Z",
        "created_by": "test-fixture",
    }
    values.update(overrides)
    return KnowledgeCandidate.create(**values)


@pytest.mark.parametrize(
    ("candidate_type", "content_class"),
    [
        (CandidateType.MEMORY_FACT, ContentClass.FACT),
        (CandidateType.EXPERIENCE, ContentClass.RECOVERY),
        (CandidateType.SKILL_CANDIDATE, ContentClass.PROCEDURE),
    ],
)
def test_supported_candidate_types_round_trip(candidate_type, content_class) -> None:
    candidate = _candidate(candidate_type=candidate_type, content_class=content_class)
    assert KnowledgeCandidate.from_dict(candidate.to_dict()) == candidate
    assert candidate.repository_scope_id == _SCOPE
    assert candidate.qualifying_sources[0].turn_id == "turn-1"
    assert candidate.qualifying_sources[0].task_success_evidence_ids == ("success-1",)


def test_tool_candidate_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        _candidate(candidate_type="tool")


def test_candidate_is_immutable_and_has_no_lifecycle_state() -> None:
    candidate = _candidate()
    with pytest.raises(FrozenInstanceError):
        candidate.title = "changed"  # type: ignore[misc]
    assert set(candidate.to_dict()).isdisjoint(
        {"active", "rejected", "pending_review", "deprecated", "lifecycle_state"}
    )


def test_content_fingerprint_ignores_creation_metadata_but_manifest_does_not() -> None:
    first = _candidate()
    later = _candidate(candidate_id="candidate-2", created_at="2026-10-02T00:00:00Z")
    assert first.content_fingerprint == later.content_fingerprint
    assert first.manifest_digest != later.manifest_digest


def test_canonical_serialization_and_digests_are_stable() -> None:
    first = _candidate()
    second = _candidate()
    assert first == second
    assert first.title == "Validate before promotion"
    assert first.reusable_content == "Check evidence.\nThen promote only after validation."
    assert KnowledgeCandidate.from_dict(first.to_dict()).manifest_digest == first.manifest_digest
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        KnowledgeCandidate.from_dict(replace(first, manifest_digest="f" * 64).to_dict())


def test_candidate_contract_requires_no_session_transcript_or_tool_payload() -> None:
    keys = set(_candidate().to_dict())
    assert keys.isdisjoint(
        {"session_transcript", "messages", "tool_arguments", "tool_results", "provider_payload"}
    )

