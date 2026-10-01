from __future__ import annotations

from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeUsageMode,
    KnowledgeUsageReceipt,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
    associate_usage_outcome,
    persist_usage,
)
from tests._knowledge_runtime_helpers import NOW, active_materialized_candidate, make_scope


def _usage(store, candidate, materialization, suffix, turn="turn-consumer", mode=KnowledgeUsageMode.INJECTED):
    receipt = KnowledgeUsageReceipt.create(
        usage_id=f"usage-{suffix}",
        turn_id=turn,
        repository_scope_id=candidate.repository_scope_id,
        candidate_id=candidate.candidate_id,
        candidate_manifest_digest=candidate.manifest_digest,
        materialization_digest=materialization.result_digest,
        knowledge_type=candidate.candidate_type,
        lifecycle_state="active",
        applicability_id=f"app-{suffix}",
        applicability_digest="a" * 64,
        retrieval_id=f"retrieval-{suffix}",
        retrieval_rank=1,
        retrieval_score=1.0,
        usage_mode=mode,
        context_identity="context:test",
        created_at=NOW,
    )
    persist_usage(store, receipt)
    return receipt


def test_usage_modes_turns_and_candidates_remain_distinct(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    first, first_mat, _ = active_materialized_candidate(
        store, scope, "fact-one", CandidateType.MEMORY_FACT, title="Fact one", content="Fact one."
    )
    second, second_mat, _ = active_materialized_candidate(
        store, scope, "fact-two", CandidateType.MEMORY_FACT, title="Fact two", content="Fact two."
    )
    injected = _usage(store, first, first_mat, "1")
    referenced = _usage(store, second, second_mat, "2", mode=KnowledgeUsageMode.REFERENCED)
    other_turn = _usage(store, first, first_mat, "3", turn="turn-other")
    assert store.read_usage(injected.usage_id) == injected
    assert injected.usage_mode is KnowledgeUsageMode.INJECTED
    assert referenced.usage_mode is KnowledgeUsageMode.REFERENCED
    assert other_turn.turn_id != injected.turn_id
    assert len({injected.usage_digest, referenced.usage_digest, other_turn.usage_digest}) == 3


@pytest.mark.parametrize("status", list(TaskSuccessStatus))
def test_outcome_association_records_status_without_mutating_usage(tmp_path: Path, status) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    candidate, materialization, _ = active_materialized_candidate(
        store, scope, "experience", CandidateType.EXPERIENCE, title="Lesson", content="Use the lesson."
    )
    usage = _usage(store, candidate, materialization, status.value)
    before = usage.to_dict()
    success = TaskSuccessEvidence.create(
        evidence_id=f"outcome-{status.value}",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id=usage.turn_id,
        status=status,
        producer_id="sealed",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest="c" * 64,
        result_digest="d" * 64,
        provenance_refs=("verification:independent",),
        created_at=NOW,
    )
    store.write_task_success(success)
    association = associate_usage_outcome(
        store,
        association_id=f"association-{status.value}",
        usage_ids=(usage.usage_id,),
        task_success=success,
        created_at=NOW,
    )
    assert association.task_success_status is status
    assert store.read_usage(usage.usage_id).to_dict() == before


def test_outcome_association_rejects_mismatched_turn(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    candidate, materialization, _ = active_materialized_candidate(
        store, scope, "fact", CandidateType.MEMORY_FACT, title="Fact", content="Fact."
    )
    usage = _usage(store, candidate, materialization, "mismatch")
    success = TaskSuccessEvidence.create(
        evidence_id="wrong-turn",
        source_kind=TaskSuccessSource.SEALED_VERIFIER,
        source_turn_id="different-turn",
        status=TaskSuccessStatus.FAIL,
        producer_id="sealed",
        producer_version="1",
        repository_scope_id=scope.repository_scope_id,
        target_state_digest=None,
        result_digest="e" * 64,
        provenance_refs=("verification:independent",),
        created_at=NOW,
    )
    store.write_task_success(success)
    with pytest.raises(ValueError, match="Turn"):
        associate_usage_outcome(
            store,
            association_id="association-wrong",
            usage_ids=(usage.usage_id,),
            task_success=success,
        )

