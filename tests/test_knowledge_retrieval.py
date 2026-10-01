from __future__ import annotations

from pathlib import Path

from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeRetriever,
    SuppressionReason,
)
from tests._knowledge_runtime_helpers import NOW, active_materialized_candidate, file_guard, make_scope


def test_scoped_fact_and_experience_retrieval_is_bounded_and_deterministic(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    guard = file_guard(workspace, "project.lock", b"locked")
    active_materialized_candidate(
        store,
        scope,
        "fact-lock",
        CandidateType.MEMORY_FACT,
        title="Project lock file",
        content="The project uses a checked lock file.",
        fingerprints=(guard,),
    )
    active_materialized_candidate(
        store,
        scope,
        "experience-lock",
        CandidateType.EXPERIENCE,
        title="Verify lock file",
        content="Verify the project lock file before dependency changes.",
        fingerprints=(guard,),
    )
    active_materialized_candidate(
        store,
        scope,
        "experience-unrelated",
        CandidateType.EXPERIENCE,
        title="Database recovery",
        content="Recover a database replica.",
        fingerprints=(guard,),
    )
    retriever = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        max_candidates=2,
    )
    first, receipt = retriever.retrieve(
        "verify project lock file",
        retrieval_id="retrieval-1",
        turn_id="turn-1",
        candidate_types=(CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
        created_at=NOW,
        latency_ms=1.0,
    )
    second, _ = retriever.retrieve(
        "verify project lock file",
        retrieval_id="retrieval-2",
        turn_id="turn-2",
        candidate_types=(CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
        created_at=NOW,
        latency_ms=1.0,
    )
    assert [item.candidate.candidate_id for item in first] == [
        item.candidate.candidate_id for item in second
    ]
    assert len(first) == 2
    assert ("experience-unrelated", SuppressionReason.IRRELEVANT) in receipt.suppressed


def test_cross_repository_and_stale_candidates_are_excluded(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    store = KnowledgeRecordStore(tmp_path / "state")
    current = make_scope(store, "current")
    other = make_scope(store, "other")
    guard = file_guard(workspace, "config.txt", b"current")
    active_materialized_candidate(
        store,
        other,
        "fact-other",
        CandidateType.MEMORY_FACT,
        title="Current config",
        content="Current config is enabled.",
        fingerprints=(guard,),
    )
    active_materialized_candidate(
        store,
        current,
        "experience-stale",
        CandidateType.EXPERIENCE,
        title="Current config",
        content="Check current config before release.",
        fingerprints=(guard,),
    )
    (workspace / "config.txt").write_bytes(b"changed")
    items, receipt = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(current.repository_scope_id, workspace),
    ).retrieve(
        "current config",
        retrieval_id="retrieval-suppressed",
        turn_id="turn-1",
        candidate_types=(CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
        created_at=NOW,
        latency_ms=1.0,
    )
    assert items == ()
    assert ("fact-other", SuppressionReason.WRONG_SCOPE) in receipt.suppressed
    assert ("experience-stale", SuppressionReason.STALE) in receipt.suppressed

