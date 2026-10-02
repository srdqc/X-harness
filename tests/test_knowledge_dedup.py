from __future__ import annotations

from pico.knowledge_evolution import (
    CandidateIndex,
    CandidateRelation,
    CandidateType,
    ContentClass,
    KnowledgeCandidate,
    KnowledgeRecordStore,
    SourceTurnReference,
)

_SCOPE_A = "a" * 64
_SCOPE_B = "b" * 64


def _candidate(
    candidate_id: str,
    *,
    scope: str = _SCOPE_A,
    candidate_type: CandidateType = CandidateType.EXPERIENCE,
    content_class: ContentClass = ContentClass.STRATEGY,
    title: str = "Verify repository changes",
    content: str = "Run deterministic tests and inspect the independent verifier result.",
    preconditions: tuple[str, ...] = ("Verifier is available",),
    fingerprints: tuple[tuple[str, str], ...] = (),
) -> KnowledgeCandidate:
    return KnowledgeCandidate.create(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        content_class=content_class,
        qualifying_sources=(
            SourceTurnReference(
                turn_id="turn-1",
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=("success-1",),
                task_success_evidence_digests=("e" * 64,),
            ),
        ),
        repository_scope_id=scope,
        title=title,
        reusable_content=content,
        preconditions=preconditions,
        applicability_fingerprints=fingerprints,
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        provenance_refs=("turn:turn-1",),
        created_at="2026-10-01T00:00:00Z",
        created_by="test",
    )


def test_exact_duplicate_uses_scoped_type_and_content_fingerprint() -> None:
    existing = _candidate("candidate-existing")
    duplicate = _candidate("candidate-proposed")
    result = CandidateIndex((existing,)).compare(duplicate)
    assert result.relation is CandidateRelation.EXACT_DUPLICATE
    assert result.existing_candidate_ids == (existing.candidate_id,)

    other_type = _candidate(
        "candidate-skill",
        candidate_type=CandidateType.SKILL_CANDIDATE,
        content_class=ContentClass.PROCEDURE,
    )
    assert CandidateIndex((existing,)).compare(other_type).relation is CandidateRelation.NEW


def test_different_title_is_not_silently_exact() -> None:
    existing = _candidate("candidate-existing")
    changed = _candidate("candidate-changed", title="A different canonical title")
    result = CandidateIndex((existing,)).compare(changed)
    assert result.relation is not CandidateRelation.EXACT_DUPLICATE


def test_lexical_near_duplicate_and_unrelated_candidate() -> None:
    existing = _candidate("candidate-existing")
    near = _candidate(
        "candidate-near",
        title="Verify changes deterministically",
        content="Run deterministic tests, then inspect the verifier result independently.",
    )
    unrelated = _candidate(
        "candidate-unrelated",
        title="Format release notes",
        content="Group user-visible release notes by audience and release category.",
        preconditions=("A release milestone exists",),
    )
    assert CandidateIndex((existing,)).compare(near).relation is CandidateRelation.NEAR_DUPLICATE
    assert CandidateIndex((existing,)).compare(unrelated).relation is CandidateRelation.NEW
    same_title_unrelated = _candidate(
        "candidate-same-title",
        title=existing.title,
        content="Group release notes by audience and release category.",
        preconditions=("A release milestone exists",),
    )
    assert CandidateIndex((existing,)).compare(same_title_unrelated).relation is CandidateRelation.NEW


def test_search_is_top_k_bounded_and_repository_scoped() -> None:
    local = tuple(
        _candidate(
            f"candidate-{index}",
            title=f"Verification strategy {index}",
            content=f"Run deterministic verification checks for module {index}.",
        )
        for index in range(6)
    )
    foreign_exact = _candidate("candidate-foreign", scope=_SCOPE_B)
    query = _candidate("candidate-query")
    result = CandidateIndex((*local, foreign_exact), top_k=2).compare(query)
    assert result.compared_candidate_count <= 2
    assert foreign_exact.candidate_id not in result.existing_candidate_ids


def test_rebuildable_index_reproduces_lookup(tmp_path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    existing = _candidate("candidate-existing")
    store.write_candidate(existing)
    query = _candidate("candidate-query")
    direct = CandidateIndex((existing,)).compare(query)
    rebuilt = CandidateIndex.rebuild(
        store, repository_scope_id=_SCOPE_A
    ).compare(query)
    assert rebuilt == direct


def test_index_has_no_embedding_or_vector_dependency() -> None:
    import pico.knowledge_evolution.index as index_module

    source = open(index_module.__file__, encoding="utf-8").read()
    assert "embedding" not in source.lower()
    assert "vector" not in source.lower()
