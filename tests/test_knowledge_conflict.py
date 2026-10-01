from __future__ import annotations

from pico.knowledge_evolution import (
    CandidateIndex,
    CandidateRelation,
    CandidateType,
    ContentClass,
    KnowledgeCandidate,
    SourceTurnReference,
)
from pico.knowledge_evolution.types import structural_digest


def _candidate(
    candidate_id: str,
    *,
    scope: str = "a" * 64,
    candidate_type: CandidateType | str = CandidateType.EXPERIENCE,
    content_class: ContentClass | str = ContentClass.STRATEGY,
    title: str = "Verify repository changes",
    content: str = "Run deterministic tests and inspect the verifier result.",
    preconditions: tuple[str, ...] = ("Verifier is available",),
    fingerprints: tuple[tuple[str, str], ...] = (),
) -> KnowledgeCandidate:
    return KnowledgeCandidate.create(
        candidate_id=candidate_id,
        candidate_type=CandidateType(candidate_type),
        content_class=ContentClass(content_class),
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


def _fact(candidate_id: str, value: str, *, content: str):
    return _candidate(
        candidate_id,
        candidate_type="memory_fact",
        content_class="fact",
        title="Repository package manager",
        content=content,
        preconditions=("Repository scope matches",),
        fingerprints=(("fact:package manager", structural_digest({"value": value})),),
    )


def test_obvious_scoped_fact_contradiction_is_conflict() -> None:
    existing = _fact("candidate-npm", "npm", content="The repository uses npm.")
    proposed = _fact("candidate-uv", "uv", content="The repository uses uv.")
    result = CandidateIndex((existing,)).compare(proposed)
    assert result.relation is CandidateRelation.CONFLICT
    assert result.existing_candidate_ids == (existing.candidate_id,)


def test_compatible_refinement_and_ambiguous_case_are_not_false_conflicts() -> None:
    existing = _fact("candidate-npm", "npm", content="The repository uses npm.")
    refinement = _fact(
        "candidate-npm-refined",
        "npm",
        content="The repository uses npm with deterministic lockfile checks.",
    )
    ambiguous = _candidate(
        "candidate-ambiguous",
        title="Package validation",
        content="Validate package metadata before release.",
    )
    assert CandidateIndex((existing,)).compare(refinement).relation is not CandidateRelation.CONFLICT
    assert CandidateIndex((existing,)).compare(ambiguous).relation is not CandidateRelation.CONFLICT


def test_conflict_does_not_overwrite_or_supersede_existing_candidate(tmp_path) -> None:
    from pico.knowledge_evolution import KnowledgeRecordStore

    store = KnowledgeRecordStore(tmp_path)
    existing = _fact("candidate-npm", "npm", content="The repository uses npm.")
    proposed = _fact("candidate-uv", "uv", content="The repository uses uv.")
    store.write_candidate(existing)
    relation = CandidateIndex.rebuild(
        store, repository_scope_id=existing.repository_scope_id
    ).compare(proposed)
    store.write_candidate(proposed)
    assert relation.relation is CandidateRelation.CONFLICT
    assert store.read_candidate(existing.candidate_id) == existing
    assert not hasattr(proposed, "supersedes")


def test_cross_repository_portability_cannot_bypass_scope() -> None:
    existing = _fact("candidate-a", "npm", content="The repository uses npm.")
    foreign = _candidate(
        "candidate-b",
        scope="b" * 64,
        candidate_type="memory_fact",
        content_class="fact",
        title=existing.title,
        content=existing.reusable_content,
        preconditions=existing.preconditions,
        fingerprints=existing.applicability_fingerprints,
    )
    assert CandidateIndex((existing,)).compare(foreign).relation is CandidateRelation.NEW
