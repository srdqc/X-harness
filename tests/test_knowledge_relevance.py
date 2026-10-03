from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live.selectivity import (
    evaluate_selectivity,
)
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeRetriever,
    KnowledgeSelectionMode,
    RelevanceDecision,
    RelevanceReason,
)
from tests._knowledge_runtime_helpers import (
    active_materialized_candidate,
    file_guard,
    make_scope,
)


def _candidate(store, scope, workspace, candidate_id, candidate_type, title, content):
    return active_materialized_candidate(
        store,
        scope,
        candidate_id,
        candidate_type,
        title=title,
        content=content,
        fingerprints=(file_guard(workspace, "project.cfg", b"safe"),),
    )[0]


def _retrieve(tmp_path: Path, query: str, *, limit: int = 5, mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1):
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    _candidate(
        store,
        scope,
        tmp_path,
        "fact-skill-flow",
        CandidateType.MEMORY_FACT,
        "Scoped Skill flow",
        "SkillForgeRouter keeps scoped Skills behind SkillRegistry, Router, Resolver, and context.",
    )
    _candidate(
        store,
        scope,
        tmp_path,
        "experience-guard",
        CandidateType.EXPERIENCE,
        "Fail-closed guard recovery",
        "Preserve fail-closed guard encoding and add a regression for applicability failures.",
    )
    _candidate(
        store,
        scope,
        tmp_path,
        "experience-path",
        CandidateType.EXPERIENCE,
        "Short physical paths",
        "Keep Windows experiment artifact paths short.",
    )
    _candidate(
        store,
        scope,
        tmp_path,
        "skill-test-suite",
        CandidateType.SKILL_CANDIDATE,
        "Register focused phase tests",
        "Inspect the canonical test matrix and register a deduplicated phase suite.",
    )
    retriever = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(scope.repository_scope_id, tmp_path),
        max_candidates=limit,
        selection_mode=mode,
    )
    items, receipt = retriever.retrieve(
        query,
        retrieval_id="retrieval-test",
        turn_id="turn-test",
        candidate_types=(
            CandidateType.MEMORY_FACT,
            CandidateType.EXPERIENCE,
            CandidateType.SKILL_CANDIDATE,
        ),
        created_at="2026-10-03T00:00:00Z",
    )
    return store, items, receipt


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("fix SkillForgeRouter multi-source behavior", {"fact-skill-flow"}),
        ("preserve fail-closed guard encoding", {"experience-guard"}),
        ("repair Windows experiment artifact paths", {"experience-path"}),
        ("register deduplicated canonical test matrix suite", {"skill-test-suite"}),
    ],
)
def test_selective_mode_keeps_strong_fact_and_experience_matches(
    tmp_path: Path, query: str, expected: set[str]
) -> None:
    store, items, _ = _retrieve(tmp_path, query)
    assert {item.candidate.candidate_id for item in items} == expected
    decisions = store.list_relevance_selections(turn_id="turn-test")
    assert {item.candidate_id for item in decisions if item.decision is RelevanceDecision.SELECT} == expected


@pytest.mark.parametrize("query", ["", "fix this", "update code", "unrelated weather forecast"])
def test_selective_mode_abstains_for_empty_or_low_signal_tasks(tmp_path: Path, query: str) -> None:
    store, items, _ = _retrieve(tmp_path, query)
    assert items == ()
    decisions = store.list_relevance_selections(turn_id="turn-test")
    assert decisions
    assert all(item.decision is RelevanceDecision.ABSTAIN for item in decisions)
    if not query:
        assert {item.reason for item in decisions} == {RelevanceReason.ABSTAIN_EMPTY_TASK_QUERY}


def test_selection_is_deterministic_and_receipts_round_trip_without_task_text(tmp_path: Path) -> None:
    first_store, first_items, _ = _retrieve(tmp_path / "first", "preserve fail-closed guard encoding")
    second_store, second_items, _ = _retrieve(tmp_path / "second", "preserve fail-closed guard encoding")
    assert [(item.candidate.candidate_id, item.score) for item in first_items] == [
        (item.candidate.candidate_id, item.score) for item in second_items
    ]
    first = first_store.list_relevance_selections(turn_id="turn-test")
    second = second_store.list_relevance_selections(turn_id="turn-test")
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert "preserve fail-closed" not in str(first[0].to_dict())
    assert first_store.read_relevance_selection(first[0].selection_id) == first[0]


def test_selection_limit_applies_after_relevance_and_marks_remaining_candidate(tmp_path: Path) -> None:
    store, items, _ = _retrieve(
        tmp_path, "SkillForgeRouter guard resolver fail-closed regression", limit=1
    )
    assert len(items) == 1
    decisions = store.list_relevance_selections(turn_id="turn-test")
    assert sum(item.decision is RelevanceDecision.SELECT for item in decisions) == 1


def test_irrelevant_higher_candidate_id_does_not_displace_relevant_candidate(tmp_path: Path) -> None:
    _, items, _ = _retrieve(tmp_path, "SkillForgeRouter scoped resolver")
    assert tuple(item.candidate.candidate_id for item in items) == ("fact-skill-flow",)


def test_legacy_mode_preserves_existing_positive_overlap_behavior_and_is_default(tmp_path: Path) -> None:
    _, explicit, _ = _retrieve(
        tmp_path / "explicit", "guard", mode=KnowledgeSelectionMode.LEGACY_APPLICABLE
    )
    store = KnowledgeRecordStore(tmp_path / "default" / "state")
    scope = make_scope(store)
    _candidate(
        store, scope, tmp_path, "experience-guard", CandidateType.EXPERIENCE,
        "Guard recovery", "Preserve guard behavior.",
    )
    default = KnowledgeRetriever(
        store, ApplicabilityEnvironment(scope.repository_scope_id, tmp_path)
    )
    implicit, _ = default.retrieve(
        "guard", retrieval_id="default", turn_id="turn-default",
        candidate_types=(CandidateType.EXPERIENCE,), created_at="2026-10-03T00:00:00Z",
    )
    assert {item.candidate.candidate_id for item in explicit} >= {"experience-guard"}
    assert tuple(item.candidate.candidate_id for item in implicit) == ("experience-guard",)
    assert store.list_relevance_selections()[0].selection_mode is KnowledgeSelectionMode.LEGACY_APPLICABLE


def test_selective_mode_never_resurrects_inactive_candidate(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    # A manifest without ACTIVE lifecycle/materialization is rejected by the pre-existing gate.
    active = _candidate(
        store, scope, tmp_path, "active", CandidateType.EXPERIENCE,
        "Checkpoint replay evidence", "Update checkpoint replay evidence.",
    )
    inactive = active.__class__.create(
        candidate_id="inactive",
        candidate_type=active.candidate_type,
        content_class=active.content_class,
        qualifying_sources=active.qualifying_sources,
        repository_scope_id=scope.repository_scope_id,
        title=active.title,
        reusable_content=active.reusable_content,
        preconditions=active.preconditions,
        applicability_fingerprints=active.applicability_fingerprints,
        extraction_policy=active.extraction_policy,
        extraction_policy_version=active.extraction_policy_version,
        provenance_refs=active.provenance_refs,
        created_at=active.created_at,
        created_by=active.created_by,
    )
    store.write_candidate(inactive)
    retriever = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(scope.repository_scope_id, tmp_path),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
    )
    items, _ = retriever.retrieve(
        "checkpoint replay evidence", retrieval_id="inactive-check", turn_id="turn",
        candidate_types=(CandidateType.EXPERIENCE,), created_at="2026-10-03T00:00:00Z",
    )
    assert tuple(item.candidate.candidate_id for item in items) == ("active",)
    assert not any(item.candidate_id == "inactive" for item in store.list_relevance_selections())


def test_malformed_selective_query_fails_to_empty_not_inject_all(tmp_path: Path) -> None:
    store, items, _ = _retrieve(tmp_path, object())  # type: ignore[arg-type]
    assert items == ()
    assert {
        item.reason for item in store.list_relevance_selections(turn_id="turn-test")
    } == {RelevanceReason.ABSTAIN_UNSUPPORTED_RELEVANCE_INPUT}


def test_frozen_corpus_selectivity_and_context_cost(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    # P3 materialization IDs are deliberately long; keep the Windows physical
    # root short while retaining the same logical candidate identities.
    state = tmp_path.parent / "p3r3-s"
    result = evaluate_selectivity(
        state_root=state, workspace=repository, reviewer_id="human:test"
    )
    assert result["corpus_digest"] == (
        "39b04fca4f6aa9e54568594b1c18dfe9f71c2292c643469197a96c8977368fbf"
    )
    legacy = result["modes"][KnowledgeSelectionMode.LEGACY_APPLICABLE.value]["tasks"]
    selective = result["modes"][KnowledgeSelectionMode.TASK_RELEVANCE_V1.value]["tasks"]
    assert all(item["selected_count"] < 6 for item in selective.values()), selective
    assert "scoped-skill-flow" in selective["p3r-int-01"]["selected_candidate_ids"]
    assert all(
        "portable-artifact-paths" not in item["selected_candidate_ids"]
        for item in selective.values()
    ), selective
    assert len(
        {tuple(item["selected_candidate_ids"]) for item in selective.values()}
    ) > 1
    assert all(
        selective[task_id]["approximate_context_tokens"]
        <= legacy[task_id]["approximate_context_tokens"]
        for task_id in selective
    )


def test_offline_selectivity_report_is_byte_deterministic(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    first = evaluate_selectivity(
        state_root=tmp_path.parent / "p3r3-a",
        workspace=repository,
        reviewer_id="human:test",
    )
    second = evaluate_selectivity(
        state_root=tmp_path.parent / "p3r3-b",
        workspace=repository,
        reviewer_id="human:test",
    )
    assert first == second
