from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live.artifacts import load_manifest, load_runs
from benchmarks.picobench.packs.knowledge_evolution_live.jev_experiment import (
    JEV_UTILITY_EXPERIMENT_ARMS,
    JEV_UTILITY_MAIN_COMPARISON,
    JevUtilityExperimentDesign,
)
from benchmarks.picobench.packs.knowledge_evolution_live.metrics import (
    extract_utility_observability,
)
from benchmarks.picobench.packs.knowledge_evolution_live.reducer import reduce_campaign
from benchmarks.picobench.packs.knowledge_evolution_live.schema import CampaignPaths
from pico.config.pico import ContextConfig
from pico.context_engine.base import AssemblyContext
from pico.context_engine.segments.skills import SkillsSegmentBuilder
from pico.decision_plane.fake import JevFakeMode, ScriptedJevBackend
from pico.decision_plane.utility import (
    JevUtilityCandidate,
    JevUtilityDecisionAdapter,
    JevUtilityRequest,
)
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeRetriever,
    KnowledgeSelectionMode,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    materialized_skill_root,
)
from pico.knowledge_evolution.runtime import KnowledgeContextSegmentBuilder
from pico.knowledge_evolution.utility import KnowledgeUtilityCoordinator
from pico.memory_engine.base import TokenBudget
from pico.memory_engine.skill_forge import SkillForgeRouter
from pico.memory_engine.skill_forge.knowledge_source import ApplicableKnowledgeSkillSource
from pico.memory_engine.skill_local.registry import SkillRegistry
from pico.tracing import evidence
from pico.tracing.store import TraceStore
from tests._knowledge_runtime_helpers import active_materialized_candidate, file_guard, make_scope


def _context(query: str) -> AssemblyContext:
    return AssemblyContext(
        session_key="session",
        current_message=query,
        media=None,
        channel="cli",
        chat_id=None,
        session_messages=[],
        budget=TokenBudget(8000, 1000, 1000, 1000, 5000),
    )


def _selected(tmp_path: Path, *, include_irrelevant: bool = False):
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    store = KnowledgeRecordStore(state)
    scope = make_scope(store)
    guard = file_guard(workspace, "project.cfg", b"safe")
    active_materialized_candidate(
        store,
        scope,
        "candidate-a",
        CandidateType.MEMORY_FACT,
        title="Guard recovery",
        content="Preserve fail-closed guard recovery behavior.",
        fingerprints=(guard,),
    )
    active_materialized_candidate(
        store,
        scope,
        "candidate-b",
        CandidateType.EXPERIENCE,
        title="Checkpoint replay",
        content="Validate checkpoint replay evidence.",
        fingerprints=(guard,),
    )
    if include_irrelevant:
        active_materialized_candidate(
            store,
            scope,
            "candidate-z",
            CandidateType.EXPERIENCE,
            title="Weather forecast",
            content="Review tomorrow weather forecast.",
            fingerprints=(guard,),
        )
    retriever = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
    )
    items, receipt = retriever.retrieve(
        "guard recovery checkpoint replay",
        retrieval_id="retrieval-utility",
        turn_id="turn-utility",
        candidate_types=(CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
        created_at="2026-10-03T00:00:00Z",
    )
    return workspace, state, store, scope, items, receipt


def _request() -> JevUtilityRequest:
    return JevUtilityRequest(
        decision_id="utility:test",
        turn_id="turn-test",
        repository_scope_id="a" * 64,
        query="bounded current task",
        query_digest="b" * 64,
        candidates=(
            JevUtilityCandidate("candidate-a", "memory_fact", "A", "summary", 1, 1.0, "c" * 64),
            JevUtilityCandidate("candidate-b", "experience", "B", "summary", 2, 0.5, "d" * 64),
        ),
    )


def test_utility_request_is_bounded_private_and_deterministic() -> None:
    first = _request()
    second = _request()
    assert first.request_digest == second.request_digest
    assert first.candidate_set_digest == second.candidate_set_digest
    assert "expected patch" not in repr(first)
    with pytest.raises(ValueError):
        JevUtilityRequest(
            decision_id="utility:bad",
            turn_id=None,
            repository_scope_id="a" * 64,
            query="x" * 4097,
            query_digest="b" * 64,
            candidates=first.candidates,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        (JevFakeMode.TIMEOUT, "backend_timeout"),
        (JevFakeMode.EXCEPTION, "backend_exception"),
        (JevFakeMode.MALFORMED, "malformed_response"),
        (JevFakeMode.UNAVAILABLE, "backend_unavailable"),
        (JevFakeMode.UNKNOWN_CANDIDATE, "unknown_candidate_id"),
        (JevFakeMode.DUPLICATE_CANDIDATE, "duplicate_candidate_id"),
        (JevFakeMode.INCOMPLETE_RANKING, "incomplete_candidate_set"),
        (JevFakeMode.INVALID_SCORE, "invalid_utility_score"),
        (JevFakeMode.INVALID_CONFIDENCE, "invalid_confidence"),
        (JevFakeMode.INVALID_DECISION, "invalid_decision"),
        (JevFakeMode.INVALID_RANK, "invalid_backend_rank"),
    ],
)
async def test_invalid_or_failed_backend_falls_back_to_relevant_set(
    tmp_path: Path, mode: JevFakeMode, reason: str
) -> None:
    _, _, _, scope, items, receipt = _selected(tmp_path)
    backend = ScriptedJevBackend(mode)
    coordinator = KnowledgeUtilityCoordinator(
        JevUtilityDecisionAdapter(
            backend,
            timeout_seconds=0.001 if mode is JevFakeMode.TIMEOUT else 0.25,
        )
    )
    result = await coordinator.refine(
        items,
        query="guard recovery checkpoint replay",
        turn_id="turn-utility",
        repository_scope_id=scope.repository_scope_id,
        group_id=receipt.retrieval_id,
    )
    assert result.items == items
    assert result.fallback_used is True
    assert result.fallback_reason == reason


@pytest.mark.asyncio
async def test_missing_backend_fallback_is_not_all_applicable(tmp_path: Path) -> None:
    _, _, store, scope, items, receipt = _selected(tmp_path, include_irrelevant=True)
    result = await KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(None)).refine(
        items,
        query="guard recovery checkpoint replay",
        turn_id="turn-utility",
        repository_scope_id=scope.repository_scope_id,
        group_id=receipt.retrieval_id,
    )
    assert result.items == items
    assert "candidate-z" not in {item.candidate.candidate_id for item in result.items}
    assert {item.candidate_id for item in store.list_relevance_selections() if item.decision.value == "select"} == {
        item.candidate.candidate_id for item in items
    }


@pytest.mark.asyncio
async def test_unexpected_adapter_failure_cannot_stall_or_fail_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, scope, items, receipt = _selected(tmp_path)
    adapter = JevUtilityDecisionAdapter(ScriptedJevBackend())

    async def fail_unexpectedly(_request):
        raise RuntimeError("unexpected adapter bug")

    monkeypatch.setattr(adapter, "decide", fail_unexpectedly)
    result = await KnowledgeUtilityCoordinator(adapter).refine(
        items,
        query="guard recovery checkpoint replay",
        turn_id="turn-utility",
        repository_scope_id=scope.repository_scope_id,
        group_id=receipt.retrieval_id,
    )
    assert result.items == items
    assert result.fallback_used is True
    assert result.fallback_reason == "utility_internal_error"


@pytest.mark.asyncio
async def test_utility_cannot_resurrect_inactive_or_stale_candidates(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    active_guard = file_guard(workspace, "active.cfg", b"safe")
    stale_guard = file_guard(workspace, "stale.cfg", b"old")
    active_materialized_candidate(
        store,
        scope,
        "active",
        CandidateType.EXPERIENCE,
        title="Guard evidence active",
        content="Validate guard evidence.",
        fingerprints=(active_guard,),
    )
    inactive, _, manager = active_materialized_candidate(
        store,
        scope,
        "inactive",
        CandidateType.EXPERIENCE,
        title="Guard evidence inactive",
        content="Validate guard evidence.",
        fingerprints=(active_guard,),
    )
    manager.transition(
        transition_id="deprecate-inactive",
        candidate_id=inactive.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.DEPRECATED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:test",
        reason=LifecycleReason.HUMAN_DEPRECATED,
        transitioned_at="2026-10-03T00:00:01Z",
    )
    active_materialized_candidate(
        store,
        scope,
        "stale",
        CandidateType.EXPERIENCE,
        title="Guard evidence stale",
        content="Validate guard evidence.",
        fingerprints=(stale_guard,),
    )
    (workspace / "stale.cfg").write_bytes(b"changed")
    retriever = KnowledgeRetriever(
        store,
        ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
    )
    items, receipt = retriever.retrieve(
        "validate guard evidence",
        retrieval_id="authority-gates",
        turn_id="turn-authority",
        candidate_types=(CandidateType.EXPERIENCE,),
        created_at="2026-10-03T00:00:02Z",
    )
    assert {item.candidate.candidate_id for item in items} == {"active"}
    result = await KnowledgeUtilityCoordinator(
        JevUtilityDecisionAdapter(ScriptedJevBackend(JevFakeMode.KEEP_ALL))
    ).refine(
        items,
        query="validate guard evidence",
        turn_id="turn-authority",
        repository_scope_id=scope.repository_scope_id,
        group_id=receipt.retrieval_id,
    )
    assert {item.candidate.candidate_id for item in result.items} == {"active"}


@pytest.mark.asyncio
async def test_zero_selection_skips_backend_and_persists_skipped_receipt(tmp_path: Path) -> None:
    backend = ScriptedJevBackend()
    coordinator = KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(backend))
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-zero",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        result = await coordinator.refine(
            (),
            query="unrelated",
            turn_id="turn-zero",
            repository_scope_id="a" * 64,
            group_id="retrieval-zero",
        )
    assert result.items == ()
    assert backend.utility_calls == 0
    assert records[0]["metadata"]["fallback_reason"] == "skipped_no_relevant_candidates"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (JevFakeMode.KEEP_ALL, {"candidate-a", "candidate-b"}),
        (JevFakeMode.ABSTAIN_ONE, {"candidate-a"}),
        (JevFakeMode.ABSTAIN_ALL, set()),
        (JevFakeMode.REORDER, {"candidate-a", "candidate-b"}),
    ],
)
async def test_valid_utility_decisions_only_remove_and_preserve_relevance_order(
    tmp_path: Path, mode: JevFakeMode, expected: set[str]
) -> None:
    _, _, _, scope, items, receipt = _selected(tmp_path)
    result = await KnowledgeUtilityCoordinator(
        JevUtilityDecisionAdapter(ScriptedJevBackend(mode))
    ).refine(
        items,
        query="guard recovery checkpoint replay",
        turn_id="turn-utility",
        repository_scope_id=scope.repository_scope_id,
        group_id=receipt.retrieval_id,
    )
    assert {item.candidate.candidate_id for item in result.items} == expected
    assert [item.candidate.candidate_id for item in result.items] == [
        item.candidate.candidate_id for item in items if item.candidate.candidate_id in expected
    ]


@pytest.mark.asyncio
async def test_context_abstention_controls_injection_and_usage(tmp_path: Path) -> None:
    workspace, state, store, scope, _, _ = _selected(tmp_path)
    backend = ScriptedJevBackend(JevFakeMode.ABSTAIN_ONE)
    builder = KnowledgeContextSegmentBuilder(
        store,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
        utility_coordinator=KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(backend)),
    )
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-context",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=TraceStore(state).append_event,
    )
    with evidence.turn_scope(recorder):
        segment = await builder.build(_context("guard recovery checkpoint replay"))
    assert segment.meta["p3_utility_abstained_candidate_ids"] == ["candidate-b"]
    assert segment.meta["p3_injected_candidate_ids"] == ["candidate-a"]
    assert {item.candidate_id for item in store.list_usages(turn_id="turn-context")} == {
        "candidate-a"
    }


@pytest.mark.asyncio
async def test_keep_all_rendering_matches_deterministic_selective_baseline(tmp_path: Path) -> None:
    workspace, _, store, scope, _, _ = _selected(tmp_path / "baseline")
    utility_workspace, _, utility_store, utility_scope, _, _ = _selected(tmp_path / "utility")

    def environment() -> ApplicabilityEnvironment:
        return ApplicabilityEnvironment(scope.repository_scope_id, workspace)

    baseline = KnowledgeContextSegmentBuilder(
        store,
        environment,
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
    )
    backend = ScriptedJevBackend(JevFakeMode.KEEP_ALL)
    utility = KnowledgeContextSegmentBuilder(
        utility_store,
        lambda: ApplicabilityEnvironment(
            utility_scope.repository_scope_id, utility_workspace
        ),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
        utility_coordinator=KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(backend)),
    )
    baseline_segment = await baseline.build(_context("guard recovery checkpoint replay"))
    utility_segment = await utility.build(_context("guard recovery checkpoint replay"))
    assert utility_segment.text == baseline_segment.text
    assert utility_segment.meta["p3_injected_candidate_ids"] == baseline_segment.meta[
        "p3_injected_candidate_ids"
    ]
    assert backend.utility_calls == 1


@pytest.mark.asyncio
async def test_concurrent_relevance_lanes_make_one_backend_call_per_turn(tmp_path: Path) -> None:
    _, _, _, scope, items, _ = _selected(tmp_path)
    backend = ScriptedJevBackend(JevFakeMode.KEEP_ALL)
    coordinator = KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(backend))
    left, right = items[:1], items[1:]
    results = await asyncio.gather(
        coordinator.refine(
            left,
            query="guard recovery checkpoint replay",
            turn_id="turn-shared",
            repository_scope_id=scope.repository_scope_id,
            group_id="facts",
        ),
        coordinator.refine(
            right,
            query="guard recovery checkpoint replay",
            turn_id="turn-shared",
            repository_scope_id=scope.repository_scope_id,
            group_id="skills",
        ),
    )
    assert backend.utility_calls == 1
    assert tuple(item.candidate.candidate_id for result in results for item in result.items) == tuple(
        item.candidate.candidate_id for item in items
    )


@pytest.mark.asyncio
async def test_skill_utility_can_reference_but_cannot_activate(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    store = KnowledgeRecordStore(state)
    scope = make_scope(store)
    guard = file_guard(workspace, "project.cfg", b"safe")
    candidate, _, _ = active_materialized_candidate(
        store,
        scope,
        "skill-release",
        CandidateType.SKILL_CANDIDATE,
        title="Release verification",
        content="Verify release evidence deterministically.",
        fingerprints=(guard,),
    )
    source_label = f"experience:{scope.repository_scope_id}"
    registry = SkillRegistry(
        workspace,
        builtin_skills_dir=tmp_path / "none",
        extra_dirs=[(materialized_skill_root(store, scope.repository_scope_id), source_label, False)],
    )
    backend = ScriptedJevBackend(JevFakeMode.KEEP_ALL)
    source = ApplicableKnowledgeSkillSource(
        store,
        registry,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        source_label,
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
        utility_coordinator=KnowledgeUtilityCoordinator(JevUtilityDecisionAdapter(backend)),
    )
    builder = SkillsSegmentBuilder(SkillForgeRouter([source]), skill_top_k=5, activation_max=0)
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-skill-utility",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=TraceStore(state).append_event,
    )
    with evidence.turn_scope(recorder):
        segment = await builder.build(_context("release verification evidence"))
    qualified = f"experience/{candidate.candidate_id}"
    assert segment.meta["injected_skill_ids"] == []
    assert segment.meta["referenced_skill_ids"] == [qualified]
    assert store.list_usages(turn_id="turn-skill-utility")[0].usage_mode.value == "referenced"


@pytest.mark.asyncio
async def test_receipt_is_turn_correlated_bounded_and_contains_backend_identity(tmp_path: Path) -> None:
    _, state, _, scope, items, receipt = _selected(tmp_path)
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-utility",
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        await KnowledgeUtilityCoordinator(
            JevUtilityDecisionAdapter(ScriptedJevBackend(JevFakeMode.ABSTAIN_ONE))
        ).refine(
            items,
            query="private full task must not persist",
            turn_id="turn-utility",
            repository_scope_id=scope.repository_scope_id,
            group_id=receipt.retrieval_id,
        )
    metadata = records[0]["metadata"]
    assert metadata["receipt_schema"] == "pico.jev-utility-decision.v1"
    assert metadata["backend_id"] == "scripted_jev"
    assert metadata["config_digest"]
    assert metadata["turn_id"] == "turn-utility"
    persisted = json.dumps(records[0])
    assert "private full task must not persist" not in persisted
    assert str(state.resolve()) not in persisted
    observability = extract_utility_observability(
        (SimpleNamespace(event_type=records[0]["event_type"], metadata=metadata),)
    )
    assert observability["utility_invoked_count"] == 1
    assert observability["utility_candidate_count"] == 2
    assert observability["utility_abstained_candidate_ids"] == ("candidate-b",)


def test_utility_mode_is_explicit_and_default_remains_legacy() -> None:
    default = ContextConfig()
    assert default.knowledge_selection_mode == "legacy_applicable"
    configured = ContextConfig(knowledge_selection_mode="task_relevance_v1_jev_utility")
    assert configured.knowledge_selection_mode == "task_relevance_v1_jev_utility"
    with pytest.raises(ValueError):
        ContextConfig(knowledge_utility_timeout_seconds=True)


def test_future_experiment_keeps_three_arms_and_compares_selective_to_utility() -> None:
    design = JevUtilityExperimentDesign()
    assert design.arms == JEV_UTILITY_EXPERIMENT_ARMS
    assert design.main_comparison == JEV_UTILITY_MAIN_COMPARISON
    assert len(design.questions) == 8
    assert len(design.design_digest) == 64


def test_utility_module_has_no_runtime_authority_or_workspace_mutation() -> None:
    source = (Path(__file__).parents[1] / "pico/decision_plane/utility.py").read_text(
        encoding="utf-8"
    )
    assert all(
        forbidden not in source
        for forbidden in (
            "ToolRegistry",
            "execute_tool",
            "tool_args",
            "activate_skill",
            "verify_replay",
            "terminal_state",
            "write_text",
            "open(",
        )
    )


@pytest.mark.parametrize(
    ("campaign_id", "classification", "selective_success"),
    [
        ("p3r-543e189ff8cfbbad", "regressive", 11 / 12),
        ("p3r-a276191655c89e8f", "neutral", 1.0),
    ],
)
def test_historical_p3r_campaigns_reduce_without_reinterpretation(
    campaign_id: str, classification: str, selective_success: float
) -> None:
    paths = CampaignPaths.at(Path(".p3r") / campaign_id)
    if not paths.manifest.exists():
        pytest.skip("immutable local historical campaign is not present")
    result = reduce_campaign(load_manifest(paths.manifest), load_runs(paths))
    assert result["classification"] == classification
    assert result["success_rates"]["no_reuse"] == 1.0
    assert result["success_rates"]["approved_reuse_selective"] == selective_success
