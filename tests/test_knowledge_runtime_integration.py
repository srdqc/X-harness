from __future__ import annotations

from pathlib import Path

import pytest

from pico.context_engine.base import AssemblyContext
from pico.context_engine.segments.skills import SkillsSegmentBuilder
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeSelectionMode,
    LifecycleActorType,
    LifecycleReason,
    LifecycleState,
    materialized_skill_root,
)
from pico.knowledge_evolution.runtime import KnowledgeContextSegmentBuilder
from pico.memory_engine.base import TokenBudget
from pico.memory_engine.skill_forge import SkillForgeRouter
from pico.memory_engine.skill_forge.knowledge_source import ApplicableKnowledgeSkillSource
from pico.memory_engine.skill_local.registry import SkillRegistry
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore
from tests._knowledge_runtime_helpers import NOW, active_materialized_candidate, file_guard, make_scope


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


def _recorder(state: Path, turn_id="turn-runtime"):
    return evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=TraceStore(state).append_event,
    )


@pytest.mark.asyncio
async def test_fact_and_experience_inject_bounded_context_and_turn_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    store = KnowledgeRecordStore(state)
    scope = make_scope(store)
    guard = file_guard(workspace, "project.cfg", b"safe")
    active_materialized_candidate(
        store, scope, "fact-config", CandidateType.MEMORY_FACT,
        title="Safe config", content="The project config is safe.", fingerprints=(guard,),
    )
    active_materialized_candidate(
        store, scope, "experience-config", CandidateType.EXPERIENCE,
        title="Verify config", content="Verify project config before release.", fingerprints=(guard,),
    )
    builder = KnowledgeContextSegmentBuilder(
        store,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        max_tokens=200,
    )
    recorder = _recorder(state)
    recorder.emit(evidence.TURN_STARTED)
    with evidence.turn_scope(recorder):
        segment = await builder.build(_context("verify safe project config"))
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )
    assert "Repository Facts" in segment.text
    assert "Experience / Lessons" in segment.text
    assert len(segment.meta["p3_injected_candidate_ids"]) == 2
    readback = evidence.read_turn_evidence(state, recorder.turn_id)
    assert [item.event_type for item in readback.events].count(evidence.KNOWLEDGE_USAGE) == 2
    reconstructed = replay.replay_turn(state, recorder.turn_id)
    assert reconstructed.evidence_status is evidence.EvidenceCompleteness.COMPLETE
    assert (
        verifier.verify_replay(reconstructed).overall_status
        is verifier.VerificationStatus.PASS
    )


@pytest.mark.asyncio
async def test_skill_reuse_uses_registry_router_and_suppresses_deprecated_before_ranking(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    store = KnowledgeRecordStore(state)
    scope = make_scope(store)
    guard = file_guard(workspace, "project.cfg", b"safe")
    active, _, _ = active_materialized_candidate(
        store, scope, "skill-active", CandidateType.SKILL_CANDIDATE,
        title="Release verification", content="Verify the release deterministically.", fingerprints=(guard,),
    )
    deprecated, _, manager = active_materialized_candidate(
        store, scope, "skill-deprecated", CandidateType.SKILL_CANDIDATE,
        title="Release verification old", content="Verify the old release process.", fingerprints=(guard,),
    )
    manager.transition(
        transition_id="deprecate-skill",
        candidate_id=deprecated.candidate_id,
        from_state=LifecycleState.ACTIVE,
        to_state=LifecycleState.DEPRECATED,
        actor_type=LifecycleActorType.HUMAN,
        actor_id="human:alice",
        reason=LifecycleReason.HUMAN_DEPRECATED,
        transitioned_at=NOW,
    )
    source_label = f"experience:{scope.repository_scope_id}"
    registry = SkillRegistry(
        workspace,
        builtin_skills_dir=tmp_path / "none",
        extra_dirs=[(materialized_skill_root(store, scope.repository_scope_id), source_label, False)],
    )
    source = ApplicableKnowledgeSkillSource(
        store,
        registry,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        source_label,
    )
    builder = SkillsSegmentBuilder(SkillForgeRouter([source]), skill_top_k=5, activation_max=1)
    recorder = _recorder(state, "turn-skill")
    with evidence.turn_scope(recorder):
        segment = await builder.build(_context("use skill active for release verification"))
    assert f"experience/{active.candidate_id}" in segment.meta["injected_skill_ids"]
    assert all(deprecated.candidate_id not in item for item in segment.meta["injected_skill_ids"])
    usages = store.list_usages(turn_id="turn-skill")
    assert len(usages) == 1
    usage = usages[0]
    assert usage.candidate_id == active.candidate_id
    assert usage.usage_mode.value == "activated"


@pytest.mark.asyncio
async def test_corrupt_optional_store_falls_back_to_empty_without_provider_or_tool_activity(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path / "state")
    scope = make_scope(store)
    store.candidates.mkdir(parents=True, exist_ok=True)
    (store.candidates / "broken.json").write_text("{", encoding="utf-8")
    builder = KnowledgeContextSegmentBuilder(
        store,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, tmp_path),
    )
    segment = await builder.build(_context("anything"))
    assert segment.text == ""
    assert segment.meta["p3_injected_candidate_ids"] == []
    assert segment.meta["p3_knowledge_failure"]


@pytest.mark.asyncio
async def test_selective_abstention_emits_no_injected_usage(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    store = KnowledgeRecordStore(state)
    scope = make_scope(store)
    guard = file_guard(workspace, "project.cfg", b"safe")
    active_materialized_candidate(
        store,
        scope,
        "experience-guard",
        CandidateType.EXPERIENCE,
        title="Fail-closed guard recovery",
        content="Preserve fail-closed guard encoding and applicability semantics.",
        fingerprints=(guard,),
    )
    builder = KnowledgeContextSegmentBuilder(
        store,
        lambda: ApplicabilityEnvironment(scope.repository_scope_id, workspace),
        selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
    )
    recorder = _recorder(state, "turn-abstain")
    with evidence.turn_scope(recorder):
        segment = await builder.build(_context("update unrelated weather forecast"))
    assert segment.text == ""
    assert segment.meta["p3_injected_candidate_ids"] == []
    assert store.list_usages(turn_id="turn-abstain") == ()
    decisions = store.list_relevance_selections(turn_id="turn-abstain")
    assert len(decisions) == 1
    assert decisions[0].decision.value == "abstain"


def test_runtime_adapter_has_no_tool_permission_terminal_or_verifier_authority() -> None:
    source = (Path(__file__).parents[1] / "pico" / "knowledge_evolution" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert all(
        forbidden not in source
        for forbidden in (
            "ToolRegistry",
            "execute_tool",
            "grant_permission",
            "sandbox_config",
            "terminal_state",
            "verify_replay",
            "Provider",
            "Delivery",
        )
    )
