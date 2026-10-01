from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from pico.context_engine.base import AssemblyContext, TokenBudget
from pico.context_engine.segments.active_skills import ActiveSkillsSegmentBuilder
from pico.context_engine.segments.skills import SkillsSegmentBuilder
from pico.decision_plane import DeterministicSkillRankingAdapter, JevDecisionAdapter
from pico.decision_plane.fake import JevFakeMode, ScriptedJevBackend
from pico.decision_plane.jev import JevBackendResponse, JevRankingAdvice
from pico.memory_engine.skill_forge import (
    LocalSkillCatalog,
    LocalSkillResolver,
    LocalSkillSource,
    RouterHit,
    SkillForgeRouter,
)
from pico.memory_engine.skill_local import LocalPool, SkillRegistry
from pico.tracing import evidence, replay, verifier
from pico.tracing.store import TraceStore


def _hit(candidate_id: str, name: str, *, skill_dir: Path | None = None) -> RouterHit:
    return RouterHit(
        qualified_id=candidate_id,
        name=name,
        content=f"body for {name}",
        score=1.0,
        meta={
            "source": candidate_id.partition("/")[0],
            "description": f"use {name}",
            "skill_dir": str(skill_dir) if skill_dir is not None else None,
        },
    )


class _StaticSource:
    name = "local"
    weight = 1.0

    def __init__(self, hits: list[RouterHit]) -> None:
        self.hits = hits

    async def search(self, query: str, history: list[dict[str, Any]], k: int) -> list[RouterHit]:
        del query, history
        return list(self.hits[:k])


class _OrderedBackend:
    def __init__(self, order: tuple[str, ...]) -> None:
        self.order = order
        self.calls = 0
        self.candidate_ids: tuple[str, ...] = ()

    async def rank_skill_candidates(self, request) -> JevBackendResponse:
        self.calls += 1
        self.candidate_ids = tuple(item.candidate_id for item in request.candidates)
        order_index = {candidate_id: index for index, candidate_id in enumerate(self.order)}
        ranked = sorted(
            self.candidate_ids,
            key=lambda candidate_id: order_index.get(candidate_id, len(order_index)),
        )
        return JevBackendResponse(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            ranking=tuple(JevRankingAdvice(candidate_id) for candidate_id in ranked),
        )


class _LateBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.returned = False

    async def rank_skill_candidates(self, request) -> JevBackendResponse:
        self.calls += 1
        try:
            await asyncio.sleep(3_600)
        except asyncio.CancelledError:
            # Simulate a non-cooperative client that produces advice after the
            # host deadline.  The adapter must discard this result.
            pass
        self.returned = True
        return JevBackendResponse(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            ranking=tuple(
                JevRankingAdvice(item.candidate_id)
                for item in reversed(request.candidates)
            ),
        )


def _ctx(message: str) -> AssemblyContext:
    return AssemblyContext(
        session_key="session",
        current_message=message,
        media=None,
        channel=None,
        chat_id=None,
        session_messages=[],
        budget=TokenBudget(
            context_length=20_000,
            reserved_output=2_000,
            reserved_tools=1_000,
            reserved_system=1_000,
            available_history=16_000,
        ),
    )


def _synthetic_router(*, enabled: bool, adapter=None) -> SkillForgeRouter:
    return SkillForgeRouter(
        [
            _StaticSource(
                [
                    _hit("local/first", "first"),
                    _hit("local/second", "second"),
                ]
            )
        ],
        decision_plane_enabled=enabled,
        decision_adapter=adapter,
    )


def _write_skill(
    root: Path,
    name: str,
    *,
    body: str | None = None,
    requires: dict[str, list[str]] | None = None,
    always: bool = False,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = [f"name: {name}", f"description: deploy with {name}"]
    if always:
        frontmatter.append("always: true")
    if requires is not None:
        frontmatter.append(f"metadata: {json.dumps({'pico': {'requires': requires}})}")
    rendered_frontmatter = "\n".join(frontmatter)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{rendered_frontmatter}\n---\n\n{body or f'body for {name}'}\n",
        encoding="utf-8",
    )
    return skill_dir


async def test_disabled_and_deterministic_modes_preserve_exact_baseline() -> None:
    baseline = await _synthetic_router(enabled=False).select("query", [], k=2)
    backend = ScriptedJevBackend()
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-disabled",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        disabled = await _synthetic_router(
            enabled=False,
            adapter=JevDecisionAdapter(backend),
        ).select("query", [], k=2)
    deterministic = await _synthetic_router(
        enabled=True,
        adapter=DeterministicSkillRankingAdapter(),
    ).select("query", [], k=2)

    assert disabled == baseline
    assert deterministic == baseline
    assert backend.calls == 0
    assert records == []


async def test_valid_jev_order_reaches_resolver_and_context_without_changing_membership() -> None:
    backend = _OrderedBackend(("local/second", "local/first"))
    router = _synthetic_router(enabled=True, adapter=JevDecisionAdapter(backend))
    resolution = await LocalSkillResolver(router, activation_limit=1).resolve("query", [])
    assert [hit.qualified_id for hit in resolution.activated] == ["local/second"]

    segment = await SkillsSegmentBuilder(router, skill_top_k=2, activation_max=1).build(_ctx("query"))
    assert segment is not None
    assert segment.meta["injected_skill_ids"] == ["local/second"]
    assert "body for second" in segment.text
    assert "body for first" not in segment.text
    assert set(backend.candidate_ids) == {
        "local/first",
        "local/second",
    }


@pytest.mark.parametrize(
    "mode",
    [
        JevFakeMode.EXCEPTION,
        JevFakeMode.MALFORMED,
        JevFakeMode.WRONG_DECISION_TYPE,
        JevFakeMode.UNKNOWN_CANDIDATE,
        JevFakeMode.DUPLICATE_CANDIDATE,
        JevFakeMode.INCOMPLETE_RANKING,
        JevFakeMode.INVALID_SCORE,
        JevFakeMode.INVALID_CONFIDENCE,
    ],
)
async def test_every_invalid_treatment_reaches_resolver_in_exact_baseline_order(mode) -> None:
    router = _synthetic_router(
        enabled=True,
        adapter=JevDecisionAdapter(ScriptedJevBackend(mode)),
    )
    resolution = await LocalSkillResolver(router, activation_limit=2).resolve("query", [])
    assert [hit.qualified_id for hit in resolution.activated] == [
        "local/first",
        "local/second",
    ]


async def test_late_timeout_result_cannot_reorder_or_repeat_resolution() -> None:
    backend = _LateBackend()
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-late",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    router = _synthetic_router(
        enabled=True,
        adapter=JevDecisionAdapter(backend, timeout_seconds=0.001),
    )
    with evidence.turn_scope(recorder):
        resolution = await LocalSkillResolver(router, activation_limit=2).resolve("query", [])
        applied = tuple(hit.qualified_id for hit in resolution.activated)
        await asyncio.sleep(0)

    assert applied == ("local/first", "local/second")
    assert tuple(hit.qualified_id for hit in resolution.activated) == applied
    assert backend.calls == 1
    assert backend.returned is True
    assert len(records) == 1
    assert records[0]["metadata"]["fallback_reason"] == "backend_timeout"


@pytest.mark.parametrize(
    ("requires", "missing_env"),
    [
        ({"bins": ["x_harness_binary_that_does_not_exist"]}, None),
        ({"env": ["X_HARNESS_MISSING_SKILL_ENV"]}, "X_HARNESS_MISSING_SKILL_ENV"),
    ],
)
async def test_jev_cannot_activate_high_ranked_skill_with_unmet_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requires: dict[str, list[str]],
    missing_env: str | None,
) -> None:
    if missing_env is not None:
        monkeypatch.delenv(missing_env, raising=False)
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    _write_skill(workspace / "skills", "blocked", requires=requires)
    _write_skill(workspace / "skills", "eligible")
    registry = SkillRegistry(workspace, builtin_skills_dir=builtin)
    backend = _OrderedBackend(("local/blocked", "local/eligible"))
    router = SkillForgeRouter(
        [LocalSkillSource(LocalPool(registry), registry)],
        decision_plane_enabled=True,
        decision_adapter=JevDecisionAdapter(backend),
    )

    resolution = await LocalSkillResolver(router, activation_limit=1).resolve(
        "use blocked and eligible to deploy",
        [],
    )

    assert backend.candidate_ids[0] == "local/blocked"
    assert [hit.qualified_id for hit in resolution.activated] == ["local/eligible"]
    assert all(hit.qualified_id != "local/blocked" for hit in resolution.references)


async def test_ranking_does_not_redefine_referenced_or_activated_categories(tmp_path: Path) -> None:
    alpha_dir = _write_skill(tmp_path, "alpha")
    beta_dir = _write_skill(tmp_path, "beta", body="release preparation")
    source = _StaticSource(
        [
            _hit("local/alpha", "alpha", skill_dir=alpha_dir),
            _hit("local/beta", "beta", skill_dir=beta_dir),
        ]
    )
    router = SkillForgeRouter(
        [source],
        decision_plane_enabled=True,
        decision_adapter=JevDecisionAdapter(ScriptedJevBackend(reverse=True)),
    )
    resolution = await LocalSkillResolver(router, activation_limit=1).resolve(
        "use alpha for release preparation",
        [],
    )
    assert [hit.qualified_id for hit in resolution.activated] == ["local/alpha"]
    assert [hit.qualified_id for hit in resolution.references] == ["local/beta"]


async def test_registry_source_precedence_and_always_path_remain_outside_jev(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    builtin = tmp_path / "builtin"
    _write_skill(workspace / "skills", "shared", body="workspace shared")
    _write_skill(external, "shared", body="external shared")
    _write_skill(builtin, "shared", body="builtin shared")
    _write_skill(workspace / "skills", "always-on", body="always body", always=True)
    _write_skill(workspace / "skills", "ordinary", body="ordinary body")
    registry = SkillRegistry(
        workspace,
        builtin_skills_dir=builtin,
        extra_dirs=[(external, "external", True)],
    )
    backend = _OrderedBackend(("local/ordinary", "local/shared"))
    router = SkillForgeRouter(
        [LocalSkillSource(LocalPool(registry), registry)],
        decision_plane_enabled=True,
        decision_adapter=JevDecisionAdapter(backend),
    )
    await router.select("shared ordinary always-on", [], k=5)

    assert registry.get("shared").source == "workspace"
    assert "local/always-on" not in backend.candidate_ids

    catalog = LocalSkillCatalog(
        workspace,
        builtin_skills_dir=builtin,
        start_watcher=False,
    )
    active = await ActiveSkillsSegmentBuilder(catalog).build(_ctx("ordinary"))
    assert active is not None
    assert "always body" in active.text


async def test_receipt_proves_applied_source_without_claiming_activation() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-receipt",
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        resolution = await LocalSkillResolver(
            _synthetic_router(
                enabled=True,
                adapter=JevDecisionAdapter(ScriptedJevBackend(reverse=True)),
            ),
            activation_limit=1,
        ).resolve("query", [])

    metadata = records[0]["metadata"]
    assert [hit.qualified_id for hit in resolution.activated] == ["local/second"]
    assert metadata["final_source"] == "jev"
    assert metadata["final_result_digest"] == metadata["experimental_result_digest"]
    assert metadata["final_result_digest"] != metadata["baseline_result_digest"]
    assert "activated" not in metadata
    assert "injected_skill_ids" not in metadata


async def test_fallback_receipt_proves_deterministic_order_was_applied() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-fallback-applied",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        resolution = await LocalSkillResolver(
            _synthetic_router(
                enabled=True,
                adapter=JevDecisionAdapter(ScriptedJevBackend(JevFakeMode.MALFORMED)),
            ),
            activation_limit=2,
        ).resolve("query", [])

    metadata = records[0]["metadata"]
    assert [hit.qualified_id for hit in resolution.activated] == [
        "local/first",
        "local/second",
    ]
    assert metadata["fallback_used"] is True
    assert metadata["final_source"] == "deterministic"
    assert metadata["experimental_result_digest"] is None
    assert metadata["final_result_digest"] == metadata["baseline_result_digest"]


async def test_decision_event_is_additive_for_read_replay_and_verification(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-p2-p1c",
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=store.append_event,
    )
    recorder.emit(evidence.TURN_STARTED)
    with evidence.turn_scope(recorder):
        await _synthetic_router(
            enabled=True,
            adapter=JevDecisionAdapter(ScriptedJevBackend()),
        ).select("query", [], k=2)
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": "completed", "lifecycle_event": "TurnEnded"},
    )

    readback = evidence.read_turn_evidence(tmp_path, recorder.turn_id)
    reconstructed = replay.replay_turn(tmp_path, recorder.turn_id)
    verification = verifier.verify_replay(reconstructed)

    assert readback.completeness is evidence.EvidenceCompleteness.COMPLETE
    assert [item.event_type for item in reconstructed.ordered_timeline] == [
        evidence.TURN_STARTED,
        evidence.DECISION_RECEIPT,
        evidence.TURN_TERMINAL,
    ]
    assert reconstructed.evidence_status is evidence.EvidenceCompleteness.COMPLETE
    assert next(item for item in verification.checks if item.check_id == "core.sequence").status is (
        verifier.VerificationStatus.PASS
    )
    assert verification.derived_facts.tool_execution_count == 0
