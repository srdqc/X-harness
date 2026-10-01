from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pico.decision_plane import (
    DECISION_SCHEMA_VERSION,
    DecisionCandidate,
    DecisionOutcome,
    DecisionRanking,
    DecisionRequest,
    DecisionResult,
    DecisionType,
    DeterministicSkillRankingAdapter,
)
from pico.memory_engine.skill_forge import LocalSkillResolver, RouterHit, SkillForgeRouter


def _hit(candidate_id: str, name: str, score: float) -> RouterHit:
    return RouterHit(
        qualified_id=candidate_id,
        name=name,
        content=f"instructions for {name}",
        score=score,
        meta={"source": candidate_id.partition("/")[0], "description": f"use {name}"},
    )


class _Source:
    name = "local"
    weight = 1.0

    def __init__(self, hits: list[RouterHit]) -> None:
        self._hits = hits

    async def search(self, query, history, k):
        del query, history
        return list(self._hits[:k])


class _CapturingAdapter:
    def __init__(self, *, reverse: bool = False) -> None:
        self.calls = 0
        self.request: DecisionRequest | None = None
        self._reverse = reverse

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        self.calls += 1
        self.request = request
        candidates = reversed(request.candidates) if self._reverse else request.candidates
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source="test",
            outcome=DecisionOutcome.SUCCESS,
            ranking=tuple(DecisionRanking(candidate.candidate_id) for candidate in candidates),
        )


class _UnknownAdapter:
    async def decide(self, request: DecisionRequest) -> DecisionResult:
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source="test",
            outcome=DecisionOutcome.SUCCESS,
            ranking=(DecisionRanking("unknown/skill"),),
        )


class _DuplicateAdapter:
    async def decide(self, request: DecisionRequest) -> DecisionResult:
        candidate_id = request.candidates[0].candidate_id
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source="test",
            outcome=DecisionOutcome.SUCCESS,
            ranking=(DecisionRanking(candidate_id), DecisionRanking(candidate_id)),
        )


class _IncompleteAdapter:
    async def decide(self, request: DecisionRequest) -> DecisionResult:
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source="test",
            outcome=DecisionOutcome.SUCCESS,
            ranking=(DecisionRanking(request.candidates[0].candidate_id),),
        )


class _TimeoutAdapter:
    async def decide(self, request: DecisionRequest) -> DecisionResult:
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source="test",
            outcome=DecisionOutcome.TIMEOUT,
            reason="deadline",
        )


class _FailingAdapter:
    async def decide(self, request: DecisionRequest) -> DecisionResult:
        del request
        raise RuntimeError("adapter unavailable")


class _MalformedAdapter:
    async def decide(self, request: DecisionRequest):
        del request
        return {"ranking": "not-a-decision-result"}


def _request(*, correlation_id: str | None = None) -> DecisionRequest:
    return DecisionRequest(
        decision_type=DecisionType.SKILL_RANKING,
        query="deploy service",
        candidates=(
            DecisionCandidate("local/deploy", "deploy", "deploy service", "local", 1, 0.9),
            DecisionCandidate("local/test", "test", "test service", "local", 2, 0.5),
        ),
        correlation_id=correlation_id,
    )


def test_skill_ranking_contract_is_immutable_and_versioned() -> None:
    request = _request()
    assert request.decision_type is DecisionType.SKILL_RANKING
    assert request.schema_version == DECISION_SCHEMA_VERSION
    with pytest.raises(FrozenInstanceError):
        request.query = "changed"  # type: ignore[misc]

    result = DecisionResult(
        decision_id=request.decision_id,
        decision_type=request.decision_type,
        request_digest=request.request_digest,
        source="deterministic",
        outcome=DecisionOutcome.SUCCESS,
    )
    with pytest.raises(FrozenInstanceError):
        result.source = "changed"  # type: ignore[misc]


def test_request_and_baseline_result_digests_are_deterministic() -> None:
    first = _request(correlation_id="turn-a")
    second = _request(correlation_id="turn-b")
    assert first.candidate_set_digest == second.candidate_set_digest
    assert first.request_digest == second.request_digest
    assert first.decision_id != second.decision_id


async def test_deterministic_adapter_preserves_candidate_order_and_digest() -> None:
    adapter = DeterministicSkillRankingAdapter()
    first = await adapter.decide(_request())
    second = await adapter.decide(_request())
    assert [item.candidate_id for item in first.ranking] == ["local/deploy", "local/test"]
    assert first.result_digest == second.result_digest


async def test_enabled_baseline_reproduces_direct_rrf_ranking() -> None:
    source = _Source([_hit("local/first", "first", 2.0), _hit("local/second", "second", 1.0)])
    direct = SkillForgeRouter([source])
    baseline = SkillForgeRouter([source], decision_plane_enabled=True)
    direct_hits = await direct.select("query", [], k=2)
    baseline_hits = await baseline.select("query", [], k=2)
    assert baseline_hits == direct_hits


async def test_disabled_mode_does_not_call_adapter_or_change_ranking() -> None:
    adapter = _CapturingAdapter(reverse=True)
    source = _Source([_hit("local/first", "first", 2.0), _hit("local/second", "second", 1.0)])
    router = SkillForgeRouter(
        [source],
        decision_plane_enabled=False,
        decision_adapter=adapter,
    )
    diagnostics: dict = {}
    hits = await router.select("query", [], k=2, diagnostics=diagnostics)
    assert [hit.qualified_id for hit in hits] == ["local/first", "local/second"]
    assert adapter.calls == 0
    assert "decision_plane" not in diagnostics


async def test_adapter_receives_only_router_owned_candidate_projection() -> None:
    adapter = _CapturingAdapter(reverse=True)
    source = _Source([_hit("local/first", "first", 2.0), _hit("local/second", "second", 1.0)])
    router = SkillForgeRouter([source], decision_plane_enabled=True, decision_adapter=adapter)
    hits = await router.select("query", [], k=2)
    assert [hit.qualified_id for hit in hits] == ["local/second", "local/first"]
    assert adapter.request is not None
    assert [candidate.candidate_id for candidate in adapter.request.candidates] == [
        "local/first",
        "local/second",
    ]
    assert not hasattr(adapter.request.candidates[0], "content")


@pytest.mark.parametrize(
    ("adapter", "reason"),
    [
        (_UnknownAdapter(), "unknown_candidate_id"),
        (_DuplicateAdapter(), "duplicate_candidate_id"),
        (_IncompleteAdapter(), "incomplete_candidate_set"),
        (_TimeoutAdapter(), "adapter_timeout"),
        (_FailingAdapter(), "adapter_exception:RuntimeError"),
        (_MalformedAdapter(), "adapter_exception:AttributeError"),
    ],
)
async def test_invalid_or_failed_adapter_falls_back_to_deterministic_order(adapter, reason) -> None:
    source = _Source([_hit("local/first", "first", 2.0), _hit("local/second", "second", 1.0)])
    router = SkillForgeRouter([source], decision_plane_enabled=True, decision_adapter=adapter)
    diagnostics: dict = {}
    hits = await router.select("query", [], k=2, diagnostics=diagnostics)
    assert [hit.qualified_id for hit in hits] == ["local/first", "local/second"]
    assert diagnostics["decision_plane"]["used_fallback"] is True
    assert diagnostics["decision_plane"]["fallback_reason"] == reason


async def test_ranking_cannot_force_resolver_activation(tmp_path: Path) -> None:
    skill_dir = tmp_path / "deploy-playbook"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("kubernetes validation steps", encoding="utf-8")
    file_skill = RouterHit(
        qualified_id="local/deploy-playbook",
        name="deploy-playbook",
        content="kubernetes validation steps",
        score=1.0,
        meta={
            "source": "local",
            "description": "release workflow",
            "skill_dir": str(skill_dir),
        },
    )
    adapter = _CapturingAdapter(reverse=True)
    router = SkillForgeRouter(
        [_Source([file_skill])],
        decision_plane_enabled=True,
        decision_adapter=adapter,
    )
    resolution = await LocalSkillResolver(router).resolve("kubernetes help", [])
    assert resolution.activated == ()
    assert [hit.qualified_id for hit in resolution.references] == ["local/deploy-playbook"]
