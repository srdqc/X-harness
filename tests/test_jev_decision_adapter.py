from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pico.agent.context import ContextBuilder
from pico.config.pico import SkillForgeRouterConfig
from pico.context_engine.factory import _build_router
from pico.decision_plane import (
    DecisionAdapter,
    DecisionCandidate,
    DecisionCost,
    DecisionOutcome,
    DecisionRequest,
    DecisionType,
    JevDecisionAdapter,
)
from pico.decision_plane.fake import JevFakeMode, ScriptedJevBackend
from pico.decision_plane.jev import JevBackendResponse, JevCostMetadata, JevRankingAdvice
from pico.memory_engine.skill_forge import RouterHit, SkillForgeRouter
from pico.tracing import evidence


def _request(*, decision_id: str = "decision:test:1") -> DecisionRequest:
    return DecisionRequest(
        decision_type=DecisionType.SKILL_RANKING,
        query="deploy the private service",
        candidates=(
            DecisionCandidate("local/deploy", "deploy", "private deploy body", "local", 1, 0.9),
            DecisionCandidate("local/test", "test", "private test body", "local", 2, 0.5),
        ),
        decision_id=decision_id,
    )


def _hit(candidate_id: str, name: str, score: float) -> RouterHit:
    return RouterHit(
        qualified_id=candidate_id,
        name=name,
        content=f"SECRET FULL BODY FOR {name}",
        score=score,
        meta={"source": "local", "description": f"use {name}"},
    )


class _Source:
    name = "local"
    weight = 1.0

    def __init__(self) -> None:
        self.hits = [_hit("local/deploy", "deploy", 2.0), _hit("local/test", "test", 1.0)]

    async def search(self, query, history, k):
        del query, history
        return self.hits[:k]


class _ResponseBackend:
    def __init__(self, response: object) -> None:
        self.response = response

    async def rank_skill_candidates(self, request: DecisionRequest) -> JevBackendResponse:
        del request
        return self.response  # type: ignore[return-value]


def _response(request: DecisionRequest) -> JevBackendResponse:
    return JevBackendResponse(
        decision_id=request.decision_id,
        decision_type=request.decision_type,
        request_digest=request.request_digest,
        ranking=tuple(JevRankingAdvice(item.candidate_id) for item in reversed(request.candidates)),
    )


def _router(backend: ScriptedJevBackend | None, *, enabled: bool = True, timeout: float = 0.05):
    return SkillForgeRouter(
        [_Source()],
        decision_plane_enabled=enabled,
        decision_adapter=JevDecisionAdapter(backend, timeout_seconds=timeout),
    )


async def test_jev_adapter_protocol_and_valid_fake_ranking() -> None:
    backend = ScriptedJevBackend()
    adapter = JevDecisionAdapter(backend)
    assert isinstance(adapter, DecisionAdapter)

    request = _request()
    result = await adapter.decide(request)
    assert result.outcome is DecisionOutcome.SUCCESS
    assert [item.candidate_id for item in result.ranking] == ["local/test", "local/deploy"]
    assert result.decision_id == request.decision_id
    assert result.request_digest == request.request_digest


async def test_disabled_router_never_calls_jev_backend() -> None:
    backend = ScriptedJevBackend()
    hits = await _router(backend, enabled=False).select("private query", [], k=2)
    assert backend.calls == 0
    assert [hit.qualified_id for hit in hits] == ["local/deploy", "local/test"]


@pytest.mark.parametrize(
    ("mode", "outcome", "reason"),
    [
        (JevFakeMode.UNAVAILABLE, DecisionOutcome.UNAVAILABLE, "backend_unavailable"),
        (JevFakeMode.EXCEPTION, DecisionOutcome.ERROR, "backend_exception"),
        (JevFakeMode.MALFORMED, DecisionOutcome.INVALID_RESULT, "malformed_response"),
        (JevFakeMode.WRONG_DECISION_TYPE, DecisionOutcome.INVALID_RESULT, "decision_type_mismatch"),
        (JevFakeMode.UNKNOWN_CANDIDATE, DecisionOutcome.INVALID_RESULT, "unknown_candidate_id"),
        (JevFakeMode.DUPLICATE_CANDIDATE, DecisionOutcome.INVALID_RESULT, "duplicate_candidate_id"),
        (JevFakeMode.INCOMPLETE_RANKING, DecisionOutcome.INVALID_RESULT, "incomplete_candidate_set"),
        (JevFakeMode.INVALID_SCORE, DecisionOutcome.INVALID_RESULT, "invalid_score"),
        (JevFakeMode.INVALID_CONFIDENCE, DecisionOutcome.INVALID_RESULT, "invalid_confidence"),
    ],
)
async def test_backend_failures_are_structured_and_fall_back(mode, outcome, reason) -> None:
    backend = ScriptedJevBackend(mode)
    diagnostics: dict = {}
    hits = await _router(backend).select("private query", [], k=2, diagnostics=diagnostics)
    assert [hit.qualified_id for hit in hits] == ["local/deploy", "local/test"]
    assert diagnostics["decision_plane"]["adapter_outcome"] == outcome.value
    assert diagnostics["decision_plane"]["fallback_used"] is True
    assert diagnostics["decision_plane"]["fallback_reason"] == reason


async def test_missing_backend_falls_back_without_failing_routing() -> None:
    diagnostics: dict = {}
    hits = await _router(None).select("query", [], k=2, diagnostics=diagnostics)
    assert [hit.qualified_id for hit in hits] == ["local/deploy", "local/test"]
    assert diagnostics["decision_plane"]["adapter_outcome"] == "unavailable"
    assert diagnostics["decision_plane"]["fallback_reason"] == "backend_unavailable"


async def test_host_timeout_cancels_backend_and_falls_back() -> None:
    backend = ScriptedJevBackend(JevFakeMode.TIMEOUT)
    diagnostics: dict = {}
    hits = await _router(backend, timeout=0.001).select("query", [], k=2, diagnostics=diagnostics)
    assert [hit.qualified_id for hit in hits] == ["local/deploy", "local/test"]
    assert backend.cancelled is True
    assert diagnostics["decision_plane"]["adapter_outcome"] == "timeout"
    assert diagnostics["decision_plane"]["fallback_reason"] == "backend_timeout"


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda response: replace(response, decision_id="wrong"), "decision_id_mismatch"),
        (lambda response: replace(response, request_digest="wrong"), "request_digest_mismatch"),
        (lambda response: replace(response, schema_version=999), "unsupported_schema"),
    ],
)
async def test_identity_digest_and_schema_mismatch_are_rejected(mutator, reason) -> None:
    request = _request()
    adapter = JevDecisionAdapter(_ResponseBackend(mutator(_response(request))))
    result = await adapter.decide(request)
    assert result.outcome is DecisionOutcome.INVALID_RESULT
    assert result.reason == reason


async def test_valid_confidence_and_real_cost_are_preserved() -> None:
    backend = ScriptedJevBackend(
        confidence=0.75,
        cost=JevCostMetadata(available=True, amount=1.25, unit="credits"),
    )
    result = await JevDecisionAdapter(backend).decide(_request())
    assert [item.confidence for item in result.ranking] == [0.75, 0.75]
    assert result.cost == DecisionCost(available=True, amount=1.25, unit="credits")


async def test_receipt_is_turn_correlated_private_and_sequenced() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-jev",
        conversation_id="conversation-1",
        trace_id="trace-1",
        root_span_id="span-1",
        writer=lambda record: records.append(record) is None,
    )
    diagnostics: dict = {}
    backend = ScriptedJevBackend(
        confidence=0.8,
        cost=JevCostMetadata(available=True, amount=2.0, unit="credits"),
    )
    with evidence.turn_scope(recorder):
        hits = await _router(backend).select(
            "private query must not persist",
            [],
            k=2,
            diagnostics=diagnostics,
        )

    assert [hit.qualified_id for hit in hits] == ["local/test", "local/deploy"]
    assert len(records) == 1
    record = records[0]
    assert record["event_type"] == evidence.DECISION_RECEIPT
    assert record["turn_id"] == "turn-jev"
    assert record["correlations"]["decision_id"] == "turn-jev:decision:1"
    assert record["metadata"]["turn_id"] == "turn-jev"
    assert record["metadata"]["fallback_used"] is False
    assert record["metadata"]["cost_available"] is True
    assert record["metadata"]["confidences"] == [["local/test", 0.8], ["local/deploy", 0.8]]
    assert record["metadata"]["latency_ms"] >= 0
    persisted = json.dumps(record)
    assert "private query must not persist" not in persisted
    assert "SECRET FULL BODY" not in persisted


async def test_fallback_receipt_is_durable_and_cost_absence_is_explicit() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-fallback",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    with evidence.turn_scope(recorder):
        await _router(ScriptedJevBackend(JevFakeMode.MALFORMED)).select("query", [], k=2)
    metadata = records[0]["metadata"]
    assert metadata["adapter_outcome"] == "invalid_result"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "malformed_response"
    assert metadata["final_ranking_source"] == "deterministic"
    assert metadata["cost_available"] is False
    assert metadata["cost_amount"] is None
    assert metadata["cost_unit"] is None


async def test_standalone_correlation_is_explicit_and_decisions_are_unique() -> None:
    router = _router(ScriptedJevBackend())
    first: dict = {}
    second: dict = {}
    await router.select("query", [], k=2, diagnostics=first)
    await router.select("query", [], k=2, diagnostics=second)
    first_receipt = first["decision_plane"]
    second_receipt = second["decision_plane"]
    assert first_receipt["turn_id"] is None
    assert second_receipt["turn_id"] is None
    assert first_receipt["decision_id"] != second_receipt["decision_id"]
    assert first_receipt["candidate_set_digest"] == second_receipt["candidate_set_digest"]
    assert first_receipt["request_digest"] == second_receipt["request_digest"]


async def test_multiple_turn_scoped_decisions_get_distinct_local_identities() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-multiple",
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    router = _router(ScriptedJevBackend())
    with evidence.turn_scope(recorder):
        await router.select("query", [], k=2)
        await router.select("query", [], k=2)
    assert [record["correlations"]["decision_id"] for record in records] == [
        "turn-multiple:decision:1",
        "turn-multiple:decision:2",
    ]
    assert [record["sequence"] for record in records] == [1, 2]


async def test_same_request_produces_deterministic_result_digest() -> None:
    request = _request()
    adapter = JevDecisionAdapter(ScriptedJevBackend())
    first = await adapter.decide(request)
    second = await adapter.decide(request)
    assert first.result_digest == second.result_digest


def test_jev_config_is_optional_bounded_and_disabled_by_default() -> None:
    default = SkillForgeRouterConfig()
    assert default.decision_plane_enabled is False
    assert default.decision_adapter == "deterministic"
    assert default.decision_timeout_seconds == 0.25
    configured = SkillForgeRouterConfig(
        decision_plane_enabled=True,
        decision_adapter="jev",
        decision_timeout_seconds=1.0,
    )
    assert configured.decision_adapter == "jev"
    with pytest.raises(ValueError):
        SkillForgeRouterConfig(decision_timeout_seconds=True)
    with pytest.raises(ValueError):
        SkillForgeRouterConfig(decision_timeout_seconds=0)


async def test_context_factory_selects_injected_jev_backend(tmp_path) -> None:
    backend = ScriptedJevBackend()
    router = _build_router(
        builder=ContextBuilder(tmp_path),
        skill_forge_router_config=SkillForgeRouterConfig(
            decision_plane_enabled=True,
            decision_adapter="jev",
        ),
        jev_backend=backend,
    )
    await router.select("query", [], k=2)
    assert backend.calls == 1
