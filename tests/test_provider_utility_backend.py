from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live.jev_experiment import (
    JevUtilityExperimentDesign,
)
from benchmarks.picobench.packs.knowledge_evolution_live.metrics import (
    extract_utility_observability,
)
from pico.config.pico import ContextConfig
from pico.decision_plane import provider_utility_smoke
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_INSTRUCTION,
    PROVIDER_UTILITY_PROMPT_DIGEST,
    ProviderUtilityBackend,
    ProviderUtilityConfig,
)
from pico.decision_plane.types import DecisionOutcome
from pico.decision_plane.utility import (
    JevUtilityCandidate,
    JevUtilityDecisionAdapter,
    JevUtilityDecisionReceipt,
    JevUtilityRequest,
    UtilityDecision,
)
from pico.providers.base import ErrorClassification, LLMProvider, LLMResponse
from pico.tracing import evidence


def _request() -> JevUtilityRequest:
    return JevUtilityRequest(
        decision_id="utility:provider:test",
        turn_id="turn-provider",
        repository_scope_id="a" * 64,
        query="repair bounded checkpoint recovery",
        query_digest="b" * 64,
        candidates=(
            JevUtilityCandidate("long-private-id-a", "memory_fact", "Guard", "Fail closed.", 1, 0.9, "c" * 64),
            JevUtilityCandidate("long-private-id-b", "skill_candidate", "Replay", "Verify evidence.", 2, 0.8, "d" * 64),
        ),
    )


class _Provider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, contents: list[str | LLMResponse | BaseException], *, delay: float = 0.0) -> None:
        super().__init__()
        self.contents = list(contents)
        self.delay = delay
        self.calls: list[dict] = []

    def get_default_model(self) -> str:
        return "deepseek/test-model"

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "model": model, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        value = self.contents.pop(0)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, LLMResponse):
            return value
        return LLMResponse(
            content=value,
            usage={"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
            model=model,
        )


def _content(*choices: tuple[str, str, float | None]) -> str:
    decisions = []
    for alias, choice, confidence in choices:
        item = {"candidate": alias, "choice": choice}
        if confidence is not None:
            item["confidence"] = confidence
        decisions.append(item)
    return json.dumps({"decisions": decisions}, separators=(",", ":"))


async def _decide(provider: LLMProvider, *, timeout: float = 0.1):
    backend = ProviderUtilityBackend(
        provider,
        ProviderUtilityConfig(provider_id="deepseek", model_id="deepseek/test-model", max_candidates=8),
    )
    return await JevUtilityDecisionAdapter(backend, timeout_seconds=timeout).decide(_request()), backend


@pytest.mark.asyncio
async def test_provider_message_is_bounded_deterministic_aliased_and_tool_free() -> None:
    provider = _Provider([_content(("candidate_0", "KEEP", 0.81), ("candidate_1", "ABSTAIN", 0.74))])
    result, backend = await _decide(provider)
    assert result.outcome is DecisionOutcome.SUCCESS
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["tools"] is None
    assert call["model"] == "deepseek/test-model"
    assert call["messages"][0]["content"] == PROVIDER_UTILITY_INSTRUCTION
    state = json.loads(call["messages"][1]["content"])
    assert state["choices"] == ["KEEP", "ABSTAIN", "UNCERTAIN"]
    assert [item["candidate"] for item in state["candidates"]] == ["candidate_0", "candidate_1"]
    serialized = json.dumps(call)
    assert "long-private-id" not in serialized
    assert all(value not in serialized for value in ("tool_definitions", "hidden verifier", "benchmark outcome", "nav-02"))
    assert backend.prompt_digest == PROVIDER_UTILITY_PROMPT_DIGEST


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choices", "raw", "effective", "retained"),
    [
        (("KEEP", "KEEP"), ("keep", "keep"), ("keep", "keep"), 2),
        (("KEEP", "ABSTAIN"), ("keep", "abstain"), ("keep", "abstain"), 1),
        (("ABSTAIN", "ABSTAIN"), ("abstain", "abstain"), ("abstain", "abstain"), 0),
        (("UNCERTAIN", "UNCERTAIN"), ("uncertain", "uncertain"), ("keep", "keep"), 2),
        (("KEEP", "UNCERTAIN"), ("keep", "uncertain"), ("keep", "keep"), 2),
    ],
)
async def test_typed_choices_and_local_uncertainty_policy(choices, raw, effective, retained) -> None:
    provider = _Provider([_content(("candidate_0", choices[0], 0.01), ("candidate_1", choices[1], 0.99))])
    result, _ = await _decide(provider)
    assert tuple(item.decision.value for item in result.decisions) == raw
    assert tuple(item.effective_decision.value for item in result.decisions) == effective
    assert sum(item.effective_decision is UtilityDecision.KEEP for item in result.decisions) == retained
    assert tuple(item.confidence for item in result.decisions) == (0.01, 0.99)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("not-json", "malformed_response"),
        ('{"decisions":[],"prose":"extra"}', "malformed_response"),
        (_content(("candidate_9", "KEEP", None), ("candidate_1", "KEEP", None)), "unexpected_candidate_alias"),
        (_content(("candidate_0", "KEEP", None), ("candidate_0", "KEEP", None)), "duplicate_candidate_alias"),
        (_content(("candidate_0", "KEEP", None)), "missing_candidate_alias"),
        (_content(("candidate_0", "PROMOTE", None), ("candidate_1", "KEEP", None)), "invalid_choice"),
        (_content(("candidate_0", "KEEP", -0.1), ("candidate_1", "KEEP", None)), "invalid_confidence"),
        (_content(("candidate_0", "KEEP", 1.1), ("candidate_1", "KEEP", None)), "invalid_confidence"),
        (_content(("candidate_0", "KEEP", float("nan")), ("candidate_1", "KEEP", None)), "invalid_confidence"),
        (_content(("candidate_0", "KEEP", float("inf")), ("candidate_1", "KEEP", None)), "invalid_confidence"),
    ],
)
async def test_strict_response_failures_return_baseline_signal(content: str, reason: str) -> None:
    result, _ = await _decide(_Provider([content]))
    assert result.outcome is DecisionOutcome.INVALID_RESULT
    assert result.reason == reason
    assert result.decisions == ()


@pytest.mark.asyncio
async def test_provider_timeout_and_exception_are_bounded_fallbacks() -> None:
    timeout, _ = await _decide(
        _Provider([_content(("candidate_0", "KEEP", None), ("candidate_1", "KEEP", None))], delay=1),
        timeout=0.001,
    )
    assert timeout.outcome is DecisionOutcome.TIMEOUT
    assert timeout.reason == "backend_timeout"

    failed, _ = await _decide(_Provider([RuntimeError("secret-bearing provider failure")]))
    assert failed.outcome is DecisionOutcome.UNAVAILABLE
    assert failed.reason == "provider_error"
    assert "secret-bearing" not in repr(failed)


@pytest.mark.asyncio
async def test_existing_provider_retry_owns_attempts_and_utility_has_one_logical_call() -> None:
    retryable = LLMResponse(
        content="temporary",
        finish_reason="error",
        error_classification=ErrorClassification("server", retryable=True),
    )
    provider = _Provider(
        [retryable, _content(("candidate_0", "KEEP", None), ("candidate_1", "KEEP", None))]
    )
    result, _ = await _decide(provider)
    assert result.outcome is DecisionOutcome.SUCCESS
    assert result.logical_calls == 1
    assert result.provider_attempts == 2
    assert len(provider.calls) == 2
    assert result.input_tokens == 17
    assert result.output_tokens == 5


@pytest.mark.asyncio
async def test_provider_attempt_receipts_are_tagged_as_utility_not_main_agent() -> None:
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id="turn-provider",
        conversation_id="conversation",
        trace_id="trace",
        root_span_id="span",
        writer=lambda record: records.append(record) is None,
    )
    provider = _Provider(
        [_content(("candidate_0", "KEEP", None), ("candidate_1", "UNCERTAIN", None))]
    )
    with evidence.turn_scope(recorder):
        result, _ = await _decide(provider)
    starts = [item for item in records if item["event_type"] == evidence.PROVIDER_ATTEMPT_STARTED]
    assert len(starts) == 1
    assert starts[0]["metadata"]["call_role"] == "utility"
    assert result.logical_call_id == starts[0]["correlations"]["logical_call_id"]
    events = tuple(
        SimpleNamespace(
            event_type=item["event_type"],
            metadata=item["metadata"],
            correlations=item["correlations"],
        )
        for item in records
    )
    projected = extract_utility_observability(events)
    assert projected["utility_provider_calls"] == 1
    assert projected["utility_provider_attempts"] == 1


@pytest.mark.asyncio
async def test_max_candidate_bound_falls_back_without_provider_invocation() -> None:
    provider = _Provider(["unused"])
    backend = ProviderUtilityBackend(provider, ProviderUtilityConfig(max_candidates=1))
    result = await JevUtilityDecisionAdapter(backend).decide(_request())
    assert result.reason == "max_candidates_exceeded"
    assert provider.calls == []


def test_provider_receipt_is_bounded_and_separates_utility_metrics() -> None:
    receipt = JevUtilityDecisionReceipt(
        turn_id="turn-provider",
        decision_id="utility:provider:test",
        repository_scope_id="a" * 64,
        backend_id="provider:deepseek",
        backend_model="deepseek/test-model",
        backend_version="1",
        config_digest="b" * 64,
        request_digest="c" * 64,
        input_candidate_ids=("a",),
        retained_candidate_ids=("a",),
        abstained_candidate_ids=(),
        decisions=(),
        latency_ms=12.0,
        fallback_used=False,
        fallback_reason=None,
        utility_logical_calls=1,
        utility_provider_attempts=2,
        utility_input_tokens=17,
        utility_output_tokens=5,
        utility_provider_latency_ms=11.0,
        utility_logical_call_id="provider-call:utility",
    )
    metadata = receipt.metadata()
    assert "prompt" not in metadata and "response" not in metadata and "api_key" not in metadata
    assert metadata["backend"] == "provider"
    assert metadata["provider_id"] == "deepseek"
    projected = extract_utility_observability(
        (SimpleNamespace(event_type=evidence.DECISION_RECEIPT, metadata=metadata),)
    )
    assert projected["utility_provider_calls"] == 1
    assert projected["utility_provider_attempts"] == 2
    assert projected["utility_input_tokens"] == 17
    assert projected["utility_output_tokens"] == 5
    assert projected["utility_logical_call_ids"] == ("provider-call:utility",)


def test_provider_backend_is_explicit_and_default_remains_disabled() -> None:
    default = ContextConfig()
    assert default.knowledge_selection_mode == "legacy_applicable"
    assert default.knowledge_utility_backend == "injected"
    enabled = ContextConfig(
        knowledge_selection_mode="task_relevance_v1_jev_utility",
        knowledge_utility_backend="provider",
        knowledge_utility_provider="agent",
        knowledge_utility_model="deepseek/test-model",
    )
    assert enabled.knowledge_utility_backend == "provider"


def test_future_provider_arm_identity_is_secret_free_and_complete() -> None:
    design = JevUtilityExperimentDesign()
    identity = design.treatment_identity(
        provider_id="deepseek",
        model_id="deepseek/test-model",
        timeout_seconds=2.0,
        max_candidates=16,
    )
    assert identity["utility_backend"] == "provider"
    assert identity["typed_choice_policy_version"] == 1
    assert identity["prompt_template_digest"] == PROVIDER_UTILITY_PROMPT_DIGEST
    assert len(design.treatment_config_digest(**{
        "provider_id": "deepseek",
        "model_id": "deepseek/test-model",
        "timeout_seconds": 2.0,
        "max_candidates": 16,
    })) == 64
    assert "api_key" not in json.dumps(identity)


def _deepseek_config():
    from pico.config.schema import Config

    config = Config()
    config.agents.defaults.provider = "deepseek"
    config.agents.defaults.model = "deepseek/deepseek-v4-flash"
    config.providers.deepseek.api_key = "test-secret-never-rendered"
    return config


def test_smoke_dry_run_resolves_registry_identity_and_never_builds_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _deepseek_config()
    monkeypatch.setattr(
        provider_utility_smoke,
        "load_pico_config",
        lambda: SimpleNamespace(base=config, context=ContextConfig()),
    )
    monkeypatch.setattr(
        provider_utility_smoke,
        "make_provider",
        lambda _config: pytest.fail("dry-run must not construct or invoke a Provider"),
    )
    assert provider_utility_smoke.main(["--provider", "deepseek", "--timeout", "15"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "candidate_count": 1,
        "live_execution": False,
        "mode": "dry_run",
        "model_id": "deepseek/deepseek-v4-flash",
        "normal_agent_turn": False,
        "provider_id": "deepseek",
        "provider_invocation_count": 0,
        "purpose": "contract_validation_only",
        "schema": "pico.jev3-provider-utility-smoke.v1",
        "tools_enabled": False,
        "typesafe_enabled": False,
        "utility_backend": "provider",
        "utility_timeout_seconds": 15.0,
        "workspace_mutation": False,
    }
    assert "test-secret" not in output


def test_smoke_rejects_unresolved_or_mismatched_provider_model() -> None:
    config = _deepseek_config()
    context = ContextConfig()
    with pytest.raises(ValueError, match="does not match"):
        provider_utility_smoke.resolve_smoke_plan(
            config,
            context,
            requested_provider="openai",
            requested_model="deepseek/deepseek-v4-flash",
            timeout_seconds=15,
        )
    with pytest.raises(ValueError, match="does not resolve"):
        provider_utility_smoke.resolve_smoke_plan(
            config,
            context,
            requested_provider="deepseek",
            requested_model="unknown-model",
            timeout_seconds=15,
        )


@pytest.mark.asyncio
async def test_smoke_live_fixture_is_one_tool_free_logical_call_with_in_memory_receipt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = _Provider([_content(("candidate_0", "UNCERTAIN", 0.9))])
    monkeypatch.setattr(provider_utility_smoke, "make_provider", lambda _config: provider)
    plan = provider_utility_smoke.resolve_smoke_plan(
        _deepseek_config(),
        ContextConfig(),
        requested_provider="deepseek",
        requested_model="deepseek/deepseek-v4-flash",
        timeout_seconds=15,
    )
    assert await provider_utility_smoke._run_live(_deepseek_config(), plan) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider_logical_calls"] == 1
    assert payload["provider_attempts"] == 1
    assert payload["receipt_produced"] is True
    assert payload["decisions"][0]["choice"] == "uncertain"
    assert payload["decisions"][0]["effective_decision"] == "keep"
    assert len(provider.calls) == 1
    assert provider.calls[0]["tools"] is None
