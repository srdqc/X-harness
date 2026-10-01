from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pico.agent.loop import AgentLoop
from pico.agent.spine_runner import AgentTurnRunner
from pico.agent.tools.base import Tool, ToolResult
from pico.agent.tools.discovery import ToolSourceKind
from pico.agent.tools.execution import ToolCapability, ToolEffect, ToolExecutionContext, ToolInvocation
from pico.agent.tools.registry import ToolRegistry
from pico.agent.tools.tool_search import TOOL_CALL_NAME, ToolCallTool, ToolSearchController
from pico.call_efficiency import CallEfficiency
from pico.call_efficiency.provider import CallEfficiencyProvider
from pico.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
from pico.spine import ChatType, Origin, OriginPools, Scheduler, Source, TurnRequest
from pico.tracing import evidence
from pico.tracing import spans as tracing_spans


@pytest.fixture
def evidence_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "traces"
    monkeypatch.setenv("PICO_TRACING", "1")
    monkeypatch.setenv("PICO_TRACING_DIR", str(state_dir))
    tracing_spans._store = None
    yield state_dir
    tracing_spans._store = None


def _recorder(turn_id: str = "turn-receipts") -> evidence.TurnEvidenceRecorder:
    return evidence.TurnEvidenceRecorder(
        turn_id=turn_id,
        conversation_id="test:chat",
        trace_id="trace-receipts",
        root_span_id="span-turn",
    )


def _start(recorder: evidence.TurnEvidenceRecorder) -> None:
    recorder.emit(evidence.TURN_STARTED)


def _finish(recorder: evidence.TurnEvidenceRecorder, outcome: str = "completed") -> None:
    recorder.emit(
        evidence.TURN_TERMINAL,
        metadata={"outcome": outcome, "lifecycle_event": "TurnEnded" if outcome == "completed" else "TurnFailed"},
    )


class _ScriptedProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, responses: list[LLMResponse | BaseException]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def get_default_model(self) -> str:
        return "model-default"


def _error(category: str = "network", *, retryable: bool = True, fallback: bool = False) -> LLMResponse:
    return LLMResponse(
        content="secret provider error",
        finish_reason="error",
        error_classification=ErrorClassification(category, retryable=retryable, should_fallback=fallback),
    )


async def test_one_logical_provider_call_has_one_stable_attempt_receipt(evidence_dir):
    recorder = _recorder()
    provider = _ScriptedProvider([LLMResponse(content="private response", usage={"prompt_tokens": 3})])
    _start(recorder)
    with evidence.turn_scope(recorder):
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "private prompt"}],
            model="model-a",
        )
    _finish(recorder)

    result = evidence.read_turn_evidence(evidence_dir, recorder.turn_id)
    attempt = result.provider_attempts[0]
    assert response.content == "private response"
    assert provider.calls == 1
    assert attempt.logical_call_id == "turn-receipts:provider-call:1"
    assert attempt.attempt_id == "turn-receipts:provider-call:1:attempt:1"
    assert attempt.attempt_ordinal == 1
    assert attempt.outcome == "success"
    assert attempt.requested_model == attempt.attempted_model == "model-a"
    assert attempt.usage == {"prompt_tokens": 3}
    assert attempt.complete is True
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE
    persisted = (evidence_dir / "logs" / "audit-events.log").read_text(encoding="utf-8")
    assert "private prompt" not in persisted
    assert "private response" not in persisted


async def test_retry_attempts_share_logical_call_and_success_does_not_fail_turn(evidence_dir):
    recorder = _recorder()
    provider = _ScriptedProvider([_error(), LLMResponse(content="recovered")])
    _start(recorder)
    with evidence.turn_scope(recorder):
        response = await provider.chat_with_retry(messages=[], model="model-a")
    _finish(recorder)

    result = evidence.read_turn_evidence(evidence_dir, recorder.turn_id)
    attempts = result.provider_attempts
    assert response.content == "recovered"
    assert [attempt.attempt_ordinal for attempt in attempts] == [1, 2]
    assert len({attempt.attempt_id for attempt in attempts}) == 2
    assert len({attempt.logical_call_id for attempt in attempts}) == 1
    assert [attempt.outcome for attempt in attempts] == ["error", "success"]
    assert result.events[-1].metadata["outcome"] == "completed"


async def test_new_provider_logical_call_resets_ordinal(evidence_dir):
    recorder = _recorder()
    provider = _ScriptedProvider([LLMResponse(content="one"), LLMResponse(content="two")])
    _start(recorder)
    with evidence.turn_scope(recorder):
        await provider.chat_with_retry(messages=[], model="model-a")
        await provider.chat_with_retry(messages=[], model="model-a")
    _finish(recorder)

    attempts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).provider_attempts
    assert [attempt.attempt_ordinal for attempt in attempts] == [1, 1]
    assert len({attempt.logical_call_id for attempt in attempts}) == 2


async def test_model_fallback_attempts_remain_one_logical_call(evidence_dir):
    recorder = _recorder()
    provider = _ScriptedProvider(
        [_error("model_unavailable", retryable=False, fallback=True), LLMResponse(content="fallback worked")]
    )
    _start(recorder)
    with evidence.turn_scope(recorder):
        response = await provider.chat_with_retry(
            messages=[],
            model="model-a",
            fallback_models=["model-b"],
        )
    _finish(recorder)

    attempts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).provider_attempts
    assert response.content == "fallback worked"
    assert [attempt.attempted_model for attempt in attempts] == ["model-a", "model-b"]
    assert [attempt.attempt_ordinal for attempt in attempts] == [1, 2]
    assert len({attempt.logical_call_id for attempt in attempts}) == 1


async def test_terminal_provider_failure_and_unavailable_usage_survive_restart(evidence_dir):
    recorder = _recorder()
    provider = _ScriptedProvider([_error("auth", retryable=False)])
    _start(recorder)
    with evidence.turn_scope(recorder):
        response = await provider.chat_with_retry(messages=[], model="model-a")
    _finish(recorder, "provider_failed")

    restarted = evidence.read_turn_evidence(Path(str(evidence_dir)), recorder.turn_id)
    attempt = restarted.provider_attempts[0]
    assert response.finish_reason == "error"
    assert attempt.outcome == "error"
    assert attempt.error_category == "auth"
    assert attempt.usage_available is False
    assert attempt.usage == {}
    assert restarted.completeness is evidence.EvidenceCompleteness.COMPLETE


def test_request_response_and_argument_digests_are_deterministic():
    left = {"b": [2, 1], "a": {"x": "value"}}
    right = {"a": {"x": "value"}, "b": [2, 1]}
    assert evidence.canonical_digest(left) == evidence.canonical_digest(right)
    assert evidence.canonical_digest(left) != evidence.canonical_digest({"a": {"x": "changed"}, "b": [2, 1]})


async def test_call_efficiency_record_joins_provider_attempt_identity(evidence_dir, tmp_path):
    recorder = _recorder()
    delegate = _ScriptedProvider([LLMResponse(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1})])
    controller = CallEfficiency(mode="observe", telemetry_dir=tmp_path, persist=False)
    provider = CallEfficiencyProvider(delegate, controller)
    _start(recorder)
    with evidence.turn_scope(recorder):
        response = await provider.chat_with_retry(messages=[], model="model-a")
    _finish(recorder)

    attempt = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).provider_attempts[0]
    assert response.call_record.turn_id == recorder.turn_id
    assert response.call_record.logical_call_id == attempt.logical_call_id
    assert response.call_record.attempt_id == attempt.attempt_id
    assert response.call_record.attempt_ordinal == attempt.attempt_ordinal


class _StreamingProvider(_ScriptedProvider):
    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        self.calls += 1
        yield StreamDelta(content="hello", model=model)
        yield StreamDelta(content=None, finish_reason="stop", usage={"prompt_tokens": 2}, model=model)


async def test_streaming_provider_produces_attempt_receipt_without_changing_chunks(evidence_dir, tmp_path):
    recorder = _recorder()
    delegate = _StreamingProvider([])
    provider = CallEfficiencyProvider(
        delegate,
        CallEfficiency(mode="observe", telemetry_dir=tmp_path, persist=False),
    )
    _start(recorder)
    with evidence.turn_scope(recorder):
        chunks = [chunk async for chunk in provider.chat_stream(messages=[], model="model-stream")]
    _finish(recorder)

    attempt = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).provider_attempts[0]
    assert [chunk.content for chunk in chunks] == ["hello", None, None]
    assert attempt.attempted_model == "model-stream"
    assert attempt.outcome == "success"
    assert attempt.usage == {"prompt_tokens": 2}


class _ReceiptTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        effect: ToolEffect = ToolEffect.READ,
        result: str | ToolResult = "ok",
        delay: float = 0,
    ) -> None:
        self._name = name
        self.capability = ToolCapability(effect=effect, concurrency_safe=effect is ToolEffect.READ)
        self._result = result
        self._delay = delay
        self.executions = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "receipt test tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, value: str) -> str:
        self.executions += 1
        await asyncio.sleep(self._delay)
        return self._result


async def test_direct_tool_receipt_preserves_requested_resolved_effect_and_digests(evidence_dir):
    recorder = _recorder()
    tool = _ReceiptTool("read_thing")
    registry = ToolRegistry()
    registry.register(tool)
    _start(recorder)
    with evidence.turn_scope(recorder):
        execution = await registry.execute_invocation(
            ToolInvocation(tool.name, {"value": "private argument"}, ToolExecutionContext(call_id="call-1"))
        )
    _finish(recorder)

    receipt = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions[0]
    assert str(execution.result) == "ok"
    assert tool.executions == 1
    assert receipt.model_call_id == "call-1"
    assert receipt.requested_name == receipt.resolved_name == tool.name
    assert receipt.routed_via is None
    assert receipt.effect == "read"
    assert receipt.outcome == "success"
    assert receipt.argument_digest == evidence.canonical_digest({"value": "private argument"})
    assert receipt.result_digest == evidence.canonical_digest("ok")
    persisted = (evidence_dir / "logs" / "audit-events.log").read_text(encoding="utf-8")
    assert "private argument" not in persisted


async def test_meta_routed_receipts_preserve_outer_child_relationship(evidence_dir):
    recorder = _recorder()
    target = _ReceiptTool("write_thing", effect=ToolEffect.WRITE)
    registry = ToolRegistry()
    registry.register(target)
    controller = ToolSearchController(registry, always_visible=set())
    registry.register(ToolCallTool(controller))
    outer = ToolInvocation(
        TOOL_CALL_NAME,
        {"name": target.name, "arguments": {"value": "x"}},
        ToolExecutionContext(call_id="outer-call"),
    )
    _start(recorder)
    with evidence.turn_scope(recorder):
        await registry.execute_invocation(outer)
    _finish(recorder)

    receipts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions
    outer_receipt = next(item for item in receipts if item.requested_name == TOOL_CALL_NAME)
    child = next(item for item in receipts if item.requested_name == target.name)
    assert outer_receipt.model_call_id == "outer-call"
    assert child.model_call_id == "outer-call:write_thing"
    assert child.parent_call_id == "outer-call"
    assert child.routed_via == TOOL_CALL_NAME
    assert child.resolved_name == target.name
    assert child.effect == "write"
    assert target.executions == 1


async def test_unresolved_meta_target_has_no_fabricated_resolved_identity(evidence_dir):
    recorder = _recorder()
    registry = ToolRegistry()
    controller = ToolSearchController(registry, always_visible=set())
    registry.register(ToolCallTool(controller))
    _start(recorder)
    with evidence.turn_scope(recorder):
        result = await registry.execute_invocation(
            ToolInvocation(
                TOOL_CALL_NAME,
                {"name": "stale_tool", "arguments": {}},
                ToolExecutionContext(call_id="outer-stale"),
            )
        )
    _finish(recorder)

    receipts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions
    unresolved = next(item for item in receipts if item.requested_name == "stale_tool")
    assert result.result.failed is True
    assert unresolved.resolved_name is None
    assert unresolved.routed_via == TOOL_CALL_NAME
    assert unresolved.failure_stage == "resolution"
    assert unresolved.failure_category == "not_found"


async def test_schema_validation_and_execution_failures_are_distinct(evidence_dir):
    recorder = _recorder()
    invalid = _ReceiptTool("invalid")
    failed = _ReceiptTool("failed", result=ToolResult("remote failed", failed=True))
    registry = ToolRegistry()
    registry.register(invalid)
    registry.register(failed, source_kind=ToolSourceKind.MCP)
    _start(recorder)
    with evidence.turn_scope(recorder):
        await registry.execute_invocation(ToolInvocation("invalid", {}, ToolExecutionContext(call_id="invalid-1")))
        await registry.execute_invocation(
            ToolInvocation("failed", {"value": "x"}, ToolExecutionContext(call_id="failed-1"))
        )
    _finish(recorder)

    receipts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions
    assert (receipts[0].failure_stage, receipts[0].failure_category) == ("schema_validation", "invalid_arguments")
    assert (receipts[1].failure_stage, receipts[1].failure_category) == ("execution", "remote")


async def test_timeout_is_structured_and_effectful_tool_is_not_replayed(evidence_dir):
    recorder = _recorder()
    tool = _ReceiptTool("external_once", effect=ToolEffect.EXTERNAL, delay=0.05)
    tool.timeout_seconds = 0.001
    registry = ToolRegistry()
    registry.register(tool)
    _start(recorder)
    with evidence.turn_scope(recorder):
        execution = await registry.execute_invocation(
            ToolInvocation(tool.name, {"value": "x"}, ToolExecutionContext(call_id="external-1"))
        )
    _finish(recorder)

    receipt = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions[0]
    assert execution.result.failed is True
    assert receipt.effect == "external"
    assert receipt.failure_stage == receipt.failure_category == "timeout"
    assert tool.executions == 1


@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.EXECUTE, ToolEffect.EXTERNAL])
async def test_effectful_receipts_are_descriptive_and_never_replay(evidence_dir, effect):
    recorder = _recorder(f"turn-{effect.value}")
    tool = _ReceiptTool(f"{effect.value}_once", effect=effect)
    registry = ToolRegistry()
    registry.register(tool)
    _start(recorder)
    with evidence.turn_scope(recorder):
        execution = await registry.execute_invocation(
            ToolInvocation(tool.name, {"value": "x"}, ToolExecutionContext(call_id=f"{effect.value}-1"))
        )
    _finish(recorder)

    receipt = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions[0]
    assert execution.result.failed is False
    assert receipt.effect == effect.value
    assert receipt.outcome == "success"
    assert tool.executions == 1


async def test_duplicate_model_call_ids_still_have_distinct_receipt_ids(evidence_dir):
    recorder = _recorder()
    tool = _ReceiptTool("read_thing")
    registry = ToolRegistry()
    registry.register(tool)
    invocation = ToolInvocation(tool.name, {"value": "x"}, ToolExecutionContext(call_id="duplicate"))
    _start(recorder)
    with evidence.turn_scope(recorder):
        await registry.execute_invocation(invocation)
        await registry.execute_invocation(invocation)
    _finish(recorder)

    receipts = evidence.read_turn_evidence(evidence_dir, recorder.turn_id).tool_executions
    assert [item.model_call_id for item in receipts] == ["duplicate", "duplicate"]
    assert len({item.receipt_id for item in receipts}) == 2
    assert tool.executions == 2


def test_missing_provider_or_tool_completion_marks_evidence_partial(evidence_dir):
    recorder = _recorder()
    _start(recorder)
    recorder.emit(
        evidence.PROVIDER_ATTEMPT_STARTED,
        correlations={"logical_call_id": "call", "attempt_id": "attempt", "attempt_ordinal": 1},
        metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA},
    )
    recorder.emit(
        evidence.TOOL_EXECUTION_STARTED,
        correlations={"receipt_id": "receipt"},
        metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "requested_name": "write_once"},
    )
    _finish(recorder)

    result = evidence.read_turn_evidence(evidence_dir, recorder.turn_id)
    assert result.completeness is evidence.EvidenceCompleteness.PARTIAL
    assert {"incomplete_provider_attempt", "incomplete_tool_execution"} <= set(result.findings)
    assert result.provider_attempts[0].complete is False
    assert result.tool_executions[0].complete is False


def test_receipts_remain_in_turn_sequence_order_after_restart(evidence_dir):
    recorder = _recorder()
    _start(recorder)
    recorder.emit(
        evidence.PROVIDER_ATTEMPT_STARTED,
        correlations={"logical_call_id": "call", "attempt_id": "attempt", "attempt_ordinal": 1},
        metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA},
    )
    recorder.emit(
        evidence.PROVIDER_ATTEMPT_COMPLETED,
        correlations={"logical_call_id": "call", "attempt_id": "attempt", "attempt_ordinal": 1},
        metadata={"receipt_schema": evidence.PROVIDER_RECEIPT_SCHEMA, "outcome": "success"},
    )
    recorder.emit(
        evidence.TOOL_EXECUTION_STARTED,
        correlations={"receipt_id": "receipt", "model_call_id": "tool-call"},
        metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "requested_name": "read"},
    )
    recorder.emit(
        evidence.TOOL_EXECUTION_COMPLETED,
        correlations={"receipt_id": "receipt"},
        metadata={"receipt_schema": evidence.TOOL_RECEIPT_SCHEMA, "outcome": "success"},
    )
    _finish(recorder)

    restarted = evidence.read_turn_evidence(Path(str(evidence_dir)), recorder.turn_id)
    assert restarted.provider_attempts[0].completed_sequence < restarted.tool_executions[0].started_sequence
    assert {event.turn_id for event in restarted.events} == {recorder.turn_id}
    assert restarted.completeness is evidence.EvidenceCompleteness.COMPLETE


async def test_real_scheduler_agent_path_joins_provider_and_tool_receipts_to_runtime_turn(evidence_dir, tmp_path):
    class AgentProvider(LLMProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    finish_reason="tool_calls",
                    tool_calls=[ToolCallRequest(id="model-tool-1", name="list_dir", arguments={"path": "."})],
                )
            return LLMResponse(content="done", finish_reason="stop")

        def get_default_model(self) -> str:
            return "agent-model"

    provider = AgentProvider()
    agent = AgentLoop(
        provider=provider,
        workspace=tmp_path,
        model="agent-model",
        max_iterations=3,
        restrict_to_workspace=True,
    )

    async def sink(_event):
        return None

    scheduler = Scheduler(
        AgentTurnRunner(agent, stream=False),
        OriginPools(user=1, system=1),
        sink,
        turn_id_factory=lambda: "turn-runtime-path",
    )
    request = TurnRequest(
        origin=Origin.USER,
        source=Source(channel="test", chat_id="chat", sender_id="user", chat_type=ChatType.DM),
        text="list the workspace",
    )
    try:
        outcome = await scheduler.submit(request).result()
    finally:
        await scheduler.shutdown(0)

    result = evidence.read_turn_evidence(evidence_dir, "turn-runtime-path")
    assert outcome is not None
    assert provider.calls == 2
    assert len(result.provider_attempts) == 2
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].model_call_id == "model-tool-1"
    assert result.tool_executions[0].resolved_name == "list_dir"
    assert {event.turn_id for event in result.events} == {"turn-runtime-path"}
    assert result.completeness is evidence.EvidenceCompleteness.COMPLETE
