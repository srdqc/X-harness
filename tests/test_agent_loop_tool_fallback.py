from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pico.agent.loop import AgentLoop
from pico.agent.tools.base import Tool, ToolResult
from pico.agent.tools.execution import ToolCapability, ToolEffect
from pico.config.schema import ToolSearchConfig
from pico.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from pico.utils.helpers import estimate_prompt_tokens


class _ScriptProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key="test")
        self.responses = list(responses)
        self.visible_tools: list[list[dict]] = []

    async def chat(self, messages, tools=None, **kwargs):
        del messages, kwargs
        self.visible_tools.append(tools or [])
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "stub"


class _ProbeTool(Tool):
    def __init__(self, name: str, *, effect: ToolEffect = ToolEffect.READ, behavior: str = "ok") -> None:
        self._name = name
        self.capability = ToolCapability(effect=effect)
        self.behavior = behavior
        self.executions = 0
        if behavior == "timeout":
            self.timeout_seconds = 0.001

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"probe {self._name}"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, value: str) -> str:
        self.executions += 1
        if self.behavior == "timeout":
            await asyncio.sleep(1)
        if self.behavior == "failure":
            return ToolResult("remote execution failed", failed=True)
        return value


def _response(name: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def _loop(tmp_path: Path, provider: LLMProvider, *, max_iterations: int = 6) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        workspace=tmp_path,
        model="stub",
        max_iterations=max_iterations,
        restrict_to_workspace=True,
        tool_search_config=ToolSearchConfig(enabled=True, compaction_threshold=5),
    )


async def _run(loop: AgentLoop):
    view = loop._effective_tool_disclosure_view()
    disclosure: list[dict] = []
    fallback = []
    result = await loop._run_agent_loop(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "go"}],
        initial_disclosure_view=view,
        disclosure_evidence_sink=disclosure,
        fallback_evidence_sink=fallback,
    )
    return result, disclosure, fallback[0]


@pytest.mark.asyncio
async def test_successful_search_does_not_activate_fallback(tmp_path: Path) -> None:
    provider = _ScriptProvider(
        [
            _response("tool_search", {"query": "read file"}, "search-1"),
            LLMResponse(content="done"),
        ]
    )
    loop = _loop(tmp_path, provider)

    _, evidence, fallback = await _run(loop)

    assert fallback.fallback_used is False
    assert [item["mode"] for item in evidence] == ["progressive", "progressive"]


@pytest.mark.asyncio
async def test_fail_open_full_does_not_consume_fallback(tmp_path: Path) -> None:
    provider = _ScriptProvider([LLMResponse(content="done")])
    loop = _loop(tmp_path, provider)
    loop.tools.unregister("tool_search")

    _, evidence, fallback = await _run(loop)

    assert fallback.fallback_used is False
    assert fallback.provider_calls_before_fallback == 1
    assert evidence[0]["mode"] == "fail_open_full"


@pytest.mark.asyncio
async def test_zero_hit_fallback_is_once_sticky_full_and_resets_next_turn(tmp_path: Path) -> None:
    provider = _ScriptProvider(
        [
            _response("tool_search", {"query": "zzzznomatch"}, "search-1"),
            _response("tool_search", {"query": "zzzznomatch"}, "search-2"),
            _response("tool_search", {"query": "zzzznomatch"}, "search-3"),
            LLMResponse(content="done"),
        ]
    )
    loop = _loop(tmp_path, provider)

    (_, tools_used, _, _), evidence, fallback = await _run(loop)

    assert fallback.fallback_used is True
    assert fallback.fallback_reason == "repeated_zero_hit_tool_search"
    assert fallback.activation_iteration == 2
    assert fallback.zero_hits_before_fallback == 2
    assert fallback.provider_calls_after_fallback == 2
    assert fallback.recovery_succeeded is True
    assert [item["mode"] for item in evidence] == [
        "progressive",
        "progressive",
        "fallback_full",
        "fallback_full",
    ]
    assert provider.visible_tools[2] == loop.tools.get_definitions()
    assert evidence[2]["tool_array_schema_tokens"] == estimate_prompt_tokens(
        [], loop.tools.get_definitions()
    )
    assert evidence[2]["available_history_tokens"] == loop._make_token_budget(
        tool_definitions=loop.tools.get_definitions()
    ).available_history
    assert tools_used == ["tool_search", "tool_search", "tool_search"]

    second_provider = _ScriptProvider([LLMResponse(content="next")])
    loop.replace_provider(second_provider)
    _, second_evidence, second_fallback = await _run(loop)
    assert second_fallback.fallback_used is False
    assert second_evidence[0]["mode"] == "progressive"


@pytest.mark.asyncio
async def test_unknown_target_activates_visibility_only_without_replay(tmp_path: Path) -> None:
    provider = _ScriptProvider(
        [
            _response(
                "tool_call",
                {"name": "stale_missing_tool", "arguments": {}},
                "call-1",
            ),
            LLMResponse(content="recovered"),
        ]
    )
    loop = _loop(tmp_path, provider)

    (_, tools_used, _, _), evidence, fallback = await _run(loop)

    assert fallback.fallback_reason == "unknown_tool_call_target"
    assert fallback.activation_iteration == 1
    assert [item["mode"] for item in evidence] == ["progressive", "fallback_full"]
    assert tools_used == ["tool_call"]


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["timeout", "failure"])
async def test_resolved_execution_timeout_or_remote_failure_does_not_fallback(
    tmp_path: Path,
    behavior: str,
) -> None:
    target = _ProbeTool(f"hidden_{behavior}", behavior=behavior)
    provider = _ScriptProvider(
        [
            _response(
                "tool_call",
                {"name": target.name, "arguments": {"value": "x"}},
                "call-1",
            ),
            LLMResponse(content="done"),
        ]
    )
    loop = _loop(tmp_path, provider)
    loop.tools.register(target)

    _, evidence, fallback = await _run(loop)

    assert fallback.fallback_used is False
    assert [item["mode"] for item in evidence] == ["progressive", "progressive"]
    assert target.executions == 1


@pytest.mark.asyncio
async def test_validation_and_write_failure_do_not_fallback_or_replay(tmp_path: Path) -> None:
    validation_target = _ProbeTool("hidden_validation")
    write_target = _ProbeTool("hidden_write", effect=ToolEffect.WRITE, behavior="failure")
    provider = _ScriptProvider(
        [
            _response(
                "tool_call",
                {"name": validation_target.name, "arguments": {}},
                "validation-1",
            ),
            _response(
                "tool_call",
                {"name": write_target.name, "arguments": {"value": "x"}},
                "write-1",
            ),
            LLMResponse(content="done"),
        ]
    )
    loop = _loop(tmp_path, provider)
    loop.tools.register(validation_target)
    loop.tools.register(write_target)

    (_, tools_used, _, _), evidence, fallback = await _run(loop)

    assert fallback.fallback_used is False
    assert all(item["mode"] == "progressive" for item in evidence)
    assert validation_target.executions == 0
    assert write_target.executions == 1
    assert tools_used == ["tool_call", "tool_call"]
