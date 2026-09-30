"""Wiring tests for tool-search registration inside AgentLoop.__init__.

Covers the ``_register_default_tools`` block: the meta-tools land in the
registry and the strategy is inserted *first* (so it filters before
CacheOptimizer marks the final tool), and the whole thing is a no-op when the
feature is disabled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pico.agent.loop import AgentLoop
from pico.agent.tools.discovery import ToolSourceKind
from pico.config.schema import ToolSearchConfig
from pico.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from pico.token_wise.base import TokenStrategy
from pico.token_wise.registry import StrategyRegistry
from pico.utils.helpers import estimate_prompt_tokens


class _StubProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.visible_tools: list[list[dict]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.visible_tools.append(tools or [])
        return LLMResponse(content="stub", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


class _MarkerStrategy(TokenStrategy):
    @property
    def name(self) -> str:
        return "marker"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_loop(workspace: Path, cfg, strategies=None, provider=None) -> AgentLoop:
    return AgentLoop(
        provider=provider or _StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        tool_search_config=cfg,
        strategies=strategies,
    )


def test_enabled_registers_meta_tools(workspace) -> None:
    loop = _make_loop(workspace, ToolSearchConfig(enabled=True))
    for name in ("tool_search", "tool_call"):
        assert loop.tools.has(name), f"{name} should be registered"
    assert loop.strategies.get("tool_search") is not None
    builtin = loop.tools.discovery_metadata("read_file")
    meta = loop.tools.discovery_metadata("tool_search")
    assert builtin is not None and builtin.source_kind is ToolSourceKind.BUILTIN
    assert meta is not None and meta.source_kind is ToolSourceKind.META


def test_strategy_registered_first(workspace) -> None:

    registry = StrategyRegistry([_MarkerStrategy()])
    loop = _make_loop(workspace, ToolSearchConfig(enabled=True), strategies=registry)
    names = [s.name for s in loop.strategies.strategies]
    assert names[0] == "tool_search", f"tool_search must run first, got {names}"
    assert "marker" in names


def test_disabled_registers_nothing(workspace) -> None:
    loop = _make_loop(workspace, ToolSearchConfig(enabled=False))
    for name in ("tool_search", "tool_call"):
        assert not loop.tools.has(name)
    assert loop.strategies.get("tool_search") is None


def test_none_config_registers_nothing(workspace) -> None:
    loop = _make_loop(workspace, None)
    assert not loop.tools.has("tool_search")
    assert loop.strategies.get("tool_search") is None


def test_default_registry_excludes_media_generation_tools(workspace) -> None:
    loop = _make_loop(workspace, None)

    assert {"web_search", "web_fetch", "message"} <= set(loop.tools.tool_names)
    assert {
        "image_generate",
        "text_to_speech",
        "video_generate",
        "read_skill",
        "use_skill",
    }.isdisjoint(loop.tools.tool_names)


@pytest.mark.asyncio
async def test_enabled_loop_keeps_interaction_primitives_visible(workspace) -> None:

    loop = _make_loop(workspace, ToolSearchConfig(enabled=True, compaction_threshold=5))
    assert loop.tools.has("ask_user") and loop.tools.has("spawn") and loop.tools.has("web_search")
    tools = loop.tools.get_definitions()
    _, out, _ = await loop.strategies.before_llm_call([], tools, "stub")
    names = {t["function"]["name"] for t in out}
    assert {"read_file", "message", "ask_user", "spawn"} <= names, "primitives must stay visible"
    assert {"tool_search", "tool_call"} <= names, "meta-tools must stay visible"
    assert "web_search" not in names, "cataloged domain tools are withheld above threshold"


@pytest.mark.asyncio
async def test_progressive_budget_and_provider_share_exact_disclosure_view(workspace) -> None:
    loop = _make_loop(workspace, ToolSearchConfig(enabled=True, compaction_threshold=5))
    view = loop._effective_tool_disclosure_view()
    tools = view.provider_tools()
    assert tools is not None

    budget = loop._make_token_budget(tool_definitions=tools)
    full_budget = loop._make_token_budget(tool_definitions=loop.tools.get_definitions())
    evidence: list[dict] = []
    await loop._run_agent_loop(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "go"}],
        initial_disclosure_view=view,
        disclosure_evidence_sink=evidence,
    )

    provider_tools = loop.provider.visible_tools[0]
    assert provider_tools == tools
    assert budget.reserved_tools == estimate_prompt_tokens([], provider_tools)
    assert budget.reserved_tools < full_budget.reserved_tools
    assert evidence[0]["visible_tool_names"] == view.visible_names
    assert evidence[0]["tool_array_schema_tokens"] == budget.reserved_tools


def test_full_disclosure_budget_matches_full_registry(workspace) -> None:
    loop = _make_loop(workspace, ToolSearchConfig(enabled=False))
    view = loop._effective_tool_disclosure_view()
    tools = view.provider_tools()
    assert tools == loop.tools.get_definitions()
    assert loop._make_token_budget(tool_definitions=tools).reserved_tools == estimate_prompt_tokens([], tools)


def test_fail_open_budget_matches_full_provider_view(workspace) -> None:
    loop = _make_loop(workspace, ToolSearchConfig(enabled=True, compaction_threshold=5))
    loop.tools.unregister("tool_search")
    loop.tools.unregister("tool_call")

    view = loop._effective_tool_disclosure_view()
    tools = view.provider_tools()
    assert view.mode == "fail_open_full"
    assert tools == loop.tools.get_definitions()
    assert loop._make_token_budget(tool_definitions=tools).reserved_tools == estimate_prompt_tokens([], tools)


@pytest.mark.asyncio
async def test_disclosure_view_refreshes_for_each_provider_iteration(workspace) -> None:
    class _TwoIterationProvider(_StubProvider):
        async def chat(self, messages, tools=None, **kwargs):
            del kwargs
            self.visible_tools.append(tools or [])
            if len(self.visible_tools) == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCallRequest(id="missing-1", name="missing_tool", arguments={})],
                )
            return LLMResponse(content="done", finish_reason="stop")

    provider = _TwoIterationProvider()
    loop = _make_loop(
        workspace,
        ToolSearchConfig(enabled=True, compaction_threshold=5),
        provider=provider,
    )
    strategy = loop._tool_search_strategy
    assert strategy is not None
    original = strategy.disclosure_view
    refreshes = 0

    def counted(tools):
        nonlocal refreshes
        refreshes += 1
        return original(tools)

    strategy.disclosure_view = counted
    initial = loop._effective_tool_disclosure_view()
    evidence: list[dict] = []
    await loop._run_agent_loop(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "go"}],
        initial_disclosure_view=initial,
        disclosure_evidence_sink=evidence,
    )

    assert refreshes == 2
    assert len(evidence) == len(provider.visible_tools) == 2
    assert [item["tool_array_schema_tokens"] for item in evidence] == [
        estimate_prompt_tokens([], tools) for tools in provider.visible_tools
    ]
