from __future__ import annotations

from pathlib import Path

import pytest

from pico.agent.loop import AgentLoop
from pico.agent.tools.base import Tool
from pico.agent.tools.registry import ToolRegistry
from pico.agent.tools.role_profile import RoleName, get_role_profile
from pico.agent.tools.tool_search import ToolSearchController
from pico.config.schema import ToolSearchConfig
from pico.providers.base import LLMProvider, LLMResponse


class _Tool(Tool):
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self.executions = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self) -> str:
        self.executions += 1
        return "ok"


class _Provider(LLMProvider):
    async def chat(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "stub"


def _controller(role: RoleName, *, enabled: bool = True) -> tuple[ToolRegistry, ToolSearchController]:
    registry = ToolRegistry()
    registry.register(_Tool("research_probe", "inspect shared record"), category="researcher")
    registry.register(_Tool("debug_probe", "inspect shared record"), category="debugger")
    controller = ToolSearchController(
        registry,
        always_visible=set(),
        role_profile=get_role_profile(role),
        role_prior_enabled=enabled,
    )
    controller.refresh()
    return registry, controller


def test_general_and_unspecified_role_preserve_base_ranking() -> None:
    _, general = _controller(RoleName.GENERAL)
    registry = ToolRegistry()
    registry.register(_Tool("research_probe", "inspect shared record"), category="researcher")
    registry.register(_Tool("debug_probe", "inspect shared record"), category="debugger")
    unspecified = ToolSearchController(registry, always_visible=set())
    unspecified.refresh()

    general_result = general.retrieve("inspect shared", 5)
    unspecified_result = unspecified.retrieve("inspect shared", 5)

    assert general_result.ranked_names == unspecified_result.ranked_names
    assert [hit.score for hit in general_result.hits] == [
        hit.score for hit in unspecified_result.hits
    ]
    assert general_result.selected_role == unspecified_result.selected_role == "general"


def test_role_prior_reranks_deterministically_with_explainable_evidence() -> None:
    _, controller = _controller(RoleName.DEBUGGER)

    first = controller.retrieve("inspect shared", 5)
    second = controller.retrieve("inspect shared", 5)

    assert first.ranked_names == second.ranked_names
    assert first.ranked_names[0] == "debug_probe"
    assert first.selected_role == "debugger"
    assert first.hits[0].role_prior > 0
    assert first.hits[0].final_score == first.hits[0].base_score + first.hits[0].role_prior


def test_strong_lexical_match_wins_and_low_priority_tool_remains_discoverable() -> None:
    registry = ToolRegistry()
    registry.register(
        _Tool("exact_crash_stack_trace", "crash stack trace root cause evidence"),
        category="researcher",
    )
    registry.register(_Tool("debug_helper", "generic helper"), category="debugger")
    controller = ToolSearchController(
        registry,
        always_visible=set(),
        role_profile=get_role_profile(RoleName.DEBUGGER),
        role_prior_enabled=True,
    )
    controller.refresh()

    strong = controller.retrieve("exact crash stack trace root cause", 5)
    low_priority = controller.retrieve("exact_crash_stack_trace", 5)

    assert strong.ranked_names[0] == "exact_crash_stack_trace"
    assert "exact_crash_stack_trace" in low_priority.ranked_names


@pytest.mark.asyncio
async def test_role_prior_does_not_change_registry_execution_authority() -> None:
    registry, controller = _controller(RoleName.DEBUGGER)
    target = registry.get("research_probe")
    assert target is not None

    result = await controller.call("research_probe", {})

    assert result == "ok"
    assert target.executions == 1
    assert await controller.call("missing", {}) == (
        "Error: tool 'missing' not found. Use tool_search to find it."
    )


@pytest.mark.parametrize("role", [RoleName.CODER, RoleName.DEBUGGER, RoleName.RESEARCHER])
def test_non_general_prompt_is_short_scoped_and_budgeted(tmp_path: Path, role: RoleName) -> None:
    loop = AgentLoop(
        provider=_Provider(),
        workspace=tmp_path,
        model="stub",
        tool_search_config=ToolSearchConfig(
            enabled=True,
            experimental_role=role.value,
            experimental_role_prompt=True,
        ),
    )
    fragment = loop._role_prompt_fragment()
    plain = AgentLoop(
        provider=_Provider(),
        workspace=tmp_path,
        model="stub",
        tool_search_config=ToolSearchConfig(enabled=True),
    )

    assert fragment.startswith(f"[Experimental Role: {role.value}]")
    assert len(fragment.split()) < 30
    assert loop._make_token_budget().reserved_system > plain._make_token_budget().reserved_system
    assert "permission" not in fragment.lower()
    assert "allow" not in fragment.lower()


def test_prompt_only_role_records_selected_role_without_reranking() -> None:
    _, controller = _controller(RoleName.CODER, enabled=False)

    result = controller.retrieve("inspect shared", 5)

    assert result.selected_role == "coder"
    assert all(hit.role_prior == 0 for hit in result.hits)
