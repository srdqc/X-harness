from __future__ import annotations

import json
import os
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from benchmarks.picobench.fixtures.mcp import (
    LIVE_MCP_CATALOG_SIZE,
    LiveFixtureEngine,
    live_catalog_definitions,
    live_catalog_definitions_v2,
    live_catalog_digest,
)
from benchmarks.picobench.packs.tool_mcp import (
    LIVE_SOLVABLE_V1_TOOL_MCP_TASK_COUNT,
    LIVE_SOLVABLE_V2_TOOL_MCP_TASK_COUNT,
    ArgumentSource,
    DeterministicMCPTrialRunner,
    MCPRuntimeTrialRunner,
    SealedLiveStateVerifier,
    SealedLiveStateVerifierV2,
    ToolMCPPack,
    ToolMCPTask,
    ToolMCPTrack,
    live_verifier_code_digest,
    load_tool_mcp_tasks,
    tool_mcp_task_set_digest,
    validate_live_task_contract,
)
from benchmarks.picobench.protocol import TrialContext
from benchmarks.picobench.records import TrialKey, VerificationState
from benchmarks.picobench.schema import ExperimentSpec
from pico.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest

_FROZEN_MECHANICS_DIGESTS = {
    ToolMCPTrack.FORMAL: "ec2df20dae92714e8a34f30ce3a49bc3477025ef94a68f4f01f7ab87a53eb8b3",
    ToolMCPTrack.CALIBRATION: "6e5817387ae09581a96d04009958236472a741cbee41b66b34c83f4b6d3cadb2",
    ToolMCPTrack.ROLE_EXPERIMENT: "8e42e8cb8bb489b82a5692debaf71136dfe2dfefca8a3fe2836103965cc72667",
}


class _VisibleInputPreflightProvider(LLMProvider):
    """Uses only the Provider-visible request, never task/evaluator metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=128, temperature=0.0)
        self.calls = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        del model, max_tokens, temperature, reasoning_effort, tool_choice
        self.calls += 1
        names = {item["function"]["name"] for item in (tools or [])}
        already_searched = any(
            message.get("role") == "tool" and message.get("name") == "tool_search" for message in messages
        )
        if "tool_search" in names and not already_searched:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"visible-search-{self.calls}",
                        name="tool_search",
                        arguments={"query": "inspect repository file content", "limit": 5},
                    )
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                model="preflight/visible-input",
            )
        return LLMResponse(
            content="preflight request constructed",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            model="preflight/visible-input",
        )

    def get_default_model(self) -> str:
        return "preflight/visible-input"


def test_frozen_mechanics_task_contracts_remain_identical() -> None:
    assert {track: tool_mcp_task_set_digest(track) for track in _FROZEN_MECHANICS_DIGESTS} == _FROZEN_MECHANICS_DIGESTS


def test_live_track_is_separate_versioned_and_balanced() -> None:
    tasks = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)
    definition = ToolMCPPack(ToolMCPTrack.LIVE_SOLVABLE_V1).definition()

    assert len(tasks) == LIVE_SOLVABLE_V1_TOOL_MCP_TASK_COUNT == 12
    assert {role: sum(task.role == role for task in tasks) for role in {task.role for task in tasks}} == {
        "coder": 4,
        "debugger": 4,
        "researcher": 4,
    }
    assert definition.pack_id == "tool-mcp-live-solvable-v1"
    assert [variant.variant_id for variant in definition.variants] == [
        "live-full",
        "live-progressive",
        "live-role-aware",
    ]


def test_live_v1_contract_remains_frozen() -> None:
    assert tool_mcp_task_set_digest(ToolMCPTrack.LIVE_SOLVABLE_V1) == (
        "455da0359832fb463cc4d70e7ff493e9457c33f5a1531744800e8a86be81b7f8"
    )
    assert live_catalog_digest() == "69d9908916d4b4385e040144f98c50a576f558ef7c44cd96e0d3a51e6d2b521b"
    assert live_verifier_code_digest() == "c726f536d6f5d34608c9e79de02b9a9878d51e0f2b76f3efd172fc388a9307df"


def test_live_v2_is_separately_versioned_balanced_and_preserves_non_research_tasks() -> None:
    v1_tasks = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)
    v2_tasks = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V2)
    definition = ToolMCPPack(ToolMCPTrack.LIVE_SOLVABLE_V2).definition()

    assert len(v2_tasks) == LIVE_SOLVABLE_V2_TOOL_MCP_TASK_COUNT == 12
    assert {role: sum(task.role == role for task in v2_tasks) for role in {task.role for task in v2_tasks}} == {
        "coder": 4,
        "debugger": 4,
        "researcher": 4,
    }
    assert definition.pack_id == "tool-mcp-live-solvable-v2"
    assert definition.identity["result_scope"] == "live_model_acceptance_v2"
    assert tool_mcp_task_set_digest(ToolMCPTrack.LIVE_SOLVABLE_V2) != tool_mcp_task_set_digest(
        ToolMCPTrack.LIVE_SOLVABLE_V1
    )
    assert [variant.variant_id for variant in definition.variants] == [
        "live-full",
        "live-progressive",
        "live-role-aware",
    ]
    assert [replace(task, track=ToolMCPTrack.LIVE_SOLVABLE_V1) for task in v2_tasks[:8]] == list(v1_tasks[:8])


def test_every_live_task_passes_observable_contract() -> None:
    catalog = live_catalog_definitions()
    for task in load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1):
        assert validate_live_task_contract(task, catalog) == ()
        assert all(set(target.arguments) == set(target.argument_sources) for target in task.targets)


def test_every_live_v2_task_passes_observable_contract() -> None:
    catalog = live_catalog_definitions_v2()
    for task in load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V2):
        assert validate_live_task_contract(task, catalog) == ()
        assert all(set(target.arguments) == set(target.argument_sources) for target in task.targets)


def test_live_v2_research_contract_is_explicit_and_does_not_seal_explanation_wording() -> None:
    task = next(
        task for task in load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V2) if task.role == "researcher"
    )
    report = task.targets[-1]
    definition = next(item for item in live_catalog_definitions_v2() if item.name == report.tool_name)

    assert set(report.arguments) == {"topic", "winning_entity", "explanation", "evidence_ids"}
    assert report.argument_sources["winning_entity"].kind == "observation"
    assert report.argument_sources["explanation"].kind == "free_text"
    assert "winning_entity" in task.prompt
    assert "explanation" in task.prompt
    assert "evidence_ids" in task.prompt
    assert set(definition.parameters["required"]) == set(report.arguments)
    assert "winning_entity" in task.expected_state["report"]
    assert "explanation" not in task.expected_state["report"]

    undocumented = replace(task, prompt=task.prompt.replace("explanation", "rationale"))
    exact_wording = replace(
        task,
        expected_state={
            "report": {
                **dict(task.expected_state["report"]),
                "explanation": report.arguments["explanation"],
            }
        },
    )
    assert "v2_research_prompt_field_missing:explanation" in validate_live_task_contract(
        undocumented,
        live_catalog_definitions_v2(),
    )
    assert "v2_research_explanation_must_not_be_exact" in validate_live_task_contract(
        exact_wording,
        live_catalog_definitions_v2(),
    )


def test_hidden_only_argument_and_exact_tool_leakage_are_rejected() -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)[0]
    target = task.targets[0]
    hidden_sources = dict(target.argument_sources)
    hidden_sources.pop(next(iter(hidden_sources)))
    hidden = replace(
        task,
        targets=(replace(target, argument_sources=hidden_sources), *task.targets[1:]),
    )
    leaked = replace(task, prompt=f"{task.prompt} Use {target.runtime_name}.")

    assert "target_0:hidden_only_argument" in validate_live_task_contract(hidden, live_catalog_definitions())
    assert "target_0:expected_tool_name_leaked" in validate_live_task_contract(leaked, live_catalog_definitions())


def test_observation_source_must_resolve_to_expected_argument() -> None:
    task = next(task for task in load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1) if task.role == "researcher")
    report = task.targets[-1]
    sources = dict(report.argument_sources)
    sources["conclusion"] = ArgumentSource(kind="observation", state_path="records.missing.label")
    invalid = replace(task, targets=(*task.targets[:-1], replace(report, argument_sources=sources)))

    assert "target_2:conclusion:observation_path_missing" in validate_live_task_contract(
        invalid, live_catalog_definitions()
    )


def test_hidden_expected_fixture_value_is_rejected() -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)[0]
    impossible = replace(task, expected_state={"content": "hidden-verifier-answer"})

    assert "expected_state_contains_hidden_value" in validate_live_task_contract(impossible, live_catalog_definitions())


def test_live_catalog_is_large_semantic_and_schema_distinguishable() -> None:
    catalog = live_catalog_definitions()
    assert len(catalog) == LIVE_MCP_CATALOG_SIZE == 64
    assert len({item.name for item in catalog}) == 64
    assert len({item.description for item in catalog}) == 64
    assert {item.category for item in catalog} >= {"coder", "debugger", "researcher", "general"}
    assert len({json.dumps(item.parameters, sort_keys=True) for item in catalog}) >= 8


def _fixture(tmp_path: Path, task_index: int) -> tuple[ToolMCPTask, LiveFixtureEngine, Path]:
    task = load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)[task_index]
    state_path = tmp_path / "state.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps(dict(task.initial_state)), encoding="utf-8")
    return task, LiveFixtureEngine(state_path, event_path), event_path


def _v2_research_fixture(tmp_path: Path) -> tuple[ToolMCPTask, LiveFixtureEngine, Path]:
    task = next(
        task for task in load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V2) if task.role == "researcher"
    )
    state_path = tmp_path / "state.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps(dict(task.initial_state)), encoding="utf-8")
    return (
        task,
        LiveFixtureEngine(
            state_path,
            event_path,
            definitions=live_catalog_definitions_v2(),
        ),
        event_path,
    )


def _execute_targets(engine: LiveFixtureEngine, task: ToolMCPTask) -> None:
    for target in task.targets:
        engine.execute(target.tool_name, dict(target.arguments))


def test_coder_fixture_requires_mutation_and_verification(tmp_path: Path) -> None:
    task, engine, events = _fixture(tmp_path, 0)
    verifier = SealedLiveStateVerifier.capture(task, engine.state_path)
    engine.execute(task.targets[0].tool_name, dict(task.targets[0].arguments))
    failed, _ = verifier.verify(events)
    assert failed.state is VerificationState.FAILED

    _execute_targets(engine, task)
    passed, _ = verifier.verify(events)
    assert passed.state is VerificationState.PASSED


def test_debugger_fixture_requires_evidence_mutation_and_verification(tmp_path: Path) -> None:
    task, engine, events = _fixture(tmp_path, 4)
    _execute_targets(engine, task)

    result, observed = SealedLiveStateVerifier.capture(task, engine.state_path).verify(events)
    assert result.state is VerificationState.PASSED
    assert [event["capability"] for event in observed] == [
        "incident.read",
        "incident.mutate",
        "incident.verify",
    ]


def test_researcher_fixture_rejects_unnecessary_mutation(tmp_path: Path) -> None:
    task, engine, events = _fixture(tmp_path, 8)
    _execute_targets(engine, task)
    engine.execute("issue_label_update", {"entity_id": "ISSUE-1", "value": "noise"})

    result, _ = SealedLiveStateVerifier.capture(task, engine.state_path).verify(events)
    assert result.state is VerificationState.FAILED
    assert "prohibited_live_capability_observed" in result.findings


def test_live_v2_researcher_accepts_varied_explanation_but_checks_structured_result(
    tmp_path: Path,
) -> None:
    task, engine, events = _v2_research_fixture(tmp_path)
    for target in task.targets[:-1]:
        engine.execute(target.tool_name, dict(target.arguments))
    report_arguments = dict(task.targets[-1].arguments)
    report_arguments["explanation"] = (
        "LAT-B reports 180 ms while LAT-A reports 240 ms, so the selected service is lower."
    )
    engine.execute(task.targets[-1].tool_name, report_arguments)

    result, _ = SealedLiveStateVerifierV2.capture(task, engine.state_path).verify(events)

    assert result.state is VerificationState.PASSED


@pytest.mark.parametrize(
    ("argument_name", "replacement"),
    [
        ("winning_entity", "service-orchid"),
        ("evidence_ids", ["LAT-B"]),
    ],
)
def test_live_v2_researcher_rejects_wrong_entity_or_missing_evidence(
    tmp_path: Path,
    argument_name: str,
    replacement: object,
) -> None:
    task, engine, events = _v2_research_fixture(tmp_path)
    for target in task.targets[:-1]:
        engine.execute(target.tool_name, dict(target.arguments))
    report_arguments = dict(task.targets[-1].arguments)
    report_arguments[argument_name] = replacement
    engine.execute(task.targets[-1].tool_name, report_arguments)

    result, _ = SealedLiveStateVerifierV2.capture(task, engine.state_path).verify(events)

    assert result.state is VerificationState.FAILED
    assert "expected_live_v2_fixture_state_missing" in result.findings


def test_live_v2_researcher_rejects_missing_explanation(tmp_path: Path) -> None:
    task, engine, events = _v2_research_fixture(tmp_path)
    for target in task.targets[:-1]:
        engine.execute(target.tool_name, dict(target.arguments))
    report_arguments = dict(task.targets[-1].arguments)
    report_arguments["explanation"] = " "
    engine.execute(task.targets[-1].tool_name, report_arguments)

    result, _ = SealedLiveStateVerifierV2.capture(task, engine.state_path).verify(events)

    assert result.state is VerificationState.FAILED
    assert "research_explanation_missing" in result.findings


def test_live_v2_researcher_rejects_prohibited_mutation(tmp_path: Path) -> None:
    task, engine, events = _v2_research_fixture(tmp_path)
    _execute_targets(engine, task)
    engine.execute("issue_label_update", {"entity_id": "ISSUE-1", "value": "noise"})

    result, _ = SealedLiveStateVerifierV2.capture(task, engine.state_path).verify(events)

    assert result.state is VerificationState.FAILED
    assert "prohibited_live_v2_capability_observed" in result.findings


def test_live_verifier_is_independent_of_retrieval_expectations(tmp_path: Path) -> None:
    task, engine, events = _fixture(tmp_path, 0)
    _execute_targets(engine, task)
    changed_retrieval_labels = replace(task, relevant_tools=("repository_symbol_search",))

    result, _ = SealedLiveStateVerifier.capture(changed_retrieval_labels, engine.state_path).verify(events)
    assert result.state is VerificationState.PASSED


def test_all_live_fixtures_reach_sealed_success(tmp_path: Path) -> None:
    for index, task in enumerate(load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V1)):
        task_root = tmp_path / str(index)
        task_root.mkdir()
        state_path = task_root / "state.json"
        event_path = task_root / "events.jsonl"
        state_path.write_text(json.dumps(dict(task.initial_state)), encoding="utf-8")
        engine = LiveFixtureEngine(state_path, event_path)
        _execute_targets(engine, task)

        result, _ = SealedLiveStateVerifier.capture(task, state_path).verify(event_path)
        assert result.state is VerificationState.PASSED, (task.task_id, result.findings)


def test_all_live_v2_fixtures_reach_sealed_success(tmp_path: Path) -> None:
    for index, task in enumerate(load_tool_mcp_tasks(ToolMCPTrack.LIVE_SOLVABLE_V2)):
        task_root = tmp_path / str(index)
        task_root.mkdir()
        state_path = task_root / "state.json"
        event_path = task_root / "events.jsonl"
        state_path.write_text(json.dumps(dict(task.initial_state)), encoding="utf-8")
        engine = LiveFixtureEngine(
            state_path,
            event_path,
            definitions=live_catalog_definitions_v2(),
        )
        _execute_targets(engine, task)

        result, _ = SealedLiveStateVerifierV2.capture(task, state_path).verify(event_path)
        assert result.state is VerificationState.PASSED, (task.task_id, result.findings)


async def test_hidden_answer_deterministic_runner_is_forbidden_for_live_track() -> None:
    pack = ToolMCPPack(
        ToolMCPTrack.LIVE_SOLVABLE_V1,
        runner=DeterministicMCPTrialRunner(),
    )
    context = _context(Path("."), pack.definition(), "live-full")

    with pytest.raises(RuntimeError, match="hidden-answer deterministic provider"):
        await pack.run_trial(context)


async def test_hidden_answer_deterministic_runner_is_forbidden_for_live_v2() -> None:
    pack = ToolMCPPack(
        ToolMCPTrack.LIVE_SOLVABLE_V2,
        runner=DeterministicMCPTrialRunner(),
    )
    context = _context(Path("."), pack.definition(), "live-full")

    with pytest.raises(RuntimeError, match="hidden-answer deterministic provider"):
        await pack.run_trial(context)


@pytest.mark.parametrize("variant_id", ["live-full", "live-progressive", "live-role-aware"])
async def test_live_variants_construct_through_normal_runtime(
    tmp_path: Path,
    variant_id: str,
) -> None:
    provider = _VisibleInputPreflightProvider()
    pack = ToolMCPPack(
        ToolMCPTrack.LIVE_SOLVABLE_V1,
        runner=MCPRuntimeTrialRunner(
            provider=provider,
            model=provider.get_default_model(),
            generation=provider.generation,
        ),
    )
    execution = await pack.run_trial(_context(tmp_path, pack.definition(), variant_id))

    assert execution.metrics["mcp_connected"] is True
    assert execution.metrics["mcp_catalog_count"] == LIVE_MCP_CATALOG_SIZE
    assert execution.metrics["model_call_records"], (
        execution.status,
        execution.runtime_state,
        execution.verification,
        execution.findings,
    )
    if variant_id == "live-full":
        assert execution.metrics["initial_visible_catalog_tool_count"] == LIVE_MCP_CATALOG_SIZE
    else:
        assert execution.metrics["tool_retrieval_query_count"] == 1
        assert execution.metrics["tool_retrieval_evidence"]
        assert execution.metrics["target_recall_at_5"] == 1.0


@pytest.mark.parametrize("variant_id", ["live-full", "live-progressive", "live-role-aware"])
async def test_live_v2_variants_construct_through_normal_runtime(
    tmp_path: Path,
    variant_id: str,
) -> None:
    provider = _VisibleInputPreflightProvider()
    pack = ToolMCPPack(
        ToolMCPTrack.LIVE_SOLVABLE_V2,
        runner=MCPRuntimeTrialRunner(
            provider=provider,
            model=provider.get_default_model(),
            generation=provider.generation,
        ),
    )
    execution = await pack.run_trial(_context(tmp_path, pack.definition(), variant_id))

    assert execution.metrics["mcp_connected"] is True
    assert execution.metrics["mcp_catalog_count"] == LIVE_MCP_CATALOG_SIZE
    assert execution.metrics["model_call_records"], (
        execution.status,
        execution.runtime_state,
        execution.verification,
        execution.findings,
    )
    if variant_id == "live-full":
        assert execution.metrics["initial_visible_catalog_tool_count"] == LIVE_MCP_CATALOG_SIZE
    else:
        assert execution.metrics["tool_retrieval_query_count"] == 1
        assert execution.metrics["tool_retrieval_evidence"]
        assert execution.metrics["target_recall_at_5"] == 1.0


def _context(tmp_path: Path, definition, variant_id: str) -> TrialContext:
    variant = next(item for item in definition.variants if item.variant_id == variant_id)
    task = definition.tasks[0]
    output_root = tmp_path
    if os.name == "nt":
        output_root = Path(".pico/picobench-tests") / sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    experiment = ExperimentSpec(
        suite="tool-mcp-live-preflight",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=output_root,
        identity={"pico_commit": "0" * 40, "model": "preflight/visible-input", "budget_cap_cny": 0},
    )
    return TrialContext(
        experiment_id="tool-mcp-live-preflight",
        plan_digest="a" * 64,
        key=TrialKey(
            experiment_id="tool-mcp-live-preflight",
            pack_id=definition.pack_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task,
        variant=variant,
    )
