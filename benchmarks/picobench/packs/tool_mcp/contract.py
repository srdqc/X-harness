from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from benchmarks.picobench.fixtures.mcp.live_catalog import LiveToolDefinition

from .models import ToolMCPTask, ToolMCPTrack


def validate_live_task_contract(
    task: ToolMCPTask,
    catalog: Iterable[LiveToolDefinition],
) -> tuple[str, ...]:
    if task.track is not ToolMCPTrack.LIVE_SOLVABLE_V1:
        return ("wrong_live_contract_track",)
    definitions = {item.name: item for item in catalog}
    findings: list[str] = []
    prompt = task.prompt.casefold()
    for target_index, target in enumerate(task.targets):
        runtime_name = target.runtime_name.casefold()
        if target.tool_name.casefold() in prompt or runtime_name in prompt:
            findings.append(f"target_{target_index}:expected_tool_name_leaked")
        if set(target.argument_sources) != set(target.arguments):
            findings.append(f"target_{target_index}:hidden_only_argument")
        for name, value in target.arguments.items():
            source = target.argument_sources.get(name)
            if source is None:
                continue
            if source.kind == "prompt":
                for observable in _observable_scalars(value):
                    if observable.casefold() not in prompt:
                        findings.append(f"target_{target_index}:{name}:prompt_value_missing")
            elif source.kind == "observation":
                if target_index == 0 or not source.state_path:
                    findings.append(f"target_{target_index}:{name}:invalid_observation_source")
                    continue
                try:
                    observed = _state_path(task.initial_state, source.state_path)
                except (KeyError, TypeError):
                    findings.append(f"target_{target_index}:{name}:observation_path_missing")
                else:
                    if observed != value:
                        findings.append(f"target_{target_index}:{name}:observation_value_mismatch")
            else:
                findings.append(f"target_{target_index}:{name}:unknown_argument_source")
        definition = definitions.get(target.tool_name)
        if definition is None:
            findings.append(f"target_{target_index}:catalog_target_missing")
        elif not _is_semantically_distinguishable(definition, definitions.values()):
            findings.append(f"target_{target_index}:catalog_target_indistinguishable")
    if not task.expected_state:
        findings.append("missing_expected_fixture_state")
    observable_values = {item.casefold() for item in _all_scalars(task.initial_state)}
    observable_values.update(
        scalar.casefold()
        for target in task.targets
        for value in target.arguments.values()
        for scalar in _observable_scalars(value)
    )
    if any(scalar.casefold() not in observable_values for scalar in _all_scalars(task.expected_state)):
        findings.append("expected_state_contains_hidden_value")
    if not task.required_capabilities:
        findings.append("missing_required_capabilities")
    if not set(task.relevant_tools).issuperset(target.tool_name for target in task.targets):
        findings.append("targets_missing_from_relevant_tools")
    return tuple(dict.fromkeys(findings))


def validate_live_task_set(
    tasks: Iterable[ToolMCPTask],
    catalog: Iterable[LiveToolDefinition],
) -> None:
    failures = {task.task_id: findings for task in tasks if (findings := validate_live_task_contract(task, catalog))}
    if failures:
        raise ValueError(f"invalid live Tool/MCP task contract: {json.dumps(failures, sort_keys=True)}")


def _observable_scalars(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _all_scalars(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(scalar for nested in value.values() for scalar in _all_scalars(nested))
    if isinstance(value, list):
        return tuple(scalar for nested in value for scalar in _all_scalars(nested))
    return (str(value),)


def _state_path(state: Any, path: str) -> Any:
    current = state
    for component in path.split("."):
        if not isinstance(current, dict):
            raise TypeError(path)
        current = current[component]
    return current


def _is_semantically_distinguishable(
    target: LiveToolDefinition,
    catalog: Iterable[LiveToolDefinition],
) -> bool:
    signature = (
        target.description.casefold(),
        json.dumps(target.parameters, sort_keys=True),
    )
    duplicates = Counter(
        (
            item.description.casefold(),
            json.dumps(item.parameters, sort_keys=True),
        )
        for item in catalog
    )
    return duplicates[signature] == 1


__all__ = ["validate_live_task_contract", "validate_live_task_set"]
