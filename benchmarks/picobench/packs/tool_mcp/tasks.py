from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.fixtures.mcp import catalog_definitions

from .models import ToolMCPTask, ToolMCPTrack, ToolTarget

FORMAL_TOOL_MCP_TASK_COUNT = 8
CALIBRATION_TOOL_MCP_TASK_COUNT = 4
ROLE_EXPERIMENT_TOOL_MCP_TASK_COUNT = 3
_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks" / "tool_mcp"


@lru_cache(maxsize=3)
def load_tool_mcp_tasks(
    track: ToolMCPTrack,
) -> tuple[ToolMCPTask, ...]:
    track = ToolMCPTrack(track)
    source_path = _TASK_ROOT / f"{track.value}.json"
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "pico.picobench.tool-mcp-tasks.v1":
        raise ValueError(f"unsupported Tool/MCP task schema in {source_path}")
    entries = raw.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"Tool/MCP task list missing in {source_path}")
    tasks = tuple(_parse_task(track, entry) for entry in entries)
    expected_count = {
        ToolMCPTrack.FORMAL: FORMAL_TOOL_MCP_TASK_COUNT,
        ToolMCPTrack.CALIBRATION: CALIBRATION_TOOL_MCP_TASK_COUNT,
        ToolMCPTrack.ROLE_EXPERIMENT: ROLE_EXPERIMENT_TOOL_MCP_TASK_COUNT,
    }[track]
    if len(tasks) != expected_count:
        raise ValueError(f"{track.value} requires exactly {expected_count} Tool/MCP tasks")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError(f"duplicate Tool/MCP task id in {source_path}")
    return tasks


def tool_mcp_task_set_digest(track: ToolMCPTrack) -> str:
    return canonical_digest([task.to_task_spec() for task in load_tool_mcp_tasks(track)])


def _parse_task(track: ToolMCPTrack, raw: Any) -> ToolMCPTask:
    if not isinstance(raw, dict):
        raise ValueError("Tool/MCP task entries must be objects")
    target_entries = raw.get("targets")
    if not isinstance(target_entries, list) or not 1 <= len(target_entries) <= 3:
        raise ValueError("Tool/MCP task requires one to three targets")
    catalog_names = {tool.name for tool in catalog_definitions()}
    targets: list[ToolTarget] = []
    for entry in target_entries:
        if not isinstance(entry, dict):
            raise ValueError("Tool/MCP target entries must be objects")
        tool_name = _required_str(entry, "tool")
        if tool_name not in catalog_names:
            raise ValueError(f"unknown Tool/MCP fixture tool: {tool_name}")
        arguments = entry.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Tool/MCP target arguments must be an object")
        target = ToolTarget(tool_name=tool_name, arguments=arguments)
        target.expected_receipt
        targets.append(target)
    role = "general"
    search_query = None
    relevant_tools: tuple[str, ...] = ()
    irrelevant_tools: tuple[str, ...] = ()
    if track is ToolMCPTrack.ROLE_EXPERIMENT:
        role = _required_str(raw, "role")
        if role not in {"coder", "debugger", "researcher"}:
            raise ValueError(f"unsupported experimental Tool/MCP role: {role}")
        search_query = _required_str(raw, "search_query")
        relevant_tools = _tool_name_list(raw, "relevant_tools", catalog_names)
        irrelevant_tools = _tool_name_list(raw, "irrelevant_tools", catalog_names)
        if not {target.tool_name for target in targets} <= set(relevant_tools):
            raise ValueError("all role-experiment targets must be relevant tools")
        if set(relevant_tools) & set(irrelevant_tools):
            raise ValueError("relevant and irrelevant tools must be disjoint")
    return ToolMCPTask(
        task_id=_required_str(raw, "task_id"),
        track=track,
        title=_required_str(raw, "title"),
        prompt=_required_str(raw, "prompt"),
        targets=tuple(targets),
        role=role,
        search_query=search_query,
        relevant_tools=relevant_tools,
        irrelevant_tools=irrelevant_tools,
    )


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Tool/MCP task field {key!r} must be a non-empty string")
    return value


def _tool_name_list(
    raw: dict[str, Any],
    key: str,
    catalog_names: set[str],
) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Tool/MCP task field {key!r} must be a non-empty list")
    names = tuple(str(value) for value in values)
    if len(set(names)) != len(names) or not set(names) <= catalog_names:
        raise ValueError(f"Tool/MCP task field {key!r} contains invalid tools")
    return names


__all__ = [
    "CALIBRATION_TOOL_MCP_TASK_COUNT",
    "FORMAL_TOOL_MCP_TASK_COUNT",
    "ROLE_EXPERIMENT_TOOL_MCP_TASK_COUNT",
    "load_tool_mcp_tasks",
    "tool_mcp_task_set_digest",
]
