from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from live_catalog import (
    LiveFixtureEngine,
    live_catalog_definitions,
    live_catalog_definitions_v2,
)

from mcp.server.fastmcp import FastMCP

_EVENT_ENV = "PICOBENCH_MCP_RECEIPTS"
_STATE_ENV = "PICOBENCH_MCP_STATE"
_CONTRACT_ENV = "PICOBENCH_MCP_LIVE_CONTRACT"
_V2_CONTRACT = "live_solvable_v2"


def _definitions():
    return (
        live_catalog_definitions_v2()
        if os.environ.get(_CONTRACT_ENV) == _V2_CONTRACT
        else live_catalog_definitions()
    )


def _engine() -> LiveFixtureEngine:
    state_path = os.environ.get(_STATE_ENV)
    event_path = os.environ.get(_EVENT_ENV)
    if not state_path or not event_path:
        raise RuntimeError("live Tool/MCP fixture paths are not configured")
    return LiveFixtureEngine(
        Path(state_path),
        Path(event_path),
        definitions=_definitions(),
    )


def _encoded(tool_name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(_engine().execute(tool_name, arguments), sort_keys=True)


def _repository_inspect(tool_name: str) -> Callable[..., Any]:
    async def handler(repository: str, path: str) -> str:
        return _encoded(tool_name, {"repository": repository, "path": path})

    return handler


def _repository_patch(tool_name: str) -> Callable[..., Any]:
    async def handler(
        repository: str,
        path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        return _encoded(
            tool_name,
            {
                "repository": repository,
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
            },
        )

    return handler


def _repository_verify(tool_name: str) -> Callable[..., Any]:
    async def handler(repository: str, suite: str) -> str:
        return _encoded(tool_name, {"repository": repository, "suite": suite})

    return handler


def _incident_read(tool_name: str) -> Callable[..., Any]:
    async def handler(incident_id: str) -> str:
        return _encoded(tool_name, {"incident_id": incident_id})

    return handler


def _incident_update(tool_name: str) -> Callable[..., Any]:
    async def handler(incident_id: str, severity: str) -> str:
        return _encoded(tool_name, {"incident_id": incident_id, "severity": severity})

    return handler


def _incident_verify(tool_name: str) -> Callable[..., Any]:
    async def handler(incident_id: str, expected_severity: str) -> str:
        return _encoded(
            tool_name,
            {"incident_id": incident_id, "expected_severity": expected_severity},
        )

    return handler


def _knowledge_read(tool_name: str) -> Callable[..., Any]:
    async def handler(record_id: str) -> str:
        return _encoded(tool_name, {"record_id": record_id})

    return handler


def _research_report(tool_name: str) -> Callable[..., Any]:
    async def handler(topic: str, conclusion: str, evidence_ids: list[str]) -> str:
        return _encoded(
            tool_name,
            {"topic": topic, "conclusion": conclusion, "evidence_ids": evidence_ids},
        )

    return handler


def _research_report_v2(tool_name: str) -> Callable[..., Any]:
    async def handler(
        topic: str,
        winning_entity: str,
        explanation: str,
        evidence_ids: list[str],
    ) -> str:
        return _encoded(
            tool_name,
            {
                "topic": topic,
                "winning_entity": winning_entity,
                "explanation": explanation,
                "evidence_ids": evidence_ids,
            },
        )

    return handler


def _generic_read(tool_name: str) -> Callable[..., Any]:
    async def handler(query: str) -> str:
        return _encoded(tool_name, {"query": query})

    return handler


def _generic_mutation(tool_name: str) -> Callable[..., Any]:
    async def handler(entity_id: str, value: str) -> str:
        return _encoded(tool_name, {"entity_id": entity_id, "value": value})

    return handler


_HANDLERS = {
    "repository_inspect": _repository_inspect,
    "repository_patch": _repository_patch,
    "repository_verify": _repository_verify,
    "incident_read": _incident_read,
    "incident_update": _incident_update,
    "incident_verify": _incident_verify,
    "knowledge_read": _knowledge_read,
    "research_report": _research_report,
    "research_report_v2": _research_report_v2,
    "generic_read": _generic_read,
    "generic_mutation": _generic_mutation,
}


def main() -> None:
    server = FastMCP("PicoBench live-solvable semantic catalogue")
    for tool in _definitions():
        server.add_tool(
            _HANDLERS[tool.handler_kind](tool.name),
            name=tool.name,
            description=tool.description,
            structured_output=False,
        )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
