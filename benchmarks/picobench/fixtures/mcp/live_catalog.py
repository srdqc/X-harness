from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LIVE_MCP_CATALOG_SIZE = 64


def _object_schema(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_TEXT = {"type": "string", "minLength": 1}


@dataclass(frozen=True)
class LiveToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    capability: str
    handler_kind: str

    @property
    def runtime_name(self) -> str:
        return f"mcp_picobench_{self.name}"


_TARGETS = (
    LiveToolDefinition(
        "repository_file_inspect",
        "Read the current text of one file in a named source repository before making a code change.",
        _object_schema({"repository": _TEXT, "path": _TEXT}),
        "coder",
        "repository.read",
        "repository_inspect",
    ),
    LiveToolDefinition(
        "repository_file_patch",
        "Replace one exact text fragment in a named repository file, rejecting stale before-text.",
        _object_schema(
            {
                "repository": _TEXT,
                "path": _TEXT,
                "old_text": _TEXT,
                "new_text": _TEXT,
            }
        ),
        "coder",
        "repository.mutate",
        "repository_patch",
    ),
    LiveToolDefinition(
        "repository_test_verify",
        "Run a named deterministic verification suite against the current repository fixture state.",
        _object_schema({"repository": _TEXT, "suite": _TEXT}),
        "coder",
        "repository.verify",
        "repository_verify",
    ),
    LiveToolDefinition(
        "incident_evidence_read",
        "Read service, severity, and diagnostic evidence for a specific incident identifier.",
        _object_schema({"incident_id": _TEXT}),
        "debugger",
        "incident.read",
        "incident_read",
    ),
    LiveToolDefinition(
        "incident_severity_update",
        "Set the severity of a specific incident after its diagnostic evidence has been inspected.",
        _object_schema({"incident_id": _TEXT, "severity": _TEXT}),
        "debugger",
        "incident.mutate",
        "incident_update",
    ),
    LiveToolDefinition(
        "incident_resolution_verify",
        "Verify that a specific incident currently has the expected severity and remains internally consistent.",
        _object_schema({"incident_id": _TEXT, "expected_severity": _TEXT}),
        "debugger",
        "incident.verify",
        "incident_verify",
    ),
    LiveToolDefinition(
        "knowledge_record_read",
        "Read one named evidence record, including its metric, value, unit, and source citation.",
        _object_schema({"record_id": _TEXT}),
        "researcher",
        "research.read",
        "knowledge_read",
    ),
    LiveToolDefinition(
        "research_report_submit",
        "Submit a comparison conclusion with the evidence record identifiers that support it; source records are not mutated.",
        _object_schema({"topic": _TEXT, "conclusion": _TEXT, "evidence_ids": {"type": "array", "items": _TEXT}}),
        "researcher",
        "research.report",
        "research_report",
    ),
)


_DISTRACTOR_NAMES = (
    "repository_symbol_search",
    "repository_branch_inspect",
    "repository_dependency_audit",
    "repository_format_check",
    "repository_commit_prepare",
    "repository_diff_read",
    "repository_file_create",
    "repository_file_delete",
    "incident_metric_query",
    "incident_trace_search",
    "incident_owner_lookup",
    "incident_note_add",
    "incident_status_update",
    "incident_priority_update",
    "incident_timeline_read",
    "incident_alert_acknowledge",
    "documentation_search",
    "documentation_page_read",
    "documentation_link_validate",
    "documentation_page_update",
    "knowledge_index_search",
    "citation_lookup",
    "source_freshness_check",
    "glossary_lookup",
    "issue_search",
    "issue_read",
    "issue_label_update",
    "issue_comment_add",
    "issue_assign",
    "sprint_read",
    "backlog_rank",
    "issue_close",
    "message_search",
    "message_thread_read",
    "message_send",
    "channel_list",
    "message_react",
    "notification_schedule",
    "contact_lookup",
    "presence_read",
    "deployment_status_read",
    "deployment_log_read",
    "deployment_start",
    "deployment_cancel",
    "release_note_read",
    "rollback_plan_read",
    "environment_compare",
    "artifact_inspect",
    "metrics_query",
    "dashboard_read",
    "alert_rule_read",
    "alert_rule_update",
    "service_health_read",
    "latency_histogram_read",
    "error_budget_read",
    "audit_event_search",
)


def _distractor(name: str) -> LiveToolDefinition:
    domain = name.split("_", 1)[0]
    category = {
        "repository": "coder",
        "incident": "debugger",
        "documentation": "researcher",
        "knowledge": "researcher",
    }.get(domain, "general")
    words = name.replace("_", " ")
    mutating = any(
        token in name
        for token in (
            "update",
            "add",
            "assign",
            "close",
            "send",
            "start",
            "cancel",
            "create",
            "delete",
            "prepare",
            "acknowledge",
            "react",
            "schedule",
            "rank",
        )
    )
    if mutating:
        parameters = _object_schema({"entity_id": _TEXT, "value": _TEXT})
        handler_kind = "generic_mutation"
        capability = f"{domain}.mutate"
        description = f"Perform the specialized {words} operation on an explicitly named entity."
    else:
        parameters = _object_schema({"query": _TEXT})
        handler_kind = "generic_read"
        capability = f"{domain}.read"
        description = f"Retrieve information using the specialized {words} capability."
    return LiveToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        category=category,
        capability=capability,
        handler_kind=handler_kind,
    )


@lru_cache(maxsize=1)
def live_catalog_definitions() -> tuple[LiveToolDefinition, ...]:
    definitions = _TARGETS + tuple(_distractor(name) for name in _DISTRACTOR_NAMES)
    if len(definitions) != LIVE_MCP_CATALOG_SIZE:
        raise RuntimeError("live Tool/MCP catalogue size drift")
    return definitions


def live_catalog_digest() -> str:
    payload = [
        {
            "name": item.name,
            "description": item.description,
            "parameters": item.parameters,
            "category": item.category,
            "capability": item.capability,
        }
        for item in live_catalog_definitions()
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class LiveFixtureEngine:
    """Small deterministic state machine owned only by the live benchmark fixture."""

    def __init__(self, state_path: Path, event_path: Path) -> None:
        self.state_path = state_path
        self.event_path = event_path
        self._definitions = {item.name: item for item in live_catalog_definitions()}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        definition = self._definitions[tool_name]
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        result = self._dispatch(definition.handler_kind, state, arguments)
        if definition.handler_kind in {
            "repository_patch",
            "incident_update",
            "research_report",
            "generic_mutation",
        }:
            self._write_state(state)
        event = {
            "tool": tool_name,
            "capability": definition.capability,
            "arguments": arguments,
            "result": result,
        }
        event["receipt"] = hashlib.sha256(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def _dispatch(
        self,
        kind: str,
        state: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if kind == "repository_inspect":
            self._require(state, "repository", arguments, "repository")
            self._require(state, "path", arguments, "path")
            return {"content": state["content"]}
        if kind == "repository_patch":
            self._require(state, "repository", arguments, "repository")
            self._require(state, "path", arguments, "path")
            old = arguments["old_text"]
            if old not in state["content"]:
                raise ValueError("stale repository before-text")
            state["content"] = state["content"].replace(old, arguments["new_text"], 1)
            return {"changed": True, "content": state["content"]}
        if kind == "repository_verify":
            self._require(state, "repository", arguments, "repository")
            self._require(state, "suite", arguments, "suite")
            return {"passed": state["content"] == state["expected_content"]}
        if kind == "incident_read":
            self._require(state, "incident_id", arguments, "incident_id")
            return {
                "service": state["service"],
                "severity": state["severity"],
                "evidence": state["evidence"],
            }
        if kind == "incident_update":
            self._require(state, "incident_id", arguments, "incident_id")
            state["severity"] = arguments["severity"]
            return {"updated": True, "severity": state["severity"]}
        if kind == "incident_verify":
            self._require(state, "incident_id", arguments, "incident_id")
            return {"passed": state["severity"] == arguments["expected_severity"]}
        if kind == "knowledge_read":
            record_id = arguments["record_id"]
            if record_id not in state["records"]:
                raise ValueError("unknown evidence record")
            return {"record_id": record_id, **state["records"][record_id]}
        if kind == "research_report":
            evidence_ids = arguments["evidence_ids"]
            if any(record_id not in state["records"] for record_id in evidence_ids):
                raise ValueError("report cites unknown evidence")
            report = {
                "topic": arguments["topic"],
                "conclusion": arguments["conclusion"],
                "evidence_ids": evidence_ids,
            }
            state["report"] = report
            return {"accepted": True, **report}
        if kind == "generic_mutation":
            state.setdefault("distractor_mutations", []).append(dict(arguments))
            return {"updated": True}
        return {"items": [], "query": arguments["query"]}

    @staticmethod
    def _require(
        state: dict[str, Any],
        state_key: str,
        arguments: dict[str, Any],
        argument_key: str,
    ) -> None:
        if state.get(state_key) != arguments.get(argument_key):
            raise ValueError(f"fixture identity mismatch: {argument_key}")

    def _write_state(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)


__all__ = [
    "LIVE_MCP_CATALOG_SIZE",
    "LiveFixtureEngine",
    "LiveToolDefinition",
    "live_catalog_definitions",
    "live_catalog_digest",
]
