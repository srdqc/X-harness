from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.fixtures.mcp import receipt_payload
from benchmarks.picobench.schema import TaskSpec


class ToolMCPTrack(StrEnum):
    FORMAL = "formal"
    CALIBRATION = "calibration"
    ROLE_EXPERIMENT = "role_experiment"


@dataclass(frozen=True)
class ToolTarget:
    tool_name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )

    @property
    def runtime_name(self) -> str:
        return f"mcp_picobench_{self.tool_name}"

    @property
    def expected_receipt(self) -> dict[str, Any]:
        return receipt_payload(self.tool_name, dict(self.arguments))


@dataclass(frozen=True)
class ToolMCPTask:
    task_id: str
    track: ToolMCPTrack
    title: str
    prompt: str
    targets: tuple[ToolTarget, ...]
    role: str = "general"
    search_query: str | None = None
    relevant_tools: tuple[str, ...] = ()
    irrelevant_tools: tuple[str, ...] = ()

    @property
    def expected_receipts_digest(self) -> str:
        return canonical_digest([target.expected_receipt for target in self.targets])

    def to_task_spec(self) -> TaskSpec:
        experimental = (
            {
                "role": self.role,
                "search_query": self.search_query,
                "relevant_tools": [f"mcp_picobench_{name}" for name in self.relevant_tools],
                "irrelevant_tools": [f"mcp_picobench_{name}" for name in self.irrelevant_tools],
            }
            if self.track is ToolMCPTrack.ROLE_EXPERIMENT
            else {}
        )
        return TaskSpec(
            task_id=self.task_id,
            payload={
                "track": self.track.value,
                "title": self.title,
                "prompt": self.prompt,
                "targets": [
                    {
                        "tool_name": target.runtime_name,
                        "arguments": dict(target.arguments),
                    }
                    for target in self.targets
                ],
                "expected_receipts_digest": self.expected_receipts_digest,
                "verifier": "external_mcp_receipt",
                **experimental,
            },
        )


@dataclass(frozen=True)
class TargetCallRecord:
    call_id: str
    target_name: str
    arguments: dict[str, Any]
    route: str
    dispatch_status: str
    canonical_key: str
    result_preview: str
    receipt: str | None


@dataclass(frozen=True)
class TargetCallSummary:
    records: tuple[TargetCallRecord, ...]
    first_target_accuracy: float | None
    invalid_target_call_rate: float | None
    exact_target_repeat_rate: float | None
    meta_tool_invocations: dict[str, int]
    meta_tool_failures: dict[str, int]


@dataclass(frozen=True)
class MCPTransportSmokeResult:
    transport: str
    catalog_count: int
    search_hit_name: str
    called_target_name: str
    receipt: dict[str, Any]
    transport_closed: bool


__all__ = [
    "MCPTransportSmokeResult",
    "TargetCallRecord",
    "TargetCallSummary",
    "ToolMCPTask",
    "ToolMCPTrack",
    "ToolTarget",
]
