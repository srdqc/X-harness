from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pico.agent.tools.execution import ToolEffect


class ToolSourceKind(StrEnum):
    UNKNOWN = "unknown"
    BUILTIN = "builtin"
    PLUGIN = "plugin"
    MCP = "mcp"
    META = "meta"
    OTHER = "other"


@dataclass(frozen=True)
class ToolDiscoveryMetadata:
    """Stable discovery-only projection of one currently registered Tool."""

    name: str
    description: str
    source_kind: ToolSourceKind = ToolSourceKind.UNKNOWN
    source_id: str | None = None
    category: str | None = None
    effect: ToolEffect = ToolEffect.UNKNOWN


__all__ = ["ToolDiscoveryMetadata", "ToolSourceKind"]
