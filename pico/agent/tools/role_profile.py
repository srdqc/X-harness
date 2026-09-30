from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from pico.agent.tools.discovery import ToolDiscoveryMetadata, ToolSourceKind

_MAX_ABSOLUTE_PRIOR = 0.15


class RoleName(StrEnum):
    GENERAL = "general"
    CODER = "coder"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"


@dataclass(frozen=True)
class RoleProfile:
    """Experimental model-guidance and retrieval prior with no execution authority."""

    name: RoleName
    prompt_fragment: str = ""
    tool_priors: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    category_priors: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    source_priors: Mapping[ToolSourceKind, float] = field(default_factory=lambda: MappingProxyType({}))

    def prior_for(self, metadata: ToolDiscoveryMetadata) -> float:
        prior = self.tool_priors.get(metadata.name, 0.0)
        if metadata.category:
            prior += self.category_priors.get(metadata.category, 0.0)
        prior += self.source_priors.get(metadata.source_kind, 0.0)
        return max(-_MAX_ABSOLUTE_PRIOR, min(_MAX_ABSOLUTE_PRIOR, prior))


def _profile(
    name: RoleName,
    prompt: str,
    *,
    tools: Mapping[str, float] | None = None,
    categories: Mapping[str, float] | None = None,
    sources: Mapping[ToolSourceKind, float] | None = None,
) -> RoleProfile:
    return RoleProfile(
        name=name,
        prompt_fragment=prompt,
        tool_priors=MappingProxyType(dict(tools or {})),
        category_priors=MappingProxyType(dict(categories or {})),
        source_priors=MappingProxyType(dict(sources or {})),
    )


_PROFILES = {
    RoleName.GENERAL: _profile(RoleName.GENERAL, ""),
    RoleName.CODER: _profile(
        RoleName.CODER,
        "Understand repository context before editing; make the smallest change and verify it.",
        tools={
            "read_file": 0.08,
            "list_dir": 0.06,
            "find": 0.06,
            "grep": 0.08,
            "edit_file": 0.08,
            "write_file": 0.06,
            "exec": 0.04,
        },
        categories={"coder": 0.12, "debugger": -0.03, "researcher": -0.03},
    ),
    RoleName.DEBUGGER: _profile(
        RoleName.DEBUGGER,
        "Gather evidence and identify the likely root cause before a narrow fix; then verify it.",
        tools={"read_file": 0.07, "grep": 0.1, "find": 0.06, "exec": 0.08},
        categories={"debugger": 0.12, "coder": -0.03, "researcher": -0.03},
    ),
    RoleName.RESEARCHER: _profile(
        RoleName.RESEARCHER,
        "Gather and compare evidence before mutation; prefer reading and distinguish evidence from inference.",
        tools={
            "read_file": 0.08,
            "grep": 0.06,
            "find": 0.06,
            "web_search": 0.1,
            "web_fetch": 0.08,
            "edit_file": -0.05,
            "write_file": -0.05,
        },
        categories={"researcher": 0.12, "coder": -0.03, "debugger": -0.03},
    ),
}


def get_role_profile(name: RoleName | str | None) -> RoleProfile:
    return _PROFILES[RoleName(name or RoleName.GENERAL)]


__all__ = ["RoleName", "RoleProfile", "get_role_profile"]
