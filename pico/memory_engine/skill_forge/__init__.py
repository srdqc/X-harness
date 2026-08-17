"""Local Skill retrieval and rendering machinery.

The :class:`SkillSource` protocol is host-internal rather than a plugin
contribution point. The active Runtime builds a router over
:class:`LocalSkillSource`; the generic router and fusion helpers remain useful
for deterministic evaluation.
"""

from __future__ import annotations

from pico.memory_engine.skill_forge.catalog import LocalSkillCatalog
from pico.memory_engine.skill_forge.fusion import RRF_K, rrf_merge_weighted
from pico.memory_engine.skill_forge.gate import LLMGateFilter
from pico.memory_engine.skill_forge.local_source import LocalSkillSource
from pico.memory_engine.skill_forge.refs import resolve_refs
from pico.memory_engine.skill_forge.resolver import LocalSkillResolver, SkillResolution
from pico.memory_engine.skill_forge.rewriter import (
    QueryRewriter,
    RewriteResult,
)
from pico.memory_engine.skill_forge.router import SkillForgeRouter
from pico.memory_engine.skill_forge.types import RouterHit, SkillSource

__all__ = [
    "LLMGateFilter",
    "LocalSkillCatalog",
    "LocalSkillSource",
    "LocalSkillResolver",
    "QueryRewriter",
    "RRF_K",
    "RewriteResult",
    "RouterHit",
    "SkillForgeRouter",
    "SkillResolution",
    "SkillSource",
    "resolve_refs",
    "rrf_merge_weighted",
]
