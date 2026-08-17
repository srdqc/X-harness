"""Context engine factory — one engine.

There is a single :class:`ContextAssembler`. Per the context-builder
design it runs three lanes per turn (the prior ``legacy`` / ``curator`` /
``default`` split is gone):

- **Curator lane** — manifest build + fast / slow / fallback history
  selection + ``# Curator Working State``. Owns ``*history``.
- **Memory lane** — ``backend.recall(user_id=...)`` (segment 3,
  ``# Memory``).
- **Local Skill lane** — :class:`SkillForgeRouter` over the operator-managed
  Local Skill catalog (segment 5, ``# Skills``).
- **Host** — identity / bootstrap / always-skills, rendered by
  :class:`ContextBuilder`.

The SkillForgeRouter always wraps the builder's existing ``LocalPool`` and
``SkillRegistry`` without a second disk scan. Memory selection does not change
Local Skill availability.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from pico.agent.context import ContextBuilder
from pico.context_engine.assembler import ContextAssembler
from pico.context_engine.base import ContextEngine
from pico.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from pico.context_engine.segments.curator import CuratorSegmentBuilder
from pico.providers.base import LLMProvider

if TYPE_CHECKING:
    from pico.config.pico import (
        ContextConfig,
        MemoryConfig,
        SkillForgeConfig,
        SkillForgeRouterConfig,
    )
    from pico.memory_engine.backend import MemoryBackend
    from pico.memory_engine.skill_forge import SkillForgeRouter


class ContextEngineFactory(Protocol):
    def __call__(
        self,
        *,
        workspace: Path,
        config: "ContextConfig",
        builder: ContextBuilder,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
        get_tool_definitions: Callable[[], list[dict]],
        now_fn: Callable[[], datetime] | None = None,
        backend: "MemoryBackend | None" = None,
        memory_config: "MemoryConfig | None" = None,
        skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
        skill_forge_config: "SkillForgeConfig | None" = None,
    ) -> ContextEngine: ...


def build_context_engine(
    *,
    workspace: Path,
    config: "ContextConfig",
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None = None,
    backend: "MemoryBackend | None" = None,
    memory_config: "MemoryConfig | None" = None,
    skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
    skill_forge_config: "SkillForgeConfig | None" = None,
) -> ContextEngine:
    """Build the one :class:`ContextAssembler` from a flat SegmentBuilder list.

    ``config.engine`` is no longer a dispatch key — there is a single
    engine. The field is retained in :class:`ContextConfig` for config
    back-compat but is ignored here. ``builder`` is used only as the
    holder of the shared ``MemoryStore`` / ``LocalSkillCatalog`` until it
    is retired.
    """
    from pico.config.pico import (
        MemoryConfig as _MemoryConfig,
    )
    from pico.config.pico import (
        SkillForgeRouterConfig as _SkillForgeRouterConfig,
    )

    if memory_config is None:
        memory_config = _MemoryConfig()
    if skill_forge_router_config is None:
        skill_forge_router_config = _SkillForgeRouterConfig()

    router = _build_router(
        builder=builder,
        skill_forge_router_config=skill_forge_router_config,
    )
    configured_inject_max = int(getattr(skill_forge_config, "inject_max", 2)) if skill_forge_config is not None else 2
    summary_only = (
        skill_forge_config is not None and getattr(skill_forge_config, "injection_mode", "full_body") == "summary"
    )
    activation_max = 0 if summary_only else configured_inject_max or skill_forge_router_config.top_k

    builders = [
        IdentitySegmentBuilder(workspace, builder.state),
        BootstrapSegmentBuilder(builder.state),
        MemorySegmentBuilder(
            builder.memory,
            backend,
            user_id=memory_config.user_id,
            memory_top_k=memory_config.memory_top_k,
            enabled=backend is not None,
        ),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(
            router,
            skill_top_k=skill_forge_router_config.top_k,
            activation_max=activation_max,
        ),
        CuratorSegmentBuilder(
            workspace=builder.state,
            config=config,
            provider=provider,
            model=model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=get_tool_definitions,
            now_fn=now_fn,
            memory_enabled=backend is not None,
        ),
    ]
    return ContextAssembler(builders, get_tool_definitions, now_fn=now_fn)


def _build_router(
    *,
    builder: ContextBuilder,
    skill_forge_router_config: "SkillForgeRouterConfig",
) -> "SkillForgeRouter":
    """Assemble the Local Skill router for segment 5."""
    from pico.memory_engine.skill_forge import (
        LocalSkillSource,
        SkillForgeRouter,
    )

    local_source = LocalSkillSource(
        pool=builder.skills.pool,
        registry=builder.skills.registry,
        min_score=skill_forge_router_config.local_min_score,
    )
    return SkillForgeRouter(sources=[local_source])
