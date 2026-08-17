"""MemoryBackend Protocol — the single contract every memory plugin implements.

MB-1 introduction. This is the **new** seam between AgentLoop and the
memory subsystem, deliberately distinct from the older
:class:`MemoryEngine` ABC in :mod:`pico.memory_engine.base` so the
two can coexist while the codebase transitions.

Three design points to flag for plugin authors:

- ``recall`` names the track explicitly: it takes ``user_id`` XOR
  ``agent_id``. The active host uses the user track for Memory; the agent
  track remains in the public plugin protocol for compatibility.

- ``Memory.metadata`` is the **escape hatch**. ``text`` and ``score``
  are normalized; everything backend-specific (categories, episode
  type, native id, source labels) goes in ``metadata``. The host's
  context assembler does **not** read ``metadata`` — only the
  pre-rendered ``text`` lands in the prompt. Skill-source adapters
  that re-emit Memory hits as ScoredSkill do read ``metadata`` for
  qualified-id construction.

- ``feedback`` is **allowed to be a no-op**. It remains in the public plugin
  protocol for compatibility, but the active host does not dispatch it.

The Protocol is :func:`typing.runtime_checkable` so ``isinstance(x,
MemoryBackend)`` works in tests — at the cost of accepting any class
whose surface matches, including duck-typed mocks. That's the trade we
want: contract tests don't have to inherit from a base class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# 数据载体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    """One hit returned by :meth:`MemoryBackend.recall`.

    ``frozen=True`` so the host can hand a list of these around without
    worrying about adapter code mutating someone else's view.
    """

    text: str
    """Pre-rendered content the LLM sees verbatim in the prompt.

    Adapters are responsible for formatting (e.g. EverMem returns
    natural-sentence facts; mem0 returns category-tagged blobs). The
    host **never** post-processes ``text`` except to join multiple
    hits into a block."""

    score: float = 0.0
    """Relevance normalized to ``[0, 1]`` by the adapter. Used by
    :class:`SkillForgeRouter` for cross-source RRF when a Memory hit is
    re-emitted as a ScoredSkill. For plain ``# Recalled memory``
    injection, ``score`` is informational only."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific escape hatch. Examples by backend:

    - EverMem: ``{"id": ..., "episode_type": ..., "name": ...,
      "owner_type": "user" | "agent"}``
    - mem0: ``{"id": ..., "categories": [...], "memory_type": ...}``
    - MemOS: ``{"id": ..., "mem_cube_id": ..., "metadata": {...}}``
    - Letta: ``{"archival_memory_id": ...}``
    """


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryBackend(Protocol):
    """The single contract every memory plugin implements.

    Five methods, ordered by hot-path:

    1. :meth:`recall` — called by ``ContextEngine.assemble`` every turn
       for user-track memory with ``user_id``.
    2. :meth:`store` — called by AgentLoop after each turn to persist
       the conversation slice.
    3. :meth:`feedback` — compatibility hook that may be a no-op.
    4. :meth:`start` / :meth:`stop` — lifecycle, awaited by the host.
    """

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        """Retrieve memories matching ``query`` for one track.

        Exactly one of ``user_id`` / ``agent_id`` is set (XOR) — the
        caller knows which track it wants at construction time, so the
        track is named explicitly rather than smuggled through a
        prefixed opaque string. Backends may use ``user_id`` and return
        ``[]`` for the ``agent_id`` call.
        Passing neither or both is a caller bug — return ``[]``.

        Empty result is a valid response (no hits); raise on transport
        errors, auth failures, etc. The host will fall back to other
        sources via ``SkillForgeRouter._safe_search``.
        """
        ...

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Persist a session slice.

        ``messages`` follows the AgentLoop ``{"role", "content", ...}``
        shape — the existing list-of-dicts form the codebase already
        produces, so adapters don't need a conversion step. Backends
        that want to chunk / deduplicate / extract are free to; the
        Protocol is fire-and-forget per call.

        Raises on transport / auth errors so AgentLoop can surface
        them; the host does **not** silently swallow store failures.
        """
        ...

    async def feedback(self, signals: dict[str, Any]) -> None:
        """Consume a free-form signal dict (e.g. injected/used skill ids).

        A no-op implementation is fully valid and idiomatic.
        """
        ...

    async def start(self) -> None:
        """One-time / idempotent initialization (open connections,
        warm caches, run migrations). The host awaits this exactly
        once during agent boot; failures abort startup."""
        ...

    async def stop(self) -> None:
        """One-time / idempotent teardown. Adapters should make this
        safe to call after a failed ``start`` (so partial-init state
        cleans up)."""
        ...


__all__ = ["Memory", "MemoryBackend"]
