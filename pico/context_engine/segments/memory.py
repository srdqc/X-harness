"""Segment 3 - host memory plus plugin recall.

The one composite segment: a single ``# Memory`` heading whose body
merges the host's slow-changing ``user.md`` dump with the backend's
query-conditioned recall hits. Two contributing sources, one owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pico.context_engine.base import AssemblyContext, Segment
from pico.context_engine.segments import render
from pico.tracing import semconv, trace

if TYPE_CHECKING:
    from pico.memory_engine.backend import MemoryBackend
    from pico.memory_engine.consolidate.consolidator import MemoryStore


class MemorySegmentBuilder:
    name = "memory"
    order = 3
    needs_prefix = False

    def __init__(
        self,
        memory_store: "MemoryStore",
        backend: "MemoryBackend | None" = None,
        user_id: str = "default",
        memory_top_k: int = 5,
        enabled: bool = True,
    ) -> None:
        self._memory_store = memory_store
        self._backend = backend
        self._user_id = user_id
        self._memory_top_k = memory_top_k
        self._enabled = enabled

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if not self._enabled:
            return Segment(text="", meta={"memory_hits": 0})
        # 合并 Host 直接读取和插件召回。召回发生硬失败时继续向上传播，
        # 让后端故障在 AgentLoop 暴露，而不是静默丢弃记忆。
        host = self._memory_store.get_memory_context(current_message=ctx.current_message)
        recall_hits = await self._recall(ctx.current_message)
        recall_bullets = render.render_recalled_memory(recall_hits)

        sections = [s for s in (host, recall_bullets) if s]
        meta: dict[str, Any] = {"memory_hits": len(recall_hits)}
        if not sections:
            return Segment(text="", meta=meta)
        return Segment(text="# Memory\n\n" + "\n\n".join(sections), meta=meta)

    @trace.instrument("memory.recall", extract=semconv.memory_recall)
    async def _recall(self, query: str) -> list[Any]:
        if self._backend is None:
            return []
        return list(
            await self._backend.recall(
                query=query,
                user_id=self._user_id,
                top_k=self._memory_top_k,
            )
        )
