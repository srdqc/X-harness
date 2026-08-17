"""SkillForgeRouter — fans :meth:`select` out to every registered source and
fuses the per-source rankings via :func:`rrf_merge_weighted`.

Two policies the router enforces (not its sources):

- **Per-source over-fetch.** :meth:`select(k)` asks every source for
  ``k * over_fetch_factor`` hits. RRF then narrows to ``k`` overall.
  Over-fetching matters because a source's #3 hit might be a great
  cross-source merge candidate even if it would never be a top-3 by
  itself. Default factor is 2 — twice the requested ``k``.

- **Single-source failure isolation.** A source that raises is caught inside
  :meth:`_safe_search` and turns into an empty list for that round.
  The other sources still feed RRF so the router never produces a
  whole-pipeline failure because of one transient.

The router's source list is fixed at construction. The active Runtime wires a
single Local source; deterministic evaluators may exercise multiple sources.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pico.memory_engine.skill_forge.fusion import rrf_merge_weighted
from pico.memory_engine.skill_forge.types import RouterHit, SkillSource

logger = logging.getLogger(__name__)


class SkillForgeRouter:
    """Compose N :class:`SkillSource` outputs into one top-K ranking."""

    def __init__(
        self,
        sources: list[SkillSource],
        *,
        over_fetch_factor: int = 2,
        dedup_by: str = "name",
    ) -> None:
        # 列表按引用捕获；如需禁止修改，调用方应传入已冻结的元组。此处刻意不代为冻结：
        # 宿主在启动时连接数据源，之后不再修改，冗长的不可变包装没有收益。
        self._sources = sources
        self._over_fetch_factor = max(1, over_fetch_factor)
        self._dedup_by = dedup_by

    async def select(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int = 5,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[RouterHit]:
        """Fan out to every source concurrently, fuse to top-K."""
        per_source_k = k * self._over_fetch_factor
        per_source = await asyncio.gather(*[self._safe_search(s, query, history, per_source_k) for s in self._sources])
        if diagnostics is not None:
            diagnostics["failed_sources"] = [
                source.name for source, (_, error_type) in zip(self._sources, per_source) if error_type is not None
            ]
            diagnostics["failure_types"] = {
                source.name: error_type
                for source, (_, error_type) in zip(self._sources, per_source)
                if error_type is not None
            }
        return rrf_merge_weighted(
            [(s.name, s.weight, hits) for s, (hits, _) in zip(self._sources, per_source)],
            k=k,
            dedup_by=self._dedup_by,
        )

    async def _safe_search(
        self,
        source: SkillSource,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> tuple[list[RouterHit], str | None]:
        try:
            return await source.search(query, history, k), None
        except Exception as e:
            # ``exception()`` 会写入堆栈追踪；使用警告级别，让短暂抖动不会淹没 ``error`` 日志，
            # 同时仍能进入常规聚合。
            logger.warning(
                "skill source %r failed; treating as empty: %s",
                source.name,
                e,
            )
            return [], type(e).__name__


__all__ = ["SkillForgeRouter"]
