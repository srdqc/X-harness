"""LocalSkillSource — wraps :class:`LocalPool` to emit :class:`RouterHit`.

Hot-path observation: :class:`LocalPool` already returns the cheap
``ScoredSkill(name, score, source)`` triple. To produce a
:class:`RouterHit` we additionally need ``content`` (the SKILL.md body)
which :class:`SkillRegistry.get` already has cached in memory — so the
per-hit lookup is O(1) dict access, not a fresh disk read.

Hits whose name no longer resolves in the registry (race against a
file-watcher delete between BM25 scoring and meta lookup) are skipped
silently. The router's contract is "at most k hits", not "exactly k",
so dropping is safer than emitting a hit with empty content.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pico.memory_engine.skill_forge.types import RouterHit

if TYPE_CHECKING:
    from pico.memory_engine.skill_local.local_pool import LocalPool
    from pico.memory_engine.skill_local.registry import SkillRegistry


class LocalSkillSource:
    """SkillSource adapter for the BM25 local pool."""

    name: str = "local"
    weight: float = 1.0

    def __init__(
        self,
        pool: "LocalPool",
        registry: "SkillRegistry",
        *,
        min_score: float = 0.0,
    ) -> None:
        self._pool = pool
        self._registry = registry
        try:
            self._min_score = float(min_score)
        except (TypeError, ValueError):
            raise ValueError(
                "LocalSkillSource: min_score must be nonnegative and finite",
            ) from None
        if isinstance(min_score, bool) or self._min_score < 0 or not math.isfinite(self._min_score):
            raise ValueError(
                "LocalSkillSource: min_score must be nonnegative and finite",
            )

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        # ``history`` 尚未使用，本地 BM25 不依赖先前对话。未来更智能的本地排序器
        # 可以引入它；当前签名与协议一致，以保留扩展边界。
        del history

        hits = self._pool.search(query, top_k=k)
        out: list[RouterHit] = []
        for h in hits:
            if h.score < self._min_score:
                continue
            meta = self._registry.get(h.name, source=h.source)
            if meta is None:
                # 文件监视器竞态：Skill 在 BM25 快照与元数据查找之间消失。
                # 直接跳过，不输出只填充了部分字段的命中结果。
                continue
            # ``skill_dir`` 让 SkillsSegmentBuilder 的门控后填充步骤能解析 {baseDir}
            # 和 Markdown 链接引用，无需再查一次注册表。
            path_obj = getattr(meta, "path", None)
            path_str = str(path_obj) if path_obj is not None else ""
            skill_dir: str | None = None
            if path_obj is not None and not path_str.startswith("sqlite:"):
                skill_dir = str(path_obj.parent)
            out.append(
                RouterHit(
                    qualified_id=f"local/{h.name}",
                    name=h.name,
                    content=meta.content,
                    score=h.score,
                    meta={
                        "source": "local",
                        # Local 内的物理来源（``workspace``、``builtin``、``external`` 或 ``mirror/*``）
                        # 偶尔会用于遥测，因此保留在 meta 中。
                        "physical_source": h.source,
                        "always": meta.always,
                        "skill_dir": skill_dir,
                        "description": meta.description,
                    },
                ),
            )
        return out


__all__ = ["LocalSkillSource"]
