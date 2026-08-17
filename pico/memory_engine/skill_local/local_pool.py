"""LocalPool — BM25 keyword retrieval over file-based skills.

The "local" pool covers everything that lives as a SKILL.md file on disk:

  - workspace/skills/ (user-authored)
  - packaged builtin/ (the 9 shipped skills)
  - configured external and mirror directories

These pools are small (tens to a few hundred skills) and frequently edited,
so BM25 over the in-memory corpus is the right shape:
  - no embedding model loaded → starts in milliseconds
  - relevance is keyword-driven → user intent on a specific tool
    ("pdf" / "weather") matches better than dense semantic
  - re-tokenize per ``select`` is cheap at this scale

Index is built eagerly in ``__init__`` and refreshed via the public
``rebuild_index()`` — :class:`SkillService` calls it from
``invalidate_skill_cache`` so every file-watcher / evolver invalidation
flows through to the BM25 state. Steady-state ``search`` therefore
costs one query-side tokenize + one BM25 dot-product over precomputed
``doc_freqs``; the per-doc tokenize and IDF accumulation only run when
files actually changed.

BM25 + tokenization come from :mod:`pico.utils.bm25` (a self-contained
Okapi BM25, no ``rank_bm25`` / ``jieba`` / ``nltk`` dependency, with CJK-aware
tokenization) — shared with the agent tool catalog.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pico.memory_engine.skill_local.types import ScoredSkill, SkillMeta
from pico.utils.bm25 import BM25Okapi as _BM25Okapi
from pico.utils.bm25 import tokenize as _tokenize

if TYPE_CHECKING:
    from pico.memory_engine.skill_local.registry import SkillRegistry


def _format_skill_text(meta: SkillMeta, body_max: int = 4000) -> str:
    """One-line representation fed into the BM25 index. Heavier on signal
    fields (name, description) than body — ``"weather"`` should fire on
    the weather skill even when the body talks about HTTP and caching."""
    body = (meta.content or "")[:body_max]
    # 重复名称和描述，使它们在 BM25 词频中的权重高于较长正文。
    return f"{meta.name} {meta.name} {meta.description or ''} {body}"


class LocalPool:
    """BM25 retrieval wrapper around a file-based ``SkillRegistry``.

    Holds a prebuilt ``_BM25Okapi`` over the current registry contents.
    :class:`SkillService` calls :meth:`rebuild_index` from its
    ``invalidate_skill_cache`` hook so file-watcher events and evolver
    writes refresh the index directly, leaving ``search`` to a single
    query-side tokenize + BM25 dot product.

    Thread-safety: an internal :class:`threading.Lock` guards the
    ``(metas, _BM25Okapi)`` pair. ``rebuild_index`` does the expensive
    tokenize + BM25 construction *outside* the lock and only takes it
    for the atomic swap; ``search`` holds the lock only long enough
    to capture the two references, then scores + sorts outside.
    """

    def __init__(self, registry: "SkillRegistry") -> None:
        self._registry = registry
        self._metas: list[SkillMeta] = []
        self._bm25: _BM25Okapi | None = None
        # 使用普通 Lock 而非 RLock，因为方法之间不会重入。
        self._lock = threading.Lock()
        # 急切执行首次构建，与服务其余部分一致：提前支付磁盘遍历成本，
        # 不让第一次用户查询承担。
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Re-read the registry and rebuild the BM25 index in place.

        Called from :meth:`SkillService.invalidate_skill_cache` on every
        watcher event / evolver write, and once from ``__init__`` for the
        initial build. Idempotent and safe to call concurrently — the
        last writer's index wins; in-flight searches retain their
        previously captured references and finish against a consistent
        snapshot.
        """
        metas = self._registry.list_all()
        if not metas:
            with self._lock:
                self._metas = []
                self._bm25 = None
            return
        tokenized_corpus = [_tokenize(_format_skill_text(m)) for m in metas]
        bm25 = _BM25Okapi(tokenized_corpus)
        # 防御性复制：注册表的 ``list_all`` 按引用返回缓存列表，后续重建会替换而非就地修改它。
        # 复制后即可解耦，保证向读取方提供的快照不会与配对的 BM25 索引分离。
        metas_snapshot = list(metas)
        with self._lock:
            self._metas = metas_snapshot
            self._bm25 = bm25

    def search(self, query: str, top_k: int = 50) -> list[ScoredSkill]:
        """Return top-K matches by BM25 over the prebuilt index."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        with self._lock:
            bm25 = self._bm25
            metas = self._metas
        if bm25 is None or not metas:
            return []
        scores = bm25.get_scores(query_tokens)
        # 丢弃零分文档，按分数降序排列后取 top_k。排除 ``m.always`` Skill，因为它们已由
        # ``ContextBuilder`` 通过 ``get_always_skills()`` 注入 ``# Active Skills`` 块。如果仍在此处出现，
        # 同一正文会在系统提示词中重复：一次作为 Active，一次作为 top-K。Mass pool 项
        # 不会重复注入，因为 ``get_always_skills`` 只读取 ``_registry``，从不读取 ``_mass_registry``，
        # 所以该过滤仅对本地池有意义。
        ranked = sorted(
            ((s, m) for s, m in zip(scores, metas) if s > 0.0 and not m.always),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        return [ScoredSkill(name=m.name, score=float(s), source=m.source) for s, m in ranked]
