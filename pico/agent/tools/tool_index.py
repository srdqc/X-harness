"""为 Agent Tool Catalog 建立 CJK-aware BM25 Keyword Index。

该索引支撑 ``tool_search``：查询同时匹配 Tool name、description 与 parameter schema，复用
:mod:`pico.utils.bm25` 的 Okapi BM25。Name 在索引文本中重复三次，让直接命中名称优先于正文
偶然出现，沿用 Skill ``LocalPool`` 的 field-weighting；Property name、description 和 enum 也
进入 body，因为 Repo owner、Channel id、Image size 等区分词常只存在于 Schema。

Catalog 的完整 ``(name, description, parameters)`` 文本集合形成 Signature。每次 ``ensure`` 会
做便宜的 flatten 比较，只有 Startup、MCP connect 或 hot-reload 真正改变已索引字段时才重建；
Steady-state ``search`` 只需 query tokenize 与预计算 ``doc_freqs`` 上的 BM25 score。

Process-level 单槽缓存最近 Built Index，使 Resident Process 创建的多个 short-lived agent loops
复用相同 BM25；Subagents 使用独立最小 Tool set，所以一个主 Signature 通常占主导。Build 在锁外
完成，Slot swap 有锁；罕见并发重复 Build 是纯且同结果的可接受工作。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pico.agent.tools.discovery import ToolDiscoveryMetadata, ToolSourceKind
from pico.utils.bm25 import BM25Okapi, tokenize

if TYPE_CHECKING:
    from pico.agent.tools.base import Tool

# 目录签名：每个工具对应一个（名称，索引文本）对。索引文本嵌入描述和参数模式，
# 任何已索引字段变化时签名都会变化，这正是重建的触发条件。
_Signature = frozenset[tuple[str, str]]

# 限制模式遍历的递归深度，避免病态嵌套的 MCP 模式撑爆栈或使索引文本膨胀；
# 实际的工具模式层级很浅。
_MAX_SCHEMA_DEPTH = 6
# 已构建的索引：名称列表与按相同顺序构建的 BM25 配对。检索方只读，可安全跨循环共享。
_BuiltIndex = tuple[list[str], BM25Okapi]

# 单一进程级槽位由独立锁保护。昂贵的构建在锁外执行，锁只保护槽位读写。
# 首次未命中时少见的并发重复构建无害：构建是纯操作，最后写入者得到相同索引，
# 与 ``LocalPool`` 的容忍策略一致。
_cache_lock = threading.Lock()
_cached_sig: _Signature = frozenset()
_cached_index: _BuiltIndex | None = None


@dataclass(frozen=True)
class ToolRetrievalHit:
    rank: int
    score: float
    metadata: ToolDiscoveryMetadata
    base_score: float
    role_prior: float = 0.0
    final_score: float | None = None


@dataclass(frozen=True)
class ToolRetrievalResult:
    query: str
    catalog_signature: str
    total_searchable_tools: int
    requested_k: int
    effective_k: int
    hits: tuple[ToolRetrievalHit, ...]
    selected_role: str = "general"

    @property
    def ranked_names(self) -> tuple[str, ...]:
        return tuple(hit.metadata.name for hit in self.hits)

    @property
    def zero_hit(self) -> bool:
        return not self.hits

    def rank_of(self, target_name: str) -> int | None:
        return next(
            (hit.rank for hit in self.hits if hit.metadata.name == target_name),
            None,
        )

    def recall_at(self, target_names: set[str] | frozenset[str], k: int) -> float | None:
        if not target_names:
            return None
        visible = {hit.metadata.name for hit in self.hits[: max(k, 0)]}
        return len(visible & target_names) / len(target_names)

    def reciprocal_rank(self, target_name: str) -> float:
        rank = self.rank_of(target_name)
        return 0.0 if rank is None else 1.0 / rank


def _schema_text(parameters: "dict[str, Any] | None") -> str:
    """从 JSON-Schema Parameter block 提取具备搜索信号的 Natural-language keywords。

    函数递归收集 property names、``description`` strings、enum values 与 array items，跳过 Types
    与 structural keywords，因为它们无法区分具体能力。递归深度受 `_MAX_SCHEMA_DEPTH` 限制，
    非 dict 输入返回空字符串；结果只用于索引，不修改模型实际看到的 Schema。
    """
    if not isinstance(parameters, dict):
        return ""
    parts: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_SCHEMA_DEPTH or not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                parts.append(str(key))
                walk(sub, depth + 1)
        desc = node.get("description")
        if isinstance(desc, str):
            parts.append(desc)
        enum = node.get("enum")
        if isinstance(enum, list):
            parts.extend(str(v) for v in enum if isinstance(v, (str, int, float)))
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, depth + 1)

    walk(parameters, 0)
    return " ".join(parts)


def _format_tool_text(tool: "Tool") -> str:
    """生成单 Tool Indexed Text：name ×3，description 与 parameter-schema keywords ×1。

    这种 TF-based field weighting 让 Query 直接命中 Tool name 时高于 body 偶然命中，同时仍能
    通过参数描述召回能力。空 Description 被当作空文本，末尾空白移除；函数不 tokenize，返回
    字符串会同时用于 Signature 与 BM25 Corpus。
    """
    name = tool.name
    body = f"{tool.description or ''} {_schema_text(tool.parameters)}".strip()
    return f"{name} {name} {name} {body}".rstrip()


def _get_or_build(sig: _Signature, items: list[tuple[str, str]]) -> _BuiltIndex:
    """返回 ``sig`` 对应的 Shared prebuilt index，Cache miss 时构建并发布。

    ``items`` 是 Caller 每次 ``ensure`` 已 flattened 的 ``(name, indexed-text)`` pairs，同一文本
    既算 Signature 又构建 Corpus，不重复读取 Tool。Slot hit 在锁内直接返回；miss 时在锁外 tokenize/build，最后
    短暂加锁替换 Cache。并发 miss 可能重复纯 Build，但不会返回半成品 Pair。
    """
    global _cached_sig, _cached_index
    with _cache_lock:
        if sig == _cached_sig and _cached_index is not None:
            return _cached_index
    names = [name for name, _ in items]
    corpus = [tokenize(text) for _, text in items]
    built: _BuiltIndex = (names, BM25Okapi(corpus))
    with _cache_lock:
        _cached_sig, _cached_index = sig, built
    return built


class ToolIndex:
    """维护 Tool Catalog 的 Prebuilt BM25，并在 name/description/parameters 变化时重建。

    Thread-safety 由实例 lock 保护 ``(names, BM25Okapi)`` Pair。昂贵 Build 委托共享
    ``_get_or_build`` slot，实例锁只用于 atomic swap 与 ``search`` 捕获一致引用。仅实例或注册
    顺序变化但完整 Signature 相同时可复用 Process cache。

    `ensure` 必须在 Search 前同步 Live Registry；Search 返回名称而非 Tool instance，使 Controller
    可再次以 Registry 为 Source of Truth 处理热删除。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: list[str] = []
        self._sig: _Signature = frozenset()
        self._bm25: BM25Okapi | None = None
        self._metadata: dict[str, ToolDiscoveryMetadata] = {}
        self._catalog_signature = _catalog_signature(())

    def ensure(
        self,
        tools: "list[Tool]",
        *,
        metadata: list[ToolDiscoveryMetadata] | None = None,
    ) -> None:
        """仅当 Tool ``(name, description, parameters)`` Set 变化时采用新 Shared Index。

        先 Flatten 当前 Tool 并计算 frozenset Signature；与实例现有 Signature 相同且 BM25 已存在
        时 no-op。否则获取或构建 Pair，再在实例锁内同时更新 names、bm25、sig，Search 不会观察
        到不匹配的名称与分数数组。
        """
        items = [(t.name, _format_tool_text(t)) for t in tools]
        sig: _Signature = frozenset(items)
        projected = (
            metadata
            if metadata is not None
            else [
                ToolDiscoveryMetadata(
                    name=tool.name,
                    description=tool.description,
                    source_kind=getattr(tool, "discovery_source_kind", ToolSourceKind.UNKNOWN),
                    source_id=getattr(tool, "discovery_source_id", None),
                    category=getattr(tool, "discovery_category", None),
                    effect=tool.capability.effect,
                )
                for tool in tools
            ]
        )
        metadata_by_name = {item.name: item for item in projected}
        if (
            len(metadata_by_name) != len(projected)
            or set(metadata_by_name) != {name for name, _ in items}
        ):
            raise ValueError("metadata must describe each indexed Tool exactly once")
        catalog_signature = _catalog_signature(
            (text, metadata_by_name[name]) for name, text in items
        )
        with self._lock:
            if sig == self._sig and self._bm25 is not None:
                self._metadata = metadata_by_name
                self._catalog_signature = catalog_signature
                return
        names, bm25 = _get_or_build(sig, items)
        with self._lock:
            self._names, self._bm25, self._sig = names, bm25, sig
            self._metadata = metadata_by_name
            self._catalog_signature = catalog_signature

    def search(self, query: str, limit: int) -> list[str]:
        """按 BM25 排名返回最多 ``limit`` 个 Tool names，并丢弃 zero-score hit。

        Query 使用共享 CJK-aware tokenizer；Index 未建立或没有 Token 时返回空列表。方法在锁内
        只捕获 BM25/names 引用，评分与排序在锁外完成，按 Score 降序返回。结果不带 Description/
        Schema，Controller 会从 Live Registry 再组装。
        """
        return list(self.retrieve(query, limit).ranked_names)

    def retrieve(self, query: str, limit: int) -> ToolRetrievalResult:
        """Return ranked scores and immutable catalogue metadata for diagnostics."""
        tokens = tokenize(query)
        with self._lock:
            bm25 = self._bm25
            names = tuple(self._names)
            metadata = dict(self._metadata)
            catalog_signature = self._catalog_signature
        requested_k = limit
        effective_k = min(max(limit, 0), len(names))
        if bm25 is None or not tokens or effective_k == 0:
            return ToolRetrievalResult(
                query=query,
                catalog_signature=catalog_signature,
                total_searchable_tools=len(names),
                requested_k=requested_k,
                effective_k=effective_k,
                hits=(),
            )
        scores = bm25.get_scores(tokens)
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        positive = [(name, float(score)) for name, score in ranked if score > 0.0][:effective_k]
        return ToolRetrievalResult(
            query=query,
            catalog_signature=catalog_signature,
            total_searchable_tools=len(names),
            requested_k=requested_k,
            effective_k=effective_k,
            hits=tuple(
                ToolRetrievalHit(
                    rank=rank,
                    score=score,
                    metadata=metadata[name],
                    base_score=score,
                    final_score=score,
                )
                for rank, (name, score) in enumerate(positive, start=1)
            ),
        )


def _catalog_signature(
    items: Iterable[tuple[str, ToolDiscoveryMetadata]],
) -> str:
    payload = [
        {
            "indexed_text": text,
            "name": metadata.name,
            "description": metadata.description,
            "source_kind": metadata.source_kind.value,
            "source_id": metadata.source_id,
            "category": metadata.category,
            "effect": metadata.effect.value,
        }
        for text, metadata in items
    ]
    encoded = json.dumps(
        sorted(payload, key=lambda item: item["name"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["ToolIndex", "ToolRetrievalHit", "ToolRetrievalResult"]
