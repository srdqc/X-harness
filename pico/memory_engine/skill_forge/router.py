"""`SkillForgeRouter` 并发 Fan-out 所有 Sources，再用 :func:`rrf_merge_weighted` 融合排名。

Router 而非各 Source 统一执行 Two Policies：

- **Per-source Over-fetch**：:meth:`select(k)` 向每个 Source 请求 ``k * over_fetch_factor`` Hits，RRF 再
  收窄到 Overall K。某 Source 的 #3 可能是优秀 Cross-source Merge Candidate，即使单独看进不了 Top-3；
  Default Factor=2。
- **Single-source Failure Isolation**：Source Raise 会在 :meth:`_safe_search` 内转成该轮 Empty List，并
  记录 Failure Type；其他 Sources 继续 Feed RRF，避免一个 Transient 变成 Whole-pipeline Failure。

Source List 在 Construction 时固定引用。Active Runtime 连接 Single Local Source；Deterministic
Evaluators 可使用 Multiple Sources。Router Ranking 不包含 LLM Gate 或最终 Skill Hydration。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pico.decision_plane import (
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    DecisionType,
    DeterministicSkillRankingAdapter,
    validate_decision_result,
)
from pico.decision_plane.evidence import DecisionReceipt, emit_decision_receipt
from pico.memory_engine.skill_forge.fusion import rrf_merge_weighted
from pico.memory_engine.skill_forge.types import RouterHit, SkillSource
from pico.tracing import evidence as turn_evidence

if TYPE_CHECKING:
    from pico.decision_plane import DecisionAdapter

logger = logging.getLogger(__name__)


class SkillForgeRouter:
    """把 N 个 :class:`SkillSource` Outputs 组合为一个 Top-K Ranking。

    实例保存 Source Order、Over-fetch Factor 与 Dedup Field。Source List 按引用捕获，Host 启动后约定不再
    修改；`select` 并发调用并填充 Diagnostics，最后用 Weighted RRF 合并。Router 不验证 Skill
    Requirements，也不执行 Provider Gate。
    """

    def __init__(
        self,
        sources: list[SkillSource],
        *,
        over_fetch_factor: int = 2,
        dedup_by: str = "name",
        decision_plane_enabled: bool = False,
        decision_adapter: "DecisionAdapter | None" = None,
    ) -> None:
        # 列表按引用捕获；如需禁止修改，调用方应传入已冻结的元组。此处刻意不代为冻结：
        # 宿主在启动时连接数据源，之后不再修改，冗长的不可变包装没有收益。
        self._sources = sources
        self._over_fetch_factor = max(1, over_fetch_factor)
        self._dedup_by = dedup_by
        self._decision_plane_enabled = decision_plane_enabled
        self._baseline_adapter = DeterministicSkillRankingAdapter()
        self._decision_adapter = decision_adapter or self._baseline_adapter

    async def select(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int = 5,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[RouterHit]:
        """并发 Fan Out 每个 Source，并 Fuse 成 Top-K。

        每个 Source 获得同一 `query`、`history` 与 Over-fetched K。`diagnostics` 提供时写入
        ``failed_sources`` / ``failure_types``；Source Failure 仍可返回其他结果。`k` 传给最终 RRF 限制输出，
        空 Sources 或全失败得到空列表，而不是异常。
        """
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
        deterministic = rrf_merge_weighted(
            [(s.name, s.weight, hits) for s, (hits, _) in zip(self._sources, per_source)],
            k=k,
            dedup_by=self._dedup_by,
        )
        if not self._decision_plane_enabled:
            # The default path deliberately does not construct a request or call
            # an adapter.  This preserves the pre-P2 ranking and latency path.
            return deterministic
        return await self._apply_decision_ranking(
            query=query,
            deterministic=deterministic,
            diagnostics=diagnostics,
        )

    async def _apply_decision_ranking(
        self,
        *,
        query: str,
        deterministic: list[RouterHit],
        diagnostics: dict[str, Any] | None,
    ) -> list[RouterHit]:
        """Apply one advisory ordering or return the authoritative baseline."""

        try:
            recorder = turn_evidence.current()
            request = DecisionRequest(
                decision_type=DecisionType.SKILL_RANKING,
                query=query[:4096],
                candidates=tuple(
                    DecisionCandidate(
                        candidate_id=hit.qualified_id,
                        name=hit.name,
                        description=str(hit.meta.get("description") or "")[:1024],
                        source=str(hit.meta.get("source") or hit.qualified_id.partition("/")[0] or "unknown"),
                        deterministic_rank=rank,
                        deterministic_score=float(hit.meta.get("rrf_score", hit.score)),
                    )
                    for rank, hit in enumerate(deterministic, start=1)
                ),
                decision_id=(
                    recorder.next_identity("decision")
                    if recorder is not None
                    else f"decision:{uuid.uuid4().hex}"
                ),
                correlation_id=recorder.turn_id if recorder is not None else None,
            )
            baseline = await self._baseline_adapter.decide(request)
        except (TypeError, ValueError) as exc:
            self._record_decision_diagnostic(
                diagnostics,
                status="invalid",
                used_fallback=True,
                reason=f"invalid_request:{type(exc).__name__}",
            )
            return deterministic

        result: DecisionResult
        started_ns = time.perf_counter_ns()
        try:
            result = (
                baseline
                if self._decision_adapter is self._baseline_adapter
                else await self._decision_adapter.decide(request)
            )
            invalid_reason = validate_decision_result(request, result)
        except Exception as exc:
            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            self._record_decision_evidence(
                diagnostics,
                status="error",
                used_fallback=True,
                reason=f"adapter_exception:{type(exc).__name__}",
                request=request,
                baseline=baseline,
                latency_ms=latency_ms,
            )
            return deterministic

        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if invalid_reason is not None:
            fallback_reason = (
                result.reason
                if getattr(self._decision_adapter, "kind", None) == "jev" and result.reason
                else invalid_reason
            )
            self._record_decision_evidence(
                diagnostics,
                status=result.outcome.value,
                used_fallback=True,
                reason=fallback_reason,
                request=request,
                result=result,
                baseline=baseline,
                latency_ms=latency_ms,
            )
            return deterministic

        by_id = {hit.qualified_id: hit for hit in deterministic}
        ranked = [by_id[item.candidate_id] for item in result.ranking]
        self._record_decision_evidence(
            diagnostics,
            status=result.outcome.value,
            used_fallback=False,
            reason=None,
            request=request,
            result=result,
            baseline=baseline,
            latency_ms=latency_ms,
        )
        return ranked

    def _record_decision_evidence(
        self,
        diagnostics: dict[str, Any] | None,
        *,
        status: str,
        used_fallback: bool,
        reason: str | None,
        request: DecisionRequest | None = None,
        result: DecisionResult | None = None,
        baseline: DecisionResult | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        if request is None or baseline is None:
            self._record_decision_diagnostic(
                diagnostics,
                status=status,
                used_fallback=used_fallback,
                reason=reason,
            )
            return

        recorder = turn_evidence.current()
        final_result = baseline if used_fallback else result
        if final_result is None:  # defensive: the successful path always has a result
            final_result = baseline
        accepted_cost = result.cost if result is not None and not used_fallback else baseline.cost
        confidences = (
            tuple(
                (item.candidate_id, item.confidence)
                for item in result.ranking
                if item.confidence is not None
            )
            if result is not None and not used_fallback
            else ()
        )
        adapter_kind = str(
            getattr(
                self._decision_adapter,
                "kind",
                result.source if result is not None else type(self._decision_adapter).__name__,
            )
        )
        receipt = DecisionReceipt(
            turn_id=recorder.turn_id if recorder is not None else None,
            decision_id=request.decision_id,
            decision_type=request.decision_type.value,
            adapter_kind=adapter_kind,
            candidate_count=len(request.candidates),
            candidate_set_digest=request.candidate_set_digest,
            request_digest=request.request_digest,
            baseline_result_digest=baseline.result_digest,
            adapter_result_digest=result.result_digest if result is not None else None,
            final_result_digest=final_result.result_digest,
            adapter_outcome=status,
            fallback_used=used_fallback,
            fallback_reason=reason,
            latency_ms=latency_ms,
            cost_available=accepted_cost.available,
            cost_amount=accepted_cost.amount,
            cost_unit=accepted_cost.unit,
            confidences=confidences,
            final_ranking_source=final_result.source,
        )
        emit_decision_receipt(receipt)
        if diagnostics is not None:
            diagnostics["decision_plane"] = dict(receipt.metadata())
            diagnostics["decision_plane"].update(
                {
                    "status": status,
                    "source": result.source if result is not None else None,
                    "used_fallback": used_fallback,
                    "result_digest": result.result_digest if result is not None else None,
                }
            )

    @staticmethod
    def _record_decision_diagnostic(
        diagnostics: dict[str, Any] | None,
        *,
        status: str,
        used_fallback: bool,
        reason: str | None,
    ) -> None:
        if diagnostics is not None:
            diagnostics["decision_plane"] = {
                "decision_id": None,
                "decision_type": DecisionType.SKILL_RANKING.value,
                "status": status,
                "source": None,
                "used_fallback": used_fallback,
                "fallback_reason": reason,
                "request_digest": None,
                "result_digest": None,
                "baseline_result_digest": None,
            }

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
