"""Optional Jev refinement over the deterministic TASK_RELEVANCE_V1 result."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from pico.decision_plane.utility import (
    MAX_UTILITY_QUERY_CHARS,
    MAX_UTILITY_SUMMARY_CHARS,
    MAX_UTILITY_TITLE_CHARS,
    JevUtilityCandidate,
    JevUtilityDecisionAdapter,
    JevUtilityDecisionReceipt,
    JevUtilityRequest,
    UtilityDecision,
    emit_utility_receipt,
    measured_latency_ms,
)
from pico.tracing import evidence

from .relevance import SELECTOR_VERSION, TaskRelevanceQuery
from .retrieval import RetrievedKnowledge


@dataclass(frozen=True)
class UtilityRefinement:
    items: tuple[RetrievedKnowledge, ...]
    retained_candidate_ids: tuple[str, ...]
    abstained_candidate_ids: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None


@dataclass
class _PendingTurn:
    query: str
    repository_scope_id: str
    groups: dict[str, tuple[RetrievedKnowledge, ...]] = field(default_factory=dict)
    futures: dict[str, asyncio.Future[UtilityRefinement]] = field(default_factory=dict)


class KnowledgeUtilityCoordinator:
    """Coalesce concurrent P3 lanes into at most one Jev request per Turn."""

    def __init__(self, adapter: JevUtilityDecisionAdapter) -> None:
        self._adapter = adapter
        self._pending: dict[str, _PendingTurn] = {}
        self._completed: set[str] = set()
        self._completed_order: deque[str] = deque()

    async def refine(
        self,
        items: tuple[RetrievedKnowledge, ...],
        *,
        query: str,
        turn_id: str | None,
        repository_scope_id: str,
        group_id: str,
    ) -> UtilityRefinement:
        key = turn_id or f"diagnostic:{group_id}"
        if key in self._completed:
            ids = tuple(item.candidate.candidate_id for item in items)
            return UtilityRefinement(items, ids, (), True, "late_candidate_group")
        loop = asyncio.get_running_loop()
        pending = self._pending.get(key)
        if pending is None:
            pending = _PendingTurn(query[:MAX_UTILITY_QUERY_CHARS], repository_scope_id)
            self._pending[key] = pending
            loop.create_task(self._flush(key, turn_id))
        if pending.repository_scope_id != repository_scope_id or pending.query != query[:MAX_UTILITY_QUERY_CHARS]:
            ids = tuple(item.candidate.candidate_id for item in items)
            return UtilityRefinement(items, ids, (), True, "turn_input_mismatch")
        token = f"{group_id}:{len(pending.groups)}"
        future: asyncio.Future[UtilityRefinement] = loop.create_future()
        pending.groups[token] = items
        pending.futures[token] = future
        return await future

    async def _flush(self, key: str, turn_id: str | None) -> None:
        # Context phase-A lanes are concurrent. Two scheduling points let both
        # relevance selections join the same bounded request without a timer.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        pending = self._pending.pop(key)
        try:
            await self._resolve_pending(key, pending, turn_id)
        except Exception:  # noqa: BLE001 -- optional utility must never stall/fail a Turn
            self._fallback_pending(pending, "utility_internal_error")
            self._emit_internal_fallback(turn_id, pending)
        finally:
            self._mark_completed(key)

    async def _resolve_pending(
        self, key: str, pending: _PendingTurn, turn_id: str | None
    ) -> None:
        combined_by_id: dict[str, RetrievedKnowledge] = {}
        for values in pending.groups.values():
            for item in values:
                combined_by_id[item.candidate.candidate_id] = item
        combined = tuple(combined_by_id[candidate_id] for candidate_id in sorted(combined_by_id))
        if not combined:
            self._emit_skipped(turn_id, pending.repository_scope_id)
            summary = UtilityRefinement((), (), (), False, "skipped_no_relevant_candidates")
            for future in pending.futures.values():
                future.set_result(summary)
            return

        relevance_query = TaskRelevanceQuery.from_task(pending.query)
        recorder = evidence.current()
        decision_id = (
            recorder.next_identity("jev_utility_decision")
            if recorder is not None and recorder.turn_id == turn_id
            else f"jev-utility:{key}"
        )
        request = JevUtilityRequest(
            decision_id=decision_id,
            turn_id=turn_id,
            repository_scope_id=pending.repository_scope_id,
            query=pending.query,
            query_digest=relevance_query.query_digest,
            candidates=tuple(
                JevUtilityCandidate(
                    candidate_id=item.candidate.candidate_id,
                    candidate_type=item.candidate.candidate_type.value,
                    title=item.candidate.title[:MAX_UTILITY_TITLE_CHARS],
                    summary=item.candidate.reusable_content[:MAX_UTILITY_SUMMARY_CHARS],
                    relevance_rank=item.rank,
                    relevance_score=item.score,
                    applicability_digest=item.applicability.applicability_digest,
                )
                for item in combined
            ),
        )
        started = time.perf_counter_ns()
        result = await self._adapter.decide(request)
        latency_ms = measured_latency_ms(started)
        fallback_used = result.outcome.value != "success"
        fallback_reason = result.reason if fallback_used else None
        keep_ids = (
            {
                item.candidate_id
                for item in result.decisions
                if item.effective_decision is UtilityDecision.KEEP
            }
            if not fallback_used
            else {item.candidate.candidate_id for item in combined}
        )
        input_ids = tuple(item.candidate.candidate_id for item in combined)
        retained_ids = tuple(candidate_id for candidate_id in input_ids if candidate_id in keep_ids)
        abstained_ids = tuple(candidate_id for candidate_id in input_ids if candidate_id not in keep_ids)
        emit_utility_receipt(
            JevUtilityDecisionReceipt(
                turn_id=turn_id,
                decision_id=request.decision_id,
                repository_scope_id=pending.repository_scope_id,
                backend_id=result.backend_id or self._adapter.backend_id,
                backend_model=result.backend_model or self._adapter.backend_model,
                backend_version=result.backend_version or self._adapter.backend_version,
                config_digest=self._adapter.config_digest,
                request_digest=request.request_digest,
                input_candidate_ids=input_ids,
                retained_candidate_ids=retained_ids,
                abstained_candidate_ids=abstained_ids,
                decisions=result.decisions,
                latency_ms=latency_ms,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                utility_logical_calls=result.logical_calls,
                utility_provider_attempts=result.provider_attempts,
                utility_input_tokens=result.input_tokens,
                utility_output_tokens=result.output_tokens,
                utility_provider_latency_ms=result.latency_ms,
                utility_logical_call_id=result.logical_call_id,
            )
        )
        for token, values in pending.groups.items():
            kept = tuple(item for item in values if item.candidate.candidate_id in keep_ids)
            value_ids = tuple(item.candidate.candidate_id for item in values)
            pending.futures[token].set_result(
                UtilityRefinement(
                    kept,
                    tuple(candidate_id for candidate_id in value_ids if candidate_id in keep_ids),
                    tuple(candidate_id for candidate_id in value_ids if candidate_id not in keep_ids),
                    fallback_used,
                    fallback_reason,
                )
            )

    @staticmethod
    def _fallback_pending(pending: _PendingTurn, reason: str) -> None:
        for token, values in pending.groups.items():
            ids = tuple(item.candidate.candidate_id for item in values)
            if not pending.futures[token].done():
                pending.futures[token].set_result(
                    UtilityRefinement(values, ids, (), True, reason)
                )

    def _mark_completed(self, key: str) -> None:
        self._completed.add(key)
        self._completed_order.append(key)
        while len(self._completed_order) > 256:
            self._completed.discard(self._completed_order.popleft())

    def _emit_skipped(self, turn_id: str | None, repository_scope_id: str) -> None:
        recorder = evidence.current()
        decision_id = (
            recorder.next_identity("jev_utility_decision")
            if recorder is not None and recorder.turn_id == turn_id
            else f"jev-utility:{turn_id or 'diagnostic'}:skipped"
        )
        emit_utility_receipt(
            JevUtilityDecisionReceipt(
                turn_id=turn_id,
                decision_id=decision_id,
                repository_scope_id=repository_scope_id,
                backend_id=self._adapter.backend_id,
                backend_model=self._adapter.backend_model,
                backend_version=self._adapter.backend_version,
                config_digest=self._adapter.config_digest,
                request_digest=None,
                input_candidate_ids=(),
                retained_candidate_ids=(),
                abstained_candidate_ids=(),
                decisions=(),
                latency_ms=0.0,
                fallback_used=False,
                fallback_reason="skipped_no_relevant_candidates",
                selector_version=SELECTOR_VERSION,
            )
        )

    def _emit_internal_fallback(
        self, turn_id: str | None, pending: _PendingTurn
    ) -> None:
        recorder = evidence.current()
        decision_id = (
            recorder.next_identity("jev_utility_decision")
            if recorder is not None and recorder.turn_id == turn_id
            else f"jev-utility:{turn_id or 'diagnostic'}:fallback"
        )
        input_ids = tuple(
            sorted(
                {
                    item.candidate.candidate_id
                    for values in pending.groups.values()
                    for item in values
                }
            )
        )
        emit_utility_receipt(
            JevUtilityDecisionReceipt(
                turn_id=turn_id,
                decision_id=decision_id,
                repository_scope_id=pending.repository_scope_id,
                backend_id=self._adapter.backend_id,
                backend_model=self._adapter.backend_model,
                backend_version=self._adapter.backend_version,
                config_digest=self._adapter.config_digest,
                request_digest=None,
                input_candidate_ids=input_ids,
                retained_candidate_ids=input_ids,
                abstained_candidate_ids=(),
                decisions=(),
                latency_ms=0.0,
                fallback_used=True,
                fallback_reason="utility_internal_error",
            )
        )


__all__ = ["KnowledgeUtilityCoordinator", "UtilityRefinement"]
