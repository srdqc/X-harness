"""Optional host adapter for an injected experimental Jev backend.

This module deliberately defines no transport.  The backend returns ranking
advice only; the Runtime validates it before the existing resolver may use it.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pico.decision_plane.types import (
    DECISION_SCHEMA,
    DECISION_SCHEMA_VERSION,
    DecisionCost,
    DecisionOutcome,
    DecisionRanking,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)


@dataclass(frozen=True)
class JevRankingAdvice:
    """Untrusted backend ranking item, validated by :class:`JevDecisionAdapter`."""

    candidate_id: object
    score: object | None = None
    confidence: object | None = None


@dataclass(frozen=True)
class JevCostMetadata:
    """Untrusted optional cost metadata supplied by a backend."""

    available: object = False
    amount: object | None = None
    unit: object | None = None


@dataclass(frozen=True)
class JevBackendResponse:
    """Transport-neutral response projection returned by a Jev backend."""

    decision_id: object
    decision_type: object
    request_digest: object
    ranking: object
    outcome: object = DecisionOutcome.SUCCESS
    schema: object = DECISION_SCHEMA
    schema_version: object = DECISION_SCHEMA_VERSION
    cost: object = JevCostMetadata()


@runtime_checkable
class JevBackend(Protocol):
    """Narrow injectable ranking backend with no Runtime authority."""

    async def rank_skill_candidates(self, request: DecisionRequest) -> JevBackendResponse: ...


class JevDecisionAdapter:
    """Validate bounded Jev advice and isolate all backend failures."""

    kind = "jev"
    source = kind

    def __init__(self, backend: JevBackend | None, *, timeout_seconds: float = 0.25) -> None:
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self._backend = backend
        self._timeout_seconds = timeout_seconds

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        if self._backend is None:
            return self._failure(request, DecisionOutcome.UNAVAILABLE, "backend_unavailable")

        task = asyncio.create_task(self._backend.rank_skill_candidates(request))
        try:
            done, _ = await asyncio.wait(
                (task,),
                timeout=self._timeout_seconds,
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            task.add_done_callback(self._discard_late_result)
            # Give well-behaved backends one scheduling point to observe
            # cancellation without waiting for a backend that suppresses it.
            await asyncio.sleep(0)
            return self._failure(request, DecisionOutcome.TIMEOUT, "backend_timeout")

        try:
            response = task.result()
        except Exception:  # noqa: BLE001 -- the experimental backend cannot fail a Turn
            return self._failure(request, DecisionOutcome.ERROR, "backend_exception")

        return self._validate_response(request, response)

    @staticmethod
    def _discard_late_result(task: asyncio.Task[object]) -> None:
        """Consume a cancelled/late result without allowing it to affect routing."""

        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    def _validate_response(self, request: DecisionRequest, response: object) -> DecisionResult:
        if not isinstance(response, JevBackendResponse):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "malformed_response")
        if response.schema != DECISION_SCHEMA or response.schema_version != DECISION_SCHEMA_VERSION:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "unsupported_schema")
        if response.decision_id != request.decision_id:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "decision_id_mismatch")
        if response.request_digest != request.request_digest:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "request_digest_mismatch")
        if response.decision_type is not request.decision_type:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "decision_type_mismatch")
        if response.outcome not in (DecisionOutcome.SUCCESS, DecisionOutcome.UNAVAILABLE):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "unsupported_outcome")
        if response.outcome is DecisionOutcome.UNAVAILABLE:
            return self._failure(request, DecisionOutcome.UNAVAILABLE, "backend_unavailable")

        ranking = self._parse_ranking(request, response.ranking)
        if isinstance(ranking, str):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, ranking)
        cost = self._parse_cost(response.cost)
        if isinstance(cost, str):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, cost)

        result = DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source=self.source,
            outcome=DecisionOutcome.SUCCESS,
            ranking=ranking,
            cost=cost,
        )
        invalid_reason = validate_decision_result(request, result)
        if invalid_reason is not None:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, invalid_reason)
        return result

    @staticmethod
    def _parse_ranking(
        request: DecisionRequest,
        raw_ranking: object,
    ) -> tuple[DecisionRanking, ...] | str:
        if not isinstance(raw_ranking, (tuple, list)):
            return "malformed_ranking"
        ranking: list[DecisionRanking] = []
        for raw in raw_ranking:
            if not isinstance(raw, JevRankingAdvice):
                return "malformed_ranking_item"
            if not isinstance(raw.candidate_id, str) or not raw.candidate_id:
                return "invalid_candidate_id"
            score = JevDecisionAdapter._optional_finite_number(raw.score)
            if score is _INVALID:
                return "invalid_score"
            confidence = JevDecisionAdapter._optional_finite_number(raw.confidence)
            if confidence is _INVALID or (confidence is not None and not 0.0 <= confidence <= 1.0):
                return "invalid_confidence"
            ranking.append(
                DecisionRanking(
                    candidate_id=raw.candidate_id,
                    score=score,
                    confidence=confidence,
                )
            )

        candidate_ids = tuple(item.candidate_id for item in ranking)
        if len(candidate_ids) != len(set(candidate_ids)):
            return "duplicate_candidate_id"
        expected_ids = {candidate.candidate_id for candidate in request.candidates}
        actual_ids = set(candidate_ids)
        if actual_ids - expected_ids:
            return "unknown_candidate_id"
        if expected_ids - actual_ids:
            return "incomplete_candidate_set"
        return tuple(ranking)

    @staticmethod
    def _parse_cost(raw: object) -> DecisionCost | str:
        if not isinstance(raw, JevCostMetadata) or not isinstance(raw.available, bool):
            return "invalid_cost"
        if not raw.available:
            if raw.amount is not None or raw.unit is not None:
                return "invalid_cost"
            return DecisionCost()
        if not isinstance(raw.unit, str):
            return "invalid_cost"
        try:
            return DecisionCost(
                available=True,
                amount=JevDecisionAdapter._required_number(raw.amount),
                unit=raw.unit,
            )
        except (TypeError, ValueError):
            return "invalid_cost"

    @staticmethod
    def _optional_finite_number(value: object | None) -> float | None | object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return _INVALID
        return float(value)

    @staticmethod
    def _required_number(value: object | None) -> float:
        parsed = JevDecisionAdapter._optional_finite_number(value)
        if parsed is None or parsed is _INVALID:
            raise ValueError("cost amount must be finite")
        return parsed

    def _failure(self, request: DecisionRequest, outcome: DecisionOutcome, reason: str) -> DecisionResult:
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source=self.source,
            outcome=outcome,
            reason=reason,
        )


_INVALID = object()


__all__ = [
    "JevBackend",
    "JevBackendResponse",
    "JevCostMetadata",
    "JevDecisionAdapter",
    "JevRankingAdvice",
]
