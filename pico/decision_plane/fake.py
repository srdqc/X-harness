"""Deterministic Jev backend fake for tests and controlled experiments."""

from __future__ import annotations

import asyncio
from enum import Enum

from pico.decision_plane.jev import JevBackendResponse, JevCostMetadata, JevRankingAdvice
from pico.decision_plane.types import DecisionOutcome, DecisionRequest


class JevFakeMode(str, Enum):
    VALID = "valid"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    MALFORMED = "malformed"
    WRONG_DECISION_TYPE = "wrong_decision_type"
    UNKNOWN_CANDIDATE = "unknown_candidate"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    INCOMPLETE_RANKING = "incomplete_ranking"
    INVALID_SCORE = "invalid_score"
    INVALID_CONFIDENCE = "invalid_confidence"


class ScriptedJevBackend:
    """Return one selected response shape without inspecting expected answers."""

    def __init__(
        self,
        mode: JevFakeMode = JevFakeMode.VALID,
        *,
        reverse: bool = True,
        confidence: float | None = None,
        cost: JevCostMetadata | None = None,
    ) -> None:
        self.mode = mode
        self.reverse = reverse
        self.confidence = confidence
        self.cost = cost or JevCostMetadata()
        self.calls = 0
        self.cancelled = False

    async def rank_skill_candidates(self, request: DecisionRequest) -> JevBackendResponse:
        self.calls += 1
        if self.mode is JevFakeMode.TIMEOUT:
            try:
                await asyncio.sleep(3_600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.mode is JevFakeMode.EXCEPTION:
            raise RuntimeError("scripted Jev failure")
        if self.mode is JevFakeMode.MALFORMED:
            return object()  # type: ignore[return-value]

        candidates = list(request.candidates)
        if self.reverse:
            candidates.reverse()
        ranking = [
            JevRankingAdvice(candidate.candidate_id, confidence=self.confidence)
            for candidate in candidates
        ]
        decision_type: object = request.decision_type
        outcome: object = DecisionOutcome.SUCCESS

        if self.mode is JevFakeMode.UNAVAILABLE:
            outcome = DecisionOutcome.UNAVAILABLE
            ranking = []
        elif self.mode is JevFakeMode.WRONG_DECISION_TYPE:
            decision_type = "tool_ranking"
        elif self.mode is JevFakeMode.UNKNOWN_CANDIDATE:
            ranking[-1] = JevRankingAdvice("unknown/skill")
        elif self.mode is JevFakeMode.DUPLICATE_CANDIDATE:
            ranking[-1] = ranking[0]
        elif self.mode is JevFakeMode.INCOMPLETE_RANKING:
            ranking = ranking[:-1]
        elif self.mode is JevFakeMode.INVALID_SCORE:
            ranking[0] = JevRankingAdvice(ranking[0].candidate_id, score=float("nan"))
        elif self.mode is JevFakeMode.INVALID_CONFIDENCE:
            ranking[0] = JevRankingAdvice(ranking[0].candidate_id, confidence=2.0)

        return JevBackendResponse(
            decision_id=request.decision_id,
            decision_type=decision_type,
            request_digest=request.request_digest,
            ranking=tuple(ranking),
            outcome=outcome,
            cost=self.cost,
        )


__all__ = ["JevFakeMode", "ScriptedJevBackend"]
