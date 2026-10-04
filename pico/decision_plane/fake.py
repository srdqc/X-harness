"""Deterministic Jev backend fake for tests and controlled experiments."""

from __future__ import annotations

import asyncio
from enum import Enum

from pico.decision_plane.jev import JevBackendResponse, JevCostMetadata, JevRankingAdvice
from pico.decision_plane.types import DecisionOutcome, DecisionRequest
from pico.decision_plane.utility import (
    JevUtilityAdvice,
    JevUtilityBackendResponse,
    JevUtilityRequest,
    UtilityDecision,
)


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
    INVALID_DECISION = "invalid_decision"
    INVALID_RANK = "invalid_rank"
    KEEP_ALL = "keep_all"
    ABSTAIN_ONE = "abstain_one"
    ABSTAIN_ALL = "abstain_all"
    REORDER = "reorder"
    UNCERTAIN_ONE = "uncertain_one"
    UNCERTAIN_ALL = "uncertain_all"
    MIXED_UTILITY = "mixed_utility"


class ScriptedJevBackend:
    """Return one selected response shape without inspecting expected answers."""

    backend_id = "scripted_jev"
    backend_version = "1"

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
        self.utility_calls = 0
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

    async def decide_knowledge_utility(
        self, request: JevUtilityRequest
    ) -> JevUtilityBackendResponse:
        self.calls += 1
        self.utility_calls += 1
        if self.mode is JevFakeMode.TIMEOUT:
            try:
                await asyncio.sleep(3_600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.mode is JevFakeMode.EXCEPTION:
            raise RuntimeError("scripted Jev utility failure")
        if self.mode is JevFakeMode.MALFORMED:
            return object()  # type: ignore[return-value]
        if self.mode is JevFakeMode.UNAVAILABLE:
            return JevUtilityBackendResponse(
                request.decision_id,
                request.request_digest,
                (),
                outcome=DecisionOutcome.UNAVAILABLE,
            )

        values = list(request.candidates)
        decisions = [
            JevUtilityAdvice(
                item.candidate_id,
                UtilityDecision.KEEP,
                utility_score=1.0,
                confidence=self.confidence,
                reason_code="scripted_keep",
                backend_rank=index,
            )
            for index, item in enumerate(values, start=1)
        ]
        if self.mode is JevFakeMode.ABSTAIN_ONE and decisions:
            decisions[-1] = JevUtilityAdvice(
                decisions[-1].candidate_id,
                UtilityDecision.ABSTAIN,
                utility_score=0.0,
                reason_code="scripted_abstain",
                backend_rank=len(decisions),
            )
        elif self.mode is JevFakeMode.ABSTAIN_ALL:
            decisions = [
                JevUtilityAdvice(
                    item.candidate_id,
                    UtilityDecision.ABSTAIN,
                    utility_score=0.0,
                    reason_code="scripted_abstain",
                    backend_rank=index,
                )
                for index, item in enumerate(values, start=1)
            ]
        elif self.mode in {JevFakeMode.UNCERTAIN_ONE, JevFakeMode.UNCERTAIN_ALL, JevFakeMode.MIXED_UTILITY}:
            for index, decision in enumerate(decisions):
                if self.mode is JevFakeMode.UNCERTAIN_ALL or (
                    self.mode is JevFakeMode.UNCERTAIN_ONE and index == 0
                ) or (self.mode is JevFakeMode.MIXED_UTILITY and index % 3 == 2):
                    decisions[index] = JevUtilityAdvice(
                        decision.candidate_id,
                        UtilityDecision.UNCERTAIN,
                        confidence=self.confidence,
                        reason_code="scripted_uncertain",
                        backend_rank=index + 1,
                    )
                elif self.mode is JevFakeMode.MIXED_UTILITY and index % 3 == 1:
                    decisions[index] = JevUtilityAdvice(
                        decision.candidate_id,
                        UtilityDecision.ABSTAIN,
                        confidence=self.confidence,
                        reason_code="scripted_abstain",
                        backend_rank=index + 1,
                    )
        elif self.mode is JevFakeMode.REORDER:
            decisions = [
                JevUtilityAdvice(
                    item.candidate_id,
                    UtilityDecision.KEEP,
                    utility_score=1.0,
                    reason_code="scripted_keep",
                    backend_rank=index,
                )
                for index, item in enumerate(reversed(values), start=1)
            ]
        elif self.mode is JevFakeMode.UNKNOWN_CANDIDATE and decisions:
            decisions[-1] = JevUtilityAdvice("unknown/candidate", UtilityDecision.KEEP)
        elif self.mode is JevFakeMode.DUPLICATE_CANDIDATE and decisions:
            decisions[-1] = decisions[0]
        elif self.mode is JevFakeMode.INCOMPLETE_RANKING:
            decisions = decisions[:-1]
        elif self.mode is JevFakeMode.INVALID_SCORE and decisions:
            decisions[0] = JevUtilityAdvice(
                decisions[0].candidate_id,
                UtilityDecision.KEEP,
                utility_score=float("nan"),
            )
        elif self.mode is JevFakeMode.INVALID_CONFIDENCE and decisions:
            decisions[0] = JevUtilityAdvice(
                decisions[0].candidate_id,
                UtilityDecision.KEEP,
                confidence=2.0,
            )
        elif self.mode is JevFakeMode.INVALID_DECISION and decisions:
            decisions[0] = JevUtilityAdvice(decisions[0].candidate_id, "promote")
        elif self.mode is JevFakeMode.INVALID_RANK and decisions:
            decisions[0] = JevUtilityAdvice(
                decisions[0].candidate_id,
                UtilityDecision.KEEP,
                backend_rank=len(decisions) + 1,
            )
        return JevUtilityBackendResponse(
            request.decision_id,
            request.request_digest,
            tuple(decisions),
        )


__all__ = ["JevFakeMode", "ScriptedJevBackend"]
