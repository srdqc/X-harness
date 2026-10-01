"""Decision adapter protocol and the authoritative deterministic baseline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pico.decision_plane.types import (
    DecisionOutcome,
    DecisionRanking,
    DecisionRequest,
    DecisionResult,
)


@runtime_checkable
class DecisionAdapter(Protocol):
    """An advisory component that cannot mutate or activate candidates."""

    kind: str

    async def decide(self, request: DecisionRequest) -> DecisionResult: ...


class DeterministicSkillRankingAdapter:
    """Preserve the ranking already produced by the authoritative Skill router."""

    kind = "deterministic"
    source = kind

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        return DecisionResult(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            request_digest=request.request_digest,
            source=self.source,
            outcome=DecisionOutcome.SUCCESS,
            ranking=tuple(
                DecisionRanking(
                    candidate_id=candidate.candidate_id,
                    score=candidate.deterministic_score,
                )
                for candidate in request.candidates
            ),
        )


__all__ = ["DecisionAdapter", "DeterministicSkillRankingAdapter"]
