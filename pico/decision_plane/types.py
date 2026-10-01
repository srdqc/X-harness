"""Immutable contracts for narrow, advisory Runtime decisions.

The decision plane can rank an already-authoritative candidate set.  It does
not own candidate discovery, activation, execution, policy, or terminal state.
P2.1 intentionally defines only ``SKILL_RANKING``.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pico.tracing.evidence import canonical_digest

DECISION_SCHEMA = "pico.decision.v1"
DECISION_SCHEMA_VERSION = 1
MAX_QUERY_CHARS = 4096
MAX_DESCRIPTION_CHARS = 1024


class DecisionType(str, Enum):
    SKILL_RANKING = "skill_ranking"


class DecisionOutcome(str, Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"
    INVALID_RESULT = "invalid_result"
    INVALID = "invalid"


@dataclass(frozen=True)
class DecisionCost:
    """Optional backend-reported cost; absent cost is explicit and valid."""

    available: bool = False
    amount: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.available:
            if self.amount is not None or self.unit is not None:
                raise ValueError("unavailable decision cost cannot carry amount or unit")
            return
        if self.amount is None or isinstance(self.amount, bool) or not math.isfinite(self.amount) or self.amount < 0:
            raise ValueError("available decision cost requires a finite nonnegative amount")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("available decision cost requires a unit")

    def canonical_payload(self) -> dict[str, Any]:
        return {"available": self.available, "amount": self.amount, "unit": self.unit}


@dataclass(frozen=True)
class DecisionCandidate:
    """A non-authoritative projection of one already-eligible candidate."""

    candidate_id: str
    name: str
    description: str
    source: str
    deterministic_rank: int
    deterministic_score: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.name or not self.source:
            raise ValueError("decision candidate identity fields must be non-empty")
        if self.deterministic_rank < 1:
            raise ValueError("deterministic_rank must be positive")
        if self.deterministic_score is not None and not math.isfinite(self.deterministic_score):
            raise ValueError("deterministic_score must be finite")
        if len(self.description) > MAX_DESCRIPTION_CHARS:
            raise ValueError("decision candidate description exceeds the bounded projection")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "deterministic_rank": self.deterministic_rank,
            "deterministic_score": self.deterministic_score,
        }


@dataclass(frozen=True)
class DecisionRequest:
    """Versioned request with a unique identity and deterministic semantic digests."""

    decision_type: DecisionType
    query: str
    candidates: tuple[DecisionCandidate, ...]
    decision_id: str = field(default_factory=lambda: f"decision:{uuid.uuid4().hex}")
    correlation_id: str | None = None
    schema: str = DECISION_SCHEMA
    schema_version: int = DECISION_SCHEMA_VERSION
    candidate_set_digest: str = field(init=False)
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != DECISION_SCHEMA or self.schema_version != DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported decision request schema")
        if self.decision_type is not DecisionType.SKILL_RANKING:
            raise ValueError("unsupported decision type")
        if not self.decision_id:
            raise ValueError("decision_id must be non-empty")
        if len(self.query) > MAX_QUERY_CHARS:
            raise ValueError("decision query exceeds the bounded projection")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("decision request contains duplicate candidate IDs")
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        if tuple(candidate.deterministic_rank for candidate in self.candidates) != expected_ranks:
            raise ValueError("decision candidates must preserve contiguous deterministic ranks")

        candidate_payload = [candidate.canonical_payload() for candidate in self.candidates]
        candidate_set_digest = canonical_digest(candidate_payload)
        request_digest = canonical_digest(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "decision_type": self.decision_type.value,
                "query": self.query,
                "candidate_set_digest": candidate_set_digest,
            }
        )
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "request_digest", request_digest)


@dataclass(frozen=True)
class DecisionRanking:
    candidate_id: str
    score: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("ranked candidate ID must be non-empty")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("decision score must be finite")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("decision confidence must be between zero and one")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DecisionResult:
    """Advisory ranking result; Runtime validation decides whether it is usable."""

    decision_id: str
    decision_type: DecisionType
    request_digest: str
    source: str
    outcome: DecisionOutcome
    ranking: tuple[DecisionRanking, ...] = ()
    reason: str | None = None
    cost: DecisionCost = field(default_factory=DecisionCost)
    schema: str = DECISION_SCHEMA
    schema_version: int = DECISION_SCHEMA_VERSION
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != DECISION_SCHEMA or self.schema_version != DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported decision result schema")
        if not self.decision_id or not self.request_digest or not self.source:
            raise ValueError("decision result identity fields must be non-empty")
        object.__setattr__(
            self,
            "result_digest",
            canonical_digest(
                {
                    "schema": self.schema,
                    "schema_version": self.schema_version,
                    "decision_type": self.decision_type.value,
                    "request_digest": self.request_digest,
                    "source": self.source,
                    "outcome": self.outcome.value,
                    "ranking": [item.canonical_payload() for item in self.ranking],
                    "reason": self.reason,
                    "cost": self.cost.canonical_payload(),
                }
            ),
        )


def validate_decision_result(request: DecisionRequest, result: DecisionResult) -> str | None:
    """Return a stable invalidity reason, or ``None`` for a complete permutation."""

    if result.decision_id != request.decision_id:
        return "decision_id_mismatch"
    if result.decision_type is not request.decision_type:
        return "decision_type_mismatch"
    if result.request_digest != request.request_digest:
        return "request_digest_mismatch"
    if result.outcome is not DecisionOutcome.SUCCESS:
        return f"adapter_{result.outcome.value}"
    ranked_ids = tuple(item.candidate_id for item in result.ranking)
    if len(ranked_ids) != len(set(ranked_ids)):
        return "duplicate_candidate_id"
    expected_ids = {candidate.candidate_id for candidate in request.candidates}
    actual_ids = set(ranked_ids)
    if actual_ids - expected_ids:
        return "unknown_candidate_id"
    if expected_ids - actual_ids:
        return "incomplete_candidate_set"
    return None


__all__ = [
    "DECISION_SCHEMA",
    "DECISION_SCHEMA_VERSION",
    "MAX_DESCRIPTION_CHARS",
    "MAX_QUERY_CHARS",
    "DecisionCandidate",
    "DecisionCost",
    "DecisionOutcome",
    "DecisionRanking",
    "DecisionRequest",
    "DecisionResult",
    "DecisionType",
    "validate_decision_result",
]
