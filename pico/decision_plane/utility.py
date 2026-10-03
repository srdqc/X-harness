"""Typed, transport-neutral Jev utility decisions for already-relevant knowledge."""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from pico.decision_plane.types import DecisionOutcome
from pico.tracing import evidence as turn_evidence
from pico.tracing.evidence import canonical_digest

JEV_UTILITY_REQUEST_SCHEMA = "pico.jev-utility-request.v1"
JEV_UTILITY_RESPONSE_SCHEMA = "pico.jev-utility-response.v1"
JEV_UTILITY_RECEIPT_SCHEMA = "pico.jev-utility-decision.v1"
JEV_UTILITY_SCHEMA_VERSION = 1
MAX_UTILITY_QUERY_CHARS = 4096
MAX_UTILITY_TITLE_CHARS = 256
MAX_UTILITY_SUMMARY_CHARS = 1024
MAX_UTILITY_REASON_CHARS = 64
_REASON_CODE = re.compile(r"^[a-z0-9_.-]+$")


class UtilityDecision(StrEnum):
    KEEP = "keep"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class JevUtilityCandidate:
    candidate_id: str
    candidate_type: str
    title: str
    summary: str
    relevance_rank: int
    relevance_score: float
    applicability_digest: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_type:
            raise ValueError("utility candidate identity must be non-empty")
        if len(self.title) > MAX_UTILITY_TITLE_CHARS or len(self.summary) > MAX_UTILITY_SUMMARY_CHARS:
            raise ValueError("utility candidate projection exceeds its bound")
        if self.relevance_rank < 1 or not math.isfinite(self.relevance_score):
            raise ValueError("utility relevance evidence is invalid")
        if len(self.applicability_digest) != 64:
            raise ValueError("utility applicability digest is invalid")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "title": self.title,
            "summary": self.summary,
            "relevance_rank": self.relevance_rank,
            "relevance_score": self.relevance_score,
            "applicability_digest": self.applicability_digest,
        }


@dataclass(frozen=True)
class JevUtilityRequest:
    decision_id: str
    turn_id: str | None
    repository_scope_id: str
    query: str
    query_digest: str
    candidates: tuple[JevUtilityCandidate, ...]
    selector_version: int = 1
    schema: str = JEV_UTILITY_REQUEST_SCHEMA
    schema_version: int = JEV_UTILITY_SCHEMA_VERSION
    candidate_set_digest: str = field(init=False)
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != JEV_UTILITY_REQUEST_SCHEMA or self.schema_version != 1:
            raise ValueError("unsupported Jev utility request schema")
        if not self.decision_id or len(self.repository_scope_id) != 64 or len(self.query_digest) != 64:
            raise ValueError("invalid Jev utility request identity")
        if len(self.query) > MAX_UTILITY_QUERY_CHARS or self.selector_version != 1:
            raise ValueError("invalid Jev utility request bounds/version")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("utility request requires unique candidates")
        candidate_set_digest = canonical_digest([item.canonical_payload() for item in self.candidates])
        request_digest = canonical_digest(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "repository_scope_id": self.repository_scope_id,
                "query_digest": self.query_digest,
                "candidate_set_digest": candidate_set_digest,
                "selector_version": self.selector_version,
            }
        )
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "request_digest", request_digest)


@dataclass(frozen=True)
class JevUtilityAdvice:
    candidate_id: object
    decision: object
    utility_score: object | None = None
    confidence: object | None = None
    reason_code: object | None = None
    backend_rank: object | None = None


@dataclass(frozen=True)
class JevUtilityBackendResponse:
    decision_id: object
    request_digest: object
    decisions: object
    outcome: object = DecisionOutcome.SUCCESS
    schema: object = JEV_UTILITY_RESPONSE_SCHEMA
    schema_version: object = JEV_UTILITY_SCHEMA_VERSION


@dataclass(frozen=True)
class JevUtilityCandidateDecision:
    candidate_id: str
    decision: UtilityDecision
    utility_score: float | None = None
    confidence: float | None = None
    reason_code: str | None = None
    backend_rank: int | None = None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision.value,
            "utility_score": self.utility_score,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "backend_rank": self.backend_rank,
        }


@dataclass(frozen=True)
class JevUtilityResult:
    decision_id: str
    request_digest: str
    source: str
    outcome: DecisionOutcome
    decisions: tuple[JevUtilityCandidateDecision, ...] = ()
    reason: str | None = None
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_digest",
            canonical_digest(
                {
                    "decision_id": self.decision_id,
                    "request_digest": self.request_digest,
                    "source": self.source,
                    "outcome": self.outcome.value,
                    "decisions": [item.canonical_payload() for item in self.decisions],
                    "reason": self.reason,
                }
            ),
        )


class JevUtilityDecisionAdapter:
    """Validate one bounded utility response and contain every backend failure."""

    kind = "jev_utility"

    def __init__(self, backend: object | None, *, timeout_seconds: float = 0.25) -> None:
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self._backend = backend
        self.timeout_seconds = float(timeout_seconds)
        self.backend_id = str(getattr(backend, "backend_id", type(backend).__name__ if backend else "unavailable"))
        self.backend_model = getattr(backend, "model_id", None)
        self.backend_version = getattr(backend, "backend_version", None)
        self.config_digest = canonical_digest(
            {
                "mode": "task_relevance_v1_jev_utility",
                "backend_id": self.backend_id,
                "backend_model": self.backend_model,
                "backend_version": self.backend_version,
                "timeout_seconds": self.timeout_seconds,
            }
        )

    async def decide(self, request: JevUtilityRequest) -> JevUtilityResult:
        operation = getattr(self._backend, "decide_knowledge_utility", None)
        if not callable(operation):
            return self._failure(request, DecisionOutcome.UNAVAILABLE, "backend_unavailable")
        task = asyncio.create_task(operation(request))
        try:
            response = await asyncio.wait_for(task, timeout=self.timeout_seconds)
        except TimeoutError:
            return self._failure(request, DecisionOutcome.TIMEOUT, "backend_timeout")
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception:  # noqa: BLE001 -- optional external advice cannot fail a Turn
            return self._failure(request, DecisionOutcome.ERROR, "backend_exception")
        return self._validate(request, response)

    def _validate(self, request: JevUtilityRequest, response: object) -> JevUtilityResult:
        if not isinstance(response, JevUtilityBackendResponse):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "malformed_response")
        if response.schema != JEV_UTILITY_RESPONSE_SCHEMA or response.schema_version != 1:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "unsupported_schema")
        if response.decision_id != request.decision_id:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "decision_id_mismatch")
        if response.request_digest != request.request_digest:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "request_digest_mismatch")
        if response.outcome is DecisionOutcome.UNAVAILABLE:
            return self._failure(request, DecisionOutcome.UNAVAILABLE, "backend_unavailable")
        if response.outcome is not DecisionOutcome.SUCCESS or not isinstance(response.decisions, (list, tuple)):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "malformed_response")
        parsed: list[JevUtilityCandidateDecision] = []
        for raw in response.decisions:
            item = self._parse_advice(raw, len(request.candidates))
            if isinstance(item, str):
                return self._failure(request, DecisionOutcome.INVALID_RESULT, item)
            parsed.append(item)
        candidate_ids = tuple(item.candidate_id for item in parsed)
        if len(candidate_ids) != len(set(candidate_ids)):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "duplicate_candidate_id")
        expected = {item.candidate_id for item in request.candidates}
        actual = set(candidate_ids)
        if actual - expected:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "unknown_candidate_id")
        if expected - actual:
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "incomplete_candidate_set")
        ranks = tuple(item.backend_rank for item in parsed if item.backend_rank is not None)
        if len(ranks) != len(set(ranks)):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "duplicate_backend_rank")
        return JevUtilityResult(
            request.decision_id,
            request.request_digest,
            self.kind,
            DecisionOutcome.SUCCESS,
            tuple(parsed),
        )

    @staticmethod
    def _parse_advice(raw: object, candidate_count: int) -> JevUtilityCandidateDecision | str:
        if not isinstance(raw, JevUtilityAdvice):
            return "malformed_decision"
        if not isinstance(raw.candidate_id, str) or not raw.candidate_id:
            return "invalid_candidate_id"
        if not isinstance(raw.decision, UtilityDecision):
            return "invalid_decision"
        score = _finite_optional(raw.utility_score)
        if score is _INVALID:
            return "invalid_utility_score"
        confidence = _finite_optional(raw.confidence)
        if confidence is _INVALID or (confidence is not None and not 0 <= confidence <= 1):
            return "invalid_confidence"
        reason = raw.reason_code
        if reason is not None and (
            not isinstance(reason, str)
            or len(reason) > MAX_UTILITY_REASON_CHARS
            or _REASON_CODE.fullmatch(reason) is None
        ):
            return "invalid_reason_code"
        rank = raw.backend_rank
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= candidate_count):
            return "invalid_backend_rank"
        return JevUtilityCandidateDecision(raw.candidate_id, raw.decision, score, confidence, reason, rank)

    def _failure(self, request: JevUtilityRequest, outcome: DecisionOutcome, reason: str) -> JevUtilityResult:
        return JevUtilityResult(request.decision_id, request.request_digest, self.kind, outcome, reason=reason)


@dataclass(frozen=True)
class JevUtilityDecisionReceipt:
    turn_id: str | None
    decision_id: str
    repository_scope_id: str
    backend_id: str
    backend_model: str | None
    backend_version: str | None
    config_digest: str
    request_digest: str | None
    input_candidate_ids: tuple[str, ...]
    retained_candidate_ids: tuple[str, ...]
    abstained_candidate_ids: tuple[str, ...]
    decisions: tuple[JevUtilityCandidateDecision, ...]
    latency_ms: float
    fallback_used: bool
    fallback_reason: str | None
    selector_version: int = 1
    receipt_version: int = 1
    schema: str = JEV_UTILITY_RECEIPT_SCHEMA
    schema_version: int = JEV_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != JEV_UTILITY_RECEIPT_SCHEMA or self.schema_version != 1:
            raise ValueError("unsupported Jev utility receipt schema")
        if not self.decision_id or len(self.repository_scope_id) != 64 or len(self.config_digest) != 64:
            raise ValueError("invalid Jev utility receipt identity")
        if self.request_digest is not None and len(self.request_digest) != 64:
            raise ValueError("invalid Jev utility request digest")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0 or self.selector_version != 1:
            raise ValueError("invalid Jev utility receipt measurement/version")
        if any(len(value) > 128 for value in (self.backend_id, self.backend_model or "", self.backend_version or "")):
            raise ValueError("Jev utility backend identity exceeds its bound")
        input_ids = set(self.input_candidate_ids)
        retained = set(self.retained_candidate_ids)
        abstained = set(self.abstained_candidate_ids)
        if len(input_ids) != len(self.input_candidate_ids) or retained & abstained:
            raise ValueError("invalid Jev utility receipt candidate partition")
        if retained | abstained != input_ids:
            raise ValueError("incomplete Jev utility receipt candidate partition")
        if any(item.candidate_id not in input_ids for item in self.decisions):
            raise ValueError("utility receipt decision is outside the input set")
        if self.fallback_reason is not None and len(self.fallback_reason) > MAX_UTILITY_REASON_CHARS:
            raise ValueError("utility fallback reason exceeds its bound")

    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "receipt_schema": self.schema,
                "receipt_schema_version": self.schema_version,
                "turn_id": self.turn_id,
                "decision_id": self.decision_id,
                "decision_mode": "task_relevance_v1_jev_utility",
                "backend_id": self.backend_id,
                "backend_model": self.backend_model,
                "backend_version": self.backend_version,
                "config_digest": self.config_digest,
                "request_digest": self.request_digest,
                "input_candidate_ids": self.input_candidate_ids,
                "retained_candidate_ids": self.retained_candidate_ids,
                "abstained_candidate_ids": self.abstained_candidate_ids,
                "decisions": tuple(item.canonical_payload() for item in self.decisions),
                "latency_ms": self.latency_ms,
                "fallback_used": self.fallback_used,
                "fallback_reason": self.fallback_reason,
                "selector_version": self.selector_version,
                "receipt_version": self.receipt_version,
            }
        )


def emit_utility_receipt(receipt: JevUtilityDecisionReceipt) -> object | None:
    recorder = turn_evidence.current()
    if recorder is None or recorder.turn_id != receipt.turn_id:
        return None
    return recorder.emit(
        turn_evidence.DECISION_RECEIPT,
        correlations={"decision_id": receipt.decision_id},
        metadata=receipt.metadata(),
    )


def _finite_optional(value: object | None) -> float | None | object:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return _INVALID
    return float(value)


def measured_latency_ms(started_ns: int) -> float:
    return round((time.perf_counter_ns() - started_ns) / 1_000_000, 6)


_INVALID = object()


__all__ = [
    "JEV_UTILITY_RECEIPT_SCHEMA",
    "JEV_UTILITY_REQUEST_SCHEMA",
    "JEV_UTILITY_RESPONSE_SCHEMA",
    "JevUtilityAdvice",
    "JevUtilityBackendResponse",
    "JevUtilityCandidate",
    "JevUtilityCandidateDecision",
    "JevUtilityDecisionAdapter",
    "JevUtilityDecisionReceipt",
    "JevUtilityRequest",
    "JevUtilityResult",
    "UtilityDecision",
    "emit_utility_receipt",
    "measured_latency_ms",
]
