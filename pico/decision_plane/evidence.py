"""Privacy-bounded durable receipt for one advisory decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pico.tracing import evidence as turn_evidence

DECISION_RECEIPT_SCHEMA = "pico.decision-receipt.v1"
DECISION_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DecisionReceipt:
    turn_id: str | None
    decision_id: str
    decision_type: str
    adapter_kind: str
    candidate_count: int
    candidate_set_digest: str
    request_digest: str
    baseline_result_digest: str
    adapter_result_digest: str | None
    final_result_digest: str
    adapter_outcome: str
    fallback_used: bool
    fallback_reason: str | None
    latency_ms: float
    cost_available: bool
    cost_amount: float | None
    cost_unit: str | None
    confidences: tuple[tuple[str, float], ...]
    final_ranking_source: str
    schema: str = DECISION_RECEIPT_SCHEMA
    schema_version: int = DECISION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != DECISION_RECEIPT_SCHEMA or self.schema_version != DECISION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported decision receipt schema")
        if self.candidate_count < 0 or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("invalid decision receipt measurement")
        if self.cost_available != (self.cost_amount is not None and self.cost_unit is not None):
            raise ValueError("decision cost availability does not match metadata")
        if self.cost_amount is not None and (
            not math.isfinite(self.cost_amount) or self.cost_amount < 0
        ):
            raise ValueError("invalid decision cost amount")
        for candidate_id, confidence in self.confidences:
            if not candidate_id or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("invalid decision confidence metadata")

    def metadata(self) -> Mapping[str, Any]:
        """Return immutable evidence metadata without query text or Skill bodies."""

        return MappingProxyType(
            {
                "receipt_schema": self.schema,
                "receipt_schema_version": self.schema_version,
                "turn_id": self.turn_id,
                "decision_id": self.decision_id,
                "decision_type": self.decision_type,
                "adapter_kind": self.adapter_kind,
                "candidate_count": self.candidate_count,
                "candidate_set_digest": self.candidate_set_digest,
                "request_digest": self.request_digest,
                "baseline_result_digest": self.baseline_result_digest,
                "adapter_result_digest": self.adapter_result_digest,
                "final_result_digest": self.final_result_digest,
                "adapter_outcome": self.adapter_outcome,
                "fallback_used": self.fallback_used,
                "fallback_reason": self.fallback_reason,
                "latency_ms": self.latency_ms,
                "cost_available": self.cost_available,
                "cost_amount": self.cost_amount,
                "cost_unit": self.cost_unit,
                "confidences": self.confidences,
                "final_ranking_source": self.final_ranking_source,
            }
        )


def emit_decision_receipt(receipt: DecisionReceipt) -> turn_evidence.TurnEvidenceEvent | None:
    """Append to the current Turn-local evidence stream when one exists."""

    recorder = turn_evidence.current()
    if recorder is None or receipt.turn_id != recorder.turn_id:
        return None
    return recorder.emit(
        turn_evidence.DECISION_RECEIPT,
        correlations={"decision_id": receipt.decision_id},
        metadata=receipt.metadata(),
    )


__all__ = [
    "DECISION_RECEIPT_SCHEMA",
    "DECISION_RECEIPT_SCHEMA_VERSION",
    "DecisionReceipt",
    "emit_decision_receipt",
]
