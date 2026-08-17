"""Stable request and evidence models for Provider calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CALL_RECORD_SCHEMA = "pico.call-efficiency.call.v1"


@dataclass(frozen=True)
class PreparedCall:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str
    cache_policy: str


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int | None = None
    complete: bool = False


@dataclass(frozen=True)
class CallRecord:
    requested_model: str
    attempted_model: str
    actual_model: str
    accounting_model: str
    usage: CallUsage
    estimated_cost_usd: float | None
    outcome: str
    finish_reason: str
    error_category: str | None
    session_key: str | None
    trace_id: str | None
    turn_span_id: str | None
    mode: str
    cache_policy: str
    observed_at: str
    findings: tuple[str, ...] = field(default_factory=tuple)
    schema: str = CALL_RECORD_SCHEMA

    @property
    def cost_complete(self) -> bool:
        return self.estimated_cost_usd is not None

    def to_legacy_snapshot(self):
        from pico.token_wise.base import UsageSnapshot

        return UsageSnapshot(
            model=self.accounting_model,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens,
            reasoning_tokens=self.usage.reasoning_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
            session_key=self.session_key,
            trace_id=self.trace_id,
            turn_span_id=self.turn_span_id,
        )


__all__ = [
    "CALL_RECORD_SCHEMA",
    "CallRecord",
    "CallUsage",
    "PreparedCall",
]
