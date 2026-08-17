"""TokenWise core abstractions colocated with implementations.

Strategies live next to the ABC they implement.

Strategies are additive — multiple can be installed. The agent calls each
hook in registration order. A strategy that is not interested in a given
hook inherits the default no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageSnapshot:
    """Token usage and cost for a single LLM call.

    Convention: ``input_tokens`` is *fresh* (non-cached) prompt tokens.
    ``pico.call_efficiency`` normalizes Provider-specific total/fresh
    conventions before projecting a record into this historical schema.

    ``trace_id`` / ``turn_span_id`` correlate a persisted usage row back to the
    Turn that spent it. Both
    stay ``None`` when tracing is disabled, and rows written before they existed
    read back as unjoinable rather than mis-joined.

    ``estimated_cost_usd`` is ``None`` when pricing is unavailable. Zero is
    reserved for calls whose known rates produce a real zero cost.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None
    session_key: str | None = None
    trace_id: str | None = None
    turn_span_id: str | None = None


class TokenStrategy(ABC):
    """Cross-cutting hooks for token and cost optimization.

    Strategies are additive — multiple can be installed. The agent calls each
    hook in registration order. A strategy that is not interested in a given
    hook inherits the default no-op.

    This is a single unified interface rather than four tiny ABCs to keep the
    install point simple. Concrete strategies will typically implement just
    one or two hooks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g. 'cache_optimizer', 'smart_router')."""

    async def before_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str]:
        """Pre-process the outgoing request. Return (messages, tools, model).

        Used by CacheOptimizer (marks cache_control), SmartRouter (chooses
        model), ToolResultPruner (rewrites old tool output blocks).
        Default: pass through.
        """
        return messages, tools, model

    async def after_llm_call(
        self,
        response: dict[str, Any],
        usage: UsageSnapshot,
    ) -> None:
        """Post-call hook. Used by UsageTracker, BudgetAlerter. Default: no-op."""


__all__ = ["TokenStrategy", "UsageSnapshot"]
