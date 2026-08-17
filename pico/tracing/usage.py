"""Tracing projection of canonical CallEfficiency usage and cost."""

from __future__ import annotations

from typing import Any


def normalize(usage: dict[str, Any] | None, model: str | None) -> dict[str, Any]:
    from pico.call_efficiency.pricing import estimate_cost_usd
    from pico.call_efficiency.usage import normalize_usage

    raw = usage or {}
    normalized, findings = normalize_usage(raw, model or "")
    total = normalized.total_tokens or (
        normalized.input_tokens
        + normalized.cache_read_tokens
        + normalized.cache_write_tokens
        + normalized.output_tokens
    )
    cost = None
    if model and normalized.complete:
        try:
            cost = estimate_cost_usd(
                model,
                normalized.input_tokens,
                normalized.output_tokens,
                normalized.cache_read_tokens,
                normalized.cache_write_tokens,
                allow_litellm_import=False,
            )
        except Exception:  # noqa: BLE001
            cost = None
    return {
        "input_tokens": normalized.input_tokens,
        "output_tokens": normalized.output_tokens,
        "cache_read_tokens": normalized.cache_read_tokens,
        "cache_write_tokens": normalized.cache_write_tokens,
        "total_tokens": total,
        "cost_usd": cost,
        "usage_complete": normalized.complete,
        "findings": findings,
        "raw": dict(raw),
    }
