"""Provider-aware usage normalization."""

from __future__ import annotations

from typing import Any

from pico.call_efficiency.models import CallUsage
from pico.providers.registry import find_by_model


def normalize_usage(raw_usage: dict[str, Any] | None, model: str) -> tuple[CallUsage, tuple[str, ...]]:
    raw = raw_usage or {}
    findings: list[str] = []
    prompt, prompt_present = _token(raw, "prompt_tokens", findings)
    output, output_present = _token(raw, "completion_tokens", findings)
    cache_read, _ = _token(raw, "cache_read_input_tokens", findings)
    cache_write, _ = _token(raw, "cache_creation_input_tokens", findings)
    cache_miss, cache_miss_present = _token(raw, "cache_miss_input_tokens", findings)
    reasoning, _ = _token(raw, "reasoning_tokens", findings)
    total, total_present = _token(raw, "total_tokens", findings)

    complete = prompt_present and output_present and not findings
    fresh = prompt
    cached = cache_read + cache_write
    if cache_miss_present:
        fresh = cache_miss
        if prompt_present and prompt != fresh + cached:
            findings.append("provider_usage_total_mismatch")
            complete = False
    elif cached:
        semantics = _prompt_token_semantics(model)
        if semantics == "fresh":
            fresh = prompt
        elif semantics == "total":
            if prompt >= cached:
                fresh = prompt - cached
            else:
                findings.append("provider_usage_cache_exceeds_prompt")
                complete = False
        else:
            findings.append("input_token_semantics_ambiguous")
            complete = False

    return (
        CallUsage(
            input_tokens=fresh,
            output_tokens=output,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
            total_tokens=total if total_present else None,
            complete=complete and not findings,
        ),
        tuple(findings),
    )


def _prompt_token_semantics(model: str) -> str:
    key = model.lower()
    if key.startswith("openrouter/"):
        return "total"
    spec = find_by_model(key)
    if spec is not None and spec.name == "anthropic":
        return "fresh"
    if spec is not None and spec.name in {"deepseek", "openai", "openai_codex"}:
        return "total"
    return "unknown"


def _token(raw: dict[str, Any], key: str, findings: list[str]) -> tuple[int, bool]:
    if key not in raw or raw[key] is None:
        return 0, False
    value = raw[key]
    if isinstance(value, bool):
        findings.append(f"invalid_usage_field:{key}")
        return 0, True
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        findings.append(f"invalid_usage_field:{key}")
        return 0, True
    if parsed < 0:
        findings.append(f"negative_usage_field:{key}")
        return 0, True
    return parsed, True


__all__ = ["normalize_usage"]
