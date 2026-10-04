"""System-One-inspired typed utility decisions over the existing Provider boundary."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pico.decision_plane.types import DecisionOutcome
from pico.decision_plane.utility import (
    JevUtilityAdvice,
    JevUtilityBackendResponse,
    JevUtilityRequest,
    UtilityDecision,
)
from pico.providers.base import LLMProvider, LLMResponse
from pico.tracing.evidence import canonical_digest

PROVIDER_UTILITY_POLICY_VERSION = 1
PROVIDER_UTILITY_TEMPLATE_VERSION = 1
PROVIDER_UTILITY_INSTRUCTION = """You are making a bounded utility judgment, not solving the task.

For each candidate below, decide whether this already-relevant reusable knowledge should be included for the current task.

Choose exactly one:
KEEP: likely to materially help complete the task.
ABSTAIN: relevant, but unlikely to materially help and likely to add unnecessary context or exploration.
UNCERTAIN: not enough evidence to safely remove it.

Do not propose code, tools, patches, task solutions, or reasoning. Return only the requested JSON object."""
PROVIDER_UTILITY_PROMPT_DIGEST = canonical_digest(
    {"version": PROVIDER_UTILITY_TEMPLATE_VERSION, "instruction": PROVIDER_UTILITY_INSTRUCTION}
)


class UtilityFinishReason(StrEnum):
    STOP = "STOP"
    LENGTH = "LENGTH"
    CONTENT_FILTER = "CONTENT_FILTER"
    TOOL_CALL = "TOOL_CALL"
    UNKNOWN = "UNKNOWN"


class UtilityMalformedCategory(StrEnum):
    NON_JSON = "NON_JSON"
    JSON_SYNTAX_ERROR = "JSON_SYNTAX_ERROR"
    TRUNCATED_OUTPUT = "TRUNCATED_OUTPUT"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_ALIAS = "MISSING_ALIAS"
    DUPLICATE_ALIAS = "DUPLICATE_ALIAS"
    UNKNOWN_ALIAS = "UNKNOWN_ALIAS"
    INVALID_CHOICE = "INVALID_CHOICE"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    EXTRA_FIELDS = "EXTRA_FIELDS"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class _ParseFailure:
    reason: str
    category: UtilityMalformedCategory
    stage: str
    top_level_shape: str | None = None


def normalize_finish_reason(value: object) -> UtilityFinishReason:
    if not isinstance(value, str):
        return UtilityFinishReason.UNKNOWN
    normalized = value.casefold().replace("-", "_")
    if normalized in {"stop", "end_turn", "complete", "completed"}:
        return UtilityFinishReason.STOP
    if normalized in {"length", "max_tokens", "max_output_tokens"}:
        return UtilityFinishReason.LENGTH
    if normalized in {"content_filter", "content_filtered", "safety"}:
        return UtilityFinishReason.CONTENT_FILTER
    if normalized in {"tool_call", "tool_calls", "function_call"}:
        return UtilityFinishReason.TOOL_CALL
    return UtilityFinishReason.UNKNOWN


@dataclass(frozen=True)
class ProviderUtilityConfig:
    provider_id: str = "agent"
    model_id: str | None = None
    max_candidates: int = 16
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.provider_id or len(self.provider_id) > 128:
            raise ValueError("provider_id must be bounded and non-empty")
        if self.model_id is not None and (not self.model_id or len(self.model_id) > 128):
            raise ValueError("model_id must be bounded when set")
        if isinstance(self.max_candidates, bool) or not 1 <= self.max_candidates <= 64:
            raise ValueError("max_candidates must be between 1 and 64")
        if isinstance(self.max_tokens, bool) or not 64 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 64 and 4096")


class ProviderUtilityBackend:
    """One logical, tool-free Provider call producing strictly validated JSON advice."""

    backend_version = "1"
    policy_version = PROVIDER_UTILITY_POLICY_VERSION

    def __init__(self, provider: LLMProvider, config: ProviderUtilityConfig | None = None) -> None:
        self._provider = provider
        self.config = config or ProviderUtilityConfig(model_id=provider.get_default_model())
        self.backend_id = f"provider:{self.config.provider_id}"
        self.model_id = self.config.model_id or provider.get_default_model()
        self.prompt_digest = PROVIDER_UTILITY_PROMPT_DIGEST

    async def decide_knowledge_utility(self, request: JevUtilityRequest) -> JevUtilityBackendResponse:
        if len(request.candidates) > self.config.max_candidates:
            return self._failure(request, DecisionOutcome.UNAVAILABLE, "max_candidates_exceeded")
        aliases = tuple(f"candidate_{index}" for index in range(len(request.candidates)))
        alias_to_id = dict(zip(aliases, (item.candidate_id for item in request.candidates), strict=True))
        messages = self._messages(request, aliases)
        attempts = 0

        def attempt_started(_model: str | None) -> None:
            nonlocal attempts
            attempts += 1

        started_ns = time.perf_counter_ns()
        try:
            response = await self._provider.chat_with_retry(
                messages=messages,
                tools=None,
                model=self.model_id,
                max_tokens=self.config.max_tokens,
                temperature=0.0,
                reasoning_effort=None,
                tool_choice=None,
                fallback_models=None,
                attempt_started=attempt_started,
                call_role="utility",
            )
        except Exception:  # noqa: BLE001 -- adapter falls back without leaking provider details
            return self._failure(
                request,
                DecisionOutcome.ERROR,
                "provider_error",
                attempts=attempts,
                latency_ms=self._latency(started_ns),
            )
        metadata = self._metadata(response, attempts, self._latency(started_ns))
        finish_reason = normalize_finish_reason(response.finish_reason)
        if response.finish_reason == "error":
            category = response.error_classification.category if response.error_classification else "provider_error"
            reason = {
                "rate_limit": "rate_limit",
                "auth": "authentication",
                "network": "unavailable",
                "server": "unavailable",
                "model_unavailable": "unavailable",
            }.get(category, "provider_error")
            return self._failure(request, DecisionOutcome.UNAVAILABLE, reason, metadata=metadata)
        if response.tool_calls:
            return self._failure(
                request, DecisionOutcome.INVALID_RESULT, "malformed_response", metadata=metadata,
                finish_reason=UtilityFinishReason.TOOL_CALL,
                malformed_category=UtilityMalformedCategory.SCHEMA_MISMATCH,
                parse_stage="provider_response",
            )
        if not isinstance(response.content, str):
            return self._failure(
                request, DecisionOutcome.INVALID_RESULT, "malformed_response", metadata=metadata,
                finish_reason=finish_reason, malformed_category=UtilityMalformedCategory.EMPTY_CONTENT,
                parse_stage="provider_response",
            )
        response_character_count = len(response.content)
        response_digest = canonical_digest(response.content)
        decisions = self._parse(response.content, alias_to_id, finish_reason=finish_reason)
        if isinstance(decisions, _ParseFailure):
            return self._failure(
                request, DecisionOutcome.INVALID_RESULT, decisions.reason, metadata=metadata,
                finish_reason=finish_reason, malformed_category=decisions.category,
                response_character_count=response_character_count, response_digest=response_digest,
                top_level_shape=decisions.top_level_shape, parse_stage=decisions.stage,
            )
        return JevUtilityBackendResponse(
            request.decision_id,
            request.request_digest,
            decisions,
            backend_id=self.backend_id,
            backend_model=response.model or self.model_id,
            backend_version=self.backend_version,
            logical_calls=1,
            provider_attempts=metadata[0],
            input_tokens=metadata[1],
            output_tokens=metadata[2],
            latency_ms=metadata[3],
            logical_call_id=metadata[4],
            finish_reason=finish_reason.value,
            response_character_count=response_character_count,
            response_digest=response_digest,
            top_level_shape="object",
            parse_stage="complete",
        )

    def _messages(self, request: JevUtilityRequest, aliases: tuple[str, ...]) -> list[dict[str, str]]:
        candidates = []
        for alias, item in zip(aliases, request.candidates, strict=True):
            candidates.append(
                {
                    "candidate": alias,
                    "type": item.candidate_type,
                    "title": item.title,
                    "summary": item.summary,
                    "preconditions": item.preconditions,
                    "relevance_rank": item.relevance_rank,
                    "relevance_score": item.relevance_score,
                    "applicability_digest": item.applicability_digest,
                }
            )
        state = {
            "schema": "pico.provider-utility-request.v1",
            "task": request.query,
            "repository_scope_id": request.repository_scope_id,
            "selector_version": request.selector_version,
            "choices": ["KEEP", "ABSTAIN", "UNCERTAIN"],
            "response_schema": {
                "decisions": [
                    {"candidate": "candidate_0", "choice": "KEEP", "confidence": 0.0}
                ]
            },
            "candidates": candidates,
        }
        return [
            {"role": "system", "content": PROVIDER_UTILITY_INSTRUCTION},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
        ]

    @staticmethod
    def _parse(
        content: str,
        alias_to_id: dict[str, str],
        *,
        finish_reason: UtilityFinishReason = UtilityFinishReason.UNKNOWN,
    ) -> tuple[JevUtilityAdvice, ...] | _ParseFailure:
        if not content.strip():
            return _ParseFailure("malformed_response", UtilityMalformedCategory.EMPTY_CONTENT, "json_decode")
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            category = (
                UtilityMalformedCategory.TRUNCATED_OUTPUT
                if finish_reason is UtilityFinishReason.LENGTH
                else UtilityMalformedCategory.JSON_SYNTAX_ERROR
            )
            return _ParseFailure("malformed_response", category, "json_decode")
        shape = type(payload).__name__.lower()
        if not isinstance(payload, dict):
            return _ParseFailure("malformed_response", UtilityMalformedCategory.SCHEMA_MISMATCH, "schema_validation", shape)
        if set(payload) != {"decisions"}:
            category = UtilityMalformedCategory.EXTRA_FIELDS if "decisions" in payload else UtilityMalformedCategory.SCHEMA_MISMATCH
            return _ParseFailure("malformed_response", category, "schema_validation", "object")
        if not isinstance(payload["decisions"], list):
            return _ParseFailure("malformed_response", UtilityMalformedCategory.SCHEMA_MISMATCH, "schema_validation", "object")
        if len(payload["decisions"]) > len(alias_to_id):
            return _ParseFailure("unexpected_candidate_alias", UtilityMalformedCategory.UNKNOWN_ALIAS, "alias_validation", "object")
        parsed: list[JevUtilityAdvice] = []
        seen: set[str] = set()
        for raw in payload["decisions"]:
            if not isinstance(raw, dict) or not {"candidate", "choice"} <= set(raw) or set(raw) - {"candidate", "choice", "confidence"}:
                return _ParseFailure("malformed_decision", UtilityMalformedCategory.EXTRA_FIELDS, "schema_validation", "object")
            alias = raw["candidate"]
            if not isinstance(alias, str) or alias not in alias_to_id:
                return _ParseFailure("unexpected_candidate_alias", UtilityMalformedCategory.UNKNOWN_ALIAS, "alias_validation", "object")
            if alias in seen:
                return _ParseFailure("duplicate_candidate_alias", UtilityMalformedCategory.DUPLICATE_ALIAS, "alias_validation", "object")
            seen.add(alias)
            choice = raw["choice"]
            try:
                decision = UtilityDecision(str(choice).lower()) if isinstance(choice, str) else None
            except ValueError:
                decision = None
            if decision is None or choice not in {"KEEP", "ABSTAIN", "UNCERTAIN"}:
                return _ParseFailure("invalid_choice", UtilityMalformedCategory.INVALID_CHOICE, "decision_validation", "object")
            confidence = raw.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return _ParseFailure("invalid_confidence", UtilityMalformedCategory.INVALID_CONFIDENCE, "decision_validation", "object")
            parsed.append(JevUtilityAdvice(alias_to_id[alias], decision, confidence=confidence))
        if seen != set(alias_to_id):
            return _ParseFailure("missing_candidate_alias", UtilityMalformedCategory.MISSING_ALIAS, "alias_validation", "object")
        return tuple(parsed)

    @staticmethod
    def _latency(started_ns: int) -> float:
        return round((time.perf_counter_ns() - started_ns) / 1_000_000, 6)

    @staticmethod
    def _tokens(usage: dict[str, Any], *names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    def _metadata(self, response: LLMResponse, attempts: int, latency_ms: float) -> tuple[int, int | None, int | None, float, str | None]:
        return (
            attempts,
            self._tokens(response.usage, "input_tokens", "prompt_tokens"),
            self._tokens(response.usage, "output_tokens", "completion_tokens"),
            latency_ms,
            response.logical_call_id,
        )

    def _failure(
        self,
        request: JevUtilityRequest,
        outcome: DecisionOutcome,
        reason: str,
        *,
        attempts: int = 0,
        latency_ms: float | None = None,
        metadata: tuple[int, int | None, int | None, float, str | None] | None = None,
        finish_reason: UtilityFinishReason = UtilityFinishReason.UNKNOWN,
        malformed_category: UtilityMalformedCategory | None = None,
        response_character_count: int | None = None,
        response_digest: str | None = None,
        top_level_shape: str | None = None,
        parse_stage: str | None = None,
    ) -> JevUtilityBackendResponse:
        values = metadata or (attempts, None, None, latency_ms, None)
        return JevUtilityBackendResponse(
            request.decision_id,
            request.request_digest,
            (),
            outcome=outcome,
            failure_reason=reason,
            backend_id=self.backend_id,
            backend_model=self.model_id,
            backend_version=self.backend_version,
            logical_calls=1 if attempts else 0,
            provider_attempts=values[0],
            input_tokens=values[1],
            output_tokens=values[2],
            latency_ms=values[3],
            logical_call_id=values[4],
            finish_reason=finish_reason.value,
            malformed_category=malformed_category.value if malformed_category else None,
            response_character_count=response_character_count,
            response_digest=response_digest,
            top_level_shape=top_level_shape,
            parse_stage=parse_stage,
        )


__all__ = [
    "PROVIDER_UTILITY_INSTRUCTION",
    "PROVIDER_UTILITY_POLICY_VERSION",
    "PROVIDER_UTILITY_PROMPT_DIGEST",
    "PROVIDER_UTILITY_TEMPLATE_VERSION",
    "ProviderUtilityBackend",
    "ProviderUtilityConfig",
    "UtilityFinishReason",
    "UtilityMalformedCategory",
    "normalize_finish_reason",
]
