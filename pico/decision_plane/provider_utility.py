"""System-One-inspired typed utility decisions over the existing Provider boundary."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
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
        if response.tool_calls or not isinstance(response.content, str):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, "malformed_response", metadata=metadata)
        decisions = self._parse(response.content, alias_to_id)
        if isinstance(decisions, str):
            return self._failure(request, DecisionOutcome.INVALID_RESULT, decisions, metadata=metadata)
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
    def _parse(content: str, alias_to_id: dict[str, str]) -> tuple[JevUtilityAdvice, ...] | str:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return "malformed_response"
        if not isinstance(payload, dict) or set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
            return "malformed_response"
        if len(payload["decisions"]) > len(alias_to_id):
            return "unexpected_candidate_alias"
        parsed: list[JevUtilityAdvice] = []
        seen: set[str] = set()
        for raw in payload["decisions"]:
            if not isinstance(raw, dict) or not {"candidate", "choice"} <= set(raw) or set(raw) - {"candidate", "choice", "confidence"}:
                return "malformed_decision"
            alias = raw["candidate"]
            if not isinstance(alias, str) or alias not in alias_to_id:
                return "unexpected_candidate_alias"
            if alias in seen:
                return "duplicate_candidate_alias"
            seen.add(alias)
            choice = raw["choice"]
            try:
                decision = UtilityDecision(str(choice).lower()) if isinstance(choice, str) else None
            except ValueError:
                decision = None
            if decision is None or choice not in {"KEEP", "ABSTAIN", "UNCERTAIN"}:
                return "invalid_choice"
            confidence = raw.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return "invalid_confidence"
            parsed.append(JevUtilityAdvice(alias_to_id[alias], decision, confidence=confidence))
        if seen != set(alias_to_id):
            return "missing_candidate_alias"
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
        )


__all__ = [
    "PROVIDER_UTILITY_INSTRUCTION",
    "PROVIDER_UTILITY_POLICY_VERSION",
    "PROVIDER_UTILITY_PROMPT_DIGEST",
    "PROVIDER_UTILITY_TEMPLATE_VERSION",
    "ProviderUtilityBackend",
    "ProviderUtilityConfig",
]
