"""Permanent, reproducible JEV.3 Provider-utility contract smoke entry point.

Dry-run is the default and performs no Provider invocation. Live execution is
explicitly gated and validates only the transport/typed-decision contract; its
result is not evidence for prompt, confidence, candidate, or P3R tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from pico.cli._helpers import make_provider
from pico.config.pico import ContextConfig, load_pico_config
from pico.config.schema import Config
from pico.decision_plane.provider_utility import ProviderUtilityBackend, ProviderUtilityConfig
from pico.decision_plane.types import DecisionOutcome
from pico.decision_plane.utility import (
    JevUtilityCandidate,
    JevUtilityDecisionAdapter,
    JevUtilityDecisionReceipt,
    JevUtilityRequest,
    UtilityDecision,
    emit_utility_receipt,
    measured_latency_ms,
)
from pico.providers.registry import find_by_model, find_by_name
from pico.tracing import evidence
from pico.tracing.evidence import canonical_digest

SMOKE_SCHEMA = "pico.jev3-provider-utility-smoke.v1"
SMOKE_TURN_ID = "jev3-provider-utility-smoke"
SMOKE_PURPOSE = "contract_validation_only"
SMOKE_CANDIDATE_COUNT = 1


@dataclass(frozen=True)
class SmokePlan:
    provider_id: str
    model_id: str
    utility_timeout_seconds: float
    candidate_count: int = SMOKE_CANDIDATE_COUNT
    utility_backend: str = "provider"
    tools_enabled: bool = False
    typesafe_enabled: bool = False
    normal_agent_turn: bool = False
    workspace_mutation: bool = False
    purpose: str = SMOKE_PURPOSE

    def dry_run_payload(self) -> dict[str, object]:
        return {
            "schema": SMOKE_SCHEMA,
            "mode": "dry_run",
            **asdict(self),
            "live_execution": False,
            "provider_invocation_count": 0,
        }


def resolve_smoke_plan(
    config: Config,
    context: ContextConfig,
    *,
    requested_provider: str | None,
    requested_model: str | None,
    timeout_seconds: float | None,
) -> SmokePlan:
    """Resolve a configured registry identity without constructing a Provider."""

    model_id = requested_model or config.agents.defaults.model
    provider_id = config.get_provider_name(model_id)
    if provider_id is None:
        raise ValueError(f"model {model_id!r} has no configured Provider")
    if requested_provider is not None and requested_provider != provider_id:
        raise ValueError(
            f"requested Provider {requested_provider!r} does not match configured Provider {provider_id!r}"
        )
    spec = find_by_name(provider_id)
    model_spec = find_by_model(model_id)
    if spec is None:
        raise ValueError(f"Provider {provider_id!r} is not present in the Provider registry")
    if model_spec is None or model_spec.name != spec.name:
        raise ValueError(
            f"model {model_id!r} does not resolve to registry Provider {provider_id!r}"
        )
    agent_provider = config.get_provider_name(config.agents.defaults.model)
    if agent_provider != provider_id:
        raise ValueError(
            "the smoke reuses the configured Agent Provider instance; requested Provider must match it"
        )
    provider_config = config.get_provider(model_id)
    if provider_config is None or (
        not provider_config.api_key and not spec.is_oauth and not spec.is_local
    ):
        raise ValueError(f"Provider {provider_id!r} is not configured with usable credentials")
    timeout = context.knowledge_utility_timeout_seconds if timeout_seconds is None else timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
        raise ValueError("utility timeout must be a finite number")
    if not 0 < timeout <= 30:
        raise ValueError("utility timeout must be greater than zero and at most 30 seconds")
    return SmokePlan(provider_id, model_id, float(timeout))


def _request(plan: SmokePlan) -> JevUtilityRequest:
    query = "Validate one bounded typed Provider utility decision."
    return JevUtilityRequest(
        decision_id="jev3:provider-utility-smoke",
        turn_id=SMOKE_TURN_ID,
        repository_scope_id=canonical_digest({"scope": "jev3-smoke"}),
        query=query,
        query_digest=canonical_digest({"query": query}),
        candidates=(
            JevUtilityCandidate(
                candidate_id="smoke_candidate",
                candidate_type="experience",
                title="Provider utility smoke candidate",
                summary="Preserve deterministic fallback when utility evidence is uncertain.",
                relevance_rank=1,
                relevance_score=1.0,
                applicability_digest=canonical_digest({"applicable": True}),
            ),
        ),
    )


async def _run_live(config: Config, plan: SmokePlan) -> int:
    provider = make_provider(config)
    backend = ProviderUtilityBackend(
        provider,
        ProviderUtilityConfig(
            provider_id=plan.provider_id,
            model_id=plan.model_id,
            max_candidates=plan.candidate_count,
        ),
    )
    adapter = JevUtilityDecisionAdapter(backend, timeout_seconds=plan.utility_timeout_seconds)
    request = _request(plan)
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=SMOKE_TURN_ID,
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    started_ns = time.perf_counter_ns()
    with evidence.turn_scope(recorder):
        result = await adapter.decide(request)
        fallback_used = result.outcome is not DecisionOutcome.SUCCESS
        decisions = result.decisions
        retained_ids = tuple(
            item.candidate_id
            for item in decisions
            if item.effective_decision is UtilityDecision.KEEP
        )
        if fallback_used:
            retained_ids = tuple(item.candidate_id for item in request.candidates)
        abstained_ids = tuple(
            item.candidate_id
            for item in request.candidates
            if item.candidate_id not in retained_ids
        )
        emit_utility_receipt(
            JevUtilityDecisionReceipt(
                turn_id=SMOKE_TURN_ID,
                decision_id=request.decision_id,
                repository_scope_id=request.repository_scope_id,
                backend_id=result.backend_id or adapter.backend_id,
                backend_model=result.backend_model or adapter.backend_model,
                backend_version=result.backend_version or adapter.backend_version,
                config_digest=adapter.config_digest,
                request_digest=request.request_digest,
                input_candidate_ids=tuple(item.candidate_id for item in request.candidates),
                retained_candidate_ids=retained_ids,
                abstained_candidate_ids=abstained_ids,
                decisions=decisions,
                latency_ms=measured_latency_ms(started_ns),
                fallback_used=fallback_used,
                fallback_reason=result.reason if fallback_used else None,
                utility_logical_calls=result.logical_calls,
                utility_provider_attempts=result.provider_attempts,
                utility_input_tokens=result.input_tokens,
                utility_output_tokens=result.output_tokens,
                utility_provider_latency_ms=result.latency_ms,
                utility_logical_call_id=result.logical_call_id,
            )
        )
    utility_starts = tuple(
        item
        for item in records
        if item.get("event_type") == evidence.PROVIDER_ATTEMPT_STARTED
        and item.get("metadata", {}).get("call_role") == "utility"
    )
    logical_call_ids = {
        item.get("correlations", {}).get("logical_call_id") for item in utility_starts
    } - {None}
    receipt_produced = any(
        item.get("event_type") == evidence.DECISION_RECEIPT
        and item.get("metadata", {}).get("receipt_schema") == "pico.jev-utility-decision.v1"
        for item in records
    )
    print(
        json.dumps(
            {
                "schema": SMOKE_SCHEMA,
                "mode": "live",
                **asdict(plan),
                "live_execution": True,
                "provider_logical_calls": len(logical_call_ids),
                "provider_attempts": len(utility_starts),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "provider_latency_ms": result.latency_ms,
                "outcome": result.outcome.value,
                "fallback_used": fallback_used,
                "fallback_reason": result.reason if fallback_used else None,
                "receipt_produced": receipt_produced,
                "decisions": [
                    {
                        "candidate": f"candidate_{index}",
                        "choice": item.decision.value,
                        "effective_decision": item.effective_decision.value,
                        "confidence": item.confidence,
                    }
                    for index, item in enumerate(decisions)
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result.outcome is DecisionOutcome.SUCCESS and receipt_produced else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly execute one JEV.3 Provider utility contract smoke."
    )
    parser.add_argument("--execute-live", action="store_true", help="opt in to one live logical call")
    parser.add_argument("--provider", help="expected configured Provider registry name")
    parser.add_argument("--model", help="configured model; defaults to the Agent model")
    parser.add_argument(
        "--timeout",
        type=float,
        help="independent utility timeout in seconds; defaults to typed context config",
    )
    args = parser.parse_args(argv)
    pico_config = load_pico_config()
    try:
        plan = resolve_smoke_plan(
            pico_config.base,
            pico_config.context,
            requested_provider=args.provider,
            requested_model=args.model,
            timeout_seconds=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute_live:
        print(json.dumps(plan.dry_run_payload(), sort_keys=True))
        return 0
    return asyncio.run(_run_live(pico_config.base, plan))


if __name__ == "__main__":
    raise SystemExit(main())
