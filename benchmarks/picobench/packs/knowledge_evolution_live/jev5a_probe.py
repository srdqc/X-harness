"""JEV.5A utility-only structured-generation contract probe.

The probe never runs an Agent Turn or tools. Expected behavioral directions are
kept in evaluator metadata and are not included in Provider requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pico.cli._helpers import make_provider
from pico.config.pico import load_pico_config
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_PROMPT_DIGEST,
    ProviderUtilityBackend,
    ProviderUtilityConfig,
    UtilityInferencePolicy,
)
from pico.decision_plane.provider_utility_smoke import resolve_smoke_plan
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
from pico.providers.base import LLMProvider
from pico.tracing import evidence
from pico.tracing.evidence import canonical_digest

PROBE_SCHEMA = "pico.jev5a-utility-probe.v1"
PROVIDER_ID = "deepseek"
MODEL_ID = "deepseek/deepseek-v4-flash"
UTILITY_TIMEOUT_SECONDS = 15.0
MAX_OUTPUT_TOKENS = 1024
PLANNED_LOGICAL_CALLS = 3


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    query: str
    candidates: tuple[dict[str, object], ...]
    expected_direction: tuple[str, ...]


CASES = (
    ProbeCase(
        "A",
        "Prepare a Python package release and validate its metadata before publishing.",
        ({
            "title": "Validate package metadata",
            "summary": "Run the repository's metadata validation before publishing the package.",
            "preconditions": "A Python package is ready for a release check.",
        },),
        ("KEEP",),
    ),
    ProbeCase(
        "B",
        "Parse a small CSV file deterministically with Python's existing csv module.",
        ({
            "title": "CSV module availability",
            "summary": "Python includes a csv module in its standard library.",
            "preconditions": "The task already specifies use of the csv module.",
        },),
        ("ABSTAIN",),
    ),
    ProbeCase(
        "C",
        "Implement a crash-safe update for a local JSON settings file.",
        (
            {
                "title": "Atomic local replacement",
                "summary": "Write to a sibling temporary file, flush it, then atomically replace the destination.",
                "preconditions": "The destination and temporary file are on the same filesystem.",
            },
            {
                "title": "Stable JSON display order",
                "summary": "Sort JSON object keys to make human-readable output stable.",
                "preconditions": "Stable display order is desired but crash safety is handled separately.",
            },
        ),
        ("KEEP", "ABSTAIN"),
    ),
)


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- repository-native executable
        cwd=repository,
        text=True,
    ).strip()


def _plan_payload(repository: Path) -> dict[str, object]:
    return {
        "schema": PROBE_SCHEMA,
        "provider_id": PROVIDER_ID,
        "model_id": MODEL_ID,
        "utility_timeout_seconds": UTILITY_TIMEOUT_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "planned_logical_calls": PLANNED_LOGICAL_CALLS,
        "tools_enabled": False,
        "typesafe_enabled": False,
        "normal_agent_turn": False,
        "provider_workspace_mutation": False,
        "prompt_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
        "base_commit": _git_head(repository),
        "cases": tuple(
            {
                "case_id": case.case_id,
                "candidate_count": len(case.candidates),
                "expected_direction": case.expected_direction,
                "case_digest": canonical_digest(
                    {"query": case.query, "candidates": case.candidates}
                ),
            }
            for case in CASES
        ),
    }


def _request(case: ProbeCase) -> JevUtilityRequest:
    query_digest = canonical_digest({"query": case.query})
    return JevUtilityRequest(
        decision_id=f"jev5a:probe:{case.case_id}",
        turn_id=f"jev5a-probe-{case.case_id}",
        repository_scope_id=canonical_digest({"scope": "jev5a-synthetic-probe"}),
        query=case.query,
        query_digest=query_digest,
        candidates=tuple(
            JevUtilityCandidate(
                candidate_id=f"jev5a_{case.case_id.lower()}_{index}",
                candidate_type="synthetic_experience",
                title=str(candidate["title"]),
                summary=str(candidate["summary"]),
                preconditions=str(candidate["preconditions"]),
                relevance_rank=index + 1,
                relevance_score=1.0 - index * 0.1,
                applicability_digest=canonical_digest(
                    {"case_id": case.case_id, "candidate": index}
                ),
            )
            for index, candidate in enumerate(case.candidates)
        ),
    )


async def run_probe(provider: LLMProvider) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for case in CASES:
        backend = ProviderUtilityBackend(
            provider,
            ProviderUtilityConfig(
                provider_id=PROVIDER_ID,
                model_id=MODEL_ID,
                max_candidates=len(case.candidates),
                max_tokens=MAX_OUTPUT_TOKENS,
                inference_policy=UtilityInferencePolicy(max_output_tokens=MAX_OUTPUT_TOKENS),
            ),
        )
        adapter = JevUtilityDecisionAdapter(
            backend, timeout_seconds=UTILITY_TIMEOUT_SECONDS
        )
        request = _request(case)
        records: list[dict[str, Any]] = []
        recorder = evidence.TurnEvidenceRecorder(
            turn_id=request.turn_id,
            conversation_id=None,
            trace_id=None,
            root_span_id=None,
            writer=lambda record: records.append(record) is None,
        )
        started_ns = time.perf_counter_ns()
        with evidence.turn_scope(recorder):
            result = await adapter.decide(request)
            fallback = result.outcome is not DecisionOutcome.SUCCESS
            retained = tuple(
                item.candidate_id
                for item in result.decisions
                if item.effective_decision is UtilityDecision.KEEP
            )
            if fallback:
                retained = tuple(item.candidate_id for item in request.candidates)
            abstained = tuple(
                item.candidate_id
                for item in request.candidates
                if item.candidate_id not in retained
            )
            emit_utility_receipt(
                JevUtilityDecisionReceipt(
                    turn_id=request.turn_id,
                    decision_id=request.decision_id,
                    repository_scope_id=request.repository_scope_id,
                    backend_id=result.backend_id or adapter.backend_id,
                    backend_model=result.backend_model or adapter.backend_model,
                    backend_version=result.backend_version or adapter.backend_version,
                    config_digest=adapter.config_digest,
                    request_digest=request.request_digest,
                    input_candidate_ids=tuple(item.candidate_id for item in request.candidates),
                    retained_candidate_ids=retained,
                    abstained_candidate_ids=abstained,
                    decisions=result.decisions,
                    latency_ms=measured_latency_ms(started_ns),
                    fallback_used=fallback,
                    fallback_reason=result.reason if fallback else None,
                    utility_logical_calls=result.logical_calls,
                    utility_provider_attempts=result.provider_attempts,
                    utility_input_tokens=result.input_tokens,
                    utility_output_tokens=result.output_tokens,
                    utility_provider_latency_ms=result.latency_ms,
                    utility_logical_call_id=result.logical_call_id,
                    utility_finish_reason=result.finish_reason,
                    utility_malformed_category=result.malformed_category,
                    utility_response_character_count=result.response_character_count,
                    utility_response_digest=result.response_digest,
                    utility_top_level_shape=result.top_level_shape,
                    utility_parse_stage=result.parse_stage,
                    utility_generation_outcome=result.generation_outcome,
                    utility_payload_outcome=result.payload_outcome,
                    utility_reasoning_mode_requested=result.reasoning_mode_requested,
                    utility_reasoning_mode_effective=result.reasoning_mode_effective,
                    utility_structured_output_requested=result.structured_output_requested,
                    utility_reasoning_tokens=result.reasoning_tokens,
                    utility_visible_output_tokens=result.visible_output_tokens,
                    utility_reasoning_content_present=result.reasoning_content_present,
                    utility_reasoning_character_count=result.reasoning_character_count,
                )
            )
        receipt_produced = any(
            item.get("event_type") == evidence.DECISION_RECEIPT for item in records
        )
        raw_choices = tuple(item.decision.value.upper() for item in result.decisions)
        effective_choices = tuple(
            item.effective_decision.value.upper() for item in result.decisions
        )
        contract_success = bool(
            result.outcome is DecisionOutcome.SUCCESS
            and result.logical_calls == 1
            and result.finish_reason != "LENGTH"
            and (result.response_character_count or 0) > 0
            and len(result.decisions) == len(request.candidates)
            and not fallback
            and receipt_produced
        )
        results.append(
            {
                "case_id": case.case_id,
                "expected_direction": case.expected_direction,
                "candidate_count": len(request.candidates),
                "outcome": result.outcome.value,
                "raw_choices": raw_choices,
                "effective_choices": effective_choices,
                "finish_reason": result.finish_reason,
                "generation_outcome": result.generation_outcome,
                "payload_outcome": result.payload_outcome,
                "reasoning_mode_requested": result.reasoning_mode_requested,
                "reasoning_mode_effective": result.reasoning_mode_effective,
                "structured_output_requested": result.structured_output_requested,
                "reasoning_tokens": result.reasoning_tokens,
                "output_tokens": result.output_tokens,
                "visible_output_tokens": result.visible_output_tokens,
                "visible_response_character_count": result.response_character_count,
                "reasoning_content_present": result.reasoning_content_present,
                "reasoning_character_count": result.reasoning_character_count,
                "input_tokens": result.input_tokens,
                "provider_latency_ms": result.latency_ms,
                "logical_calls": result.logical_calls,
                "provider_attempts": result.provider_attempts,
                "fallback_used": fallback,
                "fallback_reason": result.reason if fallback else None,
                "receipt_produced": receipt_produced,
                "contract_success": contract_success,
            }
        )
    actual_calls = sum(int(item["logical_calls"]) for item in results)
    return {
        "schema": PROBE_SCHEMA,
        "provider_id": PROVIDER_ID,
        "requested_model_id": MODEL_ID,
        "utility_timeout_seconds": UTILITY_TIMEOUT_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "planned_logical_calls": PLANNED_LOGICAL_CALLS,
        "actual_logical_calls": actual_calls,
        "fallback_count": sum(bool(item["fallback_used"]) for item in results),
        "contract_success_count": sum(bool(item["contract_success"]) for item in results),
        "total_input_tokens": sum(int(item["input_tokens"] or 0) for item in results),
        "total_output_tokens": sum(int(item["output_tokens"] or 0) for item in results),
        "total_provider_latency_ms": round(
            sum(float(item["provider_latency_ms"] or 0.0) for item in results), 6
        ),
        "results": tuple(results),
    }


def validate_artifact(artifact: dict[str, object]) -> None:
    if artifact.get("schema") != PROBE_SCHEMA:
        raise ValueError("unsupported JEV.5A probe artifact schema")
    results = artifact.get("results")
    if not isinstance(results, (list, tuple)) or len(results) != PLANNED_LOGICAL_CALLS:
        raise ValueError("JEV.5A artifact must contain exactly three planned cases")
    actual = artifact.get("actual_logical_calls")
    if isinstance(actual, bool) or not isinstance(actual, int) or not 0 <= actual <= PLANNED_LOGICAL_CALLS:
        raise ValueError("JEV.5A logical-call bound violated")
    if any(item.get("case_id") != case.case_id for item, case in zip(results, CASES, strict=True)):
        raise ValueError("JEV.5A case order/identity mismatch")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded JEV.5A utility-only probe.")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    plan = _plan_payload(repository)
    campaign_id = f"jev5a-{canonical_digest(plan)[:16]}"
    campaign_root = repository / ".p3r" / campaign_id
    pico_config = load_pico_config()
    try:
        resolved = resolve_smoke_plan(
            pico_config.base,
            pico_config.context,
            requested_provider=PROVIDER_ID,
            requested_model=MODEL_ID,
            timeout_seconds=UTILITY_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        parser.error(str(exc))
    dry_run = {
        **plan,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "resolved_provider_id": resolved.provider_id,
        "resolved_model_id": resolved.model_id,
        "live_execution": False,
        "provider_invocation_count": 0,
    }
    if not args.execute_live:
        print(json.dumps(dry_run, sort_keys=True))
        return 0
    if campaign_root.exists():
        parser.error(f"campaign already exists and cannot be rerun: {campaign_root}")
    campaign_root.mkdir(parents=True)
    _atomic_json(campaign_root / "manifest.json", dry_run | {"live_execution": True})
    provider = make_provider(pico_config.base)
    artifact = asyncio.run(run_probe(provider))
    artifact.update(campaign_id=campaign_id, campaign_root=str(campaign_root))
    validate_artifact(artifact)
    artifact["artifact_digest"] = canonical_digest(artifact)
    _atomic_json(campaign_root / "result.json", artifact)
    print(json.dumps(artifact, sort_keys=True))
    return 0 if artifact["contract_success_count"] == PLANNED_LOGICAL_CALLS else 1


if __name__ == "__main__":
    raise SystemExit(main())
