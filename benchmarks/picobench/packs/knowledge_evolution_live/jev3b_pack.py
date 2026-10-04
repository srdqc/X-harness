"""Frozen eight-case synthetic behavior pack for Provider utility decisions.

This is exploratory and non-claim-eligible. It validates typed behavior only;
it is not an Agent benchmark and must not be used to tune the frozen prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from pico.cli._helpers import make_provider
from pico.config.pico import load_pico_config
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_PROMPT_DIGEST,
    ProviderUtilityBackend,
    ProviderUtilityConfig,
)
from pico.decision_plane.provider_utility_smoke import resolve_smoke_plan
from pico.decision_plane.types import DecisionOutcome
from pico.decision_plane.utility import (
    JEV_UTILITY_RESPONSE_SCHEMA,
    JEV_UTILITY_SCHEMA_VERSION,
    JevUtilityCandidate,
    JevUtilityDecisionAdapter,
    JevUtilityDecisionReceipt,
    JevUtilityRequest,
    UtilityDecision,
    emit_utility_receipt,
    measured_latency_ms,
)
from pico.tracing import evidence
from pico.tracing.evidence import canonical_digest

PACK_SCHEMA = "pico.jev3b-synthetic-utility-pack.v1"
PACK_VERSION = 1
PROVIDER_ID = "deepseek"
MODEL_ID = "deepseek/deepseek-v4-flash"
UTILITY_TIMEOUT_SECONDS = 15.0
UTILITY_MAX_TOKENS = 1024
MAX_LOGICAL_CALLS = 8
REPOSITORY_SCOPE_ID = canonical_digest({"scope": "jev3b-synthetic-placeholder"})


@dataclass(frozen=True)
class SyntheticCandidate:
    candidate_id: str
    candidate_type: str
    title: str
    summary: str
    preconditions: str
    handcrafted_expectation: str
    expectation_reason: str


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    task: str
    candidates: tuple[SyntheticCandidate, ...]


def _candidate(
    case: int,
    suffix: str,
    title: str,
    summary: str,
    preconditions: str,
    expectation: str,
    reason: str,
    *,
    candidate_type: str = "experience",
) -> SyntheticCandidate:
    return SyntheticCandidate(
        f"synthetic-{case:02d}-{suffix}",
        candidate_type,
        title,
        summary,
        preconditions,
        expectation,
        reason,
    )


PACK_CASES = (
    SyntheticCase(
        "case_01",
        "Extend a configuration parser with a new enum field. Existing valid values must remain compatible, while unknown values must be rejected.",
        (
            _candidate(
                1,
                "a",
                "Canonical enum validation",
                "When extending enum parsing, validate input against the canonical enum and retain regression coverage for previously valid values.",
                "The task changes parsing or validation of an enumerated configuration value.",
                "KEEP",
                "The guidance directly addresses the behavior and compatibility constraint.",
            ),
        ),
    ),
    SyntheticCase(
        "case_02",
        "Make a pure table-formatting helper return rows in deterministic key order.",
        (
            _candidate(
                2,
                "a",
                "Cross-system presentation audit",
                "When changing output formatting, inspect logging, tracing, telemetry, and downstream presentation modules before editing to avoid hidden coupling.",
                "The formatting change affects a cross-system presentation pipeline.",
                "ABSTAIN",
                "Broad exploration is not materially justified for this narrow pure-helper task.",
            ),
        ),
    ),
    SyntheticCase(
        "case_03",
        "Reduce intermittent failures in a third-party request path. The failure cause has not yet been identified.",
        (
            _candidate(
                3,
                "a",
                "Transient failure backoff",
                "Use bounded exponential backoff when failures are caused by rate limits or transient transport errors.",
                "The observed failures are rate-limit or transient transport failures.",
                "UNCERTAIN",
                "The guidance may help, but the required precondition is unknown.",
            ),
        ),
    ),
    SyntheticCase(
        "case_04",
        "Add a file-backed cache update that must not expose partially written content to readers.",
        (
            _candidate(
                4,
                "a",
                "Atomic file replacement",
                "Write new content to a temporary file and atomically replace the destination after the write is complete.",
                "The task persists replaceable file-backed state.",
                "KEEP",
                "The guidance directly provides the required atomic update pattern.",
            ),
            _candidate(
                4,
                "b",
                "CLI presentation consistency audit",
                "Before modifying file-backed state, audit unrelated CLI theme and naming configuration for consistency.",
                "The task also changes CLI presentation conventions.",
                "ABSTAIN",
                "The candidate is unrelated to the required atomic file behavior.",
            ),
        ),
    ),
    SyntheticCase(
        "case_05",
        "Optimize lookup performance for an immutable in-memory table that is built once and never modified after initialization.",
        (
            _candidate(
                5,
                "a",
                "Concurrent mutable-map locking",
                "Guard shared map writes with a lock when multiple tasks may mutate the map concurrently.",
                "The shared in-memory map is mutable after initialization and receives concurrent writes.",
                "ABSTAIN",
                "The explicit mutability and concurrent-write precondition does not hold.",
            ),
        ),
    ),
    SyntheticCase(
        "case_06",
        "Add request deduplication to a local CLI command. Requirements do not state whether deduplication must survive process restarts.",
        (
            _candidate(
                6,
                "a",
                "Durable deduplication fingerprints",
                "Persist request fingerprints so duplicate detection remains effective across process restarts.",
                "Deduplication must survive process termination or restart.",
                "UNCERTAIN",
                "The durability requirement cannot be determined from the task.",
            ),
        ),
    ),
    SyntheticCase(
        "case_07",
        "Validate an incoming JSON object against an exact schema and reject unknown keys.",
        (
            _candidate(
                7,
                "a",
                "Strict schema validation",
                "Use exact schema validation that rejects properties not declared by the contract.",
                "The input is validated against a structured schema.",
                "KEEP",
                "The guidance directly implements the requested validation contract.",
            ),
            _candidate(
                7,
                "b",
                "Duplicate manual key validation",
                "After exact schema validation has already rejected unknown properties, walk every input key again and independently reject unknown keys a second time.",
                "Exact schema validation already enforces unknown-key rejection.",
                "ABSTAIN",
                "The second validation pass is semantically related but redundant.",
            ),
        ),
    ),
    SyntheticCase(
        "case_08",
        "Prepare a release checklist for a small Python library before publishing a new version.",
        (
            _candidate(
                8,
                "a",
                "Library release verification",
                "Before publishing, run focused tests, build the package, and verify version metadata and generated artifacts.",
                "The task is preparing a library release.",
                "KEEP",
                "The reusable checklist directly supports the release task.",
                candidate_type="skill_candidate",
            ),
        ),
    ),
)


def semantic_payload() -> dict[str, Any]:
    return {
        "schema": PACK_SCHEMA,
        "pack_version": PACK_VERSION,
        "status": "exploratory_non_claim_eligible",
        "provider_id": PROVIDER_ID,
        "model_id": MODEL_ID,
        "utility_timeout_seconds": UTILITY_TIMEOUT_SECONDS,
        "utility_max_tokens": UTILITY_MAX_TOKENS,
        "max_logical_calls": MAX_LOGICAL_CALLS,
        "prompt_template_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
        "response_schema": JEV_UTILITY_RESPONSE_SCHEMA,
        "response_schema_version": JEV_UTILITY_SCHEMA_VERSION,
        "repository_scope_id": REPOSITORY_SCOPE_ID,
        "cases": [
            {
                "case_id": item.case_id,
                "task": item.task,
                "candidates": [asdict(candidate) for candidate in item.candidates],
            }
            for item in PACK_CASES
        ],
    }


def pack_manifest() -> dict[str, Any]:
    payload = semantic_payload()
    digest = canonical_digest(payload)
    return {**payload, "semantic_digest": digest, "freeze_digest": digest}


def artifact_root(repository: Path) -> Path:
    digest = pack_manifest()["semantic_digest"]
    return repository / ".p3r" / f"jev3b-{str(digest)[:16]}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_artifact(repository: Path) -> Path:
    root = artifact_root(repository)
    manifest_path = root / "manifest.json"
    manifest = pack_manifest()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing JEV.3B manifest differs from the frozen pack")
    else:
        _atomic_json(manifest_path, manifest)
    return root


def build_request(case: SyntheticCase) -> JevUtilityRequest:
    return JevUtilityRequest(
        decision_id=f"jev3b:{case.case_id}",
        turn_id=f"jev3b:{case.case_id}",
        repository_scope_id=REPOSITORY_SCOPE_ID,
        query=case.task,
        query_digest=canonical_digest({"query": case.task}),
        candidates=tuple(
            JevUtilityCandidate(
                candidate_id=item.candidate_id,
                candidate_type=item.candidate_type,
                title=item.title,
                summary=item.summary,
                relevance_rank=index,
                relevance_score=1.0,
                applicability_digest=canonical_digest(
                    {"candidate_id": item.candidate_id, "applicable": True}
                ),
                preconditions=item.preconditions,
            )
            for index, item in enumerate(case.candidates, start=1)
        ),
    )


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values: list[float], *, include_p95: bool) -> dict[str, float | None]:
    if not values:
        base: dict[str, float | None] = {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    else:
        base = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    if include_p95:
        base["p95"] = _percentile_95(values)
    return {key: round(value, 6) if value is not None else None for key, value in base.items()}


def reduce_results(
    results: Sequence[dict[str, Any]],
    *,
    systemic_failure: bool = False,
) -> dict[str, Any]:
    cases_by_id = {item.case_id: item for item in PACK_CASES}
    raw_counts: Counter[str] = Counter()
    expected_observed: dict[str, Counter[str]] = defaultdict(Counter)
    decision_matches = 0
    decision_count = 0
    case_matches = 0
    latencies: list[float] = []
    confidences: list[float] = []
    confidence_by_choice: dict[str, list[float]] = defaultdict(list)
    for result in results:
        case = cases_by_id[result["case_id"]]
        expected = {
            item.candidate_id: item.handcrafted_expectation for item in case.candidates
        }
        observed = {
            item["candidate_id"]: item["raw_choice"]
            for item in result.get("decisions", ())
        }
        exact = result.get("decision_source") == "MODEL_DECISION" and observed == expected
        case_matches += int(exact)
        for candidate in case.candidates:
            choice = observed.get(candidate.candidate_id, "FALLBACK")
            expected_observed[candidate.handcrafted_expectation][choice] += 1
            decision_count += 1
            decision_matches += int(choice == candidate.handcrafted_expectation)
        for item in result.get("decisions", ()):
            raw_counts[item["raw_choice"]] += 1
            if item.get("confidence") is not None:
                confidence = float(item["confidence"])
                confidences.append(confidence)
                confidence_by_choice[item["raw_choice"]].append(confidence)
        latency = result.get("provider_latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))

    complete = len(results) == len(PACK_CASES)
    contract_ok = complete and not systemic_failure and all(
        item.get("receipt_produced") is True
        and item.get("provider_logical_calls") == 1
        and item.get("tool_call_count") == 0
        and item.get("normal_agent_turn_count") == 0
        and item.get("workspace_mutation") is False
        and item.get("secret_leak") is False
        for item in results
    )
    contract_status = "PASS" if contract_ok else "INVALID"
    choices = {choice for choice, count in raw_counts.items() if count}
    clear_keep = any(
        candidate.handcrafted_expectation == "KEEP"
        and expected_observed["KEEP"]["KEEP"] > 0
        for case in PACK_CASES
        for candidate in case.candidates
    )
    clear_abstain = any(
        candidate.handcrafted_expectation == "ABSTAIN"
        and expected_observed["ABSTAIN"]["ABSTAIN"] > 0
        for case in PACK_CASES
        for candidate in case.candidates
    )
    agreement_rate = decision_matches / decision_count if decision_count else 0.0
    if contract_status != "PASS" or len(choices) <= 1:
        behavioral_signal = "UNINFORMATIVE"
    elif clear_keep and clear_abstain and agreement_rate >= 0.5:
        behavioral_signal = "PROMISING"
    else:
        behavioral_signal = "MIXED"
    recommendation = {
        "PROMISING": "GO_JEV4_PILOT",
        "MIXED": "HOLD_AND_ANALYZE",
        "UNINFORMATIVE": "STOP_PROVIDER_UTILITY",
    }[behavioral_signal]
    if contract_status == "INVALID":
        recommendation = "STOP_PROVIDER_UTILITY"
    return {
        "case_count": len(PACK_CASES),
        "completed_case_count": len(results),
        "decision_count": decision_count,
        "schema_valid_case_count": sum(item.get("schema_valid") is True for item in results),
        "fallback_case_count": sum(item.get("fallback_used") is True for item in results),
        "raw_choice_counts": {choice: raw_counts[choice] for choice in ("KEEP", "ABSTAIN", "UNCERTAIN")},
        "handcrafted_case_exact_match_count": case_matches,
        "handcrafted_case_exact_match_rate": round(case_matches / len(PACK_CASES), 6),
        "handcrafted_decision_exact_match_count": decision_matches,
        "handcrafted_decision_exact_match_rate": round(agreement_rate, 6),
        "per_choice_agreement": {
            expected: dict(sorted(values.items()))
            for expected, values in sorted(expected_observed.items())
        },
        "provider_logical_call_total": sum(int(item.get("provider_logical_calls", 0)) for item in results),
        "provider_attempt_total": sum(int(item.get("provider_attempts", 0)) for item in results),
        "input_token_total": sum(int(item.get("input_tokens") or 0) for item in results),
        "output_token_total": sum(int(item.get("output_tokens") or 0) for item in results),
        "provider_latency_ms": _stats(latencies, include_p95=True),
        "confidence": {
            **_stats(confidences, include_p95=False),
            "by_raw_choice": {
                choice: _stats(values, include_p95=False)
                for choice, values in sorted(confidence_by_choice.items())
            },
        },
        "receipt_count": sum(item.get("receipt_produced") is True for item in results),
        "skill_activation_count": 0,
        "normal_agent_turn_count": 0,
        "tool_call_count": 0,
        "workspace_mutation": False,
        "secret_leak": False,
        "systemic_failure": systemic_failure,
        "rerun_case_count": 0,
        "contract_status": contract_status,
        "behavioral_signal": behavioral_signal,
        "recommendation": recommendation,
    }


def _receipt(
    request: JevUtilityRequest,
    result: Any,
    adapter: JevUtilityDecisionAdapter,
    total_latency_ms: float,
) -> JevUtilityDecisionReceipt:
    fallback_used = result.outcome is not DecisionOutcome.SUCCESS
    retained = tuple(
        item.candidate_id
        for item in result.decisions
        if item.effective_decision is UtilityDecision.KEEP
    )
    if fallback_used:
        retained = tuple(item.candidate_id for item in request.candidates)
    abstained = tuple(
        item.candidate_id for item in request.candidates if item.candidate_id not in retained
    )
    return JevUtilityDecisionReceipt(
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
        latency_ms=total_latency_ms,
        fallback_used=fallback_used,
        fallback_reason=result.reason if fallback_used else None,
        utility_logical_calls=result.logical_calls,
        utility_provider_attempts=result.provider_attempts,
        utility_input_tokens=result.input_tokens,
        utility_output_tokens=result.output_tokens,
        utility_provider_latency_ms=result.latency_ms,
        utility_logical_call_id=result.logical_call_id,
    )


async def _execute_case(
    case: SyntheticCase,
    adapter: JevUtilityDecisionAdapter,
) -> dict[str, Any]:
    request = build_request(case)
    records: list[dict] = []
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=request.turn_id or case.case_id,
        conversation_id=None,
        trace_id=None,
        root_span_id=None,
        writer=lambda record: records.append(record) is None,
    )
    started_ns = time.perf_counter_ns()
    with evidence.turn_scope(recorder):
        result = await adapter.decide(request)
        total_latency_ms = measured_latency_ms(started_ns)
        emit_utility_receipt(_receipt(request, result, adapter, total_latency_ms))
    starts = tuple(
        item
        for item in records
        if item.get("event_type") == evidence.PROVIDER_ATTEMPT_STARTED
        and item.get("metadata", {}).get("call_role") == "utility"
    )
    logical_ids = {
        item.get("correlations", {}).get("logical_call_id") for item in starts
    } - {None}
    receipt_event = next(
        (
            item
            for item in records
            if item.get("event_type") == evidence.DECISION_RECEIPT
            and item.get("metadata", {}).get("receipt_schema")
            == "pico.jev-utility-decision.v1"
        ),
        None,
    )
    fallback_used = result.outcome is not DecisionOutcome.SUCCESS
    decisions = [
        {
            "candidate_id": item.candidate_id,
            "raw_choice": item.decision.value.upper(),
            "effective_decision": item.effective_decision.value.upper(),
            "confidence": item.confidence,
        }
        for item in result.decisions
    ]
    return {
        "case_id": case.case_id,
        "decision_source": "FALLBACK" if fallback_used else "MODEL_DECISION",
        "schema_valid": result.outcome is DecisionOutcome.SUCCESS,
        "exact_candidate_coverage": (
            {item["candidate_id"] for item in decisions}
            == {item.candidate_id for item in case.candidates}
            if not fallback_used
            else False
        ),
        "decisions": decisions,
        "fallback_used": fallback_used,
        "fallback_reason": result.reason if fallback_used else None,
        "provider_logical_calls": len(logical_ids),
        "provider_attempts": len(starts),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "provider_latency_ms": result.latency_ms,
        "total_utility_latency_ms": total_latency_ms,
        "receipt_produced": receipt_event is not None,
        "receipt": receipt_event.get("metadata") if receipt_event is not None else None,
        "tool_call_count": 0,
        "normal_agent_turn_count": 0,
        "workspace_mutation": False,
        "skill_activation_count": 0,
        "secret_leak": False,
    }


async def execute_pack(repository: Path) -> tuple[Path, dict[str, Any]]:
    root = prepare_artifact(repository)
    run_path = root / "run.json"
    if run_path.exists():
        raise RuntimeError("JEV.3B run artifact already exists; live reruns are forbidden")
    pico_config = load_pico_config()
    plan = resolve_smoke_plan(
        pico_config.base,
        pico_config.context,
        requested_provider=PROVIDER_ID,
        requested_model=MODEL_ID,
        timeout_seconds=UTILITY_TIMEOUT_SECONDS,
    )
    provider = make_provider(pico_config.base)
    backend = ProviderUtilityBackend(
        provider,
        ProviderUtilityConfig(
            provider_id=plan.provider_id,
            model_id=plan.model_id,
            max_candidates=2,
            max_tokens=UTILITY_MAX_TOKENS,
        ),
    )
    adapter = JevUtilityDecisionAdapter(backend, timeout_seconds=plan.utility_timeout_seconds)
    results: list[dict[str, Any]] = []
    systemic_failure = False
    consecutive_unavailable = 0
    _atomic_json(
        run_path,
        {
            "schema": PACK_SCHEMA,
            "semantic_digest": pack_manifest()["semantic_digest"],
            "status": "running",
            "results": results,
        },
    )
    for case in PACK_CASES:
        if len(results) >= MAX_LOGICAL_CALLS:
            raise RuntimeError("JEV.3B logical call limit reached")
        result = await _execute_case(case, adapter)
        results.append(result)
        reason = result.get("fallback_reason")
        if reason == "authentication":
            systemic_failure = True
        if reason in {"unavailable", "rate_limit"}:
            consecutive_unavailable += 1
        else:
            consecutive_unavailable = 0
        if consecutive_unavailable >= 2:
            systemic_failure = True
        _atomic_json(
            run_path,
            {
                "schema": PACK_SCHEMA,
                "semantic_digest": pack_manifest()["semantic_digest"],
                "status": "infra_invalid" if systemic_failure else "running",
                "results": results,
            },
        )
        if systemic_failure:
            break
    summary = reduce_results(results, systemic_failure=systemic_failure)
    artifact = {
        "schema": PACK_SCHEMA,
        "semantic_digest": pack_manifest()["semantic_digest"],
        "status": "complete" if not systemic_failure else "infra_invalid",
        "results": results,
        "summary": summary,
    }
    serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True).casefold()
    forbidden = ("api_key", "authorization", "bearer ", "raw_response", "chain-of-thought")
    if any(value in serialized for value in forbidden):
        raise RuntimeError("secret-bearing or raw transport field detected in JEV.3B artifact")
    _atomic_json(run_path, artifact)
    return root, artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or execute the frozen JEV.3B pack.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="persist the frozen manifest only")
    mode.add_argument("--execute-live", action="store_true", help="run at most eight live utility calls")
    parser.add_argument("--provider", default=PROVIDER_ID)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--timeout", type=float, default=UTILITY_TIMEOUT_SECONDS)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.provider != PROVIDER_ID or args.model != MODEL_ID or args.timeout != UTILITY_TIMEOUT_SECONDS:
        parser.error("JEV.3B provider, model, and timeout are frozen")
    root = artifact_root(args.repository.resolve())
    if args.prepare:
        root = prepare_artifact(args.repository.resolve())
        print(
            json.dumps(
                {
                    "artifact_root": str(root),
                    "case_count": len(PACK_CASES),
                    "planned_logical_calls": len(PACK_CASES),
                    "semantic_digest": pack_manifest()["semantic_digest"],
                    "live_execution": False,
                    "provider_invocation_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_live:
        print(
            json.dumps(
                {
                    "artifact_root": str(root),
                    "case_count": len(PACK_CASES),
                    "planned_logical_calls": len(PACK_CASES),
                    "semantic_digest": pack_manifest()["semantic_digest"],
                    "live_execution": False,
                    "provider_invocation_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    root, artifact = asyncio.run(execute_pack(args.repository.resolve()))
    print(
        json.dumps(
            {
                "artifact_root": str(root),
                "schema": artifact["schema"],
                "semantic_digest": artifact["semantic_digest"],
                "status": artifact["status"],
                "summary": artifact["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["summary"]["contract_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_LOGICAL_CALLS",
    "MODEL_ID",
    "PACK_CASES",
    "PACK_SCHEMA",
    "PACK_VERSION",
    "PROVIDER_ID",
    "UTILITY_TIMEOUT_SECONDS",
    "artifact_root",
    "build_request",
    "pack_manifest",
    "prepare_artifact",
    "reduce_results",
    "semantic_payload",
]
