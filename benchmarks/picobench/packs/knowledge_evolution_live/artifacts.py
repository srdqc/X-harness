"""Immutable local artifact I/O and versioned P3R decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_digest, to_primitive
from benchmarks.picobench.schema import ExperimentRef

from .schema import (
    Arm,
    Availability,
    CampaignManifest,
    CampaignMode,
    CampaignPaths,
    InfraInvalidReason,
    MetricValue,
    PlannedRun,
    RunMetrics,
    RunRecord,
    RuntimeBudget,
    RunValidity,
    TokenAccountingStatus,
)


def store_for(paths: CampaignPaths) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(paths.root.name, paths.root))


def freeze_manifest(paths: CampaignPaths, manifest: CampaignManifest) -> None:
    store_for(paths).append_immutable(paths.manifest, to_primitive(manifest))


def write_run(paths: CampaignPaths, record: RunRecord) -> None:
    store_for(paths).append_immutable(paths.runs / f"{record.run_id}.json", to_primitive(record))


def load_manifest(path: Path) -> CampaignManifest:
    raw = _object(path)
    budget_raw = dict(raw["budget"])
    if "max_agent_steps" in budget_raw:
        budget_raw = {
            "max_agent_iterations": budget_raw["max_agent_steps"],
            "provider_logical_calls_observational": budget_raw.get("max_provider_logical_calls"),
            "tool_calls_observational": budget_raw.get("max_tool_calls"),
            "final_synthesis_call_allowed": True,
            "wall_clock_timeout_seconds": budget_raw["wall_clock_timeout_seconds"],
            "context_window_tokens": budget_raw["context_window_tokens"],
            "retry_policy": budget_raw["retry_policy"],
        }
    manifest = CampaignManifest(
        campaign_id=str(raw["campaign_id"]),
        mode=CampaignMode(raw["mode"]),
        base_commit_sha=str(raw["base_commit_sha"]),
        campaign_seed=int(raw["campaign_seed"]),
        provider_id=str(raw["provider_id"]),
        actual_model_id=str(raw["actual_model_id"]),
        provider_model_config_digest=str(raw["provider_model_config_digest"]),
        tool_config_digest=str(raw["tool_config_digest"]),
        runtime_config_digest=str(raw["runtime_config_digest"]),
        budget=RuntimeBudget(**budget_raw),
        task_ids=tuple(raw["task_ids"]),
        task_prompt_digests=tuple(tuple(item) for item in raw["task_prompt_digests"]),
        verifier_digests=tuple(tuple(item) for item in raw["verifier_digests"]),
        knowledge_corpus_digest=str(raw["knowledge_corpus_digest"]),
        planned_runs=tuple(
            PlannedRun(
                run_id=str(item["run_id"]),
                task_id=str(item["task_id"]),
                arm=Arm(item["arm"]),
                repetition=int(item["repetition"]),
                order=int(item["order"]),
                replacement_for_run_id=item.get("replacement_for_run_id"),
                replacement_ordinal=int(item.get("replacement_ordinal", 0)),
            )
            for item in raw["planned_runs"]
        ),
        created_at=str(raw["created_at"]),
        validity_policy_version=int(raw.get("validity_policy_version", 1)),
        task_contract_version=int(raw.get("task_contract_version", 1)),
        benchmark_version=str(raw["benchmark_version"]),
        manifest_digest=str(raw["manifest_digest"]),
        schema=str(raw["schema"]),
        schema_version=int(raw["schema_version"]),
    )
    manifest.validate()
    return manifest


def load_run(path: Path) -> RunRecord:
    raw = _object(path)
    metric_raw = dict(raw["metrics"])
    metric_fields = {
        name: MetricValue(value["value"], Availability(value["availability"]))
        for name, value in metric_raw.items()
        if isinstance(value, dict) and "availability" in value
    }
    if "token_accounting_status" in metric_raw:
        metric_fields["token_accounting_status"] = TokenAccountingStatus(
            metric_raw["token_accounting_status"]
        )
    for name in ("iteration_exhausted", "final_synthesis_call_present"):
        if name in metric_raw:
            metric_fields[name] = bool(metric_raw[name])
    metrics = RunMetrics(**metric_fields)
    additive = {
        "run_validity": RunValidity(raw.get("run_validity", RunValidity.VALID.value)),
        "infra_invalid_reason": (
            InfraInvalidReason(raw["infra_invalid_reason"])
            if raw.get("infra_invalid_reason") is not None
            else None
        ),
        "normalized_provider_failure_categories": tuple(
            raw.get("normalized_provider_failure_categories", ())
        ),
        "recovered_provider_failure_count": int(raw.get("recovered_provider_failure_count", 0)),
        "terminal_provider_failure": bool(raw.get("terminal_provider_failure", False)),
        "verifier_findings": tuple(raw.get("verifier_findings", ())),
        "replacement_for_run_id": raw.get("replacement_for_run_id"),
    }
    record = RunRecord(
        **{
            **{
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "arm",
                    "metrics",
                    "run_validity",
                    "infra_invalid_reason",
                    "normalized_provider_failure_categories",
                    "recovered_provider_failure_count",
                    "terminal_provider_failure",
                    "verifier_findings",
                    "replacement_for_run_id",
                }
            },
            "arm": Arm(raw["arm"]),
            "metrics": metrics,
            "safety_findings": tuple(raw["safety_findings"]),
            "knowledge_retrieval_refs": tuple(raw["knowledge_retrieval_refs"]),
            "knowledge_usage_refs": tuple(raw["knowledge_usage_refs"]),
            "outcome_association_refs": tuple(raw["outcome_association_refs"]),
            "retrieved_candidate_ids": tuple(raw["retrieved_candidate_ids"]),
            "injected_candidate_ids": tuple(raw["injected_candidate_ids"]),
            "referenced_candidate_ids": tuple(raw["referenced_candidate_ids"]),
            "activated_candidate_ids": tuple(raw["activated_candidate_ids"]),
            "trace_evidence_refs": tuple(raw["trace_evidence_refs"]),
            **additive,
        }
    )
    record.validate()
    return record


def load_runs(paths: CampaignPaths) -> tuple[RunRecord, ...]:
    if not paths.runs.exists():
        return ()
    return tuple(load_run(path) for path in sorted(paths.runs.glob("*.json")))


def completed_run_ids(paths: CampaignPaths) -> frozenset[str]:
    return frozenset(item.run_id for item in load_runs(paths))


def load_replacements(paths: CampaignPaths) -> tuple[PlannedRun, ...]:
    if not paths.replacements.exists():
        return ()
    values: list[PlannedRun] = []
    for path in sorted(paths.replacements.glob("*.json")):
        raw = _object(path)
        digest = raw.pop("integrity_digest", None)
        if digest != canonical_digest(raw):
            raise ValueError("P3R replacement plan integrity mismatch")
        values.append(
            PlannedRun(
                run_id=str(raw["run_id"]),
                task_id=str(raw["task_id"]),
                arm=Arm(raw["arm"]),
                repetition=int(raw["repetition"]),
                order=int(raw["order"]),
                replacement_for_run_id=str(raw["replacement_for_run_id"]),
                replacement_ordinal=int(raw["replacement_ordinal"]),
            )
        )
    return tuple(values)


def write_replacement(paths: CampaignPaths, planned: PlannedRun) -> None:
    payload = to_primitive(planned)
    payload["schema"] = "pico.picobench.p3r-replacement.v1"
    payload["integrity_digest"] = canonical_digest(payload)
    store_for(paths).append_immutable(
        paths.replacements / f"{planned.run_id}.json",
        payload,
    )


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid P3R artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"P3R artifact is not an object: {path}")
    return value


__all__ = [
    "completed_run_ids",
    "freeze_manifest",
    "load_manifest",
    "load_run",
    "load_runs",
    "load_replacements",
    "store_for",
    "write_run",
    "write_replacement",
]
