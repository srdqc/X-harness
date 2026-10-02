"""Immutable local artifact I/O and versioned P3R decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import to_primitive
from benchmarks.picobench.schema import ExperimentRef

from .schema import (
    Arm,
    Availability,
    CampaignManifest,
    CampaignMode,
    CampaignPaths,
    MetricValue,
    PlannedRun,
    RunMetrics,
    RunRecord,
    RuntimeBudget,
)


def store_for(paths: CampaignPaths) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(paths.root.name, paths.root))


def freeze_manifest(paths: CampaignPaths, manifest: CampaignManifest) -> None:
    store_for(paths).append_immutable(paths.manifest, to_primitive(manifest))


def write_run(paths: CampaignPaths, record: RunRecord) -> None:
    store_for(paths).append_immutable(paths.runs / f"{record.run_id}.json", to_primitive(record))


def load_manifest(path: Path) -> CampaignManifest:
    raw = _object(path)
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
        budget=RuntimeBudget(**raw["budget"]),
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
            )
            for item in raw["planned_runs"]
        ),
        created_at=str(raw["created_at"]),
        benchmark_version=str(raw["benchmark_version"]),
        manifest_digest=str(raw["manifest_digest"]),
        schema=str(raw["schema"]),
        schema_version=int(raw["schema_version"]),
    )
    manifest.validate()
    return manifest


def load_run(path: Path) -> RunRecord:
    raw = _object(path)
    metrics = RunMetrics(
        **{
            name: MetricValue(
                value["value"],
                Availability(value["availability"]),
            )
            for name, value in raw["metrics"].items()
        }
    )
    record = RunRecord(
        **{
            **{key: value for key, value in raw.items() if key not in {"arm", "metrics"}},
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
    "store_for",
    "write_run",
]
