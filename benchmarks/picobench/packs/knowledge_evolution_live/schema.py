"""Versioned immutable contracts for the P3R live paired campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest, to_primitive

SCHEMA_VERSION = 1
BENCHMARK_VERSION = "p3r-live-experience-reuse-v1"
MANIFEST_SCHEMA = "pico.picobench.p3r-campaign.v1"
RUN_RECORD_SCHEMA = "pico.picobench.p3r-run.v1"


class CampaignMode(StrEnum):
    PILOT = "pilot"
    OFFICIAL_SINGLE = "official-single"
    OFFICIAL_REPEAT2 = "official-repeat2"


class Arm(StrEnum):
    NO_REUSE = "no_reuse"
    APPROVED_REUSE = "approved_reuse"


class BenefitClassification(StrEnum):
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    REGRESSIVE = "regressive"
    INVALID = "invalid"
    NOT_EVALUATED = "not_evaluated"


class Availability(StrEnum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class RuntimeBudget:
    max_agent_steps: int = 12
    max_provider_logical_calls: int = 12
    max_tool_calls: int = 40
    wall_clock_timeout_seconds: int = 900
    context_window_tokens: int = 98_304
    retry_policy: str = "configured_runtime_bounded"


@dataclass(frozen=True)
class LiveTask:
    task_id: str
    category: str
    prompt: str
    verifier_id: str
    verifier_digest: str
    expected_knowledge_classes: tuple[str, ...]
    rationale: str
    tool_profile: str = "normal_workspace_tools"
    setup_id: str | None = None

    @property
    def prompt_digest(self) -> str:
        return canonical_digest({"task_id": self.task_id, "prompt": self.prompt})


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    task_id: str
    arm: Arm
    repetition: int
    order: int


@dataclass(frozen=True)
class CampaignManifest:
    campaign_id: str
    mode: CampaignMode
    base_commit_sha: str
    campaign_seed: int
    provider_id: str
    actual_model_id: str
    provider_model_config_digest: str
    tool_config_digest: str
    runtime_config_digest: str
    budget: RuntimeBudget
    task_ids: tuple[str, ...]
    task_prompt_digests: tuple[tuple[str, str], ...]
    verifier_digests: tuple[tuple[str, str], ...]
    knowledge_corpus_digest: str
    planned_runs: tuple[PlannedRun, ...]
    created_at: str
    benchmark_version: str = BENCHMARK_VERSION
    manifest_digest: str = ""
    schema: str = MANIFEST_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> CampaignManifest:
        value = cls(**values, manifest_digest="")
        return replace(value, manifest_digest=canonical_digest(value._payload()))

    def _payload(self) -> dict[str, Any]:
        payload = to_primitive(self)
        payload.pop("manifest_digest", None)
        return payload

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported P3R campaign manifest schema")
        if self.manifest_digest != canonical_digest(self._payload()):
            raise ValueError("P3R campaign manifest digest mismatch")
        if len(self.base_commit_sha) != 40:
            raise ValueError("base_commit_sha must be a full Git SHA")
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError("unsupported P3R benchmark version")
        run_ids = tuple(item.run_id for item in self.planned_runs)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("P3R planned run IDs must be unique")


@dataclass(frozen=True)
class MetricValue:
    value: int | float | None
    availability: Availability


@dataclass(frozen=True)
class RunMetrics:
    provider_logical_calls: MetricValue
    provider_attempts: MetricValue
    tool_calls_total: MetricValue
    unique_repo_files_read: MetricValue
    total_repo_file_reads: MetricValue
    repeated_repo_file_reads: MetricValue
    skill_candidates_retrieved: MetricValue
    skills_referenced: MetricValue
    skills_activated: MetricValue
    repeated_skill_reads: MetricValue
    input_tokens: MetricValue
    output_tokens: MetricValue
    cached_tokens: MetricValue
    turn_latency_ms: MetricValue
    provider_retries: MetricValue
    failed_tool_attempts: MetricValue
    validation_failures: MetricValue
    p3_approximate_context_tokens: MetricValue


@dataclass(frozen=True)
class RunRecord:
    campaign_id: str
    task_id: str
    arm: Arm
    repetition: int
    run_id: str
    workspace_identity: str
    turn_id: str
    provider_id: str
    actual_model_id: str
    started_at: str
    terminal_at: str
    runtime_outcome: str
    verifier_outcome: str
    safety_outcome: str
    safety_findings: tuple[str, ...]
    task_success_evidence_ref: str | None
    knowledge_retrieval_refs: tuple[str, ...]
    knowledge_usage_refs: tuple[str, ...]
    outcome_association_refs: tuple[str, ...]
    retrieved_candidate_ids: tuple[str, ...]
    injected_candidate_ids: tuple[str, ...]
    referenced_candidate_ids: tuple[str, ...]
    activated_candidate_ids: tuple[str, ...]
    metrics: RunMetrics
    patch_digest: str
    patch_artifact_ref: str
    trace_evidence_refs: tuple[str, ...]
    fairness_digest: str
    integrity_digest: str = ""
    schema: str = RUN_RECORD_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> RunRecord:
        value = cls(**values, integrity_digest="")
        payload = to_primitive(value)
        payload.pop("integrity_digest", None)
        return replace(value, integrity_digest=canonical_digest(payload))

    def validate(self) -> None:
        payload = to_primitive(self)
        digest = payload.pop("integrity_digest", None)
        if self.schema != RUN_RECORD_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported P3R run-record schema")
        if digest != canonical_digest(payload):
            raise ValueError("P3R run-record integrity mismatch")


@dataclass(frozen=True)
class CampaignPaths:
    root: Path
    manifest: Path
    runs: Path
    worktrees: Path
    states: Path
    patches: Path
    reduced: Path

    @classmethod
    def at(cls, root: Path) -> CampaignPaths:
        root = Path(root)
        return cls(
            root=root,
            manifest=root / "campaign.json",
            runs=root / "runs",
            worktrees=root / "w",
            states=root / "s",
            patches=root / "patches",
            reduced=root / "reduced.json",
        )


__all__ = [
    "BENCHMARK_VERSION",
    "MANIFEST_SCHEMA",
    "RUN_RECORD_SCHEMA",
    "Arm",
    "Availability",
    "BenefitClassification",
    "CampaignManifest",
    "CampaignMode",
    "CampaignPaths",
    "LiveTask",
    "MetricValue",
    "PlannedRun",
    "RunMetrics",
    "RunRecord",
    "RuntimeBudget",
]
