"""Versioned immutable contracts for the P3R live paired campaign."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest, to_primitive

SCHEMA_VERSION = 5
BENCHMARK_VERSION = "p3r-live-experience-reuse-v5"
LEGACY_BENCHMARK_VERSIONS = {
    "p3r-live-experience-reuse-v1",
    "p3r-live-experience-reuse-v2",
    "p3r-live-experience-reuse-v3",
    "p3r-live-experience-reuse-v4",
}
MANIFEST_SCHEMA = "pico.picobench.p3r-campaign.v1"
RUN_RECORD_SCHEMA = "pico.picobench.p3r-run.v1"


class CampaignMode(StrEnum):
    PILOT = "pilot"
    OFFICIAL_SINGLE = "official-single"
    OFFICIAL_REPEAT2 = "official-repeat2"
    P3R3_EXPLORATORY = "p3r3-exploratory"


class Arm(StrEnum):
    NO_REUSE = "no_reuse"
    APPROVED_REUSE = "approved_reuse"
    APPROVED_REUSE_LEGACY = "approved_reuse_legacy"
    APPROVED_REUSE_SELECTIVE = "approved_reuse_selective"


class BenefitClassification(StrEnum):
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    REGRESSIVE = "regressive"
    INVALID = "invalid"
    NOT_EVALUATED = "not_evaluated"


class Availability(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    NOT_AVAILABLE = "not_available"


class TokenAccountingStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_AVAILABLE = "not_available"


class RunValidity(StrEnum):
    VALID = "valid"
    INFRA_INVALID = "infra_invalid"


class InfraInvalidReason(StrEnum):
    PROVIDER_TRANSPORT = "provider_transport"
    PROVIDER_SERVER = "provider_server"
    PROVIDER_MALFORMED_RESPONSE = "provider_malformed_response"
    BENCHMARK_HOST_FAILURE = "benchmark_host_failure"
    WORKTREE_SETUP_FAILURE = "worktree_setup_failure"
    VERIFIER_HOST_FAILURE = "verifier_host_failure"
    MANDATORY_EVIDENCE_FAILURE = "mandatory_evidence_failure"


@dataclass(frozen=True)
class RuntimeBudget:
    max_agent_iterations: int = 12
    provider_logical_calls_observational: int | None = None
    tool_calls_observational: int | None = None
    final_synthesis_call_allowed: bool = True
    wall_clock_timeout_seconds: int = 900
    context_window_tokens: int = 98_304
    retry_policy: str = "configured_runtime_bounded"

    @property
    def max_agent_steps(self) -> int:
        """Compatibility alias; the enforced control is Agent iterations."""

        return self.max_agent_iterations


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
    replacement_for_run_id: str | None = None
    replacement_ordinal: int = 0


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
    pilot_repetition: int = 1
    replication_questions: tuple[str, ...] = ()
    validity_policy_version: int = 1
    task_contract_version: int = 2
    suite_version: str = ""
    mechanical_solvability_digests: tuple[tuple[str, str], ...] = ()
    selector_mode: str = ""
    selector_version: int = 0
    selector_config_digest: str = ""
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
        if self.schema_version == 1:
            payload.pop("validity_policy_version", None)
            payload.pop("task_contract_version", None)
            budget = payload["budget"]
            payload["budget"] = {
                "max_agent_steps": budget["max_agent_iterations"],
                "max_provider_logical_calls": budget["provider_logical_calls_observational"],
                "max_tool_calls": budget["tool_calls_observational"],
                "wall_clock_timeout_seconds": budget["wall_clock_timeout_seconds"],
                "context_window_tokens": budget["context_window_tokens"],
                "retry_policy": budget["retry_policy"],
            }
            for planned in payload["planned_runs"]:
                planned.pop("replacement_for_run_id", None)
                planned.pop("replacement_ordinal", None)
        if self.schema_version < 3:
            payload.pop("pilot_repetition", None)
            payload.pop("replication_questions", None)
        if self.schema_version < 5:
            payload.pop("suite_version", None)
            payload.pop("mechanical_solvability_digests", None)
            payload.pop("selector_mode", None)
            payload.pop("selector_version", None)
            payload.pop("selector_config_digest", None)
        return payload

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA or self.schema_version not in {1, 2, 3, 4, SCHEMA_VERSION}:
            raise ValueError("unsupported P3R campaign manifest schema")
        if self.manifest_digest != canonical_digest(self._payload()):
            raise ValueError("P3R campaign manifest digest mismatch")
        if len(self.base_commit_sha) != 40:
            raise ValueError("base_commit_sha must be a full Git SHA")
        if self.benchmark_version not in {*LEGACY_BENCHMARK_VERSIONS, BENCHMARK_VERSION}:
            raise ValueError("unsupported P3R benchmark version")
        if self.pilot_repetition < 1:
            raise ValueError("pilot_repetition must be positive")
        if self.schema_version >= 5 and self.mode in {
            CampaignMode.OFFICIAL_SINGLE,
            CampaignMode.OFFICIAL_REPEAT2,
        }:
            if not self.suite_version or len(self.mechanical_solvability_digests) != 12:
                raise ValueError("official P3R manifest lacks frozen suite evidence")
            if not self.selector_mode or self.selector_version < 1:
                raise ValueError("official P3R manifest lacks selector identity")
            if len(self.selector_config_digest) != 64:
                raise ValueError("official P3R selector config digest is invalid")
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
    provider_failed_attempts: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    known_input_tokens_sum: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    known_output_tokens_sum: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    known_cached_tokens_sum: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    attempts_with_usage: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    attempts_without_usage: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    token_accounting_status: TokenAccountingStatus = TokenAccountingStatus.NOT_AVAILABLE
    agent_iterations: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    first_edit_iteration: MetricValue = field(default_factory=lambda: MetricValue(None, Availability.NOT_AVAILABLE))
    iteration_exhausted: bool = False
    final_synthesis_call_present: bool = False


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
    run_validity: RunValidity = RunValidity.VALID
    infra_invalid_reason: InfraInvalidReason | None = None
    normalized_provider_failure_categories: tuple[str, ...] = ()
    recovered_provider_failure_count: int = 0
    terminal_provider_failure: bool = False
    verifier_findings: tuple[str, ...] = ()
    replacement_for_run_id: str | None = None
    repository_read_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    relevance_selection_refs: tuple[str, ...] = ()
    relevance_selected_candidate_ids: tuple[str, ...] = ()
    relevance_abstained_candidate_ids: tuple[str, ...] = ()
    relevance_abstention_reason_counts: tuple[tuple[str, int], ...] = ()
    integrity_digest: str = ""
    schema: str = RUN_RECORD_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> RunRecord:
        value = cls(**values, integrity_digest="")
        return replace(value, integrity_digest=canonical_digest(value._payload()))

    def _payload(self) -> dict[str, Any]:
        payload = to_primitive(self)
        payload.pop("integrity_digest", None)
        if self.schema_version < 3:
            payload.pop("repository_read_paths", None)
            payload.pop("changed_paths", None)
            payload["metrics"].pop("first_edit_iteration", None)
        if self.schema_version < 4:
            payload.pop("relevance_selection_refs", None)
            payload.pop("relevance_selected_candidate_ids", None)
            payload.pop("relevance_abstained_candidate_ids", None)
            payload.pop("relevance_abstention_reason_counts", None)
        if self.schema_version == 1:
            for key in (
                "run_validity",
                "infra_invalid_reason",
                "normalized_provider_failure_categories",
                "recovered_provider_failure_count",
                "terminal_provider_failure",
                "verifier_findings",
                "replacement_for_run_id",
            ):
                payload.pop(key, None)
            for key in (
                "provider_failed_attempts",
                "known_input_tokens_sum",
                "known_output_tokens_sum",
                "known_cached_tokens_sum",
                "attempts_with_usage",
                "attempts_without_usage",
                "token_accounting_status",
                "agent_iterations",
                "iteration_exhausted",
                "final_synthesis_call_present",
            ):
                payload["metrics"].pop(key, None)
        return payload

    def validate(self) -> None:
        digest = self.integrity_digest
        if self.schema != RUN_RECORD_SCHEMA or self.schema_version not in {1, 2, 3, 4, SCHEMA_VERSION}:
            raise ValueError("unsupported P3R run-record schema")
        if digest != canonical_digest(self._payload()):
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
    replacements: Path

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
            replacements=root / "replacements",
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
    "InfraInvalidReason",
    "LiveTask",
    "MetricValue",
    "PlannedRun",
    "RunMetrics",
    "RunRecord",
    "RunValidity",
    "RuntimeBudget",
    "TokenAccountingStatus",
]
