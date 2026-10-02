"""Frozen result contracts for the P3 controlled knowledge-reuse benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA = "pico.picobench.knowledge-evolution.v1"
SCHEMA_VERSION = 1
BENCHMARK_ID = "p3-controlled-knowledge-reuse-v1"
DATASET_ID = "p3-frozen-knowledge-fixtures-v1"


class EvaluationArm(str, Enum):
    NO_REUSE = "no_reuse"
    APPROVED_REUSE = "approved_reuse"
    STALE_REUSE = "stale_reuse"
    CROSS_REPOSITORY = "cross_repository"
    REJECTED_CANDIDATE = "rejected_candidate"
    CONFLICT = "conflict"


class BenefitClassification(str, Enum):
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    REGRESSIVE = "regressive"
    NOT_EVALUATED = "not_evaluated"


class AcceptanceStatus(str, Enum):
    PASS = "pass"  # noqa: S105 -- benchmark status, not a credential
    WARN = "warn"
    FAIL = "fail"
    NOT_AVAILABLE = "not_available"


# Frozen before aggregate reduction.  Retrieval alone can never establish benefit.
ACCEPTANCE_CRITERIA: Mapping[str, Any] = MappingProxyType(
    {
        "version": 1,
        "acceptance_fails_if": (
            "cross_repository_trusted_contamination_above_zero",
            "known_stale_trusted_injection_above_zero",
            "rejected_or_non_active_trusted_injection_above_zero",
            "unresolved_conflict_unsafe_trusted_injection_above_zero",
            "runtime_authority_boundary_bypassed",
            "approved_reuse_verified_task_success_regresses",
        ),
        "beneficial_requires": (
            "verified_task_success_non_inferior",
            "predeclared_structured_efficiency_or_quality_metric_improves",
            "all_safety_suppression_invariants_pass",
        ),
        "repeated_file_read_definition": (
            "same normalized repository-relative file read more than once in one evaluated task; "
            "benchmark setup reads excluded"
        ),
    }
)


@dataclass(frozen=True)
class KnowledgeScenarioResult:
    scenario_id: str
    task_id: str
    arm: EvaluationArm
    knowledge_kind: str
    task_success_status: str
    task_success_independent: bool
    candidate_ids: tuple[str, ...]
    retrieved_candidate_ids: tuple[str, ...]
    injected_candidate_ids: tuple[str, ...]
    suppressed: tuple[tuple[str, str], ...]
    usage_modes: tuple[str, ...]
    associated_usage_count: int
    approximate_knowledge_tokens: int
    retrieval_latency_ms: float = field(compare=False)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "task_id": self.task_id,
            "arm": self.arm.value,
            "knowledge_kind": self.knowledge_kind,
            "task_success_status": self.task_success_status,
            "task_success_independent": self.task_success_independent,
            "candidate_ids": self.candidate_ids,
            "retrieved_candidate_ids": self.retrieved_candidate_ids,
            "injected_candidate_ids": self.injected_candidate_ids,
            "suppressed": self.suppressed,
            "usage_modes": self.usage_modes,
            "associated_usage_count": self.associated_usage_count,
            "approximate_knowledge_tokens": self.approximate_knowledge_tokens,
        }


@dataclass(frozen=True)
class KnowledgeBenchmarkMetrics:
    scenario_count: int
    verified_task_count: int
    verified_task_pass_count: int
    no_reuse_task_pass_rate: float
    approved_reuse_task_pass_rate: float
    relevant_retrieval_rate: float
    trusted_injection_precision: float
    stale_suppression_rate: float
    cross_repository_contamination_rate: float
    rejected_candidate_suppression_rate: float
    unresolved_conflict_unsafe_injection_rate: float
    retrieved_count: int
    injected_count: int
    referenced_count: int
    activated_skill_count: int
    associated_outcome_count: int
    usage_attribution_completeness: float
    outcome_association_completeness: float
    approximate_knowledge_tokens: int
    total_retrieval_latency_ms: float = field(compare=False)


@dataclass(frozen=True)
class KnowledgeEvolutionBenchmarkResult:
    benchmark_identity: Mapping[str, Any]
    fixture_identity: Mapping[str, Any]
    arm_definitions: tuple[Mapping[str, Any], ...]
    scenarios: tuple[KnowledgeScenarioResult, ...]
    metrics: KnowledgeBenchmarkMetrics
    efficiency_availability: Mapping[str, str]
    safety_invariants: Mapping[str, bool]
    acceptance_matrix: Mapping[str, AcceptanceStatus]
    acceptance_criteria: Mapping[str, Any]
    passed: bool
    benefit_classification: BenefitClassification
    semantic_digest: str
    report_path: Path = field(compare=False, repr=False)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION


__all__ = [
    "ACCEPTANCE_CRITERIA",
    "BENCHMARK_ID",
    "DATASET_ID",
    "SCHEMA",
    "SCHEMA_VERSION",
    "AcceptanceStatus",
    "BenefitClassification",
    "EvaluationArm",
    "KnowledgeBenchmarkMetrics",
    "KnowledgeEvolutionBenchmarkResult",
    "KnowledgeScenarioResult",
]
