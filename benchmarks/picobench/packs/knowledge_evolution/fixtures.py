"""Frozen, non-secret P3 evaluation scenario manifest."""

from __future__ import annotations

from dataclasses import dataclass

from pico.knowledge_evolution import CandidateType, ContentClass

from .schema import EvaluationArm


@dataclass(frozen=True)
class KnowledgeScenario:
    scenario_id: str
    task_id: str
    arm: EvaluationArm
    candidate_type: CandidateType | None
    content_class: ContentClass | None
    title: str
    reusable_content: str
    query: str


SCENARIOS = (
    KnowledgeScenario(
        "no-reuse-known-config",
        "locate-repository-test-command",
        EvaluationArm.NO_REUSE,
        CandidateType.MEMORY_FACT,
        ContentClass.FACT,
        "",
        "",
        "locate the repository test command",
    ),
    KnowledgeScenario(
        "no-reuse-schema-recovery",
        "recover-schema-validation",
        EvaluationArm.NO_REUSE,
        CandidateType.EXPERIENCE,
        ContentClass.RECOVERY,
        "",
        "",
        "recover from schema validation failure",
    ),
    KnowledgeScenario(
        "no-reuse-verification-skill",
        "follow-repository-verification-procedure",
        EvaluationArm.NO_REUSE,
        CandidateType.SKILL_CANDIDATE,
        ContentClass.PROCEDURE,
        "",
        "",
        "use c5a to follow the repository verification procedure",
    ),
    KnowledgeScenario(
        "approved-memory-fact",
        "locate-repository-test-command",
        EvaluationArm.APPROVED_REUSE,
        CandidateType.MEMORY_FACT,
        ContentClass.FACT,
        "Repository test command location",
        "The deterministic test entry point is scripts/run_tests.py.",
        "locate the repository test command",
    ),
    KnowledgeScenario(
        "approved-experience",
        "recover-schema-validation",
        EvaluationArm.APPROVED_REUSE,
        CandidateType.EXPERIENCE,
        ContentClass.RECOVERY,
        "Recover from schema validation",
        "After schema validation fails, correct the arguments and issue a fresh Tool intent.",
        "recover from schema validation failure",
    ),
    KnowledgeScenario(
        "approved-skill",
        "follow-repository-verification-procedure",
        EvaluationArm.APPROVED_REUSE,
        CandidateType.SKILL_CANDIDATE,
        ContentClass.PROCEDURE,
        "Repository verification procedure",
        "Run the scoped deterministic checks.\n\nValidation expectations:\n- Checks pass",
        "use c5a to follow the repository verification procedure",
    ),
    KnowledgeScenario(
        "stale-memory-fact",
        "locate-repository-test-command",
        EvaluationArm.STALE_REUSE,
        CandidateType.MEMORY_FACT,
        ContentClass.FACT,
        "Repository test command location",
        "The deterministic test entry point is scripts/run_tests.py.",
        "locate the repository test command",
    ),
    KnowledgeScenario(
        "cross-repository-fact",
        "locate-repository-test-command",
        EvaluationArm.CROSS_REPOSITORY,
        CandidateType.MEMORY_FACT,
        ContentClass.FACT,
        "Repository test command location",
        "The deterministic test entry point is scripts/run_tests.py.",
        "locate the repository test command",
    ),
    KnowledgeScenario(
        "rejected-experience",
        "recover-schema-validation",
        EvaluationArm.REJECTED_CANDIDATE,
        CandidateType.EXPERIENCE,
        ContentClass.RECOVERY,
        "Recover from schema validation",
        "After schema validation fails, correct the arguments and issue a fresh Tool intent.",
        "recover from schema validation failure",
    ),
    KnowledgeScenario(
        "unreviewed-skill",
        "follow-repository-verification-procedure",
        EvaluationArm.REJECTED_CANDIDATE,
        CandidateType.SKILL_CANDIDATE,
        ContentClass.PROCEDURE,
        "Repository verification procedure",
        "Run the scoped deterministic checks.\n\nValidation expectations:\n- Checks pass",
        "follow the repository verification procedure",
    ),
    KnowledgeScenario(
        "unresolved-conflicting-facts",
        "select-repository-package-manager",
        EvaluationArm.CONFLICT,
        CandidateType.MEMORY_FACT,
        ContentClass.FACT,
        "Repository package manager",
        "The repository uses uv.",
        "select the repository package manager",
    ),
)


__all__ = ["KnowledgeScenario", "SCENARIOS"]
