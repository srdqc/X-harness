"""Exactly twelve frozen, Agent-visible P3R held-out tasks."""

from __future__ import annotations

from benchmarks.picobench.canonical import canonical_digest

from .schema import LiveTask
from .verifiers import VERIFIERS


def _task(task_id: str, category: str, prompt: str, verifier_id: str, knowledge: tuple[str, ...], rationale: str) -> LiveTask:
    return LiveTask(
        task_id=task_id,
        category=category,
        prompt=prompt,
        verifier_id=verifier_id,
        verifier_digest=VERIFIERS[verifier_id].digest,
        expected_knowledge_classes=knowledge,
        rationale=rationale,
    )


TASKS = (
    _task("p3r-nav-01", "repository_navigation", "Add a deterministic, optionally Turn-filtered retrieval-receipt listing API to the authoritative P3 store. Follow existing immutable-record and ordering conventions and add focused tests.", "p3r-v-nav-retrieval-list", ("memory_fact",), "Requires locating store serialization, retrieval contracts, and ordering conventions rather than a single obvious edit."),
    _task("p3r-nav-02", "repository_navigation", "Add a deterministic outcome-association listing API to the P3 store with optional Turn filtering, corruption-safe reads, and focused tests consistent with existing usage APIs.", "p3r-v-nav-association-list", ("memory_fact",), "Exercises repository navigation across store and usage evidence without exposing a final patch."),
    _task("p3r-nav-03", "repository_navigation", "Register a new narrow phase-level test suite named p3r_example that composes existing focused targets without duplicating target paths, and add deterministic runner tests for resolution and deduplication.", "p3r-v-nav-campaign-matrix", ("skill_candidate",), "Requires understanding the canonical test matrix and runner composition conventions."),
    _task("p3r-impl-01", "implementation_extension", "Extend KnowledgeRetrievalReceipt with a compact deterministic selection summary suitable for reducers, preserving serialization compatibility and adding round-trip tests.", "p3r-v-impl-retrieval-summary", ("memory_fact", "experience"), "Generalizes the immutable receipt pattern without embedding benchmark answers."),
    _task("p3r-impl-02", "implementation_extension", "Add a small immutable aggregate for knowledge-usage dispositions that can be computed from receipt objects without reading Agent prose. Export it through the existing package boundary and add focused tests.", "p3r-v-impl-usage-summary", ("memory_fact", "experience"), "Requires following established dataclass, enum, digest, and export conventions."),
    _task("p3r-impl-03", "implementation_extension", "Add a validated reader for the deterministic P3 PicoBench report that rejects schema or semantic-digest mismatches. Preserve existing report writing and add corruption tests.", "p3r-v-impl-report-reader", ("experience",), "Exercises benchmark artifact conventions and deterministic integrity validation."),
    _task("p3r-debug-01", "debugging_recovery", "Diagnose why a machine-checkable applicability predicate produced by extraction can remain inconclusive at Runtime. Implement the smallest compatible fix and regression tests without weakening fail-closed behavior.", "p3r-v-debug-guard", ("experience",), "Tests recovery across extraction and applicability boundaries without disclosing exact code changes."),
    _task("p3r-debug-02", "debugging_recovery", "Harden usage-to-outcome association against duplicate usage IDs while preserving immutable evidence and same-Turn validation. Add focused positive and negative tests.", "p3r-v-debug-association", ("experience",), "Requires tracing an integrity edge case through durable association semantics."),
    _task("p3r-debug-03", "debugging_recovery", "Make the deterministic P3 benchmark robust to short Windows test roots without changing scenario identity or semantic digest. Add a regression test that distinguishes logical identity from physical paths.", "p3r-v-debug-path", ("experience",), "Exercises a recurring portability failure and requires preserving semantic evidence."),
    _task("p3r-int-01", "cross_module_integration", "Inspect the multi-source scoped Skill path and prove applicable P3 Skills still flow through SkillRegistry, Router, Resolver, and Skills context when a second ordinary local Skill source is present. Fix a genuine production wiring defect if one exists; otherwise add deterministic integration proof without changing production code.", "p3r-v-integration-skill", ("skill_candidate",), "Requires coordinating P3 applicability with the existing multi-source Skill path while permitting a proof-only result when the wiring is already correct."),
    _task("p3r-int-02", "cross_module_integration", "Extend replay-derived facts with bounded P3 knowledge-usage counts sourced from durable evidence, preserving legacy replay digests and verifier independence. Add compatibility tests.", "p3r-v-integration-trace", ("memory_fact", "experience"), "Crosses tracing and knowledge evidence while retaining authority separation."),
    _task("p3r-int-03", "cross_module_integration", "Wire the deterministic knowledge-evolution PicoBench test into the closest reusable PicoBench suite and prove phase resolution remains deduplicated and stable.", "p3r-v-integration-suite", ("skill_candidate",), "Requires test-infrastructure and benchmark integration rather than a standalone file edit."),
)

PILOT_TASK_IDS = ("p3r-nav-01", "p3r-debug-01", "p3r-int-01")


def task_set_digest() -> str:
    return canonical_digest(TASKS)


def task_by_id(task_id: str) -> LiveTask:
    return next(task for task in TASKS if task.task_id == task_id)


__all__ = ["PILOT_TASK_IDS", "TASKS", "task_by_id", "task_set_digest"]
