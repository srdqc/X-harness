"""Deterministic offline P3R.3 selector comparison; never invokes a Provider."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateEvidenceIdentity,
    CandidateType,
    KnowledgeLifecycleManager,
    KnowledgeRecordStore,
    KnowledgeRetriever,
    KnowledgeSelectionMode,
    RepositoryScopeResolver,
)

from .knowledge import CORPUS, corpus_digest, prepare_approved_corpus
from .schema import LiveTask
from .tasks import OFFICIAL_TASKS, PILOT_TASK_IDS, task_by_id


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _context_tokens(candidate_ids: set[str], by_corpus_id: dict[str, Any]) -> int:
    return sum(
        max(1, (len(item.title) + len(item.reusable_content) + 3) // 4)
        for corpus_id, item in by_corpus_id.items()
        if corpus_id in candidate_ids
    )


def evaluate_selectivity(
    *,
    state_root: Path,
    workspace: Path,
    reviewer_id: str = "human:offline",
    task_ids: tuple[str, ...] = PILOT_TASK_IDS,
    tasks: tuple[LiveTask, ...] | None = None,
) -> dict[str, Any]:
    """Compare modes with identical corpus, applicability inputs, and limits."""

    candidate_ids = prepare_approved_corpus(
        state_root=state_root,
        workspace=workspace,
        reviewer_id=reviewer_id,
    )
    scope = RepositoryScopeResolver(workspace, state_root).resolve().identity
    if scope is None:
        raise RuntimeError("offline P3R.3 evaluation requires repository scope")
    store = KnowledgeRecordStore(state_root)
    corpus_id_by_candidate = dict(zip(candidate_ids, (item.corpus_id for item in CORPUS), strict=True))
    proposal_by_corpus = {item.corpus_id: item.proposal for item in CORPUS}
    prompt_by_task = {task.task_id: task.prompt for task in tasks or ()}
    modes: dict[str, Any] = {}
    # Historical P3R selector audits compare only the frozen deterministic
    # baseline modes. Jev utility is a later additive experiment, not a third
    # selector implementation in these immutable reports.
    for mode in (
        KnowledgeSelectionMode.LEGACY_APPLICABLE,
        KnowledgeSelectionMode.TASK_RELEVANCE_V1,
    ):
        task_results: dict[str, Any] = {}
        for task_id in task_ids:
            selected: set[str] = set()
            for index, candidate_types in enumerate(
                (
                    (CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
                    (CandidateType.SKILL_CANDIDATE,),
                )
            ):
                retriever = KnowledgeRetriever(
                    store,
                    ApplicabilityEnvironment(
                        scope.repository_scope_id,
                        workspace,
                        available_tools=("read_file",),
                    ),
                    max_candidates=len(CORPUS),
                    selection_mode=mode,
                )
                prompt = (
                    prompt_by_task[task_id]
                    if task_id in prompt_by_task
                    else task_by_id(task_id).prompt
                )
                items, _ = retriever.retrieve(
                    prompt,
                    retrieval_id=f"offline-{mode.value}-{task_id}-{index}",
                    turn_id=f"offline-{mode.value}-{task_id}",
                    candidate_types=candidate_types,
                    created_at="2026-10-03T00:00:00Z",
                )
                selected.update(corpus_id_by_candidate[item.candidate.candidate_id] for item in items)
            injected = {
                corpus_id
                for corpus_id in selected
                if proposal_by_corpus[corpus_id].candidate_type != CandidateType.SKILL_CANDIDATE.value
            }
            selections = store.list_relevance_selections(turn_id=f"offline-{mode.value}-{task_id}")
            task_results[task_id] = {
                "candidate_count_before_relevance": len(CORPUS),
                "selected_count": len(selected),
                "abstained_count": len(CORPUS) - len(selected),
                "selected_candidate_ids": tuple(sorted(selected)),
                "injected_candidate_ids": tuple(sorted(injected)),
                "selected_candidate_types": tuple(
                    sorted({proposal_by_corpus[corpus_id].candidate_type for corpus_id in selected})
                ),
                "abstention_reason_counts": {
                    reason: sum(item.reason.value == reason for item in selections)
                    for reason in sorted({item.reason.value for item in selections})
                    if reason.startswith("abstain_")
                },
                "approximate_context_tokens": _context_tokens(selected, proposal_by_corpus),
            }
        selected_pairs = []
        injected_pairs = []
        for left, right in combinations(task_ids, 2):
            selected_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "jaccard": _jaccard(
                        set(task_results[left]["selected_candidate_ids"]),
                        set(task_results[right]["selected_candidate_ids"]),
                    ),
                }
            )
            injected_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "jaccard": _jaccard(
                        set(task_results[left]["injected_candidate_ids"]),
                        set(task_results[right]["injected_candidate_ids"]),
                    ),
                }
            )
        modes[mode.value] = {
            "tasks": task_results,
            "selected_set_jaccard": tuple(selected_pairs),
            "injected_set_jaccard": tuple(injected_pairs),
        }
    result = {
        "schema": (
            "pico.picobench.p3r3-selectivity.v1" if task_ids == PILOT_TASK_IDS else "pico.picobench.p3r4-selectivity.v1"
        ),
        "corpus_digest": corpus_digest(),
        "selector_version": 1,
        "modes": modes,
    }
    return {**result, "semantic_digest": canonical_digest(result)}


def evaluate_official_selectivity(
    *,
    state_root: Path,
    workspace: Path,
    reviewer_id: str = "human:offline",
    tasks: tuple[LiveTask, ...] = OFFICIAL_TASKS,
) -> dict[str, Any]:
    return evaluate_selectivity(
        state_root=state_root,
        workspace=workspace,
        reviewer_id=reviewer_id,
        task_ids=tuple(task.task_id for task in tasks),
        tasks=tasks,
    )


def build_corpus_identity_catalog(
    *,
    state_root: Path,
    workspace: Path,
    reviewer_id: str = "human:offline",
) -> tuple[CandidateEvidenceIdentity, ...]:
    """Build the frozen corpus through production APIs and retain all identity layers."""

    candidate_ids = prepare_approved_corpus(
        state_root=state_root,
        workspace=workspace,
        reviewer_id=reviewer_id,
    )
    store = KnowledgeRecordStore(state_root)
    values: list[CandidateEvidenceIdentity] = []
    for item, candidate_id in zip(CORPUS, candidate_ids, strict=True):
        candidate = store.read_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("prepared corpus candidate is missing")
        lifecycle = KnowledgeLifecycleManager(store).rebuild(candidate_id)
        values.append(
            CandidateEvidenceIdentity.from_candidate(
                candidate,
                lifecycle_state=lifecycle.state,
                portable_label=item.corpus_id,
            )
        )
    return tuple(values)


def evaluate_candidate_identity_audit(
    *,
    state_root: Path,
    workspace: Path,
    tasks: tuple[LiveTask, ...],
    reviewer_id: str = "human:offline",
) -> dict[str, Any]:
    """Run TASK_RELEVANCE_V1 and persist ordered durable+semantic identities."""

    catalog = build_corpus_identity_catalog(
        state_root=state_root,
        workspace=workspace,
        reviewer_id=reviewer_id,
    )
    by_id = {item.candidate_id: item for item in catalog}
    store = KnowledgeRecordStore(state_root)
    scope = RepositoryScopeResolver(workspace, state_root).resolve().identity
    if scope is None:
        raise RuntimeError("candidate identity audit requires repository scope")
    task_results: dict[str, Any] = {}
    for task in tasks:
        selected: list[CandidateEvidenceIdentity] = []
        evidence: list[dict[str, Any]] = []
        turn_id = f"identity-audit-{task.task_id}"
        for index, candidate_types in enumerate(
            (
                (CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
                (CandidateType.SKILL_CANDIDATE,),
            )
        ):
            retrieval_id = f"identity-audit-{task.task_id}-{index}"
            retriever = KnowledgeRetriever(
                store,
                ApplicabilityEnvironment(
                    scope.repository_scope_id,
                    workspace,
                    available_tools=("read_file",),
                ),
                max_candidates=len(CORPUS),
                selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
            )
            items, _ = retriever.retrieve(
                task.prompt,
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                candidate_types=candidate_types,
                created_at="2026-10-03T00:00:00Z",
            )
            selected.extend(by_id[item.candidate.candidate_id] for item in items)
            receipts = store.list_relevance_selections(turn_id=turn_id)
            for receipt in receipts:
                if receipt.retrieval_id != retrieval_id:
                    continue
                evidence.append(
                    {
                        "candidate_id": receipt.candidate_id,
                        "portable_label": by_id[receipt.candidate_id].portable_label,
                        "semantic_digest": by_id[receipt.candidate_id].semantic_digest,
                        "rank": receipt.rank,
                        "relevance_score": receipt.relevance_score,
                        "meaningful_overlap": receipt.meaningful_overlap,
                        "identifier_overlap": receipt.identifier_overlap,
                        "decision": receipt.decision.value,
                        "reason": receipt.reason.value,
                        "selector_config_digest": receipt.selector_config_digest,
                    }
                )
        task_results[task.task_id] = {
            "ordered_selected_identities": tuple(item.to_dict() for item in selected),
            "selected_candidate_ids": tuple(item.candidate_id for item in selected),
            "selected_portable_labels": tuple(
                item.portable_label for item in selected
            ),
            "selection_evidence": tuple(evidence),
        }
    payload = {
        "schema": "pico.jev6-selector-identity-audit.v1",
        "schema_version": 1,
        "historical_corpus_digest": corpus_digest(),
        "canonical_corpus_identity_digest": canonical_digest(
            tuple(item.to_dict() for item in catalog)
        ),
        "selector_version": 1,
        "tasks": task_results,
    }
    return {**payload, "semantic_digest": canonical_digest(payload)}


__all__ = [
    "build_corpus_identity_catalog",
    "evaluate_candidate_identity_audit",
    "evaluate_official_selectivity",
    "evaluate_selectivity",
]
