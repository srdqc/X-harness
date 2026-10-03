"""Deterministic offline P3R.3 selector comparison; never invokes a Provider."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from benchmarks.picobench.canonical import canonical_digest
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
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
    for mode in KnowledgeSelectionMode:
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
                items, _ = retriever.retrieve(
                    prompt_by_task.get(task_id, task_by_id(task_id).prompt),
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


__all__ = ["evaluate_official_selectivity", "evaluate_selectivity"]
