"""Narrow Runtime adapters for trusted P3 context and Skill reuse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pico.context_engine.base import AssemblyContext, Segment
from pico.tracing import evidence, spans

from .applicability import ApplicabilityEnvironment
from .relevance import KnowledgeSelectionMode
from .retrieval import KnowledgeRetriever, RetrievedKnowledge
from .store import KnowledgeRecordStore
from .types import CandidateType, structural_digest
from .usage import KnowledgeUsageMode, KnowledgeUsageReceipt, persist_usage

if TYPE_CHECKING:
    from .utility import KnowledgeUtilityCoordinator


def _usage_receipt(
    store: KnowledgeRecordStore,
    item: RetrievedKnowledge,
    *,
    retrieval_id: str,
    turn_id: str,
    mode: KnowledgeUsageMode,
    context_identity: str,
) -> KnowledgeUsageReceipt:
    materialization_digest = item.applicability.materialization_digest
    if materialization_digest is None:
        raise ValueError("applicable knowledge lacks materialization evidence")
    recorder = evidence.current()
    usage_id = (
        "usage-" + structural_digest({"identity": recorder.next_identity("knowledge_usage")})
        if recorder is not None and recorder.turn_id == turn_id
        else "usage-"
        + structural_digest(
            {"turn_id": turn_id, "candidate_id": item.candidate.candidate_id, "mode": mode.value}
        )
    )
    receipt = KnowledgeUsageReceipt.create(
        usage_id=usage_id,
        turn_id=turn_id,
        repository_scope_id=item.candidate.repository_scope_id,
        candidate_id=item.candidate.candidate_id,
        candidate_manifest_digest=item.candidate.manifest_digest,
        materialization_digest=materialization_digest,
        knowledge_type=item.candidate.candidate_type,
        lifecycle_state=item.applicability.lifecycle_state.value,
        applicability_id=item.applicability.applicability_id,
        applicability_digest=item.applicability.applicability_digest,
        retrieval_id=retrieval_id,
        retrieval_rank=item.rank,
        retrieval_score=item.score,
        usage_mode=mode,
        context_identity=context_identity,
        created_at=spans.now_iso(),
    )
    persist_usage(store, receipt)
    return receipt


class KnowledgeContextSegmentBuilder:
    """Inject bounded applicable Facts and Experiences; fail isolated to empty."""

    name = "trusted_repository_knowledge"
    order = 4
    needs_prefix = False

    def __init__(
        self,
        store: KnowledgeRecordStore,
        environment_factory: Callable[[], ApplicabilityEnvironment],
        *,
        max_facts: int = 3,
        max_experiences: int = 3,
        max_tokens: int = 900,
        selection_mode: KnowledgeSelectionMode = KnowledgeSelectionMode.LEGACY_APPLICABLE,
        utility_coordinator: KnowledgeUtilityCoordinator | None = None,
    ) -> None:
        self._store = store
        self._environment_factory = environment_factory
        self._max_facts = max(0, max_facts)
        self._max_experiences = max(0, max_experiences)
        self._max_tokens = max(0, max_tokens)
        self._selection_mode = selection_mode
        self._utility_coordinator = utility_coordinator

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        recorder = evidence.current()
        turn_id = recorder.turn_id if recorder is not None else None
        retrieval_id = (
            "retrieval-"
            + structural_digest({"identity": recorder.next_identity("knowledge_retrieval")})
            if recorder is not None
            else f"diagnostic-{self._environment_factory().repository_scope_id}"
        )
        try:
            retriever = KnowledgeRetriever(
                self._store,
                self._environment_factory(),
                max_candidates=self._max_facts + self._max_experiences,
                selection_mode=self._selection_mode,
            )
            items, receipt = retriever.retrieve(
                ctx.current_message,
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                candidate_types=(CandidateType.MEMORY_FACT, CandidateType.EXPERIENCE),
            )
            refinement = None
            if self._utility_coordinator is not None:
                refinement = await self._utility_coordinator.refine(
                    items,
                    query=ctx.current_message,
                    turn_id=turn_id,
                    repository_scope_id=receipt.repository_scope_id,
                    group_id=receipt.retrieval_id,
                )
                items = refinement.items
        except Exception as exc:  # noqa: BLE001 -- optional knowledge must fail isolated
            return Segment(
                text="",
                meta={
                    "p3_knowledge_failure": type(exc).__name__,
                    "p3_injected_candidate_ids": [],
                },
            )

        facts = [item for item in items if item.candidate.candidate_type is CandidateType.MEMORY_FACT][
            : self._max_facts
        ]
        experiences = [
            item for item in items if item.candidate.candidate_type is CandidateType.EXPERIENCE
        ][: self._max_experiences]
        remaining = min(self._max_tokens, max(0, ctx.budget.available_history)) * 4
        sections: list[str] = []
        injected: list[RetrievedKnowledge] = []
        for heading, values, label in (
            ("Repository Facts", facts, "Fact"),
            ("Experience / Lessons", experiences, "Guidance"),
        ):
            bullets: list[str] = []
            for item in values:
                text = f"- {label}: {item.candidate.title} — {item.candidate.reusable_content}"
                if len(text) > remaining:
                    continue
                bullets.append(text)
                injected.append(item)
                remaining -= len(text)
            if bullets:
                sections.append(f"## {heading}\n\n" + "\n".join(bullets))
        if turn_id is not None:
            for item in injected:
                _usage_receipt(
                    self._store,
                    item,
                    retrieval_id=receipt.retrieval_id,
                    turn_id=turn_id,
                    mode=KnowledgeUsageMode.INJECTED,
                    context_identity="context:p3-trusted-knowledge",
                )
        return Segment(
            text=("# Trusted Repository Knowledge\n\n" + "\n\n".join(sections) if sections else ""),
            meta={
                "p3_retrieval_id": receipt.retrieval_id,
                "p3_selection_mode": self._selection_mode.value,
                "p3_retrieved_candidate_ids": list(receipt.selected_candidate_ids),
                "p3_injected_candidate_ids": [item.candidate.candidate_id for item in injected],
                "p3_suppressed_count": receipt.suppressed_count,
                "p3_utility_kept_candidate_ids": (
                    list(refinement.retained_candidate_ids) if refinement is not None else []
                ),
                "p3_utility_abstained_candidate_ids": (
                    list(refinement.abstained_candidate_ids) if refinement is not None else []
                ),
                "p3_utility_fallback_used": (
                    refinement.fallback_used if refinement is not None else False
                ),
            },
        )


def persist_skill_hit_usage(hit: Any, mode: KnowledgeUsageMode) -> None:
    candidate_id = hit.meta.get("p3_candidate_id")
    recorder = evidence.current()
    if not candidate_id or recorder is None:
        return
    store = KnowledgeRecordStore(Path(str(hit.meta["p3_state_root"])))
    receipt = KnowledgeUsageReceipt.create(
        usage_id="usage-"
        + structural_digest({"identity": recorder.next_identity("knowledge_usage")}),
        turn_id=recorder.turn_id,
        repository_scope_id=str(hit.meta["p3_repository_scope_id"]),
        candidate_id=str(candidate_id),
        candidate_manifest_digest=str(hit.meta["p3_candidate_manifest_digest"]),
        materialization_digest=str(hit.meta["p3_materialization_digest"]),
        knowledge_type=CandidateType.SKILL_CANDIDATE,
        lifecycle_state="active",
        applicability_id=str(hit.meta["p3_applicability_id"]),
        applicability_digest=str(hit.meta["p3_applicability_digest"]),
        retrieval_id=str(hit.meta["p3_retrieval_id"]),
        retrieval_rank=int(hit.meta["p3_retrieval_rank"]),
        retrieval_score=float(hit.meta["p3_retrieval_score"]),
        usage_mode=mode,
        context_identity=f"skill:{hit.qualified_id}",
        created_at=spans.now_iso(),
    )
    persist_usage(store, receipt)
__all__ = [
    "KnowledgeContextSegmentBuilder",
    "persist_skill_hit_usage",
]
