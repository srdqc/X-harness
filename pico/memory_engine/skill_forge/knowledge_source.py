"""SkillRegistry-backed router source for ACTIVE/APPLICABLE P3 Skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pico.knowledge_evolution.applicability import ApplicabilityEnvironment
from pico.knowledge_evolution.retrieval import KnowledgeRetriever
from pico.knowledge_evolution.store import KnowledgeRecordStore
from pico.knowledge_evolution.types import CandidateType, structural_digest
from pico.tracing import evidence

from .types import RouterHit


@dataclass
class ApplicableKnowledgeSkillSource:
    """Filter P3 eligibility before hits reach Router or optional P2 ranking."""

    store: KnowledgeRecordStore
    registry: Any
    environment_factory: Callable[[], ApplicabilityEnvironment]
    physical_source: str
    name: str = "experience"
    weight: float = 1.0

    async def search(self, query: str, history: list[dict[str, Any]], k: int) -> list[RouterHit]:
        del history
        recorder = evidence.current()
        turn_id = recorder.turn_id if recorder is not None else None
        retrieval_id = (
            "retrieval-"
            + structural_digest(
                {"identity": recorder.next_identity("knowledge_skill_retrieval")}
            )
            if recorder is not None
            else f"diagnostic-skill-{self.environment_factory().repository_scope_id}"
        )
        retriever = KnowledgeRetriever(
            self.store,
            self.environment_factory(),
            max_candidates=max(1, k),
        )
        items, receipt = retriever.retrieve(
            query,
            retrieval_id=retrieval_id,
            turn_id=turn_id,
            candidate_types=(CandidateType.SKILL_CANDIDATE,),
            top_k=k,
        )
        hits: list[RouterHit] = []
        for item in items:
            meta = self.registry.get(item.candidate.candidate_id, source=self.physical_source)
            if meta is None:
                continue
            hits.append(
                RouterHit(
                    qualified_id=f"experience/{item.candidate.candidate_id}",
                    name=item.candidate.candidate_id,
                    content=meta.content,
                    score=item.score,
                    meta={
                        "source": self.name,
                        "physical_source": self.physical_source,
                        "requirements_met": self.registry.check_available(
                            item.candidate.candidate_id,
                            source=self.physical_source,
                        ),
                        "skill_dir": str(meta.path.parent),
                        "description": meta.description,
                        "p3_state_root": str(self.store.root.parent.parent),
                        "p3_retrieval_id": receipt.retrieval_id,
                        "p3_candidate_id": item.candidate.candidate_id,
                        "p3_candidate_manifest_digest": item.candidate.manifest_digest,
                        "p3_materialization_digest": item.applicability.materialization_digest,
                        "p3_applicability_id": item.applicability.applicability_id,
                        "p3_applicability_digest": item.applicability.applicability_digest,
                        "p3_repository_scope_id": item.candidate.repository_scope_id,
                        "p3_retrieval_rank": item.rank,
                        "p3_retrieval_score": item.score,
                    },
                )
            )
        return hits


__all__ = ["ApplicableKnowledgeSkillSource"]
