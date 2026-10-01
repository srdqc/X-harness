"""Rebuildable, repository-scoped candidate comparison index."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pico.utils.bm25 import BM25Okapi, tokenize

from .types import KnowledgeCandidate

if TYPE_CHECKING:
    from .store import KnowledgeRecordStore


class CandidateRelation(str, Enum):
    NEW = "new"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class CandidateComparison:
    relation: CandidateRelation
    candidate_id: str
    existing_candidate_ids: tuple[str, ...]
    compared_candidate_count: int
    max_bm25_score: float | None = None
    max_lexical_similarity: float | None = None


def _search_text(candidate: KnowledgeCandidate) -> str:
    return "\n".join((candidate.title, candidate.reusable_content, *candidate.preconditions))


def _meaning_tokens(candidate: KnowledgeCandidate) -> set[str]:
    return set(tokenize("\n".join((candidate.reusable_content, *candidate.preconditions))))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _fact_map(candidate: KnowledgeCandidate) -> dict[str, str]:
    return {
        key: digest
        for key, digest in candidate.applicability_fingerprints
        if key.startswith("fact:")
    }


class CandidateIndex:
    """Non-authoritative in-memory projection rebuilt from immutable candidates."""

    def __init__(self, candidates: tuple[KnowledgeCandidate, ...], *, top_k: int = 5) -> None:
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        self._candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        self.top_k = top_k

    @classmethod
    def rebuild(
        cls,
        store: KnowledgeRecordStore,
        *,
        repository_scope_id: str,
        top_k: int = 5,
    ) -> CandidateIndex:
        return cls(store.list_candidates(repository_scope_id=repository_scope_id), top_k=top_k)

    def compare(self, candidate: KnowledgeCandidate) -> CandidateComparison:
        compatible = tuple(
            item
            for item in self._candidates
            if item.repository_scope_id == candidate.repository_scope_id
            and item.candidate_type is candidate.candidate_type
            and item.content_class is candidate.content_class
        )
        exact = tuple(
            item.candidate_id
            for item in compatible
            if item.content_fingerprint == candidate.content_fingerprint
        )
        if exact:
            return CandidateComparison(
                CandidateRelation.EXACT_DUPLICATE,
                candidate.candidate_id,
                exact,
                len(compatible),
            )

        candidate_facts = _fact_map(candidate)
        conflicts = tuple(
            item.candidate_id
            for item in compatible
            if any(
                key in _fact_map(item) and _fact_map(item)[key] != value
                for key, value in candidate_facts.items()
            )
        )
        if conflicts:
            return CandidateComparison(
                CandidateRelation.CONFLICT,
                candidate.candidate_id,
                conflicts,
                len(compatible),
            )

        if not compatible:
            return CandidateComparison(CandidateRelation.NEW, candidate.candidate_id, (), 0)

        corpus = [tokenize(_search_text(item)) for item in compatible]
        scores = BM25Okapi(corpus).get_scores(tokenize(_search_text(candidate)))
        ranked = sorted(
            range(len(compatible)),
            key=lambda index: (-scores[index], compatible[index].candidate_id),
        )[: self.top_k]
        meaning = _meaning_tokens(candidate)
        near: list[tuple[str, float, float]] = []
        for index in ranked:
            similarity = _jaccard(meaning, _meaning_tokens(compatible[index]))
            if scores[index] > 0 and similarity >= 0.45:
                near.append((compatible[index].candidate_id, scores[index], similarity))
        if near:
            return CandidateComparison(
                CandidateRelation.NEAR_DUPLICATE,
                candidate.candidate_id,
                tuple(item[0] for item in near),
                len(ranked),
                round(max(item[1] for item in near), 8),
                round(max(item[2] for item in near), 8),
            )
        return CandidateComparison(
            CandidateRelation.NEW,
            candidate.candidate_id,
            (),
            len(ranked),
            round(max((scores[index] for index in ranked), default=0.0), 8),
            round(
                max(
                    (_jaccard(meaning, _meaning_tokens(compatible[index])) for index in ranked),
                    default=0.0,
                ),
                8,
            ),
        )


__all__ = ["CandidateComparison", "CandidateIndex", "CandidateRelation"]
