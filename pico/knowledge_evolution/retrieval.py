"""Repository-scoped deterministic retrieval for applicable P3 knowledge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from pico.tracing import spans
from pico.utils.bm25 import tokenize

from .applicability import (
    ApplicabilityEnvironment,
    ApplicabilityStatus,
    KnowledgeApplicabilityResult,
    evaluate_applicability,
)
from .relevance import (
    KnowledgeRelevanceSelection,
    KnowledgeSelectionMode,
    RelevanceDecision,
    RelevanceReason,
    TaskRelevanceQuery,
    select_relevant_candidates,
)
from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .types import CandidateType, KnowledgeCandidate, require_digest, structural_digest

RETRIEVAL_SCHEMA = "pico.knowledge-retrieval.v1"
SCHEMA_VERSION = 1


class SuppressionReason(str, Enum):
    STALE = "stale"
    WRONG_SCOPE = "wrong_scope"
    NOT_ACTIVE = "not_active"
    SUPERSEDED = "superseded"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    MISSING_TOOL = "missing_tool"
    MISSING_BINARY = "missing_binary"
    DIGEST_MISMATCH = "digest_mismatch"
    INCONCLUSIVE_GUARD = "inconclusive_guard"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class RetrievedKnowledge:
    candidate: KnowledgeCandidate
    applicability: KnowledgeApplicabilityResult
    rank: int
    score: float


@dataclass(frozen=True)
class KnowledgeRetrievalReceipt:
    retrieval_id: str
    turn_id: str | None
    repository_scope_id: str
    query_digest: str
    candidate_set_digest: str
    ranked_candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    suppressed: tuple[tuple[str, SuppressionReason], ...]
    applicable_count: int
    suppressed_count: int
    latency_ms: float
    created_at: str
    retrieval_policy: str
    retrieval_version: int
    retrieval_digest: str
    schema: str = RETRIEVAL_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, **values: Any) -> "KnowledgeRetrievalReceipt":
        base = cls(**values, retrieval_digest="0" * 64)
        return cls(**{**base.__dict__, "retrieval_digest": structural_digest(base._payload())})

    def __post_init__(self) -> None:
        if self.schema != RETRIEVAL_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported retrieval schema")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.query_digest, "query_digest")
        require_digest(self.candidate_set_digest, "candidate_set_digest")
        require_digest(self.retrieval_digest, "retrieval_digest")
        if self.turn_id is not None and not self.turn_id.strip():
            raise ValueError("turn_id cannot be empty")
        if self.applicable_count != len(self.ranked_candidate_ids):
            raise ValueError("applicable count mismatch")
        if self.suppressed_count != len(self.suppressed):
            raise ValueError("suppressed count mismatch")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "retrieval_id": self.retrieval_id,
            "turn_id": self.turn_id,
            "repository_scope_id": self.repository_scope_id,
            "query_digest": self.query_digest,
            "candidate_set_digest": self.candidate_set_digest,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "suppressed": [[candidate_id, reason.value] for candidate_id, reason in self.suppressed],
            "applicable_count": self.applicable_count,
            "suppressed_count": self.suppressed_count,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
            "retrieval_policy": self.retrieval_policy,
            "retrieval_version": self.retrieval_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "retrieval_digest": self.retrieval_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeRetrievalReceipt":
        receipt = cls(
            retrieval_id=str(value["retrieval_id"]),
            turn_id=(str(value["turn_id"]) if value.get("turn_id") else None),
            repository_scope_id=str(value["repository_scope_id"]),
            query_digest=str(value["query_digest"]),
            candidate_set_digest=str(value["candidate_set_digest"]),
            ranked_candidate_ids=tuple(str(item) for item in value["ranked_candidate_ids"]),
            selected_candidate_ids=tuple(str(item) for item in value["selected_candidate_ids"]),
            suppressed=tuple(
                (str(item[0]), SuppressionReason(item[1])) for item in value["suppressed"]
            ),
            applicable_count=int(value["applicable_count"]),
            suppressed_count=int(value["suppressed_count"]),
            latency_ms=float(value["latency_ms"]),
            created_at=str(value["created_at"]),
            retrieval_policy=str(value["retrieval_policy"]),
            retrieval_version=int(value["retrieval_version"]),
            retrieval_digest=str(value["retrieval_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if receipt.retrieval_digest != structural_digest(receipt._payload()):
            raise ValueError("retrieval receipt digest mismatch")
        return receipt


def _score(query: str, candidate: KnowledgeCandidate) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document = " ".join((candidate.title, candidate.reusable_content, *candidate.preconditions))
    document_tokens = set(tokenize(document))
    overlap = query_tokens & document_tokens
    return len(overlap) / max(1, len(query_tokens))


def _suppression(result: KnowledgeApplicabilityResult) -> SuppressionReason:
    values = {item.value for item in result.reasons}
    for name, reason in (
        ("wrong_scope", SuppressionReason.WRONG_SCOPE),
        ("deprecated", SuppressionReason.NOT_ACTIVE),
        ("superseded", SuppressionReason.SUPERSEDED),
        ("fingerprint_changed", SuppressionReason.STALE),
        ("dependency_mismatch", SuppressionReason.DEPENDENCY_MISMATCH),
        ("missing_tool", SuppressionReason.MISSING_TOOL),
        ("missing_binary", SuppressionReason.MISSING_BINARY),
        ("digest_mismatch", SuppressionReason.DIGEST_MISMATCH),
        ("not_active", SuppressionReason.NOT_ACTIVE),
    ):
        if name in values:
            return reason
    return SuppressionReason.INCONCLUSIVE_GUARD


class KnowledgeRetriever:
    def __init__(
        self,
        store: KnowledgeRecordStore,
        environment: ApplicabilityEnvironment,
        *,
        max_candidates: int = 5,
        selection_mode: KnowledgeSelectionMode = KnowledgeSelectionMode.LEGACY_APPLICABLE,
    ) -> None:
        self.store = store
        self.environment = environment
        self.max_candidates = max(1, max_candidates)
        self.selection_mode = selection_mode

    def retrieve(
        self,
        query: str,
        *,
        retrieval_id: str,
        turn_id: str | None,
        candidate_types: tuple[CandidateType, ...],
        created_at: str | None = None,
        top_k: int | None = None,
        latency_ms: float | None = None,
    ) -> tuple[tuple[RetrievedKnowledge, ...], KnowledgeRetrievalReceipt]:
        started = time.perf_counter_ns()
        candidates = tuple(
            item for item in self.store.list_candidates() if item.candidate_type in candidate_types
        )
        applicable: list[tuple[KnowledgeCandidate, KnowledgeApplicabilityResult]] = []
        suppressed: list[tuple[str, SuppressionReason]] = []
        for candidate in candidates:
            applicability = evaluate_applicability(
                self.store,
                candidate_id=candidate.candidate_id,
                environment=self.environment,
                applicability_id="app-"
                + structural_digest(
                    {"retrieval_id": retrieval_id, "candidate_id": candidate.candidate_id}
                ),
                evaluated_at=created_at or spans.now_iso(),
            )
            if applicability.status is not ApplicabilityStatus.APPLICABLE:
                suppressed.append((candidate.candidate_id, _suppression(applicability)))
                continue
            applicable.append((candidate, applicability))
        limit = min(self.max_candidates, top_k or self.max_candidates)
        unsupported_query = False
        try:
            relevance_query = TaskRelevanceQuery.from_task(query)
        except (TypeError, ValueError):
            if self.selection_mode is KnowledgeSelectionMode.LEGACY_APPLICABLE:
                raise
            relevance_query = TaskRelevanceQuery.from_task("")
            unsupported_query = True
        selections: tuple[KnowledgeRelevanceSelection, ...]
        if self.selection_mode in {
            KnowledgeSelectionMode.TASK_RELEVANCE_V1,
            KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
        }:
            selected, selections = select_relevant_candidates(
                relevance_query,
                (item[0] for item in applicable),
                repository_scope_id=self.environment.repository_scope_id,
                turn_id=turn_id,
                retrieval_id=retrieval_id,
                limit=limit,
                unsupported_input=unsupported_query,
            )
            applicability_by_id = {item.candidate_id: result for item, result in applicable}
            selected_values = [
                (score, candidate, applicability_by_id[candidate.candidate_id])
                for candidate, score in selected
            ]
            suppressed.extend(
                (item.candidate_id, SuppressionReason.IRRELEVANT)
                for item in selections
                if item.decision is RelevanceDecision.ABSTAIN
            )
            ranked_ids = tuple(
                item.candidate_id
                for item in sorted(selections, key=lambda value: value.rank)
                if item.decision is RelevanceDecision.SELECT
            )
        else:
            ranked = [(_score(query, candidate), candidate, result) for candidate, result in applicable]
            ranked.sort(key=lambda item: (-item[0], item[1].candidate_id))
            positive = [item for item in ranked if item[0] > 0]
            selected_values = positive[:limit]
            selected_ids = {item[1].candidate_id for item in selected_values}
            selection_values: list[KnowledgeRelevanceSelection] = []
            for rank, (score, candidate, _result) in enumerate(ranked, start=1):
                selected = candidate.candidate_id in selected_ids
                reason = (
                    RelevanceReason.SELECT_RELEVANT
                    if selected
                    else (
                        RelevanceReason.ABSTAIN_NO_MEANINGFUL_OVERLAP
                        if score <= 0
                        else RelevanceReason.ABSTAIN_SELECTION_LIMIT
                    )
                )
                selection_values.append(
                    KnowledgeRelevanceSelection.create(
                        selection_id="selection-"
                        + structural_digest(
                            {"retrieval_id": retrieval_id, "candidate_id": candidate.candidate_id}
                        ),
                        retrieval_id=retrieval_id,
                        turn_id=turn_id,
                        repository_scope_id=self.environment.repository_scope_id,
                        selection_mode=self.selection_mode,
                        query_digest=relevance_query.query_digest,
                        candidate_id=candidate.candidate_id,
                        candidate_type=candidate.candidate_type,
                        rank=rank,
                        relevance_score=round(score, 8),
                        meaningful_overlap=(),
                        identifier_overlap=(),
                        decision=(RelevanceDecision.SELECT if selected else RelevanceDecision.ABSTAIN),
                        reason=reason,
                        selector_version=1,
                    )
                )
                if score <= 0:
                    suppressed.append((candidate.candidate_id, SuppressionReason.IRRELEVANT))
            selections = tuple(selection_values)
            ranked_ids = tuple(item[1].candidate_id for item in positive)
        for selection in selections:
            if self.store.write_relevance_selection(selection) is ImmutableWriteStatus.CONFLICT:
                raise KnowledgeStoreError("immutable relevance selection conflict")
        retrieved = tuple(
            RetrievedKnowledge(candidate, applicability, rank, score)
            for rank, (score, candidate, applicability) in enumerate(selected_values, start=1)
        )
        elapsed = (
            latency_ms
            if latency_ms is not None
            else (time.perf_counter_ns() - started) / 1_000_000
        )
        receipt = KnowledgeRetrievalReceipt.create(
            retrieval_id=retrieval_id,
            turn_id=turn_id,
            repository_scope_id=self.environment.repository_scope_id,
            query_digest=(
                relevance_query.query_digest
                if self.selection_mode
                in {
                    KnowledgeSelectionMode.TASK_RELEVANCE_V1,
                    KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY,
                }
                else structural_digest({"query": query})
            ),
            candidate_set_digest=structural_digest(
                {"candidate_ids": sorted(item.candidate_id for item in candidates)}
            ),
            ranked_candidate_ids=ranked_ids,
            selected_candidate_ids=tuple(item.candidate.candidate_id for item in retrieved),
            suppressed=tuple(sorted(suppressed, key=lambda item: item[0])),
            applicable_count=len(ranked_ids),
            suppressed_count=len(suppressed),
            latency_ms=round(float(elapsed), 6),
            created_at=created_at or spans.now_iso(),
            retrieval_policy="p3.scoped-keyword",
            retrieval_version=1,
        )
        if self.store.write_retrieval(receipt) is ImmutableWriteStatus.CONFLICT:
            raise KnowledgeStoreError("immutable retrieval receipt conflict")
        return retrieved, receipt


__all__ = [
    "KnowledgeRetrievalReceipt",
    "KnowledgeRetriever",
    "RetrievedKnowledge",
    "SuppressionReason",
]
