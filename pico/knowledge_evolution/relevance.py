"""Deterministic task-relevance selection for already-applicable P3 knowledge."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping

from pico.utils.bm25 import BM25Okapi

from .types import CandidateType, KnowledgeCandidate, require_digest, structural_digest

RELEVANCE_SELECTION_SCHEMA = "pico.knowledge-relevance-selection.v1"
RELEVANCE_QUERY_SCHEMA = "pico.task-relevance-query.v1"
SELECTOR_VERSION = 1
MAX_QUERY_CHARS = 4096
MAX_QUERY_TOKENS = 128
MAX_EVIDENCE_TOKENS = 12


class KnowledgeSelectionMode(StrEnum):
    LEGACY_APPLICABLE = "legacy_applicable"
    TASK_RELEVANCE_V1 = "task_relevance_v1"


class RelevanceDecision(StrEnum):
    SELECT = "select"
    ABSTAIN = "abstain"


class RelevanceReason(StrEnum):
    SELECT_RELEVANT = "select_relevant"
    ABSTAIN_NO_MEANINGFUL_OVERLAP = "abstain_no_meaningful_overlap"
    ABSTAIN_BELOW_RELEVANCE_FLOOR = "abstain_below_relevance_floor"
    ABSTAIN_SELECTION_LIMIT = "abstain_selection_limit"
    ABSTAIN_EMPTY_TASK_QUERY = "abstain_empty_task_query"
    ABSTAIN_UNSUPPORTED_RELEVANCE_INPUT = "abstain_unsupported_relevance_input"


# These terms describe almost any repository task.  They can contribute to BM25
# ordering, but never satisfy the relevance floor on their own.
_BOILERPLATE = frozenset(
    {
        "a", "add", "after", "an", "and", "are", "as", "at", "be", "before",
        "behavior", "by", "can", "change", "code", "current", "deterministic",
        "do", "does", "existing", "file", "fix", "focused", "for", "from",
        "has", "have", "if", "implement", "in", "into", "is", "it", "keep",
        "make", "may", "must", "new", "of", "on", "or", "p3", "project",
        "repository", "should", "small", "smallest", "task", "test", "tests",
        "that", "the", "this", "through", "to", "update", "use", "uses",
        "using", "when", "while", "with", "without", "work", "working",
    }
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|[0-9]+|[\u4e00-\u9fff]")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalized_tokens(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw in _WORD_RE.findall(text[:MAX_QUERY_CHARS]):
        lowered = raw.casefold()
        parts = tuple(
            part.casefold()
            for chunk in re.split(r"[_\-]+", raw)
            for part in _CAMEL_BOUNDARY.split(chunk)
            if len(part) >= 2 or "\u4e00" <= part <= "\u9fff"
        )
        if len(lowered) >= 2:
            values.append(lowered)
        values.extend(part for part in parts if part != lowered)
    return tuple(dict.fromkeys(values))


def _identifier_tokens(text: str) -> tuple[str, ...]:
    identifiers: list[str] = []
    for raw in _WORD_RE.findall(text[:MAX_QUERY_CHARS]):
        if "_" in raw or "-" in raw or _CAMEL_BOUNDARY.search(raw):
            parts = tuple(
                part.casefold()
                for chunk in re.split(r"[_\-]+", raw)
                for part in _CAMEL_BOUNDARY.split(chunk)
                if part
            )
            identifiers.append(raw.casefold())
            joined = "".join(parts)
            if joined != raw.casefold():
                identifiers.append(joined)
    return tuple(dict.fromkeys(identifiers))


@dataclass(frozen=True)
class TaskRelevanceQuery:
    query_digest: str
    normalized_tokens: tuple[str, ...]
    meaningful_tokens: tuple[str, ...]
    identifier_tokens: tuple[str, ...]
    truncated: bool
    schema: str = RELEVANCE_QUERY_SCHEMA
    schema_version: int = 1

    @classmethod
    def from_task(cls, task: str) -> "TaskRelevanceQuery":
        if not isinstance(task, str):
            raise ValueError("task relevance query must be text")
        bounded = task[:MAX_QUERY_CHARS]
        tokens = _normalized_tokens(bounded)[:MAX_QUERY_TOKENS]
        meaningful = tuple(item for item in tokens if item not in _BOILERPLATE)
        identifiers = tuple(item for item in _identifier_tokens(bounded) if item in tokens)
        payload = {
            "schema": RELEVANCE_QUERY_SCHEMA,
            "schema_version": 1,
            "normalized_tokens": list(tokens),
            "meaningful_tokens": list(meaningful),
            "identifier_tokens": list(identifiers),
            "truncated": len(task) > MAX_QUERY_CHARS,
        }
        return cls(
            query_digest=structural_digest(payload),
            normalized_tokens=tokens,
            meaningful_tokens=meaningful,
            identifier_tokens=identifiers,
            truncated=len(task) > MAX_QUERY_CHARS,
        )

    def __post_init__(self) -> None:
        require_digest(self.query_digest, "query_digest")
        if self.schema != RELEVANCE_QUERY_SCHEMA or self.schema_version != 1:
            raise ValueError("unsupported task relevance query schema")
        if len(self.normalized_tokens) > MAX_QUERY_TOKENS:
            raise ValueError("task relevance query is not bounded")


@dataclass(frozen=True)
class CandidateRelevanceDocument:
    candidate_id: str
    candidate_type: CandidateType
    document_digest: str
    tokens: tuple[str, ...]
    title_tokens: tuple[str, ...]
    identifier_tokens: tuple[str, ...]

    @classmethod
    def from_candidate(cls, candidate: KnowledgeCandidate) -> "CandidateRelevanceDocument":
        text = " ".join(
            (
                candidate.title,
                candidate.candidate_type.value,
                candidate.reusable_content,
                *candidate.preconditions,
            )
        )
        tokens = _normalized_tokens(text)
        title_tokens = _normalized_tokens(candidate.title)
        identifiers = _identifier_tokens(text)
        if not candidate.candidate_id or not tokens:
            raise ValueError("candidate relevance document is malformed")
        return cls(
            candidate_id=candidate.candidate_id,
            candidate_type=candidate.candidate_type,
            document_digest=structural_digest(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type.value,
                    "tokens": list(tokens),
                    "title_tokens": list(title_tokens),
                }
            ),
            tokens=tokens,
            title_tokens=title_tokens,
            identifier_tokens=identifiers,
        )


@dataclass(frozen=True)
class KnowledgeRelevanceSelection:
    selection_id: str
    retrieval_id: str
    turn_id: str | None
    repository_scope_id: str
    selection_mode: KnowledgeSelectionMode
    query_digest: str
    candidate_id: str
    candidate_type: CandidateType
    rank: int
    relevance_score: float
    meaningful_overlap: tuple[str, ...]
    identifier_overlap: tuple[str, ...]
    decision: RelevanceDecision
    reason: RelevanceReason
    selector_version: int
    selector_config_digest: str
    selection_digest: str = ""
    schema: str = RELEVANCE_SELECTION_SCHEMA
    schema_version: int = 1

    @classmethod
    def create(cls, **values: Any) -> "KnowledgeRelevanceSelection":
        values.setdefault(
            "selector_config_digest",
            structural_digest(
                {
                    "selection_mode": values["selection_mode"].value,
                    "selector_version": values["selector_version"],
                }
            ),
        )
        value = cls(**values)
        return replace(value, selection_digest=structural_digest(value._payload()))

    def __post_init__(self) -> None:
        if self.schema != RELEVANCE_SELECTION_SCHEMA or self.schema_version != 1:
            raise ValueError("unsupported relevance selection schema")
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.query_digest, "query_digest")
        require_digest(self.selector_config_digest, "selector_config_digest")
        if self.selection_digest:
            require_digest(self.selection_digest, "selection_digest")
        if self.rank < 1 or self.selector_version != SELECTOR_VERSION:
            raise ValueError("invalid relevance selection rank/version")
        if len(self.meaningful_overlap) > MAX_EVIDENCE_TOKENS:
            raise ValueError("relevance evidence is not bounded")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "retrieval_id": self.retrieval_id,
            "turn_id": self.turn_id,
            "repository_scope_id": self.repository_scope_id,
            "selection_mode": self.selection_mode.value,
            "query_digest": self.query_digest,
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type.value,
            "rank": self.rank,
            "relevance_score": self.relevance_score,
            "meaningful_overlap": list(self.meaningful_overlap),
            "identifier_overlap": list(self.identifier_overlap),
            "decision": self.decision.value,
            "reason": self.reason.value,
            "selector_version": self.selector_version,
            "selector_config_digest": self.selector_config_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "selection_digest": self.selection_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeRelevanceSelection":
        result = cls(
            selection_id=str(value["selection_id"]),
            retrieval_id=str(value["retrieval_id"]),
            turn_id=str(value["turn_id"]) if value.get("turn_id") else None,
            repository_scope_id=str(value["repository_scope_id"]),
            selection_mode=KnowledgeSelectionMode(value["selection_mode"]),
            query_digest=str(value["query_digest"]),
            candidate_id=str(value["candidate_id"]),
            candidate_type=CandidateType(value["candidate_type"]),
            rank=int(value["rank"]),
            relevance_score=float(value["relevance_score"]),
            meaningful_overlap=tuple(str(item) for item in value["meaningful_overlap"]),
            identifier_overlap=tuple(str(item) for item in value["identifier_overlap"]),
            decision=RelevanceDecision(value["decision"]),
            reason=RelevanceReason(value["reason"]),
            selector_version=int(value["selector_version"]),
            selector_config_digest=str(value["selector_config_digest"]),
            selection_digest=str(value["selection_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if result.selection_digest != structural_digest(result._payload()):
            raise ValueError("relevance selection digest mismatch")
        return result


def select_relevant_candidates(
    query: TaskRelevanceQuery,
    candidates: Iterable[KnowledgeCandidate],
    *,
    repository_scope_id: str,
    turn_id: str | None,
    retrieval_id: str,
    limit: int,
    unsupported_input: bool = False,
) -> tuple[tuple[KnowledgeCandidate, float], tuple[KnowledgeRelevanceSelection, ...]]:
    """Rank and abstain without changing any upstream eligibility decision."""

    candidate_values = tuple(candidates)
    documents: list[CandidateRelevanceDocument | None] = []
    for candidate in candidate_values:
        try:
            documents.append(CandidateRelevanceDocument.from_candidate(candidate))
        except (TypeError, ValueError):
            documents.append(None)
    valid_docs = [item for item in documents if item is not None]
    corpus = [list(item.tokens) for item in valid_docs]
    bm25_scores = BM25Okapi(corpus).get_scores(list(query.normalized_tokens))
    score_by_id = {
        item.candidate_id: score for item, score in zip(valid_docs, bm25_scores, strict=True)
    }
    evaluated: list[tuple[float, KnowledgeCandidate, tuple[str, ...], tuple[str, ...], RelevanceReason]] = []
    invalid: list[KnowledgeCandidate] = []
    query_meaningful = set(query.meaningful_tokens)
    query_identifiers = set(query.identifier_tokens)
    for candidate, document in zip(candidate_values, documents, strict=True):
        if document is None:
            invalid.append(candidate)
            continue
        meaningful = tuple(sorted(query_meaningful & (set(document.tokens) - _BOILERPLATE)))
        identifiers = tuple(sorted(query_identifiers & set(document.identifier_tokens)))
        title_overlap = query_meaningful & set(document.title_tokens)
        coverage = len(meaningful) / max(1, len(query_meaningful))
        score = round(score_by_id[candidate.candidate_id] + coverage + 0.5 * len(identifiers), 8)
        passes = bool(identifiers or title_overlap or len(meaningful) >= 2)
        reason = (
            RelevanceReason.SELECT_RELEVANT
            if passes
            else (
                RelevanceReason.ABSTAIN_NO_MEANINGFUL_OVERLAP
                if not meaningful
                else RelevanceReason.ABSTAIN_BELOW_RELEVANCE_FLOOR
            )
        )
        evaluated.append((score, candidate, meaningful, identifiers, reason))
    evaluated.sort(key=lambda item: (-item[0], item[1].candidate_id))
    selected_ids = {
        item[1].candidate_id
        for item in evaluated
        if item[4] is RelevanceReason.SELECT_RELEVANT
    }
    selected_ids = set(
        item[1].candidate_id
        for item in [value for value in evaluated if value[1].candidate_id in selected_ids][
            : max(0, limit)
        ]
    )
    receipts: list[KnowledgeRelevanceSelection] = []
    selected: list[tuple[KnowledgeCandidate, float]] = []
    empty_query = not query.meaningful_tokens
    for rank, (score, candidate, meaningful, identifiers, reason) in enumerate(evaluated, start=1):
        if unsupported_input:
            decision = RelevanceDecision.ABSTAIN
            reason = RelevanceReason.ABSTAIN_UNSUPPORTED_RELEVANCE_INPUT
        elif empty_query:
            decision = RelevanceDecision.ABSTAIN
            reason = RelevanceReason.ABSTAIN_EMPTY_TASK_QUERY
        elif candidate.candidate_id in selected_ids:
            decision = RelevanceDecision.SELECT
            selected.append((candidate, score))
        else:
            decision = RelevanceDecision.ABSTAIN
            if reason is RelevanceReason.SELECT_RELEVANT:
                reason = RelevanceReason.ABSTAIN_SELECTION_LIMIT
        receipts.append(
            KnowledgeRelevanceSelection.create(
                selection_id="selection-"
                + structural_digest({"retrieval_id": retrieval_id, "candidate_id": candidate.candidate_id}),
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                repository_scope_id=repository_scope_id,
                selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
                query_digest=query.query_digest,
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                rank=rank,
                relevance_score=score,
                meaningful_overlap=meaningful[:MAX_EVIDENCE_TOKENS],
                identifier_overlap=identifiers[:MAX_EVIDENCE_TOKENS],
                decision=decision,
                reason=reason,
                selector_version=SELECTOR_VERSION,
            )
        )
    for candidate in invalid:
        receipts.append(
            KnowledgeRelevanceSelection.create(
                selection_id="selection-"
                + structural_digest({"retrieval_id": retrieval_id, "candidate_id": candidate.candidate_id}),
                retrieval_id=retrieval_id,
                turn_id=turn_id,
                repository_scope_id=repository_scope_id,
                selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
                query_digest=query.query_digest,
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                rank=len(receipts) + 1,
                relevance_score=0.0,
                meaningful_overlap=(),
                identifier_overlap=(),
                decision=RelevanceDecision.ABSTAIN,
                reason=RelevanceReason.ABSTAIN_UNSUPPORTED_RELEVANCE_INPUT,
                selector_version=SELECTOR_VERSION,
            )
        )
    return tuple(selected), tuple(receipts)


__all__ = [
    "CandidateRelevanceDocument",
    "KnowledgeRelevanceSelection",
    "KnowledgeSelectionMode",
    "RelevanceDecision",
    "RelevanceReason",
    "TaskRelevanceQuery",
    "select_relevant_candidates",
]
