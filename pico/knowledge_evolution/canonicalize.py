"""Deterministic validation and canonicalization of untrusted proposals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .types import CandidateType, ContentClass, structural_digest

MAX_TITLE_LENGTH = 256
MAX_CONTENT_LENGTH = 32_768
MAX_PRECONDITIONS = 32
MAX_PRECONDITION_LENGTH = 1_024

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}", re.IGNORECASE),
    re.compile(r"https?://[^\s/:]+:[^\s/@]+@", re.IGNORECASE),
)
_REASONING_PATTERNS = (
    re.compile(r"<\/?(?:thinking|reasoning)>", re.IGNORECASE),
    re.compile(r"\b(?:chain[- ]of[- ]thought|private reasoning|internal reasoning)\b", re.IGNORECASE),
)
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?:Users|Documents and Settings|home)\\", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])/(?:home|Users|private|tmp)/[^\s]+"),
)
_ENV_DUMP = re.compile(r"(?:^|\n)(?:[A-Z][A-Z0-9_]{2,}=.*(?:\n|$)){3,}")


class ProposalRejectionReason(str, Enum):
    MALFORMED_PROPOSAL = "malformed_proposal"
    UNSUPPORTED_CANDIDATE_TYPE = "unsupported_candidate_type"
    INVALID_CONTENT_CLASS = "invalid_content_class"
    OVERSIZED_CONTENT = "oversized_content"
    UNSAFE_CONTENT = "unsafe_content"
    MISSING_APPLICABILITY = "missing_applicability"
    INVALID_DESTINATION_CONTENT = "invalid_destination_content"


class UnsafeKnowledgeContentError(ValueError):
    def __init__(self, finding: str) -> None:
        super().__init__(finding)
        self.finding = finding


@dataclass(frozen=True)
class KnowledgeProposal:
    """Untrusted extractor proposal; deliberately excludes authority fields."""

    candidate_type: str
    content_class: str
    title: str
    reusable_content: str
    preconditions: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    fact_subject: str | None = None
    fact_value: str | None = None
    validation_expectations: tuple[str, ...] = ()
    portable_hint: bool = False


@dataclass(frozen=True)
class CanonicalProposal:
    candidate_type: CandidateType
    content_class: ContentClass
    title: str
    reusable_content: str
    preconditions: tuple[str, ...]
    applicability_fingerprints: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]
    portable_hint: bool
    fact_subject: str | None
    fact_value: str | None


@dataclass(frozen=True)
class ProposalValidation:
    proposal: CanonicalProposal | None
    reason: ProposalRejectionReason | None
    sanitization_findings: tuple[str, ...] = ()


def normalize_inline(value: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    if "\x00" in value:
        raise ValueError("text contains NUL")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > maximum:
        raise ValueError("text is empty or oversized")
    return normalized


def normalize_body(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("content must be a string")
    if len(value) > MAX_CONTENT_LENGTH:
        raise OverflowError("content is oversized")
    if "\x00" in value:
        raise ValueError("content contains NUL")
    lines = [" ".join(line.split()) for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    normalized = "\n".join(compact)
    if not normalized:
        raise ValueError("content is empty")
    return normalized


def inspect_unsafe_text(value: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            raise UnsafeKnowledgeContentError("suspected_secret")
    for pattern in _REASONING_PATTERNS:
        if pattern.search(value):
            raise UnsafeKnowledgeContentError("provider_reasoning")
    for pattern in _LOCAL_PATH_PATTERNS:
        if pattern.search(value):
            raise UnsafeKnowledgeContentError("absolute_local_path")
    if _ENV_DUMP.search(value):
        raise UnsafeKnowledgeContentError("environment_dump")


def _normalized_items(values: tuple[str, ...], *, maximum_count: int) -> tuple[str, ...]:
    if len(values) > maximum_count:
        raise OverflowError("too many bounded items")
    normalized = {normalize_inline(value, maximum=MAX_PRECONDITION_LENGTH) for value in values}
    return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))


def validate_and_canonicalize(proposal: object) -> ProposalValidation:
    if not isinstance(proposal, KnowledgeProposal):
        return ProposalValidation(None, ProposalRejectionReason.MALFORMED_PROPOSAL)
    sequence_fields = (
        proposal.preconditions,
        proposal.applicability,
        proposal.evidence_refs,
        proposal.validation_expectations,
    )
    if (
        not isinstance(proposal.candidate_type, str)
        or not isinstance(proposal.content_class, str)
        or any(
            not isinstance(items, tuple) or any(not isinstance(item, str) for item in items)
            for items in sequence_fields
        )
        or not isinstance(proposal.portable_hint, bool)
        or (proposal.fact_subject is not None and not isinstance(proposal.fact_subject, str))
        or (proposal.fact_value is not None and not isinstance(proposal.fact_value, str))
    ):
        return ProposalValidation(None, ProposalRejectionReason.MALFORMED_PROPOSAL)
    try:
        candidate_type = CandidateType(proposal.candidate_type)
    except (TypeError, ValueError):
        return ProposalValidation(None, ProposalRejectionReason.UNSUPPORTED_CANDIDATE_TYPE)
    try:
        content_class = ContentClass(proposal.content_class)
    except (TypeError, ValueError):
        return ProposalValidation(None, ProposalRejectionReason.INVALID_CONTENT_CLASS)
    try:
        title = normalize_inline(proposal.title, maximum=MAX_TITLE_LENGTH)
        content = normalize_body(proposal.reusable_content)
        preconditions = _normalized_items(
            tuple(proposal.preconditions), maximum_count=MAX_PRECONDITIONS
        )
        applicability = _normalized_items(
            tuple(proposal.applicability), maximum_count=MAX_PRECONDITIONS
        )
        evidence_refs = _normalized_items(tuple(proposal.evidence_refs), maximum_count=64)
        expectations = _normalized_items(
            tuple(proposal.validation_expectations), maximum_count=MAX_PRECONDITIONS
        )
    except OverflowError:
        return ProposalValidation(None, ProposalRejectionReason.OVERSIZED_CONTENT)
    except (TypeError, ValueError):
        return ProposalValidation(None, ProposalRejectionReason.MALFORMED_PROPOSAL)

    all_text = (title, content, *preconditions, *applicability, *evidence_refs, *expectations)
    try:
        for item in all_text:
            inspect_unsafe_text(item)
    except UnsafeKnowledgeContentError as exc:
        return ProposalValidation(
            None,
            ProposalRejectionReason.UNSAFE_CONTENT,
            (exc.finding,),
        )

    allowed_classes = {
        CandidateType.MEMORY_FACT: {ContentClass.FACT},
        CandidateType.EXPERIENCE: {
            ContentClass.STRATEGY,
            ContentClass.RECOVERY,
            ContentClass.WARNING,
            ContentClass.ANTI_PATTERN,
        },
        CandidateType.SKILL_CANDIDATE: {ContentClass.PROCEDURE},
    }
    if content_class not in allowed_classes[candidate_type]:
        return ProposalValidation(None, ProposalRejectionReason.INVALID_DESTINATION_CONTENT)

    fact_subject: str | None = None
    fact_value: str | None = None
    fingerprints: list[tuple[str, str]] = []
    if candidate_type is CandidateType.MEMORY_FACT:
        if not proposal.fact_subject or not proposal.fact_value or not (preconditions or applicability):
            return ProposalValidation(None, ProposalRejectionReason.MISSING_APPLICABILITY)
        try:
            fact_subject = normalize_inline(proposal.fact_subject, maximum=256).casefold()
            fact_value = normalize_inline(proposal.fact_value, maximum=2_048)
            inspect_unsafe_text(fact_subject)
            inspect_unsafe_text(fact_value)
        except UnsafeKnowledgeContentError as exc:
            return ProposalValidation(None, ProposalRejectionReason.UNSAFE_CONTENT, (exc.finding,))
        except (TypeError, ValueError):
            return ProposalValidation(None, ProposalRejectionReason.MALFORMED_PROPOSAL)
        fingerprints.append((f"fact:{fact_subject}", structural_digest({"value": fact_value})))
    elif candidate_type is CandidateType.SKILL_CANDIDATE:
        if not preconditions or not expectations:
            return ProposalValidation(None, ProposalRejectionReason.MISSING_APPLICABILITY)
        content = "\n".join(
            (
                content,
                "",
                "Validation expectations:",
                *(f"- {item}" for item in expectations),
            )
        )

    for index, guard in enumerate(applicability):
        # Preserve the small machine-checkable namespaces understood by the
        # Runtime applicability evaluator.  Free-form predicates remain
        # opaque indexed guards and therefore fail closed without explicit
        # environment evidence.
        key = (
            f"guard:{guard}"
            if guard.startswith(("tool:", "binary:"))
            else f"guard:{index}"
        )
        fingerprints.append((key, structural_digest({"guard": guard})))
    return ProposalValidation(
        CanonicalProposal(
            candidate_type=candidate_type,
            content_class=content_class,
            title=title,
            reusable_content=content,
            preconditions=preconditions,
            applicability_fingerprints=tuple(fingerprints),
            evidence_refs=evidence_refs,
            portable_hint=bool(proposal.portable_hint),
            fact_subject=fact_subject,
            fact_value=fact_value,
        ),
        None,
    )


__all__ = [
    "CanonicalProposal",
    "KnowledgeProposal",
    "ProposalRejectionReason",
    "ProposalValidation",
    "UnsafeKnowledgeContentError",
    "inspect_unsafe_text",
    "normalize_body",
    "normalize_inline",
    "validate_and_canonicalize",
]
