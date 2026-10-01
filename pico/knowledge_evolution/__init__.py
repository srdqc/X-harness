"""P3 controlled knowledge-evolution evidence and persistence contracts."""

from .canonicalize import (
    CanonicalProposal,
    KnowledgeProposal,
    ProposalRejectionReason,
    ProposalValidation,
    UnsafeKnowledgeContentError,
    validate_and_canonicalize,
)
from .eligibility import (
    SUPPORTED_POLICY,
    CandidateEligibilityInput,
    CandidateEligibilityResult,
    EligibilityReason,
    EligibilityStatus,
    evaluate_candidate_eligibility,
)
from .extraction import (
    CandidateExtractionOutcome,
    ExtractionFailureReason,
    ExtractionStatus,
    KnowledgeExtractionContext,
    KnowledgeExtractionRequest,
    KnowledgeExtractionResult,
    KnowledgeExtractor,
    RejectedProposal,
    build_extraction_context,
    extract_candidates,
)
from .index import CandidateComparison, CandidateIndex, CandidateRelation
from .scope import (
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    RepositoryScopeReason,
    RepositoryScopeResolution,
    RepositoryScopeResolver,
    RepositoryScopeStatus,
    normalize_git_remote,
    normalize_project_key,
)
from .store import (
    ImmutableWriteStatus,
    KnowledgeRecordStore,
    KnowledgeStoreError,
)
from .task_success import (
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
)
from .types import (
    CandidateType,
    ContentClass,
    KnowledgeCandidate,
    SourceTurnReference,
)

__all__ = [
    "SUPPORTED_POLICY",
    "CandidateEligibilityInput",
    "CandidateEligibilityResult",
    "CandidateExtractionOutcome",
    "CandidateComparison",
    "CandidateIndex",
    "CandidateRelation",
    "CandidateType",
    "CanonicalProposal",
    "ContentClass",
    "EligibilityReason",
    "EligibilityStatus",
    "ExtractionFailureReason",
    "ExtractionStatus",
    "ImmutableWriteStatus",
    "KnowledgeCandidate",
    "KnowledgeExtractionContext",
    "KnowledgeExtractionRequest",
    "KnowledgeExtractionResult",
    "KnowledgeExtractor",
    "KnowledgeProposal",
    "KnowledgeRecordStore",
    "KnowledgeStoreError",
    "RepositoryIdentityMethod",
    "RepositoryScopeIdentity",
    "RepositoryScopeReason",
    "RepositoryScopeResolution",
    "RepositoryScopeResolver",
    "RepositoryScopeStatus",
    "ProposalRejectionReason",
    "ProposalValidation",
    "RejectedProposal",
    "SourceTurnReference",
    "TaskSuccessEvidence",
    "TaskSuccessSource",
    "TaskSuccessStatus",
    "UnsafeKnowledgeContentError",
    "build_extraction_context",
    "evaluate_candidate_eligibility",
    "extract_candidates",
    "normalize_git_remote",
    "normalize_project_key",
    "validate_and_canonicalize",
]
