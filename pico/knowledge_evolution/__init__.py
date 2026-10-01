"""P3 controlled knowledge-evolution evidence and persistence contracts."""

from .eligibility import (
    SUPPORTED_POLICY,
    CandidateEligibilityInput,
    CandidateEligibilityResult,
    EligibilityReason,
    EligibilityStatus,
    evaluate_candidate_eligibility,
)
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
    "CandidateType",
    "ContentClass",
    "EligibilityReason",
    "EligibilityStatus",
    "ImmutableWriteStatus",
    "KnowledgeCandidate",
    "KnowledgeRecordStore",
    "KnowledgeStoreError",
    "RepositoryIdentityMethod",
    "RepositoryScopeIdentity",
    "RepositoryScopeReason",
    "RepositoryScopeResolution",
    "RepositoryScopeResolver",
    "RepositoryScopeStatus",
    "SourceTurnReference",
    "TaskSuccessEvidence",
    "TaskSuccessSource",
    "TaskSuccessStatus",
    "evaluate_candidate_eligibility",
    "normalize_git_remote",
    "normalize_project_key",
]
