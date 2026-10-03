"""Human-reviewable P3R.4 v2 prompt/verifier contract audit."""

from __future__ import annotations

from dataclasses import dataclass

from .tasks import OFFICIAL_TASKS_V2

EXPLICITLY_SUPPORTED = "EXPLICITLY_SUPPORTED"
REPOSITORY_CONTRACT_SUPPORTED = "REPOSITORY_CONTRACT_SUPPORTED"
DEFECT_HIDDEN_REQUIREMENT = "DEFECT / HIDDEN_REQUIREMENT"


@dataclass(frozen=True)
class OfficialContractAudit:
    task_id: str
    visible_prompt_requirement: str
    verifier_check: str
    known_valid_fixture_behavior: str
    explicitly_supported: tuple[str, ...]
    repository_contract_supported: tuple[str, ...] = ()
    hidden_requirements: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return DEFECT_HIDDEN_REQUIREMENT if self.hidden_requirements else EXPLICITLY_SUPPORTED


OFFICIAL_V2_CONTRACT_AUDIT = (
    OfficialContractAudit(
        "p3r4-nav-01",
        "Public list_retrievals API; validated retrieval-ID ordering; optional turn_id; identity and malformed-record safety; empty/no-match; focused tests.",
        "Calls list_retrievals, checks retrieval-ID order and turn filtering, then runs the existing retrieval test target.",
        "Adds the store method using the record loader, sorted paths, identity validation, and turn filtering.",
        ("method name", "return behavior", "ordering", "turn filter", "validation", "empty/no-match", "tests"),
        ("KnowledgeRecordStore owner", "retrieval model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-nav-02",
        "Public list_outcome_associations API; validated association-ID ordering; optional turn_id; identity and malformed-record safety; empty/no-match; focused tests.",
        "Calls list_outcome_associations, checks association-ID order and turn filtering, then runs the existing usage test target.",
        "Adds the store method using the record loader, sorted paths, identity validation, and turn filtering.",
        ("method name", "return behavior", "ordering", "turn filter", "validation", "empty/no-match", "tests"),
        ("KnowledgeRecordStore owner", "usage model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-nav-03",
        "Public list_applicability_results API; validated applicability-ID ordering; composable candidate_id and ApplicabilityStatus filters; identity safety; empty/no-match; focused tests.",
        "Checks API existence through calls, ordering, each filter, combined filters, empty/no-match, reload validation, and mismatch rejection.",
        "Canonical inline loop and structurally different helper-based implementation both satisfy the probe.",
        ("method name", "return behavior", "ordering", "candidate filter", "status enum filter", "composition", "reload", "identity", "empty/no-match", "tests"),
        ("KnowledgeRecordStore owner", "applicability model and test module paths", "KnowledgeStoreError for stored-record corruption"),
    ),
    OfficialContractAudit(
        "p3r4-impl-01",
        "Derived selection_summary with ranked, selected, and suppressed counts; unchanged serialization; compatibility tests.",
        "Reads selection_summary and checks all three counts, then runs retrieval tests.",
        "Adds a derived property without adding a serialized field.",
        ("property name", "keys and counts", "serialization compatibility", "tests"),
        ("retrieval model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-impl-02",
        "Derived is_model_visible property with retrieved false and injected/referenced/activated true; unchanged serialization; every mode tested.",
        "Reads the property for all four KnowledgeUsageMode values, then runs usage tests.",
        "Adds a derived enum-membership property without changing serialized evidence.",
        ("property name", "mode truth table", "serialization compatibility", "tests"),
        ("usage model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-impl-03",
        "Derived usage_count and candidate_count properties; unchanged serialization; round-trip tests.",
        "Checks both property values on a valid association, then runs usage tests.",
        "Adds two length-derived properties without changing serialized evidence.",
        ("property names", "count behavior", "serialization compatibility", "tests"),
        ("usage model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-debug-01",
        "Reject duplicate usage IDs while preserving valid outcome associations; positive and negative tests.",
        "Builds a duplicate association and requires the repository-standard validation exception, then runs usage tests.",
        "Adds the duplicate invariant to association construction.",
        ("duplicate rejection", "valid compatibility", "tests"),
        ("ValueError is the existing immutable-record validation contract", "usage model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-debug-02",
        "Selected candidate IDs are unique and members of the ranked set; valid receipts preserved; regression tests.",
        "Requires duplicate and out-of-ranked selections to raise the repository-standard validation exception, then runs retrieval tests.",
        "Adds both invariants to retrieval receipt construction.",
        ("uniqueness", "ranked membership", "valid compatibility", "tests"),
        ("ValueError is the existing immutable-record validation contract", "retrieval model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-debug-03",
        "Reject NaN and both infinities while preserving finite scores and deterministic digests; regression tests.",
        "Requires all three non-finite forms to raise the repository-standard validation exception, then runs usage tests.",
        "Adds a finite-number invariant to usage receipt construction.",
        ("non-finite rejection", "finite compatibility", "digest stability", "tests"),
        ("ValueError is the existing immutable-record validation contract", "usage model and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-int-01",
        "Composable candidate_id and usage-mode filters on existing durable usage listing, including turn filtering and deterministic order; integration tests.",
        "Calls existing list_usages with each new enum-backed filter, then runs usage tests.",
        "Extends the existing public list_usages method with two optional predicates.",
        ("candidate filter", "usage-mode filter", "composition", "turn compatibility", "ordering", "tests"),
        ("existing list_usages public API", "KnowledgeUsageMode enum", "store and usage test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-int-02",
        "Public count_turn_retrievals(store, turn_id), backed by deterministic listing and exported at the package boundary; focused tests.",
        "Imports the exact public helper, checks matching and missing turns, then runs retrieval tests.",
        "Counts the public listing result and exports the helper from the established package module.",
        ("helper name and signature", "listing-backed behavior", "package export", "zero/matching counts", "tests"),
        ("retrieval implementation, package export, and test module paths"),
    ),
    OfficialContractAudit(
        "p3r4-int-03",
        "p3_official_contract suite with retrieval then usage targets, each exactly once; stable resolution tests.",
        "Reads the canonical test matrix and checks the exact ordered target list and uniqueness, then runs runner tests.",
        "Registers the named suite with the two declared targets in declared order.",
        ("suite name", "exact target paths", "target order", "uniqueness", "resolution tests"),
        ("canonical test-matrix and runner-test ownership paths"),
    ),
)


def audit_official_v2_contracts() -> tuple[OfficialContractAudit, ...]:
    """Return the frozen matrix after checking task coverage and verifier binding."""

    tasks = {task.task_id: task for task in OFFICIAL_TASKS_V2}
    rows = {row.task_id: row for row in OFFICIAL_V2_CONTRACT_AUDIT}
    if tasks.keys() != rows.keys():
        raise ValueError("official v2 contract audit does not cover the frozen task set")
    return OFFICIAL_V2_CONTRACT_AUDIT


__all__ = [
    "DEFECT_HIDDEN_REQUIREMENT",
    "EXPLICITLY_SUPPORTED",
    "OFFICIAL_V2_CONTRACT_AUDIT",
    "OfficialContractAudit",
    "REPOSITORY_CONTRACT_SUPPORTED",
    "audit_official_v2_contracts",
]
