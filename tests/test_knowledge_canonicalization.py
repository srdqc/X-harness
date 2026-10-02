from __future__ import annotations

from pico.knowledge_evolution import (
    CandidateType,
    ContentClass,
    KnowledgeProposal,
    ProposalRejectionReason,
    validate_and_canonicalize,
)


def _proposal(**overrides):
    values = {
        "candidate_type": CandidateType.EXPERIENCE.value,
        "content_class": ContentClass.STRATEGY.value,
        "title": "  Stable   title ",
        "reusable_content": "First   line.\r\n\r\n Second line.  ",
        "preconditions": (" zeta  condition ", "Alpha condition", "Alpha condition"),
        "applicability": ("Scope matches",),
    }
    values.update(overrides)
    return KnowledgeProposal(**values)


def test_canonicalization_normalizes_whitespace_and_orders_preconditions() -> None:
    result = validate_and_canonicalize(_proposal())
    assert result.reason is None
    assert result.proposal.title == "Stable title"
    assert result.proposal.reusable_content == "First line.\n\nSecond line."
    assert result.proposal.preconditions == ("Alpha condition", "zeta condition")


def test_equivalent_proposals_have_identical_canonical_form() -> None:
    first = validate_and_canonicalize(_proposal()).proposal
    second = validate_and_canonicalize(
        _proposal(
            title="Stable title",
            reusable_content="First line.\n\nSecond line.",
            preconditions=("Alpha condition", "zeta condition"),
        )
    ).proposal
    assert first == second


def test_destination_specific_contracts_are_enforced() -> None:
    fact = validate_and_canonicalize(
        _proposal(candidate_type="memory_fact", content_class="fact")
    )
    skill = validate_and_canonicalize(
        _proposal(candidate_type="skill_candidate", content_class="procedure")
    )
    wrong = validate_and_canonicalize(
        _proposal(candidate_type="experience", content_class="fact")
    )
    assert fact.reason is ProposalRejectionReason.MISSING_APPLICABILITY
    assert skill.reason is ProposalRejectionReason.MISSING_APPLICABILITY
    assert wrong.reason is ProposalRejectionReason.INVALID_DESTINATION_CONTENT


def test_skill_validation_expectations_survive_canonicalization() -> None:
    result = validate_and_canonicalize(
        _proposal(
            candidate_type="skill_candidate",
            content_class="procedure",
            validation_expectations=("Verifier reports PASS",),
        )
    )
    assert "Validation expectations:\n- Verifier reports PASS" in result.proposal.reusable_content


def test_runtime_shape_errors_are_rejected_as_malformed() -> None:
    result = validate_and_canonicalize(
        _proposal(preconditions="not-a-tuple")
    )
    assert result.reason is ProposalRejectionReason.MALFORMED_PROPOSAL
