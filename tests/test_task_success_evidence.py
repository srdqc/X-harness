from __future__ import annotations

from dataclasses import replace

import pytest

from pico.knowledge_evolution import (
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
)

_DIGEST = "a" * 64
_SCOPE = "b" * 64


def _evidence(**overrides):
    values = {
        "evidence_id": "success-1",
        "source_kind": TaskSuccessSource.SEALED_VERIFIER,
        "source_turn_id": "turn-1",
        "status": TaskSuccessStatus.PASS,
        "producer_id": "sealed-verifier",
        "producer_version": "1",
        "repository_scope_id": _SCOPE,
        "target_state_digest": "c" * 64,
        "result_digest": _DIGEST,
        "provenance_refs": ("artifact:result",),
        "created_at": "2026-10-01T00:00:00Z",
    }
    values.update(overrides)
    return TaskSuccessEvidence.create(**values)


@pytest.mark.parametrize("status", tuple(TaskSuccessStatus))
def test_sealed_evidence_has_explicit_status_and_deterministic_digest(status) -> None:
    first = _evidence(status=status)
    second = _evidence(status=status)
    assert first == second
    assert TaskSuccessEvidence.from_dict(first.to_dict()) == first


def test_workspace_test_requires_state_binding_at_eligibility_boundary() -> None:
    evidence = _evidence(
        source_kind=TaskSuccessSource.WORKSPACE_TEST,
        target_state_digest=None,
    )
    assert evidence.has_state_binding is False


def test_human_validation_requires_actor_and_frozen_benchmark_is_representable() -> None:
    human = _evidence(
        source_kind=TaskSuccessSource.HUMAN_VALIDATION,
        human_actor_id="reviewer@example.test",
        comment="Validated the sealed artifact.",
    )
    benchmark = _evidence(
        evidence_id="benchmark-1",
        source_kind=TaskSuccessSource.FROZEN_BENCHMARK,
        producer_id="picobench:fixture-v1",
    )
    assert human.human_actor_id == "reviewer@example.test"
    assert benchmark.source_kind is TaskSuccessSource.FROZEN_BENCHMARK
    with pytest.raises(ValueError):
        _evidence(source_kind=TaskSuccessSource.HUMAN_VALIDATION)


def test_corrupted_digest_is_rejected() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="digest mismatch"):
        TaskSuccessEvidence.from_dict(replace(evidence, evidence_digest="d" * 64).to_dict())


def test_agent_runtime_tool_and_delivery_claims_are_not_evidence_fields() -> None:
    fields = set(_evidence().to_dict())
    assert fields.isdisjoint(
        {"agent_answer", "runtime_success", "tool_success", "delivery_success", "confidence"}
    )

