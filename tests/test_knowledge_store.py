from __future__ import annotations

import json
from pathlib import Path

import pytest

from pico.knowledge_evolution import (
    CandidateType,
    ContentClass,
    ImmutableWriteStatus,
    KnowledgeCandidate,
    KnowledgeRecordStore,
    KnowledgeStoreError,
    RepositoryIdentityMethod,
    RepositoryScopeIdentity,
    SourceTurnReference,
    TaskSuccessEvidence,
    TaskSuccessSource,
    TaskSuccessStatus,
)
from pico.knowledge_evolution.types import structural_digest


def _scope() -> RepositoryScopeIdentity:
    return RepositoryScopeIdentity.create(
        method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
        canonical_identity="project:store-test",
        evidence=(("project_key", "store-test"),),
    )


def _success(scope_id: str) -> TaskSuccessEvidence:
    return TaskSuccessEvidence.create(
        evidence_id="success-1",
        source_kind=TaskSuccessSource.EXPECTED_ARTIFACT,
        source_turn_id="turn-1",
        status=TaskSuccessStatus.PASS,
        producer_id="artifact-verifier",
        producer_version="1",
        repository_scope_id=scope_id,
        target_state_digest="a" * 64,
        result_digest="b" * 64,
        provenance_refs=("artifact:1",),
        created_at="2026-10-01T00:00:00Z",
    )


def _candidate(scope_id: str, content: str = "Reusable content") -> KnowledgeCandidate:
    return KnowledgeCandidate.create(
        candidate_id="candidate-1",
        candidate_type=CandidateType.EXPERIENCE,
        content_class=ContentClass.STRATEGY,
        qualifying_sources=(
            SourceTurnReference(
                turn_id="turn-1",
                replay_digest="c" * 64,
                verification_digest="d" * 64,
                task_success_evidence_ids=("success-1",),
                task_success_evidence_digests=("e" * 64,),
            ),
        ),
        repository_scope_id=scope_id,
        title="Store test",
        reusable_content=content,
        extraction_policy="p3.manual-candidate",
        extraction_policy_version=1,
        provenance_refs=("turn:turn-1",),
        created_at="2026-10-01T00:00:00Z",
        created_by="test",
    )


def test_immutable_records_create_round_trip_and_survive_reload(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    scope = _scope()
    success = _success(scope.repository_scope_id)
    candidate = _candidate(scope.repository_scope_id)
    assert store.write_scope(scope) is ImmutableWriteStatus.CREATED
    assert store.write_task_success(success) is ImmutableWriteStatus.CREATED
    assert store.write_candidate(candidate) is ImmutableWriteStatus.CREATED

    reloaded = KnowledgeRecordStore(tmp_path)
    assert reloaded.read_scope(scope.repository_scope_id) == scope
    assert reloaded.read_task_success(success.evidence_id) == success
    assert reloaded.read_candidate(candidate.candidate_id) == candidate


def test_identical_writes_are_idempotent_and_conflicts_do_not_overwrite(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    scope = _scope()
    candidate = _candidate(scope.repository_scope_id)
    assert store.write_scope(scope) is ImmutableWriteStatus.CREATED
    assert store.write_scope(scope) is ImmutableWriteStatus.UNCHANGED
    assert store.write_candidate(candidate) is ImmutableWriteStatus.CREATED
    assert store.write_candidate(candidate) is ImmutableWriteStatus.UNCHANGED
    changed = _candidate(scope.repository_scope_id, "Different reusable content")
    assert store.write_candidate(changed) is ImmutableWriteStatus.CONFLICT
    assert store.read_candidate(candidate.candidate_id) == candidate


def test_same_scope_id_with_different_manifest_is_a_conflict(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    scope = _scope()
    changed_evidence = (("project_key", "different"),)
    changed_payload = {
        "schema": scope.schema,
        "schema_version": scope.schema_version,
        "repository_scope_id": scope.repository_scope_id,
        "method": scope.method.value,
        "evidence": [list(item) for item in changed_evidence],
    }
    conflicting = RepositoryScopeIdentity(
        repository_scope_id=scope.repository_scope_id,
        method=scope.method,
        evidence=changed_evidence,
        structural_digest=structural_digest(changed_payload),
    )
    assert store.write_scope(scope) is ImmutableWriteStatus.CREATED
    assert store.write_scope(conflicting) is ImmutableWriteStatus.CONFLICT
    assert store.read_scope(scope.repository_scope_id) == scope


@pytest.mark.parametrize("corruption", ["{", "[]"])
def test_malformed_records_fail_closed(tmp_path: Path, corruption: str) -> None:
    store = KnowledgeRecordStore(tmp_path)
    candidate = _candidate(_scope().repository_scope_id)
    assert store.write_candidate(candidate) is ImmutableWriteStatus.CREATED
    path = store.candidates / f"{candidate.candidate_id}.json"
    path.write_text(corruption, encoding="utf-8")
    with pytest.raises(KnowledgeStoreError):
        store.read_candidate(candidate.candidate_id)


def test_schema_and_digest_corruption_fail_closed(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    candidate = _candidate(_scope().repository_scope_id)
    store.write_candidate(candidate)
    path = store.candidates / f"{candidate.candidate_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KnowledgeStoreError):
        store.read_candidate(candidate.candidate_id)


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    store = KnowledgeRecordStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe path"):
        store.read_candidate("../escape")
    with pytest.raises(ValueError, match="unsafe path"):
        store.read_task_success("a/b")


def test_p3_package_has_no_runtime_authority_dependencies() -> None:
    package = Path(__file__).parents[1] / "pico" / "knowledge_evolution"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    forbidden = (
        "pico.providers",
        "pico.agent.tools",
        "pico.delivery",
        "pico.memory_engine",
        "workspace_restore",
        "selective_rewind",
    )
    assert all(name not in source for name in forbidden)
