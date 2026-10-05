from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.picobench.packs.knowledge_evolution_live import (
    jev6_benchmark as benchmark,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    SELECTOR_CONFIG_DIGEST,
    TASKS,
)
from benchmarks.picobench.packs.knowledge_evolution_live.selectivity import (
    evaluate_candidate_identity_audit,
)
from pico.knowledge_evolution import (
    CandidateEvidenceIdentity,
    KnowledgeLifecycleManager,
    KnowledgeRecordStore,
    ordered_candidate_evidence_equal,
)
from pico.knowledge_evolution.types import structural_digest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT
    / "benchmarks/picobench/packs/knowledge_evolution_live/jev6_frozen_manifest.json"
)
LEDGER = (
    ROOT
    / "benchmarks/picobench/packs/knowledge_evolution_live/jev6_exposure_ledger.json"
)
CAMPAIGN = ROOT / ".p3r/jev6b2-ec0eb2cafd43defe"
RUNTIME_STATE = CAMPAIGN / "s/r01"
RUN = (
    CAMPAIGN
    / "runs/jev6-nav-02-skill-variants-task_relevance_v1-rep1.json"
)


@pytest.fixture(scope="module")
def identity_audit() -> dict[str, object]:
    temporary_root = ROOT / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jev6f-", dir=temporary_root) as root:
        yield evaluate_candidate_identity_audit(
            state_root=Path(root),
            workspace=ROOT,
            tasks=TASKS,
            reviewer_id="human:jev6f-offline",
        )


def _identities(value: dict[str, object]) -> tuple[CandidateEvidenceIdentity, ...]:
    return tuple(
        CandidateEvidenceIdentity.from_dict(item)
        for item in value["ordered_selected_identities"]
    )


def _runtime_nav02_identities(
    identity_audit: dict[str, object],
) -> tuple[CandidateEvidenceIdentity, ...]:
    run = json.loads(RUN.read_text(encoding="utf-8"))
    catalog = {
        item["candidate_id"]: item["portable_label"]
        for task in identity_audit["tasks"].values()
        for item in task["ordered_selected_identities"]
    }
    store = KnowledgeRecordStore(RUNTIME_STATE)
    values = []
    for candidate_id in run["relevance_selected_candidate_ids"]:
        candidate = store.read_candidate(candidate_id)
        assert candidate is not None
        lifecycle = KnowledgeLifecycleManager(store).rebuild(candidate_id)
        values.append(
            CandidateEvidenceIdentity.from_candidate(
                candidate,
                lifecycle_state=lifecycle.state,
                portable_label=catalog[candidate_id],
            )
        )
    return tuple(values)


def test_frozen_portable_labels_map_to_production_durable_ids(
    identity_audit: dict[str, object],
) -> None:
    nav02 = identity_audit["tasks"]["jev6-nav-02-skill-variants"]
    identities = _identities(nav02)
    assert tuple(item.portable_label for item in identities) == (
        "scoped-skill-flow",
        "portable-artifact-paths",
    )
    assert tuple(item.candidate_id for item in identities) == (
        "candidate-2efd7bff19eb5245f9efb1fef44346c5f135aaef04319fe1c55bba5cd09038b1",
        "candidate-e724ec2089470b836dff8aa6ac4a283fe2ba8253d078f8d1a6d5feab0a310a68",
    )


def test_runtime_hash_id_uses_exact_production_construction_inputs() -> None:
    store = KnowledgeRecordStore(RUNTIME_STATE)
    candidate = store.read_candidate(
        "candidate-2efd7bff19eb5245f9efb1fef44346c5f135aaef04319fe1c55bba5cd09038b1"
    )
    assert candidate is not None
    payload = {
        "repository_scope_id": candidate.repository_scope_id,
        "source_turn_id": candidate.qualifying_sources[0].turn_id,
        "policy": candidate.extraction_policy,
        "policy_version": candidate.extraction_policy_version,
        "candidate_type": candidate.candidate_type.value,
        "content_class": candidate.content_class.value,
        "title": candidate.title,
        "content": candidate.reusable_content,
        "preconditions": list(candidate.preconditions),
        "applicability_fingerprints": [
            list(item) for item in candidate.applicability_fingerprints
        ],
    }
    assert candidate.candidate_id == f"candidate-{structural_digest(payload)}"


def test_nav02_runtime_and_offline_selector_are_semantically_identical(
    identity_audit: dict[str, object],
) -> None:
    expected = _identities(
        identity_audit["tasks"]["jev6-nav-02-skill-variants"]
    )
    actual = _runtime_nav02_identities(identity_audit)
    assert ordered_candidate_evidence_equal(expected, actual)
    assert tuple(item.payload_digest for item in expected) == tuple(
        item.payload_digest for item in actual
    )
    assert tuple(item.source_identity_digest for item in expected) == tuple(
        item.source_identity_digest for item in actual
    )


def test_all_eight_frozen_sets_match_semantics_but_not_raw_id_representation(
    identity_audit: dict[str, object],
) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["selector_audit"][
        "selected_candidate_ids"
    ]
    for task in TASKS:
        row = identity_audit["tasks"][task.task_id]
        labels = tuple(row["selected_portable_labels"])
        raw_ids = tuple(row["selected_candidate_ids"])
        assert set(labels) == set(frozen[task.task_id])
        assert (raw_ids == tuple(frozen[task.task_id])) is (not raw_ids)


def test_nav02_score_rank_and_decisions_match_persisted_runtime(
    identity_audit: dict[str, object],
) -> None:
    expected = {
        item["candidate_id"]: (
            item["rank"],
            item["relevance_score"],
            tuple(item["meaningful_overlap"]),
            tuple(item["identifier_overlap"]),
            item["decision"],
        )
        for item in identity_audit["tasks"]["jev6-nav-02-skill-variants"][
            "selection_evidence"
        ]
    }
    store = KnowledgeRecordStore(RUNTIME_STATE)
    selected_ids = set(json.loads(RUN.read_text(encoding="utf-8"))["relevance_selected_candidate_ids"])
    runtime = {
        item.candidate_id: (
            item.rank,
            item.relevance_score,
            item.meaningful_overlap,
            item.identifier_overlap,
            item.decision.value,
        )
        for item in store.list_relevance_selections()
        if item.candidate_id in selected_ids
    }
    assert {candidate_id: expected[candidate_id] for candidate_id in selected_ids} == runtime
    assert {
        item["selector_config_digest"]
        for item in identity_audit["tasks"]["jev6-nav-02-skill-variants"][
            "selection_evidence"
        ]
    } == {SELECTOR_CONFIG_DIGEST}


def test_candidate_identity_json_round_trip(identity_audit: dict[str, object]) -> None:
    identity = _identities(
        identity_audit["tasks"]["jev6-nav-02-skill-variants"]
    )[0]
    reloaded = CandidateEvidenceIdentity.from_dict(
        json.loads(json.dumps(identity.to_dict()))
    )
    assert reloaded == identity


def test_campaign_fairness_uses_canonical_evidence_when_available(
    identity_audit: dict[str, object],
) -> None:
    task_id = "jev6-nav-02-skill-variants"
    evidence = identity_audit["tasks"][task_id]["ordered_selected_identities"]
    manifest = {
        "frozen_selected_candidate_ids": {task_id: ("presentation-only",)},
        "frozen_selected_candidate_evidence": {task_id: evidence},
    }
    record = {
        "task_id": task_id,
        "relevance_selected_candidate_ids": ("different-presentation",),
        "relevance_selected_candidate_evidence": evidence,
    }
    assert benchmark._selection_identity_matches(manifest, record)
    changed = [dict(item) for item in evidence]
    changed[0]["payload_digest"] = structural_digest("changed")
    assert not benchmark._selection_identity_matches(
        manifest, {**record, "relevance_selected_candidate_evidence": changed}
    )


def test_identity_mutations_and_ambiguity_fail_closed(
    identity_audit: dict[str, object],
) -> None:
    identity = _identities(
        identity_audit["tasks"]["jev6-nav-02-skill-variants"]
    )[0]
    digest_fields = (
        "payload_digest",
        "source_identity_digest",
        "repository_scope_id",
        "preconditions_digest",
    )
    for field in digest_fields:
        with pytest.raises(ValueError, match="semantic identity digest mismatch"):
            replace(identity, **{field: structural_digest(field + "-changed")})
    with pytest.raises(ValueError, match="semantic identity digest mismatch"):
        replace(identity, payload_digest=structural_digest("different payload"))
    relabelled = replace(identity, portable_label="debug-only-label")
    assert relabelled.semantic_digest == identity.semantic_digest
    other = _identities(
        identity_audit["tasks"]["jev6-nav-02-skill-variants"]
    )[1]
    assert not ordered_candidate_evidence_equal((identity, other), (other, identity))
    with pytest.raises(ValueError, match="ambiguous"):
        ordered_candidate_evidence_equal((identity, identity), (identity, identity))


def test_historical_artifacts_and_exposure_ledger_are_frozen() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    invalid = json.loads(RUN.read_text(encoding="utf-8"))
    reduced = json.loads((CAMPAIGN / "reduced.json").read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert frozen["anti_tuning_digests"]["corpus"] == (
        "39b04fca4f6aa9e54568594b1c18dfe9f71c2292c643469197a96c8977368fbf"
    )
    assert invalid["integrity_digest"] == (
        "1d4fdef963967559f9d3d2af9a67b3648c0ef4a0036ddfb4acaa25979e59c6e7"
    )
    assert reduced["integrity_digest"] == (
        "4daf1e9f7b4954992a6462c256579a94b6d4fc132d08dbdfcae18ecaafb56081"
    )
    assert ledger["tasks"]["jev6-nav-02-skill-variants"]["live_agent_exposed"] is True
    assert sum(
        row["live_agent_exposed"] for row in ledger["tasks"].values()
    ) == 1
    assert ledger["pristine_confirmatory_reuse_allowed"] is False
