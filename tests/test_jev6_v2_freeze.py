from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live import jev4_pilot, jev5a_probe
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    BENEFIT_CRITERIA as V1_BENEFIT_CRITERIA,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    RUN_ORDER_DIGEST as V1_RUN_ORDER_DIGEST,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    TASKS as V1_TASKS,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_suite import (
    UTILITY_POLICY as V1_UTILITY_POLICY,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v2_fixtures import reference_digests
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v2_freeze import (
    audit_mechanical_solvability,
    evaluate_selector_identity_audit,
    load_exposure_snapshot,
    solvability_payload,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v2_suite import (
    CONTRACT_AUDIT,
    NEW_TASK,
    NEW_TASK_ID,
    RETAINED_TASK_IDS,
    RETIRED_TASK_ID,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    SUITE_VERSION,
    TASKS,
    Arm,
    Category,
    suite_payload,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v2_verifiers import SPECS
from pico.knowledge_evolution import CandidateEvidenceIdentity

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/picobench/packs/knowledge_evolution_live"
FROZEN = PACK / "jev6_v2_frozen_manifest.json"


@pytest.fixture(scope="module")
def selector_audit() -> dict[str, object]:
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="j6v2s-", dir=ROOT / ".tmp") as state:
        yield evaluate_selector_identity_audit(state_root=Path(state), workspace=ROOT)


@pytest.fixture(scope="module")
def mechanical_audit() -> dict[str, object]:
    return solvability_payload(audit_mechanical_solvability(ROOT))


def test_v2_has_exactly_seven_unchanged_retained_tasks_and_one_new_nav() -> None:
    assert SUITE_VERSION == "jev6-held-out-v2"
    assert len(TASKS) == len({task.task_id for task in TASKS}) == 8
    assert Counter(task.category for task in TASKS) == {category.value: 2 for category in Category}
    v1 = {task.task_id: task for task in V1_TASKS}
    assert tuple(task_id for task_id in RETAINED_TASK_IDS if v1[task_id] in TASKS) == RETAINED_TASK_IDS
    assert all(next(task for task in TASKS if task.task_id == task_id) == v1[task_id] for task_id in RETAINED_TASK_IDS)
    assert RETIRED_TASK_ID not in {task.task_id for task in TASKS}
    assert NEW_TASK in TASKS and NEW_TASK_ID not in {task.task_id for task in V1_TASKS}
    historical = json.dumps(jev4_pilot.TASKS, default=str) + json.dumps(jev5a_probe.CASES, default=str)
    assert NEW_TASK.prompt not in historical


def test_contract_exposure_and_verifier_coverage_are_fail_closed() -> None:
    assert len(CONTRACT_AUDIT) == 8
    assert all(row.status == "EXPLICITLY_SUPPORTED" for row in CONTRACT_AUDIT)
    assert {task.verifier_id for task in TASKS} == {spec.verifier_id for spec in SPECS}
    snapshot = load_exposure_snapshot(ROOT)
    assert set(snapshot["tasks"]) == {task.task_id for task in TASKS}
    assert all(not row["live_agent_exposed"] for row in snapshot["tasks"].values())
    assert snapshot["retired_history"][RETIRED_TASK_ID]["live_agent_exposed"] is True


def test_mechanical_audit_passes_fresh_eight_of_eight(mechanical_audit: dict[str, object]) -> None:
    assert mechanical_audit["passed_count"] == mechanical_audit["total_count"] == 8
    assert all(row["passed"] and not row["findings"] for row in mechanical_audit["results"])
    assert tuple((row["task_id"], row["reference_digest"]) for row in mechanical_audit["results"]) == reference_digests()


def test_canonical_selector_audit_is_deterministic_and_uses_evidence_identity(selector_audit: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="j6v2r-", dir=ROOT / ".tmp") as state:
        repeated = evaluate_selector_identity_audit(state_root=Path(state), workspace=ROOT)
    assert repeated == selector_audit
    assert selector_audit["schema"] == "pico.jev6-selector-identity-audit.v1"
    counts = {task_id: len(row["ordered_selected_identities"]) for task_id, row in selector_audit["tasks"].items()}
    assert counts == json.loads(FROZEN.read_text(encoding="utf-8"))["selector_audit"]["selected_counts"]
    for row in selector_audit["tasks"].values():
        for raw in row["ordered_selected_identities"]:
            identity = CandidateEvidenceIdentity.from_dict(raw)
            assert CandidateEvidenceIdentity.from_dict(json.loads(json.dumps(identity.to_dict()))) == identity
            assert identity.candidate_id.startswith("candidate-")
            assert identity.portable_label not in (identity.candidate_id, identity.semantic_digest)


def test_two_arms_three_repetitions_and_new_full_run_order_are_frozen() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert len(tuple(Arm)) == 2
    assert len(RUN_ORDER) == 48
    assert {(row.task_id, row.arm, row.repetition) for row in RUN_ORDER} == {
        (task.task_id, arm.value, repetition)
        for task in TASKS for arm in Arm for repetition in range(1, 4)
    }
    assert frozen["run_order_ids"] == [row.run_id for row in RUN_ORDER]
    assert RUN_ORDER_DIGEST == canonical_digest(RUN_ORDER)
    assert RUN_ORDER_DIGEST != V1_RUN_ORDER_DIGEST


def test_v2_freeze_digests_match_recomputed_payload(selector_audit: dict[str, object], mechanical_audit: dict[str, object]) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    payload = suite_payload(
        selector_audit=selector_audit,
        mechanical_audit=mechanical_audit,
        exposure_snapshot=load_exposure_snapshot(ROOT),
    )
    expected = {
        "suite": payload["semantic_digest"],
        "task_set": payload["task_set_digest"],
        "verifier_set": payload["verifier_set_digest"],
        "fixtures": payload["fixture_set_digest"],
        "mechanical_audit": payload["mechanical_audit_digest"],
        "contract_audit": payload["contract_audit_digest"],
        "selector_audit": payload["selector_audit_digest"],
        "canonical_selector_identities": payload["canonical_selector_identities_digest"],
        "corpus": payload["corpus_digest"],
        "selector_config": payload["selector_config_digest"],
        "utility_prompt": payload["utility_prompt_digest"],
        "utility_inference_policy": payload["utility_inference_policy_digest"],
        "provider_model": payload["provider_model_digest"],
        "treatment_config": payload["treatment_config_digest"],
        "run_order": payload["run_order_digest"],
        "classification_policy": payload["benefit_criteria_digest"],
        "exposure_ledger_snapshot": payload["exposure_snapshot_digest"],
    }
    assert frozen["anti_tuning_digests"] == expected
    assert frozen["provider_calls"] == frozen["agent_turns"] == 0
    assert V1_UTILITY_POLICY == payload["utility_inference_policy"]
    assert canonical_digest(V1_BENEFIT_CRITERIA) == payload["benefit_criteria_digest"]


def test_historical_v1_and_invalid_campaign_artifacts_are_immutable() -> None:
    expected = {
        PACK / "jev6_frozen_manifest.json": "beec91ec78609b183c59736f5220b97ac091f275e766c1db9b7678b3e8f8fa3c",
        PACK / "jev6_exposure_ledger.json": "7f078f480a23aa9bdd3e1b588292be2b20ee12d8171cbdc8c2e5276365e1b437",
        ROOT / ".p3r/jev6b-041ed8144ffddbd2/manifest.json": "44dc9c5298641afc9b789e9d5208dfed2786bd6dce39df87b3dcfe1f6e743198",
        ROOT / ".p3r/jev6b-041ed8144ffddbd2/reduced.json": "0f5899f3eabb76620fe0e9c5bba0d2126d0bc7834af0cc79321b4be25ae15490",
        ROOT / ".p3r/jev6b2-ec0eb2cafd43defe/manifest.json": "290d3bb319349ca5954ebfbbfa05ae749ec6add529e55151e0d552e7498d548e",
        ROOT / ".p3r/jev6b2-ec0eb2cafd43defe/reduced.json": "a249d02ea013183e4416fa43d0c6b6514bb51ea6eeec3a79ccae50348b8bd70c",
    }
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected} == expected
