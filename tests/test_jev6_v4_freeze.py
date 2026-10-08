from __future__ import annotations

import hashlib
import json
import subprocess
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
    UTILITY_POLICY as V1_UTILITY_POLICY,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_suite import (
    RUN_ORDER_DIGEST as V3_RUN_ORDER_DIGEST,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_suite import (
    TASKS as V3_TASKS,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v4_fixtures import (
    reference_digests,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v4_freeze import (
    audit_mechanical_solvability,
    evaluate_selector_identity_audit,
    load_exposure_snapshot,
    solvability_payload,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v4_suite import (
    CONTRACT_AUDIT,
    NEW_TASK_IDS,
    NEW_TASKS,
    PREDECESSOR_SUITE,
    PROVIDER_CONFIG_SOURCE,
    RETAINED_TASK_IDS,
    RETIRED_EXPOSED_TASK_IDS,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    SANDBOX_CONFIG_SOURCE,
    SANDBOX_REQUIREMENT,
    SANDBOX_REQUIREMENT_DIGEST,
    SUITE_SCHEMA,
    SUITE_VERSION,
    TASKS,
    Arm,
    Category,
    suite_payload,
    validate_sandbox_requirement,
)
from benchmarks.picobench.packs.knowledge_evolution_live.jev6_v4_verifiers import SPECS
from pico.knowledge_evolution import CandidateEvidenceIdentity

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/picobench/packs/knowledge_evolution_live"
FROZEN = PACK / "jev6_v4_frozen_manifest.json"


@pytest.fixture(scope="module")
def selector_audit() -> dict[str, object]:
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="j6v4s-", dir=ROOT / ".tmp") as state:
        yield evaluate_selector_identity_audit(state_root=Path(state), workspace=ROOT)


@pytest.fixture(scope="module")
def mechanical_audit() -> dict[str, object]:
    return solvability_payload(audit_mechanical_solvability(ROOT))


def test_v4_has_exact_retained_retired_new_and_category_contract() -> None:
    assert SUITE_SCHEMA == "pico.jev6-heldout-suite.v4"
    assert SUITE_VERSION == "jev6-held-out-v4"
    assert PREDECESSOR_SUITE == "jev6-held-out-v3"
    assert len(TASKS) == len({task.task_id for task in TASKS}) == 8
    assert Counter(task.category for task in TASKS) == {
        category.value: 2 for category in Category
    }
    v3 = {task.task_id: task for task in V3_TASKS}
    assert len(RETAINED_TASK_IDS) == 5
    assert all(next(task for task in TASKS if task.task_id == task_id) == v3[task_id] for task_id in RETAINED_TASK_IDS)
    assert set(RETIRED_EXPOSED_TASK_IDS).isdisjoint(task.task_id for task in TASKS)
    assert len(NEW_TASKS) == len(NEW_TASK_IDS) == 3
    assert set(NEW_TASK_IDS) == {task.task_id for task in NEW_TASKS}
    assert all(task.task_id not in v3 for task in NEW_TASKS)
    assert Counter(task.category for task in NEW_TASKS) == {
        Category.IMPLEMENTATION.value: 1,
        Category.DEBUGGING_VALIDATION.value: 2,
    }
    historical = json.dumps(jev4_pilot.TASKS, default=str) + json.dumps(
        jev5a_probe.CASES, default=str
    )
    assert all(task.prompt not in historical for task in NEW_TASKS)


def test_contract_exposure_and_verifier_coverage_are_fail_closed() -> None:
    assert len(CONTRACT_AUDIT) == 8
    assert all(row.status == "EXPLICITLY_SUPPORTED" for row in CONTRACT_AUDIT)
    assert {task.verifier_id for task in TASKS} == {spec.verifier_id for spec in SPECS}
    snapshot = load_exposure_snapshot(ROOT)
    assert set(snapshot["tasks"]) == {task.task_id for task in TASKS}
    assert all(not row["live_agent_exposed"] for row in snapshot["tasks"].values())
    assert snapshot["v3_exposure_summary"] == {"unique_tasks": 3, "live_runs": 5}
    assert all(snapshot["retired_history"][task_id]["live_agent_exposed"] for task_id in RETIRED_EXPOSED_TASK_IDS)


def test_mechanical_audit_passes_fresh_eight_of_eight(
    mechanical_audit: dict[str, object],
) -> None:
    assert mechanical_audit["passed_count"] == mechanical_audit["total_count"] == 8
    assert all(row["passed"] and not row["findings"] for row in mechanical_audit["results"])
    assert tuple(
        (row["task_id"], row["reference_digest"])
        for row in mechanical_audit["results"]
    ) == reference_digests()


def test_selector_audit_is_deterministic_ordered_and_uses_canonical_identity(
    selector_audit: dict[str, object],
) -> None:
    with tempfile.TemporaryDirectory(prefix="j6v4r-", dir=ROOT / ".tmp") as state:
        repeated = evaluate_selector_identity_audit(state_root=Path(state), workspace=ROOT)
    assert repeated == selector_audit
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    counts = {
        task_id: len(row["ordered_selected_identities"])
        for task_id, row in selector_audit["tasks"].items()
    }
    assert counts == frozen["selector_audit"]["selected_counts"]
    actual_ordered = {
        task_id: [
            [item["candidate_id"], item["semantic_digest"]]
            for item in row["ordered_selected_identities"]
        ]
        for task_id, row in selector_audit["tasks"].items()
    }
    assert actual_ordered == frozen["selector_audit"]["canonical_ordered_identities"]
    assert [task_id for task_id, count in counts.items() if count == 0] == frozen[
        "selector_audit"
    ]["zero_selection_tasks"]
    for row in selector_audit["tasks"].values():
        for raw in row["ordered_selected_identities"]:
            identity = CandidateEvidenceIdentity.from_dict(raw)
            assert identity.schema == "pico.knowledge-candidate-identity.v1"


def test_ordering_drift_regression_remains_active() -> None:
    source = (PACK / "metrics.py").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_picobench_knowledge_evolution_live.py").read_text(
        encoding="utf-8"
    )
    assert "_ordered_selected_candidate_ids" in source
    assert "test_selected_candidate_metrics_preserve_retrieval_order" in tests
    assert "sorted(selected_ids)" not in source


def test_provider_and_sandbox_sources_are_separate_and_frozen() -> None:
    assert PROVIDER_CONFIG_SOURCE == "~/.pico/config.json"
    assert SANDBOX_CONFIG_SOURCE == "frozen_jev6_v4_infrastructure"
    assert PROVIDER_CONFIG_SOURCE != SANDBOX_CONFIG_SOURCE
    assert SANDBOX_REQUIREMENT["backend"] == "boxlite"
    assert SANDBOX_REQUIREMENT["boxlite_version"] == "0.9.5"
    assert SANDBOX_REQUIREMENT["allow_net"] is False
    assert SANDBOX_REQUIREMENT["extra_volumes"] == ()
    assert SANDBOX_REQUIREMENT["image_search_registry"] == "docker.m.daocloud.io"
    assert SANDBOX_REQUIREMENT["native_runtime_home"] == "/tmp/pico-jev6v4-boxlite"
    validate_sandbox_requirement(
        backend="boxlite",
        boxlite_version="0.9.5",
        allow_net=False,
        extra_volumes=[],
        image_search_registry="docker.m.daocloud.io",
        native_runtime_home="/tmp/pico-jev6v4-boxlite",
        smoke_result="PASS_REAL_BOXLITE",
    )
    with pytest.raises(ValueError, match="sandbox identity mismatch"):
        validate_sandbox_requirement(
            backend="none",
            boxlite_version="0.9.5",
            allow_net=False,
            extra_volumes=[],
            image_search_registry="docker.m.daocloud.io",
            native_runtime_home="/tmp/pico-jev6v4-boxlite",
            smoke_result="PASS_REAL_BOXLITE",
        )
    assert SANDBOX_REQUIREMENT_DIGEST == canonical_digest(SANDBOX_REQUIREMENT)


def test_two_arms_three_repetitions_and_new_full_run_order_are_frozen() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert len(tuple(Arm)) == 2
    assert len(RUN_ORDER) == 48
    assert {(row.task_id, row.arm, row.repetition) for row in RUN_ORDER} == {
        (task.task_id, arm.value, repetition)
        for task in TASKS
        for arm in Arm
        for repetition in range(1, 4)
    }
    assert frozen["run_order_ids"] == [row.run_id for row in RUN_ORDER]
    assert RUN_ORDER_DIGEST == canonical_digest(RUN_ORDER)
    assert RUN_ORDER_DIGEST != V3_RUN_ORDER_DIGEST


def test_v4_freeze_digests_match_recomputed_payload(
    selector_audit: dict[str, object],
    mechanical_audit: dict[str, object],
) -> None:
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
        "canonical_corpus_identity": payload["canonical_corpus_identity_digest"],
        "selector_config": payload["selector_config_digest"],
        "utility_prompt": payload["utility_prompt_digest"],
        "utility_inference_policy": payload["utility_inference_policy_digest"],
        "provider_model": payload["provider_model_digest"],
        "treatment_config": payload["treatment_config_digest"],
        "sandbox_config": payload["sandbox_requirement_digest"],
        "run_order": payload["run_order_digest"],
        "classification_policy": payload["benefit_criteria_digest"],
        "exposure_ledger_snapshot": payload["exposure_snapshot_digest"],
    }
    assert frozen["anti_tuning_digests"] == expected
    assert frozen["provider_calls"] == frozen["agent_turns"] == 0
    assert V1_UTILITY_POLICY == payload["utility_inference_policy"]
    assert canonical_digest(V1_BENEFIT_CRITERIA) == payload["benefit_criteria_digest"]


def test_historical_v1_v2_v3_and_invalid_campaign_are_immutable() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    expected = frozen["historical_immutability"]
    files = {
        "jev6_v1_frozen_suite_sha256": PACK / "jev6_frozen_manifest.json",
        "jev6_v1_exposure_ledger_sha256": PACK / "jev6_exposure_ledger.json",
        "jev6_v2_frozen_suite_sha256": PACK / "jev6_v2_frozen_manifest.json",
        "jev6_v2_exposure_snapshot_sha256": PACK / "jev6_v2_exposure_snapshot.json",
        "jev6_v3_frozen_suite_sha256": PACK / "jev6_v3_frozen_manifest.json",
        "jev6_v3_exposure_ledger_sha256": PACK / "jev6_v3_exposure_ledger.json",
        "jev6_v3_exposure_snapshot_sha256": PACK / "jev6_v3_exposure_snapshot.json",
        "jev6v3_invalid_manifest_sha256": ROOT / ".p3r/jev6v3-49a4590ca226cf75/manifest.json",
        "jev6v3_invalid_reduced_sha256": ROOT / ".p3r/jev6v3-49a4590ca226cf75/reduced.json",
        "jev6v3_live_exposure_sha256": ROOT / ".p3r/jev6v3-49a4590ca226cf75/live-exposure-ledger.json",
        "ordering_fix_source_sha256": PACK / "metrics.py",
        "ordering_fix_test_sha256": ROOT / "tests/test_picobench_knowledge_evolution_live.py",
    }
    assert {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()
    } == expected
    assert subprocess.check_output(
        ["git", "rev-parse", "473304a"], cwd=ROOT, text=True
    ).strip() == "473304a788acbce8be02445893435c757dd3455c"
