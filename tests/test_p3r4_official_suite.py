from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live.artifacts import (
    load_manifest,
    load_run,
    load_runs,
)
from benchmarks.picobench.packs.knowledge_evolution_live.contracts import (
    EXPLICITLY_SUPPORTED,
    audit_official_v2_contracts,
)
from benchmarks.picobench.packs.knowledge_evolution_live.knowledge import CORPUS
from benchmarks.picobench.packs.knowledge_evolution_live.official_fixtures import (
    OFFICIAL_REFERENCE_BASE,
    OFFICIAL_SUITE_VERSION,
    OFFICIAL_SUITE_VERSION_V1,
    official_reference_digests,
)
from benchmarks.picobench.packs.knowledge_evolution_live.protocol import create_manifest
from benchmarks.picobench.packs.knowledge_evolution_live.reducer import reduce_campaign
from benchmarks.picobench.packs.knowledge_evolution_live.schema import (
    Arm,
    CampaignMode,
    CampaignPaths,
)
from benchmarks.picobench.packs.knowledge_evolution_live.selectivity import (
    evaluate_official_selectivity,
)
from benchmarks.picobench.packs.knowledge_evolution_live.solvability import (
    audit_nav03_semantic_fixtures,
    audit_official_solvability,
)
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import (
    EXPLORATORY_TASKS,
    OFFICIAL_TASKS,
    OFFICIAL_TASKS_V1,
    OFFICIAL_TASKS_V2,
)
from benchmarks.picobench.packs.knowledge_evolution_live.verifiers import (
    VERIFIERS,
    OfficialVerifierSpec,
    changed_path_findings,
)


def _manifest(mode: CampaignMode):
    return create_manifest(
        mode=mode,
        base_commit_sha=OFFICIAL_REFERENCE_BASE,
        seed=17,
        provider_id="fixture",
        actual_model_id="fixture/model",
        provider_model_config_digest="a" * 64,
        tool_config_digest="b" * 64,
        runtime_config_digest="c" * 64,
        created_at="2026-10-03T00:00:00Z",
    )


def test_official_suite_is_exactly_twelve_balanced_held_out_tasks() -> None:
    assert canonical_digest(EXPLORATORY_TASKS) == "5daf4a6049414ddb9921e3104dca6f12c17464d7954b5e91e95436569e3b8a9b"
    assert canonical_digest(OFFICIAL_TASKS_V1) == "386d1f94c32771eca8ef0ea66304a778098b5708894cd3af51a1a977f81a6f61"
    assert canonical_digest(OFFICIAL_TASKS_V2) == "47f31ddb9ecc52c6f4e7a108110f12b3b94bdb95905925c992a155ce6b5d3798"
    assert OFFICIAL_TASKS is OFFICIAL_TASKS_V2
    assert len(OFFICIAL_TASKS) == len({item.task_id for item in OFFICIAL_TASKS}) == 12
    assert Counter(item.category for item in OFFICIAL_TASKS) == {
        "repository_navigation": 3,
        "implementation_extension": 3,
        "debugging_recovery": 3,
        "cross_module_integration": 3,
    }
    assert not {item.task_id for item in OFFICIAL_TASKS} & {
        "p3r-nav-01",
        "p3r-debug-01",
        "p3r-int-01",
    }


def test_official_prompts_do_not_leak_oracles_or_corpus_titles() -> None:
    titles = tuple(item.proposal.title.casefold() for item in CORPUS)
    for task in OFFICIAL_TASKS:
        prompt = task.prompt.casefold()
        assert task.task_id.casefold() not in prompt
        assert "no_reuse" not in prompt and "approved_reuse" not in prompt
        assert "verifier" not in prompt and "fixture" not in prompt
        assert "previous arm" not in prompt and "candidate title" not in prompt
        assert all(title not in prompt for title in titles)


def test_v2_changes_only_four_genuine_contract_defects() -> None:
    old = {task.task_id: task for task in OFFICIAL_TASKS_V1}
    new = {task.task_id: task for task in OFFICIAL_TASKS_V2}
    assert {task_id for task_id in old if old[task_id] != new[task_id]} == {
        "p3r4-nav-01",
        "p3r4-nav-02",
        "p3r4-nav-03",
        "p3r4-int-03",
    }
    assert "list_retrievals" in new["p3r4-nav-01"].prompt
    assert "list_outcome_associations" in new["p3r4-nav-02"].prompt
    nav03 = new["p3r4-nav-03"].prompt
    assert "list_applicability_results" in nav03
    assert "candidate_id" in nav03 and "ApplicabilityStatus" in nav03
    assert "compose" in nav03 and "record/path identity" in nav03
    assert "tests/test_knowledge_retrieval.py followed by tests/test_knowledge_usage.py" in new[
        "p3r4-int-03"
    ].prompt


def test_v1_fixture_and_verifier_identities_remain_frozen() -> None:
    expected = (
        ("p3r4-nav-01", "8b243c2ceed6a658713b212c68fad83b65d4127934a9ac03871a38dd27cb3bce"),
        ("p3r4-nav-02", "74580ef41356600c4670d10366737579fb740388cd43c468c50d823440bd2f5a"),
        ("p3r4-nav-03", "a78bea533401a56a39143a3d071410d0418b1bdca829d085bbc54181b1ed983c"),
        ("p3r4-impl-01", "ef12eb1313851a842013918904783c643548780373ac056fd3fd9966dc63a4b6"),
        ("p3r4-impl-02", "cffbc9cd369562c89925735b68ebb061f4bab464d80c1d73e42373c47185fc92"),
        ("p3r4-impl-03", "a5f7bd8d1dfb500f8211deb93f3877cbf941445c3c05412ee293f9e4f90afc3a"),
        ("p3r4-debug-01", "517c4a49860779a916ad0fc653738583b39810c033229197ba45dcb75f92b149"),
        ("p3r4-debug-02", "cfb04168738371a03709882788c8076e2c2abe83780bbd7b54afe472046d4827"),
        ("p3r4-debug-03", "7d540856d8ecdfc59a48d19a26258ca3d7cb2a9cc3febf1c28551b9c9a019601"),
        ("p3r4-int-01", "bbdc31edc3726f90fe215b09678da5a3b24267d90772f2c166dd101491657b1f"),
        ("p3r4-int-02", "3a1193f1b158143e717ce9d5840a304ec8d7cb47b835dd35820a36771f249b52"),
        ("p3r4-int-03", "c774fc19235dec2c15c8cf3272da50dc3289b59a8034ff0ec616670cb939227c"),
    )
    assert official_reference_digests(suite_version=OFFICIAL_SUITE_VERSION_V1) == expected
    assert canonical_digest(tuple((task.task_id, task.verifier_digest) for task in OFFICIAL_TASKS_V1)) == (
        "baf2abeea35b8e91d8935c7ed227804dcccd0134ba56d8f9724f22e4b6f5d63e"
    )


def test_all_v2_contract_audit_rows_are_supported() -> None:
    rows = audit_official_v2_contracts()
    assert canonical_digest(rows) == "0820b66c7fca8f689cd23533b08e6a9d1a788e362ac1819271c6cbdc75347109"
    assert len(rows) == 12
    assert {row.task_id for row in rows} == {task.task_id for task in OFFICIAL_TASKS_V2}
    assert all(row.status == EXPLICITLY_SUPPORTED for row in rows)
    assert all(row.verifier_check and row.known_valid_fixture_behavior for row in rows)
    assert all(not row.hidden_requirements for row in rows)


def test_each_official_task_binds_a_sealed_semantic_verifier_and_regression_path() -> None:
    for task in OFFICIAL_TASKS:
        spec = VERIFIERS[task.verifier_id]
        assert isinstance(spec, OfficialVerifierSpec)
        assert spec.semantic_probe.strip()
        assert spec.targeted_tests
        assert spec.digest == task.verifier_digest
        production_only = tuple(path for path in spec.required_changed_paths if not path.startswith("tests/"))
        assert changed_path_findings(spec, production_only)
        representative = tuple(path.replace("*", "test_reference.py") for path in spec.required_changed_paths)
        assert not changed_path_findings(spec, representative)


def test_official_manifest_freezes_suite_selector_fixtures_and_selective_arms() -> None:
    single = _manifest(CampaignMode.OFFICIAL_SINGLE)
    repeat = _manifest(CampaignMode.OFFICIAL_REPEAT2)
    assert single.suite_version == OFFICIAL_SUITE_VERSION
    assert single.mechanical_solvability_digests == official_reference_digests()
    assert single.selector_mode == "task_relevance_v1"
    assert single.selector_version == 1
    assert single.selector_config_digest == "eb4cfa4573a814b1d10f3849c9cc7280dbf06df4be54c84ea99bed461af9af40"
    assert single.schema_version == 5
    assert len(single.planned_runs) == 24
    assert len(repeat.planned_runs) == 48
    for task in OFFICIAL_TASKS:
        first = tuple(item.arm for item in repeat.planned_runs if item.task_id == task.task_id and item.repetition == 1)
        second = tuple(
            item.arm for item in repeat.planned_runs if item.task_id == task.task_id and item.repetition == 2
        )
        assert set(first) == {Arm.NO_REUSE, Arm.APPROVED_REUSE_SELECTIVE}
        assert second == tuple(reversed(first))


def test_official_reference_fixtures_pass_twelve_of_twelve() -> None:
    audits = audit_official_solvability(Path.cwd())
    assert len(audits) == 12
    assert all(item.passed for item in audits), audits
    assert all(item.changed_path_count >= 2 for item in audits)
    assert all(item.changed_line_count > 0 for item in audits)
    assert len({item.reference_digest for item in audits}) == 12


def test_nav03_semantic_alternatives_pass_and_contract_defects_fail() -> None:
    audits = {item.fixture_id: item for item in audit_nav03_semantic_fixtures(Path.cwd())}
    assert audits.keys() == {
        "canonical",
        "alternate-helper",
        "missing-api",
        "broken-candidate",
        "broken-status",
        "noncomposable",
        "unordered",
        "broken-identity",
    }
    assert all(item.audit_passed for item in audits.values()), audits
    assert audits["canonical"].verifier_passed
    assert audits["alternate-helper"].verifier_passed
    assert all(
        not audits[name].verifier_passed
        for name in audits.keys() - {"canonical", "alternate-helper"}
    )


def test_official_selector_audit_is_deterministic_and_exercises_zero_one_many() -> None:
    with tempfile.TemporaryDirectory(prefix="p3r4-selector-v1-") as old_root:
        old = evaluate_official_selectivity(
            state_root=Path(old_root), workspace=Path.cwd(), tasks=OFFICIAL_TASKS_V1
        )
    with tempfile.TemporaryDirectory(prefix="p3r4-selector-v2-") as new_root:
        result = evaluate_official_selectivity(
            state_root=Path(new_root), workspace=Path.cwd(), tasks=OFFICIAL_TASKS_V2
        )
    assert old["semantic_digest"] == "87cc6b88d833239c5c8d2f1c196f08ddeea9c9d280ef4a215d64ca8fa2474f8c"
    assert result["corpus_digest"] == old["corpus_digest"] == (
        "39b04fca4f6aa9e54568594b1c18dfe9f71c2292c643469197a96c8977368fbf"
    )
    old_tasks = old["modes"]["task_relevance_v1"]["tasks"]
    tasks = result["modes"]["task_relevance_v1"]["tasks"]
    counts = {item["selected_count"] for item in tasks.values()}
    assert 0 in counts and 1 in counts and any(value > 1 for value in counts)
    assert all(item["selected_count"] + item["abstained_count"] == 6 for item in tasks.values())
    assert all(item["approximate_context_tokens"] >= 0 for item in tasks.values())
    assert result["semantic_digest"] == "482fc94400c449c97e03d28a3addc1f56d3d5298035a4b767ccedd79a18f92ad"
    changed = {
        task_id
        for task_id in tasks
        if tasks[task_id]["selected_candidate_ids"]
        != old_tasks[task_id]["selected_candidate_ids"]
    }
    assert changed == {"p3r4-nav-02", "p3r4-nav-03"}
    assert old_tasks["p3r4-nav-02"]["selected_candidate_ids"] == (
        "fail-closed-recovery",
        "immutable-record-conventions",
        "structured-evidence-first",
    )
    assert tasks["p3r4-nav-02"]["selected_candidate_ids"] == (
        "fail-closed-recovery",
        "immutable-record-conventions",
    )
    assert old_tasks["p3r4-nav-03"]["selected_candidate_ids"] == (
        "immutable-record-conventions",
    )
    assert tasks["p3r4-nav-03"]["selected_candidate_ids"] == (
        "fail-closed-recovery",
        "immutable-record-conventions",
    )


def test_historical_official_campaign_still_loads_and_reduces_unchanged() -> None:
    paths = CampaignPaths.at(Path(".p3r/p3r-543e189ff8cfbbad"))
    if not paths.manifest.exists():
        pytest.skip("immutable local historical campaign is not present")
    manifest = load_manifest(paths.manifest)
    reduced = reduce_campaign(manifest, load_runs(paths))
    assert manifest.suite_version == OFFICIAL_SUITE_VERSION_V1
    assert manifest.task_prompt_digests == tuple(
        (task.task_id, task.prompt_digest) for task in OFFICIAL_TASKS_V1
    )
    assert manifest.verifier_digests == tuple(
        (task.task_id, task.verifier_digest) for task in OFFICIAL_TASKS_V1
    )
    assert reduced["classification"] == "regressive"
    assert reduced["success_rates"]["no_reuse"] == 1.0
    assert reduced["success_rates"]["approved_reuse_selective"] == 11 / 12
    assert reduced["planned_runs"] == reduced["completed_runs"] == 24


def test_historical_v1_through_v5_artifact_shapes_remain_readable(tmp_path: Path) -> None:
    historical = CampaignPaths.at(Path(".p3r/p3r-543e189ff8cfbbad"))
    if not historical.manifest.exists():
        pytest.skip("immutable local historical campaign is not present")
    manifest = load_manifest(historical.manifest)
    record = load_run(next(iter(sorted(historical.runs.glob("*.json")))))
    for version in range(1, 6):
        candidate_manifest = replace(manifest, schema_version=version, manifest_digest="")
        manifest_payload = candidate_manifest._payload()
        manifest_payload["manifest_digest"] = canonical_digest(manifest_payload)
        manifest_path = tmp_path / f"manifest-v{version}.json"
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        assert load_manifest(manifest_path).schema_version == version

        candidate_record = replace(record, schema_version=version, integrity_digest="")
        record_payload = candidate_record._payload()
        record_payload["integrity_digest"] = canonical_digest(record_payload)
        record_path = tmp_path / f"run-v{version}.json"
        record_path.write_text(json.dumps(record_payload), encoding="utf-8")
        assert load_run(record_path).schema_version == version


def test_official_harness_and_fixtures_are_absent_from_evaluated_base() -> None:
    for relative in (
        "benchmarks/picobench/packs/knowledge_evolution_live/verifiers.py",
        "benchmarks/picobench/packs/knowledge_evolution_live/official_fixtures.py",
    ):
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{OFFICIAL_REFERENCE_BASE}:{relative}"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
