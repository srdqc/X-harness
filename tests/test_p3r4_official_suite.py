from __future__ import annotations

import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest
from benchmarks.picobench.packs.knowledge_evolution_live.knowledge import CORPUS
from benchmarks.picobench.packs.knowledge_evolution_live.official_fixtures import (
    OFFICIAL_REFERENCE_BASE,
    OFFICIAL_SUITE_VERSION,
    official_reference_digests,
)
from benchmarks.picobench.packs.knowledge_evolution_live.protocol import create_manifest
from benchmarks.picobench.packs.knowledge_evolution_live.schema import Arm, CampaignMode
from benchmarks.picobench.packs.knowledge_evolution_live.selectivity import (
    evaluate_official_selectivity,
)
from benchmarks.picobench.packs.knowledge_evolution_live.solvability import (
    audit_official_solvability,
)
from benchmarks.picobench.packs.knowledge_evolution_live.tasks import (
    EXPLORATORY_TASKS,
    OFFICIAL_TASKS,
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
    assert canonical_digest(OFFICIAL_TASKS) == "386d1f94c32771eca8ef0ea66304a778098b5708894cd3af51a1a977f81a6f61"
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
        assert all(title not in prompt for title in titles)


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
    assert len(single.selector_config_digest) == 64
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


def test_official_selector_audit_is_deterministic_and_exercises_zero_one_many() -> None:
    with tempfile.TemporaryDirectory(prefix="p3r4-selector-") as temporary:
        result = evaluate_official_selectivity(state_root=Path(temporary), workspace=Path.cwd())
    tasks = result["modes"]["task_relevance_v1"]["tasks"]
    counts = {item["selected_count"] for item in tasks.values()}
    assert 0 in counts and 1 in counts and any(value > 1 for value in counts)
    assert all(item["selected_count"] + item["abstained_count"] == 6 for item in tasks.values())
    assert all(item["approximate_context_tokens"] >= 0 for item in tasks.values())
    assert result["semantic_digest"] == "87cc6b88d833239c5c8d2f1c196f08ddeea9c9d280ef4a215d64ca8fa2474f8c"


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
