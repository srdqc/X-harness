"""Frozen three-arm JEV.4 Agent pilot.

Preparation and reduction are offline. Live execution requires the explicit
``--execute-live`` flag and never offers replacement runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_digest, canonical_json, to_primitive
from benchmarks.picobench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.picobench.schema import ExperimentRef
from pico.config.paths import RuntimePaths
from pico.decision_plane.provider_utility import (
    PROVIDER_UTILITY_PROMPT_DIGEST,
    PROVIDER_UTILITY_TEMPLATE_VERSION,
)
from pico.knowledge_evolution import (
    ApplicabilityEnvironment,
    CandidateType,
    KnowledgeRecordStore,
    KnowledgeRetriever,
    KnowledgeSelectionMode,
    RepositoryScopeResolver,
)
from pico.knowledge_evolution.relevance import SELECTOR_VERSION
from pico.spine.message import ChatType, Source
from pico.spine.turn import Origin, TurnRequest
from pico.tracing import evidence

from .knowledge import corpus_digest, prepare_approved_corpus
from .metrics import extract_run_metrics
from .prepare import configuration_identity, resolve_base_commit
from .runner import _changed_paths, _git, _workspace_patch
from .schema import RuntimeBudget
from .tasks import EXPLORATORY_TASKS, OFFICIAL_TASKS_V1, OFFICIAL_TASKS_V2
from .validity import classify_run_validity

SCHEMA = "pico.jev4-agent-pilot.v1"
SCHEMA_VERSION = 1
RUN_SCHEMA = "pico.jev4-agent-run.v1"
UTILITY_TIMEOUT_SECONDS = 15.0
CAMPAIGN_SEED = 40_904
PRIMARY_COMPARISON = ("task_relevance_v1", "task_relevance_v1_provider_utility")
PILOT_CLASSIFICATIONS = {
    "GO_NEW_HELDOUT_BENCHMARK",
    "HOLD_AND_FORENSIC",
    "STOP_PROVIDER_UTILITY",
}


class Arm(StrEnum):
    NO_REUSE = "no_reuse"
    TASK_RELEVANCE_V1 = "task_relevance_v1"
    TASK_RELEVANCE_V1_PROVIDER_UTILITY = "task_relevance_v1_provider_utility"


@dataclass(frozen=True)
class PilotTask:
    task_id: str
    description: str
    prompt: str
    production_path: str
    owner_test: str
    semantic_probe: str
    reference_fixture_id: str
    selection_condition: str

    @property
    def prompt_digest(self) -> str:
        return canonical_digest({"task_id": self.task_id, "prompt": self.prompt})

    @property
    def verifier_digest(self) -> str:
        return canonical_digest(
            {
                "task_id": self.task_id,
                "production_path": self.production_path,
                "owner_test": self.owner_test,
                "semantic_probe": self.semantic_probe,
                "policy": "required-production-and-test-change;no-other-production-change",
            }
        )

    @property
    def reference_digest(self) -> str:
        return canonical_digest(
            {"task_id": self.task_id, "reference_fixture_id": self.reference_fixture_id}
        )


_P1_PROMPT = (
    "In pico/knowledge_evolution/task_success.py, add a public "
    "TaskSuccessEvidence.is_success_for_state(target_state_digest) method. It returns true only "
    "for PASS evidence bound to that exact digest; it returns false for FAIL, INCONCLUSIVE, "
    "unbound, or different-state evidence. Reject malformed digest inputs through the existing "
    "digest contract, preserve canonical serialization, and add focused tests. Limit production "
    "changes to the task-success evidence owner."
)
_P2_PROMPT = (
    "In pico/knowledge_evolution/task_success.py, harden TaskSuccessEvidence provenance_refs "
    "validation to fail closed when a reference is empty, longer than 256 characters, or contains "
    "a NUL character. Preserve accepted unique references, deterministic evidence digests, round "
    "trips, and add positive and negative coverage. Limit production changes to the task-success "
    "evidence owner."
)
_P3_PROMPT = (
    "In pico/spine/turn.py, expose TurnRequest.has_media as a boolean reflecting whether media "
    "contains at least one item. Do not alter dataclass fields, constructor parameters, or "
    "Scheduler behavior. Cover empty and populated media, and limit production changes to the "
    "TurnRequest owner."
)

_P1_PROBE = r'''
from pico.knowledge_evolution import TaskSuccessEvidence, TaskSuccessSource, TaskSuccessStatus
def item(status=TaskSuccessStatus.PASS, target="c"*64):
    return TaskSuccessEvidence.create(evidence_id="e",source_kind=TaskSuccessSource.SEALED_VERIFIER,source_turn_id="t",status=status,producer_id="p",producer_version="1",repository_scope_id="b"*64,target_state_digest=target,result_digest="a"*64,provenance_refs=("artifact:x",),created_at="2026-10-04T00:00:00Z")
assert item().is_success_for_state("c"*64) is True
assert item().is_success_for_state("d"*64) is False
assert item(TaskSuccessStatus.FAIL).is_success_for_state("c"*64) is False
assert item(TaskSuccessStatus.INCONCLUSIVE).is_success_for_state("c"*64) is False
assert item(target=None).is_success_for_state("c"*64) is False
try: item().is_success_for_state("bad")
except ValueError: pass
else: raise AssertionError("malformed digest accepted")
assert "is_success_for_state" not in item().to_dict()
'''
_P2_PROBE = r'''
from pico.knowledge_evolution import TaskSuccessEvidence, TaskSuccessSource, TaskSuccessStatus
def item(refs):
    return TaskSuccessEvidence.create(evidence_id="e",source_kind=TaskSuccessSource.SEALED_VERIFIER,source_turn_id="t",status=TaskSuccessStatus.PASS,producer_id="p",producer_version="1",repository_scope_id="b"*64,target_state_digest="c"*64,result_digest="a"*64,provenance_refs=refs,created_at="2026-10-04T00:00:00Z")
valid=item(("artifact:x","trace:y")); assert TaskSuccessEvidence.from_dict(valid.to_dict()) == valid
for refs in (("",),("x"*257,),("bad\0ref",)):
    try: item(refs)
    except ValueError: pass
    else: raise AssertionError("malformed provenance reference accepted")
'''
_P3_PROBE = r'''
from pico.spine import ChatType, Media, Origin, Source, TurnRequest
source=Source("cli","c","u",ChatType.DM)
empty=TurnRequest(Origin.USER,source,"x")
full=TurnRequest(Origin.USER,source,"x",media=(Media("a.png","image/png","image"),))
assert empty.has_media is False and full.has_media is True
assert tuple(empty.__dataclass_fields__) == ("origin","source","text","media","message_id","conversation","busy")
'''

TASKS = (
    PilotTask("jev4-p1-state-bound-success", "State-bound verified-success helper", _P1_PROMPT,
              "pico/knowledge_evolution/task_success.py", "tests/test_task_success_evidence.py",
              _P1_PROBE, "jev4-ref-state-bound-success-v1", "nonzero_clear"),
    PilotTask("jev4-p2-provenance-guard", "Fail-closed provenance-reference validation", _P2_PROMPT,
              "pico/knowledge_evolution/task_success.py", "tests/test_task_success_evidence.py",
              _P2_PROBE, "jev4-ref-provenance-guard-v1", "nonzero_uncertain_precondition"),
    PilotTask("jev4-p3-turn-media", "TurnRequest media-presence convenience", _P3_PROMPT,
              "pico/spine/turn.py", "tests/test_spine_scheduler.py",
              _P3_PROBE, "jev4-ref-turn-media-v1", "zero_selection_control"),
)

CONTRACT_AUDIT = (
    (TASKS[0].task_id, "named method/truth table/digest rejection/serialization/tests/path", "semantic probe and owner tests"),
    (TASKS[1].task_id, "three malformed classes/valid compatibility/digest/round trip/tests/path", "semantic probe and owner tests"),
    (TASKS[2].task_id, "boolean truth table/wire fields/Scheduler unchanged/tests/path", "semantic probe and owner tests"),
)


def task_by_id(task_id: str) -> PilotTask:
    return next(item for item in TASKS if item.task_id == task_id)


def make_plan(seed: int = CAMPAIGN_SEED) -> tuple[dict[str, Any], ...]:
    arms = tuple(Arm)
    base_offset = seed % len(arms)
    result: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        offset = (base_offset + task_index) % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            result.append(
                {
                    "run_id": f"{task.task_id}-{arm.value}",
                    "task_id": task.task_id,
                    "arm": arm.value,
                    "order": len(result) + 1,
                }
            )
    return tuple(result)


def _selector_digest() -> str:
    return canonical_digest(
        {"selection_mode": KnowledgeSelectionMode.TASK_RELEVANCE_V1.value,
         "selector_version": SELECTOR_VERSION}
    )


def _treatment_identity() -> dict[str, str]:
    return {
        arm.value: canonical_digest(
            {
                "arm": arm.value,
                "corpus": arm is not Arm.NO_REUSE,
                "selector": arm is not Arm.NO_REUSE,
                "provider_utility": arm is Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY,
                "utility_timeout_seconds": (
                    UTILITY_TIMEOUT_SECONDS
                    if arm is Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY else None
                ),
            }
        )
        for arm in Arm
    }


def _semantic_payload(*, base_sha: str, identity: dict[str, str], budget: RuntimeBudget,
                      selected_sets: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "base_commit_sha": base_sha,
        "campaign_seed": CAMPAIGN_SEED,
        "primary_comparison": PRIMARY_COMPARISON,
        "task_order": tuple(task.task_id for task in TASKS),
        "task_prompt_digests": tuple((task.task_id, task.prompt_digest) for task in TASKS),
        "verifier_digests": tuple((task.task_id, task.verifier_digest) for task in TASKS),
        "reference_fixture_digests": tuple((task.task_id, task.reference_digest) for task in TASKS),
        "contract_audit_digest": canonical_digest(CONTRACT_AUDIT),
        "provider_id": identity["provider_id"],
        "actual_model_id": identity["model"],
        "provider_model_config_digest": identity["provider_digest"],
        "tool_config_digest": identity["tool_digest"],
        "runtime_config_digest": identity["runtime_digest"],
        "budget": budget,
        "utility_prompt_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
        "utility_schema_version": PROVIDER_UTILITY_TEMPLATE_VERSION,
        "utility_timeout_seconds": UTILITY_TIMEOUT_SECONDS,
        "knowledge_corpus_digest": corpus_digest(),
        "selector_digest": _selector_digest(),
        "treatment_identity": _treatment_identity(),
        "planned_runs": make_plan(),
        "preflight_relevance_selected_candidate_ids": selected_sets,
        "replacement_policy": "forbidden_stop_on_infra_invalid",
    }


def _store(root: Path) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(root.name, root))


def prepare_campaign(repository: Path, output_root: Path, reviewer_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from pico.config.loader import load_config

    config = load_config()
    budget = RuntimeBudget(
        max_agent_iterations=config.agents.defaults.max_tool_iterations,
        provider_logical_calls_observational=config.agents.defaults.max_tool_iterations,
        tool_calls_observational=40,
        context_window_tokens=config.agents.defaults.context_window_tokens,
    )
    identity = configuration_identity(config, budget)
    if identity["provider_id"] != "deepseek" or not identity["model"].startswith("deepseek/"):
        raise ValueError("JEV.4 requires the configured DeepSeek Agent provider/model")
    base_sha = resolve_base_commit(repository, "HEAD")
    audit = preflight(repository, base_sha, reviewer_id)
    semantic = _semantic_payload(
        base_sha=base_sha,
        identity=identity,
        budget=budget,
        selected_sets=audit["relevance_selected_candidate_ids"],
    )
    semantic_digest = canonical_digest(semantic)
    campaign_id = f"jev4-{semantic_digest[:16]}"
    manifest = {
        **to_primitive(semantic),
        "campaign_id": campaign_id,
        "campaign_semantic_digest": semantic_digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    root = output_root / campaign_id
    _store(root).freeze_manifest(manifest)
    return root, manifest, audit


def preflight(repository: Path, base_sha: str, reviewer_id: str) -> dict[str, Any]:
    old_ids = {item.task_id for item in (*EXPLORATORY_TASKS, *OFFICIAL_TASKS_V1, *OFFICIAL_TASKS_V2)}
    if len(TASKS) != 3 or len({item.task_id for item in TASKS}) != 3 or old_ids & {item.task_id for item in TASKS}:
        raise ValueError("JEV.4 requires three unique non-P3R task IDs")
    old_prompts = {item.prompt for item in (*EXPLORATORY_TASKS, *OFFICIAL_TASKS_V1, *OFFICIAL_TASKS_V2)}
    if old_prompts & {item.prompt for item in TASKS}:
        raise ValueError("JEV.4 prompt reuses an official P3R prompt")
    forbidden = ("verifier", "candidate_id", "expected utility", "arm winner", "jev.3b", "p3r result")
    if any(term in task.prompt.casefold() for task in TASKS for term in forbidden):
        raise ValueError("JEV.4 visible prompt leaks experiment-only information")
    if len(CONTRACT_AUDIT) != 3 or {row[0] for row in CONTRACT_AUDIT} != {item.task_id for item in TASKS}:
        raise ValueError("JEV.4 prompt/verifier contract audit is incomplete")
    plan = make_plan()
    if len(plan) != 9 or Counter(item["arm"] for item in plan) != Counter({arm.value: 3 for arm in Arm}):
        raise ValueError("JEV.4 plan must contain exactly three tasks and three arms")
    if tuple(item["arm"] for item in plan[:3]) == tuple(item["arm"] for item in plan[3:6]):
        raise ValueError("JEV.4 arm order is not counterbalanced")

    solvability: dict[str, str] = {}
    selected_sets: dict[str, tuple[str, ...]] = {}
    with tempfile.TemporaryDirectory(dir=repository / ".tmp") as temp:
        temp_root = Path(temp)
        selection_worktree = temp_root / "sel"
        _git(repository, "worktree", "add", "--detach", str(selection_worktree), base_sha)
        try:
            state = temp_root / "selection-state"
            prepare_approved_corpus(state_root=state, workspace=selection_worktree, reviewer_id=reviewer_id)
            store = KnowledgeRecordStore(state)
            scope = RepositoryScopeResolver(selection_worktree, state).resolve()
            if not scope.resolved or scope.identity is None:
                raise ValueError("JEV.4 preflight repository scope is unresolved")
            for task in TASKS:
                retriever = KnowledgeRetriever(
                    store,
                    ApplicabilityEnvironment(
                        scope.identity.repository_scope_id,
                        selection_worktree,
                        available_tools=("read_file",),
                    ),
                    selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1,
                )
                items, _ = retriever.retrieve(
                    task.prompt,
                    retrieval_id=f"jev4-preflight-{task.task_id}",
                    turn_id=f"jev4-preflight-{task.task_id}",
                    candidate_types=tuple(CandidateType),
                    created_at="2026-10-04T00:00:00Z",
                )
                selected_sets[task.task_id] = tuple(item.candidate.candidate_id for item in items)
        finally:
            if selection_worktree.exists():
                _git(repository, "worktree", "remove", "--force", str(selection_worktree))

        for index, task in enumerate(TASKS):
            worktree = temp_root / f"m{index}"
            _git(repository, "worktree", "add", "--detach", str(worktree), base_sha)
            try:
                _apply_reference_fixture(task, worktree)
                result = verify_workspace(task, worktree, python_executable=sys.executable)
                if not result["passed"]:
                    raise ValueError(f"JEV.4 mechanical solvability failed: {task.task_id}: {result['findings']}")
                solvability[task.task_id] = canonical_digest(
                    {"task_id": task.task_id, "patch": _workspace_patch(worktree), "verifier": result}
                )
            finally:
                if worktree.exists():
                    _git(repository, "worktree", "remove", "--force", str(worktree))

    if not selected_sets[TASKS[0].task_id] or not selected_sets[TASKS[1].task_id]:
        raise ValueError("JEV.4 P1/P2 must have nonzero deterministic selection")
    if selected_sets[TASKS[2].task_id]:
        raise ValueError("JEV.4 P3 must have zero deterministic selection")
    return {
        "task_count": 3,
        "arm_count": 3,
        "planned_agent_runs": 9,
        "primary_comparison": PRIMARY_COMPARISON,
        "contract_audit": "PASS",
        "mechanical_solvability": "3/3 PASS",
        "mechanical_solvability_digests": solvability,
        "relevance_selected_candidate_ids": selected_sets,
        "zero_selection_utility_calls_expected": 0,
        "live_provider_invoked": False,
    }


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise ValueError(f"reference fixture anchor mismatch: {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def _apply_reference_fixture(task: PilotTask, workspace: Path) -> None:
    if task.task_id == TASKS[0].task_id:
        path = workspace / task.production_path
        old = "    @property\n    def has_state_binding(self) -> bool:\n        return self.target_state_digest is not None\n"
        new = old + (
            "\n    def is_success_for_state(self, target_state_digest: str) -> bool:\n"
            "        require_digest(target_state_digest, \"target_state_digest\")\n"
            "        return (\n            self.status is TaskSuccessStatus.PASS\n"
            "            and self.target_state_digest == target_state_digest\n        )\n"
        )
        _replace_once(path, old, new)
    elif task.task_id == TASKS[1].task_id:
        path = workspace / task.production_path
        old = "        if len(self.provenance_refs) != len(set(self.provenance_refs)):\n            raise ValueError(\"provenance_refs must be unique\")\n"
        new = old + (
            "        if any(\n            not isinstance(item, str) or not item.strip() "
            "or len(item) > 256 or \"\\x00\" in item\n            for item in self.provenance_refs\n"
            "        ):\n            raise ValueError(\"provenance_refs must contain bounded non-empty strings\")\n"
        )
        _replace_once(path, old, new)
    else:
        path = workspace / task.production_path
        old = "    busy: BusyPolicy = BusyPolicy.APPEND\n"
        new = old + "\n    @property\n    def has_media(self) -> bool:\n        return bool(self.media)\n"
        _replace_once(path, old, new)
    reference_test = workspace / "tests" / f"test_{task.task_id.replace('-', '_')}_reference.py"
    reference_test.write_text(task.semantic_probe, encoding="utf-8")


def verify_workspace(task: PilotTask, workspace: Path, *, python_executable: str) -> dict[str, Any]:
    changed = _changed_paths(workspace)
    findings: list[str] = []
    if task.production_path not in changed:
        findings.append("production_change_missing")
    changed_tests = tuple(path for path in changed if path.startswith("tests/test_") and path.endswith(".py"))
    if not changed_tests:
        findings.append("regression_test_missing")
    allowed = {task.production_path, *changed_tests}
    if any(path not in allowed for path in changed):
        findings.append("out_of_scope_change")
    try:
        probe = subprocess.run(
            [python_executable, "-c", task.semantic_probe], cwd=workspace,
            check=False, capture_output=True, text=True, timeout=90,
        )
        if probe.returncode != 0:
            findings.append("semantic_contract_failed")
        targets = tuple(dict.fromkeys((task.owner_test, *changed_tests)))
        tests = subprocess.run(
            [python_executable, "-m", "pytest", "-q", *targets], cwd=workspace,
            check=False, capture_output=True, text=True, timeout=300,
        )
        if tests.returncode != 0:
            findings.append("targeted_test_failed")
    except (OSError, subprocess.TimeoutExpired):
        findings.append("verifier_host_failure")
    findings = list(dict.fromkeys(findings))
    return {
        "passed": not findings,
        "findings": tuple(findings),
        "verifier_digest": task.verifier_digest,
        "infrastructure_failure": "verifier_host_failure" in findings,
    }


def load_manifest(root: Path) -> dict[str, Any]:
    value = _store(root).read_json(root / "manifest.json")
    digest = value.pop("manifest_digest", None)
    if digest != canonical_digest(value):
        raise ValueError("JEV.4 manifest digest mismatch")
    value["manifest_digest"] = digest
    if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported JEV.4 manifest")
    semantic = {key: value[key] for key in _semantic_payload_keys()}
    if value["campaign_semantic_digest"] != canonical_digest(semantic):
        raise ValueError("JEV.4 semantic digest mismatch")
    return value


def _semantic_payload_keys() -> tuple[str, ...]:
    return (
        "schema", "schema_version", "base_commit_sha", "campaign_seed", "primary_comparison",
        "task_order", "task_prompt_digests", "verifier_digests", "reference_fixture_digests",
        "contract_audit_digest", "provider_id", "actual_model_id", "provider_model_config_digest",
        "tool_config_digest", "runtime_config_digest", "budget", "utility_prompt_digest",
        "utility_schema_version", "utility_timeout_seconds", "knowledge_corpus_digest",
        "selector_digest", "treatment_identity", "planned_runs",
        "preflight_relevance_selected_candidate_ids", "replacement_policy",
    )


def run_campaign(repository: Path, root: Path, reviewer_id: str, *, execute_live: bool) -> tuple[str, ...]:
    if not execute_live:
        raise PermissionError("JEV.4 live execution requires explicit --execute-live")
    manifest = load_manifest(root)
    run_dir = root / "runs"
    if run_dir.exists() and tuple(run_dir.glob("*.json")):
        raise FileExistsError("JEV.4 is one-shot; existing run records forbid resume or replacement")
    completed: list[str] = []
    for planned in manifest["planned_runs"]:
        record = execute_one(repository, root, manifest, planned, reviewer_id)
        completed.append(record["run_id"])
        if record["run_validity"] == "infra_invalid":
            raise RuntimeError("JEV.4 stopped after INFRA_INVALID; replacement runs are forbidden")
    return tuple(completed)


def execute_one(repository: Path, root: Path, manifest: dict[str, Any], planned: dict[str, Any], reviewer_id: str) -> dict[str, Any]:
    run_id = planned["run_id"]
    run_path = root / "runs" / f"{run_id}.json"
    if run_path.exists():
        raise FileExistsError(f"immutable JEV.4 run already exists: {run_id}")
    short = f"r{planned['order']:02d}"
    worktree, state = root / "w" / short, root / "s" / short
    try:
        _git(repository, "worktree", "add", "--detach", str(worktree), manifest["base_commit_sha"])
    except (OSError, RuntimeError):
        record = _infra_record(manifest, planned, "worktree_setup_failure")
        _write_run(root, record)
        return record
    try:
        try:
            record = asyncio.run(_execute_turn(root, manifest, planned, worktree, state, reviewer_id))
        except Exception:  # noqa: BLE001 - immutable host-failure boundary
            record = _infra_record(manifest, planned, "benchmark_host_failure")
        _write_run(root, record)
        return record
    finally:
        if worktree.exists():
            _git(repository, "worktree", "remove", "--force", str(worktree))


async def _execute_turn(root: Path, manifest: dict[str, Any], planned: dict[str, Any], worktree: Path,
                        state: Path, reviewer_id: str) -> dict[str, Any]:
    from pico.cli._helpers import make_provider
    from pico.config.loader import load_config
    from pico.config.pico import PicoConfig, load_pico_config
    from pico.proactive_engine.schedulers.cron.service import CronService

    config = load_config()
    budget = RuntimeBudget(**manifest["budget"])
    identity = configuration_identity(config, budget)
    expected = (
        manifest["provider_id"], manifest["actual_model_id"],
        manifest["provider_model_config_digest"], manifest["tool_config_digest"],
        manifest["runtime_config_digest"],
    )
    actual = (
        identity["provider_id"], identity["model"], identity["provider_digest"],
        identity["tool_digest"], identity["runtime_digest"],
    )
    if expected != actual:
        raise RuntimeError("configured Runtime/Provider differs from frozen JEV.4 campaign")
    arm = Arm(planned["arm"])
    pico_config = load_pico_config()
    context = pico_config.context.model_dump(mode="python")
    if arm is Arm.TASK_RELEVANCE_V1:
        context["knowledge_selection_mode"] = "task_relevance_v1"
    elif arm is Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY:
        context.update(
            knowledge_selection_mode="task_relevance_v1_jev_utility",
            knowledge_utility_backend="provider",
            knowledge_utility_provider="agent",
            knowledge_utility_model=manifest["actual_model_id"],
            knowledge_utility_timeout_seconds=UTILITY_TIMEOUT_SECONDS,
        )
    pico_config = PicoConfig.model_validate(
        {**pico_config.model_dump(mode="python"), "memory": {"backend": None}, "context": context}
    )
    if arm is not Arm.NO_REUSE:
        prepare_approved_corpus(state_root=state, workspace=worktree, reviewer_id=reviewer_id)

    old_trace = os.environ.get("PICO_TRACING_DIR")
    os.environ["PICO_TRACING_DIR"] = str(state)
    started_at = datetime.now(timezone.utc).isoformat()
    turn_id = f"{planned['run_id']}-turn"
    runtime_outcome = "timeout"
    provider = make_provider(config)
    outlet = RecordingOutlet("jev4-live")
    cron = CronService(state / "cron" / "jobs.json", allowed_channels={"jev4-live"})
    host = await RuntimeTrialHost.build(
        config=config, pico_config=pico_config, provider=provider, cron_service=cron,
        outlet=outlet, paths=RuntimePaths(workspace=worktree, state=state),
        turn_id_factory=lambda: turn_id,
    )
    try:
        request = TurnRequest(
            origin=Origin.USER,
            source=Source("jev4-live", planned["run_id"], "jev4-human", ChatType.DM),
            text=task_by_id(planned["task_id"]).prompt,
            conversation=f"jev4:{planned['run_id']}",
        )
        try:
            observation = await asyncio.wait_for(host.run(request), timeout=budget.wall_clock_timeout_seconds)
        except TimeoutError:
            pass
        else:
            turn_id = observation.turn_id
            runtime_outcome = observation.runtime_state.value
    finally:
        await host.close()
        if old_trace is None:
            os.environ.pop("PICO_TRACING_DIR", None)
        else:
            os.environ["PICO_TRACING_DIR"] = old_trace
    terminal_at = datetime.now(timezone.utc).isoformat()
    task = task_by_id(planned["task_id"])
    verified = verify_workspace(task, worktree, python_executable=sys.executable)
    patch = _workspace_patch(worktree)
    patch_digest = canonical_digest(patch)
    patch_path = root / "patches" / f"{planned['run_id']}.json"
    _store(root).append_immutable(patch_path, {"run_id": planned["run_id"], "patch_digest": patch_digest, **patch})
    metrics, refs = extract_run_metrics(trace_root=state, turn_id=turn_id, knowledge_state_root=state)
    readback = evidence.read_turn_evidence(state, turn_id)
    validity = classify_run_validity(
        runtime_outcome=runtime_outcome,
        normalized_provider_failures=tuple(refs["normalized_provider_failure_categories"]),
        infrastructure_reason=None,
        mandatory_evidence_complete=readback.completeness is evidence.EvidenceCompleteness.COMPLETE,
    )
    utility = _utility_projection(refs)
    main = to_primitive(metrics)
    combined = _combined_cost(main, utility)
    record = {
        "schema": RUN_SCHEMA,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        **planned,
        "workspace_identity": f"git:{manifest['base_commit_sha']}:{planned['run_id']}",
        "turn_id": turn_id,
        "started_at": started_at,
        "terminal_at": terminal_at,
        "runtime_outcome": runtime_outcome,
        "run_validity": validity.validity.value,
        "infra_invalid_reason": validity.reason.value if validity.reason else None,
        "verified_success": bool(verified["passed"]),
        "verifier_findings": verified["findings"],
        "main_agent_metrics": main,
        "utility_metrics": utility,
        "combined_provider_cost": combined,
        "retrieved_candidate_ids": refs["retrieved_candidate_ids"],
        "relevance_selected_candidate_ids": refs["relevance_selected_candidate_ids"],
        "injected_candidate_ids": refs["injected_ids"],
        "referenced_candidate_ids": refs["referenced_ids"],
        "activated_candidate_ids": refs["activated_ids"],
        "repository_read_paths": refs["repository_read_paths"],
        "changed_paths": _changed_paths(worktree),
        "patch_digest": patch_digest,
        "patch_artifact_ref": patch_path.relative_to(root).as_posix(),
        "skill_activation_count": main["skills_activated"]["value"],
        "typesafe_enabled": False,
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def _utility_projection(refs: dict[str, object]) -> dict[str, Any]:
    decisions = tuple(dict(item) for item in refs.get("utility_decisions", ()))
    raw = Counter(str(item.get("decision", "")).upper() for item in decisions)
    effective = Counter(str(item.get("effective_decision", "")).upper() for item in decisions)
    return {
        "relevance_selected_count": len(tuple(refs["relevance_selected_candidate_ids"])),
        "utility_invoked": int(refs["utility_invoked_count"]) > 0,
        "utility_logical_calls": int(refs["utility_provider_calls"]),
        "utility_provider_attempts": int(refs["utility_provider_attempts"]),
        "decisions": decisions,
        "raw_choice_counts": {key: raw[key] for key in ("KEEP", "ABSTAIN", "UNCERTAIN")},
        "effective_choice_counts": {key: effective[key] for key in ("KEEP", "ABSTAIN")},
        "kept_candidate_ids": refs["utility_kept_candidate_ids"],
        "abstained_candidate_ids": refs["utility_abstained_candidate_ids"],
        "fallback_count": int(refs["utility_fallback_count"]),
        "fallback_reasons": refs["utility_fallback_reasons"],
        "input_tokens": refs["utility_input_tokens"],
        "output_tokens": refs["utility_output_tokens"],
        "provider_latency_ms": refs["utility_provider_latency_ms"],
        "total_latency_ms": refs["utility_latency_ms"],
    }


def _metric_value(metrics: dict[str, Any], name: str) -> int | float | None:
    value = metrics[name]
    return value.get("value") if isinstance(value, dict) else None


def _sum_optional(left: int | float | None, right: int | float | None) -> int | float | None:
    return None if left is None or right is None else left + right


def _combined_cost(main: dict[str, Any], utility: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_logical_calls": _sum_optional(_metric_value(main, "provider_logical_calls"), utility["utility_logical_calls"]),
        "provider_attempts": _sum_optional(_metric_value(main, "provider_attempts"), utility["utility_provider_attempts"]),
        "input_tokens": _sum_optional(_metric_value(main, "input_tokens"), utility["input_tokens"]),
        "output_tokens": _sum_optional(_metric_value(main, "output_tokens"), utility["output_tokens"]),
        "latency_semantics": "utility latency is inside Agent Turn wall time and is reported separately; values are not added",
    }


def _write_run(root: Path, record: dict[str, Any]) -> None:
    _store(root).append_immutable(root / "runs" / f"{record['run_id']}.json", record)


def _infra_record(manifest: dict[str, Any], planned: dict[str, Any], reason: str) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA, "schema_version": 1, "campaign_id": manifest["campaign_id"],
        **planned, "run_validity": "infra_invalid", "infra_invalid_reason": reason,
        "verified_success": False, "verifier_findings": (), "main_agent_metrics": {},
        "utility_metrics": {}, "combined_provider_cost": {}, "changed_paths": (),
        "typesafe_enabled": False,
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def load_runs(root: Path) -> tuple[dict[str, Any], ...]:
    values = []
    for path in sorted((root / "runs").glob("*.json")):
        item = _store(root).read_json(path)
        digest = item.pop("integrity_digest", None)
        if digest != canonical_digest(item):
            raise ValueError(f"JEV.4 run integrity mismatch: {path.name}")
        item["integrity_digest"] = digest
        values.append(item)
    return tuple(sorted(values, key=lambda item: item["order"]))


def reduce_campaign(root: Path) -> dict[str, Any]:
    manifest = load_manifest(root)
    runs = load_runs(root)
    by_key = {(item["task_id"], item["arm"]): item for item in runs}
    pairs = []
    b_pass_c_fail = c_pass_b_fail = 0
    over_abstention = []
    for task in TASKS:
        row = {arm.value: by_key.get((task.task_id, arm.value)) for arm in Arm}
        b, c = row[Arm.TASK_RELEVANCE_V1.value], row[Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value]
        if b and c:
            bp, cp = bool(b["verified_success"]), bool(c["verified_success"])
            b_pass_c_fail += int(bp and not cp)
            c_pass_b_fail += int(cp and not bp)
            if bp and not cp and c["utility_metrics"].get("raw_choice_counts", {}).get("ABSTAIN", 0):
                over_abstention.append(task.task_id)
        pairs.append({"task_id": task.task_id, "arms": row})
    infra = sum(item["run_validity"] == "infra_invalid" for item in runs)
    zero = by_key.get((TASKS[2].task_id, Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value))
    zero_ok = bool(
        zero and not zero["relevance_selected_candidate_ids"]
        and zero["utility_metrics"]["utility_logical_calls"] == 0
    )
    fallbacks = sum(item.get("utility_metrics", {}).get("fallback_count", 0) for item in runs)
    utility_calls = sum(item.get("utility_metrics", {}).get("utility_logical_calls", 0) for item in runs)
    if infra or not zero_ok or b_pass_c_fail or over_abstention:
        classification = "HOLD_AND_FORENSIC"
        reasons = tuple(filter(None, (
            "infrastructure ambiguity" if infra else "",
            "zero-selection control failed" if not zero_ok else "",
            "B_PASS/C_FAIL observed" if b_pass_c_fail else "",
            "utility ABSTAIN coincided with B_PASS/C_FAIL" if over_abstention else "",
        )))
    else:
        classification = "GO_NEW_HELDOUT_BENCHMARK"
        reasons = ("infrastructure and zero-selection contracts held", "no B_PASS/C_FAIL observed")
    if classification not in PILOT_CLASSIFICATIONS:
        raise AssertionError("invalid JEV.4 classification")
    summary = {
        "schema": "pico.jev4-agent-pilot-summary.v1",
        "campaign_id": manifest["campaign_id"],
        "campaign_semantic_digest": manifest["campaign_semantic_digest"],
        "planned_agent_runs": 9,
        "actual_agent_runs": len(runs),
        "actual_utility_logical_calls": utility_calls,
        "infra_invalid_count": infra,
        "per_task": pairs,
        "b_pass_c_fail_count": b_pass_c_fail,
        "c_pass_b_fail_count": c_pass_b_fail,
        "suspected_over_abstention_tasks": tuple(over_abstention),
        "fallback_count": fallbacks,
        "zero_selection_control_passed": zero_ok,
        "classification": classification,
        "classification_reasons": reasons,
        "rerun_count": 0,
        "latency_semantics": "utility decision time is nested inside Turn wall time and is not added to it",
    }
    _store(root).write_summary(root / "reduced.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.picobench.packs.knowledge_evolution_live.jev4_pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository", type=Path, default=Path.cwd())
    prepare.add_argument("--output-root", type=Path, default=Path(".p3r"))
    prepare.add_argument("--reviewer-id", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--execute-live", action="store_true")
    reduce = commands.add_parser("reduce")
    reduce.add_argument("--campaign-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        root, manifest, audit = prepare_campaign(args.repository.resolve(), args.output_root.resolve(), args.reviewer_id)
        print(canonical_json({"campaign_id": manifest["campaign_id"], "campaign_root": root,
                              "semantic_digest": manifest["campaign_semantic_digest"],
                              "preflight": audit, "live_provider_invoked": False}))
        return 0
    if args.command == "run":
        runs = run_campaign(args.repository.resolve(), args.campaign_root.resolve(), args.reviewer_id,
                            execute_live=args.execute_live)
        print(canonical_json({"completed_agent_runs": runs}))
        return 0
    print(json.dumps(reduce_campaign(args.campaign_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Arm", "CAMPAIGN_SEED", "CONTRACT_AUDIT", "PRIMARY_COMPARISON", "SCHEMA",
    "TASKS", "UTILITY_TIMEOUT_SECONDS", "make_plan", "prepare_campaign", "preflight",
    "reduce_campaign", "run_campaign", "verify_workspace",
]
