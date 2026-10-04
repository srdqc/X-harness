"""Frozen three-arm JEV.4 Agent pilot.

Preparation and reduction are offline. Live execution requires the explicit
``--execute-live`` flag and never offers replacement runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    ProviderUtilityConfig,
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
from .run_isolation import (
    CONFIG_BOOTSTRAP_VERSION,
    RUNNER_VERSION,
    AgentRunRoots,
    environment_fingerprint,
    verify_trace_canary,
)
from .runner import _changed_paths, _git, _workspace_patch
from .schema import RuntimeBudget
from .tasks import EXPLORATORY_TASKS, OFFICIAL_TASKS_V1, OFFICIAL_TASKS_V2
from .validity import classify_run_validity

LEGACY_SCHEMA = "pico.jev4-agent-pilot.v1"
LEGACY_SCHEMA_VERSION = 1
REPAIRED_SCHEMA = "pico.jev4r-agent-pilot.v2"
REPAIRED_SCHEMA_VERSION = 2
REPAIRED2_SCHEMA = "pico.jev4r2-agent-pilot.v3"
REPAIRED2_SCHEMA_VERSION = 3
SCHEMA = "pico.jev4r3-agent-pilot.v4"
SCHEMA_VERSION = 4
RUN_SCHEMA = "pico.jev4-agent-run.v1"
UTILITY_TIMEOUT_SECONDS = 15.0
UTILITY_MAX_TOKENS = ProviderUtilityConfig().max_tokens
CAMPAIGN_SEED = 40_904
PRIMARY_COMPARISON = ("task_relevance_v1", "task_relevance_v1_provider_utility")
PILOT_CLASSIFICATIONS = {
    "GO_NEW_HELDOUT_BENCHMARK",
    "HOLD_AND_FORENSIC",
    "STOP_PROVIDER_UTILITY",
}
JEV4R_CLASSIFICATIONS = {
    "INFRA_PASS_BEHAVIOR_PROMISING",
    "INFRA_PASS_BEHAVIOR_MIXED",
    "INFRA_PASS_BEHAVIOR_CONCERNING",
    "HOLD_INFRA",
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


def _behavioral_input_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize v1/v2 manifests to the frozen behavior surface."""
    return {
        key: value[key]
        for key in (
            "base_commit_sha", "campaign_seed", "primary_comparison", "task_order",
            "task_prompt_digests", "verifier_digests", "reference_fixture_digests",
            "contract_audit_digest", "provider_id", "actual_model_id",
            "provider_model_config_digest", "tool_config_digest", "runtime_config_digest",
            "budget", "utility_prompt_digest", "utility_schema_version",
            "utility_timeout_seconds", "knowledge_corpus_digest", "selector_digest",
            "treatment_identity", "planned_runs", "preflight_relevance_selected_candidate_ids",
            "replacement_policy",
        )
    } | {"utility_max_tokens": int(value.get("utility_max_tokens", 1024))}


def _semantic_payload(*, source: dict[str, Any], failed_source: dict[str, Any],
                      prior_source: dict[str, Any],
                      identity: dict[str, str], config_identity: dict[str, Any],
                      selected_sets: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    infrastructure_delta = {
        "runner_version": RUNNER_VERSION,
        "child_process_per_run": True,
        "trace_canary": True,
        "environment_drift_guard": True,
        "python_install_scope": "run_local",
        "malformed_diagnostics": True,
        "explicit_config_source": True,
        "sanitized_config_identity_guard": True,
        "offline_child_bootstrap": True,
        "config_bootstrap_version": CONFIG_BOOTSTRAP_VERSION,
        "verified_bootstrap_is_terminal": True,
        "authoritative_live_readiness": True,
    }
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "config_bootstrap_version": CONFIG_BOOTSTRAP_VERSION,
        "campaign_purpose": "non_claim_eligible_infrastructure_behavioral_rerun",
        "infrastructure_delta_digest": canonical_digest(infrastructure_delta),
        "source_campaign_id": source["campaign_id"],
        "source_campaign_semantic_digest": source["campaign_semantic_digest"],
        "failed_campaign_id": failed_source["campaign_id"],
        "failed_campaign_semantic_digest": failed_source["campaign_semantic_digest"],
        "prior_campaign_id": prior_source["campaign_id"],
        "prior_campaign_semantic_digest": prior_source["campaign_semantic_digest"],
        "historical_lineage": (
            source["campaign_id"], failed_source["campaign_id"], prior_source["campaign_id"],
        ),
        "config_identity_digest": config_identity["config_identity_digest"],
        "config_source_digest": config_identity["config_source_digest"],
        "base_commit_sha": source["base_commit_sha"],
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
        "budget": source["budget"],
        "utility_prompt_digest": PROVIDER_UTILITY_PROMPT_DIGEST,
        "utility_schema_version": PROVIDER_UTILITY_TEMPLATE_VERSION,
        "utility_timeout_seconds": UTILITY_TIMEOUT_SECONDS,
        "utility_max_tokens": UTILITY_MAX_TOKENS,
        "knowledge_corpus_digest": corpus_digest(),
        "selector_digest": _selector_digest(),
        "treatment_identity": _treatment_identity(),
        "planned_runs": make_plan(),
        "preflight_relevance_selected_candidate_ids": selected_sets,
        "replacement_policy": "forbidden_stop_on_infra_invalid",
    }
    result["behavioral_input_digest"] = canonical_digest(_behavioral_input_payload(result))
    return result


def _store(root: Path) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(root.name, root))


def _config_source_digest(path: Path) -> str:
    if not path.is_file():
        raise ValueError("explicit Provider config source does not exist")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("explicit Provider config source is not readable") from exc


def _sanitized_config_identity(config_path: Path, budget: RuntimeBudget) -> dict[str, Any]:
    """Resolve Provider configuration without retaining raw config or secrets."""
    from pico.config.loader import load_config
    from pico.providers.registry import find_by_model, find_by_name

    source_digest = _config_source_digest(config_path)
    config = load_config(config_path)
    identity = configuration_identity(config, budget)
    provider_id = identity["provider_id"]
    model_id = identity["model"]
    provider_spec = find_by_name(provider_id)
    model_spec = find_by_model(model_id)
    provider_config = config.get_provider(model_id)
    credential_available = bool(
        provider_config
        and (provider_config.api_key or (provider_spec and (provider_spec.is_oauth or provider_spec.is_local)))
    )
    if provider_id != "deepseek" or not model_id.startswith("deepseek/"):
        raise ValueError("JEV.4R2 requires the configured DeepSeek Provider/model")
    if provider_spec is None or model_spec is None or model_spec.name != provider_spec.name:
        raise ValueError("configured Provider/model does not resolve through the registry")
    if not credential_available:
        raise ValueError("configured DeepSeek credentials are unavailable")
    payload = {
        "provider_id": provider_id,
        "model_id": model_id,
        "provider_model_config_digest": identity["provider_digest"],
        "config_source_digest": source_digest,
        "registry_provider_id": provider_spec.name,
        "credential_available": True,
    }
    return {**payload, "config_identity_digest": canonical_digest(payload)}


def _validate_child_config(manifest: dict[str, Any], config_path: Path) -> dict[str, Any]:
    identity = _sanitized_config_identity(config_path, RuntimeBudget(**manifest["budget"]))
    passed = (
        identity["config_identity_digest"] == manifest["config_identity_digest"]
        and identity["config_source_digest"] == manifest["config_source_digest"]
    )
    return {**identity, "passed": passed}


def prepare_campaign(repository: Path, output_root: Path, source_root: Path,
                     failed_root: Path, prior_root: Path, reviewer_id: str, *,
                     config_path: Path | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from pico.config.loader import get_config_path, load_config

    parent_environment_before = environment_fingerprint()
    config_path = (config_path or get_config_path()).resolve()
    config = load_config(config_path)
    source = load_manifest(source_root)
    if source["schema_version"] != LEGACY_SCHEMA_VERSION:
        raise ValueError("JEV.4R source must be the frozen JEV.4 v1 campaign")
    failed_source = load_manifest(failed_root)
    if failed_source.get("schema_version") != REPAIRED_SCHEMA_VERSION:
        raise ValueError("JEV.4R2 failed source must be the frozen JEV.4R v2 campaign")
    failed_summary = json.loads((failed_root / "reduced.json").read_text(encoding="utf-8"))
    if failed_summary.get("classification") != "HOLD_INFRA":
        raise ValueError("JEV.4R2 failed source must remain HOLD_INFRA")
    prior_source = load_manifest(prior_root)
    if prior_source.get("schema_version") != REPAIRED2_SCHEMA_VERSION:
        raise ValueError("JEV.4R3 prior source must be the frozen JEV.4R2 v3 campaign")
    prior_summary = json.loads((prior_root / "reduced.json").read_text(encoding="utf-8"))
    if prior_summary.get("classification") != "HOLD_INFRA":
        raise ValueError("JEV.4R3 prior source must remain HOLD_INFRA")
    budget = RuntimeBudget(**source["budget"])
    identity = configuration_identity(config, budget)
    if identity["provider_id"] != "deepseek" or not identity["model"].startswith("deepseek/"):
        raise ValueError("JEV.4 requires the configured DeepSeek Agent provider/model")
    base_sha = resolve_base_commit(repository, source["base_commit_sha"])
    audit = preflight(repository, base_sha, reviewer_id)
    config_identity = _sanitized_config_identity(config_path, budget)
    semantic = _semantic_payload(
        source=source,
        failed_source=failed_source,
        prior_source=prior_source,
        identity=identity,
        config_identity=config_identity,
        selected_sets=audit["relevance_selected_candidate_ids"],
    )
    assert_infrastructure_only_rerun(source, semantic)
    semantic_digest = canonical_digest(semantic)
    campaign_id = f"jev4r3-{semantic_digest[:16]}"
    manifest = {
        **to_primitive(semantic),
        "campaign_id": campaign_id,
        "campaign_semantic_digest": semantic_digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    root = output_root / campaign_id
    _store(root).freeze_manifest(manifest)
    bootstrap = run_config_bootstrap(root, config_path)
    if not bootstrap["passed"]:
        raise RuntimeError("JEV.4R2 offline child bootstrap failed")
    bootstrap_artifact = root / "config-bootstrap.json"
    bootstrap_digest_before = hashlib.sha256(bootstrap_artifact.read_bytes()).hexdigest()
    bootstrap = validate_config_bootstrap(root, config_path)
    bootstrap_digest_after = hashlib.sha256(bootstrap_artifact.read_bytes()).hexdigest()
    readiness_checks = {
        "behavior_inputs_immutable": True,
        "mechanical_solvability": audit["mechanical_solvability"] == "3/3 PASS",
        "prompt_verifier_audit": audit["contract_audit"] == "PASS",
        "parent_config_resolution": identity["provider_id"] == "deepseek",
        "bootstrap_child_count": bootstrap["bootstrap_child_count"] == 1,
        "bootstrap_verified": bootstrap["passed"],
        "bootstrap_artifact_immutable": bootstrap_digest_before == bootstrap_digest_after,
        "config_identity_verified": bootstrap["config_identity_digest"] == config_identity["config_identity_digest"],
        "config_source_immutable": bootstrap["config_source_unchanged"],
        "trace_canary_verified": bootstrap["trace_canary"]["passed"],
        "environment_baseline_verified": (
            parent_environment_before["fingerprint_digest"]
            == environment_fingerprint()["fingerprint_digest"]
        ),
        "zero_selection_preflight": not audit["relevance_selected_candidate_ids"][TASKS[2].task_id],
        "provider_model_identity_verified": (
            bootstrap["provider_id"] == identity["provider_id"]
            and bootstrap["model_id"] == identity["model"]
        ),
    }
    readiness = {
        "schema": "pico.jev4r3-live-readiness.v1",
        "campaign_id": campaign_id,
        "checks": readiness_checks,
        "live_ready": all(readiness_checks.values()),
        "bootstrap_artifact_digest": bootstrap_digest_after,
        "next_run": manifest["planned_runs"][0],
        "main_provider_calls": 0,
        "utility_provider_calls": 0,
        "agent_turns": 0,
        "network_calls": 0,
    }
    readiness["integrity_digest"] = canonical_digest(readiness)
    _store(root).write_summary(root / "pre-live-readiness.json", readiness)
    if not readiness["live_ready"]:
        raise RuntimeError("JEV.4R3 authoritative pre-live readiness failed")
    audit.update(
        immutable_input_delta="PASS",
        source_campaign_id=source["campaign_id"],
        environment_fingerprint=environment_fingerprint(),
        stale_run_roots=False,
        child_process_runner_available=True,
        utility_max_tokens=UTILITY_MAX_TOKENS,
        config_identity_digest=config_identity["config_identity_digest"],
        config_source_digest=config_identity["config_source_digest"],
        child_bootstrap=bootstrap,
        live_readiness=readiness,
    )
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
    version = value.get("schema_version")
    if (value.get("schema"), version) not in {
        (LEGACY_SCHEMA, LEGACY_SCHEMA_VERSION),
        (REPAIRED_SCHEMA, REPAIRED_SCHEMA_VERSION),
        (REPAIRED2_SCHEMA, REPAIRED2_SCHEMA_VERSION),
        (SCHEMA, SCHEMA_VERSION),
    }:
        raise ValueError("unsupported JEV.4 manifest")
    semantic = {key: value[key] for key in _semantic_payload_keys(int(version))}
    if value["campaign_semantic_digest"] != canonical_digest(semantic):
        raise ValueError("JEV.4 semantic digest mismatch")
    return value


def _semantic_payload_keys(version: int = SCHEMA_VERSION) -> tuple[str, ...]:
    common = (
        "schema", "schema_version", "base_commit_sha", "campaign_seed", "primary_comparison",
        "task_order", "task_prompt_digests", "verifier_digests", "reference_fixture_digests",
        "contract_audit_digest", "provider_id", "actual_model_id", "provider_model_config_digest",
        "tool_config_digest", "runtime_config_digest", "budget", "utility_prompt_digest",
        "utility_schema_version", "utility_timeout_seconds", "knowledge_corpus_digest",
        "selector_digest", "treatment_identity", "planned_runs",
        "preflight_relevance_selected_candidate_ids", "replacement_policy",
    )
    if version == LEGACY_SCHEMA_VERSION:
        return common
    repaired = (
        "schema", "schema_version", "runner_version", "campaign_purpose",
        "behavioral_input_digest", "infrastructure_delta_digest", "source_campaign_id",
        "source_campaign_semantic_digest", *common[2:16], "utility_max_tokens", *common[16:],
    )
    if version == REPAIRED_SCHEMA_VERSION:
        return repaired
    repaired2 = (
        *repaired[:8], "config_bootstrap_version", "failed_campaign_id",
        "failed_campaign_semantic_digest", "config_identity_digest", "config_source_digest",
        *repaired[8:],
    )
    if version == REPAIRED2_SCHEMA_VERSION:
        return repaired2
    return (
        *repaired2[:11], "prior_campaign_id", "prior_campaign_semantic_digest",
        "historical_lineage", *repaired2[11:],
    )


def assert_infrastructure_only_rerun(source: dict[str, Any], candidate: dict[str, Any]) -> None:
    expected = _behavioral_input_payload(source)
    actual = _behavioral_input_payload(candidate)
    changed = tuple(
        key for key in expected
        if canonical_digest(expected[key]) != canonical_digest(actual[key])
    )
    if changed or candidate.get("behavioral_input_digest") != canonical_digest(expected):
        detail = ",".join(changed) or "behavioral_input_digest"
        raise ValueError(f"JEV.4R behavioral inputs changed: {detail}")
    if candidate.get("source_campaign_id") != source.get("campaign_id"):
        raise ValueError("JEV.4R source campaign identity changed")
    if candidate.get("source_campaign_semantic_digest") != source.get("campaign_semantic_digest"):
        raise ValueError("JEV.4R source semantic digest changed")
    if candidate.get("runner_version") != RUNNER_VERSION:
        raise ValueError("JEV.4R runner version is not current")


def _execute_config_bootstrap(root: Path, manifest: dict[str, Any], config_path: Path) -> dict[str, Any]:
    roots = AgentRunRoots.create(root, "config-bootstrap", 0)
    roots.prepare_non_worktree_roots()
    if (root / "config-bootstrap.json").exists():
        raise FileExistsError("VERIFIED bootstrap lifecycle cannot return to RUNNING")
    before = _config_source_digest(config_path)
    try:
        identity = _validate_child_config(manifest, config_path)
        canary = verify_trace_canary(
            roots.trace, canary_id=f"{manifest['campaign_id']}:bootstrap"
        )
        failure_type = None
    except Exception as exc:  # noqa: BLE001 - bounded bootstrap boundary
        identity = {"passed": False}
        canary = {"passed": False, "findings": ("config_validation_failed",)}
        failure_type = type(exc).__name__
    after = _config_source_digest(config_path)
    record = {
        "schema": "pico.jev4r2-config-bootstrap.v1",
        "bootstrap_version": CONFIG_BOOTSTRAP_VERSION,
        "lifecycle_state": "VERIFIED" if identity.get("passed") and canary.get("passed") else "FAILED",
        "lifecycle_transitions": ("NOT_STARTED", "RUNNING", "VERIFIED"),
        "bootstrap_child_count": 1,
        "child_process_id": os.getpid(),
        "campaign_id": manifest["campaign_id"],
        "passed": bool(identity.get("passed") and canary.get("passed") and before == after),
        "provider_id": identity.get("provider_id"),
        "model_id": identity.get("model_id"),
        "config_identity_digest": identity.get("config_identity_digest"),
        "config_source_digest_before": before,
        "config_source_digest_after": after,
        "config_source_unchanged": before == after,
        "trace_canary": canary,
        "environment_fingerprint": environment_fingerprint(),
        "failure_type": failure_type,
        "network_calls": 0,
        "provider_calls": 0,
        "agent_turns": 0,
        "pico_home_isolated": str(Path(os.environ.get("PICO_HOME", "")).resolve()).startswith(
            str(roots.state.resolve())
        ),
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def run_config_bootstrap(root: Path, config_path: Path) -> dict[str, Any]:
    from pico.config.loader import load_config

    manifest = load_manifest(root)
    roots = AgentRunRoots.create(root, "config-bootstrap", 0)
    roots.prepare_non_worktree_roots()
    artifact = root / "config-bootstrap.json"
    before = _config_source_digest(config_path)
    if artifact.is_file():
        return validate_config_bootstrap(root, config_path)
    command = [
        sys.executable, "-m", "benchmarks.picobench.packs.knowledge_evolution_live.jev4_pilot",
        "_bootstrap", "--campaign-root", str(root), "--config-path", str(config_path),
    ]
    completed = subprocess.run(command, cwd=Path.cwd(), env=roots.child_environment(), check=False)
    after = _config_source_digest(config_path)
    if completed.returncode != 0 or not artifact.is_file():
        raise RuntimeError("JEV.4R2 child bootstrap artifact missing")
    record = _store(root).read_json(artifact)
    digest = record.pop("integrity_digest", None)
    if digest != canonical_digest(record):
        raise ValueError("JEV.4R2 child bootstrap integrity mismatch")
    record["integrity_digest"] = digest
    secret = load_config(config_path).get_api_key(manifest["actual_model_id"])
    persisted = canonical_json({"manifest": manifest, "artifact": record, "command": command})
    for path in roots.trace.rglob("*"):
        if path.is_file():
            persisted += path.read_text(encoding="utf-8", errors="replace")
    secret_leak_free = not secret or secret not in persisted
    record.update(
        parent_config_source_digest_before=before,
        parent_config_source_digest_after=after,
        config_source_unchanged=record["config_source_unchanged"] and before == after,
        secret_leak_free=secret_leak_free,
    )
    record["passed"] = bool(
        record["passed"]
        and record["config_identity_digest"] == manifest["config_identity_digest"]
        and record["config_source_unchanged"]
        and secret_leak_free
    )
    record.pop("integrity_digest", None)
    record["integrity_digest"] = canonical_digest(record)
    _store(root).write_summary(artifact, record)
    return record


def validate_config_bootstrap(root: Path, config_path: Path) -> dict[str, Any]:
    """Read-only parent validation of a terminal bootstrap lifecycle."""
    manifest = load_manifest(root)
    artifact = root / "config-bootstrap.json"
    before_bytes = artifact.read_bytes()
    record = _store(root).read_json(artifact)
    digest = record.pop("integrity_digest", None)
    if digest != canonical_digest(record):
        raise ValueError("JEV.4R2 child bootstrap integrity mismatch")
    record["integrity_digest"] = digest
    current = _validate_child_config(manifest, config_path)
    passed = bool(
        record.get("passed")
        and record.get("lifecycle_state") == "VERIFIED"
        and record.get("secret_leak_free")
        and record.get("config_source_unchanged")
        and record.get("config_identity_digest") == current["config_identity_digest"]
        and _config_source_digest(config_path) == manifest["config_source_digest"]
        and artifact.read_bytes() == before_bytes
    )
    return {
        **record,
        "passed": passed,
        "revalidated_without_child_rerun": True,
        "artifact_file_digest": hashlib.sha256(before_bytes).hexdigest(),
    }


def _load_live_readiness(root: Path) -> dict[str, Any]:
    value = _store(root).read_json(root / "pre-live-readiness.json")
    digest = value.pop("integrity_digest", None)
    if digest != canonical_digest(value):
        raise ValueError("JEV.4R3 readiness artifact integrity mismatch")
    value["integrity_digest"] = digest
    return value


def run_campaign(repository: Path, root: Path, reviewer_id: str, *, execute_live: bool) -> tuple[str, ...]:
    from pico.config.loader import get_config_path

    if not execute_live:
        raise PermissionError("JEV.4 live execution requires explicit --execute-live")
    manifest = load_manifest(root)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("historical JEV.4 campaigns are immutable and cannot use the JEV.4R runner")
    source = load_manifest(root.parent / manifest["source_campaign_id"])
    assert_infrastructure_only_rerun(source, manifest)
    failed = load_manifest(root.parent / manifest["failed_campaign_id"])
    if failed["campaign_semantic_digest"] != manifest["failed_campaign_semantic_digest"]:
        raise ValueError("JEV.4R2 failed-campaign identity changed")
    prior = load_manifest(root.parent / manifest["prior_campaign_id"])
    if prior["campaign_semantic_digest"] != manifest["prior_campaign_semantic_digest"]:
        raise ValueError("JEV.4R3 prior-campaign identity changed")
    run_dir = root / "runs"
    if run_dir.exists() and tuple(run_dir.glob("*.json")):
        raise FileExistsError("JEV.4 is one-shot; existing run records forbid resume or replacement")
    completed: list[str] = []
    config_path = get_config_path().resolve()
    if _validate_child_config(manifest, config_path)["passed"] is not True:
        raise ValueError("parent Provider config identity differs from frozen JEV.4R2 identity")
    bootstrap = run_config_bootstrap(root, config_path)
    if not bootstrap["passed"]:
        raise RuntimeError("JEV.4R2 mandatory child bootstrap failed before live execution")
    readiness = _load_live_readiness(root)
    if not readiness["live_ready"] or not all(readiness["checks"].values()):
        raise RuntimeError("JEV.4R3 authoritative readiness is not live-ready")
    if readiness["bootstrap_artifact_digest"] != bootstrap["artifact_file_digest"]:
        raise RuntimeError("JEV.4R3 bootstrap artifact changed after readiness")
    if readiness["next_run"] != manifest["planned_runs"][0]:
        raise RuntimeError("JEV.4R3 first-run handoff changed")
    for planned in manifest["planned_runs"]:
        roots = AgentRunRoots.create(root, planned["run_id"], planned["order"])
        roots.prepare_non_worktree_roots()
        before = environment_fingerprint()
        config_source_before = _config_source_digest(config_path)
        completed_process = subprocess.run(
            [
                sys.executable, "-m",
                "benchmarks.picobench.packs.knowledge_evolution_live.jev4_pilot",
                "_run-one", "--repository", str(repository), "--campaign-root", str(root),
                "--run-id", planned["run_id"], "--reviewer-id", reviewer_id,
                "--config-path", str(config_path), "--execute-live",
            ],
            cwd=repository,
            env=roots.child_environment(),
            check=False,
        )
        after = environment_fingerprint()
        config_source_after = _config_source_digest(config_path)
        if roots.artifact.exists():
            record = _store(root).read_json(roots.artifact)
        else:
            record = _infra_record(manifest, planned, "benchmark_host_failure")
        record.pop("integrity_digest", None)
        record["environment_fingerprint_before"] = before
        record["environment_fingerprint_after"] = after
        record["environment_drift_detected"] = before["fingerprint_digest"] != after["fingerprint_digest"]
        record["config_source_digest_before"] = config_source_before
        record["config_source_digest_after"] = config_source_after
        record["config_source_unchanged"] = (
            config_source_before == config_source_after == manifest["config_source_digest"]
        )
        if record["environment_drift_detected"]:
            record.update(
                run_validity="infra_invalid",
                infra_invalid_reason="environment_drift",
                verified_success=False,
            )
        elif not record["config_source_unchanged"]:
            record.update(
                run_validity="infra_invalid",
                infra_invalid_reason="config_source_mutation",
                verified_success=False,
            )
        elif completed_process.returncode != 0:
            record.update(
                run_validity="infra_invalid",
                infra_invalid_reason="benchmark_host_failure",
                verified_success=False,
            )
        isolation = _cross_run_isolation(root, record)
        record["cross_run_isolation"] = isolation
        if not isolation["passed"] and record["run_validity"] != "infra_invalid":
            record.update(
                run_validity="infra_invalid",
                infra_invalid_reason="cross_run_trace_contamination",
                verified_success=False,
            )
        zero_failure = _zero_selection_contract_failure(record)
        if zero_failure:
            record.update(
                run_validity="infra_invalid",
                infra_invalid_reason=zero_failure,
                verified_success=False,
            )
        record["run_outcome"] = (
            "INFRA_INVALID" if record["run_validity"] == "infra_invalid"
            else "PASS" if record["verified_success"] else "FAIL"
        )
        record["integrity_digest"] = canonical_digest(record)
        _write_run(root, record)
        completed.append(record["run_id"])
        if record["run_validity"] == "infra_invalid":
            raise RuntimeError("JEV.4 stopped after INFRA_INVALID; replacement runs are forbidden")
    return tuple(completed)


def _cross_run_isolation(root: Path, current: dict[str, Any]) -> dict[str, Any]:
    turn_id = current.get("turn_id")
    current_ref = current.get("isolation_roots", {}).get("state", {}).get("ref")
    if not turn_id or not current_ref:
        return {"passed": False, "reason": "missing_current_root_identity"}
    current_root = root / str(current_ref)
    foreign_current = []
    prior_in_current = []
    for previous in load_runs(root):
        previous_ref = previous.get("isolation_roots", {}).get("state", {}).get("ref")
        previous_turn = previous.get("turn_id")
        if previous_ref and evidence.read_turn_evidence(root / str(previous_ref), str(turn_id)).events:
            foreign_current.append(previous["run_id"])
        if previous_turn and evidence.read_turn_evidence(current_root, str(previous_turn)).events:
            prior_in_current.append(previous["run_id"])
    return {
        "passed": not foreign_current and not prior_in_current,
        "current_turn_foreign_run_ids": tuple(foreign_current),
        "prior_turn_ids_in_current_root": tuple(prior_in_current),
    }


def _zero_selection_contract_failure(record: dict[str, Any]) -> str | None:
    if record.get("task_id") != TASKS[2].task_id:
        return None
    arm = record.get("arm")
    selected = tuple(record.get("relevance_selected_candidate_ids", ()))
    if arm in {Arm.TASK_RELEVANCE_V1.value, Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value} and selected:
        return "zero_selection_contract_failure"
    if arm == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value:
        calls = record.get("utility_metrics", {}).get("utility_logical_calls")
        if calls != 0:
            return "zero_selection_utility_invoked"
    return None


def execute_one(
    repository: Path,
    root: Path,
    manifest: dict[str, Any],
    planned: dict[str, Any],
    reviewer_id: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    run_id = planned["run_id"]
    run_path = root / "runs" / f"{run_id}.json"
    if run_path.exists():
        raise FileExistsError(f"immutable JEV.4 run already exists: {run_id}")
    roots = AgentRunRoots.create(root, run_id, planned["order"])
    roots.prepare_non_worktree_roots()
    worktree, state = roots.worktree, roots.state
    from pico.config.loader import get_config_path

    try:
        config_validation = _validate_child_config(manifest, get_config_path().resolve())
    except Exception as exc:  # noqa: BLE001 - no-network identity boundary
        config_validation = {"passed": False, "failure_type": type(exc).__name__}
    if not config_validation["passed"]:
        record = _infra_record(manifest, planned, "config_identity_mismatch")
        record.update(
            isolation_roots=roots.public_identity(),
            config_validation=config_validation,
            child_process_id=os.getpid(),
        )
        record["integrity_digest"] = canonical_digest(
            {key: value for key, value in record.items() if key != "integrity_digest"}
        )
        if persist:
            _write_run(root, record)
        return record
    other_trace_roots = tuple(
        path for path in (root / "s").glob("r*") if path.resolve() != roots.trace.resolve()
    )
    canary = verify_trace_canary(
        roots.trace,
        canary_id=f"{manifest['campaign_id']}:{run_id}:agent-canary",
        other_trace_roots=other_trace_roots,
    )
    if not canary["passed"]:
        record = _infra_record(manifest, planned, "trace_canary_failure")
        record.update(
            isolation_roots=roots.public_identity(),
            trace_canary=canary,
            config_validation=config_validation,
            child_process_id=os.getpid(),
        )
        record["integrity_digest"] = canonical_digest({key: value for key, value in record.items() if key != "integrity_digest"})
        if persist:
            _write_run(root, record)
        return record
    try:
        _git(repository, "worktree", "add", "--detach", str(worktree), manifest["base_commit_sha"])
    except (OSError, RuntimeError):
        record = _infra_record(manifest, planned, "worktree_setup_failure")
        record.update(
            isolation_roots=roots.public_identity(),
            trace_canary=canary,
            config_validation=config_validation,
            child_process_id=os.getpid(),
        )
        record["integrity_digest"] = canonical_digest({key: value for key, value in record.items() if key != "integrity_digest"})
        if persist:
            _write_run(root, record)
        return record
    try:
        try:
            record = asyncio.run(_execute_turn(root, manifest, planned, worktree, state, reviewer_id))
        except Exception as exc:  # noqa: BLE001 - immutable host-failure boundary
            record = _infra_record(manifest, planned, "benchmark_host_failure")
            record["bounded_host_failure_type"] = type(exc).__name__
        record.update(
            isolation_roots=roots.public_identity(),
            trace_canary=canary,
            config_validation=config_validation,
            child_process_id=os.getpid(),
        )
        record["integrity_digest"] = canonical_digest({key: value for key, value in record.items() if key != "integrity_digest"})
        if persist:
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
            knowledge_utility_max_tokens=UTILITY_MAX_TOKENS,
        )
    pico_config = PicoConfig.model_validate(
        {**pico_config.model_dump(mode="python"), "memory": {"backend": None}, "context": context}
    )
    if arm is not Arm.NO_REUSE:
        prepare_approved_corpus(state_root=state, workspace=worktree, reviewer_id=reviewer_id)

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
    terminal_at = datetime.now(timezone.utc).isoformat()
    task = task_by_id(planned["task_id"])
    verified = verify_workspace(task, worktree, python_executable=sys.executable)
    patch = _workspace_patch(worktree)
    patch_digest = canonical_digest(patch)
    patch_path = root / "patches" / f"{planned['run_id']}.json"
    _store(root).append_immutable(patch_path, {"run_id": planned["run_id"], "patch_digest": patch_digest, **patch})
    metrics, refs = extract_run_metrics(trace_root=state, turn_id=turn_id, knowledge_state_root=state)
    readback = evidence.read_turn_evidence(state, turn_id)
    mandatory = {
        "complete": readback.completeness is evidence.EvidenceCompleteness.COMPLETE,
        "completeness": readback.completeness.value,
        "findings": readback.findings,
        "event_count": len(readback.events),
        "run_root_digest": canonical_digest(str(state.resolve())),
    }
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
        "run_outcome": (
            "INFRA_INVALID" if validity.validity.value == "infra_invalid"
            else "PASS" if verified["passed"] else "FAIL"
        ),
        "mandatory_evidence": mandatory,
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
        "configured_utility_max_tokens": UTILITY_MAX_TOKENS,
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
        "finish_reasons": refs.get("utility_finish_reasons", ()),
        "malformed_categories": refs.get("utility_malformed_categories", ()),
        "response_character_counts": refs.get("utility_response_character_counts", ()),
        "response_digests": refs.get("utility_response_digests", ()),
        "top_level_shapes": refs.get("utility_top_level_shapes", ()),
        "parse_stages": refs.get("utility_parse_stages", ()),
        "generation_outcomes": refs.get("utility_generation_outcomes", ()),
        "payload_outcomes": refs.get("utility_payload_outcomes", ()),
        "reasoning_tokens": refs.get("utility_reasoning_tokens", ()),
        "visible_output_tokens": refs.get("utility_visible_output_tokens", ()),
    }


def _metric_value(metrics: dict[str, Any], name: str) -> int | float | None:
    value = metrics[name]
    return value.get("value") if isinstance(value, dict) else None


def _utility_decision_classification(metrics: dict[str, Any]) -> tuple[bool, bool]:
    """Return (all_abstain, unavailable) without conflating fallback with a decision."""
    selected = int(metrics.get("relevance_selected_count", 0))
    decisions = tuple(metrics.get("decisions", ()))
    fallback = int(metrics.get("fallback_count", 0)) > 0
    raw_abstain = int(metrics.get("raw_choice_counts", {}).get("ABSTAIN", 0))
    all_abstain = selected > 0 and not fallback and len(decisions) == selected and raw_abstain == selected
    unavailable = selected > 0 and fallback and not decisions
    return all_abstain, unavailable


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
        "run_outcome": "INFRA_INVALID",
        "verified_success": False, "verifier_findings": (), "main_agent_metrics": {},
        "utility_metrics": {}, "combined_provider_cost": {}, "changed_paths": (),
        "typesafe_enabled": False,
        "mandatory_evidence": {"complete": False, "findings": ("run_not_completed",)},
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
    pair_outcomes = Counter()
    over_abstention = []
    for task in TASKS:
        row = {arm.value: by_key.get((task.task_id, arm.value)) for arm in Arm}
        b, c = row[Arm.TASK_RELEVANCE_V1.value], row[Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value]
        if b and c:
            if b["run_validity"] == "infra_invalid" or c["run_validity"] == "infra_invalid":
                pair_outcomes["INFRA_INVALID"] += 1
            else:
                pair_outcomes[
                    f"B_{'PASS' if b['verified_success'] else 'FAIL'}_C_{'PASS' if c['verified_success'] else 'FAIL'}"
                ] += 1
            bp, cp = bool(b["verified_success"]), bool(c["verified_success"])
            b_pass_c_fail += int(bp and not cp)
            c_pass_b_fail += int(cp and not bp)
            if bp and not cp and c["utility_metrics"].get("raw_choice_counts", {}).get("ABSTAIN", 0):
                over_abstention.append(task.task_id)
        pairs.append({"task_id": task.task_id, "arms": row})
    infra = sum(item["run_validity"] == "infra_invalid" for item in runs)
    zero_b = by_key.get((TASKS[2].task_id, Arm.TASK_RELEVANCE_V1.value))
    zero = by_key.get((TASKS[2].task_id, Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value))
    zero_ok = bool(
        zero_b and not zero_b["relevance_selected_candidate_ids"]
        and zero and not zero["relevance_selected_candidate_ids"]
        and zero["utility_metrics"]["utility_logical_calls"] == 0
    )
    fallbacks = sum(item.get("utility_metrics", {}).get("fallback_count", 0) for item in runs)
    utility_calls = sum(item.get("utility_metrics", {}).get("utility_logical_calls", 0) for item in runs)
    if manifest["schema_version"] == LEGACY_SCHEMA_VERSION:
        return _reduce_legacy(
            root, manifest, runs, pairs, infra, zero_ok, b_pass_c_fail, c_pass_b_fail,
            over_abstention, fallbacks, utility_calls,
        )

    c_runs = tuple(item for item in runs if item["arm"] == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value)
    length_finish_count = sum(
        str(reason).upper() == "LENGTH"
        for item in c_runs for reason in item.get("utility_metrics", {}).get("finish_reasons", ())
    )
    truncated_output_count = sum(
        str(category).upper() == "TRUNCATED_OUTPUT"
        for item in c_runs for category in item.get("utility_metrics", {}).get("malformed_categories", ())
    )
    malformed_response_count = sum(
        bool(item.get("utility_metrics", {}).get("malformed_categories")) for item in c_runs
    )
    all_abstain = tuple(
        item["task_id"] for item in c_runs
        if _utility_decision_classification(item.get("utility_metrics", {}))[0]
    )
    utility_decision_unavailable = tuple(
        item["task_id"] for item in c_runs
        if _utility_decision_classification(item.get("utility_metrics", {}))[1]
    )
    utility_removed = sum(
        len(item.get("utility_metrics", {}).get("abstained_candidate_ids", ())) for item in c_runs
    )
    raw_uncertain = sum(
        item.get("utility_metrics", {}).get("raw_choice_counts", {}).get("UNCERTAIN", 0)
        for item in c_runs
    )
    p2_abstain = any(
        item["task_id"] == TASKS[1].task_id
        and item.get("utility_metrics", {}).get("raw_choice_counts", {}).get("ABSTAIN", 0)
        for item in c_runs
    )
    unique_pids = {item.get("child_process_id") for item in runs if item.get("child_process_id")}
    infra_checks = {
        "all_nine_artifacts_loaded": len(runs) == 9,
        "unique_child_process_per_run": len(unique_pids) == len(runs) == 9,
        "trace_canaries_passed": all(item.get("trace_canary", {}).get("passed") for item in runs),
        "mandatory_evidence_complete": all(item.get("mandatory_evidence", {}).get("complete") for item in runs),
        "environment_drift_absent": all(not item.get("environment_drift_detected", True) for item in runs),
        "cross_run_contamination_absent": all(item.get("cross_run_isolation", {}).get("passed") for item in runs),
        "typesafe_disabled": all(item.get("typesafe_enabled") is False for item in runs),
        "zero_selection_control": zero_ok,
    }
    if manifest["schema_version"] >= SCHEMA_VERSION:
        bootstrap = _store(root).read_json(root / "config-bootstrap.json")
        infra_checks.update(
            config_bootstrap_passed=bool(bootstrap.get("passed")),
            config_identity_matched=all(
                item.get("config_validation", {}).get("passed") for item in runs
            ),
            config_source_immutable=all(item.get("config_source_unchanged") for item in runs),
        )
    infrastructure_valid = infra == 0 and all(infra_checks.values())
    if not infrastructure_valid:
        classification = "HOLD_INFRA"
        reasons = tuple(name for name, passed in infra_checks.items() if not passed) or ("infra invalid run",)
    elif b_pass_c_fail:
        classification = "INFRA_PASS_BEHAVIOR_CONCERNING"
        reasons = ("infrastructure boundaries held", "B_PASS/C_FAIL correctness regression observed")
    elif fallbacks or malformed_response_count or truncated_output_count or all_abstain:
        classification = "INFRA_PASS_BEHAVIOR_MIXED"
        reasons = ("infrastructure boundaries held", "utility fallback/malformed/all-abstain variability observed")
    else:
        classification = "INFRA_PASS_BEHAVIOR_PROMISING"
        reasons = ("infrastructure boundaries held", "no B_PASS/C_FAIL or repeated harmful behavior observed")
    if classification not in JEV4R_CLASSIFICATIONS:
        raise AssertionError("invalid JEV.4R classification")
    c_diagnostics = tuple(
        {
            "task_id": item["task_id"],
            "output_tokens": item.get("utility_metrics", {}).get("output_tokens"),
            "configured_max_tokens": item.get("configured_utility_max_tokens", UTILITY_MAX_TOKENS),
            "finish_reasons": item.get("utility_metrics", {}).get("finish_reasons", ()),
            "malformed_categories": item.get("utility_metrics", {}).get("malformed_categories", ()),
        }
        for item in c_runs
    )
    summary = {
        "schema": (
            "pico.jev4r3-agent-pilot-summary.v4"
            if manifest["schema_version"] >= SCHEMA_VERSION
            else "pico.jev4r-agent-pilot-summary.v2"
        ),
        "campaign_id": manifest["campaign_id"],
        "campaign_semantic_digest": manifest["campaign_semantic_digest"],
        "source_campaign_id": manifest["source_campaign_id"],
        "failed_campaign_id": manifest.get("failed_campaign_id"),
        "prior_campaign_id": manifest.get("prior_campaign_id"),
        "historical_lineage": manifest.get("historical_lineage", ()),
        "config_identity_digest": manifest.get("config_identity_digest"),
        "behavioral_input_digest": manifest["behavioral_input_digest"],
        "infrastructure_delta_digest": manifest["infrastructure_delta_digest"],
        "planned_agent_runs": 9,
        "actual_agent_runs": len(runs),
        "child_process_count": len(unique_pids),
        "child_process_ids": tuple(sorted(unique_pids)),
        "bootstrap_child_count": (
            int(bootstrap.get("bootstrap_child_count", 0))
            if manifest["schema_version"] >= SCHEMA_VERSION else None
        ),
        "total_child_process_count": (
            len(unique_pids) + int(bootstrap.get("bootstrap_child_count", 0))
            if manifest["schema_version"] >= SCHEMA_VERSION else len(unique_pids)
        ),
        "actual_utility_logical_calls": utility_calls,
        "infra_invalid_count": infra,
        "infrastructure_checks": infra_checks,
        "per_task": pairs,
        "pair_outcomes": {name: pair_outcomes[name] for name in (
            "B_PASS_C_PASS", "B_PASS_C_FAIL", "B_FAIL_C_PASS", "B_FAIL_C_FAIL", "INFRA_INVALID"
        )},
        "b_pass_c_fail_count": b_pass_c_fail,
        "c_pass_b_fail_count": c_pass_b_fail,
        "suspected_over_abstention_tasks": tuple(over_abstention),
        "utility_removed_candidate_count": utility_removed,
        "all_abstain_tasks": all_abstain,
        "utility_decision_unavailable_tasks": utility_decision_unavailable,
        "raw_uncertain_count": raw_uncertain,
        "precondition_ambiguous_task_received_abstain": p2_abstain,
        "fallback_count": fallbacks,
        "zero_selection_control_passed": zero_ok,
        "length_finish_count": length_finish_count,
        "truncated_output_count": truncated_output_count,
        "malformed_response_count": malformed_response_count,
        "utility_truncation_diagnostics": c_diagnostics,
        "classification": classification,
        "classification_reasons": reasons,
        "rerun_count": 0,
        "latency_semantics": "utility decision time is nested inside Turn wall time and is not added to it",
    }
    _store(root).write_summary(root / "reduced.json", summary)
    return summary


def _reduce_legacy(root: Path, manifest: dict[str, Any], runs: tuple[dict[str, Any], ...],
                   pairs: list[dict[str, Any]], infra: int, zero_ok: bool,
                   b_pass_c_fail: int, c_pass_b_fail: int, over_abstention: list[str],
                   fallbacks: int, utility_calls: int) -> dict[str, Any]:
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


def rehearse_pre_live(repository: Path, source_root: Path, failed_root: Path, prior_root: Path,
                      reviewer_id: str, *, config_path: Path | None = None) -> dict[str, Any]:
    """Exercise the production pre-live path and stop before the first Agent child."""
    from pico.config.loader import get_config_path

    parent_before = environment_fingerprint()
    config_path = (config_path or get_config_path()).resolve()
    with tempfile.TemporaryDirectory(prefix="jev4r2a-", dir=repository / ".tmp") as temporary:
        output_root = Path(temporary)
        root, manifest, audit = prepare_campaign(
            repository, output_root, source_root, failed_root, prior_root, reviewer_id,
            config_path=config_path,
        )
        artifact = root / "config-bootstrap.json"
        artifact_before = hashlib.sha256(artifact.read_bytes()).hexdigest()
        trace_log = root / "s" / "r00" / "logs" / "audit-events.log"
        trace_before = len(trace_log.read_text(encoding="utf-8").splitlines())
        validations = tuple(validate_config_bootstrap(root, config_path) for _ in range(3))
        artifact_after = hashlib.sha256(artifact.read_bytes()).hexdigest()
        trace_after = len(trace_log.read_text(encoding="utf-8").splitlines())
        bootstrap = validations[-1]
        readiness_checks = {
            "bootstrap_verified": bootstrap["passed"],
            "config_identity_verified": (
                bootstrap["config_identity_digest"] == manifest["config_identity_digest"]
            ),
            "config_source_immutable": bootstrap["config_source_unchanged"],
            "trace_canary_verified": bootstrap["trace_canary"]["passed"],
            "environment_baseline_verified": (
                parent_before["fingerprint_digest"] == environment_fingerprint()["fingerprint_digest"]
            ),
            "behavioral_inputs_immutable": audit["immutable_input_delta"] == "PASS",
            "mechanical_solvability": audit["mechanical_solvability"] == "3/3 PASS",
            "prompt_verifier_audit": audit["contract_audit"] == "PASS",
            "zero_selection_preflight": not audit["relevance_selected_candidate_ids"][TASKS[2].task_id],
            "bootstrap_artifact_immutable": artifact_before == artifact_after,
            "read_only_revalidation": trace_before == trace_after,
        }
        next_run = manifest["planned_runs"][0]
        result = {
            "schema": "pico.jev4r2a-pre-live-rehearsal.v1",
            "bootstrap_child_count": bootstrap["bootstrap_child_count"],
            "bootstrap_child_process_ids": (bootstrap["child_process_id"],),
            "bootstrap_lifecycle_state": bootstrap["lifecycle_state"],
            "bootstrap_lifecycle_transitions": bootstrap["lifecycle_transitions"],
            "bootstrap_artifact_digest_before": artifact_before,
            "bootstrap_artifact_digest_after": artifact_after,
            "parent_revalidation_count": len(validations),
            "additional_trace_event_count": trace_after - trace_before,
            "bootstrap_canary_id": bootstrap["trace_canary"]["canary_id"],
            "agent_canary_policy": "<campaign-id>:<run-id>:agent-canary",
            "readiness_checks": readiness_checks,
            "live_ready": all(readiness_checks.values()),
            "next_action": {"action": "spawn_agent_child", **next_run},
            "parent_provider_id": manifest["provider_id"],
            "parent_model_id": manifest["actual_model_id"],
            "child_provider_id": bootstrap["provider_id"],
            "child_model_id": bootstrap["model_id"],
            "config_identity_digest": manifest["config_identity_digest"],
            "config_source_unchanged": bootstrap["config_source_unchanged"],
            "secret_leak_free": bootstrap["secret_leak_free"],
            "parent_environment_unchanged": readiness_checks["environment_baseline_verified"],
            "main_provider_calls": bootstrap["provider_calls"],
            "utility_provider_calls": 0,
            "network_calls": bootstrap["network_calls"],
            "agent_turns": bootstrap["agent_turns"],
            "planned_total_children": 10,
            "live_execution_started": False,
        }
        if not result["live_ready"]:
            raise RuntimeError("JEV.4R2A pre-live rehearsal readiness gate failed")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.picobench.packs.knowledge_evolution_live.jev4_pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository", type=Path, default=Path.cwd())
    prepare.add_argument("--output-root", type=Path, default=Path(".p3r"))
    prepare.add_argument("--source-campaign-root", type=Path, required=True)
    prepare.add_argument("--failed-campaign-root", type=Path, required=True)
    prepare.add_argument("--prior-campaign-root", type=Path, required=True)
    prepare.add_argument("--reviewer-id", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--execute-live", action="store_true")
    reduce = commands.add_parser("reduce")
    reduce.add_argument("--campaign-root", type=Path, required=True)
    rehearse = commands.add_parser("rehearse")
    rehearse.add_argument("--repository", type=Path, default=Path.cwd())
    rehearse.add_argument("--source-campaign-root", type=Path, required=True)
    rehearse.add_argument("--failed-campaign-root", type=Path, required=True)
    rehearse.add_argument("--prior-campaign-root", type=Path, required=True)
    rehearse.add_argument("--reviewer-id", required=True)
    internal = commands.add_parser("_run-one")
    internal.add_argument("--repository", type=Path, required=True)
    internal.add_argument("--campaign-root", type=Path, required=True)
    internal.add_argument("--run-id", required=True)
    internal.add_argument("--reviewer-id", required=True)
    internal.add_argument("--config-path", type=Path, required=True)
    internal.add_argument("--execute-live", action="store_true")
    bootstrap = commands.add_parser("_bootstrap")
    bootstrap.add_argument("--campaign-root", type=Path, required=True)
    bootstrap.add_argument("--config-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        root, manifest, audit = prepare_campaign(
            args.repository.resolve(), args.output_root.resolve(),
            args.source_campaign_root.resolve(), args.failed_campaign_root.resolve(),
            args.prior_campaign_root.resolve(),
            args.reviewer_id,
        )
        print(canonical_json({"campaign_id": manifest["campaign_id"], "campaign_root": root,
                              "semantic_digest": manifest["campaign_semantic_digest"],
                              "preflight": audit, "live_provider_invoked": False}))
        return 0
    if args.command == "run":
        runs = run_campaign(args.repository.resolve(), args.campaign_root.resolve(), args.reviewer_id,
                            execute_live=args.execute_live)
        print(canonical_json({"completed_agent_runs": runs}))
        return 0
    if args.command == "rehearse":
        result = rehearse_pre_live(
            args.repository.resolve(), args.source_campaign_root.resolve(),
            args.failed_campaign_root.resolve(), args.prior_campaign_root.resolve(),
            args.reviewer_id,
        )
        print(canonical_json(result))
        return 0
    if args.command == "_run-one":
        from pico.config.loader import set_config_path

        if not args.execute_live:
            raise PermissionError("JEV.4R child execution requires explicit --execute-live")
        set_config_path(args.config_path.resolve())
        root = args.campaign_root.resolve()
        manifest = load_manifest(root)
        planned = next(
            (item for item in manifest["planned_runs"] if item["run_id"] == args.run_id),
            None,
        )
        if planned is None:
            raise ValueError(f"unknown JEV.4R run: {args.run_id}")
        record = execute_one(
            args.repository.resolve(), root, manifest, planned, args.reviewer_id,
            persist=False,
        )
        roots = AgentRunRoots.create(root, planned["run_id"], planned["order"])
        _store(root).write_summary(roots.artifact, record)
        return 0
    if args.command == "_bootstrap":
        from pico.config.loader import set_config_path

        set_config_path(args.config_path.resolve())
        root = args.campaign_root.resolve()
        record = _execute_config_bootstrap(root, load_manifest(root), args.config_path.resolve())
        _store(root).write_summary(root / "config-bootstrap.json", record)
        return 0 if record["passed"] else 2
    print(json.dumps(reduce_campaign(args.campaign_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Arm", "CAMPAIGN_SEED", "CONTRACT_AUDIT", "PRIMARY_COMPARISON", "SCHEMA",
    "TASKS", "UTILITY_TIMEOUT_SECONDS", "make_plan", "prepare_campaign", "preflight",
    "assert_infrastructure_only_rerun", "reduce_campaign", "run_campaign", "verify_workspace",
]
