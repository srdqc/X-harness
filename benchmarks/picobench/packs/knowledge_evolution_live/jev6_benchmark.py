"""Claim-eligible JEV.6B held-out Provider Utility benchmark runner.

Preparation and reduction are offline. Live execution is one-shot, follows the
frozen JEV.6A order, and requires the explicit ``--execute-live`` flag.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_digest, canonical_json, to_primitive
from benchmarks.picobench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.picobench.schema import ExperimentRef
from pico.config.paths import RuntimePaths
from pico.knowledge_evolution import KnowledgeSelectionMode
from pico.spine.message import ChatType, Source
from pico.spine.turn import Origin, TurnRequest
from pico.tracing import evidence

from .jev4_pilot import _config_source_digest, _sanitized_config_identity
from .jev6_freeze import evaluate_selector_audit
from .jev6_suite import (
    AGENT_BUDGET,
    BASE_COMMIT,
    BENEFIT_CRITERIA,
    MODEL_ID,
    PLANNED_LIVE_RUNS,
    PROVIDER_ID,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    SUITE_NAME,
    SUITE_VERSION,
    TASKS,
    UTILITY_MAX_OUTPUT_TOKENS,
    UTILITY_TIMEOUT_SECONDS,
    Arm,
    suite_payload,
)
from .jev6_verifiers import verify_task
from .knowledge import prepare_approved_corpus
from .metrics import extract_run_metrics
from .prepare import configuration_identity, resolve_base_commit
from .run_isolation import AgentRunRoots, environment_fingerprint, verify_trace_canary
from .runner import _changed_paths, _git, _workspace_patch
from .validity import classify_run_validity

SCHEMA = "pico.jev6b-confirmatory-campaign.v1"
SCHEMA_VERSION = 1
RUN_SCHEMA = "pico.jev6b-confirmatory-run.v1"
REDUCER_SCHEMA = "pico.jev6b-confirmatory-summary.v1"
REDUCER_VERSION = 1
FROZEN_MANIFEST = Path(__file__).with_name("jev6_frozen_manifest.json")
HISTORICAL_CAMPAIGNS = (
    "jev4-9168e909e0c47dfb",
    "jev4r-3f7c246afe462f5e",
    "jev4r2-868dd3997095b407",
    "jev4r3-384e7aefbe9cca17",
    "jev5a-00e65591d2467ac3",
)
FROZEN_SOURCE_FILES = (
    "jev6_suite.py",
    "jev6_fixtures.py",
    "jev6_verifiers.py",
    "jev6_freeze.py",
    "jev6_frozen_manifest.json",
)
PRIMARY_METRICS = (
    "combined_provider_logical_calls",
    "combined_input_tokens",
    "combined_output_tokens",
    "tool_calls",
    "repeated_repository_reads",
    "turn_latency_ms",
)


def _store(root: Path) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(root.name, root))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(path: Path) -> str:
    rows = tuple(
        (item.relative_to(path).as_posix(), _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )
    return canonical_digest(rows)


def historical_artifact_digests(repository: Path) -> dict[str, str]:
    roots = {name: repository / ".p3r" / name for name in HISTORICAL_CAMPAIGNS}
    missing = tuple(name for name, path in roots.items() if not path.is_dir())
    if missing:
        raise ValueError(f"historical campaign artifact missing: {','.join(missing)}")
    return {name: _tree_digest(path) for name, path in roots.items()}


def frozen_source_digests() -> dict[str, str]:
    return {
        name: _sha256(Path(__file__).with_name(name))
        for name in FROZEN_SOURCE_FILES
    }


def _task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def _stable_environment_identity(value: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: value[key]
        for key in (
            "python_executable_digest",
            "distribution_digest",
            "distribution_count",
            "user_site_enabled",
        )
    }
    # JSON round trips tuples as lists; normalize before bootstrap/run comparison.
    identity["python_version"] = list(value["python_version"])
    return identity


def verify_frozen_suite(repository: Path) -> dict[str, Any]:
    """Recompute every JEV.6A anti-tuning digest without changing frozen files."""
    frozen_bytes = FROZEN_MANIFEST.read_bytes()
    frozen = json.loads(frozen_bytes)
    temp_parent = repository / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jev6b-freeze-", dir=temp_parent) as temporary:
        selector = evaluate_selector_audit(
            state_root=Path(temporary), workspace=repository
        )
    payload = suite_payload(selector)
    expected = frozen["anti_tuning_digests"]
    actual = {
        "suite": payload["semantic_digest"],
        "task_set": payload["task_set_digest"],
        "verifier_set": payload["verifier_set_digest"],
        "fixtures": payload["fixture_set_digest"],
        "contract_audit": payload["contract_audit_digest"],
        "selector_audit": payload["selector_audit_digest"],
        "corpus": payload["corpus_digest"],
        "selector_config": payload["selector_config_digest"],
        "utility_prompt": payload["utility_prompt_digest"],
        "utility_inference_policy": payload["utility_inference_policy_digest"],
        "provider_model": payload["provider_model_digest"],
        "run_order": payload["run_order_digest"],
        "classification_policy": payload["benefit_criteria_digest"],
        "treatment_config": payload["treatment_config_digest"],
    }
    checks = {name: actual[name] == expected[name] for name in expected}
    checks.update(
        frozen_manifest_immutable=FROZEN_MANIFEST.read_bytes() == frozen_bytes,
        suite_version=frozen["suite_version"] == SUITE_VERSION,
        base_commit=frozen["base_commit"] == BASE_COMMIT,
        planned_runs=frozen["planned_live_runs"] == PLANNED_LIVE_RUNS == len(RUN_ORDER),
        run_order_constant=actual["run_order"] == RUN_ORDER_DIGEST,
        provider=frozen["provider_id"] == PROVIDER_ID,
        model=frozen["model_id"] == MODEL_ID,
    )
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"JEV.6 freeze mismatch: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "digests": actual,
        "selector_audit": selector,
        "frozen_manifest_file_digest": hashlib.sha256(frozen_bytes).hexdigest(),
    }


def verify_historical_task_blindness(repository: Path) -> dict[str, Any]:
    task_ids = {task.task_id for task in TASKS}
    exposures: list[tuple[str, str]] = []
    for path in (repository / ".p3r").glob("*/runs/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("task_id") in task_ids:
            exposures.append((path.as_posix(), str(value["task_id"])))
    if exposures:
        raise ValueError("JEV.6 held-out task has prior live Agent exposure")
    return {"passed": True, "prior_live_agent_exposures": (), "scanned_task_ids": tuple(sorted(task_ids))}


def _config_identity(config_path: Path) -> dict[str, Any]:
    value = _sanitized_config_identity(config_path, AGENT_BUDGET)
    if value["provider_id"] != PROVIDER_ID or value["model_id"] != MODEL_ID:
        raise ValueError("configured Provider/model differs from frozen JEV.6 identity")
    return value


def _campaign_semantic(freeze: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "reducer_schema": REDUCER_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "campaign_purpose": "claim_eligible_confirmatory_heldout_provider_utility",
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "base_commit_sha": BASE_COMMIT,
        "frozen_suite_digest": freeze["digests"]["suite"],
        "freeze_digests": freeze["digests"],
        "frozen_manifest_file_digest": freeze["frozen_manifest_file_digest"],
        "provider_id": PROVIDER_ID,
        "actual_model_id": MODEL_ID,
        "config_identity_digest": config["config_identity_digest"],
        "config_source_digest": config["config_source_digest"],
        "provider_model_config_digest": config["provider_model_config_digest"],
        "budget": asdict(AGENT_BUDGET),
        "planned_runs": tuple(asdict(item) for item in RUN_ORDER),
        "run_order_digest": RUN_ORDER_DIGEST,
        "frozen_selected_candidate_ids": freeze["selector_audit"][
            "task_selected_candidate_ids"
        ],
        "replacement_policy": "forbidden",
        "typesafe_enabled": False,
    }


def prepare_campaign(
    repository: Path,
    output_root: Path,
    reviewer_id: str,
    *,
    config_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from pico.config.loader import get_config_path

    base_sha = resolve_base_commit(repository, BASE_COMMIT)
    if base_sha != BASE_COMMIT:
        raise ValueError("JEV.6 frozen base commit does not resolve exactly")
    freeze = verify_frozen_suite(repository)
    blindness = verify_historical_task_blindness(repository)
    config_path = (config_path or get_config_path()).resolve()
    config = _config_identity(config_path)
    semantic = _campaign_semantic(freeze, config)
    semantic_digest = canonical_digest(semantic)
    campaign_id = f"jev6b-{semantic_digest[:16]}"
    root = output_root / campaign_id
    if root.exists():
        raise FileExistsError(f"JEV.6B campaign already exists: {root}")
    root.mkdir(parents=True)
    manifest = {
        **semantic,
        "campaign_id": campaign_id,
        "campaign_semantic_digest": semantic_digest,
        "reviewer_id_digest": canonical_digest(reviewer_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_artifact_digests_before": historical_artifact_digests(repository),
        "frozen_source_digests_before": frozen_source_digests(),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    _store(root).freeze_manifest(manifest)
    bootstrap = run_bootstrap(repository, root, config_path)
    readiness_checks = {
        "freeze_verified": freeze["passed"],
        "historical_task_blindness": blindness["passed"],
        "provider_model_resolved": (
            bootstrap.get("provider_id") == PROVIDER_ID
            and bootstrap.get("model_id") == MODEL_ID
        ),
        "config_identity_verified": bootstrap.get("config_identity_digest") == config["config_identity_digest"],
        "config_source_immutable": bootstrap.get("config_source_unchanged") is True,
        "trace_canary_verified": bootstrap.get("trace_canary", {}).get("passed") is True,
        "environment_fingerprint_recorded": bool(bootstrap.get("stable_environment_identity")),
        "no_live_activity": all(bootstrap.get(name) == 0 for name in ("provider_calls", "agent_turns", "network_calls")),
    }
    readiness = {
        "schema": "pico.jev6b-pre-live-readiness.v1",
        "campaign_id": campaign_id,
        "checks": readiness_checks,
        "live_ready": all(readiness_checks.values()),
        "freeze_verification": freeze,
        "historical_task_blindness": blindness,
        "bootstrap_artifact_digest": _sha256(root / "config-bootstrap.json"),
        "next_run": semantic["planned_runs"][0],
        "provider_calls": 0,
        "agent_turns": 0,
        "network_calls": 0,
    }
    readiness["integrity_digest"] = canonical_digest(readiness)
    _store(root).write_summary(root / "pre-live-readiness.json", readiness)
    if not readiness["live_ready"]:
        raise RuntimeError("JEV.6B authoritative pre-live readiness failed")
    return root, manifest, readiness


def load_manifest(root: Path) -> dict[str, Any]:
    value = _store(root).read_json(root / "manifest.json")
    digest = value.pop("manifest_digest", None)
    if digest != canonical_digest(value):
        raise ValueError("JEV.6B manifest digest mismatch")
    value["manifest_digest"] = digest
    if (value.get("schema"), value.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ValueError("unsupported JEV.6B campaign schema")
    semantic = _campaign_semantic(
        {
            "digests": value["freeze_digests"],
            "frozen_manifest_file_digest": value["frozen_manifest_file_digest"],
            "selector_audit": {
                "task_selected_candidate_ids": value["frozen_selected_candidate_ids"]
            },
        },
        {
            "config_identity_digest": value["config_identity_digest"],
            "config_source_digest": value["config_source_digest"],
            "provider_model_config_digest": value["provider_model_config_digest"],
        },
    )
    if value["campaign_semantic_digest"] != canonical_digest(semantic):
        raise ValueError("JEV.6B campaign semantic digest mismatch")
    return value


def _bootstrap_record(root: Path, config_path: Path) -> dict[str, Any]:
    manifest = load_manifest(root)
    roots = AgentRunRoots.create(root, "config-bootstrap", 0)
    roots.prepare_non_worktree_roots()
    before = _config_source_digest(config_path)
    try:
        identity = _config_identity(config_path)
        canary = verify_trace_canary(roots.trace, canary_id=f"{manifest['campaign_id']}:bootstrap")
        failure_type = None
    except Exception as exc:  # noqa: BLE001 - bounded offline bootstrap boundary
        identity = {}
        canary = {"passed": False, "findings": ("bootstrap_failure",)}
        failure_type = type(exc).__name__
    after = _config_source_digest(config_path)
    fingerprint = environment_fingerprint()
    record = {
        "schema": "pico.jev6b-config-bootstrap.v1",
        "campaign_id": manifest["campaign_id"],
        "child_process_id": os.getpid(),
        "passed": bool(
            identity.get("config_identity_digest") == manifest["config_identity_digest"]
            and before == after == manifest["config_source_digest"]
            and canary.get("passed")
            and os.environ.get("PYTHONNOUSERSITE") == "1"
        ),
        "provider_id": identity.get("provider_id"),
        "model_id": identity.get("model_id"),
        "config_identity_digest": identity.get("config_identity_digest"),
        "config_source_digest_before": before,
        "config_source_digest_after": after,
        "config_source_unchanged": before == after,
        "trace_canary": canary,
        "stable_environment_identity": _stable_environment_identity(fingerprint),
        "failure_type": failure_type,
        "provider_calls": 0,
        "agent_turns": 0,
        "network_calls": 0,
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def run_bootstrap(repository: Path, root: Path, config_path: Path) -> dict[str, Any]:
    roots = AgentRunRoots.create(root, "config-bootstrap", 0)
    roots.prepare_non_worktree_roots()
    artifact = root / "config-bootstrap.json"
    if artifact.exists():
        raise FileExistsError("JEV.6B bootstrap is immutable")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.picobench.packs.knowledge_evolution_live.jev6_benchmark",
            "_bootstrap",
            "--repository",
            str(repository),
            "--campaign-root",
            str(root),
            "--config-path",
            str(config_path),
        ],
        cwd=repository,
        env=roots.child_environment(),
        check=False,
    )
    if completed.returncode != 0 or not artifact.is_file():
        raise RuntimeError("JEV.6B child bootstrap failed")
    record = _store(root).read_json(artifact)
    digest = record.pop("integrity_digest", None)
    if digest != canonical_digest(record):
        raise ValueError("JEV.6B bootstrap integrity mismatch")
    record["integrity_digest"] = digest
    if not record["passed"]:
        raise RuntimeError("JEV.6B child bootstrap did not pass")
    return record


def _load_readiness(root: Path) -> dict[str, Any]:
    value = _store(root).read_json(root / "pre-live-readiness.json")
    digest = value.pop("integrity_digest", None)
    if digest != canonical_digest(value):
        raise ValueError("JEV.6B readiness integrity mismatch")
    value["integrity_digest"] = digest
    return value


def _infra_record(manifest: dict[str, Any], planned: dict[str, Any], reason: str) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        **planned,
        "run_validity": "infra_invalid",
        "infra_invalid_reason": reason,
        "run_outcome": "INFRA_INVALID",
        "verified_success": False,
        "verifier_findings": (),
        "main_agent_metrics": {},
        "utility_metrics": {},
        "combined_provider_cost": {},
        "changed_paths": (),
        "typesafe_enabled": False,
        "mandatory_evidence": {"complete": False, "findings": ("run_not_completed",)},
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def _write_run(root: Path, record: dict[str, Any]) -> None:
    _store(root).append_immutable(root / "runs" / f"{record['run_id']}.json", record)


def load_runs(root: Path) -> tuple[dict[str, Any], ...]:
    values = []
    for path in (root / "runs").glob("*.json"):
        value = _store(root).read_json(path)
        digest = value.pop("integrity_digest", None)
        if digest != canonical_digest(value):
            raise ValueError(f"JEV.6B run integrity mismatch: {path.name}")
        value["integrity_digest"] = digest
        values.append(value)
    return tuple(sorted(values, key=lambda item: item["order"]))


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


def _metric_value(metrics: dict[str, Any], name: str) -> int | float | None:
    value = metrics.get(name)
    return value.get("value") if isinstance(value, dict) else None


def _sum_optional(left: int | float | None, right: int | float | None) -> int | float | None:
    return None if left is None or right is None else left + right


def _utility_projection(refs: dict[str, object]) -> dict[str, Any]:
    decisions = tuple(dict(item) for item in refs.get("utility_decisions", ()))
    raw = Counter(str(item.get("decision", "")).upper() for item in decisions)
    effective = Counter(str(item.get("effective_decision", "")).upper() for item in decisions)
    selected = len(tuple(refs["relevance_selected_candidate_ids"]))
    fallback_count = int(refs["utility_fallback_count"])
    all_abstain = selected > 0 and not fallback_count and len(decisions) == selected and raw["ABSTAIN"] == selected
    unavailable = selected > 0 and fallback_count > 0 and not decisions
    return {
        "relevance_selected_count": selected,
        "utility_invoked": int(refs["utility_invoked_count"]) > 0,
        "utility_logical_calls": int(refs["utility_provider_calls"]),
        "utility_provider_attempts": int(refs["utility_provider_attempts"]),
        "candidate_count": int(refs.get("utility_candidate_count", 0)),
        "decisions": decisions,
        "raw_choice_counts": {key: raw[key] for key in ("KEEP", "ABSTAIN", "UNCERTAIN")},
        "effective_choice_counts": {key: effective[key] for key in ("KEEP", "ABSTAIN")},
        "kept_candidate_ids": refs["utility_kept_candidate_ids"],
        "abstained_candidate_ids": refs["utility_abstained_candidate_ids"],
        "fallback_count": fallback_count,
        "fallback_reasons": refs["utility_fallback_reasons"],
        "utility_decision_unavailable": unavailable,
        "all_abstain": all_abstain,
        "input_tokens": refs["utility_input_tokens"],
        "output_tokens": refs["utility_output_tokens"],
        "provider_latency_ms": refs["utility_provider_latency_ms"],
        "total_latency_ms": refs["utility_latency_ms"],
        "finish_reasons": refs.get("utility_finish_reasons", ()),
        "malformed_categories": refs.get("utility_malformed_categories", ()),
        "generation_outcomes": refs.get("utility_generation_outcomes", ()),
        "payload_outcomes": refs.get("utility_payload_outcomes", ()),
        "reasoning_tokens": refs.get("utility_reasoning_tokens", ()),
        "visible_output_tokens": refs.get("utility_visible_output_tokens", ()),
    }


def _combined_cost(main: dict[str, Any], utility: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_logical_calls": _sum_optional(_metric_value(main, "provider_logical_calls"), utility["utility_logical_calls"]),
        "provider_attempts": _sum_optional(_metric_value(main, "provider_attempts"), utility["utility_provider_attempts"]),
        "input_tokens": _sum_optional(_metric_value(main, "input_tokens"), utility["input_tokens"]),
        "output_tokens": _sum_optional(_metric_value(main, "output_tokens"), utility["output_tokens"]),
        "latency_semantics": "utility latency is nested inside Turn latency and is not added",
    }


async def _execute_turn(
    root: Path,
    manifest: dict[str, Any],
    planned: dict[str, Any],
    worktree: Path,
    state: Path,
    reviewer_id: str,
) -> dict[str, Any]:
    from pico.cli._helpers import make_provider
    from pico.config.loader import load_config
    from pico.config.pico import PicoConfig, load_pico_config
    from pico.proactive_engine.schedulers.cron.service import CronService

    config = load_config()
    identity = configuration_identity(config, AGENT_BUDGET)
    if (
        identity["provider_id"] != PROVIDER_ID
        or identity["model"] != MODEL_ID
        or identity["provider_digest"] != manifest["provider_model_config_digest"]
    ):
        raise RuntimeError("configured Runtime/Provider differs from frozen JEV.6 campaign")
    arm = Arm(planned["arm"])
    pico_config = load_pico_config()
    context = pico_config.context.model_dump(mode="python")
    if arm is Arm.TASK_RELEVANCE_V1:
        context["knowledge_selection_mode"] = KnowledgeSelectionMode.TASK_RELEVANCE_V1.value
    else:
        context.update(
            knowledge_selection_mode=KnowledgeSelectionMode.TASK_RELEVANCE_V1_JEV_UTILITY.value,
            knowledge_utility_backend="provider",
            knowledge_utility_provider="agent",
            knowledge_utility_model=MODEL_ID,
            knowledge_utility_timeout_seconds=UTILITY_TIMEOUT_SECONDS,
            knowledge_utility_max_tokens=UTILITY_MAX_OUTPUT_TOKENS,
        )
    pico_config = PicoConfig.model_validate(
        {**pico_config.model_dump(mode="python"), "memory": {"backend": None}, "context": context}
    )
    prepare_approved_corpus(state_root=state, workspace=worktree, reviewer_id=reviewer_id)
    started_at = datetime.now(timezone.utc).isoformat()
    turn_id = f"{planned['run_id']}-turn"
    runtime_outcome = "timeout"
    provider = make_provider(config)
    outlet = RecordingOutlet("jev6b-live")
    cron = CronService(state / "cron" / "jobs.json", allowed_channels={"jev6b-live"})
    host = await RuntimeTrialHost.build(
        config=config,
        pico_config=pico_config,
        provider=provider,
        cron_service=cron,
        outlet=outlet,
        paths=RuntimePaths(workspace=worktree, state=state),
        turn_id_factory=lambda: turn_id,
    )
    try:
        request = TurnRequest(
            origin=Origin.USER,
            source=Source("jev6b-live", planned["run_id"], "jev6-human", ChatType.DM),
            text=_task_by_id(planned["task_id"]).prompt,
            conversation=f"jev6b:{planned['run_id']}",
        )
        try:
            observation = await asyncio.wait_for(
                host.run(request), timeout=AGENT_BUDGET.wall_clock_timeout_seconds
            )
        except TimeoutError:
            pass
        else:
            turn_id = observation.turn_id
            runtime_outcome = observation.runtime_state.value
    finally:
        await host.close()
    terminal_at = datetime.now(timezone.utc).isoformat()
    verified = verify_task(planned["task_id"], worktree, python_executable=sys.executable)
    patch = _workspace_patch(worktree)
    patch_digest = canonical_digest(patch)
    patch_path = root / "patches" / f"{planned['run_id']}.json"
    _store(root).append_immutable(
        patch_path, {"run_id": planned["run_id"], "patch_digest": patch_digest, **patch}
    )
    metrics, refs = extract_run_metrics(
        trace_root=state, turn_id=turn_id, knowledge_state_root=state
    )
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
        mandatory_evidence_complete=mandatory["complete"],
    )
    utility = _utility_projection(refs)
    main = to_primitive(metrics)
    combined = _combined_cost(main, utility)
    record = {
        "schema": RUN_SCHEMA,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        **planned,
        "workspace_identity": f"git:{BASE_COMMIT}:{planned['run_id']}",
        "turn_id": turn_id,
        "started_at": started_at,
        "terminal_at": terminal_at,
        "runtime_outcome": runtime_outcome,
        "run_validity": validity.validity.value,
        "infra_invalid_reason": validity.reason.value if validity.reason else None,
        "verified_success": bool(verified.passed),
        "run_outcome": (
            "INFRA_INVALID"
            if validity.validity.value == "infra_invalid"
            else "PASS" if verified.passed else "FAIL"
        ),
        "mandatory_evidence": mandatory,
        "verifier_findings": verified.findings,
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
        "typesafe_enabled": False,
        "provider_id": identity["provider_id"],
        "model_id": identity["model"],
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def execute_one(
    repository: Path,
    root: Path,
    manifest: dict[str, Any],
    planned: dict[str, Any],
    reviewer_id: str,
    *,
    config_path: Path,
) -> dict[str, Any]:
    run_id = planned["run_id"]
    roots = AgentRunRoots.create(root, run_id, planned["order"])
    roots.prepare_non_worktree_roots()
    try:
        config = _config_identity(config_path)
    except Exception as exc:  # noqa: BLE001 - bounded identity gate
        config = {"failure_type": type(exc).__name__}
    bootstrap = _store(root).read_json(root / "config-bootstrap.json")
    bootstrap_digest = bootstrap.pop("integrity_digest", None)
    if bootstrap_digest != canonical_digest(bootstrap):
        raise ValueError("JEV.6B bootstrap integrity mismatch")
    fingerprint = environment_fingerprint()
    env_match = _stable_environment_identity(fingerprint) == bootstrap["stable_environment_identity"]
    canary = verify_trace_canary(
        roots.trace,
        canary_id=f"{manifest['campaign_id']}:{run_id}:agent-canary",
        other_trace_roots=tuple(
            path for path in (root / "s").glob("r*") if path.resolve() != roots.trace.resolve()
        ),
    )
    prechecks = {
        "config_identity": config.get("config_identity_digest") == manifest["config_identity_digest"],
        "provider_identity": config.get("provider_id") == PROVIDER_ID and config.get("model_id") == MODEL_ID,
        "environment_fingerprint": env_match,
        "trace_canary": canary["passed"],
        "run_local_evidence_root": str(Path(os.environ.get("PICO_HOME", "")).resolve()).startswith(str(roots.state.resolve())),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
    }
    if not all(prechecks.values()):
        record = _infra_record(manifest, planned, "pre_run_canary_failure")
        record.update(
            pre_run_checks=prechecks,
            trace_canary=canary,
            isolation_roots=roots.public_identity(),
            child_process_id=os.getpid(),
            environment_fingerprint=fingerprint,
        )
        record["integrity_digest"] = canonical_digest(
            {key: value for key, value in record.items() if key != "integrity_digest"}
        )
        _store(root).write_summary(roots.artifact, record)
        return record
    try:
        _git(repository, "worktree", "add", "--detach", str(roots.worktree), BASE_COMMIT)
    except (OSError, RuntimeError):
        record = _infra_record(manifest, planned, "worktree_setup_failure")
    else:
        try:
            record = asyncio.run(
                _execute_turn(root, manifest, planned, roots.worktree, roots.state, reviewer_id)
            )
        except Exception as exc:  # noqa: BLE001 - immutable host boundary
            record = _infra_record(manifest, planned, "benchmark_host_failure")
            record["bounded_host_failure_type"] = type(exc).__name__
        finally:
            if roots.worktree.exists():
                _git(repository, "worktree", "remove", "--force", str(roots.worktree))
    record.update(
        pre_run_checks=prechecks,
        trace_canary=canary,
        isolation_roots=roots.public_identity(),
        child_process_id=os.getpid(),
        environment_fingerprint=fingerprint,
    )
    record["integrity_digest"] = canonical_digest(
        {key: value for key, value in record.items() if key != "integrity_digest"}
    )
    _store(root).write_summary(roots.artifact, record)
    return record


def _selection_failure(manifest: dict[str, Any], record: dict[str, Any]) -> str | None:
    expected = tuple(manifest["frozen_selected_candidate_ids"][record["task_id"]])
    actual = tuple(record.get("relevance_selected_candidate_ids", ()))
    if actual != expected:
        return "selector_drift"
    utility_calls = int(record.get("utility_metrics", {}).get("utility_logical_calls", 0))
    if record["arm"] == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value:
        if not expected and utility_calls != 0:
            return "zero_selection_utility_invoked"
        if expected and utility_calls != 1:
            return "utility_one_call_contract_failure"
    elif utility_calls != 0:
        return "baseline_utility_invoked"
    return None


def run_campaign(
    repository: Path,
    root: Path,
    reviewer_id: str,
    *,
    execute_live: bool,
) -> tuple[str, ...]:
    from pico.config.loader import get_config_path

    if not execute_live:
        raise PermissionError("JEV.6B live execution requires explicit --execute-live")
    manifest = load_manifest(root)
    if tuple(manifest["planned_runs"]) != tuple(asdict(item) for item in RUN_ORDER):
        raise ValueError("JEV.6B frozen run order changed")
    if (root / "runs").exists() and tuple((root / "runs").glob("*.json")):
        raise FileExistsError("JEV.6B is one-shot; existing runs forbid resume/replacement")
    freeze = verify_frozen_suite(repository)
    if freeze["digests"] != manifest["freeze_digests"]:
        raise ValueError("JEV.6B freeze changed after preparation")
    verify_historical_task_blindness(repository)
    config_path = get_config_path().resolve()
    if _config_identity(config_path)["config_identity_digest"] != manifest["config_identity_digest"]:
        raise ValueError("JEV.6B Provider config changed after preparation")
    readiness = _load_readiness(root)
    if not readiness["live_ready"] or not all(readiness["checks"].values()):
        raise RuntimeError("JEV.6B campaign is not live-ready")
    completed: list[str] = []
    for planned in manifest["planned_runs"]:
        roots = AgentRunRoots.create(root, planned["run_id"], planned["order"])
        roots.prepare_non_worktree_roots()
        parent_before = environment_fingerprint()
        config_before = _config_source_digest(config_path)
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.picobench.packs.knowledge_evolution_live.jev6_benchmark",
                "_run-one",
                "--repository",
                str(repository),
                "--campaign-root",
                str(root),
                "--run-id",
                planned["run_id"],
                "--reviewer-id",
                reviewer_id,
                "--config-path",
                str(config_path),
                "--execute-live",
            ],
            cwd=repository,
            env=roots.child_environment(),
            check=False,
        )
        parent_after = environment_fingerprint()
        config_after = _config_source_digest(config_path)
        record = (
            _store(root).read_json(roots.artifact)
            if roots.artifact.exists()
            else _infra_record(manifest, planned, "benchmark_host_failure")
        )
        record.pop("integrity_digest", None)
        record["parent_environment_before"] = parent_before
        record["parent_environment_after"] = parent_after
        record["environment_drift_detected"] = parent_before["fingerprint_digest"] != parent_after["fingerprint_digest"]
        record["config_source_unchanged"] = config_before == config_after == manifest["config_source_digest"]
        if process.returncode != 0:
            record.update(run_validity="infra_invalid", infra_invalid_reason="benchmark_host_failure", verified_success=False)
        elif record["environment_drift_detected"]:
            record.update(run_validity="infra_invalid", infra_invalid_reason="environment_drift", verified_success=False)
        elif not record["config_source_unchanged"]:
            record.update(run_validity="infra_invalid", infra_invalid_reason="config_source_mutation", verified_success=False)
        already_invalid = record["run_validity"] == "infra_invalid"
        isolation = (
            {"passed": False, "reason": "run_did_not_reach_turn"}
            if already_invalid
            else _cross_run_isolation(root, record)
        )
        record["cross_run_isolation"] = isolation
        if not already_invalid and not isolation["passed"]:
            record.update(run_validity="infra_invalid", infra_invalid_reason="cross_run_trace_contamination", verified_success=False)
        selection_failure = None if record["run_validity"] == "infra_invalid" else _selection_failure(manifest, record)
        if selection_failure:
            record.update(run_validity="infra_invalid", infra_invalid_reason=selection_failure, verified_success=False)
        if record.get("provider_id") not in {None, PROVIDER_ID} or record.get("model_id") not in {None, MODEL_ID}:
            record.update(run_validity="infra_invalid", infra_invalid_reason="provider_identity_mismatch", verified_success=False)
        record["run_outcome"] = (
            "INFRA_INVALID"
            if record["run_validity"] == "infra_invalid"
            else "PASS" if record["verified_success"] else "FAIL"
        )
        record["integrity_digest"] = canonical_digest(record)
        _write_run(root, record)
        completed.append(record["run_id"])
        print(canonical_json({"completed": len(completed), "run_id": record["run_id"], "outcome": record["run_outcome"]}), flush=True)
        if record["run_validity"] == "infra_invalid":
            raise RuntimeError("JEV.6B stopped after INFRA_INVALID; replacement runs are forbidden")
    return tuple(completed)


def _value(record: dict[str, Any], metric: str) -> int | float | None:
    main = record.get("main_agent_metrics", {})
    combined = record.get("combined_provider_cost", {})
    if metric == "combined_provider_logical_calls":
        return combined.get("provider_logical_calls")
    if metric == "combined_input_tokens":
        return combined.get("input_tokens")
    if metric == "combined_output_tokens":
        return combined.get("output_tokens")
    if metric == "tool_calls":
        return _metric_value(main, "tool_calls_total")
    if metric == "repeated_repository_reads":
        return _metric_value(main, "repeated_repo_file_reads")
    if metric == "turn_latency_ms":
        return _metric_value(main, "turn_latency_ms")
    raise KeyError(metric)


def _median(values: Sequence[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if len(clean) == len(values) and clean else None


def _p95(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _diagnostic_subset(runs: tuple[dict[str, Any], ...], task_ids: set[str]) -> dict[str, Any]:
    subset = tuple(item for item in runs if item["task_id"] in task_ids)
    return {
        "task_ids": tuple(sorted(task_ids)),
        "B_success": sum(item["verified_success"] for item in subset if item["arm"] == Arm.TASK_RELEVANCE_V1.value),
        "C_success": sum(item["verified_success"] for item in subset if item["arm"] == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value),
        "run_count": len(subset),
    }


def reduce_campaign(root: Path, repository: Path) -> dict[str, Any]:
    manifest = load_manifest(root)
    runs = load_runs(root)
    by_key = {(item["task_id"], item["arm"], item["repetition"]): item for item in runs}
    per_task: dict[str, Any] = {}
    pair_outcomes = Counter()
    intersections = Counter()
    task_deltas: dict[str, list[float]] = {metric: [] for metric in PRIMARY_METRICS}
    for task in TASKS:
        arms: dict[str, Any] = {}
        for arm in Arm:
            arm_runs = tuple(by_key.get((task.task_id, arm.value, rep)) for rep in range(1, 4))
            arms[arm.value] = {
                "success_count": sum(bool(item and item["verified_success"]) for item in arm_runs),
                "run_ids": tuple(item["run_id"] if item else None for item in arm_runs),
                "medians": {
                    metric: _median(tuple(_value(item, metric) if item else None for item in arm_runs))
                    for metric in PRIMARY_METRICS
                },
            }
        deltas = {}
        for metric in PRIMARY_METRICS:
            b = arms[Arm.TASK_RELEVANCE_V1.value]["medians"][metric]
            c = arms[Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value]["medians"][metric]
            relative = None if b in {None, 0} or c is None else (c - b) / b
            deltas[metric] = {"absolute": None if b is None or c is None else c - b, "relative": relative}
            if relative is not None:
                task_deltas[metric].append(relative)
        per_task[task.task_id] = {"arms": arms, "paired_deltas": deltas}
        for rep in range(1, 4):
            b = by_key.get((task.task_id, Arm.TASK_RELEVANCE_V1.value, rep))
            c = by_key.get((task.task_id, Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value, rep))
            if not b or not c or "infra_invalid" in {b["run_validity"], c["run_validity"]}:
                pair_outcomes["INFRA_INVALID_OR_MISSING"] += 1
                continue
            outcome = f"B_{'PASS' if b['verified_success'] else 'FAIL'}_C_{'PASS' if c['verified_success'] else 'FAIL'}"
            pair_outcomes[outcome] += 1
            removed = bool(c.get("utility_metrics", {}).get("abstained_candidate_ids"))
            fallback = bool(c.get("utility_metrics", {}).get("fallback_count"))
            if not c["verified_success"] and removed:
                intersections["C_failures_with_abstain"] += 1
            if not c["verified_success"] and fallback:
                intersections["C_failures_with_fallback"] += 1
            if b["verified_success"] and not c["verified_success"] and removed:
                intersections["B_PASS_C_FAIL_utility_removed"] += 1
            if c["verified_success"] and not b["verified_success"] and removed:
                intersections["C_PASS_B_FAIL_utility_removed"] += 1
    c_runs = tuple(item for item in runs if item["arm"] == Arm.TASK_RELEVANCE_V1_PROVIDER_UTILITY.value)
    utility_calls = sum(int(item.get("utility_metrics", {}).get("utility_logical_calls", 0)) for item in c_runs)
    choices = Counter()
    generation = Counter()
    payload = Counter()
    reasoning: list[int] = []
    visible_output: list[int] = []
    utility_output: list[int] = []
    utility_latency: list[float] = []
    fallbacks = unavailable = all_abstain = retained = removed = length = 0
    for item in c_runs:
        utility = item.get("utility_metrics", {})
        choices.update(utility.get("raw_choice_counts", {}))
        retained += int(utility.get("effective_choice_counts", {}).get("KEEP", 0))
        removed += int(utility.get("effective_choice_counts", {}).get("ABSTAIN", 0))
        fallbacks += int(utility.get("fallback_count", 0))
        unavailable += int(bool(utility.get("utility_decision_unavailable")))
        all_abstain += int(bool(utility.get("all_abstain")))
        generation.update(str(value) for value in utility.get("generation_outcomes", ()))
        payload.update(str(value) for value in utility.get("payload_outcomes", ()))
        reasoning.extend(int(value) for value in utility.get("reasoning_tokens", ()))
        visible_output.extend(int(value) for value in utility.get("visible_output_tokens", ()))
        if isinstance(utility.get("output_tokens"), int):
            utility_output.append(int(utility["output_tokens"]))
        if utility.get("utility_logical_calls"):
            utility_latency.append(float(utility.get("provider_latency_ms", 0.0)))
        length += sum(str(value).upper() == "LENGTH" for value in utility.get("finish_reasons", ()))
    zero_ids = {task_id for task_id, selected in manifest["frozen_selected_candidate_ids"].items() if not selected}
    nonzero_ids = {task.task_id for task in TASKS} - zero_ids
    infra = sum(item["run_validity"] == "infra_invalid" for item in runs)
    infra_reasons = Counter(
        str(item.get("infra_invalid_reason") or "unknown")
        for item in runs
        if item["run_validity"] == "infra_invalid"
    )
    pre_run_failures = Counter(
        name
        for item in runs
        for name, passed in item.get("pre_run_checks", {}).items()
        if passed is False
    )
    fairness_checks = {
        "complete_frozen_order": tuple(item["run_id"] for item in runs) == tuple(item["run_id"] for item in manifest["planned_runs"]),
        "selector_identity": all(tuple(item.get("relevance_selected_candidate_ids", ())) == tuple(manifest["frozen_selected_candidate_ids"][item["task_id"]]) for item in runs),
        "provider_model_identity": all(item.get("provider_id") == PROVIDER_ID and item.get("model_id") == MODEL_ID for item in runs),
        "zero_selection_controls": all(not item.get("relevance_selected_candidate_ids") and int(item.get("utility_metrics", {}).get("utility_logical_calls", 0)) == 0 for item in c_runs if item["task_id"] in zero_ids),
    }
    safety_checks = {
        "typesafe_disabled": all(item.get("typesafe_enabled") is False for item in runs),
        "utility_at_most_one_call": all(int(item.get("utility_metrics", {}).get("utility_logical_calls", 0)) <= 1 for item in c_runs),
        "no_replacement_runs": len(runs) <= PLANNED_LIVE_RUNS,
    }
    infrastructure_checks = {
        "all_48_runs": len(runs) == PLANNED_LIVE_RUNS,
        "infra_invalid_zero": infra == 0,
        "unique_child_process": len({item.get("child_process_id") for item in runs}) == len(runs),
        "trace_canaries": all(item.get("trace_canary", {}).get("passed") for item in runs),
        "mandatory_evidence": all(item.get("mandatory_evidence", {}).get("complete") for item in runs),
        "environment_drift_zero": all(not item.get("environment_drift_detected", True) for item in runs),
        "cross_run_isolation": all(item.get("cross_run_isolation", {}).get("passed") for item in runs),
        "config_source_immutable": all(item.get("config_source_unchanged") for item in runs),
    }
    suite_deltas = {
        metric: _median(values) for metric, values in task_deltas.items()
    }
    b_success = sum(item["verified_success"] for item in runs if item["arm"] == Arm.TASK_RELEVANCE_V1.value)
    c_success = sum(item["verified_success"] for item in c_runs)
    repeated_regression = pair_outcomes["B_PASS_C_FAIL"] >= 2
    improvements = tuple(value for value in suite_deltas.values() if value is not None and value <= -0.15)
    regressions = tuple(value for value in suite_deltas.values() if value is not None and value > 0.20)
    utility_contract_valid = all(safety_checks.values()) and unavailable <= utility_calls
    infra_valid = all(infrastructure_checks.values())
    fairness_valid = all(fairness_checks.values())
    safety_valid = all(safety_checks.values())
    if not (infra_valid and fairness_valid and safety_valid and utility_contract_valid):
        classification = "INVALID"
    elif c_success < b_success or repeated_regression or (len(regressions) >= 2 and not improvements):
        classification = "REGRESSIVE"
    elif c_success >= b_success and not repeated_regression and improvements and len(regressions) < 2:
        classification = "BENEFICIAL"
    else:
        classification = "NEUTRAL"
    historical_after = historical_artifact_digests(repository)
    frozen_after = frozen_source_digests()
    summary = {
        "schema": REDUCER_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "campaign_id": manifest["campaign_id"],
        "suite_version": SUITE_VERSION,
        "base_commit": BASE_COMMIT,
        "planned_runs": PLANNED_LIVE_RUNS,
        "actual_runs": len(runs),
        "infra_invalid_count": infra,
        "infra_invalid_reasons": dict(infra_reasons),
        "pre_run_check_failures": dict(pre_run_failures),
        "environment_drift_count": sum(item.get("environment_drift_detected", False) for item in runs),
        "trace_canary_failure_count": sum(not item.get("trace_canary", {}).get("passed", False) for item in runs),
        "B_success": b_success,
        "C_success": c_success,
        "per_task": per_task,
        "paired_outcomes": dict(pair_outcomes),
        "zero_selection_diagnostic": _diagnostic_subset(runs, zero_ids),
        "nonzero_selection_diagnostic": _diagnostic_subset(runs, nonzero_ids),
        "relevance_selected_counts": {task_id: len(values) for task_id, values in manifest["frozen_selected_candidate_ids"].items()},
        "utility_invocation_count": utility_calls,
        "utility_choices": {key: choices[key] for key in ("KEEP", "ABSTAIN", "UNCERTAIN")},
        "effective_retained": retained,
        "effective_removed": removed,
        "fallback_count": fallbacks,
        "fallback_rate": fallbacks / utility_calls if utility_calls else 0.0,
        "utility_decision_unavailable_count": unavailable,
        "all_abstain_count": all_abstain,
        "reasoning_tokens": {"total": sum(reasoning), "median": _median(reasoning), "p95": _p95(reasoning), "max": max(reasoning, default=0)},
        "utility_output_tokens": {"total": sum(utility_output), "median": _median(utility_output), "visible_total": sum(visible_output)},
        "utility_latency_ms": {"total": sum(utility_latency), "median": _median(utility_latency), "p95": _p95(utility_latency), "max": max(utility_latency, default=0.0)},
        "length_count": length,
        "generation_outcomes": dict(generation),
        "payload_outcomes": dict(payload),
        "suite_level_median_paired_relative_deltas": suite_deltas,
        "correctness_utility_intersections": dict(intersections),
        "fairness_checks": fairness_checks,
        "fairness_valid": fairness_valid,
        "safety_checks": safety_checks,
        "safety_valid": safety_valid,
        "infrastructure_checks": infrastructure_checks,
        "infrastructure_valid": infra_valid,
        "utility_contract_valid": utility_contract_valid,
        "classification": classification,
        "frozen_benefit_criteria": asdict(BENEFIT_CRITERIA),
        "historical_artifact_compatibility": historical_after == manifest["historical_artifact_digests_before"],
        "frozen_source_compatibility": frozen_after == manifest["frozen_source_digests_before"],
        "historical_artifact_digests_after": historical_after,
        "frozen_source_digests_after": frozen_after,
        "rerun_count": 0,
        "replacement_count": 0,
        "typesafe_enabled": False,
        "latency_semantics": "utility latency is nested inside Turn latency and is not added",
    }
    summary["integrity_digest"] = canonical_digest(summary)
    _store(root).write_summary(root / "reduced.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.picobench.packs.knowledge_evolution_live.jev6_benchmark")
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
    reduce.add_argument("--repository", type=Path, default=Path.cwd())
    reduce.add_argument("--campaign-root", type=Path, required=True)
    internal = commands.add_parser("_run-one")
    internal.add_argument("--repository", type=Path, required=True)
    internal.add_argument("--campaign-root", type=Path, required=True)
    internal.add_argument("--run-id", required=True)
    internal.add_argument("--reviewer-id", required=True)
    internal.add_argument("--config-path", type=Path, required=True)
    internal.add_argument("--execute-live", action="store_true")
    bootstrap = commands.add_parser("_bootstrap")
    bootstrap.add_argument("--repository", type=Path, required=True)
    bootstrap.add_argument("--campaign-root", type=Path, required=True)
    bootstrap.add_argument("--config-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        root, manifest, readiness = prepare_campaign(
            args.repository.resolve(), args.output_root.resolve(), args.reviewer_id
        )
        print(canonical_json({"campaign_id": manifest["campaign_id"], "campaign_root": root, "readiness": readiness, "provider_calls": 0}))
        return 0
    if args.command == "run":
        completed = run_campaign(
            args.repository.resolve(), args.campaign_root.resolve(), args.reviewer_id,
            execute_live=args.execute_live,
        )
        print(canonical_json({"completed_agent_runs": completed}))
        return 0
    if args.command == "reduce":
        print(canonical_json(reduce_campaign(args.campaign_root.resolve(), args.repository.resolve())))
        return 0
    if args.command == "_bootstrap":
        from pico.config.loader import set_config_path

        set_config_path(args.config_path.resolve())
        record = _bootstrap_record(args.campaign_root.resolve(), args.config_path.resolve())
        _store(args.campaign_root.resolve()).write_summary(args.campaign_root.resolve() / "config-bootstrap.json", record)
        return 0 if record["passed"] else 2
    if args.command == "_run-one":
        from pico.config.loader import set_config_path

        if not args.execute_live:
            raise PermissionError("JEV.6B child execution requires explicit --execute-live")
        set_config_path(args.config_path.resolve())
        root = args.campaign_root.resolve()
        manifest = load_manifest(root)
        planned = next(item for item in manifest["planned_runs"] if item["run_id"] == args.run_id)
        record = execute_one(
            args.repository.resolve(), root, manifest, planned, args.reviewer_id,
            config_path=args.config_path.resolve(),
        )
        return 0 if record["run_validity"] != "infra_invalid" else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
