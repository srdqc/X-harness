"""Claim-eligible JEV.6B held-out Provider Utility benchmark runner.

Preparation and reduction are offline. Live execution is one-shot, follows the
frozen JEV.6A order, and requires the explicit ``--execute-live`` flag.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import site
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from benchmarks.picobench.artifacts import ArtifactStore
from benchmarks.picobench.canonical import canonical_digest, canonical_json, to_primitive
from benchmarks.picobench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.picobench.schema import ExperimentRef
from pico.config.paths import RuntimePaths
from pico.knowledge_evolution import (
    CandidateEvidenceIdentity,
    KnowledgeLifecycleManager,
    KnowledgeRecordStore,
    KnowledgeSelectionMode,
    ordered_candidate_evidence_equal,
)
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
from .run_isolation import (
    ENVIRONMENT_FINGERPRINT_SCHEMA,
    ENVIRONMENT_FINGERPRINT_VERSION,
    AgentRunRoots,
    canonical_environment_fingerprint,
    environment_fingerprint,
    environment_fingerprints_equal,
    verify_trace_canary,
)
from .runner import _changed_paths, _git, _workspace_patch
from .validity import classify_run_validity

SCHEMA = "pico.jev6b-confirmatory-campaign.v2"
SCHEMA_VERSION = 2
RUN_SCHEMA = "pico.jev6b-confirmatory-run.v2"
REDUCER_SCHEMA = "pico.jev6b-confirmatory-summary.v2"
REDUCER_VERSION = 2
CAMPAIGN_PREFIX = "jev6b2"
INVALID_PREDECESSOR_ID = "jev6b-041ed8144ffddbd2"
INVALIDITY_REASON = "environment_fingerprint_representation_mismatch"
CAMPAIGN_GENERATION = "JEV.6B2"
CAMPAIGN_PURPOSE = "claim_eligible_confirmatory_heldout_provider_utility"
CLAIM_ELIGIBILITY_REASON = (
    "frozen_before_execution;predecessor_exposed_zero_tasks;"
    "behavioral_delta_none;infrastructure_canonicalization_only"
)
BEHAVIORAL_DELTA = "NONE"
INFRASTRUCTURE_CHANGE = "canonical_environment_fingerprint_v1"
FROZEN_MANIFEST = Path(__file__).with_name("jev6_frozen_manifest.json")
HISTORICAL_CAMPAIGNS = (
    "jev4-9168e909e0c47dfb",
    "jev4r-3f7c246afe462f5e",
    "jev4r2-868dd3997095b407",
    "jev4r3-384e7aefbe9cca17",
    "jev5a-00e65591d2467ac3",
    INVALID_PREDECESSOR_ID,
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
MODULE_NAME = "benchmarks.picobench.packs.knowledge_evolution_live.jev6_benchmark"
PRE_RUN_SELECTOR_AUDIT: Callable[..., dict[str, Any]] | None = None
PRE_RUN_SANDBOX_AUDIT: Callable[..., dict[str, Any]] | None = None
PRE_RUN_SANDBOX_SMOKE: Callable[..., dict[str, Any]] | None = None
RESUME_SAFE_CAMPAIGN = False
PERSIST_LIVE_EXPOSURE = False
LIVE_EXPOSURE_SCHEMA = "pico.jev6-live-exposure-ledger.v1"
RUNTIME_SANDBOX_CONFIG: Any | None = None
SANDBOX_SOURCE_IDENTITY: dict[str, Any] | None = None


def _store(root: Path) -> ArtifactStore:
    return ArtifactStore(ExperimentRef(root.name, root))


def _write_forensic_artifact(path: Path, value: dict[str, Any]) -> None:
    store = ArtifactStore(ExperimentRef("jev6v3r-forensic", path.parent))
    store.write_summary(path, value)


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


def verify_historical_task_blindness(
    repository: Path, *, ignored_campaign_ids: tuple[str, ...] = ()
) -> dict[str, Any]:
    task_ids = {task.task_id for task in TASKS}
    observed_records: list[dict[str, Any]] = []
    exposures: list[dict[str, str]] = []
    for path in (repository / ".p3r").glob("*/runs/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("task_id") not in task_ids:
            continue
        campaign_id = str(value.get("campaign_id", path.parents[1].name))
        if campaign_id in ignored_campaign_ids:
            continue
        agent_turn_started = bool(
            value.get("turn_id")
            or value.get("mandatory_evidence", {}).get("complete")
            or value.get("main_agent_metrics")
            or value.get("patch_artifact_ref")
        )
        provider_calls = value.get("main_agent_metrics", {}).get(
            "provider_logical_calls", {}
        ).get("value", 0)
        exposed = bool(agent_turn_started or provider_calls)
        row = {
            "campaign_id": campaign_id,
            "task_id": str(value["task_id"]),
            "run_id": str(value.get("run_id", path.stem)),
            "agent_turn_started": agent_turn_started,
            "provider_calls": int(provider_calls or 0),
            "live_task_exposed": exposed,
        }
        observed_records.append(row)
        if exposed:
            exposures.append({key: str(item) for key, item in row.items()})
    if exposures:
        raise ValueError("JEV.6 held-out task has prior live Agent exposure")
    return {
        "passed": True,
        "live_task_exposure_count": 0,
        "prior_live_agent_exposures": (),
        "observed_pre_turn_records": tuple(observed_records),
        "scanned_task_ids": tuple(sorted(task_ids)),
    }


def _config_identity(config_path: Path) -> dict[str, Any]:
    value = _sanitized_config_identity(config_path, AGENT_BUDGET)
    if value["provider_id"] != PROVIDER_ID or value["model_id"] != MODEL_ID:
        raise ValueError("configured Provider/model differs from frozen JEV.6 identity")
    return value


def _bounded_child_environment_identity(
    config_path: Path, roots: AgentRunRoots
) -> dict[str, Any]:
    """Return the allowlisted child identity used by run-start forensics."""

    from pico.config.loader import load_config

    provider_config = load_config(config_path)
    sandbox = RUNTIME_SANDBOX_CONFIG or provider_config.tools.sandbox
    try:
        boxlite_version = importlib.metadata.version("boxlite")
    except importlib.metadata.PackageNotFoundError:
        boxlite_version = None
    kvm = Path("/dev/kvm")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "cwd": str(Path.cwd().resolve()),
        "workspace_root": str(roots.worktree.resolve()),
        "pico_home": str(Path(os.environ.get("PICO_HOME", "")).resolve()),
        "mutable_state_root": str(roots.state.resolve()),
        "trace_root": str(roots.trace.resolve()),
        "temp_root": str(roots.temporary.resolve()),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "boxlite_version": boxlite_version,
        "kvm_exists": kvm.exists(),
        "kvm_read_write": kvm.exists() and os.access(kvm, os.R_OK | os.W_OK),
        "image_search_registry": sandbox.image_search_registry,
        "sandbox_backend": sandbox.backend,
        "allow_net": sandbox.allow_net,
        "extra_volumes": list(sandbox.extra_volumes),
        "sandbox_runtime_home": (
            str(sandbox.runtime_home)
            if getattr(sandbox, "runtime_home", None) is not None
            else None
        ),
        "provider_source_identity": {
            "config_path": str(config_path.resolve()),
            "config_source_digest": _config_source_digest(config_path),
        },
        "sandbox_source_identity": SANDBOX_SOURCE_IDENTITY,
    }


def _with_runtime_sandbox(config: Any) -> Any:
    """Overlay only the explicitly injected typed sandbox config in memory."""

    if RUNTIME_SANDBOX_CONFIG is None:
        return config
    runtime_config = config.model_copy(deep=True)
    runtime_config.tools.sandbox = RUNTIME_SANDBOX_CONFIG.model_copy(deep=True)
    return runtime_config


def _sanitized_process_tail(value: str | None, *, limit: int = 2000) -> str:
    """Bound child output and redact common credential-bearing forms."""

    tail = (value or "")[-limit:]
    patterns = (
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+",
        r"(?i)((?:api[_-]?key|token|password)\s*[:=]\s*)\S+",
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
    )
    for pattern in patterns:
        tail = re.sub(pattern, r"\1<redacted>" if "(" in pattern else "<redacted>", tail)
    return tail


def _rehearsal_failure_evidence(
    *,
    campaign_id: str,
    completed: subprocess.CompletedProcess[str],
    record: dict[str, Any] | None,
    python_executable: str,
    cwd: Path,
) -> dict[str, Any]:
    """Build bounded durable evidence for a failed production child."""

    record = record or {}
    capability = record.get("sandbox_capability") or {}
    smoke = record.get("sandbox_smoke") or {}
    reason = str(record.get("infra_invalid_reason") or "child_process_failure")
    if reason in {
        "sandbox_backend_none",
        "sandbox_backend_not_explicit_boxlite",
        "linux_kvm_unavailable",
        "boxlite_dependency_unavailable",
    } or reason.startswith("unsupported_platform:"):
        last_completed = "sandbox_config_resolution"
        failing_stage = "sandbox_capability_validation"
    elif reason in {"sandbox_runtime_failure", "sandbox_smoke_failure"}:
        last_completed = "boxlite_executor_construction"
        failing_stage = "boxlite_runtime_or_microvm_startup"
    elif reason == "pre_run_selector_identity_mismatch":
        last_completed = "sandbox_smoke"
        failing_stage = "selector_identity_validation"
    elif record:
        last_completed = "child_failure_artifact_write"
        failing_stage = "run_start_lifecycle"
    else:
        last_completed = "parent_process_spawn"
        failing_stage = "child_python_bootstrap"
    failure_type = record.get("bounded_host_failure_type") or smoke.get(
        "bounded_failure_type"
    )
    failure_detail = record.get("bounded_host_failure_path") or smoke.get(
        "bounded_failure_detail"
    )
    if failure_type is None and reason != "child_process_failure":
        failure_type = "InfrastructureGateError"
        failure_detail = reason
    evidence = {
        "schema": "pico.jev6-production-child-forensic.v1",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "passed": False,
        "child_exit_code": completed.returncode,
        "last_completed_startup_stage": last_completed,
        "failing_startup_stage": failing_stage,
        "exception_type": failure_type,
        "exception_message": str(failure_detail or "")[:240],
        "stderr_tail": _sanitized_process_tail(completed.stderr),
        "stdout_tail": _sanitized_process_tail(completed.stdout),
        "python_executable": python_executable,
        "cwd": str(cwd.resolve()),
        "resolved_backend": capability.get("configured_backend")
        or record.get("child_environment_identity", {}).get("sandbox_backend"),
        "boxlite_runtime_construction_started": bool(smoke.get("resolved_executor")),
        "microvm_startup_started": bool(smoke.get("resolved_executor")),
        "microvm_started": bool(smoke.get("microvm_started")),
        "workspace_mount_started": bool(smoke.get("microvm_started")),
        "rehearsal_artifact_write_started": bool(record),
        "cleanup_outcome": (
            "completed"
            if smoke.get("sandbox_cleanup_completed") is True
            else "not_started"
            if not smoke
            else "incomplete"
        ),
        "environment_identity": record.get("child_environment_identity"),
        "sandbox_capability": capability,
        "sandbox_smoke": smoke,
        "provider_calls": int(record.get("provider_calls", 0)),
        "agent_turns": int(record.get("agent_turns", 0)),
        "live_agent_exposed": bool(record.get("live_agent_exposed", False)),
    }
    evidence["integrity_digest"] = canonical_digest(evidence)
    return evidence


def _campaign_semantic(freeze: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    selector = freeze["selector_audit"]
    selected_ids = selector["task_selected_candidate_ids"]
    semantic = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "reducer_schema": REDUCER_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "campaign_purpose": CAMPAIGN_PURPOSE,
        "campaign_generation": CAMPAIGN_GENERATION,
        "invalid_predecessor_campaign_id": INVALID_PREDECESSOR_ID,
        "invalid_predecessor_reason": INVALIDITY_REASON,
        "claim_eligible": True,
        "claim_eligibility_reason": CLAIM_ELIGIBILITY_REASON,
        "behavioral_delta": BEHAVIORAL_DELTA,
        "environment_fingerprint_schema": ENVIRONMENT_FINGERPRINT_SCHEMA,
        "environment_fingerprint_version": ENVIRONMENT_FINGERPRINT_VERSION,
        "infrastructure_delta_digest": canonical_digest(
            {
                "change": INFRASTRUCTURE_CHANGE,
                "behavioral_delta": BEHAVIORAL_DELTA,
                "invalid_predecessor": INVALID_PREDECESSOR_ID,
            }
        ),
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
        "offline_rehearsal_digest": config.get("offline_rehearsal_digest"),
        "budget": asdict(AGENT_BUDGET),
        "planned_runs": tuple(asdict(item) for item in RUN_ORDER),
        "run_order_digest": RUN_ORDER_DIGEST,
        "frozen_selected_candidate_ids": selected_ids,
        "replacement_policy": "forbidden",
        "typesafe_enabled": False,
    }
    selected_evidence = selector.get("task_selected_candidate_evidence")
    if selected_evidence is not None:
        semantic["frozen_selected_candidate_evidence"] = selected_evidence
    return semantic


def prepare_campaign(
    repository: Path,
    output_root: Path,
    reviewer_id: str,
    *,
    config_path: Path | None = None,
    rehearsal_path: Path | None = None,
    rehearsal_mode: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from pico.config.loader import get_config_path

    base_sha = resolve_base_commit(repository, BASE_COMMIT)
    if base_sha != BASE_COMMIT:
        raise ValueError("JEV.6 frozen base commit does not resolve exactly")
    freeze = verify_frozen_suite(repository)
    blindness = verify_historical_task_blindness(repository)
    config_path = (config_path or get_config_path()).resolve()
    config = _config_identity(config_path)
    rehearsal = None
    if not rehearsal_mode:
        if rehearsal_path is None or not rehearsal_path.is_file():
            raise ValueError("JEV.6B2 requires a passed offline run-start rehearsal")
        rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
        rehearsal_digest = rehearsal.pop("integrity_digest", None)
        if rehearsal_digest != canonical_digest(rehearsal) or not rehearsal.get("passed"):
            raise ValueError("JEV.6B2 rehearsal artifact is invalid")
        if any(rehearsal.get(name) != 0 for name in ("provider_calls", "agent_turns", "network_calls")):
            raise ValueError("JEV.6B2 rehearsal performed forbidden live activity")
        config["offline_rehearsal_digest"] = rehearsal_digest
    semantic = _campaign_semantic(freeze, config)
    semantic_digest = canonical_digest(semantic)
    campaign_id = f"{CAMPAIGN_PREFIX}-{semantic_digest[:16]}"
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
        "offline_rehearsal": rehearsal,
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
        "canonical_environment_fingerprint": (
            bootstrap.get("canonical_environment_fingerprint", {}).get("schema")
            == ENVIRONMENT_FINGERPRINT_SCHEMA
        ),
        "secret_leak_free": bootstrap.get("secret_leak_free") is True,
        "heldout_live_exposure_zero": blindness["live_task_exposure_count"] == 0,
        "offline_run_start_rehearsal": rehearsal_mode or bool(rehearsal and rehearsal["passed"]),
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
                "task_selected_candidate_ids": value["frozen_selected_candidate_ids"],
                **(
                    {
                        "task_selected_candidate_evidence": value[
                            "frozen_selected_candidate_evidence"
                        ]
                    }
                    if "frozen_selected_candidate_evidence" in value
                    else {}
                ),
            },
        },
        {
            "config_identity_digest": value["config_identity_digest"],
            "config_source_digest": value["config_source_digest"],
            "provider_model_config_digest": value["provider_model_config_digest"],
            "offline_rehearsal_digest": value.get("offline_rehearsal_digest"),
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
        "canonical_environment_fingerprint": canonical_environment_fingerprint(fingerprint),
        "failure_type": failure_type,
        "provider_calls": 0,
        "agent_turns": 0,
        "network_calls": 0,
    }
    record["integrity_digest"] = canonical_digest(record)
    return record


def run_bootstrap(repository: Path, root: Path, config_path: Path) -> dict[str, Any]:
    from pico.config.loader import load_config

    roots = AgentRunRoots.create(root, "config-bootstrap", 0)
    roots.prepare_non_worktree_roots()
    artifact = root / "config-bootstrap.json"
    if artifact.exists():
        raise FileExistsError("JEV.6B bootstrap is immutable")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE_NAME,
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
    secret = load_config(config_path).get_api_key(load_manifest(root)["actual_model_id"])
    persisted = canonical_json(
        {"manifest": load_manifest(root), "bootstrap": record, "command": "_bootstrap"}
    )
    for path in roots.trace.rglob("*"):
        if path.is_file():
            persisted += path.read_text(encoding="utf-8", errors="replace")
    record["secret_leak_free"] = not secret or secret not in persisted
    record["passed"] = bool(record["passed"] and record["secret_leak_free"])
    record.pop("integrity_digest", None)
    record["integrity_digest"] = canonical_digest(record)
    _store(root).write_summary(artifact, record)
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


def rehearse_run_start(
    repository: Path,
    reviewer_id: str,
    artifact_path: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Exercise the production child path and stop immediately before Agent Turn."""
    from pico.config.loader import get_config_path

    config_path = (config_path or get_config_path()).resolve()
    predecessor_before = _tree_digest(repository / ".p3r" / INVALID_PREDECESSOR_ID)
    temp_parent = repository / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="j6r-", dir=temp_parent) as temporary:
        root, manifest, readiness = prepare_campaign(
            repository,
            Path(temporary),
            reviewer_id,
            config_path=config_path,
            rehearsal_mode=True,
        )
        reloaded = load_manifest(root)
        planned = reloaded["planned_runs"][0]
        roots = AgentRunRoots.create(root, planned["run_id"], planned["order"])
        roots.prepare_non_worktree_roots()
        parent_before = environment_fingerprint()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE_NAME,
                "_rehearse-one",
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
            ],
            cwd=repository,
            env=roots.child_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        parent_after = environment_fingerprint()
        if completed.returncode != 0 or not roots.artifact.is_file():
            failure_record = None
            if roots.artifact.is_file():
                try:
                    failure_record = _store(root).read_json(roots.artifact)
                    failure_digest = failure_record.pop("integrity_digest", None)
                    if failure_digest != canonical_digest(failure_record):
                        failure_record = None
                    else:
                        failure_record["integrity_digest"] = failure_digest
                except Exception:  # noqa: BLE001 - preserve parent forensic path
                    failure_record = None
            failure = _rehearsal_failure_evidence(
                campaign_id=manifest["campaign_id"],
                completed=completed,
                record=failure_record,
                python_executable=sys.executable,
                cwd=repository,
            )
            failure.pop("integrity_digest")
            failure["parent_environment_unchanged"] = environment_fingerprints_equal(
                parent_before, parent_after
            )
            failure["failure_artifact_durable"] = True
            failure["integrity_digest"] = canonical_digest(failure)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if artifact_path.exists():
                raise FileExistsError("JEV.6V3 forensic artifact already exists")
            _write_forensic_artifact(artifact_path, failure)
            raise RuntimeError(
                "JEV.6 production child rehearsal failed at "
                f"{failure['failing_startup_stage']}; forensic={artifact_path}"
            )
        record = _store(root).read_json(roots.artifact)
        digest = record.pop("integrity_digest", None)
        if digest != canonical_digest(record):
            raise ValueError("JEV.6B2 rehearsal child integrity mismatch")
        record["integrity_digest"] = digest
        bootstrap = _store(root).read_json(root / "config-bootstrap.json")
        bootstrap_digest = bootstrap.pop("integrity_digest", None)
        if bootstrap_digest != canonical_digest(bootstrap):
            raise ValueError("JEV.6B2 rehearsal bootstrap integrity mismatch")
        canonical_match = environment_fingerprints_equal(
            record["environment_fingerprint"],
            bootstrap["canonical_environment_fingerprint"],
        )
        checks = {
            "freeze_verification": readiness["checks"]["freeze_verified"],
            "heldout_exposure_zero": readiness["checks"]["heldout_live_exposure_zero"],
            "manifest_json_round_trip": manifest["manifest_digest"] == reloaded["manifest_digest"],
            "bootstrap": bootstrap["passed"],
            "parent_child_config_identity": record["pre_run_checks"]["config_identity"],
            "canonical_environment_identity": canonical_match,
            "environment_drift_false": record["pre_run_checks"]["environment_fingerprint"],
            "trace_canary": record["trace_canary"]["passed"],
            "run_local_roots": record["pre_run_checks"]["run_local_evidence_root"],
            "python_no_user_site": record["pre_run_checks"]["python_no_user_site"],
            "mandatory_pre_run_evidence": record["mandatory_pre_run_evidence"]["complete"],
            "frozen_selector_identity": record.get(
                "pre_run_selector_identity", {}
            ).get("passed")
            is True,
            "live_ready_handoff": record["live_ready_handoff"],
            "first_run_identity": planned == tuple(asdict(item) for item in RUN_ORDER)[0],
            "secret_leak_free": bootstrap["secret_leak_free"],
            "parent_environment_unchanged": environment_fingerprints_equal(
                parent_before, parent_after
            ),
            "predecessor_immutable": predecessor_before
            == _tree_digest(repository / ".p3r" / INVALID_PREDECESSOR_ID),
        }
        result = {
            "schema": "pico.jev6b2-run-start-rehearsal.v1",
            "campaign_generation": CAMPAIGN_GENERATION,
            "passed": all(checks.values()),
            "checks": checks,
            "first_planned_run": planned,
            "canonical_parent_identity": bootstrap["canonical_environment_fingerprint"],
            "canonical_child_identity": canonical_environment_fingerprint(
                record["environment_fingerprint"]
            ),
            "provider_calls": record["provider_calls"],
            "agent_turns": record["agent_turns"],
            "network_calls": record["network_calls"],
            "live_task_exposure_count": readiness["historical_task_blindness"][
                "live_task_exposure_count"
            ],
            "claim_eligible": all(checks.values()),
            "invalid_predecessor_campaign_id": INVALID_PREDECESSOR_ID,
            "behavioral_delta": BEHAVIORAL_DELTA,
            "child_exit_code": completed.returncode,
            "last_completed_startup_stage": "rehearsal_artifact_write",
            "failing_startup_stage": None,
            "child_environment_identity": record.get("child_environment_identity"),
            "sandbox_capability": record.get("sandbox_capability"),
            "sandbox_smoke": record.get("sandbox_smoke"),
            "stderr_tail": _sanitized_process_tail(completed.stderr),
            "stdout_tail": _sanitized_process_tail(completed.stdout),
            "cleanup_outcome": (
                "completed"
                if record.get("sandbox_smoke", {}).get("sandbox_cleanup_completed")
                else "incomplete"
            ),
        }
    if not result["passed"]:
        raise RuntimeError("JEV.6B2 offline full run-start rehearsal failed")
    result["integrity_digest"] = canonical_digest(result)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError("JEV.6B2 rehearsal artifact already exists")
    artifact_path.write_text(canonical_json(result), encoding="utf-8")
    return result


def _infra_record(manifest: dict[str, Any], planned: dict[str, Any], reason: str) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
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


def _live_exposure_path(root: Path) -> Path:
    return root / "live-exposure-ledger.json"


def _load_live_exposure(root: Path) -> dict[str, Any] | None:
    path = _live_exposure_path(root)
    if not path.exists():
        return None
    value = _store(root).read_json(path)
    digest = value.pop("integrity_digest", None)
    if digest != canonical_digest(value):
        raise ValueError("JEV.6 live exposure ledger integrity mismatch")
    value["integrity_digest"] = digest
    if value.get("schema") != LIVE_EXPOSURE_SCHEMA:
        raise ValueError("unsupported JEV.6 live exposure ledger schema")
    return value


def _mark_live_exposure(
    root: Path, manifest: dict[str, Any], planned: dict[str, Any]
) -> None:
    """Persist exposure immediately before crossing the Agent Turn boundary."""

    current = _load_live_exposure(root)
    if current is None:
        value: dict[str, Any] = {
            "schema": LIVE_EXPOSURE_SCHEMA,
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "suite_version": manifest["suite_version"],
            "runs": {},
            "tasks": {},
        }
    else:
        value = {key: item for key, item in current.items() if key != "integrity_digest"}
    if value["campaign_id"] != manifest["campaign_id"]:
        raise ValueError("JEV.6 live exposure campaign mismatch")
    existing = value["runs"].get(planned["run_id"])
    identity = {
        "task_id": planned["task_id"],
        "arm": planned["arm"],
        "repetition": planned["repetition"],
        "order": planned["order"],
    }
    if existing is not None:
        comparable = {key: existing[key] for key in identity}
        if comparable != identity:
            raise ValueError("JEV.6 live exposure run identity mismatch")
        return
    value["runs"][planned["run_id"]] = {
        **identity,
        "exposed_at": datetime.now(timezone.utc).isoformat(),
    }
    value["tasks"][planned["task_id"]] = {"live_agent_exposed": True}
    value["integrity_digest"] = canonical_digest(value)
    _store(root).write_summary(_live_exposure_path(root), value)


def _resume_completed_prefix(
    root: Path, manifest: dict[str, Any]
) -> tuple[str, ...]:
    planned = tuple(manifest["planned_runs"])
    by_id = {item["run_id"]: item for item in planned}
    records = load_runs(root)
    for record in records:
        expected = by_id.get(record["run_id"])
        if expected is None:
            raise ValueError("JEV.6 resume found an unknown run artifact")
        if any(record.get(key) != expected[key] for key in ("task_id", "arm", "repetition", "order")):
            raise ValueError("JEV.6 resume run identity mismatch")
        if record.get("campaign_id") != manifest["campaign_id"]:
            raise ValueError("JEV.6 resume campaign identity mismatch")
        if record.get("run_validity") == "infra_invalid":
            raise RuntimeError("JEV.6 campaign already stopped after INFRA_INVALID")
    completed = tuple(record["run_id"] for record in records)
    expected_prefix = tuple(item["run_id"] for item in planned[: len(completed)])
    if completed != expected_prefix:
        raise ValueError("JEV.6 resume artifacts are not a frozen run-order prefix")
    if PERSIST_LIVE_EXPOSURE:
        exposure = _load_live_exposure(root)
        exposed_ids = (
            tuple(
                run_id
                for run_id, _row in sorted(
                    exposure["runs"].items(), key=lambda item: item[1]["order"]
                )
            )
            if exposure is not None
            else ()
        )
        if exposed_ids != completed:
            raise RuntimeError(
                "JEV.6 exposure/run checkpoint mismatch; rerun is forbidden"
            )
    return completed


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


def _selected_candidate_evidence(
    state: Path, candidate_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    store = KnowledgeRecordStore(state)
    lifecycle = KnowledgeLifecycleManager(store)
    values: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = store.read_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("selected candidate is missing from run-local state")
        values.append(
            CandidateEvidenceIdentity.from_candidate(
                candidate,
                lifecycle_state=lifecycle.rebuild(candidate_id).state,
            ).to_dict()
        )
    return tuple(values)


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
    config = _with_runtime_sandbox(config)
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
    if PRE_RUN_SELECTOR_AUDIT is None:
        prepare_approved_corpus(
            state_root=state, workspace=worktree, reviewer_id=reviewer_id
        )
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
    selected_candidate_evidence = _selected_candidate_evidence(
        state, tuple(refs["relevance_selected_candidate_ids"])
    )
    record = {
        "schema": RUN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
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
        "relevance_selected_candidate_evidence": selected_candidate_evidence,
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
        "live_agent_exposed": True,
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
    stop_before_agent_turn: bool = False,
) -> dict[str, Any]:
    run_id = planned["run_id"]
    roots = AgentRunRoots.create(root, run_id, planned["order"])
    roots.prepare_non_worktree_roots()
    try:
        config = _config_identity(config_path)
    except Exception as exc:  # noqa: BLE001 - bounded identity gate
        config = {"failure_type": type(exc).__name__}
    try:
        child_environment_identity = _bounded_child_environment_identity(
            config_path, roots
        )
    except Exception as exc:  # noqa: BLE001 - bounded forensic evidence
        child_environment_identity = {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "cwd": str(Path.cwd().resolve()),
            "identity_failure_type": type(exc).__name__,
        }
    bootstrap = _store(root).read_json(root / "config-bootstrap.json")
    bootstrap_digest = bootstrap.pop("integrity_digest", None)
    if bootstrap_digest != canonical_digest(bootstrap):
        raise ValueError("JEV.6B bootstrap integrity mismatch")
    fingerprint = environment_fingerprint()
    sandbox_capability = (
        PRE_RUN_SANDBOX_AUDIT(config_path=config_path, workspace=roots.worktree)
        if PRE_RUN_SANDBOX_AUDIT is not None
        else None
    )
    env_match = environment_fingerprints_equal(
        fingerprint, bootstrap["canonical_environment_fingerprint"]
    )
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
        "sandbox_capability": (
            sandbox_capability is None or sandbox_capability.get("passed") is True
        ),
    }
    mandatory_pre_run = {
        "complete": all(prechecks.values()),
        "config_identity_digest": config.get("config_identity_digest"),
        "environment_fingerprint_digest": canonical_environment_fingerprint(fingerprint)[
            "fingerprint_digest"
        ],
        "trace_canary_event_count": canary["event_count"],
        "run_root_identity_digest": canonical_digest(roots.public_identity()),
    }
    if sandbox_capability is not None and not sandbox_capability.get("passed"):
        record = _infra_record(
            manifest,
            planned,
            str(sandbox_capability.get("infra_invalid_reason") or "sandbox_capability_failure"),
        )
        record.update(
            pre_run_checks=prechecks,
            sandbox_capability=sandbox_capability,
            trace_canary=canary,
            isolation_roots=roots.public_identity(),
            child_process_id=os.getpid(),
            environment_fingerprint=fingerprint,
            mandatory_pre_run_evidence=mandatory_pre_run,
            child_environment_identity=child_environment_identity,
            provider_calls=0,
            agent_turns=0,
            network_calls=0,
            live_agent_exposed=False,
        )
        record["integrity_digest"] = canonical_digest(
            {key: value for key, value in record.items() if key != "integrity_digest"}
        )
        _store(root).write_summary(roots.artifact, record)
        return record
    if not all(prechecks.values()):
        record = _infra_record(manifest, planned, "pre_run_canary_failure")
        record.update(
            pre_run_checks=prechecks,
            trace_canary=canary,
            isolation_roots=roots.public_identity(),
            child_process_id=os.getpid(),
            environment_fingerprint=fingerprint,
            mandatory_pre_run_evidence=mandatory_pre_run,
            child_environment_identity=child_environment_identity,
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
            sandbox_smoke: dict[str, Any] | None = None
            if PRE_RUN_SANDBOX_SMOKE is not None:
                sandbox_smoke = PRE_RUN_SANDBOX_SMOKE(
                    config_path=config_path,
                    workspace=roots.worktree,
                    capability_evidence=sandbox_capability,
                )
                if not sandbox_smoke.get("passed"):
                    record = _infra_record(
                        manifest,
                        planned,
                        str(
                            sandbox_smoke.get("infra_invalid_reason")
                            or "sandbox_smoke_failure"
                        ),
                    )
                    record["sandbox_capability"] = sandbox_capability
                    record["sandbox_smoke"] = sandbox_smoke
                    record["provider_calls"] = 0
                    record["agent_turns"] = 0
                    record["network_calls"] = 0
                    record["live_agent_exposed"] = False
                    record.update(
                        pre_run_checks=prechecks,
                        trace_canary=canary,
                        isolation_roots=roots.public_identity(),
                        child_process_id=os.getpid(),
                        environment_fingerprint=fingerprint,
                        mandatory_pre_run_evidence=mandatory_pre_run,
                        child_environment_identity=child_environment_identity,
                    )
                    record["integrity_digest"] = canonical_digest(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "integrity_digest"
                        }
                    )
                    _store(root).write_summary(roots.artifact, record)
                    return record
            selector_identity: dict[str, Any] | None = None
            selector_identity_passed = True
            if PRE_RUN_SELECTOR_AUDIT is not None:
                selector_identity = PRE_RUN_SELECTOR_AUDIT(
                    state_root=roots.state,
                    workspace=roots.worktree,
                    task=_task_by_id(planned["task_id"]),
                    reviewer_id=reviewer_id,
                )
                selector_probe = {
                    "task_id": planned["task_id"],
                    "relevance_selected_candidate_ids": selector_identity[
                        "selected_candidate_ids"
                    ],
                    "relevance_selected_candidate_evidence": selector_identity[
                        "ordered_selected_identities"
                    ],
                }
                selector_identity_passed = _selection_identity_matches(
                    manifest, selector_probe
                )
            if not selector_identity_passed:
                record = _infra_record(
                    manifest, planned, "pre_run_selector_identity_mismatch"
                )
            elif stop_before_agent_turn:
                record = {
                    "schema": "pico.jev6b2-run-start-rehearsal.v1",
                    "schema_version": 1,
                    "campaign_id": manifest["campaign_id"],
                    **planned,
                    "run_validity": "valid",
                    "infra_invalid_reason": None,
                    "run_outcome": "LIVE_READY_HANDOFF",
                    "verified_success": False,
                    "verifier_findings": (),
                    "main_agent_metrics": {},
                    "utility_metrics": {},
                    "combined_provider_cost": {},
                    "changed_paths": (),
                    "typesafe_enabled": False,
                    "provider_id": config["provider_id"],
                    "model_id": config["model_id"],
                    "live_ready_handoff": True,
                    "agent_turns": 0,
                    "provider_calls": 0,
                    "network_calls": 0,
                    "mandatory_pre_run_evidence": mandatory_pre_run,
                }
            else:
                if PERSIST_LIVE_EXPOSURE:
                    _mark_live_exposure(root, manifest, planned)
                record = asyncio.run(
                    _execute_turn(root, manifest, planned, roots.worktree, roots.state, reviewer_id)
                )
            record["pre_run_selector_identity"] = {
                "passed": selector_identity_passed,
                "selected_candidate_ids": (
                    selector_identity["selected_candidate_ids"]
                    if selector_identity is not None
                    else ()
                ),
                "ordered_selected_identities": (
                    selector_identity["ordered_selected_identities"]
                    if selector_identity is not None
                    else ()
                ),
            }
            record["sandbox_capability"] = sandbox_capability
            record["sandbox_smoke"] = sandbox_smoke
        except Exception as exc:  # noqa: BLE001 - immutable host boundary
            record = _infra_record(manifest, planned, "benchmark_host_failure")
            record["bounded_host_failure_type"] = type(exc).__name__
            missing_path = getattr(exc, "filename", None)
            record["bounded_host_failure_path"] = (
                str(Path(missing_path).resolve()) if missing_path else None
            )
        finally:
            if roots.worktree.exists():
                _git(repository, "worktree", "remove", "--force", str(roots.worktree))
    record.update(
        pre_run_checks=prechecks,
        trace_canary=canary,
        isolation_roots=roots.public_identity(),
        child_process_id=os.getpid(),
        environment_fingerprint=fingerprint,
        mandatory_pre_run_evidence=mandatory_pre_run,
        child_environment_identity=child_environment_identity,
    )
    record["integrity_digest"] = canonical_digest(
        {key: value for key, value in record.items() if key != "integrity_digest"}
    )
    _store(root).write_summary(roots.artifact, record)
    return record


def _selection_identity_matches(
    manifest: dict[str, Any], record: dict[str, Any]
) -> bool:
    """Use canonical v1 evidence when present; preserve historical v1 semantics."""

    frozen = manifest.get("frozen_selected_candidate_evidence")
    if frozen is None:
        expected = tuple(
            manifest["frozen_selected_candidate_ids"][record["task_id"]]
        )
        return tuple(record.get("relevance_selected_candidate_ids", ())) == expected
    try:
        expected_identity = tuple(
            CandidateEvidenceIdentity.from_dict(item)
            for item in frozen[record["task_id"]]
        )
        actual_identity = tuple(
            CandidateEvidenceIdentity.from_dict(item)
            for item in record.get("relevance_selected_candidate_evidence", ())
        )
        return ordered_candidate_evidence_equal(expected_identity, actual_identity)
    except (KeyError, TypeError, ValueError):
        return False


def _selection_failure(manifest: dict[str, Any], record: dict[str, Any]) -> str | None:
    expected = tuple(manifest["frozen_selected_candidate_ids"][record["task_id"]])
    if not _selection_identity_matches(manifest, record):
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
    if RESUME_SAFE_CAMPAIGN:
        completed_ids = _resume_completed_prefix(root, manifest)
    elif (root / "runs").exists() and tuple((root / "runs").glob("*.json")):
        raise FileExistsError("JEV.6B is one-shot; existing runs forbid resume/replacement")
    else:
        completed_ids = ()
    freeze = verify_frozen_suite(repository)
    if freeze["digests"] != manifest["freeze_digests"]:
        raise ValueError("JEV.6B freeze changed after preparation")
    verify_historical_task_blindness(
        repository,
        ignored_campaign_ids=(manifest["campaign_id"],) if RESUME_SAFE_CAMPAIGN else (),
    )
    config_path = get_config_path().resolve()
    if _config_identity(config_path)["config_identity_digest"] != manifest["config_identity_digest"]:
        raise ValueError("JEV.6B Provider config changed after preparation")
    readiness = _load_readiness(root)
    if not readiness["live_ready"] or not all(readiness["checks"].values()):
        raise RuntimeError("JEV.6B campaign is not live-ready")
    completed = list(completed_ids)
    for planned in manifest["planned_runs"][len(completed_ids) :]:
        roots = AgentRunRoots.create(root, planned["run_id"], planned["order"])
        roots.prepare_non_worktree_roots()
        parent_before = environment_fingerprint()
        config_before = _config_source_digest(config_path)
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE_NAME,
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
        record["environment_drift_detected"] = not environment_fingerprints_equal(
            parent_before, parent_after
        )
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
        "selector_identity": all(
            _selection_identity_matches(manifest, item) for item in runs
        ),
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
    prepare.add_argument("--rehearsal-artifact", type=Path, required=True)
    rehearse = commands.add_parser("rehearse")
    rehearse.add_argument("--repository", type=Path, default=Path.cwd())
    rehearse.add_argument("--reviewer-id", required=True)
    rehearse.add_argument(
        "--artifact", type=Path, default=Path(".p3r/jev6b2-prelive-rehearsal.json")
    )
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
    rehearse_one = commands.add_parser("_rehearse-one")
    rehearse_one.add_argument("--repository", type=Path, required=True)
    rehearse_one.add_argument("--campaign-root", type=Path, required=True)
    rehearse_one.add_argument("--run-id", required=True)
    rehearse_one.add_argument("--reviewer-id", required=True)
    rehearse_one.add_argument("--config-path", type=Path, required=True)
    bootstrap = commands.add_parser("_bootstrap")
    bootstrap.add_argument("--repository", type=Path, required=True)
    bootstrap.add_argument("--campaign-root", type=Path, required=True)
    bootstrap.add_argument("--config-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        root, manifest, readiness = prepare_campaign(
            args.repository.resolve(), args.output_root.resolve(), args.reviewer_id,
            rehearsal_path=args.rehearsal_artifact.resolve(),
        )
        print(canonical_json({"campaign_id": manifest["campaign_id"], "campaign_root": root, "readiness": readiness, "provider_calls": 0}))
        return 0
    if args.command == "rehearse":
        result = rehearse_run_start(
            args.repository.resolve(), args.reviewer_id, args.artifact.resolve()
        )
        print(canonical_json(result))
        return 0
    if args.command == "run":
        root = args.campaign_root.resolve()
        with _store(root).exclusive_run_lock():
            completed = run_campaign(
                args.repository.resolve(), root, args.reviewer_id,
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
    if args.command == "_rehearse-one":
        from pico.config.loader import set_config_path

        set_config_path(args.config_path.resolve())
        root = args.campaign_root.resolve()
        manifest = load_manifest(root)
        planned = next(item for item in manifest["planned_runs"] if item["run_id"] == args.run_id)
        record = execute_one(
            args.repository.resolve(),
            root,
            manifest,
            planned,
            args.reviewer_id,
            config_path=args.config_path.resolve(),
            stop_before_agent_turn=True,
        )
        return 0 if record.get("live_ready_handoff") else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
