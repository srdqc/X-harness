"""Final confirmatory runner profile for the frozen JEV.6 held-out v3 suite."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pydantic import PrivateAttr

from benchmarks.picobench.canonical import canonical_digest
from pico.sandbox.config import SandboxConfig

from . import jev6_benchmark as runner
from .jev6_sandbox import assess_benchmark_sandbox, run_benchmark_sandbox_smoke
from .jev6_suite import UTILITY_MAX_OUTPUT_TOKENS, UTILITY_TIMEOUT_SECONDS
from .jev6_v3_freeze import (
    audit_mechanical_solvability,
    evaluate_selector_identity_audit,
    load_exposure_snapshot,
    solvability_payload,
)
from .jev6_v3_suite import (
    AGENT_BUDGET,
    BASE_COMMIT,
    BENEFIT_CRITERIA,
    MODEL_ID,
    PLANNED_LIVE_RUNS,
    PROVIDER_ID,
    REAL_SANDBOX_EVIDENCE_DIGEST,
    RUN_ORDER,
    RUN_ORDER_DIGEST,
    SANDBOX_REQUIREMENT_DIGEST,
    SUITE_NAME,
    SUITE_VERSION,
    TASKS,
    Arm,
    suite_payload,
)
from .jev6_v3_verifiers import verify_task
from .schema import LiveTask

MODULE_NAME = "benchmarks.picobench.packs.knowledge_evolution_live.jev6_v3_benchmark"
FROZEN_MANIFEST = Path(__file__).with_name("jev6_v3_frozen_manifest.json")
FROZEN_SOURCE_FILES = (
    "jev6_v3_suite.py",
    "jev6_v3_fixtures.py",
    "jev6_v3_verifiers.py",
    "jev6_v3_freeze.py",
    "jev6_v3_frozen_manifest.json",
    "jev6_v3_exposure_ledger.json",
    "jev6_v3_exposure_snapshot.json",
    "jev6_v3_benchmark.py",
)
INVALID_PREDECESSOR_ID = "jev6v2b-9627a434d427637a"
BASE_HISTORICAL_CAMPAIGNS = tuple(runner.HISTORICAL_CAMPAIGNS)
class _V3RuntimeSandboxConfig(SandboxConfig):
    """Frozen sandbox behavior plus a non-configurable native runtime home."""

    _runtime_home: Path = PrivateAttr(default=Path("/tmp/pico-jev6v3-boxlite"))

    @property
    def runtime_home(self) -> Path:
        return self._runtime_home


FROZEN_SANDBOX_CONFIG = _V3RuntimeSandboxConfig(
    backend="boxlite",
    allow_net=False,
    extra_volumes=[],
    image_search_registry="docker.m.daocloud.io",
)
SANDBOX_SOURCE_IDENTITY = {
    "schema": "pico.jev6v3-sandbox-source-identity.v1",
    "schema_version": 1,
    "source": "frozen_jev6_v3_infrastructure",
    "config_digest": canonical_digest(FROZEN_SANDBOX_CONFIG.model_dump(mode="json")),
}


def _pre_run_selector_audit(
    *, state_root: Path, workspace: Path, task: LiveTask, reviewer_id: str
) -> dict[str, Any]:
    from .selectivity import evaluate_candidate_identity_audit

    audit = evaluate_candidate_identity_audit(
        state_root=state_root,
        workspace=workspace,
        tasks=(task,),
        reviewer_id=reviewer_id,
    )
    return dict(audit["tasks"][task.task_id])


def _pre_run_sandbox_audit(*, config_path: Path, workspace: Path) -> dict[str, Any]:
    del config_path
    return assess_benchmark_sandbox(FROZEN_SANDBOX_CONFIG, workspace=workspace)


def _pre_run_sandbox_smoke(
    *, config_path: Path, workspace: Path, capability_evidence: dict[str, Any] | None
) -> dict[str, Any]:
    del config_path
    return asyncio.run(
        run_benchmark_sandbox_smoke(
            FROZEN_SANDBOX_CONFIG,
            workspace,
            capability_evidence=capability_evidence,
        )
    )


def verify_frozen_suite(repository: Path) -> dict[str, Any]:
    """Recompute every V3 anti-tuning digest without regenerating frozen data."""

    frozen_bytes = FROZEN_MANIFEST.read_bytes()
    frozen = json.loads(frozen_bytes)
    mechanical = solvability_payload(audit_mechanical_solvability(repository))
    temp_parent = repository / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jev6v3-freeze-", dir=temp_parent) as temporary:
        selector_raw = evaluate_selector_identity_audit(
            state_root=Path(temporary), workspace=repository
        )
    exposure = load_exposure_snapshot(repository)
    payload = suite_payload(
        selector_audit=selector_raw,
        mechanical_audit=mechanical,
        exposure_snapshot=exposure,
    )
    actual = {
        "suite": payload["semantic_digest"],
        "task_set": payload["task_set_digest"],
        "verifier_set": payload["verifier_set_digest"],
        "fixtures": payload["fixture_set_digest"],
        "mechanical_audit": payload["mechanical_audit_digest"],
        "contract_audit": payload["contract_audit_digest"],
        "selector_audit": payload["selector_audit_digest"],
        "canonical_selector_identities": payload[
            "canonical_selector_identities_digest"
        ],
        "corpus": payload["corpus_digest"],
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
    expected = frozen["anti_tuning_digests"]
    checks = {name: actual[name] == expected[name] for name in expected}
    checks.update(
        frozen_manifest_immutable=FROZEN_MANIFEST.read_bytes() == frozen_bytes,
        suite_version=frozen["suite_version"] == SUITE_VERSION,
        base_commit=frozen["base_commit"] == BASE_COMMIT,
        planned_runs=frozen["planned_live_runs"]
        == PLANNED_LIVE_RUNS
        == len(RUN_ORDER),
        run_order_constant=actual["run_order"] == RUN_ORDER_DIGEST,
        provider=frozen["provider_id"] == PROVIDER_ID,
        model=frozen["model_id"] == MODEL_ID,
        sandbox_digest=actual["sandbox_config"] == SANDBOX_REQUIREMENT_DIGEST,
        real_sandbox_evidence=(
            frozen["sandbox_requirement"]["real_smoke_evidence_digest"]
            == REAL_SANDBOX_EVIDENCE_DIGEST
        ),
        exposure_all_unexposed=exposure["all_v3_tasks_unexposed"] is True,
    )
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"JEV.6 V3 freeze mismatch: {failed}")
    selector = {
        **selector_raw,
        "task_selected_candidate_ids": {
            task_id: row["selected_candidate_ids"]
            for task_id, row in selector_raw["tasks"].items()
        },
        "task_selected_candidate_evidence": {
            task_id: row["ordered_selected_identities"]
            for task_id, row in selector_raw["tasks"].items()
        },
    }
    return {
        "passed": True,
        "checks": checks,
        "digests": actual,
        "selector_audit": selector,
        "mechanical_audit": mechanical,
        "exposure_snapshot": exposure,
        "frozen_manifest_file_digest": hashlib.sha256(frozen_bytes).hexdigest(),
    }


def _configure_runner() -> None:
    values = {
        "AGENT_BUDGET": AGENT_BUDGET,
        "BASE_COMMIT": BASE_COMMIT,
        "BENEFIT_CRITERIA": BENEFIT_CRITERIA,
        "MODEL_ID": MODEL_ID,
        "PLANNED_LIVE_RUNS": PLANNED_LIVE_RUNS,
        "PROVIDER_ID": PROVIDER_ID,
        "RUN_ORDER": RUN_ORDER,
        "RUN_ORDER_DIGEST": RUN_ORDER_DIGEST,
        "SUITE_NAME": SUITE_NAME,
        "SUITE_VERSION": SUITE_VERSION,
        "TASKS": TASKS,
        "UTILITY_MAX_OUTPUT_TOKENS": UTILITY_MAX_OUTPUT_TOKENS,
        "UTILITY_TIMEOUT_SECONDS": UTILITY_TIMEOUT_SECONDS,
        "Arm": Arm,
        "suite_payload": suite_payload,
        "verify_task": verify_task,
        "verify_frozen_suite": verify_frozen_suite,
        "FROZEN_MANIFEST": FROZEN_MANIFEST,
        "FROZEN_SOURCE_FILES": FROZEN_SOURCE_FILES,
        "HISTORICAL_CAMPAIGNS": (*BASE_HISTORICAL_CAMPAIGNS, INVALID_PREDECESSOR_ID),
        "INVALID_PREDECESSOR_ID": INVALID_PREDECESSOR_ID,
        "INVALIDITY_REASON": "int02_live_agent_exposure_in_invalid_jev6v2b",
        "CAMPAIGN_GENERATION": "JEV.6V3_CONFIRMATORY",
        "CLAIM_ELIGIBILITY_REASON": (
            "v3_frozen_before_execution;all_v3_tasks_unexposed;"
            "real_boxlite_required;canonical_candidate_identity"
        ),
        "BEHAVIORAL_DELTA": "V3_REPLACED_EXPOSED_INT02_BEFORE_EXECUTION",
        "INFRASTRUCTURE_CHANGE": "v3_profile_resume_checkpoint_and_exposure_ledger",
        "CAMPAIGN_PREFIX": "jev6v3",
        "SCHEMA": "pico.jev6v3-confirmatory-campaign.v1",
        "SCHEMA_VERSION": 1,
        "RUN_SCHEMA": "pico.jev6v3-confirmatory-run.v1",
        "REDUCER_SCHEMA": "pico.jev6v3-confirmatory-summary.v1",
        "REDUCER_VERSION": 1,
        "MODULE_NAME": MODULE_NAME,
        "PRE_RUN_SELECTOR_AUDIT": _pre_run_selector_audit,
        "PRE_RUN_SANDBOX_AUDIT": _pre_run_sandbox_audit,
        "PRE_RUN_SANDBOX_SMOKE": _pre_run_sandbox_smoke,
        "RESUME_SAFE_CAMPAIGN": True,
        "PERSIST_LIVE_EXPOSURE": True,
        "RUNTIME_SANDBOX_CONFIG": FROZEN_SANDBOX_CONFIG,
        "SANDBOX_SOURCE_IDENTITY": SANDBOX_SOURCE_IDENTITY,
    }
    for name, value in values.items():
        setattr(runner, name, value)


_configure_runner()


def main(argv: Sequence[str] | None = None) -> int:
    _configure_runner()
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
