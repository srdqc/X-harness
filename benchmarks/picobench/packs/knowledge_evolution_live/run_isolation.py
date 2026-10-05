"""Fresh-process and run-local environment boundary for live Agent benchmarks."""

from __future__ import annotations

import importlib.metadata
import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmarks.picobench.canonical import canonical_digest
from pico.tracing import evidence

RUNNER_VERSION = 4
CONFIG_BOOTSTRAP_VERSION = 1
ENVIRONMENT_FINGERPRINT_SCHEMA = "pico.environment-fingerprint.v1"
ENVIRONMENT_FINGERPRINT_VERSION = 1
_FINGERPRINT_FIELDS = (
    "python_executable_digest",
    "python_version",
    "sys_path_digest",
    "distribution_digest",
    "distribution_count",
    "user_site_enabled",
)


@dataclass(frozen=True)
class AgentRunRoots:
    campaign_root: Path
    run_id: str
    worktree: Path
    state: Path
    trace: Path
    session: Path
    temporary: Path
    artifact: Path
    python_user_base: Path
    pip_target: Path
    pip_cache: Path

    @classmethod
    def create(cls, campaign_root: Path, run_id: str, ordinal: int) -> "AgentRunRoots":
        short = f"r{ordinal:02d}"
        return cls(
            campaign_root=campaign_root,
            run_id=run_id,
            worktree=campaign_root / "w" / short,
            state=campaign_root / "s" / short,
            trace=campaign_root / "s" / short,
            session=campaign_root / "sessions" / short,
            temporary=campaign_root / "tmp" / short,
            artifact=campaign_root / "pending" / f"{run_id}.json",
            python_user_base=campaign_root / "python" / short / "userbase",
            pip_target=campaign_root / "python" / short / "target",
            pip_cache=campaign_root / "python" / short / "cache",
        )

    def prepare_non_worktree_roots(self) -> None:
        for path in (
            self.state,
            self.session,
            self.temporary,
            self.artifact.parent,
            self.python_user_base,
            self.pip_target,
            self.pip_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def child_environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        environment = dict(base or os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment.update(
            {
                "PICO_TRACING_DIR": str(self.trace),
                "PICO_HOME": str(self.state / "pico-home"),
                "PICO_BENCH_SESSION_ROOT": str(self.session),
                "PICO_BENCH_SESSION_ID": f"jev4r:{self.run_id}",
                "PICO_BENCH_ARTIFACT_PATH": str(self.artifact),
                "TEMP": str(self.temporary),
                "TMP": str(self.temporary),
                "TMPDIR": str(self.temporary),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUSERBASE": str(self.python_user_base),
                "PIP_TARGET": str(self.pip_target),
                "PIP_CACHE_DIR": str(self.pip_cache),
                "PYTHONPYCACHEPREFIX": str(self.temporary / "pycache"),
            }
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(self.pip_target), existing_pythonpath) if value
        )
        return environment

    def public_identity(self) -> dict[str, object]:
        values = {
            "worktree": self.worktree,
            "state": self.state,
            "trace": self.trace,
            "session": self.session,
            "temporary": self.temporary,
            "artifact": self.artifact,
            "python_user_base": self.python_user_base,
            "pip_target": self.pip_target,
            "pip_cache": self.pip_cache,
        }
        return {
            name: {
                "ref": path.relative_to(self.campaign_root).as_posix(),
                "resolved_digest": canonical_digest(str(path.resolve())),
            }
            for name, path in values.items()
        }


def canonical_environment_fingerprint(value: dict[str, object]) -> dict[str, object]:
    """Return the single JSON-safe identity used for persistence and equality."""

    missing = tuple(name for name in _FINGERPRINT_FIELDS if name not in value)
    if missing:
        raise ValueError(f"environment fingerprint fields missing: {','.join(missing)}")
    version = value["python_version"]
    if (
        not isinstance(version, (tuple, list))
        or len(version) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in version)
    ):
        raise ValueError("python_version must contain three non-negative integers")
    digests = {
        name: value[name]
        for name in (
            "python_executable_digest",
            "sys_path_digest",
            "distribution_digest",
        )
    }
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests.values()
    ):
        raise ValueError("environment fingerprint digests must be lowercase SHA-256 values")
    distribution_count = value["distribution_count"]
    if (
        isinstance(distribution_count, bool)
        or not isinstance(distribution_count, int)
        or distribution_count < 0
    ):
        raise ValueError("distribution_count must be a non-negative integer")
    if not isinstance(value["user_site_enabled"], bool):
        raise ValueError("user_site_enabled must be boolean")
    payload = {
        "schema": ENVIRONMENT_FINGERPRINT_SCHEMA,
        "schema_version": ENVIRONMENT_FINGERPRINT_VERSION,
        **digests,
        "python_version": list(version),
        "distribution_count": distribution_count,
        "user_site_enabled": value["user_site_enabled"],
    }
    return {**payload, "fingerprint_digest": canonical_digest(payload)}


def environment_fingerprints_equal(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    return (
        canonical_environment_fingerprint(left)["fingerprint_digest"]
        == canonical_environment_fingerprint(right)["fingerprint_digest"]
    )


def _canonical_sys_path_digest() -> str:
    pip_target = os.environ.get("PIP_TARGET")
    target = Path(pip_target).resolve() if pip_target else None
    values = []
    for raw in sys.path:
        if not raw:
            continue
        resolved = Path(raw).resolve()
        values.append(
            "<RUN_LOCAL_PIP_TARGET>"
            if target is not None and resolved == target
            else str(resolved)
        )
    return canonical_digest(tuple(values))


def environment_fingerprint() -> dict[str, object]:
    distributions = sorted(
        (dist.metadata.get("Name", "").casefold(), dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    payload = {
        "python_executable_digest": canonical_digest(str(Path(sys.executable).resolve())),
        "python_version": list(sys.version_info[:3]),
        "sys_path_digest": _canonical_sys_path_digest(),
        "distribution_digest": canonical_digest(distributions),
        "distribution_count": len(distributions),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
    }
    return canonical_environment_fingerprint(payload)


def verify_trace_canary(
    trace_root: Path,
    *,
    canary_id: str,
    other_trace_roots: Iterable[Path] = (),
    emit_terminal: bool = True,
) -> dict[str, object]:
    recorder = evidence.TurnEvidenceRecorder(
        turn_id=canary_id,
        conversation_id=f"canary:{canary_id}",
        trace_id=f"trace:{canary_id}",
        root_span_id=f"span:{canary_id}",
    )
    recorder.emit(evidence.TURN_STARTED, metadata={"benchmark_canary": True})
    if emit_terminal:
        recorder.emit(
            evidence.TURN_TERMINAL,
            metadata={"benchmark_canary": True, "outcome": "completed"},
        )
    readback = evidence.read_turn_evidence(trace_root, canary_id)
    foreign = tuple(
        canonical_digest(str(path.resolve()))
        for path in other_trace_roots
        if evidence.read_turn_evidence(path, canary_id).events
    )
    passed = (
        readback.completeness is evidence.EvidenceCompleteness.COMPLETE
        and recorder.write_failures == 0
        and not foreign
    )
    return {
        "passed": passed,
        "canary_id": canary_id,
        "trace_root_digest": canonical_digest(str(trace_root.resolve())),
        "evidence_completeness": readback.completeness.value,
        "findings": readback.findings,
        "event_count": len(readback.events),
        "foreign_root_digests": foreign,
    }


__all__ = [
    "AgentRunRoots",
    "CONFIG_BOOTSTRAP_VERSION",
    "ENVIRONMENT_FINGERPRINT_SCHEMA",
    "ENVIRONMENT_FINGERPRINT_VERSION",
    "RUNNER_VERSION",
    "canonical_environment_fingerprint",
    "environment_fingerprint",
    "environment_fingerprints_equal",
    "verify_trace_canary",
]
