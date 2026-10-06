"""Offline sandbox capability gate and smoke for JEV confirmatory benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import site
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from benchmarks.picobench.canonical import canonical_digest, canonical_json
from pico.sandbox import SandboxExecutor, build_executor

if TYPE_CHECKING:
    from pico.sandbox.config import SandboxConfig

SCHEMA = "pico.jev6-benchmark-sandbox-capability.v1"
SMOKE_SCHEMA = "pico.jev6-benchmark-sandbox-smoke.v1"
AVAILABLE_BACKENDS = ("none", "auto", "boxlite")
REQUIRED_BACKEND = "boxlite"
ENVIRONMENT_FINGERPRINT_SCHEMA = "pico.environment-fingerprint.v1"


@dataclass
class _ExplicitSmokeConfig:
    """Dependency-light config used only by the explicit real-smoke CLI."""

    backend: str = "boxlite"
    image: str = "ubuntu:22.04"
    image_search_registry: str | None = None
    runtime_home: Path | None = None
    cpus: int = 2
    memory_mib: int = 2048
    disk_size_gb: int | None = None
    allow_net: bool | list[str] = False
    extra_volumes: list[list[str]] = field(default_factory=list)
    default_timeout: int = 120
    verify_timeout: int = 30
    create_timeout: int = 300


def _environment_fingerprint() -> dict[str, object]:
    """Return the benchmark fingerprint without importing tracing subsystems."""
    distributions = sorted(
        (dist.metadata.get("Name", "").casefold(), dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    paths = tuple(str(Path(raw).resolve()) for raw in sys.path if raw)
    payload: dict[str, object] = {
        "schema": ENVIRONMENT_FINGERPRINT_SCHEMA,
        "schema_version": 1,
        "python_executable_digest": canonical_digest(str(Path(sys.executable).resolve())),
        "python_version": list(sys.version_info[:3]),
        "sys_path_digest": canonical_digest(paths),
        "distribution_digest": canonical_digest(distributions),
        "distribution_count": len(distributions),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
    }
    return {**payload, "fingerprint_digest": canonical_digest(payload)}


def _platform_support(
    *, platform_name: str, machine: str, kvm_available: bool | None
) -> tuple[bool, str]:
    if platform_name == "linux":
        available = (
            kvm_available
            if kvm_available is not None
            else Path("/dev/kvm").exists()
            and os.access("/dev/kvm", os.R_OK | os.W_OK)
        )
        return bool(available), "linux_kvm" if available else "linux_kvm_unavailable"
    if platform_name == "darwin":
        supported = machine.lower() in {"arm64", "aarch64"}
        return supported, "macos_apple_silicon" if supported else "macos_apple_silicon_required"
    return False, f"unsupported_platform:{platform_name}"


def assess_benchmark_sandbox(
    config: SandboxConfig,
    *,
    workspace: Path | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    dependency_available: bool | None = None,
    kvm_available: bool | None = None,
) -> dict[str, Any]:
    """Return bounded, secret-free capability evidence without starting a VM."""

    current_platform = platform_name or sys.platform
    current_machine = machine or platform.machine()
    platform_supported, platform_capability = _platform_support(
        platform_name=current_platform,
        machine=current_machine,
        kvm_available=kvm_available,
    )
    boxlite_available = (
        importlib.util.find_spec("boxlite") is not None
        if dependency_available is None
        else dependency_available
    )
    checks = {
        "explicit_boxlite_backend": config.backend == REQUIRED_BACKEND,
        "host_execution_disabled": config.backend != "none",
        "filesystem_isolation_supported": platform_supported and boxlite_available,
        "network_isolation_configured": config.allow_net is False,
        "process_isolation_supported": platform_supported and boxlite_available,
        "dependency_isolation_supported": platform_supported and boxlite_available,
        "no_extra_host_volumes": not config.extra_volumes,
        "platform_supported": platform_supported,
        "boxlite_dependency_available": boxlite_available,
    }
    if config.backend == "none":
        reason = "sandbox_backend_none"
    elif config.backend != REQUIRED_BACKEND:
        reason = "sandbox_backend_not_explicit_boxlite"
    elif not platform_supported:
        reason = platform_capability
    elif not boxlite_available:
        reason = "boxlite_dependency_unavailable"
    elif config.allow_net is not False:
        reason = "sandbox_network_not_disabled"
    elif config.extra_volumes:
        reason = "sandbox_extra_host_volumes_configured"
    else:
        reason = None
    evidence = {
        "schema": SCHEMA,
        "available_backends": AVAILABLE_BACKENDS,
        "required_backend": REQUIRED_BACKEND,
        "configured_backend": config.backend,
        "platform_family": current_platform,
        "platform_capability": platform_capability,
        "checks": checks,
        "passed": all(checks.values()),
        "infra_invalid_reason": reason,
        "host_execution_allowed": config.backend == "none",
        "workspace_root_identity_digest": (
            canonical_digest(str(workspace.resolve())) if workspace is not None else None
        ),
        "provider_calls": 0,
        "agent_turns": 0,
        "external_network_requests": 0,
        "typesafe_enabled": False,
    }
    evidence["integrity_digest"] = canonical_digest(evidence)
    return evidence


def audit_config_path(config_path: Path, *, workspace: Path | None = None) -> dict[str, Any]:
    from pico.config.loader import load_config

    return assess_benchmark_sandbox(
        load_config(config_path).tools.sandbox,
        workspace=workspace,
    )


def _probe_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "content_digest": None}
    if not path.is_file():
        return {"exists": True, "content_digest": "non_file"}
    return {"exists": True, "content_digest": hashlib.sha256(path.read_bytes()).hexdigest()}


async def run_benchmark_sandbox_smoke(
    config: SandboxConfig,
    workspace: Path,
    *,
    executor_factory: Callable[[SandboxConfig, Path], SandboxExecutor] = build_executor,
    capability_evidence: dict[str, Any] | None = None,
    require_real_boxlite: bool = True,
) -> dict[str, Any]:
    """Exercise only local VM boundaries; never starts an Agent or Provider."""

    workspace = workspace.resolve()
    capability = capability_evidence or assess_benchmark_sandbox(
        config, workspace=workspace
    )
    result: dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "sandbox_capability": capability,
        "sandbox_backend": config.backend,
        "workspace_root_identity_digest": canonical_digest(str(workspace)),
        "smoke_executed": False,
        "validation_result": "HOLD_SANDBOX",
        "real_boxlite_backend": False,
        "microvm_started": False,
        "workspace_mount_only": False,
        "sandbox_cleanup_completed": False,
        "filesystem_isolation_supported": False,
        "network_isolation_supported": False,
        "host_execution_allowed": config.backend == "none",
        "inside_workspace_write": False,
        "worktree_read_write": False,
        "outside_workspace_host_write_rejected": False,
        "host_user_site_mutation_rejected": False,
        "host_fingerprint_unchanged": False,
        "provider_calls": 0,
        "agent_turns": 0,
        "external_network_requests": 0,
        "typesafe_enabled": False,
    }
    if not capability["passed"]:
        result["passed"] = False
        result["infra_invalid_reason"] = capability["infra_invalid_reason"]
        result["integrity_digest"] = canonical_digest(result)
        return result

    workspace.mkdir(parents=True, exist_ok=True)
    inside_probe = workspace / ".jev6s-sandbox-smoke"
    outside_probe = workspace.parent / ".jev6s-outside-host-probe"
    user_site_probe = Path(site.getusersitepackages()) / ".jev6s-host-mutation-probe"
    host_before = {
        "environment": _environment_fingerprint(),
        "outside": _probe_identity(outside_probe),
        "user_site": _probe_identity(user_site_probe),
    }
    owned_ids: set[str] = set()
    executor = (
        build_executor(config, workspace, owned_ids)
        if executor_factory is build_executor
        else executor_factory(config, workspace)
    )
    real_boxlite = type(executor).__name__ == "BoxliteExecutor"
    result["resolved_executor"] = type(executor).__name__
    result["real_boxlite_backend"] = real_boxlite
    if not executor.is_sandboxed:
        result["passed"] = False
        result["infra_invalid_reason"] = "executor_reports_host_execution"
        result["integrity_digest"] = canonical_digest(result)
        return result
    if require_real_boxlite and not real_boxlite:
        result["passed"] = False
        result["infra_invalid_reason"] = "executor_is_not_real_boxlite"
        result["integrity_digest"] = canonical_digest(result)
        return result

    try:
        async with executor:
            result["microvm_started"] = bool(owned_ids) if real_boxlite else True
            created = await executor.exec("printf 'jev6s' > .jev6s-sandbox-smoke")
            read_write = await executor.exec(
                "test \"$(cat .jev6s-sandbox-smoke)\" = jev6s && printf pass"
            )
            mount = await executor.exec(
                "awk '$2 == \"/workspace\" { count++ } END { exit count == 1 ? 0 : 1 }' /proc/mounts"
            )
            outside = await executor.exec(
                "printf blocked > /__jev6s_unmounted_host__/outside"
            )
            user_site = await executor.exec(
                "printf blocked > /__jev6s_unmounted_python_user_site__/mutation"
            )
            network = await executor.exec(
                "awk 'NR > 1 && $2 == \"00000000\" { found=1 } END { exit found ? 1 : 0 }' /proc/net/route"
            )
    except Exception as exc:  # noqa: BLE001 - bounded infrastructure evidence
        result["passed"] = False
        result["infra_invalid_reason"] = "sandbox_runtime_failure"
        result["bounded_failure_type"] = type(exc).__name__
        result["bounded_failure_detail"] = str(exc)[:240]
        result["sandbox_cleanup_completed"] = not owned_ids
        result["integrity_digest"] = canonical_digest(result)
        return result

    host_after = {
        "environment": _environment_fingerprint(),
        "outside": _probe_identity(outside_probe),
        "user_site": _probe_identity(user_site_probe),
    }
    result.update(
        smoke_executed=True,
        filesystem_isolation_supported=True,
        network_isolation_supported=network.exit_code == 0,
        host_execution_allowed=False,
        workspace_mount_only=mount.exit_code == 0,
        sandbox_cleanup_completed=not owned_ids,
        inside_workspace_write=(
            created.exit_code == 0
            and inside_probe.is_file()
            and inside_probe.read_text(encoding="utf-8") == "jev6s"
        ),
        worktree_read_write=read_write.exit_code == 0 and read_write.stdout.strip() == "pass",
        outside_workspace_host_write_rejected=(
            outside.exit_code != 0 and not outside_probe.exists()
        ),
        host_user_site_mutation_rejected=(
            user_site.exit_code != 0 and _probe_identity(user_site_probe) == host_before["user_site"]
        ),
        host_fingerprint_unchanged=host_before == host_after,
    )
    inside_probe.unlink(missing_ok=True)
    result["passed"] = all(
        result[name]
        for name in (
            "filesystem_isolation_supported",
            "network_isolation_supported",
            "real_boxlite_backend" if require_real_boxlite else "filesystem_isolation_supported",
            "microvm_started",
            "workspace_mount_only",
            "sandbox_cleanup_completed",
            "inside_workspace_write",
            "worktree_read_write",
            "outside_workspace_host_write_rejected",
            "host_user_site_mutation_rejected",
            "host_fingerprint_unchanged",
        )
    )
    result["infra_invalid_reason"] = None if result["passed"] else "sandbox_smoke_failure"
    result["validation_result"] = (
        "PASS_REAL_BOXLITE"
        if result["passed"] and require_real_boxlite
        else "PASS_DETERMINISTIC"
        if result["passed"]
        else "HOLD_SANDBOX"
    )
    result["integrity_digest"] = canonical_digest(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline JEV benchmark sandbox smoke")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--backend", choices=AVAILABLE_BACKENDS)
    parser.add_argument("--disable-network", action="store_true")
    parser.add_argument("--image-search-registry")
    parser.add_argument("--runtime-home", type=Path)
    args = parser.parse_args(argv)
    explicit_real_smoke = (
        args.config_path is None
        and args.backend == "boxlite"
        and args.disable_network
        and args.image_search_registry is not None
    )
    if explicit_real_smoke:
        registry = args.image_search_registry.strip()
        if not registry or "://" in registry or any(c.isspace() for c in registry):
            parser.error("--image-search-registry must be a registry host without a URL scheme")
        runtime_home = args.runtime_home or Path(tempfile.gettempdir()) / "pico-jev6s-boxlite"
        config = _ExplicitSmokeConfig(
            image_search_registry=registry,
            runtime_home=runtime_home,
        )
    else:
        from pico.config.loader import get_config_path, load_config
        from pico.sandbox.config import SandboxConfig

        config_path = (args.config_path or get_config_path()).resolve()
        config = load_config(config_path).tools.sandbox
        updates: dict[str, Any] = {}
        if args.backend is not None:
            updates["backend"] = args.backend
        if args.disable_network:
            updates["allow_net"] = False
        if args.image_search_registry is not None:
            updates["image_search_registry"] = args.image_search_registry
        if updates:
            updates["extra_volumes"] = []
            config = SandboxConfig.model_validate({**config.model_dump(), **updates})
    result = asyncio.run(run_benchmark_sandbox_smoke(config, args.workspace))
    print(canonical_json(result))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AVAILABLE_BACKENDS",
    "REQUIRED_BACKEND",
    "assess_benchmark_sandbox",
    "audit_config_path",
    "run_benchmark_sandbox_smoke",
]
