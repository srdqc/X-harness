"""Non-Provider preparation and preflight for P3R campaigns."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from benchmarks.picobench.canonical import canonical_digest
from pico.agent.tools.registry import normalize_repository_read_path
from pico.tracing import evidence

from .artifacts import freeze_manifest
from .knowledge import prepare_approved_corpus
from .protocol import assert_fair_plan, create_manifest
from .schema import CampaignManifest, CampaignMode, CampaignPaths, RuntimeBudget
from .solvability import audit_pilot_solvability
from .tasks import TASKS
from .verifiers import VERIFIERS


@dataclass(frozen=True)
class PreflightResult:
    base_commit_sha: str
    task_count: int
    verifier_count: int
    corpus_candidate_count: int
    provider_config_present: bool
    worktree_supported: bool
    task_verifier_contract_version: int
    mechanical_solvability_passed: bool
    verifier_findings_persistence_available: bool
    repository_read_projection_passed: bool
    provider_failure_normalization_configured: bool
    validity_policy_version: int
    enforced_budget_semantics_verified: bool


def configuration_identity(config, budget: RuntimeBudget) -> dict[str, str]:
    defaults = config.agents.defaults
    provider_id = config.get_provider_name(defaults.model)
    provider_digest = canonical_digest(
        {
            "provider_id": provider_id,
            "model": defaults.model,
            "temperature": defaults.temperature,
            "max_tokens": defaults.max_tokens,
            "reasoning_effort": defaults.reasoning_effort,
        }
    )
    tool_digest = canonical_digest(
        {
            "disabled_tools": tuple(sorted(config.tools.disabled_tools)),
            "restrict_to_workspace": config.tools.restrict_to_workspace,
            "mcp_server_names": tuple(sorted(config.tools.mcp_servers)),
            "tool_search": config.tools.tool_search.model_dump(mode="json"),
            "sandbox": config.tools.sandbox.model_dump(mode="json"),
        }
    )
    runtime_digest = canonical_digest(
        {
            "budget": budget,
            "max_tool_iterations": defaults.max_tool_iterations,
            "context_window_tokens": defaults.context_window_tokens,
        }
    )
    return {
        "provider_id": provider_id,
        "model": defaults.model,
        "provider_digest": provider_digest,
        "tool_digest": tool_digest,
        "runtime_digest": runtime_digest,
    }


def prepare_campaign(
    *,
    repository: Path,
    output_root: Path,
    mode: CampaignMode,
    base_commit: str,
    seed: int,
    reviewer_id: str,
    config,
) -> tuple[CampaignPaths, CampaignManifest, PreflightResult]:
    budget = RuntimeBudget(
        max_agent_iterations=config.agents.defaults.max_tool_iterations,
        provider_logical_calls_observational=config.agents.defaults.max_tool_iterations,
        tool_calls_observational=40,
        context_window_tokens=config.agents.defaults.context_window_tokens,
    )
    identity = configuration_identity(config, budget)
    base_sha = resolve_base_commit(repository, base_commit)
    manifest = create_manifest(
        mode=mode,
        base_commit_sha=base_sha,
        seed=seed,
        provider_id=identity["provider_id"],
        actual_model_id=identity["model"],
        provider_model_config_digest=identity["provider_digest"],
        tool_config_digest=identity["tool_digest"],
        runtime_config_digest=identity["runtime_digest"],
        budget=budget,
    )
    result = preflight(
        repository=repository,
        base_commit_sha=base_sha,
        reviewer_id=reviewer_id,
        config=config,
        manifest=manifest,
    )
    paths = CampaignPaths.at(Path(output_root) / manifest.campaign_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    freeze_manifest(paths, manifest)
    return paths, manifest, result


def preflight(
    *,
    repository: Path,
    base_commit_sha: str,
    reviewer_id: str,
    config,
    manifest: CampaignManifest,
) -> PreflightResult:
    """Validate a campaign without constructing or invoking a Provider."""

    manifest.validate()
    assert_fair_plan(manifest)
    if len(TASKS) != 12 or len({item.task_id for item in TASKS}) != 12:
        raise ValueError("P3R requires exactly twelve unique frozen tasks")
    if set(item.verifier_id for item in TASKS) != set(VERIFIERS):
        raise ValueError("P3R task/verifier set mismatch")
    if not Path(__file__).with_name("verifiers.py").is_file():
        raise ValueError("sealed verifier implementation is missing")
    # The evaluated base must predate this harness so normal repository tools
    # cannot inspect the verifier implementation in the Agent worktree.
    hidden_path = "benchmarks/picobench/packs/knowledge_evolution_live/verifiers.py"
    if _git_ok(repository, "cat-file", "-e", f"{base_commit_sha}:{hidden_path}"):
        raise ValueError("base commit contains sealed P3R verifier implementation")
    if not _provider_config_present(config):
        raise ValueError("configured Provider credentials are not present")

    with tempfile.TemporaryDirectory(prefix="p3r-preflight-") as temporary:
        temp = Path(temporary)
        worktree = temp / "w"
        state = temp / "s"
        _git(repository, "worktree", "add", "--detach", str(worktree), base_commit_sha)
        try:
            actual = _git(worktree, "rev-parse", "HEAD")
            if actual != base_commit_sha:
                raise ValueError("preflight worktree base SHA mismatch")
            candidates = prepare_approved_corpus(
                state_root=state,
                workspace=worktree,
                reviewer_id=reviewer_id,
            )
            probe = temp / "write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            solvability = audit_pilot_solvability(worktree)
            if not all(item.passed for item in solvability):
                raise ValueError(f"P3R mechanical solvability failed: {solvability!r}")
            inside = worktree / "pico" / "probe.py"
            projection_ok = (
                normalize_repository_read_path(str(inside), worktree) == "pico/probe.py"
                and normalize_repository_read_path(str(worktree.parent / "outside.py"), worktree)
                is None
            )
            if not projection_ok:
                raise ValueError("P3R repository read projection self-test failed")
        finally:
            _git(repository, "worktree", "remove", "--force", str(worktree))
    return PreflightResult(
        base_commit_sha,
        len(manifest.task_ids),
        len(VERIFIERS),
        len(candidates),
        True,
        True,
        manifest.task_contract_version,
        True,
        manifest.schema_version >= 2,
        True,
        evidence.normalize_provider_failure(
            "server",
            message="Unable to get json response; Expecting value; Original Response:",
        )
        == evidence.PROVIDER_MALFORMED_RESPONSE,
        manifest.validity_policy_version,
        manifest.budget.max_agent_iterations > 0
        and manifest.budget.final_synthesis_call_allowed
        and manifest.budget.provider_logical_calls_observational is not None,
    )


def resolve_base_commit(repository: Path, value: str) -> str:
    resolved = _git(repository, "rev-parse", "--verify", f"{value}^{{commit}}")
    if len(resolved) != 40:
        raise ValueError("base commit did not resolve to a full SHA")
    return resolved


def _provider_config_present(config) -> bool:
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    provider = config.get_provider(model)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return True
    if provider_name == "azure_openai":
        return bool(provider and provider.api_key and provider.api_base)
    from pico.providers.registry import find_by_name

    spec = find_by_name(provider_name)
    return bool(provider and provider.api_key) or bool(spec and (spec.is_oauth or spec.is_local))


def _git_ok(repository: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=repository, check=False, capture_output=True, text=True
    ).returncode == 0


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],  # noqa: S607 -- repository-native executable
        cwd=repository, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


__all__ = [
    "PreflightResult",
    "configuration_identity",
    "preflight",
    "prepare_campaign",
    "resolve_base_commit",
]
