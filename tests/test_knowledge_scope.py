from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pico.knowledge_evolution import (
    RepositoryIdentityMethod,
    RepositoryScopeReason,
    RepositoryScopeResolver,
    RepositoryScopeStatus,
    normalize_git_remote,
)


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    path = tmp_path / name
    path.mkdir()
    _git(path, "init")
    return path


def test_explicit_project_key_is_normalized_and_has_precedence(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "remote", "add", "origin", "https://github.com/ignored/repo.git")
    result = RepositoryScopeResolver(
        repo, tmp_path / "state", project_key="  My Project_Key  "
    ).resolve()
    assert result.status is RepositoryScopeStatus.RESOLVED
    assert result.reason is RepositoryScopeReason.EXPLICIT_PROJECT_KEY
    assert result.identity is not None
    assert result.identity.method is RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY
    assert dict(result.identity.evidence)["project_key"] == "my-project_key"
    repeated = RepositoryScopeResolver(
        repo, tmp_path / "other-state", project_key="My Project_Key"
    ).resolve()
    assert repeated.identity == result.identity


def test_https_and_ssh_remotes_normalize_equivalently_without_credentials() -> None:
    https = normalize_git_remote(
        "https://user:secret-token@GitHub.com/org/Repo.git?access_token=hidden#fragment"
    )
    ssh = normalize_git_remote("git@github.com:org/Repo.git")
    assert https == ssh == ("github.com/org/Repo", "github.com", "org/Repo")
    assert "secret" not in repr(https)
    assert "token" not in repr(https)


def test_remote_scope_persists_only_non_secret_evidence(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    remote = "https://user:password@github.com/private/repo.git?token=top-secret"
    _git(repo, "remote", "add", "origin", remote)
    result = RepositoryScopeResolver(repo, tmp_path / "state").resolve()
    assert result.identity is not None
    persisted = next((tmp_path / "state" / "knowledge" / "v1" / "scopes").glob("*.json"))
    text = persisted.read_text(encoding="utf-8")
    assert "password" not in text
    assert "top-secret" not in text
    assert remote not in text


def test_local_uuid_fallback_is_stable_and_bound_to_repository(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    state = tmp_path / "state"
    first = RepositoryScopeResolver(repo, state, uuid_factory=lambda: "repo-local-1").resolve()
    second = RepositoryScopeResolver(repo, state, uuid_factory=lambda: "unused").resolve()
    moved = tmp_path / "moved-repo"
    repo.rename(moved)
    after_move = RepositoryScopeResolver(moved, state, uuid_factory=lambda: "unused").resolve()
    other = RepositoryScopeResolver(
        _repo(tmp_path, "other"), state, uuid_factory=lambda: "repo-local-2"
    ).resolve()
    assert (
        first.identity is not None
        and second.identity is not None
        and after_move.identity is not None
        and other.identity is not None
    )
    assert first.identity.method is RepositoryIdentityMethod.LOCAL_REPOSITORY_UUID
    assert first.identity == second.identity == after_move.identity
    assert other.identity.repository_scope_id != first.identity.repository_scope_id
    assert all("path" not in key for key, _ in first.identity.evidence)


def test_tampered_local_binding_is_invalid(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    state = tmp_path / "state"
    assert RepositoryScopeResolver(repo, state, uuid_factory=lambda: "local-1").resolve().resolved
    binding = next((state / "knowledge" / "v1" / "scope-bindings").glob("*.json"))
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["local_repository_uuid"] = "tampered"
    binding.write_text(json.dumps(payload), encoding="utf-8")
    result = RepositoryScopeResolver(repo, state).resolve()
    assert result.status is RepositoryScopeStatus.INVALID
    assert result.reason is RepositoryScopeReason.LOCAL_BINDING_INVALID


def test_non_git_path_and_git_branch_or_revision_are_not_identity(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / ".git").write_text("gitdir: missing", encoding="utf-8")
    unresolved = RepositoryScopeResolver(plain, tmp_path / "state").resolve()
    assert unresolved.status is RepositoryScopeStatus.UNAVAILABLE

    repo = _repo(tmp_path, "git")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    first = RepositoryScopeResolver(repo, tmp_path / "state2", uuid_factory=lambda: "stable").resolve()
    _git(repo, "branch", "-m", "renamed")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "next",
    )
    second = RepositoryScopeResolver(repo, tmp_path / "state2").resolve()
    assert first.identity == second.identity
