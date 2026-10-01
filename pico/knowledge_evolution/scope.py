"""Repository-scoped identity resolution without credential persistence."""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from .store import ImmutableWriteStatus, KnowledgeRecordStore, KnowledgeStoreError
from .types import require_digest, structural_digest

REPOSITORY_SCOPE_SCHEMA = "pico.repository-scope.v1"
SCHEMA_VERSION = 1
_PROJECT_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SCP_REMOTE = re.compile(r"^(?:[^@/:]+@)?(?P<host>[^:/]+):(?P<path>.+)$")


class RepositoryIdentityMethod(str, Enum):
    EXPLICIT_PROJECT_KEY = "explicit_project_key"
    GIT_REMOTE = "git_remote"
    LOCAL_REPOSITORY_UUID = "local_repository_uuid"


class RepositoryScopeStatus(str, Enum):
    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


class RepositoryScopeReason(str, Enum):
    EXPLICIT_PROJECT_KEY = "explicit_project_key"
    GIT_REMOTE = "git_remote"
    LOCAL_REPOSITORY_UUID = "local_repository_uuid"
    INVALID_PROJECT_KEY = "invalid_project_key"
    AMBIGUOUS_REMOTE = "ambiguous_remote"
    NON_GIT_REQUIRES_PROJECT_KEY = "non_git_requires_project_key"
    LOCAL_BINDING_INVALID = "local_binding_invalid"


@dataclass(frozen=True)
class RepositoryScopeIdentity:
    repository_scope_id: str
    method: RepositoryIdentityMethod
    evidence: tuple[tuple[str, str], ...]
    structural_digest: str
    schema: str = REPOSITORY_SCOPE_SCHEMA
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        method: RepositoryIdentityMethod,
        canonical_identity: str,
        evidence: tuple[tuple[str, str], ...],
    ) -> "RepositoryScopeIdentity":
        method = RepositoryIdentityMethod(method)
        scope_id = structural_digest(
            {"schema": REPOSITORY_SCOPE_SCHEMA, "method": method.value, "identity": canonical_identity}
        )
        normalized_evidence = tuple(sorted((str(key), str(value)) for key, value in evidence))
        payload = {
            "schema": REPOSITORY_SCOPE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "repository_scope_id": scope_id,
            "method": method.value,
            "evidence": [list(item) for item in normalized_evidence],
        }
        return cls(
            repository_scope_id=scope_id,
            method=method,
            evidence=normalized_evidence,
            structural_digest=structural_digest(payload),
        )

    def __post_init__(self) -> None:
        if self.schema != REPOSITORY_SCOPE_SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported repository scope schema")
        object.__setattr__(self, "method", RepositoryIdentityMethod(self.method))
        require_digest(self.repository_scope_id, "repository_scope_id")
        require_digest(self.structural_digest, "structural_digest")
        normalized = tuple(sorted((str(key), str(value)) for key, value in self.evidence))
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("repository scope evidence keys must be unique")
        object.__setattr__(self, "evidence", normalized)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "repository_scope_id": self.repository_scope_id,
            "method": self.method.value,
            "evidence": [list(item) for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "structural_digest": self.structural_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositoryScopeIdentity":
        identity = cls(
            repository_scope_id=str(value["repository_scope_id"]),
            method=RepositoryIdentityMethod(value["method"]),
            evidence=tuple((str(item[0]), str(item[1])) for item in value["evidence"]),
            structural_digest=str(value["structural_digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )
        if identity.structural_digest != structural_digest(identity._payload()):
            raise ValueError("repository scope structural digest mismatch")
        return identity


@dataclass(frozen=True)
class RepositoryScopeResolution:
    status: RepositoryScopeStatus
    reason: RepositoryScopeReason
    identity: RepositoryScopeIdentity | None = None

    @property
    def resolved(self) -> bool:
        return self.status is RepositoryScopeStatus.RESOLVED and self.identity is not None


def normalize_project_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s+", "-", normalized)
    if not _PROJECT_KEY.fullmatch(normalized):
        raise ValueError("project key must match [a-z0-9][a-z0-9._-]{0,127}")
    return normalized


def normalize_git_remote(value: str) -> tuple[str, str, str]:
    """Return canonical identity, host, and path without userinfo/query data."""

    raw = value.strip()
    match = _SCP_REMOTE.fullmatch(raw) if "://" not in raw else None
    if match is not None:
        host = match.group("host").lower().rstrip(".")
        path = match.group("path").split("?", 1)[0].split("#", 1)[0]
    else:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https", "ssh", "git"} or not parsed.hostname:
            raise ValueError("unsupported Git remote")
        host = parsed.hostname.lower().rstrip(".")
        port = parsed.port
        if port and not (
            (parsed.scheme.lower() == "ssh" and port == 22)
            or (parsed.scheme.lower() == "https" and port == 443)
            or (parsed.scheme.lower() == "http" and port == 80)
        ):
            host = f"{host}:{port}"
        path = unquote(parsed.path)
    path = path.strip().strip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    path = re.sub(r"/+", "/", path)
    if not host or not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("Git remote lacks a canonical host/repository path")
    return f"{host}/{path}", host, path


class RepositoryScopeResolver:
    def __init__(
        self,
        workspace: Path,
        state_root: Path,
        *,
        project_key: str | None = None,
        primary_remote: str = "origin",
        uuid_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.workspace = Path(workspace)
        self.project_key = project_key
        self.primary_remote = primary_remote
        self.uuid_factory = uuid_factory
        self.store = KnowledgeRecordStore(state_root)
        self._git_executable = shutil.which("git")

    def resolve(self) -> RepositoryScopeResolution:
        if self.project_key is not None:
            try:
                key = normalize_project_key(self.project_key)
            except ValueError:
                return RepositoryScopeResolution(
                    RepositoryScopeStatus.INVALID, RepositoryScopeReason.INVALID_PROJECT_KEY
                )
            return self._persist(
                RepositoryScopeIdentity.create(
                    method=RepositoryIdentityMethod.EXPLICIT_PROJECT_KEY,
                    canonical_identity=f"project:{key}",
                    evidence=(("project_key", key),),
                ),
                RepositoryScopeReason.EXPLICIT_PROJECT_KEY,
            )

        if self._git("rev-parse", "--is-inside-work-tree") != "true":
            return RepositoryScopeResolution(
                RepositoryScopeStatus.UNAVAILABLE,
                RepositoryScopeReason.NON_GIT_REQUIRES_PROJECT_KEY,
            )

        remotes = self._git_lines("remote", "get-url", "--all", self.primary_remote)
        normalized: set[tuple[str, str, str]] = set()
        for remote in remotes:
            try:
                normalized.add(normalize_git_remote(remote))
            except ValueError:
                continue
        if len(normalized) > 1:
            return RepositoryScopeResolution(
                RepositoryScopeStatus.AMBIGUOUS, RepositoryScopeReason.AMBIGUOUS_REMOTE
            )
        if len(normalized) == 1:
            canonical, host, path = normalized.pop()
            return self._persist(
                RepositoryScopeIdentity.create(
                    method=RepositoryIdentityMethod.GIT_REMOTE,
                    canonical_identity=f"remote:{canonical}",
                    evidence=(
                        ("remote_host", host),
                        ("remote_path_digest", structural_digest({"path": path})),
                    ),
                ),
                RepositoryScopeReason.GIT_REMOTE,
            )

        try:
            common_dir_text = self._git("rev-parse", "--git-common-dir")
            if not common_dir_text:
                raise ValueError("missing Git common directory")
            common_dir = Path(common_dir_text)
            if not common_dir.is_absolute():
                common_dir = self.workspace / common_dir
            common_dir = common_dir.resolve(strict=True)
            stat = common_dir.stat()
            root_commits = self._git_lines("rev-list", "--max-parents=0", "HEAD")
            root_commit = sorted(root_commits)[0] if root_commits else "unborn"
            common_identity = structural_digest(
                {"device": int(stat.st_dev), "inode": int(stat.st_ino)}
            )
            binding_evidence = {
                "common_directory_identity": common_identity,
                "root_commit": root_commit,
            }
            binding_digest = structural_digest(binding_evidence)
            binding = self.store.load_or_create_local_binding(
                binding_digest=binding_digest,
                evidence=binding_evidence,
                uuid_factory=self.uuid_factory,
            )
            local_uuid = str(binding["local_repository_uuid"])
        except (OSError, ValueError, KnowledgeStoreError):
            return RepositoryScopeResolution(
                RepositoryScopeStatus.INVALID, RepositoryScopeReason.LOCAL_BINDING_INVALID
            )
        return self._persist(
            RepositoryScopeIdentity.create(
                method=RepositoryIdentityMethod.LOCAL_REPOSITORY_UUID,
                canonical_identity=f"local:{local_uuid}",
                evidence=(
                    ("binding_digest", binding_digest),
                    ("common_directory_identity", common_identity),
                    ("root_commit", root_commit),
                ),
            ),
            RepositoryScopeReason.LOCAL_REPOSITORY_UUID,
        )

    def _persist(
        self, identity: RepositoryScopeIdentity, reason: RepositoryScopeReason
    ) -> RepositoryScopeResolution:
        status = self.store.write_scope(identity)
        if status is ImmutableWriteStatus.CONFLICT:
            return RepositoryScopeResolution(
                RepositoryScopeStatus.INVALID, RepositoryScopeReason.LOCAL_BINDING_INVALID
            )
        return RepositoryScopeResolution(RepositoryScopeStatus.RESOLVED, reason, identity)

    def _git(self, *arguments: str) -> str:
        if self._git_executable is None:
            return ""
        completed = subprocess.run(
            [self._git_executable, "-C", str(self.workspace), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    def _git_lines(self, *arguments: str) -> tuple[str, ...]:
        output = self._git(*arguments)
        return tuple(line.strip() for line in output.splitlines() if line.strip())


__all__ = [
    "REPOSITORY_SCOPE_SCHEMA",
    "RepositoryIdentityMethod",
    "RepositoryScopeIdentity",
    "RepositoryScopeReason",
    "RepositoryScopeResolution",
    "RepositoryScopeResolver",
    "RepositoryScopeStatus",
    "normalize_git_remote",
    "normalize_project_key",
]
