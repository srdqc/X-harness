"""Fail-closed immutable local persistence for P3.1 records."""

from __future__ import annotations

import json
import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from pico.utils.portable_lock import file_lock

from .task_success import TaskSuccessEvidence
from .types import KnowledgeCandidate, canonical_json, structural_digest

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_T = TypeVar("_T")


class KnowledgeStoreError(RuntimeError):
    pass


class ImmutableWriteStatus(str, Enum):
    CREATED = "created"
    UNCHANGED = "unchanged"
    CONFLICT = "conflict"


def _safe_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError("record ID contains unsafe path characters")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _atomic_create_or_compare(path: Path, payload: Mapping[str, Any]) -> ImmutableWriteStatus:
    data = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise KnowledgeStoreError(f"cannot read existing immutable record: {path}") from exc
            return (
                ImmutableWriteStatus.UNCHANGED
                if existing == data
                else ImmutableWriteStatus.CONFLICT
            )
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            if os.name == "posix":
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return ImmutableWriteStatus.CREATED


def _load(path: Path, decoder: Callable[[Mapping[str, Any]], _T]) -> _T:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnowledgeStoreError(f"invalid immutable record: {path}") from exc
    if not isinstance(raw, dict):
        raise KnowledgeStoreError(f"immutable record must be an object: {path}")
    try:
        return decoder(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise KnowledgeStoreError(f"immutable record failed validation: {path}") from exc


class KnowledgeRecordStore:
    """Own immutable scope, task-success, candidate, and local-binding records."""

    def __init__(self, state_root: Path) -> None:
        self.root = Path(state_root) / "knowledge" / "v1"
        self.scopes = self.root / "scopes"
        self.task_success = self.root / "task-success"
        self.candidates = self.root / "candidates"
        self.bindings = self.root / "scope-bindings"

    @staticmethod
    def _path(directory: Path, record_id: str) -> Path:
        safe = _safe_id(record_id)
        path = directory / f"{safe}.json"
        resolved_directory = directory.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(resolved_directory):
            raise ValueError("record path escapes store root")
        return path

    def write_scope(self, identity: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(self._path(self.scopes, identity.repository_scope_id), identity.to_dict())

    def read_scope(self, scope_id: str) -> Any | None:
        path = self._path(self.scopes, scope_id)
        if not path.exists():
            return None
        from .scope import RepositoryScopeIdentity

        value = _load(path, RepositoryScopeIdentity.from_dict)
        if value.repository_scope_id != scope_id:
            raise KnowledgeStoreError("scope record path binding mismatch")
        return value

    def write_task_success(self, evidence: TaskSuccessEvidence) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.task_success, evidence.evidence_id), evidence.to_dict()
        )

    def read_task_success(self, evidence_id: str) -> TaskSuccessEvidence | None:
        path = self._path(self.task_success, evidence_id)
        if not path.exists():
            return None
        value = _load(path, TaskSuccessEvidence.from_dict)
        if value.evidence_id != evidence_id:
            raise KnowledgeStoreError("task-success record path binding mismatch")
        return value

    def write_candidate(self, candidate: KnowledgeCandidate) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.candidates, candidate.candidate_id), candidate.to_dict()
        )

    def read_candidate(self, candidate_id: str) -> KnowledgeCandidate | None:
        path = self._path(self.candidates, candidate_id)
        if not path.exists():
            return None
        value = _load(path, KnowledgeCandidate.from_dict)
        if value.candidate_id != candidate_id:
            raise KnowledgeStoreError("candidate record path binding mismatch")
        return value

    def load_or_create_local_binding(
        self,
        *,
        binding_digest: str,
        evidence: Mapping[str, Any],
        uuid_factory: Callable[[], str],
    ) -> Mapping[str, Any]:
        path = self._path(self.bindings, binding_digest)
        if path.exists():
            value = _load(path, lambda item: dict(item))
            payload = {key: item for key, item in value.items() if key != "record_digest"}
            if value.get("record_digest") != structural_digest(payload):
                raise KnowledgeStoreError("local repository binding digest mismatch")
            if value.get("binding_digest") != binding_digest or value.get("evidence") != dict(evidence):
                raise KnowledgeStoreError("local repository binding evidence mismatch")
            return value
        payload = {
            "schema": "pico.repository-local-binding.v1",
            "schema_version": 1,
            "binding_digest": binding_digest,
            "local_repository_uuid": _safe_id(uuid_factory()),
            "evidence": dict(evidence),
        }
        record = {**payload, "record_digest": structural_digest(payload)}
        status = _atomic_create_or_compare(path, record)
        if status is ImmutableWriteStatus.CONFLICT:
            return self.load_or_create_local_binding(
                binding_digest=binding_digest,
                evidence=evidence,
                uuid_factory=uuid_factory,
            )
        return record


__all__ = [
    "ImmutableWriteStatus",
    "KnowledgeRecordStore",
    "KnowledgeStoreError",
]
