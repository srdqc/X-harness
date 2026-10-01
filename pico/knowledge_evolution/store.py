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
    return _atomic_bytes_create_or_compare(path, _canonical_bytes(payload))


def _atomic_bytes_create_or_compare(path: Path, data: bytes) -> ImmutableWriteStatus:
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
        self.extractions = self.root / "extractions"
        self.validations = self.root / "validations"
        self.reviews = self.root / "reviews"
        self.lifecycle = self.root / "lifecycle"
        self.materialized = self.root / "materialized"
        self.materialization_results = self.root / "materialization-results"
        self.applicability_results = self.root / "applicability-results"
        self.retrievals = self.root / "retrievals"
        self.usages = self.root / "usages"
        self.outcome_associations = self.root / "outcome-associations"
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

    def list_candidates(
        self, *, repository_scope_id: str | None = None
    ) -> tuple[KnowledgeCandidate, ...]:
        if repository_scope_id is not None:
            _safe_id(repository_scope_id)
        if not self.candidates.exists():
            return ()
        values: list[KnowledgeCandidate] = []
        for path in sorted(self.candidates.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeCandidate.from_dict)
            if path.stem != value.candidate_id:
                raise KnowledgeStoreError("candidate record path binding mismatch")
            if repository_scope_id is None or value.repository_scope_id == repository_scope_id:
                values.append(value)
        return tuple(values)

    def write_extraction_result(self, result: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.extractions, result.extraction_id), result.to_dict()
        )

    def read_extraction_result(self, extraction_id: str) -> Any | None:
        path = self._path(self.extractions, extraction_id)
        if not path.exists():
            return None
        from .extraction import KnowledgeExtractionResult

        value = _load(path, KnowledgeExtractionResult.from_dict)
        if value.extraction_id != extraction_id:
            raise KnowledgeStoreError("extraction result path binding mismatch")
        return value

    def write_validation(self, result: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.validations, result.validation_id), result.to_dict()
        )

    def read_validation(self, validation_id: str) -> Any | None:
        path = self._path(self.validations, validation_id)
        if not path.exists():
            return None
        from .validation import CandidateValidationResult

        value = _load(path, CandidateValidationResult.from_dict)
        if value.validation_id != validation_id:
            raise KnowledgeStoreError("validation result path binding mismatch")
        return value

    def list_validations(self, *, candidate_id: str | None = None) -> tuple[Any, ...]:
        if not self.validations.exists():
            return ()
        from .validation import CandidateValidationResult

        values = []
        for path in sorted(self.validations.glob("*.json"), key=lambda item: item.name):
            value = _load(path, CandidateValidationResult.from_dict)
            if value.validation_id != path.stem:
                raise KnowledgeStoreError("validation result path binding mismatch")
            if candidate_id is None or value.candidate_id == candidate_id:
                values.append(value)
        return tuple(values)

    def write_review(self, receipt: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.reviews, receipt.review_id), receipt.to_dict()
        )

    def read_review(self, review_id: str) -> Any | None:
        path = self._path(self.reviews, review_id)
        if not path.exists():
            return None
        from .review import KnowledgeReviewReceipt

        value = _load(path, KnowledgeReviewReceipt.from_dict)
        if value.review_id != review_id:
            raise KnowledgeStoreError("review receipt path binding mismatch")
        return value

    def list_reviews(self, *, candidate_id: str | None = None) -> tuple[Any, ...]:
        if not self.reviews.exists():
            return ()
        from .review import KnowledgeReviewReceipt

        values = []
        for path in sorted(self.reviews.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeReviewReceipt.from_dict)
            if value.review_id != path.stem:
                raise KnowledgeStoreError("review receipt path binding mismatch")
            if candidate_id is None or value.candidate_id == candidate_id:
                values.append(value)
        return tuple(values)

    def _lifecycle_path(self, repository_scope_id: str, candidate_id: str) -> Path:
        scope = _safe_id(repository_scope_id)
        candidate = _safe_id(candidate_id)
        directory = self.lifecycle / scope
        return self._path(directory, candidate).with_suffix(".jsonl")

    def read_lifecycle(
        self, repository_scope_id: str, candidate_id: str
    ) -> tuple[Any, ...]:
        path = self._lifecycle_path(repository_scope_id, candidate_id)
        if not path.exists():
            return ()
        from .lifecycle import KnowledgeLifecycleTransition

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise KnowledgeStoreError("cannot read lifecycle chain") from exc
        values = []
        for line in lines:
            if not line.strip():
                raise KnowledgeStoreError("lifecycle chain contains an empty line")
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("lifecycle record is not an object")
                values.append(KnowledgeLifecycleTransition.from_dict(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise KnowledgeStoreError("invalid lifecycle transition") from exc
        previous = None
        for sequence, value in enumerate(values, start=1):
            if (
                value.sequence != sequence
                or value.candidate_id != candidate_id
                or value.repository_scope_id != repository_scope_id
                or value.previous_transition_digest != previous
            ):
                raise KnowledgeStoreError("lifecycle digest chain mismatch")
            previous = value.transition_digest
        return tuple(values)

    def append_lifecycle_transition(self, transition: Any) -> ImmutableWriteStatus:
        path = self._lifecycle_path(transition.repository_scope_id, transition.candidate_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_bytes(transition.to_dict())
        with file_lock(path.with_suffix(path.suffix + ".lock")):
            existing = self.read_lifecycle(
                transition.repository_scope_id, transition.candidate_id
            )
            for item in existing:
                if item.transition_id == transition.transition_id:
                    return (
                        ImmutableWriteStatus.UNCHANGED
                        if item == transition
                        else ImmutableWriteStatus.CONFLICT
                    )
            previous = existing[-1].transition_digest if existing else None
            if (
                transition.sequence != len(existing) + 1
                or transition.previous_transition_digest != previous
            ):
                return ImmutableWriteStatus.CONFLICT
            created = not path.exists()
            try:
                with path.open("ab") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if created and os.name == "posix":
                    directory_fd = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except OSError as exc:
                raise KnowledgeStoreError("cannot append lifecycle transition") from exc
        return ImmutableWriteStatus.CREATED

    def write_materialized_bytes(
        self, relative_parts: tuple[str, ...], data: bytes
    ) -> tuple[ImmutableWriteStatus, Path]:
        if not relative_parts:
            raise ValueError("materialized path is empty")
        safe_parts = tuple(_safe_id(item) for item in relative_parts)
        path = self.materialized.joinpath(*safe_parts)
        resolved_root = self.materialized.resolve(strict=False)
        if not path.resolve(strict=False).is_relative_to(resolved_root):
            raise ValueError("materialized path escapes store root")
        return _atomic_bytes_create_or_compare(path, data), path

    def write_materialization_result(self, result: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.materialization_results, result.materialization_id),
            result.to_dict(),
        )

    def read_materialization_result(self, materialization_id: str) -> Any | None:
        path = self._path(self.materialization_results, materialization_id)
        if not path.exists():
            return None
        from .materialize import KnowledgeMaterializationResult

        value = _load(path, KnowledgeMaterializationResult.from_dict)
        if value.materialization_id != materialization_id:
            raise KnowledgeStoreError("materialization result path binding mismatch")
        return value

    def list_materialization_results(self, candidate_id: str | None = None) -> tuple[Any, ...]:
        if not self.materialization_results.exists():
            return ()
        from .materialize import KnowledgeMaterializationResult

        values = []
        for path in sorted(self.materialization_results.glob("*.json")):
            value = _load(path, KnowledgeMaterializationResult.from_dict)
            if path.stem != value.materialization_id:
                raise KnowledgeStoreError("materialization result path binding mismatch")
            if candidate_id is None or value.candidate_id == candidate_id:
                values.append(value)
        return tuple(values)

    def write_applicability(self, result: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.applicability_results, result.applicability_id), result.to_dict()
        )

    def read_applicability(self, applicability_id: str) -> Any | None:
        path = self._path(self.applicability_results, applicability_id)
        if not path.exists():
            return None
        from .applicability import KnowledgeApplicabilityResult

        value = _load(path, KnowledgeApplicabilityResult.from_dict)
        if value.applicability_id != applicability_id:
            raise KnowledgeStoreError("applicability result path binding mismatch")
        return value

    def write_retrieval(self, receipt: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.retrievals, receipt.retrieval_id), receipt.to_dict()
        )

    def read_retrieval(self, retrieval_id: str) -> Any | None:
        path = self._path(self.retrievals, retrieval_id)
        if not path.exists():
            return None
        from .retrieval import KnowledgeRetrievalReceipt

        value = _load(path, KnowledgeRetrievalReceipt.from_dict)
        if value.retrieval_id != retrieval_id:
            raise KnowledgeStoreError("retrieval receipt path binding mismatch")
        return value

    def write_usage(self, receipt: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(self._path(self.usages, receipt.usage_id), receipt.to_dict())

    def read_usage(self, usage_id: str) -> Any | None:
        path = self._path(self.usages, usage_id)
        if not path.exists():
            return None
        from .usage import KnowledgeUsageReceipt

        value = _load(path, KnowledgeUsageReceipt.from_dict)
        if value.usage_id != usage_id:
            raise KnowledgeStoreError("usage receipt path binding mismatch")
        return value

    def list_usages(self, *, turn_id: str | None = None) -> tuple[Any, ...]:
        if not self.usages.exists():
            return ()
        from .usage import KnowledgeUsageReceipt

        values = []
        for path in sorted(self.usages.glob("*.json"), key=lambda item: item.name):
            value = _load(path, KnowledgeUsageReceipt.from_dict)
            if value.usage_id != path.stem:
                raise KnowledgeStoreError("usage receipt path binding mismatch")
            if turn_id is None or value.turn_id == turn_id:
                values.append(value)
        return tuple(values)

    def write_outcome_association(self, receipt: Any) -> ImmutableWriteStatus:
        return _atomic_create_or_compare(
            self._path(self.outcome_associations, receipt.association_id), receipt.to_dict()
        )

    def read_outcome_association(self, association_id: str) -> Any | None:
        path = self._path(self.outcome_associations, association_id)
        if not path.exists():
            return None
        from .usage import KnowledgeUsageOutcomeAssociation

        value = _load(path, KnowledgeUsageOutcomeAssociation.from_dict)
        if value.association_id != association_id:
            raise KnowledgeStoreError("outcome association path binding mismatch")
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
