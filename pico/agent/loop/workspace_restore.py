"""Isolated materialization of validated shadow-Git checkpoint trees."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tarfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

from pico.agent.loop.checkpoint import CheckpointService
from pico.product import get_project_state_dir

_RESTORE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class WorkspaceRestoreStatus(str, Enum):
    RESTORED = "restored"
    VALIDATION_FAILED = "validation_failed"
    UNSUPPORTED = "unsupported"
    MATERIALIZATION_FAILED = "materialization_failed"


class WorkspaceCleanupState(str, Enum):
    ACTIVE = "active"
    NOT_NEEDED = "not_needed"
    CLEANED = "cleaned"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkspaceRestoreResult:
    status: WorkspaceRestoreStatus
    record_id: str
    checkpoint_id: str | None = None
    workspace_id: str | None = None
    workspace_path: Path | None = None
    reason: str | None = None
    cleanup_state: WorkspaceCleanupState = WorkspaceCleanupState.NOT_NEEDED

    @property
    def usable(self) -> bool:
        return self.status is WorkspaceRestoreStatus.RESTORED and self.workspace_path is not None


class _UnsafeSnapshotError(ValueError):
    pass


class _UnsupportedSnapshotError(RuntimeError):
    pass


class WorkspaceRestoreService:
    """Materialize checkpoint trees into owned, isolated Runtime directories."""

    def __init__(
        self,
        checkpoints: CheckpointService,
        *,
        restore_root: Path | None = None,
        restore_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._checkpoints = checkpoints
        self._source_workspace = checkpoints.workspace_path
        configured_root = restore_root or (
            get_project_state_dir(self._source_workspace) / "restored-workspaces"
        )
        self._restore_root = Path(configured_root).expanduser().resolve()
        if self._restore_root == self._source_workspace or self._restore_root.is_relative_to(
            self._source_workspace
        ):
            raise ValueError("restore_root must be outside the source Workspace")
        self._restore_id_factory = restore_id_factory or (lambda: uuid.uuid4().hex)

    def _allocate_destination(self) -> tuple[str, Path]:
        restore_id = self._restore_id_factory()
        if (
            not isinstance(restore_id, str)
            or not _RESTORE_ID_RE.fullmatch(restore_id)
            or PureWindowsPath(restore_id).is_reserved()
        ):
            raise ValueError("restore_id_factory must return a safe local identifier")
        destination = (self._restore_root / restore_id).resolve()
        if destination.parent != self._restore_root:
            raise ValueError("restore destination escaped its managed root")
        return restore_id, destination

    @staticmethod
    def _safe_member_parts(name: str) -> tuple[str, ...]:
        if not name or "\x00" in name or "\\" in name:
            raise _UnsafeSnapshotError("unsafe_archive_path")
        path = PurePosixPath(name)
        windows_path = PureWindowsPath(name)
        if (
            path.is_absolute()
            or windows_path.drive
            or windows_path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise _UnsafeSnapshotError("unsafe_archive_path")
        if os.name == "nt" and any(PureWindowsPath(part).is_reserved() for part in path.parts):
            raise _UnsupportedSnapshotError("unsupported_windows_path")
        return tuple(path.parts)

    @staticmethod
    def _normalized_link_target(parent: tuple[str, ...], target: str) -> tuple[str, ...]:
        if not target or "\x00" in target or "\\" in target:
            raise _UnsafeSnapshotError("unsafe_symlink_target")
        target_path = PurePosixPath(target)
        windows_target = PureWindowsPath(target)
        if target_path.is_absolute() or windows_target.drive or windows_target.is_absolute():
            raise _UnsafeSnapshotError("unsafe_symlink_target")
        normalized = list(parent)
        for part in target_path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not normalized:
                    raise _UnsafeSnapshotError("unsafe_symlink_target")
                normalized.pop()
            else:
                normalized.append(part)
        return tuple(normalized)

    @classmethod
    def _validate_members(
        cls,
        members: list[tarfile.TarInfo],
    ) -> dict[str, tuple[str, ...]]:
        parts_by_name: dict[str, tuple[str, ...]] = {}
        type_by_parts: dict[tuple[str, ...], str] = {}
        symlink_targets: dict[tuple[str, ...], tuple[str, ...]] = {}
        for member in members:
            parts = cls._safe_member_parts(member.name)
            if parts in type_by_parts:
                raise _UnsafeSnapshotError("duplicate_archive_path")
            if member.isdir():
                kind = "directory"
            elif member.isreg():
                kind = "file"
                if os.name == "nt" and member.mode & 0o111:
                    raise _UnsupportedSnapshotError("unsupported_executable_mode")
            elif member.issym():
                kind = "symlink"
                symlink_targets[parts] = cls._normalized_link_target(parts[:-1], member.linkname)
            else:
                raise _UnsupportedSnapshotError("unsupported_archive_entry")
            type_by_parts[parts] = kind
            parts_by_name[member.name] = parts

        for parts in type_by_parts:
            for length in range(1, len(parts)):
                ancestor_kind = type_by_parts.get(parts[:length])
                if ancestor_kind is not None and ancestor_kind != "directory":
                    raise _UnsafeSnapshotError("archive_path_has_non_directory_parent")
        symlink_paths = set(symlink_targets)
        for target in symlink_targets.values():
            for length in range(1, len(target) + 1):
                if target[:length] in symlink_paths:
                    raise _UnsupportedSnapshotError("symlink_chain_not_supported")
        return parts_by_name

    @classmethod
    def _materialize_archive(cls, archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            parts_by_name = cls._validate_members(members)
            directories: list[tuple[Path, int]] = []
            symlinks: list[tuple[tarfile.TarInfo, Path, bool]] = []
            member_kinds = {
                parts_by_name[member.name]: "directory" if member.isdir() else "other"
                for member in members
            }
            directory_members = sorted(
                (member for member in members if member.isdir()),
                key=lambda member: len(parts_by_name[member.name]),
            )
            for member in directory_members:
                target = destination.joinpath(*parts_by_name[member.name])
                target.mkdir(parents=True, exist_ok=False)
                directories.append((target, member.mode & 0o777))

            for member in members:
                parts = parts_by_name[member.name]
                target = destination.joinpath(*parts)
                if member.isdir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isreg():
                    source = archive.extractfile(member)
                    if source is None:
                        raise OSError("archive member content is unavailable")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if os.name != "nt":
                        target.chmod(member.mode & 0o777)
                    continue
                normalized_target = cls._normalized_link_target(parts[:-1], member.linkname)
                symlinks.append(
                    (
                        member,
                        target,
                        member_kinds.get(normalized_target) == "directory",
                    )
                )
            for member, target, target_is_directory in symlinks:
                try:
                    target.symlink_to(member.linkname, target_is_directory=target_is_directory)
                except (NotImplementedError, OSError) as exc:
                    raise _UnsupportedSnapshotError("symlink_creation_unsupported") from exc
            if os.name != "nt":
                for target, mode in sorted(directories, key=lambda item: len(item[0].parts), reverse=True):
                    target.chmod(mode)

    @staticmethod
    def _remove_tree(path: Path) -> bool:
        def make_writable(function, name, _excinfo) -> None:
            target = Path(name)
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
            function(name)

        try:
            shutil.rmtree(path, onexc=make_writable)
        except OSError:
            return False
        return not path.exists()

    async def restore(
        self,
        record_id: str,
        *,
        expected_workspace: Path | None = None,
        expected_session_id: str | None = None,
        expected_boundary_id: str | None = None,
        expected_turn_id: str | None = None,
    ) -> WorkspaceRestoreResult:
        validation = await self._checkpoints.validate(
            record_id,
            expected_workspace=expected_workspace,
            expected_session_id=expected_session_id,
            expected_boundary_id=expected_boundary_id,
            expected_turn_id=expected_turn_id,
        )
        if not validation.usable or validation.record is None:
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.VALIDATION_FAILED,
                record_id,
                reason=validation.reason or "checkpoint_validation_failed",
            )
        checkpoint_id = validation.record.checkpoint_id
        if checkpoint_id is None:
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.VALIDATION_FAILED,
                record_id,
                reason="checkpoint_object_missing",
            )
        try:
            restore_id, destination = self._allocate_destination()
        except ValueError as exc:
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason=str(exc),
            )
        archive_path = self._restore_root / f".{restore_id}.tar"
        try:
            self._restore_root.mkdir(parents=True, exist_ok=True)
            destination.mkdir()
        except FileExistsError:
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason="restore_destination_exists",
            )
        except OSError as exc:
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason=str(exc),
            )
        if archive_path.exists() or archive_path.is_symlink():
            cleaned = self._remove_tree(destination)
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason="restore_staging_path_exists",
                cleanup_state=(
                    WorkspaceCleanupState.CLEANED if cleaned else WorkspaceCleanupState.FAILED
                ),
            )
        try:
            ok, error = await self._checkpoints.export_archive(checkpoint_id, archive_path)
            if not ok:
                raise OSError(error or "git archive failed")
            self._materialize_archive(archive_path, destination)
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.RESTORED,
                record_id,
                checkpoint_id=checkpoint_id,
                workspace_id=restore_id,
                workspace_path=destination,
                cleanup_state=WorkspaceCleanupState.ACTIVE,
            )
        except _UnsafeSnapshotError as exc:
            cleanup_state = (
                WorkspaceCleanupState.CLEANED
                if self._remove_tree(destination)
                else WorkspaceCleanupState.FAILED
            )
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.VALIDATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason=str(exc),
                cleanup_state=cleanup_state,
            )
        except _UnsupportedSnapshotError as exc:
            cleanup_state = (
                WorkspaceCleanupState.CLEANED
                if self._remove_tree(destination)
                else WorkspaceCleanupState.FAILED
            )
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.UNSUPPORTED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason=str(exc),
                cleanup_state=cleanup_state,
            )
        except (OSError, tarfile.TarError) as exc:
            cleanup_state = (
                WorkspaceCleanupState.CLEANED
                if self._remove_tree(destination)
                else WorkspaceCleanupState.FAILED
            )
            return WorkspaceRestoreResult(
                WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
                record_id,
                checkpoint_id=checkpoint_id,
                reason=str(exc),
                cleanup_state=cleanup_state,
            )
        finally:
            try:
                archive_path.unlink(missing_ok=True)
            except OSError:
                pass

    def cleanup(self, result: WorkspaceRestoreResult) -> bool:
        """Remove one successful restore only when its identity belongs to this service."""

        if not result.usable or result.workspace_id is None or result.workspace_path is None:
            return False
        expected = (self._restore_root / result.workspace_id).resolve()
        if expected != result.workspace_path.resolve() or expected.parent != self._restore_root:
            return False
        return self._remove_tree(expected)


__all__ = [
    "WorkspaceCleanupState",
    "WorkspaceRestoreResult",
    "WorkspaceRestoreService",
    "WorkspaceRestoreStatus",
]
