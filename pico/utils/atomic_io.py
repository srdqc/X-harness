"""Crash-safe JSONL file primitives: locked append, replace, and delete.

The helpers serialize cross-process mutations with an advisory lock on a
sidecar lock kept in a hidden ``.lock/`` subdir of the target's own parent
(auto-released on process death, so no stale-lock cleanup is needed). The
lock is cross-platform (``portalocker``: POSIX ``fcntl`` + Windows
``LockFileEx``), so concurrent writers are serialized on Windows too.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pico.utils.portable_lock import file_lock


class StorageCorruptionError(ValueError):
    """Persistent generation metadata is missing or invalid."""


def _epoch_path(path: Path) -> Path:
    return path.parent / ".generation" / f"{path.name}.epoch"


def _epoch_known_path(path: Path) -> Path:
    return path.parent / ".generation" / f"{path.name}.known"


def read_epoch(path: Path) -> int:
    """Read the deletion epoch without creating lock or metadata files."""
    epoch_path = _epoch_path(path)
    try:
        raw = epoch_path.read_text(encoding="ascii")
    except FileNotFoundError:
        if _epoch_known_path(path).exists():
            raise StorageCorruptionError(f"missing deletion epoch: {epoch_path}")
        return 0
    except UnicodeError as exc:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}") from exc

    try:
        epoch = int(raw.strip())
    except ValueError as exc:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}") from exc
    if epoch < 0:
        raise StorageCorruptionError(f"invalid deletion epoch: {epoch_path}")
    return epoch


def epoch_is_known(path: Path) -> bool:
    """Return whether this path has durable generation metadata."""
    return _epoch_known_path(path).exists() or _epoch_path(path).exists()


def read_utf8_with_incomplete_tail(path: Path) -> str:
    """Decode UTF-8 while preserving an incomplete final code point as junk."""
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(payload):
            raise
        return payload[: exc.start].decode("utf-8") + "\ufffd"


def _mark_epoch_known(path: Path) -> None:
    known_path = _epoch_known_path(path)
    if known_path.exists():
        return
    known_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = known_path.with_name(known_path.name + ".tmp")
    with open(tmp_path, "w", encoding="ascii") as f:
        f.write("1")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, known_path)


def _write_epoch(path: Path, epoch: int) -> None:
    epoch_path = _epoch_path(path)
    _mark_epoch_known(path)
    tmp_path = epoch_path.with_name(epoch_path.name + ".tmp")
    with open(tmp_path, "w", encoding="ascii") as f:
        f.write(str(epoch))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, epoch_path)


def _ensure_epoch(path: Path, epoch: int) -> None:
    if not _epoch_path(path).exists():
        _write_epoch(path, epoch)


def _restore_epoch(
    path: Path,
    epoch: int,
    *,
    epoch_existed: bool,
    known_existed: bool,
    epoch_dir_existed: bool,
) -> None:
    epoch_path = _epoch_path(path)
    if epoch_existed:
        _write_epoch(path, epoch)
    else:
        epoch_path.unlink(missing_ok=True)
    if not known_existed:
        _epoch_known_path(path).unlink(missing_ok=True)
    if not epoch_dir_existed:
        try:
            epoch_path.parent.rmdir()
        except OSError:
            pass


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".lock" / (path.name + ".lock")
    with file_lock(lock_path):
        yield


def _check_epoch(
    path: Path,
    *,
    expected_epoch: int | None,
    operation: str,
) -> int:
    current = read_epoch(path)
    if not path.exists() and epoch_is_known(path) and current == 0:
        raise StorageCorruptionError(f"known generation is missing primary file: {path}")
    if expected_epoch is not None and current != expected_epoch:
        raise FileNotFoundError(f"file was deleted or replaced before {operation}: {path}")
    return current


def locked_read(path: Path) -> tuple[str | None, int, bool]:
    """Read bytes and generation state under the mutation lock."""
    # 写入方会先发布已知标记，再发布主文件，因此主文件缺失代表合法的
    # epoch-zero 快照，无需实际创建锁。
    if not path.exists() and not epoch_is_known(path):
        return None, 0, False
    with _locked(path):
        raw = read_utf8_with_incomplete_tail(path) if path.exists() else None
        return raw, read_epoch(path), epoch_is_known(path)


def locked_append(
    path: Path,
    lines: list[str],
    *,
    expected_epoch: int | None = None,
    require_existing: bool = False,
    validate_existing: Callable[[str], None] | None = None,
) -> int:
    """Append one block after optional in-lock validation of existing bytes."""
    if not lines:
        return read_epoch(path)
    with _locked(path):
        exists = path.exists()
        if require_existing and not exists:
            raise FileNotFoundError(f"file was deleted before append: {path}")
        epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="append",
        )
        if exists and validate_existing is not None:
            validate_existing(read_utf8_with_incomplete_tail(path))
        _ensure_epoch(path, epoch)
        with open(path, "a+b") as f:
            payload = "".join(line + "\n" for line in lines).encode("utf-8")
            # 写入方崩溃后可能留下没有换行符的残缺行；从新行开始写入，
            # 避免两条记录粘连。
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    payload = b"\n" + payload
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        return epoch


def atomic_replace(
    path: Path,
    data: str,
    *,
    expected_epoch: int | None = None,
    expected_exists: bool | None = None,
    require_existing: bool = False,
    increment_epoch: bool = False,
    validate_existing: Callable[[str], None] | None = None,
) -> int:
    """Replace ``path`` atomically, optionally refusing to recreate it."""
    with _locked(path):
        exists = path.exists()
        if expected_exists is not None and exists != expected_exists:
            raise FileNotFoundError(f"file existence changed before replace: {path}")
        if require_existing and not exists:
            raise FileNotFoundError(f"file was deleted before replace: {path}")
        epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="replace",
        )
        if exists and validate_existing is not None:
            validate_existing(read_utf8_with_incomplete_tail(path))
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        epoch_path = _epoch_path(path)
        epoch_existed = epoch_path.exists()
        known_existed = _epoch_known_path(path).exists()
        epoch_dir_existed = epoch_path.parent.exists()
        next_epoch = epoch + 1 if increment_epoch else epoch
        _write_epoch(path, next_epoch)
        try:
            os.replace(tmp_path, path)
        except OSError:
            _restore_epoch(
                path,
                epoch,
                epoch_existed=epoch_existed,
                known_existed=known_existed,
                epoch_dir_existed=epoch_dir_existed,
            )
            raise
        return next_epoch


def locked_delete(
    path: Path,
    *,
    expected_epoch: int | None = None,
    expected_exists: bool | None = None,
    fence_missing: bool = False,
    increment_epoch: bool = False,
) -> bool:
    """Delete ``path`` while holding the same lock used by writes."""
    with _locked(path):
        exists = path.exists()
        if expected_exists is not None and exists != expected_exists:
            raise FileNotFoundError(f"file existence changed before delete: {path}")
        prior_epoch = _check_epoch(
            path,
            expected_epoch=expected_epoch,
            operation="delete",
        )
        if not exists and not fence_missing:
            return False

        epoch_path = _epoch_path(path)
        epoch_existed = epoch_path.exists()
        known_existed = _epoch_known_path(path).exists()
        epoch_dir_existed = epoch_path.parent.exists()
        if increment_epoch:
            _write_epoch(path, prior_epoch + 1)
        if not exists:
            return False

        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            if increment_epoch:
                _restore_epoch(
                    path,
                    prior_epoch,
                    epoch_existed=epoch_existed,
                    known_existed=known_existed,
                    epoch_dir_existed=epoch_dir_existed,
                )
            raise
        return True
