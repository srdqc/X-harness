"""P1A.4 isolated Workspace checkpoint materialization tests."""

from __future__ import annotations

import io
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

from pico.agent.loop.checkpoint import CheckpointService, CheckpointStatus
from pico.agent.loop.workspace_restore import (
    WorkspaceCleanupState,
    WorkspaceRestoreService,
    WorkspaceRestoreStatus,
)
from pico.agent.tools.registry import ToolRegistry


def _checkpoint_service(workspace: Path, *record_ids: str) -> CheckpointService:
    ids = iter(record_ids)
    return CheckpointService(workspace, record_id_factory=lambda: next(ids))


async def test_valid_checkpoint_restores_exact_files_without_touching_live_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "initial", "selected")
    (workspace / "deleted.txt").write_text("old\n", encoding="utf-8")
    await service.commit_turn("initial")
    (workspace / "deleted.txt").unlink()
    (workspace / "src" / "nested").mkdir(parents=True)
    (workspace / "src" / "nested" / "main.py").write_text("selected\n", encoding="utf-8")
    binary = bytes(range(256))
    (workspace / "asset.bin").write_bytes(binary)
    selected = await service.commit_turn("selected")
    assert selected.status is CheckpointStatus.CREATED

    (workspace / "src" / "nested" / "main.py").write_text("live drift\n", encoding="utf-8")
    (workspace / "asset.bin").write_bytes(b"live")
    (workspace / "live-only.txt").write_text("keep live\n", encoding="utf-8")
    live_before = {
        "main": (workspace / "src" / "nested" / "main.py").read_bytes(),
        "binary": (workspace / "asset.bin").read_bytes(),
        "live_only": (workspace / "live-only.txt").read_bytes(),
    }
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "restore-1",
    )

    result = await restorer.restore("selected", expected_workspace=workspace)

    assert result.status is WorkspaceRestoreStatus.RESTORED
    assert result.usable
    assert result.workspace_id == "restore-1"
    assert result.workspace_path == (tmp_path / "restores" / "restore-1").resolve()
    assert result.workspace_path != workspace.resolve()
    assert (result.workspace_path / "src" / "nested" / "main.py").read_text() == "selected\n"
    assert (result.workspace_path / "asset.bin").read_bytes() == binary
    assert not (result.workspace_path / "deleted.txt").exists()
    assert not (result.workspace_path / "live-only.txt").exists()
    assert (workspace / "src" / "nested" / "main.py").read_bytes() == live_before["main"]
    assert (workspace / "asset.bin").read_bytes() == live_before["binary"]
    assert (workspace / "live-only.txt").read_bytes() == live_before["live_only"]


async def test_two_restores_have_independent_mutable_state(tmp_path: Path) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "selected")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    await service.commit_turn("selected")
    restore_ids = iter(("restore-a", "restore-b"))
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: next(restore_ids),
    )

    first = await restorer.restore("selected")
    second = await restorer.restore("selected")
    assert first.usable and second.usable
    assert first.workspace_path != second.workspace_path
    (first.workspace_path / "main.py").write_text("changed first\n", encoding="utf-8")
    assert (second.workspace_path / "main.py").read_text(encoding="utf-8") == "selected\n"
    assert (workspace / "main.py").read_text(encoding="utf-8") == "selected\n"
    assert restorer.cleanup(first)
    assert not first.workspace_path.exists()
    assert second.workspace_path.exists()
    assert (workspace / "main.py").exists()


async def test_default_restore_location_uses_project_state_outside_live_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product_home = tmp_path / "pico-home"
    monkeypatch.setenv("PICO_HOME", str(product_home))
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "selected")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    await service.commit_turn("selected")
    restorer = WorkspaceRestoreService(service, restore_id_factory=lambda: "default-root")

    result = await restorer.restore("selected")

    assert result.usable
    assert result.workspace_path.is_relative_to(product_home / "projects")
    assert not result.workspace_path.is_relative_to(workspace)


async def test_restore_rejects_mismatch_unchanged_failed_and_missing_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "created", "unchanged", "failed")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    created = await service.commit_turn("created")
    unchanged = await service.commit_turn("unchanged")
    assert unchanged.status is CheckpointStatus.UNCHANGED

    async def fail_git(*_args: str) -> tuple[int, str, str]:
        return 1, "", "injected"

    monkeypatch.setattr(service, "_git", fail_git)
    failed = await service.commit_turn("failed")
    assert failed.status is CheckpointStatus.FAILED
    monkeypatch.undo()
    restorer = WorkspaceRestoreService(service, restore_root=tmp_path / "restores")

    mismatch = await restorer.restore("created", expected_workspace=tmp_path / "other")
    unchanged_restore = await restorer.restore("unchanged")
    failed_restore = await restorer.restore("failed")
    assert mismatch.reason == "workspace_mismatch"
    assert unchanged_restore.reason == "checkpoint_unchanged"
    assert failed_restore.reason == "checkpoint_failed"
    assert all(
        result.status is WorkspaceRestoreStatus.VALIDATION_FAILED
        for result in (mismatch, unchanged_restore, failed_restore)
    )

    assert created.checkpoint_id is not None
    object_path = service._git_dir / "objects" / created.checkpoint_id[:2] / created.checkpoint_id[2:]
    object_path.chmod(object_path.stat().st_mode | stat.S_IWRITE)
    object_path.unlink()
    missing = await restorer.restore("created")
    assert missing.status is WorkspaceRestoreStatus.VALIDATION_FAILED
    assert missing.reason == "checkpoint_object_missing"
    assert not (tmp_path / "restores").exists()


async def test_unsafe_archive_path_is_rejected_and_partial_destination_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "selected")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    await service.commit_turn("selected")

    async def malicious_archive(_checkpoint_id: str, destination: Path) -> tuple[bool, str]:
        with tarfile.open(destination, "w") as archive:
            payload = b"escape"
            info = tarfile.TarInfo("../escape.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return True, ""

    monkeypatch.setattr(service, "export_archive", malicious_archive)
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "unsafe",
    )

    result = await restorer.restore("selected")

    assert result.status is WorkspaceRestoreStatus.VALIDATION_FAILED
    assert result.reason == "unsafe_archive_path"
    assert result.cleanup_state is WorkspaceCleanupState.CLEANED
    assert not (tmp_path / "restores" / "unsafe").exists()
    assert not (tmp_path / "restores" / "escape.txt").exists()


async def test_materialization_failure_cleans_partial_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "selected")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    await service.commit_turn("selected")
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "partial",
    )

    def fail_materialization(_archive: Path, destination: Path) -> None:
        (destination / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("injected materialization failure")

    monkeypatch.setattr(restorer, "_materialize_archive", fail_materialization)

    result = await restorer.restore("selected")

    assert result.status is WorkspaceRestoreStatus.MATERIALIZATION_FAILED
    assert result.cleanup_state is WorkspaceCleanupState.CLEANED
    assert result.workspace_path is None
    assert not (tmp_path / "restores" / "partial").exists()


async def test_unsafe_symlink_target_is_rejected_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    service = _checkpoint_service(workspace, "selected")
    (workspace / "main.py").write_text("selected\n", encoding="utf-8")
    await service.commit_turn("selected")

    async def malicious_archive(_checkpoint_id: str, destination: Path) -> tuple[bool, str]:
        with tarfile.open(destination, "w") as archive:
            info = tarfile.TarInfo("unsafe-link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside.txt"
            archive.addfile(info)
        return True, ""

    monkeypatch.setattr(service, "export_archive", malicious_archive)
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "unsafe-link",
    )

    result = await restorer.restore("selected")

    assert result.status is WorkspaceRestoreStatus.VALIDATION_FAILED
    assert result.reason == "unsafe_symlink_target"
    assert result.cleanup_state is WorkspaceCleanupState.CLEANED
    assert not (tmp_path / "outside.txt").exists()


async def test_restore_does_not_mutate_user_git_or_execute_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "user",
        "GIT_AUTHOR_EMAIL": "user@example.test",
        "GIT_COMMITTER_NAME": "user",
        "GIT_COMMITTER_EMAIL": "user@example.test",
    }
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True, env=env)
    (workspace / "staged.txt").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=workspace, check=True)
    service = _checkpoint_service(workspace, "selected")
    await service.commit_turn("selected", session_id="tui:branch", turn_id="turn-1")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    before = {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--porcelain=v1"),
        "stash": git("stash", "list"),
        "worktrees": git("worktree", "list", "--porcelain"),
    }
    tool_calls = 0

    async def forbidden_tool_execute(*_args, **_kwargs):
        nonlocal tool_calls
        tool_calls += 1
        raise AssertionError("restore must not execute Agent Tools")

    monkeypatch.setattr(ToolRegistry, "execute", forbidden_tool_execute)
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "git-safe",
    )

    result = await restorer.restore("selected")

    after = {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--porcelain=v1"),
        "stash": git("stash", "list"),
        "worktrees": git("worktree", "list", "--porcelain"),
    }
    assert result.usable
    assert after == before
    assert tool_calls == 0


@pytest.mark.skipif(os.name == "nt", reason="Windows does not provide POSIX mode/symlink semantics")
async def test_restore_preserves_supported_executable_and_symlink_semantics(tmp_path: Path) -> None:
    workspace = tmp_path / "live"
    workspace.mkdir()
    script = workspace / "script.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    (workspace / "script-link").symlink_to("script.sh")
    service = _checkpoint_service(workspace, "selected")
    await service.commit_turn("selected")
    restorer = WorkspaceRestoreService(
        service,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: "posix",
    )

    result = await restorer.restore("selected")

    assert result.usable
    restored_script = result.workspace_path / "script.sh"
    restored_link = result.workspace_path / "script-link"
    assert restored_script.stat().st_mode & stat.S_IXUSR
    assert restored_link.is_symlink()
    assert os.readlink(restored_link) == "script.sh"
