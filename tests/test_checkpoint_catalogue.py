"""Durable checkpoint result, catalogue, correlation, and validation tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pico.agent.loop.checkpoint import CheckpointService, CheckpointStatus


def _service(workspace: Path, *record_ids: str) -> CheckpointService:
    ids = iter(record_ids)
    return CheckpointService(
        workspace,
        record_id_factory=lambda: next(ids),
        now_fn=lambda: datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc),
    )


async def test_created_unchanged_and_failed_are_distinct_durable_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(workspace, "created-record", "unchanged-record", "failed-record")
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")

    created = await service.commit_turn("created")
    unchanged = await service.commit_turn("unchanged")

    async def fail_git(*_args: str) -> tuple[int, str, str]:
        return 1, "", "injected failure"

    monkeypatch.setattr(service, "_git", fail_git)
    failed = await service.commit_turn("failed")

    assert created.status is CheckpointStatus.CREATED
    assert created.checkpoint_id is not None
    assert len(created.checkpoint_id) in {40, 64}
    assert unchanged.status is CheckpointStatus.UNCHANGED
    assert unchanged.checkpoint_id is None
    assert failed.status is CheckpointStatus.FAILED
    assert failed.checkpoint_id is None
    assert all(
        record.session_id is None and record.boundary_id is None and record.turn_id is None
        for record in service.list_records()
    )
    assert [record.status for record in service.list_records()] == [
        CheckpointStatus.CREATED,
        CheckpointStatus.UNCHANGED,
        CheckpointStatus.FAILED,
    ]

    unchanged_validation = await service.validate("unchanged-record")
    failed_validation = await service.validate("failed-record")
    assert unchanged_validation.reason == "checkpoint_unchanged"
    assert failed_validation.reason == "checkpoint_failed"
    assert not unchanged_validation.usable
    assert not failed_validation.usable


async def test_created_record_and_full_object_correlation_survive_reload(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(workspace, "record-1")
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")

    result = await service.commit_turn(
        "turn",
        session_id="tui:branch-1",
        boundary_id="boundary-1",
        turn_id="turn-1",
    )

    reloaded = CheckpointService(workspace)
    record = reloaded.get_record("record-1")
    assert record is not None
    assert record.status is CheckpointStatus.CREATED
    assert record.checkpoint_id == result.checkpoint_id
    assert len(record.checkpoint_id or "") in {40, 64}
    assert record.session_id == "tui:branch-1"
    assert record.boundary_id == "boundary-1"
    assert record.turn_id == "turn-1"
    assert not ({"messages", "history", "conversation"} & set(record.to_dict()))
    validation = await reloaded.validate(
        "record-1",
        expected_session_id="tui:branch-1",
        expected_boundary_id="boundary-1",
        expected_turn_id="turn-1",
    )
    assert validation.usable
    assert validation.workspace_drifted is False
    assert validation.tree_id is not None


async def test_correlation_update_is_durable_and_conflicts_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(workspace, "record-1")
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")
    result = await service.commit_turn("turn", session_id="tui:branch-1", turn_id="turn-1")

    updated = service.correlate(
        result.record_id or "",
        session_id="tui:branch-1",
        boundary_id="boundary-1",
        turn_id="turn-1",
    )

    assert updated is not None
    assert updated.revision == 2
    assert CheckpointService(workspace).get_record("record-1") == updated
    assert (
        service.correlate(
            "record-1",
            session_id="tui:other-branch",
            boundary_id="boundary-1",
            turn_id="turn-1",
        )
        is None
    )


async def test_validation_rejects_workspace_mismatch_missing_object_and_detects_drift(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(workspace, "drift-record")
    tracked = workspace / "main.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    result = await service.commit_turn("turn")
    assert result.checkpoint_id is not None

    mismatch = await service.validate("drift-record", expected_workspace=tmp_path / "other")
    assert not mismatch.usable
    assert mismatch.reason == "workspace_mismatch"

    tracked.write_text("value = 2\n", encoding="utf-8")
    (workspace / "new.py").write_text("new = True\n", encoding="utf-8")
    drifted = await service.validate("drift-record")
    assert drifted.usable
    assert drifted.workspace_drifted is True
    assert set(drifted.drifted_paths) >= {"main.py", "new.py"}
    assert tracked.read_text(encoding="utf-8") == "value = 2\n"
    assert (workspace / "new.py").is_file()

    object_path = service._git_dir / "objects" / result.checkpoint_id[:2] / result.checkpoint_id[2:]
    assert object_path.is_file()
    object_path.chmod(object_path.stat().st_mode | stat.S_IWRITE)
    object_path.unlink()
    missing = await service.validate("drift-record")
    assert not missing.usable
    assert missing.reason == "checkpoint_object_missing"


@pytest.mark.parametrize("mutation", ["corrupt-json", "unsupported-version"])
async def test_catalogue_corruption_and_unknown_schema_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    workspace = tmp_path / mutation
    workspace.mkdir()
    service = _service(workspace, "record-1")
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")
    await service.commit_turn("turn")
    path = service._catalogue_path
    if mutation == "corrupt-json":
        path.write_text("{not-json}\n", encoding="utf-8")
    else:
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["schema_version"] = 999
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    validation = await service.validate("record-1")

    assert not validation.usable
    assert validation.reason == "catalogue_corrupt"


async def test_checkpoint_catalogue_never_mutates_user_git_history(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "user",
        "GIT_AUTHOR_EMAIL": "user@example.test",
        "GIT_COMMITTER_NAME": "user",
        "GIT_COMMITTER_EMAIL": "user@example.test",
    }
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "base"],
        cwd=workspace,
        check=True,
        env=env,
    )
    before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    service = _service(workspace, "record-1")
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")

    result = await service.commit_turn("turn")

    after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert result.status is CheckpointStatus.CREATED
    assert after == before
