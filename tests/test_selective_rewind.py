"""P1A.5 selective-rewind coordination and continuation binding tests."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from pico.agent.loop.checkpoint import CheckpointService
from pico.agent.loop.rewind import (
    ContinuationBindings,
    ContinuationTurnRunner,
    RewindMode,
    RewindRequest,
    RewindStatus,
    SelectiveRewindCoordinator,
    WorkspaceLeaseState,
)
from pico.agent.loop.workspace_restore import (
    WorkspaceCleanupState,
    WorkspaceRestoreResult,
    WorkspaceRestoreService,
    WorkspaceRestoreStatus,
)
from pico.session.manager import SessionManager
from pico.spine import ChatType, Origin, OriginPools, Scheduler, Source, TurnRequest, Usage
from pico.spine.runner import TurnOutcome, current_turn_id


@dataclass
class _Environment:
    live: Path
    sessions: SessionManager
    source_key: str
    boundary_id: str
    checkpoint_record_id: str
    checkpoints: CheckpointService
    restorer: WorkspaceRestoreService
    bindings: ContinuationBindings
    coordinator: SelectiveRewindCoordinator


async def _environment(
    tmp_path: Path,
    *,
    checkpoint_session_id: str | None = "cli:source",
    checkpoint_boundary_id: str | None = "boundary-1",
    checkpoint_turn_id: str | None = "turn-1",
    active=None,
) -> _Environment:
    live = tmp_path / "live"
    live.mkdir(parents=True)
    state = tmp_path / "state"
    boundary_ids = iter(("boundary-1", "boundary-2"))
    sessions = SessionManager(state, boundary_id_factory=lambda: next(boundary_ids))
    source = sessions.get_or_create("cli:source")
    source.add_message("user", "selected request")
    source.add_message("assistant", "selected answer")
    boundary = sessions.commit_turn_boundary(source, turn_id="turn-1")

    (live / "project.txt").write_text("selected\n", encoding="utf-8")
    checkpoints = CheckpointService(
        live,
        state=state,
        record_id_factory=lambda: "checkpoint-record-1",
    )
    checkpoint = await checkpoints.commit_turn(
        "selected",
        session_id=checkpoint_session_id,
        boundary_id=checkpoint_boundary_id,
        turn_id=checkpoint_turn_id,
    )
    assert checkpoint.record_id == "checkpoint-record-1"

    source.add_message("user", "later request")
    source.add_message("assistant", "later answer")
    sessions.commit_turn_boundary(source, turn_id="turn-2")
    (live / "project.txt").write_text("live current\n", encoding="utf-8")
    (live / "live-only.txt").write_text("unchanged\n", encoding="utf-8")

    restore_ids = iter(("restore-1", "restore-2", "restore-3"))
    restorer = WorkspaceRestoreService(
        checkpoints,
        restore_root=tmp_path / "restores",
        restore_id_factory=lambda: next(restore_ids),
    )
    bindings = ContinuationBindings(restorer)
    continuation_ids = iter(("continuation-1", "continuation-2", "continuation-3"))
    coordinator = SelectiveRewindCoordinator(
        sessions,
        checkpoints,
        restorer,
        bindings,
        is_session_active=active or (lambda _session_id: False),
        continuation_id_factory=lambda: next(continuation_ids),
    )
    return _Environment(
        live,
        sessions,
        source.key,
        boundary.boundary_id,
        checkpoint.record_id,
        checkpoints,
        restorer,
        bindings,
        coordinator,
    )


def _source_snapshot(env: _Environment) -> tuple[list[dict], list, dict]:
    source = env.sessions.peek(env.source_key)
    assert source is not None
    return list(source.messages), list(source.turn_boundaries), dict(source.metadata)


def _live_snapshot(env: _Environment) -> dict[str, bytes]:
    return {
        path.relative_to(env.live).as_posix(): path.read_bytes()
        for path in env.live.rglob("*")
        if path.is_file()
    }


async def test_conversation_only_forks_boundary_and_reuses_live_workspace(tmp_path: Path) -> None:
    env = await _environment(tmp_path)
    source_before = _source_snapshot(env)
    live_before = _live_snapshot(env)

    result = await env.coordinator.rewind(
        RewindRequest(RewindMode.CONVERSATION, env.source_key, boundary_id=env.boundary_id)
    )

    assert result.ready
    assert result.target is not None
    assert result.target.session_created is True
    assert result.target.workspace_created is False
    assert result.target.session_id != env.source_key
    assert result.target.workspace_path == env.live.resolve()
    child = env.sessions.peek(result.target.session_id)
    assert child is not None
    assert [message["content"] for message in child.messages] == [
        "selected request",
        "selected answer",
    ]
    assert child.metadata["parent_session_id"] == env.source_key
    assert child.metadata["parent_boundary_id"] == env.boundary_id
    assert _source_snapshot(env) == source_before
    assert _live_snapshot(env) == live_before


async def test_workspace_only_reuses_session_and_restores_isolated_workspace(tmp_path: Path) -> None:
    env = await _environment(tmp_path)
    source_before = _source_snapshot(env)
    live_before = _live_snapshot(env)

    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.WORKSPACE,
            env.source_key,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.ready
    assert result.target is not None
    assert result.target.session_id == env.source_key
    assert result.target.session_created is False
    assert result.target.workspace_created is True
    assert result.target.workspace_path != env.live.resolve()
    assert (result.target.workspace_path / "project.txt").read_text(encoding="utf-8") == "selected\n"
    assert not (result.target.workspace_path / "live-only.txt").exists()
    assert _source_snapshot(env) == source_before
    assert _live_snapshot(env) == live_before


async def test_both_requires_correlation_and_publishes_only_complete_target(tmp_path: Path) -> None:
    env = await _environment(tmp_path)
    source_before = _source_snapshot(env)
    live_before = _live_snapshot(env)

    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.BOTH,
            env.source_key,
            boundary_id=env.boundary_id,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.status is RewindStatus.READY
    assert result.target is not None
    assert result.target.session_created and result.target.workspace_created
    assert result.target.session_id != env.source_key
    assert result.target.workspace_lease_state is WorkspaceLeaseState.ACTIVE
    assert result.target.selected_boundary is not None
    assert result.target.selected_boundary.turn_id == "turn-1"
    assert result.target.selected_checkpoint is not None
    assert result.target.selected_checkpoint.turn_id == "turn-1"
    assert env.bindings.resolve(result.target.session_id) == result.target
    assert _source_snapshot(env) == source_before
    assert _live_snapshot(env) == live_before
    with pytest.raises(FrozenInstanceError):
        result.target.session_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("session_id", "boundary_id", "turn_id", "reason"),
    (
        ("cli:other", "boundary-1", "turn-1", "session_mismatch"),
        ("cli:source", "boundary-other", "turn-1", "boundary_mismatch"),
        ("cli:source", "boundary-1", "turn-other", "turn_mismatch"),
        (None, None, None, "checkpoint_correlation_missing"),
    ),
)
async def test_combined_correlation_mismatch_or_missing_fails_closed(
    tmp_path: Path,
    session_id: str | None,
    boundary_id: str | None,
    turn_id: str | None,
    reason: str,
) -> None:
    env = await _environment(
        tmp_path,
        checkpoint_session_id=session_id,
        checkpoint_boundary_id=boundary_id,
        checkpoint_turn_id=turn_id,
    )
    source_before = _source_snapshot(env)
    live_before = _live_snapshot(env)

    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.BOTH,
            env.source_key,
            boundary_id=env.boundary_id,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.status is RewindStatus.VALIDATION_FAILED
    assert result.reason == reason
    assert result.target is None
    assert _source_snapshot(env) == source_before
    assert _live_snapshot(env) == live_before
    assert not (tmp_path / "restores").exists()


async def test_child_is_cleaned_when_workspace_preparation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path)
    source_before = _source_snapshot(env)

    async def fail_restore(*_args, **_kwargs) -> WorkspaceRestoreResult:
        return WorkspaceRestoreResult(
            WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
            env.checkpoint_record_id,
            reason="injected restore failure",
        )

    monkeypatch.setattr(env.restorer, "restore", fail_restore)
    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.BOTH,
            env.source_key,
            boundary_id=env.boundary_id,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.status is RewindStatus.PREPARATION_FAILED
    assert result.target is None
    assert _source_snapshot(env) == source_before
    assert env.sessions.list_sessions() == [
        entry for entry in env.sessions.list_sessions() if entry["key"] == env.source_key
    ]


async def test_workspace_is_cleaned_when_session_preparation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path)
    source_before = _source_snapshot(env)
    monkeypatch.setattr(env.sessions, "fork_at", lambda *_args, **_kwargs: None)

    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.BOTH,
            env.source_key,
            boundary_id=env.boundary_id,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.status is RewindStatus.PREPARATION_FAILED
    assert result.target is None
    assert _source_snapshot(env) == source_before
    assert not (tmp_path / "restores" / "restore-1").exists()


async def test_cleanup_failure_is_reported_not_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = await _environment(tmp_path)
    monkeypatch.setattr(env.sessions, "discard_fork_child", lambda _child: False)

    async def fail_restore(*_args, **_kwargs) -> WorkspaceRestoreResult:
        return WorkspaceRestoreResult(
            WorkspaceRestoreStatus.MATERIALIZATION_FAILED,
            env.checkpoint_record_id,
            reason="injected",
            cleanup_state=WorkspaceCleanupState.FAILED,
        )

    monkeypatch.setattr(env.restorer, "restore", fail_restore)
    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.BOTH,
            env.source_key,
            boundary_id=env.boundary_id,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert result.cleanup_failures == ("session_cleanup_failed", "workspace_cleanup_failed")


async def test_active_source_fails_before_preparation_and_rechecks_before_activation(
    tmp_path: Path,
) -> None:
    calls = 0

    def active(_session_id: str) -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    env = await _environment(tmp_path, active=active)
    result = await env.coordinator.rewind(
        RewindRequest(RewindMode.CONVERSATION, env.source_key, boundary_id=env.boundary_id)
    )
    assert result.status is RewindStatus.VALIDATION_FAILED
    assert result.reason == "source_session_became_active"
    assert env.sessions.list_sessions() == [
        entry for entry in env.sessions.list_sessions() if entry["key"] == env.source_key
    ]

    always_active = await _environment(tmp_path / "other", active=lambda _session_id: True)
    immediate = await always_active.coordinator.rewind(
        RewindRequest(
            RewindMode.CONVERSATION,
            always_active.source_key,
            boundary_id=always_active.boundary_id,
        )
    )
    assert immediate.reason == "source_session_active"


class _ProbeRunner:
    def __init__(self, workspace: Path, calls: list[tuple[Path, str | None]]) -> None:
        self.workspace = workspace
        self.calls = calls

    async def run(self, _req, _emit, _drain) -> TurnOutcome:
        self.calls.append((self.workspace, current_turn_id()))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


async def _sink(_event) -> None:
    return None


async def test_rewind_creates_no_turn_and_next_message_uses_scheduler_with_fresh_id(
    tmp_path: Path,
) -> None:
    env = await _environment(tmp_path)
    calls: list[tuple[Path, str | None]] = []
    default = _ProbeRunner(env.live, calls)
    routed = ContinuationTurnRunner(
        default,
        env.bindings,
        lambda target: _ProbeRunner(target.workspace_path, calls),
    )
    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.WORKSPACE,
            env.source_key,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )
    assert calls == []
    assert result.target is not None

    scheduler = Scheduler(
        routed,
        OriginPools(user=1, system=1),
        _sink,
        turn_id_factory=lambda: "fresh-turn",
    )
    request = TurnRequest(
        origin=Origin.USER,
        source=Source("cli", "source", "user", ChatType.DM),
        text="continue",
        conversation=result.target.session_id,
    )
    try:
        handle = scheduler.submit(request)
        await handle.result()
    finally:
        await scheduler.shutdown(0)

    assert handle.turn_id == "fresh-turn"
    assert calls == [(result.target.workspace_path, "fresh-turn")]


async def test_multiple_restored_bindings_are_isolated_and_do_not_change_cwd(tmp_path: Path) -> None:
    env = await _environment(tmp_path)
    second = env.sessions.get_or_create("cli:second")
    second.add_message("user", "current conversation")
    env.sessions.save(second)
    cwd_before = Path.cwd()

    first_result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.WORKSPACE,
            env.source_key,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )
    second_result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.WORKSPACE,
            second.key,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )

    assert first_result.target is not None and second_result.target is not None
    assert first_result.target.workspace_path != second_result.target.workspace_path
    (first_result.target.workspace_path / "project.txt").write_text("first only\n", encoding="utf-8")
    assert (second_result.target.workspace_path / "project.txt").read_text() == "selected\n"
    assert (env.live / "project.txt").read_text() == "live current\n"
    assert Path.cwd() == cwd_before
    assert os.getcwd() == str(cwd_before)


async def test_active_workspace_lease_cannot_be_cleaned_until_runner_releases_it(
    tmp_path: Path,
) -> None:
    env = await _environment(tmp_path)
    result = await env.coordinator.rewind(
        RewindRequest(
            RewindMode.WORKSPACE,
            env.source_key,
            checkpoint_record_id=env.checkpoint_record_id,
        )
    )
    assert result.target is not None
    acquired = env.bindings.acquire(result.target.session_id)
    assert acquired == result.target

    refused = env.bindings.release(result.target.session_id)
    assert refused is not None
    assert refused.state is WorkspaceLeaseState.ACTIVE
    assert refused.reason == "continuation_in_use"
    assert result.target.workspace_path.exists()

    env.bindings.finish(result.target.continuation_id)
    released = env.bindings.release(result.target.session_id)
    assert released is not None
    assert released.state is WorkspaceLeaseState.RELEASED
    assert released.workspace_cleaned is True
    assert not result.target.workspace_path.exists()
