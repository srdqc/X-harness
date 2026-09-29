"""P1A.6 CLI/TUI adapters and normal-runner continuation binding tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from pico.agent.loop.rewind import (
    ContinuationRelease,
    ContinuationTarget,
    RewindMode,
    RewindResult,
    RewindSelectionOptions,
    RewindStatus,
    WorkspaceLeaseState,
)
from pico.cli._repl_slash import handle_repl_rewind
from pico.cli._repl_spine import build_repl
from pico.session.manager import SessionTurnBoundary
from pico.spine import ChatType, Origin, Source, TurnRequest, Usage
from pico.spine.runner import TurnOutcome
from pico.tui_rpc.methods.session import session_rewind, session_rewind_options


class _Coordinator:
    def __init__(self, result: RewindResult) -> None:
        self.result = result
        self.requests = []

    async def rewind(self, request):
        self.requests.append(request)
        return self.result

    async def selection_options(self, session_id: str) -> RewindSelectionOptions:
        return RewindSelectionOptions(
            session_id,
            (SessionTurnBoundary("boundary-1", 2, "0" * 64, "old-turn"),),
            (),
        )

    def release(self, session_id: str) -> ContinuationRelease:
        return ContinuationRelease(
            "continuation-1",
            session_id,
            WorkspaceLeaseState.RELEASED,
            True,
        )


def _ready(mode: RewindMode, tmp_path: Path) -> RewindResult:
    session_created = mode in {RewindMode.CONVERSATION, RewindMode.BOTH}
    workspace_created = mode in {RewindMode.WORKSPACE, RewindMode.BOTH}
    target = ContinuationTarget(
        "continuation-1",
        "cli:child" if session_created else "cli:source",
        "restore-1" if workspace_created else str(tmp_path),
        tmp_path / "restored" if workspace_created else tmp_path,
        "cli:source",
        SessionTurnBoundary("boundary-1", 2, "0" * 64, "old-turn")
        if session_created
        else None,
        None,
        session_created,
        workspace_created,
        WorkspaceLeaseState.ACTIVE,
    )
    return RewindResult(RewindStatus.READY, mode, target=target)


@pytest.mark.parametrize(
    ("mode", "command"),
    (
        (RewindMode.CONVERSATION, "/rewind conversation boundary-1"),
        (RewindMode.WORKSPACE, "/rewind workspace checkpoint-record-1"),
        (RewindMode.BOTH, "/rewind both boundary-1 checkpoint-record-1"),
    ),
)
async def test_cli_three_modes_delegate_to_runtime_and_present_domain_state(
    tmp_path: Path,
    mode: RewindMode,
    command: str,
) -> None:
    coordinator = _Coordinator(_ready(mode, tmp_path))
    console = Console(record=True, width=200)
    targets = []

    handled = await handle_repl_rewind(
        command,
        console=console,
        coordinator=coordinator,
        session_id="cli:source",
        on_ready=targets.append,
    )

    assert handled is True
    assert coordinator.requests[0].mode is mode
    assert targets == [coordinator.result.target]
    output = console.export_text()
    if mode is RewindMode.CONVERSATION:
        assert "workspace=" in output and "(unchanged)" in output
    if mode is RewindMode.WORKSPACE:
        assert "session=cli:source (history unchanged)" in output
    assert "external side effects were not rewound" in output


async def test_cli_surfaces_active_conflict_and_cleanup_failure_without_retry(tmp_path: Path) -> None:
    result = RewindResult(
        RewindStatus.VALIDATION_FAILED,
        RewindMode.BOTH,
        reason="source_session_active",
        cleanup_failures=("workspace_cleanup_failed",),
    )
    coordinator = _Coordinator(result)
    console = Console(record=True, width=200)

    await handle_repl_rewind(
        "/rewind both boundary-1 checkpoint-record-1",
        console=console,
        coordinator=coordinator,
        session_id="cli:source",
        on_ready=lambda _target: pytest.fail("failed rewind must not activate"),
    )

    assert len(coordinator.requests) == 1
    output = console.export_text()
    assert "source_session_active" in output
    assert "workspace_cleanup_failed" in output


async def test_cli_explicit_release_surfaces_workspace_cleanup(tmp_path: Path) -> None:
    coordinator = _Coordinator(_ready(RewindMode.WORKSPACE, tmp_path))
    console = Console(record=True, width=200)
    await handle_repl_rewind(
        "/rewind release",
        console=console,
        coordinator=coordinator,
        session_id="cli:source",
        on_ready=lambda _target: None,
    )
    assert "workspace_cleaned=True" in console.export_text()


@pytest.mark.parametrize("mode", tuple(RewindMode))
async def test_tui_rpc_exposes_same_three_typed_modes(tmp_path: Path, mode: RewindMode) -> None:
    coordinator = _Coordinator(_ready(mode, tmp_path))
    params = {"session_id": "cli:source", "mode": mode.value}
    if mode in {RewindMode.CONVERSATION, RewindMode.BOTH}:
        params["boundary_id"] = "boundary-1"
    if mode in {RewindMode.WORKSPACE, RewindMode.BOTH}:
        params["checkpoint_record_id"] = "checkpoint-record-1"

    response = await session_rewind(params, coordinator_factory=lambda: _async_value(coordinator))

    assert response["status"] == "ready"
    assert response["mode"] == mode.value
    assert response["session_created"] is (mode in {RewindMode.CONVERSATION, RewindMode.BOTH})
    assert response["workspace_created"] is (mode in {RewindMode.WORKSPACE, RewindMode.BOTH})
    assert coordinator.requests[0].mode is mode


async def _async_value(value):
    return value


async def test_tui_options_expose_durable_boundary_identity(tmp_path: Path) -> None:
    coordinator = _Coordinator(_ready(RewindMode.CONVERSATION, tmp_path))
    response = await session_rewind_options(
        {"session_id": "cli:source"},
        coordinator_factory=lambda: _async_value(coordinator),
    )
    assert response["boundaries"] == [
        {"boundary_id": "boundary-1", "message_count": 2, "turn_id": "old-turn"}
    ]


class _Loop:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    async def run_turn(self, req, emit, drain, *, stream):
        self.calls.append((req.conversation, req.text, stream))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class _Bindings:
    def __init__(self, target: ContinuationTarget) -> None:
        self.target = target
        self.finished = []

    def acquire(self, session_id: str):
        return self.target if session_id == self.target.session_id else None

    def finish(self, continuation_id: str) -> None:
        self.finished.append(continuation_id)


@pytest.mark.parametrize("mode", tuple(RewindMode))
async def test_next_cli_message_uses_effective_session_workspace_and_fresh_turn(
    tmp_path: Path,
    mode: RewindMode,
) -> None:
    result = _ready(mode, tmp_path)
    assert result.target is not None
    default_loop = _Loop("live")
    restored_loop = _Loop("restored")
    bindings = _Bindings(result.target)
    cwd_before = Path.cwd()
    scheduler, _hub, teardown = build_repl(
        default_loop,
        "cli",
        lambda _text: None,
        continuation_bindings=bindings,
        continuation_loop_factory=lambda _target: restored_loop,
    )
    request = TurnRequest(
        Origin.USER,
        Source("cli", "source", "user", ChatType.DM),
        "continue",
        conversation=result.target.session_id,
    )
    try:
        handle = scheduler.submit(request)
        await handle.result()
    finally:
        await teardown()

    selected_loop = restored_loop if result.target.workspace_created else default_loop
    assert selected_loop.calls == [(result.target.session_id, "continue", False)]
    assert handle.turn_id and handle.turn_id != "old-turn"
    assert Path.cwd() == cwd_before
