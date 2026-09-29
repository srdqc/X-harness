"""Selective-rewind coordination and per-continuation Workspace binding.

The coordinator composes Session and checkpoint owners.  It does not execute
an Agent Turn, replay tools, or mutate the process working directory.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from pico.agent.loop.checkpoint import CheckpointRecord, CheckpointService
from pico.agent.loop.workspace_restore import (
    WorkspaceCleanupState,
    WorkspaceRestoreResult,
    WorkspaceRestoreService,
    WorkspaceRestoreStatus,
)
from pico.session.manager import Session, SessionManager, SessionTurnBoundary
from pico.spine.runner import Drain, Emit, TurnOutcome, TurnRunner
from pico.spine.turn import TurnRequest
from pico.utils.atomic_io import StorageCorruptionError


class RewindMode(str, Enum):
    CONVERSATION = "conversation"
    WORKSPACE = "workspace"
    BOTH = "both"


class RewindStatus(str, Enum):
    READY = "ready"
    VALIDATION_FAILED = "validation_failed"
    PREPARATION_FAILED = "preparation_failed"
    UNSUPPORTED = "unsupported"


class WorkspaceLeaseState(str, Enum):
    PREPARED = "prepared"
    ACTIVE = "active"
    RELEASED = "released"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class RewindRequest:
    mode: RewindMode
    session_id: str
    boundary_id: str | None = None
    checkpoint_record_id: str | None = None


@dataclass(frozen=True)
class ContinuationTarget:
    """Immutable facts needed to construct or select the next normal runner."""

    continuation_id: str
    session_id: str
    workspace_id: str
    workspace_path: Path
    source_session_id: str
    selected_boundary: SessionTurnBoundary | None
    selected_checkpoint: CheckpointRecord | None
    session_created: bool
    workspace_created: bool
    workspace_lease_state: WorkspaceLeaseState


@dataclass(frozen=True)
class RewindResult:
    status: RewindStatus
    mode: RewindMode
    target: ContinuationTarget | None = None
    selected_boundary: SessionTurnBoundary | None = None
    selected_checkpoint: CheckpointRecord | None = None
    reason: str | None = None
    cleanup_failures: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is RewindStatus.READY and self.target is not None


@dataclass(frozen=True)
class ContinuationRelease:
    continuation_id: str
    session_id: str
    state: WorkspaceLeaseState
    workspace_cleaned: bool | None
    reason: str | None = None


@dataclass
class _BindingEntry:
    target: ContinuationTarget
    restore: WorkspaceRestoreResult | None
    active_turns: int = 0


class ContinuationBindingError(RuntimeError):
    """A continuation cannot be published without ambiguous ownership."""


class ContinuationBindings:
    """Process-local bindings and leases for prepared continuations.

    Bindings are indexed by Session identity, but restored Workspace identity
    is leased separately so it cannot be bound twice.  Active runner calls hold
    a usage count; release refuses to clean an in-use restored Workspace.
    """

    def __init__(self, restorer: WorkspaceRestoreService) -> None:
        self._restorer = restorer
        self._lock = threading.RLock()
        self._by_session: dict[str, _BindingEntry] = {}
        self._workspace_owners: dict[str, str] = {}
        self._continuation_owners: dict[str, str] = {}

    def activate(
        self,
        target: ContinuationTarget,
        restore: WorkspaceRestoreResult | None,
    ) -> ContinuationTarget:
        with self._lock:
            if target.session_id in self._by_session:
                raise ContinuationBindingError("session_already_bound")
            if target.continuation_id in self._continuation_owners:
                raise ContinuationBindingError("continuation_id_already_bound")
            if target.workspace_created:
                if (
                    restore is None
                    or not restore.usable
                    or restore.workspace_id != target.workspace_id
                    or restore.workspace_path != target.workspace_path
                ):
                    raise ContinuationBindingError("restored_workspace_evidence_mismatch")
                if target.workspace_id in self._workspace_owners:
                    raise ContinuationBindingError("workspace_already_bound")
            active = replace(target, workspace_lease_state=WorkspaceLeaseState.ACTIVE)
            self._by_session[active.session_id] = _BindingEntry(active, restore)
            self._continuation_owners[active.continuation_id] = active.session_id
            if active.workspace_created:
                self._workspace_owners[active.workspace_id] = active.session_id
            return active

    def resolve(self, session_id: str) -> ContinuationTarget | None:
        with self._lock:
            entry = self._by_session.get(session_id)
            if entry is None or entry.target.workspace_lease_state is not WorkspaceLeaseState.ACTIVE:
                return None
            return entry.target

    def acquire(self, session_id: str) -> ContinuationTarget | None:
        with self._lock:
            entry = self._by_session.get(session_id)
            if entry is None or entry.target.workspace_lease_state is not WorkspaceLeaseState.ACTIVE:
                return None
            entry.active_turns += 1
            return entry.target

    def finish(self, continuation_id: str) -> None:
        with self._lock:
            for entry in self._by_session.values():
                if entry.target.continuation_id == continuation_id:
                    if entry.active_turns <= 0:
                        raise ContinuationBindingError("continuation_not_acquired")
                    entry.active_turns -= 1
                    return
        raise ContinuationBindingError("continuation_binding_missing")

    def release(self, session_id: str) -> ContinuationRelease | None:
        with self._lock:
            entry = self._by_session.get(session_id)
            if entry is None:
                return None
            target = entry.target
            if entry.active_turns:
                return ContinuationRelease(
                    target.continuation_id,
                    session_id,
                    WorkspaceLeaseState.ACTIVE,
                    None,
                    "continuation_in_use",
                )
            cleaned: bool | None = None
            if target.workspace_created:
                if entry.restore is None:
                    cleaned = False
                else:
                    cleaned = self._restorer.cleanup(entry.restore)
                if not cleaned:
                    entry.target = replace(
                        target,
                        workspace_lease_state=WorkspaceLeaseState.CLEANUP_FAILED,
                    )
                    return ContinuationRelease(
                        target.continuation_id,
                        session_id,
                        WorkspaceLeaseState.CLEANUP_FAILED,
                        False,
                        "workspace_cleanup_failed",
                    )
                self._workspace_owners.pop(target.workspace_id, None)
            self._by_session.pop(session_id, None)
            self._continuation_owners.pop(target.continuation_id, None)
            return ContinuationRelease(
                target.continuation_id,
                session_id,
                WorkspaceLeaseState.RELEASED,
                cleaned,
            )


class ContinuationTurnRunner:
    """Route restored continuations through a Workspace-specific normal runner."""

    def __init__(
        self,
        default_runner: TurnRunner,
        bindings: ContinuationBindings,
        runner_factory: Callable[[ContinuationTarget], TurnRunner],
    ) -> None:
        self._default_runner = default_runner
        self._bindings = bindings
        self._runner_factory = runner_factory
        self._runners: dict[str, TurnRunner] = {}

    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome:
        session_id = req.conversation or f"{req.source.channel}:{req.source.chat_id}"
        target = self._bindings.acquire(session_id)
        if target is None:
            return await self._default_runner.run(req, emit, drain)
        try:
            runner = self._default_runner
            if target.workspace_created:
                runner = self._runners.get(target.continuation_id)
                if runner is None:
                    runner = self._runner_factory(target)
                    self._runners[target.continuation_id] = runner
            return await runner.run(req, emit, drain)
        finally:
            self._bindings.finish(target.continuation_id)


class SelectiveRewindCoordinator:
    """Validate, prepare, and atomically publish one local continuation binding."""

    def __init__(
        self,
        sessions: SessionManager,
        checkpoints: CheckpointService,
        restorer: WorkspaceRestoreService,
        bindings: ContinuationBindings,
        *,
        is_session_active: Callable[[str], bool],
        continuation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._sessions = sessions
        self._checkpoints = checkpoints
        self._restorer = restorer
        self._bindings = bindings
        self._is_session_active = is_session_active
        self._continuation_id_factory = continuation_id_factory or (lambda: uuid.uuid4().hex)
        self._prepare_lock = asyncio.Lock()

    def _result(
        self,
        status: RewindStatus,
        request: RewindRequest,
        *,
        boundary: SessionTurnBoundary | None = None,
        checkpoint: CheckpointRecord | None = None,
        reason: str | None = None,
        cleanup_failures: tuple[str, ...] = (),
    ) -> RewindResult:
        return RewindResult(
            status,
            request.mode,
            selected_boundary=boundary,
            selected_checkpoint=checkpoint,
            reason=reason,
            cleanup_failures=cleanup_failures,
        )

    def _new_continuation_id(self) -> str:
        continuation_id = self._continuation_id_factory()
        if not isinstance(continuation_id, str) or not continuation_id:
            raise ValueError("continuation_id_factory must return a non-empty string")
        return continuation_id

    def _cleanup(
        self,
        child: Session | None,
        restored: WorkspaceRestoreResult | None,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if child is not None and not self._sessions.discard_fork_child(child):
            failures.append("session_cleanup_failed")
        if restored is not None:
            if restored.usable:
                if not self._restorer.cleanup(restored):
                    failures.append("workspace_cleanup_failed")
            elif restored.cleanup_state is WorkspaceCleanupState.FAILED:
                failures.append("workspace_cleanup_failed")
        return tuple(failures)

    async def rewind(self, request: RewindRequest) -> RewindResult:
        async with self._prepare_lock:
            return await self._rewind_locked(request)

    async def _rewind_locked(self, request: RewindRequest) -> RewindResult:
        if not isinstance(request.session_id, str) or not request.session_id:
            return self._result(RewindStatus.VALIDATION_FAILED, request, reason="session_required")
        if self._is_session_active(request.session_id):
            return self._result(RewindStatus.VALIDATION_FAILED, request, reason="source_session_active")
        try:
            source = self._sessions.peek(request.session_id)
        except (OSError, StorageCorruptionError, UnicodeError):
            return self._result(RewindStatus.VALIDATION_FAILED, request, reason="session_corrupt")
        if source is None:
            return self._result(RewindStatus.VALIDATION_FAILED, request, reason="session_missing")

        needs_conversation = request.mode in {RewindMode.CONVERSATION, RewindMode.BOTH}
        needs_workspace = request.mode in {RewindMode.WORKSPACE, RewindMode.BOTH}
        boundary: SessionTurnBoundary | None = None
        checkpoint: CheckpointRecord | None = None
        validation = None

        if needs_conversation:
            if request.boundary_id is None:
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    reason="boundary_required",
                )
            try:
                boundary = self._sessions.get_turn_boundary(
                    request.session_id,
                    request.boundary_id,
                )
            except (OSError, StorageCorruptionError, UnicodeError):
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    reason="session_boundary_corrupt",
                )
            if boundary is None:
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    reason="boundary_missing",
                )

        if needs_workspace:
            if request.checkpoint_record_id is None:
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    boundary=boundary,
                    reason="checkpoint_required",
                )
            if request.mode is RewindMode.BOTH and boundary is not None and boundary.turn_id is None:
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    boundary=boundary,
                    reason="turn_correlation_missing",
                )
            validation = await self._checkpoints.validate(
                request.checkpoint_record_id,
                expected_workspace=self._checkpoints.workspace_path,
                expected_session_id=(request.session_id if request.mode is RewindMode.BOTH else None),
                expected_boundary_id=(boundary.boundary_id if request.mode is RewindMode.BOTH and boundary else None),
                expected_turn_id=(boundary.turn_id if request.mode is RewindMode.BOTH and boundary else None),
            )
            checkpoint = validation.record
            if not validation.usable or checkpoint is None:
                reason = validation.reason or "checkpoint_validation_failed"
                if request.mode is RewindMode.BOTH and checkpoint is not None:
                    required = (checkpoint.session_id, checkpoint.boundary_id, checkpoint.turn_id)
                    if any(value is None for value in required):
                        reason = "checkpoint_correlation_missing"
                return self._result(
                    RewindStatus.VALIDATION_FAILED,
                    request,
                    boundary=boundary,
                    checkpoint=checkpoint,
                    reason=reason,
                )

        child: Session | None = None
        restored: WorkspaceRestoreResult | None = None
        session_error: str | None = None
        if needs_conversation and boundary is not None:
            try:
                child = self._sessions.fork_at(request.session_id, boundary.boundary_id)
            except Exception as exc:
                session_error = str(exc) or "session_preparation_failed"
            if child is None and session_error is None:
                session_error = "session_preparation_failed"

        if needs_workspace and checkpoint is not None:
            try:
                restored = await self._restorer.restore(
                    checkpoint.record_id,
                    expected_workspace=self._checkpoints.workspace_path,
                    expected_session_id=(
                        request.session_id if request.mode is RewindMode.BOTH else None
                    ),
                    expected_boundary_id=(
                        boundary.boundary_id
                        if request.mode is RewindMode.BOTH and boundary
                        else None
                    ),
                    expected_turn_id=(
                        boundary.turn_id
                        if request.mode is RewindMode.BOTH and boundary
                        else None
                    ),
                )
            except Exception as exc:
                workspace_exception = str(exc) or "workspace_preparation_failed"
            else:
                workspace_exception = None
        else:
            workspace_exception = None

        workspace_error = workspace_exception is not None or (
            restored is not None and not restored.usable
        )
        if session_error is not None or workspace_error:
            failures = self._cleanup(child, restored)
            if restored is not None and restored.status is WorkspaceRestoreStatus.UNSUPPORTED:
                status = RewindStatus.UNSUPPORTED
            else:
                status = RewindStatus.PREPARATION_FAILED
            reason = session_error or workspace_exception or (
                restored.reason if restored is not None else None
            )
            return self._result(
                status,
                request,
                boundary=boundary,
                checkpoint=checkpoint,
                reason=reason or "preparation_failed",
                cleanup_failures=failures,
            )

        # A Scheduler snapshot is not a reservation.  Recheck immediately
        # before publication and fail closed if work appeared during preparation.
        if self._is_session_active(request.session_id):
            failures = self._cleanup(child, restored)
            return self._result(
                RewindStatus.VALIDATION_FAILED,
                request,
                boundary=boundary,
                checkpoint=checkpoint,
                reason="source_session_became_active",
                cleanup_failures=failures,
            )

        workspace_path = (
            restored.workspace_path
            if restored is not None and restored.workspace_path is not None
            else self._checkpoints.workspace_path
        )
        workspace_id = (
            restored.workspace_id
            if restored is not None and restored.workspace_id is not None
            else str(workspace_path)
        )
        try:
            continuation_id = self._new_continuation_id()
        except ValueError as exc:
            failures = self._cleanup(child, restored)
            return self._result(
                RewindStatus.PREPARATION_FAILED,
                request,
                boundary=boundary,
                checkpoint=checkpoint,
                reason=str(exc),
                cleanup_failures=failures,
            )
        target = ContinuationTarget(
            continuation_id=continuation_id,
            session_id=child.key if child is not None else request.session_id,
            workspace_id=workspace_id,
            workspace_path=workspace_path,
            source_session_id=request.session_id,
            selected_boundary=boundary,
            selected_checkpoint=checkpoint,
            session_created=child is not None,
            workspace_created=restored is not None,
            workspace_lease_state=WorkspaceLeaseState.PREPARED,
        )
        try:
            active_target = self._bindings.activate(target, restored)
        except (ContinuationBindingError, ValueError) as exc:
            failures = self._cleanup(child, restored)
            return self._result(
                RewindStatus.PREPARATION_FAILED,
                request,
                boundary=boundary,
                checkpoint=checkpoint,
                reason=str(exc) or "continuation_activation_failed",
                cleanup_failures=failures,
            )
        return RewindResult(
            RewindStatus.READY,
            request.mode,
            target=active_target,
            selected_boundary=boundary,
            selected_checkpoint=checkpoint,
        )


__all__ = [
    "ContinuationBindingError",
    "ContinuationBindings",
    "ContinuationRelease",
    "ContinuationTarget",
    "ContinuationTurnRunner",
    "RewindMode",
    "RewindRequest",
    "RewindResult",
    "RewindStatus",
    "SelectiveRewindCoordinator",
    "WorkspaceLeaseState",
]
