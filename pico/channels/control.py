"""Transport-neutral inbound control decisions for Gateway-managed channels.

The adapter translates normalized :class:`~pico.spine.turn.TurnRequest` values
into existing Scheduler and question-broker operations.  It owns no scheduling
state and never invokes an Agent loop directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from pico.spine import BusyPolicy, ConversationStatus, Scheduler, TurnRequest


class QuestionReplies(Protocol):
    """Small ask-user surface needed by channel ingress."""

    def pending_req(self, conversation_id: str) -> str | None: ...

    def reply(self, key: str, answer: str) -> bool: ...


CancelRelated = Callable[[str], Awaitable[int]]


class GatewayControlOperation(StrEnum):
    """Transport-neutral outcome of one inbound Gateway dispatch."""

    STATUS = "status"
    STOP = "stop"
    INJECT = "inject"
    SUBMIT = "submit"
    QUESTION_REPLY = "question_reply"


@dataclass(frozen=True)
class GatewayControlResult:
    """Immutable facts a channel presenter may acknowledge to its user."""

    operation: GatewayControlOperation
    conversation_id: str
    status: ConversationStatus | None = None
    stopped_count: int | None = None
    correlated_turn_id: str | None = None


class GatewayControlAdapter:
    """Route Gateway control and ordinary messages through existing owners.

    Scheduler remains authoritative for status, cancellation, and Turn
    submission.  The question broker remains authoritative for pending
    ``ask_user`` replies.  ``cancel_related`` lets the Gateway preserve its
    existing conversation-wide subagent cancellation without exposing an Agent
    loop to this adapter.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        question_replies: QuestionReplies,
        *,
        cancel_related: CancelRelated | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._question_replies = question_replies
        self._cancel_related = cancel_related

    async def dispatch(self, req: TurnRequest) -> GatewayControlResult:
        """Handle one normalized inbound request without transport assumptions."""

        conversation_id = req.conversation or f"{req.source.channel}:{req.source.chat_id}"
        command = req.text.strip().lower()

        if command == "/status":
            return GatewayControlResult(
                operation=GatewayControlOperation.STATUS,
                conversation_id=conversation_id,
                status=self._scheduler.conversation_status(conversation_id),
            )

        if command == "/stop":
            stopped = self._scheduler.cancel_conversation(conversation_id)
            if self._cancel_related is not None:
                stopped += await self._cancel_related(conversation_id)
            return GatewayControlResult(
                operation=GatewayControlOperation.STOP,
                conversation_id=conversation_id,
                stopped_count=stopped,
            )

        if self._question_replies.pending_req(conversation_id) is not None:
            self._question_replies.reply(conversation_id, req.text)
            return GatewayControlResult(
                operation=GatewayControlOperation.QUESTION_REPLY,
                conversation_id=conversation_id,
            )

        status = self._scheduler.conversation_status(conversation_id)
        if status.running_turn_id is not None:
            self._scheduler.submit(replace(req, busy=BusyPolicy.INJECT))
            return GatewayControlResult(
                operation=GatewayControlOperation.INJECT,
                conversation_id=conversation_id,
                status=self._scheduler.conversation_status(conversation_id),
                correlated_turn_id=status.running_turn_id,
            )

        self._scheduler.submit(req)
        return GatewayControlResult(
            operation=GatewayControlOperation.SUBMIT,
            conversation_id=conversation_id,
        )


__all__ = [
    "CancelRelated",
    "GatewayControlAdapter",
    "GatewayControlOperation",
    "GatewayControlResult",
    "QuestionReplies",
]
