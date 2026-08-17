"""Registration for the retained conversational TUI-RPC surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pico.tui_rpc.methods.config import register_config_methods
from pico.tui_rpc.methods.confirm import register_confirm_methods
from pico.tui_rpc.methods.image import register_image_methods
from pico.tui_rpc.methods.model import register_model_methods
from pico.tui_rpc.methods.question import register_question_methods
from pico.tui_rpc.methods.session import register_session_methods
from pico.tui_rpc.methods.setup import register_setup_methods
from pico.tui_rpc.methods.system import register_system_methods
from pico.tui_rpc.methods.terminal import register_terminal_methods
from pico.tui_rpc.methods.turn import register_turn_methods

if TYPE_CHECKING:
    from pico.spine.scheduler import Scheduler
    from pico.tui_rpc.confirm_broker import ConfirmBroker
    from pico.tui_rpc.dispatcher import Dispatcher
    from pico.tui_rpc.errors import RpcError
    from pico.tui_rpc.methods.session import AgentLoopFactory
    from pico.tui_rpc.methods.turn import SchedulerFactory
    from pico.tui_rpc.question_broker import QuestionBroker
    from pico.tui_rpc.subscriptions import SubscriptionEmitter


def register_aligned_methods(
    dispatcher: "Dispatcher",
    *,
    emitter: "SubscriptionEmitter | None" = None,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
    question_broker: "QuestionBroker | None" = None,
    scheduler: "Scheduler | None" = None,
    scheduler_factory: "SchedulerFactory | None" = None,
    turn_ids: "dict[int, str] | None" = None,
    submission_ids: "dict[int, str] | None" = None,
    build_error: "RpcError | None" = None,
) -> None:
    """Register every aligned RPC handler on a dispatcher.

    ``emitter`` and the build_tui bundle (``scheduler`` / ``turn_ids`` /
    ``build_error``) are forwarded to :func:`register_turn_methods` — when
    ``emitter`` is ``None`` the ``turn.*`` group is skipped.
    ``agent_loop_factory`` is forwarded to the session methods.
    ``confirm_broker`` is forwarded to :func:`register_confirm_methods`.
    """
    register_system_methods(dispatcher)
    register_aligned_methods_except_system(
        dispatcher,
        emitter=emitter,
        agent_loop_factory=agent_loop_factory,
        confirm_broker=confirm_broker,
        question_broker=question_broker,
        scheduler=scheduler,
        scheduler_factory=scheduler_factory,
        turn_ids=turn_ids,
        submission_ids=submission_ids,
        build_error=build_error,
    )


def register_aligned_methods_except_system(
    dispatcher: "Dispatcher",
    *,
    emitter: "SubscriptionEmitter | None" = None,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
    question_broker: "QuestionBroker | None" = None,
    scheduler: "Scheduler | None" = None,
    scheduler_factory: "SchedulerFactory | None" = None,
    turn_ids: "dict[int, str] | None" = None,
    submission_ids: "dict[int, str] | None" = None,
    build_error: "RpcError | None" = None,
) -> None:
    """Register the retained RPC handlers except ``system.*``."""
    register_setup_methods(dispatcher)
    register_config_methods(dispatcher, agent_loop_factory=agent_loop_factory)
    register_session_methods(
        dispatcher,
        agent_loop_factory=agent_loop_factory,
        confirm_broker=confirm_broker,
    )
    register_terminal_methods(dispatcher)
    register_image_methods(dispatcher)
    register_model_methods(dispatcher)
    if emitter is not None:
        register_turn_methods(
            dispatcher,
            emitter=emitter,
            scheduler=scheduler,
            scheduler_factory=scheduler_factory,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
            build_error=build_error,
        )
    if confirm_broker is not None:
        register_confirm_methods(dispatcher, confirm_broker=confirm_broker)
    if question_broker is not None:
        register_question_methods(dispatcher, question_broker=question_broker)


__all__ = [
    "register_aligned_methods",
    "register_aligned_methods_except_system",
    "register_system_methods",
    "register_setup_methods",
    "register_config_methods",
    "register_session_methods",
    "register_terminal_methods",
    "register_model_methods",
    "register_turn_methods",
    "register_confirm_methods",
    "register_image_methods",
    "register_question_methods",
]
