"""Deterministic teardown for the Scheduler and DeliveryHub pair."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from pico.spine.delivery import DeliveryHub
from pico.spine.scheduler import Scheduler


async def teardown_spine(scheduler: Scheduler, delivery: DeliveryHub, *, grace: float) -> None:
    """Attempt both shutdown barriers, then raise their highest-priority failure."""
    first_error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None

    async def attempt(name: str, cleanup: Callable[[], Awaitable[None]]) -> None:
        nonlocal cancellation, first_error
        try:
            await cleanup()
        except asyncio.CancelledError as exc:
            logger.opt(exception=exc).error("Spine teardown step {} was cancelled", name)
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            logger.opt(exception=exc).error("Spine teardown step {} failed", name)
            if first_error is None:
                first_error = exc

    await attempt("scheduler", lambda: scheduler.shutdown(grace=grace))
    await attempt("delivery", delivery.aclose)
    if cancellation is not None:
        raise cancellation
    if first_error is not None:
        raise first_error
