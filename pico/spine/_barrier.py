"""Cancellation-safe completion of an already-started Spine barrier."""

import asyncio


async def finish_barrier(
    task: asyncio.Task[None],
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Finish ``task`` and keep caller cancellation above barrier failure."""
    failure: BaseException | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            failure = exc
            break
    if failure is None:
        try:
            await task
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            failure = exc
    if cancellation is not None:
        if failure is not None:
            raise cancellation from failure
        raise cancellation
    if failure is not None:
        raise failure
