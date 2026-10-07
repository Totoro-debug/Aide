"""Shared cancellation-aware asyncio Task waiting."""

import asyncio


async def await_task_preserving_cancellation[T](task: asyncio.Task[T]) -> T:
    """Wait for a Task to finish before propagating caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as caught:
            waiter = asyncio.current_task()
            caller_cancelled = waiter is not None and waiter.cancelling() > 0
            if task.cancelled() and not caller_cancelled:
                break
            if cancellation is None:
                cancellation = caught
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    if cancellation is not None:
        raise cancellation
    return result
