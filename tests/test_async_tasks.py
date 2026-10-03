from __future__ import annotations

import asyncio

import pytest

from omni.utils.async_tasks import await_task_preserving_cancellation


@pytest.mark.asyncio
async def test_completed_task_result_is_returned() -> None:
    task = asyncio.create_task(asyncio.sleep(0, result="done"))
    await task

    assert await await_task_preserving_cancellation(task) == "done"


@pytest.mark.asyncio
async def test_incomplete_task_is_awaited_until_success() -> None:
    release = asyncio.Event()

    async def finish_later() -> str:
        await release.wait()
        return "done"

    task = asyncio.create_task(finish_later())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    assert not waiter.done()
    release.set()
    assert await waiter == "done"


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_task_success_without_cancelling_it() -> None:
    release = asyncio.Event()

    async def finish_later() -> str:
        await release.wait()
        return "done"

    task = asyncio.create_task(finish_later())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    try:
        waiter.cancel("caller cancelled")
        await asyncio.sleep(0)

        assert not waiter.done()
        assert not task.cancelled()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await waiter
        assert task.result() == "done"
    finally:
        release.set()
        await asyncio.gather(task, waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_task_failure_and_preserves_its_cause() -> None:
    release = asyncio.Event()
    failure = RuntimeError("task failed")

    async def fail_later() -> None:
        await release.wait()
        raise failure

    task = asyncio.create_task(fail_later())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    try:
        waiter.cancel("caller cancelled first")
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError, match="caller cancelled first") as raised:
            await waiter
        assert raised.value.__cause__ is failure
        assert task.exception() is failure
    finally:
        release.set()
        await asyncio.gather(task, waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_task_cancellation_is_propagated_without_a_caller_cancellation() -> None:
    release = asyncio.Event()

    async def cancel_later() -> None:
        await release.wait()
        raise asyncio.CancelledError("task cancelled")

    task = asyncio.create_task(cancel_later())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError, match="task cancelled") as raised:
        await waiter
    assert raised.value.__cause__ is None
    assert task.cancelled()


@pytest.mark.asyncio
async def test_task_failure_is_propagated_with_its_identity() -> None:
    failure = RuntimeError("task failed")

    async def fail() -> None:
        raise failure

    task = asyncio.create_task(fail())

    with pytest.raises(RuntimeError) as raised:
        await await_task_preserving_cancellation(task)
    assert raised.value is failure


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_preserves_the_first_error() -> None:
    release = asyncio.Event()

    async def finish_later() -> None:
        await release.wait()

    task = asyncio.create_task(finish_later())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    try:
        waiter.cancel("first cancellation")
        await asyncio.sleep(0)
        waiter.cancel("second cancellation")
        await asyncio.sleep(0)

        assert not waiter.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await waiter
        assert raised.value.args == ("first cancellation",)
    finally:
        release.set()
        await asyncio.gather(task, waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_later_task_cancellation() -> None:
    task = asyncio.create_task(asyncio.Event().wait())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    waiter.cancel("caller cancelled")
    await asyncio.sleep(0)
    task.cancel("task cancelled")

    with pytest.raises(asyncio.CancelledError) as raised:
        await waiter
    assert raised.value.args == ("caller cancelled",)
    assert isinstance(raised.value.__cause__, asyncio.CancelledError)
    assert raised.value.__cause__.args == ("task cancelled",)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_task_cancellation_in_the_same_tick() -> None:
    task = asyncio.create_task(asyncio.Event().wait())
    waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    task.cancel("task cancelled")
    waiter.cancel("caller cancelled")

    with pytest.raises(asyncio.CancelledError) as raised:
        await waiter
    assert raised.value.args == ("caller cancelled",)
    assert isinstance(raised.value.__cause__, asyncio.CancelledError)
    assert raised.value.__cause__.args == ("task cancelled",)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_concurrent_waiters_do_not_cancel_or_consume_each_others_result() -> None:
    release = asyncio.Event()
    result = object()

    async def finish_later() -> object:
        await release.wait()
        return result

    task = asyncio.create_task(finish_later())
    cancelled_waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    successful_waiter = asyncio.create_task(await_task_preserving_cancellation(task))
    await asyncio.sleep(0)

    try:
        cancelled_waiter.cancel("one waiter cancelled")
        await asyncio.sleep(0)

        assert not task.cancelled()
        assert not successful_waiter.done()
        release.set()
        with pytest.raises(asyncio.CancelledError, match="one waiter cancelled"):
            await cancelled_waiter
        assert await successful_waiter is result
        assert task.result() is result
    finally:
        release.set()
        await asyncio.gather(
            task,
            cancelled_waiter,
            successful_waiter,
            return_exceptions=True,
        )
