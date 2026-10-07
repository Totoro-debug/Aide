import asyncio
from io import StringIO
from typing import Literal

import pytest
from loguru import logger

from aide.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
)


class _AcquireTrackingLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.acquire_attempted = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        self.acquire_attempted.set()
        return await super().acquire()


class _ControlledCondition(asyncio.Condition):
    def __init__(self, expected_waiters: int = 0) -> None:
        lock = _AcquireTrackingLock()
        super().__init__(lock)
        self.acquire_attempted = lock.acquire_attempted
        self._expected_waiters = expected_waiters
        self.wait_count = 0
        self.all_waiters_waiting = asyncio.Event()

    async def wait(self) -> Literal[True]:
        self.wait_count += 1
        if self.wait_count == self._expected_waiters:
            self.all_waiters_waiting.set()
        return await super().wait()


async def _wait_for_acquisition(condition: _ControlledCondition) -> None:
    await condition.acquire_attempted.wait()
    condition.acquire_attempted.clear()


def test_message_metadata_defaults_are_independent_and_outbound_types_are_supported() -> None:
    first_inbound = InboundMessage(content="first")
    second_inbound = InboundMessage(content="second")
    first_inbound.metadata["source"] = "test"

    assert second_inbound.metadata == {}

    outbound = [
        OutboundMessage(type="model_reasoning", content="thinking"),
        OutboundMessage(type="model_response", content="answer"),
        OutboundMessage(type="tool_call", content="read_file"),
        OutboundMessage(type="system_control", content="cancelled"),
    ]
    assert [message.type for message in outbound] == [
        "model_reasoning",
        "model_response",
        "tool_call",
        "system_control",
    ]


def test_message_bus_has_no_synchronous_inbound_detach_mutation_seam() -> None:
    assert not hasattr(MessageBus, "_detach_inbound")


@pytest.mark.asyncio
async def test_inbound_snapshot_put_get_and_drain_publish_post_mutation_snapshots() -> None:
    bus = MessageBus()
    observed: list[tuple[InboundMessage, ...]] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        # Lock release is a required callback ordering contract with no public probe.
        assert not bus._condition.locked()
        observed.append(snapshot)

    bus.set_inbound_changed_callback(observe)
    first = InboundMessage(content="first")
    second = InboundMessage(content="second")

    assert await bus.inbound_snapshot() == ()
    assert observed == []

    await bus.put_inbound(first)
    assert list(observed) == [(first,)]
    assert await bus.inbound_snapshot() == (first,)
    assert list(observed) == [(first,)]

    await bus.put_inbound(second)
    assert list(observed[-1]) == [first, second]
    assert await bus.get_inbound() is first
    assert list(observed[-1]) == [second]

    assert await bus.drain_inbound() == (second,)
    assert list(observed[-1]) == []
    assert await bus.inbound_snapshot() == ()


@pytest.mark.asyncio
async def test_empty_drain_still_reports_the_post_operation_snapshot() -> None:
    bus = MessageBus()
    observed: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(observed.append)

    assert await bus.drain_inbound() == ()
    assert observed == [()]


@pytest.mark.asyncio
async def test_reset_clears_both_fifos_and_publishes_one_empty_snapshot() -> None:
    bus = MessageBus()
    observed: list[tuple[InboundMessage, ...]] = []
    callback_lock_states: list[bool] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        observed.append(snapshot)
        callback_lock_states.append(bus._condition.locked())

    bus.set_inbound_changed_callback(observe)
    old_inbound = InboundMessage(content="old inbound")
    old_outbound = OutboundMessage(type="model_response", content="old outbound")
    await bus.put_inbound(old_inbound)
    await bus.put_outbound(old_outbound)
    observed.clear()
    callback_lock_states.clear()

    await bus.reset()

    assert await bus.inbound_snapshot() == ()
    assert observed == [()]
    assert callback_lock_states == [False]

    new_inbound = InboundMessage(content="new inbound")
    new_outbound = OutboundMessage(type="model_response", content="new outbound")
    await bus.put_inbound(new_inbound)
    await bus.put_outbound(new_outbound)

    assert await bus.get_inbound() is new_inbound
    assert await bus.get_outbound() is new_outbound


@pytest.mark.parametrize("first_lane", ["inbound", "outbound"])
@pytest.mark.asyncio
async def test_reset_keeps_both_waiting_getters_pending_until_their_next_messages(
    first_lane: str,
) -> None:
    bus = MessageBus()
    condition = _ControlledCondition(expected_waiters=2)
    bus._condition = condition
    inbound_task = asyncio.create_task(bus.get_inbound())
    outbound_task = asyncio.create_task(bus.get_outbound())

    try:
        await condition.all_waiters_waiting.wait()
        assert condition.wait_count == 2
        assert not inbound_task.done()
        assert not outbound_task.done()

        await bus.reset()

        assert not inbound_task.done()
        assert not outbound_task.done()

        inbound = InboundMessage(content="after reset inbound")
        outbound = OutboundMessage(type="model_response", content="after reset outbound")
        if first_lane == "inbound":
            await bus.put_inbound(inbound)
            assert await inbound_task is inbound
            assert not outbound_task.done()
            await bus.put_outbound(outbound)
            assert await outbound_task is outbound
        else:
            await bus.put_outbound(outbound)
            assert await outbound_task is outbound
            assert not inbound_task.done()
            await bus.put_inbound(inbound)
            assert await inbound_task is inbound
    finally:
        for task in (inbound_task, outbound_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(inbound_task, outbound_task, return_exceptions=True)


@pytest.mark.parametrize("lane", ["inbound", "outbound"])
@pytest.mark.asyncio
async def test_cancelling_a_waiting_getter_does_not_consume_the_next_message(lane: str) -> None:
    bus = MessageBus()
    condition = _ControlledCondition(expected_waiters=1)
    bus._condition = condition
    getter = asyncio.create_task(bus.get_inbound() if lane == "inbound" else bus.get_outbound())
    await condition.all_waiters_waiting.wait()

    getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getter

    if lane == "inbound":
        inbound_message = InboundMessage(content="after cancelled inbound getter")
        await bus.put_inbound(inbound_message)
        assert await bus.get_inbound() is inbound_message
    else:
        outbound_message = OutboundMessage(
            type="model_response", content="after cancelled outbound getter"
        )
        await bus.put_outbound(outbound_message)
        assert await bus.get_outbound() is outbound_message


@pytest.mark.parametrize("lane", ["inbound", "outbound"])
@pytest.mark.asyncio
async def test_cancelling_a_put_waiting_for_coordination_does_not_admit_the_message(
    lane: str,
) -> None:
    bus = MessageBus()
    condition = _ControlledCondition()
    bus._condition = condition
    observed: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(observed.append)
    await condition.acquire()
    condition.acquire_attempted.clear()

    put_task = asyncio.create_task(
        bus.put_inbound(InboundMessage(content="cancelled inbound"))
        if lane == "inbound"
        else bus.put_outbound(OutboundMessage(type="model_response", content="cancelled outbound"))
    )
    await condition.acquire_attempted.wait()
    put_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await put_task
    condition.release()

    assert observed == []
    if lane == "inbound":
        assert await bus.inbound_snapshot() == ()
    else:
        message = OutboundMessage(type="model_response", content="next outbound")
        await bus.put_outbound(message)
        assert await bus.get_outbound() is message


@pytest.mark.asyncio
async def test_reset_cancellation_before_lock_acquisition_has_no_state_or_callback_change() -> None:
    bus = MessageBus()
    condition = _ControlledCondition()
    bus._condition = condition
    old_inbound = InboundMessage(content="must survive cancellation")
    old_outbound = OutboundMessage(type="model_response", content="must survive cancellation")
    await bus.put_inbound(old_inbound)
    await bus.put_outbound(old_outbound)
    observed: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(observed.append)

    condition.acquire_attempted.clear()
    await condition.acquire()
    condition.acquire_attempted.clear()
    reset_task = asyncio.create_task(bus.reset())
    await condition.acquire_attempted.wait()
    reset_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reset_task
    condition.release()

    assert await bus.inbound_snapshot() == (old_inbound,)
    assert await bus.get_outbound() is old_outbound
    assert observed == []


@pytest.mark.asyncio
async def test_reset_cancellation_after_linearization_keeps_the_clear_and_callback() -> None:
    bus = MessageBus()
    await bus.put_inbound(InboundMessage(content="cleared inbound"))
    await bus.put_outbound(OutboundMessage(type="model_response", content="cleared outbound"))
    observed: list[tuple[InboundMessage, ...]] = []
    reset_task: asyncio.Task[None] | None = None

    def cancel_reset(snapshot: tuple[InboundMessage, ...]) -> None:
        observed.append(snapshot)
        current = asyncio.current_task()
        assert current is not None
        assert current is reset_task
        current.cancel()

    bus.set_inbound_changed_callback(cancel_reset)
    reset_task = asyncio.create_task(bus.reset())

    with pytest.raises(asyncio.CancelledError):
        await reset_task

    assert await bus.inbound_snapshot() == ()
    assert observed == [()]
    next_outbound = OutboundMessage(type="model_response", content="next outbound")
    await bus.put_outbound(next_outbound)
    assert await bus.get_outbound() is next_outbound


@pytest.mark.asyncio
async def test_reset_callback_can_reenter_and_fail_after_observing_both_fifos_empty() -> None:
    bus = MessageBus()
    old_inbound = InboundMessage(content="stale inbound")
    old_outbound = OutboundMessage(type="model_response", content="stale outbound")
    await bus.put_inbound(old_inbound)
    await bus.put_outbound(old_outbound)
    observed: list[tuple[InboundMessage, ...]] = []
    callback_lock_states: list[bool] = []
    reentered = asyncio.Event()
    reentrant_message = InboundMessage(content="callback re-entry")
    reentry_tasks: list[asyncio.Task[None]] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        observed.append(snapshot)
        callback_lock_states.append(bus._condition.locked())
        assert tuple(bus._inbound) == ()
        assert tuple(bus._outbound) == ()
        bus.set_inbound_changed_callback(None)

        async def reenter() -> None:
            await bus.put_inbound(reentrant_message)
            reentered.set()

        reentry_tasks.append(asyncio.create_task(reenter()))
        raise RuntimeError("reset callback failure")

    bus.set_inbound_changed_callback(observe)
    await bus.reset()
    await reentered.wait()
    await asyncio.gather(*reentry_tasks)

    assert observed == [()]
    assert callback_lock_states == [False]
    assert await bus.inbound_snapshot() == (reentrant_message,)


@pytest.mark.parametrize("lane", ["inbound", "outbound"])
@pytest.mark.asyncio
async def test_put_before_reset_is_cleared_without_a_late_inbound_callback(lane: str) -> None:
    bus = MessageBus()
    condition = _ControlledCondition()
    bus._condition = condition
    callbacks: list[tuple[InboundMessage, ...]] = []
    reset_observations: list[
        tuple[tuple[InboundMessage, ...], tuple[OutboundMessage, ...], bool]
    ] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        callbacks.append(snapshot)
        if snapshot == ():
            reset_observations.append(
                (tuple(bus._inbound), tuple(bus._outbound), condition.locked())
            )

    bus.set_inbound_changed_callback(observe)

    before_inbound = InboundMessage(content="put before reset")
    before_outbound = OutboundMessage(type="model_response", content="put before reset")
    await condition.acquire()
    condition.acquire_attempted.clear()
    put_before_task = asyncio.create_task(
        bus.put_inbound(before_inbound) if lane == "inbound" else bus.put_outbound(before_outbound)
    )
    await _wait_for_acquisition(condition)
    reset_before_task = asyncio.create_task(bus.reset())
    await _wait_for_acquisition(condition)
    condition.release()
    await asyncio.gather(put_before_task, reset_before_task)

    assert await bus.inbound_snapshot() == ()
    assert callbacks == ([(before_inbound,), ()] if lane == "inbound" else [()])
    assert reset_observations == [((), (), False)]
    next_outbound = OutboundMessage(type="model_response", content="next outbound")
    await bus.put_outbound(next_outbound)
    assert await bus.get_outbound() is next_outbound


@pytest.mark.parametrize("lane", ["inbound", "outbound"])
@pytest.mark.asyncio
async def test_put_after_reset_is_retained_without_a_half_cleared_observation(lane: str) -> None:
    bus = MessageBus()
    condition = _ControlledCondition()
    bus._condition = condition
    callbacks: list[tuple[InboundMessage, ...]] = []
    reset_observations: list[
        tuple[tuple[InboundMessage, ...], tuple[OutboundMessage, ...], bool]
    ] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        callbacks.append(snapshot)
        if snapshot == ():
            reset_observations.append(
                (tuple(bus._inbound), tuple(bus._outbound), condition.locked())
            )

    bus.set_inbound_changed_callback(observe)
    after_inbound = InboundMessage(content="put after reset")
    after_outbound = OutboundMessage(type="model_response", content="put after reset")
    await bus.put_inbound(InboundMessage(content="cleared inbound"))
    await bus.put_outbound(OutboundMessage(type="model_response", content="cleared outbound"))
    callbacks.clear()
    await condition.acquire()
    condition.acquire_attempted.clear()
    reset_after_task = asyncio.create_task(bus.reset())
    await _wait_for_acquisition(condition)
    put_after_task = asyncio.create_task(
        bus.put_inbound(after_inbound) if lane == "inbound" else bus.put_outbound(after_outbound)
    )
    await _wait_for_acquisition(condition)
    condition.release()
    await asyncio.gather(reset_after_task, put_after_task)

    assert callbacks == ([(), (after_inbound,)] if lane == "inbound" else [()])
    assert reset_observations == [((), (), False)]
    assert await bus.inbound_snapshot() == ((after_inbound,) if lane == "inbound" else ())
    if lane == "outbound":
        assert await bus.get_outbound() is after_outbound


@pytest.mark.asyncio
async def test_reset_linearizes_get_before_and_after_without_returning_stale_messages() -> None:
    before_bus = MessageBus()
    before_condition = _ControlledCondition(expected_waiters=0)
    before_bus._condition = before_condition
    before_inbound = InboundMessage(content="get before reset")
    before_outbound = OutboundMessage(type="model_response", content="get before reset")
    await before_bus.put_inbound(before_inbound)
    await before_bus.put_outbound(before_outbound)
    await before_condition.acquire()
    before_condition.acquire_attempted.clear()

    get_before_inbound_task = asyncio.create_task(before_bus.get_inbound())
    await _wait_for_acquisition(before_condition)
    get_before_outbound_task = asyncio.create_task(before_bus.get_outbound())
    await _wait_for_acquisition(before_condition)
    reset_after_get_task = asyncio.create_task(before_bus.reset())
    await _wait_for_acquisition(before_condition)
    before_condition.release()

    inbound_result, outbound_result = await asyncio.gather(
        get_before_inbound_task, get_before_outbound_task
    )
    assert inbound_result is before_inbound
    assert outbound_result is before_outbound
    await reset_after_get_task
    assert await before_bus.inbound_snapshot() == ()

    after_bus = MessageBus()
    after_condition = _ControlledCondition(expected_waiters=1)
    after_bus._condition = after_condition
    stale_inbound = InboundMessage(content="stale inbound")
    stale_outbound = OutboundMessage(type="model_response", content="stale outbound")
    await after_bus.put_inbound(stale_inbound)
    await after_bus.put_outbound(stale_outbound)
    await after_condition.acquire()
    after_condition.acquire_attempted.clear()

    async def get_after_reset() -> tuple[InboundMessage, OutboundMessage]:
        return await after_bus.get_inbound(), await after_bus.get_outbound()

    reset_before_get_task = asyncio.create_task(after_bus.reset())
    await _wait_for_acquisition(after_condition)
    get_after_task = asyncio.create_task(get_after_reset())
    await _wait_for_acquisition(after_condition)
    after_condition.release()
    await reset_before_get_task
    await after_condition.all_waiters_waiting.wait()
    assert not get_after_task.done()

    new_inbound = InboundMessage(content="new inbound")
    new_outbound = OutboundMessage(type="model_response", content="new outbound")
    await after_bus.put_inbound(new_inbound)
    await after_bus.put_outbound(new_outbound)

    assert await get_after_task == (new_inbound, new_outbound)


@pytest.mark.asyncio
async def test_inbound_callback_can_be_replaced_or_cleared() -> None:
    bus = MessageBus()
    first_observed: list[tuple[InboundMessage, ...]] = []
    second_observed: list[tuple[InboundMessage, ...]] = []
    first = InboundMessage(content="first")
    second = InboundMessage(content="second")

    bus.set_inbound_changed_callback(first_observed.append)
    bus.set_inbound_changed_callback(second_observed.append)
    await bus.put_inbound(first)
    bus.set_inbound_changed_callback(None)
    await bus.put_inbound(second)

    assert first_observed == []
    assert second_observed == [(first,)]


@pytest.mark.asyncio
async def test_inbound_callback_unbind_is_identity_safe() -> None:
    bus = MessageBus()
    first_observed: list[tuple[InboundMessage, ...]] = []
    second_observed: list[tuple[InboundMessage, ...]] = []
    first_callback = first_observed.append
    second_callback = second_observed.append
    message = InboundMessage(content="still bound")

    bus.set_inbound_changed_callback(first_callback)
    bus.set_inbound_changed_callback(second_callback)
    bus.unbind_inbound_changed_callback(first_callback)
    await bus.put_inbound(message)
    bus.unbind_inbound_changed_callback(second_callback)
    await bus.drain_inbound()

    assert first_observed == []
    assert second_observed == [(message,)]


@pytest.mark.asyncio
async def test_callback_failure_is_logged_and_does_not_roll_back_queue_state() -> None:
    bus = MessageBus()

    def fail(_snapshot: tuple[InboundMessage, ...]) -> None:
        raise RuntimeError("callback failure")

    bus.set_inbound_changed_callback(fail)
    message = InboundMessage(content="survives")
    output = StringIO()
    handler_id = logger.add(output, format="{message}", level="ERROR")

    try:
        await bus.put_inbound(message)
    finally:
        logger.remove(handler_id)

    assert await bus.inbound_snapshot() == (message,)
    assert "Inbound changed callback failed" in output.getvalue()


@pytest.mark.asyncio
async def test_blocked_inbound_get_resumes_when_a_message_arrives() -> None:
    bus = MessageBus()
    get_task = asyncio.create_task(bus.get_inbound())
    await asyncio.sleep(0)
    assert not get_task.done()

    message = InboundMessage(content="released")
    await bus.put_inbound(message)

    assert await get_task is message


@pytest.mark.asyncio
async def test_paused_inbound_delivery_keeps_queued_work_unclaimed_until_resume() -> None:
    bus = MessageBus()
    message = InboundMessage(content="held for replacement")
    await bus.pause_inbound_delivery()
    getter_started = asyncio.Event()

    async def get_inbound() -> InboundMessage:
        getter_started.set()
        return await bus.get_inbound()

    getter = asyncio.create_task(get_inbound())

    try:
        await getter_started.wait()
        await bus.put_inbound(message)

        assert not getter.done()
        assert await bus.inbound_snapshot() == (message,)

        await bus.resume_inbound_delivery()
        assert await getter is message
    finally:
        if not getter.done():
            getter.cancel()
            await asyncio.gather(getter, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_inbound_puts_are_consumed_in_their_actual_admission_order() -> None:
    bus = MessageBus()
    snapshots: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(snapshots.append)
    all_ready = asyncio.Event()
    start = asyncio.Event()
    ready_count = 0

    async def put_message(message: InboundMessage) -> None:
        nonlocal ready_count
        ready_count += 1
        if ready_count == 10:
            all_ready.set()
        await start.wait()
        await bus.put_inbound(message)

    messages = [InboundMessage(content=str(index)) for index in range(10)]
    tasks = [asyncio.create_task(put_message(message)) for message in messages]
    await all_ready.wait()
    start.set()
    await asyncio.gather(*tasks)

    admitted = [snapshot[-1] for snapshot in snapshots]
    drained = await bus.drain_inbound()

    assert len(admitted) == 10
    assert len({message.content for message in admitted}) == 10
    assert drained == tuple(admitted)
    assert len({message.content for message in drained}) == 10


@pytest.mark.asyncio
async def test_outbound_messages_are_an_unbounded_single_consumer_fifo() -> None:
    bus = MessageBus()
    messages = [OutboundMessage(type="model_response", content=str(index)) for index in range(100)]

    for message in messages:
        await bus.put_outbound(message)

    for message in messages:
        assert await bus.get_outbound() is message
