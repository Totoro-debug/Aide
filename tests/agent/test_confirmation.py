from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from aide.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationAborted,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationUnavailable,
    ForegroundConfirmationOwner,
    ToolConfirmationCoordinator,
)
from aide.agent.tools.tool_gateway import ConfirmationRequest

GENERATION_ONE = UUID("00000000-0000-4000-8000-000000000001")
GENERATION_TWO = UUID("00000000-0000-4000-8000-000000000002")


def _request(call_id: str) -> ConfirmationRequest:
    return ConfirmationRequest(
        confirmation_id=uuid4(),
        tool_call_id=call_id,
        tool_name="write_file",
        reason="test confirmation",
        summary="Confirm write",
        details={"path": "note.txt", "content": "x"},
    )


def _foreground(call_id: str, *, generation_id: UUID = GENERATION_ONE) -> ConfirmationEnvelope:
    return ConfirmationEnvelope(
        request=_request(call_id),
        origin="foreground",
        owner=ForegroundConfirmationOwner(
            generation_id=generation_id,
            run_id=uuid4(),
        ),
    )


def _background(call_id: str, *, generation_id: UUID = GENERATION_ONE) -> ConfirmationEnvelope:
    return ConfirmationEnvelope(
        request=_request(call_id),
        origin="background",
        owner=BackgroundConfirmationOwner(
            generation_id=generation_id,
            job_id="job-1",
            occurrence_id=uuid4(),
        ),
        job_id="job-1",
        title="Nightly backup",
    )


class RecordingPresenter:
    def __init__(self) -> None:
        self.presented: list[tuple[ConfirmationEnvelope, object]] = []
        self.responders: dict[object, Callable[[ConfirmationDecision], bool]] = {}
        self.dismissed: list[object] = []
        self.changed = asyncio.Event()

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None | Awaitable[None]:
        self.presented.append((envelope, token))
        self.responders[token] = lambda decision: respond(token, decision)
        self.changed.set()
        return None

    async def dismiss_confirmation(self, token: object) -> None:
        self.dismissed.append(token)

    async def wait_for_count(self, count: int) -> None:
        while len(self.presented) < count:
            self.changed.clear()
            await self.changed.wait()

    def decide(self, index: int, decision: ConfirmationDecision) -> bool:
        token = self.presented[index][1]
        return self.responders[token](decision)


class BlockingAsyncPresenter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.dismissed: list[object] = []
        self.token: object | None = None

    async def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None:
        del envelope, respond
        self.token = token
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()

    async def dismiss_confirmation(self, token: object) -> None:
        self.dismissed.append(token)


class FailingOncePresenter(RecordingPresenter):
    def __init__(self, *, asynchronously: bool) -> None:
        super().__init__()
        self._asynchronously = asynchronously
        self._failed = False

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None | Awaitable[None]:
        if not self._failed:
            self._failed = True
            if self._asynchronously:
                return self._fail_asynchronously()
            raise RuntimeError("presenter failed")
        return super().present_confirmation(envelope, token, respond)

    @staticmethod
    async def _fail_asynchronously() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("presenter failed")


def _owners() -> tuple[ForegroundConfirmationOwner, BackgroundConfirmationOwner]:
    return (
        ForegroundConfirmationOwner(GENERATION_ONE, uuid4()),
        BackgroundConfirmationOwner(GENERATION_ONE, "job-1", uuid4()),
    )


def _assert_coordinator_idle(coordinator: ToolConfirmationCoordinator) -> None:
    assert coordinator._active is None
    assert not coordinator._foreground
    assert not coordinator._background


def test_owner_union_and_envelope_are_typed_runtime_values() -> None:
    foreground, background = _owners()

    assert hash(foreground) == hash(foreground)
    assert hash(background) == hash(background)
    assert not isinstance(foreground, BackgroundConfirmationOwner)

    envelope = ConfirmationEnvelope(
        request=_request("call-1"),
        origin="background",
        owner=background,
        job_id="job-1",
        title="Nightly backup",
    )

    assert envelope.request.details == {"path": "note.txt", "content": "x"}
    assert envelope.job_id == "job-1"
    assert envelope.title == "Nightly backup"
    assert not hasattr(envelope, "to_dict")

    request = _request("duck")
    duck_request = SimpleNamespace(
        confirmation_id=request.confirmation_id,
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        reason=request.reason,
        summary=request.summary,
        details=request.details,
        warnings=request.warnings,
        mcp_identity=request.mcp_identity,
    )
    with pytest.raises(TypeError, match="normalized request"):
        ConfirmationEnvelope(
            request=cast(ConfirmationRequest, duck_request),
            origin="foreground",
            owner=foreground,
        )


@pytest.mark.asyncio
async def test_active_foreground_does_not_preempt_background_and_foreground_is_prioritized() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    foreground = asyncio.create_task(coordinator.request(_foreground("foreground-1")))
    await presenter.wait_for_count(1)
    background = asyncio.create_task(coordinator.request(_background("background-1")))
    await asyncio.sleep(0)
    assert len(presenter.presented) == 1

    presenter.decide(0, "approved")
    await presenter.wait_for_count(2)
    assert presenter.presented[1][0].origin == "background"
    presenter.decide(1, "declined")

    assert await foreground == "approved"
    assert await background == "declined"
    await coordinator.close()


@pytest.mark.asyncio
async def test_active_background_finishes_before_queued_foreground_then_background() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    active_background = asyncio.create_task(coordinator.request(_background("background-1")))
    await presenter.wait_for_count(1)
    queued_background = asyncio.create_task(coordinator.request(_background("background-2")))
    queued_foreground = asyncio.create_task(coordinator.request(_foreground("foreground-1")))
    await asyncio.sleep(0)
    assert len(presenter.presented) == 1

    presenter.decide(0, "approved")
    await presenter.wait_for_count(2)
    assert presenter.presented[1][0].request.tool_call_id == "foreground-1"
    presenter.decide(1, "approved")
    await presenter.wait_for_count(3)
    assert presenter.presented[2][0].request.tool_call_id == "background-2"
    presenter.decide(2, "declined")

    assert await active_background == "approved"
    assert await queued_foreground == "approved"
    assert await queued_background == "declined"
    await coordinator.close()


@pytest.mark.asyncio
async def test_each_queue_keeps_fifo_order() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    tasks = [
        asyncio.create_task(coordinator.request(_foreground("foreground-1"))),
        asyncio.create_task(coordinator.request(_foreground("foreground-2"))),
        asyncio.create_task(coordinator.request(_background("background-1"))),
        asyncio.create_task(coordinator.request(_background("background-2"))),
    ]
    await presenter.wait_for_count(1)
    for index, expected in enumerate(
        ("foreground-1", "foreground-2", "background-1", "background-2")
    ):
        assert presenter.presented[index][0].request.tool_call_id == expected
        presenter.decide(index, "approved")
        if index < 3:
            await presenter.wait_for_count(index + 2)

    assert await asyncio.gather(*tasks) == ["approved"] * 4
    await coordinator.close()


@pytest.mark.asyncio
async def test_producer_cancellation_removes_queued_item_and_active_item_dismisses() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    active = asyncio.create_task(coordinator.request(_foreground("active")))
    await presenter.wait_for_count(1)
    queued = asyncio.create_task(coordinator.request(_background("queued")))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert len(presenter.presented) == 1

    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    await asyncio.sleep(0)
    assert presenter.dismissed == [presenter.presented[0][1]]
    _assert_coordinator_idle(coordinator)
    await coordinator.close()


@pytest.mark.asyncio
async def test_async_presenter_is_stopped_and_producer_cancellation_stays_cancelled() -> None:
    presenter = BlockingAsyncPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    pending = asyncio.create_task(coordinator.request(_foreground("active")))
    await presenter.started.wait()
    token = presenter.token
    assert token is not None
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert presenter.stopped.is_set()
    assert presenter.dismissed == [token]
    _assert_coordinator_idle(coordinator)
    await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronously", (False, True))
async def test_presenter_failures_fail_closed_and_advance_the_queue(
    asynchronously: bool,
) -> None:
    presenter = FailingOncePresenter(asynchronously=asynchronously)
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    failed = asyncio.create_task(coordinator.request(_foreground("failed")))
    next_item = asyncio.create_task(coordinator.request(_foreground("next")))
    await presenter.wait_for_count(1)

    with pytest.raises(ConfirmationUnavailable, match="presenter failed"):
        await failed
    assert presenter.presented[0][0].request.tool_call_id == "next"
    presenter.decide(0, "approved")
    assert await next_item == "approved"
    await coordinator.close()


@pytest.mark.asyncio
async def test_owner_and_generation_cancellation_raise_typed_abort() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    owner = ForegroundConfirmationOwner(GENERATION_ONE, uuid4())
    owner_request = ConfirmationEnvelope(
        request=_request("owner"),
        origin="foreground",
        owner=owner,
    )
    other_request = _foreground("other", generation_id=GENERATION_TWO)
    background_request = _background("background", generation_id=GENERATION_TWO)
    survivor_request = _background("survivor", generation_id=GENERATION_ONE)
    owner_task = asyncio.create_task(coordinator.request(owner_request))
    other_task = asyncio.create_task(coordinator.request(other_request))
    background_task = asyncio.create_task(coordinator.request(background_request))
    survivor_task = asyncio.create_task(coordinator.request(survivor_request))
    await presenter.wait_for_count(1)

    await coordinator.cancel_owner(owner)
    with pytest.raises(ConfirmationAborted):
        await owner_task
    await presenter.wait_for_count(2)
    await coordinator.cancel_generation(GENERATION_TWO)
    with pytest.raises(ConfirmationAborted):
        await other_task
    with pytest.raises(ConfirmationAborted):
        await background_task
    await presenter.wait_for_count(3)
    assert presenter.presented[2][0].request.tool_call_id == "survivor"
    presenter.decide(2, "approved")
    assert await survivor_task == "approved"
    await coordinator.close()


@pytest.mark.asyncio
async def test_duplicate_and_late_decisions_cannot_resolve_a_later_item() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)

    first = asyncio.create_task(coordinator.request(_foreground("first")))
    await presenter.wait_for_count(1)
    stale_token = presenter.presented[0][1]
    second = asyncio.create_task(coordinator.request(_foreground("second")))

    assert presenter.decide(0, "approved") is True
    await presenter.wait_for_count(2)
    assert presenter.responders[stale_token](cast(ConfirmationDecision, "invalid")) is False
    assert presenter.responders[stale_token]("declined") is False
    assert not second.done()
    presenter.decide(1, "declined")

    assert await first == "approved"
    assert await second == "declined"
    await coordinator.close()


@pytest.mark.asyncio
async def test_request_has_no_runtime_timeout_and_no_presenter_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)
    request_task = asyncio.create_task(coordinator.request(_foreground("long-lived")))
    await presenter.wait_for_count(1)
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset)
    offset = 24 * 60 * 60
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not request_task.done()
    presenter.decide(0, "approved")
    assert await request_task == "approved"

    await coordinator.unbind_presenter(presenter)
    with pytest.raises(ConfirmationUnavailable):
        await coordinator.request(_foreground("without-presenter"))
    await coordinator.close()


@pytest.mark.asyncio
async def test_unbind_current_aborts_all_and_non_current_cannot_break_binding() -> None:
    first_presenter = RecordingPresenter()
    second_presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(first_presenter)
    coordinator.bind_presenter(first_presenter)
    with pytest.raises(RuntimeError, match="already bound"):
        coordinator.bind_presenter(second_presenter)

    active = asyncio.create_task(coordinator.request(_foreground("active")))
    queued = asyncio.create_task(coordinator.request(_background("queued")))
    await first_presenter.wait_for_count(1)

    await coordinator.unbind_presenter(second_presenter)
    assert coordinator._active is not None
    await coordinator.unbind_presenter(first_presenter)
    with pytest.raises(ConfirmationAborted):
        await active
    with pytest.raises(ConfirmationAborted):
        await queued
    assert first_presenter.dismissed == [first_presenter.presented[0][1]]

    coordinator.bind_presenter(second_presenter)
    rebound = asyncio.create_task(coordinator.request(_foreground("rebound")))
    await second_presenter.wait_for_count(1)
    second_presenter.decide(0, "approved")
    assert await rebound == "approved"
    await coordinator.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_post_close_requests_fail_closed() -> None:
    presenter = RecordingPresenter()
    coordinator = ToolConfirmationCoordinator()
    coordinator.bind_presenter(presenter)
    request_task = asyncio.create_task(coordinator.request(_foreground("close")))
    await presenter.wait_for_count(1)

    await coordinator.close()
    await coordinator.close()
    with pytest.raises(ConfirmationAborted):
        await request_task
    _assert_coordinator_idle(coordinator)
    with pytest.raises(ConfirmationAborted):
        await coordinator.request(_foreground("after-close"))
