"""Runtime Lifetime coordination for transient Tool Confirmation requests."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from myclaw.agent.tools.tool_gateway import ConfirmationRequest


type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOrigin = Literal["foreground", "background"]


@dataclass(frozen=True, slots=True)
class ForegroundConfirmationOwner:
    """Identity of one foreground Agent Run in one Runtime Generation."""

    generation_id: UUID
    run_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, UUID) or not isinstance(self.run_id, UUID):
            raise TypeError("foreground confirmation owner ids must be UUIDs")


@dataclass(frozen=True, slots=True)
class BackgroundConfirmationOwner:
    """Identity of one background occurrence in one Runtime Generation."""

    generation_id: UUID
    job_id: str
    occurrence_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.generation_id, UUID) or not isinstance(self.occurrence_id, UUID):
            raise TypeError("background confirmation owner ids must be UUIDs")
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("background confirmation job_id must be non-empty")


type ConfirmationOwner = ForegroundConfirmationOwner | BackgroundConfirmationOwner


@dataclass(frozen=True, slots=True)
class ConfirmationEnvelope:
    """Runtime-only presentation data for one exact normalized Tool call."""

    request: ConfirmationRequest
    origin: ConfirmationOrigin
    owner: ConfirmationOwner
    job_id: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        from myclaw.agent.tools.tool_gateway import (
            ConfirmationRequest as RuntimeConfirmationRequest,
        )

        request = self.request
        if not isinstance(request, RuntimeConfirmationRequest):
            raise TypeError("confirmation envelope requires a normalized request")
        if self.origin == "foreground":
            if not isinstance(self.owner, ForegroundConfirmationOwner):
                raise TypeError("foreground confirmations require a foreground owner")
            if self.job_id is not None or self.title is not None:
                raise ValueError("foreground confirmations cannot carry background source data")
        elif self.origin == "background":
            if not isinstance(self.owner, BackgroundConfirmationOwner):
                raise TypeError("background confirmations require a background owner")
            if self.job_id != self.owner.job_id:
                raise ValueError("background confirmation job_id must match its owner")
            if not isinstance(self.title, str) or not self.title.strip():
                raise ValueError("background confirmations require a non-empty title")
        else:
            raise ValueError("confirmation origin is invalid")

    @property
    def confirmation_id(self) -> UUID:
        return self.request.confirmation_id

    @property
    def tool_call_id(self) -> str:
        return self.request.tool_call_id

    @property
    def tool_name(self) -> str:
        return self.request.tool_name

    @property
    def reason(self) -> str:
        return self.request.reason

    @property
    def summary(self) -> str:
        return self.request.summary

    @property
    def details(self) -> dict[str, Any]:
        return self.request.details

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.request.warnings

    @property
    def mcp_identity(self) -> Any:
        return self.request.mcp_identity


class ConfirmationAborted(Exception):
    """A lifecycle action aborted a pending confirmation before a user decision."""


class ConfirmationUnavailable(Exception):
    """No presenter was available to safely display a confirmation request."""


class ConfirmationPresenter(Protocol):
    """Narrow adapter contract for one currently bound presenter."""

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None | Awaitable[None]: ...

    def dismiss_confirmation(self, token: object) -> None | Awaitable[None]: ...


class ConfirmationPresentationCoordinator(Protocol):
    """Terminal-facing lifecycle boundary without queue ownership."""

    def bind_presenter(self, presenter: ConfirmationPresenter) -> None: ...

    async def unbind_presenter(self, presenter: ConfirmationPresenter) -> None: ...

    async def cancel_owner(self, owner: ConfirmationOwner) -> None: ...


@dataclass(slots=True)
class _ConfirmationItem:
    envelope: ConfirmationEnvelope
    token: object
    future: asyncio.Future[ConfirmationDecision]
    state: Literal["queued", "active", "finished"] = "queued"


class ToolConfirmationCoordinator:
    """Own one Runtime Lifetime confirmation slot and its two stable FIFOs."""

    def __init__(self) -> None:
        self._presenter: ConfirmationPresenter | None = None
        self._active: _ConfirmationItem | None = None
        self._foreground: deque[_ConfirmationItem] = deque()
        self._background: deque[_ConfirmationItem] = deque()
        self._pump_task: asyncio.Task[None] | None = None
        self._closed = False

    def bind_presenter(self, presenter: ConfirmationPresenter) -> None:
        if self._closed:
            raise ConfirmationAborted("confirmation coordinator is closed")
        if self._presenter is not None and self._presenter is not presenter:
            raise RuntimeError("a confirmation presenter is already bound")
        if not callable(getattr(presenter, "present_confirmation", None)):
            raise TypeError("confirmation presenter must present requests")
        if not callable(getattr(presenter, "dismiss_confirmation", None)):
            raise TypeError("confirmation presenter must dismiss requests")
        self._presenter = presenter

    async def unbind_presenter(self, presenter: ConfirmationPresenter) -> None:
        if self._presenter is not presenter:
            return
        current = self._presenter
        self._presenter = None
        active = self._abort_all("confirmation presenter unbound")
        await self._stop_pump()
        if active is not None:
            await self._dismiss(current, active.token)

    async def request(self, envelope: ConfirmationEnvelope) -> ConfirmationDecision:
        if self._closed:
            raise ConfirmationAborted("confirmation coordinator is closed")
        if self._presenter is None:
            raise ConfirmationUnavailable("confirmation presenter is not bound")

        item = _ConfirmationItem(
            envelope=envelope,
            token=object(),
            future=asyncio.get_running_loop().create_future(),
        )
        if envelope.origin == "foreground":
            self._foreground.append(item)
        else:
            self._background.append(item)
        self._ensure_pump()
        try:
            return await item.future
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cancel_item_for_producer(item))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise

    def decide(self, token: object, decision: ConfirmationDecision) -> bool:
        if decision not in {"approved", "declined"}:
            return False
        item = self._active
        if item is None or item.token is not token or item.state != "active":
            return False
        item.state = "finished"
        self._active = None
        if not item.future.done():
            item.future.set_result(decision)
        self._ensure_pump()
        return True

    async def cancel_owner(self, owner: ConfirmationOwner) -> None:
        if not isinstance(owner, (ForegroundConfirmationOwner, BackgroundConfirmationOwner)):
            raise TypeError("confirmation owner is invalid")
        await self._cancel_matching(lambda item: item.envelope.owner == owner)

    async def cancel_generation(self, generation_id: UUID) -> None:
        if not isinstance(generation_id, UUID):
            raise TypeError("confirmation generation id must be a UUID")
        await self._cancel_matching(
            lambda item: item.envelope.owner.generation_id == generation_id,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        presenter = self._presenter
        self._presenter = None
        active = self._abort_all("confirmation coordinator closed")
        await self._stop_pump()
        if presenter is not None and active is not None:
            await self._dismiss(presenter, active.token)

    async def _cancel_matching(self, predicate: Callable[[_ConfirmationItem], bool]) -> None:
        active = self._active
        if active is not None and predicate(active):
            presenter = self._presenter
            self._active = None
            self._finish_with_exception(active, ConfirmationAborted("confirmation lifecycle cancelled"))
            await self._stop_pump()
            if presenter is not None:
                await self._dismiss(presenter, active.token)

        for queue in (self._foreground, self._background):
            retained: deque[_ConfirmationItem] = deque()
            while queue:
                item = queue.popleft()
                if predicate(item):
                    self._finish_with_exception(
                        item,
                        ConfirmationAborted("confirmation lifecycle cancelled"),
                    )
                else:
                    retained.append(item)
            queue.extend(retained)
        self._ensure_pump()

    async def _cancel_item_for_producer(self, item: _ConfirmationItem) -> None:
        if item.state == "finished":
            return
        active = self._active is item
        if active:
            presenter = self._presenter
            self._active = None
            item.state = "finished"
            await self._stop_pump()
            if presenter is not None:
                await self._dismiss(presenter, item.token)
        else:
            self._remove_from_queue(item)
            item.state = "finished"
        if not item.future.done():
            item.future.cancel()
        self._ensure_pump()

    def _remove_from_queue(self, item: _ConfirmationItem) -> None:
        for queue in (self._foreground, self._background):
            for index, queued in enumerate(queue):
                if queued is item:
                    del queue[index]
                    return

    def _ensure_pump(self) -> None:
        if (
            self._closed
            or self._presenter is None
            or self._active is not None
            or not (self._foreground or self._background)
        ):
            return
        task = self._pump_task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._pump())
        self._pump_task = task
        task.add_done_callback(self._pump_finished)

    async def _pump(self) -> None:
        while self._active is None and not self._closed:
            presenter = self._presenter
            if presenter is None:
                return
            item = self._next_item()
            if item is None:
                return
            if item.future.done():
                item.state = "finished"
                continue
            self._active = item
            item.state = "active"
            try:
                result = presenter.present_confirmation(
                    item.envelope,
                    item.token,
                    self.decide,
                )
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                if self._active is not item:
                    raise
                self._active = None
                self._finish_with_exception(
                    item,
                    ConfirmationUnavailable("confirmation presenter failed"),
                )
            except BaseException:
                if self._active is item:
                    self._active = None
                    self._finish_with_exception(
                        item,
                        ConfirmationUnavailable("confirmation presenter failed"),
                    )

    def _next_item(self) -> _ConfirmationItem | None:
        if self._foreground:
            return self._foreground.popleft()
        if self._background:
            return self._background.popleft()
        return None

    def _pump_finished(self, task: asyncio.Task[None]) -> None:
        if self._pump_task is task:
            self._pump_task = None
        with suppress(BaseException):
            task.result()

    async def _stop_pump(self) -> None:
        task = self._pump_task
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with suppress(BaseException):
            await task
        if self._pump_task is task:
            self._pump_task = None

    def _abort_all(self, message: str) -> _ConfirmationItem | None:
        active = self._active
        self._active = None
        items = ([] if active is None else [active]) + list(self._foreground) + list(
            self._background
        )
        self._foreground.clear()
        self._background.clear()
        for item in items:
            self._finish_with_exception(item, ConfirmationAborted(message))
        return active

    @staticmethod
    def _finish_with_exception(item: _ConfirmationItem, error: Exception) -> None:
        item.state = "finished"
        if not item.future.done():
            item.future.set_exception(error)

    @staticmethod
    async def _dismiss(presenter: ConfirmationPresenter, token: object) -> None:
        try:
            result = presenter.dismiss_confirmation(token)
            if inspect.isawaitable(result):
                await result
        except BaseException:
            # A lifecycle cancellation must not be held hostage by a broken UI.
            return


class CallbackConfirmationRequester:
    """Compatibility adapter for tests and non-coordinator control surfaces."""

    def __init__(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback
        self._request: Any | None = None
        self._future: asyncio.Future[ConfirmationDecision] | None = None

    async def request(self, request: Any) -> ConfirmationDecision:
        if self._request is not None:
            raise RuntimeError("A foreground confirmation request is already pending")
        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        self._request = request
        self._future = future
        try:
            self._callback(request)
            return await future
        finally:
            if self._request is request:
                self._request = None
                self._future = None

    def respond(self, confirmation_id: UUID, decision: ConfirmationDecision) -> None:
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        request = self._request
        future = self._future
        if (
            request is None
            or future is None
            or request.confirmation_id != confirmation_id
            or future.done()
        ):
            raise ValueError("Confirmation response is late or unknown")
        future.set_result(decision)

    def cancel(self) -> None:
        future = self._future
        if future is not None and not future.done():
            future.cancel()

    def unbind(self, callback: Callable[[Any], None]) -> None:
        if self._callback is callback:
            self._callback = lambda _request: None
            self.cancel()


__all__ = [
    "BackgroundConfirmationOwner",
    "CallbackConfirmationRequester",
    "ConfirmationAborted",
    "ConfirmationDecision",
    "ConfirmationEnvelope",
    "ConfirmationOrigin",
    "ConfirmationOwner",
    "ConfirmationPresentationCoordinator",
    "ConfirmationPresenter",
    "ConfirmationUnavailable",
    "ForegroundConfirmationOwner",
    "ToolConfirmationCoordinator",
]
