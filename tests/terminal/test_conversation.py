from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal, Never, cast
from uuid import UUID, uuid4
from xml.etree import ElementTree

import pytest
from textual.app import App
from textual.driver import Driver
from textual.events import (
    Key,
    MouseDown,
    MouseMove,
    MouseScrollDown,
    MouseScrollRight,
    MouseScrollUp,
    MouseUp,
    Paste,
)
from textual.message import Message
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea

import aide.terminal.cli as cli
from aide.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationAborted,
    ConfirmationEnvelope,
    ToolConfirmationCoordinator,
)
from aide.agent.loop import (
    ConfirmationRequestView,
    ForegroundConversationProjection,
)
from aide.agent.memory.dream import DreamResult
from aide.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from aide.agent.session.backup_store import BackupGap, FileBackupStore
from aide.agent.session.restore import (
    RestoreFileResult,
    RestoreFileStatus,
    RestoreManager,
    RestoreMode,
    RestorePlan,
    RestoreResult,
    RestoreTarget,
)
from aide.agent.session.session import RestoreAnchor, Session, SessionRestoreResult
from aide.agent.tools.permission import MCPToolIdentity
from aide.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
)
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.management.commands import (
    MANAGEMENT_COMMANDS,
    ManagementCommandDispatcher,
    ManagementCommandResult,
)
from aide.management.service import (
    FatalManagementError,
    ManagementError,
    RestoreListingReport,
    RuntimeStatus,
)
from aide.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ModelStreamEvent,
    ReasoningDelta,
    ReasoningEffort,
    TextDelta,
)
from aide.service.client import RemoteManagementCommandDispatcher
from aide.skills.catalog import SkillMetadata
from aide.templates import render_template
from aide.terminal.conversation import (
    TerminalConversationApp,
    _ConversationInput,
    _format_activity_duration,
    _RestoreConfirmationScreen,
    _RestoreModeScreen,
)
from aide.terminal.conversation import (
    _MessageBusRunProjection as _AgentRunProjection,
)
from aide.utils.errors import ErrorInfo
from aide.utils.host_filesystem import HOST_FILESYSTEM
from aide.utils.json_types import JsonObject
from tests.agent.test_fixed_catalog import _agent_loop as _direct_agent_loop
from tests.agent.test_fixed_catalog import _FixedCatalogProvider, _response
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ProviderCall, TaskFramingRouterAdapter
from tests.fixtures.agent_loop import DrivenExecutor as AgentRunExecutor
from tests.fixtures.cli_service import cli_service
from tests.fixtures.session import seed_session_state

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
TURN_ID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
OTHER_TURN_ID = UUID("b3fbe13b-3d1f-4ee6-98e5-dc40a2c4be1c")


@dataclass(frozen=True, slots=True)
class _ResponseSegmentScript:
    content: str


type _ScriptItem = OutboundMessage | ConfirmationRequest | _ResponseSegmentScript


class _ScriptedSource:
    def run(self, text: str) -> AsyncIterator[_ScriptItem]:
        raise NotImplementedError

    async def cancel_active_turn(self) -> bool:
        return False


class ScriptedRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        pause_before_output: bool = False,
        pause_after_first_delta: bool = False,
        deltas: tuple[str, ...] = ("First ", "answer."),
        completed_content: str = "First answer.",
        deltas_by_submission: tuple[tuple[str, ...], ...] | None = None,
        completed_contents: tuple[str, ...] | None = None,
        outcomes: tuple[Literal["completed", "cancelled", "failed"], ...] = ("completed",),
        cancelled_content: str = "",
        failure_message: str = "The turn failed.",
    ) -> None:
        self.submissions: list[str] = []
        self._deltas = deltas
        self._completed_content = completed_content
        self._deltas_by_submission = deltas_by_submission
        self._completed_contents = completed_contents
        self._outcomes = outcomes
        self._cancelled_content = cancelled_content
        self._failure_message = failure_message
        self.before_output = asyncio.Event()
        self.first_delta_emitted = asyncio.Event()
        self._continue_to_output = asyncio.Event()
        self._continue_after_first_delta = asyncio.Event()
        if not pause_before_output:
            self._continue_to_output.set()
        if not pause_after_first_delta:
            self._continue_after_first_delta.set()

    def continue_to_output(self) -> None:
        self._continue_to_output.set()

    def continue_turn(self) -> None:
        self._continue_after_first_delta.set()

    def pause_after_next_first_delta(self) -> None:
        self.first_delta_emitted.clear()
        self._continue_after_first_delta.clear()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        outcome = self._outcomes[min(submission_index, len(self._outcomes) - 1)]
        deltas = (
            self._deltas
            if self._deltas_by_submission is None
            else self._deltas_by_submission[
                min(submission_index, len(self._deltas_by_submission) - 1)
            ]
        )
        completed_content = (
            self._completed_content
            if self._completed_contents is None
            else self._completed_contents[min(submission_index, len(self._completed_contents) - 1)]
        )
        self.before_output.set()
        await self._continue_to_output.wait()
        for output_index, delta in enumerate(deltas, start=1):
            yield _response_delta(delta)
            if output_index == 1:
                self.first_delta_emitted.set()
                await self._continue_after_first_delta.wait()
        if outcome == "completed":
            if not deltas and completed_content:
                yield _response_delta(completed_content)
            yield OutboundMessage("model_response", "", {"_stream_end": True})
            yield _completed_response()
        elif outcome == "cancelled":
            yield _cancelled_response(self._cancelled_content)
        else:
            yield _failed_response(self._failure_message)


class ToolActivityRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        start_summary: str = "Running read_file",
    ) -> None:
        self.submissions: list[str] = []
        self.tool_started = asyncio.Event()
        self.complete_tool = asyncio.Event()
        self._start_summary = start_summary

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        yield _tool_call("call-read-file", "read_file", self._start_summary)
        self.tool_started.set()
        await self.complete_tool.wait()
        yield OutboundMessage(
            "tool_call", "read_file", {"tool_call_id": "call-read-file", "status": "success"}
        )
        yield _completed_response()


class ConfirmationRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        tool_name: str = "read_file",
        details: dict[str, Any] | None = None,
        reason: str = "The requested path is outside the current Workspace.",
        warnings: tuple[str, ...] = (),
        mcp_identity: MCPToolIdentity | None = None,
    ) -> None:
        self.submissions: list[str] = []
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.cancel_calls = 0
        self.confirmation_requested = asyncio.Event()
        self._decision_received = asyncio.Event()
        self._request = ConfirmationRequest(
            confirmation_id=UUID("16fd2706-8baf-4334-8c7f-ada847da0314"),
            tool_call_id="call-confirm",
            tool_name=tool_name,
            summary=f"Confirm {tool_name}",
            details=({"path": "outside.txt"} if details is None else cast(JsonObject, details)),
            warnings=warnings,
            reason=reason,
            mcp_identity=mcp_identity,
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        self.submissions.append(text)
        yield _tool_call("call-confirm", self._request.tool_name, "Running")
        self.confirmation_requested.set()
        yield self._request
        await self._decision_received.wait()
        decision = self.responses[-1][1]
        del decision
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        assert confirmation_id == self._request.confirmation_id
        self.responses.append((confirmation_id, decision))
        self._decision_received.set()

    async def cancel_active_turn(self) -> bool:
        self.cancel_calls += 1
        return False


class DuplicateLateConfirmationRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.confirmation_requested = asyncio.Event()
        self.request = ConfirmationRequest(
            confirmation_id=UUID("4c9b7d40-8b5d-4a17-8140-0ce4f3511ab1"),
            tool_call_id="call-stale",
            tool_name="read_file",
            summary="Confirm read_file",
            details={"path": "outside.txt"},
            reason="The path resolves outside the current Workspace.",
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        del text
        self.confirmation_requested.set()
        yield self.request
        yield self.request
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        self.responses.append((confirmation_id, decision))
        raise ValueError("Confirmation response is late or unknown")

    async def cancel_active_turn(self) -> bool:
        return False


class MultipleConfirmationRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.confirmation_requested = (asyncio.Event(), asyncio.Event())
        self.requests = (
            ConfirmationRequest(
                confirmation_id=UUID("b378d47d-2d73-4670-badc-844245c63c3d"),
                tool_call_id="call-first",
                tool_name="read_file",
                summary="Confirm read_file",
                details={"path": "first.txt"},
                reason="First confirmation.",
            ),
            ConfirmationRequest(
                confirmation_id=UUID("3f1bb452-a8cf-4760-9cde-77b6a4b80ae9"),
                tool_call_id="call-second",
                tool_name="write_file",
                summary="Confirm write_file",
                details={"path": "second.txt", "content": "private"},
                reason="Second confirmation.",
            ),
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        del text
        for requested, request in zip(self.confirmation_requested, self.requests, strict=True):
            requested.set()
            yield request
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        self.responses.append((confirmation_id, decision))

    async def cancel_active_turn(self) -> bool:
        return False


class ToolMessageSequenceRunSource(_ScriptedSource):
    def __init__(
        self,
        *sequences: tuple[_ScriptItem, ...],
        pause_before: bool = False,
        pause_after: tuple[int, int] | None = None,
    ) -> None:
        self.submissions: list[str] = []
        self._sequences = sequences
        self._pause_before = pause_before
        self._pause_after = pause_after
        self.paused = asyncio.Event()
        self.continue_events = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        sequence = self._sequences[submission_index]
        response_delta_seen = False
        if self._pause_before:
            self.paused.set()
            await self.continue_events.wait()
        for message_index, script_item in enumerate(sequence):
            messages, response_delta_seen = _expand_script_item(script_item, response_delta_seen)
            for message in messages:
                yield message
            if self._pause_after == (submission_index, message_index):
                self.paused.set()
                await self.continue_events.wait()


class StagedToolMessageSequenceRunSource(_ScriptedSource):
    def __init__(
        self,
        *sequences: tuple[_ScriptItem, ...],
        pause_after: tuple[tuple[int, int], ...],
    ) -> None:
        self.submissions: list[str] = []
        self._sequences = sequences
        self._checkpoints = {
            checkpoint: (asyncio.Event(), asyncio.Event()) for checkpoint in pause_after
        }

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        response_delta_seen = False
        for message_index, script_item in enumerate(self._sequences[submission_index]):
            messages, response_delta_seen = _expand_script_item(script_item, response_delta_seen)
            for message in messages:
                yield message
            checkpoint = self._checkpoints.get((submission_index, message_index))
            if checkpoint is not None:
                reached, continue_events = checkpoint
                reached.set()
                await continue_events.wait()

    async def wait_after(self, submission_index: int, message_index: int) -> None:
        reached, _ = self._checkpoints[(submission_index, message_index)]
        await asyncio.wait_for(reached.wait(), timeout=1)

    def continue_after(self, submission_index: int, message_index: int) -> None:
        _, continue_events = self._checkpoints[(submission_index, message_index)]
        continue_events.set()

    def continue_all(self) -> None:
        for _, continue_events in self._checkpoints.values():
            continue_events.set()


class CancellableRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.submissions: list[str] = []
        self.cancel_calls = 0
        self.first_delta_emitted = asyncio.Event()
        self._active_task: asyncio.Task[object] | None = None

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        active_task = asyncio.current_task()
        assert active_task is not None
        self._active_task = active_task
        try:
            if len(self.submissions) == 1:
                yield _response_delta("partial response")
                self.first_delta_emitted.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    yield _cancelled_response("partial response")
                    return
            else:
                yield _response_delta("Recovered response.")
                yield _completed_response()
        finally:
            self._active_task = None

    async def cancel_active_turn(self) -> bool:
        self.cancel_calls += 1
        if self._active_task is not None:
            self._active_task.cancel()
        return True


class CancellableProvider(_FixedCatalogProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.first_delta_emitted = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._chat_calls = 0

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if messages and messages[0] == {
            "role": "system",
            "content": render_template("session-title-prompt.md"),
        }:
            async for event in super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            ):
                yield event
            return
        self.stream_requests.append(
            ProviderCall(
                messages=list(messages),
                tools=tuple(tools),
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
        )
        self._chat_calls += 1
        if self._chat_calls == 1:
            yield TextDelta(delta="partial runtime response")
            self.first_delta_emitted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return
        yield ModelCompleted(response=_response(content="Recovered runtime response."))


class FailingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("partial")
        finally:
            self.closed.set()


class ExplodingRunSource(_ScriptedSource):
    def __init__(self, *, include_tool: bool = True) -> None:
        self.closed = asyncio.Event()
        self.include_tool = include_tool

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("Unconfirmed candidate.")
            if self.include_tool:
                yield _tool_call("exploding-call", "read_file", "Running read_file")
            raise RuntimeError("event stream failed")
        finally:
            self.closed.set()


class TerminalThenBlockingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.terminal_emitted = asyncio.Event()
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("process activity")
            yield _tool_call("unfinished-call", "read_file", "Running read_file")
            self.terminal_emitted.set()
            yield _response_delta("final response")
            yield _completed_response()
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class BlockingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            self.started.set()
            yield _response_delta("partial")
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class FailingCancellationCleanupRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            self.started.set()
            yield _response_delta("partial")
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("worker cleanup failed")


class FailingMarkdownStream:
    async def write(self, markdown_fragment: str) -> None:
        del markdown_fragment
        raise RuntimeError("markdown write failed")

    async def stop(self) -> None:
        raise RuntimeError("markdown stop failed")


class _ScriptedControl:
    """Final AgentRunExecutor control surface for a scripted MessageBus runtime."""

    def __init__(self, source: _ScriptedSource) -> None:
        self._source = source
        self._active = False
        self._pending_confirmation: UUID | None = None
        self._callback: Callable[[ConfirmationRequestView], None] | None = None
        self._seen_confirmation_ids: set[UUID] = set()
        self._confirmation_released: asyncio.Event | None = None
        self._bridge: asyncio.Task[None] | None = None

    @property
    def has_active_run(self) -> bool:
        return self._active

    def foreground_input_admitted(self) -> bool:
        return True

    def bind_confirmation_callback(
        self, callback: Callable[[ConfirmationRequestView], None]
    ) -> None:
        self._callback = callback

    async def cancel_active_run(self) -> None:
        handled = await self._source.cancel_active_turn()
        if handled:
            return
        bridge = self._bridge
        if bridge is not None and not bridge.done():
            bridge.cancel()
            await asyncio.gather(bridge, return_exceptions=True)

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return ForegroundConversationProjection(session_id="test-session", messages=())

    def respond_to_confirmation(
        self, confirmation_id: UUID, decision: ConfirmationDecision
    ) -> None:
        try:
            respond = getattr(self._source, "respond_to_confirmation", None)
            if callable(respond):
                cast(Callable[[UUID, ConfirmationDecision], None], respond)(
                    confirmation_id, decision
                )
        finally:
            if self._pending_confirmation == confirmation_id:
                self._pending_confirmation = None
                released = self._confirmation_released
                self._confirmation_released = None
                if released is not None:
                    released.set()

    async def publish(
        self,
        item: _ScriptItem,
        bus: MessageBus,
    ) -> bool:
        if isinstance(item, ConfirmationRequest):
            if item.confirmation_id in self._seen_confirmation_ids:
                return False
            self._seen_confirmation_ids.add(item.confirmation_id)
            self._pending_confirmation = item.confirmation_id
            released = asyncio.Event()
            self._confirmation_released = released
            callback = self._callback
            if callback is None:
                raise AssertionError("confirmation callback was not bound")
            callback(item)
            await released.wait()
            return False
        if isinstance(item, _ResponseSegmentScript):
            raise TypeError("response segment scripts must be expanded by a run source")
        await bus.put_outbound(item)
        if item.type == "model_response":
            return item.metadata.get("_streamed") is True
        if item.type == "system_control":
            return item.metadata.get("_streamed") is True
        return False


class _UnavailableManagement:
    @staticmethod
    def _unavailable() -> Never:
        raise ManagementError(ErrorInfo("route_unavailable", "Runtime Generation is unavailable."))

    async def config_view(self) -> Any:
        self._unavailable()

    async def status(self) -> Any:
        self._unavailable()

    async def memory_view(self) -> str:
        self._unavailable()

    async def dream(self) -> DreamResult:
        self._unavailable()

    async def resumable_listing(self) -> Any:
        self._unavailable()

    async def resume(self, _session_id: str, *, force: bool = False) -> Any:
        del force
        self._unavailable()


def _management_dispatcher() -> ManagementCommandDispatcher:
    return ManagementCommandDispatcher(cast(Any, _UnavailableManagement()))


class FakeTerminalBackend:
    def __init__(self, source: _ScriptedSource) -> None:
        self.source = source
        self.bus = MessageBus()
        self.inbound_history: list[InboundMessage] = []
        self.outbound_history: list[OutboundMessage] = []
        put_outbound = self.bus.put_outbound

        async def record_outbound(message: OutboundMessage) -> None:
            self.outbound_history.append(message)
            await put_outbound(message)

        self.bus.put_outbound = record_outbound  # type: ignore[method-assign]
        self.control = _ScriptedControl(source)
        self.management_dispatcher = _management_dispatcher()
        self.session_id = "test-session"
        self.start_calls = 0
        self.close_calls = 0
        self._closing = False
        self._bridge: asyncio.Task[None] | None = None

    async def _consume_inbound(self) -> None:
        while not self._closing:
            message = await self.bus.get_inbound()
            self.inbound_history.append(message)
            self.control._active = True
            self.control._bridge = asyncio.current_task()
            terminal = False
            try:
                async for item in self.source.run(message.content):
                    terminal = await self.control.publish(item, self.bus) or terminal
            except asyncio.CancelledError:
                if self._closing:
                    raise
                if not terminal:
                    await self.bus.put_outbound(
                        OutboundMessage(
                            type="system_control",
                            content="Turn cancelled.",
                            metadata={"_streamed": True, "finish_reason": "cancelled"},
                        )
                    )
                terminal = True
            except Exception as error:
                if not terminal:
                    await self.bus.put_outbound(
                        OutboundMessage(
                            type="system_control",
                            content=str(error),
                            metadata={"_streamed": True, "finish_reason": "failed"},
                        )
                    )
                terminal = True
            finally:
                self.control._active = False
                self.control._bridge = None
            if not terminal:
                await self.bus.put_outbound(
                    OutboundMessage(
                        type="system_control",
                        content="Turn failed.",
                        metadata={"_streamed": True, "finish_reason": "failed"},
                    )
                )

    async def start(self) -> None:
        self.start_calls += 1
        self._bridge = asyncio.create_task(self._consume_inbound())

    def bind_confirmation_callback(
        self, callback: Callable[[ConfirmationRequestView], None]
    ) -> None:
        self.control.bind_confirmation_callback(callback)

    async def close(self) -> None:
        self.close_calls += 1
        self._closing = True
        if self._bridge is not None:
            self._bridge.cancel()
            with suppress(asyncio.CancelledError):
                await self._bridge

    def session_messages(self) -> tuple[dict[str, object], ...]:
        return ()


class _TerminalTestDriver:
    """Keep test backend ownership outside the Textual application lifecycle."""

    def __init__(
        self,
        app: TerminalConversationApp,
        runtime: Any,
        cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._app = app
        self._runtime = runtime
        self._cleanup = cleanup

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_app", "_runtime", "_cleanup"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._app, name, value)

    async def _close_backend(self) -> None:
        await self._runtime.close()
        if self._cleanup is not None:
            await self._cleanup()

    @asynccontextmanager
    async def run_test(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        primary_error: BaseException | None = None
        try:
            await self._runtime.start()
            async with self._app.run_test(*args, **kwargs) as pilot:
                yield pilot
        except BaseException as error:
            primary_error = error
        finally:
            try:
                await self._close_backend()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    cause = primary_error.__cause__
                    if cause is None:
                        raise primary_error from cleanup_error
                    raise primary_error from BaseExceptionGroup(
                        "Terminal Conversation cleanup failed",
                        (cause, cleanup_error),
                    )
                raise
        if primary_error is not None:
            raise primary_error

    async def run_async(self, *args: Any, **kwargs: Any) -> None:
        primary_error: BaseException | None = None
        try:
            await self._runtime.start()
            await self._app.run_async(*args, **kwargs)
        except BaseException as error:
            primary_error = error
        finally:
            try:
                await self._close_backend()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    raise primary_error from cleanup_error
                raise
        if primary_error is not None:
            raise primary_error


class _DirectControl:
    """Minimal public AgentRunExecutor control seam for direct MessageBus UI tests."""

    def __init__(self) -> None:
        self.confirmation_callback: Callable[[ConfirmationRequestView], None] | None = None

    @property
    def has_active_run(self) -> bool:
        return False

    def foreground_input_admitted(self) -> bool:
        return True

    async def cancel_active_run(self) -> None:
        return None

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return ForegroundConversationProjection(session_id="direct-test", messages=())

    def bind_confirmation_callback(
        self,
        callback: Callable[[ConfirmationRequestView], None],
    ) -> None:
        self.confirmation_callback = callback

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        del confirmation_id, decision
        raise ValueError("Confirmation response is late or unknown")


class KeyboardLifecycleDriver(Driver):
    operations: ClassVar[list[tuple[str, str]]] = []

    def write(self, data: str) -> None:
        self.operations.append(("write", data))

    def flush(self) -> None:
        self.operations.append(("flush", ""))

    def start_application_mode(self) -> None:
        self.write("application:start")
        self.write("\x1b[>1u")
        self.flush()

    def disable_input(self) -> None:
        self.operations.append(("disable_input", ""))

    def stop_application_mode(self) -> None:
        self.write("\x1b[<u")
        self.write("application:stop")
        self.flush()

    def close(self) -> None:
        self.operations.append(("close", ""))


class CloseOrderingBackend(FakeTerminalBackend):
    def __init__(self, source: BlockingRunSource) -> None:
        super().__init__(source)
        self._blocking_source = source
        self.close_saw_stream_closed = False

    async def close(self) -> None:
        await super().close()
        self.close_saw_stream_closed = self._blocking_source.closed.is_set()


class FailingStartBackend(FakeTerminalBackend):
    async def start(self) -> None:
        self.start_calls += 1
        raise RuntimeError("runtime startup failed")


class FailingCloseBackend(FakeTerminalBackend):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("runtime cleanup failed")


def _terminal_backend(
    source: _ScriptedSource,
    management_dispatcher: ManagementCommandDispatcher | None = None,
) -> FakeTerminalBackend:
    backend = FakeTerminalBackend(source)
    if management_dispatcher is not None:
        backend.management_dispatcher = management_dispatcher
    return backend


def _direct_terminal_loop(
    agent_home: Path,
    workspace: Path,
    provider: _FixedCatalogProvider,
) -> tuple[AgentRunExecutor, MessageBus]:
    loop, router, schedule, bus = _direct_agent_loop(agent_home, workspace, provider)

    async def close_components() -> None:
        await schedule.close()
        await router.close()

    object.__setattr__(loop, "_terminal_test_cleanup", close_components)
    return loop, bus


def _terminal_app(
    runtime: Any,
    *,
    bus: MessageBus | None = None,
    app_type: type[TerminalConversationApp] = TerminalConversationApp,
    monotonic: Callable[[], float] | None = None,
    skill_metadata: tuple[SkillMetadata, ...] = (),
    management_dispatcher: ManagementCommandDispatcher | None = None,
    cleanup: Callable[[], Awaitable[None]] | None = None,
    confirmation_coordinator: ToolConfirmationCoordinator | None = None,
) -> TerminalConversationApp:
    dispatcher = (
        getattr(runtime, "management_dispatcher", _management_dispatcher())
        if management_dispatcher is None
        else management_dispatcher
    )
    kwargs: dict[str, object] = {
        "bus": runtime.bus if bus is None else bus,
        "control": runtime.control,
        "management_dispatcher": dispatcher,
        "skill_metadata": skill_metadata,
    }
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    app = app_type(**kwargs)  # type: ignore[arg-type]
    if confirmation_coordinator is not None:
        bind_confirmation_coordinator = getattr(app, "bind_confirmation_coordinator", None)
        assert callable(bind_confirmation_coordinator)
        bind_confirmation_coordinator(confirmation_coordinator)
    test_cleanup = cleanup or getattr(runtime, "_terminal_test_cleanup", None)
    return cast(TerminalConversationApp, _TerminalTestDriver(app, runtime, test_cleanup))


async def _run_cli_terminal_case(
    *,
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Callable[[TerminalConversationApp, Pilot[None]], Awaitable[None]],
    provider: _FixedCatalogProvider | None = None,
    size: tuple[int, int] = (80, 24),
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    selected_provider = provider or _FixedCatalogProvider(())

    from aide.agent.loop import AgentRunExecutor as ProductionExecutor

    class DeterministicExecutor(ProductionExecutor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["model_router"] = TaskFramingRouterAdapter(kwargs["model_router"])
            super().__init__(*args, **kwargs)

    class ScenarioApp(TerminalConversationApp):
        async def run_async(self, **options: Any) -> None:
            del options
            async with self.run_test(size=size) as pilot:
                await scenario(self, pilot)

    monkeypatch.setattr("aide.service.runtime.workspace.AgentRunExecutor", DeterministicExecutor)
    monkeypatch.setattr(cli, "TerminalConversationApp", ScenarioApp)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        "aide.service.runtime.service.create_provider", lambda _configuration: selected_provider
    )
    async with cli_service(home):
        await cli._run_service_cli_conversation(agent_home=home, workspace=workspace)


@pytest.mark.asyncio
async def test_service_terminal_recalls_only_queued_inputs_through_up_key(
    agent_home: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = CancellableProvider()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("active A"), "enter")
        await asyncio.wait_for(provider.first_delta_emitted.wait(), 2)
        for text in ("queued B", "queued C"):
            await pilot.press(*list(text), "enter")
            await pilot.pause()
        assert app.has_pending_input
        assert "Pending (2)" in str(app.query_one("#pending-queue", Static).content)
        await pilot.press("up")
        async with asyncio.timeout(3):
            while input_area.text != "queued B\nqueued C":
                await pilot.pause()
        assert not app.has_pending_input
        assert app._control.has_active_run
        assert not provider.cancelled.is_set()
        await pilot.press("ctrl+c")
        await _wait_for_turn(app)
        input_area.text = "next D"
        await pilot.press("enter")
        await _wait_for_turn(app)
        assert len(provider.stream_requests) == 2
        assert all(
            "queued B" not in str(call.messages) and "queued C" not in str(call.messages)
            for call in provider.stream_requests
        )
        assert "Recovered runtime response." in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=provider,
    )


@pytest.mark.asyncio
async def test_service_terminal_preserves_output_delivered_before_submission_ack(
    agent_home: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.service.client import ServiceClient

    finished = asyncio.Event()
    original_event = ServiceClient._handle_event
    original_submit = ServiceClient.submit_user_input

    async def handle_event(
        client: ServiceClient, event: Mapping[str, object], *, recover_display: bool = False
    ) -> None:
        await original_event(client, event, recover_display=recover_display)
        if event.get("type") == "run.completed":
            finished.set()

    async def delayed_ack(
        client: ServiceClient, text: str, *, request_id: str | None = None
    ) -> dict[str, object]:
        result = await original_submit(client, text, request_id=request_id)
        await asyncio.wait_for(finished.wait(), 2)
        return result

    monkeypatch.setattr(ServiceClient, "_handle_event", handle_event)
    monkeypatch.setattr(ServiceClient, "submit_user_input", delayed_ack)

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        await pilot.press(*list("fast answer"), "enter")
        await _wait_for_turn(app)
        assert finished.is_set()
        assert not app._consumed_runs
        assert "early response" in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=_FixedCatalogProvider((_response(content="early response"),)),
    )


def _tool_call(
    tool_call_id: str,
    tool_name: str,
    arguments: str,
) -> OutboundMessage:
    return OutboundMessage(
        type="tool_call",
        content=tool_name,
        metadata={"tool_call_id": tool_call_id, "arguments": arguments},
    )


def _response_delta(delta: str) -> OutboundMessage:
    return OutboundMessage(
        type="model_response",
        content=delta,
        metadata={"_stream_delta": True},
    )


def _response_segment(content: str) -> _ResponseSegmentScript:
    return _ResponseSegmentScript(content=content)


def _completed_response() -> OutboundMessage:
    return OutboundMessage(
        type="model_response",
        content="",
        metadata={"_streamed": True},
    )


def _cancelled_response(partial_content: str) -> OutboundMessage:
    return OutboundMessage(
        type="system_control",
        content=partial_content,
        metadata={"_streamed": True, "finish_reason": "cancelled"},
    )


def _failed_response(error_message: str) -> OutboundMessage:
    return OutboundMessage(
        type="system_control",
        content=error_message,
        metadata={"_streamed": True, "finish_reason": "failed"},
    )


def _expand_script_item(
    script_item: _ScriptItem,
    response_delta_seen: bool,
) -> tuple[tuple[OutboundMessage, ...], bool]:
    if isinstance(script_item, _ResponseSegmentScript):
        messages: list[OutboundMessage] = []
        if script_item.content and not response_delta_seen:
            messages.append(
                OutboundMessage("model_response", script_item.content, {"_stream_delta": True})
            )
        messages.append(OutboundMessage("model_response", "", {"_stream_end": True}))
        return tuple(messages), False
    if isinstance(script_item, ConfirmationRequest):
        raise TypeError("confirmation scripts must be consumed by the run source")
    is_delta = (
        script_item.type == "model_response" and script_item.metadata.get("_stream_delta") is True
    )
    return (script_item,), is_delta


def _tool_row_texts(app: TerminalConversationApp) -> list[str]:
    rows = app.query(".tool-row")
    assert all(isinstance(row, Static) for row in rows)
    return [str(cast(Static, row).content) for row in rows]


def _tool_row_tail_matches(
    app: TerminalConversationApp,
    expected: Sequence[str],
) -> bool:
    return _tool_row_texts(app)[-1:] == list(expected)


def _visible_screen_text(app: TerminalConversationApp) -> str:
    text = "".join(element.text or "" for element in _screenshot_text_elements(app))
    return text.replace("\xa0", " ")


def _screenshot(app: TerminalConversationApp) -> ElementTree.Element:
    return ElementTree.fromstring(app.export_screenshot(simplify=True))


def _screenshot_text_elements(app: TerminalConversationApp) -> list[ElementTree.Element]:
    return [
        element
        for element in _screenshot(app).iter()
        if element.tag.endswith("text")
        and element.text
        and not element.attrib.get("class", "").endswith("-title")
    ]


def _content_text_nodes(app: TerminalConversationApp, marker: str) -> list[str]:
    return [
        (element.text or "").replace("\xa0", " ")
        for element in _screenshot_text_elements(app)
        if all(character in marker for character in (element.text or "").replace("\xa0", " "))
    ]


def _screenshot_text_nodes(app: TerminalConversationApp) -> list[tuple[str, float, float]]:
    return [
        (
            (element.text or "").replace("\xa0", " "),
            float(element.attrib["x"]),
            float(element.attrib["y"]),
        )
        for element in _screenshot_text_elements(app)
    ]


def _screenshot_width(app: TerminalConversationApp) -> float:
    return float(_screenshot(app).attrib["viewBox"].split()[2])


async def _wait_for_turn(app: TerminalConversationApp) -> None:
    text_area = app.query_one("#conversation-input", TextArea)
    async with asyncio.timeout(2):
        while (
            text_area.read_only
            or getattr(text_area, "active_turn_token", None) is not None
            or app._active_run_projection is not None
        ):
            await asyncio.sleep(0)
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()


async def _wait_for_refresh_condition(
    app: TerminalConversationApp,
    predicate: Callable[[], bool],
) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            refreshed = asyncio.Event()
            app.call_after_refresh(refreshed.set)
            await refreshed.wait()


def test_terminal_constructor_does_not_expose_runtime_lifecycle_parameters() -> None:
    parameter_names = set(inspect.signature(TerminalConversationApp).parameters)

    assert parameter_names.isdisjoint({"start_runtime", "close_runtime", "runtime_host"})


@pytest.mark.asyncio
async def test_terminal_unmount_clears_ui_owned_generation_state() -> None:
    app = TerminalConversationApp(
        bus=MessageBus(),
        control=_DirectControl(),
        management_dispatcher=_management_dispatcher(),
    )

    async with app.run_test(size=(80, 24)):
        app._pending_inputs.append("stale input")
        app._bus_snapshot = (InboundMessage(content="stale inbound"),)
        app._cancel_requested_turn = object()
        app._active_confirmation_id = uuid4()
        app._completion_dismissed_text = "stale completion"
        app._run_ready.clear()
        app.exit()

    assert not app._pending_inputs
    assert not app._bus_snapshot
    assert app._cancel_requested_turn is None
    assert app._active_confirmation_id is None
    assert app._completion_dismissed_text is None
    assert app._run_ready.is_set() is False
    assert app._outbound_worker is None
    assert app._resume_worker is None


def _constant_datetime(value: datetime) -> Callable[[], datetime]:
    return lambda: value


def _constant_uuid(value: UUID) -> Callable[[], UUID]:
    return lambda: value


async def _wait_for_session_picker(
    app: TerminalConversationApp,
    pilot: Pilot[None],
) -> None:
    async with asyncio.timeout(2):
        while app.screen.id != "session-picker" or not app.screen.query("#session-picker-options"):
            await pilot.pause()


async def _wait_for_screen_id(
    app: TerminalConversationApp,
    pilot: Pilot[None],
    screen_id: str,
) -> None:
    async with asyncio.timeout(2):
        while app.screen.id != screen_id:
            await pilot.pause()


async def _wait_for_confirmation(
    app: TerminalConversationApp,
    pilot: Pilot[None],
    *,
    timeout: float = 5,
) -> None:
    async with asyncio.timeout(timeout):
        while (
            app.screen.id is None
            or not app.screen.id.startswith("confirmation-")
            or not app.screen.is_mounted
            or not app.screen.query(".confirmation-details")
        ):
            await pilot.pause()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24), (120, 32)))
async def test_restore_mode_and_final_actions_fit_terminal_sizes(size: tuple[int, int]) -> None:
    plan = RestorePlan(
        session_id="test-session",
        anchor_id=1,
        session_digest="digest",
        journal_revision=1,
        removed_users=2,
        removed_messages=8,
        targets=(),
        external_target_count=0,
        backup_gaps=(),
        integrity_issues=(),
        conflict_targets=(),
        discarded_run_tokens=(),
        available_modes=(RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES),
    )
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))

    async with app.run_test(size=size) as pilot:
        await app.push_screen(_RestoreModeScreen(plan))
        await pilot.pause()
        options = app.screen.query_one("#restore-mode-options", OptionList)
        assert options.region.height > 0
        assert options.region.bottom <= size[1]
        assert app.screen.focused is options
        await pilot.press("escape")

        await app.push_screen(_RestoreConfirmationScreen(plan, RestoreMode.FILES))
        await pilot.pause()
        cancel = app.screen.query_one("#restore-confirmation-cancel", Button)
        approve = app.screen.query_one("#restore-confirmation-approve", Button)
        assert cast(Any, app.screen.focused) is cancel
        assert cancel.region.width > 0 and approve.region.width > 0
        assert cancel.region.bottom <= size[1] and approve.region.bottom <= size[1]
        assert not cancel.region.overlaps(approve.region)
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_restore_workflow_refills_selected_anchor_after_partial_file_failure() -> None:
    selected_modes: list[RestoreMode | str] = []
    cancel_calls = 0
    acknowledgement_calls = 0
    anchor_token = UUID("550e8400-e29b-41d4-a716-446655440001")
    newer_anchor_token = UUID("550e8400-e29b-41d4-a716-446655440011")
    selected_content = "  selected input\nsecond line  "
    newer_content = "newer discarded input"
    projection = ForegroundConversationProjection(
        session_id="20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000",
        messages=(),
    )
    target = RestoreTarget(
        canonical_target=Path("tracked.txt"),
        operation_id=1,
        requested_targets=(Path("tracked.txt"),),
        before_exists=True,
        before_sha256="before",
        before_bytes=b"before",
        latest_after_exists=True,
        latest_after_sha256="after",
        external=False,
        session_owned=False,
    )
    plan = RestorePlan(
        session_id=projection.session_id,
        anchor_id=2,
        session_digest="digest",
        journal_revision=1,
        removed_users=1,
        removed_messages=3,
        targets=(target,),
        external_target_count=0,
        backup_gaps=(),
        integrity_issues=(),
        conflict_targets=(target.canonical_target,),
        discarded_run_tokens=(anchor_token,),
        available_modes=(RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES),
    )
    restore_result = RestoreResult(
        session_id=projection.session_id,
        anchor_id=2,
        mode=RestoreMode.FILES,
        removed_users=1,
        removed_messages=3,
        file_results=(
            RestoreFileResult(
                target=target.canonical_target,
                operation_id=1,
                status=RestoreFileStatus.RESTORED,
                conflict=True,
            ),
            RestoreFileResult(
                target=Path("failed.txt"),
                operation_id=2,
                status=RestoreFileStatus.FAILED,
                conflict=False,
                error="injected failure",
            ),
        ),
        session_result=SessionRestoreResult(
            session_id=projection.session_id,
            anchor_id=2,
            removed_messages=3,
            updated_at=NOW,
        ),
        failure_notification_acknowledged=False,
    )

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=projection.session_id,
                anchors=(
                    RestoreAnchor(
                        anchor_id=3,
                        run_token=newer_anchor_token,
                        content=newer_content,
                        timestamp="2026-09-26T12:01:00.000+08:00",
                    ),
                    RestoreAnchor(
                        anchor_id=2,
                        run_token=anchor_token,
                        content=selected_content,
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_inspect(self, _anchor_id: int) -> RestorePlan:
            return plan

        async def restore_commit(
            self,
            _anchor_id: int,
            mode: RestoreMode | str,
        ) -> RestoreResult:
            selected_modes.append(mode)
            object.__setattr__(
                projection,
                "messages",
                ({"role": "user", "content": "kept input"},),
            )
            return restore_result

        async def restore_result(self) -> RestoreResult | None:
            return None

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

        async def restore_acknowledge_failure(self) -> RestoreResult | None:
            nonlocal acknowledgement_calls
            acknowledgement_calls += 1
            return restore_result

    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return True

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return projection

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        input_area = app.query_one("#conversation-input", _ConversationInput)
        input_area.remember_submission("kept input")
        input_area.remember_submission(selected_content)
        input_area.remember_submission(newer_content)

        await pilot.press(*list("/restore"), "enter")
        await _wait_for_screen_id(app, pilot, "restore-anchor-picker")
        await pilot.press(*list("ordinary input"))
        assert app.screen.id == "restore-anchor-picker"
        assert app.screen.query_one("#restore-anchor-options", OptionList).option_count == 0
        app.screen.query_one("#restore-anchor-filter", Input).value = "selected"
        await pilot.pause()
        assert app.screen.query_one("#restore-anchor-options", OptionList).options[0].id == "2"
        await pilot.pause()
        await pilot.click("#restore-anchor-options", offset=(4, 1))
        await _wait_for_screen_id(app, pilot, "restore-mode-picker")
        await pilot.pause()

        mode_options = app.screen.query_one("#restore-mode-options", OptionList)
        assert app.screen.focused is mode_options
        assert "Impact: 3 messages removed; 1 file targets" in str(
            app.screen.query_one("#restore-mode-impact", Static).content
        )
        assert str(mode_options.options[0].prompt).startswith("Conversation only")
        assert str(mode_options.options[1].prompt) == "Conversation + files"
        await pilot.press("down", "enter")
        await _wait_for_screen_id(app, pilot, "restore-confirmation")

        confirmation_cancel = app.screen.query_one("#restore-confirmation-cancel", Button)
        assert app.screen.focused is not None
        assert app.screen.focused.id == confirmation_cancel.id
        confirmation_text = _visible_screen_text(app)
        assert "Impact: 3 messages removed; 1 file targets" in str(
            app.screen.query_one("#restore-confirmation-details").query_one(
                ".restore-confirmation-detail", Static
            ).content
        )
        assert "Remove: 1 User and 3 total messages" in confirmation_text
        assert "Files: 1 tracked, 0 external" in confirmation_text
        assert "Not independently rolled back:" in confirmation_text
        assert "may still overwrite later manual, Exec, or MCP changes" in confirmation_text
        await pilot.click(offset=(1, 1))
        assert app.screen.id == "restore-confirmation"
        await pilot.click("#restore-confirmation-approve")

        await _wait_for_screen_id(app, pilot, "restore-failure")
        await pilot.pause()
        assert input_area.text == selected_content
        assert input_area.cursor_location == (1, len("second line  "))
        assert input_area.read_only is True
        assert "failed.txt" in _visible_screen_text(app)
        await pilot.click("#restore-failure-acknowledge")
        await _wait_for_screen_id(app, pilot, "_default")
        async with asyncio.timeout(2):
            while app._restore_workflow_active:
                await pilot.pause()

        visible_text = _visible_screen_text(app)
        assert "Restored conflicting files:" in visible_text
        assert "tracked.txt" in visible_text
        assert "restore-failure" not in str(app.screen.id)
        assert selected_modes == [RestoreMode.FILES]
        assert cancel_calls == 0
        assert acknowledgement_calls == 1
        assert input_area.text == selected_content
        assert input_area.cursor_location == (1, len("second line  "))
        assert input_area.has_focus
        assert input_area.read_only is False
        assert await bus.inbound_snapshot() == ()

        await pilot.press("up")
        assert input_area.text == selected_content
        input_area.text = ""
        await pilot.press("up")
        assert input_area.text == "/restore"
        assert input_area._history.count(selected_content) == 0
        assert input_area._history.count(newer_content) == 0
        assert input_area._history.count("/restore") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("has_gap", "cancel_stage", "cancel_key"),
    (
        (False, "picker", "escape"),
        (True, "mode", "ctrl+c"),
        (False, "confirmation", "escape"),
        (True, "confirmation", "ctrl+c"),
    ),
)
async def test_restore_workflow_cancellation_branches_do_not_refill_draft(
    has_gap: bool,
    cancel_stage: str,
    cancel_key: str,
) -> None:
    cancel_calls = 0
    anchor_token = UUID("550e8400-e29b-41d4-a716-446655440002")
    session_id = "20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000"
    plan = RestorePlan(
        session_id=session_id,
        anchor_id=1,
        session_digest="digest",
        journal_revision=1,
        removed_users=1,
        removed_messages=1,
        targets=(),
        external_target_count=0,
        backup_gaps=(
            BackupGap(
                operation_id=1,
                revision=1,
                run_token=anchor_token,
                requested_target="missing.txt",
                canonical_target="missing.txt",
                reason="target_read_failed",
            ),
        )
        if has_gap
        else (),
        integrity_issues=(),
        conflict_targets=(),
        discarded_run_tokens=(anchor_token,),
        available_modes=(RestoreMode.CONVERSATION_ONLY,),
    )
    projection = ForegroundConversationProjection(session_id=session_id, messages=())

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=session_id,
                anchors=(
                    RestoreAnchor(
                        anchor_id=1,
                        run_token=anchor_token,
                        content="discarded input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_inspect(self, _anchor_id: int) -> RestorePlan:
            return plan

        async def restore_commit(
            self,
            _anchor_id: int,
            _mode: RestoreMode | str,
        ) -> RestoreResult:
            raise AssertionError("cancelled restore must not commit")

        async def restore_result(self) -> RestoreResult | None:
            return None

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return cancel_calls > 0

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return projection

    app = TerminalConversationApp(
        bus=MessageBus(),
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        input_area = app.query_one("#conversation-input", _ConversationInput)
        await pilot.press(*list("/restore"), "enter")
        await _wait_for_screen_id(app, pilot, "restore-anchor-picker")
        if cancel_stage == "picker":
            await pilot.press(cancel_key)
        else:
            await pilot.press("enter")
            if has_gap:
                await _wait_for_screen_id(app, pilot, "restore-mode-picker")
                async with asyncio.timeout(2):
                    while "incomplete backup coverage" not in _visible_screen_text(app):
                        await pilot.pause()
                assert "incomplete backup coverage" in _visible_screen_text(app)
                if cancel_stage == "mode":
                    await pilot.press(cancel_key)
                else:
                    await pilot.press("enter")
            if cancel_stage == "confirmation":
                await _wait_for_screen_id(app, pilot, "restore-confirmation")
                async with asyncio.timeout(2):
                    while "Files: 0 tracked, 0 external" not in _visible_screen_text(app):
                        await pilot.pause()
                assert "Files: 0 tracked, 0 external" in _visible_screen_text(app)
                await pilot.press(cancel_key)
        async with asyncio.timeout(2):
            while app._restore_workflow_active:
                await pilot.pause()

        assert cancel_calls == 1
        assert app.screen.id == "_default"
        assert input_area.text == ""
        assert "discarded input" not in input_area.text


@pytest.mark.asyncio
async def test_restore_workflow_conversation_only_refills_unsubmitted_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = " conversation-only\nanchor text "
    anchor = RestoreAnchor(
        anchor_id=1,
        run_token=UUID("550e8400-e29b-41d4-a716-446655440012"),
        content=content,
        timestamp="2026-09-26T12:00:00.000+08:00",
    )
    plan = RestorePlan(
        session_id="direct-test",
        anchor_id=anchor.anchor_id,
        session_digest="digest",
        journal_revision=1,
        removed_users=1,
        removed_messages=1,
        targets=(),
        external_target_count=0,
        backup_gaps=(),
        integrity_issues=(),
        conflict_targets=(),
        discarded_run_tokens=(anchor.run_token,),
        available_modes=(RestoreMode.CONVERSATION_ONLY,),
    )
    result = RestoreResult(
        session_id=plan.session_id,
        anchor_id=plan.anchor_id,
        mode=RestoreMode.CONVERSATION_ONLY,
        removed_users=1,
        removed_messages=1,
        file_results=(),
        session_result=SessionRestoreResult(
            session_id=plan.session_id,
            anchor_id=plan.anchor_id,
            removed_messages=1,
            updated_at=NOW,
        ),
        failure_notification_acknowledged=True,
    )

    class Management:
        async def restore_commit(
            self,
            _anchor_id: int,
            _mode: RestoreMode | str,
        ) -> RestoreResult:
            return result

        async def restore_result(self) -> RestoreResult | None:
            return None

        async def restore_cancel(self) -> None:
            return None

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=_DirectControl(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async def select_anchor(_anchors: tuple[RestoreAnchor, ...]) -> int:
        return anchor.anchor_id

    async def inspect_anchor(
        _anchor_id: int,
    ) -> tuple[ManagementCommandResult, bool]:
        return ManagementCommandResult(handled=True, output=None, restore_plan=plan), False

    async def confirm_restore(_plan: RestorePlan, _mode: RestoreMode) -> bool:
        return True

    monkeypatch.setattr(app, "_restore_picker_selection", select_anchor)
    monkeypatch.setattr(app, "_restore_inspection", inspect_anchor)
    monkeypatch.setattr(app, "_restore_confirmation", confirm_restore)

    async with app.run_test(size=(100, 30)):
        input_area = app.query_one("#conversation-input", _ConversationInput)
        input_area.remember_submission(content)
        input_area.remember_submission("/restore")

        await app._run_restore_workflow((anchor,), input_area)

        assert input_area.text == content
        assert input_area.cursor_location == (1, len("anchor text "))
        assert input_area._history == ["/restore"]
        assert input_area.read_only is False
        assert input_area.has_focus
        assert await bus.inbound_snapshot() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "missing_selected_anchor",
        "missing_plan",
        "plan_anchor_mismatch",
        "missing_result",
        "result_anchor_mismatch",
        "result_session_mismatch",
        "projection_session_mismatch",
    ),
)
async def test_restore_workflow_identity_and_failure_guards_do_not_refill_draft(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "SECRET anchor body"
    anchor = RestoreAnchor(
        anchor_id=1,
        run_token=UUID("550e8400-e29b-41d4-a716-446655440013"),
        content=content,
        timestamp="2026-09-26T12:00:00.000+08:00",
    )
    base_plan = RestorePlan(
        session_id="direct-test",
        anchor_id=anchor.anchor_id,
        session_digest="digest",
        journal_revision=1,
        removed_users=1,
        removed_messages=1,
        targets=(),
        external_target_count=0,
        backup_gaps=(),
        integrity_issues=(),
        conflict_targets=(),
        discarded_run_tokens=(anchor.run_token,),
        available_modes=(RestoreMode.CONVERSATION_ONLY,),
    )
    plan = base_plan
    if case == "plan_anchor_mismatch":
        plan = replace(plan, anchor_id=2)
    elif case == "projection_session_mismatch":
        plan = replace(plan, session_id="other-session")
    base_result = RestoreResult(
        session_id=plan.session_id,
        anchor_id=plan.anchor_id,
        mode=RestoreMode.CONVERSATION_ONLY,
        removed_users=1,
        removed_messages=1,
        file_results=(),
        session_result=SessionRestoreResult(
            session_id=plan.session_id,
            anchor_id=plan.anchor_id,
            removed_messages=1,
            updated_at=NOW,
        ),
        failure_notification_acknowledged=True,
    )
    result: RestoreResult | None = base_result
    if case == "missing_result":
        result = None
    elif case == "result_anchor_mismatch":
        result = replace(base_result, anchor_id=2)
    elif case == "result_session_mismatch":
        result = replace(base_result, session_id="other-session")
    inspection_plan = None if case == "missing_plan" else plan
    commit_calls = 0
    cancel_calls = 0

    class Management:
        async def restore_result(self) -> RestoreResult | None:
            return None

    app = TerminalConversationApp(
        bus=MessageBus(),
        control=_DirectControl(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async def select_anchor(_anchors: tuple[RestoreAnchor, ...]) -> int:
        return 2 if case == "missing_selected_anchor" else anchor.anchor_id

    async def inspect_anchor(
        _anchor_id: int,
    ) -> tuple[ManagementCommandResult, bool]:
        return (
            ManagementCommandResult(
                handled=True,
                output="Inspection failed safely." if inspection_plan is None else None,
                restore_plan=inspection_plan,
            ),
            False,
        )

    async def confirm_restore(_plan: RestorePlan, _mode: RestoreMode) -> bool:
        return True

    async def commit_restore(
        _anchor_id: int,
        _mode: RestoreMode | str,
    ) -> ManagementCommandResult:
        nonlocal commit_calls
        commit_calls += 1
        return ManagementCommandResult(
            handled=True,
            output="Commit failed safely." if result is None else "Restore completed.",
            restore_result=result,
        )

    async def cancel_restore() -> ManagementCommandResult:
        nonlocal cancel_calls
        cancel_calls += 1
        return ManagementCommandResult(handled=True, output="Session Restore cancelled.")

    monkeypatch.setattr(app, "_restore_picker_selection", select_anchor)
    monkeypatch.setattr(app, "_restore_inspection", inspect_anchor)
    monkeypatch.setattr(app, "_restore_confirmation", confirm_restore)
    monkeypatch.setattr(app._management_dispatcher, "restore_commit", commit_restore)
    monkeypatch.setattr(app._management_dispatcher, "restore_cancel", cancel_restore)

    async with app.run_test(size=(100, 30)):
        input_area = app.query_one("#conversation-input", _ConversationInput)
        input_area.remember_submission(content)
        input_area.remember_submission("/restore")

        await app._run_restore_workflow((anchor,), input_area)

        assert input_area.text == ""
        assert content not in _visible_screen_text(app)
        assert commit_calls == (
            0
            if case in {"missing_selected_anchor", "missing_plan", "plan_anchor_mismatch"}
            else 1
        )
        assert cancel_calls == (
            1 if case in {"missing_selected_anchor", "plan_anchor_mismatch"} else 0
        )
        if case == "projection_session_mismatch":
            assert input_area._history == ["/restore"]
        else:
            assert input_area._history == [content, "/restore"]


@pytest.mark.asyncio
async def test_restore_waiting_barrier_can_be_cancelled_before_inspection_finishes() -> None:
    inspect_started = asyncio.Event()
    cancel_calls = 0
    anchor_token = UUID("550e8400-e29b-41d4-a716-446655440003")
    session_id = "20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000"
    plan = RestorePlan(
        session_id=session_id,
        anchor_id=1,
        session_digest="digest",
        journal_revision=1,
        removed_users=1,
        removed_messages=1,
        targets=(),
        external_target_count=0,
        backup_gaps=(),
        integrity_issues=(),
        conflict_targets=(),
        discarded_run_tokens=(anchor_token,),
        available_modes=(RestoreMode.CONVERSATION_ONLY,),
    )

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=session_id,
                anchors=(
                    RestoreAnchor(
                        anchor_id=1,
                        run_token=anchor_token,
                        content="discarded input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_inspect(self, _anchor_id: int) -> RestorePlan:
            inspect_started.set()
            await asyncio.Event().wait()
            return plan

        async def restore_commit(
            self,
            _anchor_id: int,
            _mode: RestoreMode | str,
        ) -> RestoreResult:
            raise AssertionError("cancelled inspection must not commit")

        async def restore_result(self) -> RestoreResult | None:
            return None

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return cancel_calls > 0

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(session_id=session_id, messages=())

    app = TerminalConversationApp(
        bus=MessageBus(),
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/restore"), "enter")
        await _wait_for_screen_id(app, pilot, "restore-anchor-picker")
        await pilot.press("enter")
        await _wait_for_screen_id(app, pilot, "restore-waiting")
        await asyncio.wait_for(inspect_started.wait(), timeout=1)
        await pilot.press("escape")
        async with asyncio.timeout(2):
            while app._restore_workflow_active:
                await pilot.pause()

        assert cancel_calls == 1
        assert app.screen.id != "restore-waiting"


@pytest.mark.asyncio
async def test_restore_inspection_failure_dismisses_waiting_screen() -> None:
    cancel_calls = 0
    anchor_token = UUID("550e8400-e29b-41d4-a716-446655440004")
    session_id = "20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000"

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=session_id,
                anchors=(
                    RestoreAnchor(
                        anchor_id=1,
                        run_token=anchor_token,
                        content="discarded input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_inspect(self, _anchor_id: int) -> RestorePlan:
            raise RuntimeError("injected inspection failure")

        async def restore_result(self) -> RestoreResult | None:
            return None

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

    app = TerminalConversationApp(
        bus=MessageBus(),
        control=_DirectControl(),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, Management())),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/restore"), "enter")
        await _wait_for_screen_id(app, pilot, "restore-anchor-picker")
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app._restore_workflow_active:
                await pilot.pause()

        assert app.screen.id == "_default"
        assert cancel_calls == 1
        assert app.query_one("#conversation-input", _ConversationInput).read_only is False
        assert any(
            "Session Restore failed." in str(cast(Static, row).content)
            for row in app.query(".management-row")
        )


@pytest.mark.asyncio
async def test_restore_failure_notification_acknowledgement_survives_restart(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discarded_input = "discard this persisted input"
    blocked = workspace / "blocked.txt"
    successful_conflict = workspace / "conflict.txt"
    blocked.write_bytes(b"blocked before")
    successful_conflict.write_bytes(b"conflict before")
    session_ids: list[str] = []
    original_replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_blocked_restore(target: Path, content: bytes) -> None:
        if Path(target) == blocked and content == b"blocked before":
            raise PermissionError("injected restore permission failure")
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_blocked_restore)

    async def restore_with_failure(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        await pilot.press(*list(discarded_input), "enter")
        await _wait_for_turn(app)
        session = Session.load(
            WorkspaceState(workspace), app._control.project_foreground_conversation().session_id
        )
        session_ids.append(session.session_id)
        run_token = session.restore_candidates()[0].run_token
        store = FileBackupStore(session.workspace_state, session.session_id)
        for target, tool_content, later_content in (
            (blocked, b"blocked by tool", b"blocked later"),
            (successful_conflict, b"conflict by tool", b"conflict later"),
        ):
            ticket = store.before_write(run_token, target)
            assert ticket is not None
            target.write_bytes(tool_content)
            store.after_write(ticket)
            target.write_bytes(later_content)
        await session.wait_for_pending_persist()

        input_area = app.query_one("#conversation-input", _ConversationInput)
        await pilot.press(*list("/restore"), "enter")
        await _wait_for_screen_id(app, pilot, "restore-anchor-picker")
        await pilot.press("enter")
        await _wait_for_screen_id(app, pilot, "restore-mode-picker")
        await pilot.press("down", "enter")
        await _wait_for_screen_id(app, pilot, "restore-confirmation")
        confirmation_text = _visible_screen_text(app)
        assert "Not independently rolled back:" in confirmation_text
        assert "may still overwrite later manual, Exec, or MCP changes" in confirmation_text
        assert app.screen.focused is app.screen.query_one("#restore-confirmation-cancel")
        await pilot.press("right", "enter")
        await _wait_for_screen_id(app, pilot, "restore-failure")

        notification_details = tuple(
            str(cast(Static, detail).content)
            for detail in app.screen.query(".restore-failure-detail")
        )
        assert f"Failed: {blocked}" in notification_details
        assert f"Restored conflict: {successful_conflict}" in notification_details
        assert input_area.read_only is True
        async with asyncio.timeout(3):
            while discarded_input in input_area._history:
                await pilot.pause()
        assert discarded_input not in input_area._history
        assert "/restore" in input_area._history
        assert app._control.project_foreground_conversation().session_id == session.session_id
        assert Session.load(session.workspace_state, session.session_id).messages == []
        assert blocked.read_bytes() == b"blocked later"
        assert successful_conflict.read_bytes() == b"conflict before"
        assert any(
            "Restored conflicting files:" in str(cast(Static, row).content)
            and str(successful_conflict) in str(cast(Static, row).content)
            for row in app.query(".management-row")
        )

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=restore_with_failure,
        provider=_FixedCatalogProvider((_response(content="Completed before restore."),)),
        size=(100, 30),
    )

    session_id = session_ids[0]
    pending = await RestoreManager(WorkspaceState(workspace), session_id).recover_pending()
    assert pending is not None
    assert pending.failure_notification_pending is True

    async def acknowledge_after_restart(
        app: TerminalConversationApp,
        pilot: Pilot[None],
    ) -> None:
        await _wait_for_screen_id(app, pilot, "restore-failure")
        assert f"Failed: {blocked}" in {
            str(cast(Static, detail).content)
            for detail in app.screen.query(".restore-failure-detail")
        }
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app._restore_workflow_active:
                await pilot.pause()
        assert app.screen.id != "restore-failure"

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=acknowledge_after_restart,
        size=(100, 30),
    )

    acknowledged = await RestoreManager(WorkspaceState(workspace), session_id).recover_pending()
    assert acknowledged is not None
    assert acknowledged.failure_notification_acknowledged is True

    async def verify_acknowledged_restart(
        app: TerminalConversationApp,
        pilot: Pilot[None],
    ) -> None:
        async with asyncio.timeout(2):
            while (
                app._restore_result_worker is not None
                and not app._restore_result_worker.is_finished
            ):
                await pilot.pause()
        assert app.screen.id != "restore-failure"

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=verify_acknowledged_restart,
        size=(100, 30),
    )


@pytest.mark.asyncio
async def test_resume_picker_orders_sessions_and_cancellation_preserves_display(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        initial = app._control
        older = Session.create(
            WorkspaceState(workspace),
            now=lambda: NOW.replace(hour=10),
            new_uuid=lambda: UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
        )
        seed_session_state(
            older,
            messages=[
                {
                    "role": "user",
                    "content": "Older persisted question.",
                    "timestamp": "2026-08-11T10:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Older session",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        older.close()
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: NOW,
            new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Persisted question.",
                    "timestamp": "2026-08-11T12:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Target session",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/status"), "enter", "ctrl+home")
        before_resume = _visible_screen_text(app)
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        picker_text = _visible_screen_text(app)
        assert picker_text.index("Target session") < picker_text.index("Older session")
        assert target.session_id not in picker_text
        assert older.session_id not in picker_text
        assert target.updated_at.astimezone().strftime("%Y-%m-%d %H:%M") in picker_text
        assert app.screen.focused is app.screen.query_one("#session-picker-filter", Input)

        app.screen.query_one("#session-picker-filter", Input).value = "Target"
        await pilot.pause()
        filtered = app.screen.query_one("#session-picker-options", OptionList)
        assert filtered.option_count == 1
        assert filtered.options[0].id == target.session_id
        assert "1 message" in str(filtered.options[0].prompt)

        await pilot.click(offset=(1, 1))
        assert app.screen.id == "session-picker"
        await pilot.press("escape", "ctrl+home")
        await pilot.pause()
        assert app._control is initial
        assert _visible_screen_text(app) == before_resume

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
    )


@pytest.mark.asyncio
async def test_resume_selection_rebinds_sanitized_session_projection(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: NOW,
            new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
        )
        timestamp = "2026-08-11T12:00:00.000+00:00"
        seed_session_state(
            target,
            messages=[
                {"role": "user", "content": "Persisted question.", "timestamp": timestamp},
                {
                    "role": "assistant",
                    "content": "Persisted **answer**.",
                    "timestamp": timestamp,
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "total_tokens": 5,
                    },
                },
                {
                    "role": "assistant",
                    "content": "",
                    "timestamp": timestamp,
                    "tool_calls": [
                        {
                            "id": "call-restored",
                            "name": "read_file",
                            "arguments": '{"api_key":"private"}',
                        },
                        {
                            "id": "call-error",
                            "name": "exec",
                            "arguments": '{"command":"private"}',
                        },
                        {
                            "id": "call-refused",
                            "name": "web_fetch",
                            "arguments": '{"url":"private"}',
                        },
                    ],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                {
                    "role": "tool",
                    "content": "private tool result",
                    "timestamp": timestamp,
                    "tool_call_id": "call-restored",
                    "name": "read_file",
                    "status": "success",
                    "artifact": None,
                },
                {
                    "role": "tool",
                    "content": "STDERR permission denied; STDOUT secret bytes",
                    "timestamp": timestamp,
                    "tool_call_id": "call-error",
                    "name": "exec",
                    "status": "error",
                    "artifact": None,
                },
                {
                    "role": "tool",
                    "content": "private refusal detail",
                    "timestamp": timestamp,
                    "tool_call_id": "call-refused",
                    "name": "web_fetch",
                    "status": "refused",
                    "artifact": None,
                },
                {
                    "role": "assistant",
                    "content": "Persisted partial answer.",
                    "timestamp": timestamp,
                    "tool_calls": [],
                    "status": "interrupted",
                    "error": {
                        "code": "turn_cancelled",
                        "message": "Turn interrupted by user.",
                    },
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                {
                    "role": "assistant",
                    "content": "",
                    "timestamp": timestamp,
                    "tool_calls": [],
                    "status": "error",
                    "error": {
                        "code": "model_failed",
                        "message": "Persisted model failure.",
                    },
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    },
                },
            ],
            metadata={
                "title": "Restored session",
                "token_usage": {
                    "model_calls": 4,
                    "input_tokens": 5,
                    "output_tokens": 5,
                    "total_tokens": 10,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/status"), "enter")
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(3):
            while (
                app._control.project_foreground_conversation().session_id != target.session_id
                or "Persisted model failure." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        visible_text = _visible_screen_text(app)
        expected_order = (
            "Persisted answer.",
            "Completed: read_file",
            "Failed: exec - The operation did not complete.",
            "Rejected: web_fetch",
            "Persisted partial answer.",
            "Persisted model failure.",
        )
        assert all(value in visible_text for value in expected_order)
        assert [visible_text.index(value) for value in expected_order] == sorted(
            visible_text.index(value) for value in expected_order
        )
        for secret in (
            "private tool result",
            "private refusal detail",
            "call-restored",
            "call-error",
            "call-refused",
            "api_key",
            "STDERR",
            "STDOUT",
            "secret bytes",
        ):
            assert secret not in visible_text
        assert "Command: /status" not in visible_text
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        size=(100, 40),
    )


@pytest.mark.asyncio
async def test_active_resume_decline_then_force_rebinds_the_same_bus(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CancellableProvider()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        old = app._control
        shared_bus = app._bus
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            new_uuid=uuid4,
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Approve target",
                    "timestamp": "2027-01-01T00:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Untitled session",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("active work"), "enter")
        await asyncio.wait_for(provider.first_delta_emitted.wait(), timeout=2)
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-switch-confirmation":
                await pilot.pause()

        assert app._control is old
        await pilot.press("escape")
        await pilot.pause()
        assert app._control is old
        assert old.has_active_run

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-switch-confirmation":
                await pilot.pause()
        await pilot.press("right", "enter")

        async with asyncio.timeout(3):
            while app._control.project_foreground_conversation().session_id != target.session_id:
                await pilot.pause()
        selected = app._control
        assert selected.project_foreground_conversation().session_id == target.session_id
        assert app._bus is shared_bus
        assert not selected.has_active_run
        assert not provider.cancelled.is_set()
        await pilot.pause(0.1)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=provider,
    )
    assert provider.cancelled.is_set()


@pytest.mark.asyncio
async def test_resume_picker_mouse_selection_rebinds_the_clicked_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            new_uuid=uuid4,
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Mouse-selected content.",
                    "timestamp": "2027-01-01T00:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Mouse target",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.click("#session-picker-options", offset=(4, 1))

        async with asyncio.timeout(3):
            while (
                app._control.project_foreground_conversation().session_id != target.session_id
                or "Mouse-selected content." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert "Mouse-selected content." in _visible_screen_text(app)
        for message in app.query(".message"):
            row = cast(Widget, message.parent)
            assert abs(message.outer_size.width - row.content_region.width * 0.95) <= 1

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
    )


@pytest.mark.asyncio
async def test_resume_picker_scrolls_in_management_order_and_selects_by_keyboard(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        sessions: list[Session] = []
        for index in range(50):
            session = Session.create(
                WorkspaceState(workspace),
                now=_constant_datetime(datetime(2027, 1, 1, 0, index, tzinfo=UTC)),
                new_uuid=_constant_uuid(UUID(f"00000000-0000-4000-8000-{index + 1:012x}")),
            )
            seed_session_state(
                session,
                messages=[
                    {
                        "role": "user",
                        "content": f"Content {index:02d}.",
                        "timestamp": datetime(2027, 1, 1, 0, index, tzinfo=UTC).isoformat(
                            timespec="milliseconds"
                        ),
                    }
                ],
                metadata={
                    "title": f"Session {index:02d}",
                    "token_usage": {
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "summary": "",
                },
                last_compacted=0,
            )
            session.close()
            sessions.append(session)

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        visible_text = _visible_screen_text(app)
        assert visible_text.index("Session 49") < visible_text.index("Session 48")
        options = app.screen.query_one("#session-picker-options", OptionList)
        async with asyncio.timeout(1):
            while options.max_scroll_y <= 0:
                await pilot.pause()

        await pilot._post_mouse_events(
            [MouseScrollDown], offset=(options.region.x + 2, options.region.y + 2), times=3
        )
        await pilot.pause()
        assert options.scroll_y > 0
        await pilot.press(*(("down",) * 50), "enter")

        async with asyncio.timeout(3):
            while app._control.project_foreground_conversation().session_id != sessions[
                0
            ].session_id or "Content 00." not in _visible_screen_text(app):
                await pilot.pause()

        assert "Content 00." in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        size=(80, 20),
    )


@pytest.mark.asyncio
async def test_resume_serializes_input_until_cli_rebind_finishes(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            new_uuid=uuid4,
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Delayed restored content.",
                    "timestamp": "2027-01-01T00:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Delayed target",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()
        dispatcher = app._management_dispatcher
        original_resume = dispatcher.resume
        resume_started = asyncio.Event()
        continue_resume = asyncio.Event()
        resume_finished = asyncio.Event()
        resume_errors: list[BaseException] = []

        async def delayed_resume(
            session_id: str,
            *,
            force: bool = False,
        ) -> ManagementCommandResult:
            resume_started.set()
            await continue_resume.wait()
            try:
                return await original_resume(session_id, force=force)
            except BaseException as error:
                resume_errors.append(error)
                raise
            finally:
                resume_finished.set()

        monkeypatch.setattr(dispatcher, "resume", delayed_resume)
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        await asyncio.wait_for(resume_started.wait(), timeout=1)

        input_area = app.query_one("#conversation-input", TextArea)
        assert input_area.read_only
        assert app.screen.id == "_default"
        await pilot.press(*list("racing turn"), "enter", *list("/resume"), "enter")
        assert input_area.text == ""
        assert app.screen.id == "_default"

        continue_resume.set()
        await asyncio.wait_for(resume_finished.wait(), timeout=2)
        assert resume_errors == []
        async with asyncio.timeout(3):
            while (
                app._control.project_foreground_conversation().session_id != target.session_id
                or input_area.read_only
                or "Delayed restored content." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        await pilot.press(*list("/status"), "enter")
        await pilot.press("ctrl+home")
        assert "Command: /status" in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
    )


@pytest.mark.asyncio
async def test_resumed_long_history_starts_latest_and_preserves_input_history(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            new_uuid=uuid4,
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": f"Restored line {index:02d} " + "x" * 40,
                    "timestamp": "2027-01-01T00:00:00.000+00:00",
                }
                for index in range(60)
            ],
            metadata={
                "title": "Long target",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(3):
            while (
                app._control.project_foreground_conversation().session_id != target.session_id
                or not display.is_vertical_scroll_end
                or "Restored line 59" not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert not app.query_one("#new-content").display
        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert display.is_vertical_scroll_end
        input_area = app.query_one("#conversation-input", TextArea)
        assert app.screen.focused is input_area
        await pilot.press("up")
        assert input_area.text == "/resume"

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        size=(60, 20),
    )


@pytest.mark.asyncio
async def test_resume_projects_unknown_reversed_and_unclassifiable_history_safely(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
            new_uuid=uuid4,
        )
        timestamp = "2027-01-01T00:00:00.000+00:00"
        seed_session_state(
            target,
            messages=[
                {
                    "role": "assistant",
                    "content": "Before the first user message.",
                    "timestamp": timestamp,
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                {
                    "role": "tool",
                    "content": "private pre-user result",
                    "timestamp": timestamp,
                    "tool_call_id": "orphan-pre-user",
                    "name": "read_file",
                    "status": "success",
                    "artifact": None,
                },
                {
                    "role": "user",
                    "content": "Question with an orphan tool result.",
                    "timestamp": timestamp,
                },
                {
                    "role": "tool",
                    "content": "private orphan result",
                    "timestamp": timestamp,
                    "tool_call_id": "orphan-in-run",
                    "name": "read_file",
                    "status": "success",
                    "artifact": None,
                },
                {
                    "role": "user",
                    "content": "Wait for the operation.",
                    "timestamp": timestamp,
                },
                {
                    "role": "assistant",
                    "content": "Still waiting for a result.",
                    "timestamp": "2027-01-01T00:00:01.000+00:00",
                    "tool_calls": [{"id": "call-pending", "name": "read_file", "arguments": "{}"}],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                {
                    "role": "user",
                    "content": "Run with reversed timestamps.",
                    "timestamp": "2027-01-01T00:00:10.000+00:00",
                },
                {
                    "role": "assistant",
                    "content": "Historical activity.",
                    "timestamp": "2027-01-01T00:00:05.000+00:00",
                    "tool_calls": [{"id": "call-reversed", "name": "read_file", "arguments": "{}"}],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ],
            metadata={
                "title": "Historical edge cases",
                "token_usage": {
                    "model_calls": 3,
                    "input_tokens": 3,
                    "output_tokens": 3,
                    "total_tokens": 6,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        async with asyncio.timeout(3):
            while "Historical activity." not in _visible_screen_text(app):
                await pilot.pause()

        visible_text = _visible_screen_text(app)
        assert "Before the first user message." in visible_text
        assert "Question with an orphan tool result." in visible_text
        assert visible_text.count("Completed: read_file") == 2
        assert "private pre-user result" not in visible_text
        assert "private orphan result" not in visible_text
        groups = list(app.query(".agent-run-activity-group"))
        assert len(groups) == 2
        headings = [
            str(group.query_one(".agent-run-activity-heading", Static).content) for group in groups
        ]
        assert headings == ["\u25bc Activity | 1s", "\u25bc Activity | 0s"]
        assert all(group.query_one(".agent-run-activity-content").display for group in groups)
        assert "Turn cancelled." not in visible_text
        assert "Turn failed." not in visible_text
        assert "Completed with no response." not in visible_text
        await pilot.click(".agent-run-activity-heading")
        assert not groups[0].query_one(".agent-run-activity-content").display

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        size=(100, 30),
    )


@pytest.mark.asyncio
async def test_resume_stale_selection_preserves_current_display_and_interaction(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        initial = app._control
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: NOW,
            new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Should not be restored.",
                    "timestamp": "2026-08-11T12:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Stale target",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()
        target_path = target.workspace_state.sessions_directory / f"{target.session_id}.jsonl"

        await pilot.press(*list("/status"), "enter")
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        target_path.unlink()
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "model_invalid_request:" not in _visible_screen_text(app):
                await pilot.pause()
        failure_text = _visible_screen_text(app)
        input_area = app.query_one("#conversation-input", TextArea)
        assert app._control is initial
        assert app.is_running
        assert not input_area.read_only
        assert "Should not be restored." not in failure_text
        assert "not\nresumable." in failure_text

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
    )


@pytest.mark.asyncio
async def test_fatal_resume_failure_exits_without_rendering_private_error(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_error = "reset secret C:\\sensitive\\bus"
    visible_after_failure = ""

    async def fail_resume(
        _dispatcher: RemoteManagementCommandDispatcher, _session_id: str, *, force: bool = False
    ) -> ManagementCommandResult:
        del force
        raise FatalManagementError(
            ErrorInfo("persistence_error", "Runtime Session replacement could not be completed.")
        ) from RuntimeError(private_error)

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        nonlocal visible_after_failure
        target = Session.create(
            WorkspaceState(workspace),
            now=lambda: NOW,
            new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
        )
        seed_session_state(
            target,
            messages=[
                {
                    "role": "user",
                    "content": "Must not be rendered after fatal replacement failure.",
                    "timestamp": "2026-08-11T12:00:00.000+00:00",
                }
            ],
            metadata={
                "title": "Fatal target",
                "token_usage": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "summary": "",
            },
            last_compacted=0,
        )
        target.close()

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        async with asyncio.timeout(3):
            while app.is_running:
                await pilot.pause()
        visible_after_failure = _visible_screen_text(app)
        assert isinstance(app.fatal_management_error, FatalManagementError)

    monkeypatch.setattr(RemoteManagementCommandDispatcher, "resume", fail_resume)
    with pytest.raises(FatalManagementError) as raised:
        await _run_cli_terminal_case(
            agent_home=agent_home,
            workspace=workspace,
            monkeypatch=monkeypatch,
            scenario=scenario,
        )

    assert raised.value.error.code == "persistence_error"
    assert private_error not in str(raised.value)
    assert private_error not in visible_after_failure
    assert "Must not be rendered after fatal replacement failure." not in visible_after_failure


@pytest.mark.asyncio
async def test_terminal_conversation_starts_blank_and_focuses_input() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)):
        visible_text = _visible_screen_text(app)

        assert "Message Aide" in visible_text
        assert "Welcome" not in visible_text
        assert "Session" not in visible_text
        assert "Ready" in visible_text
        assert isinstance(app.screen.focused, TextArea)
        assert runtime.start_calls == 1

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_up_drain_keeps_an_inbound_consumed_during_the_atomic_drain() -> None:
    class CoordinatedDrainBus(MessageBus):
        def __init__(self) -> None:
            super().__init__()
            self.drain_started = asyncio.Event()
            self.allow_drain = asyncio.Event()

        async def drain_inbound(self) -> tuple[InboundMessage, ...]:
            self.drain_started.set()
            await self.allow_drain.wait()
            return await super().drain_inbound()

    bus = CoordinatedDrainBus()
    control = _DirectControl()

    app = TerminalConversationApp(
        bus=bus,
        control=control,
        management_dispatcher=_management_dispatcher(),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await pilot.press(*list("second"), "enter")

        drain = asyncio.create_task(pilot.press("up"))
        await asyncio.wait_for(bus.drain_started.wait(), timeout=1)
        consumed = await bus.get_inbound()
        assert consumed == InboundMessage(content="first")
        bus.allow_drain.set()
        await asyncio.wait_for(drain, timeout=1)

        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="done",
                metadata={"_streamed": True},
            )
        )
        async with asyncio.timeout(1):
            while not app.query(".user-message"):
                await pilot.pause()

        assert app.query_one("#conversation-input", TextArea).text == "second"
        user_messages = [
            str(cast(Static, message).content) for message in app.query(".user-message")
        ]
        assert user_messages == ["first"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"tool_call_id": "call-malformed"},
        {"tool_call_id": "call-malformed", "status": "success", "result": 1},
        {"tool_call_id": "call-malformed", "status": "success", "unexpected": True},
    ],
)
async def test_sparse_protocol_errors_and_duplicate_terminal_do_not_poison_next_run(
    metadata: dict[str, object],
) -> None:
    bus = MessageBus()
    control = _DirectControl()

    app = TerminalConversationApp(
        bus=bus,
        control=control,
        management_dispatcher=_management_dispatcher(),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("malformed"), "enter")
        assert await bus.get_inbound() == InboundMessage(content="malformed")
        await bus.put_outbound(
            OutboundMessage(
                type="tool_call",
                content="read_file",
                metadata=metadata,
            )
        )
        async with asyncio.timeout(1):
            while "Turn failed." not in _visible_screen_text(app):
                await pilot.pause()

        await bus.put_outbound(
            OutboundMessage(
                type="system_control",
                content="duplicate",
                metadata={"_streamed": True, "finish_reason": "failed"},
            )
        )
        await pilot.pause()

        await pilot.press(*list("next"), "enter")
        assert await bus.get_inbound() == InboundMessage(content="next")
        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="next answer",
                metadata={"_stream_delta": True},
            )
        )
        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="",
                metadata={"_streamed": True},
            )
        )
        async with asyncio.timeout(1):
            while "next answer" not in _visible_screen_text(app):
                await pilot.pause()

        assert _visible_screen_text(app).count("next answer") == 1


@pytest.mark.asyncio
async def test_terminal_conversation_inherits_the_terminal_background() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)):
        input_area = app.screen.focused

        assert isinstance(input_area, TextArea)
        assert app.screen.styles.background.a == 0
        assert input_area.styles.background.a == 0


@pytest.mark.asyncio
async def test_nonblank_enter_echoes_user_before_consuming_agent_events() -> None:
    app: TerminalConversationApp
    conversation = ScriptedRunSource(pause_before_output=True)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.before_output.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "hi" not in _visible_screen_text(app):
                    await pilot.pause()

            assert conversation.submissions == ["hi"]
            assert "hi" in _visible_screen_text(app)
            assert runtime.inbound_history == [InboundMessage(content="hi")]
        finally:
            conversation.continue_to_output()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_unknown_slash_input_remains_an_ordinary_foreground_turn() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/unknown"), "enter")
        await _wait_for_turn(app)

        assert conversation.submissions == ["/unknown"]
        assert runtime.inbound_history == [InboundMessage(content="/unknown")]


@pytest.mark.asyncio
async def test_active_turn_keeps_input_editable_and_cancellable_before_a_later_turn() -> None:
    conversation = CancellableRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
        await asyncio.sleep(0)

        input_area = app.query_one("#conversation-input", TextArea)
        working = app.query_one("#status-bar", Static)
        assert not input_area.read_only
        async with asyncio.timeout(1):
            while not str(working.content).startswith("Working"):
                await asyncio.sleep(0)
        assert str(working.content).startswith("Working")

        for text in ("queued one", "queued two", "queued three"):
            await pilot.press(*list(text), "enter")
        assert conversation.submissions == ["first"]
        assert "Pending (3)" in _visible_screen_text(app)
        assert "+1 more" in _visible_screen_text(app)

        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        async with asyncio.timeout(1):
            while conversation.submissions != [
                "first",
                "queued one",
                "queued two",
                "queued three",
            ]:
                await asyncio.sleep(0)
        await _wait_for_turn(app)

        assert conversation.cancel_calls == 1
        assert not input_area.read_only
        assert str(working.content).startswith("Ready")
        await pilot.press("ctrl+home")
        visible_text = _visible_screen_text(app)
        assert "partial response" in visible_text
        assert "Turn cancelled." in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert conversation.submissions == [
            "first",
            "queued one",
            "queued two",
            "queued three",
            "again",
        ]
        assert "Recovered response." in _visible_screen_text(app)
        assert all(message.metadata == {} for message in runtime.inbound_history)


@pytest.mark.asyncio
async def test_pending_queue_summary_stays_two_lines_at_narrow_width() -> None:
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))

    async with app.run_test(size=(40, 15)) as pilot:
        queue = app.query_one("#pending-queue", Static)
        for count in range(4):
            app._pending_inputs.clear()
            app._pending_inputs.extend(
                ("first line\n" + "x" * 90, "second line", "third line")[:count]
            )
            app._refresh_pending_queue()
            await pilot.pause()
            if count == 0:
                assert not queue.display
                continue
            assert queue.display
            assert f"Pending ({count})" in str(queue.content)
            assert queue.outer_size.height <= 2
            assert "first line" in str(queue.content)
            if count == 3:
                assert "+1 more" in str(queue.content)
                assert "third line" not in str(queue.content)


@pytest.mark.asyncio
async def test_up_drains_pending_fifo_and_reenter_submits_one_multiline_inbound() -> None:
    conversation = CancellableRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
        for text in ("queued one", "queued two", "queued three"):
            await pilot.press(*list(text), "enter")

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert input_area.text == "queued one\nqueued two\nqueued three"
        assert "Pending (" not in _visible_screen_text(app)
        assert conversation.submissions == ["first"]
        assert runtime.inbound_history == [InboundMessage(content="first")]

        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)
        await pilot.press("enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first", "queued one\nqueued two\nqueued three"]
    assert runtime.inbound_history == [
        InboundMessage(content="first"),
        InboundMessage(content="queued one\nqueued two\nqueued three"),
    ]
    assert all(message.metadata == {} for message in runtime.inbound_history)


@pytest.mark.asyncio
async def test_repeated_or_delayed_active_ctrl_c_stays_bound_to_the_cancelled_turn() -> None:
    conversation = CancellableRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)
        await pilot.press("ctrl+c")

        assert conversation.cancel_calls == 1
        assert app.is_running
        assert input_area.text == ""
        assert not input_area.read_only
        assert app.screen.focused is input_area

        await pilot.press("x", "backspace", "ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_c_clears_an_idle_draft_without_exiting() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("draft"))
        await pilot.press("ctrl+c")

        assert input_area.text == ""
        assert app.is_running


@pytest.mark.asyncio
async def test_ctrl_c_on_empty_idle_input_exits() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_deletes_forward_when_draft_is_nonempty() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("draft"))
        input_area.move_cursor((0, 0))
        await pilot.press("ctrl+d")

        assert input_area.text == "raft"
        assert app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_on_empty_input_exits() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_during_an_active_turn_settles_stream_before_runtime_close() -> None:
    conversation = BlockingRunSource()
    runtime = CloseOrderingBackend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("active"), "enter"))
        await asyncio.wait_for(conversation.started.wait(), timeout=1)
        await pilot.press("ctrl+d")
        await asyncio.wait_for(submission, timeout=1)
        await pilot.pause()

        assert not app.is_running

    assert conversation.closed.is_set()
    assert runtime.close_saw_stream_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [" EXIT ", " qUiT "])
async def test_exit_and_quit_commands_exit_without_submitting(command: str) -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await pilot.pause()

        assert conversation.submissions == []
        assert not app.is_running


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["exit now", "quitter"])
async def test_exit_like_text_is_submitted_as_an_ordinary_turn(command: str) -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await _wait_for_turn(app)

        assert conversation.submissions == [command]
        assert app.is_running


@pytest.mark.asyncio
async def test_multiline_submission_preserves_text_and_ctrl_j_inserts_a_newline() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first line"), "ctrl+j", *list("second line"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first line\nsecond line"]
    assert runtime.inbound_history == [InboundMessage(content="first line\nsecond line")]


@pytest.mark.asyncio
async def test_supported_modifier_enter_sequences_insert_newlines_without_submitting() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("first line"))
        input_area.post_message(Key("\x1b[13;66u", None))
        await pilot.pause()
        await pilot.press(*list("second line"))
        input_area.post_message(Key("\x1b[13;67u", None))
        await pilot.pause()
        await pilot.press(*list("third line"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first line\nsecond line\nthird line"]


@pytest.mark.asyncio
async def test_textual_driver_lifecycle_balances_enhanced_keyboard_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    KeyboardLifecycleDriver.operations = []
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = KeyboardLifecycleDriver

    async def exit_when_ready(_: object) -> None:
        app.exit()

    await app.run_async(headless=False, size=(80, 24), auto_pilot=exit_when_ready)

    keyboard_writes = [
        value
        for operation, value in KeyboardLifecycleDriver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]
    assert KeyboardLifecycleDriver.operations[-1] == ("close", "")
    assert runtime.start_calls == 1
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_malformed_enhanced_keyboard_report_does_not_break_ordinary_submission() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Key("\x1b[13;" + ("9" * 5000) + "u", None))
        await pilot.pause()
        await pilot.press(*list("still works"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["still works"]


@pytest.mark.asyncio
async def test_whitespace_only_submission_is_ignored() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Paste(" \n\t ").stop())
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert conversation.submissions == []
        assert input_area.text == " \n\t "


@pytest.mark.asyncio
async def test_multiline_paste_is_inserted_without_implicit_submission() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Paste("pasted first\npasted second").stop())
        await pilot.pause()

        assert input_area.text == "pasted first\npasted second"
        assert conversation.submissions == []

        await pilot.press("enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["pasted first\npasted second"]


@pytest.mark.asyncio
async def test_multiline_input_grows_to_six_rows_then_scrolls_internally() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_region = app.query_one("#conversation-input-region")

        input_area.post_message(Paste("\n".join(f"line {index}" for index in range(6))).stop())
        await pilot.pause()
        six_row_region_height = input_region.size.height

        assert input_area.size.height == 6
        assert input_area.max_scroll_y == 0

        input_area.post_message(Paste("\nline 6\nline 7").stop())
        await pilot.pause()

        assert input_area.size.height == 6
        assert input_area.max_scroll_y > 0
        assert input_region.size.height == six_row_region_height


@pytest.mark.asyncio
async def test_accepted_submissions_are_recalled_only_within_the_runtime_lifetime() -> None:
    conversation = ScriptedRunSource(
        deltas_by_submission=((), (), ()),
        completed_contents=("", "", ""),
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("second"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert input_area.text == "second"
        await pilot.press("up")
        assert input_area.text == "first"
        await pilot.press("down")
        assert input_area.text == "second"
        await pilot.press("down")
        assert input_area.text == ""

    next_runtime = _terminal_backend(ScriptedRunSource())
    next_app = _terminal_app(cast(Any, next_runtime))
    async with next_app.run_test(size=(80, 24)) as pilot:
        next_input = next_app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert next_input.text == ""


@pytest.mark.asyncio
async def test_scrolling_conversation_keeps_composer_focus_and_draft() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("draft"))
        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=3)

        input_area = app.query_one("#conversation-input", TextArea)
        assert input_area.text == "draft"
        assert isinstance(app.screen.focused, TextArea)


@pytest.mark.asyncio
async def test_text_deltas_update_one_assistant_markdown_and_terminal_marker_closes() -> None:
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        completed_content="First answer.",
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "First" not in _visible_screen_text(app):
                    await pilot.pause()
            partial_text = _visible_screen_text(app)

            assert partial_text.count("First") == 1
            assert "answer." not in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

        completed_text = _visible_screen_text(app)
        assert completed_text.count("First answer.") == 1
        assert not list(app.query(".agent-run-activity-group"))

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_intermediate_model_output_and_tools_share_one_activity_group() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment("Planning **the tool call**."),
            _tool_call(
                "call-read-file",
                "read_file",
                "Running read_file",
            ),
            _response_delta("Final **answer**."),
            _response_segment("Final **answer**."),
            _completed_response(),
        )
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        groups = list(app.query(".agent-run-activity-group"))
        assert len(groups) == 1
        group = groups[0]
        activity_content = group.query_one(".agent-run-activity-content")
        assert not activity_content.display
        assert len(list(activity_content.query(".assistant-row"))) == 1
        assert len(list(activity_content.query(".tool-row"))) == 1
        activity_markdown = activity_content.query_one(".assistant-row").query_one(Markdown)
        assert "Planning" in activity_markdown.source

        display = app.query_one("#conversation-display")
        direct_assistant_rows = [
            row for row in display.query(".assistant-row") if row.parent is display
        ]
        if len(direct_assistant_rows) != 1:
            raise AssertionError(
                f"history={[(item.type, item.content, item.metadata) for item in runtime.outbound_history]} "
                f"children={[type(child).__name__ for child in display.children]}"
            )
        assert len(direct_assistant_rows) == 1
        final_markdown = direct_assistant_rows[0].query_one(Markdown)
        assert "Final" in final_markdown.source
        assert _visible_screen_text(app).count("Final") == 1


@pytest.mark.asyncio
async def test_intermediate_completion_reparents_stream_candidate_and_empty_output_is_not_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("candidate"),
            _response_segment(""),
            _tool_call(
                "call-empty",
                "read_file",
                "Running read_file",
            ),
            _response_segment(""),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))
    mounted_assistants: list[tuple[Markdown, Widget | None]] = []
    mount_assistant = app._mount_assistant

    async def record_mounted_assistant(*args: object, **kwargs: object) -> Markdown:
        assistant = await mount_assistant(*args, **kwargs)  # type: ignore[arg-type]
        mounted_assistants.append((assistant, cast(Widget | None, assistant.parent)))
        return assistant

    monkeypatch.setattr(app, "_mount_assistant", record_mounted_assistant)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        activity_markdown = activity_content.query(".assistant-row").first().query_one(Markdown)
        assert activity_markdown.source == "candidate"
        assert len(mounted_assistants) == 1
        streamed_markdown, streamed_row = mounted_assistants[0]
        assert activity_markdown is streamed_markdown
        assert activity_markdown.parent is streamed_row
        assert streamed_row is not None
        assert streamed_row.parent is activity_content
        assert len(list(activity_content.query(".assistant-row"))) == 1
        assert len(list(app.query("#conversation-display > .assistant-row"))) == 0


@pytest.mark.asyncio
async def test_tool_start_reclassifies_a_streamed_candidate_without_model_completion() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("Unclassified candidate."),
            _tool_call("call-read", "read_file", "Running read_file"),
            _response_delta("Final answer."),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        assert activity_content.query_one(Markdown).source == "Unclassified candidate."
        assert _tool_row_texts(app) == ["Status unavailable: read_file"]
        assert app.query("#conversation-display > .assistant-row").first().query_one(
            Markdown
        ).source == ("Final answer.")
        assert not activity_content.display


@pytest.mark.asyncio
async def test_direct_terminal_marker_preserves_the_current_candidate() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("streamed candidate"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert app.query("#conversation-display > .assistant-row").first().query_one(
            Markdown
        ).source == ("streamed candidate")
        assert "streamed candidate" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_first_terminal_event_finishes_without_waiting_for_more_events() -> None:
    conversation = TerminalThenBlockingRunSource()
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await asyncio.wait_for(conversation.terminal_emitted.wait(), timeout=1)
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert not group.query_one(".agent-run-activity-content").display
        assert "final response" in _visible_screen_text(app)
        assert _tool_row_texts(app) == ["Status unavailable: read_file"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        (
            _response_segment("process activity"),
            _response_segment("process activity"),
            _tool_call(
                "call-read",
                "read_file",
                "Running read_file",
            ),
            _completed_response(),
        ),
        (
            _response_delta("process activity"),
            _tool_call(
                "call-read",
                "read_file",
                "Running read_file",
            ),
            _response_segment("process activity"),
            _completed_response(),
        ),
    ],
)
async def test_duplicate_or_late_model_completion_reconciles_grouped_candidate(
    events: tuple[_ScriptItem, ...],
) -> None:
    conversation = ToolMessageSequenceRunSource(events)
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        content = app.query_one(".agent-run-activity-content")
        assert [markdown.source for markdown in content.query(Markdown)] == ["process activity"]
        assert _tool_row_texts(app) == ["Status unavailable: read_file"]


@pytest.mark.asyncio
async def test_reasoning_transition_reopens_the_completed_response_stream() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            OutboundMessage("model_reasoning", "Reasoning A.", {"_stream_delta": True}),
            OutboundMessage("model_reasoning", "", {"_stream_end": True}),
            _response_delta("First answer."),
            _response_segment(""),
            OutboundMessage("model_reasoning", "Reasoning B.", {"_stream_delta": True}),
            OutboundMessage("model_reasoning", "", {"_stream_end": True}),
            _response_delta("Second answer."),
            _response_segment(""),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assistant_rows = list(app.query("#conversation-display > .assistant-row"))
        assert len(assistant_rows) == 1
        assert assistant_rows[0].query_one(Markdown).source == "First answer.Second answer."
        activity = app.query_one(".agent-run-activity-content")
        assert [markdown.source for markdown in activity.query(Markdown)] == [
            "Reasoning A.",
            "Reasoning B.",
        ]


@pytest.mark.asyncio
async def test_late_delta_after_completed_candidate_does_not_reopen_its_stream() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("authoritative complete content"),
            _response_segment(""),
            _response_delta("late fragment"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assistant = app.query_one("#conversation-display > .assistant-row").query_one(Markdown)
        assert assistant.source == "authoritative complete content"
        assert "late fragment" not in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_successful_empty_terminal_content_shows_status_without_activity() -> None:
    conversation = ScriptedRunSource(deltas=(), completed_content="")
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("empty"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Completed with no response." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_first_terminal_event_remains_authoritative() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("candidate"),
            _cancelled_response("authoritative partial"),
            _failed_response("Late failure."),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(".agent-run-activity-content").query_one(Markdown).source == (
            "candidate"
        )
        visible_text = _visible_screen_text(app)
        assert "Turn cancelled." in visible_text
        assert "Late failure." not in visible_text
        assert "late success" not in visible_text
        assert group.query_one(".agent-run-activity-content").display


@pytest.mark.asyncio
async def test_event_stream_failure_groups_candidate_and_finishes_unfinished_tool() -> None:
    conversation = ExplodingRunSource()
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        assert activity_content.display
        assert activity_content.query_one(Markdown).source == "Unconfirmed candidate."
        assert _tool_row_texts(app) == ["Status unavailable: read_file"]
        assert "event stream failed" in _visible_screen_text(app)
        assert "Unconfirmed candidate." in _visible_screen_text(app)
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert conversation.closed.is_set()


@pytest.mark.asyncio
async def test_event_stream_failure_without_activity_does_not_create_empty_group() -> None:
    conversation = ToolMessageSequenceRunSource(())
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Turn failed." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_failed_no_tool_candidate_becomes_expanded_activity() -> None:
    conversation = ScriptedRunSource(
        deltas=("draft content",),
        outcomes=("failed",),
        failure_message="Model unavailable.",
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(".agent-run-activity-content").query_one(Markdown).source == (
            "draft content"
        )
        assert group.query_one(".agent-run-activity-content").display
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert "Model unavailable." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_failure_before_visible_activity_does_not_create_empty_group() -> None:
    conversation = ScriptedRunSource(
        deltas=(),
        outcomes=("failed",),
        failure_message="Model unavailable.",
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Model unavailable." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_empty_cancelled_content_removes_candidate_without_empty_group() -> None:
    conversation = ScriptedRunSource(
        deltas=("streamed candidate",),
        outcomes=("cancelled",),
        cancelled_content="",
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(Markdown).source == "streamed candidate"
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert "Turn cancelled." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_replacing_the_display_stops_a_frozen_run_timer(
    agent_home: Path,
    workspace: Path,
) -> None:
    runtime, bus = _direct_terminal_loop(
        agent_home,
        workspace,
        _FixedCatalogProvider(()),
    )
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)):
        projection = _AgentRunProjection(app, TURN_ID)
        app._active_run_projection = projection
        projection._start_timing()
        assert projection._timer is not None

        replaced = await app._replace_display_from_session(runtime.session.session_id)

        assert replaced
        assert projection._timer is None


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.99, "0s"),
        (59.99, "59s"),
        (60, "1min 0s"),
        (65.9, "1min 5s"),
        (3605, "1h 0min 5s"),
        (24 * 3600 + 5, "24h 0min 5s"),
    ],
)
def test_activity_duration_uses_floored_accumulated_seconds(elapsed: float, expected: str) -> None:
    assert _format_activity_duration(elapsed) == expected


@pytest.mark.asyncio
async def test_activity_timing_includes_wait_before_first_outbound() -> None:
    clock = [0.0]
    conversation = ToolMessageSequenceRunSource(
        (
            _tool_call("call-read", "read_file", "{}"),
            _completed_response(),
        ),
        pause_before=True,
    )
    app = _terminal_app(
        cast(Any, _terminal_backend(conversation)),
        monotonic=lambda: clock[0],
    )

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.paused.wait(), timeout=1)
        clock[0] = 10.9
        conversation.continue_events.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        heading = app.query_one(".agent-run-activity-heading", Static)
        assert str(heading.content) == "\u25b6 Completed | 10s"


@pytest.mark.asyncio
async def test_activity_heading_starts_with_accumulated_time_and_freezes_on_success() -> None:
    clock = [0.0]
    events = (
        _response_segment("intermediate"),
        _tool_call("call-read", "read_file", "Running read_file"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(events, pause_after=(0, 1))
    app = _terminal_app(
        cast(Any, _terminal_backend(conversation)),
        monotonic=lambda: clock[0],
    )
    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.paused.wait(), timeout=1)
        async with asyncio.timeout(1):
            while app._active_run_projection is None:
                await pilot.pause()
        projection = app._active_run_projection
        assert projection is not None
        async with asyncio.timeout(1):
            while projection._started_at is None or projection._activity_group is None:
                await pilot.pause()

        clock[0] = 5.9
        projection._refresh_elapsed()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()
        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert not heading.can_focus
        assert str(heading.content) == "\u25bc Running | 5s"
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.click(".agent-run-activity-heading")
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        clock[0] = 3605.9
        projection._refresh_elapsed()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()
        assert str(heading.content) == "\u25bc Running | 1h 0min 5s"

        conversation.continue_events.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert str(heading.content) == "\u25b6 Completed | 1h 0min 5s"
        assert not content.display
        projection._refresh_elapsed()
        assert str(heading.content) == "\u25b6 Completed | 1h 0min 5s"


@pytest.mark.asyncio
async def test_manual_activity_disclosure_keeps_heading_position_while_content_grows_downward() -> (
    None
):
    intermediate = "\n".join(f"activity {index:02d} " + "a" * 35 for index in range(45))
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(intermediate),
            _tool_call("call-read", "read_file", "Running read_file"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert not content.display
        async with asyncio.timeout(2):
            while heading.region.y < 0 or heading.region.bottom > app.screen.size.height:
                await pilot.pause()

        await pilot.click(".agent-run-activity-heading")
        async with asyncio.timeout(1):
            while not app.query_one("#new-content").display:
                await pilot.pause()
        expanded_heading_y = heading.region.y
        assert content.display
        display = app.query_one("#conversation-display")
        assert not display.is_vertical_scroll_end
        assert app.query_one("#new-content").display

        # The heading remains in the viewport after the group shrinks, so the
        # next click can still target the same visible control.
        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        assert not content.display
        assert heading.region.y == expanded_heading_y
        assert display.is_vertical_scroll_end

        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        assert content.display
        assert heading.region.y == expanded_heading_y
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_failed_activity_group_is_mouse_toggleable_without_moving_composer_focus() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment("intermediate"),
            _tool_call("call-read", "read_file", "Running read_file"),
            _failed_response("The Agent Run failed."),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert heading.can_focus
        assert str(heading.content).startswith("\u25bc Failed | ")
        assert heading.has_class("-failed")
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.press("enter")
        assert content.display
        heading.focus()
        await pilot.press("enter")
        assert not content.display
        await pilot.press("space")
        assert content.display
        app.query_one("#conversation-input", TextArea).focus()
        await pilot.click(".agent-run-activity-heading")
        assert not content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.click(".agent-run-activity-heading")
        assert content.display
        await pilot.click(".tool-row")
        assert content.display


@pytest.mark.asyncio
async def test_tool_confirmation_defaults_to_decline_and_shows_effective_operation() -> None:
    conversation = ConfirmationRunSource(
        details={"path": "outside.txt", "limit": 20},
        reason="The path resolves outside the current Workspace.",
        warnings=("Review the target before allowing access.",),
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert str(app.screen.query_one("#confirmation-heading", Static).content) == (
            "Tool Confirmation"
        )
        tool_label = app.screen.query_one("#confirmation-tool", Static)
        assert str(tool_label.content) == "Tool: Read File"
        assert tool_label.region.height > 0
        assert 0 <= tool_label.region.y < app.screen.size.height
        assert "Reason: The path resolves outside the current" in visible_text
        assert "Workspace." in visible_text
        assert "Warning: Review the target before allowing access." in visible_text
        assert "Path: outside.txt" in visible_text
        assert "Limit: 20" in visible_text
        assert '"path"' not in visible_text
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        assert conversation.responses == []

        await pilot.press("tab")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("shift+tab")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24), (120, 32)))
async def test_tool_confirmation_actions_stay_visible_with_long_details(
    size: tuple[int, int],
) -> None:
    conversation = ConfirmationRunSource(
        details={"path": "nested/" + "long-name-" * 24, "limit": 20},
        reason="Review access to the requested path. " * 8,
        warnings=("The requested path is outside the Workspace. " * 4,),
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=size) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        decline = app.screen.query_one("#confirmation-decline", Button)
        approve = app.screen.query_one("#confirmation-approve", Button)
        details = app.screen.query_one("#confirmation-details-scroll")
        assert app.screen.focused is decline
        assert decline.region.width > 0 and approve.region.width > 0
        assert decline.region.bottom <= size[1] and approve.region.bottom <= size[1]
        assert not decline.region.overlaps(approve.region)
        if size == (40, 15):
            assert details.max_scroll_y > 0
            details.scroll_end(animate=False)
            await pilot.pause()
            assert details.scroll_y > 0
            assert decline.region.bottom <= size[1]
        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)
        assert conversation.responses[0][1] == "declined"


@pytest.mark.asyncio
async def test_coordinator_background_confirmation_uses_stable_modal_projection() -> None:
    coordinator = ToolConfirmationCoordinator()
    runtime = _terminal_backend(ConfirmationRunSource())
    app = _terminal_app(
        runtime,
        confirmation_coordinator=coordinator,
    )
    request = ConfirmationRequest(
        confirmation_id=uuid4(),
        tool_call_id="background-call",
        tool_name="write_file",
        summary="Confirm background write",
        reason="A background Job requires confirmation.",
        details={"path": "nightly.txt"},
    )
    envelope = ConfirmationEnvelope(
        request=request,
        origin="background",
        owner=BackgroundConfirmationOwner(uuid4(), "job-1", uuid4()),
        job_id="job-1",
        title="Nightly backup",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        pending = asyncio.create_task(coordinator.request(envelope))
        await _wait_for_confirmation(app, pilot)

        assert str(app.screen.query_one("#confirmation-heading", Static).content) == (
            "Background Tool Confirmation"
        )
        assert "Source: job-1 + Nightly backup" in _visible_screen_text(app)
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        input_area = app._conversation_input
        display = app._conversation_display
        assert input_area is not None
        assert display is not None
        assert input_area.read_only
        projection = _AgentRunProjection(app, uuid4(), display=display)
        app._active_run_projection = projection
        projection.start()
        await projection.consume(
            OutboundMessage(
                "model_response",
                "Streaming under modal",
                {"_stream_delta": True},
            )
        )
        await projection.consume(
            OutboundMessage(
                "tool_call",
                "write_file",
                {"tool_call_id": "foreground-call", "arguments": '{"path":"note.txt"}'},
            )
        )
        await projection.consume(
            OutboundMessage(
                "tool_call",
                "write_file",
                {"tool_call_id": "foreground-call", "status": "success"},
            )
        )
        await projection.consume(
            OutboundMessage(
                "system_control",
                "Foreground failed under modal",
                {
                    "finish_reason": "failed",
                    "error_code": "model_failed",
                    "_streamed": True,
                },
            )
        )

        assert any(
            markdown.source == "Streaming under modal" for markdown in display.query(Markdown)
        )
        tool_rows = [row for row in display.query(".tool-row") if isinstance(row, Static)]
        assert len(tool_rows) == 1
        assert str(tool_rows[0].content) == "Completed: write_file"
        assert any(
            str(widget.content) == "Foreground failed under modal"
            for widget in display.query(".turn-status")
            if isinstance(widget, Static)
        )
        assert projection.terminal_seen

        await pilot.press("right", "enter")
        assert await asyncio.wait_for(pending, timeout=1) == "approved"
        await pilot.pause()
        assert not input_area.read_only

    await coordinator.close()


@pytest.mark.asyncio
async def test_coordinator_cancellation_before_message_delivery_cannot_open_a_stale_modal() -> None:
    instances: list[Any] = []

    class DelayedConfirmationApp(TerminalConversationApp):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.confirmation_posted = asyncio.Event()
            self.deferred_confirmation: Message | None = None
            instances.append(self)

        def post_message(self, message: Message) -> bool:
            if isinstance(message, self.CoordinatorConfirmationRequested):
                self.deferred_confirmation = message
                self.confirmation_posted.set()
                return True
            return super().post_message(message)

        def release_confirmation(self) -> None:
            message = self.deferred_confirmation
            assert message is not None
            self.deferred_confirmation = None
            assert super().post_message(message)

    coordinator = ToolConfirmationCoordinator()
    runtime = _terminal_backend(ConfirmationRunSource())
    app = _terminal_app(
        runtime,
        app_type=DelayedConfirmationApp,
        confirmation_coordinator=coordinator,
    )
    envelope = ConfirmationEnvelope(
        request=ConfirmationRequest(
            confirmation_id=uuid4(),
            tool_call_id="cancelled-call",
            tool_name="write_file",
            summary="Confirm cancelled write",
            reason="This request will be cancelled before delivery.",
            details={"path": "cancelled.txt"},
        ),
        origin="background",
        owner=BackgroundConfirmationOwner(uuid4(), "job-1", uuid4()),
        job_id="job-1",
        title="Cancelled job",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        delayed = instances[0]
        pending = asyncio.create_task(coordinator.request(envelope))
        await delayed.confirmation_posted.wait()
        await coordinator.cancel_owner(envelope.owner)
        with pytest.raises(ConfirmationAborted):
            await pending

        delayed.release_confirmation()
        await pilot.pause()
        assert not app.query("#confirmation-heading")
        assert app._active_confirmation_token is None
        assert not app._dismissed_confirmation_tokens

    await coordinator.close()


@pytest.mark.asyncio
async def test_coordinator_open_modal_is_aborted_and_drained_on_unmount() -> None:
    coordinator = ToolConfirmationCoordinator()
    runtime = _terminal_backend(ConfirmationRunSource())
    app = _terminal_app(runtime, confirmation_coordinator=coordinator)
    envelope = ConfirmationEnvelope(
        request=ConfirmationRequest(
            confirmation_id=uuid4(),
            tool_call_id="unmount-call",
            tool_name="write_file",
            summary="Confirm unmount write",
            reason="This request remains open until unmount.",
            details={"path": "unmount.txt"},
        ),
        origin="background",
        owner=BackgroundConfirmationOwner(uuid4(), "job-1", uuid4()),
        job_id="job-1",
        title="Unmount job",
    )
    pending: asyncio.Task[ConfirmationDecision]

    async with app.run_test(size=(80, 24)) as pilot:
        pending = asyncio.create_task(coordinator.request(envelope))
        await _wait_for_confirmation(app, pilot)

    with pytest.raises(ConfirmationAborted):
        await pending
    assert coordinator._active is None
    assert not coordinator._foreground
    assert not coordinator._background
    await coordinator.close()


@pytest.mark.asyncio
async def test_write_confirmation_hides_content_and_unknown_details() -> None:
    secret = "Authorization: Bearer sk-sensitive-value"
    conversation = ConfirmationRunSource(
        tool_name="write_file",
        details={"path": "outside.txt", "content": secret, "internal": "raw-result"},
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(42, 16)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("write"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert str(app.screen.query_one("#confirmation-tool", Static).content) == "Tool: Write File"
        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert "Path: outside.txt" in details
        assert f"Content: {len(secret)} characters" in details
        assert "sk-sensitive-value" not in visible_text
        assert "raw-result" not in visible_text
        assert all("sk-sensitive-value" not in detail for detail in details)
        assert all("raw-result" not in detail for detail in details)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_mcp_confirmation_shows_stable_identity_and_complete_arguments() -> None:
    arguments = {
        "nested": {"unicode": "\u5b8c\u6574\u53c2\u6570", "nullable": None},
        "items": [1, True, {"value": "x" * 512}],
    }
    identity = MCPToolIdentity(
        server_name="calendar-server",
        remote_name="create_event",
        model_name="mcp_calendar-server_create_event",
    )
    conversation = ConfirmationRunSource(
        tool_name=identity.model_name,
        details=arguments,
        reason="MCP Tool invocation requires confirmation.",
        mcp_identity=identity,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(100, 30)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("create event"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert details == [
            "MCP Server: calendar-server",
            "Remote Tool: create_event",
            "Model Tool: mcp_calendar-server_create_event",
            "Arguments: " + json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        ]

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_web_fetch_confirmation_redacts_url_credentials_and_query_values() -> None:
    conversation = ConfirmationRunSource(
        tool_name="web_fetch",
        details={
            "url": "http://user:password@127.0.0.1/private?token=secret-value",
            "format": "markdown",
        },
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("fetch"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert "URL: http://127.0.0.1/private?<redacted>" in details
        assert "Format: markdown" in details
        assert all("password" not in detail for detail in details)
        assert all("secret-value" not in detail for detail in details)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_exec_confirmation_shows_exact_command_and_arrow_keys_select_approval() -> None:
    command = 'rm -rf "build output" && printf done'
    conversation = ConfirmationRunSource(
        tool_name="exec",
        details={"command": command, "cwd": ".", "timeout": 45},
        reason="The Exec command matches a known destructive operation.",
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(100, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert f"Command: {command}" in visible_text
        assert "CWD: ." in visible_text
        assert "Timeout: 45" in visible_text
        assert "The Exec command matches a known destructive operation." in visible_text
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)

        await pilot.press("right")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("left")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("down")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("up")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("right")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "approved"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "decision"),
    [
        ("confirmation-approve", "approved"),
        ("confirmation-decline", "declined"),
    ],
)
async def test_confirmation_buttons_resolve_the_pending_tool_with_mouse(
    button_id: str,
    decision: ConfirmationDecision,
) -> None:
    conversation = ConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        assert await pilot.click(f"#{button_id}")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), decision),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("escape", "ctrl+c"))
async def test_confirmation_escape_and_ctrl_c_decline_only_the_pending_tool(key: str) -> None:
    conversation = ConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        input_area = app.query_one("#conversation-input", TextArea)

        await pilot.press(key)
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert app.is_running
        assert app.screen.focused is input_area
        assert not input_area.read_only

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
async def test_clicking_outside_confirmation_keeps_it_open_without_a_decision() -> None:
    conversation = ConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        assert await pilot.click(offset=(1, 1))
        await pilot.pause()
        assert conversation.responses == []
        assert "Tool: Read File" in _visible_screen_text(app)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
async def test_duplicate_late_confirmation_is_shown_once_and_does_not_fail_the_turn() -> None:
    conversation = DuplicateLateConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await _wait_for_confirmation(app, pilot)
        assert conversation.confirmation_requested.is_set()
        await pilot.press("right", "enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert "A foreground turn failed" not in _visible_screen_text(app)

    assert conversation.responses == [(conversation.request.confirmation_id, "approved")]


@pytest.mark.asyncio
async def test_multiple_confirmations_are_resolved_in_order_with_a_fresh_safe_default() -> None:
    conversation = MultipleConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested[0].wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        await pilot.press("right", "enter")

        await asyncio.wait_for(conversation.confirmation_requested[1].wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")

        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (conversation.requests[0].confirmation_id, "approved"),
        (conversation.requests[1].confirmation_id, "declined"),
    ]


@pytest.mark.asyncio
async def test_application_teardown_cancels_an_open_confirmation_without_a_decision() -> None:
    conversation = ConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    submission: asyncio.Task[None]

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        assert conversation.responses == []

    await asyncio.gather(submission, return_exceptions=True)
    assert conversation.responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("include_result", [False, True])
@pytest.mark.parametrize(
    ("status", "expected"),
    [("success", "Completed: work"), ("error", "Failed: work"), ("refused", "Rejected: work")],
)
async def test_tool_completion_updates_only_its_row_without_showing_results(
    status: str,
    expected: str,
    include_result: bool,
) -> None:
    completion_metadata: dict[str, object] = {"tool_call_id": "first", "status": status}
    if include_result:
        completion_metadata["result"] = "Private tool result."
    conversation = ToolMessageSequenceRunSource(
        (
            _tool_call("first", "work", "{}"),
            OutboundMessage("tool_call", "work", completion_metadata),
            OutboundMessage("tool_call", "work", {"tool_call_id": "first", "status": "success"}),
            _tool_call("second", "work", "{}"),
            OutboundMessage("tool_call", "work", {"tool_call_id": "second", "status": "success"}),
            _response_delta("Final answer."),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("go"), "enter")
        await _wait_for_turn(app)
        rows = _tool_row_texts(app)
        assert len(rows) == 2
        assert rows[0].startswith(expected)
        assert rows[1] == "Completed: work"
        assert all("Running:" not in row and "Arguments:" not in row for row in rows)
        assert "Private tool result." not in _visible_screen_text(app)
        assert "Final answer." in _visible_screen_text(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_unfinished_tool_row_does_not_claim_success_after_run_termination(
    cancelled: bool,
) -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _tool_call("finished", "first", "{}"),
            OutboundMessage(
                "tool_call", "first", {"tool_call_id": "finished", "status": "success"}
            ),
            _tool_call("unfinished", "second", "{}"),
            OutboundMessage(
                "system_control",
                "Turn stopped.",
                {"_streamed": True, "finish_reason": "cancelled" if cancelled else "failed"},
            ),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("go"), "enter")
        await _wait_for_turn(app)
        assert _tool_row_texts(app) == [
            "Completed: first",
            "Cancelled: second" if cancelled else "Status unavailable: second",
        ]


@pytest.mark.asyncio
async def test_tool_activity_renders_raw_arguments_until_terminal_marker() -> None:
    conversation = ToolActivityRunSource(
        start_summary='Running read_file {"arguments":{"path":"C:/private.txt"}}',
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.tool_started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while not app.query(".tool-row"):
                await pilot.pause()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()

        running_text = _visible_screen_text(app)
        assert "Running: read_file" in running_text
        assert "call-read-file" not in running_text
        assert 'Running read_file {"arguments":{"path":"C:/private.txt"}}' in running_text
        row = app.query_one(".tool-row", Static)
        assert row.outer_size.width == app.query_one(".message-row").outer_size.width
        assert row.parent is app.query_one(".agent-run-activity-content")
        assert not row.has_class("message")
        assert not row.can_focus
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)
        running_row_content = str(row.content)
        await pilot.click(".tool-row")
        assert not row.has_focus
        assert str(row.content) == running_row_content

        conversation.complete_tool.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        final_text = _visible_screen_text(app)
        assert "Running: read_file" not in final_text
        assert str(row.content) == "Completed: read_file"
        assert not row.parent.display
        assert "Completed with no response." in final_text


@pytest.mark.asyncio
async def test_tool_rows_isolate_calls_and_turns_without_tool_result_projection() -> None:
    first_turn = (
        _tool_call("completion-first", "glob", "Running glob"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _tool_call("missing-completion", "write_file", "Running write_file"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _completed_response(),
    )
    second_turn = (
        _tool_call("shared-call", "read_file", "Running read_file"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(first_turn, second_turn)
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        assert _tool_row_texts(app) == [
            "Status unavailable: glob",
            "Status unavailable: read_file",
            "Status unavailable: write_file",
        ]

        await pilot.press(*list("second"), "enter")
        await _wait_for_turn(app)
        assert _tool_row_texts(app) == [
            "Status unavailable: glob",
            "Status unavailable: read_file",
            "Status unavailable: write_file",
            "Status unavailable: read_file",
        ]
        assert "completion-first" not in _visible_screen_text(app)
        assert "shared-call" not in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_tool_row_updates_preserve_historical_follow_and_resize_state() -> None:
    first_events: list[_ScriptItem] = []
    for index in range(24):
        tool_name = f"tool_{index:02d}"
        tool_call_id = f"call-{index:02d}"
        first_events.append(_tool_call(tool_call_id, tool_name, f"Running {tool_name}"))
    first_events.append(_failed_response("The Agent Run failed."))
    second_events = (
        _tool_call("later-call", "web_search", "Running web_search"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(
        tuple(first_events),
        second_events,
        pause_after=(1, 0),
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()
        assert display.is_vertical_scroll_end

        await pilot.press("pageup", "pageup")
        assert not display.is_vertical_scroll_end
        submission = asyncio.create_task(pilot.press(*list("later"), "enter"))
        try:
            await asyncio.wait_for(conversation.paused.wait(), timeout=1)
            await pilot.pause()
            historical_scroll_y = display.scroll_y
            async with asyncio.timeout(1):
                while not app.query_one("#new-content").display:
                    await pilot.pause()
            assert app.query_one("#new-content").display
            conversation.continue_events.set()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)
        finally:
            conversation.continue_events.set()

        assert not display.is_vertical_scroll_end
        assert display.scroll_y == historical_scroll_y
        assert app.query_one("#new-content").display

        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert not display.is_vertical_scroll_end
        message_width = app.query_one(".message-row").outer_size.width
        assert all(
            row.outer_size.width == message_width
            for row in app.query(".tool-row")
            if row.display and row.parent.display
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reading_history", [False, True])
async def test_activity_layout_changes_preserve_follow_or_historical_anchor_at_each_stage(
    reading_history: bool,
) -> None:
    history = "\n".join(f"history {index:02d} " + "h" * 35 for index in range(70))
    second_events = (
        _response_delta("candidate"),
        _response_segment("candidate"),
        _tool_call(
            "activity-call",
            "read_file",
            "Running read_file",
        ),
        _response_segment("latest answer"),
        _completed_response(),
    )
    conversation = StagedToolMessageSequenceRunSource(
        (
            _response_delta(history),
            _completed_response(),
        ),
        second_events,
        pause_after=tuple((1, event_index) for event_index in range(5)),
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("history"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+end")
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        try:
            await conversation.wait_after(1, 0)
            await pilot.pause()
            refreshed = asyncio.Event()
            app.call_after_refresh(refreshed.set)
            await refreshed.wait()
            await _wait_for_refresh_condition(
                app,
                lambda: "candidate" in _visible_screen_text(app),
            )
            await _wait_for_refresh_condition(
                app,
                lambda: display.is_vertical_scroll_end,
            )
            assert display.is_vertical_scroll_end
            historical_position = display.scroll_y
            if reading_history:
                await pilot.press("pageup")
                await _wait_for_refresh_condition(
                    app,
                    lambda: not display.is_vertical_scroll_end,
                )
                historical_position = display.scroll_y
                assert not display.is_vertical_scroll_end

            expected_tool_rows: dict[int, list[str]] = {
                1: [],
                2: ["Running: read_file\nArguments: Running read_file"],
                3: ["Running: read_file\nArguments: Running read_file"],
                4: ["Status unavailable: read_file"],
            }
            for previous_event, next_event in zip(range(4), range(1, 5), strict=True):
                conversation.continue_after(1, previous_event)
                await conversation.wait_after(1, next_event)
                await pilot.pause()
                refreshed = asyncio.Event()
                app.call_after_refresh(refreshed.set)
                await refreshed.wait()
                expected_tool_row = expected_tool_rows[next_event][-1:]
                if expected_tool_row:
                    await _wait_for_refresh_condition(
                        app,
                        partial(_tool_row_tail_matches, app, expected_tool_row),
                    )
                assert _tool_row_texts(app)[-1:] == expected_tool_row
                if reading_history and expected_tool_row:
                    await _wait_for_refresh_condition(
                        app,
                        lambda: app.query_one("#new-content").display,
                    )
                    assert not display.is_vertical_scroll_end
                    assert display.scroll_y == historical_position
                    assert app.query_one("#new-content").display
                elif reading_history:
                    assert not display.is_vertical_scroll_end
                else:
                    await _wait_for_refresh_condition(
                        app,
                        lambda: display.is_vertical_scroll_end,
                    )
                    assert not app.query_one("#new-content").display

            conversation.continue_after(1, 4)
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

            if reading_history:
                assert not display.is_vertical_scroll_end
                assert display.scroll_y == historical_position
                assert app.query_one("#new-content").display
            else:
                assert display.is_vertical_scroll_end
                assert "latest answer" in _visible_screen_text(app)
                assert not app.query_one("#new-content").display
            assert not app.query_one(".agent-run-activity-content").display
        finally:
            conversation.continue_all()
            if not submission.done():
                await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expanded", "reading_history"),
    [(False, False), (False, True), (True, False), (True, True)],
)
async def test_activity_group_resize_preserves_scroll_mode_anchor_and_input_focus(
    expanded: bool,
    reading_history: bool,
) -> None:
    history = "\n".join(f"history {index:02d}" for index in range(80))
    activity = "\n".join(f"activity {index:02d}" for index in range(50))
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(history),
            _completed_response(),
        ),
        (
            _response_segment(activity),
            _tool_call(
                "activity-call",
                "read_file",
                "Running read_file",
            ),
            _completed_response(),
        ),
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(70, 22)) as pilot:
        await pilot.press(*list("history"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        content = app.query_one(".agent-run-activity-content")
        if expanded:
            heading = app.query_one(".agent-run-activity-heading")
            heading.scroll_visible()
            await pilot.pause()
            await pilot.click(heading)
            await pilot.pause()
        assert content.display is expanded

        await pilot.press("pageup" if reading_history else "ctrl+end")
        await pilot.pause()
        scroll_position = display.scroll_y
        assert display.is_vertical_scroll_end is not reading_history

        await pilot.resize_terminal(55, 22)
        await pilot.pause()

        assert content.display is expanded
        assert display.is_vertical_scroll_end is not reading_history
        if reading_history:
            assert display.scroll_y == scroll_position
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_activity_content_uses_main_vertical_scroll_and_keeps_code_horizontal_scroll() -> (
    None
):
    code_line = "very_long_variable_name = " + "0123456789" * 8 + "TAIL"
    intermediate = f"```python\n{code_line}\n```\n\n" + "\n".join(
        f"activity {index:02d}" for index in range(60)
    )
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(intermediate),
            _tool_call("activity-call", "read_file", "Running read_file"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(48, 20)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.05)
        await pilot.press("ctrl+end")
        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        display = app.query_one("#conversation-display")
        markdown = app.query_one(".agent-run-activity-content").query_one(Markdown)
        horizontal_scroll_y = display.scroll_y

        assert "TAIL" not in _visible_screen_text(app)
        await pilot._post_mouse_events(
            [MouseScrollRight],
            offset=(10, markdown.region.y + 2),
            times=30,
        )
        assert "TAIL" in _visible_screen_text(app)
        assert display.scroll_y == horizontal_scroll_y

        await pilot.press("pageup")
        await pilot.pause()
        before_down = display.scroll_y

        await pilot._post_mouse_events(
            [MouseScrollDown],
            offset=(display.region.x + 5, display.region.y + 3),
            times=3,
        )
        assert display.scroll_y > before_down
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_streamed_markdown_preserves_reading_structure_and_link_urls() -> None:
    content = (
        "# Heading\n\n"
        "- first item\n"
        "- second item\n\n"
        "> quoted text\n\n"
        "```python\n"
        "long_value = " + "'x'" * 30 + "\n```\n\n"
        "[documentation](https://example.com/docs)\n"
        "![architecture image](https://example.com/asset.png)\n"
    )
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press(*list("read"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.1)

        visible_text = _visible_screen_text(app)
        assert "Heading" in visible_text
        assert "first item" in visible_text
        assert "second item" in visible_text
        assert "quoted text" in visible_text
        assert "long_value" in visible_text
        assert "documentation (https://example.com/docs)" in visible_text
        assert "architecture image" in visible_text
        assert "https://example.com/asset.png" in visible_text
        assert "@click" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_markdown_structure_is_visible_while_the_fenced_block_is_incomplete() -> None:
    partial = "# Heading\n\n- first item\n\n> quoted text\n\n```python\nvalue = 1"
    content = partial + "\n```\n"
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        deltas=(partial, "\n```\n"),
        completed_content=content,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("progress"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            async with asyncio.timeout(2):
                while not all(
                    marker in _visible_screen_text(app)
                    for marker in ("Heading", "first item", "quoted text", "value=1")
                ):
                    await pilot.pause()

            visible_text = _visible_screen_text(app)
            assert "Heading" in visible_text
            assert "first item" in visible_text
            assert "quoted text" in visible_text
            assert "value=1" in visible_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_high_frequency_deltas_are_preserved_until_the_terminal_marker() -> None:
    streamed_content = "draft-" * 300
    conversation = ScriptedRunSource(
        deltas=tuple(streamed_content),
        completed_content="# Complete\n\nExact final content.",
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await asyncio.wait_for(pilot.press(*list("burst"), "enter"), timeout=5)
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assistant = app.query_one("#conversation-display > .assistant-row").query_one(Markdown)
        assert assistant.source == streamed_content
        assert "draft-" in visible_text
        assert "Complete" not in visible_text
        assert "Exact final content." not in visible_text


@pytest.mark.asyncio
async def test_long_markdown_code_lines_remain_unwrapped_in_a_narrow_terminal() -> None:
    code_line = "very_long_variable_name = " + "0123456789" * 8 + "TAIL"
    content = f"```python\n{code_line}\n```"
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(48, 20)) as pilot:
        await pilot.press(*list("code"), "enter")
        await asyncio.sleep(0.1)

        nodes = [
            (text, y)
            for text, _, y in _screenshot_text_nodes(app)
            if text and all(character in code_line for character in text)
        ]
        assert "".join(text for text, _ in nodes).startswith("very_long_variable_name")
        assert any("0" in text for text, _ in nodes)
        assert len({y for _, y in nodes}) == 1

        assert "TAIL" not in _visible_screen_text(app)
        await pilot._post_mouse_events([MouseScrollRight], offset=(10, 5), times=30)
        assert "TAIL" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_incomplete_streamed_markdown_remains_visible_before_completion() -> None:
    content = "[documentation](https://example.com/docs)\n\n```python\nvalue = 1\n```"
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        deltas=("[documentation](", "https://example.com/docs)\n\n```python\n", "value = 1\n```"),
        completed_content=content,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("partial"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "[documentation](" not in _visible_screen_text(app):
                    await pilot.pause()

            partial_text = _visible_screen_text(app)
            assert "[documentation](" in partial_text
            assert "partial" in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

        await asyncio.sleep(0.05)
        assert "documentation (https://example.com/docs)" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_streamed_markdown_reflows_cjk_content_after_resize() -> None:
    content = "界" * 20
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cjk"), "enter")
        await asyncio.sleep(0.1)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "cancelled_content", "expected_partial", "expected_status"),
    [
        ("cancelled", "Cancelled exact content.", "draft content", "Turn cancelled."),
        ("failed", "", "draft content", "Model unavailable."),
    ],
)
async def test_terminal_outcomes_settle_markdown_and_allow_a_subsequent_turn(
    outcome: Literal["cancelled", "failed"],
    cancelled_content: str,
    expected_partial: str,
    expected_status: str | None,
) -> None:
    conversation = ScriptedRunSource(
        deltas=("draft ", "content"),
        completed_content="Recovered response.",
        outcomes=(outcome, "completed"),
        cancelled_content=cancelled_content,
        failure_message="Model unavailable.",
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        visible_text = _visible_screen_text(app)
        assert expected_partial in visible_text
        if expected_status is not None:
            assert expected_status in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert conversation.submissions == ["first", "again"]
        assert "draft content" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_cancelled_terminal_marker_renders_the_cancelled_status() -> None:
    conversation = ScriptedRunSource(
        deltas=(),
        outcomes=("cancelled",),
        cancelled_content="Retained partial response.",
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "Retained partial response." not in visible_text
        assert "Turn cancelled." in visible_text


@pytest.mark.asyncio
async def test_cancelled_terminal_marker_keeps_the_streamed_candidate_in_activity() -> None:
    conversation = ScriptedRunSource(
        deltas=("streamed candidate",),
        outcomes=("cancelled",),
        cancelled_content="authoritative partial",
    )
    app = _terminal_app(cast(Any, _terminal_backend(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        content = group.query_one(".agent-run-activity-content")
        heading = group.query_one(".agent-run-activity-heading", Static)
        assert str(heading.content).startswith("\u25bc Cancelled | ")
        assert heading.has_class("-cancelled")
        assert content.display
        assert content.query_one(Markdown).source == "streamed candidate"
        assert not app.query("#conversation-display > .assistant-row")
        visible_text = _visible_screen_text(app)
        assert "streamed candidate" in visible_text
        assert "authoritative partial" not in visible_text
        assert visible_text.index("streamed candidate") < visible_text.index("Turn cancelled.")


@pytest.mark.asyncio
async def test_markdown_failure_still_closes_the_conversation_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FailingRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    markdown_stream = FailingMarkdownStream()
    monkeypatch.setattr(
        Markdown,
        "get_stream",
        classmethod(lambda cls, markdown: markdown_stream),
    )

    with pytest.raises(RuntimeError, match="markdown write failed") as raised:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(*list("fail"), "enter")

    assert conversation.closed.is_set()
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "markdown stop failed"


@pytest.mark.asyncio
async def test_terminal_cleanup_failure_does_not_mask_an_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FailingRunSource()
    runtime = FailingCloseBackend(conversation)
    app = _terminal_app(cast(Any, runtime))
    markdown_stream = FailingMarkdownStream()
    monkeypatch.setattr(
        Markdown,
        "get_stream",
        classmethod(lambda cls, markdown: markdown_stream),
    )

    with pytest.raises(RuntimeError, match="markdown write failed") as raised:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(*list("fail"), "enter")

    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    cleanup_messages = {str(error) for error in raised.value.__cause__.exceptions}
    assert cleanup_messages == {"markdown stop failed", "runtime cleanup failed"}
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_conversation_uses_the_direct_terminal_loop_lifecycle(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _FixedCatalogProvider((_response(content="Prepared runtime answer."),))
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("h", "i", "enter")
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "hi" in visible_text
        assert "Prepared runtime answer." in visible_text

    assert provider.closed


@pytest.mark.asyncio
async def test_terminal_renders_reasoning_from_each_tool_loop_model_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    class MultiReasoningProvider(_FixedCatalogProvider):
        async def stream(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[dict[str, Any]],
            model: str,
            max_output: int,
            temperature: float,
            reasoning_effort: ReasoningEffort | None,
            timeout: int,
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            if not (
                messages
                and messages[0]
                == {"role": "system", "content": render_template("session-title-prompt.md")}
            ):
                yield ReasoningDelta(delta=f"Reasoning segment {len(self.stream_requests) + 1}.")
            async for event in super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            ):
                yield event

    (workspace / "note.txt").write_text("note", encoding="utf-8")
    provider = MultiReasoningProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments='{"path":"note.txt"}',
                ),
            ),
            _response(content="Final answer."),
        )
    )
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        await pilot.click(".agent-run-activity-heading")
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert "Reasoning segment 1." in visible_text
        assert "Reasoning segment 2." in visible_text
        assert "Final answer." in visible_text


@pytest.mark.asyncio
async def test_direct_terminal_loop_exec_confirmation_preserves_the_exact_long_command(
    agent_home: Path,
    workspace: Path,
) -> None:
    command = f'printf "{"x" * 300}" && rm -rf "{workspace.resolve()}"'
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_long_exec",
                    name="exec",
                    arguments=json.dumps({"command": command, "cwd": ".", "timeout": 45}),
                ),
            ),
            _response(content="The command was declined."),
        )
    )
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run it"), "enter"))
        await _wait_for_confirmation(app, pilot, timeout=15)

        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert f"Command: {command}" in details
        assert f"CWD: {workspace.resolve()}" in details
        assert "Timeout: 45" in details

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=2)
        await _wait_for_turn(app)
        async with asyncio.timeout(2):
            while "The command was declined." not in _visible_screen_text(app):
                await pilot.pause()
        assert "The command was declined." in _visible_screen_text(app)

    assert provider.closed


@pytest.mark.asyncio
async def test_direct_terminal_loop_close_cancels_the_pending_confirmation_future(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_close",
                    name="exec",
                    arguments=json.dumps(
                        {"command": f'rm -rf "{workspace.resolve()}"', "cwd": "."}
                    ),
                ),
            ),
        )
    )
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run it"), "enter"))
        await _wait_for_confirmation(app, pilot, timeout=15)

    await asyncio.gather(submission, return_exceptions=True)
    assert provider.closed


@pytest.mark.asyncio
async def test_direct_terminal_loop_cancellation_preserves_partial_and_allows_next_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = CancellableProvider()
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    app = _terminal_app(runtime, bus=bus)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(provider.first_delta_emitted.wait(), timeout=1)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "partial runtime response" in visible_text
        assert "Turn cancelled." in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert "Recovered runtime response." in _visible_screen_text(app)

    assert provider.closed


@pytest.mark.asyncio
async def test_narrow_terminal_uses_full_message_width_for_readable_content() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    content = "0123456789ABCDEFGHIJ"

    async with app.run_test(size=(30, 16)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)

        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24), (120, 32)))
async def test_messages_use_95_percent_of_row_width(size: tuple[int, int]) -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    content = "X" * 100

    async with app.run_test(size=size) as pilot:
        await pilot.press(*list(content), "enter")
        await _wait_for_refresh_condition(
            app,
            lambda: (
                len(app.query(".message")) == 2
                and all(message.outer_size.width > 0 for message in app.query(".message"))
                and "".join(_content_text_nodes(app, content)) == content
            ),
        )

        lines = _content_text_nodes(app, content)
        assert "".join(lines) == content
        for message in app.query(".message"):
            row = cast(Widget, message.parent)
            assert row is not None
            assert abs(message.outer_size.width - row.content_region.width * 0.95) <= 1


@pytest.mark.asyncio
async def test_messages_are_side_aligned_with_role_accents_on_wide_terminals() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("user"), "enter")
        await asyncio.sleep(0.05)

        nodes = _screenshot_text_nodes(app)
        user_x = next(x for text, x, _ in nodes if text == "user")
        assistant_x = next(x for text, x, _ in nodes if text == "First answer.")
        accents = [(x, y) for text, x, y in nodes if text == "│"]
        screenshot_width = _screenshot_width(app)

        assert user_x > screenshot_width * 0.65
        assert assistant_x < screenshot_width * 0.25
        assert min(x for x, _ in accents) < screenshot_width * 0.25
        assert max(x for x, _ in accents) > screenshot_width * 0.75


@pytest.mark.asyncio
async def test_cjk_double_width_content_reflows_when_terminal_is_resized() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    content = "界" * 20

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
async def test_role_accents_remain_visible_with_limited_ansi_colors() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))
    app.ansi_color = True

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await asyncio.sleep(0.05)

        visible_text = _visible_screen_text(app)
        assert app.native_ansi_color
        assert "hello│" in visible_text
        assert "│First answer." in visible_text


@pytest.mark.asyncio
async def test_role_accents_remain_visible_without_terminal_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await _wait_for_refresh_condition(
            app,
            lambda: "│First answer." in _visible_screen_text(app),
        )

        visible_text = _visible_screen_text(app)
        assert "nocolor" in app.screen.pseudo_classes
        assert "hello│" in visible_text
        assert "│First answer." in visible_text


@pytest.mark.asyncio
async def test_conversation_navigation_keys_keep_input_focus() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 30 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("navigate"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        assert display.max_scroll_y > 0
        assert display.is_vertical_scroll_end

        await pilot.press("ctrl+home")
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("pagedown")
        assert 0 < display.scroll_y < display.max_scroll_y
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("pageup")
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("ctrl+end")
        assert display.is_vertical_scroll_end
        assert isinstance(app.screen.focused, TextArea)


@pytest.mark.asyncio
async def test_mouse_scroll_keeps_input_focus_and_scrollbar_is_overflow_only() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 30 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("scroll"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        assert display.max_scroll_y > 0
        assert display.vertical_scrollbar.display

        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=3)
        assert display.scroll_y < display.max_scroll_y
        assert isinstance(app.screen.focused, TextArea)

        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=100)
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot._post_mouse_events([MouseScrollDown], offset=(10, 5), times=100)
        assert display.is_vertical_scroll_end
        assert isinstance(app.screen.focused, TextArea)

    empty_runtime = _terminal_backend(ScriptedRunSource())
    empty_app = _terminal_app(cast(Any, empty_runtime))
    async with empty_app.run_test(size=(60, 20)):
        empty_display = empty_app.query_one("#conversation-display")
        assert not empty_display.vertical_scrollbar.display


@pytest.mark.asyncio
async def test_conversation_scrollbar_supports_pointer_dragging() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("drag"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        scrollbar = display.vertical_scrollbar
        async with asyncio.timeout(2):
            while scrollbar.region.height < 6 or scrollbar.region.x <= 0:
                await pilot.pause()
        start = (scrollbar.region.x, scrollbar.region.bottom - 2)
        end = (scrollbar.region.x, scrollbar.region.bottom - 5)

        await pilot._post_mouse_events([MouseDown], offset=start, button=1)
        await pilot._post_mouse_events([MouseMove], offset=end, button=1)
        await pilot._post_mouse_events([MouseUp], offset=end, button=1)

        assert display.scroll_y > 0
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_dragging_scrollbar_to_bottom_resumes_follow_mode() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")

        scrollbar = display.vertical_scrollbar
        start = (scrollbar.region.x, scrollbar.region.y + 1)
        end = (scrollbar.region.x, scrollbar.region.bottom - 2)
        await pilot._post_mouse_events([MouseDown], offset=start, button=1)
        await pilot._post_mouse_events([MouseMove], offset=end, button=1)
        await pilot._post_mouse_events([MouseUp], offset=end, button=1)
        await asyncio.sleep(0.2)
        assert display.is_vertical_scroll_end

        await pilot.press(*list("next"), "enter")
        await _wait_for_turn(app)
        await pilot.pause()

        assert display.is_vertical_scroll_end
        assert not app.query_one("#new-content").display


@pytest.mark.asyncio
async def test_historical_streaming_pauses_follow_and_exposes_new_content() -> None:
    history = " ".join(f"history {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(
        deltas_by_submission=((history,), ("New ", "content.")),
        completed_contents=(history, "New content."),
    )
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)

        conversation.pause_after_next_first_delta()
        submission = asyncio.create_task(pilot.press(*list("stream"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.1)
            await pilot.pause()
            display = app.query_one("#conversation-display")
            assert display.max_scroll_y > 0
            async with asyncio.timeout(1):
                while not display.is_vertical_scroll_end:
                    await pilot.pause()
            assert display.is_vertical_scroll_end

            await pilot.press("ctrl+home")
            await asyncio.sleep(0.05)
            historical_position = display.scroll_y
            assert historical_position == 0
            assert not display.is_vertical_scroll_end

            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=2)
            await _wait_for_turn(app)
            await asyncio.sleep(0.1)

            assert display.scroll_y == historical_position
            assert not display.is_vertical_scroll_end
            assert app.query_one("#new-content").display

            await pilot.press("ctrl+end")
            assert display.is_vertical_scroll_end
            assert not app.query_one("#new-content").display
        finally:
            conversation.continue_turn()
            if not submission.done():
                await asyncio.wait_for(submission, timeout=2)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_historical_resize_preserves_a_visible_message_anchor() -> None:
    content = "\n".join(f"anchor {index:02d} " + "z" * 35 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")
        await pilot.press("pagedown")
        await pilot.pause()
        before_resize = _visible_screen_text(app)
        anchor = next(
            f"anchor {index:02d}" for index in range(50) if f"anchor {index:02d}" in before_resize
        )

        await pilot.resize_terminal(40, 24)
        async with asyncio.timeout(2):
            while anchor not in _visible_screen_text(app):
                await pilot.pause()

        assert anchor in _visible_screen_text(app)
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_bottom_follow_is_preserved_when_resize_reflows_content() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 60 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()
        assert display.is_vertical_scroll_end

        await pilot.resize_terminal(40, 24)
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()

        assert display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_user_scroll_takes_over_from_rapid_resize_callbacks() -> None:
    content = "\n".join(f"resize {index:03d} " + "x" * 70 for index in range(90))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.05)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(3):
            while display.max_scroll_y == 0:
                await pilot.pause()

        for width, height in ((65, 20), (42, 18), (76, 22), (50, 16), (80, 24)):
            await pilot.press("ctrl+home")
            async with asyncio.timeout(3):
                while display.is_vertical_scroll_end:
                    await pilot.pause()
            assert not display.is_vertical_scroll_end
            resize = asyncio.create_task(pilot.resize_terminal(width, height))
            await asyncio.sleep(0)
            await pilot.press("ctrl+end")
            await resize
            async with asyncio.timeout(3):
                while not display.is_vertical_scroll_end:
                    await pilot.pause()
            assert display.is_vertical_scroll_end
            assert display.following


@pytest.mark.asyncio
async def test_historical_resize_preserves_anchor_within_one_long_message() -> None:
    content = "\n".join(f"anchor {index:03d} " + "z" * 55 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home", "pagedown", "pagedown")
        await pilot.pause()
        before_resize = _visible_screen_text(app)
        visible_anchors = {
            f"anchor {index:03d}" for index in range(80) if f"anchor {index:03d}" in before_resize
        }
        assert visible_anchors

        await pilot.resize_terminal(25, 24)
        await pilot.pause()
        await asyncio.sleep(0.2)
        after_resize = _visible_screen_text(app)

        assert any(anchor.split()[-1] in after_resize for anchor in visible_anchors)
        assert not cast(Any, display).following


@pytest.mark.asyncio
async def test_shutdown_settles_stream_worker_before_runtime_close() -> None:
    conversation = BlockingRunSource()
    runtime = CloseOrderingBackend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)

    assert conversation.closed.is_set()
    assert runtime.close_saw_stream_closed
    assert not list(app.query(".turn-status"))


@pytest.mark.asyncio
async def test_failed_stream_terminal_still_closes_runtime_once() -> None:
    conversation = FailingCancellationCleanupRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)

    assert runtime.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "undersized"),
    [
        ((19, 10), True),
        ((20, 9), True),
        ((20, 10), False),
    ],
)
async def test_minimum_terminal_size_has_an_exact_20_by_10_boundary(
    size: tuple[int, int],
    undersized: bool,
) -> None:
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))

    async with app.run_test(size=size):
        visible_text = _visible_screen_text(app)
        assert ("Resize to" in visible_text) is undersized
        assert app.query_one("#conversation-display").display is not undersized
        assert app.query_one("#conversation-input-region").display is not undersized


@pytest.mark.asyncio
async def test_undersized_terminal_replaces_presentation_and_recovers_input() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(19, 9)) as pilot:
        size_state = app.query_one("#size-insufficient", Static)
        assert size_state.display
        assert not app.query_one("#conversation-display").display
        assert not app.query_one("#conversation-input-region").display

        await pilot.press(*list("ignored"), "enter")
        await pilot.pause()
        assert conversation.submissions == []

        await pilot.resize_terminal(80, 24)
        await pilot.pause()

        assert not size_state.display
        assert app.query_one("#conversation-display").display
        assert app.query_one("#conversation-input-region").display

        await pilot.press(*list("ready"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["ready"]


@pytest.mark.asyncio
async def test_undersized_recovery_preserves_completion_draft_history_and_focus() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("remember"), "enter")
        await _wait_for_turn(app)
        await pilot.press("/")
        assert app.command_completion_visible

        await pilot.resize_terminal(19, 9)
        await pilot.press("down", "enter", *list("ignored"))
        await pilot.pause()
        assert input_area.text == "/"

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert app.command_completion_visible
        assert "/config" in _visible_screen_text(app)
        assert input_area.text == "/"
        assert app.screen.focused is input_area
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        await pilot.press("up")
        assert input_area.text == "remember"


@pytest.mark.asyncio
async def test_undersized_recovery_preserves_active_turn_state() -> None:
    conversation = BlockingRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)
        input_area = app.query_one("#conversation-input", TextArea)
        assert not input_area.read_only

        await pilot.resize_terminal(19, 9)
        await pilot.press(*list("ignored"), "enter", "ctrl+c")
        await pilot.pause()

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert not input_area.read_only
        assert "Working" in _visible_screen_text(app)
        assert "partial" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_undersized_terminal_blocks_an_open_confirmation_until_recovery() -> None:
    conversation = ConfirmationRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await pilot.pause()

        await pilot.resize_terminal(19, 9)
        await pilot.pause()

        undersized_text = _visible_screen_text(app)
        assert "Terminal window" in undersized_text
        assert "Resize to" in undersized_text
        assert "Tool Confirmation" not in undersized_text
        await pilot.press("enter")
        await pilot.pause()
        assert conversation.responses == []

        await pilot.resize_terminal(80, 24)
        await _wait_for_confirmation(app, pilot)
        await pilot.pause(0.05)

        assert "Tool Confirmation" in _visible_screen_text(app)
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]


@pytest.mark.asyncio
async def test_resize_back_from_undersized_terminal_restores_historical_anchor() -> None:
    content = "\n".join(f"anchor {index:02d} " + "z" * 35 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("anchor"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")
        await pilot.pause()
        assert not display.is_vertical_scroll_end
        before_resize = _visible_screen_text(app)
        anchor = next(
            f"anchor {index:02d}" for index in range(50) if f"anchor {index:02d}" in before_resize
        )

        await pilot.resize_terminal(19, 9)
        await pilot.pause()
        assert app.query_one("#size-insufficient", Static).display

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert not app.query_one("#size-insufficient", Static).display
        assert anchor in _visible_screen_text(app)
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_resize_back_from_undersized_terminal_restores_bottom_follow() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 50 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("bottom"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(2):
            while not display.is_vertical_scroll_end:
                await pilot.pause()

        await pilot.resize_terminal(19, 9)
        await pilot.pause()
        assert app.query_one("#size-insufficient", Static).display

        await pilot.resize_terminal(80, 24)
        async with asyncio.timeout(2):
            while (
                app.query_one("#size-insufficient", Static).display
                or not display.is_vertical_scroll_end
            ):
                await pilot.pause()


@pytest.mark.asyncio
async def test_terminal_start_failure_closes_after_external_driver_start() -> None:
    terminal_state = {"restored": False}

    class RecordingDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            terminal_state["restored"] = False
            super().start_application_mode()

        def stop_application_mode(self) -> None:
            super().stop_application_mode()
            terminal_state["restored"] = True

    class RecordingBackend(FailingStartBackend):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state["restored"]
            await super().close()

    runtime = RecordingBackend(ScriptedRunSource())
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = RecordingDriver

    with pytest.raises(RuntimeError, match="runtime startup failed"):
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert not runtime.close_saw_terminal_restored
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_cleanup_failure_still_restores_terminal_first() -> None:
    terminal_state = {"restored": False}

    class RecordingDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            terminal_state["restored"] = False
            super().start_application_mode()

        def stop_application_mode(self) -> None:
            super().stop_application_mode()
            terminal_state["restored"] = True

    class RecordingBackend(FailingCloseBackend):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state["restored"]
            await super().close()

    runtime = RecordingBackend(ScriptedRunSource())
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = RecordingDriver

    with pytest.raises(RuntimeError, match="runtime cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    assert runtime.close_saw_terminal_restored


@pytest.mark.asyncio
async def test_terminal_start_and_cleanup_failure_preserves_the_start_error() -> None:
    class FailingStartAndCloseBackend(FakeTerminalBackend):
        async def start(self) -> None:
            self.start_calls += 1
            raise RuntimeError("runtime startup failed")

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("runtime cleanup failed")

    runtime = FailingStartAndCloseBackend(ScriptedRunSource())
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = KeyboardLifecycleDriver

    with pytest.raises(RuntimeError, match="runtime startup failed") as raised:
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "runtime cleanup failed"
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_start_failure_attempts_application_mode_restore() -> None:
    class FailingStartDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            self.write("application:start")
            self.write("\x1b[>1u")
            self.flush()
            raise RuntimeError("terminal startup failed")

    KeyboardLifecycleDriver.operations = []
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))
    app.driver_class = FailingStartDriver

    with pytest.raises(RuntimeError, match="terminal startup failed"):
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert ("write", "application:stop") in KeyboardLifecycleDriver.operations


@pytest.mark.asyncio
async def test_terminal_stop_failure_still_restores_enhanced_keyboard_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")

    class FailingStopDriver(KeyboardLifecycleDriver):
        def stop_application_mode(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    KeyboardLifecycleDriver.operations = []
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))
    app.driver_class = FailingStopDriver

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    keyboard_writes = [
        value
        for operation, value in KeyboardLifecycleDriver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]


@pytest.mark.asyncio
async def test_terminal_stop_failure_does_not_mask_an_application_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingStopDriver(KeyboardLifecycleDriver):
        def stop_application_mode(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    def failing_compose(_app: TerminalConversationApp) -> object:
        raise RuntimeError("application body failed")

    monkeypatch.setattr(TerminalConversationApp, "compose", failing_compose)

    runtime = _terminal_backend(ScriptedRunSource())
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = FailingStopDriver

    with pytest.raises(RuntimeError, match="application body failed") as raised:
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "terminal cleanup failed"


@pytest.mark.asyncio
async def test_partial_terminal_stop_failure_restores_all_modes_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    terminal_state = {
        "application_mode": False,
        "alternate_screen": False,
        "mouse_reporting": False,
        "cursor_hidden": False,
        "focus_reporting": False,
        "bracketed_paste": False,
        "kitty_depth": 0,
    }

    class PartialStopDriver(Driver):
        def __init__(
            self,
            app: App[object],
            *,
            debug: bool = False,
            mouse: bool = True,
            size: tuple[int, int] | None = None,
        ) -> None:
            super().__init__(app, debug=debug, mouse=mouse, size=size)
            self._restore_console: Callable[[], None] | None = None

        def write(self, data: str) -> None:
            transitions = {
                "\x1b[?1049h": ("alternate_screen", True),
                "\x1b[?1049l": ("alternate_screen", False),
                "\x1b[?1000h": ("mouse_reporting", True),
                "\x1b[?1000l": ("mouse_reporting", False),
                "\x1b[?25l": ("cursor_hidden", True),
                "\x1b[?25h": ("cursor_hidden", False),
                "\x1b[?1004h": ("focus_reporting", True),
                "\x1b[?1004l": ("focus_reporting", False),
                "\x1b[?2004h": ("bracketed_paste", True),
                "\x1b[?2004l": ("bracketed_paste", False),
            }
            for sequence, (name, enabled) in transitions.items():
                if sequence in data:
                    terminal_state[name] = enabled
            if "\x1b[>1u" in data:
                terminal_state["kitty_depth"] += 1
            if "\x1b[<u" in data:
                terminal_state["kitty_depth"] -= 1

        def flush(self) -> None:
            pass

        def start_application_mode(self) -> None:
            terminal_state["application_mode"] = True

            def restore_console() -> None:
                terminal_state["application_mode"] = False

            self._restore_console = restore_console
            for sequence in (
                "\x1b[?1049h",
                "\x1b[?1000h",
                "\x1b[?25l",
                "\x1b[?1004h",
                "\x1b[>1u",
                "\x1b[?2004h",
            ):
                self.write(sequence)

        def disable_input(self) -> None:
            self.write("\x1b[?1000l")

        def stop_application_mode(self) -> None:
            self.write("\x1b[?2004l")
            raise RuntimeError("terminal cleanup failed")

        def close(self) -> None:
            if self._restore_console is not None:
                self._restore_console()

    class RecordingBackend(FakeTerminalBackend):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state == {
                "application_mode": False,
                "alternate_screen": False,
                "mouse_reporting": False,
                "cursor_hidden": False,
                "focus_reporting": False,
                "bracketed_paste": False,
                "kitty_depth": 0,
            }
            await super().close()

    runtime = RecordingBackend(ScriptedRunSource())
    app = _terminal_app(cast(Any, runtime))
    app.driver_class = PartialStopDriver

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    assert runtime.close_saw_terminal_restored


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24)))
async def test_config_output_keeps_original_text_and_scrolls(size: tuple[int, int]) -> None:
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))
    output = "Configuration\n\n" + "\n".join(
        f"    key_{index:02d} = 状态 {'x' * 80}" for index in range(30)
    )

    async with app.run_test(size=size) as pilot:
        await app._mount_management_rows("/config", output)
        await pilot.pause()
        heading = app.query_one(".management-heading", Static)
        rendered = app.query_one(".management-output", Static)
        display = app.query_one("#conversation-display")
        assert str(rendered.content) == output
        assert heading.region.y < rendered.region.y
        assert display.max_scroll_y > 0
        await pilot.press("ctrl+home")
        assert "Configuration" in _visible_screen_text(app)
        await pilot.press("ctrl+end")
        assert "key_29" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_status_view_and_bar_show_current_values_and_clear_failed_reads() -> None:
    status_data = RuntimeStatus(
        version="0.1.0",
        chat_model="test/chat-model",
        chat_reasoning_effort="mid",
        uptime_seconds=10,
        context_window=100,
        max_output=10,
        available_context=90,
        compact_ratio=0.9,
        compact_context_window=81,
        projected_next_request_tokens=20,
        projection_source="estimated",
        input_budget_used_percent=22.2,
        session_message_count=2,
        last_compacted=0,
        cumulative_usage={"input_tokens": 12},
        schedule={"status": "available"},
    )

    class Management:
        current = status_data
        failing = False

        async def status(self) -> RuntimeStatus:
            if self.failing:
                raise ManagementError(ErrorInfo("route_unavailable", "Status unavailable."))
            return self.current

        async def permission_level(self) -> str:
            return self.current.current_permission_level

        async def update_permission_level(self, level: str) -> str:
            self.current = replace(self.current, current_permission_level=cast(Any, level))
            return level

    management = Management()
    app = _terminal_app(
        cast(Any, _terminal_backend(ScriptedRunSource())),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        bar = app.query_one("#status-bar", Static)
        async with asyncio.timeout(1):
            while "test/chat-model" not in str(bar.content):
                await pilot.pause()
        assert "workspace-write" in str(bar.content)

        await pilot.press(*list("/status"), "enter")
        await pilot.pause()
        output = "\n".join(
            str(cast(Static, row).content) for row in app.query(".management-output")
        )
        assert "Context budget" in output
        assert "input_budget_used_percent: 22.2%" in output
        assert all(key in output for key in status_data.to_dict())

        await pilot.press(*list("/permission"), "enter", "left", "enter")
        async with asyncio.timeout(1):
            while "read-only" not in str(bar.content):
                await pilot.pause()

        management.failing = True
        app._schedule_status_refresh()
        async with asyncio.timeout(1):
            while "Model: -" not in str(bar.content):
                await pilot.pause()
        assert "test/chat-model" not in str(bar.content)


@pytest.mark.asyncio
async def test_management_completion_supports_keyboard_filtering_and_escape() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)

        await pilot.press("/")

        completion = app.query_one("#command-completion", OptionList)
        assert [option.id for option in completion.options] == [
            command.token for command in MANAGEMENT_COMMANDS
        ]
        assert all("Management" in str(option.prompt) for option in completion.options)
        assert "/config" in _visible_screen_text(app)
        assert any(text == "/" for text, _x, _y in _screenshot_text_nodes(app))
        assert app.screen.focused is input_area

        await pilot.press("down", "enter")

        assert input_area.text == "/status"
        assert app.screen.focused is input_area
        assert runtime.inbound_history == []
        assert conversation.submissions == []

        await pilot.press("ctrl+c", "/", "down", "up", "enter")

        assert input_area.text == "/config"
        assert runtime.inbound_history == []
        assert conversation.submissions == []

        await pilot.press("ctrl+c", "/", "escape")

        assert input_area.text == "/"
        assert not completion.display
        assert app.screen.focused is input_area

        await pilot.press("m")
        await pilot.pause()

        assert [option.id for option in completion.options] == ["/memory"]
        assert "/memory" in _visible_screen_text(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24), (120, 32), (30, 12), (20, 10)))
async def test_management_completion_keeps_the_composer_visible(
    size: tuple[int, int],
) -> None:
    app = _terminal_app(cast(Any, _terminal_backend(ScriptedRunSource())))

    async with app.run_test(size=size) as pilot:
        await pilot.press("/")

        completion = app.query_one("#command-completion", OptionList)
        input_area = app.query_one("#conversation-input", TextArea)
        async with asyncio.timeout(1):
            while (
                completion.option_count != len(MANAGEMENT_COMMANDS)
                or completion.region.bottom > input_area.region.y
                or completion.region.overlaps(input_area.region)
            ):
                await pilot.pause()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()
        visible_nodes = _screenshot_text_nodes(app)
        assert completion.option_count == len(MANAGEMENT_COMMANDS)
        assert completion.virtual_size.height == completion.option_count
        assert all("\n" not in str(option.prompt) for option in completion.options)
        assert completion.region.bottom <= input_area.region.y
        assert not completion.region.overlaps(input_area.region)
        if size[0] >= 80:
            assert all(
                command.token in _visible_screen_text(app) for command in MANAGEMENT_COMMANDS[:6]
            )
            assert completion.region.height <= 7
        else:
            assert all(
                completion.render_line(index).text.endswith("\u2026")
                for index in range(completion.option_count)
            )
        assert any(text == "/" for text, _x, _y in visible_nodes)
        assert input_area.display
        assert app.query_one("#conversation-display").region.height >= 3
        assert app.screen.focused is input_area


@pytest.mark.asyncio
async def test_management_completion_mouse_selection_updates_the_composer() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")

        await pilot.click("#command-completion", offset=(2, 2))

        assert app.query_one("#conversation-input", TextArea).text == "/status"
        assert isinstance(app.screen.focused, TextArea)
        assert not app.query_one("#command-completion", OptionList).display


@pytest.mark.asyncio
async def test_skill_completion_merges_after_management_commands_with_safe_labels() -> None:
    conversation = ScriptedRunSource()
    runtime = cast(Any, _terminal_backend(conversation))
    skills = (
        SkillMetadata(
            name="alpha",
            description="First line\n\t[bold] stays literal",
            path=Path("C:/agent-home/skills/alpha/SKILL.md"),
        ),
        SkillMetadata(
            name="bravo",
            description="Second\u2003line",
            path=Path("C:/agent-home/skills/bravo/SKILL.md"),
        ),
    )
    app = _terminal_app(runtime, skill_metadata=skills)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        await pilot.pause()

        completion = app.query_one("#command-completion", OptionList)
        assert [str(option.prompt) for option in completion.options] == [
            *(
                f"{command.token}  Management  {command.description}"
                for command in MANAGEMENT_COMMANDS
            ),
            "/alpha  Skill  First line [bold] stays literal",
            "/bravo  Skill  Second line",
        ]
        assert app.query_one("#conversation-input", TextArea).text == "/"
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (30, 12), (20, 10)))
async def test_skill_completion_labels_are_single_line_at_narrow_sizes_and_stop_on_whitespace(
    size: tuple[int, int],
) -> None:
    long_name = "a" * 64
    description_chunk = "segment\n\t[bold] \\path\u2003"
    long_description = (description_chunk * ((1024 // len(description_chunk)) + 1))[:1024]
    assert len(long_description) == 1024
    runtime = cast(Any, _terminal_backend(ScriptedRunSource()))
    app = _terminal_app(
        runtime,
        skill_metadata=(
            SkillMetadata(
                name=long_name,
                description=long_description,
                path=Path("C:/agent-home/skills/long/SKILL.md"),
            ),
        ),
    )

    async with app.run_test(size=size) as pilot:
        await pilot.press("/")
        await pilot.pause()

        completion = app.query_one("#command-completion", OptionList)
        input_area = app.query_one("#conversation-input", TextArea)
        skill_label = str(completion.options[-1].prompt)
        assert skill_label.startswith(f"/{long_name}  Skill  ")
        assert "\n" not in skill_label
        assert "\t" not in skill_label
        assert "[bold]" in skill_label
        assert "\\path" in skill_label
        assert completion.virtual_size.height == completion.option_count
        assert completion.region.bottom <= input_area.region.y
        assert not completion.region.overlaps(input_area.region)
        assert input_area.text == "/"
        assert input_area.display
        assert app.screen.focused is input_area

        await pilot.press(*(("down",) * len(MANAGEMENT_COMMANDS)))
        await pilot.pause()
        assert completion.render_line(
            completion.scrollable_content_region.height - 1
        ).text.endswith("\u2026")

        input_area.text = "/\u2003"
        await pilot.pause()
        assert not completion.display
        assert not completion.options


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ("enter", "exact-enter", "mouse"))
async def test_skill_completion_selection_inserts_only_the_skill_invocation(
    selection: Literal["enter", "exact-enter", "mouse"],
) -> None:
    conversation = ScriptedRunSource()
    fake_runtime = _terminal_backend(conversation)
    runtime = cast(Any, fake_runtime)
    app = _terminal_app(
        runtime,
        skill_metadata=(
            SkillMetadata(
                name="alpha",
                description="First skill",
                path=Path("C:/agent-home/skills/alpha/SKILL.md"),
            ),
        ),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        await pilot.pause()

        if selection == "mouse":
            await pilot.press("a")
            await pilot.pause()
            await pilot.click("#command-completion", offset=(2, 1))
        elif selection == "exact-enter":
            await pilot.press(*list("alpha"), "enter")
        else:
            await pilot.press(*(("down",) * len(MANAGEMENT_COMMANDS)), selection)

        input_area = app.query_one("#conversation-input", TextArea)
        completion = app.query_one("#command-completion", OptionList)
        assert input_area.text == "/alpha "
        assert not completion.display
        assert not completion.options
        assert app.screen.focused is input_area
        assert fake_runtime.inbound_history == []
        assert conversation.submissions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("down_count", "candidate_text"),
    ((0, "/config"), (len(MANAGEMENT_COMMANDS), "/alpha ")),
)
async def test_completion_tab_does_not_accept_or_submit_highlighted_candidate(
    down_count: int,
    candidate_text: str,
) -> None:
    conversation = ScriptedRunSource()
    fake_runtime = _terminal_backend(conversation)
    runtime = cast(Any, fake_runtime)
    app = _terminal_app(
        runtime,
        skill_metadata=(
            SkillMetadata(
                name="alpha",
                description="First skill",
                path=Path("C:/agent-home/skills/alpha/SKILL.md"),
            ),
        ),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", *(("down",) * down_count), "tab")
        await pilot.pause()

        input_area = app.query_one("#conversation-input", TextArea)
        assert input_area.text != candidate_text
        assert fake_runtime.inbound_history == []
        assert conversation.submissions == []


@pytest.mark.asyncio
async def test_management_error_row_preserves_later_command_interaction(
    agent_home: Path,
    workspace: Path,
) -> None:
    class Management:
        async def memory_view(self) -> str:
            raise ManagementError(
                ErrorInfo("persistence_error", "Long-term Memory could not be read.")
            )

        async def dream(self) -> DreamResult:
            return DreamResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=0,
                error=None,
            )

    provider = _FixedCatalogProvider(())
    runtime, bus = _direct_terminal_loop(agent_home, workspace, provider)
    dispatcher = ManagementCommandDispatcher(cast(Any, Management()))
    app = _terminal_app(runtime, bus=bus, management_dispatcher=dispatcher)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/memory"), "enter")
        await pilot.press(*list("/dream"), "enter")
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert "persistence_error: Long-term Memory could not be read." in visible_text
        assert "No pending summaries" in visible_text
        assert runtime.session.messages == []
        assert provider.stream_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (False, True))
async def test_reload_skill_submission_updates_only_terminal_skill_cache_on_success(
    failure: bool,
) -> None:
    initial_metadata = (
        SkillMetadata(
            name="initial",
            description="Initial skill",
            path=Path("C:/agent-home/skills/initial/SKILL.md"),
        ),
    )
    target_metadata = (
        SkillMetadata(
            name="target",
            description="Target skill",
            path=Path("C:/agent-home/skills/target/SKILL.md"),
        ),
    )

    class ReloadManagement:
        async def reload_skill(self) -> tuple[SkillMetadata, ...]:
            if failure:
                raise RuntimeError(
                    "SECRET-SKILL-BODY D:\\private\\skills\\planner\\SKILL.md traceback",
                )
            return target_metadata

    conversation = ScriptedRunSource()
    fake_runtime = _terminal_backend(conversation)
    app = _terminal_app(
        cast(Any, fake_runtime),
        skill_metadata=initial_metadata,
        management_dispatcher=ManagementCommandDispatcher(cast(Any, ReloadManagement())),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        await pilot.pause()
        input_area = app.query_one("#conversation-input", _ConversationInput)
        input_area.text = "/reload_skill"
        await pilot.pause()
        completion = app.query_one("#command-completion", OptionList)
        before_cache = app._completion_options
        before_options = tuple(str(option.prompt) for option in completion.options)
        before_display = completion.display
        before_dismissed_text = app._completion_dismissed_text
        before_input = input_area.text

        await pilot.press("enter")
        await pilot.pause()

        assert fake_runtime.inbound_history == []
        assert conversation.submissions == []
        if failure:
            assert app._skill_metadata == initial_metadata
            assert "skill_reload_failed: Skill reload failed." in _visible_screen_text(app)
            assert app._completion_options == before_cache
            assert tuple(str(option.prompt) for option in completion.options) == before_options
            assert completion.display is before_display
            assert app._completion_dismissed_text == before_dismissed_text
            assert input_area.text == before_input
        else:
            assert app._skill_metadata == target_metadata
            assert "Skill count: 1" in _visible_screen_text(app)
            assert not completion.display
            assert not completion.options


@pytest.mark.asyncio
async def test_completion_direction_keys_take_precedence_over_composer_and_input_history() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("previous"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("/")
        for direction in ("left", "right"):
            cursor_before = input_area.cursor_location
            await pilot.press(direction)
            assert input_area.cursor_location == cursor_before
            assert input_area.text == "/"
            assert any(text.startswith("/config") for text, _x, _y in _screenshot_text_nodes(app))

        await pilot.press("up", "down")
        assert input_area.text == "/"

        await pilot.press("escape", "ctrl+c", "up")
        assert input_area.text == "previous"


@pytest.mark.asyncio
async def test_completion_ctrl_c_closes_completion_before_idle_draft_behavior() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    app = _terminal_app(cast(Any, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("previous"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("/")
        await pilot.pause()
        assert any(text.startswith("/config") for text, _x, _y in _screenshot_text_nodes(app))

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert input_area.text == "/"
        assert not app.query_one("#command-completion", OptionList).display
        assert app.is_running
        assert app.screen.focused is input_area

        await pilot.press("ctrl+c")
        assert input_area.text == ""
        assert app.is_running

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


class _EffortManagement:
    def __init__(self, effort: str) -> None:
        self.effort = effort
        self.updated: list[str] = []

    async def reasoning_effort(self) -> str:
        return self.effort

    async def update_reasoning_effort(self, effort: str) -> str:
        self.updated.append(effort)
        self.effort = effort
        return effort


class _PermissionManagement:
    def __init__(self, level: str) -> None:
        self.level = level
        self.updated: list[str] = []

    async def permission_level(self) -> str:
        return self.level

    async def update_permission_level(self, level: str) -> str:
        self.updated.append(level)
        self.level = level
        return level


class _BlockingEffortManagement(_EffortManagement):
    def __init__(self, effort: str) -> None:
        super().__init__(effort)
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()

    async def update_reasoning_effort(self, effort: str) -> str:
        self.updated.append(effort)
        self.update_started.set()
        await self.release_update.wait()
        self.effort = effort
        return effort


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ("low", "mid", "high", "xhigh", "max"))
@pytest.mark.parametrize("cancel_key", ("escape", "ctrl+c"))
async def test_effort_selector_cancels_without_rows_or_runtime_updates(
    initial: str,
    cancel_key: str,
) -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _EffortManagement(initial)
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()

        input_area = app.query_one("#conversation-input", TextArea)
        selector = app.query_one("#reasoning-effort-selector", Static)
        selector_state = cast(Any, selector)
        assert not input_area.display
        assert selector.display
        assert selector_state.selected_effort == initial
        assert app.screen.focused is selector
        visible_text = cast(Any, selector.render()).plain
        assert f"Current: {initial} | Pending: {initial}" in visible_text
        assert all(
            re.search(rf"(?<![a-z]){level}(?![a-z])", visible_text) is not None
            for level in ("low", "mid", "high", "xhigh", "max")
        )

        await pilot.press("right", cancel_key)
        await pilot.pause()

        assert management.updated == []
        assert input_area.display
        assert not selector.display
        assert input_area.text == ""
        assert app.screen.focused is input_area
        assert len(app.query(".management-row")) == 0
        assert runtime.inbound_history == []
        assert conversation.submissions == []

        await pilot.press("up")
        assert input_area.text == "/effort"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((40, 15), (80, 24), (120, 32)))
async def test_effort_selector_mouse_commits_selected_value(size: tuple[int, int]) -> None:
    management = _EffortManagement("mid")
    app = _terminal_app(
        cast(Any, _terminal_backend(ScriptedRunSource())),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=size) as pilot:
        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()
        selector = cast(Any, app.query_one("#reasoning-effort-selector", Static))
        options_line = selector.render().plain.splitlines()[1]
        left = selector.content_region.x - selector.region.x
        centered = max(0, (selector.content_region.width - len(options_line)) // 2)
        high_x = left + centered + options_line.index("high") + 1
        assert await pilot.click("#reasoning-effort-selector", offset=(high_x, 2))
        async with asyncio.timeout(1):
            while management.updated != ["high"]:
                await pilot.pause()
        assert app.query_one("#conversation-input", TextArea).display


@pytest.mark.asyncio
async def test_effort_selector_commits_selected_value_and_restores_composer() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _EffortManagement("mid")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()
        await pilot.press("right", "right", "enter")
        await pilot.pause()

        input_area = app.query_one("#conversation-input", TextArea)
        selector = app.query_one("#reasoning-effort-selector", Static)
        assert management.updated == ["xhigh"]
        assert input_area.display
        assert not selector.display
        assert input_area.text == ""
        assert app.screen.focused is input_area
        assert "Command: /effort" in _visible_screen_text(app)
        assert _visible_screen_text(app).count("Chat reasoning effort: xhigh") == 1
        assert len(app.query(".management-row")) == 2


@pytest.mark.asyncio
async def test_effort_selector_accepts_only_one_confirm_while_update_is_pending() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _BlockingEffortManagement("mid")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()
        confirm = asyncio.create_task(pilot.press("right", "enter"))
        await management.update_started.wait()

        input_area = app.query_one("#conversation-input", TextArea)
        selector = app.query_one("#reasoning-effort-selector", Static)
        assert input_area.display
        assert not selector.display

        duplicate = asyncio.create_task(pilot.press("enter"))
        await asyncio.sleep(0)
        management.release_update.set()
        await asyncio.gather(confirm, duplicate)
        await pilot.pause()

        assert management.updated == ["high"]
        assert _visible_screen_text(app).count("Chat reasoning effort: high") == 1
        assert len(app.query(".management-row")) == 2


@pytest.mark.asyncio
async def test_effort_selector_clamps_navigation_at_both_boundaries() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _EffortManagement("mid")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()
        selector = cast(Any, app.query_one("#reasoning-effort-selector", Static))

        await pilot.press("left", "left", "left", "left", "left")
        assert selector.selected_effort == "low"
        await pilot.press("left")
        assert selector.selected_effort == "low"

        await pilot.press("right", "right", "right", "right", "right")
        assert selector.selected_effort == "max"
        await pilot.press("right")
        assert selector.selected_effort == "max"

        await pilot.press("escape")
        await pilot.pause()
        assert management.updated == []


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ("read-only", "workspace-write", "full-access"))
async def test_permission_selector_reflects_current_level_and_cancels_without_update(
    initial: str,
) -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement(initial)
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(24, 12)) as pilot:
        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()

        input_area = app.query_one("#conversation-input", TextArea)
        selector = cast(Any, app.query_one("#permission-selector", Static))
        assert not input_area.display
        assert selector.display
        assert selector.selected_permission_level == initial
        assert all(
            label in selector.render().plain
            for label in ("Read-Only", "Workspace-Write", "Full-Access")
        )
        rendered_lines = selector.render().plain.splitlines()
        assert rendered_lines[1:] == ["Read-Only", "Workspace-Write", "Full-Access"]
        assert rendered_lines[0].startswith("Current:")
        assert max(map(len, rendered_lines)) <= selector.content_region.width
        assert selector.outer_size.height <= 5
        assert app.screen.focused is selector

        await pilot.press("right" if initial == "read-only" else "left")
        assert cast(Any, selector).selected_permission_level != initial
        assert "Current:" in cast(Any, selector).render().plain.splitlines()[0]
        await pilot.press("escape")
        await pilot.pause()

        assert management.updated == []
        assert management.level == initial
        assert input_area.display
        assert not selector.display
        assert app.screen.focused is input_area


@pytest.mark.asyncio
async def test_permission_full_access_requires_cancel_focused_warning_and_explicit_confirmation() -> (
    None
):
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement("workspace-write")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()

        assert app.screen.id == "permission-warning"
        assert app.screen.focused is not None
        assert app.screen.focused.id == "permission-warning-cancel"
        warning = " ".join(_visible_screen_text(app).split())
        assert "Full-Access" in warning
        assert "ordinary permission prompts" in warning
        assert "OS sandbox" in warning
        assert "validation" in warning.casefold()
        for tool in ("File", "Exec", "Web Fetch", "MCP", "Schedule"):
            assert tool in warning
        assert "private or non-global" in warning
        for protection in (
            "Catastrophic or uncertain Exec",
            "unavailable",
            "capabilities",
            "business refusals",
            "Tool errors",
            "confirmation",
        ):
            assert protection in warning
        assert "keep their existing behavior" not in warning
        assert management.updated == []

        await pilot.press("enter")
        await pilot.pause()
        assert management.level == "workspace-write"
        assert management.updated == []
        assert app.screen.id == "_default"

        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()
        assert app.screen.id == "permission-warning"
        await pilot.press("right", "enter")
        await pilot.pause()

        assert management.level == "full-access"
        assert management.updated == ["full-access"]
        assert app.screen.id == "_default"
        assert "Foreground permission level: full-access" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_permission_full_access_warning_actions_fit_narrow_viewport() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement("workspace-write")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()

        assert app.screen.id == "permission-warning"
        assert app.screen.focused is not None
        assert app.screen.focused.id == "permission-warning-cancel"
        cancel = app.screen.query_one("#permission-warning-cancel", Button)
        confirm = app.screen.query_one("#permission-warning-confirm", Button)
        assert cancel.region.width > 0 and confirm.region.width > 0
        assert cancel.region.y >= 0 and confirm.region.y >= 0
        assert cancel.region.bottom <= 18 and confirm.region.bottom <= 18
        assert not cancel.region.overlaps(confirm.region)
        assert management.updated == []

        assert await pilot.click(
            confirm, offset=(confirm.size.width // 2, confirm.size.height // 2)
        )
        await pilot.pause()
        assert management.updated == ["full-access"]


@pytest.mark.asyncio
async def test_permission_selector_and_warning_support_mouse_without_narrow_overlap() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement("workspace-write")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:

        async def click_full_access() -> None:
            await pilot.press(*list("/permission"), "enter")
            await pilot.pause()
            selector = cast(Any, app.query_one("#permission-selector", Static))
            rendered = selector.render().plain.splitlines()[1]
            content_x = selector.content_region.x - selector.region.x
            aligned_x = max(0, (selector.content_region.width - len(rendered)) // 2)
            full_access_x = content_x + aligned_x + rendered.index("Full-Access") + 1
            assert await pilot.click("#permission-selector", offset=(full_access_x, 2))
            await pilot.pause()

        await click_full_access()
        assert app.screen.id == "permission-warning"
        assert management.updated == []
        cancel = app.screen.query_one("#permission-warning-cancel", Button)
        confirm = app.screen.query_one("#permission-warning-confirm", Button)
        assert not cancel.region.overlaps(confirm.region)
        assert await pilot.click(
            cancel,
            offset=(cancel.size.width // 2, cancel.size.height // 2),
        )
        await pilot.pause()
        assert management.updated == []

        await click_full_access()
        assert app.screen.id == "permission-warning"
        confirm = app.screen.query_one("#permission-warning-confirm", Button)
        assert await pilot.click(
            confirm,
            offset=(confirm.size.width // 2, confirm.size.height // 2),
        )
        await pilot.pause()

        assert management.level == "full-access"
        assert management.updated == ["full-access"]


@pytest.mark.asyncio
async def test_permission_warning_external_dismiss_and_unmount_leave_selection_unchanged() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement("workspace-write")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()
        assert app.screen.id == "permission-warning"

        await app.screen.dismiss(None)
        await pilot.pause()
        assert app.screen.id == "_default"
        assert management.updated == []

        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("right", "enter")
        await pilot.pause()
        assert app.screen.id == "permission-warning"

    assert management.level == "workspace-write"
    assert management.updated == []
    assert app._permission_warning_result is None


@pytest.mark.asyncio
async def test_permission_downgrade_skips_warning_and_same_level_is_stable() -> None:
    conversation = ScriptedRunSource()
    runtime = _terminal_backend(conversation)
    management = _PermissionManagement("full-access")
    app = _terminal_app(
        cast(Any, runtime),
        management_dispatcher=ManagementCommandDispatcher(cast(Any, management)),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("left", "enter")
        await pilot.pause()

        assert app.screen.id == "_default"
        assert management.level == "workspace-write"
        assert management.updated == ["workspace-write"]

        await pilot.press(*list("/permission"), "enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.screen.id == "_default"
        assert management.level == "workspace-write"
        assert management.updated == ["workspace-write"]


@pytest.mark.asyncio
async def test_effort_persistence_failure_does_not_interrupt_active_run_or_next_call(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CancellableProvider()
    config_path = agent_home / "config.toml"
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    def fail_config_replace(target: Path, content: str) -> None:
        if target == config_path:
            raise OSError("injected config replacement failure")
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_config_replace)

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        initial = app._control
        submission = asyncio.create_task(pilot.press(*list("active work"), "enter"))
        await asyncio.wait_for(provider.first_delta_emitted.wait(), timeout=2)
        assert initial.has_active_run

        await pilot.press(*list("/effort"), "enter")
        await pilot.pause()
        selector = app.query_one("#reasoning-effort-selector", Static)
        assert selector.display
        await pilot.press("right", "enter")
        await pilot.pause()

        assert initial.has_active_run
        assert _visible_screen_text(app).count("Chat reasoning effort: high") == 1

        await pilot.press(*list("/status"), "enter")
        await pilot.pause()
        assert any(
            "chat_reasoning_effort: high" in str(cast(Static, row).content)
            for row in app.query(".management-row")
        )
        assert initial.has_active_run

        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=2)
        await _wait_for_turn(app)

        await pilot.press(*list("next"), "enter")
        await _wait_for_turn(app)

        assert [request.reasoning_effort for request in provider.stream_requests] == [
            "mid",
            "high",
        ]
        assert app._control is initial

    await _run_cli_terminal_case(
        agent_home=agent_home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=provider,
    )
