"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

import asyncio
import sys
from asyncio import CancelledError, Event, create_task
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic as monotonic_now
from typing import Final, Protocol, cast
from uuid import UUID, uuid4

from rich.cells import cell_len
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.driver import Driver
from textual.events import Resize, Unmount
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Markdown, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerError

from aide.agent.confirmation import (
    ConfirmationDecision as CoordinatorConfirmationDecision,
)
from aide.agent.confirmation import (
    ConfirmationEnvelope,
    ConfirmationPresentationCoordinator,
    ConfirmationUnavailable,
)
from aide.agent.loop import (
    ConfirmationRequestView,
    ForegroundConversationProjection,
    TerminalAgentRunExecutorControl,
)
from aide.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from aide.agent.permission import ToolPermissionLevel
from aide.agent.session.restore import RestoreMode, RestorePlan, RestoreResult
from aide.agent.session.session import RestoreAnchor
from aide.management.commands import (
    RELOAD_SKILL_MANAGEMENT_COMMAND,
    RESUME_MANAGEMENT_COMMAND,
    ManagementCommandDispatcher,
    ManagementCommandResult,
)
from aide.management.service import FatalManagementError, RuntimeStatus, SessionListingEntry
from aide.provider.models import ReasoningEffort
from aide.skills.catalog import SkillMetadata
from aide.terminal.keyboard import EnhancedKeyboardAdapter
from aide.terminal.ui.activity import (
    _activity_group_heading_text,
    _ActivityGroupHeading,
    _ActivityGroupState,
    _tool_row_content,
    _ToolRowStatus,
)
from aide.terminal.ui.dialogs.confirmation import (
    ConfirmationDecision,
    _FullAccessWarningScreen,
    _ToolConfirmationScreen,
)
from aide.terminal.ui.dialogs.restore import (
    _RestoreAnchorPickerScreen,
    _RestoreConfirmationScreen,
    _RestoreFailureScreen,
    _RestoreModeScreen,
    _RestoreWaitingScreen,
)
from aide.terminal.ui.dialogs.sessions import (
    _ConversationRecoveryScreen,
    _SessionPickerScreen,
    _SessionSwitchConfirmationScreen,
)
from aide.terminal.ui.dialogs.size import (
    _MIN_TERMINAL_HEIGHT,
    _MIN_TERMINAL_WIDTH,
    _SizeInsufficientScreen,
)
from aide.terminal.ui.display import _COMPACT_MESSAGE_MAX_WIDTH, _ConversationDisplay
from aide.terminal.ui.input import (
    _CommandCompletion,
    _completion_candidates,
    _CompletionCandidate,
    _CompletionCandidateKind,
    _ConversationInput,
)
from aide.terminal.ui.rendering import (
    _classify_historical_partition,
    _markdown_parser,
    _MessageBusRunProjection,
    _persisted_assistant_status,
    _persisted_message_partitions,
    _persisted_role_and_content,
    _PersistedMessageProjection,
    _queue_excerpt,
    _status_view_text,
)
from aide.terminal.ui.selectors import _PermissionSelector, _ReasoningEffortSelector

_RELOAD_SKILL_MANAGEMENT_COMMAND_TOKEN = RELOAD_SKILL_MANAGEMENT_COMMAND.token



_RESUME_MANAGEMENT_COMMAND_TOKEN = RESUME_MANAGEMENT_COMMAND.token



_RESTORE_MANAGEMENT_COMMAND_TOKEN = "/restore"



_TERMINAL_MODE_RESETS: Final = (
    ("\x1b[?2004h", "\x1b[?2004l"),
    ("\x1b[?1000h", "\x1b[?1000l"),
    ("\x1b[?1003h", "\x1b[?1003l"),
    ("\x1b[?1015h", "\x1b[?1015l"),
    ("\x1b[?1006h", "\x1b[?1006l"),
    ("\x1b[?1004h", "\x1b[?1004l"),
    ("\x1b[?1049h", "\x1b[?1049l"),
    ("\x1b[?25l", "\x1b[?25h"),
)



class _DriverLifecycleHooks(Protocol):
    write: Callable[[str], None]
    flush: Callable[[], None]
    start_application_mode: Callable[[], None]
    stop_application_mode: Callable[[], None]



class _ConsoleRestoreHooks(Protocol):
    _restore_console: Callable[[], None] | None



@dataclass(slots=True)
class _ConsumedRun:
    turn_id: UUID
    user_text: str
    projection: _MessageBusRunProjection
    run_id: str | None = None
    started: bool = False



class TerminalConversationApp(App[None]):
    """The two-region Textual application for one foreground Message Bus."""

    class CoordinatorConfirmationRequested(Message):
        def __init__(
            self,
            envelope: ConfirmationEnvelope,
            token: object,
            respond: Callable[[object, CoordinatorConfirmationDecision], bool],
        ) -> None:
            super().__init__()
            self.envelope = envelope
            self.token = token
            self.respond = respond

    class ConfirmationRequested(Message):
        def __init__(
            self,
            request: ConfirmationRequestView,
            *,
            control: TerminalAgentRunExecutorControl,
            bus: MessageBus,
        ) -> None:
            super().__init__()
            self.request = request
            self.bound_control = control
            self.bound_bus = bus

    class InboundSnapshotChanged(Message):
        def __init__(
            self,
            bus: MessageBus,
            snapshot: tuple[InboundMessage, ...],
            *,
            promote_removed: bool,
            callback: Callable[[tuple[InboundMessage, ...]], None],
        ) -> None:
            super().__init__()
            self.bus = bus
            self.snapshot = snapshot
            self.promote_removed = promote_removed
            self.callback = callback

    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #conversation-display {
        height: 1fr;
        min-height: 3;
        width: 100%;
        padding: 1 2;
        scrollbar-size-vertical: 1;
        background: transparent;
    }

    #conversation-input {
        height: auto;
        min-height: 3;
        max-height: 8;
        width: 100%;
        border-top: solid $panel;
        padding: 0 1;
        background: transparent;
    }

    #reasoning-effort-selector {
        display: none;
        height: 3;
        min-height: 3;
        width: 100%;
        border-top: solid $panel;
        padding: 0 1;
        overflow-x: auto;
        text-wrap: nowrap;
        content-align: center middle;
        text-align: center;
        background: transparent;
    }

    #permission-selector {
        display: none;
        height: auto;
        min-height: 3;
        max-height: 5;
        width: 100%;
        border-top: solid $panel;
        padding: 0 1;
        overflow: hidden;
        text-wrap: nowrap;
        content-align: center middle;
        text-align: center;
        background: transparent;
        pointer: pointer;
    }

    .message {
        width: 95%;
        max-width: 100%;
        min-width: 0;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: transparent;
    }

    .message-compact {
        width: 95%;
    }

    .user-message {
        text-align: right;
        border-right: solid $foreground;
    }

    .assistant-message {
        border-left: solid $foreground;
    }

    .message-row {
        width: 100%;
        height: auto;
    }

    .tool-row {
        width: 100%;
        min-width: 0;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }

    .agent-run-activity-group {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        padding: 0;
        background: transparent;
    }

    .agent-run-activity-heading {
        width: 100%;
        height: auto;
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }

    .agent-run-activity-heading:focus {
        background: $panel;
    }

    .agent-run-activity-heading.-running {
        color: $primary;
    }

    .agent-run-activity-heading.-failed {
        color: $error;
    }

    .agent-run-activity-heading.-cancelled {
        color: $warning;
    }

    .agent-run-activity-content {
        width: 100%;
        height: auto;
        padding: 0;
        background: transparent;
    }

    #command-completion {
        display: none;
        width: 100%;
        height: 7;
        max-height: 7;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        background: transparent;
        border-top: solid $panel;
    }

    .management-row {
        width: 100%;
        min-width: 0;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }

    .user-row {
        align: right top;
    }

    .assistant-row {
        align: left top;
    }

    #conversation-input-region {
        height: auto;
        min-height: 3;
        width: 100%;
    }

    #size-insufficient {
        display: none;
        width: 100%;
        height: 1fr;
        padding: 1 2;
        align: center middle;
        content-align: center middle;
        text-align: center;
        background: transparent;
    }

    #new-content {
        display: none;
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }

    #pending-queue {
        display: none;
        height: auto;
        max-height: 2;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }

    #status-bar {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }

    .management-heading {
        color: $primary;
        text-style: bold;
        margin-bottom: 0;
    }

    .management-output {
        color: $text;
    }

    .turn-status {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        control: TerminalAgentRunExecutorControl,
        management_dispatcher: ManagementCommandDispatcher,
        monotonic: Callable[[], float] = monotonic_now,
        skill_metadata: tuple[SkillMetadata, ...] = (),
        management_command_tokens: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self._skill_metadata = tuple(skill_metadata)
        self._management_command_tokens = management_command_tokens
        self._service_run_projection = callable(
            getattr(management_dispatcher, "recall_queued_inputs", None)
        )
        self._submitting_input = False
        self._bus = bus
        self._control = control
        self._management_dispatcher = management_dispatcher
        self._monotonic = monotonic
        self._confirmation_coordinator: ConfirmationPresentationCoordinator | None = None
        self._conversation_display: _ConversationDisplay | None = None
        self._conversation_input: _ConversationInput | None = None
        self._size_insufficient = False
        self._driver_mode_started = False
        self._driver_mode_stopped = True
        self._viable_size = Event()
        self._viable_size.set()
        self._size_screen: _SizeInsufficientScreen | None = None
        self._outbound_worker: Worker[None] | None = None
        self._resume_worker: Worker[None] | None = None
        self._service_connection_state: str | None = None
        self._restore_worker: Worker[None] | None = None
        self._restore_result_worker: Worker[None] | None = None
        self._restore_workflow_active = False
        self._restore_anchors: tuple[RestoreAnchor, ...] = ()
        self._restore_plan: RestorePlan | None = None
        self._cancel_requested_turn: object | None = None
        self._active_run_projection: _MessageBusRunProjection | None = None
        self._active_confirmation_id: UUID | None = None
        self._active_confirmation_token: object | None = None
        self._dismissed_confirmation_tokens: set[object] = set()
        self._confirmation_result: asyncio.Future[ConfirmationDecision | None] | None = None
        self._session_switch_result: asyncio.Future[bool | None] | None = None
        self._pending_inputs: deque[str] = deque()
        self._bus_snapshot: tuple[InboundMessage, ...] = ()
        self._consumed_runs: deque[_ConsumedRun] = deque()
        self._run_ready = Event()
        self._draining_inputs = False
        self._completion_options: tuple[_CompletionCandidate, ...] = ()
        self._completion_dismissed_text: str | None = None
        self._working = False
        self._status_view: RuntimeStatus | None = None
        self._status_generation = 0
        self._permission_current_level: ToolPermissionLevel | None = None
        self._permission_warning_result: asyncio.Future[bool | None] | None = None
        self._closing = False
        self._presentation_quiesced = False
        self._bus_callback: Callable[[tuple[InboundMessage, ...]], None] | None = None
        self._bus_callback_bus: MessageBus | None = None
        self._confirmation_callback: Callable[[ConfirmationRequestView], None] | None = None
        self._confirmation_control: TerminalAgentRunExecutorControl | None = None
        self._application_error: Exception | None = None
        self._fatal_management_error: FatalManagementError | None = None

    def bind_confirmation_coordinator(
        self,
        coordinator: ConfirmationPresentationCoordinator,
    ) -> None:
        """Bind the Runtime Lifetime coordinator without changing the app constructor contract."""
        if self._confirmation_coordinator is not None and self._confirmation_coordinator is not coordinator:
            raise RuntimeError("a different confirmation coordinator is already bound")
        self._confirmation_coordinator = coordinator
        if self._conversation_display is not None:
            coordinator.bind_presenter(self)

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, CoordinatorConfirmationDecision], bool],
    ) -> None:
        """Project a coordinator item onto the mounted Textual conversation."""
        if self._closing or self._presentation_quiesced:
            raise ConfirmationUnavailable("confirmation presenter is not available")
        accepted = self.post_message(
            self.CoordinatorConfirmationRequested(envelope, token, respond)
        )
        if not accepted:
            raise ConfirmationUnavailable("confirmation presenter is not available")

    async def dismiss_confirmation(self, token: object) -> None:
        """Dismiss only the modal owned by this coordinator item."""
        self._dismissed_confirmation_tokens.add(token)
        if self._active_confirmation_token is not token:
            return
        if isinstance(self.screen, _ToolConfirmationScreen):
            self.screen.dismiss(None)

    def _handle_exception(self, error: Exception) -> None:
        if self._application_error is None:
            self._application_error = error
        super()._handle_exception(error)

    @property
    def fatal_management_error(self) -> FatalManagementError | None:
        return self._fatal_management_error

    def _build_driver(
        self,
        headless: bool,
        inline: bool,
        mouse: bool,
        size: tuple[int, int] | None,
    ) -> Driver:
        driver = super()._build_driver(headless, inline, mouse, size)
        if driver.is_headless:
            return driver

        # Textual has no public hook around its Kitty push/pop writes. Keep this
        # compatibility boundary narrow and exercise it through App.run_async tests.
        EnhancedKeyboardAdapter.install_on_driver(driver)
        self._install_driver_lifecycle(driver)
        return driver

    def _install_driver_lifecycle(self, driver: Driver) -> None:
        hooks = cast(_DriverLifecycleHooks, driver)
        original_write = hooks.write
        original_flush = hooks.flush
        original_start = hooks.start_application_mode
        original_stop = hooks.stop_application_mode
        self._driver_mode_started = False
        self._driver_mode_stopped = True
        active_mode_resets: set[str] = set()

        def write(value: str) -> None:
            original_write(value)
            transitions: list[tuple[int, str, bool]] = []
            for enable, reset in _TERMINAL_MODE_RESETS:
                enable_at = value.find(enable)
                if enable_at >= 0:
                    transitions.append((enable_at, reset, True))
                reset_at = value.find(reset)
                if reset_at >= 0:
                    transitions.append((reset_at, reset, False))
            for _, reset, enabled in sorted(transitions):
                if enabled:
                    active_mode_resets.add(reset)
                else:
                    active_mode_resets.discard(reset)

        def restore_terminal_modes() -> None:
            if active_mode_resets:
                for _, reset in _TERMINAL_MODE_RESETS:
                    if reset in active_mode_resets:
                        write(reset)
                original_flush()

            restore_console = getattr(driver, "_restore_console", None)
            if callable(restore_console):
                restore_console()
                cast(_ConsoleRestoreHooks, driver)._restore_console = None

        def stop_application_mode() -> None:
            if not self._driver_mode_started or self._driver_mode_stopped:
                return
            primary_error = sys.exception()
            try:
                original_stop()
            except BaseException as stop_error:
                try:
                    restore_terminal_modes()
                except BaseException as restore_error:
                    stop_error.__cause__ = restore_error
                else:
                    self._driver_mode_stopped = True
                if primary_error is not None:
                    if stop_error is primary_error:
                        raise
                    raise primary_error from stop_error
                raise stop_error
            try:
                restore_terminal_modes()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    raise primary_error from cleanup_error
                raise
            self._driver_mode_stopped = True

        def start_application_mode() -> None:
            self._driver_mode_started = True
            self._driver_mode_stopped = False
            try:
                original_start()
            except BaseException as primary_error:
                try:
                    stop_application_mode()
                except BaseException as cleanup_error:
                    if cleanup_error is primary_error:
                        raise
                    raise primary_error from cleanup_error
                raise

        hooks.write = write
        hooks.start_application_mode = start_application_mode
        hooks.stop_application_mode = stop_application_mode

    def compose(self) -> ComposeResult:
        yield _ConversationDisplay(id="conversation-display")
        yield Vertical(
            _CommandCompletion(
                id="command-completion",
                markup=False,
                compact=True,
            ),
            Static("New content below", id="new-content", markup=False),
            Static("", id="pending-queue", markup=False),
            _ReasoningEffortSelector(id="reasoning-effort-selector"),
            _PermissionSelector(id="permission-selector"),
            _ConversationInput(id="conversation-input", placeholder="Message Aide"),
            Static("Ready", id="status-bar", markup=False),
            id="conversation-input-region",
        )
        yield Static(
            "Terminal window is too small. Resize to continue.",
            id="size-insufficient",
            markup=False,
        )

    async def on_mount(self) -> None:
        self._conversation_display = self.query_one("#conversation-display", _ConversationDisplay)
        self._conversation_input = self.query_one("#conversation-input", _ConversationInput)
        self._bind_bus_callback(self._bus)
        self._bus_snapshot = await self._bus.inbound_snapshot()
        if self._confirmation_coordinator is None:
            self._bind_confirmation_callback(self._control, self._bus)
        else:
            self._confirmation_coordinator.bind_presenter(self)
        self._outbound_worker = self.run_worker(
            self._consume_outbound(),
            name="conversation-outbound",
            group="conversation-outbound",
            exclusive=True,
            exit_on_error=False,
        )
        self._restore_result_worker = self.run_worker(
            self._restore_startup_notification(),
            name="restore-startup-notification",
            group="restore-startup-notification",
            exclusive=False,
            exit_on_error=False,
        )
        if not self._size_insufficient:
            self.query_one(_ConversationInput).focus()
        self._schedule_status_refresh()

    async def on_unmount(self, event: Unmount) -> None:
        del event
        self._closing = True
        confirmation_result = self._confirmation_result
        if confirmation_result is not None and not confirmation_result.done():
            confirmation_result.cancel()
        permission_warning_result = self._permission_warning_result
        if permission_warning_result is not None and not permission_warning_result.done():
            permission_warning_result.cancel()
        session_switch_result = self._session_switch_result
        if session_switch_result is not None and not session_switch_result.done():
            session_switch_result.cancel()
        if self._confirmation_coordinator is None:
            self._unbind_confirmation_callback(self._control)
        else:
            await self._confirmation_coordinator.unbind_presenter(self)
        self._unbind_bus_callback(self._bus)
        projection = self._active_run_projection
        self._active_run_projection = None
        if projection is not None:
            projection.stop()
        cleanup_errors: list[BaseException] = []
        try:
            if self._outbound_worker is not None:
                self._outbound_worker.cancel()
                with suppress(WorkerError):
                    await self._outbound_worker.wait()
            if self._resume_worker is not None:
                self._resume_worker.cancel()
                with suppress(WorkerError):
                    await self._resume_worker.wait()
            if self._restore_worker is not None:
                self._restore_worker.cancel()
                with suppress(WorkerError):
                    await self._restore_worker.wait()
            if self._restore_result_worker is not None:
                self._restore_result_worker.cancel()
                with suppress(WorkerError):
                    await self._restore_result_worker.wait()
        except BaseException as worker_error:
            cleanup_errors.append(worker_error)

        for run in self._consumed_runs:
            run.projection.stop()
        self._consumed_runs.clear()
        self._pending_inputs.clear()
        self._bus_snapshot = ()
        self._run_ready = Event()
        self._cancel_requested_turn = None
        self._active_confirmation_id = None
        self._confirmation_result = None
        self._active_confirmation_token = None
        self._dismissed_confirmation_tokens.clear()
        self._session_switch_result = None
        self._permission_current_level = None
        self._permission_warning_result = None
        self._completion_options = ()
        self._completion_dismissed_text = None
        self._outbound_worker = None
        self._resume_worker = None
        self._restore_worker = None
        self._restore_result_worker = None
        self._restore_workflow_active = False
        self._restore_anchors = ()
        self._restore_plan = None
        self._presentation_quiesced = True
        self._active_confirmation_token = None
        self._conversation_display = None
        self._conversation_input = None

        primary_error = self._application_error
        if primary_error is not None and cleanup_errors:
            causes: list[BaseException] = []
            if primary_error.__cause__ is not None:
                causes.append(primary_error.__cause__)
            causes.extend(cleanup_errors)
            unique_causes: list[BaseException] = []
            for error in causes:
                if not any(error is existing for existing in unique_causes):
                    unique_causes.append(error)
            cause: BaseException = (
                unique_causes[0]
                if len(unique_causes) == 1
                else BaseExceptionGroup("Terminal Conversation cleanup failed", unique_causes)
            )
            raise primary_error from cause
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise cleanup_errors[0] from BaseExceptionGroup(
                "Additional Terminal Conversation cleanup failures",
                cleanup_errors[1:],
            )

    def _bind_bus_callback(self, bus: MessageBus) -> None:
        def on_snapshot(snapshot: tuple[InboundMessage, ...]) -> None:
            if self._closing or self._presentation_quiesced:
                return
            self.post_message(
                self.InboundSnapshotChanged(
                    bus,
                    snapshot,
                    promote_removed=not self._draining_inputs,
                    callback=on_snapshot,
                )
            )

        self._bus_callback = on_snapshot
        self._bus_callback_bus = bus
        bus.set_inbound_changed_callback(on_snapshot)

    def _unbind_bus_callback(self, bus: MessageBus) -> None:
        callback = self._bus_callback
        if callback is None or self._bus_callback_bus is not bus:
            return
        bus.unbind_inbound_changed_callback(callback)
        self._bus_callback = None
        self._bus_callback_bus = None

    def _bind_confirmation_callback(
        self,
        control: TerminalAgentRunExecutorControl,
        bus: MessageBus,
    ) -> None:
        def on_confirmation(request: ConfirmationRequestView) -> None:
            self.post_message(
                self.ConfirmationRequested(
                    request,
                    control=control,
                    bus=bus,
                )
            )

        self._confirmation_callback = on_confirmation
        self._confirmation_control = control
        control.bind_confirmation_callback(on_confirmation)

    def _unbind_confirmation_callback(self, control: TerminalAgentRunExecutorControl) -> None:
        callback = self._confirmation_callback
        if callback is None or self._confirmation_control is not control:
            return
        unbind = getattr(control, "unbind_confirmation_callback", None)
        if callable(unbind):
            unbind(callback)
        self._confirmation_callback = None
        self._confirmation_control = None

    @property
    def command_completion_visible(self) -> bool:
        return bool(self._completion_options)

    @property
    def has_pending_input(self) -> bool:
        return bool(self._pending_inputs) or any(not run.started for run in self._consumed_runs)

    @on(TextArea.Changed, "#conversation-input")
    def _input_changed(self, message: TextArea.Changed) -> None:
        if isinstance(message.text_area, _ConversationInput):
            message.text_area._ctrl_c_turn_token = None
        self._refresh_command_completion(message.text_area.text)

    @on(_CommandCompletion.OptionSelected)
    def _completion_selected(self, message: _CommandCompletion.OptionSelected) -> None:
        if message.option_list.id != "command-completion":
            return
        self._select_command_completion(message.option_index)

    @on(_CommandCompletion.Dismissed)
    def _completion_dismissed(self, message: _CommandCompletion.Dismissed) -> None:
        message.stop()
        self.dismiss_command_completion()

    @on(_ReasoningEffortSelector.Confirmed)
    async def _reasoning_effort_confirmed(
        self,
        message: _ReasoningEffortSelector.Confirmed,
    ) -> None:
        message.stop()
        self._close_reasoning_effort_selector()
        try:
            result = await self._management_dispatcher.update_reasoning_effort(message.effort)
        except Exception as error:
            self._handle_exception(error)
            return
        if result.output is not None:
            await self._mount_management_rows("/effort", result.output)
        self._schedule_status_refresh()

    @on(_ReasoningEffortSelector.Cancelled)
    def _reasoning_effort_cancelled(self, message: _ReasoningEffortSelector.Cancelled) -> None:
        message.stop()
        self._close_reasoning_effort_selector()

    @on(_PermissionSelector.Confirmed)
    def _permission_confirmed(
        self,
        message: _PermissionSelector.Confirmed,
    ) -> None:
        message.stop()
        selected = message.permission_level
        current = self._permission_current_level
        self._close_permission_selector()
        if current is None or selected == current:
            return
        self.run_worker(
            self._apply_permission_selection(selected, current),
            name="permission-selection",
            group="permission-selection",
            exclusive=True,
            exit_on_error=False,
        )

    async def _apply_permission_selection(
        self,
        selected: ToolPermissionLevel,
        current: ToolPermissionLevel,
    ) -> None:
        if selected == "full-access" and current != "full-access":
            if not await self._confirm_full_access_warning():
                return
        try:
            result = await self._management_dispatcher.update_permission_level(selected)
        except Exception as error:
            self._handle_exception(error)
            return
        if result.output is not None:
            await self._mount_management_rows("/permission", result.output)
        self._schedule_status_refresh()

    @on(_PermissionSelector.Cancelled)
    def _permission_cancelled(self, message: _PermissionSelector.Cancelled) -> None:
        message.stop()
        self._close_permission_selector()

    @on(_ActivityGroupHeading.Clicked)
    def _activity_group_clicked(self, message: _ActivityGroupHeading.Clicked) -> None:
        message.stop()
        state = message.heading.activity_group
        if state is not None and state.toggleable:
            display = self.query_one("#conversation-display", _ConversationDisplay)
            display.layout_changed(state.heading)
            self._set_activity_group_expanded(state, not state.expanded)
        if not message.heading.has_focus:
            with suppress(NoMatches, NoScreen, ScreenStackError):
                self.screen.set_focus(
                    self.query_one("#conversation-input", _ConversationInput),
                    scroll_visible=False,
                )

    def on_resize(self, event: Resize) -> None:
        self._resize_completion()
        self._render_status_bar()
        self._refresh_pending_queue()
        too_small = (
            event.size.width < _MIN_TERMINAL_WIDTH or event.size.height < _MIN_TERMINAL_HEIGHT
        )
        if too_small == self._size_insufficient:
            return

        display = self.query_one("#conversation-display", _ConversationDisplay)
        if too_small:
            display.suspend_for_size()
        self._size_insufficient = too_small
        if too_small:
            self._viable_size.clear()
        else:
            self._viable_size.set()
        input_region = self.query_one("#conversation-input-region", Vertical)
        size_state = self.query_one("#size-insufficient", Static)
        display.display = not too_small
        input_region.display = not too_small
        size_state.display = too_small
        if too_small:
            size_screen = _SizeInsufficientScreen()
            self._size_screen = size_screen
            self.push_screen(size_screen)
            return

        active_size_screen = self._size_screen
        self._size_screen = None
        if active_size_screen is not None and active_size_screen is self.screen:
            active_size_screen.dismiss()
        if not too_small and not self._closing and not self._presentation_quiesced:
            self.refresh(layout=True)
            display.resume_from_size()
            display.restore_resize_anchor()
            # Let the layout-driven scroll range settle before the final retry.
            display.schedule_resize_anchor_retry()
            self.call_after_refresh(self._restore_input_focus_after_size)

    def _restore_input_focus_after_size(self) -> None:
        if self._size_insufficient or self._closing or self._presentation_quiesced:
            return
        self.screen.refresh(layout=True)
        if isinstance(self.screen, _ToolConfirmationScreen):
            self.screen.restore_after_size()
        if len(self.screen_stack) != 1:
            return
        with suppress(Exception):
            effort_selector = self.query_one(
                "#reasoning-effort-selector",
                _ReasoningEffortSelector,
            )
            permission_selector = self.query_one("#permission-selector", _PermissionSelector)
            if effort_selector.display:
                effort_selector.focus()
            elif permission_selector.display:
                permission_selector.focus()
            else:
                self.query_one(_ConversationInput).focus()

    def move_command_completion(self, direction: int) -> None:
        if not self._completion_options:
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        highlighted = completion.highlighted
        current = 0 if highlighted is None else highlighted
        completion.highlighted = max(
            0,
            min(len(self._completion_options) - 1, current + direction),
        )

    def dismiss_command_completion(self) -> None:
        input_area = self.query_one("#conversation-input", _ConversationInput)
        self._hide_command_completion(remember_text=input_area.text)
        input_area.focus()

    def accept_command_completion(self) -> None:
        if not self._completion_options:
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        highlighted = completion.highlighted
        index = 0 if highlighted is None else highlighted
        selected = self._completion_options[index]
        input_area = self.query_one("#conversation-input", _ConversationInput)
        should_submit = (
            selected.kind is _CompletionCandidateKind.MANAGEMENT
            and input_area.text == selected.insert_text
        )
        if should_submit and selected.token == _RELOAD_SKILL_MANAGEMENT_COMMAND_TOKEN:
            self.post_message(_ConversationInput.Submitted(input_area, selected.insert_text))
            return
        self._select_command_completion(index)
        if should_submit:
            self.post_message(_ConversationInput.Submitted(input_area, selected.insert_text))

    def _select_command_completion(self, index: int) -> None:
        if not self._completion_options:
            return
        selected = self._completion_options[index]
        input_area = self.query_one("#conversation-input", _ConversationInput)
        input_area.text = selected.insert_text
        input_area.move_cursor(
            (len(input_area.document.lines) - 1, len(input_area.document.lines[-1]))
        )
        self._hide_command_completion(remember_text=selected.insert_text)
        input_area.focus()

    def _refresh_command_completion(self, text: str) -> None:
        if text == self._completion_dismissed_text:
            self._hide_command_completion()
            return
        self._completion_dismissed_text = None
        candidates = _completion_candidates(
            text,
            self._skill_metadata,
            management_command_tokens=self._management_command_tokens,
        )
        if not candidates:
            self._hide_command_completion()
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        completion.set_options(
            Option(candidate.display_label, id=candidate.token) for candidate in candidates
        )
        completion.highlighted = 0
        completion.display = True
        self._completion_options = candidates
        self._resize_completion()

    def _resize_completion(self) -> None:
        with suppress(NoMatches, NoScreen, ScreenStackError):
            completion = self.query_one("#command-completion", _CommandCompletion)
            new_content = self.query_one("#new-content", Static)
            reserved = 7 + (2 if self._pending_inputs else 0) + int(new_content.display)
            completion.styles.height = max(1, min(7, self.size.height - reserved))

    def _schedule_status_refresh(self) -> None:
        self._status_generation += 1
        generation = self._status_generation
        self._status_view = None
        self._render_status_bar()
        self.run_worker(
            self._refresh_status_bar(generation),
            name="status-bar-refresh",
            group="status-bar-refresh",
            exclusive=False,
            exit_on_error=False,
        )

    async def _refresh_status_bar(self, generation: int) -> None:
        try:
            result = await self._management_dispatcher.dispatch("/status")
        except Exception:
            result = None
        if generation != self._status_generation or self._closing:
            return
        self._status_view = None if result is None else result.status_view
        self._render_status_bar()

    def _render_status_bar(self) -> None:
        with suppress(NoMatches, NoScreen, ScreenStackError):
            status = self._status_view
            state = "Working" if self._working else "Ready"
            if self._service_run_projection and not self._control.foreground_input_admitted():
                state = (
                    "Reconnecting" if self._service_connection_state == "recovering"
                    else "Session unavailable"
                )
            width = self.size.width
            available = max(1, width - 2)
            content = state
            if width >= 40:
                model = "-" if status is None else status.chat_model
                permission = "-" if status is None else status.current_permission_level
                full = f"{state} | Model: {model} | Permission: {permission}"
                if cell_len(full) <= available:
                    content = full
                else:
                    model_width = max(1, available - cell_len(state) - 3)
                    content = f"{state} | {_queue_excerpt(f'Model: {model}', model_width)}"
            elif width >= 28:
                permission = "-" if status is None else status.current_permission_level
                compact = f"{state} | {permission}"
                if cell_len(compact) <= available:
                    content = compact
            self.query_one("#status-bar", Static).update(content)

    def _hide_command_completion(self, *, remember_text: str | None = None) -> None:
        self._completion_options = ()
        if remember_text is not None:
            self._completion_dismissed_text = remember_text
        with suppress(NoMatches, NoScreen, ScreenStackError):
            completion = self.query_one("#command-completion", _CommandCompletion)
            completion.set_options(())
            completion.display = False

    def _open_reasoning_effort_selector(
        self,
        effort: ReasoningEffort,
        input_area: _ConversationInput,
    ) -> None:
        selector = self.query_one("#reasoning-effort-selector", _ReasoningEffortSelector)
        self._hide_command_completion()
        selector.set_effort(effort)
        input_area.text = ""
        input_area.display = False
        selector.display = True
        selector.focus()

    def _close_reasoning_effort_selector(self) -> None:
        selector = self.query_one("#reasoning-effort-selector", _ReasoningEffortSelector)
        input_area = self.query_one("#conversation-input", _ConversationInput)
        selector.display = False
        input_area.display = True
        input_area.text = ""
        input_area.focus()

    def _open_permission_selector(
        self,
        permission_level: ToolPermissionLevel,
        input_area: _ConversationInput,
    ) -> None:
        selector = self.query_one("#permission-selector", _PermissionSelector)
        self._hide_command_completion()
        self._permission_current_level = permission_level
        selector.set_permission_level(permission_level)
        input_area.text = ""
        input_area.display = False
        selector.display = True
        selector.focus()

    def _close_permission_selector(self) -> None:
        selector = self.query_one("#permission-selector", _PermissionSelector)
        input_area = self.query_one("#conversation-input", _ConversationInput)
        selector.display = False
        input_area.display = True
        input_area.text = ""
        input_area.focus()
        self._permission_current_level = None

    async def _confirm_full_access_warning(self) -> bool:
        if self._permission_warning_result is not None:
            return False
        await self._viable_size.wait()
        result = asyncio.get_running_loop().create_future()
        self._permission_warning_result = result

        def on_dismissed(value: bool | None) -> None:
            if self._permission_warning_result is result:
                self._permission_warning_result = None
            if not result.done():
                result.set_result(value)

        try:
            await self.push_screen(_FullAccessWarningScreen(), callback=on_dismissed)
            return (await result) is True
        finally:
            if self._permission_warning_result is result:
                self._permission_warning_result = None

    @on(_ConversationDisplay.Resized)
    def _display_resized(self, message: _ConversationDisplay.Resized) -> None:
        compact = message.width <= _COMPACT_MESSAGE_MAX_WIDTH
        for content in message.display.query(".message"):
            content.set_class(compact, "message-compact")
        if not self._size_insufficient:
            message.display.restore_resize_anchor(message.generation)
            message.display.schedule_resize_anchor_retry(message.generation)

    @on(_ConversationInput.Submitted)
    async def _submit_input(self, message: _ConversationInput.Submitted) -> None:
        text = message.text
        if (
            not text.strip()
            or self._size_insufficient
            or (self._resume_worker is not None and not self._resume_worker.is_finished)
            or self._restore_workflow_active
        ):
            return
        if text.strip().casefold() in {"exit", "quit"}:
            message.text_area.text = ""
            self.exit()
            return

        submit_user_input = getattr(self._management_dispatcher, "submit_user_input", None)
        self._draining_inputs = True
        self._submitting_input = True
        try:
            result = (
                await submit_user_input(text)
                if callable(submit_user_input)
                else await self._management_dispatcher.dispatch(text)
            )
        except Exception as error:
            if not self._service_run_projection:
                raise
            await self._mount_management_rows(
                "input", f"{getattr(error, 'code', 'submission_failed')}: {error}"
            )
            return
        finally:
            self._draining_inputs = False
            self._submitting_input = False
            self._run_ready.set()
        if result.handled:
            self._remove_pending_text(text)
            if result.effort_selection is not None:
                message.text_area.remember_submission(text)
                self._open_reasoning_effort_selector(result.effort_selection, message.text_area)
                return
            if result.permission_selection is not None:
                message.text_area.remember_submission(text)
                self._open_permission_selector(result.permission_selection, message.text_area)
                return
            if result.skill_metadata is not None:
                self._skill_metadata = tuple(result.skill_metadata)
                self._hide_command_completion()
            elif text == _RELOAD_SKILL_MANAGEMENT_COMMAND_TOKEN:
                message.text_area.remember_submission(text)
                await self._mount_management_rows(text, result.output)
                return
            message.text_area.remember_submission(text)
            message.text_area.text = ""
            if result.restore_listing is not None:
                listing = result.restore_listing
                if not listing.anchors:
                    await self._management_dispatcher.restore_cancel()
                    await self._mount_management_rows(text, result.output)
                    return
                self._restore_workflow_active = True
                self._restore_anchors = listing.anchors
                message.text_area.read_only = True
                self._restore_worker = self.run_worker(
                    self._run_restore_workflow(listing.anchors, message.text_area),
                    name="restore-session",
                    group="restore-session",
                    exclusive=False,
                    exit_on_error=False,
                )
            elif result.resume_sessions is not None:
                await self._open_resume_picker(
                    result.resume_sessions,
                    message.text_area,
                    skipped_count=result.resume_skipped_count,
                )
            else:
                await self._mount_management_rows(
                    text, result.output, status_view=result.status_view
                )
            return

        foreground_input_admitted = getattr(self._control, "foreground_input_admitted", None)
        if (
            not result.submitted
            and callable(foreground_input_admitted)
            and not foreground_input_admitted()
        ):
            self._remove_pending_text(text)
            message.text_area.text = ""
            await self._mount_management_rows(
                "input",
                "restore_in_progress: Session Restore is waiting for confirmation.",
            )
            return
        message.text_area.remember_submission(text)
        if message.text_area.text == text:
            message.text_area.text = ""
        self._pending_inputs.append(text)
        self._refresh_pending_queue()
        if result.submitted:
            self._promote_consumed_inputs(1, run_id=result.submitted_run_id)
        else:
            await self._bus.put_inbound(InboundMessage(content=text))

    @on(_ConversationInput.ControlAction)
    async def _handle_control_action(self, message: _ConversationInput.ControlAction) -> None:
        message.stop()
        text_area = message.text_area
        if message.action == "cancel_active_turn":
            turn_token = message.turn_token
            if (
                turn_token is None
                or turn_token is not text_area.active_turn_token
                or self._cancel_requested_turn is turn_token
            ):
                return
            self._cancel_requested_turn = turn_token
            try:
                await self._control.cancel_active_run()
            except Exception as error:
                if self._cancel_requested_turn is turn_token:
                    self._cancel_requested_turn = None
                self._handle_exception(error)
            return
        if message.action == "drain_pending":
            await self._drain_pending_inputs(text_area)
            return
        if message.action == "clear_draft":
            text_area.text = ""
            return
        self.exit()

    async def _drain_pending_inputs(self, text_area: _ConversationInput) -> None:
        if text_area.text:
            return
        recall_queued_inputs = getattr(self._management_dispatcher, "recall_queued_inputs", None)
        if callable(recall_queued_inputs):
            self._draining_inputs = True
            try:
                recalled = await recall_queued_inputs()
            finally:
                self._draining_inputs = False
            await self._restore_recalled_inputs(recalled, text_area)
            return
        if not self._pending_inputs:
            return
        pending_before = len(self._pending_inputs)
        self._draining_inputs = True
        try:
            drained = await self._bus.drain_inbound()
        finally:
            self._draining_inputs = False
        consumed_during_drain = max(0, pending_before - len(drained))
        if consumed_during_drain:
            self._promote_consumed_inputs(consumed_during_drain)
        if not drained:
            return
        for _message in drained:
            if self._pending_inputs:
                self._pending_inputs.popleft()
        text_area.text = "\n".join(message.content for message in drained)
        text_area.move_cursor(
            (len(text_area.document.lines) - 1, len(text_area.document.lines[-1]))
        )
        self._refresh_pending_queue()

    async def _restore_recalled_inputs(
        self,
        recalled: list[dict[str, str]],
        text_area: _ConversationInput,
    ) -> None:
        if not recalled:
            return
        recalled_ids = {item["run_id"] for item in recalled}
        removed_ids: set[str] = set()
        retained: deque[_ConsumedRun] = deque()
        for run in self._consumed_runs:
            if run.run_id in recalled_ids:
                await run.projection.close()
                removed_ids.add(run.run_id)
            else:
                retained.append(run)
        self._consumed_runs = retained
        recalled_texts: list[str] = []
        for item in recalled:
            if item["run_id"] not in removed_ids:
                for index, pending in enumerate(self._pending_inputs):
                    if pending == item["text"]:
                        del self._pending_inputs[index]
                        break
            recalled_texts.append(item["text"])
        input_area = self._conversation_input or text_area
        input_area.active_turn_token = (
            self._consumed_runs[0].turn_id if self._consumed_runs else None
        )
        self._set_working(bool(self._consumed_runs))
        if self._consumed_runs:
            self._run_ready.set()
        text_area.text = "\n".join([*recalled_texts, *([text_area.text] if text_area.text else [])])
        text_area.move_cursor(
            (len(text_area.document.lines) - 1, len(text_area.document.lines[-1]))
        )
        self._refresh_pending_queue()

    def _remove_pending_text(self, text: str) -> None:
        with suppress(ValueError):
            self._pending_inputs.remove(text)
        self._refresh_pending_queue()

    def _on_inbound_snapshot_for(
        self,
        bus: MessageBus,
        snapshot: tuple[InboundMessage, ...],
        *,
        promote_removed: bool,
    ) -> None:
        if bus is not self._bus or self._closing or self._presentation_quiesced:
            return
        previous = self._bus_snapshot
        self._bus_snapshot = snapshot
        removed = max(0, len(previous) - len(snapshot))
        if removed and promote_removed:
            self._promote_consumed_inputs(removed)
        self._refresh_pending_queue()

    @on(InboundSnapshotChanged)
    def _inbound_snapshot_changed(self, message: InboundSnapshotChanged) -> None:
        if message.callback is not self._bus_callback:
            return
        self._on_inbound_snapshot_for(
            message.bus,
            message.snapshot,
            promote_removed=message.promote_removed,
        )

    def _promote_consumed_inputs(self, count: int, *, run_id: str | None = None) -> None:
        promoted = False
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        input_area = self._conversation_input
        if input_area is None:
            input_area = self.query_one("#conversation-input", _ConversationInput)
        for _ in range(count):
            if not self._pending_inputs:
                break
            text = self._pending_inputs.popleft()
            turn_id = UUID(int=uuid4().int)
            projection = _MessageBusRunProjection(self, turn_id, display=display)
            projection.start()
            self._consumed_runs.append(
                _ConsumedRun(
                    turn_id=turn_id,
                    user_text=text,
                    projection=projection,
                    run_id=run_id if _ == 0 else None,
                )
            )
            promoted = True
            if input_area.active_turn_token is None:
                input_area.active_turn_token = turn_id
                self._set_working(True)
        if promoted:
            self._run_ready.set()

    async def _consume_service_state(self, outbound: OutboundMessage) -> bool:
        if "_remote_connection_state" in outbound.metadata:
            self._service_connection_state = str(outbound.metadata["_remote_connection_state"])
            await self._mount_management_output(outbound.content, scroll=False)
            self._render_status_bar()
            if outbound.metadata.get("_remote_recovery_error") is True:
                self.push_screen(
                    _ConversationRecoveryScreen(outbound.content),
                    callback=self._conversation_recovery_selected,
                )
            return True
        snapshot = outbound.metadata.get("_remote_state_snapshot")
        if not isinstance(snapshot, dict):
            return False
        session_id = snapshot.get("session_id")
        if (not isinstance(session_id, str)
                or session_id != self._control.project_foreground_conversation().session_id):
            return True
        messages = snapshot.get("messages")
        if not isinstance(messages, list):
            return True
        projection = ForegroundConversationProjection(
            session_id, tuple(message for message in messages if isinstance(message, dict)),
        )
        live = snapshot.get("live_state")
        runs = live.get("runs") if isinstance(live, dict) else None
        if not isinstance(runs, list):
            return True
        live_ids = [run.get("run_id") for run in runs if isinstance(run, dict)]
        if (outbound.metadata.get("_remote_snapshot_reason") == "initial_subscribe"
                and outbound.metadata.get("_remote_snapshot_rebuild") is not True
                and live_ids == [run.run_id for run in self._consumed_runs]):
            return True
        previous = {run.run_id: run for run in self._consumed_runs}
        for run in self._consumed_runs:
            await run.projection.close()
        self._consumed_runs.clear()
        await self._replace_display_from_projection(projection)
        for state in runs:
            if not isinstance(state, dict) or not isinstance(state.get("run_id"), str):
                continue
            old = previous.get(state["run_id"])
            turn_id = old.turn_id if old is not None else uuid4()
            current = _ConsumedRun(
                turn_id, str(state.get("prompt", "")), _MessageBusRunProjection(self, turn_id),
                run_id=state["run_id"],
            )
            self._consumed_runs.append(current)
            if state.get("status") == "accepted":
                continue
            await self._start_consumed_run(current)
            segments = state.get("response_segments", [])
            tools = state.get("tools", [])
            if not isinstance(segments, list) or not isinstance(tools, list):
                continue
            for index, tool in enumerate(tools):
                if index < len(segments) and isinstance(segments[index], str) and segments[index]:
                    await current.projection.consume(OutboundMessage(
                        "model_response", segments[index], {"_stream_delta": True},
                    ))
                if not isinstance(tool, dict):
                    continue
                await current.projection.consume(OutboundMessage(
                    "tool_call", str(tool.get("name", "")),
                    {"tool_call_id": tool.get("tool_call_id"), "arguments": tool.get("arguments")},
                ))
                status = {"completed": "success", "failed": "error", "rejected": "refused"}.get(
                    str(tool.get("status")),
                )
                if status is not None:
                    await current.projection.consume(OutboundMessage(
                        "tool_call", str(tool.get("name", "")),
                        {"tool_call_id": tool.get("tool_call_id"), "status": status},
                    ))
            if segments and isinstance(segments[-1], str) and segments[-1]:
                await current.projection.consume(OutboundMessage(
                    "model_response", segments[-1], {"_stream_delta": True},
                ))
        input_area = self._conversation_input
        if input_area is not None:
            input_area.active_turn_token = self._consumed_runs[0].turn_id if self._consumed_runs else None
        self._set_working(bool(self._consumed_runs))
        self._refresh_pending_queue()
        if self._consumed_runs:
            self._run_ready.set()
        return True

    def _conversation_recovery_selected(self, action: str | None) -> None:
        if action is None or self._closing:
            return
        self._resume_worker = self.run_worker(
            self._recover_selected_conversation(action == "new"),
            name="recover-conversation", group="resume-session", exit_on_error=False,
        )

    async def _recover_selected_conversation(self, create_new: bool) -> None:
        recover = getattr(self._management_dispatcher, "recover_conversation", None)
        if recover is None:
            return
        try:
            await recover(create_new=create_new)
        except Exception as error:
            message = str(error)
            await self._mount_management_output(message, scroll=False)
            self.push_screen(
                _ConversationRecoveryScreen(message), callback=self._conversation_recovery_selected,
            )

    async def _consume_outbound(self) -> None:
        try:
            buffered: OutboundMessage | None = None
            while not self._closing and not self._presentation_quiesced:
                if not self._consumed_runs:
                    buffered = await self._wait_for_consumed_run_or_discard_orphan()
                if self._closing or self._presentation_quiesced:
                    return
                if not self._consumed_runs:
                    continue
                run = self._consumed_runs[0]
                if not run.started and not self._service_run_projection:
                    await self._start_consumed_run(run)
                outbound = buffered
                buffered = None
                if outbound is None:
                    outbound = await self._bus.get_outbound()
                if await self._consume_service_state(outbound):
                    continue
                remote_run_id = outbound.metadata.pop("_remote_run_id", None)
                if isinstance(remote_run_id, str):
                    remote_run = next(
                        (item for item in self._consumed_runs if item.run_id == remote_run_id),
                        None,
                    )
                    if remote_run is None:
                        continue
                    run = remote_run
                remote_started = outbound.metadata.pop("_remote_run_started", False)
                if not run.started:
                    await self._start_consumed_run(run)
                if remote_started:
                    continue
                await run.projection.consume(outbound)
                if run.projection.terminal_seen:
                    await run.projection.close()
                    self._consumed_runs.popleft()
                    self._finish_consumed_run(run)
        except CancelledError:
            if not self._closing and not self._presentation_quiesced:
                raise
        except Exception as error:
            self._handle_exception(error)

    async def _wait_for_consumed_run_or_discard_orphan(
        self,
    ) -> OutboundMessage | None:
        while not self._closing and not self._presentation_quiesced and not self._consumed_runs:
            self._run_ready.clear()
            if self._consumed_runs:
                break
            run_ready = create_task(self._run_ready.wait())
            outbound = create_task(self._bus.get_outbound())
            try:
                done, _ = await asyncio.wait(
                    {run_ready, outbound},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if outbound in done:
                    message = outbound.result()
                    if await self._consume_service_state(message):
                        continue
                    if self._submitting_input and not self._consumed_runs:
                        await self._run_ready.wait()
                    if run_ready in done or self._consumed_runs:
                        return message
                    continue
            finally:
                for task in (run_ready, outbound):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(run_ready, outbound, return_exceptions=True)
        return None

    async def _start_consumed_run(self, run: _ConsumedRun) -> None:
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await self._mount_user_message(run.user_text, display)
        run.started = True
        input_area = self._conversation_input
        if input_area is None:
            input_area = self.query_one("#conversation-input", _ConversationInput)
        input_area.active_turn_token = run.turn_id
        self._active_run_projection = run.projection
        self._set_working(True)
        self._refresh_pending_queue()
        display.content_changed()

    def _finish_consumed_run(self, run: _ConsumedRun) -> None:
        if self._active_run_projection is run.projection:
            self._active_run_projection = None
        input_area = self._conversation_input
        if input_area is None:
            input_area = self.query_one("#conversation-input", _ConversationInput)
        if self._consumed_runs:
            input_area.active_turn_token = self._consumed_runs[0].turn_id
            self._set_working(True)
            self._refresh_pending_queue()
            return
        if input_area.active_turn_token is run.turn_id:
            input_area.active_turn_token = None
        if self._cancel_requested_turn is run.turn_id:
            self._cancel_requested_turn = None
        self._set_working(False)
        self._refresh_pending_queue()
        if not self._closing and not self._presentation_quiesced:
            with suppress(Exception):
                input_area.focus()

    @on(ConfirmationRequested)
    def _confirmation_requested(self, message: ConfirmationRequested) -> None:
        self.run_worker(
            self._request_confirmation(
                message.request,
                message.bound_control,
                message.bound_bus,
            ),
            name="tool-confirmation",
            group="tool-confirmation",
            exclusive=True,
            exit_on_error=False,
        )

    @on(CoordinatorConfirmationRequested)
    def _coordinator_confirmation_requested(
        self,
        message: CoordinatorConfirmationRequested,
    ) -> None:
        if message.token in self._dismissed_confirmation_tokens:
            self._dismissed_confirmation_tokens.discard(message.token)
            return
        self.run_worker(
            self._request_coordinator_confirmation(
                message.envelope,
                message.token,
                message.respond,
            ),
            name="coordinator-tool-confirmation",
            group="coordinator-tool-confirmation",
            exclusive=True,
            exit_on_error=False,
        )

    async def _request_coordinator_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, CoordinatorConfirmationDecision], bool],
    ) -> None:
        coordinator = self._confirmation_coordinator
        if coordinator is None:
            return
        if token in self._dismissed_confirmation_tokens:
            self._dismissed_confirmation_tokens.discard(token)
            return
        if self._closing or self._presentation_quiesced:
            await coordinator.cancel_owner(envelope.owner)
            return
        self._active_confirmation_token = token
        input_area = self._conversation_input
        if input_area is None:
            input_area = self.query_one("#conversation-input", _ConversationInput)
        input_was_read_only = input_area.read_only
        input_area.read_only = True
        result: asyncio.Future[ConfirmationDecision | None] | None = None
        try:
            await self._viable_size.wait()
            if self._closing or self._presentation_quiesced:
                await coordinator.cancel_owner(envelope.owner)
                return
            if token in self._dismissed_confirmation_tokens:
                self._dismissed_confirmation_tokens.discard(token)
                return
            result = asyncio.get_running_loop().create_future()
            self._confirmation_result = result

            def on_dismissed(value: ConfirmationDecision | None) -> None:
                if not result.done():
                    result.set_result(value)

            await self.push_screen(
                _ToolConfirmationScreen(envelope),
                callback=on_dismissed,
            )
            if token in self._dismissed_confirmation_tokens:
                self._dismissed_confirmation_tokens.discard(token)
                if isinstance(self.screen, _ToolConfirmationScreen):
                    self.screen.dismiss(None)
            decision = await result
            if decision in {"approved", "declined"}:
                respond(token, decision)
            else:
                await coordinator.cancel_owner(envelope.owner)
        except CancelledError:
            await coordinator.cancel_owner(envelope.owner)
            raise
        except Exception:
            await coordinator.cancel_owner(envelope.owner)
            raise
        finally:
            input_area.read_only = input_was_read_only
            if result is not None and self._confirmation_result is result:
                self._confirmation_result = None
            if self._active_confirmation_token is token:
                self._active_confirmation_token = None
            self._dismissed_confirmation_tokens.discard(token)
            self._refresh_pending_queue()
            if not self._closing and not self._presentation_quiesced:
                with suppress(Exception):
                    input_area.focus()

    async def _request_confirmation(
        self,
        request: ConfirmationRequestView,
        control: TerminalAgentRunExecutorControl,
        bus: MessageBus,
    ) -> None:
        if (
            self._closing
            or self._presentation_quiesced
            or control is not self._control
            or bus is not self._bus
        ):
            return
        if self._active_confirmation_id is not None:
            self._respond_to_confirmation_if_pending(
                request.confirmation_id,
                "declined",
                control=control,
            )
            return
        self._active_confirmation_id = request.confirmation_id
        input_area = self._conversation_input
        if input_area is None:
            input_area = self.query_one("#conversation-input", _ConversationInput)
        input_was_read_only = input_area.read_only
        input_area.read_only = True
        result: asyncio.Future[ConfirmationDecision | None] | None = None
        try:
            await self._viable_size.wait()
            result = asyncio.get_running_loop().create_future()
            self._confirmation_result = result

            def on_dismissed(value: ConfirmationDecision | None) -> None:
                if not result.done():
                    result.set_result(value)

            await self.push_screen(
                _ToolConfirmationScreen(request),
                callback=on_dismissed,
            )
            decision = await result
            if decision not in {"approved", "declined"}:
                decision = "declined"
        except BaseException:
            if not self._closing and not self._presentation_quiesced:
                with suppress(Exception):
                    self._respond_to_confirmation_if_pending(
                        request.confirmation_id,
                        "declined",
                        control=control,
                    )
            raise
        else:
            self._respond_to_confirmation_if_pending(
                request.confirmation_id,
                decision,
                control=control,
            )
        finally:
            input_area.read_only = input_was_read_only
            if result is not None and self._confirmation_result is result:
                self._confirmation_result = None
            if bus is self._bus:
                self._bus_snapshot = await bus.inbound_snapshot()
            self._refresh_pending_queue()
            if self._active_confirmation_id == request.confirmation_id:
                self._active_confirmation_id = None

    def _respond_to_confirmation_if_pending(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
        *,
        control: TerminalAgentRunExecutorControl | None = None,
    ) -> bool:
        try:
            target_control = self._control if control is None else control
            target_control.respond_to_confirmation(confirmation_id, decision)
        except ValueError:
            return False
        return True

    async def _mount_user_message(
        self,
        content: str,
        display: _ConversationDisplay,
    ) -> None:
        row = Horizontal(classes="message-row user-row")
        await display.mount(row)
        if not row.is_attached:
            await row._mounted_event.wait()
        await row.mount(
            Static(
                content,
                markup=False,
                classes=self._message_classes("user-message", display),
            )
        )

    async def _mount_assistant(
        self,
        content: str = "",
        display: _ConversationDisplay | None = None,
        parent: Widget | None = None,
    ) -> Markdown:
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        if parent is None:
            parent = display
        assistant = Markdown(
            content,
            classes=self._message_classes("assistant-message", display),
            open_links=False,
            parser_factory=_markdown_parser,
        )
        row = Horizontal(classes="message-row assistant-row")
        await parent.mount(row)
        if not row.is_attached:
            await row._mounted_event.wait()
        await row.mount(assistant)
        return assistant

    async def _mount_activity_group(
        self,
        display: _ConversationDisplay,
        *,
        expanded: bool,
        toggleable: bool,
        elapsed: float,
    ) -> _ActivityGroupState:
        heading = _ActivityGroupHeading(
            _activity_group_heading_text(
                expanded=expanded,
                elapsed=elapsed,
                toggleable=toggleable,
            ),
            markup=False,
            classes="agent-run-activity-heading",
        )
        container = Vertical(classes="agent-run-activity-group")
        await display.mount(container)
        if not container.is_attached:
            await container._mounted_event.wait()
        content = Vertical(classes="agent-run-activity-content")
        await container.mount(heading, content)
        if not content.is_attached:
            await content._mounted_event.wait()
        state = _ActivityGroupState(
            heading=heading,
            content=content,
            expanded=expanded,
            toggleable=toggleable,
            elapsed=elapsed,
        )
        heading.activity_group = state
        heading.can_focus = toggleable
        heading.set_class(not toggleable, "-running")
        if not expanded:
            content.display = False
        self._scroll_to_latest()
        return state

    def _set_activity_group_expanded(
        self,
        activity_group: _ActivityGroupState,
        expanded: bool,
    ) -> None:
        if not activity_group.toggleable:
            expanded = True
        activity_group.expanded = expanded
        activity_group.content.display = expanded
        activity_group.heading.update(
            _activity_group_heading_text(
                expanded=expanded,
                elapsed=activity_group.elapsed,
                outcome=activity_group.outcome,
                toggleable=activity_group.toggleable,
            )
        )
        self._scroll_to_latest()

    @staticmethod
    def _reparent_mounted_widget(widget: Widget, parent: Widget) -> None:
        current_parent = widget.parent
        if current_parent is parent:
            return
        if not isinstance(current_parent, Widget):
            raise RuntimeError("Widget is not mounted under another widget")

        # Textual's public move_child only reorders siblings, while remove() prunes
        # the mounted subtree. Update the DOM links directly to preserve its state.
        current_parent._nodes._remove(widget)
        widget._detach()
        widget._attach(parent)
        parent._nodes._append(widget)

        current_parent.update_node_styles(animate=False)
        parent.update_node_styles(animate=False)
        current_parent.refresh(layout=True)
        parent.refresh(layout=True)

    async def _mount_tool_message(
        self,
        tool_name: str,
        status: _ToolRowStatus,
        summary: str,
        display: _ConversationDisplay,
        parent: Widget | None = None,
        raw_arguments: str | None = None,
    ) -> Static:
        row = Static(
            _tool_row_content(status, tool_name, summary, raw_arguments=raw_arguments),
            markup=False,
            classes="tool-row",
        )
        if parent is None:
            parent = display
        await parent.mount(row)
        return row

    def _scroll_to_latest(self, display: _ConversationDisplay | None = None) -> None:
        if display is None:
            display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        display.content_changed()

    async def _mount_status(
        self,
        content: str,
        display: _ConversationDisplay | None = None,
    ) -> None:
        if display is None:
            display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(Static(content, markup=False, classes="turn-status"))
        self._scroll_to_latest(display)

    async def _mount_management_rows(
        self, command: str, output: str | None, *, status_view: RuntimeStatus | None = None
    ) -> None:
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(
            Static(
                f"Command: {command}",
                markup=False,
                classes="management-row management-heading",
            )
        )
        if output is not None:
            await self._mount_management_output(
                _status_view_text(status_view) if status_view is not None else output,
                scroll=False,
            )
        self._scroll_to_latest()

    async def _mount_management_output(self, output: str, *, scroll: bool = True) -> None:
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(
            Static(output, markup=False, classes="management-row management-output")
        )
        if scroll:
            self._scroll_to_latest()

    async def _replace_display_from_session(self, expected_session_id: str) -> bool:
        conversation_projection = self._control.project_foreground_conversation()
        if conversation_projection.session_id != expected_session_id:
            return False
        replaced = await self._replace_display_from_projection(conversation_projection)
        if replaced:
            self._schedule_status_refresh()
        return replaced

    async def _replace_display_from_projection(
        self,
        conversation_projection: ForegroundConversationProjection,
    ) -> bool:
        projected_messages = conversation_projection.messages
        run_projection = self._active_run_projection
        self._active_run_projection = None
        if run_projection is not None:
            run_projection.stop()
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.remove_children()
        for partition in _persisted_message_partitions(projected_messages):
            historical = _classify_historical_partition(partition)
            if historical is None:
                for message in partition:
                    role, content = _persisted_role_and_content(message)
                    await self._mount_persisted_message(message, role, content, display)
                continue

            user = partition[0]
            role, content = _persisted_role_and_content(user)
            await self._mount_persisted_message(user, role, content, display)
            if historical.activity:
                group = await self._mount_activity_group(
                    display,
                    expanded=historical.outcome != "completed",
                    toggleable=True,
                    elapsed=historical.elapsed,
                )
                group.outcome = historical.outcome
                if historical.outcome is not None:
                    group.heading.set_class(True, f"-{historical.outcome}")
                group.heading.update(
                    _activity_group_heading_text(
                        expanded=group.expanded,
                        elapsed=group.elapsed,
                        outcome=group.outcome,
                        toggleable=True,
                    )
                )
                for item in historical.activity:
                    await self._mount_persisted_activity_message(item, display, group.content)
            if historical.final is not None and historical.final.content.strip():
                final = historical.final
                await self._mount_persisted_message(
                    final.message,
                    final.role,
                    final.content,
                    display,
                )
            if historical.terminal_status is not None:
                await display.mount(
                    Static(historical.terminal_status, markup=False, classes="turn-status")
                )
        display.reset_to_latest()
        return True

    async def _mount_persisted_message(
        self,
        message: Mapping[str, object],
        role: str,
        content: str,
        display: _ConversationDisplay,
        *,
        parent: Widget | None = None,
    ) -> None:
        if role == "user":
            await self._mount_user_message(content, display)
            return
        if role == "assistant":
            if content:
                await self._mount_assistant(content, display, parent=parent)
            status = _persisted_assistant_status(message)
            if status is not None:
                await display.mount(Static(status, markup=False, classes="turn-status"))
            return
        if role == "tool":
            await self._mount_persisted_tool_message(message, display, parent=parent)

    async def _mount_persisted_activity_message(
        self,
        item: _PersistedMessageProjection,
        display: _ConversationDisplay,
        parent: Widget,
    ) -> None:
        if item.role == "assistant":
            if item.content:
                await self._mount_assistant(item.content, display, parent=parent)
            return
        if item.role == "tool":
            await self._mount_persisted_tool_message(item.message, display, parent=parent)

    async def _mount_persisted_tool_message(
        self,
        message: Mapping[str, object],
        display: _ConversationDisplay,
        *,
        parent: Widget | None = None,
    ) -> None:
        # Persisted Tool content is a raw result, not a display-safe activity summary.
        await self._mount_tool_message(
            cast(str, message["name"]),
            cast(_ToolRowStatus, message["status"]),
            "",
            display,
            parent=parent,
        )

    async def _restore_picker_selection(
        self,
        anchors: tuple[RestoreAnchor, ...],
    ) -> int | None:
        result: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()

        def on_dismissed(value: int | None) -> None:
            if not result.done():
                result.set_result(value)

        await self.push_screen(
            _RestoreAnchorPickerScreen(anchors),
            callback=on_dismissed,
        )
        return await result

    async def _restore_inspection(
        self,
        anchor_id: int,
    ) -> tuple[ManagementCommandResult | None, bool]:
        cancelled: asyncio.Future[bool | None] = asyncio.get_running_loop().create_future()
        waiting = _RestoreWaitingScreen()

        def on_dismissed(value: bool | None) -> None:
            if not cancelled.done():
                cancelled.set_result(value)

        await self.push_screen(waiting, callback=on_dismissed)
        inspection_task = asyncio.create_task(
            self._management_dispatcher.restore_inspect(anchor_id)
        )
        try:
            done, _ = await asyncio.wait(
                (inspection_task, cancelled),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancelled in done and cancelled.result() is False:
                with suppress(Exception, CancelledError):
                    await self._management_dispatcher.restore_cancel()
                if not inspection_task.done():
                    inspection_task.cancel()
                with suppress(Exception, CancelledError):
                    await inspection_task
                return None, True
            inspection = await inspection_task
            return inspection, False
        except CancelledError:
            if not inspection_task.done():
                inspection_task.cancel()
            with suppress(Exception, CancelledError):
                await inspection_task
            raise
        finally:
            if self.screen is waiting:
                with suppress(Exception):
                    await waiting.dismiss(True)

    async def _restore_mode_selection(self, plan: RestorePlan) -> RestoreMode | None:
        result: asyncio.Future[RestoreMode | None] = asyncio.get_running_loop().create_future()

        def on_dismissed(value: RestoreMode | None) -> None:
            if not result.done():
                result.set_result(value)

        await self.push_screen(
            _RestoreModeScreen(plan),
            callback=on_dismissed,
        )
        return await result

    async def _restore_confirmation(self, plan: RestorePlan, mode: RestoreMode) -> bool:
        result: asyncio.Future[bool | None] = asyncio.get_running_loop().create_future()

        def on_dismissed(value: bool | None) -> None:
            if not result.done():
                result.set_result(value)

        await self.push_screen(
            _RestoreConfirmationScreen(plan, mode),
            callback=on_dismissed,
        )
        return (await result) is True

    async def _show_restore_failure_notification(self, result: RestoreResult) -> None:
        if not result.failure_notification_pending:
            return
        acknowledged: asyncio.Future[bool | None] = asyncio.get_running_loop().create_future()

        def on_dismissed(value: bool | None) -> None:
            if not acknowledged.done():
                acknowledged.set_result(value)

        await self.push_screen(
            _RestoreFailureScreen(result),
            callback=on_dismissed,
        )
        if (await acknowledged) is not True:
            return
        acknowledge = getattr(
            self._management_dispatcher,
            "restore_acknowledge_failure",
            None,
        )
        if callable(acknowledge):
            await acknowledge()

    async def _restore_startup_notification(self) -> None:
        input_area: _ConversationInput | None = None
        try:
            result_command = await self._management_dispatcher.restore_result()
            result = result_command.restore_result
            if not isinstance(result, RestoreResult) or not result.failure_notification_pending:
                return
            input_area = self._conversation_input
            if input_area is None:
                input_area = self.query_one("#conversation-input", _ConversationInput)
            self._restore_workflow_active = True
            input_area.read_only = True
            await self._show_restore_failure_notification(result)
        except CancelledError:
            raise
        except Exception:
            return
        finally:
            if self._restore_worker is None:
                self._restore_workflow_active = False
                if input_area is not None:
                    input_area.read_only = False
                    if not self._closing and not self._presentation_quiesced:
                        with suppress(Exception):
                            input_area.focus()

    async def _run_restore_workflow(
        self,
        anchors: tuple[RestoreAnchor, ...],
        input_area: _ConversationInput,
    ) -> None:
        committed = False
        try:
            anchor_id = await self._restore_picker_selection(anchors)
            if anchor_id is None:
                await self._management_dispatcher.restore_cancel()
                return
            selected_anchor = next(
                (anchor for anchor in anchors if anchor.anchor_id == anchor_id),
                None,
            )
            if selected_anchor is None:
                await self._management_dispatcher.restore_cancel()
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    "Session Restore selection is no longer available.",
                )
                return

            inspection, cancelled = await self._restore_inspection(anchor_id)
            if cancelled:
                return
            assert inspection is not None
            plan = inspection.restore_plan
            if plan is None:
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    inspection.output or "Session Restore could not be inspected.",
                )
                return
            if plan.anchor_id != selected_anchor.anchor_id:
                await self._management_dispatcher.restore_cancel()
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    "Session Restore selection changed before confirmation.",
                )
                return
            self._restore_plan = plan

            if plan.targets or plan.backup_gaps or plan.integrity_issues:
                mode = await self._restore_mode_selection(plan)
                if mode is None:
                    await self._management_dispatcher.restore_cancel()
                    return
            else:
                mode = RestoreMode.CONVERSATION_ONLY

            if not await self._restore_confirmation(plan, mode):
                await self._management_dispatcher.restore_cancel()
                return

            committed = True
            commit = await self._management_dispatcher.restore_commit(plan.anchor_id, mode)
            result = commit.restore_result
            if not isinstance(result, RestoreResult):
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    commit.output or "Session Restore could not be completed.",
                )
                return
            if result.anchor_id != plan.anchor_id or result.session_id != plan.session_id:
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    "Session Restore result did not match the inspected plan.",
                )
                return

            input_area.forget_submissions(
                tuple(anchor.content for anchor in anchors if anchor.anchor_id >= plan.anchor_id)
            )
            self._restore_plan = None
            if not await self._replace_display_from_session(result.session_id):
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    "Conversation Session authority changed before display replacement.",
                )
                return
            input_area.text = selected_anchor.content
            input_area.move_cursor(
                (len(input_area.document.lines) - 1, len(input_area.document.lines[-1]))
            )
            await self._mount_management_rows(
                _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                commit.output,
            )
            if result.failure_notification_pending:
                await self._show_restore_failure_notification(result)
        except FatalManagementError as fatal_error:
            self._fatal_management_error = fatal_error
            if not self._closing:
                self.exit(return_code=1)
        except CancelledError:
            if not committed:
                with suppress(Exception):
                    await self._management_dispatcher.restore_cancel()
            raise
        except Exception:
            if not committed:
                with suppress(Exception):
                    await self._management_dispatcher.restore_cancel()
            if not self._closing:
                await self._mount_management_rows(
                    _RESTORE_MANAGEMENT_COMMAND_TOKEN,
                    "Session Restore failed.",
                )
        finally:
            self._restore_plan = None
            self._restore_anchors = ()
            self._restore_workflow_active = False
            input_area.read_only = False
            if not self._closing and not self._presentation_quiesced:
                with suppress(Exception):
                    input_area.focus()

    async def _open_resume_picker(
        self,
        sessions: tuple[SessionListingEntry, ...],
        input_area: _ConversationInput,
        *,
        skipped_count: int,
    ) -> None:
        await self._viable_size.wait()
        await self.push_screen(
            _SessionPickerScreen(sessions, skipped_count=skipped_count),
            callback=lambda session_id: self._resume_picker_dismissed(
                session_id,
                input_area,
            ),
        )

    def _resume_picker_dismissed(
        self,
        session_id: str | None,
        input_area: _ConversationInput,
    ) -> None:
        if session_id is None:
            with suppress(Exception):
                input_area.focus()
            return
        if self._resume_worker is not None and not self._resume_worker.is_finished:
            return
        input_area.read_only = True
        self._resume_worker = self.run_worker(
            self._resume_selected_session(session_id, input_area),
            name="resume-session",
            group="resume-session",
            exclusive=False,
            exit_on_error=False,
        )

    async def _confirm_active_session_switch(self) -> bool:
        if self._session_switch_result is not None:
            return False
        await self._viable_size.wait()
        result = asyncio.get_running_loop().create_future()
        self._session_switch_result = result

        def on_dismissed(value: bool | None) -> None:
            if not result.done():
                result.set_result(value)

        try:
            await self.push_screen(
                _SessionSwitchConfirmationScreen(),
                callback=on_dismissed,
            )
            return (await result) is True
        finally:
            if self._session_switch_result is result:
                self._session_switch_result = None

    async def _resume_selected_session(
        self,
        session_id: str,
        input_area: _ConversationInput,
    ) -> None:
        try:
            dispatcher = self._management_dispatcher
            previous_control = self._control
            force = False
            if self._control.has_active_run:
                if not await self._confirm_active_session_switch():
                    return
                force = True
            try:
                result = await dispatcher.resume(session_id, force=force)
            except FatalManagementError as fatal_error:
                self._fatal_management_error = fatal_error
                self.exit(return_code=1)
                return
            except Exception as error:
                await self._mount_management_rows(
                    _RESUME_MANAGEMENT_COMMAND_TOKEN,
                    str(error) if getattr(error, "code", None) == "result_unknown"
                    else "Session resume failed.",
                )
                self._render_status_bar()
                if getattr(error, "code", None) == "result_unknown":
                    self.push_screen(
                        _ConversationRecoveryScreen(str(error)),
                        callback=self._conversation_recovery_selected,
                    )
                return
            resumed_session_id = result.resumed_session_id
            if resumed_session_id != session_id:
                await self._mount_management_rows(_RESUME_MANAGEMENT_COMMAND_TOKEN, result.output)
                return
            if self._control.project_foreground_conversation().session_id != resumed_session_id:
                await self._mount_management_rows(
                    _RESUME_MANAGEMENT_COMMAND_TOKEN,
                    "Session resume did not select the requested Conversation Session.",
                )
                return
            if self._control is not previous_control:
                return
            if not await self._replace_display_from_session(resumed_session_id):
                await self._mount_management_rows(
                    _RESUME_MANAGEMENT_COMMAND_TOKEN,
                    "Conversation Session authority changed before display replacement.",
                )
        finally:
            input_area.read_only = False
            if not self._closing and not self._presentation_quiesced:
                with suppress(Exception):
                    input_area.focus()

    def _refresh_pending_queue(self) -> None:
        with suppress(NoMatches, NoScreen, ScreenStackError):
            queue = self.query_one("#pending-queue", Static)
            pending = [run.user_text for run in self._consumed_runs if not run.started]
            pending.extend(self._pending_inputs)
            if not pending:
                queue.update("")
                queue.display = False
                return
            count = len(pending)
            available = max(1, self.size.width - 2)
            suffix = f" +{count - 2} more" if count > 2 else ""
            shown = pending[:2]
            separator_width = 3 if len(shown) == 2 else 0
            share = max(1, (available - cell_len(suffix) - separator_width) // len(shown))
            summaries = [_queue_excerpt(" ".join(value.split()), min(48, share)) for value in shown]
            detail = " | ".join(summaries) + suffix
            queue.update(f"Pending ({count})\n{detail}")
            queue.display = True

    def _set_working(self, working: bool) -> None:
        self._working = working
        self._render_status_bar()

    @staticmethod
    def _message_classes(role: str, display: _ConversationDisplay) -> str:
        compact = display.size.width <= _COMPACT_MESSAGE_MAX_WIDTH
        suffix = " message-compact" if compact else ""
        return f"message {role}{suffix}"






def _stream_is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False



def is_interactive_terminal() -> bool:
    """Return whether the standard streams can support a full-screen conversation."""

    def stream_is_interactive(current: object, driver_stream: object | None) -> bool:
        return _stream_is_tty(current) and _stream_is_tty(driver_stream)

    return all(
        (
            stream_is_interactive(sys.stdin, sys.__stdin__),
            stream_is_interactive(sys.stdout, sys.__stdout__),
            stream_is_interactive(sys.stderr, sys.__stderr__),
        )
    )


__all__ = ["TerminalConversationApp", "is_interactive_terminal"]
