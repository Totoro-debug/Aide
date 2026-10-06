"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from asyncio import CancelledError, Event, Task, create_task, sleep
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import partial
from time import monotonic as monotonic_now
from typing import ClassVar, Final, Literal, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token
from rich.cells import cell_len, set_cell_size
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.driver import Driver
from textual.events import Click, Key, MouseScrollDown, MouseScrollUp, Resize, Unmount
from textual.message import Message
from textual.screen import ModalScreen
from textual.scrollbar import ScrollTo
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerError

from omni.agent.confirmation import (
    ConfirmationDecision as CoordinatorConfirmationDecision,
)
from omni.agent.confirmation import (
    ConfirmationEnvelope,
    ConfirmationPresentationCoordinator,
    ConfirmationUnavailable,
)
from omni.agent.loop import (
    ConfirmationRequestView,
    ForegroundConversationProjection,
    TerminalAgentRunExecutorControl,
)
from omni.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from omni.agent.permission import PERMISSION_LEVELS, ToolPermissionLevel
from omni.agent.session.restore import RestoreMode, RestorePlan, RestoreResult
from omni.agent.session.session import RestoreAnchor
from omni.management.commands import (
    MANAGEMENT_COMMANDS,
    RELOAD_SKILL_MANAGEMENT_COMMAND,
    RESUME_MANAGEMENT_COMMAND,
    ManagementCommandDispatcher,
    ManagementCommandResult,
    format_restore_preview,
)
from omni.management.service import FatalManagementError, RuntimeStatus, SessionListingEntry
from omni.provider.models import REASONING_EFFORT_LEVELS, ReasoningEffort
from omni.skills.catalog import SkillMetadata
from omni.terminal.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter

__all__ = [
    "TerminalConversationApp",
    "is_interactive_terminal",
]

_COMPACT_MESSAGE_MAX_WIDTH = 60
_MIN_TERMINAL_WIDTH = 20
_MIN_TERMINAL_HEIGHT = 10
_CONVERSATION_NAVIGATION_KEYS = frozenset({"pageup", "pagedown", "ctrl+home", "ctrl+end"})
_FAILURE_REASON_MAX_CHARS = 120
_TOOL_NAME_MAX_CHARS = 80
_GENERIC_TOOL_FAILURE_REASON = "The operation did not complete."
_RELOAD_SKILL_MANAGEMENT_COMMAND_TOKEN = RELOAD_SKILL_MANAGEMENT_COMMAND.token
_RESUME_MANAGEMENT_COMMAND_TOKEN = RESUME_MANAGEMENT_COMMAND.token
_RESTORE_MANAGEMENT_COMMAND_TOKEN = "/restore"
_UNSAFE_TOOL_DETAIL_PATTERN = re.compile(
    r"(?:^\s*[\[{])|(?:[\"'][^\"']+[\"']\s*:)|"
    r"(?:\b(?:api[_-]?key|authorization|bearer|password|secret|token)\b)|"
    r"(?:\b(?:arguments?|parameters?|result|output|content)\b\s*[:=])|"
    r"(?:\bcall[-_][A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_ACTIVITY_EXPANDED_SYMBOL = "\u25bc"
_ACTIVITY_COLLAPSED_SYMBOL = "\u25b6"
_SPARSE_MARKERS = ("_stream_delta", "_stream_end", "_streamed")
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
_PERMISSION_LABELS: Final[tuple[str, ...]] = (
    "Read-Only",
    "Workspace-Write",
    "Full-Access",
)
type _ControlAction = Literal["cancel_active_turn", "clear_draft", "drain_pending", "exit"]
type ConfirmationDecision = Literal["approved", "declined"]
type _ToolRowStatus = Literal["running", "success", "error", "refused", "cancelled", "unknown"]
type _TerminalOutcome = Literal["completed", "cancelled", "failed"]


class _DriverLifecycleHooks(Protocol):
    write: Callable[[str], None]
    flush: Callable[[], None]
    start_application_mode: Callable[[], None]
    stop_application_mode: Callable[[], None]


class _ConsoleRestoreHooks(Protocol):
    _restore_console: Callable[[], None] | None


@dataclass(slots=True)
class _ToolRowState:
    widget: Static
    tool_name: str
    status: _ToolRowStatus


class _ActivityGroupHeading(Static):
    """Disclosure title for one Agent Run Activity Group."""

    FOCUS_ON_CLICK = False
    can_focus = True
    activity_group: _ActivityGroupState | None = None

    class Clicked(Message):
        def __init__(self, heading: _ActivityGroupHeading) -> None:
            super().__init__()
            self.heading = heading

    @on(Click)
    async def _on_click(self, event: Click) -> None:
        if event.widget is not self:
            return
        event.stop()
        event.prevent_default()
        self.post_message(self.Clicked(self))

    async def _on_key(self, event: Key) -> None:
        if event.key in {"enter", "space"} and self.activity_group is not None:
            event.stop()
            event.prevent_default()
            if self.activity_group.toggleable:
                self.post_message(self.Clicked(self))
            return
        await super()._on_key(event)

    def on_unmount(self, event: Unmount) -> None:
        del event
        self.activity_group = None


@dataclass(slots=True)
class _ActivityGroupState:
    heading: _ActivityGroupHeading
    content: Vertical
    expanded: bool = True
    toggleable: bool = False
    elapsed: float = 0.0
    outcome: _TerminalOutcome | None = None


@dataclass(frozen=True, slots=True)
class _PersistedMessageProjection:
    message: Mapping[str, object]
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class _HistoricalRunProjection:
    activity: tuple[_PersistedMessageProjection, ...]
    final: _PersistedMessageProjection | None
    terminal_status: str | None
    outcome: _TerminalOutcome | None
    elapsed: float


class _ConversationInput(TextArea):
    """Multiline input whose ordinary Enter key submits the current draft."""

    class ControlAction(Message):
        def __init__(
            self,
            text_area: _ConversationInput,
            action: _ControlAction,
            *,
            turn_token: object | None = None,
        ) -> None:
            super().__init__()
            self.text_area = text_area
            self.action = action
            self.turn_token = turn_token

    class Submitted(Message):
        def __init__(self, text_area: _ConversationInput, text: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.text = text

    def on_mount(self) -> None:
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self.active_turn_token: object | None = None
        self._ctrl_c_turn_token: object | None = None

    def remember_submission(self, text: str) -> None:
        """Keep accepted input only for this live application instance."""
        self._history.append(text)
        self._history_index = None
        self._history_draft = ""

    def forget_submissions(self, submissions: Sequence[str]) -> None:
        """Remove the newest matching accepted inputs after Session truncation."""
        history = list(self._history)
        for submission in submissions:
            for index in range(len(history) - 1, -1, -1):
                if history[index] == submission:
                    del history[index]
                    break
        self._history = history
        self._leave_history()

    def _navigate_history(self, direction: int) -> bool:
        if not self._history or (self.text and self._history_index is None):
            return False

        if self._history_index is None:
            self._history_draft = self.text
            self._history_index = len(self._history) - 1
        else:
            next_index = self._history_index + direction
            if next_index < 0:
                next_index = 0
            if next_index >= len(self._history):
                self._history_index = None
                self.text = self._history_draft
                return True
            self._history_index = next_index

        self.text = self._history[self._history_index]
        self.move_cursor((len(self.document.lines) - 1, len(self.document.lines[-1])))
        return True

    def _leave_history(self) -> None:
        self._history_index = None
        self._history_draft = ""

    async def _on_key(self, event: Key) -> None:
        completion = cast(_CommandCompletionHost, self.app)
        if event.key != "ctrl+c":
            self._ctrl_c_turn_token = None
        if event.key in {"up", "down", "left", "right"} and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            if event.key in {"up", "down"}:
                completion.move_command_completion(-1 if event.key == "up" else 1)
            return
        if event.key == "escape" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.dismiss_command_completion()
            return
        if event.key == "enter" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.accept_command_completion()
            return
        if event.key == "ctrl+c" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.dismiss_command_completion()
            return
        if event.key in _CONVERSATION_NAVIGATION_KEYS:
            event.stop()
            event.prevent_default()
            self.app.query_one("#conversation-display", _ConversationDisplay).navigate(event.key)
            return
        if event.key == "up" and not self.text and getattr(self.app, "has_pending_input", False):
            event.stop()
            event.prevent_default()
            self.post_message(self.ControlAction(self, "drain_pending"))
            return
        control_action: _ControlAction | None = None
        turn_token: object | None = None
        if event.key == "ctrl+c":
            if self.active_turn_token is not None:
                control_action = "cancel_active_turn"
                turn_token = self.active_turn_token
                if turn_token is not None:
                    self._ctrl_c_turn_token = turn_token
            elif self._ctrl_c_turn_token is not None:
                control_action = "cancel_active_turn"
                turn_token = self._ctrl_c_turn_token
            elif self.text:
                control_action = "clear_draft"
            else:
                control_action = "exit"
        elif event.key == "ctrl+d" and not self.text:
            control_action = "exit"
        if control_action is not None:
            event.stop()
            event.prevent_default()
            self.post_message(self.ControlAction(self, control_action, turn_token=turn_token))
            return
        if event.key in {"up", "down"} and self._navigate_history(-1 if event.key == "up" else 1):
            event.stop()
            event.prevent_default()
            return
        action = EnhancedKeyboardAdapter.parse(event.key)
        if self._history_index is not None and (
            event.is_printable
            or event.key in {"backspace", "delete", "ctrl+backspace", "ctrl+delete"}
            or action is not None
        ):
            self._leave_history()

        if action is EnhancedKeyboardAction.NEWLINE:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if action is EnhancedKeyboardAction.SUBMIT:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        await super()._on_key(event)


class _CommandCompletionHost(Protocol):
    @property
    def command_completion_visible(self) -> bool: ...

    def move_command_completion(self, direction: int) -> None: ...

    def dismiss_command_completion(self) -> None: ...

    def accept_command_completion(self) -> None: ...


class _CompletionCandidateKind(StrEnum):
    MANAGEMENT = "management"
    SKILL = "skill"


@dataclass(frozen=True, slots=True)
class _CompletionCandidate:
    """Typed presentation data for one completion option."""

    kind: _CompletionCandidateKind
    token: str
    description: str
    insert_text: str

    @property
    def display_label(self) -> Text:
        """Return a markup-disabled, single-line label for OptionList."""
        description = " ".join(self.description.split())
        label = Text(no_wrap=True, overflow="ellipsis", end="")
        label.append(self.token, style="bold")
        label.append(f"  {self.kind.value.title()}  ", style="dim")
        label.append(description, style="dim")
        return label


class _CommandCompletion(OptionList):
    class Dismissed(Message):
        pass

    async def _on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.Dismissed())
            return
        await super()._on_key(event)


class _ReasoningEffortSelector(Static):
    """Focused horizontal selector for the Runtime-Lifetime Reasoning Effort."""

    can_focus = True

    class Confirmed(Message):
        def __init__(self, selector: _ReasoningEffortSelector) -> None:
            super().__init__()
            self.effort = selector.selected_effort

    class Cancelled(Message):
        pass

    def __init__(self, effort: ReasoningEffort = "medium", *, id: str | None = None) -> None:
        super().__init__("", id=id, markup=False)
        self._selected_index = 0
        self._current_effort = effort
        self.set_effort(effort)

    @property
    def selected_effort(self) -> ReasoningEffort:
        return REASONING_EFFORT_LEVELS[self._selected_index]

    def set_effort(self, effort: ReasoningEffort) -> None:
        self._current_effort = effort
        self._selected_index = REASONING_EFFORT_LEVELS.index(effort)
        self._refresh_content()

    def _refresh_content(self) -> None:
        content = Text(f"Current: {self._current_effort} | Pending: {self.selected_effort}\n")
        for index, effort in enumerate(REASONING_EFFORT_LEVELS):
            if index:
                content.append("  ")
            if index == self._selected_index:
                content.append(
                    effort,
                    style=Style(reverse=True, bold=True, meta={"effort": effort}),
                )
            else:
                content.append(effort, style=Style(meta={"effort": effort}))
        self.update(content)

    @on(Click)
    async def _on_click(self, event: Click) -> None:
        if event.widget is not self:
            return
        selected = event.style.meta.get("effort")
        if selected not in REASONING_EFFORT_LEVELS:
            return
        event.stop()
        event.prevent_default()
        self._selected_index = REASONING_EFFORT_LEVELS.index(selected)
        self._refresh_content()
        self.post_message(self.Confirmed(self))

    async def _on_key(self, event: Key) -> None:
        if event.key in {"left", "right"}:
            event.stop()
            event.prevent_default()
            direction = -1 if event.key == "left" else 1
            self._selected_index = max(
                0,
                min(len(REASONING_EFFORT_LEVELS) - 1, self._selected_index + direction),
            )
            self._refresh_content()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Confirmed(self))
            return
        if event.key in {"escape", "ctrl+c"}:
            event.stop()
            event.prevent_default()
            self.post_message(self.Cancelled())
            return
        await super()._on_key(event)


class _PermissionSelector(Static):
    """Focused horizontal selector for the foreground Tool Permission Level."""

    can_focus = True

    class Confirmed(Message):
        def __init__(self, selector: _PermissionSelector) -> None:
            super().__init__()
            self.permission_level = selector.selected_permission_level

    class Cancelled(Message):
        pass

    def __init__(
        self,
        permission_level: ToolPermissionLevel = "workspace-write",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__("", id=id, markup=False)
        self._selected_index = 0
        self._current_level = permission_level
        self.set_permission_level(permission_level)

    @property
    def selected_permission_level(self) -> ToolPermissionLevel:
        return PERMISSION_LEVELS[self._selected_index]

    def set_permission_level(self, permission_level: ToolPermissionLevel) -> None:
        self._current_level = permission_level
        self._selected_index = PERMISSION_LEVELS.index(permission_level)
        self._refresh_content()

    def _refresh_content(self) -> None:
        available_width = self.content_region.width
        inline_width = sum(map(len, _PERMISSION_LABELS)) + 2 * (len(_PERMISSION_LABELS) - 1)
        separator = "\n" if available_width and available_width < inline_width else "  "
        current_label = _PERMISSION_LABELS[PERMISSION_LEVELS.index(self._current_level)]
        pending_label = _PERMISSION_LABELS[self._selected_index]
        if available_width and available_width < 50:
            short_labels = ("Read", "Write", "Full")
            current_short = short_labels[PERMISSION_LEVELS.index(self._current_level)]
            pending_short = short_labels[self._selected_index]
            prefix = "Current:" if available_width >= 22 else ""
            heading = f"{prefix}{current_short} -> {pending_short}"
        else:
            heading = f"Current: {current_label} | Pending: {pending_label}"
        content = Text(f"{heading}\n", no_wrap=True)
        for index, (level, label) in enumerate(
            zip(PERMISSION_LEVELS, _PERMISSION_LABELS, strict=True)
        ):
            if index:
                content.append(separator)
            content.append(
                label,
                style=Style(
                    bold=index == self._selected_index,
                    reverse=index == self._selected_index,
                    meta={"permission_level": level},
                ),
            )
        self.update(content)

    def on_resize(self, event: Resize) -> None:
        del event
        self._refresh_content()

    @on(Click)
    async def _on_click(self, event: Click) -> None:
        if event.widget is not self:
            return
        selected = event.style.meta.get("permission_level")
        if selected not in PERMISSION_LEVELS:
            return
        event.stop()
        event.prevent_default()
        self.set_permission_level(cast(ToolPermissionLevel, selected))
        self.post_message(self.Confirmed(self))

    async def _on_key(self, event: Key) -> None:
        if event.key in {"left", "right"}:
            event.stop()
            event.prevent_default()
            direction = -1 if event.key == "left" else 1
            self._selected_index = max(
                0,
                min(len(PERMISSION_LEVELS) - 1, self._selected_index + direction),
            )
            self._refresh_content()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Confirmed(self))
            return
        if event.key in {"escape", "ctrl+c"}:
            event.stop()
            event.prevent_default()
            self.post_message(self.Cancelled())
            return
        await super()._on_key(event)


class _FullAccessWarningScreen(ModalScreen[bool]):
    """Require an explicit foreground Full-Access risk acknowledgement."""

    CSS = """
    _FullAccessWarningScreen {
        align: center middle;
        padding: 1 2;
    }

    #permission-warning-panel {
        width: 80%;
        max-width: 72;
        height: 90%;
        max-height: 90%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
        overflow-y: hidden;
    }

    #permission-warning-copy {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }

    #permission-warning-heading,
    #permission-warning-message,
    #permission-warning-details {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #permission-warning-heading {
        text-style: bold;
    }

    #permission-warning-details {
        color: $text-warning;
    }

    #permission-warning-actions {
        width: 100%;
        height: 3;
        align: center middle;
    }

    #permission-warning-actions Button {
        width: 1fr;
        min-width: 0;
        margin: 0;
        height: 3;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
        Binding("left,up", "focus_cancel", "Cancel", show=False),
        Binding("right,down", "focus_confirm", "Enable Full-Access", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id="permission-warning")

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-warning-panel"):
            yield Static("Enable Full-Access?", id="permission-warning-heading", markup=False)
            with VerticalScroll(id="permission-warning-copy"):
                yield Static(
                    "Full-Access skips ordinary permission prompts for valid foreground File,"
                    " eligible non-catastrophic Exec, private or non-global Web Fetch, MCP,"
                    " and available Schedule Tool calls. Process-local; not persisted.",
                    id="permission-warning-message",
                    markup=False,
                )
                yield Static(
                    "No OS sandbox. Validation, unavailable capabilities, business refusals,"
                    " and Tool errors still apply. Catastrophic or uncertain Exec still"
                    " requires confirmation.",
                    id="permission-warning-details",
                    markup=False,
                )
            with Horizontal(id="permission-warning-actions"):
                yield Button("Cancel", id="permission-warning-cancel")
                yield Button(
                    "Enable",
                    variant="warning",
                    id="permission-warning-confirm",
                )

    def on_mount(self) -> None:
        self.query_one("#permission-warning-cancel", Button).focus()

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "permission-warning-confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_focus_cancel(self) -> None:
        self.query_one("#permission-warning-cancel", Button).focus()

    def action_focus_confirm(self) -> None:
        self.query_one("#permission-warning-confirm", Button).focus()


class _SessionPickerScreen(ModalScreen[str | None]):
    """Choose one validated Conversation Session without changing it yet."""

    CSS = """
    _SessionPickerScreen {
        align: center middle;
        padding: 1 2;
    }

    #session-picker-panel {
        width: 80%;
        max-width: 72;
        height: 80%;
        max-height: 90%;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #session-picker-heading {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
    }

    .session-picker-notice {
        width: 100%;
        margin-bottom: 1;
        color: $text-muted;
    }

    #session-picker-options {
        width: 100%;
        height: 1fr;
        min-height: 3;
        overflow-y: auto;
    }

    #session-picker-filter {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        sessions: tuple[SessionListingEntry, ...],
        *,
        skipped_count: int,
    ) -> None:
        super().__init__(id="session-picker")
        self._sessions = sessions
        self._skipped_count = skipped_count

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-panel"):
            yield Static("Resume Session", id="session-picker-heading", markup=False)
            if self._skipped_count:
                noun = "Session" if self._skipped_count == 1 else "Sessions"
                yield Static(
                    f"Skipped {self._skipped_count} corrupt Conversation {noun}.",
                    markup=False,
                    classes="session-picker-notice",
                )
            if not self._sessions:
                yield Static(
                    "No resumable Conversation Sessions.",
                    markup=False,
                    classes="session-picker-notice",
                )
            yield Input(placeholder="Filter by title", id="session-picker-filter")
            yield OptionList(
                *(
                    Option(_session_picker_label(session), id=session.id)
                    for session in self._sessions
                ),
                id="session-picker-options",
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#session-picker-filter", Input).focus()

    @on(Input.Changed, "#session-picker-filter")
    def _filter_changed(self, message: Input.Changed) -> None:
        search = message.value.casefold().strip()
        options = self.query_one("#session-picker-options", OptionList)
        options.set_options(
            Option(_session_picker_label(session), id=session.id)
            for session in self._sessions
            if search in session.title.casefold()
        )
        if options.option_count:
            options.highlighted = 0

    @on(Key)
    def _filter_key(self, event: Key) -> None:
        if not self.query_one("#session-picker-filter", Input).has_focus:
            return
        options = self.query_one("#session-picker-options", OptionList)
        if event.key == "down" and options.option_count:
            event.stop()
            event.prevent_default()
            options.focus()
        elif event.key == "enter" and options.option_count:
            event.stop()
            event.prevent_default()
            option = options.get_option_at_index(options.highlighted or 0)
            self.dismiss(option.id)

    @on(OptionList.OptionSelected, "#session-picker-options")
    def _option_selected(self, message: OptionList.OptionSelected) -> None:
        message.stop()
        self.dismiss(message.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _RestoreAnchorPickerScreen(ModalScreen[int | None]):
    """Choose one persisted User input without dismissing on outside clicks."""

    CSS = """
    _RestoreAnchorPickerScreen {
        align: center middle;
        padding: 1 2;
    }

    #restore-anchor-panel {
        width: 86%;
        max-width: 84;
        height: 80%;
        max-height: 90%;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #restore-anchor-heading,
    #restore-anchor-notice {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #restore-anchor-heading {
        text-style: bold;
    }

    #restore-anchor-notice {
        color: $text-muted;
    }

    #restore-anchor-options {
        width: 100%;
        height: 1fr;
        min-height: 3;
        overflow-y: auto;
    }

    #restore-anchor-filter {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, anchors: tuple[RestoreAnchor, ...]) -> None:
        super().__init__(id="restore-anchor-picker")
        self._anchors = anchors

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-anchor-panel"):
            yield Static("Restore Session", id="restore-anchor-heading", markup=False)
            yield Static(
                "Persisted User inputs",
                id="restore-anchor-notice",
                markup=False,
            )
            yield Input(placeholder="Filter ID, time or preview", id="restore-anchor-filter")
            yield OptionList(
                *(
                    Option(_restore_anchor_label(anchor), id=str(anchor.anchor_id))
                    for anchor in self._anchors
                ),
                id="restore-anchor-options",
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#restore-anchor-filter", Input).focus()

    @on(Input.Changed, "#restore-anchor-filter")
    def _filter_changed(self, message: Input.Changed) -> None:
        search = message.value.casefold().strip()
        options = self.query_one("#restore-anchor-options", OptionList)
        options.set_options(
            Option(_restore_anchor_label(anchor), id=str(anchor.anchor_id))
            for anchor in self._anchors
            if search in _restore_anchor_label(anchor).casefold()
        )
        if options.option_count:
            options.highlighted = 0

    @on(Key)
    def _filter_key(self, event: Key) -> None:
        if not self.query_one("#restore-anchor-filter", Input).has_focus:
            return
        options = self.query_one("#restore-anchor-options", OptionList)
        if event.key == "down" and options.option_count:
            event.stop()
            event.prevent_default()
            options.focus()
        elif event.key == "enter" and options.option_count:
            event.stop()
            event.prevent_default()
            option = options.get_option_at_index(options.highlighted or 0)
            if isinstance(option.id, str):
                self.dismiss(int(option.id))

    @on(OptionList.OptionSelected, "#restore-anchor-options")
    def _option_selected(self, message: OptionList.OptionSelected) -> None:
        message.stop()
        option_id = message.option_id
        if not isinstance(option_id, str):
            return
        try:
            anchor_id = int(option_id)
        except (TypeError, ValueError):
            return
        self.dismiss(anchor_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _RestoreWaitingScreen(ModalScreen[bool]):
    """Keep the pre-confirmation restore barrier cancellable while it settles."""

    CSS = """
    _RestoreWaitingScreen {
        align: center middle;
        padding: 1 2;
    }

    #restore-waiting-panel {
        width: 70%;
        max-width: 64;
        height: auto;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #restore-waiting-heading,
    #restore-waiting-message {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #restore-waiting-heading {
        text-style: bold;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__(id="restore-waiting")

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-waiting-panel"):
            yield Static("Preparing Session Restore", id="restore-waiting-heading", markup=False)
            yield Static(
                "Waiting for active work to finish.",
                id="restore-waiting-message",
                markup=False,
            )

    def action_cancel(self) -> None:
        self.dismiss(False)


class _RestoreModeScreen(ModalScreen[RestoreMode | None]):
    """Choose the scope of one inspected Session Restore."""

    CSS = """
    _RestoreModeScreen {
        align: center middle;
        padding: 1 2;
    }

    #restore-mode-panel {
        width: 86%;
        max-width: 84;
        height: 90%;
        max-height: 90%;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #restore-mode-details {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }

    #restore-mode-heading,
    #restore-mode-notice {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #restore-mode-heading {
        text-style: bold;
    }

    #restore-mode-notice {
        color: $text-warning;
    }

    #restore-mode-options {
        width: 100%;
        height: 3;
    }

    #restore-mode-impact {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        color: $text-warning;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, plan: RestorePlan) -> None:
        super().__init__(id="restore-mode-picker")
        self._plan = plan
        self._modes = plan.available_modes
        self._has_gap = bool(plan.backup_gaps or plan.integrity_issues)

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-mode-panel"):
            yield Static("Restore scope", id="restore-mode-heading", markup=False)
            with VerticalScroll(id="restore-mode-details"):
                yield Static(
                    _restore_impact_text(self._plan), id="restore-mode-impact", markup=False
                )
                if self._has_gap:
                    yield Static(
                        "File Restore unavailable: the selected range has incomplete backup coverage.",
                        id="restore-mode-notice",
                        markup=False,
                    )
            yield OptionList(
                *(Option(_restore_mode_label(mode), id=mode.value) for mode in self._modes),
                id="restore-mode-options",
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#restore-mode-options", OptionList).focus()

    @on(OptionList.OptionSelected, "#restore-mode-options")
    def _option_selected(self, message: OptionList.OptionSelected) -> None:
        message.stop()
        option_id = message.option_id
        if not isinstance(option_id, str):
            return
        try:
            mode = RestoreMode(option_id)
        except (TypeError, ValueError):
            return
        if mode in self._modes:
            self.dismiss(mode)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _RestoreConfirmationScreen(ModalScreen[bool]):
    """Confirm the entire Session Restore operation once."""

    CSS = """
    _RestoreConfirmationScreen {
        align: center middle;
        padding: 1 2;
    }

    #restore-confirmation-panel {
        width: 86%;
        max-width: 84;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
        overflow-y: hidden;
    }

    #restore-confirmation-heading,
    .restore-confirmation-detail,
    #restore-confirmation-scope {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #restore-confirmation-heading {
        text-style: bold;
    }

    #restore-confirmation-scope {
        color: $text-warning;
    }

    #restore-confirmation-actions {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    #restore-confirmation-actions Button {
        width: 1fr;
        min-width: 0;
        margin: 0;
        height: 3;
    }

    #restore-confirmation-details {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
        Binding("left,up", "focus_cancel", "Cancel", show=False),
        Binding("right,down", "focus_confirm", "Restore", show=False),
    ]

    def __init__(self, plan: RestorePlan, mode: RestoreMode) -> None:
        super().__init__(id="restore-confirmation")
        self._plan = plan
        self._mode = mode

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-confirmation-panel"):
            yield Static("Confirm Session Restore", id="restore-confirmation-heading", markup=False)
            with VerticalScroll(id="restore-confirmation-details"):
                yield Static(
                    _restore_impact_text(self._plan),
                    classes="restore-confirmation-detail",
                    markup=False,
                )
                yield Static(
                    f"Remove: {self._plan.removed_users} User and "
                    f"{self._plan.removed_messages} total messages",
                    classes="restore-confirmation-detail",
                    markup=False,
                )
                yield Static(
                    f"Files: {len(self._plan.targets)} tracked, "
                    f"{self._plan.external_target_count} external",
                    classes="restore-confirmation-detail",
                    markup=False,
                )
                if self._mode is RestoreMode.FILES:
                    scope = "Scope: Conversation Session and eligible File Restore."
                else:
                    scope = "Scope: Conversation Session only; files remain unchanged."
                yield Static(scope, id="restore-confirmation-scope", markup=False)
                yield Static(
                    "Not independently rolled back: Conversation Summary, Long-term Memory, Schedule, "
                    "Exec/MCP effects, Dream, Tool Artifacts, Session Log, and manual edits.",
                    classes="restore-confirmation-detail",
                    markup=False,
                )
                if self._mode is RestoreMode.FILES:
                    yield Static(
                        "File Restore may still overwrite later manual, Exec, or MCP changes "
                        "to tracked files.",
                        classes="restore-confirmation-detail",
                        markup=False,
                    )
            with Horizontal(id="restore-confirmation-actions"):
                yield Button("Cancel", id="restore-confirmation-cancel")
                yield Button("Restore", variant="warning", id="restore-confirmation-approve")

    def on_mount(self) -> None:
        self.query_one("#restore-confirmation-cancel", Button).focus()

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "restore-confirmation-approve")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_focus_cancel(self) -> None:
        self.query_one("#restore-confirmation-cancel", Button).focus()

    def action_focus_confirm(self) -> None:
        self.query_one("#restore-confirmation-approve", Button).focus()


class _RestoreFailureScreen(ModalScreen[bool]):
    """Require acknowledgement for a partially failed File Restore."""

    CSS = """
    _RestoreFailureScreen {
        align: center middle;
        padding: 1 2;
    }

    #restore-failure-panel {
        width: 86%;
        max-width: 84;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $error;
        background: $surface;
        overflow-y: auto;
    }

    #restore-failure-heading,
    .restore-failure-detail {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #restore-failure-heading {
        text-style: bold;
    }

    #restore-failure-actions {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "acknowledge", "Acknowledge", show=False),
        Binding("ctrl+c", "acknowledge", "Acknowledge", show=False, priority=True),
    ]

    def __init__(self, result: RestoreResult) -> None:
        super().__init__(id="restore-failure")
        self._result = result

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-failure-panel"):
            yield Static("File Restore incomplete", id="restore-failure-heading", markup=False)
            for item in self._result.failures:
                yield Static(
                    f"Failed: {item.target}",
                    classes="restore-failure-detail",
                    markup=False,
                )
            for target in self._result.successful_conflicts:
                yield Static(
                    f"Restored conflict: {target}",
                    classes="restore-failure-detail",
                    markup=False,
                )
            with Horizontal(id="restore-failure-actions"):
                yield Button("Acknowledge", id="restore-failure-acknowledge")

    def on_mount(self) -> None:
        self.query_one("#restore-failure-acknowledge", Button).focus()

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    def action_acknowledge(self) -> None:
        self.dismiss(True)


class _SessionSwitchConfirmationScreen(ModalScreen[bool]):
    """Confirm replacing an active foreground Runtime Generation."""

    CSS = """
    _SessionSwitchConfirmationScreen {
        align: center middle;
        padding: 1 2;
    }

    #session-switch-panel {
        width: 80%;
        max-width: 72;
        height: auto;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    #session-switch-heading,
    #session-switch-message {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #session-switch-heading {
        text-style: bold;
    }

    #session-switch-actions {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #session-switch-actions Button {
        margin: 0 1;
        height: 3;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "decline", "Decline", show=False),
        Binding("ctrl+c", "decline", "Decline", show=False, priority=True),
        Binding("left,up", "focus_decline", "Decline", show=False),
        Binding("right,down", "focus_approve", "Approve", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id="session-switch-confirmation")

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="session-switch-panel"):
                yield Static("Switch Conversation Session?", id="session-switch-heading")
                yield Static(
                    "The active foreground run will be abandoned and its pending input discarded.",
                    id="session-switch-message",
                    markup=False,
                )
                with Horizontal(id="session-switch-actions"):
                    yield Button("Decline", id="session-switch-decline")
                    yield Button("Approve", variant="warning", id="session-switch-approve")

    def on_mount(self) -> None:
        self.query_one("#session-switch-decline", Button).focus()

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "session-switch-approve")

    def action_decline(self) -> None:
        self.dismiss(False)

    def action_focus_decline(self) -> None:
        self.query_one("#session-switch-decline", Button).focus()

    def action_focus_approve(self) -> None:
        self.query_one("#session-switch-approve", Button).focus()


class _ConversationRecoveryScreen(ModalScreen[str]):
    """Offer explicit actions when the service cannot reopen the selected Session."""

    CSS = """
    _ConversationRecoveryScreen { align: center middle; }
    #conversation-recovery-panel {
        width: 80%; max-width: 76; height: auto; padding: 1 2;
        border: round $warning; background: $surface;
    }
    #conversation-recovery-panel Static { height: auto; margin-bottom: 1; }
    #conversation-recovery-actions { height: auto; }
    #conversation-recovery-actions Button { margin-right: 1; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="conversation-recovery-panel"):
            yield Static("Conversation Session unavailable")
            yield Static(self._message, markup=False)
            with Horizontal(id="conversation-recovery-actions"):
                yield Button("Retry original Session", id="conversation-recovery-retry")
                yield Button("New Session", id="conversation-recovery-new")

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("new" if event.button.id == "conversation-recovery-new" else "retry")


class _MarkdownStream(Protocol):
    async def write(self, markdown_fragment: str) -> None: ...

    async def stop(self) -> None: ...


class _CoalescedMarkdownStream:
    """Batch adjacent provider deltas into one Textual Markdown refresh."""

    def __init__(
        self,
        stream: _MarkdownStream,
        *,
        content_changed: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._content_changed = content_changed
        self._pending: list[str] = []
        self._flush_task: Task[None] | None = None
        self._flush_error: BaseException | None = None
        self._stopped = False

    def write(self, fragment: str) -> None:
        if self._stopped:
            raise RuntimeError("Markdown stream is already stopped")
        self._pending.append(fragment)
        if self._flush_task is None and self._flush_error is None:
            self._flush_task = create_task(self._flush_after_event_loop_turn())

    async def stop(self) -> None:
        self._stopped = True
        try:
            if self._flush_task is not None:
                await self._flush_task
            if self._pending and self._flush_error is None:
                await self._flush_pending()
        except BaseException as error:
            self._flush_error = error

        try:
            await self._stream.stop()
        except BaseException as stop_error:
            if self._flush_error is not None:
                raise self._flush_error from stop_error
            raise
        if self._flush_error is not None:
            raise self._flush_error

    async def _flush_after_event_loop_turn(self) -> None:
        try:
            await sleep(0)
            while self._pending:
                await self._flush_pending()
        except BaseException as error:
            self._flush_error = error
        finally:
            self._flush_task = None

    async def _flush_pending(self) -> None:
        fragments = self._pending
        self._pending = []
        await self._stream.write("".join(fragments))
        self._content_changed()


def _make_links_visible(state: StateCore) -> None:
    """Render Markdown links as ordinary label-and-URL text."""
    for token in state.tokens:
        if token.type != "inline" or token.children is None:
            continue

        children: list[Token] = []
        link_starts: list[tuple[str, int]] = []
        for child in token.children:
            if child.type == "link_open":
                href = child.attrGet("href")
                link_starts.append((str(href) if href is not None else "", len(children)))
                continue
            if child.type == "link_close":
                if link_starts:
                    href, start = link_starts.pop()
                    label = "".join(
                        item.content
                        for item in children[start:]
                        if item.type in {"text", "code_inline"}
                    )
                    if href and label != href:
                        children.append(Token("text", "", 0, content=f" ({href})"))
                continue
            if child.type == "image":
                source = child.attrGet("src")
                href = str(source) if source is not None else ""
                alt = child.content
                content = alt if not href else f"{alt} ({href})"
                children.append(Token("text", "", 0, content=content))
                continue
            children.append(child)
        token.children = children


def _markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("gfm-like")
    parser.core.ruler.after("linkify", "omni_visible_links", _make_links_visible)
    return parser


@dataclass(frozen=True, slots=True)
class _ScrollAnchor:
    widget: Widget
    relative_position: float
    child_index: int
    fallbacks: tuple[tuple[Widget, float], ...]


class _ConversationDisplay(VerticalScroll):
    """Conversation viewport with bottom-following and historical navigation."""

    FOCUS_ON_CLICK = False

    class Resized(Message):
        def __init__(self, display: _ConversationDisplay, width: int, generation: int) -> None:
            super().__init__()
            self.display = display
            self.width = width
            self.generation = generation

    _following = True
    _new_content = False
    _last_scroll_state: tuple[bool, bool] | None = None
    _historical_anchor: _ScrollAnchor | None = None
    _resize_anchor: _ScrollAnchor | None = None
    _resize_generation = 0
    _resize_seen = False
    _size_suspended = False
    _size_restore_pending = False
    _resize_restore_pending = False
    _restoring_resize = False
    _follow_generation = 0
    _content_restore_generation = 0
    _content_restore_pending = False
    _restoring_content = False
    _layout_restore_generation = 0
    _layout_restore_pending = False
    _layout_restore_anchor: _ScrollAnchor | None = None
    _layout_restore_scheduled_generation: int | None = None
    _layout_restore_following = True
    _layout_restore_new_content = False
    _suspended_following = True
    _suspended_new_content = False
    _suspended_anchor: _ScrollAnchor | None = None

    @property
    def following(self) -> bool:
        """Whether new content should keep the viewport at the bottom."""
        return self._following

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._cancel_follow_latest()
        self._take_over_resize_restore()
        self._take_over_content_restore()
        super()._on_mouse_scroll_up(event)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._cancel_follow_latest()
        self._take_over_resize_restore()
        self._take_over_content_restore()
        super()._on_mouse_scroll_down(event)
        self._sync_after_user_scroll()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        self._cancel_follow_latest()
        self._take_over_resize_restore()
        self._take_over_content_restore()
        super()._on_scroll_to(message)
        self.call_after_refresh(self._sync_after_user_scroll)

    def on_resize(self, event: Resize) -> None:
        self._cancel_follow_latest()
        self._resize_generation += 1
        self._cancel_content_restores()
        if not self._size_restore_pending:
            self._restoring_resize = False
        first_resize = not self._resize_seen
        self._resize_seen = True
        if not self._size_suspended and not self._size_restore_pending:
            self._resize_anchor = None if self.following else self._historical_anchor
            if not first_resize:
                self._resize_restore_pending = True
        self.post_message(self.Resized(self, event.size.width, self._resize_generation))

    def restore_resize_anchor(self, generation: int | None = None) -> None:
        if self._size_suspended:
            return
        if generation is None and not (self._size_restore_pending or self._resize_restore_pending):
            return
        if generation is not None and generation != self._resize_generation:
            return
        current_generation = self._resize_generation
        anchor = self._resize_anchor
        self._restoring_resize = True
        if self.following:
            self.scroll_end(
                animate=False,
                immediate=not self._size_restore_pending,
            )

            def follow_latest(attempt: int = 0) -> None:
                if current_generation != self._resize_generation:
                    return
                self.scroll_end(animate=False, immediate=True)
                if attempt < 2:
                    self.call_after_refresh(follow_latest, attempt + 1)
                    return
                self._finish_resize_restore(emit_state=True)

            self.call_after_refresh(follow_latest)
            return
        if anchor is None:
            self._finish_resize_restore()
            return

        def restore(attempt: int = 0) -> None:
            if current_generation != self._resize_generation or self._resize_anchor is not anchor:
                return
            target = self._resolve_scroll_anchor(anchor)
            if target is None:
                self._finish_resize_restore()
                return
            self.scroll_to(
                y=round(target),
                animate=False,
                immediate=True,
            )
            if attempt < 2:
                self.call_after_refresh(restore, attempt + 1)
                return
            self._finish_resize_restore(emit_state=True)

        self.call_after_refresh(restore)

    def schedule_resize_anchor_retry(self, generation: int | None = None) -> None:
        retry_generation = self._resize_generation if generation is None else generation
        self.set_timer(0.001, partial(self.restore_resize_anchor, retry_generation))

    def navigate(self, key: str) -> None:
        self._cancel_follow_latest()
        self._take_over_resize_restore()
        self._take_over_content_restore()
        if key == "pageup":
            page_height = max(1, self.scrollable_content_region.height)
            self.scroll_to(
                y=self.scroll_y - page_height,
                animate=False,
                immediate=True,
            )
        elif key == "pagedown":
            page_height = max(1, self.scrollable_content_region.height)
            self.scroll_to(
                y=self.scroll_y + page_height,
                animate=False,
                immediate=True,
            )
        elif key == "ctrl+home":
            self.scroll_home(animate=False, immediate=True)
        elif key == "ctrl+end":
            self.scroll_end(animate=False, immediate=True)
        else:
            raise ValueError(f"Unsupported conversation navigation key: {key}")
        self._sync_after_user_scroll()

    def _take_over_resize_restore(self) -> None:
        if self._size_suspended or not (
            self._size_restore_pending or self._resize_restore_pending or self._restoring_resize
        ):
            return
        self._finish_resize_restore()
        self._resize_generation += 1

    def _take_over_content_restore(self) -> None:
        if not (
            self._content_restore_pending or self._restoring_content or self._layout_restore_pending
        ):
            return
        self._cancel_content_restores()

    def _cancel_content_restores(self) -> None:
        self._content_restore_pending = False
        self._restoring_content = False
        self._content_restore_generation += 1
        self._layout_restore_pending = False
        self._layout_restore_anchor = None
        self._layout_restore_generation += 1
        self._layout_restore_scheduled_generation = None

    def _finish_resize_restore(self, *, emit_state: bool = False) -> None:
        self._resize_anchor = None
        self._size_restore_pending = False
        self._resize_restore_pending = False
        self._restoring_resize = False
        if emit_state:
            self._emit_scroll_state()

    def _cancel_follow_latest(self) -> None:
        self._follow_generation += 1

    def _schedule_follow_latest(self) -> None:
        self._follow_generation += 1
        self.call_after_refresh(partial(self._follow_latest, self._follow_generation))

    def content_changed(self) -> None:
        """Follow new content only while the user is at the conversation bottom."""
        if self._size_suspended or self._size_restore_pending:
            if not self._suspended_following:
                self._suspended_new_content = True
            self._emit_scroll_state()
            return
        if self._layout_restore_pending:
            self._new_content = self._layout_restore_new_content
            self._schedule_layout_anchor_restore()
            self._emit_scroll_state()
            return
        if self.following:
            self.scroll_end(animate=False, immediate=True)
            self._following = True
            self._new_content = False
            self._historical_anchor = None
            self._schedule_follow_latest()
        else:
            self._new_content = True
            self._schedule_content_anchor_restore()
        self._emit_scroll_state()

    def layout_changed(self, anchor: Widget) -> None:
        """Keep an explicitly selected widget stable while its layout changes."""
        self._cancel_follow_latest()
        if self._size_suspended or self._size_restore_pending:
            return
        scroll_anchor = self._capture_widget_anchor(anchor)
        if scroll_anchor is None:
            return
        self._content_restore_pending = False
        self._content_restore_generation += 1
        self._layout_restore_anchor = scroll_anchor
        self._layout_restore_following = self._following
        self._layout_restore_new_content = self._new_content
        self._layout_restore_pending = True
        self._layout_restore_generation += 1
        self._schedule_layout_anchor_restore()
        self._emit_scroll_state()

    def _schedule_layout_anchor_restore(self) -> None:
        if not self._layout_restore_pending:
            return
        generation = self._layout_restore_generation
        if self._layout_restore_scheduled_generation == generation:
            return
        self._layout_restore_scheduled_generation = generation
        self.call_after_refresh(partial(self.restore_layout_anchor, generation))

    def restore_layout_anchor(self, generation: int, attempt: int = 0) -> None:
        if generation != self._layout_restore_generation:
            return
        self._layout_restore_scheduled_generation = None
        anchor = self._layout_restore_anchor
        if anchor is None:
            self._layout_restore_pending = False
            self._following = self._layout_restore_following
            self._new_content = self._layout_restore_new_content
            return
        if self._size_suspended:
            return
        target = self._resolve_scroll_anchor(anchor)
        if target is None:
            self._layout_restore_pending = False
            self._layout_restore_anchor = None
            self._following = self._layout_restore_following
            self._new_content = self._layout_restore_new_content
            self._historical_anchor = self._capture_scroll_anchor()
            return
        self._restoring_content = True
        self.scroll_to(y=target, animate=False, immediate=True)
        self._restoring_content = False
        if attempt < 2:
            self.call_after_refresh(partial(self.restore_layout_anchor, generation, attempt + 1))
            return
        self._layout_restore_pending = False
        self._layout_restore_anchor = None
        self._layout_restore_scheduled_generation = None
        self._following = self._layout_restore_following and self.is_vertical_scroll_end
        self._new_content = self._layout_restore_new_content or (
            self._layout_restore_following and not self._following
        )
        if not self._following:
            self._historical_anchor = self._capture_scroll_anchor()
        else:
            self._historical_anchor = None
        self._emit_scroll_state()

    def _schedule_content_anchor_restore(self) -> None:
        if self._resize_restore_pending or self._size_restore_pending or self._restoring_resize:
            return
        if self._historical_anchor is None:
            self._historical_anchor = self._capture_scroll_anchor()
        if self._historical_anchor is None or self._content_restore_pending:
            return
        self._content_restore_pending = True
        generation = self._content_restore_generation
        self.call_after_refresh(partial(self.restore_content_anchor, generation))

    def restore_content_anchor(self, generation: int | None = None, attempt: int = 0) -> None:
        if self._size_suspended or self._restoring_resize or self.following:
            self._content_restore_pending = False
            return
        if generation is not None and generation != self._content_restore_generation:
            return
        anchor = self._historical_anchor
        if anchor is None:
            self._content_restore_pending = False
            return
        target = self._resolve_scroll_anchor(anchor)
        if target is None:
            self._content_restore_pending = False
            return

        current_generation = self._content_restore_generation
        self._restoring_content = True
        self.scroll_to(y=target, animate=False, immediate=True)
        self._restoring_content = False
        if attempt < 2:
            self.call_after_refresh(
                partial(self.restore_content_anchor, current_generation, attempt + 1)
            )
            return
        self._content_restore_pending = False
        self._historical_anchor = self._capture_scroll_anchor()
        self._emit_scroll_state()

    def reset_to_latest(self) -> None:
        """Start a replaced Session at its latest persisted content."""
        if self._size_suspended:
            self._suspended_following = True
            self._suspended_new_content = False
            self._suspended_anchor = None
        self._size_restore_pending = False
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._cancel_content_restores()
        self._following = True
        self._new_content = False
        self._historical_anchor = None
        self._resize_anchor = None
        self._resize_generation += 1
        self.scroll_end(animate=False, immediate=True)
        self._following = True
        self._schedule_follow_latest()
        self._emit_scroll_state()

    def _follow_latest(self, generation: int, attempt: int = 0) -> None:
        if generation != self._follow_generation or not self.following:
            return
        self.scroll_end(animate=False, immediate=True)
        self._following = True
        self._new_content = False
        self._historical_anchor = None
        self._emit_scroll_state()
        if attempt < 2:
            self.call_after_refresh(partial(self._follow_latest, generation, attempt + 1))

    def _sync_after_user_scroll(self) -> None:
        if self._size_suspended or self._restoring_resize or self._restoring_content:
            return
        if self._size_restore_pending:
            self._size_restore_pending = False
            self._resize_anchor = None
            self._resize_generation += 1
        elif self._resize_restore_pending:
            self._resize_restore_pending = False
            self._resize_anchor = None
            self._resize_generation += 1
        self._following = self.is_vertical_scroll_end
        if self._following:
            self._new_content = False
            self._historical_anchor = None
        else:
            self._historical_anchor = self._capture_scroll_anchor()
        self._emit_scroll_state()

    def suspend_for_size(self) -> None:
        """Freeze the visible scroll state while the normal layout is hidden."""
        if self._size_suspended:
            return
        self._size_suspended = True
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._suspended_following = self._following
        self._suspended_new_content = self._new_content
        self._suspended_anchor = (
            None if self._following else (self._resize_anchor or self._historical_anchor)
        )
        self._resize_anchor = self._suspended_anchor
        self._resize_generation += 1

    def resume_from_size(self) -> None:
        """Restore the scroll state captured before the layout was hidden."""
        if not self._size_suspended:
            return
        self._size_suspended = False
        self._size_restore_pending = True
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._following = self._suspended_following
        self._new_content = self._suspended_new_content
        self._historical_anchor = self._suspended_anchor
        self._resize_anchor = None if self._following else self._suspended_anchor
        self._resize_generation += 1
        self._emit_scroll_state()

    def _capture_scroll_anchor(self) -> _ScrollAnchor | None:
        scroll_top = round(self.scroll_y)
        candidates: list[tuple[int, Widget, float, float]] = []
        for child_index, child in enumerate(self.children):
            widget = self._anchor_widget(child)
            metrics = self._anchor_metrics(widget)
            if metrics is None:
                continue
            y, height = metrics
            candidates.append((child_index, widget, y, height))
        if not candidates:
            return None
        selected = next(
            (candidate for candidate in candidates if candidate[2] + candidate[3] > scroll_top),
            candidates[-1],
        )
        child_index, widget, y, height = selected
        relative_position = (scroll_top - y) / height
        selected_index = next(
            index for index, candidate in enumerate(candidates) if candidate[1] is widget
        )
        fallback_candidates = (
            *candidates[selected_index + 1 :],
            *candidates[:selected_index][::-1],
        )
        fallbacks = tuple(
            (candidate_widget, (scroll_top - candidate_y) / candidate_height)
            for _, candidate_widget, candidate_y, candidate_height in fallback_candidates
        )
        return _ScrollAnchor(widget, relative_position, child_index, fallbacks)

    @staticmethod
    def _anchor_widget(child: Widget) -> Widget:
        if child.has_class("agent-run-activity-group"):
            with suppress(NoMatches):
                return child.query_one(".agent-run-activity-heading", _ActivityGroupHeading)
        return child

    def _capture_widget_anchor(self, widget: Widget) -> _ScrollAnchor | None:
        top_level = widget
        while top_level.parent is not self:
            parent = top_level.parent
            if not isinstance(parent, Widget):
                return None
            top_level = parent
        try:
            child_index = list(self.children).index(top_level)
        except ValueError:
            return None
        metrics = self._anchor_metrics(widget)
        if metrics is None:
            return None
        y, height = metrics
        base = self._historical_anchor or self._capture_scroll_anchor()
        fallbacks = () if base is None else ((base.widget, base.relative_position), *base.fallbacks)
        return _ScrollAnchor(
            widget,
            (round(self.scroll_y) - y) / height,
            child_index,
            tuple(
                (candidate, relative)
                for candidate, relative in fallbacks
                if candidate is not widget
            ),
        )

    def _anchor_metrics(self, widget: Widget) -> tuple[float, float] | None:
        y = 0.0
        current: Widget | None = widget
        while current is not None and current is not self:
            if not current.display:
                return None
            region = current.virtual_region
            if region.height <= 0:
                return None
            y += region.y
            parent = current.parent
            current = parent if isinstance(parent, Widget) else None
        if current is not self:
            return None
        return y, float(widget.virtual_region.height)

    def _resolve_scroll_anchor(self, anchor: _ScrollAnchor) -> float | None:
        targets = ((anchor.widget, anchor.relative_position), *anchor.fallbacks)
        for widget, relative_position in targets:
            metrics = self._anchor_metrics(widget)
            if metrics is not None:
                y, height = metrics
                return y + relative_position * height

        children = list(self.children)
        if not children:
            return None
        child_index = min(anchor.child_index, len(children) - 1)
        widget = children[child_index]
        metrics = self._anchor_metrics(widget)
        if metrics is None:
            return None
        y, height = metrics
        return y + anchor.relative_position * height

    def _emit_scroll_state(self) -> None:
        at_bottom = self.is_vertical_scroll_end
        if at_bottom and self.following:
            self._new_content = False
        state = (self.following, self._new_content)
        if state == self._last_scroll_state:
            return
        self._last_scroll_state = state
        try:
            self.app.query_one("#new-content", Static).display = state[1]
        except (NoMatches, NoScreen, ScreenStackError):
            pass


class _SizeInsufficientScreen(ModalScreen[None]):
    """Block all interaction until the terminal can render the application."""

    CSS = """
    _SizeInsufficientScreen {
        align: center middle;
        background: $background;
    }

    #size-insufficient-modal {
        width: 100%;
        height: 100%;
        padding: 1 2;
        content-align: center middle;
        text-align: center;
        background: $background;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Terminal window is too small. Resize to continue.",
            id="size-insufficient-modal",
            markup=False,
        )

    async def _on_key(self, event: Key) -> None:
        event.stop()
        event.prevent_default()


class _ToolConfirmationScreen(ModalScreen[ConfirmationDecision]):
    """Present one normalized Tool Confirmation without leaving the conversation."""

    CSS = """
    _ToolConfirmationScreen {
        align: center middle;
        padding: 1 2;
    }

    _ToolConfirmationScreen Center {
        width: 100%;
        height: 100%;
    }

    #confirmation-panel {
        width: 80%;
        max-width: 72;
        height: 90%;
        max-height: 90%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
        overflow-y: hidden;
    }

    #confirmation-details-scroll {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }

    #confirmation-heading {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
    }

    #confirmation-tool,
    #confirmation-reason,
    .confirmation-details,
    .confirmation-warning {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .confirmation-warning {
        color: $text-warning;
    }

    #confirmation-actions {
        width: 100%;
        height: 3;
        align: center middle;
    }

    #confirmation-actions Button {
        width: 1fr;
        min-width: 0;
        margin: 0;
        height: 3;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "decline", "Decline", show=False),
        Binding("ctrl+c", "decline", "Decline", show=False, priority=True),
        Binding("left,up", "focus_decline", "Decline", show=False),
        Binding("right,down", "focus_approve", "Approve", show=False),
    ]

    def __init__(self, request: ConfirmationRequestView) -> None:
        super().__init__(id=f"confirmation-{request.confirmation_id.hex}")
        self._request = request

    def compose(self) -> ComposeResult:
        is_background = getattr(self._request, "origin", "foreground") == "background"
        heading = "Background Tool Confirmation" if is_background else "Tool Confirmation"
        with Center():
            with Vertical(id="confirmation-panel"):
                yield Static(heading, id="confirmation-heading", markup=False)
                with VerticalScroll(id="confirmation-details-scroll"):
                    if is_background:
                        job_id = getattr(self._request, "job_id", "")
                        title = getattr(self._request, "title", "")
                        yield Static(
                            f"Source: {job_id} + {title}",
                            id="confirmation-source",
                            markup=False,
                        )
                    yield Static(
                        f"Tool: {_friendly_name(self._request.tool_name, fallback='Tool')}",
                        id="confirmation-tool",
                        markup=False,
                    )
                    reason = self._request.reason or self._request.summary
                    yield Static(f"Reason: {reason}", id="confirmation-reason", markup=False)
                    for warning in self._request.warnings:
                        yield Static(
                            f"Warning: {warning}",
                            markup=False,
                            classes="confirmation-warning",
                        )
                    for detail in _confirmation_detail_lines(self._request):
                        yield Static(detail, markup=False, classes="confirmation-details")
                with Horizontal(id="confirmation-actions"):
                    yield Button("Decline", id="confirmation-decline")
                    yield Button("Approve", variant="success", id="confirmation-approve")

    def on_mount(self) -> None:
        self.query_one("#confirmation-decline", Button).focus()

    def restore_after_size(self) -> None:
        """Reveal the confirmation heading after a constrained background layout."""
        self.query_one("#confirmation-details-scroll", VerticalScroll).scroll_home(
            animate=False,
            immediate=True,
        )

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        decision: ConfirmationDecision = (
            "approved" if event.button.id == "confirmation-approve" else "declined"
        )
        self.dismiss(decision)

    def action_decline(self) -> None:
        self.dismiss("declined")

    def action_focus_decline(self) -> None:
        self.query_one("#confirmation-decline", Button).focus()

    def action_focus_approve(self) -> None:
        self.query_one("#confirmation-approve", Button).focus()


class _MessageBusRunProjection:
    """Project one consumed MessageBus foreground run into the Terminal UI."""

    def __init__(
        self,
        app: TerminalConversationApp,
        turn_id: UUID,
        *,
        display: _ConversationDisplay | None = None,
    ) -> None:
        self._app = app
        if display is None:
            display = app._conversation_display
        if display is None:
            display = app.query_one("#conversation-display", _ConversationDisplay)
        self._display = display
        self.turn_id = turn_id
        self._assistant: Markdown | None = None
        self._response_stream: _CoalescedMarkdownStream | None = None
        self._response_fragments: list[str] = []
        self._response_reopen_allowed = False
        self._reasoning: Markdown | None = None
        self._reasoning_stream: _CoalescedMarkdownStream | None = None
        self._activity_group: _ActivityGroupState | None = None
        self._tool_rows: dict[str, _ToolRowState] = {}
        self._terminal_content = ""
        self._terminal_status: str | None = None
        self._outcome: _TerminalOutcome | None = None
        self._terminal_seen = False
        self._started_at: float | None = None
        self._elapsed = 0.0
        self._timer: Timer | None = None

    @property
    def terminal_seen(self) -> bool:
        return self._terminal_seen

    def start(self) -> None:
        """Start elapsed timing when AgentRunExecutor consumes the inbound message."""
        self._start_timing()

    async def consume(self, outbound: OutboundMessage) -> None:
        if self._terminal_seen:
            return
        self._start_timing()
        marker = self._sparse_marker(outbound.metadata)
        if outbound.type == "model_reasoning":
            if marker == "_stream_delta":
                await self._append_reasoning(outbound.content)
            elif marker == "_stream_end":
                reasoning_stream_active = self._reasoning_stream is not None
                await self._stop_reasoning_stream()
                if reasoning_stream_active:
                    self._response_reopen_allowed = True
            else:
                await self._fail_sparse_protocol()
            return
        if outbound.type == "model_response":
            if marker == "_stream_delta":
                await self._append_response(outbound.content)
            elif marker == "_stream_end":
                await self._stop_response_stream()
            elif marker == "_streamed":
                await self._finish_terminal("completed", "")
            else:
                await self._fail_sparse_protocol()
            return
        if outbound.type == "tool_call":
            if marker is not None or any(key in outbound.metadata for key in _SPARSE_MARKERS):
                await self._fail_sparse_protocol()
                return
            tool_call_id = outbound.metadata.get("tool_call_id")
            if "status" in outbound.metadata:
                status = outbound.metadata["status"]
                if (
                    not isinstance(tool_call_id, str)
                    or not isinstance(status, str)
                    or status not in {"success", "error", "refused"}
                    or set(outbound.metadata) != {"tool_call_id", "status"}
                ):
                    await self._fail_sparse_protocol()
                    return
                tool_row = self._tool_rows.get(tool_call_id)
                if tool_row is None or tool_row.tool_name != outbound.content:
                    await self._fail_sparse_protocol()
                    return
                if tool_row.status == "running":
                    tool_row.status = cast(_ToolRowStatus, status)
                    tool_row.widget.update(
                        _tool_row_content(tool_row.status, tool_row.tool_name, "")
                    )
                    self._app._scroll_to_latest(self._display)
                return
            arguments = outbound.metadata.get("arguments")
            if not isinstance(tool_call_id, str) or not isinstance(arguments, str):
                await self._fail_sparse_protocol()
                return
            await self._stop_response_stream()
            if self._assistant is not None or self._response_fragments:
                await self._move_response_to_activity("".join(self._response_fragments))
            group = await self._ensure_activity_group()
            if tool_call_id not in self._tool_rows:
                row = await self._app._mount_tool_message(
                    outbound.content,
                    "running",
                    "",
                    self._display,
                    parent=group.content,
                    raw_arguments=arguments,
                )
                self._tool_rows[tool_call_id] = _ToolRowState(row, outbound.content, "running")
                self._app._scroll_to_latest(self._display)
            return
        if outbound.type == "system_control" and marker == "_streamed":
            finish_reason = outbound.metadata.get("finish_reason")
            if finish_reason not in {"cancelled", "failed", "max_iterations"}:
                await self._fail_sparse_protocol()
                return
            outcome: _TerminalOutcome = "cancelled" if finish_reason == "cancelled" else "failed"
            self._terminal_status = (
                "Turn cancelled." if outcome == "cancelled" else outbound.content
            )
            await self._finish_terminal(outcome, outbound.content)
            return
        await self._fail_sparse_protocol()

    @staticmethod
    def _sparse_marker(metadata: Mapping[str, object]) -> str | None:
        present = [key for key in _SPARSE_MARKERS if key in metadata]
        if len(present) != 1 or metadata[present[0]] is not True:
            return None
        return present[0]

    async def _fail_sparse_protocol(self) -> None:
        self._terminal_status = "Turn failed."
        await self._finish_terminal("failed", "")

    async def close(self) -> None:
        await self._stop_response_stream()
        await self._stop_reasoning_stream()
        self.stop()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    async def _append_response(self, fragment: str) -> None:
        if self._assistant is not None and self._response_stream is None:
            if not self._response_reopen_allowed:
                return
            self._response_stream = _CoalescedMarkdownStream(
                Markdown.get_stream(self._assistant),
                content_changed=partial(self._app._scroll_to_latest, self._display),
            )
        self._response_reopen_allowed = False
        self._response_fragments.append(fragment)
        if self._assistant is None:
            self._assistant = await self._app._mount_assistant(display=self._display)
            self._response_stream = _CoalescedMarkdownStream(
                Markdown.get_stream(self._assistant),
                content_changed=partial(self._app._scroll_to_latest, self._display),
            )
        assert self._response_stream is not None
        self._response_stream.write(fragment)

    async def _append_reasoning(self, fragment: str) -> None:
        group = await self._ensure_activity_group()
        if self._reasoning is None:
            self._reasoning = await self._app._mount_assistant(
                display=self._display,
                parent=group.content,
            )
            self._reasoning_stream = _CoalescedMarkdownStream(
                Markdown.get_stream(self._reasoning),
                content_changed=partial(self._app._scroll_to_latest, self._display),
            )
        assert self._reasoning_stream is not None
        self._reasoning_stream.write(fragment)

    async def _stop_response_stream(self) -> None:
        self._response_reopen_allowed = False
        stream = self._response_stream
        self._response_stream = None
        if stream is not None:
            await stream.stop()

    async def _stop_reasoning_stream(self) -> None:
        stream = self._reasoning_stream
        self._reasoning_stream = None
        if stream is not None:
            await stream.stop()
        self._reasoning = None

    async def _finish_terminal(self, outcome: _TerminalOutcome, content: str) -> None:
        if self._terminal_seen:
            return
        self._terminal_seen = True
        self._terminal_content = content or "".join(self._response_fragments)
        self._outcome = outcome
        if self._started_at is not None:
            self._elapsed = max(0.0, self._app._monotonic() - self._started_at)
        if self._activity_group is not None:
            self._activity_group.elapsed = self._elapsed
        self.stop()
        await self._stop_response_stream()
        await self._stop_reasoning_stream()
        await self._reconcile_terminal()

    async def _reconcile_terminal(self) -> None:
        if self._app._closing or self._app._presentation_quiesced:
            return
        for row in self._tool_rows.values():
            if row.status == "running":
                row.status = "cancelled" if self._outcome == "cancelled" else "unknown"
                row.widget.update(_tool_row_content(row.status, row.tool_name, ""))
        if self._outcome == "completed":
            if self._terminal_content:
                if self._assistant is None:
                    self._assistant = await self._app._mount_assistant(
                        self._terminal_content,
                        display=self._display,
                    )
                elif self._terminal_content != "".join(self._response_fragments):
                    await self._assistant.update(self._terminal_content)
                self._app._scroll_to_latest(self._display)
            else:
                if self._assistant is not None:
                    await self._remove_assistant(self._assistant)
                    self._assistant = None
                await self._app._mount_status("Completed with no response.", display=self._display)
        else:
            if self._assistant is not None or self._response_fragments:
                await self._move_response_to_activity("".join(self._response_fragments))
            reason = self._terminal_status or (
                "Turn cancelled." if self._outcome == "cancelled" else "Turn failed."
            )
            await self._app._mount_status(reason, display=self._display)
        if self._activity_group is not None:
            self._set_activity_group_terminal(self._outcome or "failed")
            self._app._scroll_to_latest(self._display)

    async def _move_response_to_activity(self, content: str) -> None:
        if not content:
            if self._assistant is not None:
                await self._remove_assistant(self._assistant)
                self._assistant = None
            self._response_fragments.clear()
            return
        group = await self._ensure_activity_group()
        assistant = self._assistant
        if assistant is None:
            await self._app._mount_assistant(content, display=self._display, parent=group.content)
        else:
            await assistant.update(content)
            row = assistant.parent
            if not isinstance(row, Widget):
                raise RuntimeError("Assistant Markdown is not mounted in a row")
            self._app._reparent_mounted_widget(row, group.content)
        self._assistant = None
        self._response_fragments.clear()
        self._app._scroll_to_latest(self._display)

    async def _remove_assistant(self, assistant: Markdown) -> None:
        parent = assistant.parent
        if isinstance(parent, Widget):
            await parent.remove()
        self._app._scroll_to_latest(self._display)

    async def _ensure_activity_group(self) -> _ActivityGroupState:
        if self._activity_group is not None:
            return self._activity_group
        self._activity_group = await self._app._mount_activity_group(
            self._display,
            expanded=True,
            toggleable=False,
            elapsed=self._elapsed,
        )
        return self._activity_group

    def _start_timing(self) -> None:
        if self._started_at is not None:
            return
        self._started_at = self._app._monotonic()
        self._timer = self._app.set_interval(1.0, self._refresh_elapsed, name="agent-run-duration")

    def _refresh_elapsed(self) -> None:
        if self._started_at is None or self._outcome is not None:
            return
        self._elapsed = max(0.0, self._app._monotonic() - self._started_at)
        if self._activity_group is not None:
            self._activity_group.elapsed = self._elapsed
            self._activity_group.heading.update(
                _activity_group_heading_text(
                    expanded=self._activity_group.expanded,
                    elapsed=self._elapsed,
                )
            )

    def _set_activity_group_terminal(self, outcome: _TerminalOutcome) -> None:
        group = self._activity_group
        if group is None:
            return
        group.toggleable = True
        group.outcome = outcome
        group.heading.can_focus = True
        group.heading.set_class(False, "-running")
        group.heading.set_class(True, f"-{outcome}")
        group.expanded = outcome != "completed"
        group.content.display = group.expanded
        group.heading.update(
            _activity_group_heading_text(
                expanded=group.expanded, elapsed=group.elapsed, outcome=outcome
            )
        )


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
            _ConversationInput(id="conversation-input", placeholder="Message Omni"),
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

    async def _stop_outbound_worker(self) -> None:
        worker = self._outbound_worker
        self._outbound_worker = None
        if worker is None:
            return
        worker.cancel()
        with suppress(WorkerError, CancelledError):
            await worker.wait()

    async def quiesce_for_rebind(self) -> None:
        """Stop presentation work before the owning composition driver switches generations."""
        if self._presentation_quiesced:
            return
        self._presentation_quiesced = True
        confirmation_result = self._confirmation_result
        if confirmation_result is not None and not confirmation_result.done():
            confirmation_result.cancel()
        session_switch_result = self._session_switch_result
        if session_switch_result is not None and not session_switch_result.done():
            session_switch_result.cancel()
        permission_warning_result = self._permission_warning_result
        if permission_warning_result is not None and not permission_warning_result.done():
            if isinstance(self.screen, _FullAccessWarningScreen):
                await self.screen.dismiss(False)
            else:
                permission_warning_result.cancel()
        if self._confirmation_coordinator is None:
            self._unbind_confirmation_callback(self._control)
        self._unbind_bus_callback(self._bus)
        await self._stop_outbound_worker()
        await self._clear_generation_projection()

    async def _clear_generation_projection(self) -> None:
        self._pending_inputs.clear()
        self._bus_snapshot = ()
        active_projection = self._active_run_projection
        if active_projection is not None:
            await active_projection.close()
        for run in self._consumed_runs:
            await run.projection.close()
        self._consumed_runs.clear()
        self._run_ready = Event()
        self._cancel_requested_turn = None
        self._active_run_projection = None
        self._active_confirmation_id = None
        self._active_confirmation_token = None
        self._completion_dismissed_text = None
        self._permission_current_level = None
        self._permission_warning_result = None
        with suppress(NoMatches, NoScreen, ScreenStackError):
            self.query_one("#permission-selector", _PermissionSelector).display = False
            self.query_one("#conversation-input", _ConversationInput).display = True
        confirmation_result = self._confirmation_result
        self._confirmation_result = None
        if confirmation_result is not None and not confirmation_result.done():
            confirmation_result.cancel()
        if self._confirmation_coordinator is None and isinstance(self.screen, _ToolConfirmationScreen):
            self.screen.dismiss("declined")
        self._set_working(False)
        self._refresh_pending_queue()
        with suppress(NoMatches, NoScreen, ScreenStackError):
            input_area = self.query_one("#conversation-input", _ConversationInput)
            input_area.active_turn_token = None
            input_area.read_only = self._restore_workflow_active
            input_area.text = ""
        self._hide_command_completion()
        display = self._conversation_display
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.remove_children()

    async def rebind_agent_loop(
        self,
        *,
        control: TerminalAgentRunExecutorControl,
        skill_metadata: tuple[SkillMetadata, ...],
        session_projection: ForegroundConversationProjection,
    ) -> None:
        """Replace generation presentation state without owning business lifecycle."""
        if self._closing and not self._presentation_quiesced:
            raise RuntimeError("Terminal Conversation is closing")
        if not isinstance(session_projection, ForegroundConversationProjection):
            raise TypeError("Terminal Conversation requires a session projection")
        if not self._presentation_quiesced:
            await self.quiesce_for_rebind()

        target_skill_metadata = tuple(skill_metadata)
        try:
            # Switch generation-bound control before the first DOM await so observers
            # cannot see the new Runtime generation paired with the old control.
            self._control = control
            self._skill_metadata = target_skill_metadata
            await self._replace_display_from_projection(session_projection)
            if self._confirmation_coordinator is None:
                self._bind_confirmation_callback(self._control, self._bus)
            self._bind_bus_callback(self._bus)
            self._bus_snapshot = await self._bus.inbound_snapshot()
            self._closing = False
            self._outbound_worker = self.run_worker(
                self._consume_outbound(),
                name="conversation-outbound",
                group="conversation-outbound",
                exclusive=True,
                exit_on_error=False,
            )
            self._presentation_quiesced = False
            self._schedule_status_refresh()
        except BaseException:
            self._closing = True
            self._presentation_quiesced = True
            if self._confirmation_coordinator is None:
                self._unbind_confirmation_callback(control)
            self._unbind_bus_callback(self._bus)
            await self._stop_outbound_worker()
            with suppress(Exception):
                await self._clear_generation_projection()
            raise

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
            commit = await self._management_dispatcher.restore_commit(plan, mode)
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


_FRIENDLY_INITIALISMS = {"cwd": "CWD", "id": "ID", "url": "URL"}


def _friendly_name(name: str, *, fallback: str) -> str:
    words = name.replace("_", " ").split()
    return (
        " ".join(
            _FRIENDLY_INITIALISMS.get(word.casefold(), word[:1].upper() + word[1:])
            for word in words
        )
        or fallback
    )


def _completion_candidates(
    text: str,
    skill_metadata: tuple[SkillMetadata, ...],
    *,
    management_command_tokens: tuple[str, ...] | None = None,
) -> tuple[_CompletionCandidate, ...]:
    if not text.startswith("/") or any(character.isspace() for character in text):
        return ()
    management = tuple(
        _CompletionCandidate(
            kind=_CompletionCandidateKind.MANAGEMENT,
            token=command.token,
            description=command.description,
            insert_text=command.token,
        )
        for command in MANAGEMENT_COMMANDS
        if command.token.startswith(text)
        and (management_command_tokens is None or command.token in management_command_tokens)
    )
    skills = tuple(
        _CompletionCandidate(
            kind=_CompletionCandidateKind.SKILL,
            token=f"/{metadata.name}",
            description=metadata.description,
            insert_text=f"/{metadata.name} ",
        )
        for metadata in skill_metadata
        if f"/{metadata.name}".startswith(text)
    )
    return management + skills


def _queue_excerpt(value: str, width: int) -> str:
    if cell_len(value) <= width:
        return value
    if width <= 1:
        return "…"
    return f"{set_cell_size(value, width - 1).rstrip()}…"


def _status_view_text(status: RuntimeStatus) -> str:
    values = status.to_dict()
    groups = (
        ("Runtime", ("version", "uptime_seconds")),
        ("Model", ("chat_model", "chat_reasoning_effort")),
        (
            "Context budget",
            (
                "context_window",
                "max_output",
                "available_context",
                "compact_ratio",
                "compact_context_window",
                "projected_next_request_tokens",
                "projection_source",
                "input_budget_used_percent",
            ),
        ),
        ("Permission", ("configured_permission_level", "current_permission_level")),
        ("Session", ("session_message_count", "last_compacted")),
        ("Cumulative usage", ("cumulative_usage",)),
        ("Schedule", ("schedule",)),
    )
    lines: list[str] = []
    for title, keys in groups:
        present = [key for key in keys if key in values]
        if not present:
            continue
        if lines:
            lines.append("")
        lines.append(title)
        for key in present:
            value = values[key]
            if key == "input_budget_used_percent":
                filled = round(min(100.0, status.input_budget_used_percent) / 5)
                value = f"{value}% [{'#' * filled}{'.' * (20 - filled)}]"
            elif isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _session_picker_label(session: SessionListingEntry) -> str:
    local_updated_at = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
    noun = "message" if session.message_count == 1 else "messages"
    return f"{session.title} | {local_updated_at} | {session.message_count} {noun}"


def _restore_anchor_label(anchor: RestoreAnchor) -> str:
    try:
        local_timestamp = (
            datetime.fromisoformat(anchor.timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
        )
    except (TypeError, ValueError):
        local_timestamp = anchor.timestamp
    return f"{anchor.anchor_id}. {local_timestamp} | {format_restore_preview(anchor.content)}"


def _restore_mode_label(mode: RestoreMode) -> str:
    if mode is RestoreMode.CONVERSATION_ONLY:
        return "Conversation only (default)"
    return "Conversation + files"


def _restore_impact_text(plan: RestorePlan) -> str:
    text = (
        f"Impact: {plan.removed_messages} messages removed; "
        f"{len(plan.targets)} file targets ({plan.external_target_count} external)."
    )
    if plan.backup_gaps or plan.integrity_issues:
        text += " File Restore unavailable: backup coverage is incomplete."
    text += " Memory, Schedule and Exec/MCP effects cannot be restored."
    return text


def _persisted_role_and_content(message: Mapping[str, object]) -> tuple[str, str]:
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant", "tool"}:
        raise TypeError("Unsupported persisted Session message role")
    if not isinstance(content, str):
        raise TypeError("Persisted Session message content must be a string")
    if role == "assistant" and message.get("status") not in {
        "completed",
        "interrupted",
        "error",
    }:
        raise TypeError("Persisted Assistant message is malformed")
    if role == "tool" and (
        not isinstance(message.get("name"), str)
        or message.get("status") not in {"success", "error", "refused"}
    ):
        raise TypeError("Persisted Tool message is malformed")
    return role, content


def _persisted_assistant_status(message: Mapping[str, object]) -> str | None:
    status = message.get("status")
    if status == "interrupted":
        return "Turn cancelled."
    if status != "error":
        return None
    error = message.get("error")
    if isinstance(error, dict):
        detail = error.get("message")
        if isinstance(detail, str) and detail:
            return detail
    return "Turn failed."


def _persisted_message_partitions(
    messages: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Split persisted messages into user-owned historical run candidates."""
    partitions: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    seen_user = False
    for message in messages:
        if message.get("role") == "user":
            if current:
                partitions.append(tuple(current))
            current = [message]
            seen_user = True
        elif seen_user:
            current.append(message)
        else:
            partitions.append((message,))
    if current:
        partitions.append(tuple(current))
    return tuple(partitions)


def _classify_historical_partition(
    partition: Sequence[Mapping[str, object]],
) -> _HistoricalRunProjection | None:
    """Infer one tolerant historical Agent Run projection from persisted messages."""
    if not partition or partition[0].get("role") != "user":
        return None

    projected: list[_PersistedMessageProjection] = []
    timestamps: list[datetime | None] = []
    for message in partition:
        try:
            role, content = _persisted_role_and_content(message)
        except (TypeError, ValueError):
            return None
        if role == "assistant" and not isinstance(message.get("tool_calls"), list):
            return None
        projected.append(_PersistedMessageProjection(message, role, content))
        timestamps.append(_persisted_message_timestamp(message))

    declared_tool_call_ids: set[str] = set()
    completed_tool_call_ids: set[str] = set()
    for item in projected:
        if item.role == "assistant":
            for tool_call in cast(list[object], item.message["tool_calls"]):
                if not isinstance(tool_call, dict) or not isinstance(tool_call.get("id"), str):
                    return None
                tool_call_id = cast(str, tool_call["id"])
                if tool_call_id in declared_tool_call_ids:
                    return None
                declared_tool_call_ids.add(tool_call_id)
            continue
        if item.role != "tool":
            continue
        result_tool_call_id = item.message.get("tool_call_id")
        if (
            not isinstance(result_tool_call_id, str)
            or result_tool_call_id not in declared_tool_call_ids
            or result_tool_call_id in completed_tool_call_ids
        ):
            return None
        completed_tool_call_ids.add(result_tool_call_id)

    user_timestamp = timestamps[0]
    if user_timestamp is None:
        return None

    terminal_index: int | None = None
    terminal_kind: _TerminalOutcome | None = None
    for index, item in enumerate(projected[1:], start=1):
        if item.role != "assistant":
            continue
        status = item.message.get("status")
        if status == "completed" and not cast(list[object], item.message["tool_calls"]):
            terminal_index = index
            terminal_kind = "completed"
        elif status == "interrupted":
            terminal_index = index
            terminal_kind = "cancelled"
        elif status == "error":
            terminal_index = index
            terminal_kind = "failed"

    final = (
        projected[terminal_index]
        if terminal_kind == "completed" and terminal_index is not None
        else None
    )
    activity_items = projected[1:]
    if final is not None and terminal_index is not None:
        activity_items = [
            item for index, item in enumerate(projected[1:], start=1) if index != terminal_index
        ]
    activity = tuple(
        item
        for item in activity_items
        if item.role == "tool" or (item.role == "assistant" and bool(item.content.strip()))
    )

    terminal_status: str | None = None
    endpoint: datetime | None
    if terminal_kind == "completed" and final is not None:
        endpoint = timestamps[terminal_index] if terminal_index is not None else None
        if endpoint is None:
            return None
        if not final.content.strip():
            terminal_status = "Completed with no response."
    elif terminal_kind in {"cancelled", "failed"} and terminal_index is not None:
        endpoint = timestamps[terminal_index]
        if endpoint is None:
            return None
        terminal_status = _persisted_assistant_status(projected[terminal_index].message)
    else:
        endpoint = timestamps[-1]
        if endpoint is None:
            return None

    if not activity and final is None and terminal_status is None:
        return None
    elapsed = max(0.0, (endpoint - user_timestamp).total_seconds())
    return _HistoricalRunProjection(
        activity=activity,
        final=final,
        terminal_status=terminal_status,
        outcome=terminal_kind,
        elapsed=elapsed,
    )


def _persisted_message_timestamp(message: Mapping[str, object]) -> datetime | None:
    timestamp = message.get("timestamp")
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _friendly_parameter_name(name: str) -> str:
    return _friendly_name(name, fallback="Parameter")


def _friendly_parameter_value(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "None"
        return "; ".join(
            f"{_friendly_parameter_name(str(name))}: {_friendly_parameter_value(item)}"
            for name, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "None"
        return ", ".join(_friendly_parameter_value(item) for item in value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


_VISIBLE_CONFIRMATION_PARAMETERS: dict[str, tuple[str, ...]] = {
    "read_file": ("path", "offset", "limit"),
    "list_dir": ("path", "recursive", "max_entries"),
    "glob": ("pattern", "path", "kind", "head_limit", "offset"),
    "grep": (
        "pattern",
        "path",
        "glob",
        "type",
        "output_mode",
        "context",
        "head_limit",
        "offset",
    ),
    "web_fetch": ("url", "format"),
}


def _selected_parameter_lines(
    details: Mapping[str, object],
    names: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    for name in names:
        if name not in details:
            continue
        value = details[name]
        if name == "url":
            value = _safe_confirmation_url(value)
        lines.append(f"{_friendly_parameter_name(name)}: {_friendly_parameter_value(value)}")
    return lines


def _text_size_line(label: str, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    unit = "character" if len(value) == 1 else "characters"
    return f"{label}: {len(value)} {unit}"


def _safe_confirmation_url(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "Invalid URL"
    if not parsed.scheme or hostname is None:
        return "Invalid URL"
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        host = f"{host}:{port}"
    rendered = f"{parsed.scheme}://{host}{parsed.path}"
    if parsed.query:
        rendered = f"{rendered}?<redacted>"
    return rendered


def _confirmation_detail_lines(request: ConfirmationRequestView) -> tuple[str, ...]:
    details = request.details
    if request.mcp_identity is not None:
        identity = request.mcp_identity
        return (
            f"MCP Server: {identity.server_name}",
            f"Remote Tool: {identity.remote_name}",
            f"Model Tool: {identity.model_name}",
            "Arguments: "
            + json.dumps(details, ensure_ascii=False, separators=(",", ":")),
        )
    tool_name = request.tool_name.casefold()
    if tool_name == "exec" and "command" in details:
        lines = [f"Command: {_friendly_parameter_value(details['command'])}"]
        for name in ("cwd", "timeout"):
            if name in details:
                lines.append(
                    f"{_friendly_parameter_name(name)}: {_friendly_parameter_value(details[name])}"
                )
        return tuple(lines)
    if tool_name == "write_file":
        lines = _selected_parameter_lines(details, ("path",))
        content_size = _text_size_line("Content", details.get("content"))
        if content_size is not None:
            lines.append(content_size)
        return tuple(lines) or ("Parameters: None",)
    if tool_name == "edit_file":
        lines = _selected_parameter_lines(details, ("path", "replace_all"))
        for label, name in (("Existing Text", "old_text"), ("Replacement Text", "new_text")):
            text_size = _text_size_line(label, details.get(name))
            if text_size is not None:
                lines.append(text_size)
        return tuple(lines) or ("Parameters: None",)
    visible_names = _VISIBLE_CONFIRMATION_PARAMETERS.get(tool_name)
    if visible_names is None:
        return ("Parameters: Not displayed",)
    return tuple(_selected_parameter_lines(details, visible_names)) or ("Parameters: None",)


def _tool_row_content(
    status: _ToolRowStatus,
    tool_name: str,
    summary: str,
    *,
    raw_arguments: str | None = None,
) -> str:
    display_name = _concise_tool_name(tool_name)
    if status == "running":
        if raw_arguments is None:
            return f"Running: {display_name}"
        return f"Running: {display_name}\nArguments: {raw_arguments}"
    if status == "success":
        return f"Completed: {display_name}"
    if status == "refused":
        return f"Rejected: {display_name}"
    if status == "cancelled":
        return f"Cancelled: {display_name}"
    if status == "unknown":
        return f"Status unavailable: {display_name}"
    return f"Failed: {display_name} - {_safe_failure_reason(summary, display_name)}"


def _format_activity_duration(elapsed: float) -> str:
    total_seconds = max(0, int(elapsed))
    seconds = total_seconds % 60
    if total_seconds < 60:
        return f"{seconds}s"
    minutes = (total_seconds // 60) % 60
    if total_seconds < 3600:
        return f"{total_seconds // 60}min {seconds}s"
    hours = total_seconds // 3600
    return f"{hours}h {minutes}min {seconds}s"


def _activity_group_heading_text(
    *,
    expanded: bool,
    elapsed: float,
    outcome: _TerminalOutcome | None = None,
    toggleable: bool = False,
) -> str:
    symbol = _ACTIVITY_EXPANDED_SYMBOL if expanded else _ACTIVITY_COLLAPSED_SYMBOL
    label = {
        None: "Activity" if toggleable else "Running",
        "completed": "Completed",
        "failed": "Failed",
        "cancelled": "Cancelled",
    }[outcome]
    return f"{symbol} {label} | {_format_activity_duration(elapsed)}"


def _concise_tool_name(tool_name: str) -> str:
    display_name = " ".join(tool_name.split()) or "Tool"
    if len(display_name) <= _TOOL_NAME_MAX_CHARS:
        return display_name
    return f"{display_name[: _TOOL_NAME_MAX_CHARS - 3].rstrip()}..."


def _safe_failure_reason(summary: str, tool_name: str) -> str:
    detail = " ".join(summary.split())
    if not detail or _UNSAFE_TOOL_DETAIL_PATTERN.search(detail):
        return _GENERIC_TOOL_FAILURE_REASON

    for prefix in ("Failed", "Error", "Finished", "Completed"):
        if detail.casefold().startswith(prefix.casefold()):
            detail = detail[len(prefix) :].lstrip(" :-")
            break
    if detail.casefold().startswith(tool_name.casefold()):
        detail = detail[len(tool_name) :].lstrip(" :-")
    if not detail:
        return _GENERIC_TOOL_FAILURE_REASON
    if len(detail) <= _FAILURE_REASON_MAX_CHARS:
        return detail
    return f"{detail[: _FAILURE_REASON_MAX_CHARS - 3].rstrip()}..."


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
