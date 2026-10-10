"""Terminal Restore preview and confirmation."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from aide.agent.session.restore import RestoreMode, RestorePlan, RestoreResult
from aide.agent.session.session import RestoreAnchor
from aide.management.commands import (
    format_restore_preview,
)


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
