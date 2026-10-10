"""Terminal Session selection and actions."""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from aide.management.service import SessionListingEntry


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


def _session_picker_label(session: SessionListingEntry) -> str:
    local_updated_at = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
    noun = "message" if session.message_count == 1 else "messages"
    return f"{session.title} | {local_updated_at} | {session.message_count} {noun}"
