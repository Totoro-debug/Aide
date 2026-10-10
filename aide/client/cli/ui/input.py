"""Conversation input and completion widgets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, cast

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widgets import OptionList, TextArea

from aide.client.cli.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter
from aide.client.cli.ui.display import _CONVERSATION_NAVIGATION_KEYS, _ConversationDisplay
from aide.management.commands import (
    MANAGEMENT_COMMANDS,
)
from aide.skills.catalog import SkillMetadata

type _ControlAction = Literal["cancel_active_turn", "clear_draft", "drain_pending", "exit"]


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
