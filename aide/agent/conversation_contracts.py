"""Presentation and control contracts independent of Agent Run execution."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ForegroundConversationProjection:
    """Presentation-safe snapshot of the active foreground conversation."""

    session_id: str
    messages: tuple[dict[str, Any], ...]


class TerminalAgentRunExecutorControl(Protocol):
    """Foreground control surface including Terminal history projection."""

    @property
    def has_active_run(self) -> bool: ...

    def foreground_input_admitted(self) -> bool: ...

    async def cancel_active_run(self) -> None: ...

    def project_foreground_conversation(self) -> ForegroundConversationProjection: ...
