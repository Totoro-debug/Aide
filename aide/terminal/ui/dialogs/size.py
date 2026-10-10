"""Terminal size recovery screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Static

_MIN_TERMINAL_WIDTH = 20


_MIN_TERMINAL_HEIGHT = 10


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
