"""Terminal confirmation screens and presentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from aide.agent.loop import (
    ConfirmationRequestView,
)

type ConfirmationDecision = Literal["approved", "declined"]


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


def _friendly_name(name: str, *, fallback: str) -> str:
    words = name.replace("_", " ").split()
    return (
        " ".join(
            _FRIENDLY_INITIALISMS.get(word.casefold(), word[:1].upper() + word[1:])
            for word in words
        )
        or fallback
    )


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


_FRIENDLY_INITIALISMS = {"cwd": "CWD", "id": "ID", "url": "URL"}
