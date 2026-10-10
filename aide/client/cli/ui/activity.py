"""Tool and SubAgent activity presentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from textual import on
from textual.containers import Vertical
from textual.events import Click, Key, Unmount
from textual.message import Message
from textual.widgets import Static

_FAILURE_REASON_MAX_CHARS = 120


_TOOL_NAME_MAX_CHARS = 80


_GENERIC_TOOL_FAILURE_REASON = "The operation did not complete."


_UNSAFE_TOOL_DETAIL_PATTERN = re.compile(
    r"(?:^\s*[\[{])|(?:[\"'][^\"']+[\"']\s*:)|"
    r"(?:\b(?:api[_-]?key|authorization|bearer|password|secret|token)\b)|"
    r"(?:\b(?:arguments?|parameters?|result|output|content)\b\s*[:=])|"
    r"(?:\bcall[-_][A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


_ACTIVITY_EXPANDED_SYMBOL = "\u25bc"


_ACTIVITY_COLLAPSED_SYMBOL = "\u25b6"


type _ToolRowStatus = Literal["running", "success", "error", "refused", "cancelled", "unknown"]


type _TerminalOutcome = Literal["completed", "cancelled", "failed"]


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
