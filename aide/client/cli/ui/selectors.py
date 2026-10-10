"""Conversation model, reasoning and permission selectors."""

from __future__ import annotations

from typing import Final, cast

from rich.style import Style
from rich.text import Text
from textual import on
from textual.events import Click, Key, Resize
from textual.message import Message
from textual.widgets import Static

from aide.agent.permission import PERMISSION_LEVELS, ToolPermissionLevel
from aide.provider.models import REASONING_EFFORT_LEVELS, ReasoningEffort

_PERMISSION_LABELS: Final[tuple[str, ...]] = (
    "Read-Only",
    "Workspace-Write",
    "Full-Access",
)


class _ReasoningEffortSelector(Static):
    """Focused horizontal selector for the Runtime-Lifetime Reasoning Effort."""

    can_focus = True

    class Confirmed(Message):
        def __init__(self, selector: _ReasoningEffortSelector) -> None:
            super().__init__()
            self.effort = selector.selected_effort

    class Cancelled(Message):
        pass

    def __init__(self, effort: ReasoningEffort = "mid", *, id: str | None = None) -> None:
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
