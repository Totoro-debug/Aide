"""Conversation scrolling display and anchors."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import partial

from textual.app import ScreenStackError
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.events import MouseScrollDown, MouseScrollUp, Resize
from textual.message import Message
from textual.scrollbar import ScrollTo
from textual.widget import Widget
from textual.widgets import Static

from aide.client.cli.ui.activity import _ActivityGroupHeading

_COMPACT_MESSAGE_MAX_WIDTH = 60


_CONVERSATION_NAVIGATION_KEYS = frozenset({"pageup", "pagedown", "ctrl+home", "ctrl+end"})


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
