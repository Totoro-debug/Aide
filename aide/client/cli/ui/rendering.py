"""Conversation message and stream projections."""

from __future__ import annotations

import json
from asyncio import Task, create_task, sleep
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast
from uuid import UUID

from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token
from rich.cells import cell_len, set_cell_size
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Markdown

from aide.agent.message_bus import OutboundMessage
from aide.client.cli.ui.activity import (
    _activity_group_heading_text,
    _ActivityGroupState,
    _TerminalOutcome,
    _tool_row_content,
    _ToolRowState,
    _ToolRowStatus,
)
from aide.client.cli.ui.display import _ConversationDisplay
from aide.management.service import RuntimeStatus

if TYPE_CHECKING:
    from aide.client.cli.conversation import TerminalConversationApp


_SPARSE_MARKERS = ("_stream_delta", "_stream_end", "_streamed")


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
    parser.core.ruler.after("linkify", "aide_visible_links", _make_links_visible)
    return parser


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
                    or set(outbound.metadata) - {"tool_call_id", "status", "result"}
                    or (
                        "result" in outbound.metadata
                        and not isinstance(outbound.metadata["result"], str)
                    )
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
