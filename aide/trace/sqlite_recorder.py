"""SQLite-backed asynchronous storage for published Execution Traces."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import threading
import zlib
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar, cast
from uuid import uuid4

from aide.provider.session_configuration import REASONING_EFFORT_LEVELS, ReasoningEffort
from aide.trace.context import TraceContext, TraceRootSource
from aide.trace.recorder import TraceRecorder
from aide.trace.types import (
    TraceAttemptHandle,
    TraceAttemptMetadata,
    TraceBreakdown,
    TraceCallMetadata,
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureLayer,
    TraceCaptureState,
    TraceDirection,
    TraceErrorBreakdown,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePage,
    TracePurpose,
    TraceRawHandle,
    TraceRawPage,
    TraceRawSource,
    TraceRecordDetail,
    TraceRecordFilters,
    TraceRecordSummary,
    TraceSessionPage,
    TraceSessionSummary,
    TraceStats,
    TraceStatsTotals,
    TraceStatus,
    TraceTimelinePoint,
    TraceTokenTotals,
    TraceTokenUsage,
    TraceUsageState,
)

__all__ = [
    "TRACE_DATABASE_FILENAME",
    "TRACE_MAX_QUEUE_EVENTS",
    "TRACE_MAX_QUEUE_PAYLOAD_BYTES",
    "TRACE_MAX_RAW_CHUNK_BYTES",
    "TRACE_RETENTION_DAYS",
    "SQLiteTraceRecorder",
]

TRACE_DATABASE_FILENAME = "traces.sqlite3"
TRACE_MAX_QUEUE_PAYLOAD_BYTES = 64 * 1024 * 1024
TRACE_MAX_QUEUE_EVENTS = 4096
TRACE_MAX_RAW_CHUNK_BYTES = 64 * 1024
TRACE_RETENTION_DAYS = 365
_WRITER_RETRY_DELAYS = (0.1, 1.0, 5.0)
_WRITER_BATCH_SIZE = 128
_WRITER_BATCH_DELAY = 0.002
_PURGE_BATCH_SIZE = 500
_TRACE_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
)
_SCHEMA_VERSION = 1
_PAGE_LIMIT_MAX = 1000
_T = TypeVar("_T")
_METHODLOGY = (
    "Statistics include published records retained in this database. Model failures "
    "retain final error metadata but not failed request or response candidates. "
    "Reported token values are summed independently; unknown and inapplicable usage "
    "is not counted as zero usage."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    publish_sequence INTEGER NOT NULL CHECK (publish_sequence >= 0)
);
INSERT OR IGNORE INTO trace_meta(singleton, publish_sequence) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS trace_roots (
    root_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    workspace_key TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    origin TEXT NOT NULL,
    purpose TEXT NOT NULL,
    owner_session_id TEXT,
    subject_session_id TEXT,
    run_id TEXT,
    restore_run_token TEXT,
    agent_id TEXT,
    parent_root_id TEXT,
    schedule_run_id TEXT
);

CREATE TABLE IF NOT EXISTS trace_records (
    record_id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL REFERENCES trace_roots(root_id) ON DELETE CASCADE,
    workspace_key TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    origin TEXT NOT NULL,
    owner_session_id TEXT,
    subject_session_id TEXT,
    run_id TEXT,
    schedule_run_id TEXT,
    restore_run_token TEXT,
    agent_id TEXT,
    parent_root_id TEXT,
    parent_record_id TEXT,
    tool_call_id TEXT,
    kind TEXT NOT NULL,
    purpose TEXT NOT NULL,
    tool_name TEXT,
    model TEXT,
    reasoning_effort TEXT,
    endpoint TEXT,
    status TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    final_attempt_duration_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_category TEXT,
    error_code TEXT,
    error_message TEXT,
    usage_state TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cached_input_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    request_json TEXT,
    request_set INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    result_set INTEGER NOT NULL DEFAULT 0,
    capture_state TEXT NOT NULL DEFAULT 'complete',
    restored INTEGER NOT NULL DEFAULT 0,
    published INTEGER NOT NULL DEFAULT 0,
    publish_sequence INTEGER
);

CREATE TABLE IF NOT EXISTS trace_attempts (
    record_id TEXT NOT NULL REFERENCES trace_records(record_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    model TEXT,
    reasoning_effort TEXT,
    endpoint TEXT,
    request_json TEXT,
    request_set INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (record_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS trace_raw_streams (
    stream_id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES trace_records(record_id) ON DELETE CASCADE,
    attempt_id TEXT,
    source TEXT NOT NULL,
    direction TEXT NOT NULL,
    layer TEXT NOT NULL,
    content_type TEXT,
    content_encoding TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_raw_stream_direct
    ON trace_raw_streams(record_id, source) WHERE attempt_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_raw_stream_attempt
    ON trace_raw_streams(record_id, attempt_id, source) WHERE attempt_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS trace_raw_chunks (
    stream_id INTEGER NOT NULL REFERENCES trace_raw_streams(stream_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    part INTEGER NOT NULL,
    original_size INTEGER NOT NULL CHECK (original_size >= 0),
    compressed_data BLOB NOT NULL,
    PRIMARY KEY (stream_id, sequence, part)
);

CREATE TABLE IF NOT EXISTS trace_capture_issues (
    issue_id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES trace_records(record_id) ON DELETE CASCADE,
    attempt_id TEXT,
    state TEXT NOT NULL,
    message TEXT NOT NULL,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_issues_record ON trace_capture_issues(record_id);

CREATE INDEX IF NOT EXISTS idx_trace_records_published_sequence
    ON trace_records(publish_sequence DESC) WHERE published = 1;
CREATE INDEX IF NOT EXISTS idx_trace_records_finished
    ON trace_records(finished_at DESC, record_id DESC) WHERE published = 1;
CREATE INDEX IF NOT EXISTS idx_trace_records_owner_finished
    ON trace_records(workspace_key, owner_session_id, finished_at DESC)
    WHERE published = 1;
CREATE INDEX IF NOT EXISTS idx_trace_records_subject_finished
    ON trace_records(workspace_key, subject_session_id, finished_at DESC)
    WHERE published = 1;
CREATE INDEX IF NOT EXISTS idx_trace_records_root
    ON trace_records(root_id, publish_sequence DESC) WHERE published = 1;
PRAGMA user_version = 1;
"""


@dataclass(slots=True)
class _CaptureState:
    context: TraceContext
    metadata: TraceCallMetadata
    attempts: dict[str, tuple[int, TraceAttemptMetadata]] = field(default_factory=dict)
    latest_attempt_id: str | None = None
    finished: bool = False
    issues: dict[tuple[str | None, str, str | None, str], TraceCaptureIssue] = field(
        default_factory=dict
    )

    def add_issue(self, attempt_id: str | None, issue: TraceCaptureIssue) -> None:
        self.issues[(attempt_id, issue.state.value, issue.source, issue.message)] = issue


@dataclass(frozen=True, slots=True)
class _Event:
    ticket: int
    kind: str
    record_id: str
    payload: dict[str, object]
    payload_bytes: int


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _enum_value(value: object | None) -> str | None:
    return None if value is None else cast(str, getattr(value, "value", value))


def _json_payload_size(value: object, *, maximum: int = TRACE_MAX_QUEUE_PAYLOAD_BYTES) -> int:
    """Bound retained event content without serializing on the capture thread."""
    stack: list[Iterator[object]] = [iter((value,))]
    total = 0
    visited = 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visited += 1
        if visited > 100_000:
            return maximum + 1
        if item is None:
            total += 4
        elif item is True:
            total += 4
        elif item is False:
            total += 5
        elif isinstance(item, str):
            total += 2
            for char in item:
                codepoint = ord(char)
                if char in '"\\':
                    total += 2
                elif codepoint < 0x20:
                    total += 2 if char in "\b\t\n\f\r" else 6
                elif 0xD800 <= codepoint <= 0xDFFF:
                    total += 6
                else:
                    total += (
                        1
                        if codepoint < 0x80
                        else 2
                        if codepoint < 0x800
                        else 3
                        if codepoint < 0x10000
                        else 4
                    )
                if total > maximum:
                    return maximum + 1
        elif isinstance(item, bytes):
            total += len(item)
        elif isinstance(item, int):
            total += (item.bit_length() * 30103 // 100000) + 1 + (item < 0)
        elif isinstance(item, float):
            total += 24
        elif isinstance(item, dict):
            total += 2 + max(0, len(item) - 1) + 3 * len(item)
            stack.append(iter(item.keys()))
            stack.append(iter(item.values()))
        elif isinstance(item, (list, tuple)):
            total += 2 + max(0, len(item) - 1)
            stack.append(iter(item))
        elif is_dataclass(item) and not isinstance(item, type):
            stack.append(iter(tuple(getattr(item, entry.name) for entry in fields(item))))
        elif isinstance(item, datetime):
            total += 32
        else:
            total += 64
        if total > maximum:
            return maximum + 1
    return total


def _encode_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded.encode("utf-8")
    return encoded


def _cursor_encode(snapshot: int, last: int) -> str:
    raw = json.dumps({"v": 1, "snapshot": snapshot, "last": last}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("ascii")).decode("ascii").rstrip("=")


def _cursor_decode(cursor: str) -> tuple[int, int]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        snapshot = value["snapshot"]
        last = value["last"]
        if (
            value["v"] != 1
            or isinstance(snapshot, bool)
            or isinstance(last, bool)
            or not isinstance(snapshot, int)
            or not isinstance(last, int)
            or snapshot < 0
            or last < 0
        ):
            raise ValueError
        return snapshot, last
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid Trace cursor") from error


def _record_cursor_encode(snapshot: int, finished_at: str, record_id: str) -> str:
    raw = json.dumps(
        {"v": 1, "snapshot": snapshot, "finished_at": finished_at, "record_id": record_id}
    )
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _record_cursor_decode(cursor: str) -> tuple[int, str, str]:
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        snapshot, finished_at, record_id = (
            value["snapshot"],
            value["finished_at"],
            value["record_id"],
        )
        if (
            value["v"] != 1
            or isinstance(snapshot, bool)
            or not isinstance(snapshot, int)
            or snapshot < 0
            or not isinstance(finished_at, str)
            or not isinstance(record_id, str)
        ):
            raise ValueError
        _parse_utc(finished_at)
        return snapshot, finished_at, record_id
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid Trace cursor") from error


def _filter_sql(filters: TraceRecordFilters | None, alias: str = "r") -> tuple[str, list[object]]:
    if filters is None:
        return "", []
    conditions: list[str] = []
    parameters: list[object] = []
    fields = (
        ("workspace_key", filters.workspace_key),
        ("owner_session_id", filters.owner_session_id),
        ("subject_session_id", filters.subject_session_id),
        ("root_id", filters.root_id),
        ("run_id", filters.run_id),
        ("agent_id", filters.agent_id),
        ("kind", filters.kind),
        ("status", filters.status),
        ("purpose", filters.purpose),
        ("tool_name", filters.tool_name),
        ("model", filters.model),
    )
    for name, value in fields:
        if value is not None:
            conditions.append(f"{alias}.{name} = ?")
            parameters.append(_enum_value(value))
    if filters.from_at is not None:
        conditions.append(f"{alias}.finished_at >= ?")
        parameters.append(_utc_text(filters.from_at))
    if filters.to_at is not None:
        conditions.append(f"{alias}.finished_at <= ?")
        parameters.append(_utc_text(filters.to_at))
    return (" AND " + " AND ".join(conditions) if conditions else "", parameters)


class SQLiteTraceRecorder(TraceRecorder):
    """Capture without disk waits, then publish completed calls from one writer thread."""

    def __init__(
        self,
        path: Path,
        *,
        max_queue_events: int = TRACE_MAX_QUEUE_EVENTS,
        max_queue_payload_bytes: int = TRACE_MAX_QUEUE_PAYLOAD_BYTES,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            isinstance(max_queue_events, bool)
            or not isinstance(max_queue_events, int)
            or max_queue_events <= 0
        ):
            raise ValueError("max_queue_events must be a positive integer")
        if (
            isinstance(max_queue_payload_bytes, bool)
            or not isinstance(max_queue_payload_bytes, int)
            or max_queue_payload_bytes <= 0
        ):
            raise ValueError("max_queue_payload_bytes must be a positive integer")
        self.path = Path(path)
        self._now = now if now is not None else lambda: datetime.now(UTC)
        self._max_queue_events = max_queue_events
        self._max_queue_payload_bytes = max_queue_payload_bytes
        self._condition = threading.Condition()
        self._queue: deque[_Event] = deque()
        self._queue_bytes = 0
        self._inflight_events = 0
        self._inflight_bytes = 0
        self._next_ticket = 0
        self._completed_ticket = 0
        self._completed_out_of_order: set[int] = set()
        self._closing = False
        self._stop_writer = threading.Event()
        self._worker_done = False
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._states: dict[str, _CaptureState] = {}
        self._closed_sessions: set[tuple[str, str]] = set()
        self._worker = threading.Thread(
            target=self._run_writer,
            name="aide-trace-sqlite-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def startup_error(self) -> Exception | None:
        return self._startup_error

    async def wait_until_ready(self, timeout_seconds: float = 5.0) -> bool:
        """Wait briefly for schema initialization without blocking the event loop."""
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        ready = await asyncio.to_thread(self._ready.wait, timeout_seconds)
        return ready and self._startup_error is None and not self._worker_done

    def new_root(self, source: TraceRootSource) -> TraceContext:
        return TraceContext.create(str(uuid4()), source)

    def begin_call(self, context: TraceContext, metadata: TraceCallMetadata) -> TraceCaptureHandle:
        handle = TraceCaptureHandle(record_id=str(uuid4()), root_id=context.root_id)
        state = _CaptureState(context=context, metadata=metadata)
        with self._condition:
            closed = (
                (context.workspace_key, context.owner_session_id) in self._closed_sessions
                if context.owner_session_id is not None
                else False
            )
            closed = closed or self._closing or self._worker_done
            if not closed:
                self._states[handle.record_id] = state
        if not closed:
            self._enqueue(
                "begin",
                handle.record_id,
                {"context": context, "metadata": metadata},
            )
        return handle

    def begin_attempt(
        self, handle: TraceCaptureHandle, metadata: TraceAttemptMetadata
    ) -> TraceAttemptHandle:
        attempt = TraceAttemptHandle(record_id=handle.record_id, attempt_id=str(uuid4()))
        with self._condition:
            state = self._states.get(handle.record_id)
            if state is None or state.finished:
                return attempt
            ordinal = len(state.attempts) + 1
            state.attempts[attempt.attempt_id] = (ordinal, metadata)
            state.latest_attempt_id = attempt.attempt_id
        self._enqueue(
            "attempt",
            handle.record_id,
            {"attempt_id": attempt.attempt_id, "ordinal": ordinal, "metadata": metadata},
            attempt_id=attempt.attempt_id,
        )
        return attempt

    def set_endpoint(self, handle: TraceRawHandle, endpoint: str) -> None:
        attempt_id = handle.attempt_id if isinstance(handle, TraceAttemptHandle) else None
        if not isinstance(endpoint, str) or not endpoint:
            self._note_loss(
                handle.record_id,
                attempt_id,
                "endpoint",
                "Call endpoint metadata was invalid.",
            )
            return
        with self._condition:
            state = self._states.get(handle.record_id)
            if state is None or state.finished:
                return
            if attempt_id is None:
                state.metadata = replace(state.metadata, endpoint=endpoint)
            else:
                attempt = state.attempts.get(attempt_id)
                if attempt is None:
                    return
                ordinal, metadata = attempt
                state.attempts[attempt_id] = (ordinal, replace(metadata, endpoint=endpoint))
        issue = TraceCaptureIssue(
            state=TraceCaptureState.PARTIAL,
            message="Call endpoint metadata could not be queued.",
            source="provider_transport" if attempt_id is not None else "tool_endpoint",
        )
        self._enqueue(
            "attempt_endpoint" if attempt_id is not None else "call_endpoint",
            handle.record_id,
            {"attempt_id": attempt_id, "endpoint": endpoint},
            attempt_id=attempt_id,
            source="endpoint",
            dropped_issue=issue,
        )

    def set_request(self, handle: TraceRawHandle, request: object) -> None:
        attempt_id = handle.attempt_id if isinstance(handle, TraceAttemptHandle) else None
        self._enqueue(
            "request",
            handle.record_id,
            {"attempt_id": attempt_id, "request": request},
            attempt_id=attempt_id,
            source="request",
        )

    def report_capture_issue(self, handle: TraceRawHandle, issue: TraceCaptureIssue) -> None:
        attempt_id = handle.attempt_id if isinstance(handle, TraceAttemptHandle) else None
        self._enqueue(
            "issue",
            handle.record_id,
            {"attempt_id": attempt_id, "issue": issue},
            attempt_id=attempt_id,
            source=issue.source,
            dropped_issue=issue,
        )

    def append_raw(
        self,
        handle: TraceRawHandle,
        *,
        source: str,
        direction: TraceDirection,
        layer: TraceCaptureLayer,
        sequence: int,
        data: bytes,
        content_type: str | None = None,
        content_encoding: str | None = None,
    ) -> None:
        attempt_id = handle.attempt_id if isinstance(handle, TraceAttemptHandle) else None
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            self._note_loss(handle.record_id, attempt_id, source, "Invalid raw data sequence.")
            return
        if not isinstance(data, bytes):
            self._note_loss(handle.record_id, attempt_id, source, "Raw data was not bytes.")
            return
        part = 0
        start = 0
        while start < len(data) or (not data and part == 0):
            end = min(start + TRACE_MAX_RAW_CHUNK_BYTES, len(data))
            chunk = data[start:end]
            self._enqueue(
                "raw",
                handle.record_id,
                {
                    "attempt_id": attempt_id,
                    "source": source,
                    "direction": direction,
                    "layer": layer,
                    "sequence": sequence,
                    "part": part,
                    "data": chunk,
                    "content_type": content_type,
                    "content_encoding": content_encoding,
                },
                attempt_id=attempt_id,
                source=source,
            )
            start = end
            part += 1
            if not data:
                break

    def finish_call(
        self,
        handle: TraceCaptureHandle,
        outcome: TraceOutcome,
        *,
        result: object,
        usage: TraceTokenUsage,
        retry_count: int,
    ) -> None:
        with self._condition:
            state = self._states.get(handle.record_id)
            if state is None or state.finished:
                return
            state.finished = True
            attempt_id = state.latest_attempt_id
            attempt = state.attempts.get(attempt_id) if attempt_id is not None else None
            issues = tuple(
                (issue_attempt, issue) for (issue_attempt, _, _, _), issue in state.issues.items()
            )
            payload: dict[str, object] = {
                "context": state.context,
                "metadata": state.metadata,
                "outcome": outcome,
                "result": result,
                "usage": usage,
                "retry_count": retry_count,
                "final_attempt_id": attempt_id,
                "final_attempt": attempt[1] if attempt is not None else None,
                "final_attempt_ordinal": attempt[0] if attempt is not None else 0,
                "issues": issues,
            }
        if outcome.status is TraceStatus.FAILED and state.metadata.kind is TraceKind.MODEL:
            payload["result"] = None
        ticket = self._enqueue(
            "finish",
            handle.record_id,
            payload,
            attempt_id=attempt_id,
            source="result",
        )
        if ticket is None:
            with self._condition:
                self._states.pop(handle.record_id, None)

    def close_session_scope(self, workspace_key: str, owner_session_id: str) -> None:
        with self._condition:
            self._closed_sessions.add((workspace_key, owner_session_id))

    async def purge_session(self, workspace_key: str, owner_session_id: str) -> None:
        self.close_session_scope(workspace_key, owner_session_id)
        await self._run_maintenance_command(
            "purge",
            {"workspace_key": workspace_key, "owner_session_id": owner_session_id},
        )

    async def purge_expired(self) -> None:
        cutoff = self._retention_cutoff()
        while True:
            payload: dict[str, object] = {"cutoff": cutoff, "deleted": 0}
            await self._run_maintenance_command("expire", payload)
            if cast(int, payload["deleted"]) < _PURGE_BATCH_SIZE:
                return

    async def _run_maintenance_command(self, kind: str, payload: dict[str, object]) -> None:
        ticket = await asyncio.to_thread(self._enqueue_control, kind, "", payload)
        await asyncio.to_thread(self._wait_for_ticket, ticket, ordered=False)
        error = payload.get("error")
        if isinstance(error, Exception):
            raise error

    def mark_restored(
        self,
        workspace_key: str,
        owner_session_id: str,
        discarded_run_tokens: Sequence[str],
    ) -> None:
        tokens = tuple(discarded_run_tokens)
        self._enqueue(
            "restore",
            "",
            {
                "workspace_key": workspace_key,
                "owner_session_id": owner_session_id,
                "tokens": tokens,
            },
        )

    async def list_sessions(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TraceSessionPage:
        self._validate_limit(limit)
        return await self._run_query(self._list_sessions_sync, filters, cursor, limit)

    async def list_records(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TracePage:
        self._validate_limit(limit)
        return await self._run_query(self._list_records_sync, filters, cursor, limit)

    async def get_record(self, record_id: str) -> TraceRecordDetail | None:
        return await self._run_query(self._get_record_sync, record_id)

    async def read_raw(
        self,
        record_id: str,
        *,
        source: str,
        offset: int,
        limit_bytes: int,
    ) -> TraceRawPage:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes <= 0:
            raise ValueError("limit_bytes must be a positive integer")
        return await self._run_query(
            self._read_raw_sync,
            record_id,
            source,
            offset,
            limit_bytes,
        )

    async def get_stats(self, filters: TraceRecordFilters | None = None) -> TraceStats:
        return await self._run_query(self._get_stats_sync, filters)

    async def close(self, timeout_seconds: float = 2.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        await asyncio.to_thread(self._worker.join, max(0.0, timeout_seconds))
        if self._worker.is_alive():
            self._stop_writer.set()
            with self._condition:
                self._condition.notify_all()

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _PAGE_LIMIT_MAX
        ):
            raise ValueError(f"limit must be between 1 and {_PAGE_LIMIT_MAX}")

    def _retention_cutoff(self) -> str:
        now = self._now()
        if (
            not isinstance(now, datetime)
            or now.tzinfo is None
            or now.utcoffset() != UTC.utcoffset(now)
        ):
            raise ValueError("Trace clock must return an aware UTC datetime")
        return _utc_text(now - timedelta(days=TRACE_RETENTION_DAYS))

    def _record_visibility_sql(
        self, connection: sqlite3.Connection, alias: str = "r"
    ) -> tuple[str, list[object]]:
        with self._condition:
            closed_sessions = tuple(self._closed_sessions)
        closed_sql = ""
        if closed_sessions:
            connection.execute(
                "CREATE TEMP TABLE trace_closed_session_scopes ("
                "workspace_key TEXT NOT NULL, owner_session_id TEXT NOT NULL, "
                "PRIMARY KEY (workspace_key, owner_session_id)) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT INTO trace_closed_session_scopes(workspace_key, owner_session_id) "
                "VALUES (?, ?)",
                closed_sessions,
            )
            closed_sql = (
                " AND NOT EXISTS (SELECT 1 FROM trace_closed_session_scopes AS c "
                f"WHERE c.workspace_key = {alias}.workspace_key "
                f"AND c.owner_session_id = {alias}.owner_session_id)"
            )
        return (
            f" AND {alias}.finished_at >= ? "
            + closed_sql,
            [self._retention_cutoff()],
        )

    def _session_scope_closed(self, workspace_key: str, owner_session_id: str | None) -> bool:
        with self._condition:
            return (workspace_key, owner_session_id) in self._closed_sessions

    def _enqueue(
        self,
        kind: str,
        record_id: str,
        payload: dict[str, object],
        *,
        attempt_id: str | None = None,
        source: str | None = None,
        dropped_issue: TraceCaptureIssue | None = None,
    ) -> int | None:
        payload_bytes = _json_payload_size(payload, maximum=self._max_queue_payload_bytes)
        with self._condition:
            state = self._states.get(record_id)
            if kind != "restore":
                if state is None or (state.finished and kind != "finish"):
                    return None
                if (
                    state.context.owner_session_id is not None
                    and (
                        state.context.workspace_key,
                        state.context.owner_session_id,
                    )
                    in self._closed_sessions
                ):
                    return None
            if (
                self._closing
                or self._worker_done
                or len(self._queue) + self._inflight_events >= self._max_queue_events
                or self._queue_bytes + self._inflight_bytes + payload_bytes
                > self._max_queue_payload_bytes
            ):
                if state is not None:
                    issue = dropped_issue or TraceCaptureIssue(
                        state=TraceCaptureState.PARTIAL,
                        message="Trace data was dropped because the bounded queue was full.",
                        source=source,
                    )
                    state.add_issue(attempt_id, issue)
                return None
            self._next_ticket += 1
            event = _Event(self._next_ticket, kind, record_id, payload, payload_bytes)
            self._queue.append(event)
            self._queue_bytes += payload_bytes
            self._condition.notify()
            return event.ticket

    def _enqueue_control(self, kind: str, record_id: str, payload: dict[str, object]) -> int:
        with self._condition:
            while (
                not self._worker_done
                and not self._closing
                and (
                    len(self._queue) + self._inflight_events >= self._max_queue_events
                    or self._queue_bytes + self._inflight_bytes >= self._max_queue_payload_bytes
                )
            ):
                self._condition.wait()
            if self._closing or self._worker_done:
                raise RuntimeError("Trace SQLite writer is closed")
            self._next_ticket += 1
            event = _Event(self._next_ticket, kind, record_id, payload, 0)
            if kind == "purge":
                self._queue.appendleft(event)
            else:
                self._queue.append(event)
            self._condition.notify()
            return event.ticket

    def _note_loss(self, record_id: str, attempt_id: str | None, source: str, message: str) -> None:
        issue = TraceCaptureIssue(
            state=TraceCaptureState.PARTIAL,
            message=message,
            source=source,
        )
        with self._condition:
            state = self._states.get(record_id)
            if state is not None:
                state.add_issue(attempt_id, issue)

    def _wait_for_ticket(self, ticket: int, *, ordered: bool = True) -> None:
        self._ready.wait()
        with self._condition:
            while (
                self._completed_ticket < ticket
                and (ordered or ticket not in self._completed_out_of_order)
                and not self._worker_done
            ):
                self._condition.wait()
            error = self._startup_error
        if error is not None:
            raise RuntimeError("Trace SQLite writer is unavailable") from error

    def _complete_ticket(self, ticket: int) -> None:
        self._completed_out_of_order.add(ticket)
        while self._completed_ticket + 1 in self._completed_out_of_order:
            self._completed_ticket += 1
            self._completed_out_of_order.remove(self._completed_ticket)

    async def _run_query(self, function: Callable[..., _T], *args: object) -> _T:
        await asyncio.to_thread(self._ready.wait)
        if self._startup_error is not None:
            raise RuntimeError("Trace SQLite storage is unavailable") from self._startup_error
        return await asyncio.to_thread(function, *args)

    def _connect(self, *, writer: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        if writer:
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported Trace schema version {version}")
        if version == 0:
            connection.executescript(_SCHEMA)
        elif version != _SCHEMA_VERSION:
            raise RuntimeError(f"No migration path for Trace schema version {version}")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_records_sessions "
                "ON trace_records(workspace_key, owner_session_id, subject_session_id, origin, "
                "publish_sequence DESC, finished_at DESC, workspace_path) "
                "WHERE published = 1 AND owner_session_id IS NOT NULL"
            )
            connection.execute("DELETE FROM trace_records WHERE published = 0")
            connection.execute(
                "DELETE FROM trace_roots WHERE NOT EXISTS "
                "(SELECT 1 FROM trace_records WHERE trace_records.root_id = trace_roots.root_id "
                "AND trace_records.published = 1)"
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _run_writer(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect(writer=True)
            self._migrate(connection)
        except Exception as error:
            self._startup_error = error
            self._ready.set()
            with self._condition:
                for event in self._queue:
                    self._complete_ticket(event.ticket)
                    if event.kind == "finish":
                        self._states.pop(event.record_id, None)
                self._queue.clear()
                self._queue_bytes = 0
                self._states.clear()
                self._worker_done = True
                self._condition.notify_all()
            return

        self._ready.set()
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closing:
                        self._condition.wait()
                    if self._stop_writer.is_set():
                        break
                    if not self._queue and self._closing:
                        break
                    if len(self._queue) < _WRITER_BATCH_SIZE and not self._closing:
                        self._condition.wait(timeout=_WRITER_BATCH_DELAY)
                    events: list[_Event] = []
                    while self._queue and len(events) < _WRITER_BATCH_SIZE:
                        if events and (
                            (events[0].kind == "purge") != (self._queue[0].kind == "purge")
                        ):
                            break
                        event = self._queue.popleft()
                        self._queue_bytes -= event.payload_bytes
                        self._inflight_events += 1
                        self._inflight_bytes += event.payload_bytes
                        events.append(event)
                    self._condition.notify_all()

                self._process_events_with_retries(connection, events)
                with self._condition:
                    for event in events:
                        self._complete_ticket(event.ticket)
                        if event.kind == "finish":
                            self._states.pop(event.record_id, None)
                        elif event.kind == "purge":
                            workspace_key = cast(str, event.payload["workspace_key"])
                            owner_session_id = cast(str, event.payload["owner_session_id"])
                            for record_id, state in tuple(self._states.items()):
                                if (
                                    state.context.workspace_key == workspace_key
                                    and state.context.owner_session_id == owner_session_id
                                ):
                                    self._states.pop(record_id, None)
                    self._inflight_events -= len(events)
                    self._inflight_bytes -= sum(event.payload_bytes for event in events)
                    self._condition.notify_all()
        finally:
            connection.close()
            with self._condition:
                if self._queue:
                    for event in self._queue:
                        self._mark_writer_failure(event)
                        self._complete_ticket(event.ticket)
                    self._queue.clear()
                    self._queue_bytes = 0
                self._states.clear()
                self._worker_done = True
                self._condition.notify_all()

    def _process_event_with_retries(
        self, connection: sqlite3.Connection, event: _Event, *, initial_failed: bool = False
    ) -> None:
        delays = _WRITER_RETRY_DELAYS if initial_failed else (0.0, *_WRITER_RETRY_DELAYS)
        for delay in delays:
            if self._stop_writer.wait(delay):
                self._mark_writer_failure(event)
                return
            try:
                self._apply_events_transaction(connection, (event,))
            except Exception:
                if delay == _WRITER_RETRY_DELAYS[-1]:
                    self._mark_writer_failure(event)
                    return
            else:
                return

    def _process_events_with_retries(
        self,
        connection: sqlite3.Connection,
        events: list[_Event],
    ) -> None:
        if self._stop_writer.is_set():
            for event in events:
                self._mark_writer_failure(event)
            return
        if len(events) == 1:
            self._process_event_with_retries(connection, events[0])
            return
        try:
            self._apply_events_transaction(connection, events)
        except Exception:
            # Isolate a failed batch so one bad event cannot discard unrelated calls.
            for event in events:
                self._process_event_with_retries(connection, event, initial_failed=True)

    def _mark_writer_failure(self, event: _Event) -> None:
        with self._condition:
            if event.kind in ("purge", "expire"):
                event.payload["error"] = RuntimeError("Trace maintenance failed in the writer")
            state = self._states.get(event.record_id)
            if state is not None and event.kind != "finish":
                state.add_issue(
                    self._event_attempt_id(event),
                    TraceCaptureIssue(
                        state=TraceCaptureState.PARTIAL,
                        message="Trace data could not be stored after writer retries.",
                        source=self._event_source(event),
                    ),
                )

    @staticmethod
    def _event_attempt_id(event: _Event) -> str | None:
        value = event.payload.get("attempt_id")
        return cast(str | None, value)

    @staticmethod
    def _event_source(event: _Event) -> str | None:
        value = event.payload.get("source")
        return cast(str | None, value)

    def _apply_events_transaction(
        self,
        connection: sqlite3.Connection,
        events: Sequence[_Event],
    ) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for event in events:
                self._apply_event(connection, event)
            if self._stop_writer.is_set():
                raise RuntimeError("Trace drain deadline expired")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _apply_event(self, connection: sqlite3.Connection, event: _Event) -> None:
        if self._stop_writer.is_set():
            raise RuntimeError("Trace drain deadline expired")
        if event.record_id:
            with self._condition:
                state = self._states.get(event.record_id)
                if state is None or self._session_scope_closed(
                    state.context.workspace_key,
                    state.context.owner_session_id,
                ):
                    return
        if event.kind == "begin":
            self._write_begin(connection, event.record_id, event.payload)
        elif event.kind == "attempt":
            self._write_attempt(connection, event.record_id, event.payload)
        elif event.kind == "attempt_endpoint":
            connection.execute(
                "UPDATE trace_attempts SET endpoint = ? WHERE record_id = ? AND attempt_id = ?",
                (
                    cast(str, event.payload["endpoint"]),
                    event.record_id,
                    cast(str, event.payload["attempt_id"]),
                ),
            )
        elif event.kind == "request":
            self._write_request(connection, event.record_id, event.payload)
        elif event.kind == "call_endpoint":
            self._ensure_call_record(connection, event.record_id)
            connection.execute(
                "UPDATE trace_records SET endpoint = ? WHERE record_id = ? AND published = 0",
                (cast(str, event.payload["endpoint"]), event.record_id),
            )
        elif event.kind == "issue":
            self._write_issue_event(connection, event.record_id, event.payload)
        elif event.kind == "raw":
            self._write_raw(connection, event.record_id, event.payload)
        elif event.kind == "finish":
            self._write_finish(connection, event.record_id, event.payload)
        elif event.kind == "restore":
            self._write_restore(connection, event.payload)
        elif event.kind == "purge":
            self._write_purge(connection, event.payload)
        elif event.kind == "expire":
            event.payload["deleted"] = self._write_expired(connection, event.payload)
        else:
            raise ValueError(f"Unknown Trace writer event {event.kind}")

    @staticmethod
    def _write_begin(
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> None:
        context = cast(TraceContext, payload["context"])
        metadata = cast(TraceCallMetadata, payload["metadata"])
        connection.execute(
            """INSERT OR IGNORE INTO trace_roots(
                root_id, created_at, workspace_key, workspace_path, origin, purpose,
                owner_session_id, subject_session_id, run_id, restore_run_token,
                agent_id, parent_root_id, schedule_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.root_id,
                _utc_text(context.created_at),
                context.workspace_key,
                context.workspace_path,
                context.origin.value,
                context.purpose.value,
                context.owner_session_id,
                context.subject_session_id,
                context.run_id,
                context.restore_run_token,
                context.agent_id,
                context.parent_root_id,
                context.schedule_run_id,
            ),
        )
        connection.execute(
            """INSERT OR IGNORE INTO trace_records(
                record_id, root_id, workspace_key, workspace_path, origin,
                owner_session_id, subject_session_id, run_id, schedule_run_id,
                restore_run_token, agent_id, parent_root_id, parent_record_id,
                tool_call_id, kind, purpose, tool_name, model, reasoning_effort,
                endpoint, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                context.root_id,
                context.workspace_key,
                context.workspace_path,
                context.origin.value,
                context.owner_session_id,
                context.subject_session_id,
                context.run_id,
                context.schedule_run_id,
                context.restore_run_token,
                context.agent_id,
                context.parent_root_id,
                metadata.parent_record_id,
                metadata.tool_call_id,
                metadata.kind.value,
                metadata.purpose.value,
                metadata.tool_name,
                metadata.model,
                metadata.reasoning_effort,
                metadata.endpoint,
                _utc_text(metadata.started_at),
            ),
        )

    def _write_attempt(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> None:
        self._ensure_call_record(connection, record_id)
        metadata = cast(TraceAttemptMetadata, payload["metadata"])
        connection.execute(
            """INSERT OR IGNORE INTO trace_attempts(
                record_id, attempt_id, ordinal, started_at, model, reasoning_effort, endpoint
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                cast(str, payload["attempt_id"]),
                cast(int, payload["ordinal"]),
                _utc_text(metadata.started_at),
                metadata.model,
                metadata.reasoning_effort,
                metadata.endpoint,
            ),
        )

    def _write_request(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> None:
        self._ensure_call_record(connection, record_id)
        attempt_id = cast(str | None, payload["attempt_id"])
        try:
            request_json = _encode_json(payload["request"])
        except (TypeError, ValueError, OverflowError, RecursionError):
            self._insert_issue(
                connection,
                record_id,
                attempt_id,
                TraceCaptureIssue(
                    state=TraceCaptureState.PARTIAL,
                    message="Request content could not be serialized as JSON.",
                    source="request",
                ),
            )
            return
        if attempt_id is None:
            result = connection.execute(
                "UPDATE trace_records SET request_json = ?, request_set = 1 "
                "WHERE record_id = ? AND published = 0",
                (request_json, record_id),
            )
        else:
            result = connection.execute(
                "UPDATE trace_attempts SET request_json = ?, request_set = 1 "
                "WHERE record_id = ? AND attempt_id = ?",
                (request_json, record_id, attempt_id),
            )
        if result.rowcount != 1:
            raise sqlite3.IntegrityError("Trace request arrived before its call candidate")

    def _write_issue_event(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> None:
        self._ensure_call_record(connection, record_id)
        self._insert_issue(
            connection,
            record_id,
            cast(str | None, payload["attempt_id"]),
            cast(TraceCaptureIssue, payload["issue"]),
        )

    @staticmethod
    def _insert_issue(
        connection: sqlite3.Connection,
        record_id: str,
        attempt_id: str | None,
        issue: TraceCaptureIssue,
    ) -> None:
        existing = connection.execute(
            """SELECT 1 FROM trace_capture_issues
               WHERE record_id = ? AND attempt_id IS ? AND state = ? AND message = ?
                 AND source IS ? LIMIT 1""",
            (record_id, attempt_id, issue.state.value, issue.message, issue.source),
        ).fetchone()
        if existing is None:
            connection.execute(
                """INSERT INTO trace_capture_issues(record_id, attempt_id, state, message, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (record_id, attempt_id, issue.state.value, issue.message, issue.source),
            )

    def _write_raw(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> None:
        self._ensure_call_record(connection, record_id)
        attempt_id = cast(str | None, payload["attempt_id"])
        source = cast(str, payload["source"])
        direction = cast(TraceDirection, payload["direction"])
        layer = cast(TraceCaptureLayer, payload["layer"])
        stream = connection.execute(
            """SELECT stream_id, direction, layer, content_type, content_encoding
               FROM trace_raw_streams
               WHERE record_id = ? AND attempt_id IS ? AND source = ?""",
            (record_id, attempt_id, source),
        ).fetchone()
        content_type = cast(str | None, payload["content_type"])
        content_encoding = cast(str | None, payload["content_encoding"])
        if stream is None:
            cursor = connection.execute(
                """INSERT INTO trace_raw_streams(
                    record_id, attempt_id, source, direction, layer, content_type, content_encoding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    attempt_id,
                    source,
                    direction.value,
                    layer.value,
                    content_type,
                    content_encoding,
                ),
            )
            stream_id = int(cursor.lastrowid or 0)
        else:
            stream_id = int(stream["stream_id"])
            if (
                stream["direction"] != direction.value
                or stream["layer"] != layer.value
                or stream["content_type"] != content_type
                or stream["content_encoding"] != content_encoding
            ):
                SQLiteTraceRecorder._insert_issue(
                    connection,
                    record_id,
                    attempt_id,
                    TraceCaptureIssue(
                        state=TraceCaptureState.PARTIAL,
                        message="Raw source metadata changed during capture.",
                        source=source,
                    ),
                )
                return
        data = cast(bytes, payload["data"])
        if cast(int, payload["part"]) == 0:
            connection.execute(
                "DELETE FROM trace_raw_chunks WHERE stream_id = ? AND sequence = ?",
                (stream_id, cast(int, payload["sequence"])),
            )
        connection.execute(
            """INSERT OR REPLACE INTO trace_raw_chunks(
                stream_id, sequence, part, original_size, compressed_data
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                stream_id,
                cast(int, payload["sequence"]),
                cast(int, payload["part"]),
                len(data),
                zlib.compress(data),
            ),
        )

    def _ensure_call_record(self, connection: sqlite3.Connection, record_id: str) -> None:
        with self._condition:
            state = self._states.get(record_id)
        if state is None:
            raise sqlite3.IntegrityError("Trace call identity is no longer available")
        self._write_begin(
            connection,
            record_id,
            {"context": state.context, "metadata": state.metadata},
        )

    def _write_finish(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> bool:
        self._write_begin(
            connection,
            record_id,
            {"context": payload["context"], "metadata": payload["metadata"]},
        )
        row = connection.execute(
            "SELECT * FROM trace_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise sqlite3.IntegrityError("Trace call metadata was not available at finish")
        if row["published"]:
            return False

        outcome = cast(TraceOutcome, payload["outcome"])
        usage = cast(TraceTokenUsage, payload["usage"])
        metadata = cast(TraceCallMetadata, payload["metadata"])
        final_attempt_id = cast(str | None, payload["final_attempt_id"])
        final_attempt = cast(TraceAttemptMetadata | None, payload["final_attempt"])
        attempt_ordinal = cast(int, payload["final_attempt_ordinal"])
        if final_attempt_id is not None and final_attempt is not None:
            connection.execute(
                """INSERT OR IGNORE INTO trace_attempts(
                    record_id, attempt_id, ordinal, started_at, model, reasoning_effort, endpoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    final_attempt_id,
                    attempt_ordinal,
                    _utc_text(final_attempt.started_at),
                    final_attempt.model,
                    final_attempt.reasoning_effort,
                    final_attempt.endpoint,
                ),
            )

        successful_model = (
            outcome.status is TraceStatus.SUCCESS and metadata.kind is TraceKind.MODEL
        )
        if successful_model:
            connection.execute(
                """DELETE FROM trace_raw_streams
                   WHERE record_id = ? AND attempt_id IS NOT NULL AND attempt_id != ?""",
                (record_id, final_attempt_id),
            )
            selected_sources = connection.execute(
                "SELECT source FROM trace_raw_streams WHERE record_id = ? AND attempt_id = ?",
                (record_id, final_attempt_id),
            ).fetchall()
            for source_row in selected_sources:
                connection.execute(
                    "DELETE FROM trace_raw_streams WHERE record_id = ? AND attempt_id IS NULL AND source = ?",
                    (record_id, source_row["source"]),
                )
            connection.execute(
                "UPDATE trace_raw_streams SET attempt_id = NULL "
                "WHERE record_id = ? AND attempt_id = ?",
                (record_id, final_attempt_id),
            )
            connection.execute(
                "DELETE FROM trace_capture_issues WHERE record_id = ? "
                "AND attempt_id IS NOT NULL AND attempt_id != ?",
                (record_id, final_attempt_id),
            )
            connection.execute(
                "UPDATE trace_capture_issues SET attempt_id = NULL "
                "WHERE record_id = ? AND attempt_id = ?",
                (record_id, final_attempt_id),
            )
        elif metadata.kind is TraceKind.MODEL:
            connection.execute(
                "DELETE FROM trace_raw_streams WHERE record_id = ?",
                (record_id,),
            )
            connection.execute(
                "DELETE FROM trace_capture_issues WHERE record_id = ? AND attempt_id IS NOT NULL",
                (record_id,),
            )
        else:
            connection.execute(
                "UPDATE trace_capture_issues SET attempt_id = NULL "
                "WHERE record_id = ? AND attempt_id IS NOT NULL",
                (record_id,),
            )

        with self._condition:
            state = self._states.get(record_id)
            pending_issues = tuple(state.issues.items()) if state is not None else ()
        final_issues: list[tuple[str | None, TraceCaptureIssue]] = []
        for (issue_attempt_id, _, _, _), issue in pending_issues:
            if issue_attempt_id is None or issue_attempt_id == final_attempt_id:
                final_issues.append((issue_attempt_id, issue))
        for issue_attempt_id, issue in cast(
            tuple[tuple[str | None, TraceCaptureIssue], ...], payload["issues"]
        ):
            if issue_attempt_id is None or issue_attempt_id == final_attempt_id:
                final_issues.append((issue_attempt_id, issue))
        for issue_attempt_id, issue in final_issues:
            normalized_attempt_id = None if successful_model else issue_attempt_id
            if metadata.kind is TraceKind.MODEL and not successful_model and issue_attempt_id:
                continue
            self._insert_issue(connection, record_id, normalized_attempt_id, issue)

        request_json: str | None = None
        request_set = 0
        if successful_model and final_attempt_id is not None:
            request = connection.execute(
                """SELECT request_json, request_set FROM trace_attempts
                   WHERE record_id = ? AND attempt_id = ?""",
                (record_id, final_attempt_id),
            ).fetchone()
            if request is not None:
                request_json = cast(str | None, request["request_json"])
                request_set = int(request["request_set"])
        elif metadata.kind is TraceKind.TOOL:
            request_json = cast(str | None, row["request_json"])
            request_set = int(row["request_set"])

        result_json: str | None = None
        result_set = 0
        if outcome.status is TraceStatus.SUCCESS or metadata.kind is TraceKind.TOOL:
            try:
                result_json = _encode_json(payload["result"])
                result_set = 1
            except (TypeError, ValueError, OverflowError, RecursionError):
                self._insert_issue(
                    connection,
                    record_id,
                    None,
                    TraceCaptureIssue(
                        state=TraceCaptureState.PARTIAL,
                        message="Result content could not be serialized as JSON.",
                        source="result",
                    ),
                )

        issue_rows = connection.execute(
            "SELECT state FROM trace_capture_issues WHERE record_id = ?",
            (record_id,),
        ).fetchall()
        raw_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM trace_raw_streams WHERE record_id = ? AND attempt_id IS NULL",
                (record_id,),
            ).fetchone()[0]
        )
        has_content = bool(request_set or result_set or raw_count)
        capture_state = (
            TraceCaptureState.COMPLETE
            if not issue_rows
            else TraceCaptureState.PARTIAL
            if has_content
            else TraceCaptureState.UNAVAILABLE
        )

        selected_metadata = (
            final_attempt if metadata.kind is TraceKind.MODEL and final_attempt else metadata
        )
        connection.execute(
            "UPDATE trace_meta SET publish_sequence = publish_sequence + 1 WHERE singleton = 1"
        )
        publish_sequence = int(
            connection.execute(
                "SELECT publish_sequence FROM trace_meta WHERE singleton = 1"
            ).fetchone()[0]
        )
        updated = connection.execute(
            """UPDATE trace_records SET
                model = ?, reasoning_effort = ?, endpoint = ?,
                status = ?, finished_at = ?, duration_ms = ?, final_attempt_duration_ms = ?,
                retry_count = ?, error_category = ?, error_code = ?, error_message = ?,
                usage_state = ?, input_tokens = ?, output_tokens = ?, total_tokens = ?,
                cached_input_tokens = ?, cache_creation_input_tokens = ?,
                request_json = ?, request_set = ?, result_json = ?, result_set = ?,
                capture_state = ?, published = 1, publish_sequence = ?
               WHERE record_id = ? AND published = 0""",
            (
                selected_metadata.model,
                selected_metadata.reasoning_effort,
                selected_metadata.endpoint,
                outcome.status.value,
                _utc_text(outcome.finished_at),
                outcome.duration_ms,
                outcome.final_attempt_duration_ms,
                max(0, cast(int, payload["retry_count"])),
                _enum_value(outcome.error_category),
                outcome.error_code,
                outcome.error_message,
                usage.state.value,
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                request_json,
                request_set,
                result_json,
                result_set,
                capture_state.value,
                publish_sequence,
                record_id,
            ),
        )
        connection.execute("DELETE FROM trace_attempts WHERE record_id = ?", (record_id,))
        return updated.rowcount == 1

    @staticmethod
    def _write_restore(connection: sqlite3.Connection, payload: dict[str, object]) -> None:
        tokens = cast(tuple[str, ...], payload["tokens"])
        if not tokens:
            return
        connection.executemany(
            """UPDATE trace_records SET restored = 1
               WHERE workspace_key = ? AND owner_session_id = ? AND restore_run_token = ?
                 AND published = 1""",
            [(payload["workspace_key"], payload["owner_session_id"], token) for token in tokens],
        )

    @staticmethod
    def _write_purge(connection: sqlite3.Connection, payload: dict[str, object]) -> None:
        connection.execute(
            "DELETE FROM trace_records WHERE workspace_key = ? AND owner_session_id = ?",
            (payload["workspace_key"], payload["owner_session_id"]),
        )
        connection.execute(
            "DELETE FROM trace_roots WHERE NOT EXISTS "
            "(SELECT 1 FROM trace_records WHERE trace_records.root_id = trace_roots.root_id)"
        )

    @staticmethod
    def _write_expired(connection: sqlite3.Connection, payload: dict[str, object]) -> int:
        deleted = connection.execute(
            """DELETE FROM trace_records
               WHERE record_id IN (
                   SELECT record_id FROM trace_records
                   WHERE published = 1 AND finished_at < ?
                   ORDER BY finished_at LIMIT ?
               )""",
            (payload["cutoff"], _PURGE_BATCH_SIZE),
        ).rowcount
        if deleted:
            connection.execute(
                "DELETE FROM trace_roots WHERE NOT EXISTS "
                "(SELECT 1 FROM trace_records WHERE trace_records.root_id = trace_roots.root_id)"
            )
        return deleted

    def _list_records_sync(
        self,
        filters: TraceRecordFilters | None,
        cursor: str | None,
        limit: int,
    ) -> TracePage:
        connection = self._connect()
        try:
            visibility_sql, visibility_parameters = self._record_visibility_sql(connection)
            if cursor is None:
                snapshot = int(
                    connection.execute(
                        "SELECT publish_sequence FROM trace_meta WHERE singleton = 1"
                    ).fetchone()[0]
                )
                position_sql = ""
                position_parameters: tuple[object, ...] = ()
            else:
                snapshot, finished_at, record_id = _record_cursor_decode(cursor)
                position_sql = " AND (r.finished_at, r.record_id) < (?, ?)"
                position_parameters = (finished_at, record_id)
            filter_sql, filter_parameters = _filter_sql(filters)
            rows = connection.execute(
                """SELECT r.record_id, r.root_id, r.workspace_key, r.workspace_path,
                          r.origin, r.purpose, r.kind, r.status, r.started_at, r.finished_at,
                          r.duration_ms, r.retry_count, r.capture_state, r.usage_state,
                          r.input_tokens, r.output_tokens, r.total_tokens, r.cached_input_tokens,
                          r.cache_creation_input_tokens, r.owner_session_id, r.subject_session_id,
                          r.run_id, r.schedule_run_id, r.restore_run_token, r.agent_id,
                          r.parent_root_id, r.parent_record_id, r.tool_call_id, r.tool_name,
                          r.model, r.reasoning_effort, r.endpoint, r.final_attempt_duration_ms,
                          r.error_category, r.error_code, r.error_message, r.restored
                   FROM trace_records AS r
                   WHERE r.published = 1 AND r.publish_sequence <= ?"""
                + visibility_sql
                + position_sql
                + filter_sql
                + " ORDER BY r.finished_at DESC, r.record_id DESC LIMIT ?",
                (
                    snapshot,
                    *visibility_parameters,
                    *position_parameters,
                    *filter_parameters,
                    limit + 1,
                ),
            ).fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            next_cursor = (
                _record_cursor_encode(
                    snapshot, page_rows[-1]["finished_at"], page_rows[-1]["record_id"]
                )
                if has_more and page_rows
                else None
            )
            return TracePage(
                items=tuple(self._summary_from_row(row) for row in page_rows),
                next_cursor=next_cursor,
                snapshot_sequence=snapshot,
            )
        finally:
            connection.close()

    def _list_sessions_sync(
        self,
        filters: TraceRecordFilters | None,
        cursor: str | None,
        limit: int,
    ) -> TraceSessionPage:
        connection = self._connect()
        try:
            visibility_sql, visibility_parameters = self._record_visibility_sql(connection)
            if cursor is None:
                snapshot = int(
                    connection.execute(
                        "SELECT publish_sequence FROM trace_meta WHERE singleton = 1"
                    ).fetchone()[0]
                )
                last = snapshot + 1
            else:
                snapshot, last = _cursor_decode(cursor)
            filter_sql, filter_parameters = _filter_sql(filters)
            rows = connection.execute(
                """SELECT r.workspace_key, r.workspace_path, r.owner_session_id,
                          r.subject_session_id, r.origin, MAX(r.finished_at) AS last_finished_at,
                          COUNT(*) AS record_count, MAX(r.publish_sequence) AS group_sequence
                   FROM trace_records AS r
                   WHERE r.published = 1 AND r.owner_session_id IS NOT NULL
                     AND r.publish_sequence <= ?"""
                + visibility_sql
                + filter_sql
                + " GROUP BY r.workspace_key, r.owner_session_id, r.subject_session_id, r.origin "
                " HAVING MAX(r.publish_sequence) < ?"
                " ORDER BY group_sequence DESC LIMIT ?",
                (snapshot, *visibility_parameters, *filter_parameters, last, limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            next_cursor = (
                _cursor_encode(snapshot, int(page_rows[-1]["group_sequence"]))
                if has_more and page_rows
                else None
            )
            return TraceSessionPage(
                items=tuple(
                    TraceSessionSummary(
                        workspace_key=cast(str, row["workspace_key"]),
                        workspace_path=cast(str, row["workspace_path"]),
                        owner_session_id=cast(str, row["owner_session_id"]),
                        subject_session_id=cast(str | None, row["subject_session_id"]),
                        origin=TraceOrigin(row["origin"]),
                        last_finished_at=_parse_utc(row["last_finished_at"]),
                        record_count=int(row["record_count"]),
                    )
                    for row in page_rows
                ),
                next_cursor=next_cursor,
            )
        finally:
            connection.close()

    def _get_record_sync(self, record_id: str) -> TraceRecordDetail | None:
        connection = self._connect()
        try:
            visibility_sql, visibility_parameters = self._record_visibility_sql(connection)
            row = connection.execute(
                "SELECT r.* FROM trace_records AS r "
                "WHERE r.record_id = ? AND r.published = 1" + visibility_sql,
                (record_id, *visibility_parameters),
            ).fetchone()
            if row is None:
                return None
            streams = connection.execute(
                """SELECT s.source, s.direction, s.layer, s.content_type, s.content_encoding,
                          COALESCE(SUM(c.original_size), 0) AS total_bytes
                   FROM trace_raw_streams AS s
                   LEFT JOIN trace_raw_chunks AS c ON c.stream_id = s.stream_id
                   WHERE s.record_id = ? AND s.attempt_id IS NULL
                   GROUP BY s.stream_id ORDER BY s.source""",
                (record_id,),
            ).fetchall()
            issues = connection.execute(
                """SELECT state, message, source FROM trace_capture_issues
                   WHERE record_id = ? ORDER BY issue_id""",
                (record_id,),
            ).fetchall()
            return TraceRecordDetail(
                record=self._summary_from_row(row),
                request=(
                    json.loads(row["request_json"])
                    if row["request_set"] and row["request_json"] is not None
                    else None
                ),
                result=(
                    json.loads(row["result_json"])
                    if row["result_set"] and row["result_json"] is not None
                    else None
                ),
                raw_sources=tuple(
                    TraceRawSource(
                        source=cast(str, stream["source"]),
                        direction=TraceDirection(stream["direction"]),
                        layer=TraceCaptureLayer(stream["layer"]),
                        content_type=cast(str | None, stream["content_type"]),
                        content_encoding=cast(str | None, stream["content_encoding"]),
                        total_bytes=int(stream["total_bytes"]),
                    )
                    for stream in streams
                ),
                capture_issues=tuple(
                    TraceCaptureIssue(
                        state=TraceCaptureState(issue["state"]),
                        message=issue["message"],
                        source=issue["source"],
                    )
                    for issue in issues
                ),
            )
        finally:
            connection.close()

    def _read_raw_sync(
        self,
        record_id: str,
        source: str,
        offset: int,
        limit_bytes: int,
    ) -> TraceRawPage:
        connection = self._connect()
        try:
            visibility_sql, visibility_parameters = self._record_visibility_sql(connection)
            stream = connection.execute(
                """SELECT s.stream_id, s.direction, s.layer, s.content_type, s.content_encoding,
                          COALESCE(SUM(c.original_size), 0) AS total_bytes
                   FROM trace_raw_streams AS s
                   JOIN trace_records AS r ON r.record_id = s.record_id
                   LEFT JOIN trace_raw_chunks AS c ON c.stream_id = s.stream_id
                   WHERE s.record_id = ? AND s.source = ? AND s.attempt_id IS NULL
                     AND r.published = 1"""
                + visibility_sql
                + " GROUP BY s.stream_id",
                (record_id, source, *visibility_parameters),
            ).fetchone()
            if stream is None:
                return TraceRawPage(
                    source=source,
                    direction=None,
                    layer=None,
                    offset=offset,
                    data=b"",
                    total_bytes=0,
                    next_offset=None,
                )
            total_bytes = int(stream["total_bytes"])
            output = bytearray()
            end_offset = offset + limit_bytes
            position = 0
            if offset < total_bytes:
                chunks = connection.execute(
                    """SELECT sequence, part, original_size FROM trace_raw_chunks
                       WHERE stream_id = ? ORDER BY sequence, part""",
                    (stream["stream_id"],),
                )
                for chunk in chunks:
                    chunk_end = position + int(chunk["original_size"])
                    if chunk_end > position and chunk_end > offset and position < end_offset:
                        body = connection.execute(
                            """SELECT compressed_data FROM trace_raw_chunks
                               WHERE stream_id = ? AND sequence = ? AND part = ?""",
                            (stream["stream_id"], chunk["sequence"], chunk["part"]),
                        ).fetchone()
                        if body is None:
                            break
                        data = zlib.decompress(body["compressed_data"])
                        start = max(0, offset - position)
                        stop = min(len(data), end_offset - position)
                        output.extend(data[start:stop])
                    position = chunk_end
                    if position >= end_offset:
                        break
            next_offset = offset + len(output)
            return TraceRawPage(
                source=source,
                direction=TraceDirection(stream["direction"]),
                layer=TraceCaptureLayer(stream["layer"]),
                offset=offset,
                data=bytes(output),
                total_bytes=total_bytes,
                next_offset=next_offset if next_offset < total_bytes else None,
                content_type=cast(str | None, stream["content_type"]),
                content_encoding=cast(str | None, stream["content_encoding"]),
            )
        finally:
            connection.close()

    def _get_stats_sync(self, filters: TraceRecordFilters | None) -> TraceStats:
        connection = self._connect()
        total = model_calls = tool_calls = success_count = failed_count = 0
        retry_count = retried_calls = unknown_usage_count = 0
        duration_total = duration_count = 0
        token_totals = [0] * len(_TRACE_TOKEN_FIELDS)
        daily_values: dict[str, list[int]] = {}
        model_values: dict[str, list[int]] = {}
        tool_values: dict[str, list[int]] = {}
        purpose_values: dict[str, list[int]] = {}
        error_values: dict[str, int] = {}

        def add_breakdown(
            values: dict[str, list[int]], key: str | None, calls: int, tokens: Sequence[int]
        ) -> None:
            if key is None:
                return
            aggregate = values.setdefault(key, [0] * (len(_TRACE_TOKEN_FIELDS) + 1))
            aggregate[0] += calls
            for index, value in enumerate(tokens, start=1):
                aggregate[index] += value

        try:
            connection.execute("PRAGMA temp_store = MEMORY")
            connection.row_factory = None
            visibility_sql, visibility_parameters = self._record_visibility_sql(connection)
            filter_sql, filter_parameters = _filter_sql(filters)
            token_sum_sql = ", ".join(
                "COALESCE(SUM(CASE WHEN r.kind = 'model' AND r.usage_state = 'reported' "
                f"THEN COALESCE(r.{field}, 0) ELSE 0 END), 0) AS {field}"
                for field in _TRACE_TOKEN_FIELDS
            )
            rows = connection.execute(
                "SELECT substr(r.finished_at, 1, 10) AS date, r.model, r.tool_name, "
                "r.purpose, r.error_category, COUNT(*) AS calls, "
                "COALESCE(SUM(CASE WHEN r.status <> 'success' "
                "AND r.error_category IS NOT NULL THEN 1 ELSE 0 END), 0) AS error_calls, "
                "COALESCE(SUM(r.kind = 'model'), 0) AS model_calls, "
                "COALESCE(SUM(r.kind = 'tool'), 0) AS tool_calls, "
                "COALESCE(SUM(r.status = 'success'), 0) AS successful_calls, "
                "COALESCE(SUM(r.status <> 'success'), 0) AS failed_calls, "
                "COALESCE(SUM(r.retry_count), 0) AS retry_count, "
                "COALESCE(SUM(r.retry_count > 0), 0) AS retried_calls, "
                "COALESCE(SUM(r.kind = 'model' AND r.usage_state = 'unknown'), 0) "
                "AS unknown_usage_count, COALESCE(SUM(r.duration_ms), 0) AS duration_total, "
                "COUNT(r.duration_ms) AS duration_count, "
                + token_sum_sql
                + " FROM trace_records AS r WHERE r.published = 1"
                + visibility_sql
                + filter_sql
                + " GROUP BY substr(r.finished_at, 1, 10), r.model, r.tool_name, "
                "r.purpose, r.error_category",
                (*visibility_parameters, *filter_parameters),
            )
            for row in rows:
                date = cast(str, row[0])
                model = cast(str | None, row[1])
                tool_name = cast(str | None, row[2])
                purpose = cast(str | None, row[3])
                category = cast(str | None, row[4])
                calls = int(row[5])
                error_calls = int(row[6])
                row_model_calls = int(row[7])
                row_tool_calls = int(row[8])
                row_success_count = int(row[9])
                row_failed_count = int(row[10])
                row_retry_count = int(row[11])
                row_retried_calls = int(row[12])
                row_unknown_usage_count = int(row[13])
                row_duration_total = int(row[14])
                row_duration_count = int(row[15])
                row_tokens = [int(row[16 + index]) for index in range(len(_TRACE_TOKEN_FIELDS))]

                total += calls
                model_calls += row_model_calls
                tool_calls += row_tool_calls
                success_count += row_success_count
                failed_count += row_failed_count
                retry_count += row_retry_count
                retried_calls += row_retried_calls
                unknown_usage_count += row_unknown_usage_count
                duration_total += row_duration_total
                duration_count += row_duration_count
                for index, value in enumerate(row_tokens):
                    token_totals[index] += value

                day = daily_values.setdefault(date, [0] * (4 + len(_TRACE_TOKEN_FIELDS)))
                day[0] += row_model_calls
                day[1] += row_tool_calls
                day[2] += row_success_count
                day[3] += row_failed_count
                for index, value in enumerate(row_tokens, start=4):
                    day[index] += value

                add_breakdown(model_values, model, calls, row_tokens)
                add_breakdown(tool_values, tool_name, calls, row_tokens)
                add_breakdown(purpose_values, purpose, calls, row_tokens)
                if category is not None and error_calls:
                    error_values[category] = error_values.get(category, 0) + error_calls
        finally:
            connection.close()

        failure_rate = failed_count / total if total else None
        average_duration = duration_total / duration_count if duration_count else None

        return TraceStats(
            totals=TraceStatsTotals(
                model_calls=model_calls,
                tool_calls=tool_calls,
                success_count=success_count,
                failed_count=failed_count,
                failure_rate=failure_rate,
                retry_count=retry_count,
                retried_calls=retried_calls,
                unknown_usage_count=unknown_usage_count,
                average_duration_ms=average_duration,
            ),
            tokens=TraceTokenTotals(*token_totals),
            by_model=self._breakdowns(model_values),
            by_tool=self._breakdowns(tool_values),
            by_purpose=self._breakdowns(purpose_values),
            by_error_category=tuple(
                TraceErrorBreakdown(category=TraceErrorCategory(category), calls=calls)
                for category, calls in sorted(error_values.items())
            ),
            daily=tuple(
                TraceTimelinePoint(
                    date=date,
                    model_calls=values[0],
                    tool_calls=values[1],
                    successful_calls=values[2],
                    failed_calls=values[3],
                    tokens=TraceTokenTotals(*values[4:]),
                )
                for date, values in sorted(daily_values.items())
            ),
            methodology=_METHODLOGY,
        )

    @staticmethod
    def _breakdowns(values: dict[str, list[int]]) -> tuple[TraceBreakdown, ...]:
        return tuple(
            TraceBreakdown(
                key=key,
                calls=values[key][0],
                tokens=TraceTokenTotals(*values[key][1:]),
            )
            for key in sorted(values, key=lambda item: (-values[item][0], item))
        )

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> TraceRecordSummary:
        effort = row["reasoning_effort"]
        if effort is not None and effort not in REASONING_EFFORT_LEVELS:
            raise ValueError("Trace reasoning_effort is unsupported")
        return TraceRecordSummary(
            record_id=row["record_id"],
            root_id=row["root_id"],
            workspace_key=row["workspace_key"],
            workspace_path=row["workspace_path"],
            origin=TraceOrigin(row["origin"]),
            purpose=TracePurpose(row["purpose"]),
            kind=TraceKind(row["kind"]),
            status=TraceStatus(row["status"]),
            started_at=_parse_utc(row["started_at"]),
            finished_at=_parse_utc(row["finished_at"]),
            duration_ms=int(row["duration_ms"]),
            retry_count=int(row["retry_count"]),
            capture_state=TraceCaptureState(row["capture_state"]),
            usage=TraceTokenUsage(
                state=TraceUsageState(row["usage_state"]),
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                total_tokens=row["total_tokens"],
                cached_input_tokens=row["cached_input_tokens"],
                cache_creation_input_tokens=row["cache_creation_input_tokens"],
            ),
            owner_session_id=row["owner_session_id"],
            subject_session_id=row["subject_session_id"],
            run_id=row["run_id"],
            schedule_run_id=row["schedule_run_id"],
            restore_run_token=row["restore_run_token"],
            agent_id=row["agent_id"],
            parent_root_id=row["parent_root_id"],
            parent_record_id=row["parent_record_id"],
            tool_call_id=row["tool_call_id"],
            tool_name=row["tool_name"],
            model=row["model"],
            reasoning_effort=cast(ReasoningEffort | None, effort),
            endpoint=row["endpoint"],
            final_attempt_duration_ms=row["final_attempt_duration_ms"],
            error_category=(
                TraceErrorCategory(row["error_category"])
                if row["error_category"] is not None
                else None
            ),
            error_code=row["error_code"],
            error_message=row["error_message"],
            restored=bool(row["restored"]),
        )
