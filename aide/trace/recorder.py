"""Recorder boundary and behavior-preserving empty implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import uuid4

from aide.trace.context import TraceContext, TraceRootSource
from aide.trace.types import (
    TraceAttemptHandle,
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureLayer,
    TraceDirection,
    TraceOutcome,
    TracePage,
    TraceRawHandle,
    TraceRawPage,
    TraceRecordDetail,
    TraceRecordFilters,
    TraceSessionPage,
    TraceStats,
    TraceTokenUsage,
)


class TraceRecorder(Protocol):
    """Capture never waits for disk or replaces business outcomes; reads/cleanup are async.

    Requests and issues bind to a Tool call or a specific model attempt. The
    adapter supplies detached JSON-compatible request/result snapshots; the recorder
    does not interpret adapter payloads. Only the successful final model attempt's
    request, raw sources and issues are published. Ultimately failed model calls
    discard request/response candidates and retain errors and metadata.
    Failed Tools retain their arguments and the output already captured.

    A source names a stream within a record; its layer states the capture boundary:
    HTTP body before/after content decoding, JSON-RPC message, SDK result (DDGS),
    process stream (Exec), or Tool result before formatting. Issues degrade capture
    completeness independently of execution status; writer losses can also add
    issues. An unavailable source alongside other captured sources makes the
    record partial; unavailable at record level means no capture is available.
    A lack of retained model payloads by policy is not a capture failure.

    Query DTOs encode datetimes as UTC Z strings, error_* fields as one error
    object, detail request/result as content, and raw data as data_base64. Sources,
    layers, capture issues and other metadata retain their names and meanings.
    """

    def new_root(self, source: TraceRootSource) -> TraceContext: ...

    def begin_call(
        self, context: TraceContext, metadata: TraceCallMetadata
    ) -> TraceCaptureHandle: ...

    def begin_attempt(
        self, handle: TraceCaptureHandle, metadata: TraceAttemptMetadata
    ) -> TraceAttemptHandle: ...

    def set_request(self, handle: TraceRawHandle, request: object) -> None: ...

    def report_capture_issue(self, handle: TraceRawHandle, issue: TraceCaptureIssue) -> None: ...

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
    ) -> None: ...

    def finish_call(
        self,
        handle: TraceCaptureHandle,
        outcome: TraceOutcome,
        *,
        result: object,
        usage: TraceTokenUsage,
        retry_count: int,
    ) -> None: ...

    def close_session_scope(self, workspace_key: str, owner_session_id: str) -> None: ...

    async def purge_session(self, workspace_key: str, owner_session_id: str) -> None: ...

    async def purge_expired(self) -> None: ...

    def mark_restored(
        self,
        workspace_key: str,
        owner_session_id: str,
        discarded_run_tokens: Sequence[str],
    ) -> None: ...

    async def list_sessions(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TraceSessionPage: ...

    async def list_records(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TracePage: ...

    async def get_record(self, record_id: str) -> TraceRecordDetail | None: ...

    async def read_raw(
        self,
        record_id: str,
        *,
        source: str,
        offset: int,
        limit_bytes: int,
    ) -> TraceRawPage: ...

    async def get_stats(self, filters: TraceRecordFilters | None = None) -> TraceStats: ...

    async def close(self, timeout_seconds: float = 2.0) -> None: ...


class NoOpTraceRecorder:
    """Drop trace data while returning the same handles as a real recorder."""

    def new_root(self, source: TraceRootSource) -> TraceContext:
        return TraceContext.create(str(uuid4()), source)

    def begin_call(self, context: TraceContext, metadata: TraceCallMetadata) -> TraceCaptureHandle:
        del metadata
        return TraceCaptureHandle(record_id=str(uuid4()), root_id=context.root_id)

    def begin_attempt(
        self, handle: TraceCaptureHandle, metadata: TraceAttemptMetadata
    ) -> TraceAttemptHandle:
        del metadata
        return TraceAttemptHandle(record_id=handle.record_id, attempt_id=str(uuid4()))

    def set_request(self, handle: TraceRawHandle, request: object) -> None:
        del handle, request

    def report_capture_issue(self, handle: TraceRawHandle, issue: TraceCaptureIssue) -> None:
        del handle, issue

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
        del handle, source, direction, layer, sequence, data, content_type, content_encoding

    def finish_call(
        self,
        handle: TraceCaptureHandle,
        outcome: TraceOutcome,
        *,
        result: object,
        usage: TraceTokenUsage,
        retry_count: int,
    ) -> None:
        del handle, outcome, result, usage, retry_count

    def close_session_scope(self, workspace_key: str, owner_session_id: str) -> None:
        del workspace_key, owner_session_id

    async def purge_session(self, workspace_key: str, owner_session_id: str) -> None:
        del workspace_key, owner_session_id

    async def purge_expired(self) -> None:
        return None

    def mark_restored(
        self,
        workspace_key: str,
        owner_session_id: str,
        discarded_run_tokens: Sequence[str],
    ) -> None:
        del workspace_key, owner_session_id, discarded_run_tokens

    async def list_sessions(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TraceSessionPage:
        del filters, cursor, limit
        return TraceSessionPage()

    async def list_records(
        self,
        filters: TraceRecordFilters | None = None,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> TracePage:
        del filters, cursor, limit
        return TracePage()

    async def get_record(self, record_id: str) -> TraceRecordDetail | None:
        del record_id
        return None

    async def read_raw(
        self,
        record_id: str,
        *,
        source: str,
        offset: int,
        limit_bytes: int,
    ) -> TraceRawPage:
        del record_id, limit_bytes
        return TraceRawPage(
            source=source,
            direction=None,
            layer=None,
            offset=offset,
            data=b"",
            total_bytes=0,
            next_offset=None,
        )

    async def get_stats(self, filters: TraceRecordFilters | None = None) -> TraceStats:
        del filters
        return TraceStats()

    async def close(self, timeout_seconds: float = 2.0) -> None:
        del timeout_seconds


__all__ = ["NoOpTraceRecorder", "TraceRecorder"]
