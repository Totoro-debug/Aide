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
    TraceMetadataHandle,
    TraceOutcome,
    TracePage,
    TraceRecordDetail,
    TraceRecordFilters,
    TraceSessionPage,
    TraceStats,
    TraceTokenUsage,
)


class TraceRecorder(Protocol):
    """Best-effort semantic outcomes; writes never wait for disk.

    Only Tool arguments use set_request. Model inputs and protocol payloads are
    never captured. Attempts select final routing metadata and issues; partial
    model output survives failure independently of capture completeness.
    """

    def new_root(self, source: TraceRootSource) -> TraceContext: ...

    def begin_call(
        self, context: TraceContext, metadata: TraceCallMetadata
    ) -> TraceCaptureHandle: ...

    def begin_attempt(
        self, handle: TraceCaptureHandle, metadata: TraceAttemptMetadata
    ) -> TraceAttemptHandle: ...

    def set_endpoint(self, handle: TraceMetadataHandle, endpoint: str) -> None: ...

    def set_protocol(self, handle: TraceMetadataHandle, protocol: str) -> None: ...

    def set_request(self, handle: TraceMetadataHandle, request: object) -> None: ...

    def report_capture_issue(self, handle: TraceMetadataHandle, issue: TraceCaptureIssue) -> None: ...


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

    def set_endpoint(self, handle: TraceMetadataHandle, endpoint: str) -> None:
        del handle, endpoint

    def set_protocol(self, handle: TraceMetadataHandle, protocol: str) -> None:
        del handle, protocol

    def set_request(self, handle: TraceMetadataHandle, request: object) -> None:
        del handle, request

    def report_capture_issue(self, handle: TraceMetadataHandle, issue: TraceCaptureIssue) -> None:
        del handle, issue


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


    async def get_stats(self, filters: TraceRecordFilters | None = None) -> TraceStats:
        del filters
        return TraceStats()

    async def close(self, timeout_seconds: float = 2.0) -> None:
        del timeout_seconds


__all__ = ["NoOpTraceRecorder", "TraceRecorder"]
