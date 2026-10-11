"""Execution trace contracts and the default no-op recorder."""

from aide.trace.context import TraceContext, TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder, TraceRecorder
from aide.trace.sqlite_recorder import (
    TRACE_MAX_QUEUE_EVENTS,
    TRACE_MAX_QUEUE_PAYLOAD_BYTES,
    TRACE_MAX_RAW_CHUNK_BYTES,
    SQLiteTraceRecorder,
    TraceRecorderDiagnostics,
)
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceCaptureIssue,
    TraceCaptureLayer,
    TraceCaptureState,
    TraceDirection,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TracePurpose,
    TraceReasoningEffort,
    TraceSessionPage,
    TraceStatus,
    TraceUsageState,
)

__all__ = [
    "TRACE_MAX_QUEUE_EVENTS",
    "TRACE_MAX_QUEUE_PAYLOAD_BYTES",
    "TRACE_MAX_RAW_CHUNK_BYTES",
    "NoOpTraceRecorder",
    "SQLiteTraceRecorder",
    "TraceAttemptMetadata",
    "TraceCallMetadata",
    "TraceCaptureIssue",
    "TraceCaptureLayer",
    "TraceCaptureState",
    "TraceContext",
    "TraceDirection",
    "TraceErrorCategory",
    "TraceKind",
    "TraceOrigin",
    "TracePurpose",
    "TraceReasoningEffort",
    "TraceRecorder",
    "TraceRecorderDiagnostics",
    "TraceRootSource",
    "TraceSessionPage",
    "TraceStatus",
    "TraceUsageState",
    "workspace_key_for",
]
