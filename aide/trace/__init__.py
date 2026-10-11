"""Execution trace contracts and the default no-op recorder."""

from aide.trace.context import TraceContext, TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder, TraceRecorder
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
    "NoOpTraceRecorder",
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
    "TraceRootSource",
    "TraceSessionPage",
    "TraceStatus",
    "TraceUsageState",
    "workspace_key_for",
]
