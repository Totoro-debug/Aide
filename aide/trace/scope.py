"""Task-local execution identity for model and Tool calls in one Agent Run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from aide.trace.context import TraceContext, TraceRootSource
from aide.trace.sanitization import sanitize_trace_url
from aide.trace.types import (
    TraceAttemptHandle,
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureState,
    TracePurpose,
)

if TYPE_CHECKING:
    from aide.trace.recorder import TraceRecorder


@dataclass(frozen=True, slots=True)
class TraceExecutionScope:
    recorder: TraceRecorder
    context: TraceContext
    purpose: TracePurpose
    parent_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class TraceToolCallScope:
    recorder: TraceRecorder
    handle: TraceCaptureHandle

    def set_protocol(self, protocol: str) -> None:
        try:
            self.recorder.set_protocol(self.handle, protocol)
        except Exception:
            self.report_capture_issue("Tool protocol metadata could not be captured.", source="protocol")

    def set_endpoint(self, endpoint: str) -> None:
        try:
            self.recorder.set_endpoint(self.handle, sanitize_trace_url(endpoint))
        except Exception:
            self.report_capture_issue("Tool endpoint metadata could not be captured.",
                                      source="tool_endpoint")


    def report_capture_issue(self, message: str, *, source: str) -> None:
        try:
            self.recorder.report_capture_issue(
                self.handle,
                TraceCaptureIssue(
                    state=TraceCaptureState.PARTIAL,
                    message=message,
                    source=source,
                ),
            )
        except Exception:
            logger.warning("Execution Trace Tool capture issue could not be reported")


@dataclass(slots=True)
class TraceProviderAttempt:
    recorder: TraceRecorder
    handle: TraceAttemptHandle


    def set_endpoint(self, endpoint: str) -> None:
        try:
            self.recorder.set_endpoint(self.handle, endpoint)
        except Exception:
            self.report_capture_issue("Provider endpoint metadata could not be captured.")


    def report_capture_issue(self, message: str) -> None:
        try:
            self.recorder.report_capture_issue(
                self.handle,
                TraceCaptureIssue(
                    state=TraceCaptureState.PARTIAL,
                    message=message,
                    source="provider_transport",
                ),
            )
        except Exception:
            logger.warning("Execution Trace provider capture issue could not be reported")


_ACTIVE_TRACE_SCOPE: ContextVar[TraceExecutionScope | None] = ContextVar(
    "aide_active_trace_scope", default=None
)
_ACTIVE_TRACE_TOOL_CALL: ContextVar[TraceToolCallScope | None] = ContextVar(
    "aide_active_trace_tool_call", default=None
)
_ACTIVE_TRACE_PROVIDER_ATTEMPT: ContextVar[TraceProviderAttempt | None] = ContextVar(
    "aide_active_trace_provider_attempt", default=None
)


def current_trace_scope() -> TraceExecutionScope | None:
    """Return the immutable trace identity active in this task, if any."""
    return _ACTIVE_TRACE_SCOPE.get()


def current_trace_tool_call() -> TraceToolCallScope | None:
    """Return the immutable capture handle active during one Tool execution."""
    return _ACTIVE_TRACE_TOOL_CALL.get()


def current_trace_provider_attempt() -> TraceProviderAttempt | None:
    """Return the provider attempt active in this task, if any."""
    return _ACTIVE_TRACE_PROVIDER_ATTEMPT.get()


def set_trace_parent_record_id(record_id: str, *, scope: TraceExecutionScope) -> None:
    """Bind the latest model record as parent for following Tool calls in this task."""
    current = _ACTIVE_TRACE_SCOPE.get()
    if (
        current is not None
        and current.context == scope.context
        and current.recorder is scope.recorder
    ):
        _ACTIVE_TRACE_SCOPE.set(replace(current, parent_record_id=record_id))


@contextmanager
def trace_tool_call(scope: TraceToolCallScope | None) -> Iterator[None]:
    """Bind one Tool capture handle through its awaited execution and restore its parent."""
    if scope is None:
        yield
        return
    token = _ACTIVE_TRACE_TOOL_CALL.set(scope)
    try:
        yield
    finally:
        _ACTIVE_TRACE_TOOL_CALL.reset(token)


@contextmanager
def trace_provider_attempt(attempt: TraceProviderAttempt | None) -> Iterator[None]:
    """Bind one provider attempt to its transport work and restore its parent on exit."""
    if attempt is None:
        yield
        return
    token = _ACTIVE_TRACE_PROVIDER_ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _ACTIVE_TRACE_PROVIDER_ATTEMPT.reset(token)


@contextmanager
def trace_scope(
    recorder: TraceRecorder,
    source: TraceRootSource,
    *,
    purpose: TracePurpose | None = None,
) -> Iterator[TraceContext]:
    """Create and bind a new root while restoring any parent task scope on exit."""
    try:
        context = recorder.new_root(source)
    except Exception:
        logger.warning("Execution Trace root could not be created")
        context = TraceContext.create(str(uuid4()), source)
    scope = TraceExecutionScope(
        recorder=recorder,
        context=context,
        purpose=source.purpose if purpose is None else purpose,
    )
    token = _ACTIVE_TRACE_SCOPE.set(scope)
    try:
        yield context
    finally:
        _ACTIVE_TRACE_SCOPE.reset(token)


@contextmanager
def trace_purpose(purpose: TracePurpose) -> Iterator[None]:
    """Override only the call purpose while retaining its root and ownership."""
    current = _ACTIVE_TRACE_SCOPE.get()
    if current is None:
        yield
        return
    token: Token[TraceExecutionScope | None] = _ACTIVE_TRACE_SCOPE.set(
        replace(current, purpose=purpose)
    )
    try:
        yield
    finally:
        _ACTIVE_TRACE_SCOPE.reset(token)
