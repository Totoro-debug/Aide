"""Task-local execution identity for model calls recorded by a shared Router."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from aide.trace.context import TraceContext, TraceRootSource
from aide.trace.types import (
    TraceAttemptHandle,
    TraceCaptureIssue,
    TraceCaptureLayer,
    TraceCaptureState,
    TraceDirection,
    TracePurpose,
)

if TYPE_CHECKING:
    from aide.trace.recorder import TraceRecorder


@dataclass(frozen=True, slots=True)
class TraceExecutionScope:
    recorder: TraceRecorder
    context: TraceContext
    purpose: TracePurpose


@dataclass(slots=True)
class TraceProviderAttempt:
    recorder: TraceRecorder
    handle: TraceAttemptHandle
    capture_expected: bool = True
    request_captured: bool = False
    response_captured: bool = False
    response_metadata_captured: bool = False
    response_body_started: bool = False
    http_exchange_count: int = 0

    def begin_http_exchange(self) -> int:
        index = self.http_exchange_count
        self.http_exchange_count += 1
        self.request_captured = False
        self.response_captured = False
        self.response_metadata_captured = False
        self.response_body_started = False
        return index

    def set_endpoint(self, endpoint: str) -> None:
        try:
            self.recorder.set_endpoint(self.handle, endpoint)
        except Exception:
            self.report_capture_issue("Provider endpoint metadata could not be captured.")

    def append_raw(
        self,
        *,
        source: str,
        direction: TraceDirection,
        layer: TraceCaptureLayer,
        sequence: int,
        data: bytes,
        content_type: str | None = None,
        content_encoding: str | None = None,
    ) -> bool:
        try:
            self.recorder.append_raw(
                self.handle,
                source=source,
                direction=direction,
                layer=layer,
                sequence=sequence,
                data=data,
                content_type=content_type,
                content_encoding=content_encoding,
            )
        except Exception:
            self.report_capture_issue(f"Provider {source} could not be captured.")
            return False
        return True

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

    def report_incomplete(self) -> bool:
        if not self.capture_expected:
            return False
        if not self.request_captured or not self.response_captured:
            self.report_capture_issue("Provider HTTP request or response capture was incomplete.")
            return True
        return False


_ACTIVE_TRACE_SCOPE: ContextVar[TraceExecutionScope | None] = ContextVar(
    "aide_active_trace_scope", default=None
)
_ACTIVE_TRACE_PROVIDER_ATTEMPT: ContextVar[TraceProviderAttempt | None] = ContextVar(
    "aide_active_trace_provider_attempt", default=None
)


def current_trace_scope() -> TraceExecutionScope | None:
    """Return the immutable trace identity active in this task, if any."""
    return _ACTIVE_TRACE_SCOPE.get()


def current_trace_provider_attempt() -> TraceProviderAttempt | None:
    """Return the provider attempt active in this task, if any."""
    return _ACTIVE_TRACE_PROVIDER_ATTEMPT.get()


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
