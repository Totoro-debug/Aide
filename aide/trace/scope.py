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
from aide.trace.types import TracePurpose

if TYPE_CHECKING:
    from aide.trace.recorder import TraceRecorder


@dataclass(frozen=True, slots=True)
class TraceExecutionScope:
    recorder: TraceRecorder
    context: TraceContext
    purpose: TracePurpose


_ACTIVE_TRACE_SCOPE: ContextVar[TraceExecutionScope | None] = ContextVar(
    "aide_active_trace_scope", default=None
)


def current_trace_scope() -> TraceExecutionScope | None:
    """Return the immutable trace identity active in this task, if any."""
    return _ACTIVE_TRACE_SCOPE.get()


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
