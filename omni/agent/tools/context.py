"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.agent.tools.core.exec_host import ExecHost
    from omni.schedule.service import ScheduleService


@dataclass(frozen=True, slots=True)
class ToolRunContext:
    """Workspace and runtime targets used by one Tool invocation path."""

    workspace: Path
    schedule_service: ScheduleService | None = None
    exec_host: ExecHost | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("Tool Run Context workspace must be a Path")


_BOUND_TOOL_RUN_CONTEXT: ContextVar[ToolRunContext | None] = ContextVar(
    "omni_bound_tool_run_context",
    default=None,
)


@contextmanager
def bind_tool_run_context(context: ToolRunContext) -> Iterator[None]:
    """Bind one context only while a legacy execution hook is awaited."""
    if not isinstance(context, ToolRunContext):
        raise TypeError("Bound Tool Run Context must be a ToolRunContext")
    token = _BOUND_TOOL_RUN_CONTEXT.set(context)
    try:
        yield
    finally:
        _BOUND_TOOL_RUN_CONTEXT.reset(token)


def bound_tool_run_context() -> ToolRunContext | None:
    """Return the task-local context exposed to a legacy Tool hook."""
    return _BOUND_TOOL_RUN_CONTEXT.get()


__all__ = [
    "ToolRunContext",
    "bind_tool_run_context",
    "bound_tool_run_context",
]
