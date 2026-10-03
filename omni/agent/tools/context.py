"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from omni.agent.tools.core.exec_host import ExecHost
    from omni.schedule.service import ScheduleService


@dataclass(frozen=True, slots=True)
class ToolRunContext:
    """Workspace and runtime targets used by one Tool invocation path."""

    workspace: Path
    schedule_service: ScheduleService | None = None
    exec_host: ExecHost | None = None
    session_id: str | None = None
    run_id: str | None = None
    run_token: UUID | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("Tool Run Context workspace must be a Path")
        for name in ("session_id", "run_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"Tool Run Context {name} must be a non-empty string or None")
        if self.run_token is not None and not isinstance(self.run_token, UUID):
            raise TypeError("Tool Run Context run_token must be a UUID or None")

    def for_run(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        run_token: UUID | None = None,
    ) -> ToolRunContext:
        """Return a detached context with the identity of one Agent Run."""
        return replace(
            self,
            session_id=session_id,
            run_id=run_id,
            run_token=run_token,
        )


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
