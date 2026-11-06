"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aide.agent.subagents.context import SubAgentToolContext
    from aide.agent.tools.core.exec_host import ExecHost
    from aide.schedule.service import ScheduleService


@dataclass(frozen=True, slots=True)
class AgentRunSuspension:
    """Typed control result that ends an Agent Run at a safe boundary."""

    kind: str
    task_id: str
    generation: int
    execution_id: str
    question_id: str | None
    tool_call_id: str | None

    def __post_init__(self) -> None:
        if self.kind not in {"human_question", "subagent_attention", "worker_reports"}:
            raise ValueError("Agent Run suspension kind is invalid")
        if any(
            not isinstance(value, str) or not value
            for value in (self.task_id, self.execution_id)
        ):
            raise ValueError("Agent Run suspension identities must be non-empty strings")
        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id
        ):
            raise ValueError("Agent Run suspension tool call identity is invalid")
        if self.question_id is not None and (
            not isinstance(self.question_id, str) or not self.question_id
        ):
            raise ValueError("Agent Run suspension question identity is invalid")
        if self.kind != "worker_reports" and self.question_id is None:
            raise ValueError("Question suspension requires a question identity")
        if self.kind != "worker_reports" and self.tool_call_id is None:
            raise ValueError("Question suspension requires a tool call identity")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ValueError("Agent Run suspension generation must be positive")


@dataclass(frozen=True, slots=True)
class ToolRunContext:
    """Workspace and runtime targets used by one Tool invocation path."""

    workspace: Path
    schedule_service: ScheduleService | None = None
    exec_host: ExecHost | None = None
    subagent: SubAgentToolContext | None = None
    foreground_question_handler: Callable[[str, Sequence[Mapping[str, str]], str], str] | None = (
        None
    )
    run_suspension_handler: Callable[
        [str, Mapping[str, object], str, str], AgentRunSuspension | None
    ] | None = None
    task_id: str | None = None
    generation: int | None = None
    execution_id: str | None = None
    foreground_task_is_current: Callable[[str, int, str], bool] | None = None
    tool_call_id: str | None = None
    worker_dispatch_handler: Callable[[str, str, Sequence[str], str], Awaitable[str]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("Tool Run Context workspace must be a Path")
        if self.subagent is not None and any(
            not hasattr(self.subagent, name)
            for name in ("coordinator", "parent_run_id", "source", "creator_snapshot")
        ):
            raise TypeError("Tool Run Context SubAgent facts are invalid")
        foreground_identity = (self.task_id, self.generation, self.execution_id)
        if self.foreground_question_handler is not None and any(
            value is None for value in foreground_identity
        ):
            raise TypeError("Foreground question handling requires Task and execution identity")
        if self.generation is not None and (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ValueError("Tool Run Context generation must be positive")
        if self.foreground_task_is_current is not None and (
            self.task_id is None or self.generation is None or self.execution_id is None
        ):
            raise TypeError("Foreground Task validation requires Task and execution identity")
        if self.worker_dispatch_handler is not None and not callable(self.worker_dispatch_handler):
            raise TypeError("Worker dispatch handler must be callable")


__all__ = ["AgentRunSuspension", "ToolRunContext"]
