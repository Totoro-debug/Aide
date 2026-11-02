"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    question_id: str
    tool_call_id: str

    def __post_init__(self) -> None:
        if self.kind != "human_question":
            raise ValueError("Agent Run suspension kind is invalid")
        if any(
            not isinstance(value, str) or not value
            for value in (self.task_id, self.execution_id, self.question_id, self.tool_call_id)
        ):
            raise ValueError("Agent Run suspension identities must be non-empty strings")
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
    task_id: str | None = None
    generation: int | None = None
    execution_id: str | None = None
    foreground_task_is_current: Callable[[str, int, str], bool] | None = None
    tool_call_id: str | None = None

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


__all__ = ["AgentRunSuspension", "ToolRunContext"]
