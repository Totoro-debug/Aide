"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aide.agent.subagents.context import SubAgentToolContext
    from aide.agent.tools.core.exec_host import ExecHost
    from aide.schedule.service import ScheduleService


@dataclass(frozen=True, slots=True)
class ToolRunContext:
    """Workspace and runtime targets used by one Tool invocation path."""

    workspace: Path
    schedule_service: ScheduleService | None = None
    exec_host: ExecHost | None = None
    subagent: SubAgentToolContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("Tool Run Context workspace must be a Path")
        if self.subagent is not None and any(
            not hasattr(self.subagent, name)
            for name in ("coordinator", "parent_run_id", "source", "creator_snapshot")
        ):
            raise TypeError("Tool Run Context SubAgent facts are invalid")


__all__ = ["ToolRunContext"]
