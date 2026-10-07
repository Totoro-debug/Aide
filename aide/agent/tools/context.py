"""Immutable facts bound to one Tool Gateway view and Agent Run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aide.agent.tools.core.exec_host import ExecHost
    from aide.schedule.service import ScheduleService


@dataclass(frozen=True, slots=True)
class ToolRunContext:
    """Workspace and runtime targets used by one Tool invocation path."""

    workspace: Path
    schedule_service: ScheduleService | None = None
    exec_host: ExecHost | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("Tool Run Context workspace must be a Path")


__all__ = ["ToolRunContext"]
