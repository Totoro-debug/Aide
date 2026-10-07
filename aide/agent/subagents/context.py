"""Immutable Main Agent facts used when it submits or queries SubAgents."""

from __future__ import annotations

from dataclasses import dataclass

from aide.agent.subagents.models import SubAgentCreatorSnapshot, SubAgentSource
from aide.agent.subagents.ports import SubAgentSessionCoordinator


@dataclass(frozen=True, slots=True)
class SubAgentToolContext:
    """Capture the Session coordinator and creator facts for one Main Agent Run."""

    coordinator: SubAgentSessionCoordinator
    parent_run_id: str
    source: SubAgentSource
    creator_snapshot: SubAgentCreatorSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip():
            raise ValueError("SubAgent Tool Context parent_run_id must not be empty")
        if not isinstance(self.source, SubAgentSource):
            raise TypeError("SubAgent Tool Context requires a valid task source")
        if not isinstance(self.creator_snapshot, SubAgentCreatorSnapshot):
            raise TypeError("SubAgent Tool Context requires a creator snapshot")
        if not isinstance(getattr(self.coordinator, "session_id", None), str):
            raise TypeError("SubAgent Tool Context requires a Session coordinator")
        for name in ("submit", "get", "list", "wait"):
            if not callable(getattr(self.coordinator, name, None)):
                raise TypeError("SubAgent Tool Context requires a Session coordinator")


__all__ = ["SubAgentToolContext"]
