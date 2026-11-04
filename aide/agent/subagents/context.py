"""Immutable Main Agent facts used when it submits or queries SubAgents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentSource,
    SubAgentSourceKind,
)
from aide.agent.subagents.ports import SubAgentSessionCoordinator
from aide.utils.validation import require_uuid4_string


@dataclass(frozen=True, slots=True)
class SubAgentToolContext:
    """Capture the Session coordinator and immutable caller ownership for one Run."""

    coordinator: SubAgentSessionCoordinator
    parent_run_id: str
    source: SubAgentSource
    creator_snapshot: SubAgentCreatorSnapshot | None
    caller_role: Literal["main", "worker", "subagent"] = "main"
    owner_agent_id: str | None = None
    owner_is_current: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip():
            raise ValueError("SubAgent Tool Context parent_run_id must not be empty")
        if not isinstance(self.source, SubAgentSource):
            raise TypeError("SubAgent Tool Context requires a valid task source")
        if self.caller_role not in {"main", "worker", "subagent"}:
            raise ValueError("SubAgent Tool Context caller role is invalid")
        if self.caller_role == "worker":
            if self.source.kind is not SubAgentSourceKind.WORKER or not callable(
                self.owner_is_current
            ):
                raise TypeError("Worker SubAgent Tools require a current Worker owner")
            if self.owner_agent_id is not None:
                raise ValueError("Worker SubAgent Tools cannot claim a SubAgent owner")
        elif self.caller_role == "subagent":
            if self.owner_agent_id is None:
                raise ValueError("SubAgent Tools require the current SubAgent owner")
            require_uuid4_string(self.owner_agent_id, field="owner_agent_id")
            if self.owner_is_current is not None:
                raise ValueError("SubAgent Tools cannot claim a Worker liveness check")
        elif self.source.kind is SubAgentSourceKind.WORKER or self.owner_agent_id is not None:
            raise ValueError("Main Agent SubAgent Tools cannot claim Worker ownership")
        if self.creator_snapshot is not None and not isinstance(
            self.creator_snapshot, SubAgentCreatorSnapshot
        ):
            raise TypeError("SubAgent Tool Context requires a creator snapshot")
        if not isinstance(getattr(self.coordinator, "session_id", None), str):
            raise TypeError("SubAgent Tool Context requires a Session coordinator")
        for name in ("submit", "get", "list", "wait"):
            if not callable(getattr(self.coordinator, name, None)):
                raise TypeError("SubAgent Tool Context requires a Session coordinator")

    @property
    def query_source(self) -> SubAgentSource | None:
        return self.source if self.caller_role in {"worker", "subagent"} else None

    @property
    def query_owner_agent_id(self) -> str | None:
        return self.owner_agent_id if self.caller_role == "subagent" else None

    def require_current_worker(self) -> None:
        if self.caller_role != "worker" or self.owner_is_current is None:
            raise ValueError("Only a Worker can create SubAgents for this source")
        if not self.owner_is_current():
            raise ValueError("Worker is no longer active")


__all__ = ["SubAgentToolContext"]
