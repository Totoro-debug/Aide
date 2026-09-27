from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from myclaw.config.agent_home import AgentHome
from myclaw.management.service import (
    ManagementViewService,
    RuntimeStatusInput,
)
from myclaw.memory.dream import DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.permission.state import RuntimePermissionControl
from myclaw.provider.models import ReasoningEffort
from myclaw.skills.catalog import SkillMetadata
from myclaw.workspace.state import WorkspaceState


class _DefaultLoop:
    def reload_skill(self) -> tuple[SkillMetadata, ...]:
        return ()

    def runtime_status_input(self) -> RuntimeStatusInput:
        return RuntimeStatusInput(
            chat_model="test/model",
            context_window=1,
        )


class _DefaultDream:
    async def run(self) -> DreamResult:
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )


class _DefaultReasoningEffortControl:
    def __init__(self, effort: ReasoningEffort = "medium") -> None:
        self.effort = effort

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        return self.effort

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        self.effort = effort


async def _replace_agent_loop(_session_id: str, _force: bool) -> None:
    return None


async def _prepare_session_resume(_session_id: str) -> None:
    return None


def management_service(
    agent_home: AgentHome,
    *,
    current_agent_loop: Callable[[], Any] | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_agent_loop: Callable[[str, bool], Awaitable[None]] = _replace_agent_loop,
    prepare_session_resume: Callable[[str], Awaitable[None]] = _prepare_session_resume,
    memory_manager: Any | None = None,
    dream: Any | None = None,
    reasoning_effort_control: Any | None = None,
    permission_control: RuntimePermissionControl | None = None,
    schedule_status: Callable[[], dict[str, object]] = lambda: {
        "status": "available",
        "active_job_count": 0,
    },
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = lambda: 0.0,
) -> ManagementViewService:
    state = workspace_state
    if state is None:
        workspace = agent_home.path.parent / f"{agent_home.path.name}-management-workspace"
        workspace.mkdir(exist_ok=True)
        state = WorkspaceState(workspace)
        state.initialize(agent_home_root=agent_home.path)
    return ManagementViewService(
        agent_home,
        current_agent_loop=current_agent_loop or (lambda: _DefaultLoop()),
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        prepare_session_resume=prepare_session_resume,
        memory_manager=memory_manager or MemoryManager(state),
        dream=dream or _DefaultDream(),
        reasoning_effort_control=reasoning_effort_control or _DefaultReasoningEffortControl(),
        permission_control=permission_control or RuntimePermissionControl("workspace-write"),
        schedule_status=schedule_status,
        now=now,
        monotonic=monotonic,
    )
