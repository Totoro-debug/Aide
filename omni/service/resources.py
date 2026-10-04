"""Service-owned Memory, Dream, and Schedule resources per Workspace."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from omni.agent.memory.dream import Dream
from omni.agent.memory.manager import MemoryManager
from omni.agent.tools.mcp_keywords import MCPKeywordPreparer
from omni.agent.tools.mcp_runtime import (
    MCPStartupReport,
    MCPToolSnapshot,
    MCPWorkspaceRuntimeManager,
)
from omni.agent.workspace_state import WorkspaceState
from omni.provider.model_router import ModelRouter
from omni.schedule.service import ScheduleDispatcher, ScheduleService

ForegroundCloser = Callable[[], Awaitable[object]]


@dataclass(slots=True)
class WorkspaceResources:
    """The durable and execution resources associated with one Workspace."""

    workspace_id: str
    workspace_state: WorkspaceState
    memory_manager: MemoryManager
    dream: Dream
    schedule_service: ScheduleService
    workspace_path: Path
    mcp_manager: MCPWorkspaceRuntimeManager
    mcp_startup_report: MCPStartupReport
    mcp_snapshot: MCPToolSnapshot
    mcp_keywords: Mapping[str, tuple[str, ...]]
    mcp_keyword_preparer: MCPKeywordPreparer
    router: ModelRouter


class WorkspaceResourceManager:
    """Own one Memory/Dream/Schedule set for each active Workspace."""

    def __init__(self) -> None:
        self._resources: dict[str, WorkspaceResources] = {}
        self._dispatcher = ScheduleDispatcher()
        self._closed = False

    @property
    def dispatcher(self) -> ScheduleDispatcher:
        return self._dispatcher

    @property
    def resources(self) -> Mapping[str, WorkspaceResources]:
        return self._resources

    def register_resources(self, resources: WorkspaceResources) -> WorkspaceResources:
        if self._closed:
            raise RuntimeError("Workspace Resource Manager is closed")
        workspace_id = resources.workspace_id
        existing = self._resources.get(workspace_id)
        if existing is not None:
            if existing.workspace_state is not resources.workspace_state:
                raise RuntimeError("Workspace Resource registration changed its state")
            return existing
        self._dispatcher.register(workspace_id, resources.schedule_service)
        self._resources[workspace_id] = resources
        return resources

    def get(self, workspace_id: str) -> WorkspaceResources:
        try:
            return self._resources[workspace_id]
        except KeyError as error:
            raise RuntimeError("Workspace resources are not registered") from error

    async def close_workspace(
        self,
        workspace_id: str,
        *,
        close_foreground: ForegroundCloser | None = None,
        drain_confirmation_aborts: bool = True,
    ) -> None:
        resources = self._resources.get(workspace_id)
        if resources is None:
            return
        errors: list[BaseException] = []
        await _collect_cleanup(errors, resources.dream.abort_and_wait)
        if drain_confirmation_aborts:
            await _collect_cleanup(errors, resources.schedule_service.drain_confirmation_aborts)
        await _collect_cleanup(errors, resources.schedule_service.pause_and_drain)
        await _collect_cleanup(errors, resources.schedule_service.close)
        if close_foreground is not None:
            await _collect_cleanup(errors, close_foreground)
        await _collect_cleanup(errors, resources.dream.close)
        if resources.mcp_manager is not None:
            await _collect_cleanup(errors, resources.mcp_manager.close)
        if errors:
            raise BaseExceptionGroup("Workspace resource cleanup failed", errors)
        self._resources.pop(workspace_id, None)

    async def close(self) -> None:
        errors: list[BaseException] = []
        for workspace_id in tuple(self._resources):
            try:
                await self.close_workspace(workspace_id)
            except BaseException as error:
                errors.append(error)
        try:
            await self._dispatcher.close()
        except BaseException as error:
            errors.append(error)
        self._closed = True
        if errors:
            raise BaseExceptionGroup("Workspace Resource Manager cleanup failed", errors)


async def _collect_cleanup(
    errors: list[BaseException],
    cleanup: Callable[[], Awaitable[object]],
) -> None:
    try:
        await cleanup()
    except BaseException as error:
        errors.append(error)


__all__ = ["WorkspaceResourceManager", "WorkspaceResources"]
