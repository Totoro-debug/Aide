"""Shared execution resources retained by immutable configuration snapshots."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.mcp_keywords import MCPKeywordPreparer
from aide.agent.tools.mcp_runtime import (
    MCPRuntimeManager,
    MCPStartupReport,
    MCPWorkspaceRuntimeManager,
)
from aide.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES, BuiltInToolCatalog
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.model_router import ModelRouter, ProviderFactory
from aide.service.configuration import ConfigurationSnapshot
from aide.skills.catalog import SkillLoader


@dataclass(slots=True, eq=False)
class SharedConfigurationResources:
    snapshot: ConfigurationSnapshot
    router: ModelRouter
    mcp_manager: MCPRuntimeManager
    skill_loader: SkillLoader
    exec_host: ExecHost
    built_in_catalog: BuiltInToolCatalog
    users: int = 1


@dataclass(slots=True, eq=False)
class WorkspaceConfigurationResources:
    shared: SharedConfigurationResources
    mcp_manager: MCPWorkspaceRuntimeManager
    mcp_report: MCPStartupReport
    keywords: Mapping[str, tuple[str, ...]]
    keyword_preparer: MCPKeywordPreparer
    users: int = 1


class ConfigurationResourceManager:
    """Prepare once per version and scope; never replace a live Run's collaborators."""

    def __init__(self, agent_home: AgentHome, provider_factory: ProviderFactory) -> None:
        self._agent_home = agent_home
        self._provider_factory = provider_factory
        self._shared: SharedConfigurationResources | None = None
        self._versions: dict[int, SharedConfigurationResources] = {}
        self._pending_versions: set[SharedConfigurationResources] = set()
        self._workspaces: dict[str, WorkspaceConfigurationResources] = {}
        self._shared_lock = asyncio.Lock()
        self._workspace_locks: dict[str, asyncio.Lock] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_errors: list[Exception] = []
        self._closed = False

    async def shared(
        self,
        snapshot: ConfigurationSnapshot,
        skill_loader: SkillLoader,
        exec_host: ExecHost,
    ) -> SharedConfigurationResources:
        async with self._shared_lock:
            if self._closed:
                raise RuntimeError("Configuration resources are closed")
            previous = self._shared
            cached = self._versions.get(snapshot.generation)
            if cached is not None and cached.snapshot == snapshot:
                return cached
            if previous is not None and previous.snapshot == snapshot:
                return previous
            router = (
                previous.router.fork(snapshot.configuration) if previous is not None else
                ModelRouter(configuration=snapshot.configuration, provider_factory=self._provider_factory)
            )
            mcp = (
                previous.mcp_manager.fork() if previous is not None else
                MCPRuntimeManager(None, transport="streamable-http", built_in_names=BUILT_IN_TOOL_NAMES)
            )
            try:
                await mcp.start(snapshot.configuration.mcp)
            except BaseException:
                await asyncio.gather(router.close(), mcp.close(), return_exceptions=True)
                raise
            catalog = (
                previous.built_in_catalog
                if previous is not None and previous.exec_host is exec_host else
                BuiltInToolCatalog(skill_root=skill_loader.root, exec_host=exec_host)
            )
            selected = SharedConfigurationResources(
                snapshot, router, mcp, skill_loader, exec_host, catalog,
            )
            self._versions[snapshot.generation] = selected
            if previous is None or snapshot.generation >= previous.snapshot.generation:
                self._shared = selected
                if previous is not None:
                    self._release_shared(previous)
            else:
                self._pending_versions.add(selected)
            return selected

    async def workspace(
        self,
        workspace_id: str,
        workspace_path: Path,
        shared: SharedConfigurationResources,
    ) -> WorkspaceConfigurationResources:
        shared.users += 1
        if shared in self._pending_versions:
            self._pending_versions.remove(shared)
            self._release_shared(shared)
        lock = self._workspace_locks.setdefault(workspace_id, asyncio.Lock())
        try:
            await lock.acquire()
        except BaseException:
            self._release_shared(shared)
            raise
        try:
            if self._closed:
                self._release_shared(shared)
                raise RuntimeError("Configuration resources are closed")
            previous = self._workspaces.get(workspace_id)
            if previous is not None and previous.shared is shared:
                self._release_shared(shared)
                return previous
            mcp = (
                previous.mcp_manager.fork(shared.mcp_manager) if previous is not None else
                MCPWorkspaceRuntimeManager(
                    workspace_path, shared_runtime=shared.mcp_manager,
                    built_in_names=BUILT_IN_TOOL_NAMES,
                )
            )
            preparer = (
                previous.keyword_preparer.for_router(shared.router) if previous is not None else
                MCPKeywordPreparer(shared.router, ConfigLoader(self._agent_home))
            )
            try:
                report = await mcp.start(shared.snapshot.configuration.mcp)
                keywords = await preparer.prepare(report.snapshot, shared.snapshot.configuration.mcp)
            except BaseException:
                try:
                    await mcp.close()
                finally:
                    self._release_shared(shared)
                raise
            selected = WorkspaceConfigurationResources(shared, mcp, report, keywords, preparer)
            self._workspaces[workspace_id] = selected
            if previous is not None:
                self.release(previous)
            return selected
        finally:
            lock.release()

    def retain_shared(self, resources: SharedConfigurationResources) -> Callable[[], None]:
        """Keep shared collaborators alive while a Workspace awaits local recovery."""
        resources.users += 1
        if resources in self._pending_versions:
            self._pending_versions.remove(resources)
            self._release_shared(resources)
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._release_shared(resources)

        return release

    def retain(self, resources: WorkspaceConfigurationResources) -> Callable[[], None]:
        resources.users += 1
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self.release(resources)

        return release

    def release(self, resources: WorkspaceConfigurationResources) -> None:
        resources.users -= 1
        if resources.users == 0:
            self._schedule_cleanup(self._close_workspace(resources))

    async def _close_workspace(self, resources: WorkspaceConfigurationResources) -> None:
        try:
            await resources.mcp_manager.close()
        finally:
            self._release_shared(resources.shared)

    def _release_shared(self, resources: SharedConfigurationResources) -> None:
        resources.users -= 1
        if resources.users == 0:
            if self._versions.get(resources.snapshot.generation) is resources:
                del self._versions[resources.snapshot.generation]
            self._schedule_cleanup(self._close_shared(resources))

    async def _close_shared(self, resources: SharedConfigurationResources) -> None:
        results = await asyncio.gather(
            resources.mcp_manager.close(), resources.router.close(), return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("Configuration resource cleanup failed", errors)

    def _schedule_cleanup(self, cleanup: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(cleanup)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_done)

    def _cleanup_done(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            if isinstance(error, Exception):
                self._cleanup_errors.append(error)

    async def close_workspace(self, workspace_id: str) -> None:
        resources = self._workspaces.pop(workspace_id, None)
        if resources is not None:
            self.release(resources)
        self._workspace_locks.pop(workspace_id, None)
        await self._drain_cleanup()

    async def _drain_cleanup(self) -> None:
        while self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        if self._cleanup_errors:
            errors, self._cleanup_errors = self._cleanup_errors, []
            raise ExceptionGroup("Configuration resource cleanup failed", errors)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            for workspace_id in tuple(self._workspaces):
                resources = self._workspaces.pop(workspace_id)
                self.release(resources)
            if self._shared is not None:
                self._release_shared(self._shared)
                self._shared = None
            for shared_resources in tuple(self._pending_versions):
                self._release_shared(shared_resources)
            self._pending_versions.clear()
        await self._drain_cleanup()
