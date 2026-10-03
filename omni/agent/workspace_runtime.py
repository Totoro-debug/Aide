"""Shared Workspace-owned runtime resources for CLI and future clients."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar, Self, TypeVar, cast

from tzlocal import get_localzone_name

from omni.agent.confirmation import ConfirmationOwner
from omni.agent.memory.dream import Dream
from omni.agent.memory.manager import MemoryManager
from omni.agent.permission import (
    PermissionExecShell,
    PermissionSnapshot,
    ToolPermissionLevel,
    validate_permission_level,
)
from omni.agent.session.deletion import recover_session_deletions
from omni.agent.session.restore import RestoreManager, RestoreResult
from omni.agent.tools.mcp_keywords import MCPKeywordPreparer
from omni.agent.tools.mcp_runtime import (
    MCPRuntimeManager,
    MCPStartupReport,
    MCPToolSnapshot,
)
from omni.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES
from omni.agent.workspace_state import WorkspaceState, normalize_workspace_path
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader, ProviderConfiguration, UserConfiguration
from omni.provider.factory import create_provider
from omni.provider.model_router import ModelRouter
from omni.provider.models import ModelProvider
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.schedule.service import (
    ScheduleClock,
    ScheduleOccurrence,
    ScheduleService,
)
from omni.utils.scheduler import AsyncioSchedulerClock
from omni.utils.time import local_now

WorkspaceJobExecutor = Callable[[ScheduleJob], Awaitable[None]]
WorkspaceOccurrenceExecutor = Callable[[ScheduleOccurrence], Awaitable[None]]
ConfirmationOwnerCanceller = Callable[[ConfirmationOwner], Awaitable[None]]
ProviderFactory = Callable[[ProviderConfiguration], ModelProvider]
ForegroundCloser = Callable[[], Awaitable[None]]
RuntimeFactory = Callable[..., Any]
RuntimeResource = TypeVar("RuntimeResource")


@dataclass(frozen=True, slots=True)
class WorkspaceRuntimeFactories:
    """Construction seams used by a composition root and its tests."""

    workspace_state: RuntimeFactory = WorkspaceState
    restore_manager: RuntimeFactory = RestoreManager
    mcp_runtime: RuntimeFactory = MCPRuntimeManager
    router: RuntimeFactory = ModelRouter
    mcp_keyword_preparer: RuntimeFactory = MCPKeywordPreparer
    memory_manager: RuntimeFactory = MemoryManager
    dream: RuntimeFactory = Dream
    schedule_service: RuntimeFactory = ScheduleService


class WorkspaceRuntimeError(ValueError):
    """Raised when a Workspace cannot be assigned a safe runtime identity."""


class WorkspaceRuntimeRestoreError(RuntimeError):
    """Raised when startup Restore recovery cannot complete safely."""


class WorkspaceRuntimeRegistry:
    """Own Workspace Runtime identity and lifecycle coordination for one owner."""

    def __init__(self) -> None:
        self.runtimes: dict[str, WorkspaceRuntime] = {}
        self.lock = RLock()

    def publish_replacements(
        self, replacements: tuple[tuple[WorkspaceRuntime, WorkspaceRuntime], ...]
    ) -> None:
        """Validate all owners before publishing any Runtime generation."""
        with self.lock:
            for previous, candidate in replacements:
                key = _workspace_key(candidate.workspace_path)
                if self.runtimes.get(key) is not previous or previous._closed or candidate._closed:
                    raise WorkspaceRuntimeError("Workspace Runtime replacement owner is stale")
            for _previous, candidate in replacements:
                self.runtimes[_workspace_key(candidate.workspace_path)] = candidate


_LEGACY_WORKSPACE_RUNTIME_REGISTRY = WorkspaceRuntimeRegistry()


async def _noop_cancel_confirmation(_owner: ConfirmationOwner) -> None:
    return None


class WorkspaceRuntime:
    """Own one initialized Workspace and its shared Runtime-Lifetime resources.

    The owner-provided registry is keyed by the host-resolved directory. It
    prevents path aliases from creating a second owner while keeping Session,
    Message Bus, and Agent Loop state in the caller that owns those concerns.
    """

    # Direct callers without an explicit owner retain the historical acquire API.
    _legacy_registry: ClassVar[WorkspaceRuntimeRegistry] = _LEGACY_WORKSPACE_RUNTIME_REGISTRY
    _registry: ClassVar[dict[str, WorkspaceRuntime]] = _LEGACY_WORKSPACE_RUNTIME_REGISTRY.runtimes

    def __init__(
        self,
        *,
        workspace_path: Path,
        agent_home: AgentHome,
        configuration: UserConfiguration,
        execute_user_job: WorkspaceJobExecutor,
        execute_user_occurrence: WorkspaceOccurrenceExecutor | None,
        cancel_confirmation_owner: ConfirmationOwnerCanceller,
        configured_schedule_level: ToolPermissionLevel,
        resolved_exec_shell: PermissionExecShell,
        now: Callable[[], datetime],
        timezone_name: str,
        provider_factory: ProviderFactory,
        built_in_names: tuple[str, ...],
        schedule_clock: ScheduleClock | None,
        factories: WorkspaceRuntimeFactories,
        registry: WorkspaceRuntimeRegistry,
        shared_router: ModelRouter | None,
    ) -> None:
        self.workspace_path = workspace_path
        self.agent_home = agent_home
        self.configuration = configuration
        self._execute_user_job = execute_user_job
        self._execute_user_occurrence = execute_user_occurrence
        self._cancel_confirmation_owner = cancel_confirmation_owner
        self._configured_schedule_level = configured_schedule_level
        self._resolved_exec_shell = resolved_exec_shell
        self._now = now
        self._timezone_name = timezone_name
        self._provider_factory = provider_factory
        self._built_in_names = built_in_names
        self._schedule_clock = schedule_clock
        self._factories = factories
        self._runtime_registry = registry
        self._shared_router = shared_router
        self._router_owned = shared_router is None
        self._lifecycle_lock = asyncio.Lock()

        self._workspace_state: WorkspaceState | None = None
        self._restore_result: RestoreResult | None = None
        self._mcp_manager: MCPRuntimeManager | None = None
        self._mcp_startup_report: MCPStartupReport | None = None
        self._mcp_snapshot: MCPToolSnapshot = ()
        self._mcp_keywords: Mapping[str, tuple[str, ...]] = MappingProxyType({})
        self._mcp_keyword_preparer: MCPKeywordPreparer | None = None
        self._router: ModelRouter | None = None
        self._memory_manager: MemoryManager | None = None
        self._dream: Dream | None = None
        self._schedule_service: ScheduleService | None = None
        self._schedule_prepared = False
        self._started = False
        self._closed = False
        self._close_failed = False

    @classmethod
    def acquire(
        cls,
        *,
        workspace: Path,
        agent_home: AgentHome,
        configuration: UserConfiguration,
        execute_user_job: WorkspaceJobExecutor,
        execute_user_occurrence: WorkspaceOccurrenceExecutor | None = None,
        cancel_confirmation_owner: ConfirmationOwnerCanceller | None = None,
        configured_schedule_level: ToolPermissionLevel | None = None,
        resolved_exec_shell: PermissionExecShell | None = None,
        now: Callable[[], datetime] = local_now,
        timezone_name: str | None = None,
        provider_factory: ProviderFactory = create_provider,
        built_in_names: Iterable[str] = BUILT_IN_TOOL_NAMES,
        schedule_clock: ScheduleClock | None = None,
        factories: WorkspaceRuntimeFactories | None = None,
        registry: WorkspaceRuntimeRegistry | None = None,
        shared_router: ModelRouter | None = None,
    ) -> Self:
        """Return the sole active Runtime for one resolved Workspace path."""
        workspace_path = _resolve_workspace_identity(workspace)
        key = _workspace_key(workspace_path)
        selected_registry = cls._legacy_registry if registry is None else registry
        with selected_registry.lock:
            existing = selected_registry.runtimes.get(key)
            if existing is not None and not existing._closed:
                if _workspace_key(_resolve_workspace_identity(existing.workspace_path)) != key:
                    raise WorkspaceRuntimeError("Workspace Runtime identity changed")
                return cast(Self, existing)

            selected_configuration = cast(
                ToolPermissionLevel,
                (
                    getattr(configuration.runtime, "permission_level", "workspace-write")
                    if configured_schedule_level is None
                    else configured_schedule_level
                ),
            )
            validate_permission_level(selected_configuration)
            selected_shell = resolved_exec_shell
            if selected_shell is None:
                from omni.agent.tools.core.exec_host import resolve_exec_shell

                selected_shell = resolve_exec_shell(
                    getattr(configuration.runtime, "exec_shell", "auto")
                )
            runtime = cls(
                workspace_path=workspace_path,
                agent_home=agent_home,
                configuration=configuration,
                execute_user_job=execute_user_job,
                execute_user_occurrence=execute_user_occurrence,
                cancel_confirmation_owner=(
                    _noop_cancel_confirmation
                    if cancel_confirmation_owner is None
                    else cancel_confirmation_owner
                ),
                configured_schedule_level=selected_configuration,
                resolved_exec_shell=selected_shell,
                now=now,
                timezone_name=(get_localzone_name() if timezone_name is None else timezone_name),
                provider_factory=provider_factory,
                built_in_names=tuple(built_in_names),
                schedule_clock=schedule_clock,
                factories=WorkspaceRuntimeFactories() if factories is None else factories,
                registry=selected_registry,
                shared_router=shared_router,
            )
            selected_registry.runtimes[key] = runtime
            return runtime

    async def start(self) -> Self:
        """Initialize all shared resources once and return this Runtime."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Workspace Runtime is closed")
            if self._started:
                return self
            try:
                state = cast(WorkspaceState, self._factories.workspace_state(self.workspace_path))
                state.initialize(agent_home_root=self.agent_home.path)
                self._workspace_state = state
                if isinstance(state, WorkspaceState):
                    try:
                        recover_session_deletions(state)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        raise WorkspaceRuntimeRestoreError from error

                if (
                    isinstance(state, WorkspaceState)
                    or self._factories.restore_manager is not RestoreManager
                ):
                    restore_manager = cast(RestoreManager, self._factories.restore_manager(state))
                    try:
                        self._restore_result = await restore_manager.recover_pending()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        raise WorkspaceRuntimeRestoreError from error

                await self._start_resources()
                return self
            except BaseException as error:
                cleanup_errors = await self._close_owned_resources()
                self._mark_closed()
                if cleanup_errors:
                    raise error from _cleanup_exception(cleanup_errors)
                raise

    async def _start_resources(self) -> None:
        """Start generation-owned resources against an already selected Workspace state."""
        state = self.workspace_state
        manager = cast(
            MCPRuntimeManager,
            self._factories.mcp_runtime(
                self.workspace_path,
                built_in_names=self._built_in_names,
            ),
        )
        self._mcp_manager = manager
        startup_report = await manager.start(self.configuration.mcp)
        self._mcp_startup_report = startup_report
        self._mcp_snapshot = startup_report.snapshot

        if self._shared_router is None:
            router = cast(
                ModelRouter,
                self._factories.router(
                    configuration=self.configuration,
                    provider_factory=self._provider_factory,
                ),
            )
            self._router_owned = True
        else:
            router = self._shared_router
        self._router = router
        keyword_preparer = cast(
            MCPKeywordPreparer,
            self._factories.mcp_keyword_preparer(
                model_router=router,
                config_loader=ConfigLoader(self.agent_home),
            ),
        )
        self._mcp_keyword_preparer = keyword_preparer
        self._mcp_keywords = await keyword_preparer.prepare(
            self._mcp_snapshot,
            self.configuration.mcp,
        )

        memory_manager = cast(MemoryManager, self._factories.memory_manager(state))
        self._memory_manager = memory_manager
        dream = cast(
            Dream,
            self._factories.dream(
                memory_manager=memory_manager,
                model_router=router,
                batch_size=self.configuration.memory.batch_size,
                memory_route_status=router.route_status("memory"),
            ),
        )
        self._dream = dream

        schedule = cast(
            ScheduleService,
            self._factories.schedule_service(
                workspace_state=state,
                clock=(
                    AsyncioSchedulerClock(now=self._now)
                    if self._schedule_clock is None
                    else self._schedule_clock
                ),
                execute_user_job=self._execute_user_job,
                execute_user_occurrence=self._execute_user_occurrence,
                permission_snapshot_factory=self._capture_schedule_permission_snapshot,
                cancel_confirmation_owner=self._cancel_confirmation_owner,
                execute_dream=dream.run,
                timezone_name=self._timezone_name,
            ),
        )
        self._schedule_service = schedule
        self._started = True

    def create_replacement(self, configuration: UserConfiguration) -> Self:
        """Build an unregistered Runtime generation sharing this Workspace state."""
        from omni.agent.tools.core.exec_host import resolve_exec_shell

        return type(self)(
            workspace_path=self.workspace_path,
            agent_home=self.agent_home,
            configuration=configuration,
            execute_user_job=self._execute_user_job,
            execute_user_occurrence=self._execute_user_occurrence,
            cancel_confirmation_owner=self._cancel_confirmation_owner,
            configured_schedule_level=configuration.runtime.permission_level,
            resolved_exec_shell=resolve_exec_shell(configuration.runtime.exec_shell),
            now=self._now,
            timezone_name=self._timezone_name,
            provider_factory=self._provider_factory,
            built_in_names=self._built_in_names,
            schedule_clock=self._schedule_clock,
            factories=self._factories,
            registry=self._runtime_registry,
            shared_router=self._shared_router,
        )

    async def start_replacement(self, previous: WorkspaceRuntime) -> Self:
        """Start a replacement without reinitializing Workspace state or registry ownership."""
        if previous.workspace_path != self.workspace_path:
            raise WorkspaceRuntimeError("Workspace Runtime replacement path does not match")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Workspace Runtime is closed")
            if self._started:
                return self
            self._workspace_state = previous.workspace_state
            self._restore_result = previous.startup_restore_result
            try:
                await self._start_resources()
            except BaseException as error:
                cleanup_errors = await self._close_owned_resources()
                self._mark_closed()
                if cleanup_errors:
                    raise error from _cleanup_exception(cleanup_errors)
                raise
            return self

    @classmethod
    def publish_replacements(
        cls, replacements: tuple[tuple[WorkspaceRuntime, WorkspaceRuntime], ...]
    ) -> None:
        """Validate all owners before publishing any generation."""
        cls._legacy_registry.publish_replacements(replacements)

    async def prepare_schedule(self, schedule: JobSchedule) -> None:
        """Prepare the single Schedule dispatcher and register Dream once."""
        if not isinstance(schedule, JobSchedule):
            raise TypeError("Workspace Runtime Schedule preparation requires a JobSchedule")
        if not self._started:
            raise RuntimeError("Workspace Runtime has not been started")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Workspace Runtime is closed")
            if self._schedule_prepared:
                return
            schedule_service = self.schedule_service
            schedule_service._prepare_start()
            await schedule_service.register_dream_job(schedule=schedule)
            self._schedule_prepared = True

    def prepare_replacement_schedule(self) -> None:
        """Preflight a replacement dispatcher without changing persistent Job state."""
        self.schedule_service._prepare_start()
        self._schedule_prepared = True

    @property
    def workspace_state(self) -> WorkspaceState:
        return self._require(self._workspace_state, "Workspace State")

    @property
    def startup_restore_result(self) -> RestoreResult | None:
        self._require_started()
        return self._restore_result

    @property
    def startup_session_id(self) -> str | None:
        result = self.startup_restore_result
        return None if result is None else result.session_id

    @property
    def mcp_manager(self) -> MCPRuntimeManager:
        return self._require(self._mcp_manager, "MCP Runtime Manager")

    @property
    def mcp_startup_report(self) -> MCPStartupReport:
        return self._require(self._mcp_startup_report, "MCP startup report")

    @property
    def mcp_snapshot(self) -> MCPToolSnapshot:
        self._require_started()
        return self._mcp_snapshot

    @property
    def mcp_keywords(self) -> Mapping[str, tuple[str, ...]]:
        self._require_started()
        return self._mcp_keywords

    @property
    def mcp_keyword_preparer(self) -> MCPKeywordPreparer:
        return self._require(self._mcp_keyword_preparer, "MCP Keyword Preparer")

    @property
    def router(self) -> ModelRouter:
        return self._require(self._router, "Model Router")

    @property
    def memory_manager(self) -> MemoryManager:
        return self._require(self._memory_manager, "Memory Manager")

    @property
    def dream(self) -> Dream:
        return self._require(self._dream, "Dream")

    @property
    def schedule_service(self) -> ScheduleService:
        return self._require(self._schedule_service, "Schedule Service")

    async def drain_confirmation_aborts(self) -> None:
        """Drain Schedule-owned confirmations before the broader shutdown sequence."""
        async with self._lifecycle_lock:
            if self._closed or self._schedule_service is None:
                return
            await self._schedule_service.drain_confirmation_aborts()

    async def close(
        self,
        *,
        close_foreground: ForegroundCloser | None = None,
        drain_confirmation_aborts: bool = True,
    ) -> None:
        """Drain shared resources, optionally between Schedule and MCP cleanup."""
        async with self._lifecycle_lock:
            if self._closed and not self._close_failed:
                return
            cleanup_errors = await self._close_owned_resources(
                close_foreground=close_foreground,
                drain_confirmation_aborts=drain_confirmation_aborts,
            )
            if cleanup_errors:
                self._close_failed = True
                raise _cleanup_exception(cleanup_errors)
            self._mark_closed()
            self._close_failed = False

    async def abort_dream(self) -> None:
        """Cancel Dream work before a Workspace removal drains shared resources."""
        if self._dream is not None and not self._closed:
            await self._dream.abort_and_wait()

    async def _close_owned_resources(
        self,
        *,
        close_foreground: ForegroundCloser | None = None,
        drain_confirmation_aborts: bool = True,
    ) -> list[BaseException]:
        errors: list[BaseException] = []

        schedule = self._schedule_service
        if schedule is not None:
            if drain_confirmation_aborts:
                await _collect_cleanup(errors, schedule.drain_confirmation_aborts)
            await _collect_cleanup(errors, schedule.pause_and_drain)
            await _collect_cleanup(errors, schedule.close)

        if close_foreground is not None:
            await _collect_cleanup(errors, close_foreground)

        if self._mcp_manager is not None:
            await _collect_cleanup(errors, self._mcp_manager.close)
        if self._dream is not None:
            await _collect_cleanup(errors, self._dream.close)
        if self._router is not None and self._router_owned:
            await _collect_cleanup(errors, self._router.close)
        return errors

    def _capture_schedule_permission_snapshot(self) -> PermissionSnapshot:
        return PermissionSnapshot(
            level=self._configured_schedule_level,
            exec_shell=self._resolved_exec_shell,
        )

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("Workspace Runtime is unavailable")

    @staticmethod
    def _require(value: RuntimeResource | None, label: str) -> RuntimeResource:
        if value is None:
            raise RuntimeError(f"{label} is unavailable")
        return value

    def _mark_closed(self) -> None:
        self._closed = True
        self._started = False
        key = _workspace_key(self.workspace_path)
        with self._runtime_registry.lock:
            if self._runtime_registry.runtimes.get(key) is self:
                del self._runtime_registry.runtimes[key]


async def _collect_cleanup(
    errors: list[BaseException],
    cleanup: Callable[[], Awaitable[object]],
) -> None:
    try:
        await cleanup()
    except BaseException as error:
        errors.append(error)


def _cleanup_exception(errors: list[BaseException]) -> BaseException:
    if len(errors) == 1:
        return errors[0]
    return BaseExceptionGroup("Workspace Runtime shutdown failed", errors)


def _resolve_workspace_identity(path: Path) -> Path:
    normalized = normalize_workspace_path(path)
    try:
        resolved = normalized.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise WorkspaceRuntimeError("Workspace directory could not be resolved") from error
    return resolved


def _workspace_key(path: Path) -> str:
    return os.path.normcase(str(path))


__all__ = [
    "WorkspaceRuntime",
    "WorkspaceRuntimeError",
    "WorkspaceRuntimeFactories",
    "WorkspaceRuntimeRegistry",
    "WorkspaceRuntimeRestoreError",
]
