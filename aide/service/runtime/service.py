"""Service resources, clients and lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast
from uuid import uuid4

from tzlocal import get_localzone_name

from aide.agent.confirmation import (
    ConfirmationDecision,
    ConfirmationOwner,
    ToolConfirmationCoordinator,
)
from aide.agent.memory.dream import Dream
from aide.agent.memory.manager import MemoryManager
from aide.agent.permission import RuntimePermissionControl
from aide.agent.session.deletion import (
    recover_session_deletions,
    session_deletion_pending,
    session_deletion_status,
)
from aide.agent.session.restore import (
    RestoreManager,
)
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.subagents.models import (
    SubAgentRecord,
    SubAgentStatus,
)
from aide.agent.subagents.store import (
    SubAgentRecordStore,
    SubAgentRequestError,
    SubAgentStoreError,
)
from aide.agent.tools.core.exec_host import ExecHost, create_exec_host, resolve_exec_shell
from aide.agent.tools.mcp_runtime import MCPRuntimeManager
from aide.agent.tools.tool_gateway import (
    BuiltInToolCatalog,
)
from aide.agent.workspace_state import WorkspaceState, WorkspaceStateError
from aide.config.agent_home import AgentHome
from aide.config.config import (
    ConfigError,
    ConfigLoader,
    ReasoningEffort,
    UserConfiguration,
)
from aide.management.commands import MANAGEMENT_COMMANDS
from aide.provider.factory import create_provider
from aide.provider.model_router import ModelRouter
from aide.provider.models import REASONING_EFFORT_LEVELS, SessionModelConfiguration
from aide.schedule.history import (
    ScheduleHistoryPersistenceError,
    ScheduleHistoryRequestError,
    read_schedule_history,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.schedule.service import (
    ScheduleOccurrence,
    ScheduleService,
    ScheduleStaleRemovalError,
)
from aide.schedule.store import (
    ScheduleStateError,
    ScheduleStoreFaultedError,
    WorkspaceScheduleStore,
)
from aide.service.configuration import ConfigurationEdit, ConfigurationEditor, ConfigurationSnapshot
from aide.service.configuration_resources import (
    ConfigurationResourceManager,
    SharedConfigurationResources,
    WorkspaceConfigurationResources,
)
from aide.service.contracts import (
    ConversationClaimDTO,
    ConversationOpenDTO,
    ConversationOpenError,
    ConversationOpenFailureDTO,
    DreamRunDTO,
    MemoryViewDTO,
    ProjectListDTO,
    ProjectRegistrationDTO,
    ProjectSummaryDTO,
    RuntimeStatusDTO,
    ServiceStatusDTO,
    ServiceStopDTO,
    SkillMetadataDTO,
    SkillReloadDTO,
    WebLaunchTicketDTO,
)
from aide.service.conversation_workspaces import (
    ConversationWorkspaceCatalog,
    ConversationWorkspaceCatalogError,
)
from aide.service.errors import ServiceError, service_error
from aide.service.projects import ProjectCatalog, ProjectCatalogError, ProjectRecord
from aide.service.resources import WorkspaceResourceManager, WorkspaceResources
from aide.service.runtime.confirmation import ServiceConfirmationPresenter
from aide.service.runtime.pagination import (
    _MAX_SESSION_PAGE_SIZE,
    _decode_chat_session_cursor,
    _encode_chat_session_cursor,
)
from aide.service.runtime.projections import (
    _encode_management_result,
    _project_catalog_service_error,
    _project_job_summary,
    _safe_wire_value,
    _schedule_job_projection,
)
from aide.service.runtime.records import (
    ClientState,
    ServiceSink,
    SessionDeletionClaim,
    _consume_task_result,
    _LoopState,
    _ProjectRemoval,
)
from aide.service.runtime.schedule_input import (
    _schedule_epoch_milliseconds,
    _schedule_job_input,
    _schedule_request_fingerprint,
)
from aide.service.runtime.workspace import WorkspaceRecord
from aide.skills.catalog import SkillLoader, SkillMetadata
from aide.utils.scheduler import AsyncioSchedulerClock
from aide.utils.time import local_now

_PROJECT_REMOVAL_FAILURE_MESSAGE = (
    "Project work could not be stopped; the registration remains blocked."
)


WEB_TICKET_TTL_SECONDS = 25.0


class AgentService:
    """Own the Agent Home service lifecycle and schedule Agent Runs."""

    def __init__(
        self,
        agent_home: AgentHome,
        configuration: UserConfiguration | None = None,
        *,
        reconnect_timeout: float = 30.0,
        monotonic_now: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.agent_home = agent_home
        self.configuration = configuration
        self.reconnect_timeout = reconnect_timeout
        self._monotonic = monotonic_now
        self._sleep = sleep
        self.service_instance_id = str(uuid4())
        self.protocol_version = 1
        self._started_at: float | None = None
        self.state = "starting"
        self.confirmation = ToolConfirmationCoordinator()
        self._presenter = ServiceConfirmationPresenter(self)
        self._clients: dict[str, ClientState] = {}
        self._client_by_reconnect: dict[str, str] = {}
        self._workspaces: dict[str, WorkspaceRecord] = {}
        self._workspace_keys: dict[str, str] = {}
        self._workspace_resources = WorkspaceResourceManager()
        self._skill_loader: SkillLoader | None = None
        self._model_router: ModelRouter | None = None
        self._mcp_manager: MCPRuntimeManager | None = None
        self._exec_host: ExecHost | None = None
        self._built_in_tool_catalog: BuiltInToolCatalog | None = None
        self._global_reconnect_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_operation_id: str | None = None
        self._stop_request_results: dict[str, ServiceStopDTO] = {}
        self._stop_failed = False
        self.restart_requested = False
        self._restart_request_results: dict[str, tuple[str, ServiceStopDTO]] = {}
        self._closed = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._schedule_admission_lock = asyncio.Lock()
        self._schedule_mutation_lock = asyncio.Lock()
        self._schedule_mutation_results: dict[str, tuple[str, dict[str, object]]] = {}
        self._schedule_removal_jobs: dict[tuple[str, str], bool] = {}
        self._project_lifecycle_lock = asyncio.Lock()
        self._project_removals: dict[str, _ProjectRemoval] = {}
        self._configuration_editor = ConfigurationEditor(ConfigLoader(agent_home), configuration)
        self._configuration_resources = ConfigurationResourceManager(
            agent_home, lambda candidate: create_provider(candidate),
        )
        self._exec_hosts: dict[str, ExecHost] = {}
        self._skill_always_load: bool | None = None
        self._memory_schedules: dict[str, str] = {}
        self._chat_effort_override: ReasoningEffort | None = None
        self.projects = ProjectCatalog(agent_home)
        self.conversation_workspaces = ConversationWorkspaceCatalog(agent_home)

    def get_service_status(self) -> ServiceStatusDTO:
        return {
            "service_instance_id": self.service_instance_id,
            "protocol_version": self.protocol_version,
            "state": self.state,
            "active_workspace_count": len(self._workspaces),
        }

    def open_web_interface(self, client_id: str, request_id: str) -> WebLaunchTicketDTO:
        client = self._require_client(client_id)
        if client.kind != "cli":
            raise service_error(
                "forbidden", "Only CLI clients may open the Web Interface.", status=403
            )
        if not request_id:
            raise service_error("validation_error", "request_id is required.", status=422)
        return {
            "request_id": request_id,
            "ticket": secrets.token_urlsafe(32),
            "expires_in": int(WEB_TICKET_TTL_SECONDS),
        }

    def request_service_stop(self, request_id: str) -> ServiceStopDTO:
        if not request_id:
            raise service_error("validation_error", "request_id is required.", status=422)
        existing = self._stop_request_results.get(request_id)
        if existing is not None:
            return existing.copy()
        if self._stop_operation_id is None:
            self._stop_operation_id = str(uuid4())
        if self._stop_task is not None and self._stop_task.done():
            try:
                self._stop_task.result()
            except BaseException:
                self._stop_task = None
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_owned())
            self._stop_task.add_done_callback(_consume_task_result)
        result: ServiceStopDTO = {
            "request_id": request_id,
            "accepted": True,
            "operation_id": self._stop_operation_id,
        }
        self._stop_request_results[request_id] = result
        return result.copy()

    async def request_service_restart(
        self, client_id: str, request_id: str, saved_revision: str
    ) -> ServiceStopDTO:
        """Validate the saved configuration before ending this Runtime Lifetime."""
        self._require_client(client_id)
        if not request_id or not saved_revision:
            raise service_error(
                "validation_error", "request_id and saved_revision are required.", status=422
            )
        def restart() -> ServiceStopDTO:
            previous = self._restart_request_results.get(request_id)
            if previous is not None:
                if previous[0] != saved_revision:
                    raise service_error("request_reused", "request_id was already used.")
                return previous[1].copy()
            if self.state in {"draining", "stopped"} or self._stop_task is not None:
                raise service_error("admission_closed", "The local service is stopping.")
            view = self.config_view()
            application = cast(dict[str, object], view["application"])
            configuration = cast(dict[str, object], view["configuration"])
            if configuration["repair_required"]:
                raise service_error("config_invalid", "Repair configuration before restarting.")
            if application["saved_revision"] != saved_revision:
                raise service_error(
                    "config_conflict", "Saved configuration changed; refresh before restarting."
                )
            self.restart_requested = True
            result = self.request_service_stop(request_id)
            self._restart_request_results[request_id] = (saved_revision, result)
            return result.copy()

        return await self._configuration_editor.coordinate_restart(restart)

    def replacement_after_restart(self) -> AgentService:
        """Create the next Runtime Lifetime with the restart receipt retained."""
        if not self.restart_requested or self.state != "stopped" or self._stop_failed:
            raise service_error("admission_closed", "The previous Runtime has not stopped.")
        replacement = AgentService(
            self.agent_home, reconnect_timeout=self.reconnect_timeout,
            monotonic_now=self._monotonic, sleep=self._sleep,
        )
        replacement._restart_request_results = self._restart_request_results.copy()
        return replacement

    async def _activate_workspace(self, workspace: WorkspaceRecord) -> None:
        async with workspace._lock:
            if workspace._closed:
                raise RuntimeError("Workspace service runtime is closed")
            if workspace._started:
                return
            snapshot = self._capture_configuration()
            shared = await self._prepare_shared_configuration(snapshot)
            workspace.configuration = snapshot.configuration
            workspace._exec_host = shared.exec_host

            async def execute_user_job(job: ScheduleJob) -> None:
                # Admission was checked at reservation; accepted occurrences must drain.
                loop = await workspace._get_schedule_loop(job.job_id, title=job.title)
                await loop.loop.run_schedule_job(job)

            async def execute_user_occurrence(occurrence: ScheduleOccurrence) -> None:
                loop = await workspace._get_schedule_loop(occurrence.job.job_id, title=occurrence.job.title)
                await loop.loop.run_schedule_job(occurrence.job, occurrence)

            async def cancel_subagents_for_job(job_id: str) -> None:
                results = await asyncio.gather(
                    *(
                        coordinator.cancel_source_and_wait(job_id=job_id)
                        for coordinator in tuple(workspace._subagent_coordinators.values())
                    ),
                    return_exceptions=True,
                )
                failures = [result for result in results if isinstance(result, BaseException)]
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup("Schedule SubAgent cleanup failed", failures)

            registered_resources = False
            release_shared = self._configuration_resources.retain_shared(shared)
            try:
                state = WorkspaceState(workspace.workspace_path)
                state.initialize(
                    agent_home_root=self.agent_home.path,
                    allow_agent_home_chat=workspace.allow_agent_home_chat,
                )
                recover_session_deletions(state)
                SubAgentRecordStore.recover_workspace(state)
                restore_manager = RestoreManager(state)
                workspace._restore_result = await restore_manager.recover_pending()
                workspace.workspace_state = state

                configured = await self._configuration_resources.workspace(
                    workspace.workspace_id, workspace.workspace_path, shared,
                )
                manager = configured.mcp_manager
                workspace._mcp_manager = manager
                workspace._mcp_startup_report = configured.mcp_report
                workspace._mcp_snapshot = workspace._mcp_startup_report.snapshot
                workspace._router = shared.router
                workspace._mcp_keyword_preparer = configured.keyword_preparer
                workspace._mcp_keywords = configured.keywords
                memory_manager = MemoryManager(state)

                async def prepare_dream() -> Any:
                    current = self._capture_configuration()
                    prepared, activated = await self._prepare_execution_resources(workspace, current)
                    release = self._configuration_resources.retain(prepared)
                    try:
                        if activated:
                            await self._emit_configuration_event()
                        return (
                            prepared.shared.router, prepared.shared.router.route_status("memory"),
                            current.configuration.memory.batch_size, release,
                        )
                    except BaseException:
                        release()
                        raise

                dream = Dream(
                    memory_manager=memory_manager,
                    model_router=workspace._router,
                    batch_size=workspace.configuration.memory.batch_size,
                    memory_route_status=workspace._router.route_status("memory"),
                    prepare_execution=prepare_dream,
                )
                workspace._memory_manager = memory_manager
                workspace._dream = dream
                schedule = ScheduleService(
                    workspace_state=state,
                    clock=AsyncioSchedulerClock(now=local_now),
                    execute_user_job=execute_user_job,
                    execute_user_occurrence=execute_user_occurrence,
                    permission_snapshot_factory=workspace._capture_schedule_permission_snapshot,
                    cancel_confirmation_owner=self.confirmation.cancel_owner,
                    block_subagent_source=workspace.block_subagent_source,
                    unblock_subagent_source=workspace.unblock_subagent_source,
                    cancel_subagents_for_job=cancel_subagents_for_job,
                    execute_dream=dream.run,
                    timezone_name=get_localzone_name(),
                )
                workspace._schedule_service = schedule
                self.workspace_resources.register_resources(
                    WorkspaceResources(
                        workspace_id=workspace.workspace_id,
                        workspace_state=state,
                        memory_manager=memory_manager,
                        dream=dream,
                        schedule_service=schedule,
                        workspace_path=workspace.workspace_path,
                        mcp_manager=manager,
                        mcp_startup_report=workspace._mcp_startup_report,
                        mcp_snapshot=workspace._mcp_snapshot,
                        mcp_keywords=workspace._mcp_keywords,
                        mcp_keyword_preparer=workspace._mcp_keyword_preparer,
                        router=workspace._router,
                    )
                )
                registered_resources = True
                schedule.set_admission_guard(lambda: workspace.schedule_admitted)
                schedule._prepare_start()
                await schedule.register_dream_job(
                    schedule=JobSchedule.from_cron_input(
                        workspace.configuration.memory.schedule,
                        get_localzone_name(),
                    )
                )
                self._memory_schedules[workspace.workspace_id] = workspace.configuration.memory.schedule
                workspace._started = True
                if self._configuration_editor.activate(snapshot):
                    await self._emit_configuration_event()
            except BaseException as error:
                cleanup_targets: list[Awaitable[None]] = []
                if registered_resources:
                    cleanup_targets.append(self._close_workspace_resources(workspace))
                else:
                    for resource in (workspace._schedule_service, workspace._dream, workspace._mcp_manager):
                        if resource is not None:
                            cleanup_targets.append(resource.close())
                cleanup_results = await asyncio.gather(*cleanup_targets, return_exceptions=True)
                await self._configuration_resources.close_workspace(workspace.workspace_id)
                cleanup_errors = [
                    result for result in cleanup_results if isinstance(result, BaseException)
                ]
                workspace._started = False
                if not cleanup_errors:
                    workspace._schedule_service = None
                    workspace._dream = None
                    workspace._memory_manager = None
                    workspace._mcp_manager = None
                    workspace._mcp_startup_report = None
                    workspace._mcp_snapshot = ()
                    workspace._mcp_keywords = {}
                    workspace._mcp_keyword_preparer = None
                    workspace._router = None
                if cleanup_errors:
                    raise error from BaseExceptionGroup(
                        "Workspace startup cleanup failed", cleanup_errors
                    )
                raise
            finally:
                release_shared()

    async def _close_workspace_resources(self, workspace: WorkspaceRecord) -> None:
        """Close this Workspace's resources without closing the service globals."""
        if workspace.workspace_id in self.workspace_resources.resources:
            await self.workspace_resources.close_workspace(
                workspace.workspace_id,
                close_foreground=workspace._close_all_loops,
                drain_confirmation_aborts=True,
            )
        else:
            await workspace._close_all_loops()
        await self._configuration_resources.close_workspace(workspace.workspace_id)
        self._memory_schedules.pop(workspace.workspace_id, None)

    async def start(self) -> None:
        async with self._start_lock:
            await self._start_owned()

    async def _start_owned(self) -> None:
        if self.state != "starting":
            return
        self.agent_home.initialize()
        self.configuration = self._configuration_editor.start()
        if self.configuration is not None:
            await self._prepare_shared_configuration(self._capture_configuration())
        self.confirmation.bind_presenter(self._presenter)
        for record in self.projects.list():
            if record.schedule_state == "removing" and record.removal_error is None:
                self.projects.record_removal_failure(
                    record.project_id,
                    "Project removal was interrupted; retry to finish stopping its work.",
                )
            if (
                self.configuration is not None
                and record.schedule_state == "available"
                and record.path.is_dir()
            ):
                await self._get_or_create_workspace(record.path)
        self._started_at = monotonic()
        self.state = "ready"
        await self._reconcile_schedule_admission()
        self._global_reconnect_task = asyncio.create_task(self._stop_after_grace())

    @property
    def workspace_resources(self) -> WorkspaceResourceManager:
        return self._workspace_resources

    @property
    def skill_loader(self) -> SkillLoader:
        if self._skill_loader is None:
            raise RuntimeError("Skill Loader is unavailable")
        return self._skill_loader

    @property
    def model_router(self) -> ModelRouter:
        if self._model_router is None:
            raise RuntimeError("Model Router is unavailable")
        return self._model_router

    @property
    def mcp_manager(self) -> MCPRuntimeManager:
        if self._mcp_manager is None:
            raise RuntimeError("MCP Runtime Manager is unavailable")
        return self._mcp_manager

    @property
    def exec_host(self) -> ExecHost:
        if self._exec_host is None:
            raise RuntimeError("Exec Host is unavailable")
        return self._exec_host

    @property
    def built_in_tool_catalog(self) -> BuiltInToolCatalog:
        if self._built_in_tool_catalog is None:
            raise RuntimeError("Built-in Tool Catalog is unavailable")
        return self._built_in_tool_catalog

    def _initialize_shared_resources(self, configuration: UserConfiguration) -> None:
        """Prepare local projections before asynchronously opening shared resources."""
        if self._skill_loader is None:
            skill_loader = SkillLoader(
                root=self.agent_home.skills_directory,
                reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
                enable_always_load=configuration.runtime.enable_skill_always_load,
            )
            skill_loader.load()
            self._skill_loader = skill_loader
        elif self._skill_always_load != configuration.runtime.enable_skill_always_load:
            self._skill_loader = self._skill_loader.with_always_load(
                configuration.runtime.enable_skill_always_load,
            )
        self._skill_always_load = configuration.runtime.enable_skill_always_load

    def _exec_host_for(self, configuration: UserConfiguration) -> ExecHost:
        selector = configuration.runtime.exec_shell
        if selector not in self._exec_hosts:
            self._exec_hosts[selector] = create_exec_host(resolve_exec_shell(selector))
        return self._exec_hosts[selector]

    def _capture_configuration(self) -> ConfigurationSnapshot:
        """Publish valid external edits without changing collaborators owned by live Runs."""
        snapshot = self._configuration_editor.capture()
        self.configuration = snapshot.configuration
        for client in self._clients.values():
            client.permission_control.reconfigure(snapshot.configuration.runtime.permission_level)
        for workspace in self._workspaces.values():
            workspace.configuration = snapshot.configuration
            workspace._schedule_permission.reconfigure(snapshot.configuration.runtime.permission_level)
        return snapshot

    async def _prepare_shared_configuration(
        self, snapshot: ConfigurationSnapshot,
    ) -> SharedConfigurationResources:
        self._initialize_shared_resources(snapshot.configuration)
        shared = await self._configuration_resources.shared(
            snapshot, self.skill_loader, self._exec_host_for(snapshot.configuration),
        )
        if self.configuration == snapshot.configuration:
            self._model_router = shared.router
            self._mcp_manager = shared.mcp_manager
            self._exec_host = shared.exec_host
            self._built_in_tool_catalog = shared.built_in_catalog
            if self._chat_effort_override is not None:
                shared.router.set_reasoning_effort(self._chat_effort_override)
        return shared

    async def _prepare_execution_resources(
        self, workspace: WorkspaceRecord, snapshot: ConfigurationSnapshot,
    ) -> tuple[WorkspaceConfigurationResources, bool]:
        await self._synchronize_memory_schedules()
        shared = await self._prepare_shared_configuration(snapshot)
        configured = await self._configuration_resources.workspace(
            workspace.workspace_id, workspace.workspace_path, shared,
        )
        resources = workspace.resources
        resources.router = shared.router
        resources.mcp_manager = configured.mcp_manager
        resources.mcp_startup_report = configured.mcp_report
        resources.mcp_snapshot = configured.mcp_report.snapshot
        resources.mcp_keywords = configured.keywords
        resources.mcp_keyword_preparer = configured.keyword_preparer
        workspace._router = shared.router
        workspace._exec_host = shared.exec_host
        workspace._mcp_manager = configured.mcp_manager
        workspace._mcp_startup_report = configured.mcp_report
        workspace._mcp_snapshot = configured.mcp_report.snapshot
        workspace._mcp_keywords = configured.keywords
        workspace._mcp_keyword_preparer = configured.keyword_preparer
        activated = self._configuration_editor.activate(snapshot)
        return configured, activated

    async def _synchronize_memory_schedules(self) -> None:
        for workspace in tuple(self._workspaces.values()):
            if workspace._closed or not workspace._started:
                continue
            schedule = workspace.configuration.memory.schedule
            if self._memory_schedules.get(workspace.workspace_id) != schedule:
                await workspace.schedule_service.register_dream_job(
                    schedule=JobSchedule.from_cron_input(schedule, get_localzone_name()),
                )
                self._memory_schedules[workspace.workspace_id] = schedule

    def reload_skills(self) -> tuple[SkillMetadata, ...]:
        """Reload and publish the globally validated Skill snapshot."""
        self.skill_loader.load()
        return self.skill_loader.metadata

    @property
    def configuration_ready(self) -> bool:
        """Return whether new Agent and Schedule work may be admitted."""
        return (
            self.configuration is not None
            and self._configuration_editor.ready
        )

    def config_view(self) -> dict[str, object]:
        return self._configuration_editor.view()

    def configuration_text_view(self) -> dict[str, object]:
        return self._configuration_editor.text_view()

    def configuration_startup_view(self) -> dict[str, object]:
        return self._configuration_editor.startup_view()


    def available_models_view(self) -> dict[str, object]:
        """Return active provider models with capacities and the effective chat route."""
        configuration: UserConfiguration | None
        try:
            configuration = self._capture_configuration().configuration
        except ServiceError as error:
            if error.code != "config_invalid":
                raise
            configuration = self.configuration
        if configuration is None:
            return {"models": [], "default_combination": None}

        try:
            default = configuration.resolve_route("chat")
        except ConfigError:
            default_combination = None
        else:
            default_combination = {
                "provider_id": default.provider.provider_id,
                "model": default.route.model,
                "reasoning_effort": self.reasoning_effort,
            }
        models = [
            {
                "provider_id": provider_id,
                "model": model,
                "context_window": parameters.context_window,
            }
            for provider_id, provider in configuration.models.providers.items()
            if provider.is_usable
            for model, parameters in provider.models.items()
            if default_combination is not None
        ]
        return {"models": models, "default_combination": default_combination}

    def get_input_capabilities(self) -> dict[str, object]:
        """Return published Management Command and Skill metadata for clients."""
        from aide.management.commands import MANAGEMENT_COMMANDS

        skills = () if self._skill_loader is None else self._skill_loader.metadata
        return {
            "management_commands": [command.token for command in MANAGEMENT_COMMANDS],
            "skill_metadata": [
                {"name": item.name, "description": item.description, "path": str(item.path)}
                for item in skills
            ],
        }


    def configuration_status_text(self) -> str:
        return self._configuration_editor.status_text()

    async def update_configuration(
        self,
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
        baseline: Mapping[str, object] | None = None,
        baseline_secrets: Mapping[str, object] | None = None,
        overwrite_conflicts: bool = False,
        editor_id: str | None = None,
        edit_sequence: int | None = None,
    ) -> dict[str, object]:
        """Persist one safe configuration patch for subsequent Agent Runs."""
        return await self._persist_configuration_edit(
            "patch",
            request_id, expected_revision, fields, secrets,
            client_id=client_id, baseline=baseline, baseline_secrets=baseline_secrets,
            overwrite_conflicts=overwrite_conflicts, editor_id=editor_id, edit_sequence=edit_sequence,
        )

    async def repair_configuration(
        self,
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
        baseline: Mapping[str, object] | None = None,
        baseline_secrets: Mapping[str, object] | None = None,
        overwrite_conflicts: bool = False,
        editor_id: str | None = None,
        edit_sequence: int | None = None,
    ) -> dict[str, object]:
        """Persist a first-use or malformed-file repair for subsequent Agent Runs."""
        return await self._persist_configuration_edit(
            "repair",
            request_id, expected_revision, fields, secrets,
            client_id=client_id, baseline=baseline, baseline_secrets=baseline_secrets,
            overwrite_conflicts=overwrite_conflicts, editor_id=editor_id, edit_sequence=edit_sequence,
        )

    async def _persist_configuration_edit(
        self,
        action: Literal["patch", "repair"],
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
        baseline: Mapping[str, object] | None = None,
        baseline_secrets: Mapping[str, object] | None = None,
        overwrite_conflicts: bool = False,
        editor_id: str | None = None,
        edit_sequence: int | None = None,
    ) -> dict[str, object]:
        """Validate service admission and publish a completed configuration save."""
        if not request_id:
            raise service_error("validation_error", "Request ID is required.", status=422)
        if client_id is not None:
            self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        saved = await self._configuration_editor.save(ConfigurationEdit(
            action=action, request_id=request_id, expected_revision=expected_revision,
            fields=fields, secrets=secrets, client_id=client_id, baseline=baseline,
            baseline_secrets=baseline_secrets, overwrite_conflicts=overwrite_conflicts,
            editor_id=editor_id, edit_sequence=edit_sequence,
        ))
        if saved.changed:
            self._capture_configuration()
            self._initialize_shared_resources(cast(UserConfiguration, self.configuration))
            await self._synchronize_memory_schedules()
            await self._emit_configuration_event()
        return saved.view

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """Return the global chat control independently of saved configuration."""
        if self._chat_effort_override is not None:
            return self._chat_effort_override
        if self.configuration is None:
            raise service_error("config_invalid", "User Configuration is unavailable.", status=422)
        return self.configuration.resolve_route("chat").route.reasoning_effort

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Publish the shared chat control for every current and future Workspace."""
        self._chat_effort_override = effort
        if self._model_router is not None:
            self._model_router.set_reasoning_effort(effort)

    async def persist_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Best-effort persistence does not activate other saved settings."""
        await self._configuration_editor.persist_reasoning_effort(effort)
        await self._emit_configuration_event()

    async def _emit_configuration_event(self) -> None:
        application = self._configuration_editor.application()
        if application is None:
            return
        await self.emit(
            "config.application",
            workspace_id=None,
            session_id=None,
            run_id=None,
            payload=application,
            target_client_ids=tuple(self._clients),
        )

    def client(self, client_id: str) -> ClientState:
        return self._require_client(client_id)

    def workspace_audience(self, workspace_id: str) -> tuple[str, ...]:
        """Return clients that may receive Workspace-scoped metadata events."""
        return tuple(
            client.client_id
            for client in self._clients.values()
            if workspace_id in client.attached_workspaces
            or any(candidate_workspace == workspace_id for candidate_workspace, _ in client.claimed)
            or client.current_workspace_id == workspace_id
        )

    def require_web_client_available(self) -> None:
        """Reject another Web participant until the current one's cleanup finishes."""
        if any(client.kind == "web" for client in self._clients.values()):
            raise service_error(
                "web_client_exists", "Another Web Client is already open.", status=409
            )

    async def register_client(
        self,
        kind: str,
        reconnect_credential: str | None = None,
    ) -> ClientState:
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        if reconnect_credential is not None:
            client_id = self._client_by_reconnect.get(reconnect_credential)
            if client_id is None:
                raise service_error(
                    "stale_client", "This Client reconnect credential is no longer valid.",
                    retryable=True,
                )
            client = self._require_client(client_id)
            if client.reconnect_blocked:
                raise service_error(
                    "project_reentry_required",
                    "This CLI must explicitly re-enter the Project after removal.",
                )
            if client.kind == kind == "web" and client.connected:
                self.require_web_client_available()
            if client.kind != kind or client.connected:
                raise service_error(
                    "client_already_connected",
                    "This Client is already connected or has a different kind.",
                )
            self._client_by_reconnect.pop(client.reconnect_credential, None)
            client.reconnect_credential = str(uuid4())
            client.web_control_credential = str(uuid4()) if kind == "web" else None
            self._client_by_reconnect[client.reconnect_credential] = client_id
            return client
        if kind == "web":
            self.require_web_client_available()
        permission = (
            self.configuration.runtime.permission_level if self.configuration else "workspace-write"
        )
        client = ClientState(str(uuid4()), kind, str(uuid4()), RuntimePermissionControl(permission))
        if kind == "web":
            client.web_control_credential = str(uuid4())
        self._clients[client.client_id] = client
        self._client_by_reconnect[client.reconnect_credential] = client.client_id
        if kind == "web":
            deadline = self._monotonic() + self.reconnect_timeout
            client.reconnect_deadline = deadline
            client.disconnect_task = asyncio.create_task(
                self._expire_client_later(client.client_id, deadline)
            )
        return client

    def client_permission(self, client_id: str | None) -> RuntimePermissionControl:
        if client_id is None:
            return RuntimePermissionControl(
                self.configuration.runtime.permission_level
                if self.configuration
                else "workspace-write"
            )
        client = self._clients.get(client_id)
        if client is None:
            raise service_error("unauthenticated", "Client identity is not recognized.", status=401)
        return client.permission_control

    def client_claim_released(self, client_id: str, workspace_id: str, session_id: str) -> None:
        client = self._clients.get(client_id)
        if client is not None:
            client.claimed.discard((workspace_id, session_id))
            if (
                client.current_workspace_id == workspace_id
                and client.current_session_id == session_id
            ):
                client.current_session_id = None

    async def connect_client(
        self,
        client_id: str,
        sink: ServiceSink,
        *,
        wait_for_subscribe: bool = False,
    ) -> None:
        client = self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        if client.connected:
            raise service_error("client_already_connected", "This Client already has a connection.")
        client.connected = True
        client.ever_connected = True
        client.sink = sink
        client.subscribed = False
        client.expired = False
        client.reconnect_deadline = None
        if client.disconnect_task is not None:
            client.disconnect_task.cancel()
            client.disconnect_task = None
        if self._global_reconnect_task is not None:
            self._global_reconnect_task.cancel()
            self._global_reconnect_task = None
        if self.state == "reconnecting":
            self.state = "ready"
        for workspace in self._workspaces.values():
            workspace.set_client_connection(client_id, connected=True)
        await self._reconcile_schedule_admission()
        if not wait_for_subscribe:
            async with client.delivery_lock:
                client.subscribed = True
                if client.resync_required:
                    await self._send_snapshot_required(client, reason="slow_consumer")
                    if client.connected:
                        client.resync_required = False
                else:
                    for event in tuple(client.events):
                        await self._send_event(client, event)

    async def disconnect_client(self, client_id: str, *, sink: ServiceSink | None = None) -> None:
        client = self._clients.get(client_id)
        if client is None or (sink is not None and client.sink is not sink):
            return
        client.connected = False
        client.sink = None
        client.subscribed = False
        for workspace in self._workspaces.values():
            workspace.set_client_connection(client_id, connected=False)
        if client.disconnect_task is not None:
            client.disconnect_task.cancel()
        if self.state in {"draining", "stopped"}:
            return
        expiry_deadline = self._monotonic() + self.reconnect_timeout
        client.reconnect_deadline = expiry_deadline
        client.disconnect_task = asyncio.create_task(
            self._expire_client_later(client_id, expiry_deadline)
        )
        if not any(candidate.connected for candidate in self._clients.values()):
            if self.state == "ready":
                self.state = "reconnecting"
            await self._reconcile_schedule_admission()
            if self.state == "reconnecting":
                pending_web_deadline = next(
                    (
                        candidate.reconnect_deadline
                        for candidate in self._clients.values()
                        if candidate.kind == "web"
                        and not candidate.ever_connected
                        and candidate.reconnect_deadline is not None
                        and candidate.reconnect_deadline > self._monotonic()
                    ),
                    None,
                )
                self._global_reconnect_task = asyncio.create_task(
                    self._stop_after_grace(pending_web_deadline or self._monotonic())
                )
        else:
            await self._reconcile_schedule_admission()

    async def attach_workspace(self, client_id: str, path: Path) -> WorkspaceRecord:
        async with self._project_lifecycle_lock:
            return await self._attach_workspace(client_id, path)

    async def enter_workspace(self, client_id: str, path: Path) -> dict[str, object]:
        if self._require_client(client_id).kind != "cli":
            raise service_error("forbidden", "Only CLI clients may attach a directory.", status=403)
        workspace = await self.attach_workspace(client_id, path)
        return {"workspace_id": workspace.workspace_id, "project_id": None}

    async def _attach_workspace(self, client_id: str, path: Path) -> WorkspaceRecord:
        client = self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        if not path.is_absolute():
            raise service_error("validation_error", "Workspace path must be absolute.", status=422)
        if not path.exists() or not path.is_dir():
            raise service_error("not_found", "Workspace directory is unavailable.", status=404)
        try:
            normalized = path.resolve(strict=True)
        except OSError as error:
            raise service_error(
                "persistence_error", "Workspace directory could not be resolved"
            ) from error
        workspace_key = os.path.normcase(str(normalized))
        if workspace_key in client.blocked_workspace_keys:
            raise service_error(
                "project_reentry_required",
                "This CLI must explicitly re-enter the Project after removal.",
            )
        registered = next(
            (
                record
                for record in self.projects.list()
                if os.path.normcase(str(record.path.resolve(strict=False))) == workspace_key
            ),
            None,
        )
        if registered is not None and registered.schedule_state in {"removing", "failed"}:
            raise service_error(
                "admission_closed",
                "Project work is being removed and cannot be admitted.",
                retryable=True,
            )
        runtime = await self._get_or_create_workspace(normalized)
        client.attached_workspaces.add(runtime.workspace_id)
        await self._reconcile_schedule_admission()
        client.reconnect_blocked = False
        return runtime

    async def _get_or_create_workspace(
        self, path: Path, *, allow_agent_home_chat: bool = False
    ) -> WorkspaceRecord:
        self._capture_configuration()
        key = os.path.normcase(str(path.resolve(strict=True)))
        async with self._lock:
            workspace_id = self._workspace_keys.get(key)
            if workspace_id is not None:
                runtime = self._workspaces[workspace_id]
                if runtime._closed:
                    raise service_error(
                        "admission_closed", "Workspace cleanup has not completed.", retryable=True
                    )
                return runtime
            if self.configuration is None or not self._configuration_editor.ready:
                raise service_error(
                    "config_invalid", "User Configuration is unavailable.", status=422
                )
            workspace_id = str(uuid4())
            runtime = WorkspaceRecord(
                self,
                path,
                self.configuration,
                workspace_id=workspace_id,
                allow_agent_home_chat=allow_agent_home_chat,
            )
            await runtime.start()
            self._workspace_keys[key] = workspace_id
            self._workspaces[workspace_id] = runtime
            return runtime

    def _schedule_allowed(self, workspace: WorkspaceRecord) -> bool:
        key = os.path.normcase(str(workspace.workspace_path))
        for record in self.projects.list():
            if os.path.normcase(str(record.path.resolve(strict=False))) == key:
                return record.schedule_state == "available"
        return any(
            self._clients[client_id].connected and not self._clients[client_id].expired
            for client_id in self._workspace_clients(workspace.workspace_id)
        )

    def _schedule_admission_open(self) -> bool:
        return (
            self.state == "ready"
            and self.configuration_ready
            and any(client.connected for client in self._clients.values())
        )

    async def _reconcile_schedule_admission(self) -> None:
        """Apply the single service-wide Schedule admission gate to every Workspace."""
        async with self._schedule_admission_lock:
            for workspace in tuple(self._workspaces.values()):
                if self._schedule_admission_open() and self._schedule_allowed(workspace):
                    await workspace.activate_schedule()
                else:
                    await workspace.pause_schedule_admission()

    async def register_project(
        self, client_id: str, path: Path
    ) -> tuple[ProjectRecord, WorkspaceRecord, tuple[ScheduleJob, ...]]:
        async with self._project_lifecycle_lock:
            self._require_client(client_id)
            if self.configuration is None or not self._configuration_editor.ready:
                raise service_error(
                    "config_invalid", "User Configuration is unavailable.", status=422
                )
            try:
                record = self.projects.register(path, schedule_state="awaiting_resume")
            except ProjectCatalogError as error:
                raise _project_catalog_service_error(error) from error
            except OSError as error:
                raise service_error(
                    "persistence_error", "The Project catalog could not be saved.", status=500
                ) from error
            if record.schedule_state in {"removing", "failed"}:
                raise service_error(
                    "project_removal_blocked",
                    "Project removal has not completed; retry the removal before re-entering.",
                    retryable=True,
                )
            client = self._require_client(client_id)
            client.blocked_workspace_keys.discard(
                os.path.normcase(str(record.path.resolve(strict=False)))
            )
            if record.schedule_state == "awaiting_resume":
                key = os.path.normcase(str(record.path.resolve(strict=False)))
                workspace_id = self._workspace_keys.get(key)
                if workspace_id is not None:
                    await self._workspaces[workspace_id].pause_schedule_admission()
            workspace = await self._attach_workspace(client_id, record.path)
            jobs = await workspace.schedule_service.public_snapshot()
            if not jobs and record.schedule_state == "awaiting_resume":
                record = self.projects.set_schedule_state(record.project_id, "available")
                await self._reconcile_schedule_admission()
            return record, workspace, jobs

    async def register_project_entry(self, client_id: str, path: Path) -> ProjectRegistrationDTO:
        record, workspace, jobs = await self.register_project(client_id, path)
        return {
            "project_id": record.project_id,
            "workspace_id": workspace.workspace_id,
            "schedule_state": record.schedule_state,
            "saved_jobs": [_project_job_summary(job) for job in jobs],
        }

    async def list_projects(self, client_id: str) -> ProjectListDTO:
        self._require_client(client_id)
        try:
            records = self.projects.list()
        except ProjectCatalogError as error:
            raise service_error(
                "persistence_error", "The Project catalog could not be read safely.", status=500
            ) from error
        projects: list[ProjectSummaryDTO] = []
        for record in records:
            saved_jobs, schedule_status = await self.project_schedule_snapshot(record)
            project: ProjectSummaryDTO = {
                "project_id": record.project_id,
                "path": str(record.path),
                "name": record.path.name,
                "schedule_state": record.schedule_state,
                "available": record.path.is_dir() and self.configuration_ready,
                "saved_jobs": [_project_job_summary(job) for job in saved_jobs],
                "schedule_status": schedule_status,
            }
            if record.removal_operation_id is not None:
                project["removal_operation_id"] = record.removal_operation_id
            if record.removal_error is not None:
                project["removal_error"] = record.removal_error
            projects.append(project)
        return {"projects": projects}

    async def project_schedule_snapshot(
        self, record: ProjectRecord
    ) -> tuple[tuple[ScheduleJob, ...], dict[str, object] | None]:
        """Read one Project's Jobs and live status across the removal boundary."""
        async with self._project_lifecycle_lock:
            current = next(
                (item for item in self.projects.list() if item.project_id == record.project_id),
                None,
            )
            if current is None or not current.path.is_dir():
                return (), None
            record = current
            key = os.path.normcase(str(record.path.resolve(strict=False)))
            workspace_id = self._workspace_keys.get(key)
            if workspace_id is None:
                if self.state in {"draining", "stopped"}:
                    return (), None
                if not self.configuration_ready or record.schedule_state != "available":
                    state = WorkspaceState(record.path)
                    try:
                        state.path.lstat()
                    except FileNotFoundError:
                        return (), None
                    try:
                        jobs = await WorkspaceScheduleStore(state).public_snapshot()
                    except FileNotFoundError:
                        jobs = ()
                    return jobs, None
                workspace = await self._get_or_create_workspace(record.path)
            else:
                workspace = self._workspaces[workspace_id]
            jobs = await workspace.schedule_service.public_snapshot()
            return jobs, workspace.schedule_status()

    def _schedule_workspace(self, client_id: str, workspace_id: str) -> WorkspaceRecord:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        if (
            workspace_id not in client.attached_workspaces
            and client.current_workspace_id != workspace_id
            and not any(
                candidate_workspace == workspace_id for candidate_workspace, _ in client.claimed
            )
        ):
            raise service_error(
                "forbidden",
                "This Client is not attached to the requested Workspace.",
                status=403,
            )
        return workspace

    async def list_schedule_jobs(self, client_id: str, workspace_id: str) -> dict[str, object]:
        workspace = self._schedule_workspace(client_id, workspace_id)
        jobs = await workspace.schedule_service.public_snapshot()
        return {
            "workspace_id": workspace_id,
            "jobs": [
                _schedule_job_projection(
                    job,
                    active=workspace.schedule_service.is_job_active(job.job_id),
                )
                for job in jobs
            ],
            "status": workspace.schedule_status(),
        }

    async def get_schedule_job(
        self, client_id: str, workspace_id: str, job_id: str
    ) -> dict[str, object]:
        workspace = self._schedule_workspace(client_id, workspace_id)
        jobs = await workspace.schedule_service.public_snapshot()
        job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)
        if job is None:
            raise service_error("not_found", "Schedule Job was not found.", status=404)
        return {
            "workspace_id": workspace_id,
            "job": _schedule_job_projection(
                job,
                active=workspace.schedule_service.is_job_active(job.job_id),
            ),
            "status": workspace.schedule_status(),
        }

    async def get_schedule_job_history(
        self,
        client_id: str,
        workspace_id: str,
        job_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        workspace = self._schedule_workspace(client_id, workspace_id)
        jobs = await workspace.schedule_service.public_snapshot()
        job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)
        if job is None:
            raise service_error("not_found", "Schedule Job was not found.", status=404)
        try:
            history = read_schedule_history(
                workspace.workspace_state,
                job.job_id,
                workspace_id=workspace_id,
                cursor=cursor,
                limit=limit,
            )
        except ScheduleHistoryRequestError as error:
            raise service_error("validation_error", str(error), status=422) from error
        except (OSError, ScheduleHistoryPersistenceError) as error:
            raise service_error(
                "persistence_error",
                "Schedule history could not be loaded safely.",
                status=500,
            ) from error
        return {
            "workspace_id": workspace_id,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "job": _schedule_job_projection(
                job,
                active=workspace.schedule_service.is_job_active(job.job_id),
            ),
            "status": workspace.schedule_status(),
            **history,
        }

    async def create_schedule_job(
        self,
        client_id: str,
        workspace_id: str,
        payload: Mapping[str, object],
        request_id: str,
    ) -> dict[str, object]:
        workspace = self._schedule_workspace(client_id, workspace_id)
        fingerprint = _schedule_request_fingerprint(workspace_id, "create", payload)
        async with self._schedule_mutation_lock, self._project_lifecycle_lock:
            workspace = self._schedule_workspace(client_id, workspace_id)
            previous = self._schedule_mutation_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error(
                        "request_reused",
                        "request_id was already used for a different Schedule request.",
                        status=409,
                    )
                return previous[1]
            message, title, schedule = _schedule_job_input(payload)
            timestamp = _schedule_epoch_milliseconds(workspace.schedule_service.current_time())
            job = ScheduleJob(
                job_id=str(uuid4()),
                message=message,
                title=title,
                schedule=schedule,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
            )
            try:
                workspace.schedule_service.validate_job_schedule(job)
            except (ValueError, OverflowError) as error:
                field_name = "every_seconds" if schedule.kind == "every" else "cron_expr"
                raise service_error(
                    "validation_error",
                    "Schedule Job input is invalid.",
                    status=400,
                    field_errors={field_name: "must define a representable next occurrence"},
                ) from error
            try:
                await workspace.schedule_service.add_user_job(job)
            except ScheduleStoreFaultedError as error:
                raise service_error(
                    "schedule_unavailable",
                    "Schedule state is unavailable; retry after it is repaired.",
                    retryable=True,
                ) from error
            except (ScheduleStateError, OSError, RuntimeError) as error:
                raise service_error(
                    "schedule_update_failed",
                    "Schedule Job could not be created.",
                    retryable=True,
                ) from error
            except ValueError as error:
                raise service_error("validation_error", str(error), status=400) from error
            result: dict[str, object] = {
                "request_id": request_id,
                "workspace_id": workspace_id,
                "job": _schedule_job_projection(job, active=False),
                "status": workspace.schedule_status(),
            }
            self._schedule_mutation_results[request_id] = (fingerprint, result)
            return result

    async def delete_schedule_job(
        self,
        client_id: str,
        workspace_id: str,
        job_id: str,
        request_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        workspace = self._schedule_workspace(client_id, workspace_id)
        fingerprint = _schedule_request_fingerprint(
            workspace_id,
            "delete",
            {**payload, "job_id": job_id},
        )
        async with self._schedule_mutation_lock, self._project_lifecycle_lock:
            workspace = self._schedule_workspace(client_id, workspace_id)
            previous = self._schedule_mutation_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error(
                        "request_reused",
                        "request_id was already used for a different Schedule request.",
                        status=409,
                    )
                return previous[1]
            job = await workspace.schedule_service.job_for_removal(job_id)
            removal_key = (workspace_id, job_id)
            pending = self._schedule_removal_jobs.get(removal_key)
            if job is None:
                raise service_error("not_found", "Schedule Job was not found.", status=404)
            was_active = (
                pending if pending is not None else workspace.schedule_service.is_job_active(job_id)
            )
            self._schedule_removal_jobs[removal_key] = was_active
            try:
                removed = await workspace.schedule_service.remove_user_job(job_id, expected=job)
            except ScheduleStaleRemovalError as error:
                raise service_error(
                    "schedule_changed",
                    "Schedule Job changed before removal; reload and try again.",
                    retryable=True,
                ) from error
            except ScheduleStoreFaultedError as error:
                raise service_error(
                    "schedule_unavailable",
                    "Schedule state is unavailable; retry after it is repaired.",
                    retryable=True,
                ) from error
            except Exception as error:
                raise service_error(
                    "schedule_update_failed",
                    "Schedule Job could not be deleted.",
                    retryable=True,
                ) from error
            if not removed:
                raise service_error(
                    "schedule_changed",
                    "Schedule Job changed before removal; reload and try again.",
                    retryable=True,
                )
            result: dict[str, object] = {
                "request_id": request_id,
                "workspace_id": workspace_id,
                "job_id": job_id,
                "deleted": True,
                "canceled": was_active,
                "job": _schedule_job_projection(job, active=False, status="deleted"),
                "status": workspace.schedule_status(),
            }
            self._schedule_mutation_results[request_id] = (fingerprint, result)
            self._schedule_removal_jobs.pop(removal_key, None)
            return result

    async def _project_workspace_owned(
        self, client_id: str, project_id: str
    ) -> tuple[ProjectRecord, WorkspaceRecord]:
        client = self._require_client(client_id)
        record = next(
            (item for item in self.projects.list() if item.project_id == project_id), None
        )
        if record is None:
            raise service_error("not_found", "Project registration was not found.", status=404)
        if record.schedule_state in {"removing", "failed"}:
            raise service_error(
                "admission_closed",
                "Project work is being removed and cannot be admitted.",
                retryable=True,
            )
        if not record.path.is_dir():
            raise service_error("not_found", "Project directory is unavailable.", status=404)
        workspace = await self._get_or_create_workspace(record.path)
        client.attached_workspaces.add(workspace.workspace_id)
        return record, workspace

    async def list_project_sessions_page(
        self,
        client_id: str,
        project_id: str,
        *,
        title: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[ProjectRecord, WorkspaceRecord, dict[str, object]]:
        async with self._project_lifecycle_lock:
            record, workspace = await self._project_workspace_owned(client_id, project_id)
            return (
                record,
                workspace,
                await workspace.list_sessions_page(
                    client_id,
                    title=title,
                    cursor=cursor,
                    limit=limit,
                    creation_scope="project",
                ),
            )

    async def rename_project_session(
        self,
        client_id: str,
        project_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        title: str,
        expected_metadata_version: int,
        request_id: str,
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            record, workspace = await self._project_workspace_owned(client_id, project_id)
            del record
            return await workspace.rename_session(
                client_id,
                session_id,
                claim_version,
                claim_credential,
                title,
                expected_metadata_version,
                request_id,
            )

    async def delete_project_session(
        self,
        client_id: str,
        project_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            record, workspace = await self._project_workspace_owned(client_id, project_id)
            del record
            result = await workspace.delete_session(
                client_id,
                session_id,
                claim_version,
                claim_credential,
                request_id,
            )
            return {"project_id": project_id, **result}

    async def project_session_deletion_status(
        self, client_id: str, project_id: str, session_id: str
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            result = await self.session_deletion_status(
                client_id, workspace.workspace_id, session_id
            )
            return {"project_id": project_id, **result}

    async def claim_project_session_deletion(
        self, client_id: str, project_id: str, session_id: str
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            result = await self.claim_session_deletion(
                client_id, workspace.workspace_id, session_id
            )
            return {"project_id": project_id, **result}

    async def enter_default_conversation_workspace(
        self, client_id: str, *, directory: str | None = None
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        if client.kind != "web":
            raise service_error(
                "forbidden", "Only Web clients may enter a Conversation Workspace.", status=403
            )
        try:
            known_directories = self.conversation_workspaces.list()
            if directory is None:
                configured_path = Path(
                    self._configuration_editor.default_chat_workspace()
                ).expanduser()
                path = self.conversation_workspaces.validate(configured_path)
                path.mkdir(parents=True, exist_ok=True)
            else:
                requested_path = self.conversation_workspaces.validate(Path(directory))
                requested_identity = os.path.normcase(str(requested_path))
                if not any(
                    os.path.normcase(str(known)) == requested_identity
                    for known in known_directories
                ):
                    raise service_error(
                        "not_found",
                        "Conversation Workspace was not found in the saved history.",
                        status=404,
                    )
                path = requested_path
                if not path.is_dir():
                    raise service_error(
                        "not_found", "Conversation Workspace is unavailable.", status=404
                    )
            path = path.resolve(strict=True)
            if not path.is_dir():
                raise service_error(
                    "workspace_unavailable",
                    "The default conversation directory is not a directory.",
                    status=422,
                )
            workspace = await self._get_or_create_workspace(path, allow_agent_home_chat=True)
            self.conversation_workspaces.remember(path)
        except ServiceError:
            raise
        except ConfigError as error:
            raise service_error(
                "config_invalid",
                "The default conversation directory setting is invalid.",
                status=422,
            ) from error
        except ConversationWorkspaceCatalogError as error:
            raise service_error(
                "persistence_error",
                "The saved Conversation Workspace list could not be read or updated safely.",
                status=500,
            ) from error
        except (OSError, RuntimeError, ValueError, WorkspaceStateError) as error:
            raise service_error(
                "workspace_unavailable",
                "The default conversation directory could not be created or opened. Check its path and permissions.",
                status=422,
            ) from error
        client.attached_workspaces.add(workspace.workspace_id)
        await self._reconcile_schedule_admission()
        return {
            "workspace_id": workspace.workspace_id,
            "directory": str(path),
            "project_id": None,
        }

    async def open_conversation(
        self,
        client_id: str,
        *,
        project_id: str | None = None,
        workspace_id: str | None = None,
        directory: str | None = None,
        session_id: str | None = None,
        create_new: bool = False,
        request_id: str | None = None,
    ) -> ConversationOpenDTO:
        client = self._require_client(client_id)
        fingerprint = (project_id, workspace_id, directory, session_id, create_new)
        async with client.conversation_lock:
            self._require_client(client_id)
            try:
                if request_id is not None:
                    if not request_id:
                        raise service_error(
                            "validation_error", "request_id is required.", status=422
                        )
                    previous = client.conversation_open_results.get(request_id)
                    if previous is not None:
                        if previous[0] != fingerprint:
                            raise service_error(
                                "request_reused", "Conversation request_id was reused."
                            )
                        claimed = previous[1]["claim"]
                        self.workspace(claimed["workspace_id"]).require_claim(
                            client_id,
                            claimed["session_id"],
                            claimed["claim_version"],
                            claimed["reconnect_credential"],
                        )
                        return deepcopy(previous[1])
                result = await self._open_conversation_once(
                    client,
                    project_id=project_id,
                    workspace_id=workspace_id,
                    directory=directory,
                    session_id=session_id,
                    create_new=create_new,
                )
            except ServiceError as error:
                current: ConversationClaimDTO | None = None
                active_workspace = self._workspaces.get(client.current_workspace_id or "")
                if active_workspace is not None:
                    claim = active_workspace._claims.get(client.current_session_id or "")
                    if (
                        claim is not None
                        and claim.client_id == client_id
                        and claim.status == "claimed"
                    ):
                        current = {
                            "workspace_id": claim.workspace_id,
                            "session_id": claim.session_id,
                            "claim_version": claim.version,
                            "reconnect_credential": claim.credential,
                        }
                failure: ConversationOpenFailureDTO = {
                    "target": {
                        "project_id": project_id,
                        "workspace_id": workspace_id,
                        "directory": directory,
                        "session_id": session_id,
                        "status": error.code,
                    },
                    "current_context": current,
                }
                if current is not None and active_workspace is not None:
                    try:
                        snapshot = active_workspace.session_snapshot(current["session_id"])
                    except (ServiceError, OSError, ValueError):
                        pass
                    else:
                        failure["current_conversation"] = {
                            "request_id": request_id or str(uuid4()),
                            "project_id": client.current_project_id,
                            "workspace_id": current["workspace_id"],
                            "directory": str(active_workspace.workspace_path),
                            "session_id": current["session_id"],
                            "claim": current,
                            "snapshot": snapshot,
                        }
                raise ConversationOpenError(error, failure) from error
            client.current_project_id = result["project_id"]
            if request_id is not None:
                client.conversation_open_results[request_id] = (fingerprint, deepcopy(result))
            return result

    async def _open_conversation_once(
        self,
        client: ClientState,
        *,
        project_id: str | None,
        workspace_id: str | None,
        directory: str | None,
        session_id: str | None,
        create_new: bool,
    ) -> ConversationOpenDTO:
        client_id = client.client_id
        if sum(value is not None for value in (project_id, workspace_id, directory)) > 1:
            raise service_error("validation_error", "Conversation scope is ambiguous.", status=422)
        if session_id is not None and not session_id:
            raise service_error("validation_error", "Session ID is invalid.", status=422)
        if create_new and session_id is not None:
            raise service_error(
                "validation_error", "A new Conversation cannot include a Session ID.", status=422
            )

        if project_id is not None:
            async with self._project_lifecycle_lock:
                _record, workspace = await self._project_workspace_owned(client_id, project_id)
                return await self._open_conversation_in_workspace(
                    client,
                    workspace,
                    project_id=project_id,
                    session_id=session_id,
                    create_new=create_new,
                )

        if directory is not None:
            if client.kind == "cli":
                entry = await self.enter_workspace(client_id, Path(directory))
                workspace_id = cast(str, entry["workspace_id"])
            else:
                entry = await self.enter_default_conversation_workspace(
                    client_id, directory=directory
                )
                workspace_id = cast(str, entry["workspace_id"])
        elif workspace_id is None:
            if client.kind != "web":
                workspace_id = client.current_workspace_id
            else:
                entry = await self.enter_default_conversation_workspace(client_id)
                workspace_id = cast(str, entry["workspace_id"])
            if workspace_id is None:
                raise service_error("validation_error", "Workspace ID is required.", status=422)

        workspace = self._schedule_workspace(client_id, workspace_id)
        return await self._open_conversation_in_workspace(
            client, workspace, project_id=None, session_id=session_id, create_new=create_new
        )

    async def _open_conversation_in_workspace(
        self,
        client: ClientState,
        workspace: WorkspaceRecord,
        *,
        project_id: str | None,
        session_id: str | None,
        create_new: bool,
    ) -> ConversationOpenDTO:
        if (
            session_id is None
            and not create_new
            and client.kind == "cli"
            and client.current_workspace_id == workspace.workspace_id
        ):
            session_id = client.current_session_id
        created = session_id is None
        if session_id is None:
            creation_scope: Literal["chat", "project"] = "chat"
            if project_id is not None or (
                client.kind == "cli"
                and any(
                    os.path.normcase(str(record.path))
                    == os.path.normcase(str(workspace.workspace_path))
                    for record in self.projects.list()
                )
            ):
                creation_scope = "project"
            reuse_startup_session = project_id is None and client.kind != "web" and not create_new
            session_id = await workspace.create_draft(
                client.client_id,
                reuse_startup_session=reuse_startup_session,
                creation_scope=creation_scope,
            )
        created_draft = created and workspace._draft_clients.get(session_id) == client.client_id
        try:
            claimed = await self.claim(client.client_id, workspace.workspace_id, session_id)
        except BaseException as error:
            if isinstance(error, ServiceError) and error.code == "not_found":
                error.field_errors["session_id"] = "Conversation Session was not found."
            if created_draft:
                async with workspace._lock:
                    if session_id not in workspace._claims:
                        await workspace._close_loop(session_id, abort=True)
                        workspace._draft_clients.pop(session_id, None)
            raise
        claim = cast(ConversationClaimDTO, claimed["claim"])
        snapshot = cast(dict[str, object], claimed["snapshot"])
        return {
            "project_id": project_id,
            "workspace_id": workspace.workspace_id,
            "directory": str(workspace.workspace_path),
            "session_id": session_id,
            "claim": claim,
            "snapshot": snapshot,
        }

    async def create_project_session(self, client_id: str, project_id: str) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            session_id = await workspace.create_draft(
                client_id,
                reuse_startup_session=False,
                creation_scope="project",
            )
            return {
                "project_id": project_id,
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
            }

    async def claim_project_session(
        self, client_id: str, project_id: str, session_id: str
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            return await self.claim(client_id, workspace.workspace_id, session_id)

    async def get_project_session(
        self,
        client_id: str,
        project_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> dict[str, object]:
        return await self.get_session_snapshot(
            client_id,
            None,
            session_id,
            claim_version,
            claim_credential,
            project_id=project_id,
        )

    async def get_session_snapshot(
        self,
        client_id: str,
        workspace_id: str | None,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, object]:
        if project_id is not None:
            async with self._project_lifecycle_lock:
                _record, workspace = await self._project_workspace_owned(client_id, project_id)
                return self._session_snapshot_for_workspace(
                    client_id,
                    workspace,
                    session_id,
                    claim_version,
                    claim_credential,
                    project_id=project_id,
                )
        else:
            if workspace_id is None:
                raise service_error("validation_error", "Workspace ID is required.", status=422)
            workspace = self._schedule_workspace(client_id, workspace_id)
        return self._session_snapshot_for_workspace(
            client_id,
            workspace,
            session_id,
            claim_version,
            claim_credential,
        )

    @staticmethod
    def _session_snapshot_for_workspace(
        client_id: str,
        workspace: WorkspaceRecord,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, object]:
        claim = workspace.require_claim(client_id, session_id, claim_version, claim_credential)
        workspace._ensure_session_available(session_id)
        result: dict[str, object] = {
            "workspace_id": workspace.workspace_id,
            "session_id": session_id,
            "claim_version": claim.version,
            "snapshot": workspace.session_snapshot(session_id),
        }
        if project_id is not None:
            result["project_id"] = project_id
        return result

    async def release_conversation(
        self,
        client_id: str,
        workspace_id: str | None,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        *,
        project_id: str | None = None,
    ) -> None:
        if project_id is None:
            if workspace_id is None:
                raise service_error("validation_error", "Workspace ID is required.", status=422)
            await self.release_claim(
                client_id, workspace_id, session_id, claim_version, claim_credential
            )
            return
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            if workspace_id is not None and workspace.workspace_id != workspace_id:
                raise service_error(
                    "validation_error", "Workspace does not match Project.", status=422
                )
            workspace.require_claim(client_id, session_id, claim_version, claim_credential)
            await workspace.release(client_id, session_id)

    async def resume_project_schedule(
        self, client_id: str, project_id: str, expected_job_ids: set[str]
    ) -> str:
        async with self._project_lifecycle_lock:
            self._require_client(client_id)
            if not self._schedule_admission_open():
                raise service_error("admission_closed", "Schedule admission is closed.")
            record = next(
                (item for item in self.projects.list() if item.project_id == project_id), None
            )
            if record is None:
                raise service_error("not_found", "Project registration was not found.", status=404)
            if record.schedule_state != "awaiting_resume":
                return record.schedule_state
            if not record.path.is_dir():
                raise service_error("not_found", "Project directory is unavailable.", status=404)
            workspace = await self._get_or_create_workspace(record.path)
            jobs = await workspace.schedule_service.public_snapshot()
            if {job.job_id for job in jobs} != expected_job_ids:
                raise service_error(
                    "stale_schedule_review", "Saved Schedule Jobs changed; review them again."
                )
            self.projects.set_schedule_state(project_id, "available")
            await self._reconcile_schedule_admission()
            return "available"

    async def start_project_removal(self, client_id: str, project_id: str) -> dict[str, object]:
        """Persist and start one idempotent Project removal operation."""
        self._require_client(client_id)
        async with self._project_lifecycle_lock:
            existing = self._project_removals.get(project_id)
            if existing is not None:
                if existing.task is not None and not existing.task.done():
                    return self._project_removal_response(existing)
                if existing.status == "completed":
                    return self._project_removal_response(existing)
            record = next(
                (item for item in self.projects.list() if item.project_id == project_id), None
            )
            if record is None:
                raise service_error(
                    "not_found", "Project registration was not found.", status=404
                )
            try:
                record = self.projects.begin_removal(
                    project_id,
                    record.removal_operation_id,
                )
            except (ProjectCatalogError, OSError) as error:
                raise service_error(
                    "persistence_error",
                    "The Project removal barrier could not be saved.",
                    status=500,
                ) from error
            try:
                await self._reconcile_schedule_admission()
            except asyncio.CancelledError:
                self.projects.record_removal_failure(
                    project_id,
                    "Project removal was interrupted; retry to finish stopping its work.",
                )
                raise
            except Exception as error:
                try:
                    self.projects.record_removal_failure(
                        project_id,
                        _PROJECT_REMOVAL_FAILURE_MESSAGE,
                    )
                except Exception:
                    pass
                raise service_error(
                    "project_removal_failed",
                    _PROJECT_REMOVAL_FAILURE_MESSAGE,
                    status=500,
                    retryable=True,
                ) from error

            key = os.path.normcase(str(record.path.resolve(strict=False)))
            workspace_id = self._workspace_keys.get(key)
            affected = self._workspace_clients(workspace_id)
            operation = _ProjectRemoval(
                project_id=project_id,
                operation_id=record.removal_operation_id or str(uuid4()),
                path=record.path,
                workspace_id=workspace_id,
                affected_client_ids=affected,
                notification_client_ids=tuple(dict.fromkeys((*affected, client_id))),
            )
            self._project_removals[project_id] = operation
            task = asyncio.create_task(self._run_project_removal(operation))
            operation.task = task
            task.add_done_callback(_consume_task_result)
            try:
                await self.emit(
                    "project.removal.started",
                    workspace_id=workspace_id,
                    session_id=None,
                    run_id=None,
                    payload={"project_id": project_id, "operation_id": operation.operation_id},
                    target_client_ids=operation.notification_client_ids,
                )
            except Exception:
                # Event delivery must not orphan a persisted removal operation.
                pass
            return self._project_removal_response(operation)

    async def project_removal_status(
        self, client_id: str, project_id: str, operation_id: str
    ) -> dict[str, object]:
        """Read one removal's current or persisted state by operation identity."""
        self._require_client(client_id)
        async with self._project_lifecycle_lock:
            operation = self._project_removals.get(project_id)
            if operation is not None and operation.operation_id == operation_id:
                return self._project_removal_response(operation)
            record = next(
                (item for item in self.projects.list() if item.project_id == project_id), None
            )
            if record is not None and record.removal_operation_id == operation_id:
                return {
                    "project_id": project_id,
                    "operation_id": operation_id,
                    "status": "failed" if record.removal_error is not None else "removing",
                }
            raise service_error("not_found", "Project removal was not found.", status=404)

    @staticmethod
    def _project_removal_response(operation: _ProjectRemoval) -> dict[str, object]:
        return {
            "project_id": operation.project_id,
            "operation_id": operation.operation_id,
            "status": operation.status,
        }

    def _workspace_clients(self, workspace_id: str | None) -> tuple[str, ...]:
        if workspace_id is None:
            return ()
        return tuple(
            client.client_id
            for client in self._clients.values()
            if workspace_id in client.attached_workspaces
            or client.current_workspace_id == workspace_id
            or any(candidate_workspace == workspace_id for candidate_workspace, _ in client.claimed)
        )

    async def _run_project_removal(self, operation: _ProjectRemoval) -> None:
        try:
            workspace = (
                None
                if operation.workspace_id is None
                else self._workspaces.get(operation.workspace_id)
            )
            if workspace is None and operation.path.is_dir():
                workspace = await self._get_or_create_workspace(operation.path)
                operation.workspace_id = workspace.workspace_id
            if workspace is not None:
                workspace_id = operation.workspace_id
                assert workspace_id is not None
                await workspace.close(interrupted=False)
                claims = workspace.clear_claims_for_removal()
                key = os.path.normcase(str(operation.path.resolve(strict=False)))
                self._workspace_keys.pop(key, None)
                self._workspaces.pop(workspace_id, None)
                for affected_id in operation.affected_client_ids:
                    client = self._clients.get(affected_id)
                    if client is None:
                        continue
                    client.claimed = {
                        claim
                        for claim in client.claimed
                        if claim[0] != workspace_id
                    }
                    client.attached_workspaces.discard(workspace_id)
                    if client.kind == "cli":
                        client.reconnect_blocked = True
                    client.blocked_workspace_keys.add(key)
                    if client.current_workspace_id == workspace_id:
                        client.current_workspace_id = None
                        client.current_session_id = None
                for claim in claims:
                    await self.emit(
                        "session.released",
                        workspace_id=workspace_id,
                        session_id=claim.session_id,
                        run_id=None,
                        payload={},
                        target_client_ids=operation.affected_client_ids,
                    )
                await self.emit(
                    "project.removed",
                    workspace_id=workspace_id,
                    session_id=None,
                    run_id=None,
                    payload={
                        "project_id": operation.project_id,
                        "operation_id": operation.operation_id,
                    },
                    target_client_ids=operation.notification_client_ids,
                )
            self.projects.remove(operation.project_id)
            operation.status = "completed"
            try:
                await self.emit(
                    "project.removal.completed",
                    workspace_id=operation.workspace_id,
                    session_id=None,
                    run_id=None,
                    payload={
                        "project_id": operation.project_id,
                        "operation_id": operation.operation_id,
                        "status": "completed",
                    },
                    target_client_ids=operation.notification_client_ids,
                )
            except Exception:
                # Removal is complete even if a disconnected client misses the event.
                pass
        except asyncio.CancelledError:
            raise
        except Exception:
            operation.status = "failed"
            operation.error = _PROJECT_REMOVAL_FAILURE_MESSAGE
            try:
                self.projects.record_removal_failure(
                    operation.project_id,
                    operation.error,
                )
            except Exception:
                pass
            key = os.path.normcase(str(operation.path.resolve(strict=False)))
            for affected_id in operation.affected_client_ids:
                client = self._clients.get(affected_id)
                if client is not None:
                    if client.kind == "cli":
                        client.reconnect_blocked = True
                    client.blocked_workspace_keys.add(key)
            try:
                await self.emit(
                    "project.removal.failed",
                    workspace_id=operation.workspace_id,
                    session_id=None,
                    run_id=None,
                    payload={
                        "project_id": operation.project_id,
                        "operation_id": operation.operation_id,
                        "status": "failed",
                        "message": operation.error,
                    },
                    target_client_ids=operation.notification_client_ids,
                )
            except Exception:
                # The persisted failure remains retryable if notification delivery fails.
                pass

    def workspace(self, workspace_id: str) -> WorkspaceRecord:
        try:
            workspace = self._workspaces[workspace_id]
        except KeyError as error:
            raise service_error("not_found", "Workspace was not found.", status=404) from error
        key = os.path.normcase(str(workspace.workspace_path.resolve(strict=False)))
        record = next(
            (
                item
                for item in self.projects.list()
                if os.path.normcase(str(item.path.resolve(strict=False))) == key
            ),
            None,
        )
        if record is not None and record.schedule_state in {"removing", "failed"}:
            raise service_error(
                "admission_closed",
                "Project work is being removed and cannot be admitted.",
                retryable=True,
            )
        return workspace

    async def create_session(self, client_id: str, workspace_id: str) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        creation_scope: Literal["chat", "project"] = (
            "project"
            if client.kind == "cli"
            and any(
                os.path.normcase(str(record.path))
                == os.path.normcase(str(workspace.workspace_path))
                for record in self.projects.list()
            )
            else "chat"
        )
        session_id = await workspace.create_draft(
            client_id, reuse_startup_session=client.kind != "web", creation_scope=creation_scope
        )
        return {"workspace_id": workspace_id, "session_id": session_id}

    async def claim(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        previous_workspace_id = client.current_workspace_id
        previous_session_id = client.current_session_id
        already_claimed = (workspace_id, session_id) in client.claimed
        draft_owner = workspace._draft_clients.get(session_id)
        prior_state = workspace._loops.get(session_id)
        prior_owner = None if prior_state is None else prior_state.owner_client_id
        claim = await workspace.claim(client_id, session_id)
        try:
            if not already_claimed:
                await self.emit(
                    "session.claimed",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    run_id=None,
                    payload={"occupied": True},
                    target_client_ids=self.workspace_audience(workspace_id),
                )
            snapshot = workspace.session_snapshot(session_id)
        except BaseException as error:
            if not already_claimed:
                workspace._claims.pop(session_id, None)
                workspace._loops[session_id].owner_client_id = prior_owner
                if draft_owner is not None:
                    workspace._draft_clients[session_id] = draft_owner
                await self.emit(
                    "session.released",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    run_id=None,
                    payload={},
                    target_client_ids=self.workspace_audience(workspace_id),
                )
            if isinstance(error, (OSError, ValueError)):
                raise service_error(
                    "persistence_error",
                    "Conversation snapshot could not be read safely.",
                    status=500,
                ) from error
            raise
        client.claimed.add((workspace_id, session_id))
        client.current_workspace_id = workspace_id
        client.current_session_id = session_id
        client.current_project_id = None
        if (
            previous_workspace_id is not None
            and previous_session_id is not None
            and (previous_workspace_id, previous_session_id) != (workspace_id, session_id)
        ):
            previous_workspace = self._workspaces.get(previous_workspace_id)
            if previous_workspace is not None:
                await previous_workspace._release_if_idle(client_id, previous_session_id)
        return {
            "claim": {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "claim_version": claim.version,
                "reconnect_credential": claim.credential,
            },
            "snapshot": snapshot,
        }

    def list_subagents(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        workspace._require_subagent_session(session_id)
        repository = workspace.subagent_repository(session_id)
        coordinator = workspace._subagent_coordinator(session_id)
        try:
            page = (
                coordinator.list(status=status, cursor=cursor, limit=limit)
                if coordinator is not None
                else repository.list(status=status, cursor=cursor, limit=limit)
            )
            items: list[dict[str, object]] = []
            for item in page.items:
                record = repository.get(item.agent_id)
                result_preview = None if record is None else record.result
                if result_preview is not None and len(result_preview) > 240:
                    result_preview = result_preview[:237] + "..."
                items.append(
                    {
                        "agent_id": item.agent_id,
                        "title": item.title,
                        "status": item.status.value,
                        "created_at": item.created_at.isoformat(),
                        "finished_at": (
                            None if item.finished_at is None else item.finished_at.isoformat()
                        ),
                        "result_preview": result_preview,
                        "error": (
                            None
                            if record is None or record.error is None
                            else record.error.to_dict()
                        ),
                        "usage": {} if record is None else dict(record.usage or {}),
                    }
                )
            return {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "items": items,
                "next_cursor": page.next_cursor,
            }
        except SubAgentRequestError as error:
            raise service_error("validation_error", str(error), status=422) from error
        except SubAgentStoreError as error:
            raise service_error(
                "persistence_error",
                "SubAgent records could not be read safely.",
                status=500,
                retryable=True,
            ) from error

    def get_subagent(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        agent_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        workspace._require_subagent_session(session_id)
        repository = workspace.subagent_repository(session_id)
        coordinator = workspace._subagent_coordinator(session_id)
        try:
            record = (
                coordinator.get(agent_id)
                if coordinator is not None
                else repository.get(agent_id)
            )
        except SubAgentRequestError as error:
            raise service_error("validation_error", str(error), status=422) from error
        except SubAgentStoreError as error:
            raise service_error(
                "persistence_error",
                "SubAgent records could not be read safely.",
                status=500,
                retryable=True,
            ) from error
        if record is None or record.session_id != session_id:
            raise service_error("not_found", "SubAgent was not found.", status=404)
        return self._subagent_detail(workspace_id, session_id, record)

    async def cancel_subagent(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        agent_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        workspace._require_subagent_session(session_id)
        repository = workspace.subagent_repository(session_id)
        coordinator = workspace._subagent_coordinator(session_id)
        try:
            record = repository.get(agent_id)
            if record is None or record.session_id != session_id:
                raise service_error("not_found", "SubAgent was not found.", status=404)
            if record.status in {SubAgentStatus.QUEUED, SubAgentStatus.RUNNING}:
                if coordinator is None:
                    raise service_error(
                        "subagent_unavailable",
                        "Active SubAgent execution is unavailable for cancellation.",
                        status=503,
                        retryable=True,
                    )
                record = await coordinator.cancel_and_wait(agent_id)
                if record is None:
                    raise service_error("not_found", "SubAgent was not found.", status=404)
            return {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "cancelled": record.status is SubAgentStatus.CANCELLED,
                "agent": self._subagent_detail(workspace_id, session_id, record),
            }
        except SubAgentRequestError as error:
            raise service_error("validation_error", str(error), status=422) from error
        except SubAgentStoreError as error:
            raise service_error(
                "persistence_error",
                "SubAgent cancellation could not be persisted safely.",
                status=500,
                retryable=True,
            ) from error

    @staticmethod
    def _subagent_detail(
        workspace_id: str,
        session_id: str,
        record: SubAgentRecord,
    ) -> dict[str, object]:
        return {
            "workspace_id": workspace_id,
            "session_id": session_id,
            "agent_id": record.agent_id,
            "title": record.title,
            "task": record.task,
            "source": record.source.kind.value,
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "started_at": None if record.started_at is None else record.started_at.isoformat(),
            "finished_at": None if record.finished_at is None else record.finished_at.isoformat(),
            "revision": record.revision,
            "conversation": list(record.conversation),
            "result": record.result,
            "error": None if record.error is None else record.error.to_dict(),
            "usage": dict(record.usage or {}),
        }

    async def list_sessions(self, client_id: str, workspace_id: str) -> list[dict[str, object]]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        return await workspace.list_sessions(client_id)

    async def list_sessions_page(
        self,
        client_id: str,
        workspace_id: str,
        *,
        title: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        creation_scope: str | None = None
        if client.kind == "web" and self.conversation_workspaces.contains(workspace.workspace_path):
            creation_scope = "chat"
        return await workspace.list_sessions_page(
            client_id,
            title=title,
            cursor=cursor,
            limit=limit,
            creation_scope=creation_scope,
        )

    def list_chat_sessions_page(
        self,
        client_id: str,
        *,
        title: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        if client.kind != "web":
            raise service_error(
                "forbidden", "Chat history is only available to Web clients.", status=403
            )
        if limit is not None and (
            isinstance(limit, bool) or limit < 1 or limit > _MAX_SESSION_PAGE_SIZE
        ):
            raise service_error(
                "validation_error",
                f"limit must be between 1 and {_MAX_SESSION_PAGE_SIZE}.",
                status=422,
            )
        page_limit = 50 if limit is None else limit
        title_filter = "" if title is None else title.strip().casefold()
        cursor_key = None if cursor is None else _decode_chat_session_cursor(cursor, title_filter)
        try:
            directories = self.conversation_workspaces.list()
        except ConversationWorkspaceCatalogError as error:
            raise service_error(
                "persistence_error",
                "Conversation Workspace history could not be read safely.",
                status=500,
            ) from error

        entries: list[
            tuple[
                tuple[datetime, datetime, str, str],
                dict[str, object],
            ]
        ] = []
        unavailable_directories: list[str] = []
        for path in directories:
            state = WorkspaceState(path)
            try:
                sessions_directory = state.existing_sessions_directory()
                if sessions_directory is None:
                    continue
                session_paths = tuple(
                    session_path
                    for session_path in sessions_directory.iterdir()
                    if os.path.normcase(session_path.suffix) == ".jsonl"
                )
            except (OSError, RuntimeError, ValueError):
                unavailable_directories.append(str(path))
                continue
            for session_path in session_paths:
                session_id = session_path.stem
                try:
                    if session_deletion_pending(state, session_id):
                        continue
                    loaded_id, created_at, updated_at, metadata = Session.load_header(
                        state,
                        session_id,
                        partition=SessionStoragePartition.FOREGROUND,
                    )
                except (OSError, UnicodeError, ValueError, RuntimeError):
                    continue
                if metadata.get("creation_scope") != "chat":
                    continue
                session_title = metadata.get("title", "Untitled session")
                if not isinstance(session_title, str):
                    session_title = "Untitled session"
                if title_filter and title_filter not in session_title.casefold():
                    continue
                key = (
                    updated_at,
                    created_at,
                    loaded_id,
                    str(path),
                )
                if cursor_key is not None and key >= cursor_key:
                    continue
                entries.append(
                    (
                        key,
                        {
                            "id": loaded_id,
                            "title": session_title,
                            "created_at": created_at.isoformat(),
                            "updated_at": updated_at.isoformat(),
                            "directory": str(path),
                            "available": path.is_dir(),
                        },
                    )
                )

        entries.sort(key=lambda item: item[0], reverse=True)
        page = entries[:page_limit]
        next_cursor = (
            _encode_chat_session_cursor(page[-1][0], title_filter)
            if len(entries) > len(page) and page
            else None
        )
        return {
            "sessions": [entry for _key, entry in page],
            "next_cursor": next_cursor,
            "unavailable_directories": unavailable_directories,
        }

    async def release_claim(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> None:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        workspace.require_claim(client_id, session_id, claim_version, claim_credential)
        await workspace.release(client_id, session_id)

    async def session_deletion_status(
        self, client_id: str, workspace_id: str, session_id: str
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        async with workspace._lock:
            try:
                state = session_deletion_status(workspace.workspace_state, session_id)
            except ValueError as error:
                raise service_error(
                    "validation_error", "Session ID is invalid.", status=422
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "Session deletion status could not be read safely.",
                    status=500,
                    retryable=True,
                ) from error
            if state == "deleted" and session_id in workspace._loops:
                state = "present"
            return {
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "state": state,
            }

    async def claim_session_deletion(
        self, client_id: str, workspace_id: str, session_id: str
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        async with workspace._lock:
            if workspace._closed:
                raise service_error("admission_closed", "Workspace admission is closed.")
            try:
                pending = session_deletion_pending(workspace.workspace_state, session_id)
            except ValueError as error:
                raise service_error(
                    "validation_error", "Session ID is invalid.", status=422
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "Session deletion state could not be read safely.",
                    status=500,
                    retryable=True,
                ) from error
            if not pending:
                raise service_error("not_found", "Session deletion is not pending.", status=404)
            active = workspace._claims.get(session_id)
            cleanup = workspace._deletion_claims.get(session_id)
            if active is not None:
                active = workspace.require_claim(client_id, session_id, active.version)
                version, credential = active.version, active.credential
            else:
                if cleanup is not None and cleanup.client_id != client_id:
                    raise service_error(
                        "session_claimed", "Session deletion is claimed by another client."
                    )
                if cleanup is None:
                    version = workspace._claim_versions.get(session_id, 0) + 1
                    workspace._claim_versions[session_id] = version
                    cleanup = SessionDeletionClaim(session_id, client_id, version, str(uuid4()))
                    workspace._deletion_claims[session_id] = cleanup
                version, credential = cleanup.version, cleanup.credential
            return {
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim": {
                    "workspace_id": workspace.workspace_id,
                    "session_id": session_id,
                    "claim_version": version,
                    "reconnect_credential": credential,
                },
            }

    async def rename_session(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        title: str,
        expected_metadata_version: int,
        request_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        return await workspace.rename_session(
            client_id,
            session_id,
            claim_version,
            claim_credential,
            title,
            expected_metadata_version,
            request_id,
        )

    async def delete_session(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        client.attached_workspaces.add(workspace_id)
        return await workspace.delete_session(
            client_id,
            session_id,
            claim_version,
            claim_credential,
            request_id,
        )

    async def inspect_restore(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
        anchor_id: int,
    ) -> dict[str, object]:
        return await self._restore_management(
            client_id,
            workspace_id,
            session_id,
            claim_version,
            claim_credential,
            "restore/inspect",
            request_id,
            {"anchor_id": anchor_id},
        )

    async def commit_restore(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
        anchor_id: int,
        mode: str,
    ) -> dict[str, object]:
        result = await self._restore_management(
            client_id,
            workspace_id,
            session_id,
            claim_version,
            claim_credential,
            "restore/execute",
            request_id,
            {"plan": {"anchor_id": anchor_id}, "mode": mode},
        )
        if not isinstance(result.get("restore_result"), dict):
            raise service_error("restore_failed", "Session Restore returned no result.", status=500)
        return result

    async def get_restore_result(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        return await self._restore_management(
            client_id,
            workspace_id,
            session_id,
            claim_version,
            claim_credential,
            "restore/result",
            request_id,
            {},
        )

    async def cancel_restore(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        return await self._restore_management(
            client_id,
            workspace_id,
            session_id,
            claim_version,
            claim_credential,
            "restore/cancel",
            request_id,
            {},
        )

    async def acknowledge_restore_failure(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        return await self._restore_management(
            client_id,
            workspace_id,
            session_id,
            claim_version,
            claim_credential,
            "restore/acknowledge",
            request_id,
            {},
        )

    async def _restore_management(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        action: str,
        request_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        return await self.handle_management(
            client_id,
            workspace_id,
            session_id,
            action,
            {"request_id": request_id, **payload},
            claim_version=claim_version,
            claim_credential=claim_credential,
        )

    async def _run_named_session_management(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        action: str,
        request_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> dict[str, object]:
        result = await self.handle_management(
            client_id,
            workspace_id,
            session_id,
            action,
            {"request_id": request_id},
            claim_version=claim_version,
            claim_credential=claim_credential,
        )
        failure = result.get("management_error")
        if isinstance(failure, dict):
            code = failure.get("code")
            message = failure.get("message")
            if isinstance(code, str) and isinstance(message, str):
                field_errors = failure.get("field_errors")
                raise service_error(
                    code,
                    message,
                    status=409 if code == "model_invalid_request" else 500,
                    retryable=failure.get("retryable") is True,
                    field_errors=field_errors if isinstance(field_errors, dict) else None,
                )
        return result

    async def project_memory_operation(
        self, client_id: str, project_id: str, request_id: str, action: str
    ) -> dict[str, object]:
        """Operate on Project-owned Memory without creating a Conversation Session."""
        client = self._require_client(client_id)
        if action not in {"read", "dream"} or not request_id:
            raise service_error("validation_error", "Memory operation is invalid.", status=422)
        fingerprint = f"project-memory:{project_id}:{action}"
        async with client.management_lock:
            async with self._project_lifecycle_lock:
                _, workspace = await self._project_workspace_owned(client_id, project_id)
                workspace._require_admitted()
            previous = client.management_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error("request_reused", "request_id was already used.")
                return deepcopy(previous[1])
            if action == "read":
                try:
                    content = await workspace.memory_manager.read_long_term()
                except (OSError, UnicodeError, ValueError) as error:
                    raise service_error(
                        "persistence_error", "Long-term Memory could not be read.", status=500
                    ) from error
                result: dict[str, object] = {
                    "request_id": request_id, "workspace_id": workspace.workspace_id,
                    "content": content,
                }
            else:
                dream_result = await workspace.dream.run()
                result = {
                    "request_id": request_id, "workspace_id": workspace.workspace_id,
                    "result": _safe_wire_value(dream_result),
                }
            client.management_results[request_id] = (fingerprint, result)
            return deepcopy(result)

    async def get_memory_view(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        request_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> MemoryViewDTO:
        result = await self._run_named_session_management(
            client_id,
            workspace_id,
            session_id,
            "memory",
            request_id,
            claim_version,
            claim_credential,
        )
        content = result.get("memory_content")
        if not isinstance(content, str):
            raise service_error("service_protocol_error", "Memory result is invalid.", status=500)
        return {"request_id": request_id, "workspace_id": workspace_id, "content": content}

    async def run_dream(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        request_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> DreamRunDTO:
        result = await self._run_named_session_management(
            client_id,
            workspace_id,
            session_id,
            "dream",
            request_id,
            claim_version,
            claim_credential,
        )
        dream_result = result.get("dream_result")
        if not isinstance(dream_result, dict):
            raise service_error("service_protocol_error", "Dream result is invalid.", status=500)
        return {"request_id": request_id, "workspace_id": workspace_id, "result": dream_result}

    async def get_runtime_status(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        request_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> RuntimeStatusDTO:
        result = await self._run_named_session_management(
            client_id,
            workspace_id,
            session_id,
            "status",
            request_id,
            claim_version,
            claim_credential,
        )
        status = result.get("status_view")
        if not isinstance(status, dict):
            raise service_error("service_protocol_error", "Runtime status is invalid.", status=500)
        return {"request_id": request_id, "workspace_id": workspace_id, "status": status}

    async def reload_skill_catalog(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        request_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> SkillReloadDTO:
        result = await self._run_named_session_management(
            client_id,
            workspace_id,
            session_id,
            "skills/reload",
            request_id,
            claim_version,
            claim_credential,
        )
        metadata = result.get("skill_metadata")
        if not isinstance(metadata, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("description"), str)
            or not isinstance(item.get("path"), str)
            for item in metadata
        ):
            raise service_error("service_protocol_error", "Skill metadata is invalid.", status=500)
        skills: list[SkillMetadataDTO] = [
            {"name": item["name"], "description": item["description"], "path": item["path"]}
            for item in metadata
        ]
        return {"request_id": request_id, "workspace_id": workspace_id, "skills": skills}

    async def handle_management(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        action: str,
        payload: Mapping[str, object],
        *,
        claim_version: int | None = None,
        claim_credential: str | None = None,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise service_error("validation_error", "request_id is required.", status=422)
        fingerprint_payload = {key: value for key, value in payload.items() if key != "request_id"}
        try:
            fingerprint = json.dumps(
                {
                    "workspace_id": workspace_id,
                    "session_id": session_id,
                    "action": action,
                    "payload": fingerprint_payload,
                    "claim_version": claim_version,
                    "claim_credential": claim_credential,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise service_error(
                "validation_error", "Management payload is invalid.", status=422
            ) from error
        async with client.management_lock:
            self._require_client(client_id)
            previous = client.management_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error(
                        "request_reused",
                        "request_id was already used for a different management request.",
                        status=409,
                    )
                if (
                    action in {"status", "permission", "effort", "memory", "dream", "skills/reload"}
                    or action.startswith("restore/")
                    or (action == "dispatch" and payload.get("command") == "/restore")
                ):
                    workspace = self.workspace(workspace_id)
                    workspace._ensure_session_available(session_id)
                    replay_version = previous[1].get("claim_version", claim_version)
                    replay_credential = previous[1].get("claim_credential", claim_credential)
                    if not isinstance(replay_version, int) or not isinstance(
                        replay_credential, str
                    ):
                        raise service_error("stale_claim", "Conversation Session Claim is missing.")
                    workspace.require_claim(
                        client_id, session_id, replay_version, replay_credential
                    )
                return previous[1]
            result = await self._handle_management_once(
                client_id,
                workspace_id,
                session_id,
                action,
                payload,
                claim_version=claim_version,
                claim_credential=claim_credential,
            )
            client.management_results[request_id] = (fingerprint, result)
            return result

    async def _handle_management_once(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        action: str,
        payload: Mapping[str, object],
        *,
        claim_version: int | None,
        claim_credential: str | None,
    ) -> dict[str, object]:
        self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        restore_actions = {
            "restore/inspect",
            "restore/execute",
            "restore/result",
            "restore/cancel",
            "restore/acknowledge",
        }
        requires_restore_claim = action in restore_actions or (
            action == "dispatch" and payload.get("command") == "/restore"
        )
        requires_typed_management_claim = action in {
            "status",
            "permission",
            "effort",
            "memory",
            "dream",
            "skills/reload",
        }
        requires_claim = requires_restore_claim or requires_typed_management_claim
        if requires_claim and (claim_version is None or claim_credential is None):
            raise service_error(
                "stale_claim",
                "Conversation Session Claim is missing or stale.",
                retryable=True,
            )
        resume_context: dict[str, object] = {}
        dispatcher = workspace.management_dispatcher(
            client_id,
            session_id,
            claim_version if requires_claim else None,
            claim_credential if requires_claim else None,
            resume_context=resume_context,
        )
        if action == "dispatch":
            command = payload.get("command")
            if not isinstance(command, str):
                raise service_error(
                    "validation_error", "Management command is required.", status=422
                )
            result = await dispatcher.dispatch(command)
        elif action == "status":
            result = await dispatcher.status()
        elif action == "effort":
            effort = payload.get("effort")
            if not isinstance(effort, str):
                raise service_error("validation_error", "Reasoning effort is required.", status=422)
            result = await dispatcher.update_reasoning_effort(cast(Any, effort))
        elif action == "permission":
            permission = payload.get("permission_level")
            if not isinstance(permission, str):
                raise service_error("validation_error", "Permission level is required.", status=422)
            result = await dispatcher.update_permission_level(cast(Any, permission))
        elif action == "memory":
            result = await dispatcher.memory_view()
        elif action == "dream":
            result = await dispatcher.dream()
        elif action == "skills/reload":
            result = await dispatcher.reload_skill()
        elif action == "resume":
            resume_id = payload.get("session_id")
            if not isinstance(resume_id, str):
                raise service_error("validation_error", "Session ID is required.", status=422)
            result = await dispatcher.resume(resume_id, force=payload.get("force") is True)
        elif action == "restore/inspect":
            anchor_id = payload.get("anchor_id")
            if isinstance(anchor_id, bool) or not isinstance(anchor_id, int):
                raise service_error("validation_error", "Restore anchor ID is invalid.", status=422)
            result = await dispatcher.restore_inspect(anchor_id)
        elif action == "restore/execute":
            wire_plan = payload.get("plan")
            mode = payload.get("mode")
            if not isinstance(wire_plan, dict) or isinstance(wire_plan.get("anchor_id"), bool):
                raise service_error("validation_error", "Restore plan is required.", status=422)
            anchor_id = wire_plan.get("anchor_id")
            if not isinstance(anchor_id, int) or anchor_id < 1:
                raise service_error(
                    "validation_error", "Restore plan anchor ID is invalid.", status=422
                )
            if not isinstance(mode, str) or not mode:
                raise service_error("validation_error", "Restore mode is required.", status=422)
            try:
                result = await dispatcher.restore_commit(anchor_id, mode)
            except Exception as error:
                raise service_error(
                    "restore_failed",
                    "Session Restore could not be completed safely.",
                    status=500,
                ) from error
        elif action == "restore/result":
            result = await dispatcher.restore_result()
        elif action == "restore/cancel":
            result = await dispatcher.restore_cancel()
        elif action == "restore/acknowledge":
            result = await dispatcher.restore_acknowledge_failure()
        else:
            raise service_error("validation_error", "Unsupported management action.", status=422)
        encoded = _encode_management_result(result)
        if encoded.get("resumed_session_id") is not None:
            encoded.update(resume_context)
        if action == "restore/execute" and getattr(result, "restore_result", None) is not None:
            claim = workspace._claims.get(session_id)
            if claim is not None and claim.client_id == client_id:
                encoded["claim_version"] = claim.version
                encoded["claim_credential"] = claim.credential
                encoded["claim"] = {
                    "workspace_id": workspace_id,
                    "session_id": session_id,
                    "claim_version": claim.version,
                    "reconnect_credential": claim.credential,
                }
                encoded["snapshot"] = workspace.session_snapshot(session_id)
        return encoded

    async def configure_conversation_model(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        workspace.require_claim(client_id, session_id, claim_version)
        provider_id = payload.get("provider_id")
        model = payload.get("model")
        reasoning_effort = payload.get("reasoning_effort")
        expected_version = payload.get("expected_model_configuration_version")
        if (
            not isinstance(provider_id, str)
            or not provider_id.strip()
            or not isinstance(model, str)
            or not model.strip()
            or not isinstance(reasoning_effort, str)
            or reasoning_effort not in REASONING_EFFORT_LEVELS
        ):
            raise service_error(
                "validation_error", "Session Model Configuration is invalid.", status=422
            )
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise service_error(
                "validation_error",
                "expected_model_configuration_version is invalid.",
                status=422,
                field_errors={"expected_model_configuration_version": "must be nonnegative"},
            )
        configuration = self.configuration
        if configuration is None:
            raise service_error(
                "model_unavailable",
                "The selected model is not available for this Agent Service.",
                status=422,
                field_errors={"model": "choose an available model with a valid context window"},
            )
        try:
            selection = SessionModelConfiguration(
                provider_id,
                model,
                reasoning_effort,
            )
            configuration.resolve_session_model_route(
                selection.provider_id,
                selection.model,
                selection.reasoning_effort,
            )
        except (AttributeError, ConfigError, TypeError, ValueError) as error:
            raise service_error(
                "model_unavailable",
                "The selected model is not available for this Agent Service.",
                status=422,
                field_errors={"model": "choose an available model with a valid context window"},
            ) from error
        claim = workspace.require_claim(client_id, session_id, claim_version)
        session = claim.loop.session
        try:
            await session.wait_for_pending_persist()
            workspace.require_claim(client_id, session_id, claim_version)
            version = session.configure_model_durably(
                selection,
                expected_version=expected_version,
            )
        except ValueError as error:
            if "version is stale" in str(error):
                raise service_error(
                    "model_configuration_conflict",
                    "Session Model Configuration changed; reload it before retrying.",
                    status=409,
                ) from error
            raise service_error(
                "validation_error",
                "Session Model Configuration is invalid.",
                status=422,
            ) from error
        except (OSError, RuntimeError) as error:
            raise service_error(
                "persistence_error",
                "Session Model Configuration could not be saved.",
                status=500,
            ) from error
        result: dict[str, object] = {
            "model_configuration": selection.to_dict(),
            "model_configuration_version": version,
        }
        if version != expected_version:
            await self.emit(
                "session.model_configuration",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=None,
                payload={
                    **selection.to_dict(),
                    "model_configuration_version": version,
                },
            )
        return result

    async def execute_management_command(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        command: str,
        request_id: str,
    ) -> dict[str, object]:
        claim = self.workspace(workspace_id).require_claim(client_id, session_id, claim_version)
        return await self.handle_management(
            client_id,
            workspace_id,
            session_id,
            "dispatch",
            {"request_id": request_id, "command": command},
            claim_version=claim_version,
            claim_credential=claim.credential,
        )

    async def recall_queued_inputs(
        self, client_id: str, workspace_id: str, session_id: str, claim_version: int
    ) -> dict[str, object]:
        self._require_client(client_id)
        return await self.workspace(workspace_id).recall_queued_inputs(
            client_id, session_id, claim_version
        )

    async def cancel_run(
        self, client_id: str, workspace_id: str, session_id: str, claim_version: int, run_id: str
    ) -> None:
        self._require_client(client_id)
        await self.workspace(workspace_id).cancel(client_id, session_id, claim_version, run_id)

    def decide_confirmation(
        self, client_id: str, token: str, decision: ConfirmationDecision
    ) -> dict[str, object]:
        self._require_client(client_id)
        if not self._presenter.decide(client_id, token, decision):
            raise service_error("confirmation_resolved", "Confirmation is already resolved.")
        return {"decided": True}

    async def submit_user_input(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        claim_version: int,
        text: str,
        request_id: str,
    ) -> dict[str, object]:
        """Classify a raw submission before admitting ordinary Session input."""
        from aide.management.commands import MANAGEMENT_COMMANDS

        workspace = self.workspace(workspace_id)
        workspace.require_claim(client_id, session_id, claim_version)
        if text in {command.token for command in MANAGEMENT_COMMANDS}:
            response = await self.execute_management_command(
                client_id, workspace_id, session_id, claim_version, text, request_id
            )
            if response.get("handled") is not True:
                raise service_error(
                    "service_protocol_error", "Management result is invalid.", status=500
                )
            return {"kind": "management", "management_result": response}

        self._capture_configuration()
        run_id = str(uuid4())
        await workspace.input(
            client_id,
            session_id,
            claim_version,
            text,
            run_id,
            request_id,
        )
        await self.emit(
            "input.accepted",
            workspace_id=workspace_id,
            session_id=session_id,
            run_id=run_id,
            payload={"text": text, "request_id": request_id},
            target_client_ids=(client_id,),
        )
        return {
            "kind": "conversation_input",
            "run_id": run_id,
            "live_state": workspace.session_snapshot(session_id)["live_state"],
        }

    async def handle_command(
        self, client_id: str, command: Mapping[str, object]
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        request_id = command.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise service_error("validation_error", "request_id is required.", status=422)
        previous = client.results.get(request_id)
        if previous is not None:
            return previous
        inflight = client.inflight.get(request_id)
        if inflight is not None:
            return await asyncio.shield(inflight)
        task = asyncio.create_task(self._handle_command_once(client_id, command))
        client.inflight[request_id] = task

        def forget(done: asyncio.Task[dict[str, object]]) -> None:
            if client.inflight.get(request_id) is done:
                client.inflight.pop(request_id, None)
            _consume_task_result(done)

        task.add_done_callback(forget)
        return await asyncio.shield(task)

    async def _handle_command_once(
        self, client_id: str, command: Mapping[str, object]
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        request_id = command.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise service_error("validation_error", "request_id is required.", status=422)
        command_type = command.get("type")
        workspace_id = command.get("workspace_id")
        session_id = command.get("session_id")
        claim_version = command.get("claim_version")
        payload = command.get("payload")
        if not isinstance(payload, dict):
            raise service_error("validation_error", "payload must be an object.", status=422)
        if command_type == "claim":
            if not isinstance(workspace_id, str) or not isinstance(session_id, str):
                raise service_error(
                    "validation_error", "claim requires Workspace and Session IDs.", status=422
                )
            result = await self.claim(client_id, workspace_id, session_id)
        elif command_type == "release":
            self._validate_claim_fields(client_id, workspace_id, session_id, claim_version)
            assert (
                isinstance(workspace_id, str)
                and isinstance(session_id, str)
                and isinstance(claim_version, int)
            )
            workspace = self.workspace(workspace_id)
            workspace.require_claim(client_id, session_id, claim_version)
            await workspace.release(client_id, session_id)
            result = {"released": True}
        elif command_type == "input":
            self._validate_claim_fields(client_id, workspace_id, session_id, claim_version)
            text = payload.get("text")
            if not isinstance(text, str):
                raise service_error("validation_error", "input text is required.", status=422)
            assert (
                isinstance(workspace_id, str)
                and isinstance(session_id, str)
                and isinstance(claim_version, int)
            )
            result = await self.submit_user_input(
                client_id,
                workspace_id,
                session_id,
                claim_version,
                text,
                request_id,
            )
        elif command_type == "recall_queued_inputs":
            self._validate_claim_fields(client_id, workspace_id, session_id, claim_version)
            assert (
                isinstance(workspace_id, str)
                and isinstance(session_id, str)
                and isinstance(claim_version, int)
            )
            result = await self.recall_queued_inputs(
                client_id,
                workspace_id,
                session_id,
                claim_version,
            )
        elif command_type == "session_model_configure":
            self._validate_claim_fields(client_id, workspace_id, session_id, claim_version)
            assert (
                isinstance(workspace_id, str)
                and isinstance(session_id, str)
                and isinstance(claim_version, int)
            )
            workspace = self.workspace(workspace_id)
            workspace.require_claim(client_id, session_id, claim_version)
            result = await self.configure_conversation_model(
                client_id, workspace_id, session_id, claim_version, payload
            )
        elif command_type == "cancel":
            self._validate_claim_fields(client_id, workspace_id, session_id, claim_version)
            cancel_run_id = payload.get("run_id")
            if not isinstance(cancel_run_id, str):
                raise service_error("validation_error", "cancel run_id is required.", status=422)
            assert (
                isinstance(workspace_id, str)
                and isinstance(session_id, str)
                and isinstance(claim_version, int)
            )
            await self.cancel_run(client_id, workspace_id, session_id, claim_version, cancel_run_id)
            result = {"cancelled": True}
        elif command_type == "confirmation_decide":
            token = payload.get("token")
            decision = payload.get("decision")
            if not isinstance(token, str) or decision not in {"approved", "declined"}:
                raise service_error(
                    "validation_error", "confirmation decision is invalid.", status=422
                )
            result = self.decide_confirmation(
                client_id, token, cast(ConfirmationDecision, decision)
            )
        elif command_type == "subscribe":
            last_seq = payload.get("last_seq")
            await self._replay(client, last_seq, payload.get("stream_id"))
            result = {"subscribed": True, "stream_id": client.stream_id, "seq": client.sequence}
        else:
            raise service_error("validation_error", "Unsupported client command.", status=422)
        ack = {"request_id": request_id, "accepted": True, "result": result}
        client.results[request_id] = ack
        return ack

    async def emit(
        self,
        event_type: str,
        *,
        workspace_id: str | None,
        session_id: str | None,
        run_id: str | None,
        payload: dict[str, object],
        target_client_ids: tuple[str, ...] | None = None,
    ) -> None:
        project_id = None
        if workspace_id is not None:
            workspace = self._workspaces.get(workspace_id)
            if workspace is not None:
                workspace_key = os.path.normcase(str(workspace.workspace_path))
                project_id = next(
                    (
                        record.project_id
                        for record in self.projects.list()
                        if os.path.normcase(str(record.path.resolve(strict=False))) == workspace_key
                    ),
                    None,
                )
        if event_type in {
            "project.removal.started",
            "project.removal.failed",
            "project.removal.completed",
            "project.removed",
        } and isinstance(payload.get("project_id"), str):
            project_id = cast(str, payload["project_id"])
        targets = (
            tuple(self._clients.values())
            if target_client_ids is None
            else tuple(
                self._clients[client_id]
                for client_id in target_client_ids
                if client_id in self._clients
            )
        )
        for client in targets:
            if workspace_id is not None and session_id is not None:
                cancellation_ack = event_type == "run.cancelled" and target_client_ids is not None
                if (
                    (workspace_id, session_id) not in client.claimed
                    and event_type.startswith("run.")
                    and not cancellation_ack
                ):
                    continue
            async with client.delivery_lock:
                if workspace_id is not None and session_id is not None and run_id is not None:
                    workspace = self._workspaces.get(workspace_id)
                    state = workspace._loops.get(session_id) if workspace is not None else None
                    if state is not None and state.owner_client_id == client.client_id:
                        self._update_live_run(state, event_type, run_id, payload)
                client.sequence += 1
                event = {
                    "protocol_version": self.protocol_version,
                    "service_instance_id": self.service_instance_id,
                    "stream_id": client.stream_id,
                    "seq": client.sequence,
                    "type": event_type,
                    "workspace_id": workspace_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "payload": payload,
                }
                client.events.append(event)
                if client.connected and client.subscribed and client.sink is not None:
                    await self._send_event(client, event)

    @staticmethod
    def _update_live_run(
        state: _LoopState, event_type: str, run_id: str, payload: dict[str, object]
    ) -> None:
        if event_type == "input.recalled":
            state.live_runs.pop(run_id, None)
            return
        if event_type == "run.started":
            run = state.live_runs.get(run_id)
            if run is not None:
                run["status"] = "running"
                run["cancellable"] = True
            return
        if event_type in {"run.completed", "run.failed"}:
            if state.live_runs.pop(run_id, None) is not None:
                if state.completed_user_count is not None:
                    state.completed_user_count += 1
                if not state.live_runs:
                    state.completed_user_count = None
            return
        run = state.live_runs.get(run_id)
        if run is None or event_type != "run.output":
            return
        message = cast(dict[str, Any], payload["message"])
        metadata = message["metadata"]
        run["status"] = "running"
        if message["type"] == "model_response" and metadata.get("_stream_delta") is True:
            run["assistant_content"] += message["content"]
            run["response_segments"][-1] += message["content"]
        elif message["type"] == "tool_call":
            tool_id = metadata.get("tool_call_id")
            if not isinstance(tool_id, str) or not tool_id:
                return
            tools = run["tools"]
            tool = next((item for item in tools if item["tool_call_id"] == tool_id), None)
            status = {"success": "completed", "error": "failed", "refused": "rejected",
                      "cancelled": "canceled", "canceled": "canceled"}.get(metadata.get("status"))
            if tool is None:
                run["response_segments"].append("")
                tools.append({"tool_call_id": tool_id, "name": message["content"],
                              "arguments": metadata.get("arguments", ""),
                              "status": status or "running"})
                tool = tools[-1]
            elif status is not None:
                tool["status"] = status
            if isinstance(metadata.get("result"), str):
                tool["result"] = metadata["result"]

    def confirmation_source(
        self, owner: ConfirmationOwner
    ) -> tuple[str | None, str | None, str | None]:
        for workspace in self._workspaces.values():
            result = workspace.confirmation_source(owner)
            if result[0] is not None:
                return result
        return None, None, None

    async def stop(self) -> None:
        if self._stop_task is not None:
            task = self._stop_task
            if not task.done():
                await task
                return
            try:
                task.result()
            except BaseException:
                self._stop_task = None
            else:
                return
        task = asyncio.create_task(self._stop_owned())
        self._stop_task = task
        try:
            await task
        except BaseException:
            if self._stop_task is task:
                self._stop_task = None
            raise

    async def _stop_owned(self) -> None:
        if self.state == "stopped":
            return
        self.state = "draining"
        errors: list[Exception] = []
        for workspace in tuple(self._workspaces.values()):
            workspace.close_subagent_admission()
        if self._global_reconnect_task is not None:
            self._global_reconnect_task.cancel()
        await self._reconcile_schedule_admission()
        for operation in tuple(self._project_removals.values()):
            task = operation.task
            if task is not None and not task.done():
                try:
                    await asyncio.shield(task)
                except Exception as error:
                    errors.append(error)
        for client in self._clients.values():
            if client.disconnect_task is not None:
                client.disconnect_task.cancel()
        for workspace in tuple(self._workspaces.values()):
            try:
                await workspace.pause_schedule_admission()
            except Exception as error:
                errors.append(error)
        for workspace in tuple(self._workspaces.values()):
            try:
                await workspace.close()
            except Exception as error:
                errors.append(error)
        if any(workspace._subagent_shutdown_failed for workspace in self._workspaces.values()):
            self._stop_failed = True
            raise service_error(
                "service_stop_failed",
                "The local service could not persist SubAgent shutdown state; retry after storage is repaired.",
                status=500,
                retryable=True,
            ) from ExceptionGroup("Local service SubAgent cleanup failed", errors)
        try:
            await self.confirmation.close()
        except Exception as error:
            errors.append(error)
        try:
            await self._workspace_resources.close()
        except Exception as error:
            errors.append(error)
        try:
            await self._configuration_resources.close()
        except Exception as error:
            errors.append(error)
        self._stop_failed = bool(errors)
        self.state = "stopped"
        self._closed.set()
        if errors:
            raise service_error(
                "service_stop_failed",
                "The local service stopped with a workspace cleanup error.",
                status=500,
            ) from ExceptionGroup("Local service cleanup failed", errors)

    async def wait_closed(self) -> None:
        await self._closed.wait()
        if self._stop_failed:
            raise service_error(
                "service_stop_failed",
                "The local service stopped with a workspace cleanup error.",
                status=500,
            )

    def _require_client(self, client_id: str) -> ClientState:
        client = self._clients.get(client_id)
        if client is None:
            raise service_error("unauthenticated", "Client identity is not recognized.", status=401)
        if client.expired or (
            not client.connected
            and client.reconnect_deadline is not None
            and self._monotonic() >= client.reconnect_deadline
        ):
            raise service_error(
                "stale_client", "This Client reconnect grace period has expired.", retryable=True
            )
        return client

    async def _expire_client_later(self, client_id: str, deadline: float) -> None:
        try:
            await self._wait_until(deadline)
        except asyncio.CancelledError:
            return
        client = self._clients.get(client_id)
        if client is None or client.connected:
            return
        client.expired = True
        self._client_by_reconnect.pop(client.reconnect_credential, None)
        errors: list[Exception] = []
        for workspace in tuple(self._workspaces.values()):
            try:
                await workspace.expire_client(client_id)
                async with self._project_lifecycle_lock:
                    if self._workspaces.get(workspace.workspace_id) is not workspace:
                        continue
                    key = os.path.normcase(str(workspace.workspace_path))
                    registered = any(
                        os.path.normcase(str(record.path.resolve(strict=False))) == key
                        for record in self.projects.list()
                    )
                    users = (
                        self._clients[user_id]
                        for user_id in self._workspace_clients(workspace.workspace_id)
                    )
                    if registered or any(
                        not user.expired
                        and (
                            user.connected
                            or user.reconnect_deadline is None
                            or self._monotonic() < user.reconnect_deadline
                        )
                        for user in users
                    ):
                        continue
                    await workspace.pause_schedule_admission()
                    await workspace.schedule_service.pause_and_drain()
                    await workspace.dream.abort_and_wait()
            except Exception as error:
                errors.append(error)
        if errors:
            self.state = "draining"
            for workspace in self._workspaces.values():
                try:
                    await workspace.pause_schedule_admission()
                except Exception:
                    pass
            await self.emit(
                "service.cleanup_failed",
                workspace_id=None,
                session_id=None,
                run_id=None,
                payload={"reason": "client_expiry"},
                target_client_ids=tuple(
                    candidate.client_id
                    for candidate in self._clients.values()
                    if candidate.connected and candidate.client_id != client_id
                ),
            )
            return
        client.claimed.clear()
        self._clients.pop(client_id, None)
        await self._reconcile_schedule_admission()

    async def _stop_after_grace(self, deadline: float | None = None) -> None:
        try:
            if deadline is None:
                deadline = self._monotonic() + self.reconnect_timeout
            await self._wait_until(deadline)
        except asyncio.CancelledError:
            return
        if not any(client.connected for client in self._clients.values()):
            self._global_reconnect_task = None
            self.state = "draining"
            await self.stop()

    async def _wait_until(self, deadline: float) -> None:
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return
            await self._sleep(remaining)

    async def _replay(
        self, client: ClientState, last_seq: object, last_stream_id: object = None
    ) -> None:
        if last_seq is not None and (
            isinstance(last_seq, bool) or not isinstance(last_seq, int) or last_seq < 0
        ):
            raise service_error("validation_error", "last_seq is invalid.", status=422)
        if last_stream_id is not None and not isinstance(last_stream_id, str):
            raise service_error("validation_error", "stream_id is invalid.", status=422)
        async with client.delivery_lock:
            client.subscribed = True
            if client.resync_required:
                await self._send_snapshot_required(client, reason="slow_consumer")
                if client.connected:
                    client.resync_required = False
                return
            if last_stream_id is not None and last_stream_id != client.stream_id:
                await self._send_snapshot_required(client, reason="stream_changed")
                return
            if last_seq is None:
                await self._send_snapshot_required(client, reason="initial_subscribe")
                return
            if last_seq > client.sequence:
                await self._send_snapshot_required(client, reason="cursor_ahead")
                return
            first_seq = int(cast(int, client.events[0]["seq"])) if client.events else 0
            if not client.events or last_seq < first_seq - 1:
                await self._send_snapshot_required(client, reason="event_cache_exhausted")
                return
            for event in tuple(client.events):
                if int(cast(int, event["seq"])) > last_seq and client.sink is not None:
                    await self._send_event(client, event)
            if client.kind == "web":
                await self._send_snapshot_required(client, reason="reconnected")

    async def _send_snapshot_required(self, client: ClientState, *, reason: str) -> None:
        client.sequence += 1
        snapshots: list[dict[str, object]] = []
        for workspace_id, session_id in tuple(client.claimed):
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                continue
            claim = workspace._claims.get(session_id)
            if claim is None or claim.client_id != client.client_id:
                continue
            try:
                snapshot = workspace.session_snapshot(session_id)
            except ServiceError:
                continue
            snapshots.append(
                {
                    "workspace_id": workspace_id,
                    "claim_version": claim.version,
                    "snapshot": snapshot,
                }
            )
        event = {
            "protocol_version": self.protocol_version,
            "service_instance_id": self.service_instance_id,
            "stream_id": client.stream_id,
            "seq": client.sequence,
            "type": "snapshot.required",
            "workspace_id": None,
            "project_id": None,
            "session_id": None,
            "run_id": None,
            "payload": {
                "reason": reason,
                "stream_id": client.stream_id,
                "snapshot": {"sessions": snapshots,
                             "pending_confirmation": self._presenter.snapshot(client.client_id)},
            },
        }
        client.events.append(event)
        if client.connected and client.subscribed and client.sink is not None:
            await self._send_event(client, event)

    async def _send_event(self, client: ClientState, event: dict[str, object]) -> None:
        sink = client.sink
        if not client.connected or sink is None:
            return
        try:
            await asyncio.wait_for(sink.send_event(event), timeout=1.0)
        except Exception:
            client.resync_required = True
            await self.disconnect_client(client.client_id, sink=sink)

    @staticmethod
    def _validate_claim_fields(
        client_id: str,
        workspace_id: object,
        session_id: object,
        claim_version: object,
    ) -> None:
        del client_id
        if not isinstance(workspace_id, str) or not isinstance(session_id, str):
            raise service_error(
                "validation_error", "Workspace and Session IDs are required.", status=422
            )
        if (
            isinstance(claim_version, bool)
            or not isinstance(claim_version, int)
            or claim_version < 1
        ):
            raise service_error(
                "stale_claim", "Conversation Session Claim is missing or stale.", retryable=True
            )
