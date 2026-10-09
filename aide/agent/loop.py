"""Serial foreground Agent Runner orchestration over the Runtime Message Bus."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, cast
from uuid import UUID

from loguru import logger
from tzlocal import get_localzone_name

from aide.agent.blackboard import Blackboard
from aide.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationAborted,
    ConfirmationEnvelope,
    ConfirmationUnavailable,
    ForegroundConfirmationOwner,
)
from aide.agent.context.budget import ContextBudget, ContextUsageSnapshot, estimate_request_tokens
from aide.agent.context.builder import ContextBuilder
from aide.agent.context.run_context import (
    AgentRunContextController,
    AgentRunContextRequestPreparer,
    CompactionProjection,
    agent_run_attempt_guard,
    latest_main_agent_usage_anchor,
)
from aide.agent.memory.manager import MemoryManager
from aide.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    OutboundMessageType,
)
from aide.agent.permission import (
    PermissionSnapshot,
    RuntimePermissionControl,
    ToolPermissionLevel,
)
from aide.agent.run_errors import CommittableAgentRunError
from aide.agent.runner import (
    AgentRunner,
    AgentRunnerResponseSegmentEnd,
    AgentRunnerResult,
    AgentRunnerToolCallFinished,
    AgentRunnerToolCallStarted,
    _build_assistant_repair_message,
)
from aide.agent.session.backup_store import FileBackupStore
from aide.agent.session.execution_state import SessionRunState, TitleWork
from aide.agent.session.session import (
    Session,
    SessionRestoreBefore,
    SessionStoragePartition,
)
from aide.agent.subagents.context import SubAgentToolContext
from aide.agent.subagents.executor import SubAgentRunnerExecutor
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
)
from aide.agent.subagents.ports import SubAgentRecordRepository, SubAgentSessionCoordinator
from aide.agent.tools.base import BaseTool
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.core.subagents import build_subagent_tools
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.permission import MCPToolIdentity, PermissionContext
from aide.agent.tools.tool_gateway import (
    BuiltInToolCatalog,
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ToolGateway,
    ToolResult,
)
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigError, UserConfiguration
from aide.errors import (
    MODEL_CONTEXT_OVERFLOW_MESSAGE,
    TURN_CANCELLED_MESSAGE,
    ErrorInfo,
)
from aide.logging.session import session_log
from aide.management.service import RuntimeStatusInput
from aide.provider.errors import ModelCallError
from aide.provider.model_router import (
    ModelRouter,
    ModelRouterDelegate,
    ModelRouteStatus,
    RunModelRouter,
)
from aide.provider.models import (
    ModelCompleted,
    ModelRoute,
    ReasoningDelta,
    ReasoningEffort,
    TextDelta,
)
from aide.provider.session_configuration import SessionModelConfiguration
from aide.schedule.model import ScheduleJob
from aide.schedule.service import (
    ScheduleJobExecutionError,
    ScheduleOccurrence,
    ScheduleService,
)
from aide.skills.catalog import LoadedSkill, ManualSkillInvocation, SkillLoader
from aide.utils.async_tasks import await_task_preserving_cancellation
from aide.utils.text import normalize_title, normalize_title_candidate


class ModelContextOverflowError(Exception):
    """The complete Model request exceeds the chat input budget."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class ConfirmationRequestView(Protocol):
    """Stable confirmation data exposed to a foreground control consumer."""

    @property
    def confirmation_id(self) -> UUID: ...

    @property
    def tool_call_id(self) -> str: ...

    @property
    def tool_name(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def details(self) -> dict[str, Any]: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...

    @property
    def mcp_identity(self) -> MCPToolIdentity | None: ...


@dataclass(frozen=True, slots=True)
class ForegroundConversationProjection:
    """Presentation-safe snapshot of the active foreground conversation."""

    session_id: str
    messages: tuple[dict[str, Any], ...]


class TerminalAgentRunExecutorControl(Protocol):
    """Foreground control surface including Terminal history projection."""

    @property
    def has_active_run(self) -> bool: ...

    def foreground_input_admitted(self) -> bool: ...

    async def cancel_active_run(self) -> None: ...

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None: ...

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None: ...

    def project_foreground_conversation(self) -> ForegroundConversationProjection: ...


type ConfirmationCallback = Callable[[ConfirmationRequestView], None]
type RuntimeConfirmationRequester = Callable[
    [ConfirmationEnvelope], Awaitable[ConfirmationDecision]
]


@dataclass(slots=True)
class _AgentRunContext:
    """Run-local budget, projection and guarded Router collaborators."""

    route: Literal["chat", "schedule"]
    current_user: dict[str, Any]
    project_messages: CompactionProjection
    router: RunModelRouter
    controller: AgentRunContextController
    runner: AgentRunner


class AgentRunExecutor:
    """Own the complete serial foreground execution path."""

    def __init__(
        self,
        *,
        workspace_path: Path,
        workspace_state: WorkspaceState,
        agent_home: AgentHome,
        configuration: UserConfiguration,
        bus: MessageBus,
        schedule_service: ScheduleService,
        model_router: ModelRouterDelegate,
        memory_manager: MemoryManager,
        session_id: str | None,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        monotonic_now: Callable[[], float],
        exec_host: ExecHost,
        permission_control: RuntimePermissionControl,
        configured_schedule_level: ToolPermissionLevel | None = None,
        mcp_tools: Sequence[BaseTool] = (),
        mcp_keywords: Mapping[str, Sequence[str]] | None = None,
        skill_loader: SkillLoader,
        session: Session | None = None,
        built_in_catalog: BuiltInToolCatalog | None = None,
        session_run_state: SessionRunState | None = None,
        subagent_model_router: ModelRouter | None = None,
    ) -> None:
        if workspace_state.workspace_path != workspace_path:
            raise ValueError("Agent Loop Workspace State must belong to the Workspace")
        if memory_manager.workspace_state is not workspace_state:
            raise ValueError("Agent Loop Memory Manager must belong to the Workspace State")

        # Build every generation-local collaborator before publishing any Loop field.
        context_builder = ContextBuilder(
            workspace_path,
            schedule_service.context_timezone_name() or get_localzone_name(),
            agent_home=agent_home.path,
            memory_manager=memory_manager,
            skill_loader=skill_loader,
        )
        tool_gateway = ToolGateway(
            catalog=built_in_catalog,
            workspace=workspace_path,
            schedule_service=schedule_service,
            skill_root=skill_loader.root,
            additional_tools=(
                (*tuple(mcp_tools), *build_subagent_tools())
                if subagent_model_router is not None
                else tuple(mcp_tools)
            ),
            exec_host=exec_host,
            tool_context=ToolRunContext(
                workspace=workspace_path,
                schedule_service=schedule_service,
                exec_host=exec_host,
            ),
            permission_context=PermissionContext(
                workspace_root=workspace_path,
                configured_schedule_level=(
                    permission_control.configured()
                    if configured_schedule_level is None
                    else configured_schedule_level
                ),
            ),
        )
        selected_mcp_keywords = {} if mcp_keywords is None else dict(mcp_keywords)
        baseline_gateway = build_agent_run_gateway(
            tool_gateway,
            mcp_keywords=selected_mcp_keywords,
        )
        baseline_tool_schemas = tuple(baseline_gateway.schemas)
        active_session = (
            session
            if session is not None
            else (
                Session.create(workspace_state, now=now, new_uuid=new_uuid)
                if session_id is None
                else Session.load(
                    workspace_state,
                    session_id,
                    partition=SessionStoragePartition.FOREGROUND,
                    now=now,
                )
            )
        )

        self._session_run_state = (
            SessionRunState(new_uuid()) if session_run_state is None else session_run_state
        )
        self._workspace_state = workspace_state
        self._configuration = configuration
        self._session = active_session
        self._skill_loader = skill_loader
        self._schedule_service = schedule_service
        self._context_builder = context_builder
        self._memory_manager = memory_manager
        self._now = now
        self._new_uuid = new_uuid
        self._monotonic_now = monotonic_now
        self._schedule_now = schedule_service.current_time
        self._tool_gateway = tool_gateway
        self._exec_host = exec_host
        self._permission_control = permission_control
        self._baseline_tool_schemas = baseline_tool_schemas
        self._mcp_keywords = selected_mcp_keywords
        self._model_router = model_router
        self._subagent_model_router = subagent_model_router
        self._subagent_coordinator: SubAgentSessionCoordinator | None = None
        self._max_iterations = configuration.runtime.max_iterations
        self._bus = bus
        self._generation_started_at: float | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._schedule_tasks: set[asyncio.Task[None]] = set()
        self._aborted_tasks: set[asyncio.Task[Any]] = set()
        self._abort_task: asyncio.Task[None] | None = None
        self._execution_ready: asyncio.Event | None = None
        self._confirmation_requester: RuntimeConfirmationRequester | None = None
        self._active_foreground_owner: ForegroundConfirmationOwner | None = None
        self._cancel_requested = False
        self._aborted = False
        self._started = False
        self._preflighted = False
        self._preflight_error: Exception | None = None
        self._session_abandoned = False
        self._run_model_configuration: SessionModelConfiguration | None = None
        self._captured_foreground: tuple[
            PermissionSnapshot, SessionModelConfiguration | None,
        ] | None = None

    def capture_run_inputs(
        self, permission: PermissionSnapshot, model: SessionModelConfiguration | None,
    ) -> None:
        """Retain inputs captured before asynchronous resource preparation."""
        self._captured_foreground = (permission, model)

    @property
    def session(self) -> Session:
        return self._session

    @property
    def run_model_configuration(self) -> SessionModelConfiguration | None:
        """Return the explicit combination captured before this foreground Run waits."""
        return self._run_model_configuration

    @property
    def generation_id(self) -> UUID:
        """Return the immutable identity of this Runtime Generation."""
        return self._session_run_state.generation_id

    @property
    def tool_schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(schema) for schema in self._baseline_tool_schemas)

    def _new_run_gateway(
        self,
        *,
        excluded_names: Sequence[str] = (),
        permission_snapshot: PermissionSnapshot | None = None,
        permission_context: PermissionContext | None = None,
        tool_context: ToolRunContext | None = None,
    ) -> ToolGateway:
        return build_agent_run_gateway(
            self._tool_gateway,
            excluded_names=excluded_names,
            mcp_keywords=self._mcp_keywords,
            permission_snapshot=permission_snapshot,
            permission_context=permission_context,
            tool_context=tool_context,
        )

    def bind_subagent_coordinator(
        self,
        coordinator: SubAgentSessionCoordinator,
    ) -> None:
        """Bind the Service-owned pool for this executor's Session."""
        if self._subagent_model_router is None:
            raise RuntimeError("SubAgent tools are not enabled for this Agent Loop")
        if coordinator.session_id != self._session.session_id:
            raise ValueError("SubAgent coordinator belongs to a different Session")
        if self._subagent_coordinator is not None:
            raise RuntimeError("Agent Loop SubAgent coordinator is already bound")
        self._subagent_coordinator = coordinator

    def create_subagent_runner_executor(
        self,
        repository: SubAgentRecordRepository,
        *,
        workspace_id: str,
        confirmation_for: Callable[[SubAgentRecord], ConfirmationRequester | None],
    ) -> SubAgentRunnerExecutor:
        """Build the child executor from this Service Loop's shared runtime resources."""
        model_router = self._subagent_model_router
        if model_router is None:
            raise RuntimeError("SubAgent execution requires the shared Service Model Router")
        coordinator = self._subagent_coordinator
        workspace_state = self._workspace_state

        def child_context(record: SubAgentRecord) -> SubAgentToolContext:
            if coordinator is None:
                raise RuntimeError("SubAgent coordinator is unavailable for this Agent Run")
            return SubAgentToolContext(
                coordinator=coordinator, parent_run_id=record.parent_run_id,
                source=record.source, creator_snapshot=record.creator_snapshot,
            )

        return SubAgentRunnerExecutor(
            workspace_id=workspace_id,
            workspace_state=self._workspace_state,
            repository=repository,
            model_router=model_router,
            tool_gateway=self._tool_gateway,
            compact_ratio=self._configuration.runtime.compact_ratio,
            max_iterations=self._configuration.runtime.max_iterations,
            max_tool_result_chars=self._configuration.runtime.max_tool_result_chars,
            enable_tool_micro_compression=(
                self._configuration.runtime.enable_tool_micro_compression
            ),
            mcp_keywords=self._mcp_keywords,
            confirmation_for=confirmation_for,
            tool_context_for=(self._subagent_tool_context_for if coordinator is None else child_context),
            file_mutation_recorder_for=lambda record: FileBackupStore(
                workspace_state,
                record.session_id,
            ),
            now=self._now,
        )

    def _subagent_tool_context_for(self, record: SubAgentRecord) -> SubAgentToolContext:
        coordinator = self._subagent_coordinator
        if coordinator is None:
            raise RuntimeError("SubAgent coordinator is unavailable for this Agent Run")
        return SubAgentToolContext(
            coordinator=coordinator,
            parent_run_id=record.parent_run_id,
            source=record.source,
            creator_snapshot=record.creator_snapshot,
        )

    def _subagent_run_gateway(
        self,
        context: _AgentRunContext,
        run_gateway: ToolGateway,
        *,
        source: SubAgentSource,
        parent_run_id: str,
        permission_snapshot: PermissionSnapshot,
        permission_context: PermissionContext | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        excluded_names: Sequence[str] = (),
        system_prompt: str,
    ) -> ToolGateway:
        coordinator = self._subagent_coordinator
        if coordinator is None:
            raise RuntimeError("SubAgent coordinator is unavailable for this Agent Run")
        shared_router = self._subagent_model_router
        if shared_router is None:
            raise RuntimeError("SubAgent execution requires the shared Service Model Router")
        try:
            route_status = shared_router.call_route_status("subagent", continuation=None)
        except ConfigError as error:
            if error.error.code != "route_unavailable":
                raise
            snapshot = None
        else:
            snapshot = SubAgentCreatorSnapshot(
                provider_id=route_status.provider_id,
                model=route_status.model,
                reasoning_effort=(
                    self._configuration.resolve_route("subagent").route.reasoning_effort
                ),
                permission_level=permission_snapshot.level,
                shell=self._exec_host.resolved_shell.selector,
                tool_names=tuple(tool.name for tool in run_gateway.catalog),
                system_prompt=system_prompt,
            )
        base_context = self._tool_gateway.tool_context
        if base_context is None:
            raise RuntimeError("SubAgent execution requires a Tool Run context")
        tool_context = replace(
            base_context,
            subagent=SubAgentToolContext(
                coordinator=coordinator,
                parent_run_id=parent_run_id,
                source=source,
                creator_snapshot=snapshot,
            ),
        )
        return self._new_run_gateway(
            excluded_names=(
                (*excluded_names, "spawn_agent") if snapshot is None else excluded_names
            ),
            permission_context=permission_context,
            permission_snapshot=(None if permission_context is not None else permission_snapshot),
            tool_context=tool_context,
        )

    async def wait_for_restore_idle(self) -> None:
        """Drain title work before the strict Session restore write."""
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        await self._session_run_state.wait_for_title_idle(self._session.wait_for_pending_persist)

    def bind_confirmation_requester(self, requester: RuntimeConfirmationRequester) -> None:
        """Bind the Runtime Lifetime requester without taking ownership of its queue."""
        if self._confirmation_requester is not None:
            raise RuntimeError("Agent Loop confirmation requester is already bound")
        if self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if not callable(requester):
            raise TypeError("confirmation requester must be callable")
        self._confirmation_requester = requester

    async def start(self) -> None:
        if self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        self.preflight()
        self._activate_prepared()

    def preflight(self) -> None:
        """Validate this generation synchronously without external side effects."""
        if self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        if self._preflighted:
            return
        if self._preflight_error is not None:
            raise self._preflight_error
        try:
            self._validate_model_context_budget(self._skill_loader.skills)
        except Exception as error:
            self._preflight_error = error
            raise
        self._preflighted = True

    def _validate_model_context_budget(
        self,
        skills: tuple[LoadedSkill, ...],
    ) -> None:
        chat_route = self._configuration.resolve_route("chat").route
        tool_schemas = self.tool_schemas
        with self._context_builder.foreground_projection_scope(skills):
            status_input = _foreground_runtime_status_input(
                context_builder=self._context_builder,
                history=(),
                session_id=self._session.session_id,
                tool_schemas=tool_schemas,
                summary=_action_summary_from_metadata(self._session.metadata),
            )
        budget = ContextBudget(
            context_window=chat_route.context_window,
            max_output=chat_route.max_output,
            compact_ratio=self._configuration.runtime.compact_ratio,
        )
        projected_messages = status_input.projected_messages
        estimated = estimate_request_tokens(projected_messages, status_input.projected_tools)
        if budget.exceeds_available_context(estimated):
            raise ModelContextOverflowError(
                ErrorInfo(
                    "model_context_overflow",
                    MODEL_CONTEXT_OVERFLOW_MESSAGE,
                )
            )

    def _activate_prepared(self) -> None:
        """Sample uptime and atomically publish the preflighted Loop activation."""
        if self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        if not self._preflighted:
            raise RuntimeError("Agent Loop was not preflighted")
        started_at = self._monotonic_now()
        self._generation_started_at = started_at
        self._started = True

    async def abort(self) -> None:
        """Cancel and await every Session-scoped task before abandoning the Session."""
        if self._aborted:
            task = self._abort_task
            if task is None:
                task = asyncio.create_task(self._finish_abort())
                self._abort_task = task
            await await_task_preserving_cancellation(task)
            return
        self._request_abort()
        task = self._abort_task
        if task is None:
            task = asyncio.create_task(self._finish_abort())
            self._abort_task = task
        await await_task_preserving_cancellation(task)

    def _request_abort(self) -> None:
        """Synchronously stop new work before the awaited abort barrier runs."""
        if self._aborted:
            return
        self._aborted = True
        self._confirmation_requester = None
        self._active_foreground_owner = None
        if not self._started:
            self._abandon_session()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in self._owned_tasks():
            if task is current or task.done():
                continue
            self._retain_aborted_task(task)

    async def _finish_abort(self) -> None:
        try:
            await self._drain_owned_tasks()
            self._abandon_session()
            await self._session.wait_for_pending_persist()
        finally:
            self._clear_owned_task_references()

    def _owned_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        tasks: list[asyncio.Task[Any]] = []
        for task in (self._execution_task,):
            if task is not None:
                tasks.append(task)
        tasks.extend(self._session_run_state.title_tasks)
        tasks.extend(self._schedule_tasks)
        return tuple(dict.fromkeys(tasks))

    async def _drain_owned_tasks(self) -> None:
        tasks = self._owned_tasks()
        current = asyncio.current_task()
        awaitable_tasks = tuple(task for task in tasks if task is not current)
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)
        for task in awaitable_tasks:
            self._aborted_tasks.discard(task)
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except BaseException as error:
                    logger.warning(
                        "Drained Agent Loop task failed type={}",
                        type(error).__name__,
                    )

    def _clear_owned_task_references(self) -> None:
        self._execution_task = None
        self._execution_ready = None
        self._session_run_state.clear_title_work()
        self._schedule_tasks.clear()
        self._aborted_tasks.clear()

    def _abandon_session(self) -> None:
        if self._session_abandoned:
            return
        self._session.abandon()
        self._session_abandoned = True

    def _retain_aborted_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done():
            return
        self._aborted_tasks.add(task)
        task.add_done_callback(self._aborted_task_finished)
        task.cancel()

    def _aborted_task_finished(self, task: asyncio.Task[Any]) -> None:
        self._aborted_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as error:
            logger.warning(
                "Aborted Agent Loop task failed type={}",
                type(error).__name__,
            )

    async def cancel_active_run(self) -> None:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        active = self._execution_task
        if active is None or active.done():
            return
        self._cancel_requested = True
        ready = self._execution_ready
        if ready is not None and not ready.is_set():
            await ready.wait()
        await asyncio.sleep(0)
        if active.done():
            return
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)

    async def run_foreground(self, inbound: InboundMessage) -> None:
        """Execute one foreground input without owning a persistent consumer."""
        if self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if not self._started:
            raise RuntimeError("Agent Loop is not started")
        if self._execution_task is not None and not self._execution_task.done():
            raise RuntimeError("Agent Loop already has an active foreground Run")
        execution_ready = asyncio.Event()
        execution = asyncio.create_task(
            self._execute_foreground(inbound, execution_ready=execution_ready)
        )
        self._execution_task = execution
        self._execution_ready = execution_ready
        try:
            await execution
        finally:
            if self._execution_task is execution:
                self._execution_task = None
            if self._execution_ready is execution_ready:
                self._execution_ready = None
            self._cancel_requested = False

    async def run_schedule_job(
        self,
        job: ScheduleJob,
        occurrence: ScheduleOccurrence | None = None,
    ) -> None:
        """Execute one Schedule Job without using foreground state or output."""
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        if job.source != "user":
            raise ScheduleJobExecutionError(
                ErrorInfo(
                    "schedule_state_error",
                    "Only User Schedule Jobs may run through Agent Loop.",
                )
            )
        if occurrence is not None and occurrence.job != job:
            raise ValueError("Schedule occurrence Job does not match its callback Job")
        background_owner: BackgroundConfirmationOwner | None = None
        if occurrence is not None and occurrence.permission_snapshot is not None:
            background_owner = BackgroundConfirmationOwner(
                generation_id=self._session_run_state.generation_id,
                job_id=job.job_id,
                occurrence_id=occurrence.occurrence_id,
            )
            self._schedule_service.bind_occurrence_owner(occurrence, background_owner)
        current_task = asyncio.current_task()
        if current_task is not None:
            self._schedule_tasks.add(current_task)
        try:
            await self._execute_schedule_job(job, occurrence)
        finally:
            if occurrence is not None and background_owner is not None:
                self._schedule_service.unbind_occurrence_owner(occurrence, background_owner)
            if current_task is not None:
                self._schedule_tasks.discard(current_task)

    async def _execute_schedule_job(
        self,
        job: ScheduleJob,
        occurrence: ScheduleOccurrence | None = None,
    ) -> None:
        schedule_session: Session | None = None
        workspace_state = self._session.workspace_state
        with session_log(workspace_state, job.session_id):
            try:
                try:
                    if self._session.session_id == job.session_id:
                        schedule_session = self._session
                    else:
                        schedule_session = Session.load(
                            workspace_state,
                            job.session_id,
                            partition=SessionStoragePartition.SCHEDULE,
                            now=self._schedule_now,
                        )
                except FileNotFoundError:
                    schedule_session = Session.create_schedule(
                        workspace_state,
                        job.job_id,
                        now=self._schedule_now,
                        title=cast(str, job.title),
                    )
                try:
                    if occurrence is None:
                        await self._run_schedule_agent(schedule_session, job)
                    else:
                        await self._run_schedule_agent(schedule_session, job, occurrence)
                except ScheduleJobExecutionError as failure:
                    logger.warning(
                        "Schedule Job failed job_id={} kind={} code={}",
                        job.job_id,
                        job.schedule.kind,
                        failure.error.code,
                    )
                    raise
            finally:
                if schedule_session is not None:
                    try:
                        if schedule_session is not self._session:
                            if self._aborted:
                                schedule_session.abandon()
                            else:
                                schedule_session.close()
                        persist_drain = asyncio.create_task(
                            schedule_session.wait_for_pending_persist()
                        )
                        await await_task_preserving_cancellation(persist_drain)
                    except Exception as error:
                        logger.error(
                            "Schedule Session close failed job_id={} type={}",
                            job.job_id,
                            type(error).__name__,
                        )

    async def _run_schedule_agent(
        self,
        session: Session,
        job: ScheduleJob,
        occurrence: ScheduleOccurrence | None = None,
    ) -> None:
        with self._context_builder.schedule_projection_scope():
            await self._run_schedule_agent_scoped(session, job, occurrence)

    async def _run_schedule_agent_scoped(
        self,
        session: Session,
        job: ScheduleJob,
        occurrence: ScheduleOccurrence | None = None,
    ) -> None:
        current_user = {"role": "user", "content": job.message}
        permission_snapshot = (
            self._captured_foreground[0] if self._captured_foreground is not None else
            None if occurrence is None else occurrence.permission_snapshot
        )
        permission_context = (
            PermissionContext.from_snapshot(
                permission_snapshot,
                workspace_root=self._workspace_state.workspace_path,
                origin="schedule",
                configured_schedule_level=self._permission_control.configured(),
            )
            if permission_snapshot is not None
            else PermissionContext(
                origin="schedule",
                workspace_root=self._workspace_state.workspace_path,
            )
        )
        subagents_eligible = (
            self._subagent_coordinator is not None
            and occurrence is not None
            and permission_snapshot is not None
        )
        run_gateway = self._new_run_gateway(
            excluded_names=(
                ("schedule",)
                if subagents_eligible
                else ("schedule", "spawn_agent", "wait_agent", "list_agents")
            ),
            permission_context=permission_context,
        )

        def project_messages(
            history: Sequence[dict[str, Any]],
            current_user: dict[str, Any] | None,
            increment: Sequence[dict[str, Any]],
            compaction_cursor: int,
            action_summary: str | None,
        ) -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {"session_id": session.session_id, "summary": ""}
            if permission_snapshot is not None:
                kwargs["permission_snapshot"] = permission_snapshot
            return self._context_builder.build_run_messages(
                history,
                current_user=current_user,
                increment=increment,
                compaction_cursor=compaction_cursor,
                action_summary=action_summary,
                project_messages=lambda messages: self._context_builder.build_schedule_messages(
                    messages,
                    **kwargs,
                ),
            )

        run_context = self._new_agent_run_context(
            session,
            current_user=deepcopy(current_user),
            route="schedule",
            project_messages=project_messages,
        )
        if subagents_eligible:
            assert occurrence is not None and permission_snapshot is not None
            occurrence_id = str(occurrence.occurrence_id)
            run_gateway = self._subagent_run_gateway(
                run_context,
                run_gateway,
                source=SubAgentSource(
                    kind=SubAgentSourceKind.SCHEDULE,
                    job_id=job.job_id,
                    occurrence_id=occurrence_id,
                ),
                parent_run_id=occurrence_id,
                permission_snapshot=permission_snapshot,
                permission_context=permission_context,
                excluded_names=("schedule",),
                system_prompt=self._context_builder.schedule_system_prompt(),
            )
        schedule_confirmation: (
            Callable[[ConfirmationRequest], Awaitable[ConfirmationDecision]] | None
        ) = None
        if occurrence is not None and permission_snapshot is not None:
            background_owner = self._schedule_service.occurrence_owner(occurrence)

            async def request_schedule_confirmation(
                request: ConfirmationRequest,
            ) -> ConfirmationDecision:
                lifecycle_aborted = False
                try:
                    self._schedule_service.confirmation_waiting(occurrence)
                    requester = self._confirmation_requester
                    if requester is None:
                        raise ConfirmationUnavailable("confirmation requester is not bound")
                    return await requester(
                        ConfirmationEnvelope(
                            request=request,
                            origin="background",
                            owner=background_owner,
                            job_id=job.job_id,
                            title=cast(str, job.title),
                        )
                    )
                except ConfirmationAborted:
                    lifecycle_aborted = True
                    self._schedule_service.confirmation_aborted(occurrence)
                    raise
                finally:
                    if not lifecycle_aborted:
                        self._schedule_service.confirmation_finished(occurrence)

            schedule_confirmation = request_schedule_confirmation

        try:
            initial_messages = await self._prepare_agent_run(
                run_context,
                tool_gateway=run_gateway,
            )
        except asyncio.CancelledError:
            if not self._aborted:
                self._record_schedule_failure(
                    session,
                    run_context,
                    current_user,
                    ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                    job,
                )
            raise
        except CommittableAgentRunError as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            self._commit_schedule_failure(session, run_context, current_user, failure.error, job)
        except ModelCallError as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            if failure.error.code == "model_context_overflow":
                raise ScheduleJobExecutionError(failure.error) from failure
            self._commit_schedule_failure(session, run_context, current_user, failure.error, job)
        except ConfigError as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            self._commit_schedule_failure(session, run_context, current_user, failure.error, job)
        except Exception as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            _runtime_logger().error(
                "Schedule Agent Run preparation failed unexpectedly job_id={} type={}",
                job.job_id,
                type(failure).__name__,
            )
            self._commit_schedule_failure(
                session,
                run_context,
                current_user,
                ErrorInfo("model_failed", "The model request failed."),
                job,
            )

        try:
            result = await run_context.runner.run(
                initial_messages,
                model="schedule",
                tool_gateway=run_gateway,
                on_output=None,
                confirmation=schedule_confirmation,
                externalize_result=self._result_externalizer_for(session),
                cancel_requested=lambda: self._schedule_service.job_cancellation_requested(
                    job.job_id
                ),
                max_iterations=self._max_iterations,
            )
        except ConfirmationAborted:
            if self._aborted:
                raise asyncio.CancelledError() from None
            self._record_schedule_failure(
                session,
                run_context,
                current_user,
                ErrorInfo(
                    "tool_failed",
                    "Schedule Tool confirmation was aborted.",
                ),
                job,
            )
            raise
        except ModelCallError as failure:
            raise ScheduleJobExecutionError(failure.error) from failure
        if self._aborted:
            raise asyncio.CancelledError()
        if result.finish_reason == "cancelled" and not result.messages:
            self._commit_schedule_failure(
                session,
                run_context,
                current_user,
                result.error or ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                job,
            )
        self._commit_schedule_run(
            session,
            run_context,
            [deepcopy(current_user), *deepcopy(result.messages)],
            job=job,
        )

        if result.finish_reason == "cancelled":
            raise asyncio.CancelledError()
        if result.finish_reason != "completed":
            error = result.error or ErrorInfo("model_failed", "The model request failed.")
            raise ScheduleJobExecutionError(error)

    def _new_agent_run_context(
        self,
        session: Session,
        *,
        current_user: dict[str, Any],
        route: Literal["chat", "schedule"],
        project_messages: CompactionProjection,
        session_model_configuration: SessionModelConfiguration | None = None,
    ) -> _AgentRunContext:
        run_router = RunModelRouter(
            self._model_router,
            guard=agent_run_attempt_guard,
            session_model_configuration=session_model_configuration,
        )
        controller = AgentRunContextController.from_session(
            session,
            provider=run_router,
            append_summary=self._memory_manager.append_summary,
            now=self._now,
        )
        request_preparer = AgentRunContextRequestPreparer(
            controller,
            router=run_router,
            requested_route=route,
            project_messages=project_messages,
            project_tool_results=self._context_builder.project_tool_results,
            current_user=current_user,
            compact_ratio=self._configuration.runtime.compact_ratio,
            enable_tool_micro_compression=self._configuration.runtime.enable_tool_micro_compression,
        )
        return _AgentRunContext(
            route=route,
            current_user=deepcopy(current_user),
            project_messages=project_messages,
            router=run_router,
            controller=controller,
            runner=AgentRunner(run_router, request_preparer),
        )

    async def _prepare_agent_run(
        self,
        context: _AgentRunContext,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        route_status = context.router.call_route_status(context.route, continuation=None)
        memory_route_status = context.router.call_route_status("memory", continuation=None)
        retained_messages = await context.controller.prepare_run_start(
            project_messages=context.project_messages,
            route_status=route_status,
            memory_route_status=memory_route_status,
            tools=tool_gateway.schemas,
            current_user=deepcopy(context.current_user),
            compact_ratio=self._configuration.runtime.compact_ratio,
        )
        return list(deepcopy(retained_messages))

    def _commit_agent_run(
        self,
        session: Session,
        context: _AgentRunContext,
        messages: list[dict[str, Any]],
        *,
        usage_delta: dict[str, int] | None = None,
        metadata_updates: dict[str, Any] | None = None,
        metadata_removals: tuple[str, ...] = (),
        restore_before: SessionRestoreBefore | None = None,
        restore_run_token: UUID | None = None,
    ) -> None:
        values = context.controller.terminal_commit_values()
        combined_usage = _merge_usage_deltas(values.usage_delta, usage_delta)
        session.commit_agent_run(
            messages,
            pending_last_compacted=values.pending_last_compacted,
            pending_action_summary=values.pending_action_summary,
            usage_delta=combined_usage or None,
            metadata_updates=metadata_updates,
            metadata_removals=metadata_removals,
            restore_before=restore_before,
            restore_run_token=restore_run_token,
        )

    def _commit_schedule_run(
        self,
        session: Session,
        context: _AgentRunContext,
        messages: list[dict[str, Any]],
        *,
        job: ScheduleJob,
    ) -> None:
        try:
            self._commit_agent_run(session, context, messages)
        except (OSError, UnicodeError) as error:
            logger.error(
                "Schedule Agent Run commit failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )
            raise ScheduleJobExecutionError(
                ErrorInfo("persistence_error", "The Conversation Session could not be updated.")
            ) from error
        except Exception as error:
            _runtime_logger().error(
                "Schedule Agent Run commit contract failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )
            raise ScheduleJobExecutionError(
                ErrorInfo("model_failed", "The model request failed.")
            ) from error

    def _commit_schedule_failure(
        self,
        session: Session,
        context: _AgentRunContext,
        current_user: dict[str, Any],
        error: ErrorInfo,
        job: ScheduleJob,
    ) -> NoReturn:
        self._record_schedule_failure(session, context, current_user, error, job)
        if error.code == "turn_cancelled":
            raise asyncio.CancelledError()
        raise ScheduleJobExecutionError(error)

    def _record_schedule_failure(
        self,
        session: Session,
        context: _AgentRunContext,
        current_user: dict[str, Any],
        error: ErrorInfo,
        job: ScheduleJob,
    ) -> None:
        self._commit_schedule_run(
            session,
            context,
            [
                deepcopy(current_user),
                _build_assistant_repair_message(
                    content=(TURN_CANCELLED_MESSAGE if error.code == "turn_cancelled" else ""),
                    status="interrupted" if error.code == "turn_cancelled" else "error",
                    error=error,
                    model_calls=0,
                ),
            ],
            job=job,
        )

    async def _execute_foreground(
        self,
        inbound: InboundMessage,
        *,
        execution_ready: asyncio.Event,
    ) -> None:
        active_session = self._session
        if not inbound.content.strip():
            execution_ready.set()
            return
        restore_before = active_session.capture_restore_before()
        restore_run_token = self._new_uuid()
        if self._captured_foreground is None:
            permission_snapshot = self._permission_control.snapshot(self._exec_host.resolved_shell)
            session_model_configuration = active_session.model_configuration
        else:
            permission_snapshot, session_model_configuration = self._captured_foreground
        self._run_model_configuration = session_model_configuration
        skill_state = self._skill_loader.skills
        manual_invocation = self._skill_loader.resolve_manual(inbound.content)
        start_title = not active_session.messages
        created_title_work = (
            self._session_run_state.start_title(
                active_session, inbound.content, self._resolve_title, lambda: self._aborted
            ) if start_title else None
        )
        title_work = created_title_work or self._session_run_state.title_for(active_session.session_id)
        if title_work is not None and title_work.task.done():
            title_work = None
        title_coordination = None if title_work is None else title_work.coordination
        if title_coordination is not None:
            title_coordination.attach_foreground()
        committed = False
        try:
            with self._context_builder.foreground_projection_scope(skill_state):
                if title_work is None:
                    with session_log(active_session):
                        committed = await self._execute_foreground_logged(
                            active_session,
                            inbound,
                            title_work=None,
                            manual_invocation=manual_invocation,
                            execution_ready=execution_ready,
                            permission_snapshot=permission_snapshot,
                            session_model_configuration=session_model_configuration,
                            restore_before=restore_before,
                            restore_run_token=restore_run_token,
                        )
                else:
                    assert title_coordination is not None
                    await title_coordination.log_ready.wait()
                    with logger.contextualize(session_id=active_session.session_id):
                        committed = await self._execute_foreground_logged(
                            active_session,
                            inbound,
                            title_work=title_work,
                            manual_invocation=manual_invocation,
                            execution_ready=execution_ready,
                            permission_snapshot=permission_snapshot,
                            session_model_configuration=session_model_configuration,
                            restore_before=restore_before,
                            restore_run_token=restore_run_token,
                        )
        finally:
            execution_ready.set()
            if title_coordination is not None:
                title_coordination.release_foreground()
            if created_title_work is not None:
                created_coordination = created_title_work.coordination
                if not created_coordination.prepared.done():
                    created_coordination.prepared.set_result(False)
                if not committed:
                    await self._session_run_state.discard_uncommitted_title(
                        active_session.session_id, created_title_work
                    )

    async def _execute_foreground_logged(
        self,
        active_session: Session,
        inbound: InboundMessage,
        *,
        title_work: TitleWork | None,
        manual_invocation: ManualSkillInvocation | None = None,
        execution_ready: asyncio.Event,
        permission_snapshot: PermissionSnapshot,
        session_model_configuration: SessionModelConfiguration | None = None,
        restore_before: SessionRestoreBefore | None = None,
        restore_run_token: UUID | None = None,
    ) -> bool:
        current_user = {"role": "user", "content": inbound.content}
        if title_work is not None:
            title_work.coordination.preparation_started.set()
        execution_ready.set()
        if not inbound.content.strip():
            return False

        staged_blackboard: Blackboard | None = None

        def project_messages(
            history: Sequence[dict[str, Any]],
            current_user: dict[str, Any] | None,
            increment: Sequence[dict[str, Any]],
            compaction_cursor: int,
            action_summary: str | None,
        ) -> list[dict[str, Any]]:
            return self._context_builder.build_run_messages(
                history,
                current_user=current_user,
                increment=increment,
                compaction_cursor=compaction_cursor,
                action_summary=action_summary,
                project_messages=lambda messages: self._context_builder.build_foreground_messages(
                    messages,
                    session_id=active_session.session_id,
                    blackboard=staged_blackboard,
                    manual_invocation=manual_invocation,
                    summary="",
                    permission_snapshot=permission_snapshot,
                ),
            )

        run_context = self._new_agent_run_context(
            active_session,
            current_user=deepcopy(current_user),
            route="chat",
            project_messages=project_messages,
            session_model_configuration=session_model_configuration,
        )
        framing_usage: dict[str, int] | None = None

        def metadata_patch() -> tuple[dict[str, Any] | None, tuple[str, ...]]:
            if manual_invocation is not None:
                return None, ()
            if staged_blackboard is None:
                return None, ("blackboard",)
            return {"blackboard": staged_blackboard.to_dict()}, ()

        if manual_invocation is None:
            previous_blackboard = Blackboard.from_dict(active_session.metadata.get("blackboard"))
            last_assistant_content = _latest_assistant_content(active_session)
            try:
                framing_result = await Blackboard.generate(
                    self._model_router,
                    previous=previous_blackboard,
                    last_assistant_content=last_assistant_content,
                    current_user_input=inbound.content,
                )
            except asyncio.CancelledError:
                if not self._cancel_requested:
                    raise
                return await self._finish_foreground_terminal(
                    active_session,
                    run_context,
                    current_user,
                    error=ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                    framing_usage=framing_usage,
                    metadata_updates=metadata_patch()[0],
                    metadata_removals=metadata_patch()[1],
                    restore_before=restore_before,
                    restore_run_token=restore_run_token,
                )

            if framing_result.status != "resolved":
                _runtime_logger().warning(
                    "Task Framing degraded status={}",
                    framing_result.status,
                )
            staged_blackboard = framing_result.blackboard
            framing_usage = framing_result.usage_delta
        else:
            staged_blackboard = None
            framing_usage = None
        subagents_eligible = (
            self._subagent_coordinator is not None
            and restore_run_token is not None
        )
        run_gateway = self._new_run_gateway(
            excluded_names=(
                () if subagents_eligible else ("spawn_agent", "wait_agent", "list_agents")
            ),
            permission_snapshot=permission_snapshot,
        )
        if subagents_eligible:
            assert restore_run_token is not None
            parent_run_id = str(restore_run_token)
            run_gateway = self._subagent_run_gateway(
                run_context,
                run_gateway,
                source=SubAgentSource(
                    kind=SubAgentSourceKind.FOREGROUND,
                    restore_run_token=parent_run_id,
                ),
                parent_run_id=parent_run_id,
                permission_snapshot=permission_snapshot,
                session_model_configuration=session_model_configuration,
                system_prompt=self._context_builder.foreground_system_prompt(),
            )
        try:
            initial_messages = await self._prepare_agent_run(
                run_context,
                tool_gateway=run_gateway,
            )
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        except CommittableAgentRunError as failure:
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=failure.error,
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        except ModelCallError as failure:
            if failure.error.code == "model_context_overflow":
                await self._publish_preparation_failure(failure.error)
                return True
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=failure.error,
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        except Exception as error:
            _runtime_logger().error(
                "Agent Run preparation failed unexpectedly type={}",
                type(error).__name__,
            )
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=ErrorInfo("model_failed", "The model request failed."),
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )

        if title_work is not None and not title_work.coordination.prepared.done():
            title_work.coordination.prepared.set_result(True)
        foreground_owner = ForegroundConfirmationOwner(
            generation_id=self._session_run_state.generation_id,
            run_id=self._new_uuid(),
        )
        self._active_foreground_owner = foreground_owner
        file_mutation_recorder = FileBackupStore(
            active_session.workspace_state,
            active_session.session_id,
        )
        try:
            result = await run_context.runner.run(
                initial_messages,
                model="chat",
                tool_gateway=run_gateway,
                on_output=self._publish_runner_output,
                confirmation=self._request_confirmation,
                externalize_result=self._result_externalizer_for(active_session),
                cancel_requested=lambda: self._cancel_requested,
                max_iterations=self._max_iterations,
                file_mutation_recorder=file_mutation_recorder,
                run_token=restore_run_token,
            )
        except ModelCallError as failure:
            if failure.error.code == "turn_cancelled":
                return await self._finish_foreground_terminal(
                    active_session,
                    run_context,
                    current_user,
                    error=failure.error,
                    framing_usage=framing_usage,
                    metadata_updates=metadata_patch()[0],
                    metadata_removals=metadata_patch()[1],
                    restore_before=restore_before,
                    restore_run_token=restore_run_token,
                )
            await self._publish_preparation_failure(failure.error)
            return True
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        finally:
            if self._active_foreground_owner is foreground_owner:
                self._active_foreground_owner = None

        if result.finish_reason == "cancelled" and not result.messages:
            return await self._finish_foreground_terminal(
                active_session,
                run_context,
                current_user,
                error=result.error or ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                framing_usage=framing_usage,
                metadata_updates=metadata_patch()[0],
                metadata_removals=metadata_patch()[1],
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )

        if self._aborted:
            return False

        metadata_removals: tuple[str, ...]
        if manual_invocation is not None:
            metadata_updates = None
            metadata_removals = ()
        elif staged_blackboard is None:
            metadata_updates = None
            metadata_removals = ("blackboard",)
        else:
            metadata_updates = {"blackboard": staged_blackboard.to_dict()}
            metadata_removals = ()

        run_messages = deepcopy(result.messages)
        if result.finish_reason == "cancelled" and not any(
            message.get("status") == "interrupted" for message in run_messages
        ):
            run_messages.append(_build_assistant_repair_message(
                content=(result.error.message if result.error else TURN_CANCELLED_MESSAGE),
                status="interrupted",
                error=result.error or ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                model_calls=0,
            ))
        try:
            if self._aborted:
                return False
            self._commit_agent_run(
                active_session,
                run_context,
                [deepcopy(current_user), *run_messages],
                usage_delta=framing_usage,
                metadata_updates=metadata_updates,
                metadata_removals=metadata_removals,
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        except (OSError, UnicodeError) as failure:
            _runtime_logger().error(
                "Agent Run Session increment failed code=persistence_error type={}",
                type(failure).__name__,
            )
            await self._publish_commit_failure()
            return False
        except Exception as failure:
            _runtime_logger().error(
                "Agent Run Session increment contract failed type={}",
                type(failure).__name__,
            )
            await self._publish_preparation_failure(
                ErrorInfo("model_failed", "The model request failed.")
            )
            return False
        await self._publish_terminal(result)
        return True

    async def _finish_foreground_terminal(
        self,
        active_session: Session,
        context: _AgentRunContext,
        current_user: dict[str, Any],
        *,
        error: ErrorInfo,
        framing_usage: dict[str, int] | None,
        metadata_updates: dict[str, Any] | None,
        metadata_removals: tuple[str, ...],
        restore_before: SessionRestoreBefore | None,
        restore_run_token: UUID | None,
    ) -> bool:
        if self._aborted:
            return False
        try:
            if self._aborted:
                return False
            self._commit_agent_run(
                active_session,
                context,
                [
                    deepcopy(current_user),
                    _build_assistant_repair_message(
                        content=(
                            TURN_CANCELLED_MESSAGE if error.code == "turn_cancelled" else ""
                        ),
                        status="interrupted" if error.code == "turn_cancelled" else "error",
                        error=error,
                        model_calls=0,
                    ),
                ],
                usage_delta=framing_usage,
                metadata_updates=metadata_updates,
                metadata_removals=metadata_removals,
                restore_before=restore_before,
                restore_run_token=restore_run_token,
            )
        except (OSError, UnicodeError) as failure:
            _runtime_logger().error(
                "Agent Run preparation commit failed code=persistence_error type={}",
                type(failure).__name__,
            )
            await self._publish_commit_failure()
            return False
        except Exception as failure:
            _runtime_logger().error(
                "Agent Run preparation commit contract failed type={}",
                type(failure).__name__,
            )
            await self._publish_preparation_failure(
                ErrorInfo("model_failed", "The model request failed.")
            )
            return False
        await self._publish_preparation_failure(error)
        return True

    def _result_externalizer_for(
        self,
        active_session: Session,
    ) -> Callable[[ToolResult], ToolResult] | None:
        max_tool_result_chars = self._configuration.runtime.max_tool_result_chars

        def externalize(result: ToolResult) -> ToolResult:
            if result.status != "success" or len(result.content) <= max_tool_result_chars:
                return result
            output = BaseTool.handle_result(
                result.content,
                workspace=active_session.workspace_state.workspace_path,
                session_id=active_session.session_id,
                tool_call_id=result.tool_call_id,
                limit=max_tool_result_chars,
            )
            return replace(result, content=output.content, artifact=output.artifact)

        return externalize

    async def _publish_runner_output(self, event: object) -> None:
        if self._aborted:
            return
        if isinstance(event, ReasoningDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_reasoning",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, TextDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_response",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, AgentRunnerResponseSegmentEnd):
            outbound_type: OutboundMessageType = (
                "model_reasoning" if event.segment == "reasoning" else "model_response"
            )
            await self._bus.put_outbound(OutboundMessage(outbound_type, "", {"_stream_end": True}))
            return
        if isinstance(event, AgentRunnerToolCallStarted):
            await self._bus.put_outbound(
                OutboundMessage(
                    "tool_call",
                    event.tool_name,
                    {
                        "tool_call_id": event.tool_call_id,
                        "arguments": event.arguments,
                    },
                )
            )
            return
        if isinstance(event, AgentRunnerToolCallFinished):
            await self._bus.put_outbound(
                OutboundMessage(
                    "tool_call",
                    event.tool_name,
                    {"tool_call_id": event.tool_call_id, "status": event.status,
                     "result": event.result},
                )
            )
            return
        raise TypeError(f"Unsupported Agent Runner output: {type(event).__name__}")

    async def _publish_terminal(self, result: AgentRunnerResult) -> None:
        if self._aborted:
            return
        if result.finish_reason == "completed":
            await self._bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))
            return
        error = result.error
        if error is None:
            error = ErrorInfo("model_failed", "The model request failed.")
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": result.finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_preparation_failure(self, error: ErrorInfo) -> None:
        if self._aborted:
            return
        if error.code != "turn_cancelled":
            _log_agent_failure(error)
        finish_reason = "cancelled" if error.code == "turn_cancelled" else "failed"
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_commit_failure(self) -> None:
        if self._aborted:
            return
        error = ErrorInfo(
            "persistence_error",
            "The Conversation Session could not be updated.",
        )
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": "failed",
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _request_confirmation(
        self,
        request: ConfirmationRequest,
    ) -> ConfirmationDecision:
        if self._aborted:
            raise asyncio.CancelledError()
        requester = self._confirmation_requester
        if requester is not None:
            owner = self._active_foreground_owner
            if owner is None:
                raise RuntimeError("foreground confirmation owner is not bound")
            return await requester(
                ConfirmationEnvelope(
                    request=request,
                    origin="foreground",
                    owner=owner,
                )
            )
        raise RuntimeError("Agent Loop confirmation requester is not bound")


    async def _resolve_title(self, content: str) -> tuple[str, dict[str, int] | None]:
        title = normalize_title(content)
        usage_delta: dict[str, int] | None = None
        events: Any = None
        try:
            events = self._router_stream_title(content)
            async for event in events:
                if not isinstance(event, ModelCompleted):
                    continue
                response = event.response
                usage_delta = {"model_calls": 1, **response.usage.to_dict()}
                if response.message.tool_calls:
                    continue
                candidate = normalize_title_candidate(response.message.content)
                if candidate:
                    title = candidate
                break
        except Exception as error:
            _runtime_logger().opt(exception=error).warning(
                "Session title fallback selected type={}", type(error).__name__
            )
        finally:
            if events is not None:
                close = getattr(events, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except RuntimeError:
                        pass
        return title, usage_delta

    def _router_stream_title(self, content: str) -> Any:
        messages = self._context_builder.build_title_messages(normalize_title(content))
        return self._model_router.stream(
            "title",
            messages=messages,
            tools=(),
            continuation=None,
        )


def session_runtime_status_input(
    session: Session,
    *,
    configuration: UserConfiguration,
    context_builder: ContextBuilder,
    tool_schemas: tuple[dict[str, Any], ...],
    generation_started_at: float | None,
) -> RuntimeStatusInput:
    """Project resident history for management without constructing an executor."""
    session_model_configuration = session.model_configuration
    model_configuration_available = True
    try:
        route_status = _configured_model_route_status(
            configuration,
            "chat",
            session_model_configuration=session_model_configuration,
        )
    except ValueError:
        if session_model_configuration is None:
            raise
        model_configuration_available = False
        route_status = _configured_model_route_status(configuration, "chat")
    session_id = session.session_id
    messages = session.messages
    metadata = session.metadata
    last_compacted = session.last_compacted
    title = metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("Active Session title is malformed")
    usage_value = metadata.get("token_usage")
    if not isinstance(usage_value, dict):
        raise ValueError("Active Session token usage is malformed")
    summary = _action_summary_from_metadata(metadata)
    usage_fields = ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    usage = tuple((field, usage_value.get(field)) for field in usage_fields)
    if any(isinstance(value, bool) or not isinstance(value, int) for _, value in usage):
        raise ValueError("Active Session token usage is malformed")
    usage_anchor = latest_main_agent_usage_anchor(messages)
    last_request_usage = next(
        (
            {
                "input_tokens": message["token_usage"]["input_tokens"],
                "cached_input_tokens": message.get("cached_input_tokens"),
            }
            for message in reversed(messages)
            if message.get("role") == "assistant"
            and message.get("status") == "completed"
            and message["token_usage"]["model_calls"] == 1
        ),
        None,
    )
    latest_usage_context: ContextUsageSnapshot | None = None
    latest_reported_usage: tuple[tuple[str, int], ...] = ()
    if usage_anchor is not None:
        latest_usage_context, reported_usage = usage_anchor
        latest_reported_usage = tuple((field, reported_usage[field]) for field in usage_fields)
    status_input = _foreground_runtime_status_input(
        context_builder=context_builder,
        history=messages[last_compacted:],
        session_id=session_id,
        tool_schemas=tool_schemas,
        summary=summary,
        blackboard=Blackboard.from_dict(metadata.get("blackboard")),
        session_title=title,
        session_message_count=len(messages),
        last_compacted=last_compacted,
        cumulative_usage=tuple((field, cast(int, value)) for field, value in usage),
        chat_model=f"{route_status.provider_id}/{route_status.model}",
        chat_reasoning_effort=(
            None
            if session_model_configuration is None
            else session_model_configuration.reasoning_effort
        ),
        context_window=route_status.context_window,
        generation_started_at=generation_started_at,
        max_output=route_status.max_output,
        compact_ratio=configuration.runtime.compact_ratio,
        requested_route="chat",
        selected_route=route_status.selected_route,
        provider_id=route_status.provider_id,
        model=route_status.model,
        latest_usage_context=latest_usage_context,
        latest_reported_usage=latest_reported_usage,
    )
    return replace(
        status_input,
        model_configuration_available=model_configuration_available,
        last_request_usage=last_request_usage,
    )


__all__ = [
    "AgentRunExecutor",
    "ConfirmationCallback",
    "ConfirmationRequestView",
    "ModelContextOverflowError",
]


def _runtime_logger() -> Any:
    def set_runtime_name(record: Any) -> None:
        record["name"] = "aide.agent.loop"

    return logger.patch(set_runtime_name)


def _log_agent_failure(error: ErrorInfo) -> None:
    failure = ModelCallError(error)
    _runtime_logger().opt(exception=failure).error(
        "Agent Run failed code={} type={}",
        error.code,
        type(failure).__name__,
    )


def _project_terminal_message(message: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(message)
    projected.pop("restore_anchor_id", None)
    projected.pop("restore_run_token", None)
    projected.pop("restore_before", None)
    return projected


def _latest_assistant_content(session: Session) -> str:
    for message in reversed(session.messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _merge_usage_deltas(
    first: Mapping[str, int] | None,
    second: Mapping[str, int] | None,
) -> dict[str, int]:
    result = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for delta in (first, second):
        if delta is None:
            continue
        for field in result:
            result[field] += delta[field]
    return result


def _configured_model_route_status(
    configuration: UserConfiguration,
    route: ModelRoute,
    *,
    session_model_configuration: SessionModelConfiguration | None = None,
) -> ModelRouteStatus:
    resolved = (
        configuration.resolve_session_model_route(
            session_model_configuration.provider_id,
            session_model_configuration.model,
            session_model_configuration.reasoning_effort,
        )
        if session_model_configuration is not None and route == "chat"
        else configuration.resolve_route(route)
    )
    return ModelRouteStatus(
        requested_route=route,
        selected_route=cast(ModelRoute, resolved.selected_route),
        provider_id=resolved.provider.provider_id,
        model=resolved.route.model,
        context_window=resolved.route.context_window,
        max_output=resolved.route.max_output,
        used_fallback=resolved.used_fallback,
    )


def _foreground_runtime_status_input(
    *,
    context_builder: ContextBuilder,
    history: Sequence[dict[str, Any]],
    session_id: str,
    tool_schemas: tuple[dict[str, Any], ...],
    summary: str = "",
    blackboard: Blackboard | None = None,
    session_title: str = "",
    session_message_count: int = 0,
    last_compacted: int = 0,
    cumulative_usage: tuple[tuple[str, int], ...] = (),
    chat_model: str = "",
    chat_reasoning_effort: ReasoningEffort | None = None,
    context_window: int = 0,
    max_output: int = 0,
    compact_ratio: float = 0.9,
    requested_route: str = "chat",
    selected_route: str = "chat",
    provider_id: str = "",
    model: str = "",
    latest_usage_context: ContextUsageSnapshot | None = None,
    latest_reported_usage: tuple[tuple[str, int], ...] = (),
    generation_started_at: float | None = None,
) -> RuntimeStatusInput:
    """Project and serialize a minimum foreground request for status and preflight."""
    if blackboard is None:
        projected = context_builder.build_status_messages(
            history,
            session_id=session_id,
            summary=summary,
        )
    else:
        projected = context_builder.build_status_messages(
            history,
            session_id=session_id,
            summary=summary,
            blackboard=blackboard,
        )
    return RuntimeStatusInput(
        session_id=session_id,
        session_title=session_title,
        session_message_count=session_message_count,
        last_compacted=last_compacted,
        cumulative_usage=cumulative_usage,
        chat_model=chat_model,
        chat_reasoning_effort=chat_reasoning_effort,
        context_window=context_window,
        max_output=max_output,
        compact_ratio=compact_ratio,
        requested_route=requested_route,
        selected_route=selected_route,
        provider_id=provider_id,
        model=model,
        projected_messages=tuple(deepcopy(projected)),
        projected_tools=tuple(deepcopy(tool_schemas)),
        latest_usage_context=latest_usage_context,
        latest_reported_usage=latest_reported_usage,
        generation_started_at=generation_started_at,
    )


def _action_summary_from_metadata(metadata: dict[str, Any]) -> str:
    summary = metadata.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError("Active Session action summary is malformed")
    return summary
