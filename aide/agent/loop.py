"""Serial foreground Agent Runner orchestration over the Runtime Message Bus."""

from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
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
from aide.agent.context.budget import ContextUsageSnapshot
from aide.agent.context.builder import ContextBuilder
from aide.agent.context.run_context import (
    AgentRunContextSnapshot,
    CompactionProjection,
    ContextController,
    latest_main_agent_usage_anchor,
)
from aide.agent.conversation_contracts import (
    ForegroundConversationProjection as ForegroundConversationProjection,
)
from aide.agent.conversation_contracts import (
    TerminalAgentRunExecutorControl as TerminalAgentRunExecutorControl,
)
from aide.agent.foreground_work.store import ForegroundWorkStore
from aide.agent.memory.manager import MemoryManager
from aide.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    OutboundMessageType,
)
from aide.agent.permission import (
    PERMISSION_LEVELS,
    PermissionSnapshot,
    RuntimePermissionControl,
    ToolPermissionLevel,
)
from aide.agent.router import (
    DispatchWorkerTool,
    FinishRouterTool,
    RouterAskUserTool,
    RouterCatalog,
    RouterDecisionValidator,
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
from aide.agent.tools.base import BaseTool, ToolError
from aide.agent.tools.context import AgentRunSuspension, ToolRunContext
from aide.agent.tools.core.ask_user import AskUserTool
from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.core.subagents import build_subagent_tools
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import (
    BuiltInToolCatalog,
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ToolGateway,
    ToolResult,
)
from aide.agent.workers.executor import WorkerRunnerExecutor
from aide.agent.workers.models import WorkerRecord
from aide.agent.workers.store import WorkerStore
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigError, UserConfiguration
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
from aide.skills.catalog import ManualSkillInvocation, SkillLoader
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder, TraceRecorder
from aide.trace.scope import current_trace_scope, trace_purpose, trace_scope
from aide.trace.types import TraceOrigin, TracePurpose
from aide.utils.async_tasks import await_task_preserving_cancellation
from aide.utils.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from aide.utils.text import normalize_title, normalize_title_candidate
from aide.utils.validation import empty_token_usage

type RuntimeConfirmationRequester = Callable[
    [ConfirmationEnvelope], Awaitable[ConfirmationDecision]
]
type WorkerScheduler = Callable[[WorkerRecord, Callable[[], Awaitable[None]]], None]
type WorkerGenerationFencer = Callable[[str, int], Awaitable[None]]


@dataclass(slots=True)
class _AgentRunContext:
    """Run-local budget, projection and guarded Router collaborators."""

    controller: ContextController
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
        trace_recorder: TraceRecorder | None = None,
        foreground_work_store: ForegroundWorkStore | None = None,
        worker_store: WorkerStore | None = None,
        worker_scheduler: WorkerScheduler | None = None,
        worker_generation_fencer: WorkerGenerationFencer | None = None,
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
        self._trace_recorder = (
            NoOpTraceRecorder() if trace_recorder is None else trace_recorder
        )
        self._foreground_work_store = foreground_work_store
        self._worker_store = worker_store
        self._worker_scheduler = worker_scheduler
        self._worker_generation_fencer = worker_generation_fencer
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
        self._schedule_outbound_suppressed = False
        self._aborted = False
        self._started = False
        self._preflighted = False
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
        allowed_names: Sequence[str] | None = None,
        permission_snapshot: PermissionSnapshot | None = None,
        permission_context: PermissionContext | None = None,
        tool_context: ToolRunContext | None = None,
        run_tools: Sequence[BaseTool] = (),
    ) -> ToolGateway:
        return build_agent_run_gateway(
            self._tool_gateway,
            excluded_names=excluded_names,
            allowed_names=allowed_names,
            mcp_keywords=self._mcp_keywords,
            permission_snapshot=permission_snapshot,
            permission_context=permission_context,
            tool_context=tool_context,
            run_tools=run_tools,
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
        coordinator_for: Callable[[], SubAgentSessionCoordinator | None] | None = None,
    ) -> SubAgentRunnerExecutor:
        """Build the child executor from this Service Loop's shared runtime resources."""
        model_router = self._subagent_model_router
        if model_router is None:
            raise RuntimeError("SubAgent execution requires the shared Service Model Router")
        coordinator = self._subagent_coordinator
        workspace_state = self._workspace_state
        foreground_work_store = self._foreground_work_store
        new_uuid = self._new_uuid
        parent_executor_reference = weakref.ref(self)

        def child_context(record: SubAgentRecord) -> SubAgentToolContext:
            active_coordinator = coordinator
            if active_coordinator is None and coordinator_for is not None:
                active_coordinator = coordinator_for()
            if active_coordinator is None:
                parent_executor = parent_executor_reference()
                active_coordinator = (
                    None if parent_executor is None else parent_executor._subagent_coordinator
                )
            if active_coordinator is None:
                raise RuntimeError("SubAgent coordinator is unavailable for this Agent Run")
            return SubAgentToolContext(
                coordinator=active_coordinator, parent_run_id=record.parent_run_id,
                source=record.source, creator_snapshot=record.creator_snapshot,
                caller_role="subagent", owner_agent_id=record.agent_id,
            )

        def child_question_handler_for(
            record: SubAgentRecord,
        ) -> Callable[[str, Sequence[Mapping[str, str]], str], str] | None:
            store = foreground_work_store
            if (
                store is None
                or record.source.kind not in {
                    SubAgentSourceKind.FOREGROUND,
                    SubAgentSourceKind.WORKER,
                }
                or "ask_user" not in record.creator_snapshot.tool_names
                or record.source.task_id is None
                or record.source.task_generation is None
            ):
                return None

            def register_question(
                question: str,
                options: Sequence[Mapping[str, str]],
                tool_call_id: str,
            ) -> str:
                question_id = str(new_uuid())
                store.prepare_question(
                    task_id=cast(str, record.source.task_id),
                    execution_id=cast(str, record.execution_id),
                    question_id=question_id,
                    tool_call_id=tool_call_id,
                    text=question,
                    options=[{"id": option["id"], "label": option["label"]} for option in options],
                    owner_agent_id=record.agent_id,
                )
                return question_id

            return register_question

        def publish_child_question(
            record: SubAgentRecord,
            question_id: str,
            committed_tool_call_ids: set[str],
        ) -> None:
            store = foreground_work_store
            if store is not None:
                store.publish_question(
                    question_id,
                    committed_tool_call_ids=committed_tool_call_ids,
                )

        def finish_child_execution(record: SubAgentRecord) -> None:
            store = foreground_work_store
            if (
                store is not None
                and record.source.kind in {
                    SubAgentSourceKind.FOREGROUND,
                    SubAgentSourceKind.WORKER,
                }
                and "ask_user" in record.creator_snapshot.tool_names
                and record.source.task_id is not None
                and record.source.task_generation is not None
            ):
                store.finish_subagent_execution(
                    record.source.task_id,
                    generation=record.source.task_generation,
                    agent_id=record.agent_id,
                    execution_id=cast(str, record.execution_id),
                )

        return SubAgentRunnerExecutor(
            workspace_id=workspace_id,
            workspace_state=workspace_state,
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
            tool_context_for=child_context,
            foreground_task_is_current=(
                None if self._foreground_work_store is None
                else self._foreground_work_store.is_active_generation
            ),
            foreground_question_handler_for=child_question_handler_for,
            foreground_question_publisher=publish_child_question,
            foreground_task_finished=finish_child_execution,
            file_mutation_recorder_for=(
                None if self._session._storage_partition is SessionStoragePartition.SCHEDULE
                else lambda record: FileBackupStore(workspace_state, record.session_id)
            ),
            now=self._now,
            trace_recorder=self._trace_recorder,
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
            caller_role="subagent",
            owner_agent_id=record.agent_id,
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
            creator_tool_names = tuple(tool.name for tool in run_gateway.catalog)
            if source.kind is SubAgentSourceKind.FOREGROUND and source.task_id is not None:
                creator_tool_names = tuple(dict.fromkeys((*creator_tool_names, "ask_user")))
            snapshot = SubAgentCreatorSnapshot(
                provider_id=route_status.provider_id,
                model=route_status.model,
                reasoning_effort=(
                    self._configuration.resolve_route("subagent").route.reasoning_effort
                ),
                permission_level=permission_snapshot.level,
                shell=self._exec_host.resolved_shell.selector,
                tool_names=creator_tool_names,
                system_prompt=system_prompt,
            )
        base_context = run_gateway.tool_context
        if base_context is None:
            raise RuntimeError("SubAgent execution requires a Tool Run context")
        tool_context = replace(
            base_context,
            subagent=SubAgentToolContext(
                coordinator=coordinator,
                parent_run_id=parent_run_id,
                source=source,
                creator_snapshot=snapshot,
                caller_role="main",
            ),
        )
        return self._new_run_gateway(
            excluded_names=(
                (*excluded_names, "spawn_agent") if snapshot is None else excluded_names
            ),
            permission_context=permission_context,
            permission_snapshot=(None if permission_context is not None else permission_snapshot),
            tool_context=tool_context,
            run_tools=tuple(tool for tool in run_gateway.catalog if tool.name == "ask_user"),
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
        self._preflighted = True

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

    async def run_automatic_continuation(self, inbound: InboundMessage) -> None:
        """Run an internal result through its owning Session authority."""
        if inbound.metadata.get("foreground_automatic_continuation") is not True:
            raise ValueError("automatic continuation metadata is required")
        await self.run_foreground(inbound)

    async def run_schedule_job(
        self,
        job: ScheduleJob,
        occurrence: ScheduleOccurrence | None = None,
    ) -> None:
        """Execute one Schedule Job in its isolated Session without outbound output."""
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
            if (
                occurrence is not None
                and self._session._storage_partition is SessionStoragePartition.SCHEDULE
                and self._foreground_work_store is not None
                and self._worker_store is not None
                and self._worker_scheduler is not None
            ):
                occurrence_id = str(occurrence.occurrence_id)
                inbound = InboundMessage(
                    job.message,
                    metadata={
                        "run_id": occurrence_id,
                        "schedule_occurrence_id": occurrence_id,
                        "schedule_permission_snapshot": occurrence.permission_snapshot,
                        "schedule_job_id": job.job_id,
                        "schedule_title": job.title,
                        "foreground_task_id": occurrence_id,
                        "foreground_task_generation": 1,
                    },
                )
                with self._context_builder.schedule_projection_scope():
                    await self.run_foreground(inbound)
            else:
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
        workspace_path = self._workspace_state.workspace_path
        schedule_run_id = None if occurrence is None else str(occurrence.occurrence_id)
        with trace_scope(
            self._trace_recorder,
            TraceRootSource(
                workspace_key=workspace_key_for(workspace_path),
                workspace_path=str(workspace_path),
                origin=TraceOrigin.SCHEDULE,
                purpose=TracePurpose.SCHEDULE_RUN,
                owner_session_id=session.session_id,
                subject_session_id=session.session_id,
                run_id=schedule_run_id,
                schedule_run_id=schedule_run_id,
            ),
        ):
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
            initial_messages = run_context.controller.initial_messages()
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

        first_request_prepared = False

        def notify_schedule_prepared() -> None:
            nonlocal first_request_prepared
            first_request_prepared = True

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
                on_first_request_prepared=notify_schedule_prepared,
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
        except ConfigError as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            if first_request_prepared:
                raise
            self._commit_schedule_failure(
                session,
                run_context,
                current_user,
                failure.error,
                job,
            )
        except asyncio.CancelledError:
            if not self._aborted and not first_request_prepared:
                self._record_schedule_failure(
                    session,
                    run_context,
                    current_user,
                    ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                    job,
                )
            raise
        except Exception as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            if first_request_prepared:
                raise
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
            session_model_configuration=session_model_configuration,
        )
        controller = ContextController(
            snapshot=AgentRunContextSnapshot.from_session(session),
            append_summary=self._memory_manager.append_summary,
            now=self._now,
            request_router=run_router,
            requested_route=route,
            project_messages=project_messages,
            project_tool_results=self._context_builder.project_tool_results,
            current_user=current_user,
            compact_ratio=self._configuration.runtime.compact_ratio,
            enable_tool_micro_compression=(
                self._configuration.runtime.enable_tool_micro_compression
            ),
        )
        return _AgentRunContext(
            controller=controller,
            runner=AgentRunner(run_router, controller),
        )

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
        persist_current_user: bool = True,
    ) -> None:
        values = context.controller.terminal_commit_values()
        combined_usage = _merge_usage_deltas(values.usage_delta, usage_delta)
        pending_last_compacted = values.pending_last_compacted
        if not persist_current_user and pending_last_compacted > len(session.messages):
            pending_last_compacted -= 1
        session.commit_agent_run(
            messages,
            pending_last_compacted=pending_last_compacted,
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
        schedule_occurrence_id: UUID | None = None
        if active_session._storage_partition is SessionStoragePartition.SCHEDULE:
            occurrence_value = inbound.metadata.get("schedule_occurrence_id")
            if isinstance(occurrence_value, str):
                try:
                    candidate_occurrence_id = UUID(occurrence_value)
                except ValueError as error:
                    execution_ready.set()
                    raise ValueError("Schedule input has an invalid occurrence identity") from error
                if candidate_occurrence_id.version != 4 or str(candidate_occurrence_id) != occurrence_value:
                    execution_ready.set()
                    raise ValueError("Schedule input has an invalid occurrence identity")
                schedule_occurrence_id = candidate_occurrence_id
        automatic_continuation = (
            inbound.metadata.get("foreground_automatic_continuation") is True
        )
        if automatic_continuation:
            token_value = inbound.metadata.get("foreground_effect_run_token")
            try:
                restore_run_token = UUID(token_value) if isinstance(token_value, str) else None
            except ValueError:
                restore_run_token = None
            if restore_run_token is None or restore_run_token.version != 4:
                execution_ready.set()
                raise ValueError("automatic continuation is missing a valid Task Run token")
            if schedule_occurrence_id is not None:
                if restore_run_token != schedule_occurrence_id:
                    execution_ready.set()
                    raise ValueError("Schedule continuation does not match its occurrence")
                restore_run_token = None
            restore_before = None
        elif schedule_occurrence_id is not None:
            restore_before = None
            restore_run_token = None
        else:
            restore_before = active_session.capture_restore_before()
            restore_run_token = self._new_uuid()
        run_id = inbound.metadata.get("run_id")
        if not isinstance(run_id, str):
            run_id = None
        workspace_path = self._workspace_state.workspace_path
        previous_suppression = self._schedule_outbound_suppressed
        if schedule_occurrence_id is not None:
            self._schedule_outbound_suppressed = True
        try:
            with trace_scope(
                self._trace_recorder,
                TraceRootSource(
                    workspace_key=workspace_key_for(workspace_path),
                    workspace_path=str(workspace_path),
                    origin=(
                        TraceOrigin.SCHEDULE
                        if schedule_occurrence_id is not None
                        else TraceOrigin.FOREGROUND
                    ),
                    purpose=(
                        TracePurpose.SCHEDULE_RUN
                        if schedule_occurrence_id is not None
                        else TracePurpose.AGENT_RUN
                    ),
                    owner_session_id=active_session.session_id,
                    subject_session_id=active_session.session_id,
                    run_id=run_id,
                    restore_run_token=(
                        None if restore_run_token is None else str(restore_run_token)
                    ),
                    schedule_run_id=(
                        None
                        if schedule_occurrence_id is None
                        else str(schedule_occurrence_id)
                    ),
                ),
            ):
                await self._execute_foreground_scoped(
                    inbound,
                    execution_ready=execution_ready,
                    restore_before=restore_before,
                    restore_run_token=restore_run_token,
                )
        finally:
            self._schedule_outbound_suppressed = previous_suppression

    async def _execute_foreground_scoped(
        self,
        inbound: InboundMessage,
        *,
        execution_ready: asyncio.Event,
        restore_before: SessionRestoreBefore | None,
        restore_run_token: UUID | None,
    ) -> None:
        active_session = self._session
        schedule_occurrence_value = inbound.metadata.get("schedule_occurrence_id")
        schedule_occurrence_id = (
            schedule_occurrence_value
            if active_session._storage_partition is SessionStoragePartition.SCHEDULE
            and isinstance(schedule_occurrence_value, str)
            else None
        )
        scheduled_run = schedule_occurrence_id is not None
        if self._captured_foreground is None:
            permission_snapshot = self._permission_control.snapshot(self._exec_host.resolved_shell)
            session_model_configuration = active_session.model_configuration
        else:
            permission_snapshot, session_model_configuration = self._captured_foreground
        captured_schedule_permission = inbound.metadata.get("schedule_permission_snapshot")
        if isinstance(captured_schedule_permission, PermissionSnapshot):
            permission_snapshot = captured_schedule_permission
        if scheduled_run:
            session_model_configuration = None
        self._run_model_configuration = session_model_configuration
        skill_state = () if scheduled_run else self._skill_loader.skills
        manual_invocation = (
            None if scheduled_run else self._skill_loader.resolve_manual(inbound.content)
        )
        start_title = not active_session.messages and not scheduled_run

        async def resolve_title(content: str) -> tuple[str, dict[str, int] | None]:
            parent = current_trace_scope()
            if parent is None:
                return await self._resolve_title(content)
            context = parent.context
            with trace_scope(
                self._trace_recorder,
                TraceRootSource(
                    workspace_key=context.workspace_key,
                    workspace_path=context.workspace_path,
                    origin=TraceOrigin.AUXILIARY,
                    purpose=TracePurpose.TITLE_GENERATION,
                    owner_session_id=active_session.session_id,
                    subject_session_id=active_session.session_id,
                    run_id=context.run_id,
                    restore_run_token=context.restore_run_token,
                    parent_root_id=context.root_id,
                ),
            ):
                return await self._resolve_title(content)

        created_title_work = (
            self._session_run_state.start_title(
                active_session, inbound.content, resolve_title, lambda: self._aborted
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
        schedule_occurrence_value = inbound.metadata.get("schedule_occurrence_id")
        scheduled_run = (
            active_session._storage_partition is SessionStoragePartition.SCHEDULE
            and isinstance(schedule_occurrence_value, str)
        )
        question_id = inbound.metadata.get("foreground_question_id")
        task_id = inbound.metadata.get("foreground_task_id")
        execution_id = inbound.metadata.get("run_id")
        explicit_continuation = inbound.metadata.get("foreground_task_continue") is True
        automatic_continuation = (
            inbound.metadata.get("foreground_automatic_continuation") is True
        )
        if automatic_continuation:
            current_user["_runtime_input_kind"] = (
                "worker_result"
                if inbound.metadata.get("foreground_worker_report") is True
                else "subagent_result"
            )
        runtime_subagent_result = inbound.metadata.get("foreground_subagent_result")
        if isinstance(runtime_subagent_result, str):
            current_user["_runtime_subagent_result"] = runtime_subagent_result
        delivery_receipts: list[tuple[str, int, int, str]] = []
        raw_delivery_receipts = inbound.metadata.get("foreground_delivery_receipts")
        if isinstance(raw_delivery_receipts, list):
            for item in raw_delivery_receipts:
                if not isinstance(item, dict):
                    continue
                agent_id = item.get("agent_id")
                record_revision = item.get("terminal_record_revision")
                generation = item.get("task_generation")
                receipt_id = item.get("receipt_id")
                if (
                    isinstance(agent_id, str)
                    and isinstance(record_revision, int)
                    and not isinstance(record_revision, bool)
                    and isinstance(generation, int)
                    and not isinstance(generation, bool)
                    and isinstance(receipt_id, str)
                ):
                    delivery_receipts.append((agent_id, record_revision, generation, receipt_id))
        if not delivery_receipts:
            delivery_receipt_id = inbound.metadata.get("foreground_delivery_receipt_id")
            delivery_agent_id = inbound.metadata.get("foreground_delivery_agent_id")
            delivery_record_revision = inbound.metadata.get(
                "foreground_delivery_record_revision"
            )
            delivery_generation = inbound.metadata.get("foreground_task_generation", 1)
            if (
                isinstance(delivery_receipt_id, str)
                and isinstance(delivery_agent_id, str)
                and isinstance(delivery_record_revision, int)
                and not isinstance(delivery_record_revision, bool)
                and isinstance(delivery_generation, int)
                and not isinstance(delivery_generation, bool)
            ):
                delivery_receipts.append(
                    (
                        delivery_agent_id,
                        delivery_record_revision,
                        delivery_generation,
                        delivery_receipt_id,
                    )
                )
        task_generation = 1
        if self._foreground_work_store is not None and inbound.content.strip():
            if not isinstance(execution_id, str):
                execution_id = str(self._new_uuid())
            if isinstance(question_id, str):
                if not isinstance(task_id, str):
                    raise ValueError("Foreground question resume is missing its Task ID")
                current_user["foreground_question_id"] = question_id
                current_user["foreground_task_id"] = task_id
                self._foreground_work_store.start_resume(question_id, execution_id)
                task = next(
                    (
                        item
                        for item in self._foreground_work_store.load().tasks
                        if item.task_id == task_id
                    ),
                    None,
                )
                if task is None:
                    raise ValueError("Foreground question Task is no longer available")
                task_generation = task.generation
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
            if isinstance(question_id, str) and current_user is not None:
                assert self._foreground_work_store is not None
                answered_question = self._foreground_work_store.load().question(question_id)
                if answered_question is not None:
                    current_user = {
                        **current_user,
                        "content": (
                            f"Human Answer to question {question_id}\n"
                            f"Question: {answered_question.text}\n"
                            f"Answer kind: {answered_question.answer_kind}\n"
                            + (
                                f"Choice ID: {answered_question.answer_option_id}\n"
                                if answered_question.answer_kind == "choice"
                                else ""
                            )
                            + f"Answer: {current_user['content']}"
                        ),
                    }

            def project_foreground(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
                return self._context_builder.build_foreground_messages(
                    messages,
                    session_id=active_session.session_id,
                    blackboard=staged_blackboard,
                    manual_invocation=manual_invocation,
                    summary="",
                    permission_snapshot=permission_snapshot,
                )

            def project_schedule(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
                return self._context_builder.build_schedule_messages(
                    messages,
                    session_id=active_session.session_id,
                    summary="",
                    permission_snapshot=permission_snapshot,
                )
            return self._context_builder.build_run_messages(
                history,
                current_user=current_user,
                increment=increment,
                compaction_cursor=compaction_cursor,
                action_summary=action_summary,
                project_messages=project_schedule if scheduled_run else project_foreground,
            )

        run_context = self._new_agent_run_context(
            active_session,
            current_user=deepcopy(current_user),
            route="schedule" if scheduled_run else "chat",
            project_messages=project_messages,
            session_model_configuration=(
                None if scheduled_run else session_model_configuration
            ),
        )
        framing_usage: dict[str, int] | None = None

        def metadata_patch() -> tuple[dict[str, Any] | None, tuple[str, ...]]:
            if scheduled_run:
                return None, ()
            if manual_invocation is not None or explicit_continuation or automatic_continuation:
                return None, ()
            if staged_blackboard is None:
                return None, ("blackboard",)
            return {"blackboard": staged_blackboard.to_dict()}, ()

        if (
            not scheduled_run
            and manual_invocation is None
            and not explicit_continuation
            and not automatic_continuation
        ):
            previous_blackboard = Blackboard.from_dict(active_session.metadata.get("blackboard"))
            last_assistant_content = _latest_assistant_content(active_session)
            try:
                with trace_purpose(TracePurpose.TASK_FRAMING):
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
            staged_blackboard = (
                Blackboard.from_dict(active_session.metadata.get("blackboard"))
                if automatic_continuation
                else None
            )
            framing_usage = None
            framing_result = None

        foreground_work_store = self._foreground_work_store
        if foreground_work_store is not None and inbound.content.strip() and not isinstance(
            question_id, str
        ):
            assert isinstance(execution_id, str)
            anchor_id = next(
                (
                    message.get("restore_anchor_id")
                    for message in reversed(active_session.messages)
                    if type(message.get("restore_anchor_id")) is int
                ),
                None,
            )
            if automatic_continuation:
                generation_value = inbound.metadata.get("foreground_task_generation")
                if (
                    not isinstance(task_id, str)
                    or isinstance(generation_value, bool)
                    or not isinstance(generation_value, int)
                ):
                    raise ValueError("Automatic SubAgent result is missing its Task generation")
                task_generation = generation_value
                foreground_work_store.start_execution(
                    task_id,
                    execution_id,
                    expected_generation=task_generation,
                    explicit_user_operation=False,
                )
            elif scheduled_run:
                occurrence_id = cast(str, schedule_occurrence_value)
                if not isinstance(task_id, str) or task_id != occurrence_id:
                    raise ValueError("Schedule Router Task does not match its occurrence")
                current_task = next(
                    (
                        item
                        for item in foreground_work_store.load().tasks
                        if item.task_id == task_id
                    ),
                    None,
                )
                if current_task is None:
                    foreground_work_store.begin_task(
                        task_id,
                        execution_id,
                        generation=1,
                        permission_level=permission_snapshot.level,
                        iteration_budget_limit=self._max_iterations,
                        execution_role="router",
                    )
                    task_generation = 1
                elif current_task.state == "completed":
                    execution_ready.set()
                    return False
                else:
                    task_generation = current_task.generation
                    foreground_work_store.start_execution(
                        task_id,
                        execution_id,
                        expected_generation=task_generation,
                        explicit_user_operation=False,
                    )
                current_user["foreground_task_id"] = task_id
            elif explicit_continuation:
                if not isinstance(task_id, str):
                    raise ValueError("Foreground Task continuation is missing its Task ID")
                current_task = next(
                    (
                        item
                        for item in foreground_work_store.load().tasks
                        if item.task_id == task_id
                    ),
                    None,
                )
                if current_task is None:
                    raise ValueError("Foreground Task is no longer available")
                expected_generation = inbound.metadata.get("foreground_task_generation")
                if (
                    expected_generation is not None
                    and (
                        isinstance(expected_generation, bool)
                        or not isinstance(expected_generation, int)
                        or expected_generation != current_task.generation
                    )
                ):
                    raise ValueError("Foreground Task generation changed before continuation")
                foreground_work_store.start_execution(
                    task_id,
                    execution_id,
                    expected_generation=current_task.generation,
                )
            elif manual_invocation is not None:
                task_id = str(self._new_uuid())
                foreground_work_store.begin_task(
                    task_id,
                    execution_id,
                    generation=1,
                    origin_anchor_id=anchor_id,
                    permission_level=permission_snapshot.level,
                    iteration_budget_limit=self._max_iterations,
                    execution_role="router",
                )
            elif framing_result is not None and framing_result.status == "resolved":
                if framing_result.action == "keep":
                    latest_task = next(reversed(foreground_work_store.load().tasks), None)
                    if latest_task is not None and latest_task.state not in {
                        "cancelled",
                        "superseded",
                    }:
                        task_id = latest_task.task_id
                        foreground_work_store.start_execution(
                            task_id,
                            execution_id,
                            expected_generation=latest_task.generation,
                        )
                    else:
                        task_id = str(self._new_uuid())
                        foreground_work_store.begin_task(
                            task_id,
                            execution_id,
                            generation=1,
                            origin_anchor_id=anchor_id,
                            permission_level=permission_snapshot.level,
                            iteration_budget_limit=self._max_iterations,
                            execution_role="router",
                        )
                elif framing_result.action in {"replace", "clear"}:
                    prior_task = next(reversed(foreground_work_store.load().tasks), None)
                    task_id = str(self._new_uuid())
                    foreground_work_store.replace_current_task(
                        task_id,
                        execution_id,
                        origin_anchor_id=anchor_id,
                        permission_level=permission_snapshot.level,
                        iteration_budget_limit=self._max_iterations,
                        execution_role="router",
                    )
                    if prior_task is not None and self._subagent_coordinator is not None:
                        await self._subagent_coordinator.cancel_task_and_wait(
                            prior_task.task_id, prior_task.generation
                        )
                    if (
                        prior_task is not None
                        and self._worker_generation_fencer is not None
                    ):
                        await self._worker_generation_fencer(
                            prior_task.task_id, prior_task.generation
                        )
                else:
                    task_id = str(self._new_uuid())
                    foreground_work_store.begin_task(
                        task_id,
                        execution_id,
                        generation=1,
                        origin_anchor_id=anchor_id,
                        permission_level=permission_snapshot.level,
                        iteration_budget_limit=self._max_iterations,
                        execution_role="router",
                    )
            else:
                foreground_work_store.interrupt_uncertain_tasks()
                task_id = str(self._new_uuid())
                foreground_work_store.begin_task(
                    task_id,
                    execution_id,
                    generation=1,
                    origin_anchor_id=anchor_id,
                    permission_level=permission_snapshot.level,
                    iteration_budget_limit=self._max_iterations,
                    execution_role="router",
                )
        router_restricted = False
        if foreground_work_store is not None and isinstance(task_id, str):
            task = next(
                (
                    item for item in foreground_work_store.load().tasks
                    if item.task_id == task_id
                ),
                None,
            )
            if task is None:
                raise ValueError("Foreground Task registration was not persisted")
            task_generation = task.generation
            router_restricted = task.execution_role == "router"
            permission_level = min(
                (task.permission_level, permission_snapshot.level),
                key=PERMISSION_LEVELS.index,
            )
            permission_snapshot = PermissionSnapshot(
                level=permission_level,
                exec_shell=permission_snapshot.exec_shell,
            )
            current_user["foreground_task_id"] = task_id

        subagents_eligible = (
            self._subagent_coordinator is not None
            and restore_run_token is not None
            and not router_restricted
        )
        foreground_tools: tuple[BaseTool, ...] = ()
        foreground_context: ToolRunContext | None = None
        worker_report_receipts: list[tuple[str, str, str, str]] = []
        inbound_worker_receipts = inbound.metadata.get("worker_report_receipts")
        if isinstance(inbound_worker_receipts, list):
            for item in inbound_worker_receipts:
                if not isinstance(item, dict):
                    continue
                worker_id = item.get("worker_id")
                report_id = item.get("report_id")
                receipt_id = item.get("receipt_id")
                if all(isinstance(value, str) for value in (worker_id, report_id, receipt_id)):
                    worker_report_receipts.append(
                        ("", cast(str, worker_id), cast(str, report_id), cast(str, receipt_id))
                    )
        if foreground_work_store is not None and task_id is not None:
            assert isinstance(execution_id, str)
            executor_reference = weakref.ref(self)

            def register_question(
                question: str,
                options: Sequence[Mapping[str, str]],
                tool_call_id: str,
            ) -> str:
                question_id = str(self._new_uuid())
                foreground_work_store.prepare_question(
                    task_id=cast(str, task_id),
                    execution_id=execution_id,
                    question_id=question_id,
                    tool_call_id=tool_call_id,
                    text=question,
                    options=[{"id": option["id"], "label": option["label"]} for option in options],
                )
                return question_id

            def suspend_for_subagent_attention(
                tool_name: str,
                arguments: Mapping[str, object],
                tool_call_id: str,
                content: str,
            ) -> AgentRunSuspension | None:
                if tool_name != "wait_agent":
                    return None
                try:
                    results = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    return None
                requested = arguments.get("agent_ids")
                if (
                    not isinstance(results, list)
                    or not isinstance(requested, list)
                    or any(not isinstance(agent_id, str) for agent_id in requested)
                ):
                    return None
                requested_ids = set(requested)
                for result in results:
                    if (
                        not isinstance(result, dict)
                        or result.get("status") != "waiting_for_answer"
                        or result.get("agent_id") not in requested_ids
                        or not isinstance(result.get("question_id"), str)
                    ):
                        continue
                    child_question = foreground_work_store.load().question(
                        result["question_id"]
                    )
                    if (
                        child_question is None
                        or child_question.state != "pending"
                        or child_question.owner_agent_id != result.get("agent_id")
                    ):
                        continue
                    foreground_work_store.suspend_for_child_question(
                        cast(str, task_id),
                        generation=task_generation,
                        execution_id=execution_id,
                    )
                    return AgentRunSuspension(
                        kind="subagent_attention",
                        task_id=cast(str, task_id),
                        generation=task_generation,
                        execution_id=execution_id,
                        question_id=result["question_id"],
                        tool_call_id=tool_call_id,
                    )
                return None

            def suspend_for_worker_reports(
                tool_name: str,
                arguments: Mapping[str, object],
                tool_call_id: str,
                content: str,
            ) -> AgentRunSuspension | None:
                del arguments, content
                agent_executor = executor_reference()
                if (
                    tool_name != "finish"
                    or not router_restricted
                    or agent_executor is None
                    or agent_executor._worker_scheduler is None
                    or agent_executor._worker_store is None
                ):
                    return None
                reports_in_input = {
                    (item.get("worker_id"), item.get("report_id"))
                    for item in (
                        inbound.metadata.get("worker_report_receipts", [])
                        if isinstance(inbound.metadata.get("worker_report_receipts", []), list)
                        else []
                    )
                    if isinstance(item, dict)
                }
                workers = tuple(
                    worker
                    for worker in agent_executor._worker_store.load().workers
                    if worker.parent_task_id == task_id
                    and worker.parent_generation == task_generation
                )
                if not workers or all(
                    worker.report is not None
                    and (
                        worker.report_delivery_state == "consumed"
                        or (worker.worker_id, worker.report.report_id) in reports_in_input
                    )
                    for worker in workers
                ):
                    return None
                foreground_work_store.suspend_for_worker_reports(
                    cast(str, task_id),
                    generation=task_generation,
                    execution_id=execution_id,
                )
                return AgentRunSuspension(
                    kind="worker_reports",
                    task_id=cast(str, task_id),
                    generation=task_generation,
                    execution_id=execution_id,
                    question_id=None,
                    tool_call_id=tool_call_id,
                )

            def suspend_foreground_run(
                tool_name: str,
                arguments: Mapping[str, object],
                tool_call_id: str,
                content: str,
            ) -> AgentRunSuspension | None:
                return suspend_for_subagent_attention(
                    tool_name, arguments, tool_call_id, content
                ) or suspend_for_worker_reports(tool_name, arguments, tool_call_id, content)

            async def dispatch_worker(
                worker_task_id: str,
                worker_task: str,
                explicit_context: Sequence[str],
                tool_call_id: str,
            ) -> str:
                agent_executor = executor_reference()
                if agent_executor is None:
                    raise ToolError("The foreground execution is no longer available.")
                worker_store = agent_executor._worker_store
                if not router_restricted or worker_store is None:
                    raise ToolError("Persistent Worker execution is unavailable.")
                parent_task_id = cast(str, task_id)
                parent_generation = task_generation
                if not foreground_work_store.is_current_execution(
                    parent_task_id, parent_generation, execution_id
                ):
                    raise ToolError("The foreground Task is no longer current.")
                try:
                    durable_session = Session.load(
                        agent_executor._workspace_state, active_session.session_id
                    )
                    worker_store.reconcile_deliveries(durable_session.messages)
                except FileNotFoundError:
                    worker_store.reconcile_deliveries(())
                except (OSError, UnicodeError, ValueError) as error:
                    raise ToolError("Worker report receipt state could not be recovered.") from error

                runtime_facts = {
                    "workspace": str(agent_executor._workspace_state.workspace_path),
                    "session_id": active_session.session_id,
                    "foreground_task_id": parent_task_id,
                    "foreground_task_generation": str(parent_generation),
                    "foreground_execution_id": execution_id,
                    "permission_level": permission_snapshot.level,
                    "exec_shell": permission_snapshot.exec_shell.selector,
                }
                worker_context = list(explicit_context)
                if manual_invocation is not None:
                    skill_metadata = json.dumps(
                        {
                            "name": manual_invocation.metadata.name,
                            "description": manual_invocation.metadata.description,
                            "path": str(manual_invocation.metadata.path),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    worker_context.append(
                        "Manual Skill metadata: "
                        + skill_metadata
                        + "\nManual Skill instructions:\n"
                        + manual_invocation.body
                        + "\nManual Skill request:\n"
                        + manual_invocation.request
                    )
                captured_worker_context = tuple(worker_context)
                parent_trace = current_trace_scope()
                trace_parent_root_id = (
                    None if parent_trace is None else parent_trace.context.root_id
                )
                initial_message = "Assigned task:\n" + worker_task
                if captured_worker_context:
                    initial_message += "\n\nExplicit context:\n" + "\n".join(
                        f"- {item}" for item in captured_worker_context
                    )
                system_prompt = (
                    "You are an Aide Worker. Perform only the assigned task. You do not have the "
                    "user's Conversation Session history. Treat the assigned task, explicit "
                    "context, and runtime facts below as the complete input. Use only the Tools "
                    "available in this run; Tool Gateway authorization applies to every call. "
                    "Return a concise factual report.\n\n"
                    "Captured authorization snapshot: "
                    f"permission_level={permission_snapshot.level}; "
                    f"exec_shell={permission_snapshot.exec_shell.selector}.\n"
                    "Runtime facts: "
                    + json.dumps(runtime_facts, ensure_ascii=True, separators=(",", ":"))
                )
                existing = worker_store.find_for_dispatch(
                    parent_task_id, parent_generation, worker_task_id
                )
                if existing is not None and (
                    existing.task != worker_task
                    or existing.explicit_context != captured_worker_context
                ):
                    raise ToolError("Worker dispatch identity was reused with different input.")

                if existing is None:
                    worker_model_configuration = session_model_configuration
                    if scheduled_run:
                        worker_model_configuration = next(
                            (item.model_configuration for item in worker_store.load().workers
                             if item.parent_task_id == parent_task_id
                             and item.parent_generation == parent_generation),
                            worker_model_configuration,
                        )
                    if worker_model_configuration is None:
                        route = RunModelRouter(agent_executor._model_router)
                        route_status = route.call_route_status(
                            "schedule" if scheduled_run else "chat",
                            continuation=None,
                        )
                        worker_model_configuration = SessionModelConfiguration(
                            route_status.provider_id,
                            route_status.model,
                            (
                                agent_executor._configuration.resolve_route("schedule").route.reasoning_effort
                                if scheduled_run
                                else agent_executor._model_router.reasoning_effort
                            ),
                        )
                    try:
                        worker = worker_store.create(
                            parent_task_id=parent_task_id,
                            parent_generation=parent_generation,
                            task_id=worker_task_id,
                            task=worker_task,
                            explicit_context=captured_worker_context,
                            runtime_facts=runtime_facts,
                            permission_snapshot=permission_snapshot,
                            session_model_configuration=worker_model_configuration,
                            system_prompt=system_prompt,
                            initial_message=initial_message,
                            restore_run_token=(
                                cast(str, schedule_occurrence_value)
                                if scheduled_run
                                else None
                                if restore_run_token is None
                                else str(restore_run_token)
                            ),
                            trace_parent_root_id=trace_parent_root_id,
                        )
                    except (OSError, UnicodeError, ValueError) as error:
                        raise ToolError("Worker assignment could not be persisted.") from error
                else:
                    worker = existing

                if worker.report is None:
                    executor = WorkerRunnerExecutor(
                        workspace_state=agent_executor._workspace_state,
                        store=worker_store,
                        model_router=cast(ModelRouter, agent_executor._model_router),
                        tool_gateway=agent_executor._tool_gateway,
                        exec_host=agent_executor._exec_host,
                        schedule_service=agent_executor._schedule_service,
                        compact_ratio=agent_executor._configuration.runtime.compact_ratio,
                        max_iterations=agent_executor._max_iterations,
                        max_tool_result_chars=agent_executor._configuration.runtime.max_tool_result_chars,
                        enable_tool_micro_compression=(
                            agent_executor._configuration.runtime.enable_tool_micro_compression
                        ),
                        mcp_keywords=agent_executor._mcp_keywords,
                        schedule_session=scheduled_run,
                        subagent_coordinator=agent_executor._subagent_coordinator,
                        trace_recorder=agent_executor._trace_recorder,
                        now=agent_executor._now,
                    )
                    requester = agent_executor._confirmation_requester
                    owner: ForegroundConfirmationOwner | BackgroundConfirmationOwner
                    if scheduled_run:
                        job_id = inbound.metadata.get("schedule_job_id")
                        if not isinstance(job_id, str):
                            job_id = active_session.session_id.removeprefix("schedule_")
                        owner = BackgroundConfirmationOwner(
                            generation_id=agent_executor._session_run_state.generation_id,
                            job_id=job_id,
                            occurrence_id=UUID(cast(str, schedule_occurrence_value)),
                        )
                    else:
                        owner = ForegroundConfirmationOwner(
                            generation_id=agent_executor._session_run_state.generation_id,
                            run_id=UUID(worker.execution_id),
                        )

                    async def request_worker_confirmation(
                        request: ConfirmationRequest,
                    ) -> ConfirmationDecision:
                        if requester is None:
                            raise ConfirmationUnavailable(
                                "foreground confirmation requester is not bound"
                            )
                        return await requester(
                            ConfirmationEnvelope(
                                request=request,
                                origin="background" if scheduled_run else "foreground",
                                owner=owner,
                                job_id=(
                                    owner.job_id
                                    if isinstance(owner, BackgroundConfirmationOwner)
                                    else None
                                ),
                                title=(
                                    inbound.metadata.get("schedule_title")
                                    if scheduled_run
                                    and isinstance(inbound.metadata.get("schedule_title"), str)
                                    else None
                                ),
                            )
                        )

                    if agent_executor._worker_scheduler is not None:
                        async def run_worker() -> None:
                            await executor.execute(
                                worker,
                                foreground_task_is_current=lambda parent_task, generation, _execution: (
                                    foreground_work_store.is_active_generation(
                                        parent_task, generation
                                    )
                                ),
                                confirmation=request_worker_confirmation,
                                cancel_requested=lambda: not foreground_work_store.is_active_generation(
                                    parent_task_id, parent_generation
                                ),
                                externalize_result=agent_executor._result_externalizer_for(
                                    active_session
                                ),
                            )

                        try:
                            agent_executor._worker_scheduler(worker, run_worker)
                        except (RuntimeError, ValueError) as error:
                            raise ToolError("Worker could not be scheduled for execution.") from error
                        return json.dumps(
                            {
                                "worker_id": worker.worker_id,
                                "task_id": worker.task_id,
                                "status": worker.state,
                            },
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )

                    try:
                        worker = await executor.execute(
                            worker,
                            foreground_task_is_current=lambda parent_task, generation, _execution: (
                                foreground_work_store.is_active_generation(
                                    parent_task, generation
                                )
                            ),
                            confirmation=request_worker_confirmation,
                            cancel_requested=lambda: not foreground_work_store.is_active_generation(
                                parent_task_id, parent_generation
                            ),
                            externalize_result=agent_executor._result_externalizer_for(active_session),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        raise ToolError("Worker execution did not produce a durable report.") from error

                if worker.report is None:
                    raise ToolError("Worker execution did not produce a durable report.")
                delivery = worker_store.claim_report(worker.worker_id, worker.report.report_id)
                if delivery is None:
                    return "Worker report was already consumed by the Router."
                worker_report_receipts.append(
                    (
                        tool_call_id,
                        worker.worker_id,
                        delivery.report.report_id,
                        delivery.receipt_id,
                    )
                )
                return json.dumps(
                    {
                        "worker_id": delivery.report.worker_id,
                        "report_id": delivery.report.report_id,
                        "receipt_id": delivery.receipt_id,
                        "sequence": delivery.report.sequence,
                        "status": delivery.report.status,
                        "summary": delivery.report.summary,
                        "error": (
                            None if delivery.report.error_code is None else {
                                "code": delivery.report.error_code,
                                "message": delivery.report.error_message,
                            }
                        ),
                        "usage": delivery.report.usage,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )

            foreground_tools = (RouterAskUserTool() if router_restricted else AskUserTool(),)
            if router_restricted:
                foreground_tools += (
                    FinishRouterTool(),
                    DispatchWorkerTool(
                        RouterDecisionValidator(
                            session_id=active_session.session_id,
                            workspace_id=str(self._workspace_state.workspace_path),
                        )
                    ),
                )
            foreground_context = ToolRunContext(
                workspace=self._workspace_state.workspace_path,
                schedule_service=self._schedule_service,
                exec_host=self._exec_host,
                foreground_question_handler=register_question,
                run_suspension_handler=suspend_foreground_run,
                task_id=task_id,
                generation=task_generation,
                execution_id=execution_id,
                foreground_task_is_current=foreground_work_store.is_current_execution,
                worker_dispatch_handler=dispatch_worker if router_restricted else None,
            )
        run_gateway = self._new_run_gateway(
            excluded_names=(
                () if subagents_eligible else ("spawn_agent", "wait_agent", "list_agents")
            ),
            permission_snapshot=None if scheduled_run else permission_snapshot,
            permission_context=(
                PermissionContext.from_snapshot(
                    permission_snapshot,
                    workspace_root=self._workspace_state.workspace_path,
                    origin="schedule",
                    configured_schedule_level=self._permission_control.configured(),
                )
                if scheduled_run
                else None
            ),
            tool_context=foreground_context,
            run_tools=foreground_tools,
            allowed_names=RouterCatalog().tool_names
            if router_restricted
            else None,
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
                    task_id=task_id,
                    task_generation=task_generation,
                ),
                parent_run_id=parent_run_id,
                permission_snapshot=permission_snapshot,
                session_model_configuration=session_model_configuration,
                system_prompt=self._context_builder.foreground_system_prompt(),
            )
        if (
            automatic_continuation
            and foreground_work_store is not None
            and isinstance(task_id, str)
            and not foreground_work_store.is_current_execution(
                task_id, task_generation, cast(str, execution_id)
            )
        ):
            return False
        try:
            initial_messages = run_context.controller.initial_messages()
        except asyncio.CancelledError:
            if not self._cancel_requested and not scheduled_run:
                raise
            cancelled_result = await self._finish_foreground_terminal(
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
            if scheduled_run and not self._cancel_requested:
                await active_session.wait_for_pending_persist()
                raise
            return cancelled_result
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

        wait_observations: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        externalize_result = self._result_externalizer_for(active_session)

        def externalize_foreground_result(value: ToolResult) -> ToolResult:
            final = externalize_result(value) if externalize_result is not None else value
            if value.name == "wait_agent" and value.status == "success" and final.status == "success":
                try:
                    items = json.loads(value.content)
                except json.JSONDecodeError:
                    items = None
                if isinstance(items, list):
                    observations = [
                        {key: item[key] for key in ("agent_id", "status", "terminal_record_revision")}
                        for item in items
                        if isinstance(item, dict)
                        and isinstance(item.get("agent_id"), str)
                        and item.get("status") in {"completed", "failed"}
                        and type(item.get("terminal_record_revision")) is int
                    ]
                    wait_observations[value.tool_call_id] = (final.content, observations)
            return final

        foreground_owner = ForegroundConfirmationOwner(
            generation_id=self._session_run_state.generation_id,
            run_id=self._new_uuid(),
        )
        self._active_foreground_owner = foreground_owner
        file_mutation_recorder = (
            None
            if scheduled_run
            else FileBackupStore(
                active_session.workspace_state,
                active_session.session_id,
            )
        )

        first_request_prepared = False

        def notify_title_prepared() -> None:
            nonlocal first_request_prepared
            first_request_prepared = True
            if title_work is not None and not title_work.coordination.prepared.done():
                title_work.coordination.prepared.set_result(True)

        try:
            result = await run_context.runner.run(
                initial_messages,
                model="schedule" if scheduled_run else "chat",
                tool_gateway=run_gateway,
                on_output=None if scheduled_run else self._publish_runner_output,
                confirmation=self._request_confirmation,
                externalize_result=externalize_foreground_result,
                cancel_requested=lambda: self._cancel_requested,
                max_iterations=self._max_iterations,
                remaining_iterations=(
                    min(self._max_iterations, foreground_work_store.remaining_iteration_budget(task_id, generation=task_generation))
                    if automatic_continuation and foreground_work_store is not None and isinstance(task_id, str)
                    else None
                ),
                file_mutation_recorder=file_mutation_recorder,
                run_token=restore_run_token,
                on_first_request_prepared=notify_title_prepared,
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
        except ConfigError:
            if first_request_prepared:
                raise
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
        except asyncio.CancelledError:
            if not self._cancel_requested and not scheduled_run:
                raise
            cancelled_result = await self._finish_foreground_terminal(
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
            if scheduled_run and not self._cancel_requested:
                await active_session.wait_for_pending_persist()
                raise
            return cancelled_result
        except Exception as failure:
            if first_request_prepared:
                raise
            _runtime_logger().error(
                "Agent Run preparation failed unexpectedly type={}",
                type(failure).__name__,
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
        if manual_invocation is not None or explicit_continuation or automatic_continuation:
            metadata_updates = None
            metadata_removals = ()
        elif staged_blackboard is None:
            metadata_updates = None
            metadata_removals = ("blackboard",)
        else:
            metadata_updates = {"blackboard": staged_blackboard.to_dict()}
            metadata_removals = ()

        run_messages = deepcopy(result.messages)
        failed_workers = False
        if scheduled_run and result.suspension is not None and result.suspension.kind == "human_question":
            run_messages.append(_build_assistant_repair_message(
                content="",
                status="error",
                error=ErrorInfo("interactive_terminal_required", "Schedule Router requires user input before it can continue."),
                model_calls=0,
            ))
        if scheduled_run and result.finish_reason == "completed" and result.suspension is None:
            failed_workers = (
                self._worker_store is not None
                and any(
                    worker.parent_task_id == task_id
                    and worker.report is not None
                    and worker.report.status != "ready"
                    for worker in self._worker_store.load().workers
                )
            )
            if failed_workers:
                worker_failure = ErrorInfo("model_failed", "One or more Schedule Workers failed.")
                result = replace(result, finish_reason="failed", error=worker_failure, final_content="")
                terminal = next(
                    (message for message in reversed(run_messages) if message.get("role") == "assistant"),
                    None,
                )
                if terminal is not None:
                    terminal.pop("router_finish_call_id", None)
                    terminal.update(status="error", content="", error={"code": worker_failure.code, "message": worker_failure.message})
        if scheduled_run:
            for message in run_messages:
                if message.get("role") == "assistant" and message.get("status") == "error":
                    message["schedule_occurrence_id"] = schedule_occurrence_value
        for message in run_messages:
            observation = wait_observations.get(message.get("tool_call_id", ""))
            if (message.get("role") == "tool" and message.get("name") == "wait_agent"
                    and observation is not None and message.get("content") == observation[0]):
                message["foreground_wait_results"] = observation[1]
            if message.get("role") == "tool" and message.get("name") == "dispatch_worker":
                tool_call_id = message.get("tool_call_id")
                worker_receipt_metadata = [
                    {
                        "worker_id": worker_id,
                        "report_id": report_id,
                        "receipt_id": receipt_id,
                    }
                    for receipt_tool_call_id, worker_id, report_id, receipt_id
                    in worker_report_receipts
                    if receipt_tool_call_id == tool_call_id
                ]
                if worker_receipt_metadata:
                    message["worker_report_receipts"] = worker_receipt_metadata
        if result.finish_reason == "completed" or result.suspension is not None or failed_workers:
            automatic_worker_receipts = [
                {
                    "worker_id": worker_id,
                    "report_id": report_id,
                    "receipt_id": receipt_id,
                }
                for tool_call_id, worker_id, report_id, receipt_id in worker_report_receipts
                if not tool_call_id
            ]
            if automatic_worker_receipts:
                assistant_message = next(
                    (
                        message
                        for message in reversed(run_messages)
                        if message.get("role") == "assistant"
                    ),
                    None,
                )
                if assistant_message is not None:
                    assistant_message["worker_report_receipts"] = automatic_worker_receipts
        if result.finish_reason == "cancelled" and not any(
            message.get("status") == "interrupted" for message in run_messages
        ):
            run_messages.append(_build_assistant_repair_message(
                content=(result.error.message if result.error else TURN_CANCELLED_MESSAGE),
                status="interrupted",
                error=result.error or ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
                model_calls=0,
            ))
        if result.finish_reason == "completed" and delivery_receipts:
            assistant_message = next(
                (message for message in reversed(run_messages) if message.get("role") == "assistant"),
                None,
            )
            if assistant_message is not None:
                receipts = assistant_message.get("foreground_delivery_receipts")
                receipt_list = [item for item in receipts if isinstance(item, str)] if isinstance(receipts, list) else []
                for _agent_id, _record_revision, _generation, receipt_id in delivery_receipts:
                    if receipt_id not in receipt_list:
                        receipt_list.append(receipt_id)
                assistant_message["foreground_delivery_receipts"] = receipt_list
        if (
            self._foreground_work_store is not None
            and result.finish_reason == "completed"
            and result.suspension is None
            and (framing_result is None or framing_result.status == "resolved")
            and isinstance(task_id, str)
            and type(task_generation) is int
            and isinstance(execution_id, str)
            and task is not None
        ):
            assistant_message = next(
                (message for message in reversed(run_messages) if message.get("role") == "assistant"),
                None,
            )
            if assistant_message is not None:
                assistant_message["foreground_execution_commit"] = {
                    "commit_id": f"{task_id}:{task_generation}:{execution_id}",
                    "task_id": task_id,
                    "generation": task_generation,
                    "execution_id": execution_id,
                    "model_calls": result.usage["model_calls"],
                    "iteration_stage": task.iteration_stage,
                }
        durable_current_user = deepcopy(current_user)
        durable_current_user.pop("_runtime_input_kind", None)
        durable_current_user.pop("_runtime_subagent_result", None)
        committed_messages = (
            run_messages
            if automatic_continuation
            else [durable_current_user, *run_messages]
        )
        try:
            if self._aborted:
                return False
            self._commit_agent_run(
                active_session,
                run_context,
                committed_messages,
                usage_delta=framing_usage,
                metadata_updates=metadata_updates,
                metadata_removals=metadata_removals,
                restore_before=None if automatic_continuation else restore_before,
                restore_run_token=None if automatic_continuation else restore_run_token,
                persist_current_user=not automatic_continuation,
            )
        except (OSError, UnicodeError) as failure:
            _runtime_logger().error(
                "Agent Run Session increment failed code=persistence_error type={}",
                type(failure).__name__,
            )
            await self._publish_commit_failure()
            if self._foreground_work_store is not None:
                try:
                    persisted = Session.load(self._workspace_state, active_session.session_id)
                    self._foreground_work_store.reconcile(persisted.messages)
                except (OSError, UnicodeError, ValueError):
                    pass
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
        # Session.commit_agent_run queues a best-effort save. Delivery acknowledgement
        # requires reading the durable increment, never just the resident history.
        if self._foreground_work_store is not None and (
            automatic_continuation or wait_observations or worker_report_receipts
        ):
            await active_session.wait_for_pending_persist()
            try:
                persisted = Session.load(self._workspace_state, active_session.session_id)
                if persisted.messages != active_session.messages:
                    raise OSError("Foreground result increment is not durable")
            except (OSError, UnicodeError, ValueError):
                self._foreground_work_store.interrupt_uncertain_tasks()
                await self._publish_commit_failure()
                return False
        if worker_report_receipts and self._worker_store is not None:
            try:
                persisted = Session.load(self._workspace_state, active_session.session_id)
                if persisted.messages != active_session.messages:
                    raise OSError("Worker report receipt is not durable")
                self._worker_store.reconcile_deliveries(persisted.messages)
            except (OSError, UnicodeError, ValueError) as failure:
                _runtime_logger().error(
                    "Worker report delivery receipt could not be indexed type={}",
                    type(failure).__name__,
                )
                await self._publish_commit_failure()
                return False
        if self._foreground_work_store is not None and isinstance(task_id, str):
            if result.finish_reason == "completed" and isinstance(execution_id, str):
                for agent_id, record_revision, generation, receipt_id in delivery_receipts:
                    try:
                        self._foreground_work_store.commit_delivery(
                            agent_id=agent_id,
                            terminal_record_revision=record_revision,
                            task_generation=generation,
                            execution_id=execution_id,
                            receipt_id=receipt_id,
                        )
                    except (OSError, UnicodeError, ValueError) as failure:
                        _runtime_logger().error(
                            "Foreground SubAgent delivery receipt could not be indexed type={}",
                            type(failure).__name__,
                        )
            if isinstance(task_generation, int) and isinstance(execution_id, str):
                try:
                    self._foreground_work_store.record_iteration_usage(
                        task_id,
                        generation=task_generation,
                        execution_id=execution_id,
                        model_calls=result.usage["model_calls"],
                    )
                except (OSError, UnicodeError, ValueError) as failure:
                    _runtime_logger().error(
                        "Foreground Task iteration usage could not be persisted type={}",
                        type(failure).__name__,
                    )
                    self._foreground_work_store.interrupt_uncertain_tasks()
                    await self._publish_commit_failure()
                    return False
            if result.suspension is not None or isinstance(question_id, str):
                try:
                    await active_session.wait_for_pending_persist()
                    persisted_session = Session.load(
                        self._workspace_state,
                        active_session.session_id,
                        partition=active_session._storage_partition,
                        now=self._now,
                    )
                    work_snapshot = self._foreground_work_store.reconcile(
                        persisted_session.messages
                    )
                except (OSError, UnicodeError, ValueError):
                    work_snapshot = self._foreground_work_store.load()
                    persisted_session = None
                if result.suspension is not None:
                    if result.suspension.kind == "worker_reports":
                        task = next(
                            (
                                item
                                for item in work_snapshot.tasks
                                if item.task_id == result.suspension.task_id
                                and item.generation == result.suspension.generation
                            ),
                            None,
                        )
                        valid_boundary = (
                            task is not None and task.state == "waiting_for_workers"
                        )
                    else:
                        question = (
                            None
                            if result.suspension.question_id is None
                            else work_snapshot.question(result.suspension.question_id)
                        )
                        valid_boundary = question is not None and question.state == "pending"
                    if persisted_session is None or not valid_boundary:
                        result = replace(
                            result,
                            finish_reason="failed",
                            final_content="",
                            error=ErrorInfo(
                                "persistence_error",
                                "The foreground wait could not be saved safely.",
                            ),
                            suspension=None,
                        )
                elif isinstance(question_id, str) and (
                    persisted_session is None
                    or not any(
                        message.get("foreground_question_id") == question_id
                        and message.get("role") == "user"
                        for message in persisted_session.messages
                    )
                ):
                    result = replace(
                        result,
                        finish_reason="failed",
                        final_content="",
                        error=ErrorInfo(
                            "persistence_error",
                            "The answer could not be saved to the Session.",
                        ),
                        suspension=None,
                    )
            if (
                result.finish_reason == "completed"
                and result.suspension is None
                and isinstance(execution_id, str)
            ):
                self._foreground_work_store.complete_task(task_id, execution_id)
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
        runtime_only = current_user.get("_runtime_input_kind") in {
            "subagent_result",
            "worker_result",
        }
        durable_current_user = deepcopy(current_user)
        durable_current_user.pop("_runtime_input_kind", None)
        durable_current_user.pop("_runtime_subagent_result", None)
        terminal_messages = [
            _build_assistant_repair_message(
                content=(TURN_CANCELLED_MESSAGE if error.code == "turn_cancelled" else ""),
                status="interrupted" if error.code == "turn_cancelled" else "error",
                error=error,
                model_calls=0,
            )
        ]
        if not runtime_only:
            terminal_messages.insert(0, durable_current_user)
        try:
            if self._aborted:
                return False
            self._commit_agent_run(
                active_session,
                context,
                terminal_messages,
                usage_delta=framing_usage,
                metadata_updates=metadata_updates,
                metadata_removals=metadata_removals,
                restore_before=None if runtime_only else restore_before,
                restore_run_token=None if runtime_only else restore_run_token,
                persist_current_user=not runtime_only,
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
        if self._aborted or self._schedule_outbound_suppressed:
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
        if self._aborted or self._schedule_outbound_suppressed:
            return
        if result.finish_reason == "suspended" and result.suspension is not None:
            payload: dict[str, object] = {
                "finish_reason": "suspended",
                "task_id": result.suspension.task_id,
                "execution_id": result.suspension.execution_id,
                "_streamed": True,
            }
            if result.suspension.question_id is not None:
                payload["question_id"] = result.suspension.question_id
            await self._bus.put_outbound(
                OutboundMessage(
                    "system_control",
                    "",
                    payload,
                )
            )
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
        if self._aborted or self._schedule_outbound_suppressed:
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
        if self._aborted or self._schedule_outbound_suppressed:
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
    request_messages = [
        message
        for message in messages
        if message.get("role") == "assistant"
        and message.get("status") == "completed"
        and message["token_usage"]["model_calls"] == 1
    ]
    last_request_usage = next(
        (
            {
                "input_tokens": message["token_usage"]["input_tokens"],
                "cached_input_tokens": message.get("cached_input_tokens"),
            }
            for message in reversed(request_messages)
        ),
        None,
    )
    cached_tokens = [
        message["cached_input_tokens"]
        for message in request_messages
        if "cached_input_tokens" in message
    ]
    historical_usage: dict[str, int | None] = {
        "input_tokens": sum(message["token_usage"]["input_tokens"] for message in request_messages),
        "output_tokens": sum(message["token_usage"]["output_tokens"] for message in request_messages),
        "cached_input_tokens": (
            sum(cached_tokens) if len(cached_tokens) == len(request_messages) else None
        ),
    }
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
        historical_usage=historical_usage,
    )


__all__ = [
    "AgentRunExecutor",
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
    result = empty_token_usage()
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
