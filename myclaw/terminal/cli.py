"""Command-line entry point for MyClaw."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from uuid import uuid4

import typer
from rich.console import Console
from tzlocal import get_localzone_name

from myclaw.agent.confirmation import ToolConfirmationCoordinator
from myclaw.agent.loop import AgentLoop, ModelContextOverflowError
from myclaw.agent.memory.dream import Dream
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.permission import PermissionSnapshot, RuntimePermissionControl
from myclaw.agent.session.restore import (
    RestoreError,
    RestoreManager,
    RestoreMode,
    RestorePlan,
    RestoreResult,
    StaleRestorePlan,
)
from myclaw.agent.tools.core.exec_host import (
    EXEC_CAPABILITY_ERROR,
    create_exec_host,
    resolve_exec_shell,
)
from myclaw.agent.tools.mcp_keywords import MCPKeywordPreparer
from myclaw.agent.tools.mcp_runtime import (
    MCPRuntimeManager,
    MCPServerFailure,
    MCPSnapshotReport,
    MCPStartupReport,
    MCPToolSnapshot,
)
from myclaw.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES
from myclaw.agent.workspace_state import (
    WorkspaceState,
    WorkspaceStateError,
    normalize_workspace_path,
)
from myclaw.agent.workspace_state import WorkspaceState as RuntimeWorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader, UserConfiguration
from myclaw.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    FatalManagementError,
    ManagementError,
    ManagementViewService,
    RestoreListingReport,
)
from myclaw.provider.factory import create_provider
from myclaw.provider.model_router import ModelRouter
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleOccurrence, ScheduleService
from myclaw.terminal.conversation import (
    TerminalConversationApp,
    is_interactive_terminal,
)
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.scheduler import AsyncioSchedulerClock
from myclaw.utils.time import local_now

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()

_MODEL_CONTEXT_OVERFLOW_ERROR = ErrorInfo(
    "model_context_overflow",
    MODEL_CONTEXT_OVERFLOW_MESSAGE,
)
_WORKSPACE_STATE_INITIALIZATION_ERROR = ErrorInfo(
    "persistence_error",
    "Workspace State could not be initialized at the reserved path.",
)
_TARGET_SESSION_PREPARATION_ERROR = ErrorInfo(
    "persistence_error",
    "Conversation Session could not be prepared.",
)
_MCP_GENERATION_PREPARATION_ERROR = ErrorInfo(
    "persistence_error",
    "MCP Tool Generation could not be prepared.",
)
_RUNTIME_SESSION_REPLACEMENT_ERROR = ErrorInfo(
    "persistence_error",
    "Runtime Session replacement could not be completed.",
)
_RUNTIME_STARTUP_ERROR = ErrorInfo(
    "persistence_error",
    "MyClaw runtime could not be started.",
)
_RESTORE_STARTUP_ERROR = ErrorInfo(
    "persistence_error",
    "Workspace Restore could not be recovered.",
)
_RESTORE_ADMISSION_ERROR = ErrorInfo(
    "model_invalid_request",
    "Finish or cancel the active foreground run and clear queued input before restoring.",
)
_RESTORE_IN_PROGRESS_ERROR = ErrorInfo(
    "model_invalid_request",
    "Session Restore is waiting for confirmation.",
)
_SAFE_FATAL_MANAGEMENT_ERRORS = (
    _MODEL_CONTEXT_OVERFLOW_ERROR,
    _TARGET_SESSION_PREPARATION_ERROR,
    _RUNTIME_SESSION_REPLACEMENT_ERROR,
    _RESTORE_STARTUP_ERROR,
)


def _print_error_info(error: ErrorInfo) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _print_error(error: ErrorInfo, path: object) -> None:
    _print_error_info(error)
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


def _print_mcp_notice(message: str) -> None:
    console.print(message, markup=False, highlight=False, soft_wrap=True)


def _print_exec_notice(message: str) -> None:
    console.print(message, markup=False, highlight=False, soft_wrap=True)


def _print_permission_startup_notice() -> None:
    console.print(
        "Full-Access is enabled for this process. It cancels ordinary permission confirmation "
        "for valid File, Exec, Web Fetch, MCP, and User Schedule calls; it is not an OS sandbox. "
        "Validation and Tool errors still apply; hard and uncertain Exec checks remain enforced.",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _report_mcp_generation(
    report: MCPStartupReport | MCPSnapshotReport,
) -> None:
    """Present only safe MCP lifecycle metadata to the terminal."""
    failures_by_server = {failure.mcp_name: failure for failure in report.failures}
    for mcp_name in report.failed_servers:
        failures_by_server.setdefault(
            mcp_name,
            MCPServerFailure(
                mcp_name=mcp_name,
                phase="connect",
                exception_type="MCPConnectionError",
            ),
        )
    for failure in sorted(failures_by_server.values(), key=lambda item: item.mcp_name):
        _print_mcp_notice(
            f"MCP Server {failure.mcp_name!r} unavailable during {failure.phase} "
            f"({failure.exception_type})."
        )

    for mcp_name, count in report.skipped_tool_counts:
        if count < 1:
            continue
        noun = "Tool" if count == 1 else "Tools"
        _print_mcp_notice(f"MCP Server {mcp_name!r} skipped {count} invalid MCP {noun}.")


def _approved_error_info(
    error: Exception,
    *,
    approved: tuple[ErrorInfo, ...],
    fallback: ErrorInfo,
) -> ErrorInfo:
    """Return only an exact, approved safe value from a domain exception."""
    candidate = vars(error).get("error")
    if type(candidate) is ErrorInfo and candidate in approved:
        return candidate
    return fallback


def _fatal_target_preparation_error(error: Exception) -> FatalManagementError:
    """Map only an established safe domain error across the fatal boundary."""
    if isinstance(error, ModelContextOverflowError):
        return FatalManagementError(
            _approved_error_info(
                error,
                approved=(_MODEL_CONTEXT_OVERFLOW_ERROR,),
                fallback=_TARGET_SESSION_PREPARATION_ERROR,
            )
        )
    return FatalManagementError(_TARGET_SESSION_PREPARATION_ERROR)


async def _drain_schedule_confirmation_aborts(
    schedule_service: ScheduleService,
    *,
    generation_id: object | None = None,
) -> None:
    """Drain the optional lifecycle seam while keeping old test fakes usable."""
    drain = getattr(schedule_service, "drain_confirmation_aborts", None)
    if not callable(drain):
        return
    if generation_id is None:
        await drain()
    else:
        await drain(generation_id=generation_id)


async def _run_cli_conversation(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
) -> None:
    """Compose one Runtime Lifetime and run its Terminal Conversation."""
    workspace_state: WorkspaceState | None = None
    router: ModelRouter | None = None
    dream: Dream | None = None
    schedule_service: ScheduleService | None = None
    mcp_manager: MCPRuntimeManager | None = None
    active_loop: AgentLoop | None = None
    current_loop: AgentLoop | None = None
    bus: MessageBus | None = None
    management: ManagementViewService | None = None
    terminal_app: TerminalConversationApp | None = None
    pending_target: AgentLoop | None = None
    active_mcp_snapshot: MCPToolSnapshot = ()
    active_mcp_keywords: Mapping[str, tuple[str, ...]] = {}
    mcp_keyword_preparer: MCPKeywordPreparer | None = None
    replacement_lock = asyncio.Lock()
    aborted_loops: list[AgentLoop] = []
    replacement_failed_closed = False
    started = False
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    startup_restore_result: RestoreResult | None = None
    startup_session_id: str | None = None
    restore_manager: RestoreManager | None = None
    restore_plan: RestorePlan | None = None
    restore_barrier_loop: AgentLoop | None = None
    restore_barrier_held = False
    restore_schedule_paused = False
    restore_committing = False
    restore_inspection_task: asyncio.Task[object] | None = None
    restore_blocked = False
    latest_restore_result: RestoreResult | None = None
    permission_control = RuntimePermissionControl(
        getattr(configuration.runtime, "permission_level", "workspace-write")
    )
    confirmation_coordinator = ToolConfirmationCoordinator()
    if permission_control.configured() == "full-access":
        _print_permission_startup_notice()

    async def abort_loop_once(loop: AgentLoop) -> None:
        if any(loop is existing for existing in aborted_loops):
            return
        aborted_loops.append(loop)
        await loop.abort()

    try:
        workspace_path = normalize_workspace_path(workspace)
        resolved_exec_shell = resolve_exec_shell(
            getattr(configuration.runtime, "exec_shell", "auto")
        )
        exec_host = create_exec_host(resolved_exec_shell)
        if not resolved_exec_shell.available:
            _print_exec_notice(resolved_exec_shell.diagnostic or EXEC_CAPABILITY_ERROR)
        workspace_state = WorkspaceState(workspace_path)
        workspace_state.initialize(agent_home_root=agent_home.path)
        if isinstance(workspace_state, RuntimeWorkspaceState):
            try:
                startup_restore_result = await RestoreManager(workspace_state).recover_pending()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise FatalManagementError(_RESTORE_STARTUP_ERROR) from error
            if startup_restore_result is not None:
                startup_session_id = startup_restore_result.session_id
                latest_restore_result = startup_restore_result

        mcp_manager = MCPRuntimeManager(
            workspace_path,
            built_in_names=BUILT_IN_TOOL_NAMES,
        )
        startup_report = await mcp_manager.start(configuration.mcp)
        active_mcp_snapshot = startup_report.snapshot
        _report_mcp_generation(startup_report)

        bus = MessageBus()
        router = ModelRouter(
            configuration=configuration,
            provider_factory=create_provider,
        )
        mcp_keyword_preparer = MCPKeywordPreparer(
            model_router=router,
            config_loader=ConfigLoader(agent_home),
        )
        active_mcp_keywords = await mcp_keyword_preparer.prepare(
            active_mcp_snapshot,
            configuration.mcp,
        )
        memory_manager = MemoryManager(workspace_state)
        dream = Dream(
            memory_manager=memory_manager,
            model_router=router,
            batch_size=configuration.memory.batch_size,
            memory_route_status=router.route_status("memory"),
        )

        async def execute_user_occurrence(occurrence: ScheduleOccurrence) -> None:
            if current_loop is None:
                raise RuntimeError("Schedule Service user executor is not bound")
            await current_loop.run_schedule_job(occurrence.job, occurrence)

        async def execute_user_job(job: ScheduleJob) -> None:
            if current_loop is None:
                raise RuntimeError("Schedule Service user executor is not bound")
            await current_loop.run_schedule_job(job)

        configured_schedule_level = permission_control.configured()

        def capture_schedule_permission_snapshot() -> PermissionSnapshot:
            return PermissionSnapshot(
                level=configured_schedule_level,
                exec_shell=exec_host.resolved_shell,
            )

        async def wait_for_session_persist(loop: AgentLoop, session_id: str) -> None:
            old_session = loop.session
            if old_session.session_id != session_id:
                return
            try:
                await old_session.wait_for_pending_persist()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise ManagementError(
                    ErrorInfo(
                        "persistence_error",
                        "Conversation Session could not be prepared.",
                    )
                ) from error

        async def prepare_session_resume(session_id: str) -> None:
            old_loop = current_loop
            if old_loop is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                )
            await wait_for_session_persist(old_loop, session_id)

        schedule_service = ScheduleService(
            workspace_state=workspace_state,
            clock=AsyncioSchedulerClock(now=local_now),
            execute_user_job=execute_user_job,
            execute_user_occurrence=execute_user_occurrence,
            permission_snapshot_factory=capture_schedule_permission_snapshot,
            cancel_confirmation_owner=getattr(
                confirmation_coordinator,
                "cancel_owner",
                None,
            ),
            execute_dream=dream.run,
            timezone_name=get_localzone_name(),
        )

        def create_agent_loop(
            session_id: str | None,
            *,
            mcp_snapshot: MCPToolSnapshot | None = None,
            mcp_keywords: Mapping[str, tuple[str, ...]] | None = None,
        ) -> AgentLoop:
            selected_mcp_snapshot = active_mcp_snapshot if mcp_snapshot is None else mcp_snapshot
            selected_mcp_keywords = active_mcp_keywords if mcp_keywords is None else mcp_keywords
            loop = AgentLoop(
                workspace_path=workspace_path,
                workspace_state=workspace_state,
                agent_home=agent_home,
                configuration=configuration,
                bus=bus,
                schedule_service=schedule_service,
                model_router=router,
                memory_manager=memory_manager,
                session_id=session_id,
                now=local_now,
                new_uuid=uuid4,
                monotonic_now=monotonic,
                mcp_tools=selected_mcp_snapshot,
                mcp_keywords=selected_mcp_keywords,
                exec_host=exec_host,
                permission_control=permission_control,
            )
            bind_confirmation_requester = getattr(loop, "bind_confirmation_requester", None)
            if callable(bind_confirmation_requester):
                bind_confirmation_requester(confirmation_coordinator.request)
            return loop

        def current_agent_loop() -> AgentLoop:
            if current_loop is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                )
            return current_loop

        async def replace_agent_loop(session_id: str, force: bool) -> None:
            nonlocal active_loop, active_mcp_snapshot, active_mcp_keywords
            nonlocal current_loop, pending_target
            nonlocal replacement_failed_closed
            async with replacement_lock:
                ensure_management_mutation_allowed()
                old_loop = current_loop
                if old_loop is None:
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                    )

                target: AgentLoop | None = None
                replacement_barrier_held = False
                destructive_started = False
                if mcp_manager is None:
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "MCP Runtime Manager is unavailable.")
                    )
                try:
                    candidate_report = await mcp_manager.prepare_generation()
                    if mcp_keyword_preparer is None:
                        raise RuntimeError("MCP Keyword Preparer is unavailable")
                    candidate_keywords = await mcp_keyword_preparer.prepare(
                        candidate_report.snapshot,
                        configuration.mcp,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise ManagementError(_MCP_GENERATION_PREPARATION_ERROR) from error
                _report_mcp_generation(candidate_report)

                async def release_replacement_barrier(*, resume_inbound: bool) -> None:
                    nonlocal replacement_barrier_held
                    if not replacement_barrier_held:
                        return
                    await old_loop._release_replacement_barrier(resume_inbound=resume_inbound)
                    replacement_barrier_held = False

                async def reject_prepared_target(target: AgentLoop) -> None:
                    nonlocal pending_target
                    try:
                        await abort_loop_once(target)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        raise ManagementError(
                            ErrorInfo(
                                "persistence_error",
                                "Conversation Session could not be prepared.",
                            )
                        ) from error
                    finally:
                        pending_target = None
                        await release_replacement_barrier(resume_inbound=True)

                try:
                    await old_loop._pause_for_replacement()
                    replacement_barrier_held = True
                    await wait_for_session_persist(old_loop, session_id)
                except asyncio.CancelledError:
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    raise
                except Exception as error:
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    raise ManagementError(
                        ErrorInfo(
                            "persistence_error",
                            "Conversation Session could not be prepared.",
                        )
                    ) from error

                try:
                    target = create_agent_loop(
                        session_id,
                        mcp_snapshot=candidate_report.snapshot,
                        mcp_keywords=candidate_keywords,
                    )
                    pending_target = target
                    target.preflight()
                except asyncio.CancelledError as cancellation:
                    cleanup_error: BaseException | None = None
                    if target is not None:
                        try:
                            await abort_loop_once(target)
                        except BaseException as caught:
                            cleanup_error = caught
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    if cleanup_error is not None:
                        raise cancellation from cleanup_error
                    raise
                except Exception as error:
                    target_cleanup_error: Exception | None = None
                    if target is not None:
                        try:
                            await abort_loop_once(target)
                        except asyncio.CancelledError:
                            raise
                        except Exception as caught:
                            target_cleanup_error = caught
                    pending_target = None
                    if target_cleanup_error is not None:
                        raise _fatal_target_preparation_error(error) from BaseExceptionGroup(
                            "Conversation Session preparation cleanup failed",
                            (error, target_cleanup_error),
                        )
                    raise _fatal_target_preparation_error(error) from error

                assert target is not None
                try:
                    has_active_run = old_loop.control.has_active_run
                except Exception as error:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                    ) from error
                if has_active_run and not force:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo(
                            "model_invalid_request",
                            "An active foreground run must be confirmed before switching Sessions.",
                        )
                    )

                if terminal_app is None or schedule_service is None or bus is None:
                    await reject_prepared_target(target)
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Session resume is unavailable.")
                    )

                destructive_started = True
                try:
                    generation_id = getattr(old_loop, "generation_id", None)
                    if generation_id is not None:
                        cancel_generation = getattr(
                            schedule_service,
                            "cancel_confirmation_generation",
                            None,
                        )
                        if callable(cancel_generation):
                            cancel_generation(generation_id)
                        await confirmation_coordinator.cancel_generation(generation_id)
                        await _drain_schedule_confirmation_aborts(
                            schedule_service,
                            generation_id=generation_id,
                        )
                    await terminal_app.quiesce_for_rebind()
                    await schedule_service.pause_and_drain()
                    current_loop = None
                    await abort_loop_once(old_loop)
                    await bus.reset()
                    await terminal_app.rebind_agent_loop(
                        control=target.control,
                        skill_metadata=target.skill_metadata,
                        session_projection=target.project_foreground_conversation(),
                    )
                    await target.start()
                    mcp_manager.activate_generation(candidate_report)
                    active_mcp_snapshot = candidate_report.snapshot
                    active_mcp_keywords = candidate_keywords
                    current_loop = target
                    active_loop = target
                    pending_target = None
                    await release_replacement_barrier(resume_inbound=True)
                    schedule_service.resume()
                except asyncio.CancelledError:
                    replacement_failed_closed = True
                    current_loop = None
                    if management is not None:
                        management.deactivate()
                    raise
                except BaseException as error:
                    replacement_failed_closed = True
                    current_loop = None
                    if management is not None:
                        management.deactivate()
                    raise FatalManagementError(_RUNTIME_SESSION_REPLACEMENT_ERROR) from error
                finally:
                    await release_replacement_barrier(resume_inbound=not destructive_started)

        def ensure_management_mutation_allowed() -> None:
            if restore_blocked:
                raise FatalManagementError(_RESTORE_STARTUP_ERROR)
            if restore_barrier_held or restore_committing:
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)

        async def release_restore_barriers() -> None:
            nonlocal restore_barrier_loop, restore_barrier_held
            nonlocal restore_schedule_paused, restore_manager, restore_plan
            nonlocal restore_committing
            loop = restore_barrier_loop
            release_error: BaseException | None = None
            if loop is not None and restore_barrier_held:
                try:
                    await loop._release_replacement_barrier(resume_inbound=True)
                except BaseException as error:
                    release_error = error
                finally:
                    restore_barrier_held = False
            if restore_schedule_paused and schedule_service is not None:
                try:
                    schedule_service.resume()
                except BaseException as error:
                    if release_error is None:
                        release_error = error
                finally:
                    restore_schedule_paused = False
            restore_barrier_loop = None
            restore_manager = None
            restore_plan = None
            restore_committing = False
            if release_error is not None:
                raise release_error

        async def restore_listing() -> RestoreListingReport:
            nonlocal restore_barrier_loop, restore_barrier_held
            barrier_acquired = False
            if restore_blocked:
                raise FatalManagementError(_RESTORE_STARTUP_ERROR)
            if restore_barrier_held or restore_committing:
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
            if bus is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Session Restore is unavailable.")
                )
            try:
                async with replacement_lock:
                    ensure_management_mutation_allowed()
                    loop = current_agent_loop()
                    if loop.control.has_active_run or await bus.inbound_snapshot():
                        raise ManagementError(_RESTORE_ADMISSION_ERROR)
                    await loop._pause_for_replacement()
                    restore_barrier_loop = loop
                    restore_barrier_held = True
                    barrier_acquired = True
                    if loop.control.has_active_run or await bus.inbound_snapshot():
                        raise ManagementError(_RESTORE_ADMISSION_ERROR)
                    anchors = loop.session.restore_candidates()
                    if not anchors:
                        await release_restore_barriers()
            except ManagementError:
                if barrier_acquired:
                    await release_restore_barriers()
                raise
            except asyncio.CancelledError:
                if barrier_acquired:
                    await release_restore_barriers()
                raise
            except (OSError, UnicodeError, ValueError) as error:
                if barrier_acquired:
                    await release_restore_barriers()
                raise ManagementError(
                    ErrorInfo("persistence_error", "Restore Anchors could not be listed.")
                ) from error
            return RestoreListingReport(
                session_id=loop.session.session_id,
                anchors=tuple(reversed(anchors)),
            )

        async def wait_for_restore_idle(loop: AgentLoop) -> None:
            wait = getattr(loop, "wait_for_restore_idle", None)
            if callable(wait):
                await wait()
                return
            await wait_for_session_persist(loop, loop.session.session_id)

        async def restore_inspect(anchor_id: int) -> RestorePlan:
            nonlocal restore_manager, restore_plan, restore_barrier_loop
            nonlocal restore_barrier_held, restore_schedule_paused
            nonlocal restore_inspection_task
            if restore_blocked:
                raise FatalManagementError(_RESTORE_STARTUP_ERROR)
            if (
                restore_committing
                or restore_plan is not None
                or restore_inspection_task is not None
                or not restore_barrier_held
            ):
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
            loop = current_agent_loop()
            if bus is None or schedule_service is None or workspace_state is None:
                raise ManagementError(
                    ErrorInfo("route_unavailable", "Session Restore is unavailable.")
                )
            inspection_task = asyncio.current_task()
            if inspection_task is None:
                raise RuntimeError("Session Restore inspection requires an asyncio Task")
            restore_inspection_task = inspection_task
            try:
                if loop.control.has_active_run or await bus.inbound_snapshot():
                    raise ManagementError(_RESTORE_ADMISSION_ERROR)
                if restore_barrier_loop is not loop:
                    raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
                if loop.control.has_active_run or await bus.inbound_snapshot():
                    raise ManagementError(_RESTORE_ADMISSION_ERROR)
                restore_schedule_paused = True
                await schedule_service.pause_and_wait_idle()
                await wait_for_restore_idle(loop)
                restore_manager = RestoreManager(
                    workspace_state,
                    loop.session.session_id,
                    now=local_now,
                )
                inspected = restore_manager.inspect(loop.session, anchor_id)
                restore_manager.revalidate(inspected)
                restore_plan = inspected
                return inspected
            except asyncio.CancelledError:
                await release_restore_barriers()
                raise
            except ManagementError:
                await release_restore_barriers()
                raise
            except StaleRestorePlan as error:
                await release_restore_barriers()
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                ) from error
            except (RestoreError, OSError, UnicodeError, ValueError) as error:
                await release_restore_barriers()
                raise ManagementError(
                    ErrorInfo("persistence_error", "Session Restore could not be inspected.")
                ) from error
            finally:
                if restore_inspection_task is inspection_task:
                    restore_inspection_task = None

        async def rebuild_after_restore(old_loop: AgentLoop, session_id: str) -> None:
            nonlocal active_loop, active_mcp_snapshot, active_mcp_keywords
            nonlocal current_loop, pending_target, replacement_failed_closed
            nonlocal restore_barrier_held, restore_schedule_paused
            target: AgentLoop | None = None
            try:
                if mcp_manager is None or mcp_keyword_preparer is None:
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
                    )
                if terminal_app is None or schedule_service is None or bus is None:
                    raise ManagementError(
                        ErrorInfo("route_unavailable", "Session Restore is unavailable.")
                    )
                candidate_report = await mcp_manager.prepare_generation()
                candidate_keywords = await mcp_keyword_preparer.prepare(
                    candidate_report.snapshot,
                    configuration.mcp,
                )
                _report_mcp_generation(candidate_report)
                target = create_agent_loop(
                    session_id,
                    mcp_snapshot=candidate_report.snapshot,
                    mcp_keywords=candidate_keywords,
                )
                pending_target = target
                target.preflight()
            except BaseException as error:
                replacement_failed_closed = True
                current_loop = None
                if management is not None:
                    management.deactivate()
                cleanup_errors: list[BaseException] = []
                if pending_target is not None:
                    try:
                        await abort_loop_once(pending_target)
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                    finally:
                        pending_target = None
                try:
                    await abort_loop_once(old_loop)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                if isinstance(error, asyncio.CancelledError):
                    if cleanup_errors:
                        raise error from BaseExceptionGroup(
                            "Restored Runtime Generation cleanup failed",
                            tuple(cleanup_errors),
                        )
                    raise
                if cleanup_errors:
                    raise FatalManagementError(
                        _RUNTIME_SESSION_REPLACEMENT_ERROR
                    ) from BaseExceptionGroup(
                        "Restored Runtime Generation cleanup failed",
                        (error, *cleanup_errors),
                    )
                raise FatalManagementError(_RUNTIME_SESSION_REPLACEMENT_ERROR) from error

            assert target is not None
            assert mcp_manager is not None
            assert terminal_app is not None
            assert schedule_service is not None
            assert bus is not None
            destructive_started = True
            try:
                generation_id = getattr(old_loop, "generation_id", None)
                if generation_id is not None:
                    cancel_generation = getattr(
                        schedule_service,
                        "cancel_confirmation_generation",
                        None,
                    )
                    if callable(cancel_generation):
                        cancel_generation(generation_id)
                    await confirmation_coordinator.cancel_generation(generation_id)
                    await _drain_schedule_confirmation_aborts(
                        schedule_service,
                        generation_id=generation_id,
                    )
                await terminal_app.quiesce_for_rebind()
                current_loop = None
                await abort_loop_once(old_loop)
                await bus.reset()
                await terminal_app.rebind_agent_loop(
                    control=target.control,
                    skill_metadata=target.skill_metadata,
                    session_projection=target.project_foreground_conversation(),
                )
                await target.start()
                mcp_manager.activate_generation(candidate_report)
                active_mcp_snapshot = candidate_report.snapshot
                active_mcp_keywords = candidate_keywords
                current_loop = target
                active_loop = target
                pending_target = None
                if restore_barrier_held:
                    await old_loop._release_replacement_barrier(resume_inbound=True)
                    restore_barrier_held = False
                if restore_schedule_paused:
                    schedule_service.resume()
                    restore_schedule_paused = False
            except asyncio.CancelledError:
                replacement_failed_closed = True
                current_loop = None
                if management is not None:
                    management.deactivate()
                raise
            except BaseException as error:
                replacement_failed_closed = True
                current_loop = None
                if management is not None:
                    management.deactivate()
                raise FatalManagementError(_RUNTIME_SESSION_REPLACEMENT_ERROR) from error
            finally:
                if not destructive_started and restore_barrier_held:
                    await old_loop._release_replacement_barrier(resume_inbound=True)
                    restore_barrier_held = False

        async def restore_commit(plan: RestorePlan, mode: RestoreMode | str) -> RestoreResult:
            nonlocal restore_committing, latest_restore_result, restore_manager, restore_plan
            nonlocal restore_barrier_loop, restore_blocked, replacement_failed_closed, current_loop
            if restore_blocked:
                raise FatalManagementError(_RESTORE_STARTUP_ERROR)
            if restore_manager is None or restore_plan is None or plan != restore_plan:
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                )
            if restore_committing:
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
            old_loop = restore_barrier_loop
            if old_loop is None or not restore_barrier_held:
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
            restore_committing = True
            try:
                async with replacement_lock:
                    restore_manager.revalidate(plan)
                    result = await restore_manager.execute(plan, mode)
                    await rebuild_after_restore(old_loop, plan.session_id)
                latest_restore_result = result
                restore_manager = None
                restore_plan = None
                restore_barrier_loop = None
                restore_committing = False
                return result
            except asyncio.CancelledError:
                restore_blocked = True
                replacement_failed_closed = True
                current_loop = None
                if management is not None:
                    management.deactivate()
                raise
            except StaleRestorePlan as error:
                await release_restore_barriers()
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                ) from error
            except (RestoreError, OSError, UnicodeError, ValueError) as error:
                pending_path = (
                    workspace_state.path / "restore" / plan.session_id / "pending.json"
                    if workspace_state is not None
                    else None
                )
                if pending_path is not None and pending_path.exists():
                    restore_blocked = True
                    replacement_failed_closed = True
                    current_loop = None
                    if management is not None:
                        management.deactivate()
                    raise FatalManagementError(_RESTORE_STARTUP_ERROR) from error
                await release_restore_barriers()
                raise ManagementError(
                    ErrorInfo("persistence_error", "Session Restore could not be completed.")
                ) from error

        async def restore_result() -> RestoreResult | None:
            return latest_restore_result

        async def restore_cancel() -> None:
            if restore_committing:
                raise ManagementError(_RESTORE_IN_PROGRESS_ERROR)
            inspection_task = restore_inspection_task
            if inspection_task is not None and inspection_task is not asyncio.current_task():
                inspection_task.cancel()
                try:
                    await await_task_preserving_cancellation(inspection_task)
                except asyncio.CancelledError:
                    pass
                return
            if restore_barrier_held:
                await release_restore_barriers()

        initial_loop = create_agent_loop(startup_session_id)
        active_loop = initial_loop
        initial_loop.preflight()
        schedule_service._prepare_start()
        await schedule_service.register_dream_job(
            schedule=JobSchedule.from_cron_input(
                configuration.memory.schedule,
                get_localzone_name(),
            )
        )
        current_loop = initial_loop

        management = ManagementViewService(
            agent_home,
            current_agent_loop=current_agent_loop,
            workspace_state=workspace_state,
            replace_agent_loop=replace_agent_loop,
            prepare_session_resume=prepare_session_resume,
            memory_manager=memory_manager,
            dream=dream,
            schedule_status=lambda: schedule_service.status_snapshot().to_dict(),
            now=local_now,
            monotonic=monotonic,
            reasoning_effort_control=router,
            permission_control=permission_control,
            restore_listing=restore_listing,
            restore_inspect=restore_inspect,
            restore_commit=restore_commit,
            restore_result=restore_result,
            restore_cancel=restore_cancel,
            ensure_management_mutation_allowed=ensure_management_mutation_allowed,
        )
        dispatcher = ManagementCommandDispatcher(management)
        terminal_app = TerminalConversationApp(
            bus=bus,
            control=initial_loop.control,
            management_dispatcher=dispatcher,
            skill_metadata=initial_loop.skill_metadata,
        )
        bind_confirmation_coordinator = getattr(
            terminal_app,
            "bind_confirmation_coordinator",
            None,
        )
        if callable(bind_confirmation_coordinator):
            bind_confirmation_coordinator(confirmation_coordinator)

        await initial_loop.start()
        schedule_service.start()
        started = True
        await terminal_app.run_async()
        fatal_management_error = getattr(terminal_app, "fatal_management_error", None)
        if isinstance(fatal_management_error, FatalManagementError):
            raise fatal_management_error
    except BaseException as error:
        primary_error = error
    finally:
        try:
            await confirmation_coordinator.close()
        except BaseException as error:
            cleanup_errors.append(error)

        if schedule_service is not None:
            try:
                await _drain_schedule_confirmation_aborts(schedule_service)
            except BaseException as error:
                cleanup_errors.append(error)

        if management is not None:
            try:
                management.deactivate()
            except BaseException as error:
                cleanup_errors.append(error)

        if schedule_service is not None:
            try:
                await schedule_service.pause_and_drain()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                await schedule_service.close()
            except BaseException as error:
                cleanup_errors.append(error)

        if pending_target is not None:
            try:
                await abort_loop_once(pending_target)
            except BaseException as error:
                cleanup_errors.append(error)

        if active_loop is not None:
            try:
                if started and active_loop is current_loop and not replacement_failed_closed:
                    await active_loop.close()
                else:
                    await abort_loop_once(active_loop)
            except BaseException as error:
                cleanup_errors.append(error)

        if mcp_manager is not None:
            try:
                await mcp_manager.close()
            except BaseException as error:
                cleanup_errors.append(error)

        if dream is not None:
            try:
                await dream.close()
            except BaseException as error:
                cleanup_errors.append(error)

        if router is not None:
            try:
                await router.close()
            except BaseException as error:
                cleanup_errors.append(error)

    if primary_error is not None:
        if cleanup_errors:
            cleanup = (
                cleanup_errors[0]
                if len(cleanup_errors) == 1
                else BaseExceptionGroup("CLI shutdown failed", cleanup_errors)
            )
            raise primary_error from cleanup
        raise primary_error
    if cleanup_errors:
        cleanup = (
            cleanup_errors[0]
            if len(cleanup_errors) == 1
            else BaseExceptionGroup("CLI shutdown failed", cleanup_errors)
        )
        raise cleanup


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the MyClaw Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        configuration = loader.load_for_startup()
    except ConfigError as config_error:
        _print_error(config_error.error, loader.path)
        exit_code = 1 if config_error.error.code == "persistence_error" else 2
        raise typer.Exit(code=exit_code) from None
    except OSError:
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None
    if not is_interactive_terminal():
        _print_error_info(
            ErrorInfo(
                "interactive_terminal_required",
                "Terminal Conversation requires interactive stdin, stdout, and stderr TTYs.",
            )
        )
        raise typer.Exit(code=2)
    if loader.diagnostics:
        console.print(
            "".join(f"{diagnostic.message}\n" for diagnostic in loader.diagnostics),
            markup=False,
            highlight=False,
            soft_wrap=True,
            end="",
        )
    try:
        asyncio.run(
            _run_cli_conversation(
                agent_home=loader.agent_home,
                workspace=Path.cwd(),
                configuration=configuration,
            )
        )
    except WorkspaceStateError:
        _print_error_info(_WORKSPACE_STATE_INITIALIZATION_ERROR)
        raise typer.Exit(code=1) from None
    except ModelContextOverflowError as context_error:
        _print_error_info(
            _approved_error_info(
                context_error,
                approved=(_MODEL_CONTEXT_OVERFLOW_ERROR,),
                fallback=_RUNTIME_STARTUP_ERROR,
            )
        )
        raise typer.Exit(code=1) from None
    except FatalManagementError as fatal_error:
        _print_error_info(
            _approved_error_info(
                fatal_error,
                approved=_SAFE_FATAL_MANAGEMENT_ERRORS,
                fallback=_RUNTIME_STARTUP_ERROR,
            )
        )
        raise typer.Exit(code=1) from None
    except Exception:
        _print_error_info(_RUNTIME_STARTUP_ERROR)
        raise typer.Exit(code=1) from None


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        loader.ensure_default()
        view = loader.view()
    except (OSError, UnicodeError):
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None

    console.print(
        view.header_text(),
        markup=False,
        highlight=False,
        soft_wrap=True,
        end="",
    )
    console.print(
        view.redacted_content,
        markup=False,
        highlight=False,
        soft_wrap=True,
        end="" if view.redacted_content.endswith("\n") else "\n",
    )
    if view.error is not None:
        raise typer.Exit(code=2)
