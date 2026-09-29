"""Single-process local service authority for Workspace and Session execution."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from uuid import uuid4

from tzlocal import get_localzone_name

from myclaw.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationOwner,
    ConfirmationPresenter,
    ForegroundConfirmationOwner,
    ToolConfirmationCoordinator,
)
from myclaw.agent.loop import AgentLoop, ForegroundConversationProjection
from myclaw.agent.memory.dream import Dream
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.message_bus import InboundMessage, MessageBus
from myclaw.agent.permission import RuntimePermissionControl
from myclaw.agent.session.restore import RestoreManager, RestoreRecoveryRequired, StaleRestorePlan
from myclaw.agent.session.session import Session, SessionStoragePartition
from myclaw.agent.tools.core.exec_host import ExecHost, create_exec_host, resolve_exec_shell
from myclaw.agent.tools.mcp_keywords import MCPKeywordPreparer
from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager
from myclaw.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES
from myclaw.agent.workspace_runtime import WorkspaceRuntime, WorkspaceRuntimeFactories
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.factory import create_provider
from myclaw.provider.model_router import ModelRouter
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleOccurrence, ScheduleService
from myclaw.service.errors import ServiceError, service_error
from myclaw.service.projects import ProjectCatalog, ProjectRecord
from myclaw.utils.time import local_now


class ServiceSink(Protocol):
    async def send_event(self, event: dict[str, object]) -> None: ...


def _consume_task_result(task: asyncio.Task[object]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


@dataclass(slots=True)
class ClientState:
    client_id: str
    kind: str
    reconnect_credential: str
    permission_control: RuntimePermissionControl
    connected: bool = False
    sink: ServiceSink | None = None
    stream_id: str = field(default_factory=lambda: str(uuid4()))
    sequence: int = 0
    events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=256))
    results: dict[str, dict[str, object]] = field(default_factory=dict)
    claimed: set[tuple[str, str]] = field(default_factory=set)
    current_workspace_id: str | None = None
    current_session_id: str | None = None
    disconnect_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class SessionClaim:
    workspace_id: str
    session_id: str
    client_id: str
    version: int
    credential: str
    loop: AgentLoop
    status: str = "claimed"
    disconnected_at: float | None = None


@dataclass(slots=True)
class _LoopState:
    loop: AgentLoop
    bus: MessageBus
    owner_client_id: str | None
    run_ids: deque[str] = field(default_factory=deque)
    output_task: asyncio.Task[None] | None = None
    schedule: bool = False


@dataclass(frozen=True, slots=True)
class _RestorePlanReference:
    """Wire-safe reference to the service-owned Restore plan."""

    anchor_id: int


class ServiceConfirmationPresenter(ConfirmationPresenter):
    """Bridge the existing one-shot coordinator to authenticated clients."""

    def __init__(self, service: LocalService) -> None:
        self._service = service
        self._wire_tokens: dict[object, str] = {}
        self._wire_sources: dict[object, tuple[str | None, str | None, str | None]] = {}

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None:
        wire_token = str(uuid4())
        self._wire_tokens[token] = wire_token
        workspace_id, session_id, run_id = self._service.confirmation_source(envelope.owner)
        self._wire_sources[token] = (workspace_id, session_id, run_id)
        payload: dict[str, object] = {
            "token": wire_token,
            "origin": envelope.origin,
            "request": envelope.request.to_dict(),
        }
        if envelope.job_id is not None:
            payload["job_id"] = envelope.job_id
        if envelope.title is not None:
            payload["title"] = envelope.title
        if isinstance(envelope.owner, ForegroundConfirmationOwner):
            payload["owner"] = {
                "kind": "foreground",
                "generation_id": str(envelope.owner.generation_id),
                "run_id": str(envelope.owner.run_id),
            }
        else:
            payload["owner"] = {
                "kind": "background",
                "generation_id": str(envelope.owner.generation_id),
                "job_id": envelope.owner.job_id,
                "occurrence_id": str(envelope.owner.occurrence_id),
            }
        task = asyncio.create_task(
            self._service.emit(
                "confirmation.requested",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload=payload,
                target_client_ids=self._audience(workspace_id, session_id),
            )
        )
        task.add_done_callback(_consume_task_result)

    async def dismiss_confirmation(self, token: object) -> None:
        wire_token = self._wire_tokens.pop(token, None)
        if wire_token is None:
            return
        workspace_id, session_id, run_id = self._wire_sources.pop(token, (None, None, None))
        await self._service.emit(
            "confirmation.resolved",
            workspace_id=workspace_id,
            session_id=session_id,
            run_id=run_id,
            payload={"token": wire_token},
            target_client_ids=self._audience(workspace_id, session_id),
        )

    def _audience(self, workspace_id: str | None, session_id: str | None) -> tuple[str, ...]:
        if workspace_id is None:
            return ()
        if session_id is not None:
            return tuple(
                client.client_id
                for client in self._service._clients.values()
                if (workspace_id, session_id) in client.claimed
            )
        return tuple(
            client.client_id
            for client in self._service._clients.values()
            if client.current_workspace_id == workspace_id
        )

    def decide(self, client_id: str, wire_token: str, decision: ConfirmationDecision) -> bool:
        for token, candidate in tuple(self._wire_tokens.items()):
            if candidate == wire_token:
                workspace_id, session_id, _ = self._wire_sources[token]
                if client_id not in self._audience(workspace_id, session_id):
                    raise service_error(
                        "forbidden", "Confirmation does not belong to this Client.", status=403
                    )
                accepted = self._service.confirmation.decide(token, decision)
                if accepted:
                    self._wire_tokens.pop(token, None)
                    workspace_id, session_id, run_id = self._wire_sources.pop(
                        token,
                        (None, None, None),
                    )
                    task = asyncio.create_task(
                        self._service.emit(
                            "confirmation.resolved",
                            workspace_id=workspace_id,
                            session_id=session_id,
                            run_id=run_id,
                            payload={"token": wire_token},
                            target_client_ids=self._audience(workspace_id, session_id),
                        )
                    )
                    task.add_done_callback(_consume_task_result)
                return accepted
        return False


class WorkspaceServiceRuntime:
    """Own one WorkspaceRuntime plus independent Session Agent Loops."""

    def __init__(
        self,
        service: LocalService,
        workspace_path: Path,
        configuration: UserConfiguration,
        *,
        workspace_id: str | None = None,
    ) -> None:
        self.service = service
        self.workspace_path = workspace_path
        self.configuration = configuration
        self.workspace_id = workspace_id or str(uuid4())
        self.runtime: WorkspaceRuntime | None = None
        self.workspace_state: Any = None
        self._loops: dict[str, _LoopState] = {}
        self._claims: dict[str, SessionClaim] = {}
        self._claim_versions: dict[str, int] = {}
        self._draft_clients: dict[str, str] = {}
        self._schedule_loops: dict[str, _LoopState] = {}
        self._restore_plans: dict[tuple[str, int], Any] = {}
        self._restore_results: dict[tuple[str, str], Any] = {}
        self._restore_owner: str | None = None
        self._restore_session_id: str | None = None
        self._restore_loop: AgentLoop | None = None
        self._restore_schedule_paused = False
        self._restore_blocked = False
        self._schedule_permission = RuntimePermissionControl(configuration.runtime.permission_level)
        self._exec_host: ExecHost | None = None
        self._started = False
        self._closed = False
        self._close_failed = False
        self._schedule_admitted = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Workspace service runtime is closed")
            if self._started:
                return
            resolved_shell = resolve_exec_shell(self.configuration.runtime.exec_shell)
            self._exec_host = create_exec_host(resolved_shell)

            async def execute_user_job(job: ScheduleJob) -> None:
                if not self._schedule_admitted:
                    raise ServiceError(
                        "admission_closed",
                        "Schedule admission is closed while the local service is reconnecting.",
                    )
                loop = await self._get_schedule_loop(job.job_id)
                await loop.loop.run_schedule_job(job)

            async def execute_user_occurrence(occurrence: ScheduleOccurrence) -> None:
                if not self._schedule_admitted:
                    raise ServiceError(
                        "admission_closed",
                        "Schedule admission is closed while the local service is reconnecting.",
                    )
                loop = await self._get_schedule_loop(occurrence.job.job_id)
                await loop.loop.run_schedule_job(occurrence.job, occurrence)

            self.runtime = WorkspaceRuntime.acquire(
                workspace=self.workspace_path,
                agent_home=self.service.agent_home,
                configuration=self.configuration,
                execute_user_job=execute_user_job,
                execute_user_occurrence=execute_user_occurrence,
                cancel_confirmation_owner=self.service.confirmation.cancel_owner,
                configured_schedule_level=self.configuration.runtime.permission_level,
                resolved_exec_shell=resolved_shell,
                now=local_now,
                timezone_name=get_localzone_name(),
                provider_factory=create_provider,
                built_in_names=BUILT_IN_TOOL_NAMES,
                factories=WorkspaceRuntimeFactories(
                    workspace_state=WorkspaceState,
                    restore_manager=RestoreManager,
                    mcp_runtime=MCPRuntimeManager,
                    router=ModelRouter,
                    mcp_keyword_preparer=MCPKeywordPreparer,
                    memory_manager=MemoryManager,
                    dream=Dream,
                    schedule_service=ScheduleService,
                ),
            )
            await self.runtime.start()
            self.workspace_state = self.runtime.workspace_state
            await self.runtime.prepare_schedule(
                JobSchedule.from_cron_input(
                    self.configuration.memory.schedule,
                    get_localzone_name(),
                )
            )
            self._started = True

    @property
    def schedule_service(self) -> ScheduleService:
        if self.runtime is None:
            raise RuntimeError("Workspace service runtime has not started")
        return self.runtime.schedule_service

    @property
    def loops(self) -> Mapping[str, _LoopState]:
        return self._loops

    async def activate_schedule(self) -> None:
        await self.start()
        if self._closed:
            return
        self._schedule_admitted = True
        self.schedule_service.resume()
        self.schedule_service.start()

    async def pause_schedule_admission(self) -> None:
        if not self._started or self._closed:
            return
        self._schedule_admitted = False
        await self.schedule_service.pause_admission()

    async def create_draft(self, client_id: str) -> str:
        if self._closed:
            raise service_error("admission_closed", "Workspace admission is closed.")
        startup_session_id = None if self.runtime is None else self.runtime.startup_session_id
        if startup_session_id is not None and startup_session_id not in self._loops:
            loop_state = await self._create_loop(startup_session_id, client_id=client_id)
        else:
            loop_state = await self._create_loop(None, client_id=client_id)
        session_id = loop_state.loop.session.session_id
        if startup_session_id is None:
            self._draft_clients[session_id] = client_id
        return session_id

    async def claim(self, client_id: str, session_id: str) -> SessionClaim:
        async with self._lock:
            if self._closed:
                raise service_error("admission_closed", "Workspace admission is closed.")
            existing = self._claims.get(session_id)
            if existing is not None:
                if existing.client_id != client_id:
                    raise service_error(
                        "session_claimed",
                        "Conversation Session is already claimed by another client.",
                        retryable=True,
                    )
                return existing
            draft_owner = self._draft_clients.get(session_id)
            if draft_owner is not None and draft_owner != client_id:
                raise service_error(
                    "session_claimed",
                    "Conversation Session is already claimed by another client.",
                    retryable=True,
                )
            loop_state = self._loops.get(session_id)
            if loop_state is None:
                loop_state = await self._create_loop(session_id, client_id=client_id)
            elif loop_state.owner_client_id not in {None, client_id}:
                raise service_error(
                    "session_claimed",
                    "Conversation Session is already claimed by another client.",
                    retryable=True,
                )
            loop_state.owner_client_id = client_id
            version = self._claim_versions.get(session_id, 0) + 1
            self._claim_versions[session_id] = version
            claim = SessionClaim(
                workspace_id=self.workspace_id,
                session_id=session_id,
                client_id=client_id,
                version=version,
                credential=str(uuid4()),
                loop=loop_state.loop,
            )
            self._claims[session_id] = claim
            self._draft_clients.pop(session_id, None)
            return claim

    def require_claim(
        self,
        client_id: str,
        session_id: str,
        version: int,
        credential: str | None = None,
    ) -> SessionClaim:
        claim = self._claims.get(session_id)
        if (
            claim is None
            or claim.client_id != client_id
            or claim.version != version
            or (credential is not None and claim.credential != credential)
        ):
            raise service_error(
                "stale_claim",
                "Conversation Session Claim is missing or stale.",
                retryable=True,
            )
        if claim.status != "claimed":
            raise service_error(
                "stale_claim",
                "Conversation Session Claim is no longer active.",
                retryable=True,
            )
        return claim

    async def release(self, client_id: str, session_id: str, *, close_idle: bool = True) -> None:
        if self._restore_owner == client_id and self._restore_session_id == session_id:
            await self._release_restore_barrier(client_id)
        claim = self._claims.get(session_id)
        if claim is None:
            return
        if claim.client_id != client_id:
            raise service_error(
                "stale_claim", "Conversation Session Claim is not owned by this client."
            )
        self._claims.pop(session_id, None)
        self.service.client_claim_released(client_id, self.workspace_id, session_id)
        loop_state = self._loops.get(session_id)
        if close_idle and loop_state is not None and not loop_state.loop.has_active_run:
            await self._close_loop(session_id)

    async def expire_client(self, client_id: str) -> None:
        if self._restore_owner == client_id:
            await self._release_restore_barrier(client_id)
        for session_id, claim in tuple(self._claims.items()):
            if claim.client_id != client_id:
                continue
            loop_state = self._loops.get(session_id)
            try:
                if loop_state is not None:
                    try:
                        await loop_state.loop.cancel_active_run()
                    except RuntimeError:
                        pass
                    await self._close_loop(session_id, abort=True)
            finally:
                self._claims.pop(session_id, None)
                self.service.client_claim_released(client_id, self.workspace_id, session_id)
        for session_id, owner in tuple(self._draft_clients.items()):
            if owner == client_id:
                self._draft_clients.pop(session_id, None)
                await self._close_loop(session_id, abort=True)

    def confirmation_source(
        self, owner: ConfirmationOwner
    ) -> tuple[str | None, str | None, str | None]:
        for session_id, claim in self._claims.items():
            if claim.loop.generation_id == owner.generation_id:
                run_id = (
                    str(owner.run_id) if isinstance(owner, ForegroundConfirmationOwner) else None
                )
                return self.workspace_id, session_id, run_id
        if isinstance(owner, BackgroundConfirmationOwner):
            return self.workspace_id, None, None
        return self.workspace_id, None, None

    def projection(self, session_id: str) -> ForegroundConversationProjection:
        loop = self._loops.get(session_id)
        if loop is None:
            raise service_error("not_found", "Conversation Session was not found.", status=404)
        return loop.loop.project_foreground_conversation()

    async def _release_restore_barrier(self, client_id: str) -> None:
        if self._restore_owner != client_id:
            return
        loop = self._restore_loop
        self._restore_owner = None
        self._restore_session_id = None
        self._restore_loop = None
        for key in tuple(self._restore_plans):
            if key[0] == client_id:
                self._restore_plans.pop(key, None)
        if loop is not None:
            await loop._release_replacement_barrier(resume_inbound=True)
        if self._restore_schedule_paused:
            self._restore_schedule_paused = False
            if not self._closed and not self._restore_blocked and self.service.state == "ready":
                self.schedule_service.resume()

    async def _rebuild_restored_session(self, client_id: str, session_id: str) -> SessionClaim:
        claim = self._claims.get(session_id)
        if claim is None or claim.client_id != client_id:
            raise service_error("stale_claim", "Conversation Session Claim is missing or stale.")
        await self._close_loop(session_id, abort=True)
        state = await self._create_loop(session_id, client_id=client_id)
        version = self._claim_versions.get(session_id, 0) + 1
        self._claim_versions[session_id] = version
        claim.loop = state.loop
        claim.version = version
        claim.credential = str(uuid4())
        return claim

    def management_dispatcher(self, client_id: str, session_id: str) -> Any:
        """Build the existing typed Management dispatcher for one Claim."""
        from myclaw.management.commands import ManagementCommandDispatcher
        from myclaw.management.service import (
            ManagementError,
            ManagementViewService,
            RestoreListingReport,
        )

        owned_claim = self._claims.get(session_id)
        if owned_claim is None:
            raise service_error("stale_claim", "Conversation Session Claim is missing or stale.")
        initial_claim = self.require_claim(client_id, session_id, owned_claim.version)

        def current_loop() -> AgentLoop:
            client = self.service.client(client_id)
            selected_session = client.current_session_id or session_id
            claim = self._claims.get(selected_session)
            if claim is None or claim.client_id != client_id:
                return initial_claim.loop
            return claim.loop

        async def prepare_resume(target_session_id: str) -> None:
            loop = current_loop()
            if loop.session.session_id == target_session_id:
                await loop.wait_for_restore_idle()

        async def replace_loop(target_session_id: str, force: bool) -> None:
            if self._restore_owner == client_id:
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request", "Session Restore is waiting for confirmation."
                    )
                )
            loop = current_loop()
            if loop.has_active_run and not force:
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "Finish or cancel the active foreground run before switching sessions.",
                    )
                )
            previous = self.service.client(client_id).current_session_id
            await self.claim(client_id, target_session_id)
            if previous is not None and previous != target_session_id:
                await self.release(client_id, previous, close_idle=not loop.has_active_run)

        async def restore_listing() -> RestoreListingReport:
            from myclaw.management.service import RestoreListingReport

            loop = current_loop()
            state = self._loops[loop.session.session_id]
            if (
                self._restore_owner is not None
                or loop.has_active_run
                or await state.bus.inbound_snapshot()
            ):
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "Finish the active run and clear queued input before restoring.",
                    )
                )
            self._restore_owner = client_id
            self._restore_session_id = loop.session.session_id
            self._restore_loop = loop
            try:
                await loop._pause_for_replacement()
                await loop.wait_for_restore_idle()
                anchors = loop.session.restore_candidates()
                if not anchors:
                    await self._release_restore_barrier(client_id)
                return RestoreListingReport(loop.session.session_id, tuple(reversed(anchors)))
            except BaseException:
                await self._release_restore_barrier(client_id)
                raise

        async def restore_inspect(anchor_id: int) -> Any:
            from myclaw.agent.session.restore import RestoreManager

            loop = current_loop()
            if self._restore_owner != client_id or self._restore_loop is not loop:
                raise ManagementError(
                    ErrorInfo("model_invalid_request", "Session Restore is not active.")
                )
            try:
                if not self._restore_schedule_paused:
                    await self.schedule_service.pause_and_wait_idle()
                    self._restore_schedule_paused = True
                manager = RestoreManager(
                    self.workspace_state, loop.session.session_id, now=local_now
                )
                plan = manager.inspect(loop.session, anchor_id)
                manager.revalidate(plan)
                self._restore_plans[(client_id, anchor_id)] = plan
                return plan
            except BaseException:
                await self._release_restore_barrier(client_id)
                raise

        async def restore_commit(plan: Any, mode: Any) -> Any:
            stored = self._restore_plans.get((client_id, plan.anchor_id))
            if stored is None or self._restore_owner != client_id:
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                )
            executed = False
            try:
                manager = RestoreManager(self.workspace_state, stored.session_id, now=local_now)
                manager.revalidate(stored)
                result = await manager.execute(stored, mode)
                executed = True
                await self._rebuild_restored_session(client_id, stored.session_id)
                self._restore_results[(client_id, stored.session_id)] = result
                return result
            except StaleRestorePlan as error:
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                ) from error
            except BaseException as error:
                if executed or isinstance(error, RestoreRecoveryRequired):
                    self._restore_blocked = True
                raise
            finally:
                await self._release_restore_barrier(client_id)

        async def restore_result() -> Any:
            return self._restore_results.get((client_id, current_loop().session.session_id))

        async def restore_acknowledge_failure() -> Any:
            from myclaw.agent.session.restore import RestoreManager

            session_id = current_loop().session.session_id
            result = self._restore_results.get((client_id, session_id))
            if result is None:
                return None
            acknowledged = RestoreManager(
                self.workspace_state,
                session_id,
                now=local_now,
            ).acknowledge_failure_notification()
            if acknowledged is not None:
                self._restore_results[(client_id, session_id)] = acknowledged
            return acknowledged

        async def restore_cancel() -> None:
            await self._release_restore_barrier(client_id)

        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("Workspace service runtime is not ready")
        management = ManagementViewService(
            self.service.agent_home,
            current_agent_loop=current_loop,
            workspace_state=self.workspace_state,
            replace_agent_loop=replace_loop,
            prepare_session_resume=prepare_resume,
            memory_manager=runtime.memory_manager,
            dream=runtime.dream,
            schedule_status=lambda: self.schedule_service.status_snapshot().to_dict(),
            now=local_now,
            monotonic=monotonic,
            reasoning_effort_control=runtime.router,
            permission_control=self.service.client_permission(client_id),
            restore_listing=restore_listing,
            restore_inspect=restore_inspect,
            restore_commit=restore_commit,
            restore_result=restore_result,
            restore_cancel=restore_cancel,
            ensure_management_mutation_allowed=lambda: self._require_admitted(),
        )
        management.bind_restore_acknowledge_failure(restore_acknowledge_failure)
        return ManagementCommandDispatcher(management)

    def _require_admitted(self) -> None:
        if (
            self._closed
            or self._restore_blocked
            or (not self._schedule_admitted and self.service.state in {"draining", "stopped"})
        ):
            raise ServiceError("admission_closed", "Workspace admission is closed.")

    async def input(
        self,
        client_id: str,
        session_id: str,
        version: int,
        text: str,
        run_id: str,
    ) -> SessionClaim:
        self._require_admitted()
        claim = self.require_claim(client_id, session_id, version)
        if not text.strip():
            raise service_error(
                "validation_error",
                "Conversation input must not be empty.",
                status=422,
                field_errors={"text": "must not be empty"},
            )
        if not claim.loop.foreground_input_admitted():
            raise service_error(
                "admission_closed", "Conversation input is temporarily unavailable."
            )
        state = self._loops[session_id]
        state.run_ids.append(run_id)
        await state.bus.put_inbound(InboundMessage(content=text))
        return claim

    async def cancel(self, client_id: str, session_id: str, version: int, run_id: str) -> None:
        claim = self.require_claim(client_id, session_id, version)
        if not claim.loop.has_active_run:
            return
        state = self._loops[session_id]
        if not state.run_ids or state.run_ids[0] != run_id:
            raise service_error("stale_run", "The requested Agent Run is no longer active.")
        await claim.loop.cancel_active_run()
        await self.service.emit(
            "run.cancelled",
            workspace_id=self.workspace_id,
            session_id=session_id,
            run_id=run_id,
            payload={},
        )

    async def _create_loop(self, session_id: str | None, *, client_id: str | None) -> _LoopState:
        if self.runtime is None or self.workspace_state is None or self._exec_host is None:
            raise RuntimeError("Workspace service runtime is not ready")
        permission_control = (
            self.service.client_permission(client_id)
            if client_id is not None
            else self._schedule_permission
        )
        bus = MessageBus()
        loop = AgentLoop(
            workspace_path=self.workspace_path,
            workspace_state=self.workspace_state,
            agent_home=self.service.agent_home,
            configuration=self.configuration,
            bus=bus,
            schedule_service=self.schedule_service,
            model_router=self.runtime.router,
            memory_manager=self.runtime.memory_manager,
            session_id=session_id,
            now=local_now,
            new_uuid=uuid4,
            monotonic_now=monotonic,
            mcp_tools=self.runtime.mcp_snapshot,
            mcp_keywords=self.runtime.mcp_keywords,
            exec_host=self._exec_host,
            permission_control=permission_control,
        )
        loop.bind_confirmation_requester(self.service.confirmation.request)
        loop.preflight()
        await loop.start()
        state = _LoopState(loop=loop, bus=bus, owner_client_id=client_id)
        self._loops[loop.session.session_id] = state
        if client_id is not None:
            state.output_task = asyncio.create_task(self._forward_output(state))
        return state

    async def _get_schedule_loop(self, job_id: str) -> _LoopState:
        state = self._schedule_loops.get(job_id)
        if state is not None:
            return state
        state = await self._create_loop(None, client_id=None)
        state.schedule = True
        self._schedule_loops[job_id] = state
        return state

    async def _forward_output(self, state: _LoopState) -> None:
        session_id = state.loop.session.session_id
        try:
            while not self._closed:
                message = await state.bus.get_outbound()
                run_id = state.run_ids[0] if state.run_ids else None
                await self.service.emit(
                    "run.output",
                    workspace_id=self.workspace_id,
                    session_id=session_id,
                    run_id=run_id,
                    payload={
                        "message": {
                            "type": message.type,
                            "content": message.content,
                            "metadata": dict(message.metadata),
                        }
                    },
                )
                if message.metadata.get("_streamed") is True and state.run_ids:
                    completed = state.run_ids.popleft()
                    await self.service.emit(
                        "run.completed",
                        workspace_id=self.workspace_id,
                        session_id=session_id,
                        run_id=completed,
                        payload={
                            "finish_reason": message.metadata.get("finish_reason", "completed")
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _close_loop(self, session_id: str, *, abort: bool = False) -> None:
        state = self._loops.pop(session_id, None)
        if state is None:
            return
        if state.output_task is not None:
            state.output_task.cancel()
            await asyncio.gather(state.output_task, return_exceptions=True)
        try:
            if abort:
                await state.loop.abort()
            else:
                await state.loop.close()
        finally:
            for job_id, candidate in tuple(self._schedule_loops.items()):
                if candidate is state:
                    self._schedule_loops.pop(job_id, None)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                if self._close_failed:
                    raise service_error(
                        "workspace_close_failed", "Workspace cleanup did not finish safely."
                    )
                return
            self._closed = True
            self._schedule_admitted = False
            try:
                if self._restore_owner is not None:
                    await self._release_restore_barrier(self._restore_owner)
                if self.runtime is not None:
                    await self.runtime.close(
                        close_foreground=lambda: self._close_all_loops(),
                        drain_confirmation_aborts=True,
                    )
                else:
                    await self._close_all_loops()
            except BaseException:
                self._close_failed = True
                raise

    async def _close_all_loops(self) -> None:
        for session_id in tuple(self._loops):
            await self._close_loop(session_id, abort=True)


class LocalService:
    """Own one local service instance and expose typed command behavior."""

    def __init__(
        self,
        agent_home: AgentHome,
        configuration: UserConfiguration | None = None,
        *,
        reconnect_timeout: float = 30.0,
        monotonic_now: Callable[[], float] = monotonic,
    ) -> None:
        self.agent_home = agent_home
        self.configuration = configuration
        self.reconnect_timeout = reconnect_timeout
        self._monotonic = monotonic_now
        self.service_instance_id = str(uuid4())
        self.protocol_version = 1
        self.state = "starting"
        self.confirmation = ToolConfirmationCoordinator()
        self._presenter = ServiceConfirmationPresenter(self)
        self._clients: dict[str, ClientState] = {}
        self._client_by_reconnect: dict[str, str] = {}
        self._workspaces: dict[str, WorkspaceServiceRuntime] = {}
        self._workspace_keys: dict[str, str] = {}
        self._global_reconnect_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_failed = False
        self._closed = asyncio.Event()
        self._lock = asyncio.Lock()
        self.projects = ProjectCatalog(agent_home)

    async def start(self) -> None:
        if self.state != "starting":
            return
        self.agent_home.initialize()
        self.confirmation.bind_presenter(self._presenter)
        for record in self.projects.list():
            if record.schedule_state == "available" and record.path.is_dir():
                await self._get_or_create_workspace(record.path)
        self.state = "ready"
        self._global_reconnect_task = asyncio.create_task(self._stop_after_grace())

    @property
    def workspaces(self) -> Mapping[str, WorkspaceServiceRuntime]:
        return self._workspaces

    def client(self, client_id: str) -> ClientState:
        return self._require_client(client_id)

    async def register_client(
        self,
        kind: str,
        reconnect_credential: str | None = None,
    ) -> ClientState:
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        if reconnect_credential is not None:
            client_id = self._client_by_reconnect.get(reconnect_credential)
            if client_id is not None:
                client = self._clients[client_id]
                if client.kind != kind or client.connected:
                    raise service_error(
                        "client_already_connected",
                        "This Client is already connected or has a different kind.",
                    )
                if client.disconnect_task is not None:
                    client.disconnect_task.cancel()
                    client.disconnect_task = None
                self._client_by_reconnect.pop(client.reconnect_credential, None)
                client.reconnect_credential = str(uuid4())
                self._client_by_reconnect[client.reconnect_credential] = client_id
                return client
        permission = (
            self.configuration.runtime.permission_level if self.configuration else "workspace-write"
        )
        client = ClientState(str(uuid4()), kind, str(uuid4()), RuntimePermissionControl(permission))
        self._clients[client.client_id] = client
        self._client_by_reconnect[client.reconnect_credential] = client.client_id
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

    async def connect_client(self, client_id: str, sink: ServiceSink) -> None:
        client = self._require_client(client_id)
        if client.connected:
            raise service_error("client_already_connected", "This Client already has a connection.")
        client.connected = True
        client.sink = sink
        if client.disconnect_task is not None:
            client.disconnect_task.cancel()
            client.disconnect_task = None
        if self._global_reconnect_task is not None:
            self._global_reconnect_task.cancel()
            self._global_reconnect_task = None
        if self.state == "reconnecting":
            self.state = "ready"
        if self.state == "ready":
            for workspace in self._workspaces.values():
                if self._schedule_allowed(workspace):
                    await workspace.activate_schedule()
        for event in tuple(client.events):
            await sink.send_event(event)

    async def disconnect_client(self, client_id: str, *, sink: ServiceSink | None = None) -> None:
        client = self._clients.get(client_id)
        if client is None or (sink is not None and client.sink is not sink):
            return
        client.connected = False
        client.sink = None
        if client.disconnect_task is not None:
            client.disconnect_task.cancel()
        client.disconnect_task = asyncio.create_task(self._expire_client_later(client_id))
        if not any(candidate.connected for candidate in self._clients.values()):
            if self.state == "ready":
                self.state = "reconnecting"
                for workspace in self._workspaces.values():
                    await workspace.pause_schedule_admission()
                self._global_reconnect_task = asyncio.create_task(self._stop_after_grace())

    async def attach_workspace(self, client_id: str, path: Path) -> WorkspaceServiceRuntime:
        self._require_client(client_id)
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
        runtime = await self._get_or_create_workspace(normalized)
        if any(client.connected for client in self._clients.values()) and self._schedule_allowed(
            runtime
        ):
            await runtime.activate_schedule()
        return runtime

    async def _get_or_create_workspace(self, path: Path) -> WorkspaceServiceRuntime:
        key = os.path.normcase(str(path.resolve(strict=True)))
        async with self._lock:
            workspace_id = self._workspace_keys.get(key)
            if workspace_id is not None:
                return self._workspaces[workspace_id]
            if self.configuration is None:
                raise service_error(
                    "config_invalid", "User Configuration is unavailable.", status=422
                )
            workspace_id = str(uuid4())
            runtime = WorkspaceServiceRuntime(
                self, path, self.configuration, workspace_id=workspace_id
            )
            await runtime.start()
            self._workspace_keys[key] = workspace_id
            self._workspaces[workspace_id] = runtime
            return runtime

    def _schedule_allowed(self, workspace: WorkspaceServiceRuntime) -> bool:
        key = os.path.normcase(str(workspace.workspace_path))
        for record in self.projects.list():
            if os.path.normcase(str(record.path.resolve(strict=False))) == key:
                return record.schedule_state == "available"
        return True

    async def register_project(
        self, client_id: str, path: Path
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime, tuple[ScheduleJob, ...]]:
        self._require_client(client_id)
        record = self.projects.register(path, schedule_state="awaiting_resume")
        workspace = await self.attach_workspace(client_id, record.path)
        jobs = await workspace.schedule_service.public_snapshot()
        if not jobs and record.schedule_state == "awaiting_resume":
            record = self.projects.set_schedule_state(record.project_id, "available")
            if any(client.connected for client in self._clients.values()):
                await workspace.activate_schedule()
        return record, workspace, jobs

    async def resume_project_schedule(
        self, client_id: str, project_id: str, expected_job_ids: set[str]
    ) -> str:
        self._require_client(client_id)
        record = next(
            (item for item in self.projects.list() if item.project_id == project_id), None
        )
        if record is None:
            raise service_error("not_found", "Project registration was not found.", status=404)
        if record.schedule_state != "awaiting_resume":
            return record.schedule_state
        workspace = await self._get_or_create_workspace(record.path)
        jobs = await workspace.schedule_service.public_snapshot()
        if {job.job_id for job in jobs} != expected_job_ids:
            raise service_error(
                "stale_schedule_review", "Saved Schedule Jobs changed; review them again."
            )
        self.projects.set_schedule_state(project_id, "available")
        if any(client.connected for client in self._clients.values()):
            await workspace.activate_schedule()
        return "available"

    async def remove_project(self, client_id: str, project_id: str) -> Path:
        self._require_client(client_id)
        record = next(
            (item for item in self.projects.list() if item.project_id == project_id), None
        )
        if record is None:
            raise service_error("not_found", "Project registration was not found.", status=404)
        self.projects.set_schedule_state(project_id, "removing")
        key = os.path.normcase(str(record.path.resolve(strict=False)))
        workspace_id = self._workspace_keys.get(key)
        if workspace_id is not None:
            workspace = self._workspaces[workspace_id]
            affected = tuple(
                client.client_id
                for client in self._clients.values()
                if client.current_workspace_id == workspace_id
                or any(claimed_workspace == workspace_id for claimed_workspace, _ in client.claimed)
            )
            try:
                await workspace.close()
            except Exception as error:
                raise service_error(
                    "project_removal_failed",
                    "Project work could not be stopped; the registration remains blocked.",
                    status=500,
                ) from error
            self._workspace_keys.pop(key, None)
            self._workspaces.pop(workspace_id, None)
            for affected_id in affected:
                client = self._clients.get(affected_id)
                if client is not None:
                    client.claimed = {claim for claim in client.claimed if claim[0] != workspace_id}
                    if client.current_workspace_id == workspace_id:
                        client.current_workspace_id = None
                        client.current_session_id = None
            await self.emit(
                "project.removed",
                workspace_id=workspace_id,
                session_id=None,
                run_id=None,
                payload={"project_id": project_id},
                target_client_ids=affected,
            )
        self.projects.remove(project_id)
        return record.path

    def workspace(self, workspace_id: str) -> WorkspaceServiceRuntime:
        try:
            return self._workspaces[workspace_id]
        except KeyError as error:
            raise service_error("not_found", "Workspace was not found.", status=404) from error

    async def create_session(self, client_id: str, workspace_id: str) -> dict[str, object]:
        self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        session_id = await workspace.create_draft(client_id)
        return {"workspace_id": workspace_id, "session_id": session_id}

    async def claim(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
    ) -> dict[str, object]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        claim = await workspace.claim(client_id, session_id)
        client.claimed.add((workspace_id, session_id))
        client.current_workspace_id = workspace_id
        client.current_session_id = session_id
        projection = workspace.projection(session_id)
        return {
            "claim": {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "claim_version": claim.version,
                "reconnect_credential": claim.credential,
            },
            "snapshot": {
                "session_id": projection.session_id,
                "messages": list(projection.messages),
            },
        }

    async def list_sessions(self, client_id: str, workspace_id: str) -> list[dict[str, object]]:
        client = self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        directory = workspace.workspace_state.existing_sessions_directory()
        if directory is None:
            return []
        entries: list[dict[str, object]] = []
        for path in directory.glob("*.jsonl"):
            try:
                session = Session.load(
                    workspace.workspace_state,
                    path.stem,
                    partition=SessionStoragePartition.FOREGROUND,
                    now=local_now,
                )
            except (OSError, ValueError, UnicodeError):
                continue
            claim = workspace._claims.get(session.session_id)
            entries.append(
                {
                    "id": session.session_id,
                    "title": session.metadata.get("title", "Untitled session"),
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": len(session.messages),
                    "occupied": claim is not None,
                    "occupied_by": None
                    if claim is None or claim.client_id == client.client_id
                    else "client",
                }
            )
        entries.sort(key=lambda item: (str(item["updated_at"]), str(item["id"])), reverse=True)
        return entries

    async def handle_management(
        self,
        client_id: str,
        workspace_id: str,
        session_id: str,
        action: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        self._require_client(client_id)
        workspace = self.workspace(workspace_id)
        dispatcher = workspace.management_dispatcher(client_id, session_id)
        if action == "dispatch":
            command = payload.get("command")
            if not isinstance(command, str):
                raise service_error(
                    "validation_error", "Management command is required.", status=422
                )
            result = await dispatcher.dispatch(command)
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
                result = await dispatcher.restore_commit(_RestorePlanReference(anchor_id), mode)
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
        elif action == "restore/acknowledge-failure":
            result = await dispatcher.restore_acknowledge_failure()
        else:
            raise service_error("validation_error", "Unsupported management action.", status=422)
        encoded = _encode_management_result(result)
        if action == "restore/execute" and getattr(result, "restore_result", None) is not None:
            claim = workspace._claims.get(session_id)
            if claim is not None and claim.client_id == client_id:
                encoded["claim_version"] = claim.version
                encoded["claim_credential"] = claim.credential
        return encoded

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
            assert isinstance(workspace_id, str) and isinstance(session_id, str)
            await self.workspace(workspace_id).release(client_id, session_id)
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
            run_id = str(uuid4())
            await self.workspace(workspace_id).input(
                client_id, session_id, claim_version, text, run_id
            )
            result = {"run_id": run_id}
            await self.emit(
                "input.accepted",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload={"text": text},
                target_client_ids=(client_id,),
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
            await self.workspace(workspace_id).cancel(
                client_id, session_id, claim_version, cancel_run_id
            )
            result = {"cancelled": True}
        elif command_type == "confirmation_decide":
            token = payload.get("token")
            decision = payload.get("decision")
            if not isinstance(token, str) or decision not in {"approved", "declined"}:
                raise service_error(
                    "validation_error", "confirmation decision is invalid.", status=422
                )
            accepted = self._presenter.decide(
                client_id, token, cast(ConfirmationDecision, decision)
            )
            if not accepted:
                raise service_error("confirmation_resolved", "Confirmation is already resolved.")
            result = {"decided": True}
        elif command_type == "subscribe":
            last_seq = payload.get("last_seq")
            await self._replay(client, last_seq)
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
        if event_type == "project.removed" and isinstance(payload.get("project_id"), str):
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
                if (workspace_id, session_id) not in client.claimed and event_type.startswith(
                    "run."
                ):
                    continue
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
            if client.connected and client.sink is not None:
                await self._send_event(client, event)

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
            await self._stop_task
            return
        self._stop_task = asyncio.create_task(self._stop_owned())
        await self._stop_task

    async def _stop_owned(self) -> None:
        if self.state == "stopped":
            return
        self.state = "draining"
        errors: list[Exception] = []
        if self._global_reconnect_task is not None:
            self._global_reconnect_task.cancel()
        for client in self._clients.values():
            if client.disconnect_task is not None:
                client.disconnect_task.cancel()
        for workspace in tuple(self._workspaces.values()):
            try:
                await workspace.pause_schedule_admission()
            except Exception as error:
                errors.append(error)
        try:
            await self.confirmation.close()
        except Exception as error:
            errors.append(error)
        for workspace in tuple(self._workspaces.values()):
            try:
                await workspace.close()
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
        return client

    async def _expire_client_later(self, client_id: str) -> None:
        try:
            await asyncio.sleep(self.reconnect_timeout)
        except asyncio.CancelledError:
            return
        client = self._clients.get(client_id)
        if client is None or client.connected:
            return
        for workspace in self._workspaces.values():
            await workspace.expire_client(client_id)
        client.claimed.clear()
        self._client_by_reconnect.pop(client.reconnect_credential, None)
        self._clients.pop(client_id, None)

    async def _stop_after_grace(self) -> None:
        try:
            await asyncio.sleep(self.reconnect_timeout)
        except asyncio.CancelledError:
            return
        if not any(client.connected for client in self._clients.values()):
            self._global_reconnect_task = None
            await self.stop()

    async def _replay(self, client: ClientState, last_seq: object) -> None:
        if last_seq is None:
            return
        if isinstance(last_seq, bool) or not isinstance(last_seq, int) or last_seq < 0:
            raise service_error("validation_error", "last_seq is invalid.", status=422)
        first_seq = int(cast(int, client.events[0]["seq"])) if client.events else 0
        if not client.events or last_seq >= first_seq - 1:
            for event in tuple(client.events):
                if int(cast(int, event["seq"])) > last_seq and client.sink is not None:
                    await self._send_event(client, event)
            return
        await self.emit(
            "snapshot.required",
            workspace_id=None,
            session_id=None,
            run_id=None,
            payload={"reason": "event_cache_exhausted", "stream_id": client.stream_id},
            target_client_ids=(client.client_id,),
        )

    async def _send_event(self, client: ClientState, event: dict[str, object]) -> None:
        sink = client.sink
        if not client.connected or sink is None:
            return
        try:
            await asyncio.wait_for(sink.send_event(event), timeout=1.0)
        except Exception:
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


def _encode_management_result(result: object) -> dict[str, object]:
    from myclaw.management.commands import ManagementCommandResult

    if not isinstance(result, ManagementCommandResult):
        raise service_error("service_protocol_error", "Management result is invalid.", status=500)
    encoded: dict[str, object] = {
        "handled": result.handled,
        "output": result.output,
        "effort_selection": result.effort_selection,
        "permission_selection": result.permission_selection,
        "resumed_session_id": result.resumed_session_id,
        "resume_skipped_count": result.resume_skipped_count,
    }
    if result.status_view is not None:
        encoded["status_view"] = result.status_view.to_dict()
    if result.resume_sessions is not None:
        encoded["resume_sessions"] = [
            {
                "id": item.id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "message_count": item.message_count,
            }
            for item in result.resume_sessions
        ]
    if result.skill_metadata is not None:
        encoded["skill_metadata"] = [
            {"name": item.name, "description": item.description, "path": str(item.path)}
            for item in result.skill_metadata
        ]
    if result.restore_listing is not None:
        encoded["restore_listing"] = {
            "session_id": result.restore_listing.session_id,
            "anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "run_token": str(anchor.run_token),
                    "content": anchor.content,
                    "timestamp": anchor.timestamp,
                }
                for anchor in result.restore_listing.anchors
            ],
        }
    if result.restore_plan is not None:
        encoded["restore_plan"] = _safe_wire_value(result.restore_plan)
    if result.restore_result is not None:
        encoded["restore_result"] = _safe_wire_value(result.restore_result)
    return encoded


def _safe_wire_value(value: object) -> object:
    from datetime import datetime
    from enum import Enum
    from pathlib import Path
    from uuid import UUID

    if is_dataclass(value):
        return {
            item.name: _safe_wire_value(getattr(value, item.name))
            for item in fields(value)
            if item.name != "before_bytes"
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_wire_value(item) for item in value]
    if isinstance(value, (Path, UUID, datetime, Enum)):
        return str(value)
    if isinstance(value, bytes):
        return None
    return value


__all__ = [
    "ClientState",
    "LocalService",
    "ServiceConfirmationPresenter",
    "ServiceSink",
    "SessionClaim",
    "WorkspaceServiceRuntime",
]
