"""Workspace claims, queues and execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

from tzlocal import get_localzone_name

from aide.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationOwner,
    ForegroundConfirmationOwner,
    SubAgentConfirmationOwner,
)
from aide.agent.context.builder import ContextBuilder
from aide.agent.foreground_work.store import (
    ForegroundWorkConflict,
    ForegroundWorkDelivery,
    ForegroundWorkStore,
    ForegroundWorkTask,
    StaleForegroundWorkRevision,
)
from aide.agent.loop import (
    AgentRunExecutor,
    ForegroundConversationProjection,
    session_runtime_status_input,
)
from aide.agent.memory.dream import Dream
from aide.agent.memory.manager import MemoryManager
from aide.agent.message_bus import InboundMessage, MessageBus
from aide.agent.permission import PermissionSnapshot, RuntimePermissionControl
from aide.agent.session.deletion import (
    begin_session_deletion,
    delete_session_data,
    session_deletion_pending,
    session_restore_pending,
)
from aide.agent.session.execution_state import SessionRunState
from aide.agent.session.restore import (
    RestoreManager,
    RestoreRecoveryRequired,
    RestoreResult,
    StaleRestorePlan,
)
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.subagents.coordinator import SubAgentPool
from aide.agent.subagents.models import (
    SubAgentError,
    SubAgentEvent,
    SubAgentRecord,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.ports import SubAgentRecordRepository, SubAgentSessionCoordinator
from aide.agent.subagents.store import (
    SubAgentRecordStore,
    SubAgentRequestError,
    SubAgentStoreError,
)
from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.mcp_keywords import MCPKeywordPreparer
from aide.agent.tools.mcp_runtime import MCPWorkspaceRuntimeManager
from aide.agent.tools.tool_gateway import (
    ConfirmationRequest,
    ConfirmationRequester,
)
from aide.config.config import (
    ConfigError,
    UserConfiguration,
)
from aide.provider.factory import create_provider
from aide.provider.model_router import ModelRouter
from aide.schedule.service import (
    ScheduleService,
)
from aide.service.configuration_resources import (
    WorkspaceConfigurationResources,
)
from aide.service.errors import ServiceError, service_error
from aide.service.execution import SessionExecution
from aide.service.resources import WorkspaceResources
from aide.service.runtime.pagination import (
    _MAX_SESSION_PAGE_SIZE,
    _decode_session_cursor,
    _encode_session_cursor,
)
from aide.service.runtime.records import (
    SessionClaim,
    SessionDeletionClaim,
    _consume_task_result,
    _LoopState,
)
from aide.skills.catalog import SkillLoader
from aide.trace.context import workspace_key_for
from aide.utils.errors import ErrorInfo
from aide.utils.host_filesystem import HOST_FILESYSTEM
from aide.utils.time import local_now

if TYPE_CHECKING:
    from aide.service.runtime.service import AgentService


def _effective_task_restore_token(
    messages: list[dict[str, Any]], task_id: str, fallback_token: str | None
) -> str | None:
    valid_user_tokens: set[str] = set()
    task_token: str | None = None
    for message in messages:
        token = message.get("restore_run_token")
        if (
            message.get("role") != "user"
            or type(message.get("restore_anchor_id")) is not int
            or not isinstance(token, str)
        ):
            continue
        try:
            parsed = UUID(token)
        except ValueError:
            continue
        if parsed.version != 4:
            continue
        normalized = str(parsed)
        valid_user_tokens.add(normalized)
        if message.get("foreground_task_id") == task_id:
            task_token = normalized
    if task_token is not None:
        return task_token
    if fallback_token is None:
        return None
    try:
        fallback = str(UUID(fallback_token))
    except ValueError:
        return None
    return fallback if fallback in valid_user_tokens else None


class WorkspaceRecord:
    """Compatibility record for one Workspace's data and coordination state."""

    def __init__(
        self,
        service: AgentService,
        workspace_path: Path,
        configuration: UserConfiguration,
        *,
        workspace_id: str | None = None,
        allow_agent_home_chat: bool = False,
    ) -> None:
        self.service = service
        self.workspace_path = workspace_path
        self.configuration = configuration
        self.workspace_id = workspace_id or str(uuid4())
        self.allow_agent_home_chat = allow_agent_home_chat
        self.workspace_state: Any = None
        self._restore_result: RestoreResult | None = None
        self._mcp_manager: MCPWorkspaceRuntimeManager | None = None
        self._mcp_startup_report: Any = None
        self._mcp_snapshot: tuple[Any, ...] = ()
        self._mcp_keywords: Mapping[str, tuple[str, ...]] = {}
        self._mcp_keyword_preparer: MCPKeywordPreparer | None = None
        self._router: ModelRouter | None = None
        self._memory_manager: MemoryManager | None = None
        self._dream: Dream | None = None
        self._schedule_service: ScheduleService | None = None
        self._loops: dict[str, _LoopState] = {}
        self._retained_session_closes: set[str] = set()
        self._claims: dict[str, SessionClaim] = {}
        self._deletion_claims: dict[str, SessionDeletionClaim] = {}
        self._claim_versions: dict[str, int] = {}
        self._draft_clients: dict[str, str] = {}
        self._schedule_loops: dict[str, _LoopState] = {}
        self._restore_plans: dict[tuple[str, int], Any] = {}
        self._restore_results: dict[tuple[str, str], Any] = {}
        self._restore_owner: str | None = None
        self._restore_session_id: str | None = None
        self._restore_loop: SessionExecution | None = None
        self._restore_schedule_paused = False
        self._restore_blocked = False
        self._restore_commit_task: asyncio.Task[Any] | None = None
        self._subagent_repositories: dict[str, SubAgentRecordRepository] = {}
        self._foreground_work_stores: dict[str, ForegroundWorkStore] = {}
        self._subagent_coordinators: dict[str, SubAgentSessionCoordinator] = {}
        self._subagent_admission_closed = False
        self._subagent_blocked_job_ids: set[str] = set()
        self._subagent_shutdown_failed = False
        self._schedule_permission = RuntimePermissionControl(configuration.runtime.permission_level)
        self._exec_host: ExecHost | None = None
        self._started = False
        self._closed = False
        self._close_failed = False
        self._schedule_admitted = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self.service._activate_workspace(self)

    @property
    def resources(self) -> WorkspaceResources:
        return self.service.workspace_resources.get(self.workspace_id)

    @property
    def schedule_service(self) -> ScheduleService:
        return self.resources.schedule_service if self.workspace_id in self.service.workspace_resources.resources else cast(ScheduleService, self._schedule_service)

    @property
    def memory_manager(self) -> MemoryManager:
        return self.resources.memory_manager

    @property
    def dream(self) -> Dream:
        return self.resources.dream

    def subagent_repository(self, session_id: str) -> SubAgentRecordRepository:
        """Return the Workspace-owned record repository for one Session."""
        repository = self._subagent_repositories.get(session_id)
        if repository is None:
            repository = SubAgentRecordStore(self.workspace_state, session_id)
            self._subagent_repositories[session_id] = repository
        return repository

    def foreground_work_store(self, session_id: str) -> ForegroundWorkStore:
        store = self._foreground_work_stores.get(session_id)
        if store is None:
            store = ForegroundWorkStore(self.workspace_state, session_id)
            self._foreground_work_stores[session_id] = store
        return store

    def _committed_subagents(self, session_id: str) -> tuple[SubAgentRecord, ...]:
        repository = self.subagent_repository(session_id)
        records: list[SubAgentRecord] = []
        cursor = None
        while True:
            page = repository.list(cursor=cursor, limit=100)
            for item in page.items:
                record = repository.get(item.agent_id)
                if record is not None:
                    records.append(record)
            cursor = page.next_cursor
            if cursor is None:
                return tuple(records)

    def _persisted_foreground_messages(self, session_id: str) -> list[dict[str, Any]]:
        try:
            return Session.load(self.workspace_state, session_id).messages
        except FileNotFoundError:
            return []

    def register_subagent_coordinator(
        self,
        coordinator: SubAgentSessionCoordinator,
        repository: SubAgentRecordRepository | None = None,
    ) -> None:
        """Attach one Session pool without tying it to a Claim or Agent Run."""
        session_id = getattr(coordinator, "session_id", None)
        if not isinstance(session_id, str):
            raise TypeError("SubAgent coordinator must be Session-scoped")
        if self._subagent_admission_closed or self.service.state in {"draining", "stopped"}:
            raise SubAgentRequestError("Workspace is not accepting new SubAgent coordinators")
        if self._restore_blocked or self._restore_session_id == session_id:
            raise SubAgentRequestError("Session Restore is blocking SubAgent registration")
        Session._require_id(session_id)
        if not session_id.startswith("schedule_"):
            self._ensure_session_available(session_id)
        selected_repository = repository or self.subagent_repository(session_id)
        if selected_repository.session_id != session_id:
            raise ValueError("SubAgent repository belongs to a different Session")
        existing = self._subagent_coordinators.get(session_id)
        if existing is not None and existing is not coordinator:
            raise RuntimeError("A SubAgent coordinator is already registered for this Session")
        for job_id in self._subagent_blocked_job_ids:
            coordinator.block_source(job_id)
        self._subagent_repositories[session_id] = selected_repository
        self._subagent_coordinators[session_id] = coordinator

    def _subagent_coordinator(self, session_id: str) -> SubAgentSessionCoordinator | None:
        return self._subagent_coordinators.get(session_id)

    def close_subagent_admission(self) -> None:
        """Fence every Session before any asynchronous Workspace cleanup starts."""
        self._subagent_admission_closed = True
        for coordinator in tuple(self._subagent_coordinators.values()):
            coordinator.close_admission()

    def block_subagent_source(self, job_id: str) -> None:
        """Fence a removing Job in both existing and subsequently registered pools."""
        self._subagent_blocked_job_ids.add(job_id)
        for coordinator in tuple(self._subagent_coordinators.values()):
            coordinator.block_source(job_id)

    def unblock_subagent_source(self, job_id: str) -> None:
        """Release the Job fence after its removal fails before deletion."""
        self._subagent_blocked_job_ids.discard(job_id)
        for coordinator in tuple(self._subagent_coordinators.values()):
            coordinator.unblock_source(job_id)

    def _require_subagent_session(self, session_id: str) -> None:
        try:
            Session._require_id(session_id)
            if not session_id.startswith("schedule_"):
                self._ensure_session_available(session_id)
            state = self._loops.get(session_id)
            if state is not None:
                session = state.loop.session
                if (
                    session.session_id != session_id
                    or session.workspace_state is not self.workspace_state
                ):
                    raise service_error(
                        "not_found", "Conversation Session was not found.", status=404
                    )
                return
            Session.load(self.workspace_state, session_id)
        except FileNotFoundError as error:
            raise service_error("not_found", "Conversation Session was not found.", status=404) from error
        except ValueError as error:
            raise service_error(
                "validation_error", "Conversation Session ID is invalid.", status=422
            ) from error
        except OSError as error:
            raise service_error(
                "persistence_error",
                "Conversation Session could not be read safely.",
                status=500,
                retryable=True,
            ) from error

    async def activate_schedule(self) -> None:
        await self.start()
        if (
            self._closed
            or self._restore_schedule_paused
            or self._restore_blocked
            or not self.service._schedule_admission_open()
            or not self.service._schedule_allowed(self)
        ):
            return
        self._schedule_admitted = True
        self.schedule_service.resume()
        self.schedule_service.start()

    @property
    def schedule_admitted(self) -> bool:
        """Return whether this Workspace may admit new Schedule occurrences."""
        return (
            self._schedule_admitted
            and not self._closed
            and not self._restore_schedule_paused
            and not self._restore_blocked
            and not self.schedule_service.admission_paused
            and self.service._schedule_admission_open()
            and self.service._schedule_allowed(self)
        )

    def schedule_status(self) -> dict[str, object]:
        """Return the live Schedule projection for this Workspace."""
        status = self.schedule_service.status_snapshot().to_dict()
        return {
            "admitted": self.schedule_admitted,
            "status": status["status"],
            "active_job_count": status["active_job_count"],
        }

    def _capture_schedule_permission_snapshot(self) -> PermissionSnapshot:
        if self._exec_host is None:
            raise RuntimeError("Workspace service runtime is not ready")
        return PermissionSnapshot(
            level=self._schedule_permission.configured(),
            exec_shell=self._exec_host.resolved_shell,
        )

    async def pause_schedule_admission(self) -> None:
        if not self._started or self._closed:
            return
        self._schedule_admitted = False
        await self.schedule_service.pause_admission()

    async def create_draft(
        self,
        client_id: str,
        *,
        reuse_startup_session: bool = True,
        creation_scope: Literal["chat", "project"],
    ) -> str:
        if self._closed:
            raise service_error("admission_closed", "Workspace admission is closed.")
        if creation_scope not in {"chat", "project"}:
            raise ValueError("Session creation scope must be chat or project")
        startup_session_id = (
            None if self._restore_result is None else self._restore_result.session_id
        )
        if (
            reuse_startup_session
            and startup_session_id is not None
            and startup_session_id not in self._loops
        ):
            loop_state = await self._create_loop(startup_session_id, client_id=client_id)
            is_draft = False
        else:
            loop_state = await self._create_loop(None, client_id=client_id)
            is_draft = True
        session_id = loop_state.loop.session.session_id
        if is_draft:
            loop_state.loop.session.update_metadata(creation_scope=creation_scope)
            self._draft_clients[session_id] = client_id
        return session_id

    def _ensure_session_available(self, session_id: str) -> None:
        try:
            Session._require_id(session_id, partition=SessionStoragePartition.FOREGROUND)
            pending = session_deletion_pending(self.workspace_state, session_id)
        except ValueError as error:
            raise service_error(
                "validation_error",
                "Conversation Session ID is invalid.",
                status=422,
            ) from error
        except OSError as error:
            raise service_error(
                "persistence_error",
                "Conversation Session deletion state could not be read safely.",
                status=500,
                retryable=True,
            ) from error
        if pending:
            raise service_error(
                "session_deleting",
                "Conversation Session deletion is already in progress; retry shortly.",
                retryable=True,
            )

    async def claim(self, client_id: str, session_id: str) -> SessionClaim:
        async with self._lock:
            if self._closed:
                raise service_error("admission_closed", "Workspace admission is closed.")
            self._ensure_session_available(session_id)
            existing = self._claims.get(session_id)
            if existing is not None:
                if existing.client_id != client_id:
                    raise service_error(
                        "session_claimed",
                        "Conversation Session is already claimed by another client.",
                        retryable=True,
                    )
                if existing.status != "claimed":
                    raise service_error(
                        "stale_claim",
                        "Conversation Session Claim is not active.",
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
                try:
                    Session.load(
                        self.workspace_state,
                        session_id,
                        partition=SessionStoragePartition.FOREGROUND,
                        now=local_now,
                    )
                except FileNotFoundError as error:
                    raise service_error(
                        "not_found", "Conversation Session was not found.", status=404
                    ) from error
                except (OSError, UnicodeError, ValueError) as error:
                    raise service_error(
                        "persistence_error",
                        "Conversation Session could not be loaded safely.",
                        status=500,
                    ) from error
                loop_state = await self._create_loop(session_id, client_id=client_id)
            elif loop_state.owner_client_id not in {None, client_id}:
                raise service_error(
                    "session_claimed",
                    "Conversation Session is already claimed by another client.",
                    retryable=True,
                )
            if not loop_state.loop.has_active_run and not loop_state.run_ids:
                self.foreground_work_store(session_id).reconcile(
                    self._persisted_foreground_messages(session_id),
                    subagents=self._committed_subagents(session_id),
                )
            loop_state.owner_client_id = client_id
            loop_state.foreground_permission_control = self.service.client_permission(client_id)
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

    def set_client_connection(self, client_id: str, *, connected: bool) -> None:
        """Move this client's Claims across the reconnect grace boundary."""
        for claim in self._claims.values():
            if claim.client_id != client_id:
                continue
            if connected:
                if claim.status == "reconnecting":
                    claim.status = "claimed"
                    claim.disconnected_at = None
            elif claim.status == "claimed":
                claim.status = "reconnecting"
                claim.disconnected_at = self.service._monotonic()

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

    async def release(self, client_id: str, session_id: str) -> None:
        async with self._lock:
            await self._release_unlocked(client_id, session_id)

    async def _release_unlocked(
        self, client_id: str, session_id: str
    ) -> None:
        if self._restore_owner == client_id and self._restore_session_id == session_id:
            await self._wait_restore_commit()
        if self._restore_owner == client_id and self._restore_session_id == session_id:
            await self._release_restore_barrier(client_id)
        claim = self._claims.get(session_id)
        if claim is None:
            return
        if claim.client_id != client_id:
            raise service_error(
                "stale_claim", "Conversation Session Claim is not owned by this client."
            )
        if session_deletion_pending(self.workspace_state, session_id):
            raise service_error(
                "session_deleting",
                "Finish the pending Session deletion before releasing its Claim.",
                retryable=True,
            )
        loop_state = self._loops.get(session_id)
        if loop_state is not None and (
            loop_state.loop.has_active_run
            or loop_state.run_ids
            or loop_state.output_task is not None
            or await loop_state.bus.inbound_snapshot()
        ):
            raise service_error(
                "session_busy", "Conversation Session still has accepted work.", retryable=True
            )
        if loop_state is not None:
            try:
                await loop_state.loop.wait_for_restore_idle()
                await loop_state.loop.session.persist_pending_automatic_title()
            except RuntimeError:
                pass
        self._claims.pop(session_id, None)
        self.service.client_claim_released(client_id, self.workspace_id, session_id)
        await self.service.emit(
            "session.released",
            workspace_id=self.workspace_id,
            session_id=session_id,
            run_id=None,
            payload={},
            target_client_ids=self.service.workspace_audience(self.workspace_id),
        )
        if loop_state is not None and loop_state.owner_client_id == client_id:
            loop_state.owner_client_id = None

    def clear_claims_for_removal(self) -> tuple[SessionClaim, ...]:
        """Drop all service-owned Claims after the Workspace has been drained."""
        claims = tuple(self._claims.values())
        self._claims.clear()
        self._deletion_claims.clear()
        self._draft_clients.clear()
        for claim in claims:
            self.service.client_claim_released(
                claim.client_id,
                self.workspace_id,
                claim.session_id,
            )
        return claims

    async def expire_client(self, client_id: str) -> None:
        if self._restore_owner == client_id:
            await self._wait_restore_commit()
        for session_id, deletion_claim in tuple(self._deletion_claims.items()):
            if deletion_claim.client_id == client_id:
                self._deletion_claims.pop(session_id, None)
        if self._restore_owner == client_id:
            await self._release_restore_barrier(client_id)
        owned_claims = tuple(
            claim for claim in self._claims.values() if claim.client_id == client_id
        )
        for claim in owned_claims:
            claim.status = "draining"
        errors: list[Exception] = []
        for claim in owned_claims:
            session_id = claim.session_id
            loop_state = self._loops.get(session_id)
            retain_foreground = (
                loop_state is not None
                and self.service.foreground_continuation_admission_open()
                and self._has_resumable_foreground_work(session_id)
            )
            try:
                if loop_state is not None:
                    await self.service.confirmation.cancel_generation(loop_state.loop.generation_id)
                    if retain_foreground:
                        loop_state.owner_client_id = None
                    else:
                        try:
                            await loop_state.loop.cancel_active_run()
                        except RuntimeError:
                            pass
                        self._retained_session_closes.add(session_id)
                        await self._close_loop(session_id)
            except Exception as error:
                errors.append(error)
                continue
            self._claims.pop(session_id, None)
            self.service.client_claim_released(client_id, self.workspace_id, session_id)
            await self.service.emit(
                "session.released",
                workspace_id=self.workspace_id,
                session_id=session_id,
                run_id=None,
                payload={},
                target_client_ids=self.service.workspace_audience(self.workspace_id),
            )
        for session_id, owner in tuple(self._draft_clients.items()):
            if owner == client_id:
                self._draft_clients.pop(session_id, None)
                await self._close_loop(session_id, abort=True)
        if errors:
            raise ExceptionGroup("Client Session cleanup failed", errors)

    def _has_resumable_foreground_work(self, session_id: str) -> bool:
        """Keep an expired Client's loop only while an authorized Task remains live."""
        if self._restore_blocked or self._restore_session_id == session_id:
            return False
        try:
            if session_deletion_pending(self.workspace_state, session_id):
                return False
            records = self._committed_subagents(session_id)
            snapshot = self.foreground_work_store(session_id).load()
        except (OSError, UnicodeError, ValueError, SubAgentStoreError):
            return False
        task_states = {
            (task.task_id, task.generation): task.state for task in snapshot.tasks
        }
        resumable_states = {"running", "waiting_for_answer"}
        continuable_states = resumable_states | {"completed"}
        delivery_states = {
            (delivery.agent_id, delivery.terminal_record_revision, delivery.task_generation):
            delivery.state
            for delivery in snapshot.deliveries
        }
        if any(
            delivery.state in {"pending", "claimed"}
            and task_states.get((delivery.task_id, delivery.task_generation)) in continuable_states
            for delivery in snapshot.deliveries
        ):
            return True
        return any(
            record.source.kind is SubAgentSourceKind.FOREGROUND
            and record.source.task_id is not None
            and record.source.task_generation is not None
            and task_states.get((record.source.task_id, record.source.task_generation))
            in continuable_states
            and (
                record.status
                in {
                    SubAgentStatus.QUEUED,
                    SubAgentStatus.RUNNING,
                    SubAgentStatus.WAITING_FOR_ANSWER,
                }
                or (
                    record.status in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
                    and delivery_states.get(
                        (
                            record.agent_id,
                            record.revision,
                            record.source.task_generation,
                        )
                    )
                    not in ("committed", "invalidated")
                )
            )
            for record in records
        )

    def confirmation_source(
        self, owner: ConfirmationOwner
    ) -> tuple[str | None, str | None, str | None]:
        if isinstance(owner, SubAgentConfirmationOwner):
            if owner.workspace_id != self.workspace_id:
                return None, None, None
            try:
                record = self.subagent_repository(owner.session_id).get(owner.agent_id)
            except (OSError, RuntimeError, ValueError):
                return None, None, None
            if record is not None and record.session_id == owner.session_id:
                return self.workspace_id, owner.session_id, None
            return None, None, None
        for session_id, claim in self._claims.items():
            if claim.loop.generation_id == owner.generation_id:
                run_id = (
                    str(owner.run_id) if isinstance(owner, ForegroundConfirmationOwner) else None
                )
                return self.workspace_id, session_id, run_id
        if isinstance(owner, BackgroundConfirmationOwner):
            schedule_loop = self._schedule_loops.get(owner.job_id)
            if (
                schedule_loop is not None
                and schedule_loop.loop.generation_id == owner.generation_id
            ):
                return self.workspace_id, None, None
        return None, None, None

    def projection(self, session_id: str) -> ForegroundConversationProjection:
        self._ensure_session_available(session_id)
        loop = self._loops.get(session_id)
        if loop is None:
            raise service_error("not_found", "Conversation Session was not found.", status=404)
        return loop.loop.project_foreground_conversation()

    def session_model_available(self, session_id: str) -> bool:
        selection = self._loops[session_id].loop.session.model_configuration
        if selection is None:
            return True
        try:
            self.configuration.resolve_session_model_route(
                selection.provider_id, selection.model, selection.reasoning_effort
            )
        except (ConfigError, ValueError):
            return False
        return True

    def session_snapshot(self, session_id: str) -> dict[str, object]:
        """Return the claimed Session conversation plus presentation-safe Restore Anchors."""
        self._ensure_session_available(session_id)
        loop_state = self._loops.get(session_id)
        if loop_state is None:
            raise service_error("not_found", "Conversation Session was not found.", status=404)
        work_store = self.foreground_work_store(session_id)
        if not loop_state.loop.has_active_run and not loop_state.run_ids and not loop_state.live_runs:
            work_store.reconcile(self._persisted_foreground_messages(session_id),
                            subagents=self._committed_subagents(session_id))
        foreground_work = work_store.load()
        projection = loop_state.loop.project_foreground_conversation()
        messages = list(projection.messages)
        anchors = list(loop_state.loop.session.restore_candidates())
        if loop_state.live_runs and loop_state.completed_user_count is not None:
            user_count = 0
            for index, message in enumerate(messages):
                if message.get("role") == "user":
                    user_count += 1
                    if user_count > loop_state.completed_user_count:
                        messages = messages[:index]
                        visible_anchor_ids = {
                            record.get("restore_anchor_id")
                            for record in loop_state.loop.session.messages[:index]
                        }
                        anchors = [anchor for anchor in anchors
                                   if anchor.anchor_id in visible_anchor_ids]
                        break
        claim = self._claims.get(session_id)
        client = self.service._clients.get(claim.client_id) if claim is not None else None
        live_state = None
        if client is not None:
            runs = deepcopy(list(loop_state.live_runs.values()))
            for run in runs:
                run["cancellable"] = (
                    bool(loop_state.run_ids) and loop_state.run_ids[0] == run["run_id"]
                    and loop_state.loop.has_active_run and claim is not None
                    and claim.status != "draining" and not run["cancel_requested"]
                )
            live_state = {"stream_id": client.stream_id, "seq": client.sequence, "runs": runs}
        return {
            "session_id": projection.session_id,
            "messages": messages,
            "live_state": live_state,
            "model_configuration": (
                None
                if loop_state.loop.session.model_configuration is None
                else loop_state.loop.session.model_configuration.to_dict()
            ),
            "model_configuration_version": loop_state.loop.session.model_configuration_version,
            "model_configuration_available": self.session_model_available(session_id),
            "active_model_configuration": (
                None
                if loop_state.loop.active_model_configuration is None
                else loop_state.loop.active_model_configuration.to_dict()
            ),
            "restore_anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "run_token": str(anchor.run_token),
                    "content": anchor.content,
                    "timestamp": anchor.timestamp,
                }
                for anchor in anchors
            ],
            "foreground_questions": [
                {
                    "question_id": question.question_id,
                    "task_id": question.task_id,
                    "owner_agent_id": question.owner_agent_id,
                    "generation": question.generation,
                    "text": question.text,
                    "options": [dict(option) for option in question.options],
                    "state": question.state,
                    "answer": question.answer,
                    "answer_kind": question.answer_kind,
                    "answer_option_id": question.answer_option_id,
                    "revision": foreground_work.revision,
                }
                for question in foreground_work.questions
                if question.state in {"pending", "answer_received", "interrupted"}
            ],
            "foreground_tasks": [
                {
                    "task_id": task.task_id,
                    "generation": task.generation,
                    "state": task.state,
                    "revision": foreground_work.revision,
                }
                for task in foreground_work.tasks
            ],
        }

    def _session_summary(self, session: Session, client_id: str) -> dict[str, object]:
        claim = self._claims.get(session.session_id)
        title = session.metadata.get("title", "Untitled session")
        return {
            "id": session.session_id,
            "title": title if isinstance(title, str) else "Untitled session",
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "occupied": claim is not None,
            "occupied_by": None if claim is None or claim.client_id == client_id else "client",
            "metadata_version": session.metadata_version,
        }

    async def list_sessions_page(
        self,
        client_id: str,
        *,
        title: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        creation_scope: str | None = None,
    ) -> dict[str, object]:
        """Return a filtered page of durable foreground Session metadata."""
        if limit is not None and (limit < 1 or limit > _MAX_SESSION_PAGE_SIZE):
            raise service_error(
                "validation_error",
                f"limit must be between 1 and {_MAX_SESSION_PAGE_SIZE}.",
                status=422,
            )
        title_filter = "" if title is None else title.strip().casefold()
        cursor_key = (
            None
            if cursor is None
            else _decode_session_cursor(cursor, self.workspace_id, title_filter)
        )
        directory = self.workspace_state.existing_sessions_directory()
        if directory is None:
            return {"sessions": [], "next_cursor": None}

        entries: list[tuple[tuple[datetime, datetime, str], dict[str, object]]] = []
        for path in directory.glob("*.jsonl"):
            try:
                if session_deletion_pending(self.workspace_state, path.stem):
                    continue
            except (OSError, ValueError):
                continue
            loop_state = self._loops.get(path.stem)
            session = None if loop_state is None else loop_state.loop.session
            if session is None:
                try:
                    session = Session.load(
                        self.workspace_state,
                        path.stem,
                        partition=SessionStoragePartition.FOREGROUND,
                        now=local_now,
                    )
                except (OSError, UnicodeError, ValueError):
                    continue
            summary = self._session_summary(session, client_id)
            if creation_scope is not None:
                session_scope = session.metadata.get("creation_scope")
                if session_scope != creation_scope:
                    continue
            session_title = cast(str, summary["title"])
            if title_filter and title_filter not in session_title.casefold():
                continue
            key = (session.updated_at, session.created_at, session.session_id)
            if cursor_key is not None and key >= cursor_key:
                continue
            entries.append((key, summary))

        entries.sort(key=lambda item: item[0], reverse=True)
        if limit is None:
            page = entries
            next_cursor = None
        else:
            page = entries[:limit]
            next_cursor = (
                _encode_session_cursor(page[-1][0], self.workspace_id, title_filter)
                if len(page) < len(entries) and page
                else None
            )
        return {
            "sessions": [summary for _, summary in page],
            "next_cursor": next_cursor,
        }

    async def list_sessions(self, client_id: str) -> list[dict[str, object]]:
        """Return all durable foreground Sessions owned by this Workspace."""
        page = await self.list_sessions_page(client_id)
        return cast(list[dict[str, object]], page["sessions"])

    async def rename_session(
        self,
        client_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        title: str,
        expected_metadata_version: int,
        request_id: str,
    ) -> dict[str, object]:
        """Rename one claimed, already-persisted foreground Session."""
        async with self._lock:
            claim = self.require_claim(client_id, session_id, claim_version, claim_credential)
            self._ensure_session_available(session_id)
            client = self.service.client(client_id)
            fingerprint = (self.workspace_id, session_id, title, expected_metadata_version)
            previous = client.rename_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error(
                        "validation_error",
                        "request_id was already used for another rename.",
                        status=422,
                    )
                return previous[1]
            session = claim.loop.session
            if not session.messages:
                raise service_error(
                    "session_not_persisted",
                    "Conversation Session draft has no accepted input yet.",
                    retryable=True,
                )
            await session.wait_for_pending_persist()
            self.require_claim(client_id, session_id, claim_version, claim_credential)
            self._ensure_session_available(session_id)
            directory = self.workspace_state.existing_sessions_directory()
            session_path = None if directory is None else directory / f"{session_id}.jsonl"
            if session_path is None or not session_path.is_file():
                raise service_error(
                    "session_not_persisted",
                    "Conversation Session is not persisted yet.",
                    retryable=True,
                )
            try:
                session.rename_durably(
                    title,
                    expected_metadata_version=expected_metadata_version,
                )
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "Conversation Session title could not be saved.",
                    status=500,
                    retryable=True,
                ) from error
            except ValueError as error:
                message = str(error)
                if "stale" in message:
                    raise service_error(
                        "metadata_conflict",
                        "Conversation Session metadata changed; reload before renaming.",
                        retryable=True,
                    ) from error
                raise service_error("validation_error", message, status=422) from error
            summary = self._session_summary(session, client_id)
            client.rename_results[request_id] = (fingerprint, summary)
            await self.service.emit(
                "session.metadata_updated",
                workspace_id=self.workspace_id,
                session_id=session_id,
                run_id=None,
                payload={
                    "title": summary["title"],
                    "metadata_version": summary["metadata_version"],
                },
                target_client_ids=self.service.workspace_audience(self.workspace_id),
            )
            return summary

    async def delete_session(
        self,
        client_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
        request_id: str,
    ) -> dict[str, object]:
        """Fence, drain, and delete one claimed foreground Session atomically."""
        async with self._lock:
            client = self.service.client(client_id)
            fingerprint = (self.workspace_id, session_id, claim_version, claim_credential)
            previous = client.session_delete_results.get(request_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise service_error(
                        "validation_error",
                        "request_id was already used for another Session deletion.",
                        status=422,
                    )
                return previous[1]

            try:
                Session._require_id(session_id, partition=SessionStoragePartition.FOREGROUND)
                pending = session_deletion_pending(self.workspace_state, session_id)
            except ValueError as error:
                raise service_error(
                    "validation_error",
                    "Conversation Session ID is invalid.",
                    status=422,
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "Conversation Session deletion state could not be read safely.",
                    status=500,
                    retryable=True,
                ) from error

            claim = self._claims.get(session_id)
            if claim is not None or not pending:
                claim = self.require_claim(client_id, session_id, claim_version, claim_credential)
            else:
                cleanup_claim = self._deletion_claims.get(session_id)
                if (
                    cleanup_claim is None
                    or cleanup_claim.client_id != client_id
                    or cleanup_claim.version != claim_version
                    or cleanup_claim.credential != claim_credential
                ):
                    raise service_error(
                        "stale_claim", "Session deletion Claim is missing or stale."
                    )
            try:
                restore_pending = session_restore_pending(self.workspace_state, session_id)
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "Session Restore state could not be read safely.",
                    status=500,
                    retryable=True,
                ) from error
            if (
                self._restore_owner is not None
                or self._restore_commit_task is not None
                or self._restore_blocked
                or restore_pending
            ):
                raise service_error(
                    "restore_pending",
                    "Finish or cancel the active Session Restore before deleting.",
                    retryable=True,
                )

            loop_state = self._loops.get(session_id)
            if loop_state is not None and (
                loop_state.loop.has_active_run
                or loop_state.run_ids
                or loop_state.output_task is not None
                or await loop_state.bus.inbound_snapshot()
            ):
                raise service_error(
                    "session_busy",
                    "Conversation Session still has accepted work.",
                    retryable=True,
                )
            if not pending and (claim is None or not claim.loop.session.messages):
                try:
                    directory = self.workspace_state.existing_sessions_directory()
                    if directory is None or not HOST_FILESYSTEM.entry_exists(
                        directory / f"{session_id}.jsonl"
                    ):
                        raise service_error(
                            "session_not_persisted",
                            "Conversation Session draft has no accepted input yet.",
                            retryable=True,
                        )
                    HOST_FILESYSTEM.require_owned_regular_file(
                        directory / f"{session_id}.jsonl", within=directory
                    )
                except OSError as error:
                    raise service_error(
                        "persistence_error",
                        "Session history could not be read safely.",
                        status=500,
                        retryable=True,
                    ) from error

            coordinator = self._subagent_coordinator(session_id)
            if coordinator is not None:
                coordinator.close_admission()
                if coordinator.has_active():
                    if not self._subagent_admission_closed:
                        coordinator.open_admission()
                    raise service_error(
                        "session_busy",
                        "Conversation Session still has active SubAgent work.",
                        retryable=True,
                    )
            deletion_started = False
            try:
                begin_session_deletion(self.workspace_state, session_id)
                deletion_started = True
                self.foreground_work_store(session_id).close()
                if loop_state is not None:
                    await loop_state.loop.wait_for_restore_idle()
                    if (
                        loop_state.loop.has_active_run
                        or loop_state.run_ids
                        or loop_state.output_task is not None
                        or await loop_state.bus.inbound_snapshot()
                    ):
                        raise service_error(
                            "session_busy",
                            "Conversation Session still has accepted work.",
                            retryable=True,
                        )
                    await self._close_loop(session_id)
                delete_session_data(self.workspace_state, session_id)
                self.service.request_trace_session_purge(
                    workspace_key_for(self.workspace_state.workspace_path),
                    session_id,
                )
            except ServiceError:
                if (
                    coordinator is not None
                    and not deletion_started
                    and not self._subagent_admission_closed
                ):
                    coordinator.open_admission()
                raise
            except (OSError, RuntimeError, ValueError) as error:
                if (
                    coordinator is not None
                    and not deletion_started
                    and not self._subagent_admission_closed
                ):
                    coordinator.open_admission()
                raise service_error(
                    "persistence_error",
                    "Conversation Session data could not be deleted safely; retry the operation.",
                    status=500,
                    retryable=True,
                ) from error

            self._claims.pop(session_id, None)
            self._deletion_claims.pop(session_id, None)
            self._draft_clients.pop(session_id, None)
            self._subagent_coordinators.pop(session_id, None)
            self._subagent_repositories.pop(session_id, None)
            self.service.client_claim_released(client_id, self.workspace_id, session_id)
            response = {
                "workspace_id": self.workspace_id,
                "session_id": session_id,
                "deleted": True,
            }
            client.session_delete_results[request_id] = (fingerprint, response)
            try:
                await self.service.emit(
                    "session.deleted",
                    workspace_id=self.workspace_id,
                    session_id=session_id,
                    run_id=None,
                    payload={"deleted": True},
                    target_client_ids=self.service.workspace_audience(self.workspace_id),
                )
            except Exception:
                pass
            return response

    async def _release_restore_barrier(self, client_id: str) -> None:
        if self._restore_owner != client_id:
            return
        loop = self._restore_loop
        session_id = self._restore_session_id
        self._restore_owner = None
        self._restore_session_id = None
        self._restore_loop = None
        for key in tuple(self._restore_plans):
            if key[0] == client_id:
                self._restore_plans.pop(key, None)
        if loop is not None:
            await loop._release_replacement_barrier(resume_inbound=True)
        if (
            session_id is not None
            and not self._restore_blocked
            and not self._subagent_admission_closed
        ):
            coordinator = self._subagent_coordinator(session_id)
            if coordinator is not None:
                coordinator.open_admission()
        if self._restore_schedule_paused:
            self._restore_schedule_paused = False
            if (
                not self._closed
                and not self._restore_blocked
                and self.service._schedule_admission_open()
                and self.service._schedule_allowed(self)
            ):
                self._schedule_admitted = True
                self.schedule_service.resume()
                self.schedule_service.start()

    async def _wait_restore_commit(self) -> None:
        task = self._restore_commit_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    async def _rebuild_restored_session(self, client_id: str, session_id: str) -> SessionClaim:
        claim = self._claims.get(session_id)
        if claim is None or claim.client_id != client_id:
            raise service_error("stale_claim", "Conversation Session Claim is missing or stale.")
        previous_store = self._foreground_work_stores.pop(session_id, None)
        if previous_store is not None:
            previous_store.close()
        await self._close_loop(session_id, abort=True)
        state = await self._create_loop(session_id, client_id=client_id)
        version = self._claim_versions.get(session_id, 0) + 1
        self._claim_versions[session_id] = version
        claim.loop = state.loop
        claim.version = version
        claim.credential = str(uuid4())
        return claim

    def management_dispatcher(
        self,
        client_id: str,
        session_id: str,
        claim_version: int | None = None,
        claim_credential: str | None = None,
        *,
        resume_context: dict[str, object] | None = None,
    ) -> Any:
        """Build the existing typed Management dispatcher for one Claim."""
        from aide.management.commands import ManagementCommandDispatcher
        from aide.management.service import (
            ManagementError,
            ManagementViewService,
            RestoreListingReport,
        )

        runtime = self.resources
        if not self._started:
            raise RuntimeError("Workspace service runtime is not ready")
        owned_claim = self._claims.get(session_id)
        if owned_claim is None:
            raise service_error("stale_claim", "Conversation Session Claim is missing or stale.")
        initial_claim = self.require_claim(
            client_id,
            session_id,
            owned_claim.version if claim_version is None else claim_version,
            claim_credential,
        )
        initial_version = initial_claim.version
        initial_credential = initial_claim.credential

        def current_loop() -> SessionExecution:
            self.service._require_client(client_id)
            self._ensure_session_available(session_id)
            return self.require_claim(
                client_id, session_id, initial_version, initial_credential
            ).loop

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
            context = await self.service.claim(client_id, self.workspace_id, target_session_id)
            if resume_context is not None:
                resume_context.update(context)

        async def restore_listing() -> RestoreListingReport:
            from aide.management.service import RestoreListingReport

            async with self._lock:
                self._require_admitted()
                loop = current_loop()
                state = self._loops[loop.session.session_id]
                if (
                    self._restore_owner is not None
                    or loop.has_active_run
                    or state.run_ids
                    or state.output_task is not None
                    or await state.bus.inbound_snapshot()
                ):
                    raise ManagementError(
                        ErrorInfo(
                            "model_invalid_request",
                            "Finish the active run and clear queued input before restoring.",
                        )
                    )
                coordinator = self._subagent_coordinator(loop.session.session_id)
                if coordinator is not None:
                    coordinator.close_admission()
                    try:
                        if coordinator.has_active():
                            raise ManagementError(
                                ErrorInfo(
                                    "model_invalid_request",
                                    "Cancel active SubAgent work before restoring this Session.",
                                )
                            )
                    except BaseException:
                        if not self._subagent_admission_closed:
                            coordinator.open_admission()
                        raise
                self._restore_owner = client_id
                self._restore_session_id = loop.session.session_id
                self._restore_loop = loop
            try:
                await loop._pause_for_replacement()
                await loop.wait_for_restore_idle()
                current_loop()
                anchors = loop.session.restore_candidates()
                if not anchors:
                    await self._release_restore_barrier(client_id)
                return RestoreListingReport(loop.session.session_id, tuple(reversed(anchors)))
            except BaseException:
                await self._release_restore_barrier(client_id)
                raise

        async def restore_inspect(anchor_id: int) -> Any:
            from aide.agent.session.restore import RestoreManager

            loop = current_loop()
            if self._restore_owner != client_id or self._restore_loop is not loop:
                if self._restore_owner is not None:
                    raise ManagementError(
                        ErrorInfo("model_invalid_request", "Session Restore is not active.")
                    )
                await restore_listing()
                if self._restore_owner != client_id or self._restore_loop is not loop:
                    raise ManagementError(
                        ErrorInfo("model_invalid_request", "Session Restore is not active.")
                    )
            try:
                if not self._restore_schedule_paused:
                    self._restore_schedule_paused = True
                    await self.schedule_service.pause_and_wait_idle()
                current_loop()
                manager = RestoreManager(
                    self.workspace_state,
                    loop.session.session_id,
                    now=local_now,
                    subagent_repository=self.subagent_repository(loop.session.session_id),
                )
                plan = manager.inspect(loop.session, anchor_id)
                manager.revalidate(plan)
                self._restore_plans[(client_id, anchor_id)] = plan
                return plan
            except BaseException:
                await self._release_restore_barrier(client_id)
                raise

        async def restore_commit(anchor_id: int, mode: Any) -> Any:
            loop = current_loop()
            stored = self._restore_plans.get((client_id, anchor_id))
            if (
                stored is None
                or stored.session_id != loop.session.session_id
                or self._restore_owner != client_id
                or self._restore_loop is not loop
            ):
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "The selected Restore plan is stale; no changes were made.",
                    )
                )
            executed = False
            restore_task = asyncio.current_task()
            self._restore_commit_task = restore_task
            try:
                manager = RestoreManager(
                    self.workspace_state,
                    stored.session_id,
                    now=local_now,
                    subagent_repository=self.subagent_repository(stored.session_id),
                )
                manager.revalidate(stored)
                result = await manager.execute(stored, mode)
                executed = True
                self.service.mark_trace_restored(
                    workspace_key_for(self.workspace_state.workspace_path),
                    stored.session_id,
                    tuple(str(token) for token in stored.discarded_run_tokens),
                )
                await self._rebuild_restored_session(client_id, stored.session_id)
                restored_loop = self._loops[stored.session_id].loop
                self.foreground_work_store(stored.session_id).reconcile(
                    restored_loop.session.messages,
                    subagents=self._committed_subagents(stored.session_id),
                )
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
                try:
                    await self._release_restore_barrier(client_id)
                finally:
                    if self._restore_commit_task is restore_task:
                        self._restore_commit_task = None

        async def restore_result() -> Any:
            from aide.agent.session.restore import RestoreManager

            session_id = current_loop().session.session_id
            result = RestoreManager(
                self.workspace_state, session_id, now=local_now
            ).completed_result()
            return result

        async def restore_acknowledge_failure() -> Any:
            from aide.agent.session.restore import RestoreManager

            session_id = current_loop().session.session_id
            if await restore_result() is None:
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
            loop = current_loop()
            if self._restore_session_id == loop.session.session_id:
                await self._release_restore_barrier(client_id)

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
            reasoning_effort_control=self.service,
            permission_control=self.service.client_permission(client_id),
            restore_listing=restore_listing,
            restore_inspect=restore_inspect,
            restore_commit=restore_commit,
            restore_result=restore_result,
            restore_cancel=restore_cancel,
            ensure_management_mutation_allowed=lambda: self._require_admitted(
                allow_configuration=True
            ),
        )
        management.bind_runtime_admission(lambda: self._require_admitted())
        management.bind_reasoning_effort_persistence(self.service.persist_reasoning_effort)
        management.bind_configuration_status(self.service.configuration_status_text)
        management.bind_restore_acknowledge_failure(restore_acknowledge_failure)
        return ManagementCommandDispatcher(management)

    def _require_admitted(self, *, allow_configuration: bool = False) -> None:
        del allow_configuration
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
        request_id: str | None = None,
        *,
        foreground_question_id: str | None = None,
        foreground_task_id: str | None = None,
        foreground_task_continue: bool = False,
        foreground_task_generation: int | None = None,
        foreground_delivery: Mapping[str, object] | None = None,
    ) -> SessionClaim:
        async with self._lock:
            self._require_admitted()
            claim = self.require_claim(client_id, session_id, version)
            self._ensure_session_available(session_id)
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
            if not self.session_model_available(session_id):
                raise service_error(
                    "model_unavailable",
                    "Choose an available model before sending to this Session.",
                    status=422,
                    field_errors={"model": "choose an available model"},
                )
            state = self._loops[session_id]
            client = self.service.client(client_id)
            async with state.coordination_lock:
                if run_id in state.live_runs:
                    return claim
                claimed_delivery = None
                claimed_delivery_result: object = None
                if foreground_delivery is not None:
                    agent_id = foreground_delivery.get("agent_id")
                    record_revision = foreground_delivery.get("terminal_record_revision")
                    generation = foreground_delivery.get("task_generation")
                    if (
                        not isinstance(agent_id, str)
                        or isinstance(record_revision, bool)
                        or not isinstance(record_revision, int)
                        or isinstance(generation, bool)
                        or not isinstance(generation, int)
                        or generation != foreground_task_generation
                        or foreground_task_id is None
                    ):
                        raise service_error(
                            "validation_error", "SubAgent result continuation is invalid.", status=422
                        )
                    try:
                        claimed_delivery = self.foreground_work_store(session_id).claim_delivery(
                            agent_id=agent_id,
                            terminal_record_revision=record_revision,
                            task_generation=generation,
                            execution_id=run_id,
                            allow_reclaim=True,
                        )
                    except (OSError, UnicodeError, ValueError) as error:
                        raise service_error(
                            "persistence_error", "SubAgent result could not be claimed.", status=500
                        ) from error
                    if claimed_delivery is None:
                        raise service_error(
                            "foreground_task_conflict",
                            "The SubAgent result changed; reload the Task before continuing.",
                            status=409,
                        )
                    claimed_delivery_result = foreground_delivery.get("result")
                if foreground_task_continue:
                    if (
                        foreground_task_id is None
                        or isinstance(foreground_task_generation, bool)
                        or not isinstance(foreground_task_generation, int)
                    ):
                        raise service_error(
                            "validation_error", "Task continuation is invalid.", status=422
                        )
                    try:
                        self.foreground_work_store(session_id).start_execution(
                            foreground_task_id,
                            run_id,
                            expected_generation=foreground_task_generation,
                        )
                    except (ForegroundWorkConflict, ValueError) as error:
                        raise service_error(
                            "foreground_task_conflict", "The Task changed.", status=409
                        ) from error
                    except OSError as error:
                        raise service_error(
                            "persistence_error", "Task continuation could not be saved.", status=500
                        ) from error
                async with client.delivery_lock:
                    if not state.live_runs:
                        state.completed_user_count = sum(
                            message.get("role") == "user" for message in state.loop.session.messages
                        )
                    state.run_ids.append(run_id)
                    state.live_runs[run_id] = {
                        "run_id": run_id, "request_id": request_id or run_id, "prompt": text,
                        "status": "accepted", "assistant_content": "", "tools": [],
                        "response_segments": [""],
                        "cancel_requested": False, "cancellable": False,
                    }
                    inbound_metadata: dict[str, object] = {
                        "run_id": run_id,
                        "request_id": request_id or run_id,
                    }
                    if foreground_question_id is not None:
                        inbound_metadata["foreground_question_id"] = foreground_question_id
                    if foreground_task_id is not None:
                        inbound_metadata["foreground_task_id"] = foreground_task_id
                    if foreground_task_continue:
                        inbound_metadata["foreground_task_continue"] = True
                    if foreground_task_generation is not None:
                        inbound_metadata["foreground_task_generation"] = foreground_task_generation
                    if claimed_delivery is not None:
                        inbound_metadata.update(
                            {
                                "foreground_delivery_receipt_id": claimed_delivery.receipt_id,
                                "foreground_delivery_agent_id": claimed_delivery.agent_id,
                                "foreground_delivery_record_revision": claimed_delivery.terminal_record_revision,
                                "foreground_subagent_result": claimed_delivery_result,
                            }
                        )
                    await state.bus.put_inbound(
                        InboundMessage(
                            content=text,
                            metadata=inbound_metadata,
                        )
                    )
                self._ensure_processor(state)
            return claim

    async def answer_question(
        self,
        client_id: str,
        session_id: str,
        version: int,
        question_id: str,
        answer: str | None,
        choice_id: str | None,
        expected_revision: int,
        request_id: str,
    ) -> dict[str, object]:
        """Persist and queue one explicit answer for the current claimed Session."""
        self._require_admitted()
        claim = self.require_claim(client_id, session_id, version)
        self._ensure_session_available(session_id)
        state = self._loops[session_id]
        if not claim.loop.foreground_input_admitted():
            raise service_error("admission_closed", "Conversation input is temporarily unavailable.")
        if not self.session_model_available(session_id):
            raise service_error(
                "model_unavailable",
                "Choose an available model before sending to this Session.",
                status=422,
                field_errors={"model": "choose an available model"},
            )
        if self._restore_session_id == session_id or self._restore_commit_task is not None:
            raise service_error("session_restore_active", "Session Restore is in progress.", retryable=True)
        store = self.foreground_work_store(session_id)
        if not state.loop.has_active_run and not state.run_ids and not state.live_runs:
            store.reconcile(self._persisted_foreground_messages(session_id),
                            subagents=self._committed_subagents(session_id))
        try:
            store.answer_question(
                question_id,
                answer,
                choice_id=choice_id,
                expected_revision=expected_revision,
                request_id=request_id,
            )
            snapshot = store.load()
            question = snapshot.question(question_id)
            if question is None:
                raise ForegroundWorkConflict("question does not exist")
            if question.owner_agent_id is not None:
                return await self._resume_subagent_question(
                    client_id=client_id,
                    session_id=session_id,
                    state=state,
                    store=store,
                    question=question,
                )
            if question.state == "answered":
                run_id = question.resume_execution_id
                if run_id is None:
                    raise ForegroundWorkConflict("completed question has no resume execution")
                return {
                    "run_id": run_id,
                    "answer": question.answer,
                    "live_state": self.session_snapshot(session_id)["live_state"],
                }
            if question.state in {"resuming", "executing"}:
                run_id = question.resume_execution_id
                if run_id is None:
                    raise ForegroundWorkConflict("question resume has no execution")
            else:
                run_id = str(uuid4())
                store.begin_resume(question_id, run_id)
        except StaleForegroundWorkRevision as error:
            raise service_error(
                "foreground_question_conflict",
                "The question changed; reload it before answering.",
                status=409,
            ) from error
        except (ForegroundWorkConflict, ValueError) as error:
            raise service_error(
                "foreground_question_conflict",
                "The question is no longer available for this answer.",
                status=409,
            ) from error
        except OSError as error:
            raise service_error(
                "persistence_error",
                "The question answer could not be saved.",
                status=500,
            ) from error
        await self.input(
            client_id,
            session_id,
            version,
            cast(str, question.answer),
            run_id,
            question.answer_request_id or request_id,
            foreground_question_id=question_id,
            foreground_task_id=question.task_id,
        )
        return {
            "run_id": run_id,
            "answer": question.answer,
            "live_state": self.session_snapshot(session_id)["live_state"],
        }

    async def _resume_subagent_question(
        self,
        *,
        client_id: str,
        session_id: str,
        state: _LoopState,
        store: ForegroundWorkStore,
        question: Any,
    ) -> dict[str, object]:
        if state.loop.has_active_run or state.run_ids or state.live_runs:
            raise service_error(
                "foreground_run_active",
                "Wait for the Main Agent to release this Session before answering the SubAgent.",
                retryable=True,
            )
        session = state.loop.session
        answer_message = next(
            (
                message
                for message in reversed(session.messages)
                if message.get("role") == "user"
                and message.get("foreground_question_id") == question.question_id
            ),
            None,
        )
        run_id = question.resume_execution_id or str(uuid4())
        if answer_message is None:
            if not isinstance(question.answer, str):
                raise service_error(
                    "foreground_question_conflict",
                    "The SubAgent question has no saved answer.",
                    status=409,
                )
            restore_run_token = uuid4()
            summary = session.metadata.get("summary", "")
            session.commit_agent_run(
                [
                    {
                        "role": "user",
                        "content": f"SubAgent answer: {question.answer}",
                        "foreground_question_id": question.question_id,
                        "foreground_task_id": question.task_id,
                        "subagent_answer": True,
                    }
                ],
                pending_last_compacted=session.last_compacted,
                pending_action_summary=summary if isinstance(summary, str) else "",
                restore_before=session.capture_restore_before(),
                restore_run_token=restore_run_token,
            )
            answer_message = session.messages[-1]
        await session.wait_for_pending_persist()
        persisted_answer = next(
            (
                message
                for message in self._persisted_foreground_messages(session_id)
                if message.get("role") == "user"
                and message.get("foreground_question_id") == question.question_id
            ),
            None,
        )
        if (
            persisted_answer is None
            or persisted_answer.get("restore_anchor_id") != answer_message.get("restore_anchor_id")
            or persisted_answer.get("restore_run_token") != answer_message.get("restore_run_token")
        ):
            raise service_error(
                "persistence_error",
                "The SubAgent answer Restore anchor could not be saved.",
                status=500,
            )
        answer_message = persisted_answer
        answer_anchor_id = answer_message.get("restore_anchor_id")
        token_value = answer_message.get("restore_run_token")
        if (
            isinstance(answer_anchor_id, bool)
            or not isinstance(answer_anchor_id, int)
            or not isinstance(token_value, str)
        ):
            raise service_error(
                "persistence_error",
                "The SubAgent answer Restore anchor could not be saved.",
                status=500,
            )
        try:
            if question.state == "answer_received":
                store.begin_subagent_resume(
                    question.question_id,
                    run_id,
                    answer_anchor_id=answer_anchor_id,
                )
            record = self.subagent_repository(session_id).get(question.owner_agent_id)
            if record is None:
                raise SubAgentRequestError("SubAgent execution resource is unavailable")
            creator = record.creator_snapshot
            try:
                self.configuration.resolve_session_model_route(
                    creator.provider_id, creator.model, creator.reasoning_effort
                )
            except (ConfigError, ValueError) as error:
                raise SubAgentRequestError("SubAgent model resource is unavailable") from error
            coordinator = self._subagent_coordinators.get(session_id)
            resume = getattr(coordinator, "resume_for_answer", None)
            if not callable(resume):
                raise SubAgentRequestError("SubAgent execution resource is unavailable")
            assert self._exec_host is not None
            resume(
                question.owner_agent_id,
                question_id=question.question_id,
                answer=cast(str, question.answer),
                execution_id=run_id,
                restore_run_token=token_value,
                permission_level=self.service.client_permission(client_id).snapshot(
                    self._exec_host.resolved_shell
                ).level,
            )
        except (SubAgentRequestError, SubAgentStoreError):
            record = self.subagent_repository(session_id).get(question.owner_agent_id)
            if (
                record is not None
                and record.execution_id == run_id
                and record.status
                in {
                    SubAgentStatus.RUNNING,
                    SubAgentStatus.COMPLETED,
                    SubAgentStatus.FAILED,
                    SubAgentStatus.CANCELLED,
                }
            ):
                return {
                    "kind": "subagent_resumed",
                    "agent_id": question.owner_agent_id,
                    "run_id": run_id,
                    "answer": question.answer,
                    "live_state": self.session_snapshot(session_id)["live_state"],
                }
            if record is not None and record.status in {
                SubAgentStatus.WAITING_FOR_ANSWER, SubAgentStatus.QUEUED
            }:
                self.subagent_repository(session_id).save(
                    replace(
                        record,
                        status=SubAgentStatus.INTERRUPTED,
                        finished_at=local_now(),
                        error=SubAgentError(
                            code="resource_unavailable",
                            message="The saved SubAgent execution resource is unavailable.",
                        ),
                        revision=record.revision + 1,
                    ),
                    expected_revision=record.revision,
                )
            store.interrupt_subagent_question(question.question_id)
            return {
                "kind": "subagent_interrupted",
                "agent_id": question.owner_agent_id,
                "run_id": run_id,
                "answer": question.answer,
                "live_state": self.session_snapshot(session_id)["live_state"],
            }
        return {
            "kind": "subagent_resumed",
            "agent_id": question.owner_agent_id,
            "run_id": run_id,
            "answer": question.answer,
            "live_state": self.session_snapshot(session_id)["live_state"],
        }

    async def cancel_task(
        self,
        client_id: str,
        session_id: str,
        version: int,
        task_id: str,
        expected_revision: int,
        request_id: str,
    ) -> dict[str, object]:
        """Fence one foreground Task before cancelling its queued and active work."""
        self._require_admitted()
        self.require_claim(client_id, session_id, version)
        self._ensure_session_available(session_id)
        if self._restore_session_id == session_id or self._restore_commit_task is not None:
            raise service_error("session_restore_active", "Session Restore is in progress.", retryable=True)
        state = self._loops[session_id]
        store = self.foreground_work_store(session_id)
        if not state.loop.has_active_run and not state.run_ids and not state.live_runs:
            store.reconcile(self._persisted_foreground_messages(session_id),
                            subagents=self._committed_subagents(session_id))
        current = store.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None:
            raise service_error("foreground_task_conflict", "The Task is no longer available.", status=409)
        try:
            cancelled = store.cancel_task(
                task_id,
                expected_revision=expected_revision,
                request_id=request_id,
            )
        except StaleForegroundWorkRevision as error:
            raise service_error(
                "foreground_task_conflict", "The Task changed; reload it before cancelling.", status=409
            ) from error
        except (ForegroundWorkConflict, ValueError) as error:
            raise service_error(
                "foreground_task_conflict", "The Task can no longer be cancelled.", status=409
            ) from error
        except OSError as error:
            raise service_error("persistence_error", "The Task cancellation could not be saved.", status=500) from error

        cancelled_runs: list[str] = []
        active_run_id: str | None = None
        client = self.service.client(client_id)
        async with state.coordination_lock:
            async with client.delivery_lock:
                for inbound in await state.bus.inbound_snapshot():
                    if inbound.metadata.get("foreground_task_id") != task_id:
                        continue
                    queued_request_id = inbound.metadata.get("request_id")
                    run_id = inbound.metadata.get("run_id")
                    if not isinstance(queued_request_id, str) or not isinstance(run_id, str):
                        continue
                    if await state.bus.remove_inbound(queued_request_id) is None:
                        continue
                    try:
                        state.run_ids.remove(run_id)
                    except ValueError:
                        pass
                    state.live_runs.pop(run_id, None)
                    cancelled_runs.append(run_id)
                if not state.live_runs:
                    state.completed_user_count = None
                if (
                    state.loop.has_active_run
                    and state.run_ids
                    and state.run_ids[0] == task.execution_id
                ):
                    active_run_id = task.execution_id
                    if active_run_id in state.live_runs:
                        state.live_runs[active_run_id]["cancel_requested"] = True

        if active_run_id is not None:
            await state.loop.cancel_active_run()
            cancelled_runs.append(active_run_id)
        for run_id in cancelled_runs:
            await self.service.emit(
                "run.cancelled",
                workspace_id=self.workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload={},
                target_client_ids=(client_id,),
            )
        current_task = next(item for item in cancelled.tasks if item.task_id == task_id)
        coordinator = self._subagent_coordinators.get(session_id)
        cancel_task_work = getattr(coordinator, "cancel_task_and_wait", None)
        if callable(cancel_task_work):
            await cancel_task_work(task_id, current_task.generation - 1)
        return {
            "task_id": current_task.task_id,
            "generation": current_task.generation,
            "state": current_task.state,
            "revision": cancelled.revision,
        }

    async def continue_task(
        self,
        client_id: str,
        session_id: str,
        version: int,
        task_id: str,
        expected_revision: int,
        request_id: str,
    ) -> dict[str, object]:
        """Admit a user-owned continuation for one explicitly selected Task."""
        self._require_admitted()
        self.require_claim(client_id, session_id, version)
        self._ensure_session_available(session_id)
        if not self.session_model_available(session_id):
            raise service_error(
                "model_unavailable",
                "Choose an available model before continuing this Task.",
                status=422,
                field_errors={"model": "choose an available model"},
            )
        if self._restore_session_id == session_id or self._restore_commit_task is not None:
            raise service_error("session_restore_active", "Session Restore is in progress.", retryable=True)
        store = self.foreground_work_store(session_id)
        state = self._loops[session_id]
        records = self._committed_subagents(session_id)
        if not state.loop.has_active_run and not state.run_ids and not state.live_runs:
            store.reconcile(self._persisted_foreground_messages(session_id),
                            subagents=records)
        current = store.load()
        previous_continue = next((message for message in self._persisted_foreground_messages(session_id)
            if message.get("foreground_continue_request_id") == request_id), None)
        if previous_continue is not None:
            if previous_continue.get("foreground_task_id") != task_id:
                raise service_error("foreground_task_conflict", "Continuation request belongs to another Task.", status=409)
            return {"run_id": previous_continue["foreground_continue_run_id"], "task_id": task_id,
                    "children_resumed": True}
        if current.revision != expected_revision:
            raise service_error(
                "foreground_task_conflict", "The Task changed; reload it before continuing.", status=409
            )
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        if (
            task is None
            or task.state not in {"completed", "interrupted"}
            or any(
                question.task_id == task_id and (question.state in unresolved
                    or (question.state == "interrupted" and question.answer is None))
                for question in current.questions
            )
        ):
            raise service_error(
                "foreground_task_conflict",
                "The Task is active, cancelled, replaced, or waiting for an answer.",
                status=409,
            )
        record_by_id = {record.agent_id: record for record in records}
        interrupted_children = tuple(record for record in records
            if record.source.task_id == task_id
            and record.source.task_generation == task.generation
            and record.status is SubAgentStatus.INTERRUPTED)
        if interrupted_children:
            return await self._continue_interrupted_children(
                client_id, session_id, version, request_id, task, interrupted_children, expected_revision,
            )
        foreground_delivery: dict[str, object] | None = None
        for delivery in current.deliveries:
            if (
                delivery.task_id != task_id
                or delivery.task_generation != task.generation
                or delivery.state not in {"pending", "claimed"}
            ):
                continue
            record = record_by_id.get(delivery.agent_id)
            if (
                record is None
                or record.revision != delivery.terminal_record_revision
                or record.source.kind is not SubAgentSourceKind.FOREGROUND
                or record.source.task_id != task_id
                or record.source.task_generation != task.generation
                or record.status not in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
            ):
                continue
            result_text = record.result or ""
            if record.status is SubAgentStatus.FAILED:
                failure = ""
                if record.error is not None:
                    failure = f"\nFailure: {record.error.code}: {record.error.message}"
                result_text = f"The SubAgent failed.{failure}"
            foreground_delivery = {
                "agent_id": delivery.agent_id,
                "terminal_record_revision": delivery.terminal_record_revision,
                "task_generation": delivery.task_generation,
                "result": (
                    f"Title: {record.title}\n"
                    f"Status: {record.status.value}\n"
                    f"Result: {result_text}"
                ),
            }
            break
        run_id = str(uuid4())
        text = "Continue the current foreground task."
        await self.input(
            client_id,
            session_id,
            version,
            text,
            run_id,
            request_id,
            foreground_task_id=task_id,
            foreground_task_continue=True,
            foreground_task_generation=task.generation,
            foreground_delivery=foreground_delivery,
        )
        return {"run_id": run_id, "task_id": task_id}

    async def _continue_interrupted_children(
        self, client_id: str, session_id: str, version: int, request_id: str,
        task: ForegroundWorkTask, children: tuple[SubAgentRecord, ...], expected_revision: int,
    ) -> dict[str, object]:
        """Persist a real user recovery anchor before queueing registered child segments."""
        state = self._loops[session_id]
        coordinator = self._subagent_coordinators.get(session_id)
        if not isinstance(coordinator, SubAgentPool):
            raise service_error("resource_unavailable", "SubAgent execution resources are unavailable.", status=422)
        for child in children:
            creator = child.creator_snapshot
            try:
                self.configuration.resolve_session_model_route(
                    creator.provider_id, creator.model, creator.reasoning_effort,
                )
            except (ConfigError, ValueError) as error:
                raise service_error("model_unavailable", "The registered SubAgent model is unavailable.", status=422) from error
            if self._exec_host is None or (creator.shell or "auto") != self._exec_host.resolved_shell.selector:
                raise service_error("resource_unavailable", "The registered SubAgent shell is unavailable.", status=422)
        async with state.coordination_lock:
            self.require_claim(client_id, session_id, version)
            self._ensure_session_available(session_id)
            previous = next((message for message in self._persisted_foreground_messages(session_id)
                if message.get("foreground_continue_request_id") == request_id), None)
            if previous is not None:
                if previous.get("foreground_task_id") != task.task_id:
                    raise service_error("foreground_task_conflict", "Continuation request belongs to another Task.", status=409)
                return {"run_id": previous["foreground_continue_run_id"],
                        "task_id": task.task_id, "children_resumed": True}
            store = self.foreground_work_store(session_id)
            latest = store.load()
            if latest.revision != expected_revision or next((item for item in latest.tasks if item.task_id == task.task_id), None) != task:
                raise service_error("foreground_task_conflict", "The Task changed before recovery admission.", status=409)
            if state.loop.has_active_run or state.run_ids or state.live_runs:
                raise service_error("foreground_run_active", "Wait for the current foreground input before continuing children.", retryable=True)
            run_id = str(uuid4())
            token = uuid4()
            executions = {child.agent_id: str(uuid4()) for child in children}
            session = state.loop.session
            summary = session.metadata.get("summary", "")
            session.commit_agent_run([{
                "role": "user", "content": "Continue the current foreground task.",
                "foreground_task_id": task.task_id,
                "foreground_continue_request_id": request_id,
                "foreground_continue_run_id": run_id,
                "foreground_child_executions": executions,
            }], pending_last_compacted=session.last_compacted,
                pending_action_summary=summary if isinstance(summary, str) else "",
                restore_before=session.capture_restore_before(), restore_run_token=token)
            await session.wait_for_pending_persist()
            if not any(message.get("foreground_continue_request_id") == request_id
                       for message in self._persisted_foreground_messages(session_id)):
                raise service_error("persistence_error", "The continuation intent could not be saved.", status=500)
            self.require_claim(client_id, session_id, version)
            store = self.foreground_work_store(session_id)
            self._require_admitted()
            latest = store.load()
            if self._restore_session_id == session_id or self._restore_commit_task is not None:
                raise service_error("session_restore_active", "Session Restore is in progress.", retryable=True)
            if any(question.task_id == task.task_id and question.state in {
                "preparing", "pending", "answer_received", "resuming", "executing",
            } for question in latest.questions):
                raise service_error("foreground_task_conflict", "Answer the pending question before continuing.", status=409)
            if not any(item.task_id == task.task_id and item.generation == task.generation
                       and item.state not in {"cancelled", "superseded"} for item in latest.tasks):
                raise service_error("foreground_task_conflict", "The Task was revoked while saving continuation.", status=409)
            assert self._exec_host is not None
            permission = self.service.client_permission(client_id).snapshot(self._exec_host.resolved_shell)
            try:
                store.start_execution(task.task_id, run_id, expected_generation=task.generation)
                store.complete_task(task.task_id, run_id)
                for child in children:
                    saved_answer = next((question.answer for question in store.load().questions
                        if question.owner_agent_id == child.agent_id and question.answer is not None), None)
                    coordinator.resume_interrupted(child.agent_id, execution_id=executions[child.agent_id],
                        restore_run_token=str(token), permission_level=permission.level,
                        saved_answer=saved_answer)
                    store.acknowledge_subagent_recovery(child.agent_id, executions[child.agent_id])
            except (OSError, SubAgentStoreError, SubAgentRequestError, ValueError) as error:
                repository = self.subagent_repository(session_id)
                for child in children:
                    saved = repository.get(child.agent_id)
                    if saved is not None and saved.status is SubAgentStatus.QUEUED:
                        repository.save(replace(saved, status=SubAgentStatus.INTERRUPTED,
                            finished_at=local_now(), error=SubAgentError(
                                code="resume_interrupted", message="Continuation queue publication was interrupted."),
                            revision=saved.revision + 1), expected_revision=saved.revision)
                store.interrupt_uncertain_tasks()
                raise service_error("persistence_error", "SubAgent continuation was interrupted; inspect saved state before continuing.", status=500) from error
        return {"run_id": run_id, "task_id": task.task_id, "children_resumed": True}

    async def recall_queued_inputs(
        self,
        client_id: str,
        session_id: str,
        version: int,
    ) -> dict[str, object]:
        self.require_claim(client_id, session_id, version)
        state = self._loops[session_id]
        recalled: list[tuple[str, InboundMessage]] = []
        async with state.coordination_lock:
            self._require_admitted()
            self.require_claim(client_id, session_id, version)
            queued = await state.bus.inbound_snapshot()
            for message in queued:
                run_id = message.metadata.get("run_id")
                if (
                    not isinstance(run_id, str)
                    or run_id not in state.run_ids
                    or run_id not in state.live_runs
                ):
                    raise service_error(
                        "service_state_invalid",
                        "Queued Conversation input state is inconsistent.",
                        status=500,
                    )
                recalled.append((run_id, message))
            drained = await state.bus.drain_inbound()
            if len(drained) != len(recalled):
                raise service_error(
                    "service_state_invalid",
                    "Queued Conversation input state changed during recall.",
                    status=500,
                )
            recalled_ids = {run_id for run_id, _message in recalled}
            for run_id in recalled_ids:
                state.run_ids.remove(run_id)

        for run_id, message in recalled:
            request_id = message.metadata.get("request_id")
            await self.service.emit(
                "input.recalled",
                workspace_id=self.workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload={
                    "text": message.content,
                    "request_id": request_id if isinstance(request_id, str) else run_id,
                },
                target_client_ids=(client_id,),
            )
        return {
            "recalled_inputs": [
                {"run_id": run_id, "text": message.content} for run_id, message in recalled
            ],
            "live_state": self.session_snapshot(session_id)["live_state"],
        }

    def _ensure_processor(self, state: _LoopState) -> None:
        """Start one Session processor while preserving the enqueue boundary."""
        if state.processor_stopping:
            raise service_error(
                "admission_closed", "Conversation input is temporarily unavailable."
            )
        processor = state.processor_task
        if processor is not None and not processor.done():
            state.processor_wakeup_pending = True
            return
        processor = asyncio.create_task(self._process_session(state))
        state.processor_task = processor

        def processor_finished(completed: asyncio.Task[None]) -> None:
            self._processor_finished(state, completed)

        processor.add_done_callback(processor_finished)

    def wake_foreground_continuations(self) -> None:
        """Wake retained foreground Sessions after the Service becomes available."""
        if not self.service.foreground_continuation_admission_open():
            return
        for state in tuple(self._loops.values()):
            if (
                state.schedule
                or state.processor_stopping
                or state.foreground_permission_control is None
                or not state.loop.foreground_input_admitted()
                or not self.session_model_available(state.loop.session.session_id)
            ):
                continue
            if self._has_pending_automatic_subagent_result(state):
                self._ensure_processor(state)

    def _has_pending_automatic_subagent_result(self, state: _LoopState) -> bool:
        session_id = state.loop.session.session_id
        try:
            records = self._committed_subagents(session_id)
            snapshot = self.foreground_work_store(session_id).load()
        except (OSError, UnicodeError, ValueError, SubAgentStoreError):
            return False
        tasks = {
            (task.task_id, task.generation): task for task in snapshot.tasks
        }
        deliveries = {
            (delivery.agent_id, delivery.terminal_record_revision, delivery.task_generation):
            delivery
            for delivery in snapshot.deliveries
        }
        for record in records:
            source = record.source
            if (
                source.kind is not SubAgentSourceKind.FOREGROUND
                or source.task_id is None
                or source.task_generation is None
                or record.status not in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
            ):
                continue
            task = tasks.get((source.task_id, source.task_generation))
            if (
                task is None
                or task.state not in {"running", "completed"}
                or task.iteration_budget_used >= task.iteration_budget_limit
            ):
                continue
            delivery = deliveries.get(
                (record.agent_id, record.revision, source.task_generation)
            )
            if delivery is None or delivery.state == "pending":
                return True
        return False

    def _processor_finished(self, state: _LoopState, task: asyncio.Task[None]) -> None:
        if state.processor_task is task:
            state.processor_task = None
        _consume_task_result(task)
        if state.processor_wakeup_pending:
            state.processor_wakeup_pending = False
            if (
                not state.processor_stopping
                and not state.schedule
                and self.service.foreground_continuation_admission_open()
            ):
                self._ensure_processor(state)

    async def _process_session(self, state: _LoopState) -> None:
        """Run accepted foreground inputs serially until the Session queue is empty."""
        current = asyncio.current_task()
        if current is None:
            return
        release_when_idle = False
        while True:
            inbound: InboundMessage | None = None
            async with state.coordination_lock:
                if state.processor_stopping:
                    if state.processor_task is current:
                        state.processor_task = None
                    return
                if await state.bus.inbound_snapshot():
                    inbound = await state.bus.get_inbound()
                else:
                    inbound = self._claim_automatic_subagent_result(state)
                    if inbound is None:
                        if state.processor_task is current:
                            state.processor_task = None
                        release_when_idle = not self._closed and state.owner_client_id is not None
                        break
                assert inbound is not None
                run_id = inbound.metadata.get("run_id")
                if not isinstance(run_id, str):
                    run_id = state.run_ids[0] if state.run_ids else None

            output_task = asyncio.create_task(self._forward_output(state, run_id))
            state.output_task = output_task
            terminal_forwarded = False
            execution_failed = False
            automatic_continuation = (
                inbound.metadata.get("foreground_automatic_continuation") is True
            )
            execution_task = asyncio.create_task(
                state.loop.run_automatic_continuation(inbound)
                if automatic_continuation
                else state.loop.run_foreground(inbound)
            )
            try:
                try:
                    while not state.loop.has_active_run and not execution_task.done():
                        await asyncio.sleep(0)
                    if state.loop.has_active_run and run_id is not None:
                        await self.service.emit(
                            "run.started",
                            workspace_id=self.workspace_id,
                            session_id=state.loop.session.session_id,
                            run_id=run_id,
                            payload={"source": "automatic"} if automatic_continuation else {},
                        )
                    await execution_task
                except asyncio.CancelledError:
                    execution_task.cancel()
                    await asyncio.gather(execution_task, return_exceptions=True)
                    raise
                except Exception:
                    execution_failed = True
                    if not output_task.done():
                        output_task.cancel()
                        await asyncio.gather(output_task, return_exceptions=True)
                    if run_id is not None:
                        await self._emit_unexpected_run_failure(state, run_id)
                        terminal_forwarded = True
                if not execution_failed and not output_task.done():
                    terminal_forwarded = await output_task
                elif not execution_failed:
                    terminal_forwarded = bool(output_task.result())
                if not terminal_forwarded and run_id is not None:
                    await self._emit_unexpected_run_failure(state, run_id)
            finally:
                if not output_task.done():
                    output_task.cancel()
                    await asyncio.gather(output_task, return_exceptions=True)
                if state.output_task is output_task:
                    state.output_task = None

        if release_when_idle and (
            state.release_task is None or state.release_task.done()
        ):
            state.release_task = asyncio.create_task(
                self._release_switched_claim_when_idle(state.loop.session.session_id, state)
            )
            state.release_task.add_done_callback(_consume_task_result)

    def _claim_automatic_subagent_result(self, state: _LoopState) -> InboundMessage | None:
        """Claim one persisted foreground result after the current user queue is drained."""
        session_id = state.loop.session.session_id
        if (
            self._closed
            or self._restore_blocked
            or self.service.state in {"draining", "stopped"}
            or not state.loop.foreground_input_admitted()
            or state.schedule
            or state.foreground_permission_control is None
            or not self.service.foreground_continuation_admission_open()
            or not self.session_model_available(session_id)
        ):
            return None
        store = self.foreground_work_store(session_id)
        try:
            records = self._committed_subagents(session_id)
            snapshot = store.reconcile(
                self._persisted_foreground_messages(session_id),
                subagents=records,
            )
        except (OSError, UnicodeError, ValueError, SubAgentStoreError):
            return None
        records_by_id = {record.agent_id: record for record in records}
        eligible: dict[
            tuple[str, int], list[tuple[ForegroundWorkDelivery, SubAgentRecord, str]]
        ] = {}
        for delivery in snapshot.pending_deliveries:
            record = records_by_id.get(delivery.agent_id)
            if (
                record is None
                or record.revision != delivery.terminal_record_revision
                or record.source.kind is not SubAgentSourceKind.FOREGROUND
                or record.source.task_id != delivery.task_id
                or record.source.task_generation != delivery.task_generation
                or record.status not in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
            ):
                continue
            task = next(
                (
                    item
                    for item in snapshot.tasks
                    if item.task_id == delivery.task_id
                    and item.generation == delivery.task_generation
                ),
                None,
            )
            if (
                task is None
                or task.state not in {"running", "completed"}
                or task.iteration_budget_used >= task.iteration_budget_limit
            ):
                continue
            effect_run_token = _effective_task_restore_token(
                state.loop.session.messages,
                delivery.task_id,
                record.source.restore_run_token,
            )
            if effect_run_token is None:
                continue
            eligible.setdefault((delivery.task_id, delivery.task_generation), []).append(
                (delivery, record, effect_run_token)
            )

        for (task_id, task_generation), candidates in eligible.items():
            run_id = str(uuid4())
            try:
                claimed_deliveries = store.claim_delivery_batch(
                    tuple(
                        (
                            delivery.agent_id,
                            delivery.terminal_record_revision,
                            delivery.task_generation,
                        )
                        for delivery, _record, _effect_run_token in candidates
                    ),
                    execution_id=run_id,
                )
                if claimed_deliveries is None:
                    continue
            except (OSError, UnicodeError, ValueError):
                return None
            claimed_by_key = {
                (item.agent_id, item.terminal_record_revision, item.task_generation): item
                for item in claimed_deliveries
            }
            result_sections: list[str] = []
            receipt_records: list[dict[str, object]] = []
            for delivery, record, _effect_run_token in candidates:
                claimed = claimed_by_key[
                    (delivery.agent_id, delivery.terminal_record_revision, delivery.task_generation)
                ]
                result_text = record.result or ""
                if record.status is SubAgentStatus.FAILED:
                    error_text = ""
                    if record.error is not None:
                        error_text = f"\nFailure: {record.error.code}: {record.error.message}"
                    result_text = f"The SubAgent failed.{error_text}"
                result_sections.append(
                    f"Title: {record.title}\nStatus: {record.status.value}\nResult: {result_text}"
                )
                receipt_records.append(
                    {
                        "agent_id": claimed.agent_id,
                        "terminal_record_revision": claimed.terminal_record_revision,
                        "task_generation": claimed.task_generation,
                        "receipt_id": claimed.receipt_id,
                    }
                )
            inbound = InboundMessage(
                content=("SubAgent result for automatic continuation:\n" + "\n\n".join(result_sections)),
                metadata={
                    "run_id": run_id,
                    "foreground_task_id": task_id,
                    "foreground_automatic_continuation": True,
                    "foreground_delivery_receipts": receipt_records,
                    "foreground_task_generation": task_generation,
                    "foreground_effect_run_token": candidates[0][2],
                },
            )
            state.completed_user_count = sum(
                message.get("role") == "user" for message in state.loop.session.messages
            )
            state.run_ids.append(run_id)
            state.live_runs[run_id] = {
                "run_id": run_id,
                "request_id": run_id,
                "prompt": f"SubAgent results: {', '.join(record.title for _, record, _ in candidates)}",
                "source": "automatic",
                "status": "accepted",
                "assistant_content": "",
                "tools": [],
                "response_segments": [""],
                "cancel_requested": False,
                "cancellable": False,
            }
            return inbound
        return None

    async def _emit_unexpected_run_failure(self, state: _LoopState, run_id: str) -> None:
        """Keep one accepted Run from stranding when an execution boundary fails."""
        try:
            await self.service.emit(
                "run.output",
                workspace_id=self.workspace_id,
                session_id=state.loop.session.session_id,
                run_id=run_id,
                payload={
                    "message": {
                        "type": "system_control",
                        "content": "The Agent Run failed unexpectedly.",
                        "metadata": {
                            "finish_reason": "failed",
                            "error_code": "model_failed",
                            "_streamed": True,
                        },
                    }
                },
            )
            await self.service.emit(
                "run.completed",
                workspace_id=self.workspace_id,
                session_id=state.loop.session.session_id,
                run_id=run_id,
                payload={"finish_reason": "failed"},
            )
        except Exception:
            pass
        finally:
            try:
                state.run_ids.remove(run_id)
            except ValueError:
                pass

    async def cancel(self, client_id: str, session_id: str, version: int, run_id: str) -> None:
        claim = self.require_claim(client_id, session_id, version)
        state = self._loops[session_id]
        client = self.service.client(client_id)
        async with client.delivery_lock:
            claim = self.require_claim(client_id, session_id, version)
            if not claim.loop.has_active_run:
                return
            if not state.run_ids or state.run_ids[0] != run_id:
                raise service_error("stale_run", "The requested Agent Run is no longer active.")
            if run_id in state.live_runs:
                state.live_runs[run_id]["cancel_requested"] = True
        await claim.loop.cancel_active_run()
        await self.service.emit(
            "run.cancelled",
            workspace_id=self.workspace_id,
            session_id=session_id,
            run_id=run_id,
            payload={},
            target_client_ids=(client_id,),
        )

    async def _create_loop(self, session_id: str | None, *, client_id: str | None) -> _LoopState:
        if not self._started or self.workspace_state is None or self._exec_host is None:
            raise RuntimeError("Workspace service runtime is not ready")
        permission_control = (
            self.service.client_permission(client_id)
            if client_id is not None
            else self._schedule_permission
        )
        loop, bus = await self._create_agent_loop(
            runtime=self.resources,
            configuration=self.configuration,
            schedule_service=self.schedule_service,
            exec_host=self._exec_host,
            session_id=session_id,
            permission_control=permission_control,
        )
        state = _LoopState(
            loop=loop,
            bus=bus,
            owner_client_id=client_id,
            foreground_permission_control=(
                None if client_id is None else self.service.client_permission(client_id)
            ),
        )
        self._loops[loop.session.session_id] = state
        return state

    async def _create_agent_loop(
        self,
        *,
        runtime: Any,
        configuration: UserConfiguration,
        schedule_service: ScheduleService,
        exec_host: ExecHost,
        session_id: str | None,
        permission_control: RuntimePermissionControl,
        session: Session | None = None,
        bus: MessageBus | None = None,
    ) -> tuple[SessionExecution, MessageBus]:
        selected_bus = MessageBus() if bus is None else bus
        authority = session
        if authority is None:
            authority = (
                Session.create(self.workspace_state, now=local_now, new_uuid=uuid4)
                if session_id is None
                else Session.load(self.workspace_state, session_id, now=local_now)
            )

        run_state = SessionRunState(uuid4())

        def create_executor(
            session_state: SessionRunState,
            configured: WorkspaceConfigurationResources | None = None,
            captured_skills: SkillLoader | None = None,
        ) -> AgentRunExecutor:
            state = self._loops.get(authority.session_id)
            if state is None:
                control = permission_control
            elif state.schedule:
                control = self._schedule_permission
            elif state.foreground_permission_control is not None:
                control = state.foreground_permission_control
            else:
                raise RuntimeError("Foreground execution has no permission source")
            current_configuration = (
                self.configuration if configured is None else configured.shared.snapshot.configuration
            )
            router = (
                ModelRouter(configuration=current_configuration, provider_factory=create_provider)
                if configured is None else configured.shared.router
            )
            foreground_task_store = (
                self.foreground_work_store(authority.session_id)
                if authority._storage_partition is SessionStoragePartition.FOREGROUND
                else None
            )
            loop_kwargs: dict[str, Any] = {
                "workspace_path": self.workspace_path,
                "workspace_state": self.workspace_state,
                "agent_home": self.service.agent_home,
                "configuration": current_configuration,
                "bus": selected_bus,
                "schedule_service": schedule_service,
                "model_router": router,
                "memory_manager": runtime.memory_manager,
                "trace_recorder": self.service._trace_recorder,
                "now": local_now,
                "new_uuid": uuid4,
                "monotonic_now": monotonic,
                "mcp_tools": runtime.mcp_snapshot if configured is None else configured.mcp_report.snapshot,
                "mcp_keywords": runtime.mcp_keywords if configured is None else configured.keywords,
                "exec_host": exec_host if configured is None else configured.shared.exec_host,
                "permission_control": control,
                "configured_schedule_level": current_configuration.runtime.permission_level,
                "skill_loader": self.service.skill_loader if captured_skills is None else captured_skills,
                "built_in_catalog": (
                    self.service.built_in_tool_catalog if configured is None else configured.shared.built_in_catalog
                ),
                "subagent_model_router": router,
                "foreground_work_store": foreground_task_store,
            }
            executor = AgentRunExecutor(
                session=authority, session_id=None, session_run_state=session_state, **loop_kwargs
            )
            executor.bind_confirmation_requester(self.service.confirmation.request)
            coordinator = self._subagent_coordinator(authority.session_id)
            if coordinator is None:
                repository = self.subagent_repository(authority.session_id)
                child_executor = executor.create_subagent_runner_executor(
                    repository,
                    workspace_id=self.workspace_id,
                    confirmation_for=self._subagent_confirmation_for,
                )
                coordinator = SubAgentPool(
                    repository,
                    child_executor,
                    now=local_now,
                    workspace_id=self.workspace_id,
                    event_publisher=self,
                    foreground_task_is_current=(
                        None
                        if foreground_task_store is None
                        else lambda task_id, generation: self.foreground_work_store(
                            authority.session_id
                        ).is_active_generation(task_id, generation)
                    ),
                )
                self.register_subagent_coordinator(coordinator, repository)
            executor.bind_subagent_coordinator(coordinator)
            if configured is not None and isinstance(coordinator, SubAgentPool):
                child = executor.create_subagent_runner_executor(
                    self.subagent_repository(authority.session_id),
                    workspace_id=self.workspace_id,
                    confirmation_for=self._subagent_confirmation_for,
                )
                coordinator.bind_executor(
                    child, lambda: self.service._configuration_resources.retain(configured),
                )
            return executor

        async def prepare_executor(
            session_state: SessionRunState,
        ) -> tuple[AgentRunExecutor, Callable[[], None]]:
            snapshot = self.service._capture_configuration()
            self.service._initialize_shared_resources(snapshot.configuration)
            model = authority.model_configuration
            state = self._loops.get(authority.session_id)
            if state is None:
                raise RuntimeError("Session execution state is unavailable")
            if state.schedule:
                control = self._schedule_permission
            elif state.foreground_permission_control is not None:
                control = state.foreground_permission_control
            else:
                raise RuntimeError("Foreground execution has no permission source")
            host = self.service._exec_host_for(snapshot.configuration)
            permission = control.snapshot(host.resolved_shell)
            skills = self.service.skill_loader.with_always_load(
                snapshot.configuration.runtime.enable_skill_always_load,
            )
            configured, activated = await self.service._prepare_execution_resources(self, snapshot)
            release = self.service._configuration_resources.retain(configured)
            try:
                executor = create_executor(session_state, configured, skills)
                executor.capture_run_inputs(permission, model)
                if activated:
                    await self.service._emit_configuration_event()
            except BaseException:
                release()
                raise
            return executor, release

        prepared = create_executor(run_state)
        prepared.preflight()
        await prepared.start()
        tool_schemas = prepared.tool_schemas

        def status_input() -> Any:
            builder = ContextBuilder(
                self.workspace_path, self.schedule_service.context_timezone_name() or get_localzone_name(),
                agent_home=self.service.agent_home.path, memory_manager=self.memory_manager,
                skill_loader=self.service.skill_loader,
            )
            return session_runtime_status_input(
                authority, configuration=self.configuration, context_builder=builder,
                tool_schemas=tool_schemas, generation_started_at=self.service._started_at,
            )

        handle = SessionExecution(
            authority, selected_bus, create_executor, self.service.reload_skills, status_input,
            run_state=run_state, prepare_executor=prepare_executor,
        )
        return handle, selected_bus

    async def publish(self, event: SubAgentEvent) -> None:
        """Publish a SubAgent event to the active claimant without a Main Run ID."""
        if event.workspace_id != self.workspace_id:
            raise ValueError("SubAgent event belongs to a different Workspace")
        if (
            event.kind.value == "subagent.status"
            and event.data.get("status") in {
                SubAgentStatus.COMPLETED.value,
                SubAgentStatus.FAILED.value,
            }
        ):
            state = self._loops.get(event.session_id)
            if state is not None and self.service.foreground_continuation_admission_open():
                self._ensure_processor(state)
        claim = self._claims.get(event.session_id)
        target_client_ids = () if claim is None else (claim.client_id,)
        await self.service.emit(
            event.kind.value,
            workspace_id=self.workspace_id,
            session_id=event.session_id,
            run_id=None,
            payload={
                "agent_id": event.agent_id,
                "revision": event.revision,
                "data": deepcopy(event.data),
            },
            target_client_ids=target_client_ids,
        )
        if claim is not None and event.kind.value == "subagent.status":
            client = self.service._clients.get(claim.client_id)
            if client is not None:
                await self.service._send_snapshot_required(client, reason="subagent_status_changed")

    def _subagent_confirmation_for(self, record: SubAgentRecord) -> ConfirmationRequester:
        owner = SubAgentConfirmationOwner(
            generation_id=uuid4(),
            workspace_id=self.workspace_id,
            session_id=record.session_id,
            agent_id=record.agent_id,
        )
        background = record.source.kind is SubAgentSourceKind.SCHEDULE

        async def request(request: ConfirmationRequest) -> ConfirmationDecision:
            return await self.service.confirmation.request(
                ConfirmationEnvelope(
                    request=request,
                    origin="background" if background else "foreground",
                    owner=owner,
                    job_id=record.source.job_id if background else None,
                    title=record.title if background else None,
                )
            )

        return request

    async def _get_schedule_loop(self, job_id: str, *, title: str | None = None) -> _LoopState:
        state = self._schedule_loops.get(job_id)
        if state is not None:
            return state
        try:
            session = Session.load(self.workspace_state, Session.schedule_session_id(job_id),
                                   partition=SessionStoragePartition.SCHEDULE, now=local_now)
        except FileNotFoundError:
            session = Session.create_schedule(self.workspace_state, job_id, now=local_now, title=title or "Untitled session")
        handle, bus = await self._create_agent_loop(
            runtime=self.resources, configuration=self.configuration,
            schedule_service=self.schedule_service, exec_host=self.service.exec_host,
            session_id=None, permission_control=self._schedule_permission, session=session,
        )
        state = _LoopState(loop=handle, bus=bus, owner_client_id=None, schedule=True)
        self._loops[session.session_id] = state
        self._schedule_loops[job_id] = state
        return state

    async def _forward_output(self, state: _LoopState, run_id: str | None = None) -> bool:
        session_id = state.loop.session.session_id
        try:
            while not self._closed and self._loops.get(session_id) is state:
                message = await state.bus.get_outbound()
                current_run_id = run_id or (state.run_ids[0] if state.run_ids else None)
                if (
                    message.metadata.get("_streamed") is True
                    and not state.schedule
                    and self.foreground_work_store(session_id).load().tasks
                ):
                    claim = self._claims.get(session_id)
                    client = None if claim is None else self.service._clients.get(claim.client_id)
                    if (
                        client is not None
                        and client.kind == "cli"
                        and client.current_session_id == session_id
                    ):
                        await self.service._send_snapshot_required(
                            client, reason="foreground_question_updated"
                        )
                await self.service.emit(
                    "run.output",
                    workspace_id=self.workspace_id,
                    session_id=session_id,
                    run_id=current_run_id,
                    payload={
                        "message": {
                            "type": message.type,
                            "content": message.content,
                            "metadata": dict(message.metadata),
                        }
                    },
                )
                if message.metadata.get("_streamed") is True:
                    completed = current_run_id
                    if completed is not None:
                        try:
                            state.run_ids.remove(completed)
                        except ValueError:
                            pass
                    await self.service.emit(
                        "run.completed",
                        workspace_id=self.workspace_id,
                        session_id=session_id,
                        run_id=completed,
                        payload={
                            "finish_reason": message.metadata.get("finish_reason", "completed")
                        },
                    )
                    return True
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        return False

    async def _release_switched_claim_when_idle(self, session_id: str, state: _LoopState) -> None:
        while True:
            if self._closed or self._loops.get(session_id) is not state:
                return
            try:
                active = state.loop.has_active_run
            except RuntimeError:
                return
            if not active and not state.run_ids and state.output_task is None:
                break
            await asyncio.sleep(0.01)
        claim = self._claims.get(session_id)
        if claim is None:
            return
        await self._release_switched_claim_if_idle(claim.client_id, session_id)

    async def _release_switched_claim_if_idle(self, client_id: str, session_id: str) -> None:
        claim = self._claims.get(session_id)
        if claim is None or claim.client_id != client_id:
            return
        client = self.service._clients.get(client_id)
        if (
            client is not None
            and client.current_workspace_id == self.workspace_id
            and client.current_session_id == session_id
        ):
            return
        await self._release_if_idle(client_id, session_id)

    async def _release_if_idle(self, client_id: str, session_id: str) -> None:
        state = self._loops.get(session_id)
        claim = self._claims.get(session_id)
        if state is None or claim is None or claim.client_id != client_id:
            return
        try:
            if session_deletion_pending(self.workspace_state, session_id):
                return
            if (
                state.loop.has_active_run
                or state.run_ids
                or state.output_task is not None
                or await state.bus.inbound_snapshot()
            ):
                return
        except RuntimeError:
            return
        await self.release(client_id, session_id)

    async def _close_loop_state(
        self,
        state: _LoopState,
        *,
        abort: bool = False,
        close_loop: bool = True,
    ) -> None:
        if state.release_task is not None:
            release_task = state.release_task
            state.release_task = None
            if release_task is not asyncio.current_task():
                release_task.cancel()
                await asyncio.gather(release_task, return_exceptions=True)
        async with state.coordination_lock:
            state.processor_stopping = True
        processor = state.processor_task
        if abort:
            await self.service.confirmation.cancel_generation(state.loop.generation_id)
            await state.loop.abort()
        else:
            try:
                active = state.loop.has_active_run
            except RuntimeError:
                active = False
            if active:
                await state.loop.cancel_active_run()
        if processor is not None and processor is not asyncio.current_task():
            if not processor.done():
                processor.cancel()
            await asyncio.gather(processor, return_exceptions=True)
        if state.output_task is not None:
            output_task = state.output_task
            state.output_task = None
            if output_task is not asyncio.current_task():
                output_task.cancel()
                await asyncio.gather(output_task, return_exceptions=True)
        if not abort:
            if close_loop:
                await state.loop.close()
            await self.service.confirmation.cancel_generation(state.loop.generation_id)

    async def _close_loop(self, session_id: str, *, abort: bool = False) -> None:
        state = self._loops.get(session_id)
        if state is None:
            self._retained_session_closes.discard(session_id)
            return
        retain_session = session_id in self._retained_session_closes
        try:
            await self._close_loop_state(state, abort=abort, close_loop=not retain_session)
            if retain_session:
                await state.loop.finish_work()
                await state.bus.reset()
                state.run_ids.clear()
                state.live_runs.clear()
                state.completed_user_count = None
                state.processor_stopping = False
                state.owner_client_id = None
            else:
                self._loops.pop(session_id, None)
        finally:
            self._retained_session_closes.discard(session_id)
        for job_id, candidate in tuple(self._schedule_loops.items()):
            if candidate is state:
                self._schedule_loops.pop(job_id, None)

    async def close(self, *, interrupted: bool = True) -> None:
        self.close_subagent_admission()
        # Claim release may hold this lock while awaiting a title naturally.
        for state in tuple(self._loops.values()):
            state.loop.cancel_title_work()
        async with self._lock:
            if self._closed and not self._close_failed:
                return
            self._closed = True
            self._schedule_admitted = False
            try:
                restore_task = self._restore_commit_task
                if restore_task is not None and restore_task is not asyncio.current_task():
                    await asyncio.shield(asyncio.gather(restore_task, return_exceptions=True))
                await self._shutdown_subagents(interrupted=interrupted)
                if self._restore_blocked:
                    raise service_error(
                        "restore_pending",
                        "The active Restore transaction requires recovery before Project removal.",
                        retryable=True,
                    )
                if self._restore_owner is not None:
                    await self._release_restore_barrier(self._restore_owner)
                await self.service._close_workspace_resources(self)
            except BaseException:
                self._close_failed = True
                raise
            else:
                self._close_failed = False

    async def _shutdown_subagents(self, *, interrupted: bool) -> None:
        errors: list[Exception] = []
        for coordinator in tuple(self._subagent_coordinators.values()):
            try:
                await coordinator.shutdown(interrupted=interrupted)
            except Exception as error:
                errors.append(error)
        self._subagent_shutdown_failed = bool(errors)
        if errors:
            raise SubAgentStoreError("Workspace SubAgent tasks could not be drained safely") from (
                ExceptionGroup("SubAgent shutdown failed", errors)
            )

    async def _close_all_loops(self) -> None:
        """Flush every Session, collecting failures without skipping later entries."""
        errors: list[Exception] = []
        for session_id in tuple(self._loops):
            try:
                await self._close_loop(session_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Session cleanup failed", errors)
