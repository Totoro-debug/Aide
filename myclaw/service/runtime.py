"""Single-process local service authority for Workspace and Session execution."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from types import MappingProxyType
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
from myclaw.agent.session.deletion import (
    begin_session_deletion,
    delete_session_data,
    session_deletion_pending,
    session_deletion_status,
    session_restore_pending,
)
from myclaw.agent.session.restore import RestoreManager, RestoreRecoveryRequired, StaleRestorePlan
from myclaw.agent.session.session import Session, SessionStoragePartition
from myclaw.agent.tools.core.exec_host import ExecHost, create_exec_host, resolve_exec_shell
from myclaw.agent.tools.mcp_keywords import MCPKeywordPreparer
from myclaw.agent.tools.mcp_runtime import MCPRuntimeManager
from myclaw.agent.tools.tool_gateway import BUILT_IN_TOOL_NAMES
from myclaw.agent.workspace_runtime import WorkspaceRuntime, WorkspaceRuntimeFactories
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import (
    ConfigError,
    ConfigFieldError,
    ConfigLoader,
    ConfigRevisionConflict,
    ReasoningEffort,
    UserConfiguration,
)
from myclaw.errors import ErrorInfo
from myclaw.provider.factory import create_provider
from myclaw.provider.model_router import ModelRouter
from myclaw.schedule.history import (
    ScheduleHistoryPersistenceError,
    ScheduleHistoryRequestError,
    read_schedule_history,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleOccurrence, ScheduleService, ScheduleStaleRemovalError
from myclaw.schedule.store import (
    ScheduleStateError,
    ScheduleStoreFaultedError,
    WorkspaceScheduleStore,
)
from myclaw.service.errors import ServiceError, service_error
from myclaw.service.projects import ProjectCatalog, ProjectCatalogError, ProjectRecord
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.text import normalize_title_candidate
from myclaw.utils.time import local_now


class ServiceSink(Protocol):
    async def send_event(self, event: dict[str, object]) -> None: ...


_PROJECT_REMOVAL_FAILURE_MESSAGE = (
    "Project work could not be stopped; the registration remains blocked."
)
_CONFIG_INVALID_ERROR = {
    "code": "config_invalid",
    "message": "The saved User Configuration contains invalid fields.",
}
_MAX_SESSION_PAGE_SIZE = 100
_MISSING = object()


def _schedule_job_projection(
    job: ScheduleJob,
    *,
    active: bool,
    status: str | None = None,
) -> dict[str, object]:
    """Return the public Job shape without exposing the Schedule store."""
    projection = job.to_dict()
    projection["session_id"] = job.session_id
    projection["active"] = active
    projection["status"] = status or ("running" if active else job.state.last_status or "scheduled")
    return projection


def _schedule_request_fingerprint(
    workspace_id: str,
    action: str,
    payload: Mapping[str, object],
) -> str:
    try:
        return json.dumps(
            {
                "workspace_id": workspace_id,
                "action": action,
                "payload": {key: value for key, value in payload.items() if key != "request_id"},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise service_error(
            "validation_error", "Schedule Job input is not valid JSON.", status=422
        ) from error


def _schedule_job_input(payload: Mapping[str, object]) -> tuple[str, str, JobSchedule]:
    """Normalize the existing Schedule Tool input shape for the HTTP boundary."""
    field_errors: dict[str, str] = {}
    message = payload.get("message")
    if not isinstance(message, str):
        normalized_message = ""
        field_errors["message"] = "must be a string"
    else:
        normalized_message = message.strip()
        if not normalized_message:
            field_errors["message"] = "must not be empty"
        elif len(normalized_message) > 20_000:
            field_errors["message"] = "must not exceed 20000 characters"

    title_value = payload.get("title", _MISSING)
    if title_value is _MISSING:
        normalized_title = normalize_title_candidate(normalized_message)
    elif not isinstance(title_value, str):
        normalized_title = ""
        field_errors["title"] = "must be a string"
    else:
        normalized_title = normalize_title_candidate(title_value)
        if not normalized_title:
            field_errors["title"] = "must not be empty"

    nested_schedule = payload.get("schedule")
    if nested_schedule is not None and not isinstance(nested_schedule, Mapping):
        field_errors["schedule"] = "must be an object"
        nested: Mapping[str, object] = {}
    else:
        nested = nested_schedule if isinstance(nested_schedule, Mapping) else {}

    def schedule_value(name: str) -> object:
        if name in payload:
            return payload[name]
        return nested.get(name)

    kind_value = payload.get("kind", nested.get("kind"))
    at_time = schedule_value("at_time")
    every_seconds = schedule_value("every_seconds")
    cron_expr = schedule_value("cron_expr")
    timezone = schedule_value("timezone")
    selected = [
        name
        for name, value in (
            ("at", at_time),
            ("every", every_seconds),
            ("cron", cron_expr),
        )
        if value is not None
    ]

    selected_kind: str | None = None
    if kind_value is not None:
        if not isinstance(kind_value, str) or kind_value not in {"at", "every", "cron"}:
            field_errors["kind"] = "must be at, every, or cron"
        else:
            selected_kind = kind_value
            if selected != [kind_value]:
                field_errors["schedule"] = "must select exactly one matching schedule kind"
    elif len(selected) != 1:
        field_errors["schedule"] = "must select exactly one of at_time, every_seconds, or cron_expr"
    else:
        selected_kind = selected[0]

    schedule: JobSchedule | None = None
    if selected_kind == "at" and "schedule" not in field_errors:
        if timezone is not None:
            field_errors["timezone"] = "is only valid for cron schedules"
        if not isinstance(at_time, str):
            field_errors["at_time"] = "must be a timezone-aware ISO time"
        else:
            try:
                schedule = JobSchedule.from_at_input(at_time)
            except (TypeError, ValueError):
                field_errors["at_time"] = "must be a valid timezone-aware ISO time"
    elif selected_kind == "every" and "schedule" not in field_errors:
        if timezone is not None:
            field_errors["timezone"] = "is only valid for cron schedules"
        if isinstance(every_seconds, bool) or not isinstance(every_seconds, int):
            field_errors["every_seconds"] = "must be a positive integer"
        else:
            try:
                schedule = JobSchedule.every(every_seconds)
            except (TypeError, ValueError):
                field_errors["every_seconds"] = "must be a positive integer"
    elif selected_kind == "cron" and "schedule" not in field_errors:
        if not isinstance(cron_expr, str):
            field_errors["cron_expr"] = "must be a valid five-field cron expression"
        if timezone is not None and not isinstance(timezone, str):
            field_errors["timezone"] = "must be a valid IANA timezone"
        if isinstance(cron_expr, str):
            try:
                schedule = JobSchedule.from_cron_input(cron_expr)
            except (TypeError, ValueError):
                field_errors["cron_expr"] = "must be a valid five-field cron expression"
        if isinstance(timezone, str):
            try:
                validated_timezone = JobSchedule.from_cron_input("* * * * *", timezone)
            except (TypeError, ValueError):
                field_errors["timezone"] = "must be a valid IANA timezone"
            else:
                if schedule is not None:
                    schedule = JobSchedule.cron(
                        schedule.cron_expr or "", validated_timezone.timezone or "UTC"
                    )

    if field_errors:
        raise service_error(
            "validation_error",
            "Schedule Job input is invalid.",
            status=400,
            field_errors=field_errors,
        )
    assert schedule is not None
    return normalized_message, normalized_title, schedule


def _schedule_epoch_milliseconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise service_error("schedule_unavailable", "Schedule time is unavailable.", retryable=True)
    milliseconds = int(value.timestamp() * 1000)
    if milliseconds < 0:
        raise service_error("schedule_unavailable", "Schedule time is unavailable.", retryable=True)
    return milliseconds


def _encode_session_cursor(
    key: tuple[datetime, datetime, str], workspace_id: str, title_filter: str
) -> str:
    payload = json.dumps(
        {
            "workspace_id": workspace_id,
            "title_filter": title_filter,
            "updated_at": key[0].isoformat(),
            "created_at": key[1].isoformat(),
            "id": key[2],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_session_cursor(
    value: str, workspace_id: str, title_filter: str
) -> tuple[datetime, datetime, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        )
        if payload["workspace_id"] != workspace_id or payload["title_filter"] != title_filter:
            raise ValueError("cursor scope does not match")
        updated_at = datetime.fromisoformat(payload["updated_at"])
        created_at = datetime.fromisoformat(payload["created_at"])
        session_id = payload["id"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise service_error("validation_error", "cursor is invalid.", status=422) from error
    if (
        updated_at.tzinfo is None
        or created_at.tzinfo is None
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise service_error("validation_error", "cursor is invalid.", status=422)
    return updated_at, created_at, session_id


def _project_catalog_service_error(error: ProjectCatalogError) -> ServiceError:
    message = str(error)
    if "unavailable" in message:
        return service_error(
            "not_found",
            "Project directory is unavailable.",
            status=404,
            field_errors={"path": "must name an existing directory"},
        )
    if "catalog" in message or "entries" in message or "format" in message:
        return service_error(
            "persistence_error", "The Project catalog could not be read safely.", status=500
        )
    if "overlaps Agent Home" in message:
        detail = "must not overlap Agent Home"
    elif "absolute directory" in message:
        detail = "must be an absolute directory"
    else:
        detail = "must identify a usable local directory"
    return service_error(
        "validation_error",
        "Project path is invalid.",
        status=422,
        field_errors={"path": detail},
    )


def _configuration_request_fingerprint(
    client_id: str | None,
    action: str,
    payload: object,
) -> str:
    serialized = json.dumps(
        [client_id, action, payload],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


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
    web_control_credential: str | None = None
    connected: bool = False
    sink: ServiceSink | None = None
    stream_id: str = field(default_factory=lambda: str(uuid4()))
    sequence: int = 0
    events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=256))
    results: dict[str, dict[str, object]] = field(default_factory=dict)
    inflight: dict[str, asyncio.Task[dict[str, object]]] = field(default_factory=dict)
    rename_results: dict[str, tuple[tuple[object, ...], dict[str, object]]] = field(
        default_factory=dict
    )
    session_delete_results: dict[str, tuple[tuple[object, ...], dict[str, object]]] = field(
        default_factory=dict
    )
    management_results: dict[str, tuple[str, dict[str, object]]] = field(default_factory=dict)
    claimed: set[tuple[str, str]] = field(default_factory=set)
    attached_workspaces: set[str] = field(default_factory=set)
    current_workspace_id: str | None = None
    current_session_id: str | None = None
    disconnect_task: asyncio.Task[None] | None = None
    reconnect_deadline: float | None = None
    delivery_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    management_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    subscribed: bool = True
    resync_required: bool = False
    expired: bool = False
    reconnect_blocked: bool = False
    blocked_workspace_keys: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SessionDeletionClaim:
    """A cleanup-only Claim never owns or reopens an Agent Loop."""

    session_id: str
    client_id: str
    version: int
    credential: str


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
    release_task: asyncio.Task[None] | None = None
    schedule: bool = False


@dataclass(slots=True)
class _PreparedWorkspaceGeneration:
    """Generation resources prepared before a Workspace pointer is switched."""

    configuration: UserConfiguration
    previous: WorkspaceRuntime
    runtime: WorkspaceRuntime
    exec_host: ExecHost
    schedule_permission: RuntimePermissionControl
    schedule: JobSchedule
    schedule_jobs: tuple[ScheduleJob, ...] = field(init=False)
    source_loops: tuple[tuple[str, _LoopState, AgentLoop, str | None], ...]
    loops: list[tuple[_LoopState, AgentLoop, MessageBus]]
    schedule_loops: list[tuple[str, _LoopState, AgentLoop, MessageBus]]


@dataclass(slots=True)
class _RetiredWorkspaceGeneration:
    runtime: WorkspaceRuntime
    loops: tuple[_LoopState, ...]


@dataclass(frozen=True, slots=True)
class _RestorePlanReference:
    """Wire-safe reference to the service-owned Restore plan."""

    anchor_id: int


@dataclass(slots=True)
class _ProjectRemoval:
    """One persisted Project removal and its in-process completion task."""

    project_id: str
    operation_id: str
    path: Path
    status: str = "removing"
    error: str | None = None
    workspace_id: str | None = None
    affected_client_ids: tuple[str, ...] = ()
    notification_client_ids: tuple[str, ...] = ()
    task: asyncio.Task[None] | None = None


class ServiceConfirmationPresenter(ConfirmationPresenter):
    """Bridge the existing one-shot coordinator to authenticated clients."""

    def __init__(self, service: LocalService) -> None:
        self._service = service
        self._wire_tokens: dict[object, str] = {}
        self._wire_sources: dict[object, tuple[str | None, str | None, str | None]] = {}
        self._wire_requests: dict[object, asyncio.Task[None]] = {}

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
        self._wire_requests[token] = task
        task.add_done_callback(_consume_task_result)

    async def dismiss_confirmation(self, token: object) -> None:
        wire_token = self._wire_tokens.pop(token, None)
        if wire_token is None:
            return
        workspace_id, session_id, run_id = self._wire_sources.pop(token, (None, None, None))
        await self._emit_resolved(
            self._wire_requests.pop(token),
            wire_token,
            workspace_id,
            session_id,
            run_id,
        )

    async def _emit_resolved(
        self,
        requested: asyncio.Task[None],
        wire_token: str,
        workspace_id: str | None,
        session_id: str | None,
        run_id: str | None,
    ) -> None:
        await asyncio.gather(requested, return_exceptions=True)
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
        audience = set(self._service.workspace_audience(workspace_id))
        workspace = self._service._workspaces.get(workspace_id)
        if workspace is not None:
            workspace_key = os.path.normcase(str(workspace.workspace_path))
            registered = any(
                os.path.normcase(str(record.path.resolve(strict=False))) == workspace_key
                for record in self._service.projects.list()
            )
            if registered:
                # Web clients can inspect the account-global Project catalog, while
                # CLI clients only receive confirmations for attached Workspaces.
                audience.update(
                    client.client_id
                    for client in self._service._clients.values()
                    if client.kind == "web"
                )
        return tuple(
            client.client_id
            for client in self._service._clients.values()
            if client.client_id in audience
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
                        self._emit_resolved(
                            self._wire_requests.pop(token),
                            wire_token,
                            workspace_id,
                            session_id,
                            run_id,
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
        self._deletion_claims: dict[str, SessionDeletionClaim] = {}
        self._claim_versions: dict[str, int] = {}
        self._draft_clients: dict[str, str] = {}
        self._schedule_loops: dict[str, _LoopState] = {}
        self._retired_generations: list[_RetiredWorkspaceGeneration] = []
        self._restore_plans: dict[tuple[str, int], Any] = {}
        self._restore_results: dict[tuple[str, str], Any] = {}
        self._restore_owner: str | None = None
        self._restore_session_id: str | None = None
        self._restore_loop: AgentLoop | None = None
        self._restore_schedule_paused = False
        self._restore_blocked = False
        self._restore_commit_task: asyncio.Task[Any] | None = None
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
                # Admission was checked at reservation; accepted occurrences must drain.
                loop = await self._get_schedule_loop(job.job_id)
                await loop.loop.run_schedule_job(job)

            async def execute_user_occurrence(occurrence: ScheduleOccurrence) -> None:
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
            self.runtime.schedule_service.set_admission_guard(lambda: self.schedule_admitted)
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
            and not self.service.configuration_transition_active
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

    async def pause_schedule_admission(self) -> None:
        if not self._started or self._closed:
            return
        self._schedule_admitted = False
        await self.schedule_service.pause_admission()

    async def create_draft(self, client_id: str, *, reuse_startup_session: bool = True) -> str:
        if self._closed or self.service.configuration_transition_active:
            raise service_error("admission_closed", "Workspace admission is closed.")
        startup_session_id = None if self.runtime is None else self.runtime.startup_session_id
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
            if self._closed or self.service.configuration_transition_active:
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

    async def release(self, client_id: str, session_id: str, *, close_idle: bool = True) -> None:
        async with self._lock:
            await self._release_unlocked(client_id, session_id, close_idle=close_idle)

    async def _release_unlocked(
        self, client_id: str, session_id: str, *, close_idle: bool = True
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
            or await loop_state.bus.inbound_snapshot()
        ):
            raise service_error(
                "session_busy", "Conversation Session still has accepted work.", retryable=True
            )
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
        if close_idle and loop_state is not None and not loop_state.loop.has_active_run:
            await self._close_loop(session_id)

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
            try:
                if loop_state is not None:
                    await self.service.confirmation.cancel_generation(loop_state.loop.generation_id)
                    try:
                        await loop_state.loop.cancel_active_run()
                    except RuntimeError:
                        pass
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

    def session_snapshot(self, session_id: str) -> dict[str, object]:
        """Return the claimed Session conversation plus presentation-safe Restore Anchors."""
        self._ensure_session_available(session_id)
        loop_state = self._loops.get(session_id)
        if loop_state is None:
            raise service_error("not_found", "Conversation Session was not found.", status=404)
        projection = loop_state.loop.project_foreground_conversation()
        return {
            "session_id": projection.session_id,
            "messages": list(projection.messages),
            "restore_anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "run_token": str(anchor.run_token),
                    "content": anchor.content,
                    "timestamp": anchor.timestamp,
                }
                for anchor in loop_state.loop.session.restore_candidates()
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

            try:
                begin_session_deletion(self.workspace_state, session_id)
                if loop_state is not None:
                    await loop_state.loop.wait_for_restore_idle()
                    if (
                        loop_state.loop.has_active_run
                        or loop_state.run_ids
                        or await loop_state.bus.inbound_snapshot()
                    ):
                        raise service_error(
                            "session_busy",
                            "Conversation Session still has accepted work.",
                            retryable=True,
                        )
                    await self._close_loop(session_id)
                delete_session_data(self.workspace_state, session_id)
            except ServiceError:
                raise
            except (OSError, RuntimeError, ValueError) as error:
                raise service_error(
                    "persistence_error",
                    "Conversation Session data could not be deleted safely; retry the operation.",
                    status=500,
                    retryable=True,
                ) from error

            self._claims.pop(session_id, None)
            self._deletion_claims.pop(session_id, None)
            self._draft_clients.pop(session_id, None)
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
    ) -> Any:
        """Build the existing typed Management dispatcher for one Claim."""
        from myclaw.management.commands import ManagementCommandDispatcher
        from myclaw.management.service import (
            ManagementError,
            ManagementViewService,
            RestoreListingReport,
        )

        runtime = self.runtime
        if runtime is None:
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

        def current_loop() -> AgentLoop:
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
            await self.service.claim(client_id, self.workspace_id, target_session_id)

        async def restore_listing() -> RestoreListingReport:
            from myclaw.management.service import RestoreListingReport

            async with self._lock:
                self._require_admitted()
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
                current_loop()
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
            loop = current_loop()
            stored = self._restore_plans.get((client_id, plan.anchor_id))
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
                try:
                    await self._release_restore_barrier(client_id)
                finally:
                    if self._restore_commit_task is restore_task:
                        self._restore_commit_task = None

        async def restore_result() -> Any:
            from myclaw.agent.session.restore import RestoreManager

            session_id = current_loop().session.session_id
            result = RestoreManager(
                self.workspace_state, session_id, now=local_now
            ).completed_result()
            return result

        async def restore_acknowledge_failure() -> Any:
            from myclaw.agent.session.restore import RestoreManager

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
            reasoning_effort_control=runtime.router,
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
        management.bind_restore_acknowledge_failure(restore_acknowledge_failure)
        return ManagementCommandDispatcher(management)

    def _require_admitted(self, *, allow_configuration: bool = False) -> None:
        if (
            self._closed
            or self._restore_blocked
            or (not self._schedule_admitted and self.service.state in {"draining", "stopped"})
            or (self.service.configuration_transition_active and not allow_configuration)
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
            target_client_ids=(client_id,),
        )

    async def _create_loop(self, session_id: str | None, *, client_id: str | None) -> _LoopState:
        if self.runtime is None or self.workspace_state is None or self._exec_host is None:
            raise RuntimeError("Workspace service runtime is not ready")
        permission_control = (
            self.service.client_permission(client_id)
            if client_id is not None
            else self._schedule_permission
        )
        loop, bus = await self._create_agent_loop(
            runtime=self.runtime,
            configuration=self.configuration,
            schedule_service=self.schedule_service,
            exec_host=self._exec_host,
            session_id=session_id,
            permission_control=permission_control,
        )
        state = _LoopState(loop=loop, bus=bus, owner_client_id=client_id)
        self._loops[loop.session.session_id] = state
        if client_id is not None:
            state.output_task = asyncio.create_task(self._forward_output(state))
        return state

    async def _create_agent_loop(
        self,
        *,
        runtime: WorkspaceRuntime,
        configuration: UserConfiguration,
        schedule_service: ScheduleService,
        exec_host: ExecHost,
        session_id: str | None,
        permission_control: RuntimePermissionControl,
        session: Session | None = None,
        bus: MessageBus | None = None,
    ) -> tuple[AgentLoop, MessageBus]:
        selected_bus = MessageBus() if bus is None else bus
        loop_kwargs: dict[str, Any] = {
            "workspace_path": self.workspace_path,
            "workspace_state": self.workspace_state,
            "agent_home": self.service.agent_home,
            "configuration": configuration,
            "bus": selected_bus,
            "schedule_service": schedule_service,
            "model_router": runtime.router,
            "memory_manager": runtime.memory_manager,
            "session_id": session_id,
            "now": local_now,
            "new_uuid": uuid4,
            "monotonic_now": monotonic,
            "mcp_tools": runtime.mcp_snapshot,
            "mcp_keywords": runtime.mcp_keywords,
            "exec_host": exec_host,
            "permission_control": permission_control,
            "configured_schedule_level": configuration.runtime.permission_level,
        }
        loop = (
            AgentLoop.with_session(session, **loop_kwargs)
            if session is not None
            else AgentLoop(**loop_kwargs)
        )
        loop.bind_confirmation_requester(self.service.confirmation.request)
        loop.preflight()
        await loop.start()
        return loop, selected_bus

    async def _get_schedule_loop(self, job_id: str) -> _LoopState:
        state = self._schedule_loops.get(job_id)
        if state is not None:
            return state
        state = await self._create_loop(None, client_id=None)
        state.schedule = True
        self._schedule_loops[job_id] = state
        return state

    async def wait_for_generation_idle(self) -> None:
        """Pause new work and wait for active foreground, Schedule, and Dream work."""
        if not self._started or self._closed or self.runtime is None:
            return
        await self.schedule_service.pause_and_wait_idle()
        await self.runtime.dream.wait_until_idle()
        while True:
            active = self._restore_owner is not None or self._restore_commit_task is not None
            if self._closed:
                return
            for state in tuple(self._loops.values()):
                try:
                    owner = self.service._clients.get(state.owner_client_id or "")
                    active = (
                        active
                        or (state.owner_client_id is not None and (owner is None or owner.expired))
                        or state.loop.has_active_run
                        or bool(state.run_ids)
                        or bool(await state.bus.inbound_snapshot())
                    )
                    await state.loop.wait_for_restore_idle()
                except RuntimeError:
                    continue
            if not active:
                return
            await asyncio.sleep(0.01)

    async def prepare_configuration(
        self, configuration: UserConfiguration
    ) -> _PreparedWorkspaceGeneration:
        """Prepare a complete replacement generation without publishing it."""
        if self.runtime is None or self.workspace_state is None or self._exec_host is None:
            raise RuntimeError("Workspace service runtime is not ready")
        await self.wait_for_generation_idle()
        # Late automatic titles update memory after the last Run snapshot was saved.
        for state in tuple(self._loops.values()):
            await state.loop.session.persist_pending_automatic_title()
        previous = self.runtime
        source_loops = tuple(
            (session_id, state, state.loop, state.owner_client_id)
            for session_id, state in self._loops.items()
        )
        candidate = previous.create_replacement(configuration)
        schedule = JobSchedule.from_cron_input(configuration.memory.schedule, get_localzone_name())
        staged: list[tuple[_LoopState, AgentLoop, MessageBus]] = []
        schedule_job_states = list(self._schedule_loops.items())
        staged_schedule_loops: list[tuple[str, _LoopState, AgentLoop, MessageBus]] = []
        try:
            await candidate.start_replacement(previous)
            candidate.schedule_service.set_admission_guard(lambda: self.schedule_admitted)
            candidate.prepare_replacement_schedule()
            candidate_exec_host = create_exec_host(
                resolve_exec_shell(configuration.runtime.exec_shell)
            )
            candidate_schedule_permission = RuntimePermissionControl(
                configuration.runtime.permission_level
            )
            for state in tuple(self._loops.values()):
                if state.schedule:
                    continue
                permission = (
                    self.service.client_permission(state.owner_client_id)
                    if state.owner_client_id is not None
                    else candidate_schedule_permission
                )
                loop, bus = await self._create_agent_loop(
                    runtime=candidate,
                    configuration=configuration,
                    schedule_service=candidate.schedule_service,
                    exec_host=candidate_exec_host,
                    session_id=state.loop.session.session_id,
                    permission_control=permission,
                    session=state.loop.session.clone(),
                )
                staged.append((state, loop, bus))
            for job_id, state in schedule_job_states:
                loop, bus = await self._create_agent_loop(
                    runtime=candidate,
                    configuration=configuration,
                    schedule_service=candidate.schedule_service,
                    exec_host=candidate_exec_host,
                    session_id=state.loop.session.session_id,
                    permission_control=candidate_schedule_permission,
                    session=state.loop.session.clone(),
                )
                staged_schedule_loops.append((job_id, state, loop, bus))
            return _PreparedWorkspaceGeneration(
                configuration=configuration,
                previous=previous,
                runtime=candidate,
                exec_host=candidate_exec_host,
                schedule_permission=candidate_schedule_permission,
                schedule=schedule,
                source_loops=source_loops,
                loops=staged,
                schedule_loops=staged_schedule_loops,
            )
        except BaseException:
            await self._discard_configuration_resources(
                candidate,
                tuple(_LoopState(loop, bus, None) for _, loop, bus in staged)
                + tuple(_LoopState(loop, bus, None) for _, _, loop, bus in staged_schedule_loops),
            )
            raise

    async def discard_prepared_configuration(self, prepared: _PreparedWorkspaceGeneration) -> None:
        await self._discard_configuration_resources(
            prepared.runtime,
            tuple(_LoopState(loop, bus, None) for _, loop, bus in prepared.loops)
            + tuple(_LoopState(loop, bus, None) for _, _, loop, bus in prepared.schedule_loops),
        )

    async def _discard_configuration_resources(
        self, runtime: WorkspaceRuntime, loops: tuple[_LoopState, ...]
    ) -> None:
        retired = _RetiredWorkspaceGeneration(runtime, loops)
        self._retired_generations.append(retired)
        try:
            await self.retire_configuration(retired)
        except Exception:
            # Failed cleanup remains owned and is retried by Workspace shutdown.
            pass

    def prepared_configuration_is_current(self, prepared: _PreparedWorkspaceGeneration) -> bool:
        if (
            self._closed
            or self.runtime is not prepared.previous
            or self._lock.locked()
            or self._restore_owner is not None
            or self._restore_commit_task is not None
        ):
            return False
        if tuple(self._loops) != tuple(item[0] for item in prepared.source_loops):
            return False
        candidates = {state.loop.session.session_id: loop for state, loop, _bus in prepared.loops}
        candidates.update(
            {
                state.loop.session.session_id: loop
                for _job, state, loop, _bus in prepared.schedule_loops
            }
        )
        for session_id, state, loop, owner in prepared.source_loops:
            if self._loops.get(session_id) is not state or state.loop is not loop:
                return False
            if state.owner_client_id != owner:
                return False
            client = self.service._clients.get(owner or "")
            if owner is not None and (client is None or client.expired):
                return False
            candidate = candidates.get(session_id)
            if candidate is None or (
                candidate.session.messages != loop.session.messages
                or candidate.session.metadata != loop.session.metadata
                or candidate.session.updated_at != loop.session.updated_at
            ):
                return False
        return True

    def activate_configuration(
        self, prepared: _PreparedWorkspaceGeneration
    ) -> _RetiredWorkspaceGeneration:
        """Publish a prepared generation after every Workspace has passed preflight."""
        old_loops = tuple(prepared.loops)
        old_schedule_loops = tuple(prepared.schedule_loops)
        claims_by_session = {claim.session_id: claim for claim in self._claims.values()}
        retired = _RetiredWorkspaceGeneration(prepared.previous, tuple(self._loops.values()))
        self._retired_generations.append(retired)
        prepared.runtime.schedule_service.inherit_generation_state(
            prepared.previous.schedule_service,
            jobs=prepared.schedule_jobs,
        )
        for state in retired.loops:
            if state.output_task is not None:
                state.output_task.cancel()
            if state.release_task is not None:
                state.release_task.cancel()
        self.runtime = prepared.runtime
        self.configuration = prepared.configuration
        self._schedule_permission = prepared.schedule_permission
        self._exec_host = prepared.exec_host
        self._loops.clear()
        self._schedule_loops.clear()
        for old_state, new_loop, new_bus in old_loops:
            session_id = new_loop.session.session_id
            state = _LoopState(new_loop, new_bus, old_state.owner_client_id)
            self._loops[session_id] = state
            claim = claims_by_session.get(session_id)
            if claim is not None:
                claim.loop = new_loop
            if state.owner_client_id is not None:
                state.output_task = asyncio.create_task(self._forward_output(state))
        for job_id, _old_state, new_loop, new_bus in old_schedule_loops:
            state = _LoopState(new_loop, new_bus, None, schedule=True)
            self._loops[state.loop.session.session_id] = state
            self._schedule_loops[job_id] = state
        return retired

    async def retire_configuration(self, retired: _RetiredWorkspaceGeneration) -> None:
        errors: list[Exception] = []
        for state in retired.loops:
            try:
                await self._close_loop_state(state, abort=True)
            except Exception as error:
                errors.append(error)
        try:
            await retired.runtime.close()
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("Runtime generation retirement failed", errors)
        self._retired_generations.remove(retired)

    async def _forward_output(self, state: _LoopState) -> None:
        session_id = state.loop.session.session_id
        try:
            while not self._closed and self._loops.get(session_id) is state:
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
                    if state.release_task is None or state.release_task.done():
                        state.release_task = asyncio.create_task(
                            self._release_switched_claim_when_idle(session_id, state)
                        )
                        state.release_task.add_done_callback(_consume_task_result)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _release_switched_claim_when_idle(self, session_id: str, state: _LoopState) -> None:
        while True:
            if self._closed or self._loops.get(session_id) is not state:
                return
            try:
                active = state.loop.has_active_run
            except RuntimeError:
                return
            if not active and not state.run_ids:
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
            if state.loop.has_active_run or state.run_ids or await state.bus.inbound_snapshot():
                return
        except RuntimeError:
            return
        await self.release(client_id, session_id)

    async def _close_loop_state(self, state: _LoopState, *, abort: bool = False) -> None:
        if state.release_task is not None:
            release_task = state.release_task
            state.release_task = None
            if release_task is not asyncio.current_task():
                release_task.cancel()
                await asyncio.gather(release_task, return_exceptions=True)
        if state.output_task is not None:
            output_task = state.output_task
            state.output_task = None
            if output_task is not asyncio.current_task():
                output_task.cancel()
                await asyncio.gather(output_task, return_exceptions=True)
        if abort:
            await self.service.confirmation.cancel_generation(state.loop.generation_id)
            await state.loop.abort()
        else:
            await state.loop.close()
            await self.service.confirmation.cancel_generation(state.loop.generation_id)

    async def _close_loop(self, session_id: str, *, abort: bool = False) -> None:
        state = self._loops.get(session_id)
        if state is None:
            return
        await self._close_loop_state(state, abort=abort)
        self._loops.pop(session_id, None)
        for job_id, candidate in tuple(self._schedule_loops.items()):
            if candidate is state:
                self._schedule_loops.pop(job_id, None)

    async def _close_schedule_loop(
        self, job_id: str, *, expected: _LoopState | None = None, abort: bool = False
    ) -> None:
        state = self._schedule_loops.get(job_id)
        if state is None or (expected is not None and state is not expected):
            return
        await self._close_loop_state(state, abort=abort)
        if self._loops.get(state.loop.session.session_id) is state:
            self._loops.pop(state.loop.session.session_id, None)
        self._schedule_loops.pop(job_id, None)

    async def close(self) -> None:
        async with self._lock:
            if self._closed and not self._close_failed:
                return
            self._closed = True
            self._schedule_admitted = False
            try:
                restore_task = self._restore_commit_task
                if restore_task is not None and restore_task is not asyncio.current_task():
                    await asyncio.shield(asyncio.gather(restore_task, return_exceptions=True))
                if self._restore_blocked:
                    raise service_error(
                        "restore_pending",
                        "The active Restore transaction requires recovery before Project removal.",
                        retryable=True,
                    )
                if self._restore_owner is not None:
                    await self._release_restore_barrier(self._restore_owner)
                if self.runtime is not None:
                    await self.runtime.abort_dream()
                    await self.runtime.close(
                        close_foreground=lambda: self._close_all_loops(),
                        drain_confirmation_aborts=True,
                    )
                else:
                    await self._close_all_loops()
                for retired in tuple(self._retired_generations):
                    await self.retire_configuration(retired)
            except BaseException:
                self._close_failed = True
                raise
            else:
                self._close_failed = False

    async def _close_all_loops(self) -> None:
        """Drain accepted work into durable terminal outcomes during lifecycle shutdown."""
        for session_id in tuple(self._loops):
            await self._close_loop(session_id)
        for job_id in tuple(self._schedule_loops):
            await self._close_schedule_loop(job_id)


class LocalService:
    """Own one local service instance and expose typed command behavior."""

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
        self.state = "starting"
        self.confirmation = ToolConfirmationCoordinator()
        self._presenter = ServiceConfirmationPresenter(self)
        self._clients: dict[str, ClientState] = {}
        self._client_by_reconnect: dict[str, str] = {}
        self._workspaces: dict[str, WorkspaceServiceRuntime] = {}
        self._workspace_keys: dict[str, str] = {}
        self._initial_configuration_candidates: list[WorkspaceServiceRuntime] = []
        self._global_reconnect_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_failed = False
        self._closed = asyncio.Event()
        self._lock = asyncio.Lock()
        self._schedule_admission_lock = asyncio.Lock()
        self._schedule_mutation_lock = asyncio.Lock()
        self._schedule_mutation_results: dict[str, tuple[str, dict[str, object]]] = {}
        self._schedule_removal_jobs: dict[tuple[str, str], tuple[ScheduleJob, bool]] = {}
        self._project_lifecycle_lock = asyncio.Lock()
        self._project_removals: dict[str, _ProjectRemoval] = {}
        self._config_loader = ConfigLoader(agent_home)
        self._config_lock = asyncio.Lock()
        self._config_request_results: dict[str, dict[str, object]] = {}
        self._config_request_fingerprints: dict[str, str] = {}
        self._config_saved_configuration = configuration
        self._config_saved_revision: str | None = None
        self._config_active_revision: str | None = None
        self._config_pending_revision: str | None = None
        self._config_fields: dict[str, dict[str, object]] | None = None
        self._config_status = "active" if configuration is not None else "pending-repair"
        self._config_state = "active" if configuration is not None else "missing"
        self._config_repair_required = configuration is None
        self._config_backup_required = False
        self._config_requires_secret_reentry = configuration is None
        self._config_projection_error: dict[str, str] | None = None
        self._config_waiting_for: tuple[str, ...] = ()
        self._config_error: dict[str, str] | None = None
        self._config_apply_task: asyncio.Task[None] | None = None
        self._configuration_transition_active = False
        self.projects = ProjectCatalog(agent_home)

    async def start(self) -> None:
        if self.state != "starting":
            return
        self.agent_home.initialize()
        snapshot = self._config_loader.web_snapshot()
        self._config_saved_revision = snapshot.revision
        self._config_fields = {section: dict(values) for section, values in snapshot.fields.items()}
        self._config_state = snapshot.state
        self._config_repair_required = snapshot.repair_required
        self._config_backup_required = snapshot.backup_required
        self._config_requires_secret_reentry = snapshot.requires_secret_reentry
        self._config_projection_error = None if snapshot.error is None else dict(snapshot.error)
        if snapshot.state == "active":
            self.configuration = snapshot.configuration
            self._config_saved_configuration = snapshot.configuration
            self._config_active_revision = snapshot.revision
            self._config_status = "active"
        else:
            self.configuration = None
            self._config_saved_configuration = None
            self._config_active_revision = None
            self._config_status = "pending-repair"
        self.confirmation.bind_presenter(self._presenter)
        for record in self.projects.list():
            if record.schedule_state == "removing" and record.removal_error is None:
                self.projects.record_removal_failure(
                    record.project_id,
                    "Project removal was interrupted; retry to finish stopping its work.",
                )
            if (
                snapshot.state == "active"
                and record.schedule_state == "available"
                and record.path.is_dir()
            ):
                await self._get_or_create_workspace(record.path)
        self.state = "ready"
        await self._reconcile_schedule_admission()
        self._global_reconnect_task = asyncio.create_task(self._stop_after_grace())

    @property
    def workspaces(self) -> Mapping[str, WorkspaceServiceRuntime]:
        return self._workspaces

    @property
    def configuration_transition_active(self) -> bool:
        return self._configuration_transition_active

    @property
    def configuration_ready(self) -> bool:
        """Return whether new Agent and Schedule work may be admitted."""
        return (
            self.configuration is not None
            and self._config_active_revision is not None
            and not self._configuration_transition_active
        )

    def config_view(self) -> dict[str, object]:
        """Return the safe persisted projection and active-generation status."""
        snapshot = self._config_loader.web_snapshot()
        self._config_state = snapshot.state
        self._config_repair_required = snapshot.repair_required
        self._config_backup_required = snapshot.backup_required
        self._config_requires_secret_reentry = snapshot.requires_secret_reentry
        self._config_projection_error = None if snapshot.error is None else dict(snapshot.error)
        self._config_fields = {section: dict(values) for section, values in snapshot.fields.items()}
        if snapshot.state != "active":
            self._config_saved_revision = snapshot.revision
            self._config_pending_revision = snapshot.revision
            self._config_waiting_for = tuple(self._workspaces)
            if self.configuration is None or self._config_active_revision is None:
                self._config_saved_configuration = None
                self._config_status = "pending-repair"
            else:
                self._config_status = "failed-to-apply"
                self._config_error = dict(snapshot.error or _CONFIG_INVALID_ERROR)
            self._configuration_transition_active = False
            return self._config_response()

        if self._config_saved_revision is None:
            self._config_saved_revision = snapshot.revision
        elif snapshot.revision != self._config_saved_revision:
            self._config_saved_revision = snapshot.revision
            self._config_saved_configuration = snapshot.configuration
            self._config_pending_revision = snapshot.revision
            self._config_status = "pending"
            self._config_error = None
            self._config_waiting_for = tuple(self._workspaces)
            self._configuration_transition_active = True
            self._start_configuration_application()
        elif self._config_saved_configuration is None:
            self._config_saved_configuration = snapshot.configuration
            self._config_pending_revision = snapshot.revision
            self._config_status = "pending"
            self._configuration_transition_active = True
            self._start_configuration_application()
        return self._config_response()

    def _start_configuration_application(self) -> None:
        if self.state in {"draining", "stopped"}:
            return
        if self._config_apply_task is None or self._config_apply_task.done():
            self._config_apply_task = asyncio.create_task(self._apply_configuration_loop())
            self._config_apply_task.add_done_callback(_consume_task_result)

    def _configuration_request_result(
        self, request_id: str, fingerprint: str
    ) -> dict[str, object] | None:
        previous = self._config_request_fingerprints.get(request_id)
        if previous is not None and previous != fingerprint:
            raise service_error(
                "request_conflict",
                "Request ID was already used for a different operation.",
                status=409,
            )
        return self._config_request_results.get(request_id)

    def _config_response(self) -> dict[str, object]:
        saved_revision = self._config_saved_revision or ConfigLoader.revision_from_bytes(b"")
        fields = {} if self._config_fields is None else self._config_fields
        return {
            "revision": saved_revision,
            "fields": {section: dict(values) for section, values in fields.items()},
            "configuration": {
                "state": self._config_state,
                "repair_required": self._config_repair_required,
                "backup_required": self._config_backup_required,
                "requires_secret_reentry": self._config_requires_secret_reentry,
                "error": (
                    None
                    if self._config_projection_error is None
                    else dict(self._config_projection_error)
                ),
            },
            "application": {
                "status": self._config_status,
                "saved_revision": saved_revision,
                "active_revision": self._config_active_revision,
                "pending_revision": self._config_pending_revision,
                "waiting_for": self._configuration_waiting_reasons(),
                "error": None if self._config_error is None else dict(self._config_error),
            },
        }

    def _configuration_waiting_reasons(self) -> list[str]:
        if self._config_status != "pending":
            return []
        reasons: set[str] = set()
        for workspace in self._workspaces.values():
            if workspace.runtime is None:
                continue
            if workspace.schedule_service.has_pending_work:
                reasons.add("schedule")
            if workspace.runtime.dream.is_running:
                reasons.add("dream")
            if workspace._restore_owner is not None or workspace._restore_commit_task is not None:
                reasons.add("restore")
            for state in workspace.loops.values():
                try:
                    active_run = state.loop.has_active_run
                except RuntimeError:
                    active_run = False
                if active_run or state.run_ids:
                    reasons.add("schedule" if state.schedule else "foreground")
                if state.loop.has_pending_title:
                    reasons.add("title")
                if state.bus.has_pending_input:
                    reasons.add("queued-input")
                client = self._clients.get(state.owner_client_id or "")
                if state.owner_client_id is not None and (client is None or client.expired):
                    reasons.add("client-cleanup")
        return sorted(reasons) if reasons else ["preparing"]

    async def update_configuration(
        self,
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
    ) -> dict[str, object]:
        """Persist one safe configuration patch and queue generation application."""
        if not request_id:
            raise service_error("validation_error", "Request ID is required.", status=422)
        if client_id is not None:
            self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        fingerprint = _configuration_request_fingerprint(
            client_id,
            "patch",
            {"revision": expected_revision, "fields": fields, "secrets": secrets or {}},
        )
        async with self._config_lock:
            existing = self._configuration_request_result(request_id, fingerprint)
            if existing is not None:
                return existing
            try:
                result = self._config_loader.patch_editable_fields(
                    expected_revision, fields, secrets
                )
            except ConfigRevisionConflict as error:
                raise service_error(
                    "config_revision_conflict",
                    error.error.message,
                    status=409,
                    retryable=True,
                ) from error
            except ConfigFieldError as error:
                raise service_error(
                    error.error.code,
                    error.error.message,
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except ConfigError as error:
                raise service_error(
                    error.error.code,
                    "The complete User Configuration is invalid.",
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "User Configuration could not be written.",
                    status=500,
                    retryable=True,
                ) from error

            self._config_saved_configuration = result.configuration
            self._config_saved_revision = result.revision
            self._config_fields = {
                section: dict(values) for section, values in result.fields.items()
            }
            self._config_state = "active"
            self._config_repair_required = False
            self._config_backup_required = False
            self._config_requires_secret_reentry = False
            self._config_projection_error = None
            self._config_pending_revision = (
                None if result.revision == self._config_active_revision else result.revision
            )
            self._config_status = "active" if self._config_pending_revision is None else "pending"
            self._config_error = None
            self._config_waiting_for = tuple(self._workspaces)
            self._configuration_transition_active = self._config_status == "pending"
            response = self._config_response()
            self._config_request_results[request_id] = response
            self._config_request_fingerprints[request_id] = fingerprint
            if len(self._config_request_results) > 256:
                oldest = next(iter(self._config_request_results))
                self._config_request_results.pop(oldest, None)
                self._config_request_fingerprints.pop(oldest, None)
            if self._config_pending_revision is not None and (
                self._config_apply_task is None or self._config_apply_task.done()
            ):
                self._config_apply_task = asyncio.create_task(self._apply_configuration_loop())
                self._config_apply_task.add_done_callback(_consume_task_result)
        await self._reconcile_schedule_admission()
        await self._emit_configuration_event()
        return response

    async def repair_configuration(
        self,
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
    ) -> dict[str, object]:
        """Persist a first-use or malformed-file repair before activating a generation."""
        if not request_id:
            raise service_error("validation_error", "Request ID is required.", status=422)
        if client_id is not None:
            self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        fingerprint = _configuration_request_fingerprint(
            client_id,
            "repair",
            {"revision": expected_revision, "fields": fields, "secrets": secrets or {}},
        )
        async with self._config_lock:
            existing = self._configuration_request_result(request_id, fingerprint)
            if existing is not None:
                return existing
            try:
                result = self._config_loader.repair_editable_fields(
                    expected_revision, fields, secrets
                )
            except ConfigRevisionConflict as error:
                raise service_error(
                    "config_revision_conflict",
                    error.error.message,
                    status=409,
                    retryable=True,
                ) from error
            except ConfigFieldError as error:
                raise service_error(
                    error.error.code,
                    error.error.message,
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except ConfigError as error:
                raise service_error(
                    error.error.code,
                    "The complete User Configuration is invalid.",
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "User Configuration could not be written.",
                    status=500,
                    retryable=True,
                ) from error

            self._config_saved_configuration = result.configuration
            self._config_saved_revision = result.revision
            self._config_fields = {
                section: dict(values) for section, values in result.fields.items()
            }
            self._config_state = "active"
            self._config_repair_required = False
            self._config_backup_required = False
            self._config_requires_secret_reentry = False
            self._config_projection_error = None
            self._config_pending_revision = (
                None if result.revision == self._config_active_revision else result.revision
            )
            self._config_status = "active" if self._config_pending_revision is None else "pending"
            self._config_error = None
            self._config_waiting_for = tuple(self._workspaces)
            self._configuration_transition_active = self._config_status == "pending"
            response = {
                "backup_id": result.backup_id,
                **self._config_response(),
            }
            self._config_request_results[request_id] = response
            self._config_request_fingerprints[request_id] = fingerprint
            if len(self._config_request_results) > 256:
                oldest = next(iter(self._config_request_results))
                self._config_request_results.pop(oldest, None)
                self._config_request_fingerprints.pop(oldest, None)
            if self._config_pending_revision is not None and (
                self._config_apply_task is None or self._config_apply_task.done()
            ):
                self._config_apply_task = asyncio.create_task(self._apply_configuration_loop())
                self._config_apply_task.add_done_callback(_consume_task_result)
        await self._reconcile_schedule_admission()
        await self._emit_configuration_event()
        return response

    async def retry_configuration(
        self, request_id: str, revision: str, *, client_id: str | None = None
    ) -> dict[str, object]:
        """Retry the latest saved candidate after a failed generation application."""
        if not request_id:
            raise service_error("validation_error", "Request ID is required.", status=422)
        if client_id is not None:
            self._require_client(client_id)
        if self.state in {"draining", "stopped"}:
            raise service_error("admission_closed", "The local service is stopping.")
        fingerprint = json.dumps([client_id, "retry", revision])
        async with self._config_lock:
            existing = self._configuration_request_result(request_id, fingerprint)
            if existing is not None:
                return existing
            self.config_view()
            if revision != self._config_saved_revision:
                raise service_error(
                    "config_revision_conflict",
                    "The saved User Configuration revision is no longer current.",
                    status=409,
                    retryable=True,
                )
            if self._config_saved_configuration is None:
                raise service_error(
                    "config_invalid", "User Configuration is unavailable.", status=422
                )
            snapshot = self._config_loader.web_snapshot()
            if snapshot.state != "active":
                raise service_error(
                    "config_invalid", "The complete User Configuration is invalid.", status=422
                )
            self._config_status = "pending"
            self._config_pending_revision = revision
            self._config_error = None
            self._configuration_transition_active = True
            self._config_waiting_for = tuple(self._workspaces)
            response = self._config_response()
            self._config_request_results[request_id] = response
            self._config_request_fingerprints[request_id] = fingerprint
            if len(self._config_request_results) > 256:
                oldest = next(iter(self._config_request_results))
                self._config_request_results.pop(oldest, None)
                self._config_request_fingerprints.pop(oldest, None)
            if self._config_apply_task is None or self._config_apply_task.done():
                self._config_apply_task = asyncio.create_task(self._apply_configuration_loop())
                self._config_apply_task.add_done_callback(_consume_task_result)
        await self._reconcile_schedule_admission()
        await self._emit_configuration_event()
        return response

    async def persist_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Persist the legacy chat-effort control through the global config revision."""
        async with self._config_lock:
            was_pending = self._configuration_transition_active or self._config_status in {
                "pending",
                "failed-to-apply",
            }
            self._config_loader.update_reasoning_effort(effort)
            snapshot = self._config_loader.editable_snapshot()
            saved = snapshot.configuration
            if self.configuration is not None:
                active_routes = {
                    name: replace(route, reasoning_effort=effort)
                    if name in {"default", "chat"}
                    else route
                    for name, route in self.configuration.models.routes.items()
                }
                effort_only_configuration = replace(
                    self.configuration,
                    models=replace(
                        self.configuration.models, routes=MappingProxyType(active_routes)
                    ),
                )
                was_pending = was_pending or saved != effort_only_configuration
            self._config_saved_configuration = saved
            self._config_saved_revision = snapshot.revision
            self._config_fields = {
                section: dict(values) for section, values in snapshot.fields.items()
            }
            try:
                snapshot.require_valid_candidate()
            except ConfigError:
                candidate_valid = False
            else:
                candidate_valid = True
            if not candidate_valid:
                self._config_status = "failed-to-apply"
                self._config_pending_revision = snapshot.revision
                self._config_error = {
                    "code": "config_invalid",
                    "message": "The saved User Configuration contains invalid fields.",
                }
                self._configuration_transition_active = False
            elif not was_pending:
                self._config_status = "active"
                self._config_pending_revision = None
                self._configuration_transition_active = False
            else:
                self._config_status = "pending"
                self._config_pending_revision = self._config_saved_revision
                self._config_waiting_for = tuple(self._workspaces)
                self._configuration_transition_active = True
                if self._config_apply_task is None or self._config_apply_task.done():
                    self._config_apply_task = asyncio.create_task(self._apply_configuration_loop())
                    self._config_apply_task.add_done_callback(_consume_task_result)
            if self._config_status == "active":
                self._config_active_revision = self._config_saved_revision
        await self._reconcile_schedule_admission()
        await self._emit_configuration_event()

    async def _activate_initial_configuration(
        self, target: UserConfiguration, target_revision: str
    ) -> bool:
        """Build catalog Workspaces before publishing the first usable generation."""
        async with self._project_lifecycle_lock:
            return await self._prepare_initial_configuration(target, target_revision)

    async def _discard_initial_configuration_candidates(self) -> None:
        """Keep failed cleanup owned so retry and service stop can finish it."""
        candidates = tuple(self._initial_configuration_candidates)
        results = await asyncio.gather(
            *(runtime.close() for runtime in candidates), return_exceptions=True
        )
        errors: list[BaseException] = []
        for runtime, result in zip(candidates, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(result)
            else:
                self._initial_configuration_candidates.remove(runtime)
        if errors:
            raise service_error(
                "runtime_generation_cleanup_failed",
                "Prepared Workspace resources could not be stopped; retry the configuration.",
                status=500,
                retryable=True,
            ) from BaseExceptionGroup("Initial configuration cleanup failed", errors)

    async def _prepare_initial_configuration(
        self, target: UserConfiguration, target_revision: str
    ) -> bool:
        await self._discard_initial_configuration_candidates()
        async with self._config_lock:
            self.config_view()
            if target_revision != self._config_saved_revision:
                return False
        created: list[tuple[str, WorkspaceServiceRuntime]] = []
        try:
            seen: set[str] = set()
            for record in self.projects.list():
                if (
                    record.schedule_state not in {"available", "awaiting_resume"}
                    or not record.path.is_dir()
                ):
                    continue
                key = os.path.normcase(str(record.path.resolve(strict=True)))
                if key in seen or key in self._workspace_keys:
                    continue
                seen.add(key)
                runtime = WorkspaceServiceRuntime(
                    self, record.path.resolve(strict=True), target, workspace_id=str(uuid4())
                )
                self._initial_configuration_candidates.append(runtime)
                created.append((key, runtime))
                await runtime.start()
        except BaseException:
            await self._discard_initial_configuration_candidates()
            raise

        async with self._config_lock:
            self.config_view()
            if (
                self.state in {"draining", "stopped"}
                or target_revision != self._config_saved_revision
                or self._workspaces
            ):
                stale = True
            else:
                stale = False
                for key, runtime in created:
                    self._workspace_keys[key] = runtime.workspace_id
                    self._workspaces[runtime.workspace_id] = runtime
                    self._initial_configuration_candidates.remove(runtime)
                self.configuration = target
                for client in self._clients.values():
                    client.permission_control.reconfigure(target.runtime.permission_level)
                self._config_active_revision = target_revision
                self._config_status = "active"
                self._config_pending_revision = None
                self._config_error = None
                self._config_waiting_for = ()
        if stale:
            await self._discard_initial_configuration_candidates()
            return False
        return True

    async def _apply_configuration_loop(self) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                async with self._config_lock:
                    target = self._config_saved_configuration
                    target_revision = self._config_saved_revision
                    active_revision = self._config_active_revision
                if target is None or target_revision is None or target_revision == active_revision:
                    async with self._config_lock:
                        if target is not None and self.configuration is not None:
                            self._config_status = "active"
                        self._config_pending_revision = None
                        self._configuration_transition_active = False
                        self._config_waiting_for = ()
                    await self._reconcile_schedule_admission()
                    await self._emit_configuration_event()
                    if self._config_status == "pending":
                        continue
                    return

                workspaces = tuple(self._workspaces.values())
                if self.configuration is None and not workspaces:
                    try:
                        activated = await self._activate_initial_configuration(
                            target, target_revision
                        )
                    except BaseException as error:
                        if isinstance(error, asyncio.CancelledError):
                            raise
                        async with self._config_lock:
                            if target_revision != self._config_saved_revision:
                                continue
                            self._config_status = "failed-to-apply"
                            self._config_pending_revision = target_revision
                            self._config_waiting_for = ()
                            self._config_error = {
                                "code": "runtime_generation_failed",
                                "message": "The saved configuration could not be applied.",
                            }
                            self._configuration_transition_active = False
                        await self._reconcile_schedule_admission()
                        await self._emit_configuration_event()
                        return
                    if not activated:
                        continue
                    async with self._config_lock:
                        if target_revision != self._config_saved_revision:
                            continue
                        self._configuration_transition_active = False
                    await self._reconcile_schedule_admission()
                    await self._emit_configuration_event()
                    if self._config_status == "pending":
                        continue
                    return
                prepared: list[tuple[WorkspaceServiceRuntime, _PreparedWorkspaceGeneration]] = []
                committed = False
                try:
                    snapshot = self._config_loader.editable_snapshot()
                    if snapshot.revision != target_revision:
                        self.config_view()
                        continue
                    snapshot.require_valid_candidate()
                    for workspace in workspaces:
                        prepared.append((workspace, await workspace.prepare_configuration(target)))
                    async with self._config_lock:
                        self.config_view()
                        if (
                            target_revision != self._config_saved_revision
                            or workspaces != tuple(self._workspaces.values())
                            or not all(
                                workspace.prepared_configuration_is_current(candidate)
                                for workspace, candidate in prepared
                            )
                        ):
                            latest = True
                        else:
                            latest = False
                            if self.state in {"draining", "stopped"}:
                                raise asyncio.CancelledError
                            for _workspace, candidate in prepared:
                                candidate.schedule_jobs = (
                                    candidate.runtime.schedule_service.prepare_generation_state(
                                        candidate.previous.schedule_service,
                                        schedule=candidate.schedule,
                                    )
                                )
                            WorkspaceRuntime.publish_replacements(
                                tuple(
                                    (candidate.previous, candidate.runtime)
                                    for _, candidate in prepared
                                )
                            )
                            retired = [
                                (workspace, workspace.activate_configuration(candidate))
                                for workspace, candidate in prepared
                            ]
                            committed = True
                            self.configuration = target
                            for client in self._clients.values():
                                client.permission_control.reconfigure(
                                    target.runtime.permission_level
                                )
                            self._config_active_revision = target_revision
                    if latest:
                        for workspace, candidate in prepared:
                            await workspace.discard_prepared_configuration(candidate)
                        continue
                    # Cleanup cannot roll back the already published generation.
                    for workspace, generation in retired:
                        try:
                            await workspace.retire_configuration(generation)
                        except Exception:
                            # Retain ownership for a later cleanup attempt and service stop.
                            pass
                    async with self._config_lock:
                        if target_revision != self._config_saved_revision:
                            continue
                        self._configuration_transition_active = False
                        self._config_status = "active"
                        self._config_pending_revision = None
                        self._config_error = None
                        self._config_waiting_for = ()
                    await self._reconcile_schedule_admission()
                    await self._emit_configuration_event()
                    if self._config_status == "pending":
                        continue
                    return
                except BaseException as error:
                    if not committed:
                        await asyncio.gather(
                            *(
                                workspace.discard_prepared_configuration(candidate)
                                for workspace, candidate in prepared
                            ),
                            return_exceptions=True,
                        )
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    async with self._config_lock:
                        if (
                            target_revision != self._config_saved_revision
                            or workspaces != tuple(self._workspaces.values())
                            or any(workspace._closed for workspace in workspaces)
                        ):
                            continue
                        self._config_status = "failed-to-apply"
                        self._config_pending_revision = target_revision
                        self._config_waiting_for = tuple(self._workspaces)
                        self._config_error = {
                            "code": "runtime_generation_failed",
                            "message": "The saved configuration could not be applied.",
                        }
                        self._configuration_transition_active = False
                    await self._reconcile_schedule_admission()
                    await self._emit_configuration_event()
                    if self._config_status == "pending":
                        continue
                    return
        finally:
            if self._config_apply_task is current_task:
                self._config_apply_task = None

    async def _emit_configuration_event(self) -> None:
        if self._config_saved_revision is None:
            return
        application = cast(dict[str, object], self._config_response()["application"])
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
        permission = (
            self.configuration.runtime.permission_level if self.configuration else "workspace-write"
        )
        client = ClientState(str(uuid4()), kind, str(uuid4()), RuntimePermissionControl(permission))
        if kind == "web":
            client.web_control_credential = str(uuid4())
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
                self._global_reconnect_task = asyncio.create_task(
                    self._stop_after_grace(expiry_deadline)
                )
        else:
            await self._reconcile_schedule_admission()

    async def attach_workspace(self, client_id: str, path: Path) -> WorkspaceServiceRuntime:
        async with self._project_lifecycle_lock:
            return await self._attach_workspace(client_id, path)

    async def _attach_workspace(self, client_id: str, path: Path) -> WorkspaceServiceRuntime:
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

    async def _get_or_create_workspace(self, path: Path) -> WorkspaceServiceRuntime:
        key = os.path.normcase(str(path.resolve(strict=True)))
        async with self._lock:
            workspace_id = self._workspace_keys.get(key)
            if workspace_id is not None:
                return self._workspaces[workspace_id]
            if self.configuration is None or self._config_active_revision is None:
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

    def _schedule_admission_open(self) -> bool:
        return (
            self.state == "ready"
            and self.configuration_ready
            and not self._configuration_transition_active
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
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime, tuple[ScheduleJob, ...]]:
        async with self._project_lifecycle_lock:
            self._require_client(client_id)
            if self.configuration is None or self._config_active_revision is None:
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

    def _schedule_workspace(self, client_id: str, workspace_id: str) -> WorkspaceServiceRuntime:
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
            jobs = await workspace.schedule_service.public_snapshot()
            job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)
            removal_key = (workspace_id, job_id)
            pending = self._schedule_removal_jobs.get(removal_key)
            if job is None and pending is not None:
                job = pending[0]
            if job is None:
                raise service_error("not_found", "Schedule Job was not found.", status=404)
            was_active = (
                pending[1]
                if pending is not None
                else workspace.schedule_service.is_job_active(job_id)
            )
            self._schedule_removal_jobs[removal_key] = (job, was_active)
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
            except (ScheduleStateError, OSError, RuntimeError) as error:
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

    async def _project_workspace(
        self, client_id: str, project_id: str
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime]:
        async with self._project_lifecycle_lock:
            return await self._project_workspace_owned(client_id, project_id)

    async def _project_workspace_owned(
        self, client_id: str, project_id: str
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime]:
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

    async def list_project_sessions(
        self, client_id: str, project_id: str
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime, list[dict[str, object]]]:
        async with self._project_lifecycle_lock:
            record, workspace = await self._project_workspace_owned(client_id, project_id)
            return record, workspace, await workspace.list_sessions(client_id)

    async def list_project_sessions_page(
        self,
        client_id: str,
        project_id: str,
        *,
        title: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[ProjectRecord, WorkspaceServiceRuntime, dict[str, object]]:
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
                    "project_id": project_id,
                    "workspace_id": workspace.workspace_id,
                    "session_id": session_id,
                    "state": state,
                }

    async def claim_project_session_deletion(
        self, client_id: str, project_id: str, session_id: str
    ) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
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
                    "project_id": project_id,
                    "workspace_id": workspace.workspace_id,
                    "session_id": session_id,
                    "claim": {
                        "workspace_id": workspace.workspace_id,
                        "session_id": session_id,
                        "claim_version": version,
                        "reconnect_credential": credential,
                    },
                }

    async def create_project_session(self, client_id: str, project_id: str) -> dict[str, object]:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            session_id = await workspace.create_draft(client_id, reuse_startup_session=False)
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
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
            claim = workspace.require_claim(client_id, session_id, claim_version, claim_credential)
            workspace._ensure_session_available(session_id)
            return {
                "project_id": project_id,
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": claim.version,
                "snapshot": workspace.session_snapshot(session_id),
            }

    async def release_project_session(
        self,
        client_id: str,
        project_id: str,
        session_id: str,
        claim_version: int,
        claim_credential: str,
    ) -> None:
        async with self._project_lifecycle_lock:
            _record, workspace = await self._project_workspace_owned(client_id, project_id)
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
            task.add_done_callback(self._project_removal_finished)
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

    async def remove_project(self, client_id: str, project_id: str) -> Path:
        """Start a removal and wait for its terminal outcome for legacy callers."""
        await self.start_project_removal(client_id, project_id)
        operation = self._project_removals[project_id]
        task = operation.task
        if task is not None:
            await asyncio.shield(task)
        if operation.status != "completed":
            raise service_error(
                "project_removal_failed",
                "Project work could not be stopped; the registration remains blocked.",
                status=500,
                retryable=True,
            )
        return operation.path

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

    def _project_removal_finished(self, task: asyncio.Task[None]) -> None:
        _consume_task_result(task)

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
                await workspace.close()
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

    def workspace(self, workspace_id: str) -> WorkspaceServiceRuntime:
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
        previous_workspace_id = client.current_workspace_id
        previous_session_id = client.current_session_id
        already_claimed = (workspace_id, session_id) in client.claimed
        claim = await workspace.claim(client_id, session_id)
        client.claimed.add((workspace_id, session_id))
        client.current_workspace_id = workspace_id
        client.current_session_id = session_id
        if not already_claimed:
            await self.emit(
                "session.claimed",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=None,
                payload={"occupied": True},
                target_client_ids=self.workspace_audience(workspace_id),
            )
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
            "snapshot": workspace.session_snapshot(session_id),
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
        return await workspace.list_sessions_page(
            client_id,
            title=title,
            cursor=cursor,
            limit=limit,
        )

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
            "restore/acknowledge-failure",
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
        dispatcher = workspace.management_dispatcher(
            client_id,
            session_id,
            claim_version if requires_claim else None,
            claim_credential if requires_claim else None,
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
        elif action in {"restore/acknowledge", "restore/acknowledge-failure"}:
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

    def confirmation_source(
        self, owner: ConfirmationOwner
    ) -> tuple[str | None, str | None, str | None]:
        for workspace in self._workspaces.values():
            result = workspace.confirmation_source(owner)
            if result[0] is not None:
                return result
        return None, None, None

    async def stop(self) -> None:
        if (
            self._stop_task is not None
            and self._stop_task.done()
            and self._stop_failed
            and self._initial_configuration_candidates
        ):
            self._stop_task = None
        if self._stop_task is not None:
            await self._stop_task
            return
        self._stop_task = asyncio.create_task(self._stop_owned())
        await self._stop_task

    async def _stop_owned(self) -> None:
        if self.state == "stopped" and not self._initial_configuration_candidates:
            return
        self.state = "draining"
        errors: list[Exception] = []
        config_task = self._config_apply_task
        if config_task is not None and config_task is not asyncio.current_task():
            config_task.cancel()
            await asyncio.gather(config_task, return_exceptions=True)
        try:
            await self._discard_initial_configuration_candidates()
        except Exception as error:
            errors.append(error)
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
        for workspace in self._workspaces.values():
            try:
                await workspace.expire_client(client_id)
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

    async def _stop_after_grace(self, deadline: float | None = None) -> None:
        try:
            if deadline is None:
                deadline = self._monotonic() + self.reconnect_timeout
            await self._wait_until(deadline)
        except asyncio.CancelledError:
            return
        if not any(client.connected for client in self._clients.values()):
            self._global_reconnect_task = None
            pending_expiry = tuple(
                client.disconnect_task
                for client in self._clients.values()
                if client.disconnect_task is not None
                and client.disconnect_task is not asyncio.current_task()
            )
            if pending_expiry:
                await asyncio.gather(*pending_expiry, return_exceptions=True)
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

    async def _send_snapshot_required(self, client: ClientState, *, reason: str) -> None:
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
        client.sequence += 1
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
                "snapshot": {"sessions": snapshots},
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


def _encode_management_result(result: object) -> dict[str, object]:
    from myclaw.management.commands import ManagementCommandResult

    if not isinstance(result, ManagementCommandResult):
        raise service_error("service_protocol_error", "Management result is invalid.", status=500)
    encoded: dict[str, object] = {
        "handled": result.handled,
        "output": result.output,
        "memory_content": result.memory_content,
        "effort_selection": result.effort_selection,
        "permission_selection": result.permission_selection,
        "published_effort": result.published_effort,
        "published_permission_level": result.published_permission_level,
        "resumed_session_id": result.resumed_session_id,
        "resume_skipped_count": result.resume_skipped_count,
    }
    if result.dream_result is not None:
        encoded["dream_result"] = _safe_wire_value(result.dream_result)
    if result.management_error is not None:
        encoded["management_error"] = _safe_wire_value(result.management_error)
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
