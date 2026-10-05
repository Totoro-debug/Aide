"""Single-process local service authority for Workspace and Session execution."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from uuid import uuid4

from tzlocal import get_localzone_name

from omni.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationOwner,
    ConfirmationPresenter,
    ForegroundConfirmationOwner,
    ToolConfirmationCoordinator,
)
from omni.agent.context.builder import ContextBuilder
from omni.agent.loop import (
    AgentRunExecutor,
    ForegroundConversationProjection,
    session_runtime_status_input,
)
from omni.agent.memory.dream import Dream
from omni.agent.memory.manager import MemoryManager
from omni.agent.message_bus import InboundMessage, MessageBus
from omni.agent.permission import PermissionSnapshot, RuntimePermissionControl
from omni.agent.session.deletion import (
    begin_session_deletion,
    delete_session_data,
    recover_session_deletions,
    session_deletion_pending,
    session_deletion_status,
    session_restore_pending,
)
from omni.agent.session.restore import (
    RestoreManager,
    RestoreRecoveryRequired,
    RestoreResult,
    StaleRestorePlan,
)
from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.tools.core.exec_host import ExecHost, create_exec_host, resolve_exec_shell
from omni.agent.tools.mcp_keywords import MCPKeywordPreparer
from omni.agent.tools.mcp_runtime import MCPRuntimeManager, MCPWorkspaceRuntimeManager
from omni.agent.tools.tool_gateway import (
    BUILT_IN_TOOL_NAMES,
    BuiltInToolCatalog,
)
from omni.agent.workspace_state import WorkspaceState, WorkspaceStateError
from omni.config.agent_home import AgentHome
from omni.config.config import (
    ConfigError,
    ConfigFieldError,
    ConfigLoader,
    ConfigRevisionConflict,
    ReasoningEffort,
    UserConfiguration,
)
from omni.errors import ErrorInfo
from omni.management.commands import MANAGEMENT_COMMANDS
from omni.provider.factory import create_provider
from omni.provider.model_router import ModelRouter
from omni.provider.models import REASONING_EFFORT_LEVELS, SessionModelConfiguration
from omni.schedule.history import (
    ScheduleHistoryPersistenceError,
    ScheduleHistoryRequestError,
    read_schedule_history,
)
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.schedule.service import (
    ScheduleDispatcher,
    ScheduleOccurrence,
    ScheduleService,
    ScheduleStaleRemovalError,
)
from omni.schedule.store import (
    ScheduleStateError,
    ScheduleStoreFaultedError,
    WorkspaceScheduleStore,
)
from omni.service.conversation_workspaces import (
    ConversationWorkspaceCatalog,
    ConversationWorkspaceCatalogError,
)
from omni.service.errors import ServiceError, service_error
from omni.service.execution import SessionExecution
from omni.service.projects import ProjectCatalog, ProjectCatalogError, ProjectRecord
from omni.service.resources import WorkspaceResourceManager, WorkspaceResources
from omni.skills.catalog import LoadedSkill, SkillLoader, SkillMetadata
from omni.utils.host_filesystem import HOST_FILESYSTEM
from omni.utils.scheduler import AsyncioSchedulerClock
from omni.utils.text import normalize_title_candidate
from omni.utils.time import local_now


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


def _encode_chat_session_cursor(key: tuple[datetime, datetime, str, str], title_filter: str) -> str:
    payload = json.dumps(
        {
            "title_filter": title_filter,
            "updated_at": key[0].isoformat(),
            "created_at": key[1].isoformat(),
            "id": key[2],
            "directory": key[3],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_chat_session_cursor(
    value: str, title_filter: str
) -> tuple[datetime, datetime, str, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        )
        if payload["title_filter"] != title_filter:
            raise ValueError("cursor scope does not match")
        updated_at = datetime.fromisoformat(payload["updated_at"])
        created_at = datetime.fromisoformat(payload["created_at"])
        session_id = payload["id"]
        directory = payload["directory"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise service_error("validation_error", "cursor is invalid.", status=422) from error
    if (
        updated_at.tzinfo is None
        or created_at.tzinfo is None
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(directory, str)
        or not Path(directory).is_absolute()
    ):
        raise service_error("validation_error", "cursor is invalid.", status=422)
    return updated_at, created_at, session_id, directory


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
    loop: SessionExecution
    status: str = "claimed"
    disconnected_at: float | None = None


@dataclass(slots=True)
class _LoopState:
    loop: SessionExecution
    bus: MessageBus
    owner_client_id: str | None
    run_ids: deque[str] = field(default_factory=deque)
    live_runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_user_count: int | None = None
    output_task: asyncio.Task[bool] | None = None
    processor_task: asyncio.Task[None] | None = None
    coordination_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    processor_stopping: bool = False
    release_task: asyncio.Task[None] | None = None
    schedule: bool = False


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

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._wire_tokens: dict[object, str] = {}
        self._wire_sources: dict[object, tuple[str | None, str | None, str | None]] = {}
        self._wire_requests: dict[object, asyncio.Task[None]] = {}
        self._wire_payloads: dict[object, dict[str, object]] = {}

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
        self._wire_payloads[token] = payload
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
        self._wire_payloads.pop(token, None)
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
                    self._wire_payloads.pop(token, None)
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

    def snapshot(self, client_id: str) -> dict[str, object] | None:
        """Project the current display slot only for an authorized Client."""
        token = next((candidate for candidate in self._wire_payloads
                      if self._service.confirmation.is_pending(candidate)), None)
        if token is None:
            return None
        payload = self._wire_payloads[token]
        workspace_id, session_id, run_id = self._wire_sources[token]
        if client_id not in self._audience(workspace_id, session_id):
            return None
        workspace = self._service._workspaces.get(workspace_id or "")
        project_id = None
        if workspace is not None:
            key = os.path.normcase(str(workspace.workspace_path))
            project_id = next((record.project_id for record in self._service.projects.list()
                               if os.path.normcase(str(record.path.resolve(strict=False))) == key), None)
        return {"workspace_id": workspace_id, "project_id": project_id,
                "session_id": session_id, "run_id": run_id, "payload": deepcopy(payload)}


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
        creation_scope: str | None = None,
    ) -> str:
        if self._closed:
            raise service_error("admission_closed", "Workspace admission is closed.")
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
        if is_draft and creation_scope is not None:
            loop_state.loop.session.update_metadata(creation_scope=creation_scope)
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
            try:
                if loop_state is not None:
                    await self.service.confirmation.cancel_generation(loop_state.loop.generation_id)
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
        legacy_creation_scope: str | None = None,
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
                session_scope = session.metadata.get("creation_scope", legacy_creation_scope)
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

            try:
                begin_session_deletion(self.workspace_state, session_id)
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
        from omni.management.commands import ManagementCommandDispatcher
        from omni.management.service import (
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
            await self.service.claim(client_id, self.workspace_id, target_session_id)

        async def restore_listing() -> RestoreListingReport:
            from omni.management.service import RestoreListingReport

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
            from omni.agent.session.restore import RestoreManager

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
            from omni.agent.session.restore import RestoreManager

            session_id = current_loop().session.session_id
            result = RestoreManager(
                self.workspace_state, session_id, now=local_now
            ).completed_result()
            return result

        async def restore_acknowledge_failure() -> Any:
            from omni.agent.session.restore import RestoreManager

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
            selection = claim.loop.session.model_configuration
            if selection is not None:
                try:
                    self.configuration.resolve_session_model_route(
                        selection.provider_id, selection.model, selection.reasoning_effort
                    )
                except (ConfigError, ValueError) as error:
                    raise service_error(
                        "model_unavailable",
                        "Choose an available model before sending to this Session.",
                        status=422,
                        field_errors={"model": "choose an available model"},
                    ) from error
            state = self._loops[session_id]
            client = self.service.client(client_id)
            async with state.coordination_lock:
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
                    await state.bus.put_inbound(
                        InboundMessage(
                            content=text,
                            metadata={"run_id": run_id, "request_id": request_id or run_id},
                        )
                    )
                self._ensure_processor(state)
            return claim

    def _ensure_processor(self, state: _LoopState) -> None:
        """Start one Session processor while preserving the enqueue boundary."""
        if state.processor_stopping:
            raise service_error(
                "admission_closed", "Conversation input is temporarily unavailable."
            )
        processor = state.processor_task
        if processor is not None and not processor.done():
            return
        processor = asyncio.create_task(self._process_session(state))
        state.processor_task = processor

        def processor_finished(completed: asyncio.Task[None]) -> None:
            self._processor_finished(state, completed)

        processor.add_done_callback(processor_finished)

    def _processor_finished(self, state: _LoopState, task: asyncio.Task[None]) -> None:
        if state.processor_task is task:
            state.processor_task = None
        _consume_task_result(task)

    async def _process_session(self, state: _LoopState) -> None:
        """Run accepted foreground inputs serially until the Session queue is empty."""
        current = asyncio.current_task()
        if current is None:
            return
        release_when_idle = False
        while True:
            async with state.coordination_lock:
                if state.processor_stopping:
                    if state.processor_task is current:
                        state.processor_task = None
                    return
                if not await state.bus.inbound_snapshot():
                    if state.processor_task is current:
                        state.processor_task = None
                    release_when_idle = not self._closed and state.owner_client_id is not None
                    break
                inbound = await state.bus.get_inbound()
                run_id = inbound.metadata.get("run_id")
                if not isinstance(run_id, str):
                    run_id = state.run_ids[0] if state.run_ids else None

            output_task = asyncio.create_task(self._forward_output(state, run_id))
            state.output_task = output_task
            terminal_forwarded = False
            execution_failed = False
            try:
                try:
                    await state.loop.run_foreground(inbound)
                except asyncio.CancelledError:
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
        if not claim.loop.has_active_run:
            return
        state = self._loops[session_id]
        if not state.run_ids or state.run_ids[0] != run_id:
            raise service_error("stale_run", "The requested Agent Run is no longer active.")
        client = self.service.client(client_id)
        async with client.delivery_lock:
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
        state = _LoopState(loop=loop, bus=bus, owner_client_id=client_id)
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
        loop_kwargs: dict[str, Any] = {
            "workspace_path": self.workspace_path,
            "workspace_state": self.workspace_state,
            "agent_home": self.service.agent_home,
            "configuration": configuration,
            "bus": selected_bus,
            "schedule_service": schedule_service,
            "model_router": runtime.router,
            "memory_manager": runtime.memory_manager,
            "now": local_now,
            "new_uuid": uuid4,
            "monotonic_now": monotonic,
            "mcp_tools": runtime.mcp_snapshot,
            "mcp_keywords": runtime.mcp_keywords,
            "exec_host": exec_host,
            "permission_control": permission_control,
            "configured_schedule_level": configuration.runtime.permission_level,
            "skill_loader": self.service.skill_loader,
            "built_in_catalog": self.service.built_in_tool_catalog,
        }
        authority = session
        if authority is None:
            authority = (
                Session.create(self.workspace_state, now=local_now, new_uuid=uuid4)
                if session_id is None
                else Session.load(self.workspace_state, session_id, now=local_now)
            )

        def create_executor() -> AgentRunExecutor:
            state = self._loops.get(authority.session_id)
            owner = None if state is None else state.owner_client_id
            loop_kwargs["permission_control"] = (
                self._schedule_permission if owner is None else self.service.client_permission(owner)
            )
            executor = AgentRunExecutor(session=authority, session_id=None, **loop_kwargs)
            executor.bind_confirmation_requester(self.service.confirmation.request)
            return executor

        prepared = create_executor()
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

        handle = SessionExecution(authority, selected_bus, create_executor, self.service.reload_skills, status_input)
        return handle, selected_bus

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

    async def close(self) -> None:
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
        self._stop_failed = False
        self._closed = asyncio.Event()
        self._start_lock = asyncio.Lock()
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
        self._config_fields: dict[str, dict[str, object]] | None = None
        self._config_status = "active" if configuration is not None else "pending-repair"
        self._config_state = "active" if configuration is not None else "missing"
        self._config_repair_required = configuration is None
        self._config_backup_required = False
        self._config_requires_secret_reentry = configuration is None
        self._config_projection_error: dict[str, str] | None = None
        self._chat_effort_override: ReasoningEffort | None = None
        self.projects = ProjectCatalog(agent_home)
        self.conversation_workspaces = ConversationWorkspaceCatalog(agent_home)

    async def _activate_workspace(self, workspace: WorkspaceRecord) -> None:
        async with workspace._lock:
            if workspace._closed:
                raise RuntimeError("Workspace service runtime is closed")
            if workspace._started:
                return
            workspace._exec_host = self.exec_host

            async def execute_user_job(job: ScheduleJob) -> None:
                # Admission was checked at reservation; accepted occurrences must drain.
                loop = await workspace._get_schedule_loop(job.job_id, title=job.title)
                await loop.loop.run_schedule_job(job)

            async def execute_user_occurrence(occurrence: ScheduleOccurrence) -> None:
                loop = await workspace._get_schedule_loop(occurrence.job.job_id, title=occurrence.job.title)
                await loop.loop.run_schedule_job(occurrence.job, occurrence)

            registered_resources = False
            try:
                state = WorkspaceState(workspace.workspace_path)
                state.initialize(
                    agent_home_root=self.agent_home.path,
                    allow_agent_home_chat=workspace.allow_agent_home_chat,
                )
                recover_session_deletions(state)
                restore_manager = RestoreManager(state)
                workspace._restore_result = await restore_manager.recover_pending()
                workspace.workspace_state = state

                manager = MCPWorkspaceRuntimeManager(
                    workspace.workspace_path,
                    shared_runtime=self.mcp_manager,
                    built_in_names=BUILT_IN_TOOL_NAMES,
                )
                workspace._mcp_manager = manager
                workspace._mcp_startup_report = await manager.start(workspace.configuration.mcp)
                workspace._mcp_snapshot = workspace._mcp_startup_report.snapshot
                workspace._router = self.model_router
                workspace._mcp_keyword_preparer = MCPKeywordPreparer(
                    model_router=workspace._router,
                    config_loader=ConfigLoader(self.agent_home),
                )
                workspace._mcp_keywords = await workspace._mcp_keyword_preparer.prepare(
                    workspace._mcp_snapshot,
                    workspace.configuration.mcp,
                )
                memory_manager = MemoryManager(state)
                dream = Dream(
                    memory_manager=memory_manager,
                    model_router=workspace._router,
                    batch_size=workspace.configuration.memory.batch_size,
                    memory_route_status=workspace._router.route_status("memory"),
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
                workspace._started = True
            except BaseException as error:
                cleanup_targets: list[Awaitable[None]] = []
                if registered_resources:
                    cleanup_targets.append(self._close_workspace_resources(workspace))
                else:
                    for resource in (workspace._schedule_service, workspace._dream, workspace._mcp_manager):
                        if resource is not None:
                            cleanup_targets.append(resource.close())
                cleanup_results = await asyncio.gather(*cleanup_targets, return_exceptions=True)
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

    async def start(self) -> None:
        async with self._start_lock:
            await self._start_owned()

    async def _start_owned(self) -> None:
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
            self._initialize_shared_resources(snapshot.configuration)
            assert self._mcp_manager is not None
            await self._mcp_manager.start(snapshot.configuration.mcp)
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
        self._started_at = monotonic()
        self.state = "ready"
        await self._reconcile_schedule_admission()
        self._global_reconnect_task = asyncio.create_task(self._stop_after_grace())

    @property
    def workspaces(self) -> Mapping[str, WorkspaceRecord]:
        return self._workspaces

    @property
    def workspace_resources(self) -> WorkspaceResourceManager:
        return self._workspace_resources

    @property
    def schedule_dispatcher(self) -> ScheduleDispatcher:
        return self._workspace_resources.dispatcher

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
        """Publish the service-owned Skill and Model resources once per lifetime."""
        if self._skill_loader is None:
            skill_loader = SkillLoader(
                root=self.agent_home.skills_directory,
                reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
                enable_always_load=configuration.runtime.enable_skill_always_load,
            )
            skill_loader.load()
            self._skill_loader = skill_loader
        if self._model_router is None:
            self._model_router = ModelRouter(
                configuration=configuration,
                provider_factory=create_provider,
            )
        if self._exec_host is None:
            self._exec_host = create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell))
        if self._built_in_tool_catalog is None:
            self._built_in_tool_catalog = BuiltInToolCatalog(
                skill_root=self.skill_loader.root,
                exec_host=self._exec_host,
            )
        if self._mcp_manager is None:
            self._mcp_manager = MCPRuntimeManager(
                None,
                built_in_names=BUILT_IN_TOOL_NAMES,
            )

    def reload_skills(self) -> tuple[SkillMetadata, ...]:
        """Validate every loaded Session before publishing one global Skill snapshot."""

        def validate(skills: tuple[LoadedSkill, ...]) -> None:
            for workspace in self._workspaces.values():
                for state in workspace.loops.values():
                    state.loop._validate_model_context_budget(skills)

        self.skill_loader.load(validate=validate)
        return self.skill_loader.metadata

    @property
    def configuration_ready(self) -> bool:
        """Return whether new Agent and Schedule work may be admitted."""
        return (
            self.configuration is not None
            and self._config_active_revision is not None
        )

    def config_view(self) -> dict[str, object]:
        """Read saved settings without changing this service's startup configuration."""
        snapshot = self._config_loader.web_snapshot()
        self._config_saved_revision = snapshot.revision
        self._config_saved_configuration = (
            snapshot.configuration if snapshot.state == "active" else None
        )
        self._config_state = snapshot.state
        self._config_repair_required = snapshot.repair_required
        self._config_backup_required = snapshot.backup_required
        self._config_requires_secret_reentry = snapshot.requires_secret_reentry
        self._config_projection_error = None if snapshot.error is None else dict(snapshot.error)
        self._config_fields = {section: dict(values) for section, values in snapshot.fields.items()}
        self._config_status = (
            "pending-repair"
            if snapshot.repair_required
            else "active"
            if snapshot.revision == self._config_active_revision
            else "restart-required"
        )
        return self._config_response()

    def available_models_view(self) -> dict[str, object]:
        """Return active provider models with capacities and the effective chat route."""
        configuration = self.configuration
        if configuration is None:
            return {"models": [], "default_combination": None}

        capacities = configuration.effective_model_context_windows()
        try:
            default = configuration.resolve_route("chat")
        except ConfigError:
            minimum_capacity = None
            default_combination = None
        else:
            minimum_capacity = default.route.max_output
            default_combination = {
                "provider_id": default.provider.provider_id,
                "model": default.route.model,
                "reasoning_effort": self.reasoning_effort,
            }
        models = [
            {
                "provider_id": provider_id,
                "model": model,
                "context_window": capacities[provider_id][model],
            }
            for provider_id, provider in configuration.models.providers.items()
            if provider.is_usable
            for model in provider.models
            if model in capacities[provider_id]
            and minimum_capacity is not None
            and capacities[provider_id][model] > minimum_capacity
        ]
        return {"models": models, "default_combination": default_combination}

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
                "restart_required": (
                    not self._config_repair_required
                    and saved_revision != self._config_active_revision
                ),
            },
        }

    def configuration_status_text(self) -> str:
        """Render the same save/restart state for Command-line management."""
        application = cast(dict[str, object], self.config_view()["application"])
        restart = "yes" if application["restart_required"] else "no"
        return (
            f"Saved version: {application['saved_revision']}\n"
            f"Startup version: {application['active_revision'] or '-'}\n"
            f"Restart required: {restart}\n"
        )

    async def update_configuration(
        self,
        request_id: str,
        expected_revision: str,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None = None,
        *,
        client_id: str | None = None,
    ) -> dict[str, object]:
        """Persist one safe configuration patch for the next service startup."""
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
            self._config_status = (
                "active" if result.revision == self._config_active_revision else "restart-required"
            )
            response = self._config_response()
            self._config_request_results[request_id] = response
            self._config_request_fingerprints[request_id] = fingerprint
            if len(self._config_request_results) > 256:
                oldest = next(iter(self._config_request_results))
                self._config_request_results.pop(oldest, None)
                self._config_request_fingerprints.pop(oldest, None)
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
        """Persist a first-use or malformed-file repair for the next service startup."""
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
            self._config_status = (
                "active" if result.revision == self._config_active_revision else "restart-required"
            )
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
        await self._emit_configuration_event()
        return response

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
        async with self._config_lock:
            self._config_loader.update_reasoning_effort(effort)
            self.config_view()
        await self._emit_configuration_event()

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

    async def attach_workspace(self, client_id: str, path: Path) -> WorkspaceRecord:
        async with self._project_lifecycle_lock:
            return await self._attach_workspace(client_id, path)

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
            if self.configuration is None or self._config_active_revision is None:
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
                    legacy_creation_scope="project",
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
                    self._config_loader.load().web.default_chat_workspace
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
        creation_scope = (
            "chat"
            if client.kind == "web"
            and self.conversation_workspaces.contains(workspace.workspace_path)
            else None
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
        creation_scope: str | None = None
        legacy_creation_scope: str | None = None
        if client.kind == "web" and self.conversation_workspaces.contains(workspace.workspace_path):
            creation_scope = "chat"
            workspace_key = os.path.normcase(str(workspace.workspace_path.resolve(strict=False)))
            legacy_creation_scope = (
                "project"
                if any(
                    os.path.normcase(str(record.path.resolve(strict=False))) == workspace_key
                    for record in self.projects.list()
                )
                else "chat"
            )
        return await workspace.list_sessions_page(
            client_id,
            title=title,
            cursor=cursor,
            limit=limit,
            creation_scope=creation_scope,
            legacy_creation_scope=legacy_creation_scope,
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
            project_paths = {
                os.path.normcase(str(record.path.resolve(strict=False)))
                for record in self.projects.list()
            }
        except ProjectCatalogError as error:
            raise _project_catalog_service_error(error) from error
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
            directory_identity = os.path.normcase(str(path.resolve(strict=False)))
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
                if "creation_scope" in metadata:
                    if metadata.get("creation_scope") != "chat":
                        continue
                elif directory_identity in project_paths:
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
                client_id, session_id, claim_version, text, run_id, request_id
            )
            result = {"run_id": run_id}
            await self.emit(
                "input.accepted",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload={"text": text, "request_id": request_id},
                target_client_ids=(client_id,),
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
            result = {
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
        try:
            await self._workspace_resources.close()
        except Exception as error:
            errors.append(error)
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.close()
            except Exception as error:
                errors.append(error)
        if self._model_router is not None:
            try:
                await self._model_router.close()
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


def _encode_management_result(result: object) -> dict[str, object]:
    from omni.management.commands import ManagementCommandResult

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
    "AgentService",
    "ClientState",
    "ServiceConfirmationPresenter",
    "ServiceSink",
    "SessionClaim",
    "WorkspaceRecord",
]
