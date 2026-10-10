"""Shared runtime records and coordination values."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from aide.agent.message_bus import MessageBus
from aide.agent.permission import RuntimePermissionControl
from aide.service.contracts import (
    ConversationOpenDTO,
)
from aide.service.execution import SessionExecution


class ServiceSink(Protocol):
    async def send_event(self, event: dict[str, object]) -> None: ...


_CONFIG_INVALID_ERROR = {
    "code": "config_invalid",
    "message": "The saved User Configuration contains invalid fields.",
}


_MISSING = object()


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
    ever_connected: bool = False
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
    conversation_open_results: dict[str, tuple[tuple[object, ...], ConversationOpenDTO]] = field(
        default_factory=dict
    )
    conversation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    claimed: set[tuple[str, str]] = field(default_factory=set)
    attached_workspaces: set[str] = field(default_factory=set)
    current_workspace_id: str | None = None
    current_session_id: str | None = None
    current_project_id: str | None = None
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
