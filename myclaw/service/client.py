"""Python client adapter for the local service protocol."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import secrets
import socket
import subprocess
import sys
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import quote
from uuid import UUID, uuid4

import aiohttp

from myclaw.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationOwner,
    ConfirmationPresentationCoordinator,
    ConfirmationPresenter,
    ForegroundConfirmationOwner,
)
from myclaw.agent.loop import ForegroundConversationProjection, TerminalAgentLoopControl
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.agent.permission import ToolPermissionLevel
from myclaw.agent.session.backup_store import BackupGap, BackupIntegrityIssue
from myclaw.agent.session.restore import (
    RestoreFileResult,
    RestoreFileStatus,
    RestoreMode,
    RestorePlan,
    RestoreResult,
    RestoreTarget,
)
from myclaw.agent.session.session import RestoreAnchor, SessionRestoreResult
from myclaw.agent.tools.tool_gateway import ConfirmationRequest
from myclaw.config.agent_home import AgentHome
from myclaw.service.discovery import (
    DEFAULT_SERVICE_HOST,
    DEFAULT_SERVICE_PORT,
    ServiceDiscovery,
    identity_proof,
    read_credential,
    read_discovery,
)
from myclaw.service.errors import ServiceError


class ServiceStartupError(RuntimeError):
    """A safe failure while locating or starting the local service."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class RemoteMessageBus(MessageBus):
    """MessageBus compatibility surface backed by service events and commands."""

    def __init__(self, client: ServiceClient) -> None:
        super().__init__()
        self.client = client

    async def put_inbound(self, message: InboundMessage) -> None:
        await super().put_inbound(message)
        try:
            await self.client.submit_input(message.content)
        except BaseException:
            pending = await super().drain_inbound()
            for item in pending:
                if item is not message:
                    await super().put_inbound(item)
            raise

    async def accept_one_input(self) -> None:
        messages = await super().drain_inbound()
        for message in messages[1:]:
            await super().put_inbound(message)

    async def put_remote_output(self, value: Mapping[str, object]) -> None:
        message_type = value.get("type")
        if message_type not in {
            "model_reasoning",
            "model_response",
            "tool_call",
            "system_control",
        }:
            message_type = "system_control"
        content = value.get("content")
        metadata = value.get("metadata")
        await super().put_outbound(
            OutboundMessage(
                cast(Any, message_type),
                content if isinstance(content, str) else "",
                dict(metadata) if isinstance(metadata, dict) else {},
            )
        )


class RemoteControl(TerminalAgentLoopControl):
    """Foreground control projection for one claimed remote Session."""

    def __init__(self, client: ServiceClient) -> None:
        self.client = client
        self._run_ids: deque[str] = deque()
        self._completed_before_accept: deque[str] = deque(maxlen=256)
        self._projection = ForegroundConversationProjection("", ())
        self._confirmation_callback: object | None = None
        self._admitted = True

    @property
    def has_active_run(self) -> bool:
        return bool(self._run_ids)

    def foreground_input_admitted(self) -> bool:
        return self._admitted and not self.client.closed

    async def cancel_active_run(self) -> None:
        if not self._run_ids:
            return
        await self.client.cancel_run(self._run_ids[0])

    def bind_confirmation_callback(self, callback: object) -> None:
        self._confirmation_callback = callback

    def unbind_confirmation_callback(self, callback: object) -> None:
        if self._confirmation_callback is callback:
            self._confirmation_callback = None

    def respond_to_confirmation(
        self, confirmation_id: UUID, decision: ConfirmationDecision
    ) -> None:
        del confirmation_id, decision
        raise ValueError("Remote confirmation decisions are owned by the service presenter")

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return self._projection

    def set_projection(self, projection: ForegroundConversationProjection) -> None:
        self._projection = projection

    def accept_run(self, run_id: str) -> None:
        if run_id in self._completed_before_accept:
            self._completed_before_accept.remove(run_id)
            return
        if run_id not in self._run_ids:
            self._run_ids.append(run_id)

    def finish_run(self, run_id: str) -> None:
        try:
            self._run_ids.remove(run_id)
        except ValueError:
            if run_id not in self._completed_before_accept:
                self._completed_before_accept.append(run_id)

    def clear_runs(self) -> None:
        self._run_ids.clear()
        self._completed_before_accept.clear()

    def set_admitted(self, admitted: bool) -> None:
        self._admitted = admitted


class RemoteConfirmationCoordinator(ConfirmationPresentationCoordinator):
    """Present service-owned confirmation requests through the local terminal UI."""

    def __init__(self, client: ServiceClient) -> None:
        self.client = client
        self._presenter: ConfirmationPresenter | None = None
        self._items: dict[object, tuple[str, ConfirmationOwner]] = {}
        self._owners: dict[ConfirmationOwner, object] = {}

    def bind_presenter(self, presenter: ConfirmationPresenter) -> None:
        if self._presenter is not None and self._presenter is not presenter:
            raise RuntimeError("a remote confirmation presenter is already bound")
        self._presenter = presenter

    async def unbind_presenter(self, presenter: ConfirmationPresenter) -> None:
        if self._presenter is not presenter:
            return
        self._presenter = None
        for token in tuple(self._items):
            await self._dismiss_local(token)

    async def cancel_owner(self, owner: ConfirmationOwner) -> None:
        local_token = self._owners.get(owner)
        if local_token is None:
            return
        wire_token, _ = self._items.get(local_token, ("", owner))
        if wire_token:
            with suppress(Exception):
                await self.client.decide_confirmation(wire_token, "declined")
        await self._dismiss_local(local_token)

    async def handle_requested(self, event: Mapping[str, object]) -> None:
        presenter = self._presenter
        payload = event.get("payload")
        if presenter is None or not isinstance(payload, dict):
            return
        token = payload.get("token")
        request_data = payload.get("request")
        origin = payload.get("origin")
        owner_data = payload.get("owner")
        if not isinstance(token, str) or not isinstance(request_data, dict):
            return
        try:
            request = _confirmation_request(request_data)
            owner = _confirmation_owner(owner_data, origin, event.get("run_id"))
            envelope = ConfirmationEnvelope(
                request=request,
                origin=cast(Any, origin),
                owner=owner,
                job_id=payload.get("job_id") if isinstance(payload.get("job_id"), str) else None,
                title=payload.get("title") if isinstance(payload.get("title"), str) else None,
            )
        except (TypeError, ValueError, KeyError):
            return
        local_token = object()
        self._items[local_token] = (token, owner)
        self._owners[owner] = local_token

        def respond(_token: object, decision: ConfirmationDecision) -> bool:
            if _token is not local_token or local_token not in self._items:
                return False
            task = asyncio.create_task(self._decide(local_token, token, decision))
            task.add_done_callback(_consume_task_result)
            return True

        presenter.present_confirmation(envelope, local_token, respond)

    async def handle_resolved(self, event: Mapping[str, object]) -> None:
        payload = event.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("token"), str):
            return
        wire_token = payload["token"]
        for local_token, (candidate, _owner) in tuple(self._items.items()):
            if candidate == wire_token:
                await self._dismiss_local(local_token)
                return

    async def _decide(
        self,
        local_token: object,
        wire_token: str,
        decision: ConfirmationDecision,
    ) -> None:
        with suppress(ServiceError):
            await self.client.decide_confirmation(wire_token, decision)
        await self._dismiss_local(local_token)

    async def _dismiss_local(self, local_token: object) -> None:
        item = self._items.pop(local_token, None)
        if item is None:
            return
        _wire_token, owner = item
        self._owners.pop(owner, None)
        presenter = self._presenter
        if presenter is not None:
            with suppress(Exception):
                result = presenter.dismiss_confirmation(local_token)
                if inspect.isawaitable(result):
                    await result


class RemoteManagementCommandDispatcher:
    """ManagementCommandDispatcher-shaped adapter for the terminal app."""

    def __init__(self, client: ServiceClient) -> None:
        self.client = client

    async def dispatch(self, command: str) -> Any:
        return _management_result(await self.client.management("dispatch", {"command": command}))

    async def update_reasoning_effort(self, effort: str) -> Any:
        return _management_result(await self.client.management("effort", {"effort": effort}))

    async def update_permission_level(self, level: ToolPermissionLevel) -> Any:
        return _management_result(
            await self.client.management("permission", {"permission_level": level})
        )

    async def resume(self, session_id: str, *, force: bool = False) -> Any:
        result = _management_result(
            await self.client.management("resume", {"session_id": session_id, "force": force})
        )
        if result.resumed_session_id == session_id:
            await self.client.switch_session(session_id)
        return result

    async def restore_inspect(self, anchor_id: int) -> Any:
        return _management_result(
            await self.client.management("restore/inspect", {"anchor_id": anchor_id})
        )

    async def restore_commit(self, plan: RestorePlan, mode: RestoreMode | str) -> Any:
        return _management_result(
            await self.client.management(
                "restore/execute",
                {"plan": {"anchor_id": plan.anchor_id}, "mode": str(mode)},
            )
        )

    async def restore_result(self) -> Any:
        return _management_result(await self.client.management("restore/result", {}))

    async def restore_cancel(self) -> Any:
        return _management_result(await self.client.management("restore/cancel", {}))

    async def restore_acknowledge_failure(self) -> Any:
        return _management_result(await self.client.management("restore/acknowledge", {}))


class ServiceClient:
    """Own one authenticated client connection and its remote presentation adapters."""

    _startup_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        *,
        agent_home: AgentHome,
        discovery: ServiceDiscovery,
        token: str,
        http: aiohttp.ClientSession,
        client_id: str,
        reconnect_credential: str,
    ) -> None:
        self.agent_home = agent_home
        self.discovery = discovery
        self.token = token
        self.http = http
        self.client_id = client_id
        self.reconnect_credential = reconnect_credential
        self.workspace_id = ""
        self.session_id = ""
        self.claim_version = 0
        self.claim_credential = ""
        self._socket: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, object]]] = {}
        self._closed = False
        self.bus = RemoteMessageBus(self)
        self.control = RemoteControl(self)
        self.confirmation = RemoteConfirmationCoordinator(self)
        self.management_dispatcher = RemoteManagementCommandDispatcher(self)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def base_url(self) -> str:
        return f"http://{self.discovery.host}:{self.discovery.port}"

    @classmethod
    async def connect_or_start(
        cls,
        agent_home: AgentHome,
        workspace: Path,
        *,
        port: int = DEFAULT_SERVICE_PORT,
        reconnect_credential: str | None = None,
        attach_workspace: bool = True,
    ) -> ServiceClient:
        agent_home.initialize()
        if attach_workspace:
            workspace = workspace.resolve(strict=True)
        key = str(agent_home.path.resolve())
        lock = cls._startup_locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                discovery = read_discovery(agent_home)
            except (OSError, ValueError):
                discovery = None
            token = _read_token_safely(agent_home)
            if discovery is not None and token is not None:
                info = await cls._probe(discovery, token)
                if info is not None:
                    _validate_protocol(info, discovery)
                    return await cls._connect(
                        agent_home,
                        workspace,
                        discovery,
                        token,
                        reconnect_credential,
                        attach_workspace,
                    )
            process = _spawn_service(agent_home, port)
            discovery, token = await cls._wait_for_started_service(
                agent_home,
                port,
                process,
            )
            return await cls._connect(
                agent_home,
                workspace,
                discovery,
                token,
                reconnect_credential,
                attach_workspace,
            )

    @classmethod
    async def stop_existing(
        cls,
        agent_home: AgentHome,
        *,
        port: int = DEFAULT_SERVICE_PORT,
    ) -> bool:
        try:
            discovery = read_discovery(agent_home)
        except (OSError, ValueError):
            discovery = None
        token = _read_token_safely(agent_home)
        if discovery is None or token is None:
            return False
        info = await cls._probe(discovery, token)
        if info is None:
            return False
        _validate_protocol(info, discovery)
        url = f"http://{discovery.host}:{discovery.port}{'/api/v1/service/stop'}"
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            headers = _auth_headers(token, mutation=True)
            async with http.post(
                url,
                headers=headers,
                json={"request_id": str(uuid4())},
            ) as response:
                return response.status == 200

    @classmethod
    async def _connect(
        cls,
        agent_home: AgentHome,
        workspace: Path,
        discovery: ServiceDiscovery,
        token: str,
        reconnect_credential: str | None,
        attach_workspace: bool,
    ) -> ServiceClient:
        timeout = aiohttp.ClientTimeout(total=15)
        http = aiohttp.ClientSession(timeout=timeout)
        client: ServiceClient | None = None
        try:
            client_data = await _http_json(
                http,
                f"http://{discovery.host}:{discovery.port}/api/v1/clients",
                token=token,
                mutation=True,
                payload={
                    "request_id": str(uuid4()),
                    "kind": "cli",
                    "reconnect_credential": reconnect_credential,
                },
            )
            client_id = _require_string(client_data, "client_id")
            new_reconnect = _require_string(client_data, "reconnect_credential")
            client = cls(
                agent_home=agent_home,
                discovery=discovery,
                token=token,
                http=http,
                client_id=client_id,
                reconnect_credential=new_reconnect,
            )
            if not attach_workspace:
                return client
            previous_workspace_id = client_data.get("current_workspace_id")
            previous_session_id = client_data.get("current_session_id")
            attached = await client._http_request(
                "POST",
                "/api/v1/workspaces/attach",
                payload={"request_id": str(uuid4()), "path": str(workspace)},
                mutation=True,
            )
            client.workspace_id = _require_string(attached, "workspace_id")
            if (
                isinstance(previous_workspace_id, str)
                and previous_workspace_id == client.workspace_id
                and isinstance(previous_session_id, str)
                and previous_session_id
            ):
                client.session_id = previous_session_id
            else:
                draft = await client._http_request(
                    "POST",
                    f"/api/v1/workspaces/{client.workspace_id}/sessions",
                    payload={"request_id": str(uuid4())},
                    mutation=True,
                )
                client.session_id = _require_string(draft, "session_id")
            await client._open_socket()
            claim = await client._command(
                "claim",
                workspace_id=client.workspace_id,
                session_id=client.session_id,
                claim_version=None,
                payload={},
            )
            claim_data = claim.get("claim")
            snapshot = claim.get("snapshot")
            if not isinstance(claim_data, dict) or not isinstance(snapshot, dict):
                raise ServiceStartupError(
                    "service_protocol_error", "Service claim response is invalid."
                )
            client.claim_version = _require_int(claim_data, "claim_version")
            client.claim_credential = _require_string(claim_data, "reconnect_credential")
            client.control.set_projection(_projection(snapshot))
            await client._apply_snapshot(snapshot)
            return client
        except BaseException:
            if client is None:
                await http.close()
            else:
                await client.close()
            raise

    async def _open_socket(self) -> None:
        self._socket = await self.http.ws_connect(
            f"{self.base_url}/api/v1/events",
            headers={
                **_auth_headers(self.token),
                "X-MyClaw-Client": self.client_id,
                "Origin": self.base_url,
            },
            heartbeat=20.0,
        )
        self._reader_task = asyncio.create_task(self._read_events())

    async def _read_events(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            async for message in socket:
                if message.type is aiohttp.WSMsgType.TEXT:
                    value = message.json()
                    if not isinstance(value, dict):
                        continue
                    request_id = value.get("request_id")
                    if isinstance(request_id, str) and (
                        value.get("accepted") is True or isinstance(value.get("code"), str)
                    ):
                        future = self._pending.get(request_id)
                        if future is not None and not future.done():
                            if value.get("accepted") is True:
                                future.set_result(value)
                            else:
                                future.set_exception(_service_error_from_wire(value))
                        continue
                    await self._handle_event(value)
                elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                    break
        except asyncio.CancelledError:
            return
        except Exception:
            return
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        ServiceStartupError("service_disconnected", "Service connection closed.")
                    )
            self.control.set_admitted(False)
            self.control.clear_runs()
            if not self._closed:
                await self.bus.put_remote_output(
                    {
                        "type": "system_control",
                        "content": "Local service connection closed.",
                        "metadata": {},
                    }
                )

    async def _handle_event(self, event: Mapping[str, object]) -> None:
        event_type = event.get("type")
        payload = event.get("payload")
        current_session = (
            event.get("workspace_id") == self.workspace_id
            and event.get("session_id") == self.session_id
        )
        if event_type == "input.accepted":
            if not current_session:
                return
            run_id = event.get("run_id")
            if isinstance(run_id, str):
                self.control.accept_run(run_id)
            await self.bus.accept_one_input()
        elif event_type == "run.output" and isinstance(payload, dict):
            if not current_session:
                return
            message = payload.get("message")
            if isinstance(message, dict):
                await self.bus.put_remote_output(message)
        elif event_type in {"run.completed", "run.cancelled", "run.failed"}:
            if not current_session:
                return
            run_id = event.get("run_id")
            if isinstance(run_id, str):
                self.control.finish_run(run_id)
        elif event_type == "confirmation.requested":
            await self.confirmation.handle_requested(event)
        elif event_type == "confirmation.resolved":
            await self.confirmation.handle_resolved(event)
        elif event_type == "snapshot.required":
            self.control.set_admitted(False)
        elif event_type == "project.removed" and event.get("workspace_id") == self.workspace_id:
            self.control.set_admitted(False)
            self.control.clear_runs()
            self.workspace_id = ""
            self.session_id = ""
            self.claim_version = 0
            self.claim_credential = ""
            await self.bus.put_remote_output(
                {
                    "type": "system_control",
                    "content": "Project registration was removed; its work has stopped.",
                    "metadata": {},
                }
            )
        elif (
            event_type == "project.removal.failed"
            and event.get("workspace_id") == self.workspace_id
        ):
            payload = event.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("message"), str):
                self.control.set_admitted(False)
                self.control.clear_runs()
                await self.bus.put_remote_output(
                    {
                        "type": "system_control",
                        "content": payload["message"],
                        "metadata": {},
                    }
                )

    async def _apply_snapshot(self, snapshot: Mapping[str, object]) -> None:
        projection = _projection(snapshot)
        self.control.set_projection(projection)

    async def _http_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        mutation: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return await _http_json(
            self.http,
            f"{self.base_url}{path}",
            token=self.token,
            client_id=self.client_id,
            mutation=mutation,
            method=method,
            payload=payload,
            extra_headers=extra_headers,
        )

    async def _command(
        self,
        command_type: str,
        *,
        workspace_id: str | None,
        session_id: str | None,
        claim_version: int | None,
        payload: dict[str, object],
    ) -> dict[str, object]:
        request_id = str(uuid4())
        command = {
            "request_id": request_id,
            "type": command_type,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "claim_version": claim_version,
            "payload": payload,
        }
        socket = self._socket
        if socket is None or socket.closed:
            raise ServiceStartupError("service_disconnected", "Service connection is closed.")
        future: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._send_lock:
                await socket.send_json(command)
            ack = await future
            result = ack.get("result")
            if not isinstance(result, dict):
                raise ServiceStartupError(
                    "service_protocol_error", "Service command response is invalid."
                )
            return result
        finally:
            self._pending.pop(request_id, None)

    async def submit_input(self, text: str) -> dict[str, object]:
        return await self._command(
            "input",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={"text": text},
        )

    async def claim_session(self, session_id: str) -> dict[str, object]:
        result = await self._command(
            "claim",
            workspace_id=self.workspace_id,
            session_id=session_id,
            claim_version=None,
            payload={},
        )
        claim = result.get("claim")
        snapshot = result.get("snapshot")
        if not isinstance(claim, dict) or not isinstance(snapshot, dict):
            raise ServiceStartupError(
                "service_protocol_error", "Service claim response is invalid."
            )
        if session_id != self.session_id:
            self.control.clear_runs()
        self.session_id = _require_string(claim, "session_id")
        self.claim_version = _require_int(claim, "claim_version")
        self.claim_credential = _require_string(claim, "reconnect_credential")
        await self._apply_snapshot(snapshot)
        return result

    async def list_sessions(self, workspace_id: str | None = None) -> list[dict[str, object]]:
        """Return foreground Session metadata without loading any Session body."""
        selected_workspace = self.workspace_id if workspace_id is None else workspace_id
        response = await self._http_request(
            "GET", f"/api/v1/workspaces/{selected_workspace}/sessions"
        )
        sessions = response.get("sessions")
        if not isinstance(sessions, list) or any(not isinstance(item, dict) for item in sessions):
            raise ServiceStartupError("service_protocol_error", "Session listing is invalid.")
        return cast(list[dict[str, object]], sessions)

    async def create_session(self, workspace_id: str | None = None) -> dict[str, object]:
        """Create a transient foreground Session draft."""
        selected_workspace = self.workspace_id if workspace_id is None else workspace_id
        return await self._http_request(
            "POST",
            f"/api/v1/workspaces/{selected_workspace}/sessions",
            payload={"request_id": str(uuid4())},
            mutation=True,
        )

    async def release_session(self) -> None:
        """Release the current idle Session Claim through the event channel."""
        await self._command(
            "release",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={},
        )
        self.claim_version = 0
        self.claim_credential = ""

    async def switch_session(self, session_id: str) -> None:
        await self.claim_session(session_id)

    async def cancel_run(self, run_id: str) -> None:
        await self._command(
            "cancel",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={"run_id": run_id},
        )

    async def decide_confirmation(self, token: str, decision: ConfirmationDecision) -> None:
        await self._command(
            "confirmation_decide",
            workspace_id=None,
            session_id=None,
            claim_version=None,
            payload={"token": token, "decision": decision},
        )

    async def management(self, action: str, payload: dict[str, object]) -> dict[str, object]:
        request_payload: dict[str, object] = {
            "request_id": str(uuid4()),
            "current_session_id": self.session_id,
            **payload,
        }
        extra_headers: dict[str, str] = {}
        if self.claim_version >= 1:
            request_payload["claim_version"] = self.claim_version
        if self.claim_credential:
            extra_headers["X-MyClaw-Claim"] = self.claim_credential
        response = await self._http_request(
            "POST",
            f"/api/v1/workspaces/{self.workspace_id}/management/{action}",
            payload=request_payload,
            mutation=True,
            extra_headers=extra_headers,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ServiceStartupError("service_protocol_error", "Management response is invalid.")
        if action == "restore/execute" and isinstance(result.get("restore_result"), dict):
            self.claim_version = _require_int(result, "claim_version")
            self.claim_credential = _require_string(result, "claim_credential")
            current = await self._http_request(
                "GET",
                (
                    f"/api/v1/workspaces/{self.workspace_id}/sessions/{self.session_id}"
                    f"?claim_version={self.claim_version}"
                ),
                extra_headers={"X-MyClaw-Claim": self.claim_credential},
            )
            snapshot = current.get("snapshot")
            if not isinstance(snapshot, dict):
                raise ServiceStartupError(
                    "service_protocol_error", "Restored Session snapshot is invalid."
                )
            await self._apply_snapshot(snapshot)
        return result

    async def create_web_ticket(self) -> str:
        """Create a short-lived browser launch URL without exposing the service credential."""
        response = await self._http_request(
            "POST",
            "/api/v1/web/ticket",
            payload={"request_id": str(uuid4())},
            mutation=True,
        )
        ticket = response.get("ticket")
        if not isinstance(ticket, str) or not ticket:
            raise ServiceStartupError(
                "service_protocol_error", "The local service did not return a Web ticket."
            )
        return f"{self.base_url}/#ticket={quote(ticket, safe='')}"

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
        if self._socket is not None and not self._socket.closed:
            with suppress(Exception):
                await self._socket.close()
        await self.http.close()

    @classmethod
    async def _wait_for_started_service(
        cls,
        agent_home: AgentHome,
        port: int,
        process: subprocess.Popen[bytes],
    ) -> tuple[ServiceDiscovery, str]:
        deadline = asyncio.get_running_loop().time() + 15.0
        exited_at: float | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                discovery = read_discovery(agent_home)
            except (OSError, ValueError):
                discovery = None
            token = _read_token_safely(agent_home)
            if discovery is not None and token is not None:
                info = await cls._probe(discovery, token)
                if info is not None:
                    _validate_protocol(info, discovery)
                    return discovery, token
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = asyncio.get_running_loop().time()
                if asyncio.get_running_loop().time() - exited_at < 2.0:
                    await asyncio.sleep(0.1)
                    continue
                if _port_is_open(DEFAULT_SERVICE_HOST, port):
                    raise ServiceStartupError(
                        "service_port_occupied",
                        "The local service port is occupied by another process; it was not stopped.",
                    )
                raise ServiceStartupError(
                    "service_start_failed",
                    "The local service exited before publishing discovery.",
                )
            await asyncio.sleep(0.1)
        if process.poll() is not None and _port_is_open(DEFAULT_SERVICE_HOST, port):
            raise ServiceStartupError(
                "service_port_occupied",
                "The local service port is occupied by another process; it was not stopped.",
            )
        raise ServiceStartupError("service_timeout", "The local service did not become ready.")

    @staticmethod
    async def _probe(discovery: ServiceDiscovery, token: str) -> dict[str, object] | None:
        timeout = aiohttp.ClientTimeout(total=1)
        challenge = secrets.token_urlsafe(24)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.get(
                    f"http://{discovery.host}:{discovery.port}/api/v1/service/identity",
                    params={"challenge": challenge},
                ) as response:
                    if response.status != 200:
                        return None
                    value = await response.json()
                    if not isinstance(value, dict):
                        return None
                    instance_id = value.get("service_instance_id")
                    protocol_version = value.get("protocol_version")
                    proof = value.get("proof")
                    if (
                        instance_id != discovery.service_instance_id
                        or isinstance(protocol_version, bool)
                        or not isinstance(protocol_version, int)
                        or not isinstance(proof, str)
                    ):
                        return None
                    expected = identity_proof(token, challenge, instance_id, protocol_version)
                    return value if hmac.compare_digest(proof, expected) else None
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError):
            return None


async def _http_json(
    http: aiohttp.ClientSession,
    url: str,
    *,
    token: str,
    client_id: str | None = None,
    method: str = "POST",
    mutation: bool = False,
    payload: dict[str, object] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    headers = _auth_headers(token, mutation=mutation)
    if client_id is not None:
        headers["X-MyClaw-Client"] = client_id
    if extra_headers is not None:
        headers.update(extra_headers)
    try:
        async with http.request(method, url, headers=headers, json=payload) as response:
            try:
                value = await response.json(content_type=None)
            except ValueError:
                code = "service_http_error" if response.status >= 400 else "service_protocol_error"
                raise ServiceStartupError(code, "The local service response is invalid.") from None
            if response.status >= 400:
                if isinstance(value, dict):
                    raise _service_error_from_wire(value, status=response.status)
                raise ServiceStartupError(
                    "service_http_error", "The local service rejected the request."
                )
            if not isinstance(value, dict):
                raise ServiceStartupError(
                    "service_protocol_error", "The local service response is invalid."
                )
            return value
    except aiohttp.ClientError as error:
        raise ServiceStartupError(
            "service_unavailable", "The local service could not be reached."
        ) from error


def _auth_headers(token: str, *, mutation: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if mutation:
        headers["X-MyClaw-CSRF"] = token
    return headers


def _spawn_service(agent_home: AgentHome, port: int) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "myclaw.service.process",
        "--agent-home",
        str(agent_home.path),
        "--host",
        DEFAULT_SERVICE_HOST,
        "--port",
        str(port),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError as error:
        raise ServiceStartupError(
            "service_start_failed", "The local service could not be started."
        ) from error


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _read_token_safely(agent_home: AgentHome) -> str | None:
    try:
        return read_credential(agent_home)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _validate_protocol(info: Mapping[str, object], discovery: ServiceDiscovery) -> None:
    if info.get("service_instance_id") != discovery.service_instance_id:
        raise ServiceStartupError(
            "service_identity_mismatch",
            "The discovered service identity changed; no connection was made.",
        )
    if info.get("protocol_version") != 1:
        raise ServiceStartupError(
            "service_protocol_mismatch",
            "The existing local service uses an incompatible protocol and was not stopped.",
        )


def _require_string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ServiceStartupError(
            "service_protocol_error", f"Service response field {key} is invalid."
        )
    return result


def _require_int(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise ServiceStartupError(
            "service_protocol_error", f"Service response field {key} is invalid."
        )
    return result


def _service_error_from_wire(value: Mapping[str, object], *, status: int = 409) -> ServiceError:
    code = value.get("code")
    message = value.get("message")
    fields = value.get("field_errors")
    return ServiceError(
        code if isinstance(code, str) and code else "service_error",
        message
        if isinstance(message, str) and message
        else "The local service rejected the request.",
        status=status if status in {400, 401, 403, 404, 409, 422, 500} else 409,
        retryable=value.get("retryable") is True,
        field_errors=fields if isinstance(fields, dict) else {},
    )


def _projection(snapshot: Mapping[str, object]) -> ForegroundConversationProjection:
    session_id = snapshot.get("session_id")
    messages = snapshot.get("messages")
    if not isinstance(session_id, str) or not isinstance(messages, list):
        raise ServiceStartupError("service_protocol_error", "Service Session snapshot is invalid.")
    normalized = tuple(item for item in messages if isinstance(item, dict))
    return ForegroundConversationProjection(session_id, normalized)


def _confirmation_request(value: Mapping[str, object]) -> ConfirmationRequest:
    details_value = value.get("details")
    warnings_value = value.get("warnings")
    reason_value = value.get("reason")
    details = dict(details_value) if isinstance(details_value, dict) else {}
    warnings = (
        tuple(item for item in warnings_value if isinstance(item, str))
        if isinstance(warnings_value, list)
        else ()
    )
    return ConfirmationRequest(
        UUID(_require_string(value, "confirmation_id")),
        _require_string(value, "tool_call_id"),
        _require_string(value, "tool_name"),
        _require_string(value, "summary"),
        details,
        warnings,
        reason=reason_value if isinstance(reason_value, str) else "",
    )


def _confirmation_owner(
    value: object,
    origin: object,
    event_run_id: object,
) -> ConfirmationOwner:
    if not isinstance(value, dict):
        raise ValueError("confirmation owner is missing")
    generation_id = UUID(_require_string(value, "generation_id"))
    if origin == "foreground":
        run_id = (
            UUID(_require_string(value, "run_id"))
            if value.get("run_id")
            else UUID(str(event_run_id))
        )
        return ForegroundConfirmationOwner(generation_id, run_id)
    job_id = _require_string(value, "job_id")
    occurrence_id = UUID(_require_string(value, "occurrence_id"))
    return BackgroundConfirmationOwner(generation_id, job_id, occurrence_id)


def _consume_task_result(task: asyncio.Task[object]) -> None:
    with suppress(BaseException):
        task.exception()


def _wire_value(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _wire_value(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire_value(item) for item in value]
    if isinstance(value, (Path, UUID, RestoreMode)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return {key: _wire_value(item) for key, item in vars(value).items()}
    return value


def _management_result(value: Mapping[str, object]) -> Any:
    from myclaw.management.commands import ManagementCommandResult

    output_value = value.get("output")
    memory_value = value.get("memory_content")
    resumed_value = value.get("resumed_session_id")
    skipped_value = value.get("resume_skipped_count")
    return ManagementCommandResult(
        handled=value.get("handled") is True,
        output=output_value if isinstance(output_value, str) else None,
        memory_content=memory_value if isinstance(memory_value, str) else None,
        dream_result=_dream_result(value.get("dream_result")),
        management_error=_management_error(value.get("management_error")),
        effort_selection=cast(Any, value.get("effort_selection"))
        if isinstance(value.get("effort_selection"), str)
        else None,
        permission_selection=cast(Any, value.get("permission_selection"))
        if isinstance(value.get("permission_selection"), str)
        else None,
        published_effort=cast(Any, value.get("published_effort"))
        if isinstance(value.get("published_effort"), str)
        else None,
        published_permission_level=cast(Any, value.get("published_permission_level"))
        if isinstance(value.get("published_permission_level"), str)
        else None,
        status_view=_status_view(value.get("status_view")),
        resume_sessions=_session_entries(value.get("resume_sessions")),
        resumed_session_id=resumed_value if isinstance(resumed_value, str) else None,
        resume_skipped_count=skipped_value if isinstance(skipped_value, int) else 0,
        skill_metadata=_skill_metadata(value.get("skill_metadata")),
        restore_listing=_restore_listing(value.get("restore_listing")),
        restore_plan=_restore_plan(value.get("restore_plan")),
        restore_result=_restore_result(value.get("restore_result")),
    )


def _dream_result(value: object) -> Any:
    from myclaw.agent.memory.dream import DreamResult

    if not isinstance(value, dict):
        return None
    status = value.get("status")
    processed_count = value.get("processed_count")
    memory_updated = value.get("memory_updated")
    cursor = value.get("cursor")
    if (
        not isinstance(status, str)
        or isinstance(processed_count, bool)
        or not isinstance(processed_count, int)
        or not isinstance(memory_updated, bool)
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
    ):
        return None
    try:
        return DreamResult(
            status=status,
            processed_count=processed_count,
            memory_updated=memory_updated,
            cursor=cursor,
            error=_management_error(value.get("error")),
        )
    except (TypeError, ValueError):
        return None


def _management_error(value: object) -> Any:
    from myclaw.errors import ErrorCode, ErrorInfo

    if not isinstance(value, dict):
        return None
    code = value.get("code")
    message = value.get("message")
    retryable = value.get("retryable")
    retry_after_seconds = value.get("retry_after_seconds")
    if (
        not isinstance(code, str)
        or not isinstance(message, str)
        or not isinstance(retryable, bool)
        or (
            retry_after_seconds is not None
            and (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, (int, float))
            )
        )
    ):
        return None
    try:
        return ErrorInfo(
            code=cast(ErrorCode, code),
            message=message,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
        )
    except (TypeError, ValueError):
        return None


def _status_view(value: object) -> Any:
    from myclaw.management.service import RuntimeStatus

    if not isinstance(value, dict):
        return None
    try:
        return RuntimeStatus(**value)
    except (TypeError, ValueError):
        return None


def _skill_metadata(value: object) -> tuple[Any, ...] | None:
    from myclaw.skills.catalog import SkillMetadata

    if not isinstance(value, list):
        return None
    result: list[SkillMetadata] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                SkillMetadata(
                    name=_require_string(item, "name"),
                    description=_require_string(item, "description"),
                    path=Path(_require_string(item, "path")),
                )
            )
        except (TypeError, ValueError, ServiceStartupError):
            continue
    return tuple(result)


def _session_entries(value: object) -> tuple[Any, ...] | None:
    from myclaw.management.service import SessionListingEntry

    if not isinstance(value, list):
        return None
    entries: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            entries.append(
                SessionListingEntry(
                    id=_require_string(item, "id"),
                    title=_require_string(item, "title"),
                    created_at=datetime.fromisoformat(_require_string(item, "created_at")),
                    updated_at=datetime.fromisoformat(_require_string(item, "updated_at")),
                    message_count=_require_int(item, "message_count"),
                )
            )
        except (TypeError, ValueError, ServiceStartupError):
            continue
    return tuple(entries)


def _restore_listing(value: object) -> Any:
    from myclaw.management.service import RestoreListingReport

    if not isinstance(value, dict):
        return None
    session_id = value.get("session_id")
    raw_anchors = value.get("anchors")
    if not isinstance(session_id, str) or not isinstance(raw_anchors, list):
        return None
    anchors: list[RestoreAnchor] = []
    for item in raw_anchors:
        if not isinstance(item, dict):
            continue
        try:
            anchors.append(
                RestoreAnchor(
                    int(item["anchor_id"]),
                    UUID(str(item["run_token"])),
                    str(item["content"]),
                    str(item["timestamp"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return RestoreListingReport(session_id=session_id, anchors=tuple(anchors))


def _restore_plan(value: object) -> RestorePlan | None:
    if not isinstance(value, dict):
        return None
    try:
        targets_value = value["targets"]
        gaps_value = value["backup_gaps"]
        issues_value = value["integrity_issues"]
        conflicts_value = value["conflict_targets"]
        tokens_value = value["discarded_run_tokens"]
        modes_value = value["available_modes"]
        if not all(
            isinstance(item, list)
            for item in (
                targets_value,
                gaps_value,
                issues_value,
                conflicts_value,
                tokens_value,
                modes_value,
            )
        ):
            return None
        return RestorePlan(
            session_id=_require_string(value, "session_id"),
            anchor_id=_require_int(value, "anchor_id"),
            session_digest=_require_string(value, "session_digest"),
            journal_revision=_require_nonnegative_int(value, "journal_revision"),
            removed_users=_require_nonnegative_int(value, "removed_users"),
            removed_messages=_require_nonnegative_int(value, "removed_messages"),
            targets=tuple(_restore_target(item) for item in targets_value),
            external_target_count=_require_nonnegative_int(value, "external_target_count"),
            backup_gaps=tuple(_restore_gap(item) for item in gaps_value),
            integrity_issues=tuple(_restore_integrity_issue(item) for item in issues_value),
            conflict_targets=tuple(Path(_require_string_value(item)) for item in conflicts_value),
            discarded_run_tokens=tuple(UUID(_require_string_value(item)) for item in tokens_value),
            available_modes=tuple(RestoreMode(_require_string_value(item)) for item in modes_value),
        )
    except (KeyError, TypeError, ValueError, ServiceStartupError):
        return None


def _restore_target(value: object) -> RestoreTarget:
    if not isinstance(value, dict):
        raise TypeError("restore target must be an object")
    requested = value.get("requested_targets")
    if not isinstance(requested, list):
        raise TypeError("restore target requested paths are invalid")
    before_sha = value.get("before_sha256")
    latest_sha = value.get("latest_after_sha256")
    backup_error = value.get("backup_error")
    latest_exists = value.get("latest_after_exists")
    if before_sha is not None and not isinstance(before_sha, str):
        raise TypeError("restore target before digest is invalid")
    if latest_sha is not None and not isinstance(latest_sha, str):
        raise TypeError("restore target after digest is invalid")
    if latest_exists is not None and not isinstance(latest_exists, bool):
        raise TypeError("restore target after state is invalid")
    if backup_error is not None and not isinstance(backup_error, str):
        raise TypeError("restore target backup error is invalid")
    return RestoreTarget(
        canonical_target=Path(_require_string(value, "canonical_target")),
        operation_id=_require_int(value, "operation_id"),
        requested_targets=tuple(Path(_require_string_value(item)) for item in requested),
        before_exists=_require_bool(value, "before_exists"),
        before_sha256=before_sha,
        before_bytes=None,
        latest_after_exists=latest_exists,
        latest_after_sha256=latest_sha,
        external=_require_bool(value, "external"),
        session_owned=_require_bool(value, "session_owned"),
        backup_error=backup_error,
    )


def _restore_gap(value: object) -> BackupGap:
    if not isinstance(value, dict):
        raise TypeError("restore backup gap must be an object")
    return BackupGap(
        operation_id=_require_int(value, "operation_id"),
        revision=_require_int(value, "revision"),
        run_token=UUID(_require_string(value, "run_token")),
        requested_target=_require_string(value, "requested_target"),
        canonical_target=_require_string(value, "canonical_target"),
        reason=_require_string(value, "reason"),
    )


def _restore_integrity_issue(value: object) -> BackupIntegrityIssue:
    if not isinstance(value, dict):
        raise TypeError("restore integrity issue must be an object")
    run_token_value = value.get("run_token")
    if run_token_value is not None and not isinstance(run_token_value, str):
        raise TypeError("restore integrity issue run token is invalid")
    return BackupIntegrityIssue(
        operation_id=_require_int(value, "operation_id"),
        reason=_require_string(value, "reason"),
        run_token=None if run_token_value is None else UUID(run_token_value),
    )


def _restore_result(value: object) -> RestoreResult | None:
    if value is None or not isinstance(value, dict):
        return None
    files_value = value.get("file_results")
    session_value = value.get("session_result")
    acknowledged = value.get("failure_notification_acknowledged")
    if not isinstance(files_value, list) or not isinstance(session_value, dict):
        return None
    try:
        updated_at = datetime.fromisoformat(_require_string(session_value, "updated_at"))
        file_results: list[RestoreFileResult] = []
        for item in files_value:
            if not isinstance(item, dict):
                return None
            error = item.get("error")
            if error is not None and not isinstance(error, str):
                return None
            file_results.append(
                RestoreFileResult(
                    target=Path(_require_string(item, "target")),
                    operation_id=_require_int(item, "operation_id"),
                    status=RestoreFileStatus(_require_string(item, "status")),
                    conflict=_require_bool(item, "conflict"),
                    error=error,
                )
            )
        return RestoreResult(
            session_id=_require_string(value, "session_id"),
            anchor_id=_require_int(value, "anchor_id"),
            mode=RestoreMode(_require_string(value, "mode")),
            removed_users=_require_nonnegative_int(value, "removed_users"),
            removed_messages=_require_nonnegative_int(value, "removed_messages"),
            file_results=tuple(file_results),
            session_result=SessionRestoreResult(
                session_id=_require_string(session_value, "session_id"),
                anchor_id=_require_int(session_value, "anchor_id"),
                removed_messages=_require_nonnegative_int(session_value, "removed_messages"),
                updated_at=updated_at,
            ),
            failure_notification_acknowledged=acknowledged is True,
        )
    except (KeyError, TypeError, ValueError, ServiceStartupError):
        return None


def _require_string_value(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a non-empty string")
    return value


def _require_bool(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result


def _require_nonnegative_int(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ServiceStartupError(
            "service_protocol_error", f"Service response field {key} is invalid."
        )
    return result


__all__ = [
    "RemoteConfirmationCoordinator",
    "RemoteControl",
    "RemoteManagementCommandDispatcher",
    "RemoteMessageBus",
    "ServiceClient",
    "ServiceStartupError",
]
