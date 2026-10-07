"""Python client adapter for the local service protocol."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import os
import secrets
import socket
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import quote
from uuid import UUID, uuid4

import aiohttp

from aide.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationOwner,
    ConfirmationPresentationCoordinator,
    ConfirmationPresenter,
    ForegroundConfirmationOwner,
    SubAgentConfirmationOwner,
)
from aide.agent.loop import ForegroundConversationProjection, TerminalAgentRunExecutorControl
from aide.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from aide.agent.permission import ToolPermissionLevel
from aide.agent.session.backup_store import BackupGap, BackupIntegrityIssue
from aide.agent.session.restore import (
    RestoreFileResult,
    RestoreFileStatus,
    RestoreMode,
    RestorePlan,
    RestoreResult,
    RestoreTarget,
)
from aide.agent.session.session import RestoreAnchor, SessionRestoreResult
from aide.agent.tools.tool_gateway import ConfirmationRequest
from aide.config.agent_home import AgentHome
from aide.service.discovery import (
    DEFAULT_SERVICE_HOST,
    DEFAULT_SERVICE_PORT,
    ServiceDiscovery,
    identity_proof,
    read_credential,
    read_discovery,
)
from aide.service.errors import ServiceError


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
        request_id = str(uuid4())
        await self.stage_submitted_input(message.content, request_id)
        try:
            result = await self.client.submit_user_input(message.content, request_id=request_id)
        except BaseException:
            await super().remove_inbound(request_id)
            raise
        if result.get("kind") == "management":
            await super().remove_inbound(request_id)

    async def stage_submitted_input(self, text: str, request_id: str) -> None:
        await super().put_inbound(InboundMessage(content=text, metadata={"request_id": request_id}))

    async def accept_one_input(self, request_id: str | None = None) -> None:
        if request_id is not None:
            await super().remove_inbound(request_id)
            return
        messages = await super().drain_inbound()
        for message in messages[1:]:
            await super().put_inbound(message)

    async def put_remote_output(
        self, value: Mapping[str, object], *, run_id: str | None = None
    ) -> None:
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
        projected_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if run_id is not None:
            projected_metadata["_remote_run_id"] = run_id
        await super().put_outbound(
            OutboundMessage(
                cast(Any, message_type),
                content if isinstance(content, str) else "",
                projected_metadata,
            )
        )


class RemoteControl(TerminalAgentRunExecutorControl):
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
        if any(candidate == token for candidate, _owner in self._items.values()):
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
        return await self.execute_management_command(command)

    async def execute_management_command(self, command: str) -> Any:
        return _management_result(await self.client.execute_management_command(command))

    async def submit_user_input(self, text: str) -> Any:
        from aide.management.commands import ManagementCommandResult

        request_id = str(uuid4())
        result = await self.client.submit_user_input(text, request_id=request_id)
        if result.get("kind") == "management":
            management_result = result.get("management_result")
            if not isinstance(management_result, dict):
                raise ServiceStartupError("service_protocol_error", "Management result is invalid.")
            return _management_result(management_result)
        if result.get("kind") != "conversation_input" or not isinstance(result.get("run_id"), str):
            raise ServiceStartupError("service_protocol_error", "Input result is invalid.")
        run_id = cast(str, result["run_id"])
        return ManagementCommandResult(
            handled=False,
            output=None,
            submitted=True,
            submitted_run_id=run_id,
        )

    async def recall_queued_inputs(self) -> list[dict[str, str]]:
        result = await self.client.recall_queued_inputs()
        items = result.get("recalled_inputs")
        if not isinstance(items, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("run_id"), str)
            or not isinstance(item.get("text"), str)
            for item in items
        ):
            raise ServiceStartupError("service_protocol_error", "Queue recall result is invalid.")
        return cast(list[dict[str, str]], items)

    async def update_reasoning_effort(self, effort: str) -> Any:
        return _management_result(await self.client.management("effort", {"effort": effort}))

    async def update_permission_level(self, level: ToolPermissionLevel) -> Any:
        return _management_result(
            await self.client.management("permission", {"permission_level": level})
        )

    async def resume(self, session_id: str, *, force: bool = False) -> Any:
        return _management_result(
            await self.client.management("resume", {"session_id": session_id, "force": force})
        )

    async def recover_conversation(self, *, create_new: bool = False) -> None:
        await self.client.recover_conversation(create_new=create_new)

    async def restore_inspect(self, anchor_id: int) -> Any:
        return _management_result(await self.client.inspect_restore(anchor_id))

    async def restore_commit(self, plan: RestorePlan, mode: RestoreMode | str) -> Any:
        return _management_result(await self.client.commit_restore(plan.anchor_id, str(mode)))

    async def restore_result(self) -> Any:
        return _management_result(await self.client.get_restore_result())

    async def restore_cancel(self) -> Any:
        return _management_result(await self.client.cancel_restore())

    async def restore_acknowledge_failure(self) -> Any:
        return _management_result(await self.client.acknowledge_restore_failure())


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
        self._previous_workspace_id: str | None = None
        self._previous_session_id: str | None = None
        self.claim_version = 0
        self.claim_credential = ""
        self._socket: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._subscription_task: asyncio.Task[None] | None = None
        self._event_cursor: tuple[str, int] | None = None
        self._awaiting_snapshot = False
        self._recovery_target: tuple[str, str] | None = None
        self._context_generation = 0
        self._resume_retry: tuple[str, dict[str, object], dict[str, str]] | None = None
        self._needs_conversation_recovery = False
        self._recovery_request: tuple[bool, str] | None = None
        self._state_listeners: set[Callable[[Mapping[str, object]], None]] = set()
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
        port: int | None = None,
        reconnect_credential: str | None = None,
        attach_workspace: bool = True,
    ) -> ServiceClient:
        if port is None:
            configured_port = os.environ.get("AIDE_SERVICE_PORT")
            if configured_port is not None:
                try:
                    port = int(configured_port)
                except ValueError:
                    raise ServiceStartupError(
                        "service_port_invalid", "The configured local service port is invalid."
                    ) from None
            else:
                port = DEFAULT_SERVICE_PORT
        if not 1 <= port <= 65535:
            raise ServiceStartupError(
                "service_port_invalid", "The configured local service port is invalid."
            )
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
                accepted = response.status == 200
        if not accepted:
            return False
        deadline = asyncio.get_running_loop().time() + 3.0
        while asyncio.get_running_loop().time() < deadline:
            if await cls._probe(discovery, token) is None:
                return True
            await asyncio.sleep(0.02)
        return True

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
            previous_workspace_id = client_data.get("current_workspace_id")
            previous_session_id = client_data.get("current_session_id")
            client._previous_workspace_id = (
                previous_workspace_id if isinstance(previous_workspace_id, str) else None
            )
            client._previous_session_id = (
                previous_session_id if isinstance(previous_session_id, str) else None
            )
            if not attach_workspace:
                return client
            await client.attach_workspace(workspace)
            return client
        except BaseException:
            if client is None:
                await http.close()
            else:
                await client.close()
            raise

    async def get_config(self) -> dict[str, object]:
        """Read the Service-owned saved and active configuration projection."""
        return await self._http_request("GET", "/api/v1/config")

    async def get_config_text(self) -> dict[str, object]:
        """Read the Service-owned redacted text view for the CLI command."""
        return await self._http_request("GET", "/api/v1/config/text")

    async def get_startup_config(self) -> dict[str, object]:
        """Read Service startup eligibility and safe diagnostics."""
        return await self._http_request("GET", "/api/v1/config/startup")

    async def attach_workspace(self, workspace: Path) -> None:
        """Attach, open, and claim a Workspace after startup eligibility is checked."""
        if self.workspace_id:
            raise ServiceStartupError(
                "workspace_already_attached", "A Workspace is already attached."
            )
        workspace = workspace.resolve(strict=True)
        await self._open_socket()
        await self.open_conversation(directory=str(workspace))

    async def _open_socket(self) -> None:
        self._socket = await self.http.ws_connect(
            f"{self.base_url}/api/v1/events",
            headers={
                **_auth_headers(self.token),
                "X-Aide-Client": self.client_id,
                "Origin": self.base_url,
            },
            heartbeat=20.0,
        )
        self._reader_task = asyncio.create_task(self._read_events())
        if self._event_cursor is not None:
            stream_id, sequence = self._event_cursor
            await self._command(
                "subscribe", workspace_id=None, session_id=None, claim_version=None,
                payload={"last_seq": sequence, "stream_id": stream_id},
            )
        await self.subscribe_state()

    def add_state_listener(
        self, listener: Callable[[Mapping[str, object]], None]
    ) -> Callable[[], None]:
        """Deliver authenticated state events without retaining UI drafts."""
        self._state_listeners.add(listener)
        return lambda: self._state_listeners.discard(listener)

    async def subscribe_state(self) -> dict[str, object]:
        return await self._command(
            "subscribe",
            workspace_id=None,
            session_id=None,
            claim_version=None,
            payload={"last_seq": None, "stream_id": None},
        )

    async def _restore_subscription(self) -> None:
        try:
            await self.subscribe_state()
        except (ServiceError, ServiceStartupError, aiohttp.ClientError):
            if self._socket is not None:
                await self._socket.close()
        finally:
            self._subscription_task = None

    async def _receive_event(self, event: Mapping[str, object]) -> None:
        if (event.get("service_instance_id") != self.discovery.service_instance_id
                or event.get("protocol_version") != self.discovery.protocol_version):
            return
        stream_id, sequence = event.get("stream_id"), event.get("seq")
        if (not isinstance(stream_id, str) or not stream_id or isinstance(sequence, bool)
                or not isinstance(sequence, int) or sequence < 1):
            return
        snapshot = event.get("type") == "snapshot.required"
        cursor = self._event_cursor
        if cursor is not None:
            if stream_id != cursor[0]:
                return
            if stream_id == cursor[0] and sequence <= cursor[1]:
                return
            if stream_id == cursor[0] and sequence > cursor[1] + 1 and not snapshot:
                self._awaiting_snapshot = True
                self.control.set_admitted(False)
                if self._subscription_task is None:
                    self._subscription_task = asyncio.create_task(self._restore_subscription())
                return
        if self._awaiting_snapshot and not snapshot:
            return
        recover_display = self._awaiting_snapshot
        if snapshot:
            self._awaiting_snapshot = False
        self._event_cursor = (stream_id, sequence)
        await self._handle_event(event, recover_display=recover_display)
        for listener in tuple(self._state_listeners):
            with suppress(Exception):
                listener(event)

    async def _reconnect(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(1)
                try:
                    discovery = read_discovery(self.agent_home)
                    token = _read_token_safely(self.agent_home)
                    if discovery is None or token is None:
                        continue
                    info = await self._probe(discovery, token)
                    if info is None:
                        continue
                    _validate_protocol(info, discovery)
                    same_instance = discovery.service_instance_id == self.discovery.service_instance_id
                    self.discovery, self.token = discovery, token
                    reconnect = self.reconnect_credential if same_instance else None
                    try:
                        registered = await self._http_request(
                            "POST", "/api/v1/clients", mutation=True,
                            payload={"request_id": str(uuid4()), "kind": "cli",
                                     "reconnect_credential": reconnect},
                        )
                    except ServiceError as error:
                        if error.code != "stale_client":
                            raise
                        registered = await self._http_request(
                            "POST", "/api/v1/clients", mutation=True,
                            payload={"request_id": str(uuid4()), "kind": "cli",
                                     "reconnect_credential": None},
                        )
                    client_id = _require_string(registered, "client_id")
                    if not same_instance or client_id != self.client_id:
                        self._context_generation += 1
                        self._needs_conversation_recovery = self._recovery_target is not None
                        self._recovery_request = None
                        self._resume_retry = None
                        self.workspace_id = ""
                        self.session_id = ""
                        self.claim_version = 0
                        self.claim_credential = ""
                        self._event_cursor = None
                        self.control.clear_runs()
                    self.client_id = client_id
                    self.reconnect_credential = _require_string(registered, "reconnect_credential")
                    await self._open_socket()
                    if self._socket is None or self._socket.closed:
                        continue
                    if self._needs_conversation_recovery:
                        try:
                            await self.recover_conversation()
                        except ServiceError as error:
                            await self._report_recovery_error(error.message)
                        except ServiceStartupError as error:
                            if self._socket is None or self._socket.closed:
                                continue
                            await self._report_recovery_error(error.message)
                    return
                except (OSError, ValueError, ServiceError, ServiceStartupError, aiohttp.ClientError):
                    if self._socket is not None:
                        await self._socket.close()
                    continue
        finally:
            self._reconnect_task = None

    async def _report_recovery_error(self, message: str) -> None:
        self.control.set_admitted(False)
        await self.bus.put_remote_output({
            "type": "system_control", "content": message,
            "metadata": {"_remote_connection_state": "session_unavailable",
                         "_remote_recovery_error": True},
        })

    async def recover_conversation(self, *, create_new: bool = False) -> None:
        target = self._recovery_target
        if target is None:
            raise ServiceStartupError("session_missing", "No Conversation Session to recover.")
        if self._recovery_request is None or self._recovery_request[0] != create_new:
            self._recovery_request = (create_new, str(uuid4()))
        try:
            if not create_new and self._resume_retry is not None:
                retry_payload = self._resume_retry[1]
                result = await self.management("resume", {
                    "session_id": retry_payload["session_id"], "force": retry_payload["force"],
                })
            else:
                result = await self.open_conversation(
                    directory=target[0], session_id=None if create_new else target[1],
                    create_new=create_new, request_id=self._recovery_request[1],
                )
        except ServiceError:
            self._recovery_request = None
            raise
        self._recovery_request = None
        self._needs_conversation_recovery = False
        self._resume_retry = None
        await self.bus.put_remote_output({
            "type": "system_control",
            "metadata": {"_remote_state_snapshot": result["snapshot"],
                         "_remote_snapshot_rebuild": True},
        })
        await self.bus.put_remote_output({
            "type": "system_control", "content": "Conversation Session recovered.",
            "metadata": {"_remote_connection_state": "online"},
        })

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
                    await self._receive_event(value)
                elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                    break
        except asyncio.CancelledError:
            return
        except Exception:
            return
        finally:
            if self._socket is socket:
                self._context_generation += 1
                self._socket = None
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(
                            ServiceStartupError("service_disconnected", "Service connection closed.")
                        )
                self.control.set_admitted(False)
                self.control.clear_runs()
                if not socket.closed:
                    with suppress(Exception):
                        await socket.close()
                if not self._closed:
                    for local in tuple(self.confirmation._items):
                        await self.confirmation._dismiss_local(local)
                    await self.bus.put_remote_output(
                        {
                            "type": "system_control",
                            "content": "Local service connection closed.",
                            "metadata": {"_remote_connection_state": "recovering"},
                        }
                    )
                    if self._reconnect_task is None:
                        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _handle_event(
        self, event: Mapping[str, object], *, recover_display: bool = False,
    ) -> None:
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
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            if isinstance(run_id, str):
                self.control.accept_run(run_id)
            if isinstance(request_id, str):
                await self.bus.accept_one_input(request_id)
            else:
                await self.bus.accept_one_input()
        elif event_type == "run.started" and current_session:
            run_id = event.get("run_id")
            if isinstance(run_id, str):
                await self.bus.put_remote_output(
                    {"type": "system_control", "metadata": {"_remote_run_started": True}},
                    run_id=run_id,
                )
        elif event_type == "run.output" and isinstance(payload, dict):
            if not current_session:
                return
            message = payload.get("message")
            if isinstance(message, dict):
                run_id = event.get("run_id")
                await self.bus.put_remote_output(
                    message, run_id=run_id if isinstance(run_id, str) else None
                )
        elif event_type in {"run.completed", "run.cancelled", "run.failed", "input.recalled"}:
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
            snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
            if not isinstance(snapshot, dict):
                return
            sessions = snapshot.get("sessions")
            if isinstance(sessions, list):
                for entry in sessions:
                    if (
                        not isinstance(entry, dict)
                        or entry.get("workspace_id") != self.workspace_id
                    ):
                        continue
                    session = entry.get("snapshot")
                    if (
                        not isinstance(session, dict)
                        or session.get("session_id") != self.session_id
                        or entry.get("claim_version") != self.claim_version
                    ):
                        continue
                    live = session.get("live_state")
                    if not isinstance(live, dict) or not isinstance(live.get("runs"), list):
                        continue
                    self.control.set_projection(_projection(session))
                    self.control.clear_runs()
                    for run in live["runs"]:
                        if isinstance(run, dict) and isinstance(run.get("run_id"), str):
                            self.control.accept_run(run["run_id"])
                    self.control.set_admitted(True)
                    await self.bus.put_remote_output({
                        "type": "system_control",
                        "metadata": {"_remote_state_snapshot": session,
                                     "_remote_snapshot_rebuild": recover_display,
                                     "_remote_snapshot_reason": (
                                         payload.get("reason") if isinstance(payload, dict) else None
                                     )},
                    })
                    await self.bus.put_remote_output({
                        "type": "system_control", "content": "Local service connection restored.",
                        "metadata": {"_remote_connection_state": "online"},
                    })
            if "pending_confirmation" in snapshot:
                pending = snapshot["pending_confirmation"]
                token = (
                    pending.get("payload", {}).get("token") if isinstance(pending, dict) else None
                )
                for local, (candidate, _owner) in tuple(self.confirmation._items.items()):
                    if candidate != token:
                        await self.confirmation._dismiss_local(local)
                if isinstance(pending, dict):
                    await self.confirmation.handle_requested(pending)
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
        self.control.set_admitted(self.claim_version >= 1 and bool(self.claim_credential))

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
        request_id: str | None = None,
    ) -> dict[str, object]:
        request_id = str(uuid4()) if request_id is None else request_id
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

    async def submit_user_input(
        self,
        text: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        return await self._command(
            "input",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={"text": text},
            request_id=request_id,
        )

    async def recall_queued_inputs(self) -> dict[str, object]:
        return await self._command(
            "recall_queued_inputs",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={},
        )

    async def open_conversation(
        self,
        *,
        project_id: str | None = None,
        workspace_id: str | None = None,
        directory: str | None = None,
        session_id: str | None = None,
        create_new: bool = False,
        request_id: str | None = None,
    ) -> dict[str, object]:
        generation = self._context_generation
        payload: dict[str, object] = {
            "request_id": request_id or str(uuid4()),
            "create_new": create_new,
        }
        for key, value in (
            ("project_id", project_id),
            (
                "workspace_id",
                (workspace_id or self.workspace_id or None)
                if project_id is None and directory is None
                else workspace_id,
            ),
            ("directory", directory),
            ("session_id", session_id),
        ):
            if value is not None:
                payload[key] = value
        result = await self._http_request(
            "POST", "/api/v1/conversations/open", payload=payload, mutation=True
        )
        if self._closed or generation != self._context_generation:
            raise ServiceStartupError("service_disconnected", "Conversation response is outdated.")
        await self._apply_conversation_context(result, directory=_require_string(result, "directory"))
        return result

    async def _apply_conversation_context(
        self, result: Mapping[str, object], *, directory: str | None = None,
    ) -> None:
        claim = result.get("claim")
        snapshot = result.get("snapshot")
        if not isinstance(claim, dict) or not isinstance(snapshot, dict):
            raise ServiceStartupError("service_protocol_error", "Conversation response is invalid.")
        next_workspace_id = _require_string(claim, "workspace_id")
        next_session_id = _require_string(claim, "session_id")
        next_version = _require_int(claim, "claim_version")
        next_credential = _require_string(claim, "reconnect_credential")
        if (
            snapshot.get("session_id") != next_session_id
            or result.get("session_id", next_session_id) != next_session_id
            or result.get("workspace_id", next_workspace_id) != next_workspace_id
            or result.get("resumed_session_id") not in (None, next_session_id)
        ):
            raise ServiceStartupError("service_protocol_error", "Conversation snapshot is invalid.")
        projection = _projection(snapshot)
        if (next_workspace_id, next_session_id) != (self.workspace_id, self.session_id):
            self.control.clear_runs()
        self.workspace_id = next_workspace_id
        self.session_id = next_session_id
        self.claim_version = next_version
        self.claim_credential = next_credential
        if directory is None and self._recovery_target is not None:
            directory = self._recovery_target[0]
        if directory is not None:
            self._recovery_target = (directory, next_session_id)
        self._resume_retry = None
        self.control.set_projection(projection)
        await self._apply_snapshot(snapshot)

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
        await self._http_request(
            "POST",
            f"/api/v1/workspaces/{self.workspace_id}/sessions/{self.session_id}/release",
            payload={"request_id": str(uuid4()), "claim_version": self.claim_version},
            mutation=True,
            extra_headers={"X-Aide-Claim": self.claim_credential},
        )
        self.claim_version = 0
        self.claim_credential = ""

    async def cancel_run(self, run_id: str) -> None:
        await self._command(
            "cancel",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            claim_version=self.claim_version,
            payload={"run_id": run_id},
        )

    async def execute_management_command(self, command: str) -> dict[str, object]:
        return await self.management("dispatch", {"command": command})

    async def inspect_restore(self, anchor_id: int) -> dict[str, object]:
        return await self.management("restore/inspect", {"anchor_id": anchor_id})

    async def commit_restore(self, anchor_id: int, mode: str) -> dict[str, object]:
        return await self.management(
            "restore/execute", {"plan": {"anchor_id": anchor_id}, "mode": mode}
        )

    async def get_restore_result(self) -> dict[str, object]:
        return await self.management("restore/result", {})

    async def cancel_restore(self) -> dict[str, object]:
        return await self.management("restore/cancel", {})

    async def acknowledge_restore_failure(self) -> dict[str, object]:
        return await self.management("restore/acknowledge", {})

    async def _named_session_operation(self, operation: str) -> dict[str, object]:
        if (
            not self.workspace_id
            or not self.session_id
            or self.claim_version < 1
            or not self.claim_credential
        ):
            raise ServiceStartupError("stale_claim", "A current Conversation Claim is required.")
        request_id = str(uuid4())
        response = await self._http_request(
            "POST",
            f"/api/v1/workspaces/{self.workspace_id}/{operation}",
            payload={
                "request_id": request_id,
                "current_session_id": self.session_id,
                "claim_version": self.claim_version,
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": self.claim_credential},
        )
        if (
            response.get("request_id") != request_id
            or response.get("workspace_id") != self.workspace_id
        ):
            raise ServiceStartupError("service_protocol_error", "Service operation response is invalid.")
        return response

    async def get_runtime_memory(self) -> dict[str, object]:
        """Read the current Workspace's Long-term Memory through its named operation."""
        response = await self._named_session_operation("memory/read")
        if not isinstance(response.get("content"), str):
            raise ServiceStartupError("service_protocol_error", "Memory response is invalid.")
        return response

    async def run_dream(self) -> dict[str, object]:
        """Run Dream for the current foreground Session and return its typed result."""
        response = await self._named_session_operation("memory/dream")
        if not isinstance(response.get("result"), dict):
            raise ServiceStartupError("service_protocol_error", "Dream response is invalid.")
        return response

    async def reload_runtime_skills(self) -> dict[str, object]:
        """Reload the shared Skill catalog and return published metadata."""
        response = await self._named_session_operation("skills/reload")
        skills = response.get("skills")
        if not isinstance(skills, list) or any(
            not isinstance(item, dict) for item in skills
        ):
            raise ServiceStartupError("service_protocol_error", "Skill response is invalid.")
        return response

    async def get_runtime_status(self) -> dict[str, object]:
        """Read the current foreground Session's Runtime Status projection."""
        response = await self._named_session_operation("runtime/status")
        if not isinstance(response.get("status"), dict):
            raise ServiceStartupError("service_protocol_error", "Runtime status response is invalid.")
        return response

    async def get_input_capabilities(self) -> dict[str, object]:
        response = await self._http_request("GET", "/api/v1/input-capabilities")
        command_tokens = response.get("management_commands")
        metadata = _skill_metadata(response.get("skill_metadata"))
        if (
            not isinstance(command_tokens, list)
            or any(
                not isinstance(token, str) or not token.startswith("/") for token in command_tokens
            )
            or metadata is None
        ):
            raise ServiceStartupError("service_protocol_error", "Input capabilities are invalid.")
        return {"management_commands": tuple(command_tokens), "skill_metadata": metadata}

    async def decide_confirmation(self, token: str, decision: ConfirmationDecision) -> None:
        await self._command(
            "confirmation_decide",
            workspace_id=None,
            session_id=None,
            claim_version=None,
            payload={"token": token, "decision": decision},
        )

    async def management(
        self, action: str, payload: dict[str, object], *, request_id: str | None = None,
    ) -> dict[str, object]:
        generation = self._context_generation
        request_payload: dict[str, object] = {
            "request_id": request_id or str(uuid4()),
            "current_session_id": self.session_id,
            **payload,
        }
        extra_headers: dict[str, str] = {}
        if self.claim_version >= 1:
            request_payload["claim_version"] = self.claim_version
        if self.claim_credential:
            extra_headers["X-Aide-Claim"] = self.claim_credential
        path = f"/api/v1/workspaces/{self.workspace_id}/management/{action}"
        if action == "resume":
            retry = self._resume_retry
            if (retry is not None
                    and (request_id is None or retry[1]["request_id"] == request_id)
                    and all(retry[1].get(key) == value for key, value in payload.items())):
                path, request_payload, extra_headers = retry
            self._resume_retry = (path, request_payload, extra_headers)
        try:
            response = await self._http_request(
                "POST", path, payload=request_payload, mutation=True, extra_headers=extra_headers,
            )
        except ServiceError:
            if action == "resume":
                self._resume_retry = None
            raise
        except (aiohttp.ClientError, TimeoutError, ServiceStartupError) as error:
            if action == "resume":
                if isinstance(error, ServiceStartupError) and error.code == "service_http_error":
                    self._resume_retry = None
                    raise
                self.claim_version = 0
                self.claim_credential = ""
                self.control.set_admitted(False)
                raise ServiceStartupError(
                    "result_unknown", "Session resume result is unknown. Retry the same selection.",
                ) from error
            raise
        if self._closed or generation != self._context_generation:
            raise ServiceStartupError("service_disconnected", "Management response is outdated.")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ServiceStartupError("service_protocol_error", "Management response is invalid.")
        try:
            if action == "resume" and result.get("resumed_session_id") is not None:
                claim = result.get("claim")
                if (result["resumed_session_id"] != request_payload.get("session_id")
                        or not isinstance(claim, dict)
                        or claim.get("workspace_id") != self.workspace_id):
                    raise ServiceStartupError(
                        "service_protocol_error", "Resumed Conversation context is invalid.",
                    )
            if result.get("resumed_session_id") is not None or (
                action == "restore/execute" and isinstance(result.get("restore_result"), dict)
            ):
                await self._apply_conversation_context(result)
        except ServiceStartupError:
            if action == "resume":
                self.control.set_admitted(False)
                self.claim_version = 0
                self.claim_credential = ""
            raise
        if action == "resume":
            self._resume_retry = None
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
        self._context_generation += 1
        for task in (self._reconnect_task, self._subscription_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
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
        headers["X-Aide-Client"] = client_id
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
        headers["X-Aide-CSRF"] = token
    return headers


def _spawn_service(agent_home: AgentHome, port: int) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "aide.service.process",
        "--agent-home",
        str(agent_home.path),
        "--host",
        DEFAULT_SERVICE_HOST,
        "--port",
        str(port),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
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
    if value.get("kind") == "subagent":
        return SubAgentConfirmationOwner(
            generation_id,
            _require_string(value, "workspace_id"),
            _require_string(value, "session_id"),
            _require_string(value, "agent_id"),
        )
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


def _management_result(value: Mapping[str, object]) -> Any:
    from aide.management.commands import ManagementCommandResult

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
    from aide.agent.memory.dream import DreamResult

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
    from aide.errors import ErrorCode, ErrorInfo

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
    from aide.management.service import RuntimeStatus

    if not isinstance(value, dict):
        return None
    try:
        return RuntimeStatus(**value)
    except (TypeError, ValueError):
        return None


def _skill_metadata(value: object) -> tuple[Any, ...] | None:
    from aide.skills.catalog import SkillMetadata

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
    from aide.management.service import SessionListingEntry

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
    from aide.management.service import RestoreListingReport

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
