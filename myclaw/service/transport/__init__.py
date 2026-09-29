"""aiohttp transport for the authenticated local service boundary."""

from __future__ import annotations

import asyncio
import hmac
import secrets
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from ..discovery import identity_proof
from ..errors import ServiceError, service_error
from ..runtime import LocalService, ServiceSink

_API_PREFIX = "/api/v1"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
_CSRF_HEADER = "X-MyClaw-CSRF"
_CLIENT_HEADER = "X-MyClaw-Client"


@dataclass(slots=True)
class _RequestContext:
    request: web.Request
    token: str
    client_id: str | None


class _WebSocketSink(ServiceSink):
    def __init__(self, socket: web.WebSocketResponse) -> None:
        self.socket = socket
        self._send_lock = asyncio.Lock()

    async def send_event(self, event: dict[str, object]) -> None:
        async with self._send_lock:
            if not self.socket.closed:
                await self.socket.send_json(event)

    async def send_json(self, value: Mapping[str, object]) -> None:
        async with self._send_lock:
            if not self.socket.closed:
                await self.socket.send_json(value)


class LocalServiceTransport:
    """Bind HTTP, WebSocket, and safe static routes to one service instance."""

    def __init__(self, service: LocalService) -> None:
        self.service = service

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self._error_middleware])
        app["myclaw.service"] = self.service
        app.router.add_get("/", self._static_index)
        app.router.add_get(f"{_API_PREFIX}/service/identity", self._service_identity)
        app.router.add_get(f"{_API_PREFIX}/service", self._service_info)
        app.router.add_post(f"{_API_PREFIX}/clients", self._register_client)
        app.router.add_post(f"{_API_PREFIX}/workspaces/attach", self._attach_workspace)
        app.router.add_get(f"{_API_PREFIX}/projects", self._list_projects)
        app.router.add_post(f"{_API_PREFIX}/projects", self._register_project)
        app.router.add_delete(f"{_API_PREFIX}/projects/{{project_id}}", self._remove_project)
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/schedule-resume",
            self._resume_project_schedule,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions",
            self._list_sessions,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions",
            self._create_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}",
            self._get_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/management/{{action:.*}}",
            self._management,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/management/{{action:.*}}",
            self._management,
        )
        app.router.add_post(f"{_API_PREFIX}/service/stop", self._stop_service)
        app.router.add_get(f"{_API_PREFIX}/events", self._events)
        return app

    @web.middleware
    async def _error_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        try:
            return await handler(request)
        except ServiceError as error:
            request_id = _request_id_from_request(request)
            return web.json_response(error.to_dict(request_id), status=error.status)
        except web.HTTPException:
            raise
        except (OSError, ValueError, TypeError):
            service_failure = service_error(
                "validation_error",
                "The request is invalid.",
                status=422,
            )
            return web.json_response(
                service_failure.to_dict(_request_id_from_request(request)),
                status=422,
            )

    async def _static_index(self, request: web.Request) -> web.Response:
        self._check_host_origin(request, websocket=False)
        return web.Response(
            text=(
                '<!doctype html><html><head><meta charset="utf-8"><title>MyClaw</title>'
                "</head><body><main><h1>MyClaw local service</h1></main></body></html>"
            ),
            content_type="text/html",
            headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"},
        )

    async def _service_info(self, request: web.Request) -> web.Response:
        self._authenticate(request)
        return web.json_response(
            {
                "service_instance_id": self.service.service_instance_id,
                "protocol_version": self.service.protocol_version,
                "state": self.service.state,
                "active_workspace_count": len(self.service.workspaces),
            }
        )

    async def _service_identity(self, request: web.Request) -> web.Response:
        self._check_host_origin(request, websocket=False)
        challenge = request.query.get("challenge", "")
        if not 16 <= len(challenge) <= 128 or not all(
            character.isascii() and (character.isalnum() or character in "-_")
            for character in challenge
        ):
            raise service_error("validation_error", "Identity challenge is invalid.", status=422)
        try:
            token = _read_service_token(self.service)
        except (OSError, ValueError):
            raise service_error(
                "unauthenticated", "Service credential is unavailable.", status=401
            ) from None
        return web.json_response(
            {
                "service_instance_id": self.service.service_instance_id,
                "protocol_version": self.service.protocol_version,
                "proof": identity_proof(
                    token,
                    challenge,
                    self.service.service_instance_id,
                    self.service.protocol_version,
                ),
            }
        )

    async def _register_client(self, request: web.Request) -> web.Response:
        self._authenticate(request, mutation=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        kind = body.get("kind")
        reconnect = body.get("reconnect_credential")
        if kind not in {"cli", "web"}:
            raise service_error(
                "validation_error",
                "Client kind must be cli or web.",
                status=422,
                field_errors={"kind": "must be cli or web"},
            )
        if reconnect is not None and not isinstance(reconnect, str):
            raise service_error("validation_error", "Reconnect credential is invalid.", status=422)
        client = await self.service.register_client(kind, reconnect)
        return web.json_response(
            {
                "request_id": request_id,
                "client_id": client.client_id,
                "reconnect_credential": client.reconnect_credential,
                "permission_level": client.permission_control.current(),
                "current_workspace_id": client.current_workspace_id,
                "current_session_id": client.current_session_id,
            }
        )

    async def _attach_workspace(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        if self.service.client(client_id).kind != "cli":
            raise service_error("forbidden", "Only CLI clients may attach a directory.", status=403)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise service_error(
                "validation_error",
                "Workspace path is required.",
                status=422,
                field_errors={"path": "must be an absolute directory"},
            )
        workspace = await self.service.attach_workspace(client_id, _path_from_text(path))
        return web.json_response(
            {
                "request_id": request_id,
                "workspace_id": workspace.workspace_id,
                "project_id": None,
            }
        )

    async def _list_projects(self, request: web.Request) -> web.Response:
        self._authenticate(request)
        projects = []
        for record in self.service.projects.list():
            projects.append(
                {
                    "project_id": record.project_id,
                    "path": str(record.path),
                    "name": record.path.name,
                    "schedule_state": record.schedule_state,
                    "available": record.path.is_dir(),
                }
            )
        return web.json_response({"projects": projects})

    async def _register_project(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise service_error("validation_error", "Project path is required.", status=422)
        record, workspace, jobs = await self.service.register_project(
            client_id, _path_from_text(path)
        )
        return web.json_response(
            {
                "request_id": request_id,
                "project_id": record.project_id,
                "workspace_id": workspace.workspace_id,
                "schedule_state": record.schedule_state,
                "saved_jobs": [
                    {"job_id": job.job_id, "title": job.title, "schedule": job.schedule.to_dict()}
                    for job in jobs
                ],
            }
        )

    async def _remove_project(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        path = await self.service.remove_project(client_id, request.match_info["project_id"])
        return web.json_response({"request_id": request_id, "removed": True, "path": str(path)})

    async def _resume_project_schedule(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        job_ids = body.get("job_ids")
        if not isinstance(job_ids, list) or any(not isinstance(item, str) for item in job_ids):
            raise service_error("validation_error", "Reviewed Job IDs are invalid.", status=422)
        state = await self.service.resume_project_schedule(
            client_id, request.match_info["project_id"], set(job_ids)
        )
        return web.json_response({"request_id": request_id, "schedule_state": state})

    async def _list_sessions(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        sessions = await self.service.list_sessions(
            client_id,
            request.match_info["workspace_id"],
        )
        return web.json_response({"sessions": sessions})

    async def _create_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.create_session(
            client_id,
            request.match_info["workspace_id"],
        )
        return web.json_response({"request_id": request_id, **result})

    async def _get_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        workspace_id = request.match_info["workspace_id"]
        session_id = request.match_info["session_id"]
        claim_version = _integer_query(request, "claim_version")
        claim_credential = _required_header(request, "X-MyClaw-Claim")
        workspace = self.service.workspace(workspace_id)
        claim = workspace.require_claim(client_id, session_id, claim_version, claim_credential)
        projection = workspace.projection(session_id)
        return web.json_response(
            {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "claim_version": claim.version,
                "snapshot": {
                    "session_id": projection.session_id,
                    "messages": list(projection.messages),
                },
            }
        )

    async def _management(self, request: web.Request) -> web.Response:
        context = self._authenticate(
            request, mutation=request.method == "POST", client_required=True
        )
        client_id = _context_client_id(context)
        body = await _json_object(request) if request.method == "POST" else {}
        request_id = (
            _require_request_id(body)
            if request.method == "POST"
            else _request_id_from_request(request)
        )
        session_value = body.get("current_session_id")
        if not isinstance(session_value, str) or not session_value:
            session_value = request.query.get("session_id") or _required_header(
                request,
                "X-MyClaw-Session",
            )
        result = await self.service.handle_management(
            client_id,
            request.match_info["workspace_id"],
            session_value,
            request.match_info["action"],
            body,
        )
        return web.json_response({"request_id": request_id, "result": result})

    async def _stop_service(self, request: web.Request) -> web.Response:
        self._authenticate(request, mutation=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        operation_id = secrets.token_urlsafe(16)
        task = asyncio.create_task(self.service.stop())
        task.add_done_callback(_consume_task_result)
        return web.json_response(
            {"request_id": request_id, "accepted": True, "operation_id": operation_id}
        )

    async def _events(self, request: web.Request) -> web.StreamResponse:
        context = self._authenticate(request, websocket=True, client_required=True)
        client_id = _context_client_id(context)
        if self.service.client(client_id).connected:
            raise service_error("client_already_connected", "This Client already has a connection.")
        socket = web.WebSocketResponse(heartbeat=20.0, autoping=True)
        await socket.prepare(request)
        sink = _WebSocketSink(socket)
        try:
            await self.service.connect_client(client_id, sink)
            async for message in socket:
                if message.type is WSMsgType.TEXT:
                    value: object = None
                    try:
                        value = message.json()
                        if not isinstance(value, dict):
                            raise ValueError("command")
                        result = await self.service.handle_command(client_id, value)
                    except ServiceError as error:
                        result = error.to_dict(_request_id_from_value(value))
                    except (TypeError, ValueError):
                        service_failure = service_error(
                            "validation_error",
                            "The command is invalid.",
                            status=422,
                        )
                        result = service_failure.to_dict(_request_id_from_value(value))
                    await sink.send_json(result)
                elif message.type is WSMsgType.ERROR:
                    break
        finally:
            await self.service.disconnect_client(client_id, sink=sink)
            await socket.close()
        return socket

    def _authenticate(
        self,
        request: web.Request,
        *,
        mutation: bool = False,
        websocket: bool = False,
        client_required: bool = False,
    ) -> _RequestContext:
        self._check_host_origin(request, websocket=websocket)
        token = _bearer_token(request)
        try:
            expected = _read_service_token(self.service)
        except (OSError, ValueError):
            raise service_error(
                "unauthenticated", "Service credential is unavailable.", status=401
            ) from None
        if token is None or not hmac.compare_digest(token, expected):
            raise service_error(
                "unauthenticated", "Service authentication is required.", status=401
            )
        if mutation and not hmac.compare_digest(request.headers.get(_CSRF_HEADER, ""), token):
            raise service_error("forbidden", "A valid CSRF proof is required.", status=403)
        client_id = request.headers.get(_CLIENT_HEADER)
        if client_required:
            if not client_id:
                raise service_error(
                    "unauthenticated", "Client authentication is required.", status=401
                )
            self.service.client(client_id)
        return _RequestContext(request, token, client_id)

    @staticmethod
    def _check_host_origin(request: web.Request, *, websocket: bool) -> None:
        host_header = request.headers.get("Host", "")
        try:
            host = urlsplit(f"http://{host_header}")
            transport = request.transport
            address = None if transport is None else transport.get_extra_info("sockname")
            bound_port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
            valid_host = (
                host.hostname in _LOCAL_HOSTS
                and host.port == bound_port
                and host.username is None
                and host.password is None
                and not host.path
                and not host.query
                and not host.fragment
            )
        except ValueError:
            valid_host = False
        if not valid_host:
            raise service_error("forbidden", "Only local Host values are accepted.", status=403)
        origin = request.headers.get("Origin")
        if origin is None:
            if websocket:
                raise service_error("forbidden", "WebSocket Origin is required.", status=403)
            return
        try:
            parsed = urlsplit(origin)
            valid_origin = (
                parsed.scheme == "http"
                and parsed.hostname == host.hostname
                and parsed.port == bound_port
                and parsed.username is None
                and parsed.password is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid_origin = False
        if not valid_origin:
            raise service_error("forbidden", "Request Origin must match this service.", status=403)


def create_app(service: LocalService) -> web.Application:
    """Return the single local application used by tests and the service process."""
    return LocalServiceTransport(service).create_app()


async def _json_object(request: web.Request) -> dict[str, object]:
    try:
        value = await request.json()
    except (TypeError, ValueError):
        raise service_error(
            "validation_error", "Request JSON must be an object.", status=422
        ) from None
    if not isinstance(value, dict):
        raise service_error("validation_error", "Request JSON must be an object.", status=422)
    return value


def _read_service_token(service: LocalService) -> str:
    from myclaw.service.discovery import read_credential

    return read_credential(service.agent_home)


def _bearer_token(request: web.Request) -> str | None:
    value = request.headers.get("Authorization", "")
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.casefold() != "bearer" or not token:
        return None
    return token


def _request_id_from_request(request: web.Request) -> str:
    return request.headers.get("X-MyClaw-Request", "transport") or "transport"


def _request_id_from_value(value: object) -> str:
    if isinstance(value, dict) and isinstance(value.get("request_id"), str):
        request_id = value["request_id"]
        assert isinstance(request_id, str)
        return request_id
    return "transport"


def _require_request_id(body: Mapping[str, object]) -> str:
    value = body.get("request_id")
    if not isinstance(value, str) or not value:
        raise service_error("validation_error", "request_id is required.", status=422)
    return value


def _required_header(request: web.Request, name: str) -> str:
    value = request.headers.get(name)
    if not value:
        raise service_error("validation_error", f"{name} is required.", status=422)
    return value


def _integer_query(request: web.Request, name: str) -> int:
    raw = request.query.get(name)
    try:
        value = int(raw or "")
    except ValueError:
        raise service_error("validation_error", f"{name} is invalid.", status=422) from None
    if value < 1:
        raise service_error("validation_error", f"{name} is invalid.", status=422)
    return value


def _context_client_id(context: _RequestContext) -> str:
    client_id = context.client_id
    if client_id is None:
        raise service_error("unauthenticated", "Client authentication is required.", status=401)
    return client_id


def _consume_task_result(task: asyncio.Task[object]) -> None:
    with suppress(BaseException):
        task.exception()


def _path_from_text(value: str) -> Path:
    return Path(value)


__all__ = ["LocalServiceTransport", "create_app"]
