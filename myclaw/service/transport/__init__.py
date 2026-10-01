"""aiohttp transport for the authenticated local service boundary."""

from __future__ import annotations

import asyncio
import hmac
import mimetypes
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from ...config.config import ConfigError
from ...schedule.model import ScheduleJob
from ..discovery import identity_proof
from ..errors import ServiceError, service_error
from ..projects import ProjectCatalogError
from ..runtime import LocalService, ServiceSink

_API_PREFIX = "/api/v1"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
_CSRF_HEADER = "X-MyClaw-CSRF"
_CLIENT_HEADER = "X-MyClaw-Client"
_WEB_CONTROL_HEADER = "X-MyClaw-Control"
_WEB_SESSION_COOKIE = "myclaw_session"
_WEB_TICKET_TTL_SECONDS = 25.0
_WEB_SESSION_MAX_AGE_SECONDS = 24 * 60 * 60
_STATIC_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self' ws:; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)


def _project_job_summary(job: ScheduleJob) -> dict[str, object]:
    due_at: datetime | None = None
    if job.schedule.kind == "at":
        due_at = job.schedule.at_datetime
    elif job.schedule.kind == "every" and job.schedule.every_seconds is not None:
        anchor_ms = (
            job.state.last_finished_at_ms
            if job.state.last_finished_at_ms is not None
            else job.created_at_ms
        )
        due_at = datetime.fromtimestamp(anchor_ms / 1000, UTC) + timedelta(
            seconds=job.schedule.every_seconds
        )
    if job.schedule.kind == "at" and job.state.last_status is not None:
        review_status = "completed"
    elif due_at is None:
        review_status = "next_on_resume"
    else:
        review_status = "overdue" if due_at <= datetime.now(UTC) else "upcoming"
    return {
        "job_id": job.job_id,
        "title": job.title,
        "schedule": job.schedule.to_dict(),
        "due_at": due_at.isoformat() if due_at is not None else None,
        "review_status": review_status,
    }


@dataclass(slots=True)
class _RequestContext:
    request: web.Request
    token: str | None
    client_id: str | None
    web_session: _WebSession | None = None


@dataclass(slots=True)
class _WebSession:
    csrf_token: str
    expires_at: float
    client_id: str | None = None
    reconnect_credential: str | None = None


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
        self._web_tickets: dict[str, float] = {}
        self._web_sessions: dict[str, _WebSession] = {}
        self._web_auth_lock = asyncio.Lock()

    def _prune_web_tickets(self) -> None:
        now = time.monotonic()
        for ticket, expires_at in tuple(self._web_tickets.items()):
            if expires_at <= now:
                self._web_tickets.pop(ticket, None)
        for cookie, session in tuple(self._web_sessions.items()):
            if session.expires_at <= now:
                self._web_sessions.pop(cookie, None)

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self._error_middleware])
        app["myclaw.service"] = self.service
        app.router.add_get("/", self._static_index)
        app.router.add_post(f"{_API_PREFIX}/web/ticket", self._web_ticket)
        app.router.add_get(f"{_API_PREFIX}/web/session", self._web_session_info)
        app.router.add_get(f"{_API_PREFIX}/service/identity", self._service_identity)
        app.router.add_get(f"{_API_PREFIX}/service", self._service_info)
        app.router.add_get(f"{_API_PREFIX}/config", self._config)
        app.router.add_patch(f"{_API_PREFIX}/config", self._patch_config)
        app.router.add_post(f"{_API_PREFIX}/config/repair", self._repair_config)
        app.router.add_post(f"{_API_PREFIX}/config/retry", self._retry_config)
        app.router.add_post(f"{_API_PREFIX}/clients", self._register_client)
        app.router.add_post(f"{_API_PREFIX}/workspaces/attach", self._attach_workspace)
        app.router.add_get(f"{_API_PREFIX}/projects", self._list_projects)
        app.router.add_post(f"{_API_PREFIX}/projects", self._register_project)
        app.router.add_delete(f"{_API_PREFIX}/projects/{{project_id}}", self._remove_project)
        app.router.add_get(
            f"{_API_PREFIX}/projects/{{project_id}}/removal/{{operation_id}}",
            self._project_removal_status,
        )
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/schedule-resume",
            self._resume_project_schedule,
        )
        app.router.add_get(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions",
            self._list_project_sessions,
        )
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions",
            self._create_project_session,
        )
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}/claim",
            self._claim_project_session,
        )
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}/release",
            self._release_project_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}",
            self._get_project_session,
        )
        app.router.add_patch(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}",
            self._rename_project_session,
        )
        app.router.add_delete(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}",
            self._delete_project_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}/deletion-status",
            self._project_session_deletion_status,
        )
        app.router.add_post(
            f"{_API_PREFIX}/projects/{{project_id}}/sessions/{{session_id}}/deletion-claim",
            self._claim_project_session_deletion,
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
        app.router.add_patch(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}",
            self._rename_session,
        )
        app.router.add_delete(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}",
            self._delete_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/schedule/jobs",
            self._list_schedule_jobs,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/schedule/jobs",
            self._create_schedule_job,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/schedule/jobs/{{job_id}}",
            self._get_schedule_job,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/schedule/jobs/{{job_id}}/history",
            self._get_schedule_job_history,
        )
        app.router.add_delete(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/schedule/jobs/{{job_id}}",
            self._delete_schedule_job,
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
        app.router.add_get("/assets/{asset_path:.*}", self._static_asset)
        app.router.add_get("/{static_path:.*}", self._static_route)
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
        except ConfigError:
            failure = service_error("config_invalid", "User Configuration is invalid.", status=422)
            return web.json_response(failure.to_dict(_request_id_from_request(request)), status=422)
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
        return self._asset_response("index.html", cache_control="no-store")

    async def _static_asset(self, request: web.Request) -> web.Response:
        self._check_host_origin(request, websocket=False)
        asset_path = request.match_info.get("asset_path", "")
        if not _safe_asset_path(asset_path):
            raise web.HTTPNotFound()
        return self._asset_response(
            f"assets/{asset_path}",
            cache_control="public, max-age=31536000, immutable",
        )

    async def _static_route(self, request: web.Request) -> web.Response:
        self._check_host_origin(request, websocket=False)
        route_path = request.match_info.get("static_path", "").strip("/")
        if route_path.startswith("api/"):
            raise web.HTTPNotFound()
        if route_path in {"", "index.html"}:
            return self._asset_response("index.html", cache_control="no-store")
        if route_path in {"favicon.svg", "manifest.webmanifest"}:
            return self._asset_response(
                route_path,
                cache_control="public, max-age=31536000, immutable",
            )
        if "." in Path(route_path).name:
            raise web.HTTPNotFound()
        return self._asset_response("index.html", cache_control="no-store")

    @staticmethod
    def _asset_response(asset_path: str, *, cache_control: str) -> web.Response:
        try:
            content = _read_web_asset(asset_path)
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            raise web.HTTPNotFound() from None
        content_type, encoding = mimetypes.guess_type(asset_path)
        response = web.Response(
            body=content,
            content_type=content_type or "application/octet-stream",
            charset="utf-8" if content_type and content_type.startswith("text/") else None,
            headers={
                "Cache-Control": cache_control,
                "Content-Security-Policy": _STATIC_CSP,
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )
        if encoding is not None:
            response.headers["Content-Encoding"] = encoding
        return response

    async def _web_ticket(self, request: web.Request) -> web.Response:
        if _bearer_token(request) is None:
            return await self._web_exchange(request)
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        if self.service.client(client_id).kind != "cli":
            raise service_error(
                "forbidden", "Only CLI clients may open the Web Interface.", status=403
            )
        body = await _json_object(request)
        request_id = _require_request_id(body)
        ticket = secrets.token_urlsafe(32)
        async with self._web_auth_lock:
            self._prune_web_tickets()
            self._web_tickets[ticket] = time.monotonic() + _WEB_TICKET_TTL_SECONDS
        return web.json_response(
            {
                "request_id": request_id,
                "ticket": ticket,
                "expires_in": int(_WEB_TICKET_TTL_SECONDS),
            }
        )

    async def _web_exchange(self, request: web.Request) -> web.Response:
        self._check_host_origin(request, websocket=False, require_origin=True)
        body = await _json_object(request)
        ticket = body.get("ticket")
        if not isinstance(ticket, str) or not 32 <= len(ticket) <= 128:
            raise service_error("unauthenticated", "The Web launch ticket is invalid.", status=401)
        async with self._web_auth_lock:
            expires_at = self._web_tickets.pop(ticket, None)
            if expires_at is None or expires_at <= time.monotonic():
                raise service_error(
                    "unauthenticated", "The Web launch ticket is invalid or expired.", status=401
                )
            session_cookie = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            self._prune_web_tickets()
            self._web_sessions[session_cookie] = _WebSession(
                csrf_token=csrf_token,
                expires_at=time.monotonic() + _WEB_SESSION_MAX_AGE_SECONDS,
            )
        response = web.json_response({"authenticated": True, "csrf_token": csrf_token})
        response.set_cookie(
            _WEB_SESSION_COOKIE,
            session_cookie,
            max_age=_WEB_SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    async def _web_session_info(self, request: web.Request) -> web.Response:
        context = self._authenticate(request)
        if context.web_session is None:
            raise service_error("forbidden", "A browser session is required.", status=403)
        return web.json_response(
            {
                "authenticated": True,
                "csrf_token": context.web_session.csrf_token,
                "client_id": context.web_session.client_id,
            }
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

    async def _config(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.config_view())

    async def _patch_config(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        if set(body) != {"request_id", "revision", "fields", "secrets"}:
            raise service_error(
                "validation_error", "Configuration request fields are invalid.", status=422
            )
        request_id = _require_request_id(body)
        revision = body.get("revision")
        fields = body.get("fields")
        if not isinstance(revision, str) or not revision:
            raise service_error(
                "validation_error",
                "Configuration revision is required.",
                status=422,
                field_errors={"revision": "must be a nonempty string"},
            )
        if not isinstance(fields, Mapping):
            raise service_error(
                "validation_error",
                "Configuration fields must be an object.",
                status=422,
                field_errors={"fields": "must be an object"},
            )
        secrets = body.get("secrets")
        if not isinstance(secrets, Mapping):
            raise service_error(
                "validation_error",
                "Configuration secret operations must be an object.",
                status=422,
                field_errors={"secrets": "must be an object"},
            )
        result = await self.service.update_configuration(
            request_id, revision, fields, secrets, client_id=context.client_id
        )
        return web.json_response({"request_id": request_id, **result})

    async def _retry_config(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        if set(body) != {"request_id", "revision"}:
            raise service_error(
                "validation_error", "Configuration retry fields are invalid.", status=422
            )
        request_id = _require_request_id(body)
        revision = body.get("revision")
        if not isinstance(revision, str) or not revision:
            raise service_error(
                "validation_error",
                "Configuration revision is required.",
                status=422,
                field_errors={"revision": "must be a nonempty string"},
            )
        result = await self.service.retry_configuration(
            request_id, revision, client_id=context.client_id
        )
        return web.json_response({"request_id": request_id, **result})

    async def _repair_config(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        if set(body) != {"request_id", "revision", "fields", "secrets"}:
            raise service_error(
                "validation_error", "Configuration repair fields are invalid.", status=422
            )
        request_id = _require_request_id(body)
        revision = body.get("revision")
        fields = body.get("fields")
        if not isinstance(revision, str) or not revision:
            raise service_error(
                "validation_error",
                "Configuration revision is required.",
                status=422,
                field_errors={"revision": "must be a nonempty string"},
            )
        if not isinstance(fields, Mapping):
            raise service_error(
                "validation_error",
                "Configuration fields must be an object.",
                status=422,
                field_errors={"fields": "must be an object"},
            )
        secrets = body.get("secrets")
        if not isinstance(secrets, Mapping):
            raise service_error(
                "validation_error",
                "Configuration secret operations must be an object.",
                status=422,
                field_errors={"secrets": "must be an object"},
            )
        result = await self.service.repair_configuration(
            request_id, revision, fields, secrets, client_id=context.client_id
        )
        return web.json_response({"request_id": request_id, **result})

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
        context = self._authenticate(request, mutation=True)
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
        if context.web_session is not None:
            if kind != "web":
                raise service_error(
                    "forbidden", "Browser sessions may only register Web clients.", status=403
                )
            session = context.web_session
            if session.client_id is not None:
                reconnect = session.reconnect_credential
            client = await self.service.register_client(kind, reconnect)
            session.client_id = client.client_id
            session.reconnect_credential = client.reconnect_credential
        else:
            client = await self.service.register_client(kind, reconnect)
        return web.json_response(
            {
                "request_id": request_id,
                "client_id": client.client_id,
                "reconnect_credential": client.reconnect_credential,
                **(
                    {"web_control_credential": client.web_control_credential}
                    if kind == "web"
                    else {}
                ),
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
        try:
            records = self.service.projects.list()
        except ProjectCatalogError as error:
            raise service_error(
                "persistence_error", "The Project catalog could not be read safely.", status=500
            ) from error
        for record in records:
            saved_jobs, schedule_status = await self.service.project_schedule_snapshot(record)
            project = {
                "project_id": record.project_id,
                "path": str(record.path),
                "name": record.path.name,
                "schedule_state": record.schedule_state,
                "available": record.path.is_dir() and self.service.configuration_ready,
                "saved_jobs": [_project_job_summary(job) for job in saved_jobs],
                "schedule_status": schedule_status,
            }
            if record.removal_operation_id is not None:
                project["removal_operation_id"] = record.removal_operation_id
            if record.removal_error is not None:
                project["removal_error"] = record.removal_error
            projects.append(project)
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
                "saved_jobs": [_project_job_summary(job) for job in jobs],
            }
        )

    async def _remove_project(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.start_project_removal(
            client_id,
            request.match_info["project_id"],
        )
        return web.json_response({"request_id": request_id, **result})

    async def _project_removal_status(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.project_removal_status(
            _context_client_id(context),
            request.match_info["project_id"],
            request.match_info["operation_id"],
        )
        return web.json_response(result)

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

    async def _list_project_sessions(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        project_id = request.match_info["project_id"]
        record, workspace, page = await self.service.list_project_sessions_page(
            client_id,
            project_id,
            title=request.query.get("title"),
            cursor=request.query.get("cursor"),
            limit=_session_page_limit(request),
        )
        return web.json_response(
            {
                "project_id": record.project_id,
                "workspace_id": workspace.workspace_id,
                **page,
            }
        )

    async def _create_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.create_project_session(
            client_id, request.match_info["project_id"]
        )
        return web.json_response({"request_id": request_id, **result})

    async def _claim_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.claim_project_session(
            client_id,
            request.match_info["project_id"],
            request.match_info["session_id"],
        )
        return web.json_response(
            {"request_id": request_id, "project_id": request.match_info["project_id"], **result}
        )

    async def _get_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        result = await self.service.get_project_session(
            client_id,
            request.match_info["project_id"],
            request.match_info["session_id"],
            _integer_query(request, "claim_version"),
            _required_header(request, "X-MyClaw-Claim"),
        )
        return web.json_response(result)

    async def _project_session_deletion_status(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.project_session_deletion_status(
            _context_client_id(context),
            request.match_info["project_id"],
            request.match_info["session_id"],
        )
        return web.json_response(result)

    async def _claim_project_session_deletion(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.claim_project_session_deletion(
            _context_client_id(context),
            request.match_info["project_id"],
            request.match_info["session_id"],
        )
        return web.json_response({"request_id": request_id, **result})

    async def _rename_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        claim_version, title, expected_metadata_version = _rename_fields(body)
        result = await self.service.rename_project_session(
            client_id,
            request.match_info["project_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-MyClaw-Claim"),
            title,
            expected_metadata_version,
            request_id,
        )
        return web.json_response(
            {
                "request_id": request_id,
                "project_id": request.match_info["project_id"],
                "session": result,
            }
        )

    async def _release_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        claim_version = body.get("claim_version")
        if (
            isinstance(claim_version, bool)
            or not isinstance(claim_version, int)
            or claim_version < 1
        ):
            raise service_error("validation_error", "claim_version is invalid.", status=422)
        await self.service.release_project_session(
            client_id,
            request.match_info["project_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-MyClaw-Claim"),
        )
        return web.json_response({"request_id": request_id, "released": True})

    async def _delete_project_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        claim_version, confirm = _delete_fields(body)
        result = await self.service.delete_project_session(
            client_id,
            request.match_info["project_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-MyClaw-Claim"),
            request_id,
        )
        del confirm
        return web.json_response({"request_id": request_id, **result})

    async def _list_sessions(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        page = await self.service.list_sessions_page(
            client_id,
            request.match_info["workspace_id"],
            title=request.query.get("title"),
            cursor=request.query.get("cursor"),
            limit=_session_page_limit(request),
        )
        return web.json_response(page)

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
        workspace._ensure_session_available(session_id)
        return web.json_response(
            {
                "workspace_id": workspace_id,
                "session_id": session_id,
                "claim_version": claim.version,
                "snapshot": workspace.session_snapshot(session_id),
            }
        )

    async def _rename_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        claim_version, title, expected_metadata_version = _rename_fields(body)
        result = await self.service.rename_session(
            client_id,
            request.match_info["workspace_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-MyClaw-Claim"),
            title,
            expected_metadata_version,
            request_id,
        )
        return web.json_response(
            {
                "request_id": request_id,
                "workspace_id": request.match_info["workspace_id"],
                "session_id": request.match_info["session_id"],
                "session": result,
            }
        )

    async def _delete_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        claim_version, confirm = _delete_fields(body)
        result = await self.service.delete_session(
            client_id,
            request.match_info["workspace_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-MyClaw-Claim"),
            request_id,
        )
        del confirm
        return web.json_response({"request_id": request_id, **result})

    async def _list_schedule_jobs(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.list_schedule_jobs(
            _context_client_id(context),
            request.match_info["workspace_id"],
        )
        return web.json_response(result)

    async def _create_schedule_job(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.create_schedule_job(
            _context_client_id(context),
            request.match_info["workspace_id"],
            body,
            request_id,
        )
        return web.json_response(result)

    async def _get_schedule_job(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.get_schedule_job(
            _context_client_id(context),
            request.match_info["workspace_id"],
            request.match_info["job_id"],
        )
        return web.json_response(result)

    async def _get_schedule_job_history(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.get_schedule_job_history(
            _context_client_id(context),
            request.match_info["workspace_id"],
            request.match_info["job_id"],
            cursor=request.query.get("cursor"),
            limit=_schedule_history_page_limit(request),
        )
        return web.json_response(result)

    async def _delete_schedule_job(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.delete_schedule_job(
            _context_client_id(context),
            request.match_info["workspace_id"],
            request.match_info["job_id"],
            request_id,
            body,
        )
        return web.json_response(result)

    async def _management(self, request: web.Request) -> web.Response:
        if request.match_info["action"] in {"memory", "dream", "skills/reload"}:
            if request.method != "POST":
                raise web.HTTPMethodNotAllowed(request.method, ["POST"])
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
        raw_session_value = body.get("current_session_id")
        session_value: str | None = (
            raw_session_value if isinstance(raw_session_value, str) and raw_session_value else None
        )
        if session_value is None:
            query_session = request.query.get("session_id")
            if isinstance(query_session, str) and query_session:
                session_value = query_session
            else:
                session_value = _required_header(request, "X-MyClaw-Session")
        if request.method == "GET":
            body["request_id"] = request_id
            raw_query_claim_version = request.query.get("claim_version")
            claim_version = (
                _integer_query(request, "claim_version") if raw_query_claim_version else None
            )
        else:
            raw_body_claim_version = body.get("claim_version")
            if isinstance(raw_body_claim_version, int) and not isinstance(
                raw_body_claim_version, bool
            ):
                claim_version = raw_body_claim_version
            else:
                claim_version = None
        result = await self.service.handle_management(
            client_id,
            request.match_info["workspace_id"],
            session_value,
            request.match_info["action"],
            body,
            claim_version=claim_version,
            claim_credential=request.headers.get("X-MyClaw-Claim"),
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
        socket = web.WebSocketResponse(heartbeat=20.0, autoping=True, protocols=("myclaw-v1",))
        await socket.prepare(request)
        sink = _WebSocketSink(socket)
        try:
            await self.service.connect_client(
                client_id,
                sink,
                wait_for_subscribe=self.service.client(client_id).kind == "web",
            )
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
        web_session: _WebSession | None = None
        csrf_token: str
        if token is not None:
            try:
                expected = _read_service_token(self.service)
            except (OSError, ValueError):
                raise service_error(
                    "unauthenticated", "Service credential is unavailable.", status=401
                ) from None
            if not hmac.compare_digest(token, expected):
                raise service_error(
                    "unauthenticated", "Service authentication is required.", status=401
                )
            csrf_token = token
        else:
            session_cookie = request.cookies.get(_WEB_SESSION_COOKIE)
            if not session_cookie:
                raise service_error(
                    "unauthenticated", "Service authentication is required.", status=401
                )
            web_session = self._web_sessions.get(session_cookie)
            if web_session is None or web_session.expires_at <= time.monotonic():
                self._web_sessions.pop(session_cookie, None)
                raise service_error(
                    "unauthenticated", "Browser authentication has expired.", status=401
                )
            csrf_token = web_session.csrf_token
        if mutation and not hmac.compare_digest(request.headers.get(_CSRF_HEADER, ""), csrf_token):
            raise service_error("forbidden", "A valid CSRF proof is required.", status=403)
        client_id = request.headers.get(_CLIENT_HEADER)
        if web_session is not None:
            if web_session.client_id is None and client_id is not None:
                raise service_error("forbidden", "Browser Client identity is invalid.", status=403)
            if web_session.client_id is not None:
                if client_id is not None and client_id != web_session.client_id:
                    raise service_error(
                        "forbidden", "Browser Client identity is invalid.", status=403
                    )
                client_id = web_session.client_id
        if client_required:
            if not client_id:
                raise service_error(
                    "unauthenticated", "Client authentication is required.", status=401
                )
            client = self.service.client(client_id)
            if web_session is not None:
                control = client.web_control_credential
                offered = tuple(
                    protocol.strip()
                    for protocol in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
                )
                supplied = (
                    control in offered
                    if websocket
                    else request.headers.get(_WEB_CONTROL_HEADER) == control
                )
                if control is None or not supplied:
                    raise service_error(
                        "forbidden",
                        "This Web Client does not own the active control connection.",
                        status=403,
                    )
        return _RequestContext(request, token, client_id, web_session)

    @staticmethod
    def _check_host_origin(
        request: web.Request,
        *,
        websocket: bool,
        require_origin: bool = False,
    ) -> None:
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
            if websocket or require_origin:
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


def _session_page_limit(request: web.Request) -> int | None:
    raw = request.query.get("limit")
    if raw is None:
        return None
    return _integer_query(request, "limit")


def _schedule_history_page_limit(request: web.Request) -> int | None:
    raw = request.query.get("limit")
    if raw is None:
        return None
    return _integer_query(request, "limit")


def _rename_fields(body: Mapping[str, object]) -> tuple[int, str, int]:
    claim_version = body.get("claim_version")
    if isinstance(claim_version, bool) or not isinstance(claim_version, int) or claim_version < 1:
        raise service_error("validation_error", "claim_version is invalid.", status=422)
    title = body.get("title")
    if not isinstance(title, str):
        raise service_error("validation_error", "title is required.", status=422)
    version_key = (
        "expected_metadata_version" if "expected_metadata_version" in body else "metadata_version"
    )
    metadata_version = body.get(version_key)
    if (
        isinstance(metadata_version, bool)
        or not isinstance(metadata_version, int)
        or metadata_version < 0
    ):
        raise service_error(
            "validation_error",
            "expected metadata version is invalid.",
            status=422,
        )
    return claim_version, title, metadata_version


def _delete_fields(body: Mapping[str, object]) -> tuple[int, bool]:
    claim_version = body.get("claim_version")
    if isinstance(claim_version, bool) or not isinstance(claim_version, int) or claim_version < 1:
        raise service_error("validation_error", "claim_version is invalid.", status=422)
    if body.get("confirm") is not True:
        raise service_error(
            "validation_error",
            "Session deletion requires explicit confirmation.",
            status=422,
            field_errors={"confirm": "must be true"},
        )
    return claim_version, True


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


def _safe_asset_path(value: str) -> bool:
    if not value or "\x00" in value or "\\" in value or ":" in value:
        return False
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _read_web_asset(asset_path: str) -> bytes:
    if not _safe_asset_path(asset_path):
        raise FileNotFoundError(asset_path)
    package = resources.files("myclaw.web_assets")
    asset = package.joinpath(*asset_path.split("/"))
    if isinstance(package, Path) and isinstance(asset, Path):
        if not asset.resolve().is_relative_to(package.resolve()):
            raise FileNotFoundError(asset_path)
    if not asset.is_file():
        raise FileNotFoundError(asset_path)
    return asset.read_bytes()


__all__ = ["LocalServiceTransport", "create_app"]
