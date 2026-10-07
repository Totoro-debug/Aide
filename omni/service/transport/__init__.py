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
from importlib import resources
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from ...config.config import ConfigError
from ..directory_picker import DirectoryPicker
from ..discovery import identity_proof
from ..errors import ServiceError, service_error
from ..runtime import WEB_TICKET_TTL_SECONDS, AgentService, ServiceSink

_API_PREFIX = "/api/v1"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
_CSRF_HEADER = "X-Omni-CSRF"
_CLIENT_HEADER = "X-Omni-Client"
_WEB_CONTROL_HEADER = "X-Omni-Control"
_WEB_SESSION_COOKIE = "omni_session"
_WEB_TICKET_TTL_SECONDS = WEB_TICKET_TTL_SECONDS
_WEB_SESSION_MAX_AGE_SECONDS = 24 * 60 * 60
_STATIC_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self' ws:; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)


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


class AgentServiceTransport:
    """Bind HTTP, WebSocket, and safe static routes to one service instance."""

    def __init__(self, service: AgentService) -> None:
        self.service = service
        self._web_tickets: dict[str, float] = {}
        self._web_sessions: dict[str, _WebSession] = {}
        self._web_auth_lock = asyncio.Lock()
        self._directory_picker = DirectoryPicker()

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
        app["omni.service"] = self.service
        app.on_shutdown.append(self._close_directory_picker)
        app.router.add_get("/", self._static_index)
        app.router.add_post(f"{_API_PREFIX}/web/ticket", self._web_ticket)
        app.router.add_get(f"{_API_PREFIX}/web/session", self._web_session_info)
        app.router.add_get(f"{_API_PREFIX}/service/identity", self._service_identity)
        app.router.add_get(f"{_API_PREFIX}/service", self._service_info)
        app.router.add_get(f"{_API_PREFIX}/config", self._config)
        app.router.add_get(f"{_API_PREFIX}/config/text", self._config_text)
        app.router.add_get(f"{_API_PREFIX}/config/startup", self._config_startup)
        app.router.add_get(f"{_API_PREFIX}/models/available", self._available_models)
        app.router.add_get(f"{_API_PREFIX}/input-capabilities", self._input_capabilities)
        app.router.add_patch(f"{_API_PREFIX}/config", self._patch_config)
        app.router.add_post(f"{_API_PREFIX}/config/repair", self._repair_config)
        app.router.add_post(f"{_API_PREFIX}/clients", self._register_client)
        app.router.add_post(f"{_API_PREFIX}/workspaces/attach", self._attach_workspace)
        app.router.add_post(
            f"{_API_PREFIX}/chat/workspaces/enter", self._enter_conversation_workspace
        )
        app.router.add_post(f"{_API_PREFIX}/conversations/open", self._open_conversation)
        app.router.add_get(f"{_API_PREFIX}/chat/sessions", self._list_chat_sessions)
        app.router.add_get(f"{_API_PREFIX}/projects", self._list_projects)
        app.router.add_post(f"{_API_PREFIX}/projects", self._register_project)
        app.router.add_post(f"{_API_PREFIX}/projects/directory-picker", self._pick_project_directory)
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
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}/claim",
            self._claim_session,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}/release",
            self._release_session,
        )
        app.router.add_get(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}/deletion-status",
            self._session_deletion_status,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/sessions/{{session_id}}/deletion-claim",
            self._claim_session_deletion,
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
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/memory/read",
            self._get_memory_view,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/memory/dream",
            self._run_dream,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/runtime/status",
            self._get_runtime_status,
        )
        app.router.add_post(
            f"{_API_PREFIX}/workspaces/{{workspace_id}}/skills/reload",
            self._reload_skill_catalog,
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
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = self.service.open_web_interface(client_id, request_id)
        ticket = result["ticket"]
        async with self._web_auth_lock:
            self._prune_web_tickets()
            self._web_tickets[ticket] = time.monotonic() + _WEB_TICKET_TTL_SECONDS
        return web.json_response(result)

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
        return web.json_response(self.service.get_service_status())

    async def _config(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.config_view())

    async def _config_text(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.configuration_text_view())

    async def _config_startup(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.configuration_startup_view())

    async def _available_models(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.available_models_view())

    async def _input_capabilities(self, request: web.Request) -> web.Response:
        self._authenticate(request, client_required=True)
        return web.json_response(self.service.get_input_capabilities())

    async def _patch_config(self, request: web.Request) -> web.Response:
        return await self._persist_config(request, "patch")

    async def _repair_config(self, request: web.Request) -> web.Response:
        return await self._persist_config(request, "repair")

    async def _persist_config(
        self, request: web.Request, action: Literal["patch", "repair"]
    ) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        required = {"request_id", "revision", "fields", "secrets"}
        optional = {
            "baseline",
            "baseline_secrets",
            "overwrite_conflicts",
            "editor_id",
            "edit_sequence",
        }
        if not required.issubset(body) or set(body) - required - optional:
            raise service_error(
                "validation_error",
                "Configuration request fields are invalid." if action == "patch"
                else "Configuration repair fields are invalid.",
                status=422
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
        baseline = body.get("baseline")
        if baseline is not None and not isinstance(baseline, Mapping):
            raise service_error(
                "validation_error",
                "Configuration baseline must be an object.",
                status=422,
                field_errors={"baseline": "must be an object"},
            )
        baseline_secrets = body.get("baseline_secrets")
        if baseline_secrets is not None and not isinstance(baseline_secrets, Mapping):
            raise service_error(
                "validation_error",
                "Configuration secret baseline must be an object.",
                status=422,
                field_errors={"baseline_secrets": "must be an object"},
            )
        overwrite_conflicts = body.get("overwrite_conflicts", False)
        if not isinstance(overwrite_conflicts, bool):
            raise service_error(
                "validation_error",
                "Configuration conflict resolution must be a boolean.",
                status=422,
                field_errors={"overwrite_conflicts": "must be a boolean"},
            )
        persist = (
            self.service.update_configuration if action == "patch"
            else self.service.repair_configuration
        )
        result = await persist(
            request_id,
            revision,
            fields,
            secrets,
            client_id=context.client_id,
            baseline=baseline,
            baseline_secrets=baseline_secrets,
            overwrite_conflicts=overwrite_conflicts,
            editor_id=cast(str | None, body.get("editor_id")),
            edit_sequence=cast(int | None, body.get("edit_sequence")),
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
            try:
                client = await self.service.register_client(kind, reconnect)
            except ServiceError as error:
                if error.code != "stale_client":
                    raise
                client = await self.service.register_client(kind)
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
        result = await self.service.enter_workspace(client_id, _path_from_text(path))
        return web.json_response({"request_id": request_id, **result})

    async def _list_projects(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.list_projects(_context_client_id(context))
        return web.json_response(result)

    async def _open_conversation(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        allowed = {
            "request_id",
            "project_id",
            "workspace_id",
            "directory",
            "session_id",
            "create_new",
        }
        if set(body) - allowed:
            raise service_error("validation_error", "Conversation fields are invalid.", status=422)
        request_id = _require_request_id(body)
        scope_values = ("project_id", "workspace_id", "directory", "session_id")
        for field in scope_values:
            value = body.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise service_error(
                    "validation_error",
                    f"{field} is invalid.",
                    status=422,
                    field_errors={field: "must be a nonempty string"},
                )
        create_new = body.get("create_new", False)
        if not isinstance(create_new, bool):
            raise service_error(
                "validation_error",
                "create_new must be a boolean.",
                status=422,
                field_errors={"create_new": "must be a boolean"},
            )
        result = await self.service.open_conversation(
            _context_client_id(context),
            request_id=request_id,
            project_id=cast(str | None, body.get("project_id")),
            workspace_id=cast(str | None, body.get("workspace_id")),
            directory=cast(str | None, body.get("directory")),
            session_id=cast(str | None, body.get("session_id")),
            create_new=create_new,
        )
        return web.json_response({"request_id": request_id, **result})

    async def _enter_conversation_workspace(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        if set(body) - {"request_id", "directory"}:
            raise service_error(
                "validation_error", "Conversation Workspace request fields are invalid.", status=422
            )
        request_id = _require_request_id(body)
        directory = body.get("directory")
        if directory is not None and (not isinstance(directory, str) or not directory):
            raise service_error(
                "validation_error",
                "Conversation Workspace directory must be a nonempty path.",
                status=422,
                field_errors={"directory": "must be a nonempty path"},
            )
        result = await self.service.enter_default_conversation_workspace(
            _context_client_id(context), directory=directory
        )
        return web.json_response({"request_id": request_id, **result})

    async def _list_chat_sessions(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        page = self.service.list_chat_sessions_page(
            _context_client_id(context),
            title=request.query.get("title"),
            cursor=request.query.get("cursor"),
            limit=_optional_page_limit(request),
        )
        return web.json_response(page)

    async def _close_directory_picker(self, app: web.Application) -> None:
        await self._directory_picker.close()

    async def _pick_project_directory(self, request: web.Request) -> web.Response:
        self._authenticate(request, mutation=True, client_required=True)
        request_id = _require_request_id(await _json_object(request))
        selection = asyncio.create_task(self._directory_picker.pick())
        try:
            while not selection.done():
                await asyncio.wait({selection}, timeout=0.2)
                if request.transport is None or request.transport.is_closing():
                    raise asyncio.CancelledError
                self._authenticate(request, mutation=True, client_required=True)
            path = await selection
            return web.json_response({"request_id": request_id, "path": path})
        finally:
            if not selection.done():
                selection.cancel()
            await asyncio.gather(selection, return_exceptions=True)

    async def _register_project(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise service_error("validation_error", "Project path is required.", status=422)
        result = await self.service.register_project_entry(client_id, _path_from_text(path))
        return web.json_response({"request_id": request_id, **result})

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
            limit=_optional_page_limit(request),
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
            _required_header(request, "X-Omni-Claim"),
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
            _required_header(request, "X-Omni-Claim"),
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
        await self.service.release_conversation(
            client_id,
            None,
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-Omni-Claim"),
            project_id=request.match_info["project_id"],
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
            _required_header(request, "X-Omni-Claim"),
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
            limit=_optional_page_limit(request),
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
        return web.json_response({"request_id": request_id, "project_id": None, **result})

    async def _claim_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        workspace_id = request.match_info["workspace_id"]
        result = await self.service.claim(
            client_id,
            workspace_id,
            request.match_info["session_id"],
        )
        return web.json_response(
            {"request_id": request_id, "project_id": None, "workspace_id": workspace_id, **result}
        )

    async def _release_session(self, request: web.Request) -> web.Response:
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
        await self.service.release_conversation(
            client_id,
            request.match_info["workspace_id"],
            request.match_info["session_id"],
            claim_version,
            _required_header(request, "X-Omni-Claim"),
        )
        return web.json_response({"request_id": request_id, "released": True})

    async def _session_deletion_status(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        result = await self.service.session_deletion_status(
            _context_client_id(context),
            request.match_info["workspace_id"],
            request.match_info["session_id"],
        )
        return web.json_response({"project_id": None, **result})

    async def _claim_session_deletion(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        client_id = _context_client_id(context)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        result = await self.service.claim_session_deletion(
            client_id,
            request.match_info["workspace_id"],
            request.match_info["session_id"],
        )
        return web.json_response({"request_id": request_id, "project_id": None, **result})

    async def _get_session(self, request: web.Request) -> web.Response:
        context = self._authenticate(request, client_required=True)
        client_id = _context_client_id(context)
        workspace_id = request.match_info["workspace_id"]
        session_id = request.match_info["session_id"]
        claim_version = _integer_query(request, "claim_version")
        claim_credential = _required_header(request, "X-Omni-Claim")
        return web.json_response(
            await self.service.get_session_snapshot(
                client_id, workspace_id, session_id, claim_version, claim_credential
            )
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
            _required_header(request, "X-Omni-Claim"),
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
            _required_header(request, "X-Omni-Claim"),
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
            limit=_optional_page_limit(request),
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

    async def _get_memory_view(self, request: web.Request) -> web.Response:
        return await self._call_session_operation(request, self.service.get_memory_view)

    async def _run_dream(self, request: web.Request) -> web.Response:
        return await self._call_session_operation(request, self.service.run_dream)

    async def _get_runtime_status(self, request: web.Request) -> web.Response:
        return await self._call_session_operation(request, self.service.get_runtime_status)

    async def _reload_skill_catalog(self, request: web.Request) -> web.Response:
        return await self._call_session_operation(request, self.service.reload_skill_catalog)

    async def _call_session_operation(
        self,
        request: web.Request,
        operation: Callable[
            [str, str, str, str, int, str], Awaitable[Mapping[str, object]]
        ],
    ) -> web.Response:
        context = self._authenticate(request, mutation=True, client_required=True)
        body = await _json_object(request)
        if set(body) != {"request_id", "current_session_id", "claim_version"}:
            raise service_error(
                "validation_error", "Management operation fields are invalid.", status=422
            )
        request_id = _require_request_id(body)
        session_id = body.get("current_session_id")
        claim_version = body.get("claim_version")
        request["omni.request_id"] = request_id
        claim_credential = request.headers.get("X-Omni-Claim")
        if not isinstance(session_id, str) or not session_id:
            raise service_error("validation_error", "Session ID is required.", status=422)
        if (
            isinstance(claim_version, bool)
            or not isinstance(claim_version, int)
            or claim_version < 1
            or not isinstance(claim_credential, str)
            or not claim_credential
        ):
            raise service_error(
                "stale_claim", "Conversation Session Claim is missing or stale.", retryable=True
            )
        result = await operation(
            _context_client_id(context),
            request.match_info["workspace_id"],
            session_id,
            request_id,
            claim_version,
            claim_credential,
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
                session_value = _required_header(request, "X-Omni-Session")
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
        action = request.match_info["action"]
        claim_credential = request.headers.get("X-Omni-Claim")
        restore_actions = {
            "restore/inspect",
            "restore/execute",
            "restore/result",
            "restore/cancel",
            "restore/acknowledge",
        }
        if action in restore_actions:
            if claim_version is None or claim_credential is None or session_value is None:
                raise service_error(
                    "stale_claim", "Conversation Session Claim is missing or stale.", retryable=True
                )
            workspace_id = request.match_info["workspace_id"]
            if action == "restore/inspect":
                anchor_id = body.get("anchor_id")
                if isinstance(anchor_id, bool) or not isinstance(anchor_id, int):
                    raise service_error(
                        "validation_error", "Restore anchor ID is invalid.", status=422
                    )
                result = await self.service.inspect_restore(
                    client_id,
                    workspace_id,
                    session_value,
                    claim_version,
                    claim_credential,
                    request_id,
                    anchor_id,
                )
            elif action == "restore/execute":
                wire_plan = body.get("plan")
                mode = body.get("mode")
                anchor_id = wire_plan.get("anchor_id") if isinstance(wire_plan, dict) else None
                if (
                    isinstance(anchor_id, bool)
                    or not isinstance(anchor_id, int)
                    or anchor_id < 1
                    or not isinstance(mode, str)
                    or not mode
                ):
                    raise service_error(
                        "validation_error", "Restore request is invalid.", status=422
                    )
                result = await self.service.commit_restore(
                    client_id,
                    workspace_id,
                    session_value,
                    claim_version,
                    claim_credential,
                    request_id,
                    anchor_id,
                    mode,
                )
            elif action == "restore/result":
                result = await self.service.get_restore_result(
                    client_id,
                    workspace_id,
                    session_value,
                    claim_version,
                    claim_credential,
                    request_id,
                )
            elif action == "restore/cancel":
                result = await self.service.cancel_restore(
                    client_id,
                    workspace_id,
                    session_value,
                    claim_version,
                    claim_credential,
                    request_id,
                )
            else:
                result = await self.service.acknowledge_restore_failure(
                    client_id,
                    workspace_id,
                    session_value,
                    claim_version,
                    claim_credential,
                    request_id,
                )
        else:
            result = await self.service.handle_management(
                client_id,
                request.match_info["workspace_id"],
                session_value,
                action,
                body,
                claim_version=claim_version,
                claim_credential=claim_credential,
            )
        return web.json_response({"request_id": request_id, "result": result})

    async def _stop_service(self, request: web.Request) -> web.Response:
        self._authenticate(request, mutation=True)
        body = await _json_object(request)
        request_id = _require_request_id(body)
        return web.json_response(self.service.request_service_stop(request_id))

    async def _events(self, request: web.Request) -> web.StreamResponse:
        context = self._authenticate(request, websocket=True, client_required=True)
        client_id = _context_client_id(context)
        if self.service.client(client_id).connected:
            raise service_error("client_already_connected", "This Client already has a connection.")
        socket = web.WebSocketResponse(heartbeat=20.0, autoping=True, protocols=("omni-v1",))
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


AgentServiceTransport = AgentServiceTransport


def create_app(service: AgentService) -> web.Application:
    """Return the single local application used by tests and the service process."""
    return AgentServiceTransport(service).create_app()


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


def _read_service_token(service: AgentService) -> str:
    from omni.service.discovery import read_credential

    return read_credential(service.agent_home)


def _bearer_token(request: web.Request) -> str | None:
    value = request.headers.get("Authorization", "")
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.casefold() != "bearer" or not token:
        return None
    return token


def _request_id_from_request(request: web.Request) -> str:
    request_id = request.get("omni.request_id")
    if isinstance(request_id, str):
        return request_id
    return request.headers.get("X-Omni-Request", "transport") or "transport"


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


def _optional_page_limit(request: web.Request) -> int | None:
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
    package = resources.files("omni.web_assets")
    asset = package.joinpath(*asset_path.split("/"))
    if isinstance(package, Path) and isinstance(asset, Path):
        if not asset.resolve().is_relative_to(package.resolve()):
            raise FileNotFoundError(asset_path)
    if not asset.is_file():
        raise FileNotFoundError(asset_path)
    return asset.read_bytes()


__all__ = ["AgentServiceTransport", "AgentServiceTransport", "create_app"]
