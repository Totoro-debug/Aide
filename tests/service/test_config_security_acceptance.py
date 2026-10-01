from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import BaseTestServer, TestServer
from jsonschema import Draft202012Validator
from loguru import logger

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.service.discovery import create_credential
from myclaw.service.runtime import LocalService
from myclaw.service.transport import create_app
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import MINIMAL_VALID_CONFIG

SecurityHttp = tuple[LocalService, BaseTestServer, dict[str, str], dict[str, str]]


@pytest_asyncio.fixture
async def security_http(tmp_path: Path) -> AsyncIterator[SecurityHttp]:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    token = create_credential(home)
    headers = []
    for _ in range(2):
        client = await service.register_client("web")
        assert client.web_control_credential is not None
        headers.append(
            {
                "Authorization": f"Bearer {token}",
                "X-MyClaw-CSRF": token,
                "X-MyClaw-Client": client.client_id,
                "X-MyClaw-Control": client.web_control_credential,
            }
        )
    async with TestServer(create_app(service), host="127.0.0.1") as server:
        yield service, server, headers[0], headers[1]
    await service.stop()


def _patch(service: LocalService, request_id: str = "security-save") -> dict[str, object]:
    return {
        "request_id": request_id,
        "revision": service.config_view()["revision"],
        "fields": {"runtime": {"max_iterations": 87}},
    }


def _assert_schema(response: dict[str, object]) -> None:
    path = Path(__file__).resolve().parents[2] / "myclaw/service/protocol/v1.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    name = "config_mutation_response" if "request_id" in response else "config_response"
    Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]}).validate(response)


@pytest.mark.asyncio
async def test_config_http_auth_csrf_client_and_unknown_fields_preserve_bytes(
    security_http: SecurityHttp,
) -> None:
    service, server, headers, _ = security_http
    before = (service.agent_home.path / "config.toml").read_bytes()
    payload = _patch(service)
    async with aiohttp.ClientSession() as http:
        cases = [
            ({}, payload),
            ({key: value for key, value in headers.items() if key != "X-MyClaw-CSRF"}, payload),
            ({**headers, "X-MyClaw-Client": "missing-client"}, payload),
            (headers, {**payload, "unknown": "canary"}),
            (headers, {**payload, "fields": {"providers": {"api_key": "canary"}}}),
            (headers, {**payload, "fields": {"runtime": {"unknown": 1}}}),
        ]
        for request_headers, body in cases:
            response = await http.patch(
                server.make_url("/api/v1/config"), headers=request_headers, json=body
            )
            assert response.status in {401, 403, 422}
            assert (service.agent_home.path / "config.toml").read_bytes() == before
        no_client = await http.get(
            server.make_url("/api/v1/config"), headers={"Authorization": headers["Authorization"]}
        )
        assert no_client.status in {401, 422}


@pytest.mark.asyncio
async def test_browser_cookie_config_requires_current_control_and_csrf(
    security_http: SecurityHttp,
) -> None:
    service, server, bearer_headers, _ = security_http
    cli = await service.register_client("cli")
    origin = str(server.make_url("/")).rstrip("/")
    before = (service.agent_home.path / "config.toml").read_bytes()
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as http:
        ticket_response = await http.post(
            server.make_url("/api/v1/web/ticket"),
            headers={**bearer_headers, "X-MyClaw-Client": cli.client_id},
            json={"request_id": "browser-ticket"},
        )
        assert ticket_response.status == 200
        ticket = await ticket_response.json()
        exchanged_response = await http.post(
            server.make_url("/api/v1/web/ticket"),
            headers={"Origin": origin},
            json={"ticket": ticket["ticket"]},
        )
        assert exchanged_response.status == 200
        exchanged = await exchanged_response.json()
        browser_headers = {"Origin": origin, "X-MyClaw-CSRF": exchanged["csrf_token"]}
        registered_response = await http.post(
            server.make_url("/api/v1/clients"),
            headers=browser_headers,
            json={"request_id": "browser-register", "kind": "web"},
        )
        assert registered_response.status == 200
        registered = await registered_response.json()
        owned = {**browser_headers, "X-MyClaw-Control": registered["web_control_credential"]}
        for headers in (
            browser_headers,
            {**owned, "X-MyClaw-Control": "wrong-control"},
            {**owned, "X-MyClaw-Client": cli.client_id},
            {key: value for key, value in owned.items() if key != "X-MyClaw-CSRF"},
            {**owned, "Origin": "http://example.invalid"},
        ):
            response = await http.patch(
                server.make_url("/api/v1/config"),
                headers=headers,
                json=_patch(service, "browser-save"),
            )
            assert response.status == 403
            assert (service.agent_home.path / "config.toml").read_bytes() == before
        accepted = await http.patch(
            server.make_url("/api/v1/config"), headers=owned, json=_patch(service, "browser-save")
        )
        assert accepted.status == 200
        _assert_schema(await accepted.json())


@pytest.mark.asyncio
async def test_config_http_idempotency_body_action_and_client_reuse(
    security_http: SecurityHttp,
) -> None:
    service, server, first, second = security_http
    payload = _patch(service)
    async with aiohttp.ClientSession() as http:
        saved_response = await http.patch(
            server.make_url("/api/v1/config"), headers=first, json=payload
        )
        saved = await saved_response.json()
        assert saved_response.status == 200
        _assert_schema(saved)
        before = (service.agent_home.path / "config.toml").read_bytes()
        replay_response = await http.patch(
            server.make_url("/api/v1/config"), headers=first, json=payload
        )
        assert replay_response.status == 200
        assert await replay_response.json() == saved
        changed_body = await http.patch(
            server.make_url("/api/v1/config"),
            headers=first,
            json={**payload, "fields": {"memory": {"batch_size": 18}}},
        )
        other_client = await http.patch(
            server.make_url("/api/v1/config"), headers=second, json=payload
        )
        other_action = await http.post(
            server.make_url("/api/v1/config/retry"),
            headers=first,
            json={"request_id": payload["request_id"], "revision": saved["revision"]},
        )
        assert changed_body.status == other_client.status == other_action.status == 409
        assert (service.agent_home.path / "config.toml").read_bytes() == before
        current_response = await http.get(server.make_url("/api/v1/config"), headers=first)
        assert current_response.status == 200
        _assert_schema(await current_response.json())


@pytest.mark.asyncio
async def test_concurrent_http_clients_same_revision_only_one_save(
    security_http: SecurityHttp,
) -> None:
    service, server, first, second = security_http
    payload = _patch(service)
    async with aiohttp.ClientSession() as http:
        responses = await asyncio.gather(
            http.patch(server.make_url("/api/v1/config"), headers=first, json=payload),
            http.patch(
                server.make_url("/api/v1/config"),
                headers=second,
                json={
                    **payload,
                    "request_id": "second-save",
                    "fields": {"memory": {"batch_size": 19}},
                },
            ),
        )
        assert sorted(response.status for response in responses) == [200, 409]
        for response in responses:
            body = await response.json()
            assert "minimal-secret" not in json.dumps(body)
            if response.status == 200:
                _assert_schema(body)


@pytest.mark.asyncio
async def test_http_write_failure_keeps_saved_active_status_and_bytes(
    security_http: SecurityHttp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, server, headers, _ = security_http
    before = (service.agent_home.path / "config.toml").read_bytes()
    status = service.config_view()

    def fail(target: Path, content: str) -> None:
        raise OSError("write error with minimal-secret")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail)
    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"), headers=headers, json=_patch(service)
        )
        body = await response.json()
    assert response.status == 500
    assert "minimal-secret" not in json.dumps(body)
    assert (service.agent_home.path / "config.toml").read_bytes() == before
    assert service.config_view() == status
    assert service._config_apply_task is None


@pytest.mark.asyncio
async def test_http_invalid_complete_candidate_and_field_errors_do_not_leak_secrets(
    security_http: SecurityHttp,
) -> None:
    service, server, headers, _ = security_http
    async with aiohttp.ClientSession() as http:
        invalid_field = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={**_patch(service), "fields": {"memory": {"batch_size": 0}}},
        )
        field_error = await invalid_field.json()
        assert invalid_field.status == 422
        assert "memory.batch_size" in cast(dict[str, str], field_error["field_errors"])
        path = service.agent_home.path / "config.toml"
        text = path.read_text(encoding="utf-8").replace(
            "context_window = 8192", "context_window = 1"
        )
        assert text != path.read_text(encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        before = path.read_bytes()
        body = {
            "request_id": "invalid-candidate",
            "revision": ConfigLoader(service.agent_home).revision(),
            "fields": {"memory": {"batch_size": 20}},
        }
        response = await http.patch(server.make_url("/api/v1/config"), headers=headers, json=body)
        result = await response.json()
        assert response.status == 422
        assert "minimal-secret" not in json.dumps(result)
        assert "Traceback" not in json.dumps(result)
        assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_http_strict_save_rejects_untouched_invalid_mcp_without_secret_leak(
    security_http: SecurityHttp,
) -> None:
    service, server, headers, _ = security_http
    secret = "strict-mcp-secret-canary-301"
    path = service.agent_home.path / "config.toml"
    path.write_text(
        path.read_text(encoding="utf-8")
        + f'\n[mcp.servers.invalid]\nenabled = true\ntransport = "unsupported"\ncommand = "{secret}"\n',
        encoding="utf-8",
    )
    before = path.read_bytes()
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        async with aiohttp.ClientSession() as http:
            response = await http.patch(
                server.make_url("/api/v1/config"),
                headers=headers,
                json={
                    "request_id": "strict-invalid-mcp",
                    "revision": ConfigLoader(service.agent_home).revision(),
                    "fields": {"memory": {"batch_size": 23}},
                },
            )
            body = await response.json()
        assert response.status == 422
        assert path.read_bytes() == before
        assert secret not in json.dumps(body) + "".join(messages)
        assert "minimal-secret" not in json.dumps(body) + "".join(messages)
        assert "Traceback" not in json.dumps(body)
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
async def test_config_http_ws_errors_and_logs_never_expose_existing_secrets(
    security_http: SecurityHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, server, headers, _ = security_http
    secrets = ("minimal-secret", "mcp-header-canary-301", "mcp-env-canary-301")
    monkeypatch.setenv("MC301_MCP_SECRET", secrets[2])
    path = service.agent_home.path / "config.toml"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[mcp.servers.http]\nenabled = false\ntransport = "streamable-http"\n'
        + 'url = "http://127.0.0.1:1/mcp"\n'
        + f'headers = {{ Authorization = "{secrets[1]}" }}\n'
        + '\n[mcp.servers.stdio]\nenabled = false\ntransport = "stdio"\n'
        + f'command = "python"\nargs = ["{secrets[2]}"]\n',
        encoding="utf-8",
    )
    observed: list[str] = []
    sink = logger.add(lambda message: observed.append(str(message)), format="{message}")
    try:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(
                server.make_url("/api/v1/events"),
                headers={**headers, "Origin": str(server.make_url("/")).rstrip("/")},
                protocols=("myclaw-v1",),
            ) as socket:
                await socket.send_json(
                    {
                        "request_id": "secret-canary-subscribe",
                        "type": "subscribe",
                        "payload": {},
                    }
                )
                while True:
                    event = await socket.receive_json(timeout=5)
                    observed.append(json.dumps(event))
                    if event.get("request_id") == "secret-canary-subscribe":
                        assert event.get("accepted") is True
                        break
                view = await http.get(server.make_url("/api/v1/config"), headers=headers)
                projection = await view.json()
                observed.append(json.dumps(projection))
                saved = await http.patch(
                    server.make_url("/api/v1/config"),
                    headers=headers,
                    json={
                        "request_id": "secret-canary-save",
                        "revision": projection["revision"],
                        "fields": {"memory": {"batch_size": 24}},
                    },
                )
                assert saved.status == 200
                observed.append(json.dumps(await saved.json()))
                while True:
                    event = await socket.receive_json(timeout=5)
                    observed.append(json.dumps(event))
                    if event.get("type") == "config.application":
                        break
                invalid = await http.patch(
                    server.make_url("/api/v1/config"),
                    headers=headers,
                    json={
                        "request_id": "secret-canary-invalid",
                        "revision": ConfigLoader(service.agent_home).revision(),
                        "fields": {"memory": {"batch_size": secrets[2]}},
                    },
                )
                assert invalid.status == 422
                observed.append(json.dumps(await invalid.json()))
        assert all(secret not in "".join(observed) for secret in secrets)
        assert "Traceback" not in "".join(observed)
        assert all(secret in path.read_text(encoding="utf-8") for secret in secrets)
    finally:
        logger.remove(sink)
