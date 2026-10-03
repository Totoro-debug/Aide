from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import BaseTestServer, TestServer

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.service.discovery import create_credential
from omni.service.runtime import LocalService
from omni.service.transport import create_app
from tests.configuration.test_config import MINIMAL_VALID_CONFIG

ConfigHttp = tuple[LocalService, BaseTestServer, str, str]

FULL_CONFIG = """[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "transport-provider-secret-302"
models = ["small-model"]

[models.providers.retired]
protocol = "anthropic"
base_url = "https://anthropic.example"
api_key = "transport-retired-secret-302"
models = ["retired-model"]

[models.routes.default]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30

[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example/tools"
headers = { Authorization = "transport-header-secret-302" }
connect_timeout = 30
call_timeout = 60
"""


@pytest_asyncio.fixture
async def config_http(tmp_path: Path) -> AsyncIterator[ConfigHttp]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load_for_startup()
    service = LocalService(home, configuration, reconnect_timeout=3600)
    await service.start()
    create_credential(home)
    web_client = await service.register_client("web")
    assert web_client.web_control_credential is not None
    async with TestServer(create_app(service), host="127.0.0.1") as server:
        yield service, server, web_client.client_id, web_client.web_control_credential
    await service.stop()


@pytest.mark.asyncio
async def test_config_get_returns_safe_structured_fields(config_http: ConfigHttp) -> None:
    _service, server, client_id, control = config_http
    assert control is not None
    token = create_credential(_service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-Client": client_id,
        "X-MyClaw-Control": control,
    }

    async with aiohttp.ClientSession() as http:
        response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        body = await response.json()

    assert response.status == 200
    assert body["fields"]["runtime"]["max_iterations"] == 50
    assert body["fields"]["models"]["providers"]["primary"]["api_key"] == {"configured": True}
    assert "minimal-secret" not in str(body)
    assert body["application"]["status"] == "active"


@pytest.mark.asyncio
async def test_config_patch_rejects_invalid_values_without_writing(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    assert control is not None
    token = create_credential(service.agent_home)
    before = (service.agent_home.path / "config.toml").read_bytes()
    current = service.config_view()
    revision = cast(str, current["revision"])
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-CSRF": token,
        "X-MyClaw-Client": client_id,
        "X-MyClaw-Control": control,
    }

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "invalid-config-edit",
                "revision": revision,
                "fields": {"runtime": {"max_iterations": 1}},
                "secrets": {},
            },
        )

    assert response.status == 422
    assert (service.agent_home.path / "config.toml").read_bytes() == before


@pytest.mark.asyncio
async def test_config_patch_reports_pending_then_active_and_stale_conflict(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    assert control is not None
    token = create_credential(service.agent_home)
    current = service.config_view()
    revision = cast(str, current["revision"])
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-CSRF": token,
        "X-MyClaw-Client": client_id,
        "X-MyClaw-Control": control,
    }
    payload = {
        "request_id": "valid-config-edit",
        "revision": revision,
        "fields": {"runtime": {"max_iterations": 80}},
        "secrets": {},
    }

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"), headers=headers, json=payload
        )
        saved = await response.json()
        await asyncio.sleep(0)
        active_response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        active = await active_response.json()
        conflict_response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={**payload, "request_id": "stale-config-edit"},
        )
        conflict = await conflict_response.json()

    assert response.status == 200
    assert saved["fields"]["runtime"]["max_iterations"] == 80
    assert saved["application"]["status"] in {"pending", "active"}
    assert active["application"]["status"] == "active"
    assert active["fields"]["runtime"]["max_iterations"] == 80
    assert conflict_response.status == 409
    assert conflict["code"] == "config_revision_conflict"


@pytest_asyncio.fixture
async def live_service(tmp_path: Path) -> AsyncIterator[tuple[LocalService, Path]]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    yield service, tmp_path / "workspace"
    await service.stop()


@pytest.mark.asyncio
async def test_config_application_replaces_workspace_generation(
    live_service: tuple[LocalService, Path],
) -> None:
    service, workspace_path = live_service
    workspace_path.mkdir()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, workspace_path)
    session_id = await workspace.create_draft(client.client_id)
    claim = await workspace.claim(client.client_id, session_id)
    schedule_state = await workspace._get_schedule_loop("config-test-job")
    old_schedule_loop = schedule_state.loop
    schedule_session_id = old_schedule_loop.session.session_id
    assert schedule_session_id in workspace.loops
    old_runtime = workspace.runtime
    old_state = workspace.workspace_state
    current = service.config_view()
    revision = cast(str, current["revision"])

    await service.update_configuration(
        "workspace-config-edit",
        revision,
        {"runtime": {"max_iterations": 81}, "memory": {"batch_size": 11}},
    )
    for _ in range(100):
        application = cast(dict[str, object], service.config_view()["application"])
        if application["status"] == "active":
            break
        await asyncio.sleep(0.01)

    application = cast(dict[str, object], service.config_view()["application"])
    assert application["status"] == "active"
    assert workspace.runtime is not old_runtime
    assert workspace.workspace_state is old_state
    assert workspace.configuration.runtime.max_iterations == 81
    assert workspace.configuration.memory.batch_size == 11
    assert claim.loop.session.session_id == session_id
    assert workspace.loops[session_id].loop is claim.loop
    assert workspace._schedule_loops["config-test-job"].loop is not old_schedule_loop
    assert (
        workspace._schedule_loops["config-test-job"].loop.session.session_id == schedule_session_id
    )
    assert (
        workspace._schedule_loops["config-test-job"].loop._configuration.runtime.max_iterations
        == 81
    )


@pytest_asyncio.fixture
async def full_config_http(tmp_path: Path) -> AsyncIterator[ConfigHttp]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(FULL_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load_for_startup()
    service = LocalService(home, configuration, reconnect_timeout=3600)
    await service.start()
    create_credential(home)
    web_client = await service.register_client("web")
    assert web_client.web_control_credential is not None
    async with TestServer(create_app(service), host="127.0.0.1") as server:
        yield service, server, web_client.client_id, web_client.web_control_credential
    await service.stop()


@pytest.mark.asyncio
async def test_config_patch_edits_models_routes_mcp_and_write_only_secrets(
    full_config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = full_config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-CSRF": token,
        "X-MyClaw-Client": client_id,
        "X-MyClaw-Control": control,
    }
    observed: list[str] = []

    async with aiohttp.ClientSession() as http:
        current_response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        current = await current_response.json()
        observed.append(str(current))
        assert current_response.status == 200
        assert current["fields"]["models"]["providers"]["primary"]["api_key"] == {
            "configured": True
        }
        assert current["fields"]["mcp"]["http"]["headers"] == {
            "Authorization": {"configured": True}
        }
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "transport-model-mcp-edit",
                "revision": current["revision"],
                "fields": {
                    "models": {
                        "providers": {
                            "primary": {"base_url": "https://models.example/v2"},
                            "retired": {},
                        },
                        "routes": {"default": {"model": "small-model"}},
                    },
                    "mcp": {"http": {"url": "https://mcp.example/replaced"}},
                },
                "secrets": {
                    "models.providers.primary.api_key": {
                        "action": "replace",
                        "value": "transport-provider-replaced-302",
                    },
                    "mcp.http.headers.Authorization": {
                        "action": "replace",
                        "value": "transport-header-replaced-302",
                    },
                    "models.providers.retired.api_key": {"action": "keep"},
                },
            },
        )
        saved = await response.json()
        observed.append(str(saved))
        current_response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        current = await current_response.json()
        observed.append(str(current))

    assert response.status == 200
    assert current["fields"]["models"]["providers"]["primary"]["base_url"] == (
        "https://models.example/v2"
    )
    assert current["fields"]["mcp"]["http"]["url"] == "https://mcp.example/replaced"
    assert all(
        secret not in "".join(observed)
        for secret in (
            "transport-provider-secret-302",
            "transport-provider-replaced-302",
            "transport-header-secret-302",
            "transport-header-replaced-302",
            "transport-retired-secret-302",
        )
    )
    saved_text = (service.agent_home.path / "config.toml").read_text(encoding="utf-8")
    assert "transport-provider-replaced-302" in saved_text
    assert "transport-header-replaced-302" in saved_text


@pytest.mark.asyncio
async def test_config_patch_secret_clear_is_explicit_and_preserves_bytes_on_conflict(
    full_config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = full_config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-CSRF": token,
        "X-MyClaw-Client": client_id,
        "X-MyClaw-Control": control,
    }
    async with aiohttp.ClientSession() as http:
        current = await (await http.get(server.make_url("/api/v1/config"), headers=headers)).json()
        clear = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "transport-secret-clear",
                "revision": current["revision"],
                "fields": {},
                "secrets": {
                    "models.providers.retired.api_key": {"action": "clear"},
                    "mcp.http.headers.Authorization": {"action": "clear"},
                },
            },
        )
        cleared = await clear.json()
        before_conflict = (service.agent_home.path / "config.toml").read_bytes()
        conflict = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "transport-stale-secret",
                "revision": current["revision"],
                "fields": {},
                "secrets": {
                    "models.providers.retired.api_key": {"action": "replace", "value": "stale"}
                },
            },
        )
        conflict_body = await conflict.json()

    assert clear.status == 200
    assert cleared["fields"]["models"]["providers"]["retired"]["api_key"] == {"configured": False}
    assert cleared["fields"]["mcp"]["http"]["headers"] == {}
    assert conflict.status == 409
    assert conflict_body["code"] == "config_revision_conflict"
    assert (service.agent_home.path / "config.toml").read_bytes() == before_conflict
