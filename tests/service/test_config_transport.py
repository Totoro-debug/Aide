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
from omni.service.runtime import AgentService
from omni.service.transport import create_app
from tests.configuration.test_config import MINIMAL_VALID_CONFIG

ConfigHttp = tuple[AgentService, BaseTestServer, str, str]

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
    service = AgentService(home, configuration, reconnect_timeout=3600)
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
        "X-Omni-Client": client_id,
        "X-Omni-Control": control,
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
        "X-Omni-CSRF": token,
        "X-Omni-Client": client_id,
        "X-Omni-Control": control,
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
async def test_config_patch_reports_restart_required_and_stale_conflict(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    assert control is not None
    token = create_credential(service.agent_home)
    current = service.config_view()
    revision = cast(str, current["revision"])
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": client_id,
        "X-Omni-Control": control,
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
    assert saved["application"]["status"] == "restart-required"
    assert saved["application"]["restart_required"] is True
    assert active["application"]["status"] == "restart-required"
    assert active["application"]["active_revision"] == revision
    assert service.configuration is not None
    assert service.configuration.runtime.max_iterations == 50
    assert active["fields"]["runtime"]["max_iterations"] == 80
    assert conflict_response.status == 409
    assert conflict["code"] == "config_revision_conflict"


@pytest_asyncio.fixture
async def live_service(tmp_path: Path) -> AsyncIterator[tuple[AgentService, Path]]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    yield service, tmp_path / "workspace"
    await service.stop()


@pytest.mark.asyncio
async def test_config_save_preserves_workspace_and_later_activation_uses_startup_settings(
    live_service: tuple[AgentService, Path],
) -> None:
    service, workspace_path = live_service
    workspace_path.mkdir()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, workspace_path)
    old_runtime = workspace.resources
    revision = cast(str, service.config_view()["revision"])
    saved = await service.update_configuration(
        "workspace-config-edit",
        revision,
        {"runtime": {"max_iterations": 81}, "memory": {"batch_size": 11}},
    )
    second_path = workspace_path.parent / "later-workspace"
    second_path.mkdir()
    later = await service.attach_workspace(client.client_id, second_path)
    assert saved["application"] == {
        "status": "restart-required",
        "saved_revision": saved["revision"],
        "active_revision": revision,
        "restart_required": True,
    }
    assert workspace.resources is old_runtime
    for owner in (workspace, later):
        assert owner.configuration.runtime.max_iterations == 50
        assert owner.configuration.memory.batch_size == 10
    await service.stop()
    restarted = AgentService(service.agent_home, reconnect_timeout=3600)
    try:
        await restarted.start()
        new_client = await restarted.register_client("cli")
        reopened = await restarted.attach_workspace(new_client.client_id, workspace_path)
        assert reopened.configuration.runtime.max_iterations == 81
        assert reopened.configuration.memory.batch_size == 11
        assert cast(dict[str, object], restarted.config_view()["application"])["status"] == "active"
    finally:
        await restarted.stop()


@pytest_asyncio.fixture
async def full_config_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ConfigHttp]:
    from omni.agent.tools.mcp_runtime import MCPRuntimeManager, MCPStartupReport

    async def start(_manager: MCPRuntimeManager, _configuration: object) -> MCPStartupReport:
        return MCPStartupReport((), ())

    monkeypatch.setattr(MCPRuntimeManager, "start", start)
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(FULL_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load_for_startup()
    service = AgentService(home, configuration, reconnect_timeout=3600)
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
        "X-Omni-CSRF": token,
        "X-Omni-Client": client_id,
        "X-Omni-Control": control,
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
        "X-Omni-CSRF": token,
        "X-Omni-Client": client_id,
        "X-Omni-Control": control,
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
