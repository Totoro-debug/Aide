from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import BaseTestServer, TestServer

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.service.discovery import create_credential
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService
from aide.service.transport import create_app
from tests.fixtures.model_configuration import TEST_MODEL_PARAMETERS

REPAIRABLE_CONFIG = b"""[models.providers.old]
protocol = \"openai-compatible\"
base_url = \"https://old.example/v1\"
api_key = \"malformed-secret-303\"
models = [\"old-model\"]

[models.routes.default]
provider_id = \"old\"
model = \"old-model\"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30

[broken
value = true
"""


async def _wait_for_application(
    http: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    expected: str,
) -> dict[str, Any]:
    async with asyncio.timeout(5):
        while True:
            response = await http.get(url, headers=headers)
            body = cast(dict[str, Any], await response.json())
            if body["application"]["status"] == expected:
                return body
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def repair_http(
    tmp_path: Path,
) -> AsyncIterator[tuple[AgentService, BaseTestServer, dict[str, str]]]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    service = AgentService(home, None, reconnect_timeout=3600)
    await service.start()
    token = create_credential(home)
    client = await service.register_client("cli")
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-Client": client.client_id,
    }
    async with TestServer(create_app(service), host="127.0.0.1") as server:
        yield service, server, headers
    await service.stop()


def _usable_repair_fields(fields: dict[str, Any]) -> dict[str, Any]:
    models = fields["models"]
    providers = models["providers"]
    provider = providers["openai-local"]
    provider.pop("api_key", None)
    provider["base_url"] = "https://models.example/v1"
    provider["models"] = {"small-model": dict(TEST_MODEL_PARAMETERS)}
    for route in models["routes"].values():
        route["provider_id"] = "openai-local"
        route["model"] = "small-model"
    return fields


@pytest.mark.asyncio
async def test_missing_configuration_keeps_service_online_but_blocks_runtime(
    repair_http: tuple[AgentService, BaseTestServer, dict[str, str]],
    tmp_path: Path,
) -> None:
    service, server, headers = repair_http
    project = tmp_path / "project"
    project.mkdir()
    service.projects.register(project)

    async with aiohttp.ClientSession() as http:
        response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        body = await response.json()

    assert response.status == 200
    assert body["configuration"]["state"] == "missing"
    assert body["application"]["status"] == "pending-repair"
    assert not (service.agent_home.path / "config.toml").exists()
    assert service._workspaces == {}
    with pytest.raises(ServiceError) as blocked:
        await service.attach_workspace(headers["X-Aide-Client"], project)
    assert blocked.value.code == "config_invalid"
    async with aiohttp.ClientSession() as http:
        listed = await http.get(server.make_url("/api/v1/projects"), headers=headers)
        projects = await listed.json()
    assert listed.status == 200
    assert projects["projects"][0]["saved_jobs"] == []
    assert not (project / ".aide").exists()


@pytest.mark.asyncio
async def test_malformed_configuration_is_repaired_after_exact_private_backup(
    repair_http: tuple[AgentService, BaseTestServer, dict[str, str]],
) -> None:
    service, server, headers = repair_http
    config_path = service.agent_home.path / "config.toml"
    config_path.write_bytes(REPAIRABLE_CONFIG)
    instance_id = service.service_instance_id

    async with aiohttp.ClientSession() as http:
        current_response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        current = await current_response.json()
        assert current_response.status == 200
        assert current["configuration"]["state"] == "malformed"
        assert "malformed-secret-303" not in str(current)

        repair_headers = {
            **headers,
            "X-Aide-CSRF": headers["Authorization"].removeprefix("Bearer "),
        }
        fields = _usable_repair_fields(cast(dict[str, Any], current["fields"]))
        repair_response = await http.post(
            server.make_url("/api/v1/config/repair"),
            headers=repair_headers,
            json={
                "request_id": "repair-malformed-303",
                "revision": current["revision"],
                "fields": fields,
                "secrets": {
                    "models.providers.openai-local.api_key": {
                        "action": "replace",
                        "value": "repaired-secret-303",
                    }
                },
            },
        )
        repaired = await repair_response.json()
        active = await _wait_for_application(
            http,
            str(server.make_url("/api/v1/config")),
            headers,
            "next-run-required",
        )

    assert repair_response.status == 200
    assert repaired["backup_id"].startswith("sha256:")
    assert active["configuration"]["state"] == "active"
    assert active["application"]["active_revision"] is None
    assert service.configuration_ready
    assert service.service_instance_id == instance_id
    assert config_path.read_bytes() != REPAIRABLE_CONFIG
    backups = tuple(service.agent_home.path.glob("config.toml.backup.*"))
    assert backups
    assert any(backup.read_bytes() == REPAIRABLE_CONFIG for backup in backups)
    assert "repaired-secret-303" in config_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_malformed_repair_backup_failure_leaves_original_bytes_untouched(
    repair_http: tuple[AgentService, BaseTestServer, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, server, headers = repair_http
    config_path = service.agent_home.path / "config.toml"
    config_path.write_bytes(REPAIRABLE_CONFIG)
    before = config_path.read_bytes()

    def fail_backup(*_args: object, **_kwargs: object) -> object:
        raise OSError("injected backup failure")

    monkeypatch.setattr(
        "aide.config.config._create_private_backup",
        fail_backup,
    )

    async with aiohttp.ClientSession() as http:
        current = await (await http.get(server.make_url("/api/v1/config"), headers=headers)).json()
        repair_headers = {
            **headers,
            "X-Aide-CSRF": headers["Authorization"].removeprefix("Bearer "),
        }
        fields = _usable_repair_fields(cast(dict[str, Any], current["fields"]))
        response = await http.post(
            server.make_url("/api/v1/config/repair"),
            headers=repair_headers,
            json={
                "request_id": "repair-backup-failure-303",
                "revision": current["revision"],
                "fields": fields,
                "secrets": {
                    "models.providers.openai-local.api_key": {
                        "action": "replace",
                        "value": "repaired-secret-303",
                    }
                },
            },
        )
        body = await response.json()

    assert response.status == 500
    assert body["code"] == "persistence_error"
    assert config_path.read_bytes() == before
    assert service.configuration is None


@pytest.mark.asyncio
async def test_repair_http_security_validation_cas_and_secret_safe_replay(
    repair_http: tuple[AgentService, BaseTestServer, dict[str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, server, headers = repair_http
    config_path = service.agent_home.path / "config.toml"
    config_path.write_bytes(REPAIRABLE_CONFIG)
    mutation_headers = {
        **headers,
        "X-Aide-CSRF": headers["Authorization"].removeprefix("Bearer "),
    }
    async with aiohttp.ClientSession() as http:
        current = await (await http.get(server.make_url("/api/v1/config"), headers=headers)).json()
        payload: dict[str, Any] = {
            "request_id": "repair-replay",
            "revision": current["revision"],
            "fields": _usable_repair_fields(current["fields"]),
            "secrets": {
                "models.providers.openai-local.api_key": {
                    "action": "replace",
                    "value": "http-repair-secret-303",
                }
            },
        }
        url = server.make_url("/api/v1/config/repair")
        for request_headers in [
            {},
            headers,
            {key: value for key, value in mutation_headers.items() if key != "X-Aide-Client"},
            {**mutation_headers, "Origin": "https://outside.example"},
        ]:
            rejected = await http.post(url, headers=request_headers, json=payload)
            assert rejected.status in {401, 403}
            assert config_path.read_bytes() == REPAIRABLE_CONFIG
        launcher = await service.register_client("cli")
        ticket_response = await http.post(
            server.make_url("/api/v1/web/ticket"),
            headers={**mutation_headers, "X-Aide-Client": launcher.client_id},
            json={"request_id": "launch-for-repair-auth"},
        )
        ticket = (await ticket_response.json())["ticket"]
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as browser:
            exchanged = await browser.post(
                server.make_url("/api/v1/web/ticket"),
                headers={"Origin": str(server.make_url("/")).rstrip("/")},
                json={"ticket": ticket},
            )
            csrf = (await exchanged.json())["csrf_token"]
            registered = await browser.post(
                server.make_url("/api/v1/clients"),
                headers={"X-Aide-CSRF": csrf},
                json={"request_id": "browser-repair-auth", "kind": "web"},
            )
            control = (await registered.json())["web_control_credential"]
            for browser_headers in [
                {"X-Aide-Control": control},
                {"X-Aide-CSRF": csrf, "X-Aide-Control": "wrong"},
            ]:
                blocked = await browser.post(url, headers=browser_headers, json=payload)
                assert blocked.status == 403
                assert config_path.read_bytes() == REPAIRABLE_CONFIG
        unknown = await http.post(url, headers=mutation_headers, json={**payload, "unknown": True})
        assert unknown.status == 422
        invalid_payload = deepcopy(payload)
        invalid_payload["fields"]["runtime"]["max_iterations"] = 0
        invalid = await http.post(url, headers=mutation_headers, json=invalid_payload)
        assert invalid.status == 422
        assert config_path.read_bytes() == REPAIRABLE_CONFIG
        assert not tuple(service.agent_home.path.glob("config.toml.backup.*"))
        stale = await http.post(
            url, headers=mutation_headers, json={**payload, "revision": "stale"}
        )
        assert stale.status == 409
        assert config_path.read_bytes() == REPAIRABLE_CONFIG
        first = await http.post(url, headers=mutation_headers, json=payload)
        assert first.status == 200
        first_body = await first.json()
        saved = config_path.read_bytes()
        replay = await http.post(url, headers=mutation_headers, json=payload)
        assert replay.status == 200
        assert await replay.json() == first_body
        reuse = await http.post(url, headers=mutation_headers, json={**payload, "fields": {}})
        assert reuse.status == 409
        other = await service.register_client("cli")
        other_headers = {
            **mutation_headers,
            "X-Aide-Client": other.client_id,
            "X-Aide-Control": other.web_control_credential or "",
        }
        cross_client = await http.post(url, headers=other_headers, json=payload)
        assert cross_client.status == 409
        assert config_path.read_bytes() == saved
        public = await (await http.get(server.make_url("/api/v1/config"), headers=headers)).json()
        combined = str([first_body, public, await reuse.json(), await cross_client.json()])
        assert "http-repair-secret-303" not in combined + caplog.text
        assert "malformed-secret-303" not in combined + caplog.text
        assert ConfigLoader(service.agent_home).web_snapshot().state == "active"
