from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import BaseTestServer, TestServer

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader, ProviderConfiguration
from aide.service.discovery import create_credential
from aide.service.runtime import AgentService
from aide.service.transport import create_app
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.model_configuration import complete_model_settings

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
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }

    async with aiohttp.ClientSession() as http:
        response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        body = await response.json()

    assert response.status == 200
    assert body["fields"]["runtime"]["max_iterations"] == 50
    assert body["fields"]["models"]["providers"]["primary"]["api_key"] == {"configured": True}
    assert "minimal-secret" not in str(body)
    assert body["secret_revisions"]["models.providers.primary.api_key"]
    assert "minimal-secret" not in str(body["secret_revisions"])
    assert body["application"]["status"] == "active"


@pytest.mark.asyncio
async def test_available_models_exposes_active_capacity_and_default_without_secrets(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    assert control is not None
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }

    async with aiohttp.ClientSession() as http:
        initial_response = await http.get(
            server.make_url("/api/v1/models/available"), headers=headers
        )
        initial = await initial_response.json()
        saved = await service.update_configuration(
            "save-model-context-window",
            cast(str, service.config_view()["revision"]),
            {
                "models": {
                    "providers": {"primary": {"model_context_windows": {"small-model": 16384}}}
                }
            },
            client_id=client_id,
        )
        active_response = await http.get(
            server.make_url("/api/v1/models/available"), headers=headers
        )
        active = await active_response.json()

    assert initial_response.status == active_response.status == 200
    assert initial == {
        "models": [{"provider_id": "primary", "model": "small-model", "context_window": 8192}],
        "default_combination": {
            "provider_id": "primary",
            "model": "small-model",
            "reasoning_effort": "mid",
        },
    }
    assert "minimal-secret" not in str(initial)
    assert active == initial
    assert cast(dict[str, object], saved["application"])["restart_required"] is True
    assert service.configuration is not None
    assert service.configuration.resolve_route("chat").route.context_window == 8192


@pytest.mark.asyncio
async def test_available_models_excludes_unusable_providers(config_http: ConfigHttp) -> None:
    service, _server, _client_id, _control = config_http
    configuration = service.configuration
    assert configuration is not None
    provider = configuration.models.providers["primary"]
    unavailable = {
        "no-key": replace(provider, provider_id="no-key", api_key=" "),
        "bad-url": replace(provider, provider_id="bad-url", base_url="invalid"),
        "bad-protocol": replace(provider, provider_id="bad-protocol", protocol="unsupported"),
        "too-small": replace(provider, provider_id="too-small"),
    }
    known: dict[str, ProviderConfiguration] = {
        name: replace(
            value,
            model_context_windows={"small-model": 1024 if name == "too-small" else 16384},
        )
        for name, value in unavailable.items()
    }
    service.configuration = replace(
        configuration,
        models=replace(configuration.models, providers={"primary": provider, **known}),
    )

    assert service.available_models_view()["models"] == [
        {"provider_id": "primary", "model": "small-model", "context_window": 8192}
    ]

    current_configuration = service.configuration
    assert current_configuration is not None
    chat_route = replace(
        current_configuration.models.routes["default"],
        context_window=16_384,
        reasoning_effort="high",
    )
    service.configuration = replace(
        current_configuration,
        models=replace(
            current_configuration.models,
            routes={**current_configuration.models.routes, "chat": chat_route},
        ),
    )
    assert service.available_models_view()["default_combination"] == {
        "provider_id": "primary",
        "model": "small-model",
        "reasoning_effort": "high",
    }


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
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
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
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
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


@pytest.mark.asyncio
async def test_config_patch_merges_nonoverlapping_external_changes(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    ConfigLoader(service.agent_home).patch_editable_fields(revision, {"memory": {"batch_size": 14}})

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "merge-nonoverlapping-config-edit",
                "revision": revision,
                "baseline": baseline["fields"],
                "fields": {"runtime": {"max_iterations": 80}},
                "secrets": {},
            },
        )
        saved = await response.json()

    assert response.status == 200
    assert saved["fields"]["runtime"]["max_iterations"] == 80
    assert saved["fields"]["memory"]["batch_size"] == 14


@pytest.mark.asyncio
async def test_config_patch_merges_complete_browser_form_with_unchanged_model_arrays(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    browser_fields = json.loads(json.dumps(baseline["fields"]))
    for provider in browser_fields["models"]["providers"].values():
        provider.pop("api_key")
    browser_fields["runtime"]["max_iterations"] = 80
    ConfigLoader(service.agent_home).patch_editable_fields(revision, {"memory": {"batch_size": 14}})

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "merge-complete-browser-config-edit",
                "revision": revision,
                "baseline": baseline["fields"],
                "fields": browser_fields,
                "secrets": {},
            },
        )
        saved = await response.json()

    assert response.status == 200
    assert saved["fields"]["runtime"]["max_iterations"] == 80
    assert saved["fields"]["memory"]["batch_size"] == 14
    assert set(saved["fields"]["models"]["providers"]["primary"]["models"]) == {"small-model"}


@pytest.mark.asyncio
async def test_config_patch_rejects_duplicate_provider_ids_with_form_row_path(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    browser_fields = json.loads(json.dumps(baseline["fields"]))
    providers = browser_fields["models"]["providers"]
    for provider_id, provider in providers.items():
        provider.pop("api_key")
        provider["id"] = provider_id
    providers["new-provider"] = {**providers["primary"], "id": "primary"}
    before = (service.agent_home.path / "config.toml").read_bytes()

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "reject-duplicate-provider-id",
                "revision": revision,
                "baseline": baseline["fields"],
                "fields": browser_fields,
                "secrets": {},
            },
        )
        body = await response.json()

    assert response.status == 422
    assert body["field_errors"] == {"models.providers.new-provider.id": "must be unique"}
    assert (service.agent_home.path / "config.toml").read_bytes() == before


@pytest.mark.asyncio
async def test_config_patch_reports_external_secret_conflict(config_http: ConfigHttp) -> None:
    service, server, client_id, control = config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    secret_path = "models.providers.primary.api_key"
    ConfigLoader(service.agent_home).patch_editable_fields(
        revision,
        {},
        {secret_path: {"action": "replace", "value": "external-secret-326"}},
    )

    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers=headers,
            json={
                "request_id": "stale-secret-config-edit",
                "revision": revision,
                "baseline": baseline["fields"],
                "baseline_secrets": baseline["secret_revisions"],
                "fields": {"runtime": {"max_iterations": 80}},
                "secrets": {secret_path: {"action": "replace", "value": "client-secret-326"}},
            },
        )
        result = await response.json()

    assert response.status == 409
    assert result["code"] == "config_revision_conflict"
    assert result["field_errors"] == {secret_path: "changed elsewhere"}
    assert "external-secret-326" not in str(result)
    assert "client-secret-326" not in str(result)


@pytest.mark.asyncio
async def test_config_editor_orders_consecutive_edits_and_preserves_external_fields(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    ConfigLoader(service.agent_home).patch_editable_fields(
        revision, {"runtime": {"max_tool_result_chars": 5000}}
    )
    async with aiohttp.ClientSession() as http:

        async def save(
            sequence: int, value: int, source: dict[str, object] | None = None
        ) -> tuple[int, dict[str, object]]:
            source = baseline if source is None else source
            response = await http.patch(
                server.make_url("/api/v1/config"),
                headers=headers,
                json={
                    "request_id": f"ordered-edit-{sequence}",
                    "revision": source["revision"],
                    "baseline": source["fields"],
                    "baseline_secrets": source["secret_revisions"],
                    "editor_id": "one-settings-editor",
                    "edit_sequence": sequence,
                    "fields": {"runtime": {**cast(dict[str, object], cast(dict[str, object], source["fields"])["runtime"]), "max_iterations": value}},
                    "secrets": {},
                },
            )
            return response.status, await response.json()

        status, first_response = await save(1, 61)
        assert status == 200
        assert (await save(2, 62))[0] == 200
        # The UI can receive response 1 after edit 2 has already committed.
        assert (await save(3, 63, first_response))[0] == 200
        # Request 5 reaches the Service before delayed request 4.
        assert (await save(5, 65))[0] == 200
        status, rejected = await save(4, 64)
        assert status == 409 and rejected["code"] == "config_edit_superseded"
        # Changing back to the original value is still a new edit.
        assert (await save(6, 50))[0] == 200
        current = ConfigLoader(service.agent_home).load()
        assert current.runtime.max_iterations == 50
        assert current.runtime.max_tool_result_chars == 5000
        external = ConfigLoader(service.agent_home)
        external.patch_editable_fields(external.revision(), {"runtime": {"max_iterations": 70}})
        status, conflict = await save(7, 65)
        assert status == 409 and conflict["code"] == "config_revision_conflict"
        assert external.load().runtime.max_iterations == 70


@pytest.mark.asyncio
async def test_config_editor_coordinates_consecutive_mcp_keyword_arrays(
    config_http: ConfigHttp,
) -> None:
    service, server, client_id, control = config_http
    loader = ConfigLoader(service.agent_home)
    loader.path.write_text(FULL_CONFIG, encoding="utf-8")
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }
    async with aiohttp.ClientSession() as http:
        response = await http.get(server.make_url("/api/v1/config"), headers=headers)
        baseline = await response.json()
        loader.patch_editable_fields(
            baseline["revision"], {"mcp": {"http": {"call_timeout": 75}}}
        )

        async def save(
            sequence: int,
            keywords: list[str],
            source: dict[str, object] | None = None,
        ) -> tuple[int, dict[str, object]]:
            source = baseline if source is None else source
            fields = json.loads(json.dumps(source["fields"]))
            fields["mcp"]["http"]["tool_keywords"] = {"read": keywords}
            result = await http.patch(
                server.make_url("/api/v1/config"),
                headers=headers,
                json={
                    "request_id": f"ordered-keywords-{sequence}",
                    "revision": source["revision"],
                    "baseline": source["fields"],
                    "baseline_secrets": source["secret_revisions"],
                    "editor_id": "keyword-editor",
                    "edit_sequence": sequence,
                    "fields": {"mcp": fields["mcp"]},
                    "secrets": {},
                },
            )
            return result.status, await result.json()

        status, first_response = await save(1, ["resource"])
        assert status == 200
        assert (await save(2, ["resource", "file"]))[0] == 200
        assert (await save(3, ["resource", "file", "read"], first_response))[0] == 200
        current = loader.load().mcp["http"]
        assert current.tool_keywords["read"] == ("resource", "file", "read")
        assert current.call_timeout == 75
        loader.patch_editable_fields(
            loader.revision(), {"mcp": {"http": {"tool_keywords": {"read": ["external"]}}}}
        )
        status, conflict = await save(4, ["resource", "read"])
        assert status == 409 and conflict["code"] == "config_revision_conflict"
        assert loader.load().mcp["http"].tool_keywords["read"] == ("external",)


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", ["provider", "server", "header", "transport"])
async def test_config_structural_secret_removal_rejects_external_rotation(
    config_http: ConfigHttp,
    remove: str,
) -> None:
    service, server, client_id, control = config_http
    (service.agent_home.path / "config.toml").write_text(FULL_CONFIG, encoding="utf-8")
    baseline = service.config_view()
    revision = cast(str, baseline["revision"])
    fields = json.loads(json.dumps(baseline["fields"]))
    complete_model_settings(fields["models"])
    for provider in fields["models"]["providers"].values():
        provider.pop("api_key")
    secret_path = (
        "models.providers.retired.api_key"
        if remove == "provider"
        else "mcp.http.headers.Authorization"
    )
    if remove == "provider":
        del fields["models"]["providers"]["retired"]
    elif remove == "server":
        fields["mcp"] = {}
    elif remove == "header":
        fields["mcp"]["http"]["headers"] = {}
    else:
        fields["mcp"]["http"].update(transport="stdio", command="fixture", url=None, headers={})
    loader = ConfigLoader(service.agent_home)
    loader.patch_editable_fields(
        revision, {}, {secret_path: {"action": "replace", "value": "rotated-secret-326"}}
    )
    before = loader.path.read_bytes()
    token = create_credential(service.agent_home)
    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Aide-CSRF": token,
                "X-Aide-Client": client_id,
                "X-Aide-Control": control,
            },
            json={
                "request_id": f"remove-secret-{remove}",
                "revision": revision,
                "baseline": baseline["fields"],
                "baseline_secrets": baseline["secret_revisions"],
                "fields": fields,
                "secrets": {},
            },
        )
        body = await response.json()
    assert response.status == 409
    assert body["field_errors"] == {secret_path: "changed elsewhere"}
    assert "rotated-secret-326" not in str(body)
    assert loader.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", ["server", "header", "keyword"])
async def test_config_rejects_duplicate_named_mcp_form_rows(
    config_http: ConfigHttp,
    duplicate: str,
) -> None:
    service, server, client_id, control = config_http
    baseline = service.config_view()
    server_fields: dict[str, object] = {
        "name": "fixture",
        "enabled": False,
        "transport": "stdio",
        "command": "fixture",
    }
    mcp: dict[str, object] = {"first-row": server_fields}
    if duplicate == "server":
        mcp["second-row"] = server_fields
    elif duplicate == "header":
        server_fields.update(
            transport="streamable-http",
            command=None,
            url="https://mcp.example/tools",
            header_rows=[{"name": "Authorization", "secret": {"configured": False}}] * 2,
        )
    else:
        server_fields["tool_keyword_rows"] = [{"name": "read", "keywords": ["resource"]}] * 2
    before = (service.agent_home.path / "config.toml").read_bytes()
    token = create_credential(service.agent_home)
    async with aiohttp.ClientSession() as http:
        response = await http.patch(
            server.make_url("/api/v1/config"),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Aide-CSRF": token,
                "X-Aide-Client": client_id,
                "X-Aide-Control": control,
            },
            json={
                "request_id": f"duplicate-{duplicate}",
                "revision": baseline["revision"],
                "fields": {"mcp": mcp},
                "secrets": {},
            },
        )
        body = await response.json()
    assert response.status == 422
    assert "must be unique" in str(body["field_errors"])
    assert (service.agent_home.path / "config.toml").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["models", "mcp"])
async def test_partial_config_edit_preserves_nonoverlapping_external_leaf_fields(
    config_http: ConfigHttp, section: str,
) -> None:
    service, server, client_id, control = config_http
    (service.agent_home.path / "config.toml").write_text(FULL_CONFIG, encoding="utf-8")
    baseline = service.config_view()
    loader = ConfigLoader(service.agent_home)
    external: dict[str, object]
    fields: dict[str, object]
    if section == "models":
        prepared = json.loads(json.dumps(cast(dict[str, object], baseline["fields"])["models"]))
        complete_model_settings(prepared)
        for provider in prepared["providers"].values():
            provider.pop("api_key")
        loader.patch_editable_fields(loader.revision(), {"models": prepared})
        baseline = service.config_view()
        external = {"models": {"providers": {"primary": {"base_url": "https://external.example/v1"}, "retired": {}}}}
        prepared = json.loads(json.dumps(cast(dict[str, object], baseline["fields"])["models"]))
        for provider in prepared["providers"].values():
            provider.pop("api_key")
        prepared["providers"]["primary"]["models"]["small-model"]["context_window"] = 16384
        fields = {"models": {"providers": prepared["providers"]}}
    else:
        external = {"mcp": {"http": {"url": "https://external.example/tools"}}}
        fields = {"mcp": {"http": {"call_timeout": 90}}}
    loader.patch_editable_fields(cast(str, baseline["revision"]), external)
    token = create_credential(service.agent_home)
    async with aiohttp.ClientSession() as http:
        response = await http.patch(server.make_url("/api/v1/config"), headers={
            "Authorization": f"Bearer {token}", "X-Aide-CSRF": token,
            "X-Aide-Client": client_id, "X-Aide-Control": control,
        }, json={"request_id": f"partial-stale-{section}", "revision": baseline["revision"],
                 "baseline": baseline["fields"], "baseline_secrets": baseline["secret_revisions"],
                 "fields": fields, "secrets": {}})
        body = await response.json()
    assert response.status == 200, body
    configuration = loader.load()
    if section == "models":
        assert configuration.models.providers["primary"].base_url == "https://external.example/v1"
        assert configuration.models.providers["primary"].model_context_windows["small-model"] == 16384
    else:
        assert configuration.mcp["http"].url == "https://external.example/tools"
        assert configuration.mcp["http"].call_timeout == 90


@pytest.mark.asyncio
async def test_pending_service_reports_saved_config_restart_requirement(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    service = AgentService(home, reconnect_timeout=3600)
    await service.start()
    client = await service.register_client("cli")
    token = create_credential(home)
    try:
        # A repaired, valid saved document never activates the pending Service.
        (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
        async with TestServer(create_app(service), host="127.0.0.1") as server:
            async with aiohttp.ClientSession() as http:
                response = await http.get(server.make_url("/api/v1/config/startup"), headers={
                    "Authorization": f"Bearer {token}", "X-Aide-Client": client.client_id,
                })
                body = await response.json()
        assert response.status == 200
        assert body["startup"]["available"] is False
        assert body["startup"]["error"]["code"] == "config_restart_required"
        assert body["application"]["status"] == "restart-required"
        assert "minimal-secret" not in str(body)
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [True, False])
async def test_config_save_reports_safe_error_after_external_toml_corruption(
    config_http: ConfigHttp, stale: bool,
) -> None:
    service, server, client_id, control = config_http
    baseline = service.config_view()
    loader = ConfigLoader(service.agent_home)
    loader.path.write_text('[broken secret = "must-not-leak-326"', encoding="utf-8")
    before = loader.path.read_bytes()
    token = create_credential(service.agent_home)
    async with aiohttp.ClientSession() as http:
        response = await http.patch(server.make_url("/api/v1/config"), headers={
            "Authorization": f"Bearer {token}", "X-Aide-CSRF": token,
            "X-Aide-Client": client_id, "X-Aide-Control": control,
        }, json={"request_id": f"corrupted-save-{stale}",
                 "revision": baseline["revision"] if stale else loader.revision(),
                 "baseline": baseline["fields"], "fields": {"runtime": {"max_iterations": 80}},
                 "secrets": {}})
        body = await response.json()
    assert response.status == (409 if stale else 422)
    assert body["code"] == ("config_revision_conflict" if stale else "config_parse_error")
    assert "must-not-leak-326" not in str(body)
    assert loader.path.read_bytes() == before


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
async def full_config_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ConfigHttp]:
    from aide.agent.tools.mcp_runtime import MCPRuntimeManager, MCPStartupReport

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
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
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
        "X-Aide-CSRF": token,
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
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
