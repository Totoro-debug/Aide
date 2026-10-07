"""Project Memory without Sessions and hosted restart with browser reconnection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from aide.agent.memory.dream import DreamResult
from aide.service.discovery import read_credential, read_discovery
from aide.service.errors import ServiceError
from aide.service.process import serve_service
from aide.service.runtime import AgentService
from tests.service.test_protocol_contract import _validator
from tests.service.test_service_concurrency import _ConcurrentProvider
from tests.service.test_service_transport import _prepare_agent_home


@pytest.mark.asyncio
async def test_project_memory_is_scoped_and_does_not_create_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _prepare_agent_home(tmp_path / "home")
    monkeypatch.setattr("aide.service.runtime.create_provider", lambda *_: _ConcurrentProvider())
    service = AgentService(home)
    await service.start()
    try:
        client = await service.register_client("web")
        records = []
        for name in ("one", "two"):
            path = tmp_path / name
            path.mkdir()
            record, workspace, _ = await service.register_project(client.client_id, path)
            workspace.memory_manager.long_term_path.write_text(name, encoding="utf-8")
            records.append((record, workspace))
        for record, workspace in records:
            result = await service.project_memory_operation(
                client.client_id, record.project_id, record.project_id, "read"
            )
            _validator("memory_view_response").validate(result)
            assert result["content"] == record.path.name
            assert not workspace.loops
            assert client.current_session_id is None
        calls = 0

        async def dream() -> DreamResult:
            nonlocal calls
            calls += 1
            return DreamResult("No pending summaries", 0, False, 0)

        record, workspace = records[0]
        monkeypatch.setattr(workspace.dream, "run", dream)
        result = await service.project_memory_operation(client.client_id, record.project_id, "dream", "dream")
        _validator("dream_run_response").validate(result)
        assert await service.project_memory_operation(client.client_id, record.project_id, "dream", "dream") == result
        assert calls == 1
        with pytest.raises(ServiceError, match="request_id was already used"):
            await service.project_memory_operation(client.client_id, records[1][0].project_id, "dream", "dream")
        with pytest.raises(ServiceError) as missing:
            await service.project_memory_operation(client.client_id, "missing", "missing", "read")
        assert missing.value.code == "not_found"
        assert all(not workspace.loops for _, workspace in records)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_restart_refuses_unsaved_revision_and_invalid_configuration(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "home")
    service = AgentService(home)
    await service.start()
    try:
        client = await service.register_client("web")
        with pytest.raises(ServiceError) as conflict:
            await service.request_service_restart(client.client_id, "restart", "stale")
        assert conflict.value.code == "config_conflict"
        assert not service.restart_requested
        (home.path / "config.toml").write_text("invalid = [", encoding="utf-8")
        with pytest.raises(ServiceError) as invalid:
            await service.request_service_restart(client.client_id, "restart", "stale")
        assert invalid.value.code == "config_invalid"
        assert not service.restart_requested
        service.request_service_stop("stop")
        with pytest.raises(ServiceError) as stopping:
            await service.request_service_restart(client.client_id, "restart", "stale")
        assert stopping.value.code == "admission_closed"
        assert not service.restart_requested
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_hosted_restart_applies_config_and_keeps_browser_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _prepare_agent_home(tmp_path / "home")
    monkeypatch.setattr("aide.service.runtime.create_provider", lambda *_: _ConcurrentProvider())
    host = asyncio.create_task(serve_service(home, port=0, reconnect_timeout=3))
    headers: dict[str, str] = {}
    base = ""
    try:
        async with asyncio.timeout(10):
            while (discovery := read_discovery(home)) is None:
                await asyncio.sleep(0.02)
        credential = read_credential(home)
        assert credential is not None
        base = f"http://{discovery.host}:{discovery.port}"
        bearer = {"Authorization": f"Bearer {credential}", "X-Aide-CSRF": credential}
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as http:
            async def post(path: str, body: dict[str, object], proof: dict[str, str]) -> tuple[int, Any]:
                async with http.post(base + path, json=body, headers=proof) as response:
                    return response.status, await response.json()

            _, cli = await post("/api/v1/clients", {"request_id": "cli", "kind": "cli"}, bearer)
            _, ticket = await post("/api/v1/web/ticket", {"request_id": "ticket"},
                                   {**bearer, "X-Aide-Client": cli["client_id"]})
            _, exchange = await post("/api/v1/web/ticket", {"ticket": ticket["ticket"]}, {"Origin": base})
            headers = {"X-Aide-CSRF": exchange["csrf_token"], "Origin": base}
            _, web_client = await post("/api/v1/clients", {"request_id": "web", "kind": "web"}, headers)
            headers["X-Aide-Control"] = web_client["web_control_credential"]
            socket = await http.ws_connect(base + "/api/v1/events", headers={"Origin": base},
                                          protocols=("aide-v1", web_client["web_control_credential"]))
            project = tmp_path / "project"
            project.mkdir()
            status, registered = await post("/api/v1/projects", {"request_id": "project", "path": str(project)}, headers)
            assert status == 200
            async with http.get(base + "/api/v1/config", headers=headers) as response:
                config = await response.json()
            iterations = config["fields"]["runtime"]["max_iterations"] + 1
            async with http.patch(base + "/api/v1/config", headers=headers, json={
                "request_id": "save", "revision": config["revision"],
                "fields": {"runtime": {"max_iterations": iterations}}, "secrets": {},
            }) as response:
                assert response.status == 200
                saved = await response.json()
            request = {"request_id": "restart", "saved_revision": saved["application"]["saved_revision"]}
            _validator("service_restart_request").validate(request)
            refused, _ = await post("/api/v1/service/restart", request, {"Origin": base})
            assert refused == 403
            status, accepted = await post("/api/v1/service/restart", request, headers)
            assert status == 202
            _validator("service_restart_response").validate(accepted)
            async with asyncio.timeout(10):
                while (current := read_discovery(home)) is None or current.service_instance_id == discovery.service_instance_id:
                    await asyncio.sleep(0.02)
            assert current.port == discovery.port
            assert current.pid == discovery.pid
            await socket.close()
            status, reconnected = await post("/api/v1/clients", {"request_id": "reconnect", "kind": "web"}, headers)
            assert status == 200
            assert reconnected["client_id"] != web_client["client_id"]
            headers["X-Aide-Control"] = reconnected["web_control_credential"]
            new_socket = await http.ws_connect(base + "/api/v1/events", headers={"Origin": base},
                                              protocols=("aide-v1", reconnected["web_control_credential"]))
            async with http.get(base + "/api/v1/config", headers=headers) as response:
                assert response.status == 200
                active = await response.json()
            assert active["fields"]["runtime"]["max_iterations"] == iterations
            assert active["application"]["active_revision"] == request["saved_revision"]
            assert active["application"]["restart_required"] is False
            status, memory = await post(f'/api/v1/projects/{registered["project_id"]}/memory/read',
                                       {"request_id": "memory"}, headers)
            assert status == 200
            _validator("memory_view_response").validate(memory)
            status, replay = await post("/api/v1/service/restart", request, headers)
            assert status == 202 and replay == accepted
            assert read_discovery(home) == current
            await post("/api/v1/service/stop", {"request_id": "stop"}, headers)
            await new_socket.close()
        await asyncio.wait_for(host, 10)
        assert read_discovery(home) is None
    finally:
        if not host.done():
            host.cancel()
        await asyncio.gather(host, return_exceptions=True)
