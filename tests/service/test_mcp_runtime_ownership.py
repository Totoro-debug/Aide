from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web

import omni.agent.tools.mcp_runtime as mcp_runtime
from omni.agent.tools.mcp import MCPTool
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader, MCPServerConfiguration
from omni.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.mcp_wire import (
    ObservedLifetimes,
    WireServer,
    http_wire_server,
    stdio_requests,
    stdio_wire_configuration,
)
from tests.fixtures.project_removal import complete_project_removal
from tests.service.test_service_concurrency import _CollectingSink


@pytest.mark.asyncio
async def test_concurrent_service_activation_initializes_http_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG
        + '\n[mcp.servers.http]\nenabled = true\ntransport = "streamable-http"\n'
        + 'url = "https://mcp.example"\n',
        encoding="utf-8",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    connections: list[Connection] = []

    class Connection:
        unavailable = False

        def __init__(self) -> None:
            self.close_calls = 0

        async def connect(self) -> tuple[MCPTool, ...]:
            entered.set()
            await release.wait()
            return ()

        async def close(self) -> None:
            self.close_calls += 1

    def factory(configuration: MCPServerConfiguration, workspace: Path | None) -> Connection:
        assert configuration.transport == "streamable-http"
        assert workspace is None
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(mcp_runtime, "_default_connection_factory", factory)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    first = asyncio.create_task(service.start())
    second: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            second_entered = asyncio.Event()

            async def start_again() -> None:
                second_entered.set()
                await service.start()

            second = asyncio.create_task(start_again())
            await second_entered.wait()
            release.set()
            await asyncio.gather(first, second)
        assert len(connections) == 1
    finally:
        release.set()
        await asyncio.gather(first, *(() if second is None else (second,)), return_exceptions=True)
        await service.stop()

    assert [connection.close_calls for connection in connections] == [1]


@pytest.mark.asyncio
async def test_service_owns_http_and_workspace_owns_stdio_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parents[2]))
    observed = ObservedLifetimes(monkeypatch)
    home = AgentHome(tmp_path / "home")
    home.initialize()
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    stdio_configuration = stdio_wire_configuration(tmp_path, {}, name="stdio")
    release_request = asyncio.Event()
    original_http = WireServer.http

    async def controlled_http(server: WireServer, request: web.Request) -> web.StreamResponse:
        if request.method == "POST":
            message = await request.json()
            if message.get("method") == "tools/call" and message.get("params", {}).get(
                "arguments", {}
            ).get("hang"):
                server.respond(message)
                await release_request.wait()
                return web.Response(status=202)
            if message.get("method") == "notifications/cancelled":
                release_request.set()
        return await original_http(server, request)

    monkeypatch.setattr(WireServer, "http", controlled_http)

    async with http_wire_server({}) as (http_server, http_configuration):
        config = MINIMAL_VALID_CONFIG + (
            "\n[mcp.servers.http]\n"
            "enabled = true\n"
            'transport = "streamable-http"\n'
            f"url = {json.dumps(http_configuration.url)}\n"
            'tool_keywords = {echo = ["echo"]}\n'
            "\n[mcp.servers.stdio]\n"
            "enabled = true\n"
            'transport = "stdio"\n'
            f"command = {json.dumps(stdio_configuration.command)}\n"
            f"args = {json.dumps(list(stdio_configuration.args))}\n"
            'tool_keywords = {echo = ["echo"]}\n'
        )
        (home.path / "config.toml").write_text(config, encoding="utf-8")
        service = AgentService(home, ConfigLoader(home).load_for_startup())
        await service.start()
        try:
            assert [request["method"] for request in http_server.requests].count("initialize") == 1
            assert observed.processes == []
            first_record = service.projects.register(first_path)
            second_record = service.projects.register(second_path)
            client = await service.register_client("cli")
            await service.connect_client(client.client_id, _CollectingSink())
            first_workspace, alias_workspace, second_workspace = await asyncio.gather(
                service.attach_workspace(client.client_id, first_path),
                service.attach_workspace(client.client_id, first_path / "."),
                service.attach_workspace(client.client_id, second_path),
            )
            assert first_workspace is alias_workspace
            assert len(observed.processes) == 2
            assert [request["method"] for request in stdio_requests(tmp_path, "stdio")].count(
                "initialize"
            ) == 2

            assert first_workspace.resources is not None
            assert second_workspace.resources is not None
            first_http = next(
                tool for tool in first_workspace.resources.mcp_snapshot if tool.server_name == "http"
            )
            second_http = next(
                tool for tool in second_workspace.resources.mcp_snapshot if tool.server_name == "http"
            )
            assert first_http is second_http

            gateways = []
            for workspace in (first_workspace, second_workspace):
                for _ in range(10):
                    session_id = await workspace.create_draft(client.client_id)
                    loop = workspace.loops[session_id].loop
                    gateway = loop._create_executor()._tool_gateway.for_run(
                        exposed_names=(first_http.name,),
                        permission_context=PermissionContext(
                            level="full-access", workspace_root=workspace.workspace_path
                        ),
                    )
                    assert (
                        next(tool for tool in gateway.catalog if tool.name == first_http.name)
                        is first_http
                    )
                    gateways.append(gateway)
            assert len(observed.processes) == 2
            assert len(observed.closed) == 3
            assert [r["method"] for r in http_server.requests].count("initialize") == 1

            cancelled = asyncio.create_task(
                gateways[0].call(
                    ModelToolCall(id="cancel", name=first_http.name, arguments='{"hang":true}')
                )
            )
            try:
                pending_request = await http_server.wait_for("tools/call")
                results = await asyncio.gather(
                    *(
                        gateway.call(
                            ModelToolCall(
                                id=str(index),
                                name=first_http.name,
                                arguments=json.dumps({"value": str(index)}),
                            )
                        )
                        for index, gateway in enumerate(gateways[1:])
                    )
                )
                assert [(result.status, result.content) for result in results] == [
                    ("success", str(index)) for index in range(19)
                ]
                assert not cancelled.done()
                cancelled.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                notification = await http_server.wait_for("notifications/cancelled")
                assert notification["params"]["requestId"] == pending_request["id"]
            finally:
                cancelled.cancel()
                await asyncio.gather(cancelled, return_exceptions=True)
                release_request.set()
            await complete_project_removal(service, client.client_id, first_record.project_id)
            assert first_record.project_id not in {
                record.project_id for record in service.projects.list()
            }
            assert second_workspace.workspace_id in service.workspaces
            assert await second_http.execute_prepared({}) == "wire text"
            assert sum(process.returncode is not None for process in observed.processes) == 1
        finally:
            await service.stop()

    assert [request["method"] for request in http_server.requests].count("initialize") == 1
    observed.assert_closed()
    assert second_record.project_id in {record.project_id for record in service.projects.list()}
