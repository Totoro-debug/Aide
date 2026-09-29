"""Behavior tests for the real local service process and transport."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import cast
from uuid import uuid4
from xml.etree import ElementTree

import aiohttp
import pytest
from aiohttp import web

import myclaw.terminal.cli as cli
from myclaw.agent.session.restore import RestoreMode
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.service.client import (
    RemoteConfirmationCoordinator,
    RemoteControl,
    RemoteManagementCommandDispatcher,
    RemoteMessageBus,
    ServiceClient,
    ServiceStartupError,
)
from myclaw.service.discovery import (
    ServiceDiscovery,
    create_credential,
    identity_proof,
    read_credential,
    read_discovery,
    write_discovery,
)
from myclaw.service.errors import ServiceError
from myclaw.service.runtime import LocalService
from myclaw.terminal.conversation import TerminalConversationApp, _ConversationInput
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _prepare_agent_home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


@pytest.mark.asyncio
async def test_two_real_clients_use_one_service_and_claims_are_exclusive(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace, port=port)
        second = await ServiceClient.connect_or_start(home, workspace, port=port)
        assert first.discovery.service_instance_id == second.discovery.service_instance_id
        assert first.workspace_id == second.workspace_id
        assert first.session_id != second.session_id
        assert read_discovery(home) is not None
        assert read_credential(home)

        status = await first.management_dispatcher.dispatch("/status")
        assert status.handled is True
        assert status.status_view is not None
        config = await first.management_dispatcher.dispatch("/config")
        assert config.handled is True
        assert config.output is not None
        permission = await first.management_dispatcher.update_permission_level("read-only")
        assert permission.output == "Foreground permission level: read-only"
        effort = await first.management_dispatcher.update_reasoning_effort("high")
        assert effort.output == "Chat reasoning effort: high"

        with pytest.raises(ServiceError) as raised:
            await second.claim_session(first.session_id)
        assert raised.value.code == "session_claimed"

        async with aiohttp.ClientSession() as http:
            challenge = "review_identity_challenge_123"
            async with http.get(
                f"{first.base_url}/api/v1/service/identity",
                params={"challenge": challenge},
            ) as response:
                assert response.status == 200
                identity = await response.json()
                assert identity["service_instance_id"] == first.discovery.service_instance_id
                assert identity["proof"] == identity_proof(
                    first.token, challenge, first.discovery.service_instance_id, 1
                )
            async with http.get(f"{first.base_url}/api/v1/service") as response:
                assert response.status == 401
            async with http.get(
                f"{first.base_url}/api/v1/service",
                headers={"Authorization": f"Bearer {first.token}"},
            ) as response:
                assert response.status == 200
                body = await response.json()
                assert body["service_instance_id"] == first.discovery.service_instance_id
            async with http.get(
                f"{first.base_url}/api/v1/service",
                headers={
                    "Authorization": f"Bearer {first.token}",
                    "Origin": f"http://127.0.0.1:{port + 1}",
                },
            ) as response:
                assert response.status == 403
            async with http.get(
                f"{first.base_url}/api/v1/service",
                headers={
                    "Authorization": f"Bearer {first.token}",
                    "Host": f"localhost:{port + 1}",
                },
            ) as response:
                assert response.status == 403
            with pytest.raises(aiohttp.WSServerHandshakeError) as handshake:
                await http.ws_connect(
                    f"{first.base_url}/api/v1/events",
                    headers={
                        "Authorization": f"Bearer {first.token}",
                        "X-MyClaw-Client": first.client_id,
                    },
                )
            assert handshake.value.status == 403
            async with http.post(
                f"{first.base_url}/api/v1/clients",
                headers={
                    "Authorization": f"Bearer {first.token}",
                    "X-MyClaw-CSRF": first.token,
                },
                json={
                    "request_id": "duplicate-online-client",
                    "kind": "cli",
                    "reconnect_credential": first.reconnect_credential,
                },
            ) as response:
                assert response.status == 409

        reconnect_credential = first.reconnect_credential
        first_session_id = first.session_id
        await second.close()
        second = None
        await first.close()
        first = None
        reconnected = await ServiceClient.connect_or_start(
            home,
            workspace,
            port=port,
            reconnect_credential=reconnect_credential,
        )
        try:
            assert reconnected.session_id == first_session_id
            assert reconnected.claim_version == 1
        finally:
            await reconnected.close()
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_two_processes_start_or_join_one_service(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    script = """
import asyncio
import json
import sys
from pathlib import Path
from myclaw.config.agent_home import AgentHome
from myclaw.service.client import ServiceClient

async def main():
    client = await ServiceClient.connect_or_start(
        AgentHome(Path(sys.argv[1])), Path(sys.argv[2]), port=int(sys.argv[3])
    )
    try:
        print(json.dumps({"instance_id": client.discovery.service_instance_id,
                          "session_id": client.session_id}), flush=True)
        await asyncio.sleep(0.5)
    finally:
        await client.close()

asyncio.run(main())
"""
    processes = await asyncio.gather(
        *(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                str(home.path),
                str(workspace),
                str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            for _ in range(2)
        )
    )
    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*(process.communicate() for process in processes)), timeout=45
        )
        for process, (_stdout, stderr) in zip(processes, completed, strict=True):
            assert process.returncode == 0, stderr.decode(errors="replace")
        responses = [json.loads(stdout) for stdout, _stderr in completed]
        discovery = read_discovery(home)
        assert discovery is not None
        assert {response["instance_id"] for response in responses} == {
            discovery.service_instance_id
        }
        assert len({response["session_id"] for response in responses}) == 2
        assert discovery.port == port
    finally:
        for process in processes:
            if process.returncode is None:
                process.terminate()
                await process.wait()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_cli_entry_binds_terminal_to_service_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    original_connect = ServiceClient.connect_or_start
    connected: list[ServiceClient] = []

    async def connect(
        _client_type: type[ServiceClient], agent_home: AgentHome, path: Path
    ) -> ServiceClient:
        client = await original_connect(agent_home, path, port=port)
        connected.append(client)
        return client

    class FakeTerminalApp:
        def __init__(
            self,
            *,
            bus: RemoteMessageBus,
            control: RemoteControl,
            management_dispatcher: RemoteManagementCommandDispatcher,
        ) -> None:
            assert isinstance(bus, RemoteMessageBus)
            assert isinstance(control, RemoteControl)
            self.management = management_dispatcher

        def bind_confirmation_coordinator(self, coordinator: RemoteConfirmationCoordinator) -> None:
            assert isinstance(coordinator, RemoteConfirmationCoordinator)

        async def run_async(self) -> None:
            status = await self.management.dispatch("/status")
            assert status.handled is True
            assert status.status_view is not None

    monkeypatch.setattr(ServiceClient, "connect_or_start", classmethod(connect))
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeTerminalApp)
    try:
        await cli._run_service_cli_conversation(agent_home=home, workspace=workspace)
        assert len(connected) == 1
        assert connected[0].closed
    finally:
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_terminal_displays_session_projection_from_service(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    session.commit_agent_run(
        [{"role": "user", "content": "Visible through service"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        app = TerminalConversationApp(
            bus=client.bus,
            control=client.control,
            management_dispatcher=cast(ManagementCommandDispatcher, client.management_dispatcher),
        )
        app.bind_confirmation_coordinator(client.confirmation)
        async with app.run_test(size=(80, 24)) as pilot:
            await app._resume_selected_session(
                session.session_id, app.query_one("#conversation-input", _ConversationInput)
            )
            await pilot.pause()
            screen = ElementTree.fromstring(app.export_screenshot(simplify=True))
            visible = "".join(
                element.text or ""
                for element in screen.iter()
                if element.tag.endswith("text") and element.text
            )
            assert "Visible through service" in visible.replace("\xa0", " ")
    finally:
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_startup_does_not_stop_an_unrelated_port_occupant(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupant:
        occupant.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupant.bind(("127.0.0.1", port))
        occupant.listen(1)
        with pytest.raises(ServiceStartupError) as raised:
            await ServiceClient.connect_or_start(home, workspace, port=port)
        assert raised.value.code == "service_port_occupied"
        assert occupant.fileno() >= 0
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_unrelated_listener_never_receives_service_credential(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    token = create_credential(home)
    port = _free_port()
    received_authorization: list[str | None] = []

    async def unrelated(request: web.Request) -> web.Response:
        received_authorization.append(request.headers.get("Authorization"))
        return web.json_response(
            {"service_instance_id": "unrelated", "protocol_version": 1, "proof": "invalid"}
        )

    app = web.Application()
    app.router.add_get("/api/v1/service/identity", unrelated)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        write_discovery(home, ServiceDiscovery("unrelated", 1, "127.0.0.1", port, 42))
        with pytest.raises(ServiceStartupError) as raised:
            await ServiceClient.connect_or_start(home, workspace, port=port)
        assert raised.value.code == "service_port_occupied"
        assert received_authorization
        assert all(value is None for value in received_authorization)
        assert await ServiceClient.stop_existing(home, port=port) is False
        assert read_credential(home) == token
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_service_without_first_client_exits_after_connection_window(tmp_path: Path) -> None:
    service = LocalService(AgentHome(tmp_path / "agent-home"), reconnect_timeout=0.02)
    await service.start()
    await asyncio.wait_for(service.wait_closed(), timeout=1)
    assert service.state == "stopped"


@pytest.mark.asyncio
async def test_cli_restore_refreshes_claim_and_conversation_projection(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    session.commit_agent_run(
        [{"role": "user", "content": "Remove this turn"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        await client.switch_session(session.session_id)
        original_version = client.claim_version
        listing = await client.management_dispatcher.dispatch("/restore")
        assert listing.restore_listing is not None
        assert len(listing.restore_listing.anchors) == 1
        plan_result = await client.management_dispatcher.restore_inspect(1)
        assert plan_result.restore_plan is not None
        committed = await client.management_dispatcher.restore_commit(
            plan_result.restore_plan, RestoreMode.CONVERSATION_ONLY
        )
        assert committed.restore_result is not None
        assert client.claim_version == original_version + 1
        assert client.control.project_foreground_conversation().messages == ()
        assert Session.load(state, session.session_id).messages == []
    finally:
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)
