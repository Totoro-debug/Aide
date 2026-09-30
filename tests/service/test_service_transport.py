"""Behavior tests for the real local service process and transport."""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4
from xml.etree import ElementTree

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

import myclaw.terminal.cli as cli
from myclaw.agent.session.restore import RestoreMode
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_runtime import WorkspaceRuntime
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.schedule.model import JobSchedule, ScheduleJob
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
from myclaw.service.projects import ProjectCatalog
from myclaw.service.runtime import LocalService
from myclaw.service.transport import _project_job_summary
from myclaw.terminal.conversation import TerminalConversationApp, _ConversationInput
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def test_project_job_review_distinguishes_overdue_and_next_cron_occurrence() -> None:
    overdue = ScheduleJob(
        job_id=str(uuid4()),
        message="Past due task",
        schedule=JobSchedule.every(3600),
        created_at_ms=1_600_000_000_000,
        updated_at_ms=1_600_000_000_000,
    )
    cron = ScheduleJob(
        job_id=str(uuid4()),
        message="Future cron task",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        created_at_ms=1_600_000_000_000,
        updated_at_ms=1_600_000_000_000,
    )

    assert _project_job_summary(overdue)["review_status"] == "overdue"
    assert _project_job_summary(overdue)["due_at"] is not None
    assert _project_job_summary(cron)["review_status"] == "next_on_resume"
    assert _project_job_summary(cron)["due_at"] is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _prepare_agent_home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


async def _persist_session(
    workspace: Path,
    *,
    home: AgentHome,
    title: str,
    created_at: datetime,
    content: str,
) -> str:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: created_at)
    session.update_metadata(title=title)
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    return session.session_id


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
async def test_project_http_contract_reports_path_errors_and_keeps_cli_workspaces_unregistered(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    cli_workspace = tmp_path / "cli-workspace"
    cli_workspace.mkdir()
    port = _free_port()
    client: ServiceClient | None = None
    cli_client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, project, port=port)
        cli_client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
        headers = {
            "Authorization": f"Bearer {client.token}",
            "X-MyClaw-CSRF": client.token,
            "X-MyClaw-Client": client.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{client.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "relative", "path": "relative/project"},
            ) as response:
                assert response.status == 422
                error = await response.json()
                assert error["code"] == "validation_error"
                assert error["field_errors"]["path"]

            nested_home_path = home.path / "nested"
            nested_home_path.mkdir()
            async with http.post(
                f"{client.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "agent-home", "path": str(nested_home_path)},
            ) as response:
                assert response.status == 422
                error = await response.json()
                assert error["code"] == "validation_error"
                assert "Agent Home" in error["field_errors"]["path"]

            async with http.get(f"{client.base_url}/api/v1/projects", headers=headers) as response:
                assert response.status == 200
                assert await response.json() == {"projects": []}

            async with http.post(
                f"{client.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "register", "path": str(project)},
            ) as response:
                assert response.status == 200
                registered = await response.json()
                assert registered["schedule_state"] == "available"
                project_id = registered["project_id"]

            async with http.post(
                f"{client.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "alias", "path": str(project / ".")},
            ) as response:
                assert response.status == 200
                assert (await response.json())["project_id"] == project_id

            async with aiohttp.ClientSession() as status_http:
                async with status_http.get(
                    f"{client.base_url}/api/v1/projects", headers=headers
                ) as response:
                    assert response.status == 200
                    active_project = (await response.json())["projects"][0]
                    assert active_project["schedule_status"] == {
                        "admitted": True,
                        "status": "available",
                        "active_job_count": 0,
                    }

            shutil.rmtree(project)
            async with http.get(f"{client.base_url}/api/v1/projects", headers=headers) as response:
                assert response.status == 200
                listed = (await response.json())["projects"]
                assert listed == [
                    {
                        "project_id": project_id,
                        "path": str(project.resolve()),
                        "name": project.name,
                        "schedule_state": "available",
                        "available": False,
                        "saved_jobs": [],
                        "schedule_status": None,
                    }
                ]
    finally:
        if cli_client is not None:
            await cli_client.close()
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_project_session_http_scope_claim_and_empty_draft_contract(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    other_project = tmp_path / "other-project"
    project.mkdir()
    other_project.mkdir()
    older_id = await _persist_session(
        project,
        home=home,
        title="Older project session",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="project-only older content",
    )
    newer_id = await _persist_session(
        project,
        home=home,
        title="Newer project session",
        created_at=datetime(2026, 2, 1, 9, tzinfo=UTC),
        content="project-only newer content",
    )
    offset_id = await _persist_session(
        project,
        home=home,
        title="Offset project session",
        created_at=datetime(2026, 2, 1, 10, tzinfo=timezone(timedelta(hours=8))),
        content="project-only offset content",
    )
    other_id = await _persist_session(
        other_project,
        home=home,
        title="Other project session",
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        content="other-project content",
    )
    project_state = WorkspaceState(project)
    project_state.initialize(agent_home_root=home.path)
    schedule = Session.create_schedule(project_state, uuid4(), title="Schedule only")
    schedule.commit_agent_run(
        [{"role": "user", "content": "schedule-only content"}],
        pending_last_compacted=schedule.last_compacted,
        pending_action_summary="",
    )
    await schedule.wait_for_pending_persist()

    port = _free_port()
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, project, port=port)
        second = await ServiceClient.connect_or_start(home, other_project, port=port)
        headers = {
            "Authorization": f"Bearer {first.token}",
            "X-MyClaw-CSRF": first.token,
            "X-MyClaw-Client": first.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{first.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "register-project", "path": str(project)},
            ) as response:
                assert response.status == 200
                registration = await response.json()
            project_id = cast(str, registration["project_id"])

            async with http.get(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions",
                headers=headers,
            ) as response:
                assert response.status == 200
                listing = await response.json()
            assert [item["id"] for item in listing["sessions"]] == [newer_id, offset_id, older_id]
            assert all(item["occupied"] is False for item in listing["sessions"])
            assert other_id not in json.dumps(listing)
            assert "schedule-only content" not in json.dumps(listing)
            assert "reconnect_credential" not in json.dumps(listing)

            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions",
                headers=headers,
                json={"request_id": "create-empty-draft"},
            ) as response:
                assert response.status == 200
                draft = await response.json()
            draft_id = cast(str, draft["session_id"])
            assert draft_id not in {newer_id, offset_id, older_id}

            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}/claim",
                headers=headers,
                json={"request_id": "claim-newer"},
            ) as response:
                assert response.status == 200
                claim = await response.json()
            claim_data = cast(dict[str, object], claim["claim"])
            assert claim["snapshot"]["messages"]

            second_headers = {
                "Authorization": f"Bearer {second.token}",
                "X-MyClaw-CSRF": second.token,
                "X-MyClaw-Client": second.client_id,
            }
            async with http.post(
                f"{second.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}/claim",
                headers=second_headers,
                json={"request_id": "claim-contested"},
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "session_claimed"
            async with http.get(
                (
                    f"{second.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}"
                    f"?claim_version={claim_data['claim_version']}"
                ),
                headers={
                    **second_headers,
                    "X-MyClaw-Claim": cast(str, claim_data["reconnect_credential"]),
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "stale_claim"

            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}/release",
                headers={
                    **headers,
                    "X-MyClaw-Claim": cast(str, claim_data["reconnect_credential"]),
                },
                json={
                    "request_id": "release-newer",
                    "claim_version": claim_data["claim_version"],
                },
            ) as response:
                assert response.status == 200

            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{draft_id}/claim",
                headers=headers,
                json={"request_id": "claim-empty-draft"},
            ) as response:
                assert response.status == 200
                empty_claim = await response.json()
            empty_claim_data = cast(dict[str, object], empty_claim["claim"])
            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{draft_id}/release",
                headers={
                    **headers,
                    "X-MyClaw-Claim": cast(str, empty_claim_data["reconnect_credential"]),
                },
                json={
                    "request_id": "release-empty-draft",
                    "claim_version": empty_claim_data["claim_version"],
                },
            ) as response:
                assert response.status == 200

            assert not (project_state.sessions_directory / f"{draft_id}.jsonl").exists()
            async with http.get(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions",
                headers=headers,
            ) as response:
                assert response.status == 200
                final_listing = await response.json()
            assert draft_id not in {item["id"] for item in final_listing["sessions"]}
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_web_draft_stays_empty_when_workspace_has_startup_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    restored_id = await _persist_session(
        project,
        home=home,
        title="Restored history",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="Old history",
    )
    record = ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("web")
        monkeypatch.setattr(
            WorkspaceRuntime, "startup_session_id", property(lambda _runtime: restored_id)
        )
        created = await service.create_project_session(client.client_id, record.project_id)
        draft_id = cast(str, created["session_id"])
        assert draft_id != restored_id
        claim = await service.claim_project_session(client.client_id, record.project_id, draft_id)
        assert cast(dict[str, object], claim["snapshot"])["messages"] == []
        claim_data = cast(dict[str, object], claim["claim"])
        await service.release_project_session(
            client.client_id,
            record.project_id,
            draft_id,
            cast(int, claim_data["claim_version"]),
            cast(str, claim_data["reconnect_credential"]),
        )
        assert not (WorkspaceState(project).sessions_directory / f"{draft_id}.jsonl").exists()
        assert (WorkspaceState(project).sessions_directory / f"{restored_id}.jsonl").exists()
    finally:
        await service.stop()


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


@pytest.mark.asyncio
async def test_browser_ticket_is_one_time_cookie_auth_and_static_routes_are_bounded(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = await _persist_session(
        workspace,
        home=home,
        title="Private history",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="Private conversation body",
    )
    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        launch_url = await client.create_web_ticket()
        ticket = launch_url.rsplit("#ticket=", 1)[-1]
        assert ticket
        assert client.token not in launch_url

        cookie_jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(cookie_jar=cookie_jar) as browser:
            async with browser.get(f"{client.base_url}/api/v1/service") as response:
                assert response.status == 401

            async with browser.get(
                f"{client.base_url}/",
                headers={"Host": f"outside.example:{port}"},
            ) as response:
                assert response.status == 403

            async with browser.post(
                f"{client.base_url}/api/v1/web/ticket",
                headers={"Origin": "http://outside.example"},
                json={"ticket": ticket},
            ) as response:
                assert response.status == 403

            async with browser.get(f"{client.base_url}/") as response:
                assert response.status == 200
                index = await response.text()
                assert "MyClaw" in index
                assert client.token not in index

            async with browser.get(
                f"{client.base_url}/assets/C:/Windows/win.ini",
                headers={"Origin": client.base_url},
            ) as response:
                assert response.status == 404

            async with browser.post(
                f"{client.base_url}/api/v1/web/ticket",
                headers={"Origin": client.base_url},
                json={"ticket": ticket},
            ) as response:
                assert response.status == 200
                exchanged = await response.json()
                assert isinstance(exchanged["csrf_token"], str)
                assert client.token not in await response.text()
                cookies = browser.cookie_jar.filter_cookies(URL(client.base_url))
                assert "myclaw_session" in cookies
                assert "myclaw_csrf" not in cookies

            async with browser.post(
                f"{client.base_url}/api/v1/web/ticket",
                headers={"Origin": client.base_url},
                json={"ticket": ticket},
            ) as response:
                assert response.status == 401

            csrf = exchanged["csrf_token"]
            async with browser.post(
                f"{client.base_url}/api/v1/clients",
                headers={"Origin": client.base_url, "X-MyClaw-CSRF": csrf},
                json={"request_id": "browser-client", "kind": "web"},
            ) as response:
                assert response.status == 200
                web_client = await response.json()
                assert web_client["client_id"]
                control = web_client["web_control_credential"]
                assert isinstance(control, str)

            web_headers = {
                "Origin": client.base_url,
                "X-MyClaw-CSRF": csrf,
                "X-MyClaw-Control": control,
            }
            async with browser.post(
                f"{client.base_url}/api/v1/projects",
                headers=web_headers,
                json={"request_id": "browser-project", "path": str(workspace)},
            ) as response:
                assert response.status == 200
                project_id = (await response.json())["project_id"]

            socket = await browser.ws_connect(
                f"{client.base_url}/api/v1/events",
                headers={"Origin": client.base_url},
                protocols=("myclaw-v1", control),
            )
            try:
                async with browser.post(
                    f"{client.base_url}/api/v1/projects/{project_id}/sessions/{session_id}/claim",
                    headers=web_headers,
                    json={"request_id": "owner-claim"},
                ) as response:
                    assert response.status == 200
                    owned = await response.json()
                    assert "Private conversation body" in json.dumps(owned)

                async with browser.post(
                    f"{client.base_url}/api/v1/projects/{project_id}/sessions/{session_id}/claim",
                    headers={"Origin": client.base_url, "X-MyClaw-CSRF": csrf},
                    json={"request_id": "copied-tab-claim"},
                ) as response:
                    assert response.status == 403
                    assert "Private conversation body" not in await response.text()

                claim_data = owned["claim"]
                async with browser.get(
                    (
                        f"{client.base_url}/api/v1/projects/{project_id}/sessions/{session_id}"
                        f"?claim_version={claim_data['claim_version']}"
                    ),
                    headers={
                        "Origin": client.base_url,
                        "X-MyClaw-Claim": claim_data["reconnect_credential"],
                    },
                ) as response:
                    assert response.status == 403
                    assert "Private conversation body" not in await response.text()

                with pytest.raises(aiohttp.WSServerHandshakeError) as handshake:
                    await browser.ws_connect(
                        f"{client.base_url}/api/v1/events",
                        headers={"Origin": client.base_url},
                    )
                assert handshake.value.status == 403
                async with browser.post(
                    f"{client.base_url}/api/v1/clients",
                    headers={"Origin": client.base_url, "X-MyClaw-CSRF": csrf},
                    json={"request_id": "copied-tab-register", "kind": "web"},
                ) as response:
                    assert response.status == 409
            finally:
                await socket.close()

            async with browser.get(
                f"{client.base_url}/api/v1/service",
                headers={"Origin": client.base_url},
            ) as response:
                assert response.status == 200

            async with browser.post(
                f"{client.base_url}/api/v1/clients",
                headers={"Origin": client.base_url},
                json={"request_id": "missing-csrf", "kind": "web"},
            ) as response:
                assert response.status == 403

            async with browser.get(
                f"{client.base_url}/api/v1/service",
                headers={"Origin": "http://evil.example"},
            ) as response:
                assert response.status == 403

            async with browser.get(
                f"{client.base_url}/assets/../service.token",
                headers={"Origin": client.base_url},
            ) as response:
                assert response.status in {403, 404}
                assert "service.token" not in await response.text()

        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as foreign:
            with pytest.raises(aiohttp.WSServerHandshakeError) as handshake:
                await foreign.ws_connect(
                    f"{client.base_url}/api/v1/events",
                    headers={"Origin": "http://evil.example"},
                )
            assert handshake.value.status == 403
    finally:
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)
