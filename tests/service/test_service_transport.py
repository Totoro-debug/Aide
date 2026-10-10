"""Behavior tests for the real local service process and transport."""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4
from xml.etree import ElementTree

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from yarl import URL

import aide.service.runtime.projections as service_runtime
import aide.terminal.cli as cli
from aide.agent.session.restore import RestoreMode
from aide.agent.session.session import Session
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.management.commands import ManagementCommandDispatcher
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.client import (
    RemoteConfirmationCoordinator,
    RemoteControl,
    RemoteManagementCommandDispatcher,
    RemoteMessageBus,
    ServiceClient,
    ServiceStartupError,
    _port_is_open,
)
from aide.service.conversation_workspaces import ConversationWorkspaceCatalog
from aide.service.discovery import (
    ServiceDiscovery,
    create_credential,
    credential_path,
    identity_proof,
    read_credential,
    read_discovery,
    write_discovery,
)
from aide.service.errors import ServiceError
from aide.service.projects import ProjectCatalog
from aide.service.runtime import AgentService
from aide.service.runtime.projections import _project_job_summary
from aide.service.transport import create_app
from aide.terminal.conversation import TerminalConversationApp, _ConversationInput
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def test_project_job_review_uses_controlled_time_for_at_every_and_cron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_now = datetime(2026, 9, 30, 13, 0, tzinfo=UTC)

    class FrozenDateTime:
        @staticmethod
        def now(tz: timezone | None = None) -> datetime:
            return frozen_now if tz is None else frozen_now.astimezone(tz)

        @staticmethod
        def fromtimestamp(timestamp: float, tz: timezone | None = None) -> datetime:
            return datetime.fromtimestamp(timestamp, tz)

    monkeypatch.setattr(service_runtime, "datetime", FrozenDateTime)

    at = ScheduleJob(
        job_id=str(uuid4()),
        message="Past at task",
        schedule=JobSchedule.at("2026-09-30T12:00:00.000+00:00"),
        created_at_ms=1_600_000_000_000,
        updated_at_ms=1_600_000_000_000,
    )
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

    assert _project_job_summary(at)["review_status"] == "overdue"
    assert _project_job_summary(at)["due_at"] == "2026-09-30T12:00:00+00:00"
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
    creation_scope: str | None = None,
) -> str:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: created_at)
    session.update_metadata(title=title)
    if creation_scope is not None:
        session.update_metadata(creation_scope=creation_scope)
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
@pytest.mark.parametrize("kind", ["cli", "web"])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("entry", ["create", "open"])
async def test_all_foreground_creation_entries_record_explicit_scope(
    tmp_path: Path,
    kind: str,
    registered: bool,
    entry: str,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    if registered:
        service.projects.register(directory)
    await service.start()
    try:
        client = await service.register_client(kind)
        workspace = await service.attach_workspace(client.client_id, directory)
        if entry == "create":
            result = await service.create_session(client.client_id, workspace.workspace_id)
        else:
            result = cast(
                dict[str, object],
                await service.open_conversation(
                    client.client_id,
                    workspace_id=workspace.workspace_id,
                    create_new=True,
                ),
            )
        session_id = cast(str, result["session_id"])
        session = workspace._loops[session_id].loop.session
        expected = "project" if kind == "cli" and registered else "chat"
        assert session.metadata["creation_scope"] == expected
        assert not (directory / ".aide" / "sessions" / f"{session_id}.jsonl").exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_chat_draft_does_not_reuse_or_reclassify_restored_project_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    home = _prepare_agent_home(tmp_path / "agent-home")
    shared = tmp_path / "shared"
    shared.mkdir()
    restored_id = await _persist_session(
        shared,
        home=home,
        title="Project history",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="Old history",
        creation_scope="project",
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    service.projects.register(shared)
    service.conversation_workspaces.remember(shared)
    await service.start()
    try:
        client = await service.register_client("web")
        workspace = next(iter(service._workspaces.values()))
        monkeypatch.setattr(workspace, "_restore_result", SimpleNamespace(session_id=restored_id))
        created = await service.create_session(client.client_id, workspace.workspace_id)
        draft_id = cast(str, created["session_id"])
        assert draft_id != restored_id
        claimed = await service.claim(client.client_id, workspace.workspace_id, draft_id)
        assert cast(dict[str, object], claimed["snapshot"])["messages"] == []
        assert workspace._loops[draft_id].loop.session.metadata["creation_scope"] == "chat"
        assert not (shared / ".aide" / "sessions" / f"{draft_id}.jsonl").exists()
        restored = await workspace._create_loop(restored_id, client_id=client.client_id)
        assert restored.loop.session.metadata["creation_scope"] == "project"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_connected_web_chat_enters_schedule_admission(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    chat = tmp_path / "chat"
    config_path = home.path / "config.toml"
    config_path.write_text(
        MINIMAL_VALID_CONFIG + f'\n[web]\ndefault_chat_workspace = "{chat.as_posix()}"\n',
        encoding="utf-8",
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("web")

        class Sink:
            async def send_event(self, event: dict[str, object]) -> None:
                pass

        await service.connect_client(client.client_id, Sink())
        entry = await service.enter_default_conversation_workspace(client.client_id)
        workspace = service.workspace(cast(str, entry["workspace_id"]))
        assert workspace._schedule_admitted
        await service.disconnect_client(client.client_id)
        assert not workspace._schedule_admitted
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_web_can_enter_the_default_conversation_workspace_under_agent_home(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    chat = home.path / "chat"
    (home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG + f'\n[web]\ndefault_chat_workspace = "{chat.as_posix()}"\n',
        encoding="utf-8",
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    token = create_credential(home)
    server = TestServer(create_app(service))
    await service.start()
    try:
        client = await service.register_client("web")
        await server.start_server()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Aide-Client": client.client_id,
            "X-Aide-CSRF": token,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                server.make_url("/api/v1/chat/workspaces/enter"),
                headers=headers,
                json={"request_id": "enter-default-chat"},
            ) as response:
                assert response.status == 200
                result = await response.json()

            workspace_id = result["workspace_id"]
            async with http.post(
                server.make_url("/api/v1/conversations/open"),
                headers=headers,
                json={
                    "request_id": "open-chat-draft",
                    "workspace_id": workspace_id,
                    "create_new": True,
                },
            ) as response:
                assert response.status == 200
                opened = await response.json()

            session_id = opened["session_id"]
            claim = opened["claim"]
            assert opened["snapshot"]["session_id"] == session_id
            session_path = chat / ".aide" / "sessions" / f"{session_id}.jsonl"
            assert not session_path.exists()
            workspace = service.workspace(workspace_id)
            draft = workspace._loops[session_id].loop.session
            assert draft.metadata["creation_scope"] == "chat"

            draft.commit_agent_run(
                [{"role": "user", "content": "Persisted from a chat draft"}],
                pending_last_compacted=draft.last_compacted,
                pending_action_summary="",
                restore_before=draft.capture_restore_before(),
                restore_run_token=uuid4(),
            )
            await draft.wait_for_pending_persist()
            _loaded_id, _created_at, _updated_at, metadata = Session.load_header(
                workspace.workspace_state, session_id
            )
            assert metadata["creation_scope"] == "chat"

            async with http.post(
                server.make_url(f"/api/v1/workspaces/{workspace_id}/sessions/{session_id}/release"),
                headers={**headers, "X-Aide-Claim": claim["reconnect_credential"]},
                json={"request_id": "release-chat-draft", "claim_version": claim["claim_version"]},
            ) as response:
                assert response.status == 200
                assert (await response.json())["released"] is True

        assert result["directory"] == str(chat.resolve())
        assert result["project_id"] is None
        assert (chat / ".aide" / "sessions").is_dir()
        assert ConversationWorkspaceCatalog(home).list() == (chat.resolve(),)
        assert service.workspace(result["workspace_id"]).workspace_path == chat.resolve()
        assert service.projects.list() == ()
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_chat_history_lists_only_chat_sessions_without_activating_old_workspaces(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    shared = tmp_path / "project-and-chat"
    old_chat = tmp_path / "old-chat"
    shared.mkdir()
    old_chat.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    project_id = service.projects.register(shared).project_id
    service.conversation_workspaces.remember(shared)
    service.conversation_workspaces.remember(old_chat)
    now = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    expected_chat_id = await _persist_session(
        shared,
        home=home,
        title="Chat scoped",
        created_at=now,
        content="chat body",
        creation_scope="chat",
    )
    expected_project_id = await _persist_session(
        shared,
        home=home,
        title="Project scoped",
        created_at=now + timedelta(seconds=1),
        content="project body",
        creation_scope="project",
    )
    expected_legacy_project_id = await _persist_session(
        shared,
        home=home,
        title="Legacy project",
        created_at=now + timedelta(seconds=2),
        content="legacy project body",
    )
    expected_other_chat_id = await _persist_session(
        old_chat,
        home=home,
        title="Other chat",
        created_at=now + timedelta(seconds=3),
        content="other chat body",
        creation_scope="chat",
    )
    legacy_chat_path = old_chat / ".aide" / "sessions" / f"{expected_other_chat_id}.jsonl"
    legacy_chat_header = legacy_chat_path.read_bytes().split(b"\n", 1)[0]
    legacy_chat_path.write_bytes(legacy_chat_header + b"\nnot-loaded-by-history-listing\n")
    unscoped_id = await _persist_session(
        old_chat,
        home=home,
        title="Unscoped",
        created_at=now + timedelta(seconds=4),
        content="unscoped body",
    )
    unscoped_path = old_chat / ".aide" / "sessions" / f"{unscoped_id}.jsonl"
    unscoped_before = unscoped_path.read_bytes()
    legacy_project_path = shared / ".aide" / "sessions" / f"{expected_legacy_project_id}.jsonl"
    legacy_project_before = legacy_project_path.read_bytes()
    token = create_credential(home)
    server = TestServer(create_app(service))
    await service.start()
    try:
        client = await service.register_client("web")
        await server.start_server()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Aide-Client": client.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.get(
                server.make_url("/api/v1/chat/sessions"),
                headers=headers,
                params={"limit": "1"},
            ) as response:
                assert response.status == 200
                first_page = await response.json()
            assert first_page["next_cursor"] is not None
            async with http.get(
                server.make_url("/api/v1/chat/sessions"),
                headers=headers,
                params={"limit": "1", "cursor": first_page["next_cursor"]},
            ) as response:
                assert response.status == 200
                second_page = await response.json()
            async with http.get(
                server.make_url(f"/api/v1/projects/{project_id}/sessions"),
                headers=headers,
            ) as response:
                assert response.status == 200
                project_page = await response.json()
            async with http.get(
                server.make_url("/api/v1/chat/sessions"),
                headers=headers,
                params={"title": "CHAT SCOPED"},
            ) as response:
                assert response.status == 200
                filtered_page = await response.json()

        sessions = first_page["sessions"] + second_page["sessions"]
        assert [item["id"] for item in sessions] == [expected_other_chat_id, expected_chat_id]
        assert filtered_page["sessions"] == second_page["sessions"]
        assert filtered_page["next_cursor"] is None
        for item, expected_time in zip(sessions, (now + timedelta(seconds=3), now), strict=True):
            assert item["created_at"] == item["updated_at"] == expected_time.isoformat()
        assert second_page["next_cursor"] is None
        assert first_page["unavailable_directories"] == []
        assert second_page["unavailable_directories"] == []
        assert {item["id"] for item in sessions} == {
            expected_chat_id,
            expected_other_chat_id,
        }
        assert {item["title"] for item in sessions} == {
            "Chat scoped",
            "Other chat",
        }
        assert {item["id"] for item in project_page["sessions"]} == {
            expected_project_id,
        }
        assert {item["title"] for item in project_page["sessions"]} == {
            "Project scoped",
        }
        assert unscoped_path.read_bytes() == unscoped_before
        assert legacy_project_path.read_bytes() == legacy_project_before
        service.projects.remove(project_id)
        chat_page = service.list_chat_sessions_page(client.client_id)
        assert [item["id"] for item in cast(list[dict[str, object]], chat_page["sessions"])] == [
            expected_other_chat_id,
            expected_chat_id,
        ]
        reregistered = service.projects.register(shared)
        _, _, reregistered_page = await service.list_project_sessions_page(
            client.client_id,
            reregistered.project_id,
        )
        assert [
            item["id"] for item in cast(list[dict[str, object]], reregistered_page["sessions"])
        ] == [expected_project_id]
        assert unscoped_path.read_bytes() == unscoped_before
        assert legacy_project_path.read_bytes() == legacy_project_before
        assert all("messages" not in item for item in sessions)
        assert any(item["directory"] == str(old_chat.resolve()) for item in sessions)
        assert all(
            workspace.workspace_path != old_chat.resolve()
            for workspace in service._workspaces.values()
        )
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory_state", ["missing", "unreadable"])
async def test_chat_history_reports_unavailable_workspace_without_recreating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_state: str,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    old_chat = tmp_path / "old-chat"
    old_chat.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    service.conversation_workspaces.remember(old_chat)
    await _persist_session(
        old_chat,
        home=home,
        title="Old conversation",
        created_at=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
        content="old body",
        creation_scope="chat",
    )
    expected_directory = str(old_chat.resolve())
    if directory_state == "missing":
        shutil.rmtree(old_chat)
    else:
        original_iterdir = Path.iterdir

        def inaccessible_iterdir(directory: Path) -> Iterator[Path]:
            if directory == old_chat / ".aide" / "sessions":
                raise PermissionError("History directory cannot be enumerated")
            return original_iterdir(directory)

        monkeypatch.setattr(Path, "iterdir", inaccessible_iterdir)
    token = create_credential(home)
    server = TestServer(create_app(service))
    await service.start()
    try:
        client = await service.register_client("web")
        await server.start_server()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Aide-Client": client.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.get(
                server.make_url("/api/v1/chat/sessions"), headers=headers
            ) as response:
                assert response.status == 200
                page = await response.json()

        assert page["sessions"] == []
        assert page["unavailable_directories"] == [expected_directory]
        assert old_chat.exists() is (directory_state != "missing")
    finally:
        await server.close()
        await service.stop()


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

        await first._command(
            "subscribe", workspace_id=None, session_id=None, claim_version=None,
            payload={"last_seq": None},
        )
        assert first.control.foreground_input_admitted()
        assert not first.control.has_active_run
        assert first.control.project_foreground_conversation().session_id == first.session_id

        status_response = await first._http_request(
            "GET",
            (
                f"/api/v1/workspaces/{first.workspace_id}/management/status"
                f"?session_id={first.session_id}&claim_version={first.claim_version}"
            ),
            extra_headers={"X-Aide-Claim": first.claim_credential},
        )
        status_view = cast(
            dict[str, object], cast(dict[str, object], status_response["result"])["status_view"]
        )
        assert status_view["chat_model"]
        assert status_view["context_window"]
        assert status_view["current_permission_level"] == "workspace-write"
        assert "chat_reasoning_effort" in status_view
        assert first.token not in json.dumps(status_response)

        typed_permission = await first._http_request(
            "POST",
            f"/api/v1/workspaces/{first.workspace_id}/management/permission",
            payload={
                "request_id": "typed-permission",
                "current_session_id": first.session_id,
                "claim_version": first.claim_version,
                "permission_level": "read-only",
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": first.claim_credential},
        )
        typed_permission_result = cast(dict[str, object], typed_permission["result"])
        assert typed_permission_result["published_permission_level"] == "read-only"

        replay = await first._http_request(
            "POST",
            f"/api/v1/workspaces/{first.workspace_id}/management/permission",
            payload={
                "request_id": "typed-permission",
                "current_session_id": first.session_id,
                "claim_version": first.claim_version,
                "permission_level": "read-only",
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": first.claim_credential},
        )
        assert replay == typed_permission

        first_status_after_permission = await first._http_request(
            "POST",
            f"/api/v1/workspaces/{first.workspace_id}/management/status",
            payload={
                "request_id": "typed-status-after-permission",
                "current_session_id": first.session_id,
                "claim_version": first.claim_version,
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": first.claim_credential},
        )
        first_status_view = cast(
            dict[str, object],
            cast(dict[str, object], first_status_after_permission["result"])["status_view"],
        )
        assert first_status_view["current_permission_level"] == "read-only"

        second_status = await second._http_request(
            "POST",
            f"/api/v1/workspaces/{second.workspace_id}/management/status",
            payload={
                "request_id": "second-client-status",
                "current_session_id": second.session_id,
                "claim_version": second.claim_version,
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": second.claim_credential},
        )
        second_status_view = cast(
            dict[str, object], cast(dict[str, object], second_status["result"])["status_view"]
        )
        assert second_status_view["current_permission_level"] == "workspace-write"

        typed_effort = await first._http_request(
            "POST",
            f"/api/v1/workspaces/{first.workspace_id}/management/effort",
            payload={
                "request_id": "typed-effort",
                "current_session_id": first.session_id,
                "claim_version": first.claim_version,
                "effort": "high",
            },
            mutation=True,
            extra_headers={"X-Aide-Claim": first.claim_credential},
        )
        typed_effort_result = cast(dict[str, object], typed_effort["result"])
        assert typed_effort_result["published_effort"] == "high"

        with pytest.raises(ServiceError) as stale:
            await first._http_request(
                "GET",
                (
                    f"/api/v1/workspaces/{first.workspace_id}/management/status"
                    f"?session_id={first.session_id}&claim_version={first.claim_version}"
                ),
                extra_headers={
                    "X-Aide-Claim": "wrong-claim",
                    "X-Aide-Request": "stale-status",
                },
            )
        assert stale.value.code == "stale_claim"

        status = await first.management_dispatcher.dispatch("/status")
        assert status.handled is True
        assert status.status_view is not None
        config = await first.management_dispatcher.dispatch("/config")
        assert config.handled is True
        assert config.output is not None
        permission = await first.management_dispatcher.update_permission_level("read-only")
        assert permission.output == "Foreground permission level: read-only"
        assert permission.published_permission_level == "read-only"
        effort = await first.management_dispatcher.update_reasoning_effort("high")
        assert effort.output == "Chat reasoning effort: high"
        assert effort.published_effort == "high"

        with pytest.raises(ServiceError) as raised:
            await second.open_conversation(session_id=first.session_id)
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
                        "X-Aide-Client": first.client_id,
                    },
                )
            assert handshake.value.status == 403
            async with http.post(
                f"{first.base_url}/api/v1/clients",
                headers={
                    "Authorization": f"Bearer {first.token}",
                    "X-Aide-CSRF": first.token,
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
            "X-Aide-CSRF": client.token,
            "X-Aide-Client": client.client_id,
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
                assert registered["request_id"] == "register"
                assert registered["schedule_state"] == "available"
                assert registered["saved_jobs"] == []
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
async def test_project_http_delete_returns_operation_and_preserves_directory(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, project, port=port)
        headers = {
            "Authorization": f"Bearer {client.token}",
            "X-Aide-CSRF": client.token,
            "X-Aide-Client": client.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{client.base_url}/api/v1/projects",
                headers=headers,
                json={"request_id": "register", "path": str(project)},
            ) as response:
                assert response.status == 200
                project_id = (await response.json())["project_id"]

            async with http.delete(
                f"{client.base_url}/api/v1/projects/{project_id}",
                headers=headers,
                json={"request_id": "remove"},
            ) as response:
                assert response.status == 200
                removal = await response.json()
                assert removal["request_id"] == "remove"
                assert removal["project_id"] == project_id
                assert removal["operation_id"]
                assert removal["status"] in {"removing", "completed"}

            for _ in range(100):
                async with http.get(
                    f"{client.base_url}/api/v1/projects", headers=headers
                ) as response:
                    assert response.status == 200
                    projects = (await response.json())["projects"]
                if not projects:
                    break
                assert projects[0]["removal_operation_id"] == removal["operation_id"]
                await asyncio.sleep(0.01)
            else:
                pytest.fail("project removal did not reach its terminal state")

            for _ in range(100):
                if client.workspace_id == "":
                    break
                await asyncio.sleep(0.01)
            assert client.workspace_id == ""
            assert not client.control.foreground_input_admitted()

            async with http.get(
                f"{client.base_url}/api/v1/projects/{project_id}/removal/{removal['operation_id']}",
                headers=headers,
            ) as response:
                assert response.status == 200
                status = await response.json()
                assert status == {
                    "project_id": project_id,
                    "operation_id": removal["operation_id"],
                    "status": "completed",
                }

            async with http.delete(
                f"{client.base_url}/api/v1/projects/{project_id}",
                headers=headers,
                json={"request_id": "remove-retry"},
            ) as response:
                assert response.status == 200
                retried = await response.json()
                assert retried["project_id"] == project_id
                assert retried["operation_id"] == removal["operation_id"]
                assert retried["status"] == "completed"
        assert project.is_dir()
    finally:
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
        creation_scope="project",
        content="project-only older content",
    )
    newer_id = await _persist_session(
        project,
        home=home,
        title="Newer project session",
        created_at=datetime(2026, 2, 1, 9, tzinfo=UTC),
        creation_scope="project",
        content="project-only newer content",
    )
    offset_id = await _persist_session(
        project,
        home=home,
        title="Offset project session",
        created_at=datetime(2026, 2, 1, 10, tzinfo=timezone(timedelta(hours=8))),
        creation_scope="project",
        content="project-only offset content",
    )
    other_id = await _persist_session(
        other_project,
        home=home,
        title="Other project session",
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        creation_scope="project",
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
            "X-Aide-CSRF": first.token,
            "X-Aide-Client": first.client_id,
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

            async with http.patch(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}",
                headers={
                    **headers,
                    "X-Aide-Claim": cast(str, claim_data["reconnect_credential"]),
                },
                json={
                    "request_id": "rename-project-session",
                    "claim_version": claim_data["claim_version"],
                    "metadata_version": 0,
                    "title": "Renamed project session",
                },
            ) as response:
                assert response.status == 200
                renamed = await response.json()
            assert renamed["project_id"] == project_id
            assert renamed["session"]["title"] == "Renamed project session"
            assert renamed["session"]["metadata_version"] == 1

            second_headers = {
                "Authorization": f"Bearer {second.token}",
                "X-Aide-CSRF": second.token,
                "X-Aide-Client": second.client_id,
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
                    "X-Aide-Claim": cast(str, claim_data["reconnect_credential"]),
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "stale_claim"

            async with http.post(
                f"{first.base_url}/api/v1/projects/{project_id}/sessions/{newer_id}/release",
                headers={
                    **headers,
                    "X-Aide-Claim": cast(str, claim_data["reconnect_credential"]),
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
                    "X-Aide-Claim": cast(str, empty_claim_data["reconnect_credential"]),
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
@pytest.mark.parametrize("scope", ["workspaces", "projects"])
@pytest.mark.parametrize("limit", [None, "", "0", "-1", "1.5", "invalid", "1", "101"])
async def test_session_page_limit_preserves_optional_and_validation_contract(
    tmp_path: Path, scope: str, limit: str | None
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    directory = tmp_path / "project"
    directory.mkdir()
    for index in range(3):
        await _persist_session(
            directory,
            home=home,
            title=f"Session {index}",
            created_at=datetime(2026, 2, index + 1, tzinfo=UTC),
            content="Private history",
            creation_scope="project",
        )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    token = create_credential(home)
    server = TestServer(create_app(service))
    await service.start()
    try:
        client = await service.register_client("web")
        record, workspace, _jobs = await service.register_project(client.client_id, directory)
        identity = workspace.workspace_id if scope == "workspaces" else record.project_id
        await server.start_server()
        async with aiohttp.ClientSession() as http:
            async with http.get(
                server.make_url(f"/api/v1/{scope}/{identity}/sessions"),
                headers={"Authorization": f"Bearer {token}", "X-Aide-Client": client.client_id},
                params={} if limit is None else {"limit": limit},
            ) as response:
                body = await response.json()
                if limit in {None, "1"}:
                    assert response.status == 200
                    assert len(body["sessions"]) == (3 if limit is None else 1)
                    assert bool(body["next_cursor"]) == (limit == "1")
                    assert "Private history" not in str(body)
                else:
                    assert response.status == 422
                    assert body["code"] == "validation_error"
                    assert body["message"] == (
                        "limit must be between 1 and 100." if limit == "101" else "limit is invalid."
                    )
                    assert body["field_errors"] == {}
                    assert body["retryable"] is False
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_workspace_session_listing_filters_titles_and_pages_without_history(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    other_project = tmp_path / "other-project"
    project.mkdir()
    other_project.mkdir()
    first_id = await _persist_session(
        project,
        home=home,
        title="Build API",
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
        content="secret first project body",
    )
    second_id = await _persist_session(
        project,
        home=home,
        title="build api",
        created_at=datetime(2026, 2, 2, tzinfo=UTC),
        content="secret second project body",
    )
    third_id = await _persist_session(
        project,
        home=home,
        title="Build worker",
        created_at=datetime(2026, 2, 3, tzinfo=UTC),
        content="secret third project body",
    )
    other_id = await _persist_session(
        other_project,
        home=home,
        title="Build API",
        created_at=datetime(2026, 2, 4, tzinfo=UTC),
        content="secret other project body",
    )
    port = _free_port()
    first: ServiceClient | None = None
    other: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, project, port=port)
        other = await ServiceClient.connect_or_start(home, other_project, port=port)
        headers = {
            "Authorization": f"Bearer {first.token}",
            "X-Aide-Client": first.client_id,
        }
        async with aiohttp.ClientSession() as http:
            async with http.get(
                f"{first.base_url}/api/v1/workspaces/{first.workspace_id}/sessions",
                headers=headers,
                params={"title": "BUILD API", "limit": "1"},
            ) as response:
                assert response.status == 200
                first_page = await response.json()

            assert [item["id"] for item in first_page["sessions"]] == [second_id]
            assert first_page["sessions"][0]["metadata_version"] == 0
            assert first_page["next_cursor"]
            for params in (
                {"title": "worker", "cursor": first_page["next_cursor"]},
                {"title": "BUILD API", "cursor": "not-a-cursor"},
                {"title": "BUILD API", "limit": "0"},
                {"title": "BUILD API", "limit": "101"},
            ):
                async with http.get(
                    f"{first.base_url}/api/v1/workspaces/{first.workspace_id}/sessions",
                    headers=headers,
                    params=params,
                ) as response:
                    assert response.status == 422
                    assert (await response.json())["code"] == "validation_error"
            async with http.get(
                f"{other.base_url}/api/v1/workspaces/{other.workspace_id}/sessions",
                headers={
                    "Authorization": f"Bearer {other.token}",
                    "X-Aide-Client": other.client_id,
                },
                params={"title": "BUILD API", "cursor": first_page["next_cursor"]},
            ) as response:
                assert response.status == 422
            first_body = json.dumps(first_page)
            assert first_id not in first_body
            assert third_id not in first_body
            assert other_id not in first_body
            assert "secret" not in first_body

            async with http.get(
                f"{first.base_url}/api/v1/workspaces/{first.workspace_id}/sessions",
                headers=headers,
                params={"title": "build api", "cursor": first_page["next_cursor"]},
            ) as response:
                assert response.status == 200
                second_page = await response.json()

            assert [item["id"] for item in second_page["sessions"]] == [first_id]
            assert second_page["next_cursor"] is None

            async with http.get(
                f"{other.base_url}/api/v1/workspaces/{other.workspace_id}/sessions",
                headers={
                    "Authorization": f"Bearer {other.token}",
                    "X-Aide-Client": other.client_id,
                },
                params={"title": "build"},
            ) as response:
                assert response.status == 200
                other_listing = await response.json()

            assert [item["id"] for item in other_listing["sessions"]] == [other_id]
    finally:
        if other is not None:
            await other.close()
        if first is not None:
            await first.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_workspace_session_rename_requires_claim_and_persists_metadata_version(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = await _persist_session(
        workspace,
        home=home,
        title="Generated title",
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
        content="rename me",
    )
    port = _free_port()
    client: ServiceClient | None = None
    restarted: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        headers = {
            "Authorization": f"Bearer {client.token}",
            "X-Aide-CSRF": client.token,
            "X-Aide-Client": client.client_id,
        }
        async with aiohttp.ClientSession() as http:
            await client.open_conversation(session_id=session_id)
            claim_version = client.claim_version
            claim_credential = client.claim_credential
            claim_headers = {
                **headers,
                "X-Aide-Claim": claim_credential,
            }

            async with http.patch(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{session_id}",
                headers=claim_headers,
                json={
                    "request_id": "rename-session",
                    "claim_version": claim_version,
                    "metadata_version": 0,
                    "title": "Manual title",
                },
            ) as response:
                assert response.status == 200
                renamed = await response.json()
            assert renamed["session"]["title"] == "Manual title"
            assert renamed["session"]["metadata_version"] == 1

            rename_url = (
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{session_id}"
            )

            async def repeat_rename() -> object:
                async with http.patch(
                    rename_url,
                    headers=claim_headers,
                    json={
                        "request_id": "rename-session",
                        "claim_version": claim_version,
                        "metadata_version": 0,
                        "title": "Manual title",
                    },
                ) as response:
                    assert response.status == 200
                    return await response.json()

            assert tuple(await asyncio.gather(repeat_rename(), repeat_rename())) == (
                renamed,
                renamed,
            )
            async with http.patch(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{session_id}",
                headers=claim_headers,
                json={
                    "request_id": "rename-session",
                    "claim_version": claim_version,
                    "metadata_version": 1,
                    "title": "Reused ID must not write",
                },
            ) as response:
                assert response.status == 422
            async with http.patch(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{session_id}",
                headers={**claim_headers, "X-Aide-Claim": "invalid-claim"},
                json={
                    "request_id": "rename-session",
                    "claim_version": claim_version,
                    "metadata_version": 0,
                    "title": "Manual title",
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "stale_claim"

            async with http.patch(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{session_id}",
                headers=claim_headers,
                json={
                    "request_id": "rename-stale-version",
                    "claim_version": claim_version,
                    "metadata_version": 0,
                    "title": "Should conflict",
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "metadata_conflict"

            async with http.post(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions",
                headers=headers,
                json={"request_id": "rename-draft-create"},
            ) as response:
                assert response.status == 200
                draft = await response.json()
            draft_id = cast(str, draft["session_id"])
            await client.open_conversation(session_id=draft_id)
            draft_claim_version = client.claim_version
            draft_claim_credential = client.claim_credential
            async with http.patch(
                f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{draft_id}",
                headers={
                    **headers,
                    "X-Aide-Claim": draft_claim_credential,
                },
                json={
                    "request_id": "rename-draft",
                    "claim_version": draft_claim_version,
                    "metadata_version": 0,
                    "title": "Must not persist",
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "session_not_persisted"
            await client.release_session()
            assert not (WorkspaceState(workspace).sessions_directory / f"{draft_id}.jsonl").exists()
        await client.close()
        client = None
        await ServiceClient.stop_existing(home, port=port)
        restarted = await ServiceClient.connect_or_start(home, workspace, port=port)
        listing = await restarted.list_sessions()
        persisted = next(item for item in listing if item["id"] == session_id)
        assert persisted["title"] == "Manual title"
        assert persisted["metadata_version"] == 1
    finally:
        if restarted is not None:
            await restarted.close()
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_workspace_session_delete_requires_confirmation_and_cleans_only_session_data(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target_id = await _persist_session(
        workspace,
        home=home,
        title="Delete me",
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        content="target content",
    )
    other_id = await _persist_session(
        workspace,
        home=home,
        title="Keep me",
        created_at=datetime(2026, 3, 2, tzinfo=UTC),
        content="other content",
    )
    state = WorkspaceState(workspace)
    target_path = state.sessions_directory / f"{target_id}.jsonl"
    other_path = state.sessions_directory / f"{other_id}.jsonl"
    state.logs_directory.mkdir()
    (state.logs_directory / f"{target_id}.log").write_text("target log", encoding="utf-8")
    (state.logs_directory / f"{target_id}.1.log").write_text("rotated target log", encoding="utf-8")
    (state.logs_directory / f"{other_id}.log").write_text("other log", encoding="utf-8")
    artifact_root = state.path / "artifacts" / target_id
    artifact_root.mkdir(parents=True)
    (artifact_root / "tool.txt").write_text("target artifact", encoding="utf-8")
    other_artifact = state.path / "artifacts" / other_id
    other_artifact.mkdir(parents=True)
    (other_artifact / "tool.txt").write_text("other artifact", encoding="utf-8")
    restore_root = state.path / "restore" / target_id
    restore_root.mkdir(parents=True)
    (restore_root / "backup.bin").write_text("target restore backup", encoding="utf-8")
    unrelated = workspace / "user-owned.txt"
    unrelated.write_text("keep this file", encoding="utf-8")

    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        await client.open_conversation(session_id=target_id)
        headers = {
            "Authorization": f"Bearer {client.token}",
            "X-Aide-CSRF": client.token,
            "X-Aide-Client": client.client_id,
            "X-Aide-Claim": client.claim_credential,
        }
        delete_url = (
            f"{client.base_url}/api/v1/workspaces/{client.workspace_id}/sessions/{target_id}"
        )
        async with aiohttp.ClientSession() as http:
            async with http.delete(
                delete_url,
                headers=headers,
                json={
                    "request_id": "delete-without-confirmation",
                    "claim_version": client.claim_version,
                    "confirm": False,
                },
            ) as response:
                assert response.status == 422
                body = await response.json()
                assert body["code"] == "validation_error"
                assert body["field_errors"]["confirm"] == "must be true"
            assert target_path.exists()

            delete_body = {
                "request_id": "delete-target",
                "claim_version": client.claim_version,
                "confirm": True,
            }
            async with http.delete(delete_url, headers=headers, json=delete_body) as response:
                assert response.status == 200
                deleted = await response.json()
            assert deleted == {
                "request_id": "delete-target",
                "workspace_id": client.workspace_id,
                "session_id": target_id,
                "deleted": True,
            }
            async with http.delete(delete_url, headers=headers, json=delete_body) as response:
                assert response.status == 200
                assert await response.json() == deleted

        assert not target_path.exists()
        assert other_path.exists()
        assert not (state.logs_directory / f"{target_id}.log").exists()
        assert not (state.logs_directory / f"{target_id}.1.log").exists()
        assert (state.logs_directory / f"{other_id}.log").exists()
        assert not artifact_root.exists()
        assert (other_artifact / "tool.txt").exists()
        assert not restore_root.exists()
        assert unrelated.read_text(encoding="utf-8") == "keep this file"
        assert target_id not in {item["id"] for item in await client.list_sessions()}
        assert other_id in {item["id"] for item in await client.list_sessions()}
    finally:
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_project_session_delete_returns_project_identity(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    session_id = await _persist_session(
        project,
        home=home,
        title="Project deletion",
        created_at=datetime(2026, 3, 4, tzinfo=UTC),
        content="project target",
    )
    project_id = ProjectCatalog(home).register(project).project_id
    port = _free_port()
    client: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, project, port=port)
        await client.open_conversation(session_id=session_id)
        headers = {
            "Authorization": f"Bearer {client.token}",
            "X-Aide-CSRF": client.token,
            "X-Aide-Client": client.client_id,
            "X-Aide-Claim": client.claim_credential,
        }
        async with aiohttp.ClientSession() as http:
            async with http.delete(
                f"{client.base_url}/api/v1/projects/{project_id}/sessions/{session_id}",
                headers=headers,
                json={
                    "request_id": "delete-project-target",
                    "claim_version": client.claim_version,
                    "confirm": True,
                },
            ) as response:
                assert response.status == 200
                deleted = await response.json()
        assert deleted == {
            "request_id": "delete-project-target",
            "project_id": project_id,
            "workspace_id": client.workspace_id,
            "session_id": session_id,
            "deleted": True,
        }
    finally:
        if client is not None:
            await client.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_session_delete_requires_the_current_client_claim(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = await _persist_session(
        workspace,
        home=home,
        title="Claimed target",
        created_at=datetime(2026, 3, 5, tzinfo=UTC),
        content="claimed content",
    )
    port = _free_port()
    owner: ServiceClient | None = None
    other: ServiceClient | None = None
    try:
        owner = await ServiceClient.connect_or_start(home, workspace, port=port)
        other = await ServiceClient.connect_or_start(home, workspace, port=port)
        await owner.open_conversation(session_id=session_id)
        headers = {
            "Authorization": f"Bearer {other.token}",
            "X-Aide-CSRF": other.token,
            "X-Aide-Client": other.client_id,
            "X-Aide-Claim": owner.claim_credential,
        }
        async with aiohttp.ClientSession() as http:
            async with http.delete(
                f"{other.base_url}/api/v1/workspaces/{other.workspace_id}/sessions/{session_id}",
                headers=headers,
                json={
                    "request_id": "delete-foreign-claim",
                    "claim_version": owner.claim_version,
                    "confirm": True,
                },
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "stale_claim"
        assert WorkspaceState(workspace).sessions_directory.joinpath(f"{session_id}.jsonl").exists()
    finally:
        if other is not None:
            await other.close()
        if owner is not None:
            await owner.close()
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
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("web")
        workspace = next(iter(service._workspaces.values()))
        from types import SimpleNamespace
        monkeypatch.setattr(workspace, "_restore_result", SimpleNamespace(session_id=restored_id))
        created = await service.create_project_session(client.client_id, record.project_id)
        draft_id = cast(str, created["session_id"])
        assert draft_id != restored_id
        claim = await service.claim_project_session(client.client_id, record.project_id, draft_id)
        assert cast(dict[str, object], claim["snapshot"])["messages"] == []
        claim_data = cast(dict[str, object], claim["claim"])
        await service.release_conversation(
            client.client_id,
            workspace.workspace_id,
            draft_id,
            cast(int, claim_data["claim_version"]),
            cast(str, claim_data["reconnect_credential"]),
            project_id=record.project_id,
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
from aide.config.agent_home import AgentHome
from aide.service.client import ServiceClient

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
        assert len({response["instance_id"] for response in responses}) == 1
        assert len({response["session_id"] for response in responses}) == 2
        async with asyncio.timeout(5):
            while read_discovery(home) is not None:
                await asyncio.sleep(0.02)
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
        _client_type: type[ServiceClient],
        agent_home: AgentHome,
        path: Path,
        *,
        attach_workspace: bool = True,
    ) -> ServiceClient:
        client = await original_connect(
            agent_home,
            path,
            port=port,
            attach_workspace=attach_workspace,
        )
        connected.append(client)
        return client

    class FakeTerminalApp:
        def __init__(
            self,
            *,
            bus: RemoteMessageBus,
            control: RemoteControl,
            management_dispatcher: RemoteManagementCommandDispatcher,
            skill_metadata: tuple[object, ...],
            management_command_tokens: tuple[str, ...],
        ) -> None:
            assert isinstance(bus, RemoteMessageBus)
            assert isinstance(control, RemoteControl)
            assert skill_metadata == ()
            assert "/status" in management_command_tokens
            self.management = management_dispatcher

        def bind_confirmation_coordinator(self, coordinator: RemoteConfirmationCoordinator) -> None:
            assert isinstance(coordinator, RemoteConfirmationCoordinator)

        async def run_async(self) -> None:
            status = await self.management.submit_user_input("/status")
            assert status.handled is True
            assert status.status_view is not None

    monkeypatch.setattr(ServiceClient, "connect_or_start", classmethod(connect))
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeTerminalApp)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
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
    service = AgentService(AgentHome(tmp_path / "agent-home"), reconnect_timeout=0.02)
    await service.start()
    await asyncio.wait_for(service.wait_closed(), timeout=1)
    assert service.state == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cli", "web"])
async def test_last_real_client_disconnect_exits_service_process(
    tmp_path: Path, kind: str,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    launcher = await ServiceClient.connect_or_start(
        home, workspace, port=port, attach_workspace=False
    )
    instance_id = launcher.discovery.service_instance_id
    try:
        if kind == "cli":
            await launcher.attach_workspace(workspace)
            await launcher.close()
        else:
            async with aiohttp.ClientSession() as http:
                headers = {
                    "Authorization": f"Bearer {launcher.token}",
                    "X-Aide-CSRF": launcher.token,
                }
                async with http.post(
                    f"{launcher.base_url}/api/v1/clients",
                    headers=headers,
                    json={"request_id": str(uuid4()), "kind": "web"},
                ) as response:
                    assert response.status == 200
                    web_client = await response.json()
                socket = await http.ws_connect(
                    f"{launcher.base_url}/api/v1/events",
                    headers={
                        "Authorization": f"Bearer {launcher.token}",
                        "X-Aide-Client": web_client["client_id"],
                        "Origin": launcher.base_url,
                    },
                    protocols=("aide-v1",),
                )
                request_id = str(uuid4())
                await socket.send_json({
                    "request_id": request_id,
                    "type": "subscribe",
                    "workspace_id": None,
                    "session_id": None,
                    "claim_version": None,
                    "payload": {"last_seq": None, "stream_id": None},
                })
                async with asyncio.timeout(5):
                    while True:
                        event = await socket.receive_json()
                        if event.get("request_id") == request_id:
                            assert event["accepted"] is True
                            break
                await launcher.close()
                await socket.close()
        async with asyncio.timeout(5):
            while (
                read_discovery(home) is not None
                or credential_path(home).exists()
                or _port_is_open("127.0.0.1", port)
            ):
                await asyncio.sleep(0.02)
        reopened = await ServiceClient.connect_or_start(
            home, workspace, port=port, attach_workspace=False
        )
        try:
            assert reopened.discovery.service_instance_id != instance_id
        finally:
            await reopened.close()
    finally:
        await launcher.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_launcher_waits_for_draining_service_before_starting_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = AgentService(home)
    server = TestServer(create_app(service), host="127.0.0.1")
    await service.start()
    await server.start_server()
    assert server.port is not None
    port = server.port
    create_credential(home)
    write_discovery(home, ServiceDiscovery(
        service.service_instance_id, service.protocol_version, "127.0.0.1", port, 0,
    ))
    await service.stop()
    registration_rejected = asyncio.Event()
    original_register = service.register_client

    async def register_stopping_client(
        kind: str, reconnect_credential: str | None = None,
    ) -> object:
        try:
            return await original_register(kind, reconnect_credential=reconnect_credential)
        except ServiceError as error:
            if error.code == "admission_closed":
                registration_rejected.set()
            raise

    monkeypatch.setattr(service, "register_client", register_stopping_client)

    async def close_old_listener() -> None:
        await registration_rejected.wait()
        await server.close()

    closing = asyncio.create_task(close_old_listener())
    replacement: ServiceClient | None = None
    try:
        replacement = await ServiceClient.connect_or_start(
            home, workspace, port=port, attach_workspace=False
        )
        assert registration_rejected.is_set()
        assert replacement.discovery.service_instance_id != service.service_instance_id
    finally:
        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        await server.close()
        if replacement is not None:
            await replacement.close()
        await ServiceClient.stop_existing(home, port=port)


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
        await client.open_conversation(session_id=session.session_id)
        original_version = client.claim_version
        listing = await client.management_dispatcher.dispatch("/restore")
        assert listing.restore_listing is not None
        assert len(listing.restore_listing.anchors) == 1
        plan_result = await client.management_dispatcher.restore_inspect(1)
        assert plan_result.restore_plan is not None
        committed = await client.management_dispatcher.restore_commit(
            plan_result.restore_plan.anchor_id, RestoreMode.CONVERSATION_ONLY
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
                assert "Aide" in index
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
                assert "aide_session" in cookies
                assert "aide_csrf" not in cookies

            async with browser.post(
                f"{client.base_url}/api/v1/web/ticket",
                headers={"Origin": client.base_url},
                json={"ticket": ticket},
            ) as response:
                assert response.status == 401

            csrf = exchanged["csrf_token"]
            async with browser.post(
                f"{client.base_url}/api/v1/clients",
                headers={"Origin": client.base_url, "X-Aide-CSRF": csrf},
                json={"request_id": "browser-client", "kind": "web"},
            ) as response:
                assert response.status == 200
                web_client = await response.json()
                assert web_client["client_id"]
                control = web_client["web_control_credential"]
                assert isinstance(control, str)

            web_headers = {
                "Origin": client.base_url,
                "X-Aide-CSRF": csrf,
                "X-Aide-Control": control,
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
                protocols=("aide-v1", control),
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
                    headers={"Origin": client.base_url, "X-Aide-CSRF": csrf},
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
                        "X-Aide-Claim": claim_data["reconnect_credential"],
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
                    headers={"Origin": client.base_url, "X-Aide-CSRF": csrf},
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

            await asyncio.sleep(31)
            async with browser.post(
                f"{client.base_url}/api/v1/clients",
                headers={"Origin": client.base_url, "X-Aide-CSRF": csrf},
                json={"request_id": "browser-client-after-expiry", "kind": "web"},
            ) as response:
                assert response.status == 200, await response.text()
                replacement_client = await response.json()
                assert replacement_client["client_id"] != web_client["client_id"]

            async with browser.get(
                f"{client.base_url}/api/v1/web/session",
                headers={"Origin": client.base_url},
            ) as response:
                assert response.status == 200
                rebound_session = await response.json()
                assert rebound_session["client_id"] == replacement_client["client_id"]

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
