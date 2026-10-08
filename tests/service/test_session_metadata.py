"""Exercise Session metadata mutations through the real HTTP transport."""

import asyncio
import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from aide.agent.session.session import Session
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.session_configuration import SessionModelConfiguration
from aide.service.discovery import create_credential
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService, ClientState, SessionClaim, WorkspaceRecord
from aide.service.transport import create_app
from tests.service.test_service_transport import _persist_session, _prepare_agent_home


@dataclass
class MetadataHarness:
    service: AgentService
    workspace: WorkspaceRecord
    client: ClientState
    claim: SessionClaim
    http: aiohttp.ClientSession
    server: TestServer
    headers: dict[str, str]

    @property
    def session(self) -> Session:
        return self.claim.loop.session

    @property
    def path(self) -> Path:
        return self.session.workspace_state.sessions_directory / f"{self.session.session_id}.jsonl"

    @property
    def route(self) -> str:
        return (
            f"/api/v1/workspaces/{self.workspace.workspace_id}/sessions/{self.session.session_id}"
        )

    async def patch(
        self, *, request_id: str = "rename-1", route: str | None = None
    ) -> tuple[int, dict[str, Any]]:
        async with self.http.patch(
            self.server.make_url(self.route if route is None else route),
            headers=self.headers,
            json={
                "request_id": request_id,
                "claim_version": self.claim.version,
                "expected_metadata_version": 0,
                "title": "Manual HTTP title",
            },
        ) as response:
            return response.status, await response.json()


@asynccontextmanager
async def _metadata_service(
    tmp_path: Path,
    *,
    agent_home: AgentHome | None = None,
) -> AsyncIterator[MetadataHarness]:
    home = agent_home or _prepare_agent_home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    session_id = await _persist_session(
        directory,
        home=home,
        title="Original title",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        content="Private conversation body",
    )
    token = create_credential(home)
    active_configuration = ConfigLoader(home).load_for_startup()
    service = AgentService(home, active_configuration, reconnect_timeout=30)
    await service.start()
    server = TestServer(create_app(service), host="127.0.0.1")
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, directory)
        claim = await workspace.claim(client.client_id, session_id)
        await server.start_server()
        async with aiohttp.ClientSession() as http:
            yield MetadataHarness(
                service,
                workspace,
                client,
                claim,
                http,
                server,
                {
                    "Authorization": f"Bearer {token}",
                    "X-Aide-CSRF": token,
                    "X-Aide-Client": client.client_id,
                    "X-Aide-Claim": claim.credential,
                },
            )
    finally:
        await server.close()
        await service.stop()


def _session_model_test_configuration(home: AgentHome) -> None:
    (home.path / "config.toml").write_text(
        """[models.providers.first-provider]
protocol = "openai-compatible"
base_url = "https://first.example/v1"
api_key = "first-secret"
models = ["shared-model"]

[models.providers.first-provider.model_context_windows]
shared-model = 32000

[models.providers.second-provider]
protocol = "anthropic"
base_url = "https://second.example/v1"
api_key = "second-secret"
models = ["shared-model"]

[models.providers.second-provider.model_context_windows]
shared-model = 16000

[models.routes.chat]
provider_id = "first-provider"
model = "shared-model"
context_window = 32000
max_output = 4096
temperature = 0.2
reasoning_effort = "mid"
timeout = 60
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_session_model_configuration_is_claimed_versioned_and_persisted(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    _session_model_test_configuration(home)
    async with _metadata_service(tmp_path, agent_home=home) as harness:
        command_payload = {
            "expected_model_configuration_version": 0,
            "provider_id": "second-provider",
            "model": "shared-model",
            "reasoning_effort": "max",
        }
        command = {
            "request_id": "configure-session-model",
            "type": "session_model_configure",
            "workspace_id": harness.workspace.workspace_id,
            "session_id": harness.session.session_id,
            "claim_version": harness.claim.version,
            "payload": command_payload,
        }

        acknowledgement = await harness.service.handle_command(
            harness.client.client_id,
            command,
        )

        expected = {
            "provider_id": "second-provider",
            "model": "shared-model",
            "reasoning_effort": "max",
        }
        assert acknowledgement["accepted"] is True
        assert acknowledgement["result"] == {
            "model_configuration": expected,
            "model_configuration_version": 1,
        }
        assert harness.session.metadata_version == 0
        assert harness.session.metadata["model_configuration_version"] == 1
        assert harness.session.metadata["model_configuration"] == expected

        persisted = Session.load(harness.session.workspace_state, harness.session.session_id)
        assert persisted.metadata["model_configuration"] == expected
        assert persisted.metadata["model_configuration_version"] == 1
        event = next(
            event
            for event in reversed(harness.client.events)
            if event["type"] == "session.model_configuration"
        )
        assert event["payload"] == {
            **expected,
            "model_configuration_version": 1,
        }

        stale_command = {
            **command,
            "request_id": "configure-session-model-stale",
            "payload": {
                **command_payload,
                "provider_id": "first-provider",
                "expected_model_configuration_version": 0,
            },
        }
        with pytest.raises(ServiceError) as error:
            await harness.service.handle_command(harness.client.client_id, stale_command)
        assert error.value.code == "model_configuration_conflict"
        assert error.value.status == 409
        assert harness.session.model_configuration is not None
        assert harness.session.model_configuration.to_dict() == expected
        assert harness.session.model_configuration_version == 1
        assert (
            len(
                [
                    event
                    for event in harness.client.events
                    if event["type"] == "session.model_configuration"
                ]
            )
            == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    ["unknown_model", "stale_claim", "missing_claim", "invalid_effort", "legacy_effort"],
)
async def test_session_model_configuration_rejects_invalid_updates_without_mutation(
    tmp_path: Path,
    invalid: str,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    _session_model_test_configuration(home)
    async with _metadata_service(tmp_path, agent_home=home) as harness:
        before = harness.path.read_bytes()
        payload = {
            "expected_model_configuration_version": 0,
            "provider_id": "second-provider",
            "model": "unknown" if invalid == "unknown_model" else "shared-model",
            "reasoning_effort": (
                "unknown"
                if invalid == "invalid_effort"
                else "medium"
                if invalid == "legacy_effort"
                else "high"
            ),
        }
        command = {
            "request_id": f"invalid-{invalid}",
            "type": "session_model_configure",
            "workspace_id": harness.workspace.workspace_id,
            "session_id": harness.session.session_id,
            "claim_version": harness.claim.version + (1 if invalid == "stale_claim" else 0),
            "payload": payload,
        }
        if invalid == "missing_claim":
            await harness.workspace.release(harness.client.client_id, harness.session.session_id)
        with pytest.raises(ServiceError) as error:
            await harness.service.handle_command(harness.client.client_id, command)
        assert (
            error.value.code
            == {
                "unknown_model": "model_unavailable",
                "stale_claim": "stale_claim",
                "missing_claim": "stale_claim",
                "invalid_effort": "validation_error",
                "legacy_effort": "validation_error",
            }[invalid]
        )
        assert harness.session.model_configuration is None
        assert harness.session.model_configuration_version == 0
        assert harness.path.read_bytes() == before


@pytest.mark.asyncio
async def test_removed_session_model_keeps_history_readable_and_rejects_input(
    tmp_path: Path,
) -> None:
    async with _metadata_service(tmp_path) as harness:
        selection = SessionModelConfiguration("removed-provider", "removed-model", "high")
        harness.session.configure_model_durably(selection, expected_version=0)
        before = harness.path.read_bytes()
        snapshot = harness.workspace.session_snapshot(harness.session.session_id)
        assert snapshot["model_configuration"] == selection.to_dict()
        assert snapshot["messages"]
        status = harness.claim.loop.runtime_status_input()
        assert status.model_configuration_available is False
        assert status.context_window > 0
        assert harness.session.model_configuration == selection
        async with harness.http.post(
            harness.server.make_url(
                f"/api/v1/workspaces/{harness.workspace.workspace_id}/management/status"
            ),
            headers=harness.headers,
            json={
                "request_id": "removed-model-status",
                "current_session_id": harness.session.session_id,
                "claim_version": harness.claim.version,
            },
        ) as response:
            assert response.status == 200
            view = (await response.json())["result"]["status_view"]
            assert view["model_configuration_available"] is False
            assert view["current_permission_level"] == "workspace-write"
        with pytest.raises(ServiceError) as error:
            await harness.service.handle_command(
                harness.client.client_id,
                {
                    "request_id": "removed-model-input",
                    "type": "input",
                    "workspace_id": harness.workspace.workspace_id,
                    "session_id": harness.session.session_id,
                    "claim_version": harness.claim.version,
                    "payload": {"text": "must be rejected"},
                },
            )
        assert error.value.code == "model_unavailable"
        assert error.value.status == 422
        assert harness.path.read_bytes() == before
        assert not any(event["type"] == "input.accepted" for event in harness.client.events)


@pytest.mark.asyncio
async def test_failed_http_rename_preserves_state_and_same_request_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _metadata_service(tmp_path) as harness:
        session = harness.session
        before_metadata = copy.deepcopy(session.metadata)
        before_updated_at = session.updated_at
        before_raw = harness.path.read_bytes()
        before_events = tuple(harness.client.events)
        write_content = session._write_content

        def fail_write(content: bytes) -> None:
            raise OSError("Controlled metadata failure")

        monkeypatch.setattr(session, "_write_content", fail_write)
        status, error = await harness.patch()
        assert status == 500
        assert error["code"] == "persistence_error"
        assert "Controlled metadata failure" not in str(error)
        assert session.metadata == before_metadata
        assert session.updated_at == before_updated_at
        assert harness.path.read_bytes() == before_raw
        assert tuple(harness.client.events) == before_events

        monkeypatch.setattr(session, "_write_content", write_content)
        status, result = await harness.patch()
        assert status == 200
        assert result["session"]["metadata_version"] == 1
        loaded = Session.load(session.workspace_state, session.session_id)
        assert loaded.metadata["title"] == "Manual HTTP title"
        assert loaded.has_manual_title


@pytest.mark.asyncio
async def test_concurrent_duplicate_http_renames_return_same_ack_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _metadata_service(tmp_path) as harness:
        writes: list[bytes] = []
        write_content = harness.session._write_content

        def record_write(content: bytes) -> None:
            writes.append(content)
            write_content(content)

        monkeypatch.setattr(harness.session, "_write_content", record_write)
        first, second = await asyncio.gather(harness.patch(), harness.patch())
        assert first[0] == second[0] == 200
        assert first[1] == second[1]
        assert len(writes) == 1
        assert harness.session.metadata_version == 1
        assert (
            len(
                [
                    event
                    for event in harness.client.events
                    if event["type"] == ("session.metadata_updated")
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_concurrent_distinct_http_renames_with_same_version_commit_once(tmp_path: Path) -> None:
    async with _metadata_service(tmp_path) as harness:
        results = await asyncio.gather(
            harness.patch(request_id="first-rename"),
            harness.patch(request_id="second-rename"),
        )
        assert sorted(status for status, _body in results) == [200, 409]
        conflict = next(body for status, body in results if status == 409)
        assert conflict["code"] == "metadata_conflict"
        assert harness.session.metadata_version == 1
        loaded = Session.load(harness.session.workspace_state, harness.session.session_id)
        assert loaded.metadata == harness.session.metadata
        assert loaded.has_manual_title


@pytest.mark.asyncio
async def test_http_rename_rechecks_automatic_title_version_after_snapshot_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _metadata_service(tmp_path) as harness:
        started, release = asyncio.Event(), asyncio.Event()
        session = harness.session
        wait_for_pending = session.wait_for_pending_persist

        async def blocked_drain() -> None:
            started.set()
            await release.wait()
            await wait_for_pending()

        monkeypatch.setattr(session, "wait_for_pending_persist", blocked_drain)
        request = asyncio.create_task(harness.patch())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            session.update_automatic_title("Automatic winner")
            session.persist()
        finally:
            release.set()
        status, error = await asyncio.wait_for(request, timeout=5)
        assert status == 409
        assert error["code"] == "metadata_conflict"
        loaded = Session.load(session.workspace_state, session.session_id)
        assert loaded.metadata["title"] == "Automatic winner"
        assert loaded.metadata_version == 1
        assert not loaded.has_manual_title


@pytest.mark.asyncio
async def test_http_rename_revalidates_claim_after_snapshot_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _metadata_service(tmp_path) as harness:
        started = asyncio.Event()
        release = asyncio.Event()
        session = harness.session
        wait_for_pending = session.wait_for_pending_persist
        before_metadata = copy.deepcopy(session.metadata)
        before_raw = harness.path.read_bytes()
        before_events = tuple(harness.client.events)

        async def blocked_drain() -> None:
            started.set()
            await release.wait()
            await wait_for_pending()

        monkeypatch.setattr(session, "wait_for_pending_persist", blocked_drain)
        request = asyncio.create_task(harness.patch())
        try:
            await started.wait()
            harness.workspace.set_client_connection(harness.client.client_id, connected=False)
        finally:
            release.set()
        status, error = await request
        assert status == 409
        assert error["code"] == "stale_claim"
        assert session.metadata == before_metadata
        assert harness.path.read_bytes() == before_raw
        assert tuple(harness.client.events) == before_events


@pytest.mark.asyncio
async def test_wrong_project_or_workspace_cannot_rename_another_sessions_metadata(
    tmp_path: Path,
) -> None:
    async with _metadata_service(tmp_path) as harness:
        other_directory = tmp_path / "other-project"
        other_directory.mkdir()
        record, other_workspace, _jobs = await harness.service.register_project(
            harness.client.client_id, other_directory
        )
        before_raw = harness.path.read_bytes()
        before_metadata = copy.deepcopy(harness.session.metadata)
        routes = [
            f"/api/v1/projects/{record.project_id}/sessions/{harness.session.session_id}",
            f"/api/v1/workspaces/{other_workspace.workspace_id}/sessions/"
            f"{harness.session.session_id}",
        ]
        for index, route in enumerate(routes):
            status, error = await harness.patch(request_id=f"wrong-scope-{index}", route=route)
            assert status == 409
            assert error["code"] == "stale_claim"
            assert "Private conversation body" not in str(error)
        assert harness.session.metadata == before_metadata
        assert harness.path.read_bytes() == before_raw
