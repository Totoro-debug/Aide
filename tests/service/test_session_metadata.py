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

from omni.agent.session.session import Session
from omni.config.config import ConfigLoader
from omni.service.discovery import create_credential
from omni.service.runtime import ClientState, LocalService, SessionClaim, WorkspaceServiceRuntime
from omni.service.transport import create_app
from tests.service.test_service_transport import _persist_session, _prepare_agent_home


@dataclass
class MetadataHarness:
    service: LocalService
    workspace: WorkspaceServiceRuntime
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
async def _metadata_service(tmp_path: Path) -> AsyncIterator[MetadataHarness]:
    home = _prepare_agent_home(tmp_path / "agent-home")
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
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=30)
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
                    "X-MyClaw-CSRF": token,
                    "X-MyClaw-Client": client.client_id,
                    "X-MyClaw-Claim": claim.credential,
                },
            )
    finally:
        await server.close()
        await service.stop()


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
