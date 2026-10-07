"""Session selection contracts through the production CLI service adapter."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aide.agent.permission import ToolPermissionLevel
from aide.agent.session.session import Session
from aide.config.agent_home import AgentHome
from aide.service.client import ServiceClient
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.cli_service import cli_service
from tests.service.test_service_concurrency import _client_output, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["full-access", "read-only", "workspace-write"])
async def test_cli_session_selection_preserves_workspace_resources_and_client_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, permission: ToolPermissionLevel
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    target = await _persist_session(
        directory,
        home=home,
        title="Target",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        content="Durable target input",
    )
    provider = _ConcurrentProvider()
    monkeypatch.setattr("aide.service.runtime.create_provider", lambda *_args: provider)
    async with cli_service(home) as service:
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            workspace = service.workspace(client.workspace_id)
            runtime = workspace.resources
            schedule = workspace.schedule_service
            await client.management_dispatcher.update_permission_level(permission)
            await client.management_dispatcher.update_reasoning_effort("max")
            result = await client.management_dispatcher.resume(target)
            assert result.resumed_session_id == target
            assert client.control.project_foreground_conversation().messages[0]["content"] == (
                "Durable target input"
            )
            first_claim = workspace.require_claim(client.client_id, target, client.claim_version)
            again = await client.management_dispatcher.resume(target)
            assert again.resumed_session_id == target
            assert workspace.require_claim(client.client_id, target, client.claim_version).loop is (
                first_claim.loop
            )
            assert workspace.resources is runtime
            assert workspace.schedule_service is schedule
            selection = await client.management_dispatcher.dispatch("/permission")
            assert selection.permission_selection == permission
            effort = await client.management_dispatcher.dispatch("/effort")
            assert effort.effort_selection == "max"
            await client.submit_user_input("session-b")
            assert "answer from session B" in await _client_output(client)
            loaded = Session.load(workspace.workspace_state, target)
            assert loaded.messages[0]["content"] == "Durable target input"
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_cli_failed_resume_leaves_current_claim_and_management_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    target = await _persist_session(
        directory,
        home=home,
        title="Target",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        content="Durable target input",
    )
    provider = _ConcurrentProvider()
    monkeypatch.setattr("aide.service.runtime.create_provider", lambda *_args: provider)
    async with cli_service(home):
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            previous = client.session_id
            path = directory / ".aide" / "sessions" / f"{target}.jsonl"
            path.write_bytes(b"PRIVATE_MALFORMED_HISTORY")
            result = await client.management_dispatcher.resume(target)
            assert result.resumed_session_id is None
            assert result.output is not None and "PRIVATE_MALFORMED_HISTORY" not in result.output
            assert client.session_id == previous
            status = await client.management_dispatcher.dispatch("/status")
            assert status.handled and status.status_view is not None
            await client.submit_user_input("session-b")
            assert "answer from session B" in await _client_output(client)
        finally:
            await client.close()
