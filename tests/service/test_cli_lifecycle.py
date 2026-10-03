"""CLI runtime ownership and preparation failures through the shared service."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from omni.agent.loop import AgentLoop
from omni.agent.session.session import Session
from omni.agent.workspace_runtime import WorkspaceRuntime
from omni.config.agent_home import AgentHome
from omni.service.client import ServiceClient, ServiceStartupError
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.cli_service import cli_service
from tests.service.test_service_concurrency import _client_output, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["constructor", "binding", "preflight", "start"])
async def test_cli_failed_target_preparation_preserves_selected_claim_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda *_args: provider)
    target = await _persist_session(
        directory,
        home=home,
        title="Target",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        content="Target input",
    )
    async with cli_service(home) as service:
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            previous = client.session_id
            workspace = service.workspace(client.workspace_id)
            runtime = workspace.runtime
            method = {
                "constructor": "__init__",
                "binding": "bind_confirmation_requester",
                "preflight": "preflight",
                "start": "start",
            }[failure_point]
            private = "PRIVATE_PREPARATION_PATH_AND_TOKEN"

            def fail(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(private)

            async def fail_start(_loop: AgentLoop) -> None:
                raise RuntimeError(private)

            with monkeypatch.context() as patched:
                patched.setattr(AgentLoop, method, fail_start if method == "start" else fail)
                with pytest.raises(ServiceStartupError) as raised:
                    await client.management_dispatcher.resume(target)
                assert private not in raised.value.message
                assert client.session_id == previous
                workspace.require_claim(
                    client.client_id, previous, client.claim_version, client.claim_credential
                )
                assert workspace.runtime is runtime
                status = await client.management_dispatcher.dispatch("/status")
                assert status.handled and status.status_view is not None
            selected = await client.management_dispatcher.resume(target)
            assert selected.resumed_session_id == target
            assert client.control.project_foreground_conversation().session_id == target
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_cli_disconnect_leaves_other_client_runtime_and_schedule_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda *_args: provider)
    async with cli_service(home) as service:
        first = await ServiceClient.connect_or_start(home, directory)
        second = await ServiceClient.connect_or_start(home, directory)
        workspace = service.workspace(first.workspace_id)
        runtime = workspace.runtime
        schedule = workspace.schedule_service
        await first.close()
        try:
            assert workspace.runtime is runtime
            assert workspace.schedule_service is schedule
            assert service.state == "ready"
            await second.submit_input("session-b")
            assert "answer from session B" in await _client_output(second)
            status = await second.management_dispatcher.dispatch("/status")
            assert status.status_view is not None
        finally:
            await second.close()
    assert not service.workspaces or all(
        workspace._closed for workspace in service.workspaces.values()
    )
    assert not WorkspaceRuntime._registry


@pytest.mark.asyncio
async def test_same_cli_session_waits_for_queued_snapshot_and_keeps_live_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.setattr(
        "omni.service.runtime.create_provider", lambda *_args: _ConcurrentProvider()
    )
    async with cli_service(home) as service:
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            workspace = service.workspace(client.workspace_id)
            claim = workspace.require_claim(
                client.client_id, client.session_id, client.claim_version
            )
            session = claim.loop.session
            started, release = asyncio.Event(), asyncio.Event()
            persist_after = session._persist_after

            async def blocked(previous: asyncio.Task[None] | None, content: bytes) -> None:
                started.set()
                await release.wait()
                await persist_after(previous, content)

            monkeypatch.setattr(session, "_persist_after", blocked)
            session.commit_agent_run(
                [{"role": "user", "content": "Durable before resume"}],
                pending_last_compacted=0,
                pending_action_summary=None,
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            resume = asyncio.create_task(client.management_dispatcher.resume(client.session_id))
            try:
                await asyncio.sleep(0)
                assert not resume.done()
            finally:
                release.set()
            result = await asyncio.wait_for(resume, timeout=5)
            assert result.resumed_session_id == session.session_id
            current = workspace.require_claim(
                client.client_id, client.session_id, client.claim_version
            )
            assert current.loop is claim.loop
            loaded = Session.load(workspace.workspace_state, session.session_id)
            assert loaded.messages == session.messages
            assert (
                cast(
                    dict[str, object], client.control.project_foreground_conversation().messages[0]
                )["content"]
                == "Durable before resume"
            )
        finally:
            await client.close()
