from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from typing import Any, cast

import pytest

from omni.agent.loop import AgentRunExecutor
from omni.agent.message_bus import InboundMessage
from omni.agent.permission import PermissionSnapshot, RuntimePermissionControl
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.service.execution import SessionExecution
from omni.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider


def _home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


@pytest.mark.asyncio
async def test_agent_service_is_the_runtime_composition_root(tmp_path: Path) -> None:
    home = _home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        executor = workspace.loops[session_id].loop

        assert type(executor) is SessionExecution
        assert not isinstance(executor, AgentRunExecutor)
        assert executor._active is None
        assert not hasattr(workspace, "runtime")
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_releasing_claim_retains_session_authority(tmp_path: Path) -> None:
    home = _home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        first = await service.claim(client.client_id, workspace.workspace_id, session_id)
        state = workspace.loops[session_id]
        authority = state.loop.session

        await workspace.release(client.client_id, session_id, close_idle=False)

        assert session_id in workspace.loops
        assert workspace.loops[session_id] is state
        assert state.loop.session is authority
        other = await service.register_client("web")
        await service.attach_workspace(other.client_id, workspace_path)
        second = await service.claim(other.client_id, workspace.workspace_id, session_id)
        assert cast(dict[str, Any], second["snapshot"])["session_id"] == session_id
        assert cast(dict[str, Any], first["claim"])["claim_version"] == 1
        assert cast(dict[str, Any], second["claim"])["claim_version"] == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_takeover_captures_current_client_permission_and_retires_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    provider = _ConcurrentProvider()
    provider.release_b.set()
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda _config: provider)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    observed: list[str] = []
    executors: list[weakref.ReferenceType[AgentRunExecutor]] = []
    original_snapshot = RuntimePermissionControl.snapshot
    original_run = AgentRunExecutor.run_foreground

    def snapshot(control: RuntimePermissionControl, shell: Any) -> PermissionSnapshot:
        result = original_snapshot(control, shell)
        observed.append(result.level)
        return result

    async def run(executor: AgentRunExecutor, inbound: InboundMessage) -> None:
        executors.append(weakref.ref(executor))
        await original_run(executor, inbound)

    monkeypatch.setattr(RuntimePermissionControl, "snapshot", snapshot)
    monkeypatch.setattr(AgentRunExecutor, "run_foreground", run)
    try:
        clients = [await service.register_client(kind) for kind in ("cli", "web")]
        sink = _CollectingSink()
        for client in clients:
            await service.connect_client(client.client_id, sink)
            client.subscribed = True
            await service.attach_workspace(client.client_id, path)
        workspace = next(iter(service.workspaces.values()))
        session_id = await workspace.create_draft(clients[0].client_id)
        authority = workspace.loops[session_id].loop.session
        for index, (client, level) in enumerate(zip(clients, ("full-access", "read-only"), strict=True)):
            service.client_permission(client.client_id).select(cast(Any, level))
            await service.claim(client.client_id, workspace.workspace_id, session_id)
            claim = workspace._claims[session_id]
            await workspace.input(client.client_id, session_id, claim.version, "warmup", str(index))
            await asyncio.wait_for(sink.wait_for("run.completed", str(index)), 5)
            await workspace.loops[session_id].loop.wait_for_restore_idle()
            await workspace.release(client.client_id, session_id)
        assert observed == ["full-access", "read-only"]
        assert workspace.loops[session_id].loop.session is authority
        assert len(executors) == 2
        gc.collect()
        assert all(reference() is None for reference in executors)
        assert len(authority.messages) == 4
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_shutdown_flushes_remaining_sessions_after_one_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, path)
    ids = [await workspace.create_draft(client.client_id, reuse_startup_session=False) for _ in range(2)]
    first, second = (workspace.loops[session_id].loop for session_id in ids)

    async def fail() -> None:
        raise OSError("injected Session flush failure")

    monkeypatch.setattr(first, "close", fail)
    with pytest.raises(Exception, match="cleanup error"):
        await service.stop()
    assert second._closed
    assert second.session._closed
    assert service.state == "stopped"


@pytest.mark.asyncio
async def test_workspace_activation_has_one_authority_for_normalized_paths(tmp_path: Path) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first, second = await asyncio.gather(
            service._get_or_create_workspace(path),
            service._get_or_create_workspace(path / ".." / "workspace"),
        )
        assert first is second
        assert len(service.workspace_resources.resources) == 1
        assert first.resources.router is service.model_router
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["expiry", "shutdown", "shutdown-with-release"])
async def test_departure_cancels_blocked_title_and_flushes_retained_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from collections.abc import AsyncIterator

    from omni.provider.models import ModelStreamEvent

    title_started, title_release, title_cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Provider(_ConcurrentProvider):
        def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
            system = kwargs["messages"][0].get("content", "")
            if not system.startswith("Generate a concise title"):
                return super().stream(**kwargs)

            async def title() -> AsyncIterator[ModelStreamEvent]:
                title_started.set()
                try:
                    await title_release.wait()
                except asyncio.CancelledError:
                    title_cancelled.set()
                    raise
                async for event in super(Provider, self).stream(**kwargs):
                    yield event

            return title()

    provider = Provider()
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda _config: provider)
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        session_id = await workspace.create_draft(client.client_id)
        await service.claim(client.client_id, workspace.workspace_id, session_id)
        authority = workspace.loops[session_id].loop.session
        claim = workspace._claims[session_id]
        await workspace.input(client.client_id, session_id, claim.version, "warmup", "run")
        await asyncio.wait_for(title_started.wait(), 5)
        await asyncio.wait_for(sink.wait_for("run.completed", "run"), 5)
        if ending == "shutdown-with-release":
            pending_release = asyncio.create_task(workspace.release(client.client_id, session_id))
            await asyncio.sleep(0)
            assert workspace._lock.locked()
        departure = workspace.expire_client(client.client_id) if ending == "expiry" else service.stop()
        departure_task = asyncio.create_task(departure)
        done, _ = await asyncio.wait((departure_task,), timeout=1)
        assert departure_task in done, "Departure waits forever for the blocked title request"
        await departure_task
        if ending == "shutdown-with-release":
            await pending_release
        assert title_cancelled.is_set()
        assert len(authority.messages) == 2
        assert authority.metadata["title"] == "warmup"
        if ending == "expiry":
            assert workspace.loops[session_id].loop.session is authority
            assert not authority._closed
            assert session_id not in workspace._claims
    finally:
        title_release.set()
        await service.stop()
