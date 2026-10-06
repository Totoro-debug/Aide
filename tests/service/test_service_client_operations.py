from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from omni.config.config import ConfigLoader
from omni.service.client import ServiceClient, ServiceStartupError
from omni.service.discovery import ServiceDiscovery, create_credential, write_discovery
from omni.service.runtime import AgentService
from omni.service.transport import _WebSocketSink, create_app
from omni.terminal.conversation import TerminalConversationApp, _ConversationInput
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home
from tests.terminal.test_conversation import _visible_screen_text


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.02)


@pytest_asyncio.fixture
async def connected_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[ServiceClient, AgentService, AsyncExitStack]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = await _persist_session(
        workspace, home=home, title="Recovery", created_at=datetime.now(UTC), content="saved history",
    )
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda *_args: _ConcurrentProvider())
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(service.stop)
        server = await stack.enter_async_context(TestServer(create_app(service), host="127.0.0.1"))
        create_credential(home)
        assert server.port is not None
        write_discovery(home, ServiceDiscovery(service.service_instance_id, 1, "127.0.0.1", server.port, 0))
        client = await ServiceClient.connect_or_start(home, workspace)
        stack.push_async_callback(client.close)
        await client.open_conversation(session_id=session_id)
        yield client, service, stack


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["short", "expired", "instance"])
async def test_client_recovers_connection_and_never_reuses_expired_claim(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack], recovery: str,
) -> None:
    client, service, stack = connected_client
    original_client = client.client_id
    original_claim = client.claim_credential
    session_id = client.session_id
    directory = str(service.workspace(client.workspace_id).workspace_path)
    snapshots: list[Mapping[str, object]] = []
    client.add_state_listener(lambda event: snapshots.append(event)
                              if event["type"] == "snapshot.required" else None)
    observer = await service.register_client("cli")
    await service.connect_client(observer.client_id, _CollectingSink())
    if recovery == "expired":
        service.reconnect_timeout = 0.05
    socket = client._socket
    assert socket is not None
    await socket.close()
    await _wait_until(lambda: not service.client(original_client).connected)
    if recovery == "instance":
        await service.stop()
        replacement = AgentService(service.agent_home, ConfigLoader(service.agent_home).load_for_startup())
        await replacement.start()
        stack.push_async_callback(replacement.stop)
        server = await stack.enter_async_context(TestServer(create_app(replacement), host="127.0.0.1"))
        create_credential(service.agent_home)
        assert server.port is not None
        write_discovery(service.agent_home, ServiceDiscovery(
            replacement.service_instance_id, 1, "127.0.0.1", server.port, 0,
        ))
    await _wait_until(lambda: len(snapshots) > 0 and client._socket is not None)
    if recovery == "short":
        assert client.client_id == original_client
        assert client.claim_credential == original_claim
        assert client.control.foreground_input_admitted()
        assert (await client.get_runtime_memory())["workspace_id"] == client.workspace_id
    else:
        assert client.client_id != original_client
        assert client.claim_version == 0 and client.claim_credential == ""
        assert not client.control.foreground_input_admitted()
        with pytest.raises(ServiceStartupError, match="current Conversation Claim"):
            await client.get_runtime_memory()
        await client.open_conversation(directory=directory, session_id=session_id)
        assert client.claim_credential != original_claim
        assert client.control.foreground_input_admitted()


@pytest.mark.asyncio
async def test_state_subscription_discards_old_duplicate_events_and_recovers_gap(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
) -> None:
    client, service, _stack = connected_client
    received: list[Mapping[str, object]] = []
    remove_listener = client.add_state_listener(received.append)
    await service.emit("test.state", workspace_id=None, session_id=None, run_id=None, payload={})
    await _wait_until(lambda: len(received) == 1)
    event = dict(received[0])
    sink = service.client(client.client_id).sink
    assert sink is not None
    await sink.send_event(event)
    await sink.send_event({**event, "service_instance_id": "old-instance", "seq": 999})
    await sink.send_event({**event, "stream_id": "old-stream", "seq": 999})
    await sink.send_event({**event, "seq": int(str(event["seq"])) + 2})
    await _wait_until(lambda: any(value["type"] == "snapshot.required" for value in received))
    assert [value["type"] for value in received] == ["test.state", "snapshot.required"]
    assert client.control.foreground_input_admitted()
    remove_listener()


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_offline", [False, True])
async def test_textual_recovers_active_run_and_keeps_unsent_draft(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch, complete_offline: bool,
) -> None:
    client, service, _stack = connected_client
    started, release = asyncio.Event(), asyncio.Event()
    original_stream = _ConcurrentProvider.stream

    def blocked_stream(provider: _ConcurrentProvider, **kwargs: Any) -> Any:
        async def emit() -> AsyncIterator[Any]:
            if not str(kwargs["messages"][0].get("content", "")).startswith("Generate a concise title"):
                started.set()
                await release.wait()
            async for event in original_stream(provider, **kwargs):
                yield event
        return emit()

    monkeypatch.setattr(_ConcurrentProvider, "stream", blocked_stream)
    app = TerminalConversationApp(
        bus=client.bus, control=client.control,
        management_dispatcher=cast(Any, client.management_dispatcher),
    )
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*list("reconnect active run"), "enter")
            await asyncio.wait_for(started.wait(), timeout=5)
            text_area = app.query_one("#conversation-input", _ConversationInput)
            if not complete_offline:
                text_area.text = "queued after recovery"
                await pilot.press("enter")
                await _wait_until(lambda: len(app._consumed_runs) == 2)
            text_area.text = "unsent draft survives"
            assert client._socket is not None
            await client._socket.close()
            await _wait_until(lambda: not service.client(client.client_id).connected)
            if complete_offline:
                release.set()
                state = service.workspace(client.workspace_id)._loops[client.session_id]
                await _wait_until(lambda: not state.live_runs)
            service.client(client.client_id).events.clear()
            await _wait_until(lambda: client._socket is not None
                              and client.control.foreground_input_admitted())
            await pilot.pause()
            assert "Turn failed." not in _visible_screen_text(app)
            if not complete_offline:
                assert "reconnect active run" in _visible_screen_text(app)
                assert len(app._consumed_runs) == 2
                assert app._consumed_runs[0].started
                assert not app._consumed_runs[1].started
            release.set()
            await _wait_until(lambda: not app._consumed_runs)
            await pilot.pause()
            assert "answer from session B" in _visible_screen_text(app)
            assert text_area.text == "unsent draft survives"
    finally:
        release.set()


@pytest.mark.asyncio
async def test_recovery_retries_when_socket_closes_immediately_after_subscribe_ack(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _service, _stack = connected_client
    original = _WebSocketSink.send_json
    subscriptions: list[Mapping[str, object]] = []

    async def close_after_ack(sink: _WebSocketSink, value: Mapping[str, object]) -> None:
        await original(sink, value)
        result = value.get("result")
        if isinstance(result, dict) and result.get("subscribed") is True:
            subscriptions.append(value)
            if len(subscriptions) == 1:
                await sink.socket.close()

    monkeypatch.setattr(_WebSocketSink, "send_json", close_after_ack)
    assert client._socket is not None
    await client._socket.close()
    await _wait_until(lambda: len(subscriptions) >= 2 and client.control.foreground_input_admitted())
    assert client._socket is not None and not client._socket.closed


@pytest.mark.asyncio
async def test_unknown_input_result_is_not_resent_during_reconnection(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _stack = connected_client
    original = service.handle_command
    submissions: list[Mapping[str, object]] = []

    async def lose_ack(client_id: str, command: Mapping[str, object]) -> dict[str, object]:
        result = await original(client_id, command)
        if command.get("type") == "input":
            submissions.append(command)
            assert client._socket is not None
            await client._socket.close()
        return result

    monkeypatch.setattr(service, "handle_command", lose_ack)
    with pytest.raises(ServiceStartupError, match="connection closed"):
        await client.submit_user_input("accepted once")
    await _wait_until(lambda: client._socket is not None and client.control.foreground_input_admitted())
    assert len(submissions) == 1


@pytest.mark.asyncio
async def test_named_session_operations_send_current_claim_and_return_operation_dtos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(ServiceClient)
    client.workspace_id = "workspace-1"
    client.session_id = "session-1"
    client.claim_version = 4
    client.claim_credential = "claim-secret"
    calls: list[tuple[str, str, dict[str, object], bool, Mapping[str, str]]] = []

    async def fake_http_request(
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        mutation: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        assert payload is not None
        headers = {} if extra_headers is None else extra_headers
        calls.append((method, path, payload, mutation, headers))
        response: dict[str, object] = {
            "request_id": payload["request_id"],
            "workspace_id": client.workspace_id,
        }
        if path.endswith("/memory/read"):
            response["content"] = "memory contents"
        elif path.endswith("/memory/dream"):
            response["result"] = {"status": "complete"}
        elif path.endswith("/skills/reload"):
            response["skills"] = []
        elif path.endswith("/runtime/status"):
            response["status"] = {"version": "test"}
        else:
            raise AssertionError(f"Unexpected operation path: {path}")
        return response

    monkeypatch.setattr(client, "_http_request", fake_http_request)

    memory = await client.get_runtime_memory()
    dream = await client.run_dream()
    skills = await client.reload_runtime_skills()
    status = await client.get_runtime_status()

    assert memory["content"] == "memory contents"
    assert dream["result"] == {"status": "complete"}
    assert skills["skills"] == []
    assert status["status"] == {"version": "test"}
    assert [call[1] for call in calls] == [
        "/api/v1/workspaces/workspace-1/memory/read",
        "/api/v1/workspaces/workspace-1/memory/dream",
        "/api/v1/workspaces/workspace-1/skills/reload",
        "/api/v1/workspaces/workspace-1/runtime/status",
    ]
    for method, _path, payload, mutation, headers in calls:
        assert method == "POST"
        assert mutation is True
        assert payload["current_session_id"] == "session-1"
        assert payload["claim_version"] == 4
        assert isinstance(payload["request_id"], str)
        assert headers == {"X-Omni-Claim": "claim-secret"}
