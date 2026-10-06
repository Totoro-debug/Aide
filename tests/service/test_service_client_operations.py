from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from omni.config.config import ConfigLoader
from omni.service.client import ServiceClient, ServiceStartupError
from omni.service.discovery import ServiceDiscovery, create_credential, write_discovery
from omni.service.errors import ServiceError
from omni.service.runtime import AgentService
from omni.service.transport import _WebSocketSink, create_app
from omni.terminal.conversation import TerminalConversationApp, _ConversationInput
from tests.service.test_protocol_contract import _validator
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, stack = connected_client
    original_client = client.client_id
    original_claim = client.claim_credential
    session_id = client.session_id
    reopened: list[dict[str, object]] = []
    original_http = client._http_request

    async def track_open(method: str, path: str, **kwargs: Any) -> dict[str, object]:
        if path.endswith("/conversations/open"):
            reopened.append(kwargs["payload"])
        return await original_http(method, path, **kwargs)

    monkeypatch.setattr(client, "_http_request", track_open)
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
    await _wait_until(lambda: len(snapshots) > 0 and client.control.foreground_input_admitted())
    if recovery == "short":
        assert client.client_id == original_client
        assert client.claim_credential == original_claim
        assert client.control.foreground_input_admitted()
    else:
        assert client.client_id != original_client
        assert client.session_id == session_id
        assert client.claim_credential != original_claim
        assert client.control.foreground_input_admitted()
    assert len(reopened) == (0 if recovery == "short" else 1)
    if reopened:
        assert reopened[0]["session_id"] == session_id
        assert "directory" in reopened[0] and "workspace_id" not in reopened[0]
        assert (await client.get_runtime_memory())["workspace_id"] == client.workspace_id
    active_service = service if recovery != "instance" else replacement
    result = await client.submit_user_input(f"after {recovery}")
    assert result["kind"] == "conversation_input"
    workspace = active_service.workspace(client.workspace_id)
    await _wait_until(lambda: any(
        message.get("role") == "assistant" and message.get("content") != "saved history"
        for message in cast(list[dict[str, object]], workspace.session_snapshot(session_id)["messages"])
    ))
    assert sum(message.get("content") == f"after {recovery}"
               for message in cast(list[dict[str, object]],
                                   workspace.session_snapshot(session_id)["messages"])) == 1


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
            await _wait_until(lambda: (
                "answer from session B" if complete_offline else "reconnect active run"
            ) in _visible_screen_text(app)
                and "Local service connection restored." in _visible_screen_text(app))
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


@pytest.mark.asyncio
async def test_resume_returns_complete_context_without_a_second_request(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _stack = connected_client
    workspace = service.workspace(client.workspace_id)
    target = await _persist_session(
        workspace.workspace_path, home=service.agent_home, title="Resume target",
        created_at=datetime.now(UTC), content="target history",
    )
    requests: list[str] = []
    original = client._http_request

    async def request(method: str, path: str, **kwargs: Any) -> dict[str, object]:
        requests.append(path)
        assert path.endswith("/management/resume"), "Resume made a supplemental business request"
        return await original(method, path, **kwargs)

    monkeypatch.setattr(client, "_http_request", request)
    result = await client.management_dispatcher.resume(target)
    assert result.resumed_session_id == target
    assert len(requests) == 1
    assert client.session_id == client.control.project_foreground_conversation().session_id == target
    assert service.client(client.client_id).current_session_id == target
    monkeypatch.setattr(client, "_http_request", original)
    await client.submit_user_input("after resume")
    await _wait_until(lambda: any(message.get("role") == "assistant"
                                for message in cast(list[dict[str, object]],
                                                    workspace.session_snapshot(target)["messages"])))
    assert sum(message.get("content") == "after resume"
               for message in cast(list[dict[str, object]],
                                   workspace.session_snapshot(target)["messages"])) == 1


@pytest.mark.asyncio
async def test_resume_replay_returns_the_same_claim_and_snapshot(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _stack = connected_client
    workspace = service.workspace(client.workspace_id)
    target = await _persist_session(
        workspace.workspace_path, home=service.agent_home, title="Replay target",
        created_at=datetime.now(UTC), content="replay history",
    )
    payload = {"request_id": "resume-replay", "session_id": target, "force": False}
    calls = 0
    original = service.claim

    async def claim(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "claim", claim)
    selected = client.session_id
    result = await service.handle_management(client.client_id, client.workspace_id, selected,
                                             "resume", payload)
    _validator("management_result").validate(result)
    assert cast(dict[str, object], result["claim"])["session_id"] == target
    assert cast(dict[str, object], result["snapshot"])["session_id"] == target
    replay = await service.handle_management(client.client_id, client.workspace_id, selected,
                                             "resume", payload)
    assert replay == result
    assert calls == 1
    with pytest.raises(ServiceError, match="request_id was already used"):
        await service.handle_management(client.client_id, client.workspace_id, selected,
                                        "resume", {**payload, "force": True})


@pytest.mark.asyncio
async def test_lost_resume_response_disables_old_claim_and_explicit_retry_replays(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _stack = connected_client
    workspace = service.workspace(client.workspace_id)
    target = await _persist_session(
        workspace.workspace_path, home=service.agent_home, title="Unknown resume",
        created_at=datetime.now(UTC), content="unknown result history",
    )
    original = client._http_request
    requests: list[dict[str, object]] = []

    async def lose_response(method: str, path: str, **kwargs: Any) -> dict[str, object]:
        result = await original(method, path, **kwargs)
        if path.endswith("/management/resume"):
            requests.append(kwargs["payload"])
            if len(requests) == 1:
                raise aiohttp.ServerDisconnectedError()
        return result

    monkeypatch.setattr(client, "_http_request", lose_response)
    with pytest.raises(ServiceStartupError, match="result is unknown"):
        await client.management_dispatcher.resume(target)
    assert service.client(client.client_id).current_session_id == target
    assert not client.control.foreground_input_admitted()
    assert client.claim_credential == ""
    await client.management_dispatcher.recover_conversation()
    assert requests[0] == requests[1]
    assert client.session_id == target
    assert client.control.foreground_input_admitted()
    await client.submit_user_input("after unknown resume")
    await _wait_until(lambda: not client.control.has_active_run)
    assert sum(message.get("content") == "after unknown resume"
               for message in cast(list[dict[str, object]],
                                   workspace.session_snapshot(target)["messages"])) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["expired", "instance"])
async def test_textual_resume_then_recovery_preserves_draft_and_accepts_one_input(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack], recovery: str,
) -> None:
    client, service, stack = connected_client
    directory = service.workspace(client.workspace_id).workspace_path
    target = await _persist_session(
        directory, home=service.agent_home, title="Journey",
        created_at=datetime.now(UTC), content="journey selected history",
    )
    original_client = client.client_id
    app = TerminalConversationApp(bus=client.bus, control=client.control,
                                  management_dispatcher=cast(Any, client.management_dispatcher))
    async with app.run_test(size=(100, 30)) as pilot:
        input_area = app.query_one("#conversation-input", _ConversationInput)
        await app._resume_selected_session(target, input_area)
        original_claim = client.claim_credential
        input_area.text = "恢复前的草稿 with spaces\nand another line"
        draft = input_area.text
        observer = await service.register_client("cli")
        await service.connect_client(observer.client_id, _CollectingSink())
        service.reconnect_timeout = 0.05
        assert client._socket is not None
        await client._socket.close()
        await _wait_until(lambda: not service.client(original_client).connected)
        active = service
        if recovery == "instance":
            await service.stop()
            active = AgentService(service.agent_home, ConfigLoader(service.agent_home).load_for_startup())
            await active.start()
            stack.push_async_callback(active.stop)
            server = await stack.enter_async_context(TestServer(create_app(active), host="127.0.0.1"))
            create_credential(active.agent_home)
            assert server.port is not None
            write_discovery(active.agent_home, ServiceDiscovery(
                active.service_instance_id, 1, "127.0.0.1", server.port, 0,
            ))
        try:
            await _wait_until(lambda: "Conversation Session recovered." in _visible_screen_text(app))
        except TimeoutError:
            pytest.fail(f"Recovery did not render: session={client.session_id}, "
                        f"admitted={client.control.foreground_input_admitted()}, "
                        f"screen={_visible_screen_text(app)}")
        await _wait_until(lambda: "journey selected history" in _visible_screen_text(app))
        assert client.client_id != original_client
        assert client.session_id == target
        assert input_area.text == draft
        assert "Ready" in _visible_screen_text(app)
        with pytest.raises(ServiceError) as stale:
            await client._http_request(
                "POST", f"/api/v1/workspaces/{client.workspace_id}/memory/read", mutation=True,
                payload={"request_id": "old-claim", "current_session_id": target, "claim_version": 1},
                extra_headers={"X-Omni-Claim": original_claim},
            )
        assert stale.value.code == "stale_claim"
        input_area.text = "one input after journey"
        await pilot.press("enter")
        await _wait_until(lambda: "answer from session B" in _visible_screen_text(app))
        messages = cast(list[dict[str, object]],
                        active.workspace(client.workspace_id).session_snapshot(target)["messages"])
        assert sum(message.get("content") == "one input after journey" for message in messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["missing", "occupied", "configuration"])
async def test_textual_recovery_refusal_offers_retry_and_new_session(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch, refusal: str,
) -> None:
    client, service, stack = connected_client
    directory = service.workspace(client.workspace_id).workspace_path
    target = client.session_id
    app = TerminalConversationApp(bus=client.bus, control=client.control,
                                  management_dispatcher=cast(Any, client.management_dispatcher))
    async with app.run_test(size=(100, 30)) as pilot:
        input_area = app.query_one("#conversation-input", _ConversationInput)
        input_area.text = "draft retained after refusal"
        original_client = client.client_id
        assert client._socket is not None
        await client._socket.close()
        await _wait_until(lambda: not service.client(original_client).connected)
        await service.stop()
        if refusal == "configuration":
            (service.agent_home.path / "config.toml").write_text("[broken", encoding="utf-8")
        active = AgentService(service.agent_home, None if refusal == "configuration"
                              else ConfigLoader(service.agent_home).load_for_startup())
        await active.start()
        stack.push_async_callback(active.stop)
        owner = await active.register_client("cli")
        await active.connect_client(owner.client_id, _CollectingSink())
        context: dict[str, Any] = {}
        if refusal != "configuration":
            context = cast(dict[str, Any], await active.open_conversation(
                owner.client_id, request_id="occupy-target", directory=str(directory),
                session_id=target,
            ))
            if refusal == "missing":
                claim = context["claim"]
                await active.delete_session(owner.client_id, claim["workspace_id"], target,
                                            claim["claim_version"], claim["reconnect_credential"],
                                            "delete-target")
        failures: list[ServiceError] = []
        original = active.open_conversation

        async def open_conversation(*args: Any, **kwargs: Any) -> Any:
            try:
                return await original(*args, **kwargs)
            except ServiceError as error:
                failures.append(error)
                raise

        monkeypatch.setattr(active, "open_conversation", open_conversation)
        server = await stack.enter_async_context(TestServer(create_app(active), host="127.0.0.1"))
        create_credential(active.agent_home)
        assert server.port is not None
        write_discovery(active.agent_home, ServiceDiscovery(
            active.service_instance_id, 1, "127.0.0.1", server.port, 0,
        ))
        try:
            await _wait_until(lambda: bool(app.screen.query("#conversation-recovery-retry")))
        except TimeoutError:
            pytest.fail(f"Recovery refusal did not render: failures={[e.code for e in failures]}, "
                        f"session={client.session_id}, screen={_visible_screen_text(app)}")
        await pilot.pause()
        await _wait_until(lambda: bool(failures) and failures[0].message in _visible_screen_text(app))
        assert failures and failures[0].message in _visible_screen_text(app)
        assert not client.control.foreground_input_admitted()
        assert "Ready" not in _visible_screen_text(app)
        assert input_area.text == "draft retained after refusal"
        if refusal == "occupied":
            claim = context["claim"]
            await active.release_claim(owner.client_id, claim["workspace_id"], target,
                                       claim["claim_version"], claim["reconnect_credential"])
            await pilot.click("#conversation-recovery-retry")
        elif refusal == "missing":
            await pilot.click("#conversation-recovery-new")
        else:
            await pilot.click("#conversation-recovery-retry")
            await _wait_until(lambda: len(failures) == 2)
            await pilot.pause()
            await _wait_until(lambda: bool(app.screen.query("#conversation-recovery-new")))
            await pilot.click("#conversation-recovery-new")
            await _wait_until(lambda: len(failures) == 3)
            await pilot.pause()
            assert not client.control.foreground_input_admitted()
            assert input_area.text == "draft retained after refusal"
            return
        await _wait_until(lambda: "Conversation Session recovered." in _visible_screen_text(app))
        assert client.control.foreground_input_admitted()
        assert input_area.text == "draft retained after refusal"
        assert (client.session_id == target) == (refusal == "occupied")


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "new_instance"])
async def test_pending_recovery_is_cancelled_or_discards_the_old_response(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch, ending: str,
) -> None:
    client, service, stack = connected_client
    original_client = client.client_id
    target = client.session_id
    observer = await service.register_client("cli")
    await service.connect_client(observer.client_id, _CollectingSink())
    service.reconnect_timeout = 0.05
    arrived, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = client._http_request
    old_context: dict[str, object] = {}

    async def hold_response(method: str, path: str, **kwargs: Any) -> dict[str, object]:
        result = await original(method, path, **kwargs)
        if path.endswith("/conversations/open") and not arrived.is_set():
            old_context.update(result)
            arrived.set()
            try:
                await release.wait()
            finally:
                finished.set()
        return result

    monkeypatch.setattr(client, "_http_request", hold_response)
    assert client._socket is not None
    await client._socket.close()
    await _wait_until(lambda: not service.client(original_client).connected)
    await asyncio.wait_for(arrived.wait(), timeout=5)
    assert not client.control.foreground_input_admitted()
    if ending == "close":
        await client.close()
        assert finished.is_set()
        assert client.closed
        assert not any(getattr(task.get_coro(), "__qualname__", None) == "ServiceClient._reconnect"
                       for task in asyncio.all_tasks())
        return
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
    assert client._socket is not None
    await client._socket.close()
    await _wait_until(lambda: client._socket is None)
    release.set()
    await _wait_until(lambda: client.control.foreground_input_admitted())
    assert client.discovery.service_instance_id == replacement.service_instance_id
    assert client.session_id == target
    assert client.claim_credential != cast(dict[str, object], old_context["claim"])["reconnect_credential"]
    assert (await client.get_runtime_memory())["workspace_id"] == client.workspace_id


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["occupied", "missing", "active", "restore"])
async def test_resume_refusal_preserves_context_and_force_preserves_the_old_run(
    connected_client: tuple[ServiceClient, AgentService, AsyncExitStack],
    monkeypatch: pytest.MonkeyPatch, refusal: str,
) -> None:
    client, service, _stack = connected_client
    workspace = service.workspace(client.workspace_id)
    target = await _persist_session(
        workspace.workspace_path, home=service.agent_home, title="Resume refusal",
        created_at=datetime.now(UTC), content="refusal target history",
    )
    previous = (client.workspace_id, client.session_id, client.claim_version, client.claim_credential)
    release, started = asyncio.Event(), asyncio.Event()
    if refusal == "occupied":
        owner = await service.register_client("cli")
        await service.connect_client(owner.client_id, _CollectingSink())
        await service.claim(owner.client_id, client.workspace_id, target)
    elif refusal == "missing":
        target = target.replace(target[-1], "0" if target[-1] != "0" else "1")
    elif refusal == "active":
        original = _ConcurrentProvider.stream

        def blocked(provider: _ConcurrentProvider, **kwargs: Any) -> Any:
            async def emit() -> AsyncIterator[Any]:
                started.set()
                await release.wait()
                async for event in original(provider, **kwargs):
                    yield event
            return emit()

        monkeypatch.setattr(_ConcurrentProvider, "stream", blocked)
        await client.submit_user_input("active before force")
        await asyncio.wait_for(started.wait(), timeout=5)
    else:
        anchors = cast(list[dict[str, Any]], workspace.session_snapshot(client.session_id)["restore_anchors"])
        assert anchors
        await client.inspect_restore(anchors[0]["anchor_id"])
    try:
        try:
            result = await client.management_dispatcher.resume(target)
            assert result.resumed_session_id is None
        except (ServiceError, ServiceStartupError):
            pass
        assert previous == (client.workspace_id, client.session_id,
                            client.claim_version, client.claim_credential)
        assert service.client(client.client_id).current_session_id == previous[1]
        if refusal == "active":
            resumed = await client.management_dispatcher.resume(target, force=True)
            assert resumed.resumed_session_id == client.session_id == target
            release.set()
            await _wait_until(lambda: any(message.get("role") == "assistant"
                for message in cast(list[dict[str, object]],
                                    workspace.session_snapshot(previous[1])["messages"])))
    finally:
        release.set()
        if refusal == "restore":
            await client.cancel_restore()
