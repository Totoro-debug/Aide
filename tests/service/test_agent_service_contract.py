from __future__ import annotations

import asyncio
import gc
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from omni.agent.loop import AgentRunExecutor
from omni.agent.message_bus import InboundMessage
from omni.agent.permission import PermissionSnapshot, RuntimePermissionControl
from omni.agent.session.session import Session
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.service.errors import ServiceError
from omni.service.execution import SessionExecution
from omni.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock
from tests.service.test_protocol_contract import _validator
from tests.service.test_service_concurrency import (
    _claim_version,
    _CollectingSink,
    _ConcurrentProvider,
)


def _home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


@pytest.mark.asyncio
async def test_backpressured_run_survives_reconnect_at_29_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    provider = _ConcurrentProvider(early_a_delta=True)
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda _config: provider)
    clock = FakeClock(datetime(2026, 10, 4, tzinfo=UTC))
    timer_started, wake_timer = asyncio.Event(), asyncio.Event()
    blocked, release_output = asyncio.Event(), asyncio.Event()
    output_delivered = asyncio.Event()

    async def sleep(_seconds: float) -> None:
        timer_started.set()
        await wake_timer.wait()

    service = AgentService(
        home,
        ConfigLoader(home).load_for_startup(),
        reconnect_timeout=30,
        monotonic_now=clock.monotonic,
        sleep=sleep,
    )
    sink, restored_sink = _CollectingSink(), _CollectingSink()
    send_event = sink.send_event

    async def slow_output(event: dict[str, object]) -> None:
        if event.get("type") == "run.output":
            message = cast(dict[str, Any], event["payload"])["message"]
            if message["metadata"].get("_stream_delta") is True:
                blocked.set()
                await release_output.wait()
        await send_event(event)
        if blocked.is_set():
            output_delivered.set()

    monkeypatch.setattr(sink, "send_event", slow_output)
    try:
        await service.start()
        client, other = await service.register_client("cli"), await service.register_client("web")
        await service.connect_client(client.client_id, sink)
        await service.connect_client(other.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, path)
        await service.attach_workspace(other.client_id, path)
        session_id = await workspace.create_draft(client.client_id)
        claimed = await service.claim(client.client_id, workspace.workspace_id, session_id)
        state = workspace.loops[session_id]
        authority = state.loop.session
        await workspace.input(
            client.client_id, session_id, _claim_version(claimed), "session-a", "run-a"
        )
        await asyncio.wait_for(blocked.wait(), 3)
        await service.disconnect_client(client.client_id, sink=sink)
        expiry = client.disconnect_task
        assert expiry is not None
        await asyncio.wait_for(timer_started.wait(), 3)
        clock.advance(29)
        assert workspace._claims[session_id].status == "reconnecting"
        with pytest.raises(ServiceError) as occupied:
            await service.claim(other.client_id, workspace.workspace_id, session_id)
        assert occupied.value.code == "session_claimed"
        assert await service.register_client("cli", client.reconnect_credential) is client
        release_output.set()
        await asyncio.wait_for(output_delivered.wait(), 3)
        await service.connect_client(client.client_id, restored_sink)
        await asyncio.wait_for(expiry, 3)
        assert workspace._claims[session_id].status == "claimed"
        restored_claim = await service.claim(client.client_id, workspace.workspace_id, session_id)
        assert restored_claim["claim"] == claimed["claim"]
        assert state.loop.session is authority
        clock.advance(1)
        wake_timer.set()
        provider.release_a.set()
        await asyncio.wait_for(restored_sink.wait_for("run.completed", "run-a"), 3)
        processor = state.processor_task
        if processor is not None:
            await asyncio.wait_for(asyncio.shield(processor), 3)
        await state.loop.wait_for_restore_idle()
        assert not provider.session_a_cancelled.is_set()
        assert not any(event["type"] == "run.cancelled" for event in restored_sink.events)
        outputs = [
            cast(dict[str, Any], event["payload"])["message"]
            for event in restored_sink.events
            if event["type"] == "run.output"
        ]
        assert [
            message["content"] for message in outputs if message["metadata"].get("_stream_delta")
        ] == [
            "early from session A",
            "answer from session A",
        ]
        assert len([message for message in outputs if message["metadata"].get("_streamed")]) == 1
        terminals = [event for event in restored_sink.events if event["type"] == "run.completed"]
        assert len(terminals) == 1
        assert cast(dict[str, object], terminals[0]["payload"])["finish_reason"] == "completed"
        assert Session.load(workspace.workspace_state, session_id).messages == authority.messages
        assert [(message["role"], message["content"]) for message in authority.messages] == [
            ("user", "session-a"),
            ("assistant", "answer from session A"),
        ]
    finally:
        release_output.set()
        provider.release_a.set()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_workspace", [False, True])
async def test_backpressured_client_expiry_preserves_another_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cross_workspace: bool
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    provider = _ConcurrentProvider(block_b=True, early_a_delta=True)
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda _config: provider)
    clock = FakeClock(datetime(2026, 10, 4, tzinfo=UTC))
    timer_started, wake_timer = asyncio.Event(), asyncio.Event()
    blocked, release_output = asyncio.Event(), asyncio.Event()

    async def sleep(_seconds: float) -> None:
        timer_started.set()
        await wake_timer.wait()

    service = AgentService(
        home,
        ConfigLoader(home).load_for_startup(),
        reconnect_timeout=30,
        monotonic_now=clock.monotonic,
        sleep=sleep,
    )
    first_sink, second_sink = _CollectingSink(), _CollectingSink()
    send_event = first_sink.send_event

    async def slow_output(event: dict[str, object]) -> None:
        if event.get("type") == "run.output":
            message = cast(dict[str, Any], event["payload"])["message"]
            if message["metadata"].get("_stream_delta") is True:
                blocked.set()
                await release_output.wait()
        await send_event(event)

    monkeypatch.setattr(first_sink, "send_event", slow_output)
    try:
        await service.start()
        first, second = await service.register_client("cli"), await service.register_client("web")
        await service.connect_client(first.client_id, first_sink)
        await service.connect_client(second.client_id, second_sink)
        workspace = await service.attach_workspace(first.client_id, path)
        second_path = tmp_path / "second-workspace" if cross_workspace else path
        second_path.mkdir(exist_ok=True)
        second_workspace = await service.attach_workspace(second.client_id, second_path)
        # Keep both Workspaces attached so expiry only cleans the target Client's work.
        await service.attach_workspace(second.client_id, path)
        session_a = await workspace.create_draft(first.client_id)
        session_b = await second_workspace.create_draft(second.client_id)
        claim_a = await service.claim(first.client_id, workspace.workspace_id, session_a)
        claim_b = await service.claim(second.client_id, second_workspace.workspace_id, session_b)
        state_a, state_b = workspace.loops[session_a], second_workspace.loops[session_b]
        authority_b = state_b.loop.session
        await second_workspace.input(
            second.client_id, session_b, _claim_version(claim_b), "session-b", "run-b"
        )
        await asyncio.wait_for(provider.session_b_started.wait(), 3)
        await workspace.input(
            first.client_id, session_a, _claim_version(claim_a), "session-a", "run-a"
        )
        await asyncio.wait_for(blocked.wait(), 3)
        await service.disconnect_client(first.client_id, sink=first_sink)
        expiry = first.disconnect_task
        assert expiry is not None
        await asyncio.wait_for(timer_started.wait(), 3)
        clock.advance(30)
        wake_timer.set()
        await asyncio.wait_for(expiry, 3)
        assert provider.session_a_cancelled.is_set()
        assert not provider.session_b_cancelled.is_set()
        assert state_b.loop.has_active_run
        assert second_workspace.loops[session_b] is state_b
        assert state_b.loop.session is authority_b
        current_claim = await service.claim(
            second.client_id, second_workspace.workspace_id, session_b
        )
        assert current_claim["claim"] == claim_b["claim"]
        assert session_a not in workspace._claims
        provider.release_b.set()
        await asyncio.wait_for(second_sink.wait_for("run.completed", "run-b"), 3)
        processor = state_b.processor_task
        if processor is not None:
            await asyncio.wait_for(asyncio.shield(processor), 3)
        await state_b.loop.wait_for_restore_idle()
        run_events = [
            event for event in second_sink.events if str(event["type"]).startswith("run.")
        ]
        assert all(
            event["run_id"] == "run-b"
            and event["session_id"] == session_b
            and event["workspace_id"] == second_workspace.workspace_id
            for event in run_events
        )
        terminals = [event for event in run_events if event["type"] == "run.completed"]
        assert len(terminals) == 1
        assert cast(dict[str, object], terminals[0]["payload"])["finish_reason"] == "completed"
        assert not any(event["type"] == "run.cancelled" for event in run_events)
        messages = [
            cast(dict[str, Any], event["payload"])["message"]
            for event in run_events
            if event["type"] == "run.output"
        ]
        assert [
            message["content"] for message in messages if message["metadata"].get("_stream_delta")
        ] == ["answer from session B"]
        assert len([message for message in messages if message["metadata"].get("_streamed")]) == 1
        assert [(message["role"], message["content"]) for message in authority_b.messages] == [
            ("user", "session-b"),
            ("assistant", "answer from session B"),
        ]
        assert (
            Session.load(second_workspace.workspace_state, session_b).messages
            == authority_b.messages
        )
        assert len(state_a.loop.session.messages) == 2
        assert state_a.loop.session.messages[-1]["error"]["code"] == "turn_cancelled"
        assert (
            Session.load(workspace.workspace_state, session_a).messages
            == state_a.loop.session.messages
        )
    finally:
        release_output.set()
        provider.release_a.set()
        provider.release_b.set()
        await service.stop()


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

        await workspace.release(client.client_id, session_id)

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
async def test_open_conversation_returns_snapshot_and_preserves_context_on_occupied_target(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path / "agent-home")
    first_path = tmp_path / "first-project"
    second_path = tmp_path / "second-project"
    first_path.mkdir()
    second_path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        owner = await service.register_client("web")
        first_record, _, _ = await service.register_project(owner.client_id, first_path)
        second_record, _, _ = await service.register_project(owner.client_id, second_path)

        first = await service.open_conversation(
            owner.client_id, project_id=first_record.project_id, create_new=True
        )
        _validator("conversation_open_response").validate({"request_id": "first", **first})
        second = await service.open_conversation(
            owner.client_id, project_id=second_record.project_id, create_new=True
        )
        other = await service.register_client("web")
        await service.open_conversation(
            other.client_id,
            project_id=first_record.project_id,
            session_id=first["session_id"],
        )

        with pytest.raises(ServiceError) as occupied:
            await service.open_conversation(
                owner.client_id,
                project_id=first_record.project_id,
                session_id=first["session_id"],
            )

        assert occupied.value.code == "session_claimed"
        _validator("error").validate(occupied.value.to_dict("occupied"))
        claim = second["claim"]
        failure = cast(dict[str, Any], occupied.value.to_dict("occupied")["conversation"])
        assert failure["current_context"] == claim
        assert failure["current_conversation"]["project_id"] == second_record.project_id
        assert failure["target"]["status"] == "session_claimed"
        assert failure["target"]["session_id"] == first["session_id"]
        assert "snapshot" not in failure["target"]
        with pytest.raises(ServiceError) as missing:
            await service.open_conversation(
                owner.client_id,
                project_id=first_record.project_id,
                session_id="20261005-000000-000000_00000000-0000-4000-8000-000000000322",
            )
        assert missing.value.code == "not_found"
        assert "session_id" in missing.value.field_errors
        _validator("error").validate(missing.value.to_dict("missing"))
        current = await service.get_session_snapshot(
            owner.client_id,
            second["workspace_id"],
            second["session_id"],
            claim["claim_version"],
            claim["reconnect_credential"],
        )
        assert first["snapshot"]["session_id"] == first["session_id"]
        assert cast(dict[str, object], current["snapshot"])["session_id"] == second["session_id"]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_open_conversation_request_replay_preserves_one_draft_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        first, replay = await asyncio.gather(
            *(
                service.open_conversation(
                    client.client_id, directory=str(path), create_new=True, request_id="open-once"
                )
                for _ in range(2)
            )
        )
        assert first == replay
        workspace = service.workspace(first["workspace_id"])
        assert list(workspace.loops) == [first["session_id"]]
        assert not (path / ".omni" / "sessions" / f"{first['session_id']}.jsonl").exists()
        with pytest.raises(ServiceError) as reused:
            await service.open_conversation(
                client.client_id, directory=str(path), request_id="open-once"
            )
        assert reused.value.code == "request_reused"
        reconnected = await service.open_conversation(client.client_id, directory=str(path))
        assert reconnected["claim"] == first["claim"]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["claim", "snapshot", "snapshot_io"])
async def test_failed_new_conversation_discards_only_the_unclaimed_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        first = await service.open_conversation(client.client_id, directory=str(path))
        workspace = service.workspace(first["workspace_id"])

        async def reject_claim(*_args: object) -> dict[str, object]:
            raise ServiceError("admission_closed", "Admission closed before claiming.")

        def reject_snapshot(_session_id: str) -> dict[str, object]:
            if failure_stage == "snapshot_io":
                raise OSError("Deletion state cannot be read.")
            raise ServiceError("admission_closed", "Snapshot could not be prepared.")

        if failure_stage == "claim":
            monkeypatch.setattr(service, "claim", reject_claim)
        else:
            monkeypatch.setattr(workspace, "session_snapshot", reject_snapshot)
        with pytest.raises(ServiceError) as rejected:
            await service.open_conversation(
                client.client_id, workspace_id=workspace.workspace_id, create_new=True
            )
        assert rejected.value.code == (
            "persistence_error" if failure_stage == "snapshot_io" else "admission_closed"
        )
        assert list(workspace.loops) == [first["session_id"]]
        assert workspace._draft_clients == {}
        failure = cast(dict[str, Any], rejected.value.to_dict("failed")["conversation"])
        assert failure["current_context"] == first["claim"]
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
