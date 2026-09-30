"""Behavior tests for concurrent Session execution through the local service."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestServer

import myclaw.service.runtime as service_runtime
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ProviderConfiguration
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.service.client import ServiceClient
from myclaw.service.discovery import ServiceDiscovery, create_credential, write_discovery
from myclaw.service.errors import ServiceError
from myclaw.service.projects import ProjectCatalog
from myclaw.service.runtime import LocalService
from myclaw.service.transport import create_app
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock


class _CollectingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.changed = asyncio.Event()

    async def send_event(self, event: dict[str, object]) -> None:
        self.events.append(event)
        self.changed.set()

    async def wait_for(self, event_type: str, run_id: str) -> dict[str, object]:
        while True:
            for event in self.events:
                if event.get("type") == event_type and event.get("run_id") == run_id:
                    return event
            self.changed.clear()
            await self.changed.wait()


class _ConcurrentProvider:
    def __init__(self, *, block_b: bool = False, early_a_delta: bool = False) -> None:
        self.session_a_started = asyncio.Event()
        self.release_a = asyncio.Event()
        self.session_a_cancelled = asyncio.Event()
        self.session_b_started = asyncio.Event()
        self.release_b = asyncio.Event()
        self.session_b_cancelled = asyncio.Event()
        self.block_b = block_b
        self.early_a_delta = early_a_delta

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> ModelResponse:
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        return _response(
            '{"action":"replace","task_goal":"answer the input",'
            '"completion_boundary":"return one answer"}'
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        system = messages[0].get("content") if messages else None
        user_value = next(
            (
                value.get("content")
                for value in reversed(messages)
                if value.get("role") == "user" and isinstance(value.get("content"), str)
            ),
            "",
        )
        user = user_value if isinstance(user_value, str) else ""

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if isinstance(system, str) and system.startswith("Generate a concise title"):
                yield ModelCompleted(_response("Concurrent session"))
                return
            if "session-a" in user:
                self.session_a_started.set()
                if self.early_a_delta:
                    yield TextDelta("early from session A")
                try:
                    await self.release_a.wait()
                except asyncio.CancelledError:
                    self.session_a_cancelled.set()
                    raise
                answer = "answer from session A"
            else:
                self.session_b_started.set()
                if self.block_b:
                    try:
                        await self.release_b.wait()
                    except asyncio.CancelledError:
                        self.session_b_cancelled.set()
                        raise
                answer = "answer from session B"
            yield TextDelta(answer)
            yield ModelCompleted(_response(answer))

        return emit()

    async def close(self) -> None:
        return None


class _ScheduleProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.complete_calls = 0

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> ModelResponse:
        self.complete_calls += 1
        self.started.set()
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        return _response(
            '{"action":"replace","task_goal":"run the schedule",'
            '"completion_boundary":"return one result"}'
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(_response("schedule completion"))

        return emit()

    async def close(self) -> None:
        return None


class _RemovalProvider(_ConcurrentProvider):
    def __init__(self) -> None:
        super().__init__(block_b=True)
        self.schedule_started = asyncio.Event()
        self.schedule_cancelled = asyncio.Event()
        self.release_schedule = asyncio.Event()

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> ModelResponse:
        del model, max_output, temperature, reasoning_effort, timeout, continuation
        if "scheduled removal job" in json.dumps(messages):
            self.schedule_started.set()
            try:
                await self.release_schedule.wait()
            except asyncio.CancelledError:
                self.schedule_cancelled.set()
                raise
            return _response(
                '{"action":"replace","task_goal":"answer the input",'
                '"completion_boundary":"return one answer"}'
            )
        return await super().complete(
            messages=messages,
            tools=tools,
            model="model",
            max_output=1,
            temperature=0.0,
            reasoning_effort=None,
            timeout=1,
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if "scheduled removal job" not in json.dumps(messages):
            return super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            self.schedule_started.set()
            try:
                await self.release_schedule.wait()
            except asyncio.CancelledError:
                self.schedule_cancelled.set()
                raise
            yield ModelCompleted(_response("schedule completion"))

        return emit()


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def _configured_home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


def _claim_version(result: dict[str, object]) -> int:
    claim = cast(dict[str, object], result["claim"])
    version = claim["claim_version"]
    assert isinstance(version, int)
    return version


async def _serve(service: LocalService, home: AgentHome) -> tuple[TestServer, int]:
    create_credential(home)
    await service.start()
    server = TestServer(create_app(service), host="127.0.0.1")
    await server.start_server()
    port = server.port
    assert port is not None
    write_discovery(
        home,
        ServiceDiscovery(
            service.service_instance_id, service.protocol_version, "127.0.0.1", port, os.getpid()
        ),
    )
    return server, port


async def _client_output(client: ServiceClient) -> list[str]:
    output: list[str] = []
    while True:
        message = await asyncio.wait_for(client.bus.get_outbound(), timeout=3)
        output.append(message.content)
        if message.metadata.get("_streamed") is True:
            return output


@pytest.mark.asyncio
async def test_two_cli_clients_complete_distinct_sessions_through_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        second = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        assert first.workspace_id == second.workspace_id
        assert first.session_id != second.session_id
        await first.submit_input("session-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        await second.submit_input("session-b")
        second_output = await _client_output(second)
        assert "answer from session B" in second_output
        assert not provider.release_a.is_set()
        provider.release_a.set()
        first_output = await _client_output(first)
        assert "answer from session A" in first_output
        assert "answer from session B" not in first_output
        assert "answer from session A" not in second_output

        workspace = service.workspace(first.workspace_id)
        first_history = Session.load(workspace.workspace_state, first.session_id)
        second_history = Session.load(workspace.workspace_state, second.session_id)
        first_content = {str(message["content"]) for message in first_history.messages}
        second_content = {str(message["content"]) for message in second_history.messages}
        assert {"session-a", "answer from session A"} <= first_content
        assert {"session-b", "answer from session B"} <= second_content
        assert first_content.isdisjoint(second_content)
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_cli_switch_does_not_display_background_session_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    client: ServiceClient | None = None
    other: ServiceClient | None = None
    try:
        client = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        other = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        background_session = client.session_id
        await client.submit_input("session-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        draft = await client._http_request(
            "POST",
            f"/api/v1/workspaces/{client.workspace_id}/sessions",
            payload={"request_id": str(uuid4())},
            mutation=True,
        )
        selected_session = cast(str, draft["session_id"])
        await client.switch_session(selected_session)
        with pytest.raises(ServiceError) as occupied:
            await other.claim_session(background_session)
        assert occupied.value.code == "session_claimed"
        provider.release_a.set()
        await client.submit_input("session-b")
        output = await _client_output(client)
        assert "answer from session B" in output
        assert "answer from session A" not in output
        for _ in range(100):
            try:
                await other.claim_session(background_session)
                break
            except ServiceError as error:
                assert error.code == "session_claimed"
                await asyncio.sleep(0.01)
        else:
            raise AssertionError("background Session Claim was not released")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.bus.get_outbound(), timeout=0.1)
    finally:
        if other is not None:
            await other.close()
        if client is not None:
            await client.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_claim_race_denies_loser_content_over_http_events_and_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        second = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        contested_session = first.session_id
        await first.submit_input("private claim marker")
        await _client_output(first)
        draft = await first._http_request(
            "POST",
            f"/api/v1/workspaces/{first.workspace_id}/sessions",
            payload={"request_id": str(uuid4())},
            mutation=True,
        )
        await first.switch_session(cast(str, draft["session_id"]))
        attempts = await asyncio.gather(
            first.claim_session(contested_session),
            second.claim_session(contested_session),
            return_exceptions=True,
        )
        assert sum(isinstance(result, dict) for result in attempts) == 1
        assert sum(isinstance(result, ServiceError) for result in attempts) == 1
        assert all(
            not isinstance(result, ServiceError) or result.code == "session_claimed"
            for result in attempts
        )
        owner, loser = (first, second) if isinstance(attempts[0], dict) else (second, first)
        assert owner.session_id == contested_session
        seen_events: list[dict[str, object]] = []
        original_handler = loser._handle_event

        async def record_event(event: dict[str, object]) -> None:
            seen_events.append(event)
            await original_handler(event)

        monkeypatch.setattr(loser, "_handle_event", record_event)
        path = f"/api/v1/workspaces/{owner.workspace_id}/sessions/{contested_session}"
        with pytest.raises(ServiceError) as denied:
            await loser._http_request(
                "GET",
                f"{path}?claim_version={owner.claim_version}",
                extra_headers={"X-MyClaw-Claim": owner.claim_credential},
            )
        assert denied.value.code == "stale_claim"
        listing = await loser._http_request(
            "GET", f"/api/v1/workspaces/{owner.workspace_id}/sessions"
        )
        assert "private claim marker" not in json.dumps(listing)
        await owner.submit_input("owner private message")
        await _client_output(owner)
        await asyncio.sleep(0)
        assert not any(
            event.get("session_id") == contested_session
            and event.get("type") in {"run.output", "input.accepted"}
            for event in seen_events
        )
        reconnect_credential = loser.reconnect_credential
        await loser.close()
        reconnected = await ServiceClient.connect_or_start(
            home, workspace_path, port=port, reconnect_credential=reconnect_credential
        )
        if loser is first:
            first = reconnected
        else:
            second = reconnected
        with pytest.raises(ServiceError) as denied_after_reconnect:
            await reconnected._http_request(
                "GET",
                f"{path}?claim_version={owner.claim_version}",
                extra_headers={"X-MyClaw-Claim": owner.claim_credential},
            )
        assert denied_after_reconnect.value.code == "stale_claim"
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_distinct_sessions_run_in_parallel_and_cancel_is_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider(block_b=True)

    def provider_factory(_configuration: ProviderConfiguration) -> _ConcurrentProvider:
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    first_sink = _CollectingSink()
    second_sink = _CollectingSink()
    try:
        first = await service.register_client("cli")
        second = await service.register_client("cli")
        workspace = await service.attach_workspace(first.client_id, workspace_path)
        await service.attach_workspace(second.client_id, workspace_path)
        await service.connect_client(first.client_id, first_sink)
        await service.connect_client(second.client_id, second_sink)

        session_a = await workspace.create_draft(first.client_id)
        session_b = await workspace.create_draft(second.client_id)
        claim_a = await service.claim(first.client_id, workspace.workspace_id, session_a)
        claim_b = await service.claim(second.client_id, workspace.workspace_id, session_b)
        version_a = _claim_version(claim_a)
        version_b = _claim_version(claim_b)

        await workspace.input(first.client_id, session_a, version_a, "session-a", "run-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        await workspace.input(second.client_id, session_b, version_b, "session-b", "run-b")
        await asyncio.wait_for(provider.session_b_started.wait(), timeout=2)

        await workspace.cancel(first.client_id, session_a, version_a, "run-a")
        await asyncio.wait_for(provider.session_a_cancelled.wait(), timeout=2)
        await asyncio.wait_for(first_sink.wait_for("run.cancelled", "run-a"), timeout=2)
        assert not any(event.get("run_id") == "run-b" for event in first_sink.events)
        provider.release_b.set()
        await asyncio.wait_for(second_sink.wait_for("run.completed", "run-b"), timeout=2)
        assert not any(event.get("type") == "run.cancelled" for event in second_sink.events)

        history_a = Session.load(workspace.workspace_state, session_a)
        history_b = Session.load(workspace.workspace_state, session_b)
        content_a = [str(message["content"]) for message in history_a.messages]
        content_b = [str(message["content"]) for message in history_b.messages]
        assert "session-a" in content_a
        assert "answer from session A" not in content_a
        assert "session-b" in content_b
        assert "answer from session B" in content_b
        assert set(content_a).isdisjoint(content_b)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_switch_keeps_active_claim_until_run_terminates_then_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()

    def provider_factory(_configuration: ProviderConfiguration) -> _ConcurrentProvider:
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    first_sink = _CollectingSink()
    second_sink = _CollectingSink()
    try:
        first = await service.register_client("cli")
        second = await service.register_client("cli")
        workspace = await service.attach_workspace(first.client_id, workspace_path)
        await service.attach_workspace(second.client_id, workspace_path)
        await service.connect_client(first.client_id, first_sink)
        await service.connect_client(second.client_id, second_sink)

        session_a = await workspace.create_draft(first.client_id)
        claim_a = await service.claim(first.client_id, workspace.workspace_id, session_a)
        version_a = _claim_version(claim_a)
        await workspace.input(first.client_id, session_a, version_a, "session-a", "run-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)

        session_b = await workspace.create_draft(first.client_id)
        await service.claim(first.client_id, workspace.workspace_id, session_b)
        with pytest.raises(Exception) as occupied:
            await service.claim(second.client_id, workspace.workspace_id, session_a)
        assert getattr(occupied.value, "code", None) == "session_claimed"
        with pytest.raises(ServiceError) as busy:
            await service.handle_command(
                first.client_id,
                {
                    "request_id": "release-running-session",
                    "type": "release",
                    "workspace_id": workspace.workspace_id,
                    "session_id": session_a,
                    "claim_version": version_a,
                    "payload": {},
                },
            )
        assert busy.value.code == "session_busy"

        await workspace.cancel(first.client_id, session_a, version_a, "run-a")
        await asyncio.wait_for(first_sink.wait_for("run.cancelled", "run-a"), timeout=2)
        await asyncio.wait_for(first_sink.wait_for("run.completed", "run-a"), timeout=2)
        terminal_output = [
            cast(dict[str, object], cast(dict[str, object], event["payload"])["message"])
            for event in first_sink.events
            if event.get("type") == "run.output" and event.get("run_id") == "run-a"
        ]
        assert any(
            isinstance(message.get("metadata"), dict)
            and cast(dict[str, object], message["metadata"]).get("finish_reason") == "cancelled"
            for message in terminal_output
        )
        for _ in range(100):
            try:
                reacquired = await service.claim(
                    second.client_id, workspace.workspace_id, session_a
                )
                break
            except Exception as error:
                if getattr(error, "code", None) != "session_claimed":
                    raise
                await asyncio.sleep(0.01)
        else:
            raise AssertionError("the switched-away Session Claim was not released")
        assert cast(dict[str, object], reacquired["claim"])["session_id"] == session_a
        assert provider.session_a_cancelled.is_set()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_queued_session_output_flows_while_next_run_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider(early_a_delta=True)
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    sink = _CollectingSink()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        await service.connect_client(client.client_id, sink)
        session_id = await workspace.create_draft(client.client_id)
        claim = await service.claim(client.client_id, workspace.workspace_id, session_id)
        version = _claim_version(claim)
        await workspace.input(client.client_id, session_id, version, "session-b", "first-run")
        await workspace.input(client.client_id, session_id, version, "session-a", "second-run")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        early = await asyncio.wait_for(sink.wait_for("run.output", "second-run"), timeout=2)
        message = cast(dict[str, object], cast(dict[str, object], early["payload"])["message"])
        assert message["content"] == "early from session A"
        assert not provider.release_a.is_set()
        provider.release_a.set()
        await asyncio.wait_for(sink.wait_for("run.completed", "second-run"), timeout=2)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_duplicate_command_request_id_does_not_start_a_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        claimed = await service.claim(client.client_id, workspace.workspace_id, session_id)
        claim_version = _claim_version(claimed)
        original_input = workspace.input
        entered = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def slow_input(
            client_id: str,
            selected_session_id: str,
            version: int,
            text: str,
            run_id: str,
        ) -> Any:
            nonlocal call_count
            call_count += 1
            entered.set()
            await release.wait()
            return await original_input(client_id, selected_session_id, version, text, run_id)

        monkeypatch.setattr(workspace, "input", slow_input)
        command = {
            "request_id": "same-input-request",
            "type": "input",
            "workspace_id": workspace.workspace_id,
            "session_id": session_id,
            "claim_version": claim_version,
            "payload": {"text": "one input"},
        }
        first = asyncio.create_task(service.handle_command(client.client_id, command))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(service.handle_command(client.client_id, command))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        first_ack, second_ack = await asyncio.gather(first, second)
        assert first_ack == second_ack
        assert call_count == 1
        assert len(workspace.loops[session_id].run_ids) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_stale_release_command_cannot_release_a_newer_claim(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        first = await service.claim(client.client_id, workspace.workspace_id, session_id)
        first_version = _claim_version(first)
        await workspace.release(client.client_id, session_id, close_idle=False)
        second = await service.claim(client.client_id, workspace.workspace_id, session_id)
        second_version = _claim_version(second)
        assert second_version == first_version + 1

        with pytest.raises(Exception) as stale:
            await service.handle_command(
                client.client_id,
                {
                    "request_id": "stale-release",
                    "type": "release",
                    "workspace_id": workspace.workspace_id,
                    "session_id": session_id,
                    "claim_version": first_version,
                    "payload": {},
                },
            )
        assert getattr(stale.value, "code", None) == "stale_claim"
        workspace.require_claim(client.client_id, session_id, second_version)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_workspace_schedule_and_memory_are_shared_across_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ScheduleProvider()

    def provider_factory(_configuration: ProviderConfiguration) -> _ScheduleProvider:
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        second = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        assert first.workspace_id == second.workspace_id
        workspace = service.workspace(first.workspace_id)
        assert workspace.runtime is not None
        session_a = first.session_id
        session_b = second.session_id
        assert session_a != session_b
        await asyncio.gather(
            first.submit_input("session A memory"), second.submit_input("session B memory")
        )
        await asyncio.gather(_client_output(first), _client_output(second))
        assert Session.load(workspace.workspace_state, session_a).messages
        assert Session.load(workspace.workspace_state, session_b).messages

        memory = workspace.runtime.memory_manager
        timestamp = datetime(2026, 9, 30, tzinfo=UTC)
        await asyncio.gather(
            memory.append_summary(f"memory from {session_a}", timestamp),
            memory.append_summary(f"memory from {session_b}", timestamp),
        )
        summaries = await MemoryManager(workspace.workspace_state).claim_summaries(limit=10)
        assert {entry.content for entry in summaries.entries} == {
            f"memory from {session_a}",
            f"memory from {session_b}",
        }
        await asyncio.sleep(0.05)
        provider.complete_calls = 0

        job = ScheduleJob(
            job_id=str(uuid4()),
            message="run the shared due job",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(job)
        for _ in range(100):
            current = await workspace.schedule_service.public_snapshot()
            if current and current[0].state.last_status is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the due Schedule Job did not reach a terminal state")
        assert current[0].state.last_status == "ok", current[0].state
        assert provider.started.is_set()
        assert provider.complete_calls == 1
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_cancels_foreground_and_schedule_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    record = ProjectCatalog(home).register(workspace_path)
    provider = _RemovalProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        second = await ServiceClient.connect_or_start(home, workspace_path, port=port)
        workspace = service.workspace(first.workspace_id)
        workspace_id = first.workspace_id

        await first.submit_input("session-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        await second.submit_input("session-b")
        await asyncio.wait_for(provider.session_b_started.wait(), timeout=2)

        job = ScheduleJob(
            job_id=str(uuid4()),
            message="scheduled removal job",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(job)
        await asyncio.wait_for(provider.schedule_started.wait(), timeout=2)
        assert len(workspace.loops) == 3
        assert len(workspace._schedule_loops) == 1
        assert workspace.schedule_service.status_snapshot().to_dict()["active_job_count"] == 1

        await service.remove_project(first.client_id, record.project_id)

        await asyncio.wait_for(provider.session_a_cancelled.wait(), timeout=2)
        await asyncio.wait_for(provider.session_b_cancelled.wait(), timeout=2)
        await asyncio.wait_for(provider.schedule_cancelled.wait(), timeout=2)
        assert not workspace.loops
        assert not workspace._schedule_loops
        assert workspace.schedule_service.status_snapshot().to_dict()["active_job_count"] == 0
        assert not workspace._claims
        assert workspace_id not in service.workspaces
        saved_jobs = await WorkspaceScheduleStore(workspace.workspace_state).public_snapshot()
        assert [saved.job_id for saved in saved_jobs] == [job.job_id]
        assert workspace_path.is_dir()
        for _ in range(100):
            if first.workspace_id == second.workspace_id == "":
                break
            await asyncio.sleep(0.01)
        assert first.workspace_id == second.workspace_id == ""
        assert not first.control.foreground_input_admitted()
        assert not second.control.foreground_input_admitted()
        for cli_client in (first, second):
            async with asyncio.timeout(2):
                while True:
                    output = await cli_client.bus.get_outbound()
                    if "Project registration was removed; its work has stopped." in output.content:
                        break
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_due_job_runs_once_in_unselected_registered_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    selected_project = tmp_path / "selected-project"
    unselected_project = tmp_path / "unselected-project"
    selected_project.mkdir()
    unselected_project.mkdir()

    ProjectCatalog(home).register(selected_project)
    ProjectCatalog(home).register(unselected_project)
    unselected_state = WorkspaceState(unselected_project)
    unselected_state.initialize(agent_home_root=home.path)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="run from an unselected project",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await WorkspaceScheduleStore(unselected_state).add_user_job(job)

    provider = _ScheduleProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    server, port = await _serve(service, home)
    client: ServiceClient | None = None
    try:
        web_client = await service.register_client("web")

        class Sink:
            async def send_event(self, event: dict[str, object]) -> None:
                del event

        await service.connect_client(web_client.client_id, Sink())
        selected_workspace = await service.attach_workspace(web_client.client_id, selected_project)
        client = await ServiceClient.connect_or_start(home, selected_project, port=port)
        assert service.workspace(client.workspace_id) is selected_workspace
        assert service.workspace(client.workspace_id).schedule_service is selected_workspace.schedule_service
        runtime = selected_workspace.runtime
        assert runtime is not None
        cli_runtime = service.workspace(client.workspace_id).runtime
        assert cli_runtime is runtime
        assert cli_runtime.memory_manager is runtime.memory_manager
        assert len(service.workspaces) == 2
        unselected_workspace = next(
            workspace
            for workspace in service.workspaces.values()
            if workspace.workspace_path == unselected_project.resolve()
        )
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        for _ in range(100):
            current = await unselected_workspace.schedule_service.public_snapshot()
            if current and current[0].state.last_status is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the unselected Project Job did not reach a terminal state")

        assert current[0].state.last_status == "ok", current[0].state
        assert provider.complete_calls == 1
        assert unselected_workspace.schedule_admitted
    finally:
        if client is not None:
            await client.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_client_expiry_keeps_claim_until_cancelled_run_cleanup_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    clock = FakeClock(datetime(2026, 9, 30, tzinfo=UTC))
    wake_timer = asyncio.Event()

    async def wait_for_timer(_seconds: float) -> None:
        await wake_timer.wait()
        wake_timer.clear()

    async def advance(seconds: float) -> None:
        clock.advance(seconds)
        wake_timer.set()
        for _ in range(4):
            await asyncio.sleep(0)

    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(
        home,
        ConfigLoader(home).load_for_startup(),
        reconnect_timeout=30,
        monotonic_now=clock.monotonic,
        sleep=wait_for_timer,
    )
    first_sink = _CollectingSink()
    second_sink = _CollectingSink()
    try:
        await service.start()
        first = await service.register_client("cli")
        second = await service.register_client("cli")
        workspace = await service.attach_workspace(first.client_id, workspace_path)
        await service.attach_workspace(second.client_id, workspace_path)
        await service.connect_client(first.client_id, first_sink)
        await service.connect_client(second.client_id, second_sink)

        session_id = await workspace.create_draft(first.client_id)
        claimed = await service.claim(first.client_id, workspace.workspace_id, session_id)
        claim_data = cast(dict[str, object], claimed["claim"])
        claim_version = cast(int, claim_data["claim_version"])
        await workspace.input(first.client_id, session_id, claim_version, "session-a", "run-a")
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)

        await service.disconnect_client(first.client_id, sink=first_sink)
        await advance(29)
        assert workspace._claims[session_id].status == "reconnecting"
        with pytest.raises(ServiceError) as still_owned:
            await service.claim(second.client_id, workspace.workspace_id, session_id)
        assert still_owned.value.code == "session_claimed"

        reconnected_sink = _CollectingSink()
        reconnect_credential = first.reconnect_credential
        assert await service.register_client("cli", reconnect_credential) is first
        with pytest.raises(ServiceError) as rotated_credential:
            await service.register_client("cli", reconnect_credential)
        assert rotated_credential.value.code == "stale_client"
        await service.connect_client(first.client_id, reconnected_sink)
        assert workspace._claims[session_id].status == "claimed"
        await service.disconnect_client(first.client_id, sink=reconnected_sink)
        await advance(29)
        assert workspace._claims[session_id].status == "reconnecting"

        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        close_loop = workspace._close_loop

        async def delayed_close(selected_session_id: str, *, abort: bool = False) -> None:
            cleanup_started.set()
            await cleanup_release.wait()
            await close_loop(selected_session_id, abort=abort)

        monkeypatch.setattr(workspace, "_close_loop", delayed_close)
        expiry_task = first.disconnect_task
        assert expiry_task is not None
        clock.advance(1)
        with pytest.raises(ServiceError) as boundary_registration:
            await service.register_client("cli", first.reconnect_credential)
        assert boundary_registration.value.code == "stale_client"
        with pytest.raises(ServiceError) as boundary_connection:
            await service.connect_client(first.client_id, _CollectingSink())
        assert boundary_connection.value.code == "stale_client"
        await advance(0)
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert workspace._claims[session_id].status == "draining"
        assert provider.session_a_cancelled.is_set()
        with pytest.raises(ServiceError) as draining:
            await workspace.input(first.client_id, session_id, claim_version, "late", "late-run")
        assert draining.value.code == "stale_claim"
        with pytest.raises(ServiceError) as cleanup_in_progress:
            await service.claim(second.client_id, workspace.workspace_id, session_id)
        assert cleanup_in_progress.value.code == "session_claimed"

        cleanup_release.set()
        await asyncio.wait_for(expiry_task, timeout=2)
        history = Session.load(workspace.workspace_state, session_id)
        assert any(message.get("role") == "user" for message in history.messages)
        assert any(
            message.get("role") == "assistant"
            and message.get("status") == "interrupted"
            and isinstance(message.get("error"), dict)
            and message["error"].get("code") == "turn_cancelled"
            for message in history.messages
        )
        with pytest.raises(ServiceError) as stale_input:
            await workspace.input(first.client_id, session_id, claim_version, "expired", "expired")
        assert stale_input.value.code == "stale_claim"
        with pytest.raises(ServiceError) as expired_client:
            await service.register_client("cli", first.reconnect_credential)
        assert expired_client.value.code == "stale_client"
        reacquired = await service.claim(second.client_id, workspace.workspace_id, session_id)
        assert cast(dict[str, object], reacquired["claim"])["session_id"] == session_id
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_event_reconnect_replays_once_and_cache_overflow_requires_snapshot(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    first_sink = _CollectingSink()
    try:
        await service.start()
        client = await service.register_client("cli")
        await service.attach_workspace(client.client_id, workspace_path)
        await service.connect_client(client.client_id, first_sink)
        await service.emit(
            "test.event",
            workspace_id=None,
            session_id=None,
            run_id=None,
            payload={"marker": "before-disconnect"},
        )
        last_seq = client.sequence
        await service.disconnect_client(client.client_id, sink=first_sink)
        old_credential = client.reconnect_credential
        assert await service.register_client("cli", old_credential) is client
        with pytest.raises(ServiceError) as rotated:
            await service.register_client("cli", old_credential)
        assert rotated.value.code == "stale_client"
        for marker in range(3):
            await service.emit(
                "test.event",
                workspace_id=None,
                session_id=None,
                run_id=None,
                payload={"marker": marker},
            )

        replay_sink = _CollectingSink()
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        replay = await service.handle_command(
            client.client_id,
            {
                "request_id": "replay",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": last_seq},
            },
        )
        assert replay["accepted"] is True
        assert [event["seq"] for event in replay_sink.events] == list(
            range(last_seq + 1, last_seq + 4)
        )
        assert [
            cast(dict[str, object], event["payload"])["marker"]
            for event in replay_sink.events
        ] == [0, 1, 2]

        await service.disconnect_client(client.client_id, sink=replay_sink)
        replay_sink = _CollectingSink()
        for marker in range(300):
            await service.emit(
                "test.event",
                workspace_id=None,
                session_id=None,
                run_id=None,
                payload={"marker": marker},
            )
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        snapshot = await service.handle_command(
            client.client_id,
            {
                "request_id": "overflow",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": last_seq},
            },
        )
        assert snapshot["accepted"] is True
        assert len(replay_sink.events) == 1
        event = replay_sink.events[0]
        assert event["type"] == "snapshot.required"
        payload = cast(dict[str, object], event["payload"])
        assert payload["reason"] == "event_cache_exhausted"
        assert isinstance(payload["snapshot"], dict)
        assert "credential" not in json.dumps(event)

        await service.disconnect_client(client.client_id, sink=replay_sink)
        replay_sink = _CollectingSink()
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        await service.handle_command(
            client.client_id,
            {
                "request_id": "different-stream",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": client.sequence, "stream_id": "old-stream"},
            },
        )
        assert replay_sink.events[0]["type"] == "snapshot.required"
        assert cast(dict[str, object], replay_sink.events[0]["payload"])["reason"] == "stream_changed"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_replay_holds_live_events_until_cached_events_are_sent(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    first_sink = _CollectingSink()
    try:
        await service.start()
        client = await service.register_client("web")
        await service.connect_client(client.client_id, first_sink, wait_for_subscribe=True)
        await service.emit(
            "test.event", workspace_id=None, session_id=None, run_id=None,
            payload={"marker": "before"},
        )
        last_seq = client.sequence
        await service.disconnect_client(client.client_id, sink=first_sink)
        for marker in ("cached-a", "cached-b"):
            await service.emit(
                "test.event", workspace_id=None, session_id=None, run_id=None,
                payload={"marker": marker},
            )

        class PausingSink(_CollectingSink):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.resume = asyncio.Event()

            async def send_event(self, event: dict[str, object]) -> None:
                if event["seq"] == last_seq + 1:
                    self.started.set()
                    await self.resume.wait()
                await super().send_event(event)

        replay_sink = PausingSink()
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        subscribe = asyncio.create_task(service.handle_command(
            client.client_id,
            {
                "request_id": "ordered-replay",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": last_seq},
            },
        ))
        await asyncio.wait_for(replay_sink.started.wait(), timeout=1)
        live = asyncio.create_task(service.emit(
            "test.event", workspace_id=None, session_id=None, run_id=None,
            payload={"marker": "live"},
        ))
        await asyncio.sleep(0)
        assert not live.done()
        replay_sink.resume.set()
        await asyncio.wait_for(asyncio.gather(subscribe, live), timeout=2)
        assert [event["seq"] for event in replay_sink.events] == [
            last_seq + 1, last_seq + 2, last_seq + 3,
        ]
        assert [cast(dict[str, object], event["payload"])["marker"] for event in replay_sink.events] == [
            "cached-a", "cached-b", "live",
        ]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_snapshot_resync_includes_selected_and_switched_away_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    initial_sink = _CollectingSink()
    try:
        await service.start()
        client = await service.register_client("web")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        await service.connect_client(client.client_id, initial_sink, wait_for_subscribe=True)
        first_session = await workspace.create_draft(client.client_id)
        first_claim = await service.claim(client.client_id, workspace.workspace_id, first_session)
        await workspace.input(
            client.client_id, first_session, _claim_version(first_claim), "session-a", "run-a"
        )
        await asyncio.wait_for(provider.session_a_started.wait(), timeout=2)
        second_session = await workspace.create_draft(
            client.client_id, reuse_startup_session=False
        )
        await service.claim(client.client_id, workspace.workspace_id, second_session)
        assert set(client.claimed) == {
            (workspace.workspace_id, first_session),
            (workspace.workspace_id, second_session),
        }
        last_seq = client.sequence
        await service.disconnect_client(client.client_id, sink=initial_sink)
        for marker in range(257):
            await service.emit(
                "test.event", workspace_id=None, session_id=None, run_id=None,
                payload={"marker": marker},
            )
        replay_sink = _CollectingSink()
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        await service.handle_command(
            client.client_id,
            {
                "request_id": "background-snapshot",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": last_seq, "stream_id": client.stream_id},
            },
        )
        assert len(replay_sink.events) == 1
        event = replay_sink.events[0]
        assert event["type"] == "snapshot.required"
        payload = cast(dict[str, object], event["payload"])
        snapshot = cast(dict[str, object], payload["snapshot"])
        sessions = cast(list[dict[str, object]], snapshot["sessions"])
        assert {cast(dict[str, object], item["snapshot"])["session_id"] for item in sessions} == {
            first_session, second_session,
        }
        assert "credential" not in json.dumps(event)
    finally:
        provider.release_a.set()
        await service.stop()


@pytest.mark.asyncio
async def test_slow_consumer_reconnect_requires_snapshot(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)

    class StalledSink:
        async def send_event(self, _event: dict[str, object]) -> None:
            await asyncio.Event().wait()

    try:
        await service.start()
        client = await service.register_client("web")
        await service.connect_client(client.client_id, StalledSink())
        await service.emit(
            "test.event", workspace_id=None, session_id=None, run_id=None, payload={},
        )
        assert not client.connected
        assert client.resync_required
        replay_sink = _CollectingSink()
        await service.connect_client(client.client_id, replay_sink, wait_for_subscribe=True)
        await service.handle_command(
            client.client_id,
            {
                "request_id": "slow-consumer-resync",
                "type": "subscribe",
                "workspace_id": None,
                "session_id": None,
                "claim_version": None,
                "payload": {"last_seq": 0, "stream_id": client.stream_id},
            },
        )
        assert [event["type"] for event in replay_sink.events] == ["snapshot.required"]
        assert cast(dict[str, object], replay_sink.events[0]["payload"])["reason"] == "slow_consumer"
        assert not client.resync_required
        assert "credential" not in json.dumps(replay_sink.events)
    finally:
        await service.stop()
