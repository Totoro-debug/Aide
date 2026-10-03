"""Exercise configuration handover with real Runs, resources, and transports."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiohttp
import pytest

import omni.service.runtime as service_runtime
from omni.agent.confirmation import ConfirmationDecision, ConfirmationEnvelope
from omni.agent.session.backup_store import FileBackupStore
from omni.agent.session.restore import RestoreManager
from omni.agent.session.session import Session
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.agent.workspace_runtime import WorkspaceRuntime
from omni.agent.workspace_state import WorkspaceState
from omni.config.config import ConfigLoader
from omni.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.service.client import ServiceClient
from omni.service.errors import ServiceError
from omni.service.runtime import LocalService
from tests.service.test_restore_management import _restore_request
from tests.service.test_service_concurrency import (
    _client_output,
    _CollectingSink,
    _configured_home,
    _RemovalProvider,
    _response,
    _serve,
)


class _GenerationProvider(_RemovalProvider):
    def __init__(self) -> None:
        super().__init__()
        self.block_b = False
        self.closed = False
        self.tool_completed = asyncio.Event()
        self.efforts: list[object] = []

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
        assert not self.closed, "an active Run used a closed generation resource"
        self.efforts.append(reasoning_effort)
        latest_user = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        tool_result = next(
            (
                item
                for item in messages[latest_user + 1 :]
                if item.get("tool_call_id") == "generation-write"
            ),
            None,
        )
        if tool_result is not None:
            self.tool_completed.set()

            async def after_tool() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(_response("old generation tool completed"))

            return after_tool()
        original = super().stream(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        )
        current_user = next(
            (item.get("content", "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )

        async def with_tool() -> AsyncIterator[ModelStreamEvent]:
            async for event in original:
                system = messages[0].get("content", "") if messages else ""
                if (
                    "session-a" in str(current_user)
                    and not str(system).startswith("Generate a concise title")
                    and isinstance(event, ModelCompleted)
                ):
                    yield ModelCompleted(
                        ModelResponse(
                            message=AssistantModelMessage(
                                content="Write the approved result",
                                tool_calls=(
                                    ModelToolCall(
                                        id="generation-write",
                                        name="write_file",
                                        arguments=json.dumps(
                                            {
                                                "path": "generation.txt",
                                                "content": "old Run survived",
                                            }
                                        ),
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    )
                else:
                    yield event

        return with_tool()

    async def close(self) -> None:
        self.closed = True


class _ConfirmationBarrier:
    def __init__(self) -> None:
        self.presented = asyncio.Event()
        self.token: object = None
        self.respond: Callable[[object, ConfirmationDecision], bool] | None = None

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None:
        assert envelope.tool_name == "write_file"
        self.token = token
        self.respond = respond
        self.presented.set()

    def dismiss_confirmation(self, token: object) -> None:
        assert token is self.token


class _BackgroundProvider(_GenerationProvider):
    def __init__(self, *, block_dream: bool = False, block_title: bool = False) -> None:
        super().__init__()
        self.block_dream = block_dream
        self.block_title = block_title
        self.dream_started = asyncio.Event()
        self.title_started = asyncio.Event()
        self.release_background = asyncio.Event()
        self.background_cancelled = asyncio.Event()

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
        assert not self.closed
        if "## Conversation Summaries" in json.dumps(messages):
            self.dream_started.set()
            if self.block_dream:
                try:
                    await self.release_background.wait()
                except asyncio.CancelledError:
                    self.background_cancelled.set()
                    raise
            assert not self.closed
            return _response("No durable update needed.")
        return await super().complete(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
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
        system = messages[0].get("content", "") if messages else ""
        if str(system).startswith("Generate a concise title"):

            async def title() -> AsyncIterator[ModelStreamEvent]:
                assert not self.closed
                self.title_started.set()
                if self.block_title:
                    try:
                        await self.release_background.wait()
                    except asyncio.CancelledError:
                        self.background_cancelled.set()
                        raise
                assert not self.closed
                yield ModelCompleted(_response("Completed background title"))

            return title()
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


def _application(service: LocalService) -> dict[str, Any]:
    return cast(dict[str, Any], service.config_view()["application"])


async def _wait_status(service: LocalService, status: str) -> None:
    async with asyncio.timeout(10):
        while _application(service)["status"] != status:
            await asyncio.sleep(0.01)


async def _wait_closed(runtimes: Sequence[WorkspaceRuntime | None]) -> None:
    async with asyncio.timeout(10):
        while not all(runtime is not None and runtime._closed for runtime in runtimes):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_http_save_drains_real_foreground_schedule_and_confirmation_without_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    providers: list[_GenerationProvider] = []

    def provider_factory(_configuration: object) -> _GenerationProvider:
        provider = _GenerationProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    browser = await service.register_client("web")
    pid = os.getpid()
    identity = service.service_instance_id
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old_runtime = workspace.runtime
        assert old_runtime is not None
        old_state = workspace.workspace_state
        socket = cli._socket
        claim = (cli.session_id, cli.claim_version, cli.claim_credential)
        await cli.management("permission", {"permission_level": "read-only"})
        confirmation = _ConfirmationBarrier()
        cli.confirmation.bind_presenter(confirmation)
        await cli.submit_input("session-a")
        old_provider = providers[0]
        await asyncio.wait_for(old_provider.session_a_started.wait(), 5)
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="scheduled removal job",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(job)
        await asyncio.wait_for(old_provider.schedule_started.wait(), 5)
        assert workspace.loops[cli.session_id].loop.has_active_run
        assert workspace.schedule_service.status_snapshot().to_dict()["active_job_count"] == 1
        revision = service.config_view()["revision"]
        headers = {
            "Authorization": f"Bearer {cli.token}",
            "X-Omni-CSRF": cli.token,
            "X-Omni-Client": browser.client_id,
            "X-Omni-Control": cast(str, browser.web_control_credential),
        }
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(
                server.make_url("/api/v1/events"),
                headers={**headers, "Origin": str(server.make_url("/")).rstrip("/")},
                protocols=("omni-v1",),
            ) as web_socket:
                response = await http.patch(
                    server.make_url("/api/v1/config"),
                    headers=headers,
                    json={
                        "request_id": "drain-real-runs",
                        "revision": revision,
                        "fields": {
                            "runtime": {"max_iterations": 81, "permission_level": "full-access"},
                            "memory": {"batch_size": 11},
                        },
                        "secrets": {},
                    },
                )
                saved = await response.json()
                assert response.status == 200, saved
                assert saved["application"]["status"] == "pending"
                assert saved["application"]["active_revision"] == revision
                assert not old_provider.closed
                with pytest.raises(ServiceError, match="admission"):
                    await cli.submit_input("new work must wait")
                old_provider.release_a.set()
                await asyncio.wait_for(confirmation.presented.wait(), 5)
                assert _application(service)["status"] == "pending"
                assert not old_provider.closed
                assert confirmation.respond is not None
                assert confirmation.respond(confirmation.token, "approved")
                assert "old generation tool completed" in await _client_output(cli)
                assert (project / "generation.txt").read_text(
                    encoding="utf-8"
                ) == "old Run survived"
                assert _application(service)["status"] == "pending"
                assert not old_provider.closed
                old_provider.release_schedule.set()
                await _wait_status(service, "active")
                await _wait_closed([old_runtime])
                assert not web_socket.closed
                await web_socket.ping(b"still-connected")
                assert service.client(browser.client_id).connected
        assert old_provider.tool_completed.is_set()
        assert not old_provider.session_a_cancelled.is_set()
        assert not old_provider.schedule_cancelled.is_set()
        assert old_provider.closed
        assert workspace.runtime is not old_runtime
        assert workspace.workspace_state is old_state
        assert workspace.configuration.runtime.max_iterations == 81
        assert workspace.configuration.memory.batch_size == 11
        assert (cli.session_id, cli.claim_version, cli.claim_credential) == claim
        assert cli._socket is socket and socket is not None and not socket.closed
        assert os.getpid() == pid and service.service_instance_id == identity
        assert service.client_permission(cli.client_id).current() == "read-only"
        assert service.client_permission(cli.client_id).configured() == "full-access"
        assert (
            workspace.loops[
                cli.session_id
            ].loop._tool_gateway._permission_context.configured_schedule_level
            == "full-access"
        )
        assert (
            workspace._schedule_loops[job.job_id].loop._permission_control.configured()
            == "full-access"
        )
        jobs = await workspace.schedule_service.public_snapshot()
        assert next(item for item in jobs if item.job_id == job.job_id).state.last_status == "ok"
        assert len(workspace._schedule_loops) == 1
        await cli.submit_input("session-b")
        assert "answer from session B" in await _client_output(cli)
        assert len(providers) == 2
        persisted = Session.load(cast(Any, old_state), cli.session_id)
        assert any(
            message.get("content") == "old generation tool completed"
            for message in persisted.messages
        )
        assert any(
            message.get("content") == "answer from session B" for message in persisted.messages
        )
    finally:
        for provider in providers:
            provider.release_a.set()
            provider.release_schedule.set()
        if cli is not None:
            await cli.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_multi_workspace_prepare_failure_keeps_every_old_generation_then_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    monkeypatch.setattr(
        service_runtime, "create_provider", lambda _configuration: _GenerationProvider()
    )
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    client = await service.register_client("cli")
    sink = _CollectingSink()
    await service.connect_client(client.client_id, sink)
    staged: list[WorkspaceRuntime] = []
    original = WorkspaceRuntime.start_replacement
    fail = True

    async def start_candidate(
        candidate: WorkspaceRuntime, previous: WorkspaceRuntime
    ) -> WorkspaceRuntime:
        staged.append(candidate)
        result = await original(candidate, previous)
        if fail and candidate.workspace_path.name == "second":
            raise RuntimeError("injected resource preparation failure: canary-secret")
        return result

    monkeypatch.setattr(WorkspaceRuntime, "start_replacement", start_candidate)
    try:
        workspaces = []
        claims = []
        for name in ("first", "second"):
            path = tmp_path / name
            path.mkdir()
            workspace = await service.attach_workspace(client.client_id, path)
            session = await workspace.create_draft(client.client_id)
            claims.append(await workspace.claim(client.client_id, session))
            workspaces.append(workspace)
        old = [workspace.runtime for workspace in workspaces]
        revision = cast(str, service.config_view()["revision"])
        await service.update_configuration(
            "prepare-failure", revision, {"memory": {"batch_size": 13}}
        )
        await _wait_status(service, "failed-to-apply")
        assert [workspace.runtime for workspace in workspaces] == old
        assert all(runtime is not None and not runtime._closed for runtime in old)
        assert len(staged) == 2 and all(candidate._closed for candidate in staged)
        assert _application(service)["active_revision"] == revision
        assert "canary-secret" not in str(service.config_view())
        assert all(workspace.configuration.memory.batch_size != 13 for workspace in workspaces)
        for index, (workspace, claim) in enumerate(zip(workspaces, claims, strict=True)):
            run_id = f"old-generation-after-failure-{index}"
            await workspace.input(
                client.client_id, claim.session_id, claim.version, "session-b", run_id
            )
            async with asyncio.timeout(5):
                while not any(
                    message.get("content") == "answer from session B"
                    for message in claim.loop.session.messages
                ):
                    await asyncio.sleep(0.01)
            await claim.loop.session.wait_for_pending_persist()
            persisted = Session.load(
                cast(WorkspaceState, workspace.workspace_state), claim.session_id
            )
            assert any(
                message.get("content") == "answer from session B" for message in persisted.messages
            )
        fail = False
        saved_revision = cast(str, service.config_view()["revision"])
        await service.retry_configuration("retry-prepare", saved_revision)
        await _wait_status(service, "active")
        assert _application(service)["active_revision"] == saved_revision
        assert all(workspace.configuration.memory.batch_size == 13 for workspace in workspaces)
        await _wait_closed(old)
        assert all(runtime is not None and runtime._closed for runtime in old)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_save_and_cli_effort_supersede_candidate_preparing_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(
        service_runtime, "create_provider", lambda _configuration: _GenerationProvider()
    )
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    prepared = asyncio.Event()
    release = asyncio.Event()
    staged: list[WorkspaceRuntime] = []
    published: list[WorkspaceRuntime] = []
    original_start = WorkspaceRuntime.start_replacement
    original_publish = WorkspaceRuntime.publish_replacements

    async def start_candidate(
        candidate: WorkspaceRuntime, previous: WorkspaceRuntime
    ) -> WorkspaceRuntime:
        result = await original_start(candidate, previous)
        staged.append(candidate)
        if candidate.configuration.runtime.max_iterations == 80:
            prepared.set()
            await release.wait()
        return result

    def publish_candidates(
        cls: type[WorkspaceRuntime],
        replacements: tuple[tuple[WorkspaceRuntime, WorkspaceRuntime], ...],
    ) -> None:
        published.extend(candidate for _previous, candidate in replacements)
        original_publish(replacements)

    monkeypatch.setattr(WorkspaceRuntime, "start_replacement", start_candidate)
    monkeypatch.setattr(WorkspaceRuntime, "publish_replacements", classmethod(publish_candidates))
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old = workspace.runtime
        await service.update_configuration(
            "first-save",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 80}},
        )
        await asyncio.wait_for(prepared.wait(), 5)
        await service.update_configuration(
            "latest-save",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 82}},
        )
        before_effort = service.config_view()["revision"]
        result = await cli.management("effort", {"effort": "high"})
        assert result["published_effort"] == "high", result
        latest_revision = service.config_view()["revision"]
        assert latest_revision != before_effort
        assert old is not None and old.router.reasoning_effort == "high"
        assert not old._closed and workspace.runtime is old
        release.set()
        await _wait_status(service, "active")
        assert len(staged) == 2 and staged[0]._closed
        assert published == [staged[1]]
        assert workspace.runtime is staged[1]
        assert workspace.configuration.runtime.max_iterations == 82
        assert workspace.runtime.router.reasoning_effort == "high"
        assert _application(service)["active_revision"] == latest_revision
        loaded = ConfigLoader(home).load_for_startup()
        assert loaded.runtime.max_iterations == 82
        assert loaded.resolve_route("chat").route.reasoning_effort == "high"
    finally:
        release.set()
        if cli is not None:
            await cli.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_save_waits_for_real_dream_and_blocks_new_dream_until_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    providers: list[_BackgroundProvider] = []

    def provider_factory(_configuration: object) -> _BackgroundProvider:
        provider = _BackgroundProvider(block_dream=not providers)
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    other: ServiceClient | None = None
    dream_task: asyncio.Task[dict[str, object]] | None = None
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        other = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old_runtime = workspace.runtime
        assert old_runtime is not None
        await old_runtime.memory_manager.append_summary(
            "A durable user preference.", datetime.now(UTC)
        )
        dream_task = asyncio.create_task(cli.management("dream", {}))
        async with asyncio.timeout(5):
            while not providers:
                await asyncio.sleep(0)
        old_provider = providers[0]
        await asyncio.wait_for(old_provider.dream_started.wait(), 5)
        await service.update_configuration(
            "save-during-dream",
            cast(str, service.config_view()["revision"]),
            {"memory": {"batch_size": 17}},
        )
        assert _application(service)["status"] == "pending"
        assert "dream" in _application(service)["waiting_for"]
        assert not old_provider.closed and not dream_task.done()
        with pytest.raises(ServiceError) as rejected:
            await other.management("dream", {})
        assert rejected.value.code == "admission_closed"
        assert workspace.runtime is old_runtime and not old_runtime._closed
        old_provider.release_background.set()
        result = await asyncio.wait_for(dream_task, 5)
        dream_result = cast(dict[str, object], result["dream_result"])
        assert dream_result["processed_count"] == 1 and dream_result["cursor"] == 1
        await _wait_status(service, "active")
        await _wait_closed([old_runtime])
        assert not old_provider.background_cancelled.is_set() and old_provider.closed
        assert workspace.runtime is not None and workspace.runtime is not old_runtime
        await workspace.runtime.memory_manager.append_summary(
            "Another preference.", datetime.now(UTC)
        )
        next_result = await cli.management("dream", {})
        assert cast(dict[str, object], next_result["dream_result"])["processed_count"] == 1
        assert len(providers) == 2 and providers[1].dream_started.is_set()
        assert workspace.configuration.memory.batch_size == 17
    finally:
        for provider in providers:
            provider.release_background.set()
        if dream_task is not None:
            await asyncio.gather(dream_task, return_exceptions=True)
        if cli is not None:
            await cli.close()
        if other is not None:
            await other.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_save_drains_real_auto_title_after_foreground_run_has_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    providers: list[_BackgroundProvider] = []

    def provider_factory(_configuration: object) -> _BackgroundProvider:
        provider = _BackgroundProvider(block_title=not providers)
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old_runtime = workspace.runtime
        assert old_runtime is not None
        await cli.submit_input("Complete this foreground message")
        async with asyncio.timeout(5):
            while not providers:
                await asyncio.sleep(0)
        old_provider = providers[0]
        await asyncio.wait_for(old_provider.title_started.wait(), 5)
        assert "answer from session B" in await _client_output(cli)
        old_loop = workspace.loops[cli.session_id].loop
        async with asyncio.timeout(5):
            while old_loop.has_active_run:
                await asyncio.sleep(0)
        title_task = old_loop._title_work[cli.session_id].task
        assert not title_task.done()
        await old_loop.session.wait_for_pending_persist()
        run_updated_at = old_loop.session.updated_at
        persisted_run_updated_at = Session.load(
            cast(WorkspaceState, workspace.workspace_state), cli.session_id
        ).updated_at
        await service.update_configuration(
            "save-during-title",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 85}},
        )
        assert _application(service)["status"] == "pending"
        assert "title" in _application(service)["waiting_for"]
        assert workspace.runtime is old_runtime and not old_provider.closed
        with pytest.raises(ServiceError) as rejected:
            await cli.submit_input("new work waits for title resource")
        assert rejected.value.code == "admission_closed"
        assert not title_task.done() and not old_runtime._closed
        old_provider.release_background.set()
        await asyncio.wait_for(asyncio.shield(title_task), 5)
        await _wait_status(service, "active")
        await _wait_closed([old_runtime])
        assert not old_provider.background_cancelled.is_set() and old_provider.closed
        current = workspace.loops[cli.session_id].loop.session
        assert current.metadata["title"] == "Completed background title"
        assert current.updated_at == run_updated_at
        persisted = Session.load(cast(WorkspaceState, workspace.workspace_state), cli.session_id)
        assert persisted.metadata["title"] == "Completed background title"
        assert persisted.updated_at == persisted_run_updated_at
        assert workspace.configuration.runtime.max_iterations == 85
    finally:
        for provider in providers:
            provider.release_background.set()
        if cli is not None:
            await cli.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_save_waits_for_real_restore_transaction_and_preserves_restored_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    target = project / "tracked.txt"
    target.write_bytes(b"before restore")
    token = uuid4()
    session.commit_agent_run(
        [{"role": "user", "content": "branch to restore"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=token,
    )
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(token, target)
    assert ticket is not None
    target.write_bytes(b"current branch")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    monkeypatch.setattr(
        service_runtime, "create_provider", lambda _configuration: _GenerationProvider()
    )
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    original_continue = RestoreManager._continue_pending

    async def continue_pending(manager: RestoreManager, pending: Any) -> Any:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return await original_continue(manager, pending)

    monkeypatch.setattr(RestoreManager, "_continue_pending", continue_pending)
    restore_task: asyncio.Task[dict[str, object]] | None = None
    try:
        client = await service.register_client("web")
        await service.connect_client(client.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, project)
        claim = await workspace.claim(client.client_id, session.session_id)
        old_runtime = workspace.runtime
        assert old_runtime is not None
        await _restore_request(
            service, workspace, client.client_id, claim, "restore/inspect", "inspect", anchor_id=1
        )
        restore_task = asyncio.create_task(
            _restore_request(
                service,
                workspace,
                client.client_id,
                claim,
                "restore/execute",
                "execute",
                plan={"anchor_id": 1},
                mode="files",
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert RestoreManager(state, session.session_id).has_pending_transaction()
        await service.update_configuration(
            "save-during-restore",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 86}},
        )
        assert _application(service)["status"] == "pending"
        assert "restore" in _application(service)["waiting_for"]
        assert workspace.runtime is old_runtime and not old_runtime._closed
        assert not restore_task.done() and target.read_bytes() == b"current branch"
        release.set()
        result = await asyncio.wait_for(restore_task, 5)
        assert result.get("restore_result") is not None
        await _wait_status(service, "active")
        await _wait_closed([old_runtime])
        assert not cancelled.is_set()
        assert target.read_bytes() == b"before restore"
        assert not RestoreManager(state, session.session_id).has_pending_transaction()
        assert workspace.configuration.runtime.max_iterations == 86
        assert workspace.loops[session.session_id].loop is claim.loop
        assert claim.loop.session.session_id == session.session_id
    finally:
        release.set()
        if restore_task is not None:
            await asyncio.gather(restore_task, return_exceptions=True)
        await service.stop()
