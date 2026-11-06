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

import aide.service.runtime.service as service_runtime
from aide.agent.confirmation import ConfirmationDecision, ConfirmationEnvelope
from aide.agent.session.backup_store import FileBackupStore
from aide.agent.session.restore import RestoreManager
from aide.agent.session.session import Session
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.agent.workspace_state import WorkspaceState
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.client import ServiceClient
from aide.service.runtime import AgentService
from tests.fixtures.service_runtime import workspace_loops
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
        system = str(messages[0].get("content", "")) if messages else ""
        tool_names = {
            name
            for item in tools
            if isinstance(item.get("function"), dict)
            and isinstance((name := item["function"].get("name")), str)
        }
        router_request = {"dispatch_worker", "finish"} <= tool_names
        worker_request = system.startswith("You are an Aide Worker.")
        last_user_index = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        latest_user = (
            str(messages[last_user_index].get("content", ""))
            if last_user_index >= 0 else ""
        )
        tool_results = [
            item for item in messages[last_user_index + 1 :]
            if item.get("role") == "tool"
        ]
        def scripted(content: str, tool_call: ModelToolCall | None = None) -> AsyncIterator[ModelStreamEvent]:
            calls = () if tool_call is None else (tool_call,)

            async def emit_scripted() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(content=content, tool_calls=calls),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls" if calls else "stop",
                    )
                )

            return emit_scripted()

        if router_request:
            if "Worker reports for automatic continuation:" in latest_user:
                reports = [
                    line.removeprefix("Report: ")
                    for line in latest_user.splitlines()
                    if line.startswith("Report: ")
                ]
                report = next(
                    (item for item in reports if item == "answer from queued input"),
                    reports[0] if reports else "The Worker completed.",
                )
                return scripted(
                    "",
                    ModelToolCall(
                        "router-finish-report",
                        "finish",
                        json.dumps({"response": report}),
                    ),
                )
            dispatch = next(
                (item for item in tool_results if item.get("name") == "dispatch_worker"),
                None,
            )
            if dispatch is None:
                user_text = latest_user.rsplit("## User Input\n\n", 1)[-1].split(
                    "\n\n## Task goal", 1
                )[0].strip()
                task_label = next(
                    (
                        value for value in ("session-a", "session-b", "new work continues")
                        if value in user_text
                    ),
                    "foreground-request",
                )
                return scripted(
                    "",
                    ModelToolCall(
                        "generation-dispatch",
                        "dispatch_worker",
                        json.dumps({
                            "task_id": f"generation-{task_label.replace(' ', '-')}",
                            "task": user_text,
                            "context": [],
                        }),
                    ),
                )
            try:
                summary = json.loads(str(dispatch.get("content", "{}"))).get("summary")
            except (TypeError, json.JSONDecodeError):
                summary = None
            response = summary if isinstance(summary, str) else "Worker completed."
            return scripted(
                "",
                ModelToolCall(
                    "router-finish-dispatch",
                    "finish",
                    json.dumps({"response": response}),
                ),
            )

        if worker_request:
            if any(item.get("tool_call_id") == "generation-write" for item in tool_results):
                self.tool_completed.set()
                return scripted("old generation tool completed")
            if "session-a" in latest_user:
                async def write_after_release() -> AsyncIterator[ModelStreamEvent]:
                    self.session_a_started.set()
                    await self.release_a.wait()
                    yield ModelCompleted(
                        ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(ModelToolCall(
                                    "generation-write",
                                    "write_file",
                                    json.dumps({
                                        "path": "generation.txt",
                                        "content": "old Run survived",
                                    }),
                                ),),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    )

                return write_after_release()
            if "new work continues" in latest_user:
                return scripted("answer from queued input")
            return scripted("answer from session B")

        latest_user_index = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        tool_result = next(
            (
                item
                for item in messages[latest_user_index + 1 :]
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


def _application(service: AgentService) -> dict[str, Any]:
    return cast(dict[str, Any], service.config_view()["application"])


@pytest.mark.asyncio
async def test_http_save_preserves_foreground_schedule_confirmation_and_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    class PreservingGenerationProvider(_GenerationProvider):
        async def complete(self, **kwargs: Any) -> ModelResponse:
            if "new work continues" in json.dumps(kwargs["messages"]):
                return _response(
                    '{"action":"keep","task_goal":null,"completion_boundary":null}'
                )
            return await super().complete(**kwargs)

    providers: list[_GenerationProvider] = []

    def provider_factory(_configuration: object) -> _GenerationProvider:
        provider = PreservingGenerationProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", provider_factory)
    service = AgentService(home, ConfigLoader(home).load(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    browser = await service.register_client("web")
    pid = os.getpid()
    identity = service.service_instance_id
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)

        async def wait_for_assistant_response(text: str) -> None:
            async with asyncio.timeout(10):
                while not any(
                    message.get("role") == "assistant" and message.get("content") == text
                    for message in cast(
                        list[dict[str, object]],
                        workspace.session_snapshot(cli.session_id)["messages"],
                    )
                ):
                    await asyncio.sleep(0.01)

        old_runtime = workspace.resources
        assert old_runtime is not None
        old_state = workspace.workspace_state
        socket = cli._socket
        claim = (cli.session_id, cli.claim_version, cli.claim_credential)
        await cli.management("permission", {"permission_level": "read-only"})
        confirmation = _ConfirmationBarrier()
        cli.confirmation.bind_presenter(confirmation)
        await cli.submit_user_input("session-a")
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
        worker_records = workspace.worker_store(cli.session_id).load().workers
        assert len(worker_records) == 1 and worker_records[0].state == "working"
        foreground_task = workspace.foreground_work_store(cli.session_id).load().tasks[-1]
        assert foreground_task.state == "waiting_for_workers"
        assert workspace.schedule_service.status_snapshot().to_dict()["active_job_count"] == 1
        revision = service.config_view()["revision"]
        headers = {
            "Authorization": f"Bearer {cli.token}",
            "X-Aide-CSRF": cli.token,
            "X-Aide-Client": browser.client_id,
            "X-Aide-Control": cast(str, browser.web_control_credential),
        }
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(
                server.make_url("/api/v1/events"),
                headers={**headers, "Origin": str(server.make_url("/")).rstrip("/")},
                protocols=("aide-v1",),
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
                assert saved["application"]["status"] == "next-run-required"
                assert saved["application"]["active_revision"] == revision
                assert not old_provider.closed
                await cli.submit_user_input("new work continues")
                old_provider.release_a.set()
                try:
                    await asyncio.wait_for(confirmation.presented.wait(), 5)
                except TimeoutError:
                    pytest.fail("The pending confirmation was not presented within 5 seconds.")
                assert _application(service)["status"] == "active"
                assert not old_provider.closed
                assert confirmation.respond is not None
                assert confirmation.respond(confirmation.token, "approved")
                async with asyncio.timeout(10):
                    while not any(
                        worker.report is not None
                        and worker.report.summary == "old generation tool completed"
                        for worker in workspace.worker_store(cli.session_id).load().workers
                    ):
                        await asyncio.sleep(0.01)
                assert (project / "generation.txt").read_text(
                    encoding="utf-8"
                ) == "old Run survived"
                assert _application(service)["status"] == "active"
                assert not old_provider.closed
                old_provider.release_schedule.set()
                assert _application(service)["status"] == "active"
                assert not web_socket.closed
                await web_socket.ping(b"still-connected")
                assert service.client(browser.client_id).connected
        assert old_provider.tool_completed.is_set()
        assert not old_provider.session_a_cancelled.is_set()
        assert not old_provider.schedule_cancelled.is_set()
        assert not old_provider.closed
        assert workspace.resources is old_runtime
        assert workspace.workspace_state is old_state
        assert workspace.configuration.runtime.max_iterations == 81
        assert workspace.configuration.memory.batch_size == 11
        assert (cli.session_id, cli.claim_version, cli.claim_credential) == claim
        assert cli._socket is socket and socket is not None and not socket.closed
        assert os.getpid() == pid and service.service_instance_id == identity
        assert service.client_permission(cli.client_id).current() == "read-only"
        assert service.client_permission(cli.client_id).configured() == "full-access"
        assert (
            workspace_loops(workspace)[
                cli.session_id
            ].loop._create_executor()._tool_gateway._permission_context.configured_schedule_level
            == "full-access"
        )
        assert (
            workspace._schedule_loops[job.job_id].loop._create_executor()._permission_control.configured()
            == "full-access"
        )
        await workspace.schedule_service.pause_and_wait_idle()
        workspace.schedule_service.resume()
        jobs = await workspace.schedule_service.public_snapshot()
        assert next(item for item in jobs if item.job_id == job.job_id).state.last_status == "ok"
        assert len(workspace._schedule_loops) == 1
        await wait_for_assistant_response("answer from queued input")
        await cli.submit_user_input("session-b")
        await wait_for_assistant_response("answer from session B")
        assert len(providers) == 1
        persisted = Session.load(cast(Any, old_state), cli.session_id)
        assert any(
            message.get("content") == "answer from session B" for message in persisted.messages
        )
        assert any(
            worker.report is not None
            and worker.report.summary == "old generation tool completed"
            for worker in workspace.worker_store(cli.session_id).load().workers
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
async def test_save_preserves_dream_and_subsequent_dream_uses_new_settings(
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
    service = AgentService(home, ConfigLoader(home).load(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    other: ServiceClient | None = None
    dream_task: asyncio.Task[dict[str, object]] | None = None
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        other = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old_runtime = workspace.resources
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
        assert _application(service)["status"] == "next-run-required"
        assert _application(service)["restart_required"] is False
        assert not old_provider.closed and not dream_task.done()
        assert workspace.resources is old_runtime and not workspace._closed
        old_provider.release_background.set()
        result = await asyncio.wait_for(dream_task, 5)
        dream_result = cast(dict[str, object], result["dream_result"])
        assert dream_result["processed_count"] == 1 and dream_result["cursor"] == 1
        assert _application(service)["status"] == "next-run-required"
        assert not old_provider.background_cancelled.is_set() and not old_provider.closed
        assert workspace.resources is not None and workspace.resources is old_runtime
        await workspace.resources.memory_manager.append_summary(
            "Another preference.", datetime.now(UTC)
        )
        next_result = await cli.management("dream", {})
        assert cast(dict[str, object], next_result["dream_result"])["processed_count"] == 1
        assert len(providers) == 1 and providers[0].dream_started.is_set()
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
async def test_save_preserves_auto_title_after_foreground_run_has_finished(
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
    service = AgentService(home, ConfigLoader(home).load(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    cli: ServiceClient | None = None
    title_idle: asyncio.Task[None] | None = None

    async def wait_for_answer(client: ServiceClient) -> None:
        outputs: list[str] = []
        try:
            async with asyncio.timeout(10):
                while "answer from session B" not in outputs:
                    try:
                        outputs.extend(await _client_output(client))
                    except TimeoutError:
                        continue
        except TimeoutError as error:
            session_id = client.session_id
            workspace = service.workspace(client.workspace_id)
            workers = workspace.worker_store(session_id).load().workers
            foreground = workspace.foreground_work_store(session_id).load()
            messages = workspace.session_snapshot(session_id)["messages"]
            raise AssertionError(
                f"outputs={outputs!r}; workers={[(item.state, None if item.report is None else item.report.summary) for item in workers]!r}; "
                f"tasks={[(item.state, item.execution_role) for item in foreground.tasks]!r}; "
                f"messages={messages!r}"
            ) from error

    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old_runtime = workspace.resources
        assert old_runtime is not None
        await cli.submit_user_input("Complete this foreground message")
        async with asyncio.timeout(5):
            while not providers:
                await asyncio.sleep(0)
        old_provider = providers[0]
        await asyncio.wait_for(old_provider.title_started.wait(), 5)
        await wait_for_answer(cli)
        old_loop = workspace_loops(workspace)[cli.session_id].loop
        async with asyncio.timeout(5):
            while old_loop.has_active_run:
                await asyncio.sleep(0)
        generation_id = old_loop.generation_id
        await cli.submit_user_input("Complete a second foreground message")
        await wait_for_answer(cli)
        assert old_loop.generation_id == generation_id
        title_idle = asyncio.create_task(old_loop.wait_for_restore_idle())
        await asyncio.sleep(0)
        assert not title_idle.done()
        await old_loop.session.wait_for_pending_persist()
        run_updated_at = old_loop.session.updated_at
        await service.update_configuration(
            "save-during-title",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 85}},
        )
        assert _application(service)["status"] == "next-run-required"
        assert _application(service)["restart_required"] is False
        assert workspace.resources is old_runtime and not old_provider.closed
        assert not title_idle.done() and not workspace._closed
        old_provider.release_background.set()
        await asyncio.wait_for(asyncio.shield(title_idle), 5)
        assert _application(service)["status"] == "next-run-required"
        assert not old_provider.background_cancelled.is_set() and not old_provider.closed
        current = workspace_loops(workspace)[cli.session_id].loop.session
        assert current.metadata["title"] == "Completed background title"
        assert current.updated_at == run_updated_at
        await workspace.release(cli.client_id, cli.session_id)
        persisted = Session.load(cast(WorkspaceState, workspace.workspace_state), cli.session_id)
        assert persisted.metadata["title"] == "Completed background title"
        assert workspace.configuration.runtime.max_iterations == 85
    finally:
        for provider in providers:
            provider.release_background.set()
        if title_idle is not None:
            await asyncio.gather(title_idle, return_exceptions=True)
        if cli is not None:
            await cli.close()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_save_preserves_real_restore_transaction(
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
    service = AgentService(home, ConfigLoader(home).load(), reconnect_timeout=3600)
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
        old_runtime = workspace.resources
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
        assert _application(service)["status"] == "next-run-required"
        assert _application(service)["restart_required"] is False
        assert workspace.resources is old_runtime and not workspace._closed
        assert not restore_task.done() and target.read_bytes() == b"current branch"
        release.set()
        result = await asyncio.wait_for(restore_task, 5)
        assert result.get("restore_result") is not None
        assert _application(service)["status"] == "next-run-required"
        assert not cancelled.is_set()
        assert target.read_bytes() == b"before restore"
        assert not RestoreManager(state, session.session_id).has_pending_transaction()
        assert workspace.configuration.runtime.max_iterations == 86
        assert workspace_loops(workspace)[session.session_id].loop is claim.loop
        assert claim.loop.session.session_id == session.session_id
    finally:
        release.set()
        if restore_task is not None:
            await asyncio.gather(restore_task, return_exceptions=True)
        await service.stop()
