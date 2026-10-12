"""T04 acceptance through Service execution and the public recorder query boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.errors import ModelCallError
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.runtime import AgentService
from aide.templates import render_template
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TraceErrorCategory, TraceKind, TraceOrigin, TracePurpose
from aide.utils.errors import ErrorInfo
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.fixtures.mcp_wire import http_wire_server
from tests.fixtures.trace import wait_for_trace_writer
from tests.service.test_service_concurrency import (
    _claim_version,
    _CollectingSink,
    _ConcurrentProvider,
)


def _response(
    content: str, *, tool: ModelToolCall | None = None, large: bool = False
) -> ModelResponse:
    return ModelResponse(
        AssistantModelMessage(content, () if tool is None else (tool,)),
        ModelUsage(20000 if large else 4, 1, 20001 if large else 5),
        "stop" if tool is None else "tool_calls",
    )


class _SourcesProvider(_ConcurrentProvider):
    """Deterministic model results selected by real request arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.child_started = asyncio.Event()
        self.child_release = asyncio.Event()
        self.schedule_finished = asyncio.Event()
        self.spawned = False

    async def complete(self, **kwargs: Any) -> ModelResponse:
        system = kwargs["messages"][0]["content"]
        if system == render_template("mcp-keyword-system-prompt.md"):
            return _response('["echo", "repeat"]')
        if kwargs["model"] == "memory-model":
            return _response("Captured summary or memory result.")
        if kwargs["model"] == "schedule-model":
            self.schedule_finished.set()
            return _response("Scheduled answer.")
        return await super().complete(**kwargs)

    async def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        model = kwargs["model"]
        if model == "title-model":
            yield ModelCompleted(_response("Trace acceptance"))
            return
        if model == "child-model":
            self.child_started.set()
            await self.child_release.wait()
            yield TextDelta("Child answer.")
            yield ModelCompleted(_response("Child answer."))
            return
        messages = kwargs["messages"]
        user = next(m["content"] for m in reversed(messages) if m.get("role") == "user")
        if "delegate" in user and not self.spawned:
            names = {tool["function"]["name"] for tool in kwargs["tools"]}
            if "spawn_agent" not in names:
                call = ModelToolCall("search", "tool_search", '{"query":"spawn agent"}')
            else:
                self.spawned = True
                call = ModelToolCall("spawn", "spawn_agent", '{"title":"Child","task":"Reply."}')
            yield ModelCompleted(_response("", tool=call))
            return
        if "delegate" in user:
            await self.child_started.wait()
        yield TextDelta("Main answer.")
        yield ModelCompleted(_response("Main answer.", large=True))


@pytest.mark.asyncio
async def test_service_nine_sources_and_concurrent_subagent_match_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        await _exercise_sources(tmp_path / str(recording), recording, monkeypatch)
        for recording in (False, True)
    ]
    assert outcomes[0] == outcomes[1] == ["Main answer.", "Main answer.", "Child answer.", 2]


async def _exercise_sources(
    root: Path,
    recording: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> list[object]:
    home = AgentHome(root / "home")
    home.initialize()
    path = root / "workspace"
    path.mkdir()
    provider = _SourcesProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    async with http_wire_server({}) as (_wire, mcp):
        config = MINIMAL_VALID_CONFIG.replace("context_window = 8192", "context_window = 32768")
        config = config.replace("compact_ratio = 0.9", "compact_ratio = 0.5")
        for route, model in (
            ("title", "title-model"),
            ("memory", "memory-model"),
            ("schedule", "schedule-model"),
            ("subagent", "child-model"),
        ):
            config += (
                f'\n[models.providers.primary.models."{model}"]\n'
                "context_window = 32768\nmax_output = 1024\ntemperature = 0\n"
                'reasoning_effort = "mid"\ntimeout = 30\n'
                f'[models.routes.{route}]\nprovider_id = "primary"\nmodel = "{model}"\n'
            )
        config += (
            '\n[mcp.servers.remote]\nenabled = true\ntransport = "streamable-http"\n'
            f"url = {json.dumps(mcp.url)}\n"
        )
        (home.path / "config.toml").write_text(config, encoding="utf-8")
        service = AgentService(home, ConfigLoader(home).load())
        await service.start()
        if not recording:
            await service._stop_trace_storage()
            service._trace_recorder = NoOpTraceRecorder()
        try:
            clients = [await service.register_client("cli") for _ in range(2)]
            sink = _CollectingSink()
            for client in clients:
                await service.connect_client(client.client_id, sink)
            workspace = await service.attach_workspace(clients[0].client_id, path)
            await service.attach_workspace(clients[1].client_id, path)
            sessions = [
                await workspace.create_draft(
                    clients[index].client_id,
                    creation_scope="chat",
                    reuse_startup_session=False,
                )
                for index in range(2)
            ]
            claims = [
                await service.claim(clients[index].client_id, workspace.workspace_id, sid)
                for index, sid in enumerate(sessions)
            ]

            async def run(index: int, text: str, run_id: str) -> None:
                await workspace.input(
                    clients[index].client_id,
                    sessions[index],
                    _claim_version(claims[index]),
                    text,
                    run_id,
                )
                await asyncio.wait_for(sink.wait_for("run.completed", run_id), 10)

            async with asyncio.timeout(15):
                await asyncio.gather(run(0, "delegate", "run-a"), run(1, "other", "run-b"))
                # The child still runs after its creating foreground run completes.
                assert provider.child_started.is_set() and not provider.child_release.is_set()
                provider.child_release.set()
                coordinator = workspace._subagent_coordinator(sessions[0])
                assert coordinator is not None
                children = coordinator.list().items
                assert len(children) == 1
                await coordinator.wait([children[0].agent_id], timeout_ms=5000)
                await run(1, "continue", "run-summary")
                assert workspace.resources is not None
                manager = workspace.resources.memory_manager
                await manager.append_summary("Dream input", datetime.now(UTC))
                claim = workspace._claims[sessions[1]]
                dream = await service.handle_management(
                    clients[1].client_id,
                    workspace.workspace_id,
                    sessions[1],
                    "dream",
                    {"request_id": "dream"},
                    claim_version=claim.version,
                    claim_credential=claim.credential,
                )
                assert cast(dict[str, Any], dream)["dream_result"]["error"] is None
                job = ScheduleJob(
                    job_id="123e4567-e89b-42d3-a456-426614174000",
                    message="scheduled",
                    schedule=JobSchedule.every(3600),
                    created_at_ms=1,
                    updated_at_ms=1,
                )
                await workspace.schedule_service.add_user_job(job)
                await provider.schedule_finished.wait()
                while workspace.schedule_service.status_snapshot().active_job_count:
                    await asyncio.sleep(0.01)

            outputs = [
                workspace._claims[sid].loop.session.messages[-1]["content"] for sid in sessions
            ]
            child_record = coordinator.get(children[0].agent_id)
            assert child_record is not None
            outputs.append(child_record.result)
            outputs.append(cast(dict[str, Any], dream)["dream_result"]["processed_count"])
            outcome = outputs
            if recording:
                recorder = service.trace_recorder
                assert isinstance(recorder, SQLiteTraceRecorder)
                await wait_for_trace_writer(recorder)
                records = (await recorder.list_records(limit=200)).items
                assert {record.purpose for record in records} == set(TracePurpose)
                assert all(record.endpoint is None for record in records)
                roots = {r.run_id: r for r in records if r.purpose is TracePurpose.AGENT_RUN}
                assert roots["run-a"].root_id != roots["run-b"].root_id
                child = next(r for r in records if r.purpose is TracePurpose.SUBAGENT_RUN)
                assert child.origin is TraceOrigin.SUBAGENT
                assert child.owner_session_id == sessions[0]
                assert child.run_id == "run-a"
                assert child.parent_root_id == roots["run-a"].root_id
                assert child.restore_run_token == roots["run-a"].restore_run_token
                assert child.agent_id == children[0].agent_id
                for record in records:
                    if record.purpose in {TracePurpose.DREAM, TracePurpose.MCP_KEYWORDS}:
                        assert record.owner_session_id is None and record.run_id is None
                    if record.purpose in {
                        TracePurpose.CONVERSATION_SUMMARY,
                        TracePurpose.ACTION_SUMMARY,
                    }:
                        assert record.run_id == "run-summary"
                        assert record.owner_session_id == sessions[1]
                    detail = await recorder.get_record(record.record_id)
                    assert detail is not None
                    if record.kind is TraceKind.MODEL:
                        assert detail.capture_issues
                    else:
                        assert detail.request is not None and detail.result is not None
                        assert record.capture_state.value == "complete"
                        assert record.parent_record_id is not None
        finally:
            provider.child_release.set()
            await service.stop()
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("success_attempt", [1, 3, 5, None])
async def test_service_retry_results_match_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    success_attempt: int | None,
) -> None:
    outcomes = [
        await _exercise_retries(tmp_path / str(recording), recording, success_attempt, monkeypatch)
        for recording in (False, True)
    ]
    assert outcomes[0] == outcomes[1]


async def _exercise_retries(
    root: Path,
    recording: bool,
    success_attempt: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object, int]:
    home = AgentHome(root / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    failures = 5 if success_attempt is None else success_attempt - 1
    error = ModelCallError(ErrorInfo("provider_timeout", "Timed out.", retryable=True))
    scripts = [StreamScript((), error) for _ in range(failures)]
    if success_attempt is not None:
        scripts.append(StreamScript((TextDelta("answer"), ModelCompleted(_response("answer")))))
    scripted = ScriptedFakeProvider(streams=scripts)

    class Provider(_ConcurrentProvider):
        def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
            if str(kwargs["messages"][0]["content"]).startswith("Generate a concise title"):
                return super().stream(**kwargs)
            return scripted.stream(**kwargs)

    provider = Provider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    if not recording:
        await service._stop_trace_storage()
        service._trace_recorder = NoOpTraceRecorder()
    service.model_router._clock = FakeClock(datetime.now(UTC))
    path = root / "workspace"
    path.mkdir()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        sid = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, sid)
        await workspace.input(client.client_id, sid, _claim_version(claim), "retry", "retry-run")
        await asyncio.wait_for(sink.wait_for("run.completed", "retry-run"), 10)
        terminal = workspace._claims[sid].loop.session.messages[-1]
        outcome = (terminal["content"], terminal["status"], len(scripted.stream_requests))
        if recording:
            recorder = service.trace_recorder
            assert isinstance(recorder, SQLiteTraceRecorder)
            await wait_for_trace_writer(recorder)
            (record,) = (
                r
                for r in (await recorder.list_records()).items
                if r.purpose is TracePurpose.AGENT_RUN
            )
            assert record.retry_count == (4 if success_attempt is None else success_attempt - 1)
            assert record.status.value == ("failed" if success_attempt is None else "success")
            detail = await recorder.get_record(record.record_id)
            assert detail is not None
            if success_attempt is None:
                assert detail.request is None and detail.result is None
                assert record.error_message and "Timed out" in record.error_message
    finally:
        await service.stop()
    return outcome


@pytest.mark.asyncio
async def test_service_cancellation_after_output_matches_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        await _exercise_cancel(tmp_path / str(recording), recording, monkeypatch)
        for recording in (False, True)
    ]
    assert outcomes[0] == outcomes[1]


async def _exercise_cancel(
    root: Path,
    recording: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object]:
    home = AgentHome(root / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    provider = _ConcurrentProvider(early_a_delta=True)
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    if not recording:
        await service._stop_trace_storage()
        service._trace_recorder = NoOpTraceRecorder()
    path = root / "workspace"
    path.mkdir()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        sid = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, sid)
        await workspace.input(
            client.client_id, sid, _claim_version(claim), "session-a", "cancel-run"
        )
        await asyncio.wait_for(provider.session_a_started.wait(), 5)
        await workspace.cancel(client.client_id, sid, _claim_version(claim), "cancel-run")
        await asyncio.wait_for(sink.wait_for("run.completed", "cancel-run"), 5)
        terminal = workspace._claims[sid].loop.session.messages[-1]
        if recording:
            recorder = service.trace_recorder
            assert isinstance(recorder, SQLiteTraceRecorder)
            await wait_for_trace_writer(recorder)
            (record,) = (
                r
                for r in (await recorder.list_records()).items
                if r.purpose is TracePurpose.AGENT_RUN
            )
            assert record.status.value == "failed"
            assert record.error_category is TraceErrorCategory.CANCELLED
            assert record.retry_count == 0
        return terminal["content"], terminal["status"]
    finally:
        provider.release_a.set()
        await service.stop()
