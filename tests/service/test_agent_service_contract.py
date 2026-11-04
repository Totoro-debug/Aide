from __future__ import annotations

import asyncio
import gc
import json
import os
import weakref
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from aide.agent.loop import AgentRunExecutor
from aide.agent.message_bus import InboundMessage
from aide.agent.permission import PermissionSnapshot, RuntimePermissionControl
from aide.agent.session.session import Session
from aide.agent.tools.core.exec_policy import ResolvedExecShell
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from aide.provider.session_configuration import SessionModelConfiguration
from aide.service.errors import ServiceError
from aide.service.execution import SessionExecution
from aide.service.runtime import AgentService
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock
from tests.fixtures.service_runtime import register_project, workspace_loops
from tests.fixtures.trace import wait_for_trace_writer
from tests.service.test_foreground_questions import _wait_for_idle
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


class _WorkerFlowProvider:
    """Drive one Router to Worker to Router exchange through a real file read."""

    def __init__(self) -> None:
        self.worker_requests: list[list[dict[str, object]]] = []
        self.retry_failed_dispatch = False
        self.worker_tool = "read_file"

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
        del tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        system = messages[0].get("content", "") if messages else ""
        if isinstance(system, str) and system.startswith("Generate a concise title"):
            return self._response("Worker handoff")
        return self._response(
            '{"action":"replace","task_goal":"delegate the file inspection",'
            '"completion_boundary":"return the Worker result"}'
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
        del model, max_output, temperature, reasoning_effort, timeout, continuation
        tool_names = {
            name
            for tool in tools
            if isinstance((name := tool.get("name")), str)
            or isinstance(
                (
                    name := tool.get("function", {}).get("name")
                    if isinstance(tool.get("function"), dict)
                    else None
                ),
                str,
            )
        }

        system_prompt = (
            messages[0].get("content", "")
            if messages and isinstance(messages[0].get("content"), str)
            else ""
        )
        worker_request = system_prompt.startswith("You are an Aide Worker.")

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if {"ask_user", "dispatch_worker", "finish"} <= tool_names:
                tool_messages = [item for item in messages if item.get("role") == "tool"]
                dispatch_result = next(
                    (
                        item
                        for item in reversed(tool_messages)
                        if item.get("name") == "dispatch_worker"
                    ),
                    None,
                )
                finish_result = next(
                    (item for item in reversed(tool_messages) if item.get("name") == "finish"),
                    None,
                )
                if dispatch_result is None or (
                    self.retry_failed_dispatch
                    and not str(dispatch_result.get("content", "")).startswith('{"worker_id":')
                ):
                    response = self._response(
                        "",
                        ModelToolCall(
                            "router-dispatch",
                            "dispatch_worker",
                            json.dumps(
                                {
                                    "task_id": "inspect-report",
                                    "task": "Read report.txt and report its first line.",
                                    "context": [{"text": "Only inspect report.txt."}],
                                }
                            ),
                        ),
                    )
                elif finish_result is None:
                    report = json.loads(str(dispatch_result.get("content", "{}")))
                    summary = report.get("summary")
                    response = self._response(
                        "",
                        ModelToolCall(
                            "router-finish",
                            "finish",
                            json.dumps(
                                {
                                    "response": (
                                        f"Worker found: {summary}"
                                        if isinstance(summary, str)
                                        else "Worker report pending."
                                    )
                                }
                            ),
                        ),
                    )
                else:
                    latest_user = next(
                        (item for item in reversed(messages) if item.get("role") == "user"),
                        {},
                    )
                    user_content = latest_user.get("content")
                    summary = next(
                        (
                            line.removeprefix("Report: ")
                            for line in str(user_content).splitlines()
                            if line.startswith("Report: ")
                        ),
                        None,
                    )
                    response = self._response(
                        f"Worker found: {summary}"
                        if summary is not None
                        else "Router delivered the Worker report to the user."
                    )
            elif worker_request:
                self.worker_requests.append([dict(item) for item in messages])
                tool_messages = [item for item in messages if item.get("role") == "tool"]
                read_result = next(
                    (
                        item
                        for item in reversed(tool_messages)
                        if item.get("name") == self.worker_tool
                    ),
                    None,
                )
                if read_result is None:
                    response = self._response(
                        "",
                        ModelToolCall(
                            "worker-read",
                            self.worker_tool,
                            json.dumps(
                                {
                                    "path": "report.txt",
                                    **(
                                        {"content": "authorized Worker write"}
                                        if self.worker_tool == "write_file"
                                        else {}
                                    ),
                                }
                            ),
                        ),
                    )
                else:
                    file_content = str(read_result.get("content", ""))
                    response = self._response(f"First line: {file_content.splitlines()[0]}")
            else:
                response = self._response("Preparation complete.")
            yield ModelCompleted(response)

        return emit()

    async def close(self) -> None:
        return None

    @staticmethod
    def _response(content: str, tool_call: ModelToolCall | None = None) -> ModelResponse:
        return ModelResponse(
            message=AssistantModelMessage(
                content=content,
                tool_calls=() if tool_call is None else (tool_call,),
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop" if tool_call is None else "tool_calls",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("client_kind", ["cli", "web"])
@pytest.mark.parametrize("trace_failure", [False, True])
async def test_worker_dispatch_runs_between_router_and_user_through_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_kind: str, trace_failure: bool
) -> None:
    provider = _WorkerFlowProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "report.txt").write_text("release 42\nsecond line\n", encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    trace_failures: list[str] = []
    if trace_failure:
        original_begin = service.trace_recorder.begin_call

        def begin_call(context: Any, metadata: Any) -> Any:
            if context.agent_id is not None:
                trace_failures.append(context.agent_id)
                raise OSError("injected trace storage failure")
            return original_begin(context, metadata)

        monkeypatch.setattr(service.trace_recorder, "begin_call", begin_call)
    try:
        client = await service.register_client(client_kind)
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        claimed = await service.claim(client.client_id, workspace.workspace_id, session_id)
        accepted = await service.submit_user_input(
            client.client_id,
            workspace.workspace_id,
            session_id,
            _claim_version(claimed),
            "Please inspect the report. Parent-only detail: hidden from the Worker.",
            f"worker-flow-{client_kind}",
        )
        run_id = accepted["run_id"]
        assert isinstance(run_id, str)
        await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=8)

        async def wait_for_report_delivery() -> None:
            while True:
                current_workers = workspace.worker_store(session_id).load().workers
                if (
                    current_workers
                    and current_workers[0].report_delivery_state == "consumed"
                ):
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_report_delivery(), timeout=8)
        await _wait_for_idle(workspace, session_id)

        session_state = workspace_loops(workspace)[session_id]
        await session_state.loop.session.wait_for_pending_persist()
        persisted = Session.load(workspace.workspace_state, session_id)
        workers = workspace.worker_store(session_id).load().workers

        assert len(workers) == 1
        worker = workers[0]
        assert worker.state == "ready"
        assert worker.report is not None
        assert worker.report.summary == "First line: release 42"
        assert worker.usage == {
            "model_calls": 2,
            "input_tokens": 2,
            "output_tokens": 2,
            "total_tokens": 4,
        }
        assert worker.report_delivery_state == "consumed"
        assert "Parent-only detail" not in json.dumps(provider.worker_requests)
        assert any(
            item.get("role") == "tool" and item.get("name") == "read_file"
            for item in worker.conversation
        )
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        records = (await recorder.list_records(limit=200)).items
        worker_records = [r for r in records if r.agent_id == worker.worker_id]
        if trace_failure:
            assert trace_failures and set(trace_failures) == {worker.worker_id}
            assert worker_records == []
        else:
            assert len(worker_records) == 3  # two model calls and the real file read
            assert all(r.owner_session_id == session_id for r in worker_records)
            assert all(r.parent_root_id is not None for r in worker_records)
            assert any(r.tool_name == "read_file" for r in worker_records)
        report_commit = next(
            item
            for item in persisted.messages
            if item.get("role") == "assistant"
            and item.get("status") == "completed"
            and item.get("worker_report_receipts")
        )
        assert report_commit["worker_report_receipts"] == [
            {
                "worker_id": worker.worker_id,
                "report_id": worker.report.report_id,
                "receipt_id": worker.report_receipt_id,
            }
        ]
        assert any(
            event.get("type") == "run.output" and "release 42" in str(event.get("payload"))
            for event in sink.events
        )
        credential = workspace.require_claim(
            client.client_id, session_id, _claim_version(claimed)
        ).credential
        await service.delete_session(
            client.client_id,
            workspace.workspace_id,
            session_id,
            _claim_version(claimed),
            credential,
            "delete-worker-session",
        )
        assert not (workspace.workspace_state.workers_directory / session_id).exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_worker_exact_call_confirmation_retains_owner_and_permission_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _WorkerFlowProvider()
    provider.worker_tool = "write_file"
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        foreign = await service.register_client("web")
        service.client_permission(client.client_id).select("read-only")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        sid = await workspace.create_draft(client.client_id, creation_scope="chat")
        claimed = await service.claim(client.client_id, workspace.workspace_id, sid)
        accepted = await service.submit_user_input(
            client.client_id,
            workspace.workspace_id,
            sid,
            _claim_version(claimed),
            "write report",
            "permission",
        )

        async def wait_confirmation() -> None:
            while not any(e.get("type") == "confirmation.requested" for e in sink.events):
                sink.changed.clear()
                await sink.changed.wait()

        await asyncio.wait_for(wait_confirmation(), 8)
        event = next(e for e in sink.events if e.get("type") == "confirmation.requested")
        payload = cast(dict[str, Any], event["payload"])
        worker = workspace.worker_store(sid).load().workers[0]
        assert worker.permission_level == "read-only"
        assert event["run_id"] == worker.execution_id
        assert event["session_id"] == sid
        assert not (path / "report.txt").exists()
        with pytest.raises(ServiceError):
            service.decide_confirmation(foreign.client_id, payload["token"], "approved")
        service.client_permission(client.client_id).select("full-access")
        service.decide_confirmation(client.client_id, payload["token"], "approved")
        await asyncio.wait_for(sink.wait_for("run.completed", cast(str, accepted["run_id"])), 8)
        assert (path / "report.txt").read_text(encoding="utf-8") == "authorized Worker write"
        worker = workspace.worker_store(sid).load().workers[0]
        assert worker.permission_level == "read-only"
        with pytest.raises(ServiceError):
            service.decide_confirmation(client.client_id, payload["token"], "approved")
        await _wait_for_idle(workspace, sid)
        credential = workspace.require_claim(
            client.client_id, sid, _claim_version(claimed)
        ).credential
        inspected = await service.inspect_restore(
            client.client_id,
            workspace.workspace_id,
            sid,
            _claim_version(claimed),
            credential,
            "inspect",
            1,
        )
        assert "restore_plan" in inspected, inspected
        restored = await service.handle_management(
            client.client_id,
            workspace.workspace_id,
            sid,
            "restore/execute",
            {"request_id": "restore", "plan": {"anchor_id": 1}, "mode": "files"},
            claim_version=_claim_version(claimed),
            claim_credential=credential,
        )
        assert "restore_result" in restored, restored
        assert not (path / "report.txt").exists()
        assert workspace.worker_store(sid).load().workers == ()
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_report_write_failure_recovers_completed_worker_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart: bool,
) -> None:
    provider = _WorkerFlowProvider()
    provider.retry_failed_dispatch = not restart
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "report.txt").write_text("release 42\n", encoding="utf-8")
    original_replace = os.replace
    report_write_failures = 0

    def replace(source: Any, target: Any) -> None:
        nonlocal report_write_failures
        target_path = Path(target)
        if "workers" in target_path.parts and target_path.name == "state.json":
            value = json.loads(Path(source).read_text(encoding="utf-8"))
            if value["workers"][0].get("report") is not None:
                report_write_failures += 1
                if restart or report_write_failures == 1:
                    raise OSError("injected report publication failure")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        sid = await workspace.create_draft(client.client_id, creation_scope="chat")
        claimed = await service.claim(client.client_id, workspace.workspace_id, sid)
        accepted = await service.submit_user_input(
            client.client_id,
            workspace.workspace_id,
            sid,
            _claim_version(claimed),
            "inspect report",
            "fault",
        )
        await asyncio.wait_for(sink.wait_for("run.completed", cast(str, accepted["run_id"])), 8)
        failure_count = 2 if restart else 1

        async def wait_for_report_write_failure() -> None:
            while report_write_failures < failure_count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_report_write_failure(), 8)
        worker = workspace.worker_store(sid).load().workers[0]
        if not restart:
            async def wait_for_consumed_report() -> None:
                while True:
                    current = workspace.worker_store(sid).get(worker.worker_id)
                    if current is not None and current.report_delivery_state == "consumed":
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_consumed_report(), 8)
            worker = workspace.worker_store(sid).get(worker.worker_id)
            assert worker is not None
            assert worker.report is not None and worker.report.summary == "First line: release 42"
            assert worker.report_delivery_state == "consumed"
            assert len(provider.worker_requests) == 2
            return
        assert worker.report is None
        assert worker.conversation[-1]["content"] == "First line: release 42"
    finally:
        await service.stop()
    monkeypatch.setattr(os, "replace", original_replace)
    provider.retry_failed_dispatch = True
    restarted = AgentService(home, ConfigLoader(home).load())
    await restarted.start()
    try:
        client = await restarted.register_client("web")
        sink = _CollectingSink()
        await restarted.connect_client(client.client_id, sink)
        workspace = await restarted.attach_workspace(client.client_id, path)
        claimed = await restarted.claim(client.client_id, workspace.workspace_id, sid)
        recovered = workspace.worker_store(sid).get(worker.worker_id)
        assert recovered is not None and recovered.report is not None
        assert recovered.report.summary == "First line: release 42"
        task = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])[-1]
        accepted = await restarted.continue_foreground_task(
            client.client_id,
            workspace.workspace_id,
            sid,
            _claim_version(claimed),
            task["task_id"],
            task["revision"],
            "recover",
        )
        await asyncio.wait_for(sink.wait_for("run.completed", cast(str, accepted["run_id"])), 8)

        async def wait_for_recovered_delivery() -> None:
            while True:
                current = workspace.worker_store(sid).get(worker.worker_id)
                if current is not None and current.report_delivery_state == "consumed":
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_recovered_delivery(), 8)
        await _wait_for_idle(workspace, sid)
        recovered = workspace.worker_store(sid).get(worker.worker_id)
        assert recovered is not None and recovered.report_delivery_state == "consumed"
        assert len(provider.worker_requests) == 2
        assert any(
            e.get("type") == "run.output" and "release 42" in str(e.get("payload"))
            for e in sink.events
        )
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_service_restart_recovers_an_interrupted_worker_report(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
    foreground = workspace.foreground_work_store(session_id)
    foreground.begin_task(
        "interrupted-task",
        "router-execution",
        generation=1,
        execution_role="router",
    )
    foreground.suspend_for_worker_reports(
        "interrupted-task", generation=1, execution_id="router-execution"
    )
    store = workspace.worker_store(session_id)
    worker = store.create(
        parent_task_id="interrupted-task",
        parent_generation=1,
        task_id="interrupted-worker",
        task="Inspect one file",
        explicit_context=("Read report.txt",),
        runtime_facts={"workspace": str(path), "foreground_execution_id": str(uuid4())},
        permission_snapshot=PermissionSnapshot(
            "workspace-write",
            ResolvedExecShell(
                selector="auto",
                platform="windows",
                family="powershell",
                executable="powershell.exe",
                flags=("-NoProfile",),
                environment=(),
                available=True,
            ),
        ),
        session_model_configuration=SessionModelConfiguration(
            provider_id="controlled", model="test-model", reasoning_effort="low"
        ),
        system_prompt="Worker system prompt",
        initial_message="Assigned task: Inspect one file",
        restore_run_token=None,
    )
    await service.stop()

    restarted = AgentService(home, ConfigLoader(home).load())
    await restarted.start()
    try:
        restarted_client = await restarted.register_client("web")
        restarted_workspace = await restarted.attach_workspace(restarted_client.client_id, path)
        recovered = restarted_workspace.worker_store(session_id).get(worker.worker_id)
        assert recovered is not None
        assert recovered.state == "blocked"
        assert recovered.report is not None
        assert recovered.report.error_code == "service_interrupted"
        assert recovered.report_delivery_state == "pending"
        task = next(
            item
            for item in restarted_workspace.foreground_work_store(session_id).load().tasks
            if item.task_id == "interrupted-task"
        )
        assert task.state == "interrupted"
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_backpressured_run_survives_reconnect_at_29_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    provider = _ConcurrentProvider(early_a_delta=True)
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    clock = FakeClock(datetime(2026, 10, 4, tzinfo=UTC))
    timer_started, wake_timer = asyncio.Event(), asyncio.Event()
    blocked, release_output = asyncio.Event(), asyncio.Event()
    output_delivered = asyncio.Event()

    async def sleep(_seconds: float) -> None:
        timer_started.set()
        await wake_timer.wait()

    service = AgentService(
        home,
        ConfigLoader(home).load(),
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
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        claimed = await service.claim(client.client_id, workspace.workspace_id, session_id)
        state = workspace_loops(workspace)[session_id]
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
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    clock = FakeClock(datetime(2026, 10, 4, tzinfo=UTC))
    timer_started, wake_timer = asyncio.Event(), asyncio.Event()
    blocked, release_output = asyncio.Event(), asyncio.Event()

    async def sleep(_seconds: float) -> None:
        timer_started.set()
        await wake_timer.wait()

    service = AgentService(
        home,
        ConfigLoader(home).load(),
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
        session_a = await workspace.create_draft(first.client_id, creation_scope="chat")
        session_b = await second_workspace.create_draft(second.client_id, creation_scope="chat")
        claim_a = await service.claim(first.client_id, workspace.workspace_id, session_a)
        claim_b = await service.claim(second.client_id, second_workspace.workspace_id, session_b)
        state_a, state_b = workspace_loops(workspace)[session_a], workspace_loops(second_workspace)[session_b]
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
        assert workspace_loops(second_workspace)[session_b] is state_b
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
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        executor = workspace_loops(workspace)[session_id].loop

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
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        first = await service.claim(client.client_id, workspace.workspace_id, session_id)
        state = workspace_loops(workspace)[session_id]
        authority = state.loop.session

        await workspace.release(client.client_id, session_id)

        assert session_id in workspace_loops(workspace)
        assert workspace_loops(workspace)[session_id] is state
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
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        owner = await service.register_client("web")
        first_record, _, _ = await register_project(service, owner.client_id, first_path)
        second_record, _, _ = await register_project(service, owner.client_id, second_path)

        first = await service.open_conversation(
            owner.client_id, project_id=first_record.project_id, create_new=True
        )
        _validator("conversation_open_response").validate({"request_id": "first", **first})
        second = await service.open_conversation(
            owner.client_id, project_id=second_record.project_id, create_new=True
        )
        other = await service.register_client("cli")
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
    service = AgentService(home, ConfigLoader(home).load())
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
        assert list(workspace_loops(workspace)) == [first["session_id"]]
        assert not (path / ".aide" / "sessions" / f"{first['session_id']}.jsonl").exists()
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
    service = AgentService(home, ConfigLoader(home).load())
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
        assert list(workspace_loops(workspace)) == [first["session_id"]]
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
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    service = AgentService(home, ConfigLoader(home).load())
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
        workspace = next(iter(service._workspaces.values()))
        session_id = await workspace.create_draft(clients[0].client_id, creation_scope="chat")
        authority = workspace_loops(workspace)[session_id].loop.session
        for index, (client, level) in enumerate(zip(clients, ("full-access", "read-only"), strict=True)):
            service.client_permission(client.client_id).select(cast(Any, level))
            await service.claim(client.client_id, workspace.workspace_id, session_id)
            claim = workspace._claims[session_id]
            await workspace.input(client.client_id, session_id, claim.version, "warmup", str(index))
            await asyncio.wait_for(sink.wait_for("run.completed", str(index)), 5)
            await workspace_loops(workspace)[session_id].loop.wait_for_restore_idle()
            await workspace.release(client.client_id, session_id)
        assert observed == ["full-access", "read-only"]
        assert workspace_loops(workspace)[session_id].loop.session is authority
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
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, path)
    ids = [await workspace.create_draft(client.client_id, reuse_startup_session=False, creation_scope="chat") for _ in range(2)]
    first, second = (workspace_loops(workspace)[session_id].loop for session_id in ids)

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
    service = AgentService(home, ConfigLoader(home).load())
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
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    home = _home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        await service.claim(client.client_id, workspace.workspace_id, session_id)
        authority = workspace_loops(workspace)[session_id].loop.session
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
            assert workspace_loops(workspace)[session_id].loop.session is authority
            assert not authority._closed
            assert session_id not in workspace._claims
    finally:
        title_release.set()
        await service.stop()
