from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import aide.service.runtime.service as service_runtime
from aide.agent.session.backup_store import FileBackupStore
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.tools.tool_gateway import BuiltInToolCatalog, ModelToolCall
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.runtime import AgentService
from aide.trace.context import workspace_key_for
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TraceErrorCategory, TraceKind
from tests.fixtures.trace import wait_for_trace_writer
from tests.service.test_shared_resources import _CountingProvider, _home, _session_case
from tests.tools.test_shared_tool_context import _RecordingExecHost


class _ToolProvider(_CountingProvider):
    def __init__(self) -> None:
        super().__init__()

    def response(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> ModelResponse:
        user_index = max(i for i, message in enumerate(messages) if message["role"] == "user")
        user_text = str(messages[user_index]["content"])
        label = next(
            (value for value in ("tool-0", "tool-1", "tool-schedule") if value in user_text),
            user_text.splitlines()[-1],
        )
        if label == "tool-schedule":
            self.schedule_started.set()
        tool_names = {
            name
            for item in tools
            if isinstance(item.get("function"), dict)
            and isinstance((name := item["function"].get("name")), str)
        }
        current_tool_results = [
            message for message in messages[user_index + 1 :]
            if message.get("role") == "tool"
        ]
        if {"dispatch_worker", "finish"} <= tool_names:
            worker_report = "Worker reports for automatic continuation:" in user_text
            dispatched = any(item.get("name") == "dispatch_worker" for item in current_tool_results)
            call = (
                ModelToolCall(
                    f"dispatch-{label}",
                    "dispatch_worker",
                    json.dumps({"task_id": label, "task": label, "context": []}),
                )
                if not dispatched and not worker_report
                else ModelToolCall(
                    f"finish-{label}", "finish", json.dumps({"response": "done"})
                )
            )
            return ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            )
        if str(messages[0].get("content", "")).startswith("You are an Aide Worker."):
            completed_calls = {item.get("tool_call_id") for item in current_tool_results}
            calls = (
                ("write", "write_file", json.dumps({"path": "same.txt", "content": label})),
                ("read", "read_file", '{"path":"same.txt"}'),
                ("exec", "exec", '{"command":"Get-Location"}'),
            )
            next_call = next((item for item in calls if item[0] not in completed_calls), None)
            if next_call is None:
                return ModelResponse(
                    message=AssistantModelMessage(content="done"),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )
            call_id, name, arguments = next_call
            if name not in tool_names:
                call_id, name, arguments = (
                    f"search-{call_id}",
                    "tool_search",
                    json.dumps({"query": name}),
                )
            return ModelResponse(
                message=AssistantModelMessage(
                    content="", tool_calls=(ModelToolCall(call_id, name, arguments),)
                ),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            )
        if current_tool_results:
            return ModelResponse(
                message=AssistantModelMessage(content="done"),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="stop",
            )
        return ModelResponse(
            message=AssistantModelMessage(
                content="",
                tool_calls=(
                    ModelToolCall(
                        "write", "write_file", json.dumps({"path": "same.txt", "content": label})
                    ),
                    ModelToolCall("read", "read_file", '{"path":"same.txt"}'),
                    ModelToolCall("exec", "exec", '{"command":"Get-Location"}'),
                ),
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="tool_calls",
        )

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        if str(kwargs["messages"][0].get("content", "")).startswith("Generate a concise title"):
            return super().stream(**kwargs)

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(self.response(kwargs["messages"], kwargs["tools"]))

        return emit()

    async def complete(self, **kwargs: Any) -> ModelResponse:
        if not kwargs["tools"]:
            return await super().complete(**kwargs)
        return self.response(kwargs["messages"], kwargs["tools"])


@pytest.mark.asyncio
async def test_service_foreground_and_schedule_reuse_catalog_with_workspace_and_backup_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    config = (home.path / "config.toml").read_text(encoding="utf-8")
    (home.path / "config.toml").write_text(
        config.replace(
            "compact_ratio = 0.9", 'compact_ratio = 0.9\npermission_level = "full-access"'
        ),
        encoding="utf-8",
    )
    provider = _ToolProvider()
    host = _RecordingExecHost()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    monkeypatch.setattr(service_runtime, "create_exec_host", lambda _shell: host)
    catalog_creations: list[BuiltInToolCatalog] = []
    original_init = BuiltInToolCatalog.__init__

    def create_catalog(catalog: BuiltInToolCatalog, **kwargs: Any) -> None:
        original_init(catalog, **kwargs)
        catalog_creations.append(catalog)

    monkeypatch.setattr(BuiltInToolCatalog, "__init__", create_catalog)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        cases = [await _session_case(service, tmp_path / f"workspace-{i}") for i in range(2)]
        await _session_case(service, tmp_path / "workspace-0")
        for i, case in enumerate(cases):
            (case.workspace.workspace_path / "same.txt").write_text(f"before-{i}", encoding="utf-8")
        await asyncio.gather(
            *(case.submit(f"tool-{i}", f"run-{i}") for i, case in enumerate(cases))
        )
        async def wait_for_task_completion(case: Any) -> None:
            store = case.workspace.foreground_work_store(case.claim.session_id)
            async with asyncio.timeout(10):
                while not any(task.state == "completed" for task in store.load().tasks):
                    await asyncio.sleep(0)

        await asyncio.gather(*(wait_for_task_completion(case) for case in cases))
        assert len(catalog_creations) == 1
        for i, case in enumerate(cases):
            path = case.workspace.workspace_path / "same.txt"
            assert path.read_text(encoding="utf-8") == f"tool-{i}"
            workers = case.workspace.worker_store(case.claim.session_id).load().workers
            assert len(workers) == 1 and workers[0].state == "ready"
            reads = [
                message
                for message in workers[0].conversation
                if message.get("tool_call_id") == "read"
            ]
            assert reads[-1]["content"] == f"tool-{i}"
            backups = FileBackupStore(case.workspace.workspace_state, case.claim.session_id)
            assert len(backups.inspect().entries) == 1
            assert backups.read_backup(1) == f"before-{i}".encode()
        assert set(host.inspected) == {case.workspace.workspace_path for case in cases}
        assert set(host.executed) == set(host.inspected)

        case = cases[1]
        schedule = case.workspace.schedule_service
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="tool-schedule",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await schedule.add_user_job(job)
        await asyncio.wait_for(provider.schedule_started.wait(), timeout=5)
        await asyncio.wait_for(schedule.pause_and_wait_idle(), timeout=5)
        history = Session.load(
            case.workspace.workspace_state,
            job.session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert history.messages[-1]["content"] == "done"
        assert (case.workspace.workspace_path / "same.txt").read_text(
            encoding="utf-8"
        ) == "tool-schedule"
        assert (cases[0].workspace.workspace_path / "same.txt").read_text(
            encoding="utf-8"
        ) == "tool-0"
        assert len(catalog_creations) == 1
        assert (
            len(
                FileBackupStore(case.workspace.workspace_state, case.claim.session_id)
                .inspect()
                .entries
            )
            == 1
        )
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        records = (await recorder.list_records(limit=200)).items
        for i, foreground in enumerate(cases):
            tools = [r for r in records if r.kind is TraceKind.TOOL and r.run_id == f"run-{i}"]
            business_tools = [
                record for record in tools
                if record.tool_name in {"write_file", "read_file", "exec"}
            ]
            assert {r.tool_name for r in business_tools} == {"write_file", "read_file", "exec"}
            assert len(business_tools) == 3
            assert {r.tool_name for r in tools} == {
                "dispatch_worker", "finish", "write_file", "read_file", "exec"
            }
            for record in business_tools:
                assert record.owner_session_id == foreground.claim.session_id
                assert record.workspace_key == workspace_key_for(
                    foreground.workspace.workspace_path
                )
                parent = next(r for r in records if r.record_id == record.parent_record_id)
                assert parent.kind is TraceKind.MODEL
                assert (parent.root_id, parent.owner_session_id) == (
                    record.root_id,
                    record.owner_session_id,
                )
                detail = await recorder.get_record(record.record_id)
                assert detail is not None and detail.request is not None
                assert detail.result is not None
        scheduled = [
            r for r in records if r.kind is TraceKind.TOOL and r.owner_session_id == job.session_id
        ]
        assert {r.tool_name for r in scheduled} - {"dispatch_worker", "finish"} == {
            "write_file", "read_file", "exec"
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_cancels_only_the_calling_run_on_a_shared_exec_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BlockingHost(_RecordingExecHost):
        def __init__(self) -> None:
            super().__init__()
            self.started = {name: asyncio.Event() for name in ("workspace-0", "workspace-1")}
            self.cancelled = {name: asyncio.Event() for name in self.started}
            self.release = asyncio.Event()

        async def execute(self, command: str, cwd: Path, timeout: int) -> Any:
            self.started[cwd.name].set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled[cwd.name].set()
                raise
            return await super().execute(command, cwd, timeout)

    home = _home(tmp_path / "home")
    config = (home.path / "config.toml").read_text(encoding="utf-8")
    (home.path / "config.toml").write_text(
        config.replace(
            "compact_ratio = 0.9", 'compact_ratio = 0.9\npermission_level = "full-access"'
        ),
        encoding="utf-8",
    )
    provider = _ToolProvider()
    host = BlockingHost()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    monkeypatch.setattr(service_runtime, "create_exec_host", lambda _shell: host)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        cases = [await _session_case(service, tmp_path / f"workspace-{i}") for i in range(2)]
        await asyncio.gather(
            *(case.submit(f"tool-{i}", f"run-{i}") for i, case in enumerate(cases))
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in host.started.values())), timeout=5
            )
        except TimeoutError:
            pytest.fail("The foreground Workers did not both reach the shared Exec Host within 5 seconds.")
        case = cases[0]
        work_store = case.workspace.foreground_work_store(case.claim.session_id)
        work_snapshot = work_store.load()
        task = next(item for item in work_snapshot.tasks if item.state == "waiting_for_workers")
        await case.workspace.cancel_task(
            case.client_id,
            case.claim.session_id,
            case.claim.version,
            task.task_id,
            work_snapshot.revision,
            "cancel-run-0",
        )
        await asyncio.wait_for(host.cancelled["workspace-0"].wait(), timeout=5)
        assert next(
            item for item in work_store.load().tasks if item.task_id == task.task_id
        ).state == "cancelled"
        assert not host.cancelled["workspace-1"].is_set()
        host.release.set()
        await cases[1].completed("run-1")
        assert not host.cancelled["workspace-1"].is_set()
        assert host.executed == [cases[1].workspace.workspace_path]
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        records = (await recorder.list_records(limit=200)).items
        cancelled = next(
            r
            for r in records
            if r.kind is TraceKind.TOOL and r.run_id == "run-0" and r.tool_name == "exec"
        )
        succeeded = next(
            r
            for r in records
            if r.kind is TraceKind.TOOL and r.run_id == "run-1" and r.tool_name == "exec"
        )
        assert cancelled.error_category is TraceErrorCategory.CANCELLED
        assert cancelled.status.value == "failed"
        assert succeeded.status.value == "success"
        assert cancelled.workspace_key != succeeded.workspace_key
    finally:
        host.release.set()
        await service.stop()
