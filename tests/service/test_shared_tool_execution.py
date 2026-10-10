from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
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
from tests.service.test_shared_resources import _CountingProvider, _home, _session_case
from tests.tools.test_shared_tool_context import _RecordingExecHost


class _ToolProvider(_CountingProvider):
    def response(self, messages: list[dict[str, Any]]) -> ModelResponse:
        user_index = max(i for i, message in enumerate(messages) if message["role"] == "user")
        label = str(messages[user_index]["content"]).splitlines()[-1]
        if label == "tool-schedule":
            self.schedule_started.set()
        if any(message["role"] == "tool" for message in messages[user_index + 1 :]):
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
            yield ModelCompleted(self.response(kwargs["messages"]))

        return emit()

    async def complete(self, **kwargs: Any) -> ModelResponse:
        if not kwargs["tools"]:
            return await super().complete(**kwargs)
        return self.response(kwargs["messages"])


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
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        cases = [await _session_case(service, tmp_path / f"workspace-{i}") for i in range(2)]
        await _session_case(service, tmp_path / "workspace-0")
        for i, case in enumerate(cases):
            (case.workspace.workspace_path / "same.txt").write_text(f"before-{i}", encoding="utf-8")
        await asyncio.gather(
            *(case.submit(f"tool-{i}", f"run-{i}") for i, case in enumerate(cases))
        )
        await asyncio.gather(*(case.completed(f"run-{i}") for i, case in enumerate(cases)))
        assert len(catalog_creations) == 1
        for i, case in enumerate(cases):
            path = case.workspace.workspace_path / "same.txt"
            assert path.read_text(encoding="utf-8") == f"tool-{i}"
            reads = [
                message
                for message in case.claim.loop.session.messages
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
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        cases = [await _session_case(service, tmp_path / f"workspace-{i}") for i in range(2)]
        await asyncio.gather(
            *(case.submit(f"tool-{i}", f"run-{i}") for i, case in enumerate(cases))
        )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in host.started.values())), timeout=5
        )
        case = cases[0]
        await case.workspace.cancel(
            case.client_id, case.claim.session_id, case.claim.version, "run-0"
        )
        await asyncio.wait_for(host.cancelled["workspace-0"].wait(), timeout=5)
        assert not host.cancelled["workspace-1"].is_set()
        host.release.set()
        await cases[1].completed("run-1")
        assert not host.cancelled["workspace-1"].is_set()
        assert host.executed == [cases[1].workspace.workspace_path]
    finally:
        host.release.set()
        await service.stop()
