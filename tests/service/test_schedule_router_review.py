from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

import aide.service.runtime.service as service_runtime
from aide.agent.confirmation import ConfirmationUnavailable
from aide.agent.session.session import Session
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.agent.workers.store import WorkerStore
from aide.config.config import ConfigLoader
from aide.provider.errors import ModelCallError
from aide.provider.models import (
    AssistantModelMessage,
    ModelMessages,
    ModelResponse,
    ModelUsage,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.schedule.service import ScheduleJobExecutionError, ScheduleOccurrence
from aide.schedule.store import WorkspaceScheduleStore
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService
from aide.utils.errors import ErrorInfo
from tests.service.test_router_worker_coordination import (
    _home,
    _ScheduledRouterWorkerProvider,
    _SilentSink,
    _wait_until,
    _WorkerSubAgentProvider,
)


class _FailureProvider(_ScheduledRouterWorkerProvider):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.worker_calls = 0

    def _response_for(
        self,
        messages: ModelMessages,
        tools: Any,
        model: str,
    ) -> ModelResponse:
        worker = str(messages[0].get("content", "")).startswith("You are an Aide Worker.")
        if worker:
            self.worker_calls += 1
        if self.mode == "ask" and not worker:
            return ModelResponse(
                message=AssistantModelMessage(
                    content="",
                    tool_calls=(ModelToolCall("ask", "ask_user", '{"question":"Which source?"}'),),
                ),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            )
        if (
            self.mode == "confirmation"
            and worker
            and not any(item.get("role") == "tool" for item in messages)
        ):
            return ModelResponse(
                message=AssistantModelMessage(
                    content="",
                    tool_calls=(
                        ModelToolCall(
                            "write", "write_file", '{"path":"output.txt","content":"no"}'
                        ),
                    ),
                ),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            )
        if (worker and self.mode == "worker_error") or (not worker and self.mode == "router_error"):
            raise ModelCallError(ErrorInfo("model_failed", "Injected model failure."))
        return super()._response_for(messages, tools, model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "worker_error",
        "router_error",
        "report_write",
        "receipt_write",
        "ask",
        "confirmation",
        "terminal_write",
        "receipt_index",
    ],
)
async def test_schedule_public_history_terminal_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    home = _home(tmp_path / "home")
    if mode == "confirmation":
        path = home.path / "config.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[runtime]", '[runtime]\npermission_level = "read-only"'
            ),
            encoding="utf-8",
        )
    with (home.path / "config.toml").open("a", encoding="utf-8") as config:
        config.write('\n[models.routes.schedule]\nprovider_id = "primary"\nmodel = "small-model"\n')
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _FailureProvider(mode)
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        # No foreground Conversation Session or claim to receive output.
        workspace = await service.attach_workspace(client.client_id, directory)
        if mode == "confirmation":

            async def unavailable(*args: Any, **kwargs: Any) -> Any:
                raise ConfirmationUnavailable("No confirmation audience")

            monkeypatch.setattr(service.confirmation, "request", unavailable)
        occurrences: list[ScheduleOccurrence] = []
        original = workspace._run_schedule_occurrence

        async def capture(occurrence: ScheduleOccurrence) -> None:
            occurrences.append(occurrence)
            await original(occurrence)

        monkeypatch.setattr(workspace, "_run_schedule_occurrence", capture)
        if mode == "report_write":
            monkeypatch.setattr(
                WorkerStore,
                "publish_report",
                lambda *a, **k: (_ for _ in ()).throw(OSError("report write")),
            )
        if mode == "receipt_write":
            monkeypatch.setattr(
                WorkerStore,
                "claim_report_batch",
                lambda *a, **k: (_ for _ in ()).throw(OSError("receipt write")),
            )
        if mode == "receipt_index":
            reconcile = WorkerStore.reconcile_deliveries

            def fail_index(store: WorkerStore, messages: Any) -> Any:
                if any(message.get("worker_report_receipts") for message in messages):
                    raise OSError("receipt index write")
                return reconcile(store, messages)

            monkeypatch.setattr(WorkerStore, "reconcile_deliveries", fail_index)
        commit_terminal = WorkspaceScheduleStore.commit_terminal
        if mode == "terminal_write":

            async def fail_terminal(*args: Any, **kwargs: Any) -> Any:
                raise OSError("terminal state write")

            monkeypatch.setattr(WorkspaceScheduleStore, "commit_terminal", fail_terminal)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="Inspect the scheduled source with a Worker.",
            schedule=JobSchedule.every(3600),
            created_at_ms=now_ms - 3_600_000,
            updated_at_ms=now_ms - 3_600_000,
            title="Review check",
        )
        await workspace.schedule_service.add_user_job(job)

        if mode == "terminal_write":
            await _wait_until(
                lambda: workspace.schedule_service.status_snapshot().status == "faulted"
            )
            with pytest.raises(ServiceError, match="cleanup error"):
                await service.stop()
            monkeypatch.setattr(WorkspaceScheduleStore, "commit_terminal", commit_terminal)
            service = AgentService(home, ConfigLoader(home).load())
            await service.start()
            client = await service.register_client("cli")
            await service.connect_client(client.client_id, _SilentSink())
            workspace = await service.attach_workspace(client.client_id, directory)

        async with asyncio.timeout(10):
            while True:
                jobs = await workspace.schedule_service.public_snapshot()
                if jobs[0].state.last_status is not None:
                    break
                await asyncio.sleep(0.01)
        success = mode in {"success", "terminal_write", "receipt_index"}
        assert jobs[0].state.last_status == ("ok" if success else "error"), jobs[0].state
        history = await service.get_schedule_job_history(
            client.client_id, workspace.workspace_id, job.job_id
        )
        groups = cast(list[dict[str, Any]], history["groups"])
        assert groups[-1]["result_state"] == ("success" if success else "failure"), (
            history
        )
        if mode == "terminal_write":
            assert provider.worker_calls == 1
        if mode in {"worker_error", "confirmation"}:
            assert all(
                worker.report_delivery_state == "consumed"
                for worker in workspace.worker_store(Session.schedule_session_id(job.job_id)).load().workers
            )
        if mode in {"success", "worker_error", "router_error", "ask"}:
            before = provider.worker_calls
            session_id = Session.schedule_session_id(job.job_id)
            history_before = Session.load(workspace.workspace_state, session_id).messages
            for _ in range(10):
                if mode == "success":
                    await original(occurrences[0])
                else:
                    with pytest.raises(ScheduleJobExecutionError):
                        await original(occurrences[0])
            await _wait_until(lambda: not workspace._has_active_workers(session_id))
            assert provider.worker_calls == before
            assert Session.load(workspace.workspace_state, session_id).messages == history_before
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "shutdown"])
async def test_schedule_deletion_drains_worker_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    home = _home(tmp_path / "home")
    with (home.path / "config.toml").open("a", encoding="utf-8") as config:
        config.write('\n[models.routes.schedule]\nprovider_id = "primary"\nmodel = "small-model"\n')
    directory = tmp_path / "workspace"
    directory.mkdir()
    (directory / "worker-input.txt").write_text("retained input", encoding="utf-8")
    from aide.provider.models import ModelCompleted

    class ScheduledChildrenProvider(_WorkerSubAgentProvider):
        async def complete(self, **kwargs: Any) -> ModelResponse:
            async for event in self.stream(**kwargs):
                if isinstance(event, ModelCompleted):
                    return event.response
            raise AssertionError("Provider did not complete")

    provider = ScheduledChildrenProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="Run both independent checks.",
            schedule=JobSchedule.every(3600),
            created_at_ms=now_ms - 3_600_000,
            updated_at_ms=now_ms - 3_600_000,
            title="Review children",
        )
        await workspace.schedule_service.add_user_job(job)
        await asyncio.wait_for(provider.two_workers_started.wait(), 10)
        provider.release_workers.set()
        try:
            await asyncio.wait_for(provider.two_subagents_started.wait(), 10)
        except TimeoutError:
            pytest.fail(
                str(
                    (
                        provider.calls,
                        workspace.worker_store(Session.schedule_session_id(job.job_id))
                        .load()
                        .workers,
                    )
                )
            )
        if operation == "delete":
            await workspace.schedule_service.remove_user_job(job.job_id)
        else:
            await service.stop()
        sid = Session.schedule_session_id(job.job_id)
        assert not workspace._has_active_workers(sid)
        records = workspace.subagent_repository(sid).list().items
        assert len(records) == 2
        assert all(item.status.value in {"cancelled", "interrupted"} for item in records)
        provider.release_subagent.set()
        await asyncio.sleep(0.05)
        assert provider.report_continuations == 0
        assert not (directory / "subagent-first.txt").exists()
        assert not (directory / "subagent-second.txt").exists()
    finally:
        provider.release_workers.set()
        provider.release_subagent.set()
        provider.release_report_continuation.set()
        await service.stop()
