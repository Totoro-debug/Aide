from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from aide.agent.confirmation import ConfirmationEnvelope, SubAgentConfirmationOwner
from aide.agent.session.session import Session
from aide.agent.subagents.coordinator import SubAgentPool
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentEvent,
    SubAgentExecutionResult,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.ports import SubAgentRecordRepository
from aide.agent.subagents.store import (
    SubAgentRecordStore,
    SubAgentRequestError,
    SubAgentStoreError,
)
from aide.agent.tools.tool_gateway import ConfirmationRequest, ModelToolCall, ToolGateway
from aide.agent.workspace_state import WorkspaceState
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.discovery import create_credential, read_credential
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService, WorkspaceRecord
from aide.service.transport import create_app
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures.project_removal import complete_project_removal
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


def _snapshot() -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="test-provider",
        model="test-model",
        reasoning_effort="mid",
        permission_level="workspace-write",
        shell="pwsh",
        tool_schemas=({"name": "read_file", "input_schema": {"type": "object"}},),
        system_prompt="You are Aide.",
    )


def _register(repository: SubAgentRecordStore, title: str) -> SubAgentRecord:
    return repository.register(
        title=title,
        task=f"Complete {title}.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )


def _complete(repository: SubAgentRecordStore, record: SubAgentRecord) -> SubAgentRecord:
    running = repository.save(
        replace(
            record,
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=record.revision + 1,
        )
    )
    return repository.save(
        replace(
            running,
            status=SubAgentStatus.COMPLETED,
            finished_at=_NOW,
            conversation=({"role": "assistant", "content": "Full child output."},),
            result="Completed answer.",
            revision=running.revision + 1,
        )
    )


def _require_record(repository: SubAgentRecordRepository, agent_id: str) -> SubAgentRecord:
    record = repository.get(agent_id)
    assert record is not None
    return record


def _schedule_child(pool: SubAgentPool, job_id: str) -> SubAgentRecord:
    return pool.submit(
        title="Schedule child",
        task="Wait for cleanup.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.SCHEDULE,
            job_id=job_id,
            occurrence_id=str(uuid4()),
        ),
        creator_snapshot=_snapshot(),
    )


async def _remove_schedule_job(
    service: AgentService,
    workspace: WorkspaceRecord,
    client_id: str,
    job_id: str,
    entry: str,
    request_id: str,
) -> bool:
    if entry == "tool":
        gateway = ToolGateway(
            workspace=workspace.workspace_path, schedule_service=workspace.schedule_service
        )
        result = await gateway.call(
            ModelToolCall(
                id=request_id,
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": job_id}),
            )
        )
        return result.status == "success"
    try:
        response = await service.delete_schedule_job(
            client_id,
            workspace.workspace_id,
            job_id,
            request_id,
            {},
        )
    except ServiceError as error:
        assert error.code in {"schedule_update_failed", "schedule_changed", "not_found"}
        return False
    return response["deleted"] is True


class _HoldingExecutor:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release: dict[str, asyncio.Event] = {}
        self.cancel_requested: dict[str, asyncio.Event] = {}
        self.interrupted: dict[str, bool] = {}

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult:
        del emit
        self.release[record.agent_id] = asyncio.Event()
        self.cancel_requested[record.agent_id] = asyncio.Event()
        self.interrupted[record.agent_id] = False
        await self.started.put(record.agent_id)
        await self.release[record.agent_id].wait()
        cancelled = self.cancel_requested[record.agent_id].is_set()
        return SubAgentExecutionResult(
            status=SubAgentStatus.CANCELLED if cancelled else SubAgentStatus.COMPLETED,
            conversation=({"role": "assistant", "content": "Finished."},),
            context_state={},
            artifact_paths=(),
            result="Finished.",
            error=(
                SubAgentError(code="cancelled", message="Cancelled by user.") if cancelled else None
            ),
            usage={"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        requested = self.cancel_requested.get(agent_id)
        if requested is None:
            return False
        requested.set()
        self.interrupted[agent_id] = interrupted
        if interrupted:
            self.release[agent_id].set()
        return True


class _DelayedExecutor(_HoldingExecutor):
    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        requested = self.cancel_requested.get(agent_id)
        if requested is None:
            return False
        requested.set()
        self.interrupted[agent_id] = interrupted
        return True


class _DelegatingProvider:
    def __init__(self) -> None:
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.old_result_seen = asyncio.Event()
        self.agent_id: str | None = None
        self.parent_tool_names: list[set[str]] = []
        self.child_tool_names: set[str] = set()
        self.child_tool_results: list[dict[str, Any]] = []
        self.write_path: str | None = None
        self.allow_write = asyncio.Event()
        self.write_seen = asyncio.Event()
        self.spawn_requested = False
        self.wait_requested = False
        self.wait_result_contents: list[str] = []
        self.direct_schedule_tool_names: list[set[str]] = []
        self.requests: list[tuple[str, set[str]]] = []

    @staticmethod
    def _response(content: str, *, tool_call: ModelToolCall | None = None) -> ModelResponse:
        calls = () if tool_call is None else (tool_call,)
        return ModelResponse(
            message=AssistantModelMessage(content=content, tool_calls=calls),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="tool_calls" if calls else "stop",
        )

    @classmethod
    def _events(
        cls, content: str, *, tool_call: ModelToolCall | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(cls._response(content, tool_call=tool_call))

        return emit()

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
            if isinstance(tool.get("function"), dict)
            and isinstance((name := tool["function"].get("name")), str)
        }
        latest_user_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        user_text = "\n".join(str(message.get("content", "")) for message in messages)
        self.requests.append((user_text, tool_names))
        current_tool_results = [
            message
            for message in messages[latest_user_index + 1 :]
            if message.get("role") == "tool"
        ]

        if "Direct schedule integration request" in user_text:
            self.direct_schedule_tool_names.append(tool_names)
            return self._events("Direct schedule completed.")

        if (
            "initial delegation request" in user_text
            and "follow-up result request" not in user_text
        ):
            self.parent_tool_names.append(tool_names)
            if self.spawn_requested:
                return self._events("Main run completed.")
            if "spawn_agent" not in tool_names:
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        "search-spawn",
                        "tool_search",
                        '{"query":"spawn agent"}',
                    ),
                )
            self.spawn_requested = True
            return self._events(
                "",
                tool_call=ModelToolCall(
                    "spawn-child",
                    "spawn_agent",
                    '{"title":"Delegated answer","task":"Make delegated integration answer."}',
                ),
            )

        if "follow-up result request" in user_text:
            self.parent_tool_names.append(tool_names)
            if self.wait_requested:
                self.wait_result_contents = [
                    str(message.get("content", "")) for message in current_tool_results
                ]
                if any("SubAgent child answer" in content for content in self.wait_result_contents):
                    self.old_result_seen.set()
                return self._events("Old result retrieved.")
            if "wait_agent" not in tool_names:
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        "search-wait",
                        "tool_search",
                        '{"query":"wait agent"}',
                    ),
                )
            self.wait_requested = True
            agent_id = self.agent_id or ""
            return self._events(
                "",
                tool_call=ModelToolCall(
                    "wait-child",
                    "wait_agent",
                    json.dumps({"agent_ids": [agent_id]}),
                ),
            )

        if "spawn_agent" not in tool_names and "list_agents" not in tool_names:
            return self._events(
                "",
                tool_call=ModelToolCall(
                    "search-list",
                    "tool_search",
                    '{"query":"list agents"}',
                ),
            )
        if "spawn_agent" not in tool_names:
            self.child_tool_names = tool_names
            self.child_tool_results = current_tool_results
            for name, call_id in (
                ("list_agents", "child-list"),
                ("spawn_agent", "child-spawn-denied"),
                ("wait_agent", "child-wait-denied"),
                ("schedule", "child-schedule-denied"),
            ):
                if not any(
                    result.get("tool_call_id") == call_id for result in current_tool_results
                ):
                    return self._events("", tool_call=ModelToolCall(call_id, name, "{}"))
            if self.write_path is not None:
                if "write_file" not in tool_names:
                    return self._events(
                        "",
                        tool_call=ModelToolCall(
                            "child-search-write", "tool_search", '{"query":"write file"}'
                        ),
                    )
                if not any(
                    result.get("tool_call_id") == "child-write" for result in current_tool_results
                ):

                    async def write_child() -> AsyncIterator[ModelStreamEvent]:
                        self.child_started.set()
                        await self.allow_write.wait()
                        yield ModelCompleted(
                            self._response(
                                "",
                                tool_call=ModelToolCall(
                                    "child-write",
                                    "write_file",
                                    json.dumps({"path": self.write_path, "content": "child edit"}),
                                ),
                            )
                        )

                    return write_child()
                self.write_seen.set()

            async def complete_child() -> AsyncIterator[ModelStreamEvent]:
                self.child_started.set()
                await self.release_child.wait()
                yield TextDelta("SubAgent ")
                yield TextDelta("child ")
                yield TextDelta("answer.")
                yield ModelCompleted(self._response("SubAgent child answer."))

            return complete_child()

        return self._events("Unexpected model request.")

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
        tool_names = {
            name
            for tool in tools
            if isinstance(tool.get("function"), dict)
            and isinstance((name := tool["function"].get("name")), str)
        }
        user_text = "\n".join(str(message.get("content", "")) for message in messages)
        if "Direct schedule integration request" in user_text:
            self.direct_schedule_tool_names.append(tool_names)
            return self._response("Direct schedule completed.")
        if (
            "initial delegation request" in user_text
            and "follow-up result request" not in user_text
            and "tool_search" in tool_names
        ):
            self.parent_tool_names.append(tool_names)
            if self.spawn_requested:
                return self._response("Scheduled Main Run completed.")
            if "spawn_agent" not in tool_names:
                return self._response(
                    "",
                    tool_call=ModelToolCall(
                        "search-spawn",
                        "tool_search",
                        '{"query":"spawn agent"}',
                    ),
                )
            self.spawn_requested = True
            return self._response(
                "",
                tool_call=ModelToolCall(
                    "spawn-child",
                    "spawn_agent",
                    '{"title":"Scheduled answer","task":"Make delegated integration answer."}',
                ),
            )
        return self._response(
            '{"action":"replace","task_goal":"answer the input",'
            '"completion_boundary":"return one answer"}'
        )

    async def close(self) -> None:
        return None


@pytest_asyncio.fixture
async def management_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AgentService, WorkspaceRecord, str, str, str]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Management",
        created_at=_NOW,
        content="Saved Session history.",
    )
    other_session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Other",
        created_at=_NOW,
        content="Other Session history.",
    )
    monkeypatch.setattr(
        "aide.service.runtime.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        yield service, workspace, client.client_id, session_id, other_session_id
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_subagent_queries_survive_claim_release_and_remain_session_scoped(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    pool = SubAgentPool(repository, _HoldingExecutor())
    workspace.register_subagent_coordinator(pool, repository)
    record = _complete(repository, _register(repository, "Stored result"))

    claim = await workspace.claim(client_id, session_id)
    await workspace.release(client_id, session_id)

    page = service.list_subagents(client_id, workspace.workspace_id, session_id)
    detail = service.get_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    page_items = cast(list[dict[str, object]], page["items"])
    assert page_items[0]["agent_id"] == record.agent_id
    assert page_items[0]["result_preview"] == "Completed answer."
    assert detail["task"] == "Complete Stored result."
    assert detail["conversation"] == [{"role": "assistant", "content": "Full child output."}]
    assert detail["usage"] == record.usage
    assert claim.session_id == session_id

    for candidate_session in (other_session_id, session_id):
        with pytest.raises(ServiceError) as error:
            service.get_subagent(
                client_id,
                workspace.workspace_id,
                candidate_session,
                "923e4567-e89b-42d3-a456-426614174000",
            )
        assert error.value.code == "not_found"


@pytest.mark.asyncio
async def test_foreground_main_run_delegates_and_later_reads_the_completed_result(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    assert service._model_router is not None
    assert workspace.resources.router is service._model_router
    service._model_router._provider_factory = lambda _configuration: provider
    service._model_router._providers.clear()
    claim_payload = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim_payload["claim_version"])
    coordinator = workspace._subagent_coordinator(session_id)
    assert coordinator is not None
    sink = cast(_CollectingSink, service.client(client_id).sink)

    try:
        await workspace.input(
            client_id,
            session_id,
            claim_version,
            "Please make an initial delegation request.",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.child_started.wait(), timeout=5)
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)

        assert any("spawn_agent" in names for names in provider.parent_tool_names)
        assert "tool_search" in provider.parent_tool_names[0]
        assert "spawn_agent" not in provider.child_tool_names
        assert "wait_agent" not in provider.child_tool_names
        assert "schedule" not in provider.child_tool_names
        assert "list_agents" in provider.child_tool_names
        child_results = {item["tool_call_id"]: item for item in provider.child_tool_results}
        assert child_results["child-list"]["status"] == "success"
        assert len(json.loads(child_results["child-list"]["content"])["items"]) == 1
        assert all(
            child_results[call_id]["status"] == "error"
            for call_id in ("child-spawn-denied", "child-wait-denied", "child-schedule-denied")
        )
        page = service.list_subagents(client_id, workspace.workspace_id, session_id)
        items = cast(list[dict[str, object]], page["items"])
        assert len(items) == 1
        agent_id = cast(str, items[0]["agent_id"])
        assert items[0]["status"] == "running"

        provider.release_child.set()
        results = await asyncio.wait_for(
            coordinator.wait([agent_id], timeout_ms=3000),
            timeout=4,
        )
        assert results[0].result == "SubAgent child answer."
        terminal = workspace.subagent_repository(session_id).get(agent_id)
        assert terminal is not None
        child_events = [
            event
            for event in sink.events
            if cast(str, event["type"]).startswith("subagent.")
            and cast(dict[str, object], event["payload"])["agent_id"] == agent_id
        ]
        assert child_events
        assert all(
            cast(int, cast(dict[str, object], event["payload"])["revision"]) <= terminal.revision
            for event in child_events
        )

        provider.agent_id = agent_id
        follow_up_run_id = str(uuid4())
        await workspace.input(
            client_id,
            session_id,
            claim_version,
            "Please make a follow-up result request.",
            follow_up_run_id,
        )
        await asyncio.wait_for(sink.wait_for("run.completed", follow_up_run_id), timeout=5)
        assert provider.old_result_seen.is_set(), provider.wait_result_contents
    finally:
        provider.release_child.set()


@pytest.mark.asyncio
async def test_schedule_occurrence_delegates_and_direct_schedule_hides_subagent_tools(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, _client_id, _session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    assert service._model_router is not None
    service._model_router._provider_factory = lambda _configuration: provider
    service._model_router._providers.clear()
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Please make an initial delegation request.",
        title="Scheduled delegation",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    workspace.schedule_service._reserve(job, current_monotonic=asyncio.get_running_loop().time())

    try:
        try:
            await asyncio.wait_for(provider.child_started.wait(), timeout=5)
        except TimeoutError:
            pytest.fail(
                f"Schedule SubAgent did not start; requests={provider.requests!r}; "
                f"jobs={await workspace.schedule_service.public_snapshot()!r}"
            )
        for _ in range(500):
            if job.job_id not in workspace.schedule_service._active_runs:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Schedule Job did not finish while its SubAgent remained active.")

        schedule_loop = workspace._schedule_loops[job.job_id].loop
        assert any(
            message.get("content") == "Scheduled Main Run completed."
            for message in schedule_loop.session.messages
        )
        repository = workspace.subagent_repository(job.session_id)
        items = repository.list().items
        assert len(items) == 1
        record = repository.get(items[0].agent_id)
        assert record is not None
        assert record.status is SubAgentStatus.RUNNING
        assert record.source.kind is SubAgentSourceKind.SCHEDULE
        assert record.source.job_id == job.job_id
        assert record.source.occurrence_id is not None
        permission_snapshot = workspace._capture_schedule_permission_snapshot()
        assert workspace._exec_host is not None
        assert record.creator_snapshot.permission_level == permission_snapshot.level
        assert record.creator_snapshot.shell == workspace._exec_host.resolved_shell.selector
        assert all("schedule" not in names for names in provider.parent_tool_names)

        provider.release_child.set()
        coordinator = workspace._subagent_coordinator(job.session_id)
        assert coordinator is not None
        result = await asyncio.wait_for(
            coordinator.wait([record.agent_id], timeout_ms=3000),
            timeout=4,
        )
        assert result[0].result == "SubAgent child answer."

        provider.spawn_requested = False
        workspace.schedule_service._reserve(
            job, current_monotonic=asyncio.get_running_loop().time()
        )
        await asyncio.wait_for(workspace.schedule_service._active_runs[job.job_id].task, timeout=5)
        assert workspace._subagent_coordinator(job.session_id) is coordinator
        records = [repository.get(item.agent_id) for item in repository.list().items]
        assert len(records) == 2
        assert len({record.source.occurrence_id for record in records if record is not None}) == 2
        for scheduled_record in records:
            assert scheduled_record is not None
            await asyncio.wait_for(coordinator.wait([scheduled_record.agent_id]), timeout=4)
            with pytest.raises(ServiceError) as cross_session:
                service.get_subagent(
                    _client_id, workspace.workspace_id, _session_id, scheduled_record.agent_id
                )
            assert cross_session.value.code == "not_found"

        direct_job = ScheduleJob(
            job_id=str(uuid4()),
            message="Direct schedule integration request.",
            title="Direct Schedule",
            schedule=JobSchedule.at("2000-01-01T00:00:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
        direct_loop = await workspace._get_schedule_loop(
            direct_job.job_id,
            title=direct_job.title,
        )
        await direct_loop.loop.run_schedule_job(direct_job)
        assert provider.direct_schedule_tool_names
        assert not (
            {"spawn_agent", "wait_agent", "list_agents"} & provider.direct_schedule_tool_names[0]
        )
    finally:
        provider.release_child.set()


@pytest.mark.asyncio
async def test_real_child_confirmation_cancel_and_file_restore_after_claim_release(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    provider = _DelegatingProvider()
    provider.write_path = "delegated.txt"
    target = workspace.workspace_path / provider.write_path
    target.write_text("before child", encoding="utf-8")
    service.client_permission(client_id).select("read-only")
    assert service._model_router is not None
    service._model_router._provider_factory = lambda _configuration: provider
    service._model_router._providers.clear()
    mcp_manager = workspace._mcp_manager
    await service.claim(client_id, workspace.workspace_id, session_id)
    claim = workspace._claims[session_id]
    repository = workspace.subagent_repository(session_id)
    assert isinstance(repository, SubAgentRecordStore)
    earlier = _complete(repository, _register(repository, "Keep earlier result"))
    sink = cast(_CollectingSink, service.client(client_id).sink)
    try:
        await workspace.input(
            client_id, session_id, claim.version, "initial delegation request", _RUN_ID
        )
        await asyncio.wait_for(provider.child_started.wait(), timeout=5)
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)
        child = next(item for item in repository.list().items if item.agent_id != earlier.agent_id)
        record = _require_record(repository, child.agent_id)
        assert record.creator_snapshot.permission_level == "read-only"
        anchor = next(
            anchor
            for anchor in claim.loop.session.restore_candidates()
            if str(anchor.run_token) == record.source.restore_run_token
        )
        await workspace.release(client_id, session_id)
        # A later permission choice must not change the already registered child's rules.
        service.client_permission(client_id).select("full-access")
        provider.allow_write.set()

        async def wait_for_confirmation() -> dict[str, object]:
            while True:
                for event in sink.events:
                    if event["type"] == "confirmation.requested":
                        return cast(dict[str, object], event["payload"])
                await asyncio.sleep(0.01)

        confirmation = await asyncio.wait_for(wait_for_confirmation(), timeout=5)
        assert cast(dict[str, object], confirmation["owner"])["agent_id"] == child.agent_id
        service.decide_confirmation(client_id, cast(str, confirmation["token"]), "approved")
        await asyncio.wait_for(provider.write_seen.wait(), timeout=5)
        assert target.read_text(encoding="utf-8") == "child edit"
        assert workspace._mcp_manager is mcp_manager
        assert len(service._model_router._providers) == 1
        claim = await workspace.claim(client_id, session_id)
        busy = await service.inspect_restore(
            client_id,
            workspace.workspace_id,
            session_id,
            claim.version,
            claim.credential,
            "restore-live-child",
            anchor.anchor_id,
        )
        assert str(busy["output"]).startswith("model_invalid_request:")
        cancelled = await service.cancel_subagent(
            client_id, workspace.workspace_id, session_id, child.agent_id
        )
        assert cast(dict[str, object], cancelled["agent"])["status"] == "cancelled"
        inspected = await service.inspect_restore(
            client_id,
            workspace.workspace_id,
            session_id,
            claim.version,
            claim.credential,
            "inspect-child-file",
            anchor.anchor_id,
        )
        assert inspected.get("restore_plan") is not None
        restored = await service.commit_restore(
            client_id,
            workspace.workspace_id,
            session_id,
            claim.version,
            claim.credential,
            "restore-child-file",
            anchor.anchor_id,
            "files",
        )
        assert restored.get("restore_result") is not None
        assert target.read_text(encoding="utf-8") == "before child"
        assert repository.get(child.agent_id) is None
        assert repository.get(earlier.agent_id) == earlier
        assert Session.load(workspace.workspace_state, other_session_id).messages
    finally:
        provider.allow_write.set()
        provider.release_child.set()


@pytest.mark.asyncio
async def test_cancel_waits_for_child_cleanup_and_is_idempotent(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Cancelable",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id

    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not cancellation.done()
    executor.release[record.agent_id].set()
    first = await cancellation
    second = await service.cancel_subagent(
        client_id, workspace.workspace_id, session_id, record.agent_id
    )
    assert first["cancelled"] is True
    assert second["cancelled"] is True
    assert cast(dict[str, object], second["agent"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_schedule_session_subagents_support_user_queries_and_cancellation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, foreground_session_id, _other_session_id = management_case
    job_id = str(uuid4())
    session = Session.create_schedule(workspace.workspace_state, job_id)
    session.commit_agent_run(
        [{"role": "user", "content": "Schedule input"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
    )
    await session.wait_for_pending_persist()
    repository = SubAgentRecordStore(workspace.workspace_state, session.session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Schedule child",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.SCHEDULE,
            job_id=job_id,
            occurrence_id=str(uuid4()),
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    page = service.list_subagents(client_id, workspace.workspace_id, session.session_id)
    assert cast(list[dict[str, object]], page["items"])[0]["agent_id"] == record.agent_id
    assert (
        service.get_subagent(
            client_id, workspace.workspace_id, session.session_id, record.agent_id
        )["source"]
        == "schedule"
    )
    with pytest.raises(ServiceError) as cross_session:
        await service.cancel_subagent(
            client_id, workspace.workspace_id, foreground_session_id, record.agent_id
        )
    assert cross_session.value.code == "not_found"
    cancellation = asyncio.create_task(
        service.cancel_subagent(
            client_id, workspace.workspace_id, session.session_id, record.agent_id
        )
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    assert (await cancellation)["cancelled"] is True


@pytest.mark.asyncio
async def test_cancel_reports_checkpoint_failure_and_can_retry_after_storage_repair(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    fail_cancelled = True

    def replace_text(path: Path, content: str) -> None:
        if fail_cancelled and json.loads(content)["status"] == "cancelled":
            raise OSError("cancel checkpoint unavailable")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    repository = SubAgentRecordStore(
        workspace.workspace_state, session_id, replace_text=replace_text
    )
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Failed cancellation checkpoint",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    try:
        with pytest.raises(ServiceError) as failure:
            await cancellation
        assert failure.value.code == "persistence_error"
        assert _require_record(repository, record.agent_id).status is SubAgentStatus.RUNNING
    finally:
        fail_cancelled = False
    repaired = await service.cancel_subagent(
        client_id, workspace.workspace_id, session_id, record.agent_id
    )
    assert repaired["cancelled"] is True


@pytest.mark.asyncio
async def test_session_delete_is_fenced_by_active_subagents_then_removes_them(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Blocks delete",
        task="Wait for cleanup.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    claim = await workspace.claim(client_id, session_id)

    with pytest.raises(ServiceError) as busy:
        await workspace.delete_session(
            client_id,
            session_id,
            claim.version,
            claim.credential,
            "delete-busy",
        )
    assert busy.value.code == "session_busy"

    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    await cancellation

    result = await workspace.delete_session(
        client_id,
        session_id,
        claim.version,
        claim.credential,
        "delete-after-cancel",
    )
    assert result["deleted"] is True
    assert repository.get(record.agent_id) is None


@pytest.mark.asyncio
async def test_schedule_job_deletion_drains_all_occurrences_before_returning(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    job_id = str(uuid4())
    job = ScheduleJob(
        job_id=job_id,
        message="Scheduled task",
        schedule=JobSchedule.every(3600),
        created_at_ms=int(_NOW.timestamp() * 1000),
        updated_at_ms=int(_NOW.timestamp() * 1000),
    )
    await workspace.schedule_service.add_user_job(job)
    records = tuple(
        pool.submit(
            title=f"Occurrence {index}",
            task="Finish the scheduled child task.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.SCHEDULE,
                job_id=job_id,
                occurrence_id=str(uuid4()),
            ),
            creator_snapshot=_snapshot(),
        )
        for index in range(2)
    )
    assert set(await asyncio.gather(executor.started.get(), executor.started.get())) == {
        record.agent_id for record in records
    }

    removal = asyncio.create_task(
        service.delete_schedule_job(client_id, workspace.workspace_id, job_id, "remove-job", {})
    )
    await asyncio.gather(
        *(
            asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
            for record in records
        )
    )
    assert not removal.done()
    for record in records:
        executor.release[record.agent_id].set()
    result = await removal

    assert result["deleted"] is True
    assert {_require_record(repository, record.agent_id).status for record in records} == {
        SubAgentStatus.CANCELLED
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
async def test_job_removal_drains_running_and_queued_children_through_both_entries(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Remove all occurrences",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    records = tuple(
        pool.submit(
            title=f"Child {index}",
            task="Wait for cancellation.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.SCHEDULE,
                job_id=job.job_id,
                occurrence_id=str(uuid4()),
            ),
            creator_snapshot=_snapshot(),
        )
        for index in range(9)
    )
    started = {await executor.started.get() for _ in range(8)}
    gateway = ToolGateway(
        workspace=workspace.workspace_path, schedule_service=workspace.schedule_service
    )
    removal = asyncio.create_task(
        gateway.call(
            ModelToolCall(
                id="remove-children",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": job.job_id}),
            )
        )
        if entry == "tool"
        else service.delete_schedule_job(
            client_id, workspace.workspace_id, job.job_id, "remove-children", {}
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(executor.cancel_requested[agent_id].wait() for agent_id in started)),
            timeout=2,
        )
        assert not removal.done()
        assert await workspace.schedule_service.public_snapshot() == ()
        assert _require_record(repository, records[-1].agent_id).status is SubAgentStatus.CANCELLED
    finally:
        pool.close_admission()
        for release in executor.release.values():
            release.set()
        await removal
    assert {_require_record(repository, record.agent_id).status for record in records} == {
        SubAgentStatus.CANCELLED
    }
    assert records[-1].agent_id not in executor.release
    assert executor.started.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
async def test_job_removal_fences_a_coordinator_registered_during_removal(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    job_id = str(uuid4())
    job = ScheduleJob(
        job_id=job_id,
        message="Scheduled task",
        schedule=JobSchedule.every(3600),
        created_at_ms=int(_NOW.timestamp() * 1000),
        updated_at_ms=int(_NOW.timestamp() * 1000),
    )
    await workspace.schedule_service.add_user_job(job)
    repository = SubAgentRecordStore(workspace.workspace_state, session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = _schedule_child(pool, job_id)
    assert await executor.started.get() == record.agent_id
    removal = asyncio.create_task(
        _remove_schedule_job(service, workspace, client_id, job_id, entry, "remove-late")
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    late_repository = SubAgentRecordStore(workspace.workspace_state, other_session_id)
    late_pool = SubAgentPool(late_repository, _HoldingExecutor())
    workspace.register_subagent_coordinator(late_pool, late_repository)
    try:
        for target in (pool, late_pool):
            with pytest.raises(SubAgentRequestError, match="Schedule Job"):
                _schedule_child(target, job_id)
    finally:
        executor.release[record.agent_id].set()
        assert await removal
    for target in (pool, late_pool):
        with pytest.raises(SubAgentRequestError, match="Schedule Job"):
            _schedule_child(target, job_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
@pytest.mark.parametrize("retry_entry", ["tool", "user"])
async def test_job_removal_waits_for_all_pools_and_retries_checkpoint_failure(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
    retry_entry: str,
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    fail_cancelled = True

    def replace_text(path: Path, content: str) -> None:
        if fail_cancelled and json.loads(content)["status"] == "cancelled":
            raise OSError("cancelled checkpoint unavailable")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Retry cleanup",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    executor = _HoldingExecutor()
    entries = []
    for target_session in (session_id, other_session_id):
        repository = SubAgentRecordStore(
            workspace.workspace_state,
            target_session,
            replace_text=(
                replace_text
                if target_session == session_id
                else HOST_FILESYSTEM.atomic_replace_text
            ),
        )
        pool = SubAgentPool(repository, executor)
        workspace.register_subagent_coordinator(pool, repository)
        record = _schedule_child(pool, job.job_id)
        assert await executor.started.get() == record.agent_id
        entries.append((repository, pool, record))
    removal = asyncio.create_task(
        _remove_schedule_job(service, workspace, client_id, job.job_id, entry, "failed-remove")
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(executor.cancel_requested[record.agent_id].wait() for _, _, record in entries)
            ),
            timeout=2,
        )
        first, second = entries
        executor.release[first[2].agent_id].set()
        with pytest.raises(SubAgentStoreError):
            await asyncio.wait_for(first[1].wait([first[2].agent_id]), timeout=2)
        assert not removal.done(), "An error in one pool must not skip the other pool's cleanup"
        executor.release[second[2].agent_id].set()
        assert not await removal
        assert await workspace.schedule_service.public_snapshot() == ()
        assert await workspace.schedule_service.job_for_removal(job.job_id) == job
        assert first[1].has_active()
        for _, pool, _ in entries:
            with pytest.raises(SubAgentRequestError, match="Schedule Job"):
                _schedule_child(pool, job.job_id)
    finally:
        fail_cancelled = False
        for release in executor.release.values():
            release.set()
    assert await _remove_schedule_job(
        service,
        workspace,
        client_id,
        job.job_id,
        retry_entry,
        "retry-remove",
    )
    for repository, pool, record in entries:
        assert _require_record(repository, record.agent_id).status is SubAgentStatus.CANCELLED
        assert len(repository.list().items) == 1
        assert not pool.has_active()
    assert await workspace.schedule_service.job_for_removal(job.job_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
async def test_job_removal_queued_checkpoint_failure_never_starts_the_cancelled_child(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    failed_id: str | None = None
    fail_cancelled = True

    def replace_text(path: Path, content: str) -> None:
        value = json.loads(content)
        if fail_cancelled and value["agent_id"] == failed_id and value["status"] == "cancelled":
            raise OSError("Queued cancellation checkpoint failed")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    repository = SubAgentRecordStore(
        workspace.workspace_state, session_id, replace_text=replace_text
    )
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Keep cancelled queue stopped",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    records = [_schedule_child(pool, job.job_id) for _ in range(9)]
    failed_id = records[-1].agent_id
    started = {await executor.started.get() for _ in range(8)}
    removal = asyncio.create_task(
        _remove_schedule_job(
            service, workspace, client_id, job.job_id, entry, "failed-queued-remove"
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(executor.cancel_requested[agent_id].wait() for agent_id in started)),
            timeout=2,
        )
        for release in executor.release.values():
            release.set()
        assert not await removal
        assert executor.started.empty(), (
            "A failed cancellation checkpoint must not let the queued child start"
        )
        assert _require_record(repository, failed_id).status is SubAgentStatus.QUEUED
        assert pool.has_active()
    finally:
        fail_cancelled = False
    assert await _remove_schedule_job(
        service,
        workspace,
        client_id,
        job.job_id,
        "user" if entry == "tool" else "tool",
        "retry-queued-remove",
    )
    assert all(
        _require_record(repository, record.agent_id).status is SubAgentStatus.CANCELLED
        for record in records
    )
    assert not pool.has_active()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
async def test_job_removal_checkpoint_failure_still_waits_for_siblings_in_the_same_pool(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    failed_id: str | None = None
    fail_cancelled = True

    def replace_text(path: Path, content: str) -> None:
        value = json.loads(content)
        if fail_cancelled and value["agent_id"] == failed_id and value["status"] == "cancelled":
            raise OSError("One sibling's terminal checkpoint failed")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    repository = SubAgentRecordStore(
        workspace.workspace_state, session_id, replace_text=replace_text
    )
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Drain siblings",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    first = _schedule_child(pool, job.job_id)
    failed_id = first.agent_id
    second = _schedule_child(pool, job.job_id)
    assert {await executor.started.get(), await executor.started.get()} == {
        first.agent_id,
        second.agent_id,
    }
    removal = asyncio.create_task(
        _remove_schedule_job(service, workspace, client_id, job.job_id, entry, "remove-siblings")
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(
                executor.cancel_requested[first.agent_id].wait(),
                executor.cancel_requested[second.agent_id].wait(),
            ),
            timeout=2,
        )
        executor.release[first.agent_id].set()
        with pytest.raises(SubAgentStoreError):
            await asyncio.wait_for(pool.wait([first.agent_id]), timeout=2)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(removal), timeout=0.1)
        executor.release[second.agent_id].set()
        assert not await removal
    finally:
        fail_cancelled = False
        for release in executor.release.values():
            release.set()
        await removal
        await _remove_schedule_job(
            service, workspace, client_id, job.job_id, entry, "retry-siblings"
        )
    assert not pool.has_active()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_concurrent_job_removal_keeps_the_source_fenced(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    entry: str,
    cancel_caller: bool,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Concurrent removal",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    repository = SubAgentRecordStore(workspace.workspace_state, session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = _schedule_child(pool, job.job_id)
    other_job_id = str(uuid4())
    unaffected = _schedule_child(pool, other_job_id)
    outside_path = workspace.workspace_path.parent / "outside-job-removal"
    outside_path.mkdir()
    outside_session_id = await _persist_session(
        outside_path,
        home=service.agent_home,
        title="Outside",
        created_at=_NOW,
        content="Unrelated Workspace",
    )
    outside_workspace = await service.attach_workspace(client_id, outside_path)
    outside_repository = SubAgentRecordStore(outside_workspace.workspace_state, outside_session_id)
    outside_pool = SubAgentPool(outside_repository, executor)
    outside_workspace.register_subagent_coordinator(outside_pool, outside_repository)
    outside = _schedule_child(outside_pool, job.job_id)
    assert {await executor.started.get() for _ in range(3)} == {
        record.agent_id,
        unaffected.agent_id,
        outside.agent_id,
    }
    removal = asyncio.create_task(
        _remove_schedule_job(service, workspace, client_id, job.job_id, entry, "concurrent-first")
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    if cancel_caller:
        removal.cancel()
    followup = asyncio.create_task(
        _remove_schedule_job(
            service,
            workspace,
            client_id,
            job.job_id,
            "user" if entry == "tool" else "tool",
            "concurrent-second",
        )
    )
    await asyncio.sleep(0)
    try:
        assert not followup.done()
        with pytest.raises(SubAgentRequestError, match="Schedule Job"):
            _schedule_child(pool, job.job_id)
        executor.release[record.agent_id].set()
        if cancel_caller:
            with pytest.raises(asyncio.CancelledError):
                await removal
            assert await followup
        else:
            assert await removal
            assert not await followup
        with pytest.raises(SubAgentRequestError, match="Schedule Job"):
            _schedule_child(pool, job.job_id)
        assert not executor.cancel_requested[unaffected.agent_id].is_set()
        assert _require_record(repository, unaffected.agent_id).status is SubAgentStatus.RUNNING
        assert not executor.cancel_requested[outside.agent_id].is_set()
        assert (
            _require_record(outside_repository, outside.agent_id).status is SubAgentStatus.RUNNING
        )
    finally:
        for release in executor.release.values():
            release.set()
        await asyncio.gather(removal, followup, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["tool", "user"])
@pytest.mark.parametrize("failure", ["before_replace", "after_replace", "unreadable"])
async def test_job_removal_store_failure_reopens_only_a_proven_unremoved_source(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    failure: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Store failure",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await workspace.schedule_service.add_user_job(job)
    repository = SubAgentRecordStore(workspace.workspace_state, session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    first = _schedule_child(pool, job.job_id)
    assert await executor.started.get() == first.agent_id

    def fail_replace(path: Path, content: str) -> None:
        if failure != "before_replace":
            HOST_FILESYSTEM.atomic_replace_text(
                path, "invalid schedule" if failure == "unreadable" else content
            )
        raise OSError("Schedule atomic replacement failed")

    monkeypatch.setattr(workspace.schedule_service._store, "_replace_text", fail_replace)
    assert not await _remove_schedule_job(
        service, workspace, client_id, job.job_id, entry, "failed-store"
    )
    assert not executor.cancel_requested[first.agent_id].is_set()
    if failure == "before_replace":
        recovered = _schedule_child(pool, job.job_id)
        assert await executor.started.get() == recovered.agent_id
    else:
        with pytest.raises(SubAgentRequestError, match="Schedule Job"):
            _schedule_child(pool, job.job_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["project", "service"])
async def test_cleanup_fences_all_pools_before_draining_and_respects_workspace_scope(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    tmp_path: Path,
    scope: str,
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    project, selected_workspace, _jobs = await service.register_project(
        client_id, workspace.workspace_path
    )
    assert selected_workspace is workspace
    other_path = tmp_path / "other-workspace"
    other_path.mkdir()
    outside_session_id = await _persist_session(
        other_path,
        home=service.agent_home,
        title="Outside removal",
        created_at=_NOW,
        content="Keep this Session.",
    )
    outside_workspace = await service.attach_workspace(client_id, other_path)
    executor = _DelayedExecutor()
    entries: list[tuple[WorkspaceRecord, SubAgentRecordStore, SubAgentPool, SubAgentRecord]] = []
    for current_workspace, current_session_id in (
        (workspace, session_id),
        (workspace, other_session_id),
        (outside_workspace, outside_session_id),
    ):
        repository = SubAgentRecordStore(current_workspace.workspace_state, current_session_id)
        pool = SubAgentPool(repository, executor)
        current_workspace.register_subagent_coordinator(pool, repository)
        record = pool.submit(
            title="Cleanup child",
            task="Wait for cleanup.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=_RESTORE_TOKEN,
            ),
            creator_snapshot=_snapshot(),
        )
        assert await executor.started.get() == record.agent_id
        entries.append((current_workspace, repository, pool, record))
    close = asyncio.create_task(
        complete_project_removal(service, client_id, project.project_id)
        if scope == "project"
        else service.stop()
    )
    first = entries[0][3]
    await asyncio.wait_for(executor.cancel_requested[first.agent_id].wait(), timeout=2)
    affected = entries[:2] if scope == "project" else entries
    try:
        assert not close.done()
        for current_workspace, repository, pool, _record in affected:
            with pytest.raises(SubAgentRequestError, match="not accepting"):
                pool.submit(
                    title="Rejected during cleanup",
                    task="Must not start.",
                    parent_run_id=_RUN_ID,
                    source=SubAgentSource(
                        kind=SubAgentSourceKind.FOREGROUND,
                        restore_run_token=_RESTORE_TOKEN,
                    ),
                    creator_snapshot=_snapshot(),
                )
            with pytest.raises(SubAgentRequestError, match="not accepting"):
                current_workspace.register_subagent_coordinator(pool, repository)
        assert service.confirmation._closed is False
        assert workspace.workspace_id in service.workspace_resources.resources
    finally:
        # Release all affected executions even if the admission assertions fail.
        for _current_workspace, _repository, _pool, record in affected:
            await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
            executor.release[record.agent_id].set()
        await close
        if scope == "project":
            executor.release[entries[2][3].agent_id].set()
    expected = SubAgentStatus.CANCELLED if scope == "project" else SubAgentStatus.INTERRUPTED
    for _current_workspace, repository, pool, record in affected:
        assert _require_record(repository, record.agent_id).status is expected
        assert not pool.has_active()
    assert workspace.workspace_id not in service.workspace_resources.resources
    if scope == "project":
        outside = entries[2]
        assert not executor.cancel_requested[outside[3].agent_id].is_set()
        result = await outside[2].wait([outside[3].agent_id])
        assert result[0].status is SubAgentStatus.COMPLETED
        assert outside_workspace.workspace_id in service.workspace_resources.resources
    else:
        assert service.confirmation._closed is True
        assert service.state == "stopped"


@pytest.mark.asyncio
async def test_service_close_marks_children_interrupted_before_closing_shared_confirmation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, _client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Service child",
        task="Wait for Service shutdown.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    close = asyncio.create_task(service.stop())
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not close.done()
    assert service.confirmation._closed is False
    executor.release[record.agent_id].set()
    await close

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.INTERRUPTED
    assert service.confirmation._closed is True
    assert service.state == "stopped"


@pytest.mark.asyncio
async def test_service_close_retries_failed_subagent_checkpoint_before_closing_resources(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, _client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Retry shutdown checkpoint",
        task="Persist the interrupted state before resources close.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id

    save = repository.save
    failures_remaining = 2

    def fail_first_interrupted_save(
        updated: SubAgentRecord, *, expected_revision: int | None = None
    ) -> SubAgentRecord:
        nonlocal failures_remaining
        if updated.status is SubAgentStatus.INTERRUPTED and failures_remaining:
            failures_remaining -= 1
            raise SubAgentStoreError("temporary storage failure")
        return save(updated, expected_revision=expected_revision)

    monkeypatch.setattr(repository, "save", fail_first_interrupted_save)
    close = asyncio.create_task(service.stop())
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()

    with pytest.raises(ServiceError, match="could not persist SubAgent shutdown state"):
        await close

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.RUNNING
    assert service.state == "draining"
    assert service.confirmation._closed is False
    assert workspace.workspace_id in service.workspace_resources.resources

    try:
        await service.stop()
    finally:
        if service.state != "stopped":
            pool.cancel(record.agent_id, interrupted=True)
            await service.stop()

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.INTERRUPTED
    assert service.confirmation._closed is True
    assert workspace.workspace_id not in service.workspace_resources.resources
    assert service.state == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["foreground", "background"])
async def test_subagent_confirmation_routes_to_session_after_claim_release(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    origin: Literal["foreground", "background"],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    record = _complete(repository, _register(repository, "Confirmation child"))
    claim = await workspace.claim(client_id, session_id)
    await workspace.release(client_id, session_id)
    owner = SubAgentConfirmationOwner(uuid4(), workspace.workspace_id, session_id, record.agent_id)
    request = ConfirmationRequest(
        uuid4(), "tool-call-1", "exec", "Run a command", {"command": "pytest"}
    )
    envelope = ConfirmationEnvelope(
        request=request,
        origin=origin,
        owner=owner,
        job_id="scheduled-job" if origin == "background" else None,
        title="Scheduled child" if origin == "background" else None,
    )
    client = service.client(client_id)
    sink = cast(_CollectingSink, client.sink)
    confirmation = asyncio.create_task(service.confirmation.request(envelope))
    async with asyncio.timeout(2):
        while not any(event["type"] == "confirmation.requested" for event in client.events):
            sink.changed.clear()
            await sink.changed.wait()
    emitted = next(event for event in client.events if event["type"] == "confirmation.requested")

    assert emitted["workspace_id"] == workspace.workspace_id
    assert emitted["session_id"] == session_id
    assert emitted["run_id"] is None
    payload = cast(dict[str, object], emitted["payload"])
    assert payload["origin"] == origin
    assert payload["owner"] == {
        "kind": "subagent",
        "generation_id": str(owner.generation_id),
        "workspace_id": workspace.workspace_id,
        "session_id": session_id,
        "agent_id": record.agent_id,
    }
    outsider = await service.register_client("cli")
    with pytest.raises(ServiceError) as forbidden:
        service.decide_confirmation(outsider.client_id, cast(str, payload["token"]), "approved")
    assert forbidden.value.code == "forbidden"
    service.decide_confirmation(client_id, cast(str, payload["token"]), "approved")
    assert await confirmation == "approved"
    assert claim.session_id == session_id


@pytest.mark.asyncio
async def test_subagent_http_routes_require_client_auth_and_csrf_for_cancel(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    record = _complete(repository, _register(repository, "HTTP child"))
    create_credential(service.agent_home)
    credential = read_credential(service.agent_home)
    server = TestServer(create_app(service), host="127.0.0.1")
    await server.start_server()
    headers = {
        "Authorization": f"Bearer {credential}",
        "X-Aide-Client": client_id,
    }
    path = (
        f"/api/v1/workspaces/{workspace.workspace_id}/sessions/{session_id}"
        f"/subagents/{record.agent_id}"
    )
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(
                server.make_url(
                    f"/api/v1/workspaces/{workspace.workspace_id}/sessions/{session_id}/subagents"
                ),
                headers=headers,
            ) as response:
                assert response.status == 200
                assert (await response.json())["items"][0]["agent_id"] == record.agent_id
            async with http.delete(server.make_url(path), headers=headers) as response:
                assert response.status == 403
            async with http.delete(
                server.make_url(path), headers={**headers, "X-Aide-CSRF": credential}
            ) as response:
                assert response.status == 200
                first = await response.json()
            async with http.delete(
                server.make_url(path), headers={**headers, "X-Aide-CSRF": credential}
            ) as response:
                assert response.status == 200
                second = await response.json()
    finally:
        await server.close()

    assert first["cancelled"] is False
    assert second == first


@pytest.mark.asyncio
async def test_workspace_startup_interrupts_persisted_subagent_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Startup",
        created_at=_NOW,
        content="Session history.",
    )
    repository = SubAgentRecordStore(WorkspaceState(workspace_path), session_id, now=lambda: _NOW)
    queued = _register(repository, "Queued before restart")
    running = repository.save(
        replace(
            _register(repository, "Running before restart"),
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=1,
        )
    )
    monkeypatch.setattr(
        "aide.service.runtime.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        recovered = workspace.subagent_repository(session_id)

        assert _require_record(recovered, queued.agent_id).status is SubAgentStatus.INTERRUPTED
        assert _require_record(recovered, running.agent_id).status is SubAgentStatus.INTERRUPTED
    finally:
        await service.stop()
