from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aide.agent.session.session import Session
from aide.agent.subagents.context import SubAgentToolContext
from aide.agent.subagents.coordinator import SubAgentPool
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentEvent,
    SubAgentExecutionResult,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.store import SubAgentRecordStore
from aide.agent.tools.base import BaseTool
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.subagents import build_subagent_tools
from aide.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


class _GateExecutor:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult:
        del emit
        await self.started.put(record.agent_id)
        await self.release.wait()
        return SubAgentExecutionResult(
            status=SubAgentStatus.COMPLETED,
            conversation=({"role": "assistant", "content": f"Finished {record.title}."},),
            context_state={},
            artifact_paths=(),
            result=f"Finished {record.title}.",
            error=None,
            usage={"model_calls": 1, "input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        del agent_id, interrupted
        self.release.set()
        return True


def _harness(
    tmp_path: Path,
    *,
    replace_text: Callable[[Path, str], None] = HOST_FILESYSTEM.atomic_replace_text,
) -> tuple[WorkspaceState, SubAgentRecordStore, SubAgentPool, _GateExecutor, ToolGateway]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session_id = Session.create(state, now=lambda: _NOW).session_id
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW, replace_text=replace_text)
    executor = _GateExecutor()
    pool = SubAgentPool(repository, executor)
    tool_context = SubAgentToolContext(
        coordinator=pool,
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=SubAgentCreatorSnapshot(
            provider_id="test-provider",
            model="test-model",
            reasoning_effort="mid",
            permission_level="workspace-write",
            shell="pwsh",
            tool_names=("read_file",),
            system_prompt="You are Aide.",
        ),
    )
    gateway = ToolGateway(
        workspace=workspace,
        additional_tools=build_subagent_tools(),
        tool_context=ToolRunContext(workspace=workspace, subagent=tool_context),
    )
    return state, repository, pool, executor, gateway


@pytest.mark.asyncio
async def test_spawn_without_creator_snapshot_does_not_register_a_task(tmp_path: Path) -> None:
    _, repository, _, _, gateway = _harness(tmp_path)
    context = gateway.tool_context
    assert context is not None and context.subagent is not None
    unavailable_gateway = ToolGateway(
        workspace=context.workspace,
        additional_tools=build_subagent_tools(),
        tool_context=replace(context, subagent=replace(context.subagent, creator_snapshot=None)),
    )
    result = await unavailable_gateway.call(ModelToolCall(
        id="spawn",
        name="spawn_agent",
        arguments=json.dumps({"title": "Inspect tests", "task": "Find focused tests."}),
    ))
    assert result.status == "error"
    assert "SubAgent Model Route is unavailable" in result.content
    assert repository.list().items == ()


@pytest.mark.asyncio
async def test_spawn_tool_returns_queued_and_list_hides_task_details(tmp_path: Path) -> None:
    _, repository, pool, executor, gateway = _harness(tmp_path)
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in gateway.schemas
        if schema["function"]["name"] in {"spawn_agent", "wait_agent", "list_agents"}
    }
    assert schemas["wait_agent"]["properties"]["agent_ids"]["type"] == "array"
    assert schemas["wait_agent"]["properties"]["agent_ids"]["minItems"] == 1

    spawned = await gateway.call(
        ModelToolCall(
            id="spawn",
            name="spawn_agent",
            arguments=json.dumps({"title": "Inspect tests", "task": "Find focused tests."}),
        )
    )
    assert spawned.status == "success"
    queued = json.loads(spawned.content)
    assert queued["status"] == "queued"
    saved = repository.get(queued["agent_id"])
    assert saved is not None and saved.status is SubAgentStatus.QUEUED

    listed = await gateway.call(ModelToolCall(id="list", name="list_agents", arguments="{}"))
    assert listed.status == "success"
    item = json.loads(listed.content)["items"][0]
    assert item["agent_id"] == queued["agent_id"]
    assert item["status"] == "queued"
    assert "task" not in item and "result" not in item

    started = await asyncio.wait_for(executor.started.get(), timeout=2)
    assert started == queued["agent_id"]
    executor.release.set()
    await pool.wait([queued["agent_id"]])


@pytest.mark.asyncio
async def test_wait_tool_returns_compact_results_and_rejects_invalid_or_foreign_ids(
    tmp_path: Path,
) -> None:
    state, _, pool, executor, gateway = _harness(tmp_path)
    invalid = await gateway.call(
        ModelToolCall(
            id="invalid",
            name="wait_agent",
            arguments=json.dumps({"agent_ids": []}),
        )
    )
    assert invalid.status == "error"

    unknown = await gateway.call(
        ModelToolCall(
            id="unknown",
            name="wait_agent",
            arguments=json.dumps({"agent_ids": [_RUN_ID]}),
        )
    )
    assert unknown.status == "error"

    spawned = await gateway.call(
        ModelToolCall(
            id="spawn",
            name="spawn_agent",
            arguments=json.dumps({"title": "Check routes", "task": "Inspect routing."}),
        )
    )
    agent_id = json.loads(spawned.content)["agent_id"]
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == agent_id
    executor.release.set()
    await pool.wait([agent_id])

    waited = await gateway.call(
        ModelToolCall(
            id="wait",
            name="wait_agent",
            arguments=json.dumps({"agent_ids": [agent_id, agent_id]}),
        )
    )
    assert waited.status == "success"
    results = json.loads(waited.content)
    assert len(results) == 1
    assert results[0]["status"] == "completed"
    assert results[0]["result"] == "Finished Check routes."
    assert "conversation" not in results[0]

    other_session = Session.create(
        state,
        now=lambda: _NOW,
    ).session_id
    other_store = SubAgentRecordStore(state, other_session, now=lambda: _NOW)
    foreign = other_store.register(
        title="Foreign task",
        task="Not owned by the current Session.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=SubAgentCreatorSnapshot(
            provider_id="test-provider",
            model="test-model",
            reasoning_effort="mid",
            permission_level="workspace-write",
            shell="pwsh",
            tool_names=("read_file",),
            system_prompt="You are Aide.",
        ),
    )
    foreign_result = await gateway.call(
        ModelToolCall(
            id="foreign",
            name="wait_agent",
            arguments=json.dumps({"agent_ids": [agent_id, foreign.agent_id]}),
        )
    )
    assert foreign_result.status == "error"


@pytest.mark.asyncio
async def test_spawn_tool_rejects_empty_fields_without_registering_a_task(tmp_path: Path) -> None:
    _, repository, _, _, gateway = _harness(tmp_path)
    result = await gateway.call(
        ModelToolCall(
            id="empty",
            name="spawn_agent",
            arguments=json.dumps({"title": " ", "task": "Do work."}),
        )
    )
    assert result.status == "error"
    assert repository.list().items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("spawn_agent", {}),
        ("spawn_agent", {"title": "Task", "task": ""}),
        ("spawn_agent", {"title": 12, "task": "Work"}),
        *(
            ("spawn_agent", {"title": "Task", "task": "Work", field: "override"})
            for field in ("session_id", "workspace", "model", "permission_level")
        ),
        ("wait_agent", {"agent_ids": "one"}),
        ("wait_agent", {"agent_ids": [" "]}),
        *(
            ("wait_agent", {"agent_ids": [_RUN_ID], "timeout_ms": value})
            for value in (-1, True, 1.5, "0")
        ),
        ("wait_agent", {"agent_ids": [_RUN_ID], "session_id": "override"}),
        ("list_agents", {"status": "unknown"}),
        ("list_agents", {"cursor": ""}),
        ("list_agents", {"cursor": 1}),
        *(("list_agents", {"limit": value}) for value in (0, 101, True, 1.5, "2")),
        ("list_agents", {"session_id": "override"}),
    ],
)
async def test_subagent_tools_reject_invalid_arguments_without_side_effects(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
) -> None:
    _, repository, _, executor, gateway = _harness(tmp_path)
    result = await gateway.call(
        ModelToolCall(id="invalid", name=name, arguments=json.dumps(arguments))
    )
    assert result.status == "error"
    assert repository.list().items == ()
    assert executor.started.empty()


@pytest.mark.asyncio
async def test_failed_registration_returns_tool_error_and_never_starts(tmp_path: Path) -> None:
    def fail_write(path: Path, content: str) -> None:
        del path, content
        raise OSError("simulated registration write failure")

    _, repository, _, executor, gateway = _harness(tmp_path, replace_text=fail_write)
    result = await gateway.call(
        ModelToolCall(
            id="spawn",
            name="spawn_agent",
            arguments=json.dumps({"title": "Task", "task": "Do work"}),
        )
    )
    assert result.status == "error"
    assert result.content == "The SubAgent task could not be registered."
    assert repository.list().items == ()
    assert executor.started.empty()


@pytest.mark.asyncio
async def test_tool_wait_timeout_and_caller_cancellation_leave_execution_running(
    tmp_path: Path,
) -> None:
    _, repository, _, executor, gateway = _harness(tmp_path)
    spawned = await gateway.call(
        ModelToolCall(
            id="spawn",
            name="spawn_agent",
            arguments=json.dumps({"title": "Independent", "task": "Continue after wait"}),
        )
    )
    agent_id = json.loads(spawned.content)["agent_id"]
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == agent_id
    for timeout_ms in (0, 1):
        timed = await gateway.call(
            ModelToolCall(
                id=f"timed_{timeout_ms}",
                name="wait_agent",
                arguments=json.dumps({"agent_ids": [agent_id], "timeout_ms": timeout_ms}),
            )
        )
        assert timed.status == "success" and json.loads(timed.content) == []
    call = ModelToolCall(
        id="wait", name="wait_agent", arguments=json.dumps({"agent_ids": [agent_id]})
    )
    pending = asyncio.create_task(gateway.call(call))
    turn = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(turn.set_result, None)
    await turn
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    saved = repository.get(agent_id)
    assert saved is not None and saved.status is SubAgentStatus.RUNNING
    executor.release.set()
    completed = await asyncio.wait_for(gateway.call(call), timeout=2)
    assert completed.status == "success"
    assert json.loads(completed.content)[0]["status"] == "completed"
    assert (await gateway.call(call)).content == completed.content


@pytest.mark.asyncio
async def test_wait_tool_reports_result_write_failure_and_keeps_checkpoint(tmp_path: Path) -> None:
    failed_write = asyncio.Event()

    def replace_text(path: Path, content: str) -> None:
        if json.loads(content)["status"] == "completed" and not failed_write.is_set():
            failed_write.set()
            raise OSError("simulated private storage failure")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    _, repository, _, executor, gateway = _harness(tmp_path, replace_text=replace_text)
    spawned = await gateway.call(
        ModelToolCall(
            id="spawn",
            name="spawn_agent",
            arguments=json.dumps({"title": "Task", "task": "Do work"}),
        )
    )
    agent_id = json.loads(spawned.content)["agent_id"]
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == agent_id
    pending = asyncio.create_task(
        gateway.call(
            ModelToolCall(
                id="wait", name="wait_agent", arguments=json.dumps({"agent_ids": [agent_id]})
            )
        )
    )
    executor.release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert failed_write.is_set()
    assert result.status == "error"
    assert result.content == "SubAgent records could not be read or saved reliably."
    saved = repository.get(agent_id)
    assert saved is not None and saved.status is SubAgentStatus.RUNNING
    assert saved.result is None


@pytest.mark.asyncio
async def test_list_tool_pages_and_later_run_reads_long_wait_result_as_artifact(
    tmp_path: Path,
) -> None:
    state, repository, pool, executor, gateway = _harness(tmp_path)
    ids = []
    for index in range(3):
        result = await gateway.call(
            ModelToolCall(
                id=f"spawn_{index}",
                name="spawn_agent",
                arguments=json.dumps({"title": f"Task {index} " + "x" * 1000, "task": "Work"}),
            )
        )
        assert result.status == "success"
        ids.append(json.loads(result.content)["agent_id"])
    for agent_id in ids:
        assert await asyncio.wait_for(executor.started.get(), timeout=2) == agent_id

    first = await gateway.call(
        ModelToolCall(id="page1", name="list_agents", arguments='{"limit":2}')
    )
    page = json.loads(first.content)
    assert [item["agent_id"] for item in page["items"]] == ids[:2]
    second = await gateway.call(
        ModelToolCall(
            id="page2",
            name="list_agents",
            arguments=json.dumps({"cursor": page["next_cursor"], "limit": 2, "status": "running"}),
        )
    )
    page = json.loads(second.content)
    assert [item["agent_id"] for item in page["items"]] == ids[2:]
    assert page["next_cursor"] is None

    executor.release.set()
    await asyncio.gather(*(pool.wait([agent_id]) for agent_id in ids))
    later_gateway = ToolGateway(
        workspace=state.workspace_path,
        additional_tools=build_subagent_tools(),
        tool_context=gateway.tool_context,
    )
    waited = await later_gateway.call(
        ModelToolCall(id="wait_later", name="wait_agent", arguments=json.dumps({"agent_ids": ids}))
    )
    assert waited.status == "success"
    results = json.loads(waited.content)
    assert [item["agent_id"] for item in results] == ids
    assert all(
        set(item) == {"agent_id", "title", "status", "result", "error", "usage"} for item in results
    )
    externalized = BaseTool.handle_result(
        waited.content,
        workspace=state.workspace_path,
        session_id=repository.session_id,
        tool_call_id=waited.tool_call_id,
        limit=200,
    )
    assert externalized.artifact is not None
    artifact = state.workspace_path / externalized.artifact.path
    assert artifact.read_text(encoding="utf-8") == waited.content
    assert len(externalized.content) < len(waited.content)
