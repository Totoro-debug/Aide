from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest

from aide.agent.memory.manager import MemoryManager
from aide.agent.session.session import Session
from aide.agent.subagents.executor import SubAgentRunnerExecutor
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentEventKind,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.store import SubAgentRecordStore
from aide.agent.tools.base import BaseTool
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.exec_host import create_exec_host, resolve_exec_shell
from aide.agent.tools.core.exec_policy import ExecShellSelector, ResolvedExecShell
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import (
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
)
from aide.agent.workspace_state import WorkspaceState
from aide.errors import ErrorInfo
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
)
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import ScriptedFakeProvider, StreamScript
from tests.test_model_router import configuration, routed_configuration

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"
_PARENT_MARKERS = (
    "parent-history-marker",
    "parent-conversation-summary-marker",
    "parent-action-summary-marker",
    "parent-blackboard-marker",
)


class _LongResultTool(BaseTool):
    name = "long_result"
    description = "Return a long result for artifact verification."

    async def execute(self) -> str:
        return "long-result-marker " * 100


class _MarkerTool(BaseTool):
    name = "marker_tool"
    description = "Find a marker for the child run."

    async def execute(self) -> str:
        return "marker"


class _ForbiddenTool(BaseTool):
    name = "spawn_agent"
    description = "Must be absent from a SubAgent run."

    async def execute(self) -> str:
        return "forbidden"


def _workspace(tmp_path: Path) -> tuple[WorkspaceState, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session_id = Session.create(state, now=lambda: _NOW).session_id
    return state, session_id


def _creator_snapshot(*names: str) -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="default-provider",
        model="default-model",
        reasoning_effort="high",
        permission_level="workspace-write",
        shell="pwsh",
        tool_schemas=tuple({"name": name} for name in names),
        system_prompt="child-only-system-prompt",
    )


def _register_running(
    store: SubAgentRecordStore,
    *,
    title: str,
    snapshot: SubAgentCreatorSnapshot,
    task: str = "child-only-task",
) -> SubAgentRecord:
    queued = store.register(
        title=title,
        task=task,
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=snapshot,
    )
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=queued.revision + 1,
    )
    return store.save(running)


def _response(content: str) -> ModelCompleted:
    return ModelCompleted(
        response=ModelResponse(
            message=AssistantModelMessage(content=content),
            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
            finish_reason="stop",
        )
    )


def _tool_call(name: str, call_id: str, arguments: dict[str, object]) -> ModelCompleted:
    return ModelCompleted(
        response=ModelResponse(
            message=AssistantModelMessage(
                content="",
                tool_calls=(
                    ModelToolCall(
                        id=call_id,
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            finish_reason="tool_calls",
        )
    )


def _router(provider: ScriptedFakeProvider) -> ModelRouter:
    return ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
    )


@pytest.mark.asyncio
async def test_subagent_runner_uses_frozen_context_and_persists_full_tool_artifact(
    tmp_path: Path,
) -> None:
    state, _ = _workspace(tmp_path)
    parent = Session.create(state, now=lambda: _NOW)
    parent.commit_agent_run(
        [{"role": "user", "content": _PARENT_MARKERS[0]}],
        pending_last_compacted=0,
        pending_action_summary=_PARENT_MARKERS[2],
        metadata_updates={
            "blackboard": {
                "goal": _PARENT_MARKERS[3],
                "completion_boundary": "parent-only completion",
            }
        },
    )
    await parent.wait_for_pending_persist()
    await MemoryManager(state).append_summary(_PARENT_MARKERS[1], _NOW)
    parent_before = deepcopy((parent.messages, parent.metadata))
    session_file = state.sessions_directory / f"{parent.session_id}.jsonl"
    session_before = session_file.read_bytes()
    summary_before = (state.memory_directory / "summary.jsonl").read_bytes()
    session_id = parent.session_id
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    tool = _LongResultTool()
    snapshot = _creator_snapshot(
        "long_result", "tool_search", "spawn_agent", "wait_agent", "schedule"
    )
    record = _register_running(store, title="Long result", snapshot=snapshot)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(_tool_call("tool_search", "search-call", {"query": "long result"}),)
            ),
            StreamScript(events=(_tool_call("long_result", "shared-call-id", {}),)),
            StreamScript(events=(_response("child-finished"),)),
        )
    )
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
        tool_context=ToolRunContext(workspace=state.workspace_path),
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=_router(provider),
        tool_gateway=gateway,
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=80,
        now=lambda: _NOW,
    )
    events = []
    checkpoint_records = []

    async def emit(event: object) -> None:
        events.append(event)
        if getattr(event, "kind", None) is SubAgentEventKind.USAGE:
            checkpoint_records.append(store.get(record.agent_id))

    result = await executor.execute(record, emit=emit)
    persisted = store.get(record.agent_id)

    assert result.status is SubAgentStatus.COMPLETED
    assert result.result == "child-finished"
    assert len(provider.stream_requests) == 3
    assert all(call.model == "default-model" for call in provider.stream_requests)
    assert all(call.reasoning_effort == "high" for call in provider.stream_requests)
    assert [message["role"] for message in result.conversation[:2]] == ["user", "assistant"]
    assert result.conversation[0]["content"] == "child-only-task"
    assistant_calls = result.conversation[3]["tool_calls"]
    assert assistant_calls[0]["id"] == "shared-call-id"
    assert result.artifact_paths == (
        f".aide/artifacts/{session_id}/{record.agent_id}_shared-call-id.txt",
    )
    artifact = state.workspace_path / Path(*result.artifact_paths[0].split("/"))
    assert artifact.read_text(encoding="utf-8") == "long-result-marker " * 100
    assert persisted is not None
    assert persisted.status is SubAgentStatus.RUNNING
    assert persisted.conversation == result.conversation
    assert persisted.usage == result.usage
    assert result.usage["model_calls"] == 3
    assert any(
        checkpoint is not None
        and checkpoint.status is SubAgentStatus.RUNNING
        and checkpoint.usage is not None
        and checkpoint.usage["model_calls"] == 2
        and checkpoint.artifact_paths == result.artifact_paths
        and any(
            message.get("role") == "tool" and message.get("name") == "long_result"
            for message in checkpoint.conversation
        )
        for checkpoint in checkpoint_records
    )
    for marker in _PARENT_MARKERS:
        assert all(marker not in json.dumps(call.messages) for call in provider.stream_requests)
    assert all(
        schema["function"]["name"] not in {"spawn_agent", "wait_agent", "schedule"}
        for call in provider.stream_requests
        for schema in call.tools
    )
    assert any(getattr(event, "kind", None) is not None for event in events)
    assert (parent.messages, parent.metadata) == parent_before
    assert session_file.read_bytes() == session_before
    assert (state.memory_directory / "summary.jsonl").read_bytes() == summary_before
    assert provider.stream_requests[0].messages[0] == {
        "role": "system",
        "content": snapshot.system_prompt,
    }


@pytest.mark.asyncio
async def test_subagent_gateway_limits_catalog_and_search_to_creator_snapshot(
    tmp_path: Path,
) -> None:
    tools = (_MarkerTool(), _ForbiddenTool())
    gateway = ToolGateway._for_memory(
        tools,
        tool_context=ToolRunContext(workspace=tmp_path),
    )
    parent = build_agent_run_gateway(gateway)
    parent_exposure = parent.exposed_names
    child = build_agent_run_gateway(
        gateway,
        allowed_names=("marker_tool", "tool_search", "spawn_agent"),
        excluded_names=("spawn_agent", "wait_agent", "schedule"),
    )

    assert {schema["function"]["name"] for schema in child.schemas} == {"tool_search"}
    search = await child.call(
        ModelToolCall(
            id="search-call",
            name="tool_search",
            arguments=json.dumps({"query": "marker"}),
        )
    )
    forbidden = await child.call(
        ModelToolCall(id="forbidden-call", name="spawn_agent", arguments="{}")
    )

    assert json.loads(search.content) == ["marker_tool"]
    assert forbidden.status == "error"
    assert "spawn_agent" not in child.exposed_names
    assert parent.exposed_names == parent_exposure
    assert "marker_tool" not in {schema["function"]["name"] for schema in parent.schemas}


@pytest.mark.asyncio
async def test_two_subagents_with_same_tool_call_id_keep_distinct_artifacts(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    tool = _LongResultTool()
    snapshot = _creator_snapshot("long_result", "tool_search")
    first = _register_running(store, title="First", snapshot=snapshot)
    second = _register_running(store, title="Second", snapshot=snapshot)
    tool_response = ModelResponse(
        message=AssistantModelMessage(
            content="",
            tool_calls=(ModelToolCall(id="same-call-id", name="long_result", arguments="{}"),),
        ),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(_tool_call("tool_search", "search-one", {"query": "long result"}),)
            ),
            StreamScript(events=(ModelCompleted(response=tool_response),)),
            StreamScript(events=(_response("first-done"),)),
            StreamScript(
                events=(_tool_call("tool_search", "search-two", {"query": "long result"}),)
            ),
            StreamScript(events=(ModelCompleted(response=tool_response),)),
            StreamScript(events=(_response("second-done"),)),
        )
    )
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
        tool_context=ToolRunContext(workspace=state.workspace_path),
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=_router(provider),
        tool_gateway=gateway,
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=80,
        now=lambda: _NOW,
    )

    first_result = await executor.execute(first, emit=_ignore_event)
    second_result = await executor.execute(second, emit=_ignore_event)

    assert len(first_result.artifact_paths) == len(second_result.artifact_paths) == 1
    assert first_result.artifact_paths[0] != second_result.artifact_paths[0]
    for path in (*first_result.artifact_paths, *second_result.artifact_paths):
        assert (state.workspace_path / Path(*path.split("/"))).read_text(encoding="utf-8") == (
            "long-result-marker " * 100
        )
    assert first.agent_id in first_result.artifact_paths[0]
    assert second.agent_id in second_result.artifact_paths[0]


@pytest.mark.asyncio
async def test_service_interruption_returns_interrupted_without_terminal_record_update(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store,
        title="Interrupted",
        snapshot=_creator_snapshot("tool_search", "marker_tool"),
    )
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(TextDelta(delta="partial"),)),))
    gateway = ToolGateway._for_memory(
        (_MarkerTool(),),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
        tool_context=ToolRunContext(workspace=state.workspace_path),
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=_router(provider),
        tool_gateway=gateway,
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=80,
        now=lambda: _NOW,
    )

    async def interrupt(event: object) -> None:
        if getattr(event, "kind", None) is SubAgentEventKind.OUTPUT:
            assert executor.request_cancel(record.agent_id, interrupted=True)

    result = await executor.execute(record, emit=interrupt)
    persisted = store.get(record.agent_id)

    assert result.status is SubAgentStatus.INTERRUPTED
    assert result.error is not None
    assert result.error.code == "service_interrupted"
    assert result.result == "partial"
    assert result.usage["model_calls"] == 1
    assert result.conversation[0] == {"role": "user", "content": record.task}
    assert persisted is not None
    assert persisted.status is SubAgentStatus.RUNNING
    assert persisted.conversation == result.conversation


@pytest.mark.asyncio
async def test_compaction_summaries_are_persisted_in_child_context_state(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store,
        title="Compacted child",
        snapshot=_creator_snapshot("tool_search", "marker_tool"),
        task="child-task-marker " * 1000,
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(_tool_call("tool_search", "search-call", {"query": "marker"}),)),
            StreamScript(events=(_response("child-finished"),)),
        ),
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="child-fact-summary"),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="child-action-summary"),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
        ),
    )
    gateway = ToolGateway._for_memory(
        (_MarkerTool(),),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
        tool_context=ToolRunContext(workspace=state.workspace_path),
    )
    config = configuration()
    chat_route = replace(config.models.routes["default"], context_window=4000, max_output=500)
    config = replace(
        config,
        models=replace(
            config.models,
            routes={**config.models.routes, "chat": chat_route},
        ),
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=ModelRouter(configuration=config, provider_factory=lambda _: provider),
        tool_gateway=gateway,
        compact_ratio=0.5,
        max_iterations=50,
        max_tool_result_chars=80,
        now=lambda: _NOW,
    )

    result = await executor.execute(record, emit=_ignore_event)
    persisted = store.get(record.agent_id)

    assert result.status is SubAgentStatus.COMPLETED
    assert len(provider.complete_requests) == 2
    assert result.context_state["conversation_summaries"] == [
        {"timestamp": _NOW.isoformat(), "content": "child-fact-summary"}
    ]
    assert result.context_state["summary"] == "child-action-summary"
    assert persisted is not None
    assert persisted.status is SubAgentStatus.RUNNING
    assert persisted.context_state == result.context_state
    assert persisted.conversation[0]["content"] == record.task
    assert result.usage["model_calls"] == 4
    assert result.usage["total_tokens"] == 24
    assert not (state.memory_directory / "summary.jsonl").exists()
    for marker in _PARENT_MARKERS:
        assert all(marker not in json.dumps(call.messages) for call in provider.complete_requests)


async def _ignore_event(event: object) -> None:
    del event


def _executor(
    state: WorkspaceState,
    store: SubAgentRecordStore,
    provider: ModelProvider,
    gateway: ToolGateway,
) -> SubAgentRunnerExecutor:
    return SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=ModelRouter(
            configuration=configuration(), provider_factory=lambda _: provider
        ),
        tool_gateway=gateway,
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=5000,
        now=lambda: _NOW,
    )


class _BlockingTool(BaseTool):
    name = "blocking_tool"
    description = "Wait until cancelled and expose completion of cleanup."

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def execute(self) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaning.set()
            await self.release_cleanup.wait()
            self.cleaned.set()
        return "unreachable"


class _BlockingProvider(ScriptedFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str = "test-model",
        max_output: int = 1024,
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int = 30,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        self.started.set()
        try:
            yield TextDelta(delta="useful-partial")
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_cancel_interrupts_a_blocked_provider_and_preserves_partial_output(
    tmp_path: Path, interrupted: bool
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(store, title="Blocked model", snapshot=_creator_snapshot())
    provider = _BlockingProvider()
    executor = _executor(state, store, provider, ToolGateway(workspace=state.workspace_path))
    execution = asyncio.create_task(executor.execute(record, emit=_ignore_event))
    await asyncio.wait_for(provider.started.wait(), timeout=5)

    assert executor.request_cancel(record.agent_id, interrupted=interrupted)
    result = await asyncio.wait_for(asyncio.shield(execution), timeout=5)

    assert provider.cleaned.is_set()
    assert result.status is (
        SubAgentStatus.INTERRUPTED if interrupted else SubAgentStatus.CANCELLED
    )
    assert result.result == "useful-partial"
    assert result.error is not None
    assert result.error.code == ("service_interrupted" if interrupted else "turn_cancelled")
    assert result.usage["model_calls"] == 1
    assert not executor.request_cancel(record.agent_id)
    persisted = store.get(record.agent_id)
    assert persisted is not None
    assert persisted.conversation == result.conversation


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_the_tool_before_execution_returns(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store, title="Blocked tool", snapshot=_creator_snapshot("blocking_tool")
    )
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(_tool_call("blocking_tool", "blocked-call", {}),)),)
    )
    tool = _BlockingTool()
    gateway = ToolGateway._for_memory(
        (tool,),
        tool_context=ToolRunContext(workspace=state.workspace_path),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
    )
    executor = _executor(state, store, provider, gateway)
    execution = asyncio.create_task(executor.execute(record, emit=_ignore_event))
    await asyncio.wait_for(tool.started.wait(), timeout=5)

    assert executor.request_cancel(record.agent_id)
    await asyncio.wait_for(tool.cleaning.wait(), timeout=5)
    for _ in range(2):
        execution.cancel()
        yielded = asyncio.Event()
        asyncio.get_running_loop().call_soon(yielded.set)
        await yielded.wait()
        assert not execution.done()
        assert executor.request_cancel(record.agent_id, interrupted=True)
        assert not tool.cleaned.is_set()
    tool.release_cleanup.set()
    result = await asyncio.wait_for(asyncio.shield(execution), timeout=5)

    assert tool.cleaned.is_set()
    assert result.status is SubAgentStatus.INTERRUPTED
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    assert not executor.request_cancel(record.agent_id)
    with pytest.raises(ValueError, match="only once"):
        await executor.execute(record, emit=_ignore_event)
    assert len(provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_model_error_keeps_partial_result_and_usage_without_restarting(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(store, title="Failed model", snapshot=_creator_snapshot())
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="useful-partial"),),
                error=ModelCallError(ErrorInfo(code="model_failed", message="Model failed.")),
            ),
        )
    )
    executor = _executor(state, store, provider, ToolGateway(workspace=state.workspace_path))

    result = await executor.execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.FAILED
    assert result.error is not None and result.error.code == "model_failed"
    assert result.result == "useful-partial"
    assert result.usage["model_calls"] == 1
    with pytest.raises(ValueError, match="only once"):
        await executor.execute(record, emit=_ignore_event)
    assert len(provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_iteration_limit_is_failed_after_exactly_fifty_model_calls(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store, title="Iteration limit", snapshot=_creator_snapshot("marker_tool")
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(_tool_call("marker_tool", f"call-{index}", {}),))
            for index in range(50)
        )
    )
    gateway = ToolGateway._for_memory(
        (_MarkerTool(),),
        tool_context=ToolRunContext(workspace=state.workspace_path),
        permission_context=PermissionContext(workspace_root=state.workspace_path),
    )

    result = await _executor(state, store, provider, gateway).execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.FAILED
    assert result.error is not None and result.error.code == "agent_iteration_limit"
    assert len(provider.stream_requests) == result.usage["model_calls"] == 50
    assert result.usage["total_tokens"] == 600
    persisted = store.get(record.agent_id)
    assert persisted is not None and persisted.conversation == result.conversation


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["spawn_agent", "wait_agent", "schedule"])
async def test_all_prohibited_tools_are_absent_in_schema_search_and_direct_calls(
    tmp_path: Path, name: str
) -> None:
    class ProhibitedTool(BaseTool):
        description = "Forbidden delegation or scheduling capability."

        async def execute(self) -> str:
            raise AssertionError("Prohibited tool must never execute")

    ProhibitedTool.name = name
    child = build_agent_run_gateway(
        ToolGateway._for_memory((ProhibitedTool(), _MarkerTool())),
        allowed_names=(name, "tool_search", "marker_tool"),
        excluded_names=("spawn_agent", "wait_agent", "schedule"),
    )
    assert name not in {schema["function"]["name"] for schema in child.schemas}
    search = await child.call(
        ModelToolCall(id="search", name="tool_search", arguments=json.dumps({"query": name}))
    )
    assert name not in json.loads(search.content)
    direct = await child.call(ModelToolCall(id="direct", name=name, arguments="{}"))
    assert direct.status == "error"


@pytest.mark.asyncio
async def test_snapshot_permission_shell_and_model_survive_parent_configuration_changes(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    snapshot = replace(_creator_snapshot("write_file"), permission_level="read-only")
    record = _register_running(store, title="Frozen settings", snapshot=snapshot)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    _tool_call(
                        "write_file", "write", {"path": "blocked.txt", "content": "blocked"}
                    ),
                )
            ),
            StreamScript(events=(_response("done"),)),
        )
    )
    router = ModelRouter(configuration=routed_configuration(), provider_factory=lambda _: provider)
    router.set_reasoning_effort("low")
    assert router.route_status("chat").model != snapshot.model
    observed_shells = []

    def resolve_snapshot_shell(selector: ExecShellSelector) -> ResolvedExecShell:
        observed_shells.append(selector)
        return resolve_exec_shell("pwsh")

    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=router,
        tool_gateway=ToolGateway(
            workspace=state.workspace_path,
            permission_context=PermissionContext(
                workspace_root=state.workspace_path, level="full-access"
            ),
            tool_context=ToolRunContext(
                workspace=state.workspace_path,
                exec_host=create_exec_host(resolve_exec_shell("powershell")),
            ),
        ),
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=5000,
        exec_shell_resolver=resolve_snapshot_shell,
        now=lambda: _NOW,
    )
    result = await executor.execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.COMPLETED
    assert observed_shells == [snapshot.shell]
    assert not (state.workspace_path / "blocked.txt").exists()
    assert any(message.get("status") == "refused" for message in result.conversation)
    assert all(
        request.model == snapshot.model and request.reasoning_effort == "high"
        for request in provider.stream_requests
    )


@pytest.mark.asyncio
async def test_checkpoint_failure_reports_storage_error_and_actual_consumed_usage(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)

    def save_before_terminal(path: Path, content: str) -> None:
        value = json.loads(content)
        if len(value["conversation"]) > 1:
            raise OSError("Checkpoint publication failed")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    store = SubAgentRecordStore(
        state, session_id, now=lambda: _NOW, replace_text=save_before_terminal
    )
    record = _register_running(store, title="Failed checkpoint", snapshot=_creator_snapshot())
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(_response("finished"),)),))
    events = []

    async def emit(event: object) -> None:
        events.append(event)

    result = await _executor(
        state, store, provider, ToolGateway(workspace=state.workspace_path)
    ).execute(record, emit=emit)

    assert result.status is SubAgentStatus.FAILED
    assert result.error is not None and result.error.code == "persistence_error"
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    persisted = store.get(record.agent_id)
    assert persisted is not None
    assert persisted.status is SubAgentStatus.RUNNING
    assert persisted.conversation == ({"role": "user", "content": record.task},)
    assert all(getattr(event, "kind", None) is not SubAgentEventKind.STATUS for event in events)


class _MutationRecorder:
    def __init__(self) -> None:
        self.before: list[tuple[UUID, Path]] = []
        self.after: list[tuple[UUID, Path]] = []

    def begin_write(self, run_token: UUID, resolved_target: Path) -> Callable[[], None]:
        self.before.append((run_token, resolved_target))

        def complete() -> None:
            self.after.append((run_token, resolved_target))

        return complete


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [False, True])
async def test_file_restore_ownership_and_prompt_follow_the_creation_source(
    tmp_path: Path, scheduled: bool
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    source = (
        SubAgentSource(
            kind=SubAgentSourceKind.SCHEDULE, job_id=_RESTORE_TOKEN, occurrence_id="occurrence"
        )
        if scheduled
        else SubAgentSource(kind=SubAgentSourceKind.FOREGROUND, restore_run_token=_RESTORE_TOKEN)
    )
    prompt = "schedule-base-prompt" if scheduled else "foreground-base-prompt with skills"
    queued = store.register(
        title="Source ownership",
        task="write child data",
        parent_run_id=_RUN_ID,
        source=source,
        creator_snapshot=replace(_creator_snapshot("write_file"), system_prompt=prompt),
    )
    record = store.save(
        replace(
            queued, status=SubAgentStatus.RUNNING, started_at=_NOW, revision=queued.revision + 1
        )
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    _tool_call("write_file", "write", {"path": "child.txt", "content": "child"}),
                )
            ),
            StreamScript(events=(_response("done"),)),
        )
    )
    recorder = _MutationRecorder()
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=_router(provider),
        tool_gateway=ToolGateway(workspace=state.workspace_path),
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=5000,
        file_mutation_recorder_for=lambda _: recorder,
        now=lambda: _NOW,
    )
    result = await executor.execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.COMPLETED
    assert (state.workspace_path / "child.txt").read_text(encoding="utf-8") == "child"
    expected = (
        []
        if scheduled
        else [(UUID(_RESTORE_TOKEN), (state.workspace_path / "child.txt").resolve())]
    )
    assert recorder.before == recorder.after == expected
    assert all(
        request.messages[0] == {"role": "system", "content": prompt}
        for request in provider.stream_requests
    )


@pytest.mark.asyncio
async def test_list_agents_remains_searchable_and_callable_in_the_child_catalog(
    tmp_path: Path,
) -> None:
    class ListAgentsTool(BaseTool):
        name = "list_agents"
        description = "List current Session agents."

        async def execute(self) -> str:
            return "session-agent-list"

    gateway = ToolGateway._for_memory((ListAgentsTool(), _LongResultTool()))
    child = build_agent_run_gateway(gateway, allowed_names=("list_agents", "tool_search"))
    search = await child.call(
        ModelToolCall(
            id="search", name="tool_search", arguments=json.dumps({"query": "list agents"})
        )
    )
    assert json.loads(search.content) == ["list_agents"]
    direct = await child.call(ModelToolCall(id="list", name="list_agents", arguments="{}"))
    assert direct.status == "success" and direct.content == "session-agent-list"
    assert "list_agents" in {schema["function"]["name"] for schema in child.schemas}
    excluded = await child.call(ModelToolCall(id="excluded", name="long_result", arguments="{}"))
    assert excluded.status == "error"


@pytest.mark.asyncio
async def test_user_cancellation_cleans_a_pending_confirmation(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store,
        title="Pending confirmation",
        snapshot=replace(_creator_snapshot("write_file"), permission_level="read-only"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    _tool_call("write_file", "write", {"path": "never.txt", "content": "never"}),
                )
            ),
        )
    )
    pending = asyncio.Event()
    cleaned = asyncio.Event()

    async def confirmation(request: ConfirmationRequest) -> Literal["approved"]:
        assert request.tool_name == "write_file"
        pending.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return "approved"

    def confirmation_for(_: SubAgentRecord) -> ConfirmationRequester:
        return confirmation

    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=_router(provider),
        tool_gateway=ToolGateway(workspace=state.workspace_path),
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=5000,
        confirmation_for=confirmation_for,
        now=lambda: _NOW,
    )
    execution = asyncio.create_task(executor.execute(record, emit=_ignore_event))
    await asyncio.wait_for(pending.wait(), timeout=5)
    assert executor.request_cancel(record.agent_id)
    result = await asyncio.wait_for(asyncio.shield(execution), timeout=5)

    assert result.status is SubAgentStatus.CANCELLED
    assert cleaned.is_set()
    assert not (state.workspace_path / "never.txt").exists()


@pytest.mark.asyncio
async def test_compaction_keeps_earlier_model_output_in_the_persisted_conversation(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register_running(
        store,
        title="Keep raw output",
        snapshot=_creator_snapshot("marker_tool"),
    )
    raw_output = "pre-compaction-output " * 400
    first_response = ModelCompleted(
        response=ModelResponse(
            message=AssistantModelMessage(
                content=raw_output,
                tool_calls=(ModelToolCall(id="first", name="marker_tool", arguments="{}"),),
            ),
            usage=ModelUsage(input_tokens=200, output_tokens=2400, total_tokens=2600),
            finish_reason="tool_calls",
        )
    )
    second_response = _tool_call("marker_tool", "second", {})
    second_response = replace(
        second_response,
        response=replace(
            second_response.response,
            usage=ModelUsage(input_tokens=2400, output_tokens=2, total_tokens=2402),
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(first_response,)),
            StreamScript(events=(second_response,)),
            StreamScript(events=(_response("done"),)),
        ),
        completions=(_response(f"summary-{index}").response for index in range(6)),
    )
    config = configuration()
    chat_route = replace(config.models.routes["default"], context_window=4000, max_output=500)
    config = replace(
        config, models=replace(config.models, routes={**config.models.routes, "chat": chat_route})
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=ModelRouter(configuration=config, provider_factory=lambda _: provider),
        tool_gateway=ToolGateway(workspace=state.workspace_path, additional_tools=(_MarkerTool(),)),
        compact_ratio=0.5,
        max_iterations=50,
        max_tool_result_chars=5000,
        now=lambda: _NOW,
    )
    result = await executor.execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.COMPLETED
    assert provider.complete_requests
    assert any(raw_output in json.dumps(request.messages) for request in provider.complete_requests)
    persisted = store.get(record.agent_id)
    assert persisted is not None
    assert any(message.get("content") == raw_output for message in persisted.conversation)
    assert all(
        message.get("content") != raw_output for message in provider.stream_requests[-1].messages
    )
    assert not (state.memory_directory / "summary.jsonl").exists()


@pytest.mark.asyncio
async def test_failed_child_summary_save_keeps_its_consumed_memory_usage(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)

    def reject_summary(path: Path, content: str) -> None:
        if json.loads(content).get("context_state"):
            raise OSError("Summary publication failed")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW, replace_text=reject_summary)
    record = _register_running(
        store,
        title="Failed summary",
        snapshot=_creator_snapshot(),
        task="long-task " * 2000,
    )
    provider = ScriptedFakeProvider(completions=(_response("fact-summary").response,))
    config = configuration()
    chat_route = replace(config.models.routes["default"], context_window=4000, max_output=500)
    config = replace(
        config, models=replace(config.models, routes={**config.models.routes, "chat": chat_route})
    )
    executor = SubAgentRunnerExecutor(
        workspace_id="workspace-id",
        workspace_state=state,
        repository=store,
        model_router=ModelRouter(configuration=config, provider_factory=lambda _: provider),
        tool_gateway=ToolGateway(workspace=state.workspace_path),
        compact_ratio=0.5,
        max_iterations=50,
        max_tool_result_chars=5000,
        now=lambda: _NOW,
    )
    result = await executor.execute(record, emit=_ignore_event)

    assert result.status is SubAgentStatus.FAILED
    assert result.error is not None and result.error.code == "persistence_error"
    assert len(provider.complete_requests) == 1
    assert not provider.stream_requests
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    persisted = store.get(record.agent_id)
    assert persisted is not None and persisted.conversation[0]["content"] == record.task
    assert not persisted.context_state
    assert not (state.memory_directory / "summary.jsonl").exists()
