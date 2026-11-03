from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
import tomlkit
from aiohttp.test_utils import TestServer

from aide.agent.confirmation import ConfirmationEnvelope, SubAgentConfirmationOwner
from aide.agent.permission import ToolPermissionLevel
from aide.agent.runner import AgentRunner, AgentRunnerResult
from aide.agent.session.deletion import begin_session_deletion
from aide.agent.session.session import Session
from aide.agent.subagents.coordinator import SubAgentPool
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentEvent,
    SubAgentEventKind,
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
from aide.config.agent_home import AgentHome
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
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceCallMetadata,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures.project_removal import complete_project_removal
from tests.fixtures.service_runtime import register_project, workspace_loops
from tests.fixtures.trace import wait_for_trace_writer
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


def _add_model_routes(home: AgentHome, *names: str) -> None:
    path = home.path / "config.toml"
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    models = cast(tomlkit.items.Table, document["models"])
    routes = cast(tomlkit.items.Table, models["routes"])
    chat = cast(tomlkit.items.Table, routes["chat"])
    for name in names:
        route = tomlkit.table()
        for key, value in chat.items():
            route[key] = value
        routes[name] = route
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


def _snapshot() -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="test-provider",
        model="test-model",
        reasoning_effort="mid",
        permission_level="workspace-write",
        shell="pwsh",
        tool_names=("read_file",),
        system_prompt="You are Aide.",
    )


def _register(repository: SubAgentRecordRepository, title: str) -> SubAgentRecord:
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


def _complete(repository: SubAgentRecordRepository, record: SubAgentRecord) -> SubAgentRecord:
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
        self.child_question_started = asyncio.Event()
        self.child_answer_seen = asyncio.Event()
        self.child_question_count = 0
        self.question_children = 2
        self.independent_ninth = False
        self.question_parent_started = asyncio.Event()
        self.question_spawn_requested = False
        self.question_wait_requested = False
        self.automatic_continuation_started = asyncio.Event()
        self.automatic_continuation_count = 0
        self.automatic_write = False
        self.keep_task = False
        self.explicit_result_continuation_started = asyncio.Event()
        self.explicit_result_continuation_context = ""

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

        if "SubAgent result for automatic continuation:" in user_text:
            self.automatic_continuation_count += 1
            self.automatic_continuation_started.set()
            if self.automatic_write and not any(m.get("tool_call_id") == "auto-write" for m in current_tool_results):
                if "write_file" not in tool_names:
                    return self._events("", tool_call=ModelToolCall(
                        "auto-search-write", "tool_search", '{"query":"write file"}'))
                return self._events("", tool_call=ModelToolCall(
                    "auto-write", "write_file", '{"path":"auto-output.txt","content":"automatic bytes"}'))
            return self._events("Main automatically handled the delegated result.")

        if "## Result from delegated work" in user_text:
            self.explicit_result_continuation_context = user_text
            self.explicit_result_continuation_started.set()
            return self._events("Main used the recovered delegated result.")

        if "Continuation recovery seed" in user_text:
            return self._events("The foreground Task is ready to continue.")

        if "Direct schedule integration request" in user_text:
            self.direct_schedule_tool_names.append(tool_names)
            return self._events("Direct schedule completed.")

        if "SubAgent question delegation request" in user_text:
            self.question_parent_started.set()
            self.parent_tool_names.append(tool_names)
            spawn_results = {
                item.get("tool_call_id"): item
                for item in current_tool_results
                if item.get("tool_call_id") in {f"question-child-{index}" for index in range(1, self.question_children + 1)}
            }
            wait_result = next(
                (item for item in current_tool_results if item.get("tool_call_id") == "question-wait"),
                None,
            )
            missing_child = next(
                (index for index in range(1, self.question_children + 1) if f"question-child-{index}" not in spawn_results),
                None,
            )
            if missing_child is not None:
                if "spawn_agent" not in tool_names:
                    return self._events(
                        "",
                        tool_call=ModelToolCall(
                            "question-search-spawn",
                            "tool_search",
                            '{"query":"spawn agent"}',
                        ),
                    )
                self.question_spawn_requested = True
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                    f"question-child-{missing_child}",
                        "spawn_agent",
                    json.dumps(
                        {
                            "title": f"Child question {missing_child}",
                            "task": ("Make delegated integration answer."
                                     if self.independent_ninth and missing_child == 9
                                     else "ask-child-question"),
                        }
                    ),
                    ),
                )
            if wait_result is None:
                if "wait_agent" not in tool_names:
                    return self._events(
                        "",
                        tool_call=ModelToolCall(
                            "question-search-wait",
                            "tool_search",
                            '{"query":"wait agent"}',
                        ),
                    )
                children = [
                    json.loads(str(spawn_results[f"question-child-{index}"].get("content", "{}")))
                    for index in range(1, self.question_children + 1)
                ]
                self.question_wait_requested = True
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        "question-wait",
                        "wait_agent",
                        json.dumps({"agent_ids": [child["agent_id"] for child in children]}),
                    ),
                )
            return self._events("Main was notified that its child needs an answer.")

        if "ask-child-question" in user_text:
            self.child_question_started.set()
            self.child_question_count += 1
            self.child_tool_names = tool_names
            if "ask_user" not in tool_names:
                return self._events("The child does not have a question Tool.")
            latest_user_text = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            if latest_user_text.startswith("Human Answer:"):
                self.child_answer_seen.set()
                if latest_user_text == "Human Answer: write-answer" and not any(
                    item.get("tool_call_id") == "answer-write" for item in current_tool_results
                ):
                    return self._events("", tool_call=ModelToolCall(
                        "answer-write", "write_file", '{"path":"child-answer.txt","content":"changed"}'))
                return self._events("The child received the answer and completed the task.")
            already_asked = any(
                item.get("tool_call_id") == "child-question" for item in current_tool_results
            )
            if already_asked:
                return self._events("The child question was submitted.")
            return self._events(
                "",
                tool_call=ModelToolCall(
                    "child-question",
                    "ask_user",
                    '{"question":"Which output should I prepare?"}',
                ),
            )

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
        if self.keep_task:
            return self._response('{"action":"keep","task_goal":null,"completion_boundary":null}')
        return self._response(
            '{"action":"replace","task_goal":"answer the input",'
            '"completion_boundary":"return one answer"}'
        )

    async def close(self) -> None:
        return None


class _FirstDraftProvider(_DelegatingProvider):
    def __init__(self, *, wait_for_child: bool = False) -> None:
        super().__init__()
        self.spawn_count = 0
        self.started: asyncio.Queue[None] = asyncio.Queue()
        self.parent_ready = asyncio.Event()
        self.allow_parent = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.wait_for_child = wait_for_child
        self.wait_emitted = asyncio.Event()

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
        user_text = "\n".join(str(message.get("content", "")) for message in messages)
        tool_names = {tool["function"]["name"] for tool in tools}
        if "first draft delegation request" in user_text:
            if "spawn_agent" not in tool_names:
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        "first-search-spawn",
                        "tool_search",
                        '{"query":"spawn agent wait agent"}',
                    ),
                )
            if self.spawn_count < 9:
                index = self.spawn_count
                self.spawn_count += 1
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        f"first-spawn-{index}",
                        "spawn_agent",
                        json.dumps(
                            {
                                "title": f"First draft child {index}",
                                "task": "Hold the delegated child.",
                            }
                        ),
                    ),
                )
            if self.wait_requested:
                self.wait_result_contents = [
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "tool" and message.get("tool_call_id") == "first-wait"
                ]
                return self._events("First main Run completed after waiting.")

            async def finish_parent() -> AsyncIterator[ModelStreamEvent]:
                self.parent_ready.set()
                await self.allow_parent.wait()
                if self.wait_for_child:
                    self.wait_requested = True
                    self.wait_emitted.set()
                    yield ModelCompleted(
                        self._response(
                            "",
                            tool_call=ModelToolCall(
                                "first-wait",
                                "wait_agent",
                                json.dumps({"agent_ids": [self.agent_id]}),
                            ),
                        )
                    )
                else:
                    yield ModelCompleted(self._response("First main Run completed."))

            return finish_parent()

        if not any(message.get("tool_call_id") == "first-child-search" for message in messages):
            return self._events(
                "Persisted first child output.",
                tool_call=ModelToolCall(
                    "first-child-search",
                    "tool_search",
                    '{"query":"list agents"}',
                ),
            )

        async def hold_child() -> AsyncIterator[ModelStreamEvent]:
            yield TextDelta("Persisted first child output.")
            self.started.put_nowait(None)
            try:
                await self.release_child.wait()
            except asyncio.CancelledError:
                self.cleanup_started.set()
                await self.allow_cleanup.wait()
                raise
            yield ModelCompleted(self._response("First child finished."))

        return hold_child()


class _BatchDelegatingProvider(_DelegatingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.batch_parent_waiting = asyncio.Event()
        self.release_batch_parent = asyncio.Event()
        self.batch_auto_requests: list[str] = []
        self.hold_batch_children: set[int] = set()
        self.batch_child_waiting = {index: asyncio.Event() for index in range(1, 4)}
        self.release_batch_children = {index: asyncio.Event() for index in range(1, 4)}
        self.hold_automatic = False
        self.release_automatic = asyncio.Event()

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
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") in {"user", "tool"}
        )
        child_prompt = next(
            (
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user"
                and str(message.get("content", "")).startswith("Make batch child ")
            ),
            None,
        )
        if child_prompt is not None:
            result_number = child_prompt.split()[3].rstrip(".")
            index = int(result_number)

            async def finish_child() -> AsyncIterator[ModelStreamEvent]:
                self.batch_child_waiting[index].set()
                if index in self.hold_batch_children:
                    await self.release_batch_children[index].wait()
                yield ModelCompleted(self._response(f"Batch child result {result_number}."))

            return finish_child()

        tool_names = {
            name
            for tool in tools
            if isinstance(tool.get("function"), dict)
            and isinstance((name := tool["function"].get("name")), str)
        }
        if "batch delegation request" in user_text:
            completed = {
                item.get("tool_call_id")
                for item in messages
                if item.get("role") == "tool"
            }
            pending = [
                index for index in range(1, 4) if f"batch-child-{index}" not in completed
            ]
            if pending and "spawn_agent" in tool_names:
                self.spawn_requested = True
                calls = tuple(
                    ModelToolCall(
                        f"batch-child-{index}",
                        "spawn_agent",
                        json.dumps(
                            {
                                "title": f"Batch child {index}",
                                "task": f"Make batch child {index}.",
                            }
                        ),
                    )
                    for index in pending
                )

                async def emit_children() -> AsyncIterator[ModelStreamEvent]:
                    yield ModelCompleted(
                        ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    )

                return emit_children()
            if not pending and not self.batch_parent_waiting.is_set():

                async def finish_parent() -> AsyncIterator[ModelStreamEvent]:
                    self.batch_parent_waiting.set()
                    await self.release_batch_parent.wait()
                    yield ModelCompleted(self._response("Main run completed."))

                return finish_parent()

        if "## Runtime Event: SubAgent Result" in user_text:
            self.batch_auto_requests.append(user_text)
            response = super().stream(
                messages=messages,
                tools=tools,
                model="test-model",
                max_output=1,
                temperature=0,
                reasoning_effort=None,
                timeout=1,
            )
            if self.hold_automatic:

                async def finish_automatic() -> AsyncIterator[ModelStreamEvent]:
                    await self.release_automatic.wait()
                    async for event in response:
                        yield event

                return finish_automatic()
            return response
        return super().stream(
            messages=messages,
            tools=tools,
            model="test-model",
            max_output=1,
            temperature=0,
            reasoning_effort=None,
            timeout=1,
        )


@pytest_asyncio.fixture
async def management_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AgentService, WorkspaceRecord, str, str, str]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    _add_model_routes(home, "schedule", "subagent")
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
        "aide.service.runtime.service.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load())
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
@pytest.mark.parametrize("scope", ["chat", "project"])
async def test_first_draft_run_allows_subagent_queries_and_cancellation_before_persist(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    scope: Literal["chat", "project"],
) -> None:
    service, workspace, client_id, _session_id, other_session_id = management_case
    if scope == "project":
        await register_project(service, client_id, workspace.workspace_path)
    provider = _FirstDraftProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    session_id = await workspace.create_draft(
        client_id,
        reuse_startup_session=False,
        creation_scope=scope,
    )
    claim_payload = cast(
        dict[str, Any],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim = workspace.require_claim(client_id, session_id, claim_payload["claim_version"])
    session = claim.loop.session
    path = workspace.workspace_state.sessions_directory / f"{session_id}.jsonl"
    sink = cast(_CollectingSink, service.client(client_id).sink)
    try:
        await workspace.input(
            client_id,
            session_id,
            claim.version,
            "First input: first draft delegation request.",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.parent_ready.wait(), timeout=20)
        for _ in range(8):
            await asyncio.wait_for(provider.started.get(), timeout=20)
        assert not path.exists()
        original_metadata = dict(session.metadata)
        original_messages = list(session.messages)
        page = service.list_subagents(client_id, workspace.workspace_id, session_id)
        items = cast(list[dict[str, Any]], page["items"])
        assert len(items) == 9
        running = next(item for item in items if item["title"] == "First draft child 0")
        queued = next(item for item in items if item["title"] == "First draft child 8")
        detail = service.get_subagent(
            client_id, workspace.workspace_id, session_id, running["agent_id"]
        )
        assert detail["status"] == "running"
        assert "Persisted first child output." in json.dumps(detail["conversation"])
        assert queued["status"] == "queued"
        assert not path.exists()
        assert session.metadata == original_metadata
        assert session.messages == original_messages
        queued_result = await service.cancel_subagent(
            client_id,
            workspace.workspace_id,
            session_id,
            queued["agent_id"],
        )
        assert queued_result["cancelled"] is True
        cancellation = asyncio.create_task(
            service.cancel_subagent(
                client_id,
                workspace.workspace_id,
                session_id,
                running["agent_id"],
            )
        )
        await asyncio.wait_for(provider.cleanup_started.wait(), timeout=3)
        assert not cancellation.done()
        assert claim.loop.has_active_run
        provider.allow_cleanup.set()
        assert (await cancellation)["cancelled"] is True
        assert not path.exists()
        assert provider.started.empty(), "The queued child must never start"
        untouched = service.list_subagents(client_id, workspace.workspace_id, session_id)
        assert (
            sum(
                item["status"] == "running"
                for item in cast(list[dict[str, Any]], untouched["items"])
            )
            == 7
        )
        for target_session in (session_id, other_session_id):
            with pytest.raises(ServiceError) as missing:
                service.get_subagent(
                    client_id,
                    workspace.workspace_id,
                    target_session,
                    str(uuid4()) if target_session == session_id else running["agent_id"],
                )
            assert missing.value.status == 404
        provider.allow_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)
        await claim.loop.wait_for_restore_idle()
        output_task = workspace_loops(workspace)[session_id].output_task
        if output_task is not None:
            await asyncio.wait_for(output_task, timeout=5)
        await workspace.release(client_id, session_id)
        assert (
            service.get_subagent(
                client_id, workspace.workspace_id, session_id, running["agent_id"]
            )["status"]
            == "cancelled"
        )
        assert (
            await service.cancel_subagent(
                client_id, workspace.workspace_id, session_id, running["agent_id"]
            )
        )["cancelled"] is True
        survivor = next(item for item in items if item["title"] == "First draft child 1")
        assert (
            await service.cancel_subagent(
                client_id, workspace.workspace_id, session_id, survivor["agent_id"]
            )
        )["cancelled"] is True
    finally:
        provider.allow_cleanup.set()
        provider.allow_parent.set()
        provider.release_child.set()


@pytest.mark.asyncio
async def test_user_cancellation_releases_wait_agent_in_the_first_draft_run(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, _session_id, _other_session_id = management_case
    provider = _FirstDraftProvider(wait_for_child=True)
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    session_id = await workspace.create_draft(
        client_id, reuse_startup_session=False, creation_scope="chat"
    )
    claim_payload = cast(
        dict[str, Any],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, service.client(client_id).sink)
    cancellation: asyncio.Task[dict[str, object]] | None = None
    try:
        await workspace.input(
            client_id,
            session_id,
            claim_payload["claim_version"],
            "first draft delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.parent_ready.wait(), timeout=20)
        for _ in range(8):
            await asyncio.wait_for(provider.started.get(), timeout=20)
        items = cast(
            list[dict[str, Any]],
            service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
        )
        provider.agent_id = next(
            item["agent_id"] for item in items if item["title"] == "First draft child 0"
        )
        provider.allow_parent.set()
        await asyncio.wait_for(provider.wait_emitted.wait(), timeout=5)
        await asyncio.sleep(0)
        path = workspace.workspace_state.sessions_directory / f"{session_id}.jsonl"
        assert not path.exists()
        cancellation = asyncio.create_task(
            service.cancel_subagent(
                client_id, workspace.workspace_id, session_id, cast(str, provider.agent_id)
            )
        )
        await asyncio.wait_for(provider.cleanup_started.wait(), timeout=3)
        assert not cancellation.done()
        assert not any(
            event["type"] == "run.completed" and event["run_id"] == _RUN_ID for event in sink.events
        )
        provider.allow_cleanup.set()
        assert (await cancellation)["cancelled"] is True
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)
        assert provider.wait_result_contents
        assert "cancelled" in provider.wait_result_contents[0]
        assert cast(str, provider.agent_id) in provider.wait_result_contents[0]
    finally:
        provider.allow_cleanup.set()
        provider.allow_parent.set()
        provider.release_child.set()
        if cancellation is not None:
            await asyncio.gather(cancellation, return_exceptions=True)


@pytest.mark.asyncio
async def test_subagent_management_requires_session_authority_and_preserves_deletion_guard(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, _session_id, _other_session_id = management_case
    orphan = Session.create(workspace.workspace_state)
    repository = SubAgentRecordStore(workspace.workspace_state, orphan.session_id)
    pool = SubAgentPool(repository, _HoldingExecutor())
    workspace.register_subagent_coordinator(pool, repository)
    record = _register(repository, "An orphan record cannot establish Session authority")
    draft = await workspace.create_draft(
        client_id, reuse_startup_session=False, creation_scope="chat"
    )
    begin_session_deletion(workspace.workspace_state, draft)
    for target, code, status in (
        (orphan.session_id, "not_found", 404),
        (Session.create(workspace.workspace_state).session_id, "not_found", 404),
        ("invalid/session", "validation_error", 422),
        (draft, "session_deleting", 409),
    ):
        for action in ("list", "get", "cancel"):
            with pytest.raises(ServiceError) as failure:
                if action == "list":
                    service.list_subagents(client_id, workspace.workspace_id, target)
                elif action == "get":
                    service.get_subagent(client_id, workspace.workspace_id, target, record.agent_id)
                else:
                    await service.cancel_subagent(
                        client_id, workspace.workspace_id, target, record.agent_id
                    )
            assert (failure.value.code, failure.value.status) == (code, status)


@pytest.mark.asyncio
@pytest.mark.parametrize("partition", ["foreground", "schedule"])
async def test_historical_subagent_session_read_failure_still_reports_persistence_error(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    directory = workspace.workspace_state.sessions_directory
    if partition == "schedule":
        session = Session.create_schedule(workspace.workspace_state, str(uuid4()))
        session.commit_agent_run(
            [{"role": "user", "content": "Schedule history"}],
            pending_last_compacted=0,
            pending_action_summary=None,
        )
        await session.wait_for_pending_persist()
        session_id = session.session_id
        directory = workspace.workspace_state.schedule_sessions_directory
    history_path = directory / f"{session_id}.jsonl"
    read_file = HOST_FILESYSTEM.require_owned_regular_file

    def fail_history_read(path: Path, *, within: Path) -> Path:
        if path == history_path:
            raise OSError("History cannot be read")
        return read_file(path, within=within)

    monkeypatch.setattr(HOST_FILESYSTEM, "require_owned_regular_file", fail_history_read)
    for action in ("list", "get", "cancel"):
        with pytest.raises(ServiceError) as failure:
            if action == "list":
                service.list_subagents(client_id, workspace.workspace_id, session_id)
            elif action == "get":
                service.get_subagent(client_id, workspace.workspace_id, session_id, str(uuid4()))
            else:
                await service.cancel_subagent(
                    client_id, workspace.workspace_id, session_id, str(uuid4())
                )
        assert (failure.value.code, failure.value.status) == ("persistence_error", 500)


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


async def _wait_for_automatic_completion(
    sink: _CollectingSink,
    *,
    excluded_run_id: str,
) -> dict[str, object]:
    while True:
        sink.changed.clear()
        started = next(
            (
                event
                for event in sink.events
                if event.get("type") == "run.started"
                and isinstance(event.get("run_id"), str)
                and event["run_id"] != excluded_run_id
                and cast(dict[str, object], event.get("payload", {})).get("source") == "automatic"
            ),
            None,
        )
        if started is not None:
            return await sink.wait_for("run.completed", cast(str, started["run_id"]))
        await asyncio.wait_for(sink.changed.wait(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["none", "partial_ledger", "session_before", "session_after"])
async def test_same_task_terminal_results_are_combined_in_one_automatic_main_request(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _BatchDelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim["claim_version"])
    sink = cast(_CollectingSink, service.client(client_id).sink)

    try:
        await workspace.input(
            client_id,
            session_id,
            claim_version,
            "initial delegation request for batch delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                items = cast(
                    list[dict[str, object]],
                    service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
                )
                if len(items) == 3 and all(
                    item["status"] in {"completed", "failed"} for item in items
                ):
                    break
                await sink.changed.wait()

        repository = workspace.subagent_repository(session_id)
        terminal_records = [
            _require_record(repository, cast(str, item["agent_id"])) for item in items
        ]
        await asyncio.gather(
            *(
                workspace.publish(
                    SubAgentEvent(
                        kind=SubAgentEventKind.STATUS,
                        workspace_id=workspace.workspace_id,
                        session_id=session_id,
                        agent_id=record.agent_id,
                        revision=record.revision + duplicate + 1_000,
                        occurred_at=_NOW,
                        data={"status": record.status.value},
                    )
                )
                for record in terminal_records
                for duplicate in range(4)
            )
        )

        provider.keep_task = True
        original_text = HOST_FILESYSTEM.atomic_replace_text
        original_bytes = HOST_FILESYSTEM.atomic_replace_bytes
        injected = asyncio.Event()

        def fail_partial_ledger(path: Path, content: str) -> None:
            if fault == "partial_ledger" and path.name == "state.json" and not injected.is_set():
                value = json.loads(content)
                deliveries = value.get("deliveries", [])
                if sum(item["state"] == "committed" for item in deliveries) == 2:
                    injected.set()
                    raise OSError("Injected second receipt confirmation failure")
            original_text(path, content)

        def fail_batch_session(path: Path, content: bytes) -> None:
            if (fault in {"session_before", "session_after"} and path.suffix == ".jsonl"
                    and b"Main automatically handled the delegated result." in content):
                injected.set()
                if fault == "session_after":
                    original_bytes(path, content)
                raise OSError("Injected batch Session persistence failure")
            original_bytes(path, content)

        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_partial_ledger)
        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_batch_session)

        async def verify_restart(expected_state: str) -> None:
            expected_keys = {
                (item.agent_id, item.terminal_record_revision, item.task_generation, item.receipt_id)
                for item in workspace.foreground_work_store(session_id).load().deliveries
            }
            await service.stop()
            monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", original_text)
            monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", original_bytes)
            recovered = AgentService(service.agent_home, ConfigLoader(service.agent_home).load())
            await recovered.start()
            recovered.model_router._provider_factory = lambda _: provider
            recovered.model_router._providers.clear()
            try:
                client = await recovered.register_client("cli")
                await recovered.connect_client(client.client_id, _CollectingSink())
                resident = await recovered.attach_workspace(client.client_id, workspace.workspace_path)
                await recovered.claim(client.client_id, resident.workspace_id, session_id)
                resident.session_snapshot(session_id)
                recovered_deliveries = resident.foreground_work_store(session_id).load().deliveries
                assert {
                    (item.agent_id, item.terminal_record_revision, item.task_generation, item.receipt_id)
                    for item in recovered_deliveries
                } == expected_keys
                assert all(item.state == expected_state for item in recovered_deliveries)
                assert provider.automatic_continuation_count == 1
                assert len(provider.batch_auto_requests) == 1
            finally:
                await recovered.stop()

        queued_runs = [str(uuid4()), str(uuid4())]
        for run_id, text in zip(
            queued_runs, ("first queued supplement", "second queued supplement"), strict=True
        ):
            await workspace.input(client_id, session_id, claim_version, text, run_id)
        provider.release_batch_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=10)
        for run_id in queued_runs:
            await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=10)
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
        await asyncio.wait_for(
            _wait_for_automatic_completion(sink, excluded_run_id=_RUN_ID), timeout=10
        )

        assert len(provider.batch_auto_requests) == 1
        if fault != "none":
            assert injected.is_set()
        if fault == "session_before":
            work = workspace.foreground_work_store(session_id).load()
            assert len(work.deliveries) == 3
            assert all(item.state == "claimed" for item in work.deliveries)
            assert work.tasks[-1].state == "interrupted"
            assert len({item.receipt_id for item in work.deliveries}) == 3
            assert not any(message.get("foreground_delivery_receipts") for message in
                           Session.load(workspace.workspace_state, session_id).messages)
            assert provider.automatic_continuation_count == 1
            await verify_restart("claimed")
            return
        merged_request = provider.batch_auto_requests[0]
        assert all(f"Batch child result {index}." in merged_request for index in range(1, 4))
        ordered_runs = [_RUN_ID, *queued_runs]
        started = [
            event
            for event in sink.events
            if event.get("type") == "run.started" and event.get("run_id") in ordered_runs
        ]
        assert [event["run_id"] for event in started] == ordered_runs
        for previous, following in pairwise(ordered_runs):
            previous_completion = next(
                index
                for index, event in enumerate(sink.events)
                if event.get("type") == "run.completed" and event.get("run_id") == previous
            )
            following_start = next(
                index
                for index, event in enumerate(sink.events)
                if event.get("type") == "run.started" and event.get("run_id") == following
            )
            assert previous_completion < following_start
        automatic_start = next(
            event
            for event in sink.events
            if event.get("type") == "run.started"
            and cast(dict[str, object], event.get("payload", {})).get("source") == "automatic"
        )
        user_completion_indexes = [
            next(
                index
                for index, event in enumerate(sink.events)
                if event.get("type") == "run.completed" and event.get("run_id") == run_id
            )
            for run_id in ordered_runs
        ]
        assert max(user_completion_indexes) < sink.events.index(automatic_start)
        assert provider.automatic_continuation_count == 1
        automatic_run_id = cast(str, automatic_start["run_id"])
        main_run_ids = {*ordered_runs, automatic_run_id}
        active_main_runs: set[str] = set()
        max_main_concurrency = 0
        for event in sink.events:
            event_run_id = event.get("run_id")
            if not isinstance(event_run_id, str) or event_run_id not in main_run_ids:
                continue
            if event.get("type") == "run.started":
                active_main_runs.add(event_run_id)
                max_main_concurrency = max(max_main_concurrency, len(active_main_runs))
            elif event.get("type") == "run.completed":
                active_main_runs.discard(event_run_id)
        assert max_main_concurrency == 1
        snapshot = workspace.session_snapshot(session_id)
        messages = cast(list[dict[str, object]], snapshot["messages"])
        user_messages = [message for message in messages if message.get("role") == "user"]
        assert [message.get("content") for message in user_messages] == [
            "Saved Session history.",
            "initial delegation request for batch delegation request",
            "first queued supplement",
            "second queued supplement",
        ]
        assert not any(
            message.get("role") == "user" and "SubAgent result" in str(message.get("content", ""))
            for message in messages
        )
        automatic_responses = [
            message
            for message in messages
            if message.get("role") == "assistant"
            and message.get("content") == "Main automatically handled the delegated result."
        ]
        assert len(automatic_responses) == 1
        assert automatic_responses[0]["token_usage"] == {
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }
        deliveries = workspace.foreground_work_store(session_id).load().deliveries
        assert len(deliveries) == 3
        assert all(delivery.state == "committed" for delivery in deliveries)
        persisted = Session.load(workspace.workspace_state, session_id).messages
        receipts = [receipt for message in persisted
                    for receipt in message.get("foreground_delivery_receipts", [])]
        assert len(receipts) == 3 and set(receipts) == {item.receipt_id for item in deliveries}
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        trace = [record for record in (await recorder.list_records()).items
                 if record.run_id == automatic_run_id and record.kind is TraceKind.MODEL]
        assert len(trace) == 1
        assert trace[0].origin is TraceOrigin.FOREGROUND
        assert trace[0].owner_session_id == session_id
        assert trace[0].usage.input_tokens == 1 and trace[0].usage.output_tokens == 1
        await verify_restart("committed")
    finally:
        provider.release_batch_parent.set()


@pytest.mark.asyncio
async def test_recalling_queued_inputs_does_not_recall_automatic_results(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _BatchDelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim["claim_version"])
    sink = cast(_CollectingSink, service.client(client_id).sink)

    try:
        await workspace.input(
            client_id,
            session_id,
            claim_version,
            "initial delegation request for batch delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                items = cast(
                    list[dict[str, object]],
                    service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
                )
                if len(items) == 3 and all(
                    item["status"] in {"completed", "failed"} for item in items
                ):
                    break
                await sink.changed.wait()

        queued = [(str(uuid4()), "recall first input"), (str(uuid4()), "recall second input")]
        for run_id, text in queued:
            await workspace.input(client_id, session_id, claim_version, text, run_id)
        recalled = await service.recall_queued_inputs(
            client_id, workspace.workspace_id, session_id, claim_version
        )
        assert recalled["recalled_inputs"] == [
            {"run_id": run_id, "text": text} for run_id, text in queued
        ]

        provider.release_batch_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=10)
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
        await asyncio.wait_for(
            _wait_for_automatic_completion(sink, excluded_run_id=_RUN_ID), timeout=10
        )
        assert provider.automatic_continuation_count == 1
        assert len(provider.batch_auto_requests) == 1
        assert all(
            f"Batch child result {index}." in provider.batch_auto_requests[0]
            for index in range(1, 4)
        )
        assert not any(
            event.get("type") == "run.started" and event.get("run_id") == run_id
            for run_id, _text in queued
            for event in sink.events
        )
    finally:
        provider.release_batch_parent.set()


@pytest.mark.asyncio
async def test_results_arriving_during_automatic_run_wait_for_next_user_first_arbitration(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _BatchDelegatingProvider()
    provider.hold_batch_children = {2, 3}
    provider.hold_automatic = True
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, service.client(client_id).sink)
    queued_run_id = str(uuid4())

    async def wait_for_child_statuses(expected: dict[str, set[str]]) -> list[dict[str, object]]:
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                items = cast(
                    list[dict[str, object]],
                    service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
                )
                statuses = {
                    cast(str, item["title"]): cast(str, item["status"])
                    for item in items
                }
                if all(statuses.get(title) in values for title, values in expected.items()):
                    return items
                await sink.changed.wait()

    async def wait_for_automatic_starts(count: int) -> list[dict[str, object]]:
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                starts = [
                    event
                    for event in sink.events
                    if event.get("type") == "run.started"
                    and cast(dict[str, object], event.get("payload", {})).get("source") == "automatic"
                ]
                if len(starts) >= count:
                    return starts
                await sink.changed.wait()

    try:
        await workspace.input(
            client_id,
            session_id,
            cast(int, claim["claim_version"]),
            "initial delegation request for batch delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        await asyncio.gather(*(event.wait() for event in provider.batch_child_waiting.values()))
        await wait_for_child_statuses(
            {
                "Batch child 1": {"completed", "failed"},
                "Batch child 2": {"running"},
                "Batch child 3": {"running"},
            }
        )

        provider.release_batch_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=10)
        first_auto = await wait_for_automatic_starts(1)
        first_auto_id = cast(str, first_auto[0]["run_id"])
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
        assert provider.automatic_continuation_count == 1
        assert len(provider.batch_auto_requests) == 1
        assert "Batch child result 1." in provider.batch_auto_requests[0]
        assert "Batch child result 2." not in provider.batch_auto_requests[0]
        assert "Batch child result 3." not in provider.batch_auto_requests[0]

        provider.keep_task = True
        await workspace.input(
            client_id,
            session_id,
            cast(int, claim["claim_version"]),
            "input queued during automatic execution",
            queued_run_id,
        )
        provider.release_batch_children[2].set()
        provider.release_batch_children[3].set()
        await wait_for_child_statuses(
            {
                "Batch child 2": {"completed", "failed"},
                "Batch child 3": {"completed", "failed"},
            }
        )
        assert provider.automatic_continuation_count == 1
        assert not any(
            event.get("type") == "run.started" and event.get("run_id") == queued_run_id
            for event in sink.events
        )

        provider.release_automatic.set()
        automatic_starts = await wait_for_automatic_starts(2)
        second_auto_id = cast(str, automatic_starts[1]["run_id"])
        await asyncio.wait_for(sink.wait_for("run.completed", first_auto_id), timeout=10)
        await asyncio.wait_for(sink.wait_for("run.completed", queued_run_id), timeout=10)
        await asyncio.wait_for(sink.wait_for("run.completed", second_auto_id), timeout=10)

        first_auto_completion = next(
            index
            for index, event in enumerate(sink.events)
            if event.get("type") == "run.completed" and event.get("run_id") == first_auto_id
        )
        user_start = next(
            index
            for index, event in enumerate(sink.events)
            if event.get("type") == "run.started" and event.get("run_id") == queued_run_id
        )
        user_completion = next(
            index
            for index, event in enumerate(sink.events)
            if event.get("type") == "run.completed" and event.get("run_id") == queued_run_id
        )
        second_auto_start = next(
            index
            for index, event in enumerate(sink.events)
            if event.get("type") == "run.started" and event.get("run_id") == second_auto_id
        )
        assert first_auto_completion < user_start < user_completion < second_auto_start
        assert provider.automatic_continuation_count == 2
        assert len(provider.batch_auto_requests) == 2
        assert "Batch child result 2." in provider.batch_auto_requests[1]
        assert "Batch child result 3." in provider.batch_auto_requests[1]
        assert "Batch child result 1." not in provider.batch_auto_requests[1]
        assert all(
            item.state == "committed"
            for item in workspace.foreground_work_store(session_id).load().deliveries
        )
    finally:
        provider.release_batch_parent.set()
        provider.release_automatic.set()
        for event in provider.release_batch_children.values():
            event.set()


@pytest.mark.asyncio
async def test_batch_claim_failure_preserves_each_result_for_reconciliation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _BatchDelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, service.client(client_id).sink)
    injected = asyncio.Event()
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    def fail_batch_claim(path: Path, content: str) -> None:
        if path.name == "state.json" and not injected.is_set():
            value = json.loads(content)
            deliveries = value.get("deliveries", []) if isinstance(value, dict) else []
            if len(deliveries) == 3 and all(item.get("state") == "claimed" for item in deliveries):
                injected.set()
                raise OSError("Injected atomic batch claim failure")
        original_replace(path, content)

    try:
        await workspace.input(
            client_id,
            session_id,
            cast(int, claim["claim_version"]),
            "initial delegation request for batch delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                items = cast(
                    list[dict[str, object]],
                    service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
                )
                if len(items) == 3 and all(
                    item["status"] in {"completed", "failed"} for item in items
                ):
                    break
                await sink.changed.wait()

        terminal_records = [
            _require_record(
                workspace.subagent_repository(session_id), cast(str, item["agent_id"])
            )
            for item in items
        ]
        expected_keys = {
            (record.agent_id, record.revision, cast(int, record.source.task_generation))
            for record in terminal_records
        }
        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_batch_claim)
        provider.release_batch_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=10)
        await asyncio.wait_for(injected.wait(), timeout=10)
        assert provider.automatic_continuation_count == 0
        pending = workspace.foreground_work_store(session_id).load().deliveries
        assert {
            (item.agent_id, item.terminal_record_revision, item.task_generation)
            for item in pending
        } == expected_keys
        assert all(item.state == "pending" for item in pending)

        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", original_replace)
        record = terminal_records[0]
        await workspace.publish(
            SubAgentEvent(
                kind=SubAgentEventKind.STATUS,
                workspace_id=workspace.workspace_id,
                session_id=session_id,
                agent_id=record.agent_id,
                revision=record.revision + 1_000,
                occurred_at=_NOW,
                data={"status": record.status.value},
            )
        )
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
        await asyncio.wait_for(
            _wait_for_automatic_completion(sink, excluded_run_id=_RUN_ID), timeout=10
        )
        workspace.session_snapshot(session_id)
        recovered = workspace.foreground_work_store(session_id).load().deliveries
        assert {
            (item.agent_id, item.terminal_record_revision, item.task_generation)
            for item in recovered
        } == expected_keys
        assert all(item.state == "committed" for item in recovered)
        assert provider.automatic_continuation_count == 1
    finally:
        provider.release_batch_parent.set()


@pytest.mark.asyncio
async def test_unanswered_task_results_stay_gated_while_independent_task_continues(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _BatchDelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, service.client(client_id).sink)
    store = workspace.foreground_work_store(session_id)
    repository = workspace.subagent_repository(session_id)

    try:
        await workspace.input(
            client_id,
            session_id,
            cast(int, claim["claim_version"]),
            "initial delegation request for batch delegation request",
            _RUN_ID,
        )
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        async with asyncio.timeout(10):
            while True:
                sink.changed.clear()
                children = cast(
                    list[dict[str, object]],
                    service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
                )
                if len(children) == 3 and all(
                    item["status"] in {"completed", "failed"} for item in children
                ):
                    break
                await sink.changed.wait()

        primary_task = store.load().tasks[0]
        owner_agent_id = cast(str, children[0]["agent_id"])
        question_id = str(uuid4())
        question_call_id = str(uuid4())
        store.prepare_question(
            task_id=primary_task.task_id,
            execution_id=primary_task.execution_id,
            question_id=question_id,
            tool_call_id=question_call_id,
            text="Which result should the Main Agent use?",
            owner_agent_id=owner_agent_id,
        )
        store.publish_question(question_id, committed_tool_call_ids={question_call_id})

        persisted_messages = Session.load(workspace.workspace_state, session_id).messages
        anchor = next(
            message
            for message in reversed(persisted_messages)
            if message.get("role") == "user"
            and isinstance(message.get("restore_anchor_id"), int)
            and isinstance(message.get("restore_run_token"), str)
        )
        independent_task_id = str(uuid4())
        independent_execution_id = str(uuid4())
        store.begin_task(
            independent_task_id,
            independent_execution_id,
            generation=1,
            origin_anchor_id=cast(int, anchor["restore_anchor_id"]),
            permission_level="workspace-write",
        )
        store.complete_task(independent_task_id, independent_execution_id)
        independent_record = repository.register(
            title="Independent task result",
            task="Finish independent work.",
            parent_run_id=independent_execution_id,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=cast(str, anchor["restore_run_token"]),
                task_id=independent_task_id,
                task_generation=1,
            ),
            creator_snapshot=_snapshot(),
        )
        independent_record = _complete(repository, independent_record)
        stale_record = repository.register(
            title="Stale generation result",
            task="Finish obsolete generation.",
            parent_run_id=independent_execution_id,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=cast(str, anchor["restore_run_token"]),
                task_id=independent_task_id,
                task_generation=2,
            ),
            creator_snapshot=_snapshot(),
        )
        stale_record = _complete(repository, stale_record)
        await workspace.publish(
            SubAgentEvent(
                kind=SubAgentEventKind.STATUS,
                workspace_id=workspace.workspace_id,
                session_id=session_id,
                agent_id=independent_record.agent_id,
                revision=independent_record.revision,
                occurred_at=_NOW,
                data={"status": "completed"},
            )
        )
        await workspace.publish(
            SubAgentEvent(
                kind=SubAgentEventKind.STATUS,
                workspace_id=workspace.workspace_id,
                session_id=session_id,
                agent_id=stale_record.agent_id,
                revision=stale_record.revision,
                occurred_at=_NOW,
                data={"status": "completed"},
            )
        )

        provider.release_batch_parent.set()
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=10)
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
        automatic = await asyncio.wait_for(
            _wait_for_automatic_completion(sink, excluded_run_id=_RUN_ID), timeout=10
        )

        assert len(provider.batch_auto_requests) == 1
        independent_request = provider.batch_auto_requests[0]
        assert "Independent task result" in independent_request
        assert "Batch child result" not in independent_request
        assert "Stale generation result" not in independent_request
        deliveries = store.load().deliveries
        primary_deliveries = [item for item in deliveries if item.task_id == primary_task.task_id]
        independent_delivery = next(
            item for item in deliveries if item.task_id == independent_task_id
        )
        assert len(primary_deliveries) == 3
        assert all(item.state == "pending" for item in primary_deliveries)
        assert independent_delivery.state == "committed"
        stale_delivery = next(item for item in deliveries if item.agent_id == stale_record.agent_id)
        assert stale_delivery.state == "invalidated"
        assert automatic["run_id"] != _RUN_ID
    finally:
        provider.release_batch_parent.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("successor_kind", ["cli", "web"])
async def test_foreground_result_continues_after_session_claim_is_released(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    successor_kind: Literal["cli", "web"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim_response = await service.claim(client_id, workspace.workspace_id, session_id)
    claim = cast(dict[str, object], claim_response["claim"])
    sink = cast(_CollectingSink, service.client(client_id).sink)

    automatic_completed = asyncio.Event()
    emit = service.emit

    async def observe(event_type: str, **kwargs: Any) -> None:
        await emit(event_type, **kwargs)
        if event_type == "run.completed" and kwargs.get("run_id") != _RUN_ID:
            automatic_completed.set()

    monkeypatch.setattr(service, "emit", observe)
    await workspace.input(
        client_id,
        session_id,
        cast(int, claim["claim_version"]),
        "Please make an initial delegation request.",
        _RUN_ID,
    )
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)
    await workspace.release(client_id, session_id)

    successor = await service.register_client(successor_kind)
    await service.connect_client(successor.client_id, _CollectingSink())
    provider.release_child.set()
    await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=12)

    async with asyncio.timeout(12):
        while True:
            sink.changed.clear()
            snapshot = workspace.session_snapshot(session_id)
            messages = cast(list[dict[str, object]], snapshot["messages"])
            if any(
                message.get("content") == "Main automatically handled the delegated result."
                for message in messages
            ):
                break
            await asyncio.sleep(0.01)

    task = cast(list[dict[str, object]], snapshot["foreground_tasks"])[0]
    for operation in (service.continue_foreground_task, service.cancel_foreground_task):
        with pytest.raises(ServiceError) as stale_claim:
            await operation(
                client_id,
                workspace.workspace_id,
                session_id,
                cast(int, claim["claim_version"]),
                cast(str, task["task_id"]),
                cast(int, task["revision"]),
                str(uuid4()),
            )
        assert stale_claim.value.code == "stale_claim"

    assert sum(
        message.get("content") == "Main automatically handled the delegated result."
        for message in messages
    ) == 1
    await asyncio.wait_for(automatic_completed.wait(), timeout=5)
    claimed = await service.claim(successor.client_id, workspace.workspace_id, session_id)
    assert cast(dict[str, object], claimed["claim"])["session_id"] == session_id
    claimed_snapshot = cast(dict[str, object], claimed["snapshot"])
    assert cast(dict[str, object], claimed_snapshot["live_state"])["runs"] == []
    claimed_messages = cast(list[dict[str, object]], claimed_snapshot["messages"])
    assert sum(
        message.get("content") == "Main automatically handled the delegated result."
        for message in claimed_messages
    ) == 1


@pytest.mark.asyncio
async def test_expired_client_does_not_destroy_foreground_work_while_service_is_online(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    original_sink = cast(_CollectingSink, service.client(client_id).sink)
    await workspace.input(
        client_id,
        session_id,
        cast(int, claim["claim_version"]),
        "Please make an initial delegation request.",
        _RUN_ID,
    )
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await asyncio.wait_for(original_sink.wait_for("run.completed", _RUN_ID), timeout=5)

    successor = await service.register_client("cli")
    successor_sink = _CollectingSink()
    await service.connect_client(successor.client_id, successor_sink)
    await service.attach_workspace(successor.client_id, workspace.workspace_path)
    expiry_waiting = asyncio.Event()
    allow_expiry = asyncio.Event()

    async def wait_for_expiry(_deadline: float) -> None:
        expiry_waiting.set()
        await allow_expiry.wait()

    monkeypatch.setattr(service, "_wait_until", wait_for_expiry)
    await service.disconnect_client(client_id, sink=original_sink)
    expired = service.client(client_id).disconnect_task
    assert expired is not None
    await asyncio.wait_for(expiry_waiting.wait(), timeout=5)
    allow_expiry.set()
    await asyncio.wait_for(expired, timeout=5)

    subagents = cast(
        list[dict[str, object]],
        service.list_subagents(successor.client_id, workspace.workspace_id, session_id)["items"],
    )
    assert subagents[0]["status"] == "running"
    provider.release_child.set()
    await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=12)
    async with asyncio.timeout(12):
        while True:
            successor_sink.changed.clear()
            snapshot = workspace.session_snapshot(session_id)
            messages = cast(list[dict[str, object]], snapshot["messages"])
            if any(
                message.get("content") == "Main automatically handled the delegated result."
                for message in messages
            ):
                break
            await asyncio.sleep(0.01)

    assert sum(
        message.get("content") == "Main automatically handled the delegated result."
        for message in messages
    ) == 1
    claimed = await service.claim(successor.client_id, workspace.workspace_id, session_id)
    assert cast(dict[str, object], claimed["claim"])["session_id"] == session_id
    claimed_snapshot = cast(dict[str, object], claimed["snapshot"])
    claimed_messages = cast(list[dict[str, object]], claimed_snapshot["messages"])
    assert sum(
        message.get("content") == "Main automatically handled the delegated result."
        for message in claimed_messages
    ) == 1


@pytest.mark.asyncio
async def test_expired_task_child_denies_confirmation_without_workspace_audience(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _ = management_case
    provider = _DelegatingProvider()
    provider.write_path = "unroutable-child.txt"
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    service.client_permission(client_id).select("read-only")
    sink = cast(_CollectingSink, service.client(client_id).sink)
    claim = cast(dict[str, Any], (await service.claim(
        client_id, workspace.workspace_id, session_id
    ))["claim"])
    await workspace.input(client_id, session_id, claim["claim_version"],
                          "initial delegation request", _RUN_ID)
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await sink.wait_for("run.completed", _RUN_ID)
    outsider = await service.register_client("cli")
    outsider_sink = _CollectingSink()
    await service.connect_client(outsider.client_id, outsider_sink)
    allow_expiry = asyncio.Event()
    waiting = asyncio.Event()

    async def wait_for_expiry(_deadline: float) -> None:
        waiting.set()
        await allow_expiry.wait()

    monkeypatch.setattr(service, "_wait_until", wait_for_expiry)
    await service.disconnect_client(client_id, sink=sink)
    expiry = service.client(client_id).disconnect_task
    assert expiry is not None
    await asyncio.wait_for(waiting.wait(), timeout=5)
    allow_expiry.set()
    await asyncio.wait_for(expiry, timeout=5)
    provider.allow_write.set()
    provider.release_child.set()
    await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
    assert not (workspace.workspace_path / "unroutable-child.txt").exists()
    assert not any(event["type"] == "confirmation.requested" for event in outsider_sink.events)


@pytest.mark.asyncio
async def test_offline_foreground_result_waits_until_client_reconnects(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    client = service.client(client_id)
    reconnect_credential = client.reconnect_credential
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, client.sink)
    await workspace.input(
        client_id,
        session_id,
        cast(int, claim["claim_version"]),
        "Please make an initial delegation request.",
        _RUN_ID,
    )
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)

    await service.disconnect_client(client_id, sink=sink)
    assert service.state == "reconnecting"
    provider.release_child.set()
    async with asyncio.timeout(5):
        while True:
            subagents = cast(
                list[dict[str, object]],
                service.list_subagents(client_id, workspace.workspace_id, session_id)["items"],
            )
            if subagents[0]["status"] == "completed":
                break
            await asyncio.sleep(0.01)
    assert provider.automatic_continuation_count == 0

    reconnected = await service.register_client("cli", reconnect_credential)
    reconnected_sink = _CollectingSink()
    await service.connect_client(reconnected.client_id, reconnected_sink)
    assert service.state == "ready"
    await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=12)
    async with asyncio.timeout(12):
        while True:
            messages = cast(
                list[dict[str, object]],
                workspace.session_snapshot(session_id)["messages"],
            )
            if any(
                message.get("content") == "Main automatically handled the delegated result."
                for message in messages
            ):
                break
            await asyncio.sleep(0.01)

    assert provider.automatic_continuation_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_permission", "current_permission", "schedule_permission"),
    [
        ("read-only", "full-access", "workspace-write"),
        ("workspace-write", "read-only", "full-access"),
    ],
)
@pytest.mark.parametrize("expire_origin", [False, True])
async def test_claimless_automatic_continuation_keeps_task_ceiling_and_denies_unroutable_confirmation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    expire_origin: bool,
    task_permission: ToolPermissionLevel,
    current_permission: ToolPermissionLevel,
    schedule_permission: ToolPermissionLevel,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    provider.automatic_write = True
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    service.client_permission(client_id).select(task_permission)
    workspace._schedule_permission.select(schedule_permission)
    target = workspace.workspace_path / "auto-output.txt"
    target.write_bytes(b"original bytes")
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    sink = cast(_CollectingSink, service.client(client_id).sink)
    await workspace.input(
        client_id,
        session_id,
        cast(int, claim["claim_version"]),
        "Please make an initial delegation request.",
        _RUN_ID,
    )
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)

    await workspace.release(client_id, session_id)
    service.client_permission(client_id).select(current_permission)
    if expire_origin:
        successor = await service.register_client("cli")
        await service.connect_client(successor.client_id, _CollectingSink())
        expiry_waiting = asyncio.Event()
        allow_expiry = asyncio.Event()

        async def wait_for_expiry(_deadline: float) -> None:
            expiry_waiting.set()
            await allow_expiry.wait()

        monkeypatch.setattr(service, "_wait_until", wait_for_expiry)
        await service.disconnect_client(client_id, sink=sink)
        expiry = service.client(client_id).disconnect_task
        assert expiry is not None
        await asyncio.wait_for(expiry_waiting.wait(), timeout=5)
        allow_expiry.set()
        await asyncio.wait_for(expiry, timeout=5)
    provider.release_child.set()
    await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
    async with asyncio.timeout(12):
        while True:
            messages = cast(
                list[dict[str, object]],
                workspace.session_snapshot(session_id)["messages"],
            )
            if any(
                message.get("content") == "Main automatically handled the delegated result."
                for message in messages
            ):
                break
            await asyncio.sleep(0.01)

    assert target.read_bytes() == b"original bytes"
    assert not any(event.get("type") == "confirmation.requested" for event in sink.events)
    assert sum(
        message.get("content") == "Main automatically handled the delegated result."
        for message in messages
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["none", "claim", "dropped_event", "session_before", "session_after", "ledger"])
async def test_completed_child_automatically_continues_the_current_task_without_a_user_anchor(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim["claim_version"])
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
        before = workspace.session_snapshot(session_id)
        before_messages = cast(list[dict[str, object]], before["messages"])
        before_anchor_count = len(cast(list[object], before["restore_anchors"]))
        assert len(cast(list[dict[str, object]], service.list_subagents(
            client_id, workspace.workspace_id, session_id
        )["items"])) == 1

        original_replace = HOST_FILESYSTEM.atomic_replace_text
        injected = asyncio.Event()

        def fail_publication(path: Path, content: str) -> None:
            value = json.loads(content) if path.name == "state.json" else None
            deliveries = value.get("deliveries", []) if isinstance(value, dict) else []
            trigger = not injected.is_set() and (
                (fault == "claim" and any(d["state"] == "claimed" for d in deliveries))
                or (fault == "ledger" and any(d["state"] == "committed" for d in deliveries))
                or (fault == "session_after" and "Main automatically handled the delegated result." in content)
            )
            if trigger:
                injected.set()
                if fault == "session_after":
                    original_replace(path, content)
                raise OSError("Injected delivery publication failure")
            original_replace(path, content)

        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_publication)
        original_bytes = HOST_FILESYSTEM.atomic_replace_bytes

        def fail_session(path: Path, content: bytes) -> None:
            if (fault in {"session_before", "session_after"} and path.suffix == ".jsonl"
                    and b"Main automatically handled the delegated result." in content):
                injected.set()
                if fault == "session_after":
                    original_bytes(path, content)
                raise OSError("Injected Session persistence failure")
            original_bytes(path, content)

        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_session)
        if fault == "dropped_event":
            original_publish = workspace.publish

            async def drop_terminal(event: SubAgentEvent) -> None:
                if event.data.get("status") in {"completed", "failed"}:
                    return
                await original_publish(event)

            monkeypatch.setattr(workspace, "publish", drop_terminal)
        provider.release_child.set()
        if fault == "dropped_event":
            from tests.service.test_foreground_questions import _wait_until
            items = cast(list[dict[str, Any]], service.list_subagents(client_id, workspace.workspace_id, session_id)["items"])
            child_id = items[0]["agent_id"]
            await _wait_until(lambda: service.get_subagent(client_id, workspace.workspace_id, session_id, child_id)["status"] == "completed")
            assert provider.automatic_continuation_count == 0
            provider.keep_task = True
            wake = str(uuid4())
            await workspace.input(client_id, session_id, claim_version, "independent supplement", wake)
            await asyncio.wait_for(sink.wait_for("run.completed", wake), timeout=10)
        if fault == "claim":
            await asyncio.wait_for(injected.wait(), timeout=10)
            work = workspace.foreground_work_store(session_id).load()
            assert provider.automatic_continuation_count == 0
            assert work.deliveries[0].state == "pending"
            record = _require_record(workspace.subagent_repository(session_id), work.deliveries[0].agent_id)
            await workspace.publish(SubAgentEvent(
                kind=SubAgentEventKind.STATUS, workspace_id=workspace.workspace_id,
                session_id=session_id, agent_id=record.agent_id, revision=record.revision + 1000,
                occurred_at=_NOW, data={"status": "completed"},
            ))
        automatic_completed = await asyncio.wait_for(
            _wait_for_automatic_completion(sink, excluded_run_id=_RUN_ID), timeout=12
        )
        await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=5)
        if fault == "session_before":
            assert injected.is_set()
            work = workspace.foreground_work_store(session_id).load()
            assert work.deliveries[0].state == "claimed"
            assert work.tasks[-1].state == "interrupted"
            assert provider.automatic_continuation_count == 1
            assert not any(message.get("foreground_delivery_receipts") for message in
                           Session.load(workspace.workspace_state, session_id).messages)
            return

        automatic_run_id = automatic_completed.get("run_id")
        automatic_started = next(
            event
            for event in sink.events
            if event.get("type") == "run.started" and event.get("run_id") == automatic_run_id
        )
        assert cast(dict[str, object], automatic_started["payload"]).get("source") == "automatic"
        automatic_request = next(
            text for text, _tools in provider.requests
            if "SubAgent result for automatic continuation:" in text
        )
        assert "## Runtime Event: SubAgent Result" in automatic_request
        workspace.session_snapshot(session_id)
        delivery = workspace.foreground_work_store(session_id).load().deliveries[0]
        assert delivery.state == "committed"
        record = _require_record(workspace.subagent_repository(session_id), delivery.agent_id)
        for index in range(100):
            await workspace.publish(SubAgentEvent(
                kind=SubAgentEventKind.STATUS, workspace_id=workspace.workspace_id,
                session_id=session_id, agent_id=record.agent_id, revision=record.revision + index + 1000,
                occurred_at=_NOW, data={"status": "completed"},
            ))
        if fault not in {"none", "dropped_event"}:
            assert injected.is_set()

        after = workspace.session_snapshot(session_id)
        after_messages = cast(list[dict[str, object]], after["messages"])
        assert provider.automatic_continuation_count == 1
        assert sum(message.get("role") == "user" for message in after_messages) == sum(
            message.get("role") == "user" for message in before_messages
        ) + int(fault == "dropped_event")
        assert len(cast(list[object], after["restore_anchors"])) == before_anchor_count + int(fault == "dropped_event")
        assert any(
            message.get("content") == "Main automatically handled the delegated result."
            for message in after_messages
        )
        final_message = next(m for m in after_messages
                             if m.get("content") == "Main automatically handled the delegated result.")
        assert final_message["token_usage"] == {"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        trace = [record for record in (await recorder.list_records()).items
                 if record.run_id == automatic_run_id and record.kind is TraceKind.MODEL]
        assert len(trace) == 1
        assert trace[0].origin is TraceOrigin.FOREGROUND
        assert trace[0].owner_session_id == session_id
        assert trace[0].usage.input_tokens == 1 and trace[0].usage.output_tokens == 1
        assert trace[0].restore_run_token == cast(list[dict[str, object]], after["restore_anchors"])[-1]["run_token"]
        result = service.get_subagent(
            client_id,
            workspace.workspace_id,
            session_id,
            cast(str, cast(list[dict[str, object]], service.list_subagents(
                client_id, workspace.workspace_id, session_id
            )["items"])[0]["agent_id"]),
        )
        assert result["result"] == "SubAgent child answer."
    finally:
        provider.release_child.set()


@pytest.mark.asyncio
async def test_explicit_task_continue_recovers_a_persisted_subagent_result(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim["claim_version"])
    sink = cast(_CollectingSink, service.client(client_id).sink)
    seed_run_id = str(uuid4())

    await workspace.input(
        client_id,
        session_id,
        claim_version,
        "Continuation recovery seed.",
        seed_run_id,
    )
    await asyncio.wait_for(sink.wait_for("run.completed", seed_run_id), timeout=5)
    before = workspace.session_snapshot(session_id)
    persisted_messages = Session.load(workspace.workspace_state, session_id).messages
    task = next(
        item
        for item in cast(list[dict[str, object]], before["foreground_tasks"])
        if item["state"] == "completed"
    )
    user_anchor = next(
        message
        for message in reversed(persisted_messages)
        if message.get("role") == "user"
    )
    task_id = cast(str, task["task_id"])
    generation = cast(int, task["generation"])
    repository = workspace.subagent_repository(session_id)
    registered = repository.register(
        title="Recovered delegated result",
        task="Return a persisted result.",
        parent_run_id=seed_run_id,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=cast(str, user_anchor["restore_run_token"]),
            task_id=task_id,
            task_generation=generation,
        ),
        creator_snapshot=_snapshot(),
    )
    running = repository.save(
        replace(
            registered,
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=registered.revision + 1,
        )
    )
    terminal = repository.save(
        replace(
            running,
            status=SubAgentStatus.COMPLETED,
            finished_at=_NOW,
            conversation=({"role": "assistant", "content": "Recovered result."},),
            result="Recovered result details.",
            revision=running.revision + 1,
        )
    )
    work_store = workspace.foreground_work_store(session_id)
    work = work_store.reconcile(
        persisted_messages,
        subagents=(terminal,),
    )
    delivery = work.deliveries[0]
    stale_execution_id = str(uuid4())
    claimed = work_store.claim_delivery(
        agent_id=delivery.agent_id,
        terminal_record_revision=delivery.terminal_record_revision,
        task_generation=delivery.task_generation,
        execution_id=stale_execution_id,
    )
    assert claimed is not None
    work_store.start_execution(
        task_id,
        stale_execution_id,
        expected_generation=generation,
        explicit_user_operation=False,
    )
    work = work_store.reconcile(persisted_messages, subagents=(terminal,))
    assert work.deliveries[0].state == "claimed"
    assert next(item for item in work.tasks if item.task_id == task_id).state == "interrupted"
    before_user_count = sum(
        message.get("role") == "user"
        for message in cast(list[dict[str, object]], before["messages"])
    )

    old_request_count = len(provider.requests)
    workspace_path = workspace.workspace_path
    await service.stop()
    recovered = AgentService(service.agent_home, ConfigLoader(service.agent_home).load())
    await recovered.start()
    recovered.model_router._provider_factory = lambda _: provider
    recovered.model_router._providers.clear()
    try:
        client = await recovered.register_client("cli")
        client_id = client.client_id
        sink = _CollectingSink()
        await recovered.connect_client(client_id, sink)
        workspace = await recovered.attach_workspace(client_id, workspace_path)
        claim = cast(dict[str, object], (await recovered.claim(client_id, workspace.workspace_id, session_id))["claim"])
        claim_version = cast(int, claim["claim_version"])
        work = workspace.foreground_work_store(session_id).load()
        assert len(provider.requests) == old_request_count
        assert work.deliveries[0].state == "claimed"
        continued = await workspace.continue_task(
            client_id,
            session_id,
            claim_version,
            task_id,
            work.revision,
            str(uuid4()),
        )
        run_id = cast(str, continued["run_id"])
        await asyncio.wait_for(provider.explicit_result_continuation_started.wait(), timeout=5)
        await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=5)

        assert "Recovered delegated result" in provider.explicit_result_continuation_context
        assert "Recovered result details." in provider.explicit_result_continuation_context
        after = workspace.session_snapshot(session_id)
        after_messages = cast(list[dict[str, object]], after["messages"])
        assert sum(message.get("role") == "user" for message in after_messages) == before_user_count + 1
        assert after_messages[-2]["content"] == "Continue the current foreground task."
        assert after_messages[-2].get("foreground_delivery_receipts") is None
        delivery = workspace.foreground_work_store(session_id).load().deliveries[0]
        assert delivery.state == "committed"
        assert any(
            message.get("foreground_delivery_receipts") == [delivery.receipt_id]
            for message in after_messages
        )
    finally:
        await recovered.stop()


@pytest.mark.asyncio
async def test_foreground_subagent_question_suspends_and_notifies_its_waiter(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    claim_version = cast(int, claim["claim_version"])
    run_id = str(uuid4())

    await workspace.input(
        client_id,
        session_id,
        claim_version,
        "SubAgent question delegation request",
        run_id,
    )
    await asyncio.wait_for(provider.question_parent_started.wait(), timeout=5)
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, session_id)

    assert "ask_user" in provider.child_tool_names
    page = service.list_subagents(client_id, workspace.workspace_id, session_id)
    items = cast(list[dict[str, object]], page["items"])
    assert len(items) == 2
    assert all(item["status"] == "waiting_for_answer" for item in items)
    assert all(item["finished_at"] is None for item in items)
    questions = cast(list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"])
    assert len(questions) == 2
    assert {question["owner_agent_id"] for question in questions} == {
        item["agent_id"] for item in items
    }
    task = cast(list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_tasks"])[0]
    assert task["state"] == "waiting_for_answer"

    coordinator = workspace._subagent_coordinator(session_id)
    assert coordinator is not None
    parent_requests = len(provider.parent_tool_names)
    first_question = questions[0]
    first_answer_request_id = str(uuid4())
    result = await service.answer_foreground_question(
        client_id,
        workspace.workspace_id,
        session_id,
        cast(int, claim["claim_version"]),
        cast(str, first_question["question_id"]),
        "Use the concise format.",
        None,
        cast(int, first_question["revision"]),
        first_answer_request_id,
    )
    assert result["kind"] == "subagent_resumed"
    first_agent_id = cast(str, first_question["owner_agent_id"])
    second_question = questions[1]
    second_agent_id = cast(str, second_question["owner_agent_id"])
    result_after_one = await asyncio.wait_for(
        coordinator.wait([first_agent_id], timeout_ms=5000), timeout=6
    )
    assert provider.child_answer_seen.is_set()
    assert result_after_one[0].status is SubAgentStatus.COMPLETED
    assert _require_record(workspace.subagent_repository(session_id), second_agent_id).status is (
        SubAgentStatus.WAITING_FOR_ANSWER
    )
    assert len(cast(list[object], workspace.session_snapshot(session_id)["foreground_questions"])) == 1
    assert len(provider.parent_tool_names) == parent_requests

    replayed = await service.answer_foreground_question(
        client_id,
        workspace.workspace_id,
        session_id,
        cast(int, claim["claim_version"]),
        cast(str, first_question["question_id"]),
        "Use the concise format.",
        None,
        cast(int, first_question["revision"]),
        first_answer_request_id,
    )
    assert replayed["kind"] == "subagent_resumed"
    assert _require_record(workspace.subagent_repository(session_id), first_agent_id).status is (
        SubAgentStatus.COMPLETED
    )
    assert len(provider.parent_tool_names) == parent_requests

    second_question = cast(
        dict[str, object], cast(list[object], workspace.session_snapshot(session_id)["foreground_questions"])[0]
    )
    second_result = await service.answer_foreground_question(
        client_id,
        workspace.workspace_id,
        session_id,
        cast(int, claim["claim_version"]),
        cast(str, second_question["question_id"]),
        "Use the detailed format.",
        None,
        cast(int, second_question["revision"]),
        str(uuid4()),
    )
    assert second_result["kind"] == "subagent_resumed"
    result_after_two = await asyncio.wait_for(
        coordinator.wait([second_agent_id], timeout_ms=5000), timeout=6
    )
    assert result_after_two[0].status is SubAgentStatus.COMPLETED
    assert len(provider.parent_tool_names) == parent_requests


@pytest.mark.asyncio
async def test_missing_subagent_question_resource_reports_interruption(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client_id, session_id, _other_session_id = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _configuration: provider
    service.model_router._providers.clear()
    claim = cast(
        dict[str, object],
        (await service.claim(client_id, workspace.workspace_id, session_id))["claim"],
    )
    await workspace.input(
        client_id,
        session_id,
        cast(int, claim["claim_version"]),
        "SubAgent question delegation request",
        str(uuid4()),
    )
    await asyncio.wait_for(provider.question_parent_started.wait(), timeout=5)
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, session_id)

    questions = cast(
        list[dict[str, object]],
        workspace.session_snapshot(session_id)["foreground_questions"],
    )
    first_question, second_question = questions
    coordinator = workspace._subagent_coordinators.pop(session_id)
    parent_requests = len(provider.parent_tool_names)
    try:
        result = await service.answer_foreground_question(
            client_id,
            workspace.workspace_id,
            session_id,
            cast(int, claim["claim_version"]),
            cast(str, first_question["question_id"]),
            "Use the concise format.",
            None,
            cast(int, first_question["revision"]),
            str(uuid4()),
        )
    finally:
        workspace._subagent_coordinators[session_id] = coordinator

    assert result["kind"] == "subagent_interrupted"
    record = _require_record(
        workspace.subagent_repository(session_id), cast(str, first_question["owner_agent_id"])
    )
    assert record.status is SubAgentStatus.INTERRUPTED
    assert record.error is not None and record.error.code == "resource_unavailable"
    remaining = workspace.session_snapshot(session_id)
    remaining_questions = cast(list[dict[str, object]], remaining["foreground_questions"])
    states = {item["question_id"]: item["state"] for item in remaining_questions}
    assert states[first_question["question_id"]] == "interrupted"
    assert states[second_question["question_id"]] == "pending"
    task = cast(list[dict[str, object]], remaining["foreground_tasks"])[0]
    assert task["state"] == "waiting_for_answer"
    assert len(provider.parent_tool_names) == parent_requests


@pytest.mark.asyncio
async def test_missing_subagent_route_keeps_query_tools_but_fences_completed_run_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    _add_model_routes(home, "schedule")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda *_args: provider)
    service = AgentService(home, ConfigLoader(home).load())
    observed: list[tuple[str, ToolGateway]] = []
    original_run = AgentRunner.run

    async def record_run(
        runner: AgentRunner,
        initial_messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> AgentRunnerResult:
        observed.append((cast(str, kwargs["model"]), kwargs["tool_gateway"]))
        return await original_run(runner, initial_messages, **kwargs)

    monkeypatch.setattr(AgentRunner, "run", record_run)
    await service.start()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(
            client.client_id, reuse_startup_session=False, creation_scope="chat",
        )
        claim_payload = cast(
            dict[str, Any],
            (await service.claim(client.client_id, workspace.workspace_id, session_id))["claim"],
        )
        claim_version = cast(int, claim_payload["claim_version"])
        await workspace.input(
            client.client_id, session_id, claim_version,
            "Run foreground without SubAgent routing.", _RUN_ID,
        )
        await asyncio.wait_for(sink.wait_for("run.completed", _RUN_ID), timeout=5)

        repository = workspace.subagent_repository(session_id)
        completed = _complete(repository, _register(repository, "Existing task"))
        chat_gateway = next(gateway for route, gateway in observed if route == "chat")
        chat_tools = {tool.name for tool in chat_gateway.catalog}
        assert "spawn_agent" not in chat_tools
        assert {"wait_agent", "list_agents"} <= chat_tools
        chat_gateway.expose(("wait_agent", "list_agents"))
        listed = await chat_gateway.call(ModelToolCall(id="list", name="list_agents", arguments="{}"))
        waited = await chat_gateway.call(ModelToolCall(
            id="wait", name="wait_agent",
            arguments=json.dumps({"agent_ids": [completed.agent_id]}),
        ))
        assert listed.status == waited.status == "error"
        assert listed.content == waited.content == (
            "The foreground Task execution is no longer active."
        )

        job = ScheduleJob(
            job_id=str(uuid4()),
            message="Run Schedule without SubAgent routing.",
            title="No SubAgent route",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(job)
        workspace.schedule_service._reserve(
            job, current_monotonic=asyncio.get_running_loop().time(),
        )
        await asyncio.wait_for(workspace.schedule_service._active_runs[job.job_id].task, timeout=5)
        schedule_gateway = next(gateway for route, gateway in observed if route == "schedule")
        schedule_tools = {tool.name for tool in schedule_gateway.catalog}
        assert "spawn_agent" not in schedule_tools
        assert {"wait_agent", "list_agents"} <= schedule_tools
    finally:
        await service.stop()


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)

    def record_call(owner_session_id: str, subject_session_id: str) -> str:
        is_subagent = subject_session_id != owner_session_id
        context = recorder.new_root(
            TraceRootSource(
                workspace_key=workspace_key_for(workspace.workspace_path),
                workspace_path=str(workspace.workspace_path),
                origin=TraceOrigin.SUBAGENT if is_subagent else TraceOrigin.FOREGROUND,
                purpose=TracePurpose.SUBAGENT_RUN if is_subagent else TracePurpose.AGENT_RUN,
                owner_session_id=owner_session_id,
                subject_session_id=subject_session_id,
            )
        )
        purpose = TracePurpose.SUBAGENT_RUN if is_subagent else TracePurpose.AGENT_RUN
        handle = recorder.begin_call(
            context,
            TraceCallMetadata(
                kind=TraceKind.TOOL,
                purpose=purpose,
                started_at=_NOW,
                tool_name="read_file",
            ),
        )
        recorder.finish_call(
            handle,
            TraceOutcome(TraceStatus.SUCCESS, _NOW, 1),
            result={"output": "done"},
            usage=TraceTokenUsage(TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        return handle.record_id

    owner_record = record_call(session_id, session_id)
    child_record = record_call(session_id, "subagent-session")
    other_record = record_call(_other_session_id, _other_session_id)
    await wait_for_trace_writer(recorder)
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
    assert {record.record_id for record in (await recorder.list_records()).items} == {
        owner_record,
        child_record,
        other_record,
    }

    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    await cancellation

    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    purge_session = recorder.purge_session

    async def hold_cleanup(workspace_key: str, owner_id: str) -> None:
        cleanup_started.set()
        await cleanup_release.wait()
        await purge_session(workspace_key, owner_id)
        cleanup_finished.set()

    monkeypatch.setattr(recorder, "purge_session", hold_cleanup)

    try:
        result = await asyncio.wait_for(
            workspace.delete_session(
                client_id,
                session_id,
                claim.version,
                claim.credential,
                "delete-after-cancel",
            ),
            timeout=1,
        )
        assert result["deleted"] is True
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert {record.record_id for record in (await recorder.list_records()).items} == {
            other_record
        }
        assert await recorder.get_record(owner_record) is None
        assert await recorder.get_record(child_record) is None
        assert not cleanup_finished.is_set()
        assert repository.get(record.agent_id) is None
    finally:
        cleanup_release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=2)


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
    project, selected_workspace, _jobs = await register_project(service,
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
        "aide.service.runtime.service.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load())
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

@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel", "replace", "clear"])
async def test_task_control_drains_owned_child_and_keeps_history(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    operation: str,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle

    service, workspace, client_id, session_id, _ = management_case
    provider = _DelegatingProvider()
    assert service._model_router is not None
    service._model_router._provider_factory = lambda _configuration: provider
    service._model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(
        client_id, workspace.workspace_id, session_id
    ))["claim"])
    version = cast(int, claim["claim_version"])
    try:
        await workspace.input(client_id, session_id, version,
                              "initial delegation request", str(uuid4()))
        await asyncio.wait_for(provider.child_started.wait(), timeout=5)
        await _wait_for_idle(workspace, session_id)
        await workspace.release(client_id, session_id)
        reclaimed = await service.claim(client_id, workspace.workspace_id, session_id)
        version = cast(int, cast(dict[str, Any], reclaimed["claim"])["claim_version"])
        snapshot = workspace.session_snapshot(session_id)
        task = cast(list[dict[str, Any]], snapshot["foreground_tasks"])[-1]
        items = cast(list[dict[str, Any]], service.list_subagents(
            client_id, workspace.workspace_id, session_id
        )["items"])
        agent_id = items[0]["agent_id"]
        if operation == "cancel":
            await service.cancel_foreground_task(client_id, workspace.workspace_id, session_id,
                                                 version, task["task_id"], task["revision"], "cancel")
        else:
            original_complete = provider.complete

            async def framing(**kwargs: Any) -> ModelResponse:
                if operation == "clear":
                    return provider._response(
                        '{"action":"clear","task_goal":null,"completion_boundary":null}'
                    )
                return await original_complete(**kwargs)

            provider.complete = framing  # type: ignore[method-assign]
            await workspace.input(client_id, session_id, version,
                                  "follow-up result request", str(uuid4()))
            await _wait_for_idle(workspace, session_id)
        detail = service.get_subagent(client_id, workspace.workspace_id, session_id, agent_id)
        assert detail["status"] == "cancelled"
        requests_after = len(provider.requests)
        provider.release_child.set()
        await _wait_for_idle(workspace, session_id)
        assert len(provider.requests) == requests_after
    finally:
        provider.release_child.set()

@pytest.mark.asyncio
async def test_restored_session_can_delegate_through_resident_pool(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from aide.agent.session.restore import RestoreMode
    from tests.service.test_foreground_questions import _wait_for_idle

    service, workspace, client_id, sid, _ = management_case
    provider = _DelegatingProvider()
    provider.release_child.set()
    assert service._model_router is not None
    service._model_router._provider_factory = lambda _: provider
    service._model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client_id, workspace.workspace_id, sid))["claim"])
    version = cast(int, claim["claim_version"])
    await workspace.input(client_id, sid, version, "initial delegation request", str(uuid4()))
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await _wait_for_idle(workspace, sid)
    pool = workspace._subagent_coordinator(sid)
    assert pool is not None
    items = cast(list[dict[str, Any]], service.list_subagents(client_id, workspace.workspace_id, sid)["items"])
    await pool.wait([items[0]["agent_id"]], timeout_ms=3000)
    anchors = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])
    credential = workspace.require_claim(client_id, sid, version).credential
    await service.inspect_restore(client_id, workspace.workspace_id, sid, version, credential,
                                  "inspect", anchors[-1]["anchor_id"])
    restored = await service.commit_restore(client_id, workspace.workspace_id, sid, version, credential,
                                 "restore", anchors[-1]["anchor_id"], RestoreMode.CONVERSATION_ONLY.value)
    version = cast(int, restored["claim_version"])
    provider.spawn_requested = False
    provider.child_started.clear()
    await workspace.input(client_id, sid, version, "initial delegation request", str(uuid4()))
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await _wait_for_idle(workspace, sid)
    items = cast(list[dict[str, Any]], service.list_subagents(client_id, workspace.workspace_id, sid)["items"])
    assert len(items) == 1

@pytest.mark.asyncio
async def test_task_fence_blocks_child_tools_before_cleanup_finishes(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle

    service, workspace, client_id, sid, _ = management_case
    provider = _DelegatingProvider()
    provider.write_path = "late-child.txt"
    service.client_permission(client_id).select("workspace-write")
    assert service._model_router is not None
    service._model_router._provider_factory = lambda _: provider
    service._model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client_id, workspace.workspace_id, sid))["claim"])
    version = cast(int, claim["claim_version"])
    await workspace.input(client_id, sid, version, "initial delegation request", str(uuid4()))
    await asyncio.wait_for(provider.child_started.wait(), timeout=5)
    await _wait_for_idle(workspace, sid)
    task = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])[-1]
    entered = asyncio.Event()
    release = asyncio.Event()
    original = SubAgentPool.cancel_task_and_wait

    async def delayed(pool: SubAgentPool, task_id: str, generation: int) -> tuple[SubAgentRecord, ...]:
        entered.set()
        await release.wait()
        return await original(pool, task_id, generation)

    monkeypatch.setattr(SubAgentPool, "cancel_task_and_wait", delayed)
    cancellation = asyncio.create_task(service.cancel_foreground_task(
        client_id, workspace.workspace_id, sid, version, task["task_id"], task["revision"], "cancel"
    ))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        provider.allow_write.set()
        await asyncio.wait_for(provider.write_seen.wait(), timeout=5)
        assert not (workspace.workspace_path / "late-child.txt").exists()
    finally:
        release.set()
        provider.release_child.set()
        await cancellation


@pytest.mark.asyncio
async def test_child_completion_does_not_acknowledge_another_running_answer(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    class HeldAnswerProvider(_DelegatingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.held = asyncio.Event()
            self.release = asyncio.Event()

        def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
            async def emit() -> AsyncIterator[ModelStreamEvent]:
                users = [m for m in kwargs["messages"] if m.get("role") == "user"]
                if users and users[-1].get("content") == "Human Answer: hold":
                    self.held.set()
                    await self.release.wait()
                async for event in super(HeldAnswerProvider, self).stream(**kwargs):
                    yield event
            return emit()

    service, workspace, client_id, sid, _ = management_case
    provider = HeldAnswerProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client_id, workspace.workspace_id, sid))["claim"])
    version = claim["claim_version"]
    await workspace.input(client_id, sid, version, "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    questions = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])
    held_id = questions[0]["question_id"]
    try:
        await service.answer_foreground_question(client_id, workspace.workspace_id, sid, version,
            held_id, "hold", None, questions[0]["revision"], str(uuid4()))
        await asyncio.wait_for(provider.held.wait(), 5)
        other = next(q for q in cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])
                     if q["question_id"] != held_id)
        await service.answer_foreground_question(client_id, workspace.workspace_id, sid, version,
            other["question_id"], "finish", None, other["revision"], str(uuid4()))
        await _wait_until(lambda: service.get_subagent(client_id, workspace.workspace_id, sid,
            other["owner_agent_id"])["status"] == "completed")
        task = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])[0]
        with pytest.raises(ServiceError) as blocked:
            await service.continue_foreground_task(client_id, workspace.workspace_id, sid, version,
                task["task_id"], task["revision"], str(uuid4()))
        assert blocked.value.code == "foreground_task_conflict"
    finally:
        provider.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("lower_permission", [False, True])
async def test_child_answer_restore_recovers_file_and_conversation_prefix(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    lower_permission: bool,
) -> None:
    from aide.agent.session.restore import RestoreMode
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    service.client_permission(client).select("workspace-write")
    target = workspace.workspace_path / "child-answer.txt"
    target.write_bytes(b"original\x00bytes")
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    version = claim["claim_version"]
    await workspace.input(client, sid, version, "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    questions = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])
    first = questions[0]
    before = service.get_subagent(client, workspace.workspace_id, sid, first["owner_agent_id"])
    before_context = _require_record(workspace.subagent_repository(sid), first["owner_agent_id"]).context_state
    if lower_permission:
        service.client_permission(client).select("read-only")
    for index, question in enumerate(questions):
        current = next(q for q in cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])
                       if q["question_id"] == question["question_id"])
        await service.answer_foreground_question(client, workspace.workspace_id, sid, version,
            current["question_id"], "write-answer" if index == 0 else "finish", None,
            current["revision"], str(uuid4()))
        if lower_permission and index == 0:
            events = service.client(client).events
            await _wait_until(lambda events=events: any(e["type"] == "confirmation.requested" for e in events))
            event = next(e for e in events if e["type"] == "confirmation.requested")
            service.decide_confirmation(client, cast(dict[str, Any], event["payload"])["token"], "declined")
        await _wait_until(lambda agent_id=current["owner_agent_id"]: service.get_subagent(client, workspace.workspace_id, sid,
            agent_id)["status"] == "completed")
    assert target.read_bytes() == (b"original\x00bytes" if lower_permission else b"changed")
    total = service.get_subagent(client, workspace.workspace_id, sid, first["owner_agent_id"])["usage"]
    assert total == {"model_calls": 3, "input_tokens": 3, "output_tokens": 3, "total_tokens": 6}
    anchors = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])
    anchor = next(a for a in anchors if a["content"] == "SubAgent answer: write-answer")
    service.client_permission(client).select("workspace-write")
    credential = workspace.require_claim(client, sid, version).credential
    await service.inspect_restore(client, workspace.workspace_id, sid, version, credential,
                                  "inspect-child-answer", anchor["anchor_id"])
    await service.commit_restore(client, workspace.workspace_id, sid, version, credential,
        "restore-child-answer", anchor["anchor_id"],
        RestoreMode.CONVERSATION_ONLY.value if lower_permission else RestoreMode.FILES.value)
    restored = service.get_subagent(client, workspace.workspace_id, sid, first["owner_agent_id"])
    assert target.read_bytes() == b"original\x00bytes"
    assert restored["status"] == "waiting_for_answer"
    assert restored["conversation"] == before["conversation"]
    assert _require_record(workspace.subagent_repository(sid), first["owner_agent_id"]).context_state == before_context


@pytest.mark.asyncio
async def test_eight_real_question_runners_release_capacity_for_ninth_child(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    provider.question_children = 9
    provider.independent_ninth = True
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    await workspace.input(client, sid, claim["claim_version"], "SubAgent question delegation request", str(uuid4()))
    try:
        await asyncio.wait_for(provider.child_started.wait(), 20)
        await _wait_until(lambda: provider.child_question_count == 8)
        await _wait_for_idle(workspace, sid)
        items = cast(list[dict[str, Any]], service.list_subagents(client, workspace.workspace_id, sid)["items"])
        assert sum(item["status"] == "waiting_for_answer" for item in items) == 8
        assert sum(item["status"] == "running" for item in items) == 1
        assert len(cast(list[object], workspace.session_snapshot(sid)["foreground_questions"])) == 8
    finally:
        provider.release_child.set()


@pytest.mark.asyncio
async def test_child_question_publication_failure_reconciles_committed_runner(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    original = HOST_FILESYSTEM.atomic_replace_text
    failures = 0

    def fail_once(path: Path, content: str) -> None:
        nonlocal failures
        if path.name == "state.json" and failures == 0:
            data = json.loads(content)
            if any(q.get("owner_agent_id") and q["state"] == "pending" for q in data["questions"]):
                failures += 1
                raise OSError("injected child question publication failure")
        original(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_once)
    await workspace.input(client, sid, claim["claim_version"], "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    snapshot = workspace.session_snapshot(sid)
    questions = cast(list[dict[str, Any]], snapshot["foreground_questions"])
    assert failures == 1
    assert len(questions) == 2
    assert all(q["state"] == "pending" for q in questions)
    calls = provider.child_question_count
    first = questions[0]
    await service.answer_foreground_question(client, workspace.workspace_id, sid, claim["claim_version"],
        first["question_id"], "finish", None, first["revision"], str(uuid4()))
    await _wait_until(lambda: service.get_subagent(client, workspace.workspace_id, sid,
        first["owner_agent_id"])["status"] == "completed")
    assert provider.child_question_count == calls + 1


@pytest.mark.asyncio
async def test_saved_answer_with_failed_queue_publication_is_explicitly_interrupted(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    await workspace.input(client, sid, claim["claim_version"], "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    question = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])[0]
    original = HOST_FILESYSTEM.atomic_replace_text
    failures = 0

    def fail_after_queue_save(path: Path, content: str) -> None:
        nonlocal failures
        original(path, content)
        if path.stem == question["owner_agent_id"] and failures == 0:
            if json.loads(content).get("status") == "queued":
                failures += 1
                raise OSError("injected failure after queue publication")

    monkeypatch.setattr(workspace.subagent_repository(sid), "_replace_text", fail_after_queue_save)
    result = await service.answer_foreground_question(client, workspace.workspace_id, sid,
        claim["claim_version"], question["question_id"], "saved answer", None, question["revision"], str(uuid4()))
    assert failures == 1
    assert result["kind"] == "subagent_interrupted"
    assert service.get_subagent(client, workspace.workspace_id, sid, question["owner_agent_id"])["status"] == "interrupted"
    assert provider.child_question_count == 2
    assert "saved answer" in str(workspace.session_snapshot(sid)["messages"])
    other = next(item for item in cast(list[dict[str, Any]],
        workspace.session_snapshot(sid)["foreground_questions"])
        if item["question_id"] != question["question_id"])
    await service.answer_foreground_question(client, workspace.workspace_id, sid,
        claim["claim_version"], other["question_id"], "finish", None, other["revision"], str(uuid4()))
    await _wait_until(lambda: service.get_subagent(client, workspace.workspace_id, sid,
        other["owner_agent_id"])["status"] == "completed")
    task = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])[0]
    await service.continue_foreground_task(client, workspace.workspace_id, sid,
        claim["claim_version"], task["task_id"], task["revision"], str(uuid4()))
    await _wait_until(lambda: service.get_subagent(client, workspace.workspace_id, sid,
        question["owner_agent_id"])["status"] == "completed")
    assert not workspace.session_snapshot(sid)["foreground_questions"]


@pytest.mark.asyncio
async def test_child_answer_does_not_replace_a_removed_registered_model(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.configuration.test_model_configuration import editable_models
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    await workspace.input(client, sid, claim["claim_version"], "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    first = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])[0]
    models = editable_models(ConfigLoader(service.agent_home))
    registered = _require_record(workspace.subagent_repository(sid), first["owner_agent_id"]).creator_snapshot
    available = models["providers"][registered.provider_id]["models"]
    available["replacement-model"] = available.pop(registered.model)
    for route in models["routes"].values():
        if route.get("provider_id") == registered.provider_id:
            route["model"] = "replacement-model"
    await service.update_configuration("remove-registered-model", cast(str, service.config_view()["revision"]),
                                       {"models": models})
    result = await service.answer_foreground_question(client, workspace.workspace_id, sid,
        claim["claim_version"], first["question_id"], "finish", None, first["revision"], str(uuid4()))
    assert result["kind"] == "subagent_interrupted"
    record = _require_record(workspace.subagent_repository(sid), first["owner_agent_id"])
    assert record.status is SubAgentStatus.INTERRUPTED
    assert record.creator_snapshot == registered
    assert provider.child_question_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("lower_permission", [False, True])
async def test_automatic_file_effect_uses_user_anchor_and_current_permission(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    lower_permission: bool,
) -> None:
    from aide.agent.session.restore import RestoreMode
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    provider.automatic_write = True
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    service.client_permission(client).select("workspace-write")
    target = workspace.workspace_path / "auto-output.txt"
    target.write_bytes(b"original\x00bytes")
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    version = claim["claim_version"]
    sink = cast(_CollectingSink, service.client(client).sink)
    run_id = str(uuid4())
    await workspace.input(client, sid, version, "initial delegation request", run_id)
    await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=10)
    if lower_permission:
        service.client_permission(client).select("read-only")
    provider.release_child.set()
    if lower_permission:
        await _wait_until(lambda: any(e["type"] == "confirmation.requested" for e in sink.events))
        event = next(e for e in sink.events if e["type"] == "confirmation.requested")
        service.decide_confirmation(client, cast(dict[str, Any], event["payload"])["token"], "declined")
    await asyncio.wait_for(_wait_for_automatic_completion(sink, excluded_run_id=run_id), timeout=15)
    assert target.read_bytes() == (b"original\x00bytes" if lower_permission else b"automatic bytes")
    await _wait_for_idle(workspace, sid)
    anchors = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])
    anchor = next(a for a in anchors if a["content"] == "initial delegation request")
    credential = workspace.require_claim(client, sid, version).credential
    await service.inspect_restore(client, workspace.workspace_id, sid, version, credential,
                                  "inspect-auto", anchor["anchor_id"])
    await service.commit_restore(client, workspace.workspace_id, sid, version, credential,
        "restore-auto", anchor["anchor_id"],
        RestoreMode.CONVERSATION_ONLY.value if lower_permission else RestoreMode.FILES.value)
    assert target.read_bytes() == b"original\x00bytes"
    work = workspace.foreground_work_store(sid).load()
    assert all(d.state == "invalidated" for d in work.deliveries)


@pytest.mark.asyncio
async def test_queued_users_and_wait_delivery_precede_automatic_arbitration(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    sink = cast(_CollectingSink, service.client(client).sink)
    runs = [str(uuid4()) for _ in range(3)]
    await workspace.input(client, sid, claim["claim_version"], "initial delegation request", runs[0])
    await asyncio.wait_for(sink.wait_for("run.completed", runs[0]), timeout=10)
    provider.keep_task = True
    items = cast(list[dict[str, Any]], service.list_subagents(client, workspace.workspace_id, sid)["items"])
    provider.agent_id = items[0]["agent_id"]
    await workspace.input(client, sid, claim["claim_version"], "follow-up result request one", runs[1])
    await _wait_until(lambda: provider.wait_requested)
    await workspace.input(client, sid, claim["claim_version"], "follow-up result request two", runs[2])
    provider.release_child.set()
    await asyncio.wait_for(sink.wait_for("run.completed", runs[2]), timeout=15)
    await _wait_for_idle(workspace, sid)
    snapshot = workspace.session_snapshot(sid)
    assert provider.automatic_continuation_count == 0
    assert provider.old_result_seen.is_set()
    assert workspace.foreground_work_store(sid).load().deliveries[0].state == "committed"
    starts = [event["run_id"] for event in sink.events if event["type"] == "run.started"]
    assert starts == runs
    for previous, following in pairwise(runs):
        completed = next(i for i, e in enumerate(sink.events) if e["type"] == "run.completed" and e["run_id"] == previous)
        started = next(i for i, e in enumerate(sink.events) if e["type"] == "run.started" and e["run_id"] == following)
        assert completed < started
    assert sum(m["role"] == "user" for m in cast(list[dict[str, Any]], snapshot["messages"])) >= 3

@pytest.mark.asyncio
async def test_restart_two_child_questions_only_resumes_answered_chain(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    monkeypatch.setattr("aide.service.runtime.workspace.create_provider", lambda _: provider)
    monkeypatch.setattr("aide.provider.factory.create_provider", lambda _: provider)
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    await workspace.input(client, sid, claim["claim_version"],
                          "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    questions = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])
    assert len(questions) == 2
    calls = len(provider.requests)
    await service.stop()
    recovered = AgentService(service.agent_home, ConfigLoader(service.agent_home).load())
    await recovered.start()
    recovered.model_router._provider_factory = lambda _: provider
    recovered.model_router._providers.clear()
    try:
        owner = await recovered.register_client("cli")
        await recovered.connect_client(owner.client_id, _CollectingSink())
        resident = await recovered.attach_workspace(owner.client_id, workspace.workspace_path)
        current = cast(dict[str, Any], (await recovered.claim(owner.client_id, resident.workspace_id, sid))["claim"])
        snapshot = resident.session_snapshot(sid)
        restored = cast(list[dict[str, Any]], snapshot["foreground_questions"])
        assert {q["question_id"] for q in restored} == {q["question_id"] for q in questions}
        assert len(provider.requests) == calls
        task = cast(list[dict[str, Any]], snapshot["foreground_tasks"])[0]
        with pytest.raises(ServiceError):
            await recovered.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
                current["claim_version"], task["task_id"], task["revision"], str(uuid4()))
        first = restored[0]
        await recovered.answer_foreground_question(owner.client_id, resident.workspace_id, sid,
            current["claim_version"], first["question_id"], "finish", None,
            first["revision"], str(uuid4()))
        await _wait_until(lambda: recovered.get_subagent(owner.client_id, resident.workspace_id,
            sid, first["owner_agent_id"])["status"] == "completed")
        other = restored[1]
        assert recovered.get_subagent(owner.client_id, resident.workspace_id, sid,
            other["owner_agent_id"])["status"] == "waiting_for_answer"
        assert provider.child_question_count == 3
        pending = cast(list[dict[str, Any]], resident.session_snapshot(sid)["foreground_questions"])
        assert [q["question_id"] for q in pending] == [other["question_id"]]
    finally:
        await recovered.stop()

@pytest.mark.asyncio
async def test_deletion_marker_restart_does_not_recreate_child_or_task_records(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aide.agent.session.deletion import begin_session_deletion
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    monkeypatch.setattr("aide.service.runtime.workspace.create_provider", lambda _: provider)
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    await workspace.input(client, sid, claim["claim_version"],
                          "SubAgent question delegation request", str(uuid4()))
    await _wait_until(lambda: provider.child_question_count == 2)
    await _wait_for_idle(workspace, sid)
    old_store = workspace.foreground_work_store(sid)
    task = old_store.load().tasks[0]
    repository = workspace.subagent_repository(sid)
    child = repository.get(repository.list().items[0].agent_id)
    assert child is not None
    await service.stop()
    begin_session_deletion(workspace.workspace_state, sid)
    replacement = AgentService(service.agent_home, ConfigLoader(service.agent_home).load())
    await replacement.start()
    try:
        owner = await replacement.register_client("cli")
        await replacement.connect_client(owner.client_id, _CollectingSink())
        await replacement.attach_workspace(owner.client_id, workspace.workspace_path)
        old_store.complete_task(task.task_id, task.execution_id)
        with pytest.raises(SubAgentStoreError):
            repository.save(replace(child, revision=child.revision + 1))
        assert not (workspace.workspace_state.path / "foreground-work" / sid).exists()
        assert not (workspace.workspace_state.subagents_directory / sid).exists()
        assert provider.child_question_count == 2
    finally:
        await replacement.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_fault", ["none", "stage", "queue_after", "missing_model", "lower_permission"])
async def test_explicit_continue_resumes_original_interrupted_child_after_restart(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
    recovery_fault: str,
) -> None:
    from tests.service.test_foreground_questions import _wait_for_idle, _wait_until

    service, workspace, client, sid, _ = management_case
    provider = _DelegatingProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    monkeypatch.setattr("aide.service.runtime.workspace.create_provider", lambda _: provider)
    service.model_router._provider_factory = lambda _: provider
    service.model_router._providers.clear()
    claim = cast(dict[str, Any], (await service.claim(client, workspace.workspace_id, sid))["claim"])
    service.client_permission(client).select("workspace-write")
    await workspace.input(client, sid, claim["claim_version"], "initial delegation request", str(uuid4()))
    await asyncio.wait_for(provider.child_started.wait(), 5)
    await _wait_for_idle(workspace, sid)
    agent = cast(list[dict[str, Any]], service.list_subagents(client, workspace.workspace_id, sid)["items"])[0]
    await service.stop()
    original = _require_record(workspace.subagent_repository(sid), agent["agent_id"])
    requests_before_restart = len(provider.requests)
    replacement = AgentService(service.agent_home, ConfigLoader(service.agent_home).load())
    await replacement.start()
    replacement.model_router._provider_factory = lambda _: provider
    replacement.model_router._providers.clear()
    try:
        owner = await replacement.register_client("cli")
        sink = _CollectingSink()
        await replacement.connect_client(owner.client_id, sink)
        resident = await replacement.attach_workspace(owner.client_id, workspace.workspace_path)
        current = cast(dict[str, Any], (await replacement.claim(owner.client_id, resident.workspace_id, sid))["claim"])
        assert replacement.get_subagent(owner.client_id, resident.workspace_id, sid, agent["agent_id"])["status"] == "interrupted"
        assert len(provider.requests) == requests_before_restart
        task = cast(list[dict[str, Any]], resident.session_snapshot(sid)["foreground_tasks"])[0]
        provider.release_child.set()
        request_id = str(uuid4())
        if recovery_fault == "missing_model":
            from tests.configuration.test_model_configuration import editable_models
            models = editable_models(ConfigLoader(replacement.agent_home))
            creator = original.creator_snapshot
            available = models["providers"][creator.provider_id]["models"]
            available["replacement-model"] = available.pop(creator.model)
            for route in models["routes"].values():
                if route.get("provider_id") == creator.provider_id:
                    route["model"] = "replacement-model"
            await replacement.update_configuration("remove-recovery-model",
                cast(str, replacement.config_view()["revision"]), {"models": models})
            with pytest.raises(ServiceError) as missing:
                await replacement.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
                    current["claim_version"], task["task_id"], task["revision"], request_id)
            assert missing.value.code == "model_unavailable"
            assert len(provider.requests) == requests_before_restart
            assert _require_record(resident.subagent_repository(sid), agent["agent_id"]) == original
            return
        if recovery_fault == "lower_permission":
            replacement.client_permission(owner.client_id).select("read-only")
            provider.write_path = "recovery-write.txt"
            provider.allow_write.set()
        if recovery_fault in {"stage", "queue_after"}:
            atomic = HOST_FILESYSTEM.atomic_replace_text
            failed = False

            def interrupt_publication(path: Path, content: str) -> None:
                nonlocal failed
                value = json.loads(content)
                stage_match = recovery_fault == "stage" and path.name == "state.json" and any(
                    item.get("iteration_stage") == 2 for item in value.get("tasks", []))
                queue_match = recovery_fault == "queue_after" and path.stem == agent["agent_id"] and value.get("status") == "queued"
                if not failed and (stage_match or queue_match):
                    failed = True
                    if queue_match:
                        atomic(path, content)
                    raise OSError("injected interrupted child recovery publication")
                atomic(path, content)

            with monkeypatch.context() as fault:
                fault.setattr(HOST_FILESYSTEM, "atomic_replace_text", interrupt_publication)
                if recovery_fault == "queue_after":
                    fault.setattr(resident.subagent_repository(sid), "_replace_text", interrupt_publication)
                with pytest.raises(ServiceError):
                    await replacement.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
                        current["claim_version"], task["task_id"], task["revision"], request_id)
            assert failed
            assert len(provider.requests) == requests_before_restart
            await replacement.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
                current["claim_version"], task["task_id"], task["revision"], request_id)
            assert len(provider.requests) == requests_before_restart
            task = cast(list[dict[str, Any]], resident.session_snapshot(sid)["foreground_tasks"])[0]
            request_id = str(uuid4())
        continued = await replacement.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
            current["claim_version"], task["task_id"], task["revision"], request_id)
        for _ in range(10):
            repeated = await replacement.continue_foreground_task(owner.client_id, resident.workspace_id, sid,
                current["claim_version"], task["task_id"], task["revision"], request_id)
            assert repeated["run_id"] == continued["run_id"]
        await _wait_for_idle(resident, sid)
        if recovery_fault == "lower_permission":
            await _wait_until(lambda: any(e["type"] == "confirmation.requested" for e in sink.events))
            event = next(e for e in sink.events if e["type"] == "confirmation.requested")
            replacement.decide_confirmation(owner.client_id, cast(dict[str, Any], event["payload"])["token"], "declined")
        await _wait_until(lambda: replacement.get_subagent(owner.client_id, resident.workspace_id,
            sid, agent["agent_id"])["status"] == "completed")
        assert not (resident.workspace_path / "recovery-write.txt").exists()
        assert len(cast(list[object], replacement.list_subagents(owner.client_id,
            resident.workspace_id, sid)["items"])) == 1
        restored = _require_record(resident.subagent_repository(sid), agent["agent_id"])
        assert restored.creator_snapshot == original.creator_snapshot
        assert restored.conversation[:len(original.conversation)] == original.conversation
        assert restored.execution_id != original.execution_id
        messages = cast(list[dict[str, Any]], resident.session_snapshot(sid)["messages"])
        assert sum(m.get("foreground_continue_request_id") == request_id for m in messages) == 1
        await _wait_until(lambda: provider.automatic_continuation_count == 1)
        await _wait_for_idle(resident, sid)
        from aide.agent.session.restore import RestoreMode
        anchor = cast(list[dict[str, Any]], resident.session_snapshot(sid)["restore_anchors"])[-1]["anchor_id"]
        credential = resident.require_claim(owner.client_id, sid, current["claim_version"]).credential
        prefix = restored.restore_prefixes[-1]
        await replacement.inspect_restore(owner.client_id, resident.workspace_id, sid,
            current["claim_version"], credential, "inspect-recovery", anchor)
        await replacement.commit_restore(owner.client_id, resident.workspace_id, sid,
            current["claim_version"], credential, "restore-recovery", anchor,
            RestoreMode.CONVERSATION_ONLY.value)
        rolled_back = _require_record(resident.subagent_repository(sid), agent["agent_id"])
        assert rolled_back.status is SubAgentStatus.INTERRUPTED
        assert rolled_back.conversation == tuple(prefix["conversation"])
        assert rolled_back.usage == prefix["usage"]
        assert rolled_back.execution_id == prefix["execution_id"]
    finally:
        provider.release_child.set()
        await replacement.stop()
