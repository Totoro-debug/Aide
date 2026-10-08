"""End-to-end Schedule acceptance through AgentRunExecutor boundaries."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import UUID

import pytest

import aide.agent.context.builder as context
from aide.agent.context.builder import ContextBuilder
from aide.agent.loop import ConfirmationRequestView
from aide.agent.memory.dream import Dream
from aide.agent.memory.manager import MemoryManager
from aide.agent.message_bus import MessageBus
from aide.agent.permission import RuntimePermissionControl
from aide.agent.runner import AgentRunner, AgentRunnerResult
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.tools.core.exec_host import create_exec_host, resolve_exec_shell
from aide.agent.tools.deferred import RUN_BASELINE_TOOL_NAMES
from aide.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.management.commands import ManagementCommandDispatcher
from aide.provider.model_router import ModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from aide.schedule.history import read_schedule_history
from aide.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from aide.schedule.service import ScheduleClock, ScheduleService
from aide.schedule.store import WorkspaceScheduleStore
from aide.templates import render_template
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ProviderCall,
    TaskFramingRouterAdapter,
    collect_foreground_outbound,
)
from tests.fixtures.agent_loop import DrivenExecutor as AgentRunExecutor
from tests.fixtures.agent_loop import loaded_skill_loader
from tests.fixtures.diagnostic_capture import capture_diagnostics
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service

NOW = datetime(2026, 8, 7, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _FrozenContextDateTime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> _FrozenContextDateTime:
        if tz is None:
            return cls.fromtimestamp(NOW.timestamp(), tz=NOW.tzinfo)
        return cls.fromtimestamp(NOW.timestamp(), tz=tz)  # type: ignore[arg-type]


class _AdvancingContextDateTime(datetime):
    _milliseconds: ClassVar[int] = 0

    @classmethod
    def reset(cls) -> None:
        cls._milliseconds = 0

    @classmethod
    def now(cls, tz: object = None) -> _AdvancingContextDateTime:
        current = NOW + timedelta(milliseconds=cls._milliseconds)
        cls._milliseconds += 1
        if tz is None:
            return cls.fromtimestamp(current.timestamp(), tz=NOW.tzinfo)
        return cls.fromtimestamp(current.timestamp(), tz=tz)  # type: ignore[arg-type]


class _BlockingClock:
    """Keep a scheduler asleep while a test exercises another boundary."""

    def __init__(self, current: datetime) -> None:
        self._current = current
        self._sleep_forever = asyncio.Event()

    def now(self) -> datetime:
        return self._current

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        await self._sleep_forever.wait()


class _ScheduleProvider:
    """Route-aware provider transcript used by Schedule and AgentRunExecutor tests."""

    def __init__(
        self,
        *,
        chat_responses: Iterable[ModelResponse] = (),
        schedule_responses: Iterable[ModelResponse] = (),
        memory_responses: Iterable[ModelResponse] = (),
        block_chat_call: int | None = None,
        block_schedule_call: int | None = None,
    ) -> None:
        self._responses = {
            "chat": deque(chat_responses),
            "schedule": deque(schedule_responses),
            "memory": deque(memory_responses),
        }
        self._block_chat_call = block_chat_call
        self._block_schedule_call = block_schedule_call
        self._chat_call_count = 0
        self._schedule_call_count = 0
        self.chat_block_started = asyncio.Event()
        self.release_chat = asyncio.Event()
        self.schedule_block_started = asyncio.Event()
        self.release_schedule = asyncio.Event()
        self.stream_requests: list[ProviderCall] = []
        self.complete_requests: list[ProviderCall] = []
        self.direct_complete_messages: list[
            tuple[list[dict[str, object]], list[dict[str, Any]]]
        ] = []
        self.closed = False

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if messages and messages[0] == {
            "role": "system",
            "content": render_template("session-title-prompt.md"),
        }:
            yield ModelCompleted(response=_response("Schedule acceptance"))
            return

        self.stream_requests.append(
            ProviderCall(
                messages=deepcopy(list(messages)),
                tools=tuple(tools),
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        )
        self._chat_call_count += 1
        if self._block_chat_call == self._chat_call_count:
            self.chat_block_started.set()
            await self.release_chat.wait()
        yield ModelCompleted(response=self._take("chat"))

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        self.direct_complete_messages.append((deepcopy(list(messages)), list(tools)))
        call = ProviderCall(
            messages=deepcopy(list(messages)),
            tools=tuple(tools),
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        self.complete_requests.append(call)
        route = "schedule" if len(tools) == 8 else "memory"
        if route == "schedule":
            self._schedule_call_count += 1
            if self._block_schedule_call == self._schedule_call_count:
                self.schedule_block_started.set()
                await self.release_schedule.wait()
        return self._take(route)

    async def close(self) -> None:
        self.closed = True

    def _take(self, route: str) -> ModelResponse:
        responses = self._responses.get(route)
        if responses is None or not responses:
            raise AssertionError(f"No scripted response remains for {route}.")
        return responses.popleft()


def _is_schedule_call(call: ProviderCall) -> bool:
    return len(call.tools) == 8


def _response(content: str, *, tool_call: ModelToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        finish_reason="tool_calls" if tool_call is not None else "stop",
    )


def _schedule_tool_response(call_id: str, arguments: dict[str, object]) -> ModelResponse:
    return _response(
        "",
        tool_call=ModelToolCall(
            id=call_id,
            name="schedule",
            arguments=json.dumps(arguments, separators=(",", ":")),
        ),
    )


def _read_file_response(call_id: str) -> ModelResponse:
    return _response(
        "",
        tool_call=ModelToolCall(
            id=call_id,
            name="read_file",
            arguments=json.dumps({"path": "large.txt", "limit": 10_000}, separators=(",", ":")),
        ),
    )


def _due_job(*, message: str = "Run this.") -> ScheduleJob:
    return ScheduleJob(
        job_id=str(JOB_UUID),
        message=message,
        schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )


def _agent_loop(
    agent_home: Path,
    workspace: Path,
    provider: _ScheduleProvider,
    *,
    schedule_clock: ScheduleClock,
    config_text: str = VALID_CONFIG,
) -> tuple[
    AgentRunExecutor,
    ModelRouter,
    ScheduleService,
    Dream,
    ManagementCommandDispatcher,
    MessageBus,
]:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
    )
    memory_manager = MemoryManager(state)
    dream = Dream(
        memory_manager=memory_manager,
        model_router=router,
        batch_size=configuration.memory.batch_size,
        memory_route_status=router.route_status("memory"),
    )
    loop: AgentRunExecutor | None = None

    async def execute_user_job(job: ScheduleJob) -> None:
        assert loop is not None
        await loop.run_schedule_job(job)

    schedule = ScheduleService(
        workspace_state=state,
        clock=schedule_clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
    )
    bus = MessageBus()
    loop = AgentRunExecutor(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=home,
        configuration=configuration,
        bus=bus,
        schedule_service=schedule,
        model_router=TaskFramingRouterAdapter(router),
        memory_manager=memory_manager,
        session_id=None,
        now=lambda: NOW,
        new_uuid=lambda: JOB_UUID,
        monotonic_now=schedule_clock.monotonic,
        exec_host=create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell)),
        permission_control=RuntimePermissionControl(configuration.runtime.permission_level),
        skill_loader=loaded_skill_loader(home, configuration),
    )
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            current_agent_loop=lambda: loop.execution,
            workspace_state=state,
            memory_manager=memory_manager,
            dream=dream,
            schedule_status=lambda: schedule.status_snapshot().to_dict(),
            now=lambda: NOW,
            monotonic=schedule_clock.monotonic,
        )
    )
    return loop, router, schedule, dream, dispatcher, bus


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for Schedule acceptance state.")
        await asyncio.sleep(0)


def _schedule_state(workspace: Path) -> WorkspaceScheduleStore:
    return WorkspaceScheduleStore(WorkspaceState(workspace))


def _tool_json(loop: AgentRunExecutor) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(cast(str, message["content"])))
        for message in loop.session.messages
        if message.get("role") == "tool" and cast(str, message["content"]).startswith("{")
    ]


async def _close_components(
    loop: AgentRunExecutor,
    router: ModelRouter,
    schedule: ScheduleService,
    dream: Dream,
) -> None:
    await schedule.close()
    await loop.close()
    await dream.close()
    await router.close()


def _capture_foreground_projections(
    monkeypatch: pytest.MonkeyPatch,
    projections: list[list[dict[str, Any]]],
) -> None:
    original_build_foreground = ContextBuilder.build_foreground_messages

    def capture_projection(
        builder: ContextBuilder,
        messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        projected = original_build_foreground(builder, messages, **kwargs)
        projections.append(deepcopy(projected))
        return projected

    monkeypatch.setattr(ContextBuilder, "build_foreground_messages", capture_projection)


def _capture_schedule_projections(
    monkeypatch: pytest.MonkeyPatch,
    projections: list[list[dict[str, Any]]],
) -> None:
    original_build_schedule = ContextBuilder.build_schedule_messages

    def capture_projection(
        builder: ContextBuilder,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
        summary: str = "",
    ) -> list[dict[str, Any]]:
        projected = original_build_schedule(
            builder,
            messages,
            session_id=session_id,
            summary=summary,
        )
        projections.append(deepcopy(projected))
        return projected

    monkeypatch.setattr(ContextBuilder, "build_schedule_messages", capture_projection)


def _runtime_current_times(
    projections: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    times: list[str] = []
    for projection in projections:
        current_users = [
            cast(str, message["content"])
            for message in projection
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and cast(str, message["content"]).startswith("## Runtime Context")
        ]
        assert len(current_users) == 1
        current_line = next(
            line for line in current_users[0].splitlines() if line.startswith("- Current time: ")
        )
        times.append(current_line.removeprefix("- Current time: "))
    return times


@pytest.mark.asyncio
async def test_agent_loop_manages_schedule_jobs_without_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_add",
                {"action": "add", "message": "  ship report  ", "every_seconds": 60},
            ),
            _response("Added."),
            _schedule_tool_response("call_list", {"action": "list"}),
            _response("Listed."),
            _schedule_tool_response(
                "call_remove",
                {"action": "remove", "job_id": str(JOB_UUID)},
            ),
            _response("Removed."),
        )
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "Schedule the report.")

        jobs = await _schedule_state(workspace).public_snapshot()
        assert len(jobs) == 1
        job_id = jobs[0].job_id
        assert jobs[0].message == "ship report"

        await collect_foreground_outbound(_bus, "List my Schedule Jobs.")
        provider._responses["chat"][0] = _schedule_tool_response(
            "call_remove",
            {"action": "remove", "job_id": job_id},
        )
        results = _tool_json(loop)
        assert results[0]["action"] == "add"
        assert results[1] == {
            "jobs": [
                {
                    "job_id": job_id,
                    "message": "ship report",
                    "title": "ship report",
                    "schedule": {"type": "every", "every_seconds": 60},
                }
            ]
        }

        await collect_foreground_outbound(_bus, "Remove that Schedule Job.")
        assert await _schedule_state(workspace).public_snapshot() == ()
        assert _tool_json(loop)[-1]["action"] == "remove"
    finally:
        await _close_components(loop, router, schedule, dream)


@pytest.mark.asyncio
async def test_foreground_schedule_add_uses_configured_level_and_current_override(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_add",
                {"action": "add", "message": "  Run it  ", "every_seconds": 60},
            ),
            _response("Added."),
        )
    )
    loop, router, schedule, dream, _dispatcher, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    loop._permission_control.select("read-only")
    confirmations: list[ConfirmationRequestView] = []

    def approve(request: ConfirmationRequestView) -> None:
        confirmations.append(request)
        loop.control.respond_to_confirmation(request.confirmation_id, "approved")

    loop.control.bind_confirmation_callback(approve)
    await loop.start()
    try:
        await collect_foreground_outbound(bus, "Schedule the report.")
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(confirmations) == 1
    assert confirmations[0].details == {
        "action": "add",
        "message": "Run it",
        "title": "Run it",
        "schedule": {"type": "every", "every_seconds": 60},
    }
    assert confirmations[0].reason.count("requires confirmation.") == 2
    assert "persistent scheduled work" in confirmations[0].reason
    assert "configured Schedule level 'workspace-write'" in confirmations[0].reason
    jobs = await _schedule_state(workspace).public_snapshot()
    assert len(jobs) == 1
    assert {
        "permission",
        "level",
        "current",
        "configured",
        "snapshot",
    }.isdisjoint(jobs[0].to_dict())


@pytest.mark.asyncio
async def test_foreground_provider_receives_builder_complete_context_projection(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(_response("First answer."), _response("Second answer.")),
    )
    projection_calls: list[list[dict[str, Any]]] = []
    _capture_foreground_projections(monkeypatch, projection_calls)
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    monkeypatch.setattr(context, "datetime", _FrozenContextDateTime)
    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "First question.")
        await collect_foreground_outbound(_bus, "Second question.")
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(provider.stream_requests) == 2
    assert all(call.messages in projection_calls for call in provider.stream_requests)
    first_messages = provider.stream_requests[0].messages
    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert "First question." in str(first_messages[-1]["content"])
    second_messages = provider.stream_requests[1].messages
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[1] == {"role": "user", "content": "First question."}
    assert second_messages[2] == {
        "role": "assistant",
        "content": "First answer.",
        "tool_calls": [],
    }
    assert "Second question." in str(second_messages[3]["content"])
    assert all("timestamp" not in message for message in second_messages)


@pytest.mark.asyncio
async def test_schedule_uses_context_builder_complete_context_projection(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Run this.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    provider = _ScheduleProvider(schedule_responses=(_response("Background result."),))
    builder_projections: list[list[dict[str, Any]]] = []

    def fail_foreground_context(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        raise AssertionError("Schedule must not use ContextBuilder")

    _capture_schedule_projections(monkeypatch, builder_projections)
    schedule_now = NOW + timedelta(hours=2)
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(schedule_now),
    )
    await loop.start()
    monkeypatch.setattr(ContextBuilder, "build_foreground_messages", fail_foreground_context)
    _AdvancingContextDateTime.reset()
    monkeypatch.setattr(context, "datetime", _AdvancingContextDateTime)
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                len(provider.direct_complete_messages) == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(provider.direct_complete_messages) == 1
    messages, tools = provider.direct_complete_messages[0]
    runner_tool_names = tuple(schema["function"]["name"] for schema in tools)
    assert "schedule" not in runner_tool_names
    assert builder_projections
    assert messages in builder_projections
    assert messages[0]["role"] == "system"
    system_content = cast(str, messages[0]["content"])
    assert str(workspace) in system_content
    assert "## Task goal" not in system_content
    assert "## Completion boundary" not in system_content
    assert system_content.splitlines().count("# Long-term Memory") == 0
    assert system_content.splitlines().count("## Long-term Memory") == 1
    assert all(
        f"### {section}" in system_content
        for section in ("User Info", "User Preference", "Project Fact", "Lesson")
    )
    assert messages[-1]["role"] == "user"
    assert messages[-1] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            f"- Current time: {NOW.astimezone().isoformat(timespec='milliseconds')}\n"
            f"- Session ID: schedule_{JOB_UUID}\n\n"
            "## User Input\n\n"
            "Run this."
        ),
    }
    assert schedule_now.isoformat(timespec="milliseconds") not in str(messages[-1]["content"])
    assert all("timestamp" not in message for message in messages)
    assert tuple(definition["function"]["name"] for definition in tools) == (
        RUN_BASELINE_TOOL_NAMES
    )


@pytest.mark.asyncio
async def test_schedule_tool_loop_persists_each_message_from_awaitable_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Read the memory template.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    provider = _ScheduleProvider(
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments=json.dumps(
                        {"path": str(state.long_term_memory_path)},
                        separators=(",", ":"),
                    ),
                ),
            ),
            _response("The memory template was read."),
        )
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                len(provider.direct_complete_messages) == 2
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    second_messages, _ = provider.direct_complete_messages[1]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2]["tool_calls"] == [
        {
            "id": "call_read",
            "name": "read_file",
            "arguments": json.dumps(
                {"path": str(state.long_term_memory_path)},
                separators=(",", ":"),
            ),
        }
    ]
    assert second_messages[3]["tool_call_id"] == "call_read"
    session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[-1]["content"] == "The memory template was read."
    history = read_schedule_history(WorkspaceState(workspace), str(JOB_UUID))
    groups = cast(list[dict[str, Any]], history["groups"])
    assert len(groups) == 1
    assert groups[0]["result_state"] == "success"
    assert groups[0]["messages"] == session.messages
    assert groups[0]["messages"][2]["status"] == "success"


@pytest.mark.asyncio
async def test_two_schedule_executions_group_after_fresh_history_load(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        schedule_responses=(
            _response("First persisted result."),
            _response("Second persisted result."),
        ),
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    try:
        await loop.run_schedule_job(_due_job(message="First scheduled input."))
        await loop.run_schedule_job(_due_job(message="Second scheduled input."))
    finally:
        await _close_components(loop, router, schedule, dream)

    fresh_state = WorkspaceState(workspace)
    history = read_schedule_history(
        fresh_state,
        str(JOB_UUID),
        workspace_id="workspace-under-test",
    )
    groups = history["groups"]
    assert isinstance(groups, list)
    assert [group["result_state"] for group in groups] == ["success", "success"]
    assert [group["messages"][0]["content"] for group in groups] == [
        "First scheduled input.",
        "Second scheduled input.",
    ]
    assert [
        [(message["role"], message["content"]) for message in group["messages"]] for group in groups
    ] == [
        [("user", "First scheduled input."), ("assistant", "First persisted result.")],
        [("user", "Second scheduled input."), ("assistant", "Second persisted result.")],
    ]
    assert all(group["complete"] is True for group in groups)
    assert list(fresh_state.schedule_sessions_directory.glob("*.jsonl")) == [
        fresh_state.schedule_sessions_directory / f"schedule_{JOB_UUID}.jsonl"
    ]


@pytest.mark.asyncio
async def test_schedule_tool_loop_does_not_prepare_compaction_inside_agent_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Continue after reading memory.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    schedule_session = Session.create_schedule(state, JOB_UUID, now=lambda: NOW)
    seed_session_state(
        schedule_session,
        messages=[
            {
                "role": "user",
                "content": "Old scheduled request.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            },
            {
                "role": "assistant",
                "content": "Old scheduled answer.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    schedule_session.close()
    provider = _ScheduleProvider(
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments=json.dumps({"path": str(state.long_term_memory_path)}),
                ),
            ),
            _response("Continuation completed."),
        ),
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    await loop.start()
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                [_is_schedule_call(request) for request in provider.complete_requests]
                == [True, True]
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    schedule_messages = [messages for messages, _tools in provider.direct_complete_messages]
    assert [message["role"] for message in schedule_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert "Old scheduled request." in json.dumps(schedule_messages[1])
    assert not (state.memory_directory / "summary.jsonl").exists()
    persisted = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert persisted.last_compacted == 0
    assert persisted.metadata["title"] == "Untitled session"


@pytest.mark.asyncio
async def test_schedule_summary_flows_through_memory_to_a_later_schedule_run(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_text = (
        VALID_CONFIG
        + """

[models.routes.schedule]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 4096
max_output = 512
temperature = 0.2
reasoning_effort = "mid"
timeout = 120

[models.routes.memory]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120
"""
    )
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    store = WorkspaceScheduleStore(state)
    created_at_ms = int((NOW - timedelta(seconds=60)).timestamp() * 1000)
    await store.add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            source="user",
            message="Continue the scheduled work.",
            schedule=JobSchedule.every(60),
            state=ScheduleJobState(),
            created_at_ms=created_at_ms,
            updated_at_ms=created_at_ms,
        )
    )
    schedule_session = Session.create(
        state,
        now=lambda: NOW,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=JOB_UUID,
    )
    history_padding = " previous schedule context" * 150
    seed_session_state(
        schedule_session,
        messages=[
            {
                "role": "user",
                "content": "Oldest scheduled request." + history_padding,
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            },
            {
                "role": "assistant",
                "content": "Oldest scheduled answer." + history_padding,
                "timestamp": NOW.isoformat(timespec="milliseconds"),
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
            {
                "role": "user",
                "content": "Earlier scheduled request." + history_padding,
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            },
            {
                "role": "assistant",
                "content": "Earlier scheduled answer." + history_padding,
                "timestamp": NOW.isoformat(timespec="milliseconds"),
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 2,
                "input_tokens": 2,
                "output_tokens": 2,
                "total_tokens": 4,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    schedule_session.close()

    memory_path = state.long_term_memory_path
    old_memory = memory_path.read_text(encoding="utf-8")
    new_user_info = "## User Info\n\nFresh schedule preference.\n"
    new_memory = old_memory.replace(
        "## User Info\n",
        new_user_info,
        1,
    )
    provider = _ScheduleProvider(
        schedule_responses=(
            _response("First scheduled result." + history_padding),
            _response("Second scheduled result." + history_padding),
        ),
        memory_responses=(
            _response("Schedule history summary."),
            _response("- Completed the scheduled work."),
            _response(
                "",
                tool_call=ModelToolCall(
                    id="edit-memory",
                    name="edit_file",
                    arguments=json.dumps(
                        {
                            "path": str(memory_path),
                            "old_text": "## User Info\n",
                            "new_text": new_user_info,
                            "replace_all": False,
                        }
                    ),
                ),
            ),
            _response("Second schedule history summary."),
            _response("None"),
        ),
    )
    builder_projections: list[list[dict[str, Any]]] = []
    _capture_schedule_projections(monkeypatch, builder_projections)
    clock = FakeClock(NOW)
    loop, router, schedule, dream_owner, dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
        config_text=config_text,
    )
    try:
        await loop.start()
        _AdvancingContextDateTime.reset()
        monkeypatch.setattr(context, "datetime", _AdvancingContextDateTime)
        schedule.start()
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if _is_schedule_call(request)
                    ]
                )
                == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )
        first_projection_count = len(builder_projections)
        assert first_projection_count > 1
        first_projection_times = _runtime_current_times(builder_projections)
        assert set(first_projection_times) == {NOW.astimezone().isoformat(timespec="milliseconds")}
        summary_records = [
            json.loads(line)
            for line in (state.memory_directory / "summary.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["content"] for record in summary_records] == ["Schedule history summary."]

        dream_result = await dispatcher.dispatch("/dream")
        assert dream_result.output is not None
        assert (state.memory_directory / ".cursor").read_bytes() == b"1\n"
        assert memory_path.read_text(encoding="utf-8") == new_memory

        clock.advance(60)
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if _is_schedule_call(request)
                    ]
                )
                == 2
                and schedule.status_snapshot().active_job_count == 0
            )
        )
        second_projection_times = _runtime_current_times(
            builder_projections[first_projection_count:]
        )
        assert len(second_projection_times) > 1
        assert len(set(second_projection_times)) == 1
        assert second_projection_times[0] != first_projection_times[0]
        schedule_requests = [
            request for request in provider.complete_requests if _is_schedule_call(request)
        ]
        assert schedule_requests[0].messages[1] == {
            "role": "user",
            "content": "- Completed the scheduled work.",
        }
        assert "Fresh schedule preference." not in cast(
            str, schedule_requests[0].messages[0]["content"]
        )
        assert "Fresh schedule preference." in cast(
            str, schedule_requests[1].messages[0]["content"]
        )
        assert all(message.get("content") != "None" for message in schedule_requests[1].messages)
        restored_schedule = Session.load(
            state,
            f"schedule_{JOB_UUID}",
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert restored_schedule.metadata["summary"] == "- Completed the scheduled work."
        assert all(message.get("content") != "None" for message in restored_schedule.messages)
        summary_records = [
            json.loads(line)
            for line in (state.memory_directory / "summary.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["content"] for record in summary_records] == ["Schedule history summary."]
    finally:
        await _close_components(loop, router, schedule, dream_owner)


@pytest.mark.asyncio
async def test_concurrent_runs_keep_summaries_and_micro_compression_state_isolated(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            *(_read_file_response(f"foreground-{index}") for index in range(10)),
            _response("Foreground result."),
        ),
        schedule_responses=(
            *(_read_file_response(f"schedule-{index}") for index in range(11)),
            _response("Schedule result."),
        ),
        block_schedule_call=12,
    )
    clock = _BlockingClock(NOW)
    loop, router, schedule, dream, _dispatcher, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
        config_text=VALID_CONFIG.replace(
            "[runtime]\n", "[runtime]\nenable_tool_micro_compression = true\n", 1
        ),
    )
    foreground_summary = "- Updated the foreground Session."
    schedule_summary = "- Updated the Schedule Session."
    loop.session.update_metadata(summary=foreground_summary)

    schedule_session = Session.create(
        loop.session.workspace_state,
        now=lambda: NOW,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=JOB_UUID,
    )
    seed_session_state(
        schedule_session,
        messages=[
            {
                "role": "user",
                "content": "Earlier scheduled work.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": schedule_summary,
        },
        last_compacted=0,
    )
    schedule_session.close()
    (workspace / "large.txt").write_text("x" * 4_000, encoding="utf-8")

    await loop.start()
    scheduled = asyncio.create_task(loop.run_schedule_job(_due_job()))
    try:
        await _wait_until(provider.schedule_block_started.is_set)
        foreground = asyncio.create_task(
            collect_foreground_outbound(bus, "Continue the foreground work.")
        )
        await asyncio.wait_for(foreground, timeout=3.0)
        provider.release_schedule.set()
        await asyncio.wait_for(scheduled, timeout=3.0)
    finally:
        provider.release_schedule.set()
        await asyncio.wait_for(
            asyncio.gather(scheduled, return_exceptions=True),
            timeout=3.0,
        )
        await asyncio.wait_for(
            _close_components(loop, router, schedule, dream),
            timeout=3.0,
        )

    assert len(provider.stream_requests) == 11
    foreground_messages = provider.stream_requests[-1].messages
    assert foreground_messages[1] == {
        "role": "user",
        "content": foreground_summary,
    }
    assert schedule_summary not in json.dumps(foreground_messages)
    foreground_tools = [message for message in foreground_messages if message.get("role") == "tool"]
    assert len(foreground_tools) == 10
    assert all(
        "result omitted from context" not in str(message["content"]) for message in foreground_tools
    )

    schedule_requests = [
        request for request in provider.complete_requests if _is_schedule_call(request)
    ]
    assert len(schedule_requests) == 12
    schedule_messages = schedule_requests[-1].messages
    assert schedule_messages[1] == {
        "role": "user",
        "content": schedule_summary,
    }
    assert foreground_summary not in json.dumps(schedule_messages)
    schedule_tools = [message for message in schedule_messages if message.get("role") == "tool"]
    assert len(schedule_tools) == 11
    assert (
        sum(
            message["content"] == "[read_file result omitted from context]"
            for message in schedule_tools
        )
        == 10
    )
    latest_schedule_content = schedule_tools[-1]["content"]
    assert isinstance(latest_schedule_content, str)
    assert len(latest_schedule_content) > 512

    restored_foreground = Session.load(loop.session.workspace_state, loop.session.session_id)
    restored_schedule = Session.load(
        loop.session.workspace_state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert restored_foreground.metadata["summary"] == foreground_summary
    assert restored_schedule.metadata["summary"] == schedule_summary
    assert all(
        message.get("content") not in {foreground_summary, schedule_summary}
        for message in (*restored_foreground.messages, *restored_schedule.messages)
    )


@pytest.mark.asyncio
async def test_schedule_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_due_add",
                {
                    "action": "add",
                    "message": "run after the dispatcher wakes",
                    "at_time": NOW.isoformat(timespec="milliseconds"),
                },
            ),
            _response("Scheduled."),
        ),
        schedule_responses=(_response("Background result."),),
    )
    clock = FakeClock(NOW)
    loop, router, schedule, dream, dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
    )
    await loop.start()
    schedule.start()
    try:
        events = await collect_foreground_outbound(_bus, "Schedule this due task.")
        assert events[-1].metadata == {"_streamed": True}
        await _wait_until(
            lambda: (
                len(provider.complete_requests) == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )

        assert _is_schedule_call(provider.complete_requests[0])
        assert (
            tuple(
                definition["function"]["name"] for definition in provider.complete_requests[0].tools
            )
            == RUN_BASELINE_TOOL_NAMES
        )
        assert await _schedule_state(workspace).public_snapshot() == ()
        schedule_session_paths = tuple(
            (workspace / ".aide" / "schedule-sessions").glob("schedule_*.jsonl")
        )
        assert len(schedule_session_paths) == 1
        schedule_session_id = schedule_session_paths[0].stem
        schedule_session = Session.load(
            WorkspaceState(workspace),
            schedule_session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert [message["role"] for message in schedule_session.messages] == [
            "user",
            "assistant",
        ]
        assert schedule_session.messages[-1]["content"] == "Background result."
        assert (
            workspace / ".aide" / "schedule-sessions" / f"{schedule_session_id}.jsonl"
        ).exists()

        resume = await dispatcher.dispatch("/resume")
        assert resume.output is not None
        assert schedule_session_id not in resume.output
        assert {message.type for message in events} <= {
            "model_reasoning",
            "model_response",
            "tool_call",
            "system_control",
        }
    finally:
        await _close_components(loop, router, schedule, dream)


def test_schedule_service_user_executor_is_bound_to_agent_loop(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider()
    loop, _router, schedule, _dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    callback = schedule._execute_user_job
    closure_values = tuple(cell.cell_contents for cell in (callback.__closure__ or ()))
    assert loop in closure_values


@pytest.mark.asyncio
async def test_foreground_and_schedule_use_distinct_runner_and_gateway_identity(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(_response("Foreground result."),),
        schedule_responses=(_response("Schedule result."),),
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    observed: list[tuple[AgentRunner, ToolGateway, str, object, object]] = []
    original_run = AgentRunner.run

    async def record_run(
        runner: AgentRunner,
        initial_messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> AgentRunnerResult:
        observed.append(
            (
                runner,
                kwargs["tool_gateway"],
                cast(str, kwargs["model"]),
                kwargs["confirmation"],
                kwargs["on_output"],
            )
        )
        return await original_run(runner, initial_messages, **kwargs)

    monkeypatch.setattr(AgentRunner, "run", record_run)
    callback = loop.run_schedule_job
    await loop.start()
    try:
        assert (await collect_foreground_outbound(_bus, "Run foreground."))[-1].metadata == {
            "_streamed": True
        }
        await callback(_due_job(message="Run Schedule."))
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(observed) == 2
    assert observed[0][0] is not observed[1][0]
    assert observed[0][1] is not observed[1][1]
    assert tuple(schema["function"]["name"] for schema in observed[0][1].schemas) == (
        RUN_BASELINE_TOOL_NAMES
    )
    assert tuple(schema["function"]["name"] for schema in observed[1][1].schemas) == (
        RUN_BASELINE_TOOL_NAMES
    )
    assert [call[2] for call in observed] == ["chat", "schedule"]
    assert observed[0][3] is not None
    assert observed[1][3] is None
    assert observed[0][4] is not observed[1][4]


@pytest.mark.asyncio
async def test_foreground_and_schedule_artifacts_remain_separate(
    agent_home: Path,
    workspace: Path,
) -> None:
    large_file = workspace / "large.txt"
    large_file.parent.mkdir(parents=True, exist_ok=True)
    large_file.write_text("x\n" * 4000, encoding="utf-8")
    provider = _ScheduleProvider(
        chat_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="foreground_artifact",
                    name="read_file",
                    arguments='{"path":"large.txt","limit":10000}',
                ),
            ),
            _response("Foreground artifact stored."),
        ),
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="schedule_artifact",
                    name="read_file",
                    arguments='{"path":"large.txt","limit":10000}',
                ),
            ),
            _response("Schedule artifact stored."),
        ),
    )
    config_text = VALID_CONFIG.replace(
        "max_tool_result_chars = 60000", "max_tool_result_chars = 1000"
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=config_text,
    )
    callback = loop.run_schedule_job
    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "Read the large file in foreground.")
        await callback(_due_job(message="Read the large file in Schedule."))
        foreground_session_id = loop.session.session_id
        foreground_tool = next(
            message for message in loop.session.messages if message["role"] == "tool"
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    state = WorkspaceState(workspace)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    schedule_tool = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    foreground_artifact = cast(dict[str, object], foreground_tool["artifact"])
    schedule_artifact = cast(dict[str, object], schedule_tool["artifact"])
    assert set(foreground_artifact) == {"path", "total_chars", "preview_chars"}
    assert set(schedule_artifact) == {"path", "total_chars", "preview_chars"}
    assert foreground_artifact["path"] == (
        f".aide/artifacts/{foreground_session_id}/foreground_artifact.txt"
    )
    assert schedule_artifact["path"] == (
        f".aide/artifacts/schedule_{JOB_UUID}/schedule_artifact.txt"
    )
    assert (workspace / foreground_artifact["path"]).exists()
    assert (workspace / schedule_artifact["path"]).exists()


@pytest.mark.asyncio
async def test_schedule_session_uses_schedule_clock_for_persisted_timestamps(
    agent_home: Path,
    workspace: Path,
) -> None:
    schedule_now = NOW + timedelta(hours=2)
    provider = _ScheduleProvider(schedule_responses=(_response("Background result."),))
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(schedule_now),
    )
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run at the Schedule clock time.",
        title="Schedule clock title",
        schedule=JobSchedule.at("2026-08-07T05:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )

    callback = loop.run_schedule_job
    try:
        await callback(job)
    finally:
        await _close_components(loop, router, schedule, dream)

    schedule_session = Session.load(
        WorkspaceState(workspace),
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    expected_timestamp = schedule_now.isoformat(timespec="milliseconds")
    assert schedule_session.created_at == schedule_now
    assert schedule_session.updated_at == schedule_now
    assert schedule_session.metadata["title"] == "Schedule clock title"
    assert {message["timestamp"] for message in schedule_session.messages} == {expected_timestamp}


@pytest.mark.asyncio
async def test_schedule_shutdown_during_model_persists_user_and_keeps_job_pending(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Block in the Schedule model.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider(
        schedule_responses=(_response("Unused."),),
        block_schedule_call=1,
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    schedule.start()
    await provider.schedule_block_started.wait()
    await _close_components(loop, router, schedule, dream)

    assert await store.snapshot() == (job,)
    schedule_session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in schedule_session.messages] == [
        ("user", job.message),
        ("assistant", schedule_session.messages[1]["content"]),
    ]
    assert schedule_session.messages[1]["status"] == "interrupted"
    assert schedule_session.messages[1]["error"]["code"] == "turn_cancelled"
    history = read_schedule_history(WorkspaceState(workspace), job.job_id)
    groups = cast(list[dict[str, Any]], history["groups"])
    assert len(groups) == 1
    assert groups[0]["result_state"] == "canceled"
    assert groups[0]["complete"] is True
    assert isinstance(groups[0]["finished_at"], str)
    assert groups[0]["messages"] == schedule_session.messages


@pytest.mark.asyncio
async def test_schedule_shutdown_during_preparation_persists_user(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Block in Schedule context preparation.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider(schedule_responses=(_response("Unused."),))
    context_started = asyncio.Event()
    context_never_completes = asyncio.Event()
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    original_prepare = loop._prepare_agent_run

    async def block_schedule_preparation(
        run_context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        if run_context.current_user["content"] == job.message:
            context_started.set()
            await context_never_completes.wait()
        return await original_prepare(run_context, tool_gateway=tool_gateway)

    object.__setattr__(loop, "_prepare_agent_run", block_schedule_preparation)

    schedule.start()
    await context_started.wait()
    await _close_components(loop, router, schedule, dream)

    assert provider.complete_requests == []
    assert await store.snapshot() == (job,)
    schedule_session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in schedule_session.messages] == [
        ("user", job.message),
        ("assistant", schedule_session.messages[1]["content"]),
    ]
    assert schedule_session.messages[1]["status"] == "interrupted"
    assert schedule_session.messages[1]["error"]["code"] == "turn_cancelled"
    history = read_schedule_history(WorkspaceState(workspace), job.job_id)
    groups = cast(list[dict[str, Any]], history["groups"])
    assert len(groups) == 1
    assert groups[0]["result_state"] == "canceled"
    assert groups[0]["complete"] is True
    assert isinstance(groups[0]["finished_at"], str)
    assert groups[0]["messages"] == schedule_session.messages


@pytest.mark.asyncio
async def test_schedule_failure_logs_one_safe_session_warning(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Fail safely during Schedule preparation.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider()
    failure_started = asyncio.Event()
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    original_prepare = loop._prepare_agent_run

    async def fail_schedule_preparation(
        run_context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        if run_context.current_user["content"] == job.message:
            failure_started.set()
            raise RuntimeError("PRIVATE_SCHEDULE_PREPARATION_BODY")
        return await original_prepare(run_context, tool_gateway=tool_gateway)

    object.__setattr__(loop, "_prepare_agent_run", fail_schedule_preparation)
    capture = capture_diagnostics()
    try:
        schedule.start()
        await failure_started.wait()
        await _wait_until(lambda: schedule.status_snapshot().active_job_count == 0)
    finally:
        await _close_components(loop, router, schedule, dream)
        capture.close()

    assert await _schedule_state(workspace).public_snapshot() == ()
    assert capture.event_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in capture.event_text
    assert "PRIVATE_SCHEDULE_PREPARATION_BODY" not in capture.text
    session_log_text = (state.logs_directory / f"{job.session_id}.log").read_text(encoding="utf-8")
    assert session_log_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in session_log_text
    assert "PRIVATE_SCHEDULE_PREPARATION_BODY" not in session_log_text
    history = read_schedule_history(WorkspaceState(workspace), job.job_id)
    groups = cast(list[dict[str, Any]], history["groups"])
    assert len(groups) == 1
    assert groups[0]["result_state"] == "failure"
    assert groups[0]["complete"] is True
    assert groups[0]["messages"][0]["content"] == job.message
    assert groups[0]["messages"][-1]["error"]["code"] == "model_failed"
    assert "PRIVATE_SCHEDULE_PREPARATION_BODY" not in json.dumps(history)
