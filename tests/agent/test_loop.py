from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadEvent
from threading import Thread
from types import TracebackType
from typing import Any, ClassVar, Literal
from uuid import UUID, uuid4

import pytest
from loguru import logger

import myclaw.agent.context.run_context as compactor_module
import myclaw.agent.loop as loop_module
from myclaw.agent.blackboard import Blackboard
from myclaw.agent.context.budget import estimate_request_tokens
from myclaw.agent.loop import AgentLoop, ConfirmationRequestView, ModelContextOverflowError
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.agent.permission import PermissionSnapshot, RuntimePermissionControl
from myclaw.agent.run_errors import CommittableAgentRunError
from myclaw.agent.runner import AgentRunner, AgentRunnerResult, AgentRunnerRouter
from myclaw.agent.session.backup_store import FileBackupStore
from myclaw.agent.session.session import Session
from myclaw.agent.tools.base import BaseTool
from myclaw.agent.tools.core.exec_host import create_exec_host, resolve_exec_shell
from myclaw.agent.tools.deferred import RUN_BASELINE_TOOL_NAMES
from myclaw.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, TURN_CANCELLED_MESSAGE, ErrorInfo
from myclaw.logging.session import session_log as real_session_log
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
)
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.service import ScheduleService
from myclaw.skills.catalog import (
    LoadedSkill,
    SkillLoader,
    SkillMetadata,
)
from tests.agent.test_context import _FrozenDateTime
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import (
    BlockingTaskFramingRouterAdapter,
    ScriptedFakeProvider,
    StreamScript,
    TaskFramingRouterAdapter,
    collect_foreground_outbound,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


class _Router:
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[str] = []

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        marker = str(messages[-1]["content"])
        self.calls.append(marker)

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            outcome = self._outcomes.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            yield ModelCompleted(response=outcome)

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        raise AssertionError("unexpected direct completion")


class _CapturingRouter(_Router):
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        super().__init__(outcomes)
        self.requests: list[list[dict[str, Any]]] = []

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(deepcopy(list(messages)))
        return super().stream(
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
        )


class _LargeSchemaTool(BaseTool):
    name = "large_schema"
    description = "A test Tool with a deliberately large request schema."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": "x" * 20_000,
            }
        },
    }

    async def execute(self) -> str:
        return "ok"


class _MaxRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        self.calls.append(str(messages[-1]["content"]))
        call = ModelToolCall(
            id=f"call-{len(self.calls)}",
            name="unknown_tool",
            arguments="{}",
        )

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(response=_response("working", tool_call=call))

        return replay()


class _BlockingRouter(_Router):
    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(())
        self._started = started

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation
        self.calls.append("call")
        first = len(self.calls) == 1

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if first:
                self._started.set()
                await asyncio.Event().wait()
            yield ModelCompleted(response=_response("after cancellation"))

        return replay()


class _ConcurrentTitleRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())
        self.chat_started = asyncio.Event()
        self.release_chat = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"
        self.calls.append("title" if is_title else "chat")

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                yield ModelCompleted(response=_response("Generated while chat blocked"))
                return
            self.chat_started.set()
            await self.release_chat.wait()
            yield ModelCompleted(response=_response("foreground completed"))

        return replay()


class _SlowTitleLogRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.chat_calls = 0

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                self.title_started.set()
                await self.release_title.wait()
                yield ModelCompleted(response=_response("Slow title"))
                return
            self.chat_calls += 1
            if self.chat_calls == 2:
                logger.warning("Second foreground marker")
            yield ModelCompleted(response=_response(f"foreground {self.chat_calls}"))

        return replay()


class _EventRouter(_Router):
    def __init__(self, streams: Sequence[Sequence[ModelStreamEvent]]) -> None:
        super().__init__(())
        self._streams = deque(tuple(stream) for stream in streams)

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        self.calls.append(str(messages[-1]["content"]))
        events = self._streams.popleft()

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            for event in events:
                yield event

        return replay()


class _TitleBehaviorRouter(_Router):
    def __init__(
        self,
        foreground: Sequence[ModelResponse],
        *,
        title: ModelResponse,
        delay_title: bool = False,
        block_first_foreground: bool = False,
    ) -> None:
        super().__init__(())
        self._foreground = deque(foreground)
        self._title = title
        self._delay_title = delay_title
        self._block_first_foreground = block_first_foreground
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.foreground_started = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"
        self.calls.append("title" if is_title else "chat")

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                self.title_started.set()
                if self._delay_title:
                    await self.release_title.wait()
                yield ModelCompleted(response=self._title)
                return
            self.foreground_started.set()
            if self._block_first_foreground:
                self._block_first_foreground = False
                await asyncio.Event().wait()
            yield ModelCompleted(response=self._foreground.popleft())

        return replay()


def test_agent_loop_constructor_is_the_generation_composition_boundary() -> None:
    assert tuple(inspect.signature(AgentLoop).parameters) == (
        "workspace_path",
        "workspace_state",
        "agent_home",
        "configuration",
        "bus",
        "schedule_service",
        "model_router",
        "memory_manager",
        "session_id",
        "now",
        "new_uuid",
        "monotonic_now",
        "exec_host",
        "permission_control",
        "configured_schedule_level",
        "mcp_tools",
        "mcp_keywords",
    )
    assert tuple(inspect.signature(AgentLoop.close).parameters) == ("self",)


def test_agent_loop_rejects_workspace_state_owned_by_another_workspace(
    tmp_path: Path,
) -> None:
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()

    with pytest.raises(
        ValueError,
        match="Agent Loop Workspace State must belong to the Workspace",
    ):
        _runtime(
            tmp_path,
            _Router(()),
            constructor_workspace_path=other_workspace,
        )


def test_agent_loop_rejects_memory_manager_owned_by_another_workspace_state(
    tmp_path: Path,
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    other_state = WorkspaceState(other_workspace)
    other_state.initialize(agent_home_root=agent_home.path)

    with pytest.raises(
        ValueError,
        match="Agent Loop Memory Manager must belong to the Workspace State",
    ):
        _runtime(
            tmp_path,
            _Router(()),
            constructor_memory_manager=MemoryManager(other_state),
        )


@pytest.mark.asyncio
async def test_agent_loop_constructs_each_generation_collaborator_once_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        "session": 0,
        "skill_loader": 0,
        "skill_load": 0,
        "context_builder": 0,
        "tool_gateway": 0,
        "runner": 0,
        "persist": 0,
    }
    constructor_args: dict[str, list[tuple[Any, ...]]] = {
        name: [] for name in ("context_builder", "tool_gateway", "runner")
    }

    original_create = Session.create
    original_persist = Session.persist

    def recording_create(*args: Any, **kwargs: Any) -> Session:
        counts["session"] += 1
        return original_create(*args, **kwargs)

    def recording_persist(session: Session) -> None:
        counts["persist"] += 1
        original_persist(session)

    class RecordingSkillLoader(SkillLoader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            counts["skill_loader"] += 1
            super().__init__(*args, **kwargs)

        def load(
            self,
            *,
            validate: Callable[[tuple[LoadedSkill, ...]], None] | None = None,
        ) -> None:
            counts["skill_load"] += 1
            super().load(validate=validate)

    def record_factory(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def recording(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            constructor_args[name].append(args)
            return original(*args, **kwargs)

        return recording

    monkeypatch.setattr(Session, "create", staticmethod(recording_create))
    monkeypatch.setattr(Session, "persist", recording_persist)
    monkeypatch.setattr(loop_module, "SkillLoader", RecordingSkillLoader)
    for name, attribute in (
        ("context_builder", "ContextBuilder"),
        ("tool_gateway", "ToolGateway"),
        ("runner", "AgentRunner"),
    ):
        original = getattr(loop_module, attribute)
        monkeypatch.setattr(loop_module, attribute, record_factory(name, original))

    router = _Router(())
    tasks_before = asyncio.all_tasks()
    loop, session, _bus = _runtime(tmp_path, router, task_framing_outcomes=None)

    assert counts == {
        "session": 1,
        "skill_loader": 1,
        "skill_load": 1,
        "context_builder": 1,
        "tool_gateway": 1,
        "runner": 0,
        "persist": 0,
    }
    assert asyncio.all_tasks() == tasks_before
    assert router.calls == []
    assert constructor_args["runner"] == []
    assert isinstance(loop._model_router, TaskFramingRouterAdapter)
    assert loop._model_router._delegate is router
    assert not (session.workspace_state.sessions_directory / f"{session.session_id}.jsonl").exists()

    await loop.abort()


def _response(
    content: str,
    *,
    tool_call: ModelToolCall | None = None,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason="stop",
    )


def _status_context_usage(
    *,
    selected_route: str,
    provider_id: str,
    model: str,
    context_window: int,
    max_output: int,
) -> dict[str, object]:
    return {
        "requested_route": "chat",
        "selected_route": selected_route,
        "provider_id": provider_id,
        "model": model,
        "context_window": context_window,
        "max_output": max_output,
        "anchor_estimated_tokens": 20,
        "estimator_version": "utf8-bytes-div4-v1",
        "run_projected_tokens": 80,
        "run_projection_source": "estimated",
    }


def _framing_response(
    blackboard: Blackboard,
    *,
    input_tokens: int,
    output_tokens: int,
) -> ModelResponse:
    return _response(
        json.dumps(
            {
                "action": "replace",
                "task_goal": blackboard.goal,
                "completion_boundary": blackboard.completion_boundary,
            },
            separators=(",", ":"),
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _runtime(
    tmp_path: Path,
    router: AgentRunnerRouter,
    *,
    task_framing_outcomes: Sequence[ModelResponse | BaseException] | None = (),
    title_prompt: str | None = None,
    skill_loader: SkillLoader | None = None,
    monotonic_now: Callable[[], float] | None = None,
    config_text: str | None = None,
    mcp_tools: Sequence[BaseTool] = (),
    prepare_session: Callable[[WorkspaceState], str] | None = None,
    constructor_workspace_path: Path | None = None,
    constructor_memory_manager: MemoryManager | None = None,
) -> tuple[AgentLoop, Session, MessageBus]:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home.path)
    selected_session_id = None if prepare_session is None else prepare_session(state)
    (agent_home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG if config_text is None else config_text,
        encoding="utf-8",
    )
    configuration = ConfigLoader(agent_home).load()

    async def execute_user_job(job: ScheduleJob) -> None:
        del job

    async def execute_dream() -> object:
        return None

    schedule = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    bus = MessageBus()
    model_router = (
        router
        if isinstance(router, TaskFramingRouterAdapter)
        else TaskFramingRouterAdapter(router, task_framing_outcomes)
    )
    model_router.bind_configuration(configuration)
    loop = AgentLoop(
        workspace_path=(
            workspace if constructor_workspace_path is None else constructor_workspace_path
        ),
        workspace_state=state,
        agent_home=agent_home,
        configuration=configuration,
        bus=bus,
        schedule_service=schedule,
        model_router=model_router,
        memory_manager=(
            MemoryManager(state)
            if constructor_memory_manager is None
            else constructor_memory_manager
        ),
        session_id=selected_session_id,
        now=_Clock().now,
        new_uuid=uuid4,
        monotonic_now=(lambda: 0.0) if monotonic_now is None else monotonic_now,
        exec_host=create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell)),
        permission_control=RuntimePermissionControl(configuration.runtime.permission_level),
        mcp_tools=mcp_tools,
    )
    if title_prompt is None:

        def disable_title(_session: Session, _content: str) -> None:
            return None

        object.__setattr__(loop, "_start_title_if_needed", disable_title)
    else:

        def build_title_messages(content: str) -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": title_prompt},
                {"role": "user", "content": content},
            ]

        object.__setattr__(loop._context_builder, "build_title_messages", build_title_messages)
    if skill_loader is not None:
        loop._skill_loader = skill_loader
        loop._context_builder._skill_loader = skill_loader
    return loop, loop.session, bus


def _planner_skill_loader(tmp_path: Path) -> SkillLoader:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    return loader


@pytest.mark.asyncio
async def test_agent_loop_status_projection_starts_uptime_only_after_activation(
    tmp_path: Path,
) -> None:
    monotonic_calls = 0

    def monotonic_now() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        return 42.5

    loop, _session, _bus = _runtime(
        tmp_path,
        _Router(()),
        monotonic_now=monotonic_now,
    )

    assert monotonic_calls == 0
    loop.preflight()
    assert monotonic_calls == 0

    await asyncio.gather(loop.start(), loop.start())

    assert monotonic_calls == 1
    assert loop.runtime_status_input().generation_started_at == 42.5
    await loop.start()
    assert monotonic_calls == 1
    await loop.close()


def test_agent_loop_preflight_uses_the_deferred_baseline_without_unused_tool_schemas(
    tmp_path: Path,
) -> None:
    config = MINIMAL_VALID_CONFIG.replace(
        "context_window = 200000", "context_window = 1024"
    ).replace("max_output = 8192", "max_output = 128")
    loop, _session, _bus = _runtime(
        tmp_path,
        _Router(()),
        config_text=config,
        mcp_tools=(_LargeSchemaTool(),),
    )

    loop.preflight()

    assert tuple(schema["function"]["name"] for schema in loop.tool_schemas) == (
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "exec",
        "tool_search",
    )
    assert all(
        schema["function"]["name"] != "large_schema"
        for schema in loop.runtime_status_input().projected_tools
    )


@pytest.mark.asyncio
async def test_agent_loop_injects_persisted_action_summary_into_foreground_and_status_context(
    tmp_path: Path,
) -> None:
    router = _CapturingRouter((_response("done"),))
    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
    )
    action_summary = "- Updated the Session context contract."
    session.update_metadata(summary=action_summary)

    status = loop.runtime_status_input()
    assert status.projected_messages[1] == {
        "role": "user",
        "content": action_summary,
    }

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("Continue the work."))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert router.requests[0][1] == {"role": "user", "content": action_summary}
    assert "Continue the work." in str(router.requests[0][-1]["content"])


@pytest.mark.asyncio
async def test_agent_loop_does_not_compact_by_message_count(
    tmp_path: Path,
) -> None:
    class CompactionRouter(_Router):
        def __init__(self) -> None:
            super().__init__(
                (
                    _response("First answer."),
                    _response("Second answer."),
                    _response("Third answer."),
                )
            )
            self.stream_requests: list[list[dict[str, Any]]] = []
            self.memory_requests: list[list[dict[str, Any]]] = []
            self.memory_responses = deque((_response("Fact summary."), _response("None")))

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(deepcopy(list(messages)))
            return super().stream(
                route,
                messages=messages,
                tools=tools,
                continuation=continuation,
            )

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del route, tools, continuation
            self.memory_requests.append(deepcopy(list(messages)))
            return self.memory_responses.popleft()

    router = CompactionRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
        config_text=MINIMAL_VALID_CONFIG,
        task_framing_outcomes=(),
    )
    previous_summary = "- Preserved the previous compacted work."
    session.update_metadata(summary=previous_summary)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("First question."))
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("Second question."))
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("Third question."))
        await _terminals(bus, 1)
        status = loop.runtime_status_input()
    finally:
        await loop.close()

    assert router.memory_requests == []
    assert session.metadata["summary"] == previous_summary
    assert session.last_compacted == 0
    assert all(message.get("content") != "None" for message in session.messages)
    assert previous_summary in json.dumps(status.projected_messages)

    assert len(router.stream_requests) == 3
    third_request = router.stream_requests[2]
    assert all(message.get("content") != "None" for message in third_request)
    assert "Third question." in str(third_request[-1]["content"])

    restored = Session.load(session.workspace_state, session.session_id)
    assert restored.metadata["summary"] == previous_summary
    assert all(message.get("content") != "None" for message in restored.messages)


@pytest.mark.asyncio
async def test_agent_loop_restores_empty_action_summary_for_foreground_status_and_preflight(
    tmp_path: Path,
) -> None:
    def prepare_session(state: WorkspaceState) -> str:
        restored = Session.create(state, now=_Clock().now, new_uuid=uuid4)
        seed_session_state(
            restored,
            messages=[
                {
                    "role": "user",
                    "content": "Persisted foreground history.",
                    "timestamp": "2026-08-21T12:00:00.000+00:00",
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
                "summary": "",
            },
            last_compacted=0,
        )
        restored.close()
        return restored.session_id

    router = _CapturingRouter((_response("Restored answer."),))
    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=(),
        prepare_session=prepare_session,
    )

    assert session.metadata["summary"] == ""
    status = loop.runtime_status_input()
    retained_messages = status.projected_messages[1:]
    assert retained_messages[0] == {
        "role": "user",
        "content": "Persisted foreground history.",
    }
    assert all(message.get("content") not in {"", "None"} for message in retained_messages)
    loop.preflight()

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("Continue the restored work."))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert router.requests[0][1] == {
        "role": "user",
        "content": "Persisted foreground history.",
    }
    assert all(message.get("content") != "None" for message in router.requests[0])
    restored = Session.load(session.workspace_state, session.session_id)
    assert restored.metadata["summary"] == ""
    assert all(message.get("content") != "None" for message in restored.messages)


@pytest.mark.asyncio
async def test_agent_loop_restores_action_summary_for_foreground_and_preflight_context(
    tmp_path: Path,
) -> None:
    action_summary = "- Restored the foreground Session summary."

    def prepare_session(state: WorkspaceState) -> str:
        restored = Session.create(state, now=_Clock().now, new_uuid=uuid4)
        seed_session_state(
            restored,
            messages=[
                {
                    "role": "user",
                    "content": "Persisted foreground history.",
                    "timestamp": "2026-08-21T12:00:00.000+00:00",
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
                "summary": action_summary,
            },
            last_compacted=0,
        )
        restored.close()
        return restored.session_id

    router = _CapturingRouter((_response("Restored answer."),))
    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=(),
        prepare_session=prepare_session,
    )

    assert session.metadata["summary"] == action_summary
    status = loop.runtime_status_input()
    assert status.projected_messages[1] == {
        "role": "user",
        "content": action_summary,
    }
    loop.preflight()

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("Continue the restored work."))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert router.requests[0][1] == {
        "role": "user",
        "content": action_summary,
    }
    assert "Continue the restored work." in str(router.requests[0][-1]["content"])
    restored = Session.load(session.workspace_state, session.session_id)
    assert restored.metadata["summary"] == action_summary
    assert all(message.get("content") != action_summary for message in restored.messages)


def test_agent_loop_preflight_budget_includes_action_summary(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    control_root.mkdir()
    control, _control_session, _control_bus = _runtime(control_root, _Router(()))
    control.preflight()

    summary_root = tmp_path / "summary"
    summary_root.mkdir()
    loop, session, _bus = _runtime(summary_root, _Router(()))
    session.update_metadata(summary="- " + ("x" * 30_000))

    with pytest.raises(ModelContextOverflowError) as raised:
        loop.preflight()

    assert raised.value.error.code == "model_context_overflow"


@pytest.mark.asyncio
async def test_foreground_context_and_runner_share_exactly_one_run_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blackboard = Blackboard(goal="Run one turn", completion_boundary="Return one answer")
    loop, _session, bus = _runtime(
        tmp_path,
        _Router((_response("done"),)),
        task_framing_outcomes=(_framing_response(blackboard, input_tokens=1, output_tokens=1),),
    )
    created_gateways: list[ToolGateway] = []
    context_gateways: list[ToolGateway] = []
    runner_gateways: list[ToolGateway] = []
    context_snapshots: list[object] = []
    gateway_snapshots: list[PermissionSnapshot] = []
    original_new_run_gateway = loop._new_run_gateway
    original_prepare = loop._prepare_agent_run
    original_run = AgentRunner.run
    original_build_foreground = loop._context_builder.build_foreground_messages

    def new_run_gateway(
        *,
        excluded_names: Sequence[str] = (),
        permission_snapshot: PermissionSnapshot | None = None,
        permission_context: object = None,
    ) -> ToolGateway:
        assert permission_snapshot is not None
        assert permission_context is None
        gateway_snapshots.append(permission_snapshot)
        gateway = original_new_run_gateway(
            excluded_names=excluded_names,
            permission_snapshot=permission_snapshot,
        )
        created_gateways.append(gateway)
        return gateway

    async def prepare(
        context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        context_gateways.append(tool_gateway)
        return await original_prepare(context, tool_gateway=tool_gateway)

    async def run(*args: Any, **kwargs: Any) -> AgentRunnerResult:
        gateway = kwargs["tool_gateway"]
        assert isinstance(gateway, ToolGateway)
        runner_gateways.append(gateway)
        return await original_run(*args, **kwargs)

    def build_foreground(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        context_snapshots.append(kwargs["permission_snapshot"])
        return original_build_foreground(*args, **kwargs)

    object.__setattr__(loop, "_new_run_gateway", new_run_gateway)
    object.__setattr__(loop, "_prepare_agent_run", prepare)
    object.__setattr__(loop._context_builder, "build_foreground_messages", build_foreground)
    monkeypatch.setattr(AgentRunner, "run", run)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("one foreground turn"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(created_gateways) == 1
    assert len(context_gateways) == 1
    assert len(runner_gateways) == 1
    assert created_gateways[0] is context_gateways[0] is runner_gateways[0]
    assert created_gateways[0].exposed_names == RUN_BASELINE_TOOL_NAMES
    foreground_snapshots = [snapshot for snapshot in context_snapshots if snapshot is not None]
    assert foreground_snapshots
    assert all(snapshot is foreground_snapshots[0] for snapshot in foreground_snapshots)
    assert gateway_snapshots == [foreground_snapshots[0]]
    assert foreground_snapshots[0] is created_gateways[0]._permission_context.snapshot


@pytest.mark.asyncio
async def test_foreground_captures_once_before_title_and_framing_and_switches_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_blackboard = Blackboard(goal="First", completion_boundary="Finish first")
    second_blackboard = Blackboard(goal="Second", completion_boundary="Finish second")
    loop, _session, bus = _runtime(
        tmp_path,
        _Router((_response("first"), _response("second"))),
        task_framing_outcomes=(
            _framing_response(first_blackboard, input_tokens=1, output_tokens=1),
            _framing_response(second_blackboard, input_tokens=1, output_tokens=1),
        ),
    )
    events: list[str] = []
    captured: list[PermissionSnapshot] = []
    context_snapshots: list[PermissionSnapshot] = []
    gateway_snapshots: list[PermissionSnapshot] = []
    control = loop._permission_control
    original_snapshot = RuntimePermissionControl.snapshot
    original_generate = Blackboard.generate
    original_build_foreground = loop._context_builder.build_foreground_messages
    original_new_run_gateway = loop._new_run_gateway

    def snapshot(
        selected_control: RuntimePermissionControl,
        exec_shell: object,
    ) -> PermissionSnapshot:
        events.append(f"snapshot:{selected_control.current()}")
        result = original_snapshot(selected_control, exec_shell)  # type: ignore[arg-type]
        captured.append(result)
        if len(captured) == 1:
            selected_control.select("read-only")
        return result

    async def generate(
        cls: type[Blackboard],
        router: AgentRunnerRouter,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> object:
        del cls
        events.append("framing")
        return await original_generate(
            router,
            previous=previous,
            last_assistant_content=last_assistant_content,
            current_user_input=current_user_input,
        )

    def start_title(_session: Session, _content: str) -> None:
        events.append("title")

    def build_foreground(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        current = kwargs.get("permission_snapshot")
        if isinstance(current, PermissionSnapshot):
            context_snapshots.append(current)
        return original_build_foreground(*args, **kwargs)

    def new_run_gateway(
        *,
        excluded_names: Sequence[str] = (),
        permission_snapshot: PermissionSnapshot | None = None,
        permission_context: object = None,
    ) -> ToolGateway:
        assert permission_snapshot is not None
        assert permission_context is None
        gateway_snapshots.append(permission_snapshot)
        return original_new_run_gateway(
            excluded_names=excluded_names,
            permission_snapshot=permission_snapshot,
        )

    monkeypatch.setattr(RuntimePermissionControl, "snapshot", snapshot)
    monkeypatch.setattr(Blackboard, "generate", classmethod(generate))
    object.__setattr__(loop, "_start_title_if_needed", start_title)
    object.__setattr__(loop._context_builder, "build_foreground_messages", build_foreground)
    object.__setattr__(loop, "_new_run_gateway", new_run_gateway)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("first input"))
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("second input"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert [snapshot.level for snapshot in captured] == ["workspace-write", "read-only"]
    assert control.current() == "read-only"
    assert gateway_snapshots == captured
    assert context_snapshots
    assert context_snapshots[0] is captured[0]
    assert context_snapshots[-1] is captured[1]
    assert all(snapshot is captured[0] or snapshot is captured[1] for snapshot in context_snapshots)
    assert events[:3] == ["snapshot:workspace-write", "title", "framing"]
    assert events.count("snapshot:workspace-write") == 1
    assert events.count("snapshot:read-only") == 1
    assert events.count("framing") == 2


@pytest.mark.asyncio
async def test_agent_loop_failed_activation_does_not_publish_or_duplicate_consumer(
    tmp_path: Path,
) -> None:
    monotonic_calls = 0

    def monotonic_now() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls == 1:
            raise RuntimeError("monotonic clock failed")
        return 84.5

    loop, _session, _bus = _runtime(
        tmp_path,
        _Router(()),
        monotonic_now=monotonic_now,
    )
    loop.preflight()

    with pytest.raises(RuntimeError, match="monotonic clock failed"):
        await loop.start()

    assert loop._consumer_task is None
    assert loop._generation_started_at is None
    assert loop._started is False

    await asyncio.gather(loop.start(), loop.start())

    consumer = loop._consumer_task
    assert consumer is not None
    assert loop._generation_started_at == 84.5
    assert monotonic_calls == 2
    await loop.start()
    assert loop._consumer_task is consumer
    assert monotonic_calls == 2
    await loop.close()


@pytest.mark.asyncio
async def test_replacement_barrier_blocks_a_late_foreground_commit_until_released(
    tmp_path: Path,
) -> None:
    class ObservableCommitLock(asyncio.Lock):
        def __init__(self) -> None:
            super().__init__()
            self.acquire_attempted = asyncio.Event()
            self.commit_completed = asyncio.Event()

        async def acquire(self) -> Literal[True]:
            self.acquire_attempted.set()
            return await super().acquire()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb
            self.release()
            self.commit_completed.set()

    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(tmp_path, router)
    commit_gate = ObservableCommitLock()
    loop._foreground_commit_gate = commit_gate
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing turn suppresses title work.",
                "timestamp": "2026-08-21T12:00:00.000+00:00",
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
            "summary": "",
        },
        last_compacted=0,
    )
    initial_messages = tuple(session.messages)
    await loop.start()
    await loop._bus.put_inbound(InboundMessage(content="late foreground"))
    await asyncio.wait_for(router.chat_started.wait(), timeout=1)

    await loop._pause_for_replacement()
    commit_gate.acquire_attempted.clear()
    router.release_chat.set()
    await asyncio.wait_for(commit_gate.acquire_attempted.wait(), timeout=1)

    assert tuple(session.messages) == initial_messages
    assert loop.control.has_active_run

    await loop._release_replacement_barrier(resume_inbound=True)
    await asyncio.wait_for(commit_gate.commit_completed.wait(), timeout=1)

    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "Existing turn suppresses title work.",
        "late foreground",
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_agent_loop_status_projection_is_one_read_immutable_and_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _Router(())
    loop, session, _bus = _runtime(tmp_path, router)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Status snapshot input",
                "timestamp": "2026-08-21T12:00:00.000+00:00",
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
            "summary": "",
            "blackboard": {
                "goal": "Project the persisted task state.",
                "completion_boundary": "The status request contains this Blackboard.",
            },
        },
        last_compacted=0,
    )

    class SessionAccessSpy:
        def __init__(self) -> None:
            self.calls = {
                "session_id": 0,
                "messages": 0,
                "metadata": 0,
                "last_compacted": 0,
            }

        @property
        def session_id(self) -> str:
            self.calls["session_id"] += 1
            return session.session_id

        @property
        def messages(self) -> list[dict[str, Any]]:
            self.calls["messages"] += 1
            return session.messages

        @property
        def metadata(self) -> dict[str, Any]:
            self.calls["metadata"] += 1
            return session.metadata

        @property
        def last_compacted(self) -> int:
            self.calls["last_compacted"] += 1
            return session.last_compacted

    spy = SessionAccessSpy()
    messages_before = deepcopy(session.messages)
    metadata_before = deepcopy(session.metadata)
    tasks_before = set(asyncio.all_tasks())
    selector_inputs: list[Sequence[dict[str, Any]]] = []
    original_selector = compactor_module.latest_main_agent_usage_anchor

    def observe_selector(
        messages: Sequence[dict[str, Any]],
    ) -> tuple[object, dict[str, int]] | None:
        selector_inputs.append(messages)
        return original_selector(messages)

    monkeypatch.setattr(loop_module, "latest_main_agent_usage_anchor", observe_selector)
    object.__setattr__(loop, "_session", spy)
    try:
        projection = loop.runtime_status_input()
    finally:
        object.__setattr__(loop, "_session", session)

    assert spy.calls == {
        "session_id": 1,
        "messages": 1,
        "metadata": 1,
        "last_compacted": 1,
    }
    assert session.messages == messages_before
    assert session.metadata == metadata_before
    assert session._pending_persist is None
    assert set(asyncio.all_tasks()) == tasks_before
    assert router.calls == []
    assert selector_inputs == [session.messages]
    assert loop._consumer_task is None
    assert isinstance(projection.projected_messages, tuple)
    assert isinstance(projection.cumulative_usage, tuple)
    assert all(isinstance(value, dict) for value in projection.projected_messages)
    assert "Project the persisted task state." in str(projection.projected_messages)
    with pytest.raises(FrozenInstanceError):
        projection.session_message_count = 99  # type: ignore[misc]
    await loop.close()


async def _terminals(bus: MessageBus, count: int) -> list[OutboundMessage]:
    terminals: list[OutboundMessage] = []
    while len(terminals) < count:
        message = await bus.get_outbound()
        if message.metadata.get("_streamed") is True:
            terminals.append(message)
    return terminals


def test_agent_loop_status_uses_the_configured_static_default_route(tmp_path: Path) -> None:
    router = _Router(())
    loop, session, _bus = _runtime(tmp_path, router)

    status = loop.runtime_status_input()

    assert (
        status.requested_route,
        status.selected_route,
        status.provider_id,
        status.model,
        status.context_window,
        status.max_output,
    ) == ("chat", "default", "primary", "small-model", 8192, 1024)
    assert status.chat_model == "primary/small-model"
    assert router.calls == []
    assert session._pending_persist is None


@pytest.mark.asyncio
async def test_status_treats_the_latest_assistant_as_the_usage_provenance_boundary(
    tmp_path: Path,
) -> None:
    router = _Router(())
    loop, session, _bus = _runtime(tmp_path, router)
    timestamp = "2026-08-21T12:00:00.000+00:00"
    older_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Older request", "timestamp": timestamp},
        {
            "role": "assistant",
            "content": "Older response",
            "timestamp": timestamp,
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            "context_usage": _status_context_usage(
                selected_route="default",
                provider_id="primary",
                model="small-model",
                context_window=8192,
                max_output=1024,
            ),
        },
    ]
    seed_session_state(
        session,
        messages=older_messages,
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    management = management_service(
        AgentHome(tmp_path / "agent-home"),
        current_agent_loop=lambda: loop,
        workspace_state=session.workspace_state,
    )

    reported = await management.status()

    seed_session_state(
        session,
        messages=[
            *older_messages,
            {"role": "user", "content": "New request", "timestamp": timestamp},
            {
                "role": "assistant",
                "content": "New response without provenance",
                "timestamp": timestamp,
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                },
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 2,
                "input_tokens": 120,
                "output_tokens": 15,
                "total_tokens": 135,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    estimated = await management.status()

    assert reported.projection_source == "reported_delta"
    assert estimated.projection_source == "estimated"
    assert router.calls == []
    assert session._pending_persist is None
    await loop.close()


@pytest.mark.asyncio
async def test_status_after_dynamic_fallback_projects_the_next_configured_chat_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = """[models.providers.chat-provider]
protocol = "openai-compatible"
base_url = "https://chat.example/v1"
api_key = "chat-secret"
models = ["chat-model"]

[models.providers.default-provider]
protocol = "anthropic"
base_url = "https://default.example/v1"
api_key = "default-secret"
models = ["default-model"]

[runtime]
compact_ratio = 0.9

[models.routes.chat]
provider_id = "chat-provider"
model = "chat-model"
context_window = 32000
max_output = 2048
temperature = 0
timeout = 30

[models.routes.default]
provider_id = "default-provider"
model = "default-model"
context_window = 16000
max_output = 1024
temperature = 0
timeout = 30
"""
    router_home = AgentHome(tmp_path / "router-home")
    router_home.initialize()
    (router_home.path / "config.toml").write_text(config, encoding="utf-8")
    router_configuration = ConfigLoader(router_home).load()
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(ErrorInfo("provider_auth_error", "Use fallback.")),
            ),
            StreamScript(events=(ModelCompleted(response=_response("Configured chat.")),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("Fallback.")),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    model_router = ModelRouter(
        configuration=router_configuration,
        provider_factory=lambda provider: providers[provider.provider_id],
        jitter=None,
    )
    loop, session, bus = _runtime(tmp_path, model_router, config_text=config)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("Use fallback for this run."))
        await _terminals(bus, 1)
        assert len(chat_provider.stream_requests) == 1
        assert len(default_provider.stream_requests) == 1
        chat_attempt = chat_provider.stream_requests[0]
        fallback_attempt = default_provider.stream_requests[0]
        assert (chat_attempt.model, chat_attempt.max_output) == ("chat-model", 2048)
        assert (fallback_attempt.model, fallback_attempt.max_output) == ("default-model", 1024)
        assert persist_calls == [None]
        fallback_context = session.messages[-1]["context_usage"]
        assert fallback_context["selected_route"] == "default"
        assert fallback_context["provider_id"] == "default-provider"

        provider_counts = (
            len(chat_provider.stream_requests),
            len(default_provider.stream_requests),
        )
        persist_count = len(persist_calls)
        status_input = loop.runtime_status_input()
        status = await management_service(
            AgentHome(tmp_path / "agent-home"),
            current_agent_loop=lambda: loop,
            workspace_state=session.workspace_state,
        ).status()

        assert (
            status_input.requested_route,
            status_input.selected_route,
            status_input.provider_id,
            status_input.model,
            status_input.context_window,
            status_input.max_output,
        ) == ("chat", "chat", "chat-provider", "chat-model", 32000, 2048)
        assert status_input.chat_model == "chat-provider/chat-model"
        assert status.projection_source == "estimated"
        assert (
            len(chat_provider.stream_requests),
            len(default_provider.stream_requests),
        ) == provider_counts
        assert len(persist_calls) == persist_count

        await bus.put_inbound(InboundMessage("Start the next independent run."))
        await _terminals(bus, 1)

        assert len(chat_provider.stream_requests) == 2
        assert len(default_provider.stream_requests) == 1
        next_context = session.messages[-1]["context_usage"]
        assert (
            next_context["requested_route"],
            next_context["selected_route"],
            next_context["provider_id"],
            next_context["model"],
            next_context["context_window"],
            next_context["max_output"],
        ) == (
            status_input.requested_route,
            status_input.selected_route,
            status_input.provider_id,
            status_input.model,
            status_input.context_window,
            status_input.max_output,
        )
        next_attempt = chat_provider.stream_requests[1]
        assert (next_attempt.model, next_attempt.max_output) == ("chat-model", 2048)
        assert persist_calls == [None, None]
    finally:
        await loop.close()


def test_agent_loop_exposes_public_control_seam(tmp_path: Path) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router((_response("unused"),)))

    assert loop.control is loop
    assert loop.control.has_active_run is False


def test_agent_loop_reload_returns_the_current_loader_metadata_and_reuses_generation_state(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, session, bus = _runtime(tmp_path, _Router(()))
    loader = loop._skill_loader
    initial_session = loop.session
    initial_context_loader = loop._context_builder._skill_loader
    before_messages = deepcopy(session.messages)
    bus_operations: list[str] = []

    async def record_bus_operation(name: str) -> None:
        bus_operations.append(name)

    object.__setattr__(bus, "reset", lambda: record_bus_operation("reset"))
    object.__setattr__(
        bus,
        "pause_inbound_delivery",
        lambda: record_bus_operation("pause"),
    )
    object.__setattr__(
        bus,
        "resume_inbound_delivery",
        lambda: record_bus_operation("resume"),
    )

    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )

    metadata = loop.reload_skill()

    assert loop is loop.control
    assert loop.session is initial_session is session
    assert loop._bus is bus
    assert loop._skill_loader is loader is initial_context_loader
    assert session.messages == before_messages
    assert bus_operations == []
    assert metadata == loader.metadata
    assert tuple(item.name for item in metadata) == ("reviewer",)
    assert loader.get("planner") is None
    invocation = loader.resolve_manual("/reviewer request")
    assert invocation is not None
    assert invocation.metadata == metadata[0]
    assert invocation.body.splitlines()[-1] == "new body"


def test_agent_loop_reload_rejects_an_always_loaded_budget_overrun_before_publication(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\nold body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9",
        "compact_ratio = 0.9\nenable_skill_always_load = true",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()), config_text=config)
    loader = loop._skill_loader
    before_skills = loader.skills
    before_metadata = loader.metadata
    before_invocation = loader.resolve_manual("/always request")
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\n"
        + ("oversized body\n" * 20_000),
        encoding="utf-8",
    )

    with pytest.raises(ModelContextOverflowError):
        loop.reload_skill()

    assert loader.skills == before_skills
    assert loader.metadata == before_metadata
    assert loader.get("always") is before_skills[0]
    assert loader.resolve_manual("/always request") == before_invocation


def test_reload_validator_candidate_projection_is_isolated_until_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\nalways: true\n---\nold body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9",
        "compact_ratio = 0.9\nenable_skill_always_load = true",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()), config_text=config)
    loader = loop._skill_loader
    before_skills = loader.skills
    before_invocation = loader.resolve_manual("/planner request")
    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\nalways: true\n---\nnew body\n",
        encoding="utf-8",
    )

    validation_started = ThreadEvent()
    release_validation = ThreadEvent()
    candidate_prompts: list[str] = []

    def block_candidate_estimate(
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> int:
        del tools
        candidate_prompts.append(str(messages[0]["content"]))
        public_messages = loop._context_builder.build_status_messages(
            (), session_id=loop.session.session_id
        )
        assert '"name":"planner"' in public_messages[0]["content"]
        assert '"name":"reviewer"' not in public_messages[0]["content"]
        validation_started.set()
        if not release_validation.wait(timeout=5):
            raise AssertionError("candidate validation was not released")
        return 0

    monkeypatch.setattr(loop_module, "estimate_request_tokens", block_candidate_estimate)
    published: list[tuple[SkillMetadata, ...]] = []
    failures: list[BaseException] = []

    def reload_in_thread() -> None:
        try:
            published.append(loop.reload_skill())
        except BaseException as error:
            failures.append(error)

    reload_thread = Thread(target=reload_in_thread)
    reload_thread.start()
    try:
        assert validation_started.wait(timeout=5)
        assert len(candidate_prompts) == 1
        assert '"name":"reviewer"' in candidate_prompts[0]
        assert '"name":"planner"' not in candidate_prompts[0]
        assert loader.skills == before_skills
        assert loader.resolve_manual("/planner request") == before_invocation
        assert loader.resolve_manual("/reviewer request") is None

        public_messages = loop._context_builder.build_status_messages(
            (),
            session_id=loop.session.session_id,
        )
        assert '"name":"planner"' in str(public_messages[0]["content"])
        assert '"name":"reviewer"' not in str(public_messages[0]["content"])
    finally:
        release_validation.set()
        reload_thread.join(timeout=5)

    assert not reload_thread.is_alive()
    assert failures == []
    assert published == [loader.metadata]
    assert tuple(item.name for item in loader.metadata) == ("reviewer",)
    assert loader.resolve_manual("/planner request") is None
    assert loader.resolve_manual("/reviewer request") is not None


@pytest.mark.parametrize("over_budget", [False, True], ids=["published", "rejected"])
def test_reload_candidate_validation_restores_the_active_skill_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    over_budget: bool,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: active\ndescription: Active snapshot\n---\nactive body\n",
        encoding="utf-8",
    )
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    loader = loop._skill_loader
    builder = loop._context_builder
    active_skills = loader.skills
    instruction.write_text(
        "---\nname: published\ndescription: Published snapshot\n---\npublished body\n",
        encoding="utf-8",
    )
    loader.load()
    published_skills = loader.skills
    published_metadata = loader.metadata
    published_invocation = loader.resolve_manual("/published request")
    candidate_document = (
        "---\nname: candidate\ndescription: Candidate snapshot\n---\ncandidate body\n"
    )
    instruction.write_text(candidate_document, encoding="utf-8", newline="")
    candidate_prompts: list[str] = []
    chat_route = loop._configuration.resolve_route("chat").route
    available_input = chat_route.context_window - chat_route.max_output

    def estimate_candidate(
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> int:
        del tools
        candidate_prompts.append(str(messages[0]["content"]))
        assert loader.skills == published_skills
        public_messages = builder.build_status_messages((), session_id=session.session_id)
        assert '"name":"active"' in public_messages[0]["content"]
        assert '"name":"candidate"' not in public_messages[0]["content"]
        assert '"name":"published"' not in public_messages[0]["content"]
        return available_input - 1 + int(over_budget)

    monkeypatch.setattr(loop_module, "estimate_request_tokens", estimate_candidate)
    with builder.foreground_projection_scope(active_skills):
        if over_budget:
            with pytest.raises(ModelContextOverflowError) as raised:
                loop.reload_skill()
            assert raised.value.error.code == "model_context_overflow"
            assert raised.value.error.message == MODEL_CONTEXT_OVERFLOW_MESSAGE
            assert loader.skills == published_skills
            assert loader.metadata == published_metadata
            assert loader.resolve_manual("/published request") == published_invocation
            assert loader.resolve_manual("/candidate request") is None
        else:
            assert loop.reload_skill() == loader.metadata
            assert tuple(item.name for item in loader.metadata) == ("candidate",)
            invocation = loader.resolve_manual("/candidate request")
            assert invocation is not None
            assert invocation.body == candidate_document
            assert loader.resolve_manual("/published request") is None

        assert len(candidate_prompts) == 1
        assert '"name":"candidate"' in candidate_prompts[0]
        assert '"name":"active"' not in candidate_prompts[0]
        assert '"name":"published"' not in candidate_prompts[0]
        public_messages = builder.build_status_messages((), session_id=session.session_id)
        assert '"name":"active"' in public_messages[0]["content"]
        assert '"name":"candidate"' not in public_messages[0]["content"]
        assert '"name":"published"' not in public_messages[0]["content"]

    public_messages = builder.build_status_messages((), session_id=session.session_id)
    expected_name = "published" if over_budget else "candidate"
    assert f'"name":"{expected_name}"' in public_messages[0]["content"]
    assert '"name":"active"' not in public_messages[0]["content"]


@pytest.mark.parametrize("operation", ["preflight", "reload"])
@pytest.mark.parametrize("empty_candidate", [False, True], ids=["skills", "empty"])
@pytest.mark.parametrize("over_budget", [False, True], ids=["below-budget", "over-budget"])
def test_skill_budget_uses_public_status_projection_and_complete_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    empty_candidate: bool,
    over_budget: bool,
) -> None:
    monkeypatch.setattr("myclaw.agent.context.builder.datetime", _FrozenDateTime)
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: active\ndescription: Active snapshot\n---\nactive body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9",
        "compact_ratio = 0.9\nenable_skill_always_load = true",
    )
    mcp_tool = _LargeSchemaTool()
    loop, session, _bus = _runtime(tmp_path, _Router(()), config_text=config, mcp_tools=(mcp_tool,))
    action_summary = "- Preserved the active Session work."
    session.update_metadata(summary=action_summary)
    builder = loop._context_builder
    loader = loop._skill_loader
    active_skills = loader.skills
    if empty_candidate:
        instruction.unlink()
    else:
        instruction.write_text(
            "---\nname: candidate\ndescription: Candidate snapshot\nalways: true\n---\n"
            "candidate instructions\n",
            encoding="utf-8",
        )
    if operation == "preflight":
        loader.load()
    published_skills = loader.skills
    expected_tools = loop.tool_schemas
    assert len(expected_tools) > 1
    original_build_status = builder.build_status_messages
    public_projections: list[list[dict[str, Any]]] = []
    observed_summaries: list[str] = []
    estimated_requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    chat_route = loop._configuration.resolve_route("chat").route
    available_input = chat_route.context_window - chat_route.max_output

    def observe_public_status(
        history: Sequence[dict[str, Any]],
        *,
        session_id: str,
        summary: str = "",
    ) -> list[dict[str, Any]]:
        assert tuple(history) == ()
        assert session_id == session.session_id
        assert loader.skills == published_skills
        observed_summaries.append(summary)
        projected = original_build_status(history, session_id=session_id, summary=summary)
        public_projections.append(projected)
        return projected

    def observe_estimate(
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> int:
        estimated_requests.append((deepcopy(list(messages)), deepcopy(list(tools))))
        projected = original_build_status((), session_id=session.session_id)
        assert '"name":"active"' in projected[0]["content"]
        assert '"name":"candidate"' not in projected[0]["content"]
        assert loader.skills == published_skills
        return available_input - 1 + int(over_budget)

    with monkeypatch.context() as patch:
        patch.setattr(builder, "build_status_messages", observe_public_status)
        patch.setattr(loop_module, "estimate_request_tokens", observe_estimate)
        with builder.foreground_projection_scope(active_skills):
            validate = loop.preflight if operation == "preflight" else loop.reload_skill
            if over_budget:
                with pytest.raises(ModelContextOverflowError) as raised:
                    validate()
                assert raised.value.error.code == "model_context_overflow"
                assert raised.value.error.message == MODEL_CONTEXT_OVERFLOW_MESSAGE
                assert loader.skills == published_skills
            else:
                validate()
                expected_names = () if empty_candidate else ("candidate",)
                assert tuple(item.name for item in loader.metadata) == expected_names
            restored = original_build_status((), session_id=session.session_id)
            assert '"name":"active"' in restored[0]["content"]
            assert '"name":"candidate"' not in restored[0]["content"]

    assert len(public_projections) == len(estimated_requests) == 1
    assert observed_summaries == [action_summary]
    assert public_projections[0][1] == {"role": "user", "content": action_summary}
    budget_messages, budget_tools = estimated_requests[0]
    assert '"name":"active"' not in budget_messages[0]["content"]
    assert ('"name":"candidate"' in budget_messages[0]["content"]) is not empty_candidate
    assert ("candidate instructions" in budget_messages[0]["content"]) is not empty_candidate
    assert budget_messages == public_projections[0]
    assert tuple(budget_tools) == expected_tools
    assert all("large_schema" not in schema for schema in budget_tools)
    assert estimate_request_tokens(budget_messages, budget_tools) > estimate_request_tokens(
        budget_messages,
    )
    if not over_budget:
        ordinary_status = loop.runtime_status_input()
        assert ordinary_status.projected_messages == tuple(budget_messages)
        assert ordinary_status.projected_tools == tuple(budget_tools)
        assert estimate_request_tokens(
            ordinary_status.projected_messages,
            ordinary_status.projected_tools,
        ) == estimate_request_tokens(budget_messages, budget_tools)


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_reload_public_projection_failure_restores_scope_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: active\ndescription: Active snapshot\n---\nactive body\n",
        encoding="utf-8",
    )
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    builder = loop._context_builder
    loader = loop._skill_loader
    active_skills = loader.skills
    instruction.write_text(
        "---\nname: published\ndescription: Published snapshot\n---\npublished body\n",
        encoding="utf-8",
    )
    loader.load()
    published_skills = loader.skills
    published_metadata = loader.metadata
    published_invocation = loader.resolve_manual("/published request")
    instruction.write_text(
        "---\nname: candidate\ndescription: Candidate snapshot\n---\ncandidate body\n",
        encoding="utf-8",
    )
    original_build_status = builder.build_status_messages
    projected_prompts: list[str] = []
    observed_summaries: list[str] = []
    error = error_type("candidate projection failed")

    def fail_public_status(
        history: Sequence[dict[str, Any]],
        *,
        session_id: str,
        summary: str = "",
    ) -> list[dict[str, Any]]:
        observed_summaries.append(summary)
        projected = original_build_status(history, session_id=session_id, summary=summary)
        projected_prompts.append(projected[0]["content"])
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(builder, "build_status_messages", fail_public_status)
        with builder.foreground_projection_scope(active_skills):
            with pytest.raises(error_type) as raised:
                loop.reload_skill()
            assert raised.value is error
            restored = original_build_status((), session_id=session.session_id)
            assert '"name":"active"' in restored[0]["content"]
            assert '"name":"candidate"' not in restored[0]["content"]
            assert '"name":"published"' not in restored[0]["content"]

    assert len(projected_prompts) == 1
    assert observed_summaries == [""]
    assert '"name":"candidate"' in projected_prompts[0]
    assert '"name":"active"' not in projected_prompts[0]
    assert '"name":"published"' not in projected_prompts[0]
    assert loader.skills == published_skills
    assert loader.metadata == published_metadata
    assert loader.resolve_manual("/published request") == published_invocation
    assert loader.resolve_manual("/candidate request") is None
    restored = builder.build_status_messages((), session_id=session.session_id)
    assert '"name":"published"' in restored[0]["content"]
    assert '"name":"active"' not in restored[0]["content"]
    assert '"name":"candidate"' not in restored[0]["content"]


@pytest.mark.asyncio
async def test_reload_during_active_run_preserves_old_request_and_updates_future_run(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ReloadBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))
            first = len(self.requests) == 1

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                if first:
                    self.first_started.set()
                    await self.release_first.wait()
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ReloadBarrierRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
    )
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner first request"))
        await asyncio.wait_for(router.first_started.wait(), timeout=1)
        first_request = deepcopy(router.requests[0])

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = loop.reload_skill()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        assert router.requests[0] == first_request
        assert '"name":"planner"' in str(first_request[0]["content"])
        assert '"name":"reviewer"' not in str(first_request[0]["content"])
        assert "old body" in str(first_request[-1]["content"])
        assert "new body" not in str(first_request[-1]["content"])

        router.release_first.set()
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("/reviewer second request"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 2
    assert '"name":"reviewer"' in str(router.requests[1][0]["content"])
    assert '"name":"planner"' not in str(router.requests[1][0]["content"])
    assert "new body" in str(router.requests[1][-1]["content"])
    assert "old body" not in str(router.requests[1][-1]["content"])


@pytest.mark.parametrize("error_type", (RuntimeError, asyncio.CancelledError))
def test_foreground_projection_scope_restores_published_state_after_failure(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    old_skills = loop._skill_loader.skills
    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )
    loop.reload_skill()

    with pytest.raises(error_type):
        with loop._context_builder.foreground_projection_scope(old_skills):
            old_prompt = loop._context_builder.foreground_system_prompt()
            assert '"name":"planner"' in old_prompt
            assert '"name":"reviewer"' not in old_prompt
            raise error_type()

    published_prompt = loop._context_builder.foreground_system_prompt()
    assert '"name":"reviewer"' in published_prompt
    assert '"name":"planner"' not in published_prompt


@pytest.mark.asyncio
async def test_reload_during_context_preparation_keeps_run_skill_snapshot(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ContextPreparationBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ContextPreparationBarrierRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
    )
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()
    original_prepare = loop._prepare_agent_run

    async def blocked_prepare(
        context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        preparation_started.set()
        await release_preparation.wait()
        return await original_prepare(context, tool_gateway=tool_gateway)

    object.__setattr__(loop, "_prepare_agent_run", blocked_prepare)
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("first request"))
        await asyncio.wait_for(preparation_started.wait(), timeout=1)

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = loop.reload_skill()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        release_preparation.set()
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert "planner" in str(router.requests[0][0]["content"])
    assert "reviewer" not in str(router.requests[0][0]["content"])


@pytest.mark.asyncio
async def test_loop_consumes_foreground_inputs_fifo_and_publishes_one_terminal_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _Router((_response("one"), _response("two"), _response("three")))
    loop, session, _bus = _runtime(tmp_path, router)
    commit_calls = 0
    persist_calls = 0
    original_commit = Session.commit_agent_run
    original_persist = Session.persist

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        for content in ("one", "two", "three"):
            await _bus.put_inbound(InboundMessage(content))

        terminals = await _terminals(_bus, 3)

        assert len(router.calls) == 3
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == [
            "one",
            "two",
            "three",
        ]
        assert [message.metadata for message in terminals] == [
            {"_streamed": True},
            {"_streamed": True},
            {"_streamed": True},
        ]
        assert commit_calls == 3
        assert persist_calls == 3
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_manual_skill_invocation_preserves_raw_order_and_projects_expanded_user(
    tmp_path: Path,
) -> None:
    raw_input = "/planner Do the work"
    body = "Follow the plan.\n"
    document = "---\nname: planner\ndescription: Plan work\n---\n" + body
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(document.encode("utf-8"))
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    base_router = _CapturingRouter((_response("Generated title"), _response("Completed")))
    router = TaskFramingRouterAdapter(base_router)

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        title_prompt="Generate a title",
        skill_loader=loader,
    )
    previous_blackboard = {
        "goal": "Preserved goal",
        "completion_boundary": "Preserved boundary",
    }
    session.update_metadata(blackboard=previous_blackboard)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage(raw_input))
        await _terminals(_bus, 1)
    finally:
        await loop.close()

    assert len(router.calls) == 2
    assert raw_input in router.calls
    assert any("Follow the plan." in call and "Do the work" in call for call in router.calls)
    assert router.framing_requests == []
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        raw_input
    ]
    assert session.metadata["blackboard"] == previous_blackboard


@pytest.mark.asyncio
async def test_published_manual_skill_loader_ignores_later_file_changes(
    tmp_path: Path,
) -> None:
    raw_input = "/planner request"
    document = "---\nname: planner\ndescription: Plan work\n---\nPRIVATE MANUAL BODY\n"
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(document.encode("utf-8"))
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    instruction.unlink()

    base_router = _CapturingRouter((_response("Generated title"), _response("Completed")))
    router = TaskFramingRouterAdapter(base_router)

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        title_prompt="Generate a title",
        skill_loader=loader,
    )
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage(raw_input))
        await _terminals(_bus, 1)
    finally:
        await loop.close()

    assert len(router.calls) == 2
    assert any("PRIVATE MANUAL BODY" in call and '"request"' in call for call in router.calls)
    assert router.framing_requests == []
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        raw_input
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_input", "expected_framing_calls"),
    [
        ("ordinary input", 1),
        ("/unknown", 1),
        (" /planner", 1),
        ("/Planner", 1),
        ("/planner", 0),
        ("/planner\u2003request", 0),
    ],
)
async def test_only_exact_manual_skill_invocations_skip_task_framing(
    tmp_path: Path,
    raw_input: str,
    expected_framing_calls: int,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    router = TaskFramingRouterAdapter(_Router((_response("Completed"),)))
    loop, _session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        skill_loader=loader,
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage(raw_input))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.framing_requests) == expected_framing_calls


@pytest.mark.asyncio
async def test_ordinary_turn_after_manual_skill_reuses_preserved_blackboard(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    previous = Blackboard(goal="Preserved goal", completion_boundary="Preserved boundary")
    router = TaskFramingRouterAdapter(
        _Router((_response("Manual result"), _response("Ordinary result"))),
        (
            _response(
                '{"action":"keep","task_goal":null,"completion_boundary":null}',
                input_tokens=0,
                output_tokens=0,
            ),
        ),
    )
    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        skill_loader=loader,
    )
    session.update_metadata(
        blackboard={
            "goal": previous.goal,
            "completion_boundary": previous.completion_boundary,
        }
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner do it"))
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("ordinary follow-up"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.framing_requests) == 1
    framing_content = router.framing_requests[0][0]["content"]
    assert isinstance(framing_content, str)
    assert "ordinary follow-up" in framing_content
    assert "Manual result" in framing_content
    assert '"task_goal":"Preserved goal"' in framing_content
    assert session.metadata["blackboard"] == {
        "goal": previous.goal,
        "completion_boundary": previous.completion_boundary,
    }


@pytest.mark.asyncio
async def test_manual_skill_context_failure_preserves_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    router = TaskFramingRouterAdapter(_Router(()))

    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        skill_loader=loader,
    )

    async def fail_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise ModelCallError(ErrorInfo("model_failed", "context failed"))

    object.__setattr__(loop, "_prepare_agent_run", fail_prepare)
    session.update_metadata(
        blackboard={
            "goal": "Preserved goal",
            "completion_boundary": "Preserved boundary",
        }
    )
    before_metadata = deepcopy(session.metadata)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner fail"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata["finish_reason"] == "failed"
    assert router.framing_requests == []
    assert session.metadata["blackboard"] == before_metadata["blackboard"]
    assert session.metadata["token_usage"] == {
        **before_metadata["token_usage"],
        "model_calls": 0,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_manual_skill_context_cancellation_preserves_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    router = TaskFramingRouterAdapter(_Router(()))
    started = asyncio.Event()

    async def block_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        skill_loader=loader,
    )
    object.__setattr__(loop, "_prepare_agent_run", block_prepare)
    session.update_metadata(
        blackboard={
            "goal": "Preserved goal",
            "completion_boundary": "Preserved boundary",
        }
    )
    before_metadata = deepcopy(session.metadata)

    await loop.start()
    await bus.put_inbound(InboundMessage("/planner wait"))
    await started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert router.framing_requests == []
    assert session.metadata["blackboard"] == before_metadata["blackboard"]
    assert session.metadata["token_usage"] == {
        **before_metadata["token_usage"],
        "model_calls": 0,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_loop_preparation_failure_has_no_session_commit_and_fifo_continues(
    tmp_path: Path,
) -> None:
    router = _Router((_response("after failure"),))
    calls = 0

    loop, session, _bus = _runtime(tmp_path, router)
    original_prepare = loop._prepare_agent_run

    async def wrapped_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))
        return await original_prepare(context, tool_gateway=tool_gateway)

    object.__setattr__(loop, "_prepare_agent_run", wrapped_prepare)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first"))
        await _bus.put_inbound(InboundMessage("second"))
        first = await _bus.get_outbound()
        while first.metadata.get("_streamed") is not True:
            first = await _bus.get_outbound()
        second = await _bus.get_outbound()
        while second.metadata.get("_streamed") is not True:
            second = await _bus.get_outbound()

        assert first.type == "system_control"
        assert first.metadata["error_code"] == "model_failed"
        assert second.metadata == {"_streamed": True}
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == ["first", "second"]
        assert session.messages[1]["status"] == "error"
        assert len(router.calls) == 1
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_foreground_run_start_hard_overflow_does_not_commit_or_call_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MINIMAL_VALID_CONFIG.replace(
        "context_window = 200000", "context_window = 8192"
    ).replace("max_output = 8192", "max_output = 1024")
    router = _Router(())
    loop, session, bus = _runtime(tmp_path, router, config_text=config)
    messages_before = deepcopy(session.messages)
    metadata_before = deepcopy(session.metadata)
    commit_calls = 0
    original_commit = Session.commit_agent_run

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("x" * 100_000))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata["error_code"] == "model_context_overflow"
    assert session.messages == messages_before
    assert session.metadata == metadata_before
    assert router.calls == []
    assert commit_calls == 0
    assert not (session.workspace_state.sessions_directory / f"{session.session_id}.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "error"),
    (
        (
            "run_start",
            ErrorInfo("model_context_overflow", MODEL_CONTEXT_OVERFLOW_MESSAGE),
        ),
        ("react", ErrorInfo("provider_unavailable", "Summary provider failed.")),
    ),
)
async def test_foreground_commits_summary_failure_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["run_start", "react"],
    error: ErrorInfo,
) -> None:
    router = _Router(())
    loop, session, bus = _runtime(tmp_path, router)
    failure = CommittableAgentRunError(error)
    commit_calls = 0
    original_commit = Session.commit_agent_run

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    async def fail_run_start(
        context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise failure

    async def fail_react(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        del args, kwargs
        raise failure

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    if stage == "run_start":
        object.__setattr__(loop, "_prepare_agent_run", fail_run_start)
    else:
        monkeypatch.setattr(
            compactor_module.AgentRunContextRequestPreparer,
            "prepare",
            fail_react,
        )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("summary failure"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.content == error.message
    assert terminal.metadata == {
        "finish_reason": "failed",
        "error_code": error.code,
        "_streamed": True,
    }
    assert commit_calls == 1
    assert router.calls == []
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[-1]["status"] == "error"
    assert session.messages[-1]["error"] == {
        "code": error.code,
        "message": error.message,
    }


@pytest.mark.asyncio
async def test_foreground_commits_cancelled_summary_failure_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, session, bus = _runtime(tmp_path, _Router(()))
    error = ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)
    commit_calls = 0
    original_commit = Session.commit_agent_run

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    async def fail_run_start(
        context: Any,
        *,
        tool_gateway: ToolGateway,
    ) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise CommittableAgentRunError(error)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    object.__setattr__(loop, "_prepare_agent_run", fail_run_start)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("cancelled summary"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert commit_calls == 1
    assert session.messages[-1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_loop_commits_failed_runner_result_once_before_one_safe_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Failed goal", completion_boundary="Failed boundary")
    router = _Router((ModelCallError(ErrorInfo("model_failed", "provider failed")),))
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=(_framing_response(staged, input_tokens=4, output_tokens=2),),
    )
    commit_calls = 0
    persist_calls = 0
    original_commit = Session.commit_agent_run
    original_persist = Session.persist

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("failed input"))
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.type == "system_control"
        assert terminal.content == "provider failed"
        assert terminal.metadata == {
            "finish_reason": "failed",
            "error_code": "model_failed",
            "_streamed": True,
        }
        assert commit_calls == 1
        assert persist_calls == 1
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.messages[0]["restore_anchor_id"] == 1
        assert session.messages[0]["restore_before"]["last_compacted"] == 0
        assert session.messages[-1]["status"] == "error"
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.metadata["token_usage"] == {
            "model_calls": 2,
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_commits_max_iteration_repair_once_before_safe_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Bounded goal", completion_boundary="Bounded boundary")
    router = _MaxRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=(_framing_response(staged, input_tokens=4, output_tokens=2),),
    )
    commit_calls = 0
    persist_calls = 0
    original_commit = Session.commit_agent_run
    original_persist = Session.persist

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("bounded input"))
        terminal = (await _terminals(_bus, 1))[0]

        assert len(router.calls) == 50
        assert terminal.type == "system_control"
        assert terminal.content == (
            "MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。"  # noqa: RUF001
            "可以再次尝试本次请求或者尝试给出更明确的任务目标。"
        )
        assert terminal.metadata == {
            "finish_reason": "max_iterations",
            "error_code": "agent_iteration_limit",
            "_streamed": True,
        }
        assert commit_calls == 1
        assert persist_calls == 1
        assert session.metadata["token_usage"] == {
            "model_calls": 51,
            "input_tokens": 54,
            "output_tokens": 52,
            "total_tokens": 106,
        }
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.messages[-1]["status"] == "error"
        assert session.messages[-1]["error"]["code"] == "agent_iteration_limit"
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_cancellation_repairs_and_keeps_the_next_queued_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    staged = Blackboard(goal="Cancelled goal", completion_boundary="Cancelled boundary")
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=(
            _framing_response(staged, input_tokens=4, output_tokens=2),
            _framing_response(staged, input_tokens=4, output_tokens=2),
        ),
    )
    commit_calls = 0
    persist_calls = 0
    original_commit = Session.commit_agent_run
    original_persist = Session.persist

    def commit_agent_run(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "commit_agent_run", commit_agent_run)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cancelled input"))
        await _bus.put_inbound(InboundMessage("queued input"))
        await started.wait()
        await loop.cancel_active_run()
        terminals = await _terminals(_bus, 2)

        assert router.calls == ["call", "call"]
        assert [terminal.metadata for terminal in terminals] == [
            {
                "finish_reason": "cancelled",
                "error_code": "turn_cancelled",
                "_streamed": True,
            },
            {"_streamed": True},
        ]
        assert commit_calls == 2
        assert persist_calls == 2
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == [
            "cancelled input",
            "queued input",
        ]
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.metadata["token_usage"] == {
            "model_calls": 3,
            "input_tokens": 9,
            "output_tokens": 5,
            "total_tokens": 14,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_confirmation_uses_one_direct_pending_future_and_cancels_it(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    router = _Router(
        (
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_confirmation",
                    name="read_file",
                    arguments=json.dumps({"path": str(outside)}),
                ),
            ),
        )
    )
    loop, _session, _bus = _runtime(tmp_path, router)
    requests: list[ConfirmationRequestView] = []
    requested = asyncio.Event()

    def on_confirmation(request: ConfirmationRequestView) -> None:
        requests.append(request)
        requested.set()

    loop.bind_confirmation_callback(on_confirmation)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("confirm this"))
        await requested.wait()

        assert len(requests) == 1
        request = requests[0]
        with pytest.raises(ValueError, match="late or unknown"):
            loop.respond_to_confirmation(uuid4(), "approved")

        await loop.cancel_active_run()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.metadata == {
            "finish_reason": "cancelled",
            "error_code": "turn_cancelled",
            "_streamed": True,
        }
        with pytest.raises(ValueError, match="late or unknown"):
            loop.respond_to_confirmation(request.confirmation_id, "approved")
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_permission_upgrade_does_not_approve_an_active_run_confirmation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    router = _Router(
        (
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_confirmation",
                    name="read_file",
                    arguments=json.dumps({"path": str(outside)}),
                ),
            ),
            _response("Declined as expected."),
        )
    )
    loop, _session, bus = _runtime(tmp_path, router)
    requested = asyncio.Event()
    requests: list[ConfirmationRequestView] = []

    def on_confirmation(request: ConfirmationRequestView) -> None:
        requests.append(request)
        requested.set()

    loop.bind_confirmation_callback(on_confirmation)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("confirm this"))
        await asyncio.wait_for(requested.wait(), timeout=1)

        loop._permission_control.select("full-access")
        await asyncio.sleep(0)

        assert loop.control.has_active_run
        assert len(requests) == 1
        loop.respond_to_confirmation(requests[0].confirmation_id, "declined")
        terminal = (await asyncio.wait_for(_terminals(bus, 1), timeout=1))[0]

        assert terminal.metadata == {"_streamed": True}
        assert loop._permission_control.current() == "full-access"
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_permission_downgrade_does_not_revoke_the_admitted_full_access_run(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    class BlockingPermissionRouter(_Router):
        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.calls.append(str(messages[-1]["content"]))
            outcome = self._outcomes.popleft()

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                if not model_started.is_set():
                    model_started.set()
                    await release_model.wait()
                assert not isinstance(outcome, BaseException)
                yield ModelCompleted(response=outcome)

            return replay()

    router = BlockingPermissionRouter(
        (
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_direct",
                    name="read_file",
                    arguments=json.dumps({"path": str(outside)}),
                ),
            ),
            _response("Read without a retroactive confirmation."),
        )
    )
    loop, _session, bus = _runtime(tmp_path, router)
    requests: list[ConfirmationRequestView] = []
    loop.bind_confirmation_callback(requests.append)
    loop._permission_control.select("full-access")
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("read this"))
        await asyncio.wait_for(model_started.wait(), timeout=1)

        loop._permission_control.select("read-only")
        release_model.set()
        terminal = (await asyncio.wait_for(_terminals(bus, 1), timeout=1))[0]

        assert terminal.metadata == {"_streamed": True}
        assert requests == []
        assert loop._permission_control.current() == "read-only"
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_preparation_cancellation_publishes_the_cancelled_terminal(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, _bus = _runtime(tmp_path, _Router(()))
    object.__setattr__(loop, "_prepare_agent_run", prepare)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cancel during preparation"))
        await started.wait()

        await loop.cancel_active_run()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.type == "system_control"
        assert terminal.content == "MyClaw 已取消本轮对话。"
        assert terminal.metadata == {
            "finish_reason": "cancelled",
            "error_code": "turn_cancelled",
            "_streamed": True,
        }
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.messages[-1]["status"] == "interrupted"
        assert session.messages[0]["restore_anchor_id"] == 1
        assert session.messages[0]["restore_before"]["last_compacted"] == 0
        assert session.metadata["token_usage"]["model_calls"] == 0
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_preparation_failure_does_not_start_title_or_accumulate_usage(
    tmp_path: Path,
) -> None:
    async def fail_preparation(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))

    router = _Router((_response("must not become a title"),))
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        title_prompt="Generate a title",
    )
    object.__setattr__(loop, "_prepare_agent_run", fail_preparation)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("uncommitted input"))
        _ = await _terminals(_bus, 1)
        for _ in range(5):
            await asyncio.sleep(0)

        assert router.calls == ["uncommitted input"]
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.metadata["title"] == "Untitled session"
        assert session.metadata["token_usage"] == {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_async_preparation_failure_discards_parallel_title_result(
    tmp_path: Path,
) -> None:
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    async def fail_after_waiting(
        context: Any, *, tool_gateway: ToolGateway
    ) -> list[dict[str, Any]]:
        del context, tool_gateway
        preparation_started.set()
        await release_preparation.wait()
        raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))

    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        title_prompt="Generate a title",
    )
    object.__setattr__(loop, "_prepare_agent_run", fail_after_waiting)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("uncommitted input"))
        await preparation_started.wait()
        for _ in range(100):
            if "title" in router.calls:
                break
            await asyncio.sleep(0)

        assert router.calls == ["title"]
        assert session.metadata["title"] == "Untitled session"

        release_preparation.set()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.metadata == {
            "finish_reason": "failed",
            "error_code": "model_failed",
            "_streamed": True,
        }
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.metadata["title"] == "Untitled session"
        assert session.metadata["token_usage"] == {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    finally:
        release_preparation.set()
        await loop.close()


@pytest.mark.asyncio
async def test_title_request_uses_context_builder_messages_without_loop_override(
    tmp_path: Path,
) -> None:
    router = _Router(())
    loop, _session, _bus = _runtime(
        tmp_path,
        router,
        title_prompt="Loop-owned title prompt",
    )
    builder_inputs: list[str] = []
    captured_messages: list[dict[str, Any]] = []

    def build_title_messages(content: str) -> list[dict[str, Any]]:
        builder_inputs.append(content)
        return [
            {"role": "system", "content": "Builder-owned title prompt"},
            {"role": "user", "content": content},
        ]

    def stream(
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        captured_messages.extend(deepcopy(messages))

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(response=_response("Generated title"))

        return replay()

    object.__setattr__(loop._context_builder, "build_title_messages", build_title_messages)
    object.__setattr__(router, "stream", stream)

    events = loop._router_stream_title("  First title input.  ")
    _ = [event async for event in events]

    assert builder_inputs == ["First title input."]
    assert captured_messages == [
        {"role": "system", "content": "Builder-owned title prompt"},
        {"role": "user", "content": "First title input."},
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_first_message_title_runs_while_foreground_chat_is_blocked(
    tmp_path: Path,
) -> None:
    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first input"))
        await router.chat_started.wait()
        for _ in range(100):
            if session.metadata["title"] != "Untitled session":
                break
            await asyncio.sleep(0)

        assert session.metadata["title"] == "Generated while chat blocked"

        router.release_chat.set()
        assert (await _terminals(_bus, 1))[0].metadata == {"_streamed": True}
    finally:
        router.release_chat.set()
        await loop.close()


@pytest.mark.asyncio
async def test_slow_title_keeps_one_session_log_owner_across_the_next_fifo_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _SlowTitleLogRouter()
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")
    active_contexts = 0
    maximum_contexts = 0
    original_session_log = real_session_log

    @contextmanager
    def observed_session_log(active_session: Session) -> Iterator[None]:
        nonlocal active_contexts, maximum_contexts
        active_contexts += 1
        maximum_contexts = max(maximum_contexts, active_contexts)
        try:
            with original_session_log(active_session):
                yield
        finally:
            active_contexts -= 1

    monkeypatch.setattr("myclaw.agent.loop.session_log", observed_session_log)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first input"))
        _ = await _terminals(_bus, 1)
        await router.title_started.wait()

        await _bus.put_inbound(InboundMessage("second input"))
        _ = await _terminals(_bus, 1)

        router.release_title.set()
        for _ in range(100):
            if session.metadata["title"] == "Slow title":
                break
            await asyncio.sleep(0)
    finally:
        router.release_title.set()
        await loop.close()

    log_path = session.workspace_state.logs_directory / f"{session.session_id}.log"
    assert maximum_contexts == 1
    assert log_path.read_text(encoding="utf-8").count("Second foreground marker") == 1


@pytest.mark.asyncio
async def test_append_failure_reports_safe_terminal_and_fifo_consumer_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")
    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        task_framing_outcomes=(
            _framing_response(staged, input_tokens=5, output_tokens=2),
            RuntimeError("Task Framing failed"),
        ),
    )
    session.update_metadata(
        title="Existing title",
        blackboard={
            "goal": previous.goal,
            "completion_boundary": previous.completion_boundary,
        },
        usage_delta={
            "model_calls": 2,
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        },
    )
    before_messages = deepcopy(session.messages)
    before_metadata = deepcopy(session.metadata)
    results = deque(
        (
            AgentRunnerResult(
                messages=[{"role": "assistant", "content": "invalid increment"}],
                final_content="invalid increment",
                usage={
                    "model_calls": 1,
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            ),
            AgentRunnerResult(
                messages=[
                    {
                        "role": "assistant",
                        "content": "next completed",
                        "tool_calls": [],
                        "status": "completed",
                        "error": None,
                        "token_usage": {
                            "model_calls": 1,
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        },
                    }
                ],
                final_content="next completed",
                usage={
                    "model_calls": 1,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            ),
        )
    )

    async def run(*args: Any, **kwargs: Any) -> AgentRunnerResult:
        del args, kwargs
        return results.popleft()

    monkeypatch.setattr(AgentRunner, "run", run)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cannot commit"))
        failed = (await _terminals(_bus, 1))[0]
        assert session.messages == before_messages
        assert session.metadata == before_metadata
        await _bus.put_inbound(InboundMessage("next input"))
        completed = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert failed.type == "system_control"
    assert failed.metadata == {
        "finish_reason": "failed",
        "error_code": "model_failed",
        "_streamed": True,
    }
    assert completed.type == "model_response"
    assert completed.metadata == {"_streamed": True}
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "next input"
    ]


@pytest.mark.asyncio
async def test_persist_request_failure_is_silent_and_terminal_stays_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router((_response("committed in memory"),)))

    def fail_persist() -> None:
        raise OSError("private persistence detail")

    monkeypatch.setattr(session, "persist", fail_persist)
    capture = capture_diagnostics()
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("persist fails"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        capture.close()
        await loop.close()

    assert terminal.metadata == {
        "finish_reason": "failed",
        "error_code": "persistence_error",
        "_streamed": True,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert "Foreground Session persist failed" not in capture.event_text
    assert "private persistence detail" not in capture.text


@pytest.mark.asyncio
async def test_loop_normal_close_saves_only_its_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, first, _bus = _runtime(tmp_path, _Router(()))
    closed: list[str] = []
    first_close = first.close

    def close_first() -> None:
        closed.append(first.session_id)
        first_close()

    monkeypatch.setattr(first, "close", close_first)

    await loop.close()

    assert closed == [first.session_id]


@pytest.mark.asyncio
async def test_loop_abort_retains_cancelled_owned_tasks_until_cleanup_finishes(
    tmp_path: Path,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cancellable_work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_cleanup.wait()
            raise
        finally:
            cleanup_finished.set()

    task = asyncio.create_task(cancellable_work())
    loop._execution_task = task
    await started.wait()

    abort_task = asyncio.create_task(loop.abort())
    await asyncio.sleep(0)

    assert task in loop._aborted_tasks
    release_cleanup.set()
    await cleanup_finished.wait()
    tasks_drained = asyncio.Event()
    task.add_done_callback(lambda _task: tasks_drained.set())
    await tasks_drained.wait()
    await abort_task
    assert loop._aborted_tasks == set()


@pytest.mark.asyncio
async def test_terminal_loop_states_reject_restart(tmp_path: Path) -> None:
    closed_loop, _closed_session, _closed_bus = _runtime(tmp_path / "closed", _Router(()))
    await closed_loop.start()
    await closed_loop.close()

    with pytest.raises(RuntimeError, match="Agent Loop is closed"):
        await closed_loop.start()

    aborted_loop, _aborted_session, _aborted_bus = _runtime(tmp_path / "aborted", _Router(()))
    await aborted_loop.start()
    await aborted_loop.abort()

    with pytest.raises(RuntimeError, match="Agent Loop is closed"):
        await aborted_loop.start()


@pytest.mark.asyncio
async def test_close_transition_blocks_concurrent_preflight_and_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_finish_close() -> None:
        close_started.set()
        await release_close.wait()

    monkeypatch.setattr(loop, "_finish_close", blocked_finish_close)
    closing = asyncio.create_task(loop.close())
    await close_started.wait()

    try:
        with pytest.raises(RuntimeError, match="Agent Loop is closed"):
            loop.preflight()
        with pytest.raises(RuntimeError, match="Agent Loop is closed"):
            await loop.start()
    finally:
        release_close.set()
        await closing


@pytest.mark.asyncio
async def test_abort_wins_before_normal_close_finalizes_the_session(
    tmp_path: Path,
) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "preserve this turn",
                "timestamp": "2026-08-21T12:00:00.000+00:00",
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
            "summary": "",
        },
        last_compacted=0,
    )
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_finish_close() -> None:
        close_started.set()
        await release_close.wait()

    object.__setattr__(loop, "_finish_close", blocked_finish_close)
    closing = asyncio.create_task(loop.close())
    await close_started.wait()

    aborting = asyncio.create_task(loop.abort())
    await asyncio.sleep(0)
    release_close.set()

    await asyncio.gather(closing, aborting)
    assert loop._session_abandoned
    assert not loop._session_closed


@pytest.mark.asyncio
async def test_blank_foreground_input_performs_zero_task_framing_attempts(
    tmp_path: Path,
) -> None:
    router = TaskFramingRouterAdapter(_Router(()))
    loop, session, _bus = _runtime(tmp_path, router, task_framing_outcomes=None)
    execution_ready = asyncio.Event()

    committed = await loop._execute_foreground_logged(
        session,
        InboundMessage(" \n\t "),
        title_work=None,
        execution_ready=execution_ready,
        permission_snapshot=loop._permission_control.snapshot(loop._exec_host.resolved_shell),
    )

    assert not committed
    assert execution_ready.is_set()
    assert router.framing_requests == []
    assert session.messages == []
    assert session.metadata["token_usage"] == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_blank_foreground_input_releases_cancel_waiter(tmp_path: Path) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    execution_ready = asyncio.Event()

    await loop._execute_foreground(InboundMessage(" \n\t "), execution_ready=execution_ready)

    assert execution_ready.is_set()
    assert session.messages == []


@pytest.mark.asyncio
async def test_direct_bus_projects_completed_turn_without_session_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router((_response("completed without deltas"),)))

    def reject_session_access(_loop: AgentLoop) -> Session:
        raise AssertionError("Terminal adapter must not access Session")

    monkeypatch.setattr(AgentLoop, "session", property(reject_session_access))

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("foreground input"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await _bus.get_outbound())
    finally:
        await loop.close()

    assert [(message.type, message.content, message.metadata) for message in observed] == [
        ("model_response", "completed without deltas", {"_stream_delta": True}),
        ("model_response", "", {"_stream_end": True}),
        ("model_response", "", {"_streamed": True}),
    ]


@pytest.mark.asyncio
async def test_loop_publishes_the_exact_sparse_outbound_protocol_without_tool_results(
    tmp_path: Path,
) -> None:
    raw_arguments = '{"missing":"preserved exactly"}'
    tool_call = ModelToolCall(
        id="call_sparse",
        name="unknown_tool",
        arguments=raw_arguments,
    )
    router = _EventRouter(
        (
            (
                ReasoningDelta("reasoning"),
                TextDelta("working"),
                ModelCompleted(response=_response("working", tool_call=tool_call)),
            ),
            (
                TextDelta("done"),
                ModelCompleted(response=_response("done")),
            ),
        )
    )
    loop, session, _bus = _runtime(tmp_path, router)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("sparse output"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await _bus.get_outbound())

        assert [(message.type, message.content, message.metadata) for message in observed] == [
            ("model_reasoning", "reasoning", {"_stream_delta": True}),
            ("model_reasoning", "", {"_stream_end": True}),
            ("model_response", "working", {"_stream_delta": True}),
            ("model_response", "", {"_stream_end": True}),
            (
                "tool_call",
                "unknown_tool",
                {"tool_call_id": "call_sparse", "arguments": raw_arguments},
            ),
            (
                "tool_call",
                "unknown_tool",
                {"tool_call_id": "call_sparse", "status": "error"},
            ),
            ("model_response", "done", {"_stream_delta": True}),
            ("model_response", "", {"_stream_end": True}),
            ("model_response", "", {"_streamed": True}),
        ]
        assert [message["role"] for message in session.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_close_normally_cancels_active_run_without_dequeuing_the_next_message(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    loop, session, _bus = _runtime(tmp_path, router)
    queued = InboundMessage("remains queued")
    await loop.start()
    await _bus.put_inbound(InboundMessage("active input"))
    await _bus.put_inbound(queued)
    await started.wait()

    await loop.close()
    terminal = (await _terminals(_bus, 1))[0]

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "active input"
    ]
    assert await _bus.inbound_snapshot() == (queued,)
    assert router.calls == ["call"]


@pytest.mark.asyncio
async def test_foreground_frames_once_with_exact_session_inputs_and_atomic_projection(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    router = TaskFramingRouterAdapter(
        _Router((_response("answer"),)),
        (_framing_response(staged, input_tokens=5, output_tokens=2),),
    )

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
    )
    seed_session_state(
        session,
        messages=[
            {
                "role": "assistant",
                "content": "Latest complete assistant content",
                "timestamp": "2026-08-21T12:00:00.000+00:00",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
            "blackboard": {
                "goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            },
        },
        last_compacted=0,
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("raw <blackboard> input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(router.framing_requests) == 1
    framing_content = router.framing_requests[0][0]["content"]
    assert isinstance(framing_content, str)
    assert "raw <blackboard> input" in framing_content
    assert "Latest complete assistant content" in framing_content
    assert '"task_goal":"Previous goal"' in framing_content
    assert any(staged.goal in call and staged.completion_boundary in call for call in router.calls)
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "raw <blackboard> input"
    ]
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }


@pytest.mark.asyncio
async def test_tool_iterations_reuse_one_framing_and_context_projection_before_one_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Stable tool goal", completion_boundary="Stable tool boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }

    class ToolLoopRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []
            self.responses = deque(
                (
                    _response(
                        "use tool",
                        tool_call=ModelToolCall(
                            id="call_stable_projection",
                            name="unknown_tool",
                            arguments="{}",
                        ),
                        input_tokens=2,
                        output_tokens=1,
                    ),
                    _response("tool loop complete", input_tokens=3, output_tokens=2),
                )
            )

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))
            response = self.responses.popleft()

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(response=response)

            return replay()

    router = ToolLoopRouter()
    framing_router = TaskFramingRouterAdapter(
        router,
        (_framing_response(staged, input_tokens=4, output_tokens=2),),
    )
    projected_blackboard = (
        f"## Task goal\n\n{staged.goal}\n\n## Completion boundary\n\n{staged.completion_boundary}"
    )

    loop, session, _bus = _runtime(
        tmp_path,
        framing_router,
        task_framing_outcomes=None,
    )
    commit_calls = 0
    original_commit = session.commit_agent_run

    def commit_agent_run(messages: list[dict[str, Any]], **kwargs: Any) -> None:
        nonlocal commit_calls
        commit_calls += 1
        assert kwargs["usage_delta"] == framing_usage
        original_commit(messages, **kwargs)

    monkeypatch.setattr(session, "commit_agent_run", commit_agent_run)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("raw tool input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(framing_router.framing_requests) == 1
    assert len(router.requests) == 2
    assert all(request[1]["content"].endswith(projected_blackboard) for request in router.requests)
    assert commit_calls == 1
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[0]["content"] == "raw tool input"
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 9,
        "output_tokens": 5,
        "total_tokens": 14,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framing_outcome", "expected_framing_usage"),
    [
        (
            _response("not valid task framing", input_tokens=4, output_tokens=1),
            {
                "model_calls": 1,
                "input_tokens": 4,
                "output_tokens": 1,
                "total_tokens": 5,
            },
        ),
        (
            RuntimeError("Task Framing failed"),
            {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        ),
    ],
)
async def test_invalid_and_model_failed_framing_statuses_fail_open_and_clear_on_commit(
    tmp_path: Path,
    framing_outcome: ModelResponse | BaseException,
    expected_framing_usage: dict[str, int],
) -> None:
    previous = Blackboard(goal="Old goal", completion_boundary="Old boundary")
    router = TaskFramingRouterAdapter(
        _Router((_response("raw answer"),)),
        (framing_outcome,),
    )

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("continue without parsed framing"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(router.framing_requests) == 1
    assert "blackboard" not in session.metadata
    assert session.metadata["token_usage"] == {
        "model_calls": expected_framing_usage["model_calls"] + 1,
        "input_tokens": expected_framing_usage["input_tokens"] + 1,
        "output_tokens": expected_framing_usage["output_tokens"] + 1,
        "total_tokens": expected_framing_usage["total_tokens"] + 2,
    }


@pytest.mark.asyncio
async def test_framing_cancellation_reclaims_first_title_task_without_commit(
    tmp_path: Path,
) -> None:
    framer = BlockingTaskFramingRouterAdapter(_ConcurrentTitleRouter())
    loop, session, _bus = _runtime(
        tmp_path,
        framer,
        task_framing_outcomes=None,
        title_prompt="Generate a title",
    )

    await loop.start()
    await _bus.put_inbound(InboundMessage("cancel before the Agent Run"))
    await framer.started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(_bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[-1]["status"] == "interrupted"
    assert session.metadata["title"] == "Untitled session"
    assert session.metadata["token_usage"] == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert framer.cancelled.is_set()
    assert not loop._title_work


@pytest.mark.asyncio
async def test_context_failure_after_framing_preserves_previous_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")

    async def fail_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise ModelCallError(ErrorInfo("model_failed", "context failed"))

    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        task_framing_outcomes=(_framing_response(staged, input_tokens=4, output_tokens=2),),
    )
    object.__setattr__(loop, "_prepare_agent_run", fail_prepare)
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("context fails after framing"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata["finish_reason"] == "failed"
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        **before_usage,
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_context_cancellation_after_framing_preserves_previous_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")
    started = asyncio.Event()

    async def block_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        task_framing_outcomes=(_framing_response(staged, input_tokens=4, output_tokens=2),),
    )
    object.__setattr__(loop, "_prepare_agent_run", block_prepare)
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    await _bus.put_inbound(InboundMessage("context cancellation after framing"))
    await started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(_bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        **before_usage,
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_title_usage_is_preserved_when_title_finishes_before_foreground_commit(
    tmp_path: Path,
) -> None:
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    router = _ConcurrentTitleRouter()
    framing_router = TaskFramingRouterAdapter(
        router,
        (_framing_response(staged, input_tokens=5, output_tokens=2),),
    )
    loop, session, _bus = _runtime(
        tmp_path,
        framing_router,
        task_framing_outcomes=None,
        title_prompt="Generate a title",
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("title and foreground race"))
        await router.chat_started.wait()
        for _ in range(100):
            if session.metadata["title"] != "Untitled session":
                break
            await asyncio.sleep(0)

        assert session.metadata["title"] == "Generated while chat blocked"
        assert session.metadata["token_usage"] == {
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }

        router.release_chat.set()
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        router.release_chat.set()
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
    }


@pytest.mark.asyncio
async def test_foreground_commit_preserves_staged_framing_until_slow_title_finishes(
    tmp_path: Path,
) -> None:
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    router = _SlowTitleLogRouter()
    framing_router = TaskFramingRouterAdapter(
        router,
        (_framing_response(staged, input_tokens=5, output_tokens=2),),
    )
    loop, session, _bus = _runtime(
        tmp_path,
        framing_router,
        task_framing_outcomes=None,
        title_prompt="Generate a title",
    )

    await loop.start()
    await _bus.put_inbound(InboundMessage("foreground commits first"))
    terminal = (await _terminals(_bus, 1))[0]
    await router.title_started.wait()

    assert terminal.metadata == {"_streamed": True}
    assert session.metadata["title"] == "Untitled session"
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }

    router.release_title.set()
    for _ in range(100):
        if session.metadata["title"] == "Slow title":
            break
        await asyncio.sleep(0)
    await loop.close()

    assert session.metadata["title"] == "Slow title"
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
    }


@pytest.mark.asyncio
async def test_manual_title_wins_over_a_late_automatic_title(tmp_path: Path) -> None:
    router = _TitleBehaviorRouter(
        (_response("First response."),),
        title=_response("Generated late title"),
        delay_title=True,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "First input.")
        await router.title_started.wait()
        await session.rename("Manual title", expected_metadata_version=0)
        assert session.metadata["title"] == "Manual title"
        assert session.metadata_version == 1

        router.release_title.set()
        await loop.wait_for_restore_idle()
        assert session.metadata["title"] == "Manual title"
        await session.wait_for_pending_persist()
    finally:
        router.release_title.set()
        await loop.close()

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Manual title"
    assert reloaded.metadata_version == 1


@pytest.mark.asyncio
async def test_late_title_is_persisted_by_the_next_completed_turn(tmp_path: Path) -> None:
    router = _TitleBehaviorRouter(
        (_response("First response."), _response("Second response.")),
        title=_response("Generated late title"),
        delay_title=True,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "First input.")
        await router.title_started.wait()
        await session.wait_for_pending_persist()
        assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
            "Untitled session"
        )

        router.release_title.set()
        for _ in range(100):
            if session.metadata["title"] == "Generated late title":
                break
            await asyncio.sleep(0)
        assert session.metadata["title"] == "Generated late title"
        assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
            "Untitled session"
        )

        await collect_foreground_outbound(_bus, "Second input.")
        await session.wait_for_pending_persist()
    finally:
        router.release_title.set()
        await loop.close()

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Generated late title"
    assert [message["role"] for message in reloaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_foreground_commit_assigns_anchor_and_keeps_internal_fields_out_of_projections(
    tmp_path: Path,
) -> None:
    router = _CapturingRouter(
        (_response("Committed response."), _response("Response after anchored history."))
    )
    loop, session, bus = _runtime(tmp_path, router)

    await loop.start()
    try:
        await collect_foreground_outbound(bus, "Persist this input.")
        await session.wait_for_pending_persist()
        await collect_foreground_outbound(bus, "Use the prior context.")
        await session.wait_for_pending_persist()
        terminal_projection = loop.project_foreground_conversation()
    finally:
        await loop.close()

    user = session.messages[0]
    assert user["restore_anchor_id"] == 1
    assert isinstance(user["restore_run_token"], str)
    assert user["restore_before"]["last_compacted"] == 0
    hidden_fields = {"restore_anchor_id", "restore_run_token", "restore_before"}
    assert len(router.requests) == 2
    assert any(message.get("content") == "Persist this input." for message in router.requests[1])
    assert all(hidden_fields.isdisjoint(message) for message in terminal_projection.messages)
    assert all(
        hidden_fields.isdisjoint(message) for request in router.requests for message in request
    )

    loaded = Session.load(session.workspace_state, session.session_id)
    assert loaded.restore_candidates()[0].anchor_id == 1


@pytest.mark.asyncio
async def test_foreground_file_backup_uses_the_committed_user_run_token(
    tmp_path: Path,
) -> None:
    tool_call = ModelToolCall(
        id="write-call",
        name="write_file",
        arguments=json.dumps({"path": "tracked.txt", "content": "after"}),
    )
    router = _Router((_response("write", tool_call=tool_call), _response("done")))
    loop, session, bus = _runtime(tmp_path, router)

    tracked = session.workspace_state.workspace_path / "tracked.txt"
    tracked.write_bytes(b"before")
    await loop.start()
    try:
        await collect_foreground_outbound(bus, "Track this file change.")
        await session.wait_for_pending_persist()
    finally:
        await loop.close()

    user = next(message for message in session.messages if message["role"] == "user")
    journal = FileBackupStore(session.workspace_state, session.session_id).inspect()
    assert len(journal.entries) == 1
    assert journal.entries[0].run_token == UUID(user["restore_run_token"])
    assert FileBackupStore(session.workspace_state, session.session_id).read_backup(1) == b"before"
    assert tracked.read_bytes() == b"after"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "finish_reason"),
    [
        (ModelCallError(ErrorInfo("model_failed", "provider failed")), "failed"),
        (ModelCallError(ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)), "cancelled"),
    ],
    ids=("failed", "cancelled"),
)
async def test_foreground_file_backup_keeps_failed_or_cancelled_run_token_association(
    tmp_path: Path,
    failure: ModelCallError,
    finish_reason: str,
) -> None:
    tool_call = ModelToolCall(
        id="write-call",
        name="write_file",
        arguments=json.dumps({"path": "tracked.txt", "content": "after"}),
    )
    router = _Router((_response("write", tool_call=tool_call), failure))
    loop, session, bus = _runtime(tmp_path, router)

    tracked = session.workspace_state.workspace_path / "tracked.txt"
    tracked.write_bytes(b"before")
    await loop.start()
    try:
        observed = await collect_foreground_outbound(bus, "Track this interrupted file change.")
        await session.wait_for_pending_persist()
    finally:
        await loop.close()

    user = next(message for message in session.messages if message["role"] == "user")
    journal = FileBackupStore(session.workspace_state, session.session_id).inspect()
    assert len(journal.entries) == 1
    assert journal.entries[0].run_token == UUID(user["restore_run_token"])
    assert observed[-1].metadata["finish_reason"] == finish_reason
    assert tracked.read_bytes() == b"after"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_title", ["tool_call", "empty"])
async def test_invalid_title_uses_first_input_fallback_and_keeps_usage(
    tmp_path: Path,
    invalid_title: str,
) -> None:
    title = (
        ModelResponse(
            message=AssistantModelMessage(
                content="Do not use this title",
                tool_calls=(
                    ModelToolCall(
                        id="invalid-title",
                        name="read_file",
                        arguments='{"path":"README.md"}',
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="tool_calls",
        )
        if invalid_title == "tool_call"
        else _response('""', input_tokens=3, output_tokens=1)
    )
    router = _TitleBehaviorRouter((_response("First response.", input_tokens=5),), title=title)
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "  Meaningful first question.  ")
        for _ in range(100):
            if session.metadata["token_usage"]["model_calls"] == 2:
                break
            await asyncio.sleep(0)
    finally:
        await loop.close()

    title_output_tokens = 2 if invalid_title == "tool_call" else 1
    assert session.metadata["title"] == "Meaningful first question."
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 8,
        "output_tokens": 1 + title_output_tokens,
        "total_tokens": 9 + title_output_tokens,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_turn", [False, True])
async def test_close_applies_first_input_title_fallback_before_final_save(
    tmp_path: Path,
    cancel_turn: bool,
) -> None:
    router = _TitleBehaviorRouter(
        (_response("Foreground response."),),
        title=_response("Unreleased title"),
        delay_title=True,
        block_first_foreground=cancel_turn,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    if cancel_turn:
        turn = asyncio.create_task(collect_foreground_outbound(_bus, "  Cancelled first title.  "))
        await router.foreground_started.wait()
        await router.title_started.wait()
        await loop.cancel_active_run()
        terminal = (await turn)[-1]
        assert terminal.metadata["finish_reason"] == "cancelled"
        expected_title = "Cancelled first title."
    else:
        await collect_foreground_outbound(_bus, "  Shutdown fallback title.  ")
        await router.title_started.wait()
        expected_title = "Shutdown fallback title."

    await loop.close()

    assert session.metadata["title"] == expected_title
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == expected_title


@pytest.mark.asyncio
async def test_loop_close_swallows_final_session_failure(tmp_path: Path) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    capture = capture_diagnostics()

    def fail_close() -> None:
        raise OSError("private final Session close failure")

    session.close = fail_close  # type: ignore[method-assign]
    try:
        await loop.close()
    finally:
        capture.close()

    assert "Agent Loop Session close failed type=OSError" in capture.event_text
    assert "private final Session close failure" not in capture.event_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "previous", "expected"),
    [
        (
            {"action": "keep", "task_goal": None, "completion_boundary": None},
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
        ),
        (
            {
                "action": "replace",
                "task_goal": "Default goal",
                "completion_boundary": "Default boundary",
            },
            None,
            Blackboard(goal="Default goal", completion_boundary="Default boundary"),
        ),
        (
            {"action": "clear", "task_goal": None, "completion_boundary": None},
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
            None,
        ),
    ],
)
async def test_default_agent_loop_wiring_reduces_keep_replace_and_clear(
    tmp_path: Path,
    decision: dict[str, object],
    previous: Blackboard | None,
    expected: Blackboard | None,
) -> None:
    class DefaultRouter(_Router):
        def __init__(self) -> None:
            super().__init__((_response("main answer"),))
            self.direct_calls: list[
                tuple[str, Sequence[dict[str, Any]], Sequence[dict[str, Any]]]
            ] = []

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del continuation
            self.direct_calls.append((route, messages, tools))
            return _response(
                json.dumps(decision, separators=(",", ":")),
                input_tokens=5,
                output_tokens=2,
            )

    router = DefaultRouter()
    loop, session, _bus = _runtime(tmp_path, router, task_framing_outcomes=None)
    if previous is not None:
        session.update_metadata(
            blackboard={
                "goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            }
        )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("default wiring input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(router.direct_calls) == 1
    assert len(router.calls) == 1
    route, messages, tools = router.direct_calls[0]
    assert route == "chat"
    assert tools == ()
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    expected_last_task = json.dumps(
        (
            None
            if previous is None
            else {
                "task_goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            }
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    system_content = messages[0]["content"]
    assert isinstance(system_content, str)
    assert "### User input\ndefault wiring input" in system_content
    assert f"### Last Task\n```json\n{expected_last_task}\n```" in system_content
    assert system_content.endswith("### Latest assistant content\n")
    if expected is None:
        assert "blackboard" not in session.metadata
    else:
        assert session.metadata["blackboard"] == {
            "goal": expected.goal,
            "completion_boundary": expected.completion_boundary,
        }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }


@pytest.mark.asyncio
async def test_default_agent_loop_wiring_skips_router_completion_for_manual_skill(
    tmp_path: Path,
) -> None:
    class DefaultRouter(_Router):
        def __init__(self) -> None:
            super().__init__((_response("manual answer"),))
            self.direct_calls: list[tuple[str, Sequence[dict[str, Any]]]] = []

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del tools, continuation
            self.direct_calls.append((route, messages))
            return _response('{"action":"clear","goal":null,"completion_boundary":null}')

    loader = _planner_skill_loader(tmp_path)
    router = DefaultRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
        task_framing_outcomes=None,
        skill_loader=loader,
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner do it"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert router.direct_calls == []
    assert len(router.calls) == 1
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "/planner do it"
    ]
