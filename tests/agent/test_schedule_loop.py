from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from mcp.types import CallToolResult

import omni.agent.context.run_context as compactor_module
from omni.agent.memory.manager import MemoryManager
from omni.agent.message_bus import InboundMessage, MessageBus
from omni.agent.permission import PermissionSnapshot, RuntimePermissionControl
from omni.agent.run_errors import CommittableAgentRunError
from omni.agent.runner import AgentRunner, AgentRunnerResult
from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.tools.base import BaseTool
from omni.agent.tools.core.exec_host import create_exec_host, resolve_exec_shell
from omni.agent.tools.deferred import RUN_BASELINE_TOOL_NAMES
from omni.agent.tools.mcp import MCPTool, MCPToolSpec
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import ModelToolCall, ToolGateway, ToolResult
from omni.agent.workspace_state import WorkspaceState
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, TURN_CANCELLED_MESSAGE, ErrorInfo
from omni.provider.errors import ModelCallError
from omni.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from omni.schedule.model import DREAM_JOB_ID, JobSchedule, ScheduleJob
from omni.schedule.service import ScheduleJobExecutionError, ScheduleService
from omni.skills.catalog import SkillLoader
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import TaskFramingRouterAdapter, collect_foreground_outbound
from tests.fixtures.agent_loop import DrivenExecutor as AgentRunExecutor
from tests.fixtures.agent_loop import loaded_skill_loader
from tests.fixtures.session import seed_session_state

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
JOB_ID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _ScheduleRouter:
    def __init__(self, *outcomes: ModelResponse | BaseException) -> None:
        self.routes: list[str] = []
        self.requests: list[tuple[list[dict[str, Any]], int]] = []
        self.tool_requests: list[tuple[str, tuple[str, ...]]] = []
        self.tool_schema_requests: list[tuple[str, tuple[dict[str, Any], ...]]] = []
        self._outcomes = list(outcomes)

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        tool_names = tuple(schema["function"]["name"] for schema in tools)
        del continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self.routes.append(route)
            self.tool_requests.append((route, tool_names))
            self.tool_schema_requests.append((route, tuple(tools)))
            self.requests.append((list(messages), 0))
            yield ModelCompleted(response=self._response())

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del continuation
        self.routes.append(route)
        self.tool_requests.append((route, tuple(schema["function"]["name"] for schema in tools)))
        self.tool_schema_requests.append((route, tuple(tools)))
        self.requests.append((list(messages), len(tools)))
        outcome = self._outcomes.pop(0) if self._outcomes else self._response()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @staticmethod
    def _response() -> ModelResponse:
        return ModelResponse(
            message=AssistantModelMessage(content="scheduled result", tool_calls=()),
            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            finish_reason="stop",
        )


class _MaxScheduleRouter(_ScheduleRouter):
    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del continuation
        self.routes.append(route)
        self.requests.append((list(messages), len(tools)))
        return ModelResponse(
            message=AssistantModelMessage(
                content="",
                tool_calls=(
                    ModelToolCall(
                        id=f"max-{len(self.requests)}",
                        name="unknown_tool",
                        arguments="{}",
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="tool_calls",
        )


class _OverlapRouter(_ScheduleRouter):
    def __init__(self) -> None:
        super().__init__()
        self.foreground_started = asyncio.Event()
        self.foreground_release = asyncio.Event()
        self.schedule_started = asyncio.Event()
        self.schedule_release = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self.foreground_started.set()
            await self.foreground_release.wait()
            yield ModelCompleted(response=self._response())

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
        self.schedule_started.set()
        await self.schedule_release.wait()
        return self._response()


class _SearchOverlapRouter(_ScheduleRouter):
    def __init__(self) -> None:
        super().__init__()
        self.foreground_after_search = asyncio.Event()
        self.schedule_after_search = asyncio.Event()
        self._foreground_calls = 0
        self._schedule_calls = 0

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del messages, continuation
        tool_names = tuple(schema["function"]["name"] for schema in tools)

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if not tools:
                yield ModelCompleted(response=self._response())
                return
            self.routes.append(route)
            self.tool_requests.append((route, tool_names))
            self.tool_schema_requests.append((route, tuple(tools)))
            self._foreground_calls += 1
            if self._foreground_calls == 1:
                yield ModelCompleted(
                    response=_tool_response(
                        call_id="foreground-search-web",
                        name="tool_search",
                        arguments={"query": "web"},
                    )
                )
                return
            self.foreground_after_search.set()
            await self.schedule_after_search.wait()
            yield ModelCompleted(response=self._response())

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del messages, continuation
        if not tools:
            return self._response()
        self.routes.append(route)
        self.tool_requests.append((route, tuple(schema["function"]["name"] for schema in tools)))
        self.tool_schema_requests.append((route, tuple(tools)))
        self._schedule_calls += 1
        if self._schedule_calls == 1:
            return _tool_response(
                call_id="schedule-search-calendar",
                name="tool_search",
                arguments={"query": "calendar"},
            )
        self.schedule_after_search.set()
        return self._response()


class _ScheduleToolOverlapRouter(_ScheduleRouter):
    def __init__(self, service: ScheduleService) -> None:
        super().__init__()
        self._service = service
        self._foreground_calls = 0
        self._schedule_calls = 0
        self.schedule_started = asyncio.Event()
        self.schedule_release = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self._foreground_calls += 1
            response = (
                _tool_response(
                    call_id="foreground_add",
                    name="schedule",
                    arguments={
                        "action": "add",
                        "message": "foreground add",
                        "every_seconds": 60,
                    },
                )
                if self._foreground_calls == 1
                else self._response()
            )
            yield ModelCompleted(response=response)

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
        self._schedule_calls += 1
        if self._schedule_calls == 1:
            self.schedule_started.set()
            await self.schedule_release.wait()
            return _tool_response(
                call_id="scheduled_add",
                name="schedule",
                arguments={
                    "action": "add",
                    "message": "recursive add",
                    "every_seconds": 60,
                },
            )
        if self._schedule_calls == 2:
            return _tool_response(
                call_id="scheduled_list",
                name="schedule",
                arguments={"action": "list"},
            )
        if self._schedule_calls == 3:
            jobs = await self._service.public_snapshot()
            assert len(jobs) == 1
            return _tool_response(
                call_id="scheduled_remove",
                name="schedule",
                arguments={"action": "remove", "job_id": jobs[0].job_id},
            )
        return self._response()


def _job(*, job_id: UUID = JOB_ID) -> ScheduleJob:
    return ScheduleJob(
        job_id=str(job_id),
        message="run the scheduled task",
        schedule=JobSchedule.at("2026-08-21T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )


def _loop(
    tmp_path: Path,
    router: _ScheduleRouter,
    *,
    config_text: str = MINIMAL_VALID_CONFIG,
    skill_loader: SkillLoader | None = None,
    externalize_result_for: Callable[[Session], Callable[[ToolResult], ToolResult]] | None = None,
    task_framing_router: TaskFramingRouterAdapter | None = None,
    mcp_tools: Sequence[BaseTool] = (),
) -> tuple[AgentRunExecutor, WorkspaceState, ScheduleService, MessageBus]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    (agent_home.path / "config.toml").write_text(config_text, encoding="utf-8")
    configuration = ConfigLoader(agent_home).load()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "agent-home")

    async def execute_user_job(job: ScheduleJob) -> None:
        del job

    async def execute_dream() -> object:
        return None

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    bus = MessageBus()
    model_router = task_framing_router or TaskFramingRouterAdapter(router)
    model_router.bind_configuration(configuration)
    loop = AgentRunExecutor(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=agent_home,
        configuration=configuration,
        bus=bus,
        schedule_service=service,
        model_router=model_router,
        memory_manager=MemoryManager(state),
        session_id=None,
        now=lambda: NOW,
        new_uuid=lambda: JOB_ID,
        monotonic_now=lambda: 0.0,
        exec_host=create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell)),
        permission_control=RuntimePermissionControl(configuration.runtime.permission_level),
        mcp_tools=mcp_tools,
        skill_loader=loaded_skill_loader(agent_home, configuration) if skill_loader is None else skill_loader,
    )
    if skill_loader is not None:
        loop._skill_loader = skill_loader
        loop._context_builder._skill_loader = skill_loader
    if externalize_result_for is not None:
        object.__setattr__(loop, "_result_externalizer_for", externalize_result_for)
    return loop, state, service, bus


class _Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        await asyncio.Event().wait()


async def _assert_no_outbound(bus: MessageBus) -> None:
    outbound = asyncio.create_task(bus.get_outbound())
    done, _ = await asyncio.wait((outbound,), timeout=0)
    assert not done
    outbound.cancel()
    await asyncio.gather(outbound, return_exceptions=True)


@pytest.mark.asyncio
async def test_schedule_run_uses_schedule_session_and_keeps_foreground_bus_empty(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter()
    framing_router = TaskFramingRouterAdapter(router)
    loop, state, _, _bus = _loop(tmp_path, router, task_framing_router=framing_router)

    outbound = asyncio.create_task(_bus.get_outbound())
    await loop.run_schedule_job(_job())

    done, _ = await asyncio.wait((outbound,), timeout=0)
    assert not done
    outbound.cancel()
    await asyncio.gather(outbound, return_exceptions=True)

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in schedule_session.messages] == [
        "user",
        "assistant",
    ]
    assert schedule_session.messages[-1]["content"] == "scheduled result"
    assert router.routes == ["schedule"]
    assert router.requests[0][1] == len(RUN_BASELINE_TOOL_NAMES)
    assert framing_router.framing_requests == []


@pytest.mark.asyncio
async def test_schedule_context_and_runner_share_exactly_one_run_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _state, _service, _bus = _loop(tmp_path, _ScheduleRouter())
    created_gateways: list[ToolGateway] = []
    context_gateways: list[ToolGateway] = []
    runner_gateways: list[ToolGateway] = []
    original_new_run_gateway = loop._new_run_gateway
    original_prepare = loop._prepare_agent_run
    original_run = AgentRunner.run

    def new_run_gateway(
        *,
        excluded_names: Sequence[str] = (),
        permission_snapshot: PermissionSnapshot | None = None,
        permission_context: PermissionContext | None = None,
    ) -> ToolGateway:
        gateway = original_new_run_gateway(
            excluded_names=excluded_names,
            permission_snapshot=permission_snapshot,
            permission_context=permission_context,
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

    object.__setattr__(loop, "_new_run_gateway", new_run_gateway)
    object.__setattr__(loop, "_prepare_agent_run", prepare)
    monkeypatch.setattr(AgentRunner, "run", run)

    await loop.run_schedule_job(_job())

    assert len(created_gateways) == 1
    assert len(context_gateways) == 1
    assert len(runner_gateways) == 1
    assert created_gateways[0] is context_gateways[0] is runner_gateways[0]
    assert created_gateways[0].exposed_names == RUN_BASELINE_TOOL_NAMES
    assert "schedule" not in {tool.name for tool in created_gateways[0].catalog}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        (
            "add",
            {"action": "add", "message": "must stay absent", "every_seconds": 60},
        ),
        ("list", {"action": "list"}),
        (
            "remove",
            {"action": "remove", "job_id": str(JOB_ID)},
        ),
    ],
)
async def test_schedule_run_excludes_schedule_tool_from_catalog_and_schema(
    tmp_path: Path,
    action: str,
    arguments: dict[str, object],
) -> None:
    router = _ScheduleRouter(
        _tool_response(call_id=f"call_{action}", name="schedule", arguments=arguments),
        _ScheduleRouter._response(),
    )
    loop, state, service, _bus = _loop(tmp_path, router)

    await loop.run_schedule_job(_job())

    assert router.requests[0][1] == len(RUN_BASELINE_TOOL_NAMES)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    assert tool_message["status"] == "error"
    assert tool_message["content"] == "The requested tool is not available."
    assert await service.public_snapshot() == ()


@pytest.mark.asyncio
async def test_schedule_run_search_activates_deferred_tools_for_the_next_request(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter(
        _tool_response(
            call_id="call_search",
            name="tool_search",
            arguments={"query": "web"},
        ),
        _ScheduleRouter._response(),
    )
    loop, _state, _service, _bus = _loop(tmp_path, router)

    await loop.run_schedule_job(_job())

    assert router.tool_requests == [
        ("schedule", RUN_BASELINE_TOOL_NAMES),
        (
            "schedule",
            (*RUN_BASELINE_TOOL_NAMES[:7], "web_search", "web_fetch", "tool_search"),
        ),
    ]


@pytest.mark.asyncio
async def test_schedule_initial_request_has_zero_schema_cost_for_100_unused_mcp_tools(
    tmp_path: Path,
) -> None:
    class SyntheticMCPSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            return CallToolResult(content=[])

    session = SyntheticMCPSession()
    mcp_tools = tuple(
        MCPTool(
            MCPToolSpec(
                server_name="synthetic",
                remote_name=f"large_tool_{index}",
                model_name=f"mcp_synthetic_large_tool_{index}",
                description="Synthetic cost measurement Tool.",
                parameters={
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "string",
                            "description": "x" * 2048,
                        }
                    },
                },
            ),
            session,
        )
        for index in range(100)
    )
    baseline_router = _ScheduleRouter()
    baseline_loop, _state, _service, _bus = _loop(
        tmp_path / "baseline",
        baseline_router,
    )
    large_router = _ScheduleRouter()
    large_loop, _state, _service, _bus = _loop(
        tmp_path / "large",
        large_router,
        mcp_tools=mcp_tools,
    )

    await baseline_loop.run_schedule_job(_job())
    await large_loop.run_schedule_job(_job())

    baseline_tools = baseline_router.tool_schema_requests[0][1]
    large_tools = large_router.tool_schema_requests[0][1]
    assert tuple(schema["function"]["name"] for schema in large_tools) == (RUN_BASELINE_TOOL_NAMES)
    assert "schedule" not in {
        tool.name for tool in large_loop._new_run_gateway(excluded_names=("schedule",)).catalog
    }
    assert len(large_loop._tool_gateway.catalog) == 110
    assert len(json.dumps(large_tools, sort_keys=True).encode("utf-8")) == len(
        json.dumps(baseline_tools, sort_keys=True).encode("utf-8")
    )


@pytest.mark.asyncio
async def test_user_schedule_run_keeps_and_executes_generation_mcp_tool(
    tmp_path: Path,
) -> None:
    class RecordingMCPSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            self.calls.append((name, arguments))
            return CallToolResult(content=[])

    mcp_session = RecordingMCPSession()
    mcp_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="schedule_meeting",
            model_name="mcp_alpha_schedule_meeting",
            description="Schedule an external meeting.",
            parameters={"type": "object"},
        ),
        mcp_session,
    )
    router = _ScheduleRouter(
        _tool_response(
            call_id="call_mcp_schedule",
            name="mcp_alpha_schedule_meeting",
            arguments={"title": "review", "timezone": "Asia/Shanghai"},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _service, bus = _loop(tmp_path, router, mcp_tools=(mcp_tool,))

    await loop.start()
    try:
        await collect_foreground_outbound(bus, "foreground input")
        await loop.run_schedule_job(_job())
    finally:
        await loop.close()

    schedule_requests = [
        (route, tool_names) for route, tool_names in router.tool_requests if route == "schedule"
    ]
    assert [route for route, _tool_names in schedule_requests] == [
        "schedule",
        "schedule",
    ]
    assert all(tool_names == RUN_BASELINE_TOOL_NAMES for _route, tool_names in schedule_requests)
    assert all(
        "mcp_alpha_schedule_meeting" not in tool_names for _route, tool_names in schedule_requests
    )
    assert mcp_session.calls == [
        ("schedule_meeting", {"title": "review", "timezone": "Asia/Shanghai"})
    ]
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    assert tool_message["status"] == "success"


@pytest.mark.asyncio
async def test_schedule_run_rejects_system_job_before_session_creation(tmp_path: Path) -> None:
    loop, state, _service, _bus = _loop(tmp_path, _ScheduleRouter())
    system_job = ScheduleJob(
        job_id=DREAM_JOB_ID,
        source="system",
        message="internal dream placeholder",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )

    with pytest.raises(ScheduleJobExecutionError) as captured:
        await loop.run_schedule_job(system_job)

    assert captured.value.error.code == "schedule_state_error"
    assert not state.schedule_sessions_directory.exists()


@pytest.mark.asyncio
async def test_schedule_run_drains_session_persist_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _state, _service, _bus = _loop(tmp_path, _ScheduleRouter())
    persist_started = asyncio.Event()
    release_persist = asyncio.Event()
    captured: list[Session] = []
    original_persist_after = Session._persist_after

    async def blocked_persist_after(
        session: Session,
        previous: asyncio.Task[None] | None,
        content: bytes,
    ) -> None:
        persist_started.set()
        await release_persist.wait()
        await original_persist_after(session, previous, content)

    async def persist_only(session: Session, job: ScheduleJob) -> None:
        del job
        captured.append(session)
        seed_session_state(
            session,
            messages=[
                {
                    "role": "user",
                    "content": "queued persistence",
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
                "summary": "",
            },
            last_compacted=0,
        )
        session.persist()

    monkeypatch.setattr(Session, "_persist_after", blocked_persist_after)
    object.__setattr__(loop, "_run_schedule_agent", persist_only)
    running = asyncio.create_task(loop.run_schedule_job(_job()))

    await persist_started.wait()
    await asyncio.sleep(0)
    assert not running.done()

    release_persist.set()
    await running
    assert captured
    assert captured[0]._persist_tasks == set()


@pytest.mark.asyncio
async def test_schedule_run_reloads_canonical_session_and_closes_each_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _ScheduleRouter()
    observed_context: list[tuple[str, int]] = []
    loop, state, _, _bus = _loop(tmp_path, router)
    original_prepare = loop._prepare_agent_run

    async def prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        try:
            current = Session.load(
                state,
                f"schedule_{JOB_ID}",
                partition=SessionStoragePartition.SCHEDULE,
            )
        except FileNotFoundError:
            message_count = 0
        else:
            message_count = len(current.messages)
        observed_context.append((f"schedule_{JOB_ID}", message_count))
        return await original_prepare(context, tool_gateway=tool_gateway)

    object.__setattr__(loop, "_prepare_agent_run", prepare)
    persisted: list[str] = []
    closed: list[str] = []
    original_persist = Session.persist
    original_close = Session.close

    def record_persist(session: Session) -> None:
        persisted.append(session.session_id)
        original_persist(session)

    def record_close(session: Session) -> None:
        closed.append(session.session_id)
        original_close(session)

    monkeypatch.setattr(Session, "persist", record_persist)
    monkeypatch.setattr(Session, "close", record_close)

    await loop.run_schedule_job(_job())
    await loop.run_schedule_job(_job())

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert observed_context == [(f"schedule_{JOB_ID}", 0), (f"schedule_{JOB_ID}", 2)]
    assert persisted == [f"schedule_{JOB_ID}", f"schedule_{JOB_ID}"]
    assert closed == [f"schedule_{JOB_ID}", f"schedule_{JOB_ID}"]
    assert [message["role"] for message in schedule_session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_schedule_failed_runner_persists_safe_error_and_maps_job_failure(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter(ModelCallError(ErrorInfo("provider_unavailable", "safe failure")))
    loop, state, _, _bus = _loop(tmp_path, router)

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(_job())

    assert raised.value.error == ErrorInfo("provider_unavailable", "safe failure")
    await _assert_no_outbound(_bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["role"] == "assistant"
    assert schedule_session.messages[-1]["status"] == "error"
    assert schedule_session.messages[-1]["error"] == {
        "code": "provider_unavailable",
        "message": "safe failure",
    }


@pytest.mark.asyncio
async def test_schedule_max_iterations_finishes_tools_without_a_fifty_first_model_call(
    tmp_path: Path,
) -> None:
    router = _MaxScheduleRouter()
    loop, state, _, _bus = _loop(tmp_path, router)

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(_job())

    assert raised.value.error.code == "agent_iteration_limit"
    assert len(router.requests) == 50
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["error"]["code"] == "agent_iteration_limit"


@pytest.mark.asyncio
async def test_schedule_cancelled_runner_persists_user_and_propagates_cancelled_error(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter()
    loop, state, service, _bus = _loop(tmp_path, router)
    await service.close()

    with pytest.raises(asyncio.CancelledError):
        await loop.run_schedule_job(_job())

    await _assert_no_outbound(_bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in schedule_session.messages] == ["user", "assistant"]
    assert schedule_session.messages[-1]["status"] == "interrupted"
    assert schedule_session.messages[-1]["error"]["code"] == "turn_cancelled"
    assert router.requests == []


@pytest.mark.asyncio
async def test_schedule_context_preparation_failures_preserve_cancel(
    tmp_path: Path,
) -> None:
    success_loop, _, _, _success_bus = _loop(tmp_path / "success", _ScheduleRouter())
    await success_loop.run_schedule_job(_job())

    failed_loop, failed_state, _, _failed_bus = _loop(tmp_path / "failed", _ScheduleRouter())

    async def unexpected_prepare(
        context: Any, *, tool_gateway: ToolGateway
    ) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise RuntimeError("unexpected preparation failure")

    object.__setattr__(failed_loop, "_prepare_agent_run", unexpected_prepare)
    with pytest.raises(ScheduleJobExecutionError) as failed:
        await failed_loop.run_schedule_job(_job())
    assert failed.value.error.code == "model_failed"

    cancelled_loop, cancelled_state, _, _cancelled_bus = _loop(
        tmp_path / "cancelled", _ScheduleRouter()
    )

    async def cancelled_prepare(context: Any, *, tool_gateway: ToolGateway) -> list[dict[str, Any]]:
        del context, tool_gateway
        raise asyncio.CancelledError()

    object.__setattr__(cancelled_loop, "_prepare_agent_run", cancelled_prepare)
    with pytest.raises(asyncio.CancelledError):
        await cancelled_loop.run_schedule_job(_job())

    failed_session = Session.load(
        failed_state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    cancelled_session = Session.load(
        cancelled_state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in failed_session.messages] == ["user", "assistant"]
    assert [message["role"] for message in cancelled_session.messages] == ["user", "assistant"]
    assert cancelled_session.messages[-1]["status"] == "interrupted"
    assert cancelled_session.messages[-1]["error"]["code"] == "turn_cancelled"


@pytest.mark.asyncio
async def test_schedule_run_start_hard_overflow_does_not_commit_or_call_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MINIMAL_VALID_CONFIG.replace("max_output = 1024", "max_output = 7000")
    router = _ScheduleRouter()
    loop, state, _, bus = _loop(tmp_path, router, config_text=config)
    job = replace(_job(), message="x" * 20_000)
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

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(job)

    assert raised.value.error.code == "model_context_overflow"
    assert router.routes == []
    assert commit_calls == 0
    await _assert_no_outbound(bus)
    assert not (state.schedule_sessions_directory / f"schedule_{JOB_ID}.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "error"),
    (
        ("run_start", ErrorInfo("provider_unavailable", "Summary provider failed.")),
        (
            "react",
            ErrorInfo("model_context_overflow", MODEL_CONTEXT_OVERFLOW_MESSAGE),
        ),
    ),
)
async def test_schedule_commits_summary_failure_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["run_start", "react"],
    error: ErrorInfo,
) -> None:
    router = _ScheduleRouter()
    loop, state, _, bus = _loop(tmp_path, router)
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

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(_job())

    assert raised.value.error == error
    assert commit_calls == 1
    assert router.routes == []
    await _assert_no_outbound(bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in schedule_session.messages] == ["user", "assistant"]
    assert schedule_session.messages[-1]["status"] == "error"
    assert schedule_session.messages[-1]["error"] == {
        "code": error.code,
        "message": error.message,
    }


@pytest.mark.asyncio
async def test_schedule_commits_cancelled_summary_failure_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, state, _, bus = _loop(tmp_path, _ScheduleRouter())
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

    with pytest.raises(asyncio.CancelledError):
        await loop.run_schedule_job(_job())

    assert commit_calls == 1
    await _assert_no_outbound(bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["status"] == "interrupted"
    assert schedule_session.messages[-1]["error"] == {
        "code": error.code,
        "message": error.message,
    }


def _tool_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content="",
            tool_calls=(
                ModelToolCall(
                    id=call_id,
                    name=name,
                    arguments=json.dumps(arguments, separators=(",", ":")),
                ),
            ),
        ),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_schedule_confirmation_required_tool_is_refused_without_foreground_control(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    router = _ScheduleRouter(
        _tool_response(
            call_id="call_external",
            name="read_file",
            arguments={"path": str(outside)},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _, _bus = _loop(tmp_path, router)
    confirmation_requests: list[object] = []
    loop.control.bind_confirmation_callback(confirmation_requests.append)

    await loop.run_schedule_job(_job())

    await _assert_no_outbound(_bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    assert tool_message["status"] == "refused"
    assert tool_message["confirmation"]["decision"] is None
    assert confirmation_requests == []


@pytest.mark.asyncio
async def test_schedule_agent_reads_known_skill_path_via_shared_gateway(tmp_path: Path) -> None:
    skill_root = tmp_path / "agent-home" / "skills"
    skill_file = skill_root / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    router = _ScheduleRouter(
        _tool_response(
            call_id="scheduled_skill_read",
            name="read_file",
            arguments={"path": str(skill_file)},
        ),
        _ScheduleRouter._response(),
    )
    skill_loader = SkillLoader(
        root=skill_root,
        reserved_names=(),
        enable_always_load=False,
    )
    skill_loader.load()
    loop, state, _, _bus = _loop(
        tmp_path,
        router,
        skill_loader=skill_loader,
    )
    confirmation_requests: list[object] = []
    loop.control.bind_confirmation_callback(confirmation_requests.append)

    await loop.run_schedule_job(_job())

    await _assert_no_outbound(_bus)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_messages = [message for message in schedule_session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["status"] == "success"
    assert tool_messages[0]["content"] == "---\nname: review\n---\nbody\n"
    assert confirmation_requests == []
    assert router.routes == ["schedule", "schedule"]
    assert [tool_count for _, tool_count in router.requests] == [
        len(RUN_BASELINE_TOOL_NAMES),
        len(RUN_BASELINE_TOOL_NAMES),
    ]


@pytest.mark.asyncio
async def test_schedule_oversized_result_uses_canonical_schedule_artifact_session(
    tmp_path: Path,
) -> None:
    def externalizer_for(session: Session) -> Callable[[ToolResult], ToolResult]:
        def externalize(result: ToolResult) -> ToolResult:
            if result.status != "success":
                return result
            output = BaseTool.handle_result(
                result.content,
                workspace=session.workspace_state.workspace_path,
                session_id=session.session_id,
                tool_call_id=result.tool_call_id,
                limit=4,
            )
            return replace(result, content=output.content, artifact=output.artifact)

        return externalize

    router = _ScheduleRouter(
        _tool_response(
            call_id="call_artifact",
            name="read_file",
            arguments={"path": "large.txt", "limit": 10000},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _, _bus = _loop(
        tmp_path,
        router,
        externalize_result_for=externalizer_for,
    )
    (state.workspace_path / "large.txt").write_text("x\n" * 4000, encoding="utf-8")

    await loop.run_schedule_job(_job())

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    artifact = tool_message["artifact"]
    assert artifact["path"].startswith(f".omni/artifacts/schedule_{JOB_ID}/")
    assert (state.workspace_path / artifact["path"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
async def test_schedule_agent_loop_micro_compression_keeps_artifacts_and_session_history_complete(
    tmp_path: Path,
    enabled: bool,
) -> None:
    router = _ScheduleRouter(
        *(
            [
                _tool_response(
                    call_id=f"call_{index}",
                    name="read_file",
                    arguments={"path": "large.txt", "limit": 10000},
                )
                for index in range(12)
            ]
            + [_ScheduleRouter._response()]
        )
    )
    loop, state, _service, _bus = _loop(
        tmp_path,
        router,
        config_text=MINIMAL_VALID_CONFIG.replace(
            "compact_ratio = 0.9",
            "compact_ratio = 0.9\nmax_tool_result_chars = 1000\n"
            f"enable_tool_micro_compression = {str(enabled).lower()}",
        ),
    )
    (state.workspace_path / "large.txt").write_text("x" * 4000, encoding="utf-8")

    try:
        await loop.run_schedule_job(_job())
    finally:
        await loop.close()

    assert len(router.requests) == 13
    previous_request = router.requests[-2][0]
    previous_tools = [message for message in previous_request if message.get("role") == "tool"]
    assert (
        sum(
            message["content"] == "[read_file result omitted from context]"
            for message in previous_tools
        )
        == (10 if enabled else 0)
    )
    assert len(previous_tools[-1]["content"]) > 512

    final_request = router.requests[-1][0]
    final_tools = [message for message in final_request if message.get("role") == "tool"]
    assert len(final_tools) == 12
    assert (
        sum(
            message["content"] == "[read_file result omitted from context]"
            for message in final_tools
        )
        == (11 if enabled else 0)
    )
    assert len(final_tools[-1]["content"]) > 512

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    persisted_tools = [
        message for message in schedule_session.messages if message.get("role") == "tool"
    ]
    assert len(persisted_tools) == 12
    assert all(len(message["content"]) > 512 for message in persisted_tools)
    assert all(
        message["content"] != "[read_file result omitted from context]"
        for message in persisted_tools
    )
    for message in persisted_tools:
        artifact = message["artifact"]
        assert isinstance(artifact, dict)
        artifact_content = (state.workspace_path / artifact["path"]).read_text(encoding="utf-8")
        assert artifact_content == "x" * 4000
        assert "result omitted from context" not in artifact_content


@pytest.mark.asyncio
async def test_foreground_cancel_does_not_cancel_overlapping_schedule_run(
    tmp_path: Path,
) -> None:
    router = _OverlapRouter()
    loop, state, _, _bus = _loop(tmp_path, router)
    await loop.start()
    await _bus.put_inbound(InboundMessage("foreground input"))
    await router.foreground_started.wait()

    schedule_task = asyncio.create_task(loop.run_schedule_job(_job()))
    await router.schedule_started.wait()
    await loop.cancel_active_run()
    assert not schedule_task.done()

    foreground_terminal = await _bus.get_outbound()
    assert foreground_terminal.type == "system_control"
    assert foreground_terminal.metadata["finish_reason"] == "cancelled"

    router.schedule_release.set()
    await schedule_task
    await loop.close()

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["content"] == "scheduled result"


@pytest.mark.asyncio
async def test_schedule_run_uses_isolated_catalog_during_concurrent_foreground_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "agent-home")
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    (agent_home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(agent_home).load()

    async def execute_user_job(job: ScheduleJob) -> None:
        del job

    async def execute_dream() -> object:
        return None

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    router = _ScheduleToolOverlapRouter(service)
    model_router = TaskFramingRouterAdapter(router)
    model_router.bind_configuration(configuration)
    bus = MessageBus()
    loop = AgentRunExecutor(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=agent_home,
        configuration=configuration,
        bus=bus,
        schedule_service=service,
        model_router=model_router,
        memory_manager=MemoryManager(state),
        session_id=None,
        now=lambda: NOW,
        new_uuid=lambda: JOB_ID,
        monotonic_now=lambda: 0.0,
        exec_host=create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell)),
        permission_control=RuntimePermissionControl(configuration.runtime.permission_level),
        skill_loader=loaded_skill_loader(agent_home, configuration),
    )
    await loop.start()
    schedule_task = asyncio.create_task(loop.run_schedule_job(_job()))
    try:
        await router.schedule_started.wait()
        assert not schedule_task.done()

        await bus.put_inbound(InboundMessage("create a foreground Schedule Job"))
        while True:
            outbound = await bus.get_outbound()
            if outbound.metadata.get("_streamed") is True:
                break

        foreground_jobs = await service.public_snapshot()
        assert len(foreground_jobs) == 1
        assert foreground_jobs[0].message == "foreground add"
        assert not schedule_task.done()

        schedule_outbound = asyncio.create_task(bus.get_outbound())
        router.schedule_release.set()
        await schedule_task
        done, _ = await asyncio.wait((schedule_outbound,), timeout=0)
        assert not done
        schedule_outbound.cancel()
        await asyncio.gather(schedule_outbound, return_exceptions=True)
    finally:
        router.schedule_release.set()
        await asyncio.gather(schedule_task, return_exceptions=True)
        await loop.close()

    foreground_jobs = await service.public_snapshot()
    assert len(foreground_jobs) == 1
    assert foreground_jobs[0].message == "foreground add"

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    scheduled_tools = [
        message for message in schedule_session.messages if message["role"] == "tool"
    ]
    assert [message["status"] for message in scheduled_tools] == ["error", "error", "error"]
    assert all(
        message["content"] == "The requested tool is not available." for message in scheduled_tools
    )


@pytest.mark.asyncio
async def test_concurrent_foreground_and_schedule_searches_keep_exposure_isolated(
    tmp_path: Path,
) -> None:
    class RecordingMCPSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            return CallToolResult(content=[])

    mcp_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object"},
        ),
        RecordingMCPSession(),
    )
    router = _SearchOverlapRouter()
    loop, _state, _service, bus = _loop(tmp_path, router, mcp_tools=(mcp_tool,))

    await loop.start()
    foreground = asyncio.create_task(
        collect_foreground_outbound(bus, "Search for web capabilities.")
    )
    try:
        await router.foreground_after_search.wait()
        scheduled = asyncio.create_task(loop.run_schedule_job(_job()))
        await asyncio.gather(foreground, scheduled)
    finally:
        await loop.close()

    foreground_requests = [names for route, names in router.tool_requests if route == "chat"]
    schedule_requests = [names for route, names in router.tool_requests if route == "schedule"]
    assert foreground_requests == [
        RUN_BASELINE_TOOL_NAMES,
        (*RUN_BASELINE_TOOL_NAMES[:7], "web_search", "web_fetch", "tool_search"),
    ]
    assert schedule_requests == [
        RUN_BASELINE_TOOL_NAMES,
        (*RUN_BASELINE_TOOL_NAMES[:7], "mcp_alpha_calendar_events", "tool_search"),
    ]
    assert "mcp_alpha_calendar_events" not in foreground_requests[-1]
    assert "web_search" not in schedule_requests[-1]
    assert "schedule" not in schedule_requests[-1]
