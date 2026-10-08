from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult

from aide.agent.context.run_context import (
    AgentRunContextController,
    AgentRunContextRequestPreparer,
    AgentRunContextSnapshot,
    agent_run_attempt_guard,
)
from aide.agent.memory.manager import MemoryManager
from aide.agent.runner import AgentRunner, AgentRunnerResult
from aide.agent.session.session import Session
from aide.agent.tools.mcp import MCPTool, MCPToolSpec
from aide.agent.tools.tool_gateway import ToolGateway
from aide.agent.workspace_state import WorkspaceState
from aide.provider.model_router import ModelRouteStatus, RunModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelRoute,
    ModelUsage,
)
from aide.schedule.service import ScheduleService
from tests.fixtures import FakeClock, ScriptedFakeProvider, ScriptedFakeRouter, StreamScript
from tests.fixtures.session import seed_session_state

NOW = datetime(2026, 9, 27, tzinfo=UTC)


async def _noop(*args: object) -> None:
    del args


class _MCPSession:
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        del name, arguments
        return CallToolResult(content=[])


def _history(count: int, length: int, status: str, name: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "earlier request"}]
    for number in range(count):
        call_id = f"old-{number}"
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": f"working {number}",
                    "tool_calls": [{"id": call_id, "name": name, "arguments": "{}"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "status": status,
                    "content": "x" * length,
                    "artifact": {"path": f"artifact-{number}"},
                },
            )
        )
    return messages


async def _run_once(
    workspace: Path,
    agent_home: Path,
    history: list[dict[str, Any]],
    *,
    last_compacted: int = 0,
    enable_tool_micro_compression: bool = False,
) -> tuple[list[dict[str, object]], AgentRunnerResult, Session, ScriptedFakeProvider]:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=history,
        metadata={
            "title": "Test",
            "summary": "",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        },
        last_compacted=last_compacted,
    )
    response = ModelResponse(
        message=AssistantModelMessage(content="done"),
        usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(ModelCompleted(response),)),))
    routes: tuple[ModelRoute, ...] = ("chat", "memory")
    router = ScriptedFakeRouter(
        provider,
        route_statuses={
            route: ModelRouteStatus(
                requested_route=route,
                selected_route=route,
                provider_id="test-provider",
                model="test-model",
                context_window=65_536,
                max_output=1_024,
                used_default=False,
            )
            for route in routes
        },
    )
    adapter = RunModelRouter(router, guard=agent_run_attempt_guard)
    controller = AgentRunContextController(
        snapshot=AgentRunContextSnapshot.from_session(session),
        provider=router,
        append_summary=MemoryManager(state).append_summary,
        now=lambda: NOW,
    )
    preparer = AgentRunContextRequestPreparer(
        controller,
        router=adapter,
        requested_route="chat",
        project_messages=lambda messages: [
            {"role": "system", "content": "system"},
            *deepcopy(list(messages)),
        ],
        current_user={"role": "user", "content": "current request"},
        enable_tool_micro_compression=enable_tool_micro_compression,
    )
    remote = MCPTool(
        MCPToolSpec(
            server_name="test",
            remote_name="remote",
            model_name="mcp_remote",
            description="remote",
            parameters={"type": "object"},
        ),
        _MCPSession(),
    )
    gateway = ToolGateway(
        workspace=workspace,
        schedule_service=ScheduleService(
            workspace_state=state,
            clock=FakeClock(NOW),
            execute_user_job=_noop,
            execute_dream=_noop,
        ),
        additional_tools=(remote,),
    ).for_run(exposed_names=())
    result = await AgentRunner(adapter, preparer).run(
        [{"role": "user", "content": "current request"}],
        model="chat",
        tool_gateway=gateway,
        on_output=None,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )
    return provider.stream_requests[0].messages, result, session, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (10, 11))
@pytest.mark.parametrize("length", (512, 513))
@pytest.mark.parametrize("status", ("success", "error", "refused"))
@pytest.mark.parametrize("enabled", (False, True))
async def test_retained_result_threshold_and_length_are_provider_only(
    workspace: Path,
    agent_home: Path,
    count: int,
    length: int,
    status: str,
    enabled: bool,
) -> None:
    history = _history(count, length, status, "read_file")
    original = deepcopy(history)

    request, result, session, provider = await _run_once(
        workspace, agent_home, history, enable_tool_micro_compression=enabled
    )

    request_tools = [message for message in request if message["role"] == "tool"]
    assert len(request_tools) == count
    assert sum(
        message["content"] == "[read_file result omitted from context]" for message in request_tools
    ) == (count - 1 if enabled and count == 11 and length == 513 else 0)
    assert request_tools[-1]["content"] == "x" * length
    assert [message["artifact"] for message in request_tools] == [
        message["artifact"] for message in history if message["role"] == "tool"
    ]
    assert result.finish_reason == "completed"
    assert result.usage["model_calls"] == 1
    assert all("result omitted from context" not in str(message) for message in result.messages)
    assert history == original == session.messages
    assert provider.complete_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "eligible"),
    [
        ("exec", True),
        ("glob", True),
        ("grep", True),
        ("list_dir", True),
        ("read_file", True),
        ("web_fetch", True),
        ("web_search", True),
        ("mcp_remote", True),
        ("write_file", False),
        ("mcp_unknown", False),
    ],
)
async def test_catalog_origin_controls_omission_even_without_tool_exposure(
    workspace: Path,
    agent_home: Path,
    name: str,
    eligible: bool,
) -> None:
    request, _, _, provider = await _run_once(
        workspace, agent_home, _history(11, 513, "success", name),
        enable_tool_micro_compression=True,
    )

    assert provider.stream_requests[0].tools == ()
    assert sum(
        message.get("content") == f"[{name} result omitted from context]" for message in request
    ) == (10 if eligible else 0)


@pytest.mark.asyncio
async def test_no_completed_cycle_keeps_all_results(
    workspace: Path,
    agent_home: Path,
) -> None:
    history = [
        message
        for message in _history(11, 513, "success", "read_file")
        if message["role"] != "assistant"
    ]

    request, _, _, _ = await _run_once(
        workspace, agent_home, history, enable_tool_micro_compression=True
    )

    assert [message["content"] for message in request if message["role"] == "tool"] == [
        "x" * 513
    ] * 11


@pytest.mark.asyncio
@pytest.mark.parametrize(("count", "cursor", "omitted"), ((11, 3, 0), (11, 0, 10), (12, 3, 10)))
async def test_retained_count_and_cycle_boundary_follow_compacted_prefix(
    workspace: Path,
    agent_home: Path,
    count: int,
    cursor: int,
    omitted: int,
) -> None:
    history = _history(count, 513, "success", "read_file")

    request, _, session, _ = await _run_once(
        workspace, agent_home, history, last_compacted=cursor, enable_tool_micro_compression=True
    )

    tools = [message for message in request if message["role"] == "tool"]
    assert len(tools) == count - (cursor // 3)
    assert (
        sum(message["content"] == "[read_file result omitted from context]" for message in tools)
        == omitted
    )
    assert tools[-1]["content"] == "x" * 513
    assert session.last_compacted == cursor
