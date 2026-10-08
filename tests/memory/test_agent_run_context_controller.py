from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from aide.agent.context.budget import (
    estimate_request_tokens,
    estimate_run_slice_tokens,
    request_fits_model_context,
)
from aide.agent.context.run_context import (
    AgentRunContextController,
    AgentRunContextRequestPreparer,
    AgentRunContextSnapshot,
    agent_run_attempt_guard,
    latest_main_agent_usage_anchor,
)
from aide.agent.memory.manager import MemoryManager
from aide.agent.run_errors import CommittableAgentRunError
from aide.agent.runner import AgentRunner
from aide.agent.session.session import Session
from aide.agent.tools.tool_gateway import ModelToolCall, ToolResult
from aide.agent.workspace_state import WorkspaceState
from aide.config.config import (
    MemoryConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from aide.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter, ModelRouteStatus, RunModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelRoute,
    ModelUsage,
)
from tests.fixtures import FakeClock, ScriptedFakeProvider, ScriptedFakeRouter, StreamScript
from tests.fixtures.session import seed_session_state

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _router_configuration(
    *,
    chat_context_window: int,
    default_context_window: int,
    max_output: int = 10,
) -> UserConfiguration:
    chat_provider = ProviderConfiguration(
        provider_id="chat-provider",
        protocol="openai-compatible",
        base_url="https://chat.example/v1",
        api_key="chat-secret",
        models=("chat-model",),
    )
    default_provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="anthropic",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models=("default-model",),
    )

    def route(provider_id: str, model: str, context_window: int) -> RouteConfiguration:
        return RouteConfiguration(
            provider_id=provider_id,
            model=model,
            context_window=context_window,
            max_output=max_output,
            temperature=0,
            reasoning_effort="mid",
            timeout=30,
        )

    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=50_000),
        memory=MemoryConfiguration(
            batch_size=10,
            schedule="0 * * * *",
        ),
        models=ModelsConfiguration(
            providers={
                chat_provider.provider_id: chat_provider,
                default_provider.provider_id: default_provider,
            },
            routes={
                "chat": route("chat-provider", "chat-model", chat_context_window),
                "default": route(
                    "default-provider",
                    "default-model",
                    default_context_window,
                ),
            },
        ),
    )


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".aide")
    return state


def _response(content: str, *, input_tokens: int = 20, output_tokens: int = 5) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason="stop",
    )


def _usage(input_tokens: int = 4, output_tokens: int = 2) -> dict[str, int]:
    return {
        "model_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _context_usage(
    *,
    requested_route: str = "chat",
    provider_id: str = "provider",
    context_window: int = 1_600,
) -> dict[str, object]:
    return {
        "requested_route": requested_route,
        "selected_route": "chat",
        "provider_id": provider_id,
        "model": "model",
        "context_window": context_window,
        "max_output": 200,
        "anchor_estimated_tokens": 20,
        "estimator_version": "utf8-bytes-div4-v1",
        "run_projected_tokens": 80,
        "run_projection_source": "estimated",
    }


def _assistant_with_usage(
    content: str,
    *,
    context_usage: object = None,
    token_usage: object = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
        "status": "completed",
        "error": None,
    }
    if context_usage is not None:
        message["context_usage"] = context_usage
    if token_usage is not None:
        message["token_usage"] = token_usage
    return message


def test_latest_main_agent_usage_anchor_returns_a_detached_latest_assistant_anchor() -> None:
    usage = _usage(100, 10)
    messages: list[dict[str, Any]] = [
        _assistant_with_usage(
            "latest answer",
            context_usage=_context_usage(),
            token_usage=usage,
        ),
        {"role": "user", "content": "later user"},
        {"role": "tool", "content": "later tool"},
    ]

    anchor = latest_main_agent_usage_anchor(messages)

    assert anchor is not None
    context, copied_usage = anchor
    assert context.provider_id == "provider"
    assert copied_usage == usage
    copied_usage["input_tokens"] = 999
    assert usage["input_tokens"] == 100


@pytest.mark.parametrize(
    "usage",
    (
        pytest.param(_usage(0, 5), id="zero-input"),
        pytest.param(_usage(5, 0), id="zero-output"),
        pytest.param(_usage(0, 0), id="zero-total"),
    ),
)
def test_latest_main_agent_usage_anchor_accepts_zero_usage_values(
    usage: dict[str, int],
) -> None:
    anchor = latest_main_agent_usage_anchor(
        [
            _assistant_with_usage(
                "answer",
                context_usage=_context_usage(),
                token_usage=usage,
            )
        ]
    )

    assert anchor is not None
    assert anchor[1] == usage


@pytest.mark.parametrize(
    "latest",
    (
        pytest.param(
            _assistant_with_usage("missing context", token_usage=_usage()),
            id="context-missing",
        ),
        pytest.param(
            _assistant_with_usage(
                "malformed context",
                context_usage={"requested_route": "chat"},
                token_usage=_usage(),
            ),
            id="context-malformed",
        ),
        pytest.param(
            _assistant_with_usage("missing usage", context_usage=_context_usage()),
            id="usage-missing",
        ),
        pytest.param(
            _assistant_with_usage(
                "malformed usage",
                context_usage=_context_usage(),
                token_usage={**_usage(), "total_tokens": 999},
            ),
            id="usage-total-mismatch",
        ),
        pytest.param(
            _assistant_with_usage(
                "boolean usage",
                context_usage=_context_usage(),
                token_usage={**_usage(), "input_tokens": True},
            ),
            id="usage-boolean",
        ),
        pytest.param(
            _assistant_with_usage(
                "negative usage",
                context_usage=_context_usage(),
                token_usage=_usage(-1, 2),
            ),
            id="usage-negative",
        ),
        pytest.param(
            _assistant_with_usage(
                "incomplete usage",
                context_usage=_context_usage(),
                token_usage={"model_calls": 1, "input_tokens": 4, "output_tokens": 2},
            ),
            id="usage-missing-field",
        ),
        pytest.param(
            _assistant_with_usage(
                "cumulative usage",
                context_usage=_context_usage(),
                token_usage={**_usage(), "model_calls": 2},
            ),
            id="usage-multiple-model-calls",
        ),
        pytest.param(
            _assistant_with_usage(
                "unsupported route",
                context_usage=_context_usage(requested_route="memory"),
                token_usage=_usage(),
            ),
            id="requested-route-unsupported",
        ),
    ),
)
def test_latest_main_agent_usage_anchor_stops_at_an_invalid_latest_assistant(
    latest: dict[str, object],
) -> None:
    messages = [
        _assistant_with_usage(
            "older valid answer",
            context_usage=_context_usage(),
            token_usage=_usage(100, 10),
        ),
        latest,
    ]

    assert latest_main_agent_usage_anchor(messages) is None


def _assistant_history_message(
    content: str,
    *,
    timestamp: str,
    token_usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "timestamp": timestamp,
        "tool_calls": [],
        "status": "completed",
        "error": None,
        "token_usage": token_usage,
    }


def _tool_run_history(
    label: str,
    *,
    timestamp: str,
    token_usage: dict[str, int],
    result_size: int = 240,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": f"{label} tool call",
            "timestamp": timestamp,
            "tool_calls": [{"id": f"call-{label}", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": token_usage,
        },
        {
            "role": "tool",
            "content": f"{label} tool result " + "r" * result_size,
            "timestamp": timestamp,
            "tool_call_id": f"call-{label}",
            "name": "read_file",
            "status": "success",
            "artifact": None,
        },
    ]


def _run_history(
    label: str,
    *,
    timestamp: str,
    token_usage: dict[str, int],
    size: int = 500,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": f"{label} user " + "u" * size,
            "timestamp": timestamp,
        },
        _assistant_history_message(
            f"{label} assistant " + "a" * size,
            timestamp=timestamp,
            token_usage=token_usage,
        ),
    ]


def _react_cycle(label: str, *, size: int = 80) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": f"{label} assistant " + "a" * size,
            "tool_calls": [{"id": f"{label}-call", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": _usage(),
        },
        {
            "role": "tool",
            "tool_call_id": f"{label}-call",
            "name": "read_file",
            "status": "success",
            "content": f"{label} result " + "r" * size,
            "artifact": None,
        },
    ]


def _controller(
    workspace: Path,
    session: Session,
    provider: ScriptedFakeProvider,
) -> AgentRunContextController:
    state = session.workspace_state
    return AgentRunContextController(
        snapshot=AgentRunContextSnapshot.from_session(session),
        provider=ScriptedFakeRouter(provider),
        append_summary=MemoryManager(state).append_summary,
        now=lambda: NOW,
    )


async def _prepare_controller(
    controller: AgentRunContextController,
    *,
    current_user: str = "new user",
    context_window: int = 1_800,
    max_output: int = 200,
    compact_ratio: float = 0.5,
    tools: Sequence[dict[str, Any]] = (),
    memory_route_status: ModelRouteStatus | None = None,
    provider_id: str = "",
    model: str = "",
) -> tuple[dict[str, Any], ...]:
    route_status = _chat_status(
        context_window=context_window,
        max_output=max_output,
        provider_id=provider_id,
        model=model,
    )
    return await controller.prepare_run_start(
        project_messages=_project_messages,
        current_user={"role": "user", "content": current_user},
        route_status=route_status,
        compact_ratio=compact_ratio,
        tools=tools,
        memory_route_status=memory_route_status
        or _memory_status(
            context_window=context_window,
            max_output=max_output,
        ),
    )


def _chat_status(
    *,
    context_window: int,
    max_output: int = 100,
    provider_id: str = "",
    model: str = "",
) -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id=provider_id,
        model=model,
        context_window=context_window,
        max_output=max_output,
        used_default=False,
    )


def _memory_status(*, context_window: int, max_output: int = 100) -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="memory-provider",
        model="memory-model",
        context_window=context_window,
        max_output=max_output,
        used_default=False,
    )


def _context_router(
    provider: ScriptedFakeProvider,
    *,
    chat_status: ModelRouteStatus | None = None,
    memory_status: ModelRouteStatus | None = None,
) -> RunModelRouter:
    statuses: dict[ModelRoute, ModelRouteStatus] = {
        "chat": chat_status
        or _chat_status(
            context_window=16_384,
            max_output=1_024,
            provider_id="test-provider",
            model="test-model",
        ),
        "memory": memory_status or _memory_status(context_window=16_384, max_output=1_024),
    }
    return RunModelRouter(
        ScriptedFakeRouter(provider, route_statuses=statuses),
        guard=agent_run_attempt_guard,
    )


def _project_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "SYSTEM"},
        *[
            {
                "role": message["role"],
                "content": message["content"],
                **(
                    {"tool_call_id": message["tool_call_id"], "name": message["name"]}
                    if message["role"] == "tool"
                    else {}
                ),
            }
            for message in messages
        ],
    ]


def _project_messages_with_tool_calls(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "SYSTEM"},
        *deepcopy(list(messages)),
    ]


def _summary_payload(
    provider: ScriptedFakeProvider,
    request_index: int = 0,
) -> list[dict[str, Any]]:
    content = provider.complete_requests[request_index].messages[1]["content"]
    assert isinstance(content, str)
    _prefix, marker, tail = content.partition("```json\n")
    assert marker
    serialized, marker, _suffix = tail.partition("\n```")
    assert marker
    return cast(list[dict[str, Any]], json.loads(serialized))


@pytest.mark.asyncio
async def test_controller_stages_run_start_compaction_from_detached_snapshot(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Old user " + "u" * 500,
                "timestamp": timestamp,
            },
            _assistant_history_message(
                "Old assistant " + "a" * 500,
                timestamp=timestamp,
                token_usage=_usage(),
            ),
            {
                "role": "user",
                "content": "Current persisted history " + "h" * 500,
                "timestamp": timestamp,
            },
            _assistant_history_message(
                "Current history assistant " + "c" * 500,
                timestamp=timestamp,
                token_usage=_usage(),
            ),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 2,
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
            },
            "summary": "Prior action",
        },
        last_compacted=0,
    )
    snapshot = AgentRunContextSnapshot.from_session(session)
    provider = ScriptedFakeProvider(
        completions=(_response("Facts"), _response("Updated action")),
    )
    manager = AgentRunContextController(
        snapshot=snapshot,
        provider=ScriptedFakeRouter(provider),
        append_summary=MemoryManager(state).append_summary,
        now=lambda: NOW,
    )

    result = await manager.prepare_run_start(
        project_messages=_project_messages,
        current_user={"role": "user", "content": "New user must stay out"},
        route_status=_chat_status(context_window=1200, max_output=100),
        memory_route_status=_memory_status(context_window=1200, max_output=100),
        compact_ratio=0.5,
        tools=(),
    )

    selected = _summary_payload(provider)
    assert selected
    assert all(message["content"] != "New user must stay out" for message in selected)
    assert "New user must stay out" not in str(provider.complete_requests[0].messages)
    assert sum(message.get("content") == "New user must stay out" for message in result) == 1
    terminal = manager.terminal_commit_values()
    assert terminal.pending_action_summary == "Updated action"
    assert terminal.pending_last_compacted > snapshot.last_compacted
    assert terminal.usage_delta == {
        "model_calls": 2,
        "input_tokens": 40,
        "output_tokens": 10,
        "total_tokens": 50,
    }
    terminal.usage_delta["model_calls"] = 99
    assert manager.terminal_commit_values().usage_delta["model_calls"] == 2


def test_snapshot_copies_transcript_and_metadata_without_retaining_session(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    initial_messages = _run_history(
        "before",
        size=20,
        timestamp=timestamp,
        token_usage=_usage(),
    )
    seed_session_state(
        session,
        messages=initial_messages,
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    snapshot = AgentRunContextSnapshot.from_session(session)
    seed_session_state(
        session,
        messages=[
            *initial_messages,
            {"role": "user", "content": "after", "timestamp": timestamp},
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "changed",
        },
        last_compacted=0,
    )

    assert len(snapshot.messages) == 2
    assert "after" not in str(snapshot.messages)
    assert snapshot.metadata.get("summary") == ""


@pytest.mark.asyncio
async def test_controller_detaches_from_the_supplied_snapshot(workspace: Path) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "original",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "original action",
        },
        last_compacted=0,
    )
    snapshot = AgentRunContextSnapshot.from_session(session)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = AgentRunContextController(
        snapshot=snapshot,
        provider=ScriptedFakeRouter(provider),
        append_summary=MemoryManager(state).append_summary,
        now=lambda: NOW,
    )

    snapshot.messages[0]["content"] = "mutated outside controller"
    assert isinstance(snapshot.metadata, dict)
    snapshot.metadata["summary"] = "mutated action"
    await _prepare_controller(
        controller,
        context_window=1_000,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert "original user" in str(_summary_payload(provider))
    assert "mutated outside controller" not in str(_summary_payload(provider))
    assert "original action" in str(provider.complete_requests[1].messages)
    assert "mutated action" not in str(provider.complete_requests[1].messages)


@pytest.mark.asyncio
async def test_prepare_run_start_returns_a_detached_message_tuple(workspace: Path) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_tool_run_history(
            "history",
            result_size=20,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "project_messages": _project_messages_with_tool_calls,
        "route_status": _chat_status(context_window=10_000, max_output=100),
        "memory_route_status": _memory_status(context_window=10_000, max_output=100),
    }

    first = await controller.prepare_run_start(**kwargs)
    terminal = controller.terminal_commit_values()
    assert isinstance(first, tuple)
    first_assistant = next(message for message in first if message["role"] == "assistant")
    first_assistant["tool_calls"][0]["arguments"] = '{"mutated": true}'

    second = await controller.prepare_run_start(**kwargs)
    second_assistant = next(message for message in second if message["role"] == "assistant")

    assert isinstance(second, tuple)
    assert second_assistant["tool_calls"][0]["arguments"] == "{}"
    assert controller.terminal_commit_values() == terminal
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_prepare_react_returns_a_detached_message_tuple(workspace: Path) -> None:
    state = _state(workspace)
    session = Session.create(state)
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)
    increment = _react_cycle("current", size=20)
    kwargs: dict[str, Any] = {
        "project_messages": _project_messages_with_tool_calls,
        "increment": increment,
        "latest_cycle_start": 0,
        "route_status": _chat_status(context_window=10_000, max_output=100),
        "memory_route_status": _memory_status(context_window=10_000, max_output=100),
        "current_user": {"role": "user", "content": "current request"},
    }

    first = await controller.prepare_react(**kwargs)
    terminal = controller.terminal_commit_values()
    assert isinstance(first, tuple)
    first_assistant = next(message for message in first if message["role"] == "assistant")
    first_assistant["tool_calls"][0]["arguments"] = '{"mutated": true}'

    second = await controller.prepare_react(**kwargs)
    second_assistant = next(message for message in second if message["role"] == "assistant")

    assert isinstance(second, tuple)
    assert second_assistant["tool_calls"][0]["arguments"] == "{}"
    assert increment[0]["tool_calls"][0]["arguments"] == "{}"
    assert controller.terminal_commit_values() == terminal
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_prepare_react_rejects_an_invalid_increment_role(workspace: Path) -> None:
    state = _state(workspace)
    controller = _controller(workspace, Session.create(state), ScriptedFakeProvider())

    with pytest.raises(ValueError, match="ReAct increment message 0 must be assistant or tool"):
        await controller.prepare_react(
            project_messages=_project_messages,
            increment=({"role": "user", "content": "invalid"},),
            latest_cycle_start=None,
            route_status=_chat_status(context_window=10_000, max_output=100),
            memory_route_status=_memory_status(context_window=10_000, max_output=100),
        )


@pytest.mark.asyncio
async def test_prepare_react_rejects_latest_cycle_start_at_a_tool_message(
    workspace: Path,
) -> None:
    state = _state(workspace)
    controller = _controller(workspace, Session.create(state), ScriptedFakeProvider())

    with pytest.raises(
        ValueError,
        match="latest_cycle_start must identify an assistant in the increment",
    ):
        await controller.prepare_react(
            project_messages=_project_messages,
            increment=_react_cycle("current", size=20),
            latest_cycle_start=1,
            route_status=_chat_status(context_window=10_000, max_output=100),
            memory_route_status=_memory_status(context_window=10_000, max_output=100),
        )


@pytest.mark.asyncio
async def test_multiple_runs_under_ten_percent_keep_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            *_run_history("old", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("middle", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("latest", size=320, timestamp=timestamp, token_usage=_usage()),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 3,
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)
    available = latest_tokens * 10

    result = await _prepare_controller(
        controller,
        context_window=available + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "old user" in fact_payload
    assert "middle user" in fact_payload
    assert "latest user" not in fact_payload
    assert "latest user" in str(result)
    assert controller.terminal_commit_values().pending_last_compacted > 0


@pytest.mark.asyncio
async def test_multiple_runs_at_ten_percent_keep_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            *_run_history("old", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("middle", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("latest", size=320, timestamp=timestamp, token_usage=_usage()),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 3,
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)

    await _prepare_controller(
        controller,
        context_window=latest_tokens * 10 + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "latest user" not in fact_payload


@pytest.mark.asyncio
async def test_multiple_runs_over_ten_percent_select_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            *_run_history("old", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("middle", size=1_400, timestamp=timestamp, token_usage=_usage()),
            *_run_history("latest", size=320, timestamp=timestamp, token_usage=_usage()),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 3,
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)

    await _prepare_controller(
        controller,
        context_window=latest_tokens * 10 - 1 + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "latest user" in fact_payload
    assert "new user" not in fact_payload


@pytest.mark.asyncio
async def test_single_completed_run_selects_its_entire_cursor_suffix(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "only",
            size=900,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        context_window=800,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert [message["role"] for message in _summary_payload(provider)] == [
        "user",
        "assistant",
    ]
    assert "only user" in str(provider.complete_requests[0].messages[1]["content"])
    assert controller.terminal_commit_values().pending_last_compacted == len(session.messages)


@pytest.mark.asyncio
async def test_react_current_run_at_exactly_fifty_percent_keeps_current_run_and_compacts_history(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "history",
            size=1_200,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current request"}
    increment = _react_cycle("current")
    available = estimate_run_slice_tokens([current_user, *increment]) * 2

    result = await controller.prepare_react(
        project_messages=_project_messages,
        increment=increment,
        latest_cycle_start=0,
        route_status=_chat_status(context_window=available + 100, max_output=100),
        current_user=current_user,
        compact_ratio=0.5,
        memory_route_status=_memory_status(context_window=10_000),
    )

    assert [message["content"] for message in _summary_payload(provider)] == [
        message["content"] for message in session.messages
    ]
    assert "current request" not in str(_summary_payload(provider))
    assert sum(message.get("content") == "current request" for message in result) == 1
    assert controller.terminal_commit_values().pending_last_compacted == len(session.messages)


@pytest.mark.asyncio
async def test_react_sole_current_run_may_compact_at_or_below_fifty_percent(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.update_metadata(summary="previous action " + "x" * 300)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current request"}
    increment = _react_cycle("current", size=300)
    available = estimate_run_slice_tokens([current_user, *increment]) * 2

    result = await controller.prepare_react(
        project_messages=_project_messages,
        increment=increment,
        latest_cycle_start=0,
        route_status=_chat_status(context_window=available + 100, max_output=100),
        current_user=current_user,
        compact_ratio=0.5,
        memory_route_status=_memory_status(context_window=10_000),
    )

    assert _summary_payload(provider)[0] == current_user
    assert result[-1]["content"].startswith("current result")
    assert sum(message.get("content") == "current request" for message in result) == 1
    assert controller.terminal_commit_values().pending_last_compacted == 1

    response = _response("final answer")
    response_message = response.message.to_dict()
    context = controller.record_main_agent_response(
        request_messages=result,
        tools=(),
        response=response,
        increment=[*increment, response_message],
        route_status=_chat_status(
            context_window=available + 100,
            max_output=100,
            provider_id="provider",
            model="model",
        ),
        estimator_version="utf8-bytes-div4-v1",
    )

    assert context.run_projected_tokens == estimate_run_slice_tokens([*increment, response_message])


@pytest.mark.asyncio
async def test_react_current_run_just_above_fifty_percent_may_select_early_current_content(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "history",
            size=1_200,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current request"}
    increment = _react_cycle("current")
    current_slice = estimate_run_slice_tokens([current_user, *increment])
    available = current_slice * 2 - 1

    result = await controller.prepare_react(
        project_messages=_project_messages,
        increment=increment,
        latest_cycle_start=0,
        route_status=_chat_status(context_window=available + 100, max_output=100),
        current_user=current_user,
        compact_ratio=0.5,
        memory_route_status=_memory_status(context_window=10_000),
    )

    assert "current request" in str(_summary_payload(provider))
    assert result[-1]["content"].startswith("current result")
    assert sum(message.get("content") == "current request" for message in result) == 1


@pytest.mark.asyncio
async def test_react_compaction_consumes_only_new_batch_after_current_user_is_compacted(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    provider = ScriptedFakeProvider(
        completions=(
            _response("facts one"),
            _response("action one"),
            _response("facts two"),
            _response("action two"),
        )
    )
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current request"}
    first_increment: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "early assistant " + "a" * 4_000,
            "tool_calls": [{"id": "early-call", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": _usage(),
        },
        {
            "role": "tool",
            "tool_call_id": "early-call",
            "name": "read_file",
            "status": "success",
            "content": "early result " + "r" * 4_000,
            "artifact": None,
        },
        *_react_cycle("latest", size=700),
    ]
    kwargs: dict[str, Any] = {
        "project_messages": _project_messages,
        "route_status": _chat_status(context_window=4_000, max_output=100),
        "current_user": current_user,
        "compact_ratio": 0.5,
        "memory_route_status": _memory_status(context_window=10_000),
    }

    first = await controller.prepare_react(
        increment=first_increment,
        latest_cycle_start=2,
        **kwargs,
    )
    second_increment: list[dict[str, Any]] = [*first_increment, *_react_cycle("new", size=4_000)]
    second = await controller.prepare_react(
        increment=second_increment,
        latest_cycle_start=4,
        **kwargs,
    )

    assert first != second
    assert len(provider.complete_requests) == 4
    assert controller.terminal_commit_values().pending_last_compacted == 5
    second_fact = str(provider.complete_requests[2].messages[1]["content"])
    second_action = str(provider.complete_requests[3].messages[1]["content"])
    assert "early assistant" not in second_fact
    assert "early result" not in second_fact
    assert "current request" not in second_fact
    assert "latest result" in second_fact
    assert "current request" not in second_action
    assert sum(message.get("content") == "current request" for message in second) == 1


@pytest.mark.asyncio
async def test_explicit_router_adapter_blocks_an_over_budget_attempt_before_provider() -> None:
    provider = ScriptedFakeProvider(completions=(_response("unexpected"),))
    router = ModelRouter(
        configuration=_router_configuration(
            chat_context_window=100,
            default_context_window=100,
        ),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    guarded = router.for_run(guard=agent_run_attempt_guard)

    with pytest.raises(ModelCallError) as raised:
        await guarded.complete(
            "chat",
            messages=[{"role": "user", "content": "x" * 1_000}],
            tools=(),
        )

    assert raised.value.error.code == "model_context_overflow"
    assert provider.complete_requests == []


@pytest.mark.parametrize("available_delta", (1, 0, -1))
def test_compactor_request_guard_matches_the_shared_context_predicate(
    available_delta: int,
) -> None:
    messages = [{"role": "user", "content": "request"}]
    tools = ({"type": "function", "function": {"name": "work", "parameters": {}}},)
    estimated = estimate_request_tokens(messages, tools)
    max_output = 10
    status = _memory_status(
        context_window=estimated + max_output + available_delta,
        max_output=max_output,
    )

    assert agent_run_attempt_guard(status, messages, tools) is request_fits_model_context(
        messages,
        tools,
        context_window=status.context_window,
        max_output=status.max_output,
    )


@pytest.mark.asyncio
async def test_explicit_router_adapter_preserves_retry_continuation_and_response() -> None:
    continuation = ModelContinuation(provider_id="chat-provider", payload=object())
    expected = _response("recovered", input_tokens=31, output_tokens=7)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo("provider_timeout", "retry", retryable=True)),
            expected,
        )
    )
    router = ModelRouter(
        configuration=_router_configuration(
            chat_context_window=4_000,
            default_context_window=4_000,
        ),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    guarded = router.for_run(guard=agent_run_attempt_guard)
    messages = [{"role": "user", "content": "request"}]
    tools = ({"type": "function", "function": {"name": "work"}},)

    observed = await guarded.complete(
        "chat",
        messages=messages,
        tools=tools,
        continuation=continuation,
    )

    assert observed is expected
    assert len(provider.complete_requests) == 2
    assert all(request.continuation is continuation for request in provider.complete_requests)
    assert all(request.messages == messages for request in provider.complete_requests)
    assert all(request.tools == tools for request in provider.complete_requests)
    assert router.current_call_status("chat") == router.route_status("chat")


@pytest.mark.asyncio
async def test_explicit_router_adapter_rechecks_smaller_fallback_before_provider() -> None:
    chat_provider = ScriptedFakeProvider(
        completions=(ModelCallError(ErrorInfo("provider_auth_error", "fallback")),)
    )
    default_provider = ScriptedFakeProvider(completions=(_response("unexpected"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=_router_configuration(
            chat_context_window=4_000,
            default_context_window=100,
        ),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )
    guarded = router.for_run(guard=agent_run_attempt_guard)

    with pytest.raises(ModelCallError) as raised:
        await guarded.complete(
            "chat",
            messages=[{"role": "user", "content": "x" * 1_000}],
            tools=(),
        )

    assert raised.value.error.code == "model_context_overflow"
    assert len(chat_provider.complete_requests) == 1
    assert default_provider.complete_requests == []
    status = router.current_call_status("chat")
    assert status is not None
    assert status.selected_route == "default"


@pytest.mark.asyncio
async def test_controller_preparer_rebuilds_runner_requests_and_preserves_opaque_continuation(
    workspace: Path,
) -> None:
    class Gateway:
        schemas: tuple[dict[str, Any], ...] = ()

        async def call(
            self,
            tool_call: ModelToolCall,
            *,
            confirmation: object = None,
        ) -> ToolResult:
            del confirmation
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                status="success",
                content="tool result",
            )

        def is_micro_compression_eligible(self, tool_name: str) -> bool:
            del tool_name
            return False

    continuation = ModelContinuation(provider_id="test-provider", payload=object())
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="First",
                                tool_calls=(
                                    ModelToolCall(id="call-1", name="work", arguments="{}"),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                            continuation=continuation,
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    state = _state(workspace)
    session = Session.create(state)
    controller = _controller(workspace, session, provider)
    router = _context_router(provider)
    preparer = AgentRunContextRequestPreparer(
        controller,
        router=router,
        requested_route="chat",
        project_messages=_project_messages,
        current_user={"role": "user", "content": "canonical task"},
    )

    result = await AgentRunner(router, preparer).run(
        [{"role": "system", "content": "stale"}, {"role": "user", "content": "stale"}],
        model="chat",
        tool_gateway=Gateway(),  # type: ignore[arg-type]
        on_output=None,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert provider.stream_requests[0].messages == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "canonical task"},
    ]
    assert all(
        message.get("content") != "stale" for message in provider.stream_requests[1].messages
    )
    assert (
        sum(
            message.get("content") == "canonical task"
            for message in provider.stream_requests[1].messages
        )
        == 1
    )
    assert provider.stream_requests[1].continuation is continuation


@pytest.mark.asyncio
async def test_react_action_failure_keeps_fact_and_retries_only_the_pending_action(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    failure = ModelCallError(ErrorInfo("model_failed", "action failed"))
    provider = ScriptedFakeProvider(
        completions=(_response("facts"), failure, _response("recovered action"))
    )
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current request"}
    increment = _react_cycle("current", size=4_000)
    kwargs: dict[str, Any] = {
        "project_messages": _project_messages,
        "increment": increment,
        "latest_cycle_start": 0,
        "route_status": _chat_status(context_window=4_000, max_output=100),
        "current_user": current_user,
        "compact_ratio": 0.5,
        "memory_route_status": _memory_status(context_window=10_000),
    }

    with pytest.raises(ModelCallError, match="action failed"):
        await controller.prepare_react(**kwargs)
    failed_values = controller.terminal_commit_values()
    assert failed_values.pending_last_compacted == 0
    assert failed_values.pending_action_summary is None
    assert failed_values.usage_delta["model_calls"] == 1
    assert len(provider.complete_requests) == 2

    with pytest.raises(ModelCallError, match="action failed"):
        await controller.prepare_react(**kwargs)
    assert len(provider.complete_requests) == 2

    recovered = await controller.prepare_react(**kwargs, continuation_revision=1)
    recovered_values = controller.terminal_commit_values()
    assert recovered_values.pending_last_compacted == 1
    assert recovered_values.pending_action_summary == "recovered action"
    assert sum(message.get("content") == "current request" for message in recovered) == 1
    assert (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8").count(
        '"content":"facts"'
    ) == 1
    assert len(provider.complete_requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "cursor", "expected_roles"),
    (
        ("before-user", 0, ("user", "assistant")),
        ("after-user", 1, ("assistant",)),
        ("inside-tools", 2, ("tool",)),
        ("run-boundary", 2, ("user", "assistant")),
    ),
    ids=("before-user", "after-user", "inside-tools", "run-boundary"),
)
async def test_cursor_intersects_recovered_run_boundary_without_selecting_fragments(
    workspace: Path,
    case: str,
    cursor: int,
    expected_roles: tuple[str, ...],
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    if case == "inside-tools":
        history_messages = [
            {"role": "user", "content": "tool run user", "timestamp": timestamp},
            *_tool_run_history(
                "tool",
                timestamp=timestamp,
                token_usage=_usage(),
            ),
        ]
        history_usage = _usage()
    else:
        history_messages = [
            *_run_history("first", size=1_000, timestamp=timestamp, token_usage=_usage()),
            *_run_history("second", size=10, timestamp=timestamp, token_usage=_usage()),
        ]
        history_usage = {
            "model_calls": 2,
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": 12,
        }
    seed_session_state(
        session,
        messages=history_messages,
        metadata={
            "title": "Untitled session",
            "token_usage": history_usage,
            "summary": "",
        },
        last_compacted=cursor,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        current_user="cursor current " + "c" * 2_000,
        context_window=1_200,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert tuple(message["role"] for message in _summary_payload(provider)) == expected_roles
    assert controller.terminal_commit_values().pending_last_compacted == cursor + len(
        expected_roles
    )


@pytest.mark.asyncio
async def test_fact_summary_contains_complete_tool_results_and_current_user_is_excluded(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "old user " + "u" * 800,
                "timestamp": timestamp,
            },
            *_tool_run_history(
                "complete",
                result_size=900,
                timestamp=timestamp,
                token_usage=_usage(),
            ),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        current_user="must not be summarized " + "n" * 1_800,
        context_window=1_200,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    action_payload = str(provider.complete_requests[1].messages[1]["content"])
    assert "complete tool result" in fact_payload
    assert "must not be summarized" not in fact_payload
    assert "must not be summarized" not in action_payload


@pytest.mark.asyncio
async def test_consecutive_staging_consumes_only_new_batch_and_replaces_action_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            *_run_history("first", size=1_000, timestamp=timestamp, token_usage=_usage()),
            *_run_history("second", size=1_000, timestamp=timestamp, token_usage=_usage()),
            *_run_history("latest", size=50, timestamp=timestamp, token_usage=_usage()),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 3,
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(
        completions=(
            _response("facts one"),
            _response("action one"),
            _response("facts two"),
            _response("action two"),
        )
    )
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_800,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    first = await _prepare_controller(
        controller,
        current_user="new user " + "n" * 3_000,
        **kwargs,
    )
    second = await _prepare_controller(
        controller,
        current_user="new user changed " + "n" * 3_000,
        **kwargs,
    )

    assert first != second
    assert "first user" in str(provider.complete_requests[0].messages[1]["content"])
    assert "first user" not in str(provider.complete_requests[2].messages[1]["content"])
    assert "latest user" in str(provider.complete_requests[2].messages[1]["content"])
    assert "action one" in str(provider.complete_requests[3].messages[1]["content"])
    assert controller.terminal_commit_values().pending_action_summary == "action two"
    assert len(provider.complete_requests) == 4


@pytest.mark.asyncio
async def test_action_none_stages_removal_of_previous_action_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "previous action",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("None")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        context_window=1_000,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert controller.terminal_commit_values().pending_action_summary is None
    assert "previous action" in str(provider.complete_requests[1].messages[1]["content"])


@pytest.mark.asyncio
async def test_fact_model_failure_does_not_stage_cursor_action_or_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "previous action",
        },
        last_compacted=0,
    )
    failure = ModelCallError(ErrorInfo(code="model_failed", message="fact failed"))
    provider = ScriptedFakeProvider(completions=(failure,))
    controller = _controller(workspace, session, provider)

    with pytest.raises(CommittableAgentRunError) as raised:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )
    with pytest.raises(CommittableAgentRunError) as repeated:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 0
    assert terminal.pending_action_summary == "previous action"
    assert terminal.usage_delta["model_calls"] == 0
    assert not (state.memory_directory / "summary.jsonl").exists()
    assert len(provider.complete_requests) == 1
    assert raised.value.__cause__ is failure
    assert repeated.value is raised.value


@pytest.mark.asyncio
async def test_fact_persistence_failure_keeps_batch_uncommitted_but_stages_usage(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "previous action",
        },
        last_compacted=0,
    )
    (state.memory_directory / "summary.jsonl").mkdir()
    provider = ScriptedFakeProvider(
        completions=(_response("facts", input_tokens=7, output_tokens=3),)
    )
    controller = _controller(workspace, session, provider)

    with pytest.raises(CommittableAgentRunError, match="could not be persisted") as raised:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 0
    assert terminal.pending_action_summary == "previous action"
    assert terminal.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert len(provider.complete_requests) == 1
    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_action_failure_keeps_fact_and_earlier_staged_state_without_advancing_cursor(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "previous action",
        },
        last_compacted=0,
    )
    failure = ModelCallError(ErrorInfo(code="model_failed", message="action failed"))
    provider = ScriptedFakeProvider(
        completions=(_response("facts"), failure, _response("recovered action"))
    )
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    with pytest.raises(CommittableAgentRunError) as raised:
        await _prepare_controller(controller, **kwargs)

    failed_values = controller.terminal_commit_values()
    assert failed_values.pending_last_compacted == 0
    assert failed_values.pending_action_summary == "previous action"
    assert failed_values.usage_delta == {
        "model_calls": 1,
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
    }
    assert "facts" in (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    assert len(provider.complete_requests) == 2
    assert raised.value.__cause__ is failure

    with pytest.raises(CommittableAgentRunError) as repeated:
        await _prepare_controller(controller, **kwargs)
    assert len(provider.complete_requests) == 2
    assert repeated.value is raised.value

    recovered = await _prepare_controller(controller, current_user="changed user", **kwargs)

    summary_content = (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    recovered_values = controller.terminal_commit_values()
    assert summary_content.count('"content":"facts"') == 1
    assert recovered_values.pending_last_compacted == len(session.messages)
    assert recovered_values.pending_action_summary == "recovered action"
    assert recovered_values.usage_delta["model_calls"] == 2
    assert "changed user" in str(recovered)
    assert len(provider.complete_requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "error_code"),
    (("length", "model_failed"), ("cancelled", "turn_cancelled")),
)
async def test_action_finish_failure_keeps_orphan_fact_and_usage_without_advancing(
    workspace: Path,
    finish_reason: Literal["length", "cancelled"],
    error_code: str,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "previous action",
        },
        last_compacted=0,
    )
    action_response = ModelResponse(
        message=AssistantModelMessage(content="incomplete action"),
        usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
        finish_reason=finish_reason,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), action_response))
    controller = _controller(workspace, session, provider)

    with pytest.raises(CommittableAgentRunError) as raised:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    assert raised.value.error.code == error_code
    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 0
    assert terminal.pending_action_summary == "previous action"
    assert terminal.usage_delta == {
        "model_calls": 2,
        "input_tokens": 28,
        "output_tokens": 8,
        "total_tokens": 36,
    }
    assert "facts" in (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_summary_hard_overflow_is_rejected_before_provider_call(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=20, max_output=10),
    }

    with pytest.raises(CommittableAgentRunError) as raised:
        await _prepare_controller(controller, **kwargs)
    with pytest.raises(CommittableAgentRunError) as repeated:
        await _prepare_controller(controller, **kwargs)

    assert raised.value.error.code == "model_context_overflow"
    assert raised.value.error.message == MODEL_CONTEXT_OVERFLOW_MESSAGE
    assert repeated.value is raised.value
    assert isinstance(raised.value.__cause__, ModelCallError)
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_action_replacement_is_checked_against_the_final_hard_limit(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    oversized_action = "action " + "x" * 4_000
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response(oversized_action)))
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=10_000),
    }

    with pytest.raises(ModelCallError) as raised:
        await _prepare_controller(controller, **kwargs)

    assert raised.value.error.code == "model_context_overflow"
    assert not isinstance(raised.value, CommittableAgentRunError)
    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == len(session.messages)
    assert terminal.pending_action_summary == oversized_action
    assert terminal.usage_delta["model_calls"] == 2
    assert len(provider.complete_requests) == 2

    with pytest.raises(ModelCallError):
        await _prepare_controller(controller, **kwargs)
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_compatible_main_agent_usage_changes_the_run_start_compaction_decision(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            {"role": "user", "content": "old user", "timestamp": timestamp},
            {
                **_assistant_history_message(
                    "old answer",
                    timestamp=timestamp,
                    token_usage=_usage(100, 10),
                ),
                "context_usage": {
                    "requested_route": "chat",
                    "selected_route": "chat",
                    "provider_id": "provider",
                    "model": "model",
                    "context_window": 360,
                    "max_output": 200,
                    "anchor_estimated_tokens": 20,
                    "estimator_version": "utf8-bytes-div4-v1",
                    "run_projected_tokens": 80,
                    "run_projection_source": "estimated",
                },
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(100, 10),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    result = await _prepare_controller(
        controller,
        context_window=360,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
        provider_id="provider",
        model="model",
    )

    assert len(provider.complete_requests) == 2
    assert "old user" in str(_summary_payload(provider))
    assert sum(message.get("content") == "new user" for message in result) == 1
    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == len(session.messages)
    assert terminal.usage_delta["model_calls"] == 2


@pytest.mark.asyncio
async def test_latest_assistant_without_provenance_forces_run_start_local_estimate(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            {"role": "user", "content": "older user", "timestamp": timestamp},
            {
                **_assistant_history_message(
                    "older answer",
                    timestamp=timestamp,
                    token_usage=_usage(100, 10),
                ),
                "context_usage": _context_usage(context_window=360),
            },
            {"role": "user", "content": "latest user", "timestamp": timestamp},
            _assistant_history_message(
                "latest answer without provenance",
                timestamp=timestamp,
                token_usage=_usage(20, 5),
            ),
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
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)

    retained = await _prepare_controller(
        controller,
        context_window=360,
        max_output=200,
        provider_id="provider",
        model="model",
    )

    assert provider.complete_requests == []
    assert "older user" in str(retained)
    assert "latest answer without provenance" in str(retained)
    assert "new user" in str(retained)
    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 0
    assert terminal.usage_delta["model_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_usage", "second_usage"),
    (
        pytest.param((20, 5), (24, 6), id="positive"),
        pytest.param((0, 5), (4, 6), id="zero-input"),
        pytest.param((5, 0), (9, 1), id="zero-output"),
        pytest.param((0, 0), (4, 1), id="zero-total"),
    ),
)
async def test_main_response_provenance_anchors_completed_response_and_then_uses_reported_delta(
    workspace: Path,
    first_usage: tuple[int, int],
    second_usage: tuple[int, int],
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    controller = _controller(workspace, session, ScriptedFakeProvider())
    current_user = {"role": "user", "content": "current request"}
    tools = ({"type": "function", "function": {"name": "read_file"}},)
    preparation = await controller.prepare_run_start(
        project_messages=_project_messages,
        current_user=current_user,
        route_status=_chat_status(
            context_window=1_600,
            max_output=200,
            provider_id="provider",
            model="model",
        ),
        memory_route_status=_memory_status(context_window=1_600, max_output=200),
        tools=tools,
    )
    first = _response(
        "first answer",
        input_tokens=first_usage[0],
        output_tokens=first_usage[1],
    )
    first_message = first.message.to_dict()

    first_context = controller.record_main_agent_response(
        request_messages=preparation,
        tools=tools,
        response=first,
        increment=[first_message],
        route_status=_chat_status(
            context_window=1_600,
            max_output=200,
            provider_id="provider",
            model="model",
        ),
        estimator_version="utf8-bytes-div4-v1",
    )

    assert first_context.run_projection_source == "estimated"
    assert first_context.anchor_estimated_tokens == estimate_request_tokens(
        [*preparation, first_message], tools
    )

    tool_message = {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read_file",
        "content": "tool result",
    }
    second_request = [*preparation, first_message, tool_message]
    second = _response(
        "second answer",
        input_tokens=second_usage[0],
        output_tokens=second_usage[1],
    )
    second_message = second.message.to_dict()
    second_context = controller.record_main_agent_response(
        request_messages=second_request,
        tools=tools,
        response=second,
        increment=[first_message, tool_message, second_message],
        route_status=_chat_status(
            context_window=1_600,
            max_output=200,
            provider_id="provider",
            model="model",
        ),
        estimator_version="utf8-bytes-div4-v1",
    )

    assert second_context.run_projection_source == "reported_delta"
    assert second_context.run_projected_tokens == first_context.run_projected_tokens + 5
    assert second_context.anchor_estimated_tokens == estimate_request_tokens(
        [*second_request, second_message], tools
    )


@pytest.mark.asyncio
async def test_incompatible_latest_main_usage_does_not_fall_back_to_an_older_anchor(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    messages: list[dict[str, Any]] = []
    for label, provider_id in (("older", "provider"), ("latest", "other-provider")):
        messages.extend(
            [
                {"role": "user", "content": f"{label} user", "timestamp": timestamp},
                {
                    **_assistant_history_message(
                        f"{label} answer",
                        timestamp=timestamp,
                        token_usage=_usage(100, 10),
                    ),
                    "context_usage": {
                        "requested_route": "chat",
                        "selected_route": "chat",
                        "provider_id": provider_id,
                        "model": "model",
                        "context_window": 360,
                        "max_output": 200,
                        "anchor_estimated_tokens": 20,
                        "estimator_version": "utf8-bytes-div4-v1",
                        "run_projected_tokens": 80,
                        "run_projection_source": "estimated",
                    },
                },
            ]
        )
    seed_session_state(
        session,
        messages=messages,
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 2,
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)

    result = await _prepare_controller(
        controller,
        context_window=360,
        max_output=200,
        provider_id="provider",
        model="model",
    )

    assert provider.complete_requests == []
    assert "older user" in str(result)
    assert "latest user" in str(result)
    assert "new user" in str(result)
    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 0
    assert terminal.usage_delta["model_calls"] == 0


@pytest.mark.asyncio
async def test_same_context_revision_is_a_noop_after_successful_staging(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    first = await _prepare_controller(controller, **kwargs)
    first_values = controller.terminal_commit_values()
    second = await _prepare_controller(controller, **kwargs)

    assert second == first
    assert controller.terminal_commit_values() == first_values
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_request_preparer_reuses_run_start_revision_without_duplicate_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    route_status = _chat_status(context_window=1_000, max_output=200)
    memory_route_status = _memory_status(context_window=4_000)
    router = _context_router(
        provider,
        chat_status=route_status,
        memory_status=memory_route_status,
    )
    preparer = AgentRunContextRequestPreparer(
        controller,
        router=router,
        requested_route="chat",
        project_messages=_project_messages,
        current_user={"role": "user", "content": "current request"},
        compact_ratio=0.5,
    )

    await controller.prepare_run_start(
        project_messages=_project_messages,
        current_user={"role": "user", "content": "current request"},
        route_status=route_status,
        compact_ratio=0.5,
        memory_route_status=memory_route_status,
    )
    prepared = await preparer.prepare(
        increment=(),
        latest_cycle_start=None,
        tools=(),
        continuation=None,
        continuation_revision=0,
        is_micro_compression_eligible=None,
    )

    assert len(provider.complete_requests) == 2
    assert prepared[0] == {"role": "system", "content": "SYSTEM"}
    assert all(message["content"] != "old user " + "u" * 800 for message in prepared)
    assert sum(message.get("content") == "current request" for message in prepared) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
async def test_runner_final_projection_changes_revision_and_repeats_stably(
    workspace: Path,
    enabled: bool,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)
    preparer = AgentRunContextRequestPreparer(
        controller,
        router=_context_router(
            provider,
            chat_status=_chat_status(context_window=16_384, max_output=100),
            memory_status=_memory_status(context_window=16_384, max_output=100),
        ),
        requested_route="chat",
        project_messages=_project_messages_with_tool_calls,
        current_user={"role": "user", "content": "current request"},
        enable_tool_micro_compression=enabled,
    )

    increment = tuple(
        message for number in range(11) for message in _react_cycle(str(number), size=513)
    )
    original_increment = deepcopy(increment)
    requests = [
        await preparer.prepare(
            increment=increment,
            latest_cycle_start=len(increment) - 2,
            tools=(),
            continuation=None,
            continuation_revision=0,
            is_micro_compression_eligible=lambda name: name == "read_file",
        )
        for _ in range(3)
    ]

    assert requests[0] == requests[1] == requests[2]
    assert (
        sum(
            message.get("content") == "[read_file result omitted from context]"
            for message in requests[0]
        )
        == (10 if enabled else 0)
    )
    assert requests[0][-1]["content"] == increment[-1]["content"]
    assert increment == original_increment
    assert provider.complete_requests == []
    assert sum(message.get("content") == "current request" for message in requests[0]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ("message", "tools", "route", "capacity", "estimator", "continuation", "micro"),
)
async def test_react_revision_changes_for_each_model_visible_input_source(
    workspace: Path,
    change: str,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=_run_history(
            "old",
            size=800,
            timestamp=NOW.isoformat(timespec="milliseconds"),
            token_usage=_usage(),
        ),
        metadata={
            "title": "Untitled session",
            "token_usage": _usage(),
            "summary": "",
        },
        last_compacted=0,
    )
    failure = ModelCallError(ErrorInfo("model_failed", "summary failed"))
    provider = ScriptedFakeProvider(completions=(failure, failure))
    controller = _controller(workspace, session, provider)
    base_status = ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id="provider",
        model="model",
        context_window=1_000,
        max_output=200,
        used_default=False,
    )
    base: dict[str, Any] = {
        "project_messages": _project_messages,
        "increment": (),
        "latest_cycle_start": None,
        "current_user": {"role": "user", "content": "current request"},
        "tools": (),
        "compact_ratio": 0.5,
        "route_status": base_status,
        "memory_route_status": _memory_status(context_window=4_000),
        "continuation_revision": 0,
    }
    with pytest.raises(CommittableAgentRunError, match="summary failed") as raised:
        await controller.prepare_react(**base)
    with pytest.raises(CommittableAgentRunError, match="summary failed") as repeated:
        await controller.prepare_react(**base)
    assert repeated.value is raised.value
    assert raised.value.__cause__ is failure
    assert len(provider.complete_requests) == 1

    changed = dict(base)
    if change == "message":
        changed["current_user"] = {"role": "user", "content": "changed request"}
    elif change == "tools":
        changed["tools"] = (
            {"type": "function", "function": {"name": "new_tool", "parameters": {}}},
        )
    elif change == "route":
        changed["route_status"] = ModelRouteStatus(
            requested_route="chat",
            selected_route="default",
            provider_id="other-provider",
            model="other-model",
            context_window=1_000,
            max_output=200,
            used_default=True,
        )
    elif change == "capacity":
        changed["route_status"] = ModelRouteStatus(
            requested_route="chat",
            selected_route="chat",
            provider_id="provider",
            model="model",
            context_window=900,
            max_output=200,
            used_default=False,
        )
    elif change == "estimator":
        changed["estimator_version"] = "another-estimator"
    elif change == "continuation":
        changed["continuation_revision"] = 1
    else:
        changed["micro_compression_enabled"] = True

    with pytest.raises(ModelCallError, match="summary failed"):
        await controller.prepare_react(**changed)

    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_request_preparer_uses_configured_capacity_after_previous_fallback(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    configuration = _router_configuration(
        chat_context_window=4_000,
        default_context_window=500,
        max_output=10,
    )
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("done")),)),),
        completions=(ModelCallError(ErrorInfo("route_unavailable", "chat unavailable")),),
    )
    default_provider = ScriptedFakeProvider(completions=(_response("fallback"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda provider: providers[provider.provider_id],
    ).for_run(guard=agent_run_attempt_guard)
    controller = AgentRunContextController(
        snapshot=AgentRunContextSnapshot.from_session(session),
        provider=router,
        append_summary=MemoryManager(state).append_summary,
        now=lambda: NOW,
    )
    await router.complete(
        "chat",
        messages=[{"role": "user", "content": "warmup"}],
        tools=(),
    )
    fallback_status = router.current_call_status("chat")
    assert fallback_status is not None
    assert fallback_status.selected_route == "default"

    preparer = AgentRunContextRequestPreparer(
        controller,
        router=router,
        requested_route="chat",
        project_messages=_project_messages,
        current_user={"role": "user", "content": "request " + "x" * 2_500},
    )

    result = await AgentRunner(router, preparer).run(
        [],
        model="chat",
        tool_gateway=None,
        on_output=None,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []


@pytest.mark.asyncio
async def test_action_summary_is_part_of_the_model_visible_revision(workspace: Path) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.update_metadata(summary="first action")
    first = _controller(workspace, session, ScriptedFakeProvider())
    first_result = await first.prepare_react(
        project_messages=_project_messages,
        increment=(),
        latest_cycle_start=None,
        route_status=_chat_status(context_window=4_000, max_output=100),
        memory_route_status=_memory_status(context_window=4_000, max_output=100),
        current_user={"role": "user", "content": "request"},
    )

    session.update_metadata(summary="second action")
    second = _controller(workspace, session, ScriptedFakeProvider())
    second_result = await second.prepare_react(
        project_messages=_project_messages,
        increment=(),
        latest_cycle_start=None,
        route_status=_chat_status(context_window=4_000, max_output=100),
        memory_route_status=_memory_status(context_window=4_000, max_output=100),
        current_user={"role": "user", "content": "request"},
    )

    assert first_result != second_result
    assert "first action" in str(first_result)
    assert "second action" in str(second_result)


@pytest.mark.asyncio
async def test_react_preparer_compacts_early_sole_run_and_preserves_latest_cycle(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    current_user = {"role": "user", "content": "current instruction"}
    increment: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "early assistant " + "a" * 4_000,
            "tool_calls": [{"id": "early-call", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": _usage(),
        },
        {
            "role": "tool",
            "tool_call_id": "early-call",
            "name": "read_file",
            "status": "success",
            "content": "complete early result " + "r" * 4_000,
            "artifact": None,
        },
        {
            "role": "assistant",
            "content": "latest assistant",
            "tool_calls": [{"id": "latest-call", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": _usage(),
        },
        {
            "role": "tool",
            "tool_call_id": "latest-call",
            "name": "read_file",
            "status": "success",
            "content": "complete latest result " + "l" * 700,
            "artifact": None,
        },
    ]

    result = await controller.prepare_react(
        project_messages=_project_messages,
        increment=increment,
        latest_cycle_start=2,
        route_status=_chat_status(context_window=4_000, max_output=100),
        current_user=current_user,
        compact_ratio=0.5,
        memory_route_status=_memory_status(context_window=10_000),
    )

    terminal = controller.terminal_commit_values()
    assert terminal.pending_last_compacted == 3
    assert [message["role"] for message in _summary_payload(provider)] == [
        "user",
        "assistant",
        "tool",
    ]
    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    action_payload = str(provider.complete_requests[1].messages[1]["content"])
    assert fact_payload.count("current instruction") == 1
    assert fact_payload.count("complete early result") == 1
    assert action_payload.count("complete early result") == 1
    assert "complete latest result" not in fact_payload
    assert "complete latest result" not in action_payload
    assert sum(message.get("content") == "current instruction" for message in result) == 1
    assert any(
        message.get("content", "").startswith("complete latest result")
        for message in result
        if message.get("role") == "tool"
    )
    repeated = await controller.prepare_react(
        project_messages=_project_messages,
        increment=increment,
        latest_cycle_start=2,
        route_status=_chat_status(context_window=4_000, max_output=100),
        current_user=current_user,
        compact_ratio=0.5,
        memory_route_status=_memory_status(context_window=10_000),
    )
    assert repeated == result
    assert controller.terminal_commit_values() == terminal
    assert len(provider.complete_requests) == 2
    assert sum(message.get("content") == "current instruction" for message in repeated) == 1


@pytest.mark.asyncio
async def test_visible_tool_schema_change_rechecks_a_new_revision(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    timestamp = NOW.isoformat(timespec="milliseconds")
    seed_session_state(
        session,
        messages=[
            *_run_history("first", size=800, timestamp=timestamp, token_usage=_usage()),
            *_run_history("second", size=800, timestamp=timestamp, token_usage=_usage()),
            *_run_history("latest", size=50, timestamp=timestamp, token_usage=_usage()),
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 3,
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    provider = ScriptedFakeProvider(
        completions=(
            _response("facts one"),
            _response("action one"),
            _response("facts two"),
            _response("action two"),
        )
    )
    controller = _controller(workspace, session, provider)
    base_kwargs: dict[str, Any] = {
        "context_window": 1_800,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }
    first = await _prepare_controller(controller, **base_kwargs)
    second = await _prepare_controller(
        controller,
        **base_kwargs,
        current_user="changed user " + "c" * 1_200,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "new_tool",
                    "description": "x" * 2_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    assert first != second
    assert len(provider.complete_requests) == 4
    assert "first user" not in str(_summary_payload(provider, 2))
    assert "latest user" in str(_summary_payload(provider, 2))
    terminal = controller.terminal_commit_values()
    assert terminal.pending_action_summary == "action two"
    assert terminal.usage_delta["model_calls"] == 4


def _estimate_latest_run(session: Session) -> int:
    user_indices = [
        index for index, message in enumerate(session.messages) if message["role"] == "user"
    ]
    return estimate_run_slice_tokens(session.messages[user_indices[-1] :])
