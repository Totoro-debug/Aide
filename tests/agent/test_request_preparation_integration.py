from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pytest

from aide.agent.session.execution_state import TitleWork
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.provider.session_configuration import SessionModelConfiguration
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import collect_foreground_outbound
from tests.fixtures.session import seed_session_state
from tests.scheduling.test_schedule_agent_loop import (
    JOB_UUID,
    NOW,
    _agent_loop,
    _BlockingClock,
    _close_components,
    _due_job,
    _is_schedule_call,
    _response,
    _ScheduleProvider,
)


def _configuration() -> str:
    return (
        VALID_CONFIG.replace(
            "context_window = 200000\nmax_output = 8192",
            "context_window = 4096\nmax_output = 512",
            1,
        )
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


def _old_run() -> list[dict[str, Any]]:
    timestamp = NOW.isoformat(timespec="milliseconds")
    usage = {"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "old request " + "u" * 5_000, "timestamp": timestamp}
    ]
    for number in range(11):
        call_id = f"old-{number}"
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [{"id": call_id, "name": "read_file", "arguments": "{}"}],
                    "status": "completed",
                    "error": None,
                    "token_usage": usage,
                    "timestamp": timestamp,
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "read_file",
                    "status": "success",
                    "content": "r" * 513,
                    "artifact": None,
                    "timestamp": timestamp,
                },
            )
        )
    messages.append(
        {
            "role": "assistant",
            "content": "old answer",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": usage,
            "timestamp": timestamp,
        }
    )
    return messages


def _selectable_configuration() -> str:
    return _configuration().replace(
        'models = ["claude-model"]',
        'models = ["claude-model", "selected-small", "large-model"]\n'
        "[models.providers.anthropic-default.model_context_windows]\n"
        "claude-model = 200000\nselected-small = 4096\nlarge-model = 200000",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model,compacted", [("selected-small", True), ("large-model", False)])
async def test_selected_model_capacity_controls_real_run_compaction(
    agent_home: Path,
    workspace: Path,
    model: str,
    compacted: bool,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(_response("done"),),
        memory_responses=(_response("fact summary"), _response("action summary")),
    )
    loop, router, schedule, dream, _dispatcher, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=_selectable_configuration(),
    )
    history = _old_run()
    seed_session_state(
        loop.session, messages=history, metadata=loop.session.metadata, last_compacted=0
    )
    loop.session.configure_model_durably(
        SessionModelConfiguration("anthropic-default", model, "high"),
        expected_version=0,
    )
    try:
        await loop.start()
        await collect_foreground_outbound(bus, "new request")
        assert provider.stream_requests[0].model == model
        assert provider.stream_requests[0].reasoning_effort == "high"
        assert len(provider.complete_requests) == (2 if compacted else 0)
        assert loop.session.last_compacted == (len(history) if compacted else 0)
    finally:
        await _close_components(loop, router, schedule, dream)


@pytest.mark.asyncio
async def test_model_is_captured_before_title_and_kept_through_tools_until_next_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "example.txt").write_text("tool contents", encoding="utf-8")
    provider = _ScheduleProvider(
        chat_responses=(
            _response(
                "read", tool_call=ModelToolCall("read", "read_file", '{"path":"example.txt"}')
            ),
            _response("first done"),
            _response("second done"),
        ),
        block_chat_call=1,
    )
    loop, router, schedule, dream, _dispatcher, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=_selectable_configuration(),
    )
    first = SessionModelConfiguration("anthropic-default", "selected-small", "high")
    second = SessionModelConfiguration("anthropic-default", "large-model", "max")
    loop.session.configure_model_durably(first, expected_version=0)
    original_title = loop._session_run_state.start_title

    def change_selection_at_title(
        session: Session,
        content: str,
        resolve_title: Callable[[str], Awaitable[tuple[str, dict[str, int] | None]]],
        is_aborted: Callable[[], bool],
    ) -> TitleWork | None:
        work = original_title(session, content, resolve_title, is_aborted)
        session.configure_model_durably(second, expected_version=1)
        return work

    object.__setattr__(loop._session_run_state, "start_title", change_selection_at_title)
    task = asyncio.create_task(collect_foreground_outbound(bus, "first request"))
    try:
        await loop.start()
        await asyncio.wait_for(provider.chat_block_started.wait(), timeout=10)
        assert loop.execution.active_model_configuration == first
        assert loop.execution.runtime_status_input().active_model_configuration == first
        assert loop.session.model_configuration == second
        provider.release_chat.set()
        await asyncio.wait_for(task, timeout=10)
        await collect_foreground_outbound(bus, "second request")
        assert [(call.model, call.reasoning_effort) for call in provider.stream_requests] == [
            ("selected-small", "high"),
            ("selected-small", "high"),
            ("large-model", "max"),
        ]
        restored = Session.load(loop.session.workspace_state, loop.session.session_id)
        assert restored.model_configuration == second
    finally:
        provider.release_chat.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _close_components(loop, router, schedule, dream)


@pytest.mark.asyncio
async def test_concurrent_sessions_share_provider_with_independent_model_combinations(
    agent_home: Path,
    workspace: Path,
) -> None:
    second_workspace = workspace / "second"
    second_workspace.mkdir()
    provider = _ScheduleProvider(
        chat_responses=(_response("done"), _response("done")),
        block_chat_call=1,
    )
    first_loop, router, schedule, dream, _dispatcher, first_bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=_selectable_configuration(),
    )
    second_loop, second_router, second_schedule, second_dream, _other, second_bus = _agent_loop(
        agent_home,
        second_workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=_selectable_configuration(),
    )
    second_loop._model_router = first_loop._model_router
    first_loop.session.configure_model_durably(
        SessionModelConfiguration("anthropic-default", "selected-small", "high"),
        expected_version=0,
    )
    second_loop.session.configure_model_durably(
        SessionModelConfiguration("anthropic-default", "large-model", "low"),
        expected_version=0,
    )
    first_task: asyncio.Task[object] | None = None
    try:
        await first_loop.start()
        await second_loop.start()
        first_task = asyncio.create_task(collect_foreground_outbound(first_bus, "first session"))
        await asyncio.wait_for(provider.chat_block_started.wait(), timeout=10)
        await asyncio.wait_for(
            collect_foreground_outbound(second_bus, "second session"), timeout=10
        )
        assert not first_task.done()
        assert [(call.model, call.reasoning_effort) for call in provider.stream_requests] == [
            ("selected-small", "high"),
            ("large-model", "low"),
        ]
        provider.release_chat.set()
        await asyncio.wait_for(first_task, timeout=10)
    finally:
        provider.release_chat.set()
        if first_task is not None:
            if not first_task.done():
                first_task.cancel()
            await asyncio.gather(first_task, return_exceptions=True)
        await _close_components(first_loop, router, schedule, dream)
        await _close_components(second_loop, second_router, second_schedule, second_dream)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ("foreground", "schedule"))
async def test_run_entry_summarizes_full_history_before_micro_compression(
    agent_home: Path,
    workspace: Path,
    lane: Literal["foreground", "schedule"],
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(_response("done"),),
        schedule_responses=(_response("done"),),
        memory_responses=(_response("fact summary"), _response("action summary")),
    )
    loop, router, schedule, dream, _dispatcher, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=_configuration(),
    )
    history = _old_run()
    if lane == "foreground":
        session = loop.session
    else:
        session = Session.create(
            loop.session.workspace_state,
            now=lambda: NOW,
            partition=SessionStoragePartition.SCHEDULE,
            job_id=JOB_UUID,
        )
    seed_session_state(
        session,
        messages=history,
        metadata={
            "title": "Existing session",
            "summary": "",
            "token_usage": {
                "model_calls": 12,
                "input_tokens": 12,
                "output_tokens": 12,
                "total_tokens": 24,
            },
        },
        last_compacted=0,
    )
    if lane == "schedule":
        session.close()
    original = deepcopy(history)

    try:
        await loop.start()
        if lane == "foreground":
            await collect_foreground_outbound(bus, "new request")
            main_requests = provider.stream_requests
            session_id = loop.session.session_id
            partition = SessionStoragePartition.FOREGROUND
        else:
            await loop.run_schedule_job(_due_job(message="new request"))
            main_requests = [
                request for request in provider.complete_requests if _is_schedule_call(request)
            ]
            session_id = f"schedule_{JOB_UUID}"
            partition = SessionStoragePartition.SCHEDULE

        summary_requests = [
            request for request in provider.complete_requests if not _is_schedule_call(request)
        ]
        assert len(main_requests) == 1
        assert len(summary_requests) == 2
        assert all("r" * 513 in json.dumps(request.messages) for request in summary_requests)
        assert all(
            "result omitted from context" not in json.dumps(request.messages)
            for request in summary_requests
        )
        assert "r" * 513 not in json.dumps(main_requests[0].messages)
        assert "result omitted from context" not in json.dumps(main_requests[0].messages)
        assert (
            sum(
                message.get("role") == "user" and "new request" in str(message.get("content"))
                for message in main_requests[0].messages
            )
            == 1
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    restored = Session.load(loop.session.workspace_state, session_id, partition=partition)
    assert restored.last_compacted == len(history)
    assert restored.metadata["summary"] == "action summary"
    assert restored.messages[: len(history)] == original
    assert all("result omitted from context" not in str(message) for message in restored.messages)
