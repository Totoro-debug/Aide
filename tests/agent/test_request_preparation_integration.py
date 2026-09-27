from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pytest

from myclaw.agent.session.session import Session, SessionStoragePartition
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
        VALID_CONFIG
        + """

[models.routes.chat]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 4096
max_output = 512
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.schedule]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 4096
max_output = 512
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.memory]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
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
