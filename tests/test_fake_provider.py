import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from aide.agent.tools.tool_gateway import ModelToolCall
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
    TextDelta,
)
from tests.fixtures.provider import ProviderCall, ScriptedFakeProvider, StreamScript


async def collect(stream: AsyncIterator[object]) -> list[object]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_scripted_fake_provider_replays_stream_events_in_order() -> None:
    response = ModelResponse(
        message=AssistantModelMessage(
            content="Hello",
            tool_calls=(
                ModelToolCall(
                    id="call_123",
                    name="read_file",
                    arguments='{"path":"CONTEXT.md"}',
                ),
            ),
        ),
        usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        finish_reason="tool_calls",
    )
    events = (
        TextDelta(delta="Hello"),
        ModelCompleted(response=response),
    )
    provider = ScriptedFakeProvider(streams=[StreamScript(events=events)])
    call: dict[str, Any] = {
        "messages": [{"role": "system", "content": "System"}],
        "tools": (),
        "model": "model",
        "max_output": 10,
        "temperature": 0.2,
        "reasoning_effort": None,
        "timeout": 30,
    }

    observed = await collect(provider.stream(**call))

    assert observed == list(events)
    assert provider.stream_requests == [ProviderCall(**call)]


@pytest.mark.asyncio
async def test_scripted_fake_provider_returns_complete_responses_in_order() -> None:
    response = ModelResponse(
        message=AssistantModelMessage(content="summary", tool_calls=()),
        usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(completions=[response])
    call: dict[str, Any] = {
        "messages": [{"role": "system", "content": "System"}],
        "tools": (),
        "model": "model",
        "max_output": 10,
        "temperature": 0.2,
        "reasoning_effort": None,
        "timeout": 30,
    }

    observed = await provider.complete(**call)

    assert observed is response
    assert provider.complete_requests == [ProviderCall(**call)]


@pytest.mark.asyncio
async def test_scripted_fake_provider_raises_scripted_failures_and_cancellation() -> None:
    retryable_error = TimeoutError("retry this provider call")
    final_error = RuntimeError("provider failed")
    provider = ScriptedFakeProvider(
        streams=[
            StreamScript(events=(), error=retryable_error),
            StreamScript(events=(), error=asyncio.CancelledError()),
        ],
        completions=[final_error],
    )

    with pytest.raises(TimeoutError, match="retry this provider call"):
        await collect(
            provider.stream(
                messages=[],
                tools=(),
                model="model",
                max_output=10,
                temperature=0.2,
                reasoning_effort=None,
                timeout=30,
            )
        )
    with pytest.raises(asyncio.CancelledError):
        await collect(
            provider.stream(
                messages=[],
                tools=(),
                model="model",
                max_output=10,
                temperature=0.2,
                reasoning_effort=None,
                timeout=30,
            )
        )
    with pytest.raises(RuntimeError, match="provider failed"):
        await provider.complete(
            messages=[],
            tools=(),
            model="model",
            max_output=10,
            temperature=0.2,
            reasoning_effort=None,
            timeout=30,
        )


@pytest.mark.asyncio
async def test_scripted_fake_provider_reports_when_it_is_closed() -> None:
    provider = ScriptedFakeProvider()

    await provider.close()

    assert provider.closed is True
