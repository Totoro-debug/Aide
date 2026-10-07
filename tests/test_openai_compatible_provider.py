"""OpenAI-compatible SDK adapter behavior through the public ModelProvider seam."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from omni.agent.tools.tool_gateway import ModelToolCall
from omni.config.config import ProviderConfiguration
from omni.provider.errors import ModelCallError
from omni.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ReasoningDelta,
    TextDelta,
)
from omni.provider.openai_compatible import OpenAICompatibleProvider
from tests.fixtures.provider import error_info_fields, model_response_fields

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


class FakeOpenAIStream:
    def __init__(self, *chunks: object, error: BaseException | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def __aiter__(self) -> AsyncIterator[object]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class FakeOpenAIError(Exception):
    def __init__(
        self,
        status_code: int | None,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(headers={} if headers is None else headers)
        super().__init__("sensitive upstream detail")


class FakeCompletions:
    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeOpenAIClient:
    def __init__(self, *results: object) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(*results))
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


@dataclass(slots=True)
class FakeOpenAIClientFactory:
    client: FakeOpenAIClient
    calls: list[dict[str, object]] = field(default_factory=list, init=False)

    def __call__(self, **kwargs: object) -> FakeOpenAIClient:
        self.calls.append(kwargs)
        return self.client


def configuration() -> ProviderConfiguration:
    return ProviderConfiguration(
        provider_id="openai-local",
        protocol="openai-compatible",
        base_url="https://openai-compatible.test/v1",
        api_key="secret-key",
        models=("model-test",),
    )


def request(*, stream: bool = True) -> dict[str, Any]:
    del stream
    return {
        "messages": [
            {"role": "system", "content": "You are Omni."},
            {"role": "user", "content": "Hello"},
        ],
        "tools": (READ_FILE_SCHEMA,),
        "model": "model-test",
        "max_output": 512,
        "temperature": 0.25,
        "reasoning_effort": "high",
        "timeout": 17,
    }


def completion_request(route: str) -> dict[str, Any]:
    del route
    return {
        "messages": [
            {"role": "system", "content": "Summarize accurately."},
            {"role": "user", "content": "Conversation transcript"},
        ],
        "tools": (),
        "model": "model-test",
        "max_output": 256,
        "temperature": 0.1,
        "reasoning_effort": None,
        "timeout": 23,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    ("cache_fields", "expected"),
    [
        ({"prompt_tokens_details": SimpleNamespace(cached_tokens=5)}, 5),
        ({"prompt_cache_hit_tokens": 6}, 6),
        ({"prompt_tokens_details": SimpleNamespace(cached_tokens=0)}, 0),
        ({}, None),
    ],
)
async def test_provider_preserves_reported_cache_hits(
    streaming: bool, cache_fields: dict[str, object], expected: int | None,
) -> None:
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2, **cache_fields)
    result = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hello", tool_calls=None),
            delta=SimpleNamespace(content="Hello", tool_calls=None),
            finish_reason="stop",
        )],
        usage=usage,
    )
    client = FakeOpenAIClient(FakeOpenAIStream(result) if streaming else result)
    provider = OpenAICompatibleProvider(
        configuration(), client_factory=FakeOpenAIClientFactory(client),
    )
    if streaming:
        events = [event async for event in provider.stream(**request())]
        terminal = events[-1]
        assert isinstance(terminal, ModelCompleted)
        response = terminal.response
    else:
        response = await provider.complete(**request())
    assert response.usage.cached_input_tokens == expected
    assert response.usage.to_dict() == {
        "input_tokens": 7, "output_tokens": 2, "total_tokens": 9,
    }


@pytest.mark.asyncio
async def test_stream_translates_text_and_usage_through_official_sdk_boundary() -> None:
    stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hel", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="lo", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        ),
    )
    client = FakeOpenAIClient(stream)
    factory = FakeOpenAIClientFactory(client)
    provider = OpenAICompatibleProvider(configuration(), client_factory=factory)

    events = [event async for event in provider.stream(**request())]

    assert events[:2] == [TextDelta(delta="Hel"), TextDelta(delta="lo")]
    completed = events[-1]
    assert isinstance(completed, ModelCompleted)
    assert completed.response.continuation is None
    assert model_response_fields(completed.response) == {
        "message": {"role": "assistant", "content": "Hello", "tool_calls": []},
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        "finish_reason": "stop",
    }
    assert factory.calls == [
        {
            "api_key": "secret-key",
            "base_url": "https://openai-compatible.test/v1",
            "max_retries": 0,
        }
    ]
    assert client.chat.completions.calls == [
        {
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": "You are Omni."},
                {"role": "user", "content": "Hello"},
            ],
            "model": "model-test",
            "stream": True,
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
            "temperature": 0.25,
            "timeout": 17,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a text file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("low", "low"),
        ("mid", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
        (None, None),
    ],
)
async def test_stream_maps_or_omits_each_reasoning_effort(
    effort: str | None,
    wire_effort: str | None,
) -> None:
    stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Done", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    )
    client = FakeOpenAIClient(stream)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    events = [event async for event in provider.stream(**{**request(), "reasoning_effort": effort})]

    assert isinstance(events[-1], ModelCompleted)
    reasoning_arguments = {
        key: value
        for key, value in client.chat.completions.calls[0].items()
        if key in {"reasoning_effort", "output_config"}
    }
    assert reasoning_arguments == ({} if effort is None else {"reasoning_effort": wire_effort})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("low", "low"),
        ("mid", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
        (None, None),
    ],
)
async def test_complete_maps_or_omits_each_reasoning_effort(
    effort: str | None,
    wire_effort: str | None,
) -> None:
    sdk_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    client = FakeOpenAIClient(sdk_response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    response = await provider.complete(**{**request(stream=False), "reasoning_effort": effort})

    assert response.message.content == "Done"
    reasoning_arguments = {
        key: value
        for key, value in client.chat.completions.calls[0].items()
        if key in {"reasoning_effort", "output_config"}
    }
    assert reasoning_arguments == ({} if effort is None else {"reasoning_effort": wire_effort})


@pytest.mark.asyncio
async def test_stream_aggregates_fragmented_tool_calls_with_mixed_content() -> None:
    stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="I can ", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_",
                                function=SimpleNamespace(name="read_", arguments='{"pa'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="inspect.",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="123",
                                function=SimpleNamespace(name="file", arguments='th":"README.md"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15),
        ),
    )
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(FakeOpenAIClient(stream)),
    )

    events = [event async for event in provider.stream(**request())]

    assert events[:2] == [TextDelta(delta="I can "), TextDelta(delta="inspect.")]
    completed = events[-1]
    assert isinstance(completed, ModelCompleted)
    assert model_response_fields(completed.response) == {
        "message": {
            "role": "assistant",
            "content": "I can inspect.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
        },
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
        "finish_reason": "tool_calls",
    }
    assert completed.response.continuation is None


@pytest.mark.asyncio
async def test_stream_preserves_interleaved_reasoning_and_replays_latest_assistant_turn() -> None:
    first_stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content="Plan",
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Answer",
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_read",
                                function=SimpleNamespace(name="read_file", arguments='{"path":"'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=" more",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='README.md"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        ),
    )
    second_stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Finished",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=19, completion_tokens=2, total_tokens=21),
        )
    )
    client = FakeOpenAIClient(first_stream, second_stream)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    first_events = [event async for event in provider.stream(**request())]

    assert first_events[:-1] == [
        ReasoningDelta(delta="Plan"),
        TextDelta(delta="Answer"),
        ReasoningDelta(delta=" more"),
    ]
    first_completed = first_events[-1]
    assert isinstance(first_completed, ModelCompleted)
    continuation = first_completed.response.continuation
    assert continuation == ModelContinuation(
        provider_id="openai-local",
        payload="Plan more",
    )

    second_request = request()
    second_request["messages"] = [
        {"role": "system", "content": "You are Omni."},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": "Answer",
            "tool_calls": [
                {
                    "id": "call_read",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "name": "read_file",
            "content": "Project documentation",
        },
    ]
    assert continuation is not None
    second_events = [
        event
        async for event in provider.stream(
            **second_request,
            continuation=continuation,
        )
    ]

    assert isinstance(second_events[-1], ModelCompleted)
    assert client.chat.completions.calls[1]["messages"] == [
        {"role": "system", "content": "You are Omni."},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": "Answer",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
            "reasoning_content": "Plan more",
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "Project documentation"},
    ]


@pytest.mark.asyncio
async def test_complete_retains_reasoning_continuation_without_stream_event() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Answer",
                    reasoning_content="Plan",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_read",
                            function=SimpleNamespace(
                                name="read_file",
                                arguments='{"path":"README.md"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    client = FakeOpenAIClient(response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    observed = await provider.complete(**request(stream=False))

    assert observed.message.content == "Answer"
    assert observed.message.tool_calls == (
        ModelToolCall(
            id="call_read",
            name="read_file",
            arguments='{"path":"README.md"}',
        ),
    )
    assert observed.continuation == ModelContinuation(
        provider_id="openai-local",
        payload="Plan",
    )


@pytest.mark.asyncio
async def test_stream_rejects_continuation_owned_by_another_provider_before_sdk_call() -> None:
    client = FakeOpenAIClient(FakeOpenAIStream())
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError):
        async for _event in provider.stream(
            **request(),
            continuation=ModelContinuation(
                provider_id="anthropic-default",
                payload=({"type": "thinking", "thinking": "opaque"},),
            ),
        ):
            pass

    assert client.chat.completions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["memory", "schedule"])
async def test_complete_normalizes_memory_and_schedule_responses(route: str) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Concise summary", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=19, completion_tokens=3, total_tokens=22),
    )
    client = FakeOpenAIClient(response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    observed = await provider.complete(**completion_request(route))

    assert model_response_fields(observed) == {
        "message": {"role": "assistant", "content": "Concise summary", "tool_calls": []},
        "usage": {"input_tokens": 19, "output_tokens": 3, "total_tokens": 22},
        "finish_reason": "stop",
    }
    assert client.chat.completions.calls == [
        {
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "Summarize accurately."},
                {"role": "user", "content": "Conversation transcript"},
            ],
            "model": "model-test",
            "stream": False,
            "temperature": 0.1,
            "timeout": 23,
            "tools": [],
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" \n", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        ),
    ],
    ids=("no-choices", "blank-message"),
)
async def test_complete_rejects_an_empty_success_response(response: object) -> None:
    client = FakeOpenAIClient(response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError) as raised:
        await provider.complete(**completion_request("memory"))

    assert error_info_fields(raised.value.error) == {
        "code": "model_failed",
        "message": (
            "OpenAI-compatible provider returned an empty response. "
            "Check its API base URL and model configuration."
        ),
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_complete_translates_tool_history_and_mixed_tool_response() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I will update memory.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_edit",
                            function=SimpleNamespace(
                                name="edit_file",
                                arguments='{"path":"memory.md","content":"fact"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=31, completion_tokens=6, total_tokens=37),
    )
    client = FakeOpenAIClient(response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )
    model_request = completion_request("memory")
    model_request["messages"] = [
        {"role": "system", "content": "Summarize accurately."},
        {"role": "user", "content": "Read memory."},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {
                    "id": "call_read",
                    "name": "read_file",
                    "arguments": '{"path":"memory.md"}',
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "name": "read_file",
            "content": "Existing memory",
        },
    ]

    observed = await provider.complete(**model_request)

    assert model_response_fields(observed) == {
        "message": {
            "role": "assistant",
            "content": "I will update memory.",
            "tool_calls": [
                {
                    "id": "call_edit",
                    "name": "edit_file",
                    "arguments": '{"path":"memory.md","content":"fact"}',
                }
            ],
        },
        "usage": {"input_tokens": 31, "output_tokens": 6, "total_tokens": 37},
        "finish_reason": "tool_calls",
    }
    assert client.chat.completions.calls[0]["messages"] == [
        {"role": "system", "content": "Summarize accurately."},
        {"role": "user", "content": "Read memory."},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"memory.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "Existing memory"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            FakeOpenAIError(401),
            {
                "code": "provider_auth_error",
                "message": "OpenAI-compatible provider authentication failed.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            FakeOpenAIError(429, headers={"Retry-After": "7.5"}),
            {
                "code": "provider_rate_limited",
                "message": "OpenAI-compatible provider rate limited the request.",
                "retryable": True,
                "retry_after_seconds": 7.5,
            },
        ),
        (
            FakeOpenAIError(408),
            {
                "code": "provider_timeout",
                "message": "OpenAI-compatible provider request timed out.",
                "retryable": True,
                "retry_after_seconds": None,
            },
        ),
        (
            FakeOpenAIError(529, headers={"Retry-After": "4.25"}),
            {
                "code": "provider_unavailable",
                "message": "OpenAI-compatible provider is temporarily unavailable.",
                "retryable": True,
                "retry_after_seconds": 4.25,
            },
        ),
        (
            FakeOpenAIError(404, code="model_not_found"),
            {
                "code": "route_unavailable",
                "message": "OpenAI-compatible model or capability is unavailable.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            FakeOpenAIError(501, code="unsupported"),
            {
                "code": "route_unavailable",
                "message": "OpenAI-compatible model or capability is unavailable.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            FakeOpenAIError(400),
            {
                "code": "model_invalid_request",
                "message": "OpenAI-compatible provider rejected the request.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            FakeOpenAIError(400, code="context_length_exceeded"),
            {
                "code": "model_context_overflow",
                "message": "OpenAI-compatible request exceeds the model context window.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            RuntimeError("sensitive unexpected failure"),
            {
                "code": "model_failed",
                "message": "OpenAI-compatible provider call failed.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
    ],
)
async def test_complete_maps_sdk_failures_once(
    failure: Exception,
    expected: dict[str, object],
) -> None:
    success = SimpleNamespace(choices=[], usage=None)
    client = FakeOpenAIClient(failure, success)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError) as raised:
        await provider.complete(**completion_request("memory"))

    assert error_info_fields(raised.value.error) == expected
    assert raised.value.__cause__ is failure
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_stream_maps_iteration_timeout_without_retrying() -> None:
    failure = TimeoutError("sensitive timeout detail")
    client = FakeOpenAIClient(FakeOpenAIStream(error=failure))
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError) as raised:
        async for _event in provider.stream(**request()):
            pass

    assert error_info_fields(raised.value.error) == {
        "code": "provider_timeout",
        "message": "OpenAI-compatible provider request timed out.",
        "retryable": True,
        "retry_after_seconds": None,
    }
    assert raised.value.__cause__ is failure
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_stream_maps_creation_error_without_retrying() -> None:
    failure = FakeOpenAIError(401)
    client = FakeOpenAIClient(failure, FakeOpenAIStream())
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError) as raised:
        async for _event in provider.stream(**request()):
            pass

    assert error_info_fields(raised.value.error) == {
        "code": "provider_auth_error",
        "message": "OpenAI-compatible provider authentication failed.",
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert raised.value.__cause__ is failure
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_stream_rejects_an_empty_success_response() -> None:
    client = FakeOpenAIClient(FakeOpenAIStream())
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(ModelCallError) as raised:
        async for _event in provider.stream(**request()):
            pass

    assert error_info_fields(raised.value.error) == {
        "code": "model_failed",
        "message": (
            "OpenAI-compatible provider returned an empty response. "
            "Check its API base URL and model configuration."
        ),
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_stream_preserves_malformed_tool_argument_text_for_the_gateway() -> None:
    stream = FakeOpenAIStream(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_invalid",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10),
        )
    )
    client = FakeOpenAIClient(stream)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    events = [event async for event in provider.stream(**request())]

    completed = events[-1]
    assert isinstance(completed, ModelCompleted)
    assert completed.response.message.tool_calls == (
        ModelToolCall(id="call_invalid", name="read_file", arguments='{"path":'),
    )
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_complete_preserves_non_object_tool_argument_text_for_the_gateway() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_invalid",
                            function=SimpleNamespace(name="read_file", arguments="[]"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10),
    )
    client = FakeOpenAIClient(response)
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    observed = await provider.complete(**completion_request("memory"))

    assert observed.message.tool_calls == (
        ModelToolCall(id="call_invalid", name="read_file", arguments="[]"),
    )
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_stream_propagates_cancellation_without_retrying() -> None:
    cancellation = asyncio.CancelledError()
    client = FakeOpenAIClient(FakeOpenAIStream(error=cancellation))
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        async for _event in provider.stream(**request()):
            pass

    assert raised.value is cancellation
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_close_closes_the_injected_official_client() -> None:
    client = FakeOpenAIClient()
    provider = OpenAICompatibleProvider(
        configuration(),
        client_factory=FakeOpenAIClientFactory(client),
    )

    await provider.close()

    assert client.closed is True
    assert client.close_calls == 1
