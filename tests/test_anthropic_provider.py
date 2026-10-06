"""Anthropic SDK adapter behavior through the public ModelProvider seam."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from httpx import Request, Response

from omni.agent.tools.tool_gateway import ModelToolCall
from omni.config.config import ProviderConfiguration
from omni.provider.anthropic import AnthropicProvider
from omni.provider.errors import EmptyModelResponseError, ModelCallError
from omni.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ReasoningDelta,
    TextDelta,
)
from tests.fixtures.provider import error_info_fields, model_response_fields

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


class FakeAnthropicStream:
    def __init__(self, *events: object) -> None:
        self._events = events

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._events:
            if isinstance(event, BaseException):
                raise event
            yield event


class FakeMessages:
    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeAnthropicClient:
    def __init__(self, *results: object) -> None:
        self.messages = FakeMessages(*results)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeAnthropicClientFactory:
    def __init__(self, client: FakeAnthropicClient) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_retries: int,
    ) -> FakeAnthropicClient:
        self.calls.append({"api_key": api_key, "base_url": base_url, "max_retries": max_retries})
        return self.client


def configuration() -> ProviderConfiguration:
    return ProviderConfiguration(
        provider_id="anthropic-default",
        protocol="anthropic",
        base_url="https://api.anthropic.test",
        api_key="secret-key",
        models=("claude-test",),
    )


def request(*, stream: bool = True) -> dict[str, Any]:
    del stream
    return {
        "messages": [
            {"role": "system", "content": "You are Omni."},
            {"role": "user", "content": "Hello"},
        ],
        "tools": (READ_FILE_SCHEMA,),
        "model": "claude-test",
        "max_output": 512,
        "temperature": 0.25,
        "reasoning_effort": "high",
        "timeout": 17,
    }


@pytest.mark.asyncio
async def test_stream_translates_text_and_usage_through_official_sdk_boundary() -> None:
    stream = FakeAnthropicStream(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hel"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="lo"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=2),
        ),
        SimpleNamespace(type="message_stop"),
    )
    client = FakeAnthropicClient(stream)
    factory = FakeAnthropicClientFactory(client)
    provider = AnthropicProvider(configuration(), client_factory=factory)

    events = [event async for event in provider.stream(**request())]

    assert events[:2] == [TextDelta(delta="Hel"), TextDelta(delta="lo")]
    assert len(events) == 3
    completed = events[-1]
    assert isinstance(completed, ModelCompleted)
    assert model_response_fields(completed.response) == {
        "message": {"role": "assistant", "content": "Hello", "tool_calls": []},
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        "finish_reason": "stop",
    }
    assert factory.calls == [
        {
            "api_key": "secret-key",
            "base_url": "https://api.anthropic.test",
            "max_retries": 0,
        }
    ]
    assert client.messages.calls == [
        {
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-test",
            "stream": True,
            "system": "You are Omni.",
            "temperature": 0.25,
            "timeout": 17,
            "output_config": {"effort": "high"},
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a text file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
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
    stream = FakeAnthropicStream(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Done"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=1),
        ),
        SimpleNamespace(type="message_stop"),
    )
    client = FakeAnthropicClient(stream)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    events = [event async for event in provider.stream(**{**request(), "reasoning_effort": effort})]

    assert isinstance(events[-1], ModelCompleted)
    reasoning_arguments = {
        key: value
        for key, value in client.messages.calls[0].items()
        if key in {"output_config", "reasoning_effort"}
    }
    assert reasoning_arguments == ({} if effort is None else {"output_config": {"effort": wire_effort}})


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
    sdk_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Done")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = FakeAnthropicClient(sdk_message)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    response = await provider.complete(**{**request(stream=False), "reasoning_effort": effort})

    assert response.message.content == "Done"
    reasoning_arguments = {
        key: value
        for key, value in client.messages.calls[0].items()
        if key in {"output_config", "reasoning_effort"}
    }
    assert reasoning_arguments == ({} if effort is None else {"output_config": {"effort": wire_effort}})


@pytest.mark.asyncio
async def test_stream_rejects_an_empty_success_response() -> None:
    client = FakeAnthropicClient(FakeAnthropicStream())
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        async for _event in provider.stream(**request()):
            pass

    assert error_info_fields(captured.value.error) == {
        "code": "model_failed",
        "message": "Anthropic provider returned an empty response. Check its model configuration.",
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert isinstance(captured.value.__cause__, EmptyModelResponseError)
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_stream_aggregates_mixed_text_and_tool_use_content() -> None:
    stream = FakeAnthropicStream(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=11, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Checking"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(
                type="tool_use",
                id="toolu_123",
                name="read_file",
                input={},
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='"README.md"}'),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=6),
        ),
        SimpleNamespace(type="message_stop"),
    )
    provider = AnthropicProvider(
        configuration(), client_factory=FakeAnthropicClientFactory(FakeAnthropicClient(stream))
    )

    events = [event async for event in provider.stream(**request())]

    assert events[:-1] == [TextDelta(delta="Checking")]
    completed = events[-1]
    assert isinstance(completed, ModelCompleted)
    assert completed.response.message.content == "Checking"
    assert completed.response.message.tool_calls == (
        ModelToolCall(id="toolu_123", name="read_file", arguments='{"path":"README.md"}'),
    )
    assert completed.response.usage.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 6,
        "total_tokens": 17,
    }
    assert completed.response.finish_reason == "tool_calls"
    assert completed.response.continuation is None


@pytest.mark.asyncio
async def test_stream_preserves_interleaved_thinking_blocks_and_replays_continuation() -> None:
    first_stream = FakeAnthropicStream(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=11, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="thinking", thinking="", signature=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking="Plan"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="text_delta", text="Answer"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=2,
            content_block=SimpleNamespace(
                type="redacted_thinking",
                data="opaque-interleaved",
            ),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=3,
            content_block=SimpleNamespace(
                type="tool_use",
                id="toolu_interleaved",
                name="read_file",
                input={},
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=3,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":"README.md"}'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking=" more"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="signature_delta", signature="sig-interleaved"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=7),
        ),
    )
    second_stream = FakeAnthropicStream(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=19, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Finished"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=2),
        ),
    )
    client = FakeAnthropicClient(first_stream, second_stream)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

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
        provider_id="anthropic-default",
        payload=(
            {"type": "thinking", "thinking": "Plan more", "signature": "sig-interleaved"},
            {"type": "text", "text": "Answer"},
            {"type": "redacted_thinking", "data": "opaque-interleaved"},
            {
                "type": "tool_use",
                "id": "toolu_interleaved",
                "name": "read_file",
                "input": {"path": "README.md"},
            },
        ),
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
                    "id": "toolu_interleaved",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_interleaved",
            "name": "read_file",
            "content": "Project documentation",
        },
    ]
    second_events = [
        event
        async for event in provider.stream(
            **second_request,
            continuation=continuation,
        )
    ]
    assert isinstance(second_events[-1], ModelCompleted)
    assert client.messages.calls[1]["messages"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer"}]},
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Plan more", "signature": "sig-interleaved"},
                {"type": "text", "text": "Answer"},
                {"type": "redacted_thinking", "data": "opaque-interleaved"},
                {
                    "type": "tool_use",
                    "id": "toolu_interleaved",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_interleaved",
                    "content": "Project documentation",
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_complete_retains_redacted_and_signature_only_continuation_without_reasoning_event() -> (
    None
):
    sdk_message = SimpleNamespace(
        content=[
            SimpleNamespace(type="redacted_thinking", data="opaque-redacted"),
            SimpleNamespace(type="thinking", thinking="", signature="sig-only"),
            SimpleNamespace(type="text", text="Done"),
        ],
        usage=SimpleNamespace(input_tokens=13, output_tokens=4),
        stop_reason="end_turn",
    )
    client = FakeAnthropicClient(sdk_message)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    response = await provider.complete(**request(stream=False))

    assert response.message.content == "Done"
    assert response.continuation == ModelContinuation(
        provider_id="anthropic-default",
        payload=(
            {"type": "redacted_thinking", "data": "opaque-redacted"},
            {"type": "thinking", "thinking": "", "signature": "sig-only"},
            {"type": "text", "text": "Done"},
        ),
    )


@pytest.mark.asyncio
async def test_stream_rejects_continuation_owned_by_another_provider_before_sdk_call() -> None:
    client = FakeAnthropicClient(FakeAnthropicStream())
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError):
        async for _event in provider.stream(
            **request(),
            continuation=ModelContinuation(
                provider_id="openai-local",
                payload="opaque-openai-state",
            ),
        ):
            pass

    assert client.messages.calls == []


@pytest.mark.asyncio
async def test_complete_translates_mixed_history_and_full_message() -> None:
    sdk_message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I found it."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_next",
                name="read_file",
                input={"path": "NEXT.md"},
            ),
        ],
        usage=SimpleNamespace(input_tokens=23, output_tokens=8),
        stop_reason="tool_use",
    )
    client = FakeAnthropicClient(sdk_message)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))
    complete_request = request(stream=False)
    complete_request["messages"] = [
        {"role": "system", "content": "You are Omni."},
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "I will read it.",
            "tool_calls": [
                {
                    "id": "toolu_prior",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_prior",
            "name": "read_file",
            "content": "Project documentation",
        },
    ]

    response = await provider.complete(**complete_request)

    assert model_response_fields(response) == {
        "message": {
            "role": "assistant",
            "content": "I found it.",
            "tool_calls": [
                {
                    "id": "toolu_next",
                    "name": "read_file",
                    "arguments": '{"path":"NEXT.md"}',
                }
            ],
        },
        "usage": {"input_tokens": 23, "output_tokens": 8, "total_tokens": 31},
        "finish_reason": "tool_calls",
    }
    assert client.messages.calls == [
        {
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will read it."},
                        {
                            "type": "tool_use",
                            "id": "toolu_prior",
                            "name": "read_file",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_prior",
                            "content": "Project documentation",
                        }
                    ],
                },
            ],
            "model": "claude-test",
            "stream": False,
            "system": "You are Omni.",
            "temperature": 0.25,
            "timeout": 17,
            "output_config": {"effort": "high"},
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a text file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_complete_rejects_an_empty_success_response() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=" \n")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        stop_reason="end_turn",
    )
    client = FakeAnthropicClient(response)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "model_failed",
        "message": "Anthropic provider returned an empty response. Check its model configuration.",
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert isinstance(captured.value.__cause__, EmptyModelResponseError)
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_complete_normalizes_timeout_without_adapter_retry() -> None:
    client = FakeAnthropicClient(
        APITimeoutError(Request("POST", "https://api.anthropic.test/v1/messages"))
    )
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "provider_timeout",
        "message": "Anthropic request timed out.",
        "retryable": True,
        "retry_after_seconds": None,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_complete_preserves_numeric_retry_after_for_rate_limit() -> None:
    response = Response(
        429,
        request=Request("POST", "https://api.anthropic.test/v1/messages"),
        headers={"Retry-After": "12.5"},
    )
    client = FakeAnthropicClient(
        RateLimitError(
            "sensitive provider response",
            response=response,
            body={"type": "error", "error": {"type": "rate_limit_error"}},
        )
    )
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "provider_rate_limited",
        "message": "Anthropic rate limit was reached.",
        "retryable": True,
        "retry_after_seconds": 12.5,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_complete_normalizes_authentication_as_permanent() -> None:
    response = Response(
        401,
        request=Request("POST", "https://api.anthropic.test/v1/messages"),
    )
    client = FakeAnthropicClient(
        AuthenticationError(
            "secret authentication detail",
            response=response,
            body={"type": "error", "error": {"type": "authentication_error"}},
        )
    )
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "provider_auth_error",
        "message": "Anthropic authentication failed.",
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sdk_error",
    [
        APIConnectionError(request=Request("POST", "https://api.anthropic.test/v1/messages")),
        InternalServerError(
            "provider overload detail",
            response=Response(
                529,
                request=Request("POST", "https://api.anthropic.test/v1/messages"),
            ),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        ),
    ],
    ids=("connection", "overloaded"),
)
async def test_complete_normalizes_temporary_unavailability(sdk_error: Exception) -> None:
    client = FakeAnthropicClient(sdk_error)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "provider_unavailable",
        "message": "Anthropic is temporarily unavailable.",
        "retryable": True,
        "retry_after_seconds": None,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_complete_preserves_retry_after_for_temporary_unavailability() -> None:
    sdk_error = InternalServerError(
        "provider overload detail",
        response=Response(
            529,
            request=Request("POST", "https://api.anthropic.test/v1/messages"),
            headers={"Retry-After": "4.25"},
        ),
        body={"type": "error", "error": {"type": "overloaded_error"}},
    )
    client = FakeAnthropicClient(sdk_error)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "provider_unavailable",
        "message": "Anthropic is temporarily unavailable.",
        "retryable": True,
        "retry_after_seconds": 4.25,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (
            NotFoundError(
                "model name was not found",
                response=Response(
                    404,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                ),
                body={"type": "error", "error": {"type": "not_found_error"}},
            ),
            {
                "code": "route_unavailable",
                "message": "Anthropic model or capability is unavailable.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            BadRequestError(
                "invalid field detail",
                response=Response(
                    400,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                ),
                body={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "invalid field"},
                },
            ),
            {
                "code": "model_invalid_request",
                "message": "Anthropic rejected the model request.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            BadRequestError(
                "sensitive token counts",
                response=Response(
                    400,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                ),
                body={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "prompt is too long: 210000 tokens > 200000 maximum",
                    },
                },
            ),
            {
                "code": "model_context_overflow",
                "message": "Anthropic model context was exceeded.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            APIStatusError(
                "request body too large",
                response=Response(
                    413,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                ),
                body={"type": "error", "error": {"type": "request_too_large"}},
            ),
            {
                "code": "model_context_overflow",
                "message": "Anthropic model context was exceeded.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
        (
            BadRequestError(
                "model does not support tools",
                response=Response(
                    400,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                ),
                body={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "This model does not support tool use.",
                    },
                },
            ),
            {
                "code": "route_unavailable",
                "message": "Anthropic model or capability is unavailable.",
                "retryable": False,
                "retry_after_seconds": None,
            },
        ),
    ],
    ids=("not-found", "invalid", "context", "too-large", "unsupported"),
)
async def test_complete_normalizes_permanent_provider_errors(
    sdk_error: Exception,
    expected: dict[str, object],
) -> None:
    client = FakeAnthropicClient(sdk_error)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == expected
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_result", "expected_code", "expected_retry_after", "expected_deltas"),
    [
        (
            RateLimitError(
                "limited",
                response=Response(
                    429,
                    request=Request("POST", "https://api.anthropic.test/v1/messages"),
                    headers={"Retry-After": "3"},
                ),
                body={"type": "error", "error": {"type": "rate_limit_error"}},
            ),
            "provider_rate_limited",
            3.0,
            [],
        ),
        (
            FakeAnthropicStream(
                SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=SimpleNamespace(type="text_delta", text="partial"),
                ),
                APIConnectionError(
                    request=Request("POST", "https://api.anthropic.test/v1/messages")
                ),
            ),
            "provider_unavailable",
            None,
            [TextDelta(delta="partial")],
        ),
    ],
    ids=("creation", "iteration"),
)
async def test_stream_normalizes_sdk_errors_without_retry_or_completed(
    sdk_result: object,
    expected_code: str,
    expected_retry_after: float | None,
    expected_deltas: list[TextDelta],
) -> None:
    client = FakeAnthropicClient(sdk_result)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))
    observed: list[TextDelta] = []

    with pytest.raises(ModelCallError) as captured:
        async for event in provider.stream(**request()):
            assert isinstance(event, TextDelta)
            observed.append(event)

    assert observed == expected_deltas
    assert captured.value.error.code == expected_code
    assert captured.value.error.retry_after_seconds == expected_retry_after
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_complete_normalizes_unclassified_sdk_failure() -> None:
    sdk_error = APIError(
        "sensitive unclassified detail",
        Request("POST", "https://api.anthropic.test/v1/messages"),
        body=None,
    )
    client = FakeAnthropicClient(sdk_error)
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "model_failed",
        "message": "Anthropic model call failed.",
        "retryable": False,
        "retry_after_seconds": None,
    }
    assert len(client.messages.calls) == 1


@pytest.mark.asyncio
async def test_close_releases_official_sdk_client() -> None:
    client = FakeAnthropicClient()
    provider = AnthropicProvider(configuration(), client_factory=FakeAnthropicClientFactory(client))

    await provider.close()

    assert client.closed is True


@pytest.mark.asyncio
async def test_stream_normalizes_malformed_tool_arguments_as_model_failure() -> None:
    stream = FakeAnthropicStream(
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use",
                id="toolu_bad",
                name="read_file",
                input={},
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":'),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=1),
        ),
    )
    provider = AnthropicProvider(
        configuration(), client_factory=FakeAnthropicClientFactory(FakeAnthropicClient(stream))
    )

    with pytest.raises(ModelCallError) as captured:
        async for _event in provider.stream(**request()):
            pass

    assert error_info_fields(captured.value.error) == {
        "code": "model_failed",
        "message": "Anthropic model call failed.",
        "retryable": False,
        "retry_after_seconds": None,
    }


@pytest.mark.asyncio
async def test_complete_normalizes_malformed_tool_arguments_as_model_failure() -> None:
    sdk_message = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_bad",
                name="read_file",
                input=["not", "an", "object"],
            )
        ],
        usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        stop_reason="tool_use",
    )
    provider = AnthropicProvider(
        configuration(),
        client_factory=FakeAnthropicClientFactory(FakeAnthropicClient(sdk_message)),
    )

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(**request(stream=False))

    assert error_info_fields(captured.value.error) == {
        "code": "model_failed",
        "message": "Anthropic model call failed.",
        "retryable": False,
        "retry_after_seconds": None,
    }


@pytest.mark.asyncio
async def test_stream_propagates_cancellation_without_completed_event() -> None:
    stream = FakeAnthropicStream(
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="partial"),
        ),
        asyncio.CancelledError(),
    )
    provider = AnthropicProvider(
        configuration(), client_factory=FakeAnthropicClientFactory(FakeAnthropicClient(stream))
    )
    observed: list[TextDelta] = []

    with pytest.raises(asyncio.CancelledError):
        async for event in provider.stream(**request()):
            assert isinstance(event, TextDelta)
            observed.append(event)

    assert observed == [TextDelta(delta="partial")]
