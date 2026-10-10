"""Direct Model Router and Provider call seams."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import APIConnectionError
from httpx import Request

from aide.config.config import (
    MemoryConfiguration,
    ModelConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from aide.provider.anthropic import AnthropicProvider
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from aide.provider.openai_compatible import OpenAICompatibleProvider
from aide.utils.errors import ErrorInfo

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


def _configuration() -> UserConfiguration:
    provider = ProviderConfiguration(
        provider_id="provider",
        protocol="openai-compatible",
        base_url="https://provider.example/v1",
        api_key="secret",
        models={'resolved-model': ModelConfiguration(100_000, 2048, 0.2, "high", 42)},
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="resolved-model",
        context_window=100_000,
        max_output=2048,
        temperature=0.2,
        reasoning_effort="high",
        timeout=42,
    )
    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=50_000),
        memory=MemoryConfiguration(
            batch_size=10,
            schedule="0 * * * *",
        ),
        models=ModelsConfiguration(
            providers={provider.provider_id: provider},
            routes={"chat": route},
        ),
    )


class _DirectProvider:
    def __init__(self, completions: list[ModelResponse | BaseException] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._completions = [] if completions is None else list(completions)

    def stream(
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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "max_output": max_output,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
            }
        )

        async def events() -> AsyncIterator[ModelCompleted]:
            yield ModelCompleted(response=_response())

        return events()

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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "max_output": max_output,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
            }
        )
        outcome = self._completions.pop(0) if self._completions else _response()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_router_direct_call_resolves_route_and_passes_only_provider_fields() -> None:
    provider = _DirectProvider()
    router = ModelRouter(
        configuration=_configuration(),
        provider_factory=cast(
            Callable[[ProviderConfiguration], ModelProvider],
            lambda _configuration: provider,
        ),
    )
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}]

    response = await router.complete("chat", messages=messages, tools=(READ_FILE_SCHEMA,))

    assert response.message.content == "done"
    assert provider.calls == [
        {
            "messages": messages,
            "tools": (READ_FILE_SCHEMA,),
            "model": "resolved-model",
            "max_output": 2048,
            "temperature": 0.2,
            "reasoning_effort": "high",
            "timeout": 42,
        }
    ]


class _RetryClock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


@pytest.mark.asyncio
async def test_router_direct_call_keeps_retry_budget_and_reuses_message_dictionaries() -> None:
    provider = _DirectProvider(
        completions=[
            ModelCallError(
                ErrorInfo(
                    code="provider_timeout",
                    message="temporary timeout",
                    retryable=True,
                )
            ),
            _response(),
        ]
    )
    clock = _RetryClock()
    router = ModelRouter(
        configuration=_configuration(),
        provider_factory=cast(
            Callable[[ProviderConfiguration], ModelProvider],
            lambda _configuration: provider,
        ),
        clock=clock,
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hi"},
    ]

    await router.complete("chat", messages=messages, tools=())

    assert clock.sleeps == [0.5]
    assert [call["messages"] for call in provider.calls] == [messages, messages]


@pytest.mark.asyncio
async def test_router_direct_stream_returns_provider_usage() -> None:
    provider = _DirectProvider()
    router = ModelRouter(
        configuration=_configuration(),
        provider_factory=cast(
            Callable[[ProviderConfiguration], ModelProvider],
            lambda _configuration: provider,
        ),
    )

    events = [
        event
        async for event in router.stream(
            "chat",
            messages=[{"role": "system", "content": "System"}],
            tools=(),
        )
    ]

    assert isinstance(events[-1], ModelCompleted)
    assert events[-1].response.usage.to_dict() == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }


def _fallback_configuration() -> UserConfiguration:
    base = _configuration()
    provider = ProviderConfiguration(
        provider_id="chat-provider",
        protocol="openai-compatible",
        base_url="https://chat.example/v1",
        api_key="secret",
        models={'chat-model': ModelConfiguration(100_000, 1024, 0.1, "low", 24)},
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="chat-model",
        context_window=100_000,
        max_output=1024,
        temperature=0.1,
        reasoning_effort="low",
        timeout=24,
    )
    return UserConfiguration(
        runtime=base.runtime,
        memory=base.memory,
        models=ModelsConfiguration(
            providers={**base.models.providers, provider.provider_id: provider},
            routes={**base.models.routes, "title": route},
        ),
    )


@pytest.mark.asyncio
async def test_router_direct_call_falls_back_to_chat_route() -> None:
    default_provider = _DirectProvider()
    requested_provider = _DirectProvider(
        completions=[
            ModelCallError(
                ErrorInfo(
                    code="provider_auth_error",
                    message="authentication failed",
                )
            )
        ]
    )
    providers = {"provider": default_provider, "chat-provider": requested_provider}
    router = ModelRouter(
        configuration=_fallback_configuration(),
        provider_factory=cast(
            Callable[[ProviderConfiguration], ModelProvider],
            lambda configuration: providers[configuration.provider_id],
        ),
    )

    response = await router.complete(
        "title",
        messages=[{"role": "system", "content": "System"}],
        tools=(),
    )

    assert response.message.content == "done"
    assert router.route_status("title").selected_route == "chat"
    assert default_provider.calls[0]["model"] == "resolved-model"


def _response() -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content="done"),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


class _FakeCompletions:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeOpenAIClient:
    def __init__(self, result: object) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(result))

    async def close(self) -> None:
        pass


class _FakeAnthropicMessages:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeAnthropicStream:
    def __init__(self, *events: object) -> None:
        self.events = events

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self.events:
            yield event


class _FakeAnthropicClient:
    def __init__(self, result: object) -> None:
        self.messages = _FakeAnthropicMessages(result)

    async def close(self) -> None:
        pass


def _provider_configuration(protocol: str) -> ProviderConfiguration:
    return ProviderConfiguration(
        provider_id="provider",
        protocol=protocol,
        base_url="https://provider.example/v1",
        api_key="secret",
        models={'model': ModelConfiguration(200_000, 8192, 0.2, "mid", 120)},
    )


@pytest.mark.asyncio
async def test_openai_provider_direct_call_preserves_system_first_message_order() -> None:
    client = _FakeOpenAIClient(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )
    )
    provider = OpenAICompatibleProvider(
        _provider_configuration("openai-compatible"),
        client_factory=lambda **_kwargs: client,
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "I will read it.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "text"},
    ]

    response = await provider.complete(
        messages=messages,
        tools=[READ_FILE_SCHEMA],
        model="model",
        max_output=512,
        temperature=0.25,
        reasoning_effort="high",
        timeout=17,
    )

    assert response.usage.to_dict() == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert client.chat.completions.calls == [
        {
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": "I will read it.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.txt"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "text"},
            ],
            "model": "model",
            "stream": False,
            "temperature": 0.25,
            "timeout": 17,
            "tools": [READ_FILE_SCHEMA],
            "reasoning_effort": "high",
        }
    ]


@pytest.mark.asyncio
async def test_openai_provider_direct_call_normalizes_sdk_failure() -> None:
    client = _FakeOpenAIClient(TimeoutError("upstream timeout"))
    provider = OpenAICompatibleProvider(
        _provider_configuration("openai-compatible"),
        client_factory=lambda **_kwargs: client,
    )

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(
            messages=[{"role": "system", "content": "System"}],
            tools=[],
            model="model",
            max_output=512,
            temperature=0.25,
            reasoning_effort=None,
            timeout=17,
        )

    assert captured.value.error.code == "provider_timeout"


@pytest.mark.asyncio
async def test_provider_requires_keyword_only_direct_arguments() -> None:
    client = _FakeOpenAIClient(object())
    provider = OpenAICompatibleProvider(
        _provider_configuration("openai-compatible"),
        client_factory=lambda **_kwargs: client,
    )
    with pytest.raises(TypeError):
        complete = cast(Any, provider.complete)
        await complete({"role": "system", "content": "System"})

    assert client.chat.completions.calls == []


@pytest.mark.asyncio
async def test_anthropic_provider_direct_call_extracts_system_and_translates_tool_history() -> None:
    client = _FakeAnthropicClient(
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Done")],
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            stop_reason="end_turn",
        )
    )
    provider = AnthropicProvider(
        _provider_configuration("anthropic"),
        client_factory=lambda **_kwargs: client,
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "I will read it.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "text"},
    ]

    await provider.complete(
        messages=messages,
        tools=[READ_FILE_SCHEMA],
        model="model",
        max_output=512,
        temperature=0.25,
        reasoning_effort=None,
        timeout=17,
    )

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
                            "id": "call-1",
                            "name": "read_file",
                            "input": {"path": "a.txt"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "text",
                        }
                    ],
                },
            ],
            "model": "model",
            "stream": False,
            "system": "System",
            "temperature": 0.25,
            "timeout": 17,
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a text file.",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_provider_direct_stream_returns_provider_usage() -> None:
    client = _FakeAnthropicClient(
        _FakeAnthropicStream(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=4, output_tokens=0)),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Done"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=2),
            ),
        )
    )
    provider = AnthropicProvider(
        _provider_configuration("anthropic"),
        client_factory=lambda **_kwargs: client,
    )

    events = [
        event
        async for event in provider.stream(
            messages=[{"role": "system", "content": "System"}],
            tools=[],
            model="model",
            max_output=512,
            temperature=0.25,
            reasoning_effort=None,
            timeout=17,
        )
    ]

    assert isinstance(events[-1], ModelCompleted)
    assert events[-1].response.usage.to_dict() == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


@pytest.mark.asyncio
async def test_anthropic_provider_direct_call_normalizes_sdk_failure() -> None:
    client = _FakeAnthropicClient(
        APIConnectionError(request=Request("POST", "https://provider.example/v1/messages"))
    )
    provider = AnthropicProvider(
        _provider_configuration("anthropic"),
        client_factory=lambda **_kwargs: client,
    )

    with pytest.raises(ModelCallError) as captured:
        await provider.complete(
            messages=[{"role": "system", "content": "System"}],
            tools=[],
            model="model",
            max_output=512,
            temperature=0.25,
            reasoning_effort=None,
            timeout=17,
        )

    assert captured.value.error.code == "provider_unavailable"
