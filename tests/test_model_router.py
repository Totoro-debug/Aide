import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from aide.agent.context.budget import ContextBudget
from aide.agent.runner import AgentRunner
from aide.agent.tools.tool_gateway import ModelToolCall, ToolResult
from aide.config.config import (
    MemoryConfiguration,
    ModelConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter, ModelRouteStatus
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelRoute,
    ModelUsage,
    SessionModelConfiguration,
    TextDelta,
)
from aide.utils.errors import (
    ErrorCode,
    ErrorInfo,
)
from tests.fixtures import DetachedRequestPreparer, FakeClock, ScriptedFakeProvider, StreamScript
from tests.fixtures.diagnostic_capture import capture_diagnostics

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"


async def collect(stream: AsyncIterator[object]) -> list[object]:
    return [event async for event in stream]


def configuration() -> UserConfiguration:
    provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="anthropic",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models={'default-model': ModelConfiguration(100_000, 4096, 0.2, "mid", 120)},
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="default-model",
        context_window=100_000,
        max_output=4096,
        temperature=0.2,
        reasoning_effort="mid",
        timeout=120,
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


def routed_configuration() -> UserConfiguration:
    default = configuration()
    chat_provider = ProviderConfiguration(
        provider_id="chat-provider",
        protocol="openai-compatible",
        base_url="https://chat.example/v1",
        api_key="chat-secret",
        models={'chat-model': ModelConfiguration(200_000, 8192, 0.1, "high", 90)},
    )
    chat_route = RouteConfiguration(
        provider_id=chat_provider.provider_id,
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        temperature=0.1,
        reasoning_effort="high",
        timeout=90,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={**default.models.providers, chat_provider.provider_id: chat_provider},
            routes={**default.models.routes, "title": chat_route},
        ),
    )


def multi_turn_configuration() -> UserConfiguration:
    configured = routed_configuration()
    routes = configured.models.routes
    return replace(
        configured,
        models=replace(configured.models, routes={
            **routes, "schedule": routes["title"], "subagent": routes["title"],
        }),
    )


def same_provider_routed_configuration() -> UserConfiguration:
    default = configuration()
    provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="anthropic",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models={'default-model': ModelConfiguration(100_000, 4096, 0.2, "mid", 120), 'chat-model': ModelConfiguration(200_000, 8192, 0.1, "high", 90)},
    )
    chat_route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        temperature=0.1,
        reasoning_effort="high",
        timeout=90,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={provider.provider_id: provider},
            routes={**default.models.routes, "title": chat_route},
        ),
    )


def test_model_router_status_and_budget_use_provider_model_context_window() -> None:
    base = configuration()
    provider = replace(
        base.models.providers["default-provider"],
        models={"default-model": replace(base.models.providers["default-provider"].models["default-model"], context_window=32_000)},
    )
    config = replace(
        base,
        models=replace(base.models, providers={"default-provider": provider}),
    )
    router = ModelRouter(
        configuration=config,
        provider_factory=lambda _: ScriptedFakeProvider(),
        clock=FakeClock(NOW),
        jitter=None,
    )

    status = router.route_status("title")
    budget = ContextBudget(
        context_window=status.context_window,
        max_output=status.max_output,
        compact_ratio=0.9,
    )

    assert status.context_window == 32_000
    assert budget.available_context == 27_904
    assert budget.compact_context_window == 25_114


def memory_configuration() -> UserConfiguration:
    default = configuration()
    memory_provider = ProviderConfiguration(
        provider_id="memory-provider",
        protocol="openai-compatible",
        base_url="https://memory.example/v1",
        api_key="memory-secret",
        models={'memory-model': ModelConfiguration(80_000, 2048, 0, "low", 60)},
    )
    memory_route = RouteConfiguration(
        provider_id=memory_provider.provider_id,
        model="memory-model",
        context_window=80_000,
        max_output=2048,
        temperature=0,
        reasoning_effort="low",
        timeout=60,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={**default.models.providers, memory_provider.provider_id: memory_provider},
            routes={**default.models.routes, "memory": memory_route},
        ),
    )


def request(*, route: str = "chat", stream: bool = True) -> dict[str, Any]:
    del route, stream
    return {
        "messages": [{"role": "system", "content": "You are Aide."}],
        "tools": (),
    }


def completed(content: str = "Done") -> ModelCompleted:
    return ModelCompleted(response=response(content))


def response(content: str = "Done") -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
        finish_reason="stop",
    )


def retryable_timeout(*, retry_after: float | None = None) -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="The provider timed out.",
            retryable=True,
            retry_after_seconds=retry_after,
        )
    )


def permanent_failure(
    code: ErrorCode = "provider_auth_error",
) -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code=code,
            message="The configured route is permanently unavailable.",
        )
    )


@pytest.mark.asyncio
async def test_router_close_settles_every_cached_provider_when_one_close_fails() -> None:
    close_order: list[str] = []

    class CloseTrackingProvider(ScriptedFakeProvider):
        def __init__(self, name: str, *, fail_close: bool = False) -> None:
            super().__init__(completions=(response(name),))
            self._name = name
            self._fail_close = fail_close

        async def close(self) -> None:
            close_order.append(self._name)
            if self._fail_close:
                raise RuntimeError(f"{self._name} close failed")

    providers = {
        "default-provider": CloseTrackingProvider("chat", fail_close=True),
        "memory-provider": CloseTrackingProvider("memory"),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.complete("memory", **request())

    with pytest.raises(RuntimeError, match="chat close failed"):
        await router.close()

    assert close_order == ["chat", "memory"]


@pytest.mark.asyncio
async def test_router_close_aggregates_multiple_provider_failures() -> None:
    class FailingCloseProvider(ScriptedFakeProvider):
        def __init__(self, name: str) -> None:
            super().__init__(completions=(response(name),))
            self._name = name

        async def close(self) -> None:
            raise RuntimeError(f"{self._name} close failed")

    providers = {
        "default-provider": FailingCloseProvider("chat"),
        "memory-provider": FailingCloseProvider("memory"),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.complete("memory", **request())

    with pytest.raises(BaseExceptionGroup) as raised:
        await router.close()

    assert str(raised.value) == "Model Provider shutdown failed (2 sub-exceptions)"
    assert [str(error) for error in raised.value.exceptions] == [
        "chat close failed",
        "memory close failed",
    ]


@pytest.mark.asyncio
async def test_router_close_settles_every_provider_when_one_close_is_cancelled() -> None:
    close_order: list[str] = []

    class CloseTrackingProvider(ScriptedFakeProvider):
        def __init__(self, name: str, *, cancel_close: bool = False) -> None:
            super().__init__(completions=(response(name),))
            self._name = name
            self._cancel_close = cancel_close

        async def close(self) -> None:
            close_order.append(self._name)
            if self._cancel_close:
                raise asyncio.CancelledError

    providers = {
        "default-provider": CloseTrackingProvider("chat", cancel_close=True),
        "memory-provider": CloseTrackingProvider("memory"),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.complete("memory", **request())

    with pytest.raises(asyncio.CancelledError):
        await router.close()

    assert close_order == ["chat", "memory"]


@pytest.mark.asyncio
async def test_router_starts_every_provider_close_before_waiting_for_one_to_finish() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    class FirstProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            first_started.set()
            await release_first.wait()

    class SecondProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            second_started.set()

    providers: dict[str, ScriptedFakeProvider] = {
        "default-provider": FirstProvider(completions=(response("chat"),)),
        "memory-provider": SecondProvider(completions=(response("memory"),)),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.complete("memory", **request())
    closing = asyncio.create_task(router.close())
    await first_started.wait()
    try:
        await asyncio.sleep(0)
        assert second_started.is_set()
    finally:
        release_first.set()
        await closing


@pytest.mark.asyncio
async def test_concurrent_router_close_waits_for_the_same_provider_shutdown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    close_calls = 0

    class BlockingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            started.set()
            await release.wait()

    provider = BlockingCloseProvider(completions=(response(),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())

    first = asyncio.create_task(router.close())
    await started.wait()
    second = asyncio.create_task(router.close())
    await asyncio.sleep(0)
    try:
        assert not second.done()
    finally:
        release.set()
        await asyncio.gather(first, second)

    assert close_calls == 1


@pytest.mark.asyncio
async def test_router_closes_shared_provider_instance_once() -> None:
    close_calls = 0

    class CloseTrackingProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    provider = CloseTrackingProvider(completions=(response("chat"), response("memory")))
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.complete("memory", **request())

    await router.close()

    assert close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_one_router_close_caller_does_not_cancel_provider_shutdown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class BlockingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            started.set()
            await release.wait()
            finished.set()

    provider = BlockingCloseProvider(completions=(response(),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())

    caller = asyncio.create_task(router.close())
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert not finished.is_set()
    release.set()
    await router.close()

    assert finished.is_set()


@pytest.mark.asyncio
async def test_router_cannot_create_a_new_provider_after_close() -> None:
    provider = ScriptedFakeProvider(completions=(response(),))
    factory_calls = 0

    def provider_factory(_configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
    )
    await router.complete("chat", **request())
    await router.close()

    with pytest.raises(RuntimeError, match="Model Router is closed"):
        await router.complete("chat", **request())

    assert factory_calls == 1


@pytest.mark.asyncio
async def test_model_router_caps_one_logical_stream_at_five_provider_attempts() -> None:
    failure = retryable_timeout()
    provider = ScriptedFakeProvider(
        streams=[StreamScript(events=(), error=failure) for _ in range(5)]
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    with pytest.raises(ModelCallError) as raised:
        await collect(router.stream("chat", **request()))

    assert raised.value.error is failure.error
    assert len(provider.stream_requests) == 5


@pytest.mark.asyncio
async def test_model_router_records_only_consumed_retry_attempts(
    agent_home: Path,
) -> None:
    private_failure_detail = "user-message prompt memory tool-args tool-result provider-body"
    failure = ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message=private_failure_detail,
            retryable=True,
        )
    )
    provider = ScriptedFakeProvider(
        streams=[StreamScript(events=(), error=failure) for _ in range(5)]
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID), pytest.raises(ModelCallError):
        await collect(router.stream("title", **request()))
    capture.close()

    records = [
        line for line in capture.text.splitlines() if "aide.provider.model_router:" in line
    ]
    assert len(records) == 5
    for attempt, delay, record in zip(
        range(1, 4),
        (0.5, 1.0, 2.0),
        records[:3],
        strict=True,
    ):
        assert f"attempt={attempt}/5" in record
        assert "code=provider_timeout" in record
        assert "provider=chat-provider" in record
        assert "requested_route=title" in record
        assert "selected_route=title" in record
        assert "model=chat-model" in record
        assert f"planned_delay_seconds={delay}" in record
    assert "attempt=4/5" in capture.text
    content = capture.text
    event_text = capture.event_text
    assert content.count("Traceback (most recent call last)") == 4
    assert content.count(f"ModelCallError: {private_failure_detail}") == 4
    assert private_failure_detail not in event_text


@pytest.mark.asyncio
async def test_model_router_uses_injected_clock_for_exponential_backoff_and_retry_after() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(), error=retryable_timeout(retry_after=0.25)),
            StreamScript(events=(), error=retryable_timeout(retry_after=7)),
            StreamScript(events=(), error=retryable_timeout(retry_after=100)),
            StreamScript(events=(completed(),)),
        )
    )
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=clock,
        jitter=None,
    )

    observed = await collect(router.stream("chat", **request()))

    assert observed == [completed()]
    assert clock.sleeps == [0.5, 1.0, 7, 60]


@pytest.mark.asyncio
async def test_model_router_uses_asyncio_sleep_when_clock_is_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(completed(),)),
        )
    )
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
    )

    observed = await collect(router.stream("chat", **request()))

    assert observed == [completed()]
    assert sleeps == [0.5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "permanent_code",
    ("provider_auth_error", "route_unavailable", "provider_unavailable"),
)
async def test_model_router_falls_back_after_a_permanent_requested_route_failure(
    permanent_code: ErrorCode,
) -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(), error=permanent_failure(permanent_code)),
        )
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    factory_calls: list[str] = []

    def provider_factory(provider: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(provider.provider_id)
        return providers[provider.provider_id]

    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=provider_factory,
        clock=clock,
        jitter=None,
    )

    observed = await collect(router.stream("title", **request()))

    assert observed == [completed()]
    assert factory_calls == ["chat-provider", "default-provider"]
    assert len(chat_provider.stream_requests) == 2
    assert len(default_provider.stream_requests) == 1
    fallback_request = default_provider.stream_requests[0]
    assert (
        fallback_request.model,
        fallback_request.max_output,
        fallback_request.temperature,
        fallback_request.reasoning_effort,
        fallback_request.timeout,
    ) == ("default-model", 4096, 0.2, "mid", 120)
    assert clock.sleeps == [0.5]


@pytest.mark.asyncio
async def test_model_router_does_not_replay_continuation_to_a_fallback_provider() -> None:
    continuation = ModelContinuation(provider_id="chat-provider", payload="chat-state")
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    observed = await collect(router.stream("title", **request(), continuation=continuation))

    assert observed == [completed()]
    assert chat_provider.stream_requests[0].continuation == continuation
    assert default_provider.stream_requests[0].continuation is None


@pytest.mark.asyncio
async def test_model_router_continues_the_selected_stream_route() -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload="default-state")
    fallback_response = ModelCompleted(
        response=ModelResponse(
            message=AssistantModelMessage(content="Working"),
            usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
            finish_reason="tool_calls",
            continuation=continuation,
        )
    )
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=permanent_failure()),
            StreamScript(events=(), error=permanent_failure()),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(fallback_response,)),
            StreamScript(events=(completed(),)),
        )
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await collect(router.stream("title", **request()))
    assert isinstance(first[-1], ModelCompleted)
    preview = router.call_route_status(
        "title",
        continuation=first[-1].response.continuation,
    )
    observed = await collect(
        router.stream(
            "title",
            **request(),
            continuation=first[-1].response.continuation,
        )
    )

    assert observed == [completed()]
    assert preview == router.current_call_status("title")
    assert len(chat_provider.stream_requests) == 1
    assert [call.model for call in default_provider.stream_requests] == [
        "default-model",
        "default-model",
    ]
    assert default_provider.stream_requests[1].continuation == continuation
    assert router.current_call_status("title") == ModelRouteStatus(
        requested_route="title",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=True,
    )


@pytest.mark.asyncio
async def test_model_router_does_not_activate_continuation_after_stream_fallback() -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload="default-state")
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    observed = await collect(router.stream("title", **request(), continuation=continuation))

    assert observed == [completed()]
    assert chat_provider.stream_requests[0].continuation is None
    assert default_provider.stream_requests[0].continuation is None


@pytest.mark.asyncio
async def test_model_router_replays_continuation_across_same_provider_retries() -> None:
    continuation = ModelContinuation(provider_id="chat-provider", payload="chat-state")
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(completed("Recovered"),)),
        )
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: chat_provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    observed = await collect(router.stream("title", **request(), continuation=continuation))

    assert observed == [completed("Recovered")]
    assert [call.continuation for call in chat_provider.stream_requests] == [
        continuation,
        continuation,
    ]


@pytest.mark.asyncio
async def test_model_router_continues_the_selected_completion_model() -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload="default-state")
    fallback_response = ModelResponse(
        message=AssistantModelMessage(content="Working"),
        usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
        finish_reason="tool_calls",
        continuation=continuation,
    )
    provider = ScriptedFakeProvider(
        completions=(permanent_failure(), fallback_response, response())
    )
    router = ModelRouter(
        configuration=same_provider_routed_configuration(),
        provider_factory=lambda configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await router.complete("title", **request())
    observed = await router.complete(
        "title",
        **request(),
        continuation=first.continuation,
    )

    assert observed == response()
    assert [call.model for call in provider.complete_requests] == [
        "chat-model",
        "default-model",
        "default-model",
    ]
    assert [call.continuation for call in provider.complete_requests] == [
        None,
        None,
        continuation,
    ]
    assert router.current_call_status("title") == ModelRouteStatus(
        requested_route="title",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=True,
    )


@pytest.mark.asyncio
async def test_model_router_does_not_replay_continuation_to_a_fallback_completion() -> None:
    continuation = ModelContinuation(provider_id="chat-provider", payload="chat-state")
    chat_provider = ScriptedFakeProvider(completions=(permanent_failure(),))
    default_provider = ScriptedFakeProvider(completions=(response("Fallback"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    observed = await router.complete("title", **request(), continuation=continuation)

    assert observed == response("Fallback")
    assert chat_provider.complete_requests[0].continuation == continuation
    assert default_provider.complete_requests[0].continuation is None


@pytest.mark.asyncio
async def test_model_router_does_not_activate_continuation_after_completion_fallback() -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload="default-state")
    chat_provider = ScriptedFakeProvider(completions=(permanent_failure(),))
    default_provider = ScriptedFakeProvider(completions=(response("Fallback"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    observed = await router.complete("title", **request(), continuation=continuation)

    assert observed == response("Fallback")
    assert chat_provider.complete_requests[0].continuation is None
    assert default_provider.complete_requests[0].continuation is None


@pytest.mark.asyncio
async def test_model_router_records_failed_attempt_and_default_fallback_separately(
    agent_home: Path,
) -> None:
    private_provider_body = "private provider response body"
    failure = ModelCallError(ErrorInfo(code="provider_auth_error", message=private_provider_body))
    chat_provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Recovered"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID):
        observed = await collect(router.stream("title", **request()))
    capture.close()

    assert observed == [completed("Recovered")]
    records = [
        line for line in capture.text.splitlines() if "aide.provider.model_router:" in line
    ]
    assert len(records) == 2
    assert "Provider attempt failed" in records[0]
    assert "attempt=1/5" in records[0]
    assert "code=provider_auth_error" in records[0]
    assert "provider=chat-provider" in records[0]
    assert "requested_route=title" in records[0]
    assert "selected_route=title" in records[0]
    assert "model=chat-model" in records[0]
    assert "planned_delay_seconds=0.0" in records[0]
    assert "Chat Model Route selected" in records[1]
    assert "requested_route=title" in records[1]
    assert "provider=default-provider" in records[1]
    assert "selected_route=chat" in records[1]
    assert "model=default-model" in records[1]
    assert all(" WARNING " in record for record in records)
    content = capture.text
    event_text = capture.event_text
    assert content.count("Traceback (most recent call last)") == 1
    assert content.count(f"ModelCallError: {private_provider_body}") == 1
    assert private_provider_body not in event_text


@pytest.mark.asyncio
async def test_model_router_route_status_updates_when_dynamic_fallback_is_selected() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    initial = router.route_status("title")
    assert initial == ModelRouteStatus(
        requested_route="title",
        selected_route="title",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        used_fallback=False,
    )
    assert "chat-secret" not in repr(initial)

    await collect(router.stream("title", **request()))

    fallback = router.route_status("title")
    assert fallback == ModelRouteStatus(
        requested_route="title",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=True,
    )
    assert "default-secret" not in repr(fallback)


def test_model_router_route_status_starts_from_static_default_fallback() -> None:
    def unexpected_factory(_: ProviderConfiguration) -> ScriptedFakeProvider:
        raise AssertionError("Reading route status must not construct a Provider adapter")

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=unexpected_factory,
        clock=FakeClock(NOW),
        jitter=None,
    )

    assert router.route_status("title") == ModelRouteStatus(
        requested_route="title",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=True,
    )


def test_model_router_call_route_status_has_no_side_effects_for_static_fallback() -> None:
    factory_calls: list[str] = []

    def provider_factory(provider: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(provider.provider_id)
        return ScriptedFakeProvider()

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    try:
        with capture.session(SESSION_ID):
            preview = router.call_route_status("title", continuation=None)
    finally:
        capture.close()

    assert preview == ModelRouteStatus(
        requested_route="title",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=True,
    )
    assert factory_calls == []
    assert router._providers == {}
    assert router._route_statuses == {}
    assert router._current_call_statuses.get() is None
    assert "Default Model Route selected" not in capture.event_text
    assert "Provider attempt failed" not in capture.event_text


@pytest.mark.asyncio
async def test_model_router_call_route_status_matches_configured_route_call() -> None:
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    preview = router.call_route_status("title", continuation=None)
    observed = await collect(router.stream("title", **request()))

    assert observed == [completed()]
    assert preview == ModelRouteStatus(
        requested_route="title",
        selected_route="title",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        used_fallback=False,
    )
    assert router.current_call_status("title") == preview


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous", "continuation"),
    (
        pytest.param(
            ModelRouteStatus(
                requested_route="title",
                selected_route="chat",
                provider_id="default-provider",
                model="default-model",
                context_window=100_000,
                max_output=4096,
                used_fallback=True,
            ),
            ModelContinuation(provider_id="chat-provider", payload="chat-state"),
            id="provider",
        ),
        pytest.param(
            ModelRouteStatus(
                requested_route="title",
                selected_route="memory",
                provider_id="default-provider",
                model="default-model",
                context_window=100_000,
                max_output=4096,
                used_fallback=True,
            ),
            ModelContinuation(provider_id="default-provider", payload="default-state"),
            id="configured-route",
        ),
        pytest.param(
            ModelRouteStatus(
                requested_route="title",
                selected_route="chat",
                provider_id="default-provider",
                model="retired-model",
                context_window=100_000,
                max_output=4096,
                used_fallback=True,
            ),
            ModelContinuation(provider_id="default-provider", payload="default-state"),
            id="model",
        ),
    ),
)
async def test_model_router_call_route_status_rejects_incompatible_continuation_route(
    previous: ModelRouteStatus,
    continuation: ModelContinuation,
) -> None:
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    router._current_call_statuses.set({"title": previous})

    preview = router.call_route_status("title", continuation=continuation)
    observed = await collect(router.stream("title", **request(), continuation=continuation))

    assert observed == [completed()]
    assert preview == ModelRouteStatus(
        requested_route="title",
        selected_route="title",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        used_fallback=False,
    )
    assert router.current_call_status("title") == preview


@pytest.mark.asyncio
async def test_model_router_records_static_default_fallback_without_provider_attempt(
    agent_home: Path,
) -> None:
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed("Static fallback"),)),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID):
        observed = await collect(router.stream("title", **request()))
    capture.close()

    assert observed == [completed("Static fallback")]
    records = [
        line for line in capture.text.splitlines() if "aide.provider.model_router:" in line
    ]
    assert len(records) == 1
    assert " WARNING " in records[0]
    assert "Chat Model Route selected code=route_unavailable" in records[0]
    assert "requested_route=title" in records[0]
    assert "provider=default-provider" in records[0]
    assert "selected_route=chat" in records[0]
    assert "model=default-model" in records[0]
    assert "Provider attempt failed" not in records[0]


@pytest.mark.asyncio
async def test_model_router_route_status_recovers_on_the_next_logical_stream() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=permanent_failure()),
            StreamScript(events=(completed("Chat recovered."),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Fallback response."),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await collect(router.stream("title", **request()))
    assert first == [completed("Fallback response.")]
    assert router.route_status("title").selected_route == "chat"

    preview = router.call_route_status("title", continuation=None)
    second = await collect(router.stream("title", **request()))

    assert second == [completed("Chat recovered.")]
    assert len(chat_provider.stream_requests) == 2
    assert len(default_provider.stream_requests) == 1
    assert preview == ModelRouteStatus(
        requested_route="title",
        selected_route="title",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        used_fallback=False,
    )
    assert router.current_call_status("title") == preview
    assert router.route_status("title") == preview


@pytest.mark.asyncio
async def test_model_router_complete_shares_one_budget_across_requested_and_default() -> None:
    memory_provider = ScriptedFakeProvider(
        completions=(
            retryable_timeout(),
            retryable_timeout(),
            permanent_failure(),
        )
    )
    expected = response("Summary complete.")
    default_provider = ScriptedFakeProvider(completions=(retryable_timeout(), expected))
    providers = {
        "memory-provider": memory_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    observed = await router.complete("memory", **request())

    assert observed is expected
    assert len(memory_provider.complete_requests) == 3
    assert len(default_provider.complete_requests) == 2
    fallback_request = default_provider.complete_requests[0]
    assert fallback_request.model == "default-model"
    assert clock.sleeps == [0.5, 1.0, 4.0]


@pytest.mark.asyncio
async def test_retryable_auxiliary_failure_reserves_the_fifth_attempt_for_chat() -> None:
    memory_provider = ScriptedFakeProvider(
        completions=tuple(retryable_timeout() for _ in range(4))
    )
    chat_provider = ScriptedFakeProvider(completions=(response("Chat fallback."),))
    providers = {
        "memory-provider": memory_provider,
        "default-provider": chat_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    result = await router.complete("memory", **request())

    assert result.message.content == "Chat fallback."
    assert len(memory_provider.complete_requests) == 4
    assert len(chat_provider.complete_requests) == 1
    assert clock.sleeps == [0.5, 1.0, 2.0]
    assert router.route_status("memory").selected_route == "chat"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "schedule", "subagent"))
@pytest.mark.parametrize("stream", (False, True))
async def test_multi_turn_routes_do_not_fallback_after_permanent_failure(
    route: ModelRoute, stream: bool,
) -> None:
    failure = permanent_failure()
    providers = {
        provider_id: ScriptedFakeProvider(
            streams=(StreamScript(events=(), error=failure),),
            completions=(failure,),
        )
        for provider_id in ("default-provider", "chat-provider")
    }
    router = ModelRouter(
        configuration=multi_turn_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    with pytest.raises(ModelCallError) as raised:
        if stream:
            await collect(router.stream(route, **request()))
        else:
            await router.complete(route, **request())

    assert raised.value.error.code == "provider_auth_error"
    selected_provider = "default-provider" if route == "chat" else "chat-provider"
    for provider_id, provider in providers.items():
        calls = provider.stream_requests if stream else provider.complete_requests
        assert len(calls) == (1 if provider_id == selected_provider else 0)
    status = router.current_call_status(route)
    assert status is not None
    assert status.selected_route == route and not status.used_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "schedule", "subagent"))
@pytest.mark.parametrize("stream", (False, True))
async def test_multi_turn_routes_retry_without_fallback_or_rechecking_budget(
    route: ModelRoute, stream: bool,
) -> None:
    failures = tuple(retryable_timeout() for _ in range(5))
    providers = {
        provider_id: ScriptedFakeProvider(
            streams=tuple(StreamScript(events=(), error=failure) for failure in failures),
            completions=failures,
        )
        for provider_id in ("default-provider", "chat-provider")
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=multi_turn_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )
    guarded: list[ModelRouteStatus] = []

    def guard(status: ModelRouteStatus, _messages: object, _tools: object) -> bool:
        guarded.append(status)
        return True

    with pytest.raises(ModelCallError) as raised:
        if stream:
            await collect(router.stream(route, **request(), guard=guard))
        else:
            await router.complete(route, **request(), guard=guard)

    assert raised.value.error.code == "provider_timeout"
    selected_provider = "default-provider" if route == "chat" else "chat-provider"
    for provider_id, provider in providers.items():
        calls = provider.stream_requests if stream else provider.complete_requests
        assert len(calls) == (5 if provider_id == selected_provider else 0)
    assert len(guarded) == 1
    assert guarded[0].selected_route == route
    assert clock.sleeps == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_model_router_route_status_recovers_on_the_next_logical_completion() -> None:
    memory_provider = ScriptedFakeProvider(
        completions=(permanent_failure(), response("Memory recovered."))
    )
    default_provider = ScriptedFakeProvider(completions=(response("Fallback summary."),))
    providers = {
        "memory-provider": memory_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await router.complete("memory", **request())
    assert first == response("Fallback summary.")
    assert router.route_status("memory").selected_route == "chat"

    second = await router.complete("memory", **request())

    assert second == response("Memory recovered.")
    assert len(memory_provider.complete_requests) == 2
    assert len(default_provider.complete_requests) == 1
    assert router.route_status("memory") == ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="memory-provider",
        model="memory-model",
        context_window=80_000,
        max_output=2048,
        used_fallback=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    ("model_invalid_request", "model_context_overflow", "turn_cancelled", "model_failed"),
)
async def test_model_router_does_not_retry_or_fallback_terminal_model_errors(
    code: ErrorCode,
) -> None:
    failure = ModelCallError(
        ErrorInfo(
            code=code,
            message="The model call cannot continue.",
            retryable=True,
        )
    )
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=failure),
            StreamScript(events=(completed("Unexpected retry"),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    with pytest.raises(ModelCallError) as raised:
        await collect(router.stream("title", **request()))

    assert raised.value is failure
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_never_retries_after_streaming_becomes_observable() -> None:
    partial = TextDelta(delta="Visible partial text")
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(partial,), error=retryable_timeout()),
            StreamScript(events=(completed("Duplicated response"),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )
    observed: list[object] = []

    with pytest.raises(ModelCallError):
        async for event in router.stream("title", **request()):
            observed.append(event)

    assert observed == [partial]
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_propagates_task_cancellation_without_another_attempt() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=asyncio.CancelledError()),)
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await collect(router.stream("title", **request()))

    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_closes_each_provider_adapter_it_constructed() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )
    await collect(router.stream("title", **request()))

    await router.close()

    assert chat_provider.closed is True
    assert default_provider.closed is True


@pytest.mark.asyncio
async def test_model_router_runtime_effort_updates_chat_and_default_routes() -> None:
    chat_provider = ScriptedFakeProvider(completions=(response("title"),))
    default_provider = ScriptedFakeProvider(completions=(response("chat"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )

    assert router.reasoning_effort == "mid"
    router.set_reasoning_effort("max")

    assert cast(str, router.reasoning_effort) == "max"
    await router.complete("title", **request())
    await router.complete("chat", **request())

    assert chat_provider.complete_requests[0].reasoning_effort == "high"
    assert default_provider.complete_requests[0].reasoning_effort == "max"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_is_snapshotted_when_request_is_created() -> None:
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("stream"),)),),
        completions=(response("complete"),),
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )

    router.set_reasoning_effort("high")
    stream = router.stream("title", **request())
    completion = router.complete("title", **request(stream=False))
    router.set_reasoning_effort("max")

    assert await collect(stream) == [completed("stream")]
    assert await completion == response("complete")
    assert provider.stream_requests[0].reasoning_effort == "high"
    assert provider.complete_requests[0].reasoning_effort == "high"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_keeps_explicit_routes_independent() -> None:
    base = memory_configuration()
    schedule_provider = ProviderConfiguration(
        provider_id="schedule-provider",
        protocol="anthropic",
        base_url="https://schedule.example/v1",
        api_key="schedule-secret",
        models={'schedule-model': ModelConfiguration(70_000, 1024, 0.2, "xhigh", 45)},
    )
    schedule_route = RouteConfiguration(
        provider_id=schedule_provider.provider_id,
        model="schedule-model",
        context_window=70_000,
        max_output=1024,
        temperature=0.2,
        reasoning_effort="xhigh",
        timeout=45,
    )
    configuration_with_schedule = UserConfiguration(
        runtime=base.runtime,
        memory=base.memory,
        models=ModelsConfiguration(
            providers={**base.models.providers, schedule_provider.provider_id: schedule_provider},
            routes={**base.models.routes, "schedule": schedule_route},
        ),
    )
    memory_provider = ScriptedFakeProvider(completions=(response("memory"),))
    schedule_provider_fake = ScriptedFakeProvider(completions=(response("schedule"),))
    default_provider = ScriptedFakeProvider(completions=(response("fallback"),))
    providers = {
        "memory-provider": memory_provider,
        "schedule-provider": schedule_provider_fake,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=configuration_with_schedule,
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )

    router.set_reasoning_effort("max")
    await router.complete("memory", **request())
    await router.complete("schedule", **request())
    await router.complete("chat", **request())

    assert memory_provider.complete_requests[0].reasoning_effort == "low"
    assert schedule_provider_fake.complete_requests[0].reasoning_effort == "xhigh"
    assert default_provider.complete_requests[0].reasoning_effort == "max"


@pytest.mark.asyncio
async def test_run_model_router_uses_its_session_model_without_changing_shared_routes() -> None:
    base = routed_configuration()
    alternate_provider = ProviderConfiguration(
        provider_id="alternate-chat-provider",
        protocol="anthropic",
        base_url="https://alternate.example/v1",
        api_key="alternate-secret",
        models={'chat-model': ModelConfiguration(16_384, 4096, 0.2, "mid", 120)},
    )
    configuration_with_alternate = replace(
        base,
        models=replace(
            base.models,
            providers={**base.models.providers, alternate_provider.provider_id: alternate_provider},
        ),
    )
    default_provider = ScriptedFakeProvider(completions=(response("chat"),))
    chat_provider = ScriptedFakeProvider(
        completions=(response("runtime route"), response("updated session route"))
    )
    alternate = ScriptedFakeProvider(completions=(response("session route"),))
    providers = {
        "default-provider": default_provider,
        "chat-provider": chat_provider,
        "alternate-chat-provider": alternate,
    }
    router = ModelRouter(
        configuration=configuration_with_alternate,
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    router.set_reasoning_effort("low")
    selected_configuration = SessionModelConfiguration(
        "alternate-chat-provider",
        "chat-model",
        "max",
    )
    selected_run = router.for_run(
        guard=lambda _status, _messages, _tools: None,
        session_model_configuration=selected_configuration,
    )
    selected_configuration = SessionModelConfiguration("chat-provider", "chat-model", "high")
    updated_session_run = router.for_run(
        guard=lambda _status, _messages, _tools: None,
        session_model_configuration=selected_configuration,
    )
    runtime_run = router.for_run(guard=lambda _status, _messages, _tools: None)

    selected_status = selected_run.call_route_status("chat", continuation=None)
    await selected_run.complete("chat", **request())
    await updated_session_run.complete("chat", **request())
    await runtime_run.complete("title", **request())

    assert selected_status == ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id="alternate-chat-provider",
        model="chat-model",
        context_window=16_384,
        max_output=4096,
        used_fallback=False,
    )
    assert len(alternate.complete_requests) == 1
    assert alternate.complete_requests[0].reasoning_effort == "max"
    assert [call.reasoning_effort for call in chat_provider.complete_requests] == ["high", "high"]
    assert router.route_status("title").provider_id == "chat-provider"


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", (False, True))
async def test_subagent_selection_is_bound_independently_of_the_chat_session_model(
    configured: bool,
) -> None:
    base = routed_configuration()
    routes = dict(base.models.routes)
    routes["subagent"] = routes["title"] if configured else routes["chat"]
    configuration_with_subagent = replace(base, models=replace(base.models, routes=routes))
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("SubAgent response"),)),)
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Default response"),)),)
    )
    providers = {"chat-provider": chat_provider, "default-provider": default_provider}
    router = ModelRouter(
        configuration=configuration_with_subagent,
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    selected_provider = "chat-provider" if configured else "default-provider"
    selected_model = "chat-model" if configured else "default-model"
    subagent_run = router.for_run(
        guard=lambda _status, _messages, _tools: None,
        subagent_model_configuration=SessionModelConfiguration(
            selected_provider, selected_model, "max",
        ),
    )
    chat_run = router.for_run(
        guard=lambda _status, _messages, _tools: None,
        session_model_configuration=SessionModelConfiguration(
            "chat-provider", "chat-model", "low",
        ),
    )

    status = subagent_run.call_route_status("subagent", continuation=None)
    await collect(subagent_run.stream("subagent", **request()))

    assert (status.requested_route, status.selected_route, status.provider_id) == (
        "subagent", "subagent", selected_provider,
    )
    assert status.used_fallback is False
    provider = chat_provider if configured else default_provider
    assert provider.stream_requests[0].reasoning_effort == "max"
    assert chat_run.call_route_status("chat", continuation=None).provider_id == "chat-provider"
    assert router.route_status("chat").provider_id == "default-provider"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_reaches_title_and_memory_inheriting_chat() -> None:
    provider = ScriptedFakeProvider(
        completions=(response("memory"), response("title")),
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )

    router.set_reasoning_effort("xhigh")
    await router.complete("memory", **request())
    await router.complete("title", **request())

    assert [call.reasoning_effort for call in provider.complete_requests] == ["xhigh", "xhigh"]
    assert router.route_status("memory").selected_route == "chat"
    assert router.route_status("title").selected_route == "chat"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_is_snapshotted_across_retry() -> None:
    class BlockingClock:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.sleeps: list[float] = []

        async def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.started.set()
            await self.release.wait()

    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(completed("retried"),)),
            StreamScript(events=(completed("next"),)),
        )
    )
    clock = BlockingClock()
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=clock,
        jitter=None,
    )

    pending = asyncio.create_task(collect(router.stream("title", **request())))
    await clock.started.wait()
    router.set_reasoning_effort("max")
    clock.release.set()

    assert await pending == [completed("retried")]
    await collect(router.stream("title", **request()))

    assert [call.reasoning_effort for call in provider.stream_requests] == [
        "high",
        "high",
        "high",
    ]


@pytest.mark.asyncio
async def test_model_router_runtime_effort_is_kept_across_stream_fallback() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    router.set_reasoning_effort("xhigh")
    assert await collect(router.stream("title", **request())) == [completed("fallback")]

    assert chat_provider.stream_requests[0].reasoning_effort == "high"
    assert default_provider.stream_requests[0].reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_is_snapshotted_across_complete_fallback() -> None:
    class BlockingClock:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def sleep(self, seconds: float) -> None:
            del seconds
            self.started.set()
            await self.release.wait()

    chat_provider = ScriptedFakeProvider(
        completions=(retryable_timeout(), permanent_failure()),
    )
    default_provider = ScriptedFakeProvider(completions=(response("fallback"),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = BlockingClock()
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    router.set_reasoning_effort("high")
    pending = asyncio.create_task(router.complete("title", **request(stream=False)))
    await clock.started.wait()
    router.set_reasoning_effort("max")
    clock.release.set()

    assert await pending == response("fallback")
    assert [call.reasoning_effort for call in chat_provider.complete_requests] == [
        "high",
        "high",
    ]
    assert default_provider.complete_requests[0].reasoning_effort == "high"


@pytest.mark.asyncio
async def test_model_router_runtime_effort_does_not_repeat_an_effectively_identical_fallback() -> (
    None
):
    base = configuration()
    default_route = base.models.routes["chat"]
    base = replace(base, models=replace(base.models, providers={
        "default-provider": replace(base.models.providers["default-provider"], models={
            "default-model": replace(base.models.providers["default-provider"].models["default-model"], reasoning_effort="max"),
        }),
    }))
    configuration_with_same_route = UserConfiguration(
        runtime=base.runtime,
        memory=base.memory,
        models=ModelsConfiguration(
            providers=base.models.providers,
            routes={
                **base.models.routes,
                "title": replace(default_route, reasoning_effort="max"),
            },
        ),
    )
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=permanent_failure()),))
    router = ModelRouter(
        configuration=configuration_with_same_route,
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    router.set_reasoning_effort("max")
    with pytest.raises(ModelCallError):
        await collect(router.stream("title", **request()))

    assert len(provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_agent_runner_next_tool_loop_request_reads_latest_runtime_effort() -> None:
    tool_call = ModelToolCall(id="call-1", name="work", arguments="{}")
    continuation = ModelContinuation(provider_id="default-provider", payload={"state": "opaque"})
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Working",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                            finish_reason="tool_calls",
                            continuation=continuation,
                        )
                    ),
                )
            ),
            StreamScript(events=(completed("Done"),)),
        )
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )

    class UpdatingGateway:
        def __init__(self) -> None:
            self.schemas: list[dict[str, object]] = []

        def is_micro_compression_eligible(self, tool_name: str) -> bool:
            del tool_name
            return False

        async def call(
            self,
            observed: ModelToolCall,
            *,
            confirmation: object = None,
        ) -> ToolResult:
            del confirmation
            router.set_reasoning_effort("max")
            return ToolResult(
                tool_call_id=observed.id,
                name=observed.name,
                status="success",
                content="updated",
            )

    initial_messages = [{"role": "user", "content": "Run work."}]
    result = await AgentRunner(router, DetachedRequestPreparer(initial_messages)).run(
        initial_messages,
        model="chat",
        tool_gateway=cast(Any, UpdatingGateway()),
        on_output=None,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.final_content == "Done"
    assert [call.reasoning_effort for call in provider.stream_requests] == ["mid", "max"]
    assert provider.stream_requests[1].continuation == continuation


@pytest.mark.asyncio
async def test_model_router_checks_one_stream_route_once_across_retries() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(completed("retried"),)),
        )
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    messages = [{"role": "user", "content": "Keep this exact request."}]
    tools = ({"type": "function", "function": {"name": "work"}},)
    guard_calls: list[tuple[ModelRouteStatus, object, object]] = []

    def guard(status: ModelRouteStatus, observed_messages: object, observed_tools: object) -> bool:
        guard_calls.append((status, observed_messages, observed_tools))
        return True

    observed = await collect(router.stream("title", messages=messages, tools=tools, guard=guard))

    assert observed == [completed("retried")]
    assert len(guard_calls) == 1
    assert [call[0] for call in guard_calls] == [
        ModelRouteStatus(
            requested_route="title",
            selected_route="title",
            provider_id="chat-provider",
            model="chat-model",
            context_window=200_000,
            max_output=8192,
            used_fallback=False,
        ),
    ]
    assert all(call[1] is messages for call in guard_calls)
    assert all(call[2] is tools for call in guard_calls)
    assert len(provider.stream_requests) == 2
    assert all(call.messages == messages for call in provider.stream_requests)
    assert all(call.tools == tools for call in provider.stream_requests)


@pytest.mark.asyncio
async def test_model_router_rejects_initial_complete_before_provider_call() -> None:
    provider = ScriptedFakeProvider(completions=(response(),))
    factory_calls: list[str] = []

    def provider_factory(provider_configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(provider_configuration.provider_id)
        return provider

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
    )
    guard_calls: list[tuple[ModelRouteStatus, object, object]] = []

    def guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        guard_calls.append((status, messages, tools))
        return False

    with pytest.raises(ModelCallError) as raised:
        await router.complete(
            "chat",
            messages=[{"role": "user", "content": "Too large."}],
            tools=(),
            guard=guard,
        )

    assert raised.value.error.code == "model_context_overflow"
    assert raised.value.error.retryable is False
    assert factory_calls == []
    assert provider.complete_requests == []
    assert len(guard_calls) == 1
    assert guard_calls[0][0] == ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        max_output=4096,
        used_fallback=False,
    )


@pytest.mark.asyncio
async def test_model_router_rejects_fallback_before_fallback_provider_call() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    factory_calls: list[str] = []

    def provider_factory(provider_configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(provider_configuration.provider_id)
        return providers[provider_configuration.provider_id]

    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
        jitter=None,
    )
    messages = [{"role": "user", "content": "Fallback input."}]
    tools = ({"type": "function", "function": {"name": "work"}},)
    guard_calls: list[tuple[ModelRouteStatus, object, object]] = []

    def guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        guard_calls.append((status, messages, tools))
        return status.selected_route == "title"

    with pytest.raises(ModelCallError) as raised:
        await collect(router.stream("title", messages=messages, tools=tools, guard=guard))

    assert raised.value.error.code == "model_context_overflow"
    assert factory_calls == ["chat-provider"]
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert [call[0] for call in guard_calls] == [
        ModelRouteStatus(
            requested_route="title",
            selected_route="title",
            provider_id="chat-provider",
            model="chat-model",
            context_window=200_000,
            max_output=8192,
            used_fallback=False,
        ),
        ModelRouteStatus(
            requested_route="title",
            selected_route="chat",
            provider_id="default-provider",
            model="default-model",
            context_window=100_000,
            max_output=4096,
            used_fallback=True,
        ),
    ]
    assert all(call[1] is messages for call in guard_calls)
    assert all(call[2] is tools for call in guard_calls)
    assert [call[1] for call in guard_calls[:1]] == [chat_provider.stream_requests[0].messages]
    assert [call[2] for call in guard_calls[:1]] == [chat_provider.stream_requests[0].tools]
    assert router.current_call_status("title") == guard_calls[-1][0]


@pytest.mark.asyncio
async def test_model_router_guard_preserves_same_provider_continuation() -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload={"state": "opaque"})
    provider = ScriptedFakeProvider(completions=(response("first"), response("continued")))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )
    guard_calls: list[ModelRouteStatus] = []

    def guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        guard_calls.append(status)
        return True

    assert (
        await router.complete(
            "chat",
            **request(stream=False),
            guard=guard,
        )
    ) == response("first")
    observed = await router.complete(
        "chat",
        **request(stream=False),
        continuation=continuation,
        guard=guard,
    )

    assert observed == response("continued")
    assert len(guard_calls) == 2
    assert all(status == guard_calls[0] for status in guard_calls)
    assert [call.continuation for call in provider.complete_requests] == [None, continuation]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("complete", "stream"))
async def test_run_model_router_guard_blocks_initial_attempt_before_provider(stream: bool) -> None:
    provider = ScriptedFakeProvider(
        completions=(response(),), streams=(StreamScript(events=(completed(),)),)
    )
    factory_calls: list[str] = []

    def provider_factory(configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(configuration.provider_id)
        return provider

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
    )
    observed_statuses: list[ModelRouteStatus] = []

    def reject(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        observed_statuses.append(status)
        return False

    run = router.for_run(guard=reject)

    with pytest.raises(ModelCallError) as raised:
        if stream:
            await collect(run.stream("title", **request()))
        else:
            await run.complete("chat", **request())

    assert raised.value.error.code == "model_context_overflow"
    assert factory_calls == []
    assert provider.stream_requests == []
    assert provider.complete_requests == []
    assert len(observed_statuses) == 1
    assert run.current_call_status("title" if stream else "chat") == observed_statuses[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("complete", "stream"))
async def test_run_model_router_guard_blocks_smaller_fallback_before_provider(
    stream: bool,
) -> None:
    chat_provider = ScriptedFakeProvider(
        completions=(permanent_failure(),),
        streams=(StreamScript(events=(), error=permanent_failure()),),
    )
    default_provider = ScriptedFakeProvider(
        completions=(response(),), streams=(StreamScript(events=(completed(),)),)
    )
    providers = {"chat-provider": chat_provider, "default-provider": default_provider}
    factory_calls: list[str] = []

    def provider_factory(configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(configuration.provider_id)
        return providers[configuration.provider_id]

    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
        jitter=None,
    )
    observed_statuses: list[ModelRouteStatus] = []

    def guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        observed_statuses.append(status)
        return status.selected_route == "title"

    run = router.for_run(guard=guard)

    with pytest.raises(ModelCallError) as raised:
        if stream:
            await collect(run.stream("title", **request()))
        else:
            await run.complete("title", **request())

    assert raised.value.error.code == "model_context_overflow"
    assert factory_calls == ["chat-provider"]
    assert len(chat_provider.stream_requests if stream else chat_provider.complete_requests) == 1
    assert default_provider.stream_requests == []
    assert default_provider.complete_requests == []
    assert [status.selected_route for status in observed_statuses] == ["title", "chat"]
    assert run.current_call_status("title") == observed_statuses[-1]


@pytest.mark.asyncio
async def test_run_model_router_keeps_final_status_within_its_run_and_preview_is_read_only() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=permanent_failure()),
            StreamScript(events=(completed("Recovered"),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Fallback"),)),)
    )
    providers = {"chat-provider": chat_provider, "default-provider": default_provider}
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    first_run = router.for_run(guard=lambda status, messages, tools: True)
    second_run = router.for_run(guard=lambda status, messages, tools: True)

    assert first_run.current_call_status("title") is None
    assert second_run.current_call_status("title") is None
    assert await collect(first_run.stream("title", **request())) == [completed("Fallback")]
    fallback_status = first_run.current_call_status("title")
    assert fallback_status is not None
    assert fallback_status.selected_route == "chat"
    assert second_run.current_call_status("title") is None

    route_statuses_before = router._route_statuses.copy()
    preview = first_run.call_route_status("title", continuation=None)
    assert preview.selected_route == "title"
    assert first_run.current_call_status("title") == fallback_status
    assert router.current_call_status("title") == fallback_status
    assert router._route_statuses == route_statuses_before

    assert await collect(second_run.stream("title", **request())) == [completed("Recovered")]
    assert second_run.current_call_status("title") == preview
    assert first_run.current_call_status("title") == fallback_status


@pytest.mark.asyncio
async def test_run_model_router_explicit_guard_overrides_bound_guard() -> None:
    provider = ScriptedFakeProvider(completions=(response("Allowed"),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )
    bound_calls: list[ModelRouteStatus] = []
    override_calls: list[ModelRouteStatus] = []

    def bound_guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        bound_calls.append(status)
        return False

    def override_guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        override_calls.append(status)
        return True

    run = router.for_run(guard=bound_guard)
    assert await run.complete("chat", **request(), guard=override_guard) == response("Allowed")
    assert bound_calls == []
    assert len(override_calls) == 1
    assert len(provider.complete_requests) == 1
    assert run.current_call_status("chat") == override_calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_selection", [False, True])
async def test_run_model_router_preserves_retry_and_continuation(explicit_selection: bool) -> None:
    continuation = ModelContinuation(provider_id="default-provider", payload="opaque-state")
    provider = ScriptedFakeProvider(
        completions=(retryable_timeout(), response("Retried"), response("Continued"))
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    guarded_attempts: list[ModelRouteStatus] = []

    def guard(status: ModelRouteStatus, messages: object, tools: object) -> bool:
        del messages, tools
        guarded_attempts.append(status)
        return True

    run = router.for_run(
        guard=guard,
        session_model_configuration=(
            SessionModelConfiguration("default-provider", "default-model", "high")
            if explicit_selection
            else None
        ),
    )
    route: ModelRoute = "chat"
    assert await run.complete(route, **request()) == response("Retried")
    assert await run.complete(route, **request(), continuation=continuation) == response(
        "Continued"
    )
    assert [call.continuation for call in provider.complete_requests] == [
        None,
        None,
        continuation,
    ]
    assert len(guarded_attempts) == 2
    assert run.current_call_status(route) == guarded_attempts[-1]
    if explicit_selection:
        assert [call.reasoning_effort for call in provider.complete_requests] == ["high"] * 3


@pytest.mark.asyncio
async def test_run_model_router_records_status_when_stream_is_cancelled() -> None:
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(TextDelta(delta="Partial"), completed())),)
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
    )
    run = router.for_run(guard=lambda status, messages, tools: True)
    stream = cast(AsyncGenerator[object, None], run.stream("title", **request()))

    assert await anext(stream) == TextDelta(delta="Partial")
    assert run.current_call_status("title") is None
    with pytest.raises(asyncio.CancelledError):
        await stream.athrow(asyncio.CancelledError())

    assert run.current_call_status("title") == router.current_call_status("title")
