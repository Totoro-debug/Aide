"""Scripted provider boundary for deterministic offline tests."""

from collections import deque
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from aide.config.config import ProviderConfiguration
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelAttemptGuard, ModelRouteStatus
from aide.provider.models import (
    ModelContinuation,
    ModelProvider,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
    ReasoningEffort,
    SessionModelConfiguration,
)
from aide.templates import render_template
from aide.utils.errors import ErrorInfo


def unexpected_provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
    del configuration
    raise AssertionError("Provider factory was unexpectedly called")


def error_info_fields(error: ErrorInfo) -> dict[str, object]:
    """Project ErrorInfo fields for adapter contract assertions."""
    return {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
        "retry_after_seconds": error.retry_after_seconds,
    }


def model_response_fields(response: ModelResponse) -> dict[str, object]:
    """Project ModelResponse fields for adapter contract assertions."""
    return {
        "message": response.message.to_dict(),
        "usage": response.usage.to_dict(),
        "finish_reason": response.finish_reason,
    }


@dataclass(frozen=True, slots=True)
class StreamScript:
    """Events yielded by one provider stream call."""

    events: tuple[ModelStreamEvent, ...]
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """Arguments captured from one direct keyword-only provider call."""

    messages: list[dict[str, object]]
    tools: tuple[dict[str, Any], ...]
    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout: int
    continuation: ModelContinuation | None = None


class ScriptedFakeProvider:
    """Replay provider behavior without loading an SDK or using the network."""

    def __init__(
        self,
        *,
        streams: Iterable[StreamScript] = (),
        completions: Iterable[ModelResponse | BaseException] = (),
    ) -> None:
        self._streams = deque(streams)
        self._completions = deque(completions)
        self.stream_requests: list[ProviderCall] = []
        self.unscripted_title_requests: list[ProviderCall] = []
        self.complete_requests: list[ProviderCall] = []
        self.closed = False

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str = "test-model",
        max_output: int = 1024,
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int = 30,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        call = _provider_call(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        )
        if call.messages and call.messages[0] == {
            "role": "system",
            "content": render_template("session-title-prompt.md"),
        }:
            self.unscripted_title_requests.append(call)
            raise ModelCallError(
                ErrorInfo(code="model_failed", message="No title response was scripted.")
            )
        self.stream_requests.append(call)
        if not self._streams:
            msg = "No scripted stream remains"
            raise AssertionError(msg)
        script = self._streams.popleft()
        for event in script.events:
            yield event
        if script.error is not None:
            raise script.error

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        model: str = "test-model",
        max_output: int = 1024,
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int = 30,
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        self.complete_requests.append(
            _provider_call(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
        )
        if not self._completions:
            msg = "No scripted completion remains"
            raise AssertionError(msg)
        outcome = self._completions.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


class ScriptedFakeRouter:
    """Adapt a direct provider test double to the route-only test seam."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        route_statuses: Mapping[ModelRoute, ModelRouteStatus] | None = None,
    ) -> None:
        self._provider = provider
        self.complete_calls = 0
        self._route_statuses = dict(route_statuses or {})
        self._current_call_statuses: dict[ModelRoute, ModelRouteStatus] = {}

    def stream(
        self,
        route: ModelRoute,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        status = self.call_route_status(route, continuation=continuation)
        self._current_call_statuses[route] = status
        if guard is not None:
            guard(status, messages, tools)
        return self._provider.stream(
            messages=messages,
            tools=tools,
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout=30,
            continuation=continuation,
        )

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse:
        status = self.call_route_status(route, continuation=continuation)
        self._current_call_statuses[route] = status
        if guard is not None:
            guard(status, messages, tools)
        self.complete_calls += 1
        return await self._provider.complete(
            messages=messages,
            tools=tools,
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout=30,
            continuation=continuation,
        )

    def current_call_status(self, route: ModelRoute) -> ModelRouteStatus | None:
        return self._current_call_statuses.get(route)

    def call_route_status(
        self,
        route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> ModelRouteStatus:
        del continuation, session_model_configuration, subagent_model_configuration
        return self._route_statuses.get(
            route,
            ModelRouteStatus(
                requested_route=route,
                selected_route=route,
                provider_id="test-provider",
                model="test-model",
                context_window=16_384,
                max_output=1_024,
                used_fallback=False,
            ),
        )

    async def close(self) -> None:
        await self._provider.close()


def _provider_call(
    *,
    messages: Sequence[dict[str, object]],
    tools: Sequence[dict[str, Any]],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    continuation: ModelContinuation | None = None,
) -> ProviderCall:
    return ProviderCall(
        messages=deepcopy(list(messages)),
        tools=tuple(tools),
        model=model,
        max_output=max_output,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        continuation=continuation,
    )
