"""Test-only adapters for Task Framing requests."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any, cast

from aide.agent.runner import AgentRunnerRoute, AgentRunnerRouter
from aide.config.config import ConfigError, UserConfiguration
from aide.provider.model_router import ModelAttemptGuard, ModelRouteStatus
from aide.provider.models import (
    ModelContinuation,
    ModelMessages,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
    SessionModelConfiguration,
)


class TaskFramingRouterAdapter:
    """Script Task Framing completions while delegating every other Router call."""

    def __init__(
        self,
        delegate: AgentRunnerRouter,
        outcomes: Sequence[ModelResponse | BaseException] | None = (),
    ) -> None:
        self._delegate = delegate
        self._outcomes = None if outcomes is None else deque(outcomes)
        self.framing_requests: list[ModelMessages] = []
        self._configured_statuses: dict[ModelRoute, ModelRouteStatus] = {}
        self._unavailable_routes: dict[ModelRoute, ConfigError] = {}
        self._last_statuses: dict[ModelRoute, ModelRouteStatus] = {}

    def bind_configuration(self, configuration: UserConfiguration) -> None:
        for route in cast(tuple[ModelRoute, ...], ("chat", "title", "schedule", "memory", "subagent")):
            try:
                resolved = configuration.resolve_route(route)
            except ConfigError as error:
                if error.error.code != "route_unavailable":
                    raise
                self._unavailable_routes[route] = error
                continue
            self._configured_statuses[route] = ModelRouteStatus(
                requested_route=route,
                selected_route=cast(ModelRoute, resolved.selected_route),
                provider_id=resolved.provider.provider_id,
                model=resolved.route.model,
                context_window=resolved.route.context_window,
                max_output=resolved.route.max_output,
                used_fallback=resolved.used_fallback,
            )

    def stream(
        self,
        route: AgentRunnerRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if _accepts_guard(self._delegate.stream):
            return cast(
                AsyncIterator[ModelStreamEvent],
                cast(Any, self._delegate).stream(
                    route,
                    messages=messages,
                    tools=tools,
                    continuation=continuation,
                    guard=guard,
                    **(
                        {"session_model_configuration": session_model_configuration}
                        if session_model_configuration is not None
                        else {}
                    ),
                ),
            )
        self._check_fake_attempt(route, messages=messages, tools=tools, guard=guard)
        return self._delegate.stream(
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse:
        if self._outcomes is not None and _is_task_framing_request(
            route, messages=messages, tools=tools
        ):
            self.framing_requests.append(deepcopy(messages))
            return await self._complete_task_framing()
        if _accepts_guard(self._delegate.complete):
            return cast(
                ModelResponse,
                await cast(Any, self._delegate).complete(
                    route,
                    messages=messages,
                    tools=tools,
                    continuation=continuation,
                    guard=guard,
                    **(
                        {"session_model_configuration": session_model_configuration}
                        if session_model_configuration is not None
                        else {}
                    ),
                ),
            )
        self._check_fake_attempt(route, messages=messages, tools=tools, guard=guard)
        return await self._delegate.complete(
            route,  # type: ignore[arg-type]
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    def current_call_status(self, route: ModelRoute) -> ModelRouteStatus | None:
        current_call_status = getattr(self._delegate, "current_call_status", None)
        if callable(current_call_status):
            status = current_call_status(route)
            if isinstance(status, ModelRouteStatus):
                return status
        return self._last_statuses.get(route)

    def call_route_status(
        self,
        route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
        session_model_configuration: SessionModelConfiguration | None = None,
    ) -> ModelRouteStatus:
        call_route_status = getattr(self._delegate, "call_route_status", None)
        if callable(call_route_status):
            status = call_route_status(
                route,
                continuation=continuation,
                **(
                    {"session_model_configuration": session_model_configuration}
                    if session_model_configuration is not None
                    else {}
                ),
            )
            if isinstance(status, ModelRouteStatus):
                return status
        status = self._configured_statuses.get(route)
        if status is None:
            if route in self._unavailable_routes:
                raise self._unavailable_routes[route]
            raise AssertionError("test Router requires a bound configuration")
        return status

    def _check_fake_attempt(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None,
    ) -> None:
        status = self._configured_statuses.get(route)
        if status is None:
            if route in self._unavailable_routes:
                raise self._unavailable_routes[route]
            raise AssertionError("test Router requires a bound configuration")
        self._last_statuses[route] = status
        if guard is not None:
            guard(status, messages, tools)

    async def _complete_task_framing(self) -> ModelResponse:
        if not self._outcomes:
            raise RuntimeError("Task Framing response is not scripted for this test")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class BlockingTaskFramingRouterAdapter(TaskFramingRouterAdapter):
    """Block a Task Framing completion until the caller cancels it."""

    def __init__(self, delegate: AgentRunnerRouter) -> None:
        super().__init__(delegate)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _complete_task_framing(self) -> ModelResponse:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def _is_task_framing_request(
    route: ModelRoute,
    *,
    messages: ModelMessages,
    tools: Sequence[dict[str, Any]],
) -> bool:
    return (
        route == "chat" and not tools and len(messages) == 1 and messages[0].get("role") == "system"
    )


def _accepts_guard(method: object) -> bool:
    return "guard" in inspect.signature(cast(Any, method)).parameters
