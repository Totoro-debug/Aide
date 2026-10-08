"""Model Route resolution and one shared Provider attempt budget."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

from loguru import logger

from aide.config.config import ProviderConfiguration, ResolvedModelRoute, UserConfiguration
from aide.provider.errors import ModelCallError, model_context_overflow_error
from aide.provider.models import (
    REASONING_EFFORT_LEVELS,
    ModelContinuation,
    ModelMessages,
    ModelProvider,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
    ReasoningEffort,
)
from aide.provider.session_configuration import SessionModelConfiguration

_MAX_ATTEMPTS = 5
_RETRYABLE_CODES = frozenset({"provider_rate_limited", "provider_timeout", "provider_unavailable"})
_FALLBACK_CODES = frozenset({"route_unavailable", "provider_auth_error"})


class RetryClock(Protocol):
    async def sleep(self, seconds: float) -> None: ...


type ProviderImplementation = ModelProvider
type ProviderFactory = Callable[[ProviderConfiguration], ProviderImplementation]
type Jitter = Callable[[float], float]


class _ProviderPool:
    """Share SDK clients across configuration versions while each Router owns a lease."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str, str], tuple[ModelProvider, int]] = {}

    @staticmethod
    def key(configuration: ProviderConfiguration) -> tuple[str, str, str, str]:
        return (
            configuration.provider_id, configuration.protocol,
            configuration.base_url, configuration.api_key,
        )

    def acquire(
        self, configuration: ProviderConfiguration, factory: ProviderFactory,
    ) -> ModelProvider:
        key = self.key(configuration)
        provider, users = self._entries.get(key, (None, 0))
        if provider is None:
            provider = factory(configuration)
        self._entries[key] = (provider, users + 1)
        return provider

    def release(self, configuration: ProviderConfiguration) -> ModelProvider | None:
        key = self.key(configuration)
        provider, users = self._entries[key]
        if users > 1:
            self._entries[key] = (provider, users - 1)
            return None
        del self._entries[key]
        return provider


@dataclass(frozen=True, slots=True)
class ModelRouteStatus:
    """Secret-free snapshot of the route selected for one logical purpose."""

    requested_route: ModelRoute
    selected_route: ModelRoute
    provider_id: str
    model: str
    context_window: int
    max_output: int
    used_fallback: bool


type ModelAttemptGuard = Callable[
    [ModelRouteStatus, ModelMessages, Sequence[dict[str, Any]]], bool | None
]


class ModelRouterDelegate(Protocol):
    """Router methods needed by a bound Agent Run."""

    def stream(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> Coroutine[Any, Any, ModelResponse]: ...

    def current_call_status(self, route: ModelRoute) -> ModelRouteStatus | None: ...

    def call_route_status(
        self,
        route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> ModelRouteStatus: ...


class RunModelRouter:
    """Bind one guard and final route-status snapshot to an Agent Run."""

    def __init__(
        self,
        router: ModelRouterDelegate,
        *,
        guard: ModelAttemptGuard,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> None:
        self._router = router
        self._guard = guard
        self._session_model_configuration = session_model_configuration
        self._subagent_model_configuration = subagent_model_configuration
        self._call_statuses: dict[ModelRoute, ModelRouteStatus] = {}

    def stream(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        call_guard = self._guard if guard is None else guard
        events = self._router.stream(
            route, messages=messages, tools=tools, continuation=continuation,
            guard=call_guard, **self._selection_kwargs(),
        )

        async def observe() -> AsyncIterator[ModelStreamEvent]:
            try:
                async for event in events:
                    yield event
            finally:
                self._remember_call_status(route)

        return observe()

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse:
        try:
            call_guard = self._guard if guard is None else guard
            return await self._router.complete(
                route, messages=messages, tools=tools, continuation=continuation,
                guard=call_guard, **self._selection_kwargs(),
            )
        finally:
            self._remember_call_status(route)

    def current_call_status(self, route: ModelRoute) -> ModelRouteStatus | None:
        return self._call_statuses.get(route)

    def call_route_status(
        self,
        route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
    ) -> ModelRouteStatus:
        return self._router.call_route_status(
            route, continuation=continuation, **self._selection_kwargs(),
        )

    def _selection_kwargs(self) -> dict[str, Any]:
        if self._session_model_configuration is not None:
            return {"session_model_configuration": self._session_model_configuration}
        if self._subagent_model_configuration is not None:
            return {"subagent_model_configuration": self._subagent_model_configuration}
        return {}

    def _remember_call_status(self, route: ModelRoute) -> None:
        status = self._router.current_call_status(route)
        if status is not None:
            self._call_statuses[route] = status


class ModelRouter:
    """Resolve a logical Model Route and coordinate Provider attempts."""

    def __init__(
        self,
        *,
        configuration: UserConfiguration,
        provider_factory: ProviderFactory,
        clock: RetryClock | None = None,
        jitter: Jitter | None = None,
    ) -> None:
        self._configuration = configuration
        self._provider_factory = provider_factory
        self._clock = clock
        self._jitter = jitter
        self._providers: dict[str, ProviderImplementation] = {}
        self._provider_pool = _ProviderPool()
        self._provider_configurations: dict[str, ProviderConfiguration] = {}
        self._route_statuses: dict[ModelRoute, ModelRouteStatus] = {}
        self._current_call_statuses: ContextVar[dict[ModelRoute, ModelRouteStatus] | None] = (
            ContextVar("aide_model_router_call_statuses", default=None)
        )
        self._reasoning_effort_override: ReasoningEffort | None = None
        self._close_task: asyncio.Task[None] | None = None

    def fork(self, configuration: UserConfiguration) -> "ModelRouter":
        """Freeze new routes while retaining unchanged, already-created Provider clients."""
        router = ModelRouter(
            configuration=configuration, provider_factory=self._provider_factory,
            clock=self._clock, jitter=self._jitter,
        )
        router._provider_pool = self._provider_pool
        router._reasoning_effort_override = self._reasoning_effort_override
        for provider_id, previous in self._provider_configurations.items():
            selected = configuration.models.providers.get(provider_id)
            if selected is not None and self._provider_pool.key(selected) == self._provider_pool.key(previous):
                router._provider(selected)
        return router

    def for_run(
        self,
        *,
        guard: ModelAttemptGuard,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> RunModelRouter:
        """Bind a per-attempt guard and final route status to an Agent Run."""
        return RunModelRouter(
            self,
            guard=guard,
            session_model_configuration=session_model_configuration,
            subagent_model_configuration=subagent_model_configuration,
        )

    def route_status(self, requested_route: ModelRoute) -> ModelRouteStatus:
        """Return the current concrete route identity without provider credentials."""
        status = self._route_statuses.get(requested_route)
        if status is None:
            status = _route_status(
                requested_route,
                self._configuration.resolve_route(requested_route),
            )
            self._route_statuses[requested_route] = status
        return status

    def current_call_status(self, requested_route: ModelRoute) -> ModelRouteStatus | None:
        """Return the route selected by the current task's latest logical call."""
        statuses = self._current_call_statuses.get()
        return None if statuses is None else statuses.get(requested_route)

    def call_route_status(
        self,
        requested_route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> ModelRouteStatus:
        """Preview the initial route for one logical call without publishing state."""
        return _route_status(
            requested_route,
            self._resolve_call_route(
                requested_route,
                continuation,
                session_model_configuration=session_model_configuration,
                subagent_model_configuration=subagent_model_configuration,
            ),
        )

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """Return the effective chat Reasoning Effort for this Runtime Lifetime."""
        override = self._reasoning_effort_override
        if override is not None:
            return override
        resolved = self._configuration.resolve_route("chat")
        return resolved.route.reasoning_effort

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Publish the Runtime-Lifetime Reasoning Effort override."""
        if effort not in REASONING_EFFORT_LEVELS:
            raise ValueError(f"Unsupported Reasoning Effort: {effort}")
        self._reasoning_effort_override = effort

    def stream(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        resolved, reasoning_effort = self._begin_call(
            route,
            continuation=continuation,
            session_model_configuration=session_model_configuration,
            subagent_model_configuration=subagent_model_configuration,
        )
        return self._stream_direct(
            resolved,
            messages=messages,
            tools=tools,
            continuation=continuation,
            reasoning_effort=reasoning_effort,
            publish_route_status=(
                session_model_configuration is None and subagent_model_configuration is None
            ),
            guard=guard,
        )

    async def _stream_direct(
        self,
        resolved: ResolvedModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        reasoning_effort: ReasoningEffort | None,
        publish_route_status: bool,
        guard: ModelAttemptGuard | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._check_attempt_guard(resolved, messages=messages, tools=tools, guard=guard)
            provider = self._provider(resolved.provider)
            emitted = False
            try:
                events = provider.stream(
                    messages=messages,
                    tools=tools,
                    model=resolved.route.model,
                    max_output=resolved.route.max_output,
                    temperature=resolved.route.temperature,
                    reasoning_effort=resolved.route.reasoning_effort,
                    timeout=resolved.route.timeout,
                    **(
                        {"continuation": continuation}
                        if continuation is not None
                        and continuation.provider_id == resolved.provider.provider_id
                        else {}
                    ),
                )
                async for event in events:
                    emitted = True
                    yield event
                return
            except ModelCallError as failure:
                if emitted:
                    raise
                resolved = await self._recover_attempt(
                    resolved,
                    failure,
                    attempt=attempt,
                    reasoning_effort=reasoning_effort,
                    publish_route_status=publish_route_status,
                )

    def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
        guard: ModelAttemptGuard | None = None,
    ) -> Coroutine[Any, Any, ModelResponse]:
        resolved, reasoning_effort = self._begin_call(
            route,
            continuation=continuation,
            session_model_configuration=session_model_configuration,
            subagent_model_configuration=subagent_model_configuration,
        )
        return self._complete_direct(
            resolved,
            messages=messages,
            tools=tools,
            continuation=continuation,
            reasoning_effort=reasoning_effort,
            publish_route_status=(
                session_model_configuration is None and subagent_model_configuration is None
            ),
            guard=guard,
        )

    async def _complete_direct(
        self,
        resolved: ResolvedModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        reasoning_effort: ReasoningEffort | None,
        publish_route_status: bool,
        guard: ModelAttemptGuard | None,
    ) -> ModelResponse:
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._check_attempt_guard(resolved, messages=messages, tools=tools, guard=guard)
            provider = self._provider(resolved.provider)
            try:
                return await provider.complete(
                    messages=messages,
                    tools=tools,
                    model=resolved.route.model,
                    max_output=resolved.route.max_output,
                    temperature=resolved.route.temperature,
                    reasoning_effort=resolved.route.reasoning_effort,
                    timeout=resolved.route.timeout,
                    **(
                        {"continuation": continuation}
                        if continuation is not None
                        and continuation.provider_id == resolved.provider.provider_id
                        else {}
                    ),
                )
            except ModelCallError as failure:
                resolved = await self._recover_attempt(
                    resolved,
                    failure,
                    attempt=attempt,
                    reasoning_effort=reasoning_effort,
                    publish_route_status=publish_route_status,
                )

        raise AssertionError("Provider attempt budget exhausted without a terminal result")

    def _check_attempt_guard(
        self,
        resolved: ResolvedModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None,
    ) -> None:
        if guard is None:
            return
        status = _route_status(cast(ModelRoute, resolved.requested_route), resolved)
        if guard(status, messages, tools) is False:
            raise model_context_overflow_error()

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_providers())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_providers(self) -> None:
        providers = tuple(
            provider for configuration in self._provider_configurations.values()
            if (provider := self._provider_pool.release(configuration)) is not None
        )
        self._providers.clear()
        self._provider_configurations.clear()
        unique = _unique_providers(providers)
        results = await asyncio.gather(
            *(provider.close() for provider in unique),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Model Provider shutdown failed", failures)

    async def _recover_attempt(
        self,
        current: ResolvedModelRoute,
        failure: ModelCallError,
        *,
        attempt: int,
        reasoning_effort: ReasoningEffort | None,
        publish_route_status: bool,
    ) -> ResolvedModelRoute:
        code = failure.error.code
        if failure.error.retryable and code in _RETRYABLE_CODES:
            if current.selected_route != "chat" and attempt == _MAX_ATTEMPTS - 1:
                return self._fallback_to_chat(
                    current, failure, attempt=attempt,
                    reasoning_effort=reasoning_effort,
                    publish_route_status=publish_route_status,
                )
            if attempt == _MAX_ATTEMPTS:
                raise failure
            backoff = min(30.0, 0.5 * 2.0 ** (attempt - 1))
            if self._jitter is not None:
                backoff += min(backoff, max(0.0, self._jitter(backoff)))
            retry_after = float(failure.error.retry_after_seconds or 0.0)
            delay = min(60.0, max(backoff, retry_after))
            _log_retry(current, failure, attempt=attempt, delay=delay)
            if self._clock is None:
                await asyncio.sleep(delay)
            else:
                await self._clock.sleep(delay)
            return current
        allows_fallback = code in _FALLBACK_CODES or (
            code == "provider_unavailable" and not failure.error.retryable
        )
        if current.selected_route == "chat" or not allows_fallback or attempt == _MAX_ATTEMPTS:
            raise failure
        return self._fallback_to_chat(
            current, failure, attempt=attempt,
            reasoning_effort=reasoning_effort,
            publish_route_status=publish_route_status,
        )

    def _fallback_to_chat(
        self,
        current: ResolvedModelRoute,
        failure: ModelCallError,
        *,
        attempt: int,
        reasoning_effort: ReasoningEffort | None,
        publish_route_status: bool,
    ) -> ResolvedModelRoute:
        fallback = self._configuration.resolve_route("chat")
        requested_route = cast(ModelRoute, current.requested_route)
        fallback_route = fallback.route
        if reasoning_effort is not None:
            fallback_route = replace(fallback_route, reasoning_effort=reasoning_effort)
        if fallback.provider == current.provider and fallback_route == current.route:
            raise failure
        fallback = replace(
            fallback,
            requested_route=requested_route,
            used_fallback=True,
            route=fallback_route,
        )
        status = _route_status(requested_route, fallback)
        if publish_route_status:
            self._route_statuses[requested_route] = status
        self._remember_current_call_status(requested_route, status)
        _log_fallback(current, fallback, failure, attempt=attempt)
        return fallback

    def _provider(self, configuration: ProviderConfiguration) -> ProviderImplementation:
        if self._close_task is not None:
            raise RuntimeError("Model Router is closed")
        provider = self._providers.get(configuration.provider_id)
        if provider is None:
            provider = self._provider_pool.acquire(configuration, self._provider_factory)
            self._providers[configuration.provider_id] = provider
            self._provider_configurations[configuration.provider_id] = configuration
        return provider

    def _begin_call(
        self,
        requested_route: ModelRoute,
        *,
        continuation: ModelContinuation | None,
        session_model_configuration: SessionModelConfiguration | None,
        subagent_model_configuration: SessionModelConfiguration | None,
    ) -> tuple[ResolvedModelRoute, ReasoningEffort | None]:
        session_chat_selection = session_model_configuration if requested_route == "chat" else None
        reasoning_effort = (
            session_chat_selection.reasoning_effort
            if session_chat_selection is not None
            else self._reasoning_effort_override
        )
        resolved = self._resolve_call_route(
            requested_route,
            continuation,
            session_model_configuration=session_chat_selection,
            subagent_model_configuration=subagent_model_configuration,
        )
        if subagent_model_configuration is not None:
            reasoning_effort = None
        if reasoning_effort is not None and resolved.selected_route == "chat":
            resolved = replace(
                resolved,
                route=replace(resolved.route, reasoning_effort=reasoning_effort),
            )
        status = _route_status(requested_route, resolved)
        if session_chat_selection is None and subagent_model_configuration is None:
            self._route_statuses[requested_route] = status
        self._remember_current_call_status(requested_route, status)
        if resolved.used_fallback:
            logger.warning(
                "Chat Model Route selected code=route_unavailable requested_route={} "
                "provider={} selected_route={} model={}",
                resolved.requested_route,
                resolved.provider.provider_id,
                resolved.selected_route,
                resolved.route.model,
            )
        return resolved, reasoning_effort

    def _resolve_call_route(
        self,
        requested_route: ModelRoute,
        continuation: ModelContinuation | None,
        *,
        session_model_configuration: SessionModelConfiguration | None,
        subagent_model_configuration: SessionModelConfiguration | None,
    ) -> ResolvedModelRoute:
        if continuation is not None:
            previous = self.current_call_status(requested_route)
            if previous is not None and previous.provider_id == continuation.provider_id:
                selected = (
                    session_model_configuration if requested_route == "chat"
                    else subagent_model_configuration if requested_route == "subagent" else None
                )
                if selected is not None and previous.selected_route == requested_route:
                    resolved = self._configuration.resolve_session_model_route(
                        selected.provider_id, selected.model, selected.reasoning_effort,
                        requested_route=cast(Literal["chat", "subagent"], requested_route),
                    )
                else:
                    resolved = self._configuration.resolve_route(previous.selected_route)
                if (
                    resolved.selected_route == previous.selected_route
                    and resolved.provider.provider_id == previous.provider_id
                    and resolved.route.model == previous.model
                ):
                    return replace(
                        resolved,
                        requested_route=requested_route,
                        used_fallback=previous.selected_route != requested_route,
                    )
        if session_model_configuration is not None and requested_route == "chat":
            return self._configuration.resolve_session_model_route(
                session_model_configuration.provider_id,
                session_model_configuration.model,
                session_model_configuration.reasoning_effort,
            )
        if subagent_model_configuration is not None and requested_route == "subagent":
            resolved = self._configuration.resolve_session_model_route(
                subagent_model_configuration.provider_id,
                subagent_model_configuration.model,
                subagent_model_configuration.reasoning_effort,
                requested_route="subagent",
            )
            configured = self._configuration.resolve_route("subagent")
            if (
                configured.used_fallback
                and configured.provider.provider_id == resolved.provider.provider_id
                and configured.route.model == resolved.route.model
            ):
                return replace(resolved, selected_route="chat", used_fallback=True)
            return resolved
        return self._configuration.resolve_route(requested_route)

    def _remember_current_call_status(
        self,
        requested_route: ModelRoute,
        status: ModelRouteStatus,
    ) -> None:
        current = self._current_call_statuses.get()
        statuses = {} if current is None else dict(current)
        statuses[requested_route] = status
        self._current_call_statuses.set(statuses)


def _unique_providers(
    providers: tuple[ProviderImplementation, ...],
) -> tuple[ProviderImplementation, ...]:
    seen: set[int] = set()
    unique: list[ProviderImplementation] = []
    for provider in providers:
        identity = id(provider)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(provider)
    return tuple(unique)


def _log_retry(
    resolved: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
    delay: float,
) -> None:
    logger.opt(exception=failure).warning(
        "Provider attempt failed; retrying attempt={}/{} code={} provider={} "
        "requested_route={} selected_route={} model={} planned_delay_seconds={}",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        resolved.provider.provider_id,
        resolved.requested_route,
        resolved.selected_route,
        resolved.route.model,
        delay,
    )


def _log_fallback(
    failed: ResolvedModelRoute,
    fallback: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
) -> None:
    logger.opt(exception=failure).warning(
        "Provider attempt failed; recovering attempt={}/{} code={} provider={} "
        "requested_route={} selected_route={} model={} planned_delay_seconds=0.0",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        failed.provider.provider_id,
        failed.requested_route,
        failed.selected_route,
        failed.route.model,
    )
    logger.warning(
        "Chat Model Route selected code={} requested_route={} provider={} "
        "selected_route={} model={}",
        failure.error.code,
        fallback.requested_route,
        fallback.provider.provider_id,
        fallback.selected_route,
        fallback.route.model,
    )


def _route_status(
    requested_route: ModelRoute,
    resolved: ResolvedModelRoute,
) -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route=requested_route,
        selected_route=cast(ModelRoute, resolved.selected_route),
        provider_id=resolved.provider.provider_id,
        model=resolved.route.model,
        context_window=resolved.route.context_window,
        max_output=resolved.route.max_output,
        used_fallback=resolved.used_fallback,
    )
