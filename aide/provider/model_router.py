"""Model Route resolution and one shared Provider attempt budget."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Protocol, cast, runtime_checkable

from loguru import logger

from aide.config.config import ProviderConfiguration, ResolvedModelRoute, UserConfiguration
from aide.provider.errors import ModelCallError, model_context_overflow_error
from aide.provider.models import (
    REASONING_EFFORT_LEVELS,
    ModelCompleted,
    ModelContinuation,
    ModelMessages,
    ModelProvider,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
    ReasoningEffort,
)
from aide.provider.session_configuration import SessionModelConfiguration
from aide.trace.scope import (
    TraceExecutionScope,
    TraceProviderAttempt,
    current_trace_scope,
    set_trace_parent_record_id,
    trace_provider_attempt,
)
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureState,
    TraceErrorCategory,
    TraceKind,
    TraceOutcome,
    TraceReasoningEffort,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from aide.utils.errors import ErrorInfo

_MAX_ATTEMPTS = 5
_RETRYABLE_CODES = frozenset({"provider_rate_limited", "provider_timeout", "provider_unavailable"})
_FALLBACK_CODES = frozenset({"route_unavailable", "provider_auth_error"})
_FALLBACK_ROUTES = frozenset({"title", "memory"})


class _TraceModelCall:
    """Best-effort capture for one logical call and its selected provider attempts."""

    def __init__(
        self,
        scope: TraceExecutionScope,
        resolved: ResolvedModelRoute | None,
        started_at: datetime,
    ) -> None:
        self.scope = scope
        self.handle: TraceCaptureHandle | None = None
        self.started_at = started_at
        self.started_monotonic = monotonic()
        self.attempt_count = 0
        self.attempt_started: float | None = None
        self.final_attempt_duration_ms: int | None = None
        self.active_provider_attempt: TraceProviderAttempt | None = None
        self.provider_capture_incomplete = False
        self.failures: list[tuple[TraceErrorCategory, str, str]] = []
        self.finished = False

        try:
            self.handle = scope.recorder.begin_call(
                scope.context,
                TraceCallMetadata(
                    kind=TraceKind.MODEL,
                    purpose=scope.purpose,
                    started_at=started_at,
                    model=None if resolved is None else resolved.route.model,
                    reasoning_effort=(
                        None
                        if resolved is None
                        else _trace_reasoning_effort(resolved.route.reasoning_effort)
                    ),
                ),
            )
            set_trace_parent_record_id(self.handle.record_id, scope=scope)
        except Exception:
            self.handle = None
            logger.warning("Execution Trace model capture could not be started")

    def begin_attempt(
        self,
        resolved: ResolvedModelRoute,
        *,
        dispatched: bool = True,
    ) -> TraceProviderAttempt | None:
        if dispatched:
            self.attempt_count += 1
        self.attempt_started = monotonic()
        self.final_attempt_duration_ms = None
        capture_handle = self.handle
        if capture_handle is None:
            return None
        try:
            attempt_handle = self.scope.recorder.begin_attempt(
                capture_handle,
                TraceAttemptMetadata(
                    started_at=datetime.now(UTC),
                    model=resolved.route.model,
                    reasoning_effort=_trace_reasoning_effort(resolved.route.reasoning_effort),
                    # Only a transport adapter can observe an actual request URL.
                    endpoint=None,
                ),
            )
            attempt = TraceProviderAttempt(
                self.scope.recorder,
                attempt_handle,
                capture_expected=dispatched,
            )
            self.active_provider_attempt = attempt
            return attempt
        except Exception:
            logger.warning("Execution Trace model attempt could not be captured")
            return None

    def set_request(
        self,
        attempt: TraceProviderAttempt | None,
        resolved: ResolvedModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
    ) -> None:
        if attempt is None:
            return
        try:
            self.scope.recorder.set_request(
                attempt.handle,
                {
                    "messages": deepcopy(list(messages)),
                    "tools": deepcopy(list(tools)),
                    "model": resolved.route.model,
                    "max_output": resolved.route.max_output,
                    "temperature": resolved.route.temperature,
                    "reasoning_effort": resolved.route.reasoning_effort,
                    "timeout": resolved.route.timeout,
                    "continuation": (
                        None
                        if continuation is None
                        else {
                            "provider_id": continuation.provider_id,
                            "payload": deepcopy(continuation.payload),
                        }
                    ),
                },
            )
        except Exception:
            self._report_issue("The logical model request could not be captured.")
            logger.warning("Execution Trace model request could not be captured")

    def end_attempt(
        self,
        attempt: TraceProviderAttempt | None = None,
        *,
        response_completed: bool = False,
    ) -> None:
        if self.attempt_started is not None:
            self.final_attempt_duration_ms = _elapsed_ms(self.attempt_started)
            self.attempt_started = None
        capture = self.active_provider_attempt if attempt is None else attempt
        if capture is not None:
            if response_completed and capture.response_metadata_captured:
                capture.response_captured = capture.response_body_started
            self.provider_capture_incomplete |= capture.report_incomplete()
        if capture is self.active_provider_attempt:
            self.active_provider_attempt = None

    def note_failure(self, error: BaseException) -> None:
        failure = _trace_failure(error)
        if not self.failures or self.failures[-1] != failure:
            self.failures.append(failure)

    def finish_success(self, response: ModelResponse | None) -> None:
        if response is not None and response.finish_reason == "cancelled":
            self.finish_failure(asyncio.CancelledError())
            return
        try:
            self._finish_success(response)
        except Exception:
            self._report_issue("The logical model response could not be captured.")
            self._finish_success(None)

    def _report_issue(self, message: str, *, source: str = "model_router") -> None:
        if self.handle is None:
            return
        try:
            self.scope.recorder.report_capture_issue(
                self.handle,
                TraceCaptureIssue(TraceCaptureState.PARTIAL, message, source=source),
            )
        except Exception:
            logger.warning("Execution Trace capture issue could not be reported")

    def reject_attempt(self, resolved: ResolvedModelRoute, error: BaseException) -> None:
        attempt = self.begin_attempt(resolved, dispatched=False)
        self.end_attempt(attempt)
        self.note_failure(error)

    def _finish_success(self, response: ModelResponse | None) -> None:
        if response is None:
            result = None
            usage = TraceTokenUsage(TraceUsageState.UNKNOWN)
        else:
            result = {
                "message": response.message.to_dict(),
                "usage": response.usage.to_dict(),
                "finish_reason": response.finish_reason,
                "continuation": (
                    None
                    if response.continuation is None
                    else {
                        "provider_id": response.continuation.provider_id,
                        "payload": deepcopy(response.continuation.payload),
                    }
                ),
            }
            usage = (
                TraceTokenUsage(
                    TraceUsageState.REPORTED,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                    cached_input_tokens=response.usage.cached_input_tokens,
                    cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
                )
                if response.usage.reported
                else TraceTokenUsage(TraceUsageState.UNKNOWN)
            )
        self._finish(
            TraceOutcome(
                status=TraceStatus.SUCCESS,
                finished_at=datetime.now(UTC),
                duration_ms=_elapsed_ms(self.started_monotonic),
                final_attempt_duration_ms=self.final_attempt_duration_ms,
            ),
            result=result,
            usage=usage,
        )

    def finish_failure(self, error: BaseException) -> None:
        self.end_attempt()
        if self.provider_capture_incomplete:
            self._report_issue(
                "Provider HTTP/SSE capture was incomplete for one or more attempts.",
                source="provider_transport",
            )
        self.note_failure(error)
        category, code, final_message = self.failures[-1]
        message = "; ".join(
            f"attempt {index}: {failure_message}"
            for index, (_, _, failure_message) in enumerate(self.failures, start=1)
        )
        self._finish(
            TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=datetime.now(UTC),
                duration_ms=_elapsed_ms(self.started_monotonic),
                final_attempt_duration_ms=self.final_attempt_duration_ms,
                error_category=category,
                error_code=code,
                error_message=message or final_message,
            ),
            result=None,
            usage=TraceTokenUsage(TraceUsageState.UNKNOWN),
        )

    def _finish(
        self,
        outcome: TraceOutcome,
        *,
        result: object,
        usage: TraceTokenUsage,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        handle = self.handle
        if handle is None:
            return
        try:
            self.scope.recorder.finish_call(
                handle,
                outcome,
                result=result,
                usage=usage,
                retry_count=max(0, self.attempt_count - 1),
            )
        except Exception:
            logger.warning("Execution Trace model outcome could not be captured")


def _start_trace_model_call(
    scope: TraceExecutionScope | None,
    resolved: ResolvedModelRoute | None,
    started_at: datetime,
) -> _TraceModelCall | None:
    if scope is None:
        return None
    try:
        return _TraceModelCall(scope, resolved, started_at)
    except Exception:
        logger.warning("Execution Trace model call could not be initialized")
        return None


def _trace_reasoning_effort(effort: ReasoningEffort | None) -> TraceReasoningEffort | None:
    return None if effort is None else TraceReasoningEffort(effort)


def _elapsed_ms(started: float) -> int:
    return max(0, round((monotonic() - started) * 1000))


def _trace_failure(
    error: BaseException,
) -> tuple[TraceErrorCategory, str, str]:
    if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        return TraceErrorCategory.CANCELLED, "cancelled", "The model call was cancelled."
    info = getattr(error, "error", None)
    code = getattr(info, "code", None)
    message = getattr(info, "message", None)
    if not isinstance(code, str) or not code:
        code = type(error).__name__
    if not isinstance(message, str) or not message:
        message = "The model provider call failed."
    normalized_code = code.casefold()
    if "cancel" in normalized_code:
        category = TraceErrorCategory.CANCELLED
    elif "timeout" in normalized_code:
        category = TraceErrorCategory.TIMEOUT
    elif "unavailable" in normalized_code:
        category = TraceErrorCategory.UNAVAILABLE
    elif any(
        term in normalized_code for term in ("overflow", "validation", "invalid", "bad_request")
    ):
        category = TraceErrorCategory.VALIDATION_ERROR
    else:
        category = TraceErrorCategory.PROVIDER_ERROR
    return category, code, message


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


@runtime_checkable
class _ClosableModelStream(Protocol):
    async def aclose(self) -> None: ...


class RunModelRouter:
    """Bind one guard and final route-status snapshot to an Agent Run."""

    def __init__(
        self,
        router: ModelRouterDelegate,
        *,
        guard: ModelAttemptGuard | None = None,
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
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
            guard=call_guard,
            **self._selection_kwargs(),
        )

        async def observe() -> AsyncIterator[ModelStreamEvent]:
            try:
                async for event in events:
                    yield event
            finally:
                try:
                    if isinstance(events, _ClosableModelStream):
                        await events.aclose()
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
        guard: ModelAttemptGuard | None = None,
        session_model_configuration: SessionModelConfiguration | None = None,
        subagent_model_configuration: SessionModelConfiguration | None = None,
    ) -> RunModelRouter:
        """Bind a per-selected-route guard and final route status to an Agent Run."""
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
        trace_scope_snapshot = current_trace_scope()
        started_at = datetime.now(UTC)
        try:
            resolved, reasoning_effort = self._begin_call(
                route,
                continuation=continuation,
                session_model_configuration=session_model_configuration,
                subagent_model_configuration=subagent_model_configuration,
            )
        except BaseException as error:
            trace = _start_trace_model_call(trace_scope_snapshot, None, started_at)
            if trace is not None:
                trace.finish_failure(error)
            raise
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
            trace_scope=trace_scope_snapshot,
            trace_started_at=started_at,
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
        trace_scope: TraceExecutionScope | None,
        trace_started_at: datetime,
    ) -> AsyncIterator[ModelStreamEvent]:
        trace = _start_trace_model_call(trace_scope, resolved, trace_started_at)
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        guarded_route: ResolvedModelRoute | None = None
        response: ModelResponse | None = None
        try:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                if guarded_route is not resolved:
                    try:
                        self._check_attempt_guard(
                            resolved, messages=messages, tools=tools, guard=guard
                        )
                    except BaseException as error:
                        if trace is not None:
                            trace.reject_attempt(resolved, error)
                        raise
                    guarded_route = resolved
                try:
                    provider = self._provider(resolved.provider)
                except BaseException as error:
                    if trace is not None:
                        trace.reject_attempt(resolved, error)
                    raise
                trace_attempt = None if trace is None else trace.begin_attempt(resolved)
                if trace is not None:
                    trace.set_request(
                        trace_attempt,
                        resolved,
                        messages=messages,
                        tools=tools,
                        continuation=(
                            continuation
                            if continuation is not None
                            and continuation.provider_id == resolved.provider.provider_id
                            else None
                        ),
                    )
                emitted = False
                try:
                    with trace_provider_attempt(trace_attempt):
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
                    while True:
                        try:
                            with trace_provider_attempt(trace_attempt):
                                event = await anext(events)
                        except StopAsyncIteration:
                            break
                        emitted = True
                        if isinstance(event, ModelCompleted):
                            response = event.response
                            if trace is not None:
                                trace.end_attempt(trace_attempt, response_completed=True)
                                trace.finish_success(response)
                        yield event
                    if trace is not None:
                        trace.end_attempt(trace_attempt)
                        if response is None:
                            trace.finish_failure(ModelCallError(ErrorInfo(
                                "model_failed", "The model stream ended without a completion.",
                            )))
                        else:
                            trace.finish_success(response)
                    return
                except ModelCallError as failure:
                    if trace is not None:
                        trace.end_attempt(trace_attempt)
                        trace.note_failure(failure)
                    if emitted:
                        raise
                    resolved = await self._recover_attempt(
                        resolved,
                        failure,
                        attempt=attempt,
                        reasoning_effort=reasoning_effort,
                        publish_route_status=publish_route_status,
                    )
        except BaseException as error:
            if trace is not None:
                trace.finish_failure(error)
            raise

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
        trace_scope = current_trace_scope()
        started_at = datetime.now(UTC)
        try:
            resolved, reasoning_effort = self._begin_call(
                route,
                continuation=continuation,
                session_model_configuration=session_model_configuration,
                subagent_model_configuration=subagent_model_configuration,
            )
        except BaseException as error:
            trace = _start_trace_model_call(trace_scope, None, started_at)
            if trace is not None:
                trace.finish_failure(error)
            raise
        trace = _start_trace_model_call(trace_scope, resolved, started_at)
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
            trace=trace,
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
        trace: _TraceModelCall | None,
    ) -> ModelResponse:
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        guarded_route: ResolvedModelRoute | None = None
        try:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                if guarded_route is not resolved:
                    try:
                        self._check_attempt_guard(
                            resolved, messages=messages, tools=tools, guard=guard
                        )
                    except BaseException as error:
                        if trace is not None:
                            trace.reject_attempt(resolved, error)
                        raise
                    guarded_route = resolved
                try:
                    provider = self._provider(resolved.provider)
                except BaseException as error:
                    if trace is not None:
                        trace.reject_attempt(resolved, error)
                    raise
                trace_attempt = None if trace is None else trace.begin_attempt(resolved)
                if trace is not None:
                    trace.set_request(
                        trace_attempt,
                        resolved,
                        messages=messages,
                        tools=tools,
                        continuation=(
                            continuation
                            if continuation is not None
                            and continuation.provider_id == resolved.provider.provider_id
                            else None
                        ),
                    )
                try:
                    with trace_provider_attempt(trace_attempt):
                        response = await provider.complete(
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
                    if trace is not None:
                        trace.end_attempt(trace_attempt, response_completed=True)
                        trace.finish_success(response)
                    return response
                except ModelCallError as failure:
                    if trace is not None:
                        trace.end_attempt(trace_attempt)
                        trace.note_failure(failure)
                    resolved = await self._recover_attempt(
                        resolved,
                        failure,
                        attempt=attempt,
                        reasoning_effort=reasoning_effort,
                        publish_route_status=publish_route_status,
                    )

            raise AssertionError("Provider attempt budget exhausted without a terminal result")
        except BaseException as error:
            if trace is not None:
                trace.finish_failure(error)
            raise

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
        may_fallback = (
            current.requested_route in _FALLBACK_ROUTES
            and current.selected_route != "chat"
        )
        if failure.error.retryable and code in _RETRYABLE_CODES:
            if may_fallback and attempt == _MAX_ATTEMPTS - 1:
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
        if not may_fallback or not allows_fallback or attempt == _MAX_ATTEMPTS:
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
        if continuation is not None and requested_route in _FALLBACK_ROUTES:
            previous = self.current_call_status(requested_route)
            if previous is not None and previous.provider_id == continuation.provider_id:
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
            return self._configuration.resolve_session_model_route(
                subagent_model_configuration.provider_id,
                subagent_model_configuration.model,
                subagent_model_configuration.reasoning_effort,
                requested_route="subagent",
            )
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
