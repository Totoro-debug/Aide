from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal, cast
from uuid import uuid4

import pytest

from aide.agent.confirmation import ConfirmationAborted
from aide.agent.run_errors import CommittableAgentRunError
from aide.agent.runner import (
    AgentRunner,
    AgentRunnerResponseSegmentEnd,
    AgentRunnerResult,
    AgentRunnerRouter,
    AgentRunnerToolCallFinished,
    AgentRunnerToolCallStarted,
)
from aide.agent.tools.base import ArtifactReference
from aide.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolResult,
)
from aide.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from aide.provider.errors import ModelCallError
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
)
from tests.fixtures import (
    DetachedRequestPreparer,
    FakeTool,
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    SingleToolGateway,
    StreamScript,
)


async def _observe(events: list[object], event: object) -> None:
    events.append(event)


async def _ignore_output(event: object) -> None:
    del event


class _DetachedRunner:
    def __init__(self, router: AgentRunnerRouter) -> None:
        self._router = router

    async def run(
        self,
        initial_messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> AgentRunnerResult:
        return await AgentRunner(
            self._router,
            DetachedRequestPreparer(initial_messages),
        ).run(initial_messages, **kwargs)


def _runner(router: AgentRunnerRouter) -> _DetachedRunner:
    return _DetachedRunner(router)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "finish_reason"),
    (
        (ErrorInfo("model_context_overflow", "summary overflow"), "failed"),
        (ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE), "cancelled"),
    ),
)
async def test_committable_preparation_failure_forms_terminal_result_without_model_call(
    error: ErrorInfo,
    finish_reason: Literal["failed", "cancelled"],
) -> None:
    class FailingSummaryPreparer(DetachedRequestPreparer):
        async def prepare(
            self,
            *,
            increment: Sequence[dict[str, Any]],
            latest_cycle_start: int | None,
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None,
            continuation_revision: int,
            is_micro_compression_eligible: Callable[[str], bool] | None,
        ) -> list[dict[str, Any]]:
            del self, increment, latest_cycle_start, tools, continuation, continuation_revision
            del is_micro_compression_eligible
            raise CommittableAgentRunError(error)

    provider = ScriptedFakeProvider()
    result = await AgentRunner(
        ScriptedFakeRouter(provider),
        FailingSummaryPreparer(),
    ).run(
        [{"role": "user", "content": "request"}],
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == finish_reason
    assert result.error == error
    assert result.usage["model_calls"] == 0
    if finish_reason == "cancelled":
        assert result.messages == []
    else:
        assert result.messages[-1]["status"] == "error"
        assert result.messages[-1]["error"] == {
            "code": error.code,
            "message": error.message,
        }
    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_runner_uses_run_local_router_and_request_provenance_recorder() -> None:
    class Router:
        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, messages, tools, continuation

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="done"),
                        usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                        finish_reason="stop",
                    )
                )

            return replay()

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del route, messages, tools, continuation
            raise AssertionError("unexpected completion")

    class RecordingPreparer(DetachedRequestPreparer):
        def __init__(self, initial_messages: Sequence[dict[str, Any]]) -> None:
            super().__init__(initial_messages)
            self.recorded = 0

        async def prepare(
            self,
            *,
            increment: Sequence[dict[str, Any]],
            latest_cycle_start: int | None,
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None,
            continuation_revision: int,
            is_micro_compression_eligible: Callable[[str], bool] | None,
        ) -> list[dict[str, Any]]:
            return await super().prepare(
                increment=increment,
                latest_cycle_start=latest_cycle_start,
                tools=tools,
                continuation=continuation,
                continuation_revision=continuation_revision,
                is_micro_compression_eligible=is_micro_compression_eligible,
            )

        def record_response(
            self,
            *,
            request_messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            response: ModelResponse,
            increment: Sequence[dict[str, Any]],
        ) -> dict[str, object]:
            del request_messages, tools, response, increment
            self.recorded += 1
            return {"selected_route": "chat"}

    initial_messages = [{"role": "user", "content": "request"}]
    router = Router()
    preparer = RecordingPreparer(initial_messages)
    result = await AgentRunner(router, preparer).run(
        initial_messages,
        model="chat",
        tool_gateway=None,
        on_output=None,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert preparer.recorded == 1
    assert result.messages[0]["context_usage"] == {"selected_route": "chat"}


class _ClosingRouter:
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: object = None,
    ) -> AsyncIterator[object]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[object]:
            try:
                yield TextDelta(delta="callback failure")
            finally:
                self.closed.set()

        return replay()

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: object = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        raise AssertionError("Unexpected complete call")


class _ConcurrentRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: object = None,
    ) -> AsyncIterator[object]:
        del route, tools
        marker = str(messages[0]["content"])
        self.calls.append((marker, continuation))

        async def replay() -> AsyncIterator[object]:
            await asyncio.sleep(0)
            if continuation is None:
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(
                            content=f"{marker} tool",
                            tool_calls=(
                                ModelToolCall(id=f"call-{marker}", name=marker, arguments="{}"),
                            ),
                        ),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                        continuation=ModelContinuation(provider_id=marker, payload=marker),
                    )
                )
            else:
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content=f"{marker} done"),
                        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                        finish_reason="stop",
                    )
                )

        return replay()

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: object = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        raise AssertionError("Unexpected complete call")


class _DirectGateway:
    def __init__(self, order: list[str], results: Sequence[ToolResult] = ()) -> None:
        self.schemas: list[dict[str, Any]] = []
        self._order = order
        self._results = list(results)
        self.confirmations: list[object] = []

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: object = None,
    ) -> ToolResult:
        self._order.append("gateway")
        self.confirmations.append(confirmation)
        if self._results:
            return self._results.pop(0)
        return ToolResult(
            tool_call_id=tool_call.id, name=tool_call.name, status="success", content="done"
        )

    def is_micro_compression_eligible(self, tool_name: str) -> bool:
        del tool_name
        return False


class _RetryingRouter:
    def __init__(self) -> None:
        self.logical_calls = 0
        self.provider_attempts = 0

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelCompleted]:
        del route, messages, tools, continuation
        self.logical_calls += 1

        async def replay() -> AsyncIterator[ModelCompleted]:
            self.provider_attempts += 1
            try:
                raise ModelCallError(ErrorInfo("provider_timeout", "retry once"))
            except ModelCallError:
                self.provider_attempts += 1
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Retried."),
                    usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                    finish_reason="stop",
                )
            )

        return replay()

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        raise AssertionError("Unexpected complete call")


class _RecordingRequestPreparer(DetachedRequestPreparer):
    def __init__(self, initial_messages: Sequence[dict[str, Any]]) -> None:
        super().__init__(initial_messages)
        self.requests: list[dict[str, Any]] = []

    async def prepare(
        self,
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        continuation_revision: int,
        is_micro_compression_eligible: Callable[[str], bool] | None,
    ) -> list[dict[str, Any]]:
        prepared = await super().prepare(
            increment=increment,
            latest_cycle_start=latest_cycle_start,
            tools=tools,
            continuation=continuation,
            continuation_revision=continuation_revision,
            is_micro_compression_eligible=is_micro_compression_eligible,
        )
        self.requests.append(
            {
                "prepared": deepcopy(prepared),
                "increment": deepcopy(list(increment)),
                "latest_cycle_start": latest_cycle_start,
                "tools": deepcopy(tuple(tools)),
                "continuation": continuation,
                "continuation_revision": continuation_revision,
                "has_micro_compression_predicate": is_micro_compression_eligible is not None,
            }
        )
        return prepared


def _tool_iteration_scripts(count: int) -> tuple[StreamScript, ...]:
    return tuple(
        StreamScript(
            events=(
                ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(
                            content="Continue",
                            tool_calls=(
                                ModelToolCall(id=f"call-{number}", name="work", arguments="{}"),
                            ),
                        ),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                ),
            )
        )
        for number in range(count)
    )


def test_runner_requires_explicit_max_iterations_input() -> None:
    parameter = inspect.signature(AgentRunner.run).parameters["max_iterations"]

    assert parameter.default is inspect.Parameter.empty


def test_runner_constructor_and_module_exclude_product_orchestration_dependencies() -> None:
    module = inspect.getmodule(AgentRunner)

    assert tuple(inspect.signature(AgentRunner).parameters) == (
        "model_router",
        "request_preparer",
    )
    assert module is not None
    assert {"Session", "MessageBus", "ContextBuilder"}.isdisjoint(vars(module))


@pytest.mark.asyncio
async def test_runner_notifies_once_after_first_request_preparation_before_provider_call() -> None:
    order: list[str] = []
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )

    class ObservedRouter:
        def __init__(self, router: ScriptedFakeRouter) -> None:
            self._router = router

        def stream(
            self,
            route: Literal["chat", "schedule", "subagent"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            order.append("provider")
            return self._router.stream(
                route,
                messages=messages,
                tools=tools,
                continuation=continuation,
            )

        async def complete(
            self,
            route: Literal["chat", "schedule", "subagent"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            return await self._router.complete(
                route,
                messages=messages,
                tools=tools,
                continuation=continuation,
            )

    def first_request_prepared() -> None:
        order.append("prepared")

    initial_messages = [{"role": "user", "content": "Run."}]
    await AgentRunner(
        ObservedRouter(ScriptedFakeRouter(provider)),
        DetachedRequestPreparer(initial_messages),
    ).run(
        initial_messages,
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
        on_first_request_prepared=first_request_prepared,
    )

    assert order == ["prepared", "provider"]


@pytest.mark.asyncio
async def test_runner_prepares_each_logical_request_with_run_local_context() -> None:
    first_call = ModelToolCall(id="call-1", name="work", arguments="{}")
    second_call = ModelToolCall(id="call-2", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="First",
                                tool_calls=(first_call,),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                            continuation=ModelContinuation(
                                provider_id="test-provider",
                                payload=object(),
                            ),
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Second",
                                tool_calls=(second_call,),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                            continuation=None,
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _DirectGateway([])
    gateway.schemas = [{"name": "work", "description": "work"}]
    initial_messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Run."},
    ]
    preparer = _RecordingRequestPreparer(initial_messages)
    prepared_notifications: list[bool] = []

    result = await AgentRunner(ScriptedFakeRouter(provider), preparer).run(
        initial_messages,
        model="chat",
        tool_gateway=gateway,  # type: ignore[arg-type]
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
        on_first_request_prepared=lambda: prepared_notifications.append(True),
    )

    assert result.finish_reason == "completed"
    assert len(provider.stream_requests) == len(preparer.requests) == 3
    assert prepared_notifications == [True]
    assert [request["latest_cycle_start"] for request in preparer.requests] == [None, 0, 2]
    assert [request["continuation_revision"] for request in preparer.requests] == [0, 1, 2]
    assert [
        None if request["continuation"] is None else request["continuation"].provider_id
        for request in preparer.requests
    ] == [None, "test-provider", None]
    assert all(request["tools"] == tuple(gateway.schemas) for request in preparer.requests)
    assert [len(request["increment"]) for request in preparer.requests] == [0, 2, 4]
    assert [request["prepared"] for request in preparer.requests] == [
        request.messages for request in provider.stream_requests
    ]
    assert all(request["has_micro_compression_predicate"] for request in preparer.requests)
    assert all(
        message["role"] in {"assistant", "tool"}
        for request in preparer.requests
        for message in request["increment"]
    )
    assert preparer.requests[1]["prepared"][-1]["content"] == "done"
    assert preparer.requests[2]["prepared"][-1]["content"] == "done"


@pytest.mark.asyncio
async def test_detached_request_preparation_is_value_equivalent() -> None:
    messages = [{"role": "user", "content": {"nested": ["original"]}}]

    prepared = await DetachedRequestPreparer(messages).prepare(
        increment=(),
        latest_cycle_start=None,
        tools=(),
        continuation=None,
        continuation_revision=0,
        is_micro_compression_eligible=None,
    )

    assert prepared == messages
    assert prepared is not messages
    prepared[0]["content"]["nested"].append("changed")
    assert messages[0]["content"] == {"nested": ["original"]}


@pytest.mark.asyncio
async def test_request_preparer_cannot_mutate_runner_messages_or_provider_tools() -> None:
    class MutatingPreparer(DetachedRequestPreparer):
        async def prepare(
            self,
            *,
            increment: Sequence[dict[str, Any]],
            latest_cycle_start: int | None,
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None,
            continuation_revision: int,
            is_micro_compression_eligible: Callable[[str], bool] | None,
        ) -> list[dict[str, Any]]:
            prepared = await super().prepare(
                increment=increment,
                latest_cycle_start=latest_cycle_start,
                tools=tools,
                continuation=continuation,
                continuation_revision=continuation_revision,
                is_micro_compression_eligible=is_micro_compression_eligible,
            )
            prepared[0]["content"]["nested"].append("projected")
            tools[0]["parameters"]["enum"].append("mutated")
            return prepared

    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _DirectGateway([])
    gateway.schemas = [{"name": "work", "parameters": {"type": "string", "enum": ["original"]}}]
    initial_messages = [{"role": "user", "content": {"nested": ["original"]}}]

    await AgentRunner(ScriptedFakeRouter(provider), MutatingPreparer(initial_messages)).run(
        initial_messages,
        model="chat",
        tool_gateway=gateway,  # type: ignore[arg-type]
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert provider.stream_requests[0].messages[0]["content"] == {
        "nested": ["original", "projected"]
    }
    assert provider.stream_requests[0].tools == tuple(gateway.schemas)
    assert gateway.schemas[0]["parameters"]["enum"] == ["original"]
    assert initial_messages[0]["content"] == {"nested": ["original"]}


@pytest.mark.asyncio
async def test_request_prepare_failure_propagates_original_without_provider_call() -> None:
    failure = RuntimeError("prepare failed")

    class FailingPreparer(DetachedRequestPreparer):
        async def prepare(
            self,
            *,
            increment: Sequence[dict[str, Any]],
            latest_cycle_start: int | None,
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None,
            continuation_revision: int,
            is_micro_compression_eligible: Callable[[str], bool] | None,
        ) -> list[dict[str, Any]]:
            del self, increment, latest_cycle_start, tools, continuation, continuation_revision
            del is_micro_compression_eligible
            raise failure

    provider = ScriptedFakeProvider()
    initial_messages = [{"role": "user", "content": "request"}]

    with pytest.raises(RuntimeError) as captured:
        await AgentRunner(
            ScriptedFakeRouter(provider),
            FailingPreparer(initial_messages),
        ).run(
            initial_messages,
            model="chat",
            tool_gateway=None,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
            propagate_unexpected_errors=True,
        )

    assert captured.value is failure
    assert str(captured.value.__cause__) == "Agent Runner request preparation failed"
    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_request_prepare_cancelled_error_stops_before_provider() -> None:
    failure = asyncio.CancelledError("preparation cancelled")

    class CancellingPreparer(DetachedRequestPreparer):
        async def prepare(
            self,
            *,
            increment: Sequence[dict[str, Any]],
            latest_cycle_start: int | None,
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None,
            continuation_revision: int,
            is_micro_compression_eligible: Callable[[str], bool] | None,
        ) -> list[dict[str, Any]]:
            del self, increment, latest_cycle_start, tools, continuation, continuation_revision
            del is_micro_compression_eligible
            raise failure

    provider = ScriptedFakeProvider()
    initial_messages = [{"role": "user", "content": "request"}]

    with pytest.raises(asyncio.CancelledError) as captured:
        await AgentRunner(
            ScriptedFakeRouter(provider),
            CancellingPreparer(initial_messages),
        ).run(
            initial_messages,
            model="chat",
            tool_gateway=None,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
            propagate_unexpected_errors=True,
        )

    assert captured.value is failure
    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_record_response_failure_propagates_original_after_one_provider_call() -> None:
    failure = RuntimeError("record failed")

    class FailingRecorderPreparer(DetachedRequestPreparer):
        def record_response(
            self,
            *,
            request_messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            response: ModelResponse,
            increment: Sequence[dict[str, Any]],
        ) -> dict[str, object] | None:
            del self, request_messages, tools, response, increment
            raise failure

    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    initial_messages = [{"role": "user", "content": "request"}]

    with pytest.raises(RuntimeError) as captured:
        await AgentRunner(
            ScriptedFakeRouter(provider),
            FailingRecorderPreparer(initial_messages),
        ).run(
            initial_messages,
            model="chat",
            tool_gateway=None,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
            propagate_unexpected_errors=True,
        )

    assert captured.value is failure
    assert str(captured.value.__cause__) == "Agent Runner request preparation failed"
    assert len(provider.stream_requests) == 1


def test_result_validates_exact_usage_and_finish_invariants() -> None:
    usage = {
        "model_calls": 1,
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }

    result = AgentRunnerResult(
        messages=[],
        final_content="done",
        usage=usage,
        finish_reason="completed",
    )

    assert result.usage == usage

    with pytest.raises(ValueError):
        AgentRunnerResult(
            messages=[],
            final_content="done",
            usage={**usage, "extra": 0},
            finish_reason="completed",
        )

    invalid_results = (
        {"finish_reason": "completed", "error": ErrorInfo("model_failed", "unexpected")},
        {"finish_reason": "failed", "error": None},
        {"finish_reason": "cancelled", "error": ErrorInfo("model_failed", "wrong")},
        {"finish_reason": "max_iterations", "error": ErrorInfo("model_failed", "wrong")},
    )
    for fields in invalid_results:
        with pytest.raises(ValueError):
            AgentRunnerResult(messages=[], final_content="done", usage=usage, **fields)

    with pytest.raises(ValueError):
        AgentRunnerResult(
            messages=[],
            final_content="done",
            usage={**usage, "input_tokens": True},
        )

    for invalid_usage in (
        {key: value for key, value in usage.items() if key != "model_calls"},
        {**usage, "model_calls": -1},
        {**usage, "total_tokens": 6},
    ):
        with pytest.raises(ValueError):
            AgentRunnerResult(messages=[], final_content="done", usage=invalid_usage)

    for finish_reason, error in (
        ("failed", ErrorInfo("model_failed", "safe failure")),
        ("cancelled", ErrorInfo("turn_cancelled", "safe cancellation")),
        ("max_iterations", ErrorInfo("agent_iteration_limit", "safe limit")),
    ):
        assert (
            AgentRunnerResult(
                messages=[],
                final_content="done",
                usage=usage,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                error=error,
            ).error
            == error
        )


@pytest.mark.asyncio
async def test_runner_returns_generated_increment_and_closes_response_segment() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Hello"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Hello"),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    events: list[object] = []
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}],
        model="chat",
        tool_gateway=None,
        on_output=lambda event: _observe(events, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert [message["role"] for message in result.messages] == ["assistant"]
    assert result.messages[0]["content"] == "Hello"
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert isinstance(events[-1], AgentRunnerResponseSegmentEnd)


@pytest.mark.asyncio
async def test_runner_emits_completed_content_when_provider_omits_text_deltas() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Completed without deltas"),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    events: list[object] = []

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Hello"}],
        model="chat",
        tool_gateway=None,
        on_output=lambda event: _observe(events, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.final_content == "Completed without deltas"
    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "Completed without deltas"
    ]
    assert isinstance(events[-1], AgentRunnerResponseSegmentEnd)


@pytest.mark.asyncio
async def test_router_internal_retry_consumes_one_runner_model_call() -> None:
    router = _RetryingRouter()
    initial_messages = [{"role": "user", "content": "Retry."}]
    preparer = _RecordingRequestPreparer(initial_messages)

    result = await AgentRunner(router, preparer).run(
        initial_messages,
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.usage["model_calls"] == 1
    assert router.logical_calls == 1
    assert router.provider_attempts == 2
    assert len(preparer.requests) == 1


@pytest.mark.asyncio
async def test_runner_executes_all_tools_in_provider_order_in_one_iteration() -> None:
    calls = tuple(
        ModelToolCall(id=f"call_{number}", name=f"tool_{number}", arguments="{}")
        for number in (1, 2, 3)
    )
    continuation = ModelContinuation(provider_id="test-provider", payload={"opaque": "state"})
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=calls),
                            usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
                            finish_reason="tool_calls",
                            continuation=continuation,
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=7, output_tokens=1, total_tokens=8),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    tools = [
        FakeTool(name=f"tool_{number}", description="test", outcomes=(f"result-{number}",))
        for number in (1, 2, 3)
    ]
    gateway = SingleToolGateway(tools)
    observed: list[object] = []
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Run the tools."}],
        model="chat",
        tool_gateway=gateway,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert [message["role"] for message in result.messages] == [
        "assistant",
        "tool",
        "tool",
        "tool",
        "assistant",
    ]
    assert [message["content"] for message in result.messages[1:4]] == [
        "result-1",
        "result-2",
        "result-3",
    ]
    assert [
        event.tool_call_id for event in observed if isinstance(event, AgentRunnerToolCallStarted)
    ] == [
        "call_1",
        "call_2",
        "call_3",
    ]
    assert result.usage == {
        "model_calls": 2,
        "input_tokens": 9,
        "output_tokens": 4,
        "total_tokens": 13,
    }
    assert provider.stream_requests[1].continuation == continuation
    assert all("continuation" not in message for message in result.messages)


@pytest.mark.asyncio
async def test_runner_passes_confirmation_requester_directly_before_tool_call() -> None:
    call = ModelToolCall(id="call", name="work", arguments='{"raw":true}')
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    order: list[str] = []
    gateway = _DirectGateway(order)

    async def requester(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "approved"

    async def observe(event: object) -> None:
        if isinstance(event, AgentRunnerToolCallStarted):
            order.append("callback")

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Run work."}],
        model="chat",
        tool_gateway=gateway,
        on_output=observe,
        confirmation=requester,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.final_content == "Done"
    assert order == ["callback", "gateway"]
    assert gateway.confirmations == [requester]


@pytest.mark.asyncio
async def test_runner_propagates_confirmation_aborted_without_turn_repair() -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )

    class AbortingGateway(_DirectGateway):
        async def call(
            self,
            tool_call: ModelToolCall,
            *,
            confirmation: object = None,
        ) -> ToolResult:
            del tool_call, confirmation
            raise ConfirmationAborted("confirmation lifecycle cancelled")

    with pytest.raises(ConfirmationAborted):
        await _runner(ScriptedFakeRouter(provider)).run(
            [{"role": "user", "content": "Run work."}],
            model="chat",
            tool_gateway=AbortingGateway([]),
            on_output=None,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("success", "error", "refused"))
async def test_runner_continues_after_provider_valid_tool_result_status(
    status: str,
) -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    tool_result = ToolResult(
        tool_call_id=call.id,
        name=call.name,
        status=status,  # type: ignore[arg-type]
        content=f"{status} result",
    )
    gateway = _DirectGateway([], (tool_result,))
    observed: list[object] = []

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Continue."}],
        model="chat",
        tool_gateway=gateway,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.messages[1]["status"] == status
    assert result.final_content == "Done"
    finished = [event for event in observed if isinstance(event, AgentRunnerToolCallFinished)]
    assert len(finished) == 1
    assert (finished[0].tool_call_id, finished[0].tool_name, finished[0].status) == (
        call.id,
        call.name,
        status,
    )
    assert finished[0].result == tool_result.content


@pytest.mark.asyncio
async def test_runner_uses_complete_for_schedule_without_stream_output() -> None:
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Scheduled."),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="stop",
            ),
        )
    )
    observed: list[object] = []
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Schedule this."}],
        model="schedule",
        tool_gateway=None,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.final_content == "Scheduled."
    assert provider.complete_requests
    assert not provider.stream_requests
    assert observed == []


@pytest.mark.asyncio
async def test_runner_honors_cancellation_after_schedule_response() -> None:
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Scheduled."),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="stop",
            ),
        )
    )
    cancellation = iter((False, False, True)).__next__

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Cancel schedule."}],
        model="schedule",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=cancellation,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.error is not None
    assert result.error.code == "turn_cancelled"
    assert result.final_content == ""
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


@pytest.mark.asyncio
async def test_runner_switches_and_closes_only_real_stream_segments() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ReasoningDelta(delta="think"),
                    TextDelta(delta="answer"),
                    TextDelta(delta=" more"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="answer more"),
                            usage=ModelUsage(input_tokens=1, output_tokens=3, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    observed: list[object] = []
    runner = _runner(ScriptedFakeRouter(provider))

    await runner.run(
        [{"role": "user", "content": "Explain."}],
        model="chat",
        tool_gateway=None,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert [type(event) for event in observed] == [
        ReasoningDelta,
        AgentRunnerResponseSegmentEnd,
        TextDelta,
        TextDelta,
        AgentRunnerResponseSegmentEnd,
    ]
    assert [
        event.segment for event in observed if isinstance(event, AgentRunnerResponseSegmentEnd)
    ] == ["reasoning", "response"]


@pytest.mark.asyncio
async def test_reasoning_cancellation_closes_segment_without_fabricating_a_message() -> None:
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ReasoningDelta(delta="visible reasoning"),)),)
    )
    cancellation = iter((False, False, True)).__next__
    observed: list[object] = []

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Cancel reasoning."}],
        model="chat",
        tool_gateway=None,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=cancellation,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.messages == []
    assert [type(event) for event in observed] == [
        ReasoningDelta,
        AgentRunnerResponseSegmentEnd,
    ]


@pytest.mark.asyncio
async def test_runner_counts_a_failed_logical_model_call_and_repairs_it() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(ErrorInfo("model_failed", "provider failed")),
            ),
        )
    )
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Fail."}],
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "failed"
    assert result.error is not None
    assert result.error.code == "model_failed"
    assert result.usage == {
        "model_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert [message["role"] for message in result.messages] == ["assistant"]


@pytest.mark.asyncio
async def test_provider_turn_cancellation_returns_structured_cancelled_result() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="partial"),),
                error=ModelCallError(ErrorInfo("turn_cancelled", "provider cancelled")),
            ),
        )
    )

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Cancel from provider."}],
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.error is not None
    assert result.error.code == "turn_cancelled"
    assert result.final_content == "partial"
    assert result.messages[-1]["status"] == "interrupted"
    assert result.usage["model_calls"] == 1


@pytest.mark.asyncio
async def test_entry_cancellation_does_not_start_a_model_call() -> None:
    provider = ScriptedFakeProvider()
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Cancel."}],
        model="chat",
        tool_gateway=None,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=lambda: True,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.error is not None
    assert result.error.code == "turn_cancelled"
    assert result.messages == []
    assert result.usage["model_calls"] == 0
    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_runner_repairs_partial_response_on_cooperative_cancellation() -> None:
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(TextDelta(delta="partial"),)),))
    cancellation = iter((False, False, True)).__next__
    observed: list[object] = []
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Cancel after text."}],
        model="chat",
        tool_gateway=None,
        on_output=lambda event: _observe(observed, event),
        confirmation=None,
        externalize_result=None,
        cancel_requested=cancellation,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.messages[-1]["content"] == "partial"
    assert result.messages[-1]["status"] == "interrupted"
    assert result.final_content == "partial"
    assert isinstance(observed[-1], AgentRunnerResponseSegmentEnd)


@pytest.mark.asyncio
async def test_runner_cancellation_repairs_only_unfinished_tools() -> None:
    calls = (
        ModelToolCall(id="first", name="first", arguments="{}"),
        ModelToolCall(id="second", name="second", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Tools", tool_calls=calls),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    first = FakeTool(name="first", description="first", outcomes=("done-first",))
    second = FakeTool(name="second", description="second", outcomes=("done-second",))
    gateway = SingleToolGateway((first, second))
    cancellation = iter((False, False, False, False, True)).__next__
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Use both."}],
        model="chat",
        tool_gateway=gateway,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=cancellation,
        max_iterations=50,
    )

    assert [message["role"] for message in result.messages] == ["assistant", "tool", "tool"]
    assert result.messages[1]["status"] == "success"
    assert result.messages[2]["status"] == "error"
    assert first.calls and not second.calls


@pytest.mark.asyncio
async def test_cancellation_after_completed_tool_response_keeps_one_assistant_sequence() -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Working"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    work = FakeTool(name="work", description="work", outcomes=("unused",))
    cancellation = iter((False, False, False, True)).__next__

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Cancel after the model response."}],
        model="chat",
        tool_gateway=SingleToolGateway((work,)),
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=cancellation,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert [message["role"] for message in result.messages] == ["assistant", "tool"]
    assert result.messages[0]["status"] == "completed"
    assert result.messages[1]["status"] == "error"
    assert not work.calls


@pytest.mark.asyncio
async def test_runner_externalizes_tool_result() -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Tool", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway((FakeTool(name="work", description="work", outcomes=("large",)),))
    runner = _runner(ScriptedFakeRouter(provider))

    def externalize(result: ToolResult) -> ToolResult:
        assert result.content == "large"
        return replace(
            result,
            content="preview",
            artifact=ArtifactReference(
                path=".aide/artifacts/session/call.txt",
                total_chars=5,
                preview_chars=5,
            ),
        )

    result = await runner.run(
        [{"role": "user", "content": "Externalize."}],
        model="chat",
        tool_gateway=gateway,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=externalize,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.messages[1]["content"] == "preview"
    assert result.messages[1]["artifact"] == {
        "path": ".aide/artifacts/session/call.txt",
        "total_chars": 5,
        "preview_chars": 5,
    }


@pytest.mark.asyncio
async def test_runner_normalizes_externalizer_failure_to_safe_tool_error() -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Tool", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done"),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _DirectGateway(
        [],
        (
            ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="success",
                content="large",
            ),
        ),
    )

    def externalize(result: ToolResult) -> ToolResult:
        del result
        raise RuntimeError("private artifact detail")

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Externalizer failure."}],
        model="chat",
        tool_gateway=gateway,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=externalize,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.messages[1] == {
        "role": "tool",
        "tool_call_id": "call",
        "name": "work",
        "status": "error",
        "content": "work result could not be stored.",
        "artifact": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", (49, 0, True, 50.0))
async def test_runner_rejects_invalid_max_iterations(value: object) -> None:
    runner = _runner(ScriptedFakeRouter(ScriptedFakeProvider()))

    with pytest.raises(ValueError):
        await runner.run(
            [],
            model="chat",
            tool_gateway=None,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=value,
        )


@pytest.mark.asyncio
async def test_runner_stops_after_fiftieth_tool_iteration_without_a_new_model_call() -> None:
    provider = ScriptedFakeProvider(streams=_tool_iteration_scripts(50))
    work = FakeTool(name="work", description="work", outcomes=tuple("ok" for _ in range(50)))
    gateway = SingleToolGateway((work,))
    runner = _runner(ScriptedFakeRouter(provider))

    result = await runner.run(
        [{"role": "user", "content": "Keep working."}],
        model="chat",
        tool_gateway=gateway,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "max_iterations"
    assert result.error is not None
    assert result.error.code == "agent_iteration_limit"
    assert result.final_content == (
        "Aide 本轮对话已经达到最大循环次数，仍没有输出最终结果。"  # noqa: RUF001
        "可以再次尝试本次请求或者尝试给出更明确的任务目标。"
    )
    assert len(provider.stream_requests) == 50
    assert len(work.calls) == 50
    assert result.usage["model_calls"] == 50
    assert result.messages[-1]["content"] == result.final_content
    assert result.messages[-1]["token_usage"]["model_calls"] == 0


@pytest.mark.asyncio
async def test_runner_completes_when_fiftieth_model_response_has_no_tools() -> None:
    final_script = StreamScript(
        events=(
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Done on fifty."),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )
            ),
        )
    )
    provider = ScriptedFakeProvider(streams=(*_tool_iteration_scripts(49), final_script))
    work = FakeTool(name="work", description="work", outcomes=tuple("ok" for _ in range(49)))

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Finish on the boundary."}],
        model="chat",
        tool_gateway=SingleToolGateway((work,)),
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "completed"
    assert result.error is None
    assert result.final_content == "Done on fifty."
    assert result.usage["model_calls"] == 50
    assert len(provider.stream_requests) == 50
    assert len(work.calls) == 49


@pytest.mark.asyncio
async def test_cancellation_after_fiftieth_tool_takes_priority_over_iteration_limit() -> None:
    provider = ScriptedFakeProvider(streams=_tool_iteration_scripts(50))
    work = FakeTool(name="work", description="work", outcomes=tuple("ok" for _ in range(50)))

    result = await _runner(ScriptedFakeRouter(provider)).run(
        [{"role": "user", "content": "Cancel at the boundary."}],
        model="chat",
        tool_gateway=SingleToolGateway((work,)),
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=lambda: len(work.calls) == 50,
        max_iterations=50,
    )

    assert result.finish_reason == "cancelled"
    assert result.error is not None
    assert result.error.code == "turn_cancelled"
    assert result.usage["model_calls"] == 50
    assert len(provider.stream_requests) == 50
    assert len(work.calls) == 50
    assert all(
        (message.get("error") or {}).get("code") != "agent_iteration_limit"
        for message in result.messages
    )


@pytest.mark.asyncio
async def test_callback_failure_propagates_and_closes_provider_iterator() -> None:
    router = _ClosingRouter()
    failure = RuntimeError("output sink failed")

    async def fail(event: object) -> None:
        del event
        raise failure

    runner = _runner(cast(AgentRunnerRouter, router))

    with pytest.raises(RuntimeError) as captured:
        await runner.run(
            [{"role": "user", "content": "Callback."}],
            model="chat",
            tool_gateway=None,
            on_output=fail,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        )

    assert captured.value is failure
    assert str(captured.value.__cause__) == "Agent Runner output callback failed"
    assert router.closed.is_set()


@pytest.mark.asyncio
async def test_callback_failure_while_repairing_model_error_stays_external() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="partial"),),
                error=ModelCallError(ErrorInfo("model_failed", "provider failed")),
            ),
        )
    )

    async def fail_on_segment_end(event: object) -> None:
        if isinstance(event, AgentRunnerResponseSegmentEnd):
            raise RuntimeError("end sink failed")

    with pytest.raises(RuntimeError, match="end sink failed"):
        await _runner(ScriptedFakeRouter(provider)).run(
            [{"role": "user", "content": "Callback during failure."}],
            model="chat",
            tool_gateway=None,
            on_output=fail_on_segment_end,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        )


@pytest.mark.asyncio
async def test_tool_start_callback_failure_does_not_start_gateway_call() -> None:
    call = ModelToolCall(id="call", name="work", arguments='{"raw":true}')
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    order: list[str] = []
    gateway = _DirectGateway(order)

    async def fail(event: object) -> None:
        if isinstance(event, AgentRunnerToolCallStarted):
            raise RuntimeError("tool output sink failed")

    with pytest.raises(RuntimeError, match="tool output sink failed"):
        await _runner(ScriptedFakeRouter(provider)).run(
            [{"role": "user", "content": "Fail before Tool."}],
            model="chat",
            tool_gateway=gateway,
            on_output=fail,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        )

    assert order == []


@pytest.mark.asyncio
async def test_task_cancellation_closes_tool_operation_and_confirmation_future() -> None:
    call = ModelToolCall(id="call", name="work", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Working", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    requested = asyncio.Event()
    decision: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
    cancel_requested = False

    class BlockingGateway(_DirectGateway):
        async def call(
            self,
            tool_call: ModelToolCall,
            *,
            confirmation: object = None,
        ) -> ToolResult:
            assert callable(confirmation)
            request = ConfirmationRequest(
                confirmation_id=uuid4(),
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason="test",
                summary="Confirm work",
                details={},
            )
            requested.set()
            await confirmation(request)
            raise AssertionError("Cancelled confirmation unexpectedly resumed")

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return await decision

    task = asyncio.create_task(
        _runner(ScriptedFakeRouter(provider)).run(
            [{"role": "user", "content": "Cancel confirmation."}],
            model="chat",
            tool_gateway=BlockingGateway([]),
            on_output=_ignore_output,
            confirmation=confirm,
            externalize_result=None,
            cancel_requested=lambda: cancel_requested,
            max_iterations=50,
        )
    )
    await requested.wait()
    cancel_requested = True
    task.cancel()

    result = await task

    assert result.finish_reason == "cancelled"
    assert decision.cancelled()
    assert [message["role"] for message in result.messages] == ["assistant", "tool"]


@pytest.mark.asyncio
async def test_noncooperative_task_cancellation_propagates_after_iterator_close() -> None:
    router = _ClosingRouter()
    started = asyncio.Event()

    def blocking_stream(
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: object = None,
    ) -> AsyncIterator[object]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[object]:
            started.set()
            try:
                await asyncio.Event().wait()
                yield TextDelta(delta="unreachable")
            finally:
                router.closed.set()

        return replay()

    router.stream = blocking_stream  # type: ignore[method-assign]
    runner = _runner(cast(AgentRunnerRouter, router))
    task = asyncio.create_task(
        runner.run(
            [{"role": "user", "content": "Cancel task."}],
            model="chat",
            tool_gateway=None,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=lambda: False,
            max_iterations=50,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert router.closed.is_set()


@pytest.mark.asyncio
async def test_task_cancellation_during_callback_honors_cooperative_cancel_request() -> None:
    router = _ClosingRouter()
    callback_started = asyncio.Event()
    blocker = asyncio.Event()
    cancel_requested = False

    async def output(event: object) -> None:
        if isinstance(event, TextDelta):
            callback_started.set()
            await blocker.wait()

    runner = _runner(cast(AgentRunnerRouter, router))
    task = asyncio.create_task(
        runner.run(
            [{"role": "user", "content": "Cancel in callback."}],
            model="chat",
            tool_gateway=None,
            on_output=output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=lambda: cancel_requested,
            max_iterations=50,
        )
    )
    await callback_started.wait()
    cancel_requested = True
    task.cancel()

    result = await task

    assert result.finish_reason == "cancelled"
    assert result.error is not None
    assert result.error.code == "turn_cancelled"
    assert result.messages[-1]["status"] == "interrupted"
    assert result.messages[-1]["content"] == "callback failure"
    assert router.closed.is_set()


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_share_increment_usage_or_continuation() -> None:
    router = _ConcurrentRouter()
    gateway = SingleToolGateway(
        (
            FakeTool(name="A", description="A", outcomes=("A result",)),
            FakeTool(name="B", description="B", outcomes=("B result",)),
        )
    )
    runner = _runner(cast(AgentRunnerRouter, router))

    results = await asyncio.gather(
        runner.run(
            [{"role": "user", "content": "A"}],
            model="chat",
            tool_gateway=gateway,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        ),
        runner.run(
            [{"role": "user", "content": "B"}],
            model="chat",
            tool_gateway=gateway,
            on_output=_ignore_output,
            confirmation=None,
            externalize_result=None,
            cancel_requested=None,
            max_iterations=50,
        ),
    )

    assert {result.final_content for result in results} == {"A done", "B done"}
    assert all(result.usage["model_calls"] == 2 for result in results)
    observed_continuations: list[tuple[str, object | None]] = []
    for marker, continuation in router.calls:
        assert continuation is None or isinstance(continuation, ModelContinuation)
        observed_continuations.append(
            (marker, None if continuation is None else continuation.payload)
        )
    assert set(observed_continuations) == {
        ("A", None),
        ("B", None),
        ("A", "A"),
        ("B", "B"),
    }
