"""Reusable, bounded Agent Runner execution without conversation ownership."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol
from uuid import UUID

from loguru import logger

from myclaw.agent.confirmation import ConfirmationAborted
from myclaw.agent.run_errors import CommittableAgentRunError
from myclaw.agent.tools.file_mutation import FileMutationRecorder
from myclaw.agent.tools.tool_gateway import (
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)
from myclaw.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
)
from myclaw.utils.validation import empty_token_usage, token_usage_validation_issue

type AgentRunnerRoute = Literal["chat", "schedule"]
type AgentRunnerSegment = Literal["reasoning", "response"]
type AgentRunnerFinishReason = Literal["completed", "failed", "cancelled", "max_iterations"]
type AgentRunnerOutput = (
    ReasoningDelta
    | TextDelta
    | AgentRunnerResponseSegmentEnd
    | AgentRunnerToolCallStarted
    | AgentRunnerToolCallFinished
)

_MAX_ITERATIONS_MESSAGE = (
    "MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。"  # noqa: RUF001
    "可以再次尝试本次请求或者尝试给出更明确的任务目标。"
)


@dataclass(frozen=True, slots=True)
class AgentRunnerResponseSegmentEnd:
    """End of one contiguous Provider-visible reasoning or response segment."""

    type: ClassVar[Literal["segment_end"]] = "segment_end"
    segment: AgentRunnerSegment


@dataclass(frozen=True, slots=True)
class AgentRunnerToolCallStarted:
    """A Tool call immediately before it is handed to the Tool Gateway."""

    type: ClassVar[Literal["tool_call_started"]] = "tool_call_started"
    tool_call_id: str
    tool_name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AgentRunnerToolCallFinished:
    """The observed Tool outcome without its result content or artifact."""

    type: ClassVar[Literal["tool_call_finished"]] = "tool_call_finished"
    tool_call_id: str
    tool_name: str
    status: Literal["success", "error", "refused"]


class AgentRunnerOutputCallback(Protocol):
    async def __call__(self, event: AgentRunnerOutput) -> None: ...


class AgentRunnerRouter(Protocol):
    def stream(
        self,
        route: AgentRunnerRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        route: AgentRunnerRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse: ...


class AgentRunRequestPreparer(Protocol):
    """Prepare one provider-neutral request from detached Agent Run snapshots."""

    async def prepare(
        self,
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        continuation_revision: int,
        is_micro_compression_eligible: Callable[[str], bool] | None,
    ) -> Sequence[dict[str, Any]]: ...

    def record_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None: ...


@dataclass(slots=True)
class AgentRunnerResult:
    messages: list[dict[str, Any]]
    final_content: str
    usage: dict[str, int] = field(default_factory=empty_token_usage)
    finish_reason: AgentRunnerFinishReason = "completed"
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.final_content, str):
            raise TypeError("final_content must be a string")
        usage_issue = token_usage_validation_issue(self.usage)
        if usage_issue == "fields":
            raise ValueError("usage must contain exactly the four Agent Runner usage fields")
        if usage_issue == "values":
            raise ValueError("usage values must be nonnegative integers")
        if usage_issue == "total":
            raise ValueError("usage total_tokens must equal input_tokens + output_tokens")

        if self.finish_reason not in {"completed", "failed", "cancelled", "max_iterations"}:
            raise ValueError("finish_reason is invalid")
        if self.finish_reason == "completed" and self.error is not None:
            raise ValueError("completed Agent Runner results cannot contain an error")
        if self.finish_reason == "failed" and not isinstance(self.error, ErrorInfo):
            raise ValueError("failed Agent Runner results require an ErrorInfo")
        if self.finish_reason == "cancelled":
            if not isinstance(self.error, ErrorInfo) or self.error.code != "turn_cancelled":
                raise ValueError("cancelled Agent Runner results require turn_cancelled")
        if self.finish_reason == "max_iterations":
            if not isinstance(self.error, ErrorInfo) or self.error.code != "agent_iteration_limit":
                raise ValueError(
                    "maximum-iteration Agent Runner results require agent_iteration_limit"
                )


class _PropagatedFailure(Exception):
    def __init__(self, error: BaseException, message: str) -> None:
        self.error = error
        super().__init__(message)


class AgentRunner:
    """Run bounded ReAct execution while owning only a Model Router reference."""

    def __init__(
        self,
        model_router: AgentRunnerRouter,
        request_preparer: AgentRunRequestPreparer,
    ) -> None:
        self._model_router = model_router
        self._request_preparer = request_preparer

    async def run(
        self,
        initial_messages: Sequence[dict[str, Any]],
        *,
        model: AgentRunnerRoute,
        tool_gateway: ToolGateway | None,
        on_output: AgentRunnerOutputCallback | None,
        confirmation: ConfirmationRequester | None,
        externalize_result: Callable[[ToolResult], ToolResult] | None,
        cancel_requested: Callable[[], bool] | None,
        max_iterations: int,
        file_mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
        stop_on_tool_error: bool = False,
        propagate_unexpected_errors: bool = False,
        tool_calls_as_tasks: bool = True,
    ) -> AgentRunnerResult:
        _validate_max_iterations(max_iterations)

        active_router = self._model_router
        active_preparer = self._request_preparer
        runtime_messages = deepcopy(list(initial_messages))
        increment: list[dict[str, Any]] = []
        pending_tool_calls: list[ModelToolCall] = []
        partial_content: list[str] = []
        usage = empty_token_usage()
        continuation: ModelContinuation | None = None
        segment: AgentRunnerSegment | None = None
        events: AsyncIterator[ModelStreamEvent] | None = None
        model_call_started = False
        is_cancel_requested = cancel_requested or _never_cancel
        externalize = externalize_result or _identity_tool_result
        latest_cycle_start: int | None = None
        continuation_revision = 0

        async def emit(event: AgentRunnerOutput) -> None:
            if on_output is None:
                return
            try:
                await on_output(event)
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                raise _PropagatedFailure(
                    error,
                    "Agent Runner output callback failed",
                ) from error
            except BaseException as error:
                raise _PropagatedFailure(
                    error,
                    "Agent Runner output callback failed",
                ) from error

        async def close_segment() -> None:
            nonlocal segment
            if segment is None:
                return
            closing = segment
            segment = None
            await emit(AgentRunnerResponseSegmentEnd(segment=closing))

        async def close_segment_for_error() -> None:
            try:
                await close_segment()
            except _PropagatedFailure as failure:
                raise failure.error from failure

        async def start_segment(next_segment: AgentRunnerSegment) -> None:
            nonlocal segment
            if segment != next_segment:
                await close_segment()
                segment = next_segment

        def finish_cancelled(*, final_content: str) -> AgentRunnerResult:
            _repair_cancelled_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                model_calls=int(model_call_started),
            )
            return _cancelled_result(increment, usage, final_content=final_content)

        if is_cancel_requested():
            return finish_cancelled(final_content="")

        try:
            while True:
                partial_content.clear()
                model_call_started = False
                response: ModelResponse | None = None
                current_cycle_start_in_increment = len(increment)
                exposed_tools = (
                    () if tool_gateway is None else tuple(deepcopy(tool_gateway.schemas))
                )
                try:
                    request_messages = await active_preparer.prepare(
                        increment=deepcopy(increment),
                        latest_cycle_start=latest_cycle_start,
                        tools=deepcopy(exposed_tools),
                        continuation=continuation,
                        continuation_revision=continuation_revision,
                        is_micro_compression_eligible=(
                            None
                            if tool_gateway is None
                            else tool_gateway.is_micro_compression_eligible
                        ),
                    )
                except (asyncio.CancelledError, ModelCallError):
                    raise
                except BaseException as error:
                    raise _PropagatedFailure(
                        error,
                        "Agent Runner request preparation failed",
                    ) from error
                model_call_started = True
                usage["model_calls"] += 1
                if model == "chat":
                    router = active_router
                    events = router.stream(
                        model,
                        messages=request_messages,
                        tools=exposed_tools,
                        continuation=continuation,
                    )
                    try:
                        async for event in events:
                            if isinstance(event, ReasoningDelta):
                                await start_segment("reasoning")
                                await emit(event)
                                if is_cancel_requested():
                                    await close_segment()
                                    return finish_cancelled(final_content="")
                                continue
                            if isinstance(event, TextDelta):
                                await start_segment("response")
                                partial_content.append(event.delta)
                                await emit(event)
                                if is_cancel_requested():
                                    cancelled_content = "".join(partial_content)
                                    await close_segment()
                                    return finish_cancelled(final_content=cancelled_content)
                                continue
                            if not isinstance(event, ModelCompleted):
                                raise _model_failure()
                            response = event.response
                            await close_segment()
                            if response.message.content and not partial_content:
                                await start_segment("response")
                                partial_content.append(response.message.content)
                                await emit(TextDelta(delta=response.message.content))
                                await close_segment()
                            break
                        if response is None:
                            raise _model_failure()
                    finally:
                        await _close_iterator(events)
                        events = None
                else:
                    response = await active_router.complete(
                        model,
                        messages=request_messages,
                        tools=exposed_tools,
                        continuation=continuation,
                    )

                _add_usage(usage, response.usage)
                assistant_message = _assistant_run_message(response)
                try:
                    context_usage = active_preparer.record_response(
                        request_messages=deepcopy(list(request_messages)),
                        tools=deepcopy(list(exposed_tools)),
                        response=response,
                        increment=deepcopy([*increment, assistant_message]),
                    )
                except (asyncio.CancelledError, ModelCallError):
                    raise
                except BaseException as error:
                    raise _PropagatedFailure(
                        error,
                        "Agent Runner request preparation failed",
                    ) from error
                if context_usage is not None:
                    assistant_message["context_usage"] = deepcopy(context_usage)
                _append_run_message(runtime_messages, increment, assistant_message)
                model_call_started = False
                partial_content.clear()
                pending_tool_calls = (
                    list(response.message.tool_calls) if tool_gateway is not None else []
                )
                continuation_for_next_call = response.continuation if pending_tool_calls else None
                continuation_revision += 1

                if is_cancel_requested():
                    return finish_cancelled(final_content="")

                if not pending_tool_calls:
                    return AgentRunnerResult(
                        messages=increment,
                        final_content=response.message.content,
                        usage=usage,
                    )

                assert tool_gateway is not None
                for tool_call in response.message.tool_calls:
                    if is_cancel_requested():
                        return finish_cancelled(final_content="")
                    await emit(
                        AgentRunnerToolCallStarted(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                    )

                    state = _ToolCallState()
                    result: ToolResult | None = None
                    try:
                        try:
                            result = await _await_tool_call(
                                tool_gateway,
                                tool_call,
                                confirmation,
                                state,
                                as_task=tool_calls_as_tasks,
                                file_mutation_recorder=file_mutation_recorder,
                                run_token=run_token,
                            )
                        except BaseException as failure:
                            if not isinstance(failure, Exception) and state.result is not None:
                                try:
                                    recovered = _externalize_tool_result(state.result, externalize)
                                    _append_run_message(
                                        runtime_messages,
                                        increment,
                                        _tool_run_message(recovered),
                                    )
                                except Exception:
                                    pass
                                else:
                                    pending_tool_calls.pop(0)
                            raise
                        if result is None:
                            raise RuntimeError("Tool Gateway ended without a result")
                    except asyncio.CancelledError:
                        raise
                    except ConfirmationAborted:
                        raise
                    except Exception:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            status="error",
                            content=f"{tool_call.name} could not complete the request.",
                            artifact=None,
                        )

                    result = _externalize_tool_result(result, externalize)
                    _append_run_message(runtime_messages, increment, _tool_run_message(result))
                    pending_tool_calls.pop(0)
                    await emit(
                        AgentRunnerToolCallFinished(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            status=result.status,
                        )
                    )
                    if stop_on_tool_error and result.status != "success":
                        return AgentRunnerResult(
                            messages=increment,
                            final_content="",
                            usage=usage,
                            finish_reason="failed",
                            error=ErrorInfo("tool_failed", result.content),
                        )
                    if is_cancel_requested():
                        return finish_cancelled(final_content="")

                latest_cycle_start = current_cycle_start_in_increment
                if usage["model_calls"] >= max_iterations:
                    limit_error = ErrorInfo("agent_iteration_limit", _MAX_ITERATIONS_MESSAGE)
                    _append_run_message(
                        runtime_messages,
                        increment,
                        _build_assistant_repair_message(
                            content=_MAX_ITERATIONS_MESSAGE,
                            status="error",
                            error=limit_error,
                            model_calls=0,
                        ),
                    )
                    return AgentRunnerResult(
                        messages=increment,
                        final_content=_MAX_ITERATIONS_MESSAGE,
                        usage=usage,
                        finish_reason="max_iterations",
                        error=limit_error,
                    )

                continuation = continuation_for_next_call
        except _PropagatedFailure as failure:
            raise failure.error from failure
        except ModelCallError as failure:
            await close_segment_for_error()
            if failure.error.code == "model_context_overflow" and model_call_started:
                usage["model_calls"] -= 1
                model_call_started = False
            if failure.error.code == "turn_cancelled" or is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            if (
                failure.error.code == "model_context_overflow"
                and usage["model_calls"] == 0
                and not isinstance(failure, CommittableAgentRunError)
            ):
                raise
            _log_agent_failure(failure)
            failed_content = "".join(partial_content) if model == "chat" else ""
            _repair_failed_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=model == "chat",
                failure=failure,
                model_calls=int(model_call_started),
            )
            return AgentRunnerResult(
                messages=increment,
                final_content=failed_content,
                usage=usage,
                finish_reason="failed",
                error=failure.error,
            )
        except asyncio.CancelledError:
            await close_segment_for_error()
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            try:
                _repair_cancelled_messages(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                    model_calls=int(model_call_started),
                )
            except BaseException:
                pass
            raise
        except ConfirmationAborted:
            raise
        except Exception:
            if propagate_unexpected_errors:
                raise
            await close_segment_for_error()
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            failed_content = "".join(partial_content) if model == "chat" else ""
            generic_failure = _model_failure()
            _repair_failed_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=model == "chat",
                failure=generic_failure,
                model_calls=int(model_call_started),
            )
            return AgentRunnerResult(
                messages=increment,
                final_content=failed_content,
                usage=usage,
                finish_reason="failed",
                error=generic_failure.error,
            )
        finally:
            await _close_iterator(events)


def _validate_max_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 50:
        raise ValueError("max_iterations must be an integer at least 50")


def _never_cancel() -> bool:
    return False


async def _await_tool_call(
    gateway: ToolGateway,
    tool_call: ModelToolCall,
    confirmation: ConfirmationRequester | None,
    state: _ToolCallState,
    *,
    as_task: bool,
    file_mutation_recorder: FileMutationRecorder | None,
    run_token: UUID | None,
) -> ToolResult:
    async def invoke_gateway() -> ToolResult:
        if file_mutation_recorder is None and run_token is None:
            return await gateway.call(tool_call, confirmation=confirmation)
        return await gateway.call(
            tool_call,
            confirmation=confirmation,
            file_mutation_recorder=file_mutation_recorder,
            run_token=run_token,
        )

    if not as_task:
        result = await invoke_gateway()
        state.result = result
        return result
    operation = asyncio.create_task(invoke_gateway())
    try:
        result = await operation
        state.result = result
        return result
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        if state.result is None and operation.done() and not operation.cancelled():
            try:
                state.result = operation.result()
            except BaseException:
                pass


def _add_usage(total: dict[str, int], usage: ModelUsage) -> None:
    total["input_tokens"] += usage.input_tokens
    total["output_tokens"] += usage.output_tokens
    total["total_tokens"] += usage.total_tokens


def _cancelled_result(
    messages: list[dict[str, Any]],
    usage: dict[str, int],
    *,
    final_content: str,
) -> AgentRunnerResult:
    return AgentRunnerResult(
        messages=messages,
        final_content=final_content,
        usage=usage,
        finish_reason="cancelled",
        error=ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE),
    )


@dataclass(slots=True)
class _ToolCallState:
    result: ToolResult | None = None


def _model_failure() -> ModelCallError:
    return ModelCallError(ErrorInfo("model_failed", "The model request failed."))


def _append_run_message(
    runtime_messages: list[dict[str, Any]],
    increment: list[dict[str, Any]],
    message: dict[str, Any],
) -> None:
    runtime_messages.append(deepcopy(message))
    increment.append(deepcopy(message))


def _assistant_run_message(response: ModelResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.message.content,
        "tool_calls": [call.to_dict() for call in response.message.tool_calls],
        "status": "completed",
        "error": None,
        "token_usage": {"model_calls": 1, **response.usage.to_dict()},
    }


def _build_assistant_repair_message(
    *,
    content: str,
    status: Literal["interrupted", "error"],
    error: ErrorInfo,
    model_calls: int = 1,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
        "status": status,
        "error": {"code": error.code, "message": error.message},
        "token_usage": {
            "model_calls": model_calls,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _tool_run_message(result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", **result.to_dict()}


def _identity_tool_result(result: ToolResult) -> ToolResult:
    return result


def _externalize_tool_result(
    result: ToolResult,
    externalize_result: Callable[[ToolResult], ToolResult],
    *,
    on_artifact_failure: Callable[[Exception, str], None] | None = None,
) -> ToolResult:
    try:
        return externalize_result(result)
    except Exception as failure:
        if on_artifact_failure is not None:
            on_artifact_failure(failure, result.name)
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status="error",
            content=f"{result.name} result could not be stored.",
            artifact=None,
            confirmation=result.confirmation,
        )


def _repair_cancelled_messages(
    runtime_messages: list[dict[str, Any]],
    increment: list[dict[str, Any]],
    partial_content: list[str],
    pending_tool_calls: list[ModelToolCall],
    *,
    model_calls: int,
) -> None:
    if partial_content:
        _append_run_message(
            runtime_messages,
            increment,
            _build_assistant_repair_message(
                content="".join(partial_content),
                status="interrupted",
                error=ErrorInfo(
                    code="turn_cancelled",
                    message="Turn interrupted by user.",
                ),
                model_calls=model_calls,
            ),
        )
    for tool_call in pending_tool_calls:
        _append_run_message(
            runtime_messages,
            increment,
            _tool_run_message(
                ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    status="error",
                    content="Tool call interrupted because the turn was cancelled.",
                    artifact=None,
                )
            ),
        )
    pending_tool_calls.clear()
    partial_content.clear()


def _repair_failed_messages(
    runtime_messages: list[dict[str, Any]],
    increment: list[dict[str, Any]],
    partial_content: list[str],
    pending_tool_calls: list[ModelToolCall],
    *,
    stream: bool,
    failure: ModelCallError,
    model_calls: int,
) -> None:
    for tool_call in pending_tool_calls:
        _append_run_message(
            runtime_messages,
            increment,
            _tool_run_message(
                ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    status="error",
                    content="Tool call interrupted because the Agent Run failed.",
                    artifact=None,
                )
            ),
        )
    pending_tool_calls.clear()
    _append_run_message(
        runtime_messages,
        increment,
        _build_assistant_repair_message(
            content="".join(partial_content) if stream else "",
            status="error",
            error=failure.error,
            model_calls=model_calls,
        ),
    )
    partial_content.clear()


async def _close_iterator(iterator: AsyncIterator[object] | None) -> None:
    if iterator is None:
        return
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:
        pass


__all__ = [
    "AgentRunRequestPreparer",
    "AgentRunner",
    "AgentRunnerFinishReason",
    "AgentRunnerOutput",
    "AgentRunnerOutputCallback",
    "AgentRunnerResponseSegmentEnd",
    "AgentRunnerResult",
    "AgentRunnerRoute",
    "AgentRunnerRouter",
    "AgentRunnerSegment",
    "AgentRunnerToolCallFinished",
    "AgentRunnerToolCallStarted",
]


def _log_agent_failure(failure: ModelCallError) -> None:
    def set_runtime_name(record: Any) -> None:
        record["name"] = "myclaw.agent.runner"

    logger.patch(set_runtime_name).opt(exception=failure).error(
        "Agent Run failed code={} type={}",
        failure.error.code,
        type(failure).__name__,
    )
