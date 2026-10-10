"""Anthropic SDK adapter for the provider-neutral model contract."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Protocol, cast

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
)

from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.config import ProviderConfiguration
from aide.provider.errors import (
    EmptyModelResponseError,
    ModelCallError,
    parse_retry_after_seconds,
)
from aide.provider.models import (
    AssistantModelMessage,
    FinishReason,
    ModelCompleted,
    ModelContinuation,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    ReasoningEffort,
    TextDelta,
    last_assistant_message_index,
    require_tool_call_sequence,
)
from aide.utils.errors import ErrorInfo
from aide.utils.json_types import JsonObject, JsonValue

_REASONING_EFFORT_MAP: Final[Mapping[ReasoningEffort, str]] = MappingProxyType(
    {
        "low": "low",
        "mid": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
)


class AnthropicMessages(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class AnthropicClient(Protocol):
    @property
    def messages(self) -> AnthropicMessages: ...

    async def close(self) -> None: ...


class AnthropicClientFactory(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_retries: int,
    ) -> AnthropicClient: ...


def _official_client_factory(
    *,
    api_key: str,
    base_url: str,
    max_retries: int,
) -> AnthropicClient:
    return cast(
        AnthropicClient,
        AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=max_retries),
    )


class AnthropicProvider:
    """Translate Anthropic SDK calls at the public ModelProvider boundary."""

    def __init__(
        self,
        configuration: ProviderConfiguration,
        *,
        client_factory: AnthropicClientFactory = _official_client_factory,
    ) -> None:
        self._provider_id = configuration.provider_id
        self._client = client_factory(
            api_key=configuration.api_key,
            base_url=configuration.base_url,
            max_retries=0,
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        return self._stream_with_arguments(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        )

    async def _stream_with_arguments(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            async for event in self._stream_once(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            ):
                yield event
        except EmptyModelResponseError as error:
            raise _empty_response_error() from error
        except APIError as error:
            raise _normalized_error(error) from error
        except (TypeError, ValueError) as error:
            raise _model_failed_error() from error

    async def _stream_once(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        sdk_stream = await self._client.messages.create(
            **_request_arguments(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                stream=True,
                provider_id=self._provider_id,
                continuation=continuation,
            )
        )
        if not hasattr(sdk_stream, "__aiter__"):
            msg = "Anthropic streaming response is not asynchronously iterable"
            raise TypeError(msg)

        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens: int | None = None
        stop_reason: str | None = None
        tool_uses: dict[int, _StreamingToolUse] = {}
        content_blocks: dict[int, dict[str, object]] = {}
        async for event in cast(AsyncIterator[object], sdk_stream):
            event_type = _string_field(event, "type")
            if event_type == "message_start":
                usage = _field(_field(event, "message"), "usage")
                input_tokens, cached_input_tokens = _input_usage(usage)
            elif event_type == "content_block_delta":
                delta = _field(event, "delta")
                delta_type = _string_field(delta, "type")
                if delta_type == "text_delta":
                    text = _string_field(delta, "text")
                    if text:
                        text_parts.append(text)
                        block = content_blocks.setdefault(
                            _integer_field(event, "index"),
                            {"type": "text", "text": ""},
                        )
                        block["text"] = f"{_string_field(block, 'text') or ''}{text}"
                        yield TextDelta(delta=text)
                elif delta_type == "thinking_delta":
                    thinking = _string_field(delta, "thinking")
                    if thinking:
                        index = _integer_field(event, "index")
                        block = content_blocks.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        block["thinking"] = f"{_string_field(block, 'thinking') or ''}{thinking}"
                        yield ReasoningDelta(delta=thinking)
                elif delta_type == "signature_delta":
                    signature = _string_field(delta, "signature")
                    if signature:
                        index = _integer_field(event, "index")
                        block = content_blocks.setdefault(
                            index,
                            {"type": "thinking", "thinking": "", "signature": ""},
                        )
                        block["signature"] = f"{_string_field(block, 'signature') or ''}{signature}"
                elif delta_type == "input_json_delta":
                    index = _integer_field(event, "index")
                    partial_json = _string_field(delta, "partial_json")
                    if index in tool_uses and partial_json:
                        tool_uses[index].json_parts.append(partial_json)
            elif event_type == "content_block_start":
                content_block = _field(event, "content_block")
                index = _integer_field(event, "index")
                block_type = _string_field(content_block, "type")
                if block_type == "text":
                    text = _string_field(content_block, "text") or ""
                    content_blocks[index] = {"type": "text", "text": text}
                    if text:
                        text_parts.append(text)
                        yield TextDelta(delta=text)
                elif block_type == "thinking":
                    thinking = _string_field(content_block, "thinking") or ""
                    content_blocks[index] = {
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": _string_field(content_block, "signature") or "",
                    }
                    if thinking:
                        yield ReasoningDelta(delta=thinking)
                elif block_type == "redacted_thinking":
                    content_blocks[index] = {
                        "type": "redacted_thinking",
                        "data": _string_field(content_block, "data") or "",
                    }
                elif block_type == "tool_use":
                    tool_uses[index] = _StreamingToolUse(
                        id=_string_field(content_block, "id") or "",
                        name=_string_field(content_block, "name") or "",
                        initial_input=_field(content_block, "input"),
                    )
                    content_blocks[index] = {
                        "type": "tool_use",
                        "id": _string_field(content_block, "id") or "",
                        "name": _string_field(content_block, "name") or "",
                        "input": _field(content_block, "input"),
                    }
            elif event_type == "message_delta":
                stop_reason = _string_field(_field(event, "delta"), "stop_reason")
                output_tokens = _integer_field(_field(event, "usage"), "output_tokens")

        continuation = _stream_continuation(
            content_blocks,
            tool_uses,
            provider_id=self._provider_id,
        )
        response = ModelResponse(
            message=AssistantModelMessage(
                content="".join(text_parts),
                tool_calls=tuple(
                    tool_uses[index].to_model_tool_call() for index in sorted(tool_uses)
                ),
            ),
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_input_tokens=cached_input_tokens,
            ),
            finish_reason=_finish_reason(stop_reason),
            continuation=continuation,
        )
        yield ModelCompleted(response=response)

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        try:
            message = await self._client.messages.create(
                **_request_arguments(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_output=max_output,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    stream=False,
                    provider_id=self._provider_id,
                    continuation=continuation,
                )
            )
            return _response_from_message(message, provider_id=self._provider_id)
        except EmptyModelResponseError as error:
            raise _empty_response_error() from error
        except APIError as error:
            raise _normalized_error(error) from error
        except (TypeError, ValueError) as error:
            raise _model_failed_error() from error

    async def close(self) -> None:
        await self._client.close()


def _request_arguments(
    *,
    messages: ModelMessages,
    tools: Sequence[dict[str, Any]],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    stream: bool,
    provider_id: str,
    continuation: ModelContinuation | None,
) -> dict[str, object]:
    system: object | None = None
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")
        messages = messages[1:]
    continuation_index = (
        last_assistant_message_index(messages) if continuation is not None else None
    )
    translated_messages = [
        _message_argument(
            message,
            continuation=continuation if index == continuation_index else None,
            provider_id=provider_id,
        )
        for index, message in enumerate(messages)
    ]
    translated_tools: list[dict[str, object]] = []
    for tool in tools:
        function = tool["function"]
        translated_tools.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    arguments: dict[str, object] = {
        "max_tokens": max_output,
        "messages": translated_messages,
        "model": model,
        "stream": stream,
        "temperature": temperature,
        "timeout": timeout,
        "tools": translated_tools,
    }
    if system is not None:
        arguments["system"] = system
    if reasoning_effort is not None:
        arguments["output_config"] = {"effort": _REASONING_EFFORT_MAP[reasoning_effort]}
    return arguments


def _message_argument(
    message: Mapping[str, object],
    *,
    continuation: ModelContinuation | None = None,
    provider_id: str,
) -> dict[str, object]:
    role = message.get("role")
    if role == "user":
        return {"role": "user", "content": message.get("content", "")}
    if role == "assistant":
        if continuation is not None:
            return {
                "role": "assistant",
                "content": _anthropic_continuation_content(continuation, provider_id),
            }
        content: list[dict[str, object]] = []
        message_content = message.get("content", "")
        if message_content:
            content.append({"type": "text", "text": message_content})
        tool_calls = message.get("tool_calls", ())
        content.extend(
            _anthropic_tool_use(tool_call) for tool_call in require_tool_call_sequence(tool_calls)
        )
        return {"role": "assistant", "content": content}
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": message.get("content", ""),
                }
            ],
        }
    raise TypeError("Model message role is unsupported")


def _anthropic_continuation_content(
    continuation: ModelContinuation,
    provider_id: str,
) -> list[dict[str, object]]:
    if continuation.provider_id != provider_id:
        raise ValueError("model continuation belongs to a different provider")
    payload = continuation.payload
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise TypeError("Anthropic continuation payload must be a sequence")
    blocks: list[dict[str, object]] = []
    for block in payload:
        if not isinstance(block, Mapping):
            raise TypeError("Anthropic continuation blocks must be mappings")
        block_type = block.get("type")
        if block_type not in {"thinking", "redacted_thinking", "text", "tool_use"}:
            raise ValueError("Anthropic continuation block type is unsupported")
        blocks.append(dict(block))
    if not blocks:
        raise ValueError("Anthropic continuation payload must not be empty")
    return blocks


def _anthropic_tool_use(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("assistant tool call must be a mapping")
    name = value.get("name")
    arguments = value.get("arguments", "")
    if not isinstance(name, str) or not isinstance(arguments, str):
        raise TypeError("assistant tool call name and arguments must be strings")
    return {
        "type": "tool_use",
        "id": value.get("id", ""),
        "name": name,
        "input": _argument_object(arguments),
    }


def _response_from_message(message: object, *, provider_id: str) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ModelToolCall] = []
    continuation_blocks: list[dict[str, object]] = []
    has_continuation = False
    content = _field(message, "content")
    if isinstance(content, (list, tuple)):
        for block in content:
            block_type = _string_field(block, "type")
            continuation_block = _continuation_block(block)
            if continuation_block is not None:
                continuation_blocks.append(continuation_block)
            if block_type in {"thinking", "redacted_thinking"}:
                has_continuation = True
            if block_type == "text":
                text = _string_field(block, "text")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        id=_string_field(block, "id") or "",
                        name=_string_field(block, "name") or "",
                        arguments=_json_text(_field(block, "input")),
                    )
                )
    usage = _field(message, "usage")
    input_tokens, cached_input_tokens = _input_usage(usage)
    output_tokens = _integer_field(usage, "output_tokens")
    return ModelResponse(
        message=AssistantModelMessage(content="".join(text_parts), tool_calls=tuple(tool_calls)),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_input_tokens=cached_input_tokens,
        ),
        finish_reason=_finish_reason(_string_field(message, "stop_reason")),
        continuation=(
            None
            if not has_continuation
            else ModelContinuation(provider_id=provider_id, payload=tuple(continuation_blocks))
        ),
    )


def _input_usage(usage: object) -> tuple[int, int | None]:
    cached = _field(usage, "cache_read_input_tokens")
    cached_input_tokens = cached if type(cached) is int and cached >= 0 else None
    input_tokens = (
        _integer_field(usage, "input_tokens")
        + _integer_field(usage, "cache_creation_input_tokens")
        + (cached_input_tokens or 0)
    )
    return input_tokens, cached_input_tokens


def _stream_continuation(
    content_blocks: Mapping[int, Mapping[str, object]],
    tool_uses: Mapping[int, _StreamingToolUse],
    *,
    provider_id: str,
) -> ModelContinuation | None:
    if not any(
        _string_field(block, "type") in {"thinking", "redacted_thinking"}
        for block in content_blocks.values()
    ):
        return None
    blocks: list[dict[str, object]] = []
    for index in sorted(content_blocks):
        block = dict(content_blocks[index])
        if _string_field(block, "type") == "tool_use" and index in tool_uses:
            block["input"] = tool_uses[index].to_input_object()
        blocks.append(block)
    return ModelContinuation(provider_id=provider_id, payload=tuple(blocks))


def _continuation_block(block: object) -> dict[str, object] | None:
    block_type = _string_field(block, "type")
    if block_type == "thinking":
        continuation_block: dict[str, object] = {
            "type": block_type,
            "thinking": _string_field(block, "thinking") or "",
        }
        signature = _string_field(block, "signature")
        if signature is not None:
            continuation_block["signature"] = signature
        return continuation_block
    if block_type == "redacted_thinking":
        return {
            "type": block_type,
            "data": _string_field(block, "data") or "",
        }
    if block_type == "text":
        return {"type": block_type, "text": _string_field(block, "text") or ""}
    if block_type == "tool_use":
        return {
            "type": block_type,
            "id": _string_field(block, "id") or "",
            "name": _string_field(block, "name") or "",
            "input": _json_object(_field(block, "input")),
        }
    return None


@dataclass(slots=True)
class _StreamingToolUse:
    id: str
    name: str
    initial_input: object
    json_parts: list[str] = field(default_factory=list)

    def to_input_object(self) -> JsonObject:
        if self.json_parts:
            return _argument_object("".join(self.json_parts))
        return _json_object(self.initial_input)

    def to_model_tool_call(self) -> ModelToolCall:
        if self.json_parts:
            arguments = "".join(self.json_parts)
            _argument_object(arguments)
        else:
            arguments = _json_text(self.initial_input)
        return ModelToolCall(id=self.id, name=self.name, arguments=arguments)


def _argument_object(arguments: str) -> JsonObject:
    try:
        value = cast(object, json.loads(arguments))
    except json.JSONDecodeError as error:
        raise ValueError("Anthropic tool arguments must be a JSON object") from error
    return _json_object(value)


def _json_text(value: object) -> str:
    return json.dumps(
        _json_object(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = "Anthropic tool arguments must be a JSON object"
        raise ValueError(msg)
    return {cast(str, key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {cast(str, key): _json_value(item) for key, item in value.items()}
    msg = "Anthropic tool arguments contain a non-JSON value"
    raise ValueError(msg)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _string_field(value: object, name: str) -> str | None:
    field = _field(value, name)
    return field if isinstance(field, str) else None


def _integer_field(value: object, name: str) -> int:
    field = _field(value, name)
    return field if isinstance(field, int) and not isinstance(field, bool) else 0


def _finish_reason(stop_reason: str | None) -> FinishReason:
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def _normalized_error(
    error: APIError,
) -> ModelCallError:
    if isinstance(error, APITimeoutError):
        return ModelCallError(
            ErrorInfo(
                code="provider_timeout",
                message="Anthropic request timed out.",
                retryable=True,
            )
        )
    if isinstance(error, APIConnectionError):
        return ModelCallError(
            ErrorInfo(
                code="provider_unavailable",
                message="Anthropic is temporarily unavailable.",
                retryable=True,
            )
        )
    if not isinstance(error, APIStatusError):
        return _model_failed_error()

    status = error.status_code
    if status in {401, 403}:
        return ModelCallError(
            ErrorInfo(
                code="provider_auth_error",
                message="Anthropic authentication failed.",
            )
        )
    if status == 408:
        return ModelCallError(
            ErrorInfo(
                code="provider_timeout",
                message="Anthropic request timed out.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    if status == 429:
        return ModelCallError(
            ErrorInfo(
                code="provider_rate_limited",
                message="Anthropic rate limit was reached.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    if status >= 500:
        return ModelCallError(
            ErrorInfo(
                code="provider_unavailable",
                message="Anthropic is temporarily unavailable.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    provider_message = _provider_error_message(error).lower()
    if status == 404 or _describes_unsupported_capability(provider_message):
        return ModelCallError(
            ErrorInfo(
                code="route_unavailable",
                message="Anthropic model or capability is unavailable.",
            )
        )
    if status == 413 or _describes_context_overflow(provider_message):
        return ModelCallError(
            ErrorInfo(
                code="model_context_overflow",
                message="Anthropic model context was exceeded.",
            )
        )
    if status in {400, 422}:
        return ModelCallError(
            ErrorInfo(
                code="model_invalid_request",
                message="Anthropic rejected the model request.",
            )
        )
    return _model_failed_error()


def _model_failed_error() -> ModelCallError:
    return ModelCallError(ErrorInfo(code="model_failed", message="Anthropic model call failed."))


def _empty_response_error() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_failed",
            message="Anthropic provider returned an empty response. Check its model configuration.",
        )
    )


def _retry_after_seconds(error: APIStatusError) -> float | None:
    return parse_retry_after_seconds(error.response.headers.get("retry-after"))


def _provider_error_message(error: APIStatusError) -> str:
    body_error = _field(error.body, "error")
    return _string_field(body_error, "message") or ""


def _describes_unsupported_capability(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "does not support",
            "not supported",
            "unsupported",
        )
    )


def _describes_context_overflow(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "prompt is too long",
            "too many tokens",
            "context window",
            "maximum context",
            "maximum allowed number of tokens",
        )
    )
