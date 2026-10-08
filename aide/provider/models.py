"""Provider-neutral model response values and direct call contracts."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from aide.agent.tools.tool_gateway import ModelToolCall
from aide.provider.errors import EmptyModelResponseError
from aide.provider.session_configuration import (
    REASONING_EFFORT_LEVELS as REASONING_EFFORT_LEVELS,
)
from aide.provider.session_configuration import (
    ReasoningEffort as ReasoningEffort,
)
from aide.provider.session_configuration import (
    SessionModelConfiguration as SessionModelConfiguration,
)
from aide.utils.validation import require_nonnegative_int

type ModelRoute = Literal["chat", "title", "memory", "schedule", "subagent"]
type FinishReason = Literal["stop", "tool_calls", "length", "cancelled"]
type ModelMessageDictionary = dict[str, Any]
type ModelMessages = Sequence[ModelMessageDictionary]


def require_tool_call_sequence(value: object) -> Sequence[object]:
    """Return a serialized assistant tool-call sequence or reject malformed input."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("assistant tool_calls must be a sequence")
    return value


def last_assistant_message_index(messages: ModelMessages) -> int:
    """Return the final assistant message position."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("continuation requires an assistant message")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Actual token usage reported by one model call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.input_tokens, field="input_tokens")
        require_nonnegative_int(self.output_tokens, field="output_tokens")
        require_nonnegative_int(self.total_tokens, field="total_tokens")
        if self.cached_input_tokens is not None:
            require_nonnegative_int(self.cached_input_tokens, field="cached_input_tokens")
            if self.cached_input_tokens > self.input_tokens:
                raise ValueError("cached_input_tokens must not exceed input_tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            msg = "total_tokens must equal input_tokens + output_tokens"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, int]:
        """Serialize shared counters; cache hits remain per-request metadata."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque Provider-owned state for the next call in the same Tool loop."""

    provider_id: str
    payload: object

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id must not be empty")


@dataclass(frozen=True, slots=True)
class AssistantModelMessage:
    """Aggregated assistant content and provider tool calls."""

    role: ClassVar[Literal["assistant"]] = "assistant"
    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()

    def __post_init__(self) -> None:
        tool_call_ids = [tool_call.id for tool_call in self.tool_calls]
        if len(set(tool_call_ids)) != len(tool_call_ids):
            raise ValueError("assistant tool call IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One complete provider-neutral model response."""

    message: AssistantModelMessage
    usage: ModelUsage
    finish_reason: FinishReason
    continuation: ModelContinuation | None = None

    def __post_init__(self) -> None:
        if not self.message.content.strip() and not self.message.tool_calls:
            raise EmptyModelResponseError("model response requires content or tool calls")


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One ordered, non-provider-specific streaming text chunk."""

    type: ClassVar[Literal["text_delta"]] = "text_delta"
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            msg = "delta must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """One ordered, Provider-returned visible reasoning chunk."""

    type: ClassVar[Literal["reasoning_delta"]] = "reasoning_delta"
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            msg = "delta must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    """The sole successful terminal event in a provider stream."""

    type: ClassVar[Literal["completed"]] = "completed"
    response: ModelResponse


type ModelStreamEvent = ReasoningDelta | TextDelta | ModelCompleted


class ModelProvider(Protocol):
    """Direct keyword-only Model Provider boundary."""

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse: ...

    async def close(self) -> None: ...
