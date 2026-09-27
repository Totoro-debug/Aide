import pytest

from myclaw.provider.errors import EmptyModelResponseError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
    last_assistant_message_index,
)
from myclaw.tools.gateway import ModelToolCall


def test_last_assistant_message_index_returns_the_last_assistant() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "tool", "content": "Tool result"},
        {"role": "assistant", "content": "Latest answer"},
        {"role": "tool", "content": "Latest tool result"},
    ]

    assert last_assistant_message_index(messages) == 3


def test_last_assistant_message_index_requires_an_assistant() -> None:
    with pytest.raises(ValueError, match=r"^continuation requires an assistant message$"):
        last_assistant_message_index(
            [
                {"role": "user", "content": "Question"},
                {"role": "tool", "content": "Tool result"},
            ]
        )


def test_model_usage_serializes_to_the_exact_contract_shape() -> None:
    usage = ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144)

    assert usage.to_dict() == {
        "input_tokens": 120,
        "output_tokens": 24,
        "total_tokens": 144,
    }


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "total_tokens"),
    [(-1, 0, -1), (1, 2, 4), (True, 0, 1)],
)
def test_model_usage_rejects_values_outside_the_counter_contract(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def test_normalized_response_exposes_provider_neutral_fields() -> None:
    tool_call = ModelToolCall(
        id="call_123",
        name="read_file",
        arguments='{"path":"CONTEXT.md"}',
    )
    response = ModelResponse(
        message=AssistantModelMessage(
            content="I will inspect it.",
            tool_calls=(tool_call,),
        ),
        usage=ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144),
        finish_reason="tool_calls",
    )

    assert response.message.content == "I will inspect it."
    assert response.message.tool_calls == (tool_call,)
    assert response.usage == ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144)
    assert response.finish_reason == "tool_calls"
    assert response.continuation is None
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "Inspect the project."},
        response.message.to_dict(),
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "read_file",
            "content": "project context",
        },
    ]
    assert messages[0]["role"] == "user"


def test_assistant_message_rejects_duplicate_tool_call_ids() -> None:
    first = ModelToolCall(
        id="duplicate-call",
        name="read_file",
        arguments='{"path":"a.txt"}',
    )
    second = ModelToolCall(
        id="duplicate-call",
        name="read_file",
        arguments='{"path":"b.txt"}',
    )

    with pytest.raises(ValueError, match="tool call IDs must be unique"):
        AssistantModelMessage(content="", tool_calls=(first, second))


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_model_response_rejects_empty_success(content: str) -> None:
    with pytest.raises(EmptyModelResponseError):
        ModelResponse(
            message=AssistantModelMessage(content=content),
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            finish_reason="stop",
        )


def test_stream_events_preserve_the_normalized_response_contract() -> None:
    response = ModelResponse(
        message=AssistantModelMessage(content="Done"),
        usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        finish_reason="stop",
    )

    text_delta = TextDelta(delta="I will")
    assert text_delta.type == "text_delta"
    assert text_delta.delta == "I will"
    completed = ModelCompleted(response=response)
    assert completed.type == "completed"
    assert completed.response is response


def test_reasoning_delta_is_distinct_from_text_and_rejects_empty_content() -> None:
    reasoning_delta = ReasoningDelta(delta="Thinking...")
    assert reasoning_delta.type == "reasoning_delta"
    assert reasoning_delta.delta == "Thinking..."

    with pytest.raises(ValueError, match="delta must not be empty"):
        ReasoningDelta(delta="")


def test_model_response_retains_opaque_continuation() -> None:
    continuation = ModelContinuation(
        provider_id="anthropic-default",
        payload=({"type": "thinking", "thinking": "private", "signature": "sig"},),
    )
    response = ModelResponse(
        message=AssistantModelMessage(content="Answer"),
        usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        finish_reason="stop",
        continuation=continuation,
    )

    assert response.continuation is continuation


def test_model_response_accepts_tool_call_without_text() -> None:
    tool_call = ModelToolCall(
        id="call_123",
        name="read_file",
        arguments='{"path":"README.md"}',
    )

    response = ModelResponse(
        message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        finish_reason="tool_calls",
    )

    assert response.message.tool_calls == (tool_call,)
