from typing import Any, cast

import pytest

from omni.errors import (
    MODEL_CONTEXT_OVERFLOW_MESSAGE,
    STABLE_ERROR_CODES,
    TURN_CANCELLED_MESSAGE,
    ErrorCode,
    ErrorInfo,
)
from omni.provider.errors import (
    ModelCallError,
    model_context_overflow_error,
    parse_retry_after_seconds,
)


def test_turn_cancelled_message_is_the_stable_user_visible_contract() -> None:
    assert TURN_CANCELLED_MESSAGE == "Omni 已取消本轮对话。"


def test_model_context_overflow_error_returns_fresh_normalized_failures() -> None:
    first = model_context_overflow_error()
    second = model_context_overflow_error()

    assert isinstance(first, ModelCallError)
    assert isinstance(second, ModelCallError)
    assert first is not second
    assert first.error is not second.error
    assert first.error.code == second.error.code == "model_context_overflow"
    assert first.error.message == second.error.message == MODEL_CONTEXT_OVERFLOW_MESSAGE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (5, 5.0),
        ("4.25", 4.25),
        (-1, None),
        ("NaN", None),
        ("Infinity", None),
        ("not-a-number", None),
    ],
)
def test_parse_retry_after_seconds_accepts_only_finite_nonnegative_values(
    value: object,
    expected: float | None,
) -> None:
    assert parse_retry_after_seconds(value) == expected


def test_error_info_uses_the_frozen_structure_and_code_vocabulary() -> None:
    assert STABLE_ERROR_CODES == frozenset(
        {
            "config_missing",
            "config_parse_error",
            "config_invalid",
            "persistence_error",
            "schedule_state_error",
            "route_unavailable",
            "provider_auth_error",
            "provider_rate_limited",
            "provider_timeout",
            "provider_unavailable",
            "model_invalid_request",
            "model_context_overflow",
            "interactive_terminal_required",
            "model_failed",
            "agent_iteration_limit",
            "turn_cancelled",
            "tool_not_found",
            "tool_invalid_arguments",
            "tool_denied",
            "tool_refused",
            "tool_failed",
            "memory_task_running",
            "skill_reload_failed",
        }
    )
    assert MODEL_CONTEXT_OVERFLOW_MESSAGE == (
        "Model request context exceeds the available input budget."
    )
    error = ErrorInfo(
        code="provider_rate_limited",
        message="Provider rate limit reached.",
        retryable=True,
        retry_after_seconds=1.5,
    )

    assert error.code == "provider_rate_limited"
    assert error.message == "Provider rate limit reached."
    assert error.retryable is True
    assert error.retry_after_seconds == 1.5
    with pytest.raises(ValueError, match="stable error code"):
        ErrorInfo(code=cast(ErrorCode, "new_unaccepted_code"), message="Not accepted.")


@pytest.mark.parametrize(
    "retry_after_seconds",
    [True, -1, float("nan"), float("inf"), "1"],
)
def test_error_info_rejects_invalid_retry_after_seconds(
    retry_after_seconds: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^retry_after_seconds must be a finite nonnegative number$",
    ):
        ErrorInfo(
            code="provider_rate_limited",
            message="Provider rate limit reached.",
            retry_after_seconds=retry_after_seconds,
        )
