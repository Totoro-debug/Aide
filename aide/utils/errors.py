"""Stable user-visible error values."""

from dataclasses import dataclass
from typing import Literal

from aide.utils.validation import require_nonnegative_number

type ErrorCode = Literal[
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
]

MODEL_CONTEXT_OVERFLOW_MESSAGE = "Model request context exceeds the available input budget."

STABLE_ERROR_CODES: frozenset[str] = frozenset(
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

TURN_CANCELLED_MESSAGE = "Aide 已取消本轮对话。"


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A safe runtime error that may cross a Port boundary."""

    code: ErrorCode
    message: str
    retryable: bool = False
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.code not in STABLE_ERROR_CODES:
            msg = "code must be a stable error code"
            raise ValueError(msg)
        if not self.message:
            msg = "message must not be empty"
            raise ValueError(msg)
        if not isinstance(self.retryable, bool):
            msg = "retryable must be a boolean"
            raise ValueError(msg)
        if self.retry_after_seconds is not None:
            require_nonnegative_number(
                self.retry_after_seconds,
                field="retry_after_seconds",
            )
