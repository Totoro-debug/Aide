"""Normalized Model Provider failures."""

from math import isfinite

from aide.utils.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo


class EmptyModelResponseError(ValueError):
    """A successful Model Response contained neither text nor Tool calls."""


class ModelCallError(Exception):
    """A normalized Model Provider failure handled by the Model Router."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


def parse_retry_after_seconds(value: object) -> float | None:
    """Parse a finite, nonnegative Retry-After scalar as seconds."""
    try:
        seconds = float(str(value))
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 and isfinite(seconds) else None


def model_context_overflow_error() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_context_overflow",
            message=MODEL_CONTEXT_OVERFLOW_MESSAGE,
        )
    )
