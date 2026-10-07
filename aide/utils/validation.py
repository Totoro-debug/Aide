"""Shared validation for runtime values and persisted records."""

from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from typing import Literal, cast
from uuid import UUID

_TOKEN_USAGE_FIELDS = frozenset({"model_calls", "input_tokens", "output_tokens", "total_tokens"})

type TokenUsageValidationIssue = Literal["fields", "values", "total"]


def empty_token_usage() -> dict[str, int]:
    """Return a fresh zero value for the shared token usage contract."""
    return {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def token_usage_validation_issue(
    value: Mapping[str, object],
) -> TokenUsageValidationIssue | None:
    """Classify violations of the shared four-field token usage contract."""
    if set(value) != _TOKEN_USAGE_FIELDS:
        return "fields"
    if any(
        type(member) is not int or member < 0
        for member in (value[field] for field in _TOKEN_USAGE_FIELDS)
    ):
        return "values"
    input_tokens = cast(int, value["input_tokens"])
    output_tokens = cast(int, value["output_tokens"])
    total_tokens = cast(int, value["total_tokens"])
    if total_tokens != input_tokens + output_tokens:
        return "total"
    return None


def require_nonnegative_int(value: int, *, field: str) -> None:
    """Validate persisted counters without accepting bool as an integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field} must be a nonnegative integer"
        raise ValueError(msg)


def require_nonnegative_number(value: float, *, field: str) -> None:
    """Validate a finite nonnegative numeric measurement without accepting bool."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or not isfinite(value)
    ):
        msg = f"{field} must be a finite nonnegative number"
        raise ValueError(msg)


def require_aware_datetime(value: datetime, *, field: str) -> None:
    """Reject datetimes that cannot carry the required persisted UTC offset."""
    if value.utcoffset() is None:
        msg = f"{field} must be timezone-aware"
        raise ValueError(msg)


def require_uuid4(value: UUID, *, field: str) -> None:
    """Reject UUIDs outside the canonical lowercase RFC 4122 UUID4 form."""
    if value.version != 4 or str(value) != str(value).lower():
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg)


def require_uuid4_string(value: str, *, field: str) -> None:
    """Validate a persisted lowercase, hyphenated UUID4 string."""
    try:
        parsed = UUID(value)
        require_uuid4(parsed, field=field)
    except (ValueError, AttributeError) as exc:
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg) from exc
    if str(parsed) != value:
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg)
