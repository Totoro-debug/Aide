"""Frozen time formatting used by persisted records and runtime events."""

from datetime import datetime

from aide.utils.validation import require_aware_datetime


def local_now() -> datetime:
    """Return the current local aware datetime."""
    return datetime.now().astimezone()


def format_rfc3339_milliseconds(value: datetime) -> str:
    """Format an aware datetime using the frozen persisted time representation."""
    require_aware_datetime(value, field="timestamp")
    return value.isoformat(timespec="milliseconds")
