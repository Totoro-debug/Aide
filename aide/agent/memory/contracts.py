"""Memory result values independent of execution implementations."""

from dataclasses import dataclass

from aide.utils.errors import ErrorInfo
from aide.utils.validation import require_nonnegative_int


@dataclass(frozen=True, slots=True)
class DreamResult:
    """Observable summary returned by one Dream execution."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            raise ValueError("memory_updated must be a boolean")
