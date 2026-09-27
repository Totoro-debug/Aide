"""Conversation Summary persisted records."""

import json
from dataclasses import dataclass
from datetime import datetime

from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_aware_datetime, require_nonnegative_int


@dataclass(frozen=True, slots=True)
class SummaryEntry:
    """One exact record in the global Conversation Summary JSONL stream."""

    index: int
    timestamp: datetime
    content: str

    def __post_init__(self) -> None:
        require_nonnegative_int(self.index, field="index")
        if self.index < 1:
            msg = "index must start at 1"
            raise ValueError(msg)
        require_aware_datetime(self.timestamp, field="timestamp")
        if not self.content:
            msg = "content must not be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "timestamp": format_rfc3339_milliseconds(self.timestamp),
            "content": self.content,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
