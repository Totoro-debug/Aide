from datetime import datetime, timedelta, timezone

import pytest

from myclaw.memory.records import SummaryEntry

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_summary_entry_serializes_with_exactly_three_keys() -> None:
    entry = SummaryEntry(
        index=1,
        timestamp=datetime(2026, 7, 11, 16, tzinfo=LOCAL_OFFSET),
        content="The user is implementing MyClaw and prefers a file-first architecture.",
    )

    assert entry.to_json_line() == (
        '{"index":1,"timestamp":"2026-07-11T16:00:00.000+08:00",'
        '"content":"The user is implementing MyClaw and prefers a file-first architecture."}\n'
    )
    assert set(entry.to_dict()) == {"index", "timestamp", "content"}


def test_summary_entry_rejects_invalid_index_time_and_content() -> None:
    with pytest.raises(ValueError, match="index must start at 1"):
        SummaryEntry(index=0, timestamp=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET), content="x")
    with pytest.raises(ValueError, match="timezone-aware"):
        SummaryEntry(index=1, timestamp=datetime(2026, 7, 11), content="x")
    with pytest.raises(ValueError, match="content"):
        SummaryEntry(index=1, timestamp=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET), content="")
