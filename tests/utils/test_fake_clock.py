from datetime import UTC, datetime, timedelta

import pytest

from tests.fixtures.clock import FakeClock


@pytest.mark.asyncio
async def test_fake_clock_advances_wall_and_monotonic_time_without_waiting() -> None:
    start = datetime(2026, 7, 12, 8, 0, tzinfo=UTC)
    clock = FakeClock(start)

    await clock.sleep(1.25)
    clock.advance(0.75)

    assert clock.now() == start + timedelta(seconds=2)
    assert clock.monotonic() == 2.0
    assert clock.sleeps == [1.25]
