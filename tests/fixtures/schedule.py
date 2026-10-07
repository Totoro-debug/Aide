"""Schedule state and coordination fixtures outside the production interfaces."""

import asyncio
import json

from aide.agent.workspace_state import WorkspaceState
from aide.schedule.model import ScheduleJob


async def wait_until(predicate: object) -> None:
    """Yield up to 100 event-loop turns until a Schedule condition holds."""
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def write_schedule_state(state: WorkspaceState, *jobs: ScheduleJob) -> None:
    """Write canonical persisted Schedule Jobs before constructing the Store."""
    state.schedule_path.write_text(
        json.dumps(
            [job.to_dict() for job in jobs],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
