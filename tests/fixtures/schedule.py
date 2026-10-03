"""Schedule state fixtures that stay outside the production Store interface."""

import json

from omni.agent.workspace_state import WorkspaceState
from omni.schedule.model import ScheduleJob


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
