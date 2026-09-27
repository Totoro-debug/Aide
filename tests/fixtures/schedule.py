"""Schedule state fixtures that stay outside the production Store interface."""

import json

from myclaw.schedule.model import ScheduleJob
from myclaw.workspace.state import WorkspaceState


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
