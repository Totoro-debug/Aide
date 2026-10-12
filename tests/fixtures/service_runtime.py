"""White-box observations for service lifecycle and resource tests."""

from collections.abc import Mapping
from pathlib import Path

from aide.schedule.model import ScheduleJob
from aide.service.projects import ProjectRecord
from aide.service.runtime import AgentService, WorkspaceRecord
from aide.service.runtime.records import _LoopState


def workspace_loops(workspace: WorkspaceRecord) -> Mapping[str, _LoopState]:
    return workspace._loops


async def register_project(
    service: AgentService, client_id: str, path: Path,
) -> tuple[ProjectRecord, WorkspaceRecord, tuple[ScheduleJob, ...]]:
    result = await service.register_project_entry(client_id, path)
    record = next(record for record in service.projects.list()
                  if record.project_id == result["project_id"])
    workspace = service._workspaces[result["workspace_id"]]
    jobs = await workspace.schedule_service.public_snapshot()
    assert [job.job_id for job in jobs] == [job["job_id"] for job in result["saved_jobs"]]
    return record, workspace, jobs
