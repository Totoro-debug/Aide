from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.session.restore import RestoreResult
from myclaw.agent.tools.mcp_runtime import MCPStartupReport
from myclaw.agent.workspace_runtime import WorkspaceRuntime, WorkspaceRuntimeFactories
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleService
from tests.fixtures import FakeClock


class _Restore:
    recoveries = 0

    def __init__(self, _state: WorkspaceState) -> None:
        pass

    async def recover_pending(self) -> RestoreResult | None:
        type(self).recoveries += 1
        return None


class _Mcp:
    starts = 0
    closes = 0

    def __init__(self, _workspace: Path, **_kwargs: object) -> None:
        pass

    async def start(self, _configuration: object) -> MCPStartupReport:
        type(self).starts += 1
        return MCPStartupReport(snapshot=(), failed_servers=())

    async def close(self) -> None:
        type(self).closes += 1


class _Router:
    starts = 0
    closes = 0

    def __init__(self, **_kwargs: object) -> None:
        type(self).starts += 1

    def route_status(self, _route: str) -> object:
        return SimpleNamespace()

    async def close(self) -> None:
        type(self).closes += 1


class _Keywords:
    starts = 0

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def prepare(self, _tools: object, _servers: object) -> dict[str, tuple[str, ...]]:
        type(self).starts += 1
        return {}


class _Dream:
    closes = 0

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def run(self) -> object:
        return None

    async def close(self) -> None:
        type(self).closes += 1


class _Schedule:
    starts = 0
    preparations = 0
    closes = 0

    def __init__(self, **_kwargs: object) -> None:
        type(self).starts += 1

    def _prepare_start(self) -> None:
        return None

    async def register_dream_job(self, *, schedule: JobSchedule) -> None:
        assert isinstance(schedule, JobSchedule)
        type(self).preparations += 1

    async def pause_and_drain(self) -> None:
        return None

    async def drain_confirmation_aborts(self) -> None:
        return None

    async def close(self) -> None:
        type(self).closes += 1


def _configuration() -> Any:
    return SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(batch_size=10, schedule="0 * * * *"),
        runtime=SimpleNamespace(permission_level="workspace-write"),
    )


def _factories(schedule_service: Any = _Schedule) -> WorkspaceRuntimeFactories:
    return WorkspaceRuntimeFactories(
        workspace_state=WorkspaceState,
        restore_manager=_Restore,
        mcp_runtime=_Mcp,
        router=_Router,
        mcp_keyword_preparer=_Keywords,
        memory_manager=MemoryManager,
        dream=_Dream,
        schedule_service=schedule_service,
    )


@pytest.mark.asyncio
async def test_workspace_runtime_shares_real_directory_owner_and_lifecycle(
    tmp_path: Path,
) -> None:
    _Restore.recoveries = 0
    _Mcp.starts = 0
    _Mcp.closes = 0
    _Router.starts = 0
    _Router.closes = 0
    _Keywords.starts = 0
    _Dream.closes = 0
    _Schedule.starts = 0
    _Schedule.preparations = 0
    _Schedule.closes = 0

    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = workspace / "."
    configuration = _configuration()

    async def execute_job(_job: object) -> None:
        return None

    async def execute_occurrence(_occurrence: object) -> None:
        return None

    async def cancel_confirmation(_owner: object) -> None:
        return None

    kwargs = {
        "workspace": workspace,
        "agent_home": agent_home,
        "configuration": configuration,
        "execute_user_job": execute_job,
        "execute_user_occurrence": execute_occurrence,
        "cancel_confirmation_owner": cancel_confirmation,
        "configured_schedule_level": "workspace-write",
        "resolved_exec_shell": SimpleNamespace(family="powershell"),
        "now": lambda: datetime(2026, 9, 29, tzinfo=UTC),
        "timezone_name": "UTC",
        "provider_factory": lambda _configuration: None,
        "built_in_names": (),
        "factories": _factories(),
    }

    first = WorkspaceRuntime.acquire(**kwargs)
    second = WorkspaceRuntime.acquire(**{**kwargs, "workspace": alias})

    assert first is second
    await first.start()
    await second.start()

    assert first.workspace_path == workspace.resolve()
    assert first.workspace_state.workspace_path == workspace.resolve()
    assert await first.memory_manager.read_long_term()
    assert first.memory_manager is second.memory_manager
    await first.memory_manager.append_summary("Session A update", datetime(2026, 9, 29, tzinfo=UTC))
    await second.memory_manager.append_summary(
        "Session B update", datetime(2026, 9, 29, tzinfo=UTC)
    )
    summaries = await first.memory_manager.claim_summaries(limit=2)
    assert tuple(entry.content for entry in summaries.entries) == (
        "Session A update",
        "Session B update",
    )
    assert _Restore.recoveries == 1
    assert _Mcp.starts == 1
    assert _Router.starts == 1
    assert _Keywords.starts == 1
    assert _Schedule.starts == 1

    await first.prepare_schedule(JobSchedule.from_cron_input("0 * * * *", "UTC"))
    await second.prepare_schedule(JobSchedule.from_cron_input("0 * * * *", "UTC"))
    assert _Schedule.preparations == 1

    await first.close()
    assert _Mcp.closes == 1
    assert _Router.closes == 1
    assert _Schedule.closes == 1
    assert _Dream.closes == 1

    replacement = WorkspaceRuntime.acquire(**kwargs)
    assert replacement is not first
    await replacement.start()
    assert _Restore.recoveries == 2
    await replacement.close()


@pytest.mark.asyncio
async def test_workspace_runtime_reuses_directory_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-alias"
    if os.name == "nt":
        created = subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(alias), str(workspace)),
            capture_output=True,
            check=False,
        )
        if created.returncode:
            pytest.skip("host cannot create a directory junction")
    else:
        try:
            alias.symlink_to(workspace, target_is_directory=True)
        except OSError:
            pytest.skip("host cannot create a directory symlink")

    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()

    async def execute_job(_job: ScheduleJob) -> None:
        return None

    first = WorkspaceRuntime.acquire(
        workspace=workspace,
        agent_home=agent_home,
        configuration=_configuration(),
        execute_user_job=execute_job,
    )
    second = WorkspaceRuntime.acquire(
        workspace=alias,
        agent_home=agent_home,
        configuration=_configuration(),
        execute_user_job=execute_job,
    )
    assert second is first
    assert second.workspace_path == workspace.resolve()
    await first.close()


def test_workspace_runtime_factory_has_public_resource_contract() -> None:
    names = {field.name for field in fields(WorkspaceRuntimeFactories)}
    assert names == {
        "workspace_state",
        "restore_manager",
        "mcp_runtime",
        "router",
        "mcp_keyword_preparer",
        "memory_manager",
        "dream",
        "schedule_service",
    }


@pytest.mark.asyncio
async def test_workspace_runtime_dispatches_one_due_job_through_one_schedule_owner(
    tmp_path: Path,
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    clock = FakeClock(datetime(2026, 9, 29, tzinfo=UTC))
    calls: list[str] = []
    finished = asyncio.Event()

    async def execute_job(job: ScheduleJob) -> None:
        calls.append(job.job_id)
        finished.set()

    async def execute_occurrence(occurrence: Any) -> None:
        calls.append(occurrence.job.job_id)
        finished.set()

    async def cancel_confirmation(_owner: object) -> None:
        return None

    def no_provider(_configuration: Any) -> Any:
        return None

    runtime = WorkspaceRuntime.acquire(
        workspace=workspace,
        agent_home=agent_home,
        configuration=_configuration(),
        execute_user_job=execute_job,
        execute_user_occurrence=execute_occurrence,
        cancel_confirmation_owner=cancel_confirmation,
        configured_schedule_level="workspace-write",
        resolved_exec_shell=SimpleNamespace(family="powershell"),
        now=clock.now,
        timezone_name="UTC",
        provider_factory=no_provider,
        built_in_names=(),
        schedule_clock=clock,
        factories=_factories(ScheduleService),
    )
    await runtime.start()
    await runtime.prepare_schedule(JobSchedule.every(3600))

    job = ScheduleJob(
        job_id=str(UUID("550e8400-e29b-41d4-a716-446655440000")),
        message="Run once.",
        schedule=JobSchedule.at("2026-09-28T00:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await runtime.schedule_service.add_user_job(job)
    runtime.schedule_service.start()
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0)

    assert calls == [job.job_id]
    await runtime.close()
