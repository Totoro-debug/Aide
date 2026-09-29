from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from myclaw.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import (
    DreamExecutor,
    ScheduleJobExecutionError,
    ScheduleJobExecutor,
    ScheduleService,
)
from myclaw.schedule.store import WorkspaceScheduleStore

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


async def _noop_user_job(job: ScheduleJob) -> None:
    del job


async def _noop_dream() -> object:
    return None


def _service(
    state: WorkspaceState,
    *,
    execute_user_job: ScheduleJobExecutor = _noop_user_job,
    execute_dream: DreamExecutor = _noop_dream,
) -> ScheduleService:
    return ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )


async def _wait_until(predicate: object) -> None:
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_schedule_service_facade_preserves_user_job_management(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    service = _service(state)
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Review the project.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )

    assert await service.add_user_job(job) == job
    assert await service.public_snapshot() == (job,)
    assert await service.remove_user_job(job.job_id, expected=job) is True
    assert await service.public_snapshot() == ()


@pytest.mark.asyncio
async def test_run_gateway_views_isolate_catalog_and_exposure(
    workspace: Path,
    agent_home: Path,
) -> None:
    identity = workspace
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    service = _service(state)
    generation = ToolGateway(workspace=identity, schedule_service=service)
    scheduled = generation.for_run(
        excluded_names=("schedule",),
        exposed_names=("read_file",),
    )
    foreground = generation.for_run(exposed_names=("schedule",))

    scheduled.expose(("exec",))
    assert scheduled.exposed_names == ("read_file", "exec")
    assert foreground.exposed_names == ("schedule",)
    assert "schedule" not in {schema["function"]["name"] for schema in scheduled.schemas}

    unavailable = await scheduled.call(
        ModelToolCall(id="scheduled_add", name="schedule", arguments='{"action":"list"}')
    )
    added = await foreground.call(
        ModelToolCall(
            id="foreground_add",
            name="schedule",
            arguments=json.dumps({"action": "add", "message": "foreground", "every_seconds": 60}),
        )
    )

    assert (unavailable.status, unavailable.content) == (
        "error",
        "The requested tool is not available.",
    )
    assert added.status == "success"
    assert len(await service.public_snapshot()) == 1


@pytest.mark.asyncio
async def test_schedule_service_start_rejects_unbound_callback_before_reservation(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    service = _service(state)
    service.start()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await service._store.snapshot() == (job,)
    assert service.status_snapshot().active_job_count == 0


@pytest.mark.asyncio
async def test_schedule_service_maps_structured_callback_failure_without_leaking_details(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="6fa459ea-ee8a-4ca4-894e-db77e160355e",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    started = asyncio.Event()

    async def callback(active_job: ScheduleJob) -> None:
        assert active_job == job
        started.set()
        raise ScheduleJobExecutionError(
            ErrorInfo(code="model_failed", message="safe model failure")
        )

    service = _service(state, execute_user_job=callback)
    service.start()
    await started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    saved = (await service._store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "safe model failure"


@pytest.mark.asyncio
async def test_schedule_service_callback_cancellation_leaves_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="9ba7b810-9dad-41d1-80b4-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    started = asyncio.Event()

    async def callback(active_job: ScheduleJob) -> None:
        assert active_job == job
        started.set()
        raise asyncio.CancelledError()

    service = _service(state, execute_user_job=callback)
    service.start()
    await started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await store.snapshot() == (job,)


@pytest.mark.asyncio
async def test_pause_admission_stops_dispatch_without_cancelling_active_job(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440001",
        message="Run this once.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    started = asyncio.Event()
    release = asyncio.Event()

    async def callback(active_job: ScheduleJob) -> None:
        assert active_job == job
        started.set()
        await release.wait()

    service = _service(state, execute_user_job=callback)
    service.start()
    await started.wait()
    await service.pause_admission()
    assert service._loop_task is None
    assert service.status_snapshot().active_job_count == 1
    release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()
