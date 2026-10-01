from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import pytest_asyncio

from myclaw.agent.workspace_runtime import WorkspaceRuntime
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.service.errors import ServiceError
from myclaw.service.runtime import LocalService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.service.test_service_concurrency import _CollectingSink


@pytest_asyncio.fixture
async def initial_service(tmp_path: Path) -> AsyncIterator[LocalService]:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    service = LocalService(home, reconnect_timeout=3600)
    for name in ("first", "second"):
        path = tmp_path / name
        path.mkdir()
        service.projects.register(path)
    await service.start()
    yield service
    await service.stop()


def _save_first_configuration(service: LocalService) -> None:
    (service.agent_home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service.config_view()


async def _application_finished(service: LocalService) -> None:
    task = service._config_apply_task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=10)


def _assert_no_runtime_owners(service: LocalService) -> None:
    assert not service.workspaces
    assert not service._initial_configuration_candidates
    assert all(
        os.path.normcase(str(record.path.resolve())) not in WorkspaceRuntime._registry
        for record in service.projects.list()
    )


@pytest.mark.asyncio
async def test_failed_second_schedule_preparation_closes_all_unpublished_resources(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = initial_service
    original = WorkspaceRuntime.prepare_schedule
    prepared: list[WorkspaceRuntime] = []

    async def fail_second(runtime: WorkspaceRuntime, schedule: JobSchedule) -> None:
        prepared.append(runtime)
        if len(prepared) == 2:
            raise OSError("Schedule persistence failed")
        await original(runtime, schedule)

    monkeypatch.setattr(WorkspaceRuntime, "prepare_schedule", fail_second)
    _save_first_configuration(service)
    await _application_finished(service)
    assert service.configuration is None
    assert service._config_status == "failed-to-apply"
    assert len(prepared) == 2
    assert all(runtime._closed for runtime in prepared)
    _assert_no_runtime_owners(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_with", ["retry", "stop"])
async def test_failed_initial_cleanup_retains_owner_until_retry_or_stop(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch, finish_with: str
) -> None:
    service = initial_service
    original_close = WorkspaceRuntime.close
    close_attempts = 0

    async def fail_schedule(_runtime: WorkspaceRuntime, _schedule: JobSchedule) -> None:
        raise OSError("Schedule preparation failed")

    async def fail_close_once(
        runtime: WorkspaceRuntime,
        *,
        close_foreground: Callable[[], Awaitable[None]] | None = None,
        drain_confirmation_aborts: bool = True,
    ) -> None:
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise OSError("Resource cleanup failed")
        await original_close(
            runtime,
            close_foreground=close_foreground,
            drain_confirmation_aborts=drain_confirmation_aborts,
        )

    monkeypatch.setattr(WorkspaceRuntime, "prepare_schedule", fail_schedule)
    monkeypatch.setattr(WorkspaceRuntime, "close", fail_close_once)
    _save_first_configuration(service)
    await _application_finished(service)
    assert service.configuration is None
    assert len(service._initial_configuration_candidates) == 1
    retained = service._initial_configuration_candidates[0]
    assert retained.runtime is not None and not retained.runtime._closed
    if finish_with == "retry":
        monkeypatch.undo()
        await service.retry_configuration(
            "retry-initial", cast(str, service.config_view()["revision"])
        )
        await _application_finished(service)
        assert service.configuration is not None
        assert len(service.workspaces) == 2
        assert retained.runtime._closed
        assert not service._initial_configuration_candidates
    else:
        await service.stop()
        assert retained.runtime._closed
        _assert_no_runtime_owners(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["external-invalid", "new-save", "stop"])
async def test_first_activation_rechecks_saved_bytes_and_closes_cancelled_candidates(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    service = initial_service
    started, release = asyncio.Event(), asyncio.Event()
    original = WorkspaceRuntime.prepare_schedule
    block_once = True

    async def blocked(runtime: WorkspaceRuntime, schedule: JobSchedule) -> None:
        nonlocal block_once
        if block_once:
            block_once = False
            started.set()
            await release.wait()
        await original(runtime, schedule)

    monkeypatch.setattr(WorkspaceRuntime, "prepare_schedule", blocked)
    _save_first_configuration(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    assert not service.configuration_ready
    assert not service.workspaces
    if change == "stop":
        await asyncio.wait_for(service.stop(), timeout=10)
        _assert_no_runtime_owners(service)
        return
    if change == "external-invalid":
        (service.agent_home.path / "config.toml").write_bytes(b"[broken")
    else:
        await service.update_configuration(
            "newer-save",
            cast(str, service.config_view()["revision"]),
            {"runtime": {"max_iterations": 83}},
        )
    release.set()
    await _application_finished(service)
    if change == "external-invalid":
        assert service.configuration is None
        assert service._config_status == "pending-repair"
        _assert_no_runtime_owners(service)
    else:
        assert service.configuration_ready
        assert service.configuration is not None
        assert service.configuration.runtime.max_iterations == 83
        assert len(service.workspaces) == 2
        assert all(
            workspace.configuration.runtime.max_iterations == 83
            for workspace in service.workspaces.values()
        )


@pytest.mark.asyncio
async def test_project_removal_serializes_with_initial_candidate_publication(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = initial_service
    client = await service.register_client("web")
    record = service.projects.list()[0]
    started, release = asyncio.Event(), asyncio.Event()
    original = WorkspaceRuntime.prepare_schedule
    block_once = True

    async def blocked(runtime: WorkspaceRuntime, schedule: JobSchedule) -> None:
        nonlocal block_once
        if block_once:
            block_once = False
            started.set()
            await release.wait()
        await original(runtime, schedule)

    monkeypatch.setattr(WorkspaceRuntime, "prepare_schedule", blocked)
    _save_first_configuration(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    removal = asyncio.create_task(service.remove_project(client.client_id, record.project_id))
    await asyncio.sleep(0)
    assert not removal.done()
    release.set()
    await _application_finished(service)
    await asyncio.wait_for(removal, timeout=10)
    assert service.configuration_ready
    assert len(service.workspaces) == 1
    assert all(runtime.workspace_path != record.path for runtime in service.workspaces.values())
    assert os.path.normcase(str(record.path.resolve())) not in WorkspaceRuntime._registry


@pytest.mark.asyncio
async def test_save_after_first_publication_is_applied_by_same_application_task(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = initial_service
    published, release = asyncio.Event(), asyncio.Event()
    original = service._reconcile_schedule_admission
    block_once = True

    async def blocked() -> None:
        nonlocal block_once
        if block_once and service.configuration_ready:
            block_once = False
            published.set()
            await release.wait()
        await original()

    monkeypatch.setattr(service, "_reconcile_schedule_admission", blocked)
    _save_first_configuration(service)
    await asyncio.wait_for(published.wait(), timeout=10)
    await service.update_configuration(
        "save-after-publication",
        cast(str, service.config_view()["revision"]),
        {"runtime": {"max_iterations": 83}},
    )
    release.set()
    await _application_finished(service)
    assert service.configuration_ready
    assert service.configuration is not None
    assert service.configuration.runtime.max_iterations == 83
    assert all(
        workspace.configuration.runtime.max_iterations == 83
        for workspace in service.workspaces.values()
    )


@pytest.mark.asyncio
async def test_external_invalid_file_preserves_old_generation_project_and_schedule_admission(
    initial_service: LocalService,
) -> None:
    service = initial_service
    client = await service.register_client("web")
    await service.connect_client(client.client_id, _CollectingSink())
    _save_first_configuration(service)
    await _application_finished(service)
    owners = tuple(service.workspaces.values())
    previous_configuration = service.configuration
    assert all(workspace.schedule_admitted for workspace in owners)
    (service.agent_home.path / "config.toml").write_bytes(b"[broken")
    view = service.config_view()
    assert cast(dict[str, object], view["application"])["status"] == "failed-to-apply"
    assert service.configuration_ready
    assert service.configuration is previous_configuration
    await service._reconcile_schedule_admission()
    assert all(workspace.schedule_admitted for workspace in owners)
    attached = await service.attach_workspace(client.client_id, owners[0].workspace_path)
    assert attached is owners[0]
    session_id = await attached.create_draft(client.client_id)
    claim = await attached.claim(client.client_id, session_id)
    assert claim.loop.session.session_id == session_id


@pytest.mark.asyncio
async def test_stop_can_retry_failed_cleanup_of_unpublished_initial_owner(
    initial_service: LocalService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = initial_service

    async def fail_schedule(_runtime: WorkspaceRuntime, _schedule: JobSchedule) -> None:
        raise OSError("Schedule preparation failed")

    async def fail_close(_runtime: WorkspaceRuntime, **_kwargs: object) -> None:
        raise OSError("Resource cleanup failed")

    monkeypatch.setattr(WorkspaceRuntime, "prepare_schedule", fail_schedule)
    monkeypatch.setattr(WorkspaceRuntime, "close", fail_close)
    _save_first_configuration(service)
    await _application_finished(service)
    with pytest.raises(ServiceError, match="workspace cleanup error"):
        await service.stop()
    assert service.state == "stopped"
    assert service._stop_failed
    assert service._initial_configuration_candidates
    monkeypatch.undo()
    await service.stop()
    assert not service._stop_failed
    _assert_no_runtime_owners(service)


@pytest.mark.asyncio
async def test_first_activation_owns_accessible_projects_without_resuming_saved_jobs(
    initial_service: LocalService, tmp_path: Path
) -> None:
    service = initial_service
    waiting_path = tmp_path / "waiting"
    waiting_path.mkdir()
    waiting = service.projects.register(waiting_path, schedule_state="awaiting_resume")
    state = WorkspaceState(waiting_path)
    state.initialize(agent_home_root=service.agent_home.path)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="Do not run automatically",
        schedule=JobSchedule.every(1),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    saved = (await store.public_snapshot())[0].to_dict()
    for name, permission in [("missing", "unavailable"), ("removing", "removing")]:
        path = tmp_path / name
        path.mkdir()
        record = service.projects.register(path)
        service.projects.set_schedule_state(record.project_id, permission)
        if name == "missing":
            path.rmdir()
    client = await service.register_client("web")
    await service.connect_client(client.client_id, _CollectingSink())
    _save_first_configuration(service)
    await _application_finished(service)
    assert len(service.workspaces) == 3
    owner = next(
        owner for owner in service.workspaces.values() if owner.workspace_path == waiting_path
    )
    assert not owner.schedule_admitted
    assert (await store.public_snapshot())[0].to_dict() == saved
    attached = await service.attach_workspace(client.client_id, waiting_path)
    assert attached is owner
    assert (
        attached.runtime
        is WorkspaceRuntime._registry[os.path.normcase(str(waiting_path.resolve()))]
    )
    await service.stop()
    restarted = LocalService(service.agent_home, reconnect_timeout=3600)
    try:
        await restarted.start()
        new_client = await restarted.register_client("web")
        await restarted.connect_client(new_client.client_id, _CollectingSink())
        record = next(
            record
            for record in restarted.projects.list()
            if record.project_id == waiting.project_id
        )
        assert record.schedule_state == "awaiting_resume"
        reopened = await restarted.attach_workspace(new_client.client_id, waiting_path)
        assert not reopened.schedule_admitted
        assert (await store.public_snapshot())[0].to_dict() == saved
    finally:
        await restarted.stop()
