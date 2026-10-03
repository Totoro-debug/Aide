from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

import myclaw.service.runtime as service_runtime
from myclaw.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationAborted,
    ConfirmationEnvelope,
    ForegroundConfirmationOwner,
)
from myclaw.agent.loop import AgentLoop
from myclaw.agent.session.restore import RestoreManager, RestoreMode
from myclaw.agent.session.session import Session
from myclaw.agent.tools.tool_gateway import ConfirmationRequest
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleOccurrence, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.service.client import RemoteControl, ServiceClient
from myclaw.service.discovery import (
    SERVICE_PROTOCOL_VERSION,
    ServiceDiscovery,
    create_credential,
    read_credential,
    read_discovery,
    startup_lock,
    write_discovery,
)
from myclaw.service.errors import ServiceError
from myclaw.service.projects import ProjectCatalog, ProjectCatalogError
from myclaw.service.runtime import LocalService, WorkspaceServiceRuntime
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock
from tests.service.test_protocol_contract import _validator
from tests.service.test_service_concurrency import _CollectingSink


def test_discovery_file_has_no_credential_and_round_trips_atomically(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    token = create_credential(home)
    write_discovery(
        home,
        ServiceDiscovery("instance-1", SERVICE_PROTOCOL_VERSION, "127.0.0.1", 8765, 42),
    )

    loaded = read_discovery(home)
    assert loaded is not None
    assert loaded.service_instance_id == "instance-1"
    assert token == read_credential(home)
    assert token not in (home.path / "service.json").read_text(encoding="utf-8")


def test_startup_lock_is_reentrant_across_sequential_starters(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    with startup_lock(home):
        assert (home.path / "service.lock").exists()
    with startup_lock(home):
        pass


def test_project_catalog_deduplicates_resolved_aliases_and_preserves_missing_records(
    tmp_path: Path,
) -> None:
    home = AgentHome(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    catalog = ProjectCatalog(home)

    first = catalog.register(project)
    second = catalog.register(project / ".")
    assert second == first
    assert len(catalog.list()) == 1

    project.rmdir()
    reopened = ProjectCatalog(home)
    record = reopened.list()[0]
    assert record.project_id == first.project_id
    assert not record.path.exists()


def test_project_catalog_rejects_agent_home_overlap_and_invalid_file(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    catalog = ProjectCatalog(home)
    home.initialize()
    with pytest.raises(ProjectCatalogError):
        catalog.register(home.path)
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ProjectCatalogError):
        catalog.register(file_path)


def test_project_catalog_deduplicates_a_native_directory_alias(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    project = tmp_path / "project"
    alias = tmp_path / "project-alias"
    project.mkdir()
    try:
        alias.symlink_to(project, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    catalog = ProjectCatalog(home)
    first = catalog.register(project)

    assert catalog.register(alias) == first
    assert len(catalog.list()) == 1


def test_project_catalog_deduplicates_a_windows_junction(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    project = tmp_path / "project"
    alias = tmp_path / "project-junction"
    project.mkdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(project)], check=True)
    catalog = ProjectCatalog(home)

    first = catalog.register(project)
    assert catalog.register(alias) == first
    assert len(ProjectCatalog(home).list()) == 1


@pytest.mark.parametrize("path", ["relative/project", "."])
def test_project_catalog_rejects_relative_persisted_paths(tmp_path: Path, path: str) -> None:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    original = json.dumps(
        {"format_version": 1, "projects": [{"project_id": "project", "path": path}]}
    ).encode()
    catalog = ProjectCatalog(home)
    catalog.path.write_bytes(original)

    with pytest.raises(ProjectCatalogError):
        catalog.list()
    assert catalog.path.read_bytes() == original


def test_project_catalog_rejects_persisted_agent_home_overlap(tmp_path: Path) -> None:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    catalog = ProjectCatalog(home)
    original = json.dumps(
        {"format_version": 1, "projects": [{"project_id": "project", "path": str(home.path)}]}
    ).encode()
    catalog.path.write_bytes(original)

    with pytest.raises(ProjectCatalogError):
        catalog.list()
    assert catalog.path.read_bytes() == original


def test_project_catalog_keeps_previous_publication_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "agent-home")
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    catalog = ProjectCatalog(home)
    catalog.register(first_path)
    original = catalog.path.read_bytes()

    original_replace = os.replace

    def fail_catalog_replace(source: Any, target: Any) -> None:
        if Path(os.fspath(target)) == catalog.path:
            raise OSError("injected publication failure")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_catalog_replace)
    with pytest.raises(OSError):
        catalog.register(second_path)

    assert catalog.path.read_bytes() == original
    assert [record.path for record in catalog.list()] == [first_path.resolve()]
    assert [record.path for record in ProjectCatalog(home).list()] == [first_path.resolve()]


def test_project_catalog_reports_corruption_without_replacing_the_original_file(
    tmp_path: Path,
) -> None:
    home = AgentHome(tmp_path / "agent-home")
    catalog = ProjectCatalog(home)
    home.initialize()
    original = b'{"format_version": 1, "projects": ['
    catalog.path.write_bytes(original)

    with pytest.raises(ProjectCatalogError):
        catalog.list()

    assert catalog.path.read_bytes() == original


def test_discovery_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        ServiceDiscovery.from_dict(
            json.loads(
                '{"service_instance_id":"x","protocol_version":1,'
                '"host":"127.0.0.1","port":8765,"pid":1,"token":"secret"}'
            )
        )


def test_remote_control_handles_completion_before_acceptance() -> None:
    control = RemoteControl(cast(ServiceClient, object()))
    control.finish_run("fast-run")
    control.accept_run("fast-run")
    assert not control.has_active_run
    control.accept_run("next-run")
    assert control.has_active_run
    control.finish_run("next-run")
    assert not control.has_active_run


def _configured_home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


@pytest.mark.asyncio
async def test_registered_projects_start_once_and_removal_releases_claims(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    record = ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        assert len(service.workspaces) == 1
        client = await service.register_client("cli")
        first, second = await asyncio.gather(
            service.attach_workspace(client.client_id, project),
            service.attach_workspace(client.client_id, project / "."),
        )
        assert first is second
        assert len(service.workspaces) == 1
        session_id = await first.create_draft(client.client_id)
        claim = await service.claim(client.client_id, first.workspace_id, session_id)
        assert cast(dict[str, object], claim["claim"])["claim_version"] == 1

        await service.remove_project(client.client_id, record.project_id)
        assert len(service.workspaces) == 0
        assert not client.claimed
        assert ProjectCatalog(home).list() == ()
        assert project.is_dir()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_closes_admission_clears_claims_and_blocks_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    record, workspace, _jobs = await service.register_project(client.client_id, project)
    session_id = await workspace.create_draft(client.client_id)
    await service.claim(client.client_id, workspace.workspace_id, session_id)
    reconnect_credential = client.reconnect_credential
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    original_close = workspace.close

    async def gated_close() -> None:
        close_started.set()
        await allow_close.wait()
        await original_close()

    monkeypatch.setattr(workspace, "close", gated_close)
    try:
        response = await service.start_project_removal(client.client_id, record.project_id)
        assert response["status"] == "removing"
        await asyncio.wait_for(close_started.wait(), timeout=1)
        assert await service.project_removal_status(
            client.client_id, record.project_id, cast(str, response["operation_id"])
        ) == {
            "project_id": record.project_id,
            "operation_id": response["operation_id"],
            "status": "removing",
        }
        with pytest.raises(ServiceError) as blocked:
            await service.attach_workspace(client.client_id, project)
        assert blocked.value.code == "admission_closed"

        allow_close.set()
        operation = service._project_removals[record.project_id]
        assert operation.task is not None
        await asyncio.wait_for(asyncio.shield(operation.task), timeout=2)
        assert (
            await service.project_removal_status(
                client.client_id, record.project_id, operation.operation_id
            )
        )["status"] == "completed"

        assert not workspace._claims
        assert not client.claimed
        event_types = [event["type"] for event in client.events]
        assert event_types.index("project.removal.started") < event_types.index("project.removed")
        assert event_types.index("project.removed") < event_types.index("project.removal.completed")
        with pytest.raises(ServiceError) as reentry:
            await service.register_client("cli", reconnect_credential=reconnect_credential)
        assert reentry.value.code == "project_reentry_required"
        reentered, _workspace, _saved_jobs = await service.register_project(
            client.client_id, project
        )
        assert reentered.project_id != record.project_id
        assert not client.reconnect_blocked
        reconnected = await service.register_client(
            "cli", reconnect_credential=client.reconnect_credential
        )
        assert reconnected is client
        await service.remove_project(client.client_id, reentered.project_id)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_notifies_unattached_web_requester_without_blocking_reconnect(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    record = ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")
    try:
        assert not client.attached_workspaces
        await service.remove_project(client.client_id, record.project_id)
        assert [
            event_type
            for event in client.events
            if isinstance(event_type := event.get("type"), str)
            and event_type.startswith("project.removal.")
        ] == ["project.removal.started", "project.removal.completed"]
        assert not client.reconnect_blocked
        assert (
            await service.register_client("web", reconnect_credential=client.reconnect_credential)
            is client
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_waits_for_restore_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    session.commit_agent_run(
        [{"role": "user", "content": "Restore this turn"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    record = ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, project)
    await service.claim(client.client_id, workspace.workspace_id, session.session_id)
    dispatcher = workspace.management_dispatcher(client.client_id, session.session_id)
    listing = await dispatcher.dispatch("/restore")
    assert listing.restore_listing is not None
    inspected = await dispatcher.restore_inspect(1)
    assert inspected.restore_plan is not None
    restore_started = asyncio.Event()
    release_restore = asyncio.Event()
    original_execute = RestoreManager.execute

    async def gated_execute(manager: RestoreManager, plan: Any, mode: RestoreMode | str) -> Any:
        restore_started.set()
        await release_restore.wait()
        return await original_execute(manager, plan, mode)

    monkeypatch.setattr(RestoreManager, "execute", gated_execute)
    restore_task = asyncio.create_task(
        dispatcher.restore_commit(inspected.restore_plan, RestoreMode.CONVERSATION_ONLY)
    )
    try:
        await asyncio.wait_for(restore_started.wait(), timeout=2)
        response = await service.start_project_removal(client.client_id, record.project_id)
        await asyncio.sleep(0)
        assert response["status"] == "removing"
        assert ProjectCatalog(home).list()[0].schedule_state == "removing"
        assert workspace.workspace_id in service.workspaces
        assert not restore_task.done()
        release_restore.set()
        committed = await asyncio.wait_for(restore_task, timeout=2)
        assert committed.restore_result is not None
        await service.remove_project(client.client_id, record.project_id)
        assert ProjectCatalog(home).list() == ()
        assert Session.load(state, session.session_id).messages == []
    finally:
        release_restore.set()
        await asyncio.gather(restore_task, return_exceptions=True)
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_keeps_failed_loop_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    record, workspace, _jobs = await service.register_project(client.client_id, project)
    session_id = await workspace.create_draft(client.client_id)
    await service.claim(client.client_id, workspace.workspace_id, session_id)
    loop = workspace.loops[session_id].loop
    original_close = loop.close
    failed = False

    async def close_once() -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected cancellation failure")
        await original_close()

    monkeypatch.setattr(loop, "close", close_once)
    try:
        with pytest.raises(ServiceError) as error:
            await service.remove_project(client.client_id, record.project_id)
        assert error.value.code == "project_removal_failed"
        assert session_id in workspace.loops
        assert session_id in workspace._claims
        await service.remove_project(client.client_id, record.project_id)
        assert not workspace.loops
        assert not workspace._claims
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_schedule_stays_paused_across_service_restart_until_resumed(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    now_ms = 1_800_000_000_000
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="saved project job",
        schedule=JobSchedule.every(3600),
        created_at_ms=now_ms,
        updated_at_ms=now_ms,
    )
    await WorkspaceScheduleStore(state).add_user_job(job)

    first_service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await first_service.start()
    first_client = await first_service.register_client("web")

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    sink = Sink()
    await first_service.connect_client(first_client.client_id, sink)
    active_workspace = await first_service.attach_workspace(first_client.client_id, project)
    assert active_workspace._schedule_admitted
    record, first_workspace, saved_jobs = await first_service.register_project(
        first_client.client_id, project
    )
    assert first_workspace is active_workspace
    assert record.schedule_state == "awaiting_resume"
    assert [saved.job_id for saved in saved_jobs] == [job.job_id]
    assert not first_workspace._schedule_admitted
    await first_service.stop()

    second_service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await second_service.start()
    second_client = await second_service.register_client("web")
    await second_service.connect_client(second_client.client_id, sink)
    try:
        assert len(second_service.workspaces) == 0
        persisted = ProjectCatalog(home).list()
        assert [(item.project_id, item.schedule_state) for item in persisted] == [
            (record.project_id, "awaiting_resume")
        ]

        project.rename(tmp_path / "moved-project")
        with pytest.raises(ServiceError) as missing_error:
            await second_service.resume_project_schedule(
                second_client.client_id, record.project_id, {job.job_id}
            )
        assert missing_error.value.code == "not_found"
        (tmp_path / "moved-project").rename(project)

        second_workspace = await second_service.attach_workspace(second_client.client_id, project)
        assert not second_workspace._schedule_admitted
        assert second_workspace.schedule_service is not None
        assert (
            await second_service.resume_project_schedule(
                second_client.client_id, record.project_id, {job.job_id}
            )
            == "available"
        )
        assert second_workspace._schedule_admitted
    finally:
        await second_service.stop()


@pytest.mark.asyncio
async def test_removed_project_re_registration_keeps_saved_jobs_paused_until_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    clock = FakeClock(datetime(2026, 9, 30, 13, 0, tzinfo=UTC))
    wake_tick = asyncio.Event()
    starts: list[str] = []
    started = asyncio.Event()

    async def wait_for_tick(_seconds: float) -> None:
        await wake_tick.wait()
        wake_tick.clear()

    async def execute(occurrence: ScheduleOccurrence) -> None:
        starts.append(occurrence.job.job_id)
        started.set()

    async def execute_dream() -> None:
        return None

    class RecordingScheduleService(ScheduleService):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                **{
                    **kwargs,
                    "clock": clock,
                    "execute_user_occurrence": execute,
                    "execute_dream": execute_dream,
                }
            )

    monkeypatch.setattr(clock, "sleep", wait_for_tick)
    monkeypatch.setattr(service_runtime, "ScheduleService", RecordingScheduleService)

    async def advance(seconds: float) -> None:
        clock.advance(seconds)
        wake_tick.set()
        for _ in range(10):
            await asyncio.sleep(0)

    async def wait_for_starts(count: int, schedule: ScheduleService) -> None:
        async with asyncio.timeout(2):
            while len(starts) < count:
                started.clear()
                await started.wait()
            while schedule.status_snapshot().active_job_count:
                await asyncio.sleep(0)

    jobs = (
        ScheduleJob(
            job_id=str(uuid4()),
            message="saved at task",
            schedule=JobSchedule.at("2026-09-30T12:00:00.000+00:00"),
            created_at_ms=1_600_000_000_000,
            updated_at_ms=1_600_000_000_000,
        ),
        ScheduleJob(
            job_id=str(uuid4()),
            message="saved every task",
            schedule=JobSchedule.every(3600),
            created_at_ms=1_600_000_000_000,
            updated_at_ms=1_600_000_000_000,
        ),
        ScheduleJob(
            job_id=str(uuid4()),
            message="saved cron task",
            schedule=JobSchedule.cron("0 * * * *", "UTC"),
            created_at_ms=1_600_000_000_000,
            updated_at_ms=1_600_000_000_000,
        ),
    )
    store = WorkspaceScheduleStore(state)
    for job in jobs:
        await store.add_user_job(job)

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    first_service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await first_service.start()
    first_client = await first_service.register_client("web")
    await first_service.connect_client(first_client.client_id, Sink())
    try:
        first_record, first_workspace, saved = await first_service.register_project(
            first_client.client_id, project
        )
        assert first_record.schedule_state == "awaiting_resume"
        assert {job.job_id for job in saved} == {job.job_id for job in jobs}
        assert not first_workspace.schedule_admitted
        await advance(7200)
        assert starts == []

        await first_service.remove_project(first_client.client_id, first_record.project_id)
        assert ProjectCatalog(home).list() == ()
        assert {job.job_id for job in await store.public_snapshot()} == {job.job_id for job in jobs}
        record, workspace, saved = await first_service.register_project(
            first_client.client_id, project
        )
        assert record.schedule_state == "awaiting_resume"
        assert {job.job_id for job in saved} == {job.job_id for job in jobs}
        await advance(7200)
        assert starts == []
    finally:
        await first_service.stop()

    second_service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await second_service.start()
    second_client = await second_service.register_client("web")
    await second_service.connect_client(second_client.client_id, Sink())
    try:
        restarted_record, workspace, saved = await second_service.register_project(
            second_client.client_id, project
        )
        assert restarted_record.project_id == record.project_id
        assert restarted_record.schedule_state == "awaiting_resume"
        assert ProjectCatalog(home).list()[0].schedule_state == "awaiting_resume"
        assert {job.job_id for job in saved} == {job.job_id for job in jobs}
        assert not workspace.schedule_admitted
        await advance(7200)
        assert starts == []
        with pytest.raises(ServiceError) as stale:
            await second_service.resume_project_schedule(
                second_client.client_id, record.project_id, {jobs[0].job_id}
            )
        assert stale.value.code == "stale_schedule_review"
        assert not workspace.schedule_admitted

        other_client = await second_service.register_client("web")
        await second_service.connect_client(other_client.client_id, Sink())
        results = await asyncio.gather(
            *(
                second_service.resume_project_schedule(
                    client.client_id, record.project_id, {job.job_id for job in jobs}
                )
                for client in (second_client, other_client)
            )
        )
        assert results == ["available", "available"]
        assert workspace.schedule_admitted
        await wait_for_starts(2, workspace.schedule_service)
        assert starts.count(jobs[0].job_id) == 1
        assert starts.count(jobs[1].job_id) == 1
        assert starts.count(jobs[2].job_id) == 0
        assert (
            await second_service.resume_project_schedule(
                other_client.client_id, record.project_id, {job.job_id for job in jobs}
            )
            == "available"
        )
        assert len(starts) == 2

        await advance(3600)
        await wait_for_starts(4, workspace.schedule_service)
        assert starts.count(jobs[0].job_id) == 1
        assert starts.count(jobs[1].job_id) == 2
        assert starts.count(jobs[2].job_id) == 1
        saved = await WorkspaceScheduleStore(state).public_snapshot()
        assert {job.job_id for job in saved} == {jobs[1].job_id, jobs[2].job_id}
        assert all(job.state.last_status == "ok" for job in saved)
    finally:
        await second_service.stop()


@pytest.mark.asyncio
async def test_stopped_service_cannot_reopen_project_schedule_admission(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="must stay paused",
        schedule=JobSchedule.every(3600),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await WorkspaceScheduleStore(state).add_user_job(job)

    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    try:
        await service.connect_client(client.client_id, Sink())
        record, workspace, saved_jobs = await service.register_project(client.client_id, project)
        assert record.schedule_state == "awaiting_resume"
        assert [saved.job_id for saved in saved_jobs] == [job.job_id]
        assert not workspace.schedule_admitted

        await service.stop()
        with pytest.raises(ServiceError) as rejected:
            await service.resume_project_schedule(client.client_id, record.project_id, {job.job_id})
        assert rejected.value.code == "admission_closed"
        assert ProjectCatalog(home).list()[0].schedule_state == "awaiting_resume"
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["disconnect", "stop"])
async def test_schedule_activation_rechecks_service_gate_after_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")
    entered = asyncio.Event()
    release = asyncio.Event()
    first = next(iter(service.workspaces.values()))
    original_activate = first.activate_schedule
    observed: list[tuple[str, bool]] = []

    async def delayed_activate() -> None:
        entered.set()
        await release.wait()
        await original_activate()
        observed.append((service.state, first._schedule_admitted))

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    monkeypatch.setattr(first, "activate_schedule", delayed_activate)
    try:
        connecting = asyncio.create_task(service.connect_client(client.client_id, Sink()))
        await asyncio.wait_for(entered.wait(), timeout=1)
        changing = asyncio.create_task(
            service.disconnect_client(client.client_id)
            if transition == "disconnect"
            else service.stop()
        )
        expected_state = "reconnecting" if transition == "disconnect" else "draining"
        for _ in range(10):
            if service.state == expected_state:
                break
            await asyncio.sleep(0)
        assert service.state == expected_state
        release.set()
        await asyncio.gather(connecting, changing)
        assert observed == [(expected_state, False)]
        assert all(not workspace.schedule_admitted for workspace in service.workspaces.values())
    finally:
        release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_other_client_disconnect_preserves_restore_schedule_pause(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    session.commit_agent_run(
        [{"role": "user", "content": "Restore this turn"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    try:
        owner = await service.register_client("cli")
        other = await service.register_client("web")
        await service.connect_client(owner.client_id, Sink())
        await service.connect_client(other.client_id, Sink())
        workspace = await service.attach_workspace(owner.client_id, project)
        await service.attach_workspace(other.client_id, project)
        await service.claim(owner.client_id, workspace.workspace_id, session.session_id)
        dispatcher = workspace.management_dispatcher(owner.client_id, session.session_id)
        assert (await dispatcher.dispatch("/restore")).restore_listing is not None
        assert (await dispatcher.restore_inspect(1)).restore_plan is not None
        assert workspace.schedule_service.admission_paused
        assert workspace.schedule_status()["admitted"] is False

        await service.disconnect_client(other.client_id)
        assert workspace.schedule_service.admission_paused
        assert workspace.schedule_status()["admitted"] is False
        await service.disconnect_client(owner.client_id)
        assert service.state == "reconnecting"
        await service.connect_client(owner.client_id, Sink())
        assert workspace.schedule_service.admission_paused
        await dispatcher.restore_cancel()
        assert workspace.schedule_admitted
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_snapshot_cannot_recreate_a_removed_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    try:
        client = await service.register_client("web")
        await service.connect_client(client.client_id, Sink())
        record, workspace, _jobs = await service.register_project(client.client_id, project)
        entered = asyncio.Event()
        release = asyncio.Event()
        original_snapshot = workspace.schedule_service.public_snapshot

        async def delayed_snapshot() -> tuple[ScheduleJob, ...]:
            entered.set()
            await release.wait()
            return await original_snapshot()

        monkeypatch.setattr(workspace.schedule_service, "public_snapshot", delayed_snapshot)
        snapshot = asyncio.create_task(service.project_schedule_snapshot(record))
        await asyncio.wait_for(entered.wait(), timeout=1)
        removing = asyncio.create_task(service.remove_project(client.client_id, record.project_id))
        await asyncio.sleep(0)
        assert not removing.done()
        release.set()
        await snapshot
        await removing
        assert not service.workspaces
        assert await service.project_schedule_snapshot(record) == ((), None)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_reacquired_claim_rejects_the_previous_version(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        first = await service.claim(client.client_id, workspace.workspace_id, session_id)
        await workspace.release(client.client_id, session_id, close_idle=False)
        second = await service.claim(client.client_id, workspace.workspace_id, session_id)
        assert cast(dict[str, object], first["claim"])["claim_version"] == 1
        assert cast(dict[str, object], second["claim"])["claim_version"] == 2
        with pytest.raises(ServiceError) as raised:
            workspace.require_claim(client.client_id, session_id, 1)
        assert raised.value.code == "stale_claim"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_foreground_confirmation_is_broadcast_to_workspace_clients_and_resolved_once(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    unrelated_workspace_path = tmp_path / "unrelated-workspace"
    workspace_path = tmp_path / "workspace"
    unrelated_workspace_path.mkdir()
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("cli")
        await service.attach_workspace(owner.client_id, unrelated_workspace_path)
        workspace = await service.attach_workspace(owner.client_id, workspace_path)
        await service.attach_workspace(other.client_id, workspace_path)
        session_id = await workspace.create_draft(owner.client_id)
        await service.claim(owner.client_id, workspace.workspace_id, session_id)
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        for _ in range(100):
            requested = [
                event for event in owner.events if event["type"] == "confirmation.requested"
            ]
            if requested:
                break
            await asyncio.sleep(0.01)
        assert len(requested) == 1
        for _ in range(100):
            if any(event["type"] == "confirmation.requested" for event in other.events):
                break
            await asyncio.sleep(0.01)
        other_requested = [
            event for event in other.events if event["type"] == "confirmation.requested"
        ]
        assert len(other_requested) == 1
        payload = cast(dict[str, object], requested[0]["payload"])
        wire_token = cast(str, payload["token"])
        assert cast(dict[str, object], other_requested[0]["payload"])["token"] == wire_token
        assert requested[0]["session_id"] == session_id
        assert requested[0]["run_id"] is not None
        assert payload["origin"] == "foreground"
        assert cast(dict[str, object], payload["request"])["tool_name"] == "exec"
        await service.handle_command(
            other.client_id,
            {
                "request_id": "other-decision",
                "type": "confirmation_decide",
                "payload": {"token": wire_token, "decision": "approved"},
            },
        )
        assert await asyncio.wait_for(pending, timeout=1) == "approved"
        with pytest.raises(ServiceError) as resolved:
            await service.handle_command(
                owner.client_id,
                {
                    "request_id": "owner-decision",
                    "type": "confirmation_decide",
                    "payload": {"token": wire_token, "decision": "declined"},
                },
            )
        assert resolved.value.code == "confirmation_resolved"
        for client in (owner, other):
            assert any(
                event["type"] == "confirmation.resolved"
                and cast(dict[str, object], event["payload"])["token"] == wire_token
                for event in client.events
            )
        next_envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-2", "exec", "Run another command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        next_pending = asyncio.create_task(service.confirmation.request(next_envelope))
        for _ in range(100):
            next_events = [
                event for event in owner.events if event["type"] == "confirmation.requested"
            ]
            if len(next_events) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(next_events) == 2
        next_token = cast(str, cast(dict[str, object], next_events[-1]["payload"])["token"])
        assert next_token != wire_token
        with pytest.raises(ServiceError) as stale:
            await service.handle_command(
                other.client_id,
                {
                    "request_id": "stale-decision",
                    "type": "confirmation_decide",
                    "payload": {"token": wire_token, "decision": "approved"},
                },
            )
        assert stale.value.code == "confirmation_resolved"
        assert not next_pending.done()
        await service.handle_command(
            owner.client_id,
            {
                "request_id": "next-decision",
                "type": "confirmation_decide",
                "payload": {"token": next_token, "decision": "declined"},
            },
        )
        assert await asyncio.wait_for(next_pending, timeout=1) == "declined"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_confirmation_resolved_cannot_overtake_requested_for_another_client(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    requested_started = asyncio.Event()
    release_requested = asyncio.Event()

    class SlowSink:
        async def send_event(self, event: dict[str, object]) -> None:
            if event["type"] == "confirmation.requested":
                requested_started.set()
                await release_requested.wait()

    try:
        first = await service.register_client("cli")
        second = await service.register_client("cli")
        workspace = await service.attach_workspace(first.client_id, workspace_path)
        await service.attach_workspace(second.client_id, workspace_path)
        session_id = await workspace.create_draft(first.client_id)
        await service.claim(first.client_id, workspace.workspace_id, session_id)
        await service.connect_client(first.client_id, SlowSink())
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        await asyncio.wait_for(requested_started.wait(), timeout=1)
        requested = next(
            event for event in first.events if event["type"] == "confirmation.requested"
        )
        token = cast(str, cast(dict[str, object], requested["payload"])["token"])
        await service.handle_command(
            first.client_id,
            {
                "request_id": "early-decision",
                "type": "confirmation_decide",
                "payload": {"token": token, "decision": "declined"},
            },
        )
        await asyncio.sleep(0)
        assert not any(event["type"] == "confirmation.resolved" for event in second.events)
        release_requested.set()
        assert await asyncio.wait_for(pending, timeout=1) == "declined"
        for _ in range(100):
            events = [
                event["type"]
                for event in second.events
                if event["type"] in {"confirmation.requested", "confirmation.resolved"}
            ]
            if len(events) == 2:
                break
            await asyncio.sleep(0.01)
        assert events == ["confirmation.requested", "confirmation.resolved"]
    finally:
        release_requested.set()
        await service.stop()


@pytest.mark.asyncio
async def test_background_confirmation_broadcast_has_job_source_without_session_scope(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    unrelated_workspace_path = tmp_path / "unrelated-workspace"
    workspace_path = tmp_path / "workspace"
    unrelated_workspace_path.mkdir()
    workspace_path.mkdir()
    ProjectCatalog(home).register(workspace_path)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()

    class Sink:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def send_event(self, event: dict[str, object]) -> None:
            self.events.append(event)

    status_sink = Sink()
    unrelated_sink = Sink()
    try:
        first = await service.register_client("web")
        second = await service.register_client("web")
        status_page = await service.register_client("web")
        unrelated_cli = await service.register_client("cli")
        await service.attach_workspace(unrelated_cli.client_id, unrelated_workspace_path)
        workspace = await service.attach_workspace(first.client_id, workspace_path)
        await service.attach_workspace(second.client_id, workspace_path)
        await service.connect_client(status_page.client_id, status_sink)
        await service.connect_client(unrelated_cli.client_id, unrelated_sink)
        schedule_loop = await workspace._get_schedule_loop("job-1")
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run scheduled command", {}),
            origin="background",
            owner=BackgroundConfirmationOwner(
                schedule_loop.loop.generation_id,
                "job-1",
                uuid4(),
            ),
            job_id="job-1",
            title="Nightly maintenance",
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        for _ in range(100):
            requested = [
                event for event in first.events if event["type"] == "confirmation.requested"
            ]
            if requested:
                break
            await asyncio.sleep(0.01)
        assert len(requested) == 1
        assert any(event["type"] == "confirmation.requested" for event in second.events)
        assert any(event["type"] == "confirmation.requested" for event in status_sink.events)
        assert not any(event["type"] == "confirmation.requested" for event in unrelated_sink.events)
        event = requested[0]
        assert event["workspace_id"] == workspace.workspace_id
        assert event["session_id"] is None
        assert event["run_id"] is None
        payload = cast(dict[str, object], event["payload"])
        assert payload["origin"] == "background"
        assert payload["job_id"] == "job-1"
        assert payload["title"] == "Nightly maintenance"
        assert not status_page.claimed
        await service.handle_command(
            second.client_id,
            {
                "request_id": "background-decision",
                "type": "confirmation_decide",
                "payload": {"token": payload["token"], "decision": "declined"},
            },
        )
        assert await asyncio.wait_for(pending, timeout=1) == "declined"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_stop_aborts_confirmation_and_invalidates_token(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        await service.claim(client.client_id, workspace.workspace_id, session_id)
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        for _ in range(100):
            requests = [
                event for event in client.events if event["type"] == "confirmation.requested"
            ]
            if requests:
                break
            await asyncio.sleep(0.01)
        assert len(requests) == 1
        token = cast(str, cast(dict[str, object], requests[0]["payload"])["token"])
        await service.stop()
        with pytest.raises(ConfirmationAborted):
            await asyncio.wait_for(pending, timeout=1)
        assert any(event["type"] == "confirmation.resolved" for event in client.events)
        with pytest.raises(ServiceError) as stale:
            await service.handle_command(
                client.client_id,
                {
                    "request_id": "late-approval",
                    "type": "confirmation_decide",
                    "payload": {"token": token, "decision": "approved"},
                },
            )
        assert stale.value.code == "confirmation_resolved"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_aborts_pending_confirmation_and_resolves_clients(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    record = ProjectCatalog(home).register(project)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("web")
        workspace = await service.attach_workspace(owner.client_id, project)
        await service.attach_workspace(other.client_id, project)
        session_id = await workspace.create_draft(owner.client_id)
        await service.claim(owner.client_id, workspace.workspace_id, session_id)
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        for _ in range(100):
            if any(event["type"] == "confirmation.requested" for event in owner.events):
                break
            await asyncio.sleep(0.01)
        await service.remove_project(owner.client_id, record.project_id)
        with pytest.raises(ConfirmationAborted):
            await asyncio.wait_for(pending, timeout=1)
        assert not service.workspaces
        assert all(
            any(event["type"] == "confirmation.resolved" for event in client.events)
            for client in (owner, other)
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_client_disconnect_expiry_aborts_owned_confirmation(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=0.05)
    await service.start()

    class Sink:
        async def send_event(self, event: dict[str, object]) -> None:
            del event

    sink = Sink()
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("web")
        workspace = await service.attach_workspace(owner.client_id, workspace_path)
        await service.attach_workspace(other.client_id, workspace_path)
        await service.connect_client(owner.client_id, sink)
        await service.connect_client(other.client_id, sink)
        session_id = await workspace.create_draft(owner.client_id)
        await service.claim(owner.client_id, workspace.workspace_id, session_id)
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call-1", "exec", "Run command", {}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(
                workspace.loops[session_id].loop.generation_id, uuid4()
            ),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        for _ in range(100):
            if any(event["type"] == "confirmation.requested" for event in other.events):
                break
            await asyncio.sleep(0.01)
        await service.disconnect_client(owner.client_id, sink=sink)
        with pytest.raises(ConfirmationAborted):
            await asyncio.wait_for(pending, timeout=1)
        assert any(event["type"] == "confirmation.resolved" for event in other.events)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_last_client_grace_pauses_and_restarts_schedule(tmp_path: Path) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=0.2)
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)

        class Sink:
            async def send_event(self, event: dict[str, object]) -> None:
                del event

        sink = Sink()
        await service.connect_client(client.client_id, sink)
        assert workspace._schedule_admitted
        await service.disconnect_client(client.client_id, sink=sink)
        assert service.state == "reconnecting"
        assert not workspace._schedule_admitted
        await service.connect_client(client.client_id, sink)
        assert service.state == "ready"
        assert workspace._schedule_admitted
        await service.disconnect_client(client.client_id, sink=sink)
        await asyncio.wait_for(service.wait_closed(), timeout=2)
        assert service.state == "stopped"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_stop_reports_workspace_cleanup_failure_and_releases_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, workspace_path)
    original_close = workspace.close

    async def failing_close() -> None:
        await original_close()
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(workspace, "close", failing_close)
    with pytest.raises(ServiceError) as stopped:
        await service.stop()
    assert stopped.value.code == "service_stop_failed"
    assert service.state == "stopped"
    with pytest.raises(ServiceError) as waited:
        await asyncio.wait_for(service.wait_closed(), timeout=1)
    assert waited.value.code == "service_stop_failed"


@pytest.mark.asyncio
async def test_project_removal_failure_keeps_registration_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    record, workspace, _jobs = await service.register_project(client.client_id, project)
    assert workspace.runtime is not None
    original_close = workspace.runtime.close

    async def failing_close(
        *, close_foreground: Any = None, drain_confirmation_aborts: bool = True
    ) -> None:
        await original_close(
            close_foreground=close_foreground,
            drain_confirmation_aborts=drain_confirmation_aborts,
        )
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(workspace.runtime, "close", failing_close)
    with pytest.raises(ServiceError) as first:
        await service.remove_project(client.client_id, record.project_id)
    assert first.value.code == "project_removal_failed"
    assert ProjectCatalog(home).list()[0].schedule_state == "removing"
    with pytest.raises(ServiceError) as retried:
        await service.remove_project(client.client_id, record.project_id)
    assert retried.value.code == "project_removal_failed"
    assert ProjectCatalog(home).list()[0].schedule_state == "removing"
    with pytest.raises(ServiceError):
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_completion_survives_event_delivery_failure(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()

    class FailingSink:
        async def send_event(self, event: dict[str, object]) -> None:
            if event["type"] in {"project.removed", "session.released"}:
                raise RuntimeError("client disconnected during removal")

    try:
        client = await service.register_client("web")
        await service.connect_client(client.client_id, FailingSink())
        record, _workspace, _jobs = await service.register_project(client.client_id, project)

        await service.remove_project(client.client_id, record.project_id)

        assert ProjectCatalog(home).list() == ()
        assert project.is_dir()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_admission_failure_is_persisted_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")
    record, _workspace, _jobs = await service.register_project(client.client_id, project)
    original_reconcile = service._reconcile_schedule_admission
    failed = False

    async def fail_once() -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected admission failure")
        await original_reconcile()

    monkeypatch.setattr(service, "_reconcile_schedule_admission", fail_once)
    try:
        with pytest.raises(ServiceError) as first:
            await service.start_project_removal(client.client_id, record.project_id)
        assert first.value.code == "project_removal_failed"
        blocked = ProjectCatalog(home).list()[0]
        assert blocked.removal_operation_id
        assert blocked.removal_error

        monkeypatch.setattr(service, "_reconcile_schedule_admission", original_reconcile)
        await service.remove_project(client.client_id, record.project_id)
        assert ProjectCatalog(home).list() == ()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_project_removal_failure_can_retry_same_persisted_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")
    record, workspace, _jobs = await service.register_project(client.client_id, project)
    assert workspace.runtime is not None
    original_close = workspace.runtime.close
    failed = False

    async def fail_once(
        *, close_foreground: Any = None, drain_confirmation_aborts: bool = True
    ) -> None:
        nonlocal failed
        await original_close(
            close_foreground=close_foreground,
            drain_confirmation_aborts=drain_confirmation_aborts,
        )
        if not failed:
            failed = True
            raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(workspace.runtime, "close", fail_once)
    with pytest.raises(ServiceError) as first:
        await service.remove_project(client.client_id, record.project_id)
    assert first.value.code == "project_removal_failed"
    failed_record = ProjectCatalog(home).list()[0]
    assert failed_record.schedule_state == "removing"
    assert failed_record.removal_operation_id

    monkeypatch.setattr(workspace.runtime, "close", original_close)
    await service.remove_project(client.client_id, record.project_id)
    assert ProjectCatalog(home).list() == ()
    assert project.is_dir()
    await service.stop()


@pytest.mark.asyncio
async def test_failed_project_removal_stays_blocked_after_service_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("cli")
    record, workspace, _jobs = await service.register_project(client.client_id, project)
    original_close = workspace.runtime.close if workspace.runtime is not None else None
    assert original_close is not None

    async def failing_close(
        *, close_foreground: Any = None, drain_confirmation_aborts: bool = True
    ) -> None:
        await original_close(
            close_foreground=close_foreground,
            drain_confirmation_aborts=drain_confirmation_aborts,
        )
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(workspace.runtime, "close", failing_close)
    with pytest.raises(ServiceError) as failed:
        await service.remove_project(client.client_id, record.project_id)
    assert failed.value.code == "project_removal_failed"
    monkeypatch.setattr(workspace.runtime, "close", original_close)
    await service.stop()

    persisted = ProjectCatalog(home).list()
    assert len(persisted) == 1
    assert persisted[0].project_id == record.project_id
    assert persisted[0].schedule_state == "removing"
    assert persisted[0].removal_operation_id

    restarted = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await restarted.start()
    restarted_client = await restarted.register_client("cli")
    try:
        assert not restarted.workspaces
        with pytest.raises(ServiceError) as blocked:
            await restarted.attach_workspace(restarted_client.client_id, project)
        assert blocked.value.code == "admission_closed"
        await restarted.remove_project(restarted_client.client_id, record.project_id)
        assert ProjectCatalog(home).list() == ()
        assert project.is_dir()
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_interrupted_project_removal_is_retryable_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    project = tmp_path / "project"
    project.mkdir()
    record = ProjectCatalog(home).register(project)
    started = ProjectCatalog(home).begin_removal(record.project_id)
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    client = await service.register_client("web")
    starts: list[Path] = []
    original_start = WorkspaceServiceRuntime.start

    async def record_start(runtime: WorkspaceServiceRuntime) -> None:
        starts.append(runtime.workspace_path)
        await original_start(runtime)

    monkeypatch.setattr(WorkspaceServiceRuntime, "start", record_start)
    try:
        assert not service.workspaces
        interrupted = ProjectCatalog(home).list()[0]
        assert interrupted.schedule_state == "removing"
        assert interrupted.removal_error is not None
        assert interrupted.removal_operation_id == started.removal_operation_id
        assert (
            await service.project_removal_status(
                client.client_id, record.project_id, cast(str, started.removal_operation_id)
            )
        )["status"] == "failed"
        await service.remove_project(client.client_id, record.project_id)
        assert starts == [project]
        assert ProjectCatalog(home).list() == ()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_stale_run_id_cannot_cancel_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        client = await service.register_client("cli")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id)
        await service.claim(client.client_id, workspace.workspace_id, session_id)
        claim = workspace._claims[session_id]
        workspace.loops[session_id].run_ids.append("current-run")
        cancelled: list[bool] = []

        async def record_cancel(_loop: AgentLoop) -> None:
            cancelled.append(True)

        with monkeypatch.context() as patched:
            patched.setattr(AgentLoop, "has_active_run", property(lambda _loop: True))
            patched.setattr(AgentLoop, "cancel_active_run", record_cancel)
            with pytest.raises(ServiceError) as stale:
                await workspace.cancel(client.client_id, session_id, claim.version, "old-run")
            assert stale.value.code == "stale_run"
            assert cancelled == []
            await workspace.cancel(client.client_id, session_id, claim.version, "current-run")
            assert cancelled == [True]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("registered_peer", [False, True])
async def test_unregistered_workspace_loses_schedule_and_runtime_at_last_user_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_peer: bool
) -> None:
    home = _configured_home(tmp_path / "home")
    clock = FakeClock(datetime(2026, 10, 3, tzinfo=UTC))
    wake = asyncio.Event()
    starts: list[str] = []

    async def sleep(_seconds: float) -> None:
        await wake.wait()
        await asyncio.sleep(0)

    async def execute(occurrence: ScheduleOccurrence) -> None:
        starts.append(occurrence.job.message)

    async def dream() -> None:
        return None

    class RecordingSchedule(ScheduleService):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**{**kwargs, "clock": clock, "execute_user_occurrence": execute,
                               "execute_dream": dream})

    monkeypatch.setattr(clock, "sleep", sleep)
    monkeypatch.setattr(service_runtime, "ScheduleService", RecordingSchedule)
    service = LocalService(home, ConfigLoader(home).load_for_startup(),
                           monotonic_now=clock.monotonic, sleep=sleep)
    await service.start()
    try:
        clients = [await service.register_client(kind)
                   for kind in ("cli", "cli", "web" if registered_peer else "cli")]
        for client in clients:
            await service.connect_client(client.client_id, _CollectingSink())
        paths = [tmp_path / name for name in ("a", "b")]
        for path in paths:
            path.mkdir()
        a = await service.attach_workspace(clients[0].client_id, paths[0])
        await service.attach_workspace(clients[1].client_id, paths[0])
        if registered_peer:
            b = (await service.register_project(clients[2].client_id, paths[1]))[1]
        else:
            b = await service.attach_workspace(clients[2].client_id, paths[1])
        await service.disconnect_client(clients[0].client_id)
        assert a.schedule_status()["admitted"] is True
        await a.schedule_service.add_user_job(ScheduleJob(
            job_id=str(uuid4()), message="A with remaining user",
            schedule=JobSchedule.at("2026-10-03T00:00:01.000+00:00"),
            created_at_ms=1, updated_at_ms=1,
        ))
        clock.advance(1)
        wake.set()
        async with asyncio.timeout(2):
            while not starts:
                await asyncio.sleep(0)
        wake.clear()
        assert starts == ["A with remaining user"]
        await service.disconnect_client(clients[1].client_id)
        for workspace, message in ((a, "A"), (b, "B")):
            await workspace.schedule_service.add_user_job(ScheduleJob(
                job_id=str(uuid4()), message=message,
                schedule=JobSchedule.at("2026-10-03T00:00:02.000+00:00"),
                created_at_ms=1, updated_at_ms=1,
            ))
        clock.advance(29)
        wake.set()
        async with asyncio.timeout(2):
            while "B" not in starts:
                await asyncio.sleep(0)
        assert starts == ["A with remaining user", "B"]
        assert a.workspace_id in service.workspaces
        clock.advance(1)
        await asyncio.wait_for(cast(asyncio.Task[None], clients[1].disconnect_task), 2)
        assert a.workspace_id not in service.workspaces
        assert b.workspace_id in service.workspaces
        assert len(await a.schedule_service.public_snapshot()) == 1
        fresh = await service.register_client("cli")
        reopened = await service.attach_workspace(fresh.client_id, paths[0])
        assert reopened is not a
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("last_seq", [None, "current", 0, "slow"])
async def test_subscribe_restores_only_valid_original_confirmation(
    tmp_path: Path, last_seq: int | str | None
) -> None:
    home = _configured_home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("web")
        workspace = await service.attach_workspace(client.client_id, path)
        session = await workspace.create_draft(client.client_id)
        await service.claim(client.client_id, workspace.workspace_id, session)
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call", "exec", "Exact command", {"command": "echo ok"}),
            origin="foreground",
            owner=ForegroundConfirmationOwner(workspace.loops[session].loop.generation_id, uuid4()),
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        async with asyncio.timeout(2):
            while not any(event["type"] == "confirmation.requested" for event in client.events):
                await asyncio.sleep(0)
        requested = next(event for event in client.events if event["type"] == "confirmation.requested")
        payload = cast(dict[str, object], requested["payload"])
        cursor = client.sequence if last_seq in {"current", "slow"} else last_seq
        if last_seq == "slow":
            client.resync_required = True
        if last_seq == 0:
            for _ in range(260):
                await service.emit("test.event", workspace_id=None, session_id=None, run_id=None, payload={})
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink, wait_for_subscribe=True)
        await service.handle_command(client.client_id, {
            "request_id": "subscribe", "type": "subscribe",
            "payload": {"last_seq": cursor, "stream_id": client.stream_id},
        })
        snapshot = cast(dict[str, Any], sink.events[-1]["payload"])["snapshot"]
        assert snapshot["pending_confirmation"]["payload"] == payload
        _validator("recovery_snapshot").validate(snapshot)
        await service.handle_command(client.client_id, {
            "request_id": "decide", "type": "confirmation_decide",
            "payload": {"token": payload["token"], "decision": "declined"},
        })
        assert await asyncio.wait_for(pending, 2) == "declined"
        await service.handle_command(client.client_id, {
            "request_id": "resubscribe", "type": "subscribe", "payload": {"last_seq": None},
        })
        assert cast(dict[str, Any], sink.events[-1]["payload"])["snapshot"]["pending_confirmation"] is None
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_confirmation_snapshot_audience_competing_decision_and_cancel(
    tmp_path: Path, background: bool
) -> None:
    home = _configured_home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        owner = await service.register_client("cli")
        peer = await service.register_client("cli")
        outsider = await service.register_client("cli")
        workspace = await service.attach_workspace(owner.client_id, path)
        await service.attach_workspace(peer.client_id, path)
        session = await workspace.create_draft(owner.client_id)
        await service.claim(owner.client_id, workspace.workspace_id, session)
        generation = workspace.loops[session].loop.generation_id
        if background:
            generation = (await workspace._get_schedule_loop("job")).loop.generation_id
        envelope = ConfirmationEnvelope(
            request=ConfirmationRequest(uuid4(), "call", "exec", "Exact operation", {}),
            origin="background" if background else "foreground",
            owner=BackgroundConfirmationOwner(generation, "job", uuid4()) if background
            else ForegroundConfirmationOwner(generation, uuid4()),
            job_id="job" if background else None,
            title="Background operation" if background else None,
        )
        pending = asyncio.create_task(service.confirmation.request(envelope))
        async with asyncio.timeout(2):
            while not any(event["type"] == "confirmation.requested" for event in owner.events):
                await asyncio.sleep(0)
        async def recover(client_id: str) -> dict[str, Any]:
            sink = _CollectingSink()
            await service.connect_client(client_id, sink, wait_for_subscribe=True)
            await service.handle_command(client_id, {"request_id": str(uuid4()),
                "type": "subscribe", "payload": {"last_seq": None}})
            return cast(dict[str, Any], cast(dict[str, Any], sink.events[-1]["payload"])["snapshot"])
        recovered = await recover(owner.client_id)
        token = recovered["pending_confirmation"]["payload"]["token"]
        assert (await recover(peer.client_id))["pending_confirmation"]["payload"]["token"] == token
        assert (await recover(outsider.client_id))["pending_confirmation"] is None
        decisions = await asyncio.gather(*(
            service.handle_command(client.client_id, {"request_id": str(uuid4()),
                "type": "confirmation_decide", "payload": {"token": token, "decision": "approved"}})
            for client in (owner, peer)
        ), return_exceptions=True)
        assert sum(isinstance(result, dict) for result in decisions) == 1
        assert await pending == "approved"
        pending = asyncio.create_task(service.confirmation.request(envelope))
        async with asyncio.timeout(2):
            while len([event for event in owner.events if event["type"] == "confirmation.requested"]) < 2:
                await asyncio.sleep(0)
        await service.confirmation.cancel_generation(generation)
        with pytest.raises(ConfirmationAborted):
            await pending
        await service.handle_command(owner.client_id, {"request_id": str(uuid4()),
            "type": "subscribe", "payload": {"last_seq": None}})
        assert cast(dict[str, Any], owner.events[-1]["payload"])["snapshot"]["pending_confirmation"] is None
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("register", [False, True])
async def test_workspace_expiry_serializes_reentry_and_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, register: bool
) -> None:
    home = _configured_home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    clock = FakeClock(datetime(2026, 10, 3, tzinfo=UTC))
    wake = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    async def sleep(_seconds: float) -> None:
        await wake.wait()
    service = LocalService(home, ConfigLoader(home).load_for_startup(),
        monotonic_now=clock.monotonic, sleep=sleep)
    await service.start()
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("web")
        await service.connect_client(owner.client_id, _CollectingSink())
        await service.connect_client(other.client_id, _CollectingSink())
        workspace = await service.attach_workspace(owner.client_id, path)
        original_close = workspace.close
        async def close() -> None:
            entered.set()
            await release.wait()
            await original_close()
        monkeypatch.setattr(workspace, "close", close)
        await service.disconnect_client(owner.client_id)
        clock.advance(30)
        wake.set()
        await asyncio.wait_for(entered.wait(), 2)
        reentry = asyncio.create_task(service.register_project(other.client_id, path) if register
            else service.attach_workspace(other.client_id, path))
        await asyncio.sleep(0)
        assert not reentry.done()
        release.set()
        result = await asyncio.wait_for(reentry, 2)
        reopened = cast(WorkspaceServiceRuntime, result[1] if isinstance(result, tuple) else result)
        assert reopened is not workspace
        assert reopened.workspace_id in service.workspaces
        assert workspace.workspace_id not in service.workspaces
    finally:
        release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_workspace_expiry_cleanup_failure_keeps_owned_runtime_and_closes_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configured_home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    clock = FakeClock(datetime(2026, 10, 3, tzinfo=UTC))
    wake = asyncio.Event()
    async def sleep(_seconds: float) -> None:
        await wake.wait()
    service = LocalService(home, ConfigLoader(home).load_for_startup(),
        monotonic_now=clock.monotonic, sleep=sleep)
    await service.start()
    original_close = None
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("web")
        sink = _CollectingSink()
        await service.connect_client(owner.client_id, _CollectingSink())
        await service.connect_client(other.client_id, sink)
        workspace = await service.attach_workspace(owner.client_id, path)
        original_close = workspace.close
        async def failing_close() -> None:
            raise RuntimeError("injected cleanup failure")
        monkeypatch.setattr(workspace, "close", failing_close)
        await service.disconnect_client(owner.client_id)
        clock.advance(30)
        wake.set()
        await asyncio.wait_for(cast(asyncio.Task[None], owner.disconnect_task), 2)
        assert workspace.workspace_id in service.workspaces
        assert service.state == "draining"
        assert any(event["type"] == "service.cleanup_failed" for event in sink.events)
        with pytest.raises(ServiceError) as unavailable:
            await service.attach_workspace(other.client_id, path)
        assert unavailable.value.code == "admission_closed"
    finally:
        if original_close is not None:
            monkeypatch.setattr(workspace, "close", original_close)
        await service.stop()
