from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from myclaw.agent.confirmation import ConfirmationEnvelope, ForegroundConfirmationOwner
from myclaw.agent.loop import AgentLoop
from myclaw.agent.tools.tool_gateway import ConfirmationRequest
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.schedule.model import JobSchedule, ScheduleJob
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
from myclaw.service.runtime import LocalService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction only")
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
async def test_foreground_confirmation_is_visible_and_decidable_only_by_claim_owner(
    tmp_path: Path,
) -> None:
    home = _configured_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=5)
    await service.start()
    try:
        owner = await service.register_client("cli")
        other = await service.register_client("cli")
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
        assert not any(event["type"] == "confirmation.requested" for event in other.events)
        payload = cast(dict[str, object], requested[0]["payload"])
        wire_token = cast(str, payload["token"])
        with pytest.raises(ServiceError) as rejected:
            await service.handle_command(
                other.client_id,
                {
                    "request_id": "other-decision",
                    "type": "confirmation_decide",
                    "payload": {"token": wire_token, "decision": "approved"},
                },
            )
        assert rejected.value.code == "forbidden"
        await service.handle_command(
            owner.client_id,
            {
                "request_id": "owner-decision",
                "type": "confirmation_decide",
                "payload": {"token": wire_token, "decision": "declined"},
            },
        )
        assert await asyncio.wait_for(pending, timeout=1) == "declined"
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
