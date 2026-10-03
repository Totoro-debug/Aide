from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio

from omni.agent.loop import AgentLoop
from omni.agent.tools.tool_gateway import ConfirmationDecision, ConfirmationRequest, ModelToolCall
from omni.agent.workspace_runtime import WorkspaceRuntime
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader, UserConfiguration
from omni.schedule.store import WorkspaceScheduleStore
from omni.service.runtime import LocalService, _PreparedWorkspaceGeneration
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.project_removal import complete_project_removal


@pytest_asyncio.fixture
async def generation_service(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[LocalService]:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    configuration_text = MINIMAL_VALID_CONFIG + cast(str, getattr(request, "param", ""))
    (home.path / "config.toml").write_text(configuration_text, encoding="utf-8")
    service = LocalService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    await service.start()
    yield service
    await service.stop()


async def _save(service: LocalService, request_id: str = "generation-change") -> None:
    await service.update_configuration(
        request_id,
        cast(str, service.config_view()["revision"]),
        {"runtime": {"max_iterations": 83}},
    )


async def _applied(service: LocalService) -> None:
    task = service._config_apply_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=10)
    assert cast(dict[str, object], service.config_view()["application"])["status"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_candidate", [False, True])
async def test_generation_preparation_preserves_unchanged_session_bytes_and_recency(
    generation_service: LocalService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_candidate: bool,
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    assert workspace.workspace_state is not None
    previous = workspace.runtime
    sessions = []
    for has_messages in (False, True):
        session_id = await workspace.create_draft(client.client_id)
        claim = await workspace.claim(client.client_id, session_id)
        session = claim.loop.session
        if has_messages:
            session.commit_agent_run(
                [{"role": "user", "content": "already completed conversation"}],
                pending_last_compacted=session.last_compacted,
                pending_action_summary="",
            )
            await session.wait_for_pending_persist()
        updated_at = session.updated_at
        monkeypatch.setattr(
            session, "_now", lambda timestamp=updated_at: timestamp + timedelta(days=1)
        )
        session_path = workspace.workspace_state.sessions_directory / f"{session_id}.jsonl"
        before = session_path.read_bytes() if session_path.exists() else None
        sessions.append((session_id, session, updated_at, session_path, before))

    if fail_candidate:

        async def fail_start(
            runtime: WorkspaceRuntime, previous_runtime: WorkspaceRuntime
        ) -> WorkspaceRuntime:
            raise OSError("candidate preparation failed")

        monkeypatch.setattr(WorkspaceRuntime, "start_replacement", fail_start)
    await _save(service)
    task = service._config_apply_task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=10)
    status = cast(dict[str, object], service.config_view()["application"])["status"]
    assert status == ("failed-to-apply" if fail_candidate else "active")
    if fail_candidate:
        assert workspace.runtime is previous
    for session_id, old_session, updated_at, session_path, before in sessions:
        assert old_session.updated_at == updated_at
        assert workspace.loops[session_id].loop.session.updated_at == updated_at
        assert (session_path.read_bytes() if session_path.exists() else None) == before


@pytest.mark.asyncio
async def test_workspace_attached_during_preparation_uses_latest_generation(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    first_path = tmp_path / "first"
    first_path.mkdir()
    first = await service.attach_workspace(client.client_id, first_path)
    started, release = asyncio.Event(), asyncio.Event()
    original = WorkspaceRuntime.start_replacement

    async def blocked(runtime: WorkspaceRuntime, previous: WorkspaceRuntime) -> WorkspaceRuntime:
        started.set()
        await release.wait()
        return await original(runtime, previous)

    monkeypatch.setattr(WorkspaceRuntime, "start_replacement", blocked)
    await _save(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    second_path = tmp_path / "second"
    second_path.mkdir()
    attach = asyncio.create_task(service.attach_workspace(client.client_id, second_path))
    await asyncio.sleep(0)
    release.set()
    second = await asyncio.wait_for(attach, timeout=10)
    await _applied(service)
    assert first.configuration.runtime.max_iterations == 83
    assert second.configuration.runtime.max_iterations == 83
    assert all(
        workspace.runtime is not None and not workspace.runtime._closed
        for workspace in service.workspaces.values()
    )


@pytest.mark.asyncio
async def test_retirement_failure_does_not_discard_published_generation(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id)
    claim = await workspace.claim(client.client_id, session_id)
    previous = workspace.runtime
    assert previous is not None
    close = previous.router.close
    attempts = 0

    async def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("retirement canary")
        await close()

    monkeypatch.setattr(previous.router, "close", fail_once)
    await _save(service)
    await _applied(service)
    assert workspace.runtime is not previous
    assert workspace.runtime is not None and not workspace.runtime._closed
    assert workspace.configuration.runtime.max_iterations == 83
    assert workspace.loops[session_id].loop is claim.loop
    assert not claim.loop._closed
    assert workspace.runtime.router.route_status("chat").model
    await service.stop()
    assert attempts >= 2


@pytest.mark.asyncio
async def test_active_configuration_status_waits_for_workspace_admission(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    previous = workspace.runtime
    assert previous is not None
    entered, release = asyncio.Event(), asyncio.Event()
    close = previous.router.close

    async def blocked_close() -> None:
        entered.set()
        await release.wait()
        await close()

    monkeypatch.setattr(previous.router, "close", blocked_close)
    await _save(service)
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        application = cast(dict[str, object], service.config_view()["application"])
        assert workspace.configuration.runtime.max_iterations == 83
        assert service.configuration_transition_active
        assert application["status"] == "pending"
        assert application["pending_revision"] == application["saved_revision"]
    finally:
        release.set()
        task = service._config_apply_task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
    await _applied(service)
    assert not service.configuration_transition_active
    draft = await service.create_session(client.client_id, workspace.workspace_id)
    claimed = await service.claim(
        client.client_id, workspace.workspace_id, cast(str, draft["session_id"])
    )
    claim = cast(dict[str, object], claimed["claim"])

    async def hold_preparation(
        loop: AgentLoop, context: object, *, tool_gateway: object
    ) -> list[dict[str, object]]:
        del loop, context, tool_gateway
        await asyncio.Event().wait()
        return []

    monkeypatch.setattr(AgentLoop, "_prepare_agent_run", hold_preparation)
    accepted = await service.handle_command(
        client.client_id,
        {
            "request_id": "input-immediately-after-active",
            "type": "input",
            "workspace_id": workspace.workspace_id,
            "session_id": draft["session_id"],
            "claim_version": claim["claim_version"],
            "payload": {"text": "new generation input"},
        },
    )
    assert accepted["accepted"] is True
    assert cast(dict[str, object], accepted["result"])["run_id"]


@pytest.mark.asyncio
async def test_stop_drains_configuration_preparation_and_candidate_resources(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    started = asyncio.Event()
    candidates: list[WorkspaceRuntime] = []
    original = WorkspaceRuntime.start_replacement

    async def blocked(runtime: WorkspaceRuntime, previous: WorkspaceRuntime) -> WorkspaceRuntime:
        result = await original(runtime, previous)
        candidates.append(runtime)
        started.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr(WorkspaceRuntime, "start_replacement", blocked)
    await _save(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    apply_task = service._config_apply_task
    assert apply_task is not None
    await asyncio.wait_for(service.stop(), timeout=10)
    assert apply_task.done()
    assert service.state == "stopped"
    assert workspace.runtime is not None and workspace.runtime._closed
    assert candidates and all(candidate._closed for candidate in candidates)


@pytest.mark.asyncio
async def test_client_release_during_preparation_does_not_resurrect_loop(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id)
    await workspace.claim(client.client_id, session_id)
    started, release = asyncio.Event(), asyncio.Event()
    prepare = workspace.prepare_configuration

    async def blocked(configuration: UserConfiguration) -> _PreparedWorkspaceGeneration:
        result = await prepare(configuration)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(workspace, "prepare_configuration", blocked)
    await _save(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    await workspace.release(client.client_id, session_id)
    release.set()
    await _applied(service)
    assert session_id not in workspace.loops
    assert session_id not in workspace._claims


@pytest.mark.asyncio
async def test_failed_candidate_keeps_old_persisted_dream_schedule(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first = await service.attach_workspace(client.client_id, first_path)
    second = await service.attach_workspace(client.client_id, second_path)
    first_runtime, second_runtime = first.runtime, second.runtime
    assert first.workspace_state is not None
    before = await WorkspaceScheduleStore(first.workspace_state).snapshot()
    original = WorkspaceRuntime.start_replacement

    async def fail_second(
        runtime: WorkspaceRuntime, previous: WorkspaceRuntime
    ) -> WorkspaceRuntime:
        if previous is second_runtime:
            raise OSError("candidate canary")
        return await original(runtime, previous)

    monkeypatch.setattr(WorkspaceRuntime, "start_replacement", fail_second)
    await service.update_configuration(
        "dream-change",
        cast(str, service.config_view()["revision"]),
        {"memory": {"schedule": "17 3 * * *"}},
    )
    task = service._config_apply_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=10)
    assert (
        cast(dict[str, object], service.config_view()["application"])["status"] == "failed-to-apply"
    )
    assert first.runtime is first_runtime
    assert second.runtime is second_runtime
    assert await WorkspaceScheduleStore(first.workspace_state).snapshot() == before


@pytest.mark.asyncio
async def test_client_expiry_during_preparation_does_not_resurrect_client_resources(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id)
    await workspace.claim(client.client_id, session_id)
    started, release = asyncio.Event(), asyncio.Event()
    prepare = workspace.prepare_configuration
    candidates: list[_PreparedWorkspaceGeneration] = []

    async def blocked(configuration: UserConfiguration) -> _PreparedWorkspaceGeneration:
        result = await prepare(configuration)
        candidates.append(result)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(workspace, "prepare_configuration", blocked)
    await _save(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    await service._expire_client_later(client.client_id, service._monotonic() - 1)
    release.set()
    await _applied(service)
    assert client.client_id not in service._clients
    assert session_id not in workspace.loops
    assert session_id not in workspace._claims
    assert candidates[0].runtime._closed
    assert all(loop._closed or loop._aborted for _, loop, _ in candidates[0].loops)


@pytest.mark.asyncio
async def test_project_removal_during_preparation_closes_unpublished_candidate(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    record = service.projects.register(path)
    started, release = asyncio.Event(), asyncio.Event()
    prepare = workspace.prepare_configuration
    candidates: list[_PreparedWorkspaceGeneration] = []

    async def blocked(configuration: UserConfiguration) -> _PreparedWorkspaceGeneration:
        result = await prepare(configuration)
        candidates.append(result)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(workspace, "prepare_configuration", blocked)
    await _save(service)
    await asyncio.wait_for(started.wait(), timeout=10)
    await asyncio.wait_for(complete_project_removal(service, client.client_id, record.project_id), timeout=10)
    release.set()
    await _applied(service)
    assert workspace.workspace_id not in service.workspaces
    assert not service.projects.list()
    assert candidates[0].runtime._closed
    assert workspace.runtime is not None and workspace.runtime._closed


@pytest.mark.parametrize(
    "generation_service",
    [
        '\n[mcp.servers.unavailable]\nenabled = true\ntransport = "stdio"\n'
        'command = "omni-acceptance-nonexistent-mcp-server"\nconnect_timeout = 1\n'
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_ordinary_mcp_unavailability_allows_replacement_generation(
    generation_service: LocalService,
    tmp_path: Path,
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    previous = workspace.runtime
    assert previous is not None
    assert previous.mcp_startup_report.failed_servers == ("unavailable",)
    await _save(service)
    await _applied(service)
    current = workspace.runtime
    assert current is not None and current is not previous
    assert current.mcp_startup_report.failed_servers == ("unavailable",)
    assert current.mcp_snapshot == ()
    assert not current._closed
    assert previous._closed
    assert workspace.configuration.runtime.max_iterations == 83


@pytest.mark.asyncio
async def test_save_during_completed_generation_notification_is_eventually_applied(
    generation_service: LocalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    notifying, release = asyncio.Event(), asyncio.Event()
    emit = service._emit_configuration_event
    blocked_once = False

    async def blocked_notification() -> None:
        nonlocal blocked_once
        await emit()
        status = cast(dict[str, object], service.config_view()["application"])["status"]
        if status == "active" and not blocked_once:
            blocked_once = True
            notifying.set()
            await release.wait()

    monkeypatch.setattr(service, "_emit_configuration_event", blocked_notification)
    initial_save = asyncio.create_task(_save(service))
    await asyncio.wait_for(notifying.wait(), timeout=10)
    await service.update_configuration(
        "next-generation",
        cast(str, service.config_view()["revision"]),
        {"runtime": {"max_iterations": 89}},
    )
    release.set()
    await asyncio.wait_for(initial_save, timeout=10)
    await _applied(service)
    assert workspace.configuration.runtime.max_iterations == 89
    application = cast(dict[str, object], service.config_view()["application"])
    assert application["active_revision"] == application["saved_revision"]


@pytest.mark.asyncio
async def test_replacement_schedule_add_confirms_new_default_above_client_override(
    generation_service: LocalService,
    tmp_path: Path,
) -> None:
    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id)
    claim = await workspace.claim(client.client_id, session_id)
    selected = await service.handle_management(
        client.client_id,
        workspace.workspace_id,
        session_id,
        "permission",
        {"request_id": "select-current-default", "permission_level": "workspace-write"},
        claim_version=claim.version,
        claim_credential=claim.credential,
    )
    assert selected["published_permission_level"] == "workspace-write"
    await service.update_configuration(
        "raise-schedule-default",
        cast(str, service.config_view()["revision"]),
        {"runtime": {"permission_level": "full-access"}},
    )
    await _applied(service)
    assert client.permission_control.current() == "workspace-write"
    assert client.permission_control.configured() == "full-access"
    gateway = claim.loop._new_run_gateway(
        permission_snapshot=client.permission_control.snapshot(claim.loop._exec_host.resolved_shell)
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id="schedule-default-escalation",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "privileged future job", "every_seconds": 3600}
            ),
        ),
        confirmation=decline,
    )
    assert len(requests) == 1
    assert "full-access" in requests[0].reason
    assert "workspace-write" in requests[0].reason
    assert result.status == "refused"
    assert await workspace.schedule_service.public_snapshot() == ()


@pytest.mark.parametrize(
    "invalid_configuration",
    [
        MINIMAL_VALID_CONFIG.replace("[runtime]\n", "[runtime]\nmax_iterations = 1\n"),
        MINIMAL_VALID_CONFIG + '\n[mcp.servers.invalid]\ntransport = "invalid"\n',
    ],
)
@pytest.mark.asyncio
async def test_invalid_external_configuration_retains_active_generation(
    generation_service: LocalService,
    tmp_path: Path,
    invalid_configuration: str,
) -> None:
    from omni.service.errors import ServiceError

    service = generation_service
    client = await service.register_client("cli")
    path = tmp_path / "workspace"
    path.mkdir()
    workspace = await service.attach_workspace(client.client_id, path)
    previous = workspace.runtime
    assert previous is not None
    active_revision = cast(dict[str, object], service.config_view()["application"])[
        "active_revision"
    ]
    loader = ConfigLoader(service.agent_home)
    loader.path.write_text(invalid_configuration, encoding="utf-8")
    invalid_bytes = loader.path.read_bytes()

    view = service.config_view()
    application = cast(dict[str, object], view["application"])
    assert application["status"] == "failed-to-apply"
    assert application["active_revision"] == active_revision
    assert application["saved_revision"] == loader.revision()
    assert workspace.runtime is previous and not previous._closed
    assert service._config_apply_task is None
    with pytest.raises(ServiceError) as rejected:
        await service.retry_configuration("retry-invalid-external", cast(str, view["revision"]))
    assert rejected.value.code == "config_invalid"
    assert loader.path.read_bytes() == invalid_bytes
    assert workspace.runtime is previous and not previous._closed

    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    await service.update_configuration(
        "corrected-external-save", loader.revision(), {"runtime": {"max_iterations": 83}}
    )
    await _applied(service)
    assert workspace.runtime is not previous
    assert workspace.configuration.runtime.max_iterations == 83
