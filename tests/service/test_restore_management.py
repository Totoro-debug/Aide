"""Service-boundary Restore tests using real Session and file state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from myclaw.agent.session.backup_store import FileBackupStore
from myclaw.agent.session.restore import RestoreManager
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.config import ConfigLoader
from myclaw.service.discovery import create_credential
from myclaw.service.errors import ServiceError
from myclaw.service.runtime import LocalService, SessionClaim, WorkspaceServiceRuntime
from myclaw.service.transport import create_app
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home


@pytest.fixture
def restore_provider(monkeypatch: pytest.MonkeyPatch) -> _ConcurrentProvider:
    provider = _ConcurrentProvider(block_b=True)
    monkeypatch.setattr("myclaw.service.runtime.create_provider", lambda *_args: provider)
    return provider


@pytest_asyncio.fixture
async def restore_case(
    tmp_path: Path,
    restore_provider: _ConcurrentProvider,
) -> AsyncIterator[tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    state = WorkspaceState(workspace_path)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC))
    target = workspace_path / "tracked.txt"
    target.write_bytes(b"before restore")
    token = uuid4()
    session.commit_agent_run(
        [{"role": "user", "content": "branch to restore"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=token,
    )
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(token, target)
    assert ticket is not None
    target.write_bytes(b"current branch")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        owner = await service.register_client("web")
        await service.connect_client(owner.client_id, _CollectingSink())
        workspace = await service.attach_workspace(owner.client_id, workspace_path)
        claim = await workspace.claim(owner.client_id, session.session_id)
        yield service, workspace, owner.client_id, claim, target
    finally:
        await service.stop()


async def _restore_request(
    service: LocalService,
    workspace: WorkspaceServiceRuntime,
    client_id: str,
    claim: SessionClaim,
    action: str,
    request_id: str,
    **payload: object,
) -> dict[str, object]:
    return await service.handle_management(
        client_id,
        workspace.workspace_id,
        claim.session_id,
        action,
        {"request_id": request_id, **payload},
        claim_version=claim.version,
        claim_credential=claim.credential,
    )


@pytest.mark.asyncio
async def test_restore_plan_and_cancel_are_bound_to_inspected_session(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, claim, target = restore_case
    other_id = await _persist_session(
        workspace.workspace_path,
        home=service.agent_home,
        title="Other",
        created_at=datetime(2026, 10, 1, 12, 1, tzinfo=UTC),
        content="Other private text",
    )
    other = await workspace.claim(owner, other_id)
    await _restore_request(
        service, workspace, owner, claim, "restore/inspect", "inspect", anchor_id=1
    )
    await _restore_request(service, workspace, owner, other, "restore/cancel", "cancel-other")
    assert workspace._restore_session_id == claim.session_id
    result = await _restore_request(
        service,
        workspace,
        owner,
        other,
        "restore/execute",
        "execute-other",
        plan={"anchor_id": 1},
        mode="files",
    )
    assert result.get("restore_result") is None
    assert target.read_bytes() == b"current branch"
    await _restore_request(service, workspace, owner, claim, "restore/cancel", "cancel")
    assert workspace._restore_owner is None
    assert not workspace._restore_schedule_paused
    assert claim.loop.foreground_input_admitted()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["restore/inspect", "restore/result", "restore/acknowledge"])
async def test_restore_cached_response_rejects_released_claim(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    action: str,
) -> None:
    service, workspace, owner, claim, _target = restore_case
    payload = {"anchor_id": 1} if action == "restore/inspect" else {}
    await _restore_request(service, workspace, owner, claim, action, "cached", **payload)
    await workspace.release(owner, claim.session_id)
    await workspace.claim(owner, claim.session_id)
    with pytest.raises(ServiceError, match="Claim") as error:
        await _restore_request(service, workspace, owner, claim, action, "cached", **payload)
    assert error.value.code == "stale_claim"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["restore/inspect", "restore/execute", "restore/result", "restore/acknowledge"]
)
async def test_other_client_cannot_inspect_execute_read_or_ack_restore(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    action: str,
) -> None:
    service, workspace, _owner, claim, target = restore_case
    stranger = await service.register_client("web")
    await service.attach_workspace(stranger.client_id, workspace.workspace_path)
    with pytest.raises(ServiceError) as error:
        await _restore_request(
            service,
            workspace,
            stranger.client_id,
            claim,
            action,
            "unauthorized",
            anchor_id=1,
            plan={"anchor_id": 1},
            mode="files",
        )
    assert error.value.code == "stale_claim"
    assert target.read_bytes() == b"current branch"
    assert workspace._restore_owner is None


@pytest.mark.asyncio
async def test_restore_request_id_cannot_be_reused_with_another_anchor_or_credential(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, claim, target = restore_case
    await _restore_request(service, workspace, owner, claim, "restore/inspect", "id", anchor_id=1)
    for payload, credential in [({"anchor_id": 2}, claim.credential), ({"anchor_id": 1}, "wrong")]:
        with pytest.raises(ServiceError) as error:
            await service.handle_management(
                owner,
                workspace.workspace_id,
                claim.session_id,
                "restore/inspect",
                {"request_id": "id", **payload},
                claim_version=claim.version,
                claim_credential=credential,
            )
        assert error.value.code == "request_reused"
    assert target.read_bytes() == b"current branch"


@pytest.mark.asyncio
async def test_claim_release_waits_for_the_restore_transaction(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, owner, claim, target = restore_case
    await _restore_request(
        service, workspace, owner, claim, "restore/inspect", "inspect", anchor_id=1
    )
    transaction_entered = asyncio.Event()
    release_waiting = asyncio.Event()
    finish_transaction = asyncio.Event()
    original_continue = RestoreManager._continue_pending
    original_wait = workspace._wait_restore_commit

    async def continue_pending(manager: RestoreManager, pending: Any) -> Any:
        transaction_entered.set()
        await finish_transaction.wait()
        return await original_continue(manager, pending)

    async def wait_commit() -> None:
        release_waiting.set()
        await original_wait()

    monkeypatch.setattr(RestoreManager, "_continue_pending", continue_pending)
    monkeypatch.setattr(workspace, "_wait_restore_commit", wait_commit)
    execute = asyncio.create_task(
        _restore_request(
            service,
            workspace,
            owner,
            claim,
            "restore/execute",
            "execute",
            plan={"anchor_id": 1},
            mode="files",
        )
    )
    release: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(transaction_entered.wait(), 5)
        release = asyncio.create_task(workspace.release(owner, claim.session_id))
        await asyncio.wait_for(release_waiting.wait(), 5)
        assert not release.done()
        assert claim.session_id in workspace._claims
        finish_transaction.set()
        result = await asyncio.wait_for(execute, 5)
        await asyncio.wait_for(release, 5)
        assert result.get("restore_result") is not None
        assert target.read_bytes() == b"before restore"
        assert claim.session_id not in workspace._claims
        assert not RestoreManager(
            workspace.workspace_state, claim.session_id
        ).has_pending_transaction()
    finally:
        finish_transaction.set()
        await asyncio.gather(
            execute, *([release] if release is not None else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_reading_absent_restore_result_creates_no_restore_storage(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, _claim, _target = restore_case
    other_id = await _persist_session(
        workspace.workspace_path,
        home=service.agent_home,
        title="No Restore",
        created_at=datetime(2026, 10, 1, 12, 1, tzinfo=UTC),
        content="Unchanged session",
    )
    other = await workspace.claim(owner, other_id)
    root = workspace.workspace_state.path / "restore" / other_id
    assert not root.exists()
    for action in ["restore/result", "restore/acknowledge"]:
        result = await _restore_request(service, workspace, owner, other, action, action)
        assert result.get("restore_result") is None
        assert not root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["release", "disconnect"])
async def test_restore_inspection_revalidates_claim_after_wait(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    monkeypatch: pytest.MonkeyPatch,
    invalidate: str,
) -> None:
    service, workspace, owner, claim, target = restore_case
    waiting = asyncio.Event()
    resume = asyncio.Event()
    original = workspace.schedule_service.pause_and_wait_idle

    async def pause() -> None:
        await original()
        waiting.set()
        await resume.wait()

    monkeypatch.setattr(workspace.schedule_service, "pause_and_wait_idle", pause)
    task = asyncio.create_task(
        _restore_request(
            service,
            workspace,
            owner,
            claim,
            "restore/inspect",
            "waiting",
            anchor_id=1,
        )
    )
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        if invalidate == "release":
            await workspace.release(owner, claim.session_id)
        else:
            await service.disconnect_client(owner)
        resume.set()
        with pytest.raises(ServiceError) as error:
            await task
        assert error.value.code == "stale_claim"
        assert target.read_bytes() == b"current branch"
        assert workspace._restore_owner is None
        assert not workspace._restore_schedule_paused
        assert not workspace._restore_plans
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["conversation-only", "files"])
async def test_restore_preserves_an_unrelated_active_run(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    restore_provider: _ConcurrentProvider,
    mode: str,
) -> None:
    service, workspace, owner, claim, target = restore_case
    provider = restore_provider
    other = await service.register_client("web")
    await service.attach_workspace(other.client_id, workspace.workspace_path)
    sink = _CollectingSink()
    await service.connect_client(other.client_id, sink)
    other_id = await workspace.create_draft(other.client_id)
    await service.claim(other.client_id, workspace.workspace_id, other_id)
    other_claim = workspace._claims[other_id]
    run_id = str(uuid4())
    await workspace.input(
        other.client_id, other_claim.session_id, other_claim.version, "session-b", run_id
    )
    try:
        await asyncio.wait_for(provider.session_b_started.wait(), 5)
        await _restore_request(
            service, workspace, owner, claim, "restore/inspect", "inspect", anchor_id=1
        )
        target.write_bytes(b"changed by other session")
        await _restore_request(
            service,
            workspace,
            owner,
            claim,
            "restore/execute",
            "execute",
            plan={"anchor_id": 1},
            mode=mode,
        )
        assert not provider.session_b_cancelled.is_set()
        assert other_claim.loop.has_active_run
        assert workspace._claims[other_claim.session_id] is other_claim
        assert target.read_bytes() == (
            b"before restore" if mode == "files" else b"changed by other session"
        )
        assert workspace.session_snapshot(claim.session_id)["messages"] == []
        provider.release_b.set()
        await asyncio.wait_for(sink.wait_for("run.completed", run_id), 5)
    finally:
        provider.release_b.set()


@pytest.mark.asyncio
async def test_failed_restore_result_and_ack_survive_new_client_and_restart(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, claim, target = restore_case
    await _restore_request(
        service, workspace, owner, claim, "restore/inspect", "inspect", anchor_id=1
    )
    target.unlink()
    target.mkdir()
    executed = await _restore_request(
        service,
        workspace,
        owner,
        claim,
        "restore/execute",
        "execute",
        plan={"anchor_id": 1},
        mode="files",
    )
    result = executed["restore_result"]
    assert isinstance(result, dict) and result["file_results"][0]["status"] == "failed"
    await workspace.release(owner, claim.session_id)
    new = await service.register_client("web")
    await service.attach_workspace(new.client_id, workspace.workspace_path)
    next_claim = await workspace.claim(new.client_id, claim.session_id)
    fetched = await _restore_request(
        service, workspace, new.client_id, next_claim, "restore/result", "result"
    )
    assert fetched["restore_result"] == result
    await service.stop()
    restarted = LocalService(
        service.agent_home, ConfigLoader(service.agent_home).load_for_startup()
    )
    await restarted.start()
    try:
        client = await restarted.register_client("web")
        active = await restarted.attach_workspace(client.client_id, workspace.workspace_path)
        recovered_claim = await active.claim(client.client_id, claim.session_id)
        fetched = await _restore_request(
            restarted, active, client.client_id, recovered_claim, "restore/result", "result"
        )
        assert fetched["restore_result"] == result
        acknowledged = await _restore_request(
            restarted, active, client.client_id, recovered_claim, "restore/acknowledge", "ack"
        )
        assert (
            cast(dict[str, Any], acknowledged["restore_result"])[
                "failure_notification_acknowledged"
            ]
            is True
        )
        assert not RestoreManager(
            active.workspace_state, claim.session_id
        ).has_pending_transaction()
        await active.release(client.client_id, claim.session_id)
        fresh = await restarted.register_client("web")
        await restarted.attach_workspace(fresh.client_id, active.workspace_path)
        fresh_claim = await active.claim(fresh.client_id, claim.session_id)
        fetched = await _restore_request(
            restarted, active, fresh.client_id, fresh_claim, "restore/result", "fresh"
        )
        assert (
            cast(dict[str, Any], fetched["restore_result"])["failure_notification_acknowledged"]
            is True
        )
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_restore_management_requires_claim_is_idempotent_and_preserves_other_claim(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    state = WorkspaceState(workspace_path)
    state.initialize(agent_home_root=home.path)

    session = Session.create(
        state,
        now=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )
    target = workspace_path / "tracked.txt"
    target.write_bytes(b"before restore")
    run_token = uuid4()
    session.commit_agent_run(
        [{"role": "user", "content": "branch to restore"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=run_token,
    )
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    target.write_bytes(b"current branch")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    other_session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Other session",
        created_at=datetime(2026, 10, 1, 12, 1, tzinfo=UTC),
        content="Other private conversation",
    )

    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        owner = await service.register_client("web")
        other = await service.register_client("web")
        workspace = await service.attach_workspace(owner.client_id, workspace_path)
        await service.attach_workspace(other.client_id, workspace_path)
        claim = await workspace.claim(owner.client_id, session.session_id)
        other_claim = await workspace.claim(other.client_id, other_session_id)

        with pytest.raises(ServiceError) as missing_claim:
            await service.handle_management(
                owner.client_id,
                workspace.workspace_id,
                session.session_id,
                "restore/inspect",
                {"request_id": "restore-missing-claim", "anchor_id": 1},
            )
        assert missing_claim.value.code == "stale_claim"

        with pytest.raises(ServiceError) as stale_claim:
            await service.handle_management(
                owner.client_id,
                workspace.workspace_id,
                session.session_id,
                "restore/inspect",
                {"request_id": "restore-stale-claim", "anchor_id": 1},
                claim_version=claim.version - 1,
                claim_credential=claim.credential,
            )
        assert stale_claim.value.code == "stale_claim"

        inspected = await service.handle_management(
            owner.client_id,
            workspace.workspace_id,
            session.session_id,
            "restore/inspect",
            {"request_id": "restore-inspect", "anchor_id": 1},
            claim_version=claim.version,
            claim_credential=claim.credential,
        )
        plan = inspected["restore_plan"]
        assert isinstance(plan, dict)
        assert plan["session_id"] == session.session_id
        assert plan["targets"][0]["canonical_target"] == str(target.resolve())

        target.write_bytes(b"changed by another Session")
        execute_version = claim.version
        execute_credential = claim.credential
        payload = {
            "request_id": "restore-execute",
            "plan": {"anchor_id": 1},
            "mode": "files",
        }
        executed = await service.handle_management(
            owner.client_id,
            workspace.workspace_id,
            session.session_id,
            "restore/execute",
            payload,
            claim_version=execute_version,
            claim_credential=execute_credential,
        )
        duplicate = await service.handle_management(
            owner.client_id,
            workspace.workspace_id,
            session.session_id,
            "restore/execute",
            payload,
            claim_version=execute_version,
            claim_credential=execute_credential,
        )

        assert duplicate == executed
        assert target.read_bytes() == b"before restore"
        assert workspace._claims[other_session_id] is other_claim
        assert other_claim.client_id == other.client_id
        restore_result = executed["restore_result"]
        assert isinstance(restore_result, dict)
        assert restore_result["session_id"] == session.session_id
        assert "Other private conversation" not in str(restore_result)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_restore_management_http_requires_csrf_and_claim_headers(tmp_path: Path) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    token = create_credential(home)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    state = WorkspaceState(workspace_path)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: datetime(2026, 10, 1, 13, 0, tzinfo=UTC))
    target = workspace_path / "tracked.txt"
    target.write_bytes(b"before")
    run_token = uuid4()
    session.commit_agent_run(
        [{"role": "user", "content": "http restore"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=run_token,
    )
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    server = TestServer(create_app(service), host="127.0.0.1")
    try:
        client = await service.register_client("web")
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        claim = await workspace.claim(client.client_id, session.session_id)
        await server.start_server()
        url = server.make_url(
            f"/api/v1/workspaces/{workspace.workspace_id}/management/restore/inspect"
        )
        payload = {
            "request_id": "http-restore-inspect",
            "current_session_id": session.session_id,
            "claim_version": claim.version,
            "anchor_id": 1,
        }
        base_headers = {
            "Authorization": f"Bearer {token}",
            "X-MyClaw-Client": client.client_id,
            "X-MyClaw-Claim": claim.credential,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(url, headers=base_headers, json=payload) as response:
                assert response.status == 403
            async with http.post(
                url,
                headers={**base_headers, "X-MyClaw-CSRF": token},
                json=payload,
            ) as response:
                assert response.status == 200
                body = await response.json()
                assert body["result"]["restore_plan"]["session_id"] == session.session_id
            result_url = server.make_url(
                f"/api/v1/workspaces/{workspace.workspace_id}/management/restore/result"
            )
            async with http.get(
                result_url,
                headers={
                    **base_headers,
                    "X-MyClaw-Request": "http-restore-result",
                    "X-MyClaw-Session": session.session_id,
                },
                params={"session_id": session.session_id, "claim_version": str(claim.version)},
            ) as response:
                assert response.status == 200
                assert (await response.json())["result"].get("restore_result") is None
            async with http.post(
                url,
                headers={**base_headers, "X-MyClaw-CSRF": token},
                json={**payload, "request_id": "http-restore-stale", "claim_version": 99},
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "stale_claim"
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_restore_listing_rejects_selected_active_and_queued_input(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    restore_provider: _ConcurrentProvider,
    queued: bool,
) -> None:
    service, workspace, owner, claim, target = restore_case
    state = workspace.loops[claim.session_id]
    if queued:
        await state.bus.pause_inbound_delivery()
    await workspace.input(owner, claim.session_id, claim.version, "session-b", "restore-busy")
    if not queued:
        await asyncio.wait_for(restore_provider.session_b_started.wait(), timeout=5)
    try:
        result = await _restore_request(
            service, workspace, owner, claim, "dispatch", "restore-busy-list", command="/restore"
        )
        assert result.get("restore_listing") is None
        assert str(result["output"]).startswith("model_invalid_request:")
        assert claim.loop.foreground_input_admitted()
        assert target.read_bytes() == b"current branch"
    finally:
        restore_provider.release_b.set()
        if queued:
            await state.bus.resume_inbound_delivery()


@pytest.mark.asyncio
async def test_restore_double_listing_and_cancel_preserve_claim_and_durable_branch(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, claim, target = restore_case
    before = Session.load(workspace.workspace_state, claim.session_id).messages
    first = await _restore_request(
        service, workspace, owner, claim, "dispatch", "first-list", command="/restore"
    )
    assert first.get("restore_listing") is not None
    assert not claim.loop.foreground_input_admitted()
    second = await _restore_request(
        service, workspace, owner, claim, "dispatch", "second-list", command="/restore"
    )
    assert second.get("restore_listing") is None
    assert not claim.loop.foreground_input_admitted()
    await _restore_request(service, workspace, owner, claim, "restore/cancel", "cancel-list")
    assert claim.loop.foreground_input_admitted()
    assert workspace.require_claim(owner, claim.session_id, claim.version) is claim
    assert Session.load(workspace.workspace_state, claim.session_id).messages == before
    assert target.read_bytes() == b"current branch"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "readiness", "inspect", "execute"])
async def test_restore_precommit_failures_release_input_and_schedule_without_mutation(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service, workspace, owner, claim, target = restore_case
    before = Session.load(workspace.workspace_state, claim.session_id).messages
    if failure == "cancel":
        started, release = asyncio.Event(), asyncio.Event()

        async def wait() -> None:
            started.set()
            await release.wait()

        monkeypatch.setattr(claim.loop, "wait_for_restore_idle", wait)
        pending = asyncio.create_task(
            _restore_request(
                service, workspace, owner, claim, "dispatch", "cancel-wait", command="/restore"
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        if failure != "readiness":
            await _restore_request(
                service, workspace, owner, claim, "dispatch", "prepare-list", command="/restore"
            )

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("PRIVATE_RESTORE_FAILURE")

        async def fail_async(*_args: Any, **_kwargs: Any) -> Any:
            return fail()

        payload: dict[str, object]
        if failure == "readiness":
            monkeypatch.setattr(claim.loop, "wait_for_restore_idle", fail_async)
            action, payload = "dispatch", {"command": "/restore"}
        elif failure == "inspect":
            monkeypatch.setattr(RestoreManager, "inspect", fail)
            action, payload = "restore/inspect", {"anchor_id": 1}
        else:
            await _restore_request(
                service, workspace, owner, claim, "restore/inspect", "prepare-inspect", anchor_id=1
            )
            monkeypatch.setattr(RestoreManager, "execute", fail_async)
            action, payload = "restore/execute", {"plan": {"anchor_id": 1}, "mode": "files"}
        with pytest.raises(ServiceError if failure == "execute" else OSError) as raised:
            await _restore_request(
                service, workspace, owner, claim, action, "failed-restore", **payload
            )
        if failure == "execute":
            assert "PRIVATE" not in str(raised.value)
    assert claim.loop.foreground_input_admitted()
    assert workspace.schedule_admitted
    assert Session.load(workspace.workspace_state, claim.session_id).messages == before
    assert target.read_bytes() == b"current branch"
    assert not RestoreManager(workspace.workspace_state, claim.session_id).has_pending_transaction()


@pytest.mark.asyncio
async def test_restore_stale_durable_plan_preserves_new_branch_and_releases_barrier(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
) -> None:
    service, workspace, owner, claim, target = restore_case
    await _restore_request(
        service, workspace, owner, claim, "restore/inspect", "inspect-stale", anchor_id=1
    )
    changed = Session.load(workspace.workspace_state, claim.session_id)
    changed.commit_agent_run(
        [{"role": "user", "content": "new persisted branch"}],
        pending_last_compacted=changed.last_compacted,
        pending_action_summary="",
    )
    await changed.wait_for_pending_persist()
    result = await _restore_request(
        service,
        workspace,
        owner,
        claim,
        "restore/execute",
        "execute-stale",
        plan={"anchor_id": 1},
        mode="files",
    )
    assert result.get("restore_result") is None
    assert "stale" in str(result["output"])
    assert claim.loop.foreground_input_admitted()
    assert workspace.schedule_admitted
    assert Session.load(workspace.workspace_state, claim.session_id).messages == changed.messages
    assert target.read_bytes() == b"current branch"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["constructor", "binding", "preflight", "start"])
async def test_restore_rebuild_failure_keeps_durable_result_and_closes_admission(
    restore_case: tuple[LocalService, WorkspaceServiceRuntime, str, SessionClaim, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from myclaw.agent.loop import AgentLoop

    service, workspace, owner, claim, target = restore_case
    await _restore_request(
        service, workspace, owner, claim, "restore/inspect", "inspect-rebuild", anchor_id=1
    )
    old = claim.loop
    method = {
        "constructor": "__init__",
        "binding": "bind_confirmation_requester",
        "preflight": "preflight",
        "start": "start",
    }[failure_point]

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("PRIVATE_REBUILD_FAILURE")

    async def fail_async(*_args: Any, **_kwargs: Any) -> None:
        fail()

    with monkeypatch.context() as patched:
        # Restore this injected failure state on scope exit so the fixture can drain its owner.
        patched.setattr(workspace, "_restore_blocked", False)
        patched.setattr(AgentLoop, method, fail_async if method == "start" else fail)
        with pytest.raises(ServiceError) as raised:
            await _restore_request(
                service,
                workspace,
                owner,
                claim,
                "restore/execute",
                "execute-rebuild",
                plan={"anchor_id": 1},
                mode="files",
            )
        assert raised.value.code == "restore_failed"
        assert "PRIVATE" not in raised.value.message
        assert workspace._restore_blocked
        assert not workspace.schedule_admitted
        assert old._aborted
        assert target.read_bytes() == b"before restore"
        assert Session.load(workspace.workspace_state, claim.session_id).messages == []
        result = RestoreManager(workspace.workspace_state, claim.session_id).completed_result()
        assert result is not None and result.session_id == claim.session_id
        with pytest.raises(ServiceError) as blocked:
            await workspace.input(
                owner, claim.session_id, claim.version, "no new work", "blocked-restore"
            )
        assert blocked.value.code == "admission_closed"
