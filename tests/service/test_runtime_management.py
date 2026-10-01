"""Typed Runtime management scope, lifetime, replay, and CLI compatibility."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from myclaw.agent.loop import AgentLoop
from myclaw.agent.tools.permission import PermissionContext
from myclaw.config.config import ConfigLoader
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.service.discovery import create_credential
from myclaw.service.errors import ServiceError
from myclaw.service.runtime import ClientState, LocalService, SessionClaim, WorkspaceServiceRuntime
from myclaw.service.transport import create_app
from tests.fixtures import FakeClock
from tests.service.test_protocol_contract import _validator
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home


@dataclass
class ManagementCase:
    service: LocalService
    workspace: WorkspaceServiceRuntime
    first: ClientState
    second: ClientState
    claim: SessionClaim
    other_claim: SessionClaim
    clock: FakeClock
    wake_timer: asyncio.Event
    provider: _ConcurrentProvider


@pytest_asyncio.fixture
async def management_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ManagementCase]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    path = tmp_path / "workspace"
    path.mkdir()
    session_id = await _persist_session(
        path,
        home=home,
        title="Selected scope",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        content="selected private content",
    )
    clock = FakeClock(datetime(2026, 10, 1, tzinfo=UTC))
    wake_timer = asyncio.Event()

    async def wait_for_timer(_seconds: float) -> None:
        await wake_timer.wait()
        wake_timer.clear()

    provider = _ConcurrentProvider()
    monkeypatch.setattr("myclaw.service.runtime.create_provider", lambda *_args: provider)
    service = LocalService(
        home,
        ConfigLoader(home).load_for_startup(),
        reconnect_timeout=30,
        monotonic_now=clock.monotonic,
        sleep=wait_for_timer,
    )
    await service.start()
    try:
        first = await service.register_client("web")
        second = await service.register_client("cli")
        await service.connect_client(first.client_id, _CollectingSink())
        await service.connect_client(second.client_id, _CollectingSink())
        workspace = await service.attach_workspace(first.client_id, path)
        await service.attach_workspace(second.client_id, path)
        await service.claim(first.client_id, workspace.workspace_id, session_id)
        claim = workspace._claims[session_id]
        other_id = await workspace.create_draft(second.client_id)
        await service.claim(second.client_id, workspace.workspace_id, other_id)
        yield ManagementCase(
            service,
            workspace,
            first,
            second,
            claim,
            workspace._claims[other_id],
            clock,
            wake_timer,
            provider,
        )
    finally:
        await service.stop()


async def _request(
    case: ManagementCase,
    action: str,
    *,
    client: ClientState | None = None,
    workspace: WorkspaceServiceRuntime | None = None,
    claim: SessionClaim | None = None,
    request_id: str | None = None,
    **payload: object,
) -> dict[str, object]:
    selected_claim = case.claim if claim is None else claim
    result = await case.service.handle_management(
        (case.first if client is None else client).client_id,
        (case.workspace if workspace is None else workspace).workspace_id,
        selected_claim.session_id,
        action,
        {"request_id": request_id or str(uuid4()), **payload},
        claim_version=selected_claim.version,
        claim_credential=selected_claim.credential,
    )
    _validator("management_response").validate(
        {"request_id": request_id or "generated", "result": result}
    )
    if "status_view" in result:
        _validator("runtime_status").validate(result["status_view"])
    return result


async def _status(case: ManagementCase, **kwargs: Any) -> dict[str, object]:
    return cast(dict[str, object], (await _request(case, "status", **kwargs))["status_view"])


@pytest.mark.asyncio
async def test_status_matches_selected_session_without_private_content_or_credentials(
    management_case: ManagementCase,
) -> None:
    case = management_case
    selected = await _status(case)
    other = await _status(case, client=case.second, claim=case.other_claim)
    assert selected["session_message_count"] == 1
    assert other["session_message_count"] == 0
    assert cast(int, selected["projected_next_request_tokens"]) > 0
    assert cast(int, selected["available_context"]) > 0
    assert selected["configured_permission_level"] == "workspace-write"
    assert selected["current_permission_level"] == "workspace-write"
    encoded = json.dumps(selected)
    assert "selected private content" not in encoded
    for credential in (case.claim.credential, case.first.reconnect_credential, "minimal-secret"):
        assert credential not in encoded
    with pytest.raises(ServiceError) as foreign:
        await _request(case, "status", client=case.second)
    assert foreign.value.code == "stale_claim"


@pytest.mark.asyncio
async def test_client_permission_survives_session_workspace_and_reconnect_without_affecting_other_client(
    management_case: ManagementCase,
    tmp_path: Path,
) -> None:
    case = management_case
    await _request(case, "permission", permission_level="read-only")
    assert (await _status(case, client=case.second, claim=case.other_claim))[
        "current_permission_level"
    ] == "workspace-write"
    next_id = await case.workspace.create_draft(case.first.client_id)
    await case.service.claim(case.first.client_id, case.workspace.workspace_id, next_id)
    next_claim = case.workspace._claims[next_id]
    assert (await _status(case, claim=next_claim))["current_permission_level"] == "read-only"

    other_path = tmp_path / "other-workspace"
    other_path.mkdir()
    other_workspace = await case.service.attach_workspace(case.first.client_id, other_path)
    draft_id = await other_workspace.create_draft(case.first.client_id)
    await case.service.claim(case.first.client_id, other_workspace.workspace_id, draft_id)
    other_claim = other_workspace._claims[draft_id]
    assert (await _status(case, workspace=other_workspace, claim=other_claim))[
        "current_permission_level"
    ] == "read-only"
    with pytest.raises(ServiceError) as wrong_workspace:
        await _request(case, "status", workspace=other_workspace, claim=case.claim)
    assert wrong_workspace.value.code == "stale_claim"

    await case.service.disconnect_client(case.first.client_id)
    case.clock.advance(29)
    reconnected = await case.service.register_client("web", case.first.reconnect_credential)
    assert reconnected is case.first
    await case.service.connect_client(reconnected.client_id, _CollectingSink())
    assert (await _status(case, workspace=other_workspace, claim=other_claim))[
        "current_permission_level"
    ] == "read-only"
    assert (await _status(case, client=case.second, claim=case.other_claim))[
        "current_permission_level"
    ] == "workspace-write"


@pytest.mark.asyncio
async def test_permission_resets_only_after_client_expiry_at_thirty_seconds(
    management_case: ManagementCase,
) -> None:
    case = management_case
    await _request(case, "permission", permission_level="full-access")
    reconnect = case.first.reconnect_credential
    await case.service.disconnect_client(case.first.client_id)
    expiry_task = case.first.disconnect_task
    assert expiry_task is not None
    case.clock.advance(29)
    assert case.service.client_permission(case.first.client_id).current() == "full-access"
    case.clock.advance(1)
    with pytest.raises(ServiceError) as expired:
        await case.service.register_client("web", reconnect)
    assert expired.value.code == "stale_client"
    case.wake_timer.set()
    await asyncio.wait_for(expiry_task, timeout=2)
    assert case.first.client_id not in case.service._clients
    replacement = await case.service.register_client("web")
    await case.service.connect_client(replacement.client_id, _CollectingSink())
    next_id = await case.workspace.create_draft(replacement.client_id)
    await case.service.claim(replacement.client_id, case.workspace.workspace_id, next_id)
    status = await _status(case, client=replacement, claim=case.workspace._claims[next_id])
    assert status["current_permission_level"] == "workspace-write"


def _action_payload(action: str) -> dict[str, Any]:
    if action == "permission":
        return {"permission_level": "read-only"}
    if action == "effort":
        return {"effort": "high"}
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "permission", "effort"])
async def test_typed_cached_response_rejects_released_and_replaced_claim(
    management_case: ManagementCase,
    action: str,
) -> None:
    case = management_case
    payload = _action_payload(action)
    initial = await _request(case, action, request_id="cached", **payload)
    assert await _request(case, action, request_id="cached", **payload) == initial
    await case.workspace.release(case.first.client_id, case.claim.session_id, close_idle=False)
    await case.workspace.claim(case.first.client_id, case.claim.session_id)
    with pytest.raises(ServiceError) as stale:
        await _request(case, action, request_id="cached", **payload)
    assert stale.value.code == "stale_claim"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "permission", "effort"])
async def test_management_revalidates_claim_after_waiting_for_client_lock(
    management_case: ManagementCase,
    action: str,
) -> None:
    case = management_case
    await case.first.management_lock.acquire()
    task = asyncio.create_task(_request(case, action, **_action_payload(action)))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        await case.workspace.release(case.first.client_id, case.claim.session_id, close_idle=False)
        await case.workspace.claim(case.first.client_id, case.claim.session_id)
    finally:
        case.first.management_lock.release()
    with pytest.raises(ServiceError) as stale:
        await task
    assert stale.value.code == "stale_claim"
    assert case.service.client_permission(case.first.client_id).current() == "workspace-write"
    assert case.workspace.runtime is not None
    assert case.workspace.runtime.router.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_request_id_reuse_cannot_change_payload_action_session_or_workspace(
    management_case: ManagementCase,
    tmp_path: Path,
) -> None:
    case = management_case
    await _request(case, "permission", request_id="unique", permission_level="read-only")
    attempts: list[dict[str, Any]] = [
        {"action": "permission", "permission_level": "full-access"},
        {"action": "effort", "effort": "high"},
        {"action": "status"},
    ]
    for attempt in attempts:
        action = attempt.pop("action")
        with pytest.raises(ServiceError) as reused:
            await _request(case, action, request_id="unique", **attempt)
        assert reused.value.code == "request_reused"
    next_id = await case.workspace.create_draft(case.first.client_id)
    next_claim = await case.workspace.claim(case.first.client_id, next_id)
    with pytest.raises(ServiceError) as session_reused:
        await _request(
            case, "permission", claim=next_claim, request_id="unique", permission_level="read-only"
        )
    assert session_reused.value.code == "request_reused"
    path = tmp_path / "different-workspace"
    path.mkdir()
    workspace = await case.service.attach_workspace(case.first.client_id, path)
    session_id = await workspace.create_draft(case.first.client_id)
    claim = await workspace.claim(case.first.client_id, session_id)
    with pytest.raises(ServiceError) as workspace_reused:
        await _request(
            case,
            "permission",
            workspace=workspace,
            claim=claim,
            request_id="unique",
            permission_level="read-only",
        )
    assert workspace_reused.value.code == "request_reused"
    assert case.service.client_permission(case.first.client_id).current() == "read-only"


@pytest.mark.asyncio
async def test_invalid_effort_does_not_publish_or_persist_and_preserves_cli_error(
    management_case: ManagementCase,
) -> None:
    case = management_case
    config_path = case.service.agent_home.path / "config.toml"
    original = config_path.read_bytes()
    result = await _request(case, "effort", effort="unsupported")
    assert result["published_effort"] is None
    assert cast(str, result["output"]).startswith("config_invalid:")
    assert config_path.read_bytes() == original
    assert (await _status(case))["chat_reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_effort_is_workspace_runtime_setting_with_shared_cli_and_persistence_contract(
    management_case: ManagementCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    result = await _request(case, "effort", effort="high")
    assert result["published_effort"] == "high"
    assert (await _status(case, client=case.second, claim=case.other_claim))[
        "chat_reasoning_effort"
    ] == "high"
    cli = case.workspace.management_dispatcher(case.second.client_id, case.other_claim.session_id)
    assert (await cli.dispatch("/effort")).effort_selection == "high"
    saved = ConfigLoader(case.service.agent_home).load_for_startup()
    assert saved.models.routes["default"].reasoning_effort == "high"

    path = tmp_path / "new-runtime"
    path.mkdir()
    workspace = await case.service.attach_workspace(case.first.client_id, path)
    draft = await workspace.create_draft(case.first.client_id)
    claim = await workspace.claim(case.first.client_id, draft)
    assert (await _status(case, workspace=workspace, claim=claim))[
        "chat_reasoning_effort"
    ] == "medium"
    original = (case.service.agent_home.path / "config.toml").read_bytes()

    def fail_save(_loader: ConfigLoader, _effort: object) -> None:
        raise OSError("test save failure")

    monkeypatch.setattr(ConfigLoader, "update_reasoning_effort", fail_save)
    failed_save = await _request(case, "effort", effort="max")
    assert failed_save["published_effort"] == "max"
    assert failed_save["output"] == "Chat reasoning effort: max"
    assert (await _status(case))["chat_reasoning_effort"] == "max"
    assert (await cli.dispatch("/effort")).effort_selection == "max"
    assert (case.service.agent_home.path / "config.toml").read_bytes() == original


@pytest.mark.asyncio
async def test_concurrent_client_effort_updates_keep_last_runtime_and_persisted_value(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    config_path = case.service.agent_home.path / "config.toml"
    with config_path.open("a", encoding="utf-8") as source:
        source.write(
            "\n# Preserve this configuration comment.\n[models.routes.chat]\n"
            "provider_id = 'primary'\nmodel = 'small-model'\ncontext_window = 8192\n"
            "max_output = 1024\ntemperature = 0\ntimeout = 30\nreasoning_effort = 'medium'\n"
        )
    published: list[str] = []
    original = ConfigLoader.update_reasoning_effort

    def save(loader: ConfigLoader, effort: Any) -> None:
        original(loader, effort)
        published.append(effort)

    monkeypatch.setattr(ConfigLoader, "update_reasoning_effort", save)
    first, second = await asyncio.gather(
        _request(case, "effort", effort="low"),
        _request(case, "effort", client=case.second, claim=case.other_claim, effort="xhigh"),
    )
    assert first["published_effort"] == "low"
    assert second["published_effort"] == "xhigh"
    assert set(published) == {"low", "xhigh"}
    final = published[-1]
    saved = ConfigLoader(case.service.agent_home).load_for_startup()
    assert saved.models.routes["default"].reasoning_effort == final
    assert saved.models.routes["chat"].reasoning_effort == final
    assert "# Preserve this configuration comment." in config_path.read_text(encoding="utf-8")
    assert (await _status(case))["chat_reasoning_effort"] == final
    assert (await _status(case, client=case.second, claim=case.other_claim))[
        "chat_reasoning_effort"
    ] == final


@pytest.mark.asyncio
async def test_effort_applies_to_next_real_cli_foreground_call_and_usage_matches_session(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    efforts: list[object] = []
    original = case.provider.stream

    def stream(**kwargs: Any) -> Any:
        efforts.append(kwargs["reasoning_effort"])
        return original(**kwargs)

    monkeypatch.setattr(case.provider, "stream", stream)
    await _request(case, "effort", effort="high")
    sink = cast(_CollectingSink, case.second.sink)
    await case.workspace.input(
        case.second.client_id,
        case.other_claim.session_id,
        case.other_claim.version,
        "cli usage evidence",
        "usage-run",
    )
    await asyncio.wait_for(sink.wait_for("run.completed", "usage-run"), timeout=2)
    assert efforts and all(effort == "high" for effort in efforts)
    selected = await _status(case)
    cli_status = await _status(case, client=case.second, claim=case.other_claim)
    assert selected["session_message_count"] == 1
    assert cli_status["session_message_count"] == 2
    assert cast(dict[str, int], cli_status["cumulative_usage"])["total_tokens"] > 0
    assert json.dumps(cli_status).find("cli usage evidence") == -1


@pytest.mark.asyncio
async def test_schedule_run_uses_configured_permission_snapshot_after_client_override(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    contexts: list[PermissionContext] = []
    provider_called = asyncio.Event()
    finished = asyncio.Event()
    original = AgentLoop._new_run_gateway
    original_complete = case.provider.complete
    original_run = AgentLoop.run_schedule_job

    async def complete(**kwargs: Any) -> Any:
        provider_called.set()
        return await original_complete(**kwargs)

    async def run_schedule(loop: AgentLoop, *args: Any, **kwargs: Any) -> None:
        await original_run(loop, *args, **kwargs)
        finished.set()

    def capture_gateway(loop: AgentLoop, **kwargs: Any) -> Any:
        context = kwargs.get("permission_context")
        if isinstance(context, PermissionContext) and context.origin == "schedule":
            contexts.append(context)
        return original(loop, **kwargs)

    monkeypatch.setattr(AgentLoop, "_new_run_gateway", capture_gateway)
    monkeypatch.setattr(AgentLoop, "run_schedule_job", run_schedule)
    monkeypatch.setattr(case.provider, "complete", complete)
    await _request(case, "permission", permission_level="read-only")
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    job = ScheduleJob(
        job_id=str(uuid4()),
        message="scheduled permission evidence",
        title="Permission evidence",
        schedule=JobSchedule.at(datetime.now(UTC).isoformat(timespec="milliseconds")),
        created_at_ms=timestamp,
        updated_at_ms=timestamp,
    )
    await case.workspace.schedule_service.add_user_job(job)
    await asyncio.wait_for(provider_called.wait(), timeout=2)
    await asyncio.wait_for(finished.wait(), timeout=2)
    assert contexts
    assert contexts[0].level == "workspace-write"
    assert contexts[0].configured_schedule_level == "workspace-write"
    assert contexts[0].snapshot is not None
    assert contexts[0].snapshot.level == "workspace-write"
    assert case.service.client_permission(case.first.client_id).current() == "read-only"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "permission", "effort"])
async def test_typed_http_actions_require_auth_csrf_and_matching_client_identity(
    management_case: ManagementCase,
    action: str,
) -> None:
    case = management_case
    token = create_credential(case.service.agent_home)
    app = create_app(case.service)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-MyClaw-CSRF": token,
        "X-MyClaw-Client": case.second.client_id,
        "X-MyClaw-Claim": case.other_claim.credential,
    }
    body = {
        "request_id": "http-runtime",
        "current_session_id": case.other_claim.session_id,
        "claim_version": case.other_claim.version,
        **_action_payload(action),
    }
    original_config = (case.service.agent_home.path / "config.toml").read_bytes()
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        url = server.make_url(
            f"/api/v1/workspaces/{case.workspace.workspace_id}/management/{action}"
        )
        for missing in ("Authorization", "X-MyClaw-CSRF", "X-MyClaw-Client"):
            async with http.post(
                url,
                headers={key: value for key, value in headers.items() if key != missing},
                json=body,
            ) as response:
                assert response.status in {401, 403}
        async with http.post(
            url, headers={**headers, "X-MyClaw-Client": "unknown-client"}, json=body
        ) as response:
            assert response.status in {401, 403}
        async with http.post(
            url, headers={**headers, "X-MyClaw-Claim": "foreign-claim"}, json=body
        ) as response:
            assert response.status == 409
            assert (await response.json())["code"] == "stale_claim"
        assert case.second.permission_control.current() == "workspace-write"
        assert (case.service.agent_home.path / "config.toml").read_bytes() == original_config
        async with http.post(url, headers=headers, json=body) as response:
            assert response.status == 200
            _validator("management_response").validate(await response.json())
