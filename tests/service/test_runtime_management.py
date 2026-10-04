"""Typed Runtime management scope, lifetime, replay, and CLI compatibility."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from omni.agent.loop import AgentRunExecutor
from omni.agent.memory.dream import DreamResult
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.config.config import ConfigLoader
from omni.provider.models import ModelCompleted, ModelStreamEvent
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.service.discovery import create_credential
from omni.service.errors import ServiceError
from omni.service.runtime import AgentService, ClientState, SessionClaim, WorkspaceRecord
from omni.service.transport import create_app
from tests.fixtures import FakeClock
from tests.memory.test_dream import _response
from tests.service.test_protocol_contract import _validator
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home


@dataclass
class ManagementCase:
    service: AgentService
    workspace: WorkspaceRecord
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
    monkeypatch.setattr("omni.service.runtime.create_provider", lambda *_args: provider)
    service = AgentService(
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
    workspace: WorkspaceRecord | None = None,
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
async def test_cli_config_reports_saved_and_startup_versions_and_restart_requirement(
    management_case: ManagementCase,
) -> None:
    case = management_case
    revision = cast(str, case.service.config_view()["revision"])
    saved = await case.service.update_configuration(
        "config-cli", revision, {"memory": {"batch_size": 17}},
    )
    result = await _request(case, "dispatch", command="/config")
    output = cast(str, result["output"])
    assert f"Saved version: {saved['revision']}" in output
    assert f"Startup version: {revision}" in output
    assert "Restart required: yes" in output
    assert "minimal-secret" not in output


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
@pytest.mark.parametrize(
    "action",
    ["status", "permission", "effort", "memory", "dream", "skills/reload"],
)
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
@pytest.mark.parametrize(
    "action",
    ["status", "permission", "effort", "memory", "dream", "skills/reload"],
)
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
    assert case.workspace.resources is not None
    assert case.workspace.resources.router.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_typed_memory_dream_and_skill_projections_return_real_safe_results(
    management_case: ManagementCase,
) -> None:
    case = management_case
    case.workspace.workspace_state.long_term_memory_path.write_text(
        "# Shared memory\n\nA readable entry.\n",
        encoding="utf-8",
    )

    memory = await _request(case, "memory", request_id="memory-result")
    dream = await _request(case, "dream", request_id="dream-result")
    skills = await _request(case, "skills/reload", request_id="skills-result")

    assert memory["memory_content"] == "# Shared memory\n\nA readable entry.\n"
    dream_result = cast(dict[str, object], dream["dream_result"])
    assert dream_result["status"] == "No pending summaries"
    assert dream_result["processed_count"] == 0
    assert dream_result["memory_updated"] is False
    assert dream_result["error"] is None
    assert skills["skill_metadata"] == []


@pytest.mark.asyncio
async def test_workspace_dream_is_single_instance_across_clients_and_reopens_after_completion(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    runtime = case.workspace.resources
    assert runtime is not None
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_run() -> DreamResult:
        started.set()
        await release.wait()
        return DreamResult(
            status="Controlled Dream complete.",
            processed_count=1,
            memory_updated=True,
            cursor=1,
        )

    monkeypatch.setattr(runtime.dream, "_run_once", controlled_run)
    first_task = asyncio.create_task(_request(case, "dream", request_id="dream-first"))
    await asyncio.wait_for(started.wait(), timeout=2)

    overlapping = await _request(
        case,
        "dream",
        client=case.second,
        claim=case.other_claim,
        request_id="dream-overlap",
    )
    overlap_result = cast(dict[str, object], overlapping["dream_result"])
    overlap_error = cast(dict[str, object], overlap_result["error"])
    assert overlap_error["code"] == "memory_task_running"

    release.set()
    completed = await asyncio.wait_for(first_task, timeout=2)
    assert cast(dict[str, object], completed["dream_result"])["memory_updated"] is True

    after = await _request(
        case,
        "dream",
        client=case.second,
        claim=case.other_claim,
        request_id="dream-after",
    )
    after_result = cast(dict[str, object], after["dream_result"])
    assert after_result["error"] is None


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
async def test_effort_is_global_runtime_control_with_shared_cli_and_persistence_contract(
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
    ] == "high"
    original = (case.service.agent_home.path / "config.toml").read_bytes()

    def fail_save(_loader: ConfigLoader, _effort: object) -> None:
        raise OSError("test save failure")

    monkeypatch.setattr(ConfigLoader, "update_reasoning_effort", fail_save)
    failed_save = await _request(case, "effort", effort="max")
    assert failed_save["published_effort"] == "max"
    assert failed_save["output"] == "Chat reasoning effort: max"
    assert (await _status(case))["chat_reasoning_effort"] == "max"
    assert (await cli.dispatch("/effort")).effort_selection == "max"
    assert (await _status(case, workspace=workspace, claim=claim))["chat_reasoning_effort"] == "max"
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
    original = AgentRunExecutor._new_run_gateway
    original_complete = case.provider.complete
    original_run = AgentRunExecutor.run_schedule_job

    async def complete(**kwargs: Any) -> Any:
        provider_called.set()
        return await original_complete(**kwargs)

    async def run_schedule(loop: AgentRunExecutor, *args: Any, **kwargs: Any) -> None:
        await original_run(loop, *args, **kwargs)
        finished.set()

    def capture_gateway(loop: AgentRunExecutor, **kwargs: Any) -> Any:
        context = kwargs.get("permission_context")
        if isinstance(context, PermissionContext) and context.origin == "schedule":
            contexts.append(context)
        return original(loop, **kwargs)

    monkeypatch.setattr(AgentRunExecutor, "_new_run_gateway", capture_gateway)
    monkeypatch.setattr(AgentRunExecutor, "run_schedule_job", run_schedule)
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
@pytest.mark.parametrize(
    "action",
    ["status", "permission", "effort", "memory", "dream", "skills/reload"],
)
async def test_typed_http_actions_require_auth_csrf_and_matching_client_identity(
    management_case: ManagementCase,
    action: str,
) -> None:
    case = management_case
    token = create_credential(case.service.agent_home)
    app = create_app(case.service)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": case.second.client_id,
        "X-Omni-Claim": case.other_claim.credential,
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
        for missing in ("Authorization", "X-Omni-CSRF", "X-Omni-Client"):
            async with http.post(
                url,
                headers={key: value for key, value in headers.items() if key != missing},
                json=body,
            ) as response:
                assert response.status in {401, 403}
        async with http.post(
            url, headers={**headers, "X-Omni-Client": "unknown-client"}, json=body
        ) as response:
            assert response.status in {401, 403}
        async with http.post(
            url, headers={**headers, "X-Omni-Claim": "foreign-claim"}, json=body
        ) as response:
            assert response.status == 409
            assert (await response.json())["code"] == "stale_claim"
        assert case.second.permission_control.current() == "workspace-write"
        assert (case.service.agent_home.path / "config.toml").read_bytes() == original_config
        async with http.post(url, headers=headers, json=body) as response:
            assert response.status == 200
            _validator("management_response").validate(await response.json())


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["memory", "dream", "skills/reload"])
async def test_new_management_http_actions_reject_get_and_head_without_execution(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    case = management_case
    token = create_credential(case.service.agent_home)
    calls: list[str] = []

    def unexpected_dispatcher(*args: object, **kwargs: object) -> None:
        calls.append(action)
        raise AssertionError("A read request executed management work")

    monkeypatch.setattr(case.workspace, "management_dispatcher", unexpected_dispatcher)
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        url = server.make_url(
            f"/api/v1/workspaces/{case.workspace.workspace_id}/management/{action}"
        )
        for method in ("GET", "HEAD"):
            async with http.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Omni-Client": case.second.client_id,
                    "X-Omni-Claim": case.other_claim.credential,
                    "X-Omni-Request": "read-cannot-mutate",
                },
                params={
                    "session_id": case.other_claim.session_id,
                    "claim_version": case.other_claim.version,
                },
            ) as response:
                assert response.status == 405
                assert response.headers["Allow"] == "POST"
        assert calls == []


@pytest.mark.asyncio
async def test_http_reload_preserves_active_run_snapshot_resources_and_next_run_skills(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    instruction = case.service.agent_home.skills_directory / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Original catalog\n---\nOLD_SKILL_BODY\n",
        encoding="utf-8",
    )
    (case.workspace.workspace_path / "fixture.txt").write_text(
        "resource survived", encoding="utf-8"
    )
    requests: list[list[dict[str, Any]]] = []
    started = asyncio.Event()
    release = asyncio.Event()
    original_stream = case.provider.stream
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    def stream(**kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        if str(messages[0].get("content", "")).startswith("Generate a concise title"):
            return original_stream(**kwargs)
        requests.append(deepcopy(list(messages)))
        first = len(requests) == 1

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if first:
                started.set()
                await release.wait()
                yield ModelCompleted(
                    _response(
                        "",
                        tool_calls=(
                            ModelToolCall("read-resource", "read_file", '{"path":"fixture.txt"}'),
                        ),
                    )
                )
            else:
                yield ModelCompleted(_response("snapshot run completed"))

        return emit()

    monkeypatch.setattr(case.provider, "stream", stream)
    monkeypatch.setattr(case.provider, "close", close)
    token = create_credential(case.service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": case.second.client_id,
        "X-Omni-Claim": case.other_claim.credential,
    }
    sink = cast(_CollectingSink, case.second.sink)
    loop = case.other_claim.loop
    runtime = case.workspace.resources
    catalog = case.service.built_in_tool_catalog
    session = loop.session
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        url = server.make_url(
            f"/api/v1/workspaces/{case.workspace.workspace_id}/management/skills/reload"
        )

        async def reload(request_id: str) -> dict[str, Any]:
            async with http.post(
                url,
                headers=headers,
                json={
                    "request_id": request_id,
                    "current_session_id": case.other_claim.session_id,
                    "claim_version": case.other_claim.version,
                },
            ) as response:
                assert response.status == 200
                body = await response.json()
                _validator("management_response").validate(body)
                return cast(dict[str, Any], body["result"])

        assert (await reload("seed-skills"))["skill_metadata"][0]["name"] == "planner"
        await case.workspace.input(
            case.second.client_id,
            case.other_claim.session_id,
            case.other_claim.version,
            "/planner first request",
            "old-snapshot",
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            instruction.write_text(
                "---\nname: reviewer\ndescription: Updated catalog\n---\nNEW_SKILL_BODY\n",
                encoding="utf-8",
            )
            result = await reload("reload-during-run")
            assert result["skill_metadata"][0]["name"] == "reviewer"
            assert loop.has_active_run
            assert case.other_claim.loop is loop
            assert case.workspace.resources is runtime
            assert loop.session is session
            assert case.service.built_in_tool_catalog is catalog
            assert closed == []
        finally:
            release.set()
        await asyncio.wait_for(sink.wait_for("run.completed", "old-snapshot"), timeout=3)
        assert len(requests) == 2
        for request in requests:
            assert '"name":"planner"' in str(request[0]["content"])
            assert '"name":"reviewer"' not in str(request[0]["content"])
            assert "OLD_SKILL_BODY" in json.dumps(request)
            assert "NEW_SKILL_BODY" not in json.dumps(request)
        assert "resource survived" in json.dumps(requests[1])
        await case.workspace.input(
            case.second.client_id,
            case.other_claim.session_id,
            case.other_claim.version,
            "/reviewer next request",
            "new-snapshot",
        )
        await asyncio.wait_for(sink.wait_for("run.completed", "new-snapshot"), timeout=3)
        assert len(requests) == 3
        assert '"name":"reviewer"' in str(requests[2][0]["content"])
        assert '"name":"planner"' not in str(requests[2][0]["content"])
        assert "NEW_SKILL_BODY" in str(requests[2][-1]["content"])
        assert closed == []


@pytest.mark.asyncio
async def test_http_dream_updates_memory_and_replay_does_not_repeat_model_work(
    management_case: ManagementCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = management_case
    runtime = case.workspace.resources
    assert runtime is not None
    manager = runtime.memory_manager
    await manager.append_summary("The user prefers concise reports.", case.clock.now())
    calls: list[bool] = []

    async def complete(**kwargs: Any) -> Any:
        calls.append(True)
        return _response(
            "",
            tool_calls=(
                ModelToolCall(
                    "edit-memory",
                    "edit_file",
                    json.dumps(
                        {
                            "path": str(manager.long_term_path),
                            "old_text": "## User Preference\n",
                            "new_text": "## User Preference\n\nPrefers concise reports.\n",
                        }
                    ),
                ),
            ),
        )

    monkeypatch.setattr(case.provider, "complete", complete)
    token = create_credential(case.service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": case.first.client_id,
        "X-Omni-Claim": case.claim.credential,
    }
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        base = f"/api/v1/workspaces/{case.workspace.workspace_id}/management"
        body = {
            "request_id": "real-dream",
            "current_session_id": case.claim.session_id,
            "claim_version": case.claim.version,
        }
        results: list[dict[str, Any]] = []
        for _ in range(2):
            async with http.post(
                server.make_url(f"{base}/dream"), headers=headers, json=body
            ) as response:
                assert response.status == 200
                value = await response.json()
                _validator("management_response").validate(value)
                results.append(value["result"])
        assert results[0] == results[1]
        assert results[0]["dream_result"]["processed_count"] == 1
        assert results[0]["dream_result"]["memory_updated"] is True
        assert results[0]["dream_result"]["error"] is None
        assert len(calls) == 1
        async with http.post(
            server.make_url(f"{base}/memory"),
            headers=headers,
            json={**body, "request_id": "inspect-updated-memory"},
        ) as response:
            assert response.status == 200
            assert "Prefers concise reports." in (await response.json())["result"]["memory_content"]
