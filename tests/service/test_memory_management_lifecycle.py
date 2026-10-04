"""HTTP Memory management concurrency, shutdown, and safe failure boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import BaseTestServer, TestServer

from omni.config.config import ConfigLoader
from omni.provider.models import ModelResponse
from omni.service.discovery import create_credential, read_credential
from omni.service.runtime import AgentService, ClientState, SessionClaim, WorkspaceRecord
from omni.service.transport import create_app
from tests.memory.test_dream import _response
from tests.service.test_protocol_contract import _validator
from tests.service.test_runtime_management import ManagementCase
from tests.service.test_runtime_management import management_case as _management_case
from tests.service.test_service_concurrency import _CollectingSink

management_case = _management_case


async def _post(
    http: aiohttp.ClientSession,
    server: BaseTestServer,
    case: ManagementCase,
    action: str,
    request_id: str,
    *,
    client: ClientState | None = None,
    claim: SessionClaim | None = None,
    workspace: WorkspaceRecord | None = None,
) -> dict[str, Any]:
    selected = claim or case.claim
    owner = client or case.first
    scoped = workspace or case.workspace
    try:
        token = read_credential(case.service.agent_home)
    except FileNotFoundError:
        token = create_credential(case.service.agent_home)
    async with http.post(
        server.make_url(f"/api/v1/workspaces/{scoped.workspace_id}/management/{action}"),
        headers={
            "Authorization": f"Bearer {token}",
            "X-Omni-CSRF": token,
            "X-Omni-Client": owner.client_id,
            "X-Omni-Claim": selected.credential,
        },
        json={
            "request_id": request_id,
            "current_session_id": selected.session_id,
            "claim_version": selected.version,
        },
    ) as response:
        assert response.status == 200
        value = await response.json()
        _validator("management_response").validate(value)
        return dict(value["result"])


@pytest.mark.asyncio
async def test_http_dream_overlap_and_inflight_replay_make_one_model_request(
    management_case: ManagementCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = management_case
    runtime = case.workspace.resources
    assert runtime is not None
    await runtime.memory_manager.append_summary("Pending shared update", case.clock.now())
    started, release, replay_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0
    accepted = 0
    original = case.service.handle_management

    async def observed(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal accepted
        if args[4]["request_id"] == "inflight-dream":
            accepted += 1
            if accepted == 2:
                replay_entered.set()
        return await original(*args, **kwargs)

    async def complete(**_kwargs: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return _response("No memory edits needed")

    monkeypatch.setattr(case.service, "handle_management", observed)
    monkeypatch.setattr(case.provider, "complete", complete)
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        first = asyncio.create_task(_post(http, server, case, "dream", "inflight-dream"))
        duplicate: asyncio.Task[dict[str, Any]] | None = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            duplicate = asyncio.create_task(_post(http, server, case, "dream", "inflight-dream"))
            await asyncio.wait_for(replay_entered.wait(), 2)
            overlap = await _post(
                http,
                server,
                case,
                "dream",
                "other-client-dream",
                client=case.second,
                claim=case.other_claim,
            )
            assert overlap["dream_result"]["error"]["code"] == "memory_task_running"
            assert calls == 1
            assert not first.done() and not duplicate.done()
            release.set()
            completed, replay = await asyncio.wait_for(asyncio.gather(first, duplicate), 2)
            assert completed == replay
            assert completed["dream_result"]["processed_count"] == 1
            assert completed["dream_result"]["error"] is None
            assert calls == 1
            await runtime.memory_manager.append_summary("Next shared update", case.clock.now())
            again = await _post(http, server, case, "dream", "next-dream")
            assert again["dream_result"]["processed_count"] == 1
            assert calls == 2
        finally:
            release.set()
            await asyncio.gather(
                first, *([] if duplicate is None else [duplicate]), return_exceptions=True
            )


@pytest.mark.asyncio
async def test_http_dreams_in_different_workspaces_reach_model_concurrently(
    management_case: ManagementCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = management_case
    other_path = tmp_path / "independent-workspace"
    other_path.mkdir()
    other = await case.service.attach_workspace(case.second.client_id, other_path)
    draft = await other.create_draft(case.second.client_id)
    await case.service.claim(case.second.client_id, other.workspace_id, draft)
    other_claim = other._claims[draft]
    assert case.workspace.resources is not None and other.resources is not None
    await case.workspace.resources.memory_manager.append_summary("FIRST-WORKSPACE", case.clock.now())
    await other.resources.memory_manager.append_summary("SECOND-WORKSPACE", case.clock.now())
    started = {key: asyncio.Event() for key in ("FIRST-WORKSPACE", "SECOND-WORKSPACE")}
    release = asyncio.Event()
    calls: list[str] = []

    async def complete(**kwargs: Any) -> ModelResponse:
        key = next(key for key in started if key in str(kwargs["messages"]))
        calls.append(key)
        started[key].set()
        await release.wait()
        return _response("No edits")

    monkeypatch.setattr(case.provider, "complete", complete)
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        requests = [
            asyncio.create_task(_post(http, server, case, "dream", "first-workspace")),
            asyncio.create_task(
                _post(
                    http,
                    server,
                    case,
                    "dream",
                    "second-workspace",
                    client=case.second,
                    claim=other_claim,
                    workspace=other,
                )
            ),
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 2)
            assert len(calls) == 2 and all(not request.done() for request in requests)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*requests), 2)
            assert all(result["dream_result"]["processed_count"] == 1 for result in results)
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_service", [False, True])
async def test_http_dream_shutdown_waits_for_cancel_cleanup_and_releases_workspace_owner(
    management_case: ManagementCase, monkeypatch: pytest.MonkeyPatch, stop_service: bool
) -> None:
    case = management_case
    runtime = case.workspace.resources
    assert runtime is not None
    await runtime.memory_manager.append_summary("Update interrupted by shutdown", case.clock.now())
    started, cancelled, cleanup_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def complete(**_kwargs: Any) -> ModelResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise
        raise AssertionError("Controlled model must be cancelled")

    monkeypatch.setattr(case.provider, "complete", complete)
    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        running = asyncio.create_task(_post(http, server, case, "dream", "shutdown-dream"))
        closing: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(started.wait(), 2)
            closing = asyncio.create_task(
                case.service.stop() if stop_service else case.workspace.close()
            )
            await asyncio.wait_for(cancelled.wait(), 2)
            assert not closing.done()
            assert runtime.dream._task is not None
            cleanup_release.set()
            await asyncio.wait_for(closing, 2)
            outcome = (await asyncio.gather(running, return_exceptions=True))[0]
            assert isinstance(outcome, aiohttp.ServerDisconnectedError)
            assert runtime.dream._task is None
            assert runtime.workspace_id not in case.service.workspace_resources.resources
        finally:
            cleanup_release.set()
            await asyncio.gather(
                running, *([] if closing is None else [closing]), return_exceptions=True
            )

    async def successful(**_kwargs: Any) -> ModelResponse:
        return _response("New runtime completed")

    monkeypatch.setattr(case.provider, "complete", successful)
    replacement = AgentService(
        case.service.agent_home, ConfigLoader(case.service.agent_home).load_for_startup()
    )
    await replacement.start()
    try:
        client = await replacement.register_client("cli")
        await replacement.connect_client(client.client_id, _CollectingSink())
        workspace = await replacement.attach_workspace(
            client.client_id, case.workspace.workspace_path
        )
        assert workspace.resources is not None and workspace.resources is not runtime
        session = await workspace.create_draft(client.client_id)
        await replacement.claim(client.client_id, workspace.workspace_id, session)
        await workspace.resources.memory_manager.append_summary(
            "Fresh update after restart", case.clock.now()
        )
        async with TestServer(create_app(replacement)) as server, aiohttp.ClientSession() as http:
            result = await _post(
                http,
                server,
                case,
                "dream",
                "restart-dream",
                client=client,
                workspace=workspace,
                claim=workspace._claims[session],
            )
            assert result["dream_result"]["processed_count"] == 1
            assert result["dream_result"]["error"] is None
    finally:
        await replacement.stop()


@pytest.mark.asyncio
async def test_http_memory_failure_and_partial_skill_discovery_do_not_expose_exception_secrets(
    management_case: ManagementCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = management_case
    runtime = case.workspace.resources
    assert runtime is not None
    secret = "PRIVATE-provider-token-keep-hidden"
    root = case.service.agent_home.path / "skills"
    for name, document in (
        ("valid", "---\nname: valid\ndescription: Safe published description\n---\nInstructions"),
        ("invalid", f"---\nname: INVALID\ndescription: {secret}\n---\nPrivate instructions"),
    ):
        path = root / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(document, encoding="utf-8")

    async def read_failure() -> str:
        raise OSError(secret)

    def reload_failure() -> object:
        raise RuntimeError(secret)

    async with TestServer(create_app(case.service)) as server, aiohttp.ClientSession() as http:
        partial = await _post(http, server, case, "skills/reload", "partial-skills")
        assert [item["name"] for item in partial["skill_metadata"]] == ["valid"]
        assert secret not in json.dumps(partial)
        monkeypatch.setattr(runtime.memory_manager, "read_long_term", read_failure)
        memory = await _post(http, server, case, "memory", "failed-memory")
        assert memory["management_error"]["code"] == "persistence_error"
        assert memory["memory_content"] is None
        monkeypatch.setattr(case.claim.loop, "reload_skill", reload_failure)
        skills = await _post(http, server, case, "skills/reload", "failed-skills")
        assert skills["management_error"]["code"] == "skill_reload_failed"
        for result in (memory, skills):
            assert secret not in json.dumps(result)
