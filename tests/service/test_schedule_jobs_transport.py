"""HTTP behavior tests for the Web Schedule Job boundary."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

import omni.service.runtime as service_runtime
from omni.agent.confirmation import BackgroundConfirmationOwner
from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.tools.core.read_file import ReadFileTool
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.agent.workspace_state import WorkspaceState
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.schedule.store import WorkspaceScheduleStore
from omni.service.client import ServiceClient
from omni.service.discovery import create_credential
from omni.service.runtime import AgentService
from omni.service.transport import create_app
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.scheduling.test_schedule_agent_loop import _response, _ScheduleProvider


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _prepare_agent_home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


def _headers(client: ServiceClient) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {client.token}",
        "X-Omni-CSRF": client.token,
        "X-Omni-Client": client.client_id,
    }


class _SilentSink:
    async def send_event(self, _event: dict[str, object]) -> None:
        return None


@pytest_asyncio.fixture
async def schedule_http(
    tmp_path: Path,
) -> AsyncIterator[tuple[AgentService, TestServer, str, dict[str, str]]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    path = tmp_path / "workspace"
    path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    server = TestServer(create_app(service))
    await service.start()
    client = await service.register_client("web")
    await service.connect_client(client.client_id, _SilentSink())
    workspace = await service.attach_workspace(client.client_id, path)
    token = create_credential(home)
    await server.start_server()
    url = str(server.make_url(f"/api/v1/workspaces/{workspace.workspace_id}/schedule/jobs"))
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": client.client_id,
    }
    try:
        yield service, server, url, headers
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_schedule_jobs_http_crud_validates_all_kinds_and_is_cross_client_idempotent(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    first: ServiceClient | None = None
    second: ServiceClient | None = None
    try:
        first = await ServiceClient.connect_or_start(home, workspace, port=port)
        second = await ServiceClient.connect_or_start(home, workspace, port=port)
        url = f"{first.base_url}/api/v1/workspaces/{first.workspace_id}/schedule/jobs"
        first_headers = _headers(first)
        second_headers = _headers(second)
        at_payload = {
            "request_id": "schedule-at-create",
            "message": "Run the one-time task",
            "title": "One-time task",
            "at_time": "2099-01-02T03:04:05.000+00:00",
        }

        async with aiohttp.ClientSession() as http:

            async def first_create(headers: dict[str, str]) -> dict[str, object]:
                async with http.post(url, headers=headers, json=at_payload) as response:
                    assert response.status == 200
                    return cast(dict[str, object], await response.json())

            created, simultaneous = await asyncio.gather(
                first_create(first_headers), first_create(second_headers)
            )
            assert simultaneous == created
            assert created["request_id"] == at_payload["request_id"]

            async def retry_create() -> object:
                async with http.post(url, headers=second_headers, json=at_payload) as response:
                    assert response.status == 200
                    return await response.json()

            assert await retry_create() == created
            assert tuple(await asyncio.gather(retry_create(), retry_create())) == (created, created)
            at_job = cast(dict[str, object], created["job"])
            at_job_id = cast(str, at_job["job_id"])
            assert cast(dict[str, object], at_job["schedule"])["kind"] == "at"
            assert cast(dict[str, object], at_job["state"]) == {
                "last_finished_at_ms": None,
                "last_status": None,
                "last_error": None,
            }

            for payload, field in (
                (
                    {
                        "request_id": "schedule-every-create",
                        "message": "Run repeatedly",
                        "every_seconds": 3600,
                    },
                    "every_seconds",
                ),
                (
                    {
                        "request_id": "schedule-cron-create",
                        "message": "Run on the hour",
                        "cron_expr": "0 * * * *",
                        "timezone": "UTC",
                    },
                    "cron_expr",
                ),
            ):
                async with http.post(url, headers=first_headers, json=payload) as response:
                    assert response.status == 200
                    body = await response.json()
                schedule = cast(dict[str, object], cast(dict[str, object], body["job"])["schedule"])
                assert schedule
                assert field in schedule

            async with http.get(url, headers=first_headers) as response:
                assert response.status == 200
                listing = await response.json()
            assert listing["workspace_id"] == first.workspace_id
            assert listing["status"] == {
                "admitted": True,
                "status": "available",
                "active_job_count": 0,
            }
            assert len(listing["jobs"]) == 3
            persisted = await WorkspaceScheduleStore(WorkspaceState(workspace)).public_snapshot()
            assert {job.job_id for job in persisted} == {job["job_id"] for job in listing["jobs"]}

            detail_url = f"{url}/{at_job_id}"
            async with http.get(detail_url, headers=second_headers) as response:
                assert response.status == 200
                detail = await response.json()
            assert detail["job"]["job_id"] == at_job_id
            assert detail["status"] == listing["status"]

            async with http.post(
                url,
                headers=first_headers,
                json={**at_payload, "message": "A different accepted request"},
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "request_reused"

            invalid_payloads = (
                ({"request_id": "invalid-message", "at_time": at_payload["at_time"]}, "message"),
                (
                    {
                        "request_id": "invalid-every",
                        "message": "Invalid every",
                        "every_seconds": 0,
                    },
                    "every_seconds",
                ),
                (
                    {
                        "request_id": "invalid-cron",
                        "message": "Invalid cron",
                        "cron_expr": "not cron",
                        "timezone": "UTC",
                    },
                    "cron_expr",
                ),
                (
                    {
                        "request_id": "invalid-timezone",
                        "message": "Invalid timezone",
                        "cron_expr": "0 * * * *",
                        "timezone": "Not/AZone",
                    },
                    "timezone",
                ),
            )
            for payload, field in invalid_payloads:
                async with http.post(url, headers=first_headers, json=payload) as response:
                    assert response.status == 400
                    body = await response.json()
                assert field in body["field_errors"]

            async with http.delete(
                detail_url,
                headers=second_headers,
                json={"request_id": "schedule-at-delete"},
            ) as response:
                assert response.status == 200
                deleted = await response.json()
            assert deleted["request_id"] == "schedule-at-delete"
            assert deleted["deleted"] is True
            assert deleted["job"]["job_id"] == at_job_id

            other_job_id = next(
                job["job_id"] for job in listing["jobs"] if job["job_id"] != at_job_id
            )
            async with http.delete(
                f"{url}/{other_job_id}",
                headers=first_headers,
                json={"request_id": "schedule-at-delete"},
            ) as response:
                assert response.status == 409
                assert (await response.json())["code"] == "request_reused"

            async with http.get(detail_url, headers=first_headers) as response:
                assert response.status == 404
                assert (await response.json())["code"] == "not_found"

            remaining_job_ids = [
                cast(str, job["job_id"]) for job in listing["jobs"] if job["job_id"] != at_job_id
            ]
            for index, remaining_job_id in enumerate(remaining_job_ids):
                async with http.delete(
                    f"{url}/{remaining_job_id}",
                    headers=first_headers,
                    json={"request_id": f"schedule-remaining-delete-{index}"},
                ) as response:
                    assert response.status == 200
                    remaining_deleted = await response.json()
                assert remaining_deleted["deleted"] is True

            async with http.get(url, headers=first_headers) as response:
                assert response.status == 200
                assert (await response.json())["jobs"] == []

            async with http.get(
                f"{first.base_url}/api/v1/workspaces/not-a-workspace/schedule/jobs",
                headers=first_headers,
            ) as response:
                assert response.status == 404
                assert (await response.json())["code"] == "not_found"
    finally:
        if second is not None:
            await second.close()
        if first is not None:
            await first.close()
        await ServiceClient.stop_existing(home, port=port)


@pytest.mark.asyncio
async def test_schedule_job_history_groups_existing_schedule_session_and_paginates(
    schedule_http: tuple[AgentService, TestServer, str, dict[str, str]],
) -> None:
    service, server, jobs_url, headers = schedule_http
    workspace = next(iter(service.workspaces.values()))
    async with aiohttp.ClientSession() as http:
        async with http.post(
            jobs_url,
            headers=headers,
            json={
                "request_id": "history-create",
                "message": "Run the scheduled history task",
                "title": "History task",
                "at_time": "2099-01-01T00:00:00Z",
            },
        ) as response:
            assert response.status == 200
            created = cast(dict[str, object], await response.json())

        job = cast(dict[str, object], created["job"])
        job_id = cast(str, job["job_id"])
        session_id = cast(str, job["session_id"])
        first_at = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
        second_at = first_at + timedelta(minutes=5)
        first_session = Session.create_schedule(
            workspace.workspace_state,
            job_id,
            now=lambda: first_at,
            title="History task",
        )
        first_session.commit_agent_run(
            [
                {"role": "user", "content": "first run"},
                {
                    "role": "assistant",
                    "content": "first result",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )
        first_session.close()
        second_session = Session.load(
            workspace.workspace_state,
            session_id,
            partition=SessionStoragePartition.SCHEDULE,
            now=lambda: second_at,
        )
        second_session.commit_agent_run(
            [
                {"role": "user", "content": "second run"},
                {
                    "role": "assistant",
                    "content": "second result",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )
        second_session.close()

        history_url = f"{jobs_url}/{job_id}/history"
        for limit in (None, "", "0", "-1", "1.5", "invalid", "1", "101"):
            async with http.get(
                history_url,
                params={} if limit is None else {"limit": limit},
                headers=headers,
            ) as response:
                body = await response.json()
                if limit in {None, "1"}:
                    assert response.status == 200
                    assert len(body["groups"]) == (2 if limit is None else 1)
                    assert bool(body["next_cursor"]) == (limit == "1")
                else:
                    assert response.status == 422
                    assert body["code"] == "validation_error"
                    assert body["message"] == (
                        "limit must be between 1 and 100." if limit == "101" else "limit is invalid."
                    )
                    assert body["field_errors"] == {}
                    assert body["retryable"] is False
        async with http.get(f"{history_url}?limit=1", headers=headers) as response:
            assert response.status == 200
            first_page = cast(dict[str, object], await response.json())
        assert first_page["workspace_id"] == workspace.workspace_id
        assert first_page["job_id"] == job_id
        assert first_page["session_id"] == session_id
        first_groups = cast(list[dict[str, object]], first_page["groups"])
        assert len(first_groups) == 1
        assert first_groups[0]["result_state"] == "success"
        assert (
            cast(list[dict[str, object]], first_groups[0]["messages"])[0]["content"] == "first run"
        )
        assert "occurrence_id" not in first_groups[0]
        assert "session_id" not in first_groups[0]
        cursor = cast(str, first_page["next_cursor"])

        async with http.get(
            history_url,
            params={"limit": "1", "cursor": cursor},
            headers=headers,
        ) as response:
            assert response.status == 200
            second_page = cast(dict[str, object], await response.json())
        second_groups = cast(list[dict[str, object]], second_page["groups"])
        assert len(second_groups) == 1
        assert second_groups[0]["result_state"] == "success"
        assert (
            cast(list[dict[str, object]], second_groups[0]["messages"])[0]["content"]
            == "second run"
        )
        assert second_page["next_cursor"] is None

        async with http.get(
            f"{server.make_url(f'/api/v1/workspaces/{workspace.workspace_id}/sessions')}",
            headers=headers,
        ) as response:
            assert response.status == 200
            conversation_sessions = cast(dict[str, object], await response.json())
        assert all(
            summary["id"] != session_id
            for summary in cast(list[dict[str, object]], conversation_sessions["sessions"])
        )

        async with http.get(
            history_url,
            params={"cursor": "not-a-valid-history-cursor"},
            headers=headers,
        ) as response:
            assert response.status == 422
            assert (await response.json())["code"] == "validation_error"

        async with http.get(
            f"{jobs_url}/00000000-0000-4000-8000-000000000000/history",
            headers=headers,
        ) as response:
            assert response.status == 404
            assert (await response.json())["code"] == "not_found"


@pytest.mark.asyncio
async def test_schedule_job_http_delete_cancels_a_running_job_and_keeps_deleted_state(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    other_workspace_path = tmp_path / "other-workspace"
    other_workspace_path.mkdir()
    service = AgentService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=30)
    server = TestServer(create_app(service))
    release = asyncio.Event()
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def blocked_execution(_occurrence: object) -> None:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            canceled.set()
            raise

    await service.start()
    client = await service.register_client("web")
    token = create_credential(home)
    try:
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        other_client = await service.register_client("web")
        other_workspace = await service.attach_workspace(
            other_client.client_id, other_workspace_path
        )
        workspace.schedule_service._execute_user_occurrence = blocked_execution
        await server.start_server()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Omni-CSRF": token,
            "X-Omni-Client": client.client_id,
        }
        jobs_url = str(
            server.make_url(f"/api/v1/workspaces/{workspace.workspace_id}/schedule/jobs")
        )
        at_time = workspace.schedule_service.current_time().astimezone(UTC)
        payload = {
            "request_id": "running-job-create",
            "message": "Block until the HTTP delete cancels this run",
            "title": "Cancelable running job",
            "at_time": at_time.isoformat(timespec="milliseconds"),
        }

        async with aiohttp.ClientSession() as http:
            async with http.post(jobs_url, headers=headers, json=payload) as response:
                assert response.status == 200
                created = await response.json()
            job = cast(dict[str, object], created["job"])
            job_id = cast(str, job["job_id"])
            await asyncio.wait_for(started.wait(), timeout=5)

            async with http.get(f"{jobs_url}/{job_id}", headers=headers) as response:
                assert response.status == 200
                active = await response.json()
            assert active["job"]["active"] is True
            assert active["status"]["active_job_count"] == 1

            async with http.delete(
                f"{jobs_url}/{job_id}",
                headers=headers,
                json={"request_id": "running-job-delete"},
            ) as response:
                assert response.status == 200
                deleted = await response.json()
            assert deleted["deleted"] is True
            assert deleted["canceled"] is True
            assert deleted["job"]["status"] == "deleted"
            assert deleted["job"]["active"] is False
            assert deleted["status"]["active_job_count"] == 0
            assert canceled.is_set()

            async with http.get(f"{jobs_url}/{job_id}", headers=headers) as response:
                assert response.status == 404
                assert (await response.json())["code"] == "not_found"

            other_jobs_url = str(
                server.make_url(f"/api/v1/workspaces/{other_workspace.workspace_id}/schedule/jobs")
            )
            async with http.get(other_jobs_url, headers=headers) as response:
                assert response.status == 403
                assert (await response.json())["code"] == "forbidden"
        assert workspace.schedule_service.status_snapshot().active_job_count == 0
        assert await workspace.schedule_service.public_snapshot() == ()
    finally:
        release.set()
        await server.close()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "errors"),
    [
        ({"message": False, "at_time": "2099-01-01T00:00:00Z"}, {"message"}),
        ({"title": None, "at_time": "2099-01-01T00:00:00Z"}, {"title"}),
        ({"at_time": "2099-01-01T00:00:00"}, {"at_time"}),
        ({"at_time": 123}, {"at_time"}),
        ({"at_time": "2099-02-30T00:00:00Z"}, {"at_time"}),
        ({"every_seconds": True}, {"every_seconds"}),
        ({"every_seconds": 1.5}, {"every_seconds"}),
        ({"every_seconds": "60"}, {"every_seconds"}),
        ({"every_seconds": -1}, {"every_seconds"}),
        ({"every_seconds": 10**100}, {"every_seconds"}),
        ({"every_seconds": 60, "timezone": "UTC"}, {"timezone"}),
        ({"cron_expr": "not cron", "timezone": "Asia/Shanghai"}, {"cron_expr"}),
        ({"cron_expr": "not cron", "timezone": "Not/AZone"}, {"cron_expr", "timezone"}),
        ({"cron_expr": "* * * * *", "timezone": False}, {"timezone"}),
        ({"cron_expr": "0 0 31 2 *", "timezone": "UTC"}, {"cron_expr"}),
        ({"kind": False, "every_seconds": 60}, {"kind"}),
        ({"kind": "at", "every_seconds": 60}, {"schedule"}),
        ({"schedule": "invalid", "every_seconds": 60}, {"schedule"}),
    ],
)
async def test_schedule_field_errors_leave_disk_unchanged(
    schedule_http: tuple[AgentService, TestServer, str, dict[str, str]],
    fields: dict[str, object],
    errors: set[str],
) -> None:
    service, _server, url, headers = schedule_http
    workspace = next(iter(service.workspaces.values()))
    before = workspace.workspace_state.schedule_path.read_bytes()
    async with aiohttp.ClientSession() as http:
        async with http.post(
            url, headers=headers, json={"request_id": "invalid", "message": "task", **fields}
        ) as response:
            assert response.status == 400
            assert set((await response.json())["field_errors"]) == errors
    assert workspace.workspace_state.schedule_path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidated", ["client", "project"])
async def test_schedule_queued_retry_rechecks_current_identity(
    schedule_http: tuple[AgentService, TestServer, str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    invalidated: str,
) -> None:
    service, _server, url, headers = schedule_http
    workspace = next(iter(service.workspaces.values()))
    payload = {"request_id": "queued", "message": "task", "at_time": "2099-01-01T00:00:00Z"}
    async with aiohttp.ClientSession() as http:
        async with http.post(url, headers=headers, json=payload) as response:
            assert response.status == 200
        checked = asyncio.Event()
        original = service._schedule_workspace

        def record_check(
            client_id: str, workspace_id: str
        ) -> service_runtime.WorkspaceRecord:
            result = original(client_id, workspace_id)
            checked.set()
            return result

        monkeypatch.setattr(service, "_schedule_workspace", record_check)
        await service._schedule_mutation_lock.acquire()
        request = asyncio.create_task(http.post(url, headers=headers, json=payload))
        try:
            await asyncio.wait_for(checked.wait(), 5)
            if invalidated == "client":
                service.client(headers["X-Omni-Client"]).expired = True
            else:
                record = service.projects.register(workspace.workspace_path)
                service.projects.begin_removal(record.project_id, None)
        finally:
            service._schedule_mutation_lock.release()
        async with await request as response:
            assert response.status == 409
            assert (await response.json())["code"] == (
                "stale_client" if invalidated == "client" else "admission_closed"
            )
    assert len(await WorkspaceScheduleStore(workspace.workspace_state).public_snapshot()) == 1


@pytest.mark.asyncio
async def test_schedule_scope_reuse_and_transport_guards(
    schedule_http: tuple[AgentService, TestServer, str, dict[str, str]],
    tmp_path: Path,
) -> None:
    service, server, url, headers = schedule_http
    payload = {"request_id": "scope", "message": "task", "at_time": "2099-01-01T08:00:00+08:00"}
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = await service.attach_workspace(headers["X-Omni-Client"], other_path)
    other_url = str(server.make_url(f"/api/v1/workspaces/{other.workspace_id}/schedule/jobs"))
    async with aiohttp.ClientSession() as http:
        async with http.post(url, headers=headers, json=payload) as response:
            assert response.status == 200
            assert (await response.json())["job"]["schedule"][
                "at_time"
            ] == "2099-01-01T08:00:00.000+08:00"
        async with http.post(other_url, headers=headers, json=payload) as response:
            assert response.status == 409
            assert (await response.json())["code"] == "request_reused"
        for denied_headers, expected in [
            ({}, 401),
            ({k: v for k, v in headers.items() if k != "X-Omni-CSRF"}, 403),
            ({**headers, "Origin": "https://example.com"}, 403),
        ]:
            async with http.post(
                other_url, headers=denied_headers, json={**payload, "request_id": "denied"}
            ) as response:
                assert response.status == expected
    assert await WorkspaceScheduleStore(other.workspace_state).public_snapshot() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tool", "confirmation", "cleanup_failure"])
async def test_schedule_http_delete_drains_real_tool_and_preserves_session_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    started = asyncio.Event()
    canceled = asyncio.Event()
    external = tmp_path / "external.txt"
    external.write_text("external data", encoding="utf-8")
    provider = _ScheduleProvider(
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="blocked-read",
                    name="read_file",
                    arguments=json.dumps(
                        {"path": str(external) if mode == "confirmation" else "task.txt"}
                    ),
                ),
            ),
        )
    )
    monkeypatch.setattr(service_runtime, "create_provider", lambda _config: provider)

    async def blocked_read(_tool: ReadFileTool, **_kwargs: object) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled.set()
            raise
        return "unreachable"

    monkeypatch.setattr(ReadFileTool, "execute", blocked_read)
    home = _prepare_agent_home(tmp_path / "home")
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "task.txt").write_text("user data", encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    server = TestServer(create_app(service))
    await service.start()
    client = await service.register_client("web")
    resolved = asyncio.Event()

    class EventSink:
        async def send_event(self, event: dict[str, object]) -> None:
            if event.get("type") == "confirmation.requested":
                started.set()
            if event.get("type") == "confirmation.resolved":
                resolved.set()

    await service.connect_client(client.client_id, EventSink())
    workspace = await service.attach_workspace(client.client_id, path)
    cleanup_calls = 0
    original_cancel = workspace.schedule_service._cancel_confirmation_owner

    async def flaky_cleanup(owner: BackgroundConfirmationOwner) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise OSError("injected cleanup failure")
        assert original_cancel is not None
        await original_cancel(owner)

    if mode == "cleanup_failure":
        workspace.schedule_service._cancel_confirmation_owner = flaky_cleanup
    token = create_credential(home)
    await server.start_server()
    url = str(server.make_url(f"/api/v1/workspaces/{workspace.workspace_id}/schedule/jobs"))
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Omni-CSRF": token,
        "X-Omni-Client": client.client_id,
    }
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                url,
                headers=headers,
                json={
                    "request_id": "real-create",
                    "message": "Read task.txt",
                    "at_time": workspace.schedule_service.current_time().isoformat(),
                },
            ) as response:
                assert response.status == 200
                job = (await response.json())["job"]
            await asyncio.wait_for(started.wait(), 5)
            if mode == "cleanup_failure":
                async with http.delete(
                    f"{url}/{job['job_id']}", headers=headers, json={"request_id": "real-delete"}
                ) as response:
                    assert response.status == 409
                    assert (await response.json())["code"] == "schedule_update_failed"
            async with http.delete(
                f"{url}/{job['job_id']}", headers=headers, json={"request_id": "real-delete"}
            ) as response:
                assert response.status == 200
                deleted = await response.json()
            if mode == "confirmation":
                await asyncio.wait_for(resolved.wait(), 5)
                assert not canceled.is_set()
            else:
                assert canceled.is_set()
            if mode == "cleanup_failure":
                assert cleanup_calls == 2
            assert deleted["canceled"] is True
            history = Session.load(
                workspace.workspace_state,
                job["session_id"],
                partition=SessionStoragePartition.SCHEDULE,
            )
            assert any(
                message["role"] == "tool" and message.get("tool_call_id") == "blocked-read"
                for message in history.messages
            )
            assert history.messages[-1]["status"] == "error"
            assert "cancelled" in history.messages[-1]["content"]
            assert await WorkspaceScheduleStore(workspace.workspace_state).public_snapshot() == ()
            async with http.delete(
                f"{url}/{job['job_id']}", headers=headers, json={"request_id": "real-delete"}
            ) as response:
                assert response.status == 200
                assert await response.json() == deleted
            assert (path / "task.txt").read_text(encoding="utf-8") == "user data"
    finally:
        await server.close()
        await service.stop()
