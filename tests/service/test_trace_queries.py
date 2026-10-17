from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import BaseTestServer, TestServer

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.service.discovery import create_credential
from aide.service.runtime import AgentService
from aide.service.transport import create_app
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.trace import wait_for_trace_writer

TraceHttp = tuple[AgentService, BaseTestServer, str, str]


@pytest_asyncio.fixture
async def trace_http(tmp_path: Path) -> AsyncIterator[TraceHttp]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    create_credential(home)
    client = await service.register_client("web")
    assert client.web_control_credential is not None
    try:
        async with TestServer(create_app(service), host="127.0.0.1") as server:
            yield service, server, client.client_id, client.web_control_credential
    finally:
        await service.stop()


def _auth_headers(service: AgentService, client_id: str, control: str) -> dict[str, str]:
    token = create_credential(service.agent_home)
    return {
        "Authorization": f"Bearer {token}",
        "X-Aide-Client": client_id,
        "X-Aide-Control": control,
    }


async def _seed_model_trace(
    service: AgentService,
    workspace: Path,
    *,
    owner_session_id: str = "fixture-session",
    finished_at: datetime | None = None,
) -> tuple[str, str]:
    finished_at = finished_at or datetime.now(UTC).replace(microsecond=0)
    started_at = finished_at - timedelta(milliseconds=120)
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id=owner_session_id,
        run_id="fixture-run",
    )
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    context = recorder.new_root(source)
    handle = recorder.begin_call(
        context,
        TraceCallMetadata(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=started_at,
            model="fixture-model",
            reasoning_effort="mid",
        ),
    )
    attempt = recorder.begin_attempt(
        handle,
        TraceAttemptMetadata(
            started_at=started_at,
            model="fixture-model",
            reasoning_effort="mid",
        ),
    )
    recorder.set_request(attempt, {"messages": [{"role": "user", "content": "fixture"}]})
    recorder.finish_call(
        handle,
        TraceOutcome(
            status=TraceStatus.SUCCESS,
            finished_at=finished_at,
            duration_ms=120,
            final_attempt_duration_ms=120,
        ),
        result={"text": "fixture result", "reasoning": "", "tool_calls": []},
        usage=TraceTokenUsage(
            state=TraceUsageState.REPORTED,
            input_tokens=9,
            output_tokens=4,
            total_tokens=13,
            cached_input_tokens=2,
        ),
        retry_count=2,
    )
    await wait_for_trace_writer(recorder)
    return handle.record_id, source.workspace_key


async def _seed_failed_model_trace(service: AgentService, workspace: Path) -> None:
    finished_at = datetime.now(UTC).replace(microsecond=0)
    started_at = finished_at - timedelta(milliseconds=60)
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="fixture-session",
        run_id="cancelled-run",
    )
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    context = recorder.new_root(source)
    handle = recorder.begin_call(
        context,
        TraceCallMetadata(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=started_at,
            model="fixture-model",
        ),
    )
    recorder.begin_attempt(
        handle, TraceAttemptMetadata(started_at=started_at, model="fixture-model")
    )
    recorder.finish_call(
        handle,
        TraceOutcome(
            status=TraceStatus.FAILED,
            finished_at=finished_at,
            duration_ms=60,
            error_category=TraceErrorCategory.CANCELLED,
            error_code="cancelled",
            error_message="User cancelled the request.",
        ),
        result={"error": "User cancelled the request."},
        usage=TraceTokenUsage(state=TraceUsageState.UNKNOWN),
        retry_count=0,
    )
    await wait_for_trace_writer(recorder)


async def _seed_tool_trace(service: AgentService, workspace: Path) -> None:
    finished_at = datetime.now(UTC).replace(microsecond=0)
    started_at = finished_at - timedelta(milliseconds=30)
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="fixture-session",
        run_id="fixture-run",
    )
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    handle = recorder.begin_call(
        recorder.new_root(source),
        TraceCallMetadata(
            kind=TraceKind.TOOL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=started_at,
            tool_name="read_file",
        ),
    )
    recorder.set_request(handle, {"path": "README.md"})
    recorder.finish_call(
        handle,
        TraceOutcome(
            status=TraceStatus.SUCCESS,
            finished_at=finished_at,
            duration_ms=30,
        ),
        result={"content": "fixture"},
        usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
        retry_count=0,
    )
    await wait_for_trace_writer(recorder)


async def _seed_unpublished_trace(
    service: AgentService,
    workspace: Path,
    owner_session_id: str,
) -> str:
    started_at = datetime.now(UTC).replace(microsecond=0)
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id=owner_session_id,
        run_id="unpublished-run",
    )
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    handle = recorder.begin_call(
        recorder.new_root(source),
        TraceCallMetadata(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=started_at,
            model="fixture-model",
        ),
    )
    recorder.set_request(handle, {"messages": [{"role": "user", "content": "pending"}]})
    await wait_for_trace_writer(recorder)
    return handle.record_id


@pytest.mark.asyncio
async def test_trace_stats_api_returns_authenticated_seed_aggregates(
    trace_http: TraceHttp,
    tmp_path: Path,
) -> None:
    service, server, client_id, control = trace_http
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _record_id, workspace_key = await _seed_model_trace(service, workspace)
    headers = _auth_headers(service, client_id, control)

    async with aiohttp.ClientSession() as http:
        status_before_response = await http.get(
            server.make_url("/api/v1/service"),
            headers=headers,
        )
        status_before = await status_before_response.json()
        response = await http.get(
            server.make_url(f"/api/v1/traces/stats?workspace_key={workspace_key}"),
            headers=headers,
        )
        body = await response.json()
        status_after_response = await http.get(
            server.make_url("/api/v1/service"),
            headers=headers,
        )
        status_after = await status_after_response.json()

    assert response.status == 200
    assert body["workspace_key"] == workspace_key
    assert body["totals"] == {
        "model_calls": 1,
        "tool_calls": 0,
        "success_count": 1,
        "failed_count": 0,
        "failure_rate": 0.0,
        "retry_count": 2,
        "retried_calls": 1,
        "unknown_usage_count": 0,
        "average_duration_ms": 120.0,
    }
    assert body["tokens"] == {
        "input_tokens": 9,
        "output_tokens": 4,
        "total_tokens": 13,
        "cached_input_tokens": 2,
        "cache_creation_input_tokens": 0,
    }
    assert body["by_model"] == [
        {
            "key": "fixture-model",
            "calls": 1,
            "tokens": body["tokens"],
        }
    ]
    assert status_before_response.status == status_after_response.status == 200
    assert status_after["active_workspace_count"] == status_before["active_workspace_count"]


@pytest.mark.asyncio
async def test_trace_session_list_detail_apis_use_stable_dto_pages(
    trace_http: TraceHttp,
    tmp_path: Path,
) -> None:
    service, server, client_id, control = trace_http
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_id, workspace_key = await _seed_model_trace(service, workspace)
    second_id, _ = await _seed_model_trace(service, workspace)
    headers = _auth_headers(service, client_id, control)

    async with aiohttp.ClientSession() as http:
        sessions_response = await http.get(
            server.make_url("/api/v1/traces/sessions"),
            headers=headers,
            params={"limit": "1"},
        )
        sessions = await sessions_response.json()

        first_response = await http.get(
            server.make_url("/api/v1/traces"),
            headers=headers,
            params={"owner_session_id": "fixture-session", "limit": "1"},
        )
        first_page = await first_response.json()
        second_response = await http.get(
            server.make_url("/api/v1/traces"),
            headers=headers,
            params={
                "owner_session_id": "fixture-session",
                "limit": "1",
                "cursor": first_page["next_cursor"],
            },
        )
        second_page = await second_response.json()

        detail_response = await http.get(
            server.make_url(f"/api/v1/traces/{first_id}"),
            headers=headers,
        )
        detail = await detail_response.json()
        removed_raw = await http.get(
            server.make_url(f"/api/v1/traces/{first_id}/raw"), headers=headers,
        )

    assert sessions_response.status == 200
    assert len(sessions["items"]) == 1
    session_summary = sessions["items"][0]
    assert {key: value for key, value in session_summary.items() if key != "last_finished_at"} == {
        "workspace_key": workspace_key,
        "workspace_path": str(workspace),
        "owner_session_id": "fixture-session",
        "subject_session_id": None,
        "origin": "foreground",
        "record_count": 2,
    }
    assert (
        datetime.fromisoformat(session_summary["last_finished_at"].replace("Z", "+00:00")).tzinfo
        is not None
    )
    assert first_response.status == 200
    assert second_response.status == 200
    assert first_page["snapshot_sequence"] > 0
    assert first_page["next_cursor"]
    assert second_page["next_cursor"] is None
    assert {first_page["items"][0]["record_id"], second_page["items"][0]["record_id"]} == {
        first_id,
        second_id,
    }
    assert "request" not in first_page["items"][0]
    assert "result" not in first_page["items"][0]
    assert detail_response.status == 200
    assert detail["record"]["record_id"] == first_id
    assert detail["content"] == {"output": {"text": "fixture result", "reasoning": "", "tool_calls": []}}
    assert detail["related_tools"] == []
    assert detail["capture_issues"] == []
    assert removed_raw.status == 404


@pytest.mark.asyncio
async def test_trace_stats_count_failures_unknown_usage_and_tool_calls_once(
    trace_http: TraceHttp,
    tmp_path: Path,
) -> None:
    service, server, client_id, control = trace_http
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _record_id, workspace_key = await _seed_model_trace(service, workspace)
    await _seed_failed_model_trace(service, workspace)
    await _seed_tool_trace(service, workspace)

    async with aiohttp.ClientSession() as http:
        response = await http.get(
            server.make_url(f"/api/v1/traces/stats?workspace_key={workspace_key}"),
            headers=_auth_headers(service, client_id, control),
        )
        body = await response.json()

    assert response.status == 200
    assert body["totals"] == {
        "model_calls": 2,
        "tool_calls": 1,
        "success_count": 2,
        "failed_count": 1,
        "failure_rate": pytest.approx(1 / 3),
        "retry_count": 2,
        "retried_calls": 1,
        "unknown_usage_count": 1,
        "average_duration_ms": 70.0,
    }
    assert body["tokens"]["total_tokens"] == 13
    assert body["tokens"]["cached_input_tokens"] == 2
    assert body["by_model"] == [
        {
            "key": "fixture-model",
            "calls": 2,
            "tokens": {
                "input_tokens": 9,
                "output_tokens": 4,
                "total_tokens": 13,
                "cached_input_tokens": 2,
                "cache_creation_input_tokens": 0,
            },
        }
    ]
    assert body["by_tool"][0]["key"] == "read_file"
    assert body["by_tool"][0]["calls"] == 1
    assert body["by_error_category"] == [{"category": "cancelled", "calls": 1}]


@pytest.mark.asyncio
async def test_trace_http_auth_validation_missing_data_and_unavailable_storage(
    trace_http: TraceHttp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, server, client_id, control = trace_http
    headers = _auth_headers(service, client_id, control)

    async with aiohttp.ClientSession() as http:
        unauthenticated = await http.get(server.make_url("/api/v1/traces/stats"))
        oversized_page = await http.get(
            server.make_url("/api/v1/traces/sessions?limit=201"), headers=headers
        )
        invalid_filter = await http.get(
            server.make_url("/api/v1/traces?kind=unknown"), headers=headers
        )
        missing_record = await http.get(
            server.make_url("/api/v1/traces/missing-record"), headers=headers
        )
        invalid_raw = await http.get(
            server.make_url("/api/v1/traces/missing-record/raw?source=x&download=maybe"),
            headers=headers,
        )
        monkeypatch.setattr(service, "_trace_storage_available", False)
        unavailable = await http.get(server.make_url("/api/v1/traces/stats"), headers=headers)

    assert unauthenticated.status == 401
    assert oversized_page.status == 400
    assert invalid_filter.status == 400
    assert missing_record.status == 404
    assert invalid_raw.status == 404
    assert unavailable.status == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/traces", {"limit": "9" * 4301}),
        ("/traces/sessions", {"limit": "9" * 4301}),
        ("/traces/stats", {"from_at": "0001-01-01T00:00:00+01:00"}),
        ("/traces", {"to_at": "9999-12-31T23:59:59-01:00"}),
        (
            "/traces",
            {
                "cursor": base64.urlsafe_b64encode(
                    json.dumps(
                        {
                            "v": 1,
                            "snapshot": 2**63,
                            "finished_at": "2026-01-01T00:00:00Z",
                            "record_id": "fixture-record",
                        }
                    ).encode()
                ).decode()
            },
        ),
        (
            "/traces/sessions",
            {
                "cursor": base64.urlsafe_b64encode(
                    json.dumps({"v": 1, "snapshot": 2**63, "last": 0}).encode()
                ).decode()
            },
        ),
        (
            "/traces",
            {
                "cursor": base64.urlsafe_b64encode(
                    json.dumps(
                        {
                            "v": 1,
                            "snapshot": 0,
                            "finished_at": "0001-01-01T00:00:00+01:00",
                            "record_id": "fixture-record",
                        }
                    ).encode()
                ).decode()
            },
        ),
    ],
    ids=[
        "record-limit-digits",
        "session-limit-digits",
        "from-utc-underflow",
        "to-utc-overflow",
        "record-cursor-snapshot",
        "session-cursor-snapshot",
        "record-cursor-date",
    ],
)
async def test_trace_invalid_numeric_and_datetime_queries_return_400(
    trace_http: TraceHttp,
    path: str,
    query: dict[str, str],
) -> None:
    service, server, client_id, control = trace_http
    async with aiohttp.ClientSession() as http:
        response = await http.get(
            server.make_url(f"/api/v1{path}"),
            params=query,
            headers=_auth_headers(service, client_id, control),
        )
        assert response.status == 400
        body = await response.json()
    assert body["code"] == "validation_error"
    assert body["retryable"] is False


@pytest.mark.asyncio
async def test_trace_queries_hide_unpublished_expired_and_closed_records(
    trace_http: TraceHttp,
    tmp_path: Path,
) -> None:
    service, server, client_id, control = trace_http
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expired_id, _ = await _seed_model_trace(
        service,
        workspace,
        owner_session_id="expired-session",
        finished_at=datetime.now(UTC) - timedelta(days=31),
    )
    closed_id, workspace_key = await _seed_model_trace(
        service,
        workspace,
        owner_session_id="closed-session",
    )
    service.trace_recorder.close_session_scope(workspace_key, "closed-session")
    unpublished_id = await _seed_unpublished_trace(service, workspace, "pending-session")
    headers = _auth_headers(service, client_id, control)

    async with aiohttp.ClientSession() as http:
        records_response = await http.get(server.make_url("/api/v1/traces"), headers=headers)
        records = await records_response.json()
        sessions_response = await http.get(
            server.make_url("/api/v1/traces/sessions"), headers=headers
        )
        sessions = await sessions_response.json()
        stats_response = await http.get(server.make_url("/api/v1/traces/stats"), headers=headers)
        stats = await stats_response.json()
        expired_detail = await http.get(
            server.make_url(f"/api/v1/traces/{expired_id}"), headers=headers
        )
        closed_raw = await http.get(
            server.make_url(f"/api/v1/traces/{closed_id}/raw?source=fixture_blob"),
            headers=headers,
        )
        unpublished_detail = await http.get(
            server.make_url(f"/api/v1/traces/{unpublished_id}"), headers=headers
        )

    assert records_response.status == 200
    assert records["items"] == []
    assert sessions_response.status == 200
    assert sessions["items"] == []
    assert stats_response.status == 200
    assert stats["totals"]["model_calls"] == 0
    assert stats["totals"]["failure_rate"] is None
    assert expired_detail.status == 404
    assert closed_raw.status == 404
    assert unpublished_detail.status == 404


@pytest.mark.asyncio
async def test_trace_http_reads_cli_claimed_session_without_mutating_claim_or_creating_workspace(
    trace_http: TraceHttp,
    tmp_path: Path,
) -> None:
    service, server, web_client_id, control = trace_http
    workspace_path = tmp_path / "cli-workspace"
    workspace_path.mkdir()
    cli = await service.register_client("cli")
    workspace = await service.attach_workspace(cli.client_id, workspace_path)
    created = await service.create_session(cli.client_id, workspace.workspace_id)
    session_id = cast(str, created["session_id"])
    claim_result = await service.claim(cli.client_id, workspace.workspace_id, session_id)
    claim_data = cast(dict[str, object], claim_result["claim"])
    claim_version = cast(int, claim_data["claim_version"])
    claim_credential = cast(str, claim_data["reconnect_credential"])
    record_id, _workspace_key = await _seed_model_trace(
        service,
        workspace_path,
        owner_session_id=session_id,
    )
    token = create_credential(service.agent_home)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-Client": web_client_id,
        "X-Aide-Control": control,
    }
    cli_headers = {
        "Authorization": f"Bearer {token}",
        "X-Aide-Client": cli.client_id,
        "X-Aide-Claim": claim_credential,
    }
    session_url = server.make_url(
        f"/api/v1/workspaces/{workspace.workspace_id}/sessions/{session_id}"
        f"?claim_version={claim_version}"
    )

    async with aiohttp.ClientSession() as http:
        status_before = await http.get(server.make_url("/api/v1/service"), headers=headers)
        status_before_body = await status_before.json()
        session_before = await http.get(session_url, headers=cli_headers)
        session_before_body = await session_before.json()
        sessions = await http.get(server.make_url("/api/v1/traces/sessions"), headers=headers)
        records = await http.get(
            server.make_url(f"/api/v1/traces?owner_session_id={session_id}"), headers=headers
        )
        detail = await http.get(server.make_url(f"/api/v1/traces/{record_id}"), headers=headers)
        stats = await http.get(server.make_url("/api/v1/traces/stats"), headers=headers)
        sessions_body = await sessions.json()
        records_body = await records.json()
        await detail.read()
        stats_body = await stats.json()
        session_after = await http.get(session_url, headers=cli_headers)
        session_after_body = await session_after.json()
        status_after = await http.get(server.make_url("/api/v1/service"), headers=headers)
        status_after_body = await status_after.json()

    assert (
        status_before.status
        == session_before.status
        == sessions.status
        == records.status
        == detail.status
        == stats.status
        == session_after.status
        == status_after.status
        == 200
    )
    assert any(item["owner_session_id"] == session_id for item in sessions_body["items"])
    assert records_body["items"][0]["record_id"] == record_id
    assert stats_body["totals"]["model_calls"] == 1
    assert session_after_body["snapshot"] == session_before_body["snapshot"]
    assert (
        status_after_body["active_workspace_count"] == status_before_body["active_workspace_count"]
    )
