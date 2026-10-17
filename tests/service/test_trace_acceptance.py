"""T14: replay Service behavior across real Trace storage failures."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any, cast

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from aide.agent.session.restore import RestoreMode
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.errors import ModelCallError
from aide.provider.models import ModelCompleted, ModelStreamEvent, TextDelta
from aide.service.discovery import create_credential
from aide.service.runtime import AgentService
from aide.service.transport import create_app
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.sqlite_recorder import SQLiteTraceRecorder, _Event
from aide.trace.types import TraceRecordFilters
from aide.utils.errors import ErrorInfo
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import FakeClock
from tests.fixtures.trace import measure_trace_writer, wait_for_trace_writer
from tests.service.test_model_trace_capture import _response
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider


class _ReplayProvider(_ConcurrentProvider):
    def __init__(self) -> None:
        super().__init__(early_a_delta=True)
        self.success_attempts = 0
        self.failure_attempts = 0

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        if str(kwargs["messages"][0]["content"]).startswith("Generate a concise title"):
            return super().stream(**kwargs)
        user = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "user")
        if "session-a" in user:
            return super().stream(**kwargs)

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if "replay failure" in user:
                self.failure_attempts += 1
                raise ModelCallError(
                    ErrorInfo("provider_timeout", "Replay timeout", retryable=True)
                )
            self.success_attempts += 1
            if self.success_attempts == 1:
                raise ModelCallError(
                    ErrorInfo("provider_timeout", "Replay timeout", retryable=True)
                )
            if self.success_attempts == 2:
                yield ModelCompleted(
                    _response(
                        "", tool=ModelToolCall("replay-read", "read_file", '{"path":"fixture.txt"}')
                    )
                )
            else:
                yield TextDelta("Replay answer")
                yield ModelCompleted(_response("Replay answer"))

        return emit()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "healthy",
        "queue_full",
        "readonly",
        "initialization",
        "writer",
        "serialization",
        "close_timeout",
    ],
)
async def test_service_business_replay_matches_noop_under_trace_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    baseline = await _replay(tmp_path / "noop", "noop", monkeypatch)
    actual = await _replay(tmp_path / fault, fault, monkeypatch)
    assert actual == baseline


async def _replay(root: Path, fault: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    with monkeypatch.context() as patches, ExitStack() as observations:
        patches.setattr("aide.trace.sqlite_recorder._WRITER_RETRY_DELAYS", (0.001,) * 3)
        home = AgentHome(root / "home")
        home.initialize()
        (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
        if fault == "initialization":
            (home.path / "traces.sqlite3").write_bytes(b"not a SQLite database")
        provider = _ReplayProvider()
        patches.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
        service = AgentService(home, ConfigLoader(home).load())
        await service.start()
        service.model_router._clock = FakeClock(datetime.now(UTC))
        if fault == "noop":
            await service._stop_trace_storage()
            service._trace_recorder = NoOpTraceRecorder()
        recorder = service.trace_recorder
        fault_hits = 0
        measurements = (
            observations.enter_context(measure_trace_writer(recorder))
            if isinstance(recorder, SQLiteTraceRecorder)
            else {}
        )
        if isinstance(recorder, SQLiteTraceRecorder):
            apply_event = recorder._apply_event

            def apply_with_fault(connection: sqlite3.Connection, event: _Event) -> None:
                nonlocal fault_hits
                if fault == "readonly":
                    connection.execute("PRAGMA query_only = ON")
                if fault == "writer" and event.kind == "request":
                    fault_hits += 1
                    raise RuntimeError("Injected writer failure")
                if fault == "serialization" and event.kind == "request":
                    # Exercise the real writer JSON encoder with an unsupported value.
                    fault_hits += 1
                    event.payload["request"] = {"unserializable": object()}
                return apply_event(connection, event)

            patches.setattr(recorder, "_apply_event", apply_with_fault)
            if fault == "queue_full":
                patches.setattr(recorder, "_max_queue_payload_bytes", 1)
        path = root / "workspace"
        path.mkdir()
        (path / "fixture.txt").write_text("fixed tool result", encoding="utf-8")
        writer_release = Event()
        stopped = False
        try:
            client = await service.register_client("cli")
            sink = _CollectingSink()
            await service.connect_client(client.client_id, sink)
            workspace = await service.attach_workspace(client.client_id, path)
            sid = await workspace.create_draft(client.client_id, creation_scope="chat")
            await service.claim(client.client_id, workspace.workspace_id, sid)
            outcomes: list[object] = []
            for run_id, text in (
                ("success", "replay success"),
                ("failure", "replay failure"),
                ("cancel", "session-a"),
            ):
                claim = workspace._claims[sid]
                await workspace.input(client.client_id, sid, claim.version, text, run_id)
                if run_id == "cancel":
                    await asyncio.wait_for(provider.session_a_started.wait(), 5)
                    await workspace.cancel(client.client_id, sid, claim.version, run_id)
                await asyncio.wait_for(sink.wait_for("run.completed", run_id), 10)
                terminal = claim.loop.session.messages[-1]
                outcomes.append((terminal["content"], terminal["status"]))
            messages = workspace._claims[sid].loop.session.messages
            tool_results = [m["content"] for m in messages if m["role"] == "tool"]
            assert tool_results and "fixed tool result" in str(tool_results[0])
            assert provider.success_attempts == 3 and provider.failure_attempts == 5
            assert provider.session_a_cancelled.is_set()
            delivered: list[tuple[object, ...]] = []
            deltas: dict[str, list[str]] = {"success": [], "failure": [], "cancel": []}
            for event in sink.events:
                event_run_id = event.get("run_id")
                if not isinstance(event_run_id, str) or event_run_id not in deltas:
                    continue
                payload = cast(dict[str, Any], event["payload"])
                if event["type"] == "run.output":
                    message = payload["message"]
                    metadata = message["metadata"]
                    delivered.append(
                        (
                            event_run_id,
                            message["type"],
                            message["content"],
                            metadata.get("_stream_delta"),
                            metadata.get("_stream_end"),
                            metadata.get("_streamed"),
                            metadata.get("status"),
                        )
                    )
                    if metadata.get("_stream_delta"):
                        deltas[event_run_id].append(message["content"])
                elif event["type"] == "run.completed":
                    delivered.append((event_run_id, event["type"], payload["finish_reason"]))
            assert deltas == {
                "success": ["Replay answer"],
                "failure": [],
                "cancel": ["early from session A"],
            }

            if isinstance(recorder, SQLiteTraceRecorder):
                await wait_for_trace_writer(recorder)
                if fault == "queue_full":
                    assert measurements["rejected_events"] > 0
                elif fault == "readonly":
                    assert measurements["writer_failed_events"] > 0
                elif fault in {"writer", "serialization"}:
                    assert fault_hits > 0
                    if fault != "serialization":
                        assert measurements["writer_failed_events"] > 0
                    records = (await recorder.list_records()).items
                    assert any(r.capture_state.value == "partial" for r in records)
                    for record in records:
                        if record.capture_state.value == "partial":
                            detail = await recorder.get_record(record.record_id)
                            assert detail is not None and detail.capture_issues
                elif fault == "healthy":
                    await _assert_authenticated_execution(service, sid)
            elif fault == "initialization":
                assert not service.trace_storage_available

            claim = workspace._claims[sid]
            inspected = await service.inspect_restore(
                client.client_id,
                workspace.workspace_id,
                sid,
                claim.version,
                claim.credential,
                "inspect",
                1,
            )
            assert cast(dict[str, Any], inspected["restore_plan"])["anchor_id"] == 1
            restored = await service.commit_restore(
                client.client_id,
                workspace.workspace_id,
                sid,
                claim.version,
                claim.credential,
                "commit",
                1,
                RestoreMode.CONVERSATION_ONLY.value,
            )
            assert cast(dict[str, Any], restored["snapshot"])["messages"] == []
            if isinstance(recorder, SQLiteTraceRecorder) and fault == "healthy":
                await wait_for_trace_writer(recorder)
                assert all(
                    r.restored
                    for r in (
                        await recorder.list_records(TraceRecordFilters(owner_session_id=sid))
                    ).items
                )
            claim = workspace._claims[sid]
            await service.delete_session(
                client.client_id,
                workspace.workspace_id,
                sid,
                claim.version,
                claim.credential,
                "delete",
            )
            assert sid not in workspace._claims
            assert not list(workspace.workspace_state.sessions_directory.glob(f"{sid}*"))
            if isinstance(recorder, SQLiteTraceRecorder):
                assert (
                    await recorder.list_records(TraceRecordFilters(owner_session_id=sid))
                ).items == ()
                if fault == "close_timeout":
                    writer_entered = Event()
                    write_expire = recorder._write_expired

                    def hold_expire(
                        connection: sqlite3.Connection, payload: dict[str, object]
                    ) -> int:
                        writer_entered.set()
                        writer_release.wait(10)
                        return write_expire(connection, payload)

                    patches.setattr(recorder, "_write_expired", hold_expire)
                    pending = asyncio.create_task(recorder.purge_expired())
                    assert await asyncio.to_thread(writer_entered.wait, 5)
                    started = monotonic()
                    await asyncio.wait_for(service.stop(), 5)
                    stopped = True
                    assert monotonic() - started < 4
                    assert recorder._worker.is_alive()
                    writer_release.set()
                    await asyncio.to_thread(recorder._worker.join, 5)
                    assert not recorder._worker.is_alive()
                    await asyncio.gather(pending, return_exceptions=True)
            return {
                "terminals": outcomes,
                "tool_results": tool_results,
                "stream_order": delivered,
                "retry_attempts": (provider.success_attempts, provider.failure_attempts),
                "cancelled": provider.session_a_cancelled.is_set(),
                "restored_messages": [],
                "session_deleted": True,
            }
        finally:
            writer_release.set()
            provider.release_a.set()
            if not stopped:
                await service.stop()


async def _assert_authenticated_execution(service: AgentService, sid: str) -> None:
    web = await service.register_client("web")
    assert web.web_control_credential is not None
    headers = {
        "Authorization": f"Bearer {create_credential(service.agent_home)}",
        "X-Aide-Client": web.client_id,
        "X-Aide-Control": web.web_control_credential,
    }
    workspace = next(w for w in service._workspaces.values() if sid in w._claims)
    claim = workspace._claims[sid]
    identity = (claim.client_id, claim.version, claim.credential)
    async with TestServer(create_app(service)) as server, aiohttp.ClientSession() as http:
        async with http.get(server.make_url("/api/v1/traces")) as unauthenticated:
            assert unauthenticated.status == 401
        async with http.get(
            server.make_url(f"/api/v1/traces?owner_session_id={sid}"), headers=headers
        ) as response:
            assert response.status == 200
            records = (await response.json())["items"]
        tools = [r for r in records if r["kind"] == "tool"]
        assert len(tools) == 1 and tools[0]["tool_name"] == "read_file"
        assert tools[0]["parent_record_id"] in {r["record_id"] for r in records}
        for record in records:
            async with http.get(
                server.make_url(f"/api/v1/traces/{record['record_id']}"), headers=headers
            ) as response:
                assert response.status == 200
                detail = await response.json()
                assert detail["record"]["record_id"] == record["record_id"]
        current_claim = workspace._claims[sid]
        assert (
            current_claim.client_id,
            current_claim.version,
            current_claim.credential,
        ) == identity
