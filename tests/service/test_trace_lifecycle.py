from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import cast

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.service.runtime import AgentService
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceCallMetadata,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from tests.fixtures.service_runtime import workspace_loops
from tests.fixtures.trace import trace_queue_load, wait_for_trace_writer
from tests.service.test_service_concurrency import (
    _claim_version,
    _CollectingSink,
    _ConcurrentProvider,
)
from tests.service.test_service_transport import _prepare_agent_home


def _publish_expired_record(
    recorder: SQLiteTraceRecorder, workspace_path: Path, finished_at: datetime
) -> str:
    context = recorder.new_root(
        TraceRootSource(
            workspace_key=workspace_key_for(workspace_path),
            workspace_path=str(workspace_path),
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id="expired-session",
            subject_session_id="expired-session",
        )
    )
    handle = recorder.begin_call(
        context,
        TraceCallMetadata(
            kind=TraceKind.TOOL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=finished_at - timedelta(seconds=1),
            tool_name="read_file",
        ),
    )
    recorder.finish_call(
        handle,
        TraceOutcome(TraceStatus.SUCCESS, finished_at, 1),
        result={"output": "old"},
        usage=TraceTokenUsage(TraceUsageState.NOT_APPLICABLE),
        retry_count=0,
    )
    return handle.record_id


@pytest.mark.asyncio
async def test_service_owns_one_agent_home_trace_database_and_closes_it(
    tmp_path: Path,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    assert recorder.path == home.path / "traces.sqlite3"
    assert service.trace_storage_available is True
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    try:
        client = await service.register_client("cli")
        first = await service.attach_workspace(client.client_id, first_path)
        second = await service.attach_workspace(client.client_id, second_path)
        assert first.service.trace_recorder is second.service.trace_recorder is recorder
        record_id = _publish_expired_record(recorder, first_path, datetime.now(UTC))
        await wait_for_trace_writer(recorder)
        old_workspace_id = first.workspace_id
    finally:
        await service.stop()
    assert trace_queue_load(recorder)[0] == 0

    restarted = AgentService(home, ConfigLoader(home).load())
    await restarted.start()
    try:
        client = await restarted.register_client("cli")
        first = await restarted.attach_workspace(client.client_id, first_path)
        assert first.workspace_id != old_workspace_id
        detail = await restarted.trace_recorder.get_record(record_id)
        assert detail is not None
        assert detail.record.workspace_key == workspace_key_for(first_path)
        assert len((await restarted.trace_recorder.list_sessions()).items) == 1
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_foreground_run_and_auxiliary_calls_keep_separate_trace_purposes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    provider = _ConcurrentProvider()
    provider.release_a.set()
    monkeypatch.setattr(
        "aide.service.runtime.service.create_provider",
        lambda _configuration: provider,
    )
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, session_id)
        ack = await service.handle_command(
            client.client_id,
            {
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": _claim_version(claim),
                "request_id": "trace-foreground-input",
                "type": "input",
                "payload": {"text": "session-a"},
            },
        )
        run_id = cast(str, cast(dict[str, object], ack["result"])["run_id"])
        await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=2)
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)

        records = (await service.trace_recorder.list_records()).items
        by_purpose = {record.purpose: record for record in records}
        foreground = by_purpose[TracePurpose.AGENT_RUN]
        framing = by_purpose[TracePurpose.TASK_FRAMING]
        title = by_purpose[TracePurpose.TITLE_GENERATION]
        assert len(records) == 3
        assert foreground.origin is TraceOrigin.FOREGROUND
        assert framing.origin is TraceOrigin.FOREGROUND
        assert title.origin is TraceOrigin.AUXILIARY
        assert foreground.root_id == framing.root_id
        assert title.parent_root_id == foreground.root_id
        assert all(record.run_id == run_id for record in records)
        assert all(record.owner_session_id == session_id for record in records)
        assert all(record.subject_session_id == session_id for record in records)
        assert title.restore_run_token == foreground.restore_run_token
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_unavailable_trace_database_does_not_block_service_start_or_clear_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    provider = _ConcurrentProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _config: provider)
    database = home.path / "traces.sqlite3"
    damaged_content = b"unavailable SQLite storage"
    database.write_bytes(damaged_content)

    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        assert service.state == "ready"
        assert isinstance(service.trace_recorder, NoOpTraceRecorder)
        assert service.trace_storage_available is False
        assert database.read_bytes() == damaged_content
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, session_id)
        ack = await service.handle_command(
            client.client_id,
            {
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": _claim_version(claim),
                "request_id": "trace-unavailable-input",
                "type": "input",
                "payload": {"text": "trace outage"},
            },
        )
        run_id = cast(str, cast(dict[str, object], ack["result"])["run_id"])
        await asyncio.wait_for(sink.wait_for("run.completed", run_id), timeout=2)
        assert provider.user_inputs == ["trace outage"]
        assert any(
            message.get("role") == "assistant" and message.get("content") == "answer from session B"
            for message in workspace_loops(workspace)[session_id].loop.session.messages
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_start_sweeps_expired_history_in_the_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    home.initialize()
    database = home.path / "traces.sqlite3"
    finished_at = datetime.now(UTC) - timedelta(days=31)
    seed = SQLiteTraceRecorder(database, now=lambda: finished_at)
    expired_id = _publish_expired_record(seed, workspace_path, finished_at)
    await wait_for_trace_writer(seed)
    await seed.close()

    maintenance_finished = asyncio.Event()
    purge_expired = SQLiteTraceRecorder.purge_expired

    async def observe_maintenance(recorder: SQLiteTraceRecorder) -> None:
        await purge_expired(recorder)
        maintenance_finished.set()

    monkeypatch.setattr(
        SQLiteTraceRecorder,
        "purge_expired",
        observe_maintenance,
        raising=False,
    )
    service = AgentService(home)
    await service.start()
    try:
        await asyncio.wait_for(maintenance_finished.wait(), timeout=2)
    finally:
        await service.stop()

    reopened = SQLiteTraceRecorder(database, now=lambda: finished_at + timedelta(days=30))
    try:
        assert await reopened.get_record(expired_id) is None
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_failed_session_cleanup_is_retried_after_storage_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aide.trace.sqlite_recorder._WRITER_RETRY_DELAYS", (0.001, 0.002, 0.003))
    monkeypatch.setattr("aide.service.runtime.service._TRACE_PURGE_RETRY_INTERVAL_SECONDS", 0.01)
    service = AgentService(AgentHome(tmp_path / "agent-home"))
    await service.start()
    recorder = service.trace_recorder
    assert isinstance(recorder, SQLiteTraceRecorder)
    workspace_path = tmp_path / "workspace"
    workspace_key = workspace_key_for(workspace_path)
    record_id = _publish_expired_record(recorder, workspace_path, datetime.now(UTC))
    await wait_for_trace_writer(recorder)
    healthy = Event()
    failed, finished = asyncio.Event(), asyncio.Event()
    write_purge = recorder._write_purge
    purge_session = recorder.purge_session

    def unavailable_write(connection: sqlite3.Connection, payload: dict[str, object]) -> None:
        if not healthy.is_set():
            raise sqlite3.OperationalError("injected storage outage")
        write_purge(connection, payload)

    async def observe_cleanup(key: str, owner_id: str) -> None:
        try:
            await purge_session(key, owner_id)
        except RuntimeError:
            failed.set()
            raise
        finished.set()

    monkeypatch.setattr(recorder, "_write_purge", unavailable_write)
    monkeypatch.setattr(recorder, "purge_session", observe_cleanup)
    try:
        service.request_trace_session_purge(workspace_key, "expired-session")
        await asyncio.wait_for(failed.wait(), timeout=1)
        assert await recorder.get_record(record_id) is None
        healthy.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        with sqlite3.connect(recorder.path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM trace_records").fetchone()[0] == 0
    finally:
        healthy.set()
        await service.stop()
