from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from aide.trace.context import TraceRootSource
from aide.trace.sqlite_recorder import SQLiteTraceRecorder, _Event
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceCaptureLayer,
    TraceDirection,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceRecordFilters,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)

_NOW = datetime(2026, 10, 11, tzinfo=UTC)
_WORKSPACE_KEY = "sha256:test-workspace"
_WORKSPACE_PATH = "D:/workspace"


def _publish(
    recorder: SQLiteTraceRecorder,
    *,
    owner_session_id: str,
    finished_at: datetime,
    subject_session_id: str | None = None,
    restore_run_token: str | None = None,
) -> str:
    context = recorder.new_root(
        TraceRootSource(
            workspace_key=_WORKSPACE_KEY,
            workspace_path=_WORKSPACE_PATH,
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id=owner_session_id,
            subject_session_id=subject_session_id or owner_session_id,
            restore_run_token=restore_run_token,
        )
    )
    handle = recorder.begin_call(
        context,
        TraceCallMetadata(
            kind=TraceKind.TOOL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=finished_at - timedelta(milliseconds=1),
            tool_name="read_file",
        ),
    )
    recorder.append_raw(
        handle,
        source="tool_result",
        direction=TraceDirection.RESPONSE,
        layer=TraceCaptureLayer.TOOL_RESULT,
        sequence=0,
        data=b"captured result",
    )
    recorder.finish_call(
        handle,
        TraceOutcome(
            status=TraceStatus.SUCCESS,
            finished_at=finished_at,
            duration_ms=1,
        ),
        result={"output": "done"},
        usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
        retry_count=0,
    )
    return handle.record_id


@pytest.mark.asyncio
async def test_retention_filters_all_queries_and_purges_at_365_day_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.sqlite3"
    recorder = SQLiteTraceRecorder(path, now=lambda: _NOW)
    cutoff = _NOW - timedelta(days=365)
    expired_id = _publish(
        recorder,
        owner_session_id="expired-session",
        finished_at=cutoff - timedelta(seconds=1),
    )
    boundary_id = _publish(
        recorder,
        owner_session_id="boundary-session",
        finished_at=cutoff,
    )
    recent_id = _publish(
        recorder,
        owner_session_id="recent-session",
        finished_at=_NOW,
    )

    try:
        await recorder.flush()

        page = await recorder.list_records()
        assert {record.record_id for record in page.items} == {boundary_id, recent_id}
        assert await recorder.get_record(expired_id) is None
        raw = await recorder.read_raw(
            expired_id,
            source="tool_result",
            offset=0,
            limit_bytes=100,
        )
        assert raw.data == b""
        assert raw.total_bytes == 0
        sessions = await recorder.list_sessions()
        assert {session.owner_session_id for session in sessions.items} == {
            "boundary-session",
            "recent-session",
        }
        stats = await recorder.get_stats()
        assert stats.totals.tool_calls == 2

        await recorder.purge_expired()
    finally:
        await recorder.close()

    reopened = SQLiteTraceRecorder(
        path,
        now=lambda: cutoff - timedelta(seconds=1) + timedelta(days=365),
    )
    try:
        assert await reopened.get_record(expired_id) is None
        assert await reopened.get_record(boundary_id) is not None
        assert await reopened.get_record(recent_id) is not None
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_closed_session_is_hidden_before_async_purge_and_stale_events_cannot_publish(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    owner_record = _publish(
        recorder,
        owner_session_id="deleted-session",
        finished_at=_NOW,
    )
    child_record = _publish(
        recorder,
        owner_session_id="deleted-session",
        subject_session_id="subagent-session",
        finished_at=_NOW,
    )
    unrelated_record = _publish(
        recorder,
        owner_session_id="other-session",
        finished_at=_NOW,
    )
    stale_context = recorder.new_root(
        TraceRootSource(
            workspace_key=_WORKSPACE_KEY,
            workspace_path=_WORKSPACE_PATH,
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id="deleted-session",
            subject_session_id="deleted-session",
        )
    )
    stale_handle = recorder.begin_call(
        stale_context,
        TraceCallMetadata(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=_NOW,
            tool_name="read_file",
        ),
    )
    stale_attempt = recorder.begin_attempt(stale_handle, TraceAttemptMetadata(started_at=_NOW))
    recorder.append_raw(
        stale_attempt,
        source="response",
        direction=TraceDirection.RESPONSE,
        layer=TraceCaptureLayer.HTTP_BODY,
        sequence=0,
        data=b"pending candidate",
    )

    try:
        await recorder.flush()
        recorder.close_session_scope(_WORKSPACE_KEY, "deleted-session")

        assert {record.record_id for record in (await recorder.list_records()).items} == {
            unrelated_record
        }
        assert await recorder.get_record(owner_record) is None
        assert await recorder.get_record(child_record) is None
        sessions = await recorder.list_sessions()
        assert {session.owner_session_id for session in sessions.items} == {"other-session"}
        assert (await recorder.get_stats()).totals.tool_calls == 1

        recorder.finish_call(
            stale_handle,
            TraceOutcome(status=TraceStatus.SUCCESS, finished_at=_NOW, duration_ms=1),
            result={"output": "late"},
            usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        await recorder.flush()
        await recorder.purge_session(_WORKSPACE_KEY, "deleted-session")
        late_handle = recorder.begin_call(
            stale_context,
            TraceCallMetadata(kind=TraceKind.TOOL, purpose=TracePurpose.AGENT_RUN, started_at=_NOW),
        )
        recorder.append_raw(
            stale_attempt,
            source="response",
            direction=TraceDirection.RESPONSE,
            layer=TraceCaptureLayer.HTTP_BODY,
            sequence=1,
            data=b"late candidate",
        )
        recorder.finish_call(
            late_handle,
            TraceOutcome(TraceStatus.SUCCESS, _NOW, 1),
            result="late call",
            usage=TraceTokenUsage(TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        await recorder.flush()

        with sqlite3.connect(recorder.path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM trace_records").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM trace_roots").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM trace_attempts").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM trace_raw_chunks").fetchone()[0] == 1

        remaining = await recorder.list_records()
        assert {record.record_id for record in remaining.items} == {unrelated_record}
        assert await recorder.get_record(owner_record) is None
        assert await recorder.get_record(child_record) is None
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_restore_marks_only_discarded_run_tokens(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    restored_id = _publish(
        recorder,
        owner_session_id="session-1",
        finished_at=_NOW,
        restore_run_token="discarded-run",
    )
    retained_id = _publish(
        recorder,
        owner_session_id="session-1",
        finished_at=_NOW,
        restore_run_token="retained-run",
    )
    try:
        await recorder.flush()
        recorder.mark_restored(_WORKSPACE_KEY, "session-1", ("discarded-run",))
        await recorder.flush()

        records = await recorder.list_records(TraceRecordFilters(owner_session_id="session-1"))
        restored = {record.record_id: record.restored for record in records.items}
        assert restored == {restored_id: True, retained_id: False}
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_queries_remain_available_after_many_session_deletions(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    retained_id = _publish(recorder, owner_session_id="retained", finished_at=_NOW)
    deleted_id = _publish(recorder, owner_session_id="deleted", finished_at=_NOW)
    try:
        await recorder.flush()
        for index in range(10_100):
            recorder.close_session_scope(_WORKSPACE_KEY, f"deleted-{index}")
        recorder.close_session_scope(_WORKSPACE_KEY, "deleted")
        assert [record.record_id for record in (await recorder.list_records()).items] == [
            retained_id
        ]
        assert len((await recorder.list_sessions()).items) == 1
        assert await recorder.get_record(retained_id) is not None
        assert await recorder.get_record(deleted_id) is None
        assert (
            await recorder.read_raw(
                retained_id,
                source="tool_result",
                offset=0,
                limit_bytes=100,
            )
        ).data == b"captured result"
        assert (
            await recorder.read_raw(
                deleted_id,
                source="tool_result",
                offset=0,
                limit_bytes=100,
            )
        ).data == b""
        assert (await recorder.get_stats()).totals.tool_calls == 1
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_expiry_maintenance_cleans_multiple_batches_and_keeps_recent_data(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    expired_at = _NOW - timedelta(days=366)
    for index in range(501):
        _publish(recorder, owner_session_id=f"expired-{index}", finished_at=expired_at)
    retained_id = _publish(recorder, owner_session_id="retained", finished_at=_NOW)
    try:
        await recorder.flush()
        await recorder.purge_expired()
        with sqlite3.connect(recorder.path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM trace_records").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM trace_roots").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM trace_raw_chunks").fetchone()[0] == 1
        assert await recorder.get_record(retained_id) is not None
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_close_deadline_discards_remaining_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    assert await recorder.wait_until_ready()
    entered, release = Event(), Event()
    write_finish = recorder._write_finish

    def hold_writer(
        connection: sqlite3.Connection,
        record_id: str,
        payload: dict[str, object],
    ) -> bool:
        entered.set()
        assert release.wait(2)
        return write_finish(connection, record_id, payload)

    monkeypatch.setattr(recorder, "_write_finish", hold_writer)
    first_id = _publish(recorder, owner_session_id="first", finished_at=_NOW)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        second_id = _publish(recorder, owner_session_id="second", finished_at=_NOW)
        await asyncio.wait_for(recorder.close(timeout_seconds=0.01), timeout=0.5)
    finally:
        release.set()
        await asyncio.to_thread(recorder._worker.join, 1)
    assert not recorder._worker.is_alive()
    assert recorder.diagnostics.queued_events == 0
    assert await recorder.get_record(first_id) is None
    assert await recorder.get_record(second_id) is None


@pytest.mark.asyncio
async def test_session_purge_precedes_backlog_without_finishing_flush_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", now=lambda: _NOW)
    _publish(recorder, owner_session_id="deleted", finished_at=_NOW)
    await recorder.flush()
    entered, release, control_queued = Event(), Event(), Event()
    backlog_entered, backlog_release = Event(), Event()
    apply_transaction = recorder._apply_events_transaction
    enqueue_control = recorder._enqueue_control
    backlog_id = ""

    def hold_writer(connection: sqlite3.Connection, events: Sequence[_Event]) -> int:
        entered.set()
        assert release.wait(2)
        if any(event.record_id == backlog_id for event in events):
            backlog_entered.set()
            assert backlog_release.wait(2)
        return apply_transaction(connection, events)

    def observe_control(kind: str, record_id: str, payload: dict[str, object]) -> int:
        ticket = enqueue_control(kind, record_id, payload)
        control_queued.set()
        return ticket

    monkeypatch.setattr(recorder, "_apply_events_transaction", hold_writer)
    monkeypatch.setattr(recorder, "_enqueue_control", observe_control)
    _publish(recorder, owner_session_id="blocker", finished_at=_NOW)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        backlog_id = _publish(recorder, owner_session_id="backlog", finished_at=_NOW)
        purge = asyncio.create_task(recorder.purge_session(_WORKSPACE_KEY, "deleted"))
        assert await asyncio.to_thread(control_queued.wait, 1)
        flush = asyncio.create_task(recorder.flush())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(purge, timeout=1)
        assert await asyncio.to_thread(backlog_entered.wait, 1)
        assert not flush.done()
        backlog_release.set()
        await asyncio.wait_for(flush, timeout=1)
        assert await recorder.get_record(backlog_id) is not None
    finally:
        release.set()
        backlog_release.set()
        await recorder.close()
