from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

from aide.provider.session_configuration import REASONING_EFFORT_LEVELS, ReasoningEffort
from aide.trace.context import TraceContext, TraceRootSource
from aide.trace.sqlite_recorder import SQLiteTraceRecorder, _Event
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePage,
    TracePurpose,
    TraceRecordFilters,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from tests.fixtures.trace import measure_trace_writer, trace_queue_load, wait_for_trace_writer

_NOW = datetime(2026, 10, 11, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_related_tools_are_independent_of_list_paging_and_session_filters(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        parents = []
        expected: dict[str, set[str]] = {}
        for session in ("one", "two"):
            context = _new_context(recorder, session_id=session)
            model = recorder.begin_call(context, TraceCallMetadata(
                kind=TraceKind.MODEL, purpose=TracePurpose.AGENT_RUN, started_at=_NOW,
                protocol="anthropic",
            ))
            # Model input is excluded even if a recorder caller supplies it.
            recorder.set_request(model, {"messages": "private history", "tools": "definitions"})
            recorder.finish_call(model, _success(), result={"text": "answer"}, usage=_usage(), retry_count=0)
            parents.append(model)
            expected[model.record_id] = set()
            for index in range(3):
                tool = recorder.begin_call(context, replace(
                    _tool_metadata(index), parent_record_id=model.record_id,
                ))
                recorder.set_request(tool, {"path": f"{session}-{index}"})
                recorder.finish_call(tool, _success(), result={"content": "done"}, usage=_usage(), retry_count=0)
                expected[model.record_id].add(tool.record_id)
        # A mismatched owner cannot inject an association even with a parent ID.
        alien = recorder.begin_call(_new_context(recorder, session_id="alien"), replace(
            _tool_metadata(), parent_record_id=parents[0].record_id,
        ))
        recorder.finish_call(alien, _success(), result="alien", usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        assert len((await recorder.list_records(TraceRecordFilters(kind=TraceKind.MODEL), limit=1)).items) == 1
        for parent in parents:
            detail = await recorder.get_record(parent.record_id)
            assert detail is not None and detail.request is None
            assert {tool.record_id for tool in detail.related_tools} == expected[parent.record_id]
            for link in detail.related_tools:
                tool_detail = await recorder.get_record(link.record_id)
                assert tool_detail is not None and tool_detail.record.parent_record_id == parent.record_id
        with sqlite3.connect(recorder.path) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "trace_raw_chunks" not in tables and "trace_attempts" not in tables
            assert connection.execute("SELECT COUNT(*) FROM trace_records WHERE kind='model' AND request_json IS NOT NULL").fetchone()[0] == 0
            assert "private history" not in str(connection.execute("SELECT request_json, result_json FROM trace_records").fetchall())
    finally:
        await recorder.close()


def _new_context(recorder: SQLiteTraceRecorder, *, session_id: str = "session-1") -> TraceContext:
    return recorder.new_root(
        TraceRootSource(
            workspace_key="sha256:workspace",
            workspace_path="D:/work/project",
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id=session_id,
            subject_session_id=session_id,
        )
    )


def _tool_metadata(index: int = 0) -> TraceCallMetadata:
    return TraceCallMetadata(
        kind=TraceKind.TOOL,
        purpose=TracePurpose.AGENT_RUN,
        started_at=_NOW,
        tool_name="read_file",
        tool_call_id=f"tool-{index}",
    )


def _success(*, finished_at: datetime = _NOW) -> TraceOutcome:
    return TraceOutcome(
        status=TraceStatus.SUCCESS,
        finished_at=finished_at,
        duration_ms=12,
    )


def _usage() -> TraceTokenUsage:
    return TraceTokenUsage(
        state=TraceUsageState.REPORTED,
        input_tokens=7,
        output_tokens=3,
        total_tokens=10,
    )


@pytest.mark.asyncio
async def test_unpublished_model_candidates_are_invisible_and_success_publishes_only_latest(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        context = _new_context(recorder)
        handle = recorder.begin_call(
            context,
            TraceCallMetadata(
                kind=TraceKind.MODEL,
                purpose=TracePurpose.AGENT_RUN,
                started_at=_NOW,
                model="provider/initial",
                endpoint="https://initial.invalid/v1",
            ),
        )
        first = recorder.begin_attempt(
            handle,
            TraceAttemptMetadata(
                started_at=_NOW,
                model="provider/initial",
                endpoint="https://initial.invalid/v1",
            ),
        )
        recorder.set_request(first, {"prompt": "discarded candidate"})

        final = recorder.begin_attempt(
            handle,
            TraceAttemptMetadata(
                started_at=_NOW,
                model="provider/fallback",
                endpoint="https://fallback.invalid/v1",
            ),
        )
        recorder.set_request(final, {"prompt": "published candidate"})

        await wait_for_trace_writer(recorder)
        assert (await recorder.list_records()).items == ()
        assert await recorder.get_record(handle.record_id) is None

        recorder.finish_call(
            handle,
            _success(),
            result={"answer": "ok"},
            usage=_usage(),
            retry_count=1,
        )
        await wait_for_trace_writer(recorder)

        detail = await recorder.get_record(handle.record_id)
        assert detail is not None
        assert detail.request is None
        assert detail.result == {"answer": "ok"}
        assert detail.record.model == "provider/fallback"
        assert detail.record.endpoint == "https://fallback.invalid/v1"
        assert detail.record.capture_state.value == "complete"

    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_failed_model_publishes_error_without_model_input(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        context = _new_context(recorder)
        handle = recorder.begin_call(
            context,
            TraceCallMetadata(
                kind=TraceKind.MODEL,
                purpose=TracePurpose.AGENT_RUN,
                started_at=_NOW,
                model="provider/model",
            ),
        )
        attempt = recorder.begin_attempt(handle, TraceAttemptMetadata(started_at=_NOW))
        recorder.set_request(attempt, {"secret": "discard on failure"})
        recorder.finish_call(
            handle,
            TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=_NOW,
                duration_ms=20,
                error_category=TraceErrorCategory.PROVIDER_ERROR,
                error_message="provider unavailable",
            ),
            result=None,
            usage=TraceTokenUsage(state=TraceUsageState.UNKNOWN),
            retry_count=2,
        )
        await wait_for_trace_writer(recorder)

        detail = await recorder.get_record(handle.record_id)
        assert detail is not None
        assert detail.request is None
        assert detail.result is None
        assert detail.record.error_message == "provider unavailable"
        assert detail.record.status is TraceStatus.FAILED
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_finish_is_idempotent_and_concurrent_sessions_do_not_cross_records(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handles = []
        errors: list[BaseException] = []
        start = Barrier(3)

        def capture_session(session_id: str) -> None:
            try:
                start.wait(timeout=2)
                for index in range(20):
                    context = _new_context(recorder, session_id=session_id)
                    handle = recorder.begin_call(context, _tool_metadata(index))
                    recorder.set_request(handle, {"session": session_id, "index": index})
                    recorder.finish_call(
                        handle,
                        _success(),
                        result={"session": session_id, "index": index},
                        usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
                        retry_count=0,
                    )
                    handles.append((session_id, index, handle))
            except BaseException as error:
                errors.append(error)

        workers = [
            Thread(target=capture_session, args=(session_id,))
            for session_id in ("session-a", "session-b")
        ]
        for worker in workers:
            worker.start()
        start.wait(timeout=2)
        await asyncio.gather(*(asyncio.to_thread(worker.join) for worker in workers))
        assert errors == []

        assert len(handles) == 40
        recorder.finish_call(
            handles[0][2],
            _success(),
            result={"session": "duplicate"},
            usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        await wait_for_trace_writer(recorder)

        for session_id, index, handle in handles:
            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.request == {"session": session_id, "index": index}
            assert detail.result == {"session": session_id, "index": index}
            assert detail.record.owner_session_id == session_id
        page = await recorder.list_records(limit=100)
        assert len(page.items) == 40
        assert page.next_cursor is None
        assert page.snapshot_sequence == 40
        sessions = await recorder.list_sessions(limit=10)
        assert len(sessions.items) == 2
        assert {session.record_count for session in sessions.items} == {20}
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_stats_do_not_count_tool_usage_as_model_tokens(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        model = recorder.begin_call(
            _new_context(recorder),
            TraceCallMetadata(
                kind=TraceKind.MODEL,
                purpose=TracePurpose.AGENT_RUN,
                started_at=_NOW,
                model="provider/model",
            ),
        )
        tool = recorder.begin_call(_new_context(recorder), _tool_metadata())
        for handle in (model, tool):
            recorder.finish_call(
                handle,
                _success(),
                result={"ok": True},
                usage=_usage(),
                retry_count=0,
            )
        await wait_for_trace_writer(recorder)

        stats = await recorder.get_stats()
        assert stats.totals.model_calls == 1
        assert stats.totals.tool_calls == 1
        assert stats.tokens.input_tokens == 7
        assert stats.tokens.output_tokens == 3
        assert stats.tokens.total_tokens == 10
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_stats_keep_totals_and_breakdowns_consistent_across_dimensions(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    next_day = _NOW + timedelta(days=1)

    def record_call(
        *,
        kind: TraceKind,
        purpose: TracePurpose,
        started_at: datetime,
        model: str | None,
        tool_name: str | None,
        outcome: TraceOutcome,
        usage: TraceTokenUsage,
        retry_count: int,
    ) -> None:
        handle = recorder.begin_call(
            _new_context(recorder),
            TraceCallMetadata(
                kind=kind,
                purpose=purpose,
                started_at=started_at,
                model=model,
                tool_name=tool_name,
            ),
        )
        recorder.finish_call(
            handle,
            outcome,
            result={"ok": True},
            usage=usage,
            retry_count=retry_count,
        )

    try:
        record_call(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=_NOW,
            model="provider/a",
            tool_name=None,
            outcome=_success(),
            usage=_usage(),
            retry_count=0,
        )
        record_call(
            kind=TraceKind.TOOL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=_NOW,
            model="provider/a",
            tool_name="read_file",
            outcome=_success(),
            usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        record_call(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.SUBAGENT_RUN,
            started_at=next_day,
            model="provider/b",
            tool_name=None,
            outcome=TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=next_day,
                duration_ms=40,
                error_category=TraceErrorCategory.PROVIDER_ERROR,
                error_message="provider unavailable",
            ),
            usage=TraceTokenUsage(state=TraceUsageState.UNKNOWN),
            retry_count=2,
        )
        record_call(
            kind=TraceKind.TOOL,
            purpose=TracePurpose.SCHEDULE_RUN,
            started_at=next_day,
            model="provider/b",
            tool_name="exec",
            outcome=TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=next_day,
                duration_ms=40,
                error_category=TraceErrorCategory.TOOL_ERROR,
                error_message="tool failed",
            ),
            usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
            retry_count=1,
        )
        await wait_for_trace_writer(recorder)

        stats = await recorder.get_stats()
        assert stats.totals.model_calls == 2
        assert stats.totals.tool_calls == 2
        assert stats.totals.success_count == 2
        assert stats.totals.failed_count == 2
        assert stats.totals.failure_rate == 0.5
        assert stats.totals.retry_count == 3
        assert stats.totals.retried_calls == 2
        assert stats.totals.unknown_usage_count == 1
        assert stats.totals.average_duration_ms == 26
        assert stats.tokens.input_tokens == 7
        assert [(item.key, item.calls, item.tokens.input_tokens) for item in stats.by_model] == [
            ("provider/a", 2, 7),
            ("provider/b", 2, 0),
        ]
        assert [(item.key, item.calls) for item in stats.by_tool] == [
            ("exec", 1),
            ("read_file", 1),
        ]
        assert [(item.key, item.calls) for item in stats.by_purpose] == [
            ("agent_run", 2),
            ("schedule_run", 1),
            ("subagent_run", 1),
        ]
        assert [(item.category, item.calls) for item in stats.by_error_category] == [
            (TraceErrorCategory.PROVIDER_ERROR, 1),
            (TraceErrorCategory.TOOL_ERROR, 1),
        ]
        assert [
            (
                item.date,
                item.model_calls,
                item.tool_calls,
                item.successful_calls,
                item.failed_calls,
                item.tokens.input_tokens,
            )
            for item in stats.daily
        ] == [
            ("2026-10-11", 1, 1, 2, 0, 7),
            ("2026-10-12", 1, 1, 0, 2, 0),
        ]

        filtered = await recorder.get_stats(
            TraceRecordFilters(from_at=next_day, to_at=next_day)
        )
        assert filtered.totals.model_calls == 1
        assert filtered.totals.tool_calls == 1
        assert filtered.totals.average_duration_ms == 40
        assert [item.date for item in filtered.daily] == ["2026-10-12"]
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_ten_thousand_finished_records_survive_bounded_batch_writes_and_pagination(
    tmp_path: Path,
) -> None:
    database = tmp_path / "traces.sqlite3"
    expected_ids: set[str] = set()
    recorder = SQLiteTraceRecorder(database)
    try:
        for batch in range(10):
            for index in range(batch * 1000, (batch + 1) * 1000):
                handle = recorder.begin_call(_new_context(recorder), _tool_metadata(index))
                recorder.finish_call(
                    handle,
                    _success(),
                    result={"index": index},
                    usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
                    retry_count=0,
                )
                expected_ids.add(handle.record_id)
            await wait_for_trace_writer(recorder)

        actual_ids: list[str] = []
        cursor = None
        while True:
            page = await recorder.list_records(
                cursor=cursor,
                limit=500,
            )
            actual_ids.extend(item.record_id for item in page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        assert len(actual_ids) == 10_000
        assert len(set(actual_ids)) == 10_000
        assert set(actual_ids) == expected_ids
        assert (await recorder.get_stats()).totals.tool_calls == 10_000
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_trace_database_uses_wal_and_foreign_keys_and_restarts_clean_staging(
    tmp_path: Path,
) -> None:
    database = tmp_path / "traces.sqlite3"
    recorder = SQLiteTraceRecorder(database)
    context = _new_context(recorder)
    handle = recorder.begin_call(context, _tool_metadata())
    recorder.set_request(handle, {"pending": True})
    await wait_for_trace_writer(recorder)
    assert await recorder.get_record(handle.record_id) is None
    await recorder.close()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_records WHERE published = 0").fetchone()[
                0
            ]
            == 1
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2

    restarted = SQLiteTraceRecorder(database)
    try:
        await wait_for_trace_writer(restarted)
        assert await restarted.get_record(handle.record_id) is None
        connection = restarted._connect()
        try:
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            connection.close()
        with sqlite3.connect(database) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM trace_records WHERE published = 0"
                ).fetchone()[0]
                == 0
            )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_queue_payload_limit_marks_capture_incomplete_without_business_exceptions(
    tmp_path: Path,
) -> None:
    """Bounded queue behavior is covered through the public capture outcomes."""
    recorder = SQLiteTraceRecorder(
        tmp_path / "traces.sqlite3",
        max_queue_events=1,
        max_queue_payload_bytes=2048,
    )
    try:
        with measure_trace_writer(recorder) as measurements:
            context = _new_context(recorder)
            handle = recorder.begin_call(context, _tool_metadata())
            await wait_for_trace_writer(recorder)
            recorder.set_request(handle, {"large": "x" * 4096})
            recorder.finish_call(
                handle,
                _success(),
                result={"ok": True},
                usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
                retry_count=0,
            )
            await wait_for_trace_writer(recorder)

            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.record.capture_state.value in {"partial", "unavailable"}
            assert detail.capture_issues
            assert measurements["rejected_events"] > 0
            assert trace_queue_load(recorder)[0] <= 1
            assert trace_queue_load(recorder)[1] <= 2048
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_queue_event_limit_drops_arguments_without_exceeding_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aide.trace.sqlite_recorder as sqlite_module

    monkeypatch.setattr(sqlite_module, "_WRITER_BATCH_DELAY", 0.1)
    recorder = SQLiteTraceRecorder(
        tmp_path / "traces.sqlite3",
        max_queue_events=1,
        max_queue_payload_bytes=1024 * 1024,
    )
    try:
        with measure_trace_writer(recorder) as measurements:
            handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
            recorder.set_request(handle, {"path": "file.txt"})
            recorder.set_request(handle, b'tool result'.decode())
            await wait_for_trace_writer(recorder)
            recorder.finish_call(
                handle,
                _success(),
                result={"ok": True},
                usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
                retry_count=0,
            )
            await wait_for_trace_writer(recorder)

            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.record.capture_state.value == "partial"
            assert detail.capture_issues
            assert measurements["rejected_events"] >= 1
            assert any(issue.source == "request" for issue in detail.capture_issues)
            assert trace_queue_load(recorder)[0] <= recorder._max_queue_events
            assert trace_queue_load(recorder)[1] <= recorder._max_queue_payload_bytes
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["events", "payload"])
async def test_default_queue_budgets_bound_retained_capture(tmp_path: Path, budget: str) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
        await wait_for_trace_writer(recorder)
        with measure_trace_writer(recorder) as measurements, recorder._condition:
            block = b"x" if budget == "events" else b"x" * (64 * 1024)
            count = 4097 if budget == "events" else 1025
            for _ in range(count):
                recorder.set_request(handle, block.decode())
            events, payload_bytes = trace_queue_load(recorder)
            assert events <= 4096
            assert payload_bytes <= 64 * 1024 * 1024
            assert measurements["rejected_events"] > 0
            if budget == "events":
                assert events == 4096
            else:
                assert payload_bytes > 63 * 1024 * 1024
        await wait_for_trace_writer(recorder)
        recorder.finish_call(handle, _success(), result="ok", usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        detail = await recorder.get_record(handle.record_id)
        assert detail is not None and detail.capture_issues
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 128])
async def test_writer_retries_three_times_then_publishes_partial_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    import aide.trace.sqlite_recorder as sqlite_module

    assert sqlite_module._WRITER_RETRY_DELAYS == (0.1, 1.0, 5.0)
    monkeypatch.setattr(sqlite_module, "_WRITER_RETRY_DELAYS", (0.001, 0.002, 0.003))
    monkeypatch.setattr(sqlite_module, "_WRITER_BATCH_SIZE", batch_size)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        with measure_trace_writer(recorder) as measurements:
            original_apply = recorder._apply_event
            request_attempts = 0

            def fail_request_for_all_retries(connection: sqlite3.Connection, event: _Event) -> None:
                nonlocal request_attempts
                if event.kind == "request":
                    request_attempts += 1
                    raise sqlite3.OperationalError("controlled write failure")
                return original_apply(connection, event)

            monkeypatch.setattr(recorder, "_apply_event", fail_request_for_all_retries)
            with recorder._condition:
                handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
                recorder.set_request(handle, {"path": "file.txt"})
                recorder.finish_call(
                    handle,
                    _success(),
                    result={"text": "output"},
                    usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
                    retry_count=0,
                )
            await wait_for_trace_writer(recorder)

            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.request is None
            assert detail.record.capture_state.value == "partial"
            assert detail.capture_issues
            assert request_attempts == 4
            assert measurements["writer_failed_events"] == 1
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_read_queries_run_without_blocking_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
        recorder.finish_call(
            handle,
            _success(),
            result={"ok": True},
            usage=TraceTokenUsage(state=TraceUsageState.NOT_APPLICABLE),
            retry_count=0,
        )
        await wait_for_trace_writer(recorder)

        started = Event()
        release = Event()
        original = recorder._list_records_sync

        def delayed_query(
            filters: TraceRecordFilters | None,
            cursor: str | None,
            limit: int,
        ) -> TracePage:
            started.set()
            assert release.wait(1)
            return original(filters, cursor, limit)

        monkeypatch.setattr(recorder, "_list_records_sync", delayed_query)
        query = asyncio.create_task(recorder.list_records())
        assert await asyncio.to_thread(started.wait, 1)
        heartbeat = asyncio.create_task(asyncio.sleep(0))
        try:
            await asyncio.wait_for(heartbeat, timeout=0.1)
        finally:
            release.set()
        assert (await query).items
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [float("nan"), float("inf"), "\ud800", 10**5000],
    ids=["nan", "infinity", "surrogate", "large-integer"],
)
async def test_unserializable_content_degrades_capture_without_escaping(
    tmp_path: Path,
    content: object,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        with measure_trace_writer(recorder) as measurements:
            handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
            recorder.set_request(handle, {"content": content})
            recorder.finish_call(handle, _success(), result=content, usage=_usage(), retry_count=0)
            await wait_for_trace_writer(recorder)
            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.record.status is TraceStatus.SUCCESS
            assert detail.record.capture_state.value == "unavailable"
            assert {issue.source for issue in detail.capture_issues} == {"request", "result"}
            assert measurements["writer_failed_events"] == 0
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_kind", ["request", "finish"])
async def test_queue_budget_includes_arguments_and_error_metadata(
    tmp_path: Path,
    event_kind: str,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", max_queue_payload_bytes=2048)
    try:
        with measure_trace_writer(recorder) as measurements:
            handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
            await wait_for_trace_writer(recorder)
            with recorder._condition:
                if event_kind == "request":
                    recorder.set_request(handle, 'x' * 16384)
                else:
                    recorder.finish_call(
                        handle,
                        TraceOutcome(
                            status=TraceStatus.FAILED,
                            finished_at=_NOW,
                            duration_ms=1,
                            error_category=TraceErrorCategory.TOOL_ERROR,
                            error_message="x" * 16384,
                        ),
                        result=None,
                        usage=_usage(),
                        retry_count=0,
                    )
                assert trace_queue_load(recorder)[0] == 0
                assert trace_queue_load(recorder)[1] == 0
                assert measurements["rejected_events"] == 1
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_dropped_finishes_release_call_state_without_new_writer_failures(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3", max_queue_events=1)
    try:
        with measure_trace_writer(recorder) as measurements:
            with recorder._condition:
                for index in range(100):
                    handle = recorder.begin_call(_new_context(recorder), _tool_metadata(index))
                    recorder.finish_call(handle, _success(), result=None, usage=_usage(), retry_count=0)
            await wait_for_trace_writer(recorder)
            assert not recorder._states
            assert measurements["rejected_events"] >= 100
            assert measurements["writer_failed_events"] == 0
            assert not (await recorder.list_records()).items
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [*REASONING_EFFORT_LEVELS, None])
async def test_reasoning_effort_round_trips(
    tmp_path: Path, effort: ReasoningEffort | None
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handle = recorder.begin_call(
            _new_context(recorder),
            TraceCallMetadata(
                kind=TraceKind.MODEL,
                purpose=TracePurpose.AGENT_RUN,
                started_at=_NOW,
                reasoning_effort=effort,
            ),
        )
        recorder.finish_call(handle, _success(), result="answer", usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        detail = await recorder.get_record(handle.record_id)
        assert detail is not None and detail.record.reasoning_effort == effort
        assert (await recorder.list_records()).items[0].reasoning_effort == effort
        with sqlite3.connect(recorder.path) as connection:
            assert connection.execute(
                "SELECT reasoning_effort FROM trace_records WHERE record_id = ?",
                (handle.record_id,),
            ).fetchone()[0] == effort
            connection.execute(
                "UPDATE trace_records SET reasoning_effort = 'invalid' WHERE record_id = ?",
                (handle.record_id,),
            )
        with pytest.raises(ValueError, match="reasoning_effort"):
            await recorder.get_record(handle.record_id)
        with pytest.raises(ValueError, match="reasoning_effort"):
            await recorder.list_records()
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TraceStatus.SUCCESS, TraceStatus.FAILED])
async def test_final_model_metadata_replaces_initial_route_even_when_values_are_null(
    tmp_path: Path,
    status: TraceStatus,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handle = recorder.begin_call(
            _new_context(recorder),
            TraceCallMetadata(
                kind=TraceKind.MODEL,
                purpose=TracePurpose.AGENT_RUN,
                started_at=_NOW,
                model="initial",
                endpoint="https://initial.invalid",
                reasoning_effort="high",
            ),
        )
        attempt = recorder.begin_attempt(
            handle, TraceAttemptMetadata(started_at=_NOW, model="final")
        )
        recorder.set_request(attempt, {"model": "final"})
        recorder.finish_call(
            handle,
            TraceOutcome(
                status=status,
                finished_at=_NOW,
                duration_ms=12,
                final_attempt_duration_ms=5,
                error_category=TraceErrorCategory.PROVIDER_ERROR
                if status is TraceStatus.FAILED
                else None,
                error_message="failure" if status is TraceStatus.FAILED else None,
            ),
            result="answer",
            usage=_usage(),
            retry_count=1,
        )
        await wait_for_trace_writer(recorder)
        detail = await recorder.get_record(handle.record_id)
        assert detail is not None
        assert detail.record.model == "final"
        assert detail.record.endpoint is None
        assert detail.record.reasoning_effort is None
        assert detail.record.final_attempt_duration_ms == 5
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_finish_freezes_content_and_late_events_do_not_change_published_record(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        with measure_trace_writer(recorder) as measurements:
            with recorder._condition:
                handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
                recorder.set_request(handle, {"original": True})
                recorder.finish_call(
                    handle, _success(), result="original", usage=_usage(), retry_count=0
                )
                recorder.set_request(handle, {"late": True})
                recorder.finish_call(handle, _success(), result="late", usage=_usage(), retry_count=0)
            await wait_for_trace_writer(recorder)
            detail = await recorder.get_record(handle.record_id)
            assert detail is not None
            assert detail.request == {"original": True}
            assert detail.result == "original"
            assert len((await recorder.list_records()).items) == 1
            assert measurements["writer_failed_events"] == 0
            assert measurements["rejected_events"] == 0
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_record_pages_order_by_completion_and_preserve_publication_snapshot(
    tmp_path: Path,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handles = []
        for hours in (2, 0, 1, 1):
            handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
            recorder.finish_call(
                handle,
                _success(finished_at=_NOW + timedelta(hours=hours)),
                result="body",
                usage=_usage(),
                retry_count=0,
            )
            handles.append((hours, handle.record_id))
        await wait_for_trace_writer(recorder)
        first = await recorder.list_records(limit=1)
        late = recorder.begin_call(_new_context(recorder), _tool_metadata())
        recorder.finish_call(late, _success(), result=None, usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        ids = [item.record_id for item in first.items]
        cursor = first.next_cursor
        while cursor:
            page = await recorder.list_records(cursor=cursor, limit=1)
            assert page.snapshot_sequence == first.snapshot_sequence
            ids.extend(item.record_id for item in page.items)
            cursor = page.next_cursor
        assert ids == [record_id for _, record_id in sorted(handles, reverse=True)]
        assert late.record_id in {item.record_id for item in (await recorder.list_records()).items}
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_model_attempts_and_tools_from_two_sessions_remain_isolated(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    barrier = Barrier(2)
    try:

        def capture(session_id: str) -> tuple[str, str]:
            context = _new_context(recorder, session_id=session_id)
            model = recorder.begin_call(
                context,
                TraceCallMetadata(
                    kind=TraceKind.MODEL, purpose=TracePurpose.AGENT_RUN, started_at=_NOW
                ),
            )
            first = recorder.begin_attempt(
                model, TraceAttemptMetadata(started_at=_NOW, model="discard")
            )
            recorder.set_request(first, {"session": "discard"})
            barrier.wait(timeout=5)
            final = recorder.begin_attempt(
                model, TraceAttemptMetadata(started_at=_NOW, model=session_id)
            )
            recorder.set_request(final, {"session": session_id})
            tool = recorder.begin_call(context, _tool_metadata())
            recorder.set_request(tool, {"session": session_id})
            recorder.finish_call(
                model, _success(), result=session_id, usage=_usage(), retry_count=1
            )
            recorder.finish_call(tool, _success(), result=session_id, usage=_usage(), retry_count=0)
            return model.record_id, tool.record_id

        results = await asyncio.gather(
            asyncio.to_thread(capture, "session-a"),
            asyncio.to_thread(capture, "session-b"),
        )
        await wait_for_trace_writer(recorder)
        for session_id, record_ids in zip(("session-a", "session-b"), results, strict=True):
            for record_id in record_ids:
                detail = await recorder.get_record(record_id)
                assert detail is not None
                assert detail.record.owner_session_id == session_id
                assert detail.request == ({"session": session_id} if detail.record.kind is TraceKind.TOOL else None)
                assert detail.result == session_id
        assert len((await recorder.list_records()).items) == 4
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_record_listing_does_not_read_request_or_result_body_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
        recorder.set_request(handle, {"body": "request"})
        recorder.finish_call(handle, _success(), result="result", usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        original_connect = recorder._connect

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if (
                action == sqlite3.SQLITE_READ
                and table == "trace_records"
                and column in {"request_json", "result_json"}
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def metadata_connection(*, writer: bool = False) -> sqlite3.Connection:
            connection = original_connect(writer=writer)
            connection.set_authorizer(authorize)
            return connection

        monkeypatch.setattr(recorder, "_connect", metadata_connection)
        assert (await recorder.list_records()).items[0].record_id == handle.record_id
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_initialization_failure_isolated_from_capture_and_reported_by_queries(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "file"
    parent.write_text("not a directory", encoding="utf-8")
    recorder = SQLiteTraceRecorder(parent / "traces.sqlite3")
    try:
        handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
        recorder.set_request(handle, {"safe": True})
        recorder.finish_call(handle, _success(), result="safe", usage=_usage(), retry_count=0)
        with pytest.raises(RuntimeError, match="unavailable"):
            await recorder.list_records()
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_large_json_encoding_uses_writer_thread_and_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aide.trace.sqlite_recorder as sqlite_module

    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    started, release = Event(), Event()
    original_encode = sqlite_module._encode_json

    def delayed_encode(value: object) -> str:
        started.set()
        assert release.wait(5)
        return original_encode(value)

    try:
        await wait_for_trace_writer(recorder)
        monkeypatch.setattr(sqlite_module, "_encode_json", delayed_encode)
        handle = recorder.begin_call(_new_context(recorder), _tool_metadata())
        recorder.set_request(handle, {"content": "x" * 100_000})
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        release.set()
        recorder.finish_call(handle, _success(), result="done", usage=_usage(), retry_count=0)
        await wait_for_trace_writer(recorder)
        detail = await recorder.get_record(handle.record_id)
        assert detail is not None
        assert detail.request == {"content": "x" * 100_000}
    finally:
        release.set()
        await recorder.close()
