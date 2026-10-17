"""Measure the accepted Execution Trace query and stream replay thresholds."""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import sqlite3
import tempfile
from collections.abc import AsyncIterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from aide.config.config import (
    MemoryConfiguration,
    ModelConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from aide.provider.model_router import ModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.scope import trace_scope
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceKind,
    TraceOrigin,
    TracePurpose,
    TraceStatus,
    TraceUsageState,
)
from tests.fixtures.trace import measure_trace_writer, wait_for_trace_writer

RECORD_COUNT = 100_000
RECORDS_PER_SESSION = 10
QUERY_WARMUPS = 3
QUERY_SAMPLES = 30
STREAM_WARMUPS = 5
STREAM_SAMPLES = 30
STREAM_REPLAYS_PER_SAMPLE = 20
STREAM_EVENT_DELAY_SECONDS = 0.005


class _ReplayProvider:
    """Replay the same paced stream without network or SDK work."""

    def __init__(self) -> None:
        self.calls = 0
        self._response = ModelResponse(
            message=AssistantModelMessage(content="benchmark answer"),
            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
            finish_reason="stop",
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        del kwargs
        self.calls += 1
        for event in (TextDelta("benchmark "), ModelCompleted(self._response)):
            await asyncio.sleep(STREAM_EVENT_DELAY_SECONDS)
            yield event

    async def complete(self, **kwargs: Any) -> ModelResponse:
        del kwargs
        raise AssertionError("The stream benchmark must not call complete().")

    async def close(self) -> None:
        return None


def _percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _seed_query_database(database: Path, workspace: Path) -> None:
    source_now = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    workspace_key = workspace_key_for(workspace)
    workspace_path = str(workspace)
    roots = [
        (
            f"root-{session_index:05d}",
            source_now,
            workspace_key,
            workspace_path,
            TraceOrigin.FOREGROUND.value,
            TracePurpose.AGENT_RUN.value,
            f"session-{session_index:05d}",
            f"session-{session_index:05d}",
        )
        for session_index in range(RECORD_COUNT // RECORDS_PER_SESSION)
    ]
    records = (
        (
            f"record-{index:06d}",
            f"root-{index // RECORDS_PER_SESSION:05d}",
            workspace_key,
            workspace_path,
            TraceOrigin.FOREGROUND.value,
            f"session-{index // RECORDS_PER_SESSION:05d}",
            f"session-{index // RECORDS_PER_SESSION:05d}",
            TracePurpose.AGENT_RUN.value,
            TraceKind.TOOL.value,
            "benchmark",
            TraceStatus.SUCCESS.value,
            source_now,
            source_now,
            1,
            0,
            TraceUsageState.NOT_APPLICABLE.value,
            1,
            index + 1,
        )
        for index in range(RECORD_COUNT)
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            "INSERT INTO trace_roots(root_id, created_at, workspace_key, workspace_path, "
            "origin, purpose, owner_session_id, subject_session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            roots,
        )
        connection.executemany(
            "INSERT INTO trace_records(record_id, root_id, workspace_key, workspace_path, "
            "origin, owner_session_id, subject_session_id, purpose, kind, tool_name, status, "
            "started_at, finished_at, duration_ms, retry_count, usage_state, published, "
            "publish_sequence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        connection.execute(
            "UPDATE trace_meta SET publish_sequence = ? WHERE singleton = 1", (RECORD_COUNT,)
        )
        connection.commit()
    finally:
        connection.close()


async def _benchmark_queries(database: Path, workspace: Path) -> dict[str, object]:
    recorder = SQLiteTraceRecorder(database)
    try:
        if not await recorder.wait_until_ready():
            raise RuntimeError("Trace database initialization failed")
        await asyncio.to_thread(_seed_query_database, database, workspace)

        print("Trace query fixture ready.", flush=True)
        verbose = os.environ.get("AIDE_TRACE_BENCHMARK_VERBOSE") == "1"

        async def sample_sessions() -> float:
            if verbose:
                print("Starting session-list query.", flush=True)
            started = perf_counter()
            page = await recorder.list_sessions(limit=50)
            elapsed_ms = (perf_counter() - started) * 1_000
            if len(page.items) != 50 or page.next_cursor is None:
                raise AssertionError("The benchmark session page did not contain 50 rows.")
            if verbose:
                print(f"Session-list query: {elapsed_ms:.1f} ms.", flush=True)
            return elapsed_ms

        async def sample_stats() -> float:
            if verbose:
                print("Starting statistics query.", flush=True)
            started = perf_counter()
            stats = await recorder.get_stats()
            elapsed_ms = (perf_counter() - started) * 1_000
            if stats.totals.tool_calls != RECORD_COUNT:
                raise AssertionError("The benchmark statistics did not include every record.")
            if verbose:
                print(f"Statistics query: {elapsed_ms:.1f} ms.", flush=True)
            return elapsed_ms

        async def sample_records() -> float:
            started = perf_counter()
            page = await recorder.list_records(limit=50)
            elapsed_ms = (perf_counter() - started) * 1_000
            if len(page.items) != 50 or page.next_cursor is None:
                raise AssertionError("The benchmark record page did not contain 50 rows.")
            return elapsed_ms

        for _ in range(QUERY_WARMUPS):
            await sample_sessions()
            await sample_records()
            await sample_stats()
        record_samples = [await sample_records() for _ in range(QUERY_SAMPLES)]
        session_samples = []
        for index in range(QUERY_SAMPLES):
            session_samples.append(await sample_sessions())
            if (index + 1) % 5 == 0:
                print(f"Session-list samples: {index + 1}/{QUERY_SAMPLES}.", flush=True)
        stats_samples = []
        for index in range(QUERY_SAMPLES):
            stats_samples.append(await sample_stats())
            if (index + 1) % 5 == 0:
                print(f"Statistics samples: {index + 1}/{QUERY_SAMPLES}.", flush=True)
    finally:
        await recorder.close()

    database_bytes = sum(path.stat().st_size for path in database.parent.glob("traces.sqlite3*"))
    return {
        "records": RECORD_COUNT,
        "sessions": RECORD_COUNT // RECORDS_PER_SESSION,
        "semantic_bytes": 0,
        "database_bytes_including_wal": database_bytes,
        "samples": QUERY_SAMPLES,
        "session_list_samples_ms": [round(value, 3) for value in session_samples],
        "record_list_samples_ms": [round(value, 3) for value in record_samples],
        "statistics_samples_ms": [round(value, 3) for value in stats_samples],
        "session_list_p95_ms": round(_percentile_95(session_samples), 3),
        "record_list_p95_ms": round(_percentile_95(record_samples), 3),
        "statistics_p95_ms": round(_percentile_95(stats_samples), 3),
    }


def _new_router() -> tuple[ModelRouter, _ReplayProvider]:
    model = ModelConfiguration(
        context_window=8_192,
        max_output=1_024,
        temperature=0.2,
        reasoning_effort="mid",
        timeout=30,
    )
    provider_configuration = ProviderConfiguration(
        provider_id="benchmark-provider",
        protocol="openai-compatible",
        base_url="https://benchmark.invalid/v1",
        api_key="unused",
        models={"benchmark-model": model},
    )
    route = RouteConfiguration(
        provider_id=provider_configuration.provider_id,
        model="benchmark-model",
        context_window=model.context_window,
        max_output=model.max_output,
        temperature=model.temperature,
        reasoning_effort=model.reasoning_effort,
        timeout=model.timeout,
    )
    configuration = UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=50_000),
        memory=MemoryConfiguration(batch_size=10, schedule="0 * * * *"),
        models=ModelsConfiguration(
            providers={provider_configuration.provider_id: provider_configuration},
            routes={"chat": route},
        ),
    )
    provider = _ReplayProvider()
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
        jitter=None,
    )
    return router, provider


async def _replay_batch(
    router: ModelRouter,
    recorder: NoOpTraceRecorder | SQLiteTraceRecorder,
    source: TraceRootSource,
    count: int,
) -> tuple[float, tuple[tuple[object, ...], ...]]:
    outcomes: list[tuple[object, ...]] = []
    started = perf_counter()
    for _ in range(count):
        events: list[object] = []
        with trace_scope(recorder, source):
            async for event in router.stream(
                "chat",
                messages=[{"role": "user", "content": "fixed replay input"}],
                tools=(),
            ):
                if isinstance(event, TextDelta):
                    events.append(("text", event.delta))
                elif isinstance(event, ModelCompleted):
                    events.append(
                        (
                            "completed",
                            event.response.message.content,
                            event.response.usage.total_tokens,
                            event.response.finish_reason,
                        )
                    )
                else:
                    events.append((type(event).__name__,))
        outcomes.append(tuple(events))
    return (perf_counter() - started) * 1_000, tuple(outcomes)


async def _benchmark_streams(database: Path, workspace: Path) -> dict[str, object]:
    no_op = NoOpTraceRecorder()
    sqlite = SQLiteTraceRecorder(database)
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="stream-benchmark-session",
    )
    no_op_router, _no_op_provider = _new_router()
    sqlite_router, _sqlite_provider = _new_router()
    no_op_samples: list[float] = []
    sqlite_samples: list[float] = []
    no_op_outcomes: list[tuple[tuple[object, ...], ...]] = []
    sqlite_outcomes: list[tuple[tuple[object, ...], ...]] = []

    try:
        with measure_trace_writer(sqlite) as measurements:
            await _replay_batch(no_op_router, no_op, source, STREAM_WARMUPS)
            await _replay_batch(sqlite_router, sqlite, source, STREAM_WARMUPS)
            await wait_for_trace_writer(sqlite)
            for sample in range(STREAM_SAMPLES):
                order = (
                    ((no_op_router, no_op, no_op_samples, no_op_outcomes),
                     (sqlite_router, sqlite, sqlite_samples, sqlite_outcomes))
                    if sample % 2 == 0
                    else ((sqlite_router, sqlite, sqlite_samples, sqlite_outcomes),
                          (no_op_router, no_op, no_op_samples, no_op_outcomes))
                )
                for router, recorder, durations, outcomes in order:
                    elapsed, result = await _replay_batch(
                        router, recorder, source, STREAM_REPLAYS_PER_SAMPLE
                    )
                    durations.append(elapsed)
                    outcomes.append(result)
                    if isinstance(recorder, SQLiteTraceRecorder):
                        await wait_for_trace_writer(recorder)
    finally:
        await no_op_router.close()
        await sqlite_router.close()
        await sqlite.close()
        await no_op.close()

    if no_op_outcomes != sqlite_outcomes:
        raise AssertionError("NoOp and SQLite stream replay results differ.")
    expected_records = STREAM_WARMUPS + STREAM_SAMPLES * STREAM_REPLAYS_PER_SAMPLE
    no_op_total = sum(no_op_samples)
    sqlite_total = sum(sqlite_samples)
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        published_records = connection.execute(
            "SELECT COUNT(*) FROM trace_records WHERE published = 1"
        ).fetchone()[0]
        semantic_bytes = connection.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(request_json AS BLOB))), 0) + COALESCE(SUM(LENGTH(CAST(result_json AS BLOB))), 0) FROM trace_records"
        ).fetchone()[0]
    if published_records != expected_records:
        raise AssertionError("The SQLite stream replay did not publish every measured call.")
    return {
        "samples": STREAM_SAMPLES,
        "replays_per_sample": STREAM_REPLAYS_PER_SAMPLE,
        "total_replays_per_mode": expected_records,
        "event_delay_ms": STREAM_EVENT_DELAY_SECONDS * 1_000,
        "no_op_total_ms": round(no_op_total, 3),
        "sqlite_total_ms": round(sqlite_total, 3),
        "no_op_samples_ms": [round(value, 3) for value in no_op_samples],
        "sqlite_samples_ms": [round(value, 3) for value in sqlite_samples],
        "semantic_bytes": semantic_bytes,
        "database_bytes": database.stat().st_size,
        "no_op_p95_ms": round(_percentile_95(no_op_samples), 3),
        "sqlite_p95_ms": round(_percentile_95(sqlite_samples), 3),
        "total_increment_percent": round((sqlite_total / no_op_total - 1) * 100, 3),
        "same_business_events": True,
        "published_records": published_records,
        "writer_dropped_events": measurements["rejected_events"],
        "writer_failed_events": measurements["writer_failed_events"],
    }


async def _run() -> None:
    with tempfile.TemporaryDirectory(prefix="aide-trace-acceptance-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        query_result = await _benchmark_queries(root / "traces.sqlite3", workspace)
        stream_result = await _benchmark_streams(root / "stream-traces.sqlite3", workspace)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "sqlite": __import__("sqlite3").sqlite_version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "queries": query_result,
                "stream_replay": stream_result,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(_run())
