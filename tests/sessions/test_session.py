import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from myclaw.agent.context_budget import CONTEXT_ESTIMATOR_VERSION, ContextUsageSnapshot
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.workspace.state import WorkspaceState
from tests.fixtures.session import seed_session_state

LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
UPDATED_AT = CREATED_AT + timedelta(seconds=5)
SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260711-153012-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e"
SCHEDULE_JOB_ID = "6fa459ea-ee8a-4ca4-894e-db77e160355e"
RESTORE_TOKENS = (
    UUID("550e8400-e29b-41d4-a716-446655440000"),
    UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    UUID("7c9e6679-7425-40de-944b-2b7b2c3d4e5f"),
    UUID("8c9e6679-7425-40de-944b-2b7b2c3d4e5f"),
)
ZERO_USAGE = {
    "model_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def _context_usage() -> dict[str, object]:
    return ContextUsageSnapshot(
        requested_route="chat",
        selected_route="chat",
        provider_id="provider-1",
        model="model-1",
        context_window=8192,
        max_output=2048,
        anchor_estimated_tokens=120,
        estimator_version=CONTEXT_ESTIMATOR_VERSION,
        run_projected_tokens=240,
        run_projection_source="reported_delta",
    ).to_dict()


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _header(**updates: Any) -> dict[str, Any]:
    header: dict[str, Any] = {
        "session_id": SESSION_ID,
        "created_at": CREATED_AT.isoformat(timespec="milliseconds"),
        "updated_at": UPDATED_AT.isoformat(timespec="milliseconds"),
        "last_compacted": 0,
        "metadata": {
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
        },
    }
    header.update(updates)
    return header


def _write_jsonl(
    state: WorkspaceState,
    records: list[dict[str, Any]],
    *,
    trailing_newline: bool = True,
) -> Path:
    content = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records
    )
    if trailing_newline:
        content += "\n"
    path = state.sessions_directory / f"{SESSION_ID}.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def _assert_local_timestamp(value: Any) -> None:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset() == datetime.now().astimezone().utcoffset()


def test_create_starts_a_memory_only_session_with_private_identity_generation(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)

    session = Session.create(state)

    timestamp, uuid_text = session.session_id.split("_", maxsplit=1)
    assert timestamp == session.created_at.strftime("%Y%m%d-%H%M%S-%f")
    assert UUID(uuid_text).version == 4
    assert str(UUID(uuid_text)) == uuid_text
    assert session.created_at.utcoffset() is not None
    assert session.updated_at == session.created_at
    assert session.messages == []
    assert session.metadata == {
        "title": "Untitled session",
        "token_usage": ZERO_USAGE,
        "summary": "",
    }
    assert session.last_compacted == 0
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()

    with pytest.raises(TypeError, match=r"Session\.create\(\) or Session\.load\(\)"):
        Session()


def test_seed_session_state_replaces_public_state_with_detached_explicit_values(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    timestamp = CREATED_AT.isoformat(timespec="milliseconds")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect the file.", "timestamp": timestamp},
        {
            "role": "assistant",
            "content": "Reading it.",
            "timestamp": timestamp,
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
        },
        {
            "role": "tool",
            "content": "contents",
            "timestamp": timestamp,
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": {"path": "artifact.txt"},
        },
    ]
    metadata: dict[str, Any] = {
        "title": "Seeded session",
        "token_usage": {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        },
        "summary": "Known state",
        "extension": {"values": ["before"]},
    }
    identity = (session.session_id, session.created_at, session.updated_at)

    seed_session_state(
        session,
        messages=messages,
        metadata=metadata,
        last_compacted=2,
    )
    messages[0]["content"] = "changed"
    messages[1]["token_usage"]["input_tokens"] = 999
    messages[2]["artifact"]["path"] = "changed.txt"
    metadata["token_usage"]["input_tokens"] = 999
    metadata["extension"]["values"].append("after")

    assert session.messages[0]["content"] == "Inspect the file."
    assert session.messages[1]["token_usage"]["input_tokens"] == 2
    assert session.messages[2]["artifact"] == {"path": "artifact.txt"}
    assert session.metadata["token_usage"]["input_tokens"] == 2
    assert session.metadata["extension"] == {"values": ["before"]}
    assert session.last_compacted == 2
    assert (session.session_id, session.created_at, session.updated_at) == identity
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()


def test_create_schedule_session_uses_a_lazy_isolated_storage_partition(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)

    session = Session.create(
        state,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=SCHEDULE_JOB_ID,
        now=lambda: CREATED_AT,
    )

    assert session.session_id == f"schedule_{SCHEDULE_JOB_ID}"
    assert not state.schedule_sessions_directory.exists()

    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Run the scheduled task.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.close()

    assert (state.schedule_sessions_directory / f"{session.session_id}.jsonl").exists()
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()
    loaded = Session.load(state, session.session_id)
    assert loaded.messages == session.messages


@pytest.mark.parametrize(
    "job_id",
    [
        "6FA459EA-EE8A-4CA4-894E-DB77E160355E",
        "550e8400-e29b-11d4-a716-446655440000",
        "not-a-uuid",
    ],
)
def test_schedule_session_requires_a_canonical_uuid4_job_id(
    agent_home: Path,
    workspace: Path,
    job_id: str,
) -> None:
    state = _state(workspace, agent_home)

    with pytest.raises(ValueError, match="canonical UUID4"):
        Session.create(
            state,
            partition=SessionStoragePartition.SCHEDULE,
            job_id=job_id,
            now=lambda: CREATED_AT,
        )


@pytest.mark.asyncio
async def test_persist_writes_one_complete_compact_utf8_snapshot_atomically(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    timestamps = iter((CREATED_AT, CREATED_AT + timedelta(seconds=1), UPDATED_AT))
    session = Session.create(
        state,
        now=timestamps.__next__,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    message_timestamp = next(timestamps)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "请读取 README。",
                "timestamp": message_timestamp.isoformat(timespec="milliseconds"),
                "extension": {"nested": ["value"]},
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "- Read the README.",
        },
        last_compacted=0,
    )
    replacements: list[tuple[Path, bytes]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append((target, content))
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)

    session.persist()
    expected = (
        '{"session_id":"20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000",'
        '"created_at":"2026-07-11T15:30:12.123+08:00",'
        '"updated_at":"2026-07-11T15:30:17.123+08:00",'
        '"last_compacted":0,'
        '"metadata":{"title":"Untitled session",'
        '"token_usage":{"model_calls":0,"input_tokens":0,'
        '"output_tokens":0,"total_tokens":0},"summary":"- Read the README."}}\n'
        '{"role":"user","content":"请读取 README。",'
        '"timestamp":"2026-07-11T15:30:13.123+08:00",'
        '"extension":{"nested":["value"]}}\n'
    ).encode()

    assert replacements == []
    await asyncio.sleep(0)

    path = state.sessions_directory / f"{SESSION_ID}.jsonl"
    raw = HOST_FILESYSTEM.path_for_io(path).read_bytes()

    assert replacements == [(path, expected)]
    assert raw == expected
    assert b"\xe8\xaf\xb7\xe8\xaf\xbb" in raw
    loaded = Session.load(state, SESSION_ID)
    assert loaded.metadata["summary"] == "- Read the README."
    assert loaded.messages == session.messages


@pytest.mark.asyncio
async def test_persist_freezes_each_call_and_finishes_snapshots_in_call_order(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Before mutation",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    snapshots: list[list[dict[str, Any]]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(_target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(records[1:])
        replace(_target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)

    session.persist()
    session.messages[0]["content"] = "After mutation"
    session.persist()
    await asyncio.sleep(0)

    assert [snapshot[0]["content"] for snapshot in snapshots] == [
        "Before mutation",
        "After mutation",
    ]
    assert Session.load(state, session.session_id).messages[0]["content"] == "After mutation"


@pytest.mark.asyncio
async def test_persist_retries_a_transient_write_with_async_backoff(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Retry this snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    attempts: list[bytes] = []
    delays: list[float] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_twice_then_replace(target: Path, content: bytes) -> None:
        attempts.append(content)
        if len(attempts) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def immediate_backoff(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_twice_then_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)

    session.persist()
    await yield_once(0)

    assert len(attempts) == 3
    assert delays == [0.1, 0.2]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_persist_retries_each_snapshot_before_starting_the_next_snapshot(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    first_message = {
        "role": "user",
        "content": "First snapshot",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
    }
    seed_session_state(
        session,
        messages=[first_message],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    snapshots: list[tuple[str, ...]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_first_snapshot_twice(target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(tuple(record["content"] for record in records[1:]))
        if len(snapshots) <= 2:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def immediate_backoff(_delay: float) -> None:
        return

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_first_snapshot_twice)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)

    session.persist()
    seed_session_state(
        session,
        messages=[
            first_message,
            {
                "role": "user",
                "content": "Second snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.persist()
    await yield_once(0)

    assert snapshots == [
        ("First snapshot",),
        ("First snapshot",),
        ("First snapshot",),
        ("First snapshot", "Second snapshot"),
    ]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_pending_persist_waiter_cancellation_does_not_cancel_snapshots(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Keep this snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    first_write_failed = asyncio.Event()
    release_backoff = asyncio.Event()
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep
    attempts = 0

    def fail_once_then_replace(target: Path, content: bytes) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_write_failed.set()
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def blocked_backoff(_delay: float) -> None:
        await release_backoff.wait()

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_once_then_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await first_write_failed.wait()
    waiter = asyncio.create_task(session.wait_for_pending_persist())
    await yield_once(0)
    waiter.cancel()
    await yield_once(0)

    assert not waiter.done()
    release_backoff.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_pending_persist_wait_retrieves_already_completed_task_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)

    class TrackingTask(asyncio.Task[None]):
        result_calls = 0

        def result(self) -> None:
            self.result_calls += 1
            return super().result()

    async def fail() -> None:
        raise RuntimeError("private persistence detail")

    task = TrackingTask(fail())
    session._persist_tasks.add(task)
    await asyncio.sleep(0)

    await session.wait_for_pending_persist()

    assert task.result_calls == 1
    assert session._persist_tasks == set()


@pytest.mark.asyncio
async def test_pending_persist_wait_drains_snapshot_queued_while_waiting(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    first_message = {
        "role": "user",
        "content": "First snapshot",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
    }
    seed_session_state(
        session,
        messages=[first_message],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    first_write_failed = asyncio.Event()
    release_backoff = asyncio.Event()
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep
    attempts = 0

    def fail_once_then_replace(target: Path, content: bytes) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_write_failed.set()
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def blocked_backoff(_delay: float) -> None:
        await release_backoff.wait()

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_once_then_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await first_write_failed.wait()
    waiter = asyncio.create_task(session.wait_for_pending_persist())
    await yield_once(0)
    seed_session_state(
        session,
        messages=[
            first_message,
            {
                "role": "user",
                "content": "Second snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.persist()
    release_backoff.set()

    await waiter

    assert Session.load(state, session.session_id).messages == session.messages
    assert session._persist_tasks == set()


@pytest.mark.asyncio
async def test_pending_persist_wait_converges_with_concurrent_abandon(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Abandoned snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    backoff_started = asyncio.Event()
    yield_once = asyncio.sleep

    def fail_write(_target: Path, _content: bytes) -> None:
        raise OSError("transient snapshot failure")

    async def blocked_backoff(_delay: float) -> None:
        backoff_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_write)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await backoff_started.wait()
    waiter = asyncio.create_task(session.wait_for_pending_persist())
    await yield_once(0)
    session.abandon()

    await waiter

    assert session._persist_tasks == set()
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()


@pytest.mark.asyncio
async def test_abandon_cancels_every_pending_snapshot_when_latest_has_not_started(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    first_message = {
        "role": "user",
        "content": "First snapshot",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
    }
    seed_session_state(
        session,
        messages=[first_message],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    writes: list[bytes] = []
    backoff_started = asyncio.Event()
    backoff_cancelled = asyncio.Event()
    release_backoff = asyncio.Event()
    yield_once = asyncio.sleep

    def fail_first_write(_target: Path, content: bytes) -> None:
        writes.append(content)
        raise OSError("transient snapshot failure")

    async def blocked_backoff(_delay: float) -> None:
        backoff_started.set()
        try:
            await release_backoff.wait()
        except asyncio.CancelledError:
            backoff_cancelled.set()
            raise

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_first_write)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await yield_once(0)
    assert backoff_started.is_set()

    seed_session_state(
        session,
        messages=[
            first_message,
            {
                "role": "user",
                "content": "Second snapshot",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.persist()
    session.abandon()
    await yield_once(0)

    try:
        assert not release_backoff.is_set()
        assert backoff_cancelled.is_set()
        assert len(writes) == 1
        assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()
    finally:
        release_backoff.set()
        await yield_once(0)

    assert len(writes) == 1
    session.abandon()


@pytest.mark.asyncio
async def test_close_wins_against_an_old_async_snapshot_in_backoff(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    old_message = {
        "role": "user",
        "content": "Old snapshot",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
    }
    seed_session_state(
        session,
        messages=[old_message],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    snapshots: list[tuple[str, ...]] = []
    backoff_started = asyncio.Event()
    release_backoff = asyncio.Event()
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_async_then_save_sync(target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(tuple(record["content"] for record in records[1:]))
        if len(snapshots) == 1:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def blocked_backoff(_delay: float) -> None:
        backoff_started.set()
        await release_backoff.wait()

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_async_then_save_sync)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await yield_once(0)
    assert backoff_started.is_set()

    seed_session_state(
        session,
        messages=[
            old_message,
            {
                "role": "user",
                "content": "Final state",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.close()
    assert snapshots == [("Old snapshot",), ("Old snapshot", "Final state")]

    release_backoff.set()
    await yield_once(0)

    assert snapshots == [("Old snapshot",), ("Old snapshot", "Final state")]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_abandon_rejects_mutators_and_close_does_not_save(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Before abandonment",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    original_messages = copy.deepcopy(session.messages)
    original_metadata = copy.deepcopy(session.metadata)

    session.abandon()

    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.commit_agent_run(
            [{"role": "user", "content": "Rejected"}],
            pending_last_compacted=0,
            pending_action_summary=None,
        )
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata(title="Rejected")
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.commit_agent_run(
            None,  # type: ignore[arg-type]
            pending_last_compacted=0,
            pending_action_summary=None,
        )
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata("Rejected")  # type: ignore[arg-type]

    session.persist()
    session.close()

    assert session.messages == original_messages
    assert session.metadata == original_metadata
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()


def test_close_then_abandon_rejects_later_mutation_without_another_save(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Saved before abandonment",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.close()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    saved = path.read_bytes()

    session.abandon()

    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.commit_agent_run(
            [{"role": "user", "content": "Rejected"}],
            pending_last_compacted=0,
            pending_action_summary=None,
        )
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata(title="Rejected")
    session.persist()
    session.close()

    assert path.read_bytes() == saved


@pytest.mark.asyncio
async def test_ordinary_persist_failure_is_silent_and_a_later_persist_is_independent(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    first_message = {
        "role": "user",
        "content": "First attempt",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
    }
    seed_session_state(
        session,
        messages=[first_message],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    attempts: list[bytes] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_replace(_target: Path, content: bytes) -> None:
        attempts.append(content)
        raise OSError("simulated snapshot failure")

    async def immediate_backoff(_delay: float) -> None:
        return

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)
    session.persist()
    await yield_once(0)
    assert len(attempts) == 3

    replacements: list[bytes] = []

    def record_later_replace(target: Path, content: bytes) -> None:
        replacements.append(content)
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_later_replace)
    seed_session_state(
        session,
        messages=[
            first_message,
            {
                "role": "user",
                "content": "Second attempt",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    session.persist()
    await yield_once(0)

    assert len(attempts) == 3
    assert len(replacements) == 1
    assert [message["content"] for message in Session.load(state, session.session_id).messages] == [
        "First attempt",
        "Second attempt",
    ]


@pytest.mark.asyncio
async def test_empty_session_persist_and_close_remain_unmaterialized(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state)

    session.persist()
    session.close()
    await asyncio.sleep(0)

    assert not state.sessions_directory.exists()


def test_close_retries_latest_snapshot_with_bounded_delays(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Save during shutdown",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    attempts: list[bytes] = []
    sleeps: list[float] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_twice_then_replace(target: Path, content: bytes) -> None:
        attempts.append(content)
        if len(attempts) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_twice_then_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", sleeps.append)

    session.close()

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]
    assert Session.load(state, session.session_id).messages == session.messages


def test_close_swallows_failure_after_three_attempts(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Best effort shutdown",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    attempts: list[bytes] = []
    sleeps: list[float] = []

    def fail_replace(_target: Path, content: bytes) -> None:
        attempts.append(content)
        raise OSError("permanent snapshot failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", sleeps.append)

    session.close()

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]


@pytest.mark.asyncio
async def test_close_supersedes_queued_persist_and_refreshes_each_attempt_timestamp(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    timestamps = iter(
        (
            CREATED_AT,
            datetime(2026, 7, 11, 15, 30, 13, 50000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 100000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 200000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 300000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 400000, tzinfo=LOCAL_OFFSET),
        )
    )
    session = Session.create(
        state,
        now=timestamps.__next__,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    message_timestamp = next(timestamps)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Final state",
                "timestamp": message_timestamp.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    replacements: list[dict[str, Any]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append(json.loads(content.splitlines()[0]))
        if len(replacements) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", lambda _delay: None)

    session.persist()
    session.close()
    await asyncio.sleep(0)

    assert [header["updated_at"] for header in replacements] == [
        "2026-07-11T15:30:13.200+08:00",
        "2026-07-11T15:30:13.300+08:00",
        "2026-07-11T15:30:13.400+08:00",
    ]
    assert Session.load(state, session.session_id).messages == session.messages


def test_session_identity_fields_are_read_only(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(AttributeError):
        session.session_id = "another-session"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.created_at = CREATED_AT  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.updated_at = UPDATED_AT  # type: ignore[misc]


def test_commit_agent_run_deep_copies_messages_before_publishing_public_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    extension = {"nested": ["before"]}
    tool_calls = [{"id": "call-1", "name": "read_file", "arguments": '{"path":"README.md"}'}]
    usage = {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }

    session.commit_agent_run(
        [
            {
                "role": "user",
                "content": "Inspect this project.",
                "extension": extension,
            },
            {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": tool_calls,
                "status": "completed",
                "error": None,
                "token_usage": usage,
            },
        ],
        pending_last_compacted=0,
        pending_action_summary="",
    )

    extension["nested"].append("after")
    tool_calls[0]["arguments"] = "changed"
    usage["input_tokens"] = 99
    session.metadata["future_key"] = {"enabled": True}
    session.last_compacted = -1

    assert session.messages[0]["extension"] == {"nested": ["before"]}
    assert session.messages[1]["tool_calls"] == [
        {
            "id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]
    assert "id" not in session.messages[0]
    assert "id" not in session.messages[1]
    _assert_local_timestamp(session.messages[0]["timestamp"])
    _assert_local_timestamp(session.messages[1]["timestamp"])
    assert session.metadata["token_usage"] == {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    assert session.metadata["future_key"] == {"enabled": True}
    assert session.last_compacted == -1


def test_update_metadata_rejects_malformed_blackboard_without_mutating_state_or_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    patch = {
        "blackboard": {"goal": "", "completion_boundary": "Boundary"},
        "future": {"nested": ["value"]},
    }
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    original_patch = copy.deepcopy(patch)

    with pytest.raises(ValueError, match=r"metadata\.blackboard"):
        session.update_metadata(patch)

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert patch == original_patch


def test_commit_agent_run_leaves_state_unchanged_when_a_middle_message_is_invalid(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_usage = copy.deepcopy(session.metadata["token_usage"])

    with pytest.raises(ValueError, match="assistant message is missing"):
        session.commit_agent_run(
            [
                {"role": "user", "content": "Before the invalid record."},
                {"role": "assistant", "content": "Missing durable fields."},
                {"role": "user", "content": "After the invalid record."},
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )

    assert session.messages == before_messages
    assert session.metadata["token_usage"] == before_usage


def test_commit_agent_run_leaves_state_unchanged_when_the_final_message_is_invalid(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_usage = copy.deepcopy(session.metadata["token_usage"])

    with pytest.raises(ValueError, match="tool_call_id"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "A valid assistant record.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                },
                {
                    "role": "tool",
                    "content": "Invalid result.",
                    "name": "read_file",
                    "status": "error",
                },
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )

    assert session.messages == before_messages
    assert session.metadata["token_usage"] == before_usage


def test_commit_agent_run_isolated_from_nested_caller_mutations(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
            "provider_extension": {"trace": ["before"]},
        },
        {
            "role": "tool",
            "content": "README.md",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": {
                "path": ".myclaw/artifacts/session-1/call-1.txt",
                "total_chars": 123,
                "preview_chars": 80,
            },
            "confirmation": {"approved": True, "details": {"source": "user"}},
        },
    ]

    session.commit_agent_run(
        messages,
        pending_last_compacted=0,
        pending_action_summary=None,
    )

    messages[0]["tool_calls"][0]["arguments"] = '{"path":"changed"}'
    messages[0]["provider_extension"]["trace"].append("after")
    messages[0]["token_usage"]["input_tokens"] = 99
    messages[1]["artifact"]["total_chars"] = 999
    messages[1]["confirmation"]["details"]["source"] = "changed"

    assert session.messages[0]["tool_calls"] == [
        {"id": "call-1", "name": "read_file", "arguments": "{}"}
    ]
    assert session.messages[0]["provider_extension"] == {"trace": ["before"]}
    assert session.messages[0]["token_usage"]["input_tokens"] == 12
    assert session.messages[1]["artifact"]["total_chars"] == 123
    assert session.messages[1]["confirmation"] == {
        "approved": True,
        "details": {"source": "user"},
    }


@pytest.mark.parametrize(
    "usage",
    [
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 99,
        },
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
        },
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "cached_tokens": 0,
        },
    ],
)
def test_commit_agent_run_rejects_usage_shape_errors_without_state_changes(
    agent_home: Path,
    workspace: Path,
    usage: dict[str, int],
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError, match="token"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Invalid usage.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": usage,
                }
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata


def test_commit_agent_run_accumulates_multiple_assistants_once_beyond_64_bit_range(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    signed_64_max = 2**63 - 1
    session.update_metadata(
        token_usage_delta={
            "model_calls": 7,
            "input_tokens": signed_64_max - 2,
            "output_tokens": 1,
            "total_tokens": signed_64_max - 1,
        }
    )

    session.commit_agent_run(
        [
            {
                "role": "assistant",
                "content": "First response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            },
            {
                "role": "assistant",
                "content": "Second response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ],
        pending_last_compacted=0,
        pending_action_summary=None,
    )

    assert session.metadata["token_usage"] == {
        "model_calls": 9,
        "input_tokens": signed_64_max + 4,
        "output_tokens": 4,
        "total_tokens": signed_64_max + 8,
    }


def test_commit_agent_run_rejects_invalid_existing_usage_without_state_changes(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.metadata["token_usage"] = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": "invalid",
    }
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError, match=r"metadata\.token_usage"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "A valid message.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ],
            pending_last_compacted=0,
            pending_action_summary=None,
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata


def test_tool_message_preserves_provider_fields_and_unknown_extensions(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    session.commit_agent_run(
        [
            {
                "role": "tool",
                "content": "README.md",
                "tool_call_id": "call-1",
                "name": "read_file",
                "status": "success",
                "artifact": {
                    "path": ".myclaw/artifacts/session-1/call-1.txt",
                    "total_chars": 123,
                    "preview_chars": 80,
                },
                "provider_extension": {"trace": [1, 2]},
            }
        ],
        pending_last_compacted=0,
        pending_action_summary="",
    )

    message = session.messages[0]
    assert {key: value for key, value in message.items() if key != "timestamp"} == {
        "role": "tool",
        "content": "README.md",
        "tool_call_id": "call-1",
        "name": "read_file",
        "status": "success",
        "artifact": {
            "path": ".myclaw/artifacts/session-1/call-1.txt",
            "total_chars": 123,
            "preview_chars": 80,
        },
        "provider_extension": {"trace": [1, 2]},
    }
    _assert_local_timestamp(message["timestamp"])


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("\n  \u300c  Project\t review  \u300d\nIgnore this line", "Project review"),
        ("\n\t\t", "Untitled session"),
        ('"  Quoted title  "', "Quoted title"),
        ("\u201c  Curly quoted title  \u201d", "Curly quoted title"),
        ("\u300cone-sided title", "\u300cone-sided title"),
        ("\u754c" * 70, "\u754c" * 60),
    ],
)
def test_update_metadata_normalizes_title_shallow_merges_and_accumulates_usage(
    agent_home: Path,
    workspace: Path,
    candidate: str,
    expected: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    extra = {"nested": ["before"]}

    session.update_metadata(
        title=candidate,
        future=extra,
        token_usage_delta={
            "model_calls": 2,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    )
    extra["nested"].append("after")

    assert session.metadata == {
        "title": expected,
        "token_usage": {
            "model_calls": 2,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
        "summary": "",
        "future": {"nested": ["before"]},
    }


@pytest.mark.parametrize("value", [object(), {"bad": object()}])
def test_mutation_helpers_reject_non_json_values(
    agent_home: Path,
    workspace: Path,
    value: object,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises((TypeError, ValueError), match="JSON"):
        session.commit_agent_run(
            [{"role": "user", "content": "Hello", "extension": value}],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises((TypeError, ValueError), match="JSON"):
        session.update_metadata(future=value)


def test_known_message_contracts_and_unsupported_legacy_fields_are_validated(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(ValueError, match="role"):
        session.commit_agent_run(
            [{"role": "system", "content": "Unsupported"}],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="timestamp is reserved"):
        session.commit_agent_run(
            [{"role": "user", "content": "Hello", "timestamp": "override"}],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="unsupported"):
        session.commit_agent_run(
            [{"role": "user", "content": "Hello", "id": "legacy-message-id"}],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="status"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "tool_calls": [],
                    "status": "unknown",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="tool_call_id"):
        session.commit_agent_run(
            [{"role": "tool", "content": "Result", "name": "read_file", "status": "success"}],
            pending_last_compacted=0,
            pending_action_summary="",
        )


@pytest.mark.parametrize("missing", ["tool_calls", "status", "error", "token_usage"])
def test_assistant_message_requires_every_provider_relevant_field(
    agent_home: Path,
    workspace: Path,
    missing: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    fields: dict[str, Any] = {
        "tool_calls": [],
        "status": "completed",
        "error": None,
        "token_usage": {
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }
    del fields[missing]

    with pytest.raises(ValueError, match=missing):
        session.commit_agent_run(
            [{"role": "assistant", "content": "Answer", **fields}],
            pending_last_compacted=0,
            pending_action_summary="",
        )


def test_assistant_status_error_content_and_model_call_contract_remains_coherent(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    usage = {"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with pytest.raises(ValueError, match="completed assistant"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "tool_calls": [],
                    "status": "completed",
                    "error": {"code": "model_failed", "message": "failed"},
                    "token_usage": usage,
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="non-completed assistant"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Partial",
                    "tool_calls": [],
                    "status": "interrupted",
                    "error": None,
                    "token_usage": usage,
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="content or tool_calls"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [],
                    "status": "interrupted",
                    "error": {"code": "turn_cancelled", "message": "interrupted"},
                    "token_usage": usage,
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    with pytest.raises(ValueError, match="model_calls"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": dict(ZERO_USAGE),
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )
    session.commit_agent_run(
        [
            {
                "role": "assistant",
                "content": "Failed before a Provider result",
                "tool_calls": [],
                "status": "error",
                "error": {"code": "model_failed", "message": "failed"},
                "token_usage": dict(ZERO_USAGE),
            },
            {
                "role": "assistant",
                "content": "Iteration limit reached",
                "tool_calls": [],
                "status": "error",
                "error": {"code": "agent_iteration_limit", "message": "limit reached"},
                "token_usage": dict(ZERO_USAGE),
            },
        ],
        pending_last_compacted=0,
        pending_action_summary="",
    )

    assert [message["error"]["code"] for message in session.messages] == [
        "model_failed",
        "agent_iteration_limit",
    ]


def test_load_current_five_field_jsonl_preserves_json_native_extensions(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        last_compacted=2,
        metadata={
            "title": "Project review",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
            "future": {"enabled": True},
        },
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Inspect this project.",
            "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            "provider_extension": {"trace": [1, 2]},
        },
        {
            "role": "assistant",
            "content": "Done.",
            "timestamp": UPDATED_AT.isoformat(timespec="milliseconds"),
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        },
    ]
    _write_jsonl(state, [header, *messages])

    loaded = Session.load(state, SESSION_ID)

    assert loaded.session_id == SESSION_ID
    assert loaded.created_at == CREATED_AT
    assert loaded.updated_at == UPDATED_AT
    assert loaded.last_compacted == 2
    assert loaded.metadata == {**header["metadata"], "summary": ""}
    assert loaded.messages == messages


def test_update_metadata_rejects_a_non_string_action_summary(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    before = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError, match=r"metadata\.summary must be a string"):
        session.update_metadata(summary=123)

    assert session.metadata == before


def test_load_rejects_a_non_string_action_summary(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(
        state,
        [
            _header(
                metadata={
                    "title": "Project review",
                    "token_usage": dict(ZERO_USAGE),
                    "summary": 123,
                }
            )
        ],
    )

    with pytest.raises(ValueError, match=r"metadata\.summary must be a string"):
        Session.load(state, SESSION_ID)


def test_load_canonicalizes_a_valid_blackboard_metadata_value(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        metadata={
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
            "blackboard": {
                "goal": "  Review the project  ",
                "completion_boundary": "  Review is complete  ",
            },
        }
    )
    _write_jsonl(state, [header])

    loaded = Session.load(state, SESSION_ID)

    assert loaded.metadata["blackboard"] == {
        "goal": "Review the project",
        "completion_boundary": "Review is complete",
    }


@pytest.mark.parametrize(
    "blackboard",
    [
        None,
        "not an object",
        {"goal": "goal"},
        {"completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": "boundary", "extra": True},
        {"goal": 1, "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": False},
        {"goal": "", "completion_boundary": "boundary"},
        {"goal": "   ", "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": ""},
        {"goal": "goal", "completion_boundary": "\t\n"},
    ],
)
def test_load_treats_malformed_blackboard_metadata_as_absent(
    agent_home: Path,
    workspace: Path,
    blackboard: object,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        metadata={
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
            "blackboard": blackboard,
        }
    )
    _write_jsonl(state, [header])

    loaded = Session.load(state, SESSION_ID)

    assert "blackboard" not in loaded.metadata


@pytest.mark.parametrize(
    "records",
    [
        [_header(extra=True)],
        [_header(last_consolidated=0)],
        [_header(session_id=OTHER_SESSION_ID)],
        [_header(created_at="2026-07-11T15:30:12")],
        [_header(last_compacted=-1)],
        [_header(metadata={"title": "Project review", "token_usage": {}})],
        [
            _header(),
            {
                "role": "assistant",
                "content": "Missing known fields",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
    ],
)
def test_load_rejects_malformed_core_or_unsupported_message_shapes(
    agent_home: Path,
    workspace: Path,
    records: list[dict[str, Any]],
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, records)

    with pytest.raises(ValueError):
        Session.load(state, SESSION_ID)


def test_load_rejects_jsonl_without_a_trailing_newline(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, [_header()], trailing_newline=False)

    with pytest.raises(ValueError, match="newline"):
        Session.load(state, SESSION_ID)


def test_assistant_context_usage_is_optional_and_round_trips_through_jsonl(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    old_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "Old response.",
        "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
        "tool_calls": [],
        "status": "completed",
        "error": None,
        "token_usage": {"model_calls": 1, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
    }
    _write_jsonl(state, [_header(), old_assistant])

    old_loaded = Session.load(state, SESSION_ID)

    assert old_loaded.messages[0].get("context_usage") is None

    new_session = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    provenance = _context_usage()
    new_session.commit_agent_run(
        [
            {
                "role": "assistant",
                "content": "New response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                },
                "context_usage": provenance,
            }
        ],
        pending_last_compacted=0,
        pending_action_summary="",
    )
    new_session.close()

    loaded = Session.load(state, new_session.session_id)

    assert loaded.messages == new_session.messages
    assert loaded.messages[0]["context_usage"] == provenance


@pytest.mark.parametrize("role", ["user", "tool"])
def test_context_usage_is_rejected_for_non_assistant_messages_at_construction(
    agent_home: Path,
    workspace: Path,
    role: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    fields: dict[str, Any] = {"context_usage": _context_usage()}
    if role == "tool":
        fields.update(tool_call_id="call-1", name="read_file", status="success")

    with pytest.raises(ValueError, match=r"context_usage.*assistant"):
        session.commit_agent_run(
            [{"role": role, "content": "Message content.", **fields}],
            pending_last_compacted=0,
            pending_action_summary="",
        )


def test_context_usage_is_rejected_for_non_assistant_messages_at_load(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(
        state,
        [
            _header(),
            {
                "role": "user",
                "content": "Invalid provenance.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
                "context_usage": _context_usage(),
            },
        ],
    )

    with pytest.raises(ValueError, match=r"context_usage.*assistant"):
        Session.load(state, SESSION_ID)


@pytest.mark.parametrize(
    "context_usage",
    [
        {key: value for key, value in _context_usage().items() if key != "model"},
        {**_context_usage(), "reported_usage": dict(ZERO_USAGE)},
    ],
)
def test_assistant_context_usage_rejects_missing_or_unknown_fields_at_construction(
    agent_home: Path,
    workspace: Path,
    context_usage: dict[str, object],
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(ValueError, match=r"context_usage.*shape"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Invalid provenance.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                    "context_usage": context_usage,
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )

    assert session.messages == []
    assert session.metadata["token_usage"] == ZERO_USAGE


def test_context_usage_is_rejected_for_a_synthetic_zero_call_assistant(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(ValueError, match="exactly one assistant model call"):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Iteration limit reached.",
                    "tool_calls": [],
                    "status": "error",
                    "error": {
                        "code": "agent_iteration_limit",
                        "message": "Iteration limit reached.",
                    },
                    "token_usage": dict(ZERO_USAGE),
                    "context_usage": _context_usage(),
                }
            ],
            pending_last_compacted=0,
            pending_action_summary="",
        )

    assert session.messages == []
    assert session.metadata["token_usage"] == ZERO_USAGE


def test_commit_agent_run_publishes_messages_cursor_summary_usage_and_metadata_once(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Concurrent title",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
            "old_extension": {"remove": True},
            "concurrent_extension": {"version": 2},
        },
        last_compacted=0,
    )
    before_state = session.__dict__
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Current request."},
        {
            "role": "assistant",
            "content": "Completed response.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
            },
            "context_usage": _context_usage(),
        },
    ]

    metadata_updates: dict[str, Any] = {
        "blackboard": {
            "goal": "  Ship the change  ",
            "completion_boundary": "  Tests pass  ",
        },
        "task_framing_status": "resolved",
        "future_extension": {"trace": ["before"]},
    }
    session.commit_agent_run(
        messages,
        pending_last_compacted=2,
        pending_action_summary=None,
        usage_delta={
            "model_calls": 3,
            "input_tokens": 30,
            "output_tokens": 10,
            "total_tokens": 40,
        },
        metadata_updates=metadata_updates,
        metadata_removals=("old_extension", "unknown_extension"),
    )
    messages[0]["content"] = "Changed after commit."
    messages[1]["context_usage"]["model"] = "changed-after-commit"
    metadata_updates["blackboard"]["goal"] = "Changed after commit"
    metadata_updates["future_extension"]["trace"].append("after")

    assert session.__dict__ is not before_state
    assert [message["content"] for message in before_state["messages"]] == ["Existing history."]
    assert [message["content"] for message in session.messages] == [
        "Existing history.",
        "Current request.",
        "Completed response.",
    ]
    assert session.last_compacted == 2
    assert session.metadata["title"] == "Concurrent title"
    assert session.metadata["summary"] == ""
    assert session.metadata["blackboard"] == {
        "goal": "Ship the change",
        "completion_boundary": "Tests pass",
    }
    assert session.metadata["task_framing_status"] == "resolved"
    assert session.metadata["concurrent_extension"] == {"version": 2}
    assert session.metadata["future_extension"] == {"trace": ["before"]}
    assert "old_extension" not in session.metadata
    assert "unknown_extension" not in session.metadata
    assert session.metadata["token_usage"] == {
        "model_calls": 4,
        "input_tokens": 41,
        "output_tokens": 13,
        "total_tokens": 54,
    }
    assert session.messages[-1]["context_usage"] == _context_usage()
    assert "context_usage" not in session.metadata
    assert persist_calls == [None]


@pytest.mark.asyncio
async def test_commit_agent_run_persists_and_round_trips_the_complete_published_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Concurrent title",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
            "old_extension": {"remove": True},
            "preserved_extension": {"version": 2},
        },
        last_compacted=0,
    )

    session.commit_agent_run(
        [
            {"role": "user", "content": "Current request."},
            {
                "role": "assistant",
                "content": "Completed response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "total_tokens": 14,
                },
                "context_usage": _context_usage(),
            },
        ],
        pending_last_compacted=2,
        pending_action_summary="Action summary.",
        usage_delta={
            "model_calls": 3,
            "input_tokens": 30,
            "output_tokens": 10,
            "total_tokens": 40,
        },
        metadata_updates={
            "blackboard": {
                "goal": "  Ship the change  ",
                "completion_boundary": "  Tests pass  ",
            },
            "future_extension": {"trace": ["persisted"]},
        },
        metadata_removals=("old_extension",),
    )
    await session.wait_for_pending_persist()

    loaded = Session.load(state, session.session_id)
    header = json.loads(
        (state.sessions_directory / f"{session.session_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert loaded.messages == session.messages
    assert loaded.last_compacted == 2
    assert loaded.metadata == {
        "title": "Concurrent title",
        "token_usage": {
            "model_calls": 4,
            "input_tokens": 41,
            "output_tokens": 13,
            "total_tokens": 54,
        },
        "summary": "Action summary.",
        "preserved_extension": {"version": 2},
        "blackboard": {
            "goal": "Ship the change",
            "completion_boundary": "Tests pass",
        },
        "future_extension": {"trace": ["persisted"]},
    }
    assert "old_extension" not in loaded.metadata
    assert header["last_compacted"] == loaded.last_compacted
    assert header["metadata"] == loaded.metadata


@pytest.mark.parametrize(
    ("metadata_updates", "metadata_removals", "match"),
    [
        ({"future": True}, ("future",), "same key"),
        ({"title": "Stale run-start title"}, (), "title"),
        ({"token_usage": dict(ZERO_USAGE)}, (), "token usage"),
        ({"token_usage_delta": dict(ZERO_USAGE)}, (), "token usage"),
        ({"usage_delta": dict(ZERO_USAGE)}, (), "token usage"),
        ({}, ("title",), "required Session metadata"),
        ({}, ("token_usage",), "required Session metadata"),
        ({}, ("token_usage_delta",), "token usage"),
        ({}, ("usage_delta",), "token usage"),
        ({"summary": "patch"}, (), "Action Summary"),
        ({}, ("summary",), "Action Summary"),
    ],
)
def test_commit_agent_run_rejects_ambiguous_or_protected_metadata_patches(
    agent_home: Path,
    workspace: Path,
    metadata_updates: dict[str, Any],
    metadata_removals: tuple[str, ...],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    with pytest.raises(ValueError, match=match):
        session.commit_agent_run(
            [],
            pending_last_compacted=0,
            pending_action_summary="",
            metadata_updates=metadata_updates,
            metadata_removals=metadata_removals,
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert persist_calls == []


@pytest.mark.parametrize(
    ("metadata_updates", "usage_delta", "match"),
    [
        (
            {"blackboard": {"goal": "", "completion_boundary": "Boundary"}},
            None,
            "metadata.blackboard",
        ),
        (
            {"future": {"nested": ["value"]}},
            {"model_calls": True, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "usage_delta",
        ),
        (
            {"future": {"nested": ["value"]}},
            {"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 99},
            "usage_delta",
        ),
        (
            {"future": {"invalid": {1, 2}}},
            None,
            "metadata_updates",
        ),
    ],
)
def test_commit_agent_run_candidate_failure_preserves_state_and_caller_inputs(
    agent_home: Path,
    workspace: Path,
    metadata_updates: dict[str, Any],
    usage_delta: dict[str, Any] | None,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Candidate increment."}]
    original_messages = copy.deepcopy(messages)
    original_updates = copy.deepcopy(metadata_updates)
    original_usage = copy.deepcopy(usage_delta)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))
    before_state = session.__dict__
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    before_updated_at = session.updated_at

    with pytest.raises((TypeError, ValueError), match=match):
        session.commit_agent_run(
            messages,
            pending_last_compacted=1,
            pending_action_summary="Pending summary.",
            usage_delta=usage_delta,
            metadata_updates=metadata_updates,
        )

    assert session.__dict__ is before_state
    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert session.updated_at == before_updated_at
    assert messages == original_messages
    assert metadata_updates == original_updates
    assert usage_delta == original_usage
    assert persist_calls == []


@pytest.mark.parametrize(
    ("metadata_updates", "metadata_removals", "usage_delta", "match"),
    [
        ([], (), None, "metadata_updates"),
        ({1: "invalid"}, (), None, "metadata_updates"),
        ({"future": {"enabled": True}}, [], None, "metadata_removals"),
        ({"future": {"enabled": True}}, (1,), None, "metadata_removals"),
        ({"future": {"enabled": True}}, (), [], "usage_delta"),
    ],
)
def test_commit_agent_run_rejects_invalid_optional_argument_containers_without_changes(
    agent_home: Path,
    workspace: Path,
    metadata_updates: object,
    metadata_removals: object,
    usage_delta: object,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Candidate increment."}]
    original_messages = copy.deepcopy(messages)
    original_updates = copy.deepcopy(metadata_updates)
    original_removals = copy.deepcopy(metadata_removals)
    original_usage = copy.deepcopy(usage_delta)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))
    before_state = session.__dict__
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    before_updated_at = session.updated_at

    with pytest.raises((TypeError, ValueError), match=match):
        session.commit_agent_run(
            messages,
            pending_last_compacted=1,
            pending_action_summary="Pending summary.",
            metadata_updates=metadata_updates,  # type: ignore[arg-type]
            metadata_removals=metadata_removals,  # type: ignore[arg-type]
            usage_delta=usage_delta,  # type: ignore[arg-type]
        )

    assert session.__dict__ is before_state
    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert session.updated_at == before_updated_at
    assert messages == original_messages
    assert metadata_updates == original_updates
    assert metadata_removals == original_removals
    assert usage_delta == original_usage
    assert persist_calls == []


def test_commit_agent_run_rejects_string_metadata_removals_without_state_changes(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    before_metadata = copy.deepcopy(session.metadata)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    with pytest.raises(TypeError, match="metadata_removals must be a tuple"):
        session.commit_agent_run(
            [],
            pending_last_compacted=0,
            pending_action_summary="",
            metadata_removals="blackboard",  # type: ignore[arg-type]
        )

    assert session.messages == []
    assert session.metadata == before_metadata
    assert session.last_compacted == 0
    assert persist_calls == []


def test_commit_agent_run_validation_failure_leaves_every_observable_field_unchanged(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    before_updated_at = session.updated_at
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Prepared response.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
            "context_usage": {**_context_usage(), "context_window": 0},
        }
    ]
    original_messages = copy.deepcopy(messages)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    with pytest.raises(ValueError, match="context_window"):
        session.commit_agent_run(
            messages,
            pending_last_compacted=99,
            pending_action_summary="Pending summary.",
            usage_delta={
                "model_calls": 1,
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert session.updated_at == before_updated_at
    assert messages == original_messages
    assert persist_calls == []


@pytest.mark.parametrize(
    ("token_usage", "match"),
    [
        (
            {"model_calls": 2, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "model_calls",
        ),
        (
            {"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 99},
            "total_tokens",
        ),
    ],
)
def test_commit_agent_run_rejects_invalid_assistant_usage_without_state_changes(
    agent_home: Path,
    workspace: Path,
    token_usage: dict[str, int],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    with pytest.raises(ValueError, match=match):
        session.commit_agent_run(
            [
                {
                    "role": "assistant",
                    "content": "Invalid usage.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": token_usage,
                }
            ],
            pending_last_compacted=1,
            pending_action_summary="Pending summary.",
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert persist_calls == []


def test_commit_agent_run_state_replacement_failure_does_not_publish_or_persist(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    before_cursor = session.last_compacted
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    def fail_state_replacement(instance: Session, name: str, value: object) -> None:
        if instance is session and name == "__dict__":
            raise RuntimeError("state replacement failed")
        object.__setattr__(instance, name, value)

    monkeypatch.setattr(Session, "__setattr__", fail_state_replacement, raising=False)

    with pytest.raises(RuntimeError, match="state replacement failed"):
        session.commit_agent_run(
            [{"role": "user", "content": "Current request."}],
            pending_last_compacted=0,
            pending_action_summary="Pending summary.",
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == before_cursor
    assert persist_calls == []


def test_commit_agent_run_keeps_main_context_provenance_separate_from_independent_usage(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.update_metadata(title="Concurrent title")
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))
    provenance = _context_usage()
    main_messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Main response.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
            "context_usage": provenance,
        }
    ]

    session.commit_agent_run(
        main_messages,
        pending_last_compacted=1,
        pending_action_summary="Action summary.",
        metadata_updates={"blackboard": {"goal": "Goal", "completion_boundary": "Done"}},
    )
    session.commit_agent_run(
        [],
        pending_last_compacted=1,
        pending_action_summary="Action summary.",
        usage_delta={
            "model_calls": 2,
            "input_tokens": 13,
            "output_tokens": 5,
            "total_tokens": 18,
        },
        metadata_updates={"task_framing_status": "resolved"},
    )

    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 20,
        "output_tokens": 7,
        "total_tokens": 27,
    }
    assert session.metadata["title"] == "Concurrent title"
    assert session.metadata["task_framing_status"] == "resolved"
    assert session.metadata["blackboard"] == {"goal": "Goal", "completion_boundary": "Done"}
    assert session.messages[0]["context_usage"] == provenance
    assert "context_usage" not in session.metadata
    assert persist_calls == [None, None]


def test_commit_agent_run_rejects_a_cursor_past_the_final_message_count(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Existing history.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    with pytest.raises(ValueError, match="final message count"):
        session.commit_agent_run(
            [{"role": "user", "content": "Current request."}],
            pending_last_compacted=3,
            pending_action_summary="Pending summary.",
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.last_compacted == 0
    assert persist_calls == []


def test_commit_agent_run_accepts_a_cursor_into_base_messages_plus_increment(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "First existing message.",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
            {
                "role": "user",
                "content": "Second existing message.",
                "timestamp": UPDATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        last_compacted=0,
    )
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    session.commit_agent_run(
        [{"role": "user", "content": "Current increment."}],
        pending_last_compacted=3,
        pending_action_summary="Pending summary.",
    )

    assert session.last_compacted == 3
    assert len(session.messages) == 3
    assert persist_calls == [None]


def test_commit_agent_run_keeps_published_state_when_persist_clock_fails(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_values = iter((CREATED_AT, CREATED_AT))

    def fail_after_message_timestamp() -> datetime:
        try:
            return next(clock_values)
        except StopIteration as error:
            raise RuntimeError("persist clock failed") from error

    session = Session.create(
        _state(workspace, agent_home),
        now=fail_after_message_timestamp,
    )

    with pytest.raises(RuntimeError, match="persist clock failed"):
        session.commit_agent_run(
            [{"role": "user", "content": "Committed before persist scheduling."}],
            pending_last_compacted=1,
            pending_action_summary="Committed Action Summary.",
        )

    assert [message["content"] for message in session.messages] == [
        "Committed before persist scheduling."
    ]
    assert session.last_compacted == 1
    assert session.metadata["summary"] == "Committed Action Summary."
    assert session.updated_at == CREATED_AT


def test_commit_agent_run_clears_action_summary_and_removes_staged_metadata(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.update_metadata(
        summary="Existing Action Summary.",
        blackboard={"goal": "Goal", "completion_boundary": "Done"},
        task_framing_status="resolved",
        future_extension={"remove": True},
    )
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    session.commit_agent_run(
        [],
        pending_last_compacted=0,
        pending_action_summary=None,
        metadata_removals=(
            "blackboard",
            "task_framing_status",
            "future_extension",
            "missing_extension",
        ),
    )

    assert session.metadata["summary"] == ""
    assert "blackboard" not in session.metadata
    assert "task_framing_status" not in session.metadata
    assert "future_extension" not in session.metadata
    assert persist_calls == [None]


def test_commit_agent_run_does_not_modify_conversation_summary_or_fact_state(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    summary_path = state.memory_directory / "summary.jsonl"
    existing_summary = b'{"index":1,"content":"Existing fact"}\n'
    summary_path.write_bytes(existing_summary)
    persist_calls: list[None] = []
    monkeypatch.setattr(session, "persist", lambda: persist_calls.append(None))

    session.commit_agent_run(
        [{"role": "user", "content": "Session-only increment."}],
        pending_last_compacted=0,
        pending_action_summary=None,
    )

    assert summary_path.read_bytes() == existing_summary
    assert persist_calls == [None]


def _commit_restore_user(
    session: Session,
    content: str,
    run_token: UUID,
) -> None:
    restore_before = session.capture_restore_before()
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
        restore_before=restore_before,
        restore_run_token=run_token,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_index", "expected_contents"),
    [
        (0, ()),
        (1, ("first",)),
        (2, ("first", "second")),
    ],
)
async def test_restore_truncates_by_anchor_id_and_never_reuses_ids_after_restart(
    agent_home: Path,
    workspace: Path,
    selected_index: int,
    expected_contents: tuple[str, ...],
) -> None:
    now_value = [CREATED_AT]
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: now_value[0])
    for index, content in enumerate(("first", "second", "third")):
        _commit_restore_user(session, content, RESTORE_TOKENS[index])
    await session.wait_for_pending_persist()

    candidates = session.restore_candidates()
    assert [candidate.anchor_id for candidate in candidates] == [1, 2, 3]
    assert [candidate.content for candidate in candidates] == ["first", "second", "third"]

    now_value[0] = UPDATED_AT
    result = session.restore_before_durably(candidates[selected_index].anchor_id)

    assert result.session_id == session.session_id
    assert result.anchor_id == selected_index + 1
    assert [message["content"] for message in session.messages] == list(expected_contents)
    assert session.updated_at == UPDATED_AT
    assert (state.sessions_directory / f"{session.session_id}.jsonl").exists()
    loaded = Session.load(state, session.session_id)
    assert loaded.session_id == session.session_id
    assert loaded.messages == session.messages
    assert loaded.updated_at == UPDATED_AT

    if selected_index == 0:
        with pytest.raises(ValueError, match="Session-owned"):
            loaded.update_metadata(restore_next_anchor_id=1)

    _commit_restore_user(loaded, "new branch", RESTORE_TOKENS[3])
    await loaded.wait_for_pending_persist()
    assert loaded.restore_candidates()[-1].anchor_id == 4


@pytest.mark.asyncio
async def test_restore_uses_anchor_id_for_repeated_input_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    for index in range(3):
        _commit_restore_user(session, "repeat", RESTORE_TOKENS[index])
    await session.wait_for_pending_persist()

    session.restore_before_durably(2)

    assert [message["content"] for message in session.messages] == ["repeat"]
    assert [candidate.anchor_id for candidate in session.restore_candidates()] == [1]


@pytest.mark.asyncio
async def test_restore_restores_complete_session_state_before_selected_user(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    session.update_metadata(
        title="Before first",
        summary="First summary",
        blackboard={"goal": "First goal", "completion_boundary": "First boundary"},
        usage_delta={"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )
    _commit_restore_user(session, "first", RESTORE_TOKENS[0])
    session.update_metadata(
        title="Before second",
        summary="Second summary",
        blackboard={"goal": "Second goal", "completion_boundary": "Second boundary"},
        usage_delta={"model_calls": 1, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
    )
    session.last_compacted = 1
    _commit_restore_user(session, "second", RESTORE_TOKENS[1])
    session.update_metadata(
        title="Current title",
        summary="Current summary",
        blackboard={"goal": "Current goal", "completion_boundary": "Current boundary"},
    )
    await session.wait_for_pending_persist()

    session.restore_before_durably(2)

    assert [message["content"] for message in session.messages] == ["first"]
    assert session.last_compacted == 1
    assert session.metadata["title"] == "Before second"
    assert session.metadata["summary"] == "Second summary"
    assert session.metadata["blackboard"] == {
        "goal": "Second goal",
        "completion_boundary": "Second boundary",
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }
    assert session.metadata["restore_next_anchor_id"] == 3


@pytest.mark.asyncio
async def test_restore_strict_write_failure_does_not_publish_truncated_state(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    _commit_restore_user(session, "first", RESTORE_TOKENS[0])
    _commit_restore_user(session, "second", RESTORE_TOKENS[1])
    await session.wait_for_pending_persist()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    before_bytes = path.read_bytes()
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    def fail_write(_content: bytes) -> None:
        raise OSError("strict restore write failed")

    monkeypatch.setattr(session, "_write_content", fail_write)
    with pytest.raises(OSError, match="strict restore write failed"):
        session.restore_before_durably(1)

    assert path.read_bytes() == before_bytes
    assert session.messages == before_messages
    assert session.metadata == before_metadata


@pytest.mark.asyncio
async def test_restore_anchor_fields_are_durable_but_hidden_from_projections(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    _commit_restore_user(session, "private anchor fields", RESTORE_TOKENS[0])
    await session.wait_for_pending_persist()

    loaded = Session.load(state, session.session_id)
    persisted_user = loaded.messages[0]
    assert persisted_user["restore_anchor_id"] == 1
    assert persisted_user["restore_run_token"] == str(RESTORE_TOKENS[0])
    assert persisted_user["restore_before"] == {
        "metadata": {
            "title": "Untitled session",
            "token_usage": dict(ZERO_USAGE),
            "summary": "",
        },
        "last_compacted": 0,
    }


@pytest.mark.asyncio
async def test_restore_excludes_user_until_snapshot_is_persisted(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    started = asyncio.Event()
    release = asyncio.Event()
    original_persist_after = session._persist_after

    async def blocked_persist(previous: asyncio.Task[None] | None, content: bytes) -> None:
        started.set()
        await release.wait()
        await original_persist_after(previous, content)

    monkeypatch.setattr(session, "_persist_after", blocked_persist)
    _commit_restore_user(session, "not yet persisted", RESTORE_TOKENS[0])
    await started.wait()

    assert session.restore_candidates() == ()
    with pytest.raises(RuntimeError, match="Pending Session snapshots"):
        session.restore_before_durably(1)

    release.set()
    await session.wait_for_pending_persist()
    assert [anchor.anchor_id for anchor in session.restore_candidates()] == [1]


@pytest.mark.asyncio
async def test_restore_waits_for_older_snapshot_before_strict_truncation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    _commit_restore_user(session, "first", RESTORE_TOKENS[0])
    _commit_restore_user(session, "second", RESTORE_TOKENS[1])
    await session.wait_for_pending_persist()

    started = asyncio.Event()
    release = asyncio.Event()
    original_persist_after = session._persist_after

    async def blocked_persist(previous: asyncio.Task[None] | None, content: bytes) -> None:
        started.set()
        await release.wait()
        await original_persist_after(previous, content)

    monkeypatch.setattr(session, "_persist_after", blocked_persist)
    session.update_metadata(title="Old snapshot title")
    session.persist()
    await started.wait()

    with pytest.raises(RuntimeError, match="Pending Session snapshots"):
        session.restore_before_durably(2)

    release.set()
    await session.wait_for_pending_persist()
    session.restore_before_durably(2)
    loaded = Session.load(state, session.session_id)
    assert [message["content"] for message in loaded.messages] == ["first"]
    assert loaded.metadata["title"] == "Untitled session"


@pytest.mark.asyncio
async def test_schedule_session_rejects_restore_anchors_on_load_and_access(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    foreground = Session.create(state, now=lambda: CREATED_AT)
    _commit_restore_user(foreground, "foreground", RESTORE_TOKENS[0])
    await foreground.wait_for_pending_persist()

    schedule = Session.create_schedule(state, SCHEDULE_JOB_ID, now=lambda: CREATED_AT)
    with pytest.raises(ValueError, match="foreground Sessions"):
        schedule.capture_restore_before()
    with pytest.raises(ValueError, match="foreground Sessions"):
        schedule.restore_candidates()
    with pytest.raises(ValueError, match="foreground Sessions"):
        schedule.restore_before_durably(1)

    seed_session_state(
        schedule,
        messages=[copy.deepcopy(foreground.messages[0])],
        metadata=copy.deepcopy(schedule.metadata),
        last_compacted=0,
    )
    schedule.close()
    with pytest.raises(ValueError, match="foreground Sessions"):
        Session.load(state, schedule.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("duplicate", "must increase"),
        ("out_of_order", "must increase"),
        ("counter_reused", "must exceed"),
        ("counter_missing", "counter is missing"),
    ],
)
async def test_load_rejects_invalid_restore_anchor_sequence(
    agent_home: Path,
    workspace: Path,
    corruption: str,
    error: str,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=lambda: RESTORE_TOKENS[0],
    )
    _commit_restore_user(session, "first", RESTORE_TOKENS[0])
    _commit_restore_user(session, "second", RESTORE_TOKENS[1])
    await session.wait_for_pending_persist()
    records = [
        json.loads(line)
        for line in (state.sessions_directory / f"{session.session_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    if corruption == "duplicate":
        records[2]["restore_anchor_id"] = 1
    elif corruption == "out_of_order":
        records[1]["restore_anchor_id"] = 2
        records[2]["restore_anchor_id"] = 1
    elif corruption == "counter_reused":
        records[0]["metadata"]["restore_next_anchor_id"] = 2
    else:
        del records[0]["metadata"]["restore_next_anchor_id"]
    _write_jsonl(state, records)

    with pytest.raises(ValueError, match=error):
        Session.load(state, session.session_id)
