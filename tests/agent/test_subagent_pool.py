from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aide.agent.session.session import Session
from aide.agent.subagents.context import SubAgentToolContext
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentEvent,
    SubAgentExecutionResult,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.store import SubAgentRecordStore, SubAgentStoreError
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.subagents import build_subagent_tools
from aide.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


def _workspace(tmp_path: Path) -> tuple[WorkspaceState, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session_id = Session.create(state, now=lambda: _NOW).session_id
    return state, session_id


def _snapshot() -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="test-provider",
        model="test-model",
        reasoning_effort="mid",
        permission_level="workspace-write",
        shell="pwsh",
        tool_schemas=({"name": "read_file"},),
        system_prompt="You are Aide.",
    )


def _record(repository: SubAgentRecordStore, agent_id: str) -> SubAgentRecord:
    record = repository.get(agent_id)
    assert record is not None
    return record


class _ControlledExecutor:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release: dict[str, asyncio.Event] = {}
        self.cancel_requested: dict[str, asyncio.Event] = {}
        self.failure_ids: set[str] = set()
        self.started_ids: list[str] = []

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult:
        del emit
        self.started_ids.append(record.agent_id)
        await self.started.put(record.agent_id)
        await self.release[record.agent_id].wait()
        if record.agent_id in self.failure_ids:
            raise RuntimeError("controlled execution failure")
        cancelled = self.cancel_requested[record.agent_id].is_set()
        return SubAgentExecutionResult(
            status=SubAgentStatus.CANCELLED if cancelled else SubAgentStatus.COMPLETED,
            conversation=({"role": "assistant", "content": "done"},),
            context_state={},
            artifact_paths=(),
            result=f"done: {record.title}",
            error=(
                SubAgentError(code="cancelled", message="Cancelled by test.") if cancelled else None
            ),
            usage={"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        del interrupted
        requested = self.cancel_requested.get(agent_id)
        if requested is None:
            return False
        requested.set()
        return True


@pytest.mark.asyncio
async def test_each_session_starts_at_most_eight_and_admits_queued_tasks_fifo(
    tmp_path: Path,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )

    gateway = ToolGateway(
        workspace=state.workspace_path,
        additional_tools=build_subagent_tools(),
        tool_context=ToolRunContext(
            workspace=state.workspace_path,
            subagent=SubAgentToolContext(
                coordinator=pool,
                parent_run_id=_RUN_ID,
                source=source,
                creator_snapshot=_snapshot(),
            ),
        ),
    )

    records: list[SubAgentRecord] = []
    for index in range(10):
        spawned = await gateway.call(
            ModelToolCall(
                id=f"spawn_{index}",
                name="spawn_agent",
                arguments=json.dumps({"title": f"Task {index}", "task": f"Do task {index}"}),
            )
        )
        assert spawned.status == "success"
        payload = json.loads(spawned.content)
        assert payload["status"] == "queued"
        record = _record(repository, payload["agent_id"])
        assert record.status is SubAgentStatus.QUEUED
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()

    persisted = [repository.get(record.agent_id) for record in records]
    assert all(record is not None for record in persisted)
    assert [record.status for record in persisted if record is not None] == [
        SubAgentStatus.QUEUED
    ] * 10

    started = [await asyncio.wait_for(executor.started.get(), timeout=2) for _ in range(8)]
    assert started == [record.agent_id for record in records[:8]]
    assert executor.started.empty()

    executor.release[records[0].agent_id].set()
    ninth_started = await asyncio.wait_for(executor.started.get(), timeout=2)
    assert ninth_started == records[8].agent_id

    executor.release[records[1].agent_id].set()
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == records[9].agent_id
    assert executor.started.empty()

    for gate in executor.release.values():
        gate.set()
    await asyncio.gather(*(pool.wait([record.agent_id]) for record in records))


@pytest.mark.asyncio
async def test_session_pools_have_independent_capacity(tmp_path: Path) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, first_session_id = _workspace(tmp_path)
    second_session_id = Session.create(state, now=lambda: _NOW).session_id
    first_repository = SubAgentRecordStore(state, first_session_id, now=lambda: _NOW)
    second_repository = SubAgentRecordStore(state, second_session_id, now=lambda: _NOW)
    first_executor = _ControlledExecutor()
    second_executor = _ControlledExecutor()
    first_pool = SubAgentPool(first_repository, first_executor)
    second_pool = SubAgentPool(second_repository, second_executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )

    def submit(pool: SubAgentPool, executor: _ControlledExecutor, prefix: str) -> list[str]:
        ids = []
        for index in range(8):
            record = pool.submit(
                title=f"{prefix} {index}",
                task=f"Do {prefix} {index}",
                parent_run_id=_RUN_ID,
                source=source,
                creator_snapshot=_snapshot(),
            )
            ids.append(record.agent_id)
            executor.release[record.agent_id] = asyncio.Event()
            executor.cancel_requested[record.agent_id] = asyncio.Event()
        return ids

    first_ids = submit(first_pool, first_executor, "First")
    second_ids = submit(second_pool, second_executor, "Second")
    first_started, second_started = await asyncio.gather(
        asyncio.gather(*(first_executor.started.get() for _ in first_ids)),
        asyncio.gather(*(second_executor.started.get() for _ in second_ids)),
    )

    assert first_started == first_ids
    assert second_started == second_ids
    for gate in (*first_executor.release.values(), *second_executor.release.values()):
        gate.set()
    await asyncio.gather(
        *(first_pool.wait([agent_id]) for agent_id in first_ids),
        *(second_pool.wait([agent_id]) for agent_id in second_ids),
    )


@pytest.mark.asyncio
async def test_queued_and_running_cancellation_release_capacity_only_after_cleanup(
    tmp_path: Path,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )
    records = []
    for index in range(10):
        record = pool.submit(
            title=f"Task {index}",
            task=f"Do task {index}",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()
    for _ in range(8):
        await asyncio.wait_for(executor.started.get(), timeout=2)

    queued_id = records[8].agent_id
    running_id = records[0].agent_id
    assert pool.cancel(queued_id)
    assert not pool.cancel(queued_id)
    assert pool.cancel(running_id)
    assert pool.cancel(running_id)
    assert executor.started.empty()
    assert _record(repository, running_id).status is SubAgentStatus.RUNNING

    executor.release[running_id].set()
    next_started = await asyncio.wait_for(executor.started.get(), timeout=2)
    assert next_started == records[9].agent_id
    assert executor.started_ids == [record.agent_id for record in records[:8]] + [
        records[9].agent_id
    ]

    for gate in executor.release.values():
        gate.set()
    await asyncio.gather(*(pool.wait([record.agent_id]) for record in records))
    assert _record(repository, queued_id).status is SubAgentStatus.CANCELLED
    assert _record(repository, running_id).status is SubAgentStatus.CANCELLED


@pytest.mark.asyncio
async def test_failed_execution_releases_one_slot_and_preserves_its_error(
    tmp_path: Path,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )
    records = []
    for index in range(9):
        record = pool.submit(
            title=f"Task {index}",
            task=f"Do task {index}",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()
    for _ in range(8):
        await asyncio.wait_for(executor.started.get(), timeout=2)

    failed_id = records[0].agent_id
    executor.failure_ids.add(failed_id)
    executor.release[failed_id].set()
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == records[8].agent_id
    failed_result = await pool.wait([failed_id])
    assert failed_result[0].status is SubAgentStatus.FAILED
    assert failed_result[0].error is not None
    assert failed_result[0].error.code == "execution_failed"
    assert not pool.cancel(failed_id)

    for gate in executor.release.values():
        gate.set()
    await asyncio.gather(*(pool.wait([record.agent_id]) for record in records))


@pytest.mark.asyncio
async def test_wait_is_non_consuming_session_scoped_and_detached_from_execution(
    tmp_path: Path,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool
    from aide.agent.subagents.store import SubAgentRequestError

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )
    records = []
    for index in range(3):
        record = pool.submit(
            title=f"Task {index}",
            task=f"Do task {index}",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()
    for _ in records:
        await asyncio.wait_for(executor.started.get(), timeout=2)

    assert await pool.wait([records[2].agent_id], timeout_ms=0) == ()
    assert await pool.wait([records[2].agent_id], timeout_ms=5) == ()
    assert _record(repository, records[2].agent_id).status is SubAgentStatus.RUNNING

    pending = asyncio.create_task(pool.wait([record.agent_id for record in records]))
    turn = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(turn.set_result, None)
    await turn
    executor.release[records[1].agent_id].set()
    executor.release[records[0].agent_id].set()
    results = await pending
    assert [result.agent_id for result in results] == [
        records[0].agent_id,
        records[1].agent_id,
    ]
    assert len(await pool.wait([record.agent_id for record in records], timeout_ms=0)) == 2

    other_session_id = Session.create(state, now=lambda: _NOW).session_id
    other_repository = SubAgentRecordStore(state, other_session_id, now=lambda: _NOW)
    foreign = other_repository.register(
        title="Foreign",
        task="Not visible here",
        parent_run_id=_RUN_ID,
        source=source,
        creator_snapshot=_snapshot(),
    )
    with pytest.raises(SubAgentRequestError):
        await pool.wait([records[0].agent_id, foreign.agent_id], timeout_ms=0)

    pending = asyncio.create_task(pool.wait([records[2].agent_id]))
    turn = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(turn.set_result, None)
    await turn
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert _record(repository, records[2].agent_id).status is SubAgentStatus.RUNNING

    executor.release[records[2].agent_id].set()
    final = await pool.wait([records[2].agent_id])
    assert final[0].status is SubAgentStatus.COMPLETED
    assert final == await pool.wait([records[2].agent_id])


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_status", ["running", "completed", "failed"])
async def test_checkpoint_write_failure_wakes_waiters_without_fabricating_a_result(
    tmp_path: Path,
    failed_status: str,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    failed_write = asyncio.Event()

    def replace_text(path: Path, content: str) -> None:
        if json.loads(content)["status"] == failed_status and not failed_write.is_set():
            failed_write.set()
            raise OSError("simulated checkpoint write failure")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    repository = SubAgentRecordStore(state, session_id, replace_text=replace_text)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    record = pool.submit(
        title="Checkpoint failure",
        task="Retain the last durable checkpoint.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    executor.release[record.agent_id] = asyncio.Event()
    executor.cancel_requested[record.agent_id] = asyncio.Event()
    pending = asyncio.create_task(pool.wait([record.agent_id]))
    if failed_status != "running":
        assert await asyncio.wait_for(executor.started.get(), timeout=2) == record.agent_id
        if failed_status == "failed":
            executor.failure_ids.add(record.agent_id)
        executor.release[record.agent_id].set()
    await asyncio.wait_for(failed_write.wait(), timeout=2)
    with pytest.raises(SubAgentStoreError, match="could not be saved reliably"):
        await asyncio.wait_for(pending, timeout=2)
    saved = _record(repository, record.agent_id)
    assert saved.status is (
        SubAgentStatus.QUEUED if failed_status == "running" else SubAgentStatus.RUNNING
    )
    assert saved.result is None
    assert executor.started_ids == ([] if failed_status == "running" else [record.agent_id])
    with pytest.raises(SubAgentStoreError):
        await pool.wait([record.agent_id], timeout_ms=0)
    assert pool.cancel(record.agent_id)
    assert not pool.cancel(record.agent_id)
    cancelled = await pool.wait([record.agent_id], timeout_ms=0)
    assert cancelled[0].status is SubAgentStatus.CANCELLED
    assert cancelled == await pool.wait([record.agent_id])


@pytest.mark.asyncio
async def test_admission_can_be_fenced_and_reopened_around_restore(tmp_path: Path) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool
    from aide.agent.subagents.store import SubAgentRequestError

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )

    pool.close_admission()
    with pytest.raises(SubAgentRequestError, match="not accepting"):
        pool.submit(
            title="Rejected during restore",
            task="Do not pass the restore fence.",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )

    pool.open_admission()
    record = pool.submit(
        title="Accepted after restore cancellation",
        task="The admission fence was released.",
        parent_run_id=_RUN_ID,
        source=source,
        creator_snapshot=_snapshot(),
    )
    executor.release[record.agent_id] = asyncio.Event()
    executor.cancel_requested[record.agent_id] = asyncio.Event()
    executor.release[record.agent_id].set()

    assert (await asyncio.wait_for(pool.wait([record.agent_id]), timeout=2))[0].status is (
        SubAgentStatus.COMPLETED
    )


@pytest.mark.asyncio
async def test_cancel_and_wait_returns_after_runner_cleanup(tmp_path: Path) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    record = pool.submit(
        title="Controlled cancellation",
        task="Remain active until cancellation cleanup is released.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    executor.release[record.agent_id] = asyncio.Event()
    executor.cancel_requested[record.agent_id] = asyncio.Event()
    assert await asyncio.wait_for(executor.started.get(), timeout=2) == record.agent_id

    cancellation = asyncio.create_task(pool.cancel_and_wait(record.agent_id))
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not cancellation.done()

    executor.release[record.agent_id].set()
    cancelled = await asyncio.wait_for(cancellation, timeout=2)

    assert cancelled is not None
    assert cancelled.status is SubAgentStatus.CANCELLED
    assert cancelled == _record(repository, record.agent_id)


@pytest.mark.asyncio
async def test_shutdown_drains_running_and_queued_tasks_as_interrupted(tmp_path: Path) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool
    from aide.agent.subagents.store import SubAgentRequestError

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    source = SubAgentSource(
        kind=SubAgentSourceKind.FOREGROUND,
        restore_run_token=_RESTORE_TOKEN,
    )
    records = []
    for index in range(9):
        record = pool.submit(
            title=f"Shutdown {index}",
            task=f"Do shutdown task {index}.",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()
    for _ in range(8):
        await asyncio.wait_for(executor.started.get(), timeout=2)

    shutdown = asyncio.create_task(pool.shutdown())
    for record in records[:8]:
        await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not shutdown.done()
    assert executor.started.empty()

    for record in records[:8]:
        executor.release[record.agent_id].set()
    await asyncio.wait_for(shutdown, timeout=2)

    assert executor.started_ids == [record.agent_id for record in records[:8]]
    assert all(
        _record(repository, record.agent_id).status is SubAgentStatus.INTERRUPTED
        for record in records
    )
    with pytest.raises(SubAgentRequestError, match="not accepting"):
        pool.submit(
            title="After shutdown",
            task="Must be rejected.",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )


@pytest.mark.asyncio
async def test_job_cancellation_covers_occurrences_without_cancelling_other_sources(
    tmp_path: Path,
) -> None:
    from aide.agent.subagents.coordinator import SubAgentPool

    state, session_id = _workspace(tmp_path)
    repository = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    executor = _ControlledExecutor()
    pool = SubAgentPool(repository, executor)
    job_id = "123e4567-e89b-42d3-a456-426614174002"

    def schedule_source(occurrence: str) -> SubAgentSource:
        return SubAgentSource(
            kind=SubAgentSourceKind.SCHEDULE,
            job_id=job_id,
            occurrence_id=occurrence,
        )

    sources = (
        schedule_source("one"),
        schedule_source("two"),
        SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
    )
    records = []
    for index, source in enumerate(sources):
        record = pool.submit(
            title=f"Job task {index}",
            task=f"Do job task {index}.",
            parent_run_id=_RUN_ID,
            source=source,
            creator_snapshot=_snapshot(),
        )
        records.append(record)
        executor.release[record.agent_id] = asyncio.Event()
        executor.cancel_requested[record.agent_id] = asyncio.Event()
    for record in records:
        assert await asyncio.wait_for(executor.started.get(), timeout=2) == record.agent_id

    removal = asyncio.create_task(pool.cancel_source_and_wait(job_id=job_id))
    for record in records[:2]:
        await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not executor.cancel_requested[records[2].agent_id].is_set()
    for record in records[:2]:
        executor.release[record.agent_id].set()
    await asyncio.wait_for(removal, timeout=2)

    assert all(
        _record(repository, record.agent_id).status is SubAgentStatus.CANCELLED
        for record in records[:2]
    )
    assert _record(repository, records[2].agent_id).status is SubAgentStatus.RUNNING
    executor.release[records[2].agent_id].set()
    assert (await asyncio.wait_for(pool.wait([records[2].agent_id]), timeout=2))[0].status is (
        SubAgentStatus.COMPLETED
    )
