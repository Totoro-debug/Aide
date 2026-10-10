from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aide.agent.memory.dream import Dream, DreamResult
from aide.agent.memory.manager import MemoryManager
from aide.agent.workspace_state import WorkspaceState
from aide.logging.session import session_log
from aide.provider.model_router import ModelRouteStatus
from aide.provider.models import AssistantModelMessage, ModelResponse, ModelUsage
from aide.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from aide.schedule.service import ScheduleService
from aide.schedule.store import ScheduleStateError, WorkspaceScheduleStore
from aide.utils.errors import ErrorInfo
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import ScriptedFakeProvider, ScriptedFakeRouter

_FAKE_MEMORY_ROUTE_STATUS = ModelRouteStatus(
    requested_route="memory",
    selected_route="memory",
    provider_id="test-provider",
    model="test-model",
    context_window=200_000,
    max_output=8_192,
    used_fallback=False,
)


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


class _AdvancingClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.current = datetime(2026, 8, 7, 12, 0, tzinfo=UTC) if start is None else start
        self.elapsed = 0.0
        self.wait_started = asyncio.Event()
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.wait_started.set()
        future = asyncio.get_running_loop().create_future()
        self._waiters.append((self.elapsed + seconds, future))
        await future

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        self.elapsed += seconds
        for deadline, future in tuple(self._waiters):
            if deadline <= self.elapsed and not future.done():
                future.set_result(None)
                self._waiters.remove((deadline, future))


class _BlockingReservation:
    def __init__(self, store: WorkspaceScheduleStore) -> None:
        self._reserve_due = store.reserve_due
        self.completed = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def reserve_due(
        self,
        candidates: tuple[ScheduleJob, ...],
    ) -> tuple[ScheduleJob, ...]:
        self.calls += 1
        if self.calls == 1:
            reserved = await self._reserve_due(candidates)
            self.completed.set()
            await self.release.wait()
            return reserved
        return await self._reserve_due(candidates)


class _BlockingOperation:
    def __init__(self, operation: Callable[..., Awaitable[object]]) -> None:
        self._operation = operation
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancellation_count = 0

    async def run(self, *args: object, **kwargs: object) -> object:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancellation_count += 1
            self.cancelled.set()
            raise
        return await self._operation(*args, **kwargs)


class _DrainObserver:
    def __init__(self, drain: Callable[..., Awaitable[None]]) -> None:
        self._drain = drain
        self.started = asyncio.Event()

    async def drain(self, *args: object, **kwargs: object) -> None:
        self.started.set()
        await self._drain(*args, **kwargs)


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _dream(state: WorkspaceState, provider: ScriptedFakeProvider) -> Dream:
    return Dream(
        memory_manager=MemoryManager(state),
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )


def _memory_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def _service(state: WorkspaceState) -> ScheduleService:
    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> object:
        raise AssertionError("Dream must not run during registration")

    return ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )


@pytest.mark.asyncio
async def test_dream_registration_persists_a_hidden_recurring_system_job(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)

    service = _service(state)

    registered = await service.register_dream_job(
        schedule=JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    )

    assert registered.job_id == "dream"
    assert registered.source == "system"
    assert registered.message
    assert registered.title == "Dream"
    assert registered.schedule == JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    assert await service.public_snapshot() == ()
    assert await WorkspaceScheduleStore(state).snapshot() == (registered,)


@pytest.mark.asyncio
async def test_store_public_mutations_reject_the_dream_system_identity(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    service = _service(state)
    store = service._store
    await service.register_dream_job(schedule=JobSchedule.every(60))

    with pytest.raises(ValueError, match="canonical UUID4"):
        await store.commit_terminal(
            "dream",
            finished_at_ms=1,
            status="ok",
        )
    with pytest.raises(ValueError, match="canonical UUID4"):
        await store.remove_user_job("dream")


@pytest.mark.asyncio
async def test_exact_dream_registration_performs_zero_store_writes(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    schedule = JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Preserve this internal message.",
        schedule=schedule,
        created_at_ms=1,
        updated_at_ms=1,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    writes: list[tuple[Path, str]] = []
    original = HOST_FILESYSTEM.atomic_replace_text

    def record_write(path: Path, content: str) -> None:
        writes.append((path, content))
        original(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_write)
    service = _service(state)

    registered = await service.register_dream_job(schedule=schedule)

    assert registered == existing
    assert writes == []
    assert (await WorkspaceScheduleStore(state).snapshot()) == (existing,)


@pytest.mark.asyncio
async def test_dream_registration_reconciles_cron_without_losing_job_history(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Keep the persisted placeholder.",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        state=ScheduleJobState(
            last_finished_at_ms=10,
            last_status="ok",
        ),
        created_at_ms=1,
        updated_at_ms=10,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    writes: list[tuple[Path, str]] = []
    original = HOST_FILESYSTEM.atomic_replace_text

    def record_write(path: Path, content: str) -> None:
        writes.append((path, content))
        original(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_write)
    service = _service(state)
    replacement_schedule = JobSchedule.cron("30 * * * *", "UTC")

    registered = await service.register_dream_job(schedule=replacement_schedule)

    assert registered.job_id == existing.job_id
    assert registered.source == existing.source
    assert registered.message == existing.message
    assert registered.created_at_ms == existing.created_at_ms
    assert registered.state == existing.state
    assert registered.updated_at_ms >= existing.updated_at_ms
    assert registered.schedule == replacement_schedule
    assert len(writes) == 1
    assert await WorkspaceScheduleStore(state).snapshot() == (registered,)


@pytest.mark.asyncio
async def test_dream_reconcile_recomputes_next_cron_occurrence_for_timezone_and_dst(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Keep this placeholder.",
        schedule=JobSchedule.cron("30 2 * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    clock = _AdvancingClock(datetime(2026, 3, 8, 6, 0, tzinfo=UTC))
    calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal calls
        calls += 1
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    replacement_schedule = JobSchedule.cron("30 2 * * *", "America/New_York")
    await service.register_dream_job(schedule=replacement_schedule)

    service.start()
    await clock.wait_started.wait()
    clock.advance(24 * 60 * 60 + 30 * 60)
    for _ in range(100):
        if calls == 1:
            break
        await asyncio.sleep(0)

    assert calls == 1
    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.schedule == replacement_schedule
    assert persisted.message == existing.message
    await service.close()


@pytest.mark.parametrize(
    ("job_id", "source", "schedule"),
    [
        (
            "dream",
            "user",
            {
                "kind": "cron",
                "at_time": None,
                "every_seconds": None,
                "cron_expr": "0 * * * *",
                "timezone": "UTC",
            },
        ),
        (
            "550e8400-e29b-41d4-a716-446655440000",
            "system",
            {
                "kind": "cron",
                "at_time": None,
                "every_seconds": None,
                "cron_expr": "0 * * * *",
                "timezone": "UTC",
            },
        ),
        (
            "unknown",
            "system",
            {
                "kind": "cron",
                "at_time": None,
                "every_seconds": None,
                "cron_expr": "0 * * * *",
                "timezone": "UTC",
            },
        ),
        (
            "dream",
            "system",
            {
                "kind": "cron",
                "at_time": None,
                "every_seconds": None,
                "cron_expr": "0 * * * *",
                "timezone": "Not/A_Timezone",
            },
        ),
        (
            "dream",
            "system",
            {
                "kind": "at",
                "at_time": "2026-08-07T12:00:00.000+00:00",
                "every_seconds": None,
                "cron_expr": None,
                "timezone": None,
            },
        ),
    ],
)
def test_schedule_store_rejects_corrupt_or_conflicting_system_state(
    workspace: Path,
    agent_home: Path,
    job_id: str,
    source: str,
    schedule: dict[str, object],
) -> None:
    state = _state(workspace, agent_home)
    state.schedule_path.write_text(
        json.dumps(
            [
                {
                    "job_id": job_id,
                    "source": source,
                    "message": "Internal Dream schedule.",
                    "schedule": schedule,
                    "state": {
                        "last_finished_at_ms": None,
                        "last_status": None,
                        "last_error": None,
                    },
                    "created_at_ms": 1,
                    "updated_at_ms": 1,
                }
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScheduleStateError):
        WorkspaceScheduleStore(state)


def test_schedule_service_rejects_an_unsupported_internal_dispatch_identity(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    service = _service(state)
    invalid = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Must not dispatch.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    object.__setattr__(invalid, "source", "system")

    with pytest.raises(ScheduleStateError):
        service._reserve(invalid, current_monotonic=0.0)

    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0


@pytest.mark.asyncio
async def test_due_dream_job_dispatches_directly_without_user_or_session_execution(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0
    user_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        nonlocal user_calls
        del job
        user_calls += 1

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.cron("* * * * *", "UTC"))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert user_calls == 0
    assert await service.public_snapshot() == ()
    persisted = await WorkspaceScheduleStore(state).snapshot()
    assert len(persisted) == 1
    assert persisted[0].source == "system"
    assert persisted[0].state.last_status == "ok"
    assert not tuple(state.schedule_sessions_directory.glob("*.jsonl"))
    await service.close()


@pytest.mark.asyncio
async def test_memory_task_running_advances_one_dream_occurrence_without_terminal_write(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="Memory Task is already running.",
            processed_count=0,
            memory_updated=False,
            cursor=0,
            error=ErrorInfo(
                code="memory_task_running",
                message="A Memory Task is already running.",
            ),
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(600)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert (await WorkspaceScheduleStore(state).snapshot())[0].state.last_status is None
    for _ in range(20):
        await asyncio.sleep(0)
    assert dream_calls == 1

    clock.advance(59)
    await asyncio.sleep(0)
    assert dream_calls == 1
    clock.advance(1)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_dream_periodic_overlap_skips_one_occurrence_without_a_second_dream_call(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        started.set()
        await release.wait()
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()
    clock.advance(60)
    for _ in range(100):
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert service.status_snapshot().active_job_count == 1
    release.set()
    for _ in range(100):
        if service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)
    assert service.status_snapshot().active_job_count == 0

    clock.advance(60)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_drain_cancels_user_and_dream_then_resume_keeps_progress(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    user_calls = 0
    dream_calls = 0
    user_started = asyncio.Event()
    dream_started = asyncio.Event()
    user_cancelled = asyncio.Event()
    dream_cancelled = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        nonlocal user_calls
        del job
        user_calls += 1
        if user_calls == 1:
            user_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                user_cancelled.set()
                raise

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        if dream_calls == 1:
            dream_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                dream_cancelled.set()
                raise
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    user_job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await service.add_user_job(user_job)
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await user_started.wait()
    await dream_started.wait()

    await service.pause_and_drain()

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0
    assert user_cancelled.is_set()
    assert dream_cancelled.is_set()
    assert user_calls == 1
    assert dream_calls == 1
    assert all(
        job.state == ScheduleJobState() for job in await WorkspaceScheduleStore(state).snapshot()
    )

    clock.advance(600)
    for _ in range(20):
        await asyncio.sleep(0)
    assert user_calls == 1
    assert dream_calls == 1

    service.resume()
    for _ in range(100):
        if user_calls == 2 and dream_calls == 2 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)
    assert user_calls == 2
    assert dream_calls == 2
    assert all(
        job.state.last_status == "ok" for job in await WorkspaceScheduleStore(state).snapshot()
    )
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_wait_idle_waits_for_dream_to_finish_naturally(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> object:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return None

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()

    paused = asyncio.create_task(service.pause_and_wait_idle())
    await asyncio.sleep(0)
    assert not paused.done()
    assert not cancelled.is_set()

    release.set()
    await paused

    assert not cancelled.is_set()
    saved = await WorkspaceScheduleStore(state).snapshot()
    assert saved[0].state.last_status == "ok"
    service.resume()
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_drain_cancels_one_shot_terminal_commit_and_retries_once(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    commit = _BlockingOperation(store._remove_terminal_job)
    monkeypatch.setattr(store, "_remove_terminal_job", commit.run)

    callbacks = 0

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callbacks
        assert active_job == job
        callbacks += 1

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    service._store = store
    drain = _DrainObserver(service._cancel_and_drain_job_tasks)
    monkeypatch.setattr(service, "_cancel_and_drain_job_tasks", drain.drain)

    service.start()
    await commit.started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await drain.started.wait()
    await commit.cancelled.wait()
    await paused

    assert callbacks == 1
    assert commit.cancellation_count == 1
    assert await store.snapshot() == (job,)
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0
    assert job.job_id not in service._consumed_at_jobs

    commit.release.set()
    clock.wait_started = asyncio.Event()
    service.resume()
    await clock.wait_started.wait()
    for _ in range(100):
        if callbacks == 2 and await store.snapshot() == ():
            break
        await asyncio.sleep(0)
    assert callbacks == 2
    assert await store.snapshot() == ()
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_drain_cancels_recurring_terminal_commit_without_immediate_replay(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a recurring user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    callback_count = 0
    next_occurrence_started = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        assert active_job.schedule == job.schedule
        callback_count += 1
        if callback_count == 2:
            next_occurrence_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    store = service._store
    service_identity = id(service)
    store_identity = id(store)
    await service.add_user_job(job)
    commit = _BlockingOperation(store.commit_terminal)
    monkeypatch.setattr(store, "commit_terminal", commit.run)
    drain = _DrainObserver(service._cancel_and_drain_job_tasks)
    monkeypatch.setattr(service, "_cancel_and_drain_job_tasks", drain.drain)

    service.start()
    await commit.started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await drain.started.wait()
    await commit.cancelled.wait()
    await paused

    persisted = (await store.snapshot())[0]
    assert persisted.state == ScheduleJobState()
    assert commit.cancellation_count == 1
    assert callback_count == 1
    assert service.status_snapshot().active_job_count == 0
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service._faulted is False
    assert service._terminal_store_error_logged is False

    try:
        commit.release.set()
        clock.wait_started = asyncio.Event()
        service.resume()
        assert id(service) == service_identity
        assert id(service._store) == store_identity
        await clock.wait_started.wait()
        assert callback_count == 1

        clock.advance(59)
        assert callback_count == 1

        clock.advance(1)
        await next_occurrence_started.wait()
        assert callback_count == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_completed_terminal_commit_wins_the_pause_race_without_replay(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Keep a completed recurring terminal commit.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    callback_count = 0
    terminal_committed = asyncio.Event()
    next_occurrence_started = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        callback_count += 1
        if callback_count == 2:
            next_occurrence_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    original_commit = service._store.commit_terminal

    async def observed_commit(*args: object, **kwargs: object) -> ScheduleJob | None:
        committed = await original_commit(*args, **kwargs)  # type: ignore[arg-type]
        terminal_committed.set()
        return committed

    monkeypatch.setattr(service._store, "commit_terminal", observed_commit)

    service.start()
    await terminal_committed.wait()
    await service.pause_and_drain()

    persisted = (await service._store.snapshot())[0]
    assert persisted.state.last_status == "ok"
    assert callback_count == 1
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0

    try:
        clock.wait_started = asyncio.Event()
        service.resume()
        await clock.wait_started.wait()
        assert callback_count == 1

        clock.advance(59)
        assert callback_count == 1

        clock.advance(1)
        await next_occurrence_started.wait()
        assert callback_count == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_and_drain_catches_terminal_commit_created_during_run_cancellation(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Finish cancellation cleanup before terminal persistence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    run_started = asyncio.Event()
    run_cancelled = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        run_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            run_cancelled.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    commit = _BlockingOperation(service._store.commit_terminal)
    monkeypatch.setattr(service._store, "commit_terminal", commit.run)

    service.start()
    await run_started.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    await run_cancelled.wait()
    await paused

    assert commit.release.is_set() is False
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0
    assert (await service._store.snapshot())[0].state == ScheduleJobState()
    await service.close()


@pytest.mark.asyncio
async def test_cancelled_pause_waiter_observes_terminal_drain_before_cancellation_propagates(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Drain terminal cancellation cleanup.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    terminal_started = asyncio.Event()
    terminal_cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()
    cancellation_count = 0

    async def blocked_remove(*args: object, **kwargs: object) -> bool:
        nonlocal cancellation_count
        del args, kwargs
        terminal_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("blocked terminal removal returned without cancellation")
        except asyncio.CancelledError:
            cancellation_count += 1
            terminal_cancelled.set()
            await cleanup_release.wait()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=lambda active_job: _completed_user_job(active_job, job),
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    monkeypatch.setattr(service._store, "_remove_terminal_job", blocked_remove)
    drain = _DrainObserver(service._cancel_and_drain_job_tasks)
    monkeypatch.setattr(service, "_cancel_and_drain_job_tasks", drain.drain)

    service.start()
    await terminal_started.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    await drain.started.wait()
    paused.cancel()
    await terminal_cancelled.wait()

    assert not paused.done()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await paused

    assert cancellation_count == 1
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0
    assert await service._store.snapshot() == (job,)
    await service.close()


@pytest.mark.asyncio
async def test_direct_close_waits_for_terminal_commit_without_cancelling_it(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Finish terminal persistence during direct close.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=lambda active_job: _completed_user_job(active_job, job),
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    commit = _BlockingOperation(service._store._remove_terminal_job)
    monkeypatch.setattr(service._store, "_remove_terminal_job", commit.run)
    close_observer = _DrainObserver(service._close_owned_tasks)
    monkeypatch.setattr(service, "_close_owned_tasks", close_observer.drain)

    service.start()
    await commit.started.wait()
    closing = asyncio.create_task(service.close())
    await close_observer.started.wait()

    assert not closing.done()
    assert commit.cancellation_count == 0
    commit.release.set()
    await closing

    assert commit.cancellation_count == 0
    assert await service._store.snapshot() == ()
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0


@pytest.mark.asyncio
async def test_pause_and_drain_preserves_next_cron_occurrence_after_terminal_cancellation(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run one Cron occurrence.",
        schedule=JobSchedule.cron("* * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    callback_count = 0
    next_occurrence_started = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        callback_count += 1
        if callback_count == 2:
            next_occurrence_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    commit = _BlockingOperation(service._store.commit_terminal)
    monkeypatch.setattr(service._store, "commit_terminal", commit.run)

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await commit.started.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    await commit.cancelled.wait()
    await paused

    assert callback_count == 1
    assert commit.cancellation_count == 1
    assert (await service._store.snapshot())[0].state == ScheduleJobState()
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0

    try:
        commit.release.set()
        clock.wait_started = asyncio.Event()
        service.resume()
        await clock.wait_started.wait()
        assert callback_count == 1

        clock.advance(59)
        assert callback_count == 1

        clock.advance(1)
        await next_occurrence_started.wait()
        assert callback_count == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancelled_one_shot_terminal_commit_stays_pending_and_retries_once(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Retry a cancelled terminal commit.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    callback_count = 0
    second_run_finished = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        callback_count += 1
        if callback_count == 2:
            run_task = asyncio.current_task()
            assert run_task is not None
            run_task.add_done_callback(lambda _task: second_run_finished.set())

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    store = service._store
    original_commit = store._remove_terminal_job
    commit_started = asyncio.Event()
    cancel_commit = asyncio.Event()
    second_commit_finished = asyncio.Event()
    commit_attempts = 0

    async def cancelled_then_successful_commit(*args: object, **kwargs: object) -> bool:
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts == 1:
            commit_started.set()
            await cancel_commit.wait()
            raise asyncio.CancelledError
        committed = await original_commit(*args, **kwargs)  # type: ignore[arg-type]
        second_commit_finished.set()
        return committed

    monkeypatch.setattr(store, "_remove_terminal_job", cancelled_then_successful_commit)
    drain = _DrainObserver(service._cancel_and_drain_job_tasks)
    monkeypatch.setattr(service, "_cancel_and_drain_job_tasks", drain.drain)

    service.start()
    await commit_started.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    await drain.started.wait()

    assert not paused.done()
    assert service.status_snapshot().active_job_count == 1
    cancel_commit.set()
    await paused

    assert callback_count == 1
    assert commit_attempts == 1
    assert await store.snapshot() == (job,)
    assert job.job_id not in service._consumed_at_jobs
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0

    try:
        clock.wait_started = asyncio.Event()
        service.resume()
        await second_commit_finished.wait()
        await second_run_finished.wait()
        await clock.wait_started.wait()

        assert callback_count == 2
        assert commit_attempts == 2
        assert await store.snapshot() == ()
        assert service._run_tasks == set()
        assert service._terminal_commit_tasks == set()
        assert service.status_snapshot().active_job_count == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_linearizes_after_an_inflight_reservation_and_preserves_its_occurrence(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    timestamp = 1
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(10),
        created_at_ms=timestamp,
        updated_at_ms=timestamp,
    )
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    reservation = _BlockingReservation(store)
    pause_started = asyncio.Event()
    callback_count = 0
    callback_started = asyncio.Event()

    monkeypatch.setattr(store, "reserve_due", reservation.reserve_due)

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        assert active_job.schedule == job.schedule
        callback_count += 1
        callback_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    service._store = store
    original_pause = service._pause_owned_tasks

    async def observed_pause() -> None:
        pause_started.set()
        await original_pause()

    monkeypatch.setattr(service, "_pause_owned_tasks", observed_pause)

    service.start()
    await reservation.completed.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    await pause_started.wait()
    assert service._paused is False
    assert not paused.done()
    reservation.release.set()
    await paused

    try:
        callbacks_before_resume = callback_count
        assert reservation.calls == 1
        assert service._paused is True
        assert service._run_tasks == set()
        assert service.status_snapshot().active_job_count == 0

        clock.wait_started = asyncio.Event()
        callback_started.clear()
        service.resume()
        await clock.wait_started.wait()

        assert reservation.calls == 1
        assert callback_count == callbacks_before_resume

        clock.wait_started = asyncio.Event()
        clock.advance(10)
        await callback_started.wait()
        await clock.wait_started.wait()
        assert callback_count == callbacks_before_resume + 1
        assert reservation.calls == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_and_resume_retries_an_unfinished_user_one_shot(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a one-shot occurrence.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    calls = 0
    blocker = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal calls
        assert active_job.job_id == job.job_id
        calls += 1
        if calls == 1:
            first_started.set()
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                raise
        else:
            second_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)

    service.start()
    await first_started.wait()
    await service.pause_and_drain()
    assert first_cancelled.is_set()
    assert calls == 1
    assert await service._store.snapshot() == (job,)

    try:
        clock.wait_started = asyncio.Event()
        service.resume()
        await second_started.wait()
        await clock.wait_started.wait()
        assert calls == 2
        assert await service._store.snapshot() == ()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_paused_reserve_gate_rejects_a_one_shot_without_consuming_it(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a one-shot occurrence.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    callback_started = asyncio.Event()
    callback_count = 0

    async def execute_user_job(active_job: ScheduleJob) -> None:
        nonlocal callback_count
        assert active_job.job_id == job.job_id
        callback_count += 1
        callback_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    await service.pause_and_drain()

    service._reserve(job, current_monotonic=clock.monotonic())

    assert not callback_started.is_set()
    assert callback_count == 0
    assert service._run_tasks == set()
    assert service._active_job_ids == set()
    assert service._consumed_at_jobs == set()
    assert await service._store.snapshot() == (job,)

    try:
        clock.wait_started = asyncio.Event()
        service.resume()
        await callback_started.wait()
        await clock.wait_started.wait()
        assert callback_count == 1
        assert await service._store.snapshot() == ()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_after_snapshot_await_does_not_start_a_new_reservation(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a one-shot occurrence.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    original_snapshot = store.snapshot
    original_reserve_due = store.reserve_due
    snapshot_started = asyncio.Event()
    snapshot_cancelled = asyncio.Event()
    callback_started = asyncio.Event()
    snapshot_calls = 0
    reserve_calls = 0

    async def cancelled_snapshot() -> tuple[ScheduleJob, ...]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls > 1:
            return await original_snapshot()
        snapshot_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            snapshot_cancelled.set()
            return await original_snapshot()
        raise AssertionError("snapshot barrier was released without cancellation")

    async def record_reservation(
        candidates: tuple[ScheduleJob, ...],
    ) -> tuple[ScheduleJob, ...]:
        nonlocal reserve_calls
        reserve_calls += 1
        return await original_reserve_due(candidates)

    monkeypatch.setattr(store, "snapshot", cancelled_snapshot)
    monkeypatch.setattr(store, "reserve_due", record_reservation)

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        callback_started.set()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    service._store = store
    service.start()
    await snapshot_started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await snapshot_cancelled.wait()
    await paused

    assert reserve_calls == 0
    assert not callback_started.is_set()
    assert await original_snapshot() == (job,)

    try:
        clock.wait_started = asyncio.Event()
        service.resume()
        await callback_started.wait()
        await clock.wait_started.wait()
        assert reserve_calls == 1
        assert await original_snapshot() == ()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pause_preserves_a_completed_dream_reservation_before_resume(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="dream",
        source="system",
        message="Internal Dream schedule.",
        schedule=JobSchedule.every(10),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store._register_system_job(job)
    reservation = _BlockingReservation(store)
    pause_started = asyncio.Event()
    dream_calls = 0
    dream_started = asyncio.Event()

    monkeypatch.setattr(store, "reserve_due", reservation.reserve_due)

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        dream_started.set()
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=lambda active_job: _unexpected_user_job(active_job),
        execute_dream=execute_dream,
    )
    service._store = store
    original_pause = service._pause_owned_tasks

    async def observed_pause() -> None:
        pause_started.set()
        await original_pause()

    monkeypatch.setattr(service, "_pause_owned_tasks", observed_pause)
    service.start()
    await reservation.completed.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await pause_started.wait()
    assert service._paused is False
    assert not paused.done()
    reservation.release.set()
    await paused

    try:
        dreams_before_resume = dream_calls
        assert reservation.calls == 1
        clock.wait_started = asyncio.Event()
        dream_started.clear()
        service.resume()
        await clock.wait_started.wait()
        assert reservation.calls == 1
        assert dream_calls == dreams_before_resume

        clock.wait_started = asyncio.Event()
        clock.advance(10)
        await dream_started.wait()
        await clock.wait_started.wait()
        assert dream_calls == dreams_before_resume + 1
        assert reservation.calls == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_concurrent_pause_waiters_share_a_cancellable_drain_barrier(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    service.start()
    await started.wait()

    first = asyncio.create_task(service.pause_and_drain())
    second = asyncio.create_task(service.pause_and_drain())
    await cancelled.wait()
    with pytest.raises(RuntimeError, match="pause is still draining"):
        service.resume()
    first.cancel()
    assert not second.done()

    cleanup_release.set()
    await second
    with pytest.raises(asyncio.CancelledError):
        await first

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    service.resume()
    service.resume()
    await service.close()


@pytest.mark.asyncio
async def test_pause_before_start_then_resume_does_not_duplicate_dispatcher(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=lambda active_job: _unexpected_user_job(active_job),
        execute_dream=lambda: _unexpected_dream(),
    )
    store = service._store

    await service.pause_and_drain()
    assert service._paused is True
    assert service._loop_task is None

    service.resume()
    service.resume()
    service.start()
    assert service._store is store
    await service.close()

    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()


@pytest.mark.asyncio
async def test_pause_and_abort_interleave_with_deterministic_owned_task_cleanup(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    service.start()
    await started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await cancelled.wait()
    service.abort()
    aborted = asyncio.create_task(service.abort_and_wait())
    cleanup_release.set()
    await asyncio.gather(paused, aborted)

    assert service._aborted is True
    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    with pytest.raises(RuntimeError, match="closed"):
        service.resume()
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_close_interleave_with_deterministic_owned_task_cleanup(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    await service.add_user_job(job)
    service.start()
    await started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await cancelled.wait()
    closing = asyncio.create_task(service.close())
    await service._closing.wait()
    assert not closing.done()

    cleanup_release.set()
    await asyncio.gather(paused, closing)

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    with pytest.raises(RuntimeError, match="closed"):
        service.resume()


@pytest.mark.asyncio
async def test_close_rejects_a_run_registered_after_its_initial_task_snapshot(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    original_reserve_due = store.reserve_due
    reserve_started = asyncio.Event()
    reservation_completed = asyncio.Event()
    run_started = asyncio.Event()

    async def completed_after_cancellation(
        candidates: tuple[ScheduleJob, ...],
    ) -> tuple[ScheduleJob, ...]:
        reserve_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            reserved = await original_reserve_due(candidates)
            reservation_completed.set()
            return reserved
        raise AssertionError("reservation barrier was released without cancellation")

    monkeypatch.setattr(store, "reserve_due", completed_after_cancellation)

    async def execute_user_job(active_job: ScheduleJob) -> None:
        assert active_job.job_id == job.job_id
        run_started.set()
        await asyncio.Event().wait()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=lambda: _unexpected_dream(),
    )
    service._store = store
    service.start()
    await reserve_started.wait()

    closing = asyncio.create_task(service.close())
    await reservation_completed.wait()
    await closing

    try:
        assert service._loop_task is None
        assert not run_started.is_set()
        assert await store.snapshot() == (job,)
        assert service._run_tasks == set()
        assert service._terminal_commit_tasks == set()
    finally:
        if service._run_tasks:
            service.abort()
            await service.abort_and_wait()


async def _unexpected_user_job(job: ScheduleJob) -> None:
    raise AssertionError(f"unexpected User Job: {job.job_id}")


async def _completed_user_job(active_job: ScheduleJob, expected: ScheduleJob) -> None:
    assert active_job == expected


async def _unexpected_dream() -> object:
    raise AssertionError("unexpected Dream execution")


@pytest.mark.asyncio
async def test_cancelled_pause_and_drain_still_finishes_the_barrier(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise
        raise AssertionError("blocked Dream executor returned without cancellation")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await asyncio.sleep(0)
    paused.cancel()
    await cancelled.wait()
    assert not paused.done()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await paused

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    service.resume()
    await service.close()


@pytest.mark.asyncio
async def test_dream_error_persists_terminal_error_and_keeps_recurrence(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="Memory failed",
            processed_count=0,
            memory_updated=False,
            cursor=0,
            error=ErrorInfo(code="model_failed", message="safe Dream failure"),
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state.last_status == "error"
    assert persisted.state.last_error == "safe Dream failure"
    assert dream_calls == 1
    clock.advance(59)
    await asyncio.sleep(0)
    assert dream_calls == 1
    clock.advance(1)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_dream_executor_exception_persists_safe_terminal_error(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        raise RuntimeError("PRIVATE_DREAM_EXCEPTION")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state.last_status == "error"
    assert persisted.state.last_error == "Schedule Job execution failed."
    assert "PRIVATE_DREAM_EXCEPTION" not in persisted.state.last_error
    await service.close()


@pytest.mark.asyncio
async def test_dream_executor_cancellation_leaves_occurrence_pending_without_terminal_state(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        raise asyncio.CancelledError()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state == ScheduleJobState()
    assert dream_calls == 1
    await service.close()


@pytest.mark.asyncio
async def test_schedule_service_runs_dream_silently_without_a_foreground_session_log(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A pending summary.", datetime(2026, 8, 7, 12, 0, tzinfo=UTC))
    provider = ScriptedFakeProvider(completions=(_memory_response("No update needed."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    clock = _AdvancingClock()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    foreground_session_id = "20260827-120000-000000_550e8400-e29b-41d4-a716-446655440000"
    with session_log(state, foreground_session_id):
        service.start()
        await clock.wait_started.wait()
        clock.advance(60)
        for _ in range(100):
            if (
                len(provider.complete_requests) == 1
                and service.status_snapshot().active_job_count == 0
            ):
                break
            await asyncio.sleep(0)

    await service.close()
    await dream.close()
    assert len(provider.complete_requests) == 1
    assert not (state.logs_directory / f"{foreground_session_id}.log").exists()
    assert not tuple(state.schedule_sessions_directory.glob("*.jsonl"))


class _BlockingDreamRouter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, *args: object, **kwargs: object) -> ModelResponse:
        del args, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("blocking Dream router completed unexpectedly")


@pytest.mark.asyncio
async def test_schedule_service_close_cancels_and_drains_an_active_dream_run(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A pending summary.", datetime(2026, 8, 7, 12, 0, tzinfo=UTC))
    router = _BlockingDreamRouter()
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    clock = _AdvancingClock()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await router.started.wait()

    await service.close()
    assert router.cancelled.is_set()
    assert dream._task is None
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    await dream.close()


@pytest.mark.asyncio
async def test_schedule_service_abort_then_close_still_drains_owned_tasks(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    started = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("aborted Dream unexpectedly completed")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()

    service.abort()
    await service.close()
    await service.abort_and_wait()

    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
