from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from myclaw.agent.confirmation import (
    BackgroundConfirmationOwner,
    ConfirmationAborted,
    ConfirmationEnvelope,
)
from myclaw.agent.permission import PermissionSnapshot, ToolPermissionLevel
from myclaw.agent.tools.core.exec_policy import ExecAssessment, ResolvedExecShell
from myclaw.agent.tools.permission import (
    FileAccess,
    MCPToolIdentity,
    NetworkAssessment,
    NormalizedNetworkTarget,
    PermissionContext,
    ToolInvocationFacts,
    ToolPermissionPolicy,
)
from myclaw.agent.tools.tool_gateway import ConfirmationDecision, ConfirmationRequest
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleOccurrence, ScheduleService
from myclaw.schedule.store import ScheduleStoreFaultedError
from myclaw.session.session import Session
from myclaw.workspace.state import WorkspaceState

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _snapshot(level: ToolPermissionLevel = "read-only") -> PermissionSnapshot:
    return PermissionSnapshot(
        level=level,
        exec_shell=ResolvedExecShell(
            selector="auto",
            platform="windows",
            family="powershell",
            executable=None,
            flags=(),
            environment=(),
            available=False,
        ),
    )


async def _noop_dream() -> object:
    return None


async def _wait_until(predicate: object) -> None:
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_user_occurrence_captures_one_immutable_permission_snapshot_at_admission(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    snapshot = _snapshot()
    snapshot_calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    occurrences: list[ScheduleOccurrence] = []

    def capture_snapshot() -> PermissionSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return snapshot

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        occurrences.append(occurrence)
        started.set()
        await release.wait()

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        permission_snapshot_factory=capture_snapshot,
        execute_dream=_noop_dream,
    )
    await service.add_user_job(job)
    service.start()
    await started.wait()

    assert snapshot_calls == 1
    assert len(occurrences) == 1
    assert occurrences[0].job == job
    assert occurrences[0].permission_snapshot is snapshot
    assert isinstance(occurrences[0].occurrence_id, UUID)
    assert "permission_snapshot" not in job.to_dict()

    release.set()
    for _ in range(100):
        if service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)
    await service.close()


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("read-only", "confirm"),
        ("workspace-write", "confirm"),
        ("full-access", "direct"),
    ],
)
def test_snapshot_schedule_context_applies_mcp_confirmation_policy(
    workspace: Path,
    level: ToolPermissionLevel,
    expected: str,
) -> None:
    snapshot = _snapshot(level)
    context = PermissionContext.from_snapshot(
        snapshot,
        workspace_root=workspace,
        origin="schedule",
    )
    facts = ToolInvocationFacts(
        "mcp_remote_echo",
        {"value": "payload"},
        mcp_identity=MCPToolIdentity(
            server_name="server",
            remote_name="echo",
            model_name="mcp_remote_echo",
        ),
    )

    authorization = ToolPermissionPolicy().open(facts, context)

    assert authorization.initial_decision() == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("read-only", "confirm"),
        ("workspace-write", "direct"),
        ("full-access", "direct"),
    ],
)
def test_snapshot_schedule_context_applies_file_policy(
    workspace: Path,
    level: ToolPermissionLevel,
    expected: str,
) -> None:
    context = PermissionContext.from_snapshot(
        _snapshot(level),
        workspace_root=workspace,
        origin="schedule",
        configured_schedule_level=level,
    )
    facts = ToolInvocationFacts(
        "write_file",
        {"path": "scheduled.txt", "content": "payload"},
        file_accesses=(
            FileAccess(
                path=workspace / "scheduled.txt",
                role="write",
                base=workspace,
                workspace_root=workspace,
            ),
        ),
    )

    authorization = ToolPermissionPolicy().open(facts, context)

    assert authorization.initial_decision() == expected


@pytest.mark.parametrize("level", ["read-only", "workspace-write", "full-access"])
def test_snapshot_schedule_context_keeps_uncertain_exec_confirmation(
    workspace: Path,
    level: ToolPermissionLevel,
) -> None:
    context = PermissionContext.from_snapshot(
        _snapshot(level),
        workspace_root=workspace,
        origin="schedule",
        configured_schedule_level=level,
    )
    facts = ToolInvocationFacts(
        "exec",
        {"command": "unknown", "cwd": str(workspace)},
        exec_assessment=ExecAssessment(
            syntax_confidence="unknown",
            syntax_uncertain=True,
        ),
    )

    authorization = ToolPermissionPolicy().open(facts, context)

    assert authorization.initial_decision() == "confirm"


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("read-only", "confirm"),
        ("workspace-write", "confirm"),
        ("full-access", "direct"),
    ],
)
def test_snapshot_schedule_context_applies_web_fetch_policy(
    workspace: Path,
    level: ToolPermissionLevel,
    expected: str,
) -> None:
    context = PermissionContext.from_snapshot(
        _snapshot(level),
        workspace_root=workspace,
        origin="schedule",
        configured_schedule_level=level,
    )
    target = NormalizedNetworkTarget(
        url="http://127.0.0.1/private",
        scheme="http",
        host="127.0.0.1",
        port=80,
    )
    facts = ToolInvocationFacts(
        "web_fetch",
        {"url": target.url},
        network_targets=(NetworkAssessment(target, "literal_non_global"),),
    )

    authorization = ToolPermissionPolicy().open(facts, context)

    assert authorization.initial_decision() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_kind", ["at", "every"])
async def test_confirmation_abort_commits_safe_terminal_lifecycle(
    workspace: Path,
    agent_home: Path,
    schedule_kind: str,
) -> None:
    state = _state(workspace, agent_home)
    created_at_ms = int((NOW.timestamp() - 120) * 1000)
    job = ScheduleJob(
        job_id="6fa459ea-ee8a-4ca4-894e-db77e160355e",
        message="Run this.",
        schedule=(
            JobSchedule.at("2026-08-07T11:59:00.000+00:00")
            if schedule_kind == "at"
            else JobSchedule.every(60)
        ),
        created_at_ms=created_at_ms,
        updated_at_ms=created_at_ms,
    )
    started = asyncio.Event()

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        del occurrence
        started.set()
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        execute_dream=_noop_dream,
    )
    await service.add_user_job(job)
    service.start()
    await started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    saved = await service._store.snapshot()
    if schedule_kind == "at":
        assert saved == ()
    else:
        assert len(saved) == 1
        assert saved[0].state.last_status == "error"
        assert saved[0].state.last_error == "Schedule Tool confirmation was aborted."


@pytest.mark.asyncio
async def test_delete_persists_absence_before_aborting_pending_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    created_at_ms = int((NOW.timestamp() - 120) * 1000)
    job = ScheduleJob(
        job_id="9ba7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=created_at_ms,
        updated_at_ms=created_at_ms,
    )
    confirmation_started = asyncio.Event()
    confirmation_cancelled = asyncio.Event()
    owner_generation = UUID("123e4567-e89b-42d3-a456-426614174000")
    service: ScheduleService | None = None

    async def cancel_owner(owner: BackgroundConfirmationOwner) -> None:
        assert owner.job_id == job.job_id
        confirmation_cancelled.set()

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        assert service is not None
        owner = BackgroundConfirmationOwner(
            generation_id=owner_generation,
            job_id=job.job_id,
            occurrence_id=occurrence.occurrence_id,
        )
        service.bind_occurrence_owner(occurrence, owner)
        service.confirmation_waiting(occurrence)
        confirmation_started.set()
        await confirmation_cancelled.wait()
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        cancel_confirmation_owner=cancel_owner,
        execute_dream=_noop_dream,
    )
    await service.add_user_job(job)
    service.start()
    await confirmation_started.wait()

    assert await service.remove_user_job(job.job_id, expected=job) is True
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await service._store.snapshot() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_bound", [False, True])
async def test_delete_cancels_and_drains_the_exact_active_occurrence(
    workspace: Path,
    agent_home: Path,
    owner_bound: bool,
) -> None:
    state = _state(workspace, agent_home)
    created_at_ms = int((NOW.timestamp() - 120) * 1000)
    job = ScheduleJob(
        job_id="7ba7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=created_at_ms,
        updated_at_ms=created_at_ms,
    )
    execution_started = asyncio.Event()
    occurrence_cancelled = asyncio.Event()
    never_finish = asyncio.Event()
    generation_id = UUID("423e4567-e89b-42d3-a456-426614174000")
    cancelled_owners: list[BackgroundConfirmationOwner] = []
    service: ScheduleService | None = None

    async def cancel_owner(owner: BackgroundConfirmationOwner) -> None:
        cancelled_owners.append(owner)

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        assert service is not None
        if owner_bound:
            owner = BackgroundConfirmationOwner(
                generation_id=generation_id,
                job_id=job.job_id,
                occurrence_id=occurrence.occurrence_id,
            )
            service.bind_occurrence_owner(occurrence, owner)
            service.confirmation_waiting(occurrence)
            service.confirmation_finished(occurrence)
        execution_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            occurrence_cancelled.set()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        cancel_confirmation_owner=cancel_owner,
        execute_dream=_noop_dream,
    )
    await service.add_user_job(job)
    service.start()
    await execution_started.wait()

    assert await service.remove_user_job(job.job_id, expected=job) is True

    assert occurrence_cancelled.is_set()
    assert service.status_snapshot().active_job_count == 0
    assert len(cancelled_owners) == int(owner_bound)
    assert await service._store.snapshot() == ()
    await service.close()


@pytest.mark.asyncio
async def test_failed_delete_does_not_cancel_the_active_occurrence(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id="8ba7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=int((NOW.timestamp() - 120) * 1000),
        updated_at_ms=int((NOW.timestamp() - 120) * 1000),
    )
    execution_started = asyncio.Event()
    release = asyncio.Event()
    occurrence_cancelled = asyncio.Event()

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        del occurrence
        execution_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            occurrence_cancelled.set()
            raise

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        execute_dream=_noop_dream,
    )
    await service.add_user_job(job)
    service.start()
    await execution_started.wait()

    async def fail_remove(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise OSError("injected delete failure")

    monkeypatch.setattr(service._store, "remove_user_job", fail_remove)
    with pytest.raises(OSError, match="injected delete failure"):
        await service.remove_user_job(job.job_id, expected=job)

    assert not occurrence_cancelled.is_set()
    release.set()
    await service.close()


@pytest.mark.asyncio
async def test_generation_abort_drain_waits_for_terminal_store_commit(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id="9ca7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=int((NOW.timestamp() - 120) * 1000),
        updated_at_ms=int((NOW.timestamp() - 120) * 1000),
    )
    generation_id = UUID("523e4567-e89b-42d3-a456-426614174000")
    confirmation_started = asyncio.Event()
    confirmation_cancelled = asyncio.Event()
    commit_started = asyncio.Event()
    allow_commit = asyncio.Event()
    service: ScheduleService | None = None

    async def cancel_owner(owner: BackgroundConfirmationOwner) -> None:
        assert owner.generation_id == generation_id
        confirmation_cancelled.set()

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        assert service is not None
        owner = BackgroundConfirmationOwner(
            generation_id=generation_id,
            job_id=job.job_id,
            occurrence_id=occurrence.occurrence_id,
        )
        service.bind_occurrence_owner(occurrence, owner)
        service.confirmation_waiting(occurrence)
        confirmation_started.set()
        await confirmation_cancelled.wait()
        service.confirmation_aborted(occurrence)
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        cancel_confirmation_owner=cancel_owner,
        execute_dream=_noop_dream,
    )
    original_commit = service._store.commit_terminal

    async def blocked_commit(*args: object, **kwargs: object) -> object:
        commit_started.set()
        await allow_commit.wait()
        return await original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service._store, "commit_terminal", blocked_commit)
    await service.add_user_job(job)
    service.start()
    await confirmation_started.wait()
    service.cancel_confirmation_generation(generation_id)
    drain = asyncio.create_task(
        service.drain_confirmation_aborts(generation_id=generation_id)
    )

    await commit_started.wait()
    assert not drain.done()
    allow_commit.set()
    await drain

    saved = await service._store.snapshot()
    assert saved[0].state.last_status == "error"
    assert saved[0].state.last_error == "Schedule Tool confirmation was aborted."
    await service.close()


@pytest.mark.asyncio
async def test_generation_abort_drain_reports_terminal_store_failure(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id="aca7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=int((NOW.timestamp() - 120) * 1000),
        updated_at_ms=int((NOW.timestamp() - 120) * 1000),
    )
    generation_id = UUID("623e4567-e89b-42d3-a456-426614174000")
    confirmation_started = asyncio.Event()
    confirmation_cancelled = asyncio.Event()
    service: ScheduleService | None = None

    async def cancel_owner(owner: BackgroundConfirmationOwner) -> None:
        del owner
        confirmation_cancelled.set()

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        assert service is not None
        service.bind_occurrence_owner(
            occurrence,
            BackgroundConfirmationOwner(
                generation_id=generation_id,
                job_id=job.job_id,
                occurrence_id=occurrence.occurrence_id,
            ),
        )
        service.confirmation_waiting(occurrence)
        confirmation_started.set()
        await confirmation_cancelled.wait()
        service.confirmation_aborted(occurrence)
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        cancel_confirmation_owner=cancel_owner,
        execute_dream=_noop_dream,
    )

    async def fail_commit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("injected terminal failure")

    monkeypatch.setattr(service._store, "commit_terminal", fail_commit)
    await service.add_user_job(job)
    service.start()
    await confirmation_started.wait()
    service.cancel_confirmation_generation(generation_id)

    with pytest.raises(ScheduleStoreFaultedError):
        await service.drain_confirmation_aborts(generation_id=generation_id)

    assert service.status_snapshot().status == "faulted"
    await service.close()


@pytest.mark.asyncio
async def test_generation_confirmation_admission_stays_closed_until_pause(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id="bca7b810-9dad-41d4-8954-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=int((NOW.timestamp() - 120) * 1000),
        updated_at_ms=int((NOW.timestamp() - 120) * 1000),
    )
    generation_id = UUID("723e4567-e89b-42d3-a456-426614174000")
    occurrence_ready = asyncio.Event()
    request_confirmation = asyncio.Event()
    admission_rejected = asyncio.Event()
    commit_started = asyncio.Event()
    allow_commit = asyncio.Event()
    service: ScheduleService | None = None

    async def execute_occurrence(occurrence: ScheduleOccurrence) -> None:
        assert service is not None
        service.bind_occurrence_owner(
            occurrence,
            BackgroundConfirmationOwner(
                generation_id=generation_id,
                job_id=job.job_id,
                occurrence_id=occurrence.occurrence_id,
            ),
        )
        occurrence_ready.set()
        await request_confirmation.wait()
        with pytest.raises(ConfirmationAborted):
            service.confirmation_waiting(occurrence)
        admission_rejected.set()
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    service = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_occurrence=execute_occurrence,
        execute_dream=_noop_dream,
    )
    original_commit = service._store.commit_terminal

    async def blocked_commit(*args: object, **kwargs: object) -> object:
        commit_started.set()
        await allow_commit.wait()
        return await original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service._store, "commit_terminal", blocked_commit)
    await service.add_user_job(job)
    service.start()
    await occurrence_ready.wait()
    service.cancel_confirmation_generation(generation_id)
    await service.drain_confirmation_aborts(generation_id=generation_id)
    request_confirmation.set()

    await admission_rejected.wait()
    await commit_started.wait()
    paused = asyncio.create_task(service.pause_and_drain())
    assert not paused.done()
    allow_commit.set()
    await paused

    saved = await service._store.snapshot()
    assert saved[0].state.last_status == "error"
    assert saved[0].state.last_error == "Schedule Tool confirmation was aborted."
    await service.close()


@pytest.mark.asyncio
async def test_agent_loop_reuses_occurrence_snapshot_for_context_gateway_and_envelope(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    from tests.scheduling.test_schedule_agent_loop import (
        _agent_loop,
        _ScheduleProvider,
    )

    provider = _ScheduleProvider()
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_Clock(),
    )
    job = ScheduleJob(
        job_id="123e4567-e89b-42d3-a456-426614174000",
        message="Ask for approval.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    snapshot = _snapshot("read-only")
    occurrence = ScheduleOccurrence(
        job=job,
        occurrence_id=UUID("223e4567-e89b-42d3-a456-426614174000"),
        permission_snapshot=snapshot,
    )
    owner = BackgroundConfirmationOwner(
        generation_id=loop.generation_id,
        job_id=job.job_id,
        occurrence_id=occurrence.occurrence_id,
    )
    captured_context: list[PermissionContext] = []
    captured_projector: list[object] = []
    envelopes: list[ConfirmationEnvelope] = []

    async def requester(envelope: ConfirmationEnvelope) -> ConfirmationDecision:
        envelopes.append(envelope)
        return "approved"

    loop.bind_confirmation_requester(requester)

    def new_gateway(**kwargs: object) -> object:
        captured_context.append(kwargs["permission_context"])  # type: ignore[arg-type]
        return object()

    class _Runner:
        async def run(self, initial_messages: object, **kwargs: object) -> object:
            del initial_messages
            confirmation = kwargs["confirmation"]
            assert callable(confirmation)
            await confirmation(
                ConfirmationRequest(
                    confirmation_id=UUID("323e4567-e89b-42d3-a456-426614174000"),
                    tool_call_id="call-1",
                    tool_name="mcp_remote_echo",
                    summary="Confirm mcp_remote_echo",
                    details={"value": "payload"},
                    reason="MCP call requires confirmation.",
                )
            )
            return SimpleNamespace(messages=(), finish_reason="completed", error=None)

    def new_run_context(*args: object, **kwargs: object) -> object:
        del args
        captured_projector.append(kwargs["project_messages"])
        return SimpleNamespace(runner=_Runner())

    async def prepare(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        return []

    def commit(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(loop, "_new_run_gateway", new_gateway)
    monkeypatch.setattr(loop, "_new_agent_run_context", new_run_context)
    monkeypatch.setattr(loop, "_prepare_agent_run", prepare)
    monkeypatch.setattr(loop, "_commit_schedule_run", commit)
    monkeypatch.setattr(schedule, "occurrence_owner", lambda active: owner)
    monkeypatch.setattr(schedule, "confirmation_waiting", lambda active: None)
    monkeypatch.setattr(schedule, "confirmation_finished", lambda active: None)

    session = Session.create_schedule(
        loop.session.workspace_state,
        job.job_id,
        now=lambda: NOW,
        title=job.title or "Scheduled Job",
    )
    await loop._run_schedule_agent_scoped(session, job, occurrence)

    assert len(captured_context) == 1
    context = captured_context[0]
    assert context.snapshot is snapshot
    assert context.level == snapshot.level
    assert context.exec_shell == snapshot.exec_shell
    assert len(captured_projector) == 1
    projected = captured_projector[0]([{"role": "user", "content": job.message}])  # type: ignore[operator]
    assert snapshot.level in str(projected)
    assert len(envelopes) == 1
    assert envelopes[0].origin == "background"
    assert envelopes[0].owner is owner
    assert envelopes[0].job_id == job.job_id
    assert envelopes[0].title == job.title

    await loop.close()
    await schedule.close()
    await dream.close()
    await router.close()


@pytest.mark.asyncio
async def test_agent_loop_records_confirmation_abort_and_preserves_its_type(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    agent_home: Path,
) -> None:
    from tests.scheduling.test_schedule_agent_loop import (
        _agent_loop,
        _ScheduleProvider,
    )

    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        _ScheduleProvider(),
        schedule_clock=_Clock(),
    )
    job = ScheduleJob(
        job_id="cba7b810-9dad-41d4-8954-00c04fd430c8",
        message="Ask for approval.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    occurrence = ScheduleOccurrence(
        job=job,
        occurrence_id=UUID("823e4567-e89b-42d3-a456-426614174000"),
        permission_snapshot=_snapshot("read-only"),
    )
    owner = BackgroundConfirmationOwner(
        generation_id=loop.generation_id,
        job_id=job.job_id,
        occurrence_id=occurrence.occurrence_id,
    )
    lifecycle: list[str] = []
    recorded_failures: list[str] = []

    async def requester(envelope: ConfirmationEnvelope) -> ConfirmationDecision:
        del envelope
        raise ConfirmationAborted("confirmation lifecycle cancelled")

    loop.bind_confirmation_requester(requester)

    class _Runner:
        async def run(self, initial_messages: object, **kwargs: object) -> object:
            del initial_messages
            confirmation = kwargs["confirmation"]
            assert callable(confirmation)
            await confirmation(
                ConfirmationRequest(
                    confirmation_id=UUID("923e4567-e89b-42d3-a456-426614174000"),
                    tool_call_id="call-abort",
                    tool_name="mcp_remote_echo",
                    summary="Confirm mcp_remote_echo",
                    details={"value": "payload"},
                    reason="MCP call requires confirmation.",
                )
            )
            raise AssertionError("confirmation abort must stop the Runner")

    def new_run_context(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return SimpleNamespace(runner=_Runner())

    async def prepare(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        return []

    def record_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        recorded_failures.append("recorded")

    monkeypatch.setattr(loop, "_new_run_gateway", lambda **kwargs: object())
    monkeypatch.setattr(loop, "_new_agent_run_context", new_run_context)
    monkeypatch.setattr(loop, "_prepare_agent_run", prepare)
    monkeypatch.setattr(loop, "_record_schedule_failure", record_failure)
    monkeypatch.setattr(schedule, "occurrence_owner", lambda active: owner)
    monkeypatch.setattr(schedule, "confirmation_waiting", lambda active: lifecycle.append("wait"))
    monkeypatch.setattr(schedule, "confirmation_aborted", lambda active: lifecycle.append("abort"))
    monkeypatch.setattr(schedule, "confirmation_finished", lambda active: lifecycle.append("finish"))

    session = Session.create_schedule(
        loop.session.workspace_state,
        job.job_id,
        now=lambda: NOW,
        title=job.title or "Scheduled Job",
    )
    with pytest.raises(ConfirmationAborted, match="confirmation lifecycle cancelled"):
        await loop._run_schedule_agent_scoped(session, job, occurrence)

    assert lifecycle == ["wait", "abort"]
    assert recorded_failures == ["recorded"]

    await loop.close()
    await schedule.close()
    await dream.close()
    await router.close()
