from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from aide.agent.confirmation import ConfirmationEnvelope, SubAgentConfirmationOwner
from aide.agent.session.session import Session
from aide.agent.subagents.coordinator import SubAgentPool
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
from aide.agent.subagents.ports import SubAgentRecordRepository
from aide.agent.subagents.store import (
    SubAgentRecordStore,
    SubAgentRequestError,
    SubAgentStoreError,
)
from aide.agent.tools.tool_gateway import ConfirmationRequest
from aide.agent.workspace_state import WorkspaceState
from aide.config.config import ConfigLoader
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.discovery import create_credential, read_credential
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService, WorkspaceRecord
from aide.service.transport import create_app
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures.project_removal import complete_project_removal
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider
from tests.service.test_service_transport import _persist_session, _prepare_agent_home

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


def _snapshot() -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="test-provider",
        model="test-model",
        reasoning_effort="mid",
        permission_level="workspace-write",
        shell="pwsh",
        tool_schemas=({"name": "read_file", "input_schema": {"type": "object"}},),
        system_prompt="You are Aide.",
    )


def _register(repository: SubAgentRecordStore, title: str) -> SubAgentRecord:
    return repository.register(
        title=title,
        task=f"Complete {title}.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )


def _complete(repository: SubAgentRecordStore, record: SubAgentRecord) -> SubAgentRecord:
    running = repository.save(
        replace(
            record,
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=record.revision + 1,
        )
    )
    return repository.save(
        replace(
            running,
            status=SubAgentStatus.COMPLETED,
            finished_at=_NOW,
            conversation=({"role": "assistant", "content": "Full child output."},),
            result="Completed answer.",
            revision=running.revision + 1,
        )
    )


def _require_record(repository: SubAgentRecordRepository, agent_id: str) -> SubAgentRecord:
    record = repository.get(agent_id)
    assert record is not None
    return record


class _HoldingExecutor:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release: dict[str, asyncio.Event] = {}
        self.cancel_requested: dict[str, asyncio.Event] = {}
        self.interrupted: dict[str, bool] = {}

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult:
        del emit
        self.release[record.agent_id] = asyncio.Event()
        self.cancel_requested[record.agent_id] = asyncio.Event()
        self.interrupted[record.agent_id] = False
        await self.started.put(record.agent_id)
        await self.release[record.agent_id].wait()
        cancelled = self.cancel_requested[record.agent_id].is_set()
        return SubAgentExecutionResult(
            status=SubAgentStatus.CANCELLED if cancelled else SubAgentStatus.COMPLETED,
            conversation=({"role": "assistant", "content": "Finished."},),
            context_state={},
            artifact_paths=(),
            result="Finished.",
            error=(
                SubAgentError(code="cancelled", message="Cancelled by user.") if cancelled else None
            ),
            usage={"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        requested = self.cancel_requested.get(agent_id)
        if requested is None:
            return False
        requested.set()
        self.interrupted[agent_id] = interrupted
        if interrupted:
            self.release[agent_id].set()
        return True


class _DelayedExecutor(_HoldingExecutor):
    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        requested = self.cancel_requested.get(agent_id)
        if requested is None:
            return False
        requested.set()
        self.interrupted[agent_id] = interrupted
        return True


@pytest_asyncio.fixture
async def management_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AgentService, WorkspaceRecord, str, str, str]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Management",
        created_at=_NOW,
        content="Saved Session history.",
    )
    other_session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Other",
        created_at=_NOW,
        content="Other Session history.",
    )
    monkeypatch.setattr(
        "aide.service.runtime.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        yield service, workspace, client.client_id, session_id, other_session_id
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_subagent_queries_survive_claim_release_and_remain_session_scoped(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    pool = SubAgentPool(repository, _HoldingExecutor())
    workspace.register_subagent_coordinator(pool, repository)
    record = _complete(repository, _register(repository, "Stored result"))

    claim = await workspace.claim(client_id, session_id)
    await workspace.release(client_id, session_id)

    page = service.list_subagents(client_id, workspace.workspace_id, session_id)
    detail = service.get_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    page_items = cast(list[dict[str, object]], page["items"])
    assert page_items[0]["agent_id"] == record.agent_id
    assert page_items[0]["result_preview"] == "Completed answer."
    assert detail["task"] == "Complete Stored result."
    assert detail["conversation"] == [{"role": "assistant", "content": "Full child output."}]
    assert detail["usage"] == record.usage
    assert claim.session_id == session_id

    for candidate_session in (other_session_id, session_id):
        with pytest.raises(ServiceError) as error:
            service.get_subagent(
                client_id,
                workspace.workspace_id,
                candidate_session,
                "923e4567-e89b-42d3-a456-426614174000",
            )
        assert error.value.code == "not_found"


@pytest.mark.asyncio
async def test_cancel_waits_for_child_cleanup_and_is_idempotent(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Cancelable",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id

    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not cancellation.done()
    executor.release[record.agent_id].set()
    first = await cancellation
    second = await service.cancel_subagent(
        client_id, workspace.workspace_id, session_id, record.agent_id
    )
    assert first["cancelled"] is True
    assert second["cancelled"] is True
    assert cast(dict[str, object], second["agent"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_schedule_session_subagents_support_user_queries_and_cancellation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, foreground_session_id, _other_session_id = management_case
    job_id = str(uuid4())
    session = Session.create_schedule(workspace.workspace_state, job_id)
    session.commit_agent_run(
        [{"role": "user", "content": "Schedule input"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
    )
    await session.wait_for_pending_persist()
    repository = SubAgentRecordStore(workspace.workspace_state, session.session_id)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Schedule child",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.SCHEDULE,
            job_id=job_id,
            occurrence_id=str(uuid4()),
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    page = service.list_subagents(client_id, workspace.workspace_id, session.session_id)
    assert cast(list[dict[str, object]], page["items"])[0]["agent_id"] == record.agent_id
    assert (
        service.get_subagent(
            client_id, workspace.workspace_id, session.session_id, record.agent_id
        )["source"]
        == "schedule"
    )
    with pytest.raises(ServiceError) as cross_session:
        await service.cancel_subagent(
            client_id, workspace.workspace_id, foreground_session_id, record.agent_id
        )
    assert cross_session.value.code == "not_found"
    cancellation = asyncio.create_task(
        service.cancel_subagent(
            client_id, workspace.workspace_id, session.session_id, record.agent_id
        )
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    assert (await cancellation)["cancelled"] is True


@pytest.mark.asyncio
async def test_cancel_reports_checkpoint_failure_and_can_retry_after_storage_repair(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    fail_cancelled = True

    def replace_text(path: Path, content: str) -> None:
        if fail_cancelled and json.loads(content)["status"] == "cancelled":
            raise OSError("cancel checkpoint unavailable")
        HOST_FILESYSTEM.atomic_replace_text(path, content)

    repository = SubAgentRecordStore(
        workspace.workspace_state, session_id, replace_text=replace_text
    )
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Failed cancellation checkpoint",
        task="Wait until cancellation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    try:
        with pytest.raises(ServiceError) as failure:
            await cancellation
        assert failure.value.code == "persistence_error"
        assert _require_record(repository, record.agent_id).status is SubAgentStatus.RUNNING
    finally:
        fail_cancelled = False
    repaired = await service.cancel_subagent(
        client_id, workspace.workspace_id, session_id, record.agent_id
    )
    assert repaired["cancelled"] is True


@pytest.mark.asyncio
async def test_session_delete_is_fenced_by_active_subagents_then_removes_them(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _HoldingExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Blocks delete",
        task="Wait for cleanup.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    claim = await workspace.claim(client_id, session_id)

    with pytest.raises(ServiceError) as busy:
        await workspace.delete_session(
            client_id,
            session_id,
            claim.version,
            claim.credential,
            "delete-busy",
        )
    assert busy.value.code == "session_busy"

    cancellation = asyncio.create_task(
        service.cancel_subagent(client_id, workspace.workspace_id, session_id, record.agent_id)
    )
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()
    await cancellation

    result = await workspace.delete_session(
        client_id,
        session_id,
        claim.version,
        claim.credential,
        "delete-after-cancel",
    )
    assert result["deleted"] is True
    assert repository.get(record.agent_id) is None


@pytest.mark.asyncio
async def test_schedule_job_deletion_drains_all_occurrences_before_returning(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    job_id = str(uuid4())
    job = ScheduleJob(
        job_id=job_id,
        message="Scheduled task",
        schedule=JobSchedule.every(3600),
        created_at_ms=int(_NOW.timestamp() * 1000),
        updated_at_ms=int(_NOW.timestamp() * 1000),
    )
    await workspace.schedule_service.add_user_job(job)
    records = tuple(
        pool.submit(
            title=f"Occurrence {index}",
            task="Finish the scheduled child task.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.SCHEDULE,
                job_id=job_id,
                occurrence_id=str(uuid4()),
            ),
            creator_snapshot=_snapshot(),
        )
        for index in range(2)
    )
    assert set(await asyncio.gather(executor.started.get(), executor.started.get())) == {
        record.agent_id for record in records
    }

    removal = asyncio.create_task(
        service.delete_schedule_job(client_id, workspace.workspace_id, job_id, "remove-job", {})
    )
    await asyncio.gather(
        *(
            asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
            for record in records
        )
    )
    assert not removal.done()
    for record in records:
        executor.release[record.agent_id].set()
    result = await removal

    assert result["deleted"] is True
    assert {_require_record(repository, record.agent_id).status for record in records} == {
        SubAgentStatus.CANCELLED
    }


@pytest.mark.asyncio
async def test_job_removal_fences_a_coordinator_registered_during_removal(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    job_id = str(uuid4())
    job = ScheduleJob(
        job_id=job_id,
        message="Scheduled task",
        schedule=JobSchedule.every(3600),
        created_at_ms=int(_NOW.timestamp() * 1000),
        updated_at_ms=int(_NOW.timestamp() * 1000),
    )
    await workspace.schedule_service.add_user_job(job)
    removal_started = asyncio.Event()
    allow_removal = asyncio.Event()
    remove = workspace.schedule_service.remove_user_job

    async def gated_remove(job_id: str, *, expected: ScheduleJob | None = None) -> bool:
        removal_started.set()
        await allow_removal.wait()
        return await remove(job_id, expected=expected)

    monkeypatch.setattr(workspace.schedule_service, "remove_user_job", gated_remove)
    removal = asyncio.create_task(
        service.delete_schedule_job(client_id, workspace.workspace_id, job_id, "remove-late", {})
    )
    await asyncio.wait_for(removal_started.wait(), timeout=2)
    repository = SubAgentRecordStore(workspace.workspace_state, session_id)
    pool = SubAgentPool(repository, _HoldingExecutor())
    try:
        workspace.register_subagent_coordinator(pool, repository)
        with pytest.raises(SubAgentRequestError, match="Schedule Job"):
            pool.submit(
                title="Rejected late child",
                task="Must not start during Job removal.",
                parent_run_id=_RUN_ID,
                source=SubAgentSource(
                    kind=SubAgentSourceKind.SCHEDULE,
                    job_id=job_id,
                    occurrence_id=str(uuid4()),
                ),
                creator_snapshot=_snapshot(),
            )
    finally:
        allow_removal.set()
        await removal


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["project", "service"])
async def test_cleanup_fences_all_pools_before_draining_and_respects_workspace_scope(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    tmp_path: Path,
    scope: str,
) -> None:
    service, workspace, client_id, session_id, other_session_id = management_case
    project, selected_workspace, _jobs = await service.register_project(
        client_id, workspace.workspace_path
    )
    assert selected_workspace is workspace
    other_path = tmp_path / "other-workspace"
    other_path.mkdir()
    outside_session_id = await _persist_session(
        other_path,
        home=service.agent_home,
        title="Outside removal",
        created_at=_NOW,
        content="Keep this Session.",
    )
    outside_workspace = await service.attach_workspace(client_id, other_path)
    executor = _DelayedExecutor()
    entries: list[tuple[WorkspaceRecord, SubAgentRecordStore, SubAgentPool, SubAgentRecord]] = []
    for current_workspace, current_session_id in (
        (workspace, session_id),
        (workspace, other_session_id),
        (outside_workspace, outside_session_id),
    ):
        repository = SubAgentRecordStore(current_workspace.workspace_state, current_session_id)
        pool = SubAgentPool(repository, executor)
        current_workspace.register_subagent_coordinator(pool, repository)
        record = pool.submit(
            title="Cleanup child",
            task="Wait for cleanup.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=_RESTORE_TOKEN,
            ),
            creator_snapshot=_snapshot(),
        )
        assert await executor.started.get() == record.agent_id
        entries.append((current_workspace, repository, pool, record))
    close = asyncio.create_task(
        complete_project_removal(service, client_id, project.project_id)
        if scope == "project"
        else service.stop()
    )
    first = entries[0][3]
    await asyncio.wait_for(executor.cancel_requested[first.agent_id].wait(), timeout=2)
    affected = entries[:2] if scope == "project" else entries
    try:
        assert not close.done()
        for current_workspace, repository, pool, _record in affected:
            with pytest.raises(SubAgentRequestError, match="not accepting"):
                pool.submit(
                    title="Rejected during cleanup",
                    task="Must not start.",
                    parent_run_id=_RUN_ID,
                    source=SubAgentSource(
                        kind=SubAgentSourceKind.FOREGROUND,
                        restore_run_token=_RESTORE_TOKEN,
                    ),
                    creator_snapshot=_snapshot(),
                )
            with pytest.raises(SubAgentRequestError, match="not accepting"):
                current_workspace.register_subagent_coordinator(pool, repository)
        assert service.confirmation._closed is False
        assert workspace.workspace_id in service.workspace_resources.resources
    finally:
        # Release all affected executions even if the admission assertions fail.
        for _current_workspace, _repository, _pool, record in affected:
            await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
            executor.release[record.agent_id].set()
        await close
        if scope == "project":
            executor.release[entries[2][3].agent_id].set()
    expected = SubAgentStatus.CANCELLED if scope == "project" else SubAgentStatus.INTERRUPTED
    for _current_workspace, repository, pool, record in affected:
        assert _require_record(repository, record.agent_id).status is expected
        assert not pool.has_active()
    assert workspace.workspace_id not in service.workspace_resources.resources
    if scope == "project":
        outside = entries[2]
        assert not executor.cancel_requested[outside[3].agent_id].is_set()
        result = await outside[2].wait([outside[3].agent_id])
        assert result[0].status is SubAgentStatus.COMPLETED
        assert outside_workspace.workspace_id in service.workspace_resources.resources
    else:
        assert service.confirmation._closed is True
        assert service.state == "stopped"


@pytest.mark.asyncio
async def test_service_close_marks_children_interrupted_before_closing_shared_confirmation(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, _client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Service child",
        task="Wait for Service shutdown.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id
    close = asyncio.create_task(service.stop())
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    assert not close.done()
    assert service.confirmation._closed is False
    executor.release[record.agent_id].set()
    await close

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.INTERRUPTED
    assert service.confirmation._closed is True
    assert service.state == "stopped"


@pytest.mark.asyncio
async def test_service_close_retries_failed_subagent_checkpoint_before_closing_resources(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, workspace, _client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    executor = _DelayedExecutor()
    pool = SubAgentPool(repository, executor, now=lambda: _NOW)
    workspace.register_subagent_coordinator(pool, repository)
    record = pool.submit(
        title="Retry shutdown checkpoint",
        task="Persist the interrupted state before resources close.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    assert await executor.started.get() == record.agent_id

    save = repository.save
    failures_remaining = 2

    def fail_first_interrupted_save(updated: SubAgentRecord) -> SubAgentRecord:
        nonlocal failures_remaining
        if updated.status is SubAgentStatus.INTERRUPTED and failures_remaining:
            failures_remaining -= 1
            raise SubAgentStoreError("temporary storage failure")
        return save(updated)

    monkeypatch.setattr(repository, "save", fail_first_interrupted_save)
    close = asyncio.create_task(service.stop())
    await asyncio.wait_for(executor.cancel_requested[record.agent_id].wait(), timeout=2)
    executor.release[record.agent_id].set()

    with pytest.raises(ServiceError, match="could not persist SubAgent shutdown state"):
        await close

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.RUNNING
    assert service.state == "draining"
    assert service.confirmation._closed is False
    assert workspace.workspace_id in service.workspace_resources.resources

    try:
        await service.stop()
    finally:
        if service.state != "stopped":
            pool.cancel(record.agent_id, interrupted=True)
            await service.stop()

    assert _require_record(repository, record.agent_id).status is SubAgentStatus.INTERRUPTED
    assert service.confirmation._closed is True
    assert workspace.workspace_id not in service.workspace_resources.resources
    assert service.state == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["foreground", "background"])
async def test_subagent_confirmation_routes_to_session_after_claim_release(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
    origin: Literal["foreground", "background"],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    record = _complete(repository, _register(repository, "Confirmation child"))
    claim = await workspace.claim(client_id, session_id)
    await workspace.release(client_id, session_id)
    owner = SubAgentConfirmationOwner(uuid4(), workspace.workspace_id, session_id, record.agent_id)
    request = ConfirmationRequest(
        uuid4(), "tool-call-1", "exec", "Run a command", {"command": "pytest"}
    )
    envelope = ConfirmationEnvelope(
        request=request,
        origin=origin,
        owner=owner,
        job_id="scheduled-job" if origin == "background" else None,
        title="Scheduled child" if origin == "background" else None,
    )
    client = service.client(client_id)
    sink = cast(_CollectingSink, client.sink)
    confirmation = asyncio.create_task(service.confirmation.request(envelope))
    async with asyncio.timeout(2):
        while not any(event["type"] == "confirmation.requested" for event in client.events):
            sink.changed.clear()
            await sink.changed.wait()
    emitted = next(event for event in client.events if event["type"] == "confirmation.requested")

    assert emitted["workspace_id"] == workspace.workspace_id
    assert emitted["session_id"] == session_id
    assert emitted["run_id"] is None
    payload = cast(dict[str, object], emitted["payload"])
    assert payload["origin"] == origin
    assert payload["owner"] == {
        "kind": "subagent",
        "generation_id": str(owner.generation_id),
        "workspace_id": workspace.workspace_id,
        "session_id": session_id,
        "agent_id": record.agent_id,
    }
    outsider = await service.register_client("cli")
    with pytest.raises(ServiceError) as forbidden:
        service.decide_confirmation(outsider.client_id, cast(str, payload["token"]), "approved")
    assert forbidden.value.code == "forbidden"
    service.decide_confirmation(client_id, cast(str, payload["token"]), "approved")
    assert await confirmation == "approved"
    assert claim.session_id == session_id


@pytest.mark.asyncio
async def test_subagent_http_routes_require_client_auth_and_csrf_for_cancel(
    management_case: tuple[AgentService, WorkspaceRecord, str, str, str],
) -> None:
    service, workspace, client_id, session_id, _other_session_id = management_case
    repository = SubAgentRecordStore(workspace.workspace_state, session_id, now=lambda: _NOW)
    record = _complete(repository, _register(repository, "HTTP child"))
    create_credential(service.agent_home)
    credential = read_credential(service.agent_home)
    server = TestServer(create_app(service), host="127.0.0.1")
    await server.start_server()
    headers = {
        "Authorization": f"Bearer {credential}",
        "X-Aide-Client": client_id,
    }
    path = (
        f"/api/v1/workspaces/{workspace.workspace_id}/sessions/{session_id}"
        f"/subagents/{record.agent_id}"
    )
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(
                server.make_url(
                    f"/api/v1/workspaces/{workspace.workspace_id}/sessions/{session_id}/subagents"
                ),
                headers=headers,
            ) as response:
                assert response.status == 200
                assert (await response.json())["items"][0]["agent_id"] == record.agent_id
            async with http.delete(server.make_url(path), headers=headers) as response:
                assert response.status == 403
            async with http.delete(
                server.make_url(path), headers={**headers, "X-Aide-CSRF": credential}
            ) as response:
                assert response.status == 200
                first = await response.json()
            async with http.delete(
                server.make_url(path), headers={**headers, "X-Aide-CSRF": credential}
            ) as response:
                assert response.status == 200
                second = await response.json()
    finally:
        await server.close()

    assert first["cancelled"] is False
    assert second == first


@pytest.mark.asyncio
async def test_workspace_startup_interrupts_persisted_subagent_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _prepare_agent_home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    session_id = await _persist_session(
        workspace_path,
        home=home,
        title="Startup",
        created_at=_NOW,
        content="Session history.",
    )
    repository = SubAgentRecordStore(WorkspaceState(workspace_path), session_id, now=lambda: _NOW)
    queued = _register(repository, "Queued before restart")
    running = repository.save(
        replace(
            _register(repository, "Running before restart"),
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=1,
        )
    )
    monkeypatch.setattr(
        "aide.service.runtime.create_provider", lambda *_args: _ConcurrentProvider()
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _CollectingSink())
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        recovered = workspace.subagent_repository(session_id)

        assert _require_record(recovered, queued.agent_id).status is SubAgentStatus.INTERRUPTED
        assert _require_record(recovered, running.agent_id).status is SubAgentStatus.INTERRUPTED
    finally:
        await service.stop()
