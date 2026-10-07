"""Session-owned SubAgent capacity, scheduling, and result waits."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from loguru import logger

from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentEvent,
    SubAgentExecutionResult,
    SubAgentPage,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
    SubAgentWaitResult,
)
from aide.agent.subagents.ports import SubAgentExecutor, SubAgentRecordRepository
from aide.agent.subagents.store import SubAgentRequestError, SubAgentStoreError
from aide.utils.validation import require_aware_datetime

SUBAGENT_POOL_CAPACITY = 8


async def _discard_event(event: SubAgentEvent) -> None:
    del event


class SubAgentPool:
    """Run up to eight SubAgents concurrently for one Session in FIFO order."""

    def __init__(
        self,
        repository: SubAgentRecordRepository,
        executor: SubAgentExecutor,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(getattr(repository, "session_id", None), str):
            raise TypeError("SubAgent Pool requires a Session-scoped record repository")
        if not callable(getattr(repository, "register", None)):
            raise TypeError("SubAgent Pool requires a record repository")
        if not callable(getattr(executor, "execute", None)) or not callable(
            getattr(executor, "request_cancel", None)
        ):
            raise TypeError("SubAgent Pool requires an executor")
        self._repository = repository
        self._executor = executor
        self._now = now or (lambda: datetime.now(UTC))
        self._queued: deque[str] = deque()
        self._active: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()
        self._interrupted_requested: set[str] = set()
        self._storage_failures: set[str] = set()
        self._blocked_job_ids: set[str] = set()
        self._waiters: set[asyncio.Event] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._accepting = True
        self._shutdown = False

    @property
    def session_id(self) -> str:
        return self._repository.session_id

    def submit(
        self,
        *,
        title: str,
        task: str,
        parent_run_id: str,
        source: SubAgentSource,
        creator_snapshot: SubAgentCreatorSnapshot,
    ) -> SubAgentRecord:
        loop = self._bind_loop()
        if not self._accepting or self._shutdown:
            raise SubAgentRequestError("SubAgent Pool is not accepting new tasks")
        if not isinstance(title, str) or not title.strip():
            raise SubAgentRequestError("SubAgent title must not be empty")
        if not isinstance(task, str) or not task.strip():
            raise SubAgentRequestError("SubAgent task must not be empty")
        if source.kind is SubAgentSourceKind.SCHEDULE and source.job_id in self._blocked_job_ids:
            raise SubAgentRequestError("Schedule Job is being removed")
        record = self._repository.register(
            title=title,
            task=task,
            parent_run_id=parent_run_id,
            source=source,
            creator_snapshot=creator_snapshot,
        )
        self._queued.append(record.agent_id)
        self._start_queued(loop)
        return record

    def get(self, agent_id: str) -> SubAgentRecord | None:
        """Read the durable checkpoint; wait reports any known persistence failure."""
        return self._repository.get(agent_id)

    def list(
        self,
        *,
        status: SubAgentStatus | str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SubAgentPage:
        """List durable checkpoints, which may predate a reported storage failure."""
        return self._repository.list(status=status, cursor=cursor, limit=limit)

    async def wait(
        self,
        agent_ids: Sequence[str],
        *,
        timeout_ms: int | None = None,
    ) -> tuple[SubAgentWaitResult, ...]:
        selected_ids = self._validate_agent_ids(agent_ids)
        if timeout_ms is not None and (
            isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 0
        ):
            raise SubAgentRequestError("SubAgent wait timeout must be a nonnegative integer")
        loop = self._bind_loop()
        self._require_all_owned(selected_ids)

        results = self._completed_results(selected_ids)
        if results or timeout_ms == 0:
            return results

        deadline = None if timeout_ms is None else loop.time() + timeout_ms / 1000
        waiter = asyncio.Event()
        self._waiters.add(waiter)
        try:
            while True:
                results = self._completed_results(selected_ids)
                if results:
                    return results
                waiter.clear()
                if deadline is None:
                    await waiter.wait()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return ()
                    try:
                        await asyncio.wait_for(waiter.wait(), timeout=remaining)
                    except TimeoutError:
                        return ()
        finally:
            self._waiters.discard(waiter)

    def cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        self._bind_loop()
        try:
            record = self._repository.get(agent_id)
        except (TypeError, ValueError) as error:
            raise SubAgentRequestError("SubAgent ID is invalid") from error
        if record is None or record.status not in {SubAgentStatus.QUEUED, SubAgentStatus.RUNNING}:
            return False
        if record.status is SubAgentStatus.QUEUED or (
            agent_id in self._storage_failures and agent_id not in self._active
        ):
            status = SubAgentStatus.INTERRUPTED if interrupted else SubAgentStatus.CANCELLED
            terminal_error = SubAgentError(
                code="service_interrupted" if interrupted else "cancelled",
                message=(
                    "The SubAgent was interrupted before execution started."
                    if interrupted
                    else "The SubAgent was cancelled before execution started."
                    if record.status is SubAgentStatus.QUEUED
                    else "The SubAgent was cancelled."
                ),
            )
            self._repository.save(
                replace(
                    record,
                    status=status,
                    finished_at=self._aware_now(),
                    error=terminal_error,
                    revision=record.revision + 1,
                )
            )
            self._storage_failures.discard(agent_id)
            self._cancel_requested.discard(agent_id)
            self._interrupted_requested.discard(agent_id)
            try:
                self._queued.remove(agent_id)
            except ValueError:
                pass
            self._signal_waiters()
            return True
        if agent_id in self._cancel_requested:
            if interrupted:
                self._interrupted_requested.add(agent_id)
                self._executor.request_cancel(agent_id, interrupted=True)
            return True
        if not self._executor.request_cancel(agent_id, interrupted=interrupted):
            return False
        self._cancel_requested.add(agent_id)
        if interrupted:
            self._interrupted_requested.add(agent_id)
        return True

    def close_admission(self) -> None:
        """Fence new registrations while an exclusive Session operation is checked."""
        self._bind_loop()
        self._accepting = False

    def open_admission(self) -> None:
        """Release a temporary Session fence unless shutdown permanently closed it."""
        self._bind_loop()
        if self._shutdown:
            raise SubAgentRequestError("SubAgent Pool is not accepting new tasks")
        self._accepting = True

    def block_source(self, job_id: str) -> None:
        """Stop future Schedule registrations for a Job whose removal has started."""
        self._bind_loop()
        if not isinstance(job_id, str) or not job_id:
            raise SubAgentRequestError("Schedule Job ID is invalid")
        self._blocked_job_ids.add(job_id)

    def unblock_source(self, job_id: str) -> None:
        """Allow Schedule registrations again after a failed Job removal."""
        self._bind_loop()
        self._blocked_job_ids.discard(job_id)

    def has_active(self) -> bool:
        """Report queued, running, or uncertain tasks that still fence Session changes."""
        return bool(self._queued or self._active or self._storage_failures)

    async def cancel_and_wait(
        self,
        agent_id: str,
        *,
        interrupted: bool = False,
    ) -> SubAgentRecord | None:
        self._bind_loop()
        record = self._repository.get(agent_id)
        if record is None:
            return None
        if record.status in {SubAgentStatus.QUEUED, SubAgentStatus.RUNNING}:
            self.cancel(agent_id, interrupted=interrupted)
        task = self._active.get(agent_id)
        if task is not None:
            await asyncio.shield(task)
        if agent_id in self._storage_failures:
            raise SubAgentStoreError("SubAgent cancellation could not be persisted safely")
        return self._repository.get(agent_id)

    async def cancel_source_and_wait(
        self,
        *,
        job_id: str,
        interrupted: bool = False,
    ) -> tuple[SubAgentRecord, ...]:
        self.block_source(job_id)
        agent_ids: list[str] = []
        cursor: str | None = None
        while True:
            page = self._repository.list(cursor=cursor, limit=100)
            for item in page.items:
                record = self._repository.get(item.agent_id)
                if (
                    record is not None
                    and record.status in {SubAgentStatus.QUEUED, SubAgentStatus.RUNNING}
                    and record.source.kind is SubAgentSourceKind.SCHEDULE
                    and record.source.job_id == job_id
                ):
                    agent_ids.append(record.agent_id)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        results = await asyncio.gather(
            *(self.cancel_and_wait(agent_id, interrupted=interrupted) for agent_id in agent_ids)
        )
        return tuple(record for record in results if record is not None)

    async def shutdown(self, *, interrupted: bool = True) -> None:
        """Stop admission and drain every task before its Workspace closes resources."""
        self._bind_loop()
        self._accepting = False
        self._shutdown = True
        shutdown_ids = set(self._queued) | set(self._active) | self._storage_failures
        errors: list[Exception] = []
        for agent_id in tuple(self._queued):
            try:
                self.cancel(agent_id, interrupted=interrupted)
            except Exception as error:
                self._storage_failures.add(agent_id)
                errors.append(error)
        active_ids = tuple(self._active)
        for agent_id in active_ids:
            try:
                self.cancel(agent_id, interrupted=interrupted)
            except Exception as error:
                errors.append(error)
        active_tasks = tuple(
            task for task in self._active.values() if task is not asyncio.current_task()
        )
        if active_tasks:
            await asyncio.gather(*active_tasks)
        for agent_id in tuple(self._storage_failures):
            try:
                record = self._repository.get(agent_id)
                if record is None or record.status not in {
                    SubAgentStatus.QUEUED,
                    SubAgentStatus.RUNNING,
                }:
                    self._storage_failures.discard(agent_id)
                else:
                    self.cancel(agent_id, interrupted=interrupted)
            except Exception as error:
                errors.append(error)
        unresolved: list[str] = []
        for agent_id in shutdown_ids | self._storage_failures:
            try:
                record = self._repository.get(agent_id)
            except Exception as error:
                errors.append(error)
                unresolved.append(agent_id)
                continue
            if record is not None and record.status in {
                SubAgentStatus.QUEUED,
                SubAgentStatus.RUNNING,
            }:
                unresolved.append(agent_id)
        if unresolved:
            shutdown_error = SubAgentStoreError(
                "A SubAgent remained active during Service shutdown: "
                + ", ".join(sorted(unresolved))
            )
            if errors:
                raise shutdown_error from ExceptionGroup(
                    "SubAgent shutdown persistence failed", errors
                )
            raise shutdown_error

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("SubAgent Pool cannot be used from another event loop")
        return loop

    def _start_queued(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._queued and not self._shutdown and len(self._active) < SUBAGENT_POOL_CAPACITY:
            agent_id = self._queued.popleft()
            execution = loop.create_task(self._run(agent_id))
            self._active[agent_id] = execution

    async def _run(self, agent_id: str) -> None:
        try:
            record = self._repository.get(agent_id)
            if record is None or record.status is not SubAgentStatus.QUEUED:
                return
            running = self._repository.save(
                replace(
                    record,
                    status=SubAgentStatus.RUNNING,
                    started_at=self._aware_now(),
                    revision=record.revision + 1,
                )
            )
            try:
                result = await self._executor.execute(running, emit=_discard_event)
                if not isinstance(result, SubAgentExecutionResult):
                    raise TypeError("SubAgent executor returned an invalid result")
            except asyncio.CancelledError:
                self._save_interrupted(agent_id)
            except SubAgentStoreError:
                raise
            except Exception:
                self._save_failure(agent_id)
            else:
                self._save_result(agent_id, result)
        except SubAgentStoreError:
            self._storage_failures.add(agent_id)
            logger.exception("SubAgent Pool could not persist task {}", agent_id)
        except Exception:
            logger.exception("SubAgent Pool could not start task {}", agent_id)
        finally:
            self._active.pop(agent_id, None)
            self._cancel_requested.discard(agent_id)
            self._interrupted_requested.discard(agent_id)
            self._signal_waiters()
            if self._loop is not None and not self._loop.is_closed():
                self._start_queued(self._loop)

    def _save_result(self, agent_id: str, result: SubAgentExecutionResult) -> None:
        current = self._repository.get(agent_id)
        if current is None or current.status is not SubAgentStatus.RUNNING:
            return
        cancellation_requested = agent_id in self._cancel_requested
        interrupted_requested = agent_id in self._interrupted_requested
        status = (
            SubAgentStatus.INTERRUPTED
            if interrupted_requested
            else SubAgentStatus.CANCELLED
            if cancellation_requested
            else result.status
        )
        error = result.error
        if interrupted_requested:
            error = SubAgentError(
                code="service_interrupted",
                message="The SubAgent was interrupted while the Service was shutting down.",
            )
        elif cancellation_requested and error is None:
            error = SubAgentError(code="cancelled", message="The SubAgent was cancelled.")
        self._repository.save(
            replace(
                current,
                status=status,
                finished_at=self._aware_now(),
                conversation=result.conversation,
                context_state=result.context_state,
                artifact_paths=result.artifact_paths,
                result=result.result,
                error=error,
                usage=result.usage,
                revision=current.revision + 1,
            )
        )

    def _save_failure(self, agent_id: str) -> None:
        current = self._repository.get(agent_id)
        if current is None or current.status is not SubAgentStatus.RUNNING:
            return
        interrupted = agent_id in self._interrupted_requested
        cancelled = agent_id in self._cancel_requested
        self._repository.save(
            replace(
                current,
                status=(
                    SubAgentStatus.INTERRUPTED
                    if interrupted
                    else SubAgentStatus.CANCELLED
                    if cancelled
                    else SubAgentStatus.FAILED
                ),
                finished_at=self._aware_now(),
                error=SubAgentError(
                    code="service_interrupted"
                    if interrupted
                    else "cancelled"
                    if cancelled
                    else "execution_failed",
                    message=(
                        "The SubAgent was interrupted while the Service was shutting down."
                        if interrupted
                        else "The SubAgent was cancelled."
                        if cancelled
                        else "The SubAgent could not complete the task."
                    ),
                ),
                revision=current.revision + 1,
            )
        )

    def _save_interrupted(self, agent_id: str) -> None:
        current = self._repository.get(agent_id)
        if current is None or current.status is not SubAgentStatus.RUNNING:
            return
        interrupted = agent_id in self._interrupted_requested
        cancelled = agent_id in self._cancel_requested
        self._repository.save(
            replace(
                current,
                status=(
                    SubAgentStatus.INTERRUPTED
                    if interrupted or not cancelled
                    else SubAgentStatus.CANCELLED
                ),
                finished_at=self._aware_now(),
                error=SubAgentError(
                    code="service_interrupted" if interrupted or not cancelled else "cancelled",
                    message=(
                        "The SubAgent was interrupted before it could finish."
                        if interrupted or not cancelled
                        else "The SubAgent was cancelled."
                    ),
                ),
                revision=current.revision + 1,
            )
        )

    def _completed_results(self, agent_ids: Sequence[str]) -> tuple[SubAgentWaitResult, ...]:
        results: list[SubAgentWaitResult] = []
        for agent_id in agent_ids:
            if agent_id in self._storage_failures:
                raise SubAgentStoreError("The SubAgent task could not be saved reliably.")
            record = self._repository.get(agent_id)
            if record is None:
                raise SubAgentRequestError("SubAgent does not belong to this Session")
            if record.status in {
                SubAgentStatus.COMPLETED,
                SubAgentStatus.FAILED,
                SubAgentStatus.CANCELLED,
                SubAgentStatus.INTERRUPTED,
            }:
                results.append(
                    SubAgentWaitResult(
                        agent_id=record.agent_id,
                        title=record.title,
                        status=record.status,
                        result=record.result,
                        error=record.error,
                        usage=record.usage or {},
                    )
                )
        return tuple(results)

    def _require_all_owned(self, agent_ids: Sequence[str]) -> None:
        for agent_id in agent_ids:
            try:
                record = self._repository.get(agent_id)
            except (TypeError, ValueError) as error:
                raise SubAgentRequestError("SubAgent ID is invalid") from error
            if record is None or record.session_id != self.session_id:
                raise SubAgentRequestError("SubAgent does not belong to this Session")

    @staticmethod
    def _validate_agent_ids(agent_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(agent_ids, (str, bytes)) or not isinstance(agent_ids, Sequence):
            raise SubAgentRequestError("SubAgent IDs must be a non-empty sequence")
        if not agent_ids:
            raise SubAgentRequestError("SubAgent IDs must be a non-empty sequence")
        selected: list[str] = []
        seen: set[str] = set()
        for agent_id in agent_ids:
            if not isinstance(agent_id, str) or not agent_id:
                raise SubAgentRequestError("SubAgent ID is invalid")
            if agent_id not in seen:
                seen.add(agent_id)
                selected.append(agent_id)
        return tuple(selected)

    def _aware_now(self) -> datetime:
        current = self._now()
        require_aware_datetime(current, field="SubAgent Pool time")
        return current

    def _signal_waiters(self) -> None:
        for waiter in tuple(self._waiters):
            waiter.set()


__all__ = ["SUBAGENT_POOL_CAPACITY", "SubAgentPool"]
