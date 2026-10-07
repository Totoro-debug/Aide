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
        self._storage_failures: set[str] = set()
        self._waiters: set[asyncio.Event] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

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
        if not isinstance(title, str) or not title.strip():
            raise SubAgentRequestError("SubAgent title must not be empty")
        if not isinstance(task, str) or not task.strip():
            raise SubAgentRequestError("SubAgent task must not be empty")
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

    def cancel(self, agent_id: str) -> bool:
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
            self._repository.save(
                replace(
                    record,
                    status=SubAgentStatus.CANCELLED,
                    finished_at=self._aware_now(),
                    error=SubAgentError(
                        code="cancelled",
                        message=(
                            "The SubAgent was cancelled before execution started."
                            if record.status is SubAgentStatus.QUEUED
                            else "The SubAgent was cancelled."
                        ),
                    ),
                    revision=record.revision + 1,
                )
            )
            self._storage_failures.discard(agent_id)
            try:
                self._queued.remove(agent_id)
            except ValueError:
                pass
            self._signal_waiters()
            return True
        if agent_id in self._cancel_requested:
            return True
        if not self._executor.request_cancel(agent_id):
            return False
        self._cancel_requested.add(agent_id)
        return True

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("SubAgent Pool cannot be used from another event loop")
        return loop

    def _start_queued(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._queued and len(self._active) < SUBAGENT_POOL_CAPACITY:
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
            self._signal_waiters()
            if self._loop is not None and not self._loop.is_closed():
                self._start_queued(self._loop)

    def _save_result(self, agent_id: str, result: SubAgentExecutionResult) -> None:
        current = self._repository.get(agent_id)
        if current is None or current.status is not SubAgentStatus.RUNNING:
            return
        cancellation_requested = agent_id in self._cancel_requested
        status = SubAgentStatus.CANCELLED if cancellation_requested else result.status
        error = result.error
        if cancellation_requested and error is None:
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
        self._repository.save(
            replace(
                current,
                status=(
                    SubAgentStatus.CANCELLED
                    if agent_id in self._cancel_requested
                    else SubAgentStatus.FAILED
                ),
                finished_at=self._aware_now(),
                error=SubAgentError(
                    code=(
                        "cancelled" if agent_id in self._cancel_requested else "execution_failed"
                    ),
                    message=(
                        "The SubAgent was cancelled."
                        if agent_id in self._cancel_requested
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
        cancelled = agent_id in self._cancel_requested
        self._repository.save(
            replace(
                current,
                status=SubAgentStatus.CANCELLED if cancelled else SubAgentStatus.INTERRUPTED,
                finished_at=self._aware_now(),
                error=SubAgentError(
                    code="cancelled" if cancelled else "service_interrupted",
                    message=(
                        "The SubAgent was cancelled."
                        if cancelled
                        else "The SubAgent was interrupted before it could finish."
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
