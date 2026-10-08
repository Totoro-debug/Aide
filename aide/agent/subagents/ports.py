"""Narrow boundaries shared by independent SubAgent implementation tasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol
from uuid import UUID

from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentEvent,
    SubAgentExecutionResult,
    SubAgentPage,
    SubAgentRecord,
    SubAgentSource,
    SubAgentStatus,
    SubAgentWaitResult,
)


class SubAgentRecordRepository(Protocol):
    """Persistence boundary for one Session's SubAgent records."""

    @property
    def session_id(self) -> str: ...

    def register(
        self,
        *,
        title: str,
        task: str,
        parent_run_id: str,
        source: SubAgentSource,
        creator_snapshot: SubAgentCreatorSnapshot,
    ) -> SubAgentRecord: ...

    def save(
        self, record: SubAgentRecord, *, expected_revision: int | None = None
    ) -> SubAgentRecord: ...

    def get(self, agent_id: str) -> SubAgentRecord | None: ...

    def list(
        self,
        *,
        status: SubAgentStatus | str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SubAgentPage: ...

    def discard_restore_run_tokens(self, restore_run_tokens: Sequence[str | UUID]) -> None: ...


class SubAgentExecutor(Protocol):
    """Run one registered task once and emit only SubAgent-scoped events."""

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult: ...

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool: ...


class SubAgentSessionCoordinator(Protocol):
    """Coordinate submission, listing, and terminal-result waits for one Session."""

    @property
    def session_id(self) -> str: ...

    def submit(
        self,
        *,
        title: str,
        task: str,
        parent_run_id: str,
        source: SubAgentSource,
        creator_snapshot: SubAgentCreatorSnapshot,
    ) -> SubAgentRecord: ...

    def get(self, agent_id: str) -> SubAgentRecord | None: ...

    def list(
        self,
        *,
        status: SubAgentStatus | str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SubAgentPage: ...

    async def wait(
        self,
        agent_ids: Sequence[str],
        *,
        timeout_ms: int | None = None,
    ) -> tuple[SubAgentWaitResult, ...]: ...

    def cancel(self, agent_id: str) -> bool: ...

    def close_admission(self) -> None: ...

    def open_admission(self) -> None: ...

    def block_source(self, job_id: str) -> None: ...

    def unblock_source(self, job_id: str) -> None: ...

    def has_active(self) -> bool: ...

    async def cancel_and_wait(
        self,
        agent_id: str,
        *,
        interrupted: bool = False,
    ) -> SubAgentRecord | None: ...

    async def cancel_source_and_wait(
        self,
        *,
        job_id: str,
        interrupted: bool = False,
    ) -> tuple[SubAgentRecord, ...]: ...

    async def shutdown(self, *, interrupted: bool = True) -> None: ...


class SubAgentEventPublisher(Protocol):
    """Publish an event independently of Main Agent Run output and busy state."""

    async def publish(self, event: SubAgentEvent) -> None: ...
