"""Resident Session coordination, with execution collaborators created per Run."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from uuid import UUID, uuid4

from omni.agent.loop import (
    AgentRunExecutor,
    ForegroundConversationProjection,
    _project_terminal_message,
)
from omni.agent.message_bus import InboundMessage, MessageBus
from omni.agent.session.execution_state import SessionRunState
from omni.agent.session.session import Session
from omni.management.service import RuntimeStatusInput
from omni.provider.session_configuration import SessionModelConfiguration
from omni.schedule.model import ScheduleJob
from omni.schedule.service import ScheduleOccurrence
from omni.skills.catalog import LoadedSkill, SkillMetadata


class SessionExecution:
    """Keep Session authority and barriers independently of any Client or Run."""

    def __init__(
        self,
        session: Session,
        bus: MessageBus,
        create_executor: Callable[[SessionRunState], AgentRunExecutor],
        reload_skills: Callable[[], tuple[SkillMetadata, ...]],
        status_input: Callable[[], RuntimeStatusInput],
        *,
        run_state: SessionRunState | None = None,
    ) -> None:
        self.session = session
        self._run_state = SessionRunState(uuid4()) if run_state is None else run_state
        self._bus = bus
        self._create_executor: Callable[[], AgentRunExecutor] = lambda: create_executor(self._run_state)
        self._reload_skills = reload_skills
        self._status_input = status_input
        self._active: AgentRunExecutor | None = None
        self._pending_flushes: set[asyncio.Task[None]] = set()
        self._flush_errors: list[Exception] = []
        self._replacement_barrier_held = False
        self._closed = False

    @property
    def generation_id(self) -> UUID:
        return self._run_state.generation_id

    @property
    def has_active_run(self) -> bool:
        return self._active is not None

    @property
    def active_model_configuration(self) -> SessionModelConfiguration | None:
        return None if self._active is None else self._active.run_model_configuration

    def foreground_input_admitted(self) -> bool:
        return not (self._closed or self._replacement_barrier_held)

    async def run_foreground(self, inbound: InboundMessage) -> None:
        if not self.foreground_input_admitted() or self._active is not None:
            raise RuntimeError("Session is unavailable for execution")
        executor = self._create_executor()
        self._active = executor
        try:
            await executor.start()
            await executor.run_foreground(inbound)
        finally:
            self._active = None
            flush = asyncio.create_task(executor.wait_for_restore_idle())
            self._pending_flushes.add(flush)
            flush.add_done_callback(self._flush_done)

    def _flush_done(self, task: asyncio.Task[None]) -> None:
        self._pending_flushes.discard(task)
        if not task.cancelled():
            error = task.exception()
            if isinstance(error, Exception):
                self._flush_errors.append(error)

    async def run_schedule_job(
        self, job: ScheduleJob, occurrence: ScheduleOccurrence | None = None
    ) -> None:
        if self._closed or self._active is not None:
            raise RuntimeError("Schedule Session is unavailable for execution")
        executor = self._create_executor()
        self._active = executor
        try:
            await executor.start()
            await executor.run_schedule_job(job, occurrence)
        finally:
            self._active = None
            await self.session.wait_for_pending_persist()

    async def cancel_active_run(self) -> None:
        active = self._active
        if active is not None:
            await active.cancel_active_run()

    async def wait_for_restore_idle(self) -> None:
        while self._pending_flushes:
            pending = tuple(self._pending_flushes)
            await asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True)
            self._pending_flushes.difference_update(pending)
        await self.session.wait_for_pending_persist()
        if self._flush_errors:
            errors, self._flush_errors = self._flush_errors, []
            raise ExceptionGroup("Session flush failed", errors)

    async def finish_work(self) -> None:
        """Cancel departing work and flush without closing the resident Session."""
        await self.cancel_active_run()
        await self._run_state.finish_title_work()
        await self.wait_for_restore_idle()
        await self.session.persist_pending_automatic_title()

    def cancel_title_work(self) -> None:
        """Release natural title waiters before shutdown acquires Workspace locks."""
        self._run_state.cancel_title_work()

    async def close(self) -> None:
        self._closed = True
        await self.finish_work()
        self.session.close()
        await self.session.wait_for_pending_persist()

    async def abort(self) -> None:
        self._closed = True
        if self._active is not None:
            await self._active.abort()
        await self._run_state.finish_title_work()
        for task in self._pending_flushes:
            task.cancel()
        await asyncio.gather(*self._pending_flushes, return_exceptions=True)
        self._pending_flushes.clear()
        self.session.abandon()

    async def _pause_for_replacement(self) -> None:
        if self._replacement_barrier_held:
            raise RuntimeError("Session Restore barrier is already held")
        self._replacement_barrier_held = True
        await self._bus.pause_inbound_delivery()

    async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
        self._replacement_barrier_held = False
        if resume_inbound:
            await self._bus.resume_inbound_delivery()

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return ForegroundConversationProjection(
            session_id=self.session.session_id,
            messages=tuple(_project_terminal_message(message) for message in self.session.messages),
        )

    def runtime_status_input(self) -> RuntimeStatusInput:
        return replace(
            self._status_input(), active_model_configuration=self.active_model_configuration
        )

    def reload_skill(self) -> tuple[SkillMetadata, ...]:
        return self._reload_skills()

    def _validate_model_context_budget(self, skills: tuple[LoadedSkill, ...]) -> None:
        self._create_executor()._validate_model_context_budget(skills)
