"""Session-owned identity and title work shared by transient Run executors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from loguru import logger

from omni.agent.session.session import Session
from omni.logging.session import session_log
from omni.utils.text import normalize_title


@dataclass(slots=True)
class TitleCoordination:
    preparation_started: asyncio.Event
    prepared: asyncio.Future[bool]
    log_ready: asyncio.Event
    foreground_idle: asyncio.Event
    active_foregrounds: int = 0

    def attach_foreground(self) -> None:
        self.active_foregrounds += 1
        self.foreground_idle.clear()

    def release_foreground(self) -> None:
        self.active_foregrounds -= 1
        if self.active_foregrounds == 0:
            self.foreground_idle.set()

    async def wait_until_foreground_idle(self) -> None:
        while True:
            await self.foreground_idle.wait()
            await asyncio.sleep(0)
            if self.active_foregrounds == 0:
                return


@dataclass(frozen=True, slots=True)
class TitleWork:
    task: asyncio.Task[None]
    coordination: TitleCoordination


class SessionRunState:
    """Keep title creation, draining, and cancellation with the resident Session."""

    def __init__(self, generation_id: UUID) -> None:
        self._generation_id = generation_id
        self._title_work: dict[str, TitleWork] = {}

    @property
    def generation_id(self) -> UUID:
        return self._generation_id

    @property
    def title_tasks(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(work.task for work in self._title_work.values())

    def title_for(self, session_id: str) -> TitleWork | None:
        return self._title_work.get(session_id)

    def clear_title_work(self) -> None:
        self._title_work.clear()

    def cancel_title_work(self) -> None:
        for task in self.title_tasks:
            if not task.done():
                task.cancel()

    async def finish_title_work(self) -> None:
        tasks = self.title_tasks
        self.cancel_title_work()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.clear_title_work()

    async def wait_for_title_idle(self, persist: Callable[[], Awaitable[None]]) -> None:
        while True:
            title_tasks = tuple(
                work.task for work in self._title_work.values() if not work.task.done()
            )
            if title_tasks:
                for title_task in title_tasks:
                    await asyncio.shield(title_task)
                continue
            await persist()
            if not any(not work.task.done() for work in self._title_work.values()):
                return

    async def discard_uncommitted_title(self, session_id: str, work: TitleWork) -> None:
        if not work.task.done():
            work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
        if self._title_work.get(session_id) is work:
            self._title_work.pop(session_id)

    def start_title(
        self,
        session: Session,
        content: str,
        resolve_title: Callable[[str], Awaitable[tuple[str, dict[str, int] | None]]],
        is_aborted: Callable[[], bool],
    ) -> TitleWork | None:
        if (
            is_aborted()
            or not content.strip()
            or session.metadata.get("title") != "Untitled session"
            or session.has_manual_title
            or session.session_id in self._title_work
        ):
            return None
        preparation_started = asyncio.Event()
        prepared: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        log_ready = asyncio.Event()
        foreground_idle = asyncio.Event()
        foreground_idle.set()
        coordination = TitleCoordination(
            preparation_started=preparation_started,
            prepared=prepared,
            log_ready=log_ready,
            foreground_idle=foreground_idle,
        )
        task = asyncio.create_task(
            self._generate_title(
                session,
                content,
                coordination=coordination,
                resolve_title=resolve_title,
                is_aborted=is_aborted,
            )
        )
        work = TitleWork(
            task=task,
            coordination=coordination,
        )
        self._title_work[session.session_id] = work
        task.add_done_callback(self._title_done)
        return work

    def _title_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.opt(exception=error).error(
                "Session title task failed type={}", type(error).__name__
            )

    async def _generate_title(
        self,
        session: Session,
        content: str,
        *,
        coordination: TitleCoordination,
        resolve_title: Callable[[str], Awaitable[tuple[str, dict[str, int] | None]]],
        is_aborted: Callable[[], bool],
    ) -> None:
        with session_log(session):
            coordination.log_ready.set()
            try:
                await coordination.preparation_started.wait()
                title, usage_delta = await resolve_title(content)
                if (
                    await coordination.prepared
                    and not session.has_manual_title
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_automatic_title(title, usage_delta=usage_delta)
            except asyncio.CancelledError:
                if (
                    not is_aborted()
                    and coordination.prepared.done()
                    and coordination.prepared.result()
                    and not session.has_manual_title
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_automatic_title(normalize_title(content))
                raise
            finally:
                await coordination.wait_until_foreground_idle()
