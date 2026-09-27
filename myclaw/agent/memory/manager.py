"""Persistence-oriented Memory Manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from myclaw.agent.memory.records import SummaryEntry
from myclaw.agent.memory.store import (
    MemoryPathDeniedError,
    WorkspaceJsonlSummaryStore,
    WorkspaceLongTermMemoryStore,
    WorkspaceSummaryCursorStore,
)
from myclaw.workspace.state import WorkspaceState


@dataclass(frozen=True, slots=True)
class SummaryClaim:
    """One cursor publication and the Summary entries claimed by it."""

    previous_cursor: int
    cursor: int
    entries: tuple[SummaryEntry, ...]


class SummaryClaimError(Exception):
    """A persistence failure while reading, selecting, or publishing a claim."""

    def __init__(self, *, cursor: int, phase: str, cause: Exception) -> None:
        self.cursor = cursor
        self.phase = phase
        self.cause = cause
        super().__init__(str(cause))
        self.__cause__ = cause


class MemoryEditMismatchError(ValueError):
    """Raised when an exact Long-term Memory edit does not match once."""


class MemoryEditReadError(Exception):
    """Raised when Long-term Memory cannot be read for an edit."""


class MemoryEditWriteError(Exception):
    """Raised when a successful Long-term Memory edit cannot be persisted."""


class MemoryManager:
    """Own Long-term Memory, Summary, Cursor, and runtime snapshot persistence."""

    def __init__(self, workspace_state: WorkspaceState) -> None:
        self.workspace_state = workspace_state
        self._summary_store = WorkspaceJsonlSummaryStore(workspace_state)
        self._cursor_store = WorkspaceSummaryCursorStore(workspace_state)
        self._long_term_store = WorkspaceLongTermMemoryStore(workspace_state)
        self._snapshot = self._long_term_store.read_sync()

    @property
    def long_term_path(self) -> Path:
        return self._long_term_store.path

    async def append_summary(self, content: str, timestamp: datetime) -> SummaryEntry:
        return await self._summary_store.append(content, timestamp)

    async def claim_summaries(self, limit: int) -> SummaryClaim:
        try:
            previous_cursor = await self._cursor_store.read()
        except (OSError, UnicodeError, ValueError) as error:
            raise SummaryClaimError(cursor=0, phase="read", cause=error) from error
        try:
            entries = await self._summary_store.after(previous_cursor, limit)
        except (OSError, UnicodeError, ValueError) as error:
            raise SummaryClaimError(
                cursor=previous_cursor,
                phase="read_summaries",
                cause=error,
            ) from error
        if not entries:
            return SummaryClaim(
                previous_cursor=previous_cursor,
                cursor=previous_cursor,
                entries=(),
            )
        cursor = entries[-1].index
        try:
            await self._cursor_store.write(cursor)
        except (OSError, UnicodeError, ValueError) as error:
            raise SummaryClaimError(
                cursor=previous_cursor,
                phase="write_cursor",
                cause=error,
            ) from error
        return SummaryClaim(
            previous_cursor=previous_cursor,
            cursor=cursor,
            entries=tuple(entries),
        )

    async def read_long_term(self) -> str:
        return await self._long_term_store.read()

    async def edit_long_term(
        self,
        *,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> str:
        try:
            content = await self._long_term_store.read()
        except MemoryPathDeniedError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise MemoryEditReadError from error
        match_count = content.count(old)
        if not old or match_count == 0 or (not replace_all and match_count != 1):
            raise MemoryEditMismatchError(
                "The requested Long-term Memory text did not match precisely."
            )
        replacement = content.replace(old, new, -1 if replace_all else 1)
        try:
            await self._long_term_store.replace(replacement)
        except MemoryPathDeniedError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise MemoryEditWriteError from error
        self._snapshot = replacement
        return replacement

    def memory_snapshot(self) -> str:
        return self._snapshot


__all__ = [
    "MemoryEditMismatchError",
    "MemoryEditReadError",
    "MemoryEditWriteError",
    "MemoryManager",
    "MemoryPathDeniedError",
    "SummaryClaim",
    "SummaryClaimError",
]
