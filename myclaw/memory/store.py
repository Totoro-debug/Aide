"""Workspace-owned persistence stores for the Memory System."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from myclaw.memory.records import SummaryEntry
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.workspace.state import WorkspaceState

type AtomicReplaceBytes = Callable[[Path, bytes], None]
type AtomicReplaceText = Callable[[Path, str], None]


class WorkspaceJsonlSummaryStore:
    """Persist Conversation Summary entries in one Workspace State."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_bytes: AtomicReplaceBytes = HOST_FILESYSTEM.atomic_replace_bytes,
    ) -> None:
        self.workspace_state = workspace_state
        self._state_root = workspace_state.path.resolve(strict=True)
        self._memory_directory = workspace_state.memory_directory
        self.path = workspace_state.memory_directory / "summary.jsonl"
        self._replace_bytes = replace_bytes
        self._lock = asyncio.Lock()

    async def append(self, content: str, timestamp: datetime) -> SummaryEntry:
        async with self._lock:
            next_index = len(self._entries()) + 1
            entry = SummaryEntry(index=next_index, timestamp=timestamp, content=content)
            self._append_exact(entry)
            return entry

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]:
        async with self._lock:
            return self._entries()[cursor : cursor + limit]

    def _append_exact(self, entry: SummaryEntry) -> None:
        entries = self._entries()
        if entry.index <= len(entries):
            if entries[entry.index - 1].to_dict() != entry.to_dict():
                raise ValueError("summary index already contains a different record")
            return
        if entry.index != len(entries) + 1:
            raise ValueError("summary index must be contiguous")
        existing = self.path.read_bytes() if self.path.exists() else b""
        self._replace_bytes(self.path, existing + entry.to_json_line().encode("utf-8"))
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self._state_root)

    def _entries(self) -> tuple[SummaryEntry, ...]:
        self._require_memory_directory()
        if not self.path.exists() and not self.path.is_symlink():
            return ()
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self._state_root)
        content = self.path.read_bytes()
        if content and not content.endswith(b"\n"):
            raise ValueError("summary stream must contain complete JSONL records")
        entries: list[SummaryEntry] = []
        for expected_index, line in enumerate(content.decode("utf-8").splitlines(), start=1):
            loaded: object = json.loads(line)
            if not isinstance(loaded, dict) or set(loaded) != {
                "index",
                "timestamp",
                "content",
            }:
                raise ValueError("summary record has an invalid schema")
            index = loaded["index"]
            timestamp = loaded["timestamp"]
            summary_content = loaded["content"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("summary index must be an integer")
            if index != expected_index:
                raise ValueError("summary indexes must be contiguous from 1")
            if not isinstance(timestamp, str):
                raise ValueError("summary timestamp must be a string")
            if not isinstance(summary_content, str):
                raise ValueError("summary content must be a string")
            parsed_timestamp = datetime.fromisoformat(timestamp)
            if format_rfc3339_milliseconds(parsed_timestamp) != timestamp:
                raise ValueError("summary timestamp must use canonical RFC 3339 milliseconds")
            entries.append(
                SummaryEntry(
                    index=index,
                    timestamp=parsed_timestamp,
                    content=summary_content,
                )
            )
        return tuple(entries)

    def _require_memory_directory(self) -> Path:
        return HOST_FILESYSTEM.require_owned_directory(
            self._memory_directory, within=self._state_root
        )


class MemoryPathDeniedError(PermissionError):
    """Raised when Long-term Memory aliases or identifies another file kind."""


class _WorkspaceMemoryStore:
    def __init__(self, workspace_state: WorkspaceState) -> None:
        self.workspace_state = workspace_state
        self._state_root = workspace_state.path.resolve(strict=True)
        self._memory_directory = workspace_state.memory_directory

    def _require_private_regular_file(self, path: Path) -> None:
        self._require_private_memory_directory()
        try:
            HOST_FILESYSTEM.require_owned_regular_file(path, within=self._state_root)
        except PermissionError as error:
            raise MemoryPathDeniedError(
                "Workspace State must be an unaliased regular file"
            ) from error

    def _require_private_memory_directory(self) -> None:
        try:
            HOST_FILESYSTEM.require_owned_directory(self._memory_directory, within=self._state_root)
        except PermissionError as error:
            raise MemoryPathDeniedError(
                "Workspace State Memory directory must remain unaliased"
            ) from error


class WorkspaceLongTermMemoryStore(_WorkspaceMemoryStore):
    """Persist Long-term Memory in one Workspace State."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_text: AtomicReplaceText = HOST_FILESYSTEM.atomic_replace_text,
    ) -> None:
        super().__init__(workspace_state)
        self._path = workspace_state.long_term_memory_path
        self._replace_text = replace_text
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def read(self) -> str:
        async with self._lock:
            self._require_private_regular_file(self._path)
            return self._path.read_text(encoding="utf-8")

    async def replace(self, content: str) -> None:
        async with self._lock:
            self._require_private_regular_file(self._path)
            self._replace_text(self._path, content)

    def read_sync(self) -> str:
        """Read the startup snapshot before asynchronous Memory work begins."""
        self._require_private_regular_file(self._path)
        return self._path.read_text(encoding="utf-8")


class WorkspaceSummaryCursorStore(_WorkspaceMemoryStore):
    """Persist the Summary Cursor in one Workspace State."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_text: AtomicReplaceText = HOST_FILESYSTEM.atomic_replace_text,
    ) -> None:
        super().__init__(workspace_state)
        self._path = workspace_state.memory_directory / ".cursor"
        self._replace_text = replace_text
        self._lock = asyncio.Lock()

    def _require_private_cursor_location(self) -> None:
        self._require_private_memory_directory()
        if self._path.exists() or self._path.is_symlink():
            self._require_private_regular_file(self._path)

    async def read(self) -> int:
        async with self._lock:
            self._require_private_cursor_location()
            if not self._path.exists():
                return 0
            content = self._path.read_bytes()
            digits = content[:-1]
            if not content.endswith(b"\n") or not digits.isdigit():
                raise ValueError("Summary Cursor must use canonical ASCII decimal text")
            index = int(digits)
            if digits != str(index).encode("ascii"):
                raise ValueError("Summary Cursor must use canonical ASCII decimal text")
            return index

    async def write(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Summary Cursor must be a nonnegative integer")
        async with self._lock:
            self._require_private_cursor_location()
            self._replace_text(self._path, f"{index}\n")
