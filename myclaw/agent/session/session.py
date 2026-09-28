"""Stateful Conversation Session public interface."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast
from uuid import UUID, uuid4

from myclaw.agent.context.budget import ContextUsageSnapshot
from myclaw.agent.tools.base import ArtifactReference
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.text import normalize_title as _normalize_title
from myclaw.utils.text import normalize_title_candidate
from myclaw.utils.time import format_rfc3339_milliseconds, local_now
from myclaw.utils.validation import (
    require_aware_datetime,
    require_nonnegative_int,
    require_uuid4,
    require_uuid4_string,
    token_usage_validation_issue,
)

__all__ = [
    "RestoreAnchor",
    "Session",
    "SessionRestoreBefore",
    "SessionRestoreResult",
    "SessionStoragePartition",
]


class SessionStoragePartition(StrEnum):
    """The Workspace-owned storage partition used by one Conversation Session."""

    FOREGROUND = "foreground"
    SCHEDULE = "schedule"


_HEADER_FIELDS = frozenset({"session_id", "created_at", "updated_at", "last_compacted", "metadata"})
_TOKEN_USAGE_PATCH_KEYS = frozenset({"token_usage", "token_usage_delta", "usage_delta"})
_RESTORE_MESSAGE_FIELDS = frozenset({"restore_anchor_id", "restore_run_token", "restore_before"})
_RESTORE_BEFORE_FIELDS = frozenset({"metadata", "last_compacted"})
_RESTORE_NEXT_ANCHOR_ID = "restore_next_anchor_id"
_SESSION_ID_PATTERN = re.compile(
    r"(?P<timestamp>\d{8}-\d{6}-\d{6})_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_SCHEDULE_SESSION_ID_PATTERN = re.compile(
    r"schedule_(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})"
)


@dataclass(frozen=True, slots=True)
class SessionRestoreBefore:
    """Detached Session-owned state captured immediately before one input."""

    metadata: dict[str, Any]
    last_compacted: int


@dataclass(frozen=True, slots=True)
class RestoreAnchor:
    """Presentation-safe details for one persisted foreground User message."""

    anchor_id: int
    run_token: UUID
    content: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class SessionRestoreResult:
    """Result of one strict durable Session truncation."""

    session_id: str
    anchor_id: int
    removed_messages: int
    updated_at: datetime


class Session:
    """Own the in-memory state and identity of one Conversation Session."""

    _workspace_state: WorkspaceState
    _session_id: str
    _storage_partition: SessionStoragePartition
    _created_at: datetime
    _updated_at: datetime
    _now: Callable[[], datetime] | None
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    last_compacted: int
    _pending_persist: asyncio.Task[None] | None
    _persist_tasks: set[asyncio.Task[None]]
    _closed: bool
    _abandoned: bool

    def __init__(self) -> None:
        raise TypeError("Use Session.create() or Session.load()")

    @classmethod
    def _from_state(
        cls,
        *,
        workspace_state: WorkspaceState,
        session_id: str,
        created_at: datetime,
        updated_at: datetime,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
        last_compacted: int,
        partition: SessionStoragePartition | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        resolved_partition = _resolve_partition(session_id, partition)
        require_aware_datetime(created_at, field="created_at")
        require_aware_datetime(updated_at, field="updated_at")
        require_nonnegative_int(last_compacted, field="last_compacted")
        _validate_restore_sequence(messages, metadata, partition=resolved_partition)
        session = object.__new__(cls)
        session._workspace_state = workspace_state
        session._session_id = session_id
        session._storage_partition = resolved_partition
        session._created_at = created_at
        session._updated_at = updated_at
        session._now = now
        session.messages = messages
        session.metadata = metadata
        session.last_compacted = last_compacted
        session._pending_persist = None
        session._persist_tasks = set()
        session._closed = False
        session._abandoned = False
        return session

    @classmethod
    def create(
        cls,
        workspace_state: WorkspaceState,
        *,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
        partition: SessionStoragePartition = SessionStoragePartition.FOREGROUND,
        job_id: UUID | str | None = None,
    ) -> Self:
        """Create a memory-only Session in the requested storage partition."""
        created_at = _clock_now(now)
        resolved_partition = _coerce_partition(partition)
        if resolved_partition is SessionStoragePartition.SCHEDULE:
            if job_id is None:
                raise ValueError("Schedule Session requires a canonical UUID4 job_id")
            session_id = cls.schedule_session_id(job_id)
        else:
            if job_id is not None:
                raise ValueError("job_id is only valid for Schedule Sessions")
            allocate_uuid = uuid4 if new_uuid is None else new_uuid
            session_id = _make_id(created_at, allocate_uuid())
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=session_id,
            created_at=created_at,
            updated_at=created_at,
            messages=[],
            metadata=_initial_metadata(),
            last_compacted=0,
            partition=resolved_partition,
            now=now,
        )

    @classmethod
    def create_schedule(
        cls,
        workspace_state: WorkspaceState,
        job_id: UUID | str,
        *,
        now: Callable[[], datetime] | None = None,
        title: str = "Untitled session",
    ) -> Self:
        """Create a memory-only Schedule Session derived from one Job UUID4."""
        session = cls.create(
            workspace_state,
            now=now,
            partition=SessionStoragePartition.SCHEDULE,
            job_id=job_id,
        )
        session.update_metadata(title=title)
        return session

    @classmethod
    def load(
        cls,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        partition: SessionStoragePartition | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        """Load one current-format Session synchronously from Workspace State."""
        resolved_partition = _resolve_partition(session_id, partition)
        sessions_directory = _existing_sessions_directory(workspace_state, resolved_partition)
        if sessions_directory is None:
            raise FileNotFoundError(_storage_directory(workspace_state, resolved_partition))
        path = sessions_directory / f"{session_id}.jsonl"
        owned_path = HOST_FILESYSTEM.require_owned_regular_file(
            path,
            within=sessions_directory,
        )
        records = _read_jsonl_records(owned_path)
        if not records:
            raise ValueError("Session must contain a header record")
        try:
            header = records[0]
            loaded_id, created_at, updated_at, last_compacted, metadata = _parse_header(header)
            if loaded_id != session_id:
                raise ValueError("Session metadata ID does not match its file name")
            messages = [_parse_message(record) for record in records[1:]]
        except TypeError as error:
            raise ValueError("Session JSONL contains malformed persisted data") from error
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=loaded_id,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
            metadata=metadata,
            last_compacted=last_compacted,
            partition=resolved_partition,
            now=now,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def workspace_state(self) -> WorkspaceState:
        return self._workspace_state

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    def capture_restore_before(self) -> SessionRestoreBefore:
        """Capture detached Session state before a foreground User input."""
        self._ensure_not_abandoned()
        self._require_foreground_restore()
        return SessionRestoreBefore(
            metadata=copy.deepcopy(self.metadata),
            last_compacted=self.last_compacted,
        )

    def restore_candidates(self) -> tuple[RestoreAnchor, ...]:
        """Return the persisted foreground User messages that can be restored."""
        self._ensure_not_abandoned()
        self._require_foreground_restore()
        candidates: list[RestoreAnchor] = []
        for message in self._persisted_restore_messages():
            _validate_message(message)
            fields = _restore_anchor_fields(message)
            if fields is None:
                continue
            anchor_id, run_token, _ = fields
            candidates.append(
                RestoreAnchor(
                    anchor_id=anchor_id,
                    run_token=UUID(run_token),
                    content=message["content"],
                    timestamp=message["timestamp"],
                )
            )
        return tuple(candidates)

    def commit_agent_run(
        self,
        messages: list[dict[str, Any]],
        *,
        pending_last_compacted: int,
        pending_action_summary: str | None,
        usage_delta: dict[str, int] | None = None,
        metadata_updates: dict[str, Any] | None = None,
        metadata_removals: tuple[str, ...] = (),
        restore_before: SessionRestoreBefore | None = None,
        restore_run_token: UUID | None = None,
    ) -> None:
        """Atomically publish one Agent Run terminal increment."""
        self._ensure_not_abandoned()
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        require_nonnegative_int(pending_last_compacted, field="pending_last_compacted")
        restore_snapshot = _copy_restore_before(restore_before)
        if (restore_snapshot is None) != (restore_run_token is None):
            raise ValueError("restore_before and restore_run_token must be supplied together")
        if restore_snapshot is not None:
            self._require_foreground_restore()
            assert restore_run_token is not None
            require_uuid4(restore_run_token, field="restore_run_token")
        anchor_id: int | None = None

        if pending_action_summary is None:
            action_summary = ""
        else:
            _validate_action_summary(pending_action_summary, field="pending_action_summary")
            action_summary = pending_action_summary

        copied_updates = _copy_metadata_updates(metadata_updates)
        removals = _validate_metadata_removals(metadata_removals)
        _validate_agent_run_metadata_patch(copied_updates, removals)
        _normalize_blackboard_metadata(copied_updates, invalid_is_absent=False)

        copied_usage_delta: dict[str, Any] | None = None
        if usage_delta is not None:
            if not isinstance(usage_delta, dict):
                raise TypeError("usage_delta must be a dictionary")
            copied_usage_delta = _copy_json_object(usage_delta, field="usage_delta")
            _validate_token_usage(copied_usage_delta, field="usage_delta")

        candidate_metadata = copy.deepcopy(self.metadata)
        _validate_metadata(candidate_metadata)
        candidate_metadata.update(copied_updates)
        for key in removals:
            candidate_metadata.pop(key, None)
        candidate_metadata["summary"] = action_summary
        _validate_metadata(candidate_metadata)

        updated_usage = copy.deepcopy(candidate_metadata["token_usage"])
        if copied_usage_delta is not None:
            updated_usage = _accumulate_token_usage(updated_usage, copied_usage_delta)

        candidate_messages = copy.deepcopy(self.messages)
        for index, record in enumerate(candidate_messages):
            if not isinstance(record, dict):
                raise TypeError(f"messages[{index}] must be a dictionary")
            _validate_message(record)

        for index, record in enumerate(messages):
            if not isinstance(record, dict):
                raise TypeError(f"messages[{index}] must be a dictionary")
            copied = _copy_json_object(record, field="message")
            if "timestamp" in copied:
                raise ValueError("timestamp is reserved for Session message timestamps")
            copied["timestamp"] = format_rfc3339_milliseconds(self._clock_now())
            if _RESTORE_MESSAGE_FIELDS.intersection(copied):
                raise ValueError("restore anchor fields are Session-owned")
            if restore_snapshot is not None and copied["role"] == "user":
                if anchor_id is not None:
                    raise ValueError("one Agent Run may commit only one User restore anchor")
                anchor_id = _next_restore_anchor_id(candidate_metadata, candidate_messages)
                copied.update(
                    {
                        "restore_anchor_id": anchor_id,
                        "restore_run_token": str(restore_run_token),
                        "restore_before": {
                            "metadata": copy.deepcopy(restore_snapshot.metadata),
                            "last_compacted": restore_snapshot.last_compacted,
                        },
                    }
                )
            try:
                _validate_message(copied)
            except KeyError as error:
                raise ValueError(f"Session message is missing {error.args[0]}") from error
            candidate_messages.append(copied)
            if copied["role"] == "assistant":
                updated_usage = _accumulate_token_usage(updated_usage, copied["token_usage"])

        if restore_snapshot is not None:
            if anchor_id is None:
                raise ValueError("restore anchor requires one User message")
            candidate_metadata[_RESTORE_NEXT_ANCHOR_ID] = anchor_id + 1

        if pending_last_compacted > len(candidate_messages):
            raise ValueError("pending_last_compacted must not exceed final message count")
        candidate_metadata["token_usage"] = updated_usage
        _validate_metadata(candidate_metadata)

        candidate_state = self.__dict__.copy()
        candidate_state.update(
            messages=candidate_messages,
            metadata=candidate_metadata,
            last_compacted=pending_last_compacted,
        )
        self.__dict__ = candidate_state
        self.persist()

    def restore_before_durably(self, anchor_id: int) -> SessionRestoreResult:
        """Strictly persist the Session state immediately before ``anchor_id``."""
        self._ensure_not_abandoned()
        self._require_foreground_restore()
        _validate_restore_anchor_id(anchor_id)
        if any(not task.done() for task in self._persist_tasks):
            raise RuntimeError("Pending Session snapshots must finish before restore")
        if not any(
            _restore_anchor_fields(message) is not None
            and message["restore_anchor_id"] == anchor_id
            for message in self._persisted_restore_messages()
        ):
            raise ValueError(f"unknown persisted restore anchor ID: {anchor_id}")

        anchor_index: int | None = None
        anchor_before: SessionRestoreBefore | None = None
        for index, message in enumerate(self.messages):
            _validate_message(message)
            fields = _restore_anchor_fields(message)
            if fields is None or fields[0] != anchor_id:
                continue
            if anchor_index is not None:
                raise ValueError(f"duplicate restore anchor ID: {anchor_id}")
            anchor_index = index
            anchor_before = fields[2]
        if anchor_index is None or anchor_before is None:
            raise ValueError(f"unknown restore anchor ID: {anchor_id}")

        retained_messages = copy.deepcopy(self.messages[:anchor_index])
        if anchor_before.last_compacted > len(retained_messages):
            raise ValueError("restore_before.last_compacted exceeds retained message count")
        removed_messages = len(self.messages) - len(retained_messages)
        current_next_id = _next_restore_anchor_id(self.metadata, self.messages)
        restored_next_id = _next_restore_anchor_id(anchor_before.metadata, retained_messages)
        restored_metadata = copy.deepcopy(anchor_before.metadata)
        restored_metadata[_RESTORE_NEXT_ANCHOR_ID] = max(
            current_next_id,
            restored_next_id,
            anchor_id + 1,
        )
        _validate_metadata(restored_metadata)

        restored_at = self._clock_now()
        content = _serialize_session_state(
            session_id=self._session_id,
            created_at=self._created_at,
            updated_at=restored_at,
            last_compacted=anchor_before.last_compacted,
            metadata=restored_metadata,
            messages=retained_messages,
        )
        candidate_state = self.__dict__.copy()
        candidate_state.update(
            messages=retained_messages,
            metadata=restored_metadata,
            last_compacted=anchor_before.last_compacted,
            _updated_at=restored_at,
        )
        self._write_content(content)
        self.__dict__ = candidate_state
        return SessionRestoreResult(
            session_id=self._session_id,
            anchor_id=anchor_id,
            removed_messages=removed_messages,
            updated_at=restored_at,
        )

    def update_metadata(self, metadata: dict[str, Any] | None = None, **updates: Any) -> None:
        """Apply a copied shallow metadata patch and accumulate token usage deltas."""
        self._ensure_not_abandoned()
        patch: dict[str, Any] = {}
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata patch must be a dictionary")
            patch.update(metadata)
        patch.update(updates)
        copied_patch = _copy_json_object(patch, field="metadata")
        if _RESTORE_NEXT_ANCHOR_ID in copied_patch:
            raise ValueError("restore anchor counter is Session-owned")
        _normalize_blackboard_metadata(copied_patch, invalid_is_absent=False)

        token_delta = copied_patch.pop("token_usage_delta", None)
        if token_delta is None and "usage_delta" in copied_patch:
            token_delta = copied_patch.pop("usage_delta")
        if "token_usage" in copied_patch:
            if token_delta is not None:
                raise ValueError("token usage delta was provided more than once")
            token_delta = copied_patch.pop("token_usage")
        if token_delta is not None:
            _validate_token_usage(token_delta, field="token_usage_delta")

        if "title" in copied_patch:
            title = copied_patch["title"]
            if not isinstance(title, str):
                raise TypeError("title must be a string")
            copied_patch["title"] = _normalize_title(title)
        if "summary" in copied_patch:
            _validate_action_summary(copied_patch["summary"], field="metadata.summary")

        updated_usage = self._usage_after_delta(token_delta)
        self.metadata.update(copied_patch)
        if updated_usage is not None:
            self.metadata["token_usage"] = updated_usage

    def persist(self) -> None:
        """Schedule a silent, ordered write of the current complete Session snapshot."""
        if self._closed:
            return
        self._updated_at = self._clock_now()
        if not self.messages:
            return
        try:
            content = self._serialized_state()
            loop = asyncio.get_running_loop()
            previous = self._pending_persist
            pending = loop.create_task(self._persist_after(previous, content))
            self._pending_persist = pending
            self._persist_tasks.add(pending)
            pending.add_done_callback(self._persist_task_finished)
        except Exception:
            return

    async def wait_for_pending_persist(self) -> None:
        """Wait for every already-scheduled ordered snapshot without starting a new save."""
        drain = asyncio.create_task(self._drain_pending_persist())
        await await_task_preserving_cancellation(drain)

    async def _drain_pending_persist(self) -> None:
        while True:
            done_tasks = tuple(task for task in self._persist_tasks if task.done())
            for task in done_tasks:
                self._consume_persist_task(task)
            pending_tasks = tuple(self._persist_tasks)
            if not pending_tasks:
                return
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    def _persist_task_finished(self, task: asyncio.Task[None]) -> None:
        self._consume_persist_task(task)

    def _consume_persist_task(self, task: asyncio.Task[None]) -> None:
        self._persist_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException:
            return

    def close(self) -> None:
        """Synchronously make a bounded best-effort final save and close the Session."""
        if self._closed:
            return
        self._closed = True
        if not self.messages:
            return

        for attempt in range(3):
            try:
                self._updated_at = self._clock_now()
                self._write_content(self._serialized_state())
                return
            except Exception:
                if attempt < 2:
                    time.sleep((0.1, 0.2)[attempt])

    def abandon(self) -> None:
        """Synchronously abandon the Session without a final persistence attempt."""
        if self._abandoned:
            return
        self._abandoned = True
        self._closed = True
        self._pending_persist = None
        pending_tasks = tuple(self._persist_tasks)
        for pending in pending_tasks:
            if not pending.done():
                pending.cancel()

    def _serialized_state(self) -> bytes:
        return _serialize_session_state(
            session_id=self._session_id,
            created_at=self._created_at,
            updated_at=self._updated_at,
            last_compacted=self.last_compacted,
            metadata=self.metadata,
            messages=self.messages,
        )

    async def _persist_after(
        self,
        previous: asyncio.Task[None] | None,
        content: bytes,
    ) -> None:
        if previous is not None and not previous.done():
            try:
                await previous
            except Exception:
                pass
        for attempt in range(3):
            if self._closed:
                return
            try:
                self._write_content(content)
                return
            except Exception:
                if attempt == 2 or self._closed:
                    return
                await asyncio.sleep((0.1, 0.2)[attempt])

    def _ensure_not_abandoned(self) -> None:
        if self._abandoned:
            raise RuntimeError("Session has been abandoned")

    def _require_foreground_restore(self) -> None:
        if self._storage_partition is not SessionStoragePartition.FOREGROUND:
            raise ValueError("Restore anchors are only supported for foreground Sessions")

    def _persisted_restore_messages(self) -> list[dict[str, Any]]:
        try:
            persisted = self.load(
                self._workspace_state,
                self._session_id,
                partition=SessionStoragePartition.FOREGROUND,
            )
        except FileNotFoundError:
            return []
        return persisted.messages

    def _write_content(self, content: bytes) -> None:
        if self._storage_partition is SessionStoragePartition.FOREGROUND:
            sessions_directory = self._workspace_state.prepare_sessions_directory()
        else:
            sessions_directory = self._workspace_state.prepare_schedule_sessions_directory()
        path = sessions_directory / f"{self._session_id}.jsonl"
        io_path = HOST_FILESYSTEM.path_for_io(path)
        try:
            io_path.lstat()
        except FileNotFoundError:
            pass
        else:
            HOST_FILESYSTEM.require_owned_regular_file(
                io_path,
                within=sessions_directory,
            )
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)
        HOST_FILESYSTEM.require_owned_regular_file(path, within=sessions_directory)

    def _usage_after_delta(self, delta: Any) -> dict[str, int] | None:
        if delta is None:
            return None
        return _accumulate_token_usage(self.metadata.get("token_usage"), delta)

    def _clock_now(self) -> datetime:
        return _clock_now(self._now)

    @classmethod
    def schedule_session_id(cls, job_id: UUID | str) -> str:
        """Return the canonical Schedule Session ID for one Job UUID4."""
        if isinstance(job_id, UUID):
            require_uuid4(job_id, field="job_id")
            canonical = str(job_id)
        elif isinstance(job_id, str):
            require_uuid4_string(job_id, field="job_id")
            canonical = job_id
        else:
            raise ValueError("job_id must be a canonical UUID4")
        return f"schedule_{canonical}"

    @classmethod
    def _require_id(
        cls,
        value: str,
        *,
        field: str = "session_id",
        partition: SessionStoragePartition | None = None,
    ) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a valid Session ID")
        resolved_partition = _coerce_partition(partition) if partition is not None else None
        if resolved_partition is SessionStoragePartition.FOREGROUND or (
            resolved_partition is None and _SCHEDULE_SESSION_ID_PATTERN.fullmatch(value) is None
        ):
            match = _SESSION_ID_PATTERN.fullmatch(value)
            if match is None:
                raise ValueError(f"{field} must be a valid Session ID")
            try:
                datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S-%f")
                require_uuid4_string(match.group("uuid"), field=field)
            except ValueError as error:
                raise ValueError(f"{field} must be a valid Session ID") from error
            return
        match = _SCHEDULE_SESSION_ID_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"{field} must be a valid Schedule Session ID")
        try:
            require_uuid4_string(match.group("uuid"), field=field)
        except ValueError as error:
            raise ValueError(f"{field} must be a valid Schedule Session ID") from error

    @staticmethod
    def _normalize_title(value: str) -> str:
        return _normalize_title(value)

    @staticmethod
    def _normalize_title_candidate(value: str) -> str:
        return normalize_title_candidate(value)


def _coerce_partition(value: SessionStoragePartition | str) -> SessionStoragePartition:
    if isinstance(value, SessionStoragePartition):
        return value
    try:
        return SessionStoragePartition(value)
    except (TypeError, ValueError) as error:
        raise ValueError("unknown Session storage partition") from error


def _resolve_partition(
    session_id: str,
    partition: SessionStoragePartition | str | None,
) -> SessionStoragePartition:
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a valid Session ID")
    if partition is None:
        resolved = (
            SessionStoragePartition.SCHEDULE
            if _SCHEDULE_SESSION_ID_PATTERN.fullmatch(session_id) is not None
            else SessionStoragePartition.FOREGROUND
        )
    else:
        resolved = _coerce_partition(partition)
    Session._require_id(session_id, partition=resolved)
    return resolved


def _storage_directory(
    workspace_state: WorkspaceState,
    partition: SessionStoragePartition,
) -> Path:
    if partition is SessionStoragePartition.FOREGROUND:
        return workspace_state.sessions_directory
    return workspace_state.schedule_sessions_directory


def _existing_sessions_directory(
    workspace_state: WorkspaceState,
    partition: SessionStoragePartition,
) -> Path | None:
    if partition is SessionStoragePartition.FOREGROUND:
        return workspace_state.existing_sessions_directory()
    return workspace_state.existing_schedule_sessions_directory()


def _clock_now(now: Callable[[], datetime] | None) -> datetime:
    return local_now() if now is None else now()


def _make_id(created_at: datetime, session_uuid: UUID) -> str:
    require_aware_datetime(created_at, field="created_at")
    require_uuid4(session_uuid, field="session_uuid")
    return f"{created_at:%Y%m%d-%H%M%S-%f}_{session_uuid}"


def _initial_metadata() -> dict[str, Any]:
    return {
        "title": "Untitled session",
        "token_usage": {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "summary": "",
    }


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        content = HOST_FILESYSTEM.path_for_io(path).read_bytes()
    except OSError:
        raise
    if not content.endswith(b"\n"):
        raise ValueError("Session JSONL must end with a newline")
    lines = content.splitlines(keepends=False)
    if not lines or any(not line for line in lines):
        raise ValueError("Session JSONL must contain complete records")
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            decoded: Any = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Session JSONL contains invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("Session JSONL records must be objects")
        records.append(cast(dict[str, Any], decoded))
    return records


def _parse_header(
    record: dict[str, Any],
) -> tuple[str, datetime, datetime, int, dict[str, Any]]:
    if set(record) != _HEADER_FIELDS:
        raise ValueError("Session header fields do not match the current format")
    session_id = record["session_id"]
    if not isinstance(session_id, str):
        raise ValueError("Session header session_id must be a string")
    Session._require_id(session_id)
    created_at = _parse_datetime(record["created_at"], field="created_at")
    updated_at = _parse_datetime(record["updated_at"], field="updated_at")
    last_compacted = record["last_compacted"]
    require_nonnegative_int(last_compacted, field="last_compacted")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Session metadata must be an object")
    metadata_copy = _copy_loaded_metadata(cast(dict[str, Any], metadata))
    _validate_metadata(metadata_copy)
    return session_id, created_at, updated_at, last_compacted, metadata_copy


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Session field '{field}' must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Session field '{field}' must be ISO 8601") from error
    require_aware_datetime(parsed, field=field)
    return parsed


def _parse_message(record: dict[str, Any]) -> dict[str, Any]:
    message = _copy_json_object(record, field="message")
    if any(key in message for key in ("record_" + "type", "schema_" + "version")):
        raise ValueError("legacy Session message fields are unsupported")
    try:
        _validate_message(message)
    except KeyError as error:
        raise ValueError(f"Session message is missing {error.args[0]}") from error
    return message


def _validate_metadata(metadata: dict[str, Any]) -> None:
    _validate_json_value(metadata, field="metadata")
    title = metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("metadata.title must be a string")
    if not title or " ".join(title.split()) != title or len(title) > 60:
        raise ValueError("metadata.title is not normalized")
    _validate_token_usage(metadata.get("token_usage"), field="metadata.token_usage")
    _validate_action_summary(metadata.get("summary", ""), field="metadata.summary")
    if _RESTORE_NEXT_ANCHOR_ID in metadata:
        _validate_restore_next_anchor_id(metadata[_RESTORE_NEXT_ANCHOR_ID])
    _normalize_blackboard_metadata(metadata, invalid_is_absent=False)


def _copy_loaded_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(metadata)
    _normalize_blackboard_metadata(copied, invalid_is_absent=True)
    copied.setdefault("summary", "")
    return _copy_json_object(copied, field="metadata")


def _validate_action_summary(value: Any, *, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")


def _normalize_blackboard_metadata(
    metadata: dict[str, Any],
    *,
    invalid_is_absent: bool,
) -> None:
    if "blackboard" not in metadata:
        return
    from myclaw.agent.blackboard import Blackboard

    blackboard = Blackboard.from_dict(metadata["blackboard"])
    if blackboard is None:
        if invalid_is_absent:
            del metadata["blackboard"]
            return
        raise ValueError("metadata.blackboard must be a valid Blackboard")
    metadata["blackboard"] = blackboard.to_dict()


def _copy_metadata_updates(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_updates must be a dictionary")
    return _copy_json_object(value, field="metadata_updates")


def _validate_agent_run_metadata_patch(
    updates: dict[str, Any],
    removals: frozenset[str],
) -> None:
    conflict = set(updates).intersection(removals)
    if conflict:
        raise ValueError("metadata updates and removals cannot target the same key")
    protected_updates = {"title", *_TOKEN_USAGE_PATCH_KEYS}.intersection(updates)
    if protected_updates:
        raise ValueError("title and token usage cannot be changed through metadata_updates")
    if "summary" in updates or "summary" in removals:
        raise ValueError("Action Summary must be supplied through pending_action_summary")
    usage_removals = (_TOKEN_USAGE_PATCH_KEYS - {"token_usage"}).intersection(removals)
    if usage_removals:
        raise ValueError("token usage must be supplied through usage_delta")
    required_removals = {"title", "token_usage"}.intersection(removals)
    if required_removals:
        raise ValueError("required Session metadata cannot be removed")
    if _RESTORE_NEXT_ANCHOR_ID in updates or _RESTORE_NEXT_ANCHOR_ID in removals:
        raise ValueError("restore anchor counter is Session-owned")


def _validate_metadata_removals(value: tuple[str, ...]) -> frozenset[str]:
    if not isinstance(value, tuple):
        raise TypeError("metadata_removals must be a tuple")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata_removals must contain only strings")
    return frozenset(value)


def _copy_json_object(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    copied = copy.deepcopy(value)
    _validate_json_value(copied, field=field)
    return copied


def _validate_json_value(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{field} must contain only JSON-compatible values")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} must contain only JSON-compatible values")
            _validate_json_value(item, field=f"{field}.{key}")
        return
    raise TypeError(f"{field} must contain only JSON-compatible values")


def _validate_message(message: dict[str, Any]) -> None:
    _validate_json_value(message, field="message")
    if "id" in message:
        raise ValueError("unsupported Session message identifiers")
    role = message["role"]
    if not isinstance(role, str):
        raise TypeError("message role must be a string")
    if not isinstance(message["content"], str):
        raise TypeError("message content must be a string")
    if not isinstance(message["timestamp"], str):
        raise TypeError("message timestamp must be a string")
    try:
        timestamp = datetime.fromisoformat(message["timestamp"])
    except ValueError as error:
        raise ValueError("message timestamp must be ISO 8601") from error
    require_aware_datetime(timestamp, field="message timestamp")
    if role != "assistant" and "context_usage" in message:
        raise ValueError("context_usage is only valid on assistant messages")
    _restore_anchor_fields(message)
    if role == "user":
        if not message["content"].strip():
            raise ValueError("user message content must not be blank")
        return
    if role == "assistant":
        _validate_assistant_message(message)
        return
    if role == "tool":
        _validate_tool_message(message)
        return
    raise ValueError("role must be user, assistant, or tool")


def _validate_assistant_message(message: dict[str, Any]) -> None:
    required = {"tool_calls", "status", "error", "token_usage"}
    missing = required.difference(message)
    if missing:
        raise ValueError(f"assistant message is missing {', '.join(sorted(missing))}")
    status = message["status"]
    if status not in {"completed", "interrupted", "error"}:
        raise ValueError("assistant status is not supported")
    error = message["error"]
    if status == "completed" and error is not None:
        raise ValueError("completed assistant must not have an error")
    if status != "completed" and error is None:
        raise ValueError("non-completed assistant requires an error")
    tool_calls = message["tool_calls"]
    if not isinstance(tool_calls, list):
        raise TypeError("assistant tool_calls must be a list")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise TypeError("assistant tool_calls must contain dictionaries")
        for field in ("id", "name", "arguments"):
            if not isinstance(tool_call.get(field), str):
                raise ValueError(f"assistant tool_calls require {field}")
    if error is not None:
        if not isinstance(error, dict) or not isinstance(error.get("code"), str):
            raise ValueError("assistant error must contain a code")
        if not isinstance(error.get("message"), str):
            raise ValueError("assistant error must contain a message")
    token_usage = message["token_usage"]
    _validate_token_usage(token_usage, field="assistant.token_usage")
    if "context_usage" in message:
        ContextUsageSnapshot.from_dict(message["context_usage"])
        if token_usage["model_calls"] != 1:
            raise ValueError("context_usage requires exactly one assistant model call")
    if token_usage["model_calls"] != 1 and not (
        status in {"error", "interrupted"} and token_usage["model_calls"] == 0 and error is not None
    ):
        raise ValueError("assistant.token_usage.model_calls must equal 1")
    if status != "error":
        if not message["content"] and not tool_calls:
            raise ValueError("assistant requires content or tool_calls unless status is error")


def _validate_tool_message(message: dict[str, Any]) -> None:
    for field in ("tool_call_id", "name", "status"):
        if not isinstance(message.get(field), str):
            raise ValueError(f"tool message requires {field}")
    if message["status"] not in {"success", "error", "refused"}:
        raise ValueError("tool status is not supported")
    artifact = message.get("artifact")
    if artifact is not None:
        if message["status"] != "success":
            raise ValueError("only successful tool messages may contain an artifact")
        if not isinstance(artifact, dict):
            raise ValueError("tool artifact must be an object or null")
        if set(artifact) != {"path", "total_chars", "preview_chars"}:
            raise ValueError("tool artifact has an invalid shape")
        try:
            ArtifactReference(
                path=artifact["path"],
                total_chars=artifact["total_chars"],
                preview_chars=artifact["preview_chars"],
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("tool artifact is malformed") from error


def _validate_token_usage(value: Any, *, field: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a dictionary")
    issue = token_usage_validation_issue(value)
    if issue == "fields":
        raise ValueError(f"{field} must contain exactly model_calls and token counters")
    if issue == "values":
        for key, member in value.items():
            require_nonnegative_int(member, field=f"{field}.{key}")
        raise AssertionError("token usage value issue did not identify an invalid field")
    if issue == "total":
        raise ValueError(f"{field}.total_tokens must equal input_tokens + output_tokens")


def _accumulate_token_usage(
    current: Any,
    delta: Any,
) -> dict[str, int]:
    _validate_token_usage(current, field="metadata.token_usage")
    _validate_token_usage(delta, field="token_usage_delta")
    assert isinstance(current, dict)
    assert isinstance(delta, dict)
    return {
        key: current[key] + delta[key]
        for key in ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    }


def _copy_restore_before(value: SessionRestoreBefore | None) -> SessionRestoreBefore | None:
    if value is None:
        return None
    if not isinstance(value, SessionRestoreBefore):
        raise TypeError("restore_before must be a SessionRestoreBefore")
    metadata = _copy_json_object(value.metadata, field="restore_before.metadata")
    _validate_metadata(metadata)
    require_nonnegative_int(value.last_compacted, field="restore_before.last_compacted")
    return SessionRestoreBefore(metadata=metadata, last_compacted=value.last_compacted)


def _validate_restore_anchor_id(value: Any) -> None:
    require_nonnegative_int(value, field="restore_anchor_id")
    if value == 0:
        raise ValueError("restore_anchor_id must be a positive integer")


def _validate_restore_next_anchor_id(value: Any) -> None:
    require_nonnegative_int(value, field=_RESTORE_NEXT_ANCHOR_ID)
    if value == 0:
        raise ValueError(f"{_RESTORE_NEXT_ANCHOR_ID} must be a positive integer")


def _next_restore_anchor_id(
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
) -> int:
    configured_value = metadata.get(_RESTORE_NEXT_ANCHOR_ID, 1)
    _validate_restore_next_anchor_id(configured_value)
    configured = cast(int, configured_value)
    maximum = 0
    for message in messages:
        fields = _restore_anchor_fields(message)
        if fields is not None:
            maximum = max(maximum, fields[0])
    return max(configured, maximum + 1)


def _validate_restore_sequence(
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    partition: SessionStoragePartition,
) -> None:
    last_anchor_id = 0
    for message in messages:
        fields = _restore_anchor_fields(message)
        if fields is None:
            continue
        if partition is not SessionStoragePartition.FOREGROUND:
            raise ValueError("Restore anchors are only supported for foreground Sessions")
        if fields[0] <= last_anchor_id:
            raise ValueError("Restore anchor IDs must increase in Session order")
        last_anchor_id = fields[0]
    next_anchor_id = metadata.get(_RESTORE_NEXT_ANCHOR_ID)
    if partition is not SessionStoragePartition.FOREGROUND and next_anchor_id is not None:
        raise ValueError("Restore anchors are only supported for foreground Sessions")
    if last_anchor_id and next_anchor_id is None:
        raise ValueError("Restore anchor counter is missing")
    if next_anchor_id is not None:
        _validate_restore_next_anchor_id(next_anchor_id)
        if next_anchor_id <= last_anchor_id:
            raise ValueError("Restore anchor counter must exceed persisted IDs")


def _restore_before_from_value(value: Any) -> SessionRestoreBefore:
    if not isinstance(value, dict) or set(value) != _RESTORE_BEFORE_FIELDS:
        raise ValueError("restore_before must contain metadata and last_compacted")
    metadata = value["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("restore_before.metadata must be an object")
    copied_metadata = _copy_json_object(metadata, field="restore_before.metadata")
    _validate_metadata(copied_metadata)
    last_compacted = value["last_compacted"]
    require_nonnegative_int(last_compacted, field="restore_before.last_compacted")
    return SessionRestoreBefore(metadata=copied_metadata, last_compacted=last_compacted)


def _restore_anchor_fields(
    message: dict[str, Any],
) -> tuple[int, str, SessionRestoreBefore] | None:
    present = _RESTORE_MESSAGE_FIELDS.intersection(message)
    if not present:
        return None
    if message.get("role") != "user":
        raise ValueError("restore anchor fields are only valid on User messages")
    if present != _RESTORE_MESSAGE_FIELDS:
        raise ValueError("restore anchor fields must be persisted together")
    anchor_id = message["restore_anchor_id"]
    _validate_restore_anchor_id(anchor_id)
    run_token = message["restore_run_token"]
    if not isinstance(run_token, str):
        raise ValueError("restore_run_token must be a canonical UUID4 string")
    require_uuid4_string(run_token, field="restore_run_token")
    return anchor_id, run_token, _restore_before_from_value(message["restore_before"])


def _serialize_session_state(
    *,
    session_id: str,
    created_at: datetime,
    updated_at: datetime,
    last_compacted: int,
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bytes:
    header = {
        "session_id": session_id,
        "created_at": format_rfc3339_milliseconds(created_at),
        "updated_at": format_rfc3339_milliseconds(updated_at),
        "last_compacted": last_compacted,
        "metadata": copy.deepcopy(metadata),
    }
    records = (header, *copy.deepcopy(messages))
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
