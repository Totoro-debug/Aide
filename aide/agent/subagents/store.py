"""Session-scoped, versioned persistence for SubAgent records."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import cast
from uuid import UUID, uuid4

from aide.agent.session.session import validate_session_id
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentListItem,
    SubAgentPage,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.tools.base import ArtifactReference
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM
from aide.utils.json import strict_json_loads
from aide.utils.validation import require_aware_datetime, require_uuid4, require_uuid4_string

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
ReplaceText = Callable[[Path, str], None]
Now = Callable[[], datetime]
NewUUID = Callable[[], UUID]


class SubAgentStoreError(RuntimeError):
    """Raised when a SubAgent record cannot be persisted or read safely."""


class SubAgentRequestError(ValueError):
    """Raised when a Session-scoped SubAgent query is invalid."""


class SubAgentRecordStore:
    """Own one Session's SubAgent files and never query another Session's records."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        now: Now | None = None,
        new_uuid: NewUUID = uuid4,
        replace_text: ReplaceText = HOST_FILESYSTEM.atomic_replace_text,
    ) -> None:
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("workspace_state must be a WorkspaceState")
        validate_session_id(session_id)
        self._workspace_state = workspace_state
        self._session_id = session_id
        self._now = now or (lambda: datetime.now(UTC))
        self._new_uuid = new_uuid
        self._replace_text = replace_text
        self._lock = RLock()
        self._recover_interrupted()

    @classmethod
    def recover_workspace(cls, workspace_state: WorkspaceState, *, now: Now | None = None) -> None:
        """Recover every persisted Session before admitting new SubAgent work in a Workspace."""
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("workspace_state must be a WorkspaceState")
        try:
            root = workspace_state.existing_subagents_directory()
            if root is None:
                return
            session_directories = tuple(root.iterdir())
        except OSError as error:
            raise SubAgentStoreError("Workspace SubAgent records could not be recovered") from error
        for directory in session_directories:
            try:
                HOST_FILESYSTEM.require_owned_directory(directory, within=root)
                validate_session_id(directory.name)
            except (OSError, ValueError) as error:
                raise SubAgentStoreError(
                    "Workspace contains an unsafe SubAgent Session entry"
                ) from error
            cls(workspace_state, directory.name, now=now)

    @property
    def session_id(self) -> str:
        """Return the Session this repository is permanently scoped to."""
        return self._session_id

    def register(
        self,
        *,
        title: str,
        task: str,
        parent_run_id: str,
        source: SubAgentSource,
        creator_snapshot: SubAgentCreatorSnapshot,
    ) -> SubAgentRecord:
        """Persist a queued registration before returning its generated identifier."""
        with self._lock:
            created_at = self._now()
            require_aware_datetime(created_at, field="created_at")
            agent_uuid = self._new_uuid()
            require_uuid4(agent_uuid, field="agent_id")
            existing = self._read_records()
            record = SubAgentRecord(
                agent_id=str(agent_uuid),
                session_id=self._session_id,
                title=title,
                task=task,
                parent_run_id=parent_run_id,
                source=source,
                creator_snapshot=creator_snapshot,
                created_at=created_at,
                registered_order=max((item.registered_order for item in existing), default=0) + 1,
                revision=0,
                status=SubAgentStatus.QUEUED,
            )
            directory = self._prepare_session_directory()
            path = directory / f"{record.agent_id}.json"
            if HOST_FILESYSTEM.entry_exists(path):
                raise SubAgentStoreError("SubAgent identifier already exists in this Session")
            try:
                self._write_record(path, record, directory=directory)
            except SubAgentStoreError as error:
                # Atomic publication can succeed before a durability check fails.
                # An unacknowledged registration must not remain available to execute.
                try:
                    if HOST_FILESYSTEM.entry_exists(path):
                        HOST_FILESYSTEM.require_owned_directory(
                            directory, within=self._workspace_state.path
                        )
                        HOST_FILESYSTEM.remove_owned_entry(path, root=directory, tree=False)
                except OSError as cleanup_error:
                    raise SubAgentStoreError(
                        f"SubAgent registration could not be rolled back: {cleanup_error}"
                    ) from error
                raise
            return record

    def save(
        self, record: SubAgentRecord, *, expected_revision: int | None = None
    ) -> SubAgentRecord:
        """Atomically replace one record after validating its revision and state transition."""
        if not isinstance(record, SubAgentRecord):
            raise TypeError("record must be a SubAgentRecord")
        record.__post_init__()
        if record.session_id != self._session_id:
            raise SubAgentRequestError("SubAgent record belongs to a different Session")
        with self._lock:
            directory = self._existing_session_directory()
            if directory is None:
                raise SubAgentStoreError("SubAgent record does not exist in this Session")
            path = self._record_path(directory, record.agent_id)
            if not HOST_FILESYSTEM.entry_exists(path):
                raise SubAgentStoreError("SubAgent record does not exist in this Session")
            existing = self._read_record(path, directory=directory)
            if record == existing:
                return existing
            self._validate_update(existing, record, expected_revision=expected_revision)
            self._write_record(path, record, directory=directory)
            return record

    def get(self, agent_id: str) -> SubAgentRecord | None:
        """Read full details for one ID, returning no record when it is unknown here."""
        path_id = self._validate_agent_id(agent_id)
        with self._lock:
            directory = self._existing_session_directory()
            if directory is None:
                return None
            path = self._record_path(directory, path_id)
            if not HOST_FILESYSTEM.entry_exists(path):
                return None
            return self._read_record(path, directory=directory)

    def list(
        self,
        *,
        status: SubAgentStatus | str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SubAgentPage:
        """Return a stable page of brief task DTOs in registration order."""
        page_limit = self._validate_limit(limit)
        selected_status = self._validate_status(status)
        after_order = self._decode_cursor(cursor) if cursor is not None else 0
        with self._lock:
            records = self._read_records()
            matching = tuple(
                record
                for record in records
                if record.registered_order > after_order
                and (selected_status is None or record.status is selected_status)
            )
            selected = matching[:page_limit]
            next_cursor = None
            if len(matching) > len(selected) and selected:
                next_cursor = self._encode_cursor(selected[-1].registered_order)
            return SubAgentPage(
                items=tuple(
                    SubAgentListItem(
                        agent_id=record.agent_id,
                        title=record.title,
                        status=record.status,
                        created_at=record.created_at,
                        finished_at=record.finished_at,
                    )
                    for record in selected
                ),
                next_cursor=next_cursor,
            )

    def discard_restore_run_tokens(self, restore_run_tokens: Sequence[str | UUID]) -> None:
        """Remove records and Tool Artifacts created by discarded foreground inputs."""
        if isinstance(restore_run_tokens, (str, bytes)) or not isinstance(
            restore_run_tokens, Sequence
        ):
            raise SubAgentRequestError("Restore Run tokens must be a sequence")
        selected_tokens: set[str] = set()
        for token in restore_run_tokens:
            token_value = str(token) if isinstance(token, UUID) else token
            try:
                require_uuid4_string(token_value, field="restore_run_token")
            except ValueError as error:
                raise SubAgentRequestError("Restore Run token is invalid") from error
            selected_tokens.add(token_value)
        if not selected_tokens:
            return

        with self._lock:
            records = tuple(
                record
                for record in self._read_records()
                if record.source.kind is SubAgentSourceKind.FOREGROUND
                and record.source.restore_run_token in selected_tokens
            )
            directory = self._existing_session_directory()
            for record in records:
                self._remove_record_artifacts(record)
                if directory is None:
                    continue
                path = self._record_path(directory, record.agent_id)
                if HOST_FILESYSTEM.entry_exists(path):
                    HOST_FILESYSTEM.require_owned_regular_file(path, within=directory)
                    HOST_FILESYSTEM.remove_owned_entry(path, root=directory, tree=False)

    def _remove_record_artifacts(self, record: SubAgentRecord) -> None:
        workspace_root = self._workspace_state.workspace_path
        for relative_path in record.artifact_paths:
            try:
                ArtifactReference(path=relative_path, total_chars=0, preview_chars=0)
                if relative_path.split("/")[2] != self._session_id:
                    raise ValueError("Tool Artifact belongs to another Session")
                path = workspace_root.joinpath(*relative_path.split("/"))
                if not HOST_FILESYSTEM.entry_exists(path):
                    continue
                HOST_FILESYSTEM.require_owned_regular_file(path, within=workspace_root)
                HOST_FILESYSTEM.remove_owned_entry(path, root=workspace_root, tree=False)
            except (OSError, PermissionError, TypeError, ValueError) as error:
                raise SubAgentStoreError(
                    "SubAgent Tool Artifact could not be removed safely"
                ) from error

    def _recover_interrupted(self) -> None:
        records = self._read_records()
        for record in records:
            if record.status not in {SubAgentStatus.QUEUED, SubAgentStatus.RUNNING}:
                continue
            updated = replace(
                record,
                status=SubAgentStatus.INTERRUPTED,
                finished_at=self._now(),
                error=record.error
                or SubAgentError(
                    code="service_interrupted",
                    message="The SubAgent was interrupted before the Service restarted.",
                ),
                revision=record.revision + 1,
            )
            self.save(updated)

    def _read_records(self) -> tuple[SubAgentRecord, ...]:
        directory = self._existing_session_directory()
        if directory is None:
            return ()
        try:
            paths = tuple(path for path in directory.iterdir() if path.suffix == ".json")
        except OSError as error:
            raise SubAgentStoreError("SubAgent records could not be listed safely") from error
        records = tuple(self._read_record(path, directory=directory) for path in paths)
        orders = [record.registered_order for record in records]
        if len(set(orders)) != len(orders):
            raise SubAgentStoreError("SubAgent registration order is duplicated")
        return tuple(sorted(records, key=lambda record: record.registered_order))

    def _read_record(self, path: Path, *, directory: Path) -> SubAgentRecord:
        try:
            HOST_FILESYSTEM.require_owned_regular_file(path, within=directory)
            value = strict_json_loads(path.read_text(encoding="utf-8"))
            record = SubAgentRecord.from_dict(value)
            if record.session_id != self._session_id:
                raise ValueError("record Session does not match its owning directory")
            if path.name != f"{record.agent_id}.json":
                raise ValueError("record identifier does not match its filename")
            return record
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            if isinstance(error, SubAgentStoreError):
                raise
            raise SubAgentStoreError(
                f"SubAgent record {path.stem} could not be loaded: {error}"
            ) from error

    def _write_record(self, path: Path, record: SubAgentRecord, *, directory: Path) -> None:
        try:
            HOST_FILESYSTEM.require_owned_directory(directory, within=self._workspace_state.path)
            if HOST_FILESYSTEM.entry_exists(path):
                HOST_FILESYSTEM.require_owned_regular_file(path, within=directory)
            content = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._replace_text(path, f"{content}\n")
            HOST_FILESYSTEM.require_owned_regular_file(path, within=directory)
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, SubAgentStoreError):
                raise
            raise SubAgentStoreError(
                f"SubAgent record {record.agent_id} could not be saved: {error}"
            ) from error

    def _existing_session_directory(self) -> Path | None:
        try:
            root = self._workspace_state.existing_subagents_directory()
        except (OSError, PermissionError) as error:
            raise SubAgentStoreError("Workspace SubAgent directory is unsafe") from error
        if root is None:
            return None
        path = root / self._session_id
        if not HOST_FILESYSTEM.entry_exists(path):
            return None
        try:
            return HOST_FILESYSTEM.require_owned_directory(path, within=root)
        except (OSError, PermissionError) as error:
            raise SubAgentStoreError("SubAgent Session directory is unsafe") from error

    def _prepare_session_directory(self) -> Path:
        try:
            root = self._workspace_state.prepare_subagents_directory()
            path = HOST_FILESYSTEM.path_for_io(root / self._session_id)
            if not HOST_FILESYSTEM.entry_exists(path):
                path.mkdir()
            return HOST_FILESYSTEM.require_owned_directory(path, within=root)
        except (OSError, PermissionError) as error:
            raise SubAgentStoreError(
                "SubAgent Session directory could not be prepared safely"
            ) from error

    @staticmethod
    def _record_path(directory: Path, agent_id: str) -> Path:
        return directory / f"{agent_id}.json"

    @staticmethod
    def _validate_agent_id(agent_id: str) -> str:
        try:
            require_uuid4_string(agent_id, field="agent_id")
        except ValueError as error:
            raise SubAgentRequestError("agent_id must be a canonical UUID4") from error
        return agent_id

    @staticmethod
    def _validate_limit(limit: int | None) -> int:
        if limit is None:
            return _DEFAULT_LIMIT
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise SubAgentRequestError(f"limit must be between 1 and {_MAX_LIMIT}")
        return limit

    @staticmethod
    def _validate_status(status: SubAgentStatus | str | None) -> SubAgentStatus | None:
        if status is None:
            return None
        try:
            return status if isinstance(status, SubAgentStatus) else SubAgentStatus(status)
        except (TypeError, ValueError) as error:
            raise SubAgentRequestError("status is invalid") from error

    def _encode_cursor(self, after_order: int) -> str:
        payload = json.dumps(
            {
                "version": 1,
                "session_id": self._session_id,
                "after_order": after_order,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: str) -> int:
        try:
            if not isinstance(cursor, str) or not cursor:
                raise ValueError("cursor must be a nonempty string")
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = strict_json_loads(
                base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
            )
            if not isinstance(payload, dict) or set(payload) != {
                "version",
                "session_id",
                "after_order",
            }:
                raise ValueError("cursor fields are invalid")
            version = payload["version"]
            session_id = payload["session_id"]
            after_order = payload["after_order"]
            if isinstance(version, bool) or not isinstance(version, int) or version != 1:
                raise ValueError("cursor version is invalid")
            if session_id != self._session_id:
                raise ValueError("cursor Session does not match")
            if isinstance(after_order, bool) or not isinstance(after_order, int) or after_order < 0:
                raise ValueError("cursor position is invalid")
            return cast(int, after_order)
        except (TypeError, ValueError, UnicodeError) as error:
            raise SubAgentRequestError("cursor is invalid for this Session") from error

    @staticmethod
    def _validate_update(
        existing: SubAgentRecord,
        candidate: SubAgentRecord,
        *,
        expected_revision: int | None = None,
    ) -> None:
        if (
            candidate.session_id != existing.session_id
            or candidate.agent_id != existing.agent_id
            or candidate.registered_order != existing.registered_order
            or candidate.created_at != existing.created_at
            or candidate.parent_run_id != existing.parent_run_id
            or candidate.source != existing.source
            or candidate.creator_snapshot != existing.creator_snapshot
            or candidate.title != existing.title
            or candidate.task != existing.task
        ):
            raise SubAgentStoreError("SubAgent registration fields cannot be changed")
        previous_revision = (
            candidate.revision - 1 if expected_revision is None else expected_revision
        )
        if previous_revision != existing.revision or candidate.revision <= existing.revision:
            raise SubAgentStoreError("SubAgent record revision is stale")
        if existing.status is SubAgentStatus.QUEUED:
            allowed = {
                SubAgentStatus.QUEUED,
                SubAgentStatus.RUNNING,
                SubAgentStatus.CANCELLED,
                SubAgentStatus.INTERRUPTED,
            }
        elif existing.status is SubAgentStatus.RUNNING:
            allowed = {
                SubAgentStatus.RUNNING,
                SubAgentStatus.COMPLETED,
                SubAgentStatus.FAILED,
                SubAgentStatus.CANCELLED,
                SubAgentStatus.INTERRUPTED,
            }
        else:
            raise SubAgentStoreError("terminal SubAgent records cannot be changed")
        if candidate.status not in allowed:
            raise SubAgentStoreError("SubAgent status transition is invalid")
