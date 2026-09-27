"""Durable pre-write file backups for foreground Session restoration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Protocol
from uuid import UUID

from myclaw.agent.session._restore_persistence import (
    canonical_json_bytes,
    sha256_hex,
    sync_created_directory,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.validation import require_uuid4, require_uuid4_string

_SCHEMA_VERSION = 1
_SESSION_ID = re.compile(
    r"(?P<timestamp>[0-9]{8}-[0-9]{6}-[0-9]{6})_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})",
    re.ASCII,
)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_BLOB_NAME = re.compile(r"[0-9a-f]{48}\.bin", re.ASCII)
_GAP_CODES = frozenset(
    {"target_read_failed", "blob_write_failed", "journal_write_failed", "store_unavailable"}
)


class BackupStoreError(Exception):
    """A generic persistence or journal validation failure."""


class BackupIntegrityError(BackupStoreError):
    """A pre-write blob is missing, unsafe, or does not match its digest."""

    def __init__(self, operation_id: int, reason: str) -> None:
        self.operation_id = operation_id
        self.reason = reason
        super().__init__(f"Backup integrity check failed for operation {operation_id}.")


@dataclass(frozen=True, slots=True)
class FileState:
    exists: bool
    sha256: str | None
    blob_name: str | None = None


@dataclass(frozen=True, slots=True)
class BackupTicket:
    operation_id: int
    run_token: UUID
    requested_target: Path
    canonical_target: Path
    recorded: bool


@dataclass(frozen=True, slots=True)
class BackupJournalEntry:
    operation_id: int
    revision: int
    run_token: UUID
    requested_target: str
    canonical_target: str
    before: FileState
    after: FileState | None


@dataclass(frozen=True, slots=True)
class BackupGap:
    operation_id: int
    revision: int
    run_token: UUID
    requested_target: str
    canonical_target: str
    reason: str


@dataclass(frozen=True, slots=True)
class BackupIntegrityIssue:
    operation_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class BackupJournal:
    revision: int
    entries: tuple[BackupJournalEntry, ...]
    gaps: tuple[BackupGap, ...]
    integrity_issues: tuple[BackupIntegrityIssue, ...]


class FileMutationRecorder(Protocol):
    """Run-local hook around an authorized file mutation."""

    def before_write(self, run_token: UUID, resolved_target: Path) -> BackupTicket | None: ...

    def after_write(self, ticket: BackupTicket | None) -> None: ...


@dataclass(frozen=True, slots=True)
class _StorePaths:
    root: Path
    entries: Path
    blobs: Path
    state: Path


@dataclass(frozen=True, slots=True)
class _StoreState:
    next_operation_id: int
    revision: int
    journal_operation_ids: tuple[int, ...]
    discarded_operation_ids: tuple[int, ...]


class FileBackupStore:
    """Persist exact pre-write states and an ordered per-Session journal."""

    def __init__(self, workspace_state: WorkspaceState, session_id: str) -> None:
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("workspace_state must be a WorkspaceState")
        _validate_session_id(session_id)
        self._workspace_state = workspace_state
        self._session_id = session_id
        self._lock = RLock()
        self._durable_directories: set[tuple[Path, int, int]] = set()

    def before_write(self, run_token: UUID, resolved_target: Path) -> BackupTicket | None:
        """Record a mutation attempt without allowing backup failures to block it."""
        try:
            require_uuid4(run_token, field="run_token")
            requested, canonical = _normalize_target(resolved_target)
        except Exception:
            return None
        try:
            if self._is_restore_target(requested, canonical):
                return None
        except Exception:
            return None

        with self._lock:
            paths: _StorePaths | None = None
            operation_id: int | None = None
            revision = 1
            try:
                paths = self._prepare_store()
                state = self._load_state(paths)
                operation_id = state.next_operation_id
                revision = state.revision + 1
                reserved = _StoreState(
                    operation_id + 1,
                    revision,
                    state.journal_operation_ids,
                    state.discarded_operation_ids,
                )
                try:
                    self._write_state(paths, reserved)
                except Exception:
                    pass

                try:
                    content = _capture_target(canonical)
                except Exception:
                    self._persist_gap(
                        paths,
                        operation_id,
                        revision,
                        run_token,
                        requested,
                        canonical,
                        "target_read_failed",
                    )
                    return BackupTicket(
                        operation_id, run_token, requested, canonical, recorded=False
                    )

                if content is None:
                    before = FileState(False, None)
                else:
                    try:
                        blob_name = self._write_blob(paths, content)
                    except Exception:
                        self._persist_gap(
                            paths,
                            operation_id,
                            revision,
                            run_token,
                            requested,
                            canonical,
                            "blob_write_failed",
                        )
                        return BackupTicket(
                            operation_id, run_token, requested, canonical, recorded=False
                        )
                    before = FileState(True, sha256_hex(content), blob_name)
                entry = BackupJournalEntry(
                    operation_id=operation_id,
                    revision=revision,
                    run_token=run_token,
                    requested_target=str(requested),
                    canonical_target=str(canonical),
                    before=before,
                    after=None,
                )
                try:
                    self._write_operation(paths, _entry_object(entry), replace_existing=False)
                    self._write_state(
                        paths,
                        _state_with_operation(reserved, operation_id, revision),
                    )
                except Exception:
                    self._persist_gap(
                        paths,
                        operation_id,
                        revision,
                        run_token,
                        requested,
                        canonical,
                        "journal_write_failed",
                    )
                    return BackupTicket(
                        operation_id, run_token, requested, canonical, recorded=False
                    )
                return BackupTicket(operation_id, run_token, requested, canonical, recorded=True)
            except Exception:
                if paths is None:
                    return None
                if operation_id is None:
                    try:
                        state = self._load_state(paths)
                        operation_id = state.next_operation_id
                        revision = state.revision + 1
                        self._write_state(
                            paths,
                            _StoreState(
                                operation_id + 1,
                                revision,
                                state.journal_operation_ids,
                                state.discarded_operation_ids,
                            ),
                        )
                    except Exception:
                        try:
                            operation_id = self._next_operation_id_from_entries(paths)
                            revision = 1
                        except Exception:
                            return None
                self._persist_gap(
                    paths,
                    operation_id,
                    revision,
                    run_token,
                    requested,
                    canonical,
                    "store_unavailable",
                )
                return BackupTicket(operation_id, run_token, requested, canonical, False)

    def after_write(self, ticket: BackupTicket | None) -> None:
        """Record the target state observed after a mutation, best effort."""
        if ticket is None or not ticket.recorded:
            return
        with self._lock:
            try:
                paths = self._prepare_store()
                entry_path = self._entry_path(paths, ticket.operation_id)
                value = self._read_operation(paths, entry_path, ticket.operation_id)
                if not isinstance(value, BackupJournalEntry) or value.run_token != ticket.run_token:
                    return
                if Path(value.canonical_target) != ticket.canonical_target:
                    return
                after = _capture_target(ticket.canonical_target)
                after_state = (
                    FileState(False, None) if after is None else FileState(True, sha256_hex(after))
                )
                state = self._load_state(paths)
                revision = state.revision + 1
                updated = replace(value, revision=revision, after=after_state)
                self._write_state(
                    paths,
                    _StoreState(
                        max(state.next_operation_id, ticket.operation_id + 1),
                        revision,
                        state.journal_operation_ids,
                        state.discarded_operation_ids,
                    ),
                )
                self._write_operation(paths, _entry_object(updated), replace_existing=True)
            except Exception:
                return

    def inspect(self) -> BackupJournal:
        """Return ordered records and identify unavailable or damaged blobs."""
        with self._lock:
            paths = self._prepare_store()
            state = self._load_state(paths)
            entries: list[BackupJournalEntry] = []
            gaps: list[BackupGap] = []
            issues: list[BackupIntegrityIssue] = []
            revision = state.revision
            operation_paths = dict(self._operation_paths(paths))
            for operation_id in state.journal_operation_ids:
                entry_path = operation_paths.get(operation_id)
                if entry_path is None:
                    issues.append(BackupIntegrityIssue(operation_id, "missing_journal_entry"))
                    continue
                operation = self._read_operation(paths, entry_path, operation_id)
                revision = max(revision, operation.revision)
                if isinstance(operation, BackupGap):
                    gaps.append(operation)
                    continue
                entries.append(operation)
                if not operation.before.exists:
                    continue
                try:
                    self._verify_backup_blob(paths, operation)
                except BackupIntegrityError as error:
                    issues.append(BackupIntegrityIssue(operation_id, error.reason))
                except Exception:
                    issues.append(BackupIntegrityIssue(operation_id, "unsafe_or_unreadable_blob"))
            return BackupJournal(
                revision=revision,
                entries=tuple(entries),
                gaps=tuple(gaps),
                integrity_issues=tuple(issues),
            )

    def read_backup(self, operation_id: int) -> bytes | None:
        """Read one verified byte backup; return None for a recorded absence."""
        _validate_operation_id(operation_id)
        with self._lock:
            paths = self._prepare_store()
            operation = self._read_operation(
                paths, self._entry_path(paths, operation_id), operation_id
            )
            if isinstance(operation, BackupGap):
                raise BackupIntegrityError(operation_id, "backup_gap")
            if not operation.before.exists:
                return None
            return self._read_backup_blob(paths, operation)

    def discard_run_tokens(self, run_tokens: Iterable[UUID]) -> int:
        """Remove matching operations from the active branch without deleting evidence."""
        tokens = frozenset(run_tokens)
        for token in tokens:
            require_uuid4(token, field="run_token")
        if not tokens:
            return 0

        with self._lock:
            paths = self._prepare_store()
            state = self._load_state(paths)
            removed: list[int] = []
            for operation_id in state.journal_operation_ids:
                operation = self._read_operation(
                    paths,
                    self._entry_path(paths, operation_id),
                    operation_id,
                )
                if operation.run_token in tokens:
                    removed.append(operation_id)
            if not removed:
                return 0
            revision = state.revision + 1
            removed_set = set(removed)
            updated = _StoreState(
                next_operation_id=state.next_operation_id,
                revision=revision,
                journal_operation_ids=tuple(
                    operation_id
                    for operation_id in state.journal_operation_ids
                    if operation_id not in removed_set
                ),
                discarded_operation_ids=tuple(
                    sorted(set(state.discarded_operation_ids).union(removed_set))
                ),
            )
            self._write_state(paths, updated)
            return len(removed)

    def _is_restore_target(self, requested: Path, canonical: Path) -> bool:
        protected = Path(os.path.abspath(self._workspace_state.path / "restore"))
        protected_resolved = protected.resolve(strict=False)
        return any(
            _is_relative_to(path, protected) or _is_relative_to(path, protected_resolved)
            for path in (requested, canonical)
        )

    def _prepare_store(self) -> _StorePaths:
        workspace = HOST_FILESYSTEM.path_for_io(self._workspace_state.workspace_path)
        workspace_root = HOST_FILESYSTEM.require_owned_directory(workspace, within=workspace)
        state_root = self._ensure_directory(self._workspace_state.path, workspace_root)
        restore_root = self._ensure_directory(self._workspace_state.path / "restore", state_root)
        session_root = self._ensure_directory(restore_root / self._session_id, restore_root)
        entries = self._ensure_directory(session_root / "entries", session_root)
        blobs = self._ensure_directory(session_root / "blobs", session_root)
        return _StorePaths(
            root=session_root,
            entries=entries,
            blobs=blobs,
            state=session_root / "state.json",
        )

    def _ensure_directory(self, path: Path, within: Path) -> Path:
        io_path = HOST_FILESYSTEM.path_for_io(path)
        if not _path_entry_exists(io_path):
            io_path.mkdir(mode=0o700)
        owned = HOST_FILESYSTEM.require_owned_directory(path, within=within)
        HOST_FILESYSTEM.restrict_private_directory(owned)
        identity = _directory_identity(owned)
        if identity not in self._durable_directories:
            sync_created_directory(owned)
            if _directory_identity(owned) != identity:
                raise OSError("Restore directory identity changed during persistence.")
            self._durable_directories.add(identity)
        return owned

    def _load_state(self, paths: _StorePaths) -> _StoreState:
        state: _StoreState | None = None
        needs_write = False
        if _path_entry_exists(HOST_FILESYSTEM.path_for_io(paths.state)):
            try:
                raw = _read_owned_file(paths.state, within=paths.root)
                state = _decode_state(raw)
            except PermissionError:
                raise
            except (OSError, ValueError, TypeError, BackupStoreError):
                state = None

        operation_ids = tuple(operation_id for operation_id, _ in self._operation_paths(paths))
        max_operation_id = max(operation_ids, default=0)
        max_revision = self._max_record_revision(paths)
        if state is None:
            state = _StoreState(max_operation_id + 1, max_revision, operation_ids, ())
            needs_write = True
        else:
            next_id = max(state.next_operation_id, max_operation_id + 1)
            revision = max(state.revision, max_revision)
            discarded_operation_ids = tuple(sorted(set(state.discarded_operation_ids)))
            journal_operation_ids = tuple(
                sorted(
                    (set(state.journal_operation_ids).union(operation_ids))
                    - set(discarded_operation_ids)
                )
            )
            if (
                next_id != state.next_operation_id
                or revision != state.revision
                or journal_operation_ids != state.journal_operation_ids
                or discarded_operation_ids != state.discarded_operation_ids
            ):
                state = _StoreState(
                    next_id,
                    revision,
                    journal_operation_ids,
                    discarded_operation_ids,
                )
                needs_write = True

        if needs_write:
            try:
                self._write_state(paths, state)
            except Exception:
                pass
        return state

    def _write_state(self, paths: _StorePaths, state: _StoreState) -> None:
        content = _encode_signed_json(
            {
                "schema_version": _SCHEMA_VERSION,
                "next_operation_id": state.next_operation_id,
                "revision": state.revision,
                "journal_operation_ids": list(state.journal_operation_ids),
                "discarded_operation_ids": list(state.discarded_operation_ids),
            }
        )
        state_path = HOST_FILESYSTEM.path_for_io(paths.state)
        if _path_entry_exists(state_path):
            HOST_FILESYSTEM.require_owned_regular_file(paths.state, within=paths.root)
        HOST_FILESYSTEM.atomic_replace_bytes(paths.state, content)
        if _path_entry_exists(state_path):
            HOST_FILESYSTEM.restrict_private_file(paths.state)

    def _write_blob(self, paths: _StorePaths, content: bytes) -> str:
        for _ in range(4):
            name = f"{secrets.token_hex(24)}.bin"
            target = paths.blobs / name
            published = HOST_FILESYSTEM.atomic_create_bytes_with_identity(target, content)
            if published is not None:
                HOST_FILESYSTEM.restrict_private_file(target)
                return name
        raise BackupStoreError("Could not allocate an opaque backup blob name.")

    def _write_operation(
        self,
        paths: _StorePaths,
        operation: dict[str, object],
        *,
        replace_existing: bool,
    ) -> None:
        operation_id = operation["operation_id"]
        if isinstance(operation_id, bool) or not isinstance(operation_id, int):
            raise BackupStoreError("Invalid operation ID.")
        target = self._entry_path(paths, operation_id)
        content = _encode_signed_json(operation)
        if replace_existing:
            HOST_FILESYSTEM.require_owned_regular_file(target, within=paths.entries)
            HOST_FILESYSTEM.atomic_replace_bytes(target, content)
        else:
            if HOST_FILESYSTEM.atomic_create_bytes_with_identity(target, content) is None:
                raise BackupStoreError("Operation ID is already present.")
        HOST_FILESYSTEM.restrict_private_file(target)

    def _persist_gap(
        self,
        paths: _StorePaths,
        operation_id: int,
        revision: int,
        run_token: UUID,
        requested: Path,
        canonical: Path,
        reason: str,
    ) -> None:
        try:
            entry_path = self._entry_path(paths, operation_id)
            if _path_entry_exists(HOST_FILESYSTEM.path_for_io(entry_path)):
                try:
                    existing = self._read_operation(paths, entry_path, operation_id)
                    if isinstance(existing, BackupJournalEntry):
                        self._read_backup_blob(paths, existing)
                        self._remember_operation(paths, operation_id, revision)
                        return
                except Exception:
                    pass
            gap = BackupGap(
                operation_id=operation_id,
                revision=revision,
                run_token=run_token,
                requested_target=str(requested),
                canonical_target=str(canonical),
                reason=reason if reason in _GAP_CODES else "store_unavailable",
            )
            self._write_operation(
                paths,
                _gap_object(gap),
                replace_existing=_path_entry_exists(HOST_FILESYSTEM.path_for_io(entry_path)),
            )
            self._remember_operation(paths, operation_id, revision)
        except Exception:
            return

    def _remember_operation(
        self,
        paths: _StorePaths,
        operation_id: int,
        revision: int,
    ) -> None:
        state = self._load_state(paths)
        self._write_state(paths, _state_with_operation(state, operation_id, revision))

    def _read_operation(
        self, paths: _StorePaths, path: Path, operation_id: int
    ) -> BackupJournalEntry | BackupGap:
        try:
            raw = _read_owned_file(path, within=paths.entries)
            operation = _decode_operation(raw)
        except BackupStoreError:
            raise
        except Exception as error:
            raise BackupStoreError("Restore journal entry is unavailable or malformed.") from error
        if operation.operation_id != operation_id:
            raise BackupStoreError("Restore journal operation ID does not match its file.")
        return operation

    def _read_backup_blob(self, paths: _StorePaths, entry: BackupJournalEntry) -> bytes:
        path, digest = _backup_blob_reference(paths, entry)
        try:
            content = _read_owned_file(path, within=paths.blobs)
        except Exception as error:
            raise BackupIntegrityError(entry.operation_id, "missing_or_unsafe_blob") from error
        if sha256_hex(content) != digest:
            raise BackupIntegrityError(entry.operation_id, "hash_mismatch")
        return content

    def _verify_backup_blob(self, paths: _StorePaths, entry: BackupJournalEntry) -> None:
        path, digest = _backup_blob_reference(paths, entry)
        try:
            actual = _hash_owned_file(path, within=paths.blobs)
        except Exception as error:
            raise BackupIntegrityError(entry.operation_id, "missing_or_unsafe_blob") from error
        if actual != digest:
            raise BackupIntegrityError(entry.operation_id, "hash_mismatch")

    @staticmethod
    def _entry_path(paths: _StorePaths, operation_id: int) -> Path:
        return paths.entries / f"{operation_id}.json"

    def _operation_paths(self, paths: _StorePaths) -> tuple[tuple[int, Path], ...]:
        found: list[tuple[int, Path]] = []
        for path in HOST_FILESYSTEM.path_for_io(paths.entries).iterdir():
            if path.suffix != ".json":
                continue
            if not path.stem.isdecimal() or str(int(path.stem)) != path.stem:
                raise BackupStoreError("Restore journal contains an invalid operation filename.")
            found.append((int(path.stem), paths.entries / path.name))
        found.sort(key=lambda item: item[0])
        return tuple(found)

    def _next_operation_id_from_entries(self, paths: _StorePaths) -> int:
        maximum = 0
        for path in HOST_FILESYSTEM.path_for_io(paths.entries).iterdir():
            if path.suffix == ".json" and path.stem.isdecimal():
                maximum = max(maximum, int(path.stem))
        return maximum + 1

    def _max_record_revision(self, paths: _StorePaths) -> int:
        maximum = 0
        for operation_id, path in self._operation_paths(paths):
            try:
                maximum = max(maximum, self._read_operation(paths, path, operation_id).revision)
            except Exception:
                continue
        return maximum


def _state_with_operation(
    state: _StoreState,
    operation_id: int,
    revision: int,
) -> _StoreState:
    operation_ids = tuple(sorted(set(state.journal_operation_ids) | {operation_id}))
    return _StoreState(
        next_operation_id=max(state.next_operation_id, operation_id + 1),
        revision=max(state.revision, revision),
        journal_operation_ids=operation_ids,
        discarded_operation_ids=state.discarded_operation_ids,
    )


def _backup_blob_reference(
    paths: _StorePaths,
    entry: BackupJournalEntry,
) -> tuple[Path, str]:
    blob_name = entry.before.blob_name
    digest = entry.before.sha256
    if not entry.before.exists or blob_name is None or digest is None:
        raise BackupIntegrityError(entry.operation_id, "missing_blob_reference")
    if _BLOB_NAME.fullmatch(blob_name) is None:
        raise BackupIntegrityError(entry.operation_id, "invalid_blob_name")
    return paths.blobs / blob_name, digest


def _validate_session_id(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("session_id must be a canonical foreground Session ID")
    match = _SESSION_ID.fullmatch(value)
    if match is None:
        raise ValueError("session_id must be a canonical foreground Session ID")
    try:
        datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S-%f")
        require_uuid4_string(match.group("uuid"), field="session_id")
    except ValueError as error:
        raise ValueError("session_id must be a canonical foreground Session ID") from error


def _validate_operation_id(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("operation_id must be a positive integer")


def _normalize_target(path: Path) -> tuple[Path, Path]:
    if not isinstance(path, Path):
        raise TypeError("resolved_target must be a Path")
    requested = Path(os.path.abspath(path))
    canonical = requested.resolve(strict=False)
    if not canonical.is_absolute():
        raise ValueError("resolved_target must resolve to an absolute path")
    return requested, canonical


def _capture_target(path: Path) -> bytes | None:
    if path.resolve(strict=False) != path:
        raise OSError("Canonical backup target now resolves through a link.")
    io_path = HOST_FILESYSTEM.path_for_io(path)
    try:
        status = io_path.lstat()
    except FileNotFoundError:
        return None
    if not HOST_FILESYSTEM.is_regular_file(status):
        raise OSError("Backup target is not a regular file.")
    with _open_owned_file(path, within=path.parent, require_single_link=True) as stream:
        return stream.read()


def _directory_identity(path: Path) -> tuple[Path, int, int]:
    status = HOST_FILESYSTEM.path_for_io(path).lstat()
    if not HOST_FILESYSTEM.is_directory(status):
        raise OSError("Restore directory identity is unsafe.")
    return path, status.st_dev, status.st_ino


@contextmanager
def _open_owned_file(
    path: Path,
    *,
    within: Path,
    require_single_link: bool,
) -> Iterator[BinaryIO]:
    io_path = HOST_FILESYSTEM.path_for_io(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(io_path, flags)
    try:
        if require_single_link:
            HOST_FILESYSTEM.require_opened_owned_regular_file(descriptor, path, within=within)
        else:
            HOST_FILESYSTEM.require_opened_contained_regular_file(descriptor, path, within=within)
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        try:
            yield stream
        finally:
            stream.close()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_owned_file(path: Path, *, within: Path) -> bytes:
    HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
    with _open_owned_file(path, within=within, require_single_link=True) as stream:
        return stream.read()


def _hash_owned_file(path: Path, *, within: Path) -> str:
    HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
    hasher = hashlib.sha256()
    with _open_owned_file(path, within=within, require_single_link=True) as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _encode_signed_json(value: dict[str, object]) -> bytes:
    body = canonical_json_bytes(value)
    signed = dict(value)
    signed["integrity_sha256"] = sha256_hex(body)
    return canonical_json_bytes(signed)


def _decode_signed_json(content: bytes) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackupStoreError("Restore state contains invalid JSON.") from error
    if not isinstance(value, dict):
        raise BackupStoreError("Restore state must contain a JSON object.")
    checksum = value.get("integrity_sha256")
    body = {key: member for key, member in value.items() if key != "integrity_sha256"}
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        raise BackupStoreError("Restore state is missing its integrity hash.")
    if sha256_hex(canonical_json_bytes(body)) != checksum:
        raise BackupStoreError("Restore state integrity check failed.")
    return body


def _decode_state(content: bytes) -> _StoreState:
    value = _decode_signed_json(content)
    if set(value) not in (
        {
            "schema_version",
            "next_operation_id",
            "revision",
            "journal_operation_ids",
        },
        {
            "schema_version",
            "next_operation_id",
            "revision",
            "journal_operation_ids",
            "discarded_operation_ids",
        },
    ):
        raise BackupStoreError("Restore state fields do not match the schema.")
    if value["schema_version"] != _SCHEMA_VERSION:
        raise BackupStoreError("Restore state schema version is unsupported.")
    next_id = value["next_operation_id"]
    revision = value["revision"]
    operation_ids_value = value["journal_operation_ids"]
    discarded_ids_value = value.get("discarded_operation_ids", [])
    if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
        raise BackupStoreError("Restore state operation counter is invalid.")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise BackupStoreError("Restore state revision is invalid.")
    if not isinstance(operation_ids_value, list) or any(
        isinstance(operation_id, bool)
        or not isinstance(operation_id, int)
        or operation_id < 1
        or operation_id >= next_id
        for operation_id in operation_ids_value
    ):
        raise BackupStoreError("Restore state journal operation IDs are invalid.")
    operation_ids = tuple(operation_ids_value)
    if operation_ids != tuple(sorted(set(operation_ids))):
        raise BackupStoreError("Restore state journal operation IDs are invalid.")
    if not isinstance(discarded_ids_value, list) or any(
        isinstance(operation_id, bool)
        or not isinstance(operation_id, int)
        or operation_id < 1
        or operation_id >= next_id
        for operation_id in discarded_ids_value
    ):
        raise BackupStoreError("Restore state discarded operation IDs are invalid.")
    discarded_ids = tuple(discarded_ids_value)
    if discarded_ids != tuple(sorted(set(discarded_ids))):
        raise BackupStoreError("Restore state discarded operation IDs are invalid.")
    if set(operation_ids).intersection(discarded_ids):
        raise BackupStoreError("Restore state operation IDs cannot be both active and discarded.")
    return _StoreState(next_id, revision, operation_ids, discarded_ids)


def _file_state_object(state: FileState) -> dict[str, object]:
    return {"exists": state.exists, "sha256": state.sha256, "blob_name": state.blob_name}


def _entry_object(entry: BackupJournalEntry) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "backup",
        "operation_id": entry.operation_id,
        "revision": entry.revision,
        "run_token": str(entry.run_token),
        "requested_target": entry.requested_target,
        "canonical_target": entry.canonical_target,
        "before": _file_state_object(entry.before),
        "after": None if entry.after is None else _file_state_object(entry.after),
    }


def _gap_object(gap: BackupGap) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "gap",
        "operation_id": gap.operation_id,
        "revision": gap.revision,
        "run_token": str(gap.run_token),
        "requested_target": gap.requested_target,
        "canonical_target": gap.canonical_target,
        "reason": gap.reason,
    }


def _decode_file_state(value: object, *, before: bool) -> FileState:
    if not isinstance(value, dict) or set(value) != {"exists", "sha256", "blob_name"}:
        raise BackupStoreError("Journal file state is malformed.")
    exists = value["exists"]
    digest = value["sha256"]
    blob_name = value["blob_name"]
    if not isinstance(exists, bool):
        raise BackupStoreError("Journal file existence marker is malformed.")
    if not exists:
        if digest is not None or blob_name is not None:
            raise BackupStoreError("Absent file state must not reference content.")
        return FileState(False, None)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise BackupStoreError("Journal file hash is malformed.")
    if before:
        if not isinstance(blob_name, str) or _BLOB_NAME.fullmatch(blob_name) is None:
            raise BackupStoreError("Journal backup blob reference is malformed.")
    elif blob_name is not None:
        raise BackupStoreError("Post-write state must not reference a backup blob.")
    return FileState(True, digest, blob_name if before else None)


def _decode_operation(content: bytes) -> BackupJournalEntry | BackupGap:
    value = _decode_signed_json(content)
    common = {
        "schema_version",
        "kind",
        "operation_id",
        "revision",
        "run_token",
        "requested_target",
        "canonical_target",
    }
    if not common.issubset(value):
        raise BackupStoreError("Journal entry is missing required fields.")
    if value["schema_version"] != _SCHEMA_VERSION:
        raise BackupStoreError("Journal entry schema version is unsupported.")
    operation_id = value["operation_id"]
    revision = value["revision"]
    if isinstance(operation_id, bool) or not isinstance(operation_id, int) or operation_id < 1:
        raise BackupStoreError("Journal operation ID is invalid.")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise BackupStoreError("Journal revision is invalid.")
    run_token_value = value["run_token"]
    if not isinstance(run_token_value, str):
        raise BackupStoreError("Journal run token is invalid.")
    try:
        require_uuid4_string(run_token_value, field="run_token")
        run_token = UUID(run_token_value)
    except ValueError as error:
        raise BackupStoreError("Journal run token is invalid.") from error
    requested = value["requested_target"]
    canonical = value["canonical_target"]
    if (
        not isinstance(requested, str)
        or not Path(requested).is_absolute()
        or not isinstance(canonical, str)
        or not Path(canonical).is_absolute()
    ):
        raise BackupStoreError("Journal target path is invalid.")
    kind = value["kind"]
    if kind == "backup":
        if set(value) != common | {"before", "after"}:
            raise BackupStoreError("Journal backup fields do not match the schema.")
        before = _decode_file_state(value["before"], before=True)
        after_value = value["after"]
        after = None if after_value is None else _decode_file_state(after_value, before=False)
        return BackupJournalEntry(
            operation_id,
            revision,
            run_token,
            requested,
            canonical,
            before,
            after,
        )
    if kind == "gap":
        if set(value) != common | {"reason"}:
            raise BackupStoreError("Journal gap fields do not match the schema.")
        reason = value["reason"]
        if not isinstance(reason, str) or reason not in _GAP_CODES:
            raise BackupStoreError("Journal gap reason is invalid.")
        return BackupGap(operation_id, revision, run_token, requested, canonical, reason)
    raise BackupStoreError("Journal entry kind is invalid.")


__all__ = [
    "BackupGap",
    "BackupIntegrityError",
    "BackupIntegrityIssue",
    "BackupJournal",
    "BackupJournalEntry",
    "BackupStoreError",
    "BackupTicket",
    "FileBackupStore",
    "FileMutationRecorder",
    "FileState",
]
