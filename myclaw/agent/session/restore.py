"""Durable, storage-only Session Restore transactions."""

from __future__ import annotations

import base64
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from myclaw.agent.session._restore_persistence import (
    canonical_json_bytes,
    sha256_hex,
    sync_created_directory,
    sync_directory,
)
from myclaw.agent.session.backup_store import (
    BackupGap,
    BackupIntegrityIssue,
    BackupJournal,
    BackupJournalEntry,
    FileBackupStore,
)
from myclaw.agent.session.session import Session, SessionRestoreResult
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.workspace.state import WorkspaceState

_SCHEMA_VERSION = 1


class RestoreError(Exception):
    """Base error for a restore operation that cannot be completed safely."""


class RestoreRecoveryRequired(RestoreError):
    """The failed transaction must be finished by startup recovery."""


class StaleRestorePlan(RestoreError, ValueError):
    """The persisted Session or active journal changed after inspection."""


class RestoreModeUnavailable(RestoreError, ValueError):
    """The requested restore mode is not safe for the inspected range."""


class PendingRestoreError(RestoreError):
    """A different restore transaction is already pending for the Session."""


class RestoreSafetyError(RestoreError):
    """Required safety state could not be durably captured."""


class RestoreMode(StrEnum):
    """The two user-selectable restore scopes."""

    CONVERSATION_ONLY = "conversation-only"
    FILES = "files"


class RestoreFileStatus(StrEnum):
    """One durable outcome for one replay target."""

    RESTORED = "restored"
    UNCHANGED = "unchanged"
    FAILED = "failed"


class _RestorePhase(StrEnum):
    PREPARED = "prepared"
    JOURNAL_PRUNED = "journal_pruned"
    FILE_INTENT = "file_intent"
    FILE_REPLAY = "file_replay"
    FILES_REPLAYED = "files_replayed"
    SESSION_WRITE = "session_write"
    SESSION_PERSISTED = "session_persisted"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class RestoreTarget:
    """Immutable replay description for one canonical target."""

    canonical_target: Path
    operation_id: int
    requested_targets: tuple[Path, ...]
    before_exists: bool
    before_sha256: str | None
    before_bytes: bytes | None = field(repr=False)
    latest_after_exists: bool | None
    latest_after_sha256: str | None
    external: bool
    session_owned: bool
    backup_error: str | None = None


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """Frozen inspection result consumed by revalidation and execution."""

    session_id: str
    anchor_id: int
    session_digest: str
    journal_revision: int
    removed_users: int
    removed_messages: int
    targets: tuple[RestoreTarget, ...]
    external_target_count: int
    backup_gaps: tuple[BackupGap, ...]
    integrity_issues: tuple[BackupIntegrityIssue, ...]
    conflict_targets: tuple[Path, ...]
    discarded_run_tokens: tuple[UUID, ...]
    available_modes: tuple[RestoreMode, ...]


@dataclass(frozen=True, slots=True)
class RestoreFileResult:
    """Public result for one file target."""

    target: Path
    operation_id: int
    status: RestoreFileStatus
    conflict: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RestoreResult:
    """Public result of a completed or recovered restore transaction."""

    session_id: str
    anchor_id: int
    mode: RestoreMode
    removed_users: int
    removed_messages: int
    file_results: tuple[RestoreFileResult, ...]
    session_result: SessionRestoreResult
    failure_notification_acknowledged: bool

    @property
    def conflicts(self) -> tuple[Path, ...]:
        return tuple(item.target for item in self.file_results if item.conflict)

    @property
    def successful_conflicts(self) -> tuple[Path, ...]:
        return tuple(
            item.target
            for item in self.file_results
            if item.conflict and item.status is not RestoreFileStatus.FAILED
        )

    @property
    def failures(self) -> tuple[RestoreFileResult, ...]:
        return tuple(item for item in self.file_results if item.status is RestoreFileStatus.FAILED)

    @property
    def failed_files(self) -> tuple[Path, ...]:
        return tuple(item.target for item in self.failures)

    @property
    def failure_notification_pending(self) -> bool:
        return bool(self.failures) and not self.failure_notification_acknowledged


@dataclass(frozen=True, slots=True)
class _ObservedTarget:
    exists: bool
    sha256: str | None
    content: bytes | None = field(repr=False)


@dataclass(slots=True)
class _PendingTarget:
    target: RestoreTarget
    initial: _ObservedTarget | None
    preflight_error: str | None
    result: RestoreFileResult | None = None


@dataclass(slots=True)
class _PendingTransaction:
    session_id: str
    anchor_id: int
    mode: RestoreMode
    session_digest: str
    journal_revision: int
    discarded_run_tokens: tuple[UUID, ...]
    safety_generation: str
    removed_users: int
    removed_messages: int
    targets: list[_PendingTarget]
    phase: _RestorePhase = _RestorePhase.PREPARED
    current_operation_id: int | None = None
    session_result: SessionRestoreResult | None = None
    failure_notification_acknowledged: bool = False


class RestoreManager:
    """Own the durable restore protocol behind a small storage seam."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        session_id: str | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("workspace_state must be a WorkspaceState")
        if session_id is not None and not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        self._workspace_state = workspace_state
        self._session_id = session_id
        self._now = now

    def inspect(self, session: Session, anchor_id: int) -> RestorePlan:
        """Inspect persisted Session anchors and its active journal."""
        if not isinstance(session, Session):
            raise TypeError("session must be a Session")
        persisted = Session.load(
            self._workspace_state,
            session.session_id,
            now=self._now,
        )
        session_bytes = _read_session_bytes(self._workspace_state, persisted.session_id)
        candidates = persisted.restore_candidates()
        anchor_index = _anchor_index(persisted.messages, anchor_id)
        selected_tokens = tuple(
            UUID(str(message["restore_run_token"]))
            for message in persisted.messages[anchor_index:]
            if message.get("role") == "user" and "restore_run_token" in message
        )
        if not selected_tokens or not any(
            candidate.anchor_id == anchor_id for candidate in candidates
        ):
            raise ValueError(f"unknown persisted restore anchor ID: {anchor_id}")

        store = FileBackupStore(self._workspace_state, persisted.session_id)
        journal = store.inspect()
        selected_token_set = frozenset(selected_tokens)
        selected_entries = tuple(
            entry for entry in journal.entries if entry.run_token in selected_token_set
        )
        selected_gaps = tuple(gap for gap in journal.gaps if gap.run_token in selected_token_set)
        selected_integrity_issues = tuple(
            issue
            for issue in journal.integrity_issues
            if issue.run_token is None or issue.run_token in selected_token_set
        )
        targets = _build_targets(
            self._workspace_state,
            persisted.session_id,
            selected_entries,
            selected_integrity_issues,
            store,
        )
        conflicts = tuple(
            target.canonical_target for target in targets if _target_is_conflicting(target)
        )
        available_modes = (
            (RestoreMode.CONVERSATION_ONLY,)
            if not targets or selected_gaps or selected_integrity_issues
            else (RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES)
        )
        self._session_id = persisted.session_id
        return RestorePlan(
            session_id=persisted.session_id,
            anchor_id=anchor_id,
            session_digest=sha256_hex(session_bytes),
            journal_revision=journal.revision,
            removed_users=sum(
                message.get("role") == "user" for message in persisted.messages[anchor_index:]
            ),
            removed_messages=len(persisted.messages) - anchor_index,
            targets=targets,
            external_target_count=sum(target.external for target in targets),
            backup_gaps=selected_gaps,
            integrity_issues=selected_integrity_issues,
            conflict_targets=conflicts,
            discarded_run_tokens=selected_tokens,
            available_modes=available_modes,
        )

    def revalidate(self, plan: RestorePlan) -> RestorePlan:
        """Reject a plan whose persisted Session or active journal is stale."""
        if not isinstance(plan, RestorePlan):
            raise TypeError("plan must be a RestorePlan")
        if self._session_id is not None and self._session_id != plan.session_id:
            raise StaleRestorePlan("restore plan belongs to a different Session")
        session = Session.load(self._workspace_state, plan.session_id, now=self._now)
        current = self.inspect(session, plan.anchor_id)
        if (
            current.session_digest != plan.session_digest
            or current.journal_revision != plan.journal_revision
        ):
            raise StaleRestorePlan("restore plan is stale")
        return plan

    async def execute(self, plan: RestorePlan, mode: RestoreMode | str) -> RestoreResult:
        """Persist safety state, replay the selected range, and truncate Session."""
        plan = self.revalidate(plan)
        pending_path = self._workspace_state.path / "restore" / plan.session_id / "pending.json"
        try:
            return await self._execute_transaction(plan, mode)
        except StaleRestorePlan:
            raise
        except Exception as error:
            try:
                no_pending = not _path_exists(pending_path)
            except Exception:
                no_pending = False
            if no_pending:
                raise
            raise RestoreRecoveryRequired("restore requires startup recovery") from error

    async def _execute_transaction(
        self, plan: RestorePlan, mode: RestoreMode | str
    ) -> RestoreResult:
        selected_mode = _coerce_mode(mode)
        if selected_mode not in plan.available_modes:
            raise RestoreModeUnavailable(
                f"restore mode {selected_mode.value!r} is unavailable for this range"
            )
        root = _restore_session_root(self._workspace_state, plan.session_id)
        pending_path = root / "pending.json"
        if _path_exists(pending_path):
            existing = _read_pending(pending_path)
            if existing.session_id != plan.session_id:
                raise PendingRestoreError("pending restore belongs to a different Session")
            if existing.phase is not _RestorePhase.COMPLETE:
                raise PendingRestoreError("a restore transaction is already pending")
            if _failure_notification_pending(existing):
                raise PendingRestoreError("restore failure notification is not acknowledged")
            _archive_completed_pending(root, existing)

        session_bytes = _read_session_bytes(self._workspace_state, plan.session_id)
        store = FileBackupStore(self._workspace_state, plan.session_id)
        journal = store.inspect()
        if (
            sha256_hex(session_bytes) != plan.session_digest
            or journal.revision != plan.journal_revision
        ):
            raise StaleRestorePlan("restore plan changed during execution")

        pending_targets: list[_PendingTarget] = []
        initial_states: dict[str, _ObservedTarget] = {}
        if selected_mode is RestoreMode.FILES:
            for target in plan.targets:
                if target.session_owned:
                    pending_targets.append(_PendingTarget(target, None, None))
                    continue
                if _is_runtime_owned_target(self._workspace_state, target.canonical_target):
                    pending_targets.append(
                        _PendingTarget(
                            target,
                            None,
                            "runtime-owned target cannot be safely reloaded",
                        )
                    )
                    continue
                try:
                    initial = _observe_target(target.canonical_target)
                except Exception as error:
                    pending_targets.append(_PendingTarget(target, None, _error_text(error)))
                else:
                    initial_states[_path_key(target.canonical_target)] = initial
                    pending_targets.append(_PendingTarget(target, initial, None))
        else:
            pending_targets = [_PendingTarget(target, None, None) for target in plan.targets]

        generation = _write_safety_snapshot(
            self._workspace_state,
            plan,
            session_bytes,
            journal,
            store,
            initial_states,
        )
        pending = _PendingTransaction(
            session_id=plan.session_id,
            anchor_id=plan.anchor_id,
            mode=selected_mode,
            session_digest=plan.session_digest,
            journal_revision=plan.journal_revision,
            discarded_run_tokens=plan.discarded_run_tokens,
            safety_generation=generation,
            removed_users=plan.removed_users,
            removed_messages=plan.removed_messages,
            targets=pending_targets,
        )
        _write_pending(self._workspace_state, pending)
        return await self._continue_pending(pending)

    async def recover_pending(self) -> RestoreResult | None:
        """Finish the current Session's pending transaction, if one exists."""
        session_id = self._session_id
        if session_id is None:
            session_id = _find_pending_session(self._workspace_state)
            if session_id is None:
                return None
            self._session_id = session_id
        root = _restore_session_root(self._workspace_state, session_id)
        pending_path = root / "pending.json"
        if not _path_exists(pending_path):
            return None
        pending = _read_pending(pending_path)
        if pending.session_id != session_id:
            raise PendingRestoreError("pending restore belongs to a different Session")
        return await self._continue_pending(pending)

    def acknowledge_failure_notification(self) -> RestoreResult | None:
        """Durably acknowledge the completed restore's file-failure notification."""
        session_id = self._session_id
        if session_id is None:
            session_id = _find_pending_session(self._workspace_state)
            if session_id is None:
                return None
            self._session_id = session_id
        root = _restore_session_root(self._workspace_state, session_id)
        pending_path = root / "pending.json"
        if not _path_exists(pending_path):
            return None
        pending = _read_pending(pending_path)
        if pending.session_id != session_id:
            raise PendingRestoreError("pending restore belongs to a different Session")
        if pending.phase is not _RestorePhase.COMPLETE:
            raise PendingRestoreError("restore transaction is not complete")
        if _failure_notification_pending(pending):
            pending.failure_notification_acknowledged = True
            _write_pending(self._workspace_state, pending)
        return _result_from_pending(pending)

    async def _continue_pending(self, pending: _PendingTransaction) -> RestoreResult:
        if pending.phase is _RestorePhase.COMPLETE:
            if pending.session_result is None:
                raise PendingRestoreError("completed restore has no Session result")
            return _result_from_pending(pending)
        _verify_safety_snapshot(self._workspace_state, pending)
        store = FileBackupStore(self._workspace_state, pending.session_id)

        if pending.phase is _RestorePhase.PREPARED:
            store.discard_run_tokens(pending.discarded_run_tokens)
            pending.phase = _RestorePhase.JOURNAL_PRUNED
            _write_pending(self._workspace_state, pending)

        if pending.mode is RestoreMode.FILES and pending.phase in {
            _RestorePhase.JOURNAL_PRUNED,
            _RestorePhase.FILE_INTENT,
            _RestorePhase.FILE_REPLAY,
        }:
            for item in pending.targets:
                if item.result is not None:
                    continue
                already_attempted = (
                    pending.phase is _RestorePhase.FILE_INTENT
                    and pending.current_operation_id == item.target.operation_id
                )
                pending.current_operation_id = item.target.operation_id
                pending.phase = _RestorePhase.FILE_INTENT
                _write_pending(self._workspace_state, pending)
                if item.preflight_error is not None:
                    item.result = RestoreFileResult(
                        target=item.target.canonical_target,
                        operation_id=item.target.operation_id,
                        status=RestoreFileStatus.FAILED,
                        error=item.preflight_error,
                    )
                elif item.target.session_owned:
                    item.result = RestoreFileResult(
                        target=item.target.canonical_target,
                        operation_id=item.target.operation_id,
                        status=RestoreFileStatus.UNCHANGED,
                        error="session-owned target is written by Session Restore",
                    )
                else:
                    item.result = _replay_target(
                        item,
                        attempted=already_attempted,
                    )
                pending.current_operation_id = None
                pending.phase = _RestorePhase.FILE_REPLAY
                _write_pending(self._workspace_state, pending)
            pending.phase = _RestorePhase.FILES_REPLAYED
            _write_pending(self._workspace_state, pending)
        elif pending.phase is _RestorePhase.JOURNAL_PRUNED:
            pending.phase = _RestorePhase.FILES_REPLAYED
            _write_pending(self._workspace_state, pending)

        if pending.phase in {_RestorePhase.FILES_REPLAYED, _RestorePhase.SESSION_WRITE}:
            if pending.phase is _RestorePhase.FILES_REPLAYED:
                pending.phase = _RestorePhase.SESSION_WRITE
                _write_pending(self._workspace_state, pending)
            if pending.session_result is None:
                session = Session.load(
                    self._workspace_state,
                    pending.session_id,
                    now=self._now,
                )
                if any(
                    message.get("restore_anchor_id") == pending.anchor_id
                    for message in session.messages
                ):
                    pending.session_result = session.restore_before_durably(pending.anchor_id)
                else:
                    pending.session_result = SessionRestoreResult(
                        session_id=pending.session_id,
                        anchor_id=pending.anchor_id,
                        removed_messages=pending.removed_messages,
                        updated_at=session.updated_at,
                    )
            pending.phase = _RestorePhase.SESSION_PERSISTED
            _write_pending(self._workspace_state, pending)

        pending.phase = _RestorePhase.COMPLETE
        _write_pending(self._workspace_state, pending)
        return _result_from_pending(pending)


def _coerce_mode(value: RestoreMode | str) -> RestoreMode:
    if isinstance(value, RestoreMode):
        return value
    if not isinstance(value, str):
        raise TypeError("restore mode must be a RestoreMode or string")
    try:
        return RestoreMode(value)
    except ValueError as error:
        raise ValueError(f"unknown restore mode: {value}") from error


def _is_runtime_owned_target(workspace_state: WorkspaceState, target: Path) -> bool:
    state_root = workspace_state.path.resolve(strict=False)
    return target.is_relative_to(state_root)


def _replay_target(item: _PendingTarget, *, attempted: bool = False) -> RestoreFileResult:
    target = item.target
    if target.backup_error is not None:
        return RestoreFileResult(
            target=target.canonical_target,
            operation_id=target.operation_id,
            status=RestoreFileStatus.FAILED,
            error=f"backup unavailable: {target.backup_error}",
        )
    if target.before_exists and target.before_bytes is None:
        return RestoreFileResult(
            target=target.canonical_target,
            operation_id=target.operation_id,
            status=RestoreFileStatus.FAILED,
            error="backup bytes are unavailable",
        )
    try:
        current = _observe_target(target.canonical_target)
    except Exception as error:
        return RestoreFileResult(
            target=target.canonical_target,
            operation_id=target.operation_id,
            status=RestoreFileStatus.FAILED,
            error=_error_text(error),
        )
    desired = target.before_bytes if target.before_exists else None
    if current.exists == target.before_exists and current.content == desired:
        was_already_matching = (
            item.initial is not None
            and item.initial.exists == target.before_exists
            and item.initial.content == desired
        )
        return RestoreFileResult(
            target=target.canonical_target,
            operation_id=target.operation_id,
            status=(
                RestoreFileStatus.UNCHANGED
                if not attempted or was_already_matching
                else RestoreFileStatus.RESTORED
            ),
            conflict=attempted and _initial_state_is_conflicting(item),
        )
    conflict = target.latest_after_exists is not None and (
        (current.exists, current.sha256) != (target.latest_after_exists, target.latest_after_sha256)
    )
    try:
        if target.before_exists:
            assert desired is not None
            HOST_FILESYSTEM.atomic_replace_bytes(target.canonical_target, desired)
        else:
            _delete_target(target.canonical_target)
    except Exception as error:
        return RestoreFileResult(
            target=target.canonical_target,
            operation_id=target.operation_id,
            status=RestoreFileStatus.FAILED,
            conflict=conflict,
            error=_error_text(error),
        )
    return RestoreFileResult(
        target=target.canonical_target,
        operation_id=target.operation_id,
        status=RestoreFileStatus.RESTORED,
        conflict=conflict,
    )


def _initial_state_is_conflicting(item: _PendingTarget) -> bool:
    initial = item.initial
    target = item.target
    if initial is None or target.latest_after_exists is None:
        return False
    return (initial.exists, initial.sha256) != (
        target.latest_after_exists,
        target.latest_after_sha256,
    )


def _delete_target(path: Path) -> None:
    current = _observe_target(path)
    if not current.exists:
        return
    owned = HOST_FILESYSTEM.require_owned_regular_file(path, within=path.parent)
    HOST_FILESYSTEM.path_for_io(owned).unlink()
    sync_directory(owned.parent)


def _error_text(error: Exception) -> str:
    text = str(error).strip()
    return text or type(error).__name__


def _restore_session_root(workspace_state: WorkspaceState, session_id: str) -> Path:
    workspace_path = HOST_FILESYSTEM.path_for_io(workspace_state.workspace_path)
    workspace_root = HOST_FILESYSTEM.require_owned_directory(
        workspace_path,
        within=workspace_path,
    )
    state_root = _ensure_private_directory(workspace_state.path, workspace_root)
    restore_root = _ensure_private_directory(workspace_state.path / "restore", state_root)
    return _ensure_private_directory(restore_root / session_id, restore_root)


def _ensure_private_directory(path: Path, within: Path) -> Path:
    io_path = HOST_FILESYSTEM.path_for_io(path)
    created = False
    if not _path_exists(path):
        io_path.mkdir(mode=0o700)
        created = True
    owned = HOST_FILESYSTEM.require_owned_directory(path, within=within)
    HOST_FILESYSTEM.restrict_private_directory(owned)
    if created:
        sync_created_directory(owned)
    return owned


def _write_internal(path: Path, content: bytes, *, within: Path) -> None:
    if _path_exists(path):
        HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
    HOST_FILESYSTEM.atomic_replace_bytes(path, content)
    HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
    HOST_FILESYSTEM.restrict_private_file(path)


def _write_json(path: Path, value: dict[str, object], *, within: Path) -> None:
    _write_internal(path, canonical_json_bytes(value), within=within)


def _write_safety_snapshot(
    workspace_state: WorkspaceState,
    plan: RestorePlan,
    session_bytes: bytes,
    journal: BackupJournal,
    store: FileBackupStore,
    initial_states: dict[str, _ObservedTarget],
) -> str:
    root = _restore_session_root(workspace_state, plan.session_id)
    latest_root = _ensure_private_directory(root / "latest-safety", root)
    generation = f"snapshot-{os.urandom(16).hex()}"
    generation_root = _ensure_private_directory(latest_root / generation, latest_root)
    try:
        journal_content = canonical_json_bytes(_journal_object(journal, store))
        targets_content = canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "targets": [
                    {
                        "canonical_target": str(path),
                        **_observed_object(observed),
                    }
                    for path, observed in sorted(initial_states.items())
                ],
            }
        )
        _write_internal(generation_root / "session.jsonl", session_bytes, within=generation_root)
        _write_internal(generation_root / "journal.json", journal_content, within=generation_root)
        _write_internal(generation_root / "targets.json", targets_content, within=generation_root)
        _write_internal(generation_root / "complete", b"complete\n", within=generation_root)
        _write_json(
            latest_root / "manifest.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "session_id": plan.session_id,
                "generation": generation,
                "session_digest": plan.session_digest,
                "journal_revision": plan.journal_revision,
                "journal_digest": sha256_hex(journal_content),
                "targets_digest": sha256_hex(targets_content),
            },
            within=latest_root,
        )
        for child in HOST_FILESYSTEM.path_for_io(latest_root).iterdir():
            if child.name in {generation, "manifest.json"}:
                continue
            if child.is_dir():
                try:
                    shutil.rmtree(child)
                except OSError:
                    pass
    except Exception as error:
        try:
            shutil.rmtree(HOST_FILESYSTEM.path_for_io(generation_root))
        except Exception:
            pass
        raise RestoreSafetyError(
            "required restore safety snapshot could not be persisted"
        ) from error
    return generation


def _journal_object(journal: BackupJournal, store: FileBackupStore) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for entry in journal.entries:
        backup: str | None = None
        backup_error: str | None = None
        if entry.before.exists:
            try:
                content = store.read_backup(entry.operation_id)
                if content is not None:
                    backup = _encode_bytes(content)
            except Exception as error:
                backup_error = _error_text(error)
        entries.append(
            {
                "operation_id": entry.operation_id,
                "revision": entry.revision,
                "run_token": str(entry.run_token),
                "requested_target": entry.requested_target,
                "canonical_target": entry.canonical_target,
                "before": {
                    "exists": entry.before.exists,
                    "sha256": entry.before.sha256,
                },
                "after": None
                if entry.after is None
                else {"exists": entry.after.exists, "sha256": entry.after.sha256},
                "backup_b64": backup,
                "backup_error": backup_error,
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "revision": journal.revision,
        "entries": entries,
        "gaps": [
            {
                "operation_id": gap.operation_id,
                "revision": gap.revision,
                "run_token": str(gap.run_token),
                "requested_target": gap.requested_target,
                "canonical_target": gap.canonical_target,
                "reason": gap.reason,
            }
            for gap in journal.gaps
        ],
        "integrity_issues": [
            {"operation_id": issue.operation_id, "reason": issue.reason}
            for issue in journal.integrity_issues
        ],
    }


def _verify_safety_snapshot(workspace_state: WorkspaceState, pending: _PendingTransaction) -> None:
    try:
        root = _restore_session_root(workspace_state, pending.session_id)
        latest_root = HOST_FILESYSTEM.require_owned_directory(root / "latest-safety", within=root)
        manifest_path = latest_root / "manifest.json"
        manifest = _read_json(manifest_path, within=latest_root)
        if (
            set(manifest)
            != {
                "schema_version",
                "session_id",
                "generation",
                "session_digest",
                "journal_revision",
                "journal_digest",
                "targets_digest",
            }
            or manifest.get("schema_version") != _SCHEMA_VERSION
            or manifest.get("session_id") != pending.session_id
            or manifest.get("generation") != pending.safety_generation
            or manifest.get("session_digest") != pending.session_digest
            or manifest.get("journal_revision") != pending.journal_revision
            or not isinstance(manifest.get("journal_digest"), str)
            or not isinstance(manifest.get("targets_digest"), str)
        ):
            raise RestoreSafetyError("pending restore safety snapshot manifest is invalid")
        generation_root = latest_root / pending.safety_generation
        HOST_FILESYSTEM.require_owned_directory(generation_root, within=latest_root)
        complete = _read_internal(generation_root / "complete", within=generation_root)
        session_content = _read_internal(generation_root / "session.jsonl", within=generation_root)
        journal_content = _read_internal(generation_root / "journal.json", within=generation_root)
        targets_content = _read_internal(generation_root / "targets.json", within=generation_root)
        if complete != b"complete\n" or sha256_hex(session_content) != pending.session_digest:
            raise RestoreSafetyError("pending restore safety snapshot content is invalid")
        if sha256_hex(journal_content) != manifest["journal_digest"]:
            raise RestoreSafetyError("pending restore safety journal digest is invalid")
        if sha256_hex(targets_content) != manifest["targets_digest"]:
            raise RestoreSafetyError("pending restore safety target digest is invalid")
        journal_value = _decode_json_object(journal_content)
        if (
            journal_value.get("schema_version") != _SCHEMA_VERSION
            or journal_value.get("revision") != pending.journal_revision
        ):
            raise RestoreSafetyError("pending restore safety journal is invalid")
        targets_value = _decode_json_object(targets_content)
        expected_targets = [
            {
                "canonical_target": _path_key(item.target.canonical_target),
                **_observed_object(item.initial),
            }
            for item in pending.targets
            if item.initial is not None
        ]
        if targets_value.get("schema_version") != _SCHEMA_VERSION or targets_value.get(
            "targets"
        ) != sorted(expected_targets, key=lambda item: str(item["canonical_target"])):
            raise RestoreSafetyError("pending restore safety targets are invalid")
    except RestoreSafetyError:
        raise
    except Exception as error:
        raise RestoreSafetyError("pending restore safety snapshot is unavailable") from error


def _find_pending_session(workspace_state: WorkspaceState) -> str | None:
    restore_root = workspace_state.path / "restore"
    if not _path_exists(restore_root):
        return None
    owned_root = HOST_FILESYSTEM.require_owned_directory(restore_root, within=workspace_state.path)
    incomplete: list[str] = []
    unacknowledged: list[str] = []
    for child in HOST_FILESYSTEM.path_for_io(owned_root).iterdir():
        try:
            owned_child = HOST_FILESYSTEM.require_owned_directory(child, within=owned_root)
        except (FileNotFoundError, OSError):
            continue
        candidate = owned_child / "pending.json"
        if _path_exists(candidate):
            pending = _read_pending(candidate)
            if pending.session_id != child.name:
                raise PendingRestoreError("pending restore belongs to a different Session")
            if pending.phase is not _RestorePhase.COMPLETE:
                incomplete.append(child.name)
            elif _failure_notification_pending(pending):
                unacknowledged.append(child.name)
    found = incomplete or unacknowledged
    return sorted(found)[0] if found else None


def _archive_completed_pending(root: Path, pending: _PendingTransaction) -> None:
    results_root = _ensure_private_directory(root / "results", root)
    _write_json(
        results_root / f"restore-{os.urandom(12).hex()}.json",
        _pending_object(pending),
        within=results_root,
    )


def _write_pending(workspace_state: WorkspaceState, pending: _PendingTransaction) -> None:
    root = _restore_session_root(workspace_state, pending.session_id)
    _write_json(root / "pending.json", _pending_object(pending), within=root)


def _read_pending(path: Path) -> _PendingTransaction:
    value = _read_json(path, within=path.parent)
    try:
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise PendingRestoreError("restore pending schema version is unsupported")
        session_id = value["session_id"]
        anchor_id = value["anchor_id"]
        session_digest = value["session_digest"]
        journal_revision = value["journal_revision"]
        safety_generation = value["safety_generation"]
        removed_users = value["removed_users"]
        removed_messages = value["removed_messages"]
        phase_value = value["phase"]
        current_operation_id = value.get("current_operation_id")
        tokens_value = value["discarded_run_tokens"]
        targets_value = value["targets"]
        acknowledged = value["failure_notification_acknowledged"]
        if (
            not isinstance(session_id, str)
            or isinstance(anchor_id, bool)
            or not isinstance(anchor_id, int)
            or anchor_id < 1
            or not isinstance(session_digest, str)
            or isinstance(journal_revision, bool)
            or not isinstance(journal_revision, int)
            or journal_revision < 0
            or not isinstance(safety_generation, str)
            or not isinstance(removed_users, int)
            or removed_users < 0
            or not isinstance(removed_messages, int)
            or removed_messages < 0
            or not isinstance(phase_value, str)
            or (
                current_operation_id is not None
                and (
                    isinstance(current_operation_id, bool)
                    or not isinstance(current_operation_id, int)
                    or current_operation_id < 1
                )
            )
            or not isinstance(tokens_value, list)
            or not isinstance(targets_value, list)
            or not isinstance(acknowledged, bool)
        ):
            raise PendingRestoreError("restore pending fields are malformed")
        tokens = tuple(UUID(str(token)) for token in tokens_value)
        mode = _coerce_mode(value["mode"])
        phase = _RestorePhase(phase_value)
        targets = [_pending_target_from_object(item) for item in targets_value]
        parsed_current_operation_id = (
            None if current_operation_id is None else int(current_operation_id)
        )
        session_result_value = value.get("session_result")
        session_result = (
            None
            if session_result_value is None
            else _session_result_from_object(session_result_value)
        )
        return _PendingTransaction(
            session_id=session_id,
            anchor_id=anchor_id,
            mode=mode,
            session_digest=session_digest,
            journal_revision=journal_revision,
            discarded_run_tokens=tokens,
            safety_generation=safety_generation,
            removed_users=removed_users,
            removed_messages=removed_messages,
            targets=targets,
            phase=phase,
            current_operation_id=parsed_current_operation_id,
            session_result=session_result,
            failure_notification_acknowledged=acknowledged,
        )
    except PendingRestoreError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise PendingRestoreError("restore pending fields are malformed") from error


def _read_json(path: Path, *, within: Path) -> dict[str, Any]:
    return _decode_json_object(_read_internal(path, within=within))


def _read_internal(path: Path, *, within: Path) -> bytes:
    owned = HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
    io_path = HOST_FILESYSTEM.path_for_io(owned)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(io_path, flags)
    try:
        HOST_FILESYSTEM.require_opened_owned_regular_file(descriptor, owned, within=within)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _decode_json_object(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PendingRestoreError("restore persistence contains invalid JSON") from error
    if not isinstance(value, dict):
        raise PendingRestoreError("restore persistence must contain a JSON object")
    return value


def _pending_object(pending: _PendingTransaction) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_id": pending.session_id,
        "anchor_id": pending.anchor_id,
        "mode": pending.mode.value,
        "session_digest": pending.session_digest,
        "journal_revision": pending.journal_revision,
        "discarded_run_tokens": [str(token) for token in pending.discarded_run_tokens],
        "safety_generation": pending.safety_generation,
        "removed_users": pending.removed_users,
        "removed_messages": pending.removed_messages,
        "phase": pending.phase.value,
        "current_operation_id": pending.current_operation_id,
        "targets": [_pending_target_object(item) for item in pending.targets],
        "session_result": None
        if pending.session_result is None
        else _session_result_object(pending.session_result),
        "failure_notification_acknowledged": pending.failure_notification_acknowledged,
    }


def _pending_target_object(item: _PendingTarget) -> dict[str, object]:
    return {
        "target": _target_object(item.target),
        "initial": None if item.initial is None else _observed_object(item.initial),
        "preflight_error": item.preflight_error,
        "result": None if item.result is None else _file_result_object(item.result),
    }


def _pending_target_from_object(value: object) -> _PendingTarget:
    if not isinstance(value, dict):
        raise PendingRestoreError("restore pending target is malformed")
    target = _target_from_object(value.get("target"))
    initial_value = value.get("initial")
    initial = None if initial_value is None else _observed_from_object(initial_value)
    preflight_error = value.get("preflight_error")
    if preflight_error is not None and not isinstance(preflight_error, str):
        raise PendingRestoreError("restore pending target error is malformed")
    result_value = value.get("result")
    result = None if result_value is None else _file_result_from_object(result_value)
    return _PendingTarget(target, initial, preflight_error, result)


def _target_object(target: RestoreTarget) -> dict[str, object]:
    return {
        "canonical_target": str(target.canonical_target),
        "operation_id": target.operation_id,
        "requested_targets": [str(path) for path in target.requested_targets],
        "before_exists": target.before_exists,
        "before_sha256": target.before_sha256,
        "before_b64": None if target.before_bytes is None else _encode_bytes(target.before_bytes),
        "latest_after_exists": target.latest_after_exists,
        "latest_after_sha256": target.latest_after_sha256,
        "external": target.external,
        "session_owned": target.session_owned,
        "backup_error": target.backup_error,
    }


def _target_from_object(value: object) -> RestoreTarget:
    if not isinstance(value, dict):
        raise PendingRestoreError("restore target is malformed")
    canonical = value.get("canonical_target")
    operation_id = value.get("operation_id")
    requested = value.get("requested_targets")
    before_exists = value.get("before_exists")
    before_sha256 = value.get("before_sha256")
    latest_exists = value.get("latest_after_exists")
    latest_sha256 = value.get("latest_after_sha256")
    external = value.get("external")
    session_owned = value.get("session_owned")
    backup_error = value.get("backup_error")
    if (
        not isinstance(canonical, str)
        or not Path(canonical).is_absolute()
        or isinstance(operation_id, bool)
        or not isinstance(operation_id, int)
        or operation_id < 1
        or not isinstance(requested, list)
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in requested)
        or not isinstance(before_exists, bool)
        or (before_sha256 is not None and not isinstance(before_sha256, str))
        or (latest_exists is not None and not isinstance(latest_exists, bool))
        or (latest_sha256 is not None and not isinstance(latest_sha256, str))
        or not isinstance(external, bool)
        or not isinstance(session_owned, bool)
        or (backup_error is not None and not isinstance(backup_error, str))
    ):
        raise PendingRestoreError("restore target fields are malformed")
    before_bytes = _decode_bytes(value.get("before_b64"))
    if before_exists:
        if before_bytes is not None and sha256_hex(before_bytes) != before_sha256:
            raise PendingRestoreError("restore target backup digest does not match its bytes")
    elif before_bytes is not None:
        raise PendingRestoreError("absent restore target must not contain backup bytes")
    return RestoreTarget(
        canonical_target=Path(canonical),
        operation_id=operation_id,
        requested_targets=tuple(Path(path) for path in requested),
        before_exists=before_exists,
        before_sha256=before_sha256,
        before_bytes=before_bytes,
        latest_after_exists=latest_exists,
        latest_after_sha256=latest_sha256,
        external=external,
        session_owned=session_owned,
        backup_error=backup_error,
    )


def _observed_object(observed: _ObservedTarget) -> dict[str, object]:
    return {
        "exists": observed.exists,
        "sha256": observed.sha256,
        "content_b64": None if observed.content is None else _encode_bytes(observed.content),
    }


def _observed_from_object(value: object) -> _ObservedTarget:
    if not isinstance(value, dict):
        raise PendingRestoreError("restore observed target is malformed")
    exists = value.get("exists")
    digest = value.get("sha256")
    if not isinstance(exists, bool) or (digest is not None and not isinstance(digest, str)):
        raise PendingRestoreError("restore observed target fields are malformed")
    content = _decode_bytes(value.get("content_b64"))
    if exists != (content is not None) or (content is not None and sha256_hex(content) != digest):
        raise PendingRestoreError("restore observed target digest does not match its bytes")
    if not exists and digest is not None:
        raise PendingRestoreError("absent observed target must not contain a digest")
    return _ObservedTarget(exists, digest, content)


def _file_result_object(result: RestoreFileResult) -> dict[str, object]:
    return {
        "target": str(result.target),
        "operation_id": result.operation_id,
        "status": result.status.value,
        "conflict": result.conflict,
        "error": result.error,
    }


def _file_result_from_object(value: object) -> RestoreFileResult:
    if not isinstance(value, dict):
        raise PendingRestoreError("restore file result is malformed")
    target = value.get("target")
    operation_id = value.get("operation_id")
    status = value.get("status")
    conflict = value.get("conflict")
    error = value.get("error")
    if (
        not isinstance(target, str)
        or not Path(target).is_absolute()
        or isinstance(operation_id, bool)
        or not isinstance(operation_id, int)
        or operation_id < 1
        or not isinstance(status, str)
        or not isinstance(conflict, bool)
        or (error is not None and not isinstance(error, str))
    ):
        raise PendingRestoreError("restore file result fields are malformed")
    try:
        parsed_status = RestoreFileStatus(status)
    except ValueError as parse_error:
        raise PendingRestoreError("restore file result status is malformed") from parse_error
    return RestoreFileResult(Path(target), operation_id, parsed_status, conflict, error)


def _session_result_object(result: SessionRestoreResult) -> dict[str, object]:
    return {
        "session_id": result.session_id,
        "anchor_id": result.anchor_id,
        "removed_messages": result.removed_messages,
        "updated_at": result.updated_at.isoformat(),
    }


def _session_result_from_object(value: object) -> SessionRestoreResult:
    if not isinstance(value, dict):
        raise PendingRestoreError("restore Session result is malformed")
    session_id = value.get("session_id")
    anchor_id = value.get("anchor_id")
    removed_messages = value.get("removed_messages")
    updated_at = value.get("updated_at")
    if (
        not isinstance(session_id, str)
        or isinstance(anchor_id, bool)
        or not isinstance(anchor_id, int)
        or anchor_id < 1
        or isinstance(removed_messages, bool)
        or not isinstance(removed_messages, int)
        or removed_messages < 0
        or not isinstance(updated_at, str)
    ):
        raise PendingRestoreError("restore Session result fields are malformed")
    try:
        parsed_updated_at = datetime.fromisoformat(updated_at)
    except ValueError as error:
        raise PendingRestoreError("restore Session result timestamp is malformed") from error
    return SessionRestoreResult(session_id, anchor_id, removed_messages, parsed_updated_at)


def _result_from_pending(pending: _PendingTransaction) -> RestoreResult:
    if pending.session_result is None:
        raise PendingRestoreError("restore transaction has no Session result")
    return RestoreResult(
        session_id=pending.session_id,
        anchor_id=pending.anchor_id,
        mode=pending.mode,
        removed_users=pending.removed_users,
        removed_messages=pending.removed_messages,
        file_results=tuple(item.result for item in pending.targets if item.result is not None),
        session_result=pending.session_result,
        failure_notification_acknowledged=pending.failure_notification_acknowledged,
    )


def _path_exists(path: Path) -> bool:
    try:
        HOST_FILESYSTEM.path_for_io(path).lstat()
    except FileNotFoundError:
        return False
    return True


def _encode_bytes(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _decode_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PendingRestoreError("restore byte field is malformed")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise PendingRestoreError("restore byte field is malformed") from error


def _failure_notification_pending(pending: _PendingTransaction) -> bool:
    return (
        any(
            item.result is not None and item.result.status is RestoreFileStatus.FAILED
            for item in pending.targets
        )
        and not pending.failure_notification_acknowledged
    )


def _anchor_index(messages: list[dict[str, Any]], anchor_id: int) -> int:
    if isinstance(anchor_id, bool) or not isinstance(anchor_id, int) or anchor_id < 1:
        raise ValueError("anchor_id must be a positive integer")
    for index, message in enumerate(messages):
        if message.get("restore_anchor_id") == anchor_id:
            return index
    raise ValueError(f"unknown persisted restore anchor ID: {anchor_id}")


def _build_targets(
    workspace_state: WorkspaceState,
    session_id: str,
    entries: tuple[BackupJournalEntry, ...],
    integrity_issues: tuple[BackupIntegrityIssue, ...],
    store: FileBackupStore,
) -> tuple[RestoreTarget, ...]:
    grouped: dict[str, list[BackupJournalEntry]] = {}
    for entry in entries:
        grouped.setdefault(_path_key(Path(entry.canonical_target)), []).append(entry)
    issue_ids = {issue.operation_id for issue in integrity_issues}
    session_path = _session_path(workspace_state, session_id)
    workspace_root = workspace_state.workspace_path.resolve(strict=False)
    result: list[RestoreTarget] = []
    for group in grouped.values():
        first = group[0]
        latest = group[-1]
        canonical = Path(first.canonical_target)
        backup_error: str | None = None
        before_bytes: bytes | None = None
        if first.operation_id in issue_ids:
            backup_error = next(
                issue.reason
                for issue in integrity_issues
                if issue.operation_id == first.operation_id
            )
        elif first.before.exists:
            try:
                before_bytes = store.read_backup(first.operation_id)
            except Exception as error:
                backup_error = type(error).__name__
        if first.before.exists and before_bytes is None and backup_error is None:
            backup_error = "backup_unavailable"
        current_after = latest.after
        result.append(
            RestoreTarget(
                canonical_target=canonical,
                operation_id=first.operation_id,
                requested_targets=tuple(Path(entry.requested_target) for entry in group),
                before_exists=first.before.exists,
                before_sha256=first.before.sha256,
                before_bytes=before_bytes,
                latest_after_exists=None if current_after is None else current_after.exists,
                latest_after_sha256=None if current_after is None else current_after.sha256,
                external=not canonical.is_relative_to(workspace_root),
                session_owned=_path_key(canonical) == _path_key(session_path),
                backup_error=backup_error,
            )
        )
    return tuple(result)


def _target_is_conflicting(target: RestoreTarget) -> bool:
    try:
        current = _observe_target(target.canonical_target)
    except Exception:
        return False
    if target.latest_after_exists is None:
        return False
    return (current.exists, current.sha256) != (
        target.latest_after_exists,
        target.latest_after_sha256,
    )


def _session_path(workspace_state: WorkspaceState, session_id: str) -> Path:
    return workspace_state.sessions_directory / f"{session_id}.jsonl"


def _read_session_bytes(workspace_state: WorkspaceState, session_id: str) -> bytes:
    sessions = workspace_state.existing_sessions_directory()
    if sessions is None:
        raise FileNotFoundError(workspace_state.sessions_directory)
    path = sessions / f"{session_id}.jsonl"
    owned = HOST_FILESYSTEM.require_owned_regular_file(path, within=sessions)
    return HOST_FILESYSTEM.path_for_io(owned).read_bytes()


def _observe_target(path: Path) -> _ObservedTarget:
    canonical = Path(os.path.abspath(path))
    if _path_key(canonical.resolve(strict=False)) != _path_key(canonical):
        raise OSError("restore target resolves through a link")
    io_path = HOST_FILESYSTEM.path_for_io(canonical)
    try:
        status = io_path.lstat()
    except FileNotFoundError:
        return _ObservedTarget(False, None, None)
    if not HOST_FILESYSTEM.is_regular_file(status) or status.st_nlink != 1:
        raise OSError("restore target is not a safe regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(io_path, flags)
    try:
        HOST_FILESYSTEM.require_opened_owned_regular_file(
            descriptor,
            canonical,
            within=canonical.parent,
        )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            content = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _ObservedTarget(True, sha256_hex(content), content)


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


__all__ = [
    "PendingRestoreError",
    "RestoreError",
    "RestoreFileResult",
    "RestoreFileStatus",
    "RestoreManager",
    "RestoreMode",
    "RestoreModeUnavailable",
    "RestorePlan",
    "RestoreResult",
    "RestoreSafetyError",
    "RestoreTarget",
    "StaleRestorePlan",
]
