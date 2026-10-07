"""Durable, Workspace-owned deletion of one foreground Conversation Session."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast
from uuid import uuid4

from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_MARKER_VERSION = 1
_LOG_SUFFIX = re.compile(r"\.([^.]+(?:\.[^.]+)*)?\.log$")


class SessionDeletionPending(FileNotFoundError):
    """Raised when a foreground Session is durably fenced for deletion."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session deletion is pending for {session_id}")


def session_deletion_pending(workspace_state: WorkspaceState, session_id: str) -> bool:
    """Return whether a valid deletion marker fences one foreground Session."""
    _require_foreground_id(session_id)
    marker_root = _existing_marker_root(workspace_state)
    if marker_root is None:
        return False
    marker = marker_root / f"{session_id}.json"
    if not HOST_FILESYSTEM.entry_exists(marker):
        return False
    _read_marker(marker, marker_root, session_id)
    return True


def begin_session_deletion(workspace_state: WorkspaceState, session_id: str) -> str:
    """Publish an idempotent deletion marker before touching Session-owned data."""
    _require_foreground_id(session_id)
    if session_restore_pending(workspace_state, session_id):
        raise RuntimeError("Session Restore must finish before deletion")
    marker_root = _prepare_marker_root(workspace_state)
    marker = marker_root / f"{session_id}.json"
    if HOST_FILESYSTEM.entry_exists(marker):
        return _read_marker(marker, marker_root, session_id)
    operation_id = str(uuid4())
    content = (
        json.dumps(
            {
                "format_version": _MARKER_VERSION,
                "session_id": session_id,
                "operation_id": operation_id,
                "state": "deleting",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    HOST_FILESYSTEM.atomic_create_text(marker, content)
    HOST_FILESYSTEM.require_owned_regular_file(marker, within=marker_root)
    HOST_FILESYSTEM.restrict_private_file(marker)
    return operation_id


def delete_session_data(workspace_state: WorkspaceState, session_id: str) -> None:
    """Delete all validated foreground Session-owned data and its marker."""
    _require_foreground_id(session_id)
    if session_restore_pending(workspace_state, session_id):
        raise RuntimeError("Session Restore must finish before deletion")
    marker_root = _existing_marker_root(workspace_state)
    if marker_root is None:
        raise FileNotFoundError("Session deletion marker is missing")
    marker = marker_root / f"{session_id}.json"
    _read_marker(marker, marker_root, session_id)

    files, trees = _deletion_targets(workspace_state, session_id)
    for tree, _parent in trees:
        _remove_owned_tree(tree, within=workspace_state.workspace_path)
    for path, _parent in files:
        _remove_owned_file(path, within=workspace_state.workspace_path)
    _remove_owned_file(marker, within=workspace_state.workspace_path)


def recover_session_deletions(workspace_state: WorkspaceState) -> tuple[str, ...]:
    """Retry every pending deletion; retain markers for failures or unsafe paths."""
    marker_root = _existing_marker_root(workspace_state)
    if marker_root is None:
        return ()
    pending = _pending_session_ids(marker_root)
    unresolved: list[str] = []
    for session_id in pending:
        try:
            delete_session_data(workspace_state, session_id)
        except Exception:
            unresolved.append(session_id)
    return tuple(unresolved)


def session_restore_pending(workspace_state: WorkspaceState, session_id: str) -> bool:
    """Keep even unreadable Restore recovery state until its owner completes it."""
    from aide.agent.session.restore import RestoreError, RestoreManager

    _require_foreground_id(session_id)
    try:
        return RestoreManager(workspace_state, session_id).has_pending_transaction()
    except RestoreError:
        return True


def session_deletion_status(workspace_state: WorkspaceState, session_id: str) -> str:
    """Resolve a lost deletion result from its fence and all owned data locations."""
    if session_deletion_pending(workspace_state, session_id):
        return "deleting"
    files, trees = _deletion_targets(workspace_state, session_id)
    return "present" if files or trees else "deleted"


def _require_foreground_id(session_id: str) -> None:
    Session._require_id(session_id, partition=SessionStoragePartition.FOREGROUND)


def _state_root(workspace_state: WorkspaceState) -> Path:
    workspace_root = HOST_FILESYSTEM.require_owned_directory(
        workspace_state.workspace_path,
        within=workspace_state.workspace_path,
    )
    return HOST_FILESYSTEM.require_owned_directory(workspace_state.path, within=workspace_root)


def _existing_marker_root(workspace_state: WorkspaceState) -> Path | None:
    if not HOST_FILESYSTEM.entry_exists(workspace_state.path):
        return None
    state_root = _state_root(workspace_state)
    marker_root = workspace_state.session_deletions_directory
    if not HOST_FILESYSTEM.entry_exists(marker_root):
        return None
    return HOST_FILESYSTEM.require_owned_directory(marker_root, within=state_root)


def _prepare_marker_root(workspace_state: WorkspaceState) -> Path:
    state_root = _state_root(workspace_state)
    marker_root = workspace_state.session_deletions_directory
    if not HOST_FILESYSTEM.entry_exists(marker_root):
        HOST_FILESYSTEM.path_for_io(marker_root).mkdir(mode=0o700)
    owned = HOST_FILESYSTEM.require_owned_directory(marker_root, within=state_root)
    HOST_FILESYSTEM.restrict_private_directory(owned)
    return owned


def _read_marker(path: Path, marker_root: Path, expected_session_id: str) -> str:
    owned = HOST_FILESYSTEM.require_owned_regular_file(path, within=marker_root)
    try:
        value = json.loads(HOST_FILESYSTEM.path_for_io(owned).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Session deletion marker is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"format_version", "session_id", "operation_id", "state"}
        or value.get("format_version") != _MARKER_VERSION
        or value.get("session_id") != expected_session_id
        or not isinstance(value.get("operation_id"), str)
        or not value["operation_id"]
        or value.get("state") != "deleting"
    ):
        raise ValueError("Session deletion marker is invalid")
    return cast(str, value["operation_id"])


def _pending_session_ids(marker_root: Path) -> tuple[str, ...]:
    result: list[str] = []
    for candidate in HOST_FILESYSTEM.path_for_io(marker_root).iterdir():
        if not candidate.name.endswith(".json"):
            raise ValueError("Session deletion directory contains an unexpected entry")
        session_id = candidate.name[:-5]
        _require_foreground_id(session_id)
        _read_marker(candidate, marker_root, session_id)
        result.append(session_id)
    return tuple(sorted(result))


def _deletion_targets(
    workspace_state: WorkspaceState,
    session_id: str,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    state_root = _state_root(workspace_state)
    files: list[tuple[Path, Path]] = []
    trees: list[tuple[Path, Path]] = []

    sessions = workspace_state.existing_sessions_directory()
    if sessions is not None:
        session_path = sessions / f"{session_id}.jsonl"
        if HOST_FILESYSTEM.entry_exists(session_path):
            HOST_FILESYSTEM.require_owned_regular_file(session_path, within=sessions)
            files.append((session_path, sessions))

    logs = _existing_owned_directory(workspace_state.logs_directory, state_root)
    if logs is not None:
        for candidate in HOST_FILESYSTEM.path_for_io(logs).iterdir():
            if _is_session_log(candidate.name, session_id):
                HOST_FILESYSTEM.require_owned_regular_file(candidate, within=logs)
                files.append((candidate, logs))

    artifacts = _existing_owned_directory(workspace_state.path / "artifacts", state_root)
    if artifacts is not None:
        artifact_root = artifacts / session_id
        if HOST_FILESYSTEM.entry_exists(artifact_root):
            _validate_owned_tree(artifact_root, within=artifacts)
            trees.append((artifact_root, artifacts))

    subagents = workspace_state.existing_subagents_directory()
    if subagents is not None:
        subagent_root = subagents / session_id
        if HOST_FILESYSTEM.entry_exists(subagent_root):
            _validate_owned_tree(subagent_root, within=subagents)
            trees.append((subagent_root, subagents))

    restore = _existing_owned_directory(workspace_state.path / "restore", state_root)
    if restore is not None:
        restore_root = restore / session_id
        if HOST_FILESYSTEM.entry_exists(restore_root):
            _validate_owned_tree(restore_root, within=restore)
            trees.append((restore_root, restore))

    return files, trees


def _existing_owned_directory(path: Path, state_root: Path) -> Path | None:
    if not HOST_FILESYSTEM.entry_exists(path):
        return None
    return HOST_FILESYSTEM.require_owned_directory(path, within=state_root)


def _validate_owned_tree(path: Path, *, within: Path) -> None:
    owned = HOST_FILESYSTEM.require_owned_directory(path, within=within)
    for child in HOST_FILESYSTEM.path_for_io(owned).iterdir():
        try:
            status = child.lstat()
        except FileNotFoundError:
            continue
        if HOST_FILESYSTEM.is_directory(status):
            _validate_owned_tree(child, within=owned)
        else:
            HOST_FILESYSTEM.require_owned_regular_file(child, within=owned)


def _remove_owned_tree(path: Path, *, within: Path) -> None:
    if not HOST_FILESYSTEM.entry_exists(path):
        return
    HOST_FILESYSTEM.remove_owned_entry(path, root=within, tree=True)


def _remove_owned_file(path: Path, *, within: Path) -> None:
    if not HOST_FILESYSTEM.entry_exists(path):
        return
    HOST_FILESYSTEM.remove_owned_entry(path, root=within, tree=False)


def _is_session_log(name: str, session_id: str) -> bool:
    prefix = f"{session_id}."
    return name == f"{session_id}.log" or (
        name.startswith(prefix) and _LOG_SUFFIX.search(name) is not None
    )


__all__ = [
    "SessionDeletionPending",
    "begin_session_deletion",
    "delete_session_data",
    "recover_session_deletions",
    "session_deletion_pending",
]
