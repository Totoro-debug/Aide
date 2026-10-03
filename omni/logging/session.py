"""Workspace-owned technical diagnostics for one Conversation Session."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from omni.agent.session.session import Session
from omni.agent.workspace_state import WorkspaceState
from omni.utils.host_filesystem import HOST_FILESYSTEM

_ROTATION_BYTES = 10_485_760
_failure_reported = False
__all__ = ["session_log", "without_session_log"]


@contextmanager
def without_session_log() -> Iterator[None]:
    """Run diagnostics without Conversation Session ownership."""
    with logger.contextualize(session_id=None):
        yield


@contextmanager
def session_log(
    workspace_or_session: WorkspaceState | Session,
    session_id: str | None = None,
) -> Iterator[None]:
    """Route WARNING+ Loguru records to one Session-owned file for this scope."""
    if isinstance(workspace_or_session, Session):
        if session_id is not None:
            raise TypeError("Session Log cannot receive both a Session and a Session ID")
        workspace_state = workspace_or_session.workspace_state
        resolved_session_id = workspace_or_session.session_id
    else:
        if session_id is None:
            raise TypeError("Session Log requires a Session ID")
        workspace_state = workspace_or_session
        resolved_session_id = session_id
    Session._require_id(resolved_session_id)
    handler_id = _add_sink(workspace_state, resolved_session_id)
    try:
        with logger.contextualize(session_id=resolved_session_id):
            yield
    finally:
        if handler_id is not None:
            logger.remove(handler_id)


def _add_sink(workspace_state: WorkspaceState, session_id: str) -> int | None:
    global _failure_reported
    try:
        logs = workspace_state.logs_directory
        _prepare_logs_directory(logs, workspace_state.path)
        path = logs / f"{session_id}.log"
        _validate_existing_log(path, logs)
        handler_id = logger.add(
            path,
            level="WARNING",
            filter=lambda record: record["extra"].get("session_id") == session_id,
            enqueue=True,
            rotation=_ROTATION_BYTES,
            retention=1,
            catch=True,
            delay=True,
            encoding="utf-8",
            diagnose=False,
            backtrace=False,
            colorize=False,
            opener=_safe_opener(logs),
        )
        _failure_reported = False
        return handler_id
    except Exception as error:
        if not _failure_reported:
            _failure_reported = True
            print(f"Session Log failure: {type(error).__name__}", file=sys.stderr)
        return None


def _prepare_logs_directory(logs: Path, state_root: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    HOST_FILESYSTEM.require_owned_directory(logs, within=state_root)
    HOST_FILESYSTEM.restrict_private_directory(logs)


def _validate_existing_log(path: Path, logs: Path) -> None:
    if path.exists() or path.is_symlink():
        HOST_FILESYSTEM.require_owned_regular_file(path, within=logs)


def _safe_opener(logs: Path) -> Callable[[str, int], int]:
    def opener(path: str, flags: int) -> int:
        candidate = Path(path)
        if candidate.parent != logs:
            raise PermissionError("Session Log path escaped logs directory")
        descriptor = os.open(path, flags, 0o600)
        try:
            HOST_FILESYSTEM.require_opened_owned_regular_file(descriptor, candidate, within=logs)
            HOST_FILESYSTEM.restrict_private_descriptor(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    return opener
