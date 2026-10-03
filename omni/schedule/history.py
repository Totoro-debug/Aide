"""Read-only projections of persisted Schedule Session history."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.workspace_state import WorkspaceState
from omni.utils.host_filesystem import HOST_FILESYSTEM

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_TERMINAL_STATES = {
    "completed": "success",
    "error": "failure",
    "interrupted": "canceled",
}


class ScheduleHistoryRequestError(ValueError):
    """Raised when a history page request or cursor is invalid."""


class ScheduleHistoryPersistenceError(RuntimeError):
    """Raised when persisted Schedule history cannot be read safely."""


@dataclass(frozen=True, slots=True)
class _HistoryGroup:
    start_index: int
    end_index: int
    result_state: str
    messages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, object]:
        first = self.messages[0] if self.messages else {}
        terminal = self.messages[-1] if self.messages else {}
        started_at = first.get("timestamp") if isinstance(first.get("timestamp"), str) else None
        finished_at = (
            terminal.get("timestamp") if isinstance(terminal.get("timestamp"), str) else None
        )
        return {
            "started_at": started_at,
            "finished_at": finished_at if self.result_state != "unknown" else None,
            "result_state": self.result_state,
            "complete": self.result_state != "unknown",
            "messages": copy.deepcopy(list(self.messages)),
        }


def read_schedule_history(
    workspace_state: WorkspaceState,
    job_id: str,
    *,
    workspace_id: str = "",
    cursor: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Return grouped Schedule history without creating a per-run identity."""
    session_id = Session.schedule_session_id(job_id)
    page_limit = _validate_limit(limit)
    messages = _load_messages(workspace_state, session_id)
    revision = hashlib.sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    after_index = (
        _decode_cursor(cursor, workspace_id, job_id, revision) if cursor is not None else 0
    )
    groups = _group_messages(messages)
    if after_index and not any(group.start_index == after_index for group in groups):
        raise ScheduleHistoryRequestError("cursor is invalid.")
    visible = [group for group in groups if group.start_index >= after_index]
    page = visible[:page_limit]
    next_cursor = None
    if len(visible) > len(page) and page:
        next_cursor = _encode_cursor(
            workspace_id,
            job_id,
            page[-1].end_index + 1,
            revision,
        )
    return {
        "session_id": session_id,
        "groups": [group.to_dict() for group in page],
        "next_cursor": next_cursor,
    }


def _validate_limit(limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
        raise ScheduleHistoryRequestError(f"limit must be between 1 and {_MAX_LIMIT}.")
    return limit


def _encode_cursor(workspace_id: str, job_id: str, after_index: int, revision: str) -> str:
    payload = json.dumps(
        {
            "workspace_id": workspace_id,
            "job_id": job_id,
            "after_index": after_index,
            "revision": revision,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str, workspace_id: str, job_id: str, revision: str) -> int:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        )
        if payload["workspace_id"] != workspace_id or payload["job_id"] != job_id:
            raise ValueError("cursor scope does not match")
        if payload["revision"] != revision:
            raise ValueError("history changed; refresh before continuing")
        after_index = payload["after_index"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ScheduleHistoryRequestError("cursor is invalid.") from error
    if isinstance(after_index, bool) or not isinstance(after_index, int) or after_index < 0:
        raise ScheduleHistoryRequestError("cursor is invalid.")
    return cast(int, after_index)


def _load_messages(workspace_state: WorkspaceState, session_id: str) -> list[dict[str, Any]]:
    try:
        session = Session.load(
            workspace_state,
            session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
    except FileNotFoundError:
        return []
    except (TypeError, UnicodeError, ValueError):
        try:
            return _load_legacy_messages(workspace_state, session_id)
        except FileNotFoundError:
            return []
        except (OSError, TypeError, UnicodeError, ValueError) as error:
            raise ScheduleHistoryPersistenceError(
                "Schedule history could not be loaded safely."
            ) from error
    return copy.deepcopy(session.messages)


def _load_legacy_messages(workspace_state: WorkspaceState, session_id: str) -> list[dict[str, Any]]:
    directory = workspace_state.existing_schedule_sessions_directory()
    if directory is None:
        return []
    path = directory / f"{session_id}.jsonl"
    owned_path = HOST_FILESYSTEM.require_owned_regular_file(path, within=directory)
    content = owned_path.read_bytes()
    if not content.endswith(b"\n"):
        raise ValueError("Schedule Session JSONL must end with a newline")
    records: list[dict[str, Any]] = []
    for line in content.splitlines(keepends=False):
        if not line:
            raise ValueError("Schedule Session JSONL contains an incomplete record")
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Schedule Session JSONL contains invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError("Schedule Session JSONL records must be objects")
        records.append(cast(dict[str, Any], record))
    if not records or records[0].get("session_id") != session_id:
        raise ValueError("Schedule Session metadata ID does not match its file name")
    records = records[1:]
    return [_legacy_message(record) for record in records]


def _legacy_message(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("role") not in ("user", "assistant", "tool") or not isinstance(
        record.get("content"), str
    ):
        raise ValueError("Schedule history contains a malformed message")
    allowed = {
        "role",
        "content",
        "timestamp",
        "status",
        "error",
        "tool_calls",
        "tool_call_id",
        "name",
        "artifact",
        "context_usage",
        "token_usage",
    }
    return {key: copy.deepcopy(value) for key, value in record.items() if key in allowed}


def _group_messages(messages: list[dict[str, Any]]) -> list[_HistoryGroup]:
    groups: list[_HistoryGroup] = []
    current: list[dict[str, Any]] = []
    current_start = 0
    has_user = False
    pending_tools: set[str] = set()
    complete_sequence = True

    def finish(end_index: int, result_state: str) -> None:
        nonlocal current, current_start, has_user, complete_sequence
        if current:
            groups.append(
                _HistoryGroup(
                    start_index=current_start,
                    end_index=end_index,
                    result_state=result_state
                    if complete_sequence and not pending_tools
                    else "unknown",
                    messages=tuple(copy.deepcopy(current)),
                )
            )
        current = []
        has_user = False
        pending_tools.clear()
        complete_sequence = True

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user" and current:
            finish(index - 1, "unknown")
        if not current:
            current_start = index
        current.append(message)
        has_user = has_user or role == "user"
        if role == "tool":
            if message.get("status") not in ("success", "error", "refused"):
                complete_sequence = False
            tool_id = message.get("tool_call_id")
            if not isinstance(tool_id, str) or tool_id not in pending_tools:
                complete_sequence = False
            else:
                pending_tools.remove(tool_id)
        if role == "assistant":
            calls = message.get("tool_calls")
            status = message.get("status")
            error = message.get("error")
            if (
                not isinstance(calls, list)
                or status not in ("completed", "error", "interrupted")
                or (status == "completed" and error is not None)
                or (
                    status in ("error", "interrupted")
                    and (
                        not isinstance(error, dict)
                        or not isinstance(error.get("code"), str)
                        or not isinstance(error.get("message"), str)
                    )
                )
            ):
                complete_sequence = False
            if calls:
                if pending_tools or not isinstance(calls, list):
                    complete_sequence = False
                for call in calls if isinstance(calls, list) else []:
                    tool_id = call.get("id") if isinstance(call, dict) else None
                    if not isinstance(tool_id, str) or not tool_id or tool_id in pending_tools:
                        complete_sequence = False
                    else:
                        pending_tools.add(tool_id)
            result_state = _TERMINAL_STATES.get(status) if isinstance(status, str) else None
            if has_user and result_state is not None and isinstance(calls, list) and not calls:
                finish(index, result_state)
    if current:
        finish(len(messages) - 1, "unknown")
    return groups
