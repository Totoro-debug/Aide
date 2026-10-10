"""Schedule history projections derive groups from persisted message facts."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.workspace_state import WorkspaceState
from aide.schedule.history import (
    ScheduleHistoryPersistenceError,
    ScheduleHistoryRequestError,
    read_schedule_history,
)

_FIRST_JOB = "123e4567-e89b-42d3-a456-426614174000"
_SECOND_JOB = "223e4567-e89b-42d3-a456-426614174000"
_BASE_TIME = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
_USAGE = {"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def _workspace(tmp_path: Path) -> WorkspaceState:
    agent_home = tmp_path / "agent-home"
    workspace = tmp_path / "workspace"
    agent_home.mkdir()
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _assistant(
    content: str,
    *,
    status: str,
    error: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
        "status": status,
        "error": error,
        "token_usage": _USAGE,
    }


def _commit(
    state: WorkspaceState,
    job_id: str,
    now: datetime,
    messages: list[dict[str, object]],
) -> None:
    session_path = state.schedule_sessions_directory / f"schedule_{job_id}.jsonl"
    if session_path.exists():
        session = Session.load(
            state,
            f"schedule_{job_id}",
            partition=SessionStoragePartition.SCHEDULE,
            now=lambda: now,
        )
    else:
        session = Session.create_schedule(
            state,
            job_id,
            now=lambda: now,
            title="History fixture",
        )
    session.commit_agent_run(
        messages,
        pending_last_compacted=0,
        pending_action_summary=None,
    )
    session.close()


def test_history_reports_only_structurally_proven_terminal_states_and_unknown_tail(
    tmp_path: Path,
) -> None:
    state = _workspace(tmp_path)
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME,
        [
            {"role": "user", "content": "success input"},
            _assistant("success output", status="completed"),
        ],
    )
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME + timedelta(minutes=1),
        [
            {"role": "user", "content": "failure input"},
            _assistant(
                "",
                status="error",
                error={"code": "model_failed", "message": "safe failure"},
            ),
        ],
    )
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME + timedelta(minutes=2),
        [
            {"role": "user", "content": "cancel input"},
            _assistant(
                "Turn cancelled.",
                status="interrupted",
                error={"code": "turn_cancelled", "message": "Turn cancelled."},
            ),
        ],
    )
    path = state.schedule_sessions_directory / f"schedule_{_FIRST_JOB}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records.append(
        {
            "role": "user",
            "content": "incomplete input",
            "timestamp": (_BASE_TIME + timedelta(minutes=3)).isoformat(timespec="milliseconds"),
        }
    )
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )

    result = read_schedule_history(state, _FIRST_JOB, workspace_id="workspace")

    groups = result["groups"]
    assert isinstance(groups, list)
    assert [group["result_state"] for group in groups] == [
        "success",
        "failure",
        "canceled",
        "unknown",
    ]
    assert groups[-1]["complete"] is False
    assert groups[-1]["finished_at"] is None
    assert groups[-1]["messages"][0]["content"] == "incomplete input"


def test_history_keeps_tool_messages_and_long_markdown_in_one_group(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    long_markdown = "# Result\n\n```python\n" + ("print('x')\n" * 200) + "```"
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME,
        [
            {"role": "user", "content": "tool input"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
                "status": "completed",
                "error": None,
                "token_usage": _USAGE,
            },
            {
                "role": "tool",
                "content": "tool output",
                "tool_call_id": "call-1",
                "name": "read_file",
                "status": "success",
            },
            _assistant(long_markdown, status="completed"),
        ],
    )

    result = read_schedule_history(state, _FIRST_JOB, workspace_id="workspace")

    groups = result["groups"]
    assert isinstance(groups, list)
    assert len(groups) == 1
    messages = groups[0]["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[-1]["content"] == long_markdown


def test_history_rejects_incomplete_records_without_reparsing_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _workspace(tmp_path)
    session_id = Session.schedule_session_id(UUID(_SECOND_JOB))
    directory = state.prepare_schedule_sessions_directory()
    header = {
        "session_id": session_id,
        "created_at": _BASE_TIME.isoformat(timespec="milliseconds"),
        "updated_at": _BASE_TIME.isoformat(timespec="milliseconds"),
        "last_compacted": 0,
        "metadata": {
            "title": "Incomplete history",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
        },
    }
    incomplete_assistant: dict[str, object] = {
        "role": "assistant",
        "content": "incomplete result",
        "tool_calls": [],
        "error": None,
        "token_usage": _USAGE,
        "timestamp": _BASE_TIME.isoformat(timespec="milliseconds"),
    }
    (directory / f"{session_id}.jsonl").write_text(
        "\n".join(
            json.dumps(record, separators=(",", ":"))
            for record in [
                header,
                {
                    "role": "user",
                    "content": "stored input",
                    "timestamp": _BASE_TIME.isoformat(timespec="milliseconds"),
                },
                incomplete_assistant,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    path = directory / f"{session_id}.jsonl"
    before = path.read_bytes()
    read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counted_read(target: Path) -> bytes:
        reads.append(target)
        return read_bytes(target)

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("History reads must not rewrite unsupported records")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", counted_read)
        patch.setattr(Path, "write_bytes", unexpected_write)
        patch.setattr(Path, "write_text", unexpected_write)
        with pytest.raises(ScheduleHistoryPersistenceError):
            read_schedule_history(state, _SECOND_JOB, workspace_id="workspace")

    assert len(reads) == 1
    assert reads[0].samefile(path)
    assert path.read_bytes() == before


def test_empty_history_has_no_groups(tmp_path: Path) -> None:
    state = _workspace(tmp_path)

    result = read_schedule_history(state, _FIRST_JOB, workspace_id="workspace")

    assert result["groups"] == []
    assert result["next_cursor"] is None


def test_history_cursor_is_bound_to_workspace_and_job_scope(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME,
        [
            {"role": "user", "content": "first"},
            _assistant("one", status="completed"),
        ],
    )
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME + timedelta(minutes=1),
        [
            {"role": "user", "content": "second"},
            _assistant("two", status="completed"),
        ],
    )
    first_page = read_schedule_history(
        state,
        _FIRST_JOB,
        workspace_id="workspace-a",
        limit=1,
    )
    cursor = first_page["next_cursor"]
    assert isinstance(cursor, str)

    with pytest.raises(ScheduleHistoryRequestError):
        read_schedule_history(
            state,
            _FIRST_JOB,
            workspace_id="workspace-b",
            cursor=cursor,
        )
    with pytest.raises(ScheduleHistoryRequestError):
        read_schedule_history(
            state,
            _SECOND_JOB,
            workspace_id="workspace-a",
            cursor=cursor,
        )


@pytest.mark.parametrize("change", ["append", "rewrite", "truncate", "finish_tail"])
def test_history_rejects_cursor_after_persisted_history_changes(
    tmp_path: Path, change: str
) -> None:
    state = _workspace(tmp_path)
    for index in range(3):
        _commit(
            state,
            _FIRST_JOB,
            _BASE_TIME + timedelta(minutes=index),
            [
                {"role": "user", "content": f"input {index}"},
                _assistant(f"output {index}", status="completed"),
            ],
        )
    path = state.schedule_sessions_directory / f"schedule_{_FIRST_JOB}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if change == "finish_tail":
        records = records[:-1]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    cursor = read_schedule_history(state, _FIRST_JOB, limit=1)["next_cursor"]
    assert isinstance(cursor, str)
    if change == "append":
        _commit(
            state,
            _FIRST_JOB,
            _BASE_TIME + timedelta(minutes=4),
            [
                {"role": "user", "content": "new input"},
                _assistant("new output", status="completed"),
            ],
        )
    else:
        if change == "rewrite":
            records[1]["content"] = "changed input"
        elif change == "truncate":
            records = records[:3]
        else:
            records.append(
                {**_assistant("finished", status="completed"), "timestamp": _BASE_TIME.isoformat()}
            )
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ScheduleHistoryRequestError):
        read_schedule_history(state, _FIRST_JOB, cursor=cursor)
    assert read_schedule_history(state, _FIRST_JOB)["groups"]


@pytest.mark.parametrize("header", ["other_job", "foreground", "missing"])
def test_history_rejects_mismatched_or_missing_persisted_identity(
    tmp_path: Path, header: str
) -> None:
    state = _workspace(tmp_path)
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME,
        [
            {"role": "user", "content": "private input"},
            _assistant("private output", status="completed"),
        ],
    )
    path = state.schedule_sessions_directory / f"schedule_{_FIRST_JOB}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if header == "missing":
        records = records[1:]
    else:
        records[0]["session_id"] = (
            Session.schedule_session_id(_SECOND_JOB) if header == "other_job" else _SECOND_JOB
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ScheduleHistoryPersistenceError):
        read_schedule_history(state, _FIRST_JOB)


@pytest.mark.parametrize("sequence", ["orphan", "missing_tool", "wrong_tool", "duplicate_tool"])
def test_history_marks_unproven_occurrences_unknown(tmp_path: Path, sequence: str) -> None:
    state = _workspace(tmp_path)
    messages: list[dict[str, object]] = []
    if sequence != "orphan":
        messages.extend(
            [
                {"role": "user", "content": "input"},
                {
                    **_assistant("calling", status="completed"),
                    "tool_calls": [{"id": "call", "name": "read_file", "arguments": "{}"}],
                },
            ]
        )
        if sequence in {"wrong_tool", "duplicate_tool"}:
            tool: dict[str, object] = {
                "role": "tool",
                "content": "success",
                "status": "success",
                "name": "read_file",
                "tool_call_id": "other" if sequence == "wrong_tool" else "call",
            }
            messages.append(tool)
            if sequence == "duplicate_tool":
                messages.append(tool)
    messages.append(_assistant("success", status="completed"))
    _commit(state, _FIRST_JOB, _BASE_TIME, messages)
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME + timedelta(minutes=1),
        [
            {"role": "user", "content": "next input"},
            _assistant("known next result", status="completed"),
        ],
    )
    groups = read_schedule_history(state, _FIRST_JOB)["groups"]
    assert isinstance(groups, list)
    assert [group["result_state"] for group in groups] == ["unknown", "success"]


def test_history_rejects_cursor_inside_an_occurrence(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    for index in range(2):
        _commit(
            state,
            _FIRST_JOB,
            _BASE_TIME,
            [
                {"role": "user", "content": f"input {index}"},
                _assistant("output", status="completed"),
            ],
        )
    cursor = read_schedule_history(state, _FIRST_JOB, limit=1)["next_cursor"]
    assert isinstance(cursor, str)
    payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    payload["after_index"] = 1
    invalid = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(ScheduleHistoryRequestError):
        read_schedule_history(state, _FIRST_JOB, cursor=invalid)


@pytest.mark.parametrize(
    "change", ["calls_type", "completed_error", "error_without_detail", "status_type"]
)
def test_history_rejects_malformed_terminal_fields(
    tmp_path: Path, change: str
) -> None:
    state = _workspace(tmp_path)
    _commit(
        state,
        _FIRST_JOB,
        _BASE_TIME,
        [
            {"role": "user", "content": "stored input"},
            _assistant("stored output", status="completed"),
        ],
    )
    path = state.schedule_sessions_directory / f"schedule_{_FIRST_JOB}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if change == "calls_type":
        records[-1]["tool_calls"] = ""
    elif change == "completed_error":
        records[-1]["error"] = {"code": "failure", "message": "failure"}
    elif change == "error_without_detail":
        records[-1]["status"] = "error"
    else:
        records[-1]["status"] = {"completed": True}
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ScheduleHistoryPersistenceError):
        read_schedule_history(state, _FIRST_JOB)
