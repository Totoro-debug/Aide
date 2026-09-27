from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent.permission import PermissionSnapshot
from myclaw.agent.tools.base import BaseTool
from myclaw.agent.tools.core.edit_file import EditFileTool
from myclaw.agent.tools.core.exec_host import resolve_exec_shell
from myclaw.agent.tools.core.write_file import WriteFileTool
from myclaw.agent.tools.permission import PermissionContext, ToolPermissionLevel, ToolRunOrigin
from myclaw.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ModelToolCall,
    ToolGateway,
)
from myclaw.session.backup_store import BackupTicket, FileBackupStore
from myclaw.workspace.state import WorkspaceState

SESSION_ID = "20260926-120000-123456_12345678-1234-4234-8234-123456789abc"


def _call(name: str, arguments: dict[str, object], *, call_id: str = "call-1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def _gateway(
    workspace: Path,
    *tools: BaseTool,
    origin: ToolRunOrigin = "foreground",
    level: ToolPermissionLevel = "full-access",
) -> ToolGateway:
    snapshot = PermissionSnapshot(level=level, exec_shell=resolve_exec_shell("auto"))
    context = PermissionContext.from_snapshot(
        snapshot,
        workspace_root=workspace,
        origin=origin,
    )
    return ToolGateway._for_memory(tuple(tools), permission_context=context)


@dataclass
class _RecordingRecorder:
    before_calls: list[tuple[UUID, Path]] = field(default_factory=list)
    after_calls: list[BackupTicket | None] = field(default_factory=list)

    def before_write(self, run_token: UUID, resolved_target: Path) -> BackupTicket:
        self.before_calls.append((run_token, resolved_target))
        return BackupTicket(
            operation_id=len(self.before_calls),
            run_token=run_token,
            requested_target=resolved_target,
            canonical_target=resolved_target,
            recorded=True,
        )

    def after_write(self, ticket: BackupTicket | None) -> None:
        self.after_calls.append(ticket)


class _NoopTool(BaseTool):
    name = "exec"
    description = "A non-file test Tool."

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self) -> str:
        self.calls += 1
        return "done"


class _FailingRecorder:
    def before_write(self, run_token: UUID, resolved_target: Path) -> BackupTicket:
        del run_token, resolved_target
        raise OSError("backup persistence failed")

    def after_write(self, ticket: BackupTicket | None) -> None:
        del ticket
        raise OSError("gap persistence failed")


@pytest.mark.asyncio
async def test_authorized_foreground_write_persists_one_backup_with_run_token(
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    target.write_bytes(b"before")
    token = uuid4()
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))

    result = await gateway.call(
        _call("write_file", {"path": "notes.txt", "content": "after"}),
        file_mutation_recorder=store,
        run_token=token,
    )

    assert (result.status, result.content) == ("success", "File written successfully.")
    journal = store.inspect()
    assert len(journal.entries) == 1
    assert journal.entries[0].run_token == token
    assert journal.entries[0].canonical_target == str(target.resolve())
    assert store.read_backup(1) == b"before"
    assert target.read_bytes() == b"after"


@pytest.mark.asyncio
async def test_matching_edit_records_once_but_unmatched_and_ambiguous_edits_record_zero(
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    target.write_text("needle\n", encoding="utf-8")
    recorder = _RecordingRecorder()
    gateway = _gateway(workspace, EditFileTool(workspace=workspace))

    matched = await gateway.call(
        _call(
            "edit_file",
            {"path": "notes.txt", "old_text": "needle", "new_text": "changed"},
        ),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )
    unmatched = await gateway.call(
        _call(
            "edit_file",
            {"path": "notes.txt", "old_text": "missing", "new_text": "changed"},
            call_id="call-2",
        ),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )
    target.write_text("duplicate duplicate", encoding="utf-8")
    ambiguous = await gateway.call(
        _call(
            "edit_file",
            {"path": "notes.txt", "old_text": "duplicate", "new_text": "changed"},
            call_id="call-3",
        ),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert matched.status == "success"
    assert unmatched.status == "error"
    assert ambiguous.status == "error"
    assert len(recorder.before_calls) == 1
    assert len(recorder.after_calls) == 1
    assert recorder.before_calls[0][1] == target.resolve()
    assert target.read_text(encoding="utf-8") == "duplicate duplicate"


@pytest.mark.asyncio
async def test_authorized_external_write_records_the_actual_external_target(
    workspace: Path,
) -> None:
    target = workspace.parent / "external.txt"
    target.write_bytes(b"before")
    recorder = _RecordingRecorder()
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))

    result = await gateway.call(
        _call("write_file", {"path": str(target), "content": "after"}),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert result.status == "success"
    assert [recorded_target for _token, recorded_target in recorder.before_calls] == [
        target.resolve()
    ]
    assert target.read_bytes() == b"after"


@pytest.mark.asyncio
async def test_declined_and_unavailable_file_confirmation_record_zero(
    workspace: Path,
) -> None:
    target = workspace.parent / "outside.txt"
    target.write_bytes(b"before")
    recorder = _RecordingRecorder()
    gateway = _gateway(
        workspace,
        WriteFileTool(workspace=workspace),
        level="read-only",
    )
    call = _call("write_file", {"path": str(target), "content": "after"})

    unavailable = await gateway.call(
        call,
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    async def decline(_request: object) -> ConfirmationDecision:
        return "declined"

    declined = await gateway.call(
        call,
        confirmation=decline,
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert unavailable.status == "refused"
    assert declined.status == "refused"
    assert recorder.before_calls == []
    assert target.read_bytes() == b"before"


@pytest.mark.asyncio
async def test_non_foreground_and_non_file_calls_do_not_record(
    workspace: Path,
) -> None:
    recorder = _RecordingRecorder()
    write = WriteFileTool(workspace=workspace)
    noop = _NoopTool()
    gateway = _gateway(workspace, write, noop, origin="memory")

    write_result = await gateway.call(
        _call("write_file", {"path": "memory.txt", "content": "written"}),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )
    exec_result = await gateway.call(
        _call("exec", {}),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert write_result.status == "success"
    assert exec_result.status == "success"
    assert recorder.before_calls == []
    assert noop.calls == 1


@pytest.mark.asyncio
async def test_protected_restore_state_is_rejected_before_authorization_or_recording(
    workspace: Path,
) -> None:
    recorder = _RecordingRecorder()
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))

    result = await gateway.call(
        _call(
            "write_file",
            {"path": ".myclaw/restore/session/state.json", "content": "unsafe"},
        ),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert result.status == "refused"
    assert "protected" in result.content.lower()
    assert recorder.before_calls == []
    assert not (workspace / ".myclaw" / "restore" / "session" / "state.json").exists()


@pytest.mark.asyncio
async def test_backup_recorder_failures_do_not_change_write_result(
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))

    result = await gateway.call(
        _call("write_file", {"path": "notes.txt", "content": "after"}),
        file_mutation_recorder=_FailingRecorder(),
        run_token=uuid4(),
    )

    assert (result.status, result.content) == ("success", "File written successfully.")
    assert target.read_text(encoding="utf-8") == "after"


@pytest.mark.asyncio
async def test_write_failure_preserves_tool_error_after_one_recording_attempt(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = workspace / "notes.txt"
    recorder = _RecordingRecorder()
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))
    original_write_bytes = Path.write_bytes

    def fail_target_write(path: Path, data: bytes) -> int:
        if path == target:
            raise OSError("injected write failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_target_write)
    result = await gateway.call(
        _call("write_file", {"path": "notes.txt", "content": "after"}),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )

    assert result.status == "error"
    assert "injected write failure" in result.content
    assert len(recorder.before_calls) == 1
    assert len(recorder.after_calls) == 1


@pytest.mark.asyncio
async def test_run_tokens_are_explicit_per_call_and_not_shared_tool_state(
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    target.write_bytes(b"before")
    recorder = _RecordingRecorder()
    gateway = _gateway(workspace, WriteFileTool(workspace=workspace))
    first_token = uuid4()
    second_token = uuid4()

    await gateway.call(
        _call("write_file", {"path": "notes.txt", "content": "first"}),
        file_mutation_recorder=recorder,
        run_token=first_token,
    )
    await gateway.call(
        _call("write_file", {"path": "notes.txt", "content": "second"}, call_id="call-2"),
        file_mutation_recorder=recorder,
        run_token=second_token,
    )

    assert [token for token, _target in recorder.before_calls] == [first_token, second_token]
