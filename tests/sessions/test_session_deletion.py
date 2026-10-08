"""Durable foreground Session deletion and recovery tests."""

from __future__ import annotations

import ctypes
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import aide.agent.session.deletion as deletion
import aide.utils._owned_deletion as native_deletion
from aide.agent.session.deletion import (
    SessionDeletionPending,
    begin_session_deletion,
    delete_session_data,
    recover_session_deletions,
    session_deletion_pending,
)
from aide.agent.session.session import Session
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.store import SubAgentRecordStore
from aide.agent.workspace_state import WorkspaceState


async def _persist_session(tmp_path: Path) -> tuple[WorkspaceState, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: datetime(2026, 4, 1, tzinfo=UTC))
    session.commit_agent_run(
        [{"role": "user", "content": "delete this"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    return state, session.session_id


@pytest.mark.asyncio
async def test_deletion_marker_fences_load_and_recovery_removes_owned_data(
    tmp_path: Path,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    artifact_root = state.path / "artifacts" / session_id
    artifact_root.mkdir(parents=True)
    (artifact_root / "result.txt").write_text("artifact", encoding="utf-8")
    state.logs_directory.mkdir()
    (state.logs_directory / f"{session_id}.log").write_text("log", encoding="utf-8")
    restore_root = state.path / "restore" / session_id
    restore_root.mkdir(parents=True)
    (restore_root / "backup.bin").write_text("backup", encoding="utf-8")

    operation_id = begin_session_deletion(state, session_id)
    assert operation_id
    assert session_deletion_pending(state, session_id)
    with pytest.raises(SessionDeletionPending):
        Session.load(state, session_id)
    with pytest.raises(SessionDeletionPending):
        Session.load_header(state, session_id)

    assert recover_session_deletions(state) == ()
    assert not state.sessions_directory.joinpath(f"{session_id}.jsonl").exists()
    assert not artifact_root.exists()
    assert not restore_root.exists()
    assert not (state.logs_directory / f"{session_id}.log").exists()
    assert not session_deletion_pending(state, session_id)


@pytest.mark.asyncio
async def test_session_deletion_removes_subagent_records_and_artifacts(
    tmp_path: Path,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    store = SubAgentRecordStore(state, session_id)
    record = store.register(
        title="Task to delete",
        task="Belongs to the deleted Session.",
        parent_run_id="123e4567-e89b-42d3-a456-426614174000",
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token="123e4567-e89b-42d3-a456-426614174001",
        ),
        creator_snapshot=SubAgentCreatorSnapshot(
            provider_id="test-provider",
            model="chat",
            reasoning_effort="mid",
            permission_level="workspace-write",
            shell="pwsh",
            tool_names=("read_file",),
            system_prompt="You are Aide.",
        ),
    )
    artifact = state.path / "artifacts" / session_id / f"{record.agent_id}_tool_call.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("full result", encoding="utf-8")
    running = replace(
        record,
        status=SubAgentStatus.RUNNING,
        started_at=datetime(2026, 4, 1, tzinfo=UTC),
        revision=record.revision + 1,
    )
    store.save(running)
    store.save(
        replace(
            running,
            status=SubAgentStatus.COMPLETED,
            finished_at=datetime(2026, 4, 1, tzinfo=UTC),
            artifact_paths=(f".aide/artifacts/{session_id}/{artifact.name}",),
            result="done",
            revision=running.revision + 1,
        )
    )
    subagent_root = state.subagents_directory / session_id
    assert subagent_root.exists()

    begin_session_deletion(state, session_id)
    delete_session_data(state, session_id)

    assert not subagent_root.exists()
    assert not artifact.exists()


def _directory_alias(alias: Path, target: Path) -> None:
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["artifacts", "nested_artifacts", "restore", "logs", "sessions", "marker"]
)
async def test_deletion_rejects_directory_aliases_and_preserves_owned_history(
    tmp_path: Path,
    boundary: str,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    begin_session_deletion(state, session_id)
    outside = tmp_path / "user-data"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    if boundary == "marker":
        alias = state.session_deletions_directory
    elif boundary == "sessions":
        alias = state.sessions_directory
    elif boundary == "logs":
        alias = state.logs_directory
    else:
        alias = state.path / boundary.removeprefix("nested_") / session_id
        if boundary == "nested_artifacts":
            alias /= "nested"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists():
        alias.rename(alias.with_name(alias.name + "-retained"))
    _directory_alias(alias, outside)
    try:
        with pytest.raises(OSError):
            delete_session_data(state, session_id)
        assert protected.read_text(encoding="utf-8") == "keep"
        if boundary not in {"sessions", "marker"}:
            assert (state.sessions_directory / f"{session_id}.jsonl").exists()
            assert session_deletion_pending(state, session_id)
    finally:
        alias.rmdir()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["history", "artifact", "restore", "log", "marker"])
async def test_deletion_rejects_hardlinked_owned_entries(tmp_path: Path, boundary: str) -> None:
    state, session_id = await _persist_session(tmp_path)
    begin_session_deletion(state, session_id)
    if boundary == "history":
        path = state.sessions_directory / f"{session_id}.jsonl"
    elif boundary == "marker":
        path = state.session_deletions_directory / f"{session_id}.json"
    elif boundary == "log":
        path = state.logs_directory / f"{session_id}.log"
    else:
        path = (
            state.path
            / ("artifacts" if boundary == "artifact" else "restore")
            / session_id
            / "data"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("owned", encoding="utf-8")
    outside = tmp_path / "user-file"
    outside.hardlink_to(path)
    content = outside.read_bytes()
    with pytest.raises(OSError):
        delete_session_data(state, session_id)
    assert outside.read_bytes() == content
    assert path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["invalid_json", "wrong_id", "extra_path", "invalid_state"])
async def test_corrupt_marker_never_deletes_data(tmp_path: Path, corruption: str) -> None:
    state, session_id = await _persist_session(tmp_path)
    begin_session_deletion(state, session_id)
    marker = state.session_deletions_directory / f"{session_id}.json"
    value = json.loads(marker.read_text(encoding="utf-8"))
    if corruption == "wrong_id":
        value["session_id"] = "../user-data"
    elif corruption == "extra_path":
        value["path"] = str(tmp_path)
    elif corruption == "invalid_state":
        value["state"] = "done"
    marker.write_text("{" if corruption == "invalid_json" else json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        recover_session_deletions(state)
    assert (state.sessions_directory / f"{session_id}.jsonl").exists()
    assert marker.exists()


@pytest.mark.asyncio
async def test_deletion_pins_ancestors_during_native_file_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    artifact = state.path / "artifacts" / session_id
    artifact.mkdir(parents=True)
    (artifact / "protected.txt").write_text("owned", encoding="utf-8")
    outside = tmp_path / "user-data"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    begin_session_deletion(state, session_id)
    original_remove = native_deletion._windows_remove_entry
    attempts = 0

    def replace_ancestor(kernel: Any, path: Path, *, tree: bool) -> None:
        nonlocal attempts
        if path.name == "protected.txt":
            attempts += 1
            with pytest.raises(OSError):
                artifact.rename(artifact.with_name("moved"))
            write_handle = kernel.CreateFileW(
                str(path.parent),
                0x40000000,
                0x1 | 0x2 | 0x4,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if write_handle != ctypes.c_void_p(-1).value:
                kernel.CloseHandle(write_handle)
                pytest.fail("Held deletion directory accepted a concurrent mutation handle")
            assert ctypes.get_last_error() == 32
        original_remove(kernel, path, tree=tree)

    monkeypatch.setattr(native_deletion, "_windows_remove_entry", replace_ancestor)
    delete_session_data(state, session_id)
    assert attempts == 1
    assert protected.read_text(encoding="utf-8") == "keep"
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_replacement_before_native_traversal_keeps_external_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    artifact = state.path / "artifacts" / session_id
    artifact.mkdir(parents=True)
    (artifact / "protected.txt").write_text("owned", encoding="utf-8")
    outside = tmp_path / "user-data"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    begin_session_deletion(state, session_id)
    original_remove = deletion._remove_owned_tree

    def replace_tree(path: Path, *, within: Path) -> None:
        path.rename(path.with_name(path.name + "-retained"))
        _directory_alias(path, outside)
        original_remove(path, within=within)

    monkeypatch.setattr(deletion, "_remove_owned_tree", replace_tree)
    try:
        with pytest.raises(OSError):
            delete_session_data(state, session_id)
        assert protected.read_text(encoding="utf-8") == "keep"
        assert session_deletion_pending(state, session_id)
        assert (state.sessions_directory / f"{session_id}.jsonl").exists()
    finally:
        artifact.rmdir()


@pytest.mark.asyncio
async def test_deletion_failure_keeps_marker_and_retry_finishes_remaining_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, session_id = await _persist_session(tmp_path)
    artifact_root = state.path / "artifacts" / session_id
    artifact_root.mkdir(parents=True)
    (artifact_root / "result.txt").write_text("artifact", encoding="utf-8")
    begin_session_deletion(state, session_id)

    original_remove_tree = deletion._remove_owned_tree
    calls = 0

    def fail_once(path: Path, *, within: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected deletion failure")
        original_remove_tree(path, within=within)

    monkeypatch.setattr(deletion, "_remove_owned_tree", fail_once)
    with pytest.raises(OSError):
        delete_session_data(state, session_id)
    assert session_deletion_pending(state, session_id)
    assert state.sessions_directory.joinpath(f"{session_id}.jsonl").exists()

    monkeypatch.setattr(deletion, "_remove_owned_tree", original_remove_tree)
    assert recover_session_deletions(state) == ()
    assert not state.sessions_directory.joinpath(f"{session_id}.jsonl").exists()
    assert not session_deletion_pending(state, session_id)


@pytest.mark.asyncio
async def test_deletion_rejects_artifact_symlink_without_touching_outside_data(
    tmp_path: Path,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    artifacts = state.path / "artifacts"
    artifacts.mkdir()
    try:
        (artifacts / session_id).symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"host cannot create a directory symlink for the security test: {error}")

    begin_session_deletion(state, session_id)
    with pytest.raises(OSError):
        delete_session_data(state, session_id)
    assert protected.read_text(encoding="utf-8") == "keep"
    assert state.sessions_directory.joinpath(f"{session_id}.jsonl").exists()
    assert session_deletion_pending(state, session_id)


@pytest.mark.asyncio
async def test_partial_tree_cleanup_stays_fenced_until_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, session_id = await _persist_session(tmp_path)
    artifact_root = state.path / "artifacts" / session_id
    artifact_root.mkdir(parents=True)
    (artifact_root / "result.txt").write_text("artifact", encoding="utf-8")
    restore_root = state.path / "restore" / session_id
    restore_root.mkdir(parents=True)
    (restore_root / "backup").write_text("backup", encoding="utf-8")
    begin_session_deletion(state, session_id)
    original = deletion._remove_owned_tree
    calls = 0

    def fail_after_artifact(path: Path, *, within: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("power lost after deleting artifact tree")
        original(path, within=within)

    monkeypatch.setattr(deletion, "_remove_owned_tree", fail_after_artifact)
    with pytest.raises(OSError):
        delete_session_data(state, session_id)
    assert not artifact_root.exists()
    assert restore_root.exists()
    with pytest.raises(SessionDeletionPending):
        Session.load(state, session_id)
    monkeypatch.setattr(deletion, "_remove_owned_tree", original)
    assert recover_session_deletions(state) == ()
    assert not restore_root.exists()
    assert not (state.sessions_directory / f"{session_id}.jsonl").exists()
    assert not session_deletion_pending(state, session_id)
