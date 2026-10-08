from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from aide.agent.session.session import Session
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentRecord,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.ports import SubAgentRecordRepository
from aide.agent.subagents.store import (
    SubAgentRecordStore,
    SubAgentRequestError,
    SubAgentStoreError,
)
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
_RUN_ID = "123e4567-e89b-42d3-a456-426614174000"
_RESTORE_TOKEN = "123e4567-e89b-42d3-a456-426614174001"


def _workspace(tmp_path: Path) -> tuple[WorkspaceState, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session_id = Session.create(state, now=lambda: _NOW).session_id
    return state, session_id


def _snapshot() -> SubAgentCreatorSnapshot:
    return SubAgentCreatorSnapshot(
        provider_id="test-provider",
        model="chat",
        reasoning_effort="mid",
        permission_level="workspace-write",
        shell="pwsh",
        tool_names=("read_file", "list_dir"),
        system_prompt="You are Aide.",
    )


def _register(store: SubAgentRecordStore, title: str) -> SubAgentRecord:
    return store.register(
        title=title,
        task=f"Task material for {title}.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )


def test_record_round_trips_through_a_new_session_scoped_store(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    repository: SubAgentRecordRepository = store
    queued = store.register(
        title="Inspect the repository",
        task="Find the relevant implementation.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=queued.revision + 1,
    )
    store.save(running)
    completed = replace(
        running,
        status=SubAgentStatus.COMPLETED,
        finished_at=_NOW,
        conversation=({"role": "assistant", "content": "Found it."},),
        context_state={"last_compacted": 0, "messages": [{"role": "system"}]},
        artifact_paths=("subagents/agent-1/tool-call.txt",),
        result="Found it.",
        usage={
            "model_calls": 1,
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
        },
        revision=running.revision + 1,
    )
    store.save(completed)

    reopened = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    loaded = reopened.get(completed.agent_id)

    assert loaded == completed
    assert loaded.creator_snapshot.provider_id == "test-provider"
    assert loaded.creator_snapshot.tool_names == ("read_file", "list_dir")
    path = state.subagents_directory / session_id / f"{completed.agent_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["creator_snapshot"] == {
        "provider_id": "test-provider",
        "model": "chat",
        "reasoning_effort": "mid",
        "permission_level": "workspace-write",
        "shell": "pwsh",
        "tool_names": ["read_file", "list_dir"],
        "system_prompt": "You are Aide.",
    }
    assert repository.session_id == session_id


def test_legacy_session_lists_no_records_without_creating_subagent_storage(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)

    page = store.list()

    assert page.items == ()
    assert page.next_cursor is None
    assert not state.subagents_directory.exists()


def test_session_scoped_lookup_cannot_read_another_sessions_record(tmp_path: Path) -> None:
    state, first_session_id = _workspace(tmp_path)
    second_session_id = Session.create(state, now=lambda: _NOW).session_id
    first_store = SubAgentRecordStore(state, first_session_id, now=lambda: _NOW)
    second_store = SubAgentRecordStore(state, second_session_id, now=lambda: _NOW)
    own = _register(first_store, "Owned task")
    other = second_store.register(
        title="Private task",
        task="Do not expose across Sessions.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )

    assert first_store.get(other.agent_id) is None
    assert first_store.get("123e4567-e89b-42d3-a456-426614174003") is None
    assert tuple(item.agent_id for item in first_store.list().items) == (own.agent_id,)
    with pytest.raises(SubAgentRequestError):
        first_store.get("../../" + other.agent_id)


def test_restore_discards_only_matching_subagents_and_their_artifacts(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    discarded = _register(store, "Discarded task")
    kept = store.register(
        title="Kept task",
        task="This task belongs to an earlier input.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token="223e4567-e89b-42d3-a456-426614174001",
        ),
        creator_snapshot=_snapshot(),
    )

    paths = {
        discarded.agent_id: f".aide/artifacts/{session_id}/{discarded.agent_id}_tool_call.txt",
        kept.agent_id: f".aide/artifacts/{session_id}/{kept.agent_id}_tool_call.txt",
    }
    for record in (discarded, kept):
        running = replace(
            record,
            status=SubAgentStatus.RUNNING,
            started_at=_NOW,
            revision=record.revision + 1,
        )
        store.save(running)
        store.save(
            replace(
                running,
                status=SubAgentStatus.COMPLETED,
                finished_at=_NOW,
                artifact_paths=(paths[record.agent_id],),
                result=f"done: {record.title}",
                revision=running.revision + 1,
            )
        )
        artifact = state.workspace_path.joinpath(*paths[record.agent_id].split("/"))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(record.title, encoding="utf-8")

    store.discard_restore_run_tokens((_RESTORE_TOKEN,))

    assert store.get(discarded.agent_id) is None
    assert not state.workspace_path.joinpath(*paths[discarded.agent_id].split("/")).exists()
    assert store.get(kept.agent_id) is not None
    assert (
        state.workspace_path.joinpath(*paths[kept.agent_id].split("/")).read_text(encoding="utf-8")
        == "Kept task"
    )


def test_list_pages_follow_registration_order_and_cursors_are_session_scoped(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    other_session_id = Session.create(state, now=lambda: _NOW).session_id
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    other_store = SubAgentRecordStore(state, other_session_id, now=lambda: _NOW)
    expected_ids = tuple(
        store.register(
            title=f"Task {index}",
            task=f"Work item {index}.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=_RESTORE_TOKEN,
            ),
            creator_snapshot=_snapshot(),
        ).agent_id
        for index in range(5)
    )

    first = store.list(limit=2)
    second = store.list(limit=2, cursor=first.next_cursor)
    third = store.list(limit=2, cursor=second.next_cursor)

    assert (
        tuple(item.agent_id for item in (*first.items, *second.items, *third.items)) == expected_ids
    )
    assert first.next_cursor is not None
    assert second.next_cursor is not None
    assert third.next_cursor is None
    assert not hasattr(first.items[0], "task")
    assert not hasattr(first.items[0], "conversation")
    with pytest.raises(SubAgentRequestError):
        other_store.list(cursor=first.next_cursor)


def test_failed_registration_does_not_return_an_id_or_leave_a_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)

    def fail_publish(source: str | bytes | Path, target: str | bytes | Path) -> None:
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(os, "replace", fail_publish)

    with pytest.raises(SubAgentStoreError, match="could not be saved"):
        store.register(
            title="Unpublished task",
            task="Must not be queued.",
            parent_run_id=_RUN_ID,
            source=SubAgentSource(
                kind=SubAgentSourceKind.FOREGROUND,
                restore_run_token=_RESTORE_TOKEN,
            ),
            creator_snapshot=_snapshot(),
        )

    assert store.list().items == ()


def test_failed_update_keeps_the_previous_complete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = store.register(
        title="Inspect",
        task="Keep the previous checkpoint.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=_RESTORE_TOKEN,
        ),
        creator_snapshot=_snapshot(),
    )
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=queued.revision + 1,
    )
    store.save(running)
    next_checkpoint = replace(
        running,
        conversation=({"role": "assistant", "content": "partial output"},),
        revision=running.revision + 1,
    )

    def fail_publish(source: str | bytes | Path, target: str | bytes | Path) -> None:
        raise OSError("simulated atomic publication failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", fail_publish)
        with pytest.raises(SubAgentStoreError, match="could not be saved"):
            store.save(next_checkpoint)

    assert store.get(running.agent_id) == running


def test_checkpoint_can_cover_stream_revisions_without_accepting_stale_writes(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Stream checkpoint")
    checkpoint = replace(queued, revision=queued.revision + 8)
    with pytest.raises(SubAgentStoreError, match="stale"):
        store.save(checkpoint)
    assert store.save(checkpoint, expected_revision=queued.revision) == checkpoint
    stale = replace(queued, revision=checkpoint.revision + 8)
    with pytest.raises(SubAgentStoreError, match="stale"):
        store.save(stale, expected_revision=queued.revision)
    assert store.get(queued.agent_id) == checkpoint


def test_reopening_marks_only_active_records_interrupted_and_keeps_outputs(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Queued")
    queued_record = queued
    running = _register(store, "Running")
    running = replace(
        running,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        conversation=({"role": "assistant", "content": "saved before restart"},),
        revision=running.revision + 1,
    )
    store.save(running)

    completed = _register(store, "Completed")
    completed_running = replace(
        completed,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=completed.revision + 1,
    )
    store.save(completed_running)
    completed = replace(
        completed_running,
        status=SubAgentStatus.COMPLETED,
        finished_at=_NOW,
        result="done",
        revision=completed_running.revision + 1,
    )
    store.save(completed)

    failed = _register(store, "Failed")
    failed_running = replace(
        failed,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=failed.revision + 1,
    )
    store.save(failed_running)
    failed = replace(
        failed_running,
        status=SubAgentStatus.FAILED,
        finished_at=_NOW,
        error=SubAgentError(code="provider_error", message="The model request failed."),
        revision=failed_running.revision + 1,
    )
    store.save(failed)

    cancelled = _register(store, "Cancelled")
    cancelled_running = replace(
        cancelled,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=cancelled.revision + 1,
    )
    store.save(cancelled_running)
    cancelled = replace(
        cancelled_running,
        status=SubAgentStatus.CANCELLED,
        finished_at=_NOW,
        revision=cancelled_running.revision + 1,
    )
    store.save(cancelled)

    interrupted = _register(store, "Interrupted")
    interrupted_running = replace(
        interrupted,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=interrupted.revision + 1,
    )
    store.save(interrupted_running)
    interrupted = replace(
        interrupted_running,
        status=SubAgentStatus.INTERRUPTED,
        finished_at=_NOW,
        error=SubAgentError(code="service_interrupted", message="Service stopped."),
        revision=interrupted_running.revision + 1,
    )
    store.save(interrupted)

    reopened = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    recovered_queued = reopened.get(queued_record.agent_id)
    recovered_running = reopened.get(running.agent_id)
    assert recovered_queued is not None
    assert recovered_running is not None
    assert recovered_queued.status is SubAgentStatus.INTERRUPTED
    assert recovered_running.status is SubAgentStatus.INTERRUPTED
    assert recovered_running.conversation == running.conversation
    assert recovered_running.revision == running.revision + 1
    assert reopened.get(completed.agent_id) == completed
    assert reopened.get(failed.agent_id) == failed
    assert reopened.get(cancelled.agent_id) == cancelled
    assert reopened.get(interrupted.agent_id) == interrupted

    reopened.recover()
    assert reopened.get(running.agent_id) == recovered_running


def test_workspace_recovery_processes_active_records_in_every_session(tmp_path: Path) -> None:
    state, first_session_id = _workspace(tmp_path)
    second_session_id = Session.create(state, now=lambda: _NOW).session_id
    first_store = SubAgentRecordStore(state, first_session_id, now=lambda: _NOW)
    second_store = SubAgentRecordStore(state, second_session_id, now=lambda: _NOW)
    first_record = _register(first_store, "First Session")
    second_queued = _register(second_store, "Second Session")
    second_running = replace(
        second_queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=second_queued.revision + 1,
    )
    second_store.save(second_running)

    SubAgentRecordStore.recover_workspace(state, now=lambda: _NOW)

    recovered_first = first_store.get(first_record.agent_id)
    recovered_second = second_store.get(second_running.agent_id)
    assert recovered_first is not None
    assert recovered_second is not None
    assert recovered_first.status is SubAgentStatus.INTERRUPTED
    assert recovered_second.status is SubAgentStatus.INTERRUPTED


def test_context_compression_update_keeps_the_complete_conversation(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Compress context")
    conversation = (
        {"role": "assistant", "content": "I'll inspect the file."},
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call-1",
            "content": "complete file output",
        },
    )
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        conversation=conversation,
        context_state={"last_compacted": 0, "messages": list(conversation)},
        revision=queued.revision + 1,
    )
    store.save(running)
    compressed = replace(
        running,
        context_state={
            "last_compacted": 2,
            "messages": [{"role": "system", "content": "summary"}],
        },
        revision=running.revision + 1,
    )

    store.save(compressed)

    loaded = store.get(compressed.agent_id)
    assert loaded is not None
    assert loaded.context_state == compressed.context_state
    assert loaded.conversation == conversation


def test_repeated_terminal_save_keeps_one_record(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Complete once")
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        revision=queued.revision + 1,
    )
    store.save(running)
    completed = replace(
        running,
        status=SubAgentStatus.COMPLETED,
        finished_at=_NOW,
        result="Done.",
        revision=running.revision + 1,
    )
    store.save(completed)

    assert store.save(completed) == completed
    assert tuple(item.agent_id for item in store.list().items) == (completed.agent_id,)


def test_list_uses_twenty_item_default_and_rejects_out_of_range_limits(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    records = tuple(_register(store, f"Task {index}") for index in range(21))

    default_page = store.list()
    maximum_page = store.list(limit=100)

    assert len(default_page.items) == 20
    assert default_page.next_cursor is not None
    assert tuple(item.agent_id for item in maximum_page.items) == tuple(
        record.agent_id for record in records
    )
    assert maximum_page.next_cursor is None
    for invalid_limit in (0, 101, True):
        with pytest.raises(SubAgentRequestError):
            store.list(limit=invalid_limit)


@pytest.mark.parametrize("schema_version", [1, 9])
def test_unknown_schema_version_is_reported_without_writing_records(
    tmp_path: Path, schema_version: int
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register(store, "Future schema")
    path = state.subagents_directory / session_id / f"{record.agent_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = schema_version
    if schema_version == 1:
        value["creator_snapshot"]["tool_schemas"] = [
            {"name": name, "input_schema": {"type": "object"}}
            for name in value["creator_snapshot"].pop("tool_names")
        ]
    path.write_text(json.dumps(value), encoding="utf-8")
    original = path.read_bytes()
    writes: list[Path] = []

    def observe_write(target: Path, content: str) -> None:
        writes.append(target)
        HOST_FILESYSTEM.atomic_replace_text(target, content)

    with pytest.raises(SubAgentStoreError, match="schema version is unsupported"):
        store.get(record.agent_id)
    with pytest.raises(SubAgentStoreError, match="schema version is unsupported"):
        SubAgentRecordStore(state, session_id, now=lambda: _NOW, replace_text=observe_write)
    assert writes == []
    assert path.read_bytes() == original


@pytest.mark.parametrize("names", [("",), (" ",), (3,), ("read_file", "read_file")])
def test_creator_snapshot_rejects_invalid_tool_names(names: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="tool_names"):
        replace(_snapshot(), tool_names=cast(tuple[str, ...], names))


def test_schedule_source_round_trips_job_and_occurrence_ownership(tmp_path: Path) -> None:
    state, foreground_session_id = _workspace(tmp_path)
    job_id = "123e4567-e89b-42d3-a456-426614174002"
    session_id = Session.create_schedule(state, job_id=job_id, now=lambda: _NOW).session_id
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    schedule_source = SubAgentSource(
        kind=SubAgentSourceKind.SCHEDULE,
        job_id=job_id,
        occurrence_id="2026-10-08T09:00:00Z",
    )
    record = store.register(
        title="Scheduled task",
        task="Preserve the Schedule source.",
        parent_run_id=_RUN_ID,
        source=schedule_source,
        creator_snapshot=_snapshot(),
    )

    assert store.get(record.agent_id) == record
    foreground_store = SubAgentRecordStore(state, foreground_session_id, now=lambda: _NOW)
    assert foreground_store.get(record.agent_id) is None
    assert foreground_store.list().items == ()


def test_session_id_and_artifact_paths_cannot_escape_their_storage_scope(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)
    with pytest.raises(ValueError):
        SubAgentRecordStore(state, "../../other-session", now=lambda: _NOW)

    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register(store, "Path validation")
    for path in (".", "./", "../outside.txt", "C:/outside.txt", "subagents\\outside.txt"):
        with pytest.raises(ValueError, match="remain within the Workspace"):
            replace(record, artifact_paths=(path,))


def test_failed_registration_after_publication_does_not_leave_a_queued_record(
    tmp_path: Path,
) -> None:
    state, session_id = _workspace(tmp_path)

    def publish_then_fail(path: Path, content: str) -> None:
        HOST_FILESYSTEM.atomic_replace_text(path, content)
        raise OSError("simulated post-publication failure")

    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW, replace_text=publish_then_fail)
    with pytest.raises(SubAgentStoreError, match="could not be saved"):
        _register(store, "Unacknowledged registration")

    assert store.list().items == ()
    assert SubAgentRecordStore(state, session_id, now=lambda: _NOW).list().items == ()


def test_registration_freezes_the_creators_tool_names(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    snapshot = _snapshot()
    queued = store.register(
        title="Captured tools",
        task="Use the registered capabilities.",
        parent_run_id=_RUN_ID,
        source=SubAgentSource(kind=SubAgentSourceKind.FOREGROUND, restore_run_token=_RESTORE_TOKEN),
        creator_snapshot=snapshot,
    )

    snapshot = replace(snapshot, tool_names=(*snapshot.tool_names, "changed_tool"))

    assert queued.creator_snapshot.tool_names == ("read_file", "list_dir")
    assert snapshot.tool_names == ("read_file", "list_dir", "changed_tool")
    assert store.get(queued.agent_id) == queued
    running = replace(
        queued, status=SubAgentStatus.RUNNING, started_at=_NOW, revision=queued.revision + 1
    )
    assert store.save(running) == running


@pytest.mark.parametrize("payload", [{"nested": {1: "value"}}, {"nested": (1, 2)}])
def test_checkpoint_rejects_values_that_would_change_after_json_round_trip(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Exact JSON checkpoint")

    with pytest.raises(ValueError, match="standard JSON values"):
        replace(queued, context_state=payload)
    with pytest.raises(ValueError, match="standard JSON values"):
        replace(queued, conversation=(payload,))

    assert store.get(queued.agent_id) == queued


def test_corrupt_artifact_path_reports_a_record_loading_error(tmp_path: Path) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    record = _register(store, "Corrupt Artifact path")
    path = state.subagents_directory / session_id / f"{record.agent_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["artifact_paths"] = ["."]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SubAgentStoreError, match="remain within the Workspace"):
        store.get(record.agent_id)


def test_failed_update_after_publication_leaves_the_new_complete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, session_id = _workspace(tmp_path)
    store = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    queued = _register(store, "Published checkpoint")
    running = replace(
        queued,
        status=SubAgentStatus.RUNNING,
        started_at=_NOW,
        conversation=({"role": "assistant", "content": "Complete saved exchange"},),
        revision=queued.revision + 1,
    )
    original_replace = os.replace

    def replace_then_fail(source: str | bytes | Path, target: str | bytes | Path) -> None:
        original_replace(source, target)
        raise OSError("simulated failure after atomic replacement")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", replace_then_fail)
        with pytest.raises(SubAgentStoreError, match="could not be saved"):
            store.save(running)

    assert store.get(running.agent_id) == running
    reopened = SubAgentRecordStore(state, session_id, now=lambda: _NOW)
    recovered = reopened.get(running.agent_id)
    assert recovered is not None
    assert recovered.status is SubAgentStatus.INTERRUPTED
    assert recovered.conversation == running.conversation
