from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import myclaw.agent.session.restore as restore_module
from myclaw.agent.session._restore_persistence import canonical_json_bytes, sha256_hex
from myclaw.agent.session.backup_store import FileBackupStore
from myclaw.agent.session.restore import (
    RestoreManager,
    RestoreMode,
    RestoreRecoveryRequired,
    RestoreSafetyError,
    StaleRestorePlan,
)
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

SESSION_ID = "20260926-120000-123456_12345678-1234-4234-8234-123456789abc"
FIRST_TOKEN = UUID("12345678-1234-4234-8234-123456789abc")
SECOND_TOKEN = UUID("22345678-1234-4234-8234-123456789abc")
NOW = datetime(2026, 9, 26, 12, 0, 0, 123000, tzinfo=UTC)
AfterRestorePhase = Callable[[str, Callable[[], None]], None]


def _commit_user(session: Session, content: str, token: UUID) -> None:
    before = session.capture_restore_before()
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
        restore_before=before,
        restore_run_token=token,
    )


def _write_v1_state(path: Path) -> None:
    body: dict[str, object] = {
        "schema_version": 1,
        "next_operation_id": 2,
        "revision": 1,
        "journal_operation_ids": [1],
        "discarded_operation_ids": [],
    }
    signed = dict(body)
    signed["integrity_sha256"] = sha256_hex(canonical_json_bytes(body))
    path.write_bytes(canonical_json_bytes(signed))


@pytest.mark.asyncio
async def test_inspect_returns_frozen_plan_with_earliest_backup(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "tracked.txt"
    target.write_bytes(b"before first write")

    _commit_user(session, "first", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    first_ticket = store.before_write(FIRST_TOKEN, target)
    assert first_ticket is not None
    target.write_bytes(b"after first write")
    store.after_write(first_ticket)

    _commit_user(session, "second", SECOND_TOKEN)
    second_ticket = store.before_write(SECOND_TOKEN, target)
    assert second_ticket is not None
    target.write_bytes(b"after second write")
    store.after_write(second_ticket)
    await session.wait_for_pending_persist()

    plan = RestoreManager(state).inspect(session, 1)

    assert plan.session_id == session.session_id
    assert plan.anchor_id == 1
    assert plan.removed_users == 2
    assert plan.removed_messages == 2
    assert plan.external_target_count == 0
    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES)
    assert len(plan.targets) == 1
    assert plan.targets[0].canonical_target == target.resolve()
    assert plan.targets[0].before_bytes == b"before first write"
    assert plan.targets[0].latest_after_sha256 is not None
    with pytest.raises(FrozenInstanceError):
        plan.anchor_id = 2  # type: ignore[misc]


@pytest.mark.asyncio
async def test_conversation_only_clears_discarded_journal_and_restores_session(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "kept.txt"
    target.write_bytes(b"before")
    store = FileBackupStore(state, session.session_id)

    _commit_user(session, "first", FIRST_TOKEN)
    first_ticket = store.before_write(FIRST_TOKEN, target)
    assert first_ticket is not None
    target.write_bytes(b"first branch")
    store.after_write(first_ticket)
    _commit_user(session, "second", SECOND_TOKEN)
    second_ticket = store.before_write(SECOND_TOKEN, target)
    assert second_ticket is not None
    target.write_bytes(b"discarded branch")
    store.after_write(second_ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    result = await manager.execute(plan, RestoreMode.CONVERSATION_ONLY)

    assert result.mode is RestoreMode.CONVERSATION_ONLY
    assert result.removed_users == 2
    assert result.removed_messages == 2
    assert result.failures == ()
    assert target.read_bytes() == b"discarded branch"
    loaded = Session.load(state, session.session_id, now=lambda: NOW)
    assert loaded.messages == []
    assert FileBackupStore(state, session.session_id).inspect().entries == ()

    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert recovered == result


@pytest.mark.asyncio
async def test_file_restore_replays_conflicts_deletes_new_files_and_continues_after_failure(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    store = FileBackupStore(state, session.session_id)
    existing = workspace / "existing.txt"
    created_directory = workspace / "created-directory"
    created_directory.mkdir()
    created = created_directory / "new.txt"
    failed = workspace / "failed.txt"
    existing.write_bytes(b"original")
    failed.write_bytes(b"failed original")

    _commit_user(session, "restore files", FIRST_TOKEN)
    for target, after in (
        (existing, b"tool result"),
        (created, b"created by tool"),
        (failed, b"failed tool result"),
    ):
        ticket = store.before_write(FIRST_TOKEN, target)
        assert ticket is not None
        target.write_bytes(after)
        store.after_write(ticket)
    existing.write_bytes(b"changed later")
    await session.wait_for_pending_persist()

    original_replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_one_target(target: Path, content: bytes) -> None:
        if Path(target) == failed:
            raise PermissionError("injected restore permission failure")
        original_replace(target, content)

    synced_directories: list[Path] = []
    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_one_target)
    monkeypatch.setattr(
        restore_module,
        "sync_directory",
        lambda path: synced_directories.append(Path(path)),
    )
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    assert plan.backup_gaps == ()
    result = await manager.execute(plan, RestoreMode.FILES)

    assert existing.read_bytes() == b"original"
    assert not created.exists()
    assert created_directory.is_dir()
    assert created_directory.resolve() in synced_directories
    assert failed.read_bytes() == b"failed tool result"
    assert result.successful_conflicts == (existing,)
    assert result.failed_files == (failed,)
    assert result.failure_notification_pending is True
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []
    assert FileBackupStore(state, session.session_id).inspect().entries == ()

    acknowledged = manager.acknowledge_failure_notification()
    assert acknowledged is not None
    assert acknowledged.failure_notification_acknowledged is True
    assert acknowledged.failure_notification_pending is False
    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert recovered == acknowledged
    assert await RestoreManager(state, now=lambda: NOW).recover_pending() is None


@pytest.mark.asyncio
async def test_revalidate_rejects_a_persisted_session_change_without_mutation(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "stale.txt"
    target.write_bytes(b"before")
    _commit_user(session, "stale", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    session.update_metadata(title="Changed after inspection")
    session.persist()
    await session.wait_for_pending_persist()

    with pytest.raises(StaleRestorePlan):
        manager.revalidate(plan)
    assert target.read_bytes() == b"after"


@pytest.mark.asyncio
async def test_execute_rechecks_session_digest_after_revalidation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "stale-during-execute.txt"
    target.write_bytes(b"before")
    _commit_user(session, "stale during execute", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    manager.revalidate(plan)
    session.update_metadata(title="Changed after the execution revalidation")
    session.persist()
    await session.wait_for_pending_persist()
    monkeypatch.setattr(manager, "revalidate", lambda value: value)

    with pytest.raises(StaleRestorePlan, match="changed during execution"):
        await manager.execute(plan, RestoreMode.FILES)

    assert target.read_bytes() == b"after"
    assert len(FileBackupStore(state, session.session_id).inspect().entries) == 1
    assert not (workspace / ".myclaw" / "restore" / session.session_id / "pending.json").exists()


@pytest.mark.asyncio
async def test_backup_gap_only_disables_file_mode_inside_selected_range(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    store = FileBackupStore(state, session.session_id)
    blocked = workspace / "blocked-directory"
    blocked.mkdir()
    valid = workspace / "valid.txt"

    _commit_user(session, "gap", FIRST_TOKEN)
    gap_ticket = store.before_write(FIRST_TOKEN, blocked)
    assert gap_ticket is not None and gap_ticket.recorded is False

    _commit_user(session, "valid", SECOND_TOKEN)
    valid.write_bytes(b"before")
    valid_ticket = store.before_write(SECOND_TOKEN, valid)
    assert valid_ticket is not None
    valid.write_bytes(b"after")
    store.after_write(valid_ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    later_plan = manager.inspect(session, 2)
    earlier_plan = manager.inspect(session, 1)

    assert later_plan.available_modes == (RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES)
    assert earlier_plan.available_modes == (RestoreMode.CONVERSATION_ONLY,)
    assert len(earlier_plan.backup_gaps) == 1


@pytest.mark.asyncio
async def test_incomplete_post_write_state_disables_files_after_external_change(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "incomplete-post-write.txt"
    target.write_bytes(b"before")
    _commit_user(session, "incomplete post-write", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"tool result")
    target.write_bytes(b"external modification")
    await session.wait_for_pending_persist()

    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 1)

    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY,)
    assert [gap.reason for gap in plan.backup_gaps] == ["post_write_state_unavailable"]


@pytest.mark.asyncio
async def test_incomplete_post_write_state_outside_selected_range_does_not_disable_files(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    incomplete = workspace / "incomplete-earlier.txt"
    complete = workspace / "complete-later.txt"
    incomplete.write_bytes(b"before incomplete")
    complete.write_bytes(b"before complete")
    store = FileBackupStore(state, session.session_id)

    _commit_user(session, "incomplete", FIRST_TOKEN)
    incomplete_ticket = store.before_write(FIRST_TOKEN, incomplete)
    assert incomplete_ticket is not None
    incomplete.write_bytes(b"tool result")
    _commit_user(session, "complete", SECOND_TOKEN)
    complete_ticket = store.before_write(SECOND_TOKEN, complete)
    assert complete_ticket is not None
    complete.write_bytes(b"after complete")
    store.after_write(complete_ticket)
    await session.wait_for_pending_persist()

    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 2)

    assert plan.backup_gaps == ()
    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES)


@pytest.mark.asyncio
async def test_missing_journal_entry_only_disables_files_inside_selected_range(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    valid = workspace / "valid-selected.txt"
    missing = workspace / "missing-selected.txt"
    valid.write_bytes(b"before valid")
    missing.write_bytes(b"before missing")
    store = FileBackupStore(state, session.session_id)

    _commit_user(session, "valid", FIRST_TOKEN)
    valid_ticket = store.before_write(FIRST_TOKEN, valid)
    assert valid_ticket is not None
    valid.write_bytes(b"after valid")
    store.after_write(valid_ticket)
    _commit_user(session, "missing", SECOND_TOKEN)
    missing_ticket = store.before_write(SECOND_TOKEN, missing)
    assert missing_ticket is not None
    (workspace / ".myclaw" / "restore" / session.session_id / "entries" / "2.json").unlink()
    await session.wait_for_pending_persist()

    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 1)

    assert len(plan.integrity_issues) == 1
    assert plan.integrity_issues[0].operation_id == missing_ticket.operation_id
    assert plan.integrity_issues[0].run_token == SECOND_TOKEN
    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY,)


@pytest.mark.asyncio
async def test_known_missing_journal_entry_outside_selected_range_does_not_disable_files(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    missing = workspace / "missing-earlier.txt"
    valid = workspace / "valid-later.txt"
    missing.write_bytes(b"before missing")
    valid.write_bytes(b"before valid")
    store = FileBackupStore(state, session.session_id)

    _commit_user(session, "missing", FIRST_TOKEN)
    missing_ticket = store.before_write(FIRST_TOKEN, missing)
    assert missing_ticket is not None
    (workspace / ".myclaw" / "restore" / session.session_id / "entries" / "1.json").unlink()
    _commit_user(session, "valid", SECOND_TOKEN)
    valid_ticket = store.before_write(SECOND_TOKEN, valid)
    assert valid_ticket is not None
    valid.write_bytes(b"after valid")
    store.after_write(valid_ticket)
    await session.wait_for_pending_persist()

    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 2)

    assert plan.integrity_issues == ()
    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY, RestoreMode.FILES)


@pytest.mark.asyncio
async def test_conversation_only_restore_tombstones_missing_entry_without_reopening_it(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "missing-known.txt"
    target.write_bytes(b"before")
    _commit_user(session, "missing known", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    (workspace / ".myclaw" / "restore" / session.session_id / "entries" / "1.json").unlink()
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    result = await manager.execute(plan, RestoreMode.CONVERSATION_ONLY)

    assert result.mode is RestoreMode.CONVERSATION_ONLY
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []
    journal = FileBackupStore(state, session.session_id).inspect()
    assert journal.entries == ()
    assert journal.integrity_issues == ()


@pytest.mark.asyncio
async def test_unknown_v1_integrity_issue_disables_files_but_allows_conversation_restore(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "legacy-unknown.txt"
    target.write_bytes(b"before")
    _commit_user(session, "legacy unknown", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    root = workspace / ".myclaw" / "restore" / session.session_id
    (root / "entries" / "1.json").unlink()
    _write_v1_state(root / "state.json")
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)

    assert len(plan.integrity_issues) == 1
    assert plan.integrity_issues[0].run_token is None
    assert plan.available_modes == (RestoreMode.CONVERSATION_ONLY,)

    result = await manager.execute(plan, RestoreMode.CONVERSATION_ONLY)

    assert result.mode is RestoreMode.CONVERSATION_ONLY
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interrupted_phase",
    [
        "pending_intent",
        "journal_pruned",
        "file_intent",
        "file_result",
        "files_replayed",
        "session_write",
        "session_persisted",
        "complete",
    ],
)
async def test_recovery_is_idempotent_after_each_durable_phase(
    workspace: Path,
    interrupted_phase: str,
    after_restore_phase: AfterRestorePhase,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "recover.txt"
    target.write_bytes(b"before")
    _commit_user(session, "recover", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 1)

    def interrupt() -> None:
        raise RuntimeError(f"injected interruption after {interrupted_phase}")

    after_restore_phase(interrupted_phase, interrupt)
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await manager.execute(plan, RestoreMode.FILES)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "injected interruption" in str(raised.value.__cause__)

    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    repeated = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()

    assert recovered is not None
    assert repeated == recovered
    assert recovered.session_id == session.session_id
    assert target.read_bytes() == b"before"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []
    assert FileBackupStore(state, session.session_id).inspect().entries == ()


@pytest.mark.asyncio
async def test_startup_recovery_prefers_incomplete_transaction_over_completed_result(
    workspace: Path,
    after_restore_phase: AfterRestorePhase,
) -> None:
    state = WorkspaceState(workspace)
    completed_session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    completed_target = workspace / "completed.txt"
    completed_target.write_bytes(b"completed before")
    _commit_user(completed_session, "completed", FIRST_TOKEN)
    completed_store = FileBackupStore(state, completed_session.session_id)
    completed_ticket = completed_store.before_write(FIRST_TOKEN, completed_target)
    assert completed_ticket is not None
    completed_target.write_bytes(b"completed after")
    completed_store.after_write(completed_ticket)
    await completed_session.wait_for_pending_persist()
    completed_manager = RestoreManager(state, completed_session.session_id, now=lambda: NOW)
    await completed_manager.execute(
        completed_manager.inspect(completed_session, 1),
        RestoreMode.FILES,
    )

    pending_session = Session.create(state, new_uuid=lambda: SECOND_TOKEN, now=lambda: NOW)
    pending_target = workspace / "pending.txt"
    pending_target.write_bytes(b"pending before")
    _commit_user(pending_session, "pending", SECOND_TOKEN)
    pending_store = FileBackupStore(state, pending_session.session_id)
    pending_ticket = pending_store.before_write(SECOND_TOKEN, pending_target)
    assert pending_ticket is not None
    pending_target.write_bytes(b"pending after")
    pending_store.after_write(pending_ticket)
    await pending_session.wait_for_pending_persist()

    def interrupt() -> None:
        raise RuntimeError("injected pending transaction")

    after_restore_phase("pending_intent", interrupt)
    pending_manager = RestoreManager(state, pending_session.session_id, now=lambda: NOW)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await pending_manager.execute(
            pending_manager.inspect(pending_session, 1),
            RestoreMode.FILES,
        )
    assert isinstance(raised.value.__cause__, RuntimeError)

    recovered = await RestoreManager(state, now=lambda: NOW).recover_pending()

    assert recovered is not None
    assert recovered.session_id == pending_session.session_id
    assert pending_target.read_bytes() == b"pending before"
    assert Session.load(state, pending_session.session_id, now=lambda: NOW).messages == []
    assert await RestoreManager(state, now=lambda: NOW).recover_pending() is None


@pytest.mark.asyncio
async def test_missing_safety_snapshot_does_not_mutate_on_recovery(
    workspace: Path,
    after_restore_phase: AfterRestorePhase,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "missing-safety.txt"
    target.write_bytes(b"before")
    _commit_user(session, "missing safety", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 1)

    def remove_safety_after_intent() -> None:
        shutil.rmtree(workspace / ".myclaw" / "restore" / session.session_id / "latest-safety")
        raise RuntimeError("injected crash after pending intent")

    after_restore_phase("pending_intent", remove_safety_after_intent)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await RestoreManager(state, session.session_id, now=lambda: NOW).execute(
            plan, RestoreMode.FILES
        )
    assert isinstance(raised.value.__cause__, RuntimeError)

    with pytest.raises(RestoreSafetyError):
        await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert target.read_bytes() == b"after"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages[-1]["content"] == (
        "missing safety"
    )
    assert len(FileBackupStore(state, session.session_id).inspect().entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damaged_name", ["session.jsonl", "journal.json", "targets.json", "complete"]
)
async def test_damaged_safety_snapshot_content_does_not_mutate_on_recovery(
    workspace: Path,
    damaged_name: str,
    after_restore_phase: AfterRestorePhase,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "damaged-safety.txt"
    target.write_bytes(b"before")
    _commit_user(session, "damaged safety", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    plan = RestoreManager(state, session.session_id, now=lambda: NOW).inspect(session, 1)

    def damage_snapshot() -> None:
        latest = workspace / ".myclaw" / "restore" / session.session_id / "latest-safety"
        manifest = json.loads((latest / "manifest.json").read_bytes())
        (latest / manifest["generation"] / damaged_name).write_bytes(b"")
        raise RuntimeError("injected crash with damaged safety snapshot")

    after_restore_phase("pending_intent", damage_snapshot)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await RestoreManager(state, session.session_id, now=lambda: NOW).execute(
            plan, RestoreMode.FILES
        )
    assert isinstance(raised.value.__cause__, RuntimeError)

    with pytest.raises(RestoreSafetyError):
        await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert target.read_bytes() == b"after"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages[-1]["content"] == (
        "damaged safety"
    )
    assert len(FileBackupStore(state, session.session_id).inspect().entries) == 1


@pytest.mark.asyncio
async def test_safety_snapshot_write_failure_causes_zero_mutation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "snapshot-failure.txt"
    target.write_bytes(b"before")
    _commit_user(session, "snapshot failure", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    original_replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_snapshot(path: Path, content: bytes) -> None:
        if Path(path).name == "journal.json":
            raise OSError("injected safety snapshot failure")
        original_replace(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_snapshot)
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    with pytest.raises(RestoreSafetyError):
        await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert target.read_bytes() == b"after"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages[-1]["content"] == (
        "snapshot failure"
    )
    assert len(FileBackupStore(state, session.session_id).inspect().entries) == 1


@pytest.mark.asyncio
async def test_pending_write_failure_after_replace_requires_recovery(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    _commit_user(session, "pending write failure", FIRST_TOKEN)
    await session.wait_for_pending_persist()
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    plan = manager.inspect(session, 1)
    original_restrict = HOST_FILESYSTEM.restrict_private_file

    def fail_after_replace(path: Path) -> None:
        if path.name == "pending.json":
            raise OSError("injected post-replace failure")
        original_restrict(path)

    monkeypatch.setattr(HOST_FILESYSTEM, "restrict_private_file", fail_after_replace)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await manager.execute(plan, RestoreMode.CONVERSATION_ONLY)
    assert isinstance(raised.value.__cause__, OSError)
    assert "post-replace failure" not in str(raised.value)
    assert Session.load(state, session.session_id, now=lambda: NOW).messages[-1]["content"] == (
        "pending write failure"
    )

    monkeypatch.undo()
    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert recovered is not None
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
async def test_published_snapshot_survives_old_generation_cleanup_failure(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_restore_phase: AfterRestorePhase,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "snapshot-cleanup.txt"
    target.write_bytes(b"before")
    _commit_user(session, "first restore", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    first_ticket = store.before_write(FIRST_TOKEN, target)
    assert first_ticket is not None
    target.write_bytes(b"first after")
    store.after_write(first_ticket)
    await session.wait_for_pending_persist()
    first_manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    await first_manager.execute(first_manager.inspect(session, 1), RestoreMode.FILES)

    latest = workspace / ".myclaw" / "restore" / session.session_id / "latest-safety"
    first_manifest = json.loads((latest / "manifest.json").read_bytes())
    first_generation = first_manifest["generation"]

    continued = Session.load(state, session.session_id, now=lambda: NOW)
    _commit_user(continued, "second restore", SECOND_TOKEN)
    second_ticket = store.before_write(SECOND_TOKEN, target)
    assert second_ticket is not None
    target.write_bytes(b"second after")
    store.after_write(second_ticket)
    await continued.wait_for_pending_persist()

    original_rmtree = shutil.rmtree

    def refuse_old_generation(path: Path) -> None:
        if Path(path).name == first_generation:
            raise PermissionError("injected old generation cleanup failure")
        original_rmtree(path)

    def interrupt() -> None:
        raise RuntimeError("injected crash after replacement snapshot")

    monkeypatch.setattr(shutil, "rmtree", refuse_old_generation)
    after_restore_phase("pending_intent", interrupt)
    second_manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await second_manager.execute(second_manager.inspect(continued, 2), RestoreMode.FILES)
    assert isinstance(raised.value.__cause__, RuntimeError)

    second_manifest = json.loads((latest / "manifest.json").read_bytes())
    assert second_manifest["generation"] != first_generation
    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert recovered is not None
    assert target.read_bytes() == b"before"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
async def test_session_write_failure_retains_pending_for_later_recovery(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "session-write-failure.txt"
    target.write_bytes(b"before")
    _commit_user(session, "session write failure", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    def fail_session_write(self: Session, anchor_id: int) -> object:
        raise OSError("injected strict Session write failure")

    monkeypatch.setattr(Session, "restore_before_durably", fail_session_write)
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    with pytest.raises(RestoreRecoveryRequired) as raised:
        await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)
    assert isinstance(raised.value.__cause__, OSError)
    assert "strict Session write failure" in str(raised.value.__cause__)
    assert target.read_bytes() == b"before"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages[-1]["content"] == (
        "session write failure"
    )

    monkeypatch.undo()
    recovered = await RestoreManager(state, session.session_id, now=lambda: NOW).recover_pending()
    assert recovered is not None
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []
    assert target.read_bytes() == b"before"


@pytest.mark.asyncio
async def test_runtime_owned_target_is_reported_as_file_failure(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    _commit_user(session, "runtime target", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    runtime_target = state.schedule_path
    runtime_target.parent.mkdir(exist_ok=True)
    runtime_target.write_bytes(b"before")
    ticket = store.before_write(FIRST_TOKEN, runtime_target)
    assert ticket is not None
    runtime_target.write_bytes(b"after")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failed_files == (runtime_target.resolve(),)
    assert runtime_target.read_bytes() == b"after"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
async def test_other_session_jsonl_is_a_runtime_owned_failure(workspace: Path) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    other = Session.create(state, new_uuid=lambda: SECOND_TOKEN, now=lambda: NOW)
    _commit_user(other, "other Session", SECOND_TOKEN)
    await other.wait_for_pending_persist()
    other_path = state.sessions_directory / f"{other.session_id}.jsonl"
    original_other_bytes = other_path.read_bytes()

    _commit_user(session, "runtime Session target", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, other_path)
    assert ticket is not None
    other_path.write_bytes(b"later runtime bytes")
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failed_files == (other_path.resolve(),)
    assert other_path.read_bytes() == b"later runtime bytes"
    assert other_path.read_bytes() != original_other_bytes
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
async def test_hard_link_target_is_left_unchanged_and_reported(workspace: Path) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "hard-linked.txt"
    alias = workspace / "hard-link-alias.txt"
    target.write_bytes(b"before")
    _commit_user(session, "hard link", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    os.link(target, alias)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failed_files == (target.resolve(),)
    assert target.read_bytes() == b"after"
    assert alias.read_bytes() == b"after"
    assert Session.load(state, session.session_id, now=lambda: NOW).messages == []


@pytest.mark.asyncio
async def test_file_restore_does_not_rewrite_a_target_already_matching_backup(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: FIRST_TOKEN, now=lambda: NOW)
    target = workspace / "already-restored.txt"
    target.write_bytes(b"before")
    _commit_user(session, "already restored", FIRST_TOKEN)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(FIRST_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"after")
    store.after_write(ticket)
    target.write_bytes(b"before")
    await session.wait_for_pending_persist()

    replacements: list[Path] = []
    original_replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replacements(path: Path, content: bytes) -> None:
        if Path(path) == target:
            replacements.append(Path(path))
        original_replace(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replacements)
    manager = RestoreManager(state, session.session_id, now=lambda: NOW)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert replacements == []
    assert result.file_results[0].status.value == "unchanged"
    assert target.read_bytes() == b"before"
