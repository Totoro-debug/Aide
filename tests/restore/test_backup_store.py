from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from omni.agent.session import backup_store as backup_store_module
from omni.agent.session._restore_persistence import canonical_json_bytes, sha256_hex
from omni.agent.session.backup_store import (
    BackupGap,
    BackupIntegrityError,
    BackupIntegrityIssue,
    BackupStoreError,
    FileBackupStore,
)
from omni.agent.workspace_state import WorkspaceState
from omni.utils.host_filesystem import HOST_FILESYSTEM

SESSION_ID = "20260926-120000-123456_12345678-1234-4234-8234-123456789abc"


def _signed_json_bytes(value: dict[str, object]) -> bytes:
    body = {key: member for key, member in value.items() if key != "integrity_sha256"}
    signed = dict(body)
    signed["integrity_sha256"] = sha256_hex(canonical_json_bytes(body))
    return canonical_json_bytes(signed)


def _write_signed_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_signed_json_bytes(value))


def _rewrite_signed_json(path: Path, **updates: object) -> None:
    value = json.loads(path.read_bytes())
    value.pop("integrity_sha256", None)
    value.update(updates)
    _write_signed_json(path, value)


def test_restore_persistence_primitives_keep_canonical_bytes_and_digest() -> None:
    content = canonical_json_bytes({"z": "雪", "a": [1, True, None]})

    assert content == b'{"a":[1,true,null],"z":"\xe9\x9b\xaa"}'
    assert sha256_hex(content) == (
        "33290394113200823ebc40344b2b193d16aefef53759dd97228b0a4fcf8762c9"
    )
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_backup_store_records_existing_bytes_and_nonexistence(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    original = b"original\x00\xff\r\n"
    existing = workspace.parent / "existing.bin"
    missing = workspace.parent / "new.bin"
    existing.write_bytes(original)

    existing_ticket = store.before_write(uuid4(), existing)
    missing_ticket = store.before_write(uuid4(), missing)

    assert existing_ticket is not None
    assert missing_ticket is not None
    missing.write_bytes(b"created after backup")
    store.after_write(missing_ticket)
    journal = store.inspect()
    assert [entry.operation_id for entry in journal.entries] == [1, 2]
    assert journal.entries[0].before.exists is True
    assert store.read_backup(existing_ticket.operation_id) == original
    assert journal.entries[1].before.exists is False
    assert store.read_backup(missing_ticket.operation_id) is None
    assert journal.entries[1].after is not None
    assert journal.entries[1].after.exists is True
    assert journal.entries[1].after.sha256 == hashlib.sha256(b"created after backup").hexdigest()


def test_repeated_target_keeps_operation_order_and_post_write_hashes(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "repeated.bin"
    first = b"first state"
    middle = b"middle state"
    final = b"final state"
    first_token = uuid4()
    second_token = uuid4()
    target.write_bytes(first)

    first_ticket = store.before_write(first_token, target)
    assert first_ticket is not None
    target.write_bytes(middle)
    store.after_write(first_ticket)
    second_ticket = store.before_write(second_token, target)
    assert second_ticket is not None
    target.write_bytes(final)
    store.after_write(second_ticket)

    entries = store.inspect().entries
    assert [entry.operation_id for entry in entries] == [1, 2]
    assert [entry.run_token for entry in entries] == [first_token, second_token]
    assert entries[0].canonical_target == entries[1].canonical_target
    assert store.read_backup(1) == first
    assert store.read_backup(2) == middle
    assert entries[0].after is not None
    assert entries[0].after.sha256 == hashlib.sha256(middle).hexdigest()
    assert entries[1].after is not None
    assert entries[1].after.sha256 == hashlib.sha256(final).hexdigest()


def test_linked_target_keeps_its_original_canonical_destination(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    original = workspace.parent / "original.txt"
    replacement = workspace.parent / "replacement.txt"
    link = workspace.parent / "alias.txt"
    original.write_bytes(b"original target")
    replacement.write_bytes(b"replacement target")
    try:
        link.symlink_to(original)
    except OSError as error:
        pytest.skip(f"host does not permit creating a file symlink: {error}")

    ticket = store.before_write(uuid4(), link)
    assert ticket is not None
    assert ticket.canonical_target == original.resolve()
    link.unlink()
    try:
        link.symlink_to(replacement)
    except OSError as error:
        pytest.skip(f"host does not permit retargeting a file symlink: {error}")
    original.write_bytes(b"updated original target")
    store.after_write(ticket)

    entry = store.inspect().entries[0]
    assert entry.requested_target == str(link.absolute())
    assert entry.canonical_target == str(original.resolve())
    assert store.read_backup(ticket.operation_id) == b"original target"
    assert entry.after is not None
    assert entry.after.sha256 == hashlib.sha256(b"updated original target").hexdigest()


def test_restore_state_subtree_is_not_recorded_as_a_file_target(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    protected = workspace / ".omni" / "restore" / "other-session" / "protected.txt"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"internal state")

    assert store.before_write(uuid4(), protected) is None
    assert store.inspect().entries == ()
    assert protected.read_bytes() == b"internal state"


def test_hard_link_to_restore_blob_is_not_recorded_as_an_ordinary_target(
    workspace: Path,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "protected-source.bin"
    target.write_bytes(b"protected backup bytes")
    first_ticket = store.before_write(uuid4(), target)
    assert first_ticket is not None
    store.after_write(first_ticket)
    first_entry = store.inspect().entries[0]
    assert first_entry.before.blob_name is not None
    blob = workspace / ".omni" / "restore" / SESSION_ID / "blobs" / first_entry.before.blob_name
    alias = workspace.parent / "protected-blob-alias.bin"
    try:
        os.link(blob, alias)
    except OSError as error:
        pytest.skip(f"host does not permit creating a hard link: {error}")

    alias_ticket = store.before_write(uuid4(), alias)

    assert alias_ticket is not None
    assert alias_ticket.recorded is False
    alias.unlink()
    journal = store.inspect()
    assert [gap.reason for gap in journal.gaps] == ["target_read_failed"]
    assert store.read_backup(first_ticket.operation_id) == b"protected backup bytes"
    assert len(tuple(blob.parent.glob("*.bin"))) == 1


def test_directory_target_records_gap_without_raising(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "directory-target"
    target.mkdir()

    ticket = store.before_write(uuid4(), target)

    assert ticket is not None
    assert ticket.recorded is False
    assert [gap.reason for gap in store.inspect().gaps] == ["target_read_failed"]


def test_new_restore_directories_enter_the_durability_sync_path(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(backup_store_module, "sync_created_directory", synced.append)
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)

    store.before_write(uuid4(), workspace.parent / "new-target.bin")

    restore_root = workspace / ".omni" / "restore"
    session_root = restore_root / SESSION_ID
    expected = {
        restore_root.resolve(),
        session_root.resolve(),
        (session_root / "entries").resolve(),
        (session_root / "blobs").resolve(),
    }
    assert expected.issubset(set(synced))


def test_failed_directory_sync_is_retried_without_blocking_the_caller(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[Path] = []
    failed = False

    def fail_restore_once(path: Path) -> None:
        nonlocal failed
        attempts.append(path)
        if path.name == "restore" and not failed:
            failed = True
            raise OSError("injected directory sync failure")

    monkeypatch.setattr(backup_store_module, "sync_created_directory", fail_restore_once)
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "sync-retry.bin"

    first_ticket = store.before_write(uuid4(), target)
    second_ticket = store.before_write(uuid4(), target)

    assert first_ticket is None
    assert second_ticket is not None
    assert second_ticket.recorded is True
    restore_root = (workspace / ".omni" / "restore").resolve()
    assert attempts.count(restore_root) == 2
    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    assert reopened.read_backup(second_ticket.operation_id) is None


def test_target_read_failure_records_gap_and_does_not_raise(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "unreadable.bin"
    target.write_bytes(b"before")
    original_open = os.open
    canonical_target = str(target.resolve()).casefold()

    def fail_target_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if os.fsdecode(path).casefold().endswith(canonical_target):
            raise PermissionError("injected target read failure")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", fail_target_open)
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    assert ticket.recorded is False
    target.write_bytes(b"tool write continues")
    store.after_write(ticket)

    journal = store.inspect()
    assert len(journal.gaps) == 1
    assert journal.gaps[0].reason == "target_read_failed"
    assert target.read_bytes() == b"tool write continues"


def test_blob_failure_records_gap_without_changing_the_write_contract(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "blob-failure.bin"
    target.write_bytes(b"before")
    create = HOST_FILESYSTEM.atomic_create_bytes_with_identity

    def fail_blob(path: Path, content: bytes) -> tuple[int, int, int, int] | None:
        if path.suffix == ".bin":
            raise OSError("injected blob write failure")
        return create(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_create_bytes_with_identity", fail_blob)
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    assert ticket.recorded is False
    target.write_bytes(b"tool write continues")
    store.after_write(ticket)

    journal = store.inspect()
    assert len(journal.gaps) == 1
    assert journal.gaps[0].reason == "blob_write_failed"
    assert target.read_bytes() == b"tool write continues"


def test_journal_write_failure_can_be_recorded_as_a_gap(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "journal-failure.bin"
    target.write_bytes(b"before")
    create = HOST_FILESYSTEM.atomic_create_bytes_with_identity
    failed_once = False

    def fail_first_entry(path: Path, content: bytes) -> tuple[int, int, int, int] | None:
        nonlocal failed_once
        if path.suffix == ".json" and not failed_once:
            failed_once = True
            raise OSError("injected journal entry failure")
        return create(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_create_bytes_with_identity", fail_first_entry)
    ticket = store.before_write(uuid4(), target)

    assert ticket is not None
    assert ticket.recorded is False
    journal = store.inspect()
    assert len(journal.gaps) == 1
    assert journal.gaps[0].reason == "journal_write_failed"


def test_backup_and_gap_write_failures_do_not_escape_to_the_caller(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "all-storage-fails.bin"
    target.write_bytes(b"before")

    def fail_storage(*args: object, **kwargs: object) -> None:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_create_bytes_with_identity", fail_storage)
    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_storage)
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    target.write_bytes(b"tool write continues")
    store.after_write(ticket)

    assert target.read_bytes() == b"tool write continues"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_integrity_check_reports_missing_or_corrupt_blob(
    workspace: Path,
    damage: str,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / f"{damage}.bin"
    target.write_bytes(b"sensitive backup bytes")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    entry = store.inspect().entries[0]
    assert entry.before.blob_name is not None
    blob = workspace / ".omni" / "restore" / SESSION_ID / "blobs" / entry.before.blob_name
    if damage == "missing":
        blob.unlink()
        reason = "missing_or_unsafe_blob"
    else:
        blob.write_bytes(b"corrupted backup")
        reason = "hash_mismatch"

    assert store.inspect().integrity_issues == (
        BackupIntegrityIssue(1, reason, run_token),
    )
    with pytest.raises(BackupIntegrityError):
        store.read_backup(ticket.operation_id)
    assert b"sensitive backup bytes" not in repr(store.inspect()).encode("utf-8")
    assert (
        b"sensitive backup bytes"
        not in (workspace / ".omni" / "restore" / SESSION_ID / "entries" / "1.json").read_bytes()
    )


def test_integrity_check_reports_missing_journal_entry_after_reopen(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "missing-entry.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    entry = workspace / ".omni" / "restore" / SESSION_ID / "entries" / "1.json"
    entry.unlink()

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    journal = reopened.inspect()

    assert journal.entries == ()
    assert journal.gaps == ()
    assert journal.integrity_issues == (
        BackupIntegrityIssue(1, "missing_journal_entry", run_token),
    )


def test_corrupt_journal_entry_is_reported_as_unreadable_after_reopen(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "corrupt-entry.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    entry = workspace / ".omni" / "restore" / SESSION_ID / "entries" / "1.json"
    entry.write_bytes(b"not valid journal JSON")

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    journal = reopened.inspect()

    assert journal.entries == ()
    assert journal.integrity_issues == (
        BackupIntegrityIssue(1, "unreadable_journal_entry", run_token),
    )


@pytest.mark.parametrize("mismatch_field", ["operation_id", "run_token"])
def test_journal_identity_mismatch_uses_authoritative_store_token(
    workspace: Path,
    mismatch_field: str,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "identity-mismatch.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    entry = workspace / ".omni" / "restore" / SESSION_ID / "entries" / "1.json"
    replacement: object = 2 if mismatch_field == "operation_id" else str(uuid4())
    _rewrite_signed_json(entry, **{mismatch_field: replacement})

    journal = store.inspect()

    assert journal.entries == ()
    assert journal.gaps == ()
    assert journal.integrity_issues == (
        BackupIntegrityIssue(1, "journal_identity_mismatch", run_token),
    )


def test_v1_state_without_entry_uses_unknown_scope_and_upgrades_to_v2(
    workspace: Path,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "legacy-missing.bin"
    target.write_bytes(b"before")
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    root = workspace / ".omni" / "restore" / SESSION_ID
    (root / "entries" / "1.json").unlink()
    _write_signed_json(
        root / "state.json",
        {
            "schema_version": 1,
            "next_operation_id": 2,
            "revision": 1,
            "journal_operation_ids": [1],
            "discarded_operation_ids": [],
        },
    )

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    journal = reopened.inspect()
    state = json.loads((root / "state.json").read_bytes())

    assert journal.integrity_issues == (
        BackupIntegrityIssue(1, "missing_journal_entry", None),
    )
    assert state["schema_version"] == 2
    assert state["active_operations"] == [{"operation_id": 1, "run_token": None}]


def test_v1_state_recovers_entry_token_and_upgrades_to_v2(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "legacy-valid.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    root = workspace / ".omni" / "restore" / SESSION_ID
    _write_signed_json(
        root / "state.json",
        {
            "schema_version": 1,
            "next_operation_id": 2,
            "revision": 1,
            "journal_operation_ids": [1],
        },
    )

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    journal = reopened.inspect()
    state = json.loads((root / "state.json").read_bytes())

    assert journal.integrity_issues == ()
    assert journal.entries[0].run_token == run_token
    assert state["schema_version"] == 2
    assert state["active_operations"] == [
        {"operation_id": 1, "run_token": str(run_token)}
    ]


@pytest.mark.parametrize(
    "state_fields",
    [
        pytest.param(
            {
                "active_operations": [
                    {
                        "operation_id": 1,
                        "run_token": "12345678-1234-4234-8234-123456789abc",
                    },
                    {
                        "operation_id": 1,
                        "run_token": "22345678-1234-4234-8234-123456789abc",
                    },
                ],
                "discarded_operation_ids": [],
                "next_operation_id": 3,
            },
            id="duplicate-operation",
        ),
        pytest.param(
            {
                "active_operations": [
                    {
                        "operation_id": 2,
                        "run_token": "22345678-1234-4234-8234-123456789abc",
                    },
                    {
                        "operation_id": 1,
                        "run_token": "12345678-1234-4234-8234-123456789abc",
                    },
                ],
                "discarded_operation_ids": [],
                "next_operation_id": 3,
            },
            id="unordered-operations",
        ),
        pytest.param(
            {
                "active_operations": [{"operation_id": 1, "run_token": "not-a-uuid"}],
                "discarded_operation_ids": [],
                "next_operation_id": 2,
            },
            id="invalid-token",
        ),
        pytest.param(
            {
                "active_operations": [
                    {
                        "operation_id": 1,
                        "run_token": "12345678-1234-4234-8234-123456789abc",
                    }
                ],
                "discarded_operation_ids": [1],
                "next_operation_id": 2,
            },
            id="active-discarded-overlap",
        ),
    ],
)
def test_state_v2_rejects_invalid_identity_metadata(state_fields: dict[str, object]) -> None:
    value = {"schema_version": 2, "revision": 1, **state_fields}

    with pytest.raises(BackupStoreError):
        backup_store_module._decode_state(_signed_json_bytes(value))


def test_backup_can_be_reopened_and_read_in_a_new_process(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "restart.bin"
    original = b"read after restart\x00\xff"
    target.write_bytes(original)
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    assert reopened.read_backup(ticket.operation_id) == original
    continued = reopened.before_write(uuid4(), target)
    assert continued is not None
    assert continued.operation_id == ticket.operation_id + 1
    script = (
        "import hashlib, sys; from pathlib import Path; "
        "from omni.agent.session.backup_store import FileBackupStore; "
        "from omni.agent.workspace_state import WorkspaceState; "
        "store = FileBackupStore(WorkspaceState(Path(sys.argv[1])), sys.argv[2]); "
        "data = store.read_backup(int(sys.argv[3])); "
        "print(len(data), hashlib.sha256(data).hexdigest())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(workspace), SESSION_ID, str(ticket.operation_id)],
        cwd=Path.cwd(),
        capture_output=True,
        check=True,
        text=True,
    )
    assert result.stdout.strip() == f"{len(original)} {hashlib.sha256(original).hexdigest()}"


def test_incomplete_post_write_state_is_reported_as_gap_after_reopen(
    workspace: Path,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "incomplete-post-write.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None

    journal = FileBackupStore(WorkspaceState(workspace), SESSION_ID).inspect()

    assert journal.entries[0].after is None
    assert journal.gaps == (
        BackupGap(
            operation_id=ticket.operation_id,
            revision=journal.entries[0].revision,
            run_token=run_token,
            requested_target=str(target.absolute()),
            canonical_target=str(target.resolve()),
            reason="post_write_state_unavailable",
        ),
    )


def test_post_write_observation_failure_is_reported_as_gap_without_tool_effect(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "post-write-observation-failure.bin"
    target.write_bytes(b"before")
    run_token = uuid4()
    ticket = store.before_write(run_token, target)
    assert ticket is not None
    target.write_bytes(b"tool result")
    capture = backup_store_module._capture_target

    def fail_observation(path: Path) -> bytes | None:
        if Path(path) == ticket.canonical_target:
            raise OSError("injected post-write observation failure")
        return capture(path)

    monkeypatch.setattr(backup_store_module, "_capture_target", fail_observation)

    store.after_write(ticket)
    journal = FileBackupStore(WorkspaceState(workspace), SESSION_ID).inspect()

    assert target.read_bytes() == b"tool result"
    assert [gap.reason for gap in journal.gaps] == ["post_write_state_unavailable"]


def test_post_write_replace_failure_is_reported_as_gap_after_reopen(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "post-write-replace-failure.bin"
    target.write_bytes(b"before")
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    entry_path = workspace / ".omni" / "restore" / SESSION_ID / "entries" / "1.json"
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_entry_replace(path: Path, content: bytes) -> None:
        if Path(path) == entry_path:
            raise OSError("injected post-write entry replace failure")
        replace(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_entry_replace)
    target.write_bytes(b"tool result")

    store.after_write(ticket)
    journal = FileBackupStore(WorkspaceState(workspace), SESSION_ID).inspect()

    assert target.read_bytes() == b"tool result"
    assert journal.entries[0].after is None
    assert [gap.reason for gap in journal.gaps] == ["post_write_state_unavailable"]


def test_backup_store_has_no_small_file_size_limit(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "large.bin"
    content = bytes(range(256)) * 32768
    target.write_bytes(content)

    ticket = store.before_write(uuid4(), target)

    assert ticket is not None
    assert store.read_backup(ticket.operation_id) == content
