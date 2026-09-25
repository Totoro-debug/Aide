from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.session import backup_store as backup_store_module
from myclaw.agent.session.backup_store import (
    BackupIntegrityError,
    BackupIntegrityIssue,
    BackupStoreError,
    FileBackupStore,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

SESSION_ID = "20260926-120000-123456_12345678-1234-4234-8234-123456789abc"


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
    protected = workspace / ".myclaw" / "restore" / "other-session" / "protected.txt"
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
    first_entry = store.inspect().entries[0]
    assert first_entry.before.blob_name is not None
    blob = workspace / ".myclaw" / "restore" / SESSION_ID / "blobs" / first_entry.before.blob_name
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
    monkeypatch.setattr(backup_store_module, "_sync_created_directory", synced.append)
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)

    store.before_write(uuid4(), workspace.parent / "new-target.bin")

    restore_root = workspace / ".myclaw" / "restore"
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

    monkeypatch.setattr(backup_store_module, "_sync_created_directory", fail_restore_once)
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "sync-retry.bin"

    first_ticket = store.before_write(uuid4(), target)
    second_ticket = store.before_write(uuid4(), target)

    assert first_ticket is None
    assert second_ticket is not None
    assert second_ticket.recorded is True
    restore_root = (workspace / ".myclaw" / "restore").resolve()
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
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    entry = store.inspect().entries[0]
    assert entry.before.blob_name is not None
    blob = workspace / ".myclaw" / "restore" / SESSION_ID / "blobs" / entry.before.blob_name
    if damage == "missing":
        blob.unlink()
        reason = "missing_or_unsafe_blob"
    else:
        blob.write_bytes(b"corrupted backup")
        reason = "hash_mismatch"

    assert store.verify_integrity() == (BackupIntegrityIssue(1, reason),)
    with pytest.raises(BackupIntegrityError):
        store.read_backup(ticket.operation_id)
    assert b"sensitive backup bytes" not in repr(store.inspect()).encode("utf-8")
    assert (
        b"sensitive backup bytes"
        not in (workspace / ".myclaw" / "restore" / SESSION_ID / "entries" / "1.json").read_bytes()
    )


def test_integrity_check_reports_missing_journal_entry_after_reopen(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "missing-entry.bin"
    target.write_bytes(b"before")
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    entry = workspace / ".myclaw" / "restore" / SESSION_ID / "entries" / "1.json"
    entry.unlink()

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    journal = reopened.inspect()

    assert journal.entries == ()
    assert journal.gaps == ()
    assert journal.integrity_issues == (BackupIntegrityIssue(1, "missing_journal_entry"),)


def test_corrupt_journal_entry_fails_closed_after_reopen(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "corrupt-entry.bin"
    target.write_bytes(b"before")
    ticket = store.before_write(uuid4(), target)
    assert ticket is not None
    entry = workspace / ".myclaw" / "restore" / SESSION_ID / "entries" / "1.json"
    entry.write_bytes(b"not valid journal JSON")

    reopened = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    with pytest.raises(BackupStoreError, match="invalid JSON"):
        reopened.inspect()


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
        "from myclaw.agent.session.backup_store import FileBackupStore; "
        "from myclaw.agent.workspace_state import WorkspaceState; "
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


def test_backup_store_has_no_small_file_size_limit(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "large.bin"
    content = bytes(range(256)) * 32768
    target.write_bytes(content)

    ticket = store.before_write(uuid4(), target)

    assert ticket is not None
    assert store.read_backup(ticket.operation_id) == content


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not available on Windows")
def test_restore_store_directories_are_private_on_posix(workspace: Path) -> None:
    store = FileBackupStore(WorkspaceState(workspace), SESSION_ID)
    target = workspace.parent / "private.bin"
    target.write_bytes(b"private")
    store.before_write(uuid4(), target)
    root = workspace / ".myclaw" / "restore" / SESSION_ID

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "entries").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "blobs").stat().st_mode) == 0o700
