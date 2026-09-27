from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.session.backup_store import FileBackupStore
from myclaw.session.restore import RestoreManager, RestoreMode
from myclaw.session.session import Session
from myclaw.workspace.state import WorkspaceState

RUN_TOKEN = UUID("12345678-1234-4234-8234-123456789abc")


def _commit_user(session: Session) -> None:
    before = session.capture_restore_before()
    session.commit_agent_run(
        [{"role": "user", "content": "restore path matrix"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
        restore_before=before,
        restore_run_token=RUN_TOKEN,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "before", "after"),
    (
        ("existing", b"original", b"tool result"),
        ("new", None, b"created by tool"),
        ("external", b"external original", b"external tool result"),
    ),
    ids=("existing", "new", "external"),
)
async def test_restore_path_matrix_restores_existing_new_and_external_targets(
    workspace: Path,
    case: str,
    before: bytes | None,
    after: bytes,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: RUN_TOKEN)
    target = workspace / f"{case}.txt" if case != "external" else workspace.parent / "external.txt"
    if before is not None:
        target.write_bytes(before)

    _commit_user(session)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(RUN_TOKEN, target)
    assert ticket is not None
    target.write_bytes(after)
    store.after_write(ticket)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failures == ()
    if before is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == before


@pytest.mark.asyncio
async def test_restore_path_matrix_does_not_follow_a_retargeted_link(workspace: Path) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: RUN_TOKEN)
    original = workspace / "original.txt"
    replacement = workspace / "replacement.txt"
    link = workspace / "alias.txt"
    original.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    try:
        link.symlink_to(original)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file symlink privilege unavailable on this host: {error}")

    _commit_user(session)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(RUN_TOKEN, link)
    assert ticket is not None
    original.write_bytes(b"tool result")
    store.after_write(ticket)
    link.unlink()
    link.symlink_to(replacement)
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failures == ()
    assert original.read_bytes() == b"original"
    assert replacement.read_bytes() == b"replacement"
    assert link.resolve() == replacement.resolve()


@pytest.mark.asyncio
async def test_restore_path_matrix_reports_a_file_failure_and_truncates_session(
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, new_uuid=lambda: RUN_TOKEN)
    target = workspace / "hard-linked.txt"
    alias = workspace / "hard-link-alias.txt"
    target.write_bytes(b"original")

    _commit_user(session)
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(RUN_TOKEN, target)
    assert ticket is not None
    target.write_bytes(b"tool result")
    store.after_write(ticket)
    try:
        os.link(target, alias)
    except OSError as error:
        pytest.skip(f"hard-link capability unavailable on this host: {error}")
    await session.wait_for_pending_persist()

    manager = RestoreManager(state, session.session_id)
    result = await manager.execute(manager.inspect(session, 1), RestoreMode.FILES)

    assert result.failed_files == (target.resolve(),)
    assert target.read_bytes() == b"tool result"
    assert alias.read_bytes() == b"tool result"
    assert Session.load(state, session.session_id).messages == []
