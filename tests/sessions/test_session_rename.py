import asyncio
import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aide.agent.session.session import Session
from aide.agent.workspace_state import WorkspaceState
from aide.provider.session_configuration import SessionModelConfiguration

CREATED_AT = datetime(2026, 10, 1, tzinfo=UTC)
RENAMED_AT = CREATED_AT + timedelta(seconds=10)


async def _rename_session(session: Session, title: str, *, expected_metadata_version: int) -> None:
    await session.wait_for_pending_persist()
    session.rename_durably(title, expected_metadata_version=expected_metadata_version)


async def _persisted_session(workspace: Path) -> Session:
    session = Session.create(WorkspaceState(workspace), now=lambda: CREATED_AT)
    session.commit_agent_run(
        [{"role": "user", "content": "Persisted input"}],
        pending_last_compacted=0,
        pending_action_summary=None,
    )
    await session.wait_for_pending_persist()
    return session


def _path(session: Session) -> Path:
    return session.workspace_state.sessions_directory / f"{session.session_id}.jsonl"


@pytest.mark.asyncio
async def test_initial_title_version_survives_persistence_and_rename(workspace: Path) -> None:
    session = await _persisted_session(workspace)
    raw = _path(session).read_bytes()
    header = json.loads(raw.splitlines()[0])
    assert "_title_version" not in header["metadata"]
    assert "_title_source" not in header["metadata"]

    loaded = Session.load(session.workspace_state, session.session_id)
    assert loaded.metadata_version == 0
    assert not loaded.has_manual_title
    assert _path(session).read_bytes() == raw
    await _rename_session(loaded, "Manual title", expected_metadata_version=0)

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Manual title"
    assert reloaded.metadata_version == 1
    assert reloaded.has_manual_title
    assert reloaded.messages == session.messages


@pytest.mark.asyncio
async def test_failed_strict_rename_preserves_disk_and_memory_and_can_retry(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _persisted_session(workspace)
    before_metadata = copy.deepcopy(session.metadata)
    before_messages = copy.deepcopy(session.messages)
    before_updated_at = session.updated_at
    before_raw = _path(session).read_bytes()
    monkeypatch.setattr(session, "_now", lambda: RENAMED_AT)
    write_content = session._write_content

    def fail_write(content: bytes) -> None:
        raise OSError("Controlled write failure")

    monkeypatch.setattr(session, "_write_content", fail_write)
    with pytest.raises(OSError, match="Controlled write failure"):
        await _rename_session(session, "Must not publish", expected_metadata_version=0)

    assert session.metadata == before_metadata
    assert session.messages == before_messages
    assert session.updated_at == before_updated_at
    assert session.metadata_version == 0
    assert not session.has_manual_title
    assert _path(session).read_bytes() == before_raw
    monkeypatch.setattr(session, "_write_content", write_content)
    await _rename_session(session, "Successful retry", expected_metadata_version=0)
    assert session.updated_at == RENAMED_AT
    assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
        "Successful retry"
    )
    assert session.metadata_version == 1


@pytest.mark.asyncio
async def test_rename_drains_queued_old_snapshot_before_publishing_manual_title(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _persisted_session(workspace)
    old_write_started = asyncio.Event()
    release_old_write = asyncio.Event()
    rename_wait_started = asyncio.Event()
    persist_after = session._persist_after
    wait_for_pending = session.wait_for_pending_persist

    async def blocked_persist(previous: asyncio.Task[None] | None, content: bytes) -> None:
        old_write_started.set()
        await release_old_write.wait()
        await persist_after(previous, content)

    async def observe_wait() -> None:
        rename_wait_started.set()
        await wait_for_pending()

    monkeypatch.setattr(session, "_persist_after", blocked_persist)
    session.persist()
    await old_write_started.wait()
    monkeypatch.setattr(session, "wait_for_pending_persist", observe_wait)
    rename = asyncio.create_task(_rename_session(session, "Manual winner", expected_metadata_version=0))
    try:
        await rename_wait_started.wait()
        assert not rename.done()
        assert session.metadata["title"] == "Untitled session"
    finally:
        release_old_write.set()
        await rename

    loaded = Session.load(session.workspace_state, session.session_id)
    assert loaded.metadata["title"] == "Manual winner"
    assert loaded.metadata_version == 1
    assert loaded.has_manual_title
    assert loaded.messages == session.messages


@pytest.mark.asyncio
async def test_concurrent_renames_with_same_version_commit_exactly_once(workspace: Path) -> None:
    session = await _persisted_session(workspace)
    results = await asyncio.gather(
        _rename_session(session, "First candidate", expected_metadata_version=0),
        _rename_session(session, "Second candidate", expected_metadata_version=0),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, BaseException)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "stale" in str(errors[0])
    assert session.metadata_version == 1
    loaded = Session.load(session.workspace_state, session.session_id)
    assert loaded.metadata == session.metadata
    assert loaded.metadata["title"] in {"First candidate", "Second candidate"}


@pytest.mark.asyncio
async def test_automatic_title_invalidates_old_rename_version(workspace: Path) -> None:
    session = await _persisted_session(workspace)
    session.update_automatic_title("Automatic title")
    session.persist()
    await session.wait_for_pending_persist()
    before = _path(session).read_bytes()

    assert session.metadata_version == 1
    with pytest.raises(ValueError, match="stale"):
        await _rename_session(session, "Stale manual edit", expected_metadata_version=0)
    assert _path(session).read_bytes() == before
    assert session.metadata["title"] == "Automatic title"
    await _rename_session(session, "Current manual edit", expected_metadata_version=1)
    assert session.metadata_version == 2


@pytest.mark.asyncio
async def test_title_changed_during_snapshot_drain_rejects_waiting_rename(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _persisted_session(workspace)
    rename_wait_started = asyncio.Event()
    release_rename_wait = asyncio.Event()
    wait_for_pending = session.wait_for_pending_persist

    async def blocked_drain() -> None:
        rename_wait_started.set()
        await release_rename_wait.wait()
        await wait_for_pending()

    monkeypatch.setattr(session, "wait_for_pending_persist", blocked_drain)
    rename = asyncio.create_task(_rename_session(session, "Stale edit", expected_metadata_version=0))
    try:
        await rename_wait_started.wait()
        session.update_automatic_title("Title resolved while waiting")
        session.persist()
    finally:
        release_rename_wait.set()
    with pytest.raises(ValueError, match="stale"):
        await rename

    loaded = Session.load(session.workspace_state, session.session_id)
    assert loaded.metadata["title"] == "Title resolved while waiting"
    assert loaded.metadata_version == 1
    assert not loaded.has_manual_title


@pytest.mark.asyncio
async def test_late_automatic_title_preserves_manual_title_after_reload(workspace: Path) -> None:
    session = await _persisted_session(workspace)
    await _rename_session(session, "Untitled session", expected_metadata_version=0)
    loaded = Session.load(session.workspace_state, session.session_id)
    loaded.update_automatic_title("Late automatic title")
    loaded.persist()
    await loaded.wait_for_pending_persist()

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Untitled session"
    assert reloaded.metadata_version == 1
    assert reloaded.has_manual_title


@pytest.mark.asyncio
async def test_rename_empty_draft_never_materializes_history(workspace: Path) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, now=lambda: CREATED_AT)
    before = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError):
        await _rename_session(session, "Must remain a draft", expected_metadata_version=0)

    assert session.metadata == before
    assert session.updated_at == CREATED_AT
    assert session.metadata_version == 0
    assert not state.path.exists()


@pytest.mark.asyncio
async def test_model_configuration_on_empty_draft_persists_with_first_turn(workspace: Path) -> None:
    state = WorkspaceState(workspace)
    session = Session.create(state, now=lambda: CREATED_AT)
    selection = SessionModelConfiguration("provider", "model", "high")

    assert session.configure_model_durably(selection, expected_version=0) == 1
    assert session.model_configuration == selection
    assert session.model_configuration_version == 1
    assert session.updated_at == CREATED_AT
    assert not state.path.exists()

    session.commit_agent_run(
        [{"role": "user", "content": "First turn"}],
        pending_last_compacted=0,
        pending_action_summary=None,
    )
    await session.wait_for_pending_persist()

    persisted = Session.load(state, session.session_id)
    assert persisted.model_configuration == selection
    assert persisted.model_configuration_version == 1
