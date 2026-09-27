import asyncio
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.management.service import ManagementError
from myclaw.session.session import Session
from myclaw.workspace.state import WorkspaceState
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service

NOW = datetime(2026, 8, 4, 14, 30, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _session(state: WorkspaceState) -> Session:
    session = Session.create(state, now=lambda: NOW, new_uuid=lambda: SESSION_UUID)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Owned Session history.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    return session


def _create_directory_alias(alias: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
            check=True,
            capture_output=True,
            text=True,
        )
        return
    alias.symlink_to(target, target_is_directory=True)


def _create_file_alias(alias: Path, target: Path, kind: str) -> None:
    if kind == "hardlink":
        alias.hardlink_to(target)
        return
    try:
        alias.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symbolic links are unavailable: {error}")


def _redirect_directory(state: WorkspaceState, workspace: Path, boundary: str) -> Path:
    if boundary == "workspace_state":
        owned = state.path
        outside = workspace.parent / "redirected-workspace-state"
    else:
        owned = state.sessions_directory
        outside = workspace.parent / "redirected-sessions"
    owned.rename(outside)
    _create_directory_alias(owned, outside)
    return outside


@pytest.mark.parametrize("boundary", ("workspace_state", "sessions"))
def test_session_load_rejects_redirected_owned_directories(
    agent_home: Path,
    workspace: Path,
    boundary: str,
) -> None:
    state = _state(workspace, agent_home)
    session = _session(state)
    session.close()
    _redirect_directory(state, workspace, boundary)

    with pytest.raises(PermissionError):
        Session.load(state, session.session_id)


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_session_load_rejects_linked_history_files(
    agent_home: Path,
    workspace: Path,
    kind: str,
) -> None:
    state = _state(workspace, agent_home)
    session = _session(state)
    session.close()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    outside = workspace.parent / f"outside-{kind}.jsonl"
    path.rename(outside)
    _create_file_alias(path, outside, kind)

    with pytest.raises(PermissionError):
        Session.load(state, session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("workspace_state", "sessions"))
async def test_session_persist_does_not_follow_redirected_owned_directories(
    agent_home: Path,
    workspace: Path,
    boundary: str,
) -> None:
    state = _state(workspace, agent_home)
    session = _session(state)
    outside = _redirect_directory(state, workspace, boundary)

    session.persist()
    await asyncio.sleep(0)

    relative = (
        Path("sessions") / f"{session.session_id}.jsonl"
        if boundary == "workspace_state"
        else Path(f"{session.session_id}.jsonl")
    )
    assert not (outside / relative).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
async def test_session_persist_preserves_linked_history_files(
    agent_home: Path,
    workspace: Path,
    kind: str,
) -> None:
    state = _state(workspace, agent_home)
    session = _session(state)
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    outside = workspace.parent / f"outside-write-{kind}.jsonl"
    outside_bytes = b"outside Session history must remain unchanged\n"
    outside.write_bytes(outside_bytes)
    _create_file_alias(path, outside, kind)

    session.persist()
    await asyncio.sleep(0)

    assert outside.read_bytes() == outside_bytes
    assert path.read_bytes() == outside_bytes
    assert path.samefile(outside)


@pytest.mark.asyncio
async def test_resume_listing_rejects_a_redirected_sessions_directory(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    session = _session(state)
    session.close()
    _redirect_directory(state, workspace, "sessions")

    with pytest.raises(ManagementError) as captured:
        await management_service(home, workspace_state=state).resumable_listing()

    assert captured.value.error.code == "persistence_error"


@pytest.mark.asyncio
async def test_session_persist_normally_replaces_an_owned_regular_file(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = _session(state)

    session.persist()
    await asyncio.sleep(0)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Owned Session history.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            },
            {
                "role": "user",
                "content": "Replacement snapshot.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    session.persist()
    await asyncio.sleep(0)

    assert [message["content"] for message in Session.load(state, session.session_id).messages] == [
        "Owned Session history.",
        "Replacement snapshot.",
    ]
