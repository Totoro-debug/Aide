import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.workspace_state import WorkspaceState
from omni.config.agent_home import AgentHome
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service

NOW = datetime(2026, 8, 1, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
FIRST_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SECOND_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _session(state: WorkspaceState, session_uuid: UUID, title: str) -> Session:
    session = Session.create(state, now=lambda: NOW, new_uuid=lambda: session_uuid)
    session.update_metadata(title=title)
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": f"History for {title}.",
                "timestamp": NOW.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": title,
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


@pytest.mark.asyncio
async def test_resume_listing_returns_current_workspace_sessions_in_update_order(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    older = _session(state, FIRST_UUID, "Older session")
    newer = _session(state, SECOND_UUID, "Newer session")
    older.close()
    newer.update_metadata(title="Newest session")
    newer.close()
    service = management_service(home, workspace_state=state)

    listing = await service.resumable_listing()

    assert [item.id for item in listing.sessions] == [newer.session_id, older.session_id]
    assert [item.title for item in listing.sessions] == ["Newest session", "Older session"]
    assert [item.message_count for item in listing.sessions] == [1, 1]


@pytest.mark.asyncio
async def test_resume_listing_excludes_schedule_session_partition(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    schedule = Session.create(
        state,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=str(FIRST_UUID),
        now=lambda: NOW,
    )
    seed_session_state(
        schedule,
        messages=[
            {
                "role": "user",
                "content": "Background work",
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
    schedule.close()

    listing = await management_service(home, workspace_state=state).resumable_listing()

    assert listing.sessions == ()
    assert listing.skipped_count == 0
    assert Session.load(state, schedule.session_id).messages == schedule.messages


@pytest.mark.asyncio
async def test_resume_listing_skips_corrupt_entries_without_mutating_them(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    valid = _session(state, FIRST_UUID, "Valid session")
    valid.close()
    corrupt_path = state.sessions_directory / (
        "20260801-120000-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e.jsonl"
    )
    corrupt_path.write_text("not-json\n", encoding="utf-8")
    before = corrupt_path.read_bytes()

    listing = await management_service(home, workspace_state=state).resumable_listing()

    assert listing.skipped_count == 1
    assert [item.id for item in listing.sessions] == [valid.session_id]
    assert corrupt_path.read_bytes() == before


@pytest.mark.asyncio
async def test_resume_listing_skips_a_session_with_malformed_field_types(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    valid = _session(state, FIRST_UUID, "Valid session")
    valid.close()
    corrupt_id = "20260801-120000-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e"
    corrupt_path = state.sessions_directory / f"{corrupt_id}.jsonl"
    corrupt_path.write_text(
        json.dumps(
            {
                "session_id": corrupt_id,
                "created_at": "2026-08-01T12:00:00.123+08:00",
                "updated_at": "2026-08-01T12:00:00.123+08:00",
                "last_compacted": 0,
                "metadata": {
                    "title": "Corrupt session",
                    "token_usage": "not-an-object",
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    listing = await management_service(home, workspace_state=state).resumable_listing()

    assert listing.skipped_count == 1
    assert [item.id for item in listing.sessions] == [valid.session_id]


@pytest.mark.asyncio
async def test_resume_selects_the_loaded_session_for_the_agent_loop_owner(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    target = _session(state, SECOND_UUID, "Target session")
    target.close()
    selected: list[tuple[str, bool]] = []

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        selected.append((session_id, force))

    service = management_service(
        home,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
    )

    result = await service.resume(target.session_id)

    assert result.session_id == target.session_id
    assert selected == [(target.session_id, False)]
