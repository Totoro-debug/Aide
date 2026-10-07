import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aide.agent.memory.manager import MemoryManager, SummaryClaimError
from aide.agent.memory.records import SummaryEntry
from aide.agent.memory.store import (
    WorkspaceJsonlSummaryStore,
    WorkspaceLongTermMemoryStore,
    WorkspaceSummaryCursorStore,
)
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome

NOW = datetime(2026, 8, 27, 10, 0, tzinfo=timezone(timedelta(hours=8)))


def _state(agent_home: Path) -> WorkspaceState:
    workspace = agent_home.parent / "memory-manager-workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".aide")
    return state


@pytest.mark.asyncio
async def test_manager_owns_persistence_and_caches_the_startup_snapshot(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    expected = state.long_term_memory_path.read_text(encoding="utf-8")

    manager = MemoryManager(state)

    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert manager.long_term_path == state.long_term_memory_path
    assert isinstance(manager._summary_store, WorkspaceJsonlSummaryStore)
    assert isinstance(manager._cursor_store, WorkspaceSummaryCursorStore)
    assert isinstance(manager._long_term_store, WorkspaceLongTermMemoryStore)
    assert (
        len(
            {
                id(manager._summary_store),
                id(manager._cursor_store),
                id(manager._long_term_store),
            }
        )
        == 3
    )


@pytest.mark.asyncio
async def test_manager_appends_and_claims_summaries_with_cursor_preadvance(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A durable preference.", NOW)
    await manager.append_summary("A second durable preference.", NOW)

    first_claim = await manager.claim_summaries(limit=1)
    second_claim = await manager.claim_summaries(limit=1)

    assert first_claim.previous_cursor == 0
    assert first_claim.cursor == 1
    assert first_claim.entries == (
        SummaryEntry(index=1, timestamp=NOW, content="A durable preference."),
    )
    assert second_claim.previous_cursor == 1
    assert second_claim.cursor == 2
    assert second_claim.entries == (
        SummaryEntry(index=2, timestamp=NOW, content="A second durable preference."),
    )
    assert (state.memory_directory / ".cursor").read_bytes() == b"2\n"


@pytest.mark.asyncio
async def test_manager_reads_disk_and_refreshes_snapshot_after_an_edit(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    manager = MemoryManager(state)
    replacement = manager.memory_snapshot().replace(
        "## User Preference\n",
        "## User Preference\n\nPrefers concise reports.\n",
    )

    updated = await manager.edit_long_term(
        old="## User Preference\n",
        new="## User Preference\n\nPrefers concise reports.\n",
    )

    assert updated == replacement
    assert await manager.read_long_term() == replacement
    assert manager.memory_snapshot() == replacement
    assert state.long_term_memory_path.read_bytes() == replacement.encode("utf-8")


@pytest.mark.parametrize(
    "cursor_bytes",
    (b"not-a-cursor\n", b"-1\n", b"1", b"1 \n", b"1\n2\n"),
)
@pytest.mark.asyncio
async def test_manager_rejects_corrupt_canonical_cursor_without_mutation(
    agent_home: Path,
    cursor_bytes: bytes,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    cursor_path = state.memory_directory / ".cursor"
    cursor_path.write_bytes(cursor_bytes)
    manager = MemoryManager(state)

    with pytest.raises(SummaryClaimError) as raised:
        await manager.claim_summaries(limit=10)

    assert raised.value.cursor == 0
    assert raised.value.phase == "read"
    assert cursor_path.read_bytes() == cursor_bytes


@pytest.mark.asyncio
async def test_manager_rejects_external_hard_linked_cursor(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    outside_cursor = agent_home.parent / "outside-cursor"
    outside_cursor.write_bytes(b"999\n")
    cursor_path = state.memory_directory / ".cursor"
    cursor_path.hardlink_to(outside_cursor)
    manager = MemoryManager(state)

    with pytest.raises(SummaryClaimError) as raised:
        await manager.claim_summaries(limit=10)

    assert raised.value.cursor == 0
    assert raised.value.phase == "read"
    assert outside_cursor.read_bytes() == b"999\n"


def test_manager_module_has_no_execution_dependencies() -> None:
    tree = ast.parse(Path("aide/agent/memory/manager.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_names = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported = imported_names | imported_from_names

    assert not any(
        name.startswith(
            (
                "aide.agent.runner",
                "aide.provider",
                "aide.agent.tools",
                "aide.schedule",
            )
        )
        for name in imported
    )
