from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aide.agent.tool_concurrency import max_tool_call_concurrency, next_tool_call_batch
from aide.agent.tools.base import ToolResourceProfile
from aide.agent.tools.core._directory import DirectoryReadTool
from aide.agent.tools.core.edit_file import EditFileTool
from aide.agent.tools.core.read_file import ReadFileTool
from aide.agent.tools.core.subagents import SpawnAgentTool
from aide.agent.tools.core.tool_search import ToolSearchTool
from aide.agent.tools.core.write_file import WriteFileTool
from aide.agent.tools.permission import FileAccess, ToolInvocationFacts


def test_concurrency_limit_uses_process_available_cpu_count() -> None:
    assert max_tool_call_concurrency(1) == 10
    assert max_tool_call_concurrency(40) == 10
    assert max_tool_call_concurrency(44) == 11


def test_only_explicit_read_only_tools_share_a_batch() -> None:
    readonly = ToolResourceProfile(reads=True, writes=False)
    writer = ToolResourceProfile(reads=False, writes=True)

    assert next_tool_call_batch((readonly, readonly, writer, readonly), limit=10) == (0, 1)
    assert next_tool_call_batch((None, readonly), limit=10) == (0,)
    assert next_tool_call_batch((writer, readonly), limit=10) == (0,)


def test_read_only_batch_respects_concurrency_limit() -> None:
    readonly = ToolResourceProfile(reads=True, writes=False)

    assert next_tool_call_batch((readonly,) * 12, limit=10) == tuple(range(10))


def test_file_access_batches_allow_disjoint_writes_and_reject_overlapping_paths() -> None:
    reader = ToolResourceProfile(reads=True, writes=False)
    writer = ToolResourceProfile(reads=False, writes=True)
    read_write = ToolResourceProfile(reads=True, writes=True)

    def call(
        profile: ToolResourceProfile,
        *paths: tuple[str, str],
    ) -> tuple[ToolResourceProfile, ToolInvocationFacts]:
        facts = ToolInvocationFacts(
            tool_name="file",
            normalized_arguments={},
            file_accesses=tuple(
                FileAccess(
                    path=Path(path),
                    role=role,  # type: ignore[arg-type]
                    base=Path("C:/workspace"),
                    workspace_root=Path("C:/workspace"),
                )
                for path, role in paths
            ),
        )
        return profile, facts

    assert next_tool_call_batch(
        (
            call(writer, ("C:/workspace/a.txt", "write")),
            call(writer, ("C:/workspace/b.txt", "write")),
        ),
        limit=10,
    ) == (0, 1)
    assert next_tool_call_batch(
        (call(reader, ("C:/workspace/a.txt", "read")), call(writer, ("C:/workspace/a.txt", "write"))),
        limit=10,
    ) == (0,)
    assert next_tool_call_batch(
        (
            call(
                read_write,
                ("C:/workspace/a.txt", "read"),
                ("C:/workspace/a.txt", "write"),
            ),
            call(writer, ("C:/workspace/a.txt", "write")),
        ),
        limit=10,
    ) == (0,)
    assert next_tool_call_batch(
        (call(reader, ("C:/workspace", "read")), call(writer, ("C:/workspace/sub/file.txt", "write"))),
        limit=10,
    ) == (0,)
    assert next_tool_call_batch(
        (
            call(writer, ("C:/workspace/a.txt", "write")),
            call(writer, ("C:/workspace/sub/a.txt", "write")),
        ),
        limit=10,
    ) == (0, 1)


def test_file_paths_are_compared_with_host_case_and_separator_rules() -> None:
    writer = ToolResourceProfile(reads=False, writes=True)

    def call(path: str) -> tuple[ToolResourceProfile, ToolInvocationFacts]:
        return writer, ToolInvocationFacts(
            tool_name="file",
            normalized_arguments={},
            file_accesses=(
                FileAccess(
                    path=Path(path),
                    role="write",
                    base=Path("C:/workspace"),
                    workspace_root=Path("C:/workspace"),
                ),
            ),
        )

    assert next_tool_call_batch(
        (call("C:/Workspace/Dir\\note.txt"), call("c:/workspace/dir/note.txt")), limit=10
    ) == (0,)
    assert next_tool_call_batch(
        (call("C:/workspace/a.txt"), call("C:/workspace/a.txt.old")), limit=10
    ) == (0, 1)


def test_tool_search_and_subagent_tools_have_explicit_access_profiles() -> None:
    assert ToolSearchTool(lambda _query: ()).resource_profile == ToolResourceProfile(
        reads=True,
        writes=False,
    )
    assert SpawnAgentTool.resource_profile == ToolResourceProfile(reads=False, writes=True)


def test_file_tools_declare_resource_profiles() -> None:
    assert ReadFileTool.resource_profile == ToolResourceProfile(reads=True, writes=False)
    assert DirectoryReadTool.resource_profile == ToolResourceProfile(reads=True, writes=False)
    assert WriteFileTool.resource_profile == ToolResourceProfile(reads=False, writes=True)
    assert EditFileTool.resource_profile == ToolResourceProfile(reads=False, writes=True)


def test_resource_profiles_are_immutable_and_invalid_flags_are_rejected() -> None:
    profile = ToolResourceProfile(reads=True, writes=False)

    with pytest.raises(FrozenInstanceError):
        profile.reads = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be booleans"):
        ToolResourceProfile(reads=1, writes=False)  # type: ignore[arg-type]
