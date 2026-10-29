from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aide.agent.tool_concurrency import max_tool_call_concurrency, next_tool_call_batch
from aide.agent.tools.base import ToolResourceProfile
from aide.agent.tools.core.subagents import SpawnAgentTool
from aide.agent.tools.core.tool_search import ToolSearchTool


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


def test_tool_search_and_subagent_tools_have_explicit_access_profiles() -> None:
    assert ToolSearchTool(lambda _query: ()).resource_profile == ToolResourceProfile(
        reads=True,
        writes=False,
    )
    assert SpawnAgentTool.resource_profile == ToolResourceProfile(reads=False, writes=True)


def test_resource_profiles_are_immutable_and_invalid_flags_are_rejected() -> None:
    profile = ToolResourceProfile(reads=True, writes=False)

    with pytest.raises(FrozenInstanceError):
        profile.reads = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be booleans"):
        ToolResourceProfile(reads=1, writes=False)  # type: ignore[arg-type]
