from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

import aide.agent.context.builder as context
from aide.agent.blackboard import Blackboard
from aide.agent.context.builder import ContextBuilder
from aide.agent.memory.manager import MemoryManager
from aide.agent.tools.core.exec_host import resolve_exec_shell
from aide.agent.tools.permission import PermissionSnapshot
from aide.agent.workspace_state import WorkspaceState
from aide.skills.catalog import (
    ManualSkillInvocation,
    SkillLoader,
    SkillMetadata,
)

FIXED_UTC = datetime(2026, 8, 16, 4, 5, 6, 789000, tzinfo=UTC)


def test_foreground_runtime_context_projects_the_permission_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    snapshot = PermissionSnapshot("read-only", resolve_exec_shell("auto"))

    messages = builder.build_foreground_messages(
        [{"role": "user", "content": "Current question."}],
        session_id="session-id",
        permission_snapshot=snapshot,
    )

    content = messages[-1]["content"]
    assert isinstance(content, str)
    assert "- Tool Permission Level: read-only" in content
    assert f"- Exec Shell: {snapshot.exec_shell.family}" in content
    assert "- Tool Confirmation: Permission checks may require one confirmation" in content


def test_schedule_runtime_context_projects_the_permission_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    snapshot = PermissionSnapshot("workspace-write", resolve_exec_shell("auto"))

    messages = builder.build_schedule_messages(
        [{"role": "user", "content": "Scheduled question."}],
        session_id="schedule-session",
        permission_snapshot=snapshot,
    )

    content = messages[-1]["content"]
    assert isinstance(content, str)
    assert "- Tool Permission Level: workspace-write" in content
    assert f"- Exec Shell: {snapshot.exec_shell.family}" in content
    assert "- Tool Confirmation: Permission checks may require one confirmation" in content


@pytest.mark.parametrize("schedule_run", [False, True], ids=["foreground", "schedule"])
def test_system_prompt_describes_run_local_tool_permission_contract(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    schedule_run: bool,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    messages = [{"role": "user", "content": "Question."}]
    rendered = (
        builder.build_schedule_messages(messages, session_id="schedule-session")
        if schedule_run
        else builder.build_foreground_messages(messages, session_id="foreground-session")
    )
    assert rendered[0]["role"] == "system"
    content = rendered[0]["content"]
    assert isinstance(content, str)

    assert "本次 Run 捕获的不可变 Tool Permission Level 快照及 Tool Gateway 约束" in content
    for tool in ("File", "Exec", "Web Fetch", "MCP", "Schedule Tool"):
        assert tool in content
    assert "私有或非全局 Web Fetch 目标" in content
    assert "灾难性或不确定的 Exec 确认仍然有效" in content
    assert "能力缺失、业务拒绝、Tool 执行错误" in content
    assert "配置的 MCP Tool 按用户配置授予的信任执行" not in content
    assert "只取消前台 File Tool 的普通权限确认" not in content
    assert "`/permission` 只影响后续前台 Run 的快照" in content
    assert "不改变已开始的 Run 或 User Schedule Agent Run 的启动配置权限" in content
    if schedule_run:
        assert "Schedule Run Boundary" in content
        assert "`schedule` Tool is unavailable" in content


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> _FrozenDateTime:
        if tz is None:
            return cls.fromtimestamp(FIXED_UTC.timestamp(), tz=UTC)
        return cls.fromtimestamp(FIXED_UTC.timestamp(), tz=tz)  # type: ignore[arg-type]


def _context_dependencies(
    workspace: Path,
    agent_home: Path,
) -> tuple[MemoryManager, SkillLoader]:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    return MemoryManager(state), loader


def _builder(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    timezone_name: str,
) -> ContextBuilder:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    agent_home = workspace.parent / "agent-home"
    memory_manager, loader = _context_dependencies(workspace, agent_home)
    return context.ContextBuilder(
        workspace,
        timezone_name,
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )


def test_context_builder_rejects_an_invalid_iana_timezone_before_building(
    workspace: Path,
) -> None:
    agent_home = workspace.parent / "agent-home"
    memory_manager, loader = _context_dependencies(workspace, agent_home)
    with pytest.raises(ZoneInfoNotFoundError):
        ContextBuilder(
            workspace,
            "Mars/Olympus",
            agent_home=agent_home,
            memory_manager=memory_manager,
            skill_loader=loader,
        )


@pytest.mark.parametrize(
    ("timezone_name", "expected_time"),
    [("UTC", "2026-08-16T04:05:06.789+00:00"), ("Asia/Shanghai", "2026-08-16T12:05:06.789+08:00")],
)
def test_context_builder_builds_system_history_and_current_user_in_order(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    timezone_name: str,
    expected_time: str,
) -> None:
    monkeypatch.setattr("aide.agent.context.builder.platform.system", lambda: "Windows")
    monkeypatch.setattr("aide.agent.context.builder.platform.machine", lambda: "AMD64")
    monkeypatch.setattr(
        "aide.agent.context.builder.platform.python_version",
        lambda: "3.12.13",
    )
    agent_home = workspace.parent / "agent-home"
    builder = _builder(monkeypatch, workspace, timezone_name)
    history: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Earlier question.",
            "timestamp": "2026-08-15T12:00:00.000+08:00",
        },
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "timestamp": "2026-08-15T12:00:01.000+08:00",
            "tool_calls": [],
            "status": "completed",
        },
    ]
    current_user: dict[str, Any] = {
        "role": "user",
        "content": "Current question.",
        "timestamp": "2026-08-16T12:00:00.000+08:00",
    }

    messages = builder.build_foreground_messages(
        [*history, current_user],
        session_id="20260816-120000-000000_550e8400-e29b-41d4-a716-446655440000",
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    system_prompt = messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert f"`{workspace}`" in system_prompt
    assert f"`{agent_home}`" in system_prompt
    assert "Windows AMD64, Python 3.12.13" in system_prompt
    assert "# Long-term Memory" in system_prompt
    assert messages[1] == {"role": "user", "content": "Earlier question."}
    assert messages[2] == {
        "role": "assistant",
        "content": "Earlier answer.",
        "tool_calls": [],
    }
    assert messages[3] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            f"- Current time: {expected_time}\n"
            "- Session ID: 20260816-120000-000000_550e8400-e29b-41d4-a716-446655440000\n\n"
            "## User Input\n\n"
            "Current question."
        ),
    }
    assert all("timestamp" not in message for message in messages)


def test_context_builder_builds_foreground_request_from_one_ordered_input(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier question.", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "tool_calls": [],
            "status": "completed",
            "timestamp": "old",
        },
        {"role": "user", "content": "Current question.", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Current tool call.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "timestamp": "old",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "Tool output.",
            "status": "success",
            "artifact": None,
            "timestamp": "old",
        },
    ]
    original_messages = deepcopy(messages)

    projected = builder.build_foreground_messages(messages, session_id="session-id")

    assert [message["role"] for message in projected] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert projected[1:3] == [
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer.", "tool_calls": []},
    ]
    assert projected[3] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            "- Current time: 2026-08-16T04:05:06.789+00:00\n"
            "- Session ID: session-id\n\n"
            "## User Input\n\n"
            "Current question."
        ),
    }
    assert projected[4:] == [
        {
            "role": "assistant",
            "content": "Current tool call.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "Tool output.",
        },
    ]
    assert messages == original_messages


def test_context_builder_builds_schedule_request_from_one_ordered_input(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "Asia/Shanghai")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier question.", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "tool_calls": [],
            "status": "completed",
            "timestamp": "old",
        },
        {
            "role": "user",
            "content": "Current scheduled question.",
            "timestamp": "1999-01-01T00:00:00.000+00:00",
        },
        {
            "role": "assistant",
            "content": "Current tool call.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "timestamp": "old",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "Tool output.",
            "status": "success",
            "artifact": None,
            "timestamp": "old",
        },
    ]
    original_messages = deepcopy(messages)

    projected = builder.build_schedule_messages(messages, session_id="schedule-session")

    assert [message["role"] for message in projected] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert projected[0] == {
        "role": "system",
        "content": builder.schedule_system_prompt(),
    }
    assert projected[1:3] == [
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Earlier answer.", "tool_calls": []},
    ]
    assert projected[3] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            "- Current time: 2026-08-16T12:05:06.789+08:00\n"
            "- Session ID: schedule-session\n\n"
            "## User Input\n\n"
            "Current scheduled question."
        ),
    }
    assert projected[4:] == [
        {
            "role": "assistant",
            "content": "Current tool call.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "Tool output.",
        },
    ]
    assert messages == original_messages


@pytest.mark.asyncio
async def test_schedule_projection_scope_is_task_local_and_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "Asia/Shanghai")
    tick = 0

    class AdvancingDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> AdvancingDateTime:
            nonlocal tick
            current = FIXED_UTC + timedelta(milliseconds=tick)
            tick += 1
            if tz is None:
                return cls.fromtimestamp(current.timestamp(), tz=UTC)
            return cls.fromtimestamp(current.timestamp(), tz=tz)  # type: ignore[arg-type]

    monkeypatch.setattr(context, "datetime", AdvancingDateTime)

    def runtime_time(projected: Sequence[dict[str, Any]]) -> str:
        content = projected[-1]["content"]
        assert isinstance(content, str)
        current_line = next(
            line for line in content.splitlines() if line.startswith("- Current time: ")
        )
        return current_line.removeprefix("- Current time: ")

    async def project_twice(
        started: asyncio.Event,
        peer_started: asyncio.Event,
    ) -> tuple[str, str]:
        with builder.schedule_projection_scope():
            first = builder.build_schedule_messages(
                [{"role": "user", "content": "Scheduled input."}],
                session_id="schedule-session",
            )
            started.set()
            await peer_started.wait()
            second = builder.build_schedule_messages(
                [{"role": "user", "content": "Scheduled input."}],
                session_id="schedule-session",
            )
        return runtime_time(first), runtime_time(second)

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_task, second_task = await asyncio.gather(
        project_twice(first_started, second_started),
        project_twice(second_started, first_started),
    )

    assert first_task[0] == first_task[1]
    assert second_task[0] == second_task[1]
    assert first_task[0] != second_task[0]

    with pytest.raises(RuntimeError, match="scope failure"):
        with builder.schedule_projection_scope():
            failed_projection = builder.build_schedule_messages(
                [{"role": "user", "content": "Scheduled input."}],
                session_id="schedule-session",
            )
            raise RuntimeError("scope failure")
    restored_projection = builder.build_schedule_messages(
        [{"role": "user", "content": "Scheduled input."}],
        session_id="schedule-session",
    )

    assert runtime_time(failed_projection) != runtime_time(restored_projection)


@pytest.mark.parametrize(
    "exit_error",
    [None, ValueError, asyncio.CancelledError],
    ids=["normal", "exception", "cancelled"],
)
def test_foreground_projection_scope_restores_nested_and_published_skills(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    exit_error: type[BaseException] | None,
) -> None:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    memory_manager, loader = _context_dependencies(workspace, agent_home)
    builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    snapshots = []
    projections = []
    for description in ("Snapshot A", "Snapshot B", "Published C"):
        instruction.write_text(
            f"---\nname: planner\ndescription: {description}\n---\nInstructions.\n",
            encoding="utf-8",
        )
        loader.load()
        snapshots.append(loader.skills)
        projected = builder.build_status_messages([], session_id="session-id")
        assert description in projected[0]["content"]
        projections.append(projected)

    with builder.foreground_projection_scope(snapshots[0]):
        assert builder.build_status_messages([], session_id="session-id") == projections[0]
        with (
            pytest.raises(exit_error, match="inner scope exit")
            if exit_error is not None
            else nullcontext()
        ):
            with builder.foreground_projection_scope(snapshots[1]):
                assert builder.build_status_messages([], session_id="session-id") == projections[1]
                if exit_error is not None:
                    raise exit_error("inner scope exit")

        assert builder.build_status_messages([], session_id="session-id") == projections[0]
        assert loader.skills == snapshots[2]

    assert builder.build_status_messages([], session_id="session-id") == projections[2]


def test_foreground_projection_scope_isolates_different_context_builders(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    memory_manager, loader = _context_dependencies(workspace, agent_home)
    first_builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )
    second_builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Scoped skills\n---\nInstructions.\n",
        encoding="utf-8",
    )
    loader.load()
    scoped_skills = loader.skills
    scoped_projection = first_builder.build_status_messages([], session_id="session-id")
    instruction.write_text(
        "---\nname: planner\ndescription: Published skills\n---\nInstructions.\n",
        encoding="utf-8",
    )
    loader.load()
    published_projection = second_builder.build_status_messages([], session_id="session-id")
    assert scoped_projection != published_projection

    with first_builder.foreground_projection_scope(scoped_skills):
        assert first_builder.build_status_messages([], session_id="session-id") == scoped_projection
        assert (
            second_builder.build_status_messages([], session_id="session-id")
            == published_projection
        )

    assert first_builder.build_status_messages([], session_id="session-id") == published_projection


def test_context_builder_advertises_catalog_metadata_in_foreground_system_prompt(
    workspace: Path,
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan the work\n---\nprivate body\n")
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        memory_manager=MemoryManager(state),
        skill_loader=loader,
    )

    messages = builder.build_foreground_messages(
        [{"role": "user", "content": "Plan this."}],
        session_id="session-id",
    )

    system_prompt = messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert "## Skill Catalog" in system_prompt
    metadata_lines = [line for line in system_prompt.splitlines() if line.startswith("{")]
    assert [json.loads(line) for line in metadata_lines] == [
        {
            "name": "planner",
            "description": "Plan the work",
            "path": str(instruction.resolve()),
        }
    ]
    assert "private body" not in system_prompt


def test_context_builder_projects_tool_and_interrupted_history_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Calling.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
            "status": "completed",
            "timestamp": "old",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "File contents.",
            "timestamp": "old",
        },
        {
            "role": "assistant",
            "content": "Partial.",
            "tool_calls": [],
            "status": "interrupted",
            "timestamp": "old",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "status": "error",
            "timestamp": "old",
        },
    ]
    current_user: dict[str, Any] = {"role": "user", "content": "Now", "timestamp": "old"}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)

    messages = builder.build_foreground_messages(
        [*history, current_user],
        session_id="session-id",
    )

    assert messages[1] == {
        "role": "assistant",
        "content": "Calling.",
        "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read_file",
        "content": "File contents.",
    }
    assert messages[3] == {
        "role": "assistant",
        "content": "Partial.\n\n[Turn interrupted by user.]",
        "tool_calls": [],
    }
    assert len(messages) == 5
    assert history == original_history
    assert current_user == original_current_user

    projected_call = messages[1]["tool_calls"]
    assert isinstance(projected_call, list)
    projected_call[0]["arguments"] = "changed"
    assert history[0]["tool_calls"][0]["arguments"] == '{"path":"a.txt"}'


def test_context_builder_projects_markdown_blackboard_only_into_current_user(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    blackboard = Blackboard(
        goal='Keep "quotes" and <tag> text.',
        completion_boundary="Finish on C:\\tmp\\done.\n完成。",
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Earlier answer",
            "tool_calls": [],
            "status": "completed",
            "timestamp": "old",
        },
    ]
    current_user: dict[str, Any] = {"role": "user", "content": "Current", "timestamp": "old"}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)
    messages = builder.build_foreground_messages(
        [*history, current_user],
        session_id="session-id",
        blackboard=blackboard,
    )

    assert messages[1:] == [
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Earlier answer", "tool_calls": []},
        {
            "role": "user",
            "content": (
                "## Runtime Context\n\n"
                "- Current time: 2026-08-16T04:05:06.789+00:00\n"
                "- Session ID: session-id\n\n"
                "## User Input\n\n"
                "Current\n\n"
                "## Task goal\n\n"
                'Keep "quotes" and <tag> text.\n\n'
                "## Completion boundary\n\n"
                "Finish on C:\\tmp\\done.\n完成。"
            ),
        },
    ]
    assert history == original_history
    assert current_user == original_current_user
    assert blackboard == Blackboard(
        goal='Keep "quotes" and <tag> text.',
        completion_boundary="Finish on C:\\tmp\\done.\n完成。",
    )


def test_context_builder_projects_manual_skill_and_request_as_safe_distinct_blocks(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    body = 'Do "this".\nPath C:\\tmp\\done.\nFence ```json and </skill_instructions> & < >'
    request = 'Need "that".\nFence ``` and </user_request> & < >'
    invocation = ManualSkillInvocation(
        metadata=SkillMetadata(
            name="planner",
            description="Plan work",
            path=workspace / "skill.md",
        ),
        request=request,
        body=body,
    )
    history = [{"role": "user", "content": "Earlier"}]
    current_user = {"role": "user", "content": "/planner  "}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)
    blackboard = Blackboard(goal="Keep the task", completion_boundary="Finish the request")

    messages = builder.build_foreground_messages(
        [*history, current_user],
        session_id="session-id",
        blackboard=blackboard,
        manual_invocation=invocation,
    )

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    system_prompt = messages[0]["content"]
    current_content = messages[-1]["content"]
    assert isinstance(system_prompt, str)
    assert isinstance(current_content, str)
    assert body not in system_prompt
    assert body not in str(messages[1:2])
    assert "/planner" not in current_content
    assert current_content.count("## Skill Instructions") == 1
    assert current_content.count("## User Request") == 1

    skill_block = current_content.split("## Skill Instructions\n\n```json\n", 1)[1].split(
        "\n```", 1
    )[0]
    request_block = current_content.split("## User Request\n\n```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(skill_block) == {"name": "planner", "body": body}
    assert json.loads(request_block) == request
    assert r"\u0060\u0060\u0060" in skill_block
    assert r"\u0060\u0060\u0060" in request_block
    assert current_content.count(body) == 0
    assert current_content.count(request) == 0
    assert "## Task goal" not in current_content
    assert "## Completion boundary" not in current_content
    assert history == original_history
    assert current_user == original_current_user


def test_context_builder_does_not_accept_tool_gateway_or_schemas(workspace: Path) -> None:
    with pytest.raises(TypeError):
        ContextBuilder(
            workspace,
            "UTC",
            agent_home=workspace.parent / "agent-home",
            tool_gateway=object(),  # type: ignore[call-arg]
        )


def test_context_builder_owns_exactly_the_five_context_dependencies(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    memory_manager = MemoryManager(state)

    builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )

    assert vars(builder) == {
        "_workspace": workspace,
        "_agent_home": agent_home,
        "_timezone": ZoneInfo("UTC"),
        "_memory_manager": memory_manager,
        "_skill_loader": loader,
    }


@pytest.mark.asyncio
async def test_context_builder_reads_live_memory_and_lane_specific_skills(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\n"
        "name: planner\n"
        "description: Plan work\n"
        "always: true\n"
        "---\n"
        "Private planner instructions.\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=True,
    )
    loader.load()
    memory_manager = MemoryManager(state)
    builder = ContextBuilder(
        workspace,
        "Asia/Shanghai",
        agent_home=agent_home,
        memory_manager=memory_manager,
        skill_loader=loader,
    )

    first_foreground = builder.foreground_system_prompt()
    first_schedule = builder.schedule_system_prompt()
    await memory_manager.edit_long_term(
        old=memory_manager.memory_snapshot(),
        new="# Long-term Memory\n\n## User Info\n\nLatest memory.",
    )
    second_foreground = builder.foreground_system_prompt()
    second_schedule = builder.schedule_system_prompt()

    assert "Latest memory." in second_foreground
    assert "Latest memory." in second_schedule
    assert "Latest memory." not in first_foreground
    assert "Latest memory." not in first_schedule
    assert '"name":"planner"' in second_foreground
    assert "Private planner instructions." in second_foreground
    assert "planner" not in second_schedule
    assert "Private planner instructions." not in second_schedule
    assert "Schedule Run Boundary" in second_schedule
    assert "`schedule` Tool is unavailable" in second_schedule
    assert "`schedule`: 仅限前台对话" in second_schedule
    assert "<skill_catalog>" not in second_foreground
    assert "<skill_always_load>" not in second_foreground
    assert "```jsonl" in second_foreground


def test_context_builder_schedule_request_excludes_foreground_skill_and_task_state(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    body = "Private foreground instructions with a unique marker."
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        f"---\nname: planner\ndescription: Plan work\nalways: true\n---\n{body}\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=True,
    )
    loader.load()
    builder = ContextBuilder(
        workspace,
        "Asia/Shanghai",
        agent_home=agent_home,
        memory_manager=MemoryManager(state),
        skill_loader=loader,
    )
    messages = [
        {
            "role": "user",
            "content": "Run the scheduled work.",
            "timestamp": "1999-01-01T00:00:00.000+00:00",
        }
    ]
    original_messages = deepcopy(messages)

    projected = builder.build_schedule_messages(messages, session_id="schedule-session")

    system_content = projected[0]["content"]
    current_content = projected[-1]["content"]
    assert isinstance(system_content, str)
    assert isinstance(current_content, str)
    assert "planner" not in system_content
    assert body not in system_content
    assert "## Skill Catalog" not in system_content
    assert "## Skill Instructions" not in current_content
    assert "## Task goal" not in current_content
    assert "## Completion boundary" not in current_content
    assert "<skill_catalog>" not in system_content
    assert "<skill_always_load>" not in system_content
    assert current_content.startswith(
        "## Runtime Context\n\n- Current time: 2026-08-16T12:05:06.789+08:00\n"
    )
    assert messages == original_messages


def test_context_builder_builds_title_and_status_minimal_messages(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    builder = ContextBuilder(
        workspace,
        "Asia/Shanghai",
        agent_home=agent_home,
        memory_manager=MemoryManager(state),
        skill_loader=loader,
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Answer",
            "tool_calls": [],
            "status": "completed",
            "timestamp": "old",
        },
    ]
    original_history = deepcopy(history)

    title_messages = builder.build_title_messages("Title input")
    status_messages = builder.build_status_messages(history, session_id="session-id")

    assert title_messages == [
        {"role": "system", "content": builder.session_title_prompt()},
        {"role": "user", "content": "Title input"},
    ]
    assert [message["role"] for message in status_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert status_messages[0] == {
        "role": "system",
        "content": builder.foreground_system_prompt(),
    }
    assert status_messages[-1] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            "- Current time: 2026-08-16T12:05:06.789+08:00\n"
            "- Session ID: session-id\n\n"
            "## User Input\n\n"
        ),
    }
    assert history == original_history


def test_context_builder_injects_action_summary_after_system_for_each_lane(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier"},
        {
            "role": "assistant",
            "content": "Answer",
            "tool_calls": [],
            "status": "completed",
        },
    ]
    summary = "- Updated the conversation compaction flow."

    foreground = builder.build_foreground_messages(
        [*history, {"role": "user", "content": "Current"}],
        session_id="session-id",
        summary=summary,
    )
    schedule = builder.build_schedule_messages(
        [*history, {"role": "user", "content": "Current"}],
        session_id="session-id",
        summary=summary,
    )
    status = builder.build_status_messages(history, session_id="session-id", summary=summary)

    for projected in (foreground, schedule, status):
        assert projected[1] == {"role": "user", "content": summary}
        assert projected[2]["content"] == "Earlier"

    without_summary = builder.build_foreground_messages(
        history,
        session_id="session-id",
        summary="",
    )
    assert len(without_summary) == 3
    assert without_summary[1]["content"] != summary


def test_context_builder_builds_run_messages_from_compaction_cursor_and_increment() -> None:
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old response"},
    ]
    current_user = {"role": "user", "content": "current request"}
    increment = [
        {"role": "assistant", "content": "compacted response"},
        {"role": "tool", "name": "read_file", "tool_call_id": "tool-1", "content": "result"},
        {"role": "assistant", "content": "latest response"},
    ]
    original = deepcopy((history, current_user, increment))

    projected = ContextBuilder.build_run_messages(
        history,
        current_user=current_user,
        increment=increment,
        compaction_cursor=4,
        action_summary="action summary",
        project_messages=lambda messages: [
            {"role": "system", "content": "run system"},
            *deepcopy(list(messages)),
        ],
    )

    assert projected == [
        {"role": "system", "content": "run system"},
        {"role": "user", "content": "action summary"},
        current_user,
        increment[1],
        increment[2],
    ]
    assert (history, current_user, increment) == original


def test_context_builder_uses_subagent_prompt_snapshot_and_projects_action_summary() -> None:
    conversation = [
        {"role": "user", "content": "child task"},
        {"role": "assistant", "content": "child response", "status": "completed"},
    ]
    original = deepcopy(conversation)

    projected = ContextBuilder.build_run_messages(
        conversation,
        current_user=None,
        increment=(),
        compaction_cursor=0,
        action_summary="child action summary",
        project_messages=lambda messages: ContextBuilder.build_subagent_messages(
            messages, system_prompt="captured creator prompt"
        ),
    )

    assert projected == [
        {"role": "system", "content": "captured creator prompt"},
        {"role": "user", "content": "child action summary"},
        *conversation,
    ]
    assert conversation == original


def test_context_builder_projects_only_selected_tool_results_on_a_detached_copy() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "tool-1",
            "content": "x" * 513,
            "artifact": {"path": "result.txt"},
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "tool-2", "content": "latest"},
    ]
    original = deepcopy(messages)

    projected = ContextBuilder.project_tool_results(messages, omission_indices={1})

    assert projected[1] == {
        **messages[1],
        "content": "[read_file result omitted from context]",
    }
    assert projected[2] == messages[2]
    assert messages == original
    assert projected is not messages
    assert projected[1] is not messages[1]


def test_runtime_lane_projections_keep_current_turn_continuation_separate(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Earlier question.",
            "timestamp": "2026-08-15T12:00:00.000+08:00",
        },
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "timestamp": "2026-08-15T12:00:01.000+08:00",
            "tool_calls": [],
            "status": "completed",
        },
        {
            "role": "user",
            "content": "Current question.",
            "timestamp": "2026-08-16T12:00:00.000+08:00",
        },
        {
            "role": "assistant",
            "content": "Calling a tool.",
            "timestamp": "2026-08-16T12:00:01.000+08:00",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
        },
        {
            "role": "tool",
            "content": "Tool output.",
            "timestamp": "2026-08-16T12:00:02.000+08:00",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": None,
        },
    ]

    foreground = builder.build_foreground_messages(
        messages,
        session_id="session-id",
        blackboard=Blackboard(goal="Current task", completion_boundary="Current boundary"),
    )
    schedule = builder.build_schedule_messages(messages, session_id="session-id")

    assert [message["role"] for message in foreground] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert f"当前工作区 `{workspace}`" in foreground[0]["content"]
    assert foreground[3]["content"].endswith(
        "## User Input\n\nCurrent question.\n\n"
        "## Task goal\n\nCurrent task\n\n"
        "## Completion boundary\n\nCurrent boundary"
    )
    assert [message["role"] for message in schedule] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert schedule[0] == {
        "role": "system",
        "content": builder.schedule_system_prompt(),
    }
    assert schedule[3] == {
        "role": "user",
        "content": (
            "## Runtime Context\n\n"
            "- Current time: 2026-08-16T04:05:06.789+00:00\n"
            "- Session ID: session-id\n\n"
            "## User Input\n\n"
            "Current question."
        ),
    }
