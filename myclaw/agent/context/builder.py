"""Build provider-neutral context for Agent Loop model requests."""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any
from zoneinfo import ZoneInfo

from myclaw.agent.blackboard import Blackboard
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.permission import PermissionSnapshot
from myclaw.skills.catalog import LoadedSkill, ManualSkillInvocation, SkillLoader
from myclaw.templates import render_template
from myclaw.utils.time import format_rfc3339_milliseconds

_MARKDOWN_SAFE_JSON_TRANSLATION: dict[int, str] = {
    ord("&"): r"\u0026",
    ord("`"): r"\u0060",
    ord("<"): r"\u003c",
    ord(">"): r"\u003e",
}

_SCHEDULE_RUN_BOUNDARY = (
    "\n\n## Schedule Run Boundary\n"
    "This is a background Schedule Agent Run. The `schedule` Tool is unavailable. "
    "Do not create, list, or remove Schedule Jobs."
)


@dataclass(frozen=True, slots=True)
class _ScheduleProjectionSnapshot:
    builder_id: int
    system_prompt: str
    current_time: datetime


@dataclass(frozen=True, slots=True)
class _ForegroundProjectionSnapshot:
    builder_id: int
    skills: tuple[LoadedSkill, ...]


_SCHEDULE_PROJECTION_SNAPSHOT: ContextVar[_ScheduleProjectionSnapshot | None] = ContextVar(
    "schedule_projection_snapshot",
    default=None,
)
_FOREGROUND_PROJECTION_SNAPSHOT: ContextVar[_ForegroundProjectionSnapshot | None] = ContextVar(
    "foreground_projection_snapshot",
    default=None,
)


class ContextBuilder:
    """Build the complete model-visible message list for one foreground turn."""

    def __init__(
        self,
        workspace: Path,
        timezone_name: str,
        *,
        agent_home: Path,
        memory_manager: MemoryManager,
        skill_loader: SkillLoader,
    ) -> None:
        self._workspace = workspace
        self._agent_home = agent_home
        self._timezone = ZoneInfo(timezone_name)
        self._memory_manager = memory_manager
        self._skill_loader = skill_loader

    def foreground_system_prompt(self) -> str:
        """Build the current foreground System Prompt from owned runtime state."""
        snapshot = _FOREGROUND_PROJECTION_SNAPSHOT.get()
        skills = (
            self._skill_loader.skills
            if snapshot is None or snapshot.builder_id != id(self)
            else snapshot.skills
        )
        return _build_foreground_system_prompt(
            workspace=self._workspace,
            agent_home=self._agent_home,
            long_term_memory=self._memory_manager.memory_snapshot(),
            skills=skills,
        )

    def schedule_system_prompt(self) -> str:
        """Build the Schedule System Prompt without foreground Skill content."""
        return (
            _build_foreground_system_prompt(
                workspace=self._workspace,
                agent_home=self._agent_home,
                long_term_memory=self._memory_manager.memory_snapshot(),
                skills=(),
            )
            + _SCHEDULE_RUN_BOUNDARY
        )

    @contextmanager
    def foreground_projection_scope(self, skills: Sequence[LoadedSkill]) -> Iterator[None]:
        """Keep one foreground request's Skill projection stable across async preparation."""
        token = _FOREGROUND_PROJECTION_SNAPSHOT.set(
            _ForegroundProjectionSnapshot(
                builder_id=id(self),
                skills=tuple(skills),
            )
        )
        try:
            yield
        finally:
            _FOREGROUND_PROJECTION_SNAPSHOT.reset(token)

    def session_title_prompt(self) -> str:
        """Return the isolated versioned prompt used for Session titles."""
        return render_template("session-title-prompt.md")

    def build_title_messages(self, content: str) -> list[dict[str, Any]]:
        """Build the minimal System/User request used for a Session title."""
        if not isinstance(content, str):
            raise TypeError("Context Builder title content must be a string")
        return [
            {"role": "system", "content": self.session_title_prompt()},
            {"role": "user", "content": deepcopy(content)},
        ]

    def build_status_messages(
        self,
        history: Sequence[dict[str, Any]],
        *,
        session_id: str,
        summary: str = "",
        blackboard: Blackboard | None = None,
        permission_snapshot: PermissionSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        """Build the minimum foreground request used by status and preflight."""
        return self.build_foreground_messages(
            [*history, {"role": "user", "content": ""}],
            session_id=session_id,
            blackboard=blackboard,
            summary=summary,
            permission_snapshot=permission_snapshot,
        )

    def build_foreground_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
        blackboard: Blackboard | None = None,
        manual_invocation: ManualSkillInvocation | None = None,
        summary: str = "",
        permission_snapshot: PermissionSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        """Build the canonical initial Model Request Context for a foreground turn."""
        if manual_invocation is not None and not isinstance(
            manual_invocation, ManualSkillInvocation
        ):
            raise TypeError("Context Builder requires a Manual Skill Invocation")
        if blackboard is not None and not isinstance(blackboard, Blackboard):
            raise TypeError("value must be a Blackboard or None")
        return self._build_messages(
            messages,
            system_prompt=self.foreground_system_prompt(),
            session_id=session_id,
            blackboard=blackboard,
            manual_invocation=manual_invocation,
            summary=summary,
            permission_snapshot=permission_snapshot,
        )

    def build_schedule_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
        summary: str = "",
        permission_snapshot: PermissionSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        """Build the canonical initial Model Request Context for a Schedule Job."""
        snapshot = _SCHEDULE_PROJECTION_SNAPSHOT.get()
        if snapshot is None or snapshot.builder_id != id(self):
            system_prompt = self.schedule_system_prompt()
            current_time = None
        else:
            system_prompt = snapshot.system_prompt
            current_time = snapshot.current_time
        return self._build_messages(
            messages,
            system_prompt=system_prompt,
            session_id=session_id,
            current_time=current_time,
            summary=summary,
            permission_snapshot=permission_snapshot,
        )

    @contextmanager
    def schedule_projection_scope(self) -> Iterator[None]:
        """Keep one Schedule request's dynamic projection stable across budgeting."""
        token = _SCHEDULE_PROJECTION_SNAPSHOT.set(
            _ScheduleProjectionSnapshot(
                builder_id=id(self),
                system_prompt=self.schedule_system_prompt(),
                current_time=datetime.now(self._timezone),
            )
        )
        try:
            yield
        finally:
            _SCHEDULE_PROJECTION_SNAPSHOT.reset(token)

    def _build_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system_prompt: str,
        session_id: str,
        blackboard: Blackboard | None = None,
        manual_invocation: ManualSkillInvocation | None = None,
        current_time: datetime | None = None,
        summary: str = "",
        permission_snapshot: PermissionSnapshot | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(summary, str):
            raise TypeError("Context Builder summary must be a string")
        current_user_index = _last_user_index(messages)
        projected = [{"role": "system", "content": system_prompt}]
        if summary:
            projected.append({"role": "user", "content": deepcopy(summary)})
        if current_user_index == len(messages):
            projected.extend(_project_history_messages(messages))
            return projected

        projected.extend(_project_history_messages(messages[:current_user_index]))
        projected.append(
            {
                "role": "user",
                "content": _build_current_user_content(
                    content=deepcopy(messages[current_user_index]["content"]),
                    current_time=(
                        datetime.now(self._timezone) if current_time is None else current_time
                    ),
                    session_id=session_id,
                    blackboard_projection=(
                        None
                        if manual_invocation is not None or blackboard is None
                        else blackboard.to_dict()
                    ),
                    manual_invocation=manual_invocation,
                    permission_snapshot=permission_snapshot,
                ),
            }
        )
        projected.extend(_project_history_messages(messages[current_user_index + 1 :]))
        return projected


def _project_history_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        projected
        for message in messages
        if (projected := _project_history_message(message)) is not None
    ]


def _last_user_index(messages: Sequence[dict[str, Any]]) -> int:
    """Return the final user message index, or the sequence length if absent."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def _project_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Project one validated Session message without durable fields."""
    role = message["role"]
    if role == "user":
        return {"role": "user", "content": deepcopy(message["content"])}

    if role == "assistant":
        content = message["content"]
        projected_tool_calls = [
            {
                "id": deepcopy(tool_call["id"]),
                "name": deepcopy(tool_call["name"]),
                "arguments": deepcopy(tool_call["arguments"]),
            }
            for tool_call in message["tool_calls"]
        ]
        if message["status"] == "error" and not content and not projected_tool_calls:
            return None
        if message["status"] == "interrupted":
            content = f"{content}\n\n[Turn interrupted by user.]"
        return {
            "role": "assistant",
            "content": deepcopy(content),
            "tool_calls": projected_tool_calls,
        }

    return {
        "role": "tool",
        "tool_call_id": deepcopy(message["tool_call_id"]),
        "name": deepcopy(message["name"]),
        "content": deepcopy(message["content"]),
    }


def _build_current_user_content(
    *,
    content: str,
    current_time: datetime,
    session_id: str,
    blackboard_projection: dict[str, str] | None = None,
    manual_invocation: ManualSkillInvocation | None = None,
    permission_snapshot: PermissionSnapshot | None = None,
) -> str:
    if manual_invocation is None:
        rendered = (
            f"{_format_runtime_context(current_time=current_time, session_id=session_id, permission_snapshot=permission_snapshot)}\n\n"
            "## User Input\n\n"
            f"{content}"
        )
    else:
        rendered = (
            f"{_format_runtime_context(current_time=current_time, session_id=session_id, permission_snapshot=permission_snapshot)}\n\n"
            "## Skill Instructions\n\n"
            "```json\n"
            f"{_markdown_safe_json({'name': manual_invocation.metadata.name, 'body': manual_invocation.body})}\n"
            "```\n\n"
            "## User Request\n\n"
            "```json\n"
            f"{_markdown_safe_json(manual_invocation.request)}\n"
            "```"
        )
    if blackboard_projection is None:
        return rendered
    blackboard = (
        "## Task goal\n\n"
        f"{blackboard_projection['goal']}\n\n"
        "## Completion boundary\n\n"
        f"{blackboard_projection['completion_boundary']}"
    )
    return f"{rendered}\n\n{blackboard}"


def _format_runtime_context(
    *,
    current_time: datetime,
    session_id: str,
    permission_snapshot: PermissionSnapshot | None = None,
) -> str:
    rendered = (
        "## Runtime Context\n\n"
        f"- Current time: {format_rfc3339_milliseconds(current_time)}\n"
        f"- Session ID: {session_id}"
    )
    if permission_snapshot is None:
        return rendered
    return (
        f"{rendered}\n"
        f"- Tool Permission Level: {permission_snapshot.level}\n"
        f"- Exec Shell: {permission_snapshot.exec_shell.family}\n"
        "- Tool Confirmation: Permission checks may require one confirmation per Tool call."
    )


def _build_foreground_system_prompt(
    *,
    workspace: PurePath,
    agent_home: PurePath,
    long_term_memory: str,
    skills: Sequence[LoadedSkill],
) -> str:
    runtime = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
    sections = [
        render_template(
            "foreground-chat-system-prompt.md",
            workspace=workspace,
            agent_home=agent_home,
            runtime=runtime,
            long_term_memory=_project_long_term_memory(long_term_memory),
        )
    ]
    loaded_skills = tuple(skills)
    if loaded_skills:
        entries = "\n".join(
            _markdown_safe_json(
                {
                    "name": skill.metadata.name,
                    "description": skill.metadata.description,
                    "path": str(skill.metadata.path),
                }
            )
            for skill in loaded_skills
        )
        sections.append(render_template("skill-catalog.md", entries=entries))
        always_entries = "\n".join(
            _markdown_safe_json({"name": skill.metadata.name, "body": skill.document})
            for skill in loaded_skills
            if skill.always
        )
        if always_entries:
            sections.append(render_template("skill-always-load.md", entries=always_entries))
    return "\n\n".join(sections)


def _project_long_term_memory(long_term_memory: str) -> str:
    heading = "# Long-term Memory"
    if long_term_memory == heading:
        projected = ""
    elif long_term_memory.startswith(f"{heading}\n"):
        projected = long_term_memory.removeprefix(f"{heading}\n").removeprefix("\n")
    else:
        projected = long_term_memory
    return projected.replace("##", "###")


def _markdown_safe_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).translate(_MARKDOWN_SAFE_JSON_TRANSLATION)


__all__ = ["ContextBuilder"]
