"""Main Agent Tools for submitting and querying Session SubAgents."""

from __future__ import annotations

import json
from collections.abc import Collection
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

from aide.agent.subagents.models import SubAgentStatus
from aide.agent.subagents.store import SubAgentRequestError, SubAgentStoreError
from aide.agent.tools.base import BaseTool, ToolError
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.permission import ToolAuthorizationSession

if TYPE_CHECKING:
    from aide.agent.subagents.context import SubAgentToolContext

_CONTEXT_REQUIRED = "SubAgent Tools require a Main Agent Session context."
_INVALID_SPAWN_ARGUMENTS = "spawn_agent requires a non-empty title and task."
_INVALID_WAIT_ARGUMENTS = "wait_agent requires non-empty SubAgent IDs and a valid timeout."
_INVALID_LIST_ARGUMENTS = "list_agents received invalid pagination or status arguments."
_REGISTRATION_FAILED = "The SubAgent task could not be registered."
_STORAGE_FAILED = "SubAgent records could not be read or saved reliably."
_STATUSES = tuple(status.value for status in SubAgentStatus)


def _arguments(
    value: dict[str, Any],
    *,
    allowed: set[str],
    required: Collection[str] = (),
) -> dict[str, Any]:
    if set(value) - allowed or set(required) - set(value):
        raise ToolError("SubAgent Tool arguments are invalid.")
    return deepcopy(value)


def _run_context(context: ToolRunContext) -> SubAgentToolContext:
    subagent = context.subagent
    if subagent is None:
        raise ToolError(_CONTEXT_REQUIRED)
    return subagent


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _SubAgentTool(BaseTool):
    _contextual: ClassVar[bool] = True

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> str:
        del arguments, authorization, context, kwargs
        raise NotImplementedError


class SpawnAgentTool(_SubAgentTool):
    name = "spawn_agent"
    description = "Register one independent task for a SubAgent in the current Session."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "task": {"type": "string", "minLength": 1},
        },
        "required": ["title", "task"],
        "additionalProperties": False,
    }

    async def execute(self, *, title: str, task: str) -> str:
        del title, task
        raise ToolError(_CONTEXT_REQUIRED)

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prepared = _arguments(arguments, allowed={"title", "task"}, required={"title", "task"})
        if (
            not isinstance(prepared["title"], str)
            or not prepared["title"].strip()
            or not isinstance(prepared["task"], str)
            or not prepared["task"].strip()
        ):
            raise ToolError(_INVALID_SPAWN_ARGUMENTS)
        return prepared

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> str:
        del authorization, kwargs
        request = _run_context(context)
        if request.creator_snapshot is None:
            raise ToolError("SubAgent Model Route is unavailable.")
        try:
            record = request.coordinator.submit(
                title=arguments["title"],
                task=arguments["task"],
                parent_run_id=request.parent_run_id,
                source=request.source,
                creator_snapshot=request.creator_snapshot,
            )
        except (SubAgentRequestError, SubAgentStoreError) as error:
            raise ToolError(_REGISTRATION_FAILED) from error
        return _json({"agent_id": record.agent_id, "status": record.status.value})


class WaitAgentTool(_SubAgentTool):
    name = "wait_agent"
    description = "Wait for one or more SubAgents to reach a terminal result."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "agent_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "SubAgent IDs in the current Session.",
            },
            "timeout_ms": {
                "type": ["integer", "null"],
                "minimum": 0,
                "default": None,
                "description": "Maximum wait in milliseconds; omit to wait without a timeout.",
            },
        },
        "required": ["agent_ids"],
        "additionalProperties": False,
    }

    async def execute(self, *, agent_ids: list[str], timeout_ms: int | None = None) -> str:
        del agent_ids, timeout_ms
        raise ToolError(_CONTEXT_REQUIRED)

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prepared = _arguments(
            arguments,
            allowed={"agent_ids", "timeout_ms"},
            required={"agent_ids"},
        )
        agent_ids = prepared["agent_ids"]
        timeout_ms = prepared.get("timeout_ms")
        if (
            not isinstance(agent_ids, list)
            or not agent_ids
            or any(not isinstance(agent_id, str) or not agent_id.strip() for agent_id in agent_ids)
            or (
                timeout_ms is not None
                and (
                    isinstance(timeout_ms, bool)
                    or not isinstance(timeout_ms, int)
                    or timeout_ms < 0
                )
            )
        ):
            raise ToolError(_INVALID_WAIT_ARGUMENTS)
        prepared["agent_ids"] = list(agent_ids)
        prepared["timeout_ms"] = timeout_ms
        return prepared

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> str:
        del authorization, kwargs
        request = _run_context(context)
        try:
            results = await request.coordinator.wait(
                arguments["agent_ids"],
                timeout_ms=arguments["timeout_ms"],
            )
        except SubAgentRequestError as error:
            raise ToolError(error.args[0]) from error
        except SubAgentStoreError as error:
            raise ToolError(_STORAGE_FAILED) from error
        return _json([result.to_dict() for result in results])


class ListAgentsTool(_SubAgentTool):
    name = "list_agents"
    description = "List brief SubAgent statuses in the current Session."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "status": {
                "type": ["string", "null"],
                "enum": [*_STATUSES, None],
                "default": None,
            },
            "cursor": {"type": ["string", "null"], "default": None},
            "limit": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 100,
                "default": None,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def execute(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> str:
        del status, cursor, limit
        raise ToolError(_CONTEXT_REQUIRED)

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prepared = _arguments(arguments, allowed={"status", "cursor", "limit"})
        status = prepared.get("status")
        cursor = prepared.get("cursor")
        limit = prepared.get("limit")
        if (
            (status is not None and status not in _STATUSES)
            or (cursor is not None and (not isinstance(cursor, str) or not cursor))
            or (
                limit is not None
                and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100)
            )
        ):
            raise ToolError(_INVALID_LIST_ARGUMENTS)
        return {"status": status, "cursor": cursor, "limit": limit}

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> str:
        del authorization, kwargs
        request = _run_context(context)
        try:
            page = request.coordinator.list(
                status=arguments["status"],
                cursor=arguments["cursor"],
                limit=arguments["limit"],
            )
        except SubAgentRequestError as error:
            raise ToolError(error.args[0]) from error
        except SubAgentStoreError as error:
            raise ToolError(_STORAGE_FAILED) from error
        return _json(
            {
                "items": [
                    {
                        "agent_id": item.agent_id,
                        "title": item.title,
                        "status": item.status.value,
                        "created_at": item.created_at.isoformat(),
                        "finished_at": (
                            None if item.finished_at is None else item.finished_at.isoformat()
                        ),
                    }
                    for item in page.items
                ],
                "next_cursor": page.next_cursor,
            }
        )


def build_subagent_tools() -> tuple[BaseTool, BaseTool, BaseTool]:
    """Create the three concrete Tools that T6 binds to eligible Main Agent Runs."""
    return SpawnAgentTool(), WaitAgentTool(), ListAgentsTool()


__all__ = [
    "ListAgentsTool",
    "SpawnAgentTool",
    "WaitAgentTool",
    "build_subagent_tools",
]
