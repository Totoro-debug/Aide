"""Small test adapters for exercising one concrete Tool through the common pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import UUID
from weakref import WeakKeyDictionary

from aide.agent.tools.base import BaseTool
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.file_mutation import FileMutationRecorder
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import (
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)
from aide.schedule.service import ScheduleService

_TEST_CONTEXTS: WeakKeyDictionary[BaseTool, ToolRunContext] = WeakKeyDictionary()


def contextual_tool[T: BaseTool](
    tool_type: type[T], *, workspace: Path | None = None,
    schedule_service: ScheduleService | None = None, **kwargs: Any,
) -> T:
    """Compose a test Tool and explicit context without binding production instances."""
    if workspace is None:
        if schedule_service is None:
            raise TypeError("Test Tool composition requires a Workspace")
        workspace = schedule_service._store.workspace_state.workspace_path
    tool = tool_type(**kwargs)
    _TEST_CONTEXTS[tool] = ToolRunContext(workspace=workspace, schedule_service=schedule_service)
    return tool


class SingleToolGateway(ToolGateway):
    """Bind a test Tool to the production preparation and result pipeline."""

    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        confirmation: ConfirmationRequester | None = None,
        permission_context: PermissionContext | None = None,
    ) -> None:
        selected_tools = tuple(tools)
        contexts = {_TEST_CONTEXTS[tool] for tool in selected_tools if tool in _TEST_CONTEXTS}
        if len(contexts) > 1:
            raise ValueError("Test Tools require one shared Tool Run Context")
        context = next(iter(contexts), None)
        self._gateway = ToolGateway._for_memory(
            selected_tools,
            tool_context=context,
            permission_context=permission_context,
        )
        self._confirmation = confirmation

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._gateway.schemas

    def is_micro_compression_eligible(self, tool_name: str) -> bool:
        return self._gateway.is_micro_compression_eligible(tool_name)

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: ConfirmationRequester | None = None,
        file_mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
    ) -> ToolResult:
        requester = self._confirmation if confirmation is None else confirmation
        return await self._gateway.call(
            tool_call,
            confirmation=requester,
            file_mutation_recorder=file_mutation_recorder,
            run_token=run_token,
        )
