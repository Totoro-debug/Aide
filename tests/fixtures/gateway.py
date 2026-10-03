"""Small test adapters for exercising one concrete Tool through the common pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from omni.agent.tools.base import BaseTool
from omni.agent.tools.file_mutation import FileMutationRecorder
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import (
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)


class SingleToolGateway(ToolGateway):
    """Bind a test Tool to the production preparation and result pipeline."""

    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        confirmation: ConfirmationRequester | None = None,
        permission_context: PermissionContext | None = None,
    ) -> None:
        self._gateway = ToolGateway._for_memory(
            tuple(tools),
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
