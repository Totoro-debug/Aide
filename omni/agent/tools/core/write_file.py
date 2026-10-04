"""Write File Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext
from omni.agent.tools.file_mutation import (
    FileMutationRecorder,
    execute_recorded_mutation,
    is_protected_restore_target,
)
from omni.agent.tools.permission import FileAccess


class WriteFileTool(BaseTool):
    """Write exact UTF-8 bytes to any host-readable file path."""

    name = "write_file"
    description = "Write UTF-8 text to a file. Paths outside the Workspace require confirmation."
    required = ("path", "content")
    _contextual = True

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]


    def build_file_accesses_for_context(
        self,
        prepared_arguments: dict[str, object],
        *,
        context: ToolRunContext,
    ) -> tuple[FileAccess, ...]:
        workspace = context.workspace
        return (
            self.canonical_file_access(
                workspace=workspace,
                base=workspace,
                requested=str(prepared_arguments["path"]),
                role="write",
            ),
        )

    def refusal_reason_for_context(
        self,
        prepared_arguments: dict[str, object],
        *,
        context: ToolRunContext,
    ) -> str | None:
        path = prepared_arguments.get("path")
        if not isinstance(path, str):
            raise ToolError("Write File arguments are invalid.")
        target = self.resolve_path_argument(workspace=context.workspace, requested=path)
        if is_protected_restore_target(context.workspace, target):
            return "Built-in File Tools cannot write to protected restore state."
        return None

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, object],
        authorization: object,
        *,
        context: ToolRunContext,
        mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
        mutation_target: Path | None = None,
        **kwargs: object,
    ) -> str:
        del authorization, kwargs
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            raise ToolError("Write File arguments are invalid.")
        target = mutation_target
        if target is None:
            target = self.resolve_path_argument(workspace=context.workspace, requested=path)
        recorder = mutation_recorder if run_token is not None else None
        run_token = run_token if mutation_recorder is not None else None
        async def mutation() -> str:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            except (OSError, UnicodeError) as error:
                raise ToolError(f"Write File failed: {error}") from error
            return "File written successfully."

        return await execute_recorded_mutation(
            mutation,
            recorder=recorder,
            run_token=run_token,
            target=target,
        )
