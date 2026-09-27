"""Write File Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from myclaw.permission.policy import FileAccess
from myclaw.session.backup_store import FileMutationRecorder
from myclaw.tools.base import BaseTool, ToolError, ToolParam
from myclaw.tools.files._file_mutation import (
    execute_recorded_mutation,
    is_protected_restore_target,
)


class WriteFileTool(BaseTool):
    """Write exact UTF-8 bytes to any host-readable file path."""

    name = "write_file"
    description = "Write UTF-8 text to a file. Paths outside the Workspace require confirmation."
    required = ("path", "content")

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        return (
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=str(prepared_arguments["path"]),
                role="write",
            ),
        )

    def refusal_reason(self, *, path: str, **arguments: object) -> str | None:
        del arguments
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        if is_protected_restore_target(self._workspace, target):
            return "Built-in File Tools cannot write to protected restore state."
        return None

    async def execute(self, *, path: str, content: str) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        return await self._execute_at_target(
            target=target,
            content=content,
            recorder=None,
            run_token=None,
        )

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: object,
        *,
        mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
        mutation_target: Path | None = None,
    ) -> str:
        del authorization
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            raise ToolError("Write File arguments are invalid.")
        target = mutation_target
        if target is None:
            target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        return await self._execute_at_target(
            target=target,
            content=content,
            recorder=(mutation_recorder if run_token is not None else None),
            run_token=(run_token if mutation_recorder is not None else None),
        )

    async def _execute_at_target(
        self,
        *,
        target: Path,
        content: str,
        recorder: FileMutationRecorder | None,
        run_token: UUID | None,
    ) -> str:
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
