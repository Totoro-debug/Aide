"""List Dir Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext, bind_tool_run_context, bound_tool_run_context
from omni.agent.tools.core._directory import (
    iter_directory_entries,
    report_path,
    requested_path_has_directory_link,
)
from omni.agent.tools.permission import FileAccess


class ListDirTool(BaseTool):
    """List visible files and directories beneath a directory root."""

    name = "list_dir"
    description = "List files and directories within a directory root."
    _contextual = True

    path: Annotated[str, ToolParam(description="Directory root.", min_length=1)] = "."
    recursive: Annotated[bool, ToolParam(description="Include nested entries.")] = False
    max_entries: Annotated[
        int,
        ToolParam(description="Maximum entries to return.", minimum=1, maximum=10000),
    ] = 200

    _workspace: Path | None

    def __init__(self, *, workspace: Path | None = None) -> None:
        if workspace is not None:
            self._workspace = workspace

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        return self._build_file_accesses(prepared_arguments, workspace=self._legacy_workspace())

    def build_file_accesses_for_context(
        self,
        prepared_arguments: dict[str, object],
        *,
        context: ToolRunContext,
    ) -> tuple[FileAccess, ...]:
        return self._build_file_accesses(prepared_arguments, workspace=context.workspace)

    def _build_file_accesses(
        self,
        prepared_arguments: dict[str, object],
        *,
        workspace: Path,
    ) -> tuple[FileAccess, ...]:
        return (
            self.canonical_file_access(
                workspace=workspace,
                base=workspace,
                requested=str(prepared_arguments["path"]),
                role="read",
            ),
        )

    async def execute(self, *, path: str, recursive: bool, max_entries: int) -> str:
        return await self._execute_at_workspace(
            workspace=self._legacy_workspace(),
            path=path,
            recursive=recursive,
            max_entries=max_entries,
        )

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, Any],
        authorization: object,
        *,
        context: ToolRunContext,
        **kwargs: object,
    ) -> str:
        del authorization, kwargs
        with bind_tool_run_context(context):
            return await self.execute(
                path=cast(str, arguments["path"]),
                recursive=cast(bool, arguments["recursive"]),
                max_entries=cast(int, arguments["max_entries"]),
            )

    async def _execute_at_workspace(
        self,
        *,
        workspace: Path,
        path: str,
        recursive: bool,
        max_entries: int,
    ) -> str:
        if requested_path_has_directory_link(workspace, path):
            return ""
        target = self.resolve_path_argument(workspace=workspace, requested=path)
        try:
            entries = list(iter_directory_entries(target, recursive=recursive))
        except OSError as error:
            raise ToolError(f"List Dir failed: {error}") from error

        reported = sorted(
            report_path(entry, workspace=workspace, search_root=target) for entry in entries
        )
        return "\n".join(reported[:max_entries])

    def _legacy_workspace(self) -> Path:
        workspace = getattr(self, "_workspace", None)
        if not isinstance(workspace, Path):
            context = bound_tool_run_context()
            if context is not None:
                return context.workspace
            raise RuntimeError("List Dir requires an explicit Tool Run Context")
        return workspace


__all__ = ["ListDirTool"]
