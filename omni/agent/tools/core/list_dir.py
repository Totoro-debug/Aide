"""List Dir Core Catalog Tool."""

from __future__ import annotations

from typing import Annotated, Any, cast

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext
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
                role="read",
            ),
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
        workspace = context.workspace
        path = cast(str, arguments["path"])
        recursive = cast(bool, arguments["recursive"])
        max_entries = cast(int, arguments["max_entries"])
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


__all__ = ["ListDirTool"]
