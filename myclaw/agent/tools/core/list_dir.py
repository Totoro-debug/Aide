"""List Dir Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from myclaw.agent.tools.base import BaseTool, ToolError, ToolParam
from myclaw.agent.tools.core._directory import (
    iter_directory_entries,
    report_path,
    requested_path_has_directory_link,
)
from myclaw.permission.policy import FileAccess


class ListDirTool(BaseTool):
    """List visible files and directories beneath a directory root."""

    name = "list_dir"
    description = "List files and directories within a directory root."

    path: Annotated[str, ToolParam(description="Directory root.", min_length=1)] = "."
    recursive: Annotated[bool, ToolParam(description="Include nested entries.")] = False
    max_entries: Annotated[
        int,
        ToolParam(description="Maximum entries to return.", minimum=1, maximum=10000),
    ] = 200

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        return (
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=str(prepared_arguments["path"]),
                role="read",
            ),
        )

    async def execute(self, *, path: str, recursive: bool, max_entries: int) -> str:
        if requested_path_has_directory_link(self._workspace, path):
            return ""
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        try:
            entries = list(iter_directory_entries(target, recursive=recursive))
        except OSError as error:
            raise ToolError(f"List Dir failed: {error}") from error

        reported = sorted(
            report_path(entry, workspace=self._workspace, search_root=target) for entry in entries
        )
        return "\n".join(reported[:max_entries])


__all__ = ["ListDirTool"]
