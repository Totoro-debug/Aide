"""Glob Core Catalog Tool."""

from __future__ import annotations

from typing import Annotated, Any, cast

from aide.agent.tools.base import ToolError, ToolParam
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core._directory import (
    DirectoryReadTool,
    iter_directory_entries,
    matches_glob_pattern,
    normalize_glob_pattern,
    report_path,
    requested_path_has_directory_link,
)


class GlobTool(DirectoryReadTool):
    """Match files and directories beneath a directory root."""

    name = "glob"
    description = "Match files and directories beneath a directory root."
    required = ("pattern",)

    pattern: Annotated[str, ToolParam(description="Relative glob pattern.", min_length=1)]
    path: Annotated[str, ToolParam(description="Directory root.", min_length=1)] = "."
    head_limit: Annotated[
        int,
        ToolParam(description="Maximum matches; zero means unlimited.", minimum=0, maximum=1000),
    ] = 200
    offset: Annotated[int, ToolParam(description="Number of matches to skip.", minimum=0)] = 0
    kind: Annotated[
        str,
        ToolParam(description="Return files, directories, or both.", min_length=1),
    ] = "files"


    def validate_arguments(  # type: ignore[override]
        self,
        *,
        pattern: str,
        path: str,
        head_limit: int,
        offset: int,
        kind: str,
    ) -> str | None:
        del path, head_limit, offset
        try:
            normalize_glob_pattern(pattern)
        except ValueError as error:
            return str(error)
        if kind not in {"files", "dirs", "both"}:
            return "Glob kind must be one of files, dirs, or both."
        return None

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
        pattern = cast(str, arguments["pattern"])
        path = cast(str, arguments["path"])
        head_limit = cast(int, arguments["head_limit"])
        offset = cast(int, arguments["offset"])
        kind = cast(str, arguments["kind"])
        if requested_path_has_directory_link(workspace, path):
            return ""
        target = self.resolve_path_argument(workspace=workspace, requested=path)
        normalized_pattern = normalize_glob_pattern(pattern)
        try:
            entries = iter_directory_entries(target)
            matched = [
                entry
                for entry in entries
                if _matches_kind(entry.is_directory, kind)
                and matches_glob_pattern(entry.relative, normalized_pattern)
            ]
        except OSError as error:
            raise ToolError(f"Glob failed: {error}") from error

        reported = sorted(
            report_path(entry, workspace=workspace, search_root=target) for entry in matched
        )
        if head_limit == 0:
            selected = reported[offset:]
        else:
            selected = reported[offset : offset + head_limit]
        return "\n".join(selected)


def _matches_kind(is_directory: bool, kind: str) -> bool:
    return (
        kind == "both"
        or (kind == "dirs" and is_directory)
        or (kind == "files" and not is_directory)
    )


__all__ = ["GlobTool"]
