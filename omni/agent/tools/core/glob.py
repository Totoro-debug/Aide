"""Glob Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext, bind_tool_run_context, bound_tool_run_context
from omni.agent.tools.core._directory import (
    iter_directory_entries,
    matches_glob_pattern,
    normalize_glob_pattern,
    report_path,
    requested_path_has_directory_link,
)
from omni.agent.tools.permission import FileAccess


class GlobTool(BaseTool):
    """Match files and directories beneath a directory root."""

    name = "glob"
    description = "Match files and directories beneath a directory root."
    required = ("pattern",)
    _contextual = True

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

    _workspace: Path | None

    def __init__(self, *, workspace: Path | None = None) -> None:
        if workspace is not None:
            self._workspace = workspace

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

    async def execute(
        self,
        *,
        pattern: str,
        path: str,
        head_limit: int,
        offset: int,
        kind: str,
    ) -> str:
        return await self._execute_at_workspace(
            workspace=self._legacy_workspace(),
            pattern=pattern,
            path=path,
            head_limit=head_limit,
            offset=offset,
            kind=kind,
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
                pattern=cast(str, arguments["pattern"]),
                path=cast(str, arguments["path"]),
                head_limit=cast(int, arguments["head_limit"]),
                offset=cast(int, arguments["offset"]),
                kind=cast(str, arguments["kind"]),
            )

    async def _execute_at_workspace(
        self,
        *,
        workspace: Path,
        pattern: str,
        path: str,
        head_limit: int,
        offset: int,
        kind: str,
    ) -> str:
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

    def _legacy_workspace(self) -> Path:
        workspace = getattr(self, "_workspace", None)
        if not isinstance(workspace, Path):
            context = bound_tool_run_context()
            if context is not None:
                return context.workspace
            raise RuntimeError("Glob requires an explicit Tool Run Context")
        return workspace


def _matches_kind(is_directory: bool, kind: str) -> bool:
    return (
        kind == "both"
        or (kind == "dirs" and is_directory)
        or (kind == "files" and not is_directory)
    )


__all__ = ["GlobTool"]
