"""Read File Core Catalog Tool."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext, bind_tool_run_context, bound_tool_run_context
from omni.agent.tools.permission import FileAccess


class ReadFileTool(BaseTool):
    """Read a strict UTF-8 line window from any host-readable file."""

    name = "read_file"
    description = (
        "Read UTF-8 text lines from a file. Paths outside the Workspace may require confirmation."
    )
    required = ("path",)
    _contextual = True

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    offset: Annotated[int, ToolParam(description="One-based first line.", minimum=1)] = 1
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    _workspace: Path | None

    def __init__(self, *, workspace: Path | None = None, skill_root: Path | None = None) -> None:
        if workspace is not None:
            self._workspace = workspace
        self._skill_root = None if skill_root is None else Path(skill_root).resolve(strict=False)

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
        access = self.canonical_file_access(
            workspace=workspace,
            base=workspace,
            requested=str(prepared_arguments["path"]),
            role="read",
        )
        if self._skill_root is not None and access.path.is_relative_to(self._skill_root):
            access = replace(access, allowed_roots=(self._skill_root,))
        return (access,)

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        return await self._execute_at_workspace(
            workspace=self._legacy_workspace(),
            path=path,
            offset=offset,
            limit=limit,
        )

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, object],
        authorization: object,
        *,
        context: ToolRunContext,
        **kwargs: object,
    ) -> str:
        del authorization, kwargs
        path = arguments.get("path")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        if (
            not isinstance(path, str)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(limit, bool)
            or not isinstance(limit, int)
        ):
            raise ToolError("Read File arguments are invalid.")
        with bind_tool_run_context(context):
            return await self.execute(path=path, offset=offset, limit=limit)

    async def _execute_at_workspace(
        self,
        *,
        workspace: Path,
        path: str,
        offset: int,
        limit: int,
    ) -> str:
        target = self.resolve_path_argument(workspace=workspace, requested=path)
        try:
            raw_content = target.read_bytes()
        except OSError as error:
            raise ToolError(f"Read File failed: {error}") from error
        try:
            content = raw_content.decode("utf-8")
        except UnicodeError as error:
            raise ToolError("Read File failed: the target is not valid UTF-8 text.") from error
        lines = content.splitlines(keepends=True)
        return "".join(lines[offset - 1 : offset - 1 + limit])

    def _legacy_workspace(self) -> Path:
        workspace = getattr(self, "_workspace", None)
        if not isinstance(workspace, Path):
            context = bound_tool_run_context()
            if context is not None:
                return context.workspace
            raise RuntimeError("Read File requires an explicit Tool Run Context")
        return workspace
