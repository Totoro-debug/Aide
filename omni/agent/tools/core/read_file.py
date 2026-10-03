"""Read File Core Catalog Tool."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.permission import FileAccess


class ReadFileTool(BaseTool):
    """Read a strict UTF-8 line window from any host-readable file."""

    name = "read_file"
    description = (
        "Read UTF-8 text lines from a file. Paths outside the Workspace may require confirmation."
    )
    required = ("path",)

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    offset: Annotated[int, ToolParam(description="One-based first line.", minimum=1)] = 1
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    def __init__(self, *, workspace: Path, skill_root: Path | None = None) -> None:
        self._workspace = workspace
        self._skill_root = None if skill_root is None else Path(skill_root).resolve(strict=False)

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        access = self.canonical_file_access(
            workspace=self._workspace,
            base=self._workspace,
            requested=str(prepared_arguments["path"]),
            role="read",
        )
        if self._skill_root is not None and access.path.is_relative_to(self._skill_root):
            access = replace(access, allowed_roots=(self._skill_root,))
        return (access,)

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
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
