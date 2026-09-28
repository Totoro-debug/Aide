"""Edit File Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from myclaw.agent.tools.base import BaseTool, ToolError, ToolParam
from myclaw.agent.tools.file_mutation import (
    FileMutationRecorder,
    execute_recorded_mutation,
    is_protected_restore_target,
)
from myclaw.agent.tools.permission import FileAccess


class EditFileTool(BaseTool):
    """Replace exact strict UTF-8 text in any host-readable file path."""

    name = "edit_file"
    description = (
        "Replace exact UTF-8 text in a file. Paths outside the Workspace require confirmation."
    )
    required = ("path", "old_text", "new_text")

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        requested = str(prepared_arguments["path"])
        return (
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=requested,
                role="read",
            ),
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=requested,
                role="write",
            ),
        )

    def refusal_reason(self, *, path: str, **arguments: object) -> str | None:
        del arguments
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        if is_protected_restore_target(self._workspace, target):
            return "Built-in File Tools cannot write to protected restore state."
        return None

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        return await self._execute_at_target(
            target=target,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
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
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        replace_all = arguments.get("replace_all")
        if (
            not isinstance(path, str)
            or not isinstance(old_text, str)
            or not isinstance(new_text, str)
            or not isinstance(replace_all, bool)
        ):
            raise ToolError("Edit File arguments are invalid.")
        target = mutation_target
        if target is None:
            target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        return await self._execute_at_target(
            target=target,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            recorder=(mutation_recorder if run_token is not None else None),
            run_token=(run_token if mutation_recorder is not None else None),
        )

    async def _execute_at_target(
        self,
        *,
        target: Path,
        old_text: str,
        new_text: str,
        replace_all: bool,
        recorder: FileMutationRecorder | None,
        run_token: UUID | None,
    ) -> str:
        try:
            raw_content = target.read_bytes()
        except OSError as error:
            raise ToolError(f"Edit File read failed: {error}") from error
        try:
            content = raw_content.decode("utf-8")
        except UnicodeError as error:
            raise ToolError("Edit File failed: the target is not valid UTF-8 text.") from error

        match_count = content.count(old_text)
        if match_count == 0:
            raise ToolError("Edit File found zero matches for the requested text.")
        if not replace_all and match_count != 1:
            raise ToolError(
                "Edit File found ambiguous text; use replace_all to replace every match."
            )

        replacement = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )

        async def mutation() -> str:
            try:
                target.write_bytes(replacement.encode("utf-8"))
            except (OSError, UnicodeError) as error:
                raise ToolError(f"Edit File write failed: {error}") from error
            return "File edited successfully."

        return await execute_recorded_mutation(
            mutation,
            recorder=recorder,
            run_token=run_token,
            target=target,
        )
