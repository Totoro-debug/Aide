"""Grep Core Catalog Tool."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final

from myclaw.agent.tools.base import BaseTool, ToolError, ToolParam
from myclaw.agent.tools.core._directory import (
    is_ignored_directory_name,
    iter_directory_entries,
    matches_glob_pattern,
    normalize_glob_pattern,
    report_path,
)
from myclaw.permission.policy import FileAccess

_OUTPUT_MODES: Final = frozenset({"content", "files_with_matches", "count"})
_TYPE_PATTERNS: Final = {
    "py": ("*.py", "*.pyi"),
    "python": ("*.py", "*.pyi"),
    "js": ("*.js", "*.mjs", "*.cjs"),
    "javascript": ("*.js", "*.mjs", "*.cjs"),
    "ts": ("*.ts",),
    "typescript": ("*.ts",),
    "tsx": ("*.tsx",),
    "jsx": ("*.jsx",),
    "json": ("*.json",),
    "md": ("*.md", "*.markdown"),
    "markdown": ("*.md", "*.markdown"),
    "go": ("*.go",),
    "rs": ("*.rs",),
    "rust": ("*.rs",),
    "java": ("*.java",),
    "sh": ("*.sh", "*.bash", "*.zsh", "*.fish"),
    "shell": ("*.sh", "*.bash", "*.zsh", "*.fish"),
    "yml": ("*.yml", "*.yaml"),
    "yaml": ("*.yml", "*.yaml"),
    "toml": ("*.toml",),
    "sql": ("*.sql",),
    "html": ("*.html", "*.htm"),
    "css": ("*.css", "*.scss", "*.sass", "*.less"),
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One visible file path and its path relative to the search root."""

    path: Path
    relative: PurePosixPath
    reported: str
    explicit: bool


@dataclass(frozen=True, slots=True)
class _FileMatches:
    """One readable candidate together with its matching line numbers."""

    candidate: _Candidate
    lines: tuple[str, ...]
    matching_lines: tuple[int, ...]


class GrepTool(BaseTool):
    """Search UTF-8 text in one file or throughout a directory root."""

    name = "grep"
    description = "Search UTF-8 text in a file or directory."
    required = ("pattern",)

    pattern: Annotated[
        str, ToolParam(description="Regular expression or fixed text.", min_length=1)
    ]
    path: Annotated[str, ToolParam(description="File or directory root.", min_length=1)] = "."
    glob: Annotated[
        str | None,
        ToolParam(description="Optional Glob filter for candidate files."),
    ] = None
    type: Annotated[
        str | None,
        ToolParam(description="Optional source type or file suffix filter."),
    ] = None
    output_mode: Annotated[str, ToolParam(description="Content, matching files, or counts.")] = (
        "content"
    )
    fixed_string: bool = False
    ignore_case: bool = False
    context: Annotated[
        int,
        ToolParam(description="Context lines before and after each selected match.", minimum=0),
    ] = 0
    head_limit: Annotated[
        int,
        ToolParam(
            description="Maximum matches or matching files; zero means unlimited.",
            minimum=0,
            maximum=1000,
        ),
    ] = 0
    offset: Annotated[
        int, ToolParam(description="Number of matches or files to skip.", minimum=0)
    ] = 0

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        pattern: str,
        path: str,
        glob: str | None,
        type: str | None,
        output_mode: str,
        fixed_string: bool,
        ignore_case: bool,
        context: int,
        head_limit: int,
        offset: int,
    ) -> str | None:
        del path, context, head_limit, offset
        if output_mode not in _OUTPUT_MODES:
            return "Grep output_mode must be one of content, files_with_matches, or count."
        if glob is not None and glob.strip():
            try:
                normalize_glob_pattern(glob.strip())
            except ValueError as error:
                return str(error)
        if not fixed_string:
            try:
                re.compile(pattern, re.IGNORECASE if ignore_case else 0)
            except re.error as error:
                return f"Grep pattern is invalid: {error}"
        return None

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        return (
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=str(prepared_arguments["path"]),
                role="read",
            ),
        )

    async def execute(
        self,
        *,
        pattern: str,
        path: str,
        glob: str | None,
        type: str | None,
        output_mode: str,
        fixed_string: bool,
        ignore_case: bool,
        context: int,
        head_limit: int,
        offset: int,
    ) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        candidates, approved_root = self._candidates(target=target, requested=path)
        glob_filter = _optional_glob(glob)
        type_filter = _type_patterns(type)
        matcher = _matcher(pattern, fixed_string=fixed_string, ignore_case=ignore_case)
        files: list[_FileMatches] = []

        for candidate in candidates:
            if glob_filter is not None and not matches_glob_pattern(
                candidate.relative, glob_filter
            ):
                continue
            if type_filter is not None and not any(
                matches_glob_pattern(candidate.relative, pattern) for pattern in type_filter
            ):
                continue
            lines = self._read_lines(
                candidate,
                approved_root=approved_root,
            )
            if lines is None:
                continue
            matching_lines = tuple(
                line_number for line_number, line in enumerate(lines) if matcher(line)
            )
            if matching_lines:
                files.append(
                    _FileMatches(
                        candidate=candidate,
                        lines=lines,
                        matching_lines=matching_lines,
                    )
                )

        if output_mode == "content":
            return _render_content(files, context=context, head_limit=head_limit, offset=offset)
        selected = _paginate(files, offset=offset, head_limit=head_limit)
        if output_mode == "files_with_matches":
            return "\n".join(item.candidate.reported for item in selected)
        return "\n".join(
            f"{item.candidate.reported}:{len(item.matching_lines)}" for item in selected
        )

    def _candidates(self, *, target: Path, requested: str) -> tuple[list[_Candidate], Path]:
        workspace_root = self._workspace.resolve(strict=True)
        lexical = _lexical_path(self._workspace, requested)
        if target.is_dir():
            if _contains_directory_link(lexical, workspace_root=workspace_root):
                return [], target
            if _contains_ignored_directory(
                target, workspace_root=workspace_root
            ) or _contains_ignored_directory(lexical, workspace_root=workspace_root):
                return [], target
            try:
                entries = iter_directory_entries(target)
                candidates = [
                    _Candidate(
                        path=entry.path,
                        relative=entry.relative,
                        reported=report_path(
                            entry,
                            workspace=self._workspace,
                            search_root=target,
                        ),
                        explicit=False,
                    )
                    for entry in entries
                    if not entry.is_directory
                ]
            except OSError as error:
                raise ToolError(f"Grep failed: {error}") from error
            return sorted(candidates, key=lambda item: item.reported), target

        if not target.is_file():
            raise ToolError("Grep failed: The requested path must identify a file or directory.")
        if _contains_directory_link(lexical.parent, workspace_root=workspace_root):
            return [], target.parent
        if _contains_ignored_directory(
            target, workspace_root=workspace_root
        ) or _contains_ignored_directory(lexical, workspace_root=workspace_root):
            return [], target.parent

        if lexical.is_relative_to(workspace_root):
            reported = lexical.relative_to(workspace_root).as_posix()
        else:
            reported = lexical.as_posix()
        return [
            _Candidate(
                path=lexical,
                relative=PurePosixPath(lexical.name),
                reported=reported,
                explicit=True,
            )
        ], target.parent

    @staticmethod
    def _read_lines(candidate: _Candidate, *, approved_root: Path) -> tuple[str, ...] | None:
        try:
            resolved = candidate.path.resolve(strict=True)
        except OSError as error:
            if candidate.explicit:
                raise ToolError(f"Grep failed: {error}") from error
            return None
        if not resolved.is_relative_to(approved_root):
            return None
        if not resolved.is_file():
            return None
        try:
            content = candidate.path.read_bytes()
            if b"\x00" in content:
                return None
            return tuple(content.decode("utf-8").splitlines())
        except (OSError, UnicodeError) as error:
            if candidate.explicit:
                raise ToolError(f"Grep failed: {error}") from error
            return None


def _lexical_path(workspace: Path, requested: str) -> Path:
    path = Path(requested)
    if not path.is_absolute():
        path = workspace / path
    return path.absolute()


def _contains_ignored_directory(path: Path, *, workspace_root: Path) -> bool:
    current = path if path.is_dir() else path.parent
    return _has_matching_path_component(
        current,
        workspace_root=workspace_root,
        predicate=lambda candidate: is_ignored_directory_name(candidate.name),
    )


def _contains_directory_link(path: Path, *, workspace_root: Path) -> bool:
    return _has_matching_path_component(
        path,
        workspace_root=workspace_root,
        predicate=lambda candidate: candidate.is_symlink() or candidate.is_junction(),
    )


def _has_matching_path_component(
    path: Path,
    *,
    workspace_root: Path,
    predicate: Callable[[Path], bool],
) -> bool:
    current = path
    boundary = workspace_root if current.is_relative_to(workspace_root) else None
    while boundary is None or current != boundary:
        if predicate(current):
            return True
        parent = current.parent
        if parent == current:
            break
        current = parent
    return False


def _optional_glob(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return normalize_glob_pattern(value.strip())


def _type_patterns(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower().lstrip(".")
    return _TYPE_PATTERNS.get(normalized, (f"*.{normalized}",))


def _matcher(
    pattern: str,
    *,
    fixed_string: bool,
    ignore_case: bool,
) -> Callable[[str], bool]:
    expression = re.escape(pattern) if fixed_string else pattern
    try:
        compiled = re.compile(expression, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        raise ToolError(f"Grep pattern is invalid: {error}") from error
    return lambda line: compiled.search(line) is not None


def _paginate[T](
    values: list[T],
    *,
    offset: int,
    head_limit: int,
) -> list[T]:
    if head_limit == 0:
        return values[offset:]
    return values[offset : offset + head_limit]


def _render_content(
    files: list[_FileMatches],
    *,
    context: int,
    head_limit: int,
    offset: int,
) -> str:
    matches = [
        (file_index, line_number)
        for file_index, item in enumerate(files)
        for line_number in item.matching_lines
    ]
    selected_matches = _paginate(matches, offset=offset, head_limit=head_limit)
    selected_by_file: dict[int, set[int]] = {}
    for file_index, line_number in selected_matches:
        selected_by_file.setdefault(file_index, set()).add(line_number)

    output: list[str] = []
    for file_index, selected_lines in selected_by_file.items():
        item = files[file_index]
        windows = _context_windows(
            selected_lines,
            line_count=len(item.lines),
            context=context,
        )
        for start, end in windows:
            if output:
                output.append("--")
            for line_number in range(start, end + 1):
                delimiter = ":" if line_number in selected_lines else "-"
                output.append(
                    f"{item.candidate.reported}{delimiter}{line_number + 1}{delimiter}"
                    f"{item.lines[line_number]}"
                )
    return "\n".join(output)


def _context_windows(
    selected_lines: set[int],
    *,
    line_count: int,
    context: int,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for line_number in sorted(selected_lines):
        window = (
            max(0, line_number - context),
            min(line_count - 1, line_number + context),
        )
        if windows and window[0] <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], window[1]))
        else:
            windows.append(window)
    return windows


__all__ = ["GrepTool"]
