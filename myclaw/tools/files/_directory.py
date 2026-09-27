"""Shared directory traversal and reporting for Core directory Tools."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
_DIRECTORY_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0)
_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_IGNORED_DIRECTORY_NAMES: Final = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
    }
)
_NORMALIZED_IGNORED_DIRECTORY_NAMES: Final = frozenset(
    os.path.normcase(name) for name in _IGNORED_DIRECTORY_NAMES
)


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """One visible child with its lexical path and directory-link status."""

    path: Path
    relative: PurePosixPath
    is_directory: bool
    is_link: bool


def iter_directory_entries(root: Path, *, recursive: bool = True) -> Iterator[DirectoryEntry]:
    """Walk descendants without following directory links or ignored directories."""
    _require_directory_root(root)
    if is_ignored_directory_name(root.name):
        return

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except OSError:
            if directory == root:
                raise
            continue

        for child in children:
            try:
                status = child.lstat()
            except OSError:
                continue
            entry = _entry_from_status(child, status, root)
            if entry is None:
                continue
            if entry.is_directory and is_ignored_directory_name(child.name):
                continue
            yield entry
            if recursive and entry.is_directory and not entry.is_link:
                pending.append(child)


def requested_path_has_directory_link(workspace: Path, requested: str) -> bool:
    """Return whether a requested root crosses a directory link or reparse point."""
    workspace_root = workspace.resolve(strict=True)
    path = Path(requested)
    if not path.is_absolute():
        path = workspace_root / path
    current = path.absolute()
    boundary = workspace_root if current.is_relative_to(workspace_root) else None

    while boundary is None or current != boundary:
        try:
            status = current.lstat()
        except OSError:
            pass
        else:
            if _is_link(status) and _link_is_directory(current, status):
                return True
        parent = current.parent
        if parent == current:
            break
        current = parent
    return False


def report_path(
    entry: DirectoryEntry,
    *,
    workspace: Path,
    search_root: Path,
) -> str:
    """Render Workspace-relative or confirmed external absolute POSIX paths."""
    workspace_root = workspace.resolve(strict=True)
    if search_root.is_relative_to(workspace_root):
        path = entry.path.relative_to(workspace_root)
    else:
        path = entry.path.absolute()
    result = path.as_posix()
    return f"{result}/" if entry.is_directory else result


def normalize_glob_pattern(pattern: str) -> str:
    """Normalize separators and reject every absolute pattern dialect."""
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE_PATTERN.match(normalized):
        raise ValueError("Glob pattern must be relative to the directory root.")
    return normalized


def matches_glob_pattern(relative: PurePosixPath, pattern: str) -> bool:
    """Apply the agreed case-sensitive directory or host-sensitive simple dialect."""
    if "/" in pattern:
        return relative.match(pattern)
    return fnmatch.fnmatch(relative.name, pattern)


def _require_directory_root(root: Path) -> None:
    try:
        status = root.lstat()
    except OSError:
        raise
    if _entry_from_status(root, status, root, include_root=True) is None:
        raise NotADirectoryError(f"The requested path is not a directory: {root}")


def _entry_from_status(
    path: Path,
    status: os.stat_result,
    root: Path,
    *,
    include_root: bool = False,
) -> DirectoryEntry | None:
    is_link = _is_link(status)
    if is_link:
        is_directory = _link_is_directory(path, status)
        if include_root and not is_directory:
            return None
        return DirectoryEntry(
            path=path,
            relative=PurePosixPath(path.relative_to(root).as_posix())
            if path != root
            else PurePosixPath("."),
            is_directory=is_directory,
            is_link=True,
        )
    if HOST_FILESYSTEM.is_directory(status):
        return DirectoryEntry(
            path=path,
            relative=PurePosixPath(path.relative_to(root).as_posix())
            if path != root
            else PurePosixPath("."),
            is_directory=True,
            is_link=False,
        )
    if include_root or not HOST_FILESYSTEM.is_regular_file(status):
        return None
    return DirectoryEntry(
        path=path,
        relative=PurePosixPath(path.relative_to(root).as_posix()),
        is_directory=False,
        is_link=False,
    )


def _is_link(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & _REPARSE_POINT)


def _link_is_directory(path: Path, status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    if attributes & _DIRECTORY_ATTRIBUTE:
        return True
    try:
        return path.is_dir()
    except OSError:
        return False


def is_ignored_directory_name(name: str) -> bool:
    """Return whether one directory basename belongs to the fixed ignore set."""
    return os.path.normcase(name) in _NORMALIZED_IGNORED_DIRECTORY_NAMES


__all__ = [
    "DirectoryEntry",
    "is_ignored_directory_name",
    "iter_directory_entries",
    "matches_glob_pattern",
    "normalize_glob_pattern",
    "report_path",
    "requested_path_has_directory_link",
]
