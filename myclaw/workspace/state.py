"""Workspace-owned persistent state boundary and startup initialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Final

from myclaw.errors import ErrorInfo
from myclaw.templates import load_template
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_GITIGNORE_CONTENT: Final = "*\n"
_LONG_TERM_MEMORY_TEMPLATE: Final = load_template("long-term-memory.md")


def normalize_workspace_path(path: Path | PurePath) -> Path:
    """Return the normalized absolute Workspace path without resolving aliases."""
    if isinstance(path, Path):
        normalized = Path(os.path.abspath(path))
    else:
        normalized = Path(os.path.normpath(path))

    if not normalized.is_absolute():
        raise ValueError("Workspace path must be absolute")

    return normalized


class WorkspaceStateError(Exception):
    """A safe startup failure tied to one Workspace State path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.error = ErrorInfo(
            "persistence_error",
            "Workspace State could not be initialized at the reserved path.",
        )
        super().__init__(self.error.message)


class _UnsafeStatePath(PermissionError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__("Workspace State path is unavailable or unsafe")


@dataclass(frozen=True, slots=True)
class WorkspaceState:
    """Canonical persistent-state paths owned by one Workspace."""

    workspace_path: Path

    def __post_init__(self) -> None:
        normalized = normalize_workspace_path(self.workspace_path)
        object.__setattr__(self, "workspace_path", normalized)

    @property
    def path(self) -> Path:
        return self.workspace_path / ".myclaw"

    @property
    def memory_directory(self) -> Path:
        return self.path / "memory"

    @property
    def sessions_directory(self) -> Path:
        return self.path / "sessions"

    @property
    def schedule_sessions_directory(self) -> Path:
        """Dedicated, lazily-created storage for Schedule Sessions."""
        return self.path / "schedule-sessions"

    @property
    def logs_directory(self) -> Path:
        """Canonical, lazily-created Workspace-owned Session Log directory."""
        return self.path / "logs"

    @property
    def schedule_path(self) -> Path:
        """Canonical Schedule Job state path."""
        return self.path / "schedule.json"

    @property
    def long_term_memory_path(self) -> Path:
        return self.memory_directory / "memory.md"

    def existing_sessions_directory(self) -> Path | None:
        """Return the validated sessions directory without materializing state."""
        return self._existing_sessions_directory(self.sessions_directory)

    def existing_schedule_sessions_directory(self) -> Path | None:
        """Return the validated Schedule Session directory without materializing state."""
        return self._existing_sessions_directory(self.schedule_sessions_directory)

    def _existing_sessions_directory(self, path: Path) -> Path | None:
        workspace_root = self._owned_workspace_root()
        state_root = self._existing_owned_directory(self.path, within=workspace_root)
        if state_root is None:
            return None
        return self._existing_owned_directory(path, within=state_root)

    def prepare_sessions_directory(self) -> Path:
        """Lazily prepare and validate the owned sessions directory for writes."""
        return self._prepare_sessions_directory(self.sessions_directory)

    def prepare_schedule_sessions_directory(self) -> Path:
        """Lazily prepare and validate the owned Schedule Session directory for writes."""
        return self._prepare_sessions_directory(self.schedule_sessions_directory)

    def _prepare_sessions_directory(self, path: Path) -> Path:
        workspace_root = self._owned_workspace_root()
        state_path = HOST_FILESYSTEM.path_for_io(self.path)
        if not _path_entry_exists(state_path):
            state_path.mkdir(exist_ok=True)
        state_root = HOST_FILESYSTEM.require_owned_directory(
            state_path,
            within=workspace_root,
        )

        sessions_path = HOST_FILESYSTEM.path_for_io(path)
        if not _path_entry_exists(sessions_path):
            sessions_path.mkdir(exist_ok=True)
        return HOST_FILESYSTEM.require_owned_directory(sessions_path, within=state_root)

    def _owned_workspace_root(self) -> Path:
        workspace_path = HOST_FILESYSTEM.path_for_io(self.workspace_path)
        return HOST_FILESYSTEM.require_owned_directory(workspace_path, within=workspace_path)

    @staticmethod
    def _existing_owned_directory(path: Path, *, within: Path) -> Path | None:
        io_path = HOST_FILESYSTEM.path_for_io(path)
        if not _path_entry_exists(io_path):
            return None
        return HOST_FILESYSTEM.require_owned_directory(io_path, within=within)

    def initialize(self, *, agent_home_root: Path) -> None:
        """Create required base state while rejecting redirected known paths."""
        try:
            normalized_agent_home = agent_home_root.resolve(strict=False)
            normalized_state_root = self.path.resolve(strict=False)
            if normalized_state_root.is_relative_to(
                normalized_agent_home
            ) or normalized_agent_home.is_relative_to(normalized_state_root):
                raise _UnsafeStatePath(self.path)

            workspace_root = self.workspace_path.resolve(strict=True)
            if not workspace_root.is_dir():
                raise _UnsafeStatePath(self.workspace_path)

            self.path.mkdir(exist_ok=True)
            state_root = HOST_FILESYSTEM.require_owned_directory(self.path, within=workspace_root)

            # Publication is create-only, so an existing policy is never read or repaired.
            HOST_FILESYSTEM.atomic_create_text(self.path / ".gitignore", _GITIGNORE_CONTENT)

            self.memory_directory.mkdir(exist_ok=True)
            HOST_FILESYSTEM.require_owned_directory(self.memory_directory, within=state_root)
            self.sessions_directory.mkdir(exist_ok=True)
            HOST_FILESYSTEM.require_owned_directory(self.sessions_directory, within=state_root)

            HOST_FILESYSTEM.atomic_create_text(
                self.long_term_memory_path, _LONG_TERM_MEMORY_TEMPLATE
            )
            HOST_FILESYSTEM.require_owned_regular_file(
                self.long_term_memory_path, within=state_root
            )
        except WorkspaceStateError:
            raise
        except _UnsafeStatePath as error:
            raise WorkspaceStateError(error.path) from error
        except OSError as error:
            affected = Path(error.filename) if isinstance(error.filename, str) else self.path
            raise WorkspaceStateError(affected) from error
        except RuntimeError as error:
            raise WorkspaceStateError(self.path) from error


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True
