"""Runtime-owned permission state shared by foreground generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

type ToolPermissionLevel = Literal["read-only", "workspace-write", "full-access"]

PERMISSION_LEVELS: tuple[ToolPermissionLevel, ...] = (
    "read-only",
    "workspace-write",
    "full-access",
)


class PermissionExecShell(Protocol):
    """The resolved shell identity needed in a run snapshot."""

    @property
    def family(self) -> str: ...


def validate_permission_level(level: object) -> ToolPermissionLevel:
    if level not in PERMISSION_LEVELS:
        raise ValueError("Tool permission level is invalid")
    return level


@dataclass(frozen=True, slots=True)
class PermissionSnapshot:
    """Immutable permission values captured for one Agent Run."""

    level: ToolPermissionLevel
    exec_shell: PermissionExecShell

    def __post_init__(self) -> None:
        validate_permission_level(self.level)
        if not isinstance(getattr(self.exec_shell, "family", None), str):
            raise TypeError("Permission snapshots require a resolved Exec shell")


class RuntimePermissionControl:
    """Runtime-owned permission state confined to the CLI event-loop thread.

    Selection and snapshot capture are synchronous, so an async caller cannot
    observe a partial update across an ``await`` boundary.
    """

    def __init__(self, configured_level: ToolPermissionLevel) -> None:
        self._configured_level = validate_permission_level(configured_level)
        self._current_level = self._configured_level

    def configured(self) -> ToolPermissionLevel:
        return self._configured_level

    def current(self) -> ToolPermissionLevel:
        return self._current_level

    def select(self, level: ToolPermissionLevel) -> None:
        self._current_level = validate_permission_level(level)

    def snapshot(self, exec_shell: PermissionExecShell) -> PermissionSnapshot:
        return PermissionSnapshot(level=self._current_level, exec_shell=exec_shell)


__all__ = [
    "PERMISSION_LEVELS",
    "PermissionExecShell",
    "PermissionSnapshot",
    "RuntimePermissionControl",
    "ToolPermissionLevel",
    "validate_permission_level",
]
