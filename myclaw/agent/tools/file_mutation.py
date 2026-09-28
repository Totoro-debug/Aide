"""File Tool recording interface and authorized mutation helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from uuid import UUID

from myclaw.utils.host_filesystem import host_path_is_within


class FileMutationRecorder(Protocol):
    """Bracket one authorized file mutation without exposing backup state."""

    def begin_write(self, run_token: UUID, resolved_target: Path) -> Callable[[], None]: ...


async def execute_recorded_mutation(
    mutation: Callable[[], Awaitable[str]],
    *,
    recorder: FileMutationRecorder | None,
    run_token: UUID | None,
    target: Path,
) -> str:
    """Record one mutation attempt without changing its execution contract."""
    if recorder is None or run_token is None:
        return await mutation()

    try:
        complete = recorder.begin_write(run_token, target)
    except Exception:
        complete = None
    try:
        return await mutation()
    finally:
        if complete is not None:
            try:
                complete()
            except Exception:
                pass


def is_protected_restore_target(workspace: Path, target: Path) -> bool:
    """Return whether a canonical target is inside protected restore state."""
    protected_root = (workspace / ".myclaw" / "restore").resolve(strict=False)
    return host_path_is_within(target, protected_root)


__all__ = ["FileMutationRecorder", "execute_recorded_mutation", "is_protected_restore_target"]
