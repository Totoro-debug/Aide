"""Helpers for recording one authorized Built-in File Tool mutation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

from myclaw.agent.session.backup_store import FileMutationRecorder
from myclaw.utils.host_filesystem import host_path_is_within


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
        ticket = recorder.before_write(run_token, target)
    except Exception:
        ticket = None
    try:
        return await mutation()
    finally:
        try:
            recorder.after_write(ticket)
        except Exception:
            pass


def is_protected_restore_target(workspace: Path, target: Path) -> bool:
    """Return whether a canonical target is inside protected restore state."""
    protected_root = (workspace / ".myclaw" / "restore").resolve(strict=False)
    return host_path_is_within(target, protected_root)


__all__ = ["execute_recorded_mutation", "is_protected_restore_target"]
