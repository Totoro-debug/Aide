from __future__ import annotations

from collections.abc import Callable

import pytest

import myclaw.session.restore as restore_module
from myclaw.workspace.state import WorkspaceState

AfterRestorePhase = Callable[[str, Callable[[], None]], None]

_PHASE_NAMES = {
    restore_module._RestorePhase.PREPARED: "pending_intent",
    restore_module._RestorePhase.JOURNAL_PRUNED: "journal_pruned",
    restore_module._RestorePhase.FILE_INTENT: "file_intent",
    restore_module._RestorePhase.FILE_REPLAY: "file_result",
    restore_module._RestorePhase.FILES_REPLAYED: "files_replayed",
    restore_module._RestorePhase.SESSION_WRITE: "session_write",
    restore_module._RestorePhase.SESSION_PERSISTED: "session_persisted",
    restore_module._RestorePhase.COMPLETE: "complete",
}


@pytest.fixture
def after_restore_phase(monkeypatch: pytest.MonkeyPatch) -> AfterRestorePhase:
    original_write_pending = restore_module._write_pending
    installed = False

    def install(expected_phase: str, action: Callable[[], None]) -> None:
        nonlocal installed
        if installed:
            raise RuntimeError("after_restore_phase can only be installed once per test")
        if expected_phase not in _PHASE_NAMES.values():
            raise ValueError(f"unknown restore phase: {expected_phase}")
        installed = True
        fired = False

        def write_pending(
            workspace_state: WorkspaceState,
            pending: restore_module._PendingTransaction,
        ) -> None:
            nonlocal fired
            original_write_pending(workspace_state, pending)
            if not fired and _PHASE_NAMES[pending.phase] == expected_phase:
                fired = True
                action()

        monkeypatch.setattr(restore_module, "_write_pending", write_pending)

    return install
