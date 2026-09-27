"""Temporary canonical paths for filesystem-isolated tests."""

from pathlib import Path

import pytest

from myclaw.workspace.state import normalize_workspace_path


@pytest.fixture
def agent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return an absent fixed Agent Home beneath a disposable user home."""
    user_home = (tmp_path / "user-home").resolve()
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    return user_home / ".myclaw"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return an existing normalized Workspace Path beneath the test directory."""
    return create_workspace(tmp_path)


def create_workspace(parent: Path) -> Path:
    """Create an existing normalized Workspace Path beneath the given directory."""
    path = normalize_workspace_path(parent / "workspace")
    path.mkdir()
    return path
