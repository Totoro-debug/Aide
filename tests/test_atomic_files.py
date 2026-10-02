import asyncio
import os
from pathlib import Path

import pytest

from myclaw.utils.host_filesystem import HOST_FILESYSTEM


def test_path_for_io_normalizes_windows_local_and_unc_paths(tmp_path: Path) -> None:
    local = tmp_path / "state.txt"
    unc = Path(r"\\server\share\state.txt")

    assert HOST_FILESYSTEM.path_for_io(local) == Path(f"\\\\?\\{local.absolute()}")
    assert HOST_FILESYSTEM.path_for_io(unc) == Path(r"\\?\UNC\server\share\state.txt")


def test_path_for_io_preserves_existing_windows_extended_path(tmp_path: Path) -> None:
    extended = Path(f"\\\\?\\{tmp_path.absolute()}\\state.txt")

    assert HOST_FILESYSTEM.path_for_io(extended) == extended


def test_failed_atomic_bytes_replace_preserves_official_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old-state")

    def fail_replace(source: Path | str, destination: Path | str) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        HOST_FILESYSTEM.atomic_replace_bytes(target, b"new-state")

    assert (target.read_bytes(), sorted(path.name for path in tmp_path.iterdir())) == (
        b"old-state",
        ["state.bin"],
    )


def test_atomic_text_replace_writes_exact_utf8_bytes(tmp_path: Path) -> None:
    target = tmp_path / "state.txt"

    HOST_FILESYSTEM.atomic_replace_text(target, "User: \u5f20\u4e09\nPreference: caf\u00e9\n")

    assert target.read_bytes() == (b"User: \xe5\xbc\xa0\xe4\xb8\x89\nPreference: caf\xc3\xa9\n")


def test_cancelled_atomic_bytes_replace_preserves_official_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old-state")

    def cancel_fsync(descriptor: int) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(os, "fsync", cancel_fsync)

    with pytest.raises(asyncio.CancelledError):
        HOST_FILESYSTEM.atomic_replace_bytes(target, b"new-state")

    assert (target.read_bytes(), sorted(path.name for path in tmp_path.iterdir())) == (
        b"old-state",
        ["state.bin"],
    )


def test_atomic_create_and_replace_use_windows_extended_paths(tmp_path: Path) -> None:
    parent = tmp_path.joinpath(*(["nested-state-directory"] * 12))
    HOST_FILESYSTEM.path_for_io(parent).mkdir(parents=True)
    target = parent / "state.txt"

    assert HOST_FILESYSTEM.atomic_create_text(target, "first\n") is True
    assert HOST_FILESYSTEM.atomic_create_text(target, "must not replace\n") is False
    HOST_FILESYSTEM.atomic_replace_text(target, "second\n")

    assert HOST_FILESYSTEM.path_for_io(target).read_bytes() == b"second\n"
