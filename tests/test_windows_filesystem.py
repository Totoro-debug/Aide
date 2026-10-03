import subprocess
from pathlib import Path

import pytest

from omni.utils.host_filesystem import HOST_FILESYSTEM


def test_require_owned_directory_returns_normalized_owned_path(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    child = owned / "child"
    child.mkdir(parents=True)

    assert HOST_FILESYSTEM.require_owned_directory(child, within=owned) == child.resolve(
        strict=True
    )


def test_require_owned_directory_rejects_junction_and_external_paths(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = owned / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=True,
        text=True,
    )

    with pytest.raises(PermissionError):
        HOST_FILESYSTEM.require_owned_directory(junction, within=owned)
    with pytest.raises(PermissionError):
        HOST_FILESYSTEM.require_owned_directory(outside, within=owned)


def test_require_owned_regular_file_returns_normalized_owned_path(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    file = owned / "state.json"
    file.write_bytes(b"{}")

    assert HOST_FILESYSTEM.require_owned_regular_file(file, within=owned) == file.resolve(
        strict=True
    )


def test_require_owned_regular_file_rejects_directories_and_hard_links(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    directory = owned / "state.json"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"private")
    hard_link = owned / "linked.json"
    hard_link.hardlink_to(outside)

    with pytest.raises(PermissionError):
        HOST_FILESYSTEM.require_owned_regular_file(directory, within=owned)
    with pytest.raises(PermissionError):
        HOST_FILESYSTEM.require_owned_regular_file(hard_link, within=owned)

    assert outside.read_bytes() == b"private"
