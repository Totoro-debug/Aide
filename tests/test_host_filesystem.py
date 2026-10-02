import errno
import os
import subprocess
import sys
import time
from pathlib import Path
from stat import S_IFLNK
from types import SimpleNamespace
from typing import cast

import pytest

from myclaw.utils.host_filesystem import (
    HOST_FILESYSTEM,
    WINDOWS_HOST_FILESYSTEM,
    host_path_is_within,
)

_LOCK_PROCESS_SCRIPT = """
import sys
from pathlib import Path

from myclaw.utils.host_filesystem import HOST_FILESYSTEM

print("started", flush=True)
try:
    with HOST_FILESYSTEM.exclusive_lock(Path(sys.argv[1]), timeout=float(sys.argv[2])):
        pass
except TimeoutError:
    print("timeout", flush=True)
else:
    print("acquired", flush=True)
"""


def _lock_process_command(lock_path: Path, timeout: float) -> list[str]:
    return [sys.executable, "-c", _LOCK_PROCESS_SCRIPT, str(lock_path), str(timeout)]


def test_host_path_is_within_accepts_child_and_rejects_sibling_prefix(tmp_path: Path) -> None:
    root = tmp_path / "workspace"

    assert host_path_is_within(root / "child", root)
    assert not host_path_is_within(tmp_path / "workspace-copy" / "child", root)


def test_host_path_is_within_uses_host_case_rules(tmp_path: Path) -> None:
    root = tmp_path / "workspace"

    assert host_path_is_within(Path(str(root).swapcase()) / "child", root)


def test_host_path_is_within_rejects_incompatible_drives() -> None:
    assert not host_path_is_within(Path("C:/workspace/child"), Path("D:/workspace"))


def test_windows_host_filesystem_prepares_local_and_unc_io_paths(tmp_path: Path) -> None:
    local = tmp_path / "state.txt"
    unc = Path(r"\\server\share\state.txt")

    assert WINDOWS_HOST_FILESYSTEM.path_for_io(local) == Path(f"\\\\?\\{local.absolute()}")
    assert WINDOWS_HOST_FILESYSTEM.path_for_io(unc) == Path(r"\\?\UNC\server\share\state.txt")
    assert WINDOWS_HOST_FILESYSTEM.path_for_io(WINDOWS_HOST_FILESYSTEM.path_for_io(local)) == (
        WINDOWS_HOST_FILESYSTEM.path_for_io(local)
    )


def test_host_entry_exists_for_existing_and_missing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("content", encoding="utf-8")

    assert HOST_FILESYSTEM.entry_exists(existing)
    assert not HOST_FILESYSTEM.entry_exists(tmp_path / "missing.txt")


def test_host_entry_exists_keeps_dangling_link_and_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dangling = HOST_FILESYSTEM.path_for_io(tmp_path / "dangling")
    denied = HOST_FILESYSTEM.path_for_io(tmp_path / "denied")
    permission_error = PermissionError(errno.EACCES, "denied", str(denied))
    original_lstat = Path.lstat

    def controlled_lstat(path: Path) -> os.stat_result:
        if path == dangling:
            return os.stat_result((S_IFLNK | 0o777, 1, 1, 1, 0, 0, 0, 0, 0, 0))
        if path == denied:
            raise permission_error
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", controlled_lstat)

    assert HOST_FILESYSTEM.entry_exists(dangling)
    with pytest.raises(PermissionError) as captured:
        HOST_FILESYSTEM.entry_exists(denied)
    assert captured.value is permission_error


def test_windows_host_filesystem_accepts_an_owned_directory(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    child = owned / "child"
    child.mkdir(parents=True)

    assert WINDOWS_HOST_FILESYSTEM.require_owned_directory(child, within=owned) == child.resolve(
        strict=True
    )


def test_windows_host_filesystem_rejects_redirected_or_external_directory(
    tmp_path: Path,
) -> None:
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

    for candidate in (junction, outside):
        with pytest.raises(PermissionError):
            WINDOWS_HOST_FILESYSTEM.require_owned_directory(candidate, within=owned)


def test_windows_host_filesystem_accepts_an_owned_regular_file(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    state = owned / "state.json"
    state.write_bytes(b"{}")

    assert WINDOWS_HOST_FILESYSTEM.require_owned_regular_file(state, within=owned) == state.resolve(
        strict=True
    )


def test_host_filesystem_create_only_publication_preserves_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.txt"

    assert HOST_FILESYSTEM.atomic_create_text(target, "first\n") is True
    assert HOST_FILESYSTEM.atomic_create_text(target, "replacement\n") is False
    assert target.read_bytes() == b"first\n"
    assert tuple(tmp_path.iterdir()) == (target,)


def test_host_filesystem_atomic_replace_publishes_exact_utf8_content(tmp_path: Path) -> None:
    target = tmp_path / "state.txt"
    target.write_bytes(b"old")

    HOST_FILESYSTEM.atomic_replace_text(target, "User: \u5f20\u4e09\nPreference: caf\u00e9\n")

    assert target.read_bytes() == (b"User: \xe5\xbc\xa0\xe4\xb8\x89\nPreference: caf\xc3\xa9\n")


def test_host_filesystem_exclusive_lock_rejects_hard_link_without_modifying_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"")
    lock_path = tmp_path / ".config.toml.lock"
    lock_path.hardlink_to(target)

    with pytest.raises(PermissionError):
        with HOST_FILESYSTEM.exclusive_lock(lock_path):
            pass

    assert target.read_bytes() == b""


@pytest.mark.parametrize(
    "timeout",
    [-0.1, float("nan"), float("inf")],
    ids=("negative", "nan", "infinite"),
)
def test_host_filesystem_exclusive_lock_rejects_invalid_timeout_before_file_creation(
    tmp_path: Path,
    timeout: float,
) -> None:
    lock_path = tmp_path / ".config.toml.lock"

    with pytest.raises(ValueError, match="finite non-negative"):
        with HOST_FILESYSTEM.exclusive_lock(lock_path, timeout=timeout):
            pass

    assert not lock_path.exists()


def test_host_filesystem_exclusive_lock_blocks_another_process(tmp_path: Path) -> None:
    lock_path = tmp_path / ".config.toml.lock"

    with HOST_FILESYSTEM.exclusive_lock(lock_path, timeout=1.0):
        contender = subprocess.Popen(
            _lock_process_command(lock_path, 2.0),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert contender.stdout is not None
        assert contender.stdout.readline().strip() == "started"
        time.sleep(0.05)
        assert contender.poll() is None

    stdout, stderr = contender.communicate(timeout=5)

    assert contender.returncode == 0, stderr
    assert stdout.strip() == "acquired"


def test_host_filesystem_exclusive_lock_times_out_while_owned(tmp_path: Path) -> None:
    lock_path = tmp_path / ".config.toml.lock"

    with HOST_FILESYSTEM.exclusive_lock(lock_path, timeout=1.0):
        contender = subprocess.run(
            _lock_process_command(lock_path, 0.05),
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )

    assert contender.stdout.splitlines() == ["started", "timeout"]


def test_host_filesystem_exclusive_lock_releases_after_body_failure(tmp_path: Path) -> None:
    lock_path = tmp_path / ".config.toml.lock"

    with pytest.raises(RuntimeError, match="injected body failure"):
        with HOST_FILESYSTEM.exclusive_lock(lock_path, timeout=1.0):
            raise RuntimeError("injected body failure")

    contender = subprocess.run(
        _lock_process_command(lock_path, 0.2),
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )

    assert contender.stdout.splitlines() == ["started", "acquired"]
    assert lock_path.is_file()
    assert lock_path.read_bytes() == b""


def test_windows_host_filesystem_rejects_an_open_file_with_mismatched_path(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    opened_path = owned / "opened.log"
    opened_path.write_bytes(b"opened")
    current_path = owned / "current.log"
    current_path.write_bytes(b"current")

    with opened_path.open("rb", buffering=0) as stream:
        with pytest.raises(PermissionError):
            WINDOWS_HOST_FILESYSTEM.require_opened_owned_regular_file(
                stream.fileno(), current_path, within=owned
            )


def test_host_filesystem_applies_only_native_reserved_component_rules() -> None:
    assert WINDOWS_HOST_FILESYSTEM.is_reserved_component("CON.txt")
    assert WINDOWS_HOST_FILESYSTEM.has_alternate_data_stream("state.json:secret")


def test_windows_filesystem_rejects_injected_reparse_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = tmp_path / "owned"
    redirected = owned / "redirected"
    redirected.mkdir(parents=True)
    original_lstat = Path.lstat

    def injected_lstat(path: Path) -> os.stat_result:
        if path == redirected:
            return cast(os.stat_result, SimpleNamespace(st_file_attributes=0x410))
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", injected_lstat)
    with pytest.raises(PermissionError):
        HOST_FILESYSTEM.require_owned_directory(redirected, within=owned)


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.EIO, errno.EACCES])
def test_windows_file_sync_ignores_only_unsupported_sync(
    monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    def failed_sync(descriptor: int) -> None:
        raise OSError(error_number, "injected sync failure")

    monkeypatch.setattr(os, "fsync", failed_sync)
    if error_number == errno.EINVAL:
        HOST_FILESYSTEM.sync_file(51)
    else:
        with pytest.raises(OSError) as error:
            HOST_FILESYSTEM.sync_file(51)
        assert error.value.errno == error_number
