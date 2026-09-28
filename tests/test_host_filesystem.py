import errno
import os
import subprocess
import sys
import time
from os import stat_result
from pathlib import Path
from stat import S_IFDIR, S_IFIFO, S_IFLNK, S_IFREG

import pytest

from myclaw.utils.host_filesystem import (
    HOST_FILESYSTEM,
    POSIX_HOST_FILESYSTEM,
    WINDOWS_HOST_FILESYSTEM,
    PosixFilesystemAdapter,
    host_path_is_within,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="requires native Windows paths")

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


@windows_only
def test_host_path_is_within_uses_host_case_rules(tmp_path: Path) -> None:
    root = tmp_path / "workspace"

    assert host_path_is_within(Path(str(root).swapcase()) / "child", root)


@windows_only
def test_host_path_is_within_rejects_incompatible_drives() -> None:
    assert not host_path_is_within(Path("C:/workspace/child"), Path("D:/workspace"))


@windows_only
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


@windows_only
def test_windows_host_filesystem_accepts_an_owned_directory(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    child = owned / "child"
    child.mkdir(parents=True)

    assert WINDOWS_HOST_FILESYSTEM.require_owned_directory(child, within=owned) == child.resolve(
        strict=True
    )


@windows_only
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


@windows_only
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


@windows_only
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


def test_posix_host_filesystem_classifies_only_ordinary_directories_and_files() -> None:
    directory = stat_result((S_IFDIR | 0o700, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    regular_file = stat_result((S_IFREG | 0o600, 2, 1, 1, 0, 0, 0, 0, 0, 0))
    symbolic_link = stat_result((S_IFLNK | 0o777, 3, 1, 1, 0, 0, 0, 0, 0, 0))
    fifo = stat_result((S_IFIFO | 0o600, 4, 1, 1, 0, 0, 0, 0, 0, 0))

    assert POSIX_HOST_FILESYSTEM.is_directory(directory)
    assert POSIX_HOST_FILESYSTEM.is_regular_file(regular_file)
    assert not POSIX_HOST_FILESYSTEM.is_directory(symbolic_link)
    assert not POSIX_HOST_FILESYSTEM.is_regular_file(symbolic_link)
    assert not POSIX_HOST_FILESYSTEM.is_regular_file(fifo)


def test_host_filesystem_applies_only_native_reserved_component_rules() -> None:
    assert WINDOWS_HOST_FILESYSTEM.is_reserved_component("CON.txt")
    assert WINDOWS_HOST_FILESYSTEM.has_alternate_data_stream("state.json:secret")
    assert not POSIX_HOST_FILESYSTEM.is_reserved_component("CON.txt")
    assert not POSIX_HOST_FILESYSTEM.has_alternate_data_stream("state.json:secret")


def test_posix_host_filesystem_rejects_hard_linked_and_escaping_files(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    ordinary = owned / "ordinary.json"
    ordinary.write_bytes(b"{}")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    hard_link = owned / "hard-link.json"
    hard_link.hardlink_to(outside)

    assert POSIX_HOST_FILESYSTEM.require_owned_regular_file(
        ordinary, within=owned
    ) == ordinary.resolve(strict=True)
    for candidate in (hard_link, outside):
        with pytest.raises(PermissionError):
            POSIX_HOST_FILESYSTEM.require_owned_regular_file(candidate, within=owned)


def test_posix_host_filesystem_rejects_injected_symbolic_link_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tmp_path / "owned"
    redirected = owned / "redirected"
    redirected.mkdir(parents=True)
    symbolic_link = stat_result((S_IFLNK | 0o777, 3, 1, 1, 0, 0, 0, 0, 0, 0))
    original_lstat = Path.lstat

    def injected_lstat(path: Path) -> stat_result:
        if path == redirected:
            return symbolic_link
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", injected_lstat)

    with pytest.raises(PermissionError):
        POSIX_HOST_FILESYSTEM.require_owned_directory(redirected, within=owned)


@pytest.mark.parametrize(
    "unsupported_errno",
    (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    ),
)
def test_posix_parent_sync_ignores_only_unsupported_open_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno: int,
) -> None:
    def unsupported_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        del path, flags, mode
        raise OSError(unsupported_errno, "directory sync is unsupported")

    monkeypatch.setattr(os, "open", unsupported_open)

    PosixFilesystemAdapter().sync_parent_directory(tmp_path)


def test_posix_parent_sync_ignores_an_unsupported_fsync_error_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(os, "open", lambda *_args: 51)
    monkeypatch.setattr(
        os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EINVAL, "unsupported")),
    )
    monkeypatch.setattr(os, "close", closed.append)

    PosixFilesystemAdapter().sync_parent_directory(tmp_path)

    assert closed == [51]


@pytest.mark.parametrize("operation", ("open", "fsync", "close"))
def test_posix_parent_sync_propagates_real_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    if operation == "open":
        monkeypatch.setattr(
            os,
            "open",
            lambda *_args: (_ for _ in ()).throw(OSError(errno.EACCES, "denied")),
        )
    else:
        monkeypatch.setattr(os, "open", lambda *_args: 52)
        monkeypatch.setattr(os, "fsync", lambda _descriptor: None)
        monkeypatch.setattr(os, "close", lambda _descriptor: None)
        monkeypatch.setattr(
            os,
            operation,
            lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "io failure")),
        )

    with pytest.raises(OSError) as error:
        PosixFilesystemAdapter().sync_parent_directory(tmp_path)

    assert error.value.errno in {errno.EACCES, errno.EIO}


def test_posix_parent_sync_failure_surfaces_after_complete_atomic_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.txt"
    target.write_bytes(b"old")
    original_open = os.open

    def reject_parent_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(os.fsdecode(path)) == tmp_path:
            raise OSError(errno.EIO, "injected directory open failure")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", reject_parent_open)

    with pytest.raises(OSError, match="injected directory open failure"):
        POSIX_HOST_FILESYSTEM.atomic_replace_text(target, "complete\n")

    assert target.read_bytes() == b"complete\n"
    assert tuple(tmp_path.iterdir()) == (target,)
