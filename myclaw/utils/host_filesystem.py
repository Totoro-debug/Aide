"""Host filesystem operations for safe local persistence."""

from __future__ import annotations

import ctypes
import errno
import importlib
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from errno import EACCES
from math import isfinite
from os import stat_result
from pathlib import Path
from stat import (
    FILE_ATTRIBUTE_DEVICE,
    FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_REPARSE_POINT,
)
from typing import Any, Final, NoReturn, Protocol, cast

type FileIdentity = tuple[int, int, int, int]

_WINDOWS_RESERVED_BASENAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_LOCK_RETRY_INTERVAL_SECONDS: Final = 0.01


def host_path_is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(str(path)), os.path.normcase(str(root)))
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


class UnsafeFilesystemPath(PermissionError):
    """An existing filesystem object violates owned-path safety rules."""


class FilesystemAdapter(Protocol):
    """Native path behavior required by the host filesystem seam."""

    def path_for_io(self, path: Path) -> Path: ...

    def is_directory(self, status: stat_result) -> bool: ...

    def is_regular_file(self, status: stat_result) -> bool: ...

    def resolved_for_comparison(self, path: Path) -> Path: ...

    def is_reserved_component(self, component: str) -> bool: ...

    def has_alternate_data_stream(self, component: str) -> bool: ...

    def accepts_native_executable_name(self, path: Path) -> bool: ...

    def sync_file(self, descriptor: int) -> None: ...

    def sync_parent_directory(self, path: Path) -> None: ...

    def restrict_private_directory(self, path: Path) -> None: ...

    def restrict_private_file(self, path: Path) -> None: ...

    def protect_private_file(self, path: Path) -> None: ...

    def restrict_private_descriptor(self, descriptor: int) -> None: ...

    def try_lock_exclusive(self, descriptor: int) -> bool: ...

    def unlock(self, descriptor: int) -> None: ...

    def remove_owned_entry(self, path: Path, *, root: Path, tree: bool) -> None: ...


class WindowsFilesystemAdapter:
    """Windows native path behavior."""

    def path_for_io(self, path: Path) -> Path:
        native = str(path.absolute())
        if native.startswith("\\\\?\\"):
            return path
        if native.startswith("\\\\"):
            return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
        return Path(f"\\\\?\\{native}")

    def remove_owned_entry(self, path: Path, *, root: Path, tree: bool) -> None:
        from myclaw.utils._owned_deletion import remove_owned_windows

        remove_owned_windows(path, root=root, tree=tree)

    def is_directory(self, status: stat_result) -> bool:
        attributes = getattr(status, "st_file_attributes", 0)
        return bool(attributes & FILE_ATTRIBUTE_DIRECTORY) and not bool(
            attributes & FILE_ATTRIBUTE_REPARSE_POINT
        )

    def is_regular_file(self, status: stat_result) -> bool:
        non_regular = (
            FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT
        )
        return not bool(getattr(status, "st_file_attributes", 0) & non_regular)

    def resolved_for_comparison(self, path: Path) -> Path:
        resolved = path.resolve(strict=True)
        native = str(resolved)
        if native.startswith("\\\\?\\UNC\\"):
            return Path(f"\\\\{native.removeprefix('\\\\?\\UNC\\')}")
        return Path(native.removeprefix("\\\\?\\"))

    def is_reserved_component(self, component: str) -> bool:
        normalized = component.rstrip(" .")
        basename = normalized.split(".", maxsplit=1)[0].upper()
        return basename in _WINDOWS_RESERVED_BASENAMES

    def has_alternate_data_stream(self, component: str) -> bool:
        return ":" in component

    def accepts_native_executable_name(self, path: Path) -> bool:
        return path.suffix.casefold() == ".exe"

    def sync_file(self, descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise

    def sync_parent_directory(self, path: Path) -> None:
        del path

    def restrict_private_directory(self, path: Path) -> None:
        del path

    def restrict_private_file(self, path: Path) -> None:
        del path

    def protect_private_file(self, path: Path) -> None:
        from ctypes import wintypes

        security = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        convert.restype = wintypes.BOOL
        set_security = security.SetFileSecurityW
        set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        set_security.restype = wintypes.BOOL
        free = kernel.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        descriptor = ctypes.c_void_p()
        # A protected ACL grants full access only to the file owner and SYSTEM.
        if not convert("D:P(A;;FA;;;OW)(A;;FA;;;SY)", 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not set_security(str(path), 0x80000004, descriptor):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            free(descriptor)

    def restrict_private_descriptor(self, descriptor: int) -> None:
        del descriptor

    def try_lock_exclusive(self, descriptor: int) -> bool:
        msvcrt = cast(Any, importlib.import_module("msvcrt"))

        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    def unlock(self, descriptor: int) -> None:
        msvcrt = cast(Any, importlib.import_module("msvcrt"))

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


class HostFilesystem:
    """Deep boundary for native persistent filesystem behavior."""

    def __init__(self, adapter: FilesystemAdapter) -> None:
        self._adapter = adapter

    def path_for_io(self, path: Path) -> Path:
        """Return the host-native path used for filesystem I/O."""
        return self._adapter.path_for_io(Path(path))

    def entry_exists(self, path: Path) -> bool:
        """Count any filesystem entry, including a dangling symbolic link."""
        try:
            self.path_for_io(path).lstat()
        except FileNotFoundError:
            return False
        return True

    def is_directory(self, status: stat_result) -> bool:
        """Return whether a status identifies an ordinary host directory."""
        return self._adapter.is_directory(status)

    def is_regular_file(self, status: stat_result) -> bool:
        """Return whether a status identifies an ordinary host regular file."""
        return self._adapter.is_regular_file(status)

    def is_reserved_component(self, component: str) -> bool:
        """Return whether a native path component names a reserved object."""
        return self._adapter.is_reserved_component(component)

    def has_alternate_data_stream(self, component: str) -> bool:
        """Return whether a component uses native alternate-stream syntax."""
        return self._adapter.has_alternate_data_stream(component)

    def accepts_native_executable_name(self, path: Path) -> bool:
        """Return whether a path uses the host's native executable naming convention."""
        return self._adapter.accepts_native_executable_name(path)

    def sync_file(self, descriptor: int) -> None:
        """Synchronize file content with host-appropriate compatibility behavior."""
        self._adapter.sync_file(descriptor)

    def sync_parent_directory(self, path: Path) -> None:
        """Synchronize a directory after removing or publishing an entry."""
        self._adapter.sync_parent_directory(self.path_for_io(path))

    def remove_owned_entry(self, path: Path, *, root: Path, tree: bool) -> None:
        """Delete below a pinned root without following replacement directory aliases."""
        self._adapter.remove_owned_entry(
            self.path_for_io(path), root=self.path_for_io(root), tree=tree
        )

    def restrict_private_directory(self, path: Path) -> None:
        """Narrow a private directory to host-appropriate owner access."""
        self._adapter.restrict_private_directory(path)

    def restrict_private_file(self, path: Path) -> None:
        """Narrow a private file to host-appropriate owner access."""
        self._adapter.restrict_private_file(path)

    def protect_private_file(self, path: Path) -> None:
        """Enforce private access before writing or reusing a secret backup."""
        self._adapter.protect_private_file(self.path_for_io(path))

    def restrict_private_descriptor(self, descriptor: int) -> None:
        """Narrow an opened private file to host-appropriate owner access."""
        self._adapter.restrict_private_descriptor(descriptor)

    @contextmanager
    def exclusive_lock(self, lock_path: Path, *, timeout: float = 1.0) -> Iterator[None]:
        """Hold an exclusive process lock on a stable sidecar path."""
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("Lock timeout must be a finite non-negative number")
        io_path = self.path_for_io(lock_path)
        owned_parent = self.require_owned_directory(io_path.parent, within=io_path.parent)
        if io_path.exists() or io_path.is_symlink():
            self.require_owned_regular_file(io_path, within=owned_parent)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        descriptor = os.open(io_path, flags, 0o600)
        locked = False
        try:
            self.require_opened_owned_regular_file(
                descriptor,
                io_path,
                within=owned_parent,
            )
            self._adapter.restrict_private_descriptor(descriptor)
            deadline = time.monotonic() + timeout
            while not self._adapter.try_lock_exclusive(descriptor):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for exclusive lock: {lock_path}")
                time.sleep(min(_LOCK_RETRY_INTERVAL_SECONDS, remaining))
            locked = True
            yield
        finally:
            try:
                if locked:
                    self._adapter.unlock(descriptor)
            finally:
                os.close(descriptor)

    def require_owned_directory(self, path: Path, *, within: Path) -> Path:
        """Return an ordinary contained directory or reject an unsafe path."""
        owned_root = self._require_owned_root(within)
        status = path.lstat()
        resolved = self._resolved_existing_path(path)
        if not self._adapter.is_directory(status) or not resolved.is_relative_to(owned_root):
            _raise_unsafe(path)
        return resolved

    def require_owned_regular_file(self, path: Path, *, within: Path) -> Path:
        """Return an ordinary singly linked contained file or reject an unsafe path."""
        owned_root = self._require_owned_root(within)
        status = path.lstat()
        resolved = self._resolved_existing_path(path)
        if (
            not self._adapter.is_regular_file(status)
            or status.st_nlink != 1
            or not resolved.is_relative_to(owned_root)
        ):
            _raise_unsafe(path)
        return resolved

    def require_opened_owned_regular_file(
        self, descriptor: int, path: Path, *, within: Path
    ) -> Path:
        """Require an open descriptor to match its stable owned regular-file path."""
        return self._require_opened_regular_file(
            descriptor,
            path,
            within=within,
            require_single_link=True,
        )

    def require_opened_contained_regular_file(
        self, descriptor: int, path: Path, *, within: Path
    ) -> Path:
        """Require an open descriptor to match its contained regular-file path."""
        return self._require_opened_regular_file(
            descriptor,
            path,
            within=within,
            require_single_link=False,
        )

    def _require_opened_regular_file(
        self,
        descriptor: int,
        path: Path,
        *,
        within: Path,
        require_single_link: bool,
    ) -> Path:
        owned_root = self._require_owned_root(within)
        opened = os.fstat(descriptor)
        current = path.lstat()
        resolved = self._resolved_existing_path(path)
        if (
            not self._adapter.is_regular_file(opened)
            or not self._adapter.is_regular_file(current)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or not resolved.is_relative_to(owned_root)
            or (require_single_link and (opened.st_nlink != 1 or current.st_nlink != 1))
        ):
            _raise_unsafe(path)
        return resolved

    def atomic_create_text(self, target: Path, content: str) -> bool:
        """Create exact UTF-8 content without replacing an existing target."""
        return self.atomic_create_text_with_identity(target, content) is not None

    def atomic_create_text_with_identity(self, target: Path, content: str) -> FileIdentity | None:
        """Create exact UTF-8 content and return its publication identity."""
        return self.atomic_create_bytes_with_identity(target, content.encode("utf-8"))

    def atomic_create_bytes_with_identity(
        self, target: Path, content: bytes
    ) -> FileIdentity | None:
        """Create complete byte content and return its publication identity."""
        io_target = self.path_for_io(target)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=io_target.parent,
            prefix=f".{io_target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            stream = os.fdopen(descriptor, "wb")
            descriptor = -1
            with stream:
                written = stream.write(content)
                if written != len(content):
                    raise OSError("atomic creation did not write the complete content")
                stream.flush()
                self._adapter.sync_file(stream.fileno())
                identity = self.file_identity(os.fstat(stream.fileno()))
            try:
                os.link(temporary, io_target)
            except FileExistsError:
                return None
            self._adapter.sync_parent_directory(io_target.parent)
            return identity
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def atomic_replace_text(self, target: Path, content: str) -> None:
        """Replace a target atomically with exact UTF-8 content."""
        self.atomic_replace_bytes(target, content.encode("utf-8"))

    def atomic_replace_bytes(self, target: Path, content: bytes) -> None:
        """Replace a target atomically with complete byte content."""
        io_target = self.path_for_io(target)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=io_target.parent,
            prefix=f".{io_target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            stream = os.fdopen(descriptor, "wb")
            descriptor = -1
            with stream:
                written = stream.write(content)
                if written != len(content):
                    raise OSError("atomic replacement did not write the complete content")
                stream.flush()
                self._adapter.sync_file(stream.fileno())
            os.replace(temporary, io_target)
            self._adapter.sync_parent_directory(io_target.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def file_identity(status: stat_result) -> FileIdentity:
        """Return identity fields stable across hard-link publication."""
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)

    def _require_owned_root(self, path: Path) -> Path:
        status = path.lstat()
        resolved = self._resolved_existing_path(path)
        if not self._adapter.is_directory(status):
            _raise_unsafe(path)
        return resolved

    def _resolved_existing_path(self, path: Path) -> Path:
        try:
            return self._adapter.resolved_for_comparison(path)
        except (OSError, RuntimeError):
            _raise_unsafe(path)


def _raise_unsafe(path: Path) -> NoReturn:
    raise UnsafeFilesystemPath(EACCES, "Owned path is unavailable or unsafe", str(path))


WINDOWS_HOST_FILESYSTEM: Final = HostFilesystem(WindowsFilesystemAdapter())
HOST_FILESYSTEM: Final = WINDOWS_HOST_FILESYSTEM
