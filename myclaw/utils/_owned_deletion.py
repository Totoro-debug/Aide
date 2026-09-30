"""Native deletion with directory identities held throughout traversal."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any


def _relative_target(path: Path, root: Path) -> Path:
    relative = path.absolute().relative_to(root.absolute())
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise PermissionError("Deletion requires an entry below the owned root")
    return relative


def remove_owned_windows(path: Path, *, root: Path, tree: bool) -> None:
    """Remove below directories held against rename by Windows handles."""
    _relative_target(path, root)
    _windows_remove(path.absolute(), root=root.absolute(), tree=tree)


def remove_owned_posix(path: Path, *, root: Path, tree: bool) -> None:
    """Remove relative to no-follow directory descriptors on POSIX."""
    relative = _relative_target(path, root)
    with _posix_parent(root, relative.parts[:-1]) as parent:
        _posix_remove(parent, relative.name, tree=tree)


@contextmanager
def _posix_parent(root: Path, parts: tuple[str, ...]) -> Iterator[int]:
    descriptors: list[int] = []
    flags = _posix_directory_flags()
    try:
        absolute_root = root.absolute()
        descriptors.append(os.open(absolute_root.anchor, flags))
        for part in (*absolute_root.parts[1:], *parts):
            descriptors.append(os.open(part, flags, dir_fd=descriptors[-1]))
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _posix_directory_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not isinstance(nofollow, int):
        raise NotImplementedError("Safe directory deletion is unavailable on this host")
    return os.O_RDONLY | directory | nofollow


def _posix_remove(parent: int, name: str, *, tree: bool) -> None:
    try:
        status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if tree:
        if not S_ISDIR(status.st_mode):
            raise PermissionError("Deletion directory is not ordinary")
        descriptor = os.open(
            name,
            _posix_directory_flags(),
            dir_fd=parent,
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino):
                raise PermissionError("Deletion directory identity changed")
            for child in os.listdir(descriptor):
                child_status = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
                _posix_remove(descriptor, child, tree=S_ISDIR(child_status.st_mode))
        finally:
            os.close(descriptor)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (status.st_dev, status.st_ino):
            raise PermissionError("Deletion directory identity changed")
        os.rmdir(name, dir_fd=parent)
    else:
        if not S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise PermissionError("Deletion file is not ordinary and singly linked")
        os.unlink(name, dir_fd=parent)
    os.fsync(parent)


class _WindowsFileInformation(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("attributes", ctypes.c_uint32),
        ("creation_time", ctypes.c_uint64),
        ("access_time", ctypes.c_uint64),
        ("write_time", ctypes.c_uint64),
        ("volume", ctypes.c_uint32),
        ("size_high", ctypes.c_uint32),
        ("size_low", ctypes.c_uint32),
        ("links", ctypes.c_uint32),
        ("index_high", ctypes.c_uint32),
        ("index_low", ctypes.c_uint32),
    ]


def _windows_api() -> Any:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_WindowsFileInformation),
    ]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


@contextmanager
def _windows_entry(kernel: Any, path: Path, *, directory: bool, deleting: bool) -> Iterator[Any]:
    # Deny rename and reparse mutations while traversing each held directory.
    handle = kernel.CreateFileW(
        str(path),
        0x10000 if deleting else 0,
        0x1,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _WindowsFileInformation()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if (
            info.attributes & (0x400 | 0x40)
            or bool(info.attributes & 0x10) != directory
            or (not directory and info.links != 1)
        ):
            raise PermissionError("Deletion entry is an alias or is not ordinary")
        yield handle
    finally:
        kernel.CloseHandle(handle)


def _windows_remove(path: Path, *, root: Path, tree: bool) -> None:
    from contextlib import ExitStack

    kernel = _windows_api()
    relative = path.relative_to(root)
    with ExitStack() as stack:
        parent = Path(root.anchor)
        stack.enter_context(_windows_entry(kernel, parent, directory=True, deleting=False))
        for part in (*root.parts[1:], *relative.parts[:-1]):
            parent /= part
            stack.enter_context(_windows_entry(kernel, parent, directory=True, deleting=False))
        _windows_remove_entry(kernel, path, tree=tree)


def _windows_remove_entry(kernel: Any, path: Path, *, tree: bool) -> None:
    try:
        with _windows_entry(kernel, path, directory=tree, deleting=True) as handle:
            if tree:
                for child in path.iterdir():
                    status = child.lstat()
                    _windows_remove_entry(
                        kernel, child, tree=bool(status.st_file_attributes & 0x10)
                    )
            disposition = ctypes.c_ubyte(1)
            if not kernel.SetFileInformationByHandle(
                handle,
                4,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
    except FileNotFoundError:
        return
