"""Native deletion with directory identities held throughout traversal."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
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
