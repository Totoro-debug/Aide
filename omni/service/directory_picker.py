"""A cancellable Windows folder dialog, isolated from the service event loop."""

from __future__ import annotations

import asyncio
import ctypes
import json
import subprocess
import sys
from contextlib import suppress
from typing import Any
from uuid import UUID

from .errors import service_error


class DirectoryPicker:
    """Own at most one native dialog process for the local service."""

    def __init__(self) -> None:
        self._active: asyncio.Task[str | None] | None = None
        self._closed = False

    async def pick(self) -> str | None:
        if self._closed:
            raise service_error("directory_picker_unavailable", "Folder selection is unavailable.")
        if self._active is not None:
            raise service_error(
                "directory_picker_busy",
                "A folder selection window is already open.",
                retryable=True,
            )
        task = asyncio.create_task(self._run())
        self._active = task
        try:
            return await task
        finally:
            self._active = None

    async def close(self) -> None:
        self._closed = True
        if self._active is not None:
            self._active.cancel()
            await asyncio.gather(self._active, return_exceptions=True)

    async def _run(self) -> str | None:
        if sys.platform != "win32":
            raise service_error(
                "directory_picker_unavailable", "Folder selection requires Windows.", status=500
            )
        process: asyncio.subprocess.Process | None = None
        try:
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "omni.service.directory_picker",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                # A cancellation during process creation still owns the resulting child.
                process = await spawn
                raise
            output, _ = await process.communicate()
            if process.returncode != 0:
                raise ValueError("Folder dialog failed")
            result = json.loads(output.decode("utf-8"))
            if not isinstance(result, dict) or "path" not in result:
                raise ValueError("Invalid folder dialog response")
            path = result["path"]
            if path is not None and (not isinstance(path, str) or not path):
                raise ValueError("Invalid selected directory")
            return path
        except (OSError, ValueError, UnicodeError):
            raise service_error(
                "directory_picker_unavailable",
                "The folder selection window could not be opened. Please try again.",
                status=500,
                retryable=True,
            ) from None
        finally:
            if process is not None and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()


class _GUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]


def _guid(value: str) -> _GUID:
    return _GUID.from_buffer_copy(UUID(value).bytes_le)


def _com_call(pointer: ctypes.c_void_p, index: int, types: tuple[Any, ...], *args: Any) -> int:
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *types)(table[index])
    return int(method(pointer, *args))


def _check_result(result: int) -> None:
    if result < 0:
        raise OSError("Windows folder dialog failed")


def _select_directory() -> str | None:
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    user32 = ctypes.WinDLL("user32")
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = ctypes.c_void_p

    _check_result(ole32.CoInitializeEx(None, 2))  # COINIT_APARTMENTTHREADED
    dialog = ctypes.c_void_p()
    item = ctypes.c_void_p()
    path = ctypes.c_void_p()
    try:
        _check_result(
            ole32.CoCreateInstance(
                ctypes.byref(_guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")),
                None,
                1,
                ctypes.byref(_guid("D57C7288-D4AD-4768-BE02-9D969532D960")),
                ctypes.byref(dialog),
            )
        )
        options = ctypes.c_uint()
        _check_result(
            _com_call(dialog, 10, (ctypes.POINTER(ctypes.c_uint),), ctypes.byref(options))
        )
        # FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR
        _check_result(
            _com_call(dialog, 9, (ctypes.c_uint,), options.value | 0x20 | 0x40 | 0x800 | 0x8)
        )
        result = _com_call(dialog, 3, (ctypes.c_void_p,), user32.GetForegroundWindow())
        if result == ctypes.c_long(0x800704C7).value:  # HRESULT_FROM_WIN32(ERROR_CANCELLED)
            return None
        _check_result(result)
        _check_result(_com_call(dialog, 20, (ctypes.POINTER(ctypes.c_void_p),), ctypes.byref(item)))
        _check_result(
            _com_call(
                item,
                5,
                (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
                0x80058000,
                ctypes.byref(path),  # SIGDN_FILESYSPATH
            )
        )
        return ctypes.wstring_at(path)
    finally:
        if path.value:
            ole32.CoTaskMemFree(path)
        if item.value:
            _com_call(item, 2, ())
        if dialog.value:
            _com_call(dialog, 2, ())
        ole32.CoUninitialize()


if __name__ == "__main__":
    try:
        selected = _select_directory()
    except OSError:
        sys.exit(1)
    sys.stdout.buffer.write(json.dumps({"path": selected}, ensure_ascii=False).encode("utf-8"))
