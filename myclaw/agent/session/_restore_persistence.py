"""Shared persistence primitives for Session Restore storage."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_POSIX_UNSUPPORTED_SYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sync_directory(path: Path) -> None:
    if not getattr(os, "O_DIRECTORY", 0):
        return
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(HOST_FILESYSTEM.path_for_io(path), flags)
    except OSError as error:
        if error.errno in _POSIX_UNSUPPORTED_SYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in _POSIX_UNSUPPORTED_SYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def sync_created_directory(path: Path) -> None:
    if not getattr(os, "O_DIRECTORY", 0):
        return
    sync_directory(path)
    sync_directory(path.parent)


__all__ = [
    "canonical_json_bytes",
    "sha256_hex",
    "sync_created_directory",
    "sync_directory",
]
