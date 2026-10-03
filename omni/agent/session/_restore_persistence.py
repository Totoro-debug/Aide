"""Shared persistence primitives for Session Restore storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omni.utils.host_filesystem import HOST_FILESYSTEM


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
    HOST_FILESYSTEM.sync_parent_directory(path)


def sync_created_directory(path: Path) -> None:
    sync_directory(path)
    sync_directory(path.parent)


__all__ = [
    "canonical_json_bytes",
    "sha256_hex",
    "sync_created_directory",
    "sync_directory",
]
