"""Stable Session pagination cursors."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

from aide.service.errors import service_error

_MAX_SESSION_PAGE_SIZE = 100


def _encode_session_cursor(
    key: tuple[datetime, datetime, str], workspace_id: str, title_filter: str
) -> str:
    payload = json.dumps(
        {
            "workspace_id": workspace_id,
            "title_filter": title_filter,
            "updated_at": key[0].isoformat(),
            "created_at": key[1].isoformat(),
            "id": key[2],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_session_cursor(
    value: str, workspace_id: str, title_filter: str
) -> tuple[datetime, datetime, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        )
        if payload["workspace_id"] != workspace_id or payload["title_filter"] != title_filter:
            raise ValueError("cursor scope does not match")
        updated_at = datetime.fromisoformat(payload["updated_at"])
        created_at = datetime.fromisoformat(payload["created_at"])
        session_id = payload["id"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise service_error("validation_error", "cursor is invalid.", status=422) from error
    if (
        updated_at.tzinfo is None
        or created_at.tzinfo is None
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise service_error("validation_error", "cursor is invalid.", status=422)
    return updated_at, created_at, session_id


def _encode_chat_session_cursor(key: tuple[datetime, datetime, str, str], title_filter: str) -> str:
    payload = json.dumps(
        {
            "title_filter": title_filter,
            "updated_at": key[0].isoformat(),
            "created_at": key[1].isoformat(),
            "id": key[2],
            "directory": key[3],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_chat_session_cursor(
    value: str, title_filter: str
) -> tuple[datetime, datetime, str, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        )
        if payload["title_filter"] != title_filter:
            raise ValueError("cursor scope does not match")
        updated_at = datetime.fromisoformat(payload["updated_at"])
        created_at = datetime.fromisoformat(payload["created_at"])
        session_id = payload["id"]
        directory = payload["directory"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise service_error("validation_error", "cursor is invalid.", status=422) from error
    if (
        updated_at.tzinfo is None
        or created_at.tzinfo is None
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(directory, str)
        or not Path(directory).is_absolute()
    ):
        raise service_error("validation_error", "cursor is invalid.", status=422)
    return updated_at, created_at, session_id, directory
