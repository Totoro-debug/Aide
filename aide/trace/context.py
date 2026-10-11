"""Immutable identity and source metadata for one trace root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePath

from aide.trace.types import TraceOrigin, TracePurpose


def workspace_key_for(path: Path | PurePath) -> str:
    """Return a stable hash of the lexically normalized Workspace path."""
    from aide.agent.workspace_state import normalize_workspace_path

    normalized = normalize_workspace_path(path)
    identity = os.path.normcase(str(normalized))
    return f"sha256:{sha256(identity.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class TraceRootSource:
    workspace_key: str
    workspace_path: str
    origin: TraceOrigin
    purpose: TracePurpose
    owner_session_id: str | None = None
    subject_session_id: str | None = None
    run_id: str | None = None
    restore_run_token: str | None = None
    agent_id: str | None = None
    parent_root_id: str | None = None
    schedule_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class TraceContext:
    root_id: str
    created_at: datetime
    workspace_key: str
    workspace_path: str
    origin: TraceOrigin
    purpose: TracePurpose
    owner_session_id: str | None = None
    subject_session_id: str | None = None
    run_id: str | None = None
    restore_run_token: str | None = None
    agent_id: str | None = None
    parent_root_id: str | None = None
    schedule_run_id: str | None = None

    @classmethod
    def create(cls, root_id: str, source: TraceRootSource) -> TraceContext:
        return cls(
            root_id=root_id,
            created_at=datetime.now(UTC),
            workspace_key=source.workspace_key,
            workspace_path=source.workspace_path,
            origin=source.origin,
            purpose=source.purpose,
            owner_session_id=source.owner_session_id,
            subject_session_id=source.subject_session_id,
            run_id=source.run_id,
            restore_run_token=source.restore_run_token,
            agent_id=source.agent_id,
            parent_root_id=source.parent_root_id,
            schedule_run_id=source.schedule_run_id,
        )
