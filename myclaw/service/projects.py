"""Durable Agent Home Project registrations used by the local service."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from myclaw.agent.workspace_state import normalize_workspace_path
from myclaw.config.agent_home import AgentHome

PROJECTS_FILENAME = "projects.json"
PROJECTS_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    path: Path
    schedule_state: str = "available"

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "path": str(self.path),
            "schedule_state": self.schedule_state,
        }


class ProjectCatalogError(ValueError):
    """A safe Project catalog validation or persistence failure."""


class ProjectCatalog:
    """Atomically persist normalized directory registrations under Agent Home."""

    def __init__(self, agent_home: AgentHome) -> None:
        self.agent_home = agent_home
        self.path = agent_home.path / PROJECTS_FILENAME
        self._records: dict[str, ProjectRecord] | None = None

    def list(self) -> tuple[ProjectRecord, ...]:
        self._load()
        assert self._records is not None
        return tuple(self._records.values())

    def register(self, path: Path, *, schedule_state: str = "available") -> ProjectRecord:
        if schedule_state not in {"available", "awaiting_resume"}:
            raise ProjectCatalogError("Project schedule state is invalid")
        normalized = self._validate_path(path)
        identity = self._identity(normalized)
        self._load()
        assert self._records is not None
        for record in self._records.values():
            if self._identity(record.path) == identity:
                return record
        record = ProjectRecord(str(uuid4()), normalized, schedule_state)
        self._records[record.project_id] = record
        self._save()
        return record

    def remove(self, project_id: str) -> ProjectRecord:
        self._load()
        assert self._records is not None
        try:
            record = self._records.pop(project_id)
        except KeyError as error:
            raise ProjectCatalogError("Project registration was not found") from error
        self._save()
        return record

    def set_schedule_state(self, project_id: str, schedule_state: str) -> ProjectRecord:
        if schedule_state not in {
            "available",
            "unavailable",
            "awaiting_resume",
            "removing",
            "failed",
        }:
            raise ProjectCatalogError("Project schedule state is invalid")
        self._load()
        assert self._records is not None
        record = self._records.get(project_id)
        if record is None:
            raise ProjectCatalogError("Project registration was not found")
        updated = ProjectRecord(record.project_id, record.path, schedule_state)
        self._records[project_id] = updated
        self._save()
        return updated

    def _load(self) -> None:
        if self._records is not None:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._records = {}
            return
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProjectCatalogError("Project catalog could not be read safely") from error
        if not isinstance(raw, dict) or raw.get("format_version") != PROJECTS_FORMAT_VERSION:
            raise ProjectCatalogError("Project catalog format is unsupported")
        entries = raw.get("projects")
        if not isinstance(entries, list):
            raise ProjectCatalogError("Project catalog entries are invalid")
        records: dict[str, ProjectRecord] = {}
        identities: set[str] = set()
        try:
            for item in entries:
                if not isinstance(item, dict):
                    raise ValueError("entry")
                project_id = item["project_id"]
                path = normalize_workspace_path(Path(item["path"]))
                schedule_state = item.get("schedule_state", "available")
                if not isinstance(project_id, str) or not project_id:
                    raise ValueError("project id")
                if schedule_state not in {
                    "available",
                    "unavailable",
                    "awaiting_resume",
                    "removing",
                    "failed",
                }:
                    raise ValueError("schedule state")
                identity = self._identity(path)
                if identity in identities:
                    raise ValueError("duplicate path")
                identities.add(identity)
                records[project_id] = ProjectRecord(project_id, path, schedule_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectCatalogError("Project catalog entries are invalid") from error
        self._records = records

    def _save(self) -> None:
        assert self._records is not None
        self.agent_home.initialize()
        payload = {
            "format_version": PROJECTS_FORMAT_VERSION,
            "projects": [record.to_dict() for record in self._records.values()],
        }
        target = self.path
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                fd = -1
                json.dump(payload, stream, ensure_ascii=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if fd >= 0:
                os.close(fd)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _identity(path: Path) -> str:
        try:
            resolved = path.resolve(strict=False)
        except OSError as error:
            raise ProjectCatalogError("Project path could not be resolved") from error
        return os.path.normcase(str(resolved))

    def _validate_path(self, path: Path) -> Path:
        normalized = normalize_workspace_path(path)
        if not normalized.exists() or not normalized.is_dir():
            raise ProjectCatalogError("Project directory is unavailable")
        resolved = normalized.resolve(strict=True)
        agent_home = self.agent_home.path.resolve(strict=False)
        if (
            resolved == agent_home
            or agent_home in resolved.parents
            or resolved in agent_home.parents
        ):
            raise ProjectCatalogError("Project directory overlaps Agent Home")
        return resolved


__all__ = [
    "PROJECTS_FILENAME",
    "PROJECTS_FORMAT_VERSION",
    "ProjectCatalog",
    "ProjectCatalogError",
    "ProjectRecord",
]
