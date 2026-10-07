"""Durable Agent Home Project registrations used by the local service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from aide.agent.workspace_state import normalize_workspace_path
from aide.config.agent_home import AgentHome
from aide.utils.host_filesystem import HOST_FILESYSTEM

PROJECTS_FILENAME = "projects.json"
PROJECTS_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    path: Path
    schedule_state: str = "available"
    removal_operation_id: str | None = None
    removal_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "project_id": self.project_id,
            "path": str(self.path),
            "schedule_state": self.schedule_state,
        }
        if self.removal_operation_id is not None:
            result["removal_operation_id"] = self.removal_operation_id
        if self.removal_error is not None:
            result["removal_error"] = self.removal_error
        return result


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
        try:
            self._save()
        except Exception:
            self._records.pop(record.project_id, None)
            raise
        return record

    def remove(self, project_id: str) -> ProjectRecord:
        self._load()
        assert self._records is not None
        try:
            record = self._records.pop(project_id)
        except KeyError as error:
            raise ProjectCatalogError("Project registration was not found") from error
        try:
            self._save()
        except Exception:
            self._records[record.project_id] = record
            raise
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
        updated = ProjectRecord(
            record.project_id,
            record.path,
            schedule_state,
            record.removal_operation_id,
            record.removal_error,
        )
        self._records[project_id] = updated
        try:
            self._save()
        except Exception:
            self._records[project_id] = record
            raise
        return updated

    def begin_removal(self, project_id: str, operation_id: str | None = None) -> ProjectRecord:
        """Persist the admission barrier and return its stable operation identity."""
        self._load()
        assert self._records is not None
        record = self._records.get(project_id)
        if record is None:
            raise ProjectCatalogError("Project registration was not found")
        stable_operation_id = record.removal_operation_id or operation_id or str(uuid4())
        updated = ProjectRecord(
            record.project_id,
            record.path,
            "removing",
            stable_operation_id,
            None,
        )
        self._records[project_id] = updated
        try:
            self._save()
        except Exception:
            self._records[project_id] = record
            raise
        return updated

    def record_removal_failure(self, project_id: str, message: str) -> ProjectRecord:
        """Keep a failed removal registered but permanently closed to admission."""
        if not message:
            raise ValueError("Project removal failure message must be non-empty")
        self._load()
        assert self._records is not None
        record = self._records.get(project_id)
        if record is None:
            raise ProjectCatalogError("Project registration was not found")
        updated = ProjectRecord(
            record.project_id,
            record.path,
            "removing",
            record.removal_operation_id,
            message,
        )
        self._records[project_id] = updated
        try:
            self._save()
        except Exception:
            self._records[project_id] = record
            raise
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
                raw_path = Path(item["path"])
                if not raw_path.is_absolute():
                    raise ValueError("relative path")
                path = normalize_workspace_path(raw_path)
                resolved = path.resolve(strict=False)
                agent_home = self.agent_home.path.resolve(strict=False)
                if (
                    resolved == agent_home
                    or agent_home in resolved.parents
                    or resolved in agent_home.parents
                ):
                    raise ValueError("Agent Home overlap")
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
                removal_operation_id = item.get("removal_operation_id")
                removal_error = item.get("removal_error")
                if removal_operation_id is not None and (
                    not isinstance(removal_operation_id, str) or not removal_operation_id
                ):
                    raise ValueError("removal operation id")
                if removal_error is not None or removal_operation_id is not None:
                    if schedule_state not in {"removing", "failed"}:
                        raise ValueError("removal metadata state")
                    if removal_error is not None and (
                        not isinstance(removal_error, str) or not removal_error
                    ):
                        raise ValueError("removal error")
                identity = self._identity(path)
                if identity in identities:
                    raise ValueError("duplicate path")
                if project_id in records:
                    raise ValueError("duplicate project id")
                identities.add(identity)
                records[project_id] = ProjectRecord(
                    project_id,
                    path,
                    schedule_state,
                    removal_operation_id,
                    removal_error,
                )
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            raise ProjectCatalogError("Project catalog entries are invalid") from error
        self._records = records

    def _save(self) -> None:
        assert self._records is not None
        self.agent_home.initialize()
        payload = {
            "format_version": PROJECTS_FORMAT_VERSION,
            "projects": [record.to_dict() for record in self._records.values()],
        }
        content = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
        HOST_FILESYSTEM.atomic_replace_text(self.path, content)

    @staticmethod
    def _identity(path: Path) -> str:
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ProjectCatalogError("Project path could not be resolved") from error
        return os.path.normcase(str(resolved))

    def _validate_path(self, path: Path) -> Path:
        if not path.is_absolute():
            raise ProjectCatalogError("Project path must be an absolute directory")
        try:
            normalized = normalize_workspace_path(path)
        except (TypeError, ValueError) as error:
            raise ProjectCatalogError("Project path must be an absolute directory") from error
        try:
            available = normalized.exists() and normalized.is_dir()
        except (OSError, ValueError) as error:
            raise ProjectCatalogError("Project directory is unavailable") from error
        if not available:
            raise ProjectCatalogError("Project directory is unavailable")
        try:
            resolved = normalized.resolve(strict=True)
            agent_home = self.agent_home.path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ProjectCatalogError("Project path could not be resolved") from error
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
