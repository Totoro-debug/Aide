"""Persist the normalized directories used by Web Conversation Sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from omni.agent.workspace_state import normalize_workspace_path
from omni.config.agent_home import AgentHome
from omni.utils.host_filesystem import HOST_FILESYSTEM

CONVERSATION_WORKSPACES_FILENAME = "conversation-workspaces.json"
CONVERSATION_WORKSPACES_FORMAT_VERSION = 1


class ConversationWorkspaceCatalogError(ValueError):
    """A safe Conversation Workspace catalog validation or persistence failure."""


class ConversationWorkspaceCatalog:
    """Atomically persist canonical directories used by non-Project Web Sessions."""

    def __init__(self, agent_home: AgentHome) -> None:
        self.agent_home = agent_home
        self.path = agent_home.path / CONVERSATION_WORKSPACES_FILENAME
        self._paths: dict[str, Path] | None = None

    def list(self) -> tuple[Path, ...]:
        self._load()
        assert self._paths is not None
        return tuple(self._paths.values())

    def contains(self, path: Path) -> bool:
        identity = self._identity(path)
        self._load()
        assert self._paths is not None
        return identity in self._paths

    def remember(self, path: Path) -> Path:
        normalized = self.validate(path)
        identity = self._identity(normalized)
        self._load()
        assert self._paths is not None
        existing = self._paths.get(identity)
        if existing is not None:
            return existing
        self._paths[identity] = normalized
        try:
            self._save()
        except Exception:
            self._paths.pop(identity, None)
            raise
        return normalized

    def _load(self) -> None:
        if self._paths is not None:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._paths = {}
            return
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ConversationWorkspaceCatalogError(
                "Conversation Workspace catalog could not be read safely"
            ) from error
        if (
            not isinstance(raw, dict)
            or raw.get("format_version") != CONVERSATION_WORKSPACES_FORMAT_VERSION
            or not isinstance(raw.get("workspaces"), list)
        ):
            raise ConversationWorkspaceCatalogError(
                "Conversation Workspace catalog format is invalid"
            )
        paths: dict[str, Path] = {}
        try:
            for value in raw["workspaces"]:
                if not isinstance(value, str):
                    raise ValueError("path")
                path = Path(value)
                if not path.is_absolute():
                    raise ValueError("relative path")
                normalized = self.validate(path)
                identity = self._identity(normalized)
                if identity in paths:
                    raise ValueError("duplicate path")
                paths[identity] = normalized
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ConversationWorkspaceCatalogError(
                "Conversation Workspace catalog entries are invalid"
            ) from error
        self._paths = paths

    def _save(self) -> None:
        assert self._paths is not None
        self.agent_home.initialize()
        payload = {
            "format_version": CONVERSATION_WORKSPACES_FORMAT_VERSION,
            "workspaces": [str(path) for path in self._paths.values()],
        }
        content = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
        HOST_FILESYSTEM.atomic_replace_text(self.path, content)

    def validate(self, path: Path) -> Path:
        """Return the canonical directory path after checking the Agent Home boundary."""
        try:
            normalized = normalize_workspace_path(path)
            resolved = normalized.resolve(strict=False)
            agent_home = self.agent_home.path.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ConversationWorkspaceCatalogError(
                "Conversation Workspace path could not be resolved"
            ) from error
        overlaps_agent_home = (
            resolved == agent_home
            or agent_home in resolved.parents
            or resolved in agent_home.parents
        )
        if overlaps_agent_home:
            chat_path = normalize_workspace_path(self.agent_home.path) / "chat"
            is_junction = getattr(chat_path, "is_junction", lambda: False)
            chat_root = chat_path.resolve(strict=False)
            if (
                resolved != chat_root
                or chat_root.parent != agent_home
                or chat_path.is_symlink()
                or is_junction()
            ):
                raise ConversationWorkspaceCatalogError(
                    "Conversation Workspace may not overlap Agent Home"
                )
        return resolved

    @staticmethod
    def _identity(path: Path) -> str:
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ConversationWorkspaceCatalogError(
                "Conversation Workspace path could not be resolved"
            ) from error
        return os.path.normcase(str(resolved))


__all__ = [
    "CONVERSATION_WORKSPACES_FILENAME",
    "CONVERSATION_WORKSPACES_FORMAT_VERSION",
    "ConversationWorkspaceCatalog",
    "ConversationWorkspaceCatalogError",
]
