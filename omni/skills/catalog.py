"""Generation-scoped loading of user-authored Skill documents."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from loguru import logger

from omni.utils.host_filesystem import HOST_FILESYSTEM

_NAME_PATTERN = re.compile(r"[a-z_-][a-z0-9_-]{0,63}")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Validated metadata retained for one Skill candidate."""

    name: str
    description: str
    path: Path


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """One complete Skill document captured for a Runtime Generation."""

    metadata: SkillMetadata
    document: str
    always: bool

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, SkillMetadata):
            raise TypeError("Loaded Skill requires Skill metadata")
        if not isinstance(self.document, str):
            raise TypeError("Loaded Skill requires a complete document")
        if not isinstance(self.always, bool):
            raise TypeError("Loaded Skill always flag must be a boolean")


@dataclass(frozen=True, slots=True)
class ManualSkillInvocation:
    """One validated manual Skill invocation for a foreground Agent Run."""

    metadata: SkillMetadata
    request: str
    body: str


class SkillLoader:
    """Discover and atomically publish valid complete Skill documents."""

    def __init__(
        self,
        *,
        root: Path,
        reserved_names: Collection[str],
        enable_always_load: bool,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("Skill Loader root must be a Path")
        if not isinstance(enable_always_load, bool):
            raise TypeError("Skill Loader always-load flag must be a boolean")
        self._root = root.resolve()
        self._reserved_names = frozenset(name.removeprefix("/") for name in reserved_names)
        self._enable_always_load = enable_always_load
        self._skills: tuple[LoadedSkill, ...] = ()

    @property
    def root(self) -> Path:
        """Return the canonical root used for Skill discovery and file access."""
        return self._root

    @property
    def skills(self) -> tuple[LoadedSkill, ...]:
        """Return the currently published immutable Skill state."""
        return self._skills

    @property
    def metadata(self) -> tuple[SkillMetadata, ...]:
        """Return metadata for the currently published Skills in discovery order."""
        return tuple(skill.metadata for skill in self._skills)

    def get(self, name: str) -> LoadedSkill | None:
        """Return one published Skill by its exact authored name."""
        return next((skill for skill in self._skills if skill.metadata.name == name), None)

    def resolve_manual(self, raw_input: str) -> ManualSkillInvocation | None:
        """Resolve one exact slash invocation without filesystem access."""
        if not isinstance(raw_input, str):
            raise TypeError("Skill input must be a string")
        if not raw_input.startswith("/"):
            return None

        token = raw_input[1:]
        delimiter = next(
            (index for index, character in enumerate(token) if character.isspace()),
            None,
        )
        if delimiter is None:
            name = token
            request = ""
        else:
            name = token[:delimiter]
            request = token[delimiter + 1 :]
        skill = self.get(name)
        if skill is None:
            return None
        return ManualSkillInvocation(
            metadata=skill.metadata,
            request=request,
            body=skill.document,
        )

    def load(
        self,
        *,
        validate: Callable[[tuple[LoadedSkill, ...]], None] | None = None,
    ) -> None:
        """Stage, validate, and atomically publish the current Skill directory."""
        root = self._root
        try:
            root_is_directory = root.is_dir()
        except (OSError, RuntimeError):
            logger.warning("Skill discovery failed path={} reason=Skill root is unavailable", root)
            raise

        if not root_is_directory:
            candidate_skills: tuple[LoadedSkill, ...] = ()
        else:
            try:
                children = tuple(root.iterdir())
            except (OSError, RuntimeError):
                logger.warning(
                    "Skill discovery failed path={} reason=Skill root is unavailable", root
                )
                raise

            candidates: list[tuple[Path, Path]] = []
            for candidate in children:
                if not candidate.is_dir():
                    continue
                try:
                    canonical_candidate = candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    _log_invalid(candidate, "Skill directory canonical path is unavailable")
                    continue
                candidates.append((canonical_candidate, candidate))

            skills: list[LoadedSkill] = []
            names: set[str] = set()
            for canonical_candidate, candidate in sorted(
                candidates,
                key=lambda item: (str(item[0]), str(item[1])),
            ):
                if not canonical_candidate.is_relative_to(root):
                    _log_invalid(candidate, "Skill directory canonical path escapes Skill root")
                    continue

                instruction = candidate / "SKILL.md"
                try:
                    status = instruction.stat()
                except OSError:
                    _log_invalid(candidate, "SKILL.md is unavailable")
                    continue
                if not HOST_FILESYSTEM.is_regular_file(status):
                    _log_invalid(candidate, "SKILL.md is not a regular file")
                    continue
                try:
                    path = instruction.resolve(strict=True)
                except (OSError, RuntimeError):
                    _log_invalid(candidate, "SKILL.md canonical path is unavailable")
                    continue
                if not path.is_relative_to(root):
                    _log_invalid(candidate, "SKILL.md canonical path escapes Skill root")
                    continue

                skill, reason = self._load_candidate(instruction, path, root)
                if skill is None:
                    _log_invalid(candidate, reason)
                    continue
                if skill.metadata.name in self._reserved_names:
                    _log_invalid(candidate, "name is reserved")
                    continue
                if skill.metadata.name in names:
                    _log_invalid(candidate, "name is duplicated")
                    continue
                names.add(skill.metadata.name)
                skills.append(skill)
            candidate_skills = tuple(skills)

        if validate is not None:
            validate(candidate_skills)
        self._skills = candidate_skills
        logger.info("Discovered Skills count={}", len(candidate_skills))

    def _load_candidate(
        self,
        instruction: Path,
        path: Path,
        root: Path,
    ) -> tuple[LoadedSkill | None, str]:
        try:
            io_path = HOST_FILESYSTEM.path_for_io(instruction)
            with io_path.open("rb") as stream:
                opened_path = HOST_FILESYSTEM.require_opened_contained_regular_file(
                    stream.fileno(),
                    path,
                    within=root,
                )
                if opened_path != path:
                    raise ValueError("Skill path does not match its canonical path")
                raw_content = stream.read()
            if not opened_path.is_relative_to(root):
                raise ValueError("Skill path escapes the canonical Skill root")
        except (OSError, RuntimeError, ValueError):
            return None, "SKILL.md could not be safely read"

        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            return None, "SKILL.md is not valid UTF-8"

        metadata, always_value, reason = _parse_document(
            content,
            path,
            interpret_always=self._enable_always_load,
        )
        if metadata is None:
            return None, reason
        if (
            self._enable_always_load
            and always_value is not _MISSING
            and not isinstance(always_value, bool)
        ):
            logger.warning(
                "Ignoring non-boolean Skill always field path={} reason=always must be a boolean",
                instruction.parent,
            )
        return (
            LoadedSkill(
                metadata=metadata,
                document=content,
                always=self._enable_always_load and always_value is True,
            ),
            "",
        )


def _parse_document(
    content: str,
    path: Path,
    *,
    interpret_always: bool,
) -> tuple[SkillMetadata | None, object, str]:
    lines = content.splitlines(keepends=True)
    if not lines or not _is_text_delimiter(lines[0]):
        return None, _MISSING, "frontmatter opening delimiter is missing"
    frontmatter: list[str] = []
    closing_delimiter_found = False
    for line in lines[1:]:
        if _is_text_delimiter(line):
            closing_delimiter_found = True
            break
        frontmatter.append(line)
    if not closing_delimiter_found:
        return None, _MISSING, "frontmatter closing delimiter is missing"
    try:
        document: object = yaml.safe_load("".join(frontmatter))
    except yaml.YAMLError:
        return None, _MISSING, "frontmatter YAML is invalid"
    metadata, reason = _validate_metadata(document, path)
    if metadata is None:
        return None, _MISSING, reason
    always_value = _always_value(document) if interpret_always else _MISSING
    return metadata, always_value, ""


def _always_value(document: object) -> object:
    if not isinstance(document, Mapping) or "always" not in document:
        return _MISSING
    return document["always"]


def _validate_metadata(document: object, path: Path) -> tuple[SkillMetadata | None, str]:
    if not isinstance(document, Mapping):
        return None, "frontmatter root is not a mapping"
    name = document.get("name")
    description = document.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        return None, "name and description must be strings"
    normalized_description = description.strip()
    if not _NAME_PATTERN.fullmatch(name) or not 1 <= len(normalized_description) <= 1024:
        return None, "name or description is outside the accepted bounds"
    return (
        SkillMetadata(
            name=name,
            description=normalized_description,
            path=path,
        ),
        "",
    )


def _is_text_delimiter(line: str) -> bool:
    return line.rstrip("\r\n") == "---"


def _log_invalid(candidate: Path, reason: str) -> None:
    logger.warning("Skipping invalid Skill candidate path={} reason={}", candidate, reason)


__all__ = [
    "LoadedSkill",
    "ManualSkillInvocation",
    "SkillLoader",
    "SkillMetadata",
]
