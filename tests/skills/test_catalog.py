import subprocess
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any, Self, cast

import pytest
from loguru import logger

from omni.management.commands import MANAGEMENT_COMMANDS
from omni.skills.catalog import LoadedSkill, SkillLoader


def _loader(
    *,
    agent_home: Path,
    reserved_names: tuple[str, ...] = (),
    enable_always_load: bool = False,
) -> SkillLoader:
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=reserved_names,
        enable_always_load=enable_always_load,
    )
    loader.load()
    return loader


def test_missing_skills_root_is_empty_without_creating_directory(agent_home: Path) -> None:
    loader = _loader(agent_home=agent_home)

    assert loader.root == (agent_home / "skills").resolve()
    assert loader.skills == ()
    assert not (agent_home / "skills").exists()


def test_existing_empty_skills_root_is_empty(agent_home: Path) -> None:
    root = agent_home / "skills"
    root.mkdir(parents=True)

    loader = _loader(agent_home=agent_home)

    assert loader.root == root.resolve()
    assert loader.skills == ()


def test_valid_candidate_retains_document_and_validated_metadata(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    document = b'---\nname: "plan"\ndescription: "  Do useful work.  "\n---\nbody\r\n'
    instruction.write_bytes(document)

    loader = _loader(agent_home=agent_home)

    assert len(loader.skills) == 1
    skill = loader.skills[0]
    assert skill.metadata.name == "plan"
    assert skill.metadata.description == "Do useful work."
    assert skill.metadata.path == instruction.resolve()
    assert skill.document == document.decode("utf-8")
    assert skill.always is False
    assert loader.metadata == (skill.metadata,)


def test_successful_reload_replaces_published_frozen_state(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nfirst body\n",
        encoding="utf-8",
    )
    loader = _loader(agent_home=agent_home)
    first_skills = loader.skills

    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nsecond body\n",
        encoding="utf-8",
    )

    loader.load()
    assert loader.skills != first_skills
    assert tuple(skill.metadata.name for skill in loader.skills) == ("reviewer",)
    assert loader.get("planner") is None
    invocation = loader.resolve_manual("/reviewer request")
    assert invocation is not None
    assert invocation.body.splitlines()[-1] == "second body"


def test_failed_publication_validation_preserves_all_published_queries(
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nfirst body\n",
        encoding="utf-8",
    )
    loader = _loader(agent_home=agent_home)
    before_skills = loader.skills
    before_metadata = loader.metadata
    before_skill = loader.get("planner")
    before_invocation = loader.resolve_manual("/planner request")

    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nsecond body\n",
        encoding="utf-8",
    )

    def reject(candidate: tuple[LoadedSkill, ...]) -> None:
        assert tuple(skill.metadata.name for skill in candidate) == ("reviewer",)
        raise ValueError("candidate rejected")

    with pytest.raises(ValueError, match="candidate rejected"):
        loader.load(validate=reject)

    assert loader.skills == before_skills
    assert loader.metadata == before_metadata
    assert loader.get("planner") == before_skill
    assert loader.resolve_manual("/planner request") == before_invocation
    assert loader.resolve_manual("/reviewer request") is None


def test_failed_skill_scan_preserves_published_state(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nfirst body\n",
        encoding="utf-8",
    )
    loader = _loader(agent_home=agent_home)
    before_skills = loader.skills
    before_metadata = loader.metadata
    before_skill = loader.get("planner")
    before_invocation = loader.resolve_manual("/planner request")

    def fail_scan(_path: Path) -> tuple[Path, ...]:
        raise OSError("scan failed")

    monkeypatch.setattr(Path, "iterdir", fail_scan)

    with pytest.raises(OSError, match="scan failed"):
        loader.load()

    assert loader.skills == before_skills
    assert loader.metadata == before_metadata
    assert loader.get("planner") == before_skill
    assert loader.resolve_manual("/planner request") == before_invocation


def test_reserved_management_command_names_are_excluded(agent_home: Path) -> None:
    for command in MANAGEMENT_COMMANDS:
        name = command.token.removeprefix("/")
        instruction = agent_home / "skills" / f"{name}-skill" / "SKILL.md"
        instruction.parent.mkdir(parents=True)
        instruction.write_bytes(
            f"---\nname: {name}\ndescription: Reserved command guide\n---\n".encode()
        )

    loader = _loader(
        agent_home=agent_home,
        reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
    )

    assert loader.skills == ()


def test_first_valid_duplicate_in_canonical_path_order_wins(agent_home: Path) -> None:
    later = agent_home / "skills" / "z-candidate" / "SKILL.md"
    earlier = agent_home / "skills" / "a-candidate" / "SKILL.md"
    for instruction in (later, earlier):
        instruction.parent.mkdir(parents=True)
        instruction.write_bytes(b"---\nname: duplicate\ndescription: Same name\n---\n")

    loader = _loader(agent_home=agent_home)

    assert tuple(skill.metadata.path for skill in loader.skills) == (earlier.resolve(),)


def test_valid_entries_follow_canonical_path_order(agent_home: Path) -> None:
    later = agent_home / "skills" / "z-candidate" / "SKILL.md"
    earlier = agent_home / "skills" / "a-candidate" / "SKILL.md"
    for instruction, name in ((later, "later"), (earlier, "earlier")):
        instruction.parent.mkdir(parents=True)
        instruction.write_bytes(f"---\nname: {name}\ndescription: Valid metadata\n---\n".encode())

    loader = _loader(agent_home=agent_home)

    assert tuple(skill.metadata.name for skill in loader.skills) == ("earlier", "later")


def test_only_direct_child_skill_directories_are_scanned(agent_home: Path) -> None:
    direct = agent_home / "skills" / "direct" / "SKILL.md"
    nested = agent_home / "skills" / "container" / "nested" / "SKILL.md"
    for instruction, name in ((direct, "direct"), (nested, "nested")):
        instruction.parent.mkdir(parents=True)
        instruction.write_bytes(f"---\nname: {name}\ndescription: Valid metadata\n---\n".encode())

    loader = _loader(agent_home=agent_home)

    assert tuple(skill.metadata.name for skill in loader.skills) == ("direct",)


@pytest.mark.parametrize(
    "document",
    (
        b"name: valid\ndescription: no opening delimiter\n",
        b"---\nname: valid\ndescription: missing closing\n",
        b"---\nname: [broken\ndescription: invalid YAML\n---\n",
        b"---\n- item\n---\n",
        b"---\ndescription: missing name\n---\n",
        b"---\nname: valid\ndescription: [not a string]\n---\n",
    ),
)
def test_invalid_frontmatter_shapes_are_excluded(agent_home: Path, document: bytes) -> None:
    instruction = agent_home / "skills" / "invalid-frontmatter" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(document)

    loader = _loader(agent_home=agent_home)

    assert loader.skills == ()


@pytest.mark.parametrize(
    ("name", "accepted"),
    (
        ("a", True),
        ("-a", True),
        ("_a", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("1a", False),
        ("A", False),
        ("a.b", False),
        (" a", False),
        ("a ", False),
    ),
)
def test_name_character_and_length_contract(
    agent_home: Path,
    name: str,
    accepted: bool,
) -> None:
    instruction = agent_home / "skills" / "name-boundary" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        f'---\nname: "{name}"\ndescription: "Valid description"\n---\n'.encode()
    )

    loader = _loader(agent_home=agent_home)

    assert bool(loader.skills) is accepted


@pytest.mark.parametrize(
    ("description", "accepted"),
    (
        ("", False),
        ("   ", False),
        ("x", True),
        ("x" * 1024, True),
        ("x" * 1025, False),
    ),
)
def test_description_trimmed_length_contract(
    agent_home: Path,
    description: str,
    accepted: bool,
) -> None:
    instruction = agent_home / "skills" / "description-boundary" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        f'---\nname: "description"\ndescription: "{description}"\n---\n'.encode()
    )

    loader = _loader(agent_home=agent_home)

    assert bool(loader.skills) is accepted
    if accepted:
        assert loader.skills[0].metadata.description == description.strip()


def test_non_utf8_document_is_excluded_without_logging_its_body(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "non-utf8" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: invalid\ndescription: Invalid UTF-8\n---\nSECRET-BODY\xff\n"
    )
    diagnostics = StringIO()
    handler = logger.add(diagnostics, format="{message}", level="WARNING")
    try:
        loader = _loader(agent_home=agent_home)
    finally:
        logger.remove(handler)

    assert loader.skills == ()
    assert "SECRET-BODY" not in diagnostics.getvalue()


def test_non_regular_instruction_path_is_excluded(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "directory-instruction" / "SKILL.md"
    instruction.mkdir(parents=True)

    loader = _loader(agent_home=agent_home)

    assert loader.skills == ()


def test_invalid_candidate_diagnostic_contains_path_and_reason_but_not_document(
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "invalid" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: invalid\ndescription: missing closing delimiter\nSECRET-SKILL-BODY\n"
    )
    diagnostics = StringIO()
    handler = logger.add(diagnostics, format="{message}", level="WARNING")
    try:
        loader = _loader(agent_home=agent_home)
    finally:
        logger.remove(handler)

    assert loader.skills == ()
    assert str(instruction.parent) in diagnostics.getvalue()
    assert "SECRET-SKILL-BODY" not in diagnostics.getvalue()


def test_loader_reads_each_candidate_once_as_one_complete_bytes_read(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = agent_home / "skills"
    documents = {
        "a-first": b"---\nname: shared\ndescription: First\n---\nfirst\n",
        "b-duplicate": b"---\nname: shared\ndescription: Duplicate\n---\nsecond\n",
        "c-invalid": b"---\nname: invalid\ndescription: Missing delimiter\ninvalid body\n",
        "d-reserved": b"---\nname: config\ndescription: Reserved\n---\nreserved\n",
    }
    for name, document in documents.items():
        instruction = root / name / "SKILL.md"
        instruction.parent.mkdir(parents=True)
        instruction.write_bytes(document)

    original_open = cast(Callable[..., Any], Path.open)
    opens = 0
    reads: list[tuple[Path, int]] = []

    class RecordingStream:
        def __init__(self, stream: Any, path: Path) -> None:
            self._stream = stream
            self._path = path

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            del exc_type, exc_value, traceback
            self._stream.close()

        def fileno(self) -> int:
            return cast(int, self._stream.fileno())

        def read(self, size: int = -1) -> bytes:
            reads.append((self._path, size))
            return cast(bytes, self._stream.read(size))

    def recording_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        stream = original_open(path, *args, **kwargs)
        if path.name != "SKILL.md":
            return stream
        opens += 1
        return RecordingStream(stream, path)

    monkeypatch.setattr(Path, "open", recording_open)

    loader = _loader(agent_home=agent_home, reserved_names=("/config",))

    assert tuple(skill.metadata.name for skill in loader.skills) == ("shared",)
    assert opens == len(documents)
    assert [(path.parent.name, size) for path, size in reads] == [(name, -1) for name in documents]


def test_loader_published_state_is_immutable_and_manual_resolution_does_not_read_disk(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    document = b"---\nname: planner\ndescription: Planner\n---\nfirst\n"
    instruction.write_bytes(document)
    loader = _loader(agent_home=agent_home)

    with pytest.raises(AttributeError):
        loader.skills[0].document = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        loader.skills.append(loader.skills[0])  # type: ignore[attr-defined]

    instruction.unlink()

    def deny_open(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("manual resolution must use the published loader state")

    monkeypatch.setattr(Path, "open", deny_open)
    invocation = loader.resolve_manual("/planner do it")

    assert invocation is not None
    assert invocation.metadata == loader.skills[0].metadata
    assert invocation.request == "do it"
    assert invocation.body == document.decode("utf-8")


@pytest.mark.parametrize(
    "delimiter",
    (" ", "\t", "\r", "\n", "\u2003"),
)
def test_manual_resolution_removes_only_first_unicode_whitespace_delimiter(
    agent_home: Path,
    delimiter: str,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan work\n---\nbody\n")
    loader = _loader(agent_home=agent_home)
    request = f"{delimiter}  keep\n\tthis"

    invocation = loader.resolve_manual(f"/planner{request}")

    assert invocation is not None
    assert invocation.request == "  keep\n\tthis"


@pytest.mark.parametrize(
    "raw_input",
    ("planner", " /planner", "/Planner", "/plan", "/unknown", "/config"),
)
def test_manual_resolution_returns_none_for_non_matching_input(
    agent_home: Path,
    raw_input: str,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan work\n---\nbody\n")
    loader = _loader(agent_home=agent_home, reserved_names=("/config",))

    assert loader.resolve_manual(raw_input) is None


def test_manual_resolution_rejects_non_string_input(agent_home: Path) -> None:
    loader = _loader(agent_home=agent_home)

    with pytest.raises(TypeError, match="Skill input must be a string"):
        loader.resolve_manual(cast(str, object()))


def test_enabled_always_skill_is_fully_loaded_once_and_frozen(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    document = b"---\nname: always\ndescription: Always loaded\nalways: true\n---\nbody\n"
    instruction.write_bytes(document)
    original_open = cast(Callable[..., Any], Path.open)
    opens = 0

    def recording_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        if path.name == "SKILL.md":
            opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    loader = _loader(agent_home=agent_home, enable_always_load=True)

    assert len(loader.skills) == 1
    assert loader.skills[0].always is True
    assert loader.skills[0].document == document.decode("utf-8")
    assert opens == 1


@pytest.mark.parametrize("always_field", ("always: false\n", ""))
def test_non_opted_in_skill_is_not_always_loaded(agent_home: Path, always_field: str) -> None:
    instruction = agent_home / "skills" / "metadata-only" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        (
            "---\nname: metadata-only\ndescription: Metadata only\n" + always_field + "---\nbody\n"
        ).encode()
    )

    loader = _loader(agent_home=agent_home, enable_always_load=True)

    assert len(loader.skills) == 1
    assert loader.skills[0].always is False


def test_disabled_always_field_is_not_interpreted_or_warned(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "always-candidate" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b'---\nname: always-candidate\ndescription: Metadata only\nalways: "true"\n---\nbody\n'
    )
    diagnostics = StringIO()
    handler = logger.add(diagnostics, format="{message}", level="WARNING")
    try:
        loader = _loader(agent_home=agent_home, enable_always_load=False)
    finally:
        logger.remove(handler)

    assert loader.skills[0].always is False
    assert "Ignoring non-boolean Skill always field" not in diagnostics.getvalue()


def test_non_boolean_always_warns_once_and_is_not_always_loaded(agent_home: Path) -> None:
    instruction = agent_home / "skills" / "invalid-always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b'---\nname: invalid-always\ndescription: Invalid policy\nalways: "true"\n---\n'
        b"SECRET DOCUMENT\n"
    )
    diagnostics = StringIO()
    handler = logger.add(diagnostics, format="{message}", level="WARNING")
    try:
        loader = _loader(agent_home=agent_home, enable_always_load=True)
    finally:
        logger.remove(handler)

    assert loader.skills[0].always is False
    assert diagnostics.getvalue().count("Ignoring non-boolean Skill always field") == 1
    assert "SECRET DOCUMENT" not in diagnostics.getvalue()


def test_duplicate_always_candidate_does_not_override_first_valid_entry(
    agent_home: Path,
) -> None:
    first = agent_home / "skills" / "a-first" / "SKILL.md"
    duplicate = agent_home / "skills" / "b-duplicate" / "SKILL.md"
    first.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    first.write_bytes(
        b"---\nname: duplicate\ndescription: First candidate\nalways: false\n---\nfirst\n"
    )
    duplicate.write_bytes(
        b"---\nname: duplicate\ndescription: Later candidate\nalways: true\n---\nsecond\n"
    )

    loader = _loader(agent_home=agent_home, enable_always_load=True)

    assert len(loader.skills) == 1
    assert loader.skills[0].metadata.description == "First candidate"
    assert loader.skills[0].always is False


def test_instruction_symlink_escape_is_excluded_when_links_are_available(
    agent_home: Path,
) -> None:
    outside = agent_home.parent / "outside" / "SKILL.md"
    outside.parent.mkdir()
    outside.write_bytes(b"---\nname: escaped\ndescription: Outside\n---\nbody\n")
    instruction = agent_home / "skills" / "escaped" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    try:
        instruction.symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")

    loader = _loader(agent_home=agent_home)

    assert loader.skills == ()


def test_skill_directory_reparse_escape_is_excluded_when_links_are_available(
    agent_home: Path,
) -> None:
    outside = agent_home.parent / "outside-directory"
    instruction = outside / "SKILL.md"
    instruction.parent.mkdir()
    instruction.write_bytes(b"---\nname: escaped\ndescription: Outside\n---\nbody\n")
    linked_directory = agent_home / "skills" / "escaped"
    linked_directory.parent.mkdir(parents=True)
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_directory), str(outside)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, NotImplementedError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory links unavailable: {error}")

    loader = _loader(agent_home=agent_home)

    assert loader.skills == ()


def test_regular_hardlink_instruction_remains_readable_when_available(agent_home: Path) -> None:
    outside = agent_home.parent / "shared-skill.md"
    outside.write_bytes(b"---\nname: linked\ndescription: Linked\n---\nbody\n")
    instruction = agent_home / "skills" / "linked" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    try:
        instruction.hardlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"hard links unavailable: {error}")

    loader = _loader(agent_home=agent_home)

    assert len(loader.skills) == 1
    assert loader.skills[0].document.endswith("body\n")
