import importlib.util
import shutil
import subprocess
from pathlib import Path, PurePath, PureWindowsPath

import pytest

from aide.agent.workspace_state import (
    WorkspaceState,
    WorkspaceStateError,
    normalize_workspace_path,
)

EXPECTED_MEMORY = (
    "# Long-term Memory\n\n## User Info\n\n## User Preference\n\n## Project Fact\n\n## Lesson\n"
)


def state_for(workspace: Path) -> WorkspaceState:
    return WorkspaceState(workspace)


def test_normalize_workspace_path_preserves_lexical_path_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "Project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    assert normalize_workspace_path(Path("Project") / "discarded" / "..") == project.absolute()
    assert normalize_workspace_path(tmp_path / "discarded" / "..") == tmp_path
    assert normalize_workspace_path(PurePath(tmp_path / "discarded" / "..")) == tmp_path
    with pytest.raises(ValueError, match="Workspace path must be absolute"):
        normalize_workspace_path(PurePath("Project") / "discarded" / "..")

    assert normalize_workspace_path.__module__ == "aide.agent.workspace_state"


def test_workspace_wrapper_module_is_removed() -> None:
    assert importlib.util.find_spec("aide.agent.workspace") is None


def test_normalized_workspace_path_uses_the_current_hosts_native_path_type(
    tmp_path: Path,
) -> None:
    normalized = normalize_workspace_path(tmp_path)

    assert normalized == tmp_path.absolute()
    assert type(normalized) is type(Path())


def test_windows_drive_workspace_path_has_the_accepted_identity() -> None:
    normalized = normalize_workspace_path(PureWindowsPath(r"D:\desktop\project\Demo-one"))

    assert normalized == Path(r"D:\desktop\project\Demo-one")


def test_unc_workspace_path_has_the_accepted_identity() -> None:
    normalized = normalize_workspace_path(PureWindowsPath(r"\\server\share\Demo-one"))

    assert normalized == Path(r"\\server\share\Demo-one")


def test_windows_workspace_path_is_lexically_normalized() -> None:
    normalized = normalize_workspace_path(
        PureWindowsPath(r"D:\desktop\project\discarded\..\current")
    )

    assert normalized == Path(r"D:\desktop\project\current")


def test_relative_pure_windows_workspace_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        normalize_workspace_path(PureWindowsPath(r"project\subdirectory"))


def create_directory_alias(alias: Path, target: Path) -> None:
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
        check=True,
        capture_output=True,
        text=True,
    )


def test_initialization_rejects_agent_home_as_workspace(
    agent_home: Path,
) -> None:
    user_home = agent_home.parent

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(user_home).initialize(agent_home_root=agent_home)

    assert captured.value.path == agent_home
    assert not (agent_home / "memory").exists()
    assert not (agent_home / "sessions").exists()


def test_initialization_rejects_workspace_beneath_agent_home_without_reading_unrelated_state(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_files = {
        agent_home / "notes" / "private.md": b"# Private notes\r\n",
        agent_home / "notes" / "attachment.bin": b"private attachment\xff",
        agent_home / "scratch.json": b"invalid scratch state\xff",
    }
    for path, content in unrelated_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    workspace = agent_home / "nested-workspace"
    workspace.mkdir()
    state = state_for(workspace)
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_unrelated_bytes(path: Path) -> bytes:
        if path in unrelated_files:
            raise AssertionError(f"unrelated state was read: {path}")
        return original_read_bytes(path)

    def reject_unrelated_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path in unrelated_files:
            raise AssertionError(f"unrelated state was read: {path}")
        return original_read_text(path, encoding=encoding, errors=errors)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_bytes", reject_unrelated_bytes)
        guarded.setattr(Path, "read_text", reject_unrelated_text)
        with pytest.raises(WorkspaceStateError) as captured:
            state.initialize(agent_home_root=agent_home)

    assert captured.value.path == state.path
    assert not state.path.exists()
    assert {path: path.read_bytes() for path in unrelated_files} == unrelated_files


def test_initialization_allows_only_the_explicit_agent_home_chat_workspace(
    agent_home: Path,
) -> None:
    chat = agent_home / "chat"
    chat.mkdir(parents=True)
    nested = chat / "nested"
    nested.mkdir()

    WorkspaceState(chat).initialize(agent_home_root=agent_home, allow_agent_home_chat=True)

    with pytest.raises(WorkspaceStateError):
        WorkspaceState(nested).initialize(
            agent_home_root=agent_home, allow_agent_home_chat=True
        )


def test_initialization_does_not_follow_agent_home_chat_redirects(
    agent_home: Path,
) -> None:
    target = agent_home / "skills"
    target.mkdir(parents=True)
    chat = agent_home / "chat"
    try:
        create_directory_alias(chat, target)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory junctions are unavailable: {error}")

    with pytest.raises(WorkspaceStateError):
        WorkspaceState(chat).initialize(
            agent_home_root=agent_home, allow_agent_home_chat=True
        )

    assert not (target / ".aide").exists()


def test_initialization_rejects_case_and_junction_aliases_of_agent_home(
    agent_home: Path,
    tmp_path: Path,
) -> None:
    user_home = agent_home.parent
    case_alias = Path(str(user_home).swapcase())
    junction_alias = tmp_path / "user-home-junction"
    create_directory_alias(junction_alias, user_home)

    for workspace in (case_alias, junction_alias):
        with pytest.raises(WorkspaceStateError):
            state_for(workspace).initialize(agent_home_root=agent_home)

    assert not agent_home.exists()


def test_initialization_accepts_project_beneath_user_home(agent_home: Path) -> None:
    workspace = agent_home.parent / "project"
    workspace.mkdir()
    state = state_for(workspace)

    state.initialize(agent_home_root=agent_home)

    assert state.memory_directory.is_dir()
    assert state.sessions_directory.is_dir()
    assert state.long_term_memory_path.read_text(encoding="utf-8") == EXPECTED_MEMORY


def test_initialization_creates_only_required_base_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = state_for(workspace)

    state.initialize(agent_home_root=agent_home)

    tree = tuple(
        sorted(
            "/".join(path.relative_to(state.path).parts) + ("/" if path.is_dir() else "")
            for path in state.path.rglob("*")
        )
    )
    assert tree == (".gitignore", "memory/", "memory/memory.md", "sessions/")
    assert (state.path / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert state.long_term_memory_path.read_text(encoding="utf-8") == EXPECTED_MEMORY


def test_repeated_initialization_preserves_policy_memory_and_unknown_entries(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = state_for(workspace)
    state.path.mkdir()
    gitignore = state.path / ".gitignore"
    gitignore.write_bytes(b"!memory/memory.md\r\n")
    unknown = state.path / "future-state.bin"
    unknown.write_bytes(b"future schema")
    state.memory_directory.mkdir()
    state.long_term_memory_path.write_bytes(b"# Existing memory\r\n")

    state.initialize(agent_home_root=agent_home)
    state.initialize(agent_home_root=agent_home)

    assert gitignore.read_bytes() == b"!memory/memory.md\r\n"
    assert unknown.read_bytes() == b"future schema"
    assert state.long_term_memory_path.read_bytes() == b"# Existing memory\r\n"


def test_copying_a_complete_workspace_retains_its_workspace_state(
    agent_home: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    state = state_for(workspace)
    state.initialize(agent_home_root=agent_home)
    state.long_term_memory_path.write_bytes(b"# Portable memory\n")
    session = state.sessions_directory / "portable-session.jsonl"
    session.write_bytes(b'{"portable_test":true}\n')
    copied_workspace = tmp_path / "copied-workspace"

    shutil.copytree(workspace, copied_workspace)
    copied_state = state_for(copied_workspace)

    assert copied_state.long_term_memory_path.read_bytes() == b"# Portable memory\n"
    assert (copied_state.sessions_directory / session.name).read_bytes() == session.read_bytes()
    assert copied_state.path != state.path
    assert not (copied_workspace / agent_home.name / "config.toml").exists()


def test_initialization_rejects_non_directory_root(
    agent_home: Path,
    workspace: Path,
) -> None:
    root = workspace / ".aide"
    root.write_text("collision", encoding="utf-8")

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(workspace).initialize(agent_home_root=agent_home)

    assert captured.value.path == root
    assert not (root / "memory").exists()
    assert root.read_text(encoding="utf-8") == "collision"


def test_initialization_rejects_junction_root(
    agent_home: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    root = workspace / ".aide"
    target = tmp_path / "outside-junction"
    target.mkdir()
    create_directory_alias(root, target)

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(workspace).initialize(agent_home_root=agent_home)

    assert captured.value.path == root
    assert root.is_junction()
    assert not (target / "memory").exists()


def test_initialization_rejects_external_memory_directory_alias(
    workspace: Path,
) -> None:
    state = state_for(workspace)
    state.path.mkdir()
    outside = workspace.parent / "outside-memory"
    outside.mkdir()
    create_directory_alias(state.memory_directory, outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".aide")

    assert captured.value.path == state.memory_directory
    assert not (outside / "memory.md").exists()


def test_initialization_rejects_external_sessions_directory_alias(
    workspace: Path,
) -> None:
    state = state_for(workspace)
    state.path.mkdir()
    outside = workspace.parent / "outside-sessions"
    outside.mkdir()
    create_directory_alias(state.sessions_directory, outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".aide")

    assert captured.value.path == state.sessions_directory


def test_initialization_rejects_hard_linked_memory_file(workspace: Path) -> None:
    state = state_for(workspace)
    state.memory_directory.mkdir(parents=True)
    outside = workspace.parent / "outside-memory.md"
    protected_content = b"outside memory must remain unchanged\n"
    outside.write_bytes(protected_content)
    state.long_term_memory_path.hardlink_to(outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".aide")

    assert captured.value.path == state.long_term_memory_path
    assert outside.read_bytes() == protected_content
