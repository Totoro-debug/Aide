import subprocess
from pathlib import Path

from tests.fixtures.paths import create_workspace


def test_agent_home_fixture_redirects_the_fixed_home_without_precreating_it(
    agent_home: Path,
) -> None:
    assert agent_home == Path.home() / ".omni"
    assert agent_home.is_absolute()
    assert not agent_home.exists()


def test_workspace_fixture_provides_an_existing_normalized_directory(
    workspace: Path,
    tmp_path: Path,
) -> None:
    assert workspace.is_absolute()
    assert workspace == tmp_path / "workspace"
    assert workspace.is_dir()


def test_workspace_fixture_preserves_lexical_directory_alias(tmp_path: Path) -> None:
    target = tmp_path / "workspace-target"
    target.mkdir()
    alias = tmp_path / "workspace-alias"
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
        check=True,
        capture_output=True,
        text=True,
    )

    workspace = create_workspace(alias)

    assert workspace == alias / "workspace"
