from pathlib import Path

from aide.config.agent_home import AgentHome


def test_production_agent_home_is_fixed(agent_home: Path) -> None:
    assert AgentHome.production().path == agent_home


def test_first_initialization_creates_only_the_global_root(agent_home: Path) -> None:
    AgentHome(agent_home).initialize()

    tree = tuple(
        sorted(
            "/".join(path.relative_to(agent_home).parts) + ("/" if path.is_dir() else "")
            for path in agent_home.rglob("*")
        )
    )
    assert tree == ()


def test_repeated_initialization_preserves_all_unrelated_state_bytes(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    unrelated_files = {
        agent_home / "notes" / "private.md": b"# Private notes\r\n",
        agent_home / "notes" / "attachment.bin": b"private attachment\xff",
        agent_home / "scratch.json": b"invalid scratch state\xff",
    }
    for path, content in unrelated_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    home.initialize()

    assert {path: path.read_bytes() for path in unrelated_files} == unrelated_files
