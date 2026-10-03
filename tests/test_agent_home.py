from pathlib import Path

from omni.config.agent_home import AgentHome


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


def test_repeated_initialization_preserves_all_legacy_state_bytes(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    legacy_files = {
        agent_home / "memory" / "memory.md": b"# Legacy memory\r\n",
        agent_home / "memory" / "summary.jsonl": b"invalid summary\xff",
        agent_home / "memory" / ".cursor": b"not-a-cursor\n",
        agent_home / "sessions" / "legacy" / "session.jsonl": b"invalid session\xff",
        agent_home / "sessions" / "legacy" / "artifacts" / "result.txt": b"artifact",
        agent_home / "obsolete-state.json": b"invalid obsolete state\xff",
    }
    for path, content in legacy_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    home.initialize()

    assert {path: path.read_bytes() for path in legacy_files} == legacy_files
