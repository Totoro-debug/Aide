from __future__ import annotations

from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader, ConfigRevisionConflict
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def _loader(tmp_path: Path, content: str = MINIMAL_VALID_CONFIG) -> ConfigLoader:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(content, encoding="utf-8")
    return ConfigLoader(home)


def test_editable_snapshot_exposes_only_safe_runtime_and_memory_fields(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    snapshot = loader.editable_snapshot()

    assert snapshot.revision.startswith("sha256:")
    assert snapshot.fields == {
        "runtime": {
            "max_tool_result_chars": 4096,
            "max_iterations": 50,
            "enable_skill_always_load": False,
            "compact_ratio": 0.9,
            "permission_level": "workspace-write",
            "exec_shell": "auto",
        },
        "memory": {"batch_size": 10, "schedule": "0 * * * *"},
    }
    assert "api_key" not in str(snapshot.fields)


def test_valid_patch_preserves_comments_and_unknown_toml(tmp_path: Path) -> None:
    content = "# keep this comment\n" + MINIMAL_VALID_CONFIG + "\n[future]\nvalue = 7\n"
    loader = _loader(tmp_path, content)
    before = loader.path.read_bytes()

    result = loader.patch_editable_fields(
        loader.editable_snapshot().revision,
        {"runtime": {"max_iterations": 77}, "memory": {"batch_size": 21}},
    )

    after = loader.path.read_text(encoding="utf-8")
    assert result.configuration.runtime.max_iterations == 77
    assert result.configuration.memory.batch_size == 21
    assert result.revision != loader.revision_from_bytes(before)
    assert "# keep this comment" in after
    assert "[future]" in after
    assert "value = 7" in after
    assert "minimal-secret" in after


def test_invalid_patch_does_not_change_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    before = loader.path.read_bytes()
    revision = loader.editable_snapshot().revision

    with pytest.raises(ConfigError):
        loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 1}})

    assert loader.path.read_bytes() == before


def test_stale_patch_does_not_change_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    revision = loader.editable_snapshot().revision
    loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 60}})
    before = loader.path.read_bytes()

    with pytest.raises(ConfigRevisionConflict):
        loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 61}})

    assert loader.path.read_bytes() == before


def test_two_writers_with_one_revision_only_one_succeeds(tmp_path: Path) -> None:
    first = _loader(tmp_path)
    second = ConfigLoader(first.agent_home)
    revision = first.editable_snapshot().revision

    first.patch_editable_fields(revision, {"memory": {"batch_size": 11}})

    with pytest.raises(ConfigRevisionConflict):
        second.patch_editable_fields(revision, {"memory": {"batch_size": 12}})

    assert first.load().memory.batch_size == 11
