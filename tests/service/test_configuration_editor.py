"""Saved-configuration behavior exercised without an Agent Service."""

import asyncio
from pathlib import Path
from typing import cast

import pytest

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.service.configuration import ConfigurationEdit, ConfigurationEditor, ConfigurationSave
from omni.service.errors import ServiceError
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


@pytest.fixture
def editor(tmp_path: Path) -> ConfigurationEditor:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    result = ConfigurationEditor(ConfigLoader(home))
    assert result.start() is not None
    return result


@pytest.mark.asyncio
async def test_save_and_retry_retain_receipt_and_startup_revision(
    editor: ConfigurationEditor,
) -> None:
    revision = cast(str, editor.view()["revision"])
    edit = ConfigurationEdit("patch", "save", revision, {"runtime": {"max_iterations": 80}})

    saved = await editor.save(edit)
    retry = await editor.save(edit)

    assert saved.changed
    assert not retry.changed
    assert retry.view == saved.view
    application = cast(dict[str, object], editor.view()["application"])
    assert application["active_revision"] == revision
    assert application["restart_required"] is True


@pytest.mark.asyncio
async def test_competing_edits_of_one_revision_have_one_winner(
    editor: ConfigurationEditor,
) -> None:
    revision = cast(str, editor.view()["revision"])
    results = await asyncio.gather(
        editor.save(
            ConfigurationEdit("patch", "one", revision, {"runtime": {"max_iterations": 80}})
        ),
        editor.save(
            ConfigurationEdit("patch", "two", revision, {"runtime": {"max_iterations": 90}})
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, ConfigurationSave) for result in results) == 1
    failures = [result for result in results if isinstance(result, ServiceError)]
    assert len(failures) == 1
    assert failures[0].code == "config_revision_conflict"


@pytest.mark.asyncio
async def test_reusing_request_for_different_edit_preserves_saved_view(
    editor: ConfigurationEditor,
) -> None:
    revision = cast(str, editor.view()["revision"])
    await editor.save(
        ConfigurationEdit("patch", "save", revision, {"runtime": {"max_iterations": 80}})
    )
    before = editor.view()

    with pytest.raises(ServiceError) as conflict:
        await editor.save(
            ConfigurationEdit("patch", "save", revision, {"runtime": {"max_iterations": 90}})
        )

    assert conflict.value.code == "request_conflict"
    assert editor.view() == before
