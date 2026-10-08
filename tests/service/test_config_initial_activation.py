"""A valid configuration permits first use without restarting the Service."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from aide.config.agent_home import AgentHome
from aide.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [None, "[broken"])
async def test_external_valid_file_permits_conversation_after_invalid_startup(
    tmp_path: Path, initial: str | None
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    config = home.path / "config.toml"
    if initial is not None:
        config.write_text(initial, encoding="utf-8")
    service = AgentService(home, reconnect_timeout=3600)
    project = tmp_path / "registered"
    project.mkdir()
    service.projects.register(project)
    await service.start()
    try:
        client = await service.register_client("web")
        config.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
        saved = service.config_view()
        assert cast(dict[str, object], saved["application"]) == {
            "status": "next-run-required",
            "saved_revision": saved["revision"],
            "active_revision": None,
            "restart_required": False,
        }
        startup = cast(dict[str, object], service.configuration_startup_view()["startup"])
        assert startup["available"] is True
        identity = service.service_instance_id
        workspace = await service.attach_workspace(client.client_id, project)
        session = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, session)
        assert cast(dict[str, object], claim["snapshot"])["session_id"] == session
        assert service.configuration_ready and service.service_instance_id == identity
        assert (project / ".aide").is_dir()
    finally:
        await service.stop()
