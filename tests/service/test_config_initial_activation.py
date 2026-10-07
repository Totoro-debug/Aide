"""First-use repair activates only on the next service startup."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from omni.config.agent_home import AgentHome
from omni.service.errors import ServiceError
from omni.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [None, "[broken"])
async def test_external_valid_file_requires_restart_after_invalid_startup(
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
            "status": "restart-required",
            "saved_revision": saved["revision"],
            "active_revision": None,
            "restart_required": True,
        }
        assert service._workspaces == {} and not service.configuration_ready
        with pytest.raises(ServiceError) as blocked:
            await service.attach_workspace(client.client_id, project)
        assert blocked.value.code == "config_invalid"
        assert not (project / ".omni").exists()
    finally:
        await service.stop()
    restarted = AgentService(home, reconnect_timeout=3600)
    try:
        await restarted.start()
        assert restarted.configuration_ready and len(restarted._workspaces) == 1
        assert (
            cast(dict[str, object], restarted.config_view()["application"])["restart_required"]
            is False
        )
    finally:
        await restarted.stop()
