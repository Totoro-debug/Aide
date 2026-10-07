"""Saved configuration has an independent lifetime from active resources."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from omni.config.agent_home import AgentHome
from omni.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.service.test_service_concurrency import _CollectingSink


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "external", [MINIMAL_VALID_CONFIG + "\n[runtime]\nmax_iterations = 83\n", "[broken"]
)
async def test_external_changes_preserve_resources_and_admission(
    tmp_path: Path, external: str
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    config = home.path / "config.toml"
    config.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = AgentService(home, reconnect_timeout=3600)
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _CollectingSink())
        first_path = tmp_path / "first"
        first_path.mkdir()
        first = await service.attach_workspace(client.client_id, first_path)
        previous, startup = first.resources, service.configuration
        revision = cast(dict[str, object], service.config_view()["application"])["active_revision"]
        config.write_text(external, encoding="utf-8")
        view = service.config_view()
        assert view["revision"] != revision
        assert cast(dict[str, object], view["application"])["active_revision"] == revision
        assert first.resources is previous and service.configuration is startup
        assert service.configuration_ready and first.schedule_admitted
        later_path = tmp_path / "later"
        later_path.mkdir()
        later = await service.attach_workspace(client.client_id, later_path)
        assert later.configuration is startup
        session = await later.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, later.workspace_id, session)
        before = later.session_snapshot(session)
        assert (await service.claim(client.client_id, later.workspace_id, session))[
            "claim"
        ] == claim["claim"]
        assert later.session_snapshot(session) == before
        config.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
        assert (
            cast(dict[str, object], service.config_view()["application"])["restart_required"]
            is False
        )
    finally:
        await service.stop()
