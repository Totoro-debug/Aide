"""Run an isolated production service for browser tests."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
from pathlib import Path

from myclaw.config.agent_home import AgentHome
from myclaw.service.client import ServiceClient

_CONFIG = """[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "e2e-fixture-only"
models = ["small-model"]

[runtime]
compact_ratio = 0.9

[models.routes.default]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="myclaw-web-e2e-") as root:
        path = Path(root)
        home = AgentHome(path / ".myclaw")
        home.initialize()
        (home.path / "config.toml").write_text(_CONFIG, encoding="utf-8")
        workspace = path / "workspace"
        workspace.mkdir()
        port = _free_port()
        client = await ServiceClient.connect_or_start(home, workspace, port=port)
        try:
            launch_url = await client.create_web_ticket()
            print(
                json.dumps(
                    {
                        "url": client.base_url,
                        "home_root": str(path),
                        "ticket": launch_url.split("#ticket=", 1)[1],
                    }
                ),
                flush=True,
            )
            await asyncio.to_thread(sys.stdin.readline)
        finally:
            await client.close()
            await ServiceClient.stop_existing(home, port=port)


if __name__ == "__main__":
    asyncio.run(main())
