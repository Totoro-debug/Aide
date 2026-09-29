"""Run an isolated production service for browser tests."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
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


async def _seed_schedule_job(home: AgentHome, workspace: Path) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    now_ms = int(time.time() * 1000)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(uuid4()),
            message="E2E saved project job",
            schedule=JobSchedule.every(3600),
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
    )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="myclaw-web-e2e-") as root:
        path = Path(root)
        home = AgentHome(path / ".myclaw")
        home.initialize()
        (home.path / "config.toml").write_text(_CONFIG, encoding="utf-8")
        cli_workspace = path / "cli-workspace"
        first_project = path / "project-one"
        project_alias = path / "project-one-alias"
        second_project = path / "project-two"
        cli_workspace.mkdir()
        first_project.mkdir()
        second_project.mkdir()
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(project_alias), str(first_project)],
                check=True,
                capture_output=True,
            )
        else:
            project_alias.symlink_to(first_project, target_is_directory=True)
        await _seed_schedule_job(home, first_project)
        port = _free_port()
        client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
        try:

            async def announce() -> None:
                launch_url = await client.create_web_ticket()
                print(
                    json.dumps(
                        {
                            "url": client.base_url,
                            "home_root": str(path),
                            "ticket": launch_url.split("#ticket=", 1)[1],
                            "cli_workspace": str(cli_workspace),
                            "first_project": str(first_project),
                            "project_alias": str(project_alias),
                            "second_project": str(second_project),
                        }
                    ),
                    flush=True,
                )

            await announce()
            while True:
                command = await asyncio.to_thread(sys.stdin.readline)
                if not command or command.strip() == "stop":
                    break
                if command.strip() != "restart":
                    continue
                await client.close()
                await ServiceClient.stop_existing(home, port=port)
                client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
                await announce()
        finally:
            await client.close()
            await ServiceClient.stop_existing(home, port=port)


if __name__ == "__main__":
    asyncio.run(main())
