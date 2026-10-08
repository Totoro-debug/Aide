"""Run one isolated service for the configuration startup browser matrix."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.schedule.store import WorkspaceScheduleStore
from aide.service.client import ServiceClient
from aide.service.discovery import DEFAULT_SERVICE_PORT, discovery_path, read_discovery
from aide.service.projects import ProjectCatalog
from web.scripts.e2e_service import _start_fixture_provider

StartupState = Literal["missing", "invalid", "malformed"]
MALFORMED_SECRET = "malformed-secret-303"


async def _launch_web(root: Path, workspace: Path) -> str:
    """Exercise the installed CLI composition path before any service exists."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", DEFAULT_SERVICE_PORT))
    source = (
        "import sys, webbrowser; "
        "from aide.terminal.process_entry import run; "
        "webbrowser.open_new_tab = lambda _url: False; "
        "sys.argv = ['aide', 'web']; run()"
    )
    environment = {
        **os.environ,
        "USERPROFILE": str(root),
        "HOME": str(root),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        source,
        cwd=workspace,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"Production aide web failed: {stderr.decode()}")
    return stdout.decode().strip()


def _semantic_invalid_config(provider_base_url: str) -> str:
    return f'''[models.providers.openai-local]
protocol = "openai-compatible"
base_url = "{provider_base_url}"
api_key = ""
models = ["small-model"]

# The required chat route is intentionally absent for the semantic-repair case.
[models.routes.title]
provider_id = "openai-local"
model = "old-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
'''


def _malformed_config() -> bytes:
    return f'''[models.providers.old]
protocol = "openai-compatible"
base_url = "https://old.example/v1"
api_key = "{MALFORMED_SECRET}"
models = ["old-model"]

[models.routes.chat]
provider_id = "old"
model = "old-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30

[broken
value = true
'''.encode()


async def _seed_project(home: AgentHome, project: Path) -> str:
    state = WorkspaceState(project)
    state.initialize(agent_home_root=home.path)
    now_ms = 1
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(uuid4()),
            message="Startup repair saved job",
            schedule=JobSchedule.every(3600),
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
    )
    return ProjectCatalog(home).register(project, schedule_state="awaiting_resume").project_id


async def _stop_service(home: AgentHome, port: int) -> None:
    discovery = read_discovery(home)
    process_handle = None
    if discovery is not None:
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        process_handle = kernel.OpenProcess(0x00100000, False, discovery.pid)
    await ServiceClient.stop_existing(home, port=port)
    deadline = asyncio.get_running_loop().time() + 15.0
    while discovery_path(home).exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("Startup E2E service did not finish shutting down.")
        await asyncio.sleep(0.05)
    if process_handle is not None:
        try:
            result = await asyncio.to_thread(kernel.WaitForSingleObject, process_handle, 15000)
            if result != 0:
                raise RuntimeError("Startup E2E service process did not exit.")
        finally:
            kernel.CloseHandle(process_handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("missing", "invalid", "malformed"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


async def _run(state: StartupState, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    home = AgentHome(root / ".aide")
    home.initialize()
    cli_workspace = root / "cli-workspace"
    cli_workspace.mkdir()
    project = root / "persistent-project"
    project.mkdir()
    available_project = root / "available-project"
    available_project.mkdir()

    provider_runner, provider_base_url = await _start_fixture_provider()
    client: ServiceClient | None = None
    try:
        project_id = await _seed_project(home, project)
        available_project_id = ProjectCatalog(home).register(available_project).project_id
        config_path = home.path / "config.toml"
        if state == "invalid":
            config_path.write_text(_semantic_invalid_config(provider_base_url), encoding="utf-8")
        elif state == "malformed":
            config_path.write_bytes(_malformed_config())
        assert not discovery_path(home).exists()
        cold_launch_url = await _launch_web(root, cli_workspace)
        assert discovery_path(home).exists()
        client = await ServiceClient.connect_or_start(
            home,
            cli_workspace,
            attach_workspace=False,
        )
        await client._open_socket()
        service_info = await client._http_request("GET", "/api/v1/service")
        raw_bytes = config_path.read_bytes() if config_path.exists() else b""
        digest = hashlib.sha256(raw_bytes).hexdigest()
        details = {
            "state": state,
            "url": client.base_url,
            "cold_launch_url": cold_launch_url,
            "home_root": str(home.path),
            "user_home_root": str(root),
            "cli_workspace": str(cli_workspace),
            "project": str(project),
            "project_id": project_id,
            "available_project_id": available_project_id,
            "pid": client.discovery.pid,
            "port": client.discovery.port,
            "backup_blocker": str(home.path / f"config.toml.backup.{digest}"),
            "provider_base_url": provider_base_url,
            "malformed_secret": MALFORMED_SECRET,
            "initial_service": service_info,
        }
        print(json.dumps(details), flush=True)
        while True:
            command = await asyncio.to_thread(sys.stdin.readline)
            if not command or command.strip() == "stop":
                break
            if command.strip() == "restart":
                port = client.discovery.port
                await client.close()
                await _stop_service(home, port)
                cold_launch_url = await _launch_web(root, cli_workspace)
                client = await ServiceClient.connect_or_start(home, cli_workspace, attach_workspace=False)
                await client._open_socket()
                details.update({
                    "url": client.base_url,
                    "cold_launch_url": cold_launch_url,
                    "pid": client.discovery.pid,
                    "port": client.discovery.port,
                })
                print(json.dumps(details), flush=True)
                continue
            if command.strip() == "service":
                print(json.dumps(await client._http_request("GET", "/api/v1/service")), flush=True)
    finally:
        if client is not None:
            port = client.discovery.port
            await client.close()
            await _stop_service(home, port)
        await provider_runner.cleanup()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args.state, args.root))


if __name__ == "__main__":
    main()
