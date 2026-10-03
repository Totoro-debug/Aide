"""Controlled resource workload for the AgentService migration (no external services)."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import mcp.client.stdio as stdio_client
import pytest
from loguru import logger

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.provider.openai_compatible import OpenAICompatibleProvider
from omni.service.runtime import LocalService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.mcp_wire import (
    ObservedLifetimes,
    http_wire_server,
    stdio_wire_configuration,
)
from tests.service.test_service_concurrency import _CollectingSink, _ConcurrentProvider


def working_set_bytes() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "pool_peak",
                "pool",
                "nonpool_peak",
                "nonpool",
                "page",
                "page_peak",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    get_memory = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.working


async def main() -> None:
    logger.remove()
    with (
        tempfile.TemporaryDirectory(prefix="omni-resource-probe-") as directory,
        pytest.MonkeyPatch.context() as patch,
    ):
        root = Path(directory)
        patch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
        lifetime = ObservedLifetimes(patch)
        stdio_working_directories = []
        observed_spawn = stdio_client._create_platform_compatible_process

        async def record_spawn(**kwargs):
            process = await observed_spawn(**kwargs)
            stdio_working_directories.append(
                {"pid": process.pid, "cwd": os.path.normcase(str(Path(kwargs["cwd"]).resolve()))}
            )
            return process

        patch.setattr(stdio_client, "_create_platform_compatible_process", record_spawn)
        controlled = _ConcurrentProvider(block_b=True)
        # Keep real SDK construction/closure; replace only external model requests.
        patch.setattr(
            OpenAICompatibleProvider, "stream", lambda self, **kwargs: controlled.stream(**kwargs)
        )
        patch.setattr(
            OpenAICompatibleProvider,
            "complete",
            lambda self, **kwargs: controlled.complete(**kwargs),
        )
        sdk_created = 0
        sdk_closed = 0
        original_init = OpenAICompatibleProvider.__init__
        original_close = OpenAICompatibleProvider.close

        def create(self, *args, **kwargs):
            nonlocal sdk_created
            original_init(self, *args, **kwargs)
            sdk_created += 1

        async def close(self):
            nonlocal sdk_closed
            await original_close(self)
            sdk_closed += 1

        patch.setattr(OpenAICompatibleProvider, "__init__", create)
        patch.setattr(OpenAICompatibleProvider, "close", close)
        stdio = stdio_wire_configuration(root, {})
        async with http_wire_server({}) as (wire, http):
            home = AgentHome(root / "home")
            home.initialize()
            config = (
                MINIMAL_VALID_CONFIG
                + f"""
[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = {json.dumps(http.url)}
[mcp.servers.stdio]
enabled = true
transport = "stdio"
command = {json.dumps(stdio.command)}
args = {json.dumps(list(stdio.args))}
"""
            )
            (home.path / "config.toml").write_text(config, encoding="utf-8")
            service = LocalService(
                home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600
            )
            snapshots = []
            initial_tasks = set(asyncio.all_tasks())

            def snapshot(label):
                tasks = [task for task in asyncio.all_tasks() - initial_tasks if not task.done()]
                snapshots.append(
                    {
                        "state": label,
                        "sdk_clients_created": sdk_created,
                        "http_initializations": sum(
                            r["method"] == "initialize" for r in wire.requests
                        ),
                        "stdio_subprocesses_created": len(lifetime.processes),
                        "live_stdio_subprocesses": sum(
                            p.returncode is None for p in lifetime.processes
                        ),
                        "tasks": dict(
                            sorted(Counter(t.get_coro().__qualname__ for t in tasks).items())
                        ),
                        "working_set_bytes": working_set_bytes(),
                    }
                )

            try:
                await service.start()
                client = await service.register_client("cli")
                sink = _CollectingSink()
                await service.connect_client(client.client_id, sink)
                client.subscribed = True
                lanes = []
                for index in range(3):
                    workspace_path = root / f"workspace-{index}"
                    workspace_path.mkdir()
                    workspace = await service.attach_workspace(client.client_id, workspace_path)
                    for _ in range(10):
                        lane_client = await service.register_client("cli")
                        await service.connect_client(lane_client.client_id, sink)
                        await service.attach_workspace(lane_client.client_id, workspace_path)
                        session = await workspace.create_draft(lane_client.client_id)
                        claim = await service.claim(
                            lane_client.client_id, workspace.workspace_id, session
                        )
                        lanes.append((workspace, session, claim, lane_client))
                for workspace, session, claim, lane_client in lanes:
                    await workspace.input(
                        lane_client.client_id,
                        session,
                        claim["claim"]["claim_version"],
                        "warmup",
                        session,
                    )
                    # Warmup must complete without the parallel barrier.
                    controlled.release_b.set()
                    await asyncio.wait_for(sink.wait_for("run.completed", session), 10)
                for workspace, session, _, _ in lanes:
                    await workspace.loops[session].loop.wait_for_restore_idle()
                    await workspace.loops[session].loop.session.wait_for_pending_persist()
                await asyncio.sleep(0)
                snapshot("idle-after-30-identical-warmups")
                controlled.release_b.clear()
                controlled.session_a_started.clear()
                controlled.session_b_started.clear()
                for index, (workspace, session, claim, lane_client) in enumerate(lanes[:2]):
                    await workspace.input(
                        lane_client.client_id,
                        session,
                        claim["claim"]["claim_version"],
                        "session-a" if index == 0 else "session-b",
                        f"parallel-{index}",
                    )
                await asyncio.wait_for(controlled.session_a_started.wait(), 10)
                await asyncio.wait_for(controlled.session_b_started.wait(), 10)
                snapshot("two-sessions-in-provider")
                workspace, session, claim, lane_client = lanes[0]
                await workspace.input(
                    lane_client.client_id,
                    session,
                    claim["claim"]["claim_version"],
                    "queued",
                    "queued",
                )
                snapshot("one-additional-input-queued")
                controlled.release_a.set()
                controlled.release_b.set()
                for run in ("parallel-0", "parallel-1", "queued"):
                    await asyncio.wait_for(sink.wait_for("run.completed", run), 10)
            finally:
                controlled.release_a.set()
                controlled.release_b.set()
                await service.stop()
            print(
                json.dumps(
                    {
                        "python": sys.version,
                        "pid": os.getpid(),
                        "workload": "3 Workspaces x 10 Sessions; one identical warmup each; 2 blocked Runs + 1 queued input; H=1 S=1",
                        "workspaces": [
                            {
                                "name": f"workspace-{index}",
                                "normalized_path": os.path.normcase(str((root / f"workspace-{index}").resolve())),
                            }
                            for index in range(3)
                        ],
                        "stdio_working_directories": stdio_working_directories,
                        "snapshots": snapshots,
                        "sdk_clients_closed": sdk_closed,
                        "mcp_client_sessions": len(lifetime.closed),
                        "mcp_client_close_counts": lifetime.close_counts,
                        "remaining_stdio_processes": sum(
                            p.returncode is None for p in lifetime.processes
                        ),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
