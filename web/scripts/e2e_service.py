"""Run an isolated production service for browser tests."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aiohttp import web

from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.service.client import ServiceClient
from myclaw.service.discovery import discovery_path

CONFIRMATION_PATH: str | None = None


def _config(base_url: str) -> str:
    return f"""[models.providers.primary]
protocol = "openai-compatible"
base_url = "{base_url}"
api_key = "e2e-fixture-only"
models = ["small-model"]

[runtime]
compact_ratio = 0.9
permission_level = "full-access"

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


def _chunk(
    *,
    request_id: str,
    delta: dict[str, object],
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "small-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _write_sse(
    response: web.StreamResponse, chunks: list[dict[str, object]], *, delay: float
) -> None:
    for chunk in chunks:
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(delay)
    await response.write(b"data: [DONE]\n\n")


async def _fixture_completion(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = body.get("messages", []) if isinstance(body, dict) else []
    user_prompt = ""
    has_tool_result = False
    last_user_index = -1
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                user_prompt = message["content"]
                last_user_index = index
            if message.get("role") == "tool":
                has_tool_result = True

    request_id = f"fixture-{uuid4()}"
    chunks: list[dict[str, object]] = []
    normalized_prompt = user_prompt.lower()
    tool_states_request = "tool states" in normalized_prompt
    has_confirmation_result = any(
        index > last_user_index
        and isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_call_id") == "call-confirmation"
        for index, message in enumerate(messages)
    )
    confirmation_request = (
        isinstance(body.get("tools"), list)
        and "confirmation" in normalized_prompt
        and not any(
            marker in normalized_prompt
            for marker in (
                "tool states",
                "streaming markdown",
                "retry once",
            )
        )
    )
    streaming_request = "streaming markdown" in user_prompt.lower()
    wait_command = (
        "Get-Content -LiteralPath .\\fixture.txt -Wait"
        if sys.platform == "win32"
        else "tail -f fixture.txt"
    )
    if confirmation_request and not has_confirmation_result:
        if CONFIRMATION_PATH is None:
            raise web.HTTPInternalServerError(text="Confirmation fixture path is not configured")
        tool_calls = [
            (
                "call-confirmation",
                "read_file",
                {"path": CONFIRMATION_PATH},
            )
        ]
        for index, (call_id, name, arguments) in enumerate(tool_calls):
            chunks.append(
                _chunk(
                    request_id=request_id,
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    },
                )
            )
        chunks.append(_chunk(request_id=request_id, delta={}, finish_reason="tool_calls"))
    elif confirmation_request and has_confirmation_result:
        chunks.append(
            _chunk(request_id=request_id, delta={"content": "Confirmation fixture completed."})
        )
        chunks.append(_chunk(request_id=request_id, delta={}, finish_reason="stop"))
    elif tool_states_request and not has_tool_result:
        tool_calls = [
            ("call-completed", "read_file", {"path": "fixture.txt"}),
            ("call-failed", "read_file", {"path": "missing-fixture.txt"}),
            (
                "call-rejected",
                "write_file",
                {"path": ".myclaw/restore/protected.txt", "content": "must not write"},
            ),
            (
                "call-cancelled",
                "exec",
                {"command": wait_command, "timeout": 180},
            ),
        ]
        for index, (call_id, name, arguments) in enumerate(tool_calls):
            chunks.append(
                _chunk(
                    request_id=request_id,
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    },
                )
            )
        chunks.append(_chunk(request_id=request_id, delta={}, finish_reason="tool_calls"))
    elif streaming_request:
        for content in (
            "# Streamed answer\n\n",
            "This is **Persisted Markdown**.\n\n",
            f"```text\n{'x' * 300}\n```\n\n",
            "[safe](https://example.com) [unsafe](javascript:alert(1)) "
            "![remote](https://example.com/remote.png)\n\n",
            "The response arrived in multiple chunks.\n",
        ):
            chunks.append(_chunk(request_id=request_id, delta={"content": content}))
        chunks.append(_chunk(request_id=request_id, delta={}, finish_reason="stop"))
    else:
        chunks.append(_chunk(request_id=request_id, delta={"content": "Fixture response."}))
        chunks.append(_chunk(request_id=request_id, delta={}, finish_reason="stop"))

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)
    try:
        await _write_sse(response, chunks, delay=0.35 if streaming_request else 0.04)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    return response


async def _start_fixture_provider() -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _fixture_completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else None
    if not sockets:
        await runner.cleanup()
        raise RuntimeError("E2E fixture provider did not bind a socket.")
    port = int(sockets[0].getsockname()[1])
    return runner, f"http://127.0.0.1:{port}/v1"


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


async def _seed_overdue_review_jobs(workspace: Path) -> None:
    store = WorkspaceScheduleStore(WorkspaceState(workspace))
    for message, schedule in (
        ("E2E overdue at job", JobSchedule.at("2020-01-01T00:00:00.000+00:00")),
        ("E2E overdue every job", JobSchedule.every(3600)),
        ("E2E next cron job", JobSchedule.cron("0 * * * *", "UTC")),
    ):
        await store.add_user_job(
            ScheduleJob(
                job_id=str(uuid4()),
                message=message,
                schedule=schedule,
                created_at_ms=1,
                updated_at_ms=1,
            )
        )


async def _seed_session(
    home: AgentHome,
    workspace: Path,
    *,
    title: str,
    content: str,
    created_at: datetime,
) -> str:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: created_at)
    session.update_metadata(title=title)
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    return session.session_id


async def _stop_service(home: AgentHome, port: int) -> None:
    await ServiceClient.stop_existing(home, port=port)
    deadline = time.monotonic() + 15.0
    while discovery_path(home).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("E2E service did not finish shutting down.")
        await asyncio.sleep(0.05)


async def _run_e2e(provider_base_url: str) -> None:
    global CONFIRMATION_PATH
    with tempfile.TemporaryDirectory(prefix="myclaw-web-e2e-") as root:
        path = Path(root)
        CONFIRMATION_PATH = str(path / "confirmation-outside.txt")
        Path(CONFIRMATION_PATH).write_text("confirmation fixture content\n", encoding="utf-8")
        home = AgentHome(path / ".myclaw")
        home.initialize()
        (home.path / "config.toml").write_text(_config(provider_base_url), encoding="utf-8")
        cli_workspace = path / "cli-workspace"
        first_project = path / "project-one"
        project_alias = path / "project-one-alias"
        second_project = path / "project-two"
        cli_workspace.mkdir()
        first_project.mkdir()
        second_project.mkdir()
        (first_project / "fixture.txt").write_text("fixture content\n", encoding="utf-8")
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(project_alias), str(first_project)],
                check=True,
                capture_output=True,
            )
        else:
            project_alias.symlink_to(first_project, target_is_directory=True)
        await _seed_schedule_job(home, first_project)
        occupied_session_id = await _seed_session(
            home,
            first_project,
            title="CLI occupied history",
            content="CLI-only history must remain private",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        available_session_id = await _seed_session(
            home,
            first_project,
            title="Web available history",
            content="Available history loaded after a successful Claim",
            created_at=datetime(2026, 9, 2, tzinfo=UTC),
        )
        port = _free_port()
        client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
        project_client = await ServiceClient.connect_or_start(home, first_project, port=port)
        await project_client.claim_session(occupied_session_id)
        try:

            async def announce() -> None:
                launch_url = await client.create_web_ticket()
                second_launch_url = await client.create_web_ticket()
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
                            "second_ticket": second_launch_url.split("#ticket=", 1)[1],
                            "confirmation_path": CONFIRMATION_PATH,
                            "occupied_session_id": occupied_session_id,
                            "available_session_id": available_session_id,
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
                await project_client.close()
                await client.close()
                await _stop_service(home, port)
                await _seed_overdue_review_jobs(first_project)
                client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
                project_client = await ServiceClient.connect_or_start(
                    home, first_project, port=port
                )
                await announce()
        finally:
            await project_client.close()
            await client.close()
            await _stop_service(home, port)


async def main() -> None:
    provider, provider_base_url = await _start_fixture_provider()
    try:
        await _run_e2e(provider_base_url)
    finally:
        await provider.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
