"""Run an isolated production service for browser tests."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aiohttp import web

from aide.agent.session.backup_store import FileBackupStore
from aide.agent.session.session import Session
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.schedule.store import WorkspaceScheduleStore
from aide.service.client import ServiceClient
from aide.service.discovery import discovery_path
from aide.service.errors import ServiceError

CONFIRMATION_PATH: str | None = None
SETTINGS_ENTERED = asyncio.Event()
SETTINGS_RELEASE = asyncio.Event()
PROCESS_ENTERED = asyncio.Event()
PROCESS_RELEASE = asyncio.Event()
SETTINGS_RELEASE_PATH: Path | None = None
PROJECT_REMOVAL_ENTERED = asyncio.Event()
PROJECT_REMOVAL_RELEASE = asyncio.Event()
INSTALLED_CONCURRENCY_RELEASE = asyncio.Event()
INSTALLED_EXPIRY_RELEASE = asyncio.Event()
QUEUE_RECALL_ENTERED = asyncio.Event()
QUEUE_RECALL_RELEASE = asyncio.Event()
MODEL_MCP_ENTERED = asyncio.Event()
MODEL_MCP_RELEASE = asyncio.Event()
PROVIDER_OBSERVATION_PATH: Path | None = None


def _config(
    base_url: str,
    *,
    mcp_command: str,
    mcp_args: list[str],
    mcp_cwd: str,
) -> str:
    model_capacities = (
        "[models.providers.primary.model_context_windows]\nlarge-model = 65536\n"
        if os.environ.get("AIDE_E2E_SESSION_MODELS") == "1"
        else ""
    )
    return f"""[models.providers.primary]
protocol = "openai-compatible"
base_url = "{base_url}"
api_key = "e2e-provider-secret-302"
models = ["small-model", "large-model"]

{model_capacities}
[models.providers.retired]
protocol = "openai-compatible"
base_url = "{base_url}"
api_key = "e2e-retired-secret-302"
models = ["retired-model"]

[runtime]
compact_ratio = 0.9
permission_level = "full-access"

[models.routes.chat]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[models.routes.title]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[models.routes.memory]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "low"
timeout = 30

[models.routes.schedule]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[mcp.servers.fixture]
enabled = true
transport = "stdio"
command = {json.dumps(mcp_command)}
args = {json.dumps(mcp_args)}
cwd = {json.dumps(mcp_cwd)}
connect_timeout = 30
call_timeout = 60

[mcp.servers.remote]
enabled = false
transport = "streamable-http"
url = "http://127.0.0.1:1/mcp"
headers = {{ Authorization = "e2e-mcp-secret-302", __proto__ = "e2e-prototype-header-canary-302" }}
connect_timeout = 30
call_timeout = 60
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _chunk(
    *,
    request_id: str,
    delta: dict[str, object],
    model: str,
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _write_sse(
    response: web.StreamResponse, chunks: list[dict[str, object]], *, delay: float,
    hold_after_first: bool = False,
) -> None:
    for index, chunk in enumerate(chunks):
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if hold_after_first and index == 0:
            SETTINGS_ENTERED.set()
            await SETTINGS_RELEASE.wait()
        await asyncio.sleep(delay)
    await response.write(b"data: [DONE]\n\n")


async def _fixture_completion(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = body.get("messages", []) if isinstance(body, dict) else []
    user_prompt = ""
    has_tool_result = False
    tool_result_ids: list[str] = []
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
                if isinstance(message.get("tool_call_id"), str):
                    tool_result_ids.append(message["tool_call_id"])

    request_id = f"fixture-{uuid4()}"
    chunks: list[dict[str, object]] = []
    normalized_prompt = user_prompt.lower()
    requested_prompt = user_prompt.rsplit("## User Input\n", 1)[-1].strip()
    normalized_requested_prompt = requested_prompt.lower()
    model = str(body.get("model", "small-model"))
    tool_names = tuple(
        str(function.get("name"))
        for tool in body.get("tools", [])
        if isinstance(tool, dict)
        and isinstance(function := tool.get("function"), dict)
        and isinstance(function.get("name"), str)
    )
    if PROVIDER_OBSERVATION_PATH is not None:
        with PROVIDER_OBSERVATION_PATH.open("a", encoding="utf-8") as observations:
            observations.write(
                json.dumps(
                    {
                        "model": model,
                        "max_output": body.get("max_tokens"),
                        "temperature": body.get("temperature"),
                        "prompt": user_prompt,
                        "reasoning_effort": body.get("reasoning_effort"),
                        "tools": tool_names,
                        "tool_results": tuple(tool_result_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if "settings generation barrier" in normalized_prompt and isinstance(body.get("tools"), list):
        SETTINGS_ENTERED.set()
        await SETTINGS_RELEASE.wait()
    if "project removal barrier" in normalized_prompt and isinstance(body.get("tools"), list):
        PROJECT_REMOVAL_ENTERED.set()
        await PROJECT_REMOVAL_RELEASE.wait()
    if "concurrent session streaming markdown" in normalized_requested_prompt:
        await INSTALLED_CONCURRENCY_RELEASE.wait()
    if "queue recall barrier" in normalized_requested_prompt:
        QUEUE_RECALL_ENTERED.set()
        await QUEUE_RECALL_RELEASE.wait()
    if "installed expiry barrier" in normalized_requested_prompt:
        await INSTALLED_EXPIRY_RELEASE.wait()
    model_mcp_barrier = "model mcp generation barrier" in normalized_prompt
    model_mcp_request = "model mcp resource" in normalized_prompt
    current_tool_result_ids = {
        message.get("tool_call_id")
        for message in messages[last_user_index + 1 :]
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    tool_search_completed = any(
        isinstance(result_id, str) and result_id.startswith("call-model-mcp-search-")
        for result_id in current_tool_result_ids
    )
    v1_tool_completed = "call-model-mcp-v1" in current_tool_result_ids
    v2_tool_completed = "call-model-mcp-v2" in current_tool_result_ids
    needs_tool_search = (
        (model_mcp_barrier or model_mcp_request)
        and "tool_search" in tool_names
        and not tool_search_completed
        and not v1_tool_completed
        and not v2_tool_completed
    )
    if "provider failure" in normalized_requested_prompt:
        return web.Response(status=502, text="Fixture provider failure")
    process_request = "process cycles" in normalized_requested_prompt
    cancel_gap_request = "process cancel gap" in normalized_requested_prompt
    if (process_request or cancel_gap_request) and "call-process-first" in current_tool_result_ids:
        if cancel_gap_request or "call-process-second" in current_tool_result_ids:
            PROCESS_ENTERED.set()
            await PROCESS_RELEASE.wait()
    if model_mcp_barrier and needs_tool_search:
        MODEL_MCP_ENTERED.set()
        await MODEL_MCP_RELEASE.wait()
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
        and "confirmation" in normalized_requested_prompt
        and not any(
            marker in normalized_requested_prompt
            for marker in (
                "tool states",
                "streaming markdown",
                "retry once",
            )
        )
    )
    streaming_request = "streaming markdown" in user_prompt.lower()
    wait_command = "Get-Content -LiteralPath .\\fixture.txt -Wait"
    if process_request or cancel_gap_request:
        if "call-process-first" not in current_tool_result_ids:
            process_text, process_call = "Before first tool.\n\n", "call-process-first"
        elif process_request and "call-process-second" not in current_tool_result_ids:
            process_text, process_call = "Between tools.\n\n", "call-process-second"
        else:
            process_text, process_call = "Process final reply.", None
        chunks.append(_chunk(request_id=request_id, model=model, delta={"content": process_text}))
        if process_call is not None:
            chunks.append(_chunk(request_id=request_id, model=model, delta={"tool_calls": [{
                "index": 0, "id": process_call, "type": "function", "function": {
                    "name": "read_file", "arguments": json.dumps({"path": "fixture.txt"}),
                },
            }]}))
        chunks.append(_chunk(request_id=request_id, model=model, delta={},
                             finish_reason="tool_calls" if process_call else "stop"))
    elif needs_tool_search:
        suffix = "v1" if model_mcp_barrier else "v2"
        chunks.append(
            _chunk(
                request_id=request_id,
                model=model,
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-model-mcp-search-{suffix}",
                            "type": "function",
                            "function": {
                                "name": "tool_search",
                                "arguments": json.dumps(
                                    {
                                        "query": (
                                            "fixture_echo_v1 model mcp generation resource"
                                            if model_mcp_barrier
                                            else "fixture_echo_v2 model mcp generation resource"
                                        )
                                    }
                                ),
                            },
                        }
                    ]
                },
            )
        )
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={}, finish_reason="tool_calls")
        )
    elif model_mcp_barrier and not v1_tool_completed:
        tool_name = next((name for name in tool_names if name.endswith("fixture_echo_v1")), None)
        if tool_name is None:
            raise web.HTTPInternalServerError(text="MCP v1 fixture tool was not discovered")
        chunks.append(
            _chunk(
                request_id=request_id,
                model=model,
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-model-mcp-v1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"value": "old"}),
                            },
                        }
                    ]
                },
            )
        )
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={}, finish_reason="tool_calls")
        )
    elif model_mcp_request and not v2_tool_completed:
        tool_name = next((name for name in tool_names if name.endswith("fixture_echo_v2")), None)
        if tool_name is None:
            raise web.HTTPInternalServerError(text="MCP v2 fixture tool was not discovered")
        chunks.append(
            _chunk(
                request_id=request_id,
                model=model,
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-model-mcp-v2",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"value": "new"}),
                            },
                        }
                    ]
                },
            )
        )
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={}, finish_reason="tool_calls")
        )
    elif (model_mcp_barrier and v1_tool_completed) or (model_mcp_request and v2_tool_completed):
        result_text = (
            "New model and MCP resource completed."
            if model_mcp_request
            else "Old model and MCP resource completed."
        )
        chunks.append(_chunk(request_id=request_id, model=model, delta={"content": result_text}))
        chunks.append(_chunk(request_id=request_id, model=model, delta={}, finish_reason="stop"))
    elif confirmation_request and not has_confirmation_result:
        if CONFIRMATION_PATH is None:
            raise web.HTTPInternalServerError(text="Confirmation fixture path is not configured")
        tool_state_calls: list[tuple[str, str, dict[str, object]]] = [
            (
                "call-confirmation",
                "read_file",
                {"path": CONFIRMATION_PATH},
            )
        ]
        for index, (call_id, name, arguments) in enumerate(tool_state_calls):
            chunks.append(
                _chunk(
                    request_id=request_id,
                    model=model,
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
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={}, finish_reason="tool_calls")
        )
    elif confirmation_request and has_confirmation_result:
        chunks.append(
            _chunk(
                request_id=request_id,
                model=model,
                delta={"content": "Confirmation fixture completed."},
            )
        )
        chunks.append(_chunk(request_id=request_id, model=model, delta={}, finish_reason="stop"))
    elif tool_states_request and not has_tool_result:
        tool_calls: list[tuple[str, str, dict[str, object]]] = [
            ("call-completed", "read_file", {"path": "fixture.txt"}),
            ("call-failed", "read_file", {"path": "missing-fixture.txt"}),
            (
                "call-rejected",
                "write_file",
                {"path": ".aide/restore/protected.txt", "content": "must not write"},
            ),
            (
                "call-cancelled",
                "exec",
                {"command": wait_command, "timeout": 600},
            ),
        ]
        for index, (call_id, name, arguments) in enumerate(tool_calls):
            chunks.append(
                _chunk(
                    request_id=request_id,
                    model=model,
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
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={}, finish_reason="tool_calls")
        )
    elif streaming_request:
        for content in (
            "# Streamed answer\n\n",
            "This is **Persisted Markdown**.\n\n",
            f"```text\n{'x' * 300}\n```\n\n",
            "[safe](https://example.com) [unsafe](javascript:alert(1)) "
            "![remote](https://example.com/remote.png)\n\n",
            "The response arrived in multiple chunks.\n",
        ):
            chunks.append(_chunk(request_id=request_id, model=model, delta={"content": content}))
        chunks.append(_chunk(request_id=request_id, model=model, delta={}, finish_reason="stop"))
    else:
        chunks.append(
            _chunk(request_id=request_id, model=model, delta={"content": "Fixture response."})
        )
        chunks.append(_chunk(request_id=request_id, model=model, delta={}, finish_reason="stop"))

    if body.get("stream") is False:
        content_parts: list[str] = []
        nonstream_tool_calls: list[dict[str, object]] = []
        nonstream_finish_reason = "stop"
        for chunk in chunks:
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    content_parts.append(delta_content)
                calls = delta.get("tool_calls")
                if isinstance(calls, list):
                    nonstream_tool_calls.extend(call for call in calls if isinstance(call, dict))
            finish_value = choice.get("finish_reason")
            if isinstance(finish_value, str):
                nonstream_finish_reason = finish_value
        nonstream_message: dict[str, object] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if nonstream_tool_calls:
            nonstream_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "function": call.get("function", {}),
                }
                for call in nonstream_tool_calls
            ]
        return web.json_response(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": nonstream_message,
                        "finish_reason": nonstream_finish_reason,
                    }
                ],
            }
        )

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
        await _write_sse(response, chunks, delay=0.35 if streaming_request else 0.04,
                         hold_after_first=streaming_request and "recovery streaming" in normalized_requested_prompt)
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
    sockets = getattr(site._server, "sockets", None) if site._server is not None else None
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


async def _seed_schedule_history(home: AgentHome, workspace: Path) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    job_id = str(uuid4())
    created_at_ms = int(time.time() * 1000)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=job_id,
            message="E2E history task",
            title="E2E schedule history job",
            schedule=JobSchedule.every(3600),
            created_at_ms=created_at_ms,
            updated_at_ms=created_at_ms,
        )
    )
    created_at = datetime(2026, 9, 6, tzinfo=UTC)
    session = Session.create_schedule(
        state,
        job_id,
        now=lambda: created_at,
        title="E2E schedule history job",
    )
    usage = {"model_calls": 1, "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    for index in range(21):
        session.commit_agent_run(
            [
                {"role": "user", "content": f"Historical execution {index + 1}"},
                {
                    "role": "assistant",
                    "content": (
                        "# Persisted schedule result\n\n"
                        "```text\n"
                        "A long enough result for the production history view.\n"
                        "```"
                        if index == 0
                        else f"Historical result {index + 1}"
                    ),
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": usage,
                },
            ],
            pending_last_compacted=session.last_compacted,
            pending_action_summary=None,
        )
    session.commit_agent_run(
        [{"role": "user", "content": "Historical execution without a terminal result"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary=None,
    )
    session.close()


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
    creation_scope: str | None,
) -> str:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path, allow_agent_home_chat=True)
    session = Session.create(state, now=lambda: created_at)
    session.update_metadata(title=title)
    if creation_scope is not None:
        session.update_metadata(creation_scope=creation_scope)
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    return session.session_id


async def _seed_restore_session(
    home: AgentHome,
    workspace: Path,
    *,
    title: str,
    content: str,
    created_at: datetime,
    target_name: str,
) -> tuple[str, Path]:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=lambda: created_at)
    session.update_metadata(title=title, creation_scope="project")
    target = workspace / target_name
    target.write_text("content before Restore\n", encoding="utf-8")
    run_token = uuid4()
    session.commit_agent_run(
        [{"role": "user", "content": content}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=session.capture_restore_before(),
        restore_run_token=run_token,
    )
    store = FileBackupStore(state, session.session_id)
    ticket = store.before_write(run_token, target)
    if ticket is None:
        raise RuntimeError("E2E Restore fixture could not create a file backup")
    target.write_text("current branch\n", encoding="utf-8")
    store.after_write(ticket)
    await session.wait_for_pending_persist()
    return session.session_id, target


async def _stop_service(home: AgentHome, port: int) -> None:
    accepted = await ServiceClient.stop_existing(home, port=port)
    timeout_seconds = int(os.environ.get("AIDE_E2E_SHUTDOWN_TIMEOUT_MS", "15000")) / 1000
    deadline = time.monotonic() + timeout_seconds
    while discovery_path(home).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("E2E service did not finish shutting down.")
        if not accepted:
            accepted = await ServiceClient.stop_existing(home, port=port)
        await asyncio.sleep(0.05)


async def _run_e2e(provider_base_url: str) -> None:
    global CONFIRMATION_PATH, MODEL_MCP_ENTERED, MODEL_MCP_RELEASE, PROVIDER_OBSERVATION_PATH
    with tempfile.TemporaryDirectory(prefix="aide-web-e2e-") as root:
        path = Path(root)
        repo_root = Path(__file__).resolve().parents[2]
        CONFIRMATION_PATH = str(path / "confirmation-outside.txt")
        Path(CONFIRMATION_PATH).write_text("confirmation fixture content\n", encoding="utf-8")
        MODEL_MCP_ENTERED.clear()
        MODEL_MCP_RELEASE.clear()
        PROVIDER_OBSERVATION_PATH = path / "provider-observations.jsonl"
        PROVIDER_OBSERVATION_PATH.write_text("", encoding="utf-8")
        mcp_wire_path = repo_root / "tests" / "fixtures" / "mcp_wire.py"
        mcp_v1_path = path / "mcp-v1.json"
        mcp_v2_path = path / "mcp-v2.json"
        mcp_v1_path.write_text(
            json.dumps(
                {
                    "pages": {
                        "": {
                            "tools": [
                                {
                                    "name": "fixture_echo_v1",
                                    "description": "E2E MCP v1 fixture tool",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                }
                            ]
                        }
                    },
                    "results": {
                        "fixture_echo_v1": {"content": [{"type": "text", "text": "mcp-v1-result"}]}
                    },
                }
            ),
            encoding="utf-8",
        )
        mcp_v2_path.write_text(
            json.dumps(
                {
                    "pages": {
                        "": {
                            "tools": [
                                {
                                    "name": "fixture_echo_v2",
                                    "description": "E2E MCP v2 fixture tool",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                }
                            ]
                        }
                    },
                    "results": {
                        "fixture_echo_v2": {"content": [{"type": "text", "text": "mcp-v2-result"}]}
                    },
                }
            ),
            encoding="utf-8",
        )
        mcp_command = Path(sys.executable).as_posix()
        mcp_args = [mcp_wire_path.as_posix(), mcp_v1_path.as_posix()]
        home = AgentHome(path / ".aide")
        home.initialize()
        (home.path / "config.toml").write_text(
            _config(
                provider_base_url,
                mcp_command=mcp_command,
                mcp_args=mcp_args,
                mcp_cwd=repo_root.as_posix(),
            ) + f'\n[web]\ndefault_chat_workspace = "{(home.path / "chat").as_posix()}"\n',
            encoding="utf-8",
        )
        cli_workspace = path / "cli-workspace"
        first_project = path / "project-one"
        project_alias = path / "project-one-alias"
        second_project = path / "project-two"
        cli_workspace.mkdir()
        first_project.mkdir()
        second_project.mkdir()
        (first_project / "fixture.txt").write_text("fixture content\n", encoding="utf-8")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(project_alias), str(first_project)],
            check=True,
            capture_output=True,
        )
        await _seed_schedule_job(home, first_project)
        await _seed_schedule_history(home, first_project)
        occupied_session_id = await _seed_session(
            home,
            first_project,
            title="CLI occupied history",
            content="CLI-only history must remain private",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
            creation_scope="project",
        )
        available_session_id = await _seed_session(
            home,
            first_project,
            title="Web available history",
            content="Available history loaded after a successful Claim",
            created_at=datetime(2026, 9, 2, tzinfo=UTC),
            creation_scope="project",
        )
        restore_session_id, restore_target = await _seed_restore_session(
            home,
            first_project,
            title="Web restore history",
            content="Restore branch should disappear from history",
            created_at=datetime(2026, 9, 3, tzinfo=UTC),
            target_name="restore-fixture.txt",
        )
        manual_restore_session_id, manual_restore_target = await _seed_restore_session(
            home,
            first_project,
            title="Web manual restore history",
            content="Manual Restore branch should disappear from history",
            created_at=datetime(2026, 9, 4, tzinfo=UTC),
            target_name="manual-restore-fixture.txt",
        )
        failure_restore_session_id, failure_restore_target = await _seed_restore_session(
            home,
            first_project,
            title="Web failed restore history",
            content="Failed Restore branch",
            created_at=datetime(2026, 9, 5, tzinfo=UTC),
            target_name="failed-restore-fixture.txt",
        )
        port = _free_port()
        client = await ServiceClient.connect_or_start(home, cli_workspace, port=port)
        project_client = await ServiceClient.connect_or_start(home, first_project, port=port)
        await project_client.open_conversation(session_id=occupied_session_id)
        peer_client: ServiceClient | None = None
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
                            "restore_session_id": restore_session_id,
                            "restore_target": str(restore_target),
                            "manual_restore_session_id": manual_restore_session_id,
                            "manual_restore_target": str(manual_restore_target),
                            "failure_restore_session_id": failure_restore_session_id,
                            "failure_restore_target": str(failure_restore_target),
                            "provider_observation_path": str(PROVIDER_OBSERVATION_PATH),
                            "mcp_v1_path": str(mcp_v1_path),
                            "mcp_v2_path": str(mcp_v2_path),
                        }
                    ),
                    flush=True,
                )

            await announce()
            while True:
                command = await asyncio.to_thread(sys.stdin.readline)
                if not command or command.strip() == "stop":
                    SETTINGS_RELEASE.set()
                    PROCESS_RELEASE.set()
                    QUEUE_RECALL_RELEASE.set()
                    MODEL_MCP_RELEASE.set()
                    break
                if command.strip() == "config-read":
                    print(json.dumps(await client.get_config()), flush=True)
                    continue
                if command.strip() == "web-ticket":
                    print(json.dumps({"url": await client.create_web_ticket()}), flush=True)
                    continue
                if command.startswith("config-patch "):
                    current = await client.get_config()
                    result = await client._http_request(
                        "PATCH", "/api/v1/config", mutation=True,
                        payload={
                            "request_id": str(uuid4()), "revision": current["revision"],
                            "fields": json.loads(command.split(" ", 1)[1]), "secrets": {},
                        },
                    )
                    print(json.dumps(result), flush=True)
                    continue
                if command.startswith("cli-claim "):
                    if peer_client is None:
                        peer_client = await ServiceClient.connect_or_start(
                            home, first_project, port=port
                        )
                    try:
                        result = await peer_client.open_conversation(
                            session_id=command.strip().split()[1]
                        )
                    except ServiceError as error:
                        result = error.to_dict("cli-claim")
                    print(json.dumps(result), flush=True)
                    continue
                if command.strip() == "cli-release":
                    assert peer_client is not None
                    await peer_client.release_session()
                    print(json.dumps({"released": True}), flush=True)
                    continue
                if command.startswith("effort "):
                    result = await client.management("effort", {"effort": command.strip().split()[1]})
                    print(json.dumps(result), flush=True)
                    continue
                if command.strip() == "queue-arm":
                    QUEUE_RECALL_ENTERED.clear()
                    QUEUE_RECALL_RELEASE.clear()
                    print(json.dumps({"armed": True}), flush=True)
                    continue
                if command.strip() == "queue-wait":
                    await asyncio.wait_for(QUEUE_RECALL_ENTERED.wait(), timeout=60)
                    print(json.dumps({"holding": True}), flush=True)
                    continue
                if command.strip() == "queue-release":
                    QUEUE_RECALL_RELEASE.set()
                    print(json.dumps({"released": True}), flush=True)
                    continue
                if command.strip() == "process-arm":
                    PROCESS_ENTERED.clear()
                    PROCESS_RELEASE.clear()
                    print(json.dumps({"armed": True}), flush=True)
                    continue
                if command.strip() == "process-wait":
                    await asyncio.wait_for(PROCESS_ENTERED.wait(), timeout=60)
                    print(json.dumps({"holding": True}), flush=True)
                    continue
                if command.strip() == "process-release":
                    PROCESS_RELEASE.set()
                    print(json.dumps({"released": True}), flush=True)
                    continue
                if command.strip() == "settings-arm":
                    SETTINGS_ENTERED.clear()
                    SETTINGS_RELEASE.clear()
                    print(json.dumps({"armed": True}), flush=True)
                    continue
                if command.strip() == "settings-wait":
                    await asyncio.wait_for(SETTINGS_ENTERED.wait(), timeout=60)
                    print(json.dumps({"holding": True}), flush=True)
                    continue
                if command.strip() == "settings-hold":
                    SETTINGS_ENTERED.clear()
                    SETTINGS_RELEASE.clear()
                    for _ in range(1200):
                        try:
                            await client.submit_user_input("settings generation barrier")
                        except ServiceError as error:
                            if error.code != "admission_closed":
                                raise
                            await asyncio.sleep(0.05)
                        else:
                            break
                    else:
                        raise RuntimeError(
                            "Settings generation barrier could not enter the workspace."
                        )
                    await asyncio.wait_for(SETTINGS_ENTERED.wait(), timeout=60)
                    print(json.dumps({"holding": True, "pid": client.discovery.pid}), flush=True)
                    continue
                if command.strip() == "settings-release":
                    SETTINGS_RELEASE.set()
                    print(json.dumps({"released": True, "pid": client.discovery.pid}), flush=True)
                    continue
                if command.strip() == "model-mcp-arm":
                    MODEL_MCP_ENTERED.clear()
                    MODEL_MCP_RELEASE.clear()
                    print(json.dumps({"armed": True}), flush=True)
                    continue
                if command.strip() == "model-mcp-wait":
                    await asyncio.wait_for(MODEL_MCP_ENTERED.wait(), timeout=60)
                    print(json.dumps({"holding": True}), flush=True)
                    continue
                if command.strip() == "model-mcp-release":
                    MODEL_MCP_RELEASE.set()
                    print(json.dumps({"released": True}), flush=True)
                    continue
                if command.strip() == "project-history-seed":
                    history = []
                    for index in range(101):
                        title = f"Paginated project {index:03d}"
                        session_id = await _seed_session(
                            home,
                            path / "chat-next",
                            title=title,
                            content=f"{title} body",
                            created_at=datetime(2026, 1, 1, 0, index // 60, index % 60, tzinfo=UTC),
                            creation_scope="project",
                        )
                        history.append({"id": session_id, "title": title})
                    print(json.dumps({"history": history}), flush=True)
                    continue
                if command.strip() == "chat-history-seed":
                    history = []
                    for index in range(101):
                        title = f"Paginated chat {index:03d}"
                        session_id = await _seed_session(
                            home,
                            home.path / "chat",
                            title=title,
                            content=f"{title} body",
                            created_at=datetime(2026, 1, 1, 0, index // 60, index % 60, tzinfo=UTC),
                            creation_scope="chat",
                        )
                        history.append({"id": session_id, "title": title})
                    for title, directory, scope in (
                        ("Saved chat", home.path / "chat", "chat"),
                        ("Unscoped chat", home.path / "chat", None),
                        ("Saved project", path / "chat-next", "project"),
                        ("Unscoped project", path / "chat-next", None),
                        ("Explicit project", path / "chat-next", "project"),
                    ):
                        session_id = await _seed_session(
                            home,
                            directory,
                            title=title,
                            content=f"{title} body",
                            created_at=datetime(2026, 1, 2, tzinfo=UTC),
                            creation_scope=scope,
                        )
                        history.append({"id": session_id, "title": title})
                    print(json.dumps({"history": history}), flush=True)
                    continue
                if command.strip() != "restart":
                    continue
                if peer_client is not None:
                    await peer_client.close()
                    peer_client = None
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
            if peer_client is not None:
                await peer_client.close()
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
