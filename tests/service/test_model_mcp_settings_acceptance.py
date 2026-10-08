"""Model/MCP settings handover through real HTTP, WS and Schedule execution."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiohttp
import pytest
from aiohttp import web
from loguru import logger

from aide.agent.session.session import Session
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.client import ServiceClient
from aide.service.runtime import AgentService
from tests.configuration.test_config_editing import FULL_EDITABLE_CONFIG
from tests.fixtures.mcp_wire import WireServer, wire_result, wire_tool
from tests.service.test_service_concurrency import _client_output, _serve


@pytest.mark.asyncio
async def test_model_and_http_mcp_save_preserves_foreground_schedule_and_existing_resources(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    observed: list[str] = []
    log_sink = logger.add(lambda message: observed.append(str(message)), format="{message}")
    gates = {name: asyncio.Event() for name in ("foreground-old", "scheduled-old")}
    arrived = {name: asyncio.Event() for name in gates}
    cancelled: list[str] = []
    model_calls: list[dict[str, Any]] = []
    header_calls: list[tuple[str, str, bool]] = []
    wires = {
        version: WireServer(
            {
                "pages": {"": {"tools": [wire_tool("echo", description="Echo resource")]}},
                "results": {"echo": wire_result(f"{version} resource")},
            }
        )
        for version in ("old", "new")
    }
    keys = {"small-model": "provider-old-canary-302", "large-model": "provider-new-canary-302"}
    header_values = {"old": "mcp-old-canary-302", "new": "mcp-new-canary-302"}

    async def mcp(request: web.Request) -> web.StreamResponse:
        version = request.match_info["version"]
        header_calls.append(
            (version, request.method, request.headers.get("X-Api-Key") == header_values[version])
        )
        return await wires[version].http(request)

    async def completion(request: web.Request) -> web.Response:
        body = await request.json()
        model = body["model"]
        messages = body["messages"]
        user = next(
            (str(item.get("content", "")) for item in reversed(messages) if item["role"] == "user"),
            "",
        )
        last_user = max(
            (index for index, item in enumerate(messages) if item["role"] == "user"), default=-1
        )
        current_messages = messages[last_user + 1 :]
        user = user.rsplit("## User Input\n\n", 1)[-1]
        tools = [item["function"]["name"] for item in body.get("tools", [])]
        key_matches = request.headers.get("Authorization") == f"Bearer {keys[model]}"
        assert key_matches, "Model request used the wrong generation credential"
        model_calls.append(
            {
                "user": user,
                "model": model,
                "max_output": body.get("max_tokens"),
                "temperature": body.get("temperature"),
                "effort": body.get("reasoning_effort"),
                "tools": tools,
                "key_matches": key_matches,
            }
        )
        if tools and user in gates and not any(item["role"] == "tool" for item in current_messages):
            arrived[user].set()
            try:
                await gates[user].wait()
            except asyncio.CancelledError:
                cancelled.append(user)
                raise
        if not tools:
            delta: dict[str, Any] = {"content": "Resource acceptance"}
            finish = "stop"
        elif any(item.get("tool_call_id") == "resource-call" for item in current_messages):
            delta = {"content": f"{user} finished"}
            finish = "stop"
        else:
            remote = next((name for name in tools if name.endswith("echo")), None)
            tool = remote or "tool_search"
            arguments = {} if remote else {"query": "resource"}
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "resource-call" if remote else "search-call",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                ]
            }
            finish = "tool_calls"
        chunk = {
            "id": "fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if not body.get("stream"):
            message = {"role": "assistant", **delta}
            for call in message.get("tool_calls", []):
                call.pop("index", None)
            return web.json_response(
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                }
            )
        return web.Response(
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n", content_type="text/event-stream"
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_route("*", "/{version}/mcp", mcp)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    base = f"http://127.0.0.1:{runner.addresses[0][1]}"
    home = AgentHome(tmp_path / "home")
    home.initialize()
    content = FULL_EDITABLE_CONFIG.replace("https://models.example/v1", f"{base}/v1").replace(
        "provider-secret-canary-302", keys["small-model"]
    )
    content = content.replace('model = "large-model"', 'model = "small-model"').replace(
        'url = "https://mcp.example/tools"', f'url = "{base}/old/mcp"'
    )
    content = (
        content.replace("mcp-header-canary-302", header_values["old"])
        .replace("Authorization =", '"X-Api-Key" =')
        .replace('search = ["query"]', 'echo = ["resource"]')
    )
    content += '\n[runtime]\npermission_level = "full-access"\n'
    path = home.path / "config.toml"
    path.write_text(content, encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server, port = await _serve(service, home)
    project = tmp_path / "project"
    project.mkdir()
    cli: ServiceClient | None = None
    pid, identity = os.getpid(), service.service_instance_id
    try:
        cli = await ServiceClient.connect_or_start(home, project, port=port)
        workspace = service.workspace(cli.workspace_id)
        old = workspace.resources
        state = workspace.workspace_state
        socket = cli._socket
        claim = (cli.session_id, cli.claim_version, cli.claim_credential)
        await cli.submit_user_input("foreground-old")
        await asyncio.wait_for(arrived["foreground-old"].wait(), 10)
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="scheduled-old",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(job)
        try:
            await asyncio.wait_for(arrived["scheduled-old"].wait(), 10)
        except TimeoutError:
            pytest.fail(f"Schedule did not reach provider: {model_calls}")
        revision = service.config_view()["revision"]
        browser = await service.register_client("web")
        headers = {
            "Authorization": f"Bearer {cli.token}",
            "X-Aide-CSRF": cli.token,
            "X-Aide-Client": browser.client_id,
            "X-Aide-Control": cast(str, browser.web_control_credential),
        }
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(
                server.make_url("/api/v1/events"),
                headers={**headers, "Origin": str(server.make_url("/")).rstrip("/")},
                protocols=("aide-v1",),
            ) as ws:
                await ws.send_json(
                    {
                        "request_id": "settings-resource-subscribe",
                        "type": "subscribe",
                        "payload": {},
                    }
                )
                async with asyncio.timeout(10):
                    while True:
                        event = await ws.receive_json()
                        observed.append(json.dumps(event))
                        if event.get("request_id") == "settings-resource-subscribe":
                            assert event.get("accepted") is True
                            break
                payload = {
                    "request_id": "model-resource-change",
                    "revision": revision,
                    "fields": {
                        "models": {
                            "routes": {
                                name: {
                                    "model": "large-model",
                                    "max_output": 1536,
                                    "temperature": 0.4,
                                    "reasoning_effort": "high",
                                    "timeout": 40,
                                }
                                for name in ("default", "chat", "memory", "schedule")
                            }
                        },
                        "mcp": {
                            "http": {
                                "url": f"{base}/new/mcp",
                                "headers": {"X-Api-Key": {"configured": True}},
                            },
                            "stdio": {"cwd": None, "args": ["--flag", "--flag", "a,b", " x ", ""]},
                        },
                    },
                    "secrets": {
                        "models.providers.primary.api_key": {
                            "action": "replace",
                            "value": keys["large-model"],
                        },
                        "mcp.http.headers.X-Api-Key": {
                            "action": "replace",
                            "value": header_values["new"],
                        },
                    },
                }
                response = await http.patch(
                    server.make_url("/api/v1/config"), headers=headers, json=payload
                )
                saved = await response.json()
                observed.append(json.dumps(saved))
                assert response.status == 200, saved
                assert saved["application"]["status"] == "next-run-required"
                assert workspace.resources is old and old is not None and not workspace._closed
                gates["foreground-old"].set()
                assert "foreground-old finished" in await _client_output(cli)
                assert workspace.resources is old and not workspace._closed
                assert (
                    cast(dict[str, Any], service.config_view()["application"])["status"]
                    == "next-run-required"
                )
                assert not any(version == "new" for version, _, _ in header_calls)
                gates["scheduled-old"].set()
                assert workspace.resources is old and not workspace._closed
                async with asyncio.timeout(10):
                    while True:
                        event = await ws.receive_json()
                        observed.append(json.dumps(event))
                        if (
                            event.get("type") == "config.application"
                            and event["payload"]["status"] == "next-run-required"
                        ):
                            break
                assert not ws.closed
                await ws.ping(b"still-live")
                for value in [True, 17, [], {"token": "invalid-header-canary-302"}]:
                    before = path.read_bytes()
                    invalid_payload = {
                        "request_id": str(uuid4()),
                        "revision": service.config_view()["revision"],
                        "fields": {},
                        "secrets": {
                            "mcp.http.headers.X-Api-Key": {"action": "replace", "value": value}
                        },
                    }
                    invalid = await http.patch(
                        server.make_url("/api/v1/config"), headers=headers, json=invalid_payload
                    )
                    observed.append(await invalid.text())
                    assert invalid.status == 422
                    assert path.read_bytes() == before
                replay = await http.patch(
                    server.make_url("/api/v1/config"), headers=headers, json=payload
                )
                assert replay.status == 200
                conflict = await http.patch(
                    server.make_url("/api/v1/config"),
                    headers=headers,
                    json={**payload, "secrets": {}},
                )
                observed.append(await conflict.text())
                assert conflict.status == 409
        await cli.submit_user_input("foreground-new")
        assert "foreground-new finished" in await _client_output(cli)
        new_job = ScheduleJob(
            job_id=str(uuid4()),
            message="scheduled-new",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await workspace.schedule_service.add_user_job(new_job)
        async with asyncio.timeout(10):
            while not any(
                item.job_id == new_job.job_id and item.state.last_status == "ok"
                for item in await workspace.schedule_service.public_snapshot()
            ):
                await asyncio.sleep(0.01)
        for user, model in (
            ("foreground-old", "small-model"),
            ("scheduled-old", "small-model"),
            ("foreground-new", "large-model"),
            ("scheduled-new", "large-model"),
        ):
            calls = [item for item in model_calls if item["user"] == user and item["tools"]]
            assert calls and all(item["model"] == model and item["key_matches"] for item in calls)
            assert any(any(name.endswith("echo") for name in item["tools"]) for item in calls)
            if model == "large-model":
                assert all(
                    item["max_output"] == 1536
                    and item["temperature"] == 0.4
                    and item["effort"] == "high"
                    for item in calls
                )
        assert sum(item["method"] == "tools/call" for item in wires["old"].requests) == 2
        assert sum(item["method"] == "initialize" for item in wires["old"].requests) == 1
        assert sum(item["method"] == "tools/call" for item in wires["new"].requests) == 2
        assert sum(item["method"] == "initialize" for item in wires["new"].requests) == 1
        assert all(matches for _, _, matches in header_calls)
        assert cancelled == []
        assert workspace.workspace_state is state
        assert cli._socket is socket and socket is not None and not socket.closed
        assert (cli.session_id, cli.claim_version, cli.claim_credential) == claim
        assert os.getpid() == pid and service.service_instance_id == identity
        assert all(
            item.state.last_status == "ok"
            for item in await workspace.schedule_service.public_snapshot()
            if item.job_id in {job.job_id, new_job.job_id}
        )
        await workspace.loops[cli.session_id].loop.session.wait_for_pending_persist()
        persisted = Session.load(cast(Any, state), cli.session_id)
        assert all(
            any(message.get("content") == f"{user} finished" for message in persisted.messages)
            for user in ("foreground-old", "foreground-new")
        )
    finally:
        for gate in gates.values():
            gate.set()
        if cli is not None:
            await cli.close()
        await server.close()
        await service.stop()
        for wire in wires.values():
            wire.close_sse.set()
        await runner.cleanup()
        logger.remove(log_sink)
    public = "\n".join(observed) + caplog.text
    for secret in [*keys.values(), *header_values.values(), "invalid-header-canary-302"]:
        assert secret not in public
