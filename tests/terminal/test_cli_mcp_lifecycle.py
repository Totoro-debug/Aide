"""MCP lifecycle through the real CLI service adapter and Workspace owner."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from aide.agent.session.session import Session
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from aide.service.client import ServiceClient
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.cli_service import cli_service
from tests.fixtures.mcp_wire import ObservedLifetimes, stdio_wire_configuration, wire_tool
from tests.service.test_service_concurrency import _client_output, _ConcurrentProvider


class _EchoProvider(_ConcurrentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.schemas: list[list[dict[str, Any]]] = []

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        if str(messages[0].get("content", "")).startswith("Generate a concise title"):
            return super().stream(**kwargs)
        tools = kwargs["tools"]
        self.schemas.append(list(tools))
        last_user = max(i for i, item in enumerate(messages) if item["role"] == "user")
        tail = messages[last_user + 1 :]
        user = messages[last_user]["content"]
        value = "second" if "second" in user else "first"
        remote = next(
            (item for item in tools if item["function"]["name"] == "mcp_local_echo"), None
        )
        done = any(
            item["role"] == "tool" and item["content"] == f"{value}:inherited" for item in tail
        )
        name = "mcp_local_echo" if remote else "tool_search"
        arguments = {"value": value} if remote else {"query": "echo"}

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if done:
                yield TextDelta("done")
            yield ModelCompleted(
                ModelResponse(
                    message=AssistantModelMessage(
                        content="done" if done else "",
                        tool_calls=()
                        if done
                        else (
                            ModelToolCall(
                                id=f"{name}-{value}", name=name, arguments=json.dumps(arguments)
                            ),
                        ),
                    ),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop" if done else "tool_calls",
                )
            )

        return emit()


@pytest.mark.asyncio
@pytest.mark.parametrize("save_before_next_run", [False, True])
async def test_cli_real_mcp_flow_persists_result_reuses_session_snapshot_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, save_before_next_run: bool
) -> None:
    observed = ObservedLifetimes(monkeypatch)
    tasks_before = asyncio.all_tasks()
    home = AgentHome(tmp_path / "home")
    home.initialize()
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.setenv("AIDE_MCP_FLOW_TEST", "inherited")
    script = "\n".join(
        (
            "import os",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('cli-flow')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return f\"{value}:{os.environ['AIDE_MCP_FLOW_TEST']}\"",
            "server.run()",
        )
    )
    content = MINIMAL_VALID_CONFIG.replace(
        "[runtime]", '[runtime]\npermission_level = "full-access"'
    ) + (
        '\n[mcp.servers.local]\nenabled = true\ntransport = "stdio"\n'
        f"command = {json.dumps(sys.executable)}\n"
        f"args = {json.dumps(['-c', script])}\n"
        'connect_timeout = 10\ncall_timeout = 5\ntool_keywords = {echo = ["echo", "text"]}\n'
    )
    (home.path / "config.toml").write_text(content, encoding="utf-8")
    provider = _EchoProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda *_args: provider)
    async with cli_service(home) as service:
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            workspace = service.workspace(client.workspace_id)
            original = workspace.resources
            assert original is not None
            old_tools = original.mcp_snapshot
            schemas = [tool.to_schema() for tool in old_tools]
            await client.submit_user_input("first echo")
            assert "done" in await _client_output(client)
            selected = await client.management_dispatcher.resume(client.session_id)
            assert selected.resumed_session_id == client.session_id
            assert workspace.resources is original
            assert original.mcp_snapshot[0] is old_tools[0]
            assert len(observed.processes) == 1
            if save_before_next_run:
                await service.update_configuration(
                    "reconnect-cli-mcp",
                    cast(str, service.config_view()["revision"]),
                    {"runtime": {"max_iterations": 83}},
                )
                assert cast(dict[str, object], service.config_view()["application"])["status"] == "next-run-required"
                assert workspace.resources is original
                assert [tool.to_schema() for tool in old_tools] == schemas
                assert len(observed.processes) == 1
            await client.submit_user_input("second echo")
            assert "done" in await _client_output(client)
            assert len(observed.processes) == 1
            assert cast(dict[str, object], service.config_view()["application"])["status"] == "active"
            if save_before_next_run:
                assert workspace.configuration.runtime.max_iterations == 83
            claim = workspace.require_claim(
                client.client_id, client.session_id, client.claim_version
            )
            await claim.loop.session.wait_for_pending_persist()
            persisted = Session.load(workspace.workspace_state, client.session_id)
            contents = [
                item.get("content") for item in persisted.messages if item["role"] == "tool"
            ]
            assert "first:inherited" in contents
            assert "second:inherited" in contents
            exposed = [
                item
                for request in provider.schemas
                for item in request
                if item["function"]["name"] == "mcp_local_echo"
            ]
            assert exposed and all(
                item["function"]["parameters"]["type"] == "object" for item in exposed
            )
        finally:
            await client.close()
    observed.assert_closed()
    assert asyncio.all_tasks() - tasks_before == set()


@pytest.mark.asyncio
async def test_cli_workspace_wire_discovery_retains_safe_aggregate_skip_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    configuration = stdio_wire_configuration(
        tmp_path,
        {
            "pages": {
                "": {
                    "tools": [
                        wire_tool(),
                        {"name": "PRIVATE_INVALID_TOOL"},
                        wire_tool("invalid", inputSchema="PRIVATE_SCHEMA"),
                    ]
                }
            }
        },
        name="local",
    )
    content = MINIMAL_VALID_CONFIG + (
        '\n[mcp.servers.local]\nenabled = true\ntransport = "stdio"\n'
        f"command = {json.dumps(configuration.command)}\n"
        f"args = {json.dumps(list(configuration.args))}\n"
        'tool_keywords = {echo = ["echo"]}\n'
    )
    (home.path / "config.toml").write_text(content, encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.setattr(
        "aide.service.runtime.service.create_provider", lambda *_args: _ConcurrentProvider()
    )
    observed = ObservedLifetimes(monkeypatch)
    async with cli_service(home) as service:
        client = await ServiceClient.connect_or_start(home, directory)
        try:
            runtime = service.workspace(client.workspace_id).resources
            assert runtime is not None
            report = runtime.mcp_startup_report
            assert report.failed_servers == ()
            assert report.skipped_tool_counts == (("local", 2),)
            assert [tool.name for tool in report.snapshot] == ["mcp_local_echo"]
            assert "PRIVATE" not in repr(report)
            assert await report.snapshot[0].execute_prepared({}) == "wire text"
        finally:
            await client.close()
    observed.assert_closed()
