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
from tests.service.test_service_concurrency import _ConcurrentProvider


class _EchoProvider(_ConcurrentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.schemas: list[list[dict[str, Any]]] = []
        self.router_dispatches = 0
        self.debug_requests: list[tuple[str, set[str], list[tuple[str, str, str]]]] = []
        self.echo_completed = {value: asyncio.Event() for value in ("first", "second")}

    @staticmethod
    def _events(
        content: str, *, tool_call: ModelToolCall | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        calls = () if tool_call is None else (tool_call,)

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if content:
                yield TextDelta(content)
            yield ModelCompleted(
                ModelResponse(
                    message=AssistantModelMessage(content=content, tool_calls=calls),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls" if calls else "stop",
                )
            )

        return emit()

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        if str(messages[0].get("content", "")).startswith("Generate a concise title"):
            return super().stream(**kwargs)
        tools = kwargs["tools"]
        self.schemas.append(list(tools))
        tool_names = {
            item["function"]["name"]
            for item in tools
            if isinstance(item.get("function"), dict)
            and isinstance(item["function"].get("name"), str)
        }
        last_user = max(i for i, item in enumerate(messages) if item["role"] == "user")
        tail = messages[last_user + 1 :]
        user = str(messages[last_user]["content"])
        self.debug_requests.append(
            (
                user[-180:],
                tool_names,
                [
                    (
                        str(item.get("role", "")),
                        str(item.get("name", "")),
                        str(item.get("content", ""))[-100:],
                    )
                    for item in messages[-5:]
                ],
            )
        )
        value = "second" if "second" in user else "first"

        if "dispatch_worker" in tool_names:
            reported = "Worker reports for automatic continuation:" in user
            dispatched = any(
                item.get("role") == "tool" and item.get("name") == "dispatch_worker"
                for item in tail
            )
            if reported:
                return self._events(
                    "done",
                    tool_call=ModelToolCall(
                        f"mcp-finish-{self.router_dispatches}",
                        "finish",
                        '{"response":"done"}',
                    ),
                )
            if dispatched:
                return self._events(
                    "",
                    tool_call=ModelToolCall(
                        f"mcp-wait-{self.router_dispatches}",
                        "finish",
                        '{"response":"The Worker is working on the assigned task."}',
                    ),
                )
            self.router_dispatches += 1
            return self._events(
                "",
                tool_call=ModelToolCall(
                    f"mcp-dispatch-{self.router_dispatches}",
                    "dispatch_worker",
                    json.dumps(
                        {
                            "task_id": f"mcp-worker-{self.router_dispatches}",
                            "task": user,
                            "context": [],
                        }
                    ),
                ),
            )

        remote = next(
            (item for item in tools if item["function"]["name"] == "mcp_local_echo"), None
        )
        done = any(
            item.get("role") == "tool" and item.get("content") == f"{value}:inherited"
            for item in messages
        )
        if done:
            self.echo_completed[value].set()
            return self._events("done")
        name = "mcp_local_echo" if remote else "tool_search"
        arguments = {"value": value} if remote else {"query": "echo"}
        return self._events(
            "",
            tool_call=ModelToolCall(
                f"{name}-{value}", name, json.dumps(arguments)
            ),
        )


async def _client_output_containing(client: ServiceClient, expected: str) -> list[str]:
    output: list[str] = []
    async with asyncio.timeout(30):
        while True:
            message = await client.bus.get_outbound()
            output.append(message.content)
            if expected in message.content:
                return output


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
            await asyncio.wait_for(provider.echo_completed["first"].wait(), timeout=15)
            try:
                output = await _client_output_containing(client, "done")
            except TimeoutError as error:
                raise AssertionError(provider.debug_requests) from error
            assert "done" in output, (output, provider.debug_requests)
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
            await asyncio.wait_for(provider.echo_completed["second"].wait(), timeout=15)
            assert "done" in await _client_output_containing(client, "done")
            async with asyncio.timeout(15):
                while True:
                    workers = workspace.worker_store(client.session_id).load().workers
                    if len(workers) == 2 and all(
                        worker.report is not None for worker in workers
                    ):
                        break
                    await asyncio.sleep(0.01)
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
                item.get("content")
                for worker in workers
                for item in worker.conversation
                if item.get("role") == "tool"
            ]
            assert "first:inherited" in contents
            assert "second:inherited" in contents
            assert any(item.get("role") == "user" for item in persisted.messages)
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
