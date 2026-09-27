from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import myclaw.agent.tools.mcp as mcp_adapter
import myclaw.agent.tools.mcp_runtime as mcp_runtime
import myclaw.terminal.cli as cli
from myclaw.agent.runner import AgentRunner
from myclaw.agent.tools.mcp import MCPServerConnection
from myclaw.agent.tools.mcp_runtime import MCPServerFailure, MCPStartupReport
from myclaw.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from myclaw.config.config import ConfigLoader, MCPServerConfiguration
from myclaw.errors import ErrorInfo
from myclaw.management.service import ManagementError
from myclaw.provider.model_router import ModelRouteStatus
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.session import Session
from myclaw.workspace.state import WorkspaceState
from tests.fixtures import DetachedRequestPreparer
from tests.fixtures.mcp_wire import ObservedLifetimes, stdio_wire_configuration, wire_tool


def _fake_memory_route_status() -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="test-provider",
        model="test-model",
        context_window=200_000,
        max_output=8_192,
        used_default=False,
    )


def _configuration() -> Any:
    return SimpleNamespace(
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
        mcp={},
    )


def test_cli_reports_one_safe_notice_for_each_failure_and_one_aggregate_skip_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)

    cli._report_mcp_generation(
        MCPStartupReport(
            snapshot=(),
            failed_servers=("fallback", "broken"),
            failures=(
                MCPServerFailure(
                    mcp_name="broken",
                    phase="connect",
                    exception_type="TimeoutError",
                ),
            ),
            skipped_tool_counts=(("partial", 2),),
        )
    )

    assert notices == [
        "MCP Server 'broken' unavailable during connect (TimeoutError).",
        "MCP Server 'fallback' unavailable during connect (MCPConnectionError).",
        "MCP Server 'partial' skipped 2 invalid MCP Tools.",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_name", ["broken-one", "broken-two", "broken-three"])
async def test_cli_starts_mcp_before_initial_loop_and_closes_it_after_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_name: str,
) -> None:
    events: list[str] = []
    notices: list[str] = []
    keyword_calls: list[dict[str, object]] = []
    remote_tool = SimpleNamespace(
        server_name="github",
        remote_name="search_issues",
        name="mcp_github_search_issues",
        description="Search GitHub issues.",
        parameters={"type": "object", "properties": {}},
    )

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            assert workspace == (tmp_path / "workspace").resolve()
            assert kwargs["built_in_names"]
            events.append("mcp_init")

        async def start(self, configuration: object) -> object:
            assert configuration == {}
            events.append("mcp_start")
            return SimpleNamespace(
                snapshot=(remote_tool,),
                failed_servers=(failed_name,),
                failures=(
                    MCPServerFailure(
                        mcp_name=failed_name,
                        phase="connect",
                        exception_type="TimeoutError",
                    ),
                ),
                skipped_tool_counts=(),
            )

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        pass

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def complete(
            self,
            route: str,
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
        ) -> ModelResponse:
            events.append("keywords_prepare")
            keyword_calls.append({"route": route, "messages": messages, "tools": tools})
            return ModelResponse(
                message=AssistantModelMessage(content='["issues", "search"]'),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="stop",
            )

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            events.append("schedule_start")

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            events.append("schedule_close")

        def resume(self) -> None:
            pass

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["mcp_tools"] == (remote_tool,)
            events.append("loop_init")
            self.control = object()
            self.skill_metadata = ()

        def preflight(self) -> None:
            events.append("loop_preflight")

        async def start(self) -> None:
            events.append("loop_start")

        async def close(self) -> None:
            events.append("loop_close")

        async def abort(self) -> None:
            events.append("loop_abort")

        def project_foreground_conversation(self) -> object:
            return object()

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            events.append("app_run")

    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )

    assert events.index("mcp_start") < events.index("keywords_prepare")
    assert events.index("keywords_prepare") < events.index("loop_init")
    assert len(keyword_calls) == 1
    keyword_call = keyword_calls[0]
    assert keyword_call["route"] == "chat"
    assert keyword_call["tools"] == ()
    serialized_messages = json.dumps(keyword_call["messages"], ensure_ascii=False)
    assert "search_issues" in serialized_messages
    assert "Search GitHub issues." in serialized_messages
    assert "input_schema" in serialized_messages
    assert "transport" not in serialized_messages
    assert events.index("loop_close") < events.index("mcp_close")
    assert events.index("mcp_close") < events.index("dream_close")
    assert events.index("dream_close") < events.index("router_close")
    assert events.index("schedule_pause") < events.index("loop_close")
    assert notices == [f"MCP Server '{failed_name}' unavailable during connect (TimeoutError)."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preparation_error",
    [
        RuntimeError("sensitive transport endpoint"),
        TimeoutError("sensitive timeout detail"),
        OSError("sensitive protocol detail"),
    ],
    ids=("transport", "timeout", "protocol"),
)
async def test_cli_keeps_old_generation_when_mcp_candidate_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preparation_error: Exception,
) -> None:
    events: list[str] = []
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Any] | None = None
    old_loop: object | None = None

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            del workspace, kwargs

        async def start(self, configuration: object) -> object:
            del configuration
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def prepare_generation(self) -> object:
            events.append("mcp_prepare")
            raise preparation_error

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            pass

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            pass

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            pass

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            pass

        def resume(self) -> None:
            pass

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal old_loop
            assert kwargs["session_id"] is None
            old_loop = self
            self.control = SimpleNamespace(has_active_run=False)
            self.skill_metadata = ()
            self.session = SimpleNamespace(
                session_id="old-session",
                wait_for_pending_persist=lambda: None,
            )

        def preflight(self) -> None:
            pass

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def abort(self) -> None:
            events.append("old_abort")

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            nonlocal current_callback, replace_callback
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(Callable[[str, bool], Any], kwargs["replace_agent_loop"])

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            with pytest.raises(ManagementError) as raised:
                await replace_callback("target-session", False)
            assert raised.value.error == ErrorInfo(
                "persistence_error",
                "MCP Tool Generation could not be prepared.",
            )
            assert "schedule_pause" not in events
            assert "old_abort" not in events
            assert current_callback is not None
            assert current_callback() is old_loop

    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", lambda: object())
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_name", ["broken-one", "broken-two", "broken-three"])
async def test_cli_uses_failed_mcp_candidate_without_mutating_old_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_name: str,
) -> None:
    events: list[str] = []
    notices: list[str] = []
    old_loop: Any | None = None
    target_loop: Any | None = None
    replace_callback: Callable[[str, bool], Any] | None = None
    current_callback: Callable[[], object] | None = None
    initial_tool = object()
    keyword_snapshots: list[tuple[object, ...]] = []

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            assert workspace == (tmp_path / "workspace").resolve()
            assert kwargs["built_in_names"]

        async def start(self, configuration: object) -> object:
            del configuration
            return SimpleNamespace(
                snapshot=(initial_tool,),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def prepare_generation(self) -> object:
            events.append("mcp_prepare")
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(failed_name,),
                failures=(
                    MCPServerFailure(
                        mcp_name=failed_name,
                        phase="connect",
                        exception_type="TimeoutError",
                    ),
                ),
                skipped_tool_counts=(),
            )

        def activate_generation(self, report: object) -> tuple[object, ...]:
            assert cast(Any, report).failed_servers == (failed_name,)
            assert current_callback is not None
            with pytest.raises(ManagementError):
                current_callback()
            events.append("mcp_activate")
            return ()

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeMCPKeywordPreparer:
        def __init__(self, model_router: object, config_loader: ConfigLoader) -> None:
            assert model_router is not None
            assert isinstance(config_loader, ConfigLoader)
            assert config_loader.path == tmp_path / "agent-home" / "config.toml"

        async def prepare(self, snapshot: Sequence[object], servers: object) -> object:
            assert servers == {}
            snapshot_tuple = tuple(snapshot)
            keyword_snapshots.append(snapshot_tuple)
            events.append(f"keywords_prepare_{len(keyword_snapshots)}")
            return {f"generation-{len(keyword_snapshots)}": ("keyword",)}

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        async def reset(self) -> None:
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            assert "mcp_tools" not in kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            pass

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            events.append("schedule_close")

        def resume(self) -> None:
            events.append("schedule_resume")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        mcp_tools: tuple[object, ...]
        mcp_keywords: dict[str, tuple[str, ...]]

        def __init__(self, **kwargs: object) -> None:
            nonlocal old_loop, target_loop
            self.mcp_tools = tuple(cast(Sequence[object], kwargs["mcp_tools"]))
            self.mcp_keywords = dict(cast(Any, kwargs["mcp_keywords"]))
            self.session = SimpleNamespace(session_id=kwargs["session_id"] or "old")
            self.control = SimpleNamespace(has_active_run=False)
            self.skill_metadata = ()
            if kwargs["session_id"] is None:
                old_loop = self
                events.append("old_init")
            else:
                target_loop = self
                events.append("target_init")

        def preflight(self) -> None:
            pass

        async def start(self) -> None:
            events.append("target_start" if self is target_loop else "old_start")

        async def close(self) -> None:
            events.append("target_close" if self is target_loop else "old_close")

        async def abort(self) -> None:
            events.append("target_abort" if self is target_loop else "old_abort")

        async def _pause_for_replacement(self) -> None:
            events.append("replacement_pause")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            assert resume_inbound is True
            assert current_callback is not None
            assert current_callback() is target_loop
            events.append("publish_observed")
            events.append("barrier_release")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            nonlocal current_callback, replace_callback
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(Callable[[str, bool], Any], kwargs["replace_agent_loop"])

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            await replace_callback("target", False)
            assert old_loop is not None
            assert target_loop is not None
            assert old_loop.mcp_tools == (initial_tool,)
            assert target_loop.mcp_tools == ()
            assert old_loop.mcp_keywords == {"generation-1": ("keyword",)}
            assert target_loop.mcp_keywords == {"generation-2": ("keyword",)}

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")

    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "MCPKeywordPreparer", FakeMCPKeywordPreparer)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )

    assert old_loop is not None
    assert target_loop is not None
    assert old_loop.mcp_tools == (initial_tool,)
    assert target_loop.mcp_tools == ()
    assert old_loop.mcp_keywords == {"generation-1": ("keyword",)}
    assert target_loop.mcp_keywords == {"generation-2": ("keyword",)}
    assert keyword_snapshots == [(initial_tool,), ()]
    assert notices == [f"MCP Server '{failed_name}' unavailable during connect (TimeoutError)."]
    assert events.index("keywords_prepare_1") < events.index("old_init")
    assert events.index("mcp_prepare") < events.index("keywords_prepare_2")
    assert events.index("keywords_prepare_2") < events.index("replacement_pause")
    assert events.index("rebind") < events.index("target_start")
    assert events.index("target_start") < events.index("mcp_activate")
    assert events.index("mcp_activate") < events.index("publish_observed")
    assert events.index("publish_observed") < events.index("barrier_release")
    assert events.index("barrier_release") < events.index("schedule_resume")
    assert events[-5:] == [
        "schedule_close",
        "target_close",
        "mcp_close",
        "dream_close",
        "router_close",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("closed_before_resume", [False, True])
async def test_cli_real_mcp_flow_persists_result_reuses_connection_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closed_before_resume: bool,
) -> None:
    observed = ObservedLifetimes(monkeypatch)
    tasks_before = asyncio.all_tasks()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MYCLAW_MCP_FLOW_TEST", "inherited")
    server_script = "\n".join(
        (
            "import os",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('cli-flow')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return f\"{value}:{os.environ['MYCLAW_MCP_FLOW_TEST']}\"",
            "server.run()",
        )
    )
    configuration = _configuration()
    configuration.mcp = {
        "local": MCPServerConfiguration(
            mcp_name="local",
            enabled=True,
            transport="stdio",
            command=sys.executable,
            args=("-c", server_script),
            cwd=workspace,
            connect_timeout=10,
            call_timeout=5,
            tool_keywords={"echo": ("echo", "text")},
        )
    }

    connections: list[MCPServerConnection] = []
    transport_events: list[str] = []
    loops: list[Any] = []
    schema_requests: list[tuple[dict[str, Any], ...]] = []
    keyword_calls: list[object] = []
    replace_callback: Callable[[str, bool], Any] | None = None
    default_connection_factory = mcp_runtime._default_connection_factory
    stdio_transport = mcp_adapter.stdio_transport

    @asynccontextmanager
    async def observed_transport(
        server_configuration: MCPServerConfiguration,
        server_workspace: Path,
    ) -> AsyncIterator[object]:
        async with stdio_transport(server_configuration, server_workspace) as streams:
            transport_events.append("entered")
            yield streams
        transport_events.append("closed")

    def connection_factory(
        server_configuration: MCPServerConfiguration,
        server_workspace: Path,
    ) -> MCPServerConnection:
        connection = default_connection_factory(server_configuration, server_workspace)
        connections.append(connection)
        return connection

    class FlowRouter:
        def __init__(self, value: str) -> None:
            self.value = value
            self.calls = 0

        def stream(
            self,
            route: str,
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelCompleted]:
            del continuation
            assert route == "chat"
            request_number = self.calls
            self.calls += 1
            schema_requests.append(tuple(tools))
            mcp_schema = next(
                schema for schema in tools if schema["function"]["name"] == "mcp_local_echo"
            )
            assert mcp_schema["function"]["parameters"]["type"] == "object"

            async def events() -> AsyncIterator[ModelCompleted]:
                if request_number == 0:
                    yield ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id=f"call-{self.value}",
                                        name="mcp_local_echo",
                                        arguments=json.dumps({"value": self.value}),
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    )
                    return
                assert messages[-1]["role"] == "tool"
                assert messages[-1]["content"] == f"{self.value}:inherited"
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="done"),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="stop",
                    )
                )

            return events()

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def complete(self, *args: object, **kwargs: object) -> ModelResponse:
            keyword_calls.append((args, kwargs))
            return ModelResponse(
                message=AssistantModelMessage(content='["must", "not", "run"]'),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="stop",
            )

        async def close(self) -> None:
            pass

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run during the MCP delivery flow")

        async def close(self) -> None:
            pass

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            pass

        async def pause_and_drain(self) -> None:
            pass

        async def close(self) -> None:
            pass

        def resume(self) -> None:
            pass

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            session_id = cast(str | None, kwargs["session_id"])
            workspace_state = cast(WorkspaceState, kwargs["workspace_state"])
            self.mcp_tools = tuple(cast(Sequence[Any], kwargs["mcp_tools"]))
            self.session = (
                Session.create(workspace_state)
                if session_id is None
                else Session.load(workspace_state, session_id)
            )
            self.control = SimpleNamespace(has_active_run=False)
            self.skill_metadata: tuple[object, ...] = ()
            self.value = f"generation-{len(loops) + 1}"
            loops.append(self)

        def preflight(self) -> None:
            assert [tool.name for tool in self.mcp_tools] == ["mcp_local_echo"]

        async def start(self) -> None:
            user_message = {"role": "user", "content": f"echo {self.value}"}
            gateway = ToolGateway._for_memory(cast(Any, self.mcp_tools))
            result = await AgentRunner(
                cast(Any, FlowRouter(self.value)),
                DetachedRequestPreparer(),
            ).run(
                [user_message],
                model="chat",
                tool_gateway=gateway,
                on_output=None,
                confirmation=None,
                externalize_result=None,
                cancel_requested=None,
                max_iterations=50,
            )
            assert result.final_content == "done"
            self.session.commit_agent_run(
                [user_message, *result.messages],
                pending_last_compacted=self.session.last_compacted,
                pending_action_summary=cast(str, self.session.metadata["summary"]),
            )
            await self.session.wait_for_pending_persist()

        async def close(self) -> None:
            self.session.close()

        async def abort(self) -> None:
            self.session.close()

        async def _pause_for_replacement(self) -> None:
            assert not connections[0].unavailable
            assert len(observed.processes) == (2 if closed_before_resume else 1)
            assert len(loops) == 1

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            del resume_inbound

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            nonlocal replace_callback
            replace_callback = cast(Callable[[str, bool], Any], kwargs["replace_agent_loop"])

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            if closed_before_resume:
                await observed.stop(0)
                assert connections[0].unavailable
            await replace_callback(loops[0].session.session_id, False)

        async def quiesce_for_rebind(self) -> None:
            pass

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs

    monkeypatch.setattr(mcp_runtime, "_default_connection_factory", connection_factory)
    monkeypatch.setattr(mcp_adapter, "stdio_transport", observed_transport)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=workspace,
        configuration=configuration,
    )

    assert len(connections) == 1
    assert len(loops) == 2
    assert (loops[0].mcp_tools[0] is loops[1].mcp_tools[0]) is not closed_before_resume
    assert len(schema_requests) == 4
    persisted = Session.load(loops[0].session.workspace_state, loops[0].session.session_id)
    assert [message["content"] for message in persisted.messages if message["role"] == "tool"] == [
        "generation-1:inherited",
        "generation-2:inherited",
    ]
    assert transport_events == ["entered", "closed"] * (2 if closed_before_resume else 1)
    assert keyword_calls == []
    observed.assert_closed()
    assert asyncio.all_tasks() - tasks_before == set()


@pytest.mark.asyncio
async def test_cli_real_wire_discovery_emits_one_aggregate_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)
    scenario = {
        "pages": {
            "": {
                "tools": [
                    wire_tool(),
                    {"name": "private-array", "inputSchema": []},
                    {"name": "private-missing"},
                    wire_tool("private-root", inputSchema={"type": "string"}),
                ]
            }
        }
    }
    manager = mcp_runtime.MCPRuntimeManager(tmp_path)
    try:
        report = await manager.start({"remote": stdio_wire_configuration(tmp_path, scenario)})
        assert [tool.server_name for tool in report.snapshot] == ["remote"]
        cli._report_mcp_generation(report)
        assert notices == ["MCP Server 'remote' skipped 3 invalid MCP Tools."]
        assert await report.snapshot[0].execute_prepared({}) == "wire text"
    finally:
        await manager.close()
