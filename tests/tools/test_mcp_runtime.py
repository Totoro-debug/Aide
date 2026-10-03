from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent

import omni.agent.tools.mcp as mcp_adapter
from omni.agent.tools.mcp import MCPServerConnection, MCPTool, MCPToolSpec
from omni.agent.tools.mcp_runtime import (
    MCPRuntimeManager,
    MCPToolSnapshot,
    allocate_mcp_tool_name,
)
from omni.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from omni.config.config import MCPServerConfiguration
from tests.fixtures.mcp_wire import (
    ObservedLifetimes,
    http_wire_server,
    stdio_requests,
    stdio_wire_configuration,
    wire_result,
)


@pytest.mark.asyncio
async def test_real_idle_stdio_eof_reconnects_on_first_generation_and_ignores_old_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = ObservedLifetimes(monkeypatch)
    manager = MCPRuntimeManager(tmp_path)
    configuration = stdio_wire_configuration(tmp_path, {})
    tasks_before = asyncio.all_tasks()
    try:
        initial = await manager.start({"remote": configuration})
        assert [tool.server_name for tool in initial.snapshot] == ["remote"]
        original_schemas = [tool.to_schema() for tool in initial.snapshot]
        assert all(r["method"] != "tools/call" for r in stdio_requests(tmp_path))
        await observed.stop(0)
        candidate = await manager.prepare_generation()
        assert candidate.failed_servers == ()
        assert [tool.to_schema() for tool in initial.snapshot] == original_schemas
        assert candidate.snapshot[0] is not initial.snapshot[0]
        assert await candidate.snapshot[0].execute_prepared({}) == "wire text"
        old_result = await ToolGateway._for_memory(initial.snapshot).call(
            ModelToolCall(id="late", name=initial.snapshot[0].name, arguments="{}")
        )
        assert old_result.content == "MCP Server connection is unavailable."
        manager.activate_generation(candidate)
        next_candidate = await manager.prepare_generation()
        assert next_candidate.snapshot[0] is candidate.snapshot[0]
        assert await next_candidate.snapshot[0].execute_prepared({}) == "wire text"
        methods = [r["method"] for r in stdio_requests(tmp_path)]
        assert methods.count("initialize") == methods.count("tools/list") == 2
    finally:
        async with asyncio.timeout(10):
            await manager.close()
    observed.assert_closed()
    assert asyncio.all_tasks() - tasks_before == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_reconnect", [False, True])
async def test_real_reconnect_reuses_healthy_server_and_preserves_old_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_reconnect: bool,
) -> None:
    observed = ObservedLifetimes(monkeypatch)
    async with http_wire_server({}) as (healthy, http_configuration):
        healthy_configuration = MCPServerConfiguration(
            mcp_name="healthy",
            enabled=True,
            transport="streamable-http",
            url=http_configuration.url,
            connect_timeout=10,
            call_timeout=5,
        )
        manager = MCPRuntimeManager(tmp_path)
        try:
            initial = await manager.start(
                {
                    "remote": stdio_wire_configuration(tmp_path, {}),
                    "healthy": healthy_configuration,
                }
            )
            assert {tool.server_name for tool in initial.snapshot} == {"healthy", "remote"}
            schemas = [tool.to_schema() for tool in initial.snapshot]
            if fail_reconnect:
                (tmp_path / "remote.json").write_text(json.dumps({"pages": {"": {}}}))
            await observed.stop(0)
            candidate = await manager.prepare_generation()
            assert candidate.failed_servers == (("remote",) if fail_reconnect else ())
            assert candidate.snapshot[0] is initial.snapshot[0]
            assert [tool.to_schema() for tool in initial.snapshot] == schemas
            assert len(candidate.snapshot) == (1 if fail_reconnect else 2)
            for tool in candidate.snapshot:
                assert await tool.execute_prepared({}) == "wire text"
            remote_methods = [r["method"] for r in stdio_requests(tmp_path)]
            healthy_methods = [r["method"] for r in healthy.requests]
            assert remote_methods.count("initialize") == remote_methods.count("tools/list") == 2
            assert healthy_methods.count("initialize") == healthy_methods.count("tools/list") == 1
        finally:
            async with asyncio.timeout(10):
                await manager.close()
    observed.assert_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [{}, {"tools": {}}, {"tools": [], "nextCursor": 42}])
async def test_real_wire_envelope_failure_is_isolated_to_one_server(
    tmp_path: Path,
    page: dict[str, Any],
) -> None:
    async with http_wire_server({}) as (healthy, http_configuration):
        manager = MCPRuntimeManager(tmp_path)
        try:
            report = await manager.start(
                {
                    "remote": http_configuration,
                    "broken": stdio_wire_configuration(
                        tmp_path, {"pages": {"": page}}, name="broken"
                    ),
                }
            )
            assert report.failed_servers == ("broken",)
            assert len(report.failures) == 1
            assert report.skipped_tool_counts == ()
            assert await report.snapshot[0].execute_prepared({}) == "wire text"
            assert [r["method"] for r in stdio_requests(tmp_path, "broken")].count(
                "tools/list"
            ) == 1
            assert [r["method"] for r in healthy.requests].count("tools/list") == 1
        finally:
            await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "is_error", "protocol_error", "cancel"])
async def test_real_sdk_call_failures_do_not_reconnect_healthy_http_session(
    tmp_path: Path,
    failure: str,
) -> None:
    scenario = {"results": {"echo": wire_result(isError=True)}} if failure == "is_error" else {}
    async with http_wire_server(scenario) as (server, configuration):
        if failure == "timeout":
            configuration = replace(configuration, call_timeout=1)
        manager = MCPRuntimeManager(tmp_path)
        try:
            report = await manager.start({"remote": configuration})
            tool = report.snapshot[0]
            arguments = (
                {"hang": True}
                if failure in {"timeout", "cancel"}
                else {"error": failure == "protocol_error"}
            )
            task = asyncio.create_task(
                ToolGateway._for_memory((tool,)).call(
                    ModelToolCall(id="failure", name=tool.name, arguments=json.dumps(arguments))
                )
            )
            try:
                await server.wait_for("tools/call")
                if failure == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    async with asyncio.timeout(10):
                        result = await task
                    assert result.status == "error"
                if failure == "cancel":
                    await server.wait_for("notifications/cancelled")
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            candidate = await manager.prepare_generation()
            assert candidate.snapshot == report.snapshot
            assert not tool.unavailable
            assert [r["method"] for r in server.requests].count("initialize") == 1
            assert [r["method"] for r in server.requests].count("tools/list") == 1
        finally:
            await manager.close()


@pytest.mark.asyncio
async def test_real_http_sse_rebuild_is_healthy_but_sdk_read_eof_requires_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = ObservedLifetimes(monkeypatch)
    streams_seen: list[Any] = []

    @asynccontextmanager
    async def observed_transport(configuration: MCPServerConfiguration, workspace: Path) -> Any:
        async with mcp_adapter.streamable_http_transport(configuration, workspace) as streams:
            streams_seen.append(streams)
            yield streams

    async with http_wire_server({}) as (server, configuration):
        connection = MCPServerConnection(
            configuration, tmp_path, transport_factory=observed_transport
        )
        try:
            tools = await connection.connect()
            async with asyncio.timeout(10):
                await server.sse_reconnected.wait()
            assert not connection.unavailable
            assert not observed.closed[0].is_set()
            assert await tools[0].execute_prepared({}) == "wire text"
            await streams_seen[0][0].aclose()
            async with asyncio.timeout(10):
                await observed.closed[0].wait()
            assert connection.unavailable
        finally:
            await connection.close()
    observed.assert_closed()


class _ResultSession:
    def __init__(self, result: object | None = None, error: BaseException | None = None) -> None:
        self.result = (
            result if result is not None else CallToolResult(content=[TextContent(text="ok")])
        )
        self.error = error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        if self.error is not None:
            raise self.error
        return self.result


class _FakeConnection:
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        tools: tuple[MCPTool, ...] = (),
        *,
        tools_factory: Callable[[], tuple[MCPTool, ...]] | None = None,
    ) -> None:
        self.configuration = configuration
        self._tools = tools
        self._tools_factory = tools_factory
        self.connect_calls = 0
        self.close_calls = 0
        self.unavailable = False
        self.skipped_tool_count = 0
        self.fail_connect: BaseException | None = None
        self.fail_close: BaseException | None = None

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return self._tools

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        if self.fail_connect is not None:
            raise self.fail_connect
        self.unavailable = False
        if self._tools_factory is not None:
            self._tools = self._tools_factory()
        return self._tools

    async def close(self) -> None:
        self.close_calls += 1
        if self.fail_close is not None:
            raise self.fail_close

    def mark_unavailable(self) -> None:
        self.unavailable = True


class _BlockingConnection(_FakeConnection):
    def __init__(self, configuration: MCPServerConfiguration) -> None:
        super().__init__(configuration)
        self.connect_started = asyncio.Event()
        self.connect_cancelled = False

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        self.connect_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("blocking connection unexpectedly resumed")
        except asyncio.CancelledError:
            self.connect_cancelled = True
            raise


class _ConcurrentConnection(_FakeConnection):
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        started: asyncio.Event,
        release: asyncio.Event,
        *,
        completed: asyncio.Event | None = None,
        failure: BaseException | None = None,
    ) -> None:
        super().__init__(configuration)
        self._started = started
        self._release = release
        self._completed = completed
        self._failure = failure

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        self._started.set()
        await self._release.wait()
        if self._failure is not None:
            raise self._failure
        self.unavailable = False
        if self._completed is not None:
            self._completed.set()
        return self._tools


class _DelayedCloseConnection(_FakeConnection):
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        release_close: asyncio.Event,
    ) -> None:
        super().__init__(configuration)
        self._release_close = release_close
        self.close_started = asyncio.Event()
        self.close_completed = False
        self.close_cancelled = False

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        try:
            await self._release_close.wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise
        self.close_completed = True


def _configuration(name: str) -> MCPServerConfiguration:
    return MCPServerConfiguration(
        mcp_name=name,
        enabled=True,
        transport="stdio",
        command="mcp-server",
    )


def _tool(
    server_name: str,
    remote_name: str,
    *,
    description: str | None = None,
    session: _ResultSession | None = None,
    on_closed: Callable[[], None] | None = None,
) -> MCPTool:
    model_name = allocate_mcp_tool_name(server_name, remote_name)
    assert model_name is not None
    return MCPTool(
        MCPToolSpec(
            server_name=server_name,
            remote_name=remote_name,
            model_name=model_name,
            description=description or remote_name,
            parameters={"type": "object"},
        ),
        session or _ResultSession(),
        on_closed=on_closed,
    )


class _ObservedManager(MCPRuntimeManager):
    @property
    def snapshot(self) -> MCPToolSnapshot:
        return self._snapshot


def _manager(
    connections: Mapping[str, _FakeConnection],
    *,
    built_in_names: tuple[str, ...] = (),
) -> _ObservedManager:
    return _ObservedManager(
        Path("."),
        built_in_names=built_in_names,
        connection_factory=lambda configuration, workspace: connections[configuration.mcp_name],
    )


@pytest.mark.asyncio
async def test_runtime_connects_two_servers_without_waiting_for_a_third_server_timeout() -> None:
    success_release = asyncio.Event()
    timeout_release = asyncio.Event()
    alpha_started = asyncio.Event()
    beta_started = asyncio.Event()
    zulu_started = asyncio.Event()
    alpha_completed = asyncio.Event()
    beta_completed = asyncio.Event()
    configurations = {name: _configuration(name) for name in ("alpha", "beta", "zulu")}
    connections = {
        "alpha": _ConcurrentConnection(
            configurations["alpha"],
            alpha_started,
            success_release,
            completed=alpha_completed,
        ),
        "beta": _ConcurrentConnection(
            configurations["beta"],
            beta_started,
            success_release,
            completed=beta_completed,
        ),
        "zulu": _ConcurrentConnection(
            configurations["zulu"],
            zulu_started,
            timeout_release,
            failure=TimeoutError("sensitive timeout detail"),
        ),
    }
    manager = _manager(connections)

    start_task = asyncio.create_task(manager.start(configurations))
    await asyncio.gather(
        alpha_started.wait(),
        beta_started.wait(),
        zulu_started.wait(),
    )

    assert start_task.done() is False
    success_release.set()
    await asyncio.wait_for(
        asyncio.gather(alpha_completed.wait(), beta_completed.wait()),
        timeout=1,
    )
    assert start_task.done() is False

    timeout_release.set()
    report = await start_task

    assert report.failed_servers == ("zulu",)
    assert all(connection.connect_calls == 1 for connection in connections.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_name", "expected_name"),
    [
        ("echo", "mcp_alpha_echo"),
        ("r" * 54, "mcp_alpha_" + "r" * 54),
        ("r" * 55, "mcp_" + "r" * 55),
        ("r" * 61, None),
        ("bad.tool", None),
    ],
)
async def test_default_connection_discovers_provider_safe_names_and_reuses_tools(
    monkeypatch: pytest.MonkeyPatch,
    remote_name: str,
    expected_name: str | None,
) -> None:
    configuration = _configuration("alpha")
    lifecycle: list[str] = []

    @asynccontextmanager
    async def transport(
        configuration: MCPServerConfiguration, workspace: Path
    ) -> AsyncIterator[object]:
        lifecycle.append("transport_entered")
        yield ("read", "write")
        lifecycle.append("transport_closed")

    class Session(_ResultSession):
        list_calls = 0

        async def __aenter__(self) -> Session:
            lifecycle.append("session_entered")
            return self

        async def __aexit__(self, *args: object) -> None:
            lifecycle.append("session_closed")

        async def initialize(self) -> object:
            lifecycle.append("initialized")
            return None

        async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
            assert params is None
            self.list_calls += 1
            return {"tools": [{"name": remote_name, "inputSchema": {"type": "object"}}]}

    session = Session()
    monkeypatch.setattr(mcp_adapter, "_transport_for", transport)
    monkeypatch.setattr(mcp_adapter, "_new_client_session", lambda read, write, **kwargs: session)
    manager = MCPRuntimeManager(Path("."))
    try:
        report = await manager.start({"alpha": configuration})

        assert [tool.name for tool in report.snapshot] == (
            [] if expected_name is None else [expected_name]
        )
        assert report.skipped_tool_counts == ((("alpha", 1),) if expected_name is None else ())
        candidate = await manager.prepare_generation()
        assert candidate.snapshot == report.snapshot
        if expected_name is not None:
            assert candidate.snapshot[0] is report.snapshot[0]
            assert await candidate.snapshot[0].execute_prepared({}) == "ok"
        assert session.list_calls == 1
        assert lifecycle == ["transport_entered", "session_entered", "initialized"]
    finally:
        await manager.close()
    assert lifecycle[-2:] == ["session_closed", "transport_closed"]


@pytest.mark.parametrize(
    ("mcp_name", "remote_name", "expected_name"),
    [
        ("alpha", "echo", "mcp_alpha_echo"),
        ("alpha", "r" * 54, "mcp_alpha_" + "r" * 54),
        ("alpha", "r" * 55, "mcp_" + "r" * 55),
        ("alpha", "r" * 61, None),
        ("alpha", "bad.tool", None),
    ],
)
def test_allocate_mcp_tool_name_matches_provider_contract(
    mcp_name: str,
    remote_name: str,
    expected_name: str | None,
) -> None:
    assert allocate_mcp_tool_name(mcp_name, remote_name) == expected_name


@pytest.mark.asyncio
async def test_runtime_snapshot_skips_builtin_and_fallback_collisions_deterministically() -> None:
    long_remote_name = "r" * 59
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", long_remote_name),))
    zulu_connection = _FakeConnection(zulu, (_tool("zulu", long_remote_name),))

    projections: list[tuple[tuple[str, str], ...]] = []
    for _ in range(3):
        alpha_connection._tools = (_tool("alpha", long_remote_name),)
        zulu_connection._tools = (_tool("zulu", long_remote_name),)
        manager = _manager(
            {"alpha": alpha_connection, "zulu": zulu_connection},
            built_in_names=("mcp_alpha_echo",),
        )
        report = await manager.start({"zulu": zulu, "alpha": alpha})
        projections.append(tuple((tool.name, tool.description) for tool in report.snapshot))

    assert projections == [(("mcp_" + long_remote_name, long_remote_name),)] * 3
    assert report.skipped_tool_counts == (("zulu", 1),)


@pytest.mark.asyncio
async def test_runtime_snapshot_skips_collision_with_a_builtin_tool() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    manager = _manager({"alpha": connection}, built_in_names=("mcp_alpha_echo",))
    report = await manager.start({"alpha": configuration})

    assert report.snapshot == ()
    assert report.skipped_tool_counts == (("alpha", 1),)


@pytest.mark.asyncio
async def test_runtime_reports_skipped_tool_counts() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    connection.skipped_tool_count = 2
    manager = _manager({"alpha": connection})

    report = await manager.start({"alpha": configuration})

    assert report.skipped_tool_counts == (("alpha", 2),)


@pytest.mark.asyncio
async def test_rebuilding_same_configuration_keeps_snapshot_schema_deterministic() -> None:
    configuration = _configuration("alpha")
    projections: list[tuple[tuple[str, object], ...]] = []

    for _ in range(100):
        connection = _FakeConnection(
            configuration,
            (
                _tool("alpha", "second", description="Second"),
                _tool("alpha", "first", description="First"),
            ),
        )
        report = await _manager({"alpha": connection}).start({"alpha": configuration})
        projections.append(tuple((tool.name, tool.to_schema()) for tool in report.snapshot))

    assert (
        projections
        == [
            (
                (
                    "mcp_alpha_first",
                    {
                        "type": "function",
                        "function": {
                            "name": "mcp_alpha_first",
                            "description": "First",
                            "parameters": {"type": "object"},
                        },
                    },
                ),
                (
                    "mcp_alpha_second",
                    {
                        "type": "function",
                        "function": {
                            "name": "mcp_alpha_second",
                            "description": "Second",
                            "parameters": {"type": "object"},
                        },
                    },
                ),
            )
        ]
        * 100
    )


@pytest.mark.asyncio
async def test_prepare_generation_reuses_healthy_connection_and_discovered_tools() -> None:
    configuration = _configuration("alpha")
    tool = _tool("alpha", "echo")
    connection = _FakeConnection(configuration, (tool,))
    manager = _manager({"alpha": connection})

    initial = await manager.start({"alpha": configuration})
    candidate = await manager.prepare_generation()

    assert connection.connect_calls == 1
    assert candidate.snapshot == initial.snapshot
    assert candidate.snapshot[0] is initial.snapshot[0]
    assert initial.snapshot[0] is tool
    assert manager.activate_generation(candidate) is candidate.snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(content=[TextContent(text="server error")], is_error=True),
        CallToolResult(content=[TextContent(text="another error")], is_error=True),
    ],
)
async def test_timeout_and_is_error_results_do_not_enter_failed_server_set(result: object) -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)
    connection._tools = (_tool("alpha", "echo", session=_ResultSession(result)),)
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call", name=connection.tools[0].name, arguments="{}")
    )
    candidate = await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 1
    assert candidate.failed_servers == ()


@pytest.mark.asyncio
async def test_timeout_does_not_enter_failed_server_set() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)
    connection._tools = (
        _tool("alpha", "echo", session=_ResultSession(error=TimeoutError("timed out"))),
    )
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call-timeout", name=connection.tools[0].name, arguments="{}")
    )
    await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 1


@pytest.mark.asyncio
async def test_closed_session_enters_failed_set_and_is_retried_for_next_generation() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)

    def replacement_tool() -> tuple[MCPTool, ...]:
        return (
            _tool(
                "alpha",
                "echo",
                session=_ResultSession(error=MCPError(types.CONNECTION_CLOSED, "closed")),
                on_closed=connection.mark_unavailable,
            ),
        )

    connection._tools_factory = replacement_tool
    connection._tools = replacement_tool()
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call-closed", name=connection.tools[0].name, arguments="{}")
    )
    candidate = await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 2
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_echo"]


@pytest.mark.asyncio
async def test_failed_candidate_keeps_previous_snapshot_unchanged() -> None:
    alpha = _configuration("alpha")
    beta = _configuration("beta")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", "alpha-tool"),))
    beta_connection = _FakeConnection(beta, (_tool("beta", "beta-tool"),))
    manager = _manager({"alpha": alpha_connection, "beta": beta_connection})

    initial = await manager.start({"alpha": alpha, "beta": beta})
    beta_connection.mark_unavailable()
    beta_connection.fail_connect = RuntimeError("connection failed")

    candidate = await manager.prepare_generation()

    assert initial.snapshot == (
        initial.snapshot[0],
        initial.snapshot[1],
    )
    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_alpha-tool"]
    assert candidate.failed_servers == ("beta",)


@pytest.mark.asyncio
async def test_healthy_collision_loser_can_win_after_previous_server_fails() -> None:
    long_remote_name = "r" * 59
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", long_remote_name),))
    zulu_connection = _FakeConnection(zulu, (_tool("zulu", long_remote_name),))
    manager = _manager({"alpha": alpha_connection, "zulu": zulu_connection})

    initial = await manager.start({"alpha": alpha, "zulu": zulu})
    assert [tool.name for tool in initial.snapshot] == ["mcp_" + long_remote_name]

    alpha_connection.mark_unavailable()
    alpha_connection.fail_connect = RuntimeError("alpha is still unavailable")
    candidate = await manager.prepare_generation()

    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_" + long_remote_name]
    assert alpha_connection.connect_calls == 2
    assert zulu_connection.connect_calls == 1
    assert candidate.failed_servers == ("alpha",)


@pytest.mark.asyncio
async def test_activate_generation_replaces_snapshot_only_after_candidate_is_selected() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "first"),))
    manager = _manager({"alpha": connection})

    initial = await manager.start({"alpha": configuration})
    connection.mark_unavailable()
    connection._tools = (_tool("alpha", "second"),)
    candidate = await manager.prepare_generation()

    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_second"]

    activated = manager.activate_generation(candidate)

    assert activated == candidate.snapshot


@pytest.mark.asyncio
async def test_cancelled_connection_cancels_and_settles_sibling_connections() -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    blocking = _BlockingConnection(alpha)
    cancelled = _FakeConnection(zulu)
    cancelled.fail_connect = asyncio.CancelledError()
    manager = _manager({"alpha": blocking, "zulu": cancelled})

    with pytest.raises(asyncio.CancelledError):
        await manager.start({"alpha": alpha, "zulu": zulu})

    assert blocking.connect_started.is_set()
    assert blocking.connect_cancelled is True
    assert blocking.close_calls == 1
    assert cancelled.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, RuntimeError])
async def test_close_attempts_every_connection_before_propagating_failure(
    error_type: type[BaseException],
) -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    first = _FakeConnection(alpha)
    second = _FakeConnection(zulu)
    manager = _manager({"alpha": first, "zulu": second})
    await manager.start({"alpha": alpha, "zulu": zulu})
    first.fail_close = error_type("close failed")

    with pytest.raises(error_type):
        await manager.close()

    assert first.close_calls == 1
    assert second.close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_manager_close_waits_for_every_connection_to_finish() -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    release_close = asyncio.Event()
    first = _DelayedCloseConnection(alpha, release_close)
    second = _DelayedCloseConnection(zulu, release_close)
    manager = _manager({"alpha": first, "zulu": second})
    await manager.start({"alpha": alpha, "zulu": zulu})
    close_task = asyncio.create_task(manager.close())
    await asyncio.gather(first.close_started.wait(), second.close_started.wait())

    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)

    assert close_task.done() is False
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert first.close_completed is True
    assert second.close_completed is True
    assert first.close_cancelled is False
    assert second.close_cancelled is False


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_candidate_from_another_manager() -> None:
    configuration = _configuration("alpha")
    first_connection = _FakeConnection(configuration, (_tool("alpha", "first"),))
    second_connection = _FakeConnection(configuration, (_tool("alpha", "second"),))
    first_manager = _manager({"alpha": first_connection})
    second_manager = _manager({"alpha": second_connection})
    await first_manager.start({"alpha": configuration})
    initial = await second_manager.start({"alpha": configuration})
    foreign_candidate = await first_manager.prepare_generation()

    with pytest.raises(ValueError, match="current prepared candidate"):
        second_manager.activate_generation(foreign_candidate)

    assert second_manager.snapshot is initial.snapshot


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_superseded_candidate() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    manager = _manager({"alpha": connection})
    initial = await manager.start({"alpha": configuration})
    stale_candidate = await manager.prepare_generation()
    current_candidate = await manager.prepare_generation()

    with pytest.raises(ValueError, match="current prepared candidate"):
        manager.activate_generation(stale_candidate)

    assert manager.snapshot is initial.snapshot
    assert manager.activate_generation(current_candidate) == current_candidate.snapshot


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_candidate_after_close() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})
    candidate = await manager.prepare_generation()
    await manager.close()

    with pytest.raises(RuntimeError, match="has not been started"):
        manager.activate_generation(candidate)

    assert manager.snapshot == ()
