from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult

from omni.agent.tools.base import BaseTool
from omni.agent.tools.deferred import RUN_BASELINE_TOOL_NAMES, build_agent_run_gateway
from omni.agent.tools.mcp import MCPTool, MCPToolSpec
from omni.agent.tools.mcp_runtime import MCPRuntimeManager, allocate_mcp_tool_name
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)
from omni.agent.workspace_state import WorkspaceState
from omni.config.config import MCPServerConfiguration
from omni.schedule.service import ScheduleService


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _gateway(
    workspace: Path,
    agent_home: Path,
    *,
    skill_root: Path | None = None,
    additional_tools: tuple[MCPTool, ...] = (),
) -> ToolGateway:
    identity = workspace
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    return ToolGateway(
        workspace=identity,
        schedule_service=ScheduleService(
            workspace_state=state,
            clock=_Clock(),
            execute_user_job=_noop,
            execute_dream=_noop,
        ),
        skill_root=skill_root,
        additional_tools=additional_tools,
    )


async def _noop(*args: object) -> None:
    del args


def _names(gateway: ToolGateway) -> list[str]:
    return [definition["function"]["name"] for definition in gateway.schemas]


class _MCPCallSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        self.calls.append((name, arguments))
        return CallToolResult(content=[])


def test_fixed_catalog_order_and_detached_definitions(
    workspace: Path,
    agent_home: Path,
) -> None:
    gateway = _gateway(workspace, agent_home)

    assert _names(gateway) == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "exec",
        "web_search",
        "web_fetch",
        "schedule",
    ]
    definitions = gateway.schemas
    next_definitions = gateway.schemas
    assert definitions == next_definitions
    assert definitions is not next_definitions
    assert definitions[0] is not next_definitions[0]
    assert isinstance(definitions, list)
    function = cast(dict[str, object], definitions[0]["function"])
    function["name"] = "changed"
    parameters = cast(dict[str, object], function["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    path = cast(dict[str, object], properties["path"])
    path["description"] = "changed"
    definitions[0]["type"] = "changed"
    assert next_definitions[0]["type"] == "function"
    assert next_definitions[0]["function"]["name"] == "read_file"
    assert (
        next_definitions[0]["function"]["parameters"]["properties"]["path"]["description"]
        != "changed"
    )
    assert _names(gateway)[0] == "read_file"
    current_function = cast(dict[str, object], gateway.schemas[0]["function"])
    current_parameters = cast(dict[str, object], current_function["parameters"])
    current_properties = cast(dict[str, object], current_parameters["properties"])
    current_path = cast(dict[str, object], current_properties["path"])
    assert current_path["description"] != "changed"
    assert not hasattr(gateway, "register_tools")
    assert hasattr(gateway, "for_run")
    assert not any(name in vars(gateway) for name in ("workspace", "schedule_store"))


def test_tool_gateway_identifies_micro_compression_eligible_tool_origins(
    workspace: Path,
    agent_home: Path,
) -> None:
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="generated",
            model_name="remote_generated_name",
            description="A dynamically named remote Tool.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    gateway = _gateway(workspace, agent_home, additional_tools=(remote_tool,))

    assert all(
        gateway.is_micro_compression_eligible(name)
        for name in (
            "exec",
            "glob",
            "grep",
            "list_dir",
            "read_file",
            "web_fetch",
            "web_search",
        )
    )
    assert gateway.is_micro_compression_eligible(remote_tool.name)
    assert not any(
        gateway.is_micro_compression_eligible(name)
        for name in ("edit_file", "write_file", "schedule", "tool_search", "unknown")
    )

    class LooksLikeMCPTool(BaseTool):
        name = "mcp_not_an_mcp_tool"
        description = "A local Tool with an MCP-looking name."
        parameters: ClassVar[dict[str, Any]] = {"type": "object"}

        async def execute(self) -> str:
            return ""

    run_gateway = gateway.for_run(exposed_names=(), run_tools=(LooksLikeMCPTool(),))
    assert not run_gateway.is_micro_compression_eligible("mcp_not_an_mcp_tool")


def test_agent_run_gateway_starts_with_search_baseline_and_activates_deferred_tools(
    workspace: Path,
    agent_home: Path,
) -> None:
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    gateway = _gateway(workspace, agent_home, additional_tools=(remote_tool,))

    run_gateway = build_agent_run_gateway(
        gateway,
        mcp_keywords={"mcp_alpha_calendar_events": ("calendar", "events")},
    )

    assert tuple(_names(run_gateway)) == RUN_BASELINE_TOOL_NAMES
    assert "mcp_alpha_calendar_events" not in _names(run_gateway)
    result = asyncio.run(
        run_gateway.call(
            ModelToolCall(
                id="search-calendar",
                name="tool_search",
                arguments=json.dumps({"query": "calendar"}),
            )
        )
    )

    assert result.status == "success"
    assert json.loads(result.content) == ["mcp_alpha_calendar_events"]
    assert "mcp_alpha_calendar_events" in _names(run_gateway)
    assert tuple(run_gateway.exposed_names) == (
        *RUN_BASELINE_TOOL_NAMES[:7],
        "mcp_alpha_calendar_events",
        "tool_search",
    )


@pytest.mark.parametrize("level", ["read-only", "workspace-write", "full-access"])
def test_mcp_catalog_exposure_activation_and_search_ignore_permission_level(
    workspace: Path,
    agent_home: Path,
    level: str,
) -> None:
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object", "properties": {}},
        ),
        _MCPCallSession(),
    )
    run_gateway = build_agent_run_gateway(
        _gateway(workspace, agent_home, additional_tools=(remote_tool,)),
        mcp_keywords={"mcp_alpha_calendar_events": ("calendar", "events")},
        permission_context=PermissionContext(
            level=level,  # type: ignore[arg-type]
            origin="foreground",
        ),
    )
    catalog_names = tuple(tool.name for tool in run_gateway.catalog)
    baseline_schema = next(
        schema
        for schema in run_gateway.schemas
        if schema["function"]["name"] == "read_file"
    )
    expected_mcp_schema = remote_tool.to_schema()

    assert "mcp_alpha_calendar_events" in catalog_names
    assert "mcp_alpha_calendar_events" not in run_gateway.exposed_names
    result = asyncio.run(
        run_gateway.call(
            ModelToolCall(
                id=f"search-calendar-{level}",
                name="tool_search",
                arguments=json.dumps({"query": "calendar"}),
            )
        )
    )

    assert result.status == "success"
    assert json.loads(result.content) == ["mcp_alpha_calendar_events"]
    assert "mcp_alpha_calendar_events" in run_gateway.exposed_names
    assert next(
        schema
        for schema in run_gateway.schemas
        if schema["function"]["name"] == "mcp_alpha_calendar_events"
    ) == expected_mcp_schema
    assert next(
        schema
        for schema in run_gateway.schemas
        if schema["function"]["name"] == "read_file"
    ) == baseline_schema


@pytest.mark.asyncio
async def test_agent_run_gateway_indexes_remote_mcp_name_and_keywords_not_allocated_name(
    workspace: Path,
    agent_home: Path,
) -> None:
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    run_gateway = build_agent_run_gateway(
        _gateway(workspace, agent_home, additional_tools=(remote_tool,)),
        mcp_keywords={"mcp_alpha_calendar_events": ("appointments",)},
    )

    allocated_name_result = await run_gateway.call(
        ModelToolCall(
            id="search-allocated-prefix",
            name="tool_search",
            arguments=json.dumps({"query": "alpha"}),
        )
    )
    keyword_result = await run_gateway.call(
        ModelToolCall(
            id="search-keywords",
            name="tool_search",
            arguments=json.dumps({"query": "calendar appointments"}),
        )
    )

    assert json.loads(allocated_name_result.content) == []
    assert json.loads(keyword_result.content) == ["mcp_alpha_calendar_events"]


@pytest.mark.asyncio
async def test_agent_run_gateway_does_not_boost_exact_remote_name_fallback(
    workspace: Path,
    agent_home: Path,
) -> None:
    earlier_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events from alpha.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    fallback_tool = MCPTool(
        MCPToolSpec(
            server_name="zulu",
            remote_name="calendar_events",
            model_name="mcp_zulu_calendar_events",
            description="Read calendar events from zulu.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    run_gateway = build_agent_run_gateway(
        _gateway(
            workspace,
            agent_home,
            additional_tools=(earlier_tool, fallback_tool),
        ),
        mcp_keywords={"mcp_zulu_calendar_events": ("calendar_events",)},
    )

    result = await run_gateway.call(
        ModelToolCall(
            id="search-calendar",
            name="tool_search",
            arguments=json.dumps({"query": "calendar"}),
        )
    )

    assert result.status == "success"
    assert json.loads(result.content) == [
        "mcp_alpha_calendar_events",
        "mcp_zulu_calendar_events",
    ]


@pytest.mark.asyncio
async def test_agent_run_gateway_direct_unexposed_call_does_not_activate_tool(
    workspace: Path,
    agent_home: Path,
) -> None:
    session = _MCPCallSession()
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object"},
        ),
        session,
    )
    run_gateway = build_agent_run_gateway(
        _gateway(workspace, agent_home, additional_tools=(remote_tool,))
    )

    result = await run_gateway.call(
        ModelToolCall(
            id="direct-calendar",
            name="mcp_alpha_calendar_events",
            arguments=json.dumps({"range": "today", "extra": [None]}),
        )
    )

    assert result.status == "success"
    assert session.calls == [("calendar_events", {"range": "today", "extra": [None]})]
    assert run_gateway.exposed_names == RUN_BASELINE_TOOL_NAMES


@pytest.mark.asyncio
async def test_agent_run_gateways_keep_search_exposure_isolated(
    workspace: Path,
    agent_home: Path,
) -> None:
    remote_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="calendar_events",
            model_name="mcp_alpha_calendar_events",
            description="Read calendar events.",
            parameters={"type": "object"},
        ),
        _MCPCallSession(),
    )
    gateway = _gateway(workspace, agent_home, additional_tools=(remote_tool,))
    foreground = build_agent_run_gateway(
        gateway,
        mcp_keywords={"mcp_alpha_calendar_events": ("calendar",)},
    )
    scheduled = build_agent_run_gateway(
        gateway,
        excluded_names=("schedule",),
        mcp_keywords={"mcp_alpha_calendar_events": ("calendar",)},
    )

    await asyncio.gather(
        foreground.call(
            ModelToolCall(
                id="search-web",
                name="tool_search",
                arguments=json.dumps({"query": "web"}),
            )
        ),
        scheduled.call(
            ModelToolCall(
                id="search-calendar",
                name="tool_search",
                arguments=json.dumps({"query": "calendar"}),
            )
        ),
    )

    assert "web_search" in foreground.exposed_names
    assert "mcp_alpha_calendar_events" not in foreground.exposed_names
    assert "mcp_alpha_calendar_events" in scheduled.exposed_names
    assert "web_search" not in scheduled.exposed_names
    assert "schedule" not in {tool.name for tool in scheduled.catalog}


def test_agent_run_gateway_schedule_lane_excludes_schedule_from_lookup_and_search(
    workspace: Path,
    agent_home: Path,
) -> None:
    gateway = _gateway(workspace, agent_home)
    run_gateway = build_agent_run_gateway(gateway, excluded_names=("schedule",))

    assert "schedule" not in run_gateway.exposed_names
    assert "schedule" not in _names(run_gateway)
    search = asyncio.run(
        run_gateway.call(
            ModelToolCall(
                id="search-schedule",
                name="tool_search",
                arguments=json.dumps({"query": "schedule"}),
            )
        )
    )
    direct = asyncio.run(
        run_gateway.call(
            ModelToolCall(
                id="direct-schedule",
                name="schedule",
                arguments=json.dumps({"action": "list"}),
            )
        )
    )

    assert json.loads(search.content) == []
    assert direct.status == "error"
    assert direct.content == "The requested tool is not available."


@pytest.mark.asyncio
async def test_run_gateway_separates_catalog_and_exposure(
    workspace: Path,
    agent_home: Path,
) -> None:
    class RunTool(BaseTool):
        name = "run_tool"
        description = "A tool added only to this run view."
        value: str

        async def execute(self, *, value: str) -> str:
            return f"run:{value}"

    gateway = _gateway(workspace, agent_home)
    first = gateway.for_run(
        excluded_names=("schedule",),
        exposed_names=("read_file",),
        run_tools=(RunTool(),),
    )
    second = gateway.for_run(
        excluded_names=("schedule",),
        exposed_names=("exec",),
    )

    assert _names(first) == ["read_file"]
    assert first.exposed_names == ("read_file",)
    assert "schedule" not in first.exposed_names
    assert _names(second) == ["exec"]
    assert _names(gateway)[-1] == "schedule"

    direct = await first.call(
        ModelToolCall(
            id="call_run_tool",
            name="run_tool",
            arguments='{"value":"payload"}',
        )
    )
    unavailable = await first.call(
        ModelToolCall(id="call_schedule", name="schedule", arguments='{"action":"list"}')
    )

    assert (direct.status, direct.content) == ("success", "run:payload")
    assert (unavailable.status, unavailable.content) == (
        "error",
        "The requested tool is not available.",
    )
    assert first.exposed_names == ("read_file",)

    first.expose(("run_tool",))

    assert _names(first) == ["read_file", "run_tool"]
    assert list(first.exposed_names) == ["read_file", "run_tool"]
    assert _names(second) == ["exec"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_order", "alpha_order", "zulu_order"),
    [
        (("zulu", "alpha"), ("b-tool", "a-tool"), ("z-tool", "a-tool")),
        (("alpha", "zulu"), ("a-tool", "b-tool"), ("a-tool", "z-tool")),
        (("zulu", "alpha"), ("a-tool", "b-tool"), ("z-tool", "a-tool")),
    ],
)
async def test_catalog_orders_builtins_servers_and_remote_tools(
    workspace: Path,
    agent_home: Path,
    server_order: tuple[str, str],
    alpha_order: tuple[str, str],
    zulu_order: tuple[str, str],
) -> None:
    class Session:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            return CallToolResult(content=[])

    class Connection:
        unavailable = False

        def __init__(self, configuration: MCPServerConfiguration, workspace: Path | None) -> None:
            self.name = configuration.mcp_name

        async def connect(self) -> tuple[MCPTool, ...]:
            return tuple(
                MCPTool(
                    MCPToolSpec(
                        server_name=self.name,
                        remote_name=name,
                        model_name=allocate_mcp_tool_name(self.name, name) or "",
                        description=name,
                        parameters={"type": "object"},
                    ),
                    Session(),
                )
                for name in {"alpha": alpha_order, "zulu": zulu_order}[self.name]
            )

        async def close(self) -> None:
            pass

    manager = MCPRuntimeManager(workspace, transport="stdio", connection_factory=Connection)
    report = await manager.start(
        {
            name: MCPServerConfiguration(
                mcp_name=name, enabled=True, transport="stdio", command="mcp-server"
            )
            for name in server_order
        }
    )
    gateway = _gateway(workspace, agent_home, additional_tools=report.snapshot)

    assert isinstance(report.snapshot, tuple)
    assert _names(gateway) == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "exec",
        "web_search",
        "web_fetch",
        "schedule",
        "mcp_alpha_a-tool",
        "mcp_alpha_b-tool",
        "mcp_zulu_a-tool",
        "mcp_zulu_z-tool",
    ]
    await manager.close()


@pytest.mark.asyncio
async def test_fixed_gateway_reads_skill_root_with_model_file_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    skill_file = agent_home / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(workspace, agent_home, skill_root=agent_home / "skills")
    result = await gateway.call(
        ModelToolCall(
            id="call_skill_read",
            name="read_file",
            arguments=json.dumps({"path": str(skill_file)}),
        ),
        confirmation=approve,
    )

    assert (result.status, result.content) == ("success", "---\nname: review\n---\nbody\n")
    assert len(requests) == 1
    assert len(gateway.schemas) == 10


def test_run_catalog_exclusion_keeps_other_tools_available(
    workspace: Path,
    agent_home: Path,
) -> None:
    foreground = _gateway(workspace, agent_home)
    scheduled = foreground.for_run(
        exposed_names=(),
        excluded_names=("schedule",),
    )
    scheduled_catalog_names = {tool.name for tool in scheduled.catalog}

    assert "schedule" in _names(foreground)
    assert scheduled.schemas == []
    assert len(scheduled.catalog) == len(foreground.catalog) - 1
    assert "schedule" not in scheduled_catalog_names
    assert "web_search" in scheduled_catalog_names
    assert "web_fetch" in scheduled_catalog_names
    assert "exec" in scheduled_catalog_names


@pytest.mark.asyncio
async def test_fixed_gateway_calls_core_tool_and_returns_unified_result(
    workspace: Path,
    agent_home: Path,
) -> None:
    (workspace / "note.txt").write_text("hello\r\nworld\r\n", encoding="utf-8", newline="")
    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_read",
            name="read_file",
            arguments='{"path":"note.txt","offset":1,"limit":1}',
        )
    )

    assert result.to_dict() == {
        "tool_call_id": "call_read",
        "name": "read_file",
        "status": "success",
        "content": "hello\r\n",
        "artifact": None,
    }


@pytest.mark.asyncio
async def test_generation_tools_share_gateway_schema_and_trusted_execution_boundary(
    workspace: Path,
    agent_home: Path,
) -> None:
    observed: list[dict[str, Any]] = []

    class Session:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            assert name == "echo"
            observed.append(arguments)
            return CallToolResult(content=[])

    remote = MCPTool(
        MCPToolSpec(
            server_name="remote",
            remote_name="echo",
            model_name="mcp_remote_echo",
            description="Echo remote arguments.",
            parameters={"type": "object", "properties": {}},
        ),
        Session(),
    )
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    gateway = ToolGateway(
        workspace=workspace,
        schedule_service=ScheduleService(
            workspace_state=state,
            clock=_Clock(),
            execute_user_job=_noop,
            execute_dream=_noop,
        ),
        additional_tools=(remote,),
    )

    assert _names(gateway)[-1] == "mcp_remote_echo"
    result = await gateway.call(
        ModelToolCall(
            id="call_remote",
            name="mcp_remote_echo",
            arguments=json.dumps({"nested": {"value": 1}, "extra": [None]}),
        ),
        confirmation=lambda request: pytest.fail(f"unexpected confirmation: {request}"),
    )

    assert result.status == "success"
    assert result.content == "(no output)"
    assert observed == [{"nested": {"value": 1}, "extra": [None]}]


@pytest.mark.asyncio
async def test_exec_confirmation_preserves_the_exact_normalized_operation(
    workspace: Path,
    agent_home: Path,
) -> None:
    command = f'printf "{"x" * 300}" && rm -rf "build output"'
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_long_exec",
            name="exec",
            arguments=json.dumps({"command": command, "cwd": ".", "timeout": 45}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {
        "command": command,
        "cwd": str(workspace.resolve()),
        "timeout": 45,
    }


@pytest.mark.asyncio
async def test_unexpected_core_tool_failure_is_redacted(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workspace / "note.txt").write_text("hello", encoding="utf-8")

    target = workspace / "note.txt"
    original_read_bytes = Path.read_bytes

    def fail_target_read(path: Path) -> bytes:
        if path == target:
            raise RuntimeError("secret implementation detail")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_target_read)
    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_failure",
            name="read_file",
            arguments='{"path":"note.txt"}',
        )
    )

    assert result.status == "error"
    assert result.content == "read_file could not complete the request."
    assert "secret implementation detail" not in result.content


def test_gateway_rebuilds_schemas_for_each_access() -> None:
    class CountingTool(BaseTool):
        name = "counting"
        description = "Count schema projections."
        value: str

        async def execute(self, *, value: str) -> str:
            return value

    tool = CountingTool()
    gateway = ToolGateway._for_memory((tool,))
    project_schema = BaseTool.to_schema

    with patch.object(
        BaseTool, "to_schema", autospec=True, side_effect=project_schema
    ) as projection:
        assert projection.call_count == 0
        assert gateway.schemas[0]["function"]["name"] == "counting"
        assert gateway.schemas[0]["function"]["name"] == "counting"
        assert gateway.schemas[0]["function"]["name"] == "counting"
        assert projection.call_count == 3


@pytest.mark.asyncio
async def test_gateway_executes_every_tool_through_execute_prepared() -> None:
    class PreparedTool(BaseTool):
        name = "prepared"
        description = "Execute through the prepared seam."
        required = ("value",)
        value: str

        async def execute(self, *, value: str) -> str:
            del value
            raise AssertionError("Gateway must not call execute directly")

        async def execute_prepared(self, arguments: dict[str, Any]) -> str:
            assert arguments == {"value": "payload"}
            return "prepared result"

    result = await ToolGateway._for_memory((PreparedTool(),)).call(
        ModelToolCall(id="call-prepared", name="prepared", arguments='{"value":"payload"}')
    )

    assert result.status == "success"
    assert result.content == "prepared result"


@pytest.mark.asyncio
async def test_gateway_default_prepared_execution_expands_builtin_arguments_three_times() -> None:
    class BuiltinTool(BaseTool):
        name = "builtin"
        description = "Use the default prepared execution."
        required = ("value",)
        value: str

        async def execute(self, *, value: str) -> str:
            return f"builtin:{value}"

    gateway = ToolGateway._for_memory((BuiltinTool(),))

    first = await gateway.call(
        ModelToolCall(id="call-builtin-1", name="builtin", arguments='{"value":"one"}')
    )
    second = await gateway.call(
        ModelToolCall(id="call-builtin-2", name="builtin", arguments='{"value":"two"}')
    )
    third = await gateway.call(
        ModelToolCall(id="call-builtin-3", name="builtin", arguments='{"value":"three"}')
    )

    assert (first.status, first.content) == ("success", "builtin:one")
    assert (second.status, second.content) == ("success", "builtin:two")
    assert (third.status, third.content) == ("success", "builtin:three")


@pytest.mark.asyncio
async def test_gateway_forwards_complete_object_arguments_three_times() -> None:
    observed: list[dict[str, Any]] = []

    class ObjectTool(BaseTool):
        name = "object"
        description = "Forward a complete object."
        parameters: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }

        async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
            return dict(arguments)

        async def execute_prepared(self, arguments: dict[str, Any]) -> str:
            observed.append(arguments)
            return f"object:{arguments['value']}"

    gateway = ToolGateway._for_memory((ObjectTool(),))

    first = await gateway.call(
        ModelToolCall(
            id="call-object-1",
            name="object",
            arguments='{"value":"one","extra":{"nested":1}}',
        )
    )
    second = await gateway.call(
        ModelToolCall(
            id="call-object-2",
            name="object",
            arguments='{"value":"two","extra":{"nested":2}}',
        )
    )
    third = await gateway.call(
        ModelToolCall(
            id="call-object-3",
            name="object",
            arguments='{"value":"three","extra":{"nested":3}}',
        )
    )

    assert (first.status, first.content) == ("success", "object:one")
    assert (second.status, second.content) == ("success", "object:two")
    assert (third.status, third.content) == ("success", "object:three")
    assert observed == [
        {"value": "one", "extra": {"nested": 1}},
        {"value": "two", "extra": {"nested": 2}},
        {"value": "three", "extra": {"nested": 3}},
    ]
