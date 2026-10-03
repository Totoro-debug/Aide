from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import mcp.types as types
import pytest
from loguru import logger
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ImageContent, TextContent, Tool
from pydantic import ValidationError

import omni.agent.tools.mcp as mcp_adapter
from omni.agent.tools.base import ToolError
from omni.agent.tools.mcp import (
    MCPServerConnection,
    MCPTool,
    MCPToolSchemaError,
    MCPToolSpec,
    discover_mcp_tool_specs,
    mcp_tool_spec_from_remote,
    normalize_nullable,
)
from omni.agent.tools.permission import MCPToolIdentity, PermissionContext
from omni.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
)
from omni.config.config import MCPServerConfiguration
from tests.fixtures.gateway import SingleToolGateway
from tests.fixtures.mcp_wire import (
    http_wire_server,
    stdio_requests,
    stdio_wire_configuration,
    wire_result,
    wire_tool,
)


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return CallToolResult(content=[TextContent(text="ok")])


@pytest.mark.asyncio
async def test_mcp_tool_forwards_the_complete_argument_object_through_gateway() -> None:
    session = _Session()
    tool = MCPTool(
        MCPToolSpec(
            server_name="local",
            remote_name="echo",
            model_name="mcp_local_echo",
            description="Echo values.",
            parameters={"type": "object", "properties": {}},
        ),
        session,
    )

    arguments = {
        "nested": {"value": "text"},
        "items": [1, None, {"extra": True}],
        "undeclared": "preserved",
    }
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-1", name="mcp_local_echo", arguments=json.dumps(arguments))
    )

    assert result.status == "success"
    assert result.content == "ok"
    assert session.calls == [("echo", arguments)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"nested": {"key": "value"}},
        {"items": [1, 2, {"nested": True}]},
        {"nullable": None},
        {"additional": {"kept": "as-is"}},
        {"undeclared": "forwarded"},
        {"self": "ok"},
        {"arguments": "ok"},
        {"not-a-python-name": "ok"},
        {"\u53c2\u6570": "ok"},
    ],
)
async def test_mcp_tool_forwards_each_complete_argument_shape(
    arguments: dict[str, Any],
) -> None:
    session = _Session()
    tool = MCPTool(
        MCPToolSpec(
            server_name="local",
            remote_name="echo",
            model_name="mcp_local_echo",
            description="Echo values.",
            parameters={"type": "object", "properties": {}},
        ),
        session,
    )

    result = await SingleToolGateway(
        (tool,), confirmation=lambda request: pytest.fail("unexpected confirmation")
    ).call(ModelToolCall(id="call-forward", name=tool.name, arguments=json.dumps(arguments)))

    assert result.status == "success"
    assert session.calls == [("echo", arguments)]


@pytest.mark.asyncio
async def test_mcp_preparation_and_remote_mutation_preserve_original_arguments() -> None:
    received: list[dict[str, Any]] = []

    class MutatingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
            received.append(deepcopy(arguments))
            arguments["nested"]["value"] = "changed"
            return CallToolResult(content=[TextContent(text="ok")])

    tool = _tool_for_session(MutatingSession())
    arguments = {"self": "ok", "nested": {"value": "before"}, "extra": "42"}
    original = deepcopy(arguments)
    facts = await tool.prepare(arguments)
    prepared = facts.normalized_arguments
    assert prepared == original
    assert await tool.execute_prepared(prepared) == "ok"
    assert received == [original]
    assert arguments == prepared == original


def test_mcp_tool_schema_uses_remote_name_when_description_is_missing() -> None:
    spec = mcp_tool_spec_from_remote(
        Tool(name="search", description=None, input_schema={"type": "object"}),
        server_name="remote",
        model_name="mcp_remote_search",
    )

    assert spec.description == "search"
    assert spec.parameters == {"type": "object"}


@pytest.mark.parametrize(
    "schema",
    [
        ["not", "a", "dictionary"],
        None,
        "object",
    ],
)
def test_mcp_tool_loading_rejects_non_dictionary_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="dictionary"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": object()}},
        {"type": "object", "properties": {"value": {"default": {1, 2}}}},
    ],
)
def test_mcp_tool_loading_rejects_non_json_serializable_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="JSON serializable"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": ["object", "null"]},
        {"properties": {}},
    ],
)
def test_mcp_tool_loading_rejects_non_object_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="object root"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    ("schema", "description", "message"),
    [
        (None, 42, "MCP Tool inputSchema must be a dictionary"),
        (
            {"type": "string", "default": object()},
            42,
            "MCP Tool inputSchema must be JSON serializable",
        ),
        ({"type": "string"}, 42, "MCP Tool inputSchema must use an object root"),
        (
            {"type": "object", "nullable": True},
            None,
            "MCP Tool inputSchema must use an object root",
        ),
        ({"type": "object", "nullable": True}, 42, "MCP Tool inputSchema must use an object root"),
        ({"type": "object"}, 42, "MCP Tool description must be a string"),
    ],
)
def test_mcp_tool_loading_preserves_validation_error_order(
    schema: object,
    description: object,
    message: str,
) -> None:
    with pytest.raises(MCPToolSchemaError) as raised:
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema, "description": description},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )

    assert type(raised.value) is MCPToolSchemaError
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string", "nullable": True}, {"type": ["string", "null"]}),
        ({"type": "null", "nullable": True}, {"type": ["null"]}),
        (
            {"type": ["string", "integer"], "nullable": True},
            {"type": ["string", "integer", "null"]},
        ),
        (
            {"type": ["string", "null"], "nullable": True},
            {"type": ["string", "null"]},
        ),
        (
            {"nullable": True, "description": "any value"},
            {"anyOf": [{"description": "any value"}, {"type": "null"}]},
        ),
        (
            {"type": "object", "properties": {"value": {"type": "boolean", "nullable": True}}},
            {"type": "object", "properties": {"value": {"type": ["boolean", "null"]}}},
        ),
        (
            {"type": "array", "items": [{"type": "string", "nullable": True}]},
            {"type": "array", "items": [{"type": ["string", "null"]}]},
        ),
        (
            {"type": "object", "properties": {"items": [{"nullable": True}]}},
            {"type": "object", "properties": {"items": [{"nullable": True}]}},
        ),
        (
            {"anyOf": [{"type": "string"}, {"type": "integer"}], "nullable": True},
            {
                "anyOf": [
                    {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    {"type": "null"},
                ]
            },
        ),
        (
            {"oneOf": [{"$ref": "#/definitions/value"}], "nullable": True},
            {"anyOf": [{"oneOf": [{"$ref": "#/definitions/value"}]}, {"type": "null"}]},
        ),
        (
            {"$ref": "#/definitions/value", "nullable": True},
            {"anyOf": [{"$ref": "#/definitions/value"}, {"type": "null"}]},
        ),
        ({"type": "string", "nullable": False}, {"type": "string"}),
    ],
)
def test_normalize_nullable_golden_cases(schema: dict[str, Any], expected: dict[str, Any]) -> None:
    original = deepcopy(schema)
    assert normalize_nullable(schema) == expected
    assert schema == original
    assert normalize_nullable(expected) == expected


@pytest.mark.parametrize(
    "keyword", ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]
)
def test_nullable_preserves_names_in_schema_maps(keyword: str) -> None:
    schema = {
        keyword: {"nullable": {"type": "string", "nullable": True}, "boolean": False},
        "required": ["nullable"],
        "$ref": f"#/{keyword}/nullable",
    }
    original = deepcopy(schema)
    expected = deepcopy(schema)
    expected[keyword]["nullable"] = {"type": ["string", "null"]}  # type: ignore[index]
    assert normalize_nullable(schema) == expected
    assert schema == original
    assert normalize_nullable(expected) == expected


@pytest.mark.parametrize(
    "keyword",
    ["default", "const", "enum", "examples", "required", "dependentRequired", "x-extension"],
)
def test_nullable_preserves_json_data(keyword: str) -> None:
    data = {"nullable": {"nullable": True, "type": "string"}, "nested": [{"nullable": True}]}
    schema = {"type": "object", keyword: [data] if keyword in {"enum", "examples"} else data}
    original = deepcopy(schema)
    normalized = normalize_nullable(schema)
    assert normalized == original
    assert normalize_nullable(normalized) == original
    normalized[keyword].clear()
    assert schema == original


@pytest.mark.parametrize(
    "keyword",
    [
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "additionalItems",
        "unevaluatedItems",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
        "items",
    ],
)
@pytest.mark.parametrize("child", [{"type": "string", "nullable": True}, False, True, [1]])
def test_nullable_visits_only_single_schema_values(keyword: str, child: object) -> None:
    schema = {keyword: child}
    original = deepcopy(schema)
    expected = {keyword: {"type": ["string", "null"]} if isinstance(child, dict) else child}
    assert normalize_nullable(schema) == expected
    assert normalize_nullable(expected) == expected
    assert schema == original


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf", "prefixItems", "items"])
def test_nullable_visits_schema_arrays(keyword: str) -> None:
    schema = {keyword: [{"nullable": True}, False, [{"nullable": True}]]}
    original = deepcopy(schema)
    expected = {keyword: [{"anyOf": [{}, {"type": "null"}]}, False, [{"nullable": True}]]}
    assert normalize_nullable(schema) == expected
    assert normalize_nullable(expected) == expected
    assert schema == original


def test_nullable_handles_legacy_dependencies_without_changing_property_arrays() -> None:
    schema = {
        "dependencies": {
            "nullable": {"type": "object", "nullable": True},
            "required": ["nullable"],
            "boolean": False,
        }
    }
    original = deepcopy(schema)
    expected = {
        "dependencies": {
            "nullable": {"type": ["object", "null"]},
            "required": ["nullable"],
            "boolean": False,
        }
    }
    assert normalize_nullable(schema) == expected
    assert normalize_nullable(expected) == expected
    assert schema == original


def test_nullable_parameter_name_and_default_reach_gateway_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "nullable": {
                "type": "object",
                "nullable": True,
                "default": {"nullable": {"type": "string", "nullable": True}},
            }
        },
        "required": ["nullable"],
    }
    original = deepcopy(schema)
    spec = mcp_tool_spec_from_remote(
        {"name": "echo", "inputSchema": schema}, server_name="remote", model_name="mcp_remote_echo"
    )
    projected = SingleToolGateway((MCPTool(spec, _Session()),)).schemas[0]["function"]["parameters"]
    expected = deepcopy(schema)
    expected["properties"]["nullable"].pop("nullable")  # type: ignore[index]
    expected["properties"]["nullable"]["type"] = ["object", "null"]  # type: ignore[index]
    assert projected == expected
    assert schema == original


class _PageSession:
    def __init__(self, pages: dict[str | None, tuple[list[object], str | None]]) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
        cursor = None if params is None else params.cursor
        self.cursors.append(cursor)
        tools, next_cursor = self.pages[cursor]
        return {"tools": tools, "nextCursor": next_cursor}


def _remote(name: str) -> Tool:
    return Tool(name=name, input_schema={"type": "object"}, description=name)


@pytest.mark.asyncio
async def test_discovery_collects_three_pages_and_sorts_remote_tools() -> None:
    session = _PageSession(
        {
            None: ([_remote("third")], "page-2"),
            "page-2": ([_remote("first")], "page-3"),
            "page-3": ([_remote("second")], None),
        }
    )

    specs = await discover_mcp_tool_specs(
        session,
        server_name="remote",
        model_name_for=lambda name: f"model_{name}",
    )

    assert session.cursors == [None, "page-2", "page-3"]
    assert [(spec.remote_name, spec.model_name) for spec in specs] == [
        ("first", "model_first"),
        ("second", "model_second"),
        ("third", "model_third"),
    ]


@pytest.mark.asyncio
async def test_discovery_handles_empty_list_without_tools() -> None:
    specs = await discover_mcp_tool_specs(
        _PageSession({None: ([], None)}),
        server_name="remote",
    )

    assert specs == ()


@pytest.mark.asyncio
async def test_discovery_stops_on_repeated_cursor_after_processing_page() -> None:
    session = _PageSession(
        {
            None: ([_remote("first")], "same"),
            "same": ([_remote("second")], "same"),
        }
    )

    specs = await discover_mcp_tool_specs(session, server_name="remote")

    assert [spec.remote_name for spec in specs] == ["first", "second"]
    assert session.cursors == [None, "same"]


@pytest.mark.asyncio
async def test_discovery_skips_invalid_tools_without_dropping_valid_tools() -> None:
    skipped: list[None] = []
    session = _PageSession(
        {
            None: (
                [
                    _remote("valid-first"),
                    {"name": "invalid", "inputSchema": {"type": "string"}},
                ],
                None,
            )
        }
    )

    specs = await discover_mcp_tool_specs(
        session,
        server_name="remote",
        on_tool_skipped=lambda: skipped.append(None),
    )

    assert [spec.remote_name for spec in specs] == ["valid-first"]
    assert len(skipped) == 1


def _tool_for_session(
    session: Any,
    *,
    call_timeout: float = 60,
    on_closed: Any = None,
) -> MCPTool:
    return MCPTool(
        MCPToolSpec(
            server_name="remote",
            remote_name="remote_echo",
            model_name="mcp_remote_echo",
            description="Remote echo.",
            parameters={"type": "object"},
        ),
        session,
        call_timeout=call_timeout,
        on_closed=on_closed,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "expects_confirmation"),
    [
        ("read-only", True),
        ("workspace-write", True),
        ("full-access", False),
    ],
)
async def test_foreground_mcp_permission_level_controls_each_call(
    level: str,
    expects_confirmation: bool,
) -> None:
    session = _Session()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level=level, origin="foreground"),  # type: ignore[arg-type]
    )
    arguments = {"nested": {"value": "normalized"}, "extra": [None]}

    result = await gateway.call(
        ModelToolCall(id="mcp-level", name=tool.name, arguments=json.dumps(arguments)),
        confirmation=approve,
    )

    assert result.status == "success"
    assert session.calls == [(tool.remote_name, arguments)]
    assert len(requests) == int(expects_confirmation)
    if expects_confirmation:
        assert requests[0].details == arguments
        assert requests[0].mcp_identity == MCPToolIdentity(
            server_name="remote",
            remote_name="remote_echo",
            model_name="mcp_remote_echo",
        )
        assert result.confirmation is not None
        assert result.confirmation.decision == "approved"
    else:
        assert result.confirmation is None


@pytest.mark.asyncio
async def test_low_permission_mcp_decline_never_reaches_transport() -> None:
    session = _Session()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id="mcp-decline", name=tool.name, arguments='{"value":"blocked"}'),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation_mode", ["missing", "failure", "invalid"])
async def test_low_permission_mcp_confirmation_failure_never_reaches_transport(
    confirmation_mode: str,
) -> None:
    session = _Session()
    tool = _tool_for_session(session)

    async def fail_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        raise RuntimeError("presenter failed")

    async def invalid_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return cast(ConfirmationDecision, "unexpected")

    confirmation: ConfirmationRequester | None
    if confirmation_mode == "missing":
        confirmation = None
    elif confirmation_mode == "failure":
        confirmation = fail_confirmation
    else:
        confirmation = invalid_confirmation

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id=f"mcp-{confirmation_mode}", name=tool.name, arguments="{}"),
        confirmation=confirmation,
    )

    assert result.status == "refused"
    assert result.confirmation is not None
    assert result.confirmation.decision is None
    assert session.calls == []


@pytest.mark.asyncio
async def test_low_permission_mcp_repeated_calls_request_independent_approvals() -> None:
    session = _Session()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="workspace-write", origin="foreground"),
    )
    first_arguments = {"value": "same"}
    second_arguments = {"value": "same"}
    first = await gateway.call(
        ModelToolCall(id="mcp-repeat-1", name=tool.name, arguments=json.dumps(first_arguments)),
        confirmation=approve,
    )
    second = await gateway.call(
        ModelToolCall(id="mcp-repeat-2", name=tool.name, arguments=json.dumps(second_arguments)),
        confirmation=approve,
    )

    assert first.status == second.status == "success"
    assert len(requests) == 2
    assert requests[0].confirmation_id != requests[1].confirmation_id
    assert [request.details for request in requests] == [first_arguments, second_arguments]
    assert session.calls == [
        (tool.remote_name, first_arguments),
        (tool.remote_name, second_arguments),
    ]


@pytest.mark.asyncio
async def test_replacement_mcp_identity_does_not_inherit_an_earlier_approval() -> None:
    first_session = _Session()
    replacement_session = _Session()
    first_tool = MCPTool(
        MCPToolSpec(
            server_name="alpha",
            remote_name="echo",
            model_name="mcp_echo",
            description="First generation echo.",
            parameters={"type": "object"},
        ),
        first_session,
    )
    replacement_tool = MCPTool(
        MCPToolSpec(
            server_name="beta",
            remote_name="echo",
            model_name="mcp_echo",
            description="Replacement generation echo.",
            parameters={"type": "object"},
        ),
        replacement_session,
    )
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    for tool in (first_tool, replacement_tool):
        gateway = ToolGateway._for_memory(
            (tool,),
            permission_context=PermissionContext(level="read-only", origin="foreground"),
        )
        result = await gateway.call(
            ModelToolCall(id=f"call-{tool.server_name}", name=tool.name, arguments="{}"),
            confirmation=approve,
        )
        assert result.status == "success"

    assert [request.mcp_identity for request in requests] == [
        MCPToolIdentity(server_name="alpha", remote_name="echo", model_name="mcp_echo"),
        MCPToolIdentity(server_name="beta", remote_name="echo", model_name="mcp_echo"),
    ]
    assert first_session.calls == [("echo", {})]
    assert replacement_session.calls == [("echo", {})]


@pytest.mark.asyncio
async def test_concurrent_foreground_mcp_calls_keep_confirmation_state_isolated() -> None:
    session = _Session()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        await asyncio.sleep(0)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    results = await asyncio.gather(
        gateway.call(
            ModelToolCall(
                id="mcp-concurrent-1",
                name=tool.name,
                arguments='{"value":"one"}',
            ),
            confirmation=approve,
        ),
        gateway.call(
            ModelToolCall(
                id="mcp-concurrent-2",
                name=tool.name,
                arguments='{"value":"two"}',
            ),
            confirmation=approve,
        ),
    )

    assert [result.status for result in results] == ["success", "success"]
    assert len(requests) == 2
    assert {request.tool_call_id for request in requests} == {
        "mcp-concurrent-1",
        "mcp-concurrent-2",
    }
    assert {tuple(sorted(request.details.items())) for request in requests} == {
        (("value", "one"),),
        (("value", "two"),),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["read-only", "workspace-write", "full-access"])
async def test_mcp_parse_and_lookup_errors_precede_permission_at_every_level(level: str) -> None:
    session = _Session()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        raise AssertionError("hard MCP errors must not request confirmation")

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level=level, origin="foreground"),  # type: ignore[arg-type]
    )
    invalid_json = await gateway.call(
        ModelToolCall(id="mcp-invalid-json", name=tool.name, arguments="not-json"),
        confirmation=unexpected_confirmation,
    )
    unknown_tool = await gateway.call(
        ModelToolCall(id="mcp-unknown", name="mcp_remote_missing", arguments="{}"),
        confirmation=unexpected_confirmation,
    )

    assert invalid_json.status == unknown_tool.status == "error"
    assert requests == []
    assert session.calls == []


@pytest.mark.asyncio
async def test_mcp_schedule_context_calls_directly_without_confirmation() -> None:
    session = _Session()
    tool = _tool_for_session(session)

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        raise AssertionError(f"Schedule MCP calls must execute without confirmation: {request}")

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="schedule"),
    )
    result = await gateway.call(
        ModelToolCall(id="mcp-schedule", name=tool.name, arguments='{"value":"direct"}'),
        confirmation=unexpected_confirmation,
    )

    assert result.status == "success"
    assert session.calls == [(tool.remote_name, {"value": "direct"})]


@pytest.mark.asyncio
async def test_approved_mcp_transport_failure_preserves_gateway_error() -> None:
    class FailingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise RuntimeError("private transport failure")

    tool = _tool_for_session(FailingSession())
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id="mcp-transport-error", name=tool.name, arguments="{}"),
        confirmation=approve,
    )

    assert result.status == "error"
    assert result.content == "mcp_remote_echo could not complete the request."
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_approved_mcp_cancellation_propagates_unchanged() -> None:
    class CancelledSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise asyncio.CancelledError

    tool = _tool_for_session(CancelledSession())

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="workspace-write", origin="foreground"),
    )
    with pytest.raises(asyncio.CancelledError):
        await gateway.call(
            ModelToolCall(id="mcp-cancelled", name=tool.name, arguments="{}"),
            confirmation=approve,
        )


class _ResultSession:
    def __init__(self, result: object) -> None:
        self.result = result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        return self.result


class _StructuredContentMustNotBeRead:
    is_error = False

    def __init__(self, content: list[object]) -> None:
        self.content = content

    @property
    def structured_content(self) -> object:
        raise AssertionError("structured content is outside the text boundary")


@pytest.mark.asyncio
async def test_mcp_result_joins_text_content_in_order() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(content=[TextContent(text="first"), TextContent(text="second")])
        )
    )

    assert await tool.execute_prepared({}) == "first\nsecond"


@pytest.mark.asyncio
async def test_mcp_result_converts_non_text_blocks_and_ignores_structured_content() -> None:
    non_text = ImageContent(data="ZmFrZQ==", mime_type="image/png")
    result = _StructuredContentMustNotBeRead([non_text, TextContent(text="text")])

    class _MixedResultSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            return result

    tool = _tool_for_session(_MixedResultSession())

    assert await tool.execute_prepared({}) == f"{non_text}\ntext"


@pytest.mark.asyncio
async def test_mcp_empty_result_has_explicit_output_marker() -> None:
    tool = _tool_for_session(_ResultSession(CallToolResult(content=[])))

    assert await tool.execute_prepared({}) == "(no output)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_fields", "status"),
    [
        ({"is_error": True}, "error"),
        ({"isError": True}, "error"),
        ({"is_error": False}, "success"),
        ({"isError": False}, "success"),
        ({"isError": 1}, "success"),
        ({"isError": "true"}, "success"),
        ({"is_error": False, "isError": True}, "success"),
    ],
)
async def test_mcp_result_preserves_content_and_requires_an_explicit_error_flag(
    error_fields: dict[str, object],
    status: str,
) -> None:
    tool = _tool_for_session(
        _ResultSession({"content": [TextContent(text="server result")], **error_fields})
    )

    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-error-result", name=tool.name, arguments="{}")
    )

    assert (result.status, result.content) == (status, "server result")
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_mcp_error_result_with_multiple_blocks_preserves_order() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(
                content=[TextContent(text="first"), TextContent(text="second")],
                is_error=True,
            )
        )
    )

    with pytest.raises(ToolError, match="first\\nsecond"):
        await tool.execute_prepared({})


@pytest.mark.asyncio
async def test_mcp_call_timeout_is_a_tool_error_without_marking_connection_unavailable() -> None:
    class _ImmediateTimeoutSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise TimeoutError("transport timeout")

    tool = _tool_for_session(_ImmediateTimeoutSession(), call_timeout=0.01)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id="call-timeout", name=tool.name, arguments="{}"),
        confirmation=approve,
    )

    assert result.status == "error"
    assert "timed out" in result.content
    assert tool.unavailable is False
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_hanging_mcp_call_is_bounded_by_tool_timeout() -> None:
    never = asyncio.Event()

    class _HangingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            await never.wait()
            return CallToolResult(content=[])

    tool = _tool_for_session(_HangingSession(), call_timeout=0.01)

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="workspace-write", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id="call-hanging-timeout", name=tool.name, arguments="{}"),
        confirmation=approve,
    )

    assert result.status == "error"
    assert result.content == "MCP Tool call timed out after 0.01 seconds."
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_approved_mcp_server_error_result_remains_a_tool_error() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(content=[TextContent(text="server failure")], is_error=True)
        )
    )
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    result = await gateway.call(
        ModelToolCall(id="call-server-error", name=tool.name, arguments="{}"),
        confirmation=approve,
    )

    assert (result.status, result.content) == ("error", "server failure")
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_closed_mcp_session_returns_safe_error_and_marks_connection() -> None:
    class _ClosedSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise MCPError(types.CONNECTION_CLOSED, "private transport detail")

    marked: list[bool] = []
    tool = _tool_for_session(_ClosedSession(), on_closed=lambda: marked.append(True))
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-closed", name=tool.name, arguments="{}")
    )

    assert (result.status, result.content) == (
        "error",
        "MCP Server connection is unavailable.",
    )
    assert tool.unavailable is True
    assert marked == [True]


@pytest.mark.asyncio
async def test_unavailable_mcp_tool_is_a_hard_error_before_later_confirmation() -> None:
    class _ClosedSession:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            self.calls += 1
            raise MCPError(types.CONNECTION_CLOSED, "private transport detail")

    session = _ClosedSession()
    tool = _tool_for_session(session)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level="read-only", origin="foreground"),
    )
    first = await gateway.call(
        ModelToolCall(id="call-disconnect", name=tool.name, arguments="{}"),
        confirmation=approve,
    )
    second = await gateway.call(
        ModelToolCall(id="call-unavailable", name=tool.name, arguments="{}"),
        confirmation=lambda request: pytest.fail(f"unexpected confirmation: {request}"),
    )

    assert (first.status, first.content) == (
        "error",
        "MCP Server connection is unavailable.",
    )
    assert (second.status, second.content) == (
        "error",
        "MCP Server connection is unavailable.",
    )
    assert len(requests) == 1
    assert session.calls == 1


@pytest.mark.asyncio
async def test_closed_mcp_session_raises_tool_error_from_direct_execution() -> None:
    class _ClosedSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise MCPError(types.CONNECTION_CLOSED, "private transport detail")

    tool = _tool_for_session(_ClosedSession())

    with pytest.raises(ToolError, match="connection is unavailable"):
        await tool.execute_prepared({})

    assert tool.unavailable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("private detail"),
        ValueError("private value"),
        MCPError(-32603, "protocol failure"),
        type(
            "ClosedLookalikeError",
            (RuntimeError,),
            {"code": types.CONNECTION_CLOSED},
        )("not an SDK error"),
    ],
)
async def test_other_mcp_exceptions_use_gateway_generic_failure(error: Exception) -> None:
    class _FailingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise error

    tool = _tool_for_session(_FailingSession())
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-generic", name=tool.name, arguments="{}")
    )

    assert result.content == "mcp_remote_echo could not complete the request."
    assert "private" not in result.content
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_mcp_cancellation_propagates_from_gateway() -> None:
    class _CancelledSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise asyncio.CancelledError

    tool = _tool_for_session(_CancelledSession())

    with pytest.raises(asyncio.CancelledError):
        await ToolGateway._for_memory((tool,)).call(
            ModelToolCall(id="call-cancelled", name=tool.name, arguments="{}")
        )


@pytest.mark.asyncio
async def test_mcp_cancellation_propagates_from_direct_execution() -> None:
    class _CancelledSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise asyncio.CancelledError

    tool = _tool_for_session(_CancelledSession())

    with pytest.raises(asyncio.CancelledError):
        await tool.execute_prepared({})


def _server_configuration(
    *,
    transport: str = "stdio",
    command: str = "python",
    args: tuple[str, ...] = (),
    cwd: Any = None,
    url: str | None = None,
    headers: Any = None,
    connect_timeout: float = 1,
    call_timeout: float = 1,
) -> MCPServerConfiguration:
    return MCPServerConfiguration(
        mcp_name="remote",
        enabled=True,
        transport=transport,  # type: ignore[arg-type]
        command=command if transport == "stdio" else None,
        args=args,
        cwd=cwd,
        url=url,
        headers={} if headers is None else headers,
        connect_timeout=connect_timeout,  # type: ignore[arg-type]
        call_timeout=call_timeout,  # type: ignore[arg-type]
    )


class _AsyncTransport:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> tuple[str, str]:
        self.entered = True
        return ("read", "write")

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.closed = True


class _ConnectionSession:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False
        self.initialized = False
        self.tools: list[object] = [_remote("echo")]

    async def __aenter__(self) -> _ConnectionSession:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.closed = True

    async def initialize(self) -> object:
        self.initialized = True
        return object()

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
        del params
        return {"tools": self.tools, "nextCursor": None}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        return CallToolResult(content=[TextContent(text="connected")])


@pytest.mark.asyncio
async def test_server_connection_initializes_discovers_and_closes_one_session() -> None:
    transport = _AsyncTransport()
    session = _ConnectionSession()
    connection = MCPServerConnection(
        _server_configuration(),
        Path("."),
        transport_factory=lambda configuration, workspace: transport,
        session_factory=lambda read_stream, write_stream: session,
        model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
    )

    tools = await connection.connect()

    assert transport.entered is True
    assert session.entered is True
    assert session.initialized is True
    assert [tool.name for tool in tools] == ["mcp_remote_echo"]
    assert await tools[0].execute_prepared({}) == "connected"
    assert session.closed is False
    assert transport.closed is False

    await connection.close()

    assert session.closed is True
    assert transport.closed is True


@pytest.mark.asyncio
async def test_server_connection_counts_invalid_discovered_tools() -> None:
    transport = _AsyncTransport()
    session = _ConnectionSession()
    session.tools = [
        _remote("valid"),
        {"name": "invalid", "inputSchema": {"type": "string"}},
    ]
    connection = MCPServerConnection(
        _server_configuration(),
        Path("."),
        transport_factory=lambda configuration, workspace: transport,
        session_factory=lambda read_stream, write_stream: session,
        model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
    )

    tools = await connection.connect()

    assert [tool.name for tool in tools] == ["mcp_remote_valid"]
    assert connection.skipped_tool_count == 1

    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initialize", "list_tools"])
async def test_connection_timeout_covers_initialization_and_discovery_and_closes_resources(
    phase: str,
) -> None:
    transport = _AsyncTransport()
    hung = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingSession(_ConnectionSession):
        async def initialize(self) -> object:
            if phase == "initialize":
                hung.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await super().initialize()

        async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
            assert self.initialized is True
            hung.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("discovery unexpectedly resumed")

    session = HangingSession()
    connection = MCPServerConnection(
        _server_configuration(connect_timeout=0.01),
        Path("."),
        transport_factory=lambda configuration, workspace: transport,
        session_factory=lambda read_stream, write_stream: session,
    )

    with pytest.raises(TimeoutError):
        await connection.connect()

    assert hung.is_set()
    assert cancelled.is_set()
    assert session.entered is True
    assert session.initialized is (phase == "list_tools")
    assert session.closed is True
    assert transport.closed is True
    await connection.close()


@pytest.mark.asyncio
async def test_stdio_transport_resolves_cwd_and_inherits_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("MYCLAW_MCP_TEST_ENV", "inherited")

    @asynccontextmanager
    async def fake_stdio(parameters: object) -> Any:
        captured["parameters"] = parameters
        yield ("read", "write")

    monkeypatch.setattr(mcp_adapter, "stdio_client", fake_stdio)
    workspace = Path("C:/workspace")
    configuration = _server_configuration(
        args=("-c", "pass"),
        cwd=Path("nested"),
    )
    async with mcp_adapter.stdio_transport(configuration, workspace) as streams:
        assert streams == ("read", "write")

    parameters = captured["parameters"]
    assert isinstance(parameters, StdioServerParameters)
    assert parameters.command == "python"
    assert parameters.args == ["-c", "pass"]
    assert parameters.cwd == workspace / "nested"
    assert parameters.env == dict(os.environ)


@pytest.mark.asyncio
async def test_http_transport_passes_headers_and_closes_created_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _HTTPClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = _HTTPClient()

    def fake_create(*, headers: dict[str, str]) -> _HTTPClient:
        captured["headers"] = headers
        return client

    @asynccontextmanager
    async def fake_http(url: str, *, http_client: _HTTPClient) -> Any:
        captured["url"] = url
        captured["client"] = http_client
        yield ("read", "write")

    monkeypatch.setattr(mcp_adapter, "create_mcp_http_client", fake_create)
    monkeypatch.setattr(mcp_adapter, "streamable_http_client", fake_http)
    configuration = _server_configuration(
        transport="streamable-http",
        url="http://127.0.0.1:8765/mcp",
        headers={"Authorization": "Bearer local"},
    )
    async with mcp_adapter.streamable_http_transport(configuration, Path(".")) as streams:
        assert streams == ("read", "write")

    assert captured == {
        "headers": {"Authorization": "Bearer local"},
        "url": "http://127.0.0.1:8765/mcp",
        "client": client,
    }
    assert client.closed is True


@pytest.mark.asyncio
async def test_stdio_transport_connects_to_a_local_real_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYCLAW_MCP_TEST_ENV", "inherited")
    server_script = "\n".join(
        (
            "import os",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('local-stdio')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return f\"{value}:{os.environ['MYCLAW_MCP_TEST_ENV']}\"",
            "server.run()",
        )
    )
    connection = MCPServerConnection(
        _server_configuration(
            command=sys.executable,
            args=("-c", server_script),
            cwd=tmp_path,
            connect_timeout=10,
            call_timeout=5,
        ),
        tmp_path,
        model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
    )

    try:
        tools = await connection.connect()
        assert [tool.remote_name for tool in tools] == ["echo"]
        result = await ToolGateway._for_memory(tools).call(
            ModelToolCall(
                id="call-stdio",
                name="mcp_remote_echo",
                arguments=json.dumps({"value": "stdio result"}),
            )
        )
        assert (result.status, result.content) == ("success", "stdio result:inherited")
    finally:
        await connection.close()


async def _read_local_server_port(process: asyncio.subprocess.Process) -> int:
    if process.stdout is None:
        raise AssertionError("local MCP HTTP server stdout is unavailable")
    async with asyncio.timeout(10):
        while line := await process.stdout.readline():
            text = line.decode().strip()
            if text.startswith("MCP_TEST_READY:"):
                return int(text.partition(":")[2])
    raise AssertionError(f"local MCP HTTP server exited with {process.returncode}")


@pytest.mark.asyncio
async def test_streamable_http_transport_connects_to_a_local_real_mcp_server() -> None:
    server_script = "\n".join(
        (
            "import asyncio",
            "import socket",
            "import sys",
            "import uvicorn",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('local-http')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return value",
            "async def run():",
            "    listener = socket.socket()",
            "    listener.bind(('127.0.0.1', 0))",
            "    listener.listen()",
            "    port = listener.getsockname()[1]",
            "    config = uvicorn.Config(",
            "        server.streamable_http_app(),",
            "        log_level='critical',",
            "    )",
            "    http_server = uvicorn.Server(config)",
            "    task = asyncio.create_task(http_server.serve(sockets=[listener]))",
            "    while not http_server.started:",
            "        if task.done():",
            "            await task",
            "        await asyncio.sleep(0.01)",
            "    print(f'MCP_TEST_READY:{port}', flush=True)",
            "    await task",
            "asyncio.run(run())",
        )
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        server_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    connection: MCPServerConnection | None = None
    try:
        port = await _read_local_server_port(process)
        connection = MCPServerConnection(
            _server_configuration(
                transport="streamable-http",
                url=f"http://127.0.0.1:{port}/mcp",
                headers={"X-Test-Header": "local"},
                connect_timeout=10,
                call_timeout=5,
            ),
            Path.cwd(),
            model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
        )
        tools = await connection.connect()
        assert [tool.remote_name for tool in tools] == ["echo"]
        result = await ToolGateway._for_memory(tools).call(
            ModelToolCall(
                id="call-http",
                name="mcp_remote_echo",
                arguments=json.dumps({"value": "http result"}),
            )
        )
        assert (result.status, result.content) == ("success", "http result")
    finally:
        if connection is not None:
            await connection.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("case", ["missing", "mismatch", "invalid_schema", "valid", "error"])
async def test_real_sdk_projects_content_without_output_schema_validation(
    tmp_path: Path,
    transport: str,
    case: str,
) -> None:
    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    if case == "invalid_schema":
        output_schema["properties"]["value"]["type"] = "not-a-json-schema-type"
    fields: dict[str, Any] = {}
    if case != "missing":
        fields["structuredContent"] = {"value": 42 if case == "valid" else "wrong"}
    if case == "error":
        fields["isError"] = True
    response = wire_result(**fields)
    response["content"].append({"type": "text", "text": "second block"})
    scenario = {
        "pages": {"": {"tools": [wire_tool(outputSchema=output_schema)]}},
        "results": {"echo": response},
    }

    async def check(configuration: MCPServerConfiguration) -> None:
        connection = MCPServerConnection(configuration, tmp_path)
        try:
            tools = await connection.connect()
            assert len(tools) == 1
            gateway = SingleToolGateway(tools)
            for index in range(2):
                result = await gateway.call(
                    ModelToolCall(id=str(index), name="echo", arguments="{}")
                )
                assert (result.status, result.content) == (
                    "error" if case == "error" else "success",
                    "wire text\nsecond block",
                )
            assert not connection.unavailable
        finally:
            await connection.close()

    if transport == "stdio":
        await check(stdio_wire_configuration(tmp_path, scenario))
        requests = stdio_requests(tmp_path)
    else:
        async with http_wire_server(scenario) as (server, configuration):
            await check(configuration)
            requests = server.requests
    assert [r["method"] for r in requests].count("tools/list") == 1
    assert [r["method"] for r in requests].count("tools/call") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("pagination", [False, True, "all_invalid"])
async def test_real_sdk_isolates_invalid_wire_tools(
    tmp_path: Path,
    transport: str,
    pagination: bool | str,
) -> None:
    invalid = [
        {"name": "private-array", "inputSchema": []},
        {"name": "private-missing"},
        wire_tool("private-root", inputSchema={"type": "string"}),
    ]
    first = invalid if pagination == "all_invalid" else [wire_tool("zulu"), *invalid]
    pages: dict[str, Any] = {"": {"tools": first}}
    if pagination is True:
        pages[""]["nextCursor"] = "second"
        pages["second"] = {"tools": invalid, "nextCursor": "third", "_meta": {"kept": True}}
        pages["third"] = {"tools": [wire_tool("alpha")]}
    scenario = {"pages": pages}
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)), format="{message}")

    async def check(configuration: MCPServerConfiguration) -> None:
        connection = MCPServerConnection(configuration, tmp_path)
        try:
            tools = await connection.connect()
            expected = (
                []
                if pagination == "all_invalid"
                else (["alpha", "zulu"] if pagination is True else ["zulu"])
            )
            assert [tool.name for tool in tools] == expected
            assert connection.skipped_tool_count == (6 if pagination is True else 3)
            for tool in tools:
                assert await tool.execute_prepared({}) == "wire text"
            assert not connection.unavailable
        finally:
            await connection.close()

    try:
        if transport == "stdio":
            await check(stdio_wire_configuration(tmp_path, scenario))
            requests = stdio_requests(tmp_path)
        else:
            async with http_wire_server(scenario) as (server, configuration):
                await check(configuration)
                requests = server.requests
        cursors = [
            r.get("params", {}).get("cursor") for r in requests if r["method"] == "tools/list"
        ]
        assert cursors == ([None, "second", "third"] if pagination is True else [None])
        skipped = [record for record in records if "MCP Tool skipped" in record]
        assert len(skipped) == (6 if pagination is True else 3)
        assert all("mcp_name=remote phase=discovery type=MCPToolSchemaError" in r for r in skipped)
        assert "private-" not in "".join(records)
        assert "inputSchema" not in "".join(records)
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
async def test_real_sdk_preserves_concurrent_request_correlation_and_progress(
    tmp_path: Path,
) -> None:
    configuration = stdio_wire_configuration(tmp_path, {})
    closed: list[bool] = []
    progress_events: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_events.append((progress, total, message))

    async with mcp_adapter.stdio_transport(configuration, tmp_path) as streams:
        read, write = cast(tuple[object, object], streams)
        session = cast(
            ClientSession,
            mcp_adapter._new_client_session(
                read,
                write,
                on_closed=lambda: closed.append(True),
                on_tool_skipped=lambda: pytest.fail("unexpected skipped tool"),
            ),
        )
        async with session:
            await session.initialize()
            await session.list_tools()
            async with asyncio.timeout(10):
                results = await asyncio.gather(
                    *(
                        session.call_tool(
                            "echo", {"value": str(index)}, progress_callback=on_progress
                        )
                        for index in range(8)
                    )
                )
            assert [result.content[0] for result in results] == [
                TextContent(text=str(index)) for index in range(8)
            ]
            assert progress_events == [(1, 1, None)] * 8
            assert closed == []
    assert closed == [True]
    calls = [r for r in stdio_requests(tmp_path) if r["method"] == "tools/call"]
    assert len({r["id"] for r in calls}) == 8
    assert all(r["id"] == r["params"]["_meta"]["progressToken"] for r in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{"content": "invalid"}, {"content": [{"type": "text"}]}])
async def test_real_sdk_rejects_malformed_protocol_results(
    tmp_path: Path,
    response: dict[str, Any],
) -> None:
    async with http_wire_server({"results": {"echo": response}}) as (_, configuration):
        connection = MCPServerConnection(configuration, tmp_path)
        try:
            tools = await connection.connect()
            with pytest.raises(ValidationError):
                await tools[0].execute_prepared({})
            assert not connection.unavailable
        finally:
            await connection.close()


@pytest.mark.asyncio
async def test_real_sdk_keeps_unsupported_input_required_result_rejection(tmp_path: Path) -> None:
    response = types.InputRequiredResult(request_state="next").model_dump(
        by_alias=True, exclude_none=True
    )
    scenario = {
        "results": {"echo": response},
        "pages": {
            "": {
                "tools": [wire_tool()],
                "resultType": "complete",
                "cacheScope": "private",
                "ttlMs": 0,
            }
        },
    }
    async with http_wire_server(scenario) as (_, configuration):
        async with mcp_adapter.streamable_http_transport(configuration, tmp_path) as streams:
            read, write = cast(tuple[object, object], streams)[:2]
            session = cast(
                ClientSession,
                mcp_adapter._new_client_session(
                    read,
                    write,
                    on_closed=lambda: None,
                    on_tool_skipped=lambda: None,
                ),
            )
            async with session:
                await session.initialize()
                session.adopt(
                    types.DiscoverResult(
                        supported_versions=[types.LATEST_PROTOCOL_VERSION],
                        capabilities=types.ServerCapabilities(tools=types.ToolsCapability()),
                    )
                )
                listing = await session.list_tools()
                assert [tool.name for tool in listing.tools] == ["echo"]
                with pytest.raises(RuntimeError, match="InputRequiredResult"):
                    await session.call_tool("echo", {})
