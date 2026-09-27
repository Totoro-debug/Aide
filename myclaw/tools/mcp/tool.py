"""MCP SDK adapter for one configured Server and its discovered Tools."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import anyio
import mcp.types as types
from loguru import logger
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import (  # type: ignore[attr-defined]
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.shared.dispatcher import CallOptions, OnNotify, OnNotifyIntercept, OnRequest
from mcp.shared.exceptions import MCPError
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
from mcp_types.methods import validate_server_result
from pydantic import ValidationError

from myclaw.config.config import MCPServerConfiguration
from myclaw.permission.policy import MCPToolIdentity, ToolInvocationFacts
from myclaw.tools.base import BaseTool, ToolError
from myclaw.utils.async_tasks import await_task_preserving_cancellation

_MISSING = object()
_NO_OUTPUT = "(no output)"
_CONNECTION_UNAVAILABLE = "MCP Server connection is unavailable."
_TIMEOUT_TEMPLATE = "MCP Tool call timed out after {timeout:g} seconds."


class MCPToolSession(Protocol):
    """The call surface required by one MCPTool."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object: ...


class MCPDiscoverySession(Protocol):
    """The discovery surface required by the paginated tools/list adapter."""

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object: ...


class MCPClientSession(MCPToolSession, MCPDiscoverySession, Protocol):
    """The session surface required to connect and discover one MCP Server."""

    async def initialize(self) -> object: ...


class MCPToolSchemaError(ValueError):
    """Raised when a discovered MCP Tool cannot enter the local Tool contract."""


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """The validated, model-facing projection of one remote MCP Tool."""

    server_name: str
    remote_name: str
    model_name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.server_name, "server_name"),
            (self.remote_name, "remote_name"),
            (self.model_name, "model_name"),
            (self.description, "description"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"MCP Tool {field_name} must be a non-empty string")
        _validate_input_schema(self.parameters)
        object.__setattr__(self, "parameters", deepcopy(self.parameters))


class MCPTool(BaseTool):
    """Expose one discovered MCP Tool through the common BaseTool boundary."""

    parameters: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        spec: MCPToolSpec,
        session: MCPToolSession,
        *,
        call_timeout: float = 60,
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(spec, MCPToolSpec):
            raise TypeError("MCPTool requires an MCPToolSpec")
        if session is None:
            raise TypeError("MCPTool requires an MCP client session")
        _validate_timeout(call_timeout, field="call_timeout")
        self.server_name = spec.server_name
        self.remote_name = spec.remote_name
        self.name = spec.model_name
        self.description = spec.description
        object.__setattr__(self, "parameters", deepcopy(spec.parameters))
        self._session = session
        self._call_timeout = float(call_timeout)
        self._on_closed = on_closed
        self._unavailable = False

    @property
    def unavailable(self) -> bool:
        """Whether a closed MCP session was observed during a Tool call."""
        return self._unavailable

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Forward the complete argument object without local schema processing."""
        if self._unavailable:
            raise ToolError(_CONNECTION_UNAVAILABLE)
        return deepcopy(arguments)

    def build_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        """Attach the complete remote identity to the normalized call facts."""
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=prepared_arguments,
            mcp_identity=MCPToolIdentity(
                server_name=self.server_name,
                remote_name=self.remote_name,
                model_name=self.name,
            ),
        )

    async def execute_prepared(self, arguments: dict[str, Any]) -> str:
        """Call the remote Tool and project its result to text."""
        if self._unavailable:
            raise ToolError(_CONNECTION_UNAVAILABLE)
        try:
            async with asyncio.timeout(self._call_timeout):
                result = await self._session.call_tool(self.remote_name, deepcopy(arguments))
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            raise ToolError(_TIMEOUT_TEMPLATE.format(timeout=self._call_timeout)) from error
        except Exception as error:
            if isinstance(error, MCPError) and error.code == types.CONNECTION_CLOSED:
                self._mark_unavailable()
                raise ToolError(_CONNECTION_UNAVAILABLE) from error
            raise

        content = _content_text(result)
        if _field(result, "is_error", "isError") is True:
            raise ToolError(content)
        return content

    def _mark_unavailable(self) -> None:
        if self._unavailable:
            return
        self._unavailable = True
        if self._on_closed is not None:
            self._on_closed()

    def _observe_connection_closed(self) -> None:
        self._unavailable = True


def normalize_nullable(value: Any) -> Any:
    """Normalize nullable at Schema positions, preserving names and JSON data."""
    if isinstance(value, dict):
        nullable = value.get("nullable") is True
        normalized = deepcopy(value)
        normalized.pop("nullable", None)
        for key, item in value.items():
            if key in {
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
                "dependencies",
            } and isinstance(item, dict):
                normalized[key] = {
                    name: normalize_nullable(schema) for name, schema in item.items()
                }
            elif key in {
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
            }:
                normalized[key] = normalize_nullable(item)
            elif key == "items" or key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
                if isinstance(item, list):
                    normalized[key] = [normalize_nullable(schema) for schema in item]
                elif key == "items":
                    normalized[key] = normalize_nullable(item)
        if not nullable:
            return normalized

        type_value = normalized.get("type", _MISSING)
        if isinstance(type_value, str):
            normalized["type"] = [type_value] if type_value == "null" else [type_value, "null"]
            return normalized
        if isinstance(type_value, list):
            if "null" not in type_value:
                normalized["type"] = [*type_value, "null"]
            return normalized
        if type_value is _MISSING:
            return {"anyOf": [normalized, {"type": "null"}]}
        return normalized
    return deepcopy(value)


def mcp_tool_spec_from_remote(
    remote_tool: object,
    *,
    server_name: str,
    model_name: str,
) -> MCPToolSpec:
    """Validate and project one SDK or fake Tool object into an ``MCPToolSpec``."""
    remote_name = _field(remote_tool, "name")
    if not isinstance(remote_name, str) or not remote_name:
        raise MCPToolSchemaError("MCP Tool name must be a non-empty string")

    input_schema = _field(remote_tool, "input_schema", "inputSchema")
    _validate_input_schema(input_schema)

    normalized = normalize_nullable(input_schema)
    if not isinstance(normalized, dict) or normalized.get("type") != "object":
        raise MCPToolSchemaError("MCP Tool inputSchema must use an object root")

    description = _field(remote_tool, "description")
    if description is _MISSING or description is None or description == "":
        description = remote_name
    if not isinstance(description, str):
        raise MCPToolSchemaError("MCP Tool description must be a string")

    return MCPToolSpec(
        server_name=server_name,
        remote_name=remote_name,
        model_name=model_name,
        description=description,
        parameters=cast(dict[str, Any], normalized),
    )


async def discover_mcp_tool_specs(
    session: MCPDiscoverySession,
    *,
    server_name: str,
    model_name_for: Callable[[str], str] | None = None,
    on_tool_skipped: Callable[[], None] | None = None,
) -> tuple[MCPToolSpec, ...]:
    """Discover every page of Tools, stopping safely on a repeated cursor."""
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("MCP server_name must be a non-empty string")
    cursor: str | None = None
    seen_cursors: set[str] = set()
    specs: list[MCPToolSpec] = []
    while True:
        params = None if cursor is None else types.PaginatedRequestParams(cursor=cursor)
        page = await session.list_tools(params=params)
        tools = _field(page, "tools")
        if not isinstance(tools, (list, tuple)):
            raise TypeError("MCP tools/list returned an invalid tools collection")
        for remote_tool in tools:
            try:
                remote_name = _field(remote_tool, "name")
                if not isinstance(remote_name, str) or not remote_name:
                    raise MCPToolSchemaError("MCP Tool name must be a non-empty string")
                allocator = model_name_for or (lambda name: name)
                model_name = allocator(remote_name)
                if not isinstance(model_name, str) or not model_name:
                    raise MCPToolSchemaError("MCP Tool model name must be a non-empty string")
                specs.append(
                    mcp_tool_spec_from_remote(
                        remote_tool,
                        server_name=server_name,
                        model_name=model_name,
                    )
                )
            except MCPToolSchemaError:
                if on_tool_skipped is not None:
                    on_tool_skipped()
                continue

        next_cursor = _field(page, "next_cursor", "nextCursor")
        if next_cursor is _MISSING or next_cursor is None:
            break
        if not isinstance(next_cursor, str):
            raise TypeError("MCP tools/list returned an invalid cursor")
        if next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    specs.sort(key=lambda item: item.remote_name)
    return tuple(specs)


async def discover_mcp_tools(
    session: MCPClientSession,
    *,
    server_name: str,
    model_name_for: Callable[[str], str] | None = None,
    call_timeout: float = 60,
    on_closed: Callable[[], None] | None = None,
    on_tool_skipped: Callable[[], None] | None = None,
) -> tuple[MCPTool, ...]:
    """Discover and wrap one Server's valid Tools."""
    specs = await discover_mcp_tool_specs(
        session,
        server_name=server_name,
        model_name_for=model_name_for,
        on_tool_skipped=on_tool_skipped,
    )
    return tuple(
        MCPTool(spec, session, call_timeout=call_timeout, on_closed=on_closed) for spec in specs
    )


class MCPServerConnection:
    """Own one SDK transport/session and expose its current discovered Tool set."""

    def __init__(
        self,
        configuration: MCPServerConfiguration,
        workspace: Path,
        *,
        transport_factory: Callable[
            [MCPServerConfiguration, Path], AbstractAsyncContextManager[object]
        ]
        | None = None,
        session_factory: Callable[[object, object], MCPClientSession] | None = None,
        model_name_for: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(configuration, MCPServerConfiguration):
            raise TypeError("MCP Server connection requires MCPServerConfiguration")
        if not isinstance(workspace, Path):
            raise TypeError("MCP Server connection requires a Path workspace")
        self.configuration = configuration
        self.workspace = workspace
        self._transport_factory = transport_factory or _transport_for
        self._session_factory = session_factory
        self._model_name_for = model_name_for
        self._lifecycle_task: asyncio.Task[None] | None = None
        self._close_requested: asyncio.Event | None = None
        self._session: MCPClientSession | None = None
        self._tools: tuple[MCPTool, ...] = ()
        self._unavailable = False
        self._skipped_tool_count = 0

    @property
    def unavailable(self) -> bool:
        return self._unavailable

    @property
    def skipped_tool_count(self) -> int:
        """Return the number of invalid remote Tools skipped during discovery."""
        return self._skipped_tool_count

    async def connect(self) -> tuple[MCPTool, ...]:
        """Open transport/session, initialize, and discover a deterministic Tool set."""
        if not self.configuration.enabled:
            await self.close()
            return ()
        await self.close()
        self._unavailable = False
        self._skipped_tool_count = 0
        ready: asyncio.Future[tuple[MCPTool, ...]] = asyncio.get_running_loop().create_future()
        close_requested = asyncio.Event()
        lifecycle_task = asyncio.create_task(self._run_lifecycle(ready, close_requested))
        self._lifecycle_task = lifecycle_task
        self._close_requested = close_requested
        try:
            return await asyncio.shield(ready)
        except asyncio.CancelledError as cancellation:
            lifecycle_task.cancel()
            try:
                await await_task_preserving_cancellation(lifecycle_task)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            try:
                await await_task_preserving_cancellation(lifecycle_task)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        finally:
            if lifecycle_task.done() and self._lifecycle_task is lifecycle_task:
                self._lifecycle_task = None
                self._close_requested = None

    async def close(self) -> None:
        """Close the session and transport, releasing all SDK resources."""
        lifecycle_task = self._lifecycle_task
        close_requested = self._close_requested
        self._lifecycle_task = None
        self._close_requested = None
        if lifecycle_task is None:
            return
        if close_requested is not None:
            close_requested.set()
        await await_task_preserving_cancellation(lifecycle_task)

    async def _run_lifecycle(
        self,
        ready: asyncio.Future[tuple[MCPTool, ...]],
        close_requested: asyncio.Event,
    ) -> None:
        stack = AsyncExitStack()

        def on_closed() -> None:
            if self._close_requested is close_requested:
                self._unavailable = True
                for tool in self._tools:
                    tool._observe_connection_closed()

        try:
            async with asyncio.timeout(float(self.configuration.connect_timeout)):
                transport = self._transport_factory(self.configuration, self.workspace)
                streams = await stack.enter_async_context(transport)
                read_stream, write_stream = _transport_streams(streams)
                session = (
                    self._session_factory(read_stream, write_stream)
                    if self._session_factory is not None
                    else _new_client_session(
                        read_stream,
                        write_stream,
                        on_closed=on_closed,
                        on_tool_skipped=self._record_tool_skip,
                    )
                )
                entered_session = await stack.enter_async_context(cast(Any, session))
                self._session = cast(MCPClientSession, entered_session)
                await self._session.initialize()
                self._tools = await discover_mcp_tools(
                    self._session,
                    server_name=self.configuration.mcp_name,
                    model_name_for=self._model_name_for,
                    call_timeout=self.configuration.call_timeout,
                    on_closed=on_closed,
                    on_tool_skipped=self._record_tool_skip,
                )
            ready.set_result(self._tools)
            await close_requested.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            else:
                raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                raise
        finally:
            self._session = None
            self._tools = ()
            await stack.aclose()

    def _record_tool_skip(self) -> None:
        self._skipped_tool_count += 1
        logger.error(
            "MCP Tool skipped mcp_name={} phase=discovery type=MCPToolSchemaError",
            self.configuration.mcp_name,
        )


@asynccontextmanager
async def stdio_transport(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AsyncIterator[object]:
    """Create the SDK stdio transport for one configured Server."""
    command = configuration.command
    if command is None:
        raise ValueError("stdio MCP Server requires a command")
    parameters = StdioServerParameters(
        command=command,
        args=list(configuration.args),
        env=dict(os.environ),
        cwd=configuration.resolve_cwd(workspace),
    )
    async with stdio_client(parameters) as streams:
        yield streams


@asynccontextmanager
async def streamable_http_transport(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AsyncIterator[object]:
    """Create the SDK Streamable HTTP transport with configured static headers."""
    del workspace
    url = configuration.url
    if url is None:
        raise ValueError("streamable-http MCP Server requires a URL")
    http_client = create_mcp_http_client(headers=dict(configuration.headers))
    try:
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams
    finally:
        await http_client.aclose()


def _transport_for(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AbstractAsyncContextManager[object]:
    if configuration.transport == "stdio":
        return stdio_transport(configuration, workspace)
    if configuration.transport == "streamable-http":
        return streamable_http_transport(configuration, workspace)
    raise ValueError(f"Unsupported MCP transport: {configuration.transport}")


class _MCPClientSession(ClientSession):
    async def validate_tool_result(self, name: str, result: types.CallToolResult) -> None:
        # ADR-0020 projects content only, without outputSchema validation or discovery.
        pass


class _MCPDispatcher(JSONRPCDispatcher[Any]):
    def __init__(
        self,
        read_stream: object,
        write_stream: object,
        *,
        on_closed: Callable[[], None],
        on_tool_skipped: Callable[[], None],
        protocol_version: Callable[[], str | None],
    ) -> None:
        super().__init__(cast(Any, read_stream), cast(Any, write_stream))
        self._on_closed = on_closed
        self._on_tool_skipped = on_tool_skipped
        self._protocol_version = protocol_version

    async def send_raw_request(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        opts: CallOptions | None = None,
        *,
        _related_request_id: types.RequestId | None = None,
    ) -> dict[str, Any]:
        result = await super().send_raw_request(
            method, params, opts, _related_request_id=_related_request_id
        )
        if method != "tools/list" or not isinstance(result.get("tools"), list):
            return result
        version = self._protocol_version()
        if version is not None:
            try:
                validate_server_result("tools/list", version, {**result, "tools": []})
            except ValidationError:
                return result
        tools = []
        for tool in result["tools"]:
            try:
                # The version-free Tool model is looser than the negotiated wire schema.
                if version is not None:
                    validate_server_result("tools/list", version, {**result, "tools": [tool]})
                types.Tool.model_validate(tool, by_name=False)
            except ValidationError:
                self._on_tool_skipped()
            else:
                tools.append(tool)
        return {**result, "tools": tools}

    async def run(
        self,
        on_request: OnRequest,
        on_notify: OnNotify,
        on_notify_intercept: OnNotifyIntercept | None = None,
        *,
        task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        try:
            await super().run(on_request, on_notify, on_notify_intercept, task_status=task_status)
        finally:
            self._on_closed()


def _new_client_session(
    read_stream: object,
    write_stream: object,
    *,
    on_closed: Callable[[], None],
    on_tool_skipped: Callable[[], None],
) -> MCPClientSession:
    session = _MCPClientSession(
        dispatcher=_MCPDispatcher(
            read_stream,
            write_stream,
            on_closed=on_closed,
            on_tool_skipped=on_tool_skipped,
            protocol_version=lambda: session.protocol_version,
        )
    )
    return session


def _transport_streams(streams: object) -> tuple[object, object]:
    if not isinstance(streams, (tuple, list)) or len(streams) < 2:
        raise TypeError("MCP transport factory must yield read and write streams")
    return streams[0], streams[1]


def _field(value: object, *names: str) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return _MISSING
    for name in names:
        candidate = getattr(value, name, _MISSING)
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _validate_input_schema(schema: object) -> None:
    if not isinstance(schema, dict):
        raise MCPToolSchemaError("MCP Tool inputSchema must be a dictionary")
    _validate_json_serializable(schema)
    if schema.get("type") != "object":
        raise MCPToolSchemaError("MCP Tool inputSchema must use an object root")


def _validate_json_serializable(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MCPToolSchemaError("MCP Tool inputSchema must be JSON serializable") from error


def _validate_timeout(value: float, *, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or not isfinite(float(value))
    ):
        raise ValueError(f"MCP {field} must be a positive number")


def _content_text(result: object) -> str:
    content = _field(result, "content")
    if not isinstance(content, (list, tuple)):
        raise TypeError("MCP Tool result content must be a sequence")
    parts = [
        block.text if isinstance(block, types.TextContent) else str(block) for block in content
    ]
    return "\n".join(parts) or _NO_OUTPUT


__all__ = [
    "MCPClientSession",
    "MCPDiscoverySession",
    "MCPServerConnection",
    "MCPTool",
    "MCPToolSchemaError",
    "MCPToolSession",
    "MCPToolSpec",
    "discover_mcp_tool_specs",
    "discover_mcp_tools",
    "mcp_tool_spec_from_remote",
    "normalize_nullable",
    "stdio_transport",
    "streamable_http_transport",
]
