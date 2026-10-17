"""MCP SDK transports, sessions and protocol tracing loaded for enabled Servers."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import mcp.types as types
from loguru import logger
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import (  # type: ignore[attr-defined]
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.shared.dispatcher import CallOptions, OnNotify, OnNotifyIntercept, OnRequest
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
from mcp_types.methods import validate_server_result
from pydantic import ValidationError

from aide.agent.tools.mcp import MCPClientSession, _field
from aide.config.config import MCPServerConfiguration
from aide.trace.scope import TraceToolCallScope, current_trace_tool_call


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
    *,
    protocol_trace: _MCPProtocolTrace | None = None,
) -> AsyncIterator[object]:
    """Create the SDK Streamable HTTP transport with configured static headers."""
    del workspace
    url = configuration.url
    if url is None:
        raise ValueError("streamable-http MCP Server requires a URL")
    http_client = create_mcp_http_client(headers=dict(configuration.headers))
    try:
        if protocol_trace is not None:
            try:
                http_client.event_hooks["request"].append(protocol_trace.http_request)
                http_client.event_hooks["response"].append(protocol_trace.http_response)
            except Exception:
                logger.warning("Execution Trace MCP HTTP hooks could not be installed")
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams
    finally:
        await http_client.aclose()


@dataclass(slots=True)
class _MCPTraceCall:
    scope: TraceToolCallScope


class _MCPProtocolTrace:
    """Associate actual HTTP endpoints with Tool calls by JSON-RPC identity."""

    def __init__(self, transport: str) -> None:
        self._source_prefix = "mcp_http" if transport == "streamable-http" else "mcp_stdio"
        self._pending: dict[int | str, _MCPTraceCall] = {}
        self._calls: dict[str, _MCPTraceCall] = {}

    async def http_request(self, request: httpx2.Request) -> None:
        try:
            request_id = request.extensions.get("aide.mcp_request_id")
            if request_id is None:
                if request.method != "POST" or not request.content:
                    return
                message = json.loads(request.content)
                if not isinstance(message, dict) or message.get("method") != "tools/call":
                    return
                request_id = message.get("id")
            if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
                return
            call = self._pending.get(request_id)
            if call is not None:
                # Request extensions follow redirects, including redirects that drop the body.
                request.extensions["aide.mcp_request_id"] = request_id
                call.scope.set_endpoint(str(request.url))
        except Exception:
            logger.warning("Execution Trace MCP HTTP request endpoint could not be captured")

    async def http_response(self, response: httpx2.Response) -> None:
        request_id = response.request.extensions.get("aide.mcp_request_id")
        if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
            return
        call = self._pending.get(request_id)
        if call is not None:
            call.scope.set_endpoint(str(response.request.url))

    def begin(self, scope: TraceToolCallScope | None) -> None:
        if scope is not None:
            scope.set_protocol(self._source_prefix)
            self._calls[scope.handle.record_id] = _MCPTraceCall(scope)

    def finish(self, scope: TraceToolCallScope | None) -> None:
        if scope is None:
            return
        self._calls.pop(scope.handle.record_id, None)
        for request_id, pending in tuple(self._pending.items()):
            if pending.scope.handle == scope.handle:
                self._pending.pop(request_id, None)

    def sent(self, item: object) -> None:
        message = _field(item, "message")
        if _field(message, "method") != "tools/call":
            return
        scope = current_trace_tool_call()
        request_id = _field(message, "id")
        if scope is None or isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
            return
        call = self._calls.setdefault(scope.handle.record_id, _MCPTraceCall(scope))
        self._pending[request_id] = call

    def received(self, item: object) -> None:
        message = _field(item, "message")
        if not isinstance(message, (types.JSONRPCResponse, types.JSONRPCError)):
            return
        request_id = _field(message, "id")
        if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
            return
        self._pending.pop(request_id, None)


class _MCPTraceReadStream:
    def __init__(self, stream: object, capture: _MCPProtocolTrace) -> None:
        self._stream = stream
        self._capture = capture

    async def receive(self) -> object:
        item = await cast(Any, self._stream).receive()
        try:
            self._capture.received(item)
        except Exception:
            logger.warning("Execution Trace MCP response could not be observed")
        return item

    def __aiter__(self) -> Any:
        async def read() -> AsyncIterator[object]:
            async for item in cast(Any, self._stream):
                try:
                    self._capture.received(item)
                except Exception:
                    logger.warning("Execution Trace MCP response could not be observed")
                yield item

        return read()

    async def __aenter__(self) -> _MCPTraceReadStream:
        await cast(Any, self._stream).__aenter__()
        return self

    async def __aexit__(self, *args: object) -> object:
        return await cast(Any, self._stream).__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _MCPTraceWriteStream:
    def __init__(self, stream: object, capture: _MCPProtocolTrace) -> None:
        self._stream = stream
        self._capture = capture

    async def send(self, item: object) -> None:
        try:
            self._capture.sent(item)
        except Exception:
            logger.warning("Execution Trace MCP request could not be observed")
        await cast(Any, self._stream).send(item)

    async def __aenter__(self) -> _MCPTraceWriteStream:
        await cast(Any, self._stream).__aenter__()
        return self

    async def __aexit__(self, *args: object) -> object:
        return await cast(Any, self._stream).__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


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
