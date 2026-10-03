"""Local wire fixtures for responses that SDK server models cannot declare."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest
from aiohttp import web
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from pydantic import TypeAdapter

from omni.config.config import MCPServerConfiguration


def wire_tool(name: str = "echo", **fields: Any) -> dict[str, Any]:
    return {"name": name, "inputSchema": {"type": "object"}, **fields}


def wire_result(text: str = "wire text", **fields: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], **fields}


class WireServer:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.requests: list[dict[str, Any]] = []
        self.received = asyncio.Queue[dict[str, Any]]()
        self.sse_reconnected = asyncio.Event()
        self.close_sse = asyncio.Event()
        self.sse_requests = 0

    def respond(self, request: dict[str, Any]) -> dict[str, Any] | None:
        self.requests.append(request)
        self.received.put_nowait(request)
        if "id" not in request:
            return None
        method = request["method"]
        params = request.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "wire-fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = self.scenario.get("pages", {}).get(
                params.get("cursor", ""), {"tools": [wire_tool()]}
            )
        elif method == "tools/call":
            arguments = params.get("arguments") or {}
            if arguments.get("hang"):
                return None
            if arguments.get("error"):
                return types.JSONRPCError(
                    jsonrpc="2.0",
                    id=request["id"],
                    error=types.ErrorData(code=-32603, message="fixture error"),
                ).model_dump(by_alias=True, exclude_none=True)
            result = self.scenario.get("results", {}).get(
                params["name"], wire_result(str(arguments.get("value", "wire text")))
            )
        else:
            result = {}
        return types.JSONRPCResponse(jsonrpc="2.0", id=request["id"], result=result).model_dump(
            by_alias=True, exclude_none=True
        )

    async def wait_for(self, method: str) -> dict[str, Any]:
        async with asyncio.timeout(10):
            while True:
                request = await self.received.get()
                if request["method"] == method:
                    return request

    async def http(self, request: web.Request) -> web.StreamResponse:
        if request.method == "GET":
            self.sse_requests += 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if self.sse_requests == 1:
                await response.write(b"retry: 1\n\n")
                await response.write_eof()
            else:
                self.sse_reconnected.set()
                await self.close_sse.wait()
            return response
        if request.method == "DELETE":
            self.close_sse.set()
            return web.Response(status=200)
        message = await request.json()
        response_data = self.respond(message)
        if response_data is None:
            return web.Response(status=202)
        return web.json_response(response_data, headers={"Mcp-Session-Id": "wire-session"})


@asynccontextmanager
async def http_wire_server(
    scenario: dict[str, Any],
) -> AsyncIterator[tuple[WireServer, MCPServerConfiguration]]:
    server = WireServer(scenario)
    app = web.Application()
    app.router.add_route("*", "/mcp", server.http)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        yield (
            server,
            MCPServerConfiguration(
                mcp_name="remote",
                enabled=True,
                transport="streamable-http",
                url=f"http://127.0.0.1:{port}/mcp",
                connect_timeout=10,
                call_timeout=5,
            ),
        )
    finally:
        server.close_sse.set()
        async with asyncio.timeout(10):
            await runner.cleanup()


def stdio_wire_configuration(
    directory: Path, scenario: dict[str, Any], *, name: str = "remote"
) -> MCPServerConfiguration:
    scenario_path = directory / f"{name}.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return MCPServerConfiguration(
        mcp_name=name,
        enabled=True,
        transport="stdio",
        command=sys.executable,
        args=(str(Path(__file__).resolve()), str(scenario_path)),
        connect_timeout=10,
        call_timeout=5,
    )


def stdio_requests(directory: Path, name: str = "remote") -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]


class ObservedLifetimes:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mcp.client.stdio as stdio

        import omni.agent.tools.mcp as adapter

        self.processes: list[Any] = []
        self.closed: list[asyncio.Event] = []
        self.close_counts: list[int] = []
        spawn = stdio._create_platform_compatible_process
        factory = adapter._new_client_session

        async def observed_spawn(**kwargs: Any) -> Any:
            process = await spawn(**kwargs)
            self.processes.append(process)
            return process

        def observed_session(read: object, write: object, **kwargs: Any) -> Any:
            closed = asyncio.Event()
            index = len(self.closed)
            self.closed.append(closed)
            self.close_counts.append(0)
            on_closed: Callable[[], None] | None = kwargs.get("on_closed")
            if on_closed is not None:

                def notify() -> None:
                    on_closed()
                    self.close_counts[index] += 1
                    closed.set()

                kwargs["on_closed"] = notify
            return factory(read, write, **kwargs)

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", observed_spawn)
        monkeypatch.setattr(adapter, "_new_client_session", observed_session)

    async def stop(self, index: int) -> None:
        process = self.processes[index]
        waiting = [asyncio.create_task(event.wait()) for event in self.closed if not event.is_set()]
        try:
            process.terminate()
            async with asyncio.timeout(10):
                await process.wait()
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)

    def assert_closed(self) -> None:
        assert all(process.returncode is not None for process in self.processes)
        assert self.close_counts == [1] * len(self.closed)


async def _run_stdio(scenario_path: Path) -> None:
    server = WireServer(json.loads(scenario_path.read_text(encoding="utf-8")))
    async with stdio_server() as (read, write):
        async for message in read:
            if isinstance(message, Exception):
                raise message
            request = message.message.model_dump(by_alias=True, exclude_none=True)
            with scenario_path.with_suffix(".jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(request) + "\n")
            response = server.respond(request)
            if response is not None:
                progress_token = (request.get("params") or {}).get("_meta", {}).get("progressToken")
                if request["method"] == "tools/call" and progress_token is not None:
                    await write.send(
                        SessionMessage(
                            types.JSONRPCNotification(
                                jsonrpc="2.0",
                                method="notifications/progress",
                                params={"progressToken": progress_token, "progress": 1, "total": 1},
                            )
                        )
                    )
                await write.send(
                    SessionMessage(TypeAdapter(types.JSONRPCMessage).validate_python(response))
                )


if __name__ == "__main__":
    asyncio.run(_run_stdio(Path(sys.argv[1])))
