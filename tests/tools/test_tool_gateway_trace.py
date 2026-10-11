from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from aide.agent.confirmation import ConfirmationAborted
from aide.agent.memory.dream import Dream
from aide.agent.runner import AgentRunner
from aide.agent.tools.base import BaseTool, ToolError
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.exec_host import PowerShellExecHost, resolve_exec_shell
from aide.agent.tools.core.exec_policy import ExecAssessment
from aide.agent.tools.core.web_fetch import HTTPResponseBoundary, WebFetchTool
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.mcp import MCPTool, MCPToolSpec
from aide.agent.tools.mcp_runtime import MCPRuntimeManager
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import (
    BuiltInToolCatalog,
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)
from aide.provider.model_router import ModelRouter
from aide.provider.models import AssistantModelMessage, ModelCompleted, ModelResponse, ModelUsage
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.scope import current_trace_tool_call, trace_scope
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureLayer,
    TraceDirection,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceRawHandle,
    TraceTokenUsage,
)
from tests.fixtures import (
    DetachedRequestPreparer,
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    StreamScript,
)
from tests.fixtures.mcp_wire import http_wire_server, stdio_requests, stdio_wire_configuration
from tests.fixtures.model_configuration import configuration
from tests.memory.test_dream import _FAKE_MEMORY_ROUTE_STATUS, _manager, _response
from tests.tools.core.test_web_fetch import FakeHTTPClient, FakeResolver, FakeResponse
from tests.tools.test_mcp import _Session
from tests.tools.test_shared_tool_context import _RecordingExecHost, _schedule_service
from tests.tools.test_subagent_tools import _harness

_RAW_RESULT = "raw-result-" * 6_000


class _TraceExecHost(PowerShellExecHost):
    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del command, cwd
        return ExecAssessment(syntax_confidence="high", syntax_uncertain=False)


def _trace_exec_gateway(
    workspace: Path,
    *,
    host_type: type[PowerShellExecHost] = _TraceExecHost,
    permission_context: PermissionContext | None = None,
) -> ToolGateway:
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=lambda name: r"C:\PowerShell\pwsh.exe" if name == "pwsh" else None,
        version_probe=lambda *_: (7, 5),
        environment=os.environ,
    )
    return build_agent_run_gateway(
        ToolGateway(
            catalog=BuiltInToolCatalog(exec_host=host_type(shell)),
            tool_context=ToolRunContext(workspace=workspace),
            permission_context=(
                PermissionContext(level="full-access", workspace_root=workspace)
                if permission_context is None
                else permission_context
            ),
        )
    )


async def _approve_exec_confirmation(_request: ConfirmationRequest) -> ConfirmationDecision:
    return "approved"


def _patch_python_process(
    monkeypatch: pytest.MonkeyPatch, script: str
) -> list[asyncio.subprocess.Process]:
    processes: list[asyncio.subprocess.Process] = []

    async def create_process(
        *, argv: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> asyncio.subprocess.Process:
        del argv
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    monkeypatch.setattr("aide.agent.tools.core.exec_host._create_process", create_process)
    return processes


async def _read_trace_raw(
    recorder: SQLiteTraceRecorder,
    record_id: str,
    source: str,
) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        page = await recorder.read_raw(
            record_id,
            source=source,
            offset=offset,
            limit_bytes=256 * 1024,
        )
        chunks.append(page.data)
        if page.next_offset is None:
            return b"".join(chunks)
        assert page.next_offset > offset
        offset = page.next_offset


async def _exec_output_handshake_server() -> tuple[asyncio.AbstractServer, asyncio.Event, int]:
    ready = asyncio.Event()

    async def receive_output_handshake(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            if await reader.readexactly(5) == b"ready":
                ready.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(receive_output_handshake, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    return server, ready, int(address[1])


def _exec_output_handshake_script(port: int) -> str:
    return "\n".join(
        (
            "import socket, sys, time",
            "sys.stdout.buffer.write(bytes(range(256)) * 1024)",
            "sys.stdout.buffer.flush()",
            "sys.stderr.buffer.write(bytes(reversed(range(256))) * 1024)",
            "sys.stderr.buffer.flush()",
            f"connection = socket.create_connection(('127.0.0.1', {port}))",
            "connection.sendall(b'ready')",
            "connection.close()",
            "time.sleep(60)",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 23])
async def test_exec_trace_preserves_large_raw_process_streams_and_tool_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stdout = bytes(range(256)) * 4096
    stderr = bytes(reversed(range(256))) * 4096
    script = "\n".join(
        (
            "import sys",
            "sys.stdout.buffer.write(bytes(range(256)) * 4096)",
            "sys.stdout.buffer.flush()",
            "sys.stderr.buffer.write(bytes(reversed(range(256))) * 4096)",
            "sys.stderr.buffer.flush()",
            f"sys.exit({exit_code})",
        )
    )
    _patch_python_process(monkeypatch, script)
    gateway = _trace_exec_gateway(workspace)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    arguments = json.dumps({"command": "Write-Output trace"})

    try:
        untraced = await gateway.call(
            ModelToolCall("exec-untraced", "exec", arguments),
            confirmation=_approve_exec_confirmation,
        )
        with trace_scope(NoOpTraceRecorder(), source):
            baseline = await gateway.call(
                ModelToolCall("exec-baseline", "exec", arguments),
                confirmation=_approve_exec_confirmation,
            )
        with trace_scope(_BrokenCaptureRecorder(), source):
            broken_capture = await gateway.call(
                ModelToolCall("exec-broken-capture", "exec", arguments),
                confirmation=_approve_exec_confirmation,
            )
        with trace_scope(recorder, source):
            traced = await gateway.call(
                ModelToolCall("exec-traced", "exec", arguments),
                confirmation=_approve_exec_confirmation,
            )
        await recorder.flush()

        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert record.tool_call_id == "exec-traced"
        assert record.status.value == "success"
        assert record.capture_state.value == "complete"
        assert detail.result == traced.to_dict()
        assert (traced.status, traced.content) == (baseline.status, baseline.content)
        assert (traced.status, traced.content) == (untraced.status, untraced.content)
        assert (traced.status, traced.content) == (broken_capture.status, broken_capture.content)
        assert traced.content.startswith(f"Exit code: {exit_code}\nstdout:\n")
        assert len(traced.content) <= 4000
        assert "\ufffd" in traced.content
        assert {raw.source: (raw.direction, raw.layer) for raw in detail.raw_sources} == {
            "exec_stdout": (TraceDirection.RESPONSE, TraceCaptureLayer.PROCESS_STREAM),
            "exec_stderr": (TraceDirection.RESPONSE, TraceCaptureLayer.PROCESS_STREAM),
            "tool_result": (TraceDirection.RESPONSE, TraceCaptureLayer.TOOL_RESULT),
        }
        assert await _read_trace_raw(recorder, record.record_id, "exec_stdout") == stdout
        assert await _read_trace_raw(recorder, record.record_id, "exec_stderr") == stderr
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_exec_trace_keeps_output_read_before_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server, output_ready, port = await _exec_output_handshake_server()
    _patch_python_process(monkeypatch, _exec_output_handshake_script(port))
    gateway = _trace_exec_gateway(workspace)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    execution: asyncio.Task[ToolResult] | None = None

    try:
        with trace_scope(recorder, source):
            execution = asyncio.create_task(
                gateway.call(
                    ModelToolCall(
                        "exec-timeout",
                        "exec",
                        json.dumps({"command": "Write-Output trace", "timeout": 1}),
                    ),
                    confirmation=_approve_exec_confirmation,
                )
            )
            await asyncio.wait_for(output_ready.wait(), timeout=5)
            timed_out = await asyncio.wait_for(execution, timeout=5)
        await recorder.flush()

        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert timed_out.status == "error"
        assert "Exec timed out after 1 seconds." in timed_out.content
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.TOOL_ERROR
        assert record.error_message == timed_out.content
        assert record.capture_state.value == "complete"
        assert await _read_trace_raw(recorder, record.record_id, "exec_stdout") == (
            bytes(range(256)) * 1024
        )
        assert await _read_trace_raw(recorder, record.record_id, "exec_stderr") == (
            bytes(reversed(range(256))) * 1024
        )
    finally:
        server.close()
        await server.wait_closed()
        if execution is not None and not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled_cleanup", [False, True])
async def test_exec_trace_keeps_output_read_before_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stalled_cleanup: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server, output_ready, port = await _exec_output_handshake_server()
    processes = _patch_python_process(monkeypatch, _exec_output_handshake_script(port))
    gateway = _trace_exec_gateway(workspace)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    execution: asyncio.Task[ToolResult] | None = None
    output_captured = asyncio.Event()
    captured_bytes = {"exec_stdout": 0, "exec_stderr": 0}
    append_raw = recorder.append_raw

    def capture_raw(handle: TraceCaptureHandle, *, source: str, data: bytes, **kwargs: Any) -> None:
        append_raw(handle, source=source, data=data, **kwargs)
        if source in captured_bytes:
            captured_bytes[source] += len(data)
            if all(size == 256 * 1024 for size in captured_bytes.values()):
                output_captured.set()

    monkeypatch.setattr(recorder, "append_raw", capture_raw)

    try:
        with trace_scope(recorder, source):
            execution = asyncio.create_task(
                gateway.call(
                    ModelToolCall(
                        "exec-cancel",
                        "exec",
                        json.dumps({"command": "Write-Output trace"}),
                    ),
                    confirmation=_approve_exec_confirmation,
                )
            )
            await asyncio.wait_for(output_ready.wait(), timeout=5)
            # Both streams must reach capture while the process is still alive and has no EOF.
            await asyncio.wait_for(output_captured.wait(), timeout=5)
            if stalled_cleanup:
                monkeypatch.setattr(processes[0], "kill", lambda: None)
                monkeypatch.setattr("aide.agent.tools.core.exec_host._PROCESS_REAP_TIMEOUT", 0.1)
            execution.cancel("cancelled after output handshake")
            with pytest.raises(asyncio.CancelledError, match="cancelled after output handshake"):
                await execution
        await recorder.flush()

        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.CANCELLED
        assert record.error_message == "cancelled after output handshake"
        assert record.capture_state.value == "complete"
        assert detail.result is None
        assert await _read_trace_raw(recorder, record.record_id, "exec_stdout") == (
            bytes(range(256)) * 1024
        )
        assert await _read_trace_raw(recorder, record.record_id, "exec_stderr") == (
            bytes(reversed(range(256))) * 1024
        )
    finally:
        server.close()
        await server.wait_closed()
        if execution is not None and not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                # Bypass the injected stalled kill so the test releases its real subprocess.
                asyncio.subprocess.Process.kill(process)
                await process.wait()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "declined"])
async def test_exec_trace_excludes_authorization_inspection_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ConfirmationDecision,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    process_count = 0

    async def create_process(
        *, argv: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> asyncio.subprocess.Process:
        nonlocal process_count
        del argv
        process_count += 1
        output = (
            json.dumps({"syntax_ok": True, "command_names": ["Set-Content"]})
            if process_count == 1
            else "command-output"
        )
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write({output!r})",
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr("aide.agent.tools.core.exec_host._create_process", create_process)
    gateway = _trace_exec_gateway(
        workspace,
        host_type=PowerShellExecHost,
        permission_context=PermissionContext(level="read-only", workspace_root=workspace),
    )
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )

    async def confirm(_request: ConfirmationRequest) -> ConfirmationDecision:
        return decision

    try:
        with trace_scope(recorder, source):
            result = await gateway.call(
                ModelToolCall(
                    "exec-inspection", "exec", '{"command":"Set-Content trace.txt data"}'
                ),
                confirmation=confirm,
            )
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        if decision == "approved":
            assert result.status == "success"
            assert process_count == 2
            assert await _read_trace_raw(recorder, record.record_id, "exec_stdout") == (
                b"command-output"
            )
        else:
            assert result.status == "refused"
            assert process_count == 1
            assert detail.raw_sources == ()
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_successful_mcp_uses_one_logical_record_and_marks_transport_gap(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = _Session()
    tool = MCPTool(
        MCPToolSpec(
            server_name="remote",
            remote_name="echo",
            model_name="mcp_remote_echo",
            description="Echo remote values.",
            parameters={"type": "object"},
        ),
        session,
    )
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(tool,))
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    try:
        with trace_scope(recorder, source):
            result = await gateway.call(ModelToolCall("mcp", tool.name, '{"value":"payload"}'))
        await recorder.flush()
        assert (result.status, result.content) == ("success", "ok")
        assert session.calls == [("echo", {"value": "payload"})]
        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert detail.result == result.to_dict()
        assert cast(dict[str, Any], detail.request)["arguments"] == {"value": "payload"}
        assert record.capture_state.value == "partial"
        assert detail.capture_issues[0].source == "mcp_transport"
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "stdio"])
async def test_mcp_runtime_trace_correlates_wire_messages_by_request_id(
    tmp_path: Path,
    transport: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scenario = {
        "result_fields": {"vendorResult": {"revision": 2, "opaque": ["kept"]}},
        "response_delay_ms_by_value": {"slow": 50},
        "ping_before_result": True,
    }
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        async with AsyncExitStack() as stack:
            if transport == "http":
                wire, configuration = await stack.enter_async_context(http_wire_server(scenario))
                configuration = replace(
                    configuration,
                    headers={"Authorization": "Bearer trace-secret"},
                )
                manager = MCPRuntimeManager(None, transport="streamable-http")
            else:
                wire = None
                configuration = stdio_wire_configuration(workspace, scenario)
                manager = MCPRuntimeManager(workspace, transport="stdio")

            try:
                await manager.start({"remote": configuration})
                tool = manager.snapshot[0]
                gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(tool,))
                shared_manager = manager.fork()
                stack.push_async_callback(shared_manager.close)
                await shared_manager.start({"remote": configuration})
                assert shared_manager.snapshot[0] is tool
                shared_gateway = ToolGateway(
                    catalog=BuiltInToolCatalog(), additional_tools=shared_manager.snapshot
                )
                source = TraceRootSource(
                    workspace_key=workspace_key_for(workspace),
                    workspace_path=str(workspace),
                    origin=TraceOrigin.FOREGROUND,
                    purpose=TracePurpose.AGENT_RUN,
                )
                calls = (
                    ModelToolCall(
                        "call-slow",
                        tool.name,
                        json.dumps({"value": "slow", "vendorArgument": {"kept": True}}),
                    ),
                    ModelToolCall(
                        "call-fast",
                        tool.name,
                        json.dumps({"value": "fast", "vendorArgument": {"kept": True}}),
                    ),
                )

                async def call_from_session(
                    call: ModelToolCall, gateway: ToolGateway
                ) -> ToolResult:
                    session_source = replace(
                        source, owner_session_id=call.id, run_id=f"run-{call.id}"
                    )
                    with trace_scope(recorder, session_source):
                        return await gateway.call(call)

                slow_result, fast_result = await asyncio.gather(
                    call_from_session(calls[0], gateway),
                    call_from_session(calls[1], shared_gateway),
                )
                await shared_manager.close()
                with trace_scope(recorder, source):
                    failed_result = await gateway.call(
                        ModelToolCall("call-failed", tool.name, '{"error":true}')
                    )

                assert (slow_result.status, slow_result.content) == ("success", "slow")
                assert (fast_result.status, fast_result.content) == ("success", "fast")
                assert failed_result.status == "error"
                await recorder.flush()

                records = (await recorder.list_records()).items
                assert len(records) == 3
                by_call_id = {record.tool_call_id: record for record in records}
                assert set(by_call_id) == {"call-slow", "call-fast", "call-failed"}

                prefix = "mcp_http" if transport == "http" else "mcp_stdio"
                for call_id, value in (("call-slow", "slow"), ("call-fast", "fast")):
                    record = by_call_id[call_id]
                    assert record.owner_session_id == call_id
                    assert record.run_id == f"run-{call_id}"
                    detail = await recorder.get_record(record.record_id)
                    assert detail is not None
                    assert cast(dict[str, Any], detail.request)["arguments"]["value"] == value
                    sources = {item.source: item for item in detail.raw_sources}
                    request_source = sources[f"{prefix}_request"]
                    response_source = sources[f"{prefix}_response"]
                    result_source = sources[f"{prefix}_result"]
                    assert (request_source.direction, request_source.layer) == (
                        TraceDirection.REQUEST,
                        TraceCaptureLayer.JSON_RPC_MESSAGE,
                    )
                    assert (response_source.direction, response_source.layer) == (
                        TraceDirection.RESPONSE,
                        TraceCaptureLayer.JSON_RPC_MESSAGE,
                    )
                    assert result_source.layer is TraceCaptureLayer.SDK_RESULT

                    request_bytes = (
                        await recorder.read_raw(
                            record.record_id,
                            source=f"{prefix}_request",
                            offset=0,
                            limit_bytes=4096,
                        )
                    ).data
                    assert b"trace-secret" not in request_bytes
                    request_message = json.loads(request_bytes)
                    response_message = json.loads(
                        (
                            await recorder.read_raw(
                                record.record_id,
                                source=f"{prefix}_response",
                                offset=0,
                                limit_bytes=4096,
                            )
                        ).data
                    )
                    raw_result = json.loads(
                        (
                            await recorder.read_raw(
                                record.record_id,
                                source=f"{prefix}_result",
                                offset=0,
                                limit_bytes=4096,
                            )
                        ).data
                    )
                    assert request_message["id"] == response_message["id"]
                    assert request_message["params"]["arguments"]["value"] == value
                    assert response_message["result"]["content"][0]["text"] == value
                    assert raw_result["vendorResult"] == {"revision": 2, "opaque": ["kept"]}
                    assert record.capture_state.value == "complete"
                assert by_call_id["call-slow"].root_id != by_call_id["call-fast"].root_id

                failed_record = by_call_id["call-failed"]
                failed_detail = await recorder.get_record(failed_record.record_id)
                assert failed_detail is not None
                failed_response = json.loads(
                    (
                        await recorder.read_raw(
                            failed_record.record_id,
                            source=f"{prefix}_response",
                            offset=0,
                            limit_bytes=4096,
                        )
                    ).data
                )
                assert failed_response["error"]["message"] == "fixture error"
                assert failed_record.status.value == "failed"
                assert failed_detail.capture_issues == ()

                wire_messages = wire.requests if wire is not None else stdio_requests(workspace)
                methods = [message["method"] for message in wire_messages if "method" in message]
                assert methods.count("initialize") == 1
                assert methods.count("tools/list") == 1
                assert methods.count("tools/call") == 3
                if wire is not None:
                    call_responses = [
                        response
                        for response in wire.responses
                        if "result" in response and "content" in response["result"]
                    ]
                    assert [
                        item["result"]["content"][0]["text"] for item in call_responses[:2]
                    ] == [
                        "fast",
                        "slow",
                    ]
            finally:
                await manager.close()
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_dream_edit_variant_records_arguments_and_actual_result(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "home")
    await manager.append_summary("Concise reports.", datetime.now(UTC))
    arguments = {
        "path": str(manager.long_term_path),
        "old_text": "## User Preference\n",
        "new_text": "## User Preference\nConcise reports.\n",
    }
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "", tool_calls=(ModelToolCall("dream-edit", "edit_file", json.dumps(arguments)),)
            ),
        )
    )
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
        trace_recorder=recorder,
    )
    try:
        outcome = await dream.run()
        assert outcome.memory_updated
        assert "Concise reports." in manager.memory_snapshot()
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert record.purpose is TracePurpose.DREAM
        assert record.owner_session_id is None
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.request)["arguments"] == arguments
        assert cast(dict[str, Any], detail.result)["content"] == "Long-term Memory updated."
    finally:
        await dream.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", {"path": "seed.txt"}),
        ("write_file", {"path": "written.txt", "content": "written"}),
        ("edit_file", {"path": "seed.txt", "old_text": "seed", "new_text": "changed"}),
        ("list_dir", {}),
        ("glob", {"pattern": "*.txt"}),
        ("grep", {"pattern": "seed"}),
        ("exec", {"command": "Get-Location"}),
        ("web_search", {"query": "trace"}),
        ("web_fetch", {"url": "https://public.example/page"}),
        ("schedule", {"action": "list"}),
        ("tool_search", {"query": "web search"}),
    ],
)
async def test_real_builtin_gateway_arguments_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, arguments: dict[str, object]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "seed.txt").write_text("seed", encoding="utf-8")
    response = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"page",))
    fetch = WebFetchTool(
        resolver=FakeResolver(("93.184.216.34",)),
        http_client=FakeHTTPClient((cast(HTTPResponseBoundary, response),)),
    )
    monkeypatch.setattr("aide.agent.tools.tool_gateway.WebFetchTool", lambda: fetch)
    monkeypatch.setattr(
        "aide.agent.tools.core.web_search._search_sync",
        lambda *_: [{"title": "trace", "href": "https://public.example", "body": "found"}],
    )
    gateway = build_agent_run_gateway(
        ToolGateway(
            catalog=BuiltInToolCatalog(exec_host=_RecordingExecHost()),
            tool_context=ToolRunContext(
                workspace=workspace, schedule_service=_schedule_service(workspace)
            ),
            permission_context=PermissionContext(level="full-access", workspace_root=workspace),
        )
    )
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="builtin-session",
        run_id="builtin-run",
    )
    call = ModelToolCall(f"call-{name}", name, json.dumps(arguments))

    async def approve(_request: ConfirmationRequest) -> ConfirmationDecision:
        return "approved"

    try:
        with trace_scope(recorder, source):
            result = await gateway.call(call, confirmation=approve)
        assert result.status == "success", result.content
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert (record.tool_name, record.owner_session_id, record.run_id) == (
            name,
            "builtin-session",
            "builtin-run",
        )
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.request)["arguments"] == arguments
        assert detail.result == result.to_dict()
        raw = await recorder.read_raw(
            record.record_id, source="tool_result", offset=0, limit_bytes=100_000
        )
        assert raw.data.decode() == result.content
        assert record.capture_state.value == (
            "partial" if name in {"exec", "web_fetch", "web_search"} else "complete"
        )
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_subagent_builtin_tools_share_the_gateway_trace_boundary(tmp_path: Path) -> None:
    state, _, pool, executor, gateway = _harness(tmp_path)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    source = TraceRootSource(
        workspace_key=workspace_key_for(state.workspace_path),
        workspace_path=str(state.workspace_path),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="subagent-owner",
        run_id="subagent-run",
    )
    results = []
    try:
        with trace_scope(recorder, source):
            results.append(
                await gateway.call(
                    ModelToolCall("spawn", "spawn_agent", '{"title":"Trace child","task":"Reply."}')
                )
            )
            agent_id = json.loads(results[0].content)["agent_id"]
            results.append(await gateway.call(ModelToolCall("list", "list_agents", "{}")))
            assert await asyncio.wait_for(executor.started.get(), 2) == agent_id
            executor.release.set()
            await pool.wait([agent_id])
            results.append(
                await gateway.call(
                    ModelToolCall("wait", "wait_agent", json.dumps({"agent_ids": [agent_id]}))
                )
            )
        await recorder.flush()
        records = (await recorder.list_records()).items
        assert len(records) == 3
        assert {r.tool_name for r in records} == {"spawn_agent", "list_agents", "wait_agent"}
        for result in results:
            assert result.status == "success"
            record = next(r for r in records if r.tool_call_id == result.tool_call_id)
            detail = await recorder.get_record(record.record_id)
            assert detail is not None and detail.request is not None
            assert detail.result == result.to_dict()
            assert record.owner_session_id == "subagent-owner"
    finally:
        executor.release.set()
        await recorder.close()


@pytest.mark.asyncio
async def test_stream_consumed_in_another_session_does_not_bind_its_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    response = ModelResponse(AssistantModelMessage("done"), ModelUsage(1, 1, 2), "stop")
    provider = ScriptedFakeProvider(streams=(StreamScript((ModelCompleted(response),)),))
    router = ModelRouter(
        configuration=configuration(), provider_factory=lambda _: provider, jitter=None
    )
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(_EchoTool(),))

    def source(session: str) -> TraceRootSource:
        return TraceRootSource(
            workspace_key=workspace_key_for(workspace),
            workspace_path=str(workspace),
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id=session,
        )

    try:
        with trace_scope(recorder, source("model-session")):
            events = router.stream("chat", messages=[{"role": "user", "content": "hi"}], tools=())
        with trace_scope(recorder, source("tool-session")):
            async for _ in events:
                pass
            await gateway.call(ModelToolCall("echo", "echo", '{"value":"payload"}'))
        await recorder.flush()
        records = (await recorder.list_records()).items
        model = next(r for r in records if r.kind is TraceKind.MODEL)
        tool = next(r for r in records if r.kind is TraceKind.TOOL)
        assert model.owner_session_id == "model-session"
        assert tool.owner_session_id == "tool-session"
        assert tool.parent_record_id is None
    finally:
        await router.close()
        await recorder.close()


class _EchoTool(BaseTool):
    name = "echo"
    description = "Return a fixed result for trace capture tests."
    required = ("value",)
    value: str

    def __init__(self) -> None:
        self.trace_record_ids: list[str | None] = []

    async def execute(self, *, value: str) -> str:
        assert value
        active_call = current_trace_tool_call()
        self.trace_record_ids.append(None if active_call is None else active_call.handle.record_id)
        return _RAW_RESULT


class _RefusingTool(BaseTool):
    name = "refuse"
    description = "Refuse execution for trace capture tests."

    def refusal_reason(self) -> str:
        return "Denied by the Tool."

    async def execute(self) -> str:
        raise AssertionError("A refused Tool must not execute")


class _CancelledTool(BaseTool):
    name = "cancel"
    description = "Cancel execution for trace capture tests."

    async def execute(self) -> str:
        raise asyncio.CancelledError()


class _BrokenCaptureRecorder(NoOpTraceRecorder):
    def set_request(self, handle: TraceRawHandle, request: object) -> None:
        del handle, request
        raise RuntimeError("request capture failed")

    def report_capture_issue(self, handle: TraceRawHandle, issue: TraceCaptureIssue) -> None:
        del handle, issue
        raise RuntimeError("issue capture failed")

    def append_raw(
        self,
        handle: TraceRawHandle,
        *,
        source: str,
        direction: TraceDirection,
        layer: TraceCaptureLayer,
        sequence: int,
        data: bytes,
        content_type: str | None = None,
        content_encoding: str | None = None,
    ) -> None:
        del handle, source, direction, layer, sequence, data, content_type, content_encoding
        raise RuntimeError("raw capture failed")

    def finish_call(
        self,
        handle: TraceCaptureHandle,
        outcome: TraceOutcome,
        *,
        result: object,
        usage: TraceTokenUsage,
        retry_count: int,
    ) -> None:
        del handle, outcome, result, usage, retry_count
        raise RuntimeError("finish capture failed")


@pytest.mark.asyncio
async def test_gateway_records_model_tool_call_and_unformatted_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    call = ModelToolCall(
        id="call-echo-1",
        name="echo",
        arguments=json.dumps({"value": "payload"}),
    )
    response = ModelResponse(
        message=AssistantModelMessage(content="", tool_calls=(call,)),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        finish_reason="tool_calls",
    )
    provider = ScriptedFakeProvider(completions=(response,))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        jitter=None,
    )
    echo = _EchoTool()
    gateway = ToolGateway(
        catalog=BuiltInToolCatalog(),
        additional_tools=(echo,),
    )
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
        subject_session_id="trace-session",
        run_id="trace-run",
    )

    try:
        with trace_scope(recorder, source):
            model_response = await router.complete(
                "chat",
                messages=[{"role": "user", "content": "call echo"}],
                tools=(),
            )
            result = await gateway.call(model_response.message.tool_calls[0])
        await recorder.flush()

        records = (await recorder.list_records()).items
        model_record = next(record for record in records if record.kind is TraceKind.MODEL)
        tool_record = next(record for record in records if record.kind is TraceKind.TOOL)
        detail = await recorder.get_record(tool_record.record_id)
        assert detail is not None
        assert result.status == "success"
        assert result.content == _RAW_RESULT
        assert tool_record.tool_call_id == call.id
        assert tool_record.parent_record_id == model_record.record_id
        assert tool_record.owner_session_id == "trace-session"
        assert tool_record.run_id == "trace-run"
        assert echo.trace_record_ids == [tool_record.record_id]
        assert current_trace_tool_call() is None
        assert detail.request == {
            "tool_call_id": call.id,
            "tool_name": call.name,
            "arguments": {"value": "payload"},
            "arguments_raw": call.arguments,
            "normalized_arguments": {"value": "payload"},
            "execution_arguments": {"value": "payload"},
        }
        assert detail.result == {
            "tool_call_id": call.id,
            "name": call.name,
            "status": "success",
            "content": _RAW_RESULT,
            "artifact": None,
        }
        assert len(detail.raw_sources) == 1
        assert detail.raw_sources[0].source == "tool_result"
        assert detail.raw_sources[0].layer is TraceCaptureLayer.TOOL_RESULT
        raw = await recorder.read_raw(
            tool_record.record_id,
            source="tool_result",
            offset=0,
            limit_bytes=100_000,
        )
        assert raw.direction is TraceDirection.RESPONSE
        assert raw.data.decode("utf-8") == _RAW_RESULT
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_fails", [False, True])
async def test_runner_trace_keeps_raw_and_final_artifact_result(
    tmp_path: Path, artifact_fails: bool
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    call = ModelToolCall("call-artifact", "echo", '{"value":"payload"}')
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                (
                    ModelCompleted(
                        ModelResponse(
                            AssistantModelMessage("", (call,)), ModelUsage(1, 1, 2), "tool_calls"
                        )
                    ),
                )
            ),
            StreamScript(
                (
                    ModelCompleted(
                        ModelResponse(AssistantModelMessage("done"), ModelUsage(1, 1, 2), "stop")
                    ),
                )
            ),
        )
    )
    router = ModelRouter(
        configuration=configuration(), provider_factory=lambda _: provider, jitter=None
    )
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(_EchoTool(),))
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
        run_id="artifact-run",
    )

    def externalize(result: ToolResult) -> ToolResult:
        if artifact_fails:
            raise OSError("artifact unavailable")
        output = BaseTool.handle_result(
            result.content,
            workspace=workspace,
            session_id="trace-session",
            tool_call_id=result.tool_call_id,
            limit=512,
        )
        return replace(result, content=output.content, artifact=output.artifact)

    try:
        initial = [{"role": "user", "content": "echo"}]
        with trace_scope(recorder, source):
            outcome = await AgentRunner(router, DetachedRequestPreparer(initial)).run(
                initial,
                model="chat",
                tool_gateway=gateway,
                on_output=None,
                confirmation=None,
                externalize_result=externalize,
                cancel_requested=None,
                max_iterations=50,
            )
        await recorder.flush()
        (record,) = [r for r in (await recorder.list_records()).items if r.kind is TraceKind.TOOL]
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        final_message = next(m for m in outcome.messages if m["role"] == "tool")
        returned = cast(dict[str, Any], detail.result)
        assert returned["content"] == final_message["content"]
        assert returned["status"] == final_message["status"]
        raw = await recorder.read_raw(
            record.record_id, source="tool_result", offset=0, limit_bytes=100_000
        )
        assert raw.data.decode() == _RAW_RESULT
        if artifact_fails:
            assert record.status.value == "failed"
            assert returned["artifact"] is None
        else:
            artifact = cast(dict[str, Any], returned["artifact"])
            assert (workspace / artifact["path"]).read_text(encoding="utf-8") == _RAW_RESULT
            assert len(returned["content"]) <= 512
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_runner_publishes_cancelled_tool_when_its_result_task_cancels_itself(
    tmp_path: Path,
) -> None:
    class SelfCancellingTool(BaseTool):
        name = "self_cancel"
        description = "Produce output while cancelling the calling Tool task."

        async def execute(self) -> str:
            task = asyncio.current_task()
            assert task is not None
            task.cancel("tool task cancelled")
            return _RAW_RESULT

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    call = ModelToolCall("call-self-cancel", "self_cancel", "{}")
    response = ModelResponse(AssistantModelMessage("", (call,)), ModelUsage(1, 1, 2), "tool_calls")
    provider = ScriptedFakeProvider(streams=(StreamScript((ModelCompleted(response),)),))
    router = ModelRouter(
        configuration=configuration(), provider_factory=lambda _: provider, jitter=None
    )
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(SelfCancellingTool(),))
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    try:
        initial = [{"role": "user", "content": "self cancel"}]
        with trace_scope(recorder, source):
            with pytest.raises(asyncio.CancelledError, match="tool task cancelled"):
                await AgentRunner(router, DetachedRequestPreparer(initial)).run(
                    initial,
                    model="chat",
                    tool_gateway=gateway,
                    on_output=None,
                    confirmation=None,
                    externalize_result=None,
                    cancel_requested=None,
                    max_iterations=50,
                )
        await recorder.flush()
        (record,) = [r for r in (await recorder.list_records()).items if r.kind is TraceKind.TOOL]
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.CANCELLED
        detail = await recorder.get_record(record.record_id)
        assert detail is not None and detail.result is None
        raw = await recorder.read_raw(
            record.record_id, source="tool_result", offset=0, limit_bytes=100_000
        )
        assert raw.data.decode() == _RAW_RESULT
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_confirmation_lifecycle_abort_is_traced_as_cancellation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    gateway = ToolGateway(
        catalog=BuiltInToolCatalog(),
        tool_context=ToolRunContext(workspace=workspace),
        permission_context=PermissionContext(level="read-only", workspace_root=workspace),
    )
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
    )
    abort = ConfirmationAborted("confirmation owner cancelled")

    async def cancel(_request: ConfirmationRequest) -> ConfirmationDecision:
        raise abort

    try:
        with trace_scope(recorder, source):
            with pytest.raises(ConfirmationAborted) as raised:
                await gateway.call(
                    ModelToolCall(
                        "call-aborted",
                        "write_file",
                        json.dumps({"path": str(workspace / "file"), "content": "blocked"}),
                    ),
                    confirmation=cancel,
                )
        assert raised.value is abort
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.CANCELLED
        assert record.error_message == str(abort)
        assert not (workspace / "file").exists()
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_gateway_records_failed_refused_and_cancelled_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    gateway = ToolGateway(
        catalog=BuiltInToolCatalog(),
        additional_tools=(_EchoTool(), _RefusingTool(), _CancelledTool()),
    )
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
    )
    invalid_call = ModelToolCall(id="call-invalid", name="echo", arguments="not-json")
    refused_call = ModelToolCall(id="call-refused", name="refuse", arguments="{}")
    cancelled_call = ModelToolCall(id="call-cancelled", name="cancel", arguments="{}")

    try:
        with trace_scope(recorder, source):
            invalid = await gateway.call(invalid_call)
            refused = await gateway.call(refused_call)
            with pytest.raises(asyncio.CancelledError):
                await gateway.call(cancelled_call)
        await recorder.flush()

        assert invalid.status == "error"
        assert refused.status == "refused"
        assert refused.content == "Denied by the Tool."
        records = {record.tool_call_id: record for record in (await recorder.list_records()).items}
        invalid_record = records[invalid_call.id]
        refused_record = records[refused_call.id]
        cancelled_record = records[cancelled_call.id]
        assert invalid_record.status.value == "failed"
        assert invalid_record.error_category is TraceErrorCategory.VALIDATION_ERROR
        assert invalid_record.error_message == "Tool arguments could not be parsed."
        assert refused_record.status.value == "failed"
        assert refused_record.error_category is TraceErrorCategory.PERMISSION_DENIED
        assert refused_record.error_message == refused.content
        assert cancelled_record.status.value == "failed"
        assert cancelled_record.error_category is TraceErrorCategory.CANCELLED
        for record, returned in ((invalid_record, invalid), (refused_record, refused)):
            detail = await recorder.get_record(record.record_id)
            assert detail is not None
            assert detail.result == returned.to_dict()
            assert detail.raw_sources == ()
        for record in (cancelled_record,):
            detail = await recorder.get_record(record.record_id)
            assert detail is not None
            assert detail.result is None
            assert detail.raw_sources == ()
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_gateway_records_declined_authorization_without_executing_mcp_tool(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")

    class Session:
        called = False

        async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            del name, arguments
            self.called = True
            raise AssertionError("A declined MCP Tool must not execute")

    session = Session()
    remote = MCPTool(
        MCPToolSpec(
            server_name="remote",
            remote_name="echo",
            model_name="mcp_remote_echo",
            description="Echo remote arguments.",
            parameters={"type": "object"},
        ),
        session,
    )
    gateway = ToolGateway(
        catalog=BuiltInToolCatalog(),
        additional_tools=(remote,),
        permission_context=PermissionContext(level="read-only"),
    )
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
    )
    call = ModelToolCall("call-declined-mcp", "mcp_remote_echo", '{"value":"payload"}')

    async def decline(_request: ConfirmationRequest) -> ConfirmationDecision:
        return "declined"

    try:
        with trace_scope(recorder, source):
            result = await gateway.call(call, confirmation=decline)
        await recorder.flush()

        assert result.status == "refused"
        assert session.called is False
        (record,) = (await recorder.list_records()).items
        assert record.tool_call_id == call.id
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.PERMISSION_DENIED
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert detail.result == result.to_dict()
        request = cast(dict[str, Any], detail.request)
        confirmation = cast(dict[str, Any], request["confirmation"])
        confirmation_request = cast(dict[str, Any], confirmation["request"])
        assert confirmation["decision"] == "declined"
        assert confirmation_request["tool_name"] == call.name
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_gateway_capture_failures_do_not_change_the_tool_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = _BrokenCaptureRecorder()
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(_EchoTool(),))
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
    )

    with trace_scope(recorder, source):
        result = await gateway.call(
            ModelToolCall(
                id="call-trace-failure",
                name="echo",
                arguments='{"value":"payload"}',
            )
        )

    assert result.status == "success"
    assert result.content == _RAW_RESULT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "gap_source"),
    [
        ("exec", "exec_process_stream"),
        ("web_fetch", "web_fetch_response"),
        ("web_search", "web_search_ddgs"),
    ],
)
async def test_gateway_marks_specialized_raw_capture_gaps(
    tmp_path: Path,
    tool_name: str,
    gap_source: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")

    class FormattedTool(BaseTool):
        name = tool_name
        description = "Return a preformatted result."

        async def execute(self) -> str:
            return "formatted result"

    gateway = ToolGateway._for_memory((FormattedTool(),))
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
    )

    try:
        with trace_scope(recorder, source):
            result = await gateway.call(ModelToolCall(f"call-{tool_name}", tool_name, "{}"))
        await recorder.flush()

        assert (result.status, result.content) == ("success", "formatted result")
        (record,) = (await recorder.list_records()).items
        assert record.capture_state.value == "partial"
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert detail.raw_sources[0].layer is TraceCaptureLayer.TOOL_RESULT
        assert detail.capture_issues[0].source == gap_source
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_gateway_marks_specialized_capture_gap_when_execution_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")

    class FailedExecTool(BaseTool):
        name = "exec"
        description = "Fail after entering the execution boundary."

        async def execute(self) -> str:
            raise ToolError("process failed")

    gateway = ToolGateway._for_memory((FailedExecTool(),))
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
    )

    try:
        with trace_scope(recorder, source):
            result = await gateway.call(ModelToolCall("call-exec-failed", "exec", "{}"))
        await recorder.flush()

        assert (result.status, result.content) == ("error", "process failed")
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.capture_state.value == "partial"
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert detail.result == result.to_dict()
        assert detail.capture_issues[0].source == "exec_process_stream"
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_shared_gateway_keeps_concurrent_session_traces_separate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    calls = (
        ModelToolCall("call-one", "echo", '{"value":"one"}'),
        ModelToolCall("call-two", "echo", '{"value":"two"}'),
    )
    provider = ScriptedFakeProvider(
        completions=tuple(
            ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            )
            for call in calls
        )
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        jitter=None,
    )
    echo = _EchoTool()
    gateway = ToolGateway(catalog=BuiltInToolCatalog(), additional_tools=(echo,))

    async def run(session_id: str) -> None:
        source = TraceRootSource(
            workspace_key=workspace_key_for(workspace),
            workspace_path=str(workspace),
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id=session_id,
            run_id=f"run-{session_id}",
        )
        with trace_scope(recorder, source):
            response = await router.complete(
                "chat",
                messages=[{"role": "user", "content": session_id}],
                tools=(),
            )
            await gateway.call(response.message.tool_calls[0])

    try:
        await asyncio.gather(run("session-one"), run("session-two"))
        await recorder.flush()

        records = (await recorder.list_records()).items
        assert len(records) == 4
        assert all(record_id is not None for record_id in echo.trace_record_ids)
        assert set(echo.trace_record_ids) == {
            record.record_id for record in records if record.kind is TraceKind.TOOL
        }
        for tool_record in (record for record in records if record.kind is TraceKind.TOOL):
            parent = next(
                record for record in records if record.record_id == tool_record.parent_record_id
            )
            assert parent.kind is TraceKind.MODEL
            assert parent.root_id == tool_record.root_id
            assert parent.owner_session_id == tool_record.owner_session_id
            assert tool_record.run_id == f"run-{tool_record.owner_session_id}"
    finally:
        await router.close()
        await recorder.close()
