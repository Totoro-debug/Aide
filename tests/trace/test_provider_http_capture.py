from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from aide.provider.anthropic import AnthropicProvider
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter
from aide.provider.models import ModelCompleted, ModelContinuation, TextDelta
from aide.provider.openai_compatible import OpenAICompatibleProvider
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.scope import trace_scope
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TraceErrorCategory, TraceOrigin, TracePurpose
from tests.fixtures import FakeClock
from tests.fixtures.model_configuration import routed_configuration

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
ProviderProtocol = Literal["openai-compatible", "anthropic"]


class _FakeAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    async def aclose(self) -> None:
        self.closed = True


class _FixedByteStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.consumed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self) -> None:
        return None


def _source(workspace: Path, session_id: str = "wire-session") -> TraceRootSource:
    return TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id=session_id,
        subject_session_id=session_id,
        run_id=f"run-{session_id}",
        restore_run_token=f"restore-{session_id}",
    )


def _router(
    transport: _FakeAsyncTransport,
    *,
    clock: FakeClock | None = None,
    protocol: ProviderProtocol = "openai-compatible",
) -> ModelRouter:
    configuration = routed_configuration()
    provider = configuration.models.providers["chat-provider"]
    provider = replace(
        provider,
        protocol=protocol,
        base_url="https://chat.example/v1?api_key=url-secret&sig=signature-secret&region=west",
    )
    configuration = replace(
        configuration,
        models=replace(
            configuration.models,
            providers={**configuration.models.providers, "chat-provider": provider},
        ),
    )
    return ModelRouter(
        configuration=configuration,
        provider_factory=lambda selected: (
            AnthropicProvider if selected.protocol == "anthropic" else OpenAICompatibleProvider
        )(
            selected,
            http_transport_factory=lambda: transport,
        ),
        clock=clock,
        jitter=None,
    )


async def _raw(
    recorder: SQLiteTraceRecorder,
    record_id: str,
    source: str,
) -> bytes:
    page = await recorder.read_raw(record_id, source=source, offset=0, limit_bytes=100_000)
    return page.data


@pytest.mark.asyncio
async def test_openai_complete_captures_wire_bytes_and_redacts_authentication(
    tmp_path: Path,
) -> None:
    response_body = (
        b'{"id":"chatcmpl-wire","object":"chat.completion","created":1,'
        b'"model":"chat-model","choices":[{"index":0,"message":{"role":"assistant",'
        b'"content":"wire answer"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}'
    )
    transport = _FakeAsyncTransport(
        lambda _request: httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "x-request-id": "response-id",
                "set-cookie": "session=response-secret",
                "location": "https://chat.example/next?api_key=redirect-secret&page=2",
                "content-location": "/result?sig=location-secret&format=json",
                "link": '<https://chat.example/next?token=link-secret&page=2>; rel="next"',
            },
            stream=_FixedByteStream(response_body),
        )
    )
    router = _router(transport)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    continuation = ModelContinuation("chat-provider", "retained reasoning")
    tools = (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            response = await router.complete(
                "title",
                messages=[
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "prior answer"},
                ],
                tools=tools,
                continuation=continuation,
            )
        await recorder.flush()

        assert response.message.content == "wire answer"
        assert len(transport.requests) == 1
        request = transport.requests[0]
        actual_request_body = request.content
        (record,) = (await recorder.list_records()).items
        detail = await recorder.get_record(record.record_id)
        assert detail is not None

        request_body = await _raw(recorder, record.record_id, "http_request")
        response_bytes = await _raw(recorder, record.record_id, "http_response")
        request_metadata = json.loads(
            (await _raw(recorder, record.record_id, "http_request_metadata")).decode("utf-8")
        )
        response_metadata = json.loads(
            (await _raw(recorder, record.record_id, "http_response_metadata")).decode("utf-8")
        )
        sent = json.loads(actual_request_body)

        assert request_body == actual_request_body
        assert response_bytes == response_body
        assert sent["messages"][-1]["reasoning_content"] == "retained reasoning"
        assert sent["tools"] == list(tools)
        assert sent["model"] == "chat-model"
        assert sent["max_tokens"] == 8192
        assert sent["temperature"] == 0.1
        assert sent["stream"] is False
        assert record.endpoint == "https://chat.example/v1?region=west%2Fchat%2Fcompletions"
        assert request_metadata["url"] == record.endpoint
        assert "authorization" not in request_metadata["headers"]
        assert "url-secret" not in json.dumps(request_metadata)
        assert "signature-secret" not in json.dumps(request_metadata)
        assert "chat-secret" not in json.dumps(request_metadata)
        assert response_metadata["status_code"] == 200
        assert response_metadata["headers"]["x-request-id"] == "response-id"
        assert "set-cookie" not in response_metadata["headers"]
        assert "response-secret" not in json.dumps(response_metadata)
        assert response_metadata["headers"]["location"] == "https://chat.example/next?page=2"
        assert response_metadata["headers"]["content-location"] == "/result?format=json"
        assert response_metadata["headers"]["link"] == (
            '<https://chat.example/next?page=2>; rel="next"'
        )
        assert "http_request" in {source.source for source in detail.raw_sources}
        assert not detail.capture_issues
    finally:
        await router.close()
        await recorder.close()
    assert transport.closed


@pytest.mark.asyncio
async def test_openai_stream_capture_does_not_prefetch_response_chunks(tmp_path: Path) -> None:
    chunks = (
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,'
        b'"model":"chat-model","choices":[{"index":0,"delta":{"content":"first"},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,'
        b'"model":"chat-model","choices":[{"index":0,"delta":{"content":"second"},'
        b'"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,'
        b'"model":"chat-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
        b"data: [DONE]\n\n",
    )
    response_stream = _FixedByteStream(*chunks)
    transport = _FakeAsyncTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": "stream-id"},
            stream=response_stream,
        )
    )
    router = _router(transport)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    try:
        with trace_scope(recorder, _source(workspace)):
            events = router.stream(
                "title",
                messages=[{"role": "user", "content": "stream question"}],
                tools=(),
            )
            first = await asyncio.wait_for(anext(events), timeout=2)
            assert first == TextDelta(delta="first")
            assert response_stream.consumed == 1
            remaining = [event async for event in events]
        await recorder.flush()

        assert remaining[0] == TextDelta(delta="second")
        assert isinstance(remaining[-1], ModelCompleted)
        assert remaining[-1].response.message.content == "firstsecond"
        (record,) = (await recorder.list_records()).items
        assert await _raw(recorder, record.record_id, "http_response") == b"".join(chunks)
        detail = await recorder.get_record(record.record_id)
        assert detail is not None and not detail.capture_issues
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_http_retries_keep_only_final_attempt_and_isolate_sessions(
    tmp_path: Path,
) -> None:
    attempts: dict[str, int] = {}
    response_bodies = {
        "session-retry": _completion_body("retry answer"),
        "session-other": _completion_body("other answer"),
    }

    def handle(request: httpx.Request) -> httpx.Response:
        label = json.loads(request.content)["messages"][0]["content"]
        attempts[label] = attempts.get(label, 0) + 1
        if label == "session-retry" and attempts[label] == 1:
            return httpx.Response(
                429,
                headers={"content-type": "application/json", "x-request-id": "retry-failure"},
                stream=_FixedByteStream(
                    b'{"error":{"message":"rate limited","type":"rate_limit_error",'
                    b'"code":"rate_limit_exceeded"}}'
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": f"ok-{label}"},
            stream=_FixedByteStream(response_bodies[label]),
        )

    transport = _FakeAsyncTransport(handle)
    router = _router(transport, clock=FakeClock(NOW))
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def invoke(session_id: str) -> str:
        with trace_scope(recorder, _source(workspace, session_id)):
            result = await router.complete(
                "title",
                messages=[{"role": "user", "content": session_id}],
                tools=(),
            )
            return result.message.content

    try:
        assert tuple(await asyncio.gather(invoke("session-retry"), invoke("session-other"))) == (
            "retry answer",
            "other answer",
        )
        await recorder.flush()

        records = {record.owner_session_id: record for record in (await recorder.list_records()).items}
        assert records["session-retry"].retry_count == 1
        assert records["session-other"].retry_count == 0
        for session_id, expected_body in response_bodies.items():
            record = records[session_id]
            assert await _raw(recorder, record.record_id, "http_response") == expected_body
            request_body = json.loads(await _raw(recorder, record.record_id, "http_request"))
            assert request_body["messages"][0]["content"] == session_id
            detail = await recorder.get_record(record.record_id)
            assert detail is not None and not detail.capture_issues
        assert attempts == {"session-retry": 2, "session-other": 1}
    finally:
        await router.close()
        await recorder.close()


def _completion_body(content: str, protocol: ProviderProtocol = "openai-compatible") -> bytes:
    if protocol == "anthropic":
        return json.dumps(
            {
                "id": "msg-complete",
                "type": "message",
                "role": "assistant",
                "model": "chat-model",
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
            separators=(",", ":"),
        ).encode("utf-8")
    return json.dumps(
        {
            "id": "chatcmpl-complete",
            "object": "chat.completion",
            "created": 1,
            "model": "chat-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _anthropic_stream_chunks() -> tuple[bytes, ...]:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-stream",'
        b'"type":"message","role":"assistant","model":"chat-model","content":[],'
        b'"usage":{"input_tokens":4,"output_tokens":0}}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"partial"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":1}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_wire_conversion_and_consumption_match_noop(
    tmp_path: Path,
    streaming: bool,
) -> None:
    continuation_blocks = (
        {"type": "thinking", "thinking": "retained reasoning", "signature": "signed-thinking"},
        {"type": "tool_use", "id": "call-1", "name": "read_file", "input": {"path": "a.txt"}},
    )
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": "use signed continuation"},
        {"role": "tool", "tool_call_id": "call-1", "content": "file text"},
    ]
    tools: tuple[dict[str, Any], ...] = (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    )
    outcomes: list[object] = []
    request_bodies: list[bytes] = []
    for traced in (False, True):
        chunks = (
            _anthropic_stream_chunks() if streaming else (_completion_body("partial", "anthropic"),)
        )
        stream = _FixedByteStream(*chunks)

        def handle(
            _: httpx.Request,
            response_stream: httpx.AsyncByteStream = stream,
        ) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream" if streaming else "application/json"},
                stream=response_stream,
            )

        transport = _FakeAsyncTransport(handle)
        router = _router(transport, protocol="anthropic")
        recorder = (
            SQLiteTraceRecorder(tmp_path / "traces.sqlite3") if traced else NoOpTraceRecorder()
        )
        try:
            with trace_scope(recorder, _source(tmp_path)):
                continuation = ModelContinuation("chat-provider", continuation_blocks)
                if streaming:
                    events = router.stream(
                        "title", messages=messages, tools=tools, continuation=continuation
                    )
                    first = await anext(events)
                    assert first == TextDelta("partial")
                    assert stream.consumed == 1  # Capturing must not pre-read the final chunk.
                    remaining = [event async for event in events]
                    assert isinstance(remaining[-1], ModelCompleted)
                    outcome = remaining[-1].response
                else:
                    outcome = await router.complete(
                        "title",
                        messages=messages,
                        tools=tools,
                        continuation=continuation,
                    )
            outcomes.append(outcome)
            assert outcome.message.content == "partial"
            assert outcome.usage.total_tokens == 5
            (request,) = transport.requests
            request_bodies.append(request.content)
            sent = json.loads(request.content)
            assert sent["system"] == "System instruction"
            assert sent["messages"] == [
                {"role": "user", "content": "Read the file"},
                {"role": "assistant", "content": list(continuation_blocks)},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "file text"}
                    ],
                },
            ]
            assert sent["tools"] == [
                {
                    "name": "read_file",
                    "description": "Read a text file.",
                    "input_schema": tools[0]["function"]["parameters"],
                }
            ]
            assert sent["max_tokens"] == 8192
            assert sent["temperature"] == 0.1
            assert sent["output_config"] == {"effort": "high"}
            assert sent["stream"] is streaming
            if isinstance(recorder, SQLiteTraceRecorder):
                await recorder.flush()
                (record,) = (await recorder.list_records()).items
                detail = await recorder.get_record(record.record_id)
                assert detail is not None and not detail.capture_issues
                assert await _raw(recorder, record.record_id, "http_request") == request.content
                assert await _raw(recorder, record.record_id, "http_response") == b"".join(chunks)
        finally:
            await router.close()
            await recorder.close()
        assert transport.closed
    assert outcomes[0] == outcomes[1]
    assert request_bodies[0] == request_bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["proxied.example", "direct.example"])
@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
async def test_default_http_client_keeps_environment_proxy_routing(
    host: str,
    protocol: ProviderProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transports: dict[httpx.AsyncHTTPTransport, object] = {}
    selected_proxies: list[object] = []
    closed: set[httpx.AsyncHTTPTransport] = set()

    def initialize(transport: httpx.AsyncHTTPTransport, **kwargs: Any) -> None:
        transports[transport] = kwargs.get("proxy")

    async def handle(transport: httpx.AsyncHTTPTransport, _: httpx.Request) -> httpx.Response:
        selected_proxies.append(transports[transport])
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_FixedByteStream(_completion_body("proxy route preserved", protocol)),
        )

    async def close(transport: httpx.AsyncHTTPTransport) -> None:
        closed.add(transport)

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8765")
    monkeypatch.setenv("NO_PROXY", "direct.example")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", initialize)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "aclose", close)
    configuration = routed_configuration().models.providers["chat-provider"]
    provider_type = AnthropicProvider if protocol == "anthropic" else OpenAICompatibleProvider
    provider = provider_type(
        replace(configuration, protocol=protocol, base_url=f"https://{host}/v1")
    )
    try:
        response = await provider.complete(
            messages=[{"role": "user", "content": "proxy test"}],
            tools=(),
            model="chat-model",
            max_output=8192,
            temperature=0.1,
            timeout=10,
        )
        assert response.message.content == "proxy route preserved"
        assert len(selected_proxies) == 1
        if host == "proxied.example":
            assert isinstance(selected_proxies[0], httpx.Proxy)
            assert str(selected_proxies[0].url) == "http://proxy.example:8765"
        else:
            assert selected_proxies == [None]
    finally:
        await provider.close()
    assert closed == transports.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
@pytest.mark.parametrize("traced", [False, True])
async def test_http_retry_exhaustion_discards_all_request_and_response_candidates(
    tmp_path: Path,
    streaming: bool,
    protocol: ProviderProtocol,
    traced: bool,
) -> None:
    transport = _FakeAsyncTransport(
        lambda _: httpx.Response(
            429,
            headers={"content-type": "application/json"},
            stream=_FixedByteStream(b'{"error":{"message":"rate limited"}}'),
        )
    )
    router = _router(transport, clock=FakeClock(NOW), protocol=protocol)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3") if traced else NoOpTraceRecorder()
    try:
        with trace_scope(recorder, _source(tmp_path)):
            with pytest.raises(ModelCallError) as failure:
                if streaming:
                    _ = [event async for event in router.stream("title", messages=(), tools=())]
                else:
                    await router.complete("title", messages=(), tools=())
        assert failure.value.error.code == "provider_rate_limited"
        assert len(transport.requests) == 5
        if not isinstance(recorder, SQLiteTraceRecorder):
            return
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.retry_count == 4
        assert record.endpoint is not None
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert detail.request is None and detail.result is None
        assert detail.raw_sources == ()
        assert not detail.capture_issues
        assert await _raw(recorder, record.record_id, "http_response") == b""
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
@pytest.mark.parametrize("traced", [False, True])
async def test_http_stream_failure_preserves_partial_output_and_original_exception(
    tmp_path: Path,
    cancelled: bool,
    protocol: ProviderProtocol,
    traced: bool,
) -> None:
    original_error = asyncio.CancelledError() if cancelled else httpx.ReadError("broken stream")

    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            if protocol == "anthropic":
                yield _anthropic_stream_chunks()[0]
            else:
                yield (
                    b'data: {"id":"partial","object":"chat.completion.chunk","created":1,'
                    b'"model":"chat-model","choices":[{"index":0,"delta":{"content":"partial"},'
                    b'"finish_reason":null}]}\n\n'
                )
            raise original_error

    transport = _FakeAsyncTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=FailingStream(),
        )
    )
    router = _router(transport, protocol=protocol)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3") if traced else NoOpTraceRecorder()
    try:
        with trace_scope(recorder, _source(tmp_path)):
            events = router.stream("title", messages=(), tools=())
            assert await anext(events) == TextDelta("partial")
            expected_error = (
                asyncio.CancelledError
                if cancelled
                else httpx.ReadError
                if protocol == "anthropic"
                else ModelCallError
            )
            with pytest.raises(expected_error) as failure:
                await anext(events)
        if cancelled or protocol == "anthropic":
            assert failure.value is original_error
        else:
            assert failure.value.__cause__ is original_error
        assert len(transport.requests) == 1  # Output already emitted, so retry is forbidden.
        if not isinstance(recorder, SQLiteTraceRecorder):
            return
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.retry_count == 0
        if cancelled:
            assert record.error_category is TraceErrorCategory.CANCELLED
        detail = await recorder.get_record(record.record_id)
        assert detail is not None and detail.raw_sources == ()
        assert detail.request is None and detail.result is None
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
async def test_http_capture_failure_keeps_success_and_marks_missing_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: ProviderProtocol,
) -> None:
    transport = _FakeAsyncTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_FixedByteStream(_completion_body("capture failure isolated", protocol)),
        )
    )
    router = _router(transport, protocol=protocol)
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")

    def broken_capture(*args: object, **kwargs: object) -> None:
        raise OSError("capture unavailable")

    monkeypatch.setattr(recorder, "append_raw", broken_capture)
    try:
        with trace_scope(recorder, _source(tmp_path)):
            response = await router.complete("title", messages=(), tools=())
        assert response.message.content == "capture failure isolated"
        await recorder.flush()
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "success"
        assert record.capture_state.value == "partial"
        detail = await recorder.get_record(record.record_id)
        assert detail is not None and detail.capture_issues
        assert detail.raw_sources == ()
    finally:
        await router.close()
        await recorder.close()
