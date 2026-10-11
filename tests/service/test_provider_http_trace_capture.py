from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader, ProviderConfiguration
from aide.provider.openai_compatible import OpenAICompatibleProvider
from aide.service.runtime import AgentService
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TraceCaptureLayer, TracePurpose
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.service.test_service_concurrency import _claim_version, _CollectingSink


class _FixedByteStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _FakeAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_service_model_run_exposes_http_trace_through_recorder_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_chunks = (
        b'data: {"id":"service-run","object":"chat.completion.chunk","created":1,'
        b'"model":"small-model","choices":[{"index":0,"delta":{"content":"Service "},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"service-run","object":"chat.completion.chunk","created":1,'
        b'"model":"small-model","choices":[{"index":0,"delta":{"content":"answer"},'
        b'"finish_reason":null}]}\n\n'
        b'data: {"id":"service-run","object":"chat.completion.chunk","created":1,'
        b'"model":"small-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
        b"data: [DONE]\n\n",
    )
    response_body = (
        b'{"id":"service-title","object":"chat.completion","created":1,'
        b'"model":"small-model","choices":[{"index":0,"message":{"role":"assistant",'
        b'"content":"Service title"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}'
    )
    transports: list[_FakeAsyncTransport] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream") is True:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "x-request-id": "service-run"},
                stream=_FixedByteStream(*response_chunks),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "service-title"},
            stream=_FixedByteStream(response_body),
        )

    def transport_factory() -> _FakeAsyncTransport:
        transport = _FakeAsyncTransport(handle)
        transports.append(transport)
        return transport

    def provider_factory(configuration: ProviderConfiguration) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            configuration,
            http_transport_factory=transport_factory,
        )

    monkeypatch.setattr("aide.service.runtime.service.create_provider", provider_factory)

    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        session_id = await workspace.create_draft(client.client_id, creation_scope="chat")
        claim = await service.claim(client.client_id, workspace.workspace_id, session_id)
        await workspace.input(
            client.client_id,
            session_id,
            _claim_version(claim),
            "capture the wire response",
            "http-trace-run",
        )
        await asyncio.wait_for(sink.wait_for("run.completed", "http-trace-run"), 10)

        terminal = workspace._claims[session_id].loop.session.messages[-1]
        assert terminal["content"] == "Service answer"
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await recorder.flush()
        records = [
            record
            for record in (await recorder.list_records()).items
            if record.purpose is TracePurpose.AGENT_RUN and record.run_id == "http-trace-run"
        ]
        assert len(records) == 1
        record = records[0]
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert not detail.capture_issues

        request_raw = await recorder.read_raw(
            record.record_id,
            source="http_request",
            offset=0,
            limit_bytes=100_000,
        )
        response_raw = await recorder.read_raw(
            record.record_id,
            source="http_response",
            offset=0,
            limit_bytes=100_000,
        )
        request_metadata = await recorder.read_raw(
            record.record_id,
            source="http_request_metadata",
            offset=0,
            limit_bytes=100_000,
        )
        assert len(transports) == 1
        streamed_requests = [
            request
            for request in transports[0].requests
            if json.loads(request.content).get("stream") is True
        ]
        matching_requests = [request for request in streamed_requests if request.content == request_raw.data]
        assert len(streamed_requests) == 2
        assert len(matching_requests) == 1
        assert response_raw.data == b"".join(response_chunks)
        assert json.loads(request_metadata.data)["url"] == record.endpoint
        assert json.loads(request_metadata.data)["url"] == "https://models.example/v1/chat/completions"
        assert {
            (source.source, source.layer) for source in detail.raw_sources
        } >= {
            ("http_request", TraceCaptureLayer.HTTP_BODY),
            ("http_response", TraceCaptureLayer.HTTP_BODY),
            ("http_request_metadata", TraceCaptureLayer.HTTP_METADATA),
        }
    finally:
        await service.stop()
