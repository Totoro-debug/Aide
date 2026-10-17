from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader, ProviderConfiguration
from aide.provider.anthropic import AnthropicProvider
from aide.provider.models import ModelProvider
from aide.provider.openai_compatible import OpenAICompatibleProvider
from aide.service.runtime import AgentService
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TracePurpose
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.trace import wait_for_trace_writer
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
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    async def aclose(self) -> None:
        self.closed = True


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
    service = AgentService(home, ConfigLoader(home).load())
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
        await wait_for_trace_writer(recorder)
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

        assert len(transports) == 1
        streamed_requests = [
            request
            for request in transports[0].requests
            if json.loads(request.content).get("stream") is True
        ]
        assert len(streamed_requests) == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_anthropic_service_run_exposes_sdk_wire_trace_through_recorder_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_chunks = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-run",'
        b'"type":"message","role":"assistant","model":"small-model","content":[],'
        b'"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":0}}}\n\n',
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"Anthropic answer"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
        b'"stop_sequence":null},"usage":{"output_tokens":2}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
    response_body = (
        b'{"id":"msg-title","type":"message","role":"assistant","model":"small-model",'
        b'"content":[{"type":"text","text":"Anthropic title"}],"stop_reason":"end_turn",'
        b'"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":2}}'
    )
    openai_response_chunks = (
        b'data: {"id":"openai-run","object":"chat.completion.chunk","created":1,'
        b'"model":"openai-model","choices":[{"index":0,"delta":{"content":"OpenAI answer"},'
        b'"finish_reason":null}]}\n\n',
        b'data: {"id":"openai-run","object":"chat.completion.chunk","created":1,'
        b'"model":"openai-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n',
        b"data: [DONE]\n\n",
    )
    openai_response_body = (
        b'{"id":"openai-title","object":"chat.completion","created":1,'
        b'"model":"openai-model","choices":[{"index":0,"message":{"role":"assistant",'
        b'"content":"OpenAI title"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}'
    )
    transports: dict[str, list[_FakeAsyncTransport]] = {"anthropic": [], "openai": []}
    anthropic_retry_sent = False
    streams_started = {"anthropic": asyncio.Event(), "openai": asyncio.Event()}

    class ConcurrentResponseStream(_FixedByteStream):
        def __init__(self, *chunks: bytes, provider: str) -> None:
            super().__init__(*chunks)
            self.provider = provider

        async def __aiter__(self) -> AsyncIterator[bytes]:
            streams_started[self.provider].set()
            peer = "openai" if self.provider == "anthropic" else "anthropic"
            await asyncio.wait_for(streams_started[peer].wait(), 5)
            async for chunk in super().__aiter__():
                await asyncio.sleep(0)
                yield chunk

    def handle_anthropic(request: httpx.Request) -> httpx.Response:
        nonlocal anthropic_retry_sent
        body = json.loads(request.content)
        if body.get("stream") is True:
            if (
                not anthropic_retry_sent
                and body.get("tools")
                and "capture the Anthropic wire response" in request.content.decode("utf-8")
            ):
                anthropic_retry_sent = True
                return httpx.Response(
                    503,
                    headers={"content-type": "application/json"},
                    json={
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "retry fixture"},
                    },
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "request-id": "msg-run"},
                stream=(
                    ConcurrentResponseStream(*response_chunks, provider="anthropic")
                    if body.get("tools")
                    else _FixedByteStream(*response_chunks)
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "request-id": "msg-title"},
            stream=_FixedByteStream(response_body),
        )

    def handle_openai(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "x-request-id": "openai-run"},
                stream=(
                    ConcurrentResponseStream(*openai_response_chunks, provider="openai")
                    if body.get("tools")
                    else _FixedByteStream(*openai_response_chunks)
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "openai-title"},
            stream=_FixedByteStream(openai_response_body),
        )

    def anthropic_transport_factory() -> _FakeAsyncTransport:
        transport = _FakeAsyncTransport(handle_anthropic)
        transports["anthropic"].append(transport)
        return transport

    def openai_transport_factory() -> _FakeAsyncTransport:
        transport = _FakeAsyncTransport(handle_openai)
        transports["openai"].append(transport)
        return transport

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        if configuration.protocol == "anthropic":
            return AnthropicProvider(
                configuration,
                http_transport_factory=anthropic_transport_factory,
            )
        return OpenAICompatibleProvider(
            configuration,
            http_transport_factory=openai_transport_factory,
        )

    monkeypatch.setattr("aide.service.runtime.service.create_provider", provider_factory)

    home = AgentHome(tmp_path / "home")
    home.initialize()
    config = MINIMAL_VALID_CONFIG.replace(
        'protocol = "openai-compatible"',
        'protocol = "anthropic"',
    ).replace('base_url = "https://models.example/v1"', 'base_url = "https://models.example"')
    config += """

[models.providers.secondary]
protocol = "openai-compatible"
base_url = "https://openai.models.example/v1"
api_key = "openai-secret"

[models.providers.secondary.models."openai-model"]
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30
"""
    (home.path / "config.toml").write_text(config, encoding="utf-8")
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    try:
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        openai_client = await service.register_client("cli")
        openai_sink = _CollectingSink()
        await service.connect_client(openai_client.client_id, openai_sink)
        workspace = await service.attach_workspace(client.client_id, workspace_path)
        openai_workspace = await service.attach_workspace(
            openai_client.client_id,
            workspace_path,
        )
        assert openai_workspace.workspace_id == workspace.workspace_id
        anthropic_session_id = await workspace.create_draft(
            client.client_id,
            creation_scope="chat",
        )
        openai_session_id = await openai_workspace.create_draft(
            openai_client.client_id,
            creation_scope="chat",
        )
        anthropic_claim = await service.claim(
            client.client_id,
            workspace.workspace_id,
            anthropic_session_id,
        )
        openai_claim = await service.claim(
            openai_client.client_id,
            workspace.workspace_id,
            openai_session_id,
        )
        await service.configure_conversation_model(
            openai_client.client_id,
            workspace.workspace_id,
            openai_session_id,
            _claim_version(openai_claim),
            {
                "provider_id": "secondary",
                "model": "openai-model",
                "reasoning_effort": "mid",
                "expected_model_configuration_version": 0,
            },
        )
        await asyncio.gather(
            workspace.input(
                client.client_id,
                anthropic_session_id,
                _claim_version(anthropic_claim),
                "capture the Anthropic wire response",
                "anthropic-http-trace-run",
            ),
            workspace.input(
                openai_client.client_id,
                openai_session_id,
                _claim_version(openai_claim),
                "capture the OpenAI wire response",
                "openai-http-trace-run",
            ),
        )
        await asyncio.gather(
            asyncio.wait_for(sink.wait_for("run.completed", "anthropic-http-trace-run"), 10),
            asyncio.wait_for(
                openai_sink.wait_for("run.completed", "openai-http-trace-run"),
                10,
            ),
        )

        anthropic_terminal = workspace._claims[anthropic_session_id].loop.session.messages[-1]
        openai_terminal = workspace._claims[openai_session_id].loop.session.messages[-1]
        assert anthropic_terminal["content"] == "Anthropic answer"
        assert openai_terminal["content"] == "OpenAI answer"
        assert all(started.is_set() for started in streams_started.values())
        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        all_records = (await recorder.list_records(limit=100)).items
        anthropic_retry_counts: list[int] = []

        async def assert_session_trace(
            *,
            run_id: str,
            session_id: str,
            expected_model: str,
            expected_endpoint: str,
            expected_prompt: str,
            expected_transports: list[_FakeAsyncTransport],
            expected_stream: bytes,
            expected_complete: bytes,
            anthropic: bool,
        ) -> list[dict[str, Any]]:
            records = [
                record
                for record in all_records
                if record.run_id == run_id and record.endpoint == expected_endpoint
            ]
            assert records
            payloads: list[dict[str, Any]] = []
            for record in records:
                assert record.subject_session_id == session_id
                detail = await recorder.get_record(record.record_id)
                assert detail is not None
                assert not detail.capture_issues

                assert detail.request is None
                assert record.protocol == ("anthropic" if anthropic else "openai-compatible")
                assert record.model == expected_model
                assert detail.result is not None
                if anthropic:
                    anthropic_retry_counts.append(record.retry_count)
                assert record.endpoint == expected_endpoint

            return payloads

        await assert_session_trace(
            run_id="anthropic-http-trace-run",
            session_id=anthropic_session_id,
            expected_model="small-model",
            expected_endpoint="https://models.example/v1/messages",
            expected_prompt="capture the Anthropic wire response",
            expected_transports=transports["anthropic"],
            expected_stream=b"".join(response_chunks),
            expected_complete=response_body,
            anthropic=True,
        )
        await assert_session_trace(
            run_id="openai-http-trace-run",
            session_id=openai_session_id,
            expected_model="openai-model",
            expected_endpoint="https://openai.models.example/v1/chat/completions",
            expected_prompt="capture the OpenAI wire response",
            expected_transports=transports["openai"],
            expected_stream=b"".join(openai_response_chunks),
            expected_complete=openai_response_body,
            anthropic=False,
        )
        assert 1 in anthropic_retry_counts
    finally:
        await service.stop()

    assert transports["anthropic"] and transports["openai"]
    assert all(
        transport.closed
        for provider_transports in transports.values()
        for transport in provider_transports
    )
