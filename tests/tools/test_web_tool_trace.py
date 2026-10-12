from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio

from aide.agent.tools.core.web_fetch import HTTPResponseBoundary, WebFetchTool
from aide.agent.tools.core.web_search import WebSearchTool
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
)
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.scope import trace_scope
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import TraceCaptureLayer, TraceErrorCategory, TraceOrigin, TracePurpose
from tests.fixtures import SingleToolGateway
from tests.tools.core.test_web_fetch import FakeHTTPClient, FakeResolver, FakeResponse
from tests.tools.core.test_web_search import FakeDDGS


@pytest_asyncio.fixture
async def recorder(tmp_path: Path) -> AsyncIterator[SQLiteTraceRecorder]:
    instance = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def source(tmp_path: Path) -> TraceRootSource:
    return TraceRootSource(
        workspace_key=workspace_key_for(tmp_path),
        workspace_path=str(tmp_path),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="web-session",
    )


def _fetch_gateway(
    responses: tuple[FakeResponse, ...],
) -> tuple[SingleToolGateway, FakeHTTPClient]:
    http = FakeHTTPClient(tuple(cast(HTTPResponseBoundary, response) for response in responses))
    gateway = SingleToolGateway(
        (WebFetchTool(resolver=FakeResolver(("93.184.216.34",)), http_client=http),),
        permission_context=PermissionContext(level="workspace-write"),
    )
    return gateway, http


@pytest.mark.asyncio
async def test_fetch_metadata_encoding_failure_keeps_business_result(
    recorder: SQLiteTraceRecorder, source: TraceRootSource
) -> None:
    call = ModelToolCall("fetch", "web_fetch", '{"url":"https://public.example/page"}')
    results = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        response = FakeResponse(headers={"content-type": "text/plain; x=\udcff"}, chunks=(b"page",))
        gateway, http = _fetch_gateway((response,))
        with trace_scope(active_recorder, source):
            results.append(await gateway.call(call))
        assert response.iterated and response.closed
        assert len(http.calls) == 1
    assert results[0] == results[1]
    assert (results[1].status, results[1].content) == ("success", "page")
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    assert record.capture_state.value == "partial"
    raw = await recorder.read_raw(
        record.record_id, source="web_fetch_response", offset=0, limit_bytes=1000
    )
    assert raw.data == b"page"


@pytest.mark.asyncio
async def test_search_preserves_all_noncredential_fields_and_url_spelling(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = [
        {
            "title": "  Original\n title  ",
            "author": "Ada",
            "monkey": 1,
            "token_count": 17,
            "secretary": "Ada",
            "href": "HTTPS://public.example/a?x=%2f&tag=a%20b&flag&empty=#part",
            "body": " original\t snippet ",
            "metadata": {"rank": 2, "unknown": [None, True, 1.25]},
        },
        {"title": "second", "url": "https://second.example/path?", "extra": "retained"},
    ]
    monkeypatch.setattr("aide.agent.tools.core.web_search.DDGS", FakeDDGS)
    monkeypatch.setattr(FakeDDGS, "records", fixture)
    monkeypatch.setattr(FakeDDGS, "calls", [])
    monkeypatch.setattr(FakeDDGS, "failure", None)
    gateway = SingleToolGateway((WebSearchTool(),))
    call = ModelToolCall("search", "web_search", '{"query":"  trace  ","count":1}')
    with trace_scope(NoOpTraceRecorder(), source):
        baseline = await gateway.call(call)
    with trace_scope(recorder, source):
        result = await gateway.call(call)
    assert result == baseline
    assert FakeDDGS.calls == [("trace", {"max_results": 1, "backend": "duckduckgo"})] * 2
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    raw = await recorder.read_raw(
        record.record_id, source="web_search_ddgs", offset=0, limit_bytes=100_000
    )
    assert json.loads(raw.data) == fixture
    assert raw.layer is TraceCaptureLayer.SDK_RESULT
    assert record.capture_state.value == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", ["a%20b", "it's"])
async def test_search_excludes_credentials_from_every_saved_result_channel(
    recorder: SQLiteTraceRecorder,
    source: TraceRootSource,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
) -> None:
    url = f"https://user:password@public.example/a?x=%2f&tag={tag}&token=search-secret#part"
    safe_url = f"https://public.example/a?x=%2f&tag={tag}#part"
    fixture = [{"title": "result", "href": url, "body": "found", "api_key": "field-secret"}]
    monkeypatch.setattr("aide.agent.tools.core.web_search.DDGS", FakeDDGS)
    monkeypatch.setattr(FakeDDGS, "records", fixture)
    monkeypatch.setattr(FakeDDGS, "calls", [])
    monkeypatch.setattr(FakeDDGS, "failure", None)
    gateway = SingleToolGateway((WebSearchTool(),))
    call = ModelToolCall("search", "web_search", '{"query":"trace"}')
    with trace_scope(NoOpTraceRecorder(), source):
        baseline = await gateway.call(call)
    with trace_scope(recorder, source):
        result = await gateway.call(call)
    assert result == baseline
    assert url in result.content
    assert fixture[0]["href"] == url
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    detail = await recorder.get_record(record.record_id)
    assert detail is not None
    assert cast(dict[str, object], detail.result)["content"] == result.content.replace(
        url, safe_url
    )
    for raw_source in detail.raw_sources:
        raw = await recorder.read_raw(
            record.record_id, source=raw_source.source, offset=0, limit_bytes=100_000
        )
        assert b"search-secret" not in raw.data
        assert b"field-secret" not in raw.data
        assert b"user:password" not in raw.data


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["text", "markdown"])
async def test_fetch_raw_survives_conversion_truncation_and_multiple_chunks(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, output_format: str
) -> None:
    chunks = (
        b"<html><body><script>hidden</script>",
        b"<p>readable</p>" * 10_000,
        b"</body></html>",
    )
    call = ModelToolCall(
        "fetch",
        "web_fetch",
        json.dumps({"url": "https://public.example/page", "format": output_format, "maxChars": 20}),
    )
    results = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        response = FakeResponse(headers={"content-type": "text/html"}, chunks=chunks)
        gateway, http = _fetch_gateway((response,))
        with trace_scope(active_recorder, source):
            results.append(await gateway.call(call))
        assert len(http.calls) == 1
        assert response.closed
    assert results[0] == results[1]
    assert len(results[1].content) == 20
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    first = await recorder.read_raw(
        record.record_id, source="web_fetch_response", offset=0, limit_bytes=80_000
    )
    rest = await recorder.read_raw(
        record.record_id, source="web_fetch_response", offset=80_000, limit_bytes=100_000
    )
    assert first.data + rest.data == b"".join(chunks)
    assert first.layer is TraceCaptureLayer.HTTP_DECODED_BODY
    assert record.capture_state.value == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["redirect", "unsupported"])
async def test_fetch_does_not_consume_skipped_response_bodies(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, case: str
) -> None:
    results = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        skipped = FakeResponse(
            status_code=302 if case == "redirect" else 415,
            headers=(
                {"location": "/final"}
                if case == "redirect"
                else {"content-type": "application/octet-stream"}
            ),
            chunks=(b"unconsumed",),
        )
        final = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"final",))
        gateway, http = _fetch_gateway((skipped, final) if case == "redirect" else (skipped,))
        with trace_scope(active_recorder, source):
            results.append(
                await gateway.call(
                    ModelToolCall("fetch", "web_fetch", '{"url":"https://public.example/start"}')
                )
            )
        assert not skipped.iterated
        assert skipped.closed
        assert len(http.calls) == (2 if case == "redirect" else 1)
    assert results[0] == results[1]
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    detail = await recorder.get_record(record.record_id)
    assert detail is not None
    if case == "redirect":
        raw = await recorder.read_raw(
            record.record_id, source="web_fetch_response", offset=0, limit_bytes=1000
        )
        assert raw.data == b"final"
    else:
        assert all(raw.source != "web_fetch_response" for raw in detail.raw_sources)
        metadata = await recorder.read_raw(
            record.record_id, source="web_fetch_response_metadata", offset=0, limit_bytes=1000
        )
        assert json.loads(metadata.data)["endpoint"] == "https://public.example/start"
        assert json.loads(metadata.data)["status_code"] == 415


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_fetch_retains_consumed_bytes_when_interrupted(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, failure: type[BaseException]
) -> None:
    class InterruptedResponse(FakeResponse):
        async def iter_bytes(self) -> AsyncIterator[bytes]:
            yield b"already consumed\xff"
            raise failure()

    results = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        response = InterruptedResponse(headers={"content-type": "text/plain"})
        gateway, http = _fetch_gateway((response,))
        with trace_scope(active_recorder, source):
            if failure is asyncio.CancelledError:
                with pytest.raises(asyncio.CancelledError):
                    await gateway.call(
                        ModelToolCall("fetch", "web_fetch", '{"url":"https://public.example/page"}')
                    )
            else:
                results.append(
                    await gateway.call(
                        ModelToolCall("fetch", "web_fetch", '{"url":"https://public.example/page"}')
                    )
                )
        assert response.closed
        assert len(http.calls) == 1
    if results:
        assert results[0] == results[1]
        assert results[1].content == "Web Fetch timed out after 30 seconds."
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    assert record.status.value == "failed"
    assert record.error_category is (
        TraceErrorCategory.TIMEOUT if failure is TimeoutError else TraceErrorCategory.CANCELLED
    )
    raw = await recorder.read_raw(
        record.record_id, source="web_fetch_response", offset=0, limit_bytes=1000
    )
    assert raw.data == b"already consumed\xff"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
async def test_search_failures_keep_noop_behavior(
    recorder: SQLiteTraceRecorder,
    source: TraceRootSource,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    monkeypatch.setattr("aide.agent.tools.core.web_search.DDGS", FakeDDGS)
    monkeypatch.setattr(FakeDDGS, "calls", [])
    monkeypatch.setattr(FakeDDGS, "failure", failure("upstream"))
    call = ModelToolCall("search", "web_search", '{"query":"trace"}')
    gateway = SingleToolGateway((WebSearchTool(),))
    results = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        with trace_scope(active_recorder, source):
            if failure is asyncio.CancelledError:
                with pytest.raises(asyncio.CancelledError):
                    await gateway.call(call)
            else:
                results.append(await gateway.call(call))
    if results:
        assert results[0] == results[1]
        assert results[1].status == "error"
    assert len(FakeDDGS.calls) == 2
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    assert record.status.value == "failed"
    assert record.error_category is {
        TimeoutError: TraceErrorCategory.TIMEOUT,
        asyncio.CancelledError: TraceErrorCategory.CANCELLED,
        RuntimeError: TraceErrorCategory.TOOL_ERROR,
    }[failure]


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approved", "declined"])
async def test_fetch_redirect_authorization_matches_noop_and_hides_credentials(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, decision: ConfirmationDecision
) -> None:
    results = []
    approvals = []
    for active_recorder in (NoOpTraceRecorder(), recorder):
        requests: list[ConfirmationRequest] = []

        async def confirm(
            request: ConfirmationRequest, *, request_log: list[ConfirmationRequest] = requests
        ) -> ConfirmationDecision:
            request_log.append(request)
            return decision

        redirect = FakeResponse(
            status_code=302,
            headers={"location": "http://private.example/admin?token=redirect-secret&view=full"},
        )
        final = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"private",))
        http = FakeHTTPClient(
            (cast(HTTPResponseBoundary, redirect), cast(HTTPResponseBoundary, final))
        )
        gateway = SingleToolGateway(
            (
                WebFetchTool(
                    resolver=FakeResolver(
                        {"public.example": ("93.184.216.34",), "private.example": ("10.0.0.7",)}
                    ),
                    http_client=http,
                ),
            ),
            confirmation=confirm,
            permission_context=PermissionContext(level="workspace-write"),
        )
        with trace_scope(active_recorder, source):
            result = await gateway.call(
                ModelToolCall("fetch", "web_fetch", '{"url":"https://public.example/start"}')
            )
        results.append((result.status, result.content))
        approvals.append(
            [(request.reason, request.summary, request.details) for request in requests]
        )
        assert len(http.calls) == (2 if decision == "approved" else 1)
        assert not redirect.iterated
        assert final.iterated is (decision == "approved")
    assert results[0] == results[1]
    assert approvals[0] == approvals[1]
    assert len(approvals[1]) == 1
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    detail = await recorder.get_record(record.record_id)
    assert detail is not None
    assert "redirect-secret" not in json.dumps(detail.request)
    assert "redirect-secret" not in json.dumps(detail.result)


@pytest.mark.asyncio
async def test_search_unserializable_extra_field_does_not_change_result(
    recorder: SQLiteTraceRecorder, source: TraceRootSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aide.agent.tools.core.web_search.DDGS", FakeDDGS)
    monkeypatch.setattr(FakeDDGS, "records", [{"title": "result", "extra": object()}])
    monkeypatch.setattr(FakeDDGS, "calls", [])
    monkeypatch.setattr(FakeDDGS, "failure", None)
    gateway = SingleToolGateway((WebSearchTool(),))
    call = ModelToolCall("search", "web_search", '{"query":"trace"}')
    with trace_scope(NoOpTraceRecorder(), source):
        baseline = await gateway.call(call)
    with trace_scope(recorder, source):
        result = await gateway.call(call)
    assert result == baseline
    assert result.status == "success"
    await recorder.flush()
    (record,) = (await recorder.list_records()).items
    assert record.capture_state.value == "partial"
    detail = await recorder.get_record(record.record_id)
    assert detail is not None
    assert any(issue.source == "web_search_ddgs" for issue in detail.capture_issues)
