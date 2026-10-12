from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder
from aide.trace.scope import trace_scope
from aide.trace.sqlite_recorder import SQLiteTraceRecorder
from aide.trace.types import (
    TraceErrorCategory,
    TraceOrigin,
    TracePurpose,
    TraceTokenUsage,
    TraceUsageState,
)
from aide.utils.errors import ErrorInfo
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.fixtures.model_configuration import configuration, routed_configuration
from tests.fixtures.trace import wait_for_trace_writer

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _retryable_timeout() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="The provider timed out.",
            retryable=True,
        )
    )


def _permanent_auth_failure() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="provider_auth_error",
            message="The configured route is permanently unavailable.",
        )
    )


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=510, output_tokens=4, total_tokens=514,
                         cached_input_tokens=200, cache_creation_input_tokens=300),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_model_router_publishes_a_completed_call_through_trace_queries(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(completions=(_response("Traced response"),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        jitter=None,
    )
    source = TraceRootSource(
        workspace_key=workspace_key_for(workspace),
        workspace_path=str(workspace),
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="trace-session",
        subject_session_id="trace-session",
        run_id="trace-run",
        restore_run_token="trace-restore-token",
    )

    try:
        with trace_scope(recorder, source):
            actual = await router.complete(
                "chat",
                messages=[{"role": "user", "content": "trace this call"}],
                tools=(),
            )
        await wait_for_trace_writer(recorder)

        page = await recorder.list_records()
        assert actual == _response("Traced response")
        assert len(page.items) == 1
        record = page.items[0]
        assert record.origin is TraceOrigin.FOREGROUND
        assert record.purpose is TracePurpose.AGENT_RUN
        assert record.owner_session_id == "trace-session"
        assert record.run_id == "trace-run"
        assert record.restore_run_token == "trace-restore-token"
        assert record.status.value == "success"
        assert record.retry_count == 0
        assert record.usage.state is TraceUsageState.REPORTED
        assert record.usage.cache_creation_input_tokens == 300
        assert (await recorder.get_stats()).tokens.cache_creation_input_tokens == 300
        detail = await recorder.get_record(record.record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.result)["message"]["content"] == "Traced response"
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("success_attempt", "retry_count"), [(1, 0), (3, 2), (5, 4)])
async def test_router_publishes_only_the_final_success_after_retries(
    tmp_path: Path,
    success_attempt: int,
    retry_count: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(
        completions=(
            *(_retryable_timeout() for _ in range(success_attempt - 1)),
            _response("final result"),
        )
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            actual = await router.complete(
                "chat",
                messages=[{"role": "user", "content": "retry this call"}],
                tools=(),
            )
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert actual == _response("final result")
        assert len(records) == 1
        assert records[0].retry_count == retry_count
        assert records[0].status.value == "success"
        assert records[0].usage.cache_creation_input_tokens == 300
        assert (await recorder.get_stats()).tokens.cache_creation_input_tokens == 300
        detail = await recorder.get_record(records[0].record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.request)["messages"][0]["content"] == "retry this call"
        assert cast(dict[str, Any], detail.result)["message"]["content"] == "final result"
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_router_publishes_one_failure_without_failed_attempt_payloads(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(completions=(_retryable_timeout() for _ in range(5)))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            with pytest.raises(ModelCallError):
                await router.complete(
                    "chat",
                    messages=[{"role": "user", "content": "do not retain this"}],
                    tools=(),
                )
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert len(records) == 1
        assert records[0].retry_count == 4
        assert records[0].status.value == "failed"
        assert records[0].error_category is TraceErrorCategory.TIMEOUT
        detail = await recorder.get_record(records[0].record_id)
        assert detail is not None
        assert detail.request is None
        assert detail.result is None
        assert "do not retain this" not in str(detail)
        assert any(issue.source == "provider_transport" for issue in detail.capture_issues)
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_router_records_fallback_metadata_from_the_successful_provider(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    primary = ScriptedFakeProvider(completions=(_permanent_auth_failure(),))
    fallback = ScriptedFakeProvider(completions=(_response("fallback result"),))
    providers = {"chat-provider": primary, "default-provider": fallback}
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            actual = await router.complete(
                "title",
                messages=[{"role": "user", "content": "fallback"}],
                tools=(),
                continuation=ModelContinuation("chat-provider", {"state": "primary only"}),
            )
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert actual == _response("fallback result")
        assert len(records) == 1
        assert records[0].retry_count == 1
        assert records[0].usage.cache_creation_input_tokens == 300
        assert (await recorder.get_stats()).tokens.cache_creation_input_tokens == 300
        assert records[0].model == "default-model"
        assert records[0].endpoint is None  # The fake Provider sends no HTTP request.
        assert records[0].reasoning_effort is not None
        assert records[0].reasoning_effort == "mid"
        detail = await recorder.get_record(records[0].record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.request)["continuation"] is None
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_router_records_stream_retry_and_preserves_event_order(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=_retryable_timeout()),
            StreamScript(events=(TextDelta(delta="answer"), ModelCompleted(_response("answer")))),
        )
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            events = [
                event
                async for event in router.stream(
                    "chat",
                    messages=[{"role": "user", "content": "stream"}],
                    tools=(),
                )
            ]
        await wait_for_trace_writer(recorder)

        assert events == [TextDelta(delta="answer"), ModelCompleted(_response("answer"))]
        records = (await recorder.list_records()).items
        assert len(records) == 1
        assert records[0].retry_count == 1
        detail = await recorder.get_record(records[0].record_id)
        assert detail is not None
        assert cast(dict[str, Any], detail.result)["message"]["content"] == "answer"
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_stream_cancellation_after_partial_output_is_failed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(TextDelta(delta="partial"), ModelCompleted(_response("done")))),
        )
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    stream = None

    try:
        with trace_scope(recorder, _source(workspace)):
            stream = cast(
                AsyncGenerator[ModelStreamEvent],
                router.stream(
                    "chat",
                    messages=[{"role": "user", "content": "cancel"}],
                    tools=(),
                ),
            )
            assert await anext(stream) == TextDelta(delta="partial")
            with pytest.raises(asyncio.CancelledError):
                await stream.athrow(asyncio.CancelledError())
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert len(records) == 1
        assert records[0].status.value == "failed"
        assert records[0].error_category is TraceErrorCategory.CANCELLED
        detail = await recorder.get_record(records[0].record_id)
        assert detail is not None
        assert detail.request is None
        assert detail.result is None
    finally:
        if stream is not None:
            await stream.aclose()
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_stream_closed_before_first_iteration_does_not_leave_an_open_trace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(TextDelta(delta="unused"),)),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            events = cast(
                AsyncGenerator[ModelStreamEvent],
                router.stream(
                    "chat",
                    messages=[{"role": "user", "content": "do not start"}],
                    tools=(),
                ),
            )
            await events.aclose()
        await wait_for_trace_writer(recorder)

        assert (await recorder.list_records()).items == ()
        assert provider.stream_requests == []
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_guard_rejection_is_failed_without_an_endpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(completions=(_response("unused"),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    try:
        with trace_scope(recorder, _source(workspace)):
            with pytest.raises(ModelCallError):
                await router.complete(
                    "chat",
                    messages=[{"role": "user", "content": "too large"}],
                    tools=(),
                    guard=lambda _status, _messages, _tools: False,
                )
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert len(records) == 1
        assert records[0].status.value == "failed"
        assert records[0].endpoint is None
        assert records[0].error_category is TraceErrorCategory.VALIDATION_ERROR
        assert provider.complete_requests == []
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_shared_router_keeps_concurrent_session_scopes_separate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(completions=(_response("one"), _response("two")))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
    )

    async def run(session_id: str) -> None:
        with trace_scope(recorder, _source(workspace, session_id=session_id)):
            await router.complete(
                "chat",
                messages=[{"role": "user", "content": session_id}],
                tools=(),
            )

    try:
        await asyncio.gather(run("session-one"), run("session-two"))
        await wait_for_trace_writer(recorder)

        records = (await recorder.list_records()).items
        assert {record.owner_session_id for record in records} == {
            "session-one",
            "session-two",
        }
        assert len({record.root_id for record in records}) == 2
    finally:
        await router.close()
        await recorder.close()


def _source(workspace: Path, *, session_id: str = "trace-session") -> TraceRootSource:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("recording", [False, True])
async def test_response_snapshot_failure_preserves_business_success(
    tmp_path: Path,
    streaming: bool,
    recording: bool,
) -> None:
    # Provider-owned opaque continuation state is not required to support deepcopy.
    response = replace(
        _response("valid answer"),
        continuation=ModelContinuation("provider", iter(value for value in (1, 2))),
    )
    provider = ScriptedFakeProvider(
        completions=(response,),
        streams=(StreamScript((ModelCompleted(response),)),),
    )
    recorder = (
        SQLiteTraceRecorder(tmp_path / "traces.sqlite3") if recording else NoOpTraceRecorder()
    )
    router = ModelRouter(configuration=configuration(), provider_factory=lambda _: provider)
    try:
        with trace_scope(recorder, _source(tmp_path)):
            if streaming:
                actual = [event async for event in router.stream("chat", messages=(), tools=())]
                assert actual == [ModelCompleted(response)]
            else:
                assert await router.complete("chat", messages=(), tools=()) is response
        if isinstance(recorder, SQLiteTraceRecorder):
            await wait_for_trace_writer(recorder)
            (record,) = (await recorder.list_records()).items
            assert record.status.value == "success"
            detail = await recorder.get_record(record.record_id)
            assert detail is not None
            assert any(issue.source == "model_router" for issue in detail.capture_issues)
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_run_router_close_records_cancellation_immediately(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    provider = ScriptedFakeProvider(streams=(StreamScript((TextDelta("partial"),)),))
    router = ModelRouter(configuration=configuration(), provider_factory=lambda _: provider)
    try:
        with trace_scope(recorder, _source(tmp_path)):
            events = cast(
                AsyncGenerator[ModelStreamEvent],
                router.for_run().stream(
                    "chat",
                    messages=(),
                    tools=(),
                ),
            )
            assert await anext(events) == TextDelta("partial")
            await events.aclose()
        await wait_for_trace_writer(recorder)
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.error_category is TraceErrorCategory.CANCELLED
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
async def test_unreported_usage_remains_unknown_in_trace(tmp_path: Path) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    response = replace(_response("answer"), usage=ModelUsage(0, 0, 0, reported=False))
    provider = ScriptedFakeProvider(completions=(response,))
    router = ModelRouter(configuration=configuration(), provider_factory=lambda _: provider)
    try:
        with trace_scope(recorder, _source(tmp_path)):
            assert await router.complete("chat", messages=(), tools=()) is response
        await wait_for_trace_writer(recorder)
        (record,) = (await recorder.list_records()).items
        assert record.usage == TraceTokenUsage(TraceUsageState.UNKNOWN)
        assert record.endpoint is None
    finally:
        await router.close()
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_response", [False, True])
async def test_incomplete_and_cancelled_streams_never_publish_success(
    tmp_path: Path,
    cancelled_response: bool,
) -> None:
    recorder = SQLiteTraceRecorder(tmp_path / "traces.sqlite3")
    events = (
        (ModelCompleted(replace(_response("partial"), finish_reason="cancelled")),)
        if cancelled_response
        else (TextDelta("partial"),)
    )
    provider = ScriptedFakeProvider(streams=(StreamScript(events),))
    router = ModelRouter(configuration=configuration(), provider_factory=lambda _: provider)
    try:
        with trace_scope(recorder, _source(tmp_path)):
            actual = [event async for event in router.stream("chat", messages=(), tools=())]
        assert actual == list(events)
        await wait_for_trace_writer(recorder)
        (record,) = (await recorder.list_records()).items
        assert record.status.value == "failed"
        assert record.error_category is (
            TraceErrorCategory.CANCELLED
            if cancelled_response
            else TraceErrorCategory.PROVIDER_ERROR
        )
    finally:
        await router.close()
        await recorder.close()
