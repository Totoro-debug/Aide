"""Behavioral contract for execution trace types and the NoOp recorder."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder, TraceRecorder
from aide.trace.types import (
    TraceAttemptMetadata,
    TraceCallMetadata,
    TraceCaptureIssue,
    TraceCaptureState,
    TraceErrorCategory,
    TraceKind,
    TraceOrigin,
    TraceOutcome,
    TracePurpose,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from tests.fixtures.trace import (
    TRACE_PURPOSE_COVERAGE,
    TRACE_SCENARIO_NAMES,
    TRACE_TOOL_COVERAGE,
    trace_contract_fixtures,
)

_NOW = datetime(2026, 10, 11, 1, 2, 3, tzinfo=UTC)


def test_trace_enums_pin_status_purpose_and_failure_vocabulary() -> None:
    assert {item.value for item in TraceStatus} == {"success", "failed"}
    assert {item.value for item in TraceOrigin} == {
        "foreground",
        "subagent",
        "schedule",
        "auxiliary",
    }
    assert {item.value for item in TracePurpose} == set(TRACE_PURPOSE_COVERAGE)
    assert {item.value for item in TraceKind} == {"model", "tool"}
    assert {item.value for item in TraceUsageState} == {
        "reported",
        "unknown",
        "not_applicable",
    }
    assert {item.value for item in TraceErrorCategory} >= {
        "cancelled",
        "permission_denied",
        "timeout",
        "provider_error",
        "tool_error",
    }


def test_trace_root_context_is_immutable_and_allows_no_session() -> None:
    recorder = NoOpTraceRecorder()
    source = TraceRootSource(
        workspace_key="workspace-sha256",
        workspace_path="D:/work/project",
        origin=TraceOrigin.AUXILIARY,
        purpose=TracePurpose.DREAM,
    )

    context = recorder.new_root(source)

    assert context.root_id
    assert context.workspace_key == source.workspace_key
    assert context.owner_session_id is None
    assert context.subject_session_id is None
    assert context.purpose is TracePurpose.DREAM
    with pytest.raises(FrozenInstanceError):
        context.root_id = "changed"  # type: ignore[misc]


def test_workspace_key_is_stable_for_windows_case_and_lexical_variants() -> None:
    first = workspace_key_for(Path("D:/work/project"))
    changed_case = workspace_key_for(Path("d:/WORK/PROJECT"))
    normalized = workspace_key_for(Path("D:/work/temporary/../project"))

    assert first.startswith("sha256:")
    assert first == changed_case == normalized


def test_trace_context_preserves_schedule_and_nested_agent_identity() -> None:
    context = NoOpTraceRecorder().new_root(
        TraceRootSource(
            workspace_key="sha256:workspace",
            workspace_path="D:/work/project",
            origin=TraceOrigin.SUBAGENT,
            purpose=TracePurpose.SUBAGENT_RUN,
            owner_session_id="owner-session",
            subject_session_id="subagent-session",
            run_id="service-run",
            restore_run_token="restore-token",
            agent_id="agent-1",
            parent_root_id="parent-root",
            schedule_run_id="schedule-run",
        )
    )

    assert context.run_id == "service-run"
    assert context.restore_run_token == "restore-token"
    assert context.agent_id == "agent-1"
    assert context.parent_root_id == "parent-root"
    assert context.schedule_run_id == "schedule-run"
    assert context.owner_session_id == "owner-session"
    assert context.subject_session_id == "subagent-session"


def test_trace_values_reject_non_utc_times_and_ambiguous_token_counts() -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        TraceCallMetadata(
            kind=TraceKind.MODEL,
            purpose=TracePurpose.AGENT_RUN,
            started_at=datetime(2026, 10, 11, 1, 2, 3),
        )
    with pytest.raises(ValueError, match="Unknown or inapplicable"):
        TraceTokenUsage(state=TraceUsageState.UNKNOWN, input_tokens=0)
    with pytest.raises(ValueError, match="non-negative integers"):
        TraceTokenUsage(state=TraceUsageState.REPORTED, input_tokens=True)
    with pytest.raises(ValueError, match="require an error category and message"):
        TraceOutcome(status=TraceStatus.FAILED, finished_at=_NOW, duration_ms=0)
    with pytest.raises(ValueError, match="incomplete state and a message"):
        TraceCaptureIssue(state=TraceCaptureState.COMPLETE, message="Missing response")


def test_noop_recorder_keeps_business_failure_and_cancellation_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = NoOpTraceRecorder()
    context = recorder.new_root(
        TraceRootSource(
            workspace_key="workspace-sha256",
            workspace_path=str(tmp_path),
            origin=TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id="owner-session",
            subject_session_id="owner-session",
            run_id="service-run",
            restore_run_token="restore-token",
        )
    )
    metadata = TraceCallMetadata(
        kind=TraceKind.MODEL,
        purpose=TracePurpose.AGENT_RUN,
        started_at=_NOW,
        model="provider/model",
        reasoning_effort="high",
        endpoint="https://provider.invalid/v1",
    )
    handle = recorder.begin_call(context, metadata)
    attempt = recorder.begin_attempt(
        handle,
        TraceAttemptMetadata(
            started_at=_NOW,
            model="provider/fallback",
            reasoning_effort="max",
            endpoint="https://fallback.invalid/v1",
        ),
    )
    request = {"messages": [{"role": "user", "content": "Original request"}]}
    recorder.set_request(attempt, request)
    recorder.report_capture_issue(
        attempt,
        TraceCaptureIssue(
            state=TraceCaptureState.PARTIAL,
            message="The response capture was interrupted.",
            source="provider_sse",
        ),
    )

    for error in (RuntimeError("business failure"), asyncio.CancelledError("cancelled by test")):
        outcome = TraceOutcome(
            status=TraceStatus.FAILED,
            finished_at=_NOW,
            duration_ms=1,
            error_category=(
                TraceErrorCategory.CANCELLED
                if isinstance(error, asyncio.CancelledError)
                else TraceErrorCategory.PROVIDER_ERROR
            ),
            error_code="cancelled"
            if isinstance(error, asyncio.CancelledError)
            else "provider_error",
            error_message=str(error),
        )
        with pytest.raises(type(error)) as raised:
            try:
                raise error
            except BaseException:
                recorder.finish_call(
                    handle,
                    outcome,
                    result=None,
                    usage=TraceTokenUsage(state=TraceUsageState.UNKNOWN),
                    retry_count=0,
                )
                raise
        assert raised.value is error

    assert request == {"messages": [{"role": "user", "content": "Original request"}]}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_noop_recorder_exposes_empty_queries_and_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recorder: TraceRecorder = NoOpTraceRecorder()
    source = TraceRootSource(
        workspace_key="workspace-sha256",
        workspace_path="D:/work/project",
        origin=TraceOrigin.FOREGROUND,
        purpose=TracePurpose.AGENT_RUN,
        owner_session_id="session-1",
    )
    context = recorder.new_root(source)

    recorder.close_session_scope(context.workspace_key, "session-1")
    await recorder.purge_session(context.workspace_key, "session-1")
    recorder.mark_restored(context.workspace_key, "session-1", ("restore-token",))
    sessions = await recorder.list_sessions()
    records = await recorder.list_records()
    detail = await recorder.get_record("missing")
    stats = await recorder.get_stats()
    await recorder.close(timeout_seconds=0.1)

    assert sessions.items == ()
    assert records.items == ()
    assert records.next_cursor is None
    assert detail is None
    assert stats.totals.model_calls == 0
    assert stats.totals.tool_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_reusable_trace_fixtures_cover_the_approved_scenarios_and_purposes() -> None:
    fixtures = trace_contract_fixtures()

    assert set(fixtures.records) == TRACE_SCENARIO_NAMES
    assert set(fixtures.purpose_matrix) == set(TracePurpose)
    assert fixtures.purpose_matrix == TRACE_PURPOSE_COVERAGE
    assert fixtures.tool_matrix == TRACE_TOOL_COVERAGE
    assert fixtures.records["retry_success"]["retry_count"] == 2
    assert fixtures.records["final_failure"]["status"] == "failed"
    assert fixtures.records["permission_denied"]["error"]["category"] == "permission_denied"
    assert fixtures.records["cancelled"]["error"]["category"] == "cancelled"
    assert fixtures.records["no_session"]["owner_session_id"] is None
    assert fixtures.records["no_session"]["subject_session_id"] is None
    assert fixtures.detail["record"]["status"] == "success"
    assert fixtures.detail["record"]["capture_state"] == "partial"
    assert fixtures.detail["capture_issues"][0]["source"] == "model_router"
