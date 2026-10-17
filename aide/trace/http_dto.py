"""Projection helpers for the versioned Execution Trace HTTP DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

from aide.trace.types import (
    TraceCaptureIssueDTO,
    TraceCaptureState,
    TraceContentDTO,
    TraceErrorDTO,
    TraceKind,
    TracePage,
    TracePageDTO,
    TraceRecordDetail,
    TraceRecordDetailDTO,
    TraceRecordDTO,
    TraceRecordFilters,
    TraceRecordSummary,
    TraceSessionPage,
    TraceSessionsResponseDTO,
    TraceStats,
    TraceStatsResponseDTO,
    TraceTokenTotals,
    TraceTokenTotalsDTO,
)


def _datetime_dto(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _optional_datetime_dto(value: datetime | None) -> str | None:
    return None if value is None else _datetime_dto(value)


def _token_totals_dto(value: TraceTokenTotals) -> TraceTokenTotalsDTO:
    return {
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "total_tokens": value.total_tokens,
        "cached_input_tokens": value.cached_input_tokens,
        "cache_creation_input_tokens": value.cache_creation_input_tokens,
    }


def trace_stats_response_dto(
    stats: TraceStats,
    filters: TraceRecordFilters,
) -> TraceStatsResponseDTO:
    return {
        "from_at": _optional_datetime_dto(filters.from_at),
        "to_at": _optional_datetime_dto(filters.to_at),
        "workspace_key": filters.workspace_key,
        "totals": {
            "model_calls": stats.totals.model_calls,
            "tool_calls": stats.totals.tool_calls,
            "success_count": stats.totals.success_count,
            "failed_count": stats.totals.failed_count,
            "failure_rate": stats.totals.failure_rate,
            "retry_count": stats.totals.retry_count,
            "retried_calls": stats.totals.retried_calls,
            "unknown_usage_count": stats.totals.unknown_usage_count,
            "average_duration_ms": stats.totals.average_duration_ms,
        },
        "tokens": _token_totals_dto(stats.tokens),
        "by_model": [
            {"key": item.key, "calls": item.calls, "tokens": _token_totals_dto(item.tokens)}
            for item in stats.by_model
        ],
        "by_tool": [
            {"key": item.key, "calls": item.calls, "tokens": _token_totals_dto(item.tokens)}
            for item in stats.by_tool
        ],
        "by_purpose": [
            {"key": item.key, "calls": item.calls, "tokens": _token_totals_dto(item.tokens)}
            for item in stats.by_purpose
        ],
        "by_error_category": [
            {"category": item.category, "calls": item.calls} for item in stats.by_error_category
        ],
        "daily": [
            {
                "date": item.date,
                "model_calls": item.model_calls,
                "tool_calls": item.tool_calls,
                "successful_calls": item.successful_calls,
                "failed_calls": item.failed_calls,
                "tokens": _token_totals_dto(item.tokens),
            }
            for item in stats.daily
        ],
        "methodology": stats.methodology,
    }


def trace_record_dto(record: TraceRecordSummary) -> TraceRecordDTO:
    error: TraceErrorDTO | None = None
    if record.error_category is not None:
        error = {
            "category": record.error_category,
            "code": record.error_code,
            "message": record.error_message or "",
        }
    return {
        "record_id": record.record_id,
        "root_id": record.root_id,
        "workspace_key": record.workspace_key,
        "workspace_path": record.workspace_path,
        "owner_session_id": record.owner_session_id,
        "subject_session_id": record.subject_session_id,
        "origin": record.origin,
        "run_id": record.run_id,
        "schedule_run_id": record.schedule_run_id,
        "restore_run_token": record.restore_run_token,
        "agent_id": record.agent_id,
        "parent_root_id": record.parent_root_id,
        "parent_record_id": record.parent_record_id,
        "tool_call_id": record.tool_call_id,
        "kind": record.kind,
        "purpose": record.purpose,
        "tool_name": record.tool_name,
        "model": record.model,
        "reasoning_effort": record.reasoning_effort,
        "endpoint": record.endpoint,
        "protocol": record.protocol,
        "output_complete": record.output_complete,
        "status": record.status,
        "error": error,
        "retry_count": record.retry_count,
        "started_at": _datetime_dto(record.started_at),
        "finished_at": _datetime_dto(record.finished_at),
        "duration_ms": record.duration_ms,
        "final_attempt_duration_ms": record.final_attempt_duration_ms,
        "usage": {
            "state": record.usage.state,
            "input_tokens": record.usage.input_tokens,
            "output_tokens": record.usage.output_tokens,
            "total_tokens": record.usage.total_tokens,
            "cached_input_tokens": record.usage.cached_input_tokens,
            "cache_creation_input_tokens": record.usage.cache_creation_input_tokens,
        },
        "capture_state": record.capture_state,
        "restored": record.restored,
    }


def trace_page_dto(page: TracePage) -> TracePageDTO:
    return {
        "items": [trace_record_dto(item) for item in page.items],
        "next_cursor": page.next_cursor,
        "snapshot_sequence": page.snapshot_sequence,
    }


def trace_sessions_response_dto(page: TraceSessionPage) -> TraceSessionsResponseDTO:
    return {
        "items": [
            {
                "workspace_key": item.workspace_key,
                "workspace_path": item.workspace_path,
                "owner_session_id": item.owner_session_id,
                "subject_session_id": item.subject_session_id,
                "origin": item.origin,
                "last_finished_at": _datetime_dto(item.last_finished_at),
                "record_count": item.record_count,
            }
            for item in page.items
        ],
        "next_cursor": page.next_cursor,
    }


def trace_record_detail_dto(detail: TraceRecordDetail) -> TraceRecordDetailDTO:
    capture_issues: list[TraceCaptureIssueDTO] = [
        {
            "state": cast(
                Literal[TraceCaptureState.PARTIAL, TraceCaptureState.UNAVAILABLE],
                item.state,
            ),
            "message": item.message,
            "source": item.source,
        }
        for item in detail.capture_issues
    ]
    content: TraceContentDTO = (
        {"output": detail.result} if detail.record.kind is TraceKind.MODEL
        else {"arguments": detail.request, "result": detail.result}
    )
    return {
        "record": trace_record_dto(detail.record),
        "content": content,
        "related_tools": [
            {"record_id": tool.record_id, "tool_call_id": tool.tool_call_id,
             "tool_name": tool.tool_name, "status": tool.status}
            for tool in detail.related_tools
        ],
        "capture_issues": capture_issues,
    }
