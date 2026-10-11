"""Reusable wire and byte fixtures for Execution Trace contract consumers."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from aide.trace.types import (
    TraceErrorCategory,
    TraceOrigin,
    TracePurpose,
    TraceStatus,
)

TRACE_PURPOSE_COVERAGE = {
    TracePurpose.AGENT_RUN: TraceOrigin.FOREGROUND,
    TracePurpose.SUBAGENT_RUN: TraceOrigin.SUBAGENT,
    TracePurpose.SCHEDULE_RUN: TraceOrigin.SCHEDULE,
    TracePurpose.TITLE_GENERATION: TraceOrigin.AUXILIARY,
    TracePurpose.TASK_FRAMING: TraceOrigin.AUXILIARY,
    TracePurpose.CONVERSATION_SUMMARY: TraceOrigin.AUXILIARY,
    TracePurpose.ACTION_SUMMARY: TraceOrigin.AUXILIARY,
    TracePurpose.DREAM: TraceOrigin.AUXILIARY,
    TracePurpose.MCP_KEYWORDS: TraceOrigin.AUXILIARY,
}

TRACE_SCENARIO_NAMES = frozenset(
    {
        "success",
        "retry_success",
        "final_failure",
        "permission_denied",
        "cancelled",
        "subagent",
        "no_session",
        "binary",
        "sse",
        "unknown_usage",
        "partial_capture",
    }
)

TRACE_TOOL_COVERAGE = {
    "builtin_gateway": ("T07", "authorization", "arguments", "result"),
    "mcp_http": ("T08", "wire_request", "wire_result"),
    "mcp_stdio": ("T08", "wire_request", "wire_result"),
    "exec": ("T09", "stdout", "stderr", "cancellation"),
    "web_fetch": ("T10", "consumed_response"),
    "web_search": ("T10", "ddgs_result"),
}

_REPORTED_USAGE = {
    "state": "reported",
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "cached_input_tokens": 10,
    "cache_creation_input_tokens": 0,
}
_NOT_APPLICABLE_USAGE = {
    "state": "not_applicable",
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
    "cached_input_tokens": None,
    "cache_creation_input_tokens": None,
}
_UNKNOWN_USAGE = {
    "state": "unknown",
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
    "cached_input_tokens": None,
    "cache_creation_input_tokens": None,
}


def _record(
    scenario: str,
    *,
    origin: str = "foreground",
    purpose: str = "agent_run",
    kind: str = "model",
    status: str = "success",
    usage: Mapping[str, object] | None = None,
    error: dict[str, str] | None = None,
    **overrides: object,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_id": f"record-{scenario}",
        "root_id": f"root-{scenario}",
        "workspace_key": "sha256:workspace",
        "workspace_path": "D:/work/project",
        "owner_session_id": "owner-session",
        "subject_session_id": "subject-session",
        "origin": origin,
        "run_id": "service-run",
        "schedule_run_id": None,
        "restore_run_token": "restore-token",
        "agent_id": "agent-1" if origin == "subagent" else None,
        "parent_root_id": "root-parent" if origin == "subagent" else None,
        "parent_record_id": None,
        "tool_call_id": None,
        "kind": kind,
        "purpose": purpose,
        "tool_name": "read_file" if kind == "tool" else None,
        "model": "provider/model" if kind == "model" else None,
        "reasoning_effort": "high" if kind == "model" else None,
        "endpoint": "https://provider.invalid/v1" if kind == "model" else None,
        "status": status,
        "error": error,
        "retry_count": 0,
        "started_at": "2026-10-11T01:00:00Z",
        "finished_at": "2026-10-11T01:00:01Z",
        "duration_ms": 1000,
        "final_attempt_duration_ms": 800 if kind == "model" else None,
        "usage": deepcopy(_REPORTED_USAGE if usage is None else usage),
        "capture_state": "complete",
        "restored": False,
    }
    record.update(overrides)
    return record


def _raw_source(
    source: str, direction: str, content_type: str, total_bytes: int, layer: str
) -> dict[str, object]:
    return {
        "source": source,
        "direction": direction,
        "layer": layer,
        "content_type": content_type,
        "content_encoding": None,
        "total_bytes": total_bytes,
    }


@dataclass(frozen=True, slots=True)
class TraceContractFixtures:
    records: dict[str, dict[str, Any]]
    purpose_matrix: dict[TracePurpose, TraceOrigin]
    tool_matrix: dict[str, tuple[str, ...]]
    detail: dict[str, Any]
    page: dict[str, Any]
    sessions: dict[str, Any]
    stats: dict[str, Any]
    raw_pages: dict[str, dict[str, object]]
    raw_content: dict[str, bytes]


def trace_contract_fixtures() -> TraceContractFixtures:
    """Return detached fixtures so consumers can safely mutate a scenario."""
    binary = b"\x00\xff\x80A"
    sse = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    records = {
        "success": _record("success"),
        "retry_success": _record(
            "retry_success",
            retry_count=2,
            model="provider/fallback",
            endpoint="https://fallback.invalid/v1",
            final_attempt_duration_ms=500,
        ),
        "final_failure": _record(
            "final_failure",
            status=TraceStatus.FAILED.value,
            error={
                "category": TraceErrorCategory.PROVIDER_ERROR.value,
                "code": "provider_error",
                "message": "The provider request failed.",
            },
            usage=_UNKNOWN_USAGE,
            retry_count=4,
        ),
        "permission_denied": _record(
            "permission_denied",
            kind="tool",
            status=TraceStatus.FAILED.value,
            usage=_NOT_APPLICABLE_USAGE,
            error={
                "category": TraceErrorCategory.PERMISSION_DENIED.value,
                "code": "permission_denied",
                "message": "The tool call was not authorized.",
            },
        ),
        "cancelled": _record(
            "cancelled",
            kind="tool",
            status=TraceStatus.FAILED.value,
            usage=_NOT_APPLICABLE_USAGE,
            error={
                "category": TraceErrorCategory.CANCELLED.value,
                "code": "cancelled",
                "message": "The tool call was cancelled.",
            },
        ),
        "subagent": _record(
            "subagent",
            origin=TraceOrigin.SUBAGENT.value,
            purpose=TracePurpose.SUBAGENT_RUN.value,
        ),
        "no_session": _record(
            "no_session",
            origin=TraceOrigin.AUXILIARY.value,
            purpose=TracePurpose.DREAM.value,
            owner_session_id=None,
            subject_session_id=None,
            run_id=None,
            schedule_run_id=None,
            restore_run_token=None,
        ),
        "binary": _record(
            "binary",
            kind="tool",
            usage=_NOT_APPLICABLE_USAGE,
            tool_name="exec",
        ),
        "sse": _record("sse"),
        "unknown_usage": _record(
            "unknown_usage",
            origin=TraceOrigin.AUXILIARY.value,
            purpose=TracePurpose.TASK_FRAMING.value,
            usage=_UNKNOWN_USAGE,
        ),
        "partial_capture": _record("partial_capture", capture_state="partial"),
    }
    detail = {
        "record": deepcopy(records["partial_capture"]),
        "content": {
            "request": {"messages": [{"role": "user", "content": "Summarize this."}]},
            "result": {"text": "Summary complete."},
        },
        "raw_sources": [
            _raw_source("provider_request", "request", "application/json", 88, "http_body"),
            _raw_source("provider_sse", "response", "text/event-stream", len(sse), "http_body"),
        ],
        "capture_issues": [
            {"state": "partial", "source": "provider_sse", "message": "A chunk was dropped."}
        ],
    }
    page = {
        "items": [deepcopy(records["success"]), deepcopy(records["retry_success"])],
        "next_cursor": "trace-cursor-1",
        "snapshot_sequence": 2,
    }
    sessions = {
        "items": [
            {
                "workspace_key": "sha256:workspace",
                "workspace_path": "D:/work/project",
                "owner_session_id": "owner-session",
                "subject_session_id": "subject-session",
                "origin": TraceOrigin.FOREGROUND.value,
                "last_finished_at": "2026-10-11T01:00:01Z",
                "record_count": 2,
            }
        ],
        "next_cursor": None,
    }
    zero_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    stats = {
        "from_at": "2026-10-04T00:00:00Z",
        "to_at": "2026-10-11T00:00:00Z",
        "workspace_key": "sha256:workspace",
        "totals": {
            "model_calls": 8,
            "tool_calls": 2,
            "success_count": 8,
            "failed_count": 2,
            "failure_rate": 0.2,
            "retry_count": 2,
            "retried_calls": 1,
            "unknown_usage_count": 1,
            "average_duration_ms": 250.5,
        },
        "tokens": {
            "input_tokens": 1000,
            "output_tokens": 400,
            "total_tokens": 1400,
            "cached_input_tokens": 75,
            "cache_creation_input_tokens": 0,
        },
        "by_model": [{"key": "provider/model", "calls": 8, "tokens": deepcopy(zero_tokens)}],
        "by_tool": [{"key": "read_file", "calls": 2, "tokens": deepcopy(zero_tokens)}],
        "by_purpose": [
            {"key": TracePurpose.AGENT_RUN.value, "calls": 4, "tokens": deepcopy(zero_tokens)}
        ],
        "by_error_category": [{"category": "provider_error", "calls": 1}],
        "daily": [
            {
                "date": "2026-10-11",
                "model_calls": 8,
                "tool_calls": 2,
                "successful_calls": 8,
                "failed_calls": 2,
                "tokens": deepcopy(zero_tokens),
            }
        ],
        "methodology": "Finalized retained records; unreported model usage is counted as unknown.",
    }
    raw_pages = {
        "binary": {
            "source": "exec_stdout",
            "direction": "response",
            "layer": "process_stream",
            "offset": 0,
            "data_base64": b64encode(binary).decode("ascii"),
            "total_bytes": len(binary),
            "next_offset": None,
            "content_type": "application/octet-stream",
            "content_encoding": None,
        },
        "sse": {
            "source": "provider_sse",
            "direction": "response",
            "layer": "http_body",
            "offset": 0,
            "data_base64": b64encode(sse).decode("ascii"),
            "total_bytes": len(sse),
            "next_offset": None,
            "content_type": "text/event-stream",
            "content_encoding": None,
        },
    }
    return TraceContractFixtures(
        records=deepcopy(records),
        purpose_matrix=dict(TRACE_PURPOSE_COVERAGE),
        tool_matrix=dict(TRACE_TOOL_COVERAGE),
        detail=detail,
        page=page,
        sessions=sessions,
        stats=stats,
        raw_pages=raw_pages,
        raw_content={"binary": binary, "sse": sse},
    )


__all__ = [
    "TRACE_PURPOSE_COVERAGE",
    "TRACE_SCENARIO_NAMES",
    "TRACE_TOOL_COVERAGE",
    "TraceContractFixtures",
    "trace_contract_fixtures",
]
