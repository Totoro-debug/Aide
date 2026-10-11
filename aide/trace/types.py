"""Execution trace values shared by recorders, adapters, and query clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict


class TraceOrigin(StrEnum):
    FOREGROUND = "foreground"
    SUBAGENT = "subagent"
    SCHEDULE = "schedule"
    AUXILIARY = "auxiliary"


class TracePurpose(StrEnum):
    AGENT_RUN = "agent_run"
    SUBAGENT_RUN = "subagent_run"
    SCHEDULE_RUN = "schedule_run"
    TITLE_GENERATION = "title_generation"
    TASK_FRAMING = "task_framing"
    CONVERSATION_SUMMARY = "conversation_summary"
    ACTION_SUMMARY = "action_summary"
    DREAM = "dream"
    MCP_KEYWORDS = "mcp_keywords"


class TraceKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


class TraceStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class TraceErrorCategory(StrEnum):
    CANCELLED = "cancelled"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    UNAVAILABLE = "unavailable"
    OTHER = "other"


class TraceCaptureState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class TraceCaptureLayer(StrEnum):
    HTTP_BODY = "http_body"
    HTTP_METADATA = "http_metadata"
    HTTP_DECODED_BODY = "http_decoded_body"
    JSON_RPC_MESSAGE = "json_rpc_message"
    SDK_RESULT = "sdk_result"
    PROCESS_STREAM = "process_stream"
    TOOL_RESULT = "tool_result"


class TraceUsageState(StrEnum):
    REPORTED = "reported"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class TraceDirection(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"


class TraceReasoningEffort(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be an aware UTC datetime")


@dataclass(frozen=True, slots=True)
class TraceTokenUsage:
    state: TraceUsageState
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.cache_creation_input_tokens,
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in values
        ):
            raise ValueError("Trace token counts must be non-negative integers or None")
        if self.state is TraceUsageState.REPORTED and not any(
            value is not None for value in values
        ):
            raise ValueError("Reported Trace usage must include a token count")
        if self.state is not TraceUsageState.REPORTED and any(
            value is not None for value in values
        ):
            raise ValueError("Unknown or inapplicable Trace usage cannot include token counts")


@dataclass(frozen=True, slots=True)
class TraceCallMetadata:
    kind: TraceKind
    purpose: TracePurpose
    started_at: datetime
    model: str | None = None
    reasoning_effort: TraceReasoningEffort | None = None
    endpoint: str | None = None
    tool_name: str | None = None
    parent_record_id: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        _require_utc(self.started_at, "started_at")


@dataclass(frozen=True, slots=True)
class TraceAttemptMetadata:
    started_at: datetime
    model: str | None = None
    reasoning_effort: TraceReasoningEffort | None = None
    endpoint: str | None = None

    def __post_init__(self) -> None:
        _require_utc(self.started_at, "started_at")


@dataclass(frozen=True, slots=True)
class TraceCaptureHandle:
    record_id: str
    root_id: str


@dataclass(frozen=True, slots=True)
class TraceAttemptHandle:
    record_id: str
    attempt_id: str


TraceRawHandle = TraceCaptureHandle | TraceAttemptHandle


@dataclass(frozen=True, slots=True)
class TraceCaptureIssue:
    state: TraceCaptureState
    message: str
    source: str | None = None

    def __post_init__(self) -> None:
        if self.state is TraceCaptureState.COMPLETE or not self.message.strip():
            raise ValueError("Capture issues require an incomplete state and a message")


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    status: TraceStatus
    finished_at: datetime
    duration_ms: int
    final_attempt_duration_ms: int | None = None
    error_category: TraceErrorCategory | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _require_utc(self.finished_at, "finished_at")
        durations = (self.duration_ms, self.final_attempt_duration_ms)
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in durations
        ):
            raise ValueError("Trace durations must be non-negative integers")
        has_error = any(
            value is not None
            for value in (self.error_category, self.error_code, self.error_message)
        )
        if self.status is TraceStatus.SUCCESS and has_error:
            raise ValueError("Successful Trace outcomes cannot include an error")
        if self.status is TraceStatus.FAILED and (
            self.error_category is None or not self.error_message
        ):
            raise ValueError("Failed Trace outcomes require an error category and message")


@dataclass(frozen=True, slots=True)
class TraceRecordFilters:
    workspace_key: str | None = None
    owner_session_id: str | None = None
    subject_session_id: str | None = None
    root_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    kind: TraceKind | None = None
    status: TraceStatus | None = None
    purpose: TracePurpose | None = None
    tool_name: str | None = None
    model: str | None = None
    from_at: datetime | None = None
    to_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.from_at is not None:
            _require_utc(self.from_at, "from_at")
        if self.to_at is not None:
            _require_utc(self.to_at, "to_at")


@dataclass(frozen=True, slots=True)
class TraceRecordSummary:
    record_id: str
    root_id: str
    workspace_key: str
    workspace_path: str
    origin: TraceOrigin
    purpose: TracePurpose
    kind: TraceKind
    status: TraceStatus
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    retry_count: int
    capture_state: TraceCaptureState
    usage: TraceTokenUsage
    owner_session_id: str | None = None
    subject_session_id: str | None = None
    run_id: str | None = None
    schedule_run_id: str | None = None
    restore_run_token: str | None = None
    agent_id: str | None = None
    parent_root_id: str | None = None
    parent_record_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    model: str | None = None
    reasoning_effort: TraceReasoningEffort | None = None
    endpoint: str | None = None
    final_attempt_duration_ms: int | None = None
    error_category: TraceErrorCategory | None = None
    error_code: str | None = None
    error_message: str | None = None
    restored: bool = False


@dataclass(frozen=True, slots=True)
class TraceRawSource:
    source: str
    direction: TraceDirection
    layer: TraceCaptureLayer
    content_type: str | None
    content_encoding: str | None
    total_bytes: int


@dataclass(frozen=True, slots=True)
class TraceRecordDetail:
    record: TraceRecordSummary
    request: object
    result: object
    raw_sources: tuple[TraceRawSource, ...]
    capture_issues: tuple[TraceCaptureIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class TracePage:
    items: tuple[TraceRecordSummary, ...] = ()
    next_cursor: str | None = None
    snapshot_sequence: int = 0


@dataclass(frozen=True, slots=True)
class TraceRawPage:
    source: str
    direction: TraceDirection | None
    layer: TraceCaptureLayer | None
    offset: int
    data: bytes
    total_bytes: int
    next_offset: int | None
    content_type: str | None = None
    content_encoding: str | None = None


@dataclass(frozen=True, slots=True)
class TraceSessionSummary:
    workspace_key: str
    workspace_path: str
    owner_session_id: str
    subject_session_id: str | None
    origin: TraceOrigin
    last_finished_at: datetime
    record_count: int


@dataclass(frozen=True, slots=True)
class TraceSessionPage:
    items: tuple[TraceSessionSummary, ...] = ()
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class TraceTokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TraceStatsTotals:
    model_calls: int = 0
    tool_calls: int = 0
    success_count: int = 0
    failed_count: int = 0
    failure_rate: float | None = None
    retry_count: int = 0
    retried_calls: int = 0
    unknown_usage_count: int = 0
    average_duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class TraceBreakdown:
    key: str
    calls: int
    tokens: TraceTokenTotals


@dataclass(frozen=True, slots=True)
class TraceErrorBreakdown:
    category: TraceErrorCategory
    calls: int


@dataclass(frozen=True, slots=True)
class TraceTimelinePoint:
    date: str
    model_calls: int
    tool_calls: int
    successful_calls: int
    failed_calls: int
    tokens: TraceTokenTotals


@dataclass(frozen=True, slots=True)
class TraceStats:
    totals: TraceStatsTotals = TraceStatsTotals()
    tokens: TraceTokenTotals = TraceTokenTotals()
    by_model: tuple[TraceBreakdown, ...] = ()
    by_tool: tuple[TraceBreakdown, ...] = ()
    by_purpose: tuple[TraceBreakdown, ...] = ()
    by_error_category: tuple[TraceErrorBreakdown, ...] = ()
    daily: tuple[TraceTimelinePoint, ...] = ()
    methodology: str = ""


class TraceUsageDTO(TypedDict):
    state: TraceUsageState
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_creation_input_tokens: int | None


class TraceErrorDTO(TypedDict):
    category: TraceErrorCategory
    code: str | None
    message: str


class TraceRecordDTO(TypedDict):
    record_id: str
    root_id: str
    workspace_key: str
    workspace_path: str
    owner_session_id: str | None
    subject_session_id: str | None
    origin: TraceOrigin
    run_id: str | None
    schedule_run_id: str | None
    restore_run_token: str | None
    agent_id: str | None
    parent_root_id: str | None
    parent_record_id: str | None
    tool_call_id: str | None
    kind: TraceKind
    purpose: TracePurpose
    tool_name: str | None
    model: str | None
    reasoning_effort: TraceReasoningEffort | None
    endpoint: str | None
    status: TraceStatus
    error: TraceErrorDTO | None
    retry_count: int
    started_at: str
    finished_at: str
    duration_ms: int
    final_attempt_duration_ms: int | None
    usage: TraceUsageDTO
    capture_state: TraceCaptureState
    restored: bool


class TraceRawSourceDTO(TypedDict):
    source: str
    direction: TraceDirection
    layer: TraceCaptureLayer
    content_type: str | None
    content_encoding: str | None
    total_bytes: int


class TraceContentDTO(TypedDict):
    request: object
    result: object


class TraceCaptureIssueDTO(TypedDict):
    state: Literal[TraceCaptureState.PARTIAL, TraceCaptureState.UNAVAILABLE]
    message: str
    source: str | None


class TraceRecordDetailDTO(TypedDict):
    record: TraceRecordDTO
    content: TraceContentDTO
    raw_sources: list[TraceRawSourceDTO]
    capture_issues: list[TraceCaptureIssueDTO]


class TracePageDTO(TypedDict):
    items: list[TraceRecordDTO]
    next_cursor: str | None
    snapshot_sequence: int


class TraceSessionSummaryDTO(TypedDict):
    workspace_key: str
    workspace_path: str
    owner_session_id: str
    subject_session_id: str | None
    origin: TraceOrigin
    last_finished_at: str
    record_count: int


class TraceSessionsResponseDTO(TypedDict):
    items: list[TraceSessionSummaryDTO]
    next_cursor: str | None


class TraceTokenTotalsDTO(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int


class TraceStatsTotalsDTO(TypedDict):
    model_calls: int
    tool_calls: int
    success_count: int
    failed_count: int
    failure_rate: float | None
    retry_count: int
    retried_calls: int
    unknown_usage_count: int
    average_duration_ms: float | None


class TraceBreakdownDTO(TypedDict):
    key: str
    calls: int
    tokens: TraceTokenTotalsDTO


class TraceErrorBreakdownDTO(TypedDict):
    category: TraceErrorCategory
    calls: int


class TraceTimelinePointDTO(TypedDict):
    date: str
    model_calls: int
    tool_calls: int
    successful_calls: int
    failed_calls: int
    tokens: TraceTokenTotalsDTO


class TraceStatsResponseDTO(TypedDict):
    from_at: str | None
    to_at: str | None
    workspace_key: str | None
    totals: TraceStatsTotalsDTO
    tokens: TraceTokenTotalsDTO
    by_model: list[TraceBreakdownDTO]
    by_tool: list[TraceBreakdownDTO]
    by_purpose: list[TraceBreakdownDTO]
    by_error_category: list[TraceErrorBreakdownDTO]
    daily: list[TraceTimelinePointDTO]
    methodology: str


class TraceRawPageDTO(TypedDict):
    source: str
    direction: TraceDirection | None
    layer: TraceCaptureLayer | None
    offset: int
    data_base64: str
    total_bytes: int
    next_offset: int | None
    content_type: str | None
    content_encoding: str | None


class TraceListQueryDTO(TypedDict, total=False):
    workspace_key: str
    owner_session_id: str
    subject_session_id: str
    root_id: str
    run_id: str
    agent_id: str
    kind: TraceKind
    status: TraceStatus
    purpose: TracePurpose
    tool_name: str
    model: str
    from_at: str
    to_at: str
    cursor: str
    limit: int


class TraceStatsQueryDTO(TypedDict, total=False):
    from_at: str
    to_at: str
    workspace_key: str
    model: str
    tool_name: str
    purpose: TracePurpose


class TraceSessionsQueryDTO(TypedDict, total=False):
    cursor: str
    limit: int


class TraceRawQueryDTO(TypedDict):
    source: str
    offset: NotRequired[int]
    limit_bytes: NotRequired[int]


__all__ = [
    "TraceAttemptHandle",
    "TraceAttemptMetadata",
    "TraceBreakdown",
    "TraceCallMetadata",
    "TraceCaptureHandle",
    "TraceCaptureIssue",
    "TraceCaptureIssueDTO",
    "TraceCaptureLayer",
    "TraceCaptureState",
    "TraceContentDTO",
    "TraceDirection",
    "TraceErrorBreakdown",
    "TraceErrorBreakdownDTO",
    "TraceErrorCategory",
    "TraceErrorDTO",
    "TraceKind",
    "TraceListQueryDTO",
    "TraceOrigin",
    "TraceOutcome",
    "TracePage",
    "TracePageDTO",
    "TracePurpose",
    "TraceRawHandle",
    "TraceRawPage",
    "TraceRawPageDTO",
    "TraceRawQueryDTO",
    "TraceRawSource",
    "TraceRawSourceDTO",
    "TraceReasoningEffort",
    "TraceRecordDTO",
    "TraceRecordDetail",
    "TraceRecordDetailDTO",
    "TraceRecordFilters",
    "TraceRecordSummary",
    "TraceSessionPage",
    "TraceSessionSummary",
    "TraceSessionSummaryDTO",
    "TraceSessionsQueryDTO",
    "TraceSessionsResponseDTO",
    "TraceStats",
    "TraceStatsQueryDTO",
    "TraceStatsResponseDTO",
    "TraceStatsTotals",
    "TraceStatsTotalsDTO",
    "TraceStatus",
    "TraceTimelinePoint",
    "TraceTimelinePointDTO",
    "TraceTokenTotals",
    "TraceTokenTotalsDTO",
    "TraceTokenUsage",
    "TraceUsageDTO",
    "TraceUsageState",
]
