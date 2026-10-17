"""The fixed Core Tool Catalog and its only invocation boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from loguru import logger

from aide.agent.confirmation import ConfirmationAborted, ConfirmationUnavailable
from aide.agent.tools.base import (
    ArtifactReference,
    BaseTool,
    ToolError,
)
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.edit_file import EditFileTool
from aide.agent.tools.core.exec import ExecTool
from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.core.glob import GlobTool
from aide.agent.tools.core.grep import GrepTool
from aide.agent.tools.core.list_dir import ListDirTool
from aide.agent.tools.core.read_file import ReadFileTool
from aide.agent.tools.core.schedule import ScheduleTool
from aide.agent.tools.core.web_fetch import WebFetchTool
from aide.agent.tools.core.web_search import WebSearchTool
from aide.agent.tools.core.write_file import WriteFileTool
from aide.agent.tools.file_mutation import FileMutationRecorder, is_protected_restore_target
from aide.agent.tools.mcp import MCPTool
from aide.agent.tools.permission import (
    MCPToolIdentity,
    NetworkConfirmationDecision,
    PermissionContext,
    PermissionSnapshot,
    ToolAuthorizationFailure,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    ToolPermissionPolicy,
)
from aide.schedule.service import ScheduleService
from aide.trace.sanitization import sanitize_trace_text_urls, sanitize_trace_value
from aide.trace.scope import (
    TraceExecutionScope,
    TraceToolCallScope,
    current_trace_scope,
    trace_tool_call,
)
from aide.trace.types import (
    TraceCallMetadata,
    TraceCaptureHandle,
    TraceCaptureIssue,
    TraceCaptureState,
    TraceErrorCategory,
    TraceKind,
    TraceOutcome,
    TraceStatus,
    TraceTokenUsage,
    TraceUsageState,
)
from aide.utils.validation import require_uuid4

type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOutcome = ConfirmationDecision | None
type ToolResultStatus = Literal["success", "error", "refused"]
type ConfirmationRequester = Callable[["ConfirmationRequest"], Awaitable[ConfirmationDecision]]

BUILT_IN_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "grep",
    "exec",
    "web_search",
    "web_fetch",
    "schedule",
    "tool_search",
    "spawn_agent",
    "wait_agent",
    "list_agents",
)
_MICRO_COMPRESSION_ELIGIBLE_BUILT_IN_NAMES = frozenset(
    {"exec", "glob", "grep", "list_dir", "read_file", "web_fetch", "web_search"}
)


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequest:
    """One immutable confirmation request bound to one normalized Tool call."""

    confirmation_id: UUID
    tool_call_id: str
    tool_name: str
    reason: str
    summary: str
    _details: dict[str, Any] = field(repr=False)
    warnings: tuple[str, ...] = ()
    mcp_identity: MCPToolIdentity | None = None

    def __init__(
        self,
        confirmation_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        details: dict[str, Any],
        warnings: tuple[str, ...] = (),
        *,
        reason: str = "",
        mcp_identity: MCPToolIdentity | None = None,
    ) -> None:
        require_uuid4(confirmation_id, field="confirmation_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TypeError("confirmation tool_call_id must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError("confirmation tool_name must be a non-empty string")
        if not isinstance(summary, str) or not summary or len(summary) > 240:
            raise ValueError("confirmation summary must contain 1 through 240 characters")
        if not isinstance(reason, str):
            raise TypeError("confirmation reason must be a string")
        if not isinstance(details, dict):
            raise TypeError("confirmation details must be a JSON object")
        if mcp_identity is not None and not isinstance(mcp_identity, MCPToolIdentity):
            raise TypeError("confirmation MCP identity must be MCPToolIdentity or None")
        if not isinstance(warnings, (tuple, list)) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise TypeError("confirmation warnings must be a sequence of strings")
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", deepcopy(details))
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "mcp_identity", deepcopy(mcp_identity))

    @property
    def details(self) -> dict[str, Any]:
        """Return a detached view of the normalized operation details."""
        return deepcopy(self._details)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "confirmation_id": str(self.confirmation_id),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "summary": self.summary,
            "details": deepcopy(self._details),
            "warnings": list(self.warnings),
        }
        if self.mcp_identity is not None:
            result["mcp_identity"] = self.mcp_identity.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ToolConfirmationMetadata:
    """The request and decision carried by a Tool Result."""

    request: ConfirmationRequest
    decision: ConfirmationOutcome

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "decision": self.decision}


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """A provider Tool call preserving its raw JSON argument text."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized result returned by the Tool Gateway."""

    tool_call_id: str
    name: str
    status: ToolResultStatus
    content: str
    artifact: ArtifactReference | None = None
    confirmation: ToolConfirmationMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("Tool result name must be a non-empty string")
        if self.status not in {"success", "error", "refused"}:
            raise ValueError("Tool result status is invalid")
        if not isinstance(self.content, str):
            raise TypeError("Tool result content must be a string")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactReference):
            raise TypeError("Tool result artifact must be an ArtifactReference")
        if self.status != "success" and self.artifact is not None:
            raise ValueError("only successful Tool results may contain an artifact")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "content": self.content,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }
        if self.confirmation is not None:
            result["confirmation"] = self.confirmation.to_dict()
        return result


@dataclass(slots=True)
class _ToolTraceCapture:
    scope: TraceExecutionScope
    handle: TraceCaptureHandle
    started_monotonic: float
    request: dict[str, object]
    arguments_parsed: bool = False
    arguments_prepared: bool = False
    tool_resolved: bool = False
    finished: bool = False
    timed_out: bool = False

    def set_request(self) -> None:
        try:
            safe_request = _trace_safe_tool_request(self.request)
            self.scope.recorder.set_request(
                self.handle, safe_request.get("arguments", safe_request.get("arguments_raw")),
            )
        except Exception:
            self._report_issue("Tool arguments could not be captured.")

    def set_parsed_arguments(self, arguments: dict[str, Any]) -> None:
        self.arguments_parsed = True
        try:
            self.request["arguments"] = deepcopy(arguments)
            self.request.pop("arguments_raw", None)
            self.set_request()
        except Exception:
            self._report_issue("Parsed Tool arguments could not be captured.")

    def mark_arguments_prepared(self) -> None:
        self.arguments_prepared = True

    def finish_result(self, result: ToolResult) -> None:
        saved_result = {
            "content": result.content,
            "artifact": None if result.artifact is None else result.artifact.to_dict(),
        }
        if result.name == "web_search":
            saved_result["content"] = sanitize_trace_text_urls(result.content)
        if result.status == "success":
            self._finish(
                TraceOutcome(
                    status=TraceStatus.SUCCESS,
                    finished_at=datetime.now(UTC),
                    duration_ms=self._duration_ms(),
                ),
                result=saved_result,
            )
            return

        if result.status == "refused":
            category = TraceErrorCategory.PERMISSION_DENIED
        elif not self.arguments_parsed or (self.tool_resolved and not self.arguments_prepared):
            category = TraceErrorCategory.VALIDATION_ERROR
        elif not self.tool_resolved:
            category = TraceErrorCategory.UNAVAILABLE
        elif self.timed_out:
            category = TraceErrorCategory.TIMEOUT
        else:
            category = TraceErrorCategory.TOOL_ERROR
        self._finish(
            TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=datetime.now(UTC),
                duration_ms=self._duration_ms(),
                error_category=category,
                error_code=category.value,
                error_message=cast(str, saved_result["content"]),
            ),
            result=saved_result,
        )


    def finish_cancelled(self, error: BaseException) -> None:
        self._finish_error(
            TraceErrorCategory.CANCELLED,
            str(error) or "Tool call was cancelled.",
            error_code="cancelled",
        )

    def finish_unexpected(self, error: Exception) -> None:
        self._finish_error(
            TraceErrorCategory.TOOL_ERROR,
            "Tool call could not be completed.",
            error_code=type(error).__name__,
        )

    def _finish_error(
        self,
        category: TraceErrorCategory,
        message: str,
        *,
        error_code: str,
    ) -> None:
        self._finish(
            TraceOutcome(
                status=TraceStatus.FAILED,
                finished_at=datetime.now(UTC),
                duration_ms=self._duration_ms(),
                error_category=category,
                error_code=error_code,
                error_message=message,
            ),
            result=None,
        )


    def _report_issue(self, message: str, *, source: str = "tool_gateway") -> None:
        try:
            self.scope.recorder.report_capture_issue(
                self.handle,
                TraceCaptureIssue(
                    state=TraceCaptureState.PARTIAL,
                    message=message,
                    source=source,
                ),
            )
        except Exception:
            logger.warning("Execution Trace Tool capture issue could not be reported")

    def _duration_ms(self) -> int:
        return max(0, int((monotonic() - self.started_monotonic) * 1000))

    def _finish(self, outcome: TraceOutcome, *, result: object) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            self.scope.recorder.finish_call(
                self.handle,
                outcome,
                result=result,
                usage=TraceTokenUsage(TraceUsageState.NOT_APPLICABLE),
                retry_count=0,
            )
        except Exception:
            logger.warning("Execution Trace Tool outcome could not be captured")


def _trace_safe_tool_request(request: dict[str, object]) -> dict[str, object]:
    safe_request = deepcopy(request)
    if safe_request.get("tool_name") == "web_fetch":
        raw_arguments = safe_request.get("arguments_raw")
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                safe_request["arguments_raw"] = "[redacted invalid Web Fetch arguments]"
            else:
                safe_arguments = sanitize_trace_value(parsed_arguments)
                safe_request["arguments_raw"] = json.dumps(
                    safe_arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        safe_request = cast(dict[str, object], sanitize_trace_value(safe_request))
    return safe_request


@dataclass(slots=True)
class ToolResultTrace:
    """Own the Trace finalizer for one Runner call, including its child task."""

    _capture: _ToolTraceCapture | None = None

    def finish(self, result: ToolResult) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.finish_result(result)
            except Exception:
                logger.warning("Execution Trace final Tool result could not be captured")

    def finish_cancelled(self, error: BaseException) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.finish_cancelled(error)
            except Exception:
                logger.warning("Execution Trace final Tool cancellation could not be captured")


_ACTIVE_TOOL_RESULT_TRACE: ContextVar[ToolResultTrace | None] = ContextVar(
    "aide_active_tool_result_trace", default=None
)


@contextmanager
def trace_tool_result(trace: ToolResultTrace) -> Iterator[None]:
    """Defer this call's publication until the Runner observes its final return channel."""
    token = _ACTIVE_TOOL_RESULT_TRACE.set(trace)
    try:
        yield
    finally:
        _ACTIVE_TOOL_RESULT_TRACE.reset(token)


def _start_tool_trace(tool_call: ModelToolCall) -> _ToolTraceCapture | None:
    scope = current_trace_scope()
    if scope is None:
        return None
    started_at = datetime.now(UTC)
    try:
        handle = scope.recorder.begin_call(
            scope.context,
            TraceCallMetadata(
                kind=TraceKind.TOOL,
                purpose=scope.purpose,
                started_at=started_at,
                tool_name=tool_call.name,
                parent_record_id=scope.parent_record_id,
                tool_call_id=tool_call.id,
            ),
        )
    except Exception:
        logger.warning("Execution Trace Tool call could not be started")
        return None
    capture = _ToolTraceCapture(
        scope=scope,
        handle=handle,
        started_monotonic=monotonic(),
        request={
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "arguments_raw": tool_call.arguments,
        },
    )
    capture.set_request()
    return capture


def _normalize_tool_names(names: Collection[str], *, label: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{label} must be a collection of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{label} must contain non-empty strings")
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return tuple(normalized)


class BuiltInToolCatalog:
    """Own one immutable set of reusable Built-in Tool implementations."""

    def __init__(
        self,
        *,
        skill_root: Path | None = None,
        exec_host: ExecHost | None = None,
    ) -> None:
        tools: tuple[BaseTool, ...] = (
            ReadFileTool(skill_root=skill_root),
            WriteFileTool(),
            EditFileTool(),
            ListDirTool(),
            GlobTool(),
            GrepTool(),
            ExecTool(host=exec_host),
            WebSearchTool(),
            WebFetchTool(),
            ScheduleTool(),
        )
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Built-in Tool names must be unique")
        self._tools = tools

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return the shared implementation objects in fixed catalog order."""
        return self._tools


class ToolGateway:
    """Create and invoke the Built-in Tool Catalog."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        schedule_service: ScheduleService | None = None,
        skill_root: Path | None = None,
        additional_tools: Sequence[BaseTool] = (),
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        exec_host: ExecHost | None = None,
        catalog: BuiltInToolCatalog | None = None,
        tool_context: ToolRunContext | None = None,
    ) -> None:
        if catalog is not None and not isinstance(catalog, BuiltInToolCatalog):
            raise TypeError("Tool Gateway catalog must be a BuiltInToolCatalog")
        if tool_context is not None and not isinstance(tool_context, ToolRunContext):
            raise TypeError("Tool Gateway context must be a ToolRunContext")
        if workspace is not None and not isinstance(workspace, Path):
            raise TypeError("Tool Gateway workspace must be a Path")
        if schedule_service is not None and not isinstance(schedule_service, ScheduleService):
            raise TypeError("Tool Gateway schedule_service must be a ScheduleService or None")

        if tool_context is None and workspace is not None:
            tool_context = ToolRunContext(
                workspace=workspace,
                schedule_service=schedule_service,
            )
        elif (
            tool_context is not None
            and workspace is not None
            and workspace != tool_context.workspace
        ):
            raise ValueError("Tool Gateway workspace must match its Tool Run Context")
        if tool_context is not None and schedule_service is not None:
            if (
                tool_context.schedule_service is not None
                and tool_context.schedule_service is not schedule_service
            ):
                raise ValueError("Tool Gateway Schedule Service must match its Tool Run Context")

        generation_tools = tuple(additional_tools)
        if any(not isinstance(tool, BaseTool) for tool in generation_tools):
            raise TypeError("Additional Tools must be BaseTool instances")
        if catalog is None:
            if workspace is None:
                raise TypeError("Tool Gateway requires a workspace or shared catalog")
            catalog = BuiltInToolCatalog(skill_root=skill_root, exec_host=exec_host)
        tools: tuple[BaseTool, ...] = (*catalog.tools, *generation_tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Tool names must be unique")
        self._catalog = tools
        self._tools = {tool.name: tool for tool in tools}
        self._exposed_names = tuple(tool.name for tool in tools)
        self._tool_context = tool_context
        self._failure_observer: Callable[[Exception], None] | None = None
        self._permission_policy = (
            ToolPermissionPolicy() if permission_policy is None else permission_policy
        )
        self._permission_context = (
            PermissionContext(
                workspace_root=(None if tool_context is None else tool_context.workspace)
            )
            if permission_context is None
            else permission_context
        )
        if (
            tool_context is not None
            and permission_context is not None
            and permission_context.workspace_root is not None
            and permission_context.workspace_root != tool_context.workspace
        ):
            raise ValueError("Tool Gateway permission context must match its Tool Run Context")

    def for_run(
        self,
        *,
        exposed_names: Collection[str],
        excluded_names: Collection[str] = (),
        allowed_names: Collection[str] | None = None,
        run_tools: Sequence[BaseTool] = (),
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        permission_snapshot: PermissionSnapshot | None = None,
        tool_context: ToolRunContext | None = None,
    ) -> ToolGateway:
        """Create an isolated Run view over this Gateway's reusable Tool instances."""
        excluded = _normalize_tool_names(excluded_names, label="Excluded Tool names")
        excluded_set = set(excluded)
        allowed = (
            None
            if allowed_names is None
            else set(_normalize_tool_names(allowed_names, label="Allowed Tool names"))
        )
        additions = tuple(run_tools)
        if any(not isinstance(tool, BaseTool) for tool in additions):
            raise TypeError("Run Tools must be BaseTool instances")

        catalog = tuple(
            tool
            for tool in (*self._catalog, *additions)
            if tool.name not in excluded_set and (allowed is None or tool.name in allowed)
        )
        if len({tool.name for tool in catalog}) != len(catalog):
            raise ValueError("Run Tool names must be unique")

        available_names = {tool.name for tool in catalog}
        requested = _normalize_tool_names(exposed_names, label="Exposed Tool names")
        unknown = set(requested) - available_names
        if unknown:
            raise ValueError("Exposed Tool names must be available in the Run Catalog")
        exposure = tuple(tool.name for tool in catalog if tool.name in requested)

        if permission_snapshot is not None and permission_context is not None:
            raise ValueError("Run permission context and snapshot are mutually exclusive")
        if tool_context is not None and not isinstance(tool_context, ToolRunContext):
            raise TypeError("Run Tool context must be a ToolRunContext")
        selected_context = (
            _context_from_snapshot(self._permission_context, permission_snapshot)
            if permission_snapshot is not None
            else (self._permission_context if permission_context is None else permission_context)
        )
        selected_tool_context = self._tool_context if tool_context is None else tool_context
        if selected_tool_context is not None and selected_context.workspace_root is not None:
            if selected_context.workspace_root != selected_tool_context.workspace:
                raise ValueError("Run permission context must match its Tool Run Context")
        return self._from_catalog(
            catalog,
            exposed_names=exposure,
            on_failure=self._failure_observer,
            permission_policy=(
                self._permission_policy if permission_policy is None else permission_policy
            ),
            permission_context=selected_context,
            tool_context=selected_tool_context,
        )

    @classmethod
    def _from_catalog(
        cls,
        catalog: tuple[BaseTool, ...],
        *,
        exposed_names: tuple[str, ...],
        on_failure: Callable[[Exception], None] | None,
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        tool_context: ToolRunContext | None = None,
    ) -> ToolGateway:
        gateway = object.__new__(cls)
        gateway._catalog = catalog
        gateway._tools = {tool.name: tool for tool in catalog}
        gateway._exposed_names = exposed_names
        gateway._tool_context = tool_context
        gateway._failure_observer = on_failure
        gateway._permission_policy = (
            ToolPermissionPolicy() if permission_policy is None else permission_policy
        )
        gateway._permission_context = (
            PermissionContext() if permission_context is None else permission_context
        )
        return gateway

    @property
    def exposed_names(self) -> tuple[str, ...]:
        """Return the names projected to the model for this Gateway view."""
        return self._exposed_names

    @property
    def catalog(self) -> tuple[BaseTool, ...]:
        """Return the complete Tool Catalog owned by this Gateway view."""
        return self._catalog

    @property
    def tool_context(self) -> ToolRunContext | None:
        """Return the immutable Workspace and runtime context for this view."""
        return self._tool_context

    def is_micro_compression_eligible(self, tool_name: str) -> bool:
        """Return whether one catalogued Tool result may be micro-compressed."""
        if not isinstance(tool_name, str):
            return False
        tool = self._tools.get(tool_name)
        return tool is not None and (
            tool_name in _MICRO_COMPRESSION_ELIGIBLE_BUILT_IN_NAMES or isinstance(tool, MCPTool)
        )

    def expose(self, names: Collection[str]) -> None:
        """Expose available Tools for subsequent model requests in this Run."""
        requested = _normalize_tool_names(names, label="Exposed Tool names")
        available_names = {tool.name for tool in self._catalog}
        if set(requested) - available_names:
            raise ValueError("Exposed Tool names must be available in the Run Catalog")
        exposed = set(self._exposed_names)
        exposed.update(requested)
        self._exposed_names = tuple(tool.name for tool in self._catalog if tool.name in exposed)

    @classmethod
    def _for_memory(
        cls,
        tools: tuple[BaseTool, ...],
        *,
        on_failure: Callable[[Exception], None] | None = None,
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        tool_context: ToolRunContext | None = None,
    ) -> ToolGateway:
        """Build the isolated Long-term Memory catalog without widening the public API."""
        if not tools or len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Memory Tool names must be unique and non-empty")
        catalog: tuple[BaseTool, ...] = tools
        return cls._from_catalog(
            catalog,
            exposed_names=tuple(tool.name for tool in catalog),
            on_failure=on_failure,
            permission_policy=permission_policy,
            permission_context=permission_context,
            tool_context=tool_context,
        )

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Build a detached schema list from each Tool in fixed Catalog order."""
        exposed = set(self._exposed_names)
        return [tool.to_schema() for tool in self._catalog if tool.name in exposed]

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: ConfirmationRequester | None = None,
        file_mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
    ) -> ToolResult:
        trace_capture = _start_tool_trace(tool_call)
        try:
            result = await self._call_untraced(
                tool_call,
                confirmation=confirmation,
                file_mutation_recorder=file_mutation_recorder,
                run_token=run_token,
                trace_capture=trace_capture,
            )
        except (asyncio.CancelledError, ConfirmationAborted) as error:
            if trace_capture is not None:
                try:
                    trace_capture.finish_cancelled(error)
                except Exception:
                    logger.warning("Execution Trace Tool cancellation could not be captured")
            raise
        except Exception as error:
            if trace_capture is not None:
                try:
                    trace_capture.finish_unexpected(error)
                except Exception:
                    logger.warning("Execution Trace Tool failure could not be captured")
            raise
        if trace_capture is not None:
            try:
                result_trace = _ACTIVE_TOOL_RESULT_TRACE.get()
                if result_trace is None:
                    trace_capture.finish_result(result)
                else:
                    result_trace._capture = trace_capture
            except Exception:
                logger.warning("Execution Trace Tool result could not be captured")
        return result

    async def _call_untraced(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: ConfirmationRequester | None = None,
        file_mutation_recorder: FileMutationRecorder | None = None,
        run_token: UUID | None = None,
        trace_capture: _ToolTraceCapture | None,
    ) -> ToolResult:
        """Parse, prepare, confirm when needed, execute, and normalize one call."""
        raw_arguments = tool_call.arguments
        if not isinstance(raw_arguments, str):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        if not isinstance(parsed, dict):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        if trace_capture is not None:
            trace_capture.set_parsed_arguments(cast(dict[str, Any], parsed))

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")
        if trace_capture is not None:
            trace_capture.tool_resolved = True

        try:
            if self._tool_context is None:
                facts = await tool.prepare(cast(dict[str, Any], parsed))
            else:
                facts = await tool.prepare_for_context(
                    cast(dict[str, Any], parsed),
                    context=self._tool_context,
                )
            if not isinstance(facts, ToolInvocationFacts):
                raise TypeError("Tool preparation returned an invalid value")
            execution_arguments = facts.execution_arguments
            mutation_target = next(
                (access.path for access in facts.file_accesses if access.role == "write"),
                None,
            )
            if trace_capture is not None:
                trace_capture.mark_arguments_prepared()
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))

        try:
            refusal = self._refusal_reason(
                tool,
                execution_arguments,
                mutation_target=mutation_target,
                tool_context=self._tool_context,
            )
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))
        if refusal is not None:
            return _result(tool_call, "refused", refusal)

        try:
            authorization = self._permission_policy.open(facts, self._permission_context)
            authorization_decision = authorization.initial_decision()
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))

        if authorization_decision not in {"direct", "confirm"}:
            return _result(tool_call, "error", "Tool authorization returned an invalid decision.")

        confirmation_state: list[ToolConfirmationMetadata | None] = [None]

        async def request_confirmation(reason: str) -> NetworkConfirmationDecision:
            request = ConfirmationRequest(
                confirmation_id=uuid4(),
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason=reason,
                summary=f"Confirm {tool.name}"[:240],
                details=facts.normalized_arguments,
                mcp_identity=facts.mcp_identity,
            )
            if confirmation is None:
                confirmation_state[0] = ToolConfirmationMetadata(request=request, decision=None)
                raise ToolAuthorizationFailure("unavailable")
            try:
                decision = await confirmation(request)
            except asyncio.CancelledError:
                raise
            except ConfirmationAborted:
                raise
            except ConfirmationUnavailable as error:
                confirmation_state[0] = ToolConfirmationMetadata(request=request, decision=None)
                raise ToolAuthorizationFailure("unavailable") from error
            except Exception as error:
                self._record_unexpected_failure(tool, error)
                confirmation_state[0] = ToolConfirmationMetadata(request=request, decision=None)
                raise ToolAuthorizationFailure("invalid") from error
            if decision not in {"approved", "declined"}:
                confirmation_state[0] = ToolConfirmationMetadata(request=request, decision=None)
                raise ToolAuthorizationFailure("invalid")
            confirmation_state[0] = ToolConfirmationMetadata(request=request, decision=decision)
            return decision

        if authorization_decision == "direct":
            _bind_network_confirmation(
                authorization,
                request_confirmation,
                already_approved=False,
            )
            return await self._execute(
                tool_call,
                tool,
                execution_arguments,
                authorization=authorization,
                confirmation_state=confirmation_state,
                file_mutation_recorder=file_mutation_recorder,
                run_token=run_token,
                mutation_target=mutation_target,
                tool_context=self._tool_context,
                trace_capture=trace_capture,
            )

        try:
            reason = authorization.confirmation_reason()
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))

        try:
            confirmation_decision = await request_confirmation(reason)
        except ToolAuthorizationFailure as error:
            metadata = confirmation_state[0]
            if metadata is None:
                return _result(tool_call, "error", _generic_tool_failure(tool.name))
            return _refused_confirmation_result(
                tool_call,
                _authorization_failure_message(error),
                request=metadata.request,
                decision=metadata.decision,
            )
        if confirmation_decision == "declined":
            metadata = confirmation_state[0]
            if metadata is None:
                return _result(tool_call, "error", _generic_tool_failure(tool.name))
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation was declined.",
                request=metadata.request,
                decision=metadata.decision,
            )
        _bind_network_confirmation(
            authorization,
            request_confirmation,
            already_approved=True,
        )
        return await self._execute(
            tool_call,
            tool,
            execution_arguments,
            authorization=authorization,
            confirmation_state=confirmation_state,
            file_mutation_recorder=file_mutation_recorder,
            run_token=run_token,
            mutation_target=mutation_target,
            tool_context=self._tool_context,
            trace_capture=trace_capture,
        )

    @staticmethod
    def _refusal_reason(
        tool: BaseTool,
        prepared_arguments: dict[str, Any],
        *,
        mutation_target: Path | None,
        tool_context: ToolRunContext | None = None,
    ) -> str | None:
        if (
            tool_context is not None
            and mutation_target is not None
            and isinstance(tool, (WriteFileTool, EditFileTool))
            and is_protected_restore_target(tool_context.workspace, mutation_target)
        ):
            return "Built-in File Tools cannot write to protected restore state."
        refusal_arguments = deepcopy(prepared_arguments)
        if mutation_target is not None and isinstance(tool, (WriteFileTool, EditFileTool)):
            refusal_arguments["path"] = str(mutation_target)
        reason: object
        if tool_context is not None:
            reason = tool.refusal_reason_for_context(refusal_arguments, context=tool_context)
        else:
            refusal = getattr(tool, "refusal_reason", None)
            if refusal is None:
                return None
            reason = cast(Callable[..., object], refusal)(**refusal_arguments)
        if reason is not None and not isinstance(reason, str):
            raise TypeError("Tool refusal checks must return a string reason or None")
        return reason

    async def _execute(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        prepared_arguments: dict[str, Any],
        *,
        authorization: ToolAuthorizationSession,
        confirmation_state: list[ToolConfirmationMetadata | None],
        file_mutation_recorder: FileMutationRecorder | None,
        run_token: UUID | None,
        mutation_target: Path | None,
        tool_context: ToolRunContext | None,
        trace_capture: _ToolTraceCapture | None = None,
    ) -> ToolResult:
        try:
            tool_scope = (
                None
                if trace_capture is None
                else TraceToolCallScope(
                    trace_capture.scope.recorder,
                    trace_capture.handle,
                )
            )
            with trace_tool_call(tool_scope):
                if tool_context is not None:
                    contextual_recorder = (
                        file_mutation_recorder
                        if self._permission_context.origin == "foreground"
                        else None
                    )
                    content = await tool.execute_authorized_for_context(
                        deepcopy(prepared_arguments),
                        authorization,
                        context=tool_context,
                        mutation_recorder=contextual_recorder,
                        run_token=run_token,
                        mutation_target=mutation_target,
                    )
                else:
                    content = await tool.execute_authorized(
                        deepcopy(prepared_arguments),
                        authorization,
                    )
            if not isinstance(content, str):
                raise TypeError("Tool execution must return a string")
        except asyncio.CancelledError:
            raise
        except ConfirmationAborted:
            raise
        except ToolAuthorizationFailure as error:
            return _result(
                tool_call,
                "refused",
                _authorization_failure_message(error),
                confirmation=confirmation_state[0],
            )
        except ToolError as error:
            if self._failure_observer is not None:
                self._failure_observer(error)
            if trace_capture is not None:
                trace_capture.timed_out = error.timed_out
            return _result(
                tool_call,
                "error",
                error.message,
                confirmation=confirmation_state[0],
            )
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(
                tool_call,
                "error",
                _generic_tool_failure(tool.name),
                confirmation=confirmation_state[0],
            )
        return _result(tool_call, "success", content, confirmation=confirmation_state[0])

    def _record_unexpected_failure(self, tool: BaseTool, error: Exception) -> None:
        if self._failure_observer is None:
            logger.opt(exception=error).error(
                "Tool execution failed name={} type={}",
                tool.name,
                type(error).__name__,
            )
            return
        self._failure_observer(error)


def _context_from_snapshot(
    context: PermissionContext,
    snapshot: PermissionSnapshot,
) -> PermissionContext:
    if not isinstance(snapshot, PermissionSnapshot):
        raise TypeError("Run permission snapshot must be a PermissionSnapshot")
    workspace_root = context.workspace_root
    if workspace_root is None:
        raise ValueError("Run permission snapshots require a workspace root")
    return PermissionContext.from_snapshot(
        snapshot,
        workspace_root=workspace_root,
        origin=context.origin,
        configured_schedule_level=context.configured_schedule_level,
    )


def _generic_tool_failure(tool_name: str) -> str:
    return f"{tool_name} could not complete the request."


def _bind_network_confirmation(
    authorization: ToolAuthorizationSession,
    requester: Callable[[str], Awaitable[NetworkConfirmationDecision]],
    *,
    already_approved: bool,
) -> None:
    binder = getattr(authorization, "bind_confirmation_requester", None)
    if callable(binder):
        binder(requester, already_approved=already_approved)


def _authorization_failure_message(error: ToolAuthorizationFailure) -> str:
    if error.outcome == "unavailable":
        return "Tool confirmation is unavailable."
    if error.outcome == "declined":
        return "Tool confirmation was declined."
    return "Tool confirmation was expired or invalid."


def _refused_confirmation_result(
    tool_call: ModelToolCall,
    content: str,
    *,
    request: ConfirmationRequest,
    decision: ConfirmationOutcome,
) -> ToolResult:
    return _result(
        tool_call,
        "refused",
        content,
        confirmation=ToolConfirmationMetadata(request=request, decision=decision),
    )


def _result(
    tool_call: ModelToolCall,
    status: ToolResultStatus,
    content: str,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status=status,
        content=content,
        artifact=None,
        confirmation=confirmation,
    )


__all__ = [
    "BUILT_IN_TOOL_NAMES",
    "BuiltInToolCatalog",
    "ConfirmationDecision",
    "ConfirmationRequest",
    "ConfirmationRequester",
    "ModelToolCall",
    "ToolConfirmationMetadata",
    "ToolGateway",
    "ToolResult",
    "ToolResultStatus",
]
