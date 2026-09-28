"""The fixed Core Tool Catalog and its only invocation boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Collection, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from loguru import logger

from myclaw.agent.confirmation import ConfirmationAborted, ConfirmationUnavailable
from myclaw.agent.tools.base import (
    ArtifactReference,
    BaseTool,
    ToolError,
)
from myclaw.agent.tools.core.edit_file import EditFileTool
from myclaw.agent.tools.core.exec import ExecTool
from myclaw.agent.tools.core.exec_host import ExecHost
from myclaw.agent.tools.core.glob import GlobTool
from myclaw.agent.tools.core.grep import GrepTool
from myclaw.agent.tools.core.list_dir import ListDirTool
from myclaw.agent.tools.core.read_file import ReadFileTool
from myclaw.agent.tools.core.schedule import ScheduleTool
from myclaw.agent.tools.core.web_fetch import WebFetchTool
from myclaw.agent.tools.core.web_search import WebSearchTool
from myclaw.agent.tools.core.write_file import WriteFileTool
from myclaw.agent.tools.file_mutation import FileMutationRecorder
from myclaw.agent.tools.mcp import MCPTool
from myclaw.agent.tools.permission import (
    MCPToolIdentity,
    NetworkConfirmationDecision,
    PermissionContext,
    PermissionSnapshot,
    ToolAuthorizationFailure,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    ToolPermissionPolicy,
)
from myclaw.schedule.service import ScheduleService
from myclaw.utils.validation import require_uuid4

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


class ToolGateway:
    """Create and invoke the Built-in Tool Catalog."""

    def __init__(
        self,
        *,
        workspace: Path,
        schedule_service: ScheduleService,
        skill_root: Path | None = None,
        additional_tools: Sequence[BaseTool] = (),
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        exec_host: ExecHost | None = None,
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("Tool Gateway requires a Path")
        if not isinstance(schedule_service, ScheduleService):
            raise TypeError("Tool Gateway requires a ScheduleService")

        generation_tools = tuple(additional_tools)
        if any(not isinstance(tool, BaseTool) for tool in generation_tools):
            raise TypeError("Additional Tools must be BaseTool instances")
        tools: tuple[BaseTool, ...] = (
            ReadFileTool(workspace=workspace, skill_root=skill_root),
            WriteFileTool(workspace=workspace),
            EditFileTool(workspace=workspace),
            ListDirTool(workspace=workspace),
            GlobTool(workspace=workspace),
            GrepTool(workspace=workspace),
            ExecTool(workspace=workspace, host=exec_host),
            WebSearchTool(),
            WebFetchTool(),
            ScheduleTool(schedule_service=schedule_service),
            *generation_tools,
        )
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Tool names must be unique")
        self._catalog = tools
        self._tools = {tool.name: tool for tool in tools}
        self._exposed_names = tuple(tool.name for tool in tools)
        self._failure_observer: Callable[[Exception], None] | None = None
        self._permission_policy = (
            ToolPermissionPolicy() if permission_policy is None else permission_policy
        )
        self._permission_context = (
            PermissionContext(workspace_root=workspace)
            if permission_context is None
            else permission_context
        )

    def for_run(
        self,
        *,
        exposed_names: Collection[str],
        excluded_names: Collection[str] = (),
        run_tools: Sequence[BaseTool] = (),
        permission_policy: ToolPermissionPolicy | None = None,
        permission_context: PermissionContext | None = None,
        permission_snapshot: PermissionSnapshot | None = None,
    ) -> ToolGateway:
        """Create an isolated Run view over this Gateway's reusable Tool instances."""
        excluded = _normalize_tool_names(excluded_names, label="Excluded Tool names")
        excluded_set = set(excluded)
        additions = tuple(run_tools)
        if any(not isinstance(tool, BaseTool) for tool in additions):
            raise TypeError("Run Tools must be BaseTool instances")

        catalog = tuple(
            tool for tool in (*self._catalog, *additions) if tool.name not in excluded_set
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
        selected_context = (
            _context_from_snapshot(self._permission_context, permission_snapshot)
            if permission_snapshot is not None
            else (self._permission_context if permission_context is None else permission_context)
        )
        return self._from_catalog(
            catalog,
            exposed_names=exposure,
            on_failure=self._failure_observer,
            permission_policy=(
                self._permission_policy if permission_policy is None else permission_policy
            ),
            permission_context=selected_context,
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
    ) -> ToolGateway:
        gateway = object.__new__(cls)
        gateway._catalog = catalog
        gateway._tools = {tool.name: tool for tool in catalog}
        gateway._exposed_names = exposed_names
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

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")

        try:
            facts = await tool.prepare(cast(dict[str, Any], parsed))
            if not isinstance(facts, ToolInvocationFacts):
                raise TypeError("Tool preparation returned an invalid value")
            execution_arguments = facts.execution_arguments
            mutation_target = _file_mutation_target(facts)
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
        )

    @staticmethod
    def _refusal_reason(
        tool: BaseTool,
        prepared_arguments: dict[str, Any],
        *,
        mutation_target: Path | None,
    ) -> str | None:
        refusal = getattr(tool, "refusal_reason", None)
        if refusal is None:
            return None
        refusal_arguments = deepcopy(prepared_arguments)
        if mutation_target is not None and isinstance(tool, (WriteFileTool, EditFileTool)):
            refusal_arguments["path"] = str(mutation_target)
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
    ) -> ToolResult:
        try:
            if (
                self._permission_context.origin == "foreground"
                and file_mutation_recorder is not None
                and run_token is not None
                and mutation_target is not None
                and isinstance(tool, (WriteFileTool, EditFileTool))
            ):
                content = await tool.execute_authorized(
                    deepcopy(prepared_arguments),
                    authorization,
                    mutation_recorder=file_mutation_recorder,
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


def _file_mutation_target(facts: ToolInvocationFacts) -> Path | None:
    return next(
        (access.path for access in facts.file_accesses if access.role == "write"),
        None,
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
    "ConfirmationDecision",
    "ConfirmationRequest",
    "ConfirmationRequester",
    "ModelToolCall",
    "ToolConfirmationMetadata",
    "ToolGateway",
    "ToolResult",
    "ToolResultStatus",
]
