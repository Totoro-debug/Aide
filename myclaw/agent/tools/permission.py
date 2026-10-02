"""Run-local Tool authorization contracts and the single permission policy."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from myclaw.agent.permission import (
    PERMISSION_LEVELS,
    PermissionSnapshot,
    RuntimePermissionControl,
    ToolPermissionLevel,
    validate_permission_level,
)
from myclaw.agent.tools.core.exec_policy import (
    EXEC_CATASTROPHIC_REASON,
    EXEC_CONFIRMATION_REASON,
    EXEC_DESTRUCTIVE_REASON,
    ExecAssessment,
    ExecGrammarClassification,
    ExecPathAccess,
    ResolvedExecShell,
    classify_powershell_command,
)
from myclaw.utils.host_filesystem import host_path_is_within

type PermissionDecision = Literal["direct", "confirm"]
type ToolRunOrigin = Literal["foreground", "schedule", "memory"]
type FileAccessRole = Literal["read", "write"]
type ScheduleActionName = Literal["list", "add", "remove"]
type IPAddress = str
type NetworkTargetRisk = Literal[
    "literal_non_global",
    "dns_failure",
    "dns_empty",
    "dns_non_global",
]
type NetworkConfirmationDecision = Literal["approved", "declined"]
type NetworkConfirmationRequester = Callable[
    [str], Awaitable[NetworkConfirmationDecision]
]


@dataclass(frozen=True, slots=True)
class ScheduleAction:
    """Typed action facts emitted by the normalized Schedule Tool call."""

    action: ScheduleActionName

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action not in {"list", "add", "remove"}:
            raise ValueError("Schedule action is invalid")


@dataclass(frozen=True, slots=True)
class NormalizedNetworkTarget:
    """Immutable URL facts used to authorize one concrete network hop."""

    url: str
    scheme: Literal["http", "https"]
    host: str
    port: int

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("Network target URL must be a non-empty string")
        if self.scheme not in {"http", "https"}:
            raise ValueError("Network target scheme is invalid")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("Network target host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("Network target port is invalid")


@dataclass(frozen=True, slots=True)
class NetworkAssessment:
    """Static facts for one network target discovered during preparation."""

    target: NormalizedNetworkTarget
    static_risk: NetworkTargetRisk | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, NormalizedNetworkTarget):
            raise TypeError("Network assessment target is invalid")
        if self.static_risk is not None and self.static_risk not in {
            "literal_non_global",
            "dns_failure",
            "dns_empty",
            "dns_non_global",
        }:
            raise ValueError("Network assessment risk is invalid")


class ToolAuthorizationFailure(Exception):
    """A confirmation outcome that must remain a canonical refused Tool result."""

    def __init__(self, outcome: Literal["declined", "unavailable", "invalid"]) -> None:
        self.outcome = outcome
        super().__init__(outcome)


@dataclass(frozen=True, slots=True)
class FileAccess:
    """One canonical host path access observed by a File Tool."""

    path: Path
    role: FileAccessRole
    base: Path
    workspace_root: Path
    allowed_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("File access path must be a Path")
        if self.role not in {"read", "write"}:
            raise ValueError("File access role is invalid")
        if not isinstance(self.base, Path) or not isinstance(self.workspace_root, Path):
            raise TypeError("File access roots must be Paths")
        if any(not isinstance(root, Path) for root in self.allowed_roots):
            raise TypeError("File access allowed roots must be Paths")
        object.__setattr__(self, "allowed_roots", tuple(self.allowed_roots))


@dataclass(frozen=True, slots=True)
class MCPToolIdentity:
    """Stable identity for one discovered MCP Tool invocation."""

    server_name: str
    remote_name: str
    model_name: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.server_name, "server_name"),
            (self.remote_name, "remote_name"),
            (self.model_name, "model_name"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"MCP Tool identity {field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "server_name": self.server_name,
            "remote_name": self.remote_name,
            "model_name": self.model_name,
        }


def canonicalize_file_access(
    *,
    workspace: Path,
    base: Path,
    requested: str | Path,
    role: FileAccessRole,
) -> FileAccess:
    """Resolve a model path using host links and the nearest existing ancestor."""
    if not isinstance(workspace, Path) or not isinstance(base, Path):
        raise TypeError("File access workspace and base must be Paths")
    if not isinstance(requested, (str, Path)):
        raise TypeError("File access requested path must be a string or Path")

    workspace_root = workspace.resolve(strict=True)
    base_root = base.resolve(strict=True)
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = base_root / candidate
    candidate = Path(os.path.abspath(str(candidate)))
    canonical = _resolve_with_missing_ancestor(candidate)
    return FileAccess(
        path=canonical,
        role=role,
        base=base_root,
        workspace_root=workspace_root,
    )


def _resolve_with_missing_ancestor(candidate: Path) -> Path:
    missing: list[str] = []
    current = candidate
    while not os.path.lexists(str(current)):
        parent = current.parent
        if parent == current:
            raise OSError(f"could not resolve path ancestor: {candidate}")
        missing.append(current.name)
        current = parent

    resolved = current.resolve(strict=True)
    for name in reversed(missing):
        resolved /= name
    return resolved


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """Run facts supplied to a Tool Permission Policy."""

    level: ToolPermissionLevel | None = None
    configured_schedule_level: ToolPermissionLevel | None = None
    origin: ToolRunOrigin = "foreground"
    workspace_root: Path | None = None
    exec_shell: ResolvedExecShell | None = None
    snapshot: PermissionSnapshot | None = None

    def __post_init__(self) -> None:
        if self.level is not None:
            validate_permission_level(self.level)
        if self.configured_schedule_level is not None:
            validate_permission_level(self.configured_schedule_level)
        if self.origin not in {"foreground", "schedule", "memory"}:
            raise ValueError("Tool run origin is invalid")
        if self.workspace_root is not None and not isinstance(self.workspace_root, Path):
            raise TypeError("Permission context workspace_root must be a Path or None")
        if self.exec_shell is not None and not isinstance(self.exec_shell, ResolvedExecShell):
            raise TypeError("Permission context exec_shell must be resolved or None")
        if self.snapshot is not None:
            if self.level is not None and self.level != self.snapshot.level:
                raise ValueError("Permission context level must match its snapshot")
            snapshot_shell = cast(ResolvedExecShell, self.snapshot.exec_shell)
            if self.exec_shell is not None and self.exec_shell != snapshot_shell:
                raise ValueError("Permission context Exec shell must match its snapshot")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PermissionSnapshot,
        *,
        workspace_root: Path,
        origin: ToolRunOrigin = "foreground",
        configured_schedule_level: ToolPermissionLevel | None = None,
    ) -> PermissionContext:
        if not isinstance(snapshot, PermissionSnapshot):
            raise TypeError("Permission context requires a PermissionSnapshot")
        return cls(
            level=snapshot.level,
            configured_schedule_level=configured_schedule_level,
            origin=origin,
            workspace_root=workspace_root,
            exec_shell=cast(ResolvedExecShell, snapshot.exec_shell),
            snapshot=snapshot,
        )


@dataclass(frozen=True, slots=True, init=False)
class ToolInvocationFacts:
    """Detached facts for one normalized Tool invocation."""

    tool_name: str
    _normalized_arguments: dict[str, Any] = field(repr=False)
    _execution_arguments: dict[str, Any] = field(repr=False)
    file_accesses: tuple[FileAccess, ...]
    exec_assessment: ExecAssessment | None
    schedule_action: ScheduleAction | None
    network_targets: tuple[NetworkAssessment, ...]
    mcp_identity: MCPToolIdentity | None

    def __init__(
        self,
        tool_name: str,
        normalized_arguments: Mapping[str, Any],
        *,
        execution_arguments: Mapping[str, Any] | None = None,
        file_accesses: tuple[FileAccess, ...] = (),
        exec_assessment: ExecAssessment | None = None,
        schedule_action: ScheduleAction | None = None,
        network_targets: tuple[NetworkAssessment, ...] = (),
        mcp_identity: MCPToolIdentity | None = None,
    ) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError("Tool invocation tool_name must be a non-empty string")
        if not isinstance(normalized_arguments, Mapping):
            raise TypeError("Tool invocation normalized_arguments must be a mapping")
        if any(not isinstance(name, str) for name in normalized_arguments):
            raise TypeError("Tool invocation argument names must be strings")
        execution_source = normalized_arguments if execution_arguments is None else execution_arguments
        if not isinstance(execution_source, Mapping):
            raise TypeError("Tool invocation execution_arguments must be a mapping")
        if any(not isinstance(name, str) for name in execution_source):
            raise TypeError("Tool invocation execution argument names must be strings")
        if mcp_identity is not None and not isinstance(mcp_identity, MCPToolIdentity):
            raise TypeError("Tool invocation MCP identity must be MCPToolIdentity or None")
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "_normalized_arguments", deepcopy(dict(normalized_arguments)))
        object.__setattr__(self, "_execution_arguments", deepcopy(dict(execution_source)))
        object.__setattr__(self, "file_accesses", tuple(deepcopy(file_accesses)))
        object.__setattr__(self, "exec_assessment", deepcopy(exec_assessment))
        object.__setattr__(self, "schedule_action", deepcopy(schedule_action))
        object.__setattr__(self, "network_targets", tuple(deepcopy(network_targets)))
        object.__setattr__(self, "mcp_identity", deepcopy(mcp_identity))

    @property
    def normalized_arguments(self) -> dict[str, Any]:
        """Return a detached view of the normalized argument object."""
        return deepcopy(self._normalized_arguments)

    @property
    def execution_arguments(self) -> dict[str, Any]:
        """Return the prepared argument object used by the Tool execution seam."""
        return deepcopy(self._execution_arguments)

class ToolAuthorizationSession(Protocol):
    """One authorization state machine owned by one Gateway call."""

    def initial_decision(self) -> PermissionDecision: ...

    def confirmation_reason(self) -> str: ...

    exec_assessment: ExecAssessment | None

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None: ...


class ToolPermissionPolicy:
    """Map normalized invocation facts to a per-call authorization session."""

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        """Classify one detached invocation without retaining mutable run state."""
        if facts.schedule_action is not None:
            return _classify_schedule_invocation(facts, context)
        if facts.mcp_identity is not None:
            if context.origin == "memory":
                return _DecisionAuthorizationSession("direct")
            if context.origin == "schedule" and context.snapshot is None:
                return _DecisionAuthorizationSession("direct")
            if context.level is None:
                return _DecisionAuthorizationSession("direct")
            if context.level == "full-access":
                return _DecisionAuthorizationSession("direct")
            return _DecisionAuthorizationSession(
                "confirm",
                _mcp_confirmation_reason(facts.mcp_identity),
            )
        if (
            facts.tool_name == "exec"
            and facts.exec_assessment is not None
            and (
                context.origin == "foreground"
                or (
                    context.origin == "schedule"
                    and context.snapshot is not None
                    and context.configured_schedule_level is not None
                )
            )
        ):
            return _classify_exec_invocation(facts, context)
        if facts.network_targets and context.origin != "memory":
            return _NetworkAuthorizationSession(facts.network_targets, context.level)
        if facts.file_accesses and context.origin != "memory":
            decision, reason = _classify_file_accesses(facts.file_accesses, context)
            return _DecisionAuthorizationSession(decision, reason)
        return _DecisionAuthorizationSession("direct")


class _NetworkAuthorizationSession:
    """Authorize Web Fetch hops while retaining one call-local approval state."""

    def __init__(
        self,
        assessments: tuple[NetworkAssessment, ...],
        level: ToolPermissionLevel | None,
    ) -> None:
        self._level = level
        self.exec_assessment: ExecAssessment | None = None
        self._requester: NetworkConfirmationRequester | None = None
        self._approved = False
        static_risk = next(
            (assessment.static_risk for assessment in assessments if assessment.static_risk is not None),
            None,
        )
        self._initial_reason = _network_confirmation_reason(
            addresses=(),
            risk=static_risk,
        )
        static_unsafe = any(item.static_risk is not None for item in assessments)
        self._initial_decision: PermissionDecision = (
            "confirm"
            if level != "full-access" and static_unsafe
            else "direct"
        )

    def initial_decision(self) -> PermissionDecision:
        return self._initial_decision

    def confirmation_reason(self) -> str:
        return self._initial_reason

    def bind_confirmation_requester(
        self,
        requester: NetworkConfirmationRequester,
        *,
        already_approved: bool,
    ) -> None:
        self._requester = requester
        self._approved = already_approved

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None:
        if self._level == "full-access":
            return
        if _addresses_are_public(resolved_addresses) or self._approved:
            return
        if self._requester is None:
            raise ToolAuthorizationFailure("unavailable")
        decision = await self._requester(
            _network_confirmation_reason(addresses=resolved_addresses)
        )
        if decision == "approved":
            self._approved = True
            return
        if decision == "declined":
            raise ToolAuthorizationFailure("declined")
        raise ToolAuthorizationFailure("invalid")


def _addresses_are_public(addresses: tuple[IPAddress, ...]) -> bool:
    if not addresses:
        return False
    # BaseTool imports this module, so defer the canonical classifier import
    # until authorization rather than maintaining a second policy copy here.
    from myclaw.agent.tools.base import is_public_ip

    return all(is_public_ip(address) for address in addresses)


def _network_confirmation_reason(
    *,
    addresses: tuple[IPAddress, ...],
    risk: NetworkTargetRisk | None = None,
    subject: str = "Web Fetch target",
) -> str:
    if not addresses and risk not in {"literal_non_global", "dns_non_global"}:
        return (
            f"{subject} DNS resolution is unavailable or returned no addresses "
            "and requires confirmation."
        )
    return f"{subject} resolves to a private or non-global address and requires confirmation."


class _DecisionAuthorizationSession:
    def __init__(
        self,
        decision: PermissionDecision,
        reason: str | None = None,
        *,
        exec_assessment: ExecAssessment | None = None,
    ) -> None:
        self._decision = decision
        self._reason = reason if reason is not None else "Tool confirmation is required."
        self.exec_assessment = exec_assessment

    def initial_decision(self) -> PermissionDecision:
        return self._decision

    def confirmation_reason(self) -> str:
        return self._reason

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None:
        del target, resolved_addresses


def _mcp_confirmation_reason(identity: MCPToolIdentity) -> str:
    return (
        "MCP Tool "
        f"server={identity.server_name} remote_tool={identity.remote_name} "
        f"model_tool={identity.model_name} requires confirmation for this call."
    )


_SCHEDULE_CRUD_CONFIRMATION_REASON = (
    "Schedule add/remove changes persistent scheduled work and requires confirmation."
)


def _classify_schedule_invocation(
    facts: ToolInvocationFacts,
    context: PermissionContext,
) -> ToolAuthorizationSession:
    action = facts.schedule_action
    if action is None or context.origin != "foreground":
        return _DecisionAuthorizationSession("direct")
    if action.action == "list":
        return _DecisionAuthorizationSession("direct")
    if context.level is None:
        return _DecisionAuthorizationSession("direct")

    reasons: list[str] = []
    if context.level == "read-only":
        reasons.append(_SCHEDULE_CRUD_CONFIRMATION_REASON)

    configured = context.configured_schedule_level
    if (
        action.action == "add"
        and configured is not None
        and PERMISSION_LEVELS.index(configured) > PERMISSION_LEVELS.index(context.level)
    ):
        reasons.append(
            "Schedule add creates a future Job with configured Schedule level "
            f"'{configured}' above current foreground level '{context.level}' "
            "and requires confirmation."
        )

    if not reasons:
        return _DecisionAuthorizationSession("direct")
    return _DecisionAuthorizationSession("confirm", _merge_confirmation_reasons(reasons))


def _merge_confirmation_reasons(reasons: list[str]) -> str:
    """Keep one stable, duplicate-free reason string for one Tool call."""
    return " ".join(dict.fromkeys(reason for reason in reasons if reason))


def _classify_exec_invocation(
    facts: ToolInvocationFacts,
    context: PermissionContext,
) -> ToolAuthorizationSession:
    assessment = facts.exec_assessment
    if assessment is None:
        return _DecisionAuthorizationSession(
            "confirm",
            "Exec command facts are incomplete.",
        )
    if assessment.catastrophic_matches:
        return _DecisionAuthorizationSession(
            "confirm",
            EXEC_CATASTROPHIC_REASON,
            exec_assessment=assessment,
        )
    if assessment.uncertain or assessment.syntax_confidence != "high":
        return _DecisionAuthorizationSession(
            "confirm",
            EXEC_CONFIRMATION_REASON,
            exec_assessment=assessment,
        )
    if context.level is None and assessment.destructive_matches:
        return _DecisionAuthorizationSession(
            "confirm",
            EXEC_DESTRUCTIVE_REASON,
            exec_assessment=assessment,
        )

    if context.level != "full-access":
        network_risk = next(
            (
                item.static_risk
                for item in facts.network_targets
                if item.static_risk is not None
            ),
            None,
        )
        if network_risk is not None:
            return _DecisionAuthorizationSession(
                "confirm",
                _network_confirmation_reason(
                    addresses=(),
                    risk=network_risk,
                    subject="Exec URL",
                ),
                exec_assessment=assessment,
            )

    shell = context.exec_shell
    if shell is None:
        if context.level is None:
            if facts.file_accesses:
                path_decision, path_reason = _classify_file_accesses(
                    facts.file_accesses,
                    context,
                )
                if path_decision == "confirm":
                    return _DecisionAuthorizationSession(
                        "confirm",
                        path_reason,
                        exec_assessment=assessment,
                    )
            return _DecisionAuthorizationSession("direct", exec_assessment=assessment)
        return _DecisionAuthorizationSession(
            "confirm",
            "Exec shell facts are incomplete.",
            exec_assessment=assessment,
        )
    if context.level == "full-access":
        return _DecisionAuthorizationSession("direct", exec_assessment=assessment)

    workspace_root = context.workspace_root
    if workspace_root is not None and any(
        identity.requested.casefold() in {"git", "git.exe"}
        and _is_workspace_git_identity(identity, workspace_root)
        for identity in assessment.command_identities
    ):
        return _DecisionAuthorizationSession(
            "confirm",
            "The Git executable resolves inside the Workspace and requires confirmation.",
            exec_assessment=assessment,
        )

    arguments = facts.normalized_arguments
    command = arguments.get("command")
    if not isinstance(command, str):
        return _DecisionAuthorizationSession(
            "confirm",
            "Exec command facts are incomplete.",
            exec_assessment=assessment,
        )
    grammar = classify_powershell_command(command, assessment)
    return _classify_exec_grammar_result(
        grammar,
        facts=facts,
        context=context,
        assessment=assessment,
    )


def _classify_exec_grammar_result(
    grammar: ExecGrammarClassification,
    *,
    facts: ToolInvocationFacts,
    context: PermissionContext,
    assessment: ExecAssessment,
) -> ToolAuthorizationSession:
    if not grammar.accepted:
        return _DecisionAuthorizationSession(
            "confirm",
            grammar.reason,
            exec_assessment=assessment,
        )

    path_decision, path_reason = _classify_exec_paths(
        grammar.file_accesses,
        facts=facts,
        context=context,
    )
    if path_decision == "confirm":
        return _DecisionAuthorizationSession(
            "confirm",
            path_reason,
            exec_assessment=assessment,
        )
    if context.level == "read-only" and any(
        access.role == "write" for access in grammar.file_accesses
    ):
        return _DecisionAuthorizationSession(
            "confirm",
            "Write access requires confirmation in read-only mode.",
            exec_assessment=assessment,
        )
    return _DecisionAuthorizationSession("direct", exec_assessment=assessment)


def _is_workspace_git_identity(identity: object, workspace_root: Path) -> bool:
    resolved = getattr(identity, "resolved", None)
    if not isinstance(resolved, str) or not os.path.isabs(resolved):
        return True
    try:
        canonical = Path(resolved).resolve(strict=False)
        root = workspace_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return True
    return host_path_is_within(canonical, root)


def _classify_exec_paths(
    accesses: tuple[ExecPathAccess, ...],
    *,
    facts: ToolInvocationFacts,
    context: PermissionContext,
) -> tuple[PermissionDecision, str | None]:
    if context.level == "full-access":
        return "direct", None
    workspace_root = context.workspace_root
    cwd = facts.normalized_arguments.get("cwd")
    if workspace_root is None or not isinstance(cwd, str):
        return "confirm", "Exec path facts are incomplete and require confirmation."
    base = Path(cwd)
    try:
        canonical_cwd = canonicalize_file_access(
            workspace=workspace_root,
            base=base,
            requested=".",
            role="read",
        )
    except (OSError, RuntimeError, ValueError):
        return "confirm", "Exec working directory could not be classified."
    if not host_path_is_within(canonical_cwd.path, canonical_cwd.workspace_root):
        return "confirm", "The Exec working directory is outside the Workspace."
    for access in accesses:
        candidate = Path(access.path)
        if not candidate.is_absolute():
            candidate = base / candidate
        if access.role == "read" and not os.path.lexists(str(candidate)):
            return "confirm", "An Exec read path has an unknown parent and requires confirmation."
        try:
            canonical = canonicalize_file_access(
                workspace=workspace_root,
                base=base,
                requested=access.path,
                role=access.role,
            )
        except (OSError, RuntimeError, ValueError):
            return "confirm", "Exec path could not be classified and requires confirmation."
        if not host_path_is_within(canonical.path, canonical.workspace_root):
            return (
                "confirm",
                "The requested path resolves outside the Workspace and requires confirmation.",
            )
    return "direct", None


def _classify_file_accesses(
    accesses: tuple[FileAccess, ...],
    context: PermissionContext,
) -> tuple[PermissionDecision, str | None]:
    level = context.level
    has_external = any(
        not host_path_is_within(access.path, access.workspace_root)
        and not _is_schedule_allowed_file_access(access, context)
        for access in accesses
    )
    has_write = any(access.role == "write" for access in accesses)

    if level == "full-access":
        return "direct", None

    reasons: list[str] = []
    if level == "read-only" and has_write:
        reasons.append("Write access requires confirmation in read-only mode.")
    if has_external:
        reasons.append(
            "The requested path resolves outside the Workspace and requires confirmation."
        )
    if reasons:
        return "confirm", _merge_confirmation_reasons(reasons)
    return "direct", None


def _is_schedule_allowed_file_access(
    access: FileAccess,
    context: PermissionContext,
) -> bool:
    if context.origin != "schedule" or context.snapshot is not None:
        return False
    return any(host_path_is_within(access.path, root) for root in access.allowed_roots)


__all__ = [
    "PERMISSION_LEVELS",
    "ExecAssessment",
    "FileAccess",
    "FileAccessRole",
    "IPAddress",
    "MCPToolIdentity",
    "NetworkAssessment",
    "NetworkConfirmationDecision",
    "NetworkConfirmationRequester",
    "NetworkTargetRisk",
    "NormalizedNetworkTarget",
    "PermissionContext",
    "PermissionDecision",
    "PermissionSnapshot",
    "ResolvedExecShell",
    "RuntimePermissionControl",
    "ScheduleAction",
    "ScheduleActionName",
    "ToolAuthorizationFailure",
    "ToolAuthorizationSession",
    "ToolInvocationFacts",
    "ToolPermissionLevel",
    "ToolPermissionPolicy",
    "ToolRunOrigin",
    "canonicalize_file_access",
]
