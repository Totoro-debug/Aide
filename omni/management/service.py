"""Concrete read-only views exposed through the Management Port."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Protocol

from loguru import logger

from omni import __version__
from omni.agent.context.budget import (
    CONTEXT_ESTIMATOR_VERSION,
    ContextBudget,
    ContextUsageSnapshot,
    ProjectionSource,
    estimate_request_tokens,
    project_next_request_tokens,
)
from omni.agent.memory.dream import DreamResult
from omni.agent.permission import (
    RuntimePermissionControl,
    ToolPermissionLevel,
    validate_permission_level,
)
from omni.agent.session.restore import RestoreMode, RestorePlan, RestoreResult
from omni.agent.session.session import (
    RestoreAnchor,
    Session,
    SessionStoragePartition,
)
from omni.agent.workspace_state import WorkspaceState
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader, ConfigView
from omni.errors import ErrorInfo
from omni.provider.models import REASONING_EFFORT_LEVELS, ReasoningEffort
from omni.provider.session_configuration import SessionModelConfiguration
from omni.skills.catalog import SkillMetadata
from omni.utils.host_filesystem import HOST_FILESYSTEM
from omni.utils.validation import require_nonnegative_int, require_nonnegative_number


class _MemoryReader(Protocol):
    async def read_long_term(self) -> str: ...


class _DreamRunner(Protocol):
    async def run(self) -> DreamResult: ...


class _ReasoningEffortControl(Protocol):
    @property
    def reasoning_effort(self) -> ReasoningEffort: ...

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None: ...


class _ManagementAgentLoop(Protocol):
    def runtime_status_input(self) -> "RuntimeStatusInput": ...

    def reload_skill(self) -> tuple[SkillMetadata, ...]: ...


@dataclass(frozen=True, slots=True)
class SessionListingEntry:
    """The fields needed to render one resumable Conversation Session."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    def __post_init__(self) -> None:
        Session._require_id(self.id, field="id", partition=SessionStoragePartition.FOREGROUND)
        if not self.title or " ".join(self.title.split()) != self.title or len(self.title) > 60:
            raise ValueError("title is not normalized")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        require_nonnegative_int(self.message_count, field="message_count")


@dataclass(frozen=True, slots=True)
class SessionListingReport:
    """Valid current-format Sessions and the entries skipped during listing."""

    sessions: tuple[SessionListingEntry, ...]
    skipped_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.skipped_count, bool)
            or not isinstance(self.skipped_count, int)
            or self.skipped_count < 0
        ):
            raise ValueError("skipped_count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class RestoreListingReport:
    """Persisted Restore Anchors for the current foreground Session."""

    session_id: str
    anchors: tuple[RestoreAnchor, ...]

    def __post_init__(self) -> None:
        Session._require_id(self.session_id, partition=SessionStoragePartition.FOREGROUND)
        if not isinstance(self.anchors, tuple) or not all(
            isinstance(anchor, RestoreAnchor) for anchor in self.anchors
        ):
            raise TypeError("anchors must be a tuple of RestoreAnchor values")


@dataclass(frozen=True, slots=True)
class RuntimeStatusInput:
    """Committed-Session baseline for the next independent Foreground Agent Run."""

    session_id: str = ""
    session_title: str = ""
    session_message_count: int = 0
    last_compacted: int = 0
    cumulative_usage: tuple[tuple[str, int], ...] = ()
    chat_model: str = ""
    chat_reasoning_effort: ReasoningEffort | None = None
    active_model_configuration: SessionModelConfiguration | None = None
    model_configuration_available: bool = True
    context_window: int = 0
    max_output: int = 0
    compact_ratio: float = 0.9
    requested_route: str = "chat"
    selected_route: str = "chat"
    provider_id: str = ""
    model: str = ""
    projected_messages: tuple[dict[str, Any], ...] = ()
    projected_tools: tuple[dict[str, Any], ...] = ()
    latest_usage_context: ContextUsageSnapshot | None = None
    latest_reported_usage: tuple[tuple[str, int], ...] = ()
    generation_started_at: float | None = None
    last_request_usage: dict[str, int | None] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """The required observable fields for the `/status` view."""

    version: str
    chat_model: str
    chat_reasoning_effort: ReasoningEffort
    uptime_seconds: int
    context_window: int
    max_output: int
    available_context: int
    compact_ratio: float
    compact_context_window: int
    projected_next_request_tokens: int
    projection_source: ProjectionSource
    input_budget_used_percent: float
    session_message_count: int
    last_compacted: int
    cumulative_usage: dict[str, int]
    schedule: dict[str, object] | None = None
    configured_permission_level: ToolPermissionLevel = "workspace-write"
    current_permission_level: ToolPermissionLevel = "workspace-write"
    active_model_configuration: SessionModelConfiguration | None = None
    model_configuration_available: bool = True
    last_request_usage: dict[str, int | None] | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.uptime_seconds, field="uptime_seconds")
        budget = ContextBudget(
            context_window=self.context_window,
            max_output=self.max_output,
            compact_ratio=self.compact_ratio,
        )
        if self.available_context != budget.available_context:
            raise ValueError("available_context does not match the route budget")
        if self.compact_context_window != budget.compact_context_window:
            raise ValueError("compact_context_window does not match the route budget")
        require_nonnegative_int(
            self.projected_next_request_tokens,
            field="projected_next_request_tokens",
        )
        if self.projection_source not in {"estimated", "reported_delta"}:
            raise ValueError("projection_source is invalid")
        require_nonnegative_number(
            self.input_budget_used_percent,
            field="input_budget_used_percent",
        )
        require_nonnegative_int(self.session_message_count, field="session_message_count")
        require_nonnegative_int(self.last_compacted, field="last_compacted")
        validate_permission_level(self.configured_permission_level)
        validate_permission_level(self.current_permission_level)
        if self.last_request_usage is not None:
            if set(self.last_request_usage) != {"input_tokens", "cached_input_tokens"}:
                raise ValueError("last_request_usage fields are invalid")
            input_tokens = self.last_request_usage["input_tokens"]
            if input_tokens is None:
                raise ValueError("last_request_usage.input_tokens must be an integer")
            require_nonnegative_int(input_tokens, field="last_request_usage.input_tokens")
            cached = self.last_request_usage["cached_input_tokens"]
            if cached is not None:
                require_nonnegative_int(cached, field="last_request_usage.cached_input_tokens")
                if cached > input_tokens:
                    raise ValueError("cached_input_tokens must not exceed input_tokens")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "version": self.version,
            "chat_model": self.chat_model,
            "chat_reasoning_effort": self.chat_reasoning_effort,
            "uptime_seconds": self.uptime_seconds,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "available_context": self.available_context,
            "compact_ratio": self.compact_ratio,
            "compact_context_window": self.compact_context_window,
            "projected_next_request_tokens": self.projected_next_request_tokens,
            "projection_source": self.projection_source,
            "input_budget_used_percent": self.input_budget_used_percent,
            "session_message_count": self.session_message_count,
            "last_compacted": self.last_compacted,
            "configured_permission_level": self.configured_permission_level,
            "current_permission_level": self.current_permission_level,
            "cumulative_usage": dict(self.cumulative_usage),
        }
        if self.schedule is not None:
            result["schedule"] = dict(self.schedule)
        if self.active_model_configuration is not None:
            result["active_model_configuration"] = self.active_model_configuration.to_dict()
        if not self.model_configuration_available:
            result["model_configuration_available"] = False
        if self.last_request_usage is not None:
            result["last_request_usage"] = dict(self.last_request_usage)
        return result


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Identity of the Conversation Session selected by a successful resume."""

    session_id: str

    def __post_init__(self) -> None:
        Session._require_id(self.session_id, partition=SessionStoragePartition.FOREGROUND)


class ManagementError(Exception):
    """A safe persistence error suitable for a Management Command."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class FatalManagementError(ManagementError):
    """A safe Management error that must terminate the owning application."""


class ManagementViewService:
    """Expose global configuration and runtime-owned Management views and actions."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        current_agent_loop: Callable[[], _ManagementAgentLoop],
        workspace_state: WorkspaceState,
        replace_agent_loop: Callable[[str, bool], Awaitable[None]],
        prepare_session_resume: Callable[[str], Awaitable[None]],
        memory_manager: _MemoryReader,
        dream: _DreamRunner,
        schedule_status: Callable[[], dict[str, object]],
        now: Callable[[], datetime],
        monotonic: Callable[[], float],
        reasoning_effort_control: _ReasoningEffortControl,
        permission_control: RuntimePermissionControl,
        restore_listing: Callable[[], Awaitable[RestoreListingReport]] | None = None,
        restore_inspect: Callable[[int], Awaitable[RestorePlan]] | None = None,
        restore_commit: Callable[[RestorePlan, RestoreMode | str], Awaitable[RestoreResult]]
        | None = None,
        restore_result: Callable[[], Awaitable[RestoreResult | None]] | None = None,
        restore_cancel: Callable[[], Awaitable[None]] | None = None,
        ensure_management_mutation_allowed: Callable[[], None] | None = None,
    ) -> None:
        self._config = ConfigLoader(agent_home)
        self._current_agent_loop = current_agent_loop
        self._workspace_state = workspace_state
        self._replace_agent_loop = replace_agent_loop
        self._prepare_session_resume = prepare_session_resume
        self._now = now
        self._monotonic = monotonic
        self._schedule_status = schedule_status
        self._memory_reader = memory_manager
        self._dream = dream
        self._reasoning_effort_control = reasoning_effort_control
        self._permission_control = permission_control
        self._restore_listing = restore_listing or _restore_unavailable_listing
        self._restore_inspect = restore_inspect or _restore_unavailable_inspect
        self._restore_commit = restore_commit or _restore_unavailable_commit
        self._restore_result = restore_result or _restore_unavailable_result
        self._restore_cancel = restore_cancel or _restore_unavailable_cancel
        self._restore_acknowledge_failure: Callable[[], Awaitable[RestoreResult | None]] = (
            _restore_unavailable_result
        )
        self._ensure_management_mutation_allowed = (
            ensure_management_mutation_allowed or _allow_management_mutation
        )
        self._ensure_runtime_admission = self._ensure_management_mutation_allowed
        self._persist_reasoning_effort: Callable[[ReasoningEffort], Awaitable[None]] | None = None
        self._configuration_status: Callable[[], str] = lambda: ""

    def bind_configuration_status(self, callback: Callable[[], str]) -> None:
        """Attach the service's saved/startup configuration status to CLI views."""
        self._configuration_status = callback

    def bind_runtime_admission(self, callback: Callable[[], None]) -> None:
        """Bind the generation-wide admission gate after construction."""
        self._ensure_runtime_admission = callback

    def bind_reasoning_effort_persistence(
        self, callback: Callable[[ReasoningEffort], Awaitable[None]]
    ) -> None:
        """Bind service-coordinated persistence for the legacy effort control."""
        self._persist_reasoning_effort = callback

    async def reload_skill(self) -> tuple[SkillMetadata, ...]:
        """Reload the current Agent Loop Skill state and return published metadata."""
        self._ensure_runtime_admission()
        try:
            current_agent_loop = self._current_agent_loop()
            metadata = current_agent_loop.reload_skill()
            if not isinstance(metadata, tuple) or not all(
                isinstance(item, SkillMetadata) for item in metadata
            ):
                raise TypeError("Agent Loop Skill metadata is malformed")
            return metadata
        except Exception as error:
            logger.warning(
                "Skill reload failed type={}",
                type(error).__name__,
            )
            raise ManagementError(
                ErrorInfo("skill_reload_failed", "Skill reload failed.")
            ) from error

    def _ensure_current_generation(self) -> None:
        self._current_agent_loop()

    async def config_view(self) -> ConfigView:
        """Return complete redacted User Configuration content."""
        try:
            self._config.ensure_default()
            return replace(self._config.view(), service_status_text=self._configuration_status())
        except (OSError, UnicodeError) as error:
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "User Configuration could not be read or written.",
                )
            ) from error

    async def memory_view(self) -> str:
        """Return the complete current Long-term Memory file."""
        self._ensure_current_generation()
        try:
            return await self._memory_reader.read_long_term()
        except ManagementError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Long-term Memory could not be read.")
            ) from error

    async def dream(self) -> DreamResult:
        """Run one foreground Memory Task and return its safe summary."""
        self._ensure_runtime_admission()
        self._ensure_current_generation()
        return await self._dream.run()

    async def reasoning_effort(self) -> ReasoningEffort:
        """Return the current Runtime-Lifetime chat Reasoning Effort."""
        effort = self._reasoning_effort_control.reasoning_effort
        if effort not in REASONING_EFFORT_LEVELS:
            raise ManagementError(
                ErrorInfo("config_invalid", "Runtime Reasoning Effort is invalid.")
            )
        return effort

    async def update_reasoning_effort(self, effort: ReasoningEffort) -> ReasoningEffort:
        """Publish one validated Runtime-Lifetime chat Reasoning Effort."""
        self._ensure_management_mutation_allowed()
        if effort not in REASONING_EFFORT_LEVELS:
            raise ManagementError(
                ErrorInfo("config_invalid", "Runtime Reasoning Effort is invalid.")
            )
        self._reasoning_effort_control.set_reasoning_effort(effort)
        try:
            if self._persist_reasoning_effort is None:
                self._config.update_reasoning_effort(effort)
            else:
                await self._persist_reasoning_effort(effort)
        except Exception as error:
            logger.warning("Reasoning Effort persistence failed type={}", type(error).__name__)
        return effort

    async def permission_level(self) -> ToolPermissionLevel:
        """Return the current process-local foreground Tool Permission Level."""
        return self._permission_control.current()

    async def update_permission_level(self, level: ToolPermissionLevel) -> ToolPermissionLevel:
        """Select a foreground level without changing User Configuration."""
        self._ensure_management_mutation_allowed()
        try:
            validated = validate_permission_level(level)
        except ValueError as error:
            raise ManagementError(
                ErrorInfo("config_invalid", "Foreground Tool Permission Level is invalid.")
            ) from error
        self._permission_control.select(validated)
        return self._permission_control.current()

    async def status(self) -> RuntimeStatus:
        """Return all required runtime and current-session status fields."""
        try:
            projection = self._current_agent_loop().runtime_status_input()
            chat_reasoning_effort = (
                projection.chat_reasoning_effort
                if projection.chat_reasoning_effort is not None
                else await self.reasoning_effort()
            )
            if projection.context_window <= 0:
                raise ValueError("Runtime status context window must be positive")
            budget = ContextBudget(
                context_window=projection.context_window,
                max_output=projection.max_output,
                compact_ratio=projection.compact_ratio,
            )
            estimated = estimate_request_tokens(
                projection.projected_messages,
                projection.projected_tools,
            )
            reported_usage = dict(projection.latest_reported_usage)
            projected = project_next_request_tokens(
                estimated,
                snapshot=projection.latest_usage_context,
                reported_usage=reported_usage,
                requested_route=projection.requested_route,
                selected_route=projection.selected_route,
                provider_id=projection.provider_id,
                model=projection.model,
                context_window=budget.context_window,
                max_output=budget.max_output,
                estimator_version=CONTEXT_ESTIMATOR_VERSION,
            )
            started_at = projection.generation_started_at
            uptime = 0 if started_at is None else max(0, int(self._monotonic() - started_at))
            schedule_snapshot = self._schedule_status()
            schedule_status = {
                key: schedule_snapshot[key]
                for key in ("status", "active_job_count")
                if key in schedule_snapshot
            }
            return RuntimeStatus(
                version=__version__,
                chat_model=projection.chat_model,
                chat_reasoning_effort=chat_reasoning_effort,
                active_model_configuration=projection.active_model_configuration,
                model_configuration_available=projection.model_configuration_available,
                uptime_seconds=uptime,
                context_window=budget.context_window,
                max_output=budget.max_output,
                available_context=budget.available_context,
                compact_ratio=budget.compact_ratio,
                compact_context_window=budget.compact_context_window,
                projected_next_request_tokens=projected.projected_tokens,
                projection_source=projected.source,
                input_budget_used_percent=(
                    projected.projected_tokens / budget.available_context * 100
                ),
                session_message_count=projection.session_message_count,
                last_compacted=projection.last_compacted,
                cumulative_usage=dict(projection.cumulative_usage),
                last_request_usage=projection.last_request_usage,
                configured_permission_level=self._permission_control.configured(),
                current_permission_level=self._permission_control.current(),
                schedule=schedule_status,
            )
        except ManagementError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Runtime status could not be read.")
            ) from error

    async def resumable_listing(self) -> SessionListingReport:
        """Return one atomic Session picker result including skipped diagnostics."""
        self._ensure_current_generation()
        return await self._resumable_listing()

    async def _resumable_listing(self) -> SessionListingReport:
        workspace_state = self._workspace_state
        summaries: list[SessionListingEntry] = []
        skipped_count = 0
        try:
            sessions_directory = workspace_state.existing_sessions_directory()
            if sessions_directory is None:
                return SessionListingReport(sessions=(), skipped_count=0)
            paths = tuple(HOST_FILESYSTEM.path_for_io(sessions_directory).glob("*.jsonl"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Conversation Sessions could not be listed.")
            ) from error
        for path in paths:
            try:
                session = Session.load(
                    workspace_state,
                    path.stem,
                    partition=SessionStoragePartition.FOREGROUND,
                    now=self._now,
                )
            except (OSError, UnicodeError, ValueError) as error:
                logger.opt(exception=error).warning(
                    "Skipped corrupt or unreadable Conversation Session entry path={} type={}",
                    path,
                    type(error).__name__,
                )
                skipped_count += 1
                continue
            summaries.append(
                SessionListingEntry(
                    id=session.session_id,
                    title=_session_title(session),
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=len(session.messages),
                )
            )
        return SessionListingReport(
            sessions=tuple(
                sorted(
                    summaries,
                    key=lambda summary: (
                        summary.updated_at,
                        summary.created_at,
                        summary.id,
                    ),
                    reverse=True,
                )
            ),
            skipped_count=skipped_count,
        )

    async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult:
        """Revalidate and select one Session from the current Workspace."""
        self._ensure_management_mutation_allowed()
        self._ensure_current_generation()
        await self._prepare_session_resume(session_id)
        listing = await self._resumable_listing()
        sessions = listing.sessions
        if session_id not in {summary.id for summary in sessions}:
            raise ManagementError(
                ErrorInfo(
                    "model_invalid_request",
                    "The selected Conversation Session is not resumable.",
                )
            )
        await self._replace_agent_loop(session_id, force)
        return ResumeResult(session_id=session_id)

    async def restore_listing(self) -> RestoreListingReport:
        """Return persisted Restore Anchors for the active foreground Session."""
        return await self._restore_listing()

    async def restore_inspect(self, anchor_id: int) -> RestorePlan:
        """Freeze restore admission and inspect one persisted Restore Anchor."""
        return await self._restore_inspect(anchor_id)

    async def restore_commit(
        self,
        plan: RestorePlan,
        mode: RestoreMode | str,
    ) -> RestoreResult:
        """Commit one previously inspected Session Restore plan."""
        return await self._restore_commit(plan, mode)

    async def restore_result(self) -> RestoreResult | None:
        """Return the latest completed restore result, if one is available."""
        return await self._restore_result()

    async def restore_cancel(self) -> None:
        """Cancel a pre-confirmation restore and release its admission barriers."""
        await self._restore_cancel()

    def bind_restore_acknowledge_failure(
        self,
        callback: Callable[[], Awaitable[RestoreResult | None]],
    ) -> None:
        """Bind the runtime-owned durable failure acknowledgement action."""
        self._restore_acknowledge_failure = callback

    async def restore_acknowledge_failure(self) -> RestoreResult | None:
        """Persist acknowledgement of the latest File Restore failure notice."""
        return await self._restore_acknowledge_failure()


def _session_title(session: Session) -> str:
    title = session.metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("Session title is malformed")
    return title


def _allow_management_mutation() -> None:
    return None


async def _restore_unavailable_listing() -> RestoreListingReport:
    raise ManagementError(ErrorInfo("route_unavailable", "Session Restore is unavailable."))


async def _restore_unavailable_inspect(_anchor_id: int) -> RestorePlan:
    raise ManagementError(ErrorInfo("route_unavailable", "Session Restore is unavailable."))


async def _restore_unavailable_commit(
    _plan: RestorePlan,
    _mode: RestoreMode | str,
) -> RestoreResult:
    raise ManagementError(ErrorInfo("route_unavailable", "Session Restore is unavailable."))


async def _restore_unavailable_result() -> RestoreResult | None:
    raise ManagementError(ErrorInfo("route_unavailable", "Session Restore is unavailable."))


async def _restore_unavailable_cancel() -> None:
    raise ManagementError(ErrorInfo("route_unavailable", "Session Restore is unavailable."))
