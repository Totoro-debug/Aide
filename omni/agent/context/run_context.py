"""Run-local context preparation, compaction, and staged commit values."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, NoReturn, Protocol

from omni.agent.context.budget import (
    CONTEXT_ESTIMATOR_VERSION,
    ContextBudget,
    ContextProjection,
    ContextUsageSnapshot,
    ProjectionSource,
    estimate_request_tokens,
    estimate_run_slice_tokens,
    project_next_request_tokens,
    reported_model_usage_total,
    request_fits_model_context,
)
from omni.agent.memory.manager import MemoryManager
from omni.agent.run_errors import CommittableAgentRunError
from omni.agent.session.session import Session
from omni.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from omni.provider.errors import ModelCallError, model_context_overflow_error
from omni.provider.model_router import ModelAttemptGuard, ModelRouteStatus, RunModelRouter
from omni.provider.models import (
    ModelContinuation,
    ModelMessages,
    ModelResponse,
    ModelRoute,
)
from omni.templates import render_template
from omni.utils.validation import empty_token_usage

type CompactionProjection = Callable[
    [Sequence[dict[str, Any]]],
    list[dict[str, Any]],
]
_COMPACTION_JSON_TRANSLATION = str.maketrans({"`": r"\u0060"})
_MICRO_COMPRESSION_TOOL_CALL_THRESHOLD = 10
_TOOL_RESULT_MICRO_COMPRESSION_CHAR_LIMIT = 512

__all__ = [
    "AgentRunContextController",
    "AgentRunContextModelRouter",
    "AgentRunContextRequestPreparer",
    "AgentRunContextSnapshot",
    "AgentRunTerminalCommitValues",
    "agent_run_attempt_guard",
    "latest_main_agent_usage_anchor",
]


class AgentRunContextModelRouter(Protocol):
    """Router seam for one guarded logical Memory completion."""

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class AgentRunContextSnapshot:
    """Detached Session state captured at Agent Run start."""

    messages: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]
    last_compacted: int

    def __post_init__(self) -> None:
        if isinstance(self.messages, (str, bytes)):
            raise TypeError("Agent Run snapshot messages must be a sequence")
        if self.last_compacted < 0 or self.last_compacted > len(self.messages):
            raise ValueError("Agent Run snapshot cursor is outside its transcript")
        object.__setattr__(self, "messages", tuple(deepcopy(list(self.messages))))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def from_session(cls, session: Session) -> AgentRunContextSnapshot:
        """Copy the complete raw Session transcript and staged metadata."""
        return cls(
            messages=tuple(session.messages),
            metadata=session.metadata,
            last_compacted=session.last_compacted,
        )


@dataclass(frozen=True, slots=True)
class AgentRunTerminalCommitValues:
    """Detached values accepted by ``Session.commit_agent_run``."""

    pending_last_compacted: int
    pending_action_summary: str | None
    usage_delta: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PendingFactBatch:
    batch: tuple[dict[str, Any], ...]
    cutoff: int
    selected_payload: str


@dataclass(frozen=True, slots=True)
class _ReactRevisionObservation:
    current_user: dict[str, Any] | None
    tools: tuple[dict[str, Any], ...]
    route_status: ModelRouteStatus
    memory_route_status: ModelRouteStatus | None
    compact_ratio: float
    estimator_version: str
    increment: tuple[dict[str, Any], ...]
    latest_cycle_start: int | None
    continuation_revision: int


class AgentRunContextController:
    """Run-local staged context state shared by Run-start and ReAct preparation."""

    def __init__(
        self,
        *,
        snapshot: AgentRunContextSnapshot,
        provider: AgentRunContextModelRouter,
        memory_manager: MemoryManager,
        now: Callable[[], datetime],
    ) -> None:
        self._snapshot = AgentRunContextSnapshot(
            messages=snapshot.messages,
            metadata=snapshot.metadata,
            last_compacted=snapshot.last_compacted,
        )
        self._provider = provider
        self._memory_manager = memory_manager
        self._now = now
        self._pending_last_compacted = snapshot.last_compacted
        self._pending_action_summary = _normalized_staged_action_summary(
            snapshot.metadata.get("summary")
        )
        self._pending_compaction_usage = empty_token_usage()
        self._current_user_compacted = False
        self._latest_usage_anchor = latest_main_agent_usage_anchor(snapshot.messages)
        self._checked_preparation_revision: str | None = None
        self._pending_fact: _PendingFactBatch | None = None
        self._failed_context_revision: str | None = None
        self._failed_exception: Exception | None = None
        self._run_anchor_context: ContextUsageSnapshot | None = None
        self._run_anchor_usage: dict[str, int] | None = None
        self._run_anchor_tools: tuple[dict[str, Any], ...] | None = None
        self._run_anchor_non_target: tuple[dict[str, Any], ...] | None = None
        self._run_current_user: dict[str, Any] | None = None

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        provider: AgentRunContextModelRouter,
        memory_manager: MemoryManager,
        now: Callable[[], datetime],
    ) -> AgentRunContextController:
        """Create a controller without retaining the writable Session object."""
        return cls(
            snapshot=AgentRunContextSnapshot.from_session(session),
            provider=provider,
            memory_manager=memory_manager,
            now=now,
        )

    def terminal_commit_values(self) -> AgentRunTerminalCommitValues:
        """Return values accepted by the terminal Session commit."""
        return AgentRunTerminalCommitValues(
            pending_last_compacted=self._pending_last_compacted,
            pending_action_summary=self._pending_action_summary,
            usage_delta=dict(self._pending_compaction_usage),
        )

    async def prepare_run_start(
        self,
        *,
        project_messages: CompactionProjection,
        route_status: ModelRouteStatus,
        memory_route_status: ModelRouteStatus,
        tools: Sequence[dict[str, Any]] = (),
        current_user: dict[str, Any] | None = None,
        compact_ratio: float = 0.9,
        estimator_version: str = CONTEXT_ESTIMATOR_VERSION,
    ) -> tuple[dict[str, Any], ...]:
        """Check and stage Run-start history compression without publishing Session state."""
        budget = ContextBudget(
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            compact_ratio=compact_ratio,
        )
        effective_tools = tuple(deepcopy(list(tools)))
        copied_user = None if current_user is None else deepcopy(current_user)
        projected = self._project_candidate(
            project_messages,
            current_user=copied_user,
        )
        projection = self._projection_from_candidate(
            projected,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )
        revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
        )
        self._run_current_user = copied_user
        if self._failed_context_revision == revision:
            assert self._failed_exception is not None
            raise self._failed_exception
        self._failed_context_revision = None
        self._failed_exception = None
        if self._checked_preparation_revision == revision:
            return tuple(deepcopy(projected))

        protected_projection = self._projected_tokens(
            project_messages,
            raw_messages=(),
            current_user=copied_user,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )
        if budget.exceeds_available_context(protected_projection.projected_tokens):
            overflow_error = model_context_overflow_error()
            self._record_failure(revision, overflow_error)
            raise overflow_error
        if self._pending_fact is None and not budget.should_compact(projection.projected_tokens):
            self._checked_preparation_revision = revision
            return tuple(deepcopy(projected))

        pending_fact = self._pending_fact
        if pending_fact is None:
            batch, cutoff = self._select_run_start_batch(budget)
        else:
            batch = tuple(deepcopy(list(pending_fact.batch)))
            cutoff = pending_fact.cutoff
        if batch and pending_fact is None:
            retained_projection = self._projected_tokens(
                project_messages,
                raw_messages=self._snapshot.messages[cutoff:],
                current_user=copied_user,
                tools=effective_tools,
                route_status=route_status,
                estimator_version=estimator_version,
            )
            if budget.exceeds_available_context(retained_projection.projected_tokens):
                all_batch, all_cutoff = self._batch_from_runs(self._eligible_runs())
                if all_batch and all_cutoff != cutoff:
                    all_projection = self._projected_tokens(
                        project_messages,
                        raw_messages=self._snapshot.messages[all_cutoff:],
                        current_user=copied_user,
                        tools=effective_tools,
                        route_status=route_status,
                        estimator_version=estimator_version,
                    )
                    if not budget.exceeds_available_context(all_projection.projected_tokens):
                        batch, cutoff = all_batch, all_cutoff
                    else:
                        overflow_error = model_context_overflow_error()
                        self._record_failure(revision, overflow_error)
                        raise overflow_error
                else:
                    overflow_error = model_context_overflow_error()
                    self._record_failure(revision, overflow_error)
                    raise overflow_error
        else:
            if not batch:
                self._checked_preparation_revision = revision
                if budget.exceeds_available_context(projection.projected_tokens):
                    overflow_error = model_context_overflow_error()
                    self._record_failure(revision, overflow_error)
                    raise overflow_error
                return tuple(deepcopy(projected))

        await self._stage_summary_pair(
            revision=revision,
            batch=batch,
            cutoff=cutoff,
            memory_route_status=memory_route_status,
        )
        final_projected = self._project_candidate(
            project_messages,
            current_user=copied_user,
        )
        final_projection = self._projection_from_candidate(
            final_projected,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )
        final_revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=final_projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
        )
        self._checked_preparation_revision = final_revision
        if budget.exceeds_available_context(final_projection.projected_tokens):
            overflow_error = model_context_overflow_error()
            self._record_failure(final_revision, overflow_error)
            raise overflow_error
        return tuple(deepcopy(final_projected))

    def record_main_agent_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        estimator_version: str,
    ) -> ContextUsageSnapshot:
        """Record one main response's route, anchor and run projection provenance."""
        anchor_estimated_tokens = estimate_request_tokens(
            [*request_messages, response.message.to_dict()],
            tools,
        )
        current_user = self._current_user_for_run()
        run_messages: list[dict[str, Any]] = []
        if current_user is not None and not self._current_user_compacted:
            run_messages.append(current_user)
        run_messages.extend(deepcopy(list(increment)))
        run_projected_tokens = estimate_run_slice_tokens(run_messages)
        projection_source: ProjectionSource = "estimated"
        baseline = self._run_anchor_context
        baseline_usage = self._run_anchor_usage
        non_target = _non_target_projection(request_messages)
        if (
            baseline is not None
            and baseline_usage is not None
            and self._run_anchor_tools == tuple(deepcopy(list(tools)))
            and self._run_anchor_non_target == non_target
            and not self._current_user_compacted
            and _usage_context_matches(baseline, route_status, estimator_version)
            and reported_model_usage_total(baseline_usage) is not None
        ):
            run_projected_tokens = max(
                0,
                baseline.run_projected_tokens
                + response.usage.total_tokens
                - baseline_usage["total_tokens"],
            )
            projection_source = "reported_delta"

        context = ContextUsageSnapshot(
            requested_route=route_status.requested_route,
            selected_route=route_status.selected_route,
            provider_id=route_status.provider_id,
            model=route_status.model,
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            anchor_estimated_tokens=anchor_estimated_tokens,
            estimator_version=estimator_version,
            run_projected_tokens=run_projected_tokens,
            run_projection_source=projection_source,
        )
        usage = {
            "model_calls": 1,
            **response.usage.to_dict(),
        }
        self._latest_usage_anchor = (context, deepcopy(usage))
        self._run_anchor_context = context
        self._run_anchor_usage = deepcopy(usage)
        self._run_anchor_tools = tuple(deepcopy(list(tools)))
        self._run_anchor_non_target = non_target
        return context

    def _current_user_for_run(self) -> dict[str, Any] | None:
        """Return the detached current User captured by the request preparer."""
        return None if self._run_current_user is None else deepcopy(self._run_current_user)

    async def prepare_react(
        self,
        *,
        project_messages: CompactionProjection,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        route_status: ModelRouteStatus,
        memory_route_status: ModelRouteStatus,
        tools: Sequence[dict[str, Any]] = (),
        current_user: dict[str, Any] | None = None,
        compact_ratio: float = 0.9,
        estimator_version: str = CONTEXT_ESTIMATOR_VERSION,
        continuation_revision: int = 0,
        micro_compression_enabled: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Prepare one ReAct request from the run's raw increment."""
        budget = ContextBudget(
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            compact_ratio=compact_ratio,
        )
        effective_tools = tuple(deepcopy(list(tools)))
        copied_user = None if current_user is None else deepcopy(current_user)
        copied_increment = tuple(deepcopy(list(increment)))
        _validate_react_increment(copied_increment)
        _validate_latest_cycle_start(latest_cycle_start, copied_increment)
        projected = self._react_project_candidate(
            copied_increment,
            current_user=copied_user,
            project_messages=project_messages,
        )
        projection = self._projection_from_candidate(
            projected,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )
        revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
            increment=copied_increment,
            latest_cycle_start=latest_cycle_start,
            continuation_revision=continuation_revision,
            micro_compression_enabled=micro_compression_enabled,
        )
        if self._failed_context_revision == revision:
            assert self._failed_exception is not None
            raise self._failed_exception
        self._failed_context_revision = None
        self._failed_exception = None
        if self._checked_preparation_revision == revision:
            return tuple(deepcopy(projected))

        protected = self._react_protected_projection(
            copied_increment,
            current_user=copied_user,
            latest_cycle_start=latest_cycle_start,
            project_messages=project_messages,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )
        if budget.exceeds_available_context(protected.projected_tokens):
            overflow_error = model_context_overflow_error()
            self._record_failure(revision, overflow_error)
            raise overflow_error
        if self._pending_fact is None and not budget.should_compact(projection.projected_tokens):
            self._checked_preparation_revision = revision
            return tuple(deepcopy(projected))

        pending_fact = self._pending_fact
        if pending_fact is None:
            batch, cutoff = self._select_react_batch(
                budget,
                copied_increment,
                current_user=copied_user,
                latest_cycle_start=latest_cycle_start,
            )
        else:
            batch = tuple(deepcopy(list(pending_fact.batch)))
            cutoff = pending_fact.cutoff
        if not batch:
            self._checked_preparation_revision = revision
            return tuple(deepcopy(projected))

        selected_user = (
            copied_user is not None
            and not self._current_user_compacted
            and self._pending_last_compacted <= len(self._snapshot.messages) < cutoff
        )
        await self._stage_summary_pair(
            revision=revision,
            batch=batch,
            cutoff=cutoff,
            memory_route_status=memory_route_status,
        )
        if selected_user:
            self._current_user_compacted = True
        final_projected = self._react_project_candidate(
            copied_increment,
            current_user=copied_user,
            project_messages=project_messages,
        )
        final_revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=final_projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
            increment=copied_increment,
            latest_cycle_start=latest_cycle_start,
            continuation_revision=continuation_revision,
            micro_compression_enabled=micro_compression_enabled,
        )
        self._checked_preparation_revision = final_revision
        return tuple(deepcopy(final_projected))

    def observe_react_request_projection(
        self,
        observation: _ReactRevisionObservation,
        messages: Sequence[dict[str, Any]],
        *,
        micro_compression_enabled: bool,
    ) -> None:
        """Record the final Provider projection before the Provider call."""
        preparation_revision = self._context_revision(
            current_user=observation.current_user,
            tools=observation.tools,
            projected=tuple(deepcopy(list(messages))),
            route_status=observation.route_status,
            memory_route_status=observation.memory_route_status,
            compact_ratio=observation.compact_ratio,
            estimator_version=observation.estimator_version,
            increment=observation.increment,
            latest_cycle_start=observation.latest_cycle_start,
            continuation_revision=observation.continuation_revision,
            micro_compression_enabled=micro_compression_enabled,
        )
        self._checked_preparation_revision = preparation_revision

    def _react_project_candidate(
        self,
        increment: Sequence[dict[str, Any]],
        *,
        current_user: dict[str, Any] | None,
        project_messages: CompactionProjection,
    ) -> list[dict[str, Any]]:
        snapshot_length = len(self._snapshot.messages)
        user_offset = 1 if current_user is not None else 0
        increment_start = max(
            0,
            self._pending_last_compacted - snapshot_length - user_offset,
        )
        source = list(
            deepcopy(self._snapshot.messages[min(self._pending_last_compacted, snapshot_length) :])
        )
        if current_user is not None:
            source.append(deepcopy(current_user))
        source.extend(deepcopy(list(increment[increment_start:])))
        return _insert_action_summary(
            project_messages(source),
            self._pending_action_summary,
        )

    def _react_protected_projection(
        self,
        increment: Sequence[dict[str, Any]],
        *,
        current_user: dict[str, Any] | None,
        latest_cycle_start: int | None,
        project_messages: CompactionProjection,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        estimator_version: str,
    ) -> ContextProjection:
        if latest_cycle_start is None:
            protected_increment: Sequence[dict[str, Any]] = ()
        else:
            increment_start = min(max(latest_cycle_start, 0), len(increment))
            protected_increment = increment[increment_start:]
        source = [] if current_user is None else [deepcopy(current_user)]
        source.extend(deepcopy(list(protected_increment)))
        projected = _insert_action_summary(project_messages(source), self._pending_action_summary)
        return self._projection_from_candidate(
            projected,
            tools=tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )

    def _select_react_batch(
        self,
        budget: ContextBudget,
        increment: Sequence[dict[str, Any]],
        *,
        current_user: dict[str, Any] | None,
        latest_cycle_start: int | None,
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        virtual = deepcopy(list(self._snapshot.messages))
        if current_user is not None:
            virtual.append(deepcopy(current_user))
        virtual.extend(deepcopy(list(increment)))
        snapshot_length = len(self._snapshot.messages)
        increment_base = snapshot_length + (1 if current_user is not None else 0)
        current_start = (
            increment_base
            if self._current_user_compacted
            else (snapshot_length if current_user is not None else increment_base)
        )
        current_start = max(current_start, self._pending_last_compacted)
        current_slice = virtual[current_start:]
        current_fits = budget.can_retain_run_slice(current_slice, percentage=50)
        eligible = self._eligible_runs()
        history_batch, history_cutoff = self._batch_from_runs(eligible)
        if current_fits and history_batch:
            return history_batch, history_cutoff
        if latest_cycle_start is None:
            return (), self._pending_last_compacted
        cycle_start = increment_base + latest_cycle_start
        cutoff = min(max(cycle_start, self._pending_last_compacted), len(virtual))
        if cutoff <= self._pending_last_compacted:
            return (), self._pending_last_compacted
        return (
            tuple(deepcopy(virtual[self._pending_last_compacted : cutoff])),
            cutoff,
        )

    def _project_candidate(
        self,
        project_messages: CompactionProjection,
        *,
        current_user: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        raw_messages = list(self._snapshot.messages[self._pending_last_compacted :])
        if current_user is not None:
            raw_messages.append(deepcopy(current_user))
        return _insert_action_summary(
            project_messages(deepcopy(raw_messages)),
            self._pending_action_summary,
        )

    def _projected_tokens(
        self,
        project_messages: CompactionProjection,
        *,
        raw_messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        estimator_version: str,
    ) -> ContextProjection:
        source = list(deepcopy(list(raw_messages)))
        if current_user is not None:
            source.append(deepcopy(current_user))
        projected = _insert_action_summary(project_messages(source), self._pending_action_summary)
        return self._projection_from_candidate(
            projected,
            tools=tools,
            route_status=route_status,
            estimator_version=estimator_version,
        )

    def _projection_from_candidate(
        self,
        projected: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        estimator_version: str,
    ) -> ContextProjection:
        usage_anchor = self._latest_usage_anchor
        if usage_anchor is None or not _usage_context_matches(
            usage_anchor[0], route_status, estimator_version
        ):
            usage_context = None
            usage = None
        else:
            usage_context, usage = usage_anchor
        return project_next_request_tokens(
            estimate_request_tokens(projected, tools),
            snapshot=usage_context,
            reported_usage=usage,
            requested_route=route_status.requested_route,
            selected_route=route_status.selected_route,
            provider_id=route_status.provider_id,
            model=route_status.model,
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            estimator_version=estimator_version,
        )

    async def _stage_summary_pair(
        self,
        *,
        batch: Sequence[dict[str, Any]],
        cutoff: int,
        revision: str,
        memory_route_status: ModelRouteStatus,
    ) -> None:
        pending_fact = self._pending_fact
        selected_payload = (
            _compaction_user_context(list(batch))
            if pending_fact is None
            else pending_fact.selected_payload
        )
        memory_budget = ContextBudget(
            context_window=memory_route_status.context_window,
            max_output=memory_route_status.max_output,
            compact_ratio=0.9,
        )
        if pending_fact is None:
            fact_messages = _summary_request_messages(
                template_name="conversation-compaction-system-prompt.md",
                selected_payload=selected_payload,
            )
            if memory_budget.exceeds_available_context(estimate_request_tokens(fact_messages)):
                overflow_error = model_context_overflow_error()
                self._raise_summary_failure(revision, overflow_error)
            try:
                fact_response = await self._provider.complete(
                    "memory",
                    messages=fact_messages,
                    tools=(),
                    guard=agent_run_attempt_guard,
                )
            except ModelCallError as provider_error:
                self._raise_summary_failure(revision, provider_error)
            except Exception as provider_error:
                self._record_failure(revision, provider_error)
                raise
            _add_pending_usage(self._pending_compaction_usage, fact_response)
            response_error = _summary_response_error(fact_response)
            if response_error is not None:
                self._raise_summary_failure(revision, response_error)
            try:
                await self._memory_manager.append_summary(
                    content=fact_response.message.content,
                    timestamp=self._persisted_now(),
                )
            except (OSError, UnicodeError, ValueError) as persistence_cause:
                persistence_error = ModelCallError(
                    ErrorInfo(
                        code="persistence_error",
                        message="Conversation Summary could not be persisted.",
                    )
                )
                self._raise_summary_failure(
                    revision,
                    persistence_error,
                    cause=persistence_cause,
                )
            self._pending_fact = _PendingFactBatch(
                batch=tuple(deepcopy(list(batch))),
                cutoff=cutoff,
                selected_payload=selected_payload,
            )

        action_messages = _summary_request_messages(
            template_name="conversation-summary-system-prompt.md",
            selected_payload=_action_summary_user_context(
                self._pending_action_summary,
                selected_payload,
            ),
        )
        if memory_budget.exceeds_available_context(estimate_request_tokens(action_messages)):
            overflow_error = model_context_overflow_error()
            self._raise_summary_failure(revision, overflow_error)
        try:
            action_response = await self._provider.complete(
                "memory",
                messages=action_messages,
                tools=(),
                guard=agent_run_attempt_guard,
            )
        except ModelCallError as provider_error:
            self._raise_summary_failure(revision, provider_error)
        except Exception as provider_error:
            self._record_failure(revision, provider_error)
            raise
        _add_pending_usage(self._pending_compaction_usage, action_response)
        response_error = _summary_response_error(action_response)
        if response_error is not None:
            self._raise_summary_failure(revision, response_error)

        self._pending_action_summary = _normalize_action_summary(action_response.message.content)
        self._pending_last_compacted = cutoff
        self._pending_fact = None

    def _record_failure(self, revision: str, error: Exception) -> None:
        self._checked_preparation_revision = revision
        self._failed_context_revision = revision
        self._failed_exception = error

    def _raise_summary_failure(
        self,
        revision: str,
        failure: ModelCallError,
        *,
        cause: BaseException | None = None,
    ) -> NoReturn:
        committable = CommittableAgentRunError(failure.error)
        self._record_failure(revision, committable)
        raise committable from (failure if cause is None else cause)

    def _select_run_start_batch(
        self, budget: ContextBudget
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        eligible = self._eligible_runs()
        if not eligible:
            return (), self._pending_last_compacted
        if len(eligible) == 1:
            selected = eligible
        else:
            latest_start, latest_end = eligible[-1]
            latest_suffix = self._snapshot.messages[
                max(self._pending_last_compacted, latest_start) : latest_end
            ]
            selected = (
                eligible[:-1]
                if budget.can_retain_run_slice(
                    latest_suffix,
                    percentage=10,
                )
                else eligible
            )
        return self._batch_from_runs(selected)

    def _batch_from_runs(
        self,
        runs: Sequence[tuple[int, int]],
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        if not runs:
            return (), self._pending_last_compacted
        start = max(self._pending_last_compacted, runs[0][0])
        cutoff = runs[-1][1]
        if cutoff <= start:
            return (), self._pending_last_compacted
        return tuple(deepcopy(self._snapshot.messages[start:cutoff])), cutoff

    def _eligible_runs(self) -> list[tuple[int, int]]:
        runs = _completed_run_ranges(self._snapshot.messages)
        return [
            (max(self._pending_last_compacted, start), end)
            for start, end in runs
            if end > self._pending_last_compacted
        ]

    def _context_revision(
        self,
        *,
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
        projected: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        memory_route_status: ModelRouteStatus | None,
        compact_ratio: float,
        estimator_version: str,
        increment: Sequence[dict[str, Any]] = (),
        latest_cycle_start: int | None = None,
        continuation_revision: int = 0,
        micro_compression_enabled: bool = False,
    ) -> str:
        value = {
            "transcript": self._snapshot.messages,
            "pending_last_compacted": self._pending_last_compacted,
            "pending_action_summary": self._pending_action_summary,
            "current_user": current_user,
            "current_user_compacted": self._current_user_compacted,
            "temporary_current_user": (current_user if self._current_user_compacted else None),
            "increment": list(increment),
            "latest_cycle_start": latest_cycle_start,
            "continuation_revision": continuation_revision,
            "micro_compression_enabled": micro_compression_enabled,
            "tools": list(tools),
            "projected": list(projected),
            "route": {
                "requested_route": route_status.requested_route,
                "selected_route": route_status.selected_route,
                "provider_id": route_status.provider_id,
                "model": route_status.model,
                "context_window": route_status.context_window,
                "max_output": route_status.max_output,
            },
            "compact_ratio": compact_ratio,
            "memory_route": (
                None
                if memory_route_status is None
                else {
                    "requested_route": memory_route_status.requested_route,
                    "selected_route": memory_route_status.selected_route,
                    "provider_id": memory_route_status.provider_id,
                    "model": memory_route_status.model,
                    "context_window": memory_route_status.context_window,
                    "max_output": memory_route_status.max_output,
                }
            ),
            "estimator_version": estimator_version,
        }
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


class AgentRunContextRequestPreparer:
    """Adapt one staged controller to the Agent Runner's narrow request seam."""

    def __init__(
        self,
        controller: AgentRunContextController,
        *,
        router: RunModelRouter,
        requested_route: Literal["chat", "schedule"],
        project_messages: CompactionProjection,
        current_user: dict[str, Any] | None = None,
        compact_ratio: float = 0.9,
        estimator_version: str = CONTEXT_ESTIMATOR_VERSION,
    ) -> None:
        self._controller = controller
        self._router = router
        self._requested_route = requested_route
        self._project_messages = project_messages
        self._current_user = None if current_user is None else deepcopy(current_user)
        self._compact_ratio = compact_ratio
        self._estimator_version = estimator_version
        self._micro_compression_enabled = False

    async def prepare(
        self,
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        continuation_revision: int,
        is_micro_compression_eligible: Callable[[str], bool] | None,
    ) -> list[dict[str, Any]]:
        route_status = self._router.call_route_status(
            self._requested_route,
            continuation=continuation,
        )
        memory_route_status = self._router.call_route_status("memory", continuation=None)
        prepared_messages = await self._controller.prepare_react(
            project_messages=self._project_messages,
            increment=deepcopy(list(increment)),
            latest_cycle_start=latest_cycle_start,
            route_status=route_status,
            memory_route_status=memory_route_status,
            tools=deepcopy(list(tools)),
            current_user=None if self._current_user is None else deepcopy(self._current_user),
            compact_ratio=self._compact_ratio,
            estimator_version=self._estimator_version,
            continuation_revision=continuation_revision,
            micro_compression_enabled=self._micro_compression_enabled,
        )
        request_messages = deepcopy(list(prepared_messages))
        micro_compression_enabled = (
            is_micro_compression_eligible is not None
            and _micro_compression_eligible_count(
                request_messages, is_micro_compression_eligible=is_micro_compression_eligible
            )
            > _MICRO_COMPRESSION_TOOL_CALL_THRESHOLD
        )
        if micro_compression_enabled:
            assert is_micro_compression_eligible is not None
            request_messages = _project_for_model_request(
                request_messages,
                omit_tool_results_before=_latest_completed_cycle_start(request_messages),
                is_micro_compression_eligible=is_micro_compression_eligible,
            )
        observation = _ReactRevisionObservation(
            current_user=None if self._current_user is None else deepcopy(self._current_user),
            tools=tuple(deepcopy(list(tools))),
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=self._compact_ratio,
            estimator_version=self._estimator_version,
            increment=tuple(deepcopy(list(increment))),
            latest_cycle_start=latest_cycle_start,
            continuation_revision=continuation_revision,
        )
        self._controller.observe_react_request_projection(
            observation,
            request_messages,
            micro_compression_enabled=micro_compression_enabled,
        )
        self._micro_compression_enabled = micro_compression_enabled
        return deepcopy(request_messages)

    def record_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        """Attach the usage anchor for the assistant response just produced."""
        route_status = self._router.current_call_status(self._requested_route)
        if route_status is None:
            raise RuntimeError("response recording requires one completed Model call")
        return self._controller.record_main_agent_response(
            request_messages=request_messages,
            tools=tools,
            response=response,
            increment=increment,
            route_status=route_status,
            estimator_version=self._estimator_version,
        ).to_dict()


def _micro_compression_eligible_count(
    messages: Sequence[dict[str, Any]],
    *,
    is_micro_compression_eligible: Callable[[str], bool],
) -> int:
    return sum(
        1
        for message in messages
        if message.get("role") == "tool"
        and isinstance(message.get("name"), str)
        and message.get("status", "success") in {"success", "error", "refused"}
        and is_micro_compression_eligible(message["name"])
    )


def _latest_completed_cycle_start(messages: Sequence[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        tool_calls = message.get("tool_calls")
        if (
            message.get("role") == "assistant"
            and isinstance(tool_calls, Sequence)
            and not isinstance(tool_calls, (str, bytes))
            and tool_calls
        ):
            return index
    return None


def _project_for_model_request(
    messages: Sequence[dict[str, Any]],
    *,
    omit_tool_results_before: int | None,
    is_micro_compression_eligible: Callable[[str], bool],
) -> list[dict[str, Any]]:
    projected = deepcopy(list(messages))
    if omit_tool_results_before is None:
        return projected

    for index, message in enumerate(projected):
        if index >= omit_tool_results_before or message.get("role") != "tool":
            continue
        name = message.get("name")
        content = message.get("content")
        if (
            not isinstance(name, str)
            or not is_micro_compression_eligible(name)
            or not isinstance(content, str)
            or len(content) <= _TOOL_RESULT_MICRO_COMPRESSION_CHAR_LIMIT
        ):
            continue
        message["content"] = f"[{name} result omitted from context]"
    return projected


def _completed_run_ranges(messages: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    starts = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
        for index, start in enumerate(starts)
    ]


def _validate_react_increment(messages: Sequence[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        if message.get("role") not in {"assistant", "tool"}:
            raise ValueError(f"ReAct increment message {index} must be assistant or tool")


def _validate_latest_cycle_start(
    latest_cycle_start: int | None,
    increment: Sequence[dict[str, Any]],
) -> None:
    if latest_cycle_start is None:
        return
    if (
        isinstance(latest_cycle_start, bool)
        or not isinstance(latest_cycle_start, int)
        or latest_cycle_start < 0
        or latest_cycle_start >= len(increment)
        or increment[latest_cycle_start].get("role") != "assistant"
    ):
        raise ValueError("latest_cycle_start must identify an assistant in the increment")


def _normalized_staged_action_summary(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Session action summary must be a string or None")
    return None if not value or value.strip() == "None" else value


def _non_target_projection(
    messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Capture the stable provider-visible prefix before this run's User."""
    copied = deepcopy(list(messages))
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            return tuple(copied[:index])
    return tuple(copied)


def latest_main_agent_usage_anchor(
    messages: Sequence[dict[str, Any]],
) -> tuple[ContextUsageSnapshot, dict[str, int]] | None:
    """Return the latest main-Agent assistant usage anchor, if it is valid."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        context_value = message.get("context_usage")
        usage_value = message.get("token_usage")
        if (
            context_value is None
            or not isinstance(usage_value, dict)
            or reported_model_usage_total(usage_value) is None
        ):
            return None
        try:
            context = ContextUsageSnapshot.from_dict(context_value)
        except (TypeError, ValueError):
            return None
        if context.requested_route not in {"chat", "schedule"}:
            return None
        return context, deepcopy(usage_value)
    return None


def _usage_context_matches(
    context: ContextUsageSnapshot,
    route_status: ModelRouteStatus,
    estimator_version: str,
) -> bool:
    return (
        context.requested_route == route_status.requested_route
        and context.selected_route == route_status.selected_route
        and context.provider_id == route_status.provider_id
        and context.model == route_status.model
        and context.context_window == route_status.context_window
        and context.max_output == route_status.max_output
        and context.estimator_version == estimator_version
    )


def _add_pending_usage(target: dict[str, int], response: ModelResponse) -> None:
    target["model_calls"] += 1
    target["input_tokens"] += response.usage.input_tokens
    target["output_tokens"] += response.usage.output_tokens
    target["total_tokens"] += response.usage.total_tokens


def _summary_response_error(response: ModelResponse) -> ModelCallError | None:
    if response.finish_reason == "stop" and not response.message.tool_calls:
        return None
    if response.finish_reason == "cancelled":
        return ModelCallError(ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE))
    return ModelCallError(
        ErrorInfo("model_failed", "Summary model response did not complete normally.")
    )


def agent_run_attempt_guard(
    status: ModelRouteStatus,
    messages: ModelMessages,
    tools: Sequence[dict[str, Any]],
) -> bool:
    return request_fits_model_context(
        messages,
        tools,
        context_window=status.context_window,
        max_output=status.max_output,
    )


def _normalize_action_summary(content: str) -> str | None:
    normalized = content.strip()
    return None if normalized == "None" else content


def _action_summary_user_context(previous: str | None, selected_payload: str) -> str:
    if previous is None:
        return selected_payload
    return f"## Previous Action Summary\n\n{previous}\n\n{selected_payload}"


def _insert_action_summary(
    projected: Sequence[dict[str, Any]],
    action_summary: str | None,
) -> list[dict[str, Any]]:
    result = deepcopy(list(projected))
    if action_summary is None:
        return result
    index = 1 if result and result[0].get("role") == "system" else 0
    result.insert(index, {"role": "user", "content": deepcopy(action_summary)})
    return result


def _summary_request_messages(*, template_name: str, selected_payload: str) -> ModelMessages:
    return [
        {"role": "system", "content": render_template(template_name)},
        {"role": "user", "content": selected_payload},
    ]


def _project_compaction_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message["role"]
    if role == "user":
        return {"role": "user", "content": deepcopy(message["content"])}

    if role == "assistant":
        content = message["content"]
        tool_calls = [
            {
                "id": deepcopy(tool_call["id"]),
                "name": deepcopy(tool_call["name"]),
                "arguments": deepcopy(tool_call["arguments"]),
            }
            for tool_call in message["tool_calls"]
        ]
        if message["status"] == "error" and not content and not tool_calls:
            return None
        if message["status"] == "interrupted":
            content = f"{content}\n\n[Turn interrupted by user.]"
        return {
            "role": "assistant",
            "content": deepcopy(content),
            "tool_calls": tool_calls,
        }

    return {
        "role": "tool",
        "tool_call_id": deepcopy(message["tool_call_id"]),
        "name": deepcopy(message["name"]),
        "content": deepcopy(message["content"]),
    }


def _compaction_user_context(messages: list[dict[str, Any]]) -> str:
    records = [
        projected
        for message in messages
        if (projected := _project_compaction_message(message)) is not None
    ]
    serialized = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).translate(_COMPACTION_JSON_TRANSLATION)
    return f"## Conversation Messages\n\n```json\n{serialized}\n```"
