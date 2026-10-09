"""Run-local context preparation, compaction, and staged commit values."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, NoReturn

from aide.agent.context.budget import (
    ContextBudget,
    ContextProjection,
    ContextUsageSnapshot,
    ProjectionSource,
    estimate_request_tokens,
    project_next_request_tokens,
    reported_model_usage_total,
    request_fits_model_context,
)
from aide.agent.context.builder import project_history_message
from aide.agent.context.tokenizer import (
    context_encoding_for_model,
    context_estimator_version_for_model,
    estimate_context_request_tokens,
    estimate_context_run_slice_tokens,
)
from aide.agent.run_errors import CommittableAgentRunError
from aide.agent.session.session import Session
from aide.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from aide.provider.errors import ModelCallError, model_context_overflow_error
from aide.provider.model_router import ModelRouteStatus, RunModelRouter
from aide.provider.models import (
    ModelContinuation,
    ModelMessages,
    ModelResponse,
)
from aide.templates import render_template
from aide.utils.validation import empty_token_usage

type CompactionProjection = Callable[
    [
        Sequence[dict[str, Any]],
        dict[str, Any] | None,
        Sequence[dict[str, Any]],
        int,
        str | None,
    ],
    list[dict[str, Any]],
]
type ToolResultProjection = Callable[
    [Sequence[dict[str, Any]], Collection[int]],
    list[dict[str, Any]],
]
_COMPACTION_JSON_TRANSLATION = str.maketrans({"`": r"\u0060"})
_MICRO_COMPRESSION_TOOL_CALL_THRESHOLD = 10
_TOOL_RESULT_MICRO_COMPRESSION_CHAR_LIMIT = 512

__all__ = [
    "AgentRunContextSnapshot",
    "AgentRunTerminalCommitValues",
    "ContextController",
    "agent_run_attempt_guard",
    "latest_main_agent_usage_anchor",
]


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


class ContextController:
    """Run-local staged context state shared by Run-start and ReAct preparation."""

    def __init__(
        self,
        *,
        snapshot: AgentRunContextSnapshot,
        append_summary: Callable[[str, datetime], Awaitable[object]],
        now: Callable[[], datetime],
        request_router: RunModelRouter,
        requested_route: Literal["chat", "schedule", "subagent"],
        project_messages: CompactionProjection,
        project_tool_results: ToolResultProjection,
        current_user: dict[str, Any] | None = None,
        compact_ratio: float = 0.9,
        enable_tool_micro_compression: bool = False,
    ) -> None:
        self._snapshot = AgentRunContextSnapshot(
            messages=snapshot.messages,
            metadata=snapshot.metadata,
            last_compacted=snapshot.last_compacted,
        )
        self._append_summary = append_summary
        self._now = now
        self._request_router = request_router
        self._requested_route = requested_route
        self._project_messages = project_messages
        self._project_tool_results = project_tool_results
        self._request_current_user = None if current_user is None else deepcopy(current_user)
        self._compact_ratio = compact_ratio
        self._enable_tool_micro_compression = enable_tool_micro_compression
        self._run_start_prepared = False
        self._micro_compression_enabled = False
        self._pending_last_compacted = snapshot.last_compacted
        self._pending_action_summary = _normalized_staged_action_summary(
            snapshot.metadata.get("summary")
        )
        self._pending_compaction_usage = empty_token_usage()
        self._current_user_compacted = False
        self._latest_usage_anchor = latest_main_agent_usage_anchor(snapshot.messages)
        self._checked_preparation_revision: str | None = None
        self._preparation_estimates: dict[tuple[str, str], int] = {}
        self._pending_fact: tuple[tuple[dict[str, Any], ...], int, str] | None = None
        self._failed_context_revision: str | None = None
        self._failed_exception: Exception | None = None
        self._run_anchor_context: ContextUsageSnapshot | None = None
        self._run_anchor_usage: dict[str, int] | None = None
        self._run_anchor_tools: tuple[dict[str, Any], ...] | None = None
        self._run_anchor_non_target: tuple[dict[str, Any], ...] | None = None

    def terminal_commit_values(self) -> AgentRunTerminalCommitValues:
        """Return values accepted by the terminal Session commit."""
        return AgentRunTerminalCommitValues(
            pending_last_compacted=self._pending_last_compacted,
            pending_action_summary=self._pending_action_summary,
            usage_delta=dict(self._pending_compaction_usage),
        )

    @staticmethod
    def estimate_request_tokens(
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        *,
        model: str,
    ) -> int:
        """Count the complete local request without preparing or publishing Run state."""
        return estimate_context_request_tokens(messages, tools, model=model)

    @staticmethod
    def project_next_request_usage(
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        *,
        snapshot: ContextUsageSnapshot | None,
        reported_usage: Mapping[str, object] | None,
        requested_route: str,
        selected_route: str,
        provider_id: str,
        model: str,
        context_window: int,
        max_output: int,
    ) -> ContextProjection:
        """Project one request without triggering compaction or Session writes."""
        return project_next_request_tokens(
            ContextController.estimate_request_tokens(messages, tools, model=model),
            snapshot=snapshot,
            reported_usage=reported_usage,
            requested_route=requested_route,
            selected_route=selected_route,
            provider_id=provider_id,
            model=model,
            context_window=context_window,
            max_output=max_output,
            estimator_version=context_estimator_version_for_model(model),
        )

    def initial_messages(self) -> list[dict[str, Any]]:
        """Return the initial runtime projection without performing budget preparation."""
        return self._project_candidate(current_user=deepcopy(self._request_current_user))

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
        """Prepare the first or next logical Agent Run request before routing it."""
        copied_increment = tuple(deepcopy(list(increment)))
        for index, message in enumerate(copied_increment):
            if message.get("role") not in {"assistant", "tool"}:
                raise ValueError(f"ReAct increment message {index} must be assistant or tool")
        if latest_cycle_start is not None and (
            isinstance(latest_cycle_start, bool)
            or not isinstance(latest_cycle_start, int)
            or latest_cycle_start < 0
            or latest_cycle_start >= len(copied_increment)
            or copied_increment[latest_cycle_start].get("role") != "assistant"
        ):
            raise ValueError("latest_cycle_start must identify an assistant in the increment")
        if not self._run_start_prepared and copied_increment:
            raise ValueError("the first request increment must be empty")

        route_status = self._request_router.call_route_status(
            self._requested_route,
            continuation=continuation,
        )
        await asyncio.to_thread(context_encoding_for_model, route_status.model)
        estimator_version = context_estimator_version_for_model(route_status.model)
        memory_route_status = self._request_router.call_route_status("memory", continuation=None)
        effective_tools = deepcopy(list(tools))
        current_user = deepcopy(self._request_current_user)
        if not self._run_start_prepared:
            prepared_messages = await self._prepare_run_start(
                route_status=route_status,
                memory_route_status=memory_route_status,
                tools=effective_tools,
            )
            self._run_start_prepared = True
        else:
            prepared_messages = await self._prepare_react(
                increment=copied_increment,
                latest_cycle_start=latest_cycle_start,
                route_status=route_status,
                memory_route_status=memory_route_status,
                tools=effective_tools,
                continuation_revision=continuation_revision,
                micro_compression_enabled=self._micro_compression_enabled,
            )

        request_messages = deepcopy(list(prepared_messages))
        micro_compression_enabled = (
            self._enable_tool_micro_compression
            and is_micro_compression_eligible is not None
            and _micro_compression_eligible_count(
                request_messages,
                is_micro_compression_eligible=is_micro_compression_eligible,
            )
            > _MICRO_COMPRESSION_TOOL_CALL_THRESHOLD
        )
        if micro_compression_enabled:
            assert is_micro_compression_eligible is not None
            request_messages = self._project_tool_results(
                request_messages,
                _micro_compression_omission_indices(
                    request_messages,
                    is_micro_compression_eligible=is_micro_compression_eligible,
                ),
            )

        final_estimated_tokens = self._estimate_candidate_tokens(
            request_messages,
            effective_tools,
            model=route_status.model,
        )
        budget = ContextBudget(
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            compact_ratio=self._compact_ratio,
        )
        revision = self._context_revision(
            current_user=current_user,
            tools=effective_tools,
            projected=request_messages,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=self._compact_ratio,
            estimator_version=estimator_version,
            increment=copied_increment,
            latest_cycle_start=latest_cycle_start,
            continuation_revision=continuation_revision,
            micro_compression_enabled=micro_compression_enabled,
        )
        if budget.exceeds_available_context(final_estimated_tokens):
            overflow_error = model_context_overflow_error()
            self._record_failure(revision, overflow_error)
            raise overflow_error

        self._checked_preparation_revision = revision
        self._micro_compression_enabled = micro_compression_enabled
        return deepcopy(request_messages)

    def record_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
    ) -> dict[str, object]:
        """Record one completed main request and return its persisted context usage."""
        route_status = self._request_router.current_call_status(self._requested_route)
        if route_status is None:
            raise RuntimeError("response recording requires one completed Model call")
        estimator_version = context_estimator_version_for_model(route_status.model)
        anchor_estimated_tokens = self.estimate_request_tokens(
            [*request_messages, response.message.to_dict()],
            tools,
            model=route_status.model,
        )
        current_user = deepcopy(self._request_current_user)
        run_messages: list[dict[str, Any]] = []
        if current_user is not None and not self._current_user_compacted:
            run_messages.append(current_user)
        run_messages.extend(deepcopy(list(increment)))
        run_projected_tokens = estimate_context_run_slice_tokens(
            run_messages, model=route_status.model
        )
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
        return context.to_dict()

    async def _prepare_run_start(
        self,
        *,
        route_status: ModelRouteStatus,
        memory_route_status: ModelRouteStatus,
        tools: Sequence[dict[str, Any]] = (),
    ) -> tuple[dict[str, Any], ...]:
        """Check and stage Run-start history compression without publishing Session state."""
        compact_ratio = self._compact_ratio
        estimator_version = context_estimator_version_for_model(route_status.model)
        self._preparation_estimates.clear()
        budget = ContextBudget(
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            compact_ratio=compact_ratio,
        )
        effective_tools = tuple(deepcopy(list(tools)))
        copied_user = deepcopy(self._request_current_user)
        projected = self._project_candidate(
            current_user=copied_user,
        )
        projection = self._compaction_projection(
            projected,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
            budget=budget,
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
        if self._failed_context_revision == revision:
            assert self._failed_exception is not None
            raise self._failed_exception
        self._failed_context_revision = None
        self._failed_exception = None
        if self._checked_preparation_revision == revision:
            return tuple(deepcopy(projected))

        protected_projection = self._projected_tokens(
            raw_messages=(),
            current_user=copied_user,
            tools=effective_tools,
            route_status=route_status,
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
            batch, cutoff = self._select_run_start_batch(budget, model=route_status.model)
        else:
            batch = tuple(deepcopy(list(pending_fact[0])))
            cutoff = pending_fact[1]
        if batch and pending_fact is None:
            retained_projection = self._projected_tokens(
                raw_messages=self._snapshot.messages[cutoff:],
                current_user=copied_user,
                tools=effective_tools,
                route_status=route_status,
            )
            if budget.exceeds_available_context(retained_projection.projected_tokens):
                all_batch, all_cutoff = self._batch_from_runs(self._eligible_runs())
                if all_batch and all_cutoff != cutoff:
                    all_projection = self._projected_tokens(
                        raw_messages=self._snapshot.messages[all_cutoff:],
                        current_user=copied_user,
                        tools=effective_tools,
                        route_status=route_status,
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
                if not self._enable_tool_micro_compression and budget.exceeds_available_context(
                    self._estimate_candidate_tokens(
                        projected, effective_tools, model=route_status.model
                    )
                ):
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
            current_user=copied_user,
        )
        final_tokens = self._estimate_candidate_tokens(
            final_projected, effective_tools, model=route_status.model
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
        if not self._enable_tool_micro_compression and budget.exceeds_available_context(
            final_tokens
        ):
            overflow_error = model_context_overflow_error()
            self._record_failure(final_revision, overflow_error)
            raise overflow_error
        return tuple(deepcopy(final_projected))

    async def _prepare_react(
        self,
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        route_status: ModelRouteStatus,
        memory_route_status: ModelRouteStatus,
        tools: Sequence[dict[str, Any]] = (),
        continuation_revision: int = 0,
        micro_compression_enabled: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Prepare one ReAct request from the run's raw increment."""
        compact_ratio = self._compact_ratio
        estimator_version = context_estimator_version_for_model(route_status.model)
        self._preparation_estimates.clear()
        budget = ContextBudget(
            context_window=route_status.context_window,
            max_output=route_status.max_output,
            compact_ratio=compact_ratio,
        )
        effective_tools = tuple(deepcopy(list(tools)))
        copied_user = deepcopy(self._request_current_user)
        projected = self._project_candidate(
            current_user=copied_user,
            increment=increment,
        )
        projection = self._compaction_projection(
            projected,
            tools=effective_tools,
            route_status=route_status,
            estimator_version=estimator_version,
            budget=budget,
        )
        revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
            increment=increment,
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
            increment,
            current_user=copied_user,
            latest_cycle_start=latest_cycle_start,
            tools=effective_tools,
            route_status=route_status,
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
                increment,
                current_user=copied_user,
                latest_cycle_start=latest_cycle_start,
                model=route_status.model,
            )
        else:
            batch = tuple(deepcopy(list(pending_fact[0])))
            cutoff = pending_fact[1]
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
        final_projected = self._project_candidate(
            current_user=copied_user,
            increment=increment,
        )
        final_revision = self._context_revision(
            current_user=copied_user,
            tools=effective_tools,
            projected=final_projected,
            route_status=route_status,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
            increment=increment,
            latest_cycle_start=latest_cycle_start,
            continuation_revision=continuation_revision,
            micro_compression_enabled=micro_compression_enabled,
        )
        self._checked_preparation_revision = final_revision
        return tuple(deepcopy(final_projected))

    def _react_protected_projection(
        self,
        increment: Sequence[dict[str, Any]],
        *,
        current_user: dict[str, Any] | None,
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
    ) -> ContextProjection:
        if latest_cycle_start is None:
            protected_increment: Sequence[dict[str, Any]] = ()
        else:
            increment_start = min(max(latest_cycle_start, 0), len(increment))
            protected_increment = increment[increment_start:]
        projected = self._project_messages(
            (),
            current_user,
            protected_increment,
            0,
            self._pending_action_summary,
        )
        return ContextProjection(
            self._estimate_candidate_tokens(projected, tools, model=route_status.model),
            "estimated",
        )

    def _select_react_batch(
        self,
        budget: ContextBudget,
        increment: Sequence[dict[str, Any]],
        *,
        current_user: dict[str, Any] | None,
        latest_cycle_start: int | None,
        model: str,
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
        current_fits = budget.can_retain_run_slice(current_slice, percentage=50, model=model)
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
        *,
        current_user: dict[str, Any] | None,
        increment: Sequence[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        return self._project_messages(
            self._snapshot.messages,
            current_user,
            increment,
            self._pending_last_compacted,
            self._pending_action_summary,
        )

    def _projected_tokens(
        self,
        *,
        raw_messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
    ) -> ContextProjection:
        projected = self._project_messages(
            raw_messages,
            current_user,
            (),
            0,
            self._pending_action_summary,
        )
        return ContextProjection(
            self._estimate_candidate_tokens(projected, tools, model=route_status.model),
            "estimated",
        )

    def _estimate_candidate_tokens(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        *,
        model: str,
    ) -> int:
        key = (
            model,
            json.dumps([messages, tools], sort_keys=True, ensure_ascii=False, allow_nan=False),
        )
        if key not in self._preparation_estimates:
            self._preparation_estimates[key] = self.estimate_request_tokens(
                messages, tools, model=model
            )
        return self._preparation_estimates[key]

    def _compaction_projection(
        self,
        projected: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        route_status: ModelRouteStatus,
        estimator_version: str,
        budget: ContextBudget,
    ) -> ContextProjection:
        anchor = self._latest_usage_anchor
        if anchor is not None and _usage_context_matches(
            anchor[0], route_status, estimator_version
        ):
            historical_tokens = reported_model_usage_total(anchor[1])
            if historical_tokens is not None and budget.should_compact(historical_tokens):
                return ContextProjection(historical_tokens, "reported_delta")
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
            self._estimate_candidate_tokens(projected, tools, model=route_status.model),
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
            _compaction_user_context(list(batch)) if pending_fact is None else pending_fact[2]
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
                fact_response = await self._request_router.complete(
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
                await self._append_summary(
                    fact_response.message.content,
                    self._persisted_now(),
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
            self._pending_fact = (tuple(deepcopy(list(batch))), cutoff, selected_payload)

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
            action_response = await self._request_router.complete(
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
        self, budget: ContextBudget, *, model: str
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
                    model=model,
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


def _micro_compression_omission_indices(
    messages: Sequence[dict[str, Any]],
    *,
    is_micro_compression_eligible: Callable[[str], bool],
) -> frozenset[int]:
    cycle_start = _latest_completed_cycle_start(messages)
    if cycle_start is None:
        return frozenset()

    return frozenset(
        index
        for index, message in enumerate(messages[:cycle_start])
        if message.get("role") == "tool"
        and isinstance(message.get("name"), str)
        and is_micro_compression_eligible(message["name"])
        and isinstance(message.get("content"), str)
        and len(message["content"]) > _TOOL_RESULT_MICRO_COMPRESSION_CHAR_LIMIT
    )


def _completed_run_ranges(messages: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    starts = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
        for index, start in enumerate(starts)
    ]


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
        if context.requested_route not in {"chat", "schedule", "subagent"}:
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


def _summary_request_messages(*, template_name: str, selected_payload: str) -> ModelMessages:
    return [
        {"role": "system", "content": render_template(template_name)},
        {"role": "user", "content": selected_payload},
    ]


def _compaction_user_context(messages: list[dict[str, Any]]) -> str:
    records = [
        projected
        for message in messages
        if (projected := project_history_message(message)) is not None
    ]
    serialized = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).translate(_COMPACTION_JSON_TRANSLATION)
    return f"## Conversation Messages\n\n```json\n{serialized}\n```"
