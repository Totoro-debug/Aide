"""Pure token-budget values and Model Request Context projections."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Self, cast

from myclaw.utils.validation import require_nonnegative_int, token_usage_validation_issue

CONTEXT_ESTIMATOR_VERSION = "utf8-bytes-div4-v1"
type ProjectionSource = Literal["estimated", "reported_delta"]
type RetentionPercentage = Literal[10, 50]

_MODEL_ROUTES = frozenset({"default", "chat", "memory", "schedule"})
_PROJECTION_SOURCES = frozenset({"estimated", "reported_delta"})
_RUN_MESSAGE_ROLES = frozenset({"user", "assistant", "tool"})
_CONTEXT_USAGE_FIELDS = frozenset(
    {
        "requested_route",
        "selected_route",
        "provider_id",
        "model",
        "context_window",
        "max_output",
        "anchor_estimated_tokens",
        "estimator_version",
        "run_projected_tokens",
        "run_projection_source",
    }
)

__all__ = [
    "CONTEXT_ESTIMATOR_VERSION",
    "ContextBudget",
    "ContextProjection",
    "ContextUsageSnapshot",
    "ProjectionSource",
    "RetentionPercentage",
    "estimate_request_tokens",
    "estimate_run_slice_tokens",
    "project_next_request_tokens",
    "reported_model_usage_total",
    "request_fits_model_context",
]


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Soft and hard input limits for one concrete Model Route."""

    context_window: int
    max_output: int
    compact_ratio: float

    def __post_init__(self) -> None:
        require_nonnegative_int(self.context_window, field="context_window")
        require_nonnegative_int(self.max_output, field="max_output")
        if self.context_window <= self.max_output:
            raise ValueError("context_window must be greater than max_output")
        if (
            isinstance(self.compact_ratio, bool)
            or not isinstance(self.compact_ratio, (int, float))
            or not math.isfinite(self.compact_ratio)
            or not 0.5 <= self.compact_ratio <= 0.95
        ):
            raise ValueError("compact_ratio must be between 0.5 and 0.95")

    @property
    def available_context(self) -> int:
        """Return input capacity after reserving the route's output budget."""
        return self.context_window - self.max_output

    @property
    def compact_context_window(self) -> int:
        """Return the inclusive soft compaction threshold."""
        return math.ceil(self.available_context * Decimal(str(self.compact_ratio)))

    def should_compact(self, projected_tokens: int) -> bool:
        """Return whether a projection reaches the soft compaction threshold."""
        require_nonnegative_int(projected_tokens, field="projected_tokens")
        return projected_tokens >= self.compact_context_window

    def exceeds_available_context(self, projected_tokens: int) -> bool:
        """Return whether a projection reaches the hard input capacity."""
        require_nonnegative_int(projected_tokens, field="projected_tokens")
        return projected_tokens >= self.available_context

    def can_retain_run_slice(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        percentage: RetentionPercentage,
    ) -> bool:
        """Return whether a target Run's raw message slice fits its retention share."""
        if type(percentage) is not int or percentage not in (10, 50):
            raise ValueError("run slice percentage must be 10 or 50")
        slice_tokens = estimate_run_slice_tokens(messages)
        return slice_tokens * 100 <= self.available_context * percentage


@dataclass(frozen=True, slots=True)
class ContextUsageSnapshot:
    """Persisted provenance for one main Agent assistant response."""

    requested_route: str
    selected_route: str
    provider_id: str
    model: str
    context_window: int
    max_output: int
    anchor_estimated_tokens: int
    estimator_version: str
    run_projected_tokens: int
    run_projection_source: ProjectionSource

    def __post_init__(self) -> None:
        if not isinstance(self.requested_route, str) or self.requested_route not in _MODEL_ROUTES:
            raise ValueError("requested_route is not supported")
        if not isinstance(self.selected_route, str) or self.selected_route not in _MODEL_ROUTES:
            raise ValueError("selected_route is not supported")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("provider_id must not be empty")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must not be empty")
        require_nonnegative_int(self.context_window, field="context_window")
        require_nonnegative_int(self.max_output, field="max_output")
        if self.context_window <= self.max_output:
            raise ValueError("context_window must be greater than max_output")
        require_nonnegative_int(self.anchor_estimated_tokens, field="anchor_estimated_tokens")
        if not isinstance(self.estimator_version, str) or not self.estimator_version:
            raise ValueError("estimator_version must not be empty")
        require_nonnegative_int(self.run_projected_tokens, field="run_projected_tokens")
        if (
            not isinstance(self.run_projection_source, str)
            or self.run_projection_source not in _PROJECTION_SOURCES
        ):
            raise ValueError("run_projection_source is not supported")

    def to_dict(self) -> dict[str, object]:
        """Encode the exact persisted context-usage shape."""
        return {
            "requested_route": self.requested_route,
            "selected_route": self.selected_route,
            "provider_id": self.provider_id,
            "model": self.model,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "anchor_estimated_tokens": self.anchor_estimated_tokens,
            "estimator_version": self.estimator_version,
            "run_projected_tokens": self.run_projected_tokens,
            "run_projection_source": self.run_projection_source,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Decode one exact persisted context-usage shape."""
        if not isinstance(value, dict):
            raise TypeError("context_usage must be a dictionary")
        if set(value) != _CONTEXT_USAGE_FIELDS:
            raise ValueError("context_usage has an invalid shape")
        try:
            return cls(
                requested_route=value["requested_route"],
                selected_route=value["selected_route"],
                provider_id=value["provider_id"],
                model=value["model"],
                context_window=value["context_window"],
                max_output=value["max_output"],
                anchor_estimated_tokens=value["anchor_estimated_tokens"],
                estimator_version=value["estimator_version"],
                run_projected_tokens=value["run_projected_tokens"],
                run_projection_source=value["run_projection_source"],
            )
        except TypeError as error:
            raise TypeError("context_usage contains invalid field types") from error


@dataclass(frozen=True, slots=True)
class ContextProjection:
    """Projected occupancy and the measurement source used to derive it."""

    projected_tokens: int
    source: ProjectionSource

    def __post_init__(self) -> None:
        require_nonnegative_int(self.projected_tokens, field="projected_tokens")
        if not isinstance(self.source, str) or self.source not in _PROJECTION_SOURCES:
            raise ValueError("projection source is not supported")


def estimate_request_tokens(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] = (),
) -> int:
    """Estimate one provider-neutral request as ceil(canonical UTF-8 bytes / 4)."""
    system_prompt = ""
    retained = messages
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise TypeError("system message content must be a string")
        system_prompt = content
        retained = messages[1:]

    components = [system_prompt]
    components.extend(_canonical_json(message) for message in retained)
    components.extend(_canonical_json(tool) for tool in tools)
    byte_count = sum(len(component.encode("utf-8")) for component in components)
    return (byte_count + 3) // 4


def request_fits_model_context(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    context_window: int,
    max_output: int,
) -> bool:
    """Return whether one request is strictly below the hard input boundary."""
    budget = ContextBudget(
        context_window=context_window,
        max_output=max_output,
        compact_ratio=0.9,
    )
    return not budget.exceeds_available_context(estimate_request_tokens(messages, tools))


def estimate_run_slice_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Estimate only a target Agent Run's uncompacted raw User/assistant/Tool slice."""
    if any(message.get("role") not in _RUN_MESSAGE_ROLES for message in messages):
        raise ValueError("run slice must contain only user, assistant, and tool messages")
    return estimate_request_tokens(messages)


def project_next_request_tokens(
    estimated_tokens: int,
    *,
    snapshot: ContextUsageSnapshot | None,
    reported_usage: Mapping[str, object] | None,
    requested_route: str,
    selected_route: str,
    provider_id: str,
    model: str,
    context_window: int,
    max_output: int,
    estimator_version: str = CONTEXT_ESTIMATOR_VERSION,
) -> ContextProjection:
    """Prefer one compatible reported-usage anchor, otherwise use the full estimate."""
    require_nonnegative_int(estimated_tokens, field="estimated_tokens")
    if snapshot is None or not _snapshot_matches(
        snapshot,
        requested_route=requested_route,
        selected_route=selected_route,
        provider_id=provider_id,
        model=model,
        context_window=context_window,
        max_output=max_output,
        estimator_version=estimator_version,
    ):
        return ContextProjection(estimated_tokens, "estimated")

    reported_total = reported_model_usage_total(reported_usage)
    if reported_total is None:
        return ContextProjection(estimated_tokens, "estimated")

    projected = max(0, reported_total + estimated_tokens - snapshot.anchor_estimated_tokens)
    return ContextProjection(projected, "reported_delta")


def _snapshot_matches(
    snapshot: ContextUsageSnapshot,
    *,
    requested_route: str,
    selected_route: str,
    provider_id: str,
    model: str,
    context_window: int,
    max_output: int,
    estimator_version: str,
) -> bool:
    return (
        snapshot.requested_route == requested_route
        and snapshot.selected_route == selected_route
        and snapshot.provider_id == provider_id
        and snapshot.model == model
        and snapshot.context_window == context_window
        and snapshot.max_output == max_output
        and snapshot.estimator_version == estimator_version
    )


def reported_model_usage_total(value: Mapping[str, object] | None) -> int | None:
    """Return one response's validated usage total, including a legitimate zero."""
    if value is None or token_usage_validation_issue(value) is not None:
        return None
    if value["model_calls"] != 1:
        return None
    return cast(int, value["total_tokens"])


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=True,
    )
