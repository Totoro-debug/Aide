from dataclasses import replace

import pytest

from aide.agent.context.budget import (
    CONTEXT_ESTIMATOR_VERSION,
    ContextBudget,
    ContextProjection,
    ContextUsageSnapshot,
    estimate_request_tokens,
    estimate_run_slice_tokens,
    project_next_request_tokens,
    reported_model_usage_total,
    request_fits_model_context,
)


def _snapshot(**updates: object) -> ContextUsageSnapshot:
    values: dict[str, object] = {
        "requested_route": "chat",
        "selected_route": "chat",
        "provider_id": "primary",
        "model": "model-a",
        "context_window": 1000,
        "max_output": 100,
        "anchor_estimated_tokens": 80,
        "estimator_version": CONTEXT_ESTIMATOR_VERSION,
        "run_projected_tokens": 20,
        "run_projection_source": "estimated",
    }
    values.update(updates)
    return ContextUsageSnapshot(**values)  # type: ignore[arg-type]


def _usage() -> dict[str, object]:
    return {
        "model_calls": 1,
        "input_tokens": 60,
        "output_tokens": 10,
        "total_tokens": 70,
    }


def _project(
    estimated_tokens: int = 100,
    *,
    snapshot: ContextUsageSnapshot | None = None,
    reported_usage: dict[str, object] | None = None,
    **route_updates: object,
) -> ContextProjection:
    route: dict[str, object] = {
        "requested_route": "chat",
        "selected_route": "chat",
        "provider_id": "primary",
        "model": "model-a",
        "context_window": 1000,
        "max_output": 100,
    }
    route.update(route_updates)
    return project_next_request_tokens(
        estimated_tokens,
        snapshot=snapshot,
        reported_usage=reported_usage,
        **route,  # type: ignore[arg-type]
    )


def test_context_budget_uses_available_input_and_ceiling_for_soft_limit() -> None:
    budget = ContextBudget(context_window=8192, max_output=2048, compact_ratio=0.9)

    assert budget.available_context == 6144
    assert budget.compact_context_window == 5530
    assert not budget.should_compact(5529)
    assert budget.should_compact(5530)
    assert not budget.exceeds_available_context(6143)
    assert budget.exceeds_available_context(6144)


def test_context_budget_ceil_uses_the_configured_decimal_ratio() -> None:
    budget = ContextBudget(context_window=500, max_output=50, compact_ratio=0.54)

    assert budget.compact_context_window == 243
    assert not budget.should_compact(242)
    assert budget.should_compact(243)


@pytest.mark.parametrize("ratio", [0.5, 0.95])
def test_context_budget_accepts_inclusive_ratio_boundaries(ratio: float) -> None:
    assert ContextBudget(100, 10, ratio).compact_ratio == ratio


@pytest.mark.parametrize("ratio", [0.49, 0.96, float("nan"), float("inf"), True])
def test_context_budget_rejects_invalid_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="compact_ratio"):
        ContextBudget(100, 10, ratio)


def test_context_budget_rejects_a_nonpositive_available_context() -> None:
    with pytest.raises(ValueError, match="context_window"):
        ContextBudget(100, 100, 0.9)


def test_context_budget_soft_trigger_includes_equality() -> None:
    budget = ContextBudget(100, 10, 0.5)

    assert not budget.should_compact(44)
    assert budget.should_compact(45)


def test_context_budget_hard_failure_includes_equality() -> None:
    budget = ContextBudget(100, 10, 0.5)

    assert not budget.exceeds_available_context(89)
    assert budget.exceeds_available_context(90)
    assert budget.exceeds_available_context(91)


def test_request_estimate_counts_canonical_messages_and_tool_schema_delta() -> None:
    messages = [
        {"role": "system", "content": "abcd"},
        {"role": "user", "content": "hello"},
    ]

    assert estimate_request_tokens(messages) == 10
    assert (
        estimate_request_tokens(
            messages,
            tools=[{"name": "read_file", "parameters": {"type": "object"}}],
        )
        == 22
    )


def test_request_estimate_is_stable_for_equivalent_tool_schema_key_orders() -> None:
    first = {
        "name": "read_file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    reordered = {
        "parameters": {"properties": {"path": {"type": "string"}}, "type": "object"},
        "name": "read_file",
    }

    assert estimate_request_tokens([], [first]) == estimate_request_tokens([], [reordered])


@pytest.mark.parametrize(
    ("available_delta", "expected"),
    (
        pytest.param(1, True, id="below"),
        pytest.param(0, False, id="equal"),
        pytest.param(-1, False, id="above"),
    ),
)
def test_request_fit_uses_a_strict_hard_context_boundary(
    available_delta: int,
    expected: bool,
) -> None:
    messages = [
        {"role": "system", "content": "fixed"},
        {"role": "user", "content": "request"},
    ]
    estimated = estimate_request_tokens(messages)
    max_output = 8

    assert (
        request_fits_model_context(
            messages,
            (),
            context_window=estimated + max_output + available_delta,
            max_output=max_output,
        )
        is expected
    )


def test_request_fit_counts_tools_at_the_hard_context_boundary() -> None:
    messages = [{"role": "user", "content": "request"}]
    tools = [{"type": "function", "function": {"name": "work", "parameters": {}}}]
    estimated_with_tools = estimate_request_tokens(messages, tools)
    max_output = 8

    assert request_fits_model_context(
        messages,
        (),
        context_window=estimated_with_tools + max_output,
        max_output=max_output,
    )
    assert not request_fits_model_context(
        messages,
        tools,
        context_window=estimated_with_tools + max_output,
        max_output=max_output,
    )
    assert request_fits_model_context(
        messages,
        tools,
        context_window=estimated_with_tools + max_output + 1,
        max_output=max_output,
    )


def test_context_usage_snapshot_round_trips_exact_shape() -> None:
    snapshot = _snapshot()

    assert snapshot.to_dict() == {
        "requested_route": "chat",
        "selected_route": "chat",
        "provider_id": "primary",
        "model": "model-a",
        "context_window": 1000,
        "max_output": 100,
        "anchor_estimated_tokens": 80,
        "estimator_version": CONTEXT_ESTIMATOR_VERSION,
        "run_projected_tokens": 20,
        "run_projection_source": "estimated",
    }
    assert ContextUsageSnapshot.from_dict(snapshot.to_dict()) == snapshot


def test_compatible_reported_usage_projects_from_one_response_anchor() -> None:
    projection = _project(snapshot=_snapshot(), reported_usage=_usage())

    assert projection == ContextProjection(90, "reported_delta")


@pytest.mark.parametrize(
    ("reported_usage", "expected_total", "expected_projection"),
    (
        pytest.param(
            {"model_calls": 1, "input_tokens": 0, "output_tokens": 10, "total_tokens": 10},
            10,
            30,
            id="zero-input",
        ),
        pytest.param(
            {"model_calls": 1, "input_tokens": 60, "output_tokens": 0, "total_tokens": 60},
            60,
            80,
            id="zero-output",
        ),
        pytest.param(
            {"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            0,
            20,
            id="zero-total",
        ),
    ),
)
def test_zero_reported_usage_values_remain_compatible(
    reported_usage: dict[str, object],
    expected_total: int,
    expected_projection: int,
) -> None:
    assert reported_model_usage_total(reported_usage) == expected_total
    assert _project(
        estimated_tokens=100,
        snapshot=_snapshot(anchor_estimated_tokens=80),
        reported_usage=reported_usage,
    ) == ContextProjection(expected_projection, "reported_delta")


def test_reported_delta_does_not_recount_the_same_user_or_tool_input() -> None:
    projection = _project(
        estimated_tokens=100,
        snapshot=_snapshot(anchor_estimated_tokens=80),
        reported_usage={
            "model_calls": 1,
            "input_tokens": 60,
            "output_tokens": 10,
            "total_tokens": 70,
        },
    )

    assert projection.projected_tokens == 90


def test_session_cumulative_usage_cannot_be_used_as_a_response_anchor() -> None:
    projection = _project(
        estimated_tokens=123,
        snapshot=_snapshot(),
        reported_usage={
            "model_calls": 2,
            "input_tokens": 60,
            "output_tokens": 10,
            "total_tokens": 70,
        },
    )

    assert projection == ContextProjection(123, "estimated")


def test_negative_reported_delta_clamps_at_zero() -> None:
    projection = _project(
        estimated_tokens=40,
        snapshot=_snapshot(anchor_estimated_tokens=120),
        reported_usage={
            "model_calls": 1,
            "input_tokens": 60,
            "output_tokens": 10,
            "total_tokens": 70,
        },
    )

    assert projection == ContextProjection(0, "reported_delta")


@pytest.mark.parametrize(
    "reported_usage",
    (
        pytest.param(None, id="absent"),
        pytest.param(
            {"model_calls": 1, "input_tokens": 60, "output_tokens": 10},
            id="missing-field",
        ),
        pytest.param(
            {"model_calls": 1, "input_tokens": 60, "output_tokens": 10, "total_tokens": 71},
            id="total-mismatch",
        ),
        pytest.param(
            {"model_calls": True, "input_tokens": 60, "output_tokens": 10, "total_tokens": 70},
            id="boolean",
        ),
        pytest.param(
            {"model_calls": 1, "input_tokens": -1, "output_tokens": 10, "total_tokens": 9},
            id="negative",
        ),
        pytest.param(
            {"model_calls": 2, "input_tokens": 60, "output_tokens": 10, "total_tokens": 70},
            id="multiple-model-calls",
        ),
        pytest.param(
            {
                "model_calls": 1,
                "input_tokens": 60,
                "output_tokens": 10,
                "total_tokens": 70,
                "unexpected": 1,
            },
            id="extra-field",
        ),
    ),
)
def test_invalid_reported_usage_falls_back_to_estimate(
    reported_usage: dict[str, object] | None,
) -> None:
    assert reported_model_usage_total(reported_usage) is None
    projection = _project(estimated_tokens=123, snapshot=_snapshot(), reported_usage=reported_usage)

    assert projection == ContextProjection(123, "estimated")


@pytest.mark.parametrize(
    "route_updates",
    [
        {"requested_route": "schedule"},
        {"selected_route": "default"},
        {"provider_id": "secondary"},
        {"model": "model-b"},
        {"context_window": 2000},
        {"max_output": 200},
        {"estimator_version": "future-v2"},
    ],
)
def test_incompatible_usage_provenance_falls_back_to_estimate(
    route_updates: dict[str, object],
) -> None:
    projection = _project(
        estimated_tokens=123,
        snapshot=_snapshot(),
        reported_usage=_usage(),
        **route_updates,
    )

    assert projection == ContextProjection(123, "estimated")


def test_tool_schema_change_is_included_in_reported_delta() -> None:
    initial = _project(estimated_tokens=80, snapshot=_snapshot(), reported_usage=_usage())
    with_tool = _project(estimated_tokens=100, snapshot=_snapshot(), reported_usage=_usage())

    assert initial == ContextProjection(70, "reported_delta")
    assert with_tool == ContextProjection(90, "reported_delta")


def test_non_target_context_change_is_included_in_the_complete_candidate_delta() -> None:
    base_request = [
        {"role": "system", "content": "fixed"},
        {"role": "user", "content": "hello"},
    ]
    changed_request = [
        {"role": "system", "content": "fixed context changed"},
        {"role": "user", "content": "hello"},
    ]
    base_tokens = estimate_request_tokens(base_request)
    changed_tokens = estimate_request_tokens(changed_request)
    usage: dict[str, object] = {
        "model_calls": 1,
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
    }
    snapshot = _snapshot(anchor_estimated_tokens=base_tokens)

    assert base_tokens == 10
    assert changed_tokens == 14
    assert _project(base_tokens, snapshot=snapshot, reported_usage=usage) == ContextProjection(
        20, "reported_delta"
    )
    assert _project(changed_tokens, snapshot=snapshot, reported_usage=usage) == ContextProjection(
        24, "reported_delta"
    )


def test_snapshot_copy_can_change_run_projection_without_changing_route_identity() -> None:
    snapshot = replace(_snapshot(), run_projected_tokens=42)

    assert snapshot.run_projected_tokens == 42
    assert snapshot.requested_route == "chat"
    assert _project(80, snapshot=snapshot, reported_usage=_usage()).source == "reported_delta"


def test_run_slice_budget_uses_only_raw_target_messages_and_available_context() -> None:
    model = "test-model"
    run_slice = [{"role": "user", "content": "hello"}]
    slice_tokens = estimate_run_slice_tokens(run_slice, model=model)

    assert ContextBudget(slice_tokens * 10, 0, 0.9).can_retain_run_slice(
        run_slice, percentage=10, model=model
    )
    assert not ContextBudget(slice_tokens * 10 - 1, 0, 0.9).can_retain_run_slice(
        run_slice, percentage=10, model=model
    )
    assert ContextBudget(slice_tokens * 2, 0, 0.9).can_retain_run_slice(
        run_slice, percentage=50, model=model
    )
    assert not ContextBudget(slice_tokens * 2 - 1, 0, 0.9).can_retain_run_slice(
        run_slice, percentage=50, model=model
    )


def test_run_slice_budget_rejects_non_run_inputs_and_unsupported_percentages() -> None:
    with pytest.raises(ValueError, match="run slice"):
        ContextBudget(100, 0, 0.9).can_retain_run_slice(
            [{"role": "system", "content": "fixed"}], percentage=10, model="test-model"
        )
    with pytest.raises(ValueError, match="percentage"):
        ContextBudget(100, 0, 0.9).can_retain_run_slice(
            [{"role": "user", "content": "hello"}],
            percentage=25,  # type: ignore[arg-type]
            model="test-model",
        )
    with pytest.raises(ValueError, match="percentage"):
        ContextBudget(100, 0, 0.9).can_retain_run_slice(
            [{"role": "user", "content": "hello"}],
            percentage=10.0,  # type: ignore[arg-type]
            model="test-model",
        )
