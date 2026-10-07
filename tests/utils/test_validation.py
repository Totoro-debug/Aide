import pytest

from aide.utils.validation import (
    TokenUsageValidationIssue,
    empty_token_usage,
    token_usage_validation_issue,
)


def test_empty_token_usage_returns_independent_zeroed_contracts() -> None:
    first = empty_token_usage()
    second = empty_token_usage()
    expected = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    assert first == second == expected
    assert first is not second

    first["model_calls"] = 1
    second["input_tokens"] = 2

    assert first == {**expected, "model_calls": 1}
    assert second == {**expected, "input_tokens": 2}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"model_calls": 1, "input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            None,
        ),
        ({"model_calls": 1, "input_tokens": 7, "output_tokens": 3}, "fields"),
        (
            {
                "model_calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "extra": 0,
            },
            "fields",
        ),
        (
            {
                "model_calls": True,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {
                "model_calls": 1,
                "input_tokens": 7.0,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {
                "model_calls": -1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {"model_calls": 1, "input_tokens": 7, "output_tokens": 3, "total_tokens": 11},
            "total",
        ),
    ],
)
def test_token_usage_validation_issue_classifies_shared_contract(
    value: dict[str, object],
    expected: TokenUsageValidationIssue | None,
) -> None:
    assert token_usage_validation_issue(value) == expected
