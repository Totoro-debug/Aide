"""Strict JSON parsing shared by persisted and model-generated documents."""

import json
from math import isfinite
from typing import Literal, NoReturn


def json_validation_issue(
    value: object,
    *,
    field: str,
) -> tuple[str, Literal["type", "value"]] | None:
    """Return the first path and category that violates standard JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return None
    if isinstance(value, float):
        return None if isfinite(value) else (field, "value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            issue = json_validation_issue(item, field=f"{field}[{index}]")
            if issue is not None:
                return issue
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return field, "type"
            issue = json_validation_issue(item, field=f"{field}.{key}")
            if issue is not None:
                return issue
        return None
    return field, "type"


def strict_json_loads(content: str) -> object:
    """Decode standard JSON while rejecting duplicate object keys."""
    return json.loads(
        content,
        object_pairs_hook=_object_from_pairs,
        parse_constant=_reject_json_constant,
    )


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant: {value}")
