"""Strict JSON parsing shared by persisted and model-generated documents."""

import json
from typing import NoReturn


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
