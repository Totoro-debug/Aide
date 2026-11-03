"""Validation for stable options attached to foreground questions."""

from __future__ import annotations

from typing import TypedDict

MAX_QUESTION_OPTION_COUNT = 4
MAX_QUESTION_OPTION_FIELD_LENGTH = 200


class ForegroundQuestionOption(TypedDict):
    id: str
    label: str


def validate_question_options(value: object) -> tuple[ForegroundQuestionOption, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_QUESTION_OPTION_COUNT:
        raise ValueError(f"question options must contain at most {MAX_QUESTION_OPTION_COUNT} items")
    normalized: list[ForegroundQuestionOption] = []
    seen_ids: set[str] = set()
    for option in value:
        if not isinstance(option, dict) or set(option) != {"id", "label"}:
            raise ValueError("each question option must contain only id and label")
        option_id = option["id"]
        label = option["label"]
        if (
            not isinstance(option_id, str)
            or not option_id
            or option_id.strip() != option_id
            or len(option_id) > MAX_QUESTION_OPTION_FIELD_LENGTH
        ):
            raise ValueError("option id must be a non-empty stable identifier")
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > MAX_QUESTION_OPTION_FIELD_LENGTH
        ):
            raise ValueError(
                f"option label must contain 1 to {MAX_QUESTION_OPTION_FIELD_LENGTH} characters"
            )
        if option_id in seen_ids:
            raise ValueError("question option IDs must be unique")
        seen_ids.add(option_id)
        normalized.append({"id": option_id, "label": label})
    return tuple(normalized)
