"""Blackboard task framing value object and direct Model Route generation."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from omni.templates import render_template
from omni.utils.json import strict_json_loads
from omni.utils.validation import token_usage_validation_issue

if TYPE_CHECKING:
    from omni.agent.runner import AgentRunnerRouter

_DECISION_KEYS = frozenset({"action", "task_goal", "completion_boundary"})
_NOT_PARSED = object()


@dataclass(frozen=True, slots=True)
class Blackboard:
    """The canonical two-field task state used by Task Framing."""

    goal: str
    completion_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not isinstance(self.completion_boundary, str):
            raise TypeError("Blackboard fields must be strings")
        goal = self.goal.strip()
        completion_boundary = self.completion_boundary.strip()
        if not goal or not completion_boundary:
            raise ValueError("Blackboard fields must not be blank")
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "completion_boundary", completion_boundary)

    @classmethod
    def from_dict(cls, value: object) -> Blackboard | None:
        """Decode an optional persisted Blackboard shape without raising."""
        if not isinstance(value, dict) or set(value) != {"goal", "completion_boundary"}:
            return None
        goal = value["goal"]
        completion_boundary = value["completion_boundary"]
        if not isinstance(goal, str) or not isinstance(completion_boundary, str):
            return None
        try:
            return cls(goal=goal, completion_boundary=completion_boundary)
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict[str, str]:
        """Encode this Blackboard using its canonical persisted shape."""
        return {
            "goal": self.goal,
            "completion_boundary": self.completion_boundary,
        }

    @classmethod
    async def generate(
        cls,
        router: AgentRunnerRouter,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult:
        """Resolve one raw input using one isolated chat completion."""
        _validate_frame_inputs(
            previous=previous,
            last_assistant_content=last_assistant_content,
            current_user_input=current_user_input,
        )
        last_task = json.dumps(
            (
                None
                if previous is None
                else {
                    "task_goal": previous.goal,
                    "completion_boundary": previous.completion_boundary,
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            response = await router.complete(
                "chat",
                messages=[
                    {
                        "role": "system",
                        "content": _build_blackboard_system_prompt(
                            user_input=current_user_input,
                            last_task=last_task,
                            latest_assistant_content=last_assistant_content,
                        ),
                    },
                ],
                tools=(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return FramingResult(
                blackboard=None,
                usage_delta=None,
                status="model_failed",
            )

        usage_delta = {
            "model_calls": 1,
            **response.usage.to_dict(),
        }
        decision = _extract_decision(response.message.content)
        resolved, blackboard = _reduce_decision(decision, previous)
        return FramingResult(
            blackboard=blackboard,
            usage_delta=usage_delta,
            status="resolved" if resolved else "invalid_response",
        )


def _build_blackboard_system_prompt(
    *,
    user_input: str,
    last_task: str,
    latest_assistant_content: str,
) -> str:
    return render_template(
        "blackboard-system-prompt.md",
        **{
            "User input": user_input,
            "Last Task": last_task,
            "Latest assistant content": latest_assistant_content,
        },
    )


@dataclass(frozen=True, slots=True)
class FramingResult:
    """The isolated result returned by one Task Framing attempt."""

    blackboard: Blackboard | None
    usage_delta: dict[str, int] | None
    status: Literal["resolved", "invalid_response", "model_failed"]

    def __post_init__(self) -> None:
        if not isinstance(self.status, str):
            raise TypeError("Framing status must be a string")
        if self.status not in {"resolved", "invalid_response", "model_failed"}:
            raise ValueError("Framing status is invalid")
        if self.blackboard is not None and not isinstance(self.blackboard, Blackboard):
            raise TypeError("Framing blackboard must be a Blackboard or None")
        if self.usage_delta is not None and not isinstance(self.usage_delta, dict):
            raise TypeError("Framing usage must be a dictionary or None")
        if self.status == "resolved":
            if self.usage_delta is None:
                raise ValueError("Resolved framing must include usage")
        elif self.status == "invalid_response":
            if self.blackboard is not None:
                raise ValueError("Invalid framing response cannot include a Blackboard")
            if self.usage_delta is None:
                raise ValueError("Invalid framing response must include usage")
        elif self.blackboard is not None or self.usage_delta is not None:
            raise ValueError("Failed framing cannot include a Blackboard or usage")

        if self.usage_delta is None:
            return
        usage_issue = token_usage_validation_issue(self.usage_delta)
        if usage_issue == "fields":
            raise ValueError("Framing usage must contain exactly four fields")
        if usage_issue == "values":
            raise ValueError("Framing usage values must be nonnegative integers")
        if usage_issue == "total":
            raise ValueError("Framing usage total_tokens must equal input plus output")


def _validate_frame_inputs(
    *,
    previous: Blackboard | None,
    last_assistant_content: str,
    current_user_input: str,
) -> None:
    if previous is not None and not isinstance(previous, Blackboard):
        raise TypeError("previous must be a Blackboard or None")
    if not isinstance(last_assistant_content, str):
        raise TypeError("last_assistant_content must be a string")
    if not isinstance(current_user_input, str):
        raise TypeError("current_user_input must be a string")


def _extract_decision(content: str) -> object:
    stripped = content.strip()
    direct = _loads_json(stripped)
    if direct is not _NOT_PARSED:
        return direct

    fenced = _fenced_json(stripped)
    if fenced is not _NOT_PARSED:
        return fenced
    if "```" in stripped:
        return _NOT_PARSED

    candidate = _first_balanced_json_object(content)
    if candidate is None:
        return _NOT_PARSED
    return _loads_json(candidate)


def _loads_json(content: str) -> object:
    if not content:
        return _NOT_PARSED
    try:
        return strict_json_loads(content)
    except (json.JSONDecodeError, ValueError):
        return _NOT_PARSED


def _fenced_json(content: str) -> object:
    match = re.fullmatch(
        r"```[ \t]*(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return _NOT_PARSED
    return _loads_json(match.group(1).strip())


def _first_balanced_json_object(content: str) -> str | None:
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(content):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return content[start : index + 1]
    return None


def _reduce_decision(
    decision: object,
    previous: Blackboard | None,
) -> tuple[bool, Blackboard | None]:
    if not isinstance(decision, dict) or set(decision) != _DECISION_KEYS:
        return False, None
    action = decision["action"]
    task_goal = decision["task_goal"]
    completion_boundary = decision["completion_boundary"]
    if not isinstance(action, str):
        return False, None
    if action in {"keep", "clear"}:
        if task_goal is not None or completion_boundary is not None:
            return False, None
        return True, previous if action == "keep" else None
    if action != "replace":
        return False, None
    if not isinstance(task_goal, str) or not isinstance(completion_boundary, str):
        return False, None
    try:
        return True, Blackboard(goal=task_goal, completion_boundary=completion_boundary)
    except (TypeError, ValueError):
        return False, None


__all__ = [
    "Blackboard",
    "FramingResult",
]
