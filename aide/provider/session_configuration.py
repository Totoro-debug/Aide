"""Session-scoped model selection values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

type ReasoningEffort = Literal["low", "mid", "high", "xhigh", "max"]
REASONING_EFFORT_LEVELS: Final[tuple[ReasoningEffort, ...]] = (
    "low",
    "mid",
    "high",
    "xhigh",
    "max",
)


@dataclass(frozen=True, slots=True)
class SessionModelConfiguration:
    """One Conversation Session's selected Available Model and Reasoning Effort."""

    provider_id: str
    model: str
    reasoning_effort: ReasoningEffort

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id must be a nonempty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if self.reasoning_effort not in REASONING_EFFORT_LEVELS:
            raise ValueError("reasoning_effort is unsupported")

    @classmethod
    def from_dict(cls, value: object) -> SessionModelConfiguration:
        if not isinstance(value, Mapping) or set(value) != {
            "provider_id",
            "model",
            "reasoning_effort",
        }:
            raise ValueError("Session Model Configuration is malformed")
        provider_id = value["provider_id"]
        model = value["model"]
        reasoning_effort = value["reasoning_effort"]
        if not isinstance(provider_id, str) or not isinstance(model, str):
            raise ValueError("Session Model Configuration is malformed")
        if not isinstance(reasoning_effort, str):
            raise ValueError("Session Model Configuration is malformed")
        return cls(provider_id, model, cast(ReasoningEffort, reasoning_effort))

    def to_dict(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }
