"""Explicit model selections used when exercising configuration editing."""

from typing import Any

TEST_MODEL_PARAMETERS = {
    "context_window": 8192, "max_output": 1024, "temperature": 0.2,
    "reasoning_effort": "mid", "timeout": 120,
}


def complete_model_settings(models: dict[str, Any]) -> None:
    """Supply explicit defaults for incomplete fixture-only models."""
    for provider in models["providers"].values():
        for parameters in provider["models"].values():
            for field, value in TEST_MODEL_PARAMETERS.items():
                if parameters.get(field) is None:
                    parameters[field] = value
