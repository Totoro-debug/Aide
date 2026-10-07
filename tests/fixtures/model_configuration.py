"""Explicit model selections used when exercising configuration editing."""

from typing import Any

TEST_MODEL_PARAMETERS = {
    "context_window": 8192, "max_output": 1024, "temperature": 0.2,
    "reasoning_effort": "mid", "timeout": 120,
}


def complete_model_settings(models: dict[str, Any]) -> None:
    """Choose a legacy candidate and supply explicit defaults for fixture-only models."""
    for provider in models["providers"].values():
        for model, parameters in provider["models"].items():
            candidates = parameters.pop("migration_candidates", [])
            if candidates:
                selected = {key: value for key, value in candidates[0].items() if key != "route"}
                provider["models"][model] = selected
            else:
                for field, value in TEST_MODEL_PARAMETERS.items():
                    if parameters.get(field) is None:
                        parameters[field] = value
