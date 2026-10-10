"""Explicit model selections used when exercising configuration editing."""

from typing import Any

from aide.config.config import (
    MemoryConfiguration,
    ModelConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)

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



def configuration() -> UserConfiguration:
    provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="anthropic",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models={'default-model': ModelConfiguration(100_000, 4096, 0.2, "mid", 120)},
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="default-model",
        context_window=100_000,
        max_output=4096,
        temperature=0.2,
        reasoning_effort="mid",
        timeout=120,
    )
    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=50_000),
        memory=MemoryConfiguration(
            batch_size=10,
            schedule="0 * * * *",
        ),
        models=ModelsConfiguration(
            providers={provider.provider_id: provider},
            routes={"chat": route},
        ),
    )


def routed_configuration() -> UserConfiguration:
    default = configuration()
    chat_provider = ProviderConfiguration(
        provider_id="chat-provider",
        protocol="openai-compatible",
        base_url="https://chat.example/v1",
        api_key="chat-secret",
        models={'chat-model': ModelConfiguration(200_000, 8192, 0.1, "high", 90)},
    )
    chat_route = RouteConfiguration(
        provider_id=chat_provider.provider_id,
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        temperature=0.1,
        reasoning_effort="high",
        timeout=90,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={**default.models.providers, chat_provider.provider_id: chat_provider},
            routes={**default.models.routes, "title": chat_route},
        ),
    )


