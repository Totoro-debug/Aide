"""Model Provider adapter construction."""

from omni.config.config import ProviderConfiguration
from omni.provider.anthropic import AnthropicProvider
from omni.provider.models import ModelProvider
from omni.provider.openai_compatible import OpenAICompatibleProvider


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Construct the adapter selected by one validated Provider configuration."""
    if configuration.protocol == "anthropic":
        return AnthropicProvider(configuration)
    if configuration.protocol == "openai-compatible":
        return OpenAICompatibleProvider(configuration)
    raise ValueError(f"Unsupported Provider protocol: {configuration.protocol}")
