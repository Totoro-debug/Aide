"""Model Provider adapter construction."""

from aide.config.config import ProviderConfiguration
from aide.provider.anthropic import AnthropicProvider
from aide.provider.models import ModelProvider
from aide.provider.openai_compatible import OpenAICompatibleProvider


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Construct the adapter selected by one validated Provider configuration."""
    if configuration.protocol == "anthropic":
        return AnthropicProvider(configuration)
    if configuration.protocol == "openai-compatible":
        return OpenAICompatibleProvider(configuration)
    raise ValueError(f"Unsupported Provider protocol: {configuration.protocol}")
