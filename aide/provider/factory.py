"""Model Provider adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from aide.config.config import ProviderConfiguration

if TYPE_CHECKING:
    from aide.provider.models import ModelProvider


def _adapter_for_protocol(protocol: str) -> Callable[[ProviderConfiguration], ModelProvider]:
    if protocol == "anthropic":
        from aide.provider.anthropic import AnthropicProvider

        return AnthropicProvider
    if protocol == "openai-compatible":
        from aide.provider.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider
    raise ValueError(f"Unsupported Provider protocol: {protocol}")


def prepare_provider_adapters(protocols: Iterable[str]) -> None:
    """Load configured adapters before readiness without constructing clients."""
    for protocol in dict.fromkeys(protocols):
        _adapter_for_protocol(protocol)


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Construct the adapter selected by one validated Provider configuration."""
    return _adapter_for_protocol(configuration.protocol)(configuration)
