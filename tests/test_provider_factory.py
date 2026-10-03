import pytest

from omni.config.config import ProviderConfiguration
from omni.provider.anthropic import AnthropicProvider
from omni.provider.factory import create_provider
from omni.provider.openai_compatible import OpenAICompatibleProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "base_url", "expected_type"),
    [
        ("anthropic", "https://api.anthropic.com", AnthropicProvider),
        ("openai-compatible", "https://models.example/v1", OpenAICompatibleProvider),
    ],
)
async def test_configured_provider_factory_selects_the_official_sdk_adapter(
    protocol: str,
    base_url: str,
    expected_type: type[AnthropicProvider] | type[OpenAICompatibleProvider],
) -> None:
    configuration = ProviderConfiguration(
        provider_id="configured-provider",
        protocol=protocol,
        base_url=base_url,
        api_key="test-api-key",
        models=("model-id",),
    )

    provider = create_provider(configuration)

    assert type(provider) is expected_type
    await provider.close()
