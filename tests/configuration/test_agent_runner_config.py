from pathlib import Path

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import (
    ConfigLoader,
    DefaultValueDiagnostic,
    RuntimeConfiguration,
)


def _config(max_iterations: str | None = None) -> str:
    runtime = "max_tool_result_chars = 4096"
    if max_iterations is not None:
        runtime += f"\nmax_iterations = {max_iterations}"
    return f"""[runtime]
{runtime}

[memory]
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "secret"

[models.providers.primary.models."model"]
context_window = 100000
max_output = 2048
temperature = 0.2
reasoning_effort = "mid"
timeout = 30

[models.routes.chat]
provider_id = "primary"
model = "model"
"""


def _load(tmp_path: Path, max_iterations: str | None = None) -> int:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config(max_iterations), encoding="utf-8")
    return loader.load().runtime.max_iterations


def test_max_iterations_defaults_to_fifty_when_omitted(tmp_path: Path) -> None:
    assert _load(tmp_path) == 50


def test_runtime_configuration_direct_construction_keeps_fifty_default() -> None:
    assert RuntimeConfiguration(max_tool_result_chars=4096).max_iterations == 50


@pytest.mark.parametrize("value", ("50", "51", "1000000000"))
def test_max_iterations_accepts_fifty_and_larger_integers(tmp_path: Path, value: str) -> None:
    assert _load(tmp_path, value) == int(value)


def test_config_view_recognizes_max_iterations_as_a_defined_runtime_field(tmp_path: Path) -> None:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config("50"), encoding="utf-8")

    assert loader.view().error is None


@pytest.mark.parametrize("value", ("49", "0", "-1", "true", "50.0", '"50"'))
def test_max_iterations_falls_back_for_invalid_values(
    tmp_path: Path, value: str
) -> None:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config(value), encoding="utf-8")

    configuration = loader.load()

    assert configuration.runtime.max_iterations == 50
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, DefaultValueDiagnostic)
    assert diagnostic.field == "runtime.max_iterations"
