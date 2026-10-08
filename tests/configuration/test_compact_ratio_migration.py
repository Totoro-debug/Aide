from pathlib import Path

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader, DefaultValueDiagnostic

VALID_CONFIG = """[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false

[memory]
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "secret"
models = ["model"]

[models.routes.chat]
provider_id = "primary"
model = "model"
context_window = 100000
max_output = 2048
temperature = 0.2
timeout = 30
"""


def _loader(tmp_path: Path, content: str = VALID_CONFIG) -> ConfigLoader:
    loader = ConfigLoader(AgentHome(tmp_path / "agent-home"))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")
    return loader


@pytest.mark.parametrize("value", (0.5, 0.75, 0.9, 0.95))
def test_compact_ratio_accepts_supported_values_without_diagnostic(
    tmp_path: Path,
    value: float,
) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", f"[runtime]\ncompact_ratio = {value}\n")
    )

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == value
    assert loader.diagnostics == ()


@pytest.mark.parametrize(
    "value",
    ("true", '"0.75"', "nan", "inf", "0.49", "0.96"),
)
def test_invalid_compact_ratio_falls_back_once_with_safe_diagnostic(
    tmp_path: Path,
    value: str,
) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", f"[runtime]\ncompact_ratio = {value}\n")
    )

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == 0.9
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, DefaultValueDiagnostic)
    assert diagnostic.field == "runtime.compact_ratio"
    assert diagnostic.message == (
        "Configuration field 'runtime.compact_ratio' is invalid; using 0.9."
    )
    assert not hasattr(diagnostic, "mcp_name")


def test_missing_compact_ratio_uses_default_without_diagnostic(tmp_path: Path) -> None:
    loader = _loader(tmp_path, VALID_CONFIG.replace("compact_ratio = 0.9\n", ""))

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == 0.9
    assert loader.diagnostics == ()


def test_config_view_exposes_effective_ratio_before_raw_content(tmp_path: Path) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", "[runtime]\ncompact_ratio = 0.75\n")
    )

    view = loader.view()

    assert view.error is None
    assert view.effective_compact_ratio == 0.75
    assert view.redacted_content.index("compact_ratio = 0.75") >= 0


def test_malformed_config_has_no_effective_ratio(tmp_path: Path) -> None:
    loader = _loader(tmp_path, "[runtime\ncompact_ratio = 0.75\n")

    view = loader.view()

    assert view.error is not None
    assert view.effective_compact_ratio is None


def test_removed_threshold_is_not_a_memory_configuration_field(tmp_path: Path) -> None:
    legacy_content = VALID_CONFIG.replace(
        "[runtime]\n",
        "[runtime]\ncompact_ratio = 0.9\n",
    ).replace(
        "[memory]\n",
        "[memory]\ncompaction_message_threshold = 40\n",
    )
    loader = _loader(tmp_path, legacy_content)

    configuration = loader.load()
    view = loader.view()

    assert not hasattr(configuration.memory, "compaction_message_threshold")
    assert view.error is None
    assert "compaction_message_threshold = 40" in view.redacted_content


def test_runtime_configuration_defaults_other_known_invalid_fields(tmp_path: Path) -> None:
    loader = _loader(tmp_path, VALID_CONFIG.replace("max_iterations = 50", "max_iterations = 49"))

    configuration = loader.load()

    assert configuration.runtime.max_iterations == 50
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, DefaultValueDiagnostic)
    assert diagnostic.field == "runtime.max_iterations"
