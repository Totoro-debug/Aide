from pathlib import Path

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import (
    ConfigError,
    ConfigLoader,
    DefaultValueDiagnostic,
    UserConfiguration,
)

BASE_CONFIG = '''[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false
enable_tool_micro_compression = false
compact_ratio = 0.9
permission_level = "workspace-write"
exec_shell = "auto"

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
'''


DEFAULTABLE_FIELDS = (
    (
        "runtime.enable_tool_micro_compression",
        "enable_tool_micro_compression = false",
        "enable_tool_micro_compression = true",
        'enable_tool_micro_compression = "true"',
        False,
        True,
        "false",
        '"true"',
    ),
    (
        "runtime.max_tool_result_chars",
        "max_tool_result_chars = 4096",
        "max_tool_result_chars = 8192",
        "max_tool_result_chars = 999",
        4096,
        8192,
        "4096",
        "999",
    ),
    (
        "runtime.max_iterations",
        "max_iterations = 50",
        "max_iterations = 75",
        "max_iterations = 49",
        50,
        75,
        "50",
        "49",
    ),
    (
        "runtime.enable_skill_always_load",
        "enable_skill_always_load = false",
        "enable_skill_always_load = true",
        'enable_skill_always_load = "true"',
        False,
        True,
        "false",
        '"true"',
    ),
    (
        "runtime.compact_ratio",
        "compact_ratio = 0.9",
        "compact_ratio = 0.75",
        "compact_ratio = 0.49",
        0.9,
        0.75,
        "0.9",
        "0.49",
    ),
    (
        "runtime.permission_level",
        'permission_level = "workspace-write"',
        'permission_level = "read-only"',
        'permission_level = "admin"',
        "workspace-write",
        "read-only",
        "'workspace-write'",
        "admin",
    ),
    (
        "runtime.exec_shell",
        'exec_shell = "auto"',
        'exec_shell = "pwsh"',
        'exec_shell = "bash"',
        "auto",
        "pwsh",
        "'auto'",
        "bash",
    ),
    (
        "memory.batch_size",
        "batch_size = 10",
        "batch_size = 25",
        "batch_size = 0",
        10,
        25,
        "10",
        "batch_size = 0",
    ),
    (
        "memory.schedule",
        'schedule = "0 * * * *"',
        'schedule = "15 * * * *"',
        'schedule = "not-a-cron"',
        "0 * * * *",
        "15 * * * *",
        "'0 * * * *'",
        "not-a-cron",
    ),
)


def _loader(tmp_path: Path, content: str = BASE_CONFIG) -> ConfigLoader:
    loader = ConfigLoader(AgentHome(tmp_path / "agent-home"))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")
    return loader


def _effective_value(configuration: UserConfiguration, field: str) -> object:
    parts = field.split(".")
    if parts[:2] == ["models", "routes"]:
        return configuration.models.routes[parts[2]].__getattribute__(parts[3])
    return getattr(getattr(configuration, parts[0]), parts[1])


@pytest.mark.parametrize("mode", ("valid", "missing", "invalid"))
@pytest.mark.parametrize(
    (
        "field",
        "line",
        "valid_line",
        "invalid_line",
        "default",
        "valid",
        "default_text",
        "invalid_text",
    ),
    DEFAULTABLE_FIELDS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_defaultable_configuration_fields_have_one_fallback_contract(
    tmp_path: Path,
    mode: str,
    field: str,
    line: str,
    valid_line: str,
    invalid_line: str,
    default: object,
    valid: object,
    default_text: str,
    invalid_text: str,
) -> None:
    content = BASE_CONFIG
    if mode == "valid":
        content = content.replace(line, valid_line)
    elif mode == "missing":
        content = content.replace(f"{line}\n", "")
    else:
        content = content.replace(line, invalid_line)

    loader = _loader(tmp_path, content)
    configuration = loader.load()

    expected = valid if mode == "valid" else default
    assert _effective_value(configuration, field) == expected
    if mode != "invalid":
        assert loader.diagnostics == ()
        return

    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, DefaultValueDiagnostic)
    assert diagnostic.field == field
    assert diagnostic.message == f"Configuration field {field!r} is invalid; using {default_text}."
    assert invalid_text not in diagnostic.message


def test_config_view_exposes_effective_permission_and_exec_shell(tmp_path: Path) -> None:
    loader = _loader(
        tmp_path,
        BASE_CONFIG.replace(
            'permission_level = "workspace-write"',
            'permission_level = "full-access"',
        ).replace('exec_shell = "auto"', 'exec_shell = "powershell"'),
    )

    view = loader.view()

    assert view.error is None
    assert view.effective_permission_level == "full-access"
    assert view.effective_exec_shell == "powershell"
    assert view.effective_values_text() == (
        "Effective runtime.compact_ratio: 0.9\n"
        "Effective runtime.permission_level: full-access\n"
        "Effective runtime.exec_shell: powershell\n"
    )


@pytest.mark.parametrize(
    ("source_line", "invalid_line", "field", "default", "default_text"),
    (
        (
            "max_tool_result_chars = 4096",
            "max_tool_result_chars = true",
            "runtime.max_tool_result_chars",
            4096,
            "4096",
        ),
        (
            "max_iterations = 50",
            "max_iterations = 50.0",
            "runtime.max_iterations",
            50,
            "50",
        ),
        (
            "enable_skill_always_load = false",
            'enable_skill_always_load = "flag-secret"',
            "runtime.enable_skill_always_load",
            False,
            "false",
        ),
        (
            "compact_ratio = 0.9",
            'compact_ratio = "ratio-secret"',
            "runtime.compact_ratio",
            0.9,
            "0.9",
        ),
        (
            'permission_level = "workspace-write"',
            'permission_level = ["not-a-level"]',
            "runtime.permission_level",
            "workspace-write",
            "'workspace-write'",
        ),
        (
            'exec_shell = "auto"',
            'exec_shell = { name = "bash-secret" }',
            "runtime.exec_shell",
            "auto",
            "'auto'",
        ),
        ("batch_size = 10", "batch_size = true", "memory.batch_size", 10, "10"),
        (
            'schedule = "0 * * * *"',
            'schedule = ["cron-secret"]',
            "memory.schedule",
            "0 * * * *",
            "'0 * * * *'",
        ),
    ),
)
def test_untyped_defaultable_values_use_sanitized_fallbacks(
    tmp_path: Path,
    source_line: str,
    invalid_line: str,
    field: str,
    default: object,
    default_text: str,
) -> None:
    loader = _loader(tmp_path, BASE_CONFIG.replace(source_line, invalid_line))

    configuration = loader.load()

    assert _effective_value(configuration, field) == default
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, DefaultValueDiagnostic)
    assert diagnostic.field == field
    assert diagnostic.message == f"Configuration field {field!r} is invalid; using {default_text}."
    assert "secret" not in diagnostic.message


def _config_with_route_reasoning(route_name: str, reasoning_line: str | None) -> str:
    replacement = "" if reasoning_line is None else reasoning_line
    content = BASE_CONFIG.replace('reasoning_effort = "mid"', replacement)
    if route_name != "chat":
        content += f'\n[models.routes.{route_name}]\nprovider_id = "primary"\nmodel = "model"\n'
    return content


@pytest.mark.parametrize("route_name", ("chat", "title", "memory", "schedule", "subagent"))
@pytest.mark.parametrize("reasoning_line", ('reasoning_effort = "high"', None, 'reasoning_effort = ["route-secret"]'))
def test_each_route_uses_explicit_valid_model_reasoning(
    tmp_path: Path, route_name: str, reasoning_line: str | None,
) -> None:
    loader = _loader(tmp_path, _config_with_route_reasoning(route_name, reasoning_line))
    before = loader.path.read_bytes()
    if reasoning_line == 'reasoning_effort = "high"':
        assert loader.load().models.routes[route_name].reasoning_effort == "high"
    else:
        with pytest.raises(ConfigError) as raised:
            loader.load()
        assert "models.providers.primary.models.model.reasoning_effort" in raised.value.field_errors
        assert "route-secret" not in str(raised.value)
    assert loader.path.read_bytes() == before


def test_repeated_load_and_view_do_not_accumulate_default_diagnostics(tmp_path: Path) -> None:
    content = BASE_CONFIG.replace(
        'permission_level = "workspace-write"',
        'permission_level = "level-secret"',
    ).replace('exec_shell = "auto"', 'exec_shell = "shell-secret"')
    loader = _loader(tmp_path, content)

    loader.load()
    first = loader.diagnostics
    loader.load()
    second = loader.diagnostics
    view = loader.view()

    expected_fields = ["runtime.permission_level", "runtime.exec_shell"]
    for diagnostics in (first, second, view.diagnostics, loader.diagnostics):
        assert all(isinstance(diagnostic, DefaultValueDiagnostic) for diagnostic in diagnostics)
        assert [
            diagnostic.field
            for diagnostic in diagnostics
            if isinstance(diagnostic, DefaultValueDiagnostic)
        ] == expected_fields
    assert "secret" not in view.diagnostics_text()


def test_config_header_keeps_fallback_diagnostics_when_later_fields_are_fatal(
    tmp_path: Path,
) -> None:
    content = BASE_CONFIG.replace(
        'permission_level = "workspace-write"',
        'permission_level = "level-secret"',
    ).replace('model = "model"\n', "", 1)
    loader = _loader(tmp_path, content)

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert view.effective_values_text() == ""
    assert view.header_text() == (
        "config_invalid: Configuration field 'models.routes.chat.model' is required.\n"
        "Configuration field 'runtime.permission_level' is invalid; "
        "using 'workspace-write'.\n"
        f"Path: {loader.path}\n"
    )
    assert "level-secret" not in view.header_text()
