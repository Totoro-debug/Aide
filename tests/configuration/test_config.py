import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigError, ConfigLoader
from aide.utils.host_filesystem import HOST_FILESYSTEM

EXPECTED_DEFAULT_CONFIG = """[runtime]
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

[web]
default_chat_workspace = "~/.aide/chat"

# Configure a trusted local MCP Server by uncommenting and editing one item.
# [mcp.servers.filesystem]
# enabled = true
# transport = "stdio"
# command = "uvx"
# args = ["mcp-server-filesystem", "."]
# cwd = "."
# connect_timeout = 30
# call_timeout = 60
# [mcp.servers.filesystem.tool_keywords]
# list_files = ["files", "directory", "list"]

# Configure a trusted Streamable HTTP MCP Server with optional static headers.
# [mcp.servers.search]
# enabled = true
# transport = "streamable-http"
# url = "https://example.com/mcp"
# connect_timeout = 30
# call_timeout = 60
# [mcp.servers.search.tool_keywords]
# search = ["search", "query"]
# [mcp.servers.search.headers]
# Authorization = "Bearer replace-with-a-token"

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
# Configure each model once; routes below select its Provider and model ID.
[models.providers.openai-local.models."replace-with-a-model-id"]
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120

# Remove any optional purpose-specific route to fall back to chat.
[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.title]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.schedule]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.subagent]
provider_id = "openai-local"
model = "replace-with-a-model-id"
"""

VALID_CONFIG = """[runtime]
max_tool_result_chars = 60000
compact_ratio = 0.9

[memory]
batch_size = 12
schedule = "15 * * * *"

[models.providers.anthropic-default]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "sk-ant-secret"
models = ["claude-model"]

[models.routes.chat]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120
"""

MINIMAL_VALID_CONFIG = """[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "minimal-secret"
models = ["small-model"]

[runtime]
compact_ratio = 0.9

[models.routes.chat]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
"""

PROJECTED_CONFIG = """unexpected = true

[runtime]
max_tool_result_chars = 60000
ignored_runtime_field = true

[models]
ignored_models_field = true

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "minimal-secret"
models = ["small-model"]
ignored_provider_field = true

[models.routes.chat]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
ignored_route_field = true

[models.routes.cron]
this_legacy_route_is_ignored = true

[models.routes.future]
this_undefined_route_is_ignored = true
"""

SCHEDULE_ROUTE = """

[models.providers.schedule-provider]
protocol = "openai-compatible"
base_url = "https://schedule.example/v1"
api_key = "schedule-secret"
models = ["schedule-model"]

[models.routes.schedule]
provider_id = "schedule-provider"
model = "schedule-model"
context_window = 100000
max_output = 4096
temperature = 0.1
reasoning_effort = "high"
timeout = 90
"""

REDACTION_CONFIG = """# User Configuration
[runtime]
max_tool_result_chars = 50000
compact_ratio = 0.9

[memory]
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "plaintext-primary-key"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.chat]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0.2
timeout = 60
"""

EXPECTED_REDACTED_CONFIG = """# User Configuration
[runtime]
max_tool_result_chars = 50000
compact_ratio = 0.9

[memory]
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.chat]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0.2
timeout = 60
"""

OVERLAPPING_API_KEY_CONFIG = """# model-id must remain visible outside api_key
[models.providers.model-id]
protocol = "openai-compatible"
base_url = "https://model-id.example/v1/model-id"
api_key = "model-id"
models = ["model-id"]

[models.routes.chat]
provider_id = "model-id"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

EXPECTED_REDACTED_OVERLAPPING_CONFIG = """# model-id must remain visible outside api_key
[models.providers.model-id]
protocol = "openai-compatible"
base_url = "https://model-id.example/v1/model-id"
api_key = "***REDACTED***"
models = ["model-id"]

[models.routes.chat]
provider_id = "model-id"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MULTILINE_DOTTED_API_KEY_CONFIG = '''models.providers.primary.protocol = "anthropic"
models.providers.primary.base_url = "https://api.anthropic.com"
models.providers.primary.api_key = """line-one-secret
line-two-secret"""
models.providers.primary.models = ["model-id"]
models.routes.chat.provider_id = "primary"
models.routes.chat.model = "model-id"
models.routes.chat.context_window = 4096
models.routes.chat.max_output = 512
models.routes.chat.temperature = 0
models.routes.chat.timeout = 60
'''

EXPECTED_REDACTED_MULTILINE_DOTTED_CONFIG = """models.providers.primary.protocol = "anthropic"
models.providers.primary.base_url = "https://api.anthropic.com"
models.providers.primary.api_key = "***REDACTED***"
models.providers.primary.models = ["model-id"]
models.routes.chat.provider_id = "primary"
models.routes.chat.model = "model-id"
models.routes.chat.context_window = 4096
models.routes.chat.max_output = 512
models.routes.chat.temperature = 0
models.routes.chat.timeout = 60
"""

MALFORMED_CONFIG = """[runtime
max_tool_result_chars = 50000
api_key = "first-plaintext-key"
  API-Key = 'second-plaintext-key' # remove this whole value
not_api_key = "not-a-provider-key"
broken = [
"""

EXPECTED_REDACTED_MALFORMED_CONFIG = """[runtime
max_tool_result_chars = 50000
api_key = "***REDACTED***"
  API-Key = "***REDACTED***"
not_api_key = "not-a-provider-key"
broken = [
"""

SCHEMA_INVALID_API_KEY_ALIAS_CONFIG = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
API-Key = "sk-schema-alias-secret"
models = ["model-id"]

[models.routes.chat]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MALFORMED_DOTTED_API_KEY_CONFIG = """models.providers.primary.API-Key = "sk-dotted-secret"
models.providers.primary.protocol = "anthropic"
broken = [
"""

MALFORMED_QUOTED_DOTTED_API_KEY_CONFIG = """"models"."providers"."primary"."API-Key" = "sk-quoted-secret"
broken = [
"""

MALFORMED_INLINE_API_KEY_CONFIG = """models = { providers = { primary = { api_key = "sk-inline-secret" } } }
broken = [
"""

MALFORMED_INLINE_MULTILINE_API_KEY_CONFIG = '''models = { providers = { primary = { api_key = """sk-inline-line-one
sk-inline-line-two""" } } }
broken = [
'''

MALFORMED_INLINE_ARRAY_API_KEY_CONFIG = """models = { providers = { primary = { api_key = ["sk-inline-array-secret"] } } }
broken = [
"""

MALFORMED_MULTILINE_ARRAY_API_KEY_CONFIG = """models.providers.primary.api_key = [
  "sk-multiline-array-secret",
]
broken = [
"""

MALFORMED_MULTILINE_API_KEY_CONFIG = '''models.providers.primary.api_key = """sk-line-one
sk-line-two"""
models.providers.primary.protocol = "anthropic"
broken = [
'''


def test_missing_configuration_is_created_exactly_once(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True
    assert (agent_home / "config.toml").read_text(encoding="utf-8") == EXPECTED_DEFAULT_CONFIG

    existing = b"# Keep this existing configuration byte-for-byte.\n"
    (agent_home / "config.toml").write_bytes(existing)

    assert loader.ensure_default() is False
    assert (agent_home / "config.toml").read_bytes() == existing


def test_generated_configuration_scaffolds_one_provider_and_all_model_routes(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True

    models = loader.load().models
    assert set(models.providers) == {"openai-local"}
    provider = models.providers["openai-local"]
    assert (provider.protocol, provider.base_url, provider.api_key, provider.models) == (
        "openai-compatible",
        "",
        "",
        ("replace-with-a-model-id",),
    )

    routes = models.routes
    assert set(routes) == {"chat", "title", "memory", "schedule", "subagent"}
    for route in routes.values():
        assert (
            route.provider_id,
            route.model,
            route.context_window,
            route.max_output,
            route.temperature,
            route.reasoning_effort,
            route.timeout,
        ) == (
            "openai-local",
            "replace-with-a-model-id",
            200000,
            8192,
            0.2,
            "mid",
            120,
        )


def test_configuration_projects_undefined_fields_and_legacy_routes(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(PROJECTED_CONFIG, encoding="utf-8")

    configuration = loader.load_for_startup()

    assert set(configuration.models.routes) == {"chat"}
    assert set(configuration.models.providers) == {"primary"}
    assert configuration.runtime.max_tool_result_chars == 60000


@pytest.mark.parametrize("operation", ["startup", "view"])
@pytest.mark.parametrize(
    ("content", "marker", "expected_mcp"),
    [
        (
            "future_setting = true\n\n" + VALID_CONFIG,
            "future_setting = true",
            False,
        ),
        (
            VALID_CONFIG.replace(
                "max_tool_result_chars = 60000",
                "max_tool_result_chars = 60000\nfuture_runtime_setting = true",
            ),
            "future_runtime_setting = true",
            False,
        ),
        (
            VALID_CONFIG + "\n[tools.web]\nenabled = true\n",
            "enabled = true",
            False,
        ),
        (
            VALID_CONFIG + "\n[mcp.servers.future]\n"
            'enabled = true\ntransport = "stdio"\ncommand = "uvx"\n'
            "future_server_setting = true\n",
            "future_server_setting = true",
            True,
        ),
    ],
    ids=("top-level", "known-table", "static-table", "mcp-server"),
)
def test_unknown_configuration_projection_is_accepted_by_startup_and_config_view(
    agent_home: Path,
    operation: str,
    content: str,
    marker: str,
    expected_mcp: bool,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    if operation == "startup":
        configuration = loader.load_for_startup()

        assert loader.diagnostics == ()
        assert set(configuration.models.routes) == {"chat"}
        assert set(configuration.mcp) == ({"future"} if expected_mcp else set())
        return

    view = loader.view()

    assert view.error is None
    assert view.diagnostics == ()
    assert marker in view.redacted_content
    assert "sk-ant-secret" not in view.redacted_content


@pytest.mark.parametrize("operation", ["startup", "view"])
@pytest.mark.parametrize("value", ["0.5", '"not-a-ratio"'], ids=("valid", "invalid"))
def test_compact_ratio_is_loaded_or_falls_back_with_a_diagnostic(
    agent_home: Path,
    operation: str,
    value: str,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = VALID_CONFIG.replace("compact_ratio = 0.9\n", "").replace(
        "[runtime]\n",
        f"[runtime]\ncompact_ratio = {value}\n",
    )
    loader.path.write_text(content, encoding="utf-8")

    if operation == "startup":
        configuration = loader.load_for_startup()

        expected = 0.5 if value == "0.5" else 0.9
        assert configuration.runtime.compact_ratio == expected
        assert len(loader.diagnostics) == (0 if value == "0.5" else 1)
        return

    view = loader.view()

    assert view.error is None
    expected = 0.5 if value == "0.5" else 0.9
    assert view.effective_compact_ratio == expected
    assert len(view.diagnostics) == (0 if value == "0.5" else 1)


@pytest.mark.parametrize("operation", ["startup", "view"])
@pytest.mark.parametrize(
    ("content", "error_code", "field"),
    [
        (
            VALID_CONFIG.replace("[runtime]\n", "runtime = true\n"),
            "config_invalid",
            "runtime",
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"\n', ""),
            "config_invalid",
            "models.routes.chat.model",
        ),
        (
            VALID_CONFIG
            + """
[models.providers.invalid]
protocol = "anthropic"
base_url = "https://invalid.example"
api_key = "invalid-secret"
""",
            "config_invalid",
            "models.providers.invalid.models",
        ),
        (
            VALID_CONFIG + "\n[broken\nvalue = true\n",
            "config_parse_error",
            None,
        ),
    ],
    ids=("structural-table", "missing-required", "eager-provider", "malformed-toml"),
)
def test_startup_and_config_view_preserve_configuration_failures(
    agent_home: Path,
    operation: str,
    content: str,
    error_code: str,
    field: str | None,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    if operation == "startup":
        with pytest.raises(ConfigError) as raised:
            loader.load_for_startup()

        error = raised.value.error
    else:
        view = loader.view()
        assert view.error is not None
        error = view.error

    assert error.code == error_code
    if field is not None:
        assert field in error.message


def test_startup_gate_does_not_resolve_provider_or_model_usability(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = VALID_CONFIG.replace('protocol = "anthropic"', 'protocol = "future-protocol"')
    loader.path.write_text(content, encoding="utf-8")

    configuration = loader.load_for_startup()

    assert configuration.models.routes["chat"].provider_id == "anthropic-default"


def test_schedule_route_resolves_directly_and_falls_back_to_default_when_unusable(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG + SCHEDULE_ROUTE, encoding="utf-8")
    configuration = loader.load()

    resolved = configuration.resolve_route("schedule")

    assert (
        resolved.requested_route,
        resolved.selected_route,
        resolved.provider.provider_id,
        resolved.route.model,
        resolved.used_fallback,
    ) == ("schedule", "schedule", "schedule-provider", "schedule-model", False)

    loader.path.write_text(
        (VALID_CONFIG + SCHEDULE_ROUTE).replace(
            'protocol = "openai-compatible"', 'protocol = "future-protocol"', 1
        ),
        encoding="utf-8",
    )
    configuration = loader.load()

    fallback = configuration.resolve_route("schedule")

    assert (
        fallback.requested_route,
        fallback.selected_route,
        fallback.provider.provider_id,
        fallback.route.model,
        fallback.used_fallback,
    ) == ("schedule", "chat", "anthropic-default", "claude-model", True)


@pytest.mark.parametrize("purpose", ("title", "memory", "schedule", "subagent"))
@pytest.mark.parametrize("availability", ("configured", "missing", "missing_provider", "missing_model", "unusable_provider"))
def test_auxiliary_routes_use_chat_when_their_model_is_unavailable(
    agent_home: Path, purpose: str, availability: str,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = VALID_CONFIG
    if availability == "unusable_provider":
        content += """
[models.providers.unusable]
protocol = "future-protocol"
base_url = "https://unusable.example/v1"
api_key = "secret"
models = ["other-model"]
"""
    if availability != "missing":
        provider_id = "unusable" if availability == "unusable_provider" else (
            "absent" if availability == "missing_provider" else "anthropic-default"
        )
        model = "absent-model" if availability == "missing_model" else (
            "other-model" if availability == "unusable_provider" else "claude-model"
        )
        content += f"""
[models.routes.{purpose}]
provider_id = "{provider_id}"
model = "{model}"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120
"""
    loader.path.write_text(content, encoding="utf-8")

    resolved = loader.load_for_startup().resolve_route(purpose)

    selected = purpose if availability == "configured" else "chat"
    assert (resolved.requested_route, resolved.selected_route, resolved.used_fallback) == (
        purpose, selected, selected == "chat",
    )
    assert resolved.route.model == "claude-model"


def test_legacy_default_route_migrates_to_chat_on_next_write(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    legacy = VALID_CONFIG.replace("[models.routes.chat]", "[models.routes.default]")
    loader.path.write_text(legacy, encoding="utf-8")

    configuration = loader.load_for_startup()
    assert set(configuration.models.routes) == {"chat"}
    assert loader.path.read_text(encoding="utf-8") == legacy
    assert loader.web_snapshot().state == "active"
    assert "default" not in loader.web_snapshot().fields["models"]["routes"]
    loader.update_reasoning_effort("high")

    saved = loader.path.read_text(encoding="utf-8")
    assert "[models.routes.default]" not in saved
    assert "[models.routes.chat]" in saved
    with pytest.raises(ConfigError):
        loader.load().resolve_route("default")


def test_explicit_chat_route_wins_over_legacy_default(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        VALID_CONFIG + "\n[models.routes.default]\nprovider_id = \"absent\"\nmodel = \"old\"\n",
        encoding="utf-8",
    )

    configuration = loader.load_for_startup()

    assert configuration.resolve_route("memory").route.model == "claude-model"
    assert set(configuration.models.routes) == {"chat"}


def test_failed_startup_generation_leaves_no_partial_configuration(
    agent_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    def fail_publish(source: Path | str, destination: Path | str) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", fail_publish)

    with pytest.raises(ConfigError) as raised:
        loader.load_for_startup()

    assert raised.value.error.code == "persistence_error"
    assert not loader.path.exists()
    assert not tuple(agent_home.glob(".config.toml.*.tmp"))


def test_valid_configuration_loads_as_typed_values(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG, encoding="utf-8")

    configuration = loader.load()

    assert (
        configuration.runtime.max_tool_result_chars,
        configuration.runtime.compact_ratio,
        configuration.runtime.permission_level,
        configuration.runtime.exec_shell,
        configuration.memory.batch_size,
        configuration.memory.schedule,
        configuration.models.providers["anthropic-default"].models,
        configuration.models.routes["chat"].reasoning_effort,
        configuration.models.routes["chat"].timeout,
    ) == (
        60000,
        0.9,
        "workspace-write",
        "auto",
        12,
        "15 * * * *",
        ("claude-model",),
        "mid",
        120,
    )


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("low", "low"),
        ("mid", "mid"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ],
)
def test_all_reasoning_effort_levels_load_as_route_values(
    agent_home: Path, effort: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        VALID_CONFIG.replace('reasoning_effort = "mid"', f'reasoning_effort = "{effort}"'),
        encoding="utf-8",
    )

    before = loader.path.read_bytes()

    def unexpected_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Reading Reasoning Effort must not write configuration")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", unexpected_write)
    monkeypatch.setattr(Path, "write_text", unexpected_write)
    monkeypatch.setattr(Path, "write_bytes", unexpected_write)
    configuration = loader.load()

    assert all(route.reasoning_effort == expected for route in configuration.models.routes.values())
    assert loader.diagnostics == ()
    providers = loader.web_snapshot().fields["models"]["providers"]
    assert isinstance(providers, dict)
    assert all(model["reasoning_effort"] == expected for provider in providers.values()
               for model in provider["models"].values())
    assert loader.path.read_bytes() == before


def test_unsupported_reasoning_effort_uses_default_policy_without_rewriting(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG.replace('reasoning_effort = "mid"', 'reasoning_effort = "medium"'), encoding="utf-8")
    before = loader.path.read_bytes()
    configuration = loader.load_for_startup()
    assert all(route.reasoning_effort == "mid" for route in configuration.models.routes.values())
    assert loader.diagnostics
    assert loader.web_snapshot().fields["models"]["routes"]
    assert loader.path.read_bytes() == before


def test_explicit_always_load_setting_loads_as_a_boolean(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        VALID_CONFIG.replace("[runtime]\n", "[runtime]\nenable_skill_always_load = true\n"),
        encoding="utf-8",
    )

    configuration = loader.load()

    assert configuration.runtime.enable_skill_always_load is True


def test_generated_configuration_disables_always_load_by_default(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True
    assert "enable_skill_always_load = false" in loader.path.read_text(encoding="utf-8")


def test_omitted_defaulted_configuration_fields_use_accepted_defaults(
    agent_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    before_load = loader.path.read_text(encoding="utf-8")
    write_operations: list[str] = []

    def record_path_write(*_args: object, **_kwargs: object) -> int:
        write_operations.append("path")
        return 0

    def record_atomic_create(*_args: object, **_kwargs: object) -> bool:
        write_operations.append("atomic_create")
        return False

    def record_atomic_replace(*_args: object, **_kwargs: object) -> None:
        write_operations.append("atomic_replace")

    monkeypatch.setattr(Path, "write_text", record_path_write)
    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_create_text", record_atomic_create)
    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_atomic_replace)

    configuration = loader.load()

    assert (
        configuration.runtime.max_tool_result_chars,
        configuration.runtime.compact_ratio,
        configuration.runtime.permission_level,
        configuration.runtime.exec_shell,
        configuration.memory.batch_size,
        configuration.memory.schedule,
        configuration.models.routes["chat"].reasoning_effort,
        configuration.runtime.enable_skill_always_load,
        configuration.runtime.enable_tool_micro_compression,
    ) == (4096, 0.9, "workspace-write", "auto", 10, "0 * * * *", "mid", False, False)
    assert write_operations == []
    assert loader.path.read_text(encoding="utf-8") == before_load


def test_config_view_redacts_nonempty_provider_keys_and_preserves_complete_content(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(REDACTION_CONFIG, encoding="utf-8")

    view = loader.view()

    assert (view.path, view.redacted_content, view.error) == (
        loader.path,
        EXPECTED_REDACTED_CONFIG,
        None,
    )


def test_config_view_redacts_only_the_api_key_when_its_text_is_reused(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(OVERLAPPING_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is None
    assert view.redacted_content == EXPECTED_REDACTED_OVERLAPPING_CONFIG


def test_config_view_redacts_multiline_api_key_in_valid_dotted_toml(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MULTILINE_DOTTED_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    if "line-one-secret" in view.redacted_content or "line-two-secret" in view.redacted_content:
        pytest.fail("ConfigView leaked a multiline plaintext provider API key", pytrace=False)
    assert view.error is None
    assert view.redacted_content == EXPECTED_REDACTED_MULTILINE_DOTTED_CONFIG


@pytest.mark.parametrize(
    ("content", "secrets", "expected_error"),
    (
        pytest.param(
            SCHEMA_INVALID_API_KEY_ALIAS_CONFIG,
            ("sk-schema-alias-secret",),
            "config_invalid",
            id="schema-alias",
        ),
        pytest.param(
            '[diagnostics]\nAPI_Key = "sk-nested-secret"\nmessage = "keep"\n',
            ("sk-nested-secret",),
            None,
            id="unknown-table",
        ),
        pytest.param(
            '[diagnostics]\napi_key = ["sk-array-secret"]\nmessage = "keep"\n',
            ("sk-array-secret",),
            None,
            id="non-string-value",
        ),
        pytest.param(
            '[[diagnostics]]\napi_key = "sk-array-table-secret"\nmessage = "keep"\n',
            ("sk-array-table-secret",),
            None,
            id="array-table",
        ),
        pytest.param(
            r""""api\u005fkey" = "sk-valid-escaped-secret"
""",
            ("sk-valid-escaped-secret",),
            None,
            id="escaped-key",
        ),
    ),
)
def test_config_view_redacts_structured_api_key_variants(
    agent_home: Path,
    content: str,
    secrets: tuple[str, ...],
    expected_error: str | None,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    if expected_error is None:
        assert view.error is None
    else:
        assert view.error is not None
        assert view.error.code == expected_error
    assert all(secret not in view.redacted_content for secret in secrets)
    assert "***REDACTED***" in view.redacted_content


@pytest.mark.parametrize(
    ("content", "secrets"),
    (
        pytest.param(
            MALFORMED_DOTTED_API_KEY_CONFIG,
            ("sk-dotted-secret",),
            id="dotted",
        ),
        pytest.param(
            MALFORMED_QUOTED_DOTTED_API_KEY_CONFIG,
            ("sk-quoted-secret",),
            id="quoted-dotted",
        ),
        pytest.param(
            r""""api\u005fkey" = "sk-escaped-key-secret"
broken = [
""",
            ("sk-escaped-key-secret",),
            id="escaped-key",
        ),
        pytest.param(
            r""""\u0041\U00000050\u0049\u002d\U0000004b\u0045\U00000059" = "sk-fully-escaped-secret"
broken = [
""",
            ("sk-fully-escaped-secret",),
            id="fully-escaped-key",
        ),
        pytest.param(
            MALFORMED_INLINE_API_KEY_CONFIG,
            ("sk-inline-secret",),
            id="inline",
        ),
        pytest.param(
            MALFORMED_INLINE_MULTILINE_API_KEY_CONFIG,
            ("sk-inline-line-one", "sk-inline-line-two"),
            id="inline-multiline",
        ),
        pytest.param(
            MALFORMED_INLINE_ARRAY_API_KEY_CONFIG,
            ("sk-inline-array-secret",),
            id="inline-array",
        ),
        pytest.param(
            MALFORMED_MULTILINE_ARRAY_API_KEY_CONFIG,
            ("sk-multiline-array-secret",),
            id="multiline-array",
        ),
        pytest.param(
            MALFORMED_MULTILINE_API_KEY_CONFIG,
            ("sk-line-one", "sk-line-two"),
            id="multiline-string",
        ),
    ),
)
def test_config_view_redacts_malformed_api_key_variants(
    agent_home: Path,
    content: str,
    secrets: tuple[str, ...],
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert all(secret not in view.redacted_content for secret in secrets)
    assert "***REDACTED***" in view.redacted_content


def test_config_view_returns_safe_parse_error_and_conservatively_redacted_raw_text(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.path == loader.path
    assert view.redacted_content == EXPECTED_REDACTED_MALFORMED_CONFIG
    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "first-plaintext-key" not in view.error.message
    assert "second-plaintext-key" not in view.error.message


def test_config_view_ignores_undefined_configuration_fields(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is None
    assert view.diagnostics == ()
    assert "misspelled_setting = true" in view.redacted_content
    assert "plaintext-primary-key" not in view.redacted_content


@pytest.mark.parametrize(
    ("content", "field"),
    [
        (
            VALID_CONFIG.replace('protocol = "anthropic"', "protocol = 1"),
            "models.providers.anthropic-default.protocol",
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"', "base_url = 1"),
            "models.providers.anthropic-default.base_url",
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"\n', ""),
            "models.providers.anthropic-default.base_url",
        ),
        (
            VALID_CONFIG.replace('api_key = "sk-ant-secret"', "api_key = 1"),
            "models.providers.anthropic-default.api_key",
        ),
        (
            VALID_CONFIG.replace(
                'models = ["claude-model"]', 'models = ["claude-model", "claude-model"]'
            ),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('models = ["claude-model"]', 'models = [""]'),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('models = ["claude-model"]', "models = [1]"),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('provider_id = "anthropic-default"\n', ""),
            "models.routes.chat.provider_id",
        ),
        (
            VALID_CONFIG.replace('provider_id = "anthropic-default"', 'provider_id = "Bad_ID"'),
            "models.routes.chat.provider_id",
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"\n', ""),
            "models.routes.chat.model",
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"', 'model = ""'),
            "models.routes.chat.model",
        ),
        (
            VALID_CONFIG.replace("context_window = 200000", "context_window = 1023"),
            "models.routes.chat.context_window",
        ),
        (
            VALID_CONFIG.replace("context_window = 200000", "context_window = true"),
            "models.routes.chat.context_window",
        ),
        (
            VALID_CONFIG.replace("max_output = 8192", "max_output = 0"),
            "models.routes.chat.max_output",
        ),
        (
            VALID_CONFIG.replace("max_output = 8192", "max_output = 200000"),
            "models.routes.chat.max_output",
        ),
        (
            VALID_CONFIG.replace("temperature = 0.2", "temperature = nan"),
            "models.routes.chat.temperature",
        ),
        (
            VALID_CONFIG.replace("temperature = 0.2", 'temperature = "0.2"'),
            "models.routes.chat.temperature",
        ),
        (
            VALID_CONFIG.replace("timeout = 120", "timeout = 601"),
            "models.routes.chat.timeout",
        ),
    ],
)
def test_configuration_rejects_schema_violations(
    agent_home: Path, content: str, field: str
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        loader.load()

    assert raised.value.error.code == "config_invalid"
    assert field in raised.value.error.message
    assert "sk-ant-secret" not in str(raised.value)


UNUSABLE_CHAT_PROVIDER = """

[models.providers.chat-provider]
protocol = "anthropic"
base_url = "not-an-absolute-url"
api_key = "chat-secret"
models = ["chat-model"]

[models.routes.title]
provider_id = "chat-provider"
model = "chat-model"
context_window = 100000
max_output = 4096
temperature = 0.1
timeout = 90
"""

UNKNOWN_PROTOCOL_CHAT_PROVIDER = UNUSABLE_CHAT_PROVIDER.replace(
    'protocol = "anthropic"', 'protocol = "future-protocol"'
).replace('base_url = "not-an-absolute-url"', 'base_url = "https://future.example/v1"')

MISSING_PROVIDER_CHAT_ROUTE = """

[models.routes.title]
provider_id = "missing-provider"
model = "chat-model"
context_window = 100000
max_output = 4096
temperature = 0.1
timeout = 90
"""

OUTSIDE_CATALOG_CHAT_ROUTE = MISSING_PROVIDER_CHAT_ROUTE.replace(
    'provider_id = "missing-provider"', 'provider_id = "anthropic-default"'
)


@pytest.mark.parametrize(
    ("content", "requested_route", "expected_selected_route"),
    [
        (VALID_CONFIG, "title", "chat"),
        (VALID_CONFIG + MISSING_PROVIDER_CHAT_ROUTE, "title", "chat"),
        (VALID_CONFIG + OUTSIDE_CATALOG_CHAT_ROUTE, "title", "chat"),
        (VALID_CONFIG + UNUSABLE_CHAT_PROVIDER, "title", "chat"),
        (VALID_CONFIG + UNKNOWN_PROTOCOL_CHAT_PROVIDER, "title", "chat"),
        (EXPECTED_DEFAULT_CONFIG, "chat", None),
        (VALID_CONFIG.replace('protocol = "anthropic"', 'protocol = "future"'), "chat", None),
        (
            VALID_CONFIG.replace(
                'base_url = "https://api.anthropic.com"', 'base_url = "not-an-absolute-url"'
            ),
            "chat",
            None,
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"', 'base_url = ""'),
            "chat",
            None,
        ),
        (VALID_CONFIG.replace('api_key = "sk-ant-secret"', 'api_key = ""'), "chat", None),
        (VALID_CONFIG.replace('models = ["claude-model"]', "models = []"), "chat", None),
        (
            VALID_CONFIG.replace(
                'provider_id = "anthropic-default"', 'provider_id = "missing-provider"'
            ),
            "chat",
            None,
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"', 'model = "outside-catalog"'),
            "chat",
            None,
        ),
    ],
)
def test_model_route_resolution_uses_only_a_usable_default(
    agent_home: Path,
    content: str,
    requested_route: str,
    expected_selected_route: str | None,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")
    configuration = loader.load()

    if expected_selected_route is None:
        with pytest.raises(ConfigError) as raised:
            configuration.resolve_route(requested_route)
        assert raised.value.error.code == "route_unavailable"
        assert "sk-ant-secret" not in str(raised.value)
    else:
        resolved = configuration.resolve_route(requested_route)
        assert (
            resolved.requested_route,
            resolved.selected_route,
            resolved.provider.provider_id,
            resolved.route.model,
            resolved.used_fallback,
        ) == (requested_route, "chat", "anthropic-default", "claude-model", True)


def test_missing_default_model_route_names_the_required_configuration_table(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content_without_routes = VALID_CONFIG.partition("\n[models.routes.chat]")[0] + "\n"
    loader.path.write_text(content_without_routes, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        loader.load_for_startup()

    assert raised.value.error.code == "route_unavailable"
    assert raised.value.error.message == (
        "Chat Model Route is missing. Add [models.routes.chat] to User Configuration."
    )


def test_update_reasoning_effort_reopens_latest_document_and_preserves_unrelated_content(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = """# Keep this comment and every unrelated route.
[runtime]
max_tool_result_chars = 60000

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "latest-secret"
models = ["chat-model", "memory-model"]

[models.routes.chat]
provider_id = "primary"
model = "chat-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "low"
timeout = 30

[models.routes.title]
provider_id = "primary"
model = "chat-model"
context_window = 8192
max_output = 1024
temperature = 0.2
reasoning_effort = "high"
timeout = 45

[models.routes.memory]
provider_id = "primary"
model = "memory-model"
context_window = 8192
max_output = 1024
temperature = 0.1
reasoning_effort = "max"
timeout = 60
"""
    loader.path.write_text(content, encoding="utf-8")

    loader.update_reasoning_effort("xhigh")

    updated = loader.path.read_text(encoding="utf-8")
    assert "# Keep this comment and every unrelated route." in updated
    assert 'api_key = "latest-secret"' in updated
    assert 'model = "memory-model"' in updated
    assert updated.count('reasoning_effort = "xhigh"') == 1
    assert 'reasoning_effort = "high"' in updated
    assert 'reasoning_effort = "max"' in updated


def test_update_reasoning_effort_holds_shared_lock_from_latest_read_through_replace(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    latest = MINIMAL_VALID_CONFIG.replace(
        "max_output = 1024",
        "max_output = 2048\n\n# Added by a cooperating writer.",
    )
    lock_held = False
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    @contextmanager
    def acquire_after_user_write(
        lock_path: Path,
        *,
        timeout: float = 1.0,
    ) -> Iterator[None]:
        nonlocal lock_held
        assert lock_path == loader.path.with_name(".config.toml.lock")
        assert timeout == 1.0
        loader.path.write_text(latest, encoding="utf-8")
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    def replace_while_locked(target: Path, content: str) -> None:
        assert lock_held
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "exclusive_lock", acquire_after_user_write)
    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", replace_while_locked)

    loader.update_reasoning_effort("high")

    updated = loader.path.read_text(encoding="utf-8")
    assert "# Added by a cooperating writer." in updated
    assert "max_output = 2048" in updated
    assert 'reasoning_effort = "high"' in updated


def test_update_reasoning_effort_keeps_external_edits_and_inherited_chat_absent(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    loader.load()

    latest_content = MINIMAL_VALID_CONFIG.replace("max_output = 1024", "max_output = 2048").replace(
        "timeout = 30", "timeout = 45\n\n# Added by an external editor."
    )
    loader.path.write_text(latest_content, encoding="utf-8")

    loader.update_reasoning_effort("max")

    updated = loader.path.read_text(encoding="utf-8")
    assert "[models.routes.title]" not in updated
    assert "# Added by an external editor." in updated
    assert 'reasoning_effort = "max"' in updated
    restarted = ConfigLoader(AgentHome(agent_home)).load_for_startup()
    assert restarted.models.routes["chat"].reasoning_effort == "max"
    assert restarted.models.routes["chat"].max_output == 2048


@pytest.mark.parametrize(
    ("raw_content", "write_bytes"),
    [
        ('[models.routes.chat\nreasoning_effort = "low"\n', False),
        (b'[models.routes.chat]\nreasoning_effort = "low"\n\xff', True),
    ],
    ids=("malformed-toml", "invalid-utf8"),
)
def test_update_reasoning_effort_rejects_unreadable_source_before_replacement(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_content: str | bytes,
    write_bytes: bool,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    if write_bytes:
        assert isinstance(raw_content, bytes)
        loader.path.write_bytes(raw_content)
    else:
        assert isinstance(raw_content, str)
        loader.path.write_text(raw_content, encoding="utf-8")
    replacement_calls: list[object] = []

    def record_replacement(*_args: object, **_kwargs: object) -> None:
        replacement_calls.append(None)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_replacement)

    with pytest.raises(ConfigError) as raised:
        loader.update_reasoning_effort("high")

    assert raised.value.error.code == "config_parse_error"
    if write_bytes:
        assert isinstance(raw_content, bytes)
        assert loader.path.read_bytes() == raw_content
    else:
        assert isinstance(raw_content, str)
        assert loader.path.read_text(encoding="utf-8") == raw_content
    assert replacement_calls == []


def test_update_reasoning_effort_rejects_missing_default_before_replacement(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content_without_routes = VALID_CONFIG.partition("\n[models.routes.chat]")[0] + "\n"
    loader.path.write_text(content_without_routes, encoding="utf-8")
    replacement_calls: list[object] = []
    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "atomic_replace_text",
        lambda *_args, **_kwargs: replacement_calls.append(None),
    )

    with pytest.raises(ConfigError) as raised:
        loader.update_reasoning_effort("high")

    assert raised.value.error.code == "route_unavailable"
    assert loader.path.read_text(encoding="utf-8") == content_without_routes
    assert replacement_calls == []


def test_update_reasoning_effort_validates_candidate_before_replacement(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    invalid_content = VALID_CONFIG.replace("max_output = 8192", "max_output = 200000")
    loader.path.write_text(invalid_content, encoding="utf-8")
    replacement_calls: list[object] = []
    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "atomic_replace_text",
        lambda *_args, **_kwargs: replacement_calls.append(None),
    )

    with pytest.raises(ConfigError) as raised:
        loader.update_reasoning_effort("high")

    assert raised.value.error.code == "config_invalid"
    assert loader.path.read_text(encoding="utf-8") == invalid_content
    assert replacement_calls == []


def test_update_reasoning_effort_propagates_atomic_replacement_failure_without_partial_write(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG, encoding="utf-8")
    original = loader.path.read_text(encoding="utf-8")
    replacement_calls: list[object] = []

    def fail_replacement(*_args: object, **_kwargs: object) -> None:
        replacement_calls.append(None)
        raise OSError("injected replacement failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_replacement)

    with pytest.raises(OSError, match="injected replacement failure"):
        loader.update_reasoning_effort("high")

    assert replacement_calls == [None]
    assert loader.path.read_text(encoding="utf-8") == original


def test_update_reasoning_effort_publishes_a_valid_candidate_once(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    replacement_calls: list[tuple[Path, str]] = []
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    def record_replacement(target: Path, content: str) -> None:
        replacement_calls.append((target, content))
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_replacement)

    loader.update_reasoning_effort("mid")

    assert len(replacement_calls) == 1
    target, candidate = replacement_calls[0]
    assert target == loader.path
    assert candidate == loader.path.read_text(encoding="utf-8")
