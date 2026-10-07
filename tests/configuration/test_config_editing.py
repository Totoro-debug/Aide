from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigError, ConfigLoader, ConfigRevisionConflict
from tests.configuration.test_config import MINIMAL_VALID_CONFIG

FULL_EDITABLE_CONFIG = """# preserve this comment
[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "provider-secret-canary-302"
models = ["small-model", "large-model"]

[models.providers.retired]
protocol = "anthropic"
base_url = "https://anthropic.example"
api_key = "retired-secret-canary-302"
models = ["retired-model"]

[models.routes.default]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[models.routes.chat]
provider_id = "primary"
model = "large-model"
context_window = 16384
max_output = 2048
temperature = 0.2
reasoning_effort = "high"
timeout = 45

[models.routes.memory]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "low"
timeout = 30

[models.routes.schedule]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example/tools"
headers = { Authorization = "mcp-header-canary-302" }
connect_timeout = 30
call_timeout = 60

[mcp.servers.http.tool_keywords]
search = ["query"]

[mcp.servers.stdio]
enabled = false
transport = "stdio"
command = "python"
args = ["server.py"]
cwd = "."
connect_timeout = 30
call_timeout = 60
"""


def _loader(tmp_path: Path, content: str = MINIMAL_VALID_CONFIG) -> ConfigLoader:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(content, encoding="utf-8")
    return ConfigLoader(home)


def test_web_snapshot_exposes_all_safe_fields_without_secrets(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    snapshot = loader.web_snapshot()

    assert snapshot.revision.startswith("sha256:")
    assert snapshot.fields["runtime"] == {
        "max_tool_result_chars": 4096,
        "max_iterations": 50,
        "enable_skill_always_load": False,
        "enable_tool_micro_compression": False,
        "compact_ratio": 0.9,
        "permission_level": "workspace-write",
        "exec_shell": "auto",
    }
    assert snapshot.fields["memory"] == {"batch_size": 10, "schedule": "0 * * * *"}
    models = snapshot.fields["models"]
    providers = cast(Mapping[str, object], models["providers"])
    primary = cast(Mapping[str, object], providers["primary"])
    assert primary["api_key"] == {"configured": True}
    assert "minimal-secret" not in str(snapshot.fields)


@pytest.mark.parametrize("enabled", (True, False))
def test_tool_micro_compression_is_editable_and_persisted(tmp_path: Path, enabled: bool) -> None:
    loader = _loader(tmp_path)
    loader.patch_editable_fields(
        loader.web_snapshot().revision,
        {"runtime": {"enable_tool_micro_compression": not enabled}},
    )

    result = loader.patch_editable_fields(
        loader.web_snapshot().revision,
        {"runtime": {"enable_tool_micro_compression": enabled}},
    )

    assert result.configuration.runtime.enable_tool_micro_compression is enabled
    assert ConfigLoader(loader.agent_home).web_snapshot().fields["runtime"][
        "enable_tool_micro_compression"
    ] is enabled


@pytest.mark.parametrize("value", ("true", 1, None))
def test_invalid_tool_micro_compression_patch_preserves_file(tmp_path: Path, value: object) -> None:
    loader = _loader(tmp_path)
    before = loader.path.read_bytes()

    with pytest.raises(ConfigError):
        loader.patch_editable_fields(
            loader.web_snapshot().revision,
            {"runtime": {"enable_tool_micro_compression": value}},
        )

    assert loader.path.read_bytes() == before


def test_default_chat_workspace_is_editable_and_persisted(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    snapshot = loader.web_snapshot()

    assert snapshot.fields["web"]["default_chat_workspace"] == "~/.omni/chat"
    target = tmp_path / "conversations"
    result = loader.patch_editable_fields(
        snapshot.revision,
        {"web": {"default_chat_workspace": str(target)}},
    )

    assert result.configuration.web.default_chat_workspace == str(target)
    assert ConfigLoader(loader.agent_home).web_snapshot().fields["web"][
        "default_chat_workspace"
    ] == str(target)


def test_valid_patch_preserves_comments_and_unknown_toml(tmp_path: Path) -> None:
    content = "# keep this comment\n" + MINIMAL_VALID_CONFIG + "\n[future]\nvalue = 7\n"
    loader = _loader(tmp_path, content)
    before = loader.path.read_bytes()

    result = loader.patch_editable_fields(
        loader.web_snapshot().revision,
        {"runtime": {"max_iterations": 77}, "memory": {"batch_size": 21}},
    )

    after = loader.path.read_text(encoding="utf-8")
    assert result.configuration.runtime.max_iterations == 77
    assert result.configuration.memory.batch_size == 21
    assert result.revision != loader.revision_from_bytes(before)
    assert "# keep this comment" in after
    assert "[future]" in after
    assert "value = 7" in after
    assert "minimal-secret" in after


def test_invalid_patch_does_not_change_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    before = loader.path.read_bytes()
    revision = loader.web_snapshot().revision

    with pytest.raises(ConfigError):
        loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 1}})

    assert loader.path.read_bytes() == before


def test_stale_patch_does_not_change_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    revision = loader.web_snapshot().revision
    loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 60}})
    before = loader.path.read_bytes()

    with pytest.raises(ConfigRevisionConflict):
        loader.patch_editable_fields(revision, {"runtime": {"max_iterations": 61}})

    assert loader.path.read_bytes() == before


def test_two_writers_with_one_revision_only_one_succeeds(tmp_path: Path) -> None:
    first = _loader(tmp_path)
    second = ConfigLoader(first.agent_home)
    revision = first.web_snapshot().revision

    first.patch_editable_fields(revision, {"memory": {"batch_size": 11}})

    with pytest.raises(ConfigRevisionConflict):
        second.patch_editable_fields(revision, {"memory": {"batch_size": 12}})

    assert first.load().memory.batch_size == 11


def test_web_snapshot_projects_all_model_route_and_mcp_fields_without_secrets(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)

    fields = loader.web_snapshot().fields
    models = fields["models"]
    providers = cast(Mapping[str, object], models["providers"])
    routes = cast(Mapping[str, object], models["routes"])
    mcp = fields["mcp"]

    assert providers["primary"] == {
        "protocol": "openai-compatible",
        "base_url": "https://models.example/v1",
        "models": ("small-model", "large-model"),
        "model_context_windows": {"small-model": 8192, "large-model": 16384},
        "api_key": {"configured": True},
    }
    assert set(routes) == {"default", "chat", "memory", "schedule"}
    chat_route = cast(Mapping[str, object], routes["chat"])
    assert chat_route["reasoning_effort"] == "high"
    assert mcp["http"] == {
        "enabled": True,
        "transport": "streamable-http",
        "command": None,
        "args": (),
        "cwd": None,
        "url": "https://mcp.example/tools",
        "headers": {"Authorization": {"configured": True}},
        "connect_timeout": 30,
        "call_timeout": 60,
        "tool_keywords": {"search": ("query",)},
    }
    assert "provider-secret-canary-302" not in repr(fields)
    assert "mcp-header-canary-302" not in repr(fields)


def test_model_context_windows_are_provider_scoped_and_preserve_comments(tmp_path: Path) -> None:
    content = FULL_EDITABLE_CONFIG + '''
[models.providers.primary.model_context_windows]
# Keep the comment with this provider/model capacity.
"large-model" = 16384

[models.providers.secondary]
protocol = "openai-compatible"
base_url = "https://secondary.example/v1"
api_key = "secondary-secret-302"
models = ["large-model"]

[models.providers.secondary.model_context_windows]
"large-model" = 65536
'''
    loader = _loader(tmp_path, content)
    before = loader.web_snapshot()
    primary = cast(
        Mapping[str, object], before.fields["models"]["providers"]
    )["primary"]
    assert cast(Mapping[str, object], primary)["model_context_windows"] == {
        "small-model": 8192,
        "large-model": 16384,
    }

    result = loader.patch_editable_fields(
        before.revision,
        {
            "models": {
                "providers": {
                    "primary": {
                        "model_context_windows": {
                            "small-model": 8192,
                            "large-model": 32768,
                        }
                    },
                    "secondary": {
                        "model_context_windows": {"large-model": 65536}
                    },
                }
            }
        },
    )

    assert result.configuration.models.providers["primary"].model_context_windows == {
        "small-model": 8192,
        "large-model": 32768,
    }
    assert result.configuration.models.providers["secondary"].model_context_windows == {
        "large-model": 65536,
    }
    assert result.configuration.resolve_route("chat").route.context_window == 32768
    saved = loader.path.read_text(encoding="utf-8")
    assert "# Keep the comment with this provider/model capacity." in saved
    assert "secondary-secret-302" in saved
    assert "provider-secret-canary-302" not in repr(result.fields)


def test_larger_model_context_window_allows_output_above_legacy_route_capacity(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    snapshot = loader.web_snapshot()
    result = loader.patch_editable_fields(
        snapshot.revision,
        {
            "models": {
                "providers": {"primary": {"model_context_windows": {"small-model": 65536}}},
                "routes": {"default": {"context_window": 8192, "max_output": 16384}},
            }
        },
    )

    resolved = result.configuration.resolve_route("default").route
    assert resolved.context_window == 65536
    assert resolved.max_output == 16384
    restarted = loader.load_for_startup().resolve_route("default").route
    assert restarted == resolved
    projected = cast(Mapping[str, object], result.fields["models"]["routes"])["default"]
    assert cast(Mapping[str, object], projected)["max_output"] == 16384


def test_model_context_window_not_greater_than_route_output_keeps_original_bytes(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()

    with pytest.raises(ConfigError) as error:
        loader.patch_editable_fields(
            loader.web_snapshot().revision,
            {
                "models": {
                    "providers": {
                        "primary": {
                            "model_context_windows": {"small-model": 1024}
                        }
                    }
                }
            },
        )

    assert "models.routes.default.max_output" in error.value.field_errors
    assert loader.path.read_bytes() == before


def test_model_route_mcp_patch_replaces_collections_and_secrets_atomically(tmp_path: Path) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG + "\n[future]\nvalue = 7\n")
    revision = loader.web_snapshot().revision

    result = loader.patch_editable_fields(
        revision,
        {
            "models": {
                "providers": {
                    "primary": {
                        "protocol": "openai-compatible",
                        "base_url": "https://new-models.example/v2",
                        "models": ["new-model"],
                    },
                    "added": {
                        "protocol": "anthropic",
                        "base_url": "https://new-anthropic.example",
                        "models": ["added-model"],
                    },
                },
                "routes": {
                    "default": {
                        "provider_id": "primary",
                        "model": "new-model",
                        "context_window": 16384,
                        "max_output": 2048,
                        "temperature": 0.1,
                        "reasoning_effort": "high",
                        "timeout": 60,
                    },
                    "chat": {
                        "provider_id": "added",
                        "model": "added-model",
                        "context_window": 16384,
                        "max_output": 2048,
                        "temperature": 0.2,
                        "reasoning_effort": "mid",
                        "timeout": 60,
                    },
                    "memory": {
                        "provider_id": "primary",
                        "model": "new-model",
                        "context_window": 16384,
                        "max_output": 2048,
                        "temperature": 0,
                        "reasoning_effort": "low",
                        "timeout": 60,
                    },
                    "schedule": {
                        "provider_id": "added",
                        "model": "added-model",
                        "context_window": 16384,
                        "max_output": 2048,
                        "temperature": 0.2,
                        "reasoning_effort": "high",
                        "timeout": 60,
                    },
                },
            },
            "mcp": {
                "http": {
                    "enabled": False,
                    "transport": "streamable-http",
                    "url": "https://mcp.example/replaced",
                    "headers": {"Authorization": {"configured": True}},
                    "connect_timeout": 40,
                    "call_timeout": 90,
                    "tool_keywords": {"search": ["lookup"]},
                },
            },
        },
        {
            "models.providers.primary.api_key": {
                "action": "replace",
                "value": "provider-secret-replaced-302",
            },
            "models.providers.added.api_key": {
                "action": "replace",
                "value": "provider-secret-added-302",
            },
            "mcp.http.headers.Authorization": {
                "action": "replace",
                "value": "mcp-header-replaced-302",
            },
        },
    )

    configuration = loader.load()
    assert set(configuration.models.providers) == {"primary", "added"}
    assert set(configuration.models.routes) == {"default", "chat", "memory", "schedule"}
    assert configuration.models.routes["chat"].provider_id == "added"
    assert configuration.models.providers["primary"].api_key == "provider-secret-replaced-302"
    assert configuration.models.providers["added"].api_key == "provider-secret-added-302"
    assert configuration.mcp["http"].headers == {"Authorization": "mcp-header-replaced-302"}
    assert configuration.mcp["http"].url == "https://mcp.example/replaced"
    saved = loader.path.read_text(encoding="utf-8")
    assert "# preserve this comment" in saved
    assert "[future]" in saved and "value = 7" in saved
    assert "retired-secret-canary-302" not in saved
    result_models = result.fields["models"]
    result_providers = cast(Mapping[str, object], result_models["providers"])
    result_primary = cast(Mapping[str, object], result_providers["primary"])
    assert result_primary["api_key"] == {"configured": True}


def test_stdio_mcp_fields_patch_preserves_http_secrets(tmp_path: Path) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)

    loader.patch_editable_fields(
        loader.web_snapshot().revision,
        {
            "mcp": {
                "http": {},
                "stdio": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "node",
                    "args": ["server.mjs"],
                    "cwd": "workspace",
                    "connect_timeout": 45,
                    "call_timeout": 75,
                    "tool_keywords": {"launch": ["run"]},
                },
            }
        },
    )

    configuration = loader.load()
    assert configuration.mcp["stdio"].command == "node"
    assert configuration.mcp["stdio"].args == ("server.mjs",)
    assert configuration.mcp["stdio"].cwd == Path("workspace")
    assert configuration.mcp["stdio"].tool_keywords == {"launch": ("run",)}
    assert configuration.mcp["http"].headers == {"Authorization": "mcp-header-canary-302"}


def test_secret_clear_removes_provider_key_and_mcp_header_without_echoing_value(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    revision = loader.web_snapshot().revision

    loader.patch_editable_fields(
        revision,
        {},
        {
            "models.providers.retired.api_key": {"action": "clear"},
            "mcp.http.headers.Authorization": {"action": "clear"},
        },
    )

    configuration = loader.load()
    assert configuration.models.providers["retired"].api_key == ""
    assert configuration.mcp["http"].headers == {}
    assert "provider-secret-canary-302" not in repr(loader.web_snapshot().fields)
    assert "mcp-header-canary-302" not in repr(loader.web_snapshot().fields)


def test_dangling_route_candidate_keeps_original_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()

    with pytest.raises(ConfigError):
        loader.patch_editable_fields(
            loader.web_snapshot().revision,
            {
                "models": {
                    "providers": {
                        "primary": {
                            "protocol": "openai-compatible",
                            "base_url": "https://models.example/v1",
                            "models": ["small-model", "large-model"],
                        },
                    },
                    "routes": {
                        "default": {
                            "provider_id": "missing",
                            "model": "small-model",
                            "context_window": 8192,
                            "max_output": 1024,
                            "temperature": 0,
                            "reasoning_effort": "mid",
                            "timeout": 30,
                        },
                    },
                },
            },
        )

    assert loader.path.read_bytes() == before


@pytest.mark.parametrize("cwd", [None, "."])
def test_stdio_optional_cwd_and_exact_arguments(tmp_path: Path, cwd: str | None) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    arguments = ["--flag", "--flag", "a,b", " x ", "", "line\nvalue"]
    result = loader.patch_editable_fields(
        loader.revision(), {"mcp": {"http": {}, "stdio": {"args": arguments, "cwd": cwd}}}
    )
    assert result.configuration.mcp["stdio"].args == tuple(arguments)
    assert result.configuration.mcp["stdio"].cwd == (None if cwd is None else Path(cwd))
    loader.patch_editable_fields(loader.revision(), {"mcp": {"http": {}, "stdio": {"cwd": None}}})
    assert loader.load().mcp["stdio"].cwd is None


@pytest.mark.parametrize("value", ["unsupported", ""])
def test_nondefault_route_cannot_fallback_from_invalid_provider(tmp_path: Path, value: str) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(
            loader.revision(),
            {
                "models": {
                    "providers": {"primary": {}, "retired": {"protocol": value}},
                    "routes": {
                        "default": {},
                        "chat": {"provider_id": "retired", "model": "retired-model"},
                        "memory": {},
                        "schedule": {},
                    },
                }
            },
        )
    assert loader.path.read_bytes() == before


def test_clearing_key_of_referenced_nondefault_provider_is_invalid(tmp_path: Path) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(
            loader.revision(),
            {
                "models": {
                    "routes": {
                        "default": {},
                        "chat": {"provider_id": "retired", "model": "retired-model"},
                        "memory": {},
                        "schedule": {},
                    }
                }
            },
            {"models.providers.retired.api_key": {"action": "clear"}},
        )
    assert loader.path.read_bytes() == before


@pytest.mark.parametrize(
    "name, patch",
    [
        ("stdio", {"url": "https://example.com/mcp"}),
        ("stdio", {"headers": {"X-Test": {"configured": False}}}),
        ("http", {"command": "python"}),
        ("http", {"args": ["server.py"]}),
        ("http", {"cwd": "."}),
    ],
)
def test_partial_mcp_patch_cannot_ignore_incompatible_fields(
    tmp_path: Path, name: str, patch: dict[str, object]
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()
    servers: dict[str, object] = {"http": {}, "stdio": {}}
    servers[name] = patch
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(loader.revision(), {"mcp": servers})
    assert loader.path.read_bytes() == before


@pytest.mark.parametrize(
    "section, name, field, value",
    [
        ("providers", "retired", "protocol", "invalid"),
        ("providers", "retired", "protocol", True),
        ("providers", "retired", "base_url", "https://example.com:99999"),
        ("providers", "retired", "models", ["same", "same"]),
        ("providers", "retired", "models", [False]),
        ("routes", "chat", "provider_id", "missing"),
        ("routes", "chat", "model", "missing"),
        ("routes", "chat", "context_window", 1023),
        ("routes", "chat", "context_window", True),
        ("routes", "chat", "max_output", 16384),
        ("routes", "chat", "max_output", 0),
        ("routes", "chat", "temperature", 2.1),
        ("routes", "chat", "temperature", float("nan")),
        ("routes", "chat", "reasoning_effort", "invalid"),
        ("routes", "chat", "reasoning_effort", "medium"),
        ("routes", "chat", "timeout", 601),
        ("routes", "chat", "timeout", False),
        ("mcp", "http", "enabled", 1),
        ("mcp", "http", "transport", "invalid"),
        ("mcp", "http", "url", "https://example.com:invalid"),
        ("mcp", "http", "headers", {"X-Key": {"configured": 1}}),
        ("mcp", "http", "connect_timeout", 0),
        ("mcp", "http", "call_timeout", True),
        ("mcp", "http", "tool_keywords", {"echo": ["中文"]}),
        ("mcp", "stdio", "command", ""),
        ("mcp", "stdio", "args", [False]),
        ("mcp", "stdio", "cwd", " "),
        ("mcp", "stdio", "env", {"TOKEN": "env-invalid-canary-302"}),
        ("mcp", "stdio", "secret_env", {"TOKEN": "secret-env-invalid-canary-302"}),
    ],
)
def test_complete_settings_value_matrix_preserves_invalid_bytes(
    tmp_path: Path, section: str, name: str, field: str, value: object
) -> None:
    loader = _loader(tmp_path, FULL_EDITABLE_CONFIG)
    before = loader.path.read_bytes()
    if section == "mcp":
        values: dict[str, object] = {"http": {}, "stdio": {}}
        values[name] = {field: value}
        patch: dict[str, object] = {"mcp": values}
    else:
        values = (
            {"primary": {}, "retired": {}}
            if section == "providers"
            else {"default": {}, "chat": {}, "memory": {}, "schedule": {}}
        )
        values[name] = {field: value}
        patch = {"models": {section: values}}
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(loader.revision(), patch)
    assert loader.path.read_bytes() == before


@pytest.mark.parametrize("transport", ["[]", "{}", "false"])
def test_partial_edit_rejects_invalid_existing_transport_without_type_error(
    tmp_path: Path, transport: str
) -> None:
    loader = _loader(
        tmp_path, FULL_EDITABLE_CONFIG.replace('transport = "stdio"', f"transport = {transport}")
    )
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(
            loader.revision(), {"mcp": {"http": {}, "stdio": {"enabled": False}}}
        )
    assert loader.path.read_bytes() == before
