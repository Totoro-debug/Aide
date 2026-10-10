from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigError, ConfigLoader
from aide.provider.model_router import ModelRouter
from aide.provider.models import AssistantModelMessage, ModelResponse, ModelUsage
from aide.provider.session_configuration import SessionModelConfiguration
from aide.service.runtime import AgentService
from tests.fixtures import ScriptedFakeProvider

MODEL_CONFIG = '''# Preserve this configuration comment.
[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "model-configuration-secret"
[models.providers.primary.models.large]
# Preserve this model comment.
context_window = 65536
max_output = 16384
temperature = 0.1
reasoning_effort = "high"
timeout = 91
[models.providers.primary.models."vendor/small.v1"]
context_window = 4096
max_output = 512
temperature = 0.7
reasoning_effort = "mid"
timeout = 17
[models.routes.chat]
provider_id = "primary"
model = "large"
[models.routes.title]
provider_id = "primary"
model = "large"
[models.routes.memory]
provider_id = "primary"
model = "vendor/small.v1"
[models.routes.schedule]
provider_id = "primary"
model = "vendor/small.v1"
'''


def model_loader(tmp_path: Path, content: str = MODEL_CONFIG) -> ConfigLoader:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    loader = ConfigLoader(home)
    loader.path.write_text(content, encoding="utf-8")
    return loader


def editable_models(loader: ConfigLoader) -> dict[str, Any]:
    models = json.loads(json.dumps(loader.web_snapshot().fields["models"]))
    for provider in models["providers"].values():
        provider.pop("api_key")
    return cast(dict[str, Any], models)


def test_model_defaults_drive_routes_session_selection_and_available_models(tmp_path: Path) -> None:
    loader = model_loader(tmp_path)
    configuration = loader.load_for_startup()
    for purpose in ("memory", "schedule"):
        route = configuration.resolve_route(purpose).route
        assert (route.context_window, route.max_output, route.temperature, route.timeout) == (
            4096, 512, 0.7, 17,
        )
    selected = configuration.resolve_session_model_route("primary", "vendor/small.v1", "xhigh")
    assert selected.route.max_output == 512
    assert selected.route.context_window == 4096
    assert selected.route.temperature == 0.7
    assert selected.route.timeout == 17
    assert selected.route.reasoning_effort == "xhigh"
    view = AgentService(loader.agent_home, configuration).available_models_view()
    assert {cast(Mapping[str, object], model)["model"] for model in cast(list[object], view["models"])} == {
        "large", "vendor/small.v1",
    }
    fields = loader.web_snapshot().fields["models"]
    assert all(set(route) == {"provider_id", "model"} for route in cast(dict[str, Any], fields["routes"]).values())
    assert "model-configuration-secret" not in repr(view) + repr(fields)


@pytest.mark.asyncio
async def test_session_switch_uses_selected_model_request_parameters(tmp_path: Path) -> None:
    loader = model_loader(tmp_path)
    response = ModelResponse(
        message=AssistantModelMessage(content="done"), usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(completions=[response, response])
    router = ModelRouter(configuration=loader.load(), provider_factory=lambda _configuration: provider)
    for selection in (SessionModelConfiguration("primary", "large", "high"),
                      SessionModelConfiguration("primary", "vendor/small.v1", "xhigh")):
        await router.complete("chat", messages=[{"role": "user", "content": "hello"}], tools=[],
                              session_model_configuration=selection)
    first, second = provider.complete_requests
    assert (first.max_output, first.temperature, first.timeout) == (16384, 0.1, 91)
    assert (second.model, second.max_output, second.temperature, second.timeout, second.reasoning_effort) == (
        "vendor/small.v1", 512, 0.7, 17, "xhigh",
    )
    await router.close()


@pytest.mark.asyncio
async def test_memory_uses_model_parameters_while_missing_schedule_route_fails(
    tmp_path: Path,
) -> None:
    loader = model_loader(tmp_path, MODEL_CONFIG.replace(
        '[models.routes.schedule]\nprovider_id = "primary"\nmodel = "vendor/small.v1"\n', "",
    ))
    response = ModelResponse(
        message=AssistantModelMessage(content="done"),
        usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11), finish_reason="stop",
    )
    provider = ScriptedFakeProvider(completions=[response])
    router = ModelRouter(configuration=loader.load(), provider_factory=lambda _configuration: provider)
    await router.complete("memory", messages=[{"role": "user", "content": "hello"}], tools=[])
    with pytest.raises(ConfigError) as raised:
        await router.complete("schedule", messages=[{"role": "user", "content": "hello"}], tools=[])
    assert raised.value.error.code == "route_unavailable"
    assert len(provider.complete_requests) == 1
    memory = provider.complete_requests[0]
    assert (memory.model, memory.max_output, memory.temperature, memory.timeout) == (
        "vendor/small.v1", 512, 0.7, 17,
    )
    await router.close()


@pytest.mark.parametrize("invalid", [
    'context_window = "broken"', 'max_output = 65536', 'temperature = "broken"',
    'unexpected = "model-configuration-secret"',
])
def test_invalid_model_repair_retains_siblings_and_explicit_route_choices(
    tmp_path: Path, invalid: str,
) -> None:
    content = MODEL_CONFIG.replace("context_window = 65536", invalid, 1) if invalid.startswith(
        "context_window"
    ) else MODEL_CONFIG.replace("max_output = 16384", invalid, 1) if invalid.startswith(
        "max_output"
    ) else MODEL_CONFIG.replace("temperature = 0.1", invalid, 1)
    loader = model_loader(tmp_path, content)
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    fields = cast(dict[str, Any], json.loads(json.dumps(dict(snapshot.fields))))
    large = fields["models"]["providers"]["primary"]["models"]["large"]
    small = fields["models"]["providers"]["primary"]["models"]["vendor/small.v1"]
    assert small == {"context_window": 4096, "max_output": 512, "temperature": 0.7,
                     "reasoning_effort": "mid", "timeout": 17}
    assert large["timeout"] == 91
    assert fields["models"]["routes"]["chat"] == {"provider_id": "primary", "model": "large"}
    assert "model-configuration-secret" not in repr(fields)
    large.update(context_window=65536, max_output=16384, temperature=0.1)
    fields["models"]["providers"]["primary"].pop("api_key")
    result = loader.repair_editable_fields(snapshot.revision, fields)
    assert result.configuration.resolve_route("chat").route.timeout == 91
    assert result.configuration.resolve_route("memory").route.max_output == 512
    assert result.configuration.models.providers["primary"].api_key == "model-configuration-secret"


@pytest.mark.parametrize("field,value", [
    ("context_window", 1023), ("context_window", 10000001), ("max_output", 0),
    ("max_output", 65536), ("temperature", -0.1), ("temperature", 2.1),
    ("reasoning_effort", "unknown"), ("timeout", 0), ("timeout", 601),
])
def test_invalid_model_parameters_preserve_bytes(tmp_path: Path, field: str, value: object) -> None:
    loader = model_loader(tmp_path)
    before = loader.path.read_bytes()
    models = editable_models(loader)
    models["providers"]["primary"]["models"]["large"][field] = value
    with pytest.raises(ConfigError) as raised:
        loader.patch_editable_fields(loader.revision(), {"models": models})
    assert f"models.providers.primary.models.large.{field}" in raised.value.field_errors
    assert loader.path.read_bytes() == before


def test_scalar_model_row_is_visible_and_can_be_repaired(tmp_path: Path) -> None:
    start = MODEL_CONFIG.index("[models.providers.primary.models.large]")
    end = MODEL_CONFIG.index('[models.providers.primary.models."vendor/small.v1"]')
    content = MODEL_CONFIG[:start] + '[models.providers.primary.models]\nlarge = "broken"\n' + MODEL_CONFIG[end:]
    loader = model_loader(tmp_path, content)
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    fields = cast(dict[str, Any], json.loads(json.dumps(dict(snapshot.fields))))
    parameters = fields["models"]["providers"]["primary"]["models"]
    assert all(value is None for value in parameters["large"].values())
    parameters["large"] = {"context_window": 65536, "max_output": 16384, "temperature": 0.1,
                           "reasoning_effort": "high", "timeout": 91}
    fields["models"]["providers"]["primary"].pop("api_key")
    loader.repair_editable_fields(snapshot.revision, fields)
    assert loader.load().resolve_route("chat").route.max_output == 16384


@pytest.mark.parametrize("field,value", [
    ("context_window", 65536), ("max_output", 512), ("temperature", 0.5),
    ("reasoning_effort", "mid"), ("timeout", 120),
])
def test_routes_reject_model_parameters(tmp_path: Path, field: str, value: object) -> None:
    loader = model_loader(tmp_path)
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(loader.revision(), {"models": {"routes": {"chat": {field: value}}}})
    assert loader.path.read_bytes() == before


def test_model_save_preserves_comments_and_removes_deleted_models(tmp_path: Path) -> None:
    loader = model_loader(tmp_path)
    models = editable_models(loader)
    models["providers"]["primary"]["models"]["large"]["max_output"] = 1024
    del models["providers"]["primary"]["models"]["vendor/small.v1"]
    for route in models["routes"].values():
        if route["provider_id"]:
            route["model"] = "large"
    loader.patch_editable_fields(loader.revision(), {"models": models})
    assert "# Preserve this model comment." in loader.path.read_text(encoding="utf-8")
    assert tuple(loader.load().models.providers["primary"].models) == ("large",)
    assert loader.load().resolve_route("schedule").route.max_output == 1024


@pytest.mark.parametrize("invalid_models", ['42', '"model"', '["model"]'])
def test_provider_requires_a_model_parameter_table(tmp_path: Path, invalid_models: str) -> None:
    start = MODEL_CONFIG.index("[models.providers.primary.models.large]")
    end = MODEL_CONFIG.index("[models.routes.chat]")
    loader = model_loader(tmp_path, MODEL_CONFIG[:start] + f"models = {invalid_models}\n" + MODEL_CONFIG[end:])
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.load()
    assert loader.web_snapshot().state == "invalid"
    assert loader.path.read_bytes() == before


def test_effort_persistence_updates_models_without_route_parameters(tmp_path: Path) -> None:
    loader = model_loader(tmp_path)
    loader.update_reasoning_effort("max")
    content = tomllib.loads(loader.path.read_text(encoding="utf-8"))
    assert content["models"]["providers"]["primary"]["models"]["large"]["reasoning_effort"] == "max"
    assert content["models"]["providers"]["primary"]["models"]["vendor/small.v1"]["reasoning_effort"] == "mid"
    assert all(set(route) == {"provider_id", "model"} for route in content["models"]["routes"].values())


def test_fixed_route_fields_include_unconfigured_targets_without_writing(tmp_path: Path) -> None:
    loader = model_loader(tmp_path)
    before = loader.path.read_bytes()
    routes = loader.web_snapshot().fields["models"]["routes"]
    assert isinstance(routes, Mapping)
    assert tuple(routes) == ("chat", "title", "memory", "schedule", "subagent")
    assert routes["subagent"] == {"provider_id": "", "model": ""}
    assert loader.path.read_bytes() == before


@pytest.mark.parametrize("stale", [False, True])
def test_partial_route_update_preserves_omitted_targets(tmp_path: Path, stale: bool) -> None:
    loader = model_loader(tmp_path)
    snapshot = loader.web_snapshot()
    if stale:
        loader.patch_editable_fields(snapshot.revision, {
            "models": {"routes": {"title": {"model": "vendor/small.v1"}}},
        })
    loader.patch_editable_fields(snapshot.revision, {
        "models": {"routes": {"chat": {"model": "vendor/small.v1"}}},
    }, baseline=snapshot.fields)
    configuration = loader.load_for_startup()
    assert configuration.models.routes["chat"].model == "vendor/small.v1"
    assert configuration.models.routes["title"].model == ("vendor/small.v1" if stale else "large")
    assert configuration.models.routes["memory"].model == "vendor/small.v1"
    assert configuration.models.routes["schedule"].model == "vendor/small.v1"


@pytest.mark.parametrize("name", ["title", "memory", "schedule", "subagent"])
def test_optional_target_can_be_unconfigured_while_its_row_remains(
    tmp_path: Path, name: str,
) -> None:
    loader = model_loader(tmp_path)
    result = loader.patch_editable_fields(loader.revision(), {
        "models": {"routes": {name: {"provider_id": "", "model": ""}}},
    })
    routes = cast(Mapping[str, object], result.fields["models"]["routes"])
    assert routes[name] == {"provider_id": "", "model": ""}
    assert name not in result.configuration.models.routes
    if name in {"title", "memory"}:
        assert result.configuration.resolve_route(name).selected_route == "chat"
    else:
        with pytest.raises(ConfigError) as raised:
            result.configuration.resolve_route(name)
        assert raised.value.error.code == "route_unavailable"


@pytest.mark.parametrize("routes", [
    {"chat": {"provider_id": "", "model": ""}},
    {"unknown": {"provider_id": "primary", "model": "large"}},
    {"title": None},
    {"title": {"delete": True}},
    {"title": {"provider_id": "", "model": "large"}},
    {"title": {"provider_id": "primary", "model": ""}},
])
def test_invalid_route_updates_preserve_configuration(
    tmp_path: Path, routes: dict[str, object],
) -> None:
    loader = model_loader(tmp_path)
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(loader.revision(), {"models": {"routes": routes}})
    assert loader.path.read_bytes() == before
