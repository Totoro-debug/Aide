from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import (
    ConfigError,
    ConfigLoader,
    ConfigurationDiagnostic,
    MCPServerConfiguration,
)
from aide.management.commands import ManagementCommandDispatcher
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.management.factories import management_service


def _loader_with_mcp(agent_home: Path, mcp_content: str) -> ConfigLoader:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG + mcp_content, encoding="utf-8")
    return loader


def test_valid_stdio_server_is_loaded_and_resolves_workspace_relative_cwd(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.filesystem]
enabled = true
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "."]
cwd = "servers"
connect_timeout = 15
call_timeout = 45
""",
        encoding="utf-8",
    )

    configuration = loader.load()
    server = configuration.mcp["filesystem"]

    assert (
        server.mcp_name,
        server.enabled,
        server.transport,
        server.command,
        server.args,
        server.cwd,
        server.connect_timeout,
        server.call_timeout,
    ) == (
        "filesystem",
        True,
        "stdio",
        "uvx",
        ("mcp-server-filesystem", "."),
        Path("servers"),
        15,
        45,
    )
    assert server.resolve_cwd(agent_home) == agent_home / "servers"


def test_mcp_tool_keywords_are_trimmed_deduplicated_and_immutable(agent_home: Path) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.github.tool_keywords]
search_issues = [" issue ", "", "issue", "github"]
""",
    )

    server = loader.load().mcp["github"]

    assert server.tool_keywords == {"search_issues": ("issue", "github")}
    with pytest.raises(TypeError):
        cast(Any, server.tool_keywords)["other"] = ("keyword",)

    direct = MCPServerConfiguration(
        mcp_name="direct",
        enabled=True,
        transport="stdio",
        command="server",
        tool_keywords={"search_issues": (" issue ", "", "issue", "github")},
    )
    assert direct.tool_keywords == {"search_issues": ("issue", "github")}


@pytest.mark.parametrize(
    ("keyword_literal", "unsafe_value"),
    [
        ('"搜索"', "搜索"),
        ('"123"', "123"),
        (r'"search\u0001"', "search\x01"),
    ],
    ids=("non-ascii", "digits-only", "control-character"),
)
def test_non_english_mcp_tool_keywords_are_isolated_with_one_safe_diagnostic(
    agent_home: Path,
    keyword_literal: str,
    unsafe_value: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.invalid.tool_keywords]
search = [{keyword_literal}]

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.valid.tool_keywords]
search = [" issues ", "issues", "github"]
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert configuration.mcp["valid"].tool_keywords == {"search": ("issues", "github")}
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, ConfigurationDiagnostic)
    assert diagnostic.mcp_name == "invalid"
    assert "tool_keywords.search" in diagnostic.message
    assert unsafe_value not in diagnostic.message


@pytest.mark.parametrize(
    "invalid_keywords",
    [
        'tool_keywords = "search"',
        "[mcp.servers.invalid.tool_keywords]\nsearch = [1]",
    ],
    ids=("not-table", "non-string-item"),
)
def test_invalid_mcp_tool_keywords_are_isolated_per_server(
    agent_home: Path,
    invalid_keywords: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "server"
{invalid_keywords}

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "server"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert "tool_keywords" in loader.diagnostics[0].message


def test_fill_mcp_tool_keywords_preserves_latest_content_and_replaces_once(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    original = """# Keep this comment.
[runtime]
max_tool_result_chars = 4096

[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
# Keep this server comment.
[mcp.servers.github.tool_keywords]
existing = ["configured"]
search_issues = []

"""
    loader.path.write_text(original, encoding="utf-8")
    replacement_calls: list[tuple[Path, str]] = []
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    def record_replacement(target: Path, content: str) -> None:
        replacement_calls.append((target, content))
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_replacement)

    effective = loader.fill_mcp_tool_keywords(
        {
            ("github", "existing"): ("generated",),
            ("github", "search_issues"): (" issues ", "", "issues", "search"),
            ("removed", "missing"): ("must-not-recreate",),
        }
    )

    saved = loader.path.read_text(encoding="utf-8")
    assert effective == {
        ("github", "existing"): ("configured",),
        ("github", "search_issues"): ("issues", "search"),
    }
    assert replacement_calls == [(loader.path, saved)]
    assert "# Keep this comment." in saved
    assert "# Keep this server comment." in saved
    assert 'existing = ["configured"]' in saved
    assert "must-not-recreate" not in saved
    with pytest.raises(TypeError):
        cast(Any, effective)[("github", "other")] = ("keyword",)


def test_fill_mcp_tool_keywords_holds_shared_lock_from_latest_read_through_replace(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.github.tool_keywords]
search_issues = []
list_issues = []
""",
    )
    latest = loader.path.read_text(encoding="utf-8").replace(
        "search_issues = []",
        'search_issues = ["configured"]\n# Added by a cooperating writer.',
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

    effective = loader.fill_mcp_tool_keywords(
        {
            ("github", "search_issues"): ("generated-search",),
            ("github", "list_issues"): ("generated-list",),
        }
    )

    saved = loader.path.read_text(encoding="utf-8")
    assert effective == {
        ("github", "search_issues"): ("configured",),
        ("github", "list_issues"): ("generated-list",),
    }
    assert "# Added by a cooperating writer." in saved
    assert "generated-search" not in saved
    assert 'list_issues = ["generated-list"]' in saved


def test_fill_mcp_tool_keywords_does_not_restore_server_deleted_before_locked_read(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
    )
    replacement_calls: list[object] = []

    @contextmanager
    def acquire_after_server_deletion(
        lock_path: Path,
        *,
        timeout: float = 1.0,
    ) -> Iterator[None]:
        assert lock_path == loader.path.with_name(".config.toml.lock")
        assert timeout == 1.0
        loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
        yield

    monkeypatch.setattr(HOST_FILESYSTEM, "exclusive_lock", acquire_after_server_deletion)
    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "atomic_replace_text",
        lambda *_args, **_kwargs: replacement_calls.append(None),
    )

    effective = loader.fill_mcp_tool_keywords({("github", "search_issues"): ("generated",)})

    assert effective == {}
    assert "mcp.servers.github" not in loader.path.read_text(encoding="utf-8")
    assert replacement_calls == []


def test_fill_mcp_tool_keywords_does_not_replace_when_values_are_already_set(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.github.tool_keywords]
search_issues = [" configured ", "configured"]
""",
    )
    original = loader.path.read_text(encoding="utf-8")
    replacement_calls: list[object] = []
    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "atomic_replace_text",
        lambda *_args, **_kwargs: replacement_calls.append(None),
    )

    effective = loader.fill_mcp_tool_keywords(
        {
            ("github", "search_issues"): ("generated",),
            ("removed", "missing"): ("must-not-recreate",),
        }
    )

    assert effective == {("github", "search_issues"): ("configured",)}
    assert loader.path.read_text(encoding="utf-8") == original
    assert replacement_calls == []


def test_fill_mcp_tool_keywords_validates_candidate_before_replacement(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    invalid_content = (
        MINIMAL_VALID_CONFIG.replace(
            "max_output = 1024",
            "max_output = 8192",
        )
        + """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
"""
    )
    loader.path.write_text(invalid_content, encoding="utf-8")
    replacement_calls: list[object] = []
    monkeypatch.setattr(
        HOST_FILESYSTEM,
        "atomic_replace_text",
        lambda *_args, **_kwargs: replacement_calls.append(None),
    )

    with pytest.raises(ConfigError) as raised:
        loader.fill_mcp_tool_keywords({("github", "search_issues"): ("issues",)})

    assert raised.value.error.code == "config_invalid"
    assert loader.path.read_text(encoding="utf-8") == invalid_content
    assert replacement_calls == []


def test_valid_streamable_http_server_preserves_headers_and_disabled_state(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.search]
enabled = false
transport = "streamable-http"
url = "https://mcp.example.test/service"
headers = { Authorization = "Bearer header-secret", X-Trace = "trace-value" }
""",
        encoding="utf-8",
    )

    server = loader.load().mcp["search"]

    assert (
        server.enabled,
        server.transport,
        server.url,
        dict(server.headers),
        server.connect_timeout,
        server.call_timeout,
    ) == (
        False,
        "streamable-http",
        "https://mcp.example.test/service",
        {"Authorization": "Bearer header-secret", "X-Trace": "trace-value"},
        30,
        60,
    )
    assert server.command is None
    assert server.resolve_cwd(agent_home) == agent_home


def test_omitted_stdio_cwd_resolves_to_the_active_workspace(agent_home: Path) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.local]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    server = loader.load().mcp["local"]

    assert server.cwd is None
    assert server.resolve_cwd(agent_home) == agent_home


def test_invalid_server_is_omitted_without_blocking_other_servers(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
env = { API_TOKEN = "must-not-be-reported" }

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
        encoding="utf-8",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    diagnostic = loader.diagnostics[0]
    assert isinstance(diagnostic, ConfigurationDiagnostic)
    assert diagnostic.mcp_name == "invalid"
    assert "env" in diagnostic.message
    assert "must-not-be-reported" not in diagnostic.message


def test_config_view_reports_ignored_servers_and_redacts_http_header_values(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    header_values = (
        "Bearer authorization-secret",
        "session-cookie-secret",
        "api-key-header-secret",
        "custom-header-secret",
        "agent-header-secret",
    )
    loader.path.write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "."]
cwd = "workspace-root"
secret_env = { API_TOKEN = "must-not-be-reported" }

[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example.test/service"

[mcp.servers.http.headers]
Authorization = "Bearer authorization-secret"
Cookie = "session-cookie-secret"
X-Api-Key = "api-key-header-secret"
X-Custom = "custom-header-secret"
User-Agent = "agent-header-secret"
""",
        encoding="utf-8",
    )

    view = loader.view()

    assert view.error is None
    assert len(view.diagnostics) == 1
    diagnostic = view.diagnostics[0]
    assert isinstance(diagnostic, ConfigurationDiagnostic)
    assert diagnostic.mcp_name == "invalid"
    assert "MCP Server 'invalid' ignored" in view.diagnostics_text()
    assert all(secret not in view.redacted_content for secret in header_values)
    assert "Authorization" in view.redacted_content
    assert view.redacted_content.count('"***REDACTED***"') >= len(header_values)
    assert 'command = "uvx"' in view.redacted_content
    assert 'args = ["mcp-server-filesystem", "."]' in view.redacted_content
    assert 'cwd = "workspace-root"' in view.redacted_content
    assert 'url = "https://mcp.example.test/service"' in view.redacted_content


def test_default_configuration_documents_disabled_mcp_examples(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True
    content = loader.path.read_text(encoding="utf-8")
    configuration = loader.load()

    assert "# [mcp.servers.filesystem]" in content
    assert '# transport = "stdio"' in content
    assert "# [mcp.servers.search]" in content
    assert '# transport = "streamable-http"' in content
    assert configuration.mcp == {}

    lines = content.splitlines()

    def uncomment_example(table_name: str) -> str:
        start = lines.index(f"# [{table_name}]")
        end = next(
            index for index in range(start, len(lines)) if index > start and not lines[index]
        )
        return "\n".join(line.removeprefix("# ") for line in lines[start:end]) + "\n"

    loader.path.write_text(
        MINIMAL_VALID_CONFIG
        + uncomment_example("mcp.servers.filesystem")
        + uncomment_example("mcp.servers.search"),
        encoding="utf-8",
    )
    examples = loader.load().mcp

    assert set(examples) == {"filesystem", "search"}
    assert (examples["search"].connect_timeout, examples["search"].call_timeout) == (30, 60)


@pytest.mark.asyncio
async def test_config_management_command_renders_diagnostics_without_header_values(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
env = { API_TOKEN = "command-env-secret" }

[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example.test/service"
headers = { Authorization = "Bearer command-header-secret" }
""",
        encoding="utf-8",
    )

    result = await ManagementCommandDispatcher(management_service(home)).dispatch("/config")

    assert result.handled is True
    assert result.output is not None
    assert "MCP Server 'invalid' ignored" in result.output
    assert "command-env-secret" not in result.output
    assert "command-header-secret" not in result.output


@pytest.mark.parametrize(
    ("invalid_fields", "expected_field"),
    [
        ('env = { API_TOKEN = "one" }', "env"),
        ('env = "three"', "env"),
        ('env = ["four"]', "env"),
        ('secret_env = { API_TOKEN = "two" }', "secret_env"),
        ('secret_env = "five"', "secret_env"),
        ('secret_env = ["six"]', "secret_env"),
    ],
)
def test_forbidden_environment_fields_are_isolated_per_server(
    agent_home: Path,
    invalid_fields: str,
    expected_field: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
{invalid_fields}

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert expected_field in loader.diagnostics[0].message


@pytest.mark.parametrize(
    ("unknown_field", "unknown_value"),
    [
        ("mystery", "true"),
        ("extra", '"value"'),
        ("nested", "{ value = true }"),
    ],
)
def test_unknown_mcp_fields_are_ignored_per_server(
    agent_home: Path,
    unknown_field: str,
    unknown_value: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
{unknown_field} = {unknown_value}

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load_for_startup()

    assert set(configuration.mcp) == {"invalid", "valid"}
    assert loader.diagnostics == ()

    view = loader.view()

    assert view.error is None
    assert view.diagnostics == ()
    assert f"{unknown_field} =" in view.redacted_content


@pytest.mark.parametrize(
    "mcp_name",
    [
        "UpperCase",
        "-leading",
        "a" * 65,
    ],
)
def test_invalid_mcp_names_are_isolated_per_server(agent_home: Path, mcp_name: str) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers."{mcp_name}"]
enabled = true
transport = "stdio"
command = "uvx"

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert mcp_name in loader.diagnostics[0].message


@pytest.mark.parametrize(
    ("server_content", "expected_field"),
    [
        (
            'transport = "stdio"\ncommand = "uvx"\nurl = "https://example.test/mcp"',
            "url",
        ),
        (
            'transport = "stdio"\ncommand = "uvx"\nheaders = { Authorization = "secret" }',
            "headers",
        ),
        (
            'transport = "streamable-http"\ncommand = "uvx"\nurl = "https://example.test/mcp"',
            "command",
        ),
    ],
)
def test_transport_incompatible_fields_are_isolated_per_server(
    agent_home: Path,
    server_content: str,
    expected_field: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
{server_content}

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert expected_field in loader.diagnostics[0].message


@pytest.mark.parametrize("timeout_field", ["connect_timeout", "call_timeout"])
@pytest.mark.parametrize("timeout_value", [0, 601])
def test_mcp_timeout_bounds_are_validated_per_server(
    agent_home: Path,
    timeout_field: str,
    timeout_value: int,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
{timeout_field} = {timeout_value}

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert timeout_field in loader.diagnostics[0].message


@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
@pytest.mark.parametrize("timeout_value", [1, 600])
def test_mcp_timeout_boundaries_are_loaded(
    agent_home: Path,
    transport: str,
    timeout_value: int,
) -> None:
    transport_fields = (
        'command = "uvx"' if transport == "stdio" else 'url = "https://mcp.example.test/service"'
    )
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.boundary]
enabled = true
transport = "{transport}"
{transport_fields}
connect_timeout = {timeout_value}
call_timeout = {timeout_value}
""",
    )

    server = loader.load().mcp["boundary"]

    assert (server.connect_timeout, server.call_timeout) == (timeout_value, timeout_value)


@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
def test_missing_required_transport_fields_are_isolated_per_server(
    agent_home: Path,
    transport: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.invalid]
enabled = true
transport = "{transport}"

[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    configuration = loader.load()

    assert set(configuration.mcp) == {"valid"}
    assert len(loader.diagnostics) == 1
    assert "required" in loader.diagnostics[0].message


@pytest.mark.parametrize("cwd", ["servers", "nested/tools"])
def test_relative_stdio_cwd_resolves_against_workspace(
    agent_home: Path,
    cwd: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.local]
enabled = true
transport = "stdio"
command = "uvx"
cwd = "{cwd}"
""",
    )

    server = loader.load().mcp["local"]

    assert server.cwd == Path(cwd)
    assert server.resolve_cwd(agent_home) == agent_home / cwd


@pytest.mark.parametrize("directory", ["absolute-one", "absolute-two"])
def test_absolute_stdio_cwd_is_preserved(
    agent_home: Path,
    directory: str,
) -> None:
    absolute_cwd = agent_home / directory
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.local]
enabled = true
transport = "stdio"
command = "uvx"
cwd = "{absolute_cwd.as_posix()}"
""",
    )

    server = loader.load().mcp["local"]

    assert server.cwd == absolute_cwd
    assert server.resolve_cwd(Path("ignored-workspace")) == absolute_cwd


def test_mcp_configuration_is_deeply_immutable(agent_home: Path) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example.test/service"
headers = { Authorization = "Bearer secret" }
""",
    )

    configuration = loader.load()
    server = configuration.mcp["http"]

    with pytest.raises(TypeError):
        cast(dict[str, object], configuration.mcp)["other"] = server
    with pytest.raises(TypeError):
        cast(dict[str, str], server.headers)["Authorization"] = "changed"
    with pytest.raises(FrozenInstanceError):
        server.enabled = False  # type: ignore[misc]


def test_malformed_toml_remains_a_fatal_configuration_error(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        '[mcp.servers.broken\ntransport = "stdio"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as raised:
        loader.load()

    assert raised.value.error.code == "config_parse_error"


@pytest.mark.parametrize(
    ("malformed_content", "secret"),
    [
        (
            MINIMAL_VALID_CONFIG
            + """
[mcp.servers.http]
transport = "streamable-http"
headers = { Authorization = "raw-header-secret" }
broken = [
""",
            "raw-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
[mcp.servers.http.headers]
Authorization = "raw-table-header-secret"
broken = [
""",
            "raw-table-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
mcp.servers.http.headers.Authorization = "raw-dotted-header-secret"
broken = [
""",
            "raw-dotted-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
mcp.servers.http."headers".Authorization = "raw-quoted-header-secret"
broken = [
""",
            "raw-quoted-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
mcp.servers.local.env.API_TOKEN = "raw-dotted-env-secret"
broken = [
""",
            "raw-dotted-env-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
[mcp.servers.http]
transport = "streamable-http"
url = "https://mcp.example.test/service"
headers = {
Authorization = "raw-multiline-header-secret"
broken = [
""",
            "raw-multiline-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
[mcp.servers.http.headers
Authorization = "raw-unclosed-table-header-secret"
""",
            "raw-unclosed-table-header-secret",
        ),
        (
            MINIMAL_VALID_CONFIG
            + """
mcp.servers.http = { transport = "streamable-http", url = "https://mcp.example.test/service", headers = { Authorization = "raw-outer-inline-secret" } }
broken = [
""",
            "raw-outer-inline-secret",
        ),
        (
            '\ufeff[mcp.servers.http.headers]\nAuthorization = "raw-bom-header-secret"\n',
            "raw-bom-header-secret",
        ),
        (
            '\u200bmcp.servers.http = { headers = { Authorization = "raw-zero-width-secret" } }\n',
            "raw-zero-width-secret",
        ),
    ],
)
def test_config_view_redacts_headers_before_reporting_toml_parse_error(
    agent_home: Path,
    malformed_content: str,
    secret: str,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(malformed_content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert secret not in view.redacted_content


def test_config_view_redacts_invalid_array_of_header_tables(agent_home: Path) -> None:
    loader = _loader_with_mcp(
        agent_home,
        """
[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example.test/service"

[[mcp.servers.http.headers]]
Authorization = "array-table-header-secret"
""",
    )

    view = loader.view()

    assert view.error is None
    assert len(view.diagnostics) == 1
    diagnostic = view.diagnostics[0]
    assert isinstance(diagnostic, ConfigurationDiagnostic)
    assert diagnostic.mcp_name == "http"
    assert "array-table-header-secret" not in view.redacted_content
    assert "***REDACTED***" in view.redacted_content


@pytest.mark.parametrize(
    ("encoded_name", "unsafe_name"),
    [
        (r"bad\nFORGED", "bad\nFORGED"),
        (r"bad\u001b[31mFORGED", "bad\x1b[31mFORGED"),
    ],
)
def test_invalid_mcp_name_diagnostic_escapes_control_characters(
    agent_home: Path,
    encoded_name: str,
    unsafe_name: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers."{encoded_name}"]
enabled = true
transport = "stdio"
command = "uvx"
""",
    )

    loader.load()
    message = loader.diagnostics[0].message

    assert unsafe_name not in message
    assert "\n" not in message
    assert "\x1b" not in message


@pytest.mark.parametrize(
    ("encoded_field", "unsafe_field"),
    [
        (r"bad\nFORGED", "bad\nFORGED"),
        (r"bad\u001b[31mFORGED", "bad\x1b[31mFORGED"),
    ],
)
def test_invalid_mcp_field_diagnostic_escapes_control_characters(
    agent_home: Path,
    encoded_field: str,
    unsafe_field: str,
) -> None:
    loader = _loader_with_mcp(
        agent_home,
        f"""
[mcp.servers.valid]
enabled = true
transport = "stdio"
command = "uvx"
[mcp.servers.valid.tool_keywords]
"{encoded_field}" = [1]
""",
    )

    loader.load()
    diagnostic = loader.diagnostics[0]

    assert isinstance(diagnostic, ConfigurationDiagnostic)
    assert unsafe_field not in diagnostic.reason
    assert "\n" not in diagnostic.reason
    assert "\x1b" not in diagnostic.reason
