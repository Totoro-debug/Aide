import importlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import omni.terminal.cli as cli
from omni.agent.loop import ModelContextOverflowError
from omni.agent.workspace_state import WorkspaceStateError
from omni.config.agent_home import AgentHome
from omni.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo
from omni.management.service import (
    FatalManagementError,
)
from omni.service.client import ServiceClient, ServiceStartupError
from omni.service.errors import ServiceError
from tests.configuration.test_config import (
    EXPECTED_DEFAULT_CONFIG,
    EXPECTED_REDACTED_CONFIG,
    EXPECTED_REDACTED_MALFORMED_CONFIG,
    MALFORMED_CONFIG,
    MINIMAL_VALID_CONFIG,
    REDACTION_CONFIG,
    VALID_CONFIG,
)


def test_legacy_runtime_module_is_not_discoverable() -> None:
    legacy_module = ".".join(("omni", "agent", "runtime"))
    assert not (Path(__file__).resolve().parents[1] / "omni" / "agent" / "runtime.py").exists()
    assert importlib.util.find_spec(legacy_module) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_module)


def test_service_stop_without_an_active_service_prints_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_service(_home: AgentHome) -> bool:
        return False

    monkeypatch.setattr(ServiceClient, "stop_existing", no_service)
    result = CliRunner().invoke(cli.app, ["service", "stop"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout.count("service_not_running: No active local service was found.") == 1
    assert result.stderr == ""


@pytest.mark.parametrize(
    "error",
    [
        ServiceStartupError("service_port_in_use", "The local service port is occupied."),
        ServiceError("admission_closed", "The local service is stopping."),
    ],
)
def test_web_service_errors_are_reported_without_a_runtime_error_code_conversion(
    monkeypatch: pytest.MonkeyPatch,
    error: ServiceStartupError | ServiceError,
) -> None:
    async def fail_launch() -> str:
        raise error

    monkeypatch.setattr(cli, "_create_web_launch_url", fail_launch)
    result = CliRunner().invoke(cli.app, ["web"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout.count(f"{error.code}: {error.message}") == 1
    assert result.stderr == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", [None, "constructor", "binding", "terminal"])
async def test_cli_service_adapter_binds_terminal_and_closes_client_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None
) -> None:
    events: list[str] = []
    client = SimpleNamespace(
        bus=object(), control=object(), management_dispatcher=object(), confirmation=object()
    )
    home = AgentHome(tmp_path / "home")
    failure = RuntimeError("Controlled terminal failure")

    async def close() -> None:
        events.append("client_close")

    client.close = close

    async def connect(actual_home: AgentHome, directory: Path) -> ServiceClient:
        assert actual_home is home and directory == tmp_path
        events.append("connect")
        return cast(ServiceClient, client)

    class App:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {
                "bus": client.bus,
                "control": client.control,
                "management_dispatcher": client.management_dispatcher,
            }
            events.append("terminal_init")
            if failure_stage == "constructor":
                raise failure

        def bind_confirmation_coordinator(self, coordinator: object) -> None:
            assert coordinator is client.confirmation
            events.append("bind")
            if failure_stage == "binding":
                raise failure

        async def run_async(self) -> None:
            events.append("run")
            if failure_stage == "terminal":
                raise failure

    monkeypatch.setattr(ServiceClient, "connect_or_start", connect)
    monkeypatch.setattr(cli, "TerminalConversationApp", App)
    if failure_stage is None:
        await cli._run_service_cli_conversation(agent_home=home, workspace=tmp_path)
        assert events == ["connect", "terminal_init", "bind", "run", "client_close"]
    else:
        with pytest.raises(RuntimeError) as raised:
            await cli._run_service_cli_conversation(agent_home=home, workspace=tmp_path)
        assert raised.value is failure
        assert events[-1] == "client_close"
    assert events.count("client_close") == 1


@pytest.mark.asyncio
async def test_cli_connection_failure_never_constructs_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = ServiceStartupError("service_port_in_use", "The local service port is occupied.")

    async def connect(_home: AgentHome, _directory: Path) -> ServiceClient:
        raise failure

    def terminal(**_kwargs: object) -> None:
        pytest.fail("Terminal must wait for a successful service connection")

    monkeypatch.setattr(ServiceClient, "connect_or_start", connect)
    monkeypatch.setattr(cli, "TerminalConversationApp", terminal)
    with pytest.raises(ServiceStartupError) as raised:
        await cli._run_service_cli_conversation(agent_home=AgentHome(tmp_path), workspace=tmp_path)
    assert raised.value is failure


def test_cli_reports_unexpected_startup_failure_without_raw_exception_output(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    secret = "sk-startup-secret C:\\sensitive\\skill\\SKILL.md"

    monkeypatch.chdir(workspace)

    class ErrorCarryingFailure(RuntimeError):
        def __init__(self) -> None:
            self.error = ErrorInfo("persistence_error", secret)
            super().__init__(secret)

    failures = (
        ErrorCarryingFailure(),
        ModelContextOverflowError(ErrorInfo("model_context_overflow", secret)),
        FatalManagementError(ErrorInfo("persistence_error", secret)),
    )
    for failure in failures:

        async def fail_startup(
            failure_to_raise: Exception = failure,
            **kwargs: object,
        ) -> None:
            del kwargs
            raise failure_to_raise

        monkeypatch.setattr(cli, "_run_service_cli_conversation", fail_startup)
        result = CliRunner().invoke(cli.app, [])

        assert result.exit_code == 1
        assert result.output.count("persistence_error: MyClaw runtime could not be started.") == 1
        assert secret not in result.output
        assert "Traceback" not in result.output


def test_cli_workspace_state_failure_outputs_one_safe_error_without_path(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    secret_path = workspace / "private-state-location"

    async def fail_workspace(**kwargs: object) -> None:
        del kwargs
        raise WorkspaceStateError(secret_path)

    monkeypatch.setattr(cli, "_run_service_cli_conversation", fail_workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
    assert result.output.count("persistence_error:") == 1
    assert str(secret_path) not in result.output
    assert "Path:" not in result.output
    assert "Traceback" not in result.output


def test_cli_reports_fatal_replacement_failure_once_without_raw_exception_output(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    secret = "reset secret C:\\sensitive\\bus"

    async def fail_replacement(**kwargs: object) -> None:
        del kwargs
        raise FatalManagementError(
            ErrorInfo("persistence_error", "Runtime Session replacement could not be completed.")
        )

    monkeypatch.setattr(cli, "_run_service_cli_conversation", fail_replacement)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
    assert result.output.count("persistence_error:") == 1
    assert secret not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("failure", "expected_code", "secret"),
    (
        (
            ModelContextOverflowError(
                ErrorInfo(
                    "model_context_overflow",
                    MODEL_CONTEXT_OVERFLOW_MESSAGE,
                )
            ),
            "model_context_overflow",
            "C:\\sensitive\\skill\\SKILL.md",
        ),
    ),
)
def test_cli_reports_runtime_context_startup_failures_without_starting_conversation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_code: str,
    secret: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    conversation_calls: list[object] = []

    async def fail_startup(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise failure

    monkeypatch.setattr(cli, "_run_service_cli_conversation", fail_startup)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
    assert result.output.count(f"{expected_code}:") == 1
    assert result.output.count(str(failure)) == 1
    assert secret not in result.output
    assert "Traceback" not in result.output
    assert conversation_calls == []


def run_installed_myclaw(
    agent_home: Path,
    *arguments: str,
    workspace: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("myclaw")
    assert executable is not None
    environment = os.environ.copy()
    environment["HOME"] = str(agent_home.parent)
    environment["USERPROFILE"] = str(agent_home.parent)
    source_root = str(Path(__file__).parent.parent)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    return subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        cwd=agent_home.parent if workspace is None else workspace,
        env=environment,
        text=True,
    )


def assert_plaintext_absent(output: str, *plaintext_values: str) -> None:
    if any(value in output for value in plaintext_values):
        pytest.fail("CLI output leaked a plaintext provider API key", pytrace=False)


def legacy_runtime_log_snapshot(agent_home: Path) -> dict[str, bytes]:
    logs = agent_home / "logs"
    return {
        path.name: path.read_bytes()
        for path in logs.iterdir()
        if path.is_file() and path.name.startswith("run.log.")
    }


def test_installed_myclaw_console_entry_starts() -> None:
    executable = shutil.which("myclaw")

    assert executable is not None
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "MyClaw Personal Agent" in result.stdout


def test_installed_myclaw_generates_missing_configuration_and_stops(
    agent_home: Path,
    workspace: Path,
) -> None:
    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert (agent_home / "config.toml").read_text(encoding="utf-8") == EXPECTED_DEFAULT_CONFIG
    assert result.stdout.count("config_missing") == 1
    assert str(agent_home / "config.toml") in result.stdout
    assert "edit" in result.stdout.lower()
    assert result.stderr == ""
    assert "configuration gate passed" not in result.stdout
    assert not (workspace / ".omni").exists()
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_does_not_modify_legacy_runtime_log_data(
    agent_home: Path,
    workspace: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"legacy slot zero\n")
    (logs / "run.log.1").write_bytes(b"legacy slot one\n")
    (logs / "run.log.cursor").write_bytes(b"1\n")
    (logs / "run.log.lock").write_bytes(b"legacy lock\n")
    before = legacy_runtime_log_snapshot(agent_home)

    result = run_installed_myclaw(agent_home, workspace=workspace)
    config_result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 2
    assert config_result.returncode == 0
    assert legacy_runtime_log_snapshot(agent_home) == before


def test_installed_config_command_generates_and_displays_missing_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 0, result.stderr
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert "Effective runtime.permission_level: workspace-write" in result.stdout
    assert "Effective runtime.exec_shell: auto" in result.stdout
    assert EXPECTED_DEFAULT_CONFIG in result.stdout
    assert "configuration gate passed" not in result.stdout
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".omni").exists()


def test_installed_config_command_redacts_valid_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(REDACTION_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 0, result.stderr
    assert EXPECTED_REDACTED_CONFIG in result.stdout
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".omni").exists()


def test_installed_config_command_reports_mcp_diagnostics_without_secrets(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(
        MINIMAL_VALID_CONFIG
        + """
[mcp.servers.invalid]
enabled = true
transport = "stdio"
command = "uvx"
env = { API_TOKEN = "installed-env-secret" }

[mcp.servers.http]
enabled = true
transport = "streamable-http"
url = "https://mcp.example.test/service"
headers = { Authorization = "Bearer installed-header-secret" }
""",
        encoding="utf-8",
    )

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    visible = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "MCP Server 'invalid' ignored" in visible
    assert "installed-env-secret" not in visible
    assert "installed-header-secret" not in visible
    assert "***REDACTED***" in visible
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".omni").exists()


def test_installed_config_command_keeps_fallback_diagnostic_before_later_fatal_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    content = MINIMAL_VALID_CONFIG.replace(
        "[runtime]\n",
        '[runtime]\npermission_level = "level-secret"\n',
    ).replace('model = "small-model"\n', "", 1)
    config_path = agent_home / "config.toml"
    config_path.write_text(content, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    error = "config_invalid: Configuration field 'models.routes.default.model' is required."
    diagnostic = (
        "Configuration field 'runtime.permission_level' is invalid; using 'workspace-write'."
    )
    path = f"Path: {config_path}"
    assert result.returncode == 2
    assert result.stdout.index(error) < result.stdout.index(diagnostic) < result.stdout.index(path)
    assert "level-secret" not in result.stdout[: result.stdout.index(path)] + result.stderr


def test_installed_config_command_shows_safe_malformed_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(MALFORMED_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("config_parse_error") == 1
    assert result.stdout.count(f"Path: {agent_home / 'config.toml'}") == 1
    assert EXPECTED_REDACTED_MALFORMED_CONFIG in result.stdout
    assert result.stderr == ""
    assert_plaintext_absent(
        result.stdout + result.stderr,
        "first-plaintext-key",
        "second-plaintext-key",
    )
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".omni").exists()


def test_installed_config_command_hides_invalid_utf8_and_traceback(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    config_path = agent_home / "config.toml"
    config_path.write_bytes(b'api_key = "sk-invalid-utf8-secret"\ninvalid = "\xff"\n')

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    visible = result.stdout + result.stderr
    assert result.returncode == 1
    assert "persistence_error" in result.stdout
    assert f"Path: {config_path}" in result.stdout
    assert "sk-invalid-utf8-secret" not in visible
    assert "Traceback" not in visible


def test_installed_config_command_ignores_undefined_content_fields(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    (agent_home / "config.toml").write_text(content, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 0
    assert "config_invalid" not in result.stdout
    assert "runtime.misspelled_setting" not in result.stdout
    assert "misspelled_setting = true" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")
    assert not (workspace / ".omni").exists()


def test_installed_myclaw_rejects_valid_configuration_without_a_tty(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2, result.stderr
    assert "interactive_terminal_required" in result.stdout
    assert "configuration gate passed" not in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "sk-ant-secret")
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".omni").exists()


def test_installed_myclaw_stops_only_on_parse_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    config_path = agent_home / "config.toml"

    config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")
    parse_result = run_installed_myclaw(agent_home, workspace=workspace)

    schema_content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    config_path.write_text(schema_content, encoding="utf-8")
    schema_result = run_installed_myclaw(agent_home, workspace=workspace)

    config_path.write_text(EXPECTED_DEFAULT_CONFIG, encoding="utf-8")
    default_result = run_installed_myclaw(agent_home, workspace=workspace)

    assert (parse_result.returncode, schema_result.returncode, default_result.returncode) == (
        2,
        2,
        2,
    )
    assert "config_parse_error" in parse_result.stdout
    assert "config_invalid" not in schema_result.stdout
    assert "configuration gate passed" not in parse_result.stdout
    assert "interactive_terminal_required" in schema_result.stdout
    assert "interactive_terminal_required" in default_result.stdout
    assert not (workspace / ".omni").exists()
    combined_output = "".join(
        result.stdout + result.stderr for result in (parse_result, schema_result, default_result)
    )
    assert all(result.stderr == "" for result in (parse_result, schema_result, default_result))
    assert_plaintext_absent(
        combined_output,
        "first-plaintext-key",
        "second-plaintext-key",
        "plaintext-primary-key",
    )
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_unsafe_workspace_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state_path = workspace / ".omni"
    state_path.write_text("private collision content", encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "Workspace State" not in result.stdout
    assert str(state_path) not in result.stdout
    assert "private collision content" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert state_path.read_text(encoding="utf-8") == "private collision content"
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_corrupt_schedule_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state_path = workspace / ".omni"
    state_path.mkdir()
    schedule_path = state_path / "schedule.json"
    schedule_path.write_text("{corrupt", encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "schedule_state_error" not in result.stdout
    assert str(schedule_path) not in result.stdout
    assert "{corrupt" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert schedule_path.read_text(encoding="utf-8") == "{corrupt"
    assert not (state_path / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_user_home_workspace_validation(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=agent_home.parent)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "Workspace State" not in result.stdout
    assert str(agent_home) not in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert not (agent_home / "memory").exists()
    assert not (agent_home / "sessions").exists()
