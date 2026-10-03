import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import pytest
from loguru import logger

from myclaw.agent.memory.dream import DreamResult
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.management.commands import (
    MANAGEMENT_COMMANDS,
    RESUME_MANAGEMENT_COMMAND,
    ManagementCommandDispatcher,
    ManagementCommandResult,
)
from myclaw.management.service import ManagementError, RuntimeStatusInput
from myclaw.skills.catalog import SkillMetadata
from tests.fixtures.diagnostic_capture import capture_diagnostics, configured_process_logging
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service

if TYPE_CHECKING:
    from loguru import Message

CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "command-secret"
models = ["model-id"]

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""


class _ResultDream:
    def __init__(self, result: DreamResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self) -> DreamResult:
        self.calls += 1
        return self.result


class _ReloadableLoop:
    def __init__(
        self,
        metadata: tuple[SkillMetadata, ...] = (),
        failure: BaseException | None = None,
    ) -> None:
        self.metadata = metadata
        self.failure = failure
        self.calls = 0

    def reload_skill(self) -> tuple[SkillMetadata, ...]:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.metadata


REDACTED_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MALFORMED_CONFIG_CONTENT = """[runtime
api_key = "first-command-secret"
  API-Key = 'second-command-secret' # this value is not safe
not_api_key = "diagnostic-content"
broken = [
"""

REDACTED_MALFORMED_CONFIG_CONTENT = """[runtime
api_key = "***REDACTED***"
  API-Key = "***REDACTED***"
not_api_key = "diagnostic-content"
broken = [
"""


def test_management_command_catalog_owns_ordered_tokens_and_descriptions() -> None:
    assert tuple((command.token, command.description) for command in MANAGEMENT_COMMANDS) == (
        ("/config", "View User Configuration"),
        ("/status", "View Runtime Status"),
        ("/effort", "Set Chat Reasoning Effort"),
        ("/permission", "Set Foreground Tool Permission Level"),
        ("/resume", "Resume a Conversation Session"),
        ("/restore", "Restore the current Conversation Session"),
        ("/memory", "View Long-term Memory"),
        ("/dream", "Process pending Conversation Summaries"),
        ("/reload_skill", "Reload Skills"),
    )
    assert all(command.token and command.description for command in MANAGEMENT_COMMANDS)
    assert any(command is RESUME_MANAGEMENT_COMMAND for command in MANAGEMENT_COMMANDS)


class _EffortManagement:
    def __init__(self, effort: str = "medium") -> None:
        self.effort = effort
        self.updated: list[str] = []

    async def reasoning_effort(self) -> str:
        return self.effort

    async def update_reasoning_effort(self, effort: str) -> str:
        self.updated.append(effort)
        self.effort = effort
        return effort


class _TypedManagement(_EffortManagement):
    def __init__(self, effort: str = "medium", permission: str = "workspace-write") -> None:
        super().__init__(effort)
        self.permission = permission

    async def status(self) -> Any:
        return "typed-status"

    async def permission_level(self) -> str:
        return self.permission

    async def update_permission_level(self, level: str) -> str:
        self.permission = level
        return level


@pytest.mark.asyncio
async def test_status_projection_and_read_commands_return_typed_selections() -> None:
    management = _TypedManagement(effort="high", permission="read-only")
    dispatcher = ManagementCommandDispatcher(cast(Any, management))

    status = await dispatcher.status()
    permission = await dispatcher.dispatch("/permission")
    effort = await dispatcher.dispatch("/effort")

    assert status.handled is True
    assert cast(Any, status.status_view) == "typed-status"
    assert permission.handled is True
    assert permission.permission_selection == "read-only"
    assert effort.handled is True
    assert effort.effort_selection == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/effort", "/permission"])
async def test_read_selection_commands_render_management_errors_without_a_selection(
    command: str,
) -> None:
    class UnavailableManagement:
        async def reasoning_effort(self) -> str:
            raise ManagementError(ErrorInfo("route_unavailable", "Runtime Generation is unavailable."))

        async def permission_level(self) -> str:
            raise ManagementError(ErrorInfo("route_unavailable", "Runtime Generation is unavailable."))

    dispatcher = ManagementCommandDispatcher(cast(Any, UnavailableManagement()))
    result = await dispatcher.dispatch(command)
    assert result.handled is True
    assert result.output == "route_unavailable: Runtime Generation is unavailable."
    assert result.effort_selection is None
    assert result.permission_selection is None


@pytest.mark.asyncio
async def test_effort_command_returns_current_selection_and_commits_a_neutral_result() -> None:
    management = _EffortManagement("high")
    dispatcher = ManagementCommandDispatcher(cast(Any, management))

    selection = await dispatcher.dispatch("/effort")
    committed = await dispatcher.update_reasoning_effort("xhigh")

    assert selection.handled is True
    assert selection.output is None
    assert selection.effort_selection == "high"
    assert committed.handled is True
    assert committed.output == "Chat reasoning effort: xhigh"
    assert committed.effort_selection is None
    assert management.updated == ["xhigh"]


class _PermissionManagement:
    def __init__(self, level: str = "workspace-write") -> None:
        self.level = level
        self.updated: list[str] = []

    async def permission_level(self) -> str:
        return self.level

    async def update_permission_level(self, level: str) -> str:
        self.updated.append(level)
        self.level = level
        return level


@pytest.mark.asyncio
async def test_permission_command_returns_current_selection_and_commits_only_selected_level() -> (
    None
):
    management = _PermissionManagement("read-only")
    dispatcher = ManagementCommandDispatcher(cast(Any, management))

    selection = await dispatcher.dispatch("/permission")
    assert management.updated == []
    committed = await dispatcher.update_permission_level("full-access")

    assert selection.handled is True
    assert selection.output is None
    assert selection.permission_selection == "read-only"
    assert committed.handled is True
    assert committed.output == "Foreground permission level: full-access"
    assert committed.permission_selection is None
    assert management.updated == ["full-access"]


@pytest.mark.asyncio
async def test_reload_skill_command_returns_published_count_and_metadata(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    metadata = (
        SkillMetadata(
            name="planner",
            description="Plan work",
            path=(agent_home / "skills" / "planner" / "SKILL.md").resolve(),
        ),
        SkillMetadata(
            name="reviewer",
            description="Review work",
            path=(agent_home / "skills" / "reviewer" / "SKILL.md").resolve(),
        ),
    )
    loop = _ReloadableLoop(metadata)
    dispatcher = ManagementCommandDispatcher(
        management_service(home, current_agent_loop=lambda: loop),
    )

    result = await dispatcher.dispatch("/reload_skill")

    assert result.handled is True
    assert result.output == "Skill count: 2"
    assert result.skill_metadata == metadata
    assert loop.calls == 1


@pytest.mark.asyncio
async def test_reload_skill_command_maps_failure_to_one_safe_stable_result(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    loop = _ReloadableLoop(
        failure=RuntimeError(
            "SECRET-SKILL-BODY D:\\private\\skills\\planner\\SKILL.md traceback details",
        ),
    )
    dispatcher = ManagementCommandDispatcher(
        management_service(home, current_agent_loop=lambda: loop),
    )

    diagnostics = capture_diagnostics()
    try:
        result = await dispatcher.dispatch("/reload_skill")
    finally:
        diagnostics.close()

    assert result.handled is True
    assert result.output == "skill_reload_failed: Skill reload failed."
    assert result.skill_metadata is None
    assert "SECRET-SKILL-BODY" not in (result.output or "")
    assert "D:\\private\\skills" not in (result.output or "")
    assert "traceback" not in (result.output or "").casefold()
    assert "SECRET-SKILL-BODY" not in diagnostics.event_text
    assert "D:\\private\\skills" not in diagnostics.event_text
    assert "traceback" not in diagnostics.event_text.casefold()
    assert "RuntimeError" in diagnostics.event_text
    assert loop.calls == 1


@pytest.mark.asyncio
async def test_concurrent_reload_dispatches_are_linearized_in_call_order(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    first = (
        SkillMetadata(
            name="first",
            description="First state",
            path=(agent_home / "skills" / "first" / "SKILL.md").resolve(),
        ),
    )
    second = (
        SkillMetadata(
            name="second",
            description="Second state",
            path=(agent_home / "skills" / "second" / "SKILL.md").resolve(),
        ),
    )

    class SequencedReloadLoop(_ReloadableLoop):
        def reload_skill(self) -> tuple[SkillMetadata, ...]:
            metadata = (first, second)[self.calls]
            self.calls += 1
            return metadata

    loop = SequencedReloadLoop()
    dispatcher = ManagementCommandDispatcher(
        management_service(home, current_agent_loop=lambda: loop),
    )

    results = await asyncio.gather(
        dispatcher.dispatch("/reload_skill"),
        dispatcher.dispatch("/reload_skill"),
    )

    assert tuple(result.skill_metadata for result in results) == (first, second)
    assert tuple(result.output for result in results) == ("Skill count: 1", "Skill count: 1")
    assert loop.calls == 2


@pytest.mark.parametrize(
    ("dream_result", "expected"),
    (
        (
            DreamResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=0,
            ),
            "No pending summaries",
        ),
        (
            DreamResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=False,
                cursor=1,
                error=ErrorInfo("model_failed", "Memory model failed."),
            ),
            (
                "model_failed: Memory model failed.\n"
                "processed_count: 0\n"
                "memory_updated: false\n"
                "cursor: 1"
            ),
        ),
        (
            DreamResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=False,
                cursor=0,
                error=ErrorInfo(
                    "persistence_error",
                    "Summary Cursor could not be updated.",
                ),
            ),
            (
                "persistence_error: Summary Cursor could not be updated.\n"
                "processed_count: 0\n"
                "memory_updated: false\n"
                "cursor: 0"
            ),
        ),
    ),
    ids=("no-pending", "model-failure", "cursor-publication-failure"),
)
@pytest.mark.asyncio
async def test_dream_command_projects_the_complete_final_result(
    agent_home: Path,
    dream_result: DreamResult,
    expected: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    dream = _ResultDream(dream_result)
    dispatcher = ManagementCommandDispatcher(management_service(home, dream=dream))

    result = await dispatcher.dispatch("/dream")

    assert result.handled is True
    assert result.output == expected
    assert dream.calls == 1


SCHEMA_INVALID_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "schema-command-secret"
models = ["model-id"]
unexpected = true

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

REDACTED_SCHEMA_INVALID_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]
unexpected = true

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""


class _StatusProjectionLoop:
    def __init__(self, projection: RuntimeStatusInput) -> None:
        self._projection = projection

    def runtime_status_input(self) -> RuntimeStatusInput:
        return self._projection


DEFAULT_CONFIG_CONTENT = """[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false
compact_ratio = 0.9
permission_level = "workspace-write"
exec_shell = "auto"

[memory]
batch_size = 10
schedule = "0 * * * *"

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
models = []

# Replace provider_id, model, and model limits with values supported by your provider.
# Remove any purpose-specific route to fall back to default.
[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.schedule]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
"""

LOCAL_OFFSET = timezone(timedelta(hours=8))
STATUS_CREATED_AT = datetime(2026, 7, 11, 15, 30, tzinfo=LOCAL_OFFSET)
STATUS_SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


class ConversationProviderSpy:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.provider_calls = 0

    async def converse(self, text: str) -> None:
        self.messages.append(text)
        self.provider_calls += 1


@pytest.mark.asyncio
async def test_config_command_returns_renderable_complete_redacted_text(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(management_service(home))

    result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output == (
        "Effective runtime.compact_ratio: 0.9\n"
        "Effective runtime.permission_level: workspace-write\n"
        "Effective runtime.exec_shell: auto\n"
        f"Path: {config_path}\n{REDACTED_CONFIG_CONTENT}"
    )
    assert "command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_keeps_fallback_diagnostic_before_later_fatal_error(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    content = '[runtime]\npermission_level = "level-secret"\n\n' + CONFIG_CONTENT.replace(
        'model = "model-id"\n', "", 1
    )
    config_path.write_text(content, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(management_service(home))

    result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output is not None
    error = "config_invalid: Configuration field 'models.routes.default.model' is required."
    diagnostic = (
        "Configuration field 'runtime.permission_level' is invalid; using 'workspace-write'."
    )
    path = f"Path: {config_path}"
    assert result.output.index(error) < result.output.index(diagnostic) < result.output.index(path)
    assert "level-secret" not in result.output[: result.output.index(path)]


@pytest.mark.asyncio
async def test_config_command_renders_safe_parse_error_and_redacted_source(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(MALFORMED_CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(management_service(home))
    with configured_process_logging():
        result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output == (
        "config_parse_error: User Configuration TOML could not be parsed.\n"
        f"Path: {config_path}\n"
        f"{REDACTED_MALFORMED_CONFIG_CONTENT}"
    )
    assert "first-command-secret" not in result.output
    assert "second-command-secret" not in result.output
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_config_command_renders_safe_persistence_failure(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_bytes(b'api_key = "raw-command-secret"\xff')
    dispatcher = ManagementCommandDispatcher(management_service(home))
    with configured_process_logging():
        result = await dispatcher.dispatch("/config")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: User Configuration could not be read or written.",
    )
    assert result.output is not None
    assert "raw-command-secret" not in result.output
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_config_command_ignores_undefined_source_fields(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(SCHEMA_INVALID_CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(management_service(home))

    result = await dispatcher.dispatch("/config")

    assert result.output == (
        "Effective runtime.compact_ratio: 0.9\n"
        "Effective runtime.permission_level: workspace-write\n"
        "Effective runtime.exec_shell: auto\n"
        f"Path: {config_path}\n{REDACTED_SCHEMA_INVALID_CONFIG_CONTENT}"
    )
    assert "schema-command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_generates_and_displays_missing_configuration(
    agent_home: Path,
) -> None:
    config_path = agent_home / "config.toml"
    dispatcher = ManagementCommandDispatcher(management_service(AgentHome(agent_home)))

    result = await dispatcher.dispatch("/config")

    assert result.output == (
        "Effective runtime.compact_ratio: 0.9\n"
        "Effective runtime.permission_level: workspace-write\n"
        "Effective runtime.exec_shell: auto\n"
        f"Path: {config_path}\n{DEFAULT_CONFIG_CONTENT}"
    )
    assert config_path.read_text(encoding="utf-8") == DEFAULT_CONFIG_CONTENT


@pytest.mark.asyncio
async def test_memory_command_returns_renderable_complete_disk_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    content = "# Long-term Memory\n\n## Lesson\n\u5b8c\u6574\u5185\u5bb9\n" + (
        "memory-line\n" * 8_000
    )
    state.long_term_memory_path.write_text(content, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            memory_manager=MemoryManager(state),
        )
    )

    result = await dispatcher.dispatch("/memory")

    assert result.handled is True
    assert result.output == content


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "error", "cancelled"))
async def test_dispatch_preserves_session_log_scope(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    management = management_service(AgentHome(agent_home))
    dispatcher = ManagementCommandDispatcher(management)
    entered = asyncio.Event()
    release = asyncio.Event()
    marker = str(uuid4())
    session_id = f"outer-session-{marker}"
    failure = RuntimeError("memory view failed")
    records: list[tuple[str, object]] = []

    def capture(message: "Message") -> None:
        record = message.record
        records.append((record["message"], record["extra"].get("session_id")))

    async def memory_view() -> str:
        logger.warning("waiting")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            logger.warning("cancelled")
            raise
        logger.warning("resumed")
        if outcome == "error":
            raise failure
        return "current memory\n"

    async def dispatch_in_session() -> ManagementCommandResult:
        with logger.contextualize(dispatch_scope=marker, session_id=session_id):
            logger.warning("before")
            try:
                return await dispatcher.dispatch("/memory")
            finally:
                logger.warning("restored")

    monkeypatch.setattr(management, "memory_view", memory_view)
    sink_id = logger.add(
        capture,
        level="WARNING",
        filter=lambda record: record["extra"].get("dispatch_scope") == marker,
        catch=False,
    )
    task = asyncio.create_task(dispatch_in_session())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if outcome == "cancelled":
            task.cancel("cancelled by test")
            with pytest.raises(asyncio.CancelledError, match="cancelled by test"):
                await asyncio.wait_for(task, timeout=5)
            assert task.cancelled()
        else:
            release.set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="memory view failed") as raised:
                    await asyncio.wait_for(task, timeout=5)
                assert raised.value is failure
            else:
                result = await asyncio.wait_for(task, timeout=5)
                assert result == ManagementCommandResult(handled=True, output="current memory\n")
    finally:
        try:
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        finally:
            logger.remove(sink_id)

    assert records == [
        ("before", session_id),
        ("waiting", None),
        ("cancelled" if outcome == "cancelled" else "resumed", None),
        ("restored", session_id),
    ]


@pytest.mark.asyncio
async def test_memory_command_renders_safe_persistence_failure(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.unlink()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
        )
    )
    with configured_process_logging():
        result = await dispatcher.dispatch("/memory")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: Long-term Memory could not be read.",
    )
    assert capsys.readouterr().err == ""
    assert not state.logs_directory.exists()


@pytest.mark.asyncio
async def test_status_command_renders_safe_persistence_failure(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingStatusProjection:
        def runtime_status_input(self) -> RuntimeStatusInput:
            raise OSError("PRIVATE_STATUS_PERSISTENCE_BODY_52")

    def failing_loop() -> FailingStatusProjection:
        return FailingStatusProjection()

    home = AgentHome(agent_home)
    home.initialize()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            current_agent_loop=failing_loop,
            monotonic=lambda: 0.0,
        )
    )
    with configured_process_logging():
        result = await dispatcher.dispatch("/status")

    assert result.output == "persistence_error: Runtime status could not be read."
    assert result.status_view is None
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_status_command_renders_actual_runtime_and_session_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session = Session.create(
        state,
        now=lambda: STATUS_CREATED_AT,
        new_uuid=iter((STATUS_SESSION_UUID,)).__next__,
    )
    seed_session_state(
        session,
        messages=[
            {
                "role": "user",
                "content": "Session state.",
                "timestamp": STATUS_CREATED_AT.isoformat(timespec="milliseconds"),
            },
            {
                "role": "assistant",
                "content": "Visible.",
                "timestamp": STATUS_CREATED_AT.isoformat(timespec="milliseconds"),
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
            },
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
            },
            "summary": "",
        },
        last_compacted=1,
    )
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            current_agent_loop=lambda: _StatusProjectionLoop(
                RuntimeStatusInput(
                    projected_messages=({"role": "system", "content": "abcd"},),
                    session_id=session.session_id,
                    session_title="New Conversation",
                    session_message_count=len(session.messages),
                    last_compacted=session.last_compacted,
                    cumulative_usage=(
                        ("model_calls", 1),
                        ("input_tokens", 10),
                        ("output_tokens", 3),
                        ("total_tokens", 13),
                    ),
                    chat_model="fallback/chat-model",
                    context_window=8,
                    generation_started_at=10.0,
                )
            ),
            monotonic=lambda: 75.8,
        )
    )

    result = await dispatcher.dispatch("/status")

    assert result.handled is True
    assert result.status_view is not None
    assert result.output == json.dumps(result.status_view.to_dict(), ensure_ascii=False, indent=2)
    assert json.loads(result.output or "") == {
        "version": "0.1.0",
        "chat_model": "fallback/chat-model",
        "chat_reasoning_effort": "medium",
        "uptime_seconds": 65,
        "context_window": 8,
        "max_output": 0,
        "available_context": 8,
        "compact_ratio": 0.9,
        "compact_context_window": 8,
        "projected_next_request_tokens": 1,
        "projection_source": "estimated",
        "input_budget_used_percent": 12.5,
        "session_message_count": 2,
        "last_compacted": 1,
        "configured_permission_level": "workspace-write",
        "current_permission_level": "workspace-write",
        "cumulative_usage": {
            "model_calls": 1,
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
        },
        "schedule": {"status": "available", "active_job_count": 0},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "/unknown",
        "/config extra",
        "/memory ",
        "/CONFIG",
        "/reload_skill extra",
        "/effort high",
        "/effort ",
        "/EFFORT",
        "/permission full-access",
        "/permission ",
        "/PERMISSION",
    ],
)
async def test_unknown_or_inexact_slash_command_is_left_unhandled(
    agent_home: Path,
    command: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    dispatcher = ManagementCommandDispatcher(management_service(home))

    result = await dispatcher.dispatch(command)

    assert (result.handled, result.output) == (False, None)


@pytest.mark.asyncio
async def test_management_commands_bypass_conversation_and_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(CONFIG_CONTENT, encoding="utf-8")
    state.long_term_memory_path.write_text("current memory\n", encoding="utf-8")
    reload_loop = _ReloadableLoop()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            current_agent_loop=lambda: reload_loop,
            workspace_state=state,
            memory_manager=MemoryManager(state),
        )
    )
    conversation = ConversationProviderSpy()

    async def dispatch_or_converse(text: str) -> bool:
        result = await dispatcher.dispatch(text)
        if not result.handled:
            await conversation.converse(text)
        return result.handled

    config_handled = await dispatch_or_converse("/config")
    memory_handled = await dispatch_or_converse("/memory")
    reload_handled = await dispatch_or_converse("/reload_skill")

    assert (config_handled, memory_handled, reload_handled) == (True, True, True)
    assert conversation.messages == []
    assert conversation.provider_calls == 0
    assert reload_loop.calls == 1

    unknown_handled = await dispatch_or_converse("/ordinary-slash")

    assert unknown_handled is False
    assert conversation.messages == ["/ordinary-slash"]
    assert conversation.provider_calls == 1
