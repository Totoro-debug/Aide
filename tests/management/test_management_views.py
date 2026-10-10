import asyncio
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
import tiktoken

from aide.agent.memory.dream import DreamResult
from aide.agent.memory.manager import MemoryManager
from aide.agent.permission import RuntimePermissionControl
from aide.agent.session.session import Session
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.management.commands import ManagementCommandDispatcher
from aide.management.service import (
    ManagementError,
    RuntimeStatus,
    RuntimeStatusInput,
)
from aide.provider.models import ReasoningEffort
from aide.utils.errors import ErrorInfo
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures.diagnostic_capture import capture_diagnostics
from tests.fixtures.session import seed_session_state
from tests.management.factories import management_service

CONFIG_WITH_PLAINTEXT_KEYS = '''# User Configuration remains source-preserved.
[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "first-plaintext-key"

[models.providers.primary.models."model-id"]
context_window = 4096
max_output = 512
temperature = 0
reasoning_effort = "mid"
timeout = 60

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""

[models.providers.empty-template.models]

[models.routes.chat]
provider_id = "primary"
model = "model-id"
'''

EXPECTED_REDACTED_CONFIG = '''# User Configuration remains source-preserved.
[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"

[models.providers.primary.models."model-id"]
context_window = 4096
max_output = 512
temperature = 0
reasoning_effort = "mid"
timeout = 60

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""

[models.providers.empty-template.models]

[models.routes.chat]
provider_id = "primary"
model = "model-id"
'''


LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123456, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _StatusProjectionLoop:
    def __init__(self, projection: RuntimeStatusInput) -> None:
        self._projection = projection
        self.projection_calls = 0

    def runtime_status_input(self) -> RuntimeStatusInput:
        self.projection_calls += 1
        return self._projection


class _MemoryReader:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def read_long_term(self) -> str:
        self.calls += 1
        return self._content


class _DreamRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self) -> DreamResult:
        self.calls += 1
        return DreamResult(
            status="Dream complete",
            processed_count=1,
            memory_updated=True,
            cursor=1,
        )


class _ReasoningEffortControl:
    def __init__(
        self,
        effort: ReasoningEffort = "mid",
        *,
        events: list[str] | None = None,
    ) -> None:
        self.effort = effort
        self.events = events

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        return self.effort

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        if self.events is not None:
            self.events.append("runtime")
        self.effort = effort


@pytest.mark.asyncio
async def test_reasoning_effort_service_publishes_and_reads_runtime_value(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    control = _ReasoningEffortControl("high")
    service = management_service(home, reasoning_effort_control=control)

    assert await service.reasoning_effort() == "high"
    assert await service.update_reasoning_effort("xhigh") == "xhigh"
    assert control.effort == "xhigh"
    assert await service.reasoning_effort() == "xhigh"


@pytest.mark.asyncio
async def test_reasoning_effort_update_publishes_before_persisting_latest_configuration(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_WITH_PLAINTEXT_KEYS, encoding="utf-8")
    events: list[str] = []
    control = _ReasoningEffortControl(events=events)
    service = management_service(home, reasoning_effort_control=control)
    original_replace = HOST_FILESYSTEM.atomic_replace_text

    def record_replace(target: Path, content: str) -> None:
        events.append("persistence")
        original_replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_replace)

    result = await service.update_reasoning_effort("xhigh")

    assert result == "xhigh"
    assert events == ["runtime", "persistence"]
    assert 'reasoning_effort = "xhigh"' in config_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reasoning_effort_persistence_failure_keeps_success_and_logs_safely(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_WITH_PLAINTEXT_KEYS, encoding="utf-8")
    original = config_path.read_text(encoding="utf-8")
    control = _ReasoningEffortControl()
    service = management_service(home, reasoning_effort_control=control)

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("replacement failed for first-plaintext-key")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_replace)
    diagnostics = capture_diagnostics()
    try:
        result = await service.update_reasoning_effort("max")
    finally:
        diagnostics.close()

    assert result == "max"
    assert await service.reasoning_effort() == "max"
    assert config_path.read_text(encoding="utf-8") == original
    assert diagnostics.event_text.splitlines() == [
        "Reasoning Effort persistence failed type=OSError"
    ]
    assert "first-plaintext-key" not in diagnostics.text
    assert "models.providers.primary" not in diagnostics.text


@pytest.mark.asyncio
async def test_reasoning_effort_configuration_failure_keeps_runtime_success_and_safe_log(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(
        'api_key = "management-secret"\n[models.routes.chat\n',
        encoding="utf-8",
    )
    control = _ReasoningEffortControl()
    service = management_service(home, reasoning_effort_control=control)
    diagnostics = capture_diagnostics()
    try:
        result = await service.update_reasoning_effort("high")
    finally:
        diagnostics.close()

    assert result == "high"
    assert await service.reasoning_effort() == "high"
    assert diagnostics.event_text.splitlines() == [
        "Reasoning Effort persistence failed type=ConfigError"
    ]
    assert "management-secret" not in diagnostics.text
    assert "models.routes.chat" not in diagnostics.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "as_bytes", "effort"),
    [
        (
            b'api_key = "management-secret"\xff',
            True,
            "high",
        ),
        (
            CONFIG_WITH_PLAINTEXT_KEYS.partition("\n[models.routes.chat]")[0] + "\n",
            False,
            "xhigh",
        ),
        (
            CONFIG_WITH_PLAINTEXT_KEYS.replace("max_output = 512", "max_output = 4096"),
            False,
            "max",
        ),
    ],
    ids=("invalid-utf8", "missing-default", "candidate-validation"),
)
async def test_reasoning_effort_failure_keeps_runtime_status_and_disk_unchanged(
    agent_home: Path,
    content: str | bytes,
    as_bytes: bool,
    effort: ReasoningEffort,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    if as_bytes:
        assert isinstance(content, bytes)
        config_path.write_bytes(content)
        original = content
    else:
        assert isinstance(content, str)
        config_path.write_text(content, encoding="utf-8")
        original = config_path.read_bytes()
    control = _ReasoningEffortControl()
    service = management_service(home, reasoning_effort_control=control)
    diagnostics = capture_diagnostics()
    try:
        result = await service.update_reasoning_effort(effort)
    finally:
        diagnostics.close()

    assert result == effort
    assert await service.reasoning_effort() == effort
    assert (await service.status()).chat_reasoning_effort == effort
    assert config_path.read_bytes() == original
    assert diagnostics.event_text.splitlines() == [
        "Reasoning Effort persistence failed type=ConfigError"
    ]
    assert "management-secret" not in diagnostics.text
    assert "models.routes.chat" not in diagnostics.text


@pytest.mark.asyncio
async def test_config_view_returns_complete_source_with_plaintext_keys_redacted(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_WITH_PLAINTEXT_KEYS, encoding="utf-8")

    view = await management_service(home).config_view()

    assert (view.path, view.redacted_content, view.error) == (
        config_path,
        EXPECTED_REDACTED_CONFIG,
        None,
    )


@pytest.mark.asyncio
async def test_config_view_converts_decode_failure_to_safe_persistence_error(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_bytes(b'api_key = "raw-secret"\xff')

    with pytest.raises(ManagementError) as raised:
        await management_service(home).config_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "User Configuration could not be read or written.",
    )
    assert "raw-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_memory_view_reads_complete_latest_utf8_content_on_every_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".aide")
    memory_path = state.long_term_memory_path
    memory_path.write_text("initial memory\n", encoding="utf-8")
    service = management_service(
        home,
        workspace_state=state,
        memory_manager=MemoryManager(state),
    )

    initial = await service.memory_view()
    updated_content = "# Long-term Memory\n\n\u7528\u6237\u504f\u597d\n" + (
        "complete-content\n" * 8_000
    )
    memory_path.write_text(updated_content, encoding="utf-8")
    updated = await service.memory_view()

    assert initial == "initial memory\n"
    assert updated == updated_content


@pytest.mark.asyncio
async def test_memory_view_converts_decode_failure_to_safe_persistence_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".aide")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.write_bytes(b"raw-secret\xff")

    with pytest.raises(ManagementError) as raised:
        await management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
        ).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )
    assert str(raised.value) == "Long-term Memory could not be read."
    assert "raw-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_memory_view_converts_read_failure_to_safe_persistence_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".aide")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.unlink()

    with pytest.raises(ManagementError) as raised:
        await management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
        ).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )


@pytest.mark.asyncio
async def test_status_encoding_load_keeps_service_responsive(
    agent_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = tiktoken.get_encoding("o200k_base")
    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()

    def load(name: str) -> tiktoken.Encoding:
        started.set()
        released.wait(timeout=1)
        finished.set()
        return official

    home = AgentHome(agent_home)
    home.initialize()
    service = management_service(home)
    monkeypatch.setattr(tiktoken, "get_encoding", load)
    task = asyncio.create_task(service.status())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert not finished.is_set(), "status encoding loading blocked the service event loop"
        released.set()
        assert (await task).projection_source == "estimated"
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_status_reports_prepared_session_and_tokenizer_estimate(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    monotonic = iter((112.9,)).__next__
    management = management_service(
        home,
        workspace_state=state,
        current_agent_loop=lambda: _StatusProjectionLoop(
            RuntimeStatusInput(
                projected_messages=(
                    {"role": "system", "content": "abcd"},
                    {"role": "user", "content": "\u00e9\u4f60"},
                ),
                projected_tools=({"name": "tool"},),
                session_id=session.session_id,
                session_title="New Conversation",
                context_window=10,
                chat_model="primary/model-id",
                cumulative_usage=(
                    ("model_calls", 0),
                    ("input_tokens", 0),
                    ("output_tokens", 0),
                    ("total_tokens", 0),
                ),
                generation_started_at=100.0,
            )
        ),
        monotonic=monotonic,
    )

    status = await management.status()

    assert status == RuntimeStatus(
        version="0.1.0",
        chat_model="primary/model-id",
        chat_reasoning_effort="mid",
        uptime_seconds=12,
        context_window=10,
        max_output=0,
        available_context=10,
        compact_ratio=0.9,
        compact_context_window=9,
        projected_next_request_tokens=16,
        projection_source="estimated",
        input_budget_used_percent=160.0,
        session_message_count=0,
        last_compacted=0,
        cumulative_usage={
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        schedule={"status": "available", "active_job_count": 0},
    )


@pytest.mark.asyncio
async def test_status_projects_the_current_runtime_reasoning_effort(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    control = _ReasoningEffortControl("xhigh")
    service = management_service(home, reasoning_effort_control=control)

    status = await service.status()

    assert status.chat_reasoning_effort == "xhigh"
    assert status.to_dict()["chat_reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "code", "message"),
    [
        ("gpt-4o", "model_failed", "Check network access"),
        ("gpt-4", "model_invalid_request", "Model"),
    ],
)
async def test_status_preserves_safe_tokenizer_failures_for_protocol_clients(
    agent_home: Path, model: str, code: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    projection = RuntimeStatusInput(
        projected_messages=({"role": "user", "content": "hello"},),
        model=model,
        context_window=200000,
    )
    service = management_service(home, current_agent_loop=lambda: _StatusProjectionLoop(projection))

    def reject_load(name: str) -> tiktoken.Encoding:
        raise OSError("raw-secret download details")

    monkeypatch.setattr(tiktoken, "get_encoding", reject_load)
    result = await ManagementCommandDispatcher(service).status()

    assert result.status_view is None
    assert result.management_error is not None
    assert result.management_error.code == code
    assert message in result.management_error.message
    assert result.output is not None and result.output.startswith(code)
    assert "raw-secret" not in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [None, 0, 8])
async def test_status_exposes_last_request_usage_separately_from_cumulative_usage(
    agent_home: Path, cached: int | None,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    projection = RuntimeStatusInput(
        context_window=200000,
        cumulative_usage=(("input_tokens", 150),),
        last_request_usage={"input_tokens": 10, "cached_input_tokens": cached},
    )
    service = management_service(
        home, current_agent_loop=lambda: _StatusProjectionLoop(projection),
    )
    status = await service.status()
    assert status.to_dict()["last_request_usage"] == projection.last_request_usage
    assert status.cumulative_usage == {"input_tokens": 150}


@pytest.mark.asyncio
async def test_status_reports_configured_and_current_foreground_permission_only(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    permission_control = RuntimePermissionControl("full-access")
    permission_control.select("read-only")
    service = management_service(
        home,
        permission_control=permission_control,
        schedule_status=lambda: {
            "status": "available",
            "active_job_count": 4,
            "permission_level": "full-access",
            "background_confirmation_count": 2,
            "future_field": "must remain private",
        },
    )

    status = await service.status()

    assert status.configured_permission_level == "full-access"
    assert status.current_permission_level == "read-only"
    rendered = status.to_dict()
    assert rendered["configured_permission_level"] == "full-access"
    assert rendered["current_permission_level"] == "read-only"
    assert status.schedule == {"status": "available", "active_job_count": 4}
    assert "background_confirmation_count" not in rendered


@pytest.mark.asyncio
async def test_permission_update_is_process_local_and_same_level_is_stable(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = home.path / "config.toml"
    config_path.write_text(
        '''[runtime]
permission_level = "full-access"

[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "secret"

[models.providers.primary.models."model-id"]
context_window = 4096
max_output = 512
temperature = 0
reasoning_effort = "mid"
timeout = 60

[models.routes.chat]
provider_id = "primary"
model = "model-id"
''',
        encoding="utf-8",
    )
    permission_control = RuntimePermissionControl("full-access")
    service = management_service(home, permission_control=permission_control)
    original = config_path.read_text(encoding="utf-8")

    assert await service.update_permission_level("read-only") == "read-only"
    assert await service.update_permission_level("read-only") == "read-only"

    assert permission_control.configured() == "full-access"
    assert permission_control.current() == "read-only"
    assert config_path.read_text(encoding="utf-8") == original
    config_view = await service.config_view()
    assert "Effective runtime.permission_level: full-access" in config_view.header_text()
    assert "Effective runtime.exec_shell: auto" in config_view.header_text()


@pytest.mark.asyncio
async def test_status_reads_one_current_loop_projection_per_request_and_resets_uptime(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    first = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "first"},),
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="First generation",
            session_message_count=2,
            last_compacted=1,
            cumulative_usage=(
                ("model_calls", 1),
                ("input_tokens", 10),
                ("output_tokens", 3),
                ("total_tokens", 13),
            ),
            chat_model="first-provider/first-model",
            context_window=100,
            generation_started_at=10.0,
        )
    )
    second = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "second"},),
            session_id="20260711-153012-123457_6fa459ea-ee8a-4ca4-894e-db77e160355e",
            session_title="Second generation",
            session_message_count=5,
            last_compacted=4,
            cumulative_usage=(
                ("model_calls", 2),
                ("input_tokens", 20),
                ("output_tokens", 6),
                ("total_tokens", 26),
            ),
            chat_model="second-provider/second-model",
            context_window=200,
            generation_started_at=50.0,
        )
    )
    current = first
    provider_calls = 0

    def current_loop() -> _StatusProjectionLoop:
        nonlocal provider_calls
        provider_calls += 1
        return current

    monotonic = iter((100.0, 60.0)).__next__
    service = management_service(
        home,
        current_agent_loop=current_loop,
        monotonic=monotonic,
    )

    first_status = await service.status()
    current = second
    second_status = await service.status()

    assert first_status.session_message_count == 2
    assert first_status.last_compacted == 1
    assert first_status.chat_model == "first-provider/first-model"
    assert first_status.context_window == 100
    assert first_status.uptime_seconds == 90
    assert first_status.cumulative_usage["total_tokens"] == 13
    assert second_status.session_message_count == 5
    assert second_status.last_compacted == 4
    assert second_status.chat_model == "second-provider/second-model"
    assert second_status.context_window == 200
    assert second_status.uptime_seconds == 10
    assert second_status.cumulative_usage["total_tokens"] == 26
    assert provider_calls == 2
    assert first.projection_calls == 1
    assert second.projection_calls == 1


@pytest.mark.asyncio
async def test_management_status_builds_once_from_the_current_loop_projection(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "status"},),
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="Current generation",
            session_message_count=3,
            last_compacted=2,
            cumulative_usage=(
                ("model_calls", 1),
                ("input_tokens", 2),
                ("output_tokens", 3),
                ("total_tokens", 5),
            ),
            chat_model="provider/model",
            context_window=100,
            generation_started_at=10.0,
        )
    )
    provider_calls = 0
    schedule_calls = 0

    def current_loop() -> _StatusProjectionLoop:
        nonlocal provider_calls
        provider_calls += 1
        return loop

    def schedule_status() -> dict[str, object]:
        nonlocal schedule_calls
        schedule_calls += 1
        return {"status": "available", "active_job_count": 2}

    service = management_service(
        home,
        current_agent_loop=current_loop,
        monotonic=lambda: 15.0,
        schedule_status=schedule_status,
    )

    status = await service.status()

    assert status.session_message_count == 3
    assert status.last_compacted == 2
    assert status.uptime_seconds == 5
    assert status.cumulative_usage["total_tokens"] == 5
    assert provider_calls == 1
    assert loop.projection_calls == 1
    assert schedule_calls == 1
    assert status.schedule == {"status": "available", "active_job_count": 2}


@pytest.mark.asyncio
async def test_status_uptime_clamps_a_monotonic_clock_rollback_to_zero(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "status"},),
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="Rollback generation",
            chat_model="provider/model",
            context_window=100,
            generation_started_at=100.0,
        )
    )
    service = management_service(
        home,
        current_agent_loop=lambda: loop,
        monotonic=lambda: 50.0,
    )

    status = await service.status()

    assert status.uptime_seconds == 0


@pytest.mark.asyncio
async def test_generation_sensitive_views_snapshot_each_current_provider_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    target = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    seed_session_state(
        target,
        messages=[
            {
                "role": "user",
                "content": "Resume target",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    target.close()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "current"},),
        )
    )
    memory = _MemoryReader("current memory")
    dream = _DreamRunner()
    current_loop_calls = 0
    replacements: list[tuple[str, bool]] = []

    def current_loop() -> _StatusProjectionLoop:
        nonlocal current_loop_calls
        current_loop_calls += 1
        return loop

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        replacements.append((session_id, force))

    service = management_service(
        home,
        current_agent_loop=current_loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        memory_manager=memory,
        dream=dream,
    )

    assert await service.memory_view() == "current memory"
    dream_result = await service.dream()
    resume_result = await service.resume(target.session_id, force=True)

    assert dream_result.status == "Dream complete"
    assert resume_result.session_id == target.session_id
    assert current_loop_calls == 3
    assert memory.calls == 1
    assert dream.calls == 1
    assert replacements == [(target.session_id, True)]

    invalid_id = "20260711-153012-123456_6fa459ea-ee8a-4ca4-894e-db77e160355e"
    with pytest.raises(ManagementError) as raised:
        await service.resume(invalid_id)

    assert raised.value.error.code == "model_invalid_request"
    assert current_loop_calls == 4
    assert replacements == [(target.session_id, True)]


@pytest.mark.asyncio
async def test_resume_prepares_before_loading_resumable_sessions(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    target = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    seed_session_state(
        target,
        messages=[
            {
                "role": "user",
                "content": "Resume target",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            }
        ],
        metadata={
            "title": "Untitled session",
            "token_usage": {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "summary": "",
        },
        last_compacted=0,
    )
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            projected_messages=({"role": "system", "content": "current"},),
        )
    )
    events: list[str] = []

    async def prepare_session_resume(session_id: str) -> None:
        assert session_id == target.session_id
        events.append("prepare")
        target.close()

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        assert session_id == target.session_id
        assert force is False
        events.append("replace")

    service = management_service(
        home,
        current_agent_loop=lambda: loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        prepare_session_resume=prepare_session_resume,
    )

    result = await service.resume(target.session_id)

    assert result.session_id == target.session_id
    assert events == ["prepare", "replace"]


@pytest.mark.asyncio
async def test_generation_sensitive_views_keep_unavailable_error_and_skip_resources(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    memory = _MemoryReader("must not be read")
    dream = _DreamRunner()
    current_calls = 0
    schedule_calls = 0
    prepare_calls = 0
    replacements: list[tuple[str, bool]] = []

    def unavailable_loop() -> _StatusProjectionLoop:
        nonlocal current_calls
        current_calls += 1
        raise ManagementError(ErrorInfo("route_unavailable", "Runtime Generation is unavailable."))

    def schedule_status() -> dict[str, object]:
        nonlocal schedule_calls
        schedule_calls += 1
        return {"status": "available"}

    async def prepare_session_resume(_session_id: str) -> None:
        nonlocal prepare_calls
        prepare_calls += 1

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        replacements.append((session_id, force))

    service = management_service(
        home,
        current_agent_loop=unavailable_loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        prepare_session_resume=prepare_session_resume,
        memory_manager=memory,
        dream=dream,
        schedule_status=schedule_status,
    )
    unavailable_id = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"

    operations = (
        service.status,
        service.memory_view,
        service.dream,
        service.resumable_listing,
        lambda: service.resume(unavailable_id),
    )
    for operation in operations:
        with pytest.raises(ManagementError) as raised:
            await operation()
        assert (raised.value.error.code, raised.value.error.message) == (
            "route_unavailable",
            "Runtime Generation is unavailable.",
        )

    assert current_calls == 5
    assert schedule_calls == 0
    assert prepare_calls == 0
    assert memory.calls == 0
    assert dream.calls == 0
    assert replacements == []
