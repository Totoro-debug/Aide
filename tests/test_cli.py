import asyncio
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from uuid import UUID

import pytest
from typer.testing import CliRunner

import myclaw.terminal.cli as cli
from myclaw.agent.loop import ModelContextOverflowError, TerminalAgentLoopControl
from myclaw.agent.message_bus import MessageBus
from myclaw.config.agent_home import AgentHome
from myclaw.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, ErrorInfo
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    FatalManagementError,
    ManagementError,
    ManagementViewService,
    ResumeResult,
)
from myclaw.permission.state import RuntimePermissionControl
from myclaw.provider.model_router import ModelRouteStatus
from myclaw.session.session import Session
from myclaw.skills.catalog import SkillMetadata
from myclaw.terminal.conversation import TerminalConversationApp
from myclaw.tools.mcp.runtime import MCPRuntimeManager
from myclaw.utils.time import local_now
from myclaw.workspace.state import WorkspaceState, WorkspaceStateError
from tests.configuration.test_config import (
    EXPECTED_DEFAULT_CONFIG,
    EXPECTED_REDACTED_CONFIG,
    EXPECTED_REDACTED_MALFORMED_CONFIG,
    MALFORMED_CONFIG,
    MINIMAL_VALID_CONFIG,
    REDACTION_CONFIG,
    VALID_CONFIG,
)
from tests.fixtures.session import seed_session_state


def _fake_memory_route_status() -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="test-provider",
        model="test-model",
        context_window=200_000,
        max_output=8_192,
        used_default=False,
    )


def test_legacy_runtime_module_is_not_discoverable() -> None:
    legacy_module = ".".join(("myclaw", "agent", "runtime"))
    assert not (Path(__file__).resolve().parents[1] / "myclaw" / "agent" / "runtime.py").exists()
    assert importlib.util.find_spec(legacy_module) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_module)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("close_fails", "terminal_fails"),
    [(False, False), (True, False), (True, True)],
    ids=["normal", "close-error", "primary-and-close-error"],
)
async def test_cli_async_root_owns_lifetime_components_and_async_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    close_fails: bool,
    terminal_fails: bool,
) -> None:
    events: list[str] = []
    close_error = RuntimeError("loop close failed")
    primary_error = RuntimeError("terminal failed")

    class FakeConfirmationCoordinator:
        def __init__(self) -> None:
            events.append("confirmation_init")

        async def request(self, request: object) -> Literal["approved", "declined"]:
            del request
            raise AssertionError("No confirmation is expected in this lifecycle test")

        async def close(self) -> None:
            events.append("confirmation_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path
            events.append("workspace_init")

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root
            events.append("workspace_initialize")

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            del workspace, kwargs
            events.append("mcp_init")

        async def start(self, configuration: object) -> object:
            del configuration
            events.append("mcp_start")
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeMessageBus:
        def __init__(self) -> None:
            events.append("bus_init")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("router_init")

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state
            events.append("memory_init")

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("dream_init")

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("schedule_init")

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs
            events.append("dream_register")

        def _prepare_start(self) -> None:
            events.append("schedule_preflight")

        def start(self) -> None:
            events.append("schedule_start")

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def drain_confirmation_aborts(self, *, generation_id: object | None = None) -> None:
            assert generation_id is None
            events.append("schedule_confirmation_drain")

        async def close(self) -> None:
            events.append("schedule_close")

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("loop_init")
            self.control = object()
            self.skill_metadata = ()

        def bind_confirmation_requester(self, requester: Callable[..., Awaitable[object]]) -> None:
            assert callable(requester)
            events.append("loop_bind_confirmation")

        def preflight(self) -> None:
            events.append("loop_preflight")

        async def start(self) -> None:
            events.append("loop_start")

        async def close(self) -> None:
            events.append("loop_close")
            if close_fails:
                raise close_error

        async def abort(self) -> None:
            events.append("loop_abort")

        def project_foreground_conversation(self) -> object:
            return object()

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("management_init")

        def deactivate(self) -> None:
            events.append("management_deactivate")

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management
            events.append("dispatcher_init")

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("app_init")

        def bind_confirmation_coordinator(self, coordinator: object) -> None:
            assert isinstance(coordinator, FakeConfirmationCoordinator)
            events.append("app_bind_confirmation")

        async def run_async(self) -> None:
            events.append("app_run")
            try:
                if terminal_fails:
                    raise primary_error
            finally:
                events.append("terminal_restore")

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "MessageBus", FakeMessageBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)
    monkeypatch.setattr(cli, "ToolConfirmationCoordinator", FakeConfirmationCoordinator)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "async def _run_cli_conversation" in source

    if close_fails:
        with pytest.raises(RuntimeError) as raised:
            await cli._run_cli_conversation(
                agent_home=home,
                workspace=tmp_path / "workspace",
                configuration=configuration,
            )
        if terminal_fails:
            assert raised.value is primary_error
            assert raised.value.__cause__ is close_error
        else:
            assert raised.value is close_error
            assert raised.value.__cause__ is None
    else:
        await cli._run_cli_conversation(
            agent_home=home,
            workspace=tmp_path / "workspace",
            configuration=configuration,
        )

    assert events == [
        "confirmation_init",
        "workspace_init",
        "workspace_initialize",
        "mcp_init",
        "mcp_start",
        "bus_init",
        "router_init",
        "memory_init",
        "dream_init",
        "schedule_init",
        "loop_init",
        "loop_bind_confirmation",
        "loop_preflight",
        "schedule_preflight",
        "dream_register",
        "management_init",
        "dispatcher_init",
        "app_init",
        "app_bind_confirmation",
        "loop_start",
        "schedule_start",
        "app_run",
        "terminal_restore",
        "confirmation_close",
        "schedule_confirmation_drain",
        "management_deactivate",
        "schedule_pause",
        "schedule_close",
        "loop_close",
        "mcp_close",
        "dream_close",
        "router_close",
    ]


@pytest.mark.asyncio
async def test_cli_async_root_cleans_partial_startup_without_registering_dream_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root
            events.append("workspace_initialize")

    class FakeBus:
        def __init__(self) -> None:
            events.append("bus_init")

        async def reset(self) -> None:
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("router_init")

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state
            events.append("memory_init")

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("dream_init")

        async def run(self) -> object:
            raise AssertionError("Dream must not run during failed startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("schedule_init")

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs
            events.append("dream_register")

        def _prepare_start(self) -> None:
            events.append("schedule_preflight")

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            events.append("schedule_close")

    class FailingAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("loop_init")
            self.control = object()
            self.skill_metadata = ()

        def preflight(self) -> None:
            events.append("loop_preflight")
            raise RuntimeError("preflight contained a secret")

        async def abort(self) -> None:
            events.append("loop_abort")

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FailingAgentLoop)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    with pytest.raises(RuntimeError, match="preflight contained a secret"):
        await cli._run_cli_conversation(
            agent_home=home,
            workspace=tmp_path / "workspace",
            configuration=configuration,
        )

    assert events == [
        "workspace_initialize",
        "bus_init",
        "router_init",
        "memory_init",
        "dream_init",
        "schedule_init",
        "loop_init",
        "loop_preflight",
        "schedule_pause",
        "schedule_close",
        "loop_abort",
        "dream_close",
        "router_close",
    ]


@dataclass(slots=True)
class _FatalResumeProbe:
    result: Any
    events: list[str]
    app: Any
    loops: list[Any]
    current_callback: Callable[[], object]
    bus: Any
    schedule: Any
    router: Any
    dream: Any
    dispatcher: ManagementCommandDispatcher
    target_id: str
    secret: str


def _invoke_cli_resume_preparation_failure(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_kind: Literal["constructor", "preflight"],
) -> _FatalResumeProbe:
    events: list[str] = []
    loops: list[Any] = []
    current_callback: Callable[[], object] | None = None
    app_instance: Any = None
    bus_instance: Any = None
    schedule_instance: Any = None
    router_instance: Any = None
    dream_instance: Any = None
    dispatcher_instance: ManagementCommandDispatcher | None = None
    persist_wait_count = 0
    persist_gate_started = asyncio.Event()
    release_persist_gate = asyncio.Event()
    target_secret = (
        "sk-target-secret C:\\sensitive\\skill\\SKILL.md Skill body: never print these instructions"
    )
    safe_preflight_error = ErrorInfo("model_context_overflow", MODEL_CONTEXT_OVERFLOW_MESSAGE)

    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    target_session = Session.create(
        state,
        now=local_now,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    seed_session_state(
        target_session,
        messages=[
            {
                "role": "user",
                "content": "Target session",
                "timestamp": target_session.created_at.isoformat(timespec="milliseconds"),
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
    target_session.close()
    target_id = target_session.session_id

    class FakeBus:
        def __init__(self) -> None:
            nonlocal bus_instance
            self.reset_calls = 0
            self.inbound_callback: Callable[[tuple[object, ...]], None] | None = None
            self.outbound_blocker = asyncio.Event()
            bus_instance = self

        async def inbound_snapshot(self) -> tuple[object, ...]:
            return ()

        def set_inbound_changed_callback(
            self,
            callback: Callable[[tuple[object, ...]], None],
        ) -> None:
            self.inbound_callback = callback

        def unbind_inbound_changed_callback(
            self,
            callback: Callable[[tuple[object, ...]], None],
        ) -> None:
            if self.inbound_callback is callback:
                self.inbound_callback = None

        async def get_outbound(self) -> object:
            await self.outbound_blocker.wait()
            raise AssertionError("Fake outbound blocker must be cancelled")

        async def reset(self) -> None:
            self.reset_calls += 1
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            nonlocal router_instance
            del kwargs
            self.close_calls = 0
            events.append("router_init")
            router_instance = self

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            self.close_calls += 1
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            nonlocal dream_instance
            del kwargs
            self.close_calls = 0
            dream_instance = self

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            self.close_calls += 1
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            nonlocal schedule_instance
            del kwargs
            self.pause_calls = 0
            self.close_calls = 0
            self.resume_calls = 0
            schedule_instance = self

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            events.append("schedule_preflight")

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            events.append("schedule_start")

        async def pause_and_drain(self) -> None:
            self.pause_calls += 1
            events.append("schedule_pause")

        def resume(self) -> None:
            self.resume_calls += 1
            events.append("schedule_resume")

        async def close(self) -> None:
            self.close_calls += 1
            events.append("schedule_close")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeControl:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            self.confirmation_callback: Callable[[object], None] | None = None

        @property
        def has_active_run(self) -> bool:
            return False

        async def cancel_active_run(self) -> None:
            raise AssertionError("No active run may be cancelled")

        def bind_confirmation_callback(self, callback: Callable[[object], None]) -> None:
            self.confirmation_callback = callback

        def unbind_confirmation_callback(self, callback: Callable[[object], None]) -> None:
            if self.confirmation_callback is callback:
                self.confirmation_callback = None

        def respond_to_confirmation(self, *args: object) -> None:
            raise AssertionError(f"Unexpected confirmation response: {args!r}")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session_id, messages=())

    class FakeSession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

        async def wait_for_pending_persist(self) -> None:
            nonlocal persist_wait_count
            persist_wait_count += 1
            events.append(f"old_persist_wait:{persist_wait_count}")
            if persist_wait_count == 2:
                persist_gate_started.set()
                await release_persist_gate.wait()
                events.append("old_persist_gate_released")

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            session_id = cast(str | None, kwargs["session_id"])
            self.is_target = session_id is not None
            self.permission_control = cast(
                RuntimePermissionControl,
                kwargs["permission_control"],
            )
            self.session_id = target_id
            self.control = FakeControl(target_id)
            self.skill_metadata = ()
            self.session = FakeSession(self.session_id)
            self.close_calls = 0
            self.abort_calls = 0
            self.replacement_barrier_held = False
            loops.append(self)
            if not self.is_target:
                self.permission_control.select("read-only")
                events.append("old_init")
            else:
                assert self.permission_control is loops[0].permission_control
                assert self.permission_control.current() == "read-only"
                events.append("target_init")
                if failure_kind == "constructor":
                    raise ModelContextOverflowError(
                        ErrorInfo("model_context_overflow", target_secret)
                    )

        def preflight(self) -> None:
            if not self.is_target:
                events.append("old_preflight")
                return
            events.append("target_preflight")
            if failure_kind == "preflight":
                try:
                    raise RuntimeError(target_secret)
                except RuntimeError as cause:
                    raise ModelContextOverflowError(safe_preflight_error) from cause

        async def start(self) -> None:
            events.append("target_start" if self.is_target else "old_start")

        async def close(self) -> None:
            self.close_calls += 1
            if not self.is_target:
                assert self.replacement_barrier_held is True
            events.append("target_close" if self.is_target else "old_close")

        async def abort(self) -> None:
            self.abort_calls += 1
            events.append("target_abort" if self.is_target else "old_abort")

        async def _pause_for_replacement(self) -> None:
            assert self.replacement_barrier_held is False
            self.replacement_barrier_held = True
            events.append("old_quiesce")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            self.replacement_barrier_held = False
            events.append(f"old_barrier_release:{resume_inbound}")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session_id, messages=())

    def recording_management_service(
        agent_home: AgentHome,
        **kwargs: Any,
    ) -> ManagementViewService:
        nonlocal current_callback
        current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
        return ManagementViewService(agent_home, **kwargs)

    class FatalResumeApp(TerminalConversationApp):
        def __init__(
            self,
            *,
            bus: MessageBus,
            control: TerminalAgentLoopControl,
            management_dispatcher: ManagementCommandDispatcher,
            skill_metadata: tuple[SkillMetadata, ...] = (),
        ) -> None:
            nonlocal app_instance, dispatcher_instance
            super().__init__(
                bus=bus,
                control=control,
                management_dispatcher=management_dispatcher,
                skill_metadata=skill_metadata,
            )
            self.exit_calls = 0
            self.remaining_task_count = -1
            self.resume_task_retrieved = False
            self.loop_error_contexts: list[dict[str, object]] = []
            self.ui_stopped = False
            app_instance = self
            dispatcher_instance = management_dispatcher

        def exit(
            self,
            result: None = None,
            return_code: int = 0,
            message: Any = None,
        ) -> None:
            self.exit_calls += 1
            events.append(f"terminal_exit:{return_code}")
            super().exit(result=result, return_code=return_code, message=message)

        async def _mount_management_rows(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("management_render")

        async def quiesce_for_rebind(self) -> None:
            events.append("presentation_quiesce")
            await super().quiesce_for_rebind()

        async def rebind_agent_loop(self, **kwargs: Any) -> None:
            events.append("presentation_rebind")
            await super().rebind_agent_loop(**kwargs)

        async def run_async(self, **options: Any) -> None:
            del options
            loop = asyncio.get_running_loop()
            previous_exception_handler = loop.get_exception_handler()

            def capture_loop_error(
                active_loop: asyncio.AbstractEventLoop,
                context: dict[str, Any],
            ) -> None:
                del active_loop
                self.loop_error_contexts.append(dict(context))

            loop.set_exception_handler(capture_loop_error)
            try:
                async with self.run_test(size=(80, 24)):
                    input_area = cast(Any, self.query_one("#conversation-input"))
                    input_area.read_only = True
                    resume_task = asyncio.create_task(
                        self._resume_selected_session(target_id, cast(Any, input_area)),
                        name="fatal-resume-test-driver",
                    )
                    try:
                        await asyncio.wait_for(persist_gate_started.wait(), timeout=1)
                        assert loops[0].replacement_barrier_held is True
                        assert "target_init" not in events
                        events.append("persist_gate_observed")
                        release_persist_gate.set()
                        await resume_task
                        resume_task.result()
                        self.resume_task_retrieved = True
                    finally:
                        if not resume_task.done():
                            resume_task.cancel()
                        await asyncio.gather(resume_task, return_exceptions=True)
                self.ui_stopped = (
                    self._closing
                    and self._presentation_quiesced
                    and self._outbound_worker is None
                    and self._resume_worker is None
                    and self._bus_callback is None
                )
                current = asyncio.current_task()
                self.remaining_task_count = sum(
                    1 for task in asyncio.all_tasks() if task is not current and not task.done()
                )
            finally:
                loop.set_exception_handler(previous_exception_handler)
            assert current_callback is not None
            assert current_callback() is loops[0]
            events.append("terminal_run_return")

    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", recording_management_service)
    monkeypatch.setattr(cli, "TerminalConversationApp", FatalResumeApp)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(cli.app, [])
    assert app_instance is not None
    assert current_callback is not None
    assert bus_instance is not None
    assert schedule_instance is not None
    assert router_instance is not None
    assert dream_instance is not None
    assert dispatcher_instance is not None
    return _FatalResumeProbe(
        result=result,
        events=events,
        app=app_instance,
        loops=loops,
        current_callback=current_callback,
        bus=bus_instance,
        schedule=schedule_instance,
        router=router_instance,
        dream=dream_instance,
        dispatcher=dispatcher_instance,
        target_id=target_id,
        secret=target_secret,
    )


def _assert_fatal_resume_preparation(
    probe: _FatalResumeProbe,
    *,
    safe_error: str,
    target_abort_count: int,
) -> None:
    result = probe.result
    events = probe.events
    app = probe.app
    old = cast(Any, probe.loops[0])
    target = cast(Any, probe.loops[1])

    assert result.exit_code == 1
    assert result.output.count(safe_error) == 1
    assert probe.secret not in result.output
    assert "Traceback" not in result.output
    assert app.exit_calls == 1
    assert app.fatal_management_error is not None
    assert app.ui_stopped is True
    assert app.resume_task_retrieved is True
    assert app.remaining_task_count == 0
    assert app.loop_error_contexts == []
    assert events.index("old_quiesce") < events.index("old_persist_wait:2")
    assert events.index("old_persist_wait:2") < events.index("persist_gate_observed")
    assert events.index("persist_gate_observed") < events.index("old_persist_gate_released")
    assert events.index("old_persist_gate_released") < events.index("target_init")
    assert events.index("terminal_exit:1") < events.index("schedule_pause")
    assert events.index("schedule_pause") < events.index("old_close")
    assert events[: events.index("terminal_exit:1")].count("schedule_pause") == 0
    assert "presentation_quiesce" not in events
    assert "presentation_rebind" not in events
    assert "target_start" not in events
    assert "old_abort" not in events
    assert "old_barrier_release:True" not in events
    assert "bus_reset" not in events
    assert "management_render" not in events
    assert old.replacement_barrier_held is True
    assert old.close_calls == 1
    assert old.abort_calls == 0
    assert target.close_calls == 0
    assert target.abort_calls == target_abort_count
    assert probe.bus.reset_calls == 0
    assert probe.bus.inbound_callback is None
    assert probe.schedule.pause_calls == 1
    assert probe.schedule.resume_calls == 0
    assert probe.schedule.close_calls == 1
    assert probe.dream.close_calls == 1
    assert probe.router.close_calls == 1
    assert (
        max(
            old.close_calls,
            target.close_calls,
            probe.schedule.close_calls,
            probe.dream.close_calls,
            probe.router.close_calls,
        )
        <= 1
    )
    assert old.control.confirmation_callback is None
    assert target.permission_control is old.permission_control
    assert old.permission_control.current() == "read-only"
    assert probe.current_callback() is old
    unavailable = asyncio.run(probe.dispatcher.resume(probe.target_id))
    assert unavailable.output == "route_unavailable: Runtime Generation is no longer active."


def test_cli_resume_constructor_failure_terminates_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _invoke_cli_resume_preparation_failure(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        failure_kind="constructor",
    )

    _assert_fatal_resume_preparation(
        probe,
        safe_error="persistence_error: Conversation Session could not be prepared.",
        target_abort_count=0,
    )
    assert "target_preflight" not in probe.events


def test_cli_resume_preflight_failure_terminates_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _invoke_cli_resume_preparation_failure(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        failure_kind="preflight",
    )

    _assert_fatal_resume_preparation(
        probe,
        safe_error=(
            "model_context_overflow: Model request context exceeds the available input budget."
        ),
        target_abort_count=1,
    )
    assert probe.events.index("target_init") < probe.events.index("target_preflight")
    assert probe.events.index("target_preflight") < probe.events.index("target_abort")


@pytest.mark.asyncio
async def test_cli_resume_publishes_current_only_after_target_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Awaitable[None]] | None = None
    initial_loop: object | None = None
    target_loop: object | None = None
    first_target: object | None = None
    pause_count = 0
    user_executor: Callable[[object], Awaitable[None]] | None = None
    bus_instances: list[object] = []
    router_instances: list[object] = []
    memory_instances: list[object] = []
    dream_instances: list[object] = []
    schedule_instances: list[object] = []
    loop_instances: list[object] = []
    management_instances: list[object] = []
    dispatcher_instances: list[object] = []
    app_instances: list[object] = []
    cancelled_generations: list[UUID] = []
    coordinator_close_calls = 0

    class FakeConfirmationCoordinator:
        async def request(self, request: object) -> Literal["approved", "declined"]:
            del request
            raise AssertionError("No confirmation is expected in this replacement test")

        async def cancel_generation(self, generation_id: UUID) -> None:
            cancelled_generations.append(generation_id)
            events.append("coordinator_cancel_generation")

        async def close(self) -> None:
            nonlocal coordinator_close_calls
            coordinator_close_calls += 1
            events.append("coordinator_close")

    def current_value() -> object:
        assert current_callback is not None
        assert callable(current_callback)
        return current_callback()

    def assert_current(expected: object) -> None:
        assert current_value() is expected

    def assert_unavailable() -> None:
        with pytest.raises(ManagementError, match="Runtime Generation is unavailable"):
            current_value()

    class FakeSession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

        async def wait_for_pending_persist(self) -> None:
            events.append("wait_for_persist")

    class FakeControl:
        has_active_run = False

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        def __init__(self) -> None:
            bus_instances.append(self)

        async def reset(self) -> None:
            assert_unavailable()
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            router_instances.append(self)

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state
            memory_instances.append(self)

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            dream_instances.append(self)

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            nonlocal user_executor
            user_executor = cast(
                Callable[[object], Awaitable[None]],
                kwargs["execute_user_job"],
            )
            schedule_instances.append(self)

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            return None

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            events.append("schedule_start")

        def cancel_confirmation_generation(self, generation_id: UUID) -> None:
            events.append(f"schedule_cancel_generation:{generation_id.int}")

        async def drain_confirmation_aborts(
            self,
            *,
            generation_id: object | None = None,
        ) -> None:
            if isinstance(generation_id, UUID):
                events.append(f"schedule_confirmation_drain:{generation_id.int}")
            else:
                events.append("schedule_confirmation_drain:all")

        async def pause_and_drain(self) -> None:
            nonlocal pause_count
            pause_count += 1
            if pause_count == 1:
                assert_current(initial_loop)
            elif pause_count == 2:
                assert_current(first_target)
            events.append("schedule_pause")

        def resume(self) -> None:
            assert_current(target_loop)
            events.append("schedule_resume")

        async def close(self) -> None:
            events.append("schedule_close")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal initial_loop, target_loop
            loop_instances.append(self)
            session_id = kwargs["session_id"]
            self.session = FakeSession("initial" if session_id is None else str(session_id))
            self.control = FakeControl()
            self.skill_metadata = ()
            self.generation_id = UUID(int=len(loop_instances))
            if session_id is None:
                initial_loop = self
                events.append("old_init")
            else:
                target_loop = self
                events.append("target_init")

        def bind_confirmation_requester(self, requester: Callable[..., Awaitable[object]]) -> None:
            assert callable(requester)

        def preflight(self) -> None:
            events.append("target_preflight" if self is target_loop else "old_preflight")

        async def start(self) -> None:
            if self is target_loop:
                assert_unavailable()
                events.append("target_start")
            else:
                events.append("old_start")

        async def close(self) -> None:
            events.append("target_close" if self is target_loop else "old_close")

        async def abort(self) -> None:
            if self is initial_loop:
                assert_unavailable()
                events.append("old_abort")
            else:
                events.append("target_abort")

        async def _pause_for_replacement(self) -> None:
            return None

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            del resume_inbound

        async def run_schedule_job(self, job: object) -> None:
            del job
            events.append("target_schedule_job" if self is target_loop else "old_schedule_job")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal current_callback, replace_callback
            del args
            management_instances.append(self)
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(
                Callable[[str, bool], Awaitable[None]],
                kwargs["replace_agent_loop"],
            )

        def deactivate(self) -> None:
            return None

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management
            dispatcher_instances.append(self)

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            app_instances.append(self)

        async def run_async(self) -> None:
            nonlocal first_target
            assert user_executor is not None
            assert callable(user_executor)
            await user_executor(object())
            assert_current(initial_loop)
            events.append("before_current")
            assert replace_callback is not None
            await replace_callback("target", False)
            assert_current(target_loop)
            events.append("after_current")
            await user_executor(object())
            first_target = target_loop
            await replace_callback("second-target", False)
            assert current_value() is target_loop
            await user_executor(object())

        async def quiesce_for_rebind(self) -> None:
            if first_target is None:
                assert_current(initial_loop)
            else:
                assert_current(first_target)
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            assert_unavailable()
            events.append("rebind")

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)
    monkeypatch.setattr(cli, "ToolConfirmationCoordinator", FakeConfirmationCoordinator)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    await cli._run_cli_conversation(
        agent_home=home,
        workspace=tmp_path / "workspace",
        configuration=configuration,
    )

    assert events[:33] == [
        "old_init",
        "old_preflight",
        "old_start",
        "schedule_start",
        "old_schedule_job",
        "before_current",
        "target_init",
        "target_preflight",
        "schedule_cancel_generation:1",
        "coordinator_cancel_generation",
        "schedule_confirmation_drain:1",
        "quiesce",
        "schedule_pause",
        "old_abort",
        "bus_reset",
        "rebind",
        "target_start",
        "schedule_resume",
        "after_current",
        "target_schedule_job",
        "target_init",
        "target_preflight",
        "schedule_cancel_generation:2",
        "coordinator_cancel_generation",
        "schedule_confirmation_drain:2",
        "quiesce",
        "schedule_pause",
        "target_abort",
        "bus_reset",
        "rebind",
        "target_start",
        "schedule_resume",
        "target_schedule_job",
    ]
    assert events[33:] == [
        "coordinator_close",
        "schedule_confirmation_drain:all",
        "schedule_pause",
        "schedule_close",
        "target_close",
        "dream_close",
        "router_close",
    ]
    assert events.count("old_init") == 1
    assert events.count("target_init") == 2
    assert len(bus_instances) == 1
    assert len(router_instances) == 1
    assert len(memory_instances) == 1
    assert len(dream_instances) == 1
    assert len(schedule_instances) == 1
    assert len(loop_instances) == 3
    assert len(management_instances) == 1
    assert len(dispatcher_instances) == 1
    assert len(app_instances) == 1
    assert cancelled_generations == [
        cast(Any, loop_instances[0]).generation_id,
        cast(Any, loop_instances[1]).generation_id,
    ]
    assert coordinator_close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_outcome", ["success", "error", "cancel"])
async def test_cli_resume_active_requires_force_before_replacing_the_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    abort_outcome: Literal["success", "error", "cancel"],
) -> None:
    events: list[str] = []
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Awaitable[None]] | None = None
    initial_loop: object | None = None
    target_loop: object | None = None
    user_executor: Callable[[object], Awaitable[None]] | None = None
    management_errors: list[ManagementError] = []
    abort_entered = asyncio.Event()
    release_abort = asyncio.Event()
    secret = "private-candidate-abort-detail"
    abort_error = RuntimeError(secret)
    target_id = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"
    rejection_event = "replacement_cancelled" if abort_outcome == "cancel" else "management_error"

    class FakeSession:
        session_id = "old-session"

        async def wait_for_pending_persist(self) -> None:
            events.append("wait_for_persist")

    class FakeControl:
        has_active_run = True

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        async def reset(self) -> None:
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            nonlocal user_executor
            user_executor = cast(
                Callable[[object], Awaitable[None]],
                kwargs["execute_user_job"],
            )

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            return None

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        def resume(self) -> None:
            events.append("schedule_resume")

        async def close(self) -> None:
            events.append("schedule_close")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal initial_loop, target_loop
            loop_instances.append(self)
            session_id = kwargs["session_id"]
            self.session = FakeSession()
            self.control = (
                FakeControl() if session_id is None else SimpleNamespace(has_active_run=False)
            )
            self.skill_metadata = ()
            self.start_calls = 0
            self.close_calls = 0
            self.abort_calls = 0
            self.replacement_barrier_held = False
            if session_id is None:
                initial_loop = self
                events.append("old_init")
            else:
                target_loop = self
                events.append("target_init")

        def preflight(self) -> None:
            events.append("old_preflight" if self is initial_loop else "target_preflight")

        async def start(self) -> None:
            self.start_calls += 1
            events.append("old_start" if self is initial_loop else "target_start")

        async def close(self) -> None:
            self.close_calls += 1
            events.append("old_close" if self is initial_loop else "target_close")

        async def abort(self) -> None:
            self.abort_calls += 1
            events.append("target_abort" if self is not initial_loop else "old_abort")
            if self is loop_instances[1]:
                if abort_outcome == "error":
                    raise abort_error
                if abort_outcome == "cancel":
                    abort_entered.set()
                    await release_abort.wait()

        async def _pause_for_replacement(self) -> None:
            assert not self.replacement_barrier_held
            self.replacement_barrier_held = True
            events.append("replacement_barrier_pause")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            assert self.replacement_barrier_held
            self.replacement_barrier_held = False
            events.append(f"replacement_barrier_release:{resume_inbound}")

        async def run_schedule_job(self, job: object) -> None:
            del job
            assert current_callback is not None
            assert current_callback() is self
            assert not self.replacement_barrier_held
            assert self.abort_calls == self.close_calls == 0
            events.append("old_schedule_job" if self is initial_loop else "target_schedule_job")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=target_id, messages=())

    loop_instances: list[FakeAgentLoop] = []

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal current_callback, replace_callback
            del args
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(
                Callable[[str, bool], Awaitable[None]],
                kwargs["replace_agent_loop"],
            )

        def deactivate(self) -> None:
            return None

        async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult:
            assert replace_callback is not None
            try:
                await replace_callback(session_id, force)
            except ManagementError as error:
                management_errors.append(error)
                raise
            return ResumeResult(session_id=session_id)

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            self.dispatcher = cast(ManagementCommandDispatcher, kwargs["management_dispatcher"])

        async def run_async(self) -> None:
            if abort_outcome == "cancel":
                replacement = asyncio.create_task(self.dispatcher.resume(target_id))
                try:
                    await asyncio.wait_for(abort_entered.wait(), timeout=1)
                    assert not replacement.done()
                    replacement.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(replacement, timeout=1)
                    assert replacement.cancelled()
                finally:
                    if not replacement.done():
                        replacement.cancel()
                    await asyncio.wait_for(
                        asyncio.gather(replacement, return_exceptions=True), timeout=1
                    )
                assert management_errors == []
            else:
                result = await self.dispatcher.resume(target_id)
                assert result.handled
                assert result.resumed_session_id is None
                assert len(management_errors) == 1
                error = management_errors[0]
                if abort_outcome == "error":
                    assert error.error.code == "persistence_error"
                    assert error.error.message == "Conversation Session could not be prepared."
                    assert error.__cause__ is abort_error
                    assert result.output == (
                        "persistence_error: Conversation Session could not be prepared."
                    )
                else:
                    assert error.error.code == "model_invalid_request"
                    assert result.output == (
                        "model_invalid_request: An active foreground run must be confirmed "
                        "before switching Sessions."
                    )
                assert result.output is not None
                assert secret not in result.output
            events.append(rejection_event)
            assert current_callback is not None
            assert current_callback() is initial_loop
            old, rejected = loop_instances
            assert not old.replacement_barrier_held
            assert events.count("replacement_barrier_release:True") == 1
            assert "replacement_barrier_release:False" not in events
            assert old.abort_calls == old.close_calls == 0
            assert rejected.abort_calls == 1
            assert rejected.start_calls == rejected.close_calls == 0
            for destructive_event in (
                "quiesce",
                "schedule_pause",
                "old_abort",
                "bus_reset",
                "rebind",
                "target_start",
            ):
                assert destructive_event not in events
            assert user_executor is not None
            await user_executor(object())
            result = await self.dispatcher.resume(target_id, force=True)
            assert result.resumed_session_id == target_id
            assert current_callback() is target_loop
            assert target_loop is not rejected
            await user_executor(object())

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    await cli._run_cli_conversation(
        agent_home=home,
        workspace=tmp_path / "workspace",
        configuration=configuration,
    )

    assert events.index("target_preflight") < events.index("target_abort")
    assert events.index("target_abort") < events.index(rejection_event)
    assert events.index(rejection_event) < events.index("old_schedule_job")
    assert events.index("old_schedule_job") < events.index("old_abort")
    assert events.index("old_abort") < events.index("bus_reset")
    assert events.index("bus_reset") < events.index("rebind")
    assert events.index("rebind") < events.index("target_start")
    assert events.index("target_start") < events.index("schedule_resume")
    assert events.count("target_init") == 2
    assert events.count("target_abort") == 1
    assert events.count("old_abort") == 1
    assert events.count("replacement_barrier_pause") == 2
    assert events.count("replacement_barrier_release:True") == 2
    assert events.count("old_schedule_job") == events.count("target_schedule_job") == 1
    old, rejected, replacement_loop = loop_instances
    assert old.abort_calls == 1
    assert old.close_calls == 0
    assert rejected.abort_calls == 1
    assert rejected.start_calls == rejected.close_calls == 0
    assert replacement_loop.start_calls == replacement_loop.close_calls == 1
    assert replacement_loop.abort_calls == 0


@pytest.mark.asyncio
async def test_cli_same_session_resume_waits_for_pending_persist_before_target_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    persist_started = asyncio.Event()
    release_persist = asyncio.Event()
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Awaitable[None]] | None = None
    initial_loop: object | None = None
    target_loop: object | None = None

    class FakeSession:
        session_id = "same-session"

        async def wait_for_pending_persist(self) -> None:
            events.append("persist_wait_started")
            persist_started.set()
            await release_persist.wait()
            events.append("persist_wait_finished")

    class FakeControl:
        has_active_run = False

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        async def reset(self) -> None:
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            return None

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            return None

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            return None

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        async def pause_and_drain(self) -> None:
            return None

        def resume(self) -> None:
            events.append("schedule_resume")

        async def close(self) -> None:
            return None

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal initial_loop, target_loop
            session_id = kwargs["session_id"]
            if session_id is not None:
                assert "persist_wait_finished" in events
                target_loop = self
                events.append("target_init")
            else:
                initial_loop = self
                events.append("old_init")
            self.session = FakeSession()
            self.control = FakeControl()
            self.skill_metadata = ()

        def preflight(self) -> None:
            events.append("target_preflight" if self is target_loop else "old_preflight")

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def abort(self) -> None:
            events.append("target_abort" if self is target_loop else "old_abort")

        async def _pause_for_replacement(self) -> None:
            events.append("replacement_barrier_pause")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            events.append(f"replacement_barrier_release:{resume_inbound}")

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal current_callback, replace_callback
            del args
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(
                Callable[[str, bool], Awaitable[None]],
                kwargs["replace_agent_loop"],
            )

        def deactivate(self) -> None:
            return None

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            replacement = asyncio.ensure_future(replace_callback("same-session", False))
            await asyncio.wait_for(persist_started.wait(), timeout=1)
            events.append("late_inbound_attempt")
            assert current_callback is not None
            assert callable(current_callback)
            assert current_callback() is initial_loop
            assert "target_init" not in events
            replacement.cancel()
            with pytest.raises(asyncio.CancelledError):
                await replacement
            assert current_callback() is initial_loop
            assert events.count("replacement_barrier_release:True") == 1
            release_persist.set()
            await replace_callback("same-session", False)
            assert current_callback() is target_loop

        async def quiesce_for_rebind(self) -> None:
            return None

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            return None

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    await cli._run_cli_conversation(
        agent_home=home,
        workspace=tmp_path / "workspace",
        configuration=configuration,
    )

    assert events.index("persist_wait_started") < events.index("late_inbound_attempt")
    assert events.index("late_inbound_attempt") < events.index("replacement_barrier_release:True")
    assert events.index("replacement_barrier_release:True") < events.index("persist_wait_finished")
    assert events.index("persist_wait_finished") < events.index("target_init")
    assert events.count("replacement_barrier_pause") == 2
    assert events.count("target_init") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_point",
    (
        "quiesce",
        "schedule_pause",
        "old_abort",
        "bus_reset",
        "rebind",
        "target_start",
        "mcp_activate",
        "cancel",
    ),
)
async def test_cli_resume_destructive_failure_fails_closed_and_aborts_each_loop_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    events: list[str] = []
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Awaitable[None]] | None = None
    initial_loop: object | None = None
    target_loop: object | None = None
    pause_count = 0

    def current_value() -> object:
        assert current_callback is not None
        assert callable(current_callback)
        return current_callback()

    def assert_unavailable() -> None:
        with pytest.raises(ManagementError, match="Runtime Generation is unavailable"):
            current_value()

    class FakeSession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

        async def wait_for_pending_persist(self) -> None:
            return None

    class FakeControl:
        has_active_run = False

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        async def reset(self) -> None:
            assert_unavailable()
            events.append("bus_reset")
            if failure_point == "bus_reset":
                raise RuntimeError("reset secret C:\\sensitive\\bus")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def route_status(self, route: str) -> ModelRouteStatus:
            assert route == "memory"
            return _fake_memory_route_status()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: FakeWorkspaceState) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            return None

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        async def pause_and_drain(self) -> None:
            nonlocal pause_count
            pause_count += 1
            if pause_count == 1:
                assert_current(initial_loop)
            events.append("schedule_pause")
            if failure_point == "schedule_pause" and pause_count == 1:
                raise RuntimeError("pause secret")

        async def close(self) -> None:
            events.append("schedule_close")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    def assert_current(expected: object) -> None:
        assert current_value() is expected

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal initial_loop, target_loop
            session_id = kwargs["session_id"]
            self.session = FakeSession("old" if session_id is None else str(session_id))
            self.control = FakeControl()
            self.skill_metadata = ()
            if session_id is None:
                initial_loop = self
            else:
                target_loop = self
            events.append("old_init" if session_id is None else "target_init")

        def preflight(self) -> None:
            events.append("old_preflight" if self is initial_loop else "target_preflight")

        async def start(self) -> None:
            events.append("old_start")
            if self is target_loop:
                if failure_point == "cancel":
                    raise asyncio.CancelledError
                if failure_point == "target_start":
                    raise RuntimeError("start secret")

        async def close(self) -> None:
            events.append("old_close" if self is initial_loop else "target_close")

        async def abort(self) -> None:
            if self is initial_loop:
                assert_unavailable()
                events.append("old_abort")
                if failure_point == "old_abort":
                    raise RuntimeError("abort secret")
            else:
                events.append("target_abort")

        async def _pause_for_replacement(self) -> None:
            return None

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            del resume_inbound

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal current_callback, replace_callback
            del args
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(
                Callable[[str, bool], Awaitable[None]],
                kwargs["replace_agent_loop"],
            )

        def deactivate(self) -> None:
            assert_unavailable()
            events.append("management_deactivate")

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            await replace_callback("target", False)

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")
            if failure_point == "quiesce":
                raise RuntimeError("quiesce secret")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")
            if failure_point == "rebind":
                raise RuntimeError("rebind secret")

    if failure_point == "mcp_activate":
        def fail_activate(_manager: object, _report: object) -> None:
            events.append("mcp_activate")
            raise RuntimeError("activation secret")

        monkeypatch.setattr(MCPRuntimeManager, "activate_generation", fail_activate)

    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    home = AgentHome(tmp_path / "agent-home")
    configuration: Any = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
    )

    expected_error = asyncio.CancelledError if failure_point == "cancel" else FatalManagementError
    with pytest.raises(expected_error) as raised:
        await cli._run_cli_conversation(
            agent_home=home,
            workspace=tmp_path / "workspace",
            configuration=configuration,
        )

    if isinstance(raised.value, FatalManagementError):
        assert raised.value.error.code == "persistence_error"
        assert "secret" not in str(raised.value)
    assert current_callback is not None
    assert_unavailable()
    assert events.count("old_abort") == 1
    assert events.count("target_abort") == 1
    assert "management_deactivate" in events
    assert "schedule_resume" not in events


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

        monkeypatch.setattr(cli, "_run_cli_conversation", fail_startup)
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

    monkeypatch.setattr(cli, "_run_cli_conversation", fail_workspace)
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

    monkeypatch.setattr(cli, "_run_cli_conversation", fail_replacement)
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

    monkeypatch.setattr(cli, "_run_cli_conversation", fail_startup)
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


@pytest.mark.skipif(
    os.name == "nt",
    reason="The Windows Python runtime has no termios/pty harness; use the Windows Terminal matrix.",
)
def test_installed_wheel_terminal_conversation_pseudo_terminal_smoke(tmp_path: Path) -> None:
    pty = pytest.importorskip("pty")
    termios = pytest.importorskip("termios")
    import select
    import time

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build_result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )
    assert build_result.returncode == 0, build_result.stderr
    wheels = tuple(wheel_dir.glob("myclaw-*.whl"))
    assert len(wheels) == 1

    venv = tmp_path / "venv"
    venv_result = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert venv_result.returncode == 0, venv_result.stderr
    venv_bin = venv / "bin"
    venv_python = venv_bin / "python"
    install_result = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        capture_output=True,
        check=False,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    agent_home = tmp_path / "home" / ".myclaw"
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HOME"] = str(agent_home.parent)

    master_fd, slave_fd = pty.openpty()
    process: subprocess.Popen[bytes] | None = None
    try:
        original_terminal = termios.tcgetattr(slave_fd)
        process = subprocess.Popen(
            [str(venv_bin / "myclaw")],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=workspace,
            env=environment,
            close_fds=True,
        )
        output = bytearray()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and b"Message MyClaw" not in output:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                output.extend(os.read(master_fd, 4096))
        assert b"Message MyClaw" in output or b"\x1b[?1049h" in output
        assert process.poll() is None

        os.write(master_fd, b"\x03")
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
        assert process.poll() == 0

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)

        terminal_output = bytes(output)
        restoration_pairs = (
            (b"\x1b[?2004h", b"\x1b[?2004l"),
            (b"\x1b[?1000h", b"\x1b[?1000l"),
            (b"\x1b[?1003h", b"\x1b[?1003l"),
            (b"\x1b[?1015h", b"\x1b[?1015l"),
            (b"\x1b[?1006h", b"\x1b[?1006l"),
            (b"\x1b[?1004h", b"\x1b[?1004l"),
            (b"\x1b[?1049h", b"\x1b[?1049l"),
            (b"\x1b[?25l", b"\x1b[?25h"),
            (b"\x1b[>1u", b"\x1b[<u"),
        )
        assert b"\x1b[?1049h" in terminal_output
        for enabled, restored in restoration_pairs:
            if enabled in terminal_output:
                assert restored in terminal_output
                assert terminal_output.rfind(restored) > terminal_output.rfind(enabled)
        assert termios.tcgetattr(slave_fd) == original_terminal
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master_fd)
        os.close(slave_fd)


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
    assert not (workspace / ".myclaw").exists()
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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()


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
    assert not (workspace / ".myclaw").exists()
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
    state_path = workspace / ".myclaw"
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
    state_path = workspace / ".myclaw"
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
