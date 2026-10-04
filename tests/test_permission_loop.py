from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from omni.agent.memory.manager import MemoryManager
from omni.agent.message_bus import MessageBus
from omni.agent.permission import RuntimePermissionControl
from omni.agent.tools.core.exec_host import create_exec_host, resolve_exec_shell
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.agent.workspace_state import WorkspaceState
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.provider.model_router import ModelRouter
from omni.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from omni.schedule.service import ScheduleService
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ScriptedFakeProvider,
    StreamScript,
    collect_foreground_outbound,
)
from tests.fixtures.agent_loop import DrivenExecutor as AgentRunExecutor
from tests.fixtures.agent_loop import loaded_skill_loader

NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=timezone(timedelta(hours=8)))


def _agent_loop(
    home: AgentHome,
    workspace: Path,
    provider: ScriptedFakeProvider,
) -> tuple[AgentRunExecutor, ModelRouter, ScheduleService, MessageBus]:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    configuration = ConfigLoader(home).load()
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
    )
    loop: AgentRunExecutor | None = None

    async def execute_user_job(job: object) -> None:
        assert loop is not None
        await loop.run_schedule_job(job)  # type: ignore[arg-type]

    async def execute_dream() -> None:
        return None

    schedule = ScheduleService(
        workspace_state=state,
        clock=FakeClock(NOW),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    bus = MessageBus()
    loop = AgentRunExecutor(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=home,
        configuration=configuration,
        bus=bus,
        schedule_service=schedule,
        model_router=router,
        memory_manager=MemoryManager(state),
        session_id=None,
        now=lambda: NOW,
        new_uuid=uuid4,
        monotonic_now=lambda: 0.0,
        exec_host=create_exec_host(resolve_exec_shell(configuration.runtime.exec_shell)),
        permission_control=RuntimePermissionControl(configuration.runtime.permission_level),
        skill_loader=loaded_skill_loader(home, configuration),
    )
    return loop, router, schedule, bus


@pytest.mark.asyncio
async def test_foreground_mutations_execute_without_a_permission_pause(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    target = workspace / "notes.txt"
    target.write_text("before", encoding="utf-8")
    calls = (
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments='{"path":"created.txt","content":"must not be written"}',
        ),
        ModelToolCall(
            id="call_edit",
            name="edit_file",
            arguments='{"path":"notes.txt","old_text":"before","new_text":"after"}',
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(
                                input_tokens=8,
                                output_tokens=2,
                                total_tokens=10,
                            ),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Mutations were refused."),
                            usage=ModelUsage(
                                input_tokens=12,
                                output_tokens=3,
                                total_tokens=15,
                            ),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    loop, router, schedule, bus = _agent_loop(home, workspace, provider)
    confirmations: list[object] = []
    loop.control.bind_confirmation_callback(confirmations.append)
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, "Change the files.")
    finally:
        await loop.close()
        await schedule.close()
        await router.close()

    assert confirmations == []
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "must not be written"
    assert target.read_text(encoding="utf-8") == "after"
    tool_messages = [message for message in loop.session.messages if message["role"] == "tool"]
    assert [message["status"] for message in tool_messages] == ["success", "success"]
    follow_up = provider.stream_requests[1]
    model_results = [message for message in follow_up.messages if message["role"] == "tool"]
    assert [(message["name"], message["content"]) for message in model_results] == [
        ("write_file", "File written successfully."),
        ("edit_file", "File edited successfully."),
    ]
    assert messages[-1].metadata == {"_streamed": True}
