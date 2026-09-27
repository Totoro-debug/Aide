from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import myclaw.terminal.cli as cli
from myclaw.agent.loop import (
    AgentLoop,
    ForegroundConversationProjection,
    TerminalAgentLoopControl,
)
from myclaw.agent.message_bus import InboundMessage, MessageBus
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, UserConfiguration
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import FatalManagementError
from myclaw.permission.state import RuntimePermissionControl
from myclaw.provider.models import AssistantModelMessage, ModelCompleted, ModelResponse, ModelUsage
from myclaw.skills.catalog import SkillMetadata
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    BlockingTaskFramingRouterAdapter,
    ScriptedFakeProvider,
    StreamScript,
    TaskFramingRouterAdapter,
    collect_foreground_outbound,
)


def _configuration(
    agent_home: AgentHome,
    config_text: str = VALID_CONFIG,
) -> UserConfiguration:
    agent_home.initialize()
    (agent_home.path / "config.toml").write_text(config_text, encoding="utf-8")
    return ConfigLoader(agent_home).load()


class _CliContractApp:
    fatal_management_error: FatalManagementError | None = None

    def __init__(
        self,
        *,
        bus: MessageBus,
        control: TerminalAgentLoopControl,
        management_dispatcher: ManagementCommandDispatcher,
        skill_metadata: tuple[SkillMetadata, ...],
    ) -> None:
        self._bus = bus
        self._initial_loop = cast(AgentLoop, control)
        self._control = control
        self._dispatcher = management_dispatcher
        self.initial_skill_metadata = skill_metadata
        self.rebound_skill_metadata: tuple[SkillMetadata, ...] | None = None
        self.events: list[str] = []
        self.target_loop: AgentLoop | None = None

    def before_same_session_resume(self, loop: AgentLoop) -> None:
        del loop

    async def verify_same_session_target(self, loop: AgentLoop) -> None:
        del loop

    async def run_async(self) -> None:
        old = self._initial_loop
        old.session.commit_agent_run(
            [{"role": "user", "content": "Persisted before same-Session replacement."}],
            pending_last_compacted=old.session.last_compacted,
            pending_action_summary=cast(str, old.session.metadata["summary"]),
        )
        await old.session.wait_for_pending_persist()
        old_session_id = old.session.session_id
        shared_bus = self._bus
        self.before_same_session_resume(old)

        result = await self._dispatcher.resume(old_session_id)

        assert result.resumed_session_id == old_session_id
        assert self._bus is shared_bus
        target = cast(AgentLoop, self._control)
        assert target is not old
        assert target.session.session_id == old_session_id
        assert [message["content"] for message in target.session.messages] == [
            "Persisted before same-Session replacement."
        ]
        assert target._consumer_task is not None
        await self.verify_same_session_target(target)
        self.target_loop = target

    async def quiesce_for_rebind(self) -> None:
        self.events.append("quiesce")

    async def rebind_agent_loop(
        self,
        *,
        control: TerminalAgentLoopControl,
        skill_metadata: tuple[SkillMetadata, ...],
        session_projection: ForegroundConversationProjection,
    ) -> None:
        self.events.append("rebind")
        unavailable = await self._dispatcher.dispatch("/status")
        assert unavailable.output == "route_unavailable: Runtime Generation is unavailable."
        target = cast(AgentLoop, control)
        assert target._consumer_task is None
        assert session_projection.session_id == target.session.session_id
        self.rebound_skill_metadata = skill_metadata
        self._control = control


async def _exercise_cli_contract(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    monkeypatch: pytest.MonkeyPatch,
    before_resume: Callable[[AgentLoop], None] | None = None,
    verify_target: Callable[[AgentLoop, MessageBus], Awaitable[None]] | None = None,
) -> tuple[list[str], AgentLoop, tuple[SkillMetadata, ...]]:
    app: _CliContractApp | None = None

    class ContractApp(_CliContractApp):
        def __init__(self, **kwargs: Any) -> None:
            nonlocal app
            super().__init__(**kwargs)
            app = self

        def before_same_session_resume(self, loop: AgentLoop) -> None:
            if before_resume is not None:
                before_resume(loop)

        async def verify_same_session_target(self, loop: AgentLoop) -> None:
            if verify_target is not None:
                await verify_target(loop, self._bus)

    monkeypatch.setattr(cli, "TerminalConversationApp", ContractApp)
    await cli._run_cli_conversation(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
    )

    assert app is not None
    assert app.target_loop is not None
    assert app.rebound_skill_metadata is not None
    return app.events, app.target_loop, app.rebound_skill_metadata


@pytest.mark.asyncio
async def test_cli_same_session_replacement_keeps_public_generation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    current_skill = agent_home.path / "skills" / "current" / "SKILL.md"
    removed_skill = agent_home.path / "skills" / "removed" / "SKILL.md"
    current_skill.parent.mkdir(parents=True)
    removed_skill.parent.mkdir(parents=True)
    current_skill.write_bytes(
        b"---\nname: current\ndescription: Original\nalways: true\n---\nOriginal body\n"
    )
    removed_skill.write_bytes(b"---\nname: removed\ndescription: Removed\n---\nRemoved body\n")
    configuration = _configuration(
        agent_home,
        VALID_CONFIG.replace(
            "[runtime]\n",
            "[runtime]\nenable_skill_always_load = true\n",
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    constructed: list[AgentLoop] = []
    routers: list[object] = []
    permission_controls: list[RuntimePermissionControl] = []
    original_init = AgentLoop.__init__
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Fresh Skill prompt observed."),
                            usage=ModelUsage(
                                input_tokens=2,
                                output_tokens=1,
                                total_tokens=3,
                            ),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )

    def recording_init(loop: AgentLoop, *args: Any, **kwargs: Any) -> None:
        routers.append(kwargs["model_router"])
        permission_control = cast(
            RuntimePermissionControl,
            kwargs["permission_control"],
        )
        if permission_controls:
            assert permission_control is permission_controls[0]
        permission_controls.append(permission_control)
        kwargs["model_router"] = TaskFramingRouterAdapter(kwargs["model_router"])
        original_init(loop, *args, **kwargs)
        constructed.append(loop)

    monkeypatch.setattr(AgentLoop, "__init__", recording_init)
    monkeypatch.setattr(cli, "create_provider", lambda _configuration: provider)

    def update_skills(old: AgentLoop) -> None:
        assert routers
        assert len(permission_controls) == 1
        permission_controls[0].select("read-only")
        cast(Any, routers[0]).set_reasoning_effort("max")
        assert [metadata.name for metadata in old.skill_metadata] == ["current", "removed"]
        assert [(metadata.name, metadata.description) for metadata in old.skill_metadata] == [
            ("current", "Original"),
            ("removed", "Removed"),
        ]
        current_skill.write_bytes(
            b"---\nname: current\ndescription: Refreshed\nalways: false\n---\nRefreshed body\n"
        )
        removed_skill.unlink()
        added_skill = agent_home.path / "skills" / "added" / "SKILL.md"
        added_skill.parent.mkdir(parents=True)
        added_skill.write_bytes(
            b"---\nname: added\ndescription: Added\nalways: true\n---\nAdded body\n"
        )

    async def assert_refreshed(target: AgentLoop, bus: MessageBus) -> None:
        assert len(permission_controls) == 2
        assert permission_controls[1] is permission_controls[0]
        assert permission_controls[1].current() == "read-only"
        assert target._tool_gateway._permission_context.configured_schedule_level == (
            configuration.runtime.permission_level
        )
        assert [(metadata.name, metadata.description) for metadata in target.skill_metadata] == [
            ("added", "Added"),
            ("current", "Refreshed"),
        ]
        messages = await collect_foreground_outbound(bus, "Inspect refreshed Skills.")
        assert any(message.content == "Fresh Skill prompt observed." for message in messages)
        assert len(provider.stream_requests) == 1
        assert provider.stream_requests[0].reasoning_effort == "max"
        system_prompt = provider.stream_requests[0].messages[0]["content"]
        assert isinstance(system_prompt, str)
        assert '"name":"added"' in system_prompt
        assert '"description":"Added"' in system_prompt
        assert "Added body" in system_prompt
        assert '"name":"current"' in system_prompt
        assert '"description":"Refreshed"' in system_prompt
        assert "Refreshed body" not in system_prompt
        assert "Original body" not in system_prompt
        assert '"name":"removed"' not in system_prompt

    events, target, rebound_metadata = await _exercise_cli_contract(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
        monkeypatch=monkeypatch,
        before_resume=update_skills,
        verify_target=assert_refreshed,
    )

    assert events == ["quiesce", "rebind"]
    assert [(metadata.name, metadata.description) for metadata in rebound_metadata] == [
        ("added", "Added"),
        ("current", "Refreshed"),
    ]
    assert len(constructed) == 2
    assert constructed == [constructed[0], target]
    assert len(routers) == 2
    assert routers == [routers[0], routers[0]]
    assert permission_controls == [permission_controls[0], permission_controls[0]]


@pytest.mark.asyncio
async def test_cli_full_access_startup_notice_is_process_lifetime_and_not_repeated_on_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    configuration = _configuration(
        agent_home,
        VALID_CONFIG.replace(
            "[runtime]\n",
            '[runtime]\npermission_level = "full-access"\n',
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    await _exercise_cli_contract(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
        monkeypatch=monkeypatch,
    )

    output = capsys.readouterr().out
    assert output.count("Full-Access is enabled") == 1
    assert "ordinary permission confirmation" in output
    assert "not an OS sandbox" in output
    assert "Validation and Tool errors still apply" in output


@pytest.mark.asyncio
async def test_cli_non_full_access_prints_no_permission_startup_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    configuration = _configuration(agent_home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    await _exercise_cli_contract(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
        monkeypatch=monkeypatch,
    )

    assert "Full-Access is enabled" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cli_force_replacement_cancels_framing_without_old_session_late_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = AgentHome(tmp_path / "agent-home")
    configuration = _configuration(agent_home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocker: BlockingTaskFramingRouterAdapter | None = None
    constructed: list[AgentLoop] = []
    original_init = AgentLoop.__init__

    def recording_init(loop: AgentLoop, *args: Any, **kwargs: Any) -> None:
        nonlocal blocker
        if not constructed:
            blocker = BlockingTaskFramingRouterAdapter(kwargs["model_router"])
            kwargs["model_router"] = blocker
        else:
            kwargs["model_router"] = TaskFramingRouterAdapter(kwargs["model_router"])
        original_init(loop, *args, **kwargs)
        constructed.append(loop)

    class FramingAbortApp:
        fatal_management_error: FatalManagementError | None = None

        def __init__(
            self,
            *,
            bus: MessageBus,
            control: TerminalAgentLoopControl,
            management_dispatcher: ManagementCommandDispatcher,
            skill_metadata: tuple[SkillMetadata, ...],
        ) -> None:
            del skill_metadata
            self._bus = bus
            self._control = control
            self._dispatcher = management_dispatcher

        async def run_async(self) -> None:
            assert blocker is not None
            old = cast(AgentLoop, self._control)
            old.session.commit_agent_run(
                [{"role": "user", "content": "Committed before blocked framing."}],
                pending_last_compacted=old.session.last_compacted,
                pending_action_summary=cast(str, old.session.metadata["summary"]),
            )
            await old.session.wait_for_pending_persist()
            before_messages = deepcopy(old.session.messages)
            before_metadata = deepcopy(old.session.metadata)

            await self._bus.put_inbound(InboundMessage("Blocked framing input."))
            await asyncio.wait_for(blocker.started.wait(), timeout=1)
            assert old.control.has_active_run
            result = await self._dispatcher.resume(old.session.session_id, force=True)
            await asyncio.wait_for(blocker.cancelled.wait(), timeout=1)

            assert result.resumed_session_id == old.session.session_id
            assert old.session.messages == before_messages
            assert old.session.metadata == before_metadata
            target = cast(AgentLoop, self._control)
            assert target is not old
            assert target.session.messages == before_messages
            assert target.session.metadata == before_metadata

        async def quiesce_for_rebind(self) -> None:
            return None

        async def rebind_agent_loop(
            self,
            *,
            control: TerminalAgentLoopControl,
            skill_metadata: tuple[SkillMetadata, ...],
            session_projection: ForegroundConversationProjection,
        ) -> None:
            del skill_metadata
            target = cast(AgentLoop, control)
            assert target._consumer_task is None
            assert session_projection.session_id == target.session.session_id
            self._control = control

    monkeypatch.setattr(AgentLoop, "__init__", recording_init)
    monkeypatch.setattr(cli, "TerminalConversationApp", FramingAbortApp)
    await cli._run_cli_conversation(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
    )

    assert len(constructed) == 2
