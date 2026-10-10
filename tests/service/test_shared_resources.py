from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest

import aide.agent.context.run_context as compactor_module
import aide.service.runtime.service as service_runtime
from aide.agent.context.budget import estimate_request_tokens
from aide.agent.message_bus import InboundMessage, MessageBus
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.service.errors import ServiceError
from aide.service.execution import SessionExecution
from aide.service.runtime import AgentService, SessionClaim, WorkspaceRecord
from aide.service.runtime.records import _LoopState
from aide.skills.catalog import LoadedSkill, SkillLoader
from tests.agent.test_loop import _response, _Router, _runtime, _terminals
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures.agent_loop import DrivenExecutor
from tests.fixtures.project_removal import wait_for_project_removal
from tests.service.test_service_concurrency import _CollectingSink


class _CountingProvider:
    def __init__(self) -> None:
        self.closed = 0
        self.schedule_started = asyncio.Event()

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: object = None,
    ) -> ModelResponse:
        if "shared schedule request" in json.dumps(messages):
            self.schedule_started.set()
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        return ModelResponse(
            message=AssistantModelMessage(content="ok"),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if "shared schedule request" in json.dumps(messages):
            self.schedule_started.set()
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout, continuation

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(
                ModelResponse(
                    message=AssistantModelMessage(content="ok"),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )
            )

        return emit()

    async def close(self) -> None:
        self.closed += 1


@dataclass
class _SessionCase:
    workspace: WorkspaceRecord
    claim: SessionClaim
    client_id: str
    sink: _CollectingSink

    async def submit(self, text: str, run_id: str) -> None:
        await self.workspace.input(
            self.client_id, self.claim.session_id, self.claim.version, text, run_id
        )

    async def completed(self, run_id: str) -> None:
        event = await asyncio.wait_for(self.sink.wait_for("run.completed", run_id), timeout=5)
        assert cast(dict[str, object], event["payload"])["finish_reason"] == "completed"

    async def reload(self, request_id: str) -> dict[str, object]:
        return await self.workspace.service.handle_management(
            self.client_id,
            self.workspace.workspace_id,
            self.claim.session_id,
            "skills/reload",
            {"request_id": request_id},
            claim_version=self.claim.version,
            claim_credential=self.claim.credential,
        )


async def _session_case(service: AgentService, path: Path) -> _SessionCase:
    path.mkdir(exist_ok=True)
    client = await service.register_client("cli")
    sink = _CollectingSink()
    await service.connect_client(client.client_id, sink)
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id, reuse_startup_session=False, creation_scope="chat")
    await service.claim(client.client_id, workspace.workspace_id, session_id)
    claim = await workspace.claim(client.client_id, session_id)
    return _SessionCase(workspace, claim, client.client_id, sink)


def _skill(home: AgentHome, name: str, body: str, *, always: bool = False) -> Path:
    instruction = home.skills_directory / "fixture" / "SKILL.md"
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        f"---\nname: {name}\ndescription: Fixture Skill\nalways: {str(always).lower()}\n---\n"
        + body,
        encoding="utf-8",
    )
    return instruction


def _home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG + """
[models.routes.schedule]
provider_id = "primary"
model = "small-model"
""",
        encoding="utf-8",
    )
    return home


@pytest.mark.asyncio
async def test_service_shares_one_skill_loader_and_model_router_across_workspaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "agent-home")
    _skill(home, "planner", "fixture instructions")
    providers: list[_CountingProvider] = []

    def create_provider(_configuration: object) -> _CountingProvider:
        provider = _CountingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", create_provider)

    load_calls: list[SkillLoader] = []
    original_load = SkillLoader.load

    def record_load(
        loader: SkillLoader,
        *,
        validate: Callable[[tuple[LoadedSkill, ...]], None] | None = None,
    ) -> None:
        load_calls.append(loader)
        original_load(loader, validate=validate)

    monkeypatch.setattr(SkillLoader, "load", record_load)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        cases = [
            await _session_case(service, tmp_path / f"workspace-{workspace}")
            for workspace in range(3)
            for _session in range(10)
        ]
        await asyncio.gather(
            *(
                case.submit(f"/planner input-{index}", f"run-{index}")
                for index, case in enumerate(cases)
            )
        )
        await asyncio.gather(*(case.completed(f"run-{index}") for index, case in enumerate(cases)))

        assert len(load_calls) == 1
        assert load_calls[0] is service.skill_loader
        assert len(providers) == 1
        assert len({case.claim.session_id for case in cases}) == 30
        for case in cases:
            assert case.workspace.resources is not None
            assert case.workspace.resources.router is service.model_router
            assert case.claim.loop.session.messages[-1]["content"] == "ok"

        await cases[0].workspace.close()
        assert providers[0].closed == 0
        await cases[10].submit("/planner after-workspace-close", "surviving-run")
        await cases[10].completed("surviving-run")
        assert len(providers) == 1
        job = ScheduleJob(
            job_id=str(uuid4()),
            message="shared schedule request",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        schedule = cases[10].workspace.schedule_service
        await schedule.add_user_job(job)
        await asyncio.wait_for(providers[0].schedule_started.wait(), timeout=5)
        await asyncio.wait_for(schedule.pause_and_wait_idle(), timeout=5)
        jobs = await schedule.public_snapshot()
        assert jobs[0].state.last_status == "ok"
        history = Session.load(
            cases[10].workspace.workspace_state,
            job.session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert history.messages[-1]["content"] == "ok"
        assert len(providers) == 1
        assert len(load_calls) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("one_shot", [False, True])
async def test_terminal_store_failure_blocks_project_removal_without_stopping_other_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_shot: bool
) -> None:
    home = _home(tmp_path / "agent-home")
    monkeypatch.setattr(
        service_runtime, "create_provider", lambda _configuration: _CountingProvider()
    )
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first = await _session_case(service, tmp_path / "workspace-a")
        second = await _session_case(service, tmp_path / "workspace-b")
        record, _, _ = await service.register_project(
            first.client_id, first.workspace.workspace_path
        )
        failing = first.workspace.schedule_service
        failing_job = ScheduleJob(
            job_id=str(uuid4()),
            message="shared schedule request",
            schedule=(
                JobSchedule.at("2000-01-01T00:00:00.000+00:00")
                if one_shot
                else JobSchedule.every(3600)
            ),
            created_at_ms=1,
            updated_at_ms=1,
        )
        await failing.add_user_job(failing_job)

        def fail_replace(_path: Path, _content: str) -> None:
            raise OSError("PRIVATE_TERMINAL_FAILURE")

        monkeypatch.setattr(failing._store, "_replace_text", fail_replace)
        async with asyncio.timeout(5):
            while failing.status_snapshot().status != "faulted":
                await asyncio.sleep(0)
        assert await failing.public_snapshot() == (failing_job,)
        operation = await service.start_project_removal(first.client_id, record.project_id)
        status = await wait_for_project_removal(
            service, first.client_id, record.project_id, cast(str, operation["operation_id"])
        )
        assert status["status"] == "failed"
        assert "PRIVATE_TERMINAL_FAILURE" not in str(status)
        assert service.projects.list()[0].schedule_state == "removing"
        assert first.workspace.workspace_id in service.workspace_resources.resources

        surviving_job = ScheduleJob(
            job_id=str(uuid4()),
            message="shared schedule request",
            schedule=JobSchedule.every(3600),
            created_at_ms=1,
            updated_at_ms=1,
        )
        surviving = second.workspace.schedule_service
        await surviving.add_user_job(surviving_job)
        async with asyncio.timeout(5):
            while (await surviving.public_snapshot())[0].state.last_status != "ok":
                await asyncio.sleep(0)
        assert surviving.status_snapshot().status == "available"
        history = Session.load(
            second.workspace.workspace_state,
            surviving_job.session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert [message["role"] for message in history.messages] == ["user", "assistant"]
        assert service.workspace_resources.dispatcher.task is not None
        assert not service.workspace_resources.dispatcher.task.done()
    finally:
        with pytest.raises(ServiceError) as failed:
            await service.stop()
        assert failed.value.code == "service_stop_failed"


@pytest.mark.asyncio
async def test_service_closes_all_shared_providers_after_workspace_and_provider_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    config = (
        MINIMAL_VALID_CONFIG
        + '''
[models.providers.secondary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "fixture-key"

[models.providers.secondary.models."small-model"]
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30

[models.routes.memory]
provider_id = "secondary"
model = "small-model"
'''
    )
    (home.path / "config.toml").write_text(config, encoding="utf-8")
    providers: list[_CountingProvider] = []

    def create_provider(_configuration: object) -> _CountingProvider:
        provider = _CountingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", create_provider)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        case = await _session_case(service, tmp_path / "workspace")
        await service.model_router.complete("chat", messages=[], tools=[])
        await service.model_router.complete("memory", messages=[], tools=[])
        assert len(providers) == 2
        close_workspace = case.workspace.close

        async def fail_workspace_close() -> None:
            await close_workspace()
            raise RuntimeError("fixture workspace cleanup failure")

        async def fail_provider_close() -> None:
            providers[0].closed += 1
            raise RuntimeError("fixture provider cleanup failure")

        monkeypatch.setattr(case.workspace, "close", fail_workspace_close)
        monkeypatch.setattr(providers[0], "close", fail_provider_close)
        with pytest.raises(ServiceError) as failed:
            await service.stop()
        assert failed.value.code == "service_stop_failed"
        assert service.state == "stopped"
        assert [provider.closed for provider in providers] == [1, 1]
    finally:
        if service.state != "stopped":
            await service.stop()

    assert providers[0].closed == 1


@pytest.mark.asyncio
async def test_independent_services_do_not_share_workspace_runtime_registry(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path / "agent-home")
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    first = AgentService(home, ConfigLoader(home).load_for_startup())
    second = AgentService(home, ConfigLoader(home).load_for_startup())
    await first.start()
    await second.start()
    first_client = await first.register_client("cli")
    second_client = await second.register_client("cli")

    try:
        first_workspace = await first.attach_workspace(first_client.client_id, workspace_path)
        second_workspace = await second.attach_workspace(second_client.client_id, workspace_path)
        assert first_workspace.resources is not None
        assert second_workspace.resources is not None
        assert first_workspace.resources is not second_workspace.resources
        assert first.model_router is not second.model_router
    finally:
        await first.stop()
        await second.stop()


class _ReloadProvider(_CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[list[dict[str, Any]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked = False

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        if str(messages[0].get("content", "")).startswith("Generate a concise title"):
            return super().stream(**kwargs)
        self.requests.append(deepcopy(list(messages)))
        block = "hold-old" in json.dumps(messages) and not self.blocked
        self.blocked = self.blocked or block

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if block:
                self.started.set()
                await self.release.wait()
                response = ModelResponse(
                    message=AssistantModelMessage(
                        content="",
                        tool_calls=(ModelToolCall("read", "read_file", '{"path":"fixture.txt"}'),),
                    ),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls",
                )
                yield ModelCompleted(response)
            else:
                async for event in super(_ReloadProvider, self).stream(**kwargs):
                    yield event

        return emit()


@pytest.mark.asyncio
async def test_global_reload_keeps_active_run_snapshot_and_updates_other_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    _skill(home, "planner", "OLD_SKILL_BODY")
    provider = _ReloadProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first = await _session_case(service, tmp_path / "workspace-a")
        second = await _session_case(service, tmp_path / "workspace-b")
        (first.workspace.workspace_path / "fixture.txt").write_text(
            "first workspace file", encoding="utf-8"
        )
        await first.submit("/planner hold-old", "old-run")
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        old_snapshot = service.skill_loader.skills
        _skill(home, "reviewer", "NEW_SKILL_BODY")
        result = await second.reload("reload-global")
        assert "management_error" not in result
        new_snapshot = service.skill_loader.skills
        assert new_snapshot is not old_snapshot
        await second.submit("/reviewer other-workspace", "other-run")
        await second.completed("other-run")
        assert first.claim.loop.has_active_run
        provider.release.set()
        await first.completed("old-run")
        assert len(provider.requests) == 3
        for request in (provider.requests[0], provider.requests[2]):
            assert "OLD_SKILL_BODY" in json.dumps(request)
            assert "NEW_SKILL_BODY" not in json.dumps(request)
        assert "NEW_SKILL_BODY" in json.dumps(provider.requests[1])
        assert "first workspace file" in json.dumps(provider.requests[2])
        assert "first workspace file" not in json.dumps(provider.requests[1])
        await first.submit("/reviewer later-first-workspace", "later-run")
        await first.completed("later-run")
        assert "NEW_SKILL_BODY" in str(provider.requests[-1][-1]["content"])
        assert service.skill_loader.skills is new_snapshot
    finally:
        provider.release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_global_reload_publishes_candidate_that_exceeds_another_workspace_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9", "compact_ratio = 0.9\nenable_skill_always_load = true"
    )
    (home.path / "config.toml").write_text(config, encoding="utf-8")
    _skill(home, "planner", "old body", always=True)
    provider = _CountingProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first = await _session_case(service, tmp_path / "workspace-a")
        second = await _session_case(service, tmp_path / "workspace-b")
        assert second.workspace.resources is not None
        memory = second.workspace.resources.memory_manager
        await memory.edit_long_term(old=memory.memory_snapshot(), new="B_MEMORY " * 1200)
        snapshot = service.skill_loader.skills
        _skill(home, "reviewer", "candidate " * 850, always=True)
        result = await first.reload("publish-global")
        assert "management_error" not in result
        assert service.skill_loader.skills is not snapshot
        assert tuple(item.name for item in service.skill_loader.metadata) == ("reviewer",)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_owns_workspace_memory_dream_and_one_schedule_dispatcher(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path / "agent-home")
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first = await _session_case(service, tmp_path / "workspace-a")
        second = await _session_case(service, tmp_path / "workspace-b")

        first_resources = service.workspace_resources.get(first.workspace.workspace_id)
        second_resources = service.workspace_resources.get(second.workspace.workspace_id)
        assert first_resources.memory_manager is first.workspace.memory_manager
        assert second_resources.memory_manager is second.workspace.memory_manager
        assert first_resources.memory_manager is not second_resources.memory_manager
        assert first_resources.dream is first.workspace.dream
        assert second_resources.dream is second.workspace.dream
        assert first_resources.schedule_service.dispatcher is service.workspace_resources.dispatcher
        assert second_resources.schedule_service.dispatcher is service.workspace_resources.dispatcher
        assert len(service.workspace_resources.dispatcher._services) == 2
        assert service.workspace_resources.dispatcher.task is not None

        same_workspace = await _session_case(service, tmp_path / "workspace-a")
        assert same_workspace.workspace.memory_manager is first_resources.memory_manager
        now = datetime.now(UTC)
        contents = {f"first-summary-{index}" for index in range(20)}
        appended = await asyncio.gather(
            *(
                owner.memory_manager.append_summary(content, now)
                for owner, content in zip(
                    [first.workspace, same_workspace.workspace] * 10,
                    sorted(contents),
                    strict=True,
                )
            )
        )
        await second.workspace.memory_manager.append_summary("second-summary", now)
        first_claim = await first_resources.memory_manager.claim_summaries(20)
        second_claim = await second_resources.memory_manager.claim_summaries(20)
        assert sorted(entry.index for entry in appended) == list(range(1, 21))
        assert {entry.content for entry in first_claim.entries} == contents
        assert [entry.index for entry in first_claim.entries] == list(range(1, 21))
        assert first_claim.cursor == 20
        assert [entry.content for entry in second_claim.entries] == ["second-summary"]
        assert second_claim.cursor == 1
        second_memory = await second_resources.memory_manager.read_long_term()
        await first_resources.memory_manager.edit_long_term(
            old="## User Preference\n", new="## User Preference\n\nFIRST-WORKSPACE-ONLY\n"
        )
        assert "FIRST-WORKSPACE-ONLY" in await first_resources.memory_manager.read_long_term()
        assert await second_resources.memory_manager.read_long_term() == second_memory
        for resources in (first_resources, second_resources):
            jobs = await resources.schedule_service._store.snapshot()
            assert len([job for job in jobs if job.source == "system"]) == 1

        await first.workspace.close()
        assert len(service.workspace_resources.dispatcher._services) == 1
        assert service.workspace_resources.get(second.workspace.workspace_id) is second_resources
    finally:
        await service.stop()


def _reload_runtime(
    tmp_path: Path, router: Any, **kwargs: Any
) -> tuple[DrivenExecutor, Session, MessageBus, AgentService]:
    """Exercise global publication against a controlled loaded Session."""
    loop, session, bus = _runtime(tmp_path, router, **kwargs)
    home = AgentHome(tmp_path / "agent-home")
    service = AgentService(home, loop._configuration)
    service._skill_loader = loop._skill_loader
    workspace = WorkspaceRecord(service, session.workspace_state.workspace_path, loop._configuration)
    handle = SessionExecution(
        session, bus, lambda _state: loop, service.reload_skills, loop.execution.runtime_status_input,
        run_state=loop._session_run_state,
    )
    workspace._loops[session.session_id] = _LoopState(loop=handle, bus=bus, owner_client_id=None)
    service._workspaces[workspace.workspace_id] = workspace
    return loop, session, bus, service


def test_agent_loop_reload_returns_the_current_loader_metadata_and_reuses_generation_state(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, session, bus, service = _reload_runtime(tmp_path, _Router(()))
    loader = loop._skill_loader
    initial_session = loop.session
    initial_context_loader = loop._context_builder._skill_loader
    before_messages = deepcopy(session.messages)
    bus_operations: list[str] = []

    async def record_bus_operation(name: str) -> None:
        bus_operations.append(name)

    object.__setattr__(bus, "reset", lambda: record_bus_operation("reset"))
    object.__setattr__(
        bus,
        "pause_inbound_delivery",
        lambda: record_bus_operation("pause"),
    )
    object.__setattr__(
        bus,
        "resume_inbound_delivery",
        lambda: record_bus_operation("resume"),
    )

    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )

    metadata = service.reload_skills()

    assert loop.session is loop.execution.session
    assert loop.session is initial_session is session
    assert loop._bus is bus
    assert loop._skill_loader is loader is initial_context_loader
    assert session.messages == before_messages
    assert bus_operations == []
    assert metadata == loader.metadata
    assert tuple(item.name for item in metadata) == ("reviewer",)
    assert loader.get("planner") is None
    invocation = loader.resolve_manual("/reviewer request")
    assert invocation is not None
    assert invocation.metadata == metadata[0]
    assert invocation.body.splitlines()[-1] == "new body"



def test_startup_and_skill_reload_do_not_estimate_context_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\nold body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9",
        "compact_ratio = 0.9\nenable_skill_always_load = true",
    )
    loop, _session, _bus, service = _reload_runtime(tmp_path, _Router(()), config_text=config)
    loader = loop._skill_loader
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\n"
        + ("oversized body\n" * 20_000),
        encoding="utf-8",
    )

    import aide.agent.context.run_context as run_context_module

    def reject_estimate(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("Skill activation must not estimate Model request tokens")

    monkeypatch.setattr(run_context_module, "estimate_request_tokens", reject_estimate)
    monkeypatch.setattr(
        run_context_module.ContextController,
        "estimate_request_tokens",
        staticmethod(reject_estimate),
    )
    monkeypatch.setattr(
        run_context_module.ContextController,
        "project_next_request_usage",
        staticmethod(reject_estimate),
    )
    loop.preflight()
    metadata = service.reload_skills()

    assert tuple(item.name for item in metadata) == ("always",)
    assert loader.resolve_manual("/always request") is not None
    assert loader.skills[0].always is True



def test_skill_reload_does_not_build_a_candidate_context_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, _session, _bus, service = _reload_runtime(tmp_path, _Router(()))
    loader = loop._skill_loader
    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )

    def reject_projection(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("Skill reload must not build a Model request context")

    monkeypatch.setattr(loop._context_builder, "build_status_messages", reject_projection)
    metadata = service.reload_skills()

    assert metadata == loader.metadata
    assert tuple(item.name for item in loader.metadata) == ("reviewer",)
    assert loader.resolve_manual("/planner request") is None
    assert loader.resolve_manual("/reviewer request") is not None



def test_runtime_status_keeps_the_complete_read_only_context_projection(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: active\ndescription: Active snapshot\nalways: true\n---\nactive body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "compact_ratio = 0.9",
        "compact_ratio = 0.9\nenable_skill_always_load = true",
    )
    loop, _session, _bus, _service = _reload_runtime(
        tmp_path,
        _Router(()),
        config_text=config,
    )
    loader = loop._skill_loader
    status = loop.execution.runtime_status_input()
    messages = list(status.projected_messages)
    tools = list(status.projected_tools)

    assert '"name":"active"' in messages[0]["content"]
    assert tuple(tools) == loop.tool_schemas
    assert estimate_request_tokens(messages, tools) > estimate_request_tokens(messages)
    assert loader.skills



@pytest.mark.asyncio
async def test_reload_during_active_run_preserves_old_request_and_updates_future_run(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ReloadBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        def stream(
            self,
            route: Literal["chat", "schedule", "subagent"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))
            first = len(self.requests) == 1

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                if first:
                    self.first_started.set()
                    await self.release_first.wait()
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ReloadBarrierRouter()
    loop, session, bus, service = _reload_runtime(
        tmp_path,
        router,
    )
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner first request"))
        await asyncio.wait_for(router.first_started.wait(), timeout=1)
        first_request = deepcopy(router.requests[0])

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = service.reload_skills()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        assert router.requests[0] == first_request
        assert '"name":"planner"' in str(first_request[0]["content"])
        assert '"name":"reviewer"' not in str(first_request[0]["content"])
        assert "old body" in str(first_request[-1]["content"])
        assert "new body" not in str(first_request[-1]["content"])

        router.release_first.set()
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("/reviewer second request"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 2
    assert '"name":"reviewer"' in str(router.requests[1][0]["content"])
    assert '"name":"planner"' not in str(router.requests[1][0]["content"])
    assert "new body" in str(router.requests[1][-1]["content"])
    assert "old body" not in str(router.requests[1][-1]["content"])



@pytest.mark.asyncio
async def test_reload_during_context_preparation_keeps_run_skill_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ContextPreparationBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []

        def stream(
            self,
            route: Literal["chat", "schedule", "subagent"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ContextPreparationBarrierRouter()
    loop, session, bus, service = _reload_runtime(
        tmp_path,
        router,
    )
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()
    original_prepare = compactor_module.ContextController.prepare

    async def blocked_prepare(
        context: compactor_module.ContextController, **options: Any
    ) -> list[dict[str, Any]]:
        preparation_started.set()
        await release_preparation.wait()
        return await original_prepare(context, **options)

    monkeypatch.setattr(compactor_module.ContextController, "prepare", blocked_prepare)
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("first request"))
        await asyncio.wait_for(preparation_started.wait(), timeout=1)

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = service.reload_skills()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        release_preparation.set()
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert "planner" in str(router.requests[0][0]["content"])
    assert "reviewer" not in str(router.requests[0][0]["content"])
