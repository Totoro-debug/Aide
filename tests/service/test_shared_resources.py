from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

import omni.service.runtime as service_runtime
from omni.agent.loop import ModelContextOverflowError
from omni.agent.session.session import Session, SessionStoragePartition
from omni.agent.tools.tool_gateway import ModelToolCall
from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.service.errors import ServiceError
from omni.service.runtime import LocalService, SessionClaim, WorkspaceServiceRuntime
from omni.skills.catalog import LoadedSkill, SkillLoader
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
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
    workspace: WorkspaceServiceRuntime
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


async def _session_case(service: LocalService, path: Path) -> _SessionCase:
    path.mkdir(exist_ok=True)
    client = await service.register_client("cli")
    sink = _CollectingSink()
    await service.connect_client(client.client_id, sink)
    workspace = await service.attach_workspace(client.client_id, path)
    session_id = await workspace.create_draft(client.client_id, reuse_startup_session=False)
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
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
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
    service = LocalService(home, ConfigLoader(home).load_for_startup())
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
            assert case.workspace.runtime is not None
            assert case.workspace.runtime.router is service.model_router
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
async def test_service_closes_all_shared_providers_after_workspace_and_provider_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    config = (
        MINIMAL_VALID_CONFIG
        + """
[models.providers.secondary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "fixture-key"
models = ["small-model"]

[models.routes.memory]
provider_id = "secondary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
"""
    )
    (home.path / "config.toml").write_text(config, encoding="utf-8")
    providers: list[_CountingProvider] = []

    def create_provider(_configuration: object) -> _CountingProvider:
        provider = _CountingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", create_provider)
    service = LocalService(home, ConfigLoader(home).load_for_startup())
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
    first = LocalService(home, ConfigLoader(home).load_for_startup())
    second = LocalService(home, ConfigLoader(home).load_for_startup())
    await first.start()
    await second.start()
    first_client = await first.register_client("cli")
    second_client = await second.register_client("cli")

    try:
        first_workspace = await first.attach_workspace(first_client.client_id, workspace_path)
        second_workspace = await second.attach_workspace(second_client.client_id, workspace_path)
        assert first_workspace.runtime is not None
        assert second_workspace.runtime is not None
        assert first_workspace.runtime is not second_workspace.runtime
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
    service = LocalService(home, ConfigLoader(home).load_for_startup())
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
async def test_global_reload_rejects_candidate_that_overflows_another_workspace(
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
    service = LocalService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    try:
        first = await _session_case(service, tmp_path / "workspace-a")
        second = await _session_case(service, tmp_path / "workspace-b")
        assert second.workspace.runtime is not None
        memory = second.workspace.runtime.memory_manager
        await memory.edit_long_term(old=memory.memory_snapshot(), new="B_MEMORY " * 1200)
        snapshot = service.skill_loader.skills
        _skill(home, "reviewer", "candidate " * 850, always=True)
        # The real projection fits the caller and overflows the other Workspace.
        candidate = SkillLoader(
            root=home.skills_directory, reserved_names=(), enable_always_load=True
        )
        candidate.load()
        first.claim.loop._validate_model_context_budget(candidate.skills)
        second.claim.loop._validate_model_context_budget(snapshot)
        with pytest.raises(ModelContextOverflowError):
            second.claim.loop._validate_model_context_budget(candidate.skills)
        result = await first.reload("reject-global")
        assert cast(dict[str, object], result["management_error"])["code"] == "skill_reload_failed"
        assert service.skill_loader.skills is snapshot
        await asyncio.gather(
            first.submit("/planner still-valid-a", "kept-a"),
            second.submit("/planner still-valid-b", "kept-b"),
        )
        await asyncio.gather(first.completed("kept-a"), second.completed("kept-b"))
        assert service.skill_loader.skills is snapshot
    finally:
        await service.stop()
