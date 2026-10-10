"""Configuration changes reach resident Sessions without interrupting existing Runs."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import aide.service.runtime.service as service_runtime
from aide.agent.session.restore import RestoreManager
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from aide.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.service.test_service_concurrency import _CollectingSink, _response


@pytest.mark.asyncio
async def test_configuration_change_preserves_resources_during_workspace_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    providers: list[_ConfigurationProvider] = []

    def factory(_configuration: object) -> _ConfigurationProvider:
        provider = _ConfigurationProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", factory)
    service = AgentService(home, reconnect_timeout=3600)
    await service.start()
    entered = asyncio.Event()
    resume = asyncio.Event()
    attachment: asyncio.Task[Any] | None = None
    try:
        path = tmp_path / "first"
        path.mkdir()
        client = await service.register_client("cli")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, path)
        session = await workspace.create_draft(client.client_id, creation_scope="chat")
        await service.claim(client.client_id, workspace.workspace_id, session)
        claim = await workspace.claim(client.client_id, session)
        await workspace.input(client.client_id, session, claim.version, "initial-0", "warm")
        await asyncio.wait_for(sink.wait_for("run.completed", "warm"), timeout=5)
        await claim.loop.wait_for_restore_idle()
        assert len(providers) == 1

        recover = RestoreManager.recover_pending

        async def blocked_recovery(manager: RestoreManager) -> Any:
            entered.set()
            await resume.wait()
            return await recover(manager)

        monkeypatch.setattr(RestoreManager, "recover_pending", blocked_recovery)
        later_path = tmp_path / "later"
        later_path.mkdir()
        later_client = await service.register_client("cli")
        attachment = asyncio.create_task(service.attach_workspace(later_client.client_id, later_path))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service.update_configuration(
            "change-provider", cast(str, service.config_view()["revision"]),
            {"models": {"providers": {"primary": {"base_url": "https://next.example/v1"}}}},
        )
        await workspace.input(client.client_id, session, claim.version, "next-0", "changed")
        await asyncio.wait_for(sink.wait_for("run.completed", "changed"), timeout=5)
        await claim.loop.wait_for_restore_idle()
        await service._configuration_resources._drain_cleanup()
        assert len(providers) == 2
        assert providers[0].close_calls == 0

        resume.set()
        later = await asyncio.wait_for(attachment, timeout=5)
        assert later.resources.mcp_manager.started
        assert providers[0].close_calls == 0
    finally:
        resume.set()
        if attachment is not None and not attachment.done():
            attachment.cancel()
            await asyncio.gather(attachment, return_exceptions=True)
        await service.stop()
    assert all(provider.close_calls == 1 for provider in providers)


class _ConfigurationProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, str, int, object]] = []
        self.close_calls = 0

    async def complete(
        self, *, messages: ModelMessages, tools: Sequence[dict[str, Any]], model: str,
        max_output: int, temperature: float, reasoning_effort: object, timeout: int,
        continuation: object = None,
    ) -> ModelResponse:
        return _response("A concise title")

    def stream(
        self, *, messages: ModelMessages, tools: Sequence[dict[str, Any]], model: str,
        max_output: int, temperature: float, reasoning_effort: object, timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        assert self.close_calls == 0
        user_index = max(i for i, item in enumerate(messages) if item.get("role") == "user")
        content = str(messages[user_index]["content"])
        matches = re.findall(r"hold-old|queued-new|limit-new|initial-\d|next-\d", content)
        prompt = matches[-1] if matches else content
        self.calls.append((prompt, model, max_output, reasoning_effort))
        has_result = any(item.get("role") == "tool" for item in messages[user_index + 1:])

        async def events() -> AsyncIterator[ModelStreamEvent]:
            if prompt == "hold-old" and not has_result:
                self.started.set()
                await self.release.wait()
            if (prompt == "hold-old" and not has_result) or prompt == "limit-new":
                yield ModelCompleted(ModelResponse(
                    AssistantModelMessage("Read the file", (ModelToolCall(
                        f"read-config-marker-{len(self.calls)}", "read_file", json.dumps({"path": "marker.txt"}),
                    ),)), ModelUsage(1, 1, 2), "tool_calls",
                ))
            else:
                yield ModelCompleted(_response(f"completed {prompt} with {model}"))
        return events()

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_saved_and_external_configuration_reach_four_resident_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    config_path = home.path / "config.toml"
    config_path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    provider = _ConfigurationProvider()
    creations = 0

    def factory(_configuration: object) -> _ConfigurationProvider:
        nonlocal creations
        creations += 1
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", factory)
    service = AgentService(home, reconnect_timeout=3600)
    await service.start()
    identities: list[tuple[str, str, str, int, str]] = []
    sinks: dict[str, _CollectingSink] = {}
    run_sinks: dict[str, _CollectingSink] = {}
    try:
        for index in range(2):
            path = tmp_path / f"workspace-{index}"
            path.mkdir()
            (path / "marker.txt").write_text("configuration marker", encoding="utf-8")
            for _ in range(2):
                client = await service.register_client("cli")
                sink = _CollectingSink()
                sinks[client.client_id] = sink
                await service.connect_client(client.client_id, sink)
                workspace = await service.attach_workspace(client.client_id, path)
                session = await workspace.create_draft(client.client_id, creation_scope="chat")
                claim = await service.claim(client.client_id, workspace.workspace_id, session)
                details = cast(dict[str, Any], claim["claim"])
                identities.append((client.client_id, workspace.workspace_id, session,
                                   details["claim_version"], details["reconnect_credential"]))

        async def submit(index: int, prompt: str) -> str:
            client_id, workspace_id, session_id, version, _ = identities[index]
            result = await service.submit_user_input(
                client_id, workspace_id, session_id, version, prompt, prompt,
            )
            run_id = cast(str, result["run_id"])
            run_sinks[run_id] = sinks[client_id]
            return run_id

        async def finish(run_id: str) -> None:
            async with asyncio.timeout(10):
                await run_sinks[run_id].wait_for("run.completed", run_id)

        initial = [await submit(index, f"initial-{index}") for index in range(4)]
        await asyncio.gather(*(finish(run_id) for run_id in initial))
        assert creations == 1
        identity = service.service_instance_id
        old_run = await submit(0, "hold-old")
        try:
            async with asyncio.timeout(10):
                await provider.started.wait()
        except TimeoutError:
            pytest.fail(f"Old Run did not start: {provider.calls!r}")
        queued = await submit(0, "queued-new")
        view = service.config_view()
        saved = await service.update_configuration(
            "save-runtime", cast(str, view["revision"]), {"runtime": {"max_iterations": 51}},
        )
        application = cast(dict[str, Any], saved["application"])
        assert application["status"] == "next-run-required"
        assert application["restart_required"] is False
        # An edit outside the service must also be visible without opening Settings.
        external = config_path.read_text(encoding="utf-8").replace("small-model", "next-model")
        external = external.replace("max_output = 1024", "max_output = 512")
        external = external.replace("context_window = 8192", "context_window = 32768")
        config_path.write_text(external, encoding="utf-8")
        new_runs = [await submit(index, f"next-{index}") for index in range(1, 4)]
        await asyncio.gather(*(finish(run_id) for run_id in new_runs))
        assert provider.close_calls == 0
        provider.release.set()
        await finish(old_run)
        await finish(queued)
        limit = await submit(3, "limit-new")
        await finish(limit)
        old_calls = [call for call in provider.calls if call[0] == "hold-old"]
        assert len(old_calls) == 2
        assert all(call[1:3] == ("small-model", 1024) for call in old_calls)
        subsequent = [call for call in provider.calls if call[0].startswith("next-") or call[0] == "queued-new"]
        assert len(subsequent) == 4
        assert all(call[1:3] == ("next-model", 512) for call in subsequent)
        assert sum(call[0] == "limit-new" for call in provider.calls) == 51
        assert creations == 1 and service.service_instance_id == identity
        assert provider.close_calls == 0
        for client_id, workspace_id, session_id, version, credential in identities:
            snapshot = await service.get_session_snapshot(
                client_id, workspace_id, session_id, version, credential,
            )
            assert cast(dict[str, Any], snapshot["snapshot"])["messages"]
            claim = await service.claim(client_id, workspace_id, session_id)
            assert cast(dict[str, Any], claim["claim"])["claim_version"] == version
    finally:
        provider.release.set()
        await service.stop()
    assert provider.close_calls == 1
