"""Service startup completes interrupted durable Restore before exposing tasks."""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from aide.agent.session.restore import RestoreMode
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.config import ConfigLoader
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService
from tests.restore.conftest import AfterRestorePhase
from tests.service.test_foreground_questions import (
    _command,
    _home,
    _open_chat,
    _QuestionProvider,
    _wait_for_idle,
)


class _RestoreRouterWorkerProvider(_QuestionProvider):
    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: object,
        timeout: int,
        continuation: object = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        system = messages[0].get("content", "") if messages else ""
        if isinstance(system, str) and system.startswith("You are an Aide Worker."):
            return super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )

        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )
        if "Worker reports for automatic continuation:" in user_text:
            call = ModelToolCall("restore-finish", "finish", '{"response":"The answer is ready."}')
        elif "write-answer" in user_text and not any(
            message.get("tool_call_id") == "restore-dispatch" for message in messages
        ):
            call = ModelToolCall(
                "restore-dispatch",
                "dispatch_worker",
                '{"task_id":"restore-answer","task":"Perform write-answer in answer.txt"}',
            )
        else:
            call = ModelToolCall("restore-finish", "finish", '{"response":"The answer is ready."}')

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            ))

        return emit()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [
    "pending_intent", "journal_pruned", "file_intent", "file_result",
    "files_replayed", "session_write", "session_persisted", "complete",
])
async def test_service_restart_finishes_restore_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    after_restore_phase: AfterRestorePhase, phase: str,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    target = directory / "answer.txt"
    target.write_bytes(b"original")
    provider = _RestoreRouterWorkerProvider()
    monkeypatch.setattr("aide.service.runtime.service.create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    replacement: AgentService | None = None
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        service.client_permission(client).select("workspace-write")
        workspace = service.workspace(wid)
        await service.handle_command(client, _command(request_id="write", command_type="input",
            workspace_id=wid, session_id=sid, claim_version=version,
            payload={"text": "write-answer"}))
        await _wait_for_idle(workspace, sid)
        assert target.read_text() == "answer effect"
        snapshot = workspace.session_snapshot(sid)
        anchor = cast(list[dict[str, Any]], snapshot["restore_anchors"])[0]["anchor_id"]
        old_store = workspace.foreground_work_store(sid)
        old_task = old_store.load().tasks[0]
        credential = workspace.require_claim(client, sid, version).credential
        await service.inspect_restore(client, wid, sid, version, credential, "inspect", anchor)

        def interrupt() -> None:
            raise RuntimeError("injected durable Restore interruption")

        after_restore_phase(phase, interrupt)
        with pytest.raises(ServiceError):
            await service.commit_restore(client, wid, sid, version, credential,
                                         "restore", anchor, RestoreMode.FILES.value)
        calls = (provider.stream_calls, provider.complete_calls)
        # Closing a workspace with an unfinished Restore reports the barrier;
        # it must not finish the transaction and conceal the persisted fault.
        with pytest.raises(ServiceError) as shutdown:
            await service.stop()
        assert shutdown.value.code == "service_stop_failed"
        replacement = AgentService(home, ConfigLoader(home).load())
        await replacement.start()
        _, reopened, _, new_wid = await _open_chat(replacement, directory, session_id=sid)
        resident = replacement.workspace(new_wid)
        assert reopened == sid
        assert target.read_bytes() == b"original"
        assert resident.session_snapshot(sid)["messages"] == []
        assert (provider.stream_calls, provider.complete_calls) == calls
        before = resident.foreground_work_store(sid).load()
        old_store.complete_task(old_task.task_id, old_task.execution_id)
        assert resident.foreground_work_store(sid).load() == before
    finally:
        if replacement is not None:
            await replacement.stop()
        if service.state != "stopped":
            await service.stop()
