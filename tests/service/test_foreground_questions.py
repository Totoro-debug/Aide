from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import aide.service.runtime.service as service_runtime
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.config.agent_home import AgentHome
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
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


class _QuestionProvider:
    def __init__(self) -> None:
        self.stream_calls = 0
        self.complete_calls = 0
        self.question_calls = 0
        self.user_texts: list[str] = []
        self.answer_started = asyncio.Event()
        self.release_answer = asyncio.Event()
        self.question_input_tokens = 1

    async def complete(
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
    ) -> ModelResponse:
        self.complete_calls += 1
        del tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        system = messages[0].get("content") if messages else None
        if isinstance(system, str) and system.startswith("Generate a concise title"):
            return _response("Question flow")
        return _response(
            '{"action":"replace","task_goal":"answer the input",'
            '"completion_boundary":"return one answer"}'
        )

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
        del model, max_output, temperature, reasoning_effort, timeout, continuation
        self.stream_calls += 1
        has_ask_user = any(
            tool.get("name") == "ask_user"
            or (
                isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == "ask_user"
            )
            for tool in tools
        )
        user = next(
            (item.get("content") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        user_text = user if isinstance(user, str) else ""
        self.user_texts.append(user_text)

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if "write-answer" in user_text and not any(
                message.get("tool_call_id") == "answer-write" for message in messages
            ):
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(
                            content="",
                            tool_calls=(
                                ModelToolCall(
                                    "answer-write",
                                    "write_file",
                                    '{"path":"answer.txt","content":"answer effect"}',
                                ),
                            ),
                        ),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                )
                return
            if has_ask_user and "ask-foreground-question" in user_text:
                self.question_calls += 1
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(
                            content="",
                            tool_calls=(
                                ModelToolCall(
                                    "question-call",
                                    "ask_user",
                                    '{"question":"Which output directory should I use?"}',
                                ),
                                ModelToolCall(
                                    "after-question",
                                    "write_file",
                                    '{"path":"after-question.txt","content":"must not run"}',
                                ),
                            ),
                        ),
                        usage=ModelUsage(
                            input_tokens=self.question_input_tokens,
                            output_tokens=1,
                            total_tokens=self.question_input_tokens + 1,
                        ),
                        finish_reason="tool_calls",
                    )
                )
                return
            if "reports-directory" in user_text:
                self.answer_started.set()
                await self.release_answer.wait()
            yield ModelCompleted(_response("The request is complete."))

        return emit()

    async def close(self) -> None:
        return None


def _home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG
        + '\n[models.routes.schedule]\nprovider_id = "primary"\nmodel = "small-model"\n',
        encoding="utf-8",
    )
    return home


async def _wait_until(predicate: Any) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def _command(
    *,
    request_id: str,
    command_type: str,
    workspace_id: str,
    session_id: str,
    claim_version: int,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "type": command_type,
        "workspace_id": workspace_id,
        "session_id": session_id,
        "claim_version": claim_version,
        "payload": payload,
    }


async def _open_chat(
    service: AgentService,
    directory: Path,
    *,
    session_id: str | None = None,
) -> tuple[str, str, int, str]:
    client = await service.register_client("cli")
    workspace = await service.attach_workspace(client.client_id, directory)
    opened = await service.open_conversation(
        client.client_id,
        workspace_id=workspace.workspace_id,
        session_id=session_id,
        create_new=session_id is None,
    )
    claim = cast(dict[str, object], opened["claim"])
    return (
        client.client_id,
        opened["session_id"],
        cast(int, claim["claim_version"]),
        workspace.workspace_id,
    )


async def _wait_for_idle(workspace: Any, session_id: str) -> None:
    state = workspace._loops[session_id]
    try:
        await _wait_until(
            lambda: (
                not state.run_ids and state.output_task is None and not state.loop.has_active_run
            )
        )
    except TimeoutError as error:
        task_stacks = {
            task.get_name(): [
                f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
                for frame in task.get_stack(limit=8)
            ]
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        }
        raise AssertionError(
            f"Session did not become idle: run_ids={state.run_ids!r}, "
            f"live_runs={state.live_runs!r}, output_task={state.output_task!r}, "
            f"processor_task={state.processor_task!r}, "
            f"active={state.loop.has_active_run!r}, messages={state.loop.session.messages!r}, "
            f"task_stacks={task_stacks!r}"
        ) from error


@pytest.mark.asyncio
async def test_question_answer_is_durable_claimed_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    replacement: AgentService | None = None
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        first = await service.handle_command(
            client_id,
            _command(
                request_id="question-input",
                command_type="input",
                workspace_id=workspace.workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        assert cast(dict[str, object], first["result"])["kind"] == "conversation_input"
        await _wait_for_idle(workspace, session_id)
        snapshot = workspace.session_snapshot(session_id)
        questions = cast(list[dict[str, object]], snapshot["foreground_questions"])
        question = questions[0]
        question_id = cast(str, question["question_id"])
        revision = cast(int, question["revision"])
        assert question["state"] == "pending"
        assert provider.question_calls == 1
        assert not (directory / "after-question.txt").exists()
        history = workspace._loops[session_id].loop.session.messages
        tool_messages = [message for message in history if message.get("role") == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0]["tool_call_id"] == "question-call"
        assert tool_messages[0]["status"] == "success"
        assert tool_messages[1]["tool_call_id"] == "after-question"
        assert tool_messages[1]["status"] == "error"
        assert "not executed" in str(tool_messages[1]["content"])
        assert workspace._loops[session_id].loop.session.metadata["token_usage"]["model_calls"] > 0
        from aide.trace.sqlite_recorder import SQLiteTraceRecorder
        from aide.trace.types import TraceKind, TraceRecordFilters
        from tests.fixtures.trace import wait_for_trace_writer

        recorder = service.trace_recorder
        assert isinstance(recorder, SQLiteTraceRecorder)
        await wait_for_trace_writer(recorder)
        records = (
            await recorder.list_records(TraceRecordFilters(owner_session_id=session_id))
        ).items
        tools = [record for record in records if record.kind == TraceKind.TOOL]
        assert [record.tool_name for record in tools] == ["ask_user"]
        detail = await recorder.get_record(tools[0].record_id)
        assert detail is not None and "waiting" in str(detail.result)
        run_models = [record for record in records if record.kind == TraceKind.MODEL]
        usage = workspace._loops[session_id].loop.session.metadata["token_usage"]
        assert usage["model_calls"] == len(run_models) == 3
        assert (
            usage["total_tokens"]
            == sum(record.usage.total_tokens or 0 for record in run_models)
            == 6
        )

        streams_before_normal_input = provider.stream_calls
        normal_input = await service.handle_command(
            client_id,
            _command(
                request_id="normal-while-question-waits",
                command_type="input",
                workspace_id=workspace.workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ordinary input while the question waits"},
            ),
        )
        assert cast(dict[str, object], normal_input["result"])["kind"] == "conversation_input"
        await _wait_for_idle(workspace, session_id)
        assert cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )
        assert provider.stream_calls == streams_before_normal_input + 1
        streams_before_answer = provider.stream_calls

        with pytest.raises(ServiceError, match="stale"):
            await service.handle_command(
                client_id,
                _command(
                    request_id="stale-claim-answer",
                    command_type="answer_question",
                    workspace_id=workspace.workspace_id,
                    session_id=session_id,
                    claim_version=claim_version + 1,
                    payload={
                        "question_id": question_id,
                        "answer": "reports-directory",
                        "expected_revision": revision,
                    },
                ),
            )
        with pytest.raises(ServiceError) as stale_revision:
            await service.handle_command(
                client_id,
                _command(
                    request_id="stale-revision-answer",
                    command_type="answer_question",
                    workspace_id=workspace.workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={
                        "question_id": question_id,
                        "answer": "reports-directory",
                        "expected_revision": revision - 1,
                    },
                ),
            )
        assert stale_revision.value.code == "foreground_question_conflict"
        assert provider.stream_calls == streams_before_answer
        completions_before = provider.complete_calls
        other = await service.register_client("cli")
        await service.attach_workspace(other.client_id, directory)
        for index, (answer_client, payload) in enumerate(
            [
                (other.client_id, {"answer": "reports-directory"}),
                (client_id, {"answer": " "}),
                (client_id, {"answer": "x" * 10001}),
                (client_id, {"answer": "reports-directory", "choice_id": "invalid"}),
            ]
        ):
            with pytest.raises(ServiceError):
                await service.handle_command(
                    answer_client,
                    _command(
                        request_id=f"invalid-answer-{index}",
                        command_type="answer_question",
                        workspace_id=workspace_id,
                        session_id=session_id,
                        claim_version=claim_version,
                        payload={
                            "question_id": question_id,
                            "expected_revision": revision,
                            **payload,
                        },
                    ),
                )
        assert provider.complete_calls == completions_before
        assert provider.stream_calls == streams_before_answer

        completions_before_restart = provider.complete_calls
        await service.stop()
        replacement = AgentService(home, ConfigLoader(home).load())
        await replacement.start()
        next_client, same_session_id, next_claim_version, next_workspace_id = await _open_chat(
            replacement, directory, session_id=session_id
        )
        assert same_session_id == session_id
        next_workspace = replacement.workspace(next_workspace_id)
        restored_questions = cast(
            list[dict[str, object]],
            next_workspace.session_snapshot(session_id)["foreground_questions"],
        )
        assert restored_questions[0]["question_id"] == question_id
        assert provider.stream_calls == streams_before_answer
        assert provider.complete_calls == completions_before_restart

        answer_command = _command(
            request_id="answer-once",
            command_type="answer_question",
            workspace_id=next_workspace.workspace_id,
            session_id=session_id,
            claim_version=next_claim_version,
            payload={
                "question_id": question_id,
                "answer": "reports-directory",
                "expected_revision": restored_questions[0]["revision"],
            },
        )
        accepted = await replacement.handle_command(next_client, answer_command)
        run_id = cast(str, cast(dict[str, object], accepted["result"])["run_id"])
        await asyncio.wait_for(provider.answer_started.wait(), timeout=5)
        with pytest.raises(ServiceError) as changed_replay:
            await replacement.handle_command(
                next_client,
                {
                    **answer_command,
                    "payload": {
                        **cast(dict[str, object], answer_command["payload"]),
                        "answer": "other",
                    },
                },
            )
        assert changed_replay.value.code == "foreground_question_conflict"

        duplicates = await asyncio.gather(
            *(
                replacement.handle_command(
                    next_client,
                    _command(
                        request_id=f"answer-retry-{index}",
                        command_type="answer_question",
                        workspace_id=next_workspace.workspace_id,
                        session_id=session_id,
                        claim_version=next_claim_version,
                        payload={
                            "question_id": question_id,
                            "answer": "reports-directory",
                            "expected_revision": restored_questions[0]["revision"],
                        },
                    ),
                )
                for index in range(10)
            )
        )
        assert {
            cast(str, cast(dict[str, object], item["result"])["run_id"]) for item in duplicates
        } == {run_id}
        assert provider.stream_calls == streams_before_answer + 1
        provider.release_answer.set()
        await _wait_for_idle(next_workspace, session_id)
        final = next_workspace.session_snapshot(session_id)
        assert final["foreground_questions"] == []
        final_history = next_workspace._loops[session_id].loop.session.messages
        assert any(
            message.get("role") == "user"
            and message.get("foreground_question_id") == question_id
            and message.get("content") == "reports-directory"
            for message in final_history
        )
        assert provider.question_calls == 1
        assert not (directory / "after-question.txt").exists()
        claim = next_workspace.require_claim(next_client, session_id, next_claim_version)
        await replacement.release_claim(
            next_client, next_workspace_id, session_id, next_claim_version, claim.credential
        )
        with pytest.raises(ServiceError):
            await replacement.handle_command(next_client, answer_command)
    finally:
        if replacement is not None:
            await replacement.stop()
        else:
            await service.stop()


@pytest.mark.asyncio
async def test_uncommitted_question_cannot_borrow_an_earlier_tool_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.agent.session.session import Session

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)

        async def ask(request: str) -> None:
            await service.handle_command(
                client,
                _command(
                    request_id=request,
                    command_type="input",
                    workspace_id=wid,
                    session_id=sid,
                    claim_version=version,
                    payload={"text": "ask-foreground-question"},
                ),
            )
            await _wait_for_idle(workspace, sid)

        await ask("first")
        original = workspace.session_snapshot(sid)["foreground_questions"]

        def fail_write(self: Session, content: bytes) -> None:
            raise OSError("injected Session write failure")

        with monkeypatch.context() as fault:
            fault.setattr(Session, "_write_content", fail_write)
            await ask("second")
            visible = [
                q
                for q in cast(
                    list[dict[str, object]], workspace.session_snapshot(sid)["foreground_questions"]
                )
                if q["state"] == "pending"
            ]
            assert [q["question_id"] for q in visible] == [
                q["question_id"] for q in cast(list[dict[str, object]], original)
            ]
        assert provider.question_calls == 2
        assert not (directory / "after-question.txt").exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_deleted_question_rejects_late_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.agent.foreground_work.store import ForegroundWorkConflict

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        await service.handle_command(
            client,
            _command(
                request_id="ask",
                command_type="input",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        workspace = service.workspace(wid)
        await _wait_for_idle(workspace, sid)
        old_store = workspace.foreground_work_store(sid)
        claim = workspace.require_claim(client, sid, version)
        await workspace.delete_session(client, sid, version, claim.credential, "delete")
        assert not old_store.path.exists()
        with pytest.raises(ForegroundWorkConflict):
            old_store.begin_task("late-task", "late-execution", generation=1)
        assert not old_store.path.parent.exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_answer_restore_returns_conversation_question_and_real_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.agent.session.restore import RestoreMode

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    target = directory / "answer.txt"
    target.write_bytes(b"original bytes")
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(
            client,
            _command(
                request_id="ask",
                command_type="input",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, sid)
        question = cast(
            list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"]
        )[0]
        await service.handle_command(
            client,
            _command(
                request_id="answer",
                command_type="answer_question",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={
                    "question_id": question["question_id"],
                    "answer": "write-answer",
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, sid)
        assert target.read_bytes() == b"answer effect"
        claim = workspace.require_claim(client, sid, version)
        anchors = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])
        answer_anchor = next(a["anchor_id"] for a in anchors if a["content"] == "write-answer")
        await service.inspect_restore(
            client, wid, sid, version, claim.credential, "inspect", answer_anchor
        )
        restored = await service.commit_restore(
            client,
            wid,
            sid,
            version,
            claim.credential,
            "restore",
            answer_anchor,
            RestoreMode.FILES.value,
        )
        snapshot = cast(dict[str, Any], restored["snapshot"])
        assert target.read_bytes() == b"original bytes"
        assert all(m.get("content") != "write-answer" for m in snapshot["messages"])
        assert snapshot["foreground_questions"][0]["question_id"] == question["question_id"]
        assert snapshot["foreground_questions"][0]["answer"] is None
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["answer_received", "resuming", "executing", "answered"])
async def test_answer_publication_failure_does_not_replay_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import json

    from aide.utils.host_filesystem import HOST_FILESYSTEM

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(
            client,
            _command(
                request_id="ask",
                command_type="input",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, sid)
        question = cast(
            list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"]
        )[0]
        command = _command(
            request_id="answer",
            command_type="answer_question",
            workspace_id=wid,
            session_id=sid,
            claim_version=version,
            payload={
                "question_id": question["question_id"],
                "answer": "write-answer",
                "expected_revision": question["revision"],
            },
        )
        original = HOST_FILESYSTEM.atomic_replace_text
        failures = 0

        def fail_phase(path: Path, content: str) -> None:
            nonlocal failures
            if path.name == "state.json" and any(
                q["state"] == phase for q in json.loads(content).get("questions", [])
            ):
                failures += 1
                raise OSError("injected coordination publication failure")
            original(path, content)

        calls_before = provider.stream_calls
        with monkeypatch.context() as fault:
            fault.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_phase)
            if phase in {"answer_received", "resuming"}:
                with pytest.raises(ServiceError):
                    await service.handle_command(client, command)
            else:
                await service.handle_command(client, command)
                await _wait_for_idle(workspace, sid)
        assert failures > 0
        if phase != "answered":
            assert provider.stream_calls == calls_before
            assert not (directory / "answer.txt").exists()
        else:
            assert (directory / "answer.txt").read_text() == "answer effect"
        calls_after_fault = provider.stream_calls
        # Reconciliation and snapshots must never rerun an uncertain fragment.
        for _ in range(3):
            workspace.session_snapshot(sid)
        assert provider.stream_calls == calls_after_fault
        if phase == "answered":
            await service.handle_command(client, command)
            await _wait_for_idle(workspace, sid)
            assert provider.stream_calls == calls_after_fault
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_uncommitted_answer_effect_is_visible_and_cannot_be_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.agent.session.session import Session

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(
            client,
            _command(
                request_id="ask",
                command_type="input",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, sid)
        question = cast(
            list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"]
        )[0]
        command = _command(
            request_id="answer",
            command_type="answer_question",
            workspace_id=wid,
            session_id=sid,
            claim_version=version,
            payload={
                "question_id": question["question_id"],
                "answer": "write-answer",
                "expected_revision": question["revision"],
            },
        )

        def fail_write(self: Session, content: bytes) -> None:
            raise OSError("injected Session commit failure")

        with monkeypatch.context() as fault:
            fault.setattr(Session, "_write_content", fail_write)
            await service.handle_command(client, command)
            await _wait_for_idle(workspace, sid)
            visible = cast(
                list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"]
            )
            assert visible and visible[0]["state"] == "interrupted"
            assert visible[0]["answer"] == "write-answer"
        assert (directory / "answer.txt").read_text() == "answer effect"
        calls = provider.stream_calls
        with pytest.raises(ServiceError):
            await service.handle_command(client, {**command, "request_id": "retry-uncertain"})
        assert provider.stream_calls == calls
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_compaction_preserves_question_and_projects_answer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.agent.session.session import Session

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.question_input_tokens = 7000
    provider.release_answer.set()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        for request, text in [
            ("ask", "ask-foreground-question"),
            ("supplement", "independent supplement"),
        ]:
            await service.handle_command(
                client,
                _command(
                    request_id=request,
                    command_type="input",
                    workspace_id=wid,
                    session_id=sid,
                    claim_version=version,
                    payload={"text": text},
                ),
            )
            await _wait_for_idle(workspace, sid)
        assert Session.load(workspace.workspace_state, sid).last_compacted > 0
        question = cast(
            list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"]
        )[0]
        await service.handle_command(
            client,
            _command(
                request_id="answer",
                command_type="answer_question",
                workspace_id=wid,
                session_id=sid,
                claim_version=version,
                payload={
                    "question_id": question["question_id"],
                    "answer": "reports-directory",
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, sid)
        assert question["question_id"] in provider.user_texts[-1]
        assert question["text"] in provider.user_texts[-1]
        assert provider.question_calls == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preparing", "pending"])
async def test_question_publication_failure_pairs_tools_and_recovers_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import json

    from aide.utils.host_filesystem import HOST_FILESYSTEM

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        original = HOST_FILESYSTEM.atomic_replace_text
        failures = 0

        def fail_phase(path: Path, content: str) -> None:
            nonlocal failures
            if path.name == "state.json" and any(
                q["state"] == phase for q in json.loads(content).get("questions", [])
            ):
                failures += 1
                raise OSError("injected question publication failure")
            original(path, content)

        with monkeypatch.context() as fault:
            fault.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_phase)
            await service.handle_command(
                client,
                _command(
                    request_id="ask",
                    command_type="input",
                    workspace_id=wid,
                    session_id=sid,
                    claim_version=version,
                    payload={"text": "ask-foreground-question"},
                ),
            )
            await _wait_for_idle(workspace, sid)
        assert failures > 0
        assert provider.question_calls == 1
        assert not (directory / "after-question.txt").exists()
        calls_after_run = provider.stream_calls
        snapshot = workspace.session_snapshot(sid)
        history = workspace._loops[sid].loop.session.messages
        assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == [
            "question-call",
            "after-question",
        ]
        visible = cast(list[dict[str, Any]], snapshot["foreground_questions"])
        assert len([q for q in visible if q["state"] == "pending"]) == (
            1 if phase == "pending" else 0
        )
        assert provider.stream_calls == calls_after_run
    finally:
        await service.stop()
