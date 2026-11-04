from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import aide.service.runtime.service as service_runtime
from aide.agent.foreground_work.options import ForegroundQuestionOption
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
        self.question_options: list[ForegroundQuestionOption] = [
            {"id": "reports", "label": "Reports folder"},
            {"id": "archives", "label": "Archives folder"},
        ]
        self.user_texts: list[str] = []
        self.answer_started = asyncio.Event()
        self.release_answer = asyncio.Event()
        self.block_choice_answers = False
        self.write_on_choice = False
        self.write_on_continue = False
        self.choice_answers_started: set[str] = set()
        self.two_choice_answers_started = asyncio.Event()
        self.question_input_tokens = 1
        self.framing_actions: list[str] = []
        self.router_tool_catalogs: list[tuple[str, ...]] = []

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
        action = self.framing_actions.pop(0) if self.framing_actions else "replace"
        if action == "replace":
            return _response(
                '{"action":"replace","task_goal":"answer the input",'
                '"completion_boundary":"return one answer"}'
            )
        return _response(
            json.dumps(
                {"action": action, "task_goal": None, "completion_boundary": None},
                separators=(",", ":"),
            )
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
        tool_names = tuple(
            cast(str, tool.get("name") or cast(dict[str, Any], tool.get("function", {})).get("name"))
            for tool in tools
        )
        self.router_tool_catalogs.append(tool_names)
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
            if (
                "write-answer" in user_text
                or (self.write_on_continue and "Continue the current foreground task." in user_text)
                or (self.write_on_choice and "Reports folder" in user_text)
            ) and not any(message.get("tool_call_id") == "answer-write" for message in messages):
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
            if has_ask_user and (
                "ask-foreground-question" in user_text or "ask-foreground-options" in user_text
            ):
                self.question_calls += 1
                question_arguments = (
                    {
                        "question": "Which output directory should I use?",
                        "options": self.question_options,
                    }
                    if "ask-foreground-options" in user_text
                    else {"question": "Which output directory should I use?"}
                )
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(
                            content="",
                            tool_calls=(
                                ModelToolCall(
                                    "question-call",
                                    "ask_user",
                                    json.dumps(question_arguments),
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
            for label in ("Reports folder", "Archives folder"):
                if label in user_text:
                    self.answer_started.set()
                    self.choice_answers_started.add(label)
                    if len(self.choice_answers_started) == 2:
                        self.two_choice_answers_started.set()
                    if self.block_choice_answers:
                        await self.release_answer.wait()
            yield ModelCompleted(_response("The request is complete."))

        return emit()

    async def close(self) -> None:
        return None


class _QuietSink:
    async def send_event(self, event: dict[str, object]) -> None:
        del event

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


def _legacy_agent_question(
    directory: Path, home: AgentHome, *, options: list[ForegroundQuestionOption] | None = None,
) -> str:
    """Seed an upgrade fixture authored by the pre-Router Agent, without a role field."""
    from uuid import uuid4

    from aide.agent.foreground_work.store import ForegroundWorkStore
    from aide.agent.session.session import Session
    from aide.agent.workspace_state import WorkspaceState
    from aide.utils.time import local_now

    state = WorkspaceState(directory)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state, now=local_now, new_uuid=uuid4)
    task_id, execution_id, question_id = str(uuid4()), str(uuid4()), str(uuid4())
    store = ForegroundWorkStore(state, session.session_id)
    store.begin_task(task_id, execution_id, generation=1, permission_level="workspace-write")
    store.prepare_question(
        task_id=task_id, execution_id=execution_id, question_id=question_id,
        tool_call_id="question-call", text="Which output directory should I use?",
        options=options,
    )
    session.commit_agent_run([
        {"role": "user", "content": "Legacy Agent work", "foreground_task_id": task_id},
        {"role": "assistant", "content": "", "status": "completed", "error": None,
         "token_usage": {"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
         "tool_calls": [{
            "id": "question-call", "name": "ask_user",
            "arguments": '{"question":"Which output directory should I use?"}',
        }]},
        {"role": "tool", "tool_call_id": "question-call", "name": "ask_user",
         "status": "success", "content": "Waiting for an answer.", "artifact": None,
         "foreground_question_id": question_id, "foreground_execution_id": execution_id},
    ], pending_last_compacted=0, pending_action_summary="",
        restore_before=session.capture_restore_before(), restore_run_token=uuid4())
    session.persist()
    store.publish_question(question_id, committed_tool_call_ids={"question-call"})
    legacy = json.loads(store.path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 6
    for task in legacy["tasks"]:
        task.pop("execution_role", None)
    store.path.write_text(json.dumps(legacy), encoding="utf-8")
    return session.session_id


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
    client_kind: str = "cli",
) -> tuple[str, str, int, str]:
    client = await service.register_client(client_kind)
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
@pytest.mark.parametrize("client_kind", ["cli", "web"])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("decision", ["finish", "dispatch"])
async def test_router_answer_keeps_restricted_catalog_and_rejects_business_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_kind: str,
    restart: bool, decision: str,
) -> None:
    class RouterProvider(_QuestionProvider):
        def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
            messages = kwargs["messages"]
            system = messages[0].get("content", "") if messages else ""
            if isinstance(system, str) and system.startswith("You are an Aide Worker."):
                names = {tool.get("name") or tool.get("function", {}).get("name")
                         for tool in kwargs["tools"]}
                assert "read_file" in names
                assert not names.intersection({"dispatch_worker", "finish", "spawn_agent"})

                async def worker_report() -> AsyncIterator[ModelStreamEvent]:
                    yield ModelCompleted(_response("Worker inspected the request."))

                return worker_report()
            user = next((m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"), "")
            worker_report = "worker reports for automatic continuation" in user.lower()
            continuing = "Continue the current foreground task." in user or worker_report
            if "router-answer" not in user and not continuing:
                return super().stream(**kwargs)
            self.router_tool_catalogs.append(tuple(
                tool.get("name") or tool.get("function", {}).get("name")
                for tool in kwargs["tools"]
            ))

            async def emit() -> AsyncIterator[ModelStreamEvent]:
                if worker_report:
                    response = ModelResponse(
                        message=AssistantModelMessage(
                            content="",
                            tool_calls=(ModelToolCall(
                                "worker-report-finish",
                                "finish",
                                '{"response":"Routed answer"}',
                            ),),
                        ),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                    yield ModelCompleted(response)
                    return
                write_id = "forbidden-continue" if continuing else "forbidden-write"
                if not any(m.get("tool_call_id") == write_id for m in messages):
                    calls = (ModelToolCall(
                        write_id, "write_file",
                        '{"path":"router-effect.txt","content":"must not run"}',
                    ),)
                elif decision == "dispatch" and not any(
                    m.get("tool_call_id") == "valid-dispatch" for m in messages
                ):
                    calls = (ModelToolCall(
                        "valid-dispatch", "dispatch_worker",
                        '{"task_id":"worker-task","task":"Inspect the request"}',
                    ),)
                else:
                    calls = (ModelToolCall(
                        "valid-finish", "finish", '{"response":"Routed answer"}',
                    ),)
                yield ModelCompleted(ModelResponse(
                    message=AssistantModelMessage(content="", tool_calls=calls),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls",
                ))

            return emit()

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = RouterProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory, client_kind=client_kind)
        await service.connect_client(client, _QuietSink())
        await service.handle_command(client, _command(
            request_id="router-question", command_type="input", workspace_id=wid,
            session_id=sid, claim_version=version, payload={"text":"ask-foreground-question"},
        ))
        await _wait_for_idle(service.workspace(wid), sid)
        if restart:
            await service.stop()
            service = AgentService(home, ConfigLoader(home).load())
            await service.start()
            client, sid, version, wid = await _open_chat(
                service, directory, session_id=sid, client_kind=client_kind,
            )
            await service.connect_client(client, _QuietSink())
        workspace = service.workspace(wid)
        before = cast(dict[str, Any], workspace.session_snapshot(sid))
        question = before["foreground_questions"][0]
        await service.handle_command(client, _command(
            request_id="router-answer", command_type="answer_question", workspace_id=wid,
            session_id=sid, claim_version=version, payload={
                "question_id": question["question_id"], "answer": "router-answer",
                "expected_revision": question["revision"],
            },
        ))
        await _wait_for_idle(workspace, sid)
        if decision == "dispatch":
            async def wait_for_worker_delivery() -> None:
                while True:
                    workers = workspace.worker_store(sid).load().workers
                    if workers and workers[0].report_delivery_state == "consumed":
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_worker_delivery(), 8)
            await _wait_for_idle(workspace, sid)
        snapshot = cast(dict[str, Any], workspace.session_snapshot(sid))
        assert not (directory / "router-effect.txt").exists()
        results = {m.get("tool_call_id"): m for m in snapshot["messages"]
                   if m.get("role") == "tool"}
        assert results["forbidden-write"]["status"] == "error"
        assert results["valid-finish"]["status"] == "success"
        assert any(m.get("role") == "assistant" and m.get("content") == "Routed answer"
                   for m in snapshot["messages"])
        if decision == "dispatch":
            assert results["valid-dispatch"]["status"] == "success"
        assert snapshot["foreground_questions"] == []
        assert snapshot["foreground_tasks"][0]["task_id"] == question["task_id"]
        assert all(set(names) == {"ask_user", "dispatch_worker", "finish"}
                   for names in provider.router_tool_catalogs if names), provider.router_tool_catalogs

        # An explicit continuation must inherit the same persisted Router role.
        task = snapshot["foreground_tasks"][0]
        await service.handle_command(client, _command(
            request_id="router-continue", command_type="continue_task", workspace_id=wid,
            session_id=sid, claim_version=version,
            payload={"task_id": task["task_id"], "expected_revision": task["revision"]},
        ))
        await _wait_for_idle(workspace, sid)
        snapshot = cast(dict[str, Any], workspace.session_snapshot(sid))
        assert any(m.get("tool_call_id") == "forbidden-continue" and m.get("status") == "error"
                   for m in snapshot["messages"])
        assert not (directory / "router-effect.txt").exists()

        # Restore the answer, then answer the retained Router question again.
        anchor = next(a["anchor_id"] for a in snapshot["restore_anchors"]
                      if a["content"] == "router-answer")
        credential = workspace.require_claim(client, sid, version).credential
        await service.inspect_restore(client, wid, sid, version, credential, "inspect", anchor)
        restored = await service.commit_restore(
            client, wid, sid, version, credential, "restore", anchor, "conversation-only",
        )
        claim = (await service.claim(client, wid, sid))["claim"]
        question = cast(dict[str, Any], restored["snapshot"])["foreground_questions"][0]
        await service.handle_command(client, _command(
            request_id="restored-router-answer", command_type="answer_question", workspace_id=wid,
            session_id=sid, claim_version=cast(int, cast(dict[str, Any], claim)["claim_version"]), payload={
                "question_id": question["question_id"], "answer": "router-answer",
                "expected_revision": question["revision"],
            },
        ))
        await _wait_for_idle(workspace, sid)
        assert not (directory / "router-effect.txt").exists()
        assert all(set(names) == {"ask_user", "dispatch_worker", "finish"}
                   for names in provider.router_tool_catalogs if names), provider.router_tool_catalogs
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("client_kind", ["cli", "web"])
async def test_question_answer_is_durable_claimed_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_kind: str
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace", "keep"]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    replacement: AgentService | None = None
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(
            service, directory, client_kind=client_kind
        )
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
        assert any(
            set(names) == {"ask_user", "dispatch_worker", "finish"}
            for names in provider.router_tool_catalogs
        ), provider.router_tool_catalogs
        assert all(
            not set(names) & {
                "read_file", "write_file", "edit_file", "exec", "schedule",
                "spawn_agent", "wait_agent", "list_agents", "tool_search",
            }
            for names in provider.router_tool_catalogs
            if set(names) == {"ask_user", "dispatch_worker", "finish"}
        )
        assert workspace.session_snapshot(session_id)["session_id"] == session_id
        assert workspace.workspace_id == workspace_id
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
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        revision = cast(int, question["revision"])
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
        unchanged = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        assert unchanged["revision"] == revision
        assert unchanged["state"] == "pending"
        assert unchanged["answer"] is None
        assert unchanged["answer_kind"] is None
        assert unchanged["answer_option_id"] is None

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
        listing = await replacement.list_sessions_page(next_client, next_workspace_id, limit=100)
        summary = next(item for item in cast(list[dict[str, object]], listing["sessions"])
                       if item["id"] == session_id)
        assert summary["foreground_attention"] == {
            "question_count": 1,
            "main_question_count": 1,
            "subagent_question_count": 0,
            "interrupted_task_count": 0,
        }
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
@pytest.mark.parametrize("framing", ["keep", "invalid"])
@pytest.mark.parametrize("has_question", [True, False])
async def test_restart_attention_distinguishes_questions_from_continuable_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framing: str, has_question: bool,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace", framing]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        initial = "ask-foreground-question" if has_question else "ordinary work"
        for index, text in enumerate((initial, "independent supplement")):
            await service.handle_command(client, _command(
                request_id=str(index), command_type="input", workspace_id=wid,
                session_id=sid, claim_version=version, payload={"text": text},
            ))
            await _wait_for_idle(service.workspace(wid), sid)
    finally:
        await service.stop()
    replacement = AgentService(home, ConfigLoader(home).load())
    await replacement.start()
    try:
        client, sid, version, wid = await _open_chat(replacement, directory, session_id=sid)
        listing = await replacement.list_sessions_page(client, wid, limit=100)
        summary = next(item for item in cast(list[dict[str, Any]], listing["sessions"])
                       if item["id"] == sid)
        assert summary["foreground_attention"] == {
            "question_count": int(has_question), "main_question_count": int(has_question),
            "subagent_question_count": 0,
            "interrupted_task_count": int(framing == "invalid" and not has_question),
        }
    finally:
        await replacement.stop()


@pytest.mark.asyncio
async def test_session_listing_reports_a_pending_foreground_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="listed-question-input",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]

        listing = await service.list_sessions_page(client_id, workspace_id, limit=100)
        sessions = cast(list[dict[str, object]], listing["sessions"])
        summary = next(item for item in sessions if item["id"] == session_id)

        assert summary["foreground_attention"] == {
            "question_count": 1,
            "main_question_count": 1,
            "subagent_question_count": 0,
            "interrupted_task_count": 0,
        }
        assert cast(str, question["text"]) not in json.dumps(summary)
        reader = await service.register_client("cli")
        readable = await service.list_sessions_page(reader.client_id, workspace_id, limit=100)
        assert next(item for item in cast(list[dict[str, object]], readable["sessions"])
                    if item["id"] == session_id)["foreground_attention"] == summary["foreground_attention"]
        with pytest.raises(ServiceError):
            await service.get_session_snapshot(reader.client_id, workspace_id, session_id,
                                               claim_version, "")
        with pytest.raises(ServiceError):
            await service.handle_command(reader.client_id, _command(
                request_id="reader-answer", command_type="answer_question",
                workspace_id=workspace_id, session_id=session_id, claim_version=claim_version,
                payload={"question_id": question["question_id"], "answer": "wrong reader",
                         "expected_revision": question["revision"]},
            ))

        provider.release_answer.set()
        await service.handle_command(
            client_id,
            _command(
                request_id="answer-listed-question",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "question_id": question["question_id"],
                    "answer": "reports-directory",
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, session_id)
        refreshed = await service.list_sessions_page(client_id, workspace_id, limit=100)
        refreshed_summary = next(
            item for item in cast(list[dict[str, object]], refreshed["sessions"])
            if item["id"] == session_id
        )
        assert refreshed_summary["foreground_attention"] == {
            "question_count": 0,
            "main_question_count": 0,
            "subagent_question_count": 0,
            "interrupted_task_count": 0,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_keep_framing_retains_task_identity_across_question_answer_and_next_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace", "keep", "keep"]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="keep-task-first-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        first_question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]

        await service.handle_command(
            client_id,
            _command(
                request_id="keep-task-answer",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "question_id": first_question["question_id"],
                    "answer": "continue this task",
                    "expected_revision": first_question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, session_id)

        await service.handle_command(
            client_id,
            _command(
                request_id="keep-task-second-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question again"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        second_question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]

        assert second_question["task_id"] == first_question["task_id"]
        assert second_question["generation"] == first_question["generation"]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["replace", "clear"])
async def test_replace_and_clear_framing_fence_old_task_and_pending_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace", action]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id=f"{action}-old-task-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = workspace.foreground_work_store(session_id).load().pending_questions[0]

        await service.handle_command(
            client_id,
            _command(
                request_id=f"{action}-new-task-input",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "start a separate task"},
            ),
        )
        await _wait_for_idle(workspace, session_id)

        snapshot = workspace.foreground_work_store(session_id).load()
        old_task = next(item for item in snapshot.tasks if item.task_id == question.task_id)
        new_task = snapshot.tasks[-1]
        invalidated_question = snapshot.question(question.question_id)
        assert old_task.state == "superseded"
        assert old_task.generation == question.generation + 1
        assert new_task.task_id != old_task.task_id
        assert invalidated_question is not None and invalidated_question.state == "invalidated"
        listing = await service.list_sessions_page(client_id, workspace_id, limit=100)
        listed_session = next(
            item for item in cast(list[dict[str, object]], listing["sessions"])
            if item["id"] == session_id
        )
        assert listed_session["foreground_attention"] == {
            "question_count": 0,
            "main_question_count": 0,
            "subagent_question_count": 0,
            "interrupted_task_count": 0,
        }
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_manual_skill_owns_new_task_without_inheriting_pending_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "agent-home")
    instruction = home.skills_directory / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nUse the planning instructions.\n",
        encoding="utf-8",
    )
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace"]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="manual-skill-old-task-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        before = workspace.foreground_work_store(session_id).load()
        question = before.pending_questions[0]

        manual_input = "/planner plan a separate task"
        await service.handle_command(
            client_id,
            _command(
                request_id="manual-skill-new-task",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": manual_input},
            ),
        )
        await _wait_for_idle(workspace, session_id)

        after = workspace.foreground_work_store(session_id).load()
        manual_task = after.tasks[-1]
        old_task = next(item for item in after.tasks if item.task_id == question.task_id)
        manual_message = next(
            message
            for message in reversed(workspace._loops[session_id].loop.session.messages)
            if message.get("role") == "user" and message.get("content") == manual_input
        )
        retained_question = after.question(question.question_id)
        assert manual_task.task_id != old_task.task_id
        assert manual_message.get("foreground_task_id") == manual_task.task_id
        assert old_task.state == "waiting_for_answer"
        assert retained_question is not None and retained_question.state == "pending"
        assert retained_question.task_id == old_task.task_id
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_cancel_task_fences_pending_question_and_repeated_cancel_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="cancel-task-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        streams_before_cancel = provider.stream_calls

        await service.handle_command(
            client_id,
            _command(
                request_id="cancel-task-once",
                command_type="cancel_task",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "task_id": question["task_id"],
                    "expected_revision": question["revision"],
                },
            ),
        )
        cancelled_snapshot = workspace.session_snapshot(session_id)
        task = next(
            item
            for item in cast(list[dict[str, object]], cancelled_snapshot["foreground_tasks"])
            if item["task_id"] == question["task_id"]
        )
        assert task["state"] == "cancelled"
        assert task["generation"] == cast(int, question["generation"]) + 1
        assert cancelled_snapshot["foreground_questions"] == []
        listing = await service.list_sessions_page(client_id, workspace_id, limit=100)
        listed_session = next(
            item for item in cast(list[dict[str, object]], listing["sessions"])
            if item["id"] == session_id
        )
        assert listed_session["foreground_attention"] == {
            "question_count": 0,
            "main_question_count": 0,
            "subagent_question_count": 0,
            "interrupted_task_count": 0,
        }

        await service.handle_command(
            client_id,
            _command(
                request_id="cancel-task-again",
                command_type="cancel_task",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "task_id": question["task_id"],
                    "expected_revision": question["revision"],
                },
            ),
        )
        assert workspace.session_snapshot(session_id)["foreground_tasks"] == cancelled_snapshot[
            "foreground_tasks"
        ]
        with pytest.raises(ServiceError) as late_answer:
            await service.handle_command(
                client_id,
                _command(
                    request_id="answer-after-task-cancel",
                    command_type="answer_question",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={
                        "question_id": question["question_id"],
                        "answer": "reports-directory",
                        "expected_revision": question["revision"],
                    },
                ),
            )
        assert late_answer.value.code == "foreground_question_conflict"
        assert provider.stream_calls == streams_before_cancel
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_cancel_task_fences_an_answer_execution_already_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="cancel-in-flight-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = workspace.foreground_work_store(session_id).load().pending_questions[0]

        await service.handle_command(
            client_id,
            _command(
                request_id="answer-before-in-flight-cancel",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "question_id": question.question_id,
                    "answer": "reports-directory",
                    "expected_revision": workspace.foreground_work_store(session_id).load().revision,
                },
            ),
        )
        await asyncio.wait_for(provider.answer_started.wait(), timeout=5)
        current = workspace.foreground_work_store(session_id).load()
        await service.handle_command(
            client_id,
            _command(
                request_id="cancel-after-answer-started",
                command_type="cancel_task",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"task_id": question.task_id, "expected_revision": current.revision},
            ),
        )
        await _wait_for_idle(workspace, session_id)

        cancelled = workspace.foreground_work_store(session_id).load()
        task = next(item for item in cancelled.tasks if item.task_id == question.task_id)
        invalidated = cancelled.question(question.question_id)
        assert task.state == "cancelled"
        assert task.generation == question.generation + 1
        assert invalidated is not None and invalidated.state == "invalidated"
        assert not (directory / "after-question.txt").exists()
    finally:
        provider.release_answer.set()
        await service.stop()


@pytest.mark.asyncio
async def test_continue_task_rejects_pending_question_and_resumes_after_explicit_user_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client_id,
            _command(
                request_id="continue-task-question",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-question"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        streams_before_continue = provider.stream_calls

        with pytest.raises(ServiceError) as pending_continue:
            await service.handle_command(
                client_id,
                _command(
                    request_id="continue-task-before-answer",
                    command_type="continue_task",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={
                        "task_id": question["task_id"],
                        "expected_revision": question["revision"],
                    },
                ),
            )
        assert pending_continue.value.code == "foreground_task_conflict"
        assert provider.stream_calls == streams_before_continue

        await service.handle_command(
            client_id,
            _command(
                request_id="continue-task-answer",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "question_id": question["question_id"],
                    "answer": "continue this task",
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, session_id)
        before = workspace.session_snapshot(session_id)
        task_before = next(
            item
            for item in cast(list[dict[str, object]], before["foreground_tasks"])
            if item["task_id"] == question["task_id"]
        )
        assert task_before["state"] == "completed"

        continued = await service.handle_command(
            client_id,
            _command(
                request_id="continue-task-after-answer",
                command_type="continue_task",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "task_id": question["task_id"],
                    "expected_revision": task_before["revision"],
                },
            ),
        )
        assert cast(dict[str, object], continued["result"])["kind"] == "task_continued"
        await _wait_for_idle(workspace, session_id)
        after_task = next(
            item
            for item in cast(
                list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_tasks"]
            )
            if item["task_id"] == question["task_id"]
        )
        assert after_task["state"] == "completed"
        assert after_task["generation"] == question["generation"]
        assert provider.stream_calls == streams_before_continue + 2
        # Explicit continuation starts a fresh bounded stage. Reading/reconciling
        # the history must not charge the preceding answer again.
        work = workspace.foreground_work_store(session_id).load()
        task = next(item for item in work.tasks if item.task_id == question["task_id"])
        assert task.iteration_budget_used == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_explicit_task_continuation_uses_lower_current_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace"]
    provider.write_on_continue = True
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    legacy_sid = _legacy_agent_question(directory, home)
    await service.start()
    try:
        client_id, session_id, claim_version, workspace_id = await _open_chat(service, directory, session_id=legacy_sid)
        service.client_permission(client_id).select("workspace-write")
        workspace = service.workspace(workspace_id)
        question = workspace.foreground_work_store(session_id).load().pending_questions[0]
        task = next(
            item
            for item in workspace.foreground_work_store(session_id).load().tasks
            if item.task_id == question.task_id
        )
        assert task.permission_level == "workspace-write"

        await service.handle_command(
            client_id,
            _command(
                request_id="permission-task-answer",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "question_id": question.question_id,
                    "answer": "continue this task",
                    "expected_revision": workspace.foreground_work_store(session_id).load().revision,
                },
            ),
        )
        await _wait_for_idle(workspace, session_id)
        service.client_permission(client_id).select("read-only")
        snapshot = workspace.session_snapshot(session_id)
        task_snapshot = next(
            item
            for item in cast(list[dict[str, object]], snapshot["foreground_tasks"])
            if item["task_id"] == question.task_id
        )
        await service.handle_command(
            client_id,
            _command(
                request_id="permission-task-continue",
                command_type="continue_task",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={
                    "task_id": question.task_id,
                    "expected_revision": task_snapshot["revision"],
                },
            ),
        )
        await _wait_until(
            lambda: any(
                event.get("type") == "confirmation.requested"
                and event.get("session_id") == session_id
                for event in service.client(client_id).events
            )
        )
        confirmation = next(
            event
            for event in service.client(client_id).events
            if event.get("type") == "confirmation.requested"
            and event.get("session_id") == session_id
        )
        confirmation_payload = cast(dict[str, object], confirmation["payload"])
        service.decide_confirmation(
            client_id,
            cast(str, confirmation_payload["token"]),
            "declined",
        )
        await _wait_for_idle(workspace, session_id)
        assert not (directory / "answer.txt").exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_question_options_survive_service_restart_and_reclaim(
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
        await service.handle_command(
            client_id,
            _command(
                request_id="question-options-input",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-options"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        assert question["options"] == [
            {"id": "reports", "label": "Reports folder"},
            {"id": "archives", "label": "Archives folder"},
        ]
        calls_before_restart = (provider.stream_calls, provider.complete_calls)
        await service.stop()

        replacement = AgentService(home, ConfigLoader(home).load())
        await replacement.start()
        replacement_client = await replacement.register_client("cli")
        attached = await replacement.attach_workspace(replacement_client.client_id, directory)
        opened = await replacement.open_conversation(
            replacement_client.client_id,
            workspace_id=attached.workspace_id,
            session_id=session_id,
            create_new=False,
        )
        resumed_claim = cast(dict[str, object], opened["claim"])
        restored_question = cast(
            list[dict[str, object]],
            replacement.workspace(attached.workspace_id).session_snapshot(session_id)[
                "foreground_questions"
            ],
        )[0]
        assert isinstance(resumed_claim["claim_version"], int)
        assert restored_question["question_id"] == question["question_id"]
        assert restored_question["options"] == question["options"]
        assert (provider.stream_calls, provider.complete_calls) == calls_before_restart

        provider.release_answer.set()
        await replacement.handle_command(
            replacement_client.client_id,
            _command(
                request_id="answer-restored-question",
                command_type="answer_question",
                workspace_id=attached.workspace_id,
                session_id=session_id,
                claim_version=resumed_claim["claim_version"],
                payload={
                    "question_id": restored_question["question_id"],
                    "answer": "reports-directory",
                    "expected_revision": restored_question["revision"],
                },
            ),
        )
        await _wait_for_idle(replacement.workspace(attached.workspace_id), session_id)
        assert "reports-directory" in provider.user_texts[-1]
        assert provider.question_calls == 1
    finally:
        if replacement is not None:
            await replacement.stop()
        elif getattr(service, "_started", False):
            await service.stop()


@pytest.mark.asyncio
async def test_restart_reconciles_a_committed_foreground_execution_without_rerunning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aide.utils.host_filesystem import HOST_FILESYSTEM

    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    replacement: AgentService | None = None
    try:
        client, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        original_replace = HOST_FILESYSTEM.atomic_replace_text
        failed_charge = False
        failed_task_id: str | None = None

        def fail_execution_charge(path: Path, content: str) -> None:
            nonlocal failed_charge, failed_task_id
            if path.name == "state.json" and not failed_charge:
                state = json.loads(content)
                for task in state.get("tasks", []):
                    execution_id = task.get("execution_id")
                    charged_ids = task.get("charged_execution_ids", [])
                    if (
                        task.get("state") == "running"
                        and isinstance(execution_id, str)
                        and execution_id in charged_ids
                    ):
                        failed_charge = True
                        failed_task_id = task.get("task_id")
                        raise OSError("injected execution charge publication failure")
            original_replace(path, content)

        with monkeypatch.context() as fault:
            fault.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail_execution_charge)
            await service.handle_command(
                client,
                _command(
                    request_id="committed-execution",
                    command_type="input",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={"text": "ordinary foreground execution"},
                ),
            )
            await _wait_for_idle(workspace, session_id)

        assert failed_charge
        assert isinstance(failed_task_id, str)
        await service.stop()
        before_restart_calls = provider.stream_calls + provider.complete_calls

        replacement = AgentService(home, ConfigLoader(home).load())
        await replacement.start()
        client, resumed_session_id, claim_version, workspace_id = await _open_chat(
            replacement, directory, session_id=session_id
        )
        assert resumed_session_id == session_id
        workspace = replacement.workspace(workspace_id)
        recovered = workspace.session_snapshot(session_id)
        recovered_task = next(
            task
            for task in cast(list[dict[str, object]], recovered["foreground_tasks"])
            if task["task_id"] == failed_task_id
        )
        assert recovered_task["state"] == "completed"
        persisted_work = workspace.foreground_work_store(session_id).load()
        persisted_task = next(
            task for task in persisted_work.tasks if task.task_id == failed_task_id
        )
        assert persisted_task.iteration_budget_used == 1
        assert provider.stream_calls + provider.complete_calls == before_restart_calls
        assert workspace.session_snapshot(session_id)["foreground_tasks"] == recovered[
            "foreground_tasks"
        ]
        persisted_after_snapshot = workspace.foreground_work_store(session_id).load()
        task_after_snapshot = next(
            task
            for task in persisted_after_snapshot.tasks
            if task.task_id == failed_task_id
        )
        assert task_after_snapshot.iteration_budget_used == 1
    finally:
        if replacement is not None:
            await replacement.stop()
        elif getattr(service, "_started", False):
            await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_labels", [False, True])
async def test_choice_answer_uses_stable_id_and_invalid_choice_preserves_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_labels: bool
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    if duplicate_labels:
        provider.question_options[1]["label"] = "Reports folder"
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    replacement: AgentService | None = None
    try:
        client, session_id, claim_version, workspace_id = await _open_chat(service, directory)
        workspace = service.workspace(workspace_id)
        await service.handle_command(
            client,
            _command(
                request_id="choice-question-input",
                command_type="input",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=claim_version,
                payload={"text": "ask-foreground-options"},
            ),
        )
        await _wait_for_idle(workspace, session_id)
        question = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        calls_before_rejection = provider.stream_calls
        with pytest.raises(ServiceError):
            await service.handle_command(
                client,
                _command(
                    request_id="choice-invalid",
                    command_type="answer_question",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={
                        "question_id": question["question_id"],
                        "choice_id": "missing-option",
                        "expected_revision": question["revision"],
                    },
                ),
            )
        unchanged = cast(
            list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_questions"]
        )[0]
        assert unchanged["revision"] == question["revision"]
        assert unchanged["state"] == "pending"
        assert unchanged["answer"] is None
        assert provider.stream_calls == calls_before_rejection

        old_claim = workspace.require_claim(client, session_id, claim_version)
        await service.release_claim(
            client, workspace_id, session_id, claim_version, old_claim.credential
        )
        current_client = await service.register_client("web")
        await service.attach_workspace(current_client.client_id, directory)
        reopened = await service.open_conversation(
            current_client.client_id,
            workspace_id=workspace_id,
            session_id=session_id,
            create_new=False,
        )
        current_claim = cast(dict[str, object], reopened["claim"])
        current_claim_version = cast(int, current_claim["claim_version"])
        with pytest.raises(ServiceError) as stale_claim:
            await service.handle_command(
                client,
                _command(
                    request_id="choice-stale-claim",
                    command_type="answer_question",
                    workspace_id=workspace_id,
                    session_id=session_id,
                    claim_version=claim_version,
                    payload={
                        "question_id": question["question_id"],
                        "choice_id": "reports",
                        "expected_revision": question["revision"],
                    },
                ),
            )
        assert stale_claim.value.code == "stale_claim"
        assert (
            cast(
                list[dict[str, object]],
                workspace.session_snapshot(session_id)["foreground_questions"],
            )[0]["state"]
            == "pending"
        )

        accepted = await service.handle_command(
            current_client.client_id,
            _command(
                request_id="choice-valid",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_id,
                claim_version=current_claim_version,
                payload={
                    "question_id": question["question_id"],
                    "choice_id": "reports",
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, session_id)
        answer_run = cast(str, cast(dict[str, object], accepted["result"])["run_id"])
        assert isinstance(answer_run, str)
        history = cast(list[dict[str, object]], workspace.session_snapshot(session_id)["messages"])
        answer_message = next(
            message
            for message in history
            if message.get("role") == "user"
            and message.get("foreground_question_id") == question["question_id"]
        )
        assert answer_message["content"] == "Reports folder"
        assert "Reports folder" in provider.user_texts[-1]
        assert "Choice ID: reports" in provider.user_texts[-1]
        assert provider.stream_calls == calls_before_rejection + 1
        for conflicting_answer in ({"answer": "Reports folder"}, {"choice_id": "archives"}):
            with pytest.raises(ServiceError):
                await service.handle_command(
                    current_client.client_id,
                    _command(
                        request_id="choice-valid",
                        command_type="answer_question",
                        workspace_id=workspace_id,
                        session_id=session_id,
                        claim_version=current_claim_version,
                        payload={
                            "question_id": question["question_id"],
                            "expected_revision": question["revision"],
                            **conflicting_answer,
                        },
                    ),
                )
        duplicates = await asyncio.gather(
            *(
                service.handle_command(
                    current_client.client_id,
                    _command(
                        request_id=f"choice-retry-{index}",
                        command_type="answer_question",
                        workspace_id=workspace_id,
                        session_id=session_id,
                        claim_version=current_claim_version,
                        payload={
                            "question_id": question["question_id"],
                            "choice_id": "reports",
                            "expected_revision": question["revision"],
                        },
                    ),
                )
                for index in range(10)
            )
        )
        assert {
            cast(str, cast(dict[str, object], result["result"])["run_id"]) for result in duplicates
        } == {answer_run}
        assert provider.stream_calls == calls_before_rejection + 1

        await service.stop()
        replacement = AgentService(home, ConfigLoader(home).load())
        await replacement.start()
        next_client, same_session_id, next_claim_version, next_workspace_id = await _open_chat(
            replacement, directory, session_id=session_id
        )
        replayed = await replacement.handle_command(
            next_client,
            _command(
                request_id="choice-valid",
                command_type="answer_question",
                workspace_id=next_workspace_id,
                session_id=same_session_id,
                claim_version=next_claim_version,
                payload={
                    "question_id": question["question_id"],
                    "choice_id": "reports",
                    "expected_revision": question["revision"],
                },
            ),
        )
        assert cast(dict[str, object], replayed["result"])["run_id"] == answer_run
        assert provider.stream_calls == calls_before_rejection + 1
    finally:
        if replacement is not None:
            await replacement.stop()
        elif getattr(service, "_started", False):
            await service.stop()


@pytest.mark.asyncio
async def test_choice_answers_are_isolated_across_concurrent_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "agent-home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.block_choice_answers = True
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client_one, session_one, version_one, workspace_id = await _open_chat(service, directory)
        client_two, session_two, version_two, second_workspace_id = await _open_chat(
            service, directory
        )
        assert workspace_id == second_workspace_id
        assert session_one != session_two
        workspace = service.workspace(workspace_id)

        await asyncio.gather(
            *(
                service.handle_command(
                    client_id,
                    _command(
                        request_id=f"ask-options-{index}",
                        command_type="input",
                        workspace_id=workspace_id,
                        session_id=session_id,
                        claim_version=claim_version,
                        payload={"text": "ask-foreground-options"},
                    ),
                )
                for index, (client_id, session_id, claim_version) in enumerate(
                    (
                        (client_one, session_one, version_one),
                        (client_two, session_two, version_two),
                    )
                )
            )
        )
        await asyncio.gather(
            _wait_for_idle(workspace, session_one),
            _wait_for_idle(workspace, session_two),
        )
        question_one = cast(
            list[dict[str, object]], workspace.session_snapshot(session_one)["foreground_questions"]
        )[0]
        question_two = cast(
            list[dict[str, object]], workspace.session_snapshot(session_two)["foreground_questions"]
        )[0]

        first_answer = await service.handle_command(
            client_one,
            _command(
                request_id="answer-options-one",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_one,
                claim_version=version_one,
                payload={
                    "question_id": question_one["question_id"],
                    "choice_id": "reports",
                    "expected_revision": question_one["revision"],
                },
            ),
        )
        await asyncio.wait_for(provider.answer_started.wait(), timeout=5)
        second_pending = cast(
            list[dict[str, object]], workspace.session_snapshot(session_two)["foreground_questions"]
        )[0]
        assert second_pending["state"] == "pending"
        assert second_pending["answer"] is None

        second_answer = await service.handle_command(
            client_two,
            _command(
                request_id="answer-options-two",
                command_type="answer_question",
                workspace_id=workspace_id,
                session_id=session_two,
                claim_version=version_two,
                payload={
                    "question_id": question_two["question_id"],
                    "choice_id": "archives",
                    "expected_revision": question_two["revision"],
                },
            ),
        )
        first_run = cast(str, cast(dict[str, object], first_answer["result"])["run_id"])
        second_run = cast(str, cast(dict[str, object], second_answer["result"])["run_id"])
        assert first_run != second_run
        await asyncio.wait_for(provider.two_choice_answers_started.wait(), timeout=5)
        first_live = cast(dict[str, object], workspace.session_snapshot(session_one)["live_state"])
        second_live = cast(dict[str, object], workspace.session_snapshot(session_two)["live_state"])
        assert first_run in {
            item["run_id"] for item in cast(list[dict[str, object]], first_live["runs"])
        }
        assert second_run in {
            item["run_id"] for item in cast(list[dict[str, object]], second_live["runs"])
        }

        provider.release_answer.set()
        await asyncio.gather(
            _wait_for_idle(workspace, session_one),
            _wait_for_idle(workspace, session_two),
        )
        messages_one = cast(
            list[dict[str, object]], workspace.session_snapshot(session_one)["messages"]
        )
        messages_two = cast(
            list[dict[str, object]], workspace.session_snapshot(session_two)["messages"]
        )
        assert any(
            item.get("role") == "user"
            and item.get("foreground_question_id") == question_one["question_id"]
            and item.get("content") == "Reports folder"
            for item in messages_one
        )
        assert any(
            item.get("role") == "user"
            and item.get("foreground_question_id") == question_two["question_id"]
            and item.get("content") == "Archives folder"
            for item in messages_two
        )
        assert provider.choice_answers_started == {"Reports folder", "Archives folder"}
    finally:
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
    provider.framing_actions = ["replace", "keep"]
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
        listing = await service.list_sessions_page(client, wid, limit=100)
        assert all(
            item["id"] != sid
            for item in cast(list[dict[str, object]], listing["sessions"])
        )
        with pytest.raises(ForegroundWorkConflict):
            old_store.begin_task("late-task", "late-execution", generation=1)
        assert not old_store.path.parent.exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer_mode", ["text", "choice"])
async def test_answer_restore_returns_conversation_question_and_real_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer_mode: str
) -> None:
    from aide.agent.session.restore import RestoreMode

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    target = directory / "answer.txt"
    target.write_bytes(b"original bytes")
    provider = _QuestionProvider()
    provider.write_on_choice = answer_mode == "choice"
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    legacy_sid = _legacy_agent_question(directory, home, options=provider.question_options if answer_mode == "choice" else None)
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory, session_id=legacy_sid)
        workspace = service.workspace(wid)
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
                    **(
                        {"choice_id": "reports"}
                        if answer_mode == "choice"
                        else {"answer": "write-answer"}
                    ),
                    "expected_revision": question["revision"],
                },
            ),
        )
        await _wait_for_idle(workspace, sid)
        assert target.read_bytes() == b"answer effect"
        claim = workspace.require_claim(client, sid, version)
        anchors = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])
        answer_text = "Reports folder" if answer_mode == "choice" else "write-answer"
        answer_anchor = next(a["anchor_id"] for a in anchors if a["content"] == answer_text)
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
        assert all(m.get("content") != answer_text for m in snapshot["messages"])
        restored_question = snapshot["foreground_questions"][0]
        assert restored_question["question_id"] == question["question_id"]
        assert restored_question["state"] == "pending"
        assert restored_question["answer"] is None
        assert restored_question["answer_kind"] is None
        assert restored_question["answer_option_id"] is None
        assert restored_question["options"] == question["options"]
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
    legacy_sid = _legacy_agent_question(directory, home, options=None)
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory, session_id=legacy_sid)
        workspace = service.workspace(wid)
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
    legacy_sid = _legacy_agent_question(directory, home, options=None)
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory, session_id=legacy_sid)
        workspace = service.workspace(wid)
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
    provider.framing_actions = ["replace", "keep"]
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
        history = cast(list[dict[str, object]], snapshot["messages"])
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

@pytest.mark.asyncio
async def test_invalid_framing_interrupts_prior_task_until_explicit_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.framing_actions = ["replace", "invalid"]
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        for index in range(2):
            await service.handle_command(client, _command(
                request_id=f"input-{index}", command_type="input", workspace_id=wid,
                session_id=sid, claim_version=version, payload={"text": "ordinary input"},
            ))
            await _wait_for_idle(workspace, sid)
        tasks = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])
        assert tasks[0]["state"] == "interrupted"
        await service.continue_foreground_task(client, wid, sid, version,
                                              tasks[0]["task_id"], tasks[0]["revision"], "continue")
        await _wait_for_idle(workspace, sid)
        assert "Continue the current foreground task." in provider.user_texts[-1]
    finally:
        await service.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel_task", "continue_task"])
async def test_task_publication_failure_admits_no_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from aide.utils.host_filesystem import HOST_FILESYSTEM

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(client, _command(
            request_id="initial", command_type="input", workspace_id=wid,
            session_id=sid, claim_version=version, payload={"text": "ordinary input"},
        ))
        await _wait_for_idle(workspace, sid)
        task = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])[-1]
        calls = provider.stream_calls
        original = HOST_FILESYSTEM.atomic_replace_text

        def fail(path: Path, content: str) -> None:
            if path.name == "state.json":
                raise OSError("injected Task publication failure")
            original(path, content)

        with monkeypatch.context() as fault:
            fault.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail)
            with pytest.raises(ServiceError) as failure:
                await service.handle_command(client, _command(
                    request_id="operation", command_type=operation, workspace_id=wid,
                    session_id=sid, claim_version=version,
                    payload={"task_id": task["task_id"], "expected_revision": task["revision"]},
                ))
            assert failure.value.code == "persistence_error"
        await _wait_for_idle(workspace, sid)
        assert provider.stream_calls == calls
        assert workspace.session_snapshot(sid)["foreground_tasks"] == [task]
    finally:
        await service.stop()

@pytest.mark.asyncio
async def test_cancel_queued_continuation_does_not_stop_another_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    skill = home.skills_directory / "planner" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: planner\ndescription: Plan\n---\nPlan work.\n", encoding="utf-8")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        async def command(kind: str, request: str, payload: dict[str, object]) -> dict[str, object]:
            return await service.handle_command(client, _command(
                request_id=request, command_type=kind, workspace_id=wid,
                session_id=sid, claim_version=version, payload=payload,
            ))
        await command("input", "ask", {"text": "ask-foreground-question"})
        await _wait_for_idle(workspace, sid)
        await command("input", "manual", {"text": "/planner independent work"})
        await _wait_for_idle(workspace, sid)
        snapshot = workspace.session_snapshot(sid)
        question = cast(list[dict[str, Any]], snapshot["foreground_questions"])[0]
        task = cast(list[dict[str, Any]], snapshot["foreground_tasks"])[-1]
        await command("answer_question", "answer", {
            "question_id": question["question_id"], "answer": "reports-directory",
            "expected_revision": question["revision"],
        })
        await asyncio.wait_for(provider.answer_started.wait(), timeout=5)
        task = next(t for t in cast(list[dict[str, Any]],
                    workspace.session_snapshot(sid)["foreground_tasks"]) if t["task_id"] == task["task_id"])
        await command("continue_task", "queued", {
            "task_id": task["task_id"], "expected_revision": task["revision"],
        })
        task = next(t for t in cast(list[dict[str, Any]],
                    workspace.session_snapshot(sid)["foreground_tasks"]) if t["task_id"] == task["task_id"])
        calls = provider.stream_calls
        await command("cancel_task", "cancel", {
            "task_id": task["task_id"], "expected_revision": task["revision"],
        })
        provider.release_answer.set()
        await _wait_for_idle(workspace, sid)
        assert provider.stream_calls == calls
        tasks = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_tasks"])
        assert next(t for t in tasks if t["task_id"] == question["task_id"])["state"] == "completed"
        assert next(t for t in tasks if t["task_id"] == task["task_id"])["state"] == "cancelled"
    finally:
        provider.release_answer.set()
        await service.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "replace"])
async def test_revoked_task_stays_revoked_after_restore_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from aide.agent.session.restore import RestoreMode

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(client, _command(
            request_id="ask", command_type="input", workspace_id=wid, session_id=sid,
            claim_version=version, payload={"text": "ask-foreground-question"},
        ))
        await _wait_for_idle(workspace, sid)
        question = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])[0]
        if ending == "cancel":
            await service.cancel_foreground_task(client, wid, sid, version,
                                                question["task_id"], question["revision"], "cancel")
        await service.handle_command(client, _command(
            request_id="new", command_type="input", workspace_id=wid, session_id=sid,
            claim_version=version, payload={"text": "separate work"},
        ))
        await _wait_for_idle(workspace, sid)
        anchor = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["restore_anchors"])[-1]
        claim = workspace.require_claim(client, sid, version)
        old_store = workspace.foreground_work_store(sid)
        await service.inspect_restore(client, wid, sid, version, claim.credential,
                                      "inspect", anchor["anchor_id"])
        await service.commit_restore(client, wid, sid, version, claim.credential,
                                     "restore", anchor["anchor_id"], RestoreMode.CONVERSATION_ONLY.value)
        assert not old_store.is_active_generation(question["task_id"], question["generation"])
        calls = provider.stream_calls
    finally:
        await service.stop()
    replacement = AgentService(home, ConfigLoader(home).load())
    await replacement.start()
    try:
        client, sid, version, wid = await _open_chat(replacement, directory, session_id=sid)
        snapshot = replacement.workspace(wid).session_snapshot(sid)
        task = next(t for t in cast(list[dict[str, Any]], snapshot["foreground_tasks"])
                    if t["task_id"] == question["task_id"])
        assert task["state"] == ("cancelled" if ending == "cancel" else "superseded")
        assert snapshot["foreground_questions"] == []
        assert provider.stream_calls == calls
        with pytest.raises(ServiceError):
            await replacement.continue_foreground_task(client, wid, sid, version,
                                                       task["task_id"], task["revision"], "late")
        assert provider.stream_calls == calls
    finally:
        await replacement.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("answer_first", [False, True])
async def test_answer_cancel_race_and_cached_request_revalidate_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer_first: bool
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _QuestionProvider()
    provider.release_answer.set()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client, sid, version, wid = await _open_chat(service, directory)
        workspace = service.workspace(wid)
        await service.handle_command(client, _command(
            request_id="ask", command_type="input", workspace_id=wid, session_id=sid,
            claim_version=version, payload={"text": "ask-foreground-question"},
        ))
        await _wait_for_idle(workspace, sid)
        question = cast(list[dict[str, Any]], workspace.session_snapshot(sid)["foreground_questions"])[0]
        commands = [_command(
            request_id="answer", command_type="answer_question", workspace_id=wid, session_id=sid,
            claim_version=version, payload={"question_id": question["question_id"],
                                          "answer": "reports-directory", "expected_revision": question["revision"]},
        ), _command(
            request_id="cancel", command_type="cancel_task", workspace_id=wid, session_id=sid,
            claim_version=version, payload={"task_id": question["task_id"], "expected_revision": question["revision"]},
        )]
        if not answer_first:
            commands.reverse()
        calls_before_race = provider.stream_calls
        results = await asyncio.gather(*(service.handle_command(client, c) for c in commands),
                                       return_exceptions=True)
        assert sum(isinstance(r, dict) for r in results) == 1
        assert sum(isinstance(r, ServiceError) for r in results) == 1
        await _wait_for_idle(workspace, sid)
        assert provider.stream_calls == calls_before_race + int(answer_first)
        credential = workspace.require_claim(client, sid, version).credential
        await service.release_claim(client, wid, sid, version, credential)
        await service.claim(client, wid, sid)
        calls = provider.stream_calls
        with pytest.raises(ServiceError) as stale:
            await service.handle_command(client, commands[0])
        assert stale.value.code == "stale_claim"
        assert provider.stream_calls == calls
    finally:
        await service.stop()
