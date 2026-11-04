from __future__ import annotations

import asyncio
import json
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
from aide.service.runtime import AgentService
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


class _ParallelWorkerProvider:
    def __init__(self) -> None:
        self.worker_started = 0
        self.framing_calls = 0
        self.two_workers_started = asyncio.Event()
        self.release_workers = asyncio.Event()
        self.report_continuations = 0
        self.report_continuation_started = asyncio.Event()
        self.release_report_continuation = asyncio.Event()
        self.followup_started = asyncio.Event()
        self.release_followup = asyncio.Event()
        self.closed = False
        self.replace_followup = False

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
        del tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        system = messages[0].get("content", "") if messages else ""
        if isinstance(system, str) and system.startswith("Generate a concise title"):
            return _response("Parallel Worker task")
        self.framing_calls += 1
        action = "replace" if self.framing_calls == 1 or self.replace_followup else "keep"
        return _response(
            json.dumps(
                {
                    "action": action,
                    "task_goal": "complete both checks" if action == "replace" else None,
                    "completion_boundary": "summarize the results" if action == "replace" else None,
                },
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
        del tools, model, max_output, temperature, reasoning_effort, timeout, continuation
        system = messages[0].get("content", "") if messages else ""
        last_user = next(
            (item.get("content", "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        user_text = last_user if isinstance(last_user, str) else ""

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if isinstance(system, str) and system.startswith("You are an Aide Worker."):
                self.worker_started += 1
                if self.worker_started >= 2:
                    self.two_workers_started.set()
                await self.release_workers.wait()
                yield ModelCompleted(_response("Independent check completed."))
                return

            if "Worker reports for automatic continuation:" in user_text:
                self.report_continuations += 1
                self.report_continuation_started.set()
                await self.release_report_continuation.wait()
                tool_call = ModelToolCall(
                    "report-finish",
                    "finish",
                    '{"response":"Both independent checks are complete."}',
                )
            elif "Follow-up update" in user_text:
                self.followup_started.set()
                await self.release_followup.wait()
                tool_call = ModelToolCall(
                    "followup-finish",
                    "finish",
                    '{"response":"Follow-up handled."}',
                )
            elif not any(
                message.get("tool_call_id") in {"dispatch-first", "dispatch-second"}
                for message in messages
            ):
                tool_calls = (
                    ModelToolCall(
                        "dispatch-first",
                        "dispatch_worker",
                        json.dumps({"task_id": "first-check", "task": "Inspect the first source"}),
                    ),
                    ModelToolCall(
                        "dispatch-second",
                        "dispatch_worker",
                        json.dumps({"task_id": "second-check", "task": "Inspect the second source"}),
                    ),
                )
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(content="", tool_calls=tool_calls),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                )
                return
            else:
                tool_call = ModelToolCall(
                    "initial-finish",
                    "finish",
                    '{"response":"The work is complete."}',
                )

            yield ModelCompleted(
                ModelResponse(
                    message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls",
                )
            )

        return emit()

    async def close(self) -> None:
        self.closed = True


class _WorkerSubAgentProvider(_ParallelWorkerProvider):
    def __init__(self, *, request_subagent_confirmation: bool = True) -> None:
        super().__init__()
        self.request_subagent_confirmation = request_subagent_confirmation
        self.release_subagent = asyncio.Event()
        self.subagent_started = asyncio.Event()
        self.two_subagents_started = asyncio.Event()
        self.subagent_started_labels: set[str] = set()
        self.followup_started = asyncio.Event()
        self.release_followup = asyncio.Event()
        self.router_report_texts: list[str] = []
        self.worker_tool_names: dict[str, set[str]] = {}
        self.subagent_tool_names: set[str] = set()
        self.calls: list[str] = []

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
        system = messages[0].get("content", "") if messages else ""
        user_text = "\n".join(
            content
            for item in messages
            if item.get("role") == "user"
            and isinstance((content := item.get("content")), str)
        )
        tool_names = {
            schema["function"]["name"]
            for schema in tools
            if isinstance(schema.get("function"), dict)
            and isinstance(schema["function"].get("name"), str)
        }

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            child_label = next(
                (label for label in ("first", "second")
                 if f"Return the private result {label}." in user_text),
                None,
            )
            if child_label is not None:
                self.calls.append(f"subagent:{child_label}")
                self.subagent_tool_names.update(tool_names)
                self.subagent_started_labels.add(child_label)
                self.subagent_started.set()
                if len(self.subagent_started_labels) >= 2:
                    self.two_subagents_started.set()
                await self.release_subagent.wait()
                call_id = f"subagent-write-{child_label}"
                tool_results = {
                    item.get("tool_call_id"): item
                    for item in messages
                    if item.get("role") == "tool"
                }
                if self.request_subagent_confirmation and call_id not in tool_results:
                    call = ModelToolCall(
                        call_id,
                        "write_file",
                        json.dumps({
                            "path": f"subagent-{child_label}.txt",
                            "content": f"SubAgent {child_label} result",
                        }),
                    )
                    yield ModelCompleted(ModelResponse(
                        message=AssistantModelMessage(content="", tool_calls=(call,)),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    ))
                    return
                yield ModelCompleted(_response(f"SUBAGENT-PRIVATE-RESULT-{child_label.upper()}"))
                return

            if isinstance(system, str) and system.startswith("You are an Aide Worker."):
                delegated = "SubAgent" in user_text
                worker_name = "delegating" if delegated else "file-reader"
                self.calls.append(f"worker:{worker_name}:{','.join(sorted(item.get('tool_call_id', '') for item in messages if item.get('role') == 'tool'))}")
                self.worker_tool_names[worker_name] = set(tool_names)
                self.worker_started += 1
                if self.worker_started >= 2:
                    self.two_workers_started.set()
                await self.two_workers_started.wait()
                await self.release_workers.wait()
                tool_results = {
                    item.get("tool_call_id"): item
                    for item in messages
                    if item.get("role") == "tool"
                }
                if delegated:
                    if "worker-search-spawn" not in tool_results:
                        call = ModelToolCall(
                            "worker-search-spawn",
                            "tool_search",
                            json.dumps({"query": "spawn_agent"}),
                        )
                    elif "worker-spawn" not in tool_results:
                        call = ModelToolCall(
                            "worker-spawn",
                            "spawn_agent",
                            json.dumps({
                                "title": "Inspect first detail",
                                "task": "Return the private result first.",
                            }),
                        )
                    elif "worker-spawn-second" not in tool_results:
                        call = ModelToolCall(
                            "worker-spawn-second",
                            "spawn_agent",
                            json.dumps({
                                "title": "Inspect second detail",
                                "task": "Return the private result second.",
                            }),
                        )
                    elif "worker-search-wait" not in tool_results:
                        call = ModelToolCall(
                            "worker-search-wait",
                            "tool_search",
                            json.dumps({"query": "wait_agent"}),
                        )
                    elif "worker-list" not in tool_results:
                        call = ModelToolCall("worker-list", "list_agents", "{}")
                    else:
                        child_ids = [
                            json.loads(str(tool_results[key].get("content", "{}")))["agent_id"]
                            for key in ("worker-spawn", "worker-spawn-second")
                        ]
                        listed = json.loads(str(tool_results["worker-list"].get("content", "{}")))
                        assert {item["agent_id"] for item in listed["items"]} == set(child_ids)
                        consumed = {
                            item["agent_id"]: item
                            for key, value in tool_results.items()
                            if isinstance(key, str) and key.startswith("worker-wait-")
                            for item in json.loads(str(value.get("content", "[]")))
                        }
                        remaining = [child_id for child_id in child_ids if child_id not in consumed]
                        if remaining:
                            call = ModelToolCall(
                                f"worker-wait-{len(consumed)}", "wait_agent",
                                json.dumps({"agent_ids": remaining}),
                            )
                        else:
                            assert all(item["status"] == "completed" for item in consumed.values())
                            yield ModelCompleted(_response(
                                "Worker incorporated " + " and ".join(
                                    consumed[child_id]["result"] for child_id in child_ids
                                ) + "."
                            ))
                            return
                elif "file-read" not in tool_results:
                    call = ModelToolCall(
                        "file-read",
                        "read_file",
                        json.dumps({"path": "worker-input.txt"}),
                    )
                else:
                    yield ModelCompleted(_response("Worker read WORKER-FILE-CONTENT."))
                    return
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(content="", tool_calls=(call,)),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                )
                return

            if "Worker reports for automatic continuation:" in user_text:
                self.calls.append("router:reports")
                self.report_continuations += 1
                self.router_report_texts.append(user_text)
                self.report_continuation_started.set()
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(
                            content="",
                            tool_calls=(ModelToolCall(
                                "report-finish",
                                "finish",
                                '{"response":"Both Worker reports are complete."}',
                            ),),
                        ),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                )
                return

            if "Follow-up update" in user_text:
                self.followup_started.set()
                await self.release_followup.wait()
                yield ModelCompleted(_response("Follow-up handled."))
                return

            if any(
                item.get("tool_call_id")
                in {"dispatch-subagent-worker", "dispatch-file-worker"}
                for item in messages
                if item.get("role") == "tool"
            ):
                call = ModelToolCall(
                    "initial-finish",
                    "finish",
                    '{"response":"The Router is waiting for Worker reports."}',
                )
                yield ModelCompleted(
                    ModelResponse(
                        message=AssistantModelMessage(content="", tool_calls=(call,)),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="tool_calls",
                    )
                )
                return

            self.calls.append(f"router:dispatch:{user_text[:120]!r}")
            tool_calls = (
                ModelToolCall(
                    "dispatch-subagent-worker",
                    "dispatch_worker",
                    json.dumps({
                        "task_id": "subagent-review",
                        "task": "Delegate to one SubAgent, wait for its result, and report it.",
                    }),
                ),
                ModelToolCall(
                    "dispatch-file-worker",
                    "dispatch_worker",
                    json.dumps({
                        "task_id": "file-review",
                        "task": "Read worker-input.txt and report its contents.",
                    }),
                ),
            )
            yield ModelCompleted(
                ModelResponse(
                    message=AssistantModelMessage(content="", tool_calls=tool_calls),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls",
                )
            )

        return emit()


class _SilentSink:
    async def send_event(self, _event: dict[str, object]) -> None:
        return None


def _home(path: Path) -> AgentHome:
    home = AgentHome(path)
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    return home


async def _wait_until(predicate: Any) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("client_kind", ["cli", "web"])
@pytest.mark.parametrize("receipt_write_failure", [False, True])
@pytest.mark.asyncio
async def test_parallel_workers_overlap_and_reports_produce_one_router_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    client_kind: str, receipt_write_failure: bool,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ParallelWorkerProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client(client_kind)
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id,
            workspace_id=workspace.workspace_id,
            create_new=True,
        )
        session_id = opened["session_id"]
        claim = cast(dict[str, object], opened["claim"])
        claim_version = cast(int, claim["claim_version"])
        await service.handle_command(
            client.client_id,
            {
                "request_id": "parallel-worker-input",
                "type": "input",
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": claim_version,
                "payload": {"text": "Run both independent checks."},
            },
        )

        await asyncio.wait_for(provider.two_workers_started.wait(), timeout=10)
        assert provider.worker_started == 2
        if receipt_write_failure:
            store = workspace.worker_store(session_id)
            acknowledge = store.acknowledge_report
            failed = False

            def fail_once(worker_id: str, receipt_id: str) -> bool:
                nonlocal failed
                if not failed:
                    failed = True
                    raise OSError("injected receipt index failure")
                return acknowledge(worker_id, receipt_id)

            monkeypatch.setattr(store, "acknowledge_report", fail_once)
        provider.release_workers.set()
        try:
            await asyncio.wait_for(provider.report_continuation_started.wait(), timeout=10)
        except TimeoutError as error:
            task_statuses = workspace.session_snapshot(session_id)["foreground_tasks"]
            worker_statuses = [
                (worker.task_id, worker.state)
                for worker in workspace.worker_store(session_id).load().workers
            ]
            raise AssertionError(
                "Worker reports did not reach the Router: "
                f"tasks={task_statuses!r}; workers={worker_statuses!r}"
            ) from error

        snapshot = workspace.session_snapshot(session_id)
        live_state = cast(dict[str, object], snapshot["live_state"])
        runs = cast(list[dict[str, object]], live_state["runs"])
        assert any(run.get("source") == "automatic" for run in runs)
        for _ in range(10):
            workspace.wake_foreground_continuations()
        provider.release_report_continuation.set()
        await _wait_until(
            lambda: not cast(
                list[dict[str, object]],
                cast(dict[str, object], workspace.session_snapshot(session_id)["live_state"])["runs"],
            )
        )

        snapshot = workspace.session_snapshot(session_id)
        messages = cast(list[dict[str, object]], snapshot["messages"])
        for _ in range(10):
            workspace.wake_foreground_continuations()
        await asyncio.sleep(0.1)
        assert provider.report_continuations == 1
        assert all(
            worker.report_delivery_state == "consumed"
            for worker in workspace.worker_store(session_id).load().workers
        )
        assert sum(
            message.get("role") == "assistant"
            and message.get("content") == "Both independent checks are complete."
            for message in messages
        ) == 1
        receipt_messages = [
            message
            for message in messages
            if message.get("role") == "assistant"
            and message.get("status") == "completed"
            and isinstance(message.get("worker_report_receipts"), list)
        ]
        assert len(receipt_messages) == 1
        receipts = cast(list[dict[str, object]], receipt_messages[0]["worker_report_receipts"])
        assert len(receipts) == 2
        task = cast(list[dict[str, object]], snapshot["foreground_tasks"])[0]
        assert task["execution_role"] == "router"
        workers = cast(list[dict[str, object]], task["workers"])
        assert len(workers) == 2
        assert {item["task_id"] for item in workers} == {"first-check", "second-check"}
        assert all(item["state"] == "ready" for item in workers)
    finally:
        provider.release_workers.set()
        provider.release_report_continuation.set()
        await service.stop()


@pytest.mark.asyncio
async def test_parallel_workers_manage_owned_subagent_results_before_router_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.service.test_service_concurrency import _CollectingSink

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    (directory / "worker-input.txt").write_text("WORKER-FILE-CONTENT", encoding="utf-8")
    provider = _WorkerSubAgentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        service.client_permission(client.client_id).select("read-only")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id,
            workspace_id=workspace.workspace_id,
            create_new=True,
        )
        session_id = str(opened["session_id"])
        claim = cast(dict[str, object], opened["claim"])
        await workspace.input(
            client.client_id,
            session_id,
            cast(int, claim["claim_version"]),
            "Run the independent checks.",
            "worker-subagent-input",
        )

        await asyncio.wait_for(provider.two_workers_started.wait(), timeout=10)
        assert provider.worker_started == 2
        assert set(provider.worker_tool_names) == {"delegating", "file-reader"}
        provider.release_workers.set()
        await asyncio.wait_for(provider.two_subagents_started.wait(), timeout=10)
        assert "spawn_agent" not in provider.subagent_tool_names
        children = cast(
            list[dict[str, object]],
            service.list_subagents(client.client_id, workspace.workspace_id, session_id)["items"],
        )
        assert len(children) == 2
        child_ids = {cast(str, child["agent_id"]) for child in children}
        repository = workspace.subagent_repository(session_id)
        owner = next(
            worker
            for worker in workspace.worker_store(session_id).load().workers
            if worker.task_id == "subagent-review"
        )
        child_records = [repository.get(child_id) for child_id in child_ids]
        assert all(child is not None for child in child_records)
        assert all(child.source.kind.value == "worker" for child in child_records if child)
        assert all(child.source.worker_id == owner.worker_id for child in child_records if child)
        assert all(child.creator_snapshot.permission_level == "read-only" for child in child_records if child)

        provider.release_subagent.set()
        def confirmation_requests() -> list[dict[str, object]]:
            return [
                event for event in sink.events
                if event.get("type") == "confirmation.requested"
            ]

        for index in range(2):
            await _wait_until(lambda index=index: len(confirmation_requests()) > index)
            payload = cast(dict[str, object], confirmation_requests()[index]["payload"])
            request = cast(dict[str, object], payload["request"])
            confirmation_owner = cast(dict[str, object], payload["owner"])
            assert confirmation_owner["kind"] == "subagent"
            assert confirmation_owner["agent_id"] in child_ids
            assert request["tool_name"] == "write_file"
            assert request["tool_call_id"] in {"subagent-write-first", "subagent-write-second"}
            service.decide_confirmation(
                client.client_id,
                cast(str, payload["token"]),
                "approved",
            )

        try:
            await asyncio.wait_for(provider.report_continuation_started.wait(), timeout=10)
        except TimeoutError as error:
            raise AssertionError(
                f"provider_calls={provider.calls[-10:]!r}; "
                f"router_report_texts={provider.router_report_texts!r}; "
                f"workers={[(item.task_id, item.state) for item in workspace.worker_store(session_id).load().workers]!r}"
            ) from error
        await _wait_until(
            lambda: not cast(
                list[dict[str, object]],
                cast(dict[str, object], workspace.session_snapshot(session_id)["live_state"])["runs"],
            )
        )

        workers = workspace.worker_store(session_id).load().workers
        assert len(workers) == 2
        assert all(worker.state == "ready" for worker in workers)
        child_records = [repository.get(child_id) for child_id in child_ids]
        assert len(child_records) == 2
        assert all(child is not None and child.status.value == "completed" for child in child_records)
        subagent_owner = next(worker for worker in workers if worker.task_id == "subagent-review")
        assert subagent_owner.report is not None
        assert "SUBAGENT-PRIVATE-RESULT-FIRST" in subagent_owner.report.summary
        assert "SUBAGENT-PRIVATE-RESULT-SECOND" in subagent_owner.report.summary
        file_worker = next(worker for worker in workers if worker.task_id == "file-review")
        assert file_worker.report is not None
        assert "WORKER-FILE-CONTENT" in file_worker.report.summary

        snapshot = workspace.session_snapshot(session_id)
        messages = cast(list[dict[str, object]], snapshot["messages"])
        assert provider.report_continuations == 1
        assert len(provider.router_report_texts) == 1
        assert all(child_id not in provider.router_report_texts[0] for child_id in child_ids)
        assert provider.router_report_texts[0].count("Worker ID:") == 2
        assert sum(
            message.get("role") == "assistant"
            and message.get("content") == "Both Worker reports are complete."
            for message in messages
        ) == 1
    finally:
        provider.release_workers.set()
        provider.release_subagent.set()
        provider.release_followup.set()
        await service.stop()


@pytest.mark.parametrize("operation", ["cancel", "replace"])
@pytest.mark.asyncio
async def test_cancel_or_replace_drains_worker_owned_subagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _WorkerSubAgentProvider(request_subagent_confirmation=False)
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        session_id = str(opened["session_id"])
        claim = cast(dict[str, object], opened["claim"])
        version = cast(int, claim["claim_version"])
        await workspace.input(client.client_id, session_id, version,
                              "Run the independent checks.", "worker-subagent-start")
        await asyncio.wait_for(provider.two_workers_started.wait(), timeout=10)
        provider.release_workers.set()
        await asyncio.wait_for(provider.two_subagents_started.wait(), timeout=10)
        await _wait_until(lambda: not cast(
            list[dict[str, object]],
            cast(dict[str, object], workspace.session_snapshot(session_id)["live_state"])["runs"],
        ))
        children = cast(
            list[dict[str, object]],
            service.list_subagents(client.client_id, workspace.workspace_id, session_id)["items"],
        )
        assert len(children) == 2
        child_ids = [cast(str, child["agent_id"]) for child in children]

        task = cast(list[dict[str, object]],
                    workspace.session_snapshot(session_id)["foreground_tasks"])[0]
        task_id = cast(str, task["task_id"])
        if operation == "cancel":
            cancellation = asyncio.create_task(workspace.cancel_task(
                client.client_id,
                session_id,
                version,
                task_id,
                cast(int, task["revision"]),
                "cancel-worker-subagents",
            ))
        else:
            provider.replace_followup = True
            provider.release_followup.set()
            await workspace.input(client.client_id, session_id, version,
                                  "Follow-up update", "replace-worker-subagents")

        await _wait_until(lambda: any(
            worker.task_id == "subagent-review"
            and worker.report is not None
            and worker.report.status == "cancelled"
            for worker in workspace.worker_store(session_id).load().workers
        ))
        provider.release_subagent.set()
        if operation == "cancel":
            result = await cancellation
            assert result["state"] == "cancelled"
        else:
            await asyncio.wait_for(provider.followup_started.wait(), timeout=10)

        await _wait_until(lambda: all(
            service.get_subagent(client.client_id, workspace.workspace_id, session_id, child_id)[
                "status"
            ] == "cancelled"
            for child_id in child_ids
        ))
        assert provider.report_continuations == 0
    finally:
        provider.release_workers.set()
        provider.release_subagent.set()
        provider.release_followup.set()
        await service.stop()


@pytest.mark.asyncio
async def test_user_input_precedes_ready_worker_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ParallelWorkerProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id,
            workspace_id=workspace.workspace_id,
            create_new=True,
        )
        session_id = opened["session_id"]
        claim = cast(dict[str, object], opened["claim"])
        claim_version = cast(int, claim["claim_version"])
        await service.handle_command(
            client.client_id,
            {
                "request_id": "worker-priority-initial",
                "type": "input",
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": claim_version,
                "payload": {"text": "Run both independent checks."},
            },
        )
        await asyncio.wait_for(provider.two_workers_started.wait(), timeout=10)
        await service.handle_command(
            client.client_id,
            {
                "request_id": "worker-priority-followup",
                "type": "input",
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": claim_version,
                "payload": {"text": "Follow-up update while checks are running."},
            },
        )
        await asyncio.wait_for(provider.followup_started.wait(), timeout=10)

        provider.release_workers.set()
        await _wait_until(
            lambda: all(
                worker.report is not None
                for worker in workspace.worker_store(session_id).load().workers
            )
        )
        await asyncio.sleep(0.05)
        assert provider.report_continuations == 0

        provider.release_followup.set()
        try:
            await asyncio.wait_for(provider.report_continuation_started.wait(), timeout=10)
        except TimeoutError as error:
            snapshot = workspace.session_snapshot(session_id)
            raise AssertionError(
                "Ready Worker reports were not resumed after user input: "
                f"tasks={snapshot['foreground_tasks']!r}; "
                f"workers={[(worker.task_id, worker.state, worker.report_delivery_state) for worker in workspace.worker_store(session_id).load().workers]!r}; "
                f"runs={snapshot['live_state']!r}"
            ) from error
        provider.release_report_continuation.set()
        await _wait_until(
            lambda: not cast(
                list[dict[str, object]],
                cast(dict[str, object], workspace.session_snapshot(session_id)["live_state"])["runs"],
            )
        )
        assert provider.report_continuations == 1
        messages = cast(list[dict[str, object]], workspace.session_snapshot(session_id)["messages"])
        assert sum(
            message.get("role") == "assistant"
            and message.get("content") == "Both independent checks are complete."
            for message in messages
        ) == 1
    finally:
        provider.release_workers.set()
        provider.release_followup.set()
        provider.release_report_continuation.set()
        await service.stop()


@pytest.mark.parametrize("operation", ["cancel", "replace", "cancel_write_failure"])
@pytest.mark.asyncio
async def test_cancelled_task_keeps_worker_diagnostics_without_auto_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ParallelWorkerProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id,
            workspace_id=workspace.workspace_id,
            create_new=True,
        )
        session_id = opened["session_id"]
        claim = cast(dict[str, object], opened["claim"])
        claim_version = cast(int, claim["claim_version"])
        await service.handle_command(
            client.client_id,
            {
                "request_id": "worker-cancel-input",
                "type": "input",
                "workspace_id": workspace.workspace_id,
                "session_id": session_id,
                "claim_version": claim_version,
                "payload": {"text": "Run both independent checks."},
            },
        )
        await asyncio.wait_for(provider.two_workers_started.wait(), timeout=10)
        await _wait_until(
            lambda: cast(list[dict[str, object]],
                workspace.session_snapshot(session_id)["foreground_tasks"])[0]["state"]
            == "waiting_for_workers"
        )
        task = cast(list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_tasks"])[0]
        task_id = cast(str, task["task_id"])
        if operation == "cancel_write_failure":
            store = workspace.worker_store(session_id)
            publish = store.publish_report
            failed = False

            def fail_once(*args: Any, **kwargs: Any) -> Any:
                nonlocal failed
                if not failed:
                    failed = True
                    raise OSError("injected cancellation report failure")
                return publish(*args, **kwargs)

            monkeypatch.setattr(store, "publish_report", fail_once)
        if operation != "replace":
            cancelled = await workspace.cancel_task(
                client.client_id, session_id, claim_version, task_id,
                cast(int, task["revision"]), "cancel-workers",
            )
            assert cancelled["state"] == "cancelled"
        else:
            provider.replace_followup = True
            provider.release_followup.set()
            await workspace.input(client.client_id, session_id, claim_version,
                                  "Follow-up update", "replace-workers")
            await asyncio.wait_for(provider.followup_started.wait(), 10)
        await _wait_until(lambda: not workspace._has_active_workers(session_id))

        await asyncio.sleep(0.05)
        assert provider.report_continuations == 0
        workers = workspace.worker_store(session_id).load().workers
        assert len(workers) == 2
        assert all(worker.report is not None and worker.state == "cancelled" for worker in workers)
        task = next(
            item
            for item in cast(list[dict[str, object]], workspace.session_snapshot(session_id)["foreground_tasks"])
            if item["task_id"] == task_id
        )
        diagnostic_workers = cast(list[dict[str, object]], task["workers"])
        assert len(diagnostic_workers) == 2
        assert {item["generation"] for item in diagnostic_workers} == {1}
        assert all(item["state"] == "cancelled" for item in diagnostic_workers)
    finally:
        provider.release_workers.set()
        await service.stop()

@pytest.mark.asyncio
async def test_workers_retain_provider_after_router_wait_and_configuration_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    providers: list[_ParallelWorkerProvider] = []

    def factory(_configuration: object) -> _ParallelWorkerProvider:
        provider = _ParallelWorkerProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(service_runtime, "create_provider", factory)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        sid = opened["session_id"]
        version = cast(int, cast(dict[str, object], opened["claim"])["claim_version"])
        await workspace.input(client.client_id, sid, version, "start", "start-workers")
        await _wait_until(lambda: bool(providers))
        provider = providers[0]
        await asyncio.wait_for(provider.two_workers_started.wait(), 10)
        await _wait_until(lambda: cast(list[dict[str, object]],
            workspace.session_snapshot(sid)["foreground_tasks"])[0]["state"] == "waiting_for_workers")
        await _wait_until(lambda: not cast(dict[str, object],
            workspace.session_snapshot(sid)["live_state"])["runs"])
        await workspace._loops[sid].loop.wait_for_restore_idle()
        await service.update_configuration(
            "change-provider", cast(str, service.config_view()["revision"]),
            {"models": {"providers": {"primary": {"base_url": "https://next.example/v1"}}}},
        )
        other = await service.register_client("cli")
        await service.connect_client(other.client_id, _SilentSink())
        await service.attach_workspace(other.client_id, directory)
        second = await service.open_conversation(
            other.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        second_sid = second["session_id"]
        second_version = cast(int, cast(dict[str, object], second["claim"])["claim_version"])
        await workspace.input(other.client_id, second_sid, second_version,
                              "Follow-up update", "new-generation")
        await _wait_until(lambda: len(providers) == 2)
        await asyncio.wait_for(providers[1].followup_started.wait(), 10)
        await service._configuration_resources._drain_cleanup()
        assert not provider.closed, "running Workers lost their captured Provider"
    finally:
        for provider in providers:
            provider.release_workers.set()
            provider.release_followup.set()
            provider.release_report_continuation.set()
        await service.stop()

@pytest.mark.asyncio
async def test_worker_reports_continue_after_owner_expiry_with_other_client_online(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ParallelWorkerProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        other = await service.register_client("cli")
        await service.connect_client(other.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        sid = opened["session_id"]
        version = cast(int, cast(dict[str, object], opened["claim"])["claim_version"])
        await workspace.input(client.client_id, sid, version, "start", "start-workers")
        await asyncio.wait_for(provider.two_workers_started.wait(), 10)
        await _wait_until(lambda: not cast(dict[str, object],
            workspace.session_snapshot(sid)["live_state"])["runs"])
        await workspace.expire_client(client.client_id)
        provider.release_workers.set()
        provider.release_report_continuation.set()
        await asyncio.wait_for(provider.report_continuation_started.wait(), 10)
        assert provider.report_continuations == 1
    finally:
        provider.release_workers.set()
        provider.release_report_continuation.set()
        await service.stop()

class _FollowOnWorkerProvider(_ParallelWorkerProvider):
    def __init__(self) -> None:
        super().__init__()
        self.report_batches: list[str] = []

    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        text = next((str(item.get("content", "")) for item in reversed(messages)
                     if item.get("role") == "user"), "")
        if "Worker reports for automatic continuation:" not in text:
            return super().stream(**kwargs)
        text = text.split("Worker reports for automatic continuation:", 1)[1].split(
            "## Task goal", 1,
        )[0]
        if not self.report_batches or self.report_batches[-1] != text:
            self.report_batches.append(text)
        if not any(item.get("tool_call_id") == "dispatch-third" for item in messages):
            call = ModelToolCall("dispatch-third", "dispatch_worker",
                                 '{"task_id":"third-check","task":"Check the combined result"}')
        else:
            call = ModelToolCall("batch-finish", "finish", '{"response":"All checks done."}')

        async def emit() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                finish_reason="tool_calls",
            ))
        return emit()


@pytest.mark.asyncio
async def test_router_follow_on_dispatch_does_not_redeliver_previous_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _FollowOnWorkerProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        sid = opened["session_id"]
        version = cast(int, cast(dict[str, object], opened["claim"])["claim_version"])
        await workspace.input(client.client_id, sid, version, "start", "start-workers")
        await asyncio.wait_for(provider.two_workers_started.wait(), 10)
        provider.release_workers.set()
        await _wait_until(lambda: len(provider.report_batches) >= 2)
        assert provider.report_batches[0].count("Worker ID:") == 2
        assert provider.report_batches[1].count("Worker ID:") == 1
        assert "Assignment ID: third-check" in provider.report_batches[1]
    finally:
        provider.release_workers.set()
        await service.stop()

@pytest.mark.asyncio
async def test_cli_displays_router_and_parallel_worker_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.widgets import Static, TextArea

    from tests.terminal.test_conversation import _run_cli_terminal_case, _wait_for_turn

    provider = _ParallelWorkerProvider()
    directory = tmp_path / "workspace"
    directory.mkdir()

    async def scenario(app: Any, pilot: Any) -> None:
        try:
            app.query_one("#conversation-input", TextArea).text = "Run both independent checks."
            await pilot.press("enter")
            await asyncio.wait_for(provider.two_workers_started.wait(), 10)
            await _wait_for_turn(app, timeout=30)
            async with asyncio.timeout(10):
                while "second-check" not in str(app.query_one("#foreground-tasks", Static).content):
                    await pilot.pause()
            text = str(app.query_one("#foreground-tasks", Static).content)
            assert "Router Task" in text
            assert text.count("Worker ") == 2
            assert "first-check" in text and "working" in text
            provider.release_workers.set()
            provider.release_report_continuation.set()
            await asyncio.wait_for(provider.report_continuation_started.wait(), 10)
        finally:
            provider.release_workers.set()
            provider.release_report_continuation.set()

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home", workspace=directory,
        monkeypatch=monkeypatch, scenario=scenario, provider=cast(Any, provider),
        size=(140, 50),
    )

class _ConfirmingWorkersProvider(_ParallelWorkerProvider):
    def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        messages = kwargs["messages"]
        if not str(messages[0].get("content", "")).startswith("You are an Aide Worker."):
            return super().stream(**kwargs)
        first = "first source" in str(messages[1].get("content", ""))
        filename = "first.txt" if first else "second.txt"
        async def emit() -> AsyncIterator[ModelStreamEvent]:
            if any(item.get("role") == "tool" for item in messages):
                yield ModelCompleted(_response("Authorized write complete."))
            else:
                yield ModelCompleted(ModelResponse(
                    message=AssistantModelMessage(content="", tool_calls=(
                        ModelToolCall("write-result", "write_file", json.dumps({
                            "path": filename, "content": "Worker result",
                        })),
                    )),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="tool_calls",
                ))
        return emit()


@pytest.mark.asyncio
async def test_parallel_worker_confirmation_tokens_keep_distinct_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aide.service.errors import ServiceError
    from tests.service.test_service_concurrency import _CollectingSink

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    provider = _ConfirmingWorkersProvider()
    provider.release_report_continuation.set()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        service.client_permission(client.client_id).select("read-only")
        sink = _CollectingSink()
        await service.connect_client(client.client_id, sink)
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        sid = opened["session_id"]
        version = cast(int, cast(dict[str, object], opened["claim"])["claim_version"])
        await workspace.input(client.client_id, sid, version, "start", "start-workers")
        def requests() -> list[dict[str, Any]]:
            return [event for event in sink.events if event.get("type") == "confirmation.requested"]
        await _wait_until(lambda: len(requests()) == 1)
        first = requests()[0]
        token = first["payload"]["token"]
        service.client_permission(client.client_id).select("full-access")
        service.decide_confirmation(client.client_id, token, "approved")
        await _wait_until(lambda: len(requests()) == 2)
        second = requests()[1]
        assert first["run_id"] != second["run_id"]
        workers = workspace.worker_store(sid).load().workers
        assert {first["run_id"], second["run_id"]} == {worker.execution_id for worker in workers}
        assert all(worker.permission_level == "read-only" for worker in workers)
        with pytest.raises(ServiceError):
            service.decide_confirmation(client.client_id, token, "approved")
        assert sum((directory / name).exists() for name in ("first.txt", "second.txt")) == 1
        service.decide_confirmation(client.client_id, second["payload"]["token"], "approved")
        await _wait_until(lambda: all((directory / name).exists()
                                      for name in ("first.txt", "second.txt")))
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["file_restore", "shutdown_restart"])
async def test_worker_children_restore_files_or_remain_interrupted_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    from aide.agent.session.restore import RestoreMode

    home = _home(tmp_path / "home")
    directory = tmp_path / "workspace"
    directory.mkdir()
    (directory / "worker-input.txt").write_text("retained input", encoding="utf-8")
    provider = _WorkerSubAgentProvider()
    monkeypatch.setattr(service_runtime, "create_provider", lambda _configuration: provider)
    service = AgentService(home, ConfigLoader(home).load())
    await service.start()
    try:
        client = await service.register_client("cli")
        service.client_permission(client.client_id).select("full-access")
        await service.connect_client(client.client_id, _SilentSink())
        workspace = await service.attach_workspace(client.client_id, directory)
        opened = await service.open_conversation(
            client.client_id, workspace_id=workspace.workspace_id, create_new=True,
        )
        sid = str(opened["session_id"])
        claim = cast(dict[str, Any], opened["claim"])
        await workspace.input(client.client_id, sid, claim["claim_version"], "start", "start")
        await asyncio.wait_for(provider.two_workers_started.wait(), 10)
        provider.release_workers.set()
        await asyncio.wait_for(provider.two_subagents_started.wait(), 10)
        children = cast(list[dict[str, Any]], service.list_subagents(
            client.client_id, workspace.workspace_id, sid,
        )["items"])
        assert len(children) == 2
        if operation == "file_restore":
            provider.release_subagent.set()
            await asyncio.wait_for(provider.report_continuation_started.wait(), 10)
            await _wait_until(lambda: not cast(dict[str, Any],
                workspace.session_snapshot(sid)["live_state"])["runs"])
            assert (directory / "subagent-first.txt").read_text() == "SubAgent first result"
            assert (directory / "subagent-second.txt").read_text() == "SubAgent second result"
            async with asyncio.timeout(10):
                attempt = 0
                while True:
                    inspected = await service.inspect_restore(
                        client.client_id, workspace.workspace_id, sid,
                        claim["claim_version"], claim["reconnect_credential"],
                        f"inspect-{attempt}", 1,
                    )
                    if "restore_plan" in inspected:
                        break
                    assert "Finish the active run" in str(inspected), inspected
                    attempt += 1
                    await asyncio.sleep(0.01)
            restored = await service.handle_management(client.client_id, workspace.workspace_id,
                sid, "restore/execute", {"request_id": "restore",
                "plan": {"anchor_id": 1}, "mode": RestoreMode.FILES.value},
                claim_version=claim["claim_version"], claim_credential=claim["reconnect_credential"])
            assert "restore_result" in restored, restored
            assert not (directory / "subagent-first.txt").exists()
            assert not (directory / "subagent-second.txt").exists()
            assert (directory / "worker-input.txt").read_text() == "retained input"
            assert not workspace.worker_store(sid).load().workers
            assert not workspace.subagent_repository(sid).list().items
        else:
            await service.stop()
            for child in children:
                record = workspace.subagent_repository(sid).get(child["agent_id"])
                assert record is not None and record.status.value == "interrupted"
            calls = len(provider.calls)
            provider.release_subagent.set()
            service = AgentService(home, ConfigLoader(home).load())
            await service.start()
            client = await service.register_client("cli")
            await service.connect_client(client.client_id, _SilentSink())
            workspace = await service.attach_workspace(client.client_id, directory)
            await service.open_conversation(client.client_id,
                workspace_id=workspace.workspace_id, session_id=sid)
            await asyncio.sleep(0.05)
            assert len(provider.calls) == calls
            assert provider.report_continuations == 0
            assert all(service.get_subagent(client.client_id, workspace.workspace_id, sid,
                child["agent_id"])["status"] == "interrupted" for child in children)
    finally:
        provider.release_workers.set()
        provider.release_subagent.set()
        await service.stop()
