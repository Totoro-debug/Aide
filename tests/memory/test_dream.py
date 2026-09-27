import asyncio
import inspect
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
from markdown_it import MarkdownIt

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import (
    ConfigLoader,
    MemoryConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from myclaw.errors import ErrorInfo
from myclaw.memory.dream import Dream, DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelAttemptGuard, ModelRouter, ModelRouteStatus
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelMessages,
    ModelResponse,
    ModelUsage,
)
from myclaw.templates import render_template
from myclaw.tools.gateway import ModelToolCall
from myclaw.workspace.state import WorkspaceState
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider, ScriptedFakeRouter
from tests.fixtures.diagnostic_capture import capture_diagnostics

NOW = datetime(2026, 8, 27, 10, 0, tzinfo=timezone(timedelta(hours=8)))
_FAKE_MEMORY_ROUTE_STATUS = ModelRouteStatus(
    requested_route="memory",
    selected_route="memory",
    provider_id="test-provider",
    model="test-model",
    context_window=200_000,
    max_output=8_192,
    used_default=False,
)


def _manager(agent_home: Path) -> MemoryManager:
    home = AgentHome(agent_home)
    home.initialize()
    workspace = agent_home.parent / "dream-workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return MemoryManager(state)


async def _cursor(manager: MemoryManager) -> int:
    return await manager._cursor_store.read()


def _response(
    content: str,
    *,
    tool_calls: tuple[ModelToolCall, ...] = (),
    finish_reason: Literal["stop", "tool_calls", "length", "cancelled"] | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content, tool_calls=tool_calls),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
    )


def _memory_configuration(
    *,
    memory_context_window: int = 200_000,
    memory_max_output: int = 8_192,
    default_context_window: int = 200_000,
    default_max_output: int = 8_192,
) -> UserConfiguration:
    memory_provider = ProviderConfiguration(
        provider_id="memory-provider",
        protocol="openai-compatible",
        base_url="https://memory.example/v1",
        api_key="memory-secret",
        models=("memory-model",),
    )
    default_provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="openai-compatible",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models=("default-model",),
    )
    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=4096),
        memory=MemoryConfiguration(
            batch_size=10,
            schedule="0 * * * *",
        ),
        models=ModelsConfiguration(
            providers={
                memory_provider.provider_id: memory_provider,
                default_provider.provider_id: default_provider,
            },
            routes={
                "memory": RouteConfiguration(
                    provider_id=memory_provider.provider_id,
                    model="memory-model",
                    context_window=memory_context_window,
                    max_output=memory_max_output,
                    temperature=0,
                    reasoning_effort="low",
                    timeout=60,
                ),
                "default": RouteConfiguration(
                    provider_id=default_provider.provider_id,
                    model="default-model",
                    context_window=default_context_window,
                    max_output=default_max_output,
                    temperature=0,
                    reasoning_effort="low",
                    timeout=60,
                ),
            },
        ),
    )


class _CountingRouter:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        self.complete_calls = 0

    async def complete(
        self,
        route: Literal["memory"],
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse:
        self.complete_calls += 1
        return await self._router.complete(route, messages=messages, tools=tools, guard=guard)


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


class _BlockingRouter:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.complete_calls = 0

    def stream(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Dream must use the non-streaming memory route")

    async def complete(self, *args: object, **kwargs: object) -> ModelResponse:
        del args, kwargs
        self.complete_calls += 1
        self.started.set()
        await self.release.wait()
        return self.response


@pytest.mark.asyncio
async def test_dream_returns_without_a_provider_call_when_no_summary_is_pending(
    agent_home: Path,
) -> None:
    provider = ScriptedFakeProvider()
    router = ScriptedFakeRouter(provider)
    dream = Dream(
        memory_manager=_manager(agent_home),
        model_router=router,
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=0,
    )
    assert router.complete_calls == 0
    assert provider.complete_requests == []


def test_dream_constructor_requires_composition_inputs() -> None:
    assert "long_term_path" not in inspect.signature(Dream).parameters
    assert (
        inspect.signature(Dream).parameters["memory_route_status"].default
        is inspect.Parameter.empty
    )


@pytest.mark.asyncio
async def test_dream_uses_the_memory_route_with_static_default_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    manager = MemoryManager(state)
    await manager.append_summary("The user prefers concise status reports.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update is needed."),))
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
    )
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=configuration.memory.batch_size,
        memory_route_status=router.route_status("memory"),
    )

    try:
        result = await dream.run()
    finally:
        await dream.close()
        await router.close()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert await _cursor(manager) == 1
    assert provider.complete_requests[0].model == "claude-model"


@pytest.mark.asyncio
async def test_dream_processes_claimed_summaries_through_restricted_memory_route(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("The user prefers concise reports.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert len(provider.complete_requests) == 1
    request = provider.complete_requests[0]
    assert [schema["function"]["name"] for schema in request.tools] == ["edit_file"]
    assert request.messages[0]["role"] == "system"
    assert manager.memory_snapshot() in str(request.messages[1]["content"])
    assert "The user prefers concise reports." in str(request.messages[1]["content"])
    assert provider.stream_requests == []
    assert not manager.workspace_state.logs_directory.exists()


@pytest.mark.asyncio
async def test_dream_builds_a_fenced_markdown_memory_request_at_its_execution_boundary(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("Already claimed.", NOW)
    initial_claim = await manager.claim_summaries(1)
    assert initial_claim.cursor == 1
    summary_contents = (
        'Unicode 雪 and "quoted" data.\n```\n<tag> & `backtick` and C:\\tmp\\done.',
        r"HTML </script>; literal \u0060 plus NaN remains text.",
    )
    for content in summary_contents:
        await manager.append_summary(content, NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 2 summaries; Long-term Memory unchanged.",
        processed_count=2,
        memory_updated=False,
        cursor=3,
    )
    request = provider.complete_requests[0]
    assert [message["role"] for message in request.messages] == ["system", "user"]
    assert request.messages[0] == {
        "role": "system",
        "content": render_template(
            "memory-task-prompt.md",
            long_term_path=manager.long_term_path,
        ),
    }
    assert str(manager.long_term_path) in str(request.messages[0]["content"])
    assert all(set(message) == {"role", "content"} for message in request.messages)

    user_context = request.messages[1]["content"]
    assert isinstance(user_context, str)
    tokens = MarkdownIt("commonmark").parse(user_context)
    assert [token.content for token in tokens if token.type == "inline"] == [
        "Long-term Memory",
        "Long-term Memory",
        "User Info",
        "User Preference",
        "Project Fact",
        "Lesson",
        "Summary Cursor",
        "1",
        "Conversation Summaries",
    ]
    fences = [token for token in tokens if token.type == "fence"]
    assert len(fences) == 1
    assert fences[0].info == "jsonl"
    payload = fences[0].content
    assert "`" not in payload
    assert [
        json.loads(line, parse_constant=_reject_nonstandard_json_constant)
        for line in payload.splitlines()
    ] == [
        {
            "index": 2,
            "timestamp": "2026-08-27T10:00:00.000+08:00",
            "content": summary_contents[0],
        },
        {
            "index": 3,
            "timestamp": "2026-08-27T10:00:00.000+08:00",
            "content": summary_contents[1],
        },
    ]
    message_payload = json.dumps(request.messages, ensure_ascii=False)
    for forbidden in ("contextbuilder", '"function"', "skill_catalog", "blackboard", "session_id"):
        assert forbidden not in message_payload.casefold()
    assert [schema["function"]["name"] for schema in request.tools] == ["edit_file"]


@pytest.mark.asyncio
async def test_dream_edit_refreshes_the_manager_snapshot_after_a_successful_edit(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("The user prefers concise reports.", NOW)
    original = manager.memory_snapshot()
    old_text = "## User Preference\n"
    new_text = "## User Preference\n\nPrefers concise reports.\n"
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": old_text,
                                "new_text": new_text,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    expected = original.replace(old_text, new_text, 1)
    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert len(provider.complete_requests) == 1
    assert [schema["function"]["name"] for schema in provider.complete_requests[0].tools] == [
        "edit_file"
    ]


@pytest.mark.asyncio
async def test_dream_edit_can_replace_all_exact_matches(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("repeat repeat")
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "repeat",
                                "new_text": "updated",
                                "replace_all": True,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    assert manager.memory_snapshot() == "updated updated"


@pytest.mark.asyncio
async def test_dream_edit_response_is_terminal_without_a_confirmation_request(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    original = manager.memory_snapshot()
    old_text = "## User Preference\n"
    new_text = "## User Preference\n\nPersisted before provider failure.\n"
    failure = ModelCallError(ErrorInfo(code="model_failed", message="provider failed"))
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": old_text,
                                "new_text": new_text,
                            }
                        ),
                    ),
                ),
            ),
            failure,
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    expected = original.replace(old_text, new_text, 1)
    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_model_failure_keeps_the_accepted_cursor(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    failure = ModelCallError(ErrorInfo(code="model_failed", message="provider failed"))
    provider = ScriptedFakeProvider(completions=(failure,))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(code="model_failed", message="provider failed"),
    )
    assert await _cursor(manager) == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_cursor_publication_failure_is_unprocessed_and_logged_once(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("One pending summary.", NOW)

    def fail_cursor(_path: Path, _content: str) -> None:
        raise OSError("injected cursor publication failure")

    monkeypatch.setattr(manager._cursor_store, "_replace_text", fail_cursor)
    provider = ScriptedFakeProvider(completions=(_response("No update."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
        capture.close()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="persistence_error",
            message="Summary Cursor could not be updated.",
        ),
    )
    assert await _cursor(manager) == 0
    assert provider.complete_requests == []
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in capture.text


@pytest.mark.asyncio
async def test_dream_tool_failure_keeps_the_accepted_cursor_without_retry(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="outside-memory",
                        name="read_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path.parent / "other.txt"),
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="The requested tool is not available.",
        ),
    )
    assert await _cursor(manager) == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_logs_an_unexpected_tool_failure_once_at_its_boundary(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)

    async def fail_edit(
        self: MemoryManager,
        *,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> str:
        del self, old, new, replace_all
        raise RuntimeError("PRIVATE unexpected memory failure")

    monkeypatch.setattr(MemoryManager, "edit_long_term", fail_edit)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="unexpected-edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "## User Info\n",
                                "new_text": "## User Info\n\nUpdated.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
        capture.close()

    assert result.error == ErrorInfo(
        code="tool_failed",
        message="edit_file could not complete the request.",
    )
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=tool_failed" in capture.text
    assert "Tool execution failed name=edit_file" not in capture.text
    assert "RuntimeError: PRIVATE unexpected memory failure" in capture.text


@pytest.mark.asyncio
async def test_dream_logs_a_corrupt_summary_failure_once_without_leaking_content(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    manager._summary_store.path.write_text(
        "PRIVATE INVALID SUMMARY STREAM",
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider()
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
        capture.close()

    assert result.error == ErrorInfo(
        code="persistence_error",
        message="Memory Task state could not be read.",
    )
    assert provider.complete_requests == []
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in capture.text
    assert "ValueError: summary stream must contain complete JSONL records" in capture.text
    assert "PRIVATE INVALID SUMMARY STREAM" not in capture.text


@pytest.mark.asyncio
async def test_dream_never_reads_through_an_external_long_term_memory_hard_link(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    outside = agent_home.parent / "outside-memory.md"
    secret = "OUTSIDE SECRET MUST NOT REACH THE MODEL"
    outside.write_text(secret, encoding="utf-8")
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="read-memory",
                        name="read_file",
                        arguments=json.dumps({"path": str(manager.long_term_path)}),
                    ),
                ),
            ),
            _response("No update needed."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    manager.long_term_path.unlink()
    manager.long_term_path.hardlink_to(outside)

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="persistence_error",
            message="Memory Task state could not be read.",
        ),
    )
    assert provider.complete_requests == []
    assert outside.read_text(encoding="utf-8") == secret
    assert await _cursor(manager) == 1


@pytest.mark.asyncio
async def test_dream_edit_failure_keeps_the_accepted_cursor(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    store = manager._long_term_store

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("disk full")

    original_replace = store._replace_text

    def replace(path: Path, content: str) -> None:
        if path == manager.long_term_path:
            fail_replace(path, content)
        original_replace(path, content)

    monkeypatch.setattr(store, "_replace_text", replace)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "## User Preference\n",
                                "new_text": "## User Preference\n\nUpdated.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory could not be updated.",
        ),
    )
    assert await _cursor(manager) == 1


@pytest.mark.asyncio
async def test_dream_cancellation_keeps_the_accepted_cursor_and_releases_run(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    task = asyncio.create_task(dream.run())

    await router.started.wait()
    assert await _cursor(manager) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _cursor(manager) == 1
    result = await dream.run()
    assert result.status == "No pending summaries"
    assert router.complete_calls == 1


@pytest.mark.asyncio
async def test_dream_concurrent_runs_claim_once_and_do_not_reenter(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    first = asyncio.create_task(dream.run())

    await router.started.wait()
    overlapping = await dream.run()
    assert overlapping == DreamResult(
        status="Memory Task is already running.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="memory_task_running",
            message="A Memory Task is already running.",
        ),
    )
    assert await _cursor(manager) == 1

    router.release.set()
    result = await first

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert router.complete_calls == 1


@pytest.mark.asyncio
async def test_dream_close_waits_for_active_work_and_releases_the_task(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )
    running = asyncio.create_task(dream.run())

    await router.started.wait()
    closing = asyncio.create_task(dream.close())
    await asyncio.sleep(0)
    assert not closing.done()

    router.release.set()
    result = await running
    await closing

    assert result.status == "Processed 1 summary; Long-term Memory unchanged."
    assert dream._task is None
    with pytest.raises(RuntimeError, match="Dream is no longer active"):
        await dream.run()


@pytest.mark.asyncio
async def test_dream_executes_multiple_edits_in_response_order(agent_home: Path) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("A")
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="first",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "A",
                                "new_text": "B",
                            }
                        ),
                    ),
                    ModelToolCall(
                        id="second",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "B",
                                "new_text": "C",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    assert await manager.read_long_term() == "C"
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_stops_after_a_later_edit_failure_and_keeps_the_first_edit(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("A")
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="first",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "A",
                                "new_text": "B",
                            }
                        ),
                    ),
                    ModelToolCall(
                        id="second",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "missing",
                                "new_text": "C",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=True,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="The requested Long-term Memory text did not match precisely.",
        ),
    )
    assert await manager.read_long_term() == "B"
    assert manager.memory_snapshot() == "B"
    assert await _cursor(manager) == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "error_code"),
    (("length", "model_failed"), ("cancelled", "turn_cancelled")),
)
async def test_dream_finish_reason_failure_does_not_apply_returned_edits(
    agent_home: Path,
    finish_reason: Literal["length", "cancelled"],
    error_code: Literal["model_failed", "turn_cancelled"],
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    original = manager.memory_snapshot()
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "truncated",
                finish_reason=finish_reason,
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path),
                                "old_text": "## User Info\n",
                                "new_text": "## User Info\n\nShould not be written.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        memory_route_status=_FAKE_MEMORY_ROUTE_STATUS,
    )

    result = await dream.run()

    assert result.status == "Memory Task failed."
    assert result.processed_count == 0
    assert result.memory_updated is False
    assert result.cursor == 1
    assert result.error is not None
    assert result.error.code == error_code
    assert await manager.read_long_term() == original
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_initial_overflow_makes_no_router_or_provider_attempt(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("x" * 6_000)
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider()
    factory_calls: list[str] = []

    def provider_factory(configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(configuration.provider_id)
        return provider

    router = ModelRouter(
        configuration=_memory_configuration(
            memory_context_window=1024,
            memory_max_output=512,
        ),
        provider_factory=provider_factory,
    )
    counting_router = _CountingRouter(router)
    dream = Dream(
        memory_manager=manager,
        model_router=counting_router,
        batch_size=10,
        memory_route_status=router.route_status("memory"),
    )

    try:
        result = await dream.run()
    finally:
        await dream.close()
        await router.close()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="model_context_overflow",
            message="Model request context exceeds the available input budget.",
        ),
    )
    assert counting_router.complete_calls == 0
    assert factory_calls == []
    assert provider.complete_requests == []
    assert await _cursor(manager) == 1


@pytest.mark.asyncio
async def test_dream_fallback_overflow_skips_the_smaller_fallback_provider(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("x" * 3_000)
    await manager.append_summary("A pending summary.", NOW)
    memory_provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(
                ErrorInfo(code="provider_auth_error", message="memory route unavailable")
            ),
        )
    )
    default_provider = ScriptedFakeProvider()
    providers = {
        "memory-provider": memory_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=_memory_configuration(
            memory_context_window=20_000,
            memory_max_output=1_000,
            default_context_window=256,
            default_max_output=64,
        ),
        provider_factory=lambda configuration: providers[configuration.provider_id],
    )
    counting_router = _CountingRouter(router)
    dream = Dream(
        memory_manager=manager,
        model_router=counting_router,
        batch_size=10,
        memory_route_status=router.route_status("memory"),
    )

    try:
        result = await dream.run()
    finally:
        await dream.close()
        await router.close()

    assert result.error == ErrorInfo(
        code="model_context_overflow",
        message="Model request context exceeds the available input budget.",
    )
    assert result.cursor == 1
    assert result.memory_updated is False
    assert counting_router.complete_calls == 1
    assert len(memory_provider.complete_requests) == 1
    assert default_provider.complete_requests == []


@pytest.mark.asyncio
async def test_dream_router_retry_is_one_logical_completion_with_multiple_attempts(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    retryable = ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="temporary timeout",
            retryable=True,
        )
    )
    provider = ScriptedFakeProvider(
        completions=(retryable, _response("No durable update.")),
    )
    router = ModelRouter(
        configuration=_memory_configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    counting_router = _CountingRouter(router)
    dream = Dream(
        memory_manager=manager,
        model_router=counting_router,
        batch_size=10,
        memory_route_status=router.route_status("memory"),
    )

    try:
        result = await dream.run()
    finally:
        await dream.close()
        await router.close()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert counting_router.complete_calls == 1
    assert len(provider.complete_requests) == 2
