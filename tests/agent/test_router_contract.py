from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from aide.agent.router import (
    DispatchWorkerTool,
    FinishRouterTool,
    RouterAskUserTool,
    RouterCatalog,
    RouterDecisionValidator,
)
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.tool_gateway import ModelToolCall, ToolGateway


def test_router_catalog_contains_only_foreground_decisions() -> None:
    catalog = RouterCatalog()
    assert catalog.tool_names == ("ask_user", "dispatch_worker", "finish")
    assert not set(catalog.tool_names) & {
        "read_file", "write_file", "edit_file", "exec", "schedule",
        "spawn_agent", "wait_agent", "list_agents", "tool_search",
    }


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "finish", "response": "done"},
        {"action": "ask_user", "question": "Which one?"},
        {"action": "dispatch", "task_id": "task-1", "task": "Inspect"},
    ],
)
def test_router_decisions_are_validated(decision: dict[str, str]) -> None:
    validator = RouterDecisionValidator(session_id="session-1", workspace_id="workspace-1")
    assert validator.validate(decision)["action"] == decision["action"]


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "dispatch", "task_id": "task-1", "task": "Inspect", "session_id": "other"},
        {"action": "dispatch", "task_id": "task-1", "task": "Inspect", "workspace_id": "other"},
        {"action": "dispatch", "task_id": "task-1", "task": "Inspect", "worker_id": "worker-1"},
        {"action": "exec", "command": "touch x"},
        {"action": "dispatch", "task_id": "", "task": "Inspect"},
    ],
)
def test_invalid_or_cross_owner_decisions_are_rejected(decision: dict[str, str]) -> None:
    validator = RouterDecisionValidator(session_id="session-1", workspace_id="workspace-1")
    with pytest.raises(ValueError):
        validator.validate(decision)


def test_dispatch_validation_is_bounded_and_does_not_start_a_worker() -> None:
    validator = RouterDecisionValidator(session_id="session-1", workspace_id="workspace-1")
    decision = validator.validate({"action": "dispatch", "task_id": "task-1", "task": "Inspect"})
    assert decision == {"action": "dispatch", "task_id": "task-1", "task": "Inspect"}

    with pytest.raises(ValueError):
        validator.validate({"action": "dispatch", "task_id": "task-1", "task": "x" * 8001})


@pytest.mark.asyncio
async def test_dispatch_tool_is_callable_through_the_public_gateway_and_is_idempotent(
    tmp_path: Path,
) -> None:
    tool = DispatchWorkerTool(
        RouterDecisionValidator(session_id="session-1", workspace_id=str(tmp_path))
    )
    context = ToolRunContext(
        workspace=tmp_path,
        task_id="foreground-task",
        generation=1,
        execution_id="execution-1",
    )
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=("dispatch_worker",),
        run_tools=(tool,),
        tool_context=context,
    )

    result = await gateway.call(
        ModelToolCall("dispatch-1", "dispatch_worker", '{"task_id":"task-1","task":"Inspect"}')
    )
    assert result.status == "success"
    assert "task-1" in result.content

    duplicate = await gateway.call(
        ModelToolCall("dispatch-2", "dispatch_worker", '{"task_id":"task-1","task":"Inspect"}')
    )
    assert duplicate.status == "error"
    assert "unique" in duplicate.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_argument",
    [
        '"session_id":"other-session"',
        '"workspace_id":"other-workspace"',
        '"worker_id":"worker-fake"',
    ],
)
async def test_dispatch_gateway_rejects_forged_ownership_references_without_side_effects(
    tmp_path: Path, extra_argument: str
) -> None:
    tool = DispatchWorkerTool(
        RouterDecisionValidator(session_id="session-1", workspace_id=str(tmp_path))
    )
    context = ToolRunContext(
        workspace=tmp_path,
        task_id="foreground-task",
        generation=1,
        execution_id="execution-1",
    )
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=("dispatch_worker",),
        run_tools=(tool,),
        tool_context=context,
    )

    result = await gateway.call(ModelToolCall(
        "forged-dispatch",
        "dispatch_worker",
        '{"task_id":"task-foreign","task":"Inspect",' + extra_argument + "}",
    ))

    assert result.status == "error"
    valid_retry = await gateway.call(ModelToolCall(
        "valid-dispatch",
        "dispatch_worker",
        '{"task_id":"task-foreign","task":"Inspect"}',
    ))
    assert valid_retry.status == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ask_user", "finish"])
@pytest.mark.parametrize("extra", ["session_id", "workspace_id", "worker_id", "unexpected"])
async def test_router_decision_tools_reject_extra_fields_without_side_effects(
    tmp_path: Path, name: str, extra: str,
) -> None:
    import json

    registered: list[str] = []

    def register(question: str, options: Sequence[Mapping[str, str]], tool_call_id: str) -> str:
        registered.append(question)
        return "question"

    context = ToolRunContext(
        workspace=tmp_path, task_id="task", generation=1, execution_id="execution",
        foreground_question_handler=register,
    )
    tool = RouterAskUserTool() if name == "ask_user" else FinishRouterTool()
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=(name,), allowed_names=(name,), run_tools=(tool,), tool_context=context,
    )
    args = {"question": "Which one?"} if name == "ask_user" else {"response": "Done"}
    result = await gateway.call(ModelToolCall("forged", name, json.dumps({**args, extra: "other"})))
    assert result.status == "error"
    assert registered == []
    valid = await gateway.call(ModelToolCall("valid", name, json.dumps(args)))
    assert valid.status == "success"
    assert registered == (["Which one?"] if name == "ask_user" else [])


@pytest.mark.asyncio
async def test_router_finish_rejects_blank_response(tmp_path: Path) -> None:
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=("finish",), run_tools=(FinishRouterTool(),),
    )
    result = await gateway.call(ModelToolCall("invalid", "finish", '{"response":"   "}'))
    assert result.status == "error"
