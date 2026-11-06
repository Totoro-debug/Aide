from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from aide.agent.router import (
    DispatchWorkerTool,
    FinishRouterTool,
    RouterAskUserTool,
    RouterDecisionValidator,
)
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.tool_gateway import ModelToolCall, ToolGateway


def test_dispatch_validation_is_bounded_and_does_not_start_a_worker() -> None:
    validator = RouterDecisionValidator()
    decision = validator.validate_dispatch("task-1", "Inspect", [])
    assert decision == {"task_id": "task-1", "task": "Inspect", "context": []}

    with pytest.raises(ValueError):
        validator.validate_dispatch("task-1", "x" * 8001, [])


@pytest.mark.asyncio
async def test_dispatch_tool_is_callable_through_the_public_gateway_and_is_idempotent(
    tmp_path: Path,
) -> None:
    dispatched: list[tuple[str, str, tuple[str, ...], str]] = []

    async def dispatch(
        task_id: str, task: str, context: Sequence[str], tool_call_id: str
    ) -> str:
        dispatched.append((task_id, task, tuple(context), tool_call_id))
        return "Worker report"

    tool = DispatchWorkerTool(RouterDecisionValidator())
    context = ToolRunContext(
        workspace=tmp_path,
        task_id="foreground-task",
        generation=1,
        execution_id="execution-1",
        worker_dispatch_handler=dispatch,
    )
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=("dispatch_worker",),
        run_tools=(tool,),
        tool_context=context,
    )

    result = await gateway.call(
        ModelToolCall(
            "dispatch-1", "dispatch_worker",
            '{"task_id":"task-1","task":"Inspect","context":[{"text":"file.txt"}]}',
        )
    )
    assert result.status == "success"
    assert result.content == "Worker report"
    assert dispatched == [("task-1", "Inspect", ("file.txt",), "dispatch-1")]

    duplicate = await gateway.call(
        ModelToolCall(
            "dispatch-2", "dispatch_worker",
            '{"task_id":"task-1","task":"Inspect","context":[]}',
        )
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
    dispatched: list[str] = []

    async def dispatch(
        task_id: str, task: str, context: Sequence[str], tool_call_id: str
    ) -> str:
        del task, context, tool_call_id
        dispatched.append(task_id)
        return "Worker report"

    tool = DispatchWorkerTool(RouterDecisionValidator())
    context = ToolRunContext(
        workspace=tmp_path,
        task_id="foreground-task",
        generation=1,
        execution_id="execution-1",
        worker_dispatch_handler=dispatch,
    )
    gateway = ToolGateway(workspace=tmp_path).for_run(
        exposed_names=("dispatch_worker",),
        run_tools=(tool,),
        tool_context=context,
    )

    result = await gateway.call(ModelToolCall(
        "forged-dispatch",
        "dispatch_worker",
        '{"task_id":"task-foreign","task":"Inspect","context":[], '
        + extra_argument + "}",
    ))

    assert result.status == "error"
    assert dispatched == []
    valid_retry = await gateway.call(ModelToolCall(
        "valid-dispatch",
        "dispatch_worker",
        '{"task_id":"task-foreign","task":"Inspect","context":[]}',
    ))
    assert valid_retry.status == "success"
    assert dispatched == ["task-foreign"]


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
