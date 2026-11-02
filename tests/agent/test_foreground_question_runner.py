from __future__ import annotations

from pathlib import Path

import pytest

from aide.agent.runner import AgentRunner
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.ask_user import AskUserTool
from aide.agent.tools.tool_gateway import ModelToolCall
from aide.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from tests.fixtures import (
    DetachedRequestPreparer,
    FakeTool,
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    SingleToolGateway,
    StreamScript,
)


async def _ignore_output(_event: object) -> None:
    return


@pytest.mark.asyncio
async def test_question_suspends_runner_and_pairs_later_calls_without_executing_them(
    tmp_path: Path,
) -> None:
    question_calls: list[tuple[str, str]] = []

    def ask(question: str, tool_call_id: str) -> str:
        question_calls.append((question, tool_call_id))
        return "question-1"

    question_tool = AskUserTool()
    later_tool = FakeTool(name="write_after_question", description="Write", outcomes=("written",))
    context = ToolRunContext(
        workspace=tmp_path,
        foreground_question_handler=ask,
        task_id="task-1",
        generation=1,
        execution_id="execution-1",
    )
    gateway = SingleToolGateway([question_tool, later_tool], tool_context=context)
    calls = (
        ModelToolCall(
            id="question-call", name="ask_user", arguments='{"question":"Which folder?"}'
        ),
        ModelToolCall(id="write-call", name="write_after_question", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )

    result = await AgentRunner(ScriptedFakeRouter(provider), DetachedRequestPreparer([])).run(
        [{"role": "user", "content": "Prepare the report."}],
        model="chat",
        tool_gateway=gateway,
        on_output=_ignore_output,
        confirmation=None,
        externalize_result=None,
        cancel_requested=None,
        max_iterations=50,
    )

    assert result.finish_reason == "suspended"
    assert result.suspension is not None
    assert result.suspension.question_id == "question-1"
    assert question_calls == [("Which folder?", "question-call")]
    assert later_tool.calls == []
    assert [item["role"] for item in result.messages] == ["assistant", "tool", "tool"]
    paired = {item.get("tool_call_id") for item in result.messages if item["role"] == "tool"}
    assert paired == {"question-call", "write-call"}
    assert result.messages[-1]["status"] == "error"
    assert "not executed" in str(result.messages[-1]["content"]).lower()
    assert result.usage["model_calls"] == 1
