from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from aide.agent.runner import AgentRunner
from aide.agent.tools.base import ToolError
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
    question_calls: list[tuple[str, list[dict[str, str]], str]] = []

    def ask(question: str, options: Sequence[Mapping[str, str]], tool_call_id: str) -> str:
        question_calls.append((question, [dict(option) for option in options], tool_call_id))
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
    assert question_calls == [("Which folder?", [], "question-call")]
    assert later_tool.calls == []
    assert [item["role"] for item in result.messages] == ["assistant", "tool", "tool"]
    paired = {item.get("tool_call_id") for item in result.messages if item["role"] == "tool"}
    assert paired == {"question-call", "write-call"}
    assert result.messages[-1]["status"] == "error"
    assert "not executed" in str(result.messages[-1]["content"]).lower()
    assert result.usage["model_calls"] == 1


def test_question_schema_exposes_bounded_stable_id_options() -> None:
    parameters = AskUserTool().to_schema()["function"]["parameters"]
    properties = parameters["properties"]
    assert properties["question"]["maxLength"] == 2000
    options = properties["options"]
    assert options["maxItems"] == 4
    assert options["items"]["type"] == "object"
    assert set(options["items"]["properties"]) == {"id", "label"}
    assert set(options["items"]["required"]) == {"id", "label"}
    assert options["items"]["properties"]["id"]["maxLength"] == 200
    assert options["items"]["properties"]["label"]["maxLength"] == 200


def test_question_argument_validation_accepts_bounds_and_rejects_invalid_options() -> None:
    tool = AskUserTool()
    four_options = [{"id": f"option-{index}", "label": "x" * 200} for index in range(4)]
    tool.validate_arguments(question="?" * 2000, options=four_options)

    for question, options in (
        (" ", []),
        ("?" * 2001, []),
        ("Valid question?", [*four_options, {"id": "extra", "label": "Extra"}]),
        ("Valid question?", [{"id": "same", "label": "One"}, {"id": "same", "label": "Two"}]),
        ("Valid question?", [{"id": "blank", "label": " "}]),
        ("Valid question?", [{"id": "long", "label": "x" * 201}]),
        ("Valid question?", [{"id": "invalid", "label": "Bad", "extra": "field"}]),
    ):
        with pytest.raises(ToolError):
            tool.validate_arguments(question=question, options=options)
