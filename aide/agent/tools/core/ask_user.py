"""Run-local Main Agent Tool for asking one foreground question."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from aide.agent.tools.base import BaseTool, ToolError, ToolParam
from aide.agent.tools.context import AgentRunSuspension, ToolRunContext
from aide.agent.tools.permission import ToolAuthorizationSession

_MAX_QUESTION_LENGTH = 2_000
_MAX_OPTIONS = 4
_MAX_OPTION_LENGTH = 200


class _AskUserQuestionOption(TypedDict):
    id: Annotated[str, ToolParam(min_length=1, max_length=_MAX_OPTION_LENGTH)]
    label: Annotated[str, ToolParam(min_length=1, max_length=_MAX_OPTION_LENGTH)]


class AskUserTool(BaseTool):
    """Request an explicit answer and suspend the current Main Agent Run."""

    name = "ask_user"
    description = "Ask the user one question that must be answered before continuing."
    required = ("question",)

    question: Annotated[
        str,
        ToolParam(
            description="One concise question that blocks safe progress until the user answers.",
            min_length=1,
            max_length=_MAX_QUESTION_LENGTH,
        ),
    ]
    options: Annotated[
        list[_AskUserQuestionOption] | None,
        ToolParam(
            description="Optional choices with stable IDs and display labels; at most four.",
            max_items=_MAX_OPTIONS,
        ),
    ] = None

    def __init__(self) -> None:
        self._suspension: AgentRunSuspension | None = None

    async def execute(
        self,
        *,
        question: str,
        options: list[_AskUserQuestionOption] | None = None,
    ) -> str:
        del question, options
        raise ToolError("ask_user is available only during a foreground Main Agent Run.")

    def validate_arguments(self, /, **arguments: Any) -> None:
        question = arguments.get("question")
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > _MAX_QUESTION_LENGTH
        ):
            raise ToolError("ask_user question must contain 1 to 2000 characters.")
        _validate_question_options(arguments.get("options"))

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, object],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: object,
    ) -> str:
        del authorization, kwargs
        question = arguments.get("question")
        options = _validate_question_options(arguments.get("options"))
        handler = context.foreground_question_handler
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > _MAX_QUESTION_LENGTH
            or handler is None
            or context.task_id is None
            or context.generation is None
            or context.execution_id is None
            or context.tool_call_id is None
        ):
            raise ToolError("ask_user requires a valid foreground question.")
        question_id = handler(
            question,
            [{"id": option["id"], "label": option["label"]} for option in options],
            context.tool_call_id,
        )
        if not isinstance(question_id, str) or not question_id:
            raise ToolError("The foreground question could not be registered.")
        self._suspension = AgentRunSuspension(
            kind="human_question",
            task_id=context.task_id,
            generation=context.generation,
            execution_id=context.execution_id,
            question_id=question_id,
            tool_call_id=context.tool_call_id,
        )
        return f"Question {question_id} is waiting for a user answer."

    def agent_run_suspension(self) -> AgentRunSuspension | None:
        return self._suspension


def _validate_question_options(value: object) -> list[_AskUserQuestionOption]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > _MAX_OPTIONS:
        raise ToolError("ask_user accepts at most four options.")
    normalized: list[_AskUserQuestionOption] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "label"}:
            raise ToolError("Each ask_user option must contain only an id and label.")
        option_id = item["id"]
        label = item["label"]
        if (
            not isinstance(option_id, str)
            or not option_id
            or option_id.strip() != option_id
            or len(option_id) > _MAX_OPTION_LENGTH
            or not isinstance(label, str)
            or not label.strip()
            or len(label) > _MAX_OPTION_LENGTH
            or option_id in seen
        ):
            raise ToolError(
                "Question option IDs must be unique and labels must contain 1 to 200 characters."
            )
        seen.add(option_id)
        normalized.append({"id": option_id, "label": label})
    return normalized


__all__ = ["AskUserTool"]
