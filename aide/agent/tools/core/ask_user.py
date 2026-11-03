"""Run-local Main Agent Tool for asking one foreground question."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, cast

from aide.agent.foreground_work.options import (
    MAX_QUESTION_OPTION_COUNT,
    MAX_QUESTION_OPTION_FIELD_LENGTH,
    validate_question_options,
)
from aide.agent.tools.base import BaseTool, ToolError, ToolParam
from aide.agent.tools.context import AgentRunSuspension, ToolRunContext
from aide.agent.tools.permission import ToolAuthorizationSession

_MAX_QUESTION_LENGTH = 2_000


class _AskUserQuestionOption(TypedDict):
    id: Annotated[str, ToolParam(min_length=1, max_length=MAX_QUESTION_OPTION_FIELD_LENGTH)]
    label: Annotated[str, ToolParam(min_length=1, max_length=MAX_QUESTION_OPTION_FIELD_LENGTH)]


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
            max_items=MAX_QUESTION_OPTION_COUNT,
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
    try:
        return [
            cast(_AskUserQuestionOption, dict(option))
            for option in validate_question_options(value)
        ]
    except ValueError as error:
        raise ToolError(f"Invalid ask_user options: {error}") from error


__all__ = ["AskUserTool"]
