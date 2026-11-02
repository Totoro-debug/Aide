"""Run-local Main Agent Tool for asking one foreground question."""

from __future__ import annotations

from typing import Annotated

from aide.agent.tools.base import BaseTool, ToolError, ToolParam
from aide.agent.tools.context import AgentRunSuspension, ToolRunContext
from aide.agent.tools.permission import ToolAuthorizationSession

_MAX_QUESTION_LENGTH = 2_000


class AskUserTool(BaseTool):
    """Request a free-text answer and suspend the current Main Agent Run."""

    name = "ask_user"
    description = "Ask the user one question that must be answered before continuing."
    required = ("question",)

    question: Annotated[
        str,
        ToolParam(
            description="One concise question that blocks safe progress until the user answers."
        ),
    ]

    def __init__(self) -> None:
        self._suspension: AgentRunSuspension | None = None

    async def execute(self, *, question: str) -> str:
        del question
        raise ToolError("ask_user is available only during a foreground Main Agent Run.")

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
        question_id = handler(question, context.tool_call_id)
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


__all__ = ["AskUserTool"]
