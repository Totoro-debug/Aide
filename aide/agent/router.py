"""Validated foreground Router decisions and its restricted tool surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast

from aide.agent.tools.base import BaseTool, ToolError, ToolParam
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.ask_user import AskUserTool
from aide.agent.tools.permission import ToolAuthorizationSession


@dataclass(frozen=True, slots=True)
class RouterCatalog:
    """The only operations a foreground Router may request."""

    tool_names: tuple[str, ...] = ("ask_user", "dispatch_worker", "finish")

    def __post_init__(self) -> None:
        if self.tool_names != tuple(dict.fromkeys(self.tool_names)):
            raise ValueError("Router Tool names must be unique")


class RouterDecisionValidator:
    """Validate decisions against the current foreground ownership boundary."""

    def __init__(self, *, session_id: str, workspace_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Router Session identifier must be non-empty")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("Router Workspace identifier must be non-empty")
        self._session_id = session_id
        self._workspace_id = workspace_id

    def validate(self, decision: object) -> dict[str, Any]:
        if not isinstance(decision, dict):
            raise ValueError("Router decision must be an object")
        action = decision.get("action")
        if action not in {"ask_user", "dispatch", "finish"}:
            raise ValueError("Router decision action is invalid")
        if any(
            key in decision and decision[key] != expected
            for key, expected in (
                ("session_id", self._session_id),
                ("workspace_id", self._workspace_id),
            )
        ):
            raise ValueError("Router decision ownership does not match the active Session")
        if action == "finish":
            self._require_fields(decision, {"action", "response", "session_id", "workspace_id"})
            response = decision.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Router finish response is required")
            return {"action": "finish", "response": response}
        if action == "ask_user":
            self._require_fields(decision, {"action", "question", "session_id", "workspace_id"})
            question = decision.get("question")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise ValueError("Router question must contain 1 to 2000 characters")
            return {"action": "ask_user", "question": question}
        self._require_fields(
            decision,
            {"action", "task_id", "task", "context", "session_id", "workspace_id"},
        )
        task_id, task = decision.get("task_id"), decision.get("task")
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
            or len(task_id) > 128
            or not isinstance(task, str)
            or not task.strip()
            or len(task) > 8000
        ):
            raise ValueError("Router dispatch requires bounded task_id and task")
        context = decision.get("context", [])
        if (
            not isinstance(context, list)
            or len(context) > 10
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 1000
                for item in context
            )
            or sum(len(item) for item in context) > 4000
        ):
            raise ValueError("Router dispatch context must contain at most 10 bounded strings")
        return {
            "action": "dispatch",
            "task_id": task_id,
            "task": task,
            **({} if "context" not in decision else {"context": list(context)}),
        }

    @staticmethod
    def _require_fields(decision: dict[object, object], allowed: set[str]) -> None:
        if set(decision) - allowed:
            raise ValueError("Router decision contains unsupported fields")


class _WorkerContextItem(TypedDict):
    text: Annotated[str, ToolParam(min_length=1, max_length=1000)]


class DispatchWorkerTool(BaseTool):
    """Execute one validated Worker dispatch through the Service-provided handler."""

    name = "dispatch_worker"
    description = "Assign one bounded task and explicit context to a persistent Worker."
    required = ("task_id", "task", "context")
    _contextual = True
    task_id: Annotated[str, ToolParam(min_length=1, max_length=128)]
    task: Annotated[str, ToolParam(min_length=1, max_length=8000)]
    context: Annotated[list[_WorkerContextItem], ToolParam(max_items=10)]

    def __init__(self, validator: RouterDecisionValidator) -> None:
        self._validator = validator
        self._task_ids: set[str] = set()

    async def execute(
        self, *, task_id: str, task: str, context: list[_WorkerContextItem]
    ) -> str:
        del task_id, task, context
        raise ToolError("dispatch_worker requires the active foreground context.")

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"task_id", "task", "context"}:
            raise ToolError("dispatch_worker accepts only task_id, task, and context.")
        return await super().prepare_arguments({"context": [], **arguments})

    def validate_arguments(self, /, **arguments: Any) -> None:
        try:
            normalized = {
                **arguments,
                "context": _normalize_context_items(arguments.get("context")),
            }
            self._validator.validate({"action": "dispatch", **normalized})
        except ValueError as error:
            raise ToolError(str(error)) from error

    async def execute_authorized_for_context(
        self,
        arguments: dict[str, object],
        authorization: ToolAuthorizationSession,
        *,
        context: ToolRunContext,
        **kwargs: object,
    ) -> str:
        del authorization, kwargs
        if context.task_id is None or context.execution_id is None:
            raise ToolError("dispatch_worker requires an active foreground Task.")
        try:
            normalized = {
                **arguments,
                "context": _normalize_context_items(arguments.get("context")),
            }
            decision = self._validator.validate({"action": "dispatch", **normalized})
        except ValueError as error:
            raise ToolError(str(error)) from error
        task_id = decision["task_id"]
        if task_id in self._task_ids:
            raise ToolError("dispatch_worker task_id must be unique in this Router run.")
        handler = context.worker_dispatch_handler
        if handler is None:
            raise ToolError("dispatch_worker is unavailable in this execution context.")
        if context.tool_call_id is None:
            raise ToolError("dispatch_worker requires an active Tool call identity.")
        try:
            result = await handler(
                task_id, decision["task"], decision["context"], context.tool_call_id
            )
        except ToolError:
            raise
        except Exception as error:
            raise ToolError("Worker dispatch failed before a report was delivered.") from error
        self._task_ids.add(task_id)
        return result


class RouterAskUserTool(AskUserTool):
    """Keep Router ownership fields outside model-authored question arguments."""

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"question", "options"}:
            raise ToolError("ask_user accepts only question and options.")
        return await super().prepare_arguments(arguments)


class FinishRouterTool(BaseTool):
    """Return a validated final response from the foreground Router."""

    name = "finish"
    description = "Finish the current foreground request with a final response."
    required = ("response",)
    response: Annotated[str, ToolParam(min_length=1, max_length=20000)]

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"response"}:
            raise ToolError("finish accepts only response.")
        response = arguments.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ToolError("Router finish response is required.")
        return await super().prepare_arguments(arguments)

    async def execute(self, *, response: str) -> str:
        return response


__all__ = [
    "DispatchWorkerTool", "FinishRouterTool", "RouterAskUserTool", "RouterCatalog",
    "RouterDecisionValidator",
]


def _normalize_context_items(value: object) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, dict)
        or set(item) != {"text"}
        or not isinstance(item.get("text"), str)
        for item in value
    ):
        raise ValueError("Worker dispatch context must be a list of text entries")
    return [cast(str, item["text"]) for item in value]
