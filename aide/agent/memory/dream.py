"""Dedicated Dream execution for semantic Long-term Memory updates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from loguru import logger

from aide.agent.context.budget import request_fits_model_context
from aide.agent.memory.manager import (
    MemoryEditMismatchError,
    MemoryEditReadError,
    MemoryEditWriteError,
    MemoryManager,
    MemoryPathDeniedError,
    SummaryClaimError,
)
from aide.agent.tools.base import BaseTool, ToolError, ToolParam
from aide.agent.tools.tool_gateway import ToolGateway
from aide.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from aide.logging.session import without_session_log
from aide.provider.errors import ModelCallError, model_context_overflow_error
from aide.provider.model_router import ModelAttemptGuard, ModelRouteStatus
from aide.provider.models import ModelMessages, ModelResponse
from aide.templates import render_template
from aide.utils.validation import require_nonnegative_int

_MEMORY_JSON_TRANSLATION = str.maketrans({"`": r"\u0060"})


@dataclass(frozen=True, slots=True)
class DreamResult:
    """Observable summary returned by one Dream execution."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            raise ValueError("memory_updated must be a boolean")


class DreamEditFileTool(BaseTool):
    """Edit only the Long-term Memory target owned by MemoryManager."""

    name = "edit_file"
    description = "Replace exact text only in the current Long-term Memory file."
    required = ("path", "old_text", "new_text")

    path: Annotated[str, ToolParam(description="Exact Long-term Memory file path.")]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        on_edit: Callable[[], None] | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._on_edit = on_edit

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        _require_long_term_path(path, expected=self._memory_manager.long_term_path)
        try:
            await self._memory_manager.edit_long_term(
                old=old_text,
                new=new_text,
                replace_all=replace_all,
            )
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
        except MemoryEditMismatchError as error:
            raise ToolError(str(error)) from error
        except MemoryEditReadError as error:
            raise ToolError("Long-term Memory could not be read.") from error
        except MemoryEditWriteError as error:
            raise ToolError("Long-term Memory could not be updated.") from error
        if self._on_edit is not None:
            self._on_edit()
        return "Long-term Memory updated."


class DreamModelRouter(Protocol):
    """The one-shot memory-route seam consumed only by Dream."""

    async def complete(
        self,
        route: Literal["memory"],
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse: ...


class Dream:
    """Run one budget-checked Memory decision and apply its edits in order."""

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        model_router: DreamModelRouter,
        batch_size: int,
        memory_route_status: ModelRouteStatus,
    ) -> None:
        require_nonnegative_int(batch_size, field="batch_size")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if memory_route_status.requested_route != "memory":
            raise ValueError("Dream requires the memory Model Route")
        self._memory_manager = memory_manager
        self._model_router = model_router
        self._batch_size = batch_size
        self._memory_route_status = memory_route_status
        self._failure_diagnostic: Exception | None = None
        self._memory_updated = False
        self._tool_gateway = ToolGateway._for_memory(
            (
                DreamEditFileTool(
                    memory_manager=memory_manager,
                    on_edit=self._record_memory_update,
                ),
            ),
            on_failure=self._capture_terminal_failure,
        )
        self._running = False
        self._running_cursor = 0
        self._task: asyncio.Task[DreamResult | None] | None = None
        self._closed = False
        self._aborted = False

    async def run(self) -> DreamResult:
        with without_session_log():
            if self._closed or self._aborted:
                raise RuntimeError("Dream is no longer active")
            if self._running:
                return DreamResult(
                    status="Memory Task is already running.",
                    processed_count=0,
                    memory_updated=False,
                    cursor=self._running_cursor,
                    error=ErrorInfo(
                        code="memory_task_running",
                        message="A Memory Task is already running.",
                    ),
                )
            self._running = True
            self._running_cursor = 0
            self._failure_diagnostic = None
            task = asyncio.current_task()
            self._task = task
            try:
                result = await self._run_once()
                self._log_failure(result)
                return result
            finally:
                self._running = False
                if self._task is task:
                    self._task = None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.wait_until_idle()

    async def wait_until_idle(self) -> None:
        """Wait for an active Dream run without cancelling its work."""
        task = self._task
        if task is None or task is asyncio.current_task() or task.done():
            return
        await asyncio.shield(task)

    @property
    def is_running(self) -> bool:
        return self._running

    def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._closed = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    async def abort_and_wait(self) -> None:
        """Cancel an active Dream and wait until its run releases ownership."""
        self.abort()
        task = self._task
        if task is None or task is asyncio.current_task() or task.done():
            return
        await asyncio.gather(task, return_exceptions=True)

    async def _run_once(self) -> DreamResult:
        self._memory_updated = False
        try:
            claim = await self._memory_manager.claim_summaries(self._batch_size)
        except SummaryClaimError as failure:
            self._capture_terminal_failure(failure.cause)
            if failure.phase == "write_cursor":
                return _cursor_write_failure(cursor=failure.cursor)
            return _state_read_failure(cursor=failure.cursor)
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return _state_read_failure(cursor=self._running_cursor)

        self._running_cursor = claim.cursor
        if not claim.entries:
            return DreamResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=claim.cursor,
            )

        try:
            long_term_memory = await self._memory_manager.read_long_term()
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return _state_read_failure(cursor=claim.cursor)

        records = "\n".join(
            json.dumps(
                entry.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).translate(_MEMORY_JSON_TRANSLATION)
            for entry in claim.entries
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "memory-task-prompt.md", long_term_path=self._memory_manager.long_term_path
                ),
            },
            {
                "role": "user",
                "content": (
                    "## Long-term Memory\n\n"
                    f"{long_term_memory}\n\n"
                    "## Summary Cursor\n\n"
                    f"{claim.previous_cursor}\n\n"
                    "## Conversation Summaries\n\n"
                    "```jsonl\n"
                    f"{records}\n"
                    "```"
                ),
            },
        ]
        tools = tuple(self._tool_gateway.schemas)
        if not request_fits_model_context(
            messages,
            tools,
            context_window=self._memory_route_status.context_window,
            max_output=self._memory_route_status.max_output,
        ):
            overflow = model_context_overflow_error()
            self._capture_terminal_failure(overflow)
            return _model_failure(cursor=claim.cursor, error=overflow.error)

        try:
            response = await self._model_router.complete(
                "memory",
                messages=messages,
                tools=tools,
                guard=lambda status, attempt_messages, attempt_tools: request_fits_model_context(
                    attempt_messages,
                    attempt_tools,
                    context_window=status.context_window,
                    max_output=status.max_output,
                ),
            )
        except asyncio.CancelledError:
            raise
        except ModelCallError as model_error:
            self._capture_terminal_failure(model_error)
            return _model_failure(
                cursor=claim.cursor,
                memory_updated=self._memory_updated,
                error=model_error.error,
            )

        if response.finish_reason == "cancelled":
            return _model_failure(
                cursor=claim.cursor,
                memory_updated=False,
                error=_finish_reason_error("cancelled"),
            )
        if response.finish_reason == "length":
            return _model_failure(
                cursor=claim.cursor,
                memory_updated=False,
                error=_finish_reason_error("length"),
            )

        for tool_call in response.message.tool_calls:
            result = await self._tool_gateway.call(tool_call)
            if result.status != "success":
                return _model_failure(
                    cursor=claim.cursor,
                    memory_updated=self._memory_updated,
                    error=ErrorInfo("tool_failed", result.content),
                )

        memory_updated = self._memory_updated
        count = len(claim.entries)
        noun = "summary" if count == 1 else "summaries"
        outcome = "updated" if memory_updated else "unchanged"
        return DreamResult(
            status=f"Processed {count} {noun}; Long-term Memory {outcome}.",
            processed_count=count,
            memory_updated=memory_updated,
            cursor=claim.cursor,
        )

    def _capture_terminal_failure(self, error: Exception) -> None:
        self._failure_diagnostic = error

    def _record_memory_update(self) -> None:
        self._memory_updated = True

    def _log_failure(self, result: DreamResult) -> None:
        if result.error is None:
            return
        if self._failure_diagnostic is None:
            logger.error("Memory Task failed code={}", result.error.code)
            return
        diagnostic = self._failure_diagnostic
        logger.opt(
            exception=(type(diagnostic), diagnostic, diagnostic.__traceback__),
        ).error("Memory Task failed code={}", result.error.code)


def _require_long_term_path(path: str, *, expected: Path) -> None:
    if Path(path) != expected:
        raise ToolError("Memory Tasks may access only Long-term Memory.")


def _finish_reason_error(finish_reason: Literal["length", "cancelled"]) -> ErrorInfo:
    if finish_reason == "cancelled":
        return ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)
    return ErrorInfo("model_failed", "Memory Task model response reached its output limit.")


def _model_failure(
    *,
    cursor: int,
    error: ErrorInfo,
    memory_updated: bool = False,
) -> DreamResult:
    return DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=memory_updated,
        cursor=cursor,
        error=error,
    )


def _state_read_failure(*, cursor: int) -> DreamResult:
    return DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=cursor,
        error=ErrorInfo(
            code="persistence_error",
            message="Memory Task state could not be read.",
        ),
    )


def _cursor_write_failure(*, cursor: int) -> DreamResult:
    return DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=cursor,
        error=ErrorInfo(
            code="persistence_error",
            message="Summary Cursor could not be updated.",
        ),
    )


__all__ = [
    "Dream",
    "DreamEditFileTool",
    "DreamModelRouter",
    "DreamResult",
]
