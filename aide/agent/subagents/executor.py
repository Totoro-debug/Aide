"""Run one registered SubAgent through an isolated Agent Runner context."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from aide.agent.context.run_context import (
    AgentRunContextController,
    AgentRunContextRequestPreparer,
    AgentRunContextSnapshot,
    AgentRunTerminalCommitValues,
    ConversationSummaryAppender,
    agent_run_attempt_guard,
)
from aide.agent.permission import PermissionSnapshot, ToolPermissionLevel
from aide.agent.runner import (
    AgentRunner,
    AgentRunnerOutput,
    AgentRunnerResponseSegmentEnd,
    AgentRunnerResult,
    AgentRunnerToolCallFinished,
    AgentRunnerToolCallStarted,
)
from aide.agent.subagents.context import SubAgentToolContext
from aide.agent.subagents.models import (
    SubAgentError,
    SubAgentEvent,
    SubAgentEventKind,
    SubAgentExecutionResult,
    SubAgentRecord,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.ports import SubAgentRecordRepository
from aide.agent.subagents.store import SubAgentStoreError
from aide.agent.tools.base import BaseTool
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.file_mutation import FileMutationRecorder
from aide.agent.tools.tool_gateway import ConfirmationRequester, ToolGateway, ToolResult
from aide.agent.workspace_state import WorkspaceState
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter, RunModelRouter
from aide.provider.models import ReasoningDelta, SessionModelConfiguration, TextDelta
from aide.utils.validation import empty_token_usage

_DISALLOWED_TOOL_NAMES = ("spawn_agent", "wait_agent", "schedule")
_ARTIFACT_TOOL_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_RUNNER_USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "total_tokens")

type SubAgentConfirmationFactory = Callable[[SubAgentRecord], ConfirmationRequester | None]
type SubAgentFileMutationRecorderFactory = Callable[[SubAgentRecord], FileMutationRecorder | None]


@dataclass(slots=True)
class _Cancellation:
    event: asyncio.Event
    interrupted: bool = False
    runner_task: asyncio.Task[AgentRunnerResult] | None = None

    def request(self, *, interrupted: bool) -> None:
        self.interrupted = self.interrupted or interrupted
        if not self.event.is_set():
            self.event.set()
            if (
                self.runner_task is not None
                and not self.runner_task.done()
                and asyncio.current_task() is not self.runner_task
            ):
                self.runner_task.cancel()


class _SubAgentSummaryAppender(ConversationSummaryAppender):
    def __init__(
        self,
        repository: SubAgentRecordRepository,
        current_record: list[SubAgentRecord],
    ) -> None:
        self._repository = repository
        self._current_record = current_record

    async def append_summary(self, content: str, timestamp: datetime) -> None:
        record = self._current_record[0]
        context_state = deepcopy(record.context_state or {})
        summaries = context_state.setdefault("conversation_summaries", [])
        if not isinstance(summaries, list):
            raise ValueError("SubAgent conversation summaries are malformed")
        summaries.append({"timestamp": timestamp.isoformat(), "content": content})
        self._current_record[0] = self._repository.save(
            replace(
                record,
                context_state=context_state,
                revision=record.revision + 1,
            )
        )


class SubAgentRunnerExecutor:
    """Execute and checkpoint one registered SubAgent without owning its terminal update."""

    def __init__(
        self,
        *,
        workspace_id: str,
        workspace_state: WorkspaceState,
        repository: SubAgentRecordRepository,
        model_router: ModelRouter,
        tool_gateway: ToolGateway,
        compact_ratio: float,
        max_iterations: int,
        max_tool_result_chars: int,
        enable_tool_micro_compression: bool = False,
        mcp_keywords: Mapping[str, Sequence[str]] | None = None,
        confirmation_for: SubAgentConfirmationFactory | None = None,
        file_mutation_recorder_for: SubAgentFileMutationRecorderFactory | None = None,
        tool_context_for: Callable[[SubAgentRecord], SubAgentToolContext] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("SubAgent Workspace ID must not be empty")
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("SubAgent execution requires a WorkspaceState")
        if not isinstance(model_router, ModelRouter):
            raise TypeError("SubAgent execution requires a shared ModelRouter")
        if not isinstance(tool_gateway, ToolGateway):
            raise TypeError("SubAgent execution requires a base ToolGateway")
        base_context = tool_gateway.tool_context
        if base_context is None or base_context.exec_host is None:
            raise ValueError("SubAgent execution requires a prepared Exec Host context")
        if base_context.workspace != workspace_state.workspace_path:
            raise ValueError("SubAgent Tool Gateway belongs to a different Workspace")
        if isinstance(compact_ratio, bool) or not 0 < compact_ratio < 1:
            raise ValueError("SubAgent compact ratio must be between zero and one")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 50
        ):
            raise ValueError("SubAgent max iterations must be an integer of at least 50")
        if (
            isinstance(max_tool_result_chars, bool)
            or not isinstance(max_tool_result_chars, int)
            or max_tool_result_chars < 1
        ):
            raise ValueError("SubAgent Tool result limit must be a positive integer")

        self._workspace_id = workspace_id
        self._workspace_state = workspace_state
        self._repository = repository
        self._model_router = model_router
        self._tool_gateway = tool_gateway
        self._tool_context = base_context
        self._exec_host = base_context.exec_host
        self._compact_ratio = compact_ratio
        self._max_iterations = max_iterations
        self._max_tool_result_chars = max_tool_result_chars
        self._enable_tool_micro_compression = enable_tool_micro_compression
        self._mcp_keywords = {} if mcp_keywords is None else deepcopy(dict(mcp_keywords))
        self._confirmation_for = confirmation_for
        self._file_mutation_recorder_for = file_mutation_recorder_for
        self._tool_context_for = tool_context_for
        self._now = now or (lambda: datetime.now(UTC))
        self._cancellations: dict[str, _Cancellation] = {}
        self._started_agent_ids: set[str] = set()

    def request_cancel(self, agent_id: str, *, interrupted: bool = False) -> bool:
        cancellation = self._cancellations.get(agent_id)
        if cancellation is None:
            return False
        cancellation.request(interrupted=interrupted)
        return True

    async def execute(
        self,
        record: SubAgentRecord,
        *,
        emit: Callable[[SubAgentEvent], Awaitable[None]],
    ) -> SubAgentExecutionResult:
        if not isinstance(record, SubAgentRecord):
            raise TypeError("SubAgent execution requires a registered record")
        if record.status is not SubAgentStatus.RUNNING:
            raise ValueError("SubAgent execution requires a running record")
        if record.session_id != self._repository.session_id:
            raise ValueError("SubAgent record belongs to a different Session")
        if record.agent_id in self._started_agent_ids:
            raise ValueError("A SubAgent can be started only once")
        if record.agent_id in self._cancellations:
            raise ValueError("SubAgent execution is already active")

        current_record = [record]
        cancellation = _Cancellation(event=asyncio.Event())
        self._cancellations[record.agent_id] = cancellation
        event_revision = record.revision
        runner_message_count = 0
        latest_runner_usage = empty_token_usage()
        controller: AgentRunContextController | None = None
        artifact_paths: list[str] = list(record.artifact_paths)
        try:
            run_router, run_gateway = self._create_run_resources(record)
            if not current_record[0].conversation:
                current = current_record[0]
                current_record[0] = self._repository.save(
                    replace(
                        current,
                        conversation=({"role": "user", "content": record.task},),
                        revision=current.revision + 1,
                    )
                )
            summary_appender = _SubAgentSummaryAppender(self._repository, current_record)
            context_state = deepcopy(record.context_state or {})
            last_compacted = _last_compacted(context_state, len(current_record[0].conversation))
            run_controller = AgentRunContextController(
                snapshot=AgentRunContextSnapshot(
                    messages=current_record[0].conversation,
                    metadata=context_state,
                    last_compacted=last_compacted,
                ),
                provider=run_router,
                memory_manager=summary_appender,
                now=self._now,
            )
            controller = run_controller
            request_preparer = AgentRunContextRequestPreparer(
                run_controller,
                router=run_router,
                requested_route="chat",
                project_messages=lambda messages: [
                    {"role": "system", "content": record.creator_snapshot.system_prompt},
                    *deepcopy(list(messages)),
                ],
                current_user=None,
                compact_ratio=self._compact_ratio,
                enable_tool_micro_compression=self._enable_tool_micro_compression,
            )
            runner = AgentRunner(run_router, request_preparer)

            async def publish(kind: SubAgentEventKind, data: dict[str, Any]) -> None:
                nonlocal event_revision
                event_revision = max(event_revision, current_record[0].revision) + 1
                await emit(
                    SubAgentEvent(
                        kind=kind,
                        workspace_id=self._workspace_id,
                        session_id=record.session_id,
                        agent_id=record.agent_id,
                        revision=event_revision,
                        occurred_at=self._now(),
                        data=deepcopy(data),
                    )
                )

            async def on_output(event: AgentRunnerOutput) -> None:
                if isinstance(event, TextDelta | ReasoningDelta):
                    await publish(
                        SubAgentEventKind.OUTPUT,
                        {"type": event.type, "delta": event.delta},
                    )
                elif isinstance(event, AgentRunnerResponseSegmentEnd):
                    await publish(
                        SubAgentEventKind.OUTPUT,
                        {"type": event.type, "segment": event.segment},
                    )
                elif isinstance(event, AgentRunnerToolCallStarted):
                    await publish(
                        SubAgentEventKind.ACTIVITY,
                        {
                            "type": event.type,
                            "tool_call_id": event.tool_call_id,
                            "tool_name": event.tool_name,
                            "arguments": event.arguments,
                        },
                    )
                elif isinstance(event, AgentRunnerToolCallFinished):
                    await publish(
                        SubAgentEventKind.ACTIVITY,
                        {
                            "type": event.type,
                            "tool_call_id": event.tool_call_id,
                            "tool_name": event.tool_name,
                            "status": event.status,
                            "result": event.result,
                        },
                    )

            async def checkpoint(
                messages: Sequence[dict[str, Any]],
                runner_usage: dict[str, int],
            ) -> None:
                nonlocal runner_message_count, latest_runner_usage
                latest_runner_usage = dict(runner_usage)
                current = current_record[0]
                appended = deepcopy(list(messages[runner_message_count:]))
                next_conversation = [*current.conversation, *appended]
                context_values = run_controller.terminal_commit_values()
                next_context_state = _context_state(current, context_values)
                usage = _combined_usage(runner_usage, context_values.usage_delta)
                next_artifact_paths = tuple(
                    dict.fromkeys((*current.artifact_paths, *artifact_paths))
                )
                if (
                    not appended
                    and next_context_state == current.context_state
                    and usage == current.usage
                    and next_artifact_paths == current.artifact_paths
                ):
                    runner_message_count = len(messages)
                    return
                current_record[0] = self._repository.save(
                    replace(
                        current,
                        conversation=tuple(next_conversation),
                        context_state=next_context_state,
                        artifact_paths=next_artifact_paths,
                        usage=usage,
                        revision=max(current.revision + 1, event_revision),
                    ),
                    expected_revision=current.revision,
                )
                runner_message_count = len(messages)
                await publish(SubAgentEventKind.USAGE, {"usage": usage})

            externalize_result = self._result_externalizer_for(record, artifact_paths)
            recorder = (
                self._file_mutation_recorder_for(record)
                if self._file_mutation_recorder_for is not None
                and record.source.kind is SubAgentSourceKind.FOREGROUND
                else None
            )
            run_token = (
                UUID(record.source.restore_run_token)
                if record.source.kind is SubAgentSourceKind.FOREGROUND
                and record.source.restore_run_token is not None
                else None
            )
            confirmation = (
                None if self._confirmation_for is None else self._confirmation_for(record)
            )
            self._started_agent_ids.add(record.agent_id)
            runner_task = asyncio.create_task(
                runner.run(
                    (),
                    model="chat",
                    tool_gateway=run_gateway,
                    on_output=on_output,
                    confirmation=confirmation,
                    externalize_result=externalize_result,
                    cancel_requested=cancellation.event.is_set,
                    max_iterations=self._max_iterations,
                    file_mutation_recorder=recorder,
                    run_token=run_token,
                    on_checkpoint=checkpoint,
                )
            )
            cancellation.runner_task = runner_task
            while True:
                try:
                    runner_result = await asyncio.shield(runner_task)
                    break
                except asyncio.CancelledError:
                    if runner_task.cancelled():
                        return _cancelled_execution(
                            current_record[0],
                            artifact_paths,
                            interrupted=cancellation.interrupted,
                        )
                    cancellation.request(interrupted=True)

            await checkpoint(runner_result.messages, runner_result.usage)
            return _execution_result(
                current_record[0],
                runner_result,
                run_controller.terminal_commit_values(),
                artifact_paths,
                interrupted=cancellation.interrupted,
            )
        except asyncio.CancelledError:
            cancellation.interrupted = True
            cancellation.event.set()
            return _cancelled_execution(
                current_record[0],
                artifact_paths,
                interrupted=True,
            )
        except Exception as error:
            usage = latest_runner_usage
            if controller is not None:
                usage = _combined_usage(usage, controller.terminal_commit_values().usage_delta)
            return _failed_execution(current_record[0], artifact_paths, error, usage=usage)
        finally:
            self._cancellations.pop(record.agent_id, None)

    def _create_run_resources(self, record: SubAgentRecord) -> tuple[RunModelRouter, ToolGateway]:
        creator = record.creator_snapshot
        run_router = RunModelRouter(
            self._model_router,
            guard=agent_run_attempt_guard,
            session_model_configuration=SessionModelConfiguration(
                creator.provider_id,
                creator.model,
                creator.reasoning_effort,
            ),
        )
        base_context = self._tool_context
        workspace = self._workspace_state.workspace_path
        shell = self._exec_host.resolved_shell
        if (creator.shell or "auto") != shell.selector:
            raise ValueError("SubAgent snapshot shell does not match the shared Exec Host")
        tool_context = ToolRunContext(
            workspace=workspace,
            schedule_service=base_context.schedule_service,
            exec_host=self._exec_host,
            subagent=(None if self._tool_context_for is None else self._tool_context_for(record)),
        )
        permission_snapshot = PermissionSnapshot(
            level=cast(ToolPermissionLevel, creator.permission_level),
            exec_shell=shell,
        )
        return run_router, build_agent_run_gateway(
            self._tool_gateway,
            excluded_names=_DISALLOWED_TOOL_NAMES,
            allowed_names=tuple(schema["name"] for schema in creator.tool_schemas),
            mcp_keywords=self._mcp_keywords,
            permission_snapshot=permission_snapshot,
            tool_context=tool_context,
        )

    def _result_externalizer_for(
        self,
        record: SubAgentRecord,
        artifact_paths: list[str],
    ) -> Callable[[ToolResult], ToolResult] | None:
        if self._max_tool_result_chars <= 0:
            return None

        def externalize(result: ToolResult) -> ToolResult:
            if (
                result.status != "success"
                or len(result.content) <= self._max_tool_result_chars
                or result.artifact is not None
            ):
                return result
            content = BaseTool.handle_result(
                result.content,
                workspace=self._workspace_state.workspace_path,
                session_id=record.session_id,
                tool_call_id=_artifact_id(record.agent_id, result.tool_call_id),
                limit=self._max_tool_result_chars,
            )
            if content.artifact is not None:
                artifact_paths.append(content.artifact.path)
            return replace(result, content=content.content, artifact=content.artifact)

        return externalize


def _artifact_id(agent_id: str, tool_call_id: str) -> str:
    safe_call_id = tool_call_id if _ARTIFACT_TOOL_CALL_ID.fullmatch(tool_call_id) else str(uuid4())
    return f"{agent_id}_{safe_call_id}"


def _last_compacted(context_state: Mapping[str, Any], message_count: int) -> int:
    value = cast(object, context_state.get("last_compacted", 0))
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= message_count:
        raise ValueError("SubAgent compaction cursor is invalid")
    return value


def _context_state(
    record: SubAgentRecord,
    values: AgentRunTerminalCommitValues,
) -> dict[str, Any]:
    context_state = deepcopy(record.context_state or {})
    context_state["last_compacted"] = values.pending_last_compacted
    if values.pending_action_summary is None:
        context_state.pop("summary", None)
    else:
        context_state["summary"] = values.pending_action_summary
    return context_state


def _combined_usage(*values: Mapping[str, int]) -> dict[str, int]:
    total = empty_token_usage()
    for value in values:
        for field in _RUNNER_USAGE_FIELDS:
            total[field] += value[field]
    return total


def _execution_result(
    record: SubAgentRecord,
    runner_result: AgentRunnerResult,
    context_values: AgentRunTerminalCommitValues,
    artifact_paths: Sequence[str],
    *,
    interrupted: bool,
) -> SubAgentExecutionResult:
    conversation = tuple(record.conversation)
    finish_reason = runner_result.finish_reason
    error = (
        None
        if runner_result.error is None
        else SubAgentError(
            code=runner_result.error.code,
            message=runner_result.error.message,
        )
    )
    if interrupted:
        status = SubAgentStatus.INTERRUPTED
        error = SubAgentError(
            code="service_interrupted",
            message="The SubAgent was interrupted while it was running.",
        )
    elif finish_reason == "completed":
        status = SubAgentStatus.COMPLETED
    elif finish_reason == "cancelled":
        status = SubAgentStatus.CANCELLED
    else:
        status = SubAgentStatus.FAILED
    result = (
        runner_result.final_content
        if status is SubAgentStatus.COMPLETED
        else runner_result.final_content or None
    )
    context_state = _context_state(record, context_values)
    return SubAgentExecutionResult(
        status=status,
        conversation=conversation,
        context_state=context_state,
        artifact_paths=tuple(dict.fromkeys((*record.artifact_paths, *artifact_paths))),
        result=result,
        error=error,
        usage=_combined_usage(runner_result.usage, context_values.usage_delta),
    )


def _failed_execution(
    record: SubAgentRecord,
    artifact_paths: Sequence[str],
    error: Exception,
    *,
    usage: Mapping[str, int],
) -> SubAgentExecutionResult:
    if isinstance(error, ModelCallError):
        failure = SubAgentError(code=error.error.code, message=error.error.message)
    elif isinstance(error, SubAgentStoreError | OSError):
        failure = SubAgentError(
            code="persistence_error", message="SubAgent state could not be saved."
        )
    else:
        failure = SubAgentError(code="agent_failed", message="SubAgent execution failed.")
    return SubAgentExecutionResult(
        status=SubAgentStatus.FAILED,
        conversation=tuple(record.conversation),
        context_state=deepcopy(record.context_state or {}),
        artifact_paths=tuple(dict.fromkeys((*record.artifact_paths, *artifact_paths))),
        result=None,
        error=failure,
        usage=dict(usage),
    )


def _cancelled_execution(
    record: SubAgentRecord,
    artifact_paths: Sequence[str],
    *,
    interrupted: bool,
) -> SubAgentExecutionResult:
    status = SubAgentStatus.INTERRUPTED if interrupted else SubAgentStatus.CANCELLED
    error = (
        SubAgentError(
            code="service_interrupted",
            message="The SubAgent was interrupted while it was running.",
        )
        if interrupted
        else SubAgentError(code="turn_cancelled", message="The SubAgent was cancelled.")
    )
    return SubAgentExecutionResult(
        status=status,
        conversation=tuple(record.conversation),
        context_state=deepcopy(record.context_state or {}),
        artifact_paths=tuple(dict.fromkeys((*record.artifact_paths, *artifact_paths))),
        result=None,
        error=error,
        usage=deepcopy(record.usage or empty_token_usage()),
    )
