"""Execute one persistent Worker through the shared Agent Runner and Tool Gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from aide.agent.context.builder import ContextBuilder
from aide.agent.context.run_context import AgentRunContextSnapshot, ContextController
from aide.agent.permission import PermissionSnapshot
from aide.agent.runner import AgentRunner
from aide.agent.session.backup_store import FileBackupStore
from aide.agent.subagents.context import SubAgentToolContext
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.ports import SubAgentSessionCoordinator
from aide.agent.tools.context import ToolRunContext
from aide.agent.tools.core.exec_host import ExecHost
from aide.agent.tools.deferred import build_agent_run_gateway
from aide.agent.tools.tool_gateway import ConfirmationRequester, ToolGateway, ToolResult
from aide.agent.workers.models import WorkerRecord, WorkerReportStatus
from aide.agent.workers.store import WorkerStore, WorkerStoreConflict, WorkerStoreError
from aide.agent.workspace_state import WorkspaceState
from aide.provider.errors import ModelCallError
from aide.provider.model_router import ModelRouter, RunModelRouter
from aide.trace.context import TraceRootSource, workspace_key_for
from aide.trace.recorder import NoOpTraceRecorder, TraceRecorder
from aide.trace.scope import current_trace_scope, trace_scope
from aide.trace.types import TraceOrigin, TracePurpose
from aide.utils.errors import ErrorInfo
from aide.utils.json import strict_json_loads
from aide.utils.validation import empty_token_usage

_SUBAGENT_TOOL_NAMES = ("spawn_agent", "wait_agent", "list_agents")
_USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "total_tokens")


class WorkerRunnerExecutor:
    """Run one Worker with its own persisted context and captured capabilities."""

    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        store: WorkerStore,
        model_router: ModelRouter,
        tool_gateway: ToolGateway,
        exec_host: ExecHost,
        schedule_service: Any,
        compact_ratio: float,
        max_iterations: int,
        max_tool_result_chars: int,
        enable_tool_micro_compression: bool,
        mcp_keywords: Mapping[str, Sequence[str]],
        schedule_session: bool = False,
        subagent_coordinator: SubAgentSessionCoordinator | None = None,
        trace_recorder: TraceRecorder | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            tool_gateway.tool_context is None
            or tool_gateway.tool_context.workspace != workspace_state.workspace_path
        ):
            raise ValueError("Worker Tool Gateway belongs to a different Workspace")
        if tool_gateway.tool_context.exec_host != exec_host:
            raise ValueError("Worker Tool Gateway must retain the shared Exec Host")
        if not 0 < compact_ratio < 1:
            raise ValueError("Worker compact ratio must be between zero and one")
        if max_iterations < 1 or max_tool_result_chars < 1:
            raise ValueError("Worker execution limits must be positive")
        if not isinstance(schedule_session, bool):
            raise ValueError("Worker Schedule Session flag must be a boolean")
        self._workspace_state = workspace_state
        self._store = store
        self._model_router = model_router
        self._tool_gateway = tool_gateway
        self._exec_host = exec_host
        self._schedule_service = schedule_service
        self._compact_ratio = compact_ratio
        self._max_iterations = max_iterations
        self._max_tool_result_chars = max_tool_result_chars
        self._enable_tool_micro_compression = enable_tool_micro_compression
        self._mcp_keywords = deepcopy(dict(mcp_keywords))
        self._schedule_session = schedule_session
        self._subagent_coordinator = subagent_coordinator
        self._trace_recorder = NoOpTraceRecorder() if trace_recorder is None else trace_recorder
        self._now = now or (lambda: datetime.now(UTC))

    async def execute(
        self,
        worker: WorkerRecord,
        *,
        foreground_task_is_current: Callable[[str, int, str], bool],
        confirmation: ConfirmationRequester | None,
        cancel_requested: Callable[[], bool],
        externalize_result: Callable[[ToolResult], ToolResult] | None = None,
    ) -> WorkerRecord:
        if worker.session_id != self._store.session_id:
            raise ValueError("Worker record belongs to a different Session")
        current = self._store.get(worker.worker_id)
        if current is None or current.revision != worker.revision or current.state != "working":
            raise WorkerStoreConflict("Worker is not ready to execute")
        worker = self._store.recover_completed_result(worker.worker_id)
        if worker.report is not None:
            return worker
        if worker.exec_shell_selector != self._exec_host.resolved_shell.selector:
            return self._failed(
                worker,
                ErrorInfo("tool_failed", "The captured Worker Exec shell is unavailable."),
            )

        remaining_iterations = self._max_iterations - worker.usage["model_calls"]
        if remaining_iterations < 1:
            return self._store.publish_report(
                worker.worker_id,
                state="blocked",
                summary="Worker reached its cumulative execution limit.",
                error_code="agent_iteration_limit",
                error_message="Worker reached its cumulative execution limit.",
                usage=worker.usage,
            )

        parent_execution_id = worker.runtime_facts.get("foreground_execution_id")
        if not isinstance(parent_execution_id, str) or not parent_execution_id:
            return self._failed(
                worker,
                ErrorInfo("model_failed", "The Worker parent execution is unavailable."),
            )
        permission_snapshot = PermissionSnapshot(
            level=worker.permission_level,
            exec_shell=self._exec_host.resolved_shell,
        )

        def worker_is_current() -> bool:
            current_worker = self._store.get(worker.worker_id)
            return (
                current_worker is not None
                and current_worker.execution_id == worker.execution_id
                and current_worker.state == "working"
                and current_worker.report is None
                and foreground_task_is_current(
                    worker.parent_task_id,
                    worker.parent_generation,
                    parent_execution_id,
                )
            )

        subagents_available = self._subagent_coordinator is not None
        allowed_names = tuple(
            tool.name
            for tool in self._tool_gateway.catalog
            if subagents_available or tool.name not in _SUBAGENT_TOOL_NAMES
        )
        if subagents_available:
            assert self._subagent_coordinator is not None
            creator_tool_names = (*allowed_names, "tool_search")
            if not self._schedule_session:
                creator_tool_names = (*creator_tool_names, "ask_user")
            source = SubAgentSource(
                kind=SubAgentSourceKind.WORKER,
                restore_run_token=worker.restore_run_token,
                task_id=worker.parent_task_id,
                task_generation=worker.parent_generation,
                worker_id=worker.worker_id,
            )
            subagent_snapshot = SubAgentCreatorSnapshot(
                provider_id=worker.model_configuration.provider_id,
                model=worker.model_configuration.model,
                reasoning_effort=worker.model_configuration.reasoning_effort,
                permission_level=worker.permission_level,
                shell=worker.exec_shell_selector,
                tool_names=tuple(dict.fromkeys(creator_tool_names)),
                system_prompt=worker.system_prompt,
            )
            subagent_context = SubAgentToolContext(
                coordinator=self._subagent_coordinator,
                parent_run_id=worker.execution_id,
                source=source,
                creator_snapshot=subagent_snapshot,
                caller_role="worker",
                owner_is_current=worker_is_current,
            )
        else:
            subagent_context = None
        tool_context = ToolRunContext(
            workspace=self._workspace_state.workspace_path,
            schedule_service=self._schedule_service,
            exec_host=self._exec_host,
            subagent=subagent_context,
            task_id=worker.parent_task_id,
            generation=worker.parent_generation,
            execution_id=parent_execution_id,
            foreground_task_is_current=foreground_task_is_current,
        )
        gateway = build_agent_run_gateway(
            self._tool_gateway,
            excluded_names=(
                () if subagents_available else _SUBAGENT_TOOL_NAMES
            ),
            allowed_names=(*allowed_names, "tool_search"),
            mcp_keywords=self._mcp_keywords,
            permission_snapshot=permission_snapshot,
            tool_context=tool_context,
        )
        run_router = RunModelRouter(
            self._model_router,
            session_model_configuration=worker.model_configuration,
        )
        current_record = [worker]
        runner_message_count = 0
        runner_usage = empty_token_usage()

        async def append_summary(content: str, timestamp: datetime) -> None:
            current_worker = current_record[0]
            next_state = deepcopy(current_worker.context_state)
            summaries = next_state.setdefault("conversation_summaries", [])
            if not isinstance(summaries, list):
                raise ValueError("Worker conversation summaries are malformed")
            summaries.append({"timestamp": timestamp.isoformat(), "content": content})
            current_record[0] = self._store.checkpoint(
                current_worker.worker_id,
                conversation=current_worker.conversation,
                context_state=next_state,
                usage=current_worker.usage,
            )

        def project_messages(
            history: Sequence[dict[str, Any]],
            current_user: dict[str, Any] | None,
            increment: Sequence[dict[str, Any]],
            compaction_cursor: int,
            action_summary: str | None,
        ) -> list[dict[str, Any]]:
            return ContextBuilder.build_run_messages(
                history,
                current_user=current_user,
                increment=increment,
                compaction_cursor=compaction_cursor,
                action_summary=action_summary,
                project_messages=lambda messages: ContextBuilder.build_subagent_messages(
                    messages,
                    system_prompt=worker.system_prompt,
                ),
            )

        try:
            cursor = worker.context_state.get("last_compacted", 0)
            if type(cursor) is not int or not 0 <= cursor <= len(worker.conversation):
                raise ValueError("Worker context cursor is invalid")
            controller = ContextController(
                snapshot=AgentRunContextSnapshot(
                    messages=worker.conversation,
                    metadata=worker.context_state,
                    last_compacted=cursor,
                ),
                append_summary=append_summary,
                now=self._now,
                request_router=run_router,
                requested_route="chat",
                project_messages=project_messages,
                project_tool_results=ContextBuilder.project_tool_results,
                compact_ratio=self._compact_ratio,
                enable_tool_micro_compression=self._enable_tool_micro_compression,
            )
            initial_messages = controller.initial_messages()
            runner = AgentRunner(run_router, controller)

            async def checkpoint(
                messages: Sequence[dict[str, Any]],
                usage: dict[str, int],
            ) -> None:
                nonlocal runner_message_count, runner_usage
                runner_usage = dict(usage)
                current_worker = current_record[0]
                appended = deepcopy(list(messages[runner_message_count:]))
                if self._schedule_session and any(
                    isinstance(item.get("confirmation"), dict)
                    and item["confirmation"].get("decision") is None
                    for item in messages if item.get("role") == "tool"
                ):
                    if appended and appended[-1].get("role") == "assistant" and not appended[-1].get("tool_calls"):
                        appended[-1].update(status="error", error={
                            "code": "interactive_terminal_required",
                            "message": "Schedule Worker Tool confirmation is unavailable.",
                        })
                context_values = controller.terminal_commit_values()
                next_state = deepcopy(current_worker.context_state)
                next_state["last_compacted"] = context_values.pending_last_compacted
                if context_values.pending_action_summary is None:
                    next_state.pop("summary", None)
                else:
                    next_state["summary"] = context_values.pending_action_summary
                combined_usage = {
                    field: worker.usage[field] + runner_usage[field] + context_values.usage_delta[field]
                    for field in _USAGE_FIELDS
                }
                current_record[0] = self._store.checkpoint(
                    current_worker.worker_id,
                    conversation=(*current_worker.conversation, *appended),
                    context_state=next_state,
                    usage=combined_usage,
                )
                runner_message_count = len(messages)

            with self._trace_scope(worker):
                result = await runner.run(
                    initial_messages,
                    model="chat",
                    tool_gateway=gateway,
                    on_output=None,
                    confirmation=confirmation,
                    externalize_result=externalize_result,
                    cancel_requested=cancel_requested,
                    max_iterations=remaining_iterations,
                    file_mutation_recorder=(
                        None
                        if self._schedule_session
                        else FileBackupStore(self._workspace_state, worker.session_id)
                    ),
                    run_token=(
                        None if worker.restore_run_token is None else UUID(worker.restore_run_token)
                    ),
                    on_checkpoint=checkpoint,
                )
            await checkpoint(result.messages, result.usage)
            if self._schedule_session and any(
                isinstance(item.get("confirmation"), dict)
                and item["confirmation"].get("decision") is None
                for item in result.messages if item.get("role") == "tool"
            ):
                return self._failed(current_record[0], ErrorInfo(
                    "interactive_terminal_required", "Schedule Worker Tool confirmation is unavailable.",
                ))
            if result.finish_reason == "completed":
                children, unread_notified = self._children_to_wait(current_record[0])
                if unread_notified:
                    return self._store.publish_report(
                        worker.worker_id,
                        state="blocked",
                        summary="Worker finished without reading a notified SubAgent result.",
                        error_code="worker_subagent_result_not_read",
                        error_message="Call wait_agent to read each notified SubAgent result before completing.",
                        usage=current_record[0].usage,
                    )
                if children:
                    current = current_record[0]
                    return self._store.suspend_for_children(
                        worker.worker_id,
                        expected_revision=current.revision,
                        conversation=current.conversation,
                        context_state=current.context_state,
                        usage=current.usage,
                        children=children,
                    )
                current = current_record[0]
                current_record[0] = self._store.finish_completed_segment(
                    worker.worker_id,
                    expected_revision=current.revision,
                    context_state=current.context_state,
                )
                state: WorkerReportStatus = "ready"
                summary = result.final_content or "Worker completed without a text response."
            elif result.finish_reason == "suspended":
                state = "blocked"
                summary = result.final_content or "Worker requires additional input."
            elif result.finish_reason == "cancelled":
                state = "cancelled"
                summary = "Worker execution was cancelled."
            elif result.finish_reason == "max_iterations":
                state = "blocked"
                summary = (
                    result.error.message
                    if result.error is not None
                    else "Worker reached its execution limit."
                )
            else:
                state = "failed"
                summary = (
                    result.error.message if result.error is not None else "Worker execution failed."
                )
            return self._store.publish_report(
                worker.worker_id,
                state=state,
                summary=summary,
                error_code=None if result.error is None else result.error.code,
                error_message=None if result.error is None else result.error.message,
                usage=current_record[0].usage,
            )
        except asyncio.CancelledError:
            self._publish_cancellation(current_record[0])
            raise
        except (OSError, WorkerStoreError):
            # Preserve the checkpoint for recovery; a failed report write must not
            # replace a completed result with an unrelated execution failure.
            raise
        except ModelCallError as error:
            return self._failed(current_record[0], error.error)
        except Exception:
            return self._failed(
                current_record[0],
                ErrorInfo("model_failed", "Worker execution failed."),
            )

    def _children_to_wait(
        self, worker: WorkerRecord
    ) -> tuple[list[dict[str, object]], bool]:
        coordinator = self._subagent_coordinator
        if coordinator is None:
            return [], False
        source = SubAgentSource(
            kind=SubAgentSourceKind.WORKER,
            restore_run_token=worker.restore_run_token,
            task_id=worker.parent_task_id,
            task_generation=worker.parent_generation,
            worker_id=worker.worker_id,
        )
        read_versions = self._read_child_versions(worker.conversation)
        notified_value = worker.context_state.get("child_notifications", [])
        notified_versions = {
            (item.get("agent_id"), item.get("terminal_revision"))
            for item in notified_value
            if isinstance(item, dict)
        } if isinstance(notified_value, list) else set()
        unread_notified = False
        waiting: list[dict[str, object]] = []
        cursor: str | None = None
        while True:
            page = coordinator.list(source=source, cursor=cursor, limit=100)
            for item in page.items:
                record = coordinator.get(item.agent_id)
                if record is None or record.source != source:
                    continue
                terminal = record.status in {
                    SubAgentStatus.COMPLETED,
                    SubAgentStatus.FAILED,
                    SubAgentStatus.CANCELLED,
                    SubAgentStatus.INTERRUPTED,
                }
                if terminal:
                    identity = (record.agent_id, record.revision)
                    if identity in read_versions:
                        continue
                    if identity in notified_versions:
                        unread_notified = True
                        continue
                    waiting.append(
                        {
                            "agent_id": record.agent_id,
                            "status": record.status.value,
                            "terminal_revision": record.revision,
                        }
                    )
                else:
                    waiting.append(
                        {
                            "agent_id": record.agent_id,
                            "status": record.status.value,
                            "terminal_revision": None,
                        }
                    )
            cursor = page.next_cursor
            if cursor is None:
                break
        return waiting, unread_notified

    @staticmethod
    def _read_child_versions(
        conversation: Sequence[Mapping[str, Any]],
    ) -> set[tuple[str, int]]:
        terminal_statuses = {
            SubAgentStatus.COMPLETED.value,
            SubAgentStatus.FAILED.value,
            SubAgentStatus.CANCELLED.value,
            SubAgentStatus.INTERRUPTED.value,
        }
        read: set[tuple[str, int]] = set()
        for message in conversation:
            if (
                message.get("role") != "tool"
                or message.get("name") != "wait_agent"
                or message.get("status") != "success"
                or not isinstance(message.get("content"), str)
            ):
                continue
            try:
                value = strict_json_loads(cast(str, message["content"]))
            except ValueError:
                continue
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                agent_id = item.get("agent_id")
                revision = item.get("terminal_record_revision")
                if (
                    isinstance(agent_id, str)
                    and item.get("status") in terminal_statuses
                    and type(revision) is int
                    and revision > 0
                ):
                    read.add((agent_id, revision))
        return read

    def _failed(self, worker: WorkerRecord, error: ErrorInfo) -> WorkerRecord:
        current = self._store.get(worker.worker_id)
        if current is None or current.report is not None:
            return worker if current is None else current
        return self._store.publish_report(
            current.worker_id,
            state="failed",
            summary=error.message,
            error_code=error.code,
            error_message=error.message,
            usage=current.usage,
        )

    def _publish_cancellation(self, worker: WorkerRecord) -> None:
        current = self._store.get(worker.worker_id)
        if (
            current is None
            or current.report is not None
            or current.state not in {"working", "waiting"}
        ):
            return
        self._store.publish_report(
            current.worker_id,
            state="cancelled",
            summary="Worker execution was cancelled.",
            error_code="turn_cancelled",
            error_message="Worker execution was cancelled before producing a report.",
            usage=current.usage,
        )

    def _trace_scope(self, worker: WorkerRecord) -> Any:
        parent = current_trace_scope()
        source = TraceRootSource(
            workspace_key=workspace_key_for(self._workspace_state.workspace_path),
            workspace_path=str(self._workspace_state.workspace_path),
            origin=TraceOrigin.SCHEDULE if self._schedule_session else TraceOrigin.FOREGROUND,
            purpose=TracePurpose.AGENT_RUN,
            owner_session_id=worker.session_id,
            subject_session_id=None,
            run_id=worker.runtime_facts.get("foreground_execution_id"),
            restore_run_token=None if self._schedule_session else worker.restore_run_token,
            schedule_run_id=worker.parent_task_id if self._schedule_session else None,
            agent_id=worker.worker_id,
            parent_root_id=worker.trace_parent_root_id or (
                None if parent is None else parent.context.root_id
            ),
        )
        return trace_scope(self._trace_recorder, source)


__all__ = ["WorkerRunnerExecutor"]
