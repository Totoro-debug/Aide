"""Atomic Workspace persistence for independent Worker contexts and reports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from aide.agent.permission import PermissionSnapshot, validate_permission_level
from aide.agent.session.session import Session
from aide.agent.workers.models import (
    WorkerRecord,
    WorkerReport,
    WorkerReportDelivery,
    WorkerReportStatus,
    _record_from_dict,
)
from aide.agent.workspace_state import WorkspaceState
from aide.provider.session_configuration import SessionModelConfiguration
from aide.utils.host_filesystem import HOST_FILESYSTEM
from aide.utils.json import strict_json_loads
from aide.utils.validation import (
    empty_token_usage,
    require_uuid4_string,
    token_usage_validation_issue,
)

_SCHEMA_VERSION = 2


class WorkerStoreError(RuntimeError):
    """A Worker record could not be read or durably published safely."""


class WorkerStoreConflict(ValueError):
    """A Worker operation conflicts with its durable identity or state."""


@dataclass(frozen=True, slots=True)
class WorkerStoreSnapshot:
    revision: int
    workers: tuple[WorkerRecord, ...] = ()


class WorkerStore:
    """Own one Conversation Session's Worker records in a single atomic file."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        replace_text: Any = HOST_FILESYSTEM.atomic_replace_text,
    ) -> None:
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("Worker Store requires WorkspaceState")
        Session._require_id(session_id)
        self.workspace_state = workspace_state
        self.session_id = session_id
        self.path = workspace_state.workers_directory / session_id / "state.json"
        self._replace_text = replace_text
        self._lock = RLock()

    @classmethod
    def recover_workspace(cls, workspace_state: WorkspaceState) -> None:
        """Turn executions interrupted by process exit into durable blocked reports."""
        root = workspace_state.existing_workers_directory()
        if root is None:
            return
        try:
            directories = tuple(HOST_FILESYSTEM.path_for_io(root).iterdir())
        except OSError as error:
            raise WorkerStoreError("Workspace Worker records could not be recovered") from error
        for directory in directories:
            try:
                owned = HOST_FILESYSTEM.require_owned_directory(directory, within=root)
                Session._require_id(owned.name)
                store = cls(workspace_state, owned.name)
                for worker in store.load().workers:
                    worker = store.recover_completed_result(worker.worker_id)
                    if worker.state == "working":
                        store.publish_report(
                            worker.worker_id,
                            state="blocked",
                            summary="Worker execution was interrupted by Service restart.",
                            error_code="service_interrupted",
                            error_message="The Worker did not reach a persisted terminal result.",
                            usage=worker.usage,
                        )
                workers = store.load().workers
                pending_tasks = {
                    (worker.parent_task_id, worker.parent_generation)
                    for worker in workers
                    if worker.report is not None
                    and worker.report_delivery_state != "consumed"
                }
                if pending_tasks:
                    from aide.agent.foreground_work.store import ForegroundWorkStore

                    foreground = ForegroundWorkStore(workspace_state, owned.name)
                    for task_id, generation in pending_tasks:
                        foreground.interrupt_worker_wait(task_id, generation=generation)
            except (OSError, ValueError, WorkerStoreError) as error:
                raise WorkerStoreError(
                    "Workspace contains an invalid Worker Session entry"
                ) from error

    def load(self) -> WorkerStoreSnapshot:
        """Read this Session's worker set without creating Workspace state."""
        root = self._existing_session_root()
        if root is None or not HOST_FILESYSTEM.entry_exists(self.path):
            return WorkerStoreSnapshot(revision=0)
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
        try:
            raw = strict_json_loads(
                HOST_FILESYSTEM.path_for_io(self.path).read_text(encoding="utf-8")
            )
            return _snapshot_from_value(raw, self.session_id)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise WorkerStoreError("Worker state is not valid UTF-8 JSON") from error

    def get(self, worker_id: str) -> WorkerRecord | None:
        require_uuid4_string(worker_id, field="worker_id")
        return next(
            (worker for worker in self.load().workers if worker.worker_id == worker_id), None
        )

    def find_for_dispatch(
        self, parent_task_id: str, parent_generation: int, task_id: str
    ) -> WorkerRecord | None:
        return next(
            (
                worker
                for worker in self.load().workers
                if worker.parent_task_id == parent_task_id
                and worker.parent_generation == parent_generation
                and worker.task_id == task_id
            ),
            None,
        )

    def create(
        self,
        *,
        parent_task_id: str,
        parent_generation: int,
        task_id: str,
        task: str,
        explicit_context: Sequence[str],
        runtime_facts: Mapping[str, str],
        permission_snapshot: PermissionSnapshot,
        session_model_configuration: SessionModelConfiguration,
        system_prompt: str,
        initial_message: str,
        restore_run_token: str | None,
        trace_parent_root_id: str | None = None,
        manual_skill_signature: str | None = None,
    ) -> WorkerRecord:
        """Persist a Worker before starting any model or Tool side effect."""
        if not isinstance(permission_snapshot, PermissionSnapshot):
            raise TypeError("Worker creation requires a captured PermissionSnapshot")
        if not isinstance(session_model_configuration, SessionModelConfiguration):
            raise TypeError("Worker creation requires captured Session model configuration")
        if (
            isinstance(parent_generation, bool)
            or not isinstance(parent_generation, int)
            or parent_generation < 1
        ):
            raise ValueError("Worker parent_generation must be positive")
        validate_permission_level(permission_snapshot.level)
        with self._lock:
            current = self.load()
            existing = next(
                (
                    worker
                    for worker in current.workers
                    if worker.parent_task_id == parent_task_id
                    and worker.parent_generation == parent_generation
                    and worker.task_id == task_id
                ),
                None,
            )
            normalized_context = tuple(explicit_context)
            normalized_facts = dict(runtime_facts)
            if existing is not None:
                resume_prompt_compatible = (
                    existing.state == "waiting"
                    and manual_skill_signature is None
                )
                if (
                    existing.task != task
                    or existing.explicit_context != normalized_context
                    or existing.initial_message != initial_message
                    or (
                        not resume_prompt_compatible
                        and (
                            existing.runtime_facts != normalized_facts
                            or existing.permission_level != permission_snapshot.level
                            or existing.exec_shell_selector != permission_snapshot.exec_shell.selector
                            or existing.model_configuration != session_model_configuration
                            or existing.system_prompt != system_prompt
                            or existing.restore_run_token != restore_run_token
                        )
                    )
                    or (
                        manual_skill_signature is not None
                        and existing.context_state.get("manual_skill_signature")
                        != manual_skill_signature
                    )
                ):
                    raise WorkerStoreConflict("Worker dispatch identity was reused with different input")
                return existing

            context_state: dict[str, Any] = {
                "last_compacted": 0,
                "segment_protocol": 2,
                "child_notifications": [],
            }
            if manual_skill_signature is not None:
                context_state["manual_skill_signature"] = manual_skill_signature
            worker = WorkerRecord(
                worker_id=str(uuid4()),
                execution_id=str(uuid4()),
                session_id=self.session_id,
                parent_task_id=parent_task_id,
                parent_generation=parent_generation,
                task_id=task_id,
                task=task,
                explicit_context=normalized_context,
                runtime_facts=normalized_facts,
                permission_level=permission_snapshot.level,
                exec_shell_selector=permission_snapshot.exec_shell.selector,
                model_configuration=session_model_configuration,
                system_prompt=system_prompt,
                restore_run_token=restore_run_token,
                initial_message=initial_message,
                conversation=({"role": "user", "content": initial_message},),
                context_state=context_state,
                usage=empty_token_usage(),
                state="working",
                revision=1,
                trace_parent_root_id=trace_parent_root_id,
            )
            return self._publish(
                replace(current, revision=current.revision + 1, workers=(*current.workers, worker))
            ).workers[-1]

    def finish_completed_segment(
        self,
        worker_id: str,
        *,
        expected_revision: int,
        context_state: Mapping[str, Any],
    ) -> WorkerRecord:
        """Durably mark one completed execution segment before report publication."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            _require_writable_revision(worker, expected_revision)
            next_state = deepcopy(dict(context_state))
            next_state["segment_protocol"] = 2
            next_state["segment_result"] = {
                "kind": "completed",
                "execution_id": worker.execution_id,
            }
            next_state.pop("waiting_children", None)
            next_state.pop("recovery_error", None)
            updated = replace(
                worker,
                context_state=next_state,
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def suspend_for_children(
        self,
        worker_id: str,
        *,
        expected_revision: int,
        conversation: Sequence[dict[str, Any]],
        context_state: Mapping[str, Any],
        usage: Mapping[str, int],
        children: Sequence[Mapping[str, object]],
    ) -> WorkerRecord:
        """Atomically save a finished segment that still awaits owned children."""
        normalized_children = _validate_child_versions(children)
        if not normalized_children:
            raise ValueError("Worker child wait requires at least one child")
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            _require_writable_revision(worker, expected_revision)
            messages = tuple(deepcopy(list(conversation)))
            if messages[: len(worker.conversation)] != worker.conversation:
                raise WorkerStoreConflict("Worker checkpoint cannot rewrite its saved context")
            next_state = deepcopy(dict(context_state))
            next_state["segment_protocol"] = 2
            next_state["segment_result"] = {
                "kind": "waiting_for_children",
                "execution_id": worker.execution_id,
            }
            next_state["waiting_children"] = normalized_children
            next_state.pop("recovery_error", None)
            updated = replace(
                worker,
                conversation=messages,
                context_state=next_state,
                usage=_validate_usage(usage),
                state="waiting",
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def resume_for_children(
        self,
        worker_id: str,
        *,
        expected_revision: int,
        notifications: Sequence[Mapping[str, object]],
    ) -> WorkerRecord:
        """Accept each terminal child identity once and queue a fresh Worker segment."""
        normalized = _validate_child_notifications(notifications)
        if not normalized:
            raise ValueError("Worker child resume requires a notification")
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            notified = worker.context_state.get("child_notifications", [])
            if not isinstance(notified, list):
                raise WorkerStoreError("Worker child notifications are invalid")
            notified_ids = {
                (item.get("agent_id"), item.get("terminal_revision"))
                for item in notified
                if isinstance(item, dict)
            }
            fresh = tuple(
                item
                for item in normalized
                if (item["agent_id"], item["terminal_revision"]) not in notified_ids
            )
            if not fresh:
                return worker
            _require_writable_revision(worker, expected_revision, allowed_state="waiting")
            waiting_children = worker.context_state.get("waiting_children", [])
            waiting_ids = {
                item.get("agent_id")
                for item in waiting_children
                if isinstance(item, dict)
            } if isinstance(waiting_children, list) else set()
            if any(item["agent_id"] not in waiting_ids for item in fresh):
                raise WorkerStoreConflict("SubAgent does not belong to this Worker wait")
            next_state = deepcopy(worker.context_state)
            next_state["child_notifications"] = [*notified, *fresh]
            next_state["segment_result"] = {
                "kind": "resumed_for_children",
                "execution_id": str(uuid4()),
            }
            next_state.pop("waiting_children", None)
            next_state.pop("recovery_error", None)
            notification_text = "SubAgent status update; results are not included. Call wait_agent to read each result:\n"
            notification_text += "\n".join(
                f"- agent_id={item['agent_id']} status={item['status']} "
                f"terminal_revision={item['terminal_revision']}"
                for item in fresh
            )
            updated = replace(
                worker,
                execution_id=cast(str, next_state["segment_result"]["execution_id"]),
                conversation=(
                    *worker.conversation,
                    {"role": "user", "content": notification_text},
                ),
                context_state=next_state,
                state="working",
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def set_recovery_error(
        self,
        worker_id: str,
        *,
        code: str,
        message: str,
        attempts: int,
    ) -> WorkerRecord:
        """Persist a visible report-recovery failure without inventing a report."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            if worker.report is not None:
                return worker
            next_state = deepcopy(worker.context_state)
            next_state["recovery_error"] = {
                "code": code,
                "message": message,
                "attempts": attempts,
            }
            updated = replace(
                worker,
                context_state=next_state,
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def clear_recovery_error(self, worker_id: str) -> WorkerRecord:
        """Clear a persisted transient failure after explicit recovery succeeds."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            if "recovery_error" not in worker.context_state:
                return worker
            next_state = deepcopy(worker.context_state)
            next_state.pop("recovery_error", None)
            updated = replace(
                worker,
                context_state=next_state,
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def checkpoint(
        self,
        worker_id: str,
        *,
        conversation: Sequence[dict[str, Any]],
        context_state: Mapping[str, Any],
        usage: Mapping[str, int],
    ) -> WorkerRecord:
        """Persist an isolated AgentRunner checkpoint before later model requests."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            if worker.state not in {"working", "waiting"}:
                raise WorkerStoreConflict("Worker context is no longer writable")
            messages = tuple(deepcopy(list(conversation)))
            if messages[: len(worker.conversation)] != worker.conversation:
                raise WorkerStoreConflict("Worker checkpoint cannot rewrite its saved context")
            normalized_usage = _validate_usage(usage)
            updated = replace(
                worker,
                conversation=messages,
                context_state=deepcopy(dict(context_state)),
                usage=normalized_usage,
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def recover_completed_result(self, worker_id: str) -> WorkerRecord:
        """Publish a terminal checkpoint without replaying model or Tool work."""
        with self._lock:
            worker = self._require_worker(self.load(), worker_id)
            if worker.context_state.get("segment_protocol") == 2:
                segment_result = worker.context_state.get("segment_result")
                if (
                    not isinstance(segment_result, dict)
                    or segment_result.get("kind") != "completed"
                    or segment_result.get("execution_id") != worker.execution_id
                ):
                    return worker
            last = worker.conversation[-1]
            if (
                worker.state == "working"
                and worker.report is None
                and last.get("role") == "assistant"
                and last.get("status") == "completed"
                and not last.get("tool_calls")
                and last.get("error") is None
            ):
                return self.publish_report(
                    worker.worker_id,
                    state="ready",
                    summary=last.get("content") or "Worker completed without a text response.",
                    usage=worker.usage,
                )
            error = last.get("error")
            if (
                worker.state == "working"
                and worker.report is None
                and last.get("role") == "assistant"
                and not last.get("tool_calls")
                and last.get("status") in {"error", "interrupted"}
                and isinstance(error, dict)
                and isinstance(error.get("code"), str)
                and isinstance(error.get("message"), str)
            ):
                state: WorkerReportStatus = (
                    "cancelled" if last.get("status") == "interrupted"
                    else "blocked" if error["code"] == "agent_iteration_limit"
                    else "failed"
                )
                return self.publish_report(
                    worker.worker_id, state=state,
                    summary=error["message"], error_code=error["code"],
                    error_message=error["message"], usage=worker.usage,
                )
            return worker

    def publish_report(
        self,
        worker_id: str,
        *,
        state: WorkerReportStatus,
        summary: str,
        error_code: str | None = None,
        error_message: str | None = None,
        usage: Mapping[str, int],
    ) -> WorkerRecord:
        """Durably publish one terminal report and pending outbox item atomically."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            normalized_usage = _validate_usage(usage)
            if worker.report is not None:
                if (
                    worker.report.status == state
                    and worker.report.summary == summary
                    and worker.report.error_code == error_code
                    and worker.report.error_message == error_message
                ):
                    return worker
                raise WorkerStoreConflict("Worker already has a different terminal report")
            if worker.state not in {"working", "waiting"}:
                raise WorkerStoreConflict("Worker cannot publish a report from its current state")
            report = WorkerReport(
                worker_id=worker.worker_id,
                report_id=f"{worker.worker_id}:1",
                sequence=1,
                status=state,
                summary=summary,
                error_code=error_code,
                error_message=error_message,
                usage=normalized_usage,
            )
            context_state = deepcopy(worker.context_state)
            context_state.pop("recovery_error", None)
            updated = replace(
                worker,
                state=state,
                usage=normalized_usage,
                report=report,
                report_delivery_state="pending",
                report_receipt_id=None,
                context_state=context_state,
                revision=worker.revision + 1,
            )
            return self._replace_worker(current, worker, updated)

    def claim_report(self, worker_id: str, report_id: str) -> WorkerReportDelivery | None:
        """Return the same pending report and receipt on retry until acknowledged."""
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            report = worker.report
            if report is None or report.report_id != report_id:
                return None
            if worker.report_delivery_state == "consumed":
                return None
            receipt_id = report.receipt_id
            if worker.report_delivery_state == "pending":
                updated = replace(
                    worker,
                    report_delivery_state="claimed",
                    report_receipt_id=receipt_id,
                    revision=worker.revision + 1,
                )
                self._replace_worker(current, worker, updated)
            elif worker.report_receipt_id != receipt_id:
                raise WorkerStoreError("Worker report outbox receipt is inconsistent")
            return WorkerReportDelivery(report=report, receipt_id=receipt_id)

    def claim_report_batch(
        self, parent_task_id: str, parent_generation: int
    ) -> tuple[WorkerReportDelivery, ...] | None:
        """Claim every unconsumed report after all Workers for a Task generation terminate."""
        if (
            not isinstance(parent_task_id, str)
            or not parent_task_id
            or isinstance(parent_generation, bool)
            or not isinstance(parent_generation, int)
            or parent_generation < 1
        ):
            raise ValueError("Worker report batch identity is invalid")
        with self._lock:
            current = self.load()
            workers = tuple(
                worker
                for worker in current.workers
                if worker.parent_task_id == parent_task_id
                and worker.parent_generation == parent_generation
            )
            if not workers or any(worker.report is None for worker in workers):
                return None

            pending = tuple(
                worker
                for worker in workers
                if worker.report_delivery_state != "consumed"
            )
            if not pending:
                return ()

            updated_by_id: dict[str, WorkerRecord] = {}
            deliveries: list[WorkerReportDelivery] = []
            for worker in pending:
                report = worker.report
                assert report is not None
                receipt_id = report.receipt_id
                if worker.report_delivery_state == "pending":
                    updated = replace(
                        worker,
                        report_delivery_state="claimed",
                        report_receipt_id=receipt_id,
                        revision=worker.revision + 1,
                    )
                    updated_by_id[worker.worker_id] = updated
                elif worker.report_delivery_state == "claimed":
                    if worker.report_receipt_id != receipt_id:
                        raise WorkerStoreError("Worker report outbox receipt is inconsistent")
                else:
                    raise WorkerStoreError("Worker report delivery state is invalid")
                deliveries.append(WorkerReportDelivery(report=report, receipt_id=receipt_id))

            if updated_by_id:
                snapshot = replace(
                    current,
                    revision=current.revision + 1,
                    workers=tuple(updated_by_id.get(item.worker_id, item) for item in current.workers),
                )
                self._publish(snapshot)
            return tuple(deliveries)

    def acknowledge_report(self, worker_id: str, receipt_id: str) -> bool:
        """Commit one stable Router receipt; duplicate acknowledgements are inert."""
        require_uuid4_string(worker_id, field="worker_id")
        with self._lock:
            current = self.load()
            worker = self._require_worker(current, worker_id)
            if worker.report is None:
                return False
            stable_receipt = worker.report.receipt_id
            if receipt_id != stable_receipt:
                raise WorkerStoreConflict("Worker report receipt does not match")
            if worker.report_delivery_state == "consumed":
                return False
            if worker.report_delivery_state != "claimed":
                raise WorkerStoreConflict("Worker report has not been delivered")
            updated = replace(
                worker,
                report_delivery_state="consumed",
                report_receipt_id=stable_receipt,
                revision=worker.revision + 1,
            )
            self._replace_worker(current, worker, updated)
            return True

    def reconcile_deliveries(self, messages: Sequence[Mapping[str, object]]) -> int:
        """Acknowledge claimed reports already present in durable Router history."""
        persisted: set[tuple[str, str, str]] = set()
        for message in messages:
            receipt_metadata = message.get("worker_report_receipts")
            is_report_commit = (
                message.get("role") == "assistant"
                and (
                    message.get("status") == "completed"
                    or (
                        message.get("status") == "error"
                        and isinstance(message.get("schedule_occurrence_id"), str)
                    )
                )
            )
            is_dispatch_result = (
                message.get("role") == "tool"
                and message.get("name") == "dispatch_worker"
                and message.get("status") == "success"
            )
            if isinstance(receipt_metadata, list) and (is_report_commit or is_dispatch_result):
                for item in receipt_metadata:
                    if not isinstance(item, dict):
                        continue
                    worker_id, report_id, receipt_id = (
                        item.get("worker_id"),
                        item.get("report_id"),
                        item.get("receipt_id"),
                    )
                    if all(isinstance(value, str) for value in (worker_id, report_id, receipt_id)):
                        persisted.add(
                            (cast(str, worker_id), cast(str, report_id), cast(str, receipt_id))
                        )
            if not is_dispatch_result:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                value = strict_json_loads(content)
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            worker_id, report_id, receipt_id = (
                value.get("worker_id"),
                value.get("report_id"),
                value.get("receipt_id"),
            )
            if all(isinstance(item, str) for item in (worker_id, report_id, receipt_id)):
                persisted.add((cast(str, worker_id), cast(str, report_id), cast(str, receipt_id)))
        consumed = 0
        for worker in self.load().workers:
            if (
                worker.report is None
                or worker.report_delivery_state != "claimed"
                or worker.report_receipt_id is None
            ):
                continue
            if (
                worker.worker_id,
                worker.report.report_id,
                worker.report_receipt_id,
            ) in persisted and self.acknowledge_report(worker.worker_id, worker.report_receipt_id):
                consumed += 1
        return consumed

    def discard_restore_run_tokens(self, tokens: Iterable[UUID | str]) -> int:
        """Drop Worker contexts and receipts belonging to a discarded Restore branch."""
        normalized = {str(token) for token in tokens}
        if not normalized:
            return 0
        with self._lock:
            current = self.load()
            retained = tuple(
                worker for worker in current.workers if worker.restore_run_token not in normalized
            )
            removed = len(current.workers) - len(retained)
            if removed:
                self._publish(replace(current, revision=current.revision + 1, workers=retained))
            return removed

    def _require_worker(self, current: WorkerStoreSnapshot, worker_id: str) -> WorkerRecord:
        require_uuid4_string(worker_id, field="worker_id")
        worker = next((item for item in current.workers if item.worker_id == worker_id), None)
        if worker is None:
            raise WorkerStoreConflict("Worker does not exist in this Session")
        return worker

    def _replace_worker(
        self,
        current: WorkerStoreSnapshot,
        previous: WorkerRecord,
        candidate: WorkerRecord,
    ) -> WorkerRecord:
        workers = tuple(
            candidate if item.worker_id == previous.worker_id else item for item in current.workers
        )
        return next(
            item
            for item in self._publish(
                replace(current, revision=current.revision + 1, workers=workers)
            ).workers
            if item.worker_id == candidate.worker_id
        )

    def _publish(self, snapshot: WorkerStoreSnapshot) -> WorkerStoreSnapshot:
        root = self._prepare_session_root()
        value = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": snapshot.revision,
            "workers": [worker.to_dict() for worker in snapshot.workers],
        }
        content = json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
        if HOST_FILESYSTEM.entry_exists(self.path):
            HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
            self._replace_text(self.path, content)
        else:
            HOST_FILESYSTEM.atomic_create_text(self.path, content)
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
        HOST_FILESYSTEM.restrict_private_file(self.path)
        return snapshot

    def _existing_session_root(self) -> Path | None:
        try:
            root = self.workspace_state.existing_workers_directory()
        except (OSError, PermissionError) as error:
            raise WorkerStoreError("Workspace Worker directory is unsafe") from error
        if root is None:
            return None
        session_root = root / self.session_id
        if not HOST_FILESYSTEM.entry_exists(session_root):
            return None
        return HOST_FILESYSTEM.require_owned_directory(session_root, within=root)

    def _prepare_session_root(self) -> Path:
        try:
            root = self.workspace_state.prepare_workers_directory()
            session_root = HOST_FILESYSTEM.path_for_io(root / self.session_id)
            if not HOST_FILESYSTEM.entry_exists(session_root):
                session_root.mkdir(mode=0o700)
            owned = HOST_FILESYSTEM.require_owned_directory(session_root, within=root)
            HOST_FILESYSTEM.restrict_private_directory(owned)
            return owned
        except (OSError, PermissionError) as error:
            raise WorkerStoreError(
                "Worker Session directory could not be prepared safely"
            ) from error


def _snapshot_from_value(value: object, session_id: str) -> WorkerStoreSnapshot:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "session_id", "revision", "workers"}
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in {1, _SCHEMA_VERSION}
        or value["session_id"] != session_id
        or type(value["revision"]) is not int
        or not isinstance(value["workers"], list)
    ):
        raise WorkerStoreError("Worker state shape is invalid")
    workers = tuple(_record_from_dict(item) for item in cast(list[object], value["workers"]))
    if any(worker.session_id != session_id for worker in workers):
        raise WorkerStoreError("Worker state contains a record from another Session")
    if len({worker.worker_id for worker in workers}) != len(workers):
        raise WorkerStoreError("Worker state contains duplicate identities")
    dispatch_ids = {
        (worker.parent_task_id, worker.parent_generation, worker.task_id)
        for worker in workers
    }
    if len(dispatch_ids) != len(workers):
        raise WorkerStoreError("Worker state contains duplicate dispatch identities")
    return WorkerStoreSnapshot(revision=value["revision"], workers=workers)


def _validate_usage(value: Mapping[str, int]) -> dict[str, int]:
    usage = dict(value)
    if token_usage_validation_issue(usage) is not None:
        raise ValueError("Worker usage must contain valid Agent Runner usage fields")
    return usage


def _validate_child_versions(
    children: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
        raise ValueError("Worker child list is invalid")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in children:
        if not isinstance(item, Mapping):
            raise ValueError("Worker child entry is invalid")
        agent_id = item.get("agent_id")
        status = item.get("status")
        revision = item.get("terminal_revision")
        if not isinstance(agent_id, str):
            raise ValueError("Worker child identity is invalid")
        require_uuid4_string(agent_id, field="agent_id")
        if agent_id in seen:
            raise ValueError("Worker child identity is duplicated")
        seen.add(agent_id)
        if not isinstance(status, str) or not status:
            raise ValueError("Worker child status is invalid")
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        ):
            raise ValueError("Worker child terminal revision is invalid")
        normalized.append(
            {"agent_id": agent_id, "status": status, "terminal_revision": revision}
        )
    return normalized


def _validate_child_notifications(
    notifications: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized = _validate_child_versions(notifications)
    if any(not isinstance(item["terminal_revision"], int) for item in normalized):
        raise ValueError("Worker child notification must identify a terminal revision")
    return normalized


def _require_writable_revision(
    worker: WorkerRecord,
    expected_revision: int,
    *,
    allowed_state: Literal["working", "waiting"] = "working",
) -> None:
    if worker.revision != expected_revision:
        raise WorkerStoreConflict("Worker revision is stale")
    if worker.state != allowed_state or worker.report is not None:
        raise WorkerStoreConflict("Worker context is no longer writable")


__all__ = ["WorkerStore", "WorkerStoreConflict", "WorkerStoreError", "WorkerStoreSnapshot"]
