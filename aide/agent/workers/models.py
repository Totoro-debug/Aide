"""Durable Worker execution and report contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from aide.agent.permission import ToolPermissionLevel, validate_permission_level
from aide.provider.session_configuration import SessionModelConfiguration
from aide.utils.json import json_validation_issue
from aide.utils.validation import (
    require_uuid4_string,
    token_usage_validation_issue,
)

WorkerStatus = Literal["working", "waiting", "ready", "blocked", "failed", "cancelled"]
WorkerReportStatus = Literal["ready", "blocked", "failed", "cancelled"]
WorkerDeliveryState = Literal["pending", "claimed", "consumed"]
_WORKER_STATES = frozenset({"working", "waiting", "ready", "blocked", "failed", "cancelled"})
_REPORT_STATES = frozenset({"ready", "blocked", "failed", "cancelled"})
_DELIVERY_STATES = frozenset({"pending", "claimed", "consumed"})
_USAGE_FIELDS = frozenset({"model_calls", "input_tokens", "output_tokens", "total_tokens"})


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """One persisted Worker outcome eligible for delivery to its Router."""

    worker_id: str
    report_id: str
    sequence: int
    status: WorkerReportStatus
    summary: str
    error_code: str | None
    error_message: str | None
    usage: dict[str, int]

    def __post_init__(self) -> None:
        require_uuid4_string(self.worker_id, field="worker_id")
        if self.report_id != f"{self.worker_id}:1":
            raise ValueError("Worker report ID is invalid")
        if type(self.sequence) is not int or self.sequence != 1:
            raise ValueError("Worker report sequence must be one")
        if self.status not in _REPORT_STATES:
            raise ValueError("Worker report status is invalid")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("Worker report summary must not be empty")
        for name, value in (("error_code", self.error_code), ("error_message", self.error_message)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Worker report {name} must be non-empty or None")
        issue = token_usage_validation_issue(self.usage)
        if issue is not None:
            raise ValueError("Worker report usage is invalid")
        object.__setattr__(self, "usage", deepcopy(self.usage))

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "report_id": self.report_id,
            "sequence": self.sequence,
            "status": self.status,
            "summary": self.summary,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "usage": deepcopy(self.usage),
        }


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    """Independent Worker context, authorization snapshot, and report outbox."""

    worker_id: str
    execution_id: str
    session_id: str
    parent_task_id: str
    parent_generation: int
    task_id: str
    task: str
    explicit_context: tuple[str, ...]
    runtime_facts: dict[str, str]
    permission_level: ToolPermissionLevel
    exec_shell_selector: Literal["auto", "powershell", "pwsh"]
    model_configuration: SessionModelConfiguration
    system_prompt: str
    restore_run_token: str | None
    initial_message: str
    conversation: tuple[dict[str, Any], ...]
    context_state: dict[str, Any]
    usage: dict[str, int]
    state: WorkerStatus
    revision: int
    report: WorkerReport | None = None
    report_delivery_state: WorkerDeliveryState | None = None
    report_receipt_id: str | None = None

    def __post_init__(self) -> None:
        require_uuid4_string(self.worker_id, field="worker_id")
        require_uuid4_string(self.execution_id, field="execution_id")
        for field_name, field_value, limit in (
            ("session_id", self.session_id, 128),
            ("parent_task_id", self.parent_task_id, 128),
            ("task_id", self.task_id, 128),
        ):
            if (
                not isinstance(field_value, str)
                or not field_value.strip()
                or len(field_value) > limit
            ):
                raise ValueError(f"Worker {field_name} must be a bounded non-empty string")
        if type(self.parent_generation) is not int or self.parent_generation < 1:
            raise ValueError("Worker parent_generation must be positive")
        if not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 8000:
            raise ValueError("Worker task must contain 1 to 8000 characters")
        if (
            not isinstance(self.explicit_context, tuple)
            or len(self.explicit_context) > 10
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 1000
                for item in self.explicit_context
            )
        ):
            raise ValueError("Worker explicit context is invalid")
        if not isinstance(self.runtime_facts, dict) or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or len(value) > 2000
            for key, value in self.runtime_facts.items()
        ):
            raise ValueError("Worker runtime facts are invalid")
        validate_permission_level(self.permission_level)
        if self.exec_shell_selector not in {"auto", "powershell", "pwsh"}:
            raise ValueError("Worker Exec shell selector is invalid")
        if not isinstance(self.model_configuration, SessionModelConfiguration):
            raise TypeError("Worker model configuration is invalid")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("Worker system prompt must not be empty")
        if self.restore_run_token is not None:
            require_uuid4_string(self.restore_run_token, field="restore_run_token")
        if not isinstance(self.initial_message, str) or not self.initial_message.strip():
            raise ValueError("Worker initial message must not be empty")
        if (
            not isinstance(self.conversation, tuple)
            or any(not isinstance(item, dict) for item in self.conversation)
            or not self.conversation
            or self.conversation[0].get("role") != "user"
            or self.conversation[0].get("content") != self.initial_message
        ):
            raise ValueError("Worker conversation must begin with its isolated task input")
        for field_name, json_value in (
            ("conversation", list(self.conversation)),
            ("context_state", self.context_state),
        ):
            json_issue = json_validation_issue(json_value, field=f"worker.{field_name}")
            if json_issue is not None:
                raise ValueError(f"Worker {field_name} is invalid: {json_issue}")
        if not isinstance(self.context_state, dict):
            raise ValueError("Worker context_state must be an object")
        usage_issue = token_usage_validation_issue(self.usage)
        if usage_issue is not None:
            raise ValueError("Worker usage is invalid")
        if self.state not in _WORKER_STATES:
            raise ValueError("Worker state is invalid")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Worker revision must be positive")
        if self.report is not None:
            if not isinstance(self.report, WorkerReport) or self.report.worker_id != self.worker_id:
                raise ValueError("Worker report belongs to a different Worker")
            if self.report.status != self.state:
                raise ValueError("Worker report status must match its Worker")
            if self.report_delivery_state not in _DELIVERY_STATES:
                raise ValueError("Worker report delivery state is required")
            stable_receipt = _receipt_id(self.worker_id, self.report.sequence)
            if self.report_delivery_state in {"claimed", "consumed"}:
                if self.report_receipt_id != stable_receipt:
                    raise ValueError("Worker report receipt is invalid")
            elif self.report_receipt_id is not None:
                raise ValueError("Pending Worker report cannot have a receipt")
        elif self.state in _REPORT_STATES or any(
            value is not None for value in (self.report_delivery_state, self.report_receipt_id)
        ):
            raise ValueError("Terminal Worker state requires a persisted report")
        object.__setattr__(self, "explicit_context", tuple(self.explicit_context))
        object.__setattr__(self, "runtime_facts", deepcopy(self.runtime_facts))
        object.__setattr__(self, "conversation", tuple(deepcopy(list(self.conversation))))
        object.__setattr__(self, "context_state", deepcopy(self.context_state))
        object.__setattr__(self, "usage", deepcopy(self.usage))

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "parent_task_id": self.parent_task_id,
            "parent_generation": self.parent_generation,
            "task_id": self.task_id,
            "task": self.task,
            "explicit_context": list(self.explicit_context),
            "runtime_facts": deepcopy(self.runtime_facts),
            "permission_level": self.permission_level,
            "exec_shell_selector": self.exec_shell_selector,
            "model_configuration": {
                "provider_id": self.model_configuration.provider_id,
                "model": self.model_configuration.model,
                "reasoning_effort": self.model_configuration.reasoning_effort,
            },
            "system_prompt": self.system_prompt,
            "restore_run_token": self.restore_run_token,
            "initial_message": self.initial_message,
            "conversation": deepcopy(list(self.conversation)),
            "context_state": deepcopy(self.context_state),
            "usage": deepcopy(self.usage),
            "state": self.state,
            "revision": self.revision,
            "report": None if self.report is None else self.report.to_dict(),
            "report_delivery_state": self.report_delivery_state,
            "report_receipt_id": self.report_receipt_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerReportDelivery:
    report: WorkerReport
    receipt_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.report, WorkerReport):
            raise TypeError("Worker report delivery requires a WorkerReport")
        if self.receipt_id != _receipt_id(self.report.worker_id, self.report.sequence):
            raise ValueError("Worker report delivery receipt is invalid")


def _receipt_id(worker_id: str, sequence: int) -> str:
    return f"worker-report:{worker_id}:{sequence}"


def _report_from_value(value: object) -> WorkerReport | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "worker_id",
        "report_id",
        "sequence",
        "status",
        "summary",
        "error_code",
        "error_message",
        "usage",
    }:
        raise ValueError("Worker report fields are invalid")
    data = cast(dict[str, object], value)
    usage = data["usage"]
    if not isinstance(usage, dict):
        raise ValueError("Worker report usage is invalid")
    return WorkerReport(
        worker_id=_string(data["worker_id"], "report.worker_id"),
        report_id=_string(data["report_id"], "report.report_id"),
        sequence=_integer(data["sequence"], "report.sequence"),
        status=cast(WorkerReportStatus, _string(data["status"], "report.status")),
        summary=_string(data["summary"], "report.summary"),
        error_code=_optional_string(data["error_code"], "report.error_code"),
        error_message=_optional_string(data["error_message"], "report.error_message"),
        usage={key: _integer(item, f"report.usage.{key}") for key, item in usage.items()},
    )


def _record_from_dict(value: object) -> WorkerRecord:
    fields = {
        "worker_id",
        "execution_id",
        "session_id",
        "parent_task_id",
        "parent_generation",
        "task_id",
        "task",
        "explicit_context",
        "runtime_facts",
        "permission_level",
        "exec_shell_selector",
        "model_configuration",
        "system_prompt",
        "restore_run_token",
        "initial_message",
        "conversation",
        "context_state",
        "usage",
        "state",
        "revision",
        "report",
        "report_delivery_state",
        "report_receipt_id",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Worker record fields do not match the current format")
    data = cast(dict[str, object], value)
    context = data["explicit_context"]
    facts = data["runtime_facts"]
    config = data["model_configuration"]
    conversation = data["conversation"]
    context_state = data["context_state"]
    usage = data["usage"]
    if (
        not isinstance(context, list)
        or any(not isinstance(item, str) for item in context)
        or not isinstance(facts, dict)
        or any(not isinstance(key, str) or not isinstance(item, str) for key, item in facts.items())
        or not isinstance(config, dict)
        or set(config) != {"provider_id", "model", "reasoning_effort"}
        or not isinstance(conversation, list)
        or not isinstance(context_state, dict)
        or not isinstance(usage, dict)
    ):
        raise ValueError("Worker record values are invalid")
    return WorkerRecord(
        worker_id=_string(data["worker_id"], "worker_id"),
        execution_id=_string(data["execution_id"], "execution_id"),
        session_id=_string(data["session_id"], "session_id"),
        parent_task_id=_string(data["parent_task_id"], "parent_task_id"),
        parent_generation=_integer(data["parent_generation"], "parent_generation"),
        task_id=_string(data["task_id"], "task_id"),
        task=_string(data["task"], "task"),
        explicit_context=tuple(context),
        runtime_facts=deepcopy(cast(dict[str, str], facts)),
        permission_level=cast(
            ToolPermissionLevel, _string(data["permission_level"], "permission_level")
        ),
        exec_shell_selector=cast(
            Literal["auto", "powershell", "pwsh"],
            _string(data["exec_shell_selector"], "exec_shell_selector"),
        ),
        model_configuration=SessionModelConfiguration(
            provider_id=_string(config["provider_id"], "model_configuration.provider_id"),
            model=_string(config["model"], "model_configuration.model"),
            reasoning_effort=cast(
                Any, _string(config["reasoning_effort"], "model_configuration.reasoning_effort")
            ),
        ),
        system_prompt=_string(data["system_prompt"], "system_prompt"),
        restore_run_token=_optional_string(data["restore_run_token"], "restore_run_token"),
        initial_message=_string(data["initial_message"], "initial_message"),
        conversation=tuple(cast(list[dict[str, Any]], conversation)),
        context_state=deepcopy(cast(dict[str, Any], context_state)),
        usage={key: _integer(item, f"usage.{key}") for key, item in usage.items()},
        state=cast(WorkerStatus, _string(data["state"], "state")),
        revision=_integer(data["revision"], "revision"),
        report=_report_from_value(data["report"]),
        report_delivery_state=(
            None
            if data["report_delivery_state"] is None
            else cast(
                WorkerDeliveryState, _string(data["report_delivery_state"], "report_delivery_state")
            )
        ),
        report_receipt_id=_optional_string(data["report_receipt_id"], "report_receipt_id"),
    )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Worker {field} must be non-empty")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"Worker {field} must be non-empty or None")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Worker {field} must be an integer")
    return value


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _USAGE_FIELDS:
        raise ValueError("Worker usage fields are invalid")
    result = {key: _integer(item, f"usage.{key}") for key, item in value.items()}
    if token_usage_validation_issue(result) is not None:
        raise ValueError("Worker usage is invalid")
    return result


__all__ = [
    "WorkerDeliveryState",
    "WorkerRecord",
    "WorkerReport",
    "WorkerReportDelivery",
    "WorkerReportStatus",
    "WorkerStatus",
]
