"""Versioned SubAgent records and query DTOs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, cast

from aide.agent.session.session import validate_session_id
from aide.provider.session_configuration import ReasoningEffort, SessionModelConfiguration
from aide.utils.json import json_validation_issue
from aide.utils.validation import (
    empty_token_usage,
    require_aware_datetime,
    require_uuid4_string,
    token_usage_validation_issue,
)

SUBAGENT_RECORD_SCHEMA_VERSION = 4
_USAGE_FIELDS = frozenset({"model_calls", "input_tokens", "output_tokens", "total_tokens"})


class SubAgentStatus(StrEnum):
    """Lifecycle state persisted for one single-use SubAgent."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_ANSWER = "waiting_for_answer"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class SubAgentSourceKind(StrEnum):
    """Main Agent lane that created a SubAgent."""

    FOREGROUND = "foreground"
    SCHEDULE = "schedule"


class SubAgentEventKind(StrEnum):
    """Independent event categories emitted for SubAgent state changes."""

    STATUS = "subagent.status"
    OUTPUT = "subagent.output"
    ACTIVITY = "subagent.activity"
    USAGE = "subagent.usage"


@dataclass(frozen=True, slots=True)
class SubAgentSource:
    """Restore or Schedule ownership that determines the task's parent operation."""

    kind: SubAgentSourceKind
    restore_run_token: str | None = None
    job_id: str | None = None
    occurrence_id: str | None = None
    task_id: str | None = None
    task_generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SubAgentSourceKind):
            raise ValueError("SubAgent source kind is invalid")
        if self.kind is SubAgentSourceKind.FOREGROUND:
            if (
                self.restore_run_token is None
                or self.job_id is not None
                or self.occurrence_id is not None
                or (self.task_id is None) != (self.task_generation is None)
            ):
                raise ValueError("foreground SubAgent source fields are invalid")
            require_uuid4_string(self.restore_run_token, field="restore_run_token")
            if self.task_id is not None:
                require_uuid4_string(self.task_id, field="task_id")
                if (
                    isinstance(self.task_generation, bool)
                    or not isinstance(self.task_generation, int)
                    or self.task_generation < 1
                ):
                    raise ValueError("task_generation must be a positive integer")
        else:
            if (
                self.restore_run_token is not None
                or self.job_id is None
                or self.occurrence_id is None
                or self.task_id is not None
                or self.task_generation is not None
            ):
                raise ValueError("Schedule SubAgent source requires Job and occurrence IDs")
            require_uuid4_string(self.job_id, field="job_id")
            _require_nonempty_string(self.occurrence_id, field="occurrence_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "restore_run_token": self.restore_run_token,
            "job_id": self.job_id,
            "occurrence_id": self.occurrence_id,
            "task_id": self.task_id,
            "task_generation": self.task_generation,
        }


@dataclass(frozen=True, slots=True)
class SubAgentCreatorSnapshot:
    """Non-resource creator settings captured when a SubAgent is registered."""

    provider_id: str
    model: str
    reasoning_effort: ReasoningEffort
    permission_level: str
    shell: str | None
    tool_names: tuple[str, ...]
    system_prompt: str

    def __post_init__(self) -> None:
        SessionModelConfiguration(self.provider_id, self.model, self.reasoning_effort)
        _require_nonempty_string(
            self.permission_level,
            field="creator_snapshot.permission_level",
        )
        if self.shell is not None:
            _require_nonempty_string(self.shell, field="creator_snapshot.shell")
        if not isinstance(self.system_prompt, str):
            raise ValueError("creator_snapshot.system_prompt must be a string")
        if not isinstance(self.tool_names, tuple) or any(
            not isinstance(name, str) or not name.strip() for name in self.tool_names
        ):
            raise ValueError("creator_snapshot.tool_names must contain nonempty strings")
        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("creator_snapshot.tool_names must not contain duplicate names")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "permission_level": self.permission_level,
            "shell": self.shell,
            "tool_names": list(self.tool_names),
            "system_prompt": self.system_prompt,
        }


@dataclass(frozen=True, slots=True)
class SubAgentError:
    """Stable failure information retained with a failed or interrupted task."""

    code: str
    message: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.code, field="error.code")
        _require_nonempty_string(self.message, field="error.message")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class SubAgentRecord:
    """Complete durable state for one SubAgent, separate from Main Agent history."""

    agent_id: str
    session_id: str
    title: str
    task: str
    parent_run_id: str
    source: SubAgentSource
    creator_snapshot: SubAgentCreatorSnapshot
    created_at: datetime
    registered_order: int
    revision: int
    status: SubAgentStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    conversation: tuple[dict[str, Any], ...] = ()
    context_state: dict[str, Any] | None = None
    artifact_paths: tuple[str, ...] = ()
    result: str | None = None
    error: SubAgentError | None = None
    usage: dict[str, int] | None = None
    execution_id: str | None = None
    restore_run_token: str | None = None
    restore_prefixes: tuple[dict[str, Any], ...] = ()
    schema_version: int = SUBAGENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_uuid4_string(self.agent_id, field="agent_id")
        validate_session_id(self.session_id)
        _require_nonempty_string(self.title, field="title")
        _require_nonempty_string(self.task, field="task")
        _require_nonempty_string(self.parent_run_id, field="parent_run_id")
        require_aware_datetime(self.created_at, field="created_at")
        if self.started_at is not None:
            require_aware_datetime(self.started_at, field="started_at")
        if self.finished_at is not None:
            require_aware_datetime(self.finished_at, field="finished_at")
        if (
            isinstance(self.registered_order, bool)
            or not isinstance(self.registered_order, int)
            or self.registered_order < 1
        ):
            raise ValueError("registered_order must be a positive integer")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a nonnegative integer")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SUBAGENT_RECORD_SCHEMA_VERSION
        ):
            raise ValueError("SubAgent record schema version is unsupported")
        if not isinstance(self.status, SubAgentStatus):
            raise ValueError("SubAgent status is invalid")
        if not isinstance(self.source, SubAgentSource):
            raise ValueError("SubAgent source is invalid")
        if not isinstance(self.creator_snapshot, SubAgentCreatorSnapshot):
            raise ValueError("SubAgent creator snapshot is invalid")
        execution_id = self.agent_id if self.execution_id is None else self.execution_id
        require_uuid4_string(execution_id, field="execution_id")
        object.__setattr__(self, "execution_id", execution_id)
        restore_run_token = self.restore_run_token
        if restore_run_token is None and self.source.kind is SubAgentSourceKind.FOREGROUND:
            restore_run_token = self.source.restore_run_token
        if restore_run_token is not None:
            require_uuid4_string(restore_run_token, field="restore_run_token")
        object.__setattr__(self, "restore_run_token", restore_run_token)
        if not isinstance(self.restore_prefixes, tuple) or any(
            not isinstance(item, dict) for item in self.restore_prefixes
        ):
            raise ValueError("restore_prefixes must contain objects")
        object.__setattr__(
            self,
            "restore_prefixes",
            tuple(_restore_prefix_from_value(item) for item in self.restore_prefixes),
        )
        if any(not isinstance(message, dict) for message in self.conversation):
            raise ValueError("SubAgent conversation entries must be objects")
        context_state = {} if self.context_state is None else self.context_state
        if not isinstance(context_state, dict):
            raise ValueError("SubAgent context_state must be an object")
        object.__setattr__(self, "context_state", context_state)
        _ensure_json_value(list(self.conversation), field="conversation")
        _ensure_json_value(context_state, field="context_state")
        creator_snapshot = copy.deepcopy(self.creator_snapshot)
        creator_snapshot.__post_init__()
        object.__setattr__(self, "creator_snapshot", creator_snapshot)
        object.__setattr__(self, "conversation", copy.deepcopy(self.conversation))
        object.__setattr__(self, "context_state", copy.deepcopy(context_state))
        if any(not isinstance(path, str) or not path for path in self.artifact_paths):
            raise ValueError("artifact_paths must contain nonempty strings")
        _validate_artifact_paths(self.artifact_paths)
        if self.result is not None and not isinstance(self.result, str):
            raise ValueError("result must be a string or null")
        if self.error is not None and not isinstance(self.error, SubAgentError):
            raise ValueError("error must be a SubAgentError or null")
        if self.usage is None:
            object.__setattr__(self, "usage", empty_token_usage())
        else:
            _validate_usage(self.usage)
            object.__setattr__(self, "usage", copy.deepcopy(self.usage))
        if self.status is SubAgentStatus.QUEUED:
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("queued SubAgent cannot have start or finish timestamps")
        elif self.status is SubAgentStatus.RUNNING:
            if self.started_at is None or self.finished_at is not None:
                raise ValueError("running SubAgent requires only a start timestamp")
        elif self.status is SubAgentStatus.WAITING_FOR_ANSWER:
            if self.source.kind is not SubAgentSourceKind.FOREGROUND:
                raise ValueError("only foreground SubAgents may wait for an answer")
            if self.started_at is None or self.finished_at is not None:
                raise ValueError("waiting SubAgent requires only a start timestamp")
            if self.result is not None or self.error is not None:
                raise ValueError("waiting SubAgent cannot have a result or error")
        elif self.finished_at is None:
            raise ValueError("terminal SubAgent requires a finish timestamp")
        if self.status is SubAgentStatus.COMPLETED and self.result is None:
            raise ValueError("completed SubAgent requires a result")
        if self.status is SubAgentStatus.FAILED and self.error is None:
            raise ValueError("failed SubAgent requires an error")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "title": self.title,
            "task": self.task,
            "parent_run_id": self.parent_run_id,
            "source": self.source.to_dict(),
            "creator_snapshot": self.creator_snapshot.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": None if self.started_at is None else self.started_at.isoformat(),
            "finished_at": None if self.finished_at is None else self.finished_at.isoformat(),
            "registered_order": self.registered_order,
            "revision": self.revision,
            "status": self.status.value,
            "conversation": copy.deepcopy(list(self.conversation)),
            "context_state": copy.deepcopy(
                {} if self.context_state is None else self.context_state
            ),
            "artifact_paths": list(self.artifact_paths),
            "result": self.result,
            "error": None if self.error is None else self.error.to_dict(),
            "usage": copy.deepcopy(self.usage),
            "execution_id": self.execution_id,
            "restore_run_token": self.restore_run_token,
            "restore_prefixes": copy.deepcopy(list(self.restore_prefixes)),
        }

    @classmethod
    def from_dict(cls, value: object) -> SubAgentRecord:
        if not isinstance(value, dict):
            raise ValueError("SubAgent record must be an object")
        fields = {
            "schema_version",
            "agent_id",
            "session_id",
            "title",
            "task",
            "parent_run_id",
            "source",
            "creator_snapshot",
            "created_at",
            "started_at",
            "finished_at",
            "registered_order",
            "revision",
            "status",
            "conversation",
            "context_state",
            "artifact_paths",
            "result",
            "error",
            "usage",
        }
        schema_version = _integer(value.get("schema_version"), field="schema_version")
        if schema_version not in {2, 3, SUBAGENT_RECORD_SCHEMA_VERSION}:
            raise ValueError("SubAgent record schema version is unsupported")
        current_fields = fields | {
            "execution_id",
            "restore_run_token",
            "restore_prefixes",
        }
        expected_fields = current_fields if schema_version == SUBAGENT_RECORD_SCHEMA_VERSION else fields
        if set(value) != expected_fields:
            raise ValueError("SubAgent record fields do not match the current format")
        source = _source_from_dict(value["source"], schema_version=schema_version)
        creator_snapshot = _snapshot_from_dict(value["creator_snapshot"])
        created_at = _datetime_from_value(value["created_at"], field="created_at")
        started_at_value = value["started_at"]
        started_at = (
            None
            if started_at_value is None
            else _datetime_from_value(started_at_value, field="started_at")
        )
        finished_at_value = value["finished_at"]
        finished_at = (
            None
            if finished_at_value is None
            else _datetime_from_value(finished_at_value, field="finished_at")
        )
        conversation = value["conversation"]
        if not isinstance(conversation, list) or any(
            not isinstance(item, dict) for item in conversation
        ):
            raise ValueError("SubAgent conversation must be a list of objects")
        context_state = value["context_state"]
        if not isinstance(context_state, dict):
            raise ValueError("SubAgent context_state must be an object")
        artifact_paths = value["artifact_paths"]
        if not isinstance(artifact_paths, list) or any(
            not isinstance(item, str) for item in artifact_paths
        ):
            raise ValueError("SubAgent artifact_paths must be a list of strings")
        result = value["result"]
        if result is not None and not isinstance(result, str):
            raise ValueError("SubAgent result must be a string or null")
        error = _error_from_value(value["error"])
        usage = _usage_from_value(value["usage"])
        execution_id = (
            None
            if schema_version < SUBAGENT_RECORD_SCHEMA_VERSION
            else _optional_string(value["execution_id"], field="execution_id")
        )
        restore_run_token = (
            None
            if schema_version < SUBAGENT_RECORD_SCHEMA_VERSION
            else _optional_string(value["restore_run_token"], field="restore_run_token")
        )
        restore_prefixes_value = (
            [] if schema_version < SUBAGENT_RECORD_SCHEMA_VERSION else value["restore_prefixes"]
        )
        if not isinstance(restore_prefixes_value, list) or any(
            not isinstance(item, dict) for item in restore_prefixes_value
        ):
            raise ValueError("SubAgent restore_prefixes must be a list of objects")
        return cls(
            agent_id=_string(value["agent_id"], field="agent_id"),
            session_id=_string(value["session_id"], field="session_id"),
            title=_string(value["title"], field="title"),
            task=_string(value["task"], field="task"),
            parent_run_id=_string(value["parent_run_id"], field="parent_run_id"),
            source=source,
            creator_snapshot=creator_snapshot,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            registered_order=_integer(value["registered_order"], field="registered_order"),
            revision=_integer(value["revision"], field="revision"),
            status=SubAgentStatus(_string(value["status"], field="status")),
            conversation=tuple(cast(dict[str, Any], item) for item in conversation),
            context_state=cast(dict[str, Any], context_state),
            artifact_paths=tuple(artifact_paths),
            result=result,
            error=error,
            usage=usage,
            execution_id=execution_id,
            restore_run_token=restore_run_token,
            restore_prefixes=tuple(cast(dict[str, Any], item) for item in restore_prefixes_value),
            schema_version=SUBAGENT_RECORD_SCHEMA_VERSION,
        )


@dataclass(frozen=True, slots=True)
class SubAgentListItem:
    """Brief task information returned by the Session-scoped list query."""

    agent_id: str
    title: str
    status: SubAgentStatus
    created_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class SubAgentPage:
    """One stable page of brief SubAgent records."""

    items: tuple[SubAgentListItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class SubAgentWaitResult:
    """Compact terminal result or nonterminal question attention for a waiting Main Agent."""

    agent_id: str
    title: str
    status: SubAgentStatus
    result: str | None
    error: SubAgentError | None
    usage: dict[str, int]
    question_id: str | None = None

    def __post_init__(self) -> None:
        require_uuid4_string(self.agent_id, field="agent_id")
        _require_nonempty_string(self.title, field="title")
        if not isinstance(self.status, SubAgentStatus):
            raise ValueError("SubAgent wait result status is invalid")
        if self.status in {
            SubAgentStatus.QUEUED,
            SubAgentStatus.RUNNING,
        }:
            raise ValueError("SubAgent wait result must be terminal or awaiting an answer")
        if (self.status is SubAgentStatus.WAITING_FOR_ANSWER) != (self.question_id is not None):
            raise ValueError("waiting SubAgent result requires its question ID")
        if self.question_id is not None:
            _require_nonempty_string(self.question_id, field="question_id")
        if self.result is not None and not isinstance(self.result, str):
            raise ValueError("result must be a string or null")
        if self.error is not None and not isinstance(self.error, SubAgentError):
            raise ValueError("error must be a SubAgentError or null")
        _validate_usage(self.usage)
        if self.status is SubAgentStatus.COMPLETED and self.result is None:
            raise ValueError("completed SubAgent wait result requires a result")
        if self.status is SubAgentStatus.FAILED and self.error is None:
            raise ValueError("failed SubAgent wait result requires an error")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "agent_id": self.agent_id,
            "title": self.title,
            "status": self.status.value,
            "result": self.result,
            "error": None if self.error is None else self.error.to_dict(),
            "usage": copy.deepcopy(self.usage),
        }
        if self.question_id is not None:
            value["question_id"] = self.question_id
        return value


@dataclass(frozen=True, slots=True)
class SubAgentExecutionResult:
    """One bounded execution outcome before it is published as a record update."""

    status: SubAgentStatus
    conversation: tuple[dict[str, Any], ...]
    context_state: dict[str, Any]
    artifact_paths: tuple[str, ...]
    result: str | None
    error: SubAgentError | None
    usage: dict[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.status, SubAgentStatus):
            raise ValueError("SubAgent execution result status is invalid")
        if self.status not in {
            SubAgentStatus.WAITING_FOR_ANSWER,
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.CANCELLED,
            SubAgentStatus.INTERRUPTED,
        }:
            raise ValueError(
                "SubAgent execution result must be waiting, completed, failed, cancelled, or interrupted"
            )
        if any(not isinstance(message, dict) for message in self.conversation):
            raise ValueError("SubAgent conversation entries must be objects")
        _ensure_json_value(list(self.conversation), field="conversation")
        if not isinstance(self.context_state, dict):
            raise ValueError("SubAgent context_state must be an object")
        _ensure_json_value(self.context_state, field="context_state")
        _validate_artifact_paths(self.artifact_paths)
        if self.result is not None and not isinstance(self.result, str):
            raise ValueError("result must be a string or null")
        if self.error is not None and not isinstance(self.error, SubAgentError):
            raise ValueError("error must be a SubAgentError or null")
        _validate_usage(self.usage)
        if self.status is SubAgentStatus.COMPLETED and self.result is None:
            raise ValueError("completed SubAgent requires a result")
        if self.status is SubAgentStatus.FAILED and self.error is None:
            raise ValueError("failed SubAgent requires an error")
        if self.status is SubAgentStatus.INTERRUPTED and self.error is None:
            raise ValueError("interrupted SubAgent requires an error")


@dataclass(frozen=True, slots=True)
class SubAgentEvent:
    """Workspace-scoped event identity and payload for a SubAgent update."""

    kind: SubAgentEventKind
    workspace_id: str
    session_id: str
    agent_id: str
    revision: int
    occurred_at: datetime
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SubAgentEventKind):
            raise ValueError("SubAgent event kind is invalid")
        _require_nonempty_string(self.workspace_id, field="workspace_id")
        validate_session_id(self.session_id)
        require_uuid4_string(self.agent_id, field="agent_id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("event revision must be a nonnegative integer")
        require_aware_datetime(self.occurred_at, field="occurred_at")
        if not isinstance(self.data, dict):
            raise ValueError("event data must be an object")
        _ensure_json_value(self.data, field="event data")


def _source_from_dict(value: object, *, schema_version: int) -> SubAgentSource:
    legacy_fields = {
        "kind",
        "restore_run_token",
        "job_id",
        "occurrence_id",
    }
    current_fields = legacy_fields | {"task_id", "task_generation"}
    expected_fields = legacy_fields if schema_version == 2 else current_fields
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("SubAgent source fields do not match the current format")
    return SubAgentSource(
        kind=SubAgentSourceKind(_string(value["kind"], field="source.kind")),
        restore_run_token=_optional_string(
            value["restore_run_token"], field="source.restore_run_token"
        ),
        job_id=_optional_string(value["job_id"], field="source.job_id"),
        occurrence_id=_optional_string(value["occurrence_id"], field="source.occurrence_id"),
        task_id=(
            None
            if schema_version == 2
            else _optional_string(value["task_id"], field="source.task_id")
        ),
        task_generation=(
            None
            if schema_version == 2 or value["task_generation"] is None
            else _integer(value["task_generation"], field="source.task_generation")
        ),
    )


def _snapshot_from_dict(value: object) -> SubAgentCreatorSnapshot:
    if not isinstance(value, dict) or set(value) != {
        "provider_id",
        "model",
        "reasoning_effort",
        "permission_level",
        "shell",
        "tool_names",
        "system_prompt",
    }:
        raise ValueError("SubAgent creator snapshot fields do not match the current format")
    tool_names = value["tool_names"]
    if not isinstance(tool_names, list) or any(not isinstance(item, str) for item in tool_names):
        raise ValueError("SubAgent creator snapshot tool_names must be a list of strings")
    return SubAgentCreatorSnapshot(
        provider_id=_string(value["provider_id"], field="creator_snapshot.provider_id"),
        model=_string(value["model"], field="creator_snapshot.model"),
        reasoning_effort=cast(
            ReasoningEffort,
            _string(value["reasoning_effort"], field="creator_snapshot.reasoning_effort"),
        ),
        permission_level=_string(
            value["permission_level"], field="creator_snapshot.permission_level"
        ),
        shell=_optional_string(value["shell"], field="creator_snapshot.shell"),
        tool_names=tuple(tool_names),
        system_prompt=_string(value["system_prompt"], field="creator_snapshot.system_prompt"),
    )


def _error_from_value(value: object) -> SubAgentError | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"code", "message"}:
        raise ValueError("SubAgent error fields do not match the current format")
    return SubAgentError(
        code=_string(value["code"], field="error.code"),
        message=_string(value["message"], field="error.message"),
    )


def _usage_from_value(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _USAGE_FIELDS:
        raise ValueError("SubAgent usage fields do not match the Agent Runner contract")
    if any(isinstance(member, bool) or not isinstance(member, int) for member in value.values()):
        raise ValueError("SubAgent usage values must be integers")
    usage = cast(dict[str, int], value)
    _validate_usage(usage)
    return usage


def _restore_prefix_from_value(value: object) -> dict[str, Any]:
    fields = {
        "restore_run_token",
        "started_at",
        "finished_at",
        "conversation",
        "context_state",
        "artifact_paths",
        "usage",
        "execution_id",
        "current_restore_run_token",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("SubAgent restore prefix fields do not match the current format")
    restore_run_token = _string(
        value["restore_run_token"], field="restore_prefix.restore_run_token"
    )
    execution_id = _string(value["execution_id"], field="restore_prefix.execution_id")
    require_uuid4_string(restore_run_token, field="restore_prefix.restore_run_token")
    require_uuid4_string(execution_id, field="restore_prefix.execution_id")
    current_restore_run_token = _optional_string(
        value["current_restore_run_token"], field="restore_prefix.current_restore_run_token"
    )
    if current_restore_run_token is not None:
        require_uuid4_string(
            current_restore_run_token, field="restore_prefix.current_restore_run_token"
        )
    _datetime_from_value(value["started_at"], field="restore_prefix.started_at")
    if value["finished_at"] is not None:
        raise ValueError("waiting restore prefix cannot have a finish timestamp")
    conversation = value["conversation"]
    if not isinstance(conversation, list) or any(
        not isinstance(item, dict) for item in conversation
    ):
        raise ValueError("SubAgent restore prefix conversation must be a list of objects")
    context_state = value["context_state"]
    if not isinstance(context_state, dict):
        raise ValueError("SubAgent restore prefix context_state must be an object")
    artifact_paths = value["artifact_paths"]
    if not isinstance(artifact_paths, list) or any(
        not isinstance(item, str) for item in artifact_paths
    ):
        raise ValueError("SubAgent restore prefix artifact_paths must be a list of strings")
    _validate_artifact_paths(tuple(artifact_paths))
    usage = _usage_from_value(value["usage"])
    _ensure_json_value(conversation, field="restore_prefix.conversation")
    _ensure_json_value(context_state, field="restore_prefix.context_state")
    return {
        "restore_run_token": restore_run_token,
        "started_at": value["started_at"],
        "finished_at": value["finished_at"],
        "conversation": copy.deepcopy(conversation),
        "context_state": copy.deepcopy(context_state),
        "artifact_paths": list(artifact_paths),
        "usage": copy.deepcopy(usage),
        "execution_id": execution_id,
        "current_restore_run_token": current_restore_run_token,
    }


def _validate_usage(value: dict[str, int]) -> None:
    issue = token_usage_validation_issue(value)
    if issue == "fields":
        raise ValueError("SubAgent usage must contain the four Agent Runner usage fields")
    if issue == "values":
        raise ValueError("SubAgent usage values must be nonnegative integers")
    if issue == "total":
        raise ValueError("SubAgent total_tokens must equal input_tokens + output_tokens")


def _validate_artifact_paths(paths: tuple[str, ...]) -> None:
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("artifact_paths must contain nonempty strings")
    for path in paths:
        path_parts = PurePosixPath(path).parts
        if (
            not path_parts
            or "\\" in path
            or ":" in path_parts[0]
            or PurePosixPath(path).is_absolute()
            or ".." in path_parts
        ):
            raise ValueError("artifact_paths must be relative and remain within the Workspace")


def _datetime_from_value(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a date-time string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid date-time") from error
    require_aware_datetime(parsed, field=field)
    return parsed


def _ensure_json_value(value: object, *, field: str) -> None:
    try:
        issue = json_validation_issue(value, field=field)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{field} must contain only standard JSON values") from error
    if issue is not None:
        raise ValueError(f"{field} must contain only standard JSON values")


def _require_nonempty_string(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value
