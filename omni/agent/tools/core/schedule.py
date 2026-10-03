"""Schedule Core Catalog Tool."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from omni.agent.tools.base import BaseTool, ToolError
from omni.agent.tools.permission import (
    ScheduleAction,
    ScheduleActionName,
    ToolInvocationFacts,
)
from omni.agent.tools.schema import Schema
from omni.schedule.model import JobSchedule, ScheduleJob
from omni.schedule.service import ScheduleService, ScheduleStaleRemovalError
from omni.utils.text import normalize_title_candidate
from omni.utils.validation import require_uuid4_string

_INVALID_ARGUMENTS = "Invalid arguments for schedule."
_NOT_FOUND = "Schedule Job was not found."
_STALE_REMOVAL = "Schedule Job changed before removal. Request removal again."
_STATE_READ_FAILED = "Schedule state could not be read."
_STATE_UPDATE_FAILED = "Schedule state could not be updated."


class _ScheduleArgumentsSchema(Schema):
    """Keep action-irrelevant Schedule fields out of schema casting."""

    def __init__(self) -> None:
        super().__init__(
            "object",
            properties={
                "action": Schema.string(
                    description="The exact action: add, list, or remove.",
                    min_length=1,
                ),
                "message": Schema.string(
                    description="The message to run for an added Schedule Job.",
                    nullable=True,
                    default=None,
                ),
                "title": Schema.string(
                    description="The stable user-visible title for an added Schedule Job.",
                ),
                "every_seconds": Schema.integer(
                    description="Run again this many seconds after completion.",
                    minimum=1,
                    nullable=True,
                    default=None,
                ),
                "cron_expr": Schema.string(
                    description="A canonical five-field Cron expression.",
                    nullable=True,
                    default=None,
                ),
                "timezone": Schema.string(
                    description="The IANA timezone for a Cron expression.",
                    nullable=True,
                    default=None,
                ),
                "at_time": Schema.string(
                    description="An absolute ISO time with a UTC offset.",
                    nullable=True,
                    default=None,
                ),
                "job_id": Schema.string(
                    description="The canonical UUID4 of a Job to remove.",
                    nullable=True,
                    default=None,
                ),
            },
            required=("action",),
        )

    def cast(self, value: object) -> object:
        if not isinstance(value, dict):
            return super().cast(value)

        action = value.get("action")
        projected: dict[str, object] = {"action": action}
        if action == "remove":
            job_id = value.get("job_id")
            if job_id is not None:
                projected["job_id"] = job_id
        elif action == "add":
            message = value.get("message")
            if message is not None:
                projected["message"] = message
            if "title" in value:
                projected["title"] = value["title"]
            for schedule_name in ("every_seconds", "cron_expr", "at_time"):
                schedule_value = value.get(schedule_name)
                if schedule_value is None:
                    continue
                projected[schedule_name] = schedule_value
                if schedule_name == "cron_expr":
                    timezone = value.get("timezone")
                    if timezone is not None:
                        projected["timezone"] = timezone
                break
        return super().cast(projected)


class ScheduleTool(BaseTool):
    """Add, list, and remove user-owned Schedule Jobs."""

    name = "schedule"
    description = "Manage one-time and recurring Schedule Jobs."
    parameters = _ScheduleArgumentsSchema().to_json_schema()

    def __init__(
        self,
        *,
        schedule_service: ScheduleService,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
    ) -> None:
        if not isinstance(schedule_service, ScheduleService):
            raise TypeError("Schedule Tool requires a ScheduleService")
        self._schedule_service = schedule_service
        self._now: Callable[[], datetime] = (lambda: datetime.now(UTC)) if now is None else now
        self._new_uuid: Callable[[], UUID] = uuid4 if new_uuid is None else new_uuid

    def _build_preparation_schema(self) -> Schema:
        return _ScheduleArgumentsSchema()

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if (
            arguments.get("action") == "add"
            and "title" in arguments
            and arguments["title"] is None
        ):
            raise ToolError(_INVALID_ARGUMENTS)
        prepared = await super().prepare_arguments(arguments)
        if prepared.get("action") == "remove":
            job_id = prepared.get("job_id")
            if isinstance(job_id, str):
                try:
                    require_uuid4_string(job_id, field="job_id")
                except ValueError:
                    return prepared
                if await self._current_public_job(job_id) is None:
                    raise ToolError(_NOT_FOUND)
        return prepared

    def build_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        action = prepared_arguments.get("action")
        if not isinstance(action, str) or action not in {"list", "add", "remove"}:
            raise ToolError(_INVALID_ARGUMENTS)
        if (
            action in {"add", "remove"}
            and self._schedule_service.status_snapshot().status == "faulted"
        ):
            raise ToolError(_STATE_UPDATE_FAILED)
        normalized_arguments = self._normalized_invocation_arguments(
            prepared_arguments,
            action=cast(Literal["list", "add", "remove"], action),
        )
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=normalized_arguments,
            execution_arguments=prepared_arguments,
            schedule_action=ScheduleAction(action=cast(ScheduleActionName, action)),
        )

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        action: str,
        message: str | None = None,
        title: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        at_time: str | None = None,
        job_id: str | None = None,
    ) -> str | None:
        if action == "add":
            self._normalize_add(
                message=message,
                title=title,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                timezone=timezone,
                at_time=at_time,
            )
            return None
        if action == "list":
            return None
        if action == "remove" and job_id is not None:
            try:
                require_uuid4_string(job_id, field="job_id")
            except ValueError:
                return _INVALID_ARGUMENTS
            return None
        return _INVALID_ARGUMENTS

    async def execute(
        self,
        *,
        action: str,
        message: str | None = None,
        title: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        at_time: str | None = None,
        job_id: str | None = None,
    ) -> str:
        if action == "add":
            normalized_message, normalized_title, schedule = self._normalize_add(
                message=message,
                title=title,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                timezone=timezone,
                at_time=at_time,
            )
            timestamp = self._epoch_milliseconds(self._aware_now())
            job = ScheduleJob(
                job_id=str(self._new_uuid()),
                message=normalized_message,
                title=normalized_title,
                schedule=schedule,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
            )
            try:
                await self._schedule_service.add_user_job(job)
            except Exception as error:
                raise ToolError(_STATE_UPDATE_FAILED) from error
            return _json_content({"action": "add", "job": _public_job(job)})

        if action == "list":
            try:
                jobs = await self._schedule_service.public_snapshot()
            except Exception as error:
                raise ToolError(_STATE_READ_FAILED) from error
            return _json_content({"jobs": [_public_job(job) for job in jobs]})

        if action == "remove":
            if job_id is None:
                raise ToolError(_INVALID_ARGUMENTS)
            try:
                require_uuid4_string(job_id, field="job_id")
            except ValueError as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            public_job = await self._current_public_job(job_id)
            if public_job is None:
                raise ToolError(_NOT_FOUND)
            try:
                removed = await self._schedule_service.remove_user_job(
                    job_id,
                    expected=public_job,
                )
            except ScheduleStaleRemovalError as error:
                raise ToolError(_STALE_REMOVAL) from error
            except Exception as error:
                raise ToolError(_STATE_UPDATE_FAILED) from error
            if not removed:
                raise ToolError(_STALE_REMOVAL)
            return _json_content({"action": "remove", "job": _remove_result_job(public_job)})

        raise ToolError(_INVALID_ARGUMENTS)

    def _normalize_add(
        self,
        *,
        message: str | None,
        title: str | None,
        every_seconds: int | None,
        cron_expr: str | None,
        timezone: str | None,
        at_time: str | None,
    ) -> tuple[str, str, JobSchedule]:
        if not isinstance(message, str):
            raise ToolError(_INVALID_ARGUMENTS)
        normalized_message = message.strip()
        if not normalized_message or len(normalized_message) > 20_000:
            raise ToolError(_INVALID_ARGUMENTS)
        if title is not None and not isinstance(title, str):
            raise ToolError(_INVALID_ARGUMENTS)
        normalized_title = normalize_title_candidate(
            normalized_message if title is None else title
        )
        if not normalized_title:
            raise ToolError(_INVALID_ARGUMENTS)

        if every_seconds is not None:
            try:
                schedule = JobSchedule.every(every_seconds)
            except (TypeError, ValueError) as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, normalized_title, schedule

        if cron_expr is not None:
            try:
                schedule = JobSchedule.from_cron_input(cron_expr, timezone)
            except (TypeError, ValueError) as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, normalized_title, schedule

        if at_time is not None:
            try:
                schedule = JobSchedule.from_at_input(at_time)
            except (TypeError, ValueError) as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, normalized_title, schedule

        raise ToolError(_INVALID_ARGUMENTS)

    def _aware_now(self) -> datetime:
        current = self._now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ToolError(_INVALID_ARGUMENTS)
        return current

    @staticmethod
    def _epoch_milliseconds(value: datetime) -> int:
        milliseconds = int(value.timestamp() * 1000)
        if milliseconds < 0:
            raise ToolError(_INVALID_ARGUMENTS)
        return milliseconds

    async def _current_public_job(self, job_id: str) -> ScheduleJob | None:
        try:
            jobs = await self._schedule_service.public_snapshot()
        except Exception as error:
            raise ToolError(_STATE_READ_FAILED) from error
        return next((job for job in jobs if job.job_id == job_id), None)

    def _normalized_invocation_arguments(
        self,
        prepared_arguments: dict[str, Any],
        *,
        action: Literal["list", "add", "remove"],
    ) -> dict[str, Any]:
        if action == "list":
            return {"action": "list"}
        if action == "remove":
            job_id = prepared_arguments.get("job_id")
            if not isinstance(job_id, str):
                raise ToolError(_INVALID_ARGUMENTS)
            return {"action": "remove", "job_id": job_id}
        normalized_message, normalized_title, schedule = self._normalize_add(
            message=cast(str | None, prepared_arguments.get("message")),
            title=cast(str | None, prepared_arguments.get("title")),
            every_seconds=cast(int | None, prepared_arguments.get("every_seconds")),
            cron_expr=cast(str | None, prepared_arguments.get("cron_expr")),
            timezone=cast(str | None, prepared_arguments.get("timezone")),
            at_time=cast(str | None, prepared_arguments.get("at_time")),
        )
        return {
            "action": "add",
            "message": normalized_message,
            "title": normalized_title,
            "schedule": _public_schedule(schedule),
        }


def _public_schedule(schedule: JobSchedule) -> dict[str, Any]:
    if schedule.kind == "at":
        return {"type": "at", "at_time": cast(str, schedule.at_time)}
    if schedule.kind == "every":
        return {"type": "every", "every_seconds": cast(int, schedule.every_seconds)}
    return {
        "type": "cron",
        "cron_expr": cast(str, schedule.cron_expr),
        "timezone": cast(str, schedule.timezone),
    }


def _public_job(job: ScheduleJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "message": job.message,
        "schedule": _public_schedule(job.schedule),
    }


def _remove_result_job(job: ScheduleJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "message": job.message,
        "schedule": _public_schedule(job.schedule),
    }


def _json_content(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["ScheduleTool"]
