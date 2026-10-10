"""Schedule command validation and fingerprints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

from aide.schedule.model import JobSchedule
from aide.service.errors import service_error
from aide.service.runtime.records import _MISSING
from aide.utils.text import normalize_title_candidate


def _schedule_request_fingerprint(
    workspace_id: str,
    action: str,
    payload: Mapping[str, object],
) -> str:
    try:
        return json.dumps(
            {
                "workspace_id": workspace_id,
                "action": action,
                "payload": {key: value for key, value in payload.items() if key != "request_id"},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise service_error(
            "validation_error", "Schedule Job input is not valid JSON.", status=422
        ) from error


def _schedule_job_input(payload: Mapping[str, object]) -> tuple[str, str, JobSchedule]:
    """Normalize the existing Schedule Tool input shape for the HTTP boundary."""
    field_errors: dict[str, str] = {}
    message = payload.get("message")
    if not isinstance(message, str):
        normalized_message = ""
        field_errors["message"] = "must be a string"
    else:
        normalized_message = message.strip()
        if not normalized_message:
            field_errors["message"] = "must not be empty"
        elif len(normalized_message) > 20_000:
            field_errors["message"] = "must not exceed 20000 characters"

    title_value = payload.get("title", _MISSING)
    if title_value is _MISSING:
        normalized_title = normalize_title_candidate(normalized_message)
    elif not isinstance(title_value, str):
        normalized_title = ""
        field_errors["title"] = "must be a string"
    else:
        normalized_title = normalize_title_candidate(title_value)
        if not normalized_title:
            field_errors["title"] = "must not be empty"

    nested_schedule = payload.get("schedule")
    if nested_schedule is not None and not isinstance(nested_schedule, Mapping):
        field_errors["schedule"] = "must be an object"
        nested: Mapping[str, object] = {}
    else:
        nested = nested_schedule if isinstance(nested_schedule, Mapping) else {}

    def schedule_value(name: str) -> object:
        if name in payload:
            return payload[name]
        return nested.get(name)

    kind_value = payload.get("kind", nested.get("kind"))
    at_time = schedule_value("at_time")
    every_seconds = schedule_value("every_seconds")
    cron_expr = schedule_value("cron_expr")
    timezone = schedule_value("timezone")
    selected = [
        name
        for name, value in (
            ("at", at_time),
            ("every", every_seconds),
            ("cron", cron_expr),
        )
        if value is not None
    ]

    selected_kind: str | None = None
    if kind_value is not None:
        if not isinstance(kind_value, str) or kind_value not in {"at", "every", "cron"}:
            field_errors["kind"] = "must be at, every, or cron"
        else:
            selected_kind = kind_value
            if selected != [kind_value]:
                field_errors["schedule"] = "must select exactly one matching schedule kind"
    elif len(selected) != 1:
        field_errors["schedule"] = "must select exactly one of at_time, every_seconds, or cron_expr"
    else:
        selected_kind = selected[0]

    schedule: JobSchedule | None = None
    if selected_kind == "at" and "schedule" not in field_errors:
        if timezone is not None:
            field_errors["timezone"] = "is only valid for cron schedules"
        if not isinstance(at_time, str):
            field_errors["at_time"] = "must be a timezone-aware ISO time"
        else:
            try:
                schedule = JobSchedule.from_at_input(at_time)
            except (TypeError, ValueError):
                field_errors["at_time"] = "must be a valid timezone-aware ISO time"
    elif selected_kind == "every" and "schedule" not in field_errors:
        if timezone is not None:
            field_errors["timezone"] = "is only valid for cron schedules"
        if isinstance(every_seconds, bool) or not isinstance(every_seconds, int):
            field_errors["every_seconds"] = "must be a positive integer"
        else:
            try:
                schedule = JobSchedule.every(every_seconds)
            except (TypeError, ValueError):
                field_errors["every_seconds"] = "must be a positive integer"
    elif selected_kind == "cron" and "schedule" not in field_errors:
        if not isinstance(cron_expr, str):
            field_errors["cron_expr"] = "must be a valid five-field cron expression"
        if timezone is not None and not isinstance(timezone, str):
            field_errors["timezone"] = "must be a valid IANA timezone"
        if isinstance(cron_expr, str):
            try:
                schedule = JobSchedule.from_cron_input(cron_expr)
            except (TypeError, ValueError):
                field_errors["cron_expr"] = "must be a valid five-field cron expression"
        if isinstance(timezone, str):
            try:
                validated_timezone = JobSchedule.from_cron_input("* * * * *", timezone)
            except (TypeError, ValueError):
                field_errors["timezone"] = "must be a valid IANA timezone"
            else:
                if schedule is not None:
                    schedule = JobSchedule.cron(
                        schedule.cron_expr or "", validated_timezone.timezone or "UTC"
                    )

    if field_errors:
        raise service_error(
            "validation_error",
            "Schedule Job input is invalid.",
            status=400,
            field_errors=field_errors,
        )
    assert schedule is not None
    return normalized_message, normalized_title, schedule


def _schedule_epoch_milliseconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise service_error("schedule_unavailable", "Schedule time is unavailable.", retryable=True)
    milliseconds = int(value.timestamp() * 1000)
    if milliseconds < 0:
        raise service_error("schedule_unavailable", "Schedule time is unavailable.", retryable=True)
    return milliseconds
