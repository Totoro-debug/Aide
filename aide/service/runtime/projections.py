"""Protocol projections of runtime results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from aide.schedule.model import ScheduleJob
from aide.service.contracts import (
    ProjectJobSummaryDTO,
)
from aide.service.errors import ServiceError, service_error
from aide.service.projects import ProjectCatalogError


def _project_job_summary(job: ScheduleJob) -> ProjectJobSummaryDTO:
    due_at: datetime | None = None
    if job.schedule.kind == "at":
        due_at = job.schedule.at_datetime
    elif job.schedule.kind == "every" and job.schedule.every_seconds is not None:
        anchor_ms = (
            job.state.last_finished_at_ms
            if job.state.last_finished_at_ms is not None
            else job.created_at_ms
        )
        due_at = datetime.fromtimestamp(anchor_ms / 1000, UTC) + timedelta(
            seconds=job.schedule.every_seconds
        )
    if job.schedule.kind == "at" and job.state.last_status is not None:
        review_status = "completed"
    elif due_at is None:
        review_status = "next_on_resume"
    else:
        review_status = "overdue" if due_at <= datetime.now(UTC) else "upcoming"
    summary: ProjectJobSummaryDTO = {
        "job_id": job.job_id,
        "title": cast(str, job.title),
        "schedule": job.schedule.to_dict(),
        "due_at": due_at.isoformat() if due_at is not None else None,
        "review_status": review_status,
    }
    return summary


def _schedule_job_projection(
    job: ScheduleJob,
    *,
    active: bool,
    status: str | None = None,
) -> dict[str, object]:
    """Return the public Job shape without exposing the Schedule store."""
    projection = job.to_dict()
    projection["session_id"] = job.session_id
    projection["active"] = active
    projection["status"] = status or ("running" if active else job.state.last_status or "scheduled")
    return projection


def _project_catalog_service_error(error: ProjectCatalogError) -> ServiceError:
    message = str(error)
    if "unavailable" in message:
        return service_error(
            "not_found",
            "Project directory is unavailable.",
            status=404,
            field_errors={"path": "must name an existing directory"},
        )
    if "catalog" in message or "entries" in message or "format" in message:
        return service_error(
            "persistence_error", "The Project catalog could not be read safely.", status=500
        )
    if "overlaps Agent Home" in message:
        detail = "must not overlap Agent Home"
    elif "absolute directory" in message:
        detail = "must be an absolute directory"
    else:
        detail = "must identify a usable local directory"
    return service_error(
        "validation_error",
        "Project path is invalid.",
        status=422,
        field_errors={"path": detail},
    )


def _encode_management_result(result: object) -> dict[str, object]:
    from aide.management.commands import ManagementCommandResult

    if not isinstance(result, ManagementCommandResult):
        raise service_error("service_protocol_error", "Management result is invalid.", status=500)
    encoded: dict[str, object] = {
        "handled": result.handled,
        "output": result.output,
        "memory_content": result.memory_content,
        "effort_selection": result.effort_selection,
        "permission_selection": result.permission_selection,
        "published_effort": result.published_effort,
        "published_permission_level": result.published_permission_level,
        "resumed_session_id": result.resumed_session_id,
        "resume_skipped_count": result.resume_skipped_count,
    }
    if result.dream_result is not None:
        encoded["dream_result"] = _safe_wire_value(result.dream_result)
    if result.management_error is not None:
        encoded["management_error"] = _safe_wire_value(result.management_error)
    if result.status_view is not None:
        encoded["status_view"] = result.status_view.to_dict()
    if result.resume_sessions is not None:
        encoded["resume_sessions"] = [
            {
                "id": item.id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "message_count": item.message_count,
            }
            for item in result.resume_sessions
        ]
    if result.skill_metadata is not None:
        encoded["skill_metadata"] = [
            {"name": item.name, "description": item.description, "path": str(item.path)}
            for item in result.skill_metadata
        ]
    if result.restore_listing is not None:
        encoded["restore_listing"] = {
            "session_id": result.restore_listing.session_id,
            "anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "run_token": str(anchor.run_token),
                    "content": anchor.content,
                    "timestamp": anchor.timestamp,
                }
                for anchor in result.restore_listing.anchors
            ],
        }
    if result.restore_plan is not None:
        encoded["restore_plan"] = _safe_wire_value(result.restore_plan)
    if result.restore_result is not None:
        encoded["restore_result"] = _safe_wire_value(result.restore_result)
    return encoded


def _safe_wire_value(value: object) -> object:
    from datetime import datetime
    from enum import Enum
    from uuid import UUID

    if is_dataclass(value):
        return {
            item.name: _safe_wire_value(getattr(value, item.name))
            for item in fields(value)
            if item.name != "before_bytes"
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_wire_value(item) for item in value]
    if isinstance(value, (Path, UUID, datetime, Enum)):
        return str(value)
    if isinstance(value, bytes):
        return None
    return value
