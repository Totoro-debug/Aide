"""Workspace-owned strict Schedule Job state and mutation boundary."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from aide.agent.workspace_state import WorkspaceState, WorkspaceStateError
from aide.schedule.model import DREAM_JOB_ID, JobStatus, ScheduleJob, ScheduleJobState
from aide.utils.errors import ErrorInfo
from aide.utils.host_filesystem import HOST_FILESYSTEM
from aide.utils.json import strict_json_loads
from aide.utils.validation import require_nonnegative_int, require_uuid4_string

StoreHealth = Literal["available", "faulted"]
ReplaceText = Callable[[Path, str], None]


class ScheduleStoreError(RuntimeError):
    """Base error for the Schedule Store boundary."""


class ScheduleStateError(ScheduleStoreError, WorkspaceStateError):
    """Raised when the Workspace Schedule document cannot be loaded strictly."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.error = ErrorInfo(
            "schedule_state_error",
            "Schedule state could not be loaded. Repair or move the file, then start Aide again.",
        )
        Exception.__init__(self, self.error.message)


class ScheduleStoreFaultedError(ScheduleStoreError):
    """Raised when a Runtime-local Store has latched a mutation failure."""


class ScheduleStaleRemovalError(ScheduleStoreError):
    """Raised when the public Job snapshot changes before removal."""


@dataclass(frozen=True, slots=True)
class ScheduleReservation:
    """One due Job paired with its durable user occurrence identity."""

    job: ScheduleJob
    occurrence_id: UUID | None


class WorkspaceScheduleStore:
    """Strict, copy-on-write Schedule state for one Runtime and Workspace."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_text: ReplaceText | None = None,
    ) -> None:
        self.workspace_state = workspace_state
        self.path = workspace_state.schedule_path
        self._state_path = workspace_state.path
        workspace_root = workspace_state.workspace_path.resolve(strict=True)
        try:
            self._state_root = HOST_FILESYSTEM.require_owned_directory(
                self._state_path,
                within=workspace_root,
            )
            self._jobs, self._pending_occurrences = self._load_once()
        except ScheduleStateError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            raise ScheduleStateError(self.path) from error
        self._replace_text = (
            HOST_FILESYSTEM.atomic_replace_text if replace_text is None else replace_text
        )
        self._condition = asyncio.Condition()
        self._revision = 0
        self._faulted = False

    @property
    def health(self) -> StoreHealth:
        return "faulted" if self._faulted else "available"

    @property
    def revision(self) -> int:
        return self._revision

    async def snapshot(self) -> tuple[ScheduleJob, ...]:
        async with self._condition:
            return copy.deepcopy(self._jobs)

    async def reserve_due(
        self,
        candidates: tuple[ScheduleJob, ...],
    ) -> tuple[ScheduleReservation, ...]:
        """Durably reserve due user occurrences while holding Store authority."""
        async with self._condition:
            self._ensure_available()
            current = {job.job_id: job for job in self._jobs}
            pending = dict(self._pending_occurrences)
            reservations: list[ScheduleReservation] = []
            changed = False
            for candidate in sorted(candidates, key=lambda item: item.job_id):
                job = current.get(candidate.job_id)
                if job != candidate:
                    continue
                if job.source != "user":
                    reservations.append(ScheduleReservation(copy.deepcopy(job), None))
                    continue
                occurrence_id = pending.get(job.job_id)
                if occurrence_id is None:
                    occurrence_id = uuid4()
                    pending[job.job_id] = occurrence_id
                    changed = True
                reservations.append(ScheduleReservation(copy.deepcopy(job), occurrence_id))
            if changed:
                self._publish(self._jobs, pending)
            return tuple(reservations)

    async def pending_occurrences(self) -> tuple[ScheduleReservation, ...]:
        """Return durable user occurrence intents eligible for restart replay."""
        async with self._condition:
            self._ensure_available()
            jobs = {job.job_id: job for job in self._jobs}
            return tuple(
                ScheduleReservation(copy.deepcopy(jobs[job_id]), occurrence_id)
                for job_id, occurrence_id in sorted(self._pending_occurrences.items())
                if job_id in jobs
            )

    async def clear_pending_occurrence(self, job_id: str, occurrence_id: UUID) -> bool:
        """Release one exact recurring reservation after an in-process pause."""
        require_uuid4_string(job_id, field="job_id")
        if not isinstance(occurrence_id, UUID) or occurrence_id.version != 4:
            raise ValueError("occurrence_id must be a UUID4")
        async with self._condition:
            self._ensure_available()
            if self._pending_occurrences.get(job_id) != occurrence_id:
                return False
            pending = dict(self._pending_occurrences)
            pending.pop(job_id)
            self._publish(self._jobs, pending)
            return True

    async def public_snapshot(self) -> tuple[ScheduleJob, ...]:
        async with self._condition:
            public_jobs = sorted(
                (job for job in self._jobs if job.source == "user"),
                key=lambda job: (job.created_at_ms, job.job_id),
            )
            return tuple(copy.deepcopy(job) for job in public_jobs)

    async def add_user_job(self, job: ScheduleJob) -> ScheduleJob:
        if job.source != "user":
            raise ValueError("public Schedule mutations can add only user Jobs")
        return await self._add_job(job)

    async def _register_system_job(self, job: ScheduleJob) -> ScheduleJob:
        _require_system_job(job)
        async with self._condition:
            self._ensure_available()
            current = next((item for item in self._jobs if item.job_id == job.job_id), None)
            if current is not None and current.source != "system":
                raise ScheduleStateError(self.path)
            reconciled = _reconcile_system_job(current, job)
            if reconciled is None:
                assert current is not None
                return copy.deepcopy(current)
            candidate = (
                (*self._jobs, reconciled)
                if current is None
                else tuple(reconciled if item.job_id == job.job_id else item for item in self._jobs)
            )
            self._publish(candidate)
            return copy.deepcopy(reconciled)

    async def _add_job(self, job: ScheduleJob) -> ScheduleJob:
        if not isinstance(job, ScheduleJob):
            raise TypeError("job must be a ScheduleJob")
        if job.created_at_ms != job.updated_at_ms:
            raise ValueError("new Schedule Job timestamps must be equal")
        if job.state != ScheduleJobState():
            raise ValueError("new Schedule Job state must be empty")
        async with self._condition:
            self._ensure_available()
            if any(existing.job_id == job.job_id for existing in self._jobs):
                raise ValueError("Schedule Job ID already exists")
            candidate = (*self._jobs, copy.deepcopy(job))
            self._publish(candidate)
            return copy.deepcopy(job)

    async def remove_user_job(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
    ) -> bool:
        return await self._remove(job_id, expected=expected, user_only=True)

    async def _remove_terminal_job(
        self,
        job_id: str,
        *,
        expected: ScheduleJob,
        occurrence_id: UUID | None = None,
    ) -> bool:
        require_uuid4_string(job_id, field="job_id")
        return await self._remove(
            job_id,
            expected=expected,
            user_only=False,
            occurrence_id=occurrence_id,
        )

    async def commit_terminal(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
        occurrence_id: UUID | None = None,
        finished_at_ms: int,
        status: JobStatus,
        error: str | None = None,
        now_ms: int | None = None,
    ) -> ScheduleJob | None:
        require_uuid4_string(job_id, field="job_id")
        return await self._commit_terminal(
            job_id,
            expected=expected,
            occurrence_id=occurrence_id,
            finished_at_ms=finished_at_ms,
            status=status,
            error=error,
            now_ms=now_ms,
            source="user",
        )

    async def _commit_system_terminal(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
        finished_at_ms: int,
        status: JobStatus,
        error: str | None = None,
        now_ms: int | None = None,
    ) -> ScheduleJob | None:
        if job_id != DREAM_JOB_ID:
            raise ValueError("only the reserved Dream system Job may use internal mutation")
        return await self._commit_terminal(
            job_id,
            expected=expected,
            occurrence_id=None,
            finished_at_ms=finished_at_ms,
            status=status,
            error=error,
            now_ms=now_ms,
            source="system",
        )

    async def _commit_terminal(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None,
        occurrence_id: UUID | None,
        finished_at_ms: int,
        status: JobStatus,
        error: str | None,
        now_ms: int | None,
        source: Literal["user", "system"],
    ) -> ScheduleJob | None:
        require_nonnegative_int(finished_at_ms, field="finished_at_ms")
        commit_now = finished_at_ms if now_ms is None else now_ms
        require_nonnegative_int(commit_now, field="now_ms")
        terminal_state = ScheduleJobState(
            last_finished_at_ms=finished_at_ms,
            last_status=status,
            last_error=error,
        )
        async with self._condition:
            self._ensure_available()
            current = next((job for job in self._jobs if job.job_id == job_id), None)
            if current is None or current.source != source:
                if current is not None and current.source != source:
                    raise ScheduleStateError(self.path)
                return None
            if expected is not None and current != expected:
                return None
            pending_occurrence_id = self._pending_occurrences.get(job_id)
            if pending_occurrence_id is not None and pending_occurrence_id != occurrence_id:
                return None
            updated = replace(
                current,
                state=terminal_state,
                updated_at_ms=max(current.updated_at_ms, commit_now),
            )
            candidate = tuple(updated if job.job_id == job_id else job for job in self._jobs)
            pending = dict(self._pending_occurrences)
            pending.pop(job_id, None)
            self._publish(candidate, pending)
            return copy.deepcopy(updated)

    async def wait_for_change(self, revision: int) -> int:
        require_nonnegative_int(revision, field="revision")
        async with self._condition:
            await self._condition.wait_for(lambda: self._revision != revision or self._faulted)
            return self._revision

    def _load_once(self) -> tuple[tuple[ScheduleJob, ...], dict[str, UUID]]:
        try:
            self.path.lstat()
        except FileNotFoundError:
            return (), {}
        owned_path = HOST_FILESYSTEM.require_owned_regular_file(
            self.path,
            within=self._state_root,
        )
        content = owned_path.read_text(encoding="utf-8")
        return _parse_document(content)

    async def _remove(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None,
        user_only: bool,
        occurrence_id: UUID | None = None,
    ) -> bool:
        if user_only:
            require_uuid4_string(job_id, field="job_id")
        else:
            _require_internal_job_id(job_id)
        async with self._condition:
            self._ensure_available()
            current = next((job for job in self._jobs if job.job_id == job_id), None)
            if current is None or (user_only and current.source != "user"):
                return False
            if expected is not None:
                if user_only:
                    if expected.source != "user" or _public_job_key(current) != _public_job_key(
                        expected
                    ):
                        raise ScheduleStaleRemovalError("Schedule Job changed before removal")
                elif current != expected:
                    return False
            pending_occurrence_id = self._pending_occurrences.get(job_id)
            if (
                not user_only
                and pending_occurrence_id is not None
                and pending_occurrence_id != occurrence_id
            ):
                return False
            candidate = tuple(job for job in self._jobs if job.job_id != job_id)
            pending = {
                pending_job_id: occurrence_id
                for pending_job_id, occurrence_id in self._pending_occurrences.items()
                if pending_job_id != job_id
            }
            self._publish(candidate, pending)
            return True

    def _ensure_available(self) -> None:
        if self._faulted:
            raise ScheduleStoreFaultedError("Schedule Store is faulted")

    def _publish(
        self,
        candidate: tuple[ScheduleJob, ...],
        pending_occurrences: dict[str, UUID] | None = None,
    ) -> None:
        pending = (
            self._pending_occurrences
            if pending_occurrences is None
            else pending_occurrences
        )
        try:
            encoded = _serialize_document(candidate, pending)
            if _parse_document(encoded) != (candidate, pending):
                raise ValueError("Schedule mutation did not produce canonical state")
            self._require_write_location()
            self._replace_text(self.path, encoded)
        except Exception:
            self._faulted = True
            self._condition.notify_all()
            raise
        self._jobs = copy.deepcopy(candidate)
        self._pending_occurrences = dict(pending)
        self._revision += 1
        self._condition.notify_all()

    def _require_write_location(self) -> None:
        workspace_root = self.workspace_state.workspace_path.resolve(strict=True)
        self._state_root = HOST_FILESYSTEM.require_owned_directory(
            self._state_path,
            within=workspace_root,
        )
        try:
            self.path.lstat()
        except FileNotFoundError:
            return
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self._state_root)


def _parse_document(content: str) -> tuple[tuple[ScheduleJob, ...], dict[str, UUID]]:
    try:
        loaded = strict_json_loads(content)
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as error:
        raise ValueError("Schedule state is not valid strict JSON") from error
    if isinstance(loaded, list):
        job_values = loaded
        pending_values: object = {}
    elif isinstance(loaded, dict) and set(loaded) == {
        "version",
        "jobs",
        "pending_occurrences",
    }:
        if loaded["version"] != 1:
            raise ValueError("Schedule state version is unsupported")
        job_values = loaded["jobs"]
        pending_values = loaded["pending_occurrences"]
    else:
        raise ValueError("Schedule state root is invalid")
    if not isinstance(job_values, list):
        raise ValueError("Schedule state Jobs must be a JSON array")
    jobs: list[ScheduleJob] = []
    seen: set[str] = set()
    for value in job_values:
        if not isinstance(value, dict):
            raise ValueError("Schedule state entries must be JSON objects")
        job = ScheduleJob.from_dict(value)
        if job.job_id in seen:
            raise ValueError("Schedule state contains duplicate Job IDs")
        seen.add(job.job_id)
        jobs.append(job)
    if [job.to_dict() for job in jobs] != job_values:
        raise ValueError("Schedule state contains non-canonical values")
    if not isinstance(pending_values, dict):
        raise ValueError("Schedule pending occurrence state must be an object")
    pending: dict[str, UUID] = {}
    job_sources = {job.job_id: job.source for job in jobs}
    for job_id, value in pending_values.items():
        if (
            not isinstance(job_id, str)
            or job_sources.get(job_id) != "user"
            or not isinstance(value, str)
        ):
            raise ValueError("Schedule pending occurrence entry is invalid")
        try:
            occurrence_id = UUID(value)
        except ValueError as error:
            raise ValueError("Schedule pending occurrence ID is invalid") from error
        if occurrence_id.version != 4 or str(occurrence_id) != value:
            raise ValueError("Schedule pending occurrence ID is not canonical UUID4")
        pending[job_id] = occurrence_id
    if not isinstance(loaded, list) and {
        "version": 1,
        "jobs": job_values,
        "pending_occurrences": {
            job_id: str(value) for job_id, value in pending.items()
        },
    } != loaded:
        raise ValueError("Schedule state contains non-canonical values")
    return tuple(jobs), pending


def _serialize_document(
    jobs: tuple[ScheduleJob, ...],
    pending_occurrences: dict[str, UUID],
) -> str:
    if not pending_occurrences:
        return json.dumps(
            [job.to_dict() for job in jobs],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return json.dumps(
        {
            "version": 1,
            "jobs": [job.to_dict() for job in jobs],
            "pending_occurrences": {
                job_id: str(occurrence_id)
                for job_id, occurrence_id in sorted(pending_occurrences.items())
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _public_job_key(job: ScheduleJob) -> tuple[object, ...]:
    return (job.job_id, job.title, job.message, tuple(job.schedule.to_dict().items()))


def _require_system_job(job: ScheduleJob) -> None:
    if not isinstance(job, ScheduleJob):
        raise TypeError("system Job must be a ScheduleJob")
    if job.source != "system" or job.job_id != DREAM_JOB_ID:
        raise ValueError("only the reserved Dream system Job may use internal mutation")


def _require_internal_job_id(job_id: str) -> None:
    if job_id == DREAM_JOB_ID:
        return
    require_uuid4_string(job_id, field="job_id")


def _reconcile_system_job(
    current: ScheduleJob | None,
    requested: ScheduleJob,
) -> ScheduleJob | None:
    if current is None:
        return requested
    if current.schedule == requested.schedule:
        return None
    updated_at_ms = (
        max(current.updated_at_ms, requested.updated_at_ms)
        if current.state.last_status is not None
        else current.created_at_ms
    )
    return replace(current, schedule=requested.schedule, updated_at_ms=updated_at_ms)


__all__ = [
    "ScheduleReservation",
    "ScheduleStaleRemovalError",
    "ScheduleStateError",
    "ScheduleStoreError",
    "ScheduleStoreFaultedError",
    "WorkspaceScheduleStore",
]
