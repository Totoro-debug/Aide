"""Conservative same-response scheduling for local Tool calls."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from aide.agent.tools.base import ToolResourceProfile
from aide.agent.tools.permission import ToolInvocationFacts

type ScheduledToolAccess = tuple[ToolResourceProfile | None, ToolInvocationFacts | None]


def available_process_cpu_count() -> int:
    """Return the CPU count available to this process where the host exposes it."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        count = process_cpu_count()
        if isinstance(count, int) and count > 0:
            return count

    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            count = len(get_affinity(0))
        except OSError:
            count = 0
        if count > 0:
            return count

    if sys.platform == "win32":
        try:
            import ctypes
        except ImportError:
            pass
        else:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_affinity = ctypes.c_size_t()
            system_affinity = ctypes.c_size_t()
            if kernel32.GetProcessAffinityMask(
                kernel32.GetCurrentProcess(),
                ctypes.byref(process_affinity),
                ctypes.byref(system_affinity),
            ):
                count = process_affinity.value.bit_count()
                if count > 0:
                    return count

    return max(1, os.cpu_count() or 1)


def max_tool_call_concurrency(cpu_count: int | None = None) -> int:
    """Compute the independently applied limit for one model response."""
    count = available_process_cpu_count() if cpu_count is None else cpu_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("cpu_count must be a positive integer")
    return max(10, count // 4)


def next_tool_call_batch(
    profiles: Sequence[ToolResourceProfile | None | ScheduledToolAccess], *, limit: int
) -> tuple[int, ...]:
    """Select the next prefix of calls that can safely execute at the same time."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not profiles:
        return ()

    first = _scheduled_access(profiles[0])
    if first is None:
        return (0,)

    batch = [0]
    for index, item in enumerate(profiles[1:limit], start=1):
        candidate = _scheduled_access(item)
        if candidate is None or any(
            _accesses_conflict(candidate, existing)
            for existing in (first, *(_scheduled_access(profiles[prior]) for prior in batch[1:]))
            if existing is not None
        ):
            break
        batch.append(index)
    return tuple(batch)


def _scheduled_access(
    item: ToolResourceProfile | None | ScheduledToolAccess,
) -> ScheduledToolAccess | None:
    if isinstance(item, tuple):
        if len(item) != 2:
            return None
        profile, facts = item
    else:
        profile, facts = item, None
    if not _is_parallel_access(profile, facts):
        return None
    return profile, facts


def _is_parallel_access(
    profile: ToolResourceProfile | None,
    facts: ToolInvocationFacts | None,
) -> bool:
    if profile is None or profile.exclusive:
        return False
    if facts is not None and facts.schedule_action is not None:
        return True
    if profile.reads and not profile.writes:
        return True
    return facts is not None and bool(facts.file_accesses)


def _accesses_conflict(first: ScheduledToolAccess, second: ScheduledToolAccess) -> bool:
    first_profile, first_facts = first
    second_profile, second_facts = second
    if first_profile is None or second_profile is None:
        return True
    if first_profile.exclusive or second_profile.exclusive:
        return True

    first_schedule = None if first_facts is None else first_facts.schedule_action
    second_schedule = None if second_facts is None else second_facts.schedule_action
    if first_schedule is not None and second_schedule is not None:
        return first_schedule.action != "list" or second_schedule.action != "list"
    if first_schedule is not None or second_schedule is not None:
        schedule, other_profile, other_facts = (
            (first_schedule, second_profile, second_facts)
            if first_schedule is not None
            else (second_schedule, first_profile, first_facts)
        )
        if schedule is not None and other_profile.reads and not other_profile.writes:
            return False
        if other_facts is not None and other_facts.file_accesses:
            return False
        return True

    if first_profile.reads and not first_profile.writes and second_profile.reads and not second_profile.writes:
        return False
    if first_facts is None or second_facts is None:
        return True
    first_accesses = first_facts.file_accesses
    second_accesses = second_facts.file_accesses
    if not first_accesses or not second_accesses:
        return True
    for left in first_accesses:
        for right in second_accesses:
            if left.role == "read" and right.role == "read":
                continue
            if _paths_overlap(left.path, right.path):
                return True
    return False


def _paths_overlap(first: Path, second: Path) -> bool:
    first_path = os.path.normcase(os.path.normpath(str(first)))
    second_path = os.path.normcase(os.path.normpath(str(second)))
    try:
        return os.path.commonpath((first_path, second_path)) in {first_path, second_path}
    except ValueError:
        return False


__all__ = ["available_process_cpu_count", "max_tool_call_concurrency", "next_tool_call_batch"]
