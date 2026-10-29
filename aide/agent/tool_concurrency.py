"""Conservative same-response scheduling for local Tool calls."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

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

    accesses: list[ScheduledToolAccess | None] = []
    for item in profiles[:limit]:
        if isinstance(item, tuple):
            if len(item) != 2:
                accesses.append(None)
                continue
            profile, facts = item
        else:
            profile, facts = item, None
        if (
            profile is None
            or profile.exclusive
            or not (
                (facts is not None and facts.schedule_action is not None)
                or (profile.reads and not profile.writes)
                or (facts is not None and bool(facts.file_accesses))
            )
        ):
            accesses.append(None)
        else:
            accesses.append((profile, facts))

    first = accesses[0]
    if first is None:
        return (0,)

    batch = [0]
    batch_accesses = [first]
    for index, candidate in enumerate(accesses[1:], start=1):
        if candidate is None:
            break
        candidate_profile, candidate_facts = candidate
        if candidate_profile is None:
            break

        conflict = False
        for prior_profile, prior_facts in batch_accesses:
            if prior_profile is None:
                conflict = True
                break

            prior_schedule = None if prior_facts is None else prior_facts.schedule_action
            candidate_schedule = (
                None if candidate_facts is None else candidate_facts.schedule_action
            )
            if prior_schedule is not None and candidate_schedule is not None:
                conflict = prior_schedule.action != "list" or candidate_schedule.action != "list"
            elif prior_schedule is not None or candidate_schedule is not None:
                other_profile, other_facts = (
                    (candidate_profile, candidate_facts)
                    if prior_schedule is not None
                    else (prior_profile, prior_facts)
                )
                conflict = not (
                    (other_profile.reads and not other_profile.writes)
                    or (other_facts is not None and bool(other_facts.file_accesses))
                )
            elif (
                prior_profile.reads
                and not prior_profile.writes
                and candidate_profile.reads
                and not candidate_profile.writes
            ):
                conflict = False
            elif prior_facts is None or candidate_facts is None:
                conflict = True
            else:
                prior_accesses = prior_facts.file_accesses
                candidate_accesses = candidate_facts.file_accesses
                conflict = not prior_accesses or not candidate_accesses
                if not conflict:
                    for left in prior_accesses:
                        for right in candidate_accesses:
                            if left.role != "read" or right.role != "read":
                                left_path = os.path.normcase(os.path.normpath(str(left.path)))
                                right_path = os.path.normcase(os.path.normpath(str(right.path)))
                                try:
                                    overlaps = os.path.commonpath((left_path, right_path)) in {
                                        left_path,
                                        right_path,
                                    }
                                except ValueError:
                                    overlaps = False
                                if overlaps:
                                    conflict = True
                                    break
                        if conflict:
                            break
            if conflict:
                break
        if conflict:
            break
        batch.append(index)
        batch_accesses.append(candidate)
    return tuple(batch)


__all__ = ["available_process_cpu_count", "max_tool_call_concurrency", "next_tool_call_batch"]
