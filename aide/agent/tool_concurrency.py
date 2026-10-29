"""Conservative same-response scheduling for local Tool calls."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from aide.agent.tools.base import ToolResourceProfile


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
    profiles: Sequence[ToolResourceProfile | None], *, limit: int
) -> tuple[int, ...]:
    """Select the next prefix of calls that can safely execute at the same time."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not profiles:
        return ()

    first = profiles[0]
    if not _is_parallel_read_only(first):
        return (0,)

    batch = [0]
    for index, profile in enumerate(profiles[1:limit], start=1):
        if not _is_parallel_read_only(profile):
            break
        batch.append(index)
    return tuple(batch)


def _is_parallel_read_only(profile: ToolResourceProfile | None) -> bool:
    return (
        profile is not None
        and profile.reads
        and not profile.writes
        and not profile.exclusive
    )


__all__ = ["available_process_cpu_count", "max_tool_call_concurrency", "next_tool_call_batch"]
