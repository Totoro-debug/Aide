"""Loguru configuration for diagnostics without Session ownership."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


def _basic_terminal_format(_record: object) -> str:
    return "{message}\n"


def _without_session_ownership(record: Record) -> bool:
    return record["extra"].get("session_id") is None


def configure_process_logging() -> None:
    """Install the non-persistent logging baseline for this process."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="ERROR",
        filter=_without_session_ownership,
        format=_basic_terminal_format,
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
