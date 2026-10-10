"""Minimal process entry that configures logging before application imports."""

import sys

from aide.logging.process import configure_process_logging
from aide.utils.platform import WINDOWS_REQUIRED_ERROR, is_windows_host


def run() -> None:
    """Configure process diagnostics, then invoke the command-line application."""
    if not is_windows_host():
        print(WINDOWS_REQUIRED_ERROR, file=sys.stderr)
        raise SystemExit(1)
    configure_process_logging()

    from aide.client.cli.cli import app

    app()
