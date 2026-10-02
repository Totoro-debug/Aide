"""Exercise an installed console entry with Textual's headless terminal adapter."""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from unittest.mock import patch

import myclaw
import myclaw.terminal.cli as cli
from myclaw.config.agent_home import AgentHome
from myclaw.service.discovery import read_discovery
from myclaw.terminal.conversation import TerminalConversationApp


async def headless_terminal(self: Any, **_kwargs: object) -> None:
    async with self.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert self.query_one("#conversation-input")
        assert read_discovery(AgentHome.production()) == before
        self.exit()


assert Path(myclaw.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert shutil.which("node") is None and shutil.which("npm") is None
before = read_discovery(AgentHome.production())
assert before is not None
assert before.service_instance_id == sys.argv[1]
assert before.pid == int(sys.argv[2])
# Replace only terminal I/O; retain the installed entry, CLI composition and client.
entry = next(iter(entry_points(group="console_scripts", name="myclaw")))
sys.argv = ["myclaw"]
with (
    patch.object(cli, "is_interactive_terminal", return_value=True),
    patch.object(TerminalConversationApp, "run_async", headless_terminal),
):
    try:
        entry.load()()
    except SystemExit as error:
        assert error.code in (None, 0), error.code
assert read_discovery(AgentHome.production()) == before
print(json.dumps({"marker": "INSTALLED_CLI_CONNECT_OK", **before.to_dict()}))
