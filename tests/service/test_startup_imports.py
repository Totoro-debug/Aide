"""Exercise startup imports in fresh processes, without timing gates on CI hosts."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest


def _fresh_process(source: str, *arguments: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", source, *arguments],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


@pytest.mark.parametrize("command", ["help", "web"])
def test_web_and_help_do_not_load_terminal_or_execution(command: str) -> None:
    result = _fresh_process(
        """
import json, sys
from aide.client.cli import cli
from typer.testing import CliRunner
if sys.argv[1] == 'web':
    from aide.service.client import ServiceClient
    class Client:
        async def create_web_ticket(self):
            return 'http://127.0.0.1/#ticket=fixture'
        async def close(self):
            pass
    async def connect(*args, **kwargs):
        return Client()
    ServiceClient.connect_or_start = connect
    cli.webbrowser.open_new_tab = lambda url: True
result = CliRunner().invoke(cli.app, ['--help'] if sys.argv[1] == 'help' else ['web'])
assert result.exit_code == 0, result.exception
print(json.dumps({'loaded': [name for name in ('textual.app', 'aide.agent.loop',
    'aide.agent.tools.tool_gateway', 'mcp', 'anthropic') if name in sys.modules]}))
""",
        command,
    )
    assert result["loaded"] == []


def test_conversation_contracts_keep_the_existing_type_identity() -> None:
    from aide.agent import conversation_contracts, loop

    assert (
        loop.ForegroundConversationProjection
        is conversation_contracts.ForegroundConversationProjection
    )
    assert (
        loop.TerminalAgentRunExecutorControl
        is conversation_contracts.TerminalAgentRunExecutorControl
    )


@pytest.mark.parametrize("with_dream_result", [False, True])
def test_terminal_and_management_results_do_not_load_execution(with_dream_result: bool) -> None:
    result = _fresh_process(
        """
import json, sys
from aide.client.cli.conversation import TerminalConversationApp
from aide.management.service import RuntimeStatus
from aide.service.client import _management_result
status = RuntimeStatus(
    version='fixture', chat_model='small', chat_reasoning_effort='mid', uptime_seconds=0,
    context_window=8192, max_output=1024, available_context=7168, compact_ratio=0.9,
    compact_context_window=6452, projected_next_request_tokens=0, projection_source='estimated',
    input_budget_used_percent=0, session_message_count=0, last_compacted=0, cumulative_usage={},
)
dream = {'status': 'completed', 'processed_count': 1, 'memory_updated': True, 'cursor': 1}
decoded = _management_result({'handled': True, 'status_view': status.to_dict(),
    'dream_result': dream if sys.argv[1] == 'True' else None, 'restore_result': None})
assert decoded.status_view == status and decoded.restore_result is None
assert (decoded.dream_result is not None) is (sys.argv[1] == 'True')
print(json.dumps({'loaded': [name for name in ('aide.agent.loop',
    'aide.agent.context.run_context', 'aide.provider.model_router',
    'aide.agent.tools.tool_gateway', 'aide.agent.memory.dream') if name in sys.modules]}))
""",
        str(with_dream_result),
    )
    assert result["loaded"] == []


def test_dream_result_keeps_the_existing_type_identity() -> None:
    from aide.agent.memory import contracts, dream

    assert dream.DreamResult is contracts.DreamResult


def test_fresh_terminal_import_in_worker_runs_textual_on_main_thread() -> None:
    result = _fresh_process(
        """
import asyncio, json, threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from aide.client.cli import cli
from aide.agent.confirmation import ToolConfirmationCoordinator
from aide.agent.message_bus import MessageBus
from aide.config.agent_home import AgentHome
from aide.service.client import ServiceClient
main_thread = threading.get_ident()
original_load = cli._load_terminal_conversation
observed = {}
def load():
    observed['import_in_worker'] = threading.get_ident() != main_thread
    app_type, _check = original_load()
    original_init = app_type.__init__
    def initialize(self, **kwargs):
        assert threading.get_ident() == main_thread
        original_init(self, **kwargs)
    async def run(self):
        assert threading.get_ident() == main_thread
        async with self.run_test(size=(100, 30)) as pilot:
            widget = self.query_one('#conversation-input')
            assert self.focused is widget and not widget.disabled
            await pilot.press('o', 'k')
            assert widget.text == 'ok'
            observed['interactive'] = True
    app_type.__init__ = initialize
    app_type.run_async = run
    return app_type, lambda: True
async def management(*args):
    from aide.management.commands import ManagementCommandResult
    return ManagementCommandResult(handled=True, output=None)
client = SimpleNamespace(bus=MessageBus(), control=SimpleNamespace(has_active_run=False),
    confirmation=ToolConfirmationCoordinator(),
    management_dispatcher=SimpleNamespace(dispatch=management, restore_result=management),
    get_startup_config=AsyncMock(return_value={'startup': {'available': True}}),
    attach_workspace=AsyncMock(), close=AsyncMock(),
    get_input_capabilities=AsyncMock(return_value={'skill_metadata': (), 'management_commands': ()}))
async def connect(*args, **kwargs):
    return client
ServiceClient.connect_or_start = connect
cli._load_terminal_conversation = load
asyncio.run(cli._run_service_cli_conversation(agent_home=AgentHome(Path.cwd()), workspace=Path.cwd()))
client.close.assert_awaited_once()
print(json.dumps(observed))
"""
    )
    assert result == {"import_in_worker": True, "interactive": True}


def _service_startup(tmp_path: Path, protocol: str, mcp_configuration: str = "") -> dict[str, object]:
    (tmp_path / "config.toml").write_text(
        f'''[models.providers.local]
protocol = "{protocol}"
base_url = "http://127.0.0.1:1/v1"
api_key = "fixture"
[models.providers.local.models.small]
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 30
[models.routes.chat]
provider_id = "local"
model = "small"
{mcp_configuration}
''',
        encoding="utf-8",
    )
    return _fresh_process(
        """
import asyncio, json, sys
from pathlib import Path
from aide.config.agent_home import AgentHome
from aide.service.runtime import AgentService
async def main():
    service = AgentService(AgentHome(Path(sys.argv[1])))
    try:
        await service.start()
        assert service.state == 'ready' and service.configuration is not None
        print(json.dumps({'anthropic': 'anthropic' in sys.modules,
            'mcp': any(name == 'mcp' or name.startswith('mcp.') or
                name == 'mcp_types' or name.startswith('mcp_types.') for name in sys.modules)}))
    finally:
        await service.stop()
asyncio.run(main())
""",
        str(tmp_path),
    )


@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
def test_service_prepares_only_configured_provider_adapters(tmp_path: Path, protocol: str) -> None:
    result = _service_startup(tmp_path, protocol)
    assert result["anthropic"] is (protocol == "anthropic")


@pytest.mark.parametrize(
    "mcp_configuration",
    [
        "",
        '''[mcp.servers.disabled]
enabled = false
transport = "streamable-http"
url = "http://127.0.0.1:1/mcp"
''',
        '''[mcp.servers.disabled]
enabled = false
transport = "stdio"
command = "nonexistent-disabled-server"
''',
    ],
)
def test_service_without_enabled_mcp_does_not_load_sdk(
    tmp_path: Path, mcp_configuration: str,
) -> None:
    result = _service_startup(tmp_path, "openai-compatible", mcp_configuration)
    assert result["mcp"] is False
