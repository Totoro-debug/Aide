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
