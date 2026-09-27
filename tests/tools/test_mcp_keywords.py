import asyncio
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import myclaw.config.config as config_module
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, MCPServerConfiguration
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelResponse,
    ModelUsage,
)
from myclaw.tools.mcp.keywords import MCPKeywordPreparer
from myclaw.tools.mcp.tool import MCPTool, MCPToolSpec


class _UnusedMCPSession:
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        raise AssertionError("Keyword preparation must not call the MCP session")


_UNUSED_MCP_SESSION = _UnusedMCPSession()


def _tool(
    *,
    server_name: str = "github",
    remote_name: str = "search_issues",
    model_name: str = "mcp_github_search_issues",
    description: str = "Search GitHub issues.",
    parameters: dict[str, Any] | None = None,
) -> MCPTool:
    return MCPTool(
        MCPToolSpec(
            server_name=server_name,
            remote_name=remote_name,
            model_name=model_name,
            description=description,
            parameters=(
                parameters if parameters is not None else {"type": "object", "properties": {}}
            ),
        ),
        _UNUSED_MCP_SESSION,
    )


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


class FakeRouter:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"route": route, "messages": messages, "tools": tools})
        return _response(self.responses.pop(0))


def _server(
    *,
    server_name: str = "github",
    keywords: dict[str, list[str]] | None = None,
) -> MCPServerConfiguration:
    return MCPServerConfiguration(
        mcp_name=server_name,
        enabled=True,
        transport="stdio",
        command="server",
        tool_keywords={name: tuple(values) for name, values in (keywords or {}).items()},
    )


def _config_loader(tmp_path: Path) -> ConfigLoader:
    return ConfigLoader(AgentHome(tmp_path))


def test_preparer_interface_contains_only_production_inputs() -> None:
    constructor = inspect.signature(MCPKeywordPreparer.__init__).parameters
    prepare = inspect.signature(MCPKeywordPreparer.prepare).parameters

    assert tuple(constructor) == ("self", "model_router", "config_loader")
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in tuple(constructor.values())[1:]
    )
    assert tuple(prepare) == ("self", "tools", "servers")


@pytest.mark.asyncio
async def test_preparer_prefers_trimmed_configured_keywords_without_model_call(
    tmp_path: Path,
) -> None:
    router = FakeRouter([json.dumps(["should-not-be-used"])])
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    result = await preparer.prepare(
        (_tool(),),
        {"github": _server(keywords={"search_issues": [" issue ", "", "issue", "github"]})},
    )

    assert result == {"mcp_github_search_issues": ("issue", "github")}
    assert router.calls == []


@pytest.mark.asyncio
async def test_preparer_generates_valid_keywords_and_keeps_model_input_tool_free(
    tmp_path: Path,
) -> None:
    tool = _tool()
    router = FakeRouter(['["issues", "search"]'])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
        encoding="utf-8",
    )
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    result = await preparer.prepare((tool,), {"github": _server()})

    assert result == {"mcp_github_search_issues": ("issues", "search")}
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["route"] == "chat"
    assert call["tools"] == ()
    messages = call["messages"]
    assert isinstance(messages, Sequence)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "search_issues" in serialized
    assert "Search GitHub issues." in serialized
    assert "input_schema" in serialized

    saved = config_path.read_text(encoding="utf-8")
    assert "tool_keywords" in saved
    assert "issues" in saved
    assert "search" in saved


@pytest.mark.asyncio
async def test_preparer_reuses_saved_keywords_in_a_fresh_process(tmp_path: Path) -> None:
    tool = _tool()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
        encoding="utf-8",
    )
    first_router = FakeRouter(['["issues", "search"]'])

    first = await MCPKeywordPreparer(first_router, _config_loader(tmp_path)).prepare(
        (tool,), {"github": _server()}
    )
    saved_configuration = ConfigLoader(AgentHome(tmp_path)).load()
    second_router = FakeRouter(['["must-not-run"]'])
    second = await MCPKeywordPreparer(second_router, _config_loader(tmp_path)).prepare(
        (tool,), saved_configuration.mcp
    )

    assert first == second == {"mcp_github_search_issues": ("issues", "search")}
    assert len(first_router.calls) == 1
    assert second_router.calls == []


@pytest.mark.asyncio
async def test_preparer_caches_the_later_user_value_selected_during_persistence(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
        encoding="utf-8",
    )

    class UserEditingRouter(FakeRouter):
        async def complete(
            self,
            route: str,
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
        ) -> ModelResponse:
            response = await super().complete(route, messages=messages, tools=tools)
            config_path.write_text(
                """# Keep the user's edit.
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
[mcp.servers.github.tool_keywords]
search_issues = ["user", "configured"]
""",
                encoding="utf-8",
            )
            return response

    router = UserEditingRouter(['["generated"]'])
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    first = await preparer.prepare((_tool(),), {"github": _server()})
    second = await preparer.prepare((_tool(),), {"github": _server()})

    expected = {"mcp_github_search_issues": ("user", "configured")}
    assert first == second == expected
    assert len(router.calls) == 1
    assert "# Keep the user's edit." in config_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_preparer_falls_back_to_remote_name_without_persisting_and_caches_it(
    tmp_path: Path,
) -> None:
    tool = _tool()
    router = FakeRouter(["not-json"])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
        encoding="utf-8",
    )
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    first = await preparer.prepare((tool,), {"github": _server()})
    second = await preparer.prepare((tool,), {"github": _server()})

    assert first == second == {"mcp_github_search_issues": ("search_issues",)}
    assert len(router.calls) == 1
    assert "tool_keywords" not in config_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "{}",
        "[]",
        '["搜索"]',
        '["123"]',
        '["search", 1]',
    ],
)
async def test_preparer_uses_only_the_remote_name_for_invalid_generated_output(
    content: str,
    tmp_path: Path,
) -> None:
    router = FakeRouter([content])
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    result = await preparer.prepare((_tool(),), {"github": _server()})

    assert result == {"mcp_github_search_issues": ("search_issues",)}


@pytest.mark.asyncio
async def test_preparer_retries_after_fingerprint_change_within_one_process(
    tmp_path: Path,
) -> None:
    router = FakeRouter(['["issues"]', '["pulls"]'])
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))

    first = await preparer.prepare(
        (_tool(description="Search GitHub issues."),),
        {"github": _server()},
    )
    second = await preparer.prepare(
        (_tool(description="Search GitHub pull requests."),),
        {"github": _server()},
    )

    assert first == {"mcp_github_search_issues": ("issues",)}
    assert second == {"mcp_github_search_issues": ("pulls",)}
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_preparer_isolates_same_remote_name_between_servers(tmp_path: Path) -> None:
    router = FakeRouter(['["github"]', '["gitlab"]'])
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path))
    tools = (
        _tool(),
        _tool(server_name="gitlab", model_name="mcp_gitlab_search_issues"),
    )

    first = await preparer.prepare(
        tools,
        {
            "github": _server(),
            "gitlab": _server(server_name="gitlab"),
        },
    )
    second = await preparer.prepare(
        tools,
        {
            "github": _server(),
            "gitlab": _server(server_name="gitlab"),
        },
    )

    assert (
        first
        == second
        == {
            "mcp_github_search_issues": ("github",),
            "mcp_gitlab_search_issues": ("gitlab",),
        }
    )
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_preparer_keeps_other_results_when_one_generation_fails(tmp_path: Path) -> None:
    class PartiallyFailingRouter(FakeRouter):
        async def complete(
            self,
            route: str,
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
        ) -> ModelResponse:
            call: dict[str, object] = {
                "route": route,
                "messages": messages,
                "tools": tools,
            }
            self.calls.append(call)
            payload = json.loads(cast(str, messages[-1]["content"]))
            if payload["name"] == "broken":
                raise RuntimeError("injected model failure")
            return _response('["working"]')

    router = PartiallyFailingRouter([])
    result = await MCPKeywordPreparer(router, _config_loader(tmp_path)).prepare(
        (
            _tool(remote_name="broken", model_name="mcp_github_broken"),
            _tool(remote_name="working", model_name="mcp_github_working"),
        ),
        {"github": _server()},
    )

    assert result == {
        "mcp_github_broken": ("broken",),
        "mcp_github_working": ("working",),
    }
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_preparer_propagates_cancellation_and_drains_sibling_tasks(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    active = 0

    class BlockingRouter(FakeRouter):
        async def complete(
            self,
            route: str,
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
        ) -> ModelResponse:
            del route, messages, tools
            nonlocal active
            active += 1
            started.set()
            try:
                return await asyncio.Future[ModelResponse]()
            finally:
                active -= 1

    preparer = MCPKeywordPreparer(BlockingRouter([]), _config_loader(tmp_path))
    task = asyncio.create_task(
        preparer.prepare(
            (
                _tool(remote_name="one", model_name="mcp_github_one"),
                _tool(remote_name="two", model_name="mcp_github_two"),
            ),
            {"github": _server()},
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_stage",
    ["lock", "read", "parse", "validation", "serialization", "replace"],
)
async def test_preparer_keeps_generated_keywords_in_memory_when_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
) -> None:
    router = FakeRouter(['["issues"]'])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp.servers.github]
enabled = true
transport = "stdio"
command = "server"
""",
        encoding="utf-8",
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {failure_stage} failure")

    if failure_stage == "lock":
        monkeypatch.setattr(
            cast(Any, config_module).HOST_FILESYSTEM,
            "exclusive_lock",
            fail,
        )
    elif failure_stage == "read":
        original_read_text = Path.read_text

        def fail_config_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == config_path:
                fail()
            return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", fail_config_read)
    elif failure_stage == "parse":
        monkeypatch.setattr(cast(Any, config_module).tomlkit, "parse", fail)
    elif failure_stage == "validation":
        monkeypatch.setattr(cast(Any, config_module), "_parse_configuration", fail)
    elif failure_stage == "serialization":
        monkeypatch.setattr(cast(Any, config_module).tomlkit, "dumps", fail)
    else:
        monkeypatch.setattr(
            cast(Any, config_module).HOST_FILESYSTEM,
            "atomic_replace_text",
            fail,
        )
    result = await MCPKeywordPreparer(router, _config_loader(tmp_path)).prepare(
        (_tool(),), {"github": _server()}
    )

    assert result == {"mcp_github_search_issues": ("issues",)}
    assert b"tool_keywords" not in config_path.read_bytes()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_preparer_limits_generation_concurrency_to_four(tmp_path: Path) -> None:
    tools = tuple(
        _tool(
            remote_name=f"tool_{index}",
            model_name=f"mcp_github_tool_{index}",
        )
        for index in range(9)
    )
    router = FakeRouter([f'["keyword{index}"]' for index in range(9)])
    active = 0
    peak = 0
    original_complete = router.complete

    async def tracked_complete(
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        response = await original_complete(route, messages=messages, tools=tools)
        active -= 1
        return response

    router.complete = tracked_complete  # type: ignore[method-assign]
    preparer = MCPKeywordPreparer(router, _config_loader(tmp_path / "missing"))

    result = await preparer.prepare(tools, {"github": _server()})

    assert peak <= 4
    assert len(router.calls) == len(tools)
    assert len(result) == 9
