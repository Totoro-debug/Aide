"""Process-level logging contract before any Conversation Session exists."""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from aide.agent.tools.mcp import MCPTool
from aide.agent.tools.mcp_runtime import MCPRuntimeManager, MCPWorkspaceRuntimeManager
from aide.client.cli.process_entry import run
from aide.config.config import MCPServerConfiguration
from aide.logging.process import configure_process_logging

if TYPE_CHECKING:
    from loguru import Record


@pytest.fixture(autouse=True)
def _remove_process_logging_handlers() -> Iterator[None]:
    logger.remove()
    logger.add(sys.stderr)
    yield
    logger.remove()


def test_process_logging_silences_records_below_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    logger.debug("debug detail")
    logger.info("ordinary progress")
    logger.warning("recoverable condition")

    assert capsys.readouterr().err == ""


def test_process_logging_emits_one_basic_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    try:
        raise RuntimeError("technical failure")
    except RuntimeError as error:
        logger.opt(exception=error).error("Startup failed")

    assert capsys.readouterr().err == "Startup failed\n"


def test_process_logging_emits_critical_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    logger.critical("Process cannot continue")

    assert capsys.readouterr().err == "Process cannot continue\n"


def test_process_logging_configuration_is_repeatable_without_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()
    configure_process_logging()

    logger.error("One diagnostic")

    assert capsys.readouterr().err == "One diagnostic\n"


@pytest.mark.asyncio
async def test_mcp_process_logging_exposes_only_sanitized_failure_metadata(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    records: list[Record] = []

    class FailingConnection:
        tools: tuple[MCPTool, ...] = ()
        unavailable = False

        async def connect(self) -> tuple[MCPTool, ...]:
            raise RuntimeError("raw-exception-secret")

        async def close(self) -> None:
            pass

    configurations = {
        "local": MCPServerConfiguration(
            mcp_name="local",
            enabled=True,
            transport="stdio",
            command="secret-command",
            args=("secret-argument",),
        ),
        "remote": MCPServerConfiguration(
            mcp_name="remote",
            enabled=True,
            transport="streamable-http",
            url="https://secret.example/mcp",
            headers={"Authorization": "Bearer header-secret"},
        ),
    }
    shared = MCPRuntimeManager(
        None,
        transport="streamable-http",
        connection_factory=lambda configuration, workspace: FailingConnection(),
    )
    manager = MCPWorkspaceRuntimeManager(
        tmp_path,
        shared_runtime=shared,
        connection_factory=lambda configuration, workspace: FailingConnection(),
    )
    configure_process_logging()
    capture_id = logger.add(
        lambda message: records.append(message.record),
        level="ERROR",
        format=lambda _record: "{message}\n",
        backtrace=False,
        diagnose=False,
    )
    try:
        await shared.start(configurations)
        report = await manager.start(configurations)
    finally:
        logger.remove(capture_id)
        await manager.close()
        await shared.close()

    process_output = capsys.readouterr().err
    assert sorted(process_output.splitlines()) == [
        "MCP Server failure mcp_name=local phase=connect type=RuntimeError",
        "MCP Server failure mcp_name=remote phase=connect type=RuntimeError",
    ]
    assert report.failed_servers == ("local", "remote")
    assert len(records) == 2
    for record in records:
        exception = record["exception"]
        assert exception is not None
        assert exception.type is RuntimeError
        assert exception.traceback is not None
    for forbidden in (
        "secret-command",
        "secret-argument",
        "https://secret.example/mcp",
        "Authorization",
        "Bearer header-secret",
        "raw-exception-secret",
    ):
        assert forbidden not in process_output
        assert all(forbidden not in record["message"] for record in records)


def test_process_entry_configures_logging_on_eager_help_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["aide", "--help"])

    with pytest.raises(SystemExit) as exited:
        run()
    logger.warning("must remain silent")

    assert exited.value.code == 0
    assert capsys.readouterr().err == ""
