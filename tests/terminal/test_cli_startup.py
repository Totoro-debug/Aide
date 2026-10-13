"""Startup scheduling and ownership at the CLI/service boundary."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from aide.client.cli import cli, conversation
from aide.config.agent_home import AgentHome
from aide.service.client import ServiceClient, ServiceStartupError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "connect_failure",
        "import_failure",
        "both_fail",
        "cancel",
        "cancel_connect",
    ],
)
async def test_terminal_preparation_overlaps_connection_and_is_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    main_thread = get_ident()
    importing = Event()
    release_import = Event()
    import_finished = Event()
    connected = asyncio.Event()
    calls: list[str] = []
    failure = ServiceStartupError("service_unavailable", "Controlled connection failure")
    import_failure = ImportError("Controlled terminal import failure")
    client = SimpleNamespace(
        bus=object(),
        control=object(),
        management_dispatcher=object(),
        confirmation=object(),
        close=AsyncMock(),
        attach_workspace=AsyncMock(),
        get_startup_config=AsyncMock(return_value={"startup": {"available": True}}),
        get_input_capabilities=AsyncMock(
            return_value={
                "skill_metadata": (),
                "management_commands": (),
            }
        ),
    )

    class App:
        def __init__(self, **kwargs: Any) -> None:
            assert get_ident() == main_thread
            assert import_finished.is_set() and kwargs["bus"] is client.bus
            calls.append("construct")

        async def run_async(self) -> None:
            assert get_ident() == main_thread
            calls.append("run")

    def load() -> tuple[type[conversation.TerminalConversationApp], Callable[[], bool]]:
        assert get_ident() != main_thread
        importing.set()
        try:
            assert release_import.wait(timeout=5), "test did not release terminal import"
            if outcome in {"import_failure", "both_fail"}:
                raise import_failure
            return cast(type[conversation.TerminalConversationApp], App), lambda: True
        finally:
            import_finished.set()

    async def connect(*args: object, **kwargs: object) -> ServiceClient:
        assert kwargs["attach_workspace"] is False
        assert await asyncio.to_thread(importing.wait, 3), "terminal preparation did not overlap"
        connected.set()
        if outcome in {"connect_failure", "both_fail"}:
            raise failure
        if outcome == "cancel_connect":
            await asyncio.Event().wait()
        return cast(ServiceClient, client)

    monkeypatch.setattr(cli, "_load_terminal_conversation", load)
    monkeypatch.setattr(ServiceClient, "connect_or_start", connect)
    task = asyncio.create_task(
        cli._run_service_cli_conversation(
            agent_home=AgentHome(tmp_path),
            workspace=tmp_path,
        )
    )
    try:
        await asyncio.wait_for(connected.wait(), timeout=4)
        if outcome in {"cancel", "cancel_connect"}:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "startup returned while terminal preparation was still running"
    finally:
        release_import.set()
        if not connected.is_set():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    if outcome == "success":
        await asyncio.wait_for(task, timeout=3)
        assert calls == ["construct", "run"]
        client.attach_workspace.assert_awaited_once_with(tmp_path)
    else:
        expected = {
            "connect_failure": ServiceStartupError,
            "import_failure": ImportError,
            "both_fail": ServiceStartupError,
            "cancel": asyncio.CancelledError,
            "cancel_connect": asyncio.CancelledError,
        }[outcome]
        with pytest.raises(expected) as raised:
            await asyncio.wait_for(task, timeout=3)
        if outcome not in {"cancel", "cancel_connect"}:
            assert raised.value is (import_failure if outcome == "import_failure" else failure)
        assert calls == []
        client.attach_workspace.assert_not_awaited()
    assert import_finished.is_set()
    assert client.close.await_count == (
        0 if outcome in {"connect_failure", "both_fail", "cancel_connect"} else 1
    )
