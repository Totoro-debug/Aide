from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from textual.widgets import Input, OptionList, Static

from aide.agent.loop import ForegroundConversationProjection
from aide.agent.message_bus import MessageBus
from aide.agent.session.execution_state import SessionRunState
from aide.agent.session.session import RestoreAnchor, Session
from aide.agent.workspace_state import WorkspaceState
from aide.client.cli.conversation import TerminalConversationApp
from aide.management.commands import (
    ManagementCommandDispatcher,
    ManagementPort,
    format_restore_preview,
)
from aide.management.service import (
    RestoreListingReport,
)

SESSION_ID = "20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000"
ANCHOR_TOKEN = UUID("550e8400-e29b-41d4-a716-446655440001")
AfterRestorePhase = Callable[[str, Callable[[], None]], None]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", ""),
        ("  first\n\tsecond  ", "first second"),
        ("x" * 96, "x" * 96),
        ("x" * 97, "x" * 93 + "..."),
        ("\u4e2d" * 97, "\u4e2d" * 93 + "..."),
    ],
)
def test_restore_preview_boundaries(content: str, expected: str) -> None:
    assert format_restore_preview(content) == expected


@pytest.mark.asyncio
async def test_restore_command_lists_persisted_anchors_without_exposing_restore_state() -> None:
    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=SESSION_ID,
                anchors=(
                    RestoreAnchor(
                        anchor_id=7,
                        run_token=ANCHOR_TOKEN,
                        content="Restore this input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

    result = await ManagementCommandDispatcher(cast(ManagementPort, Management())).dispatch(
        "/restore"
    )

    assert result.handled is True
    assert result.restore_listing is not None
    assert result.restore_listing.session_id == SESSION_ID
    assert [anchor.anchor_id for anchor in result.restore_listing.anchors] == [7]
    assert (
        result.output == "Restore anchors:\n7. 2026-09-26T12:00:00.000+08:00 | Restore this input"
    )
    assert "restore_before" not in (result.output or "")


@pytest.mark.asyncio
async def test_restore_waiter_cancellation_does_not_cancel_title_work(workspace: Path) -> None:
    title_started = asyncio.Event()
    title_finished = asyncio.Event()
    pending_persist_called = False

    async def resolve_title(_content: str) -> tuple[str, dict[str, int] | None]:
        title_started.set()
        await title_finished.wait()
        return "Generated title", None

    async def persist() -> None:
        nonlocal pending_persist_called
        pending_persist_called = True

    run_state = SessionRunState(uuid4())
    session = Session.create(WorkspaceState(workspace))
    work = run_state.start_title(session, "input", resolve_title, lambda: False)
    assert work is not None
    work.coordination.prepared.set_result(False)
    work.coordination.preparation_started.set()
    await asyncio.wait_for(title_started.wait(), timeout=1)
    waiter = asyncio.create_task(run_state.wait_for_title_idle(persist))
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not work.task.done()
    assert pending_persist_called is False
    title_finished.set()
    await work.task
    await run_state.wait_for_title_idle(persist)
    assert pending_persist_called is True


@pytest.mark.asyncio
async def test_terminal_rejects_ordinary_input_while_restore_barrier_is_held() -> None:
    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return False

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(session_id=SESSION_ID, messages=())

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(ManagementPort, object())),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("ordinary input"), "enter")
        await pilot.pause()

        assert not app._pending_inputs
        assert await bus.inbound_snapshot() == ()
        assert any(
            "restore_in_progress" in str(cast(Static, row).content)
            for row in app.query(".management-row")
        )


@pytest.mark.asyncio
async def test_restore_anchor_picker_shows_local_preview_and_cancels_without_mutation() -> None:
    cancel_calls = 0

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=SESSION_ID,
                anchors=(
                    RestoreAnchor(
                        anchor_id=1,
                        run_token=ANCHOR_TOKEN,
                        content="Restore\nthis input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return cancel_calls == 1

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(session_id=SESSION_ID, messages=())

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(ManagementPort, Management())),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/restore"), "enter")
        async with asyncio.timeout(1):
            while app.screen.id != "restore-anchor-picker":
                await pilot.pause()

        picker_text = "\n".join(
            str(option.prompt)
            for option in app.screen.query_one("#restore-anchor-options", OptionList).options
        )
        assert "1." in picker_text
        assert "Restore this input" in picker_text
        assert (
            datetime.fromisoformat("2026-09-26T12:00:00.000+08:00")
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
            in picker_text
        )
        assert app.screen.focused is app.screen.query_one("#restore-anchor-filter", Input)

        await pilot.click(offset=(1, 1))
        assert app.screen.id == "restore-anchor-picker"

        await pilot.press("escape")
        await pilot.pause()
        assert cancel_calls == 1
        assert app.screen.id != "restore-anchor-picker"

        await pilot.press(*list("ordinary input"), "enter")
        await pilot.pause()

        assert tuple(message.content for message in await bus.inbound_snapshot()) == (
            "ordinary input",
        )
