from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Static, TextArea

from aide.client.cli.conversation import TerminalConversationApp, _cli_question_answer
from tests.agent.test_fixed_catalog import _FixedCatalogProvider
from tests.service.test_foreground_questions import _QuestionProvider
from tests.terminal.test_conversation import (
    _run_cli_terminal_case,
    _visible_screen_text,
    _wait_for_turn,
)


def test_cli_question_answer_is_explicit_and_preserves_ordinary_input() -> None:
    assert _cli_question_answer("/answer reports-directory") == "reports-directory"
    assert _cli_question_answer("/answer\nreports directory") == "reports directory"
    assert _cli_question_answer("use /answer in the README") is None
    assert _cli_question_answer("ordinary input") is None


@pytest.mark.asyncio
async def test_cli_question_and_answer_through_live_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _QuestionProvider()
    provider.release_answer.set()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "ask-foreground-question"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while "Which output directory" not in str(
                app.query_one("#foreground-question", Static).content
            ):
                await pilot.pause()
        assert "Which output directory" in _visible_screen_text(app)
        input_area.text = "ordinary independent supplement"
        await pilot.press("enter")
        await _wait_for_turn(app)
        input_area.text = "/answer reports-directory"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while str(app.query_one("#foreground-question", Static).content):
                await pilot.pause()
        assert provider.answer_started.is_set()
        assert provider.question_calls == 1
        assert sum("reports-directory" in text for text in provider.user_texts) == 1
        assert not (workspace / "after-question.txt").exists()

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home",
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=cast(_FixedCatalogProvider, provider),
        size=(120, 40),
    )
