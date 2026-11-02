from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Static, TextArea

from aide.client.cli.conversation import (
    TerminalConversationApp,
    _cli_question_answer,
    _cli_task_command,
)
from tests.agent.test_fixed_catalog import _FixedCatalogProvider
from tests.service.test_foreground_questions import _QuestionProvider
from tests.terminal.test_conversation import (
    _run_cli_terminal_case,
    _visible_screen_text,
    _wait_for_turn,
)


async def _wait_for_question(app: TerminalConversationApp, pilot: Pilot[None], text: str) -> Static:
    async with asyncio.timeout(10):
        while True:
            question = next(
                (widget for widget in app.query(Static) if widget.id == "foreground-question"),
                None,
            )
            if question is not None and text in str(question.content):
                return question
            await pilot.pause()


def test_cli_question_answer_is_explicit_and_preserves_ordinary_input() -> None:
    assert _cli_question_answer("/answer reports-directory") == ("text", "reports-directory")
    assert _cli_question_answer("/answer\nreports directory") == ("text", "reports directory")
    assert _cli_question_answer("/answer --option reports") == ("choice", "reports")
    assert _cli_question_answer("/answer --text --option reports") == ("text", "--option reports")
    assert _cli_question_answer("/answer --option") == ("choice", "")
    assert _cli_question_answer("use /answer in the README") is None
    assert _cli_question_answer("ordinary input") is None


def test_cli_task_command_requires_an_explicit_task_id_for_actions() -> None:
    assert _cli_task_command("/task") == ("list", None)
    assert _cli_task_command("/task list") == ("list", None)
    assert _cli_task_command("/task cancel task-123") == ("cancel", "task-123")
    assert _cli_task_command("/task continue task-123") == ("continue", "task-123")
    assert _cli_task_command("/task cancel") == ("invalid", None)
    assert _cli_task_command("ordinary /task text") is None


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


@pytest.mark.asyncio
async def test_cli_question_options_keep_choice_and_text_answers_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _QuestionProvider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "ask-foreground-options"
        await pilot.press("enter")
        await _wait_for_turn(app)
        question_widget = await _wait_for_question(app, pilot, "Reports folder")
        assert "Reports folder" in str(question_widget.content)
        assert "reports" in str(question_widget.content)
        assert "Archives folder" in str(question_widget.content)
        assert "archives" in str(question_widget.content)

        input_area.text = "ordinary independent supplement"
        await pilot.press("enter")
        await _wait_for_turn(app)
        assert "Reports folder" in str(question_widget.content)

        input_area.text = "/answer --option reports"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while str(question_widget.content):
                await pilot.pause()
        assert "Reports folder" in provider.user_texts[-1]
        assert "--option reports" not in provider.user_texts[-1]

        input_area.text = "ask-foreground-question"
        await pilot.press("enter")
        await _wait_for_turn(app)
        question_widget = await _wait_for_question(app, pilot, "Which output directory")
        assert "Which output directory" in str(question_widget.content)
        input_area.text = "/answer --text --option custom free-text answer"
        await pilot.press("enter")
        await _wait_for_turn(app)
        assert any("--option custom free-text answer" in text for text in provider.user_texts)
        assert provider.question_calls == 2

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home",
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=cast(_FixedCatalogProvider, provider),
        size=(120, 40),
    )


@pytest.mark.asyncio
async def test_cli_task_cancel_invalidates_its_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _QuestionProvider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "ask-foreground-question"
        await pilot.press("enter")
        await _wait_for_turn(app)
        await _wait_for_question(app, pilot, "Which output directory")
        async with asyncio.timeout(10):
            while not app._foreground_tasks:
                await pilot.pause()
        task_id = str(app._foreground_tasks[-1]["task_id"])

        input_area.text = "/task cancel " + task_id
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while app._foreground_question is not None:
                await pilot.pause()

        assert app._foreground_tasks[-1]["state"] == "cancelled"
        assert "cancelled" in str(app.query_one("#foreground-tasks", Static).content)
        assert provider.question_calls == 1
        assert not (workspace / "after-question.txt").exists()

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home",
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=cast(_FixedCatalogProvider, provider),
        size=(120, 40),
    )


@pytest.mark.asyncio
async def test_cli_task_continue_is_an_explicit_user_operation(
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
        await _wait_for_question(app, pilot, "Which output directory")
        task_id = str(app._foreground_tasks[-1]["task_id"])
        input_area.text = "/answer reports-directory"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while app._foreground_tasks[-1]["state"] != "completed":
                await pilot.pause()

        input_area.text = "/task continue " + task_id
        await pilot.press("enter")
        await _wait_for_turn(app)

        assert "Continue the current foreground task." in provider.user_texts[-1]
        assert app._foreground_tasks[-1]["task_id"] == task_id
        assert app._foreground_tasks[-1]["generation"] == 1

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home",
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=scenario,
        provider=cast(_FixedCatalogProvider, provider),
        size=(120, 40),
    )

@pytest.mark.asyncio
async def test_cli_ordinary_task_is_visible_and_can_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _QuestionProvider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "ordinary task without a question"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while not app._foreground_tasks:
                await pilot.pause()
        widget = app.query_one("#foreground-tasks", Static)
        assert "completed" in str(widget.content)
        task_id = str(app._foreground_tasks[-1]["task_id"])
        input_area.text = "/task continue " + task_id
        await pilot.press("enter")
        await _wait_for_turn(app)
        assert "Continue the current foreground task." in provider.user_texts[-1]

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home", workspace=workspace, monkeypatch=monkeypatch,
        scenario=scenario, provider=cast(_FixedCatalogProvider, provider), size=(120, 40),
    )


@pytest.mark.asyncio
async def test_cli_answers_two_subagents_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    import tomlkit

    from tests.configuration.test_config import VALID_CONFIG
    from tests.service.test_subagent_management import _DelegatingProvider
    from tests.terminal import test_conversation as terminal_fixture
    document = cast(Any, tomlkit.parse(VALID_CONFIG))
    document["models"]["routes"]["subagent"] = dict(document["models"]["routes"]["chat"])
    monkeypatch.setattr(terminal_fixture, "VALID_CONFIG", tomlkit.dumps(document))
    provider = _DelegatingProvider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "SubAgent question delegation request"
        await pilot.press("enter")
        await _wait_for_turn(app)
        widget = await _wait_for_question(app, pilot, "Which output should I prepare?")
        assert app._foreground_question is not None
        first_id = app._foreground_question["question_id"]
        input_area.text = "/answer first output"
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while not provider.child_answer_seen.is_set():
                await pilot.pause()
        async with asyncio.timeout(10):
            while app._foreground_question is None or app._foreground_question["question_id"] == first_id:
                await pilot.pause()
        provider.child_answer_seen.clear()
        input_area.text = "/answer second output"
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while not provider.child_answer_seen.is_set() or str(widget.content):
                await pilot.pause()
        assert provider.child_question_count == 4

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home", workspace=workspace, monkeypatch=monkeypatch,
        scenario=scenario, provider=cast(_FixedCatalogProvider, provider), size=(120, 40),
    )
