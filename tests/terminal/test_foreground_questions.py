from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Static, TextArea

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
    _wait_for_session_picker,
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
async def test_cli_restarts_service_before_answering_and_continuing_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _QuestionProvider()
    provider.release_answer.set()
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_id: str | None = None
    calls_before_restart: tuple[int, int] | None = None

    async def create_interrupted_question(
        app: TerminalConversationApp, pilot: Pilot[None]
    ) -> None:
        nonlocal task_id, calls_before_restart
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "ask-foreground-question"
        await pilot.press("enter")
        await _wait_for_turn(app)
        await _wait_for_question(app, pilot, "Which output directory")

        task_text = str(app.query_one("#foreground-tasks", Static).content)
        match = re.search(r"Task ([^ ]+) · generation", task_text)
        assert match is not None
        task_id = match.group(1)
        provider.framing_actions.append("invalid")
        input_area.text = "independent supplement with uncertain task framing"
        await pilot.press("enter")
        await _wait_for_turn(app)
        async with asyncio.timeout(10):
            while "interrupted" not in str(app.query_one("#foreground-tasks", Static).content):
                await pilot.pause()
        calls_before_restart = (provider.stream_calls, provider.complete_calls)

    await _run_cli_terminal_case(
        agent_home=home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=create_interrupted_question,
        provider=cast(_FixedCatalogProvider, provider),
        size=(120, 40),
    )

    async def resume_and_continue(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        assert task_id is not None
        assert calls_before_restart is not None
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "/resume"
        await pilot.press("enter")
        await _wait_for_session_picker(app, pilot)
        options = app.screen.query_one("#session-picker-options", OptionList)
        selected_indices = [
            index
            for index, option in enumerate(options.options)
            if re.search(r"\b[1-9][0-9]* messages?\b", str(option.prompt))
        ]
        assert len(selected_indices) == 1
        selected = selected_indices[0]
        options.highlighted = selected
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while app.query_one("#conversation-input", TextArea).read_only:
                await pilot.pause()
        question_widget = await _wait_for_question(app, pilot, "Which output directory")
        assert "interrupted" in str(app.query_one("#foreground-tasks", Static).content)
        assert (provider.stream_calls, provider.complete_calls) == calls_before_restart

        async def wait_for_visible_response(previous_count: int) -> None:
            async with asyncio.timeout(10):
                while _visible_screen_text(app).count("The request is complete.") <= previous_count:
                    await pilot.pause()

        response_count = _visible_screen_text(app).count("The request is complete.")
        input_area.text = "/answer reports-directory"
        await pilot.press("enter")
        await asyncio.wait_for(provider.answer_started.wait(), timeout=10)
        await wait_for_visible_response(response_count)
        async with asyncio.timeout(10):
            while str(question_widget.content):
                await pilot.pause()
        assert "reports-directory" in provider.user_texts[-1]
        assert provider.question_calls == 1

        input_area.text = "/task continue " + task_id
        calls_before_continue = provider.stream_calls
        response_count = _visible_screen_text(app).count("The request is complete.")
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while provider.stream_calls == calls_before_continue:
                await pilot.pause()
        await wait_for_visible_response(response_count)
        assert "Continue the current foreground task." in provider.user_texts[-1]

    await _run_cli_terminal_case(
        agent_home=home,
        workspace=workspace,
        monkeypatch=monkeypatch,
        scenario=resume_and_continue,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("switch_session", [False, True])
async def test_cli_shows_automatic_continuation_without_user_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, switch_session: bool
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
        input_area.text = "initial delegation request"
        await pilot.press("enter")
        await _wait_for_turn(app)
        await asyncio.wait_for(provider.child_started.wait(), timeout=10)
        if switch_session:
            original_session = app._control.project_foreground_conversation().session_id
            from aide.agent.session.session import Session
            from aide.agent.workspace_state import WorkspaceState
            from aide.utils.time import local_now

            alternate = Session.create(WorkspaceState(workspace), now=local_now)
            alternate.update_metadata(title="Switch target")
            from uuid import uuid4

            alternate.commit_agent_run(
                [{"role": "user", "content": "Switch target seed."}],
                pending_last_compacted=0,
                pending_action_summary="",
                restore_before=alternate.capture_restore_before(),
                restore_run_token=uuid4(),
            )
            await alternate.wait_for_pending_persist()
            input_area.text = "/resume"
            await pilot.press("enter")
            await _wait_for_session_picker(app, pilot)
            app.screen.query_one("#session-picker-filter", Input).value = "Switch target"
            await pilot.pause()
            await pilot.press("enter")
            async with asyncio.timeout(10):
                while app._control.project_foreground_conversation().session_id == original_session:
                    await pilot.pause()
        provider.release_child.set()
        if switch_session:
            await asyncio.wait_for(provider.automatic_continuation_started.wait(), timeout=10)
            input_area.text = "/resume"
            await pilot.press("enter")
            await _wait_for_session_picker(app, pilot)
            options = app.screen.query_one("#session-picker-options", OptionList)
            options.highlighted = next(
                index for index, option in enumerate(options.options)
                if option.id == original_session
            )
            await pilot.press("enter")
            async with asyncio.timeout(10):
                while app._control.project_foreground_conversation().session_id != original_session:
                    await pilot.pause()
        async with asyncio.timeout(20):
            while "Main automatically handled the delegated result." not in _visible_screen_text(app):
                await pilot.pause()
        assert provider.automatic_continuation_count == 1
        if not switch_session:
            assert "Automatic continuation from a SubAgent result." in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home", workspace=workspace, monkeypatch=monkeypatch,
        scenario=scenario, provider=cast(_FixedCatalogProvider, provider), size=(120, 50),
    )


@pytest.mark.asyncio
async def test_cli_batches_terminal_subagent_results_in_one_automatic_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomlkit

    from tests.configuration.test_config import VALID_CONFIG
    from tests.service.test_subagent_management import _BatchDelegatingProvider
    from tests.terminal import test_conversation as terminal_fixture

    document = cast(Any, tomlkit.parse(VALID_CONFIG))
    document["models"]["routes"]["subagent"] = dict(document["models"]["routes"]["chat"])
    monkeypatch.setattr(terminal_fixture, "VALID_CONFIG", tomlkit.dumps(document))
    provider = _BatchDelegatingProvider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def scenario(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.text = "initial delegation request for batch delegation request"
        await pilot.press("enter")
        await asyncio.wait_for(provider.batch_parent_waiting.wait(), timeout=10)
        for child_started in provider.batch_child_waiting.values():
            await asyncio.wait_for(child_started.wait(), timeout=10)
        for release_child in provider.release_batch_children.values():
            release_child.set()
        provider.release_batch_parent.set()
        await _wait_for_turn(app)

        async with asyncio.timeout(20):
            while "Main automatically handled the delegated result." not in _visible_screen_text(app):
                await pilot.pause()
        assert provider.automatic_continuation_count == 1
        assert len(provider.batch_auto_requests) == 1
        assert all(
            f"Batch child result {index}." in provider.batch_auto_requests[0]
            for index in range(1, 4)
        )
        assert "Automatic continuation from a SubAgent result." in _visible_screen_text(app)

    await _run_cli_terminal_case(
        agent_home=tmp_path / "home", workspace=workspace, monkeypatch=monkeypatch,
        scenario=scenario, provider=cast(_FixedCatalogProvider, provider), size=(120, 50),
    )
