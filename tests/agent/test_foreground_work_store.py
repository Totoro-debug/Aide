from __future__ import annotations

from pathlib import Path

import pytest

from aide.agent.foreground_work.store import (
    ForegroundWorkConflict,
    ForegroundWorkStore,
    StaleForegroundWorkRevision,
)
from aide.agent.workspace_state import WorkspaceState


def _store(tmp_path: Path) -> ForegroundWorkStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "agent-home")
    return ForegroundWorkStore(state, "20261102-010203-123456_11111111-1111-4111-8111-111111111111")


def test_question_answer_survives_reload_and_same_answer_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    question = store.prepare_question(
        task_id="task-1",
        execution_id="execution-1",
        question_id="question-1",
        tool_call_id="call-1",
        text="Which directory should I use?",
    )
    store.publish_question("question-1", committed_tool_call_ids={"call-1"})

    answer = store.answer_question(
        "question-1",
        "Use the reports directory.",
        expected_revision=question.revision + 1,
        request_id="answer-request-1",
    )
    duplicate = _store(tmp_path).answer_question(
        "question-1",
        "Use the reports directory.",
        expected_revision=answer.revision,
        request_id="answer-retry-1",
    )

    assert duplicate.revision == answer.revision
    assert duplicate.questions[0].answer == "Use the reports directory."
    assert duplicate.questions[0].state == "answer_received"


def test_stale_answer_and_reused_request_with_different_payload_are_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    store.prepare_question(
        task_id="task-1",
        execution_id="execution-1",
        question_id="question-1",
        tool_call_id="call-1",
        text="Which directory should I use?",
    )
    pending = store.publish_question("question-1", committed_tool_call_ids={"call-1"})

    with pytest.raises(StaleForegroundWorkRevision):
        store.answer_question(
            "question-1",
            "reports",
            expected_revision=pending.revision - 1,
            request_id="answer-request-1",
        )

    accepted = store.answer_question(
        "question-1",
        "reports",
        expected_revision=pending.revision,
        request_id="answer-request-1",
    )
    with pytest.raises(ForegroundWorkConflict):
        _store(tmp_path).answer_question(
            "question-1",
            "archives",
            expected_revision=accepted.revision,
            request_id="answer-request-1",
        )


def test_failed_publication_preserves_previous_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    original = store.begin_task("task-1", "execution-1", generation=1)

    def fail_publish(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(
        "aide.agent.foreground_work.store.HOST_FILESYSTEM.atomic_replace_text", fail_publish
    )
    with pytest.raises(OSError, match="injected atomic publication failure"):
        store.prepare_question(
            task_id="task-1",
            execution_id="execution-1",
            question_id="question-1",
            tool_call_id="call-1",
            text="Which directory should I use?",
        )

    reloaded = _store(tmp_path).load()
    assert reloaded.revision == original.revision
    assert reloaded.questions == ()


def test_restart_and_restore_reconcile_question_and_answer_anchor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    store.prepare_question(
        task_id="task-1",
        execution_id="execution-1",
        question_id="question-1",
        tool_call_id="call-1",
        text="Which directory should I use?",
    )
    pending = store.reconcile(
        [{"role": "tool", "tool_call_id": "call-1", "foreground_question_id": "question-1"}]
    )
    pending_question = pending.question("question-1")
    assert pending_question is not None
    assert pending_question.state == "pending"

    answer = store.answer_question(
        "question-1",
        "reports",
        expected_revision=pending.revision,
        request_id="answer-1",
    )
    store.begin_resume("question-1", "execution-2")
    answered = store.reconcile(
        [
            {"role": "tool", "tool_call_id": "call-1", "foreground_question_id": "question-1"},
            {
                "role": "user",
                "foreground_question_id": "question-1",
                "restore_anchor_id": 2,
            },
        ]
    )
    answered_question = answered.question("question-1")
    assert answered_question is not None
    assert answered_question.state == "answered"
    assert answered_question.answer_anchor_id == 2

    restored_before_answer = store.reconcile(
        [{"role": "tool", "tool_call_id": "call-1", "foreground_question_id": "question-1"}]
    )
    restored_question = restored_before_answer.question("question-1")
    assert restored_question is not None
    assert restored_question.state == "pending"
    assert restored_question.answer is None

    restored_before_question = store.reconcile([])
    invalidated_question = restored_before_question.question("question-1")
    assert invalidated_question is not None
    assert invalidated_question.state == "invalidated"
    answer_question = answer.question("question-1")
    assert answer_question is not None
    assert answer_question.answer == "reports"


@pytest.mark.parametrize(
    "content", ['{"schema_version":999,"revision":1,"tasks":[],"questions":[]}', "{broken"]
)
def test_unsupported_or_corrupt_state_is_reported_without_reset(
    tmp_path: Path, content: str
) -> None:
    from aide.agent.foreground_work.store import ForegroundWorkError

    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    store.path.write_text(content, encoding="utf-8")
    with pytest.raises(ForegroundWorkError):
        store.load()
    assert store.path.read_text(encoding="utf-8") == content
