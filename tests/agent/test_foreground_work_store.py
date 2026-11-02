from __future__ import annotations

import json
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


def test_task_cancellation_invalidates_question_and_fences_late_completion(
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

    cancelled = store.cancel_task(
        "task-1",
        expected_revision=pending.revision,
        request_id="cancel-1",
    )
    task = next(item for item in cancelled.tasks if item.task_id == "task-1")
    question = cancelled.question("question-1")
    assert task.state == "cancelled"
    assert task.generation == 2
    assert question is not None and question.state == "invalidated"

    reopened = _store(tmp_path)
    reconciled = reopened.reconcile([])
    durable_task = next(item for item in reconciled.tasks if item.task_id == "task-1")
    assert durable_task.state == "cancelled"
    assert durable_task.generation == 2
    assert reopened.complete_task("task-1", "execution-1") == reconciled

    duplicate = _store(tmp_path).cancel_task(
        "task-1",
        expected_revision=pending.revision,
        request_id="cancel-1",
    )
    assert duplicate == cancelled
    assert _store(tmp_path).complete_task("task-1", "execution-1") == cancelled

    with pytest.raises(ForegroundWorkConflict):
        _store(tmp_path).answer_question(
            "question-1",
            "reports",
            expected_revision=pending.revision,
            request_id="answer-1",
        )


def test_keep_preserves_task_identity_and_replace_invalidates_its_generation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    store.complete_task("task-1", "execution-1")

    kept = store.start_execution(
        "task-1",
        "execution-2",
        expected_generation=1,
    )
    kept_task = next(item for item in kept.tasks if item.task_id == "task-1")
    assert kept_task.state == "running"
    assert kept_task.execution_id == "execution-2"
    assert kept_task.generation == 1

    store.prepare_question(
        task_id="task-1",
        execution_id="execution-2",
        question_id="question-1",
        tool_call_id="call-1",
        text="Which directory should I use?",
    )
    store.publish_question("question-1", committed_tool_call_ids={"call-1"})
    replaced = store.replace_current_task("task-2", "execution-3")

    old_task = next(item for item in replaced.tasks if item.task_id == "task-1")
    new_task = next(item for item in replaced.tasks if item.task_id == "task-2")
    question = replaced.question("question-1")
    assert old_task.state == "superseded"
    assert old_task.generation == 2
    assert new_task.state == "running"
    assert question is not None and question.state == "invalidated"
    assert _store(tmp_path).complete_task("task-1", "execution-2") == replaced


def test_automatic_segments_share_stage_budget_until_explicit_user_operation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task(
        "task-1",
        "execution-1",
        generation=1,
        iteration_budget_limit=3,
    )
    store.record_iteration_usage(
        "task-1",
        generation=1,
        execution_id="execution-1",
        model_calls=2,
    )
    charged = _store(tmp_path).load()
    charged_task = next(item for item in charged.tasks if item.task_id == "task-1")
    assert charged_task.iteration_budget_used == 2
    assert charged_task.charged_execution_ids == ("execution-1",)
    assert _store(tmp_path).record_iteration_usage(
        "task-1",
        generation=1,
        execution_id="execution-1",
        model_calls=2,
    ) == charged

    automatic = store.start_execution(
        "task-1",
        "execution-2",
        expected_generation=1,
        explicit_user_operation=False,
    )
    continued = next(item for item in automatic.tasks if item.task_id == "task-1")
    assert continued.iteration_stage == 1
    assert continued.iteration_budget_used == 2
    assert store.remaining_iteration_budget("task-1", generation=1) == 1

    store.record_iteration_usage(
        "task-1",
        generation=1,
        execution_id="execution-2",
        model_calls=1,
    )
    assert store.remaining_iteration_budget("task-1", generation=1) == 0

    explicit = store.start_execution(
        "task-1",
        "execution-3",
        expected_generation=1,
        explicit_user_operation=True,
    )
    next_stage = next(item for item in explicit.tasks if item.task_id == "task-1")
    assert next_stage.iteration_stage == 2
    assert next_stage.iteration_budget_used == 0
    assert store.remaining_iteration_budget("task-1", generation=1) == 3


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


@pytest.mark.parametrize("answered", [False, True])
@pytest.mark.parametrize("legacy_version", [1, 2, 3])
def test_legacy_question_state_upgrades_on_write_and_preserves_answer_retry(
    tmp_path: Path, answered: bool, legacy_version: int
) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)
    store.prepare_question(
        task_id="task-1",
        execution_id="execution-1",
        question_id="question-1",
        tool_call_id="call-1",
        text="Which directory?",
    )
    pending = store.publish_question("question-1", committed_tool_call_ids={"call-1"})
    if answered:
        store.answer_question(
            "question-1", "reports", expected_revision=pending.revision, request_id="answer-1"
        )
    legacy = json.loads(store.path.read_text(encoding="utf-8"))
    legacy["schema_version"] = legacy_version
    for task in legacy["tasks"]:
        if legacy_version < 3:
            task.pop("permission_level", None)
        for field in (
            "iteration_stage",
            "iteration_budget_limit",
            "iteration_budget_used",
            "charged_execution_ids",
        ):
            task.pop(field, None)
    if legacy_version == 1:
        for question in legacy["questions"]:
            for field in ("options", "answer_kind", "answer_option_id"):
                del question[field]
    store.path.write_text(json.dumps(legacy), encoding="utf-8")
    before = store.path.read_bytes()
    loaded = _store(tmp_path).load()
    assert store.path.read_bytes() == before
    question = loaded.question("question-1")
    assert question is not None and question.options == ()
    assert question.answer_kind == ("text" if answered else None)
    accepted = store.answer_question(
        "question-1", "reports", expected_revision=pending.revision, request_id="answer-1"
    )
    if answered:
        assert accepted.revision == loaded.revision
    store.begin_resume("question-1", "execution-2")
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 4
    reloaded = _store(tmp_path).load().question("question-1")
    assert reloaded is not None and reloaded.answer == "reports"
    assert reloaded.answer_kind == "text"

def test_revoked_authority_cannot_admit_tool_or_delegation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "execution", generation=1)
    assert store.is_current_execution("task", 1, "execution")
    assert store.is_active_generation("task", 1)
    store.close()
    assert not store.is_current_execution("task", 1, "execution")
    assert not store.is_active_generation("task", 1)

def test_uncertain_framing_remains_interrupted_after_question_reconciliation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "execution", generation=1)
    store.prepare_question(task_id="task", execution_id="execution", question_id="question",
                           tool_call_id="call", text="Which directory?")
    store.publish_question("question", committed_tool_call_ids={"call"})
    store.interrupt_uncertain_tasks()
    snapshot = store.reconcile([
        {"role": "tool", "tool_call_id": "call", "foreground_question_id": "question"}
    ])
    assert snapshot.tasks[0].state == "interrupted"
    assert not store.is_active_generation("task", 1)
    assert snapshot.pending_questions[0].question_id == "question"
