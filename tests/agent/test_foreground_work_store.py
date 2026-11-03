from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from aide.agent.foreground_work.options import ForegroundQuestionOption
from aide.agent.foreground_work.store import (
    ForegroundWorkConflict,
    ForegroundWorkError,
    ForegroundWorkStore,
    StaleForegroundWorkRevision,
)
from aide.agent.subagents.models import (
    SubAgentCreatorSnapshot,
    SubAgentError,
    SubAgentSource,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.subagents.store import SubAgentRecordStore
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


@pytest.mark.parametrize(
    "options",
    [
        [{"id": "duplicate", "label": "One"}, {"id": "duplicate", "label": "Two"}],
        [{"id": " padded", "label": "Label"}],
        [{"id": "valid", "label": " "}],
        [{"id": "valid", "label": "x" * 201}],
        [{"id": "valid", "label": "Label", "extra": "field"}],
        [{"id": "valid", "label": "Label"}] * 5,
    ],
)
def test_prepare_question_rejects_invalid_options(
    tmp_path: Path, options: list[dict[str, str]]
) -> None:
    store = _store(tmp_path)
    store.begin_task("task-1", "execution-1", generation=1)

    with pytest.raises(ValueError):
        store.prepare_question(
            task_id="task-1",
            execution_id="execution-1",
            question_id="question-1",
            tool_call_id="call-1",
            text="Which directory should I use?",
            options=cast(list[ForegroundQuestionOption], options),
        )

    assert store.load().questions == ()


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


def test_malformed_delivery_state_is_reported_without_reset(tmp_path: Path) -> None:
    from aide.agent.foreground_work.store import ForegroundWorkError

    store = _store(tmp_path)
    store.begin_task(str(uuid4()), str(uuid4()), generation=1)
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["deliveries"] = [
        {
            "agent_id": str(uuid4()),
            "terminal_record_revision": 1,
            "task_id": str(uuid4()),
            "task_generation": 1,
            "state": [],
            "claimed_execution_id": None,
            "receipt_id": None,
        }
    ]
    store.path.write_text(json.dumps(state), encoding="utf-8")
    before = store.path.read_bytes()

    with pytest.raises(ForegroundWorkError):
        _store(tmp_path).load()
    assert store.path.read_bytes() == before


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
    legacy.pop("deliveries", None)
    for question in legacy["questions"]:
        question.pop("owner_agent_id", None)
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
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 6
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


def test_interrupting_one_child_question_keeps_other_question_waitable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "execution", generation=1)
    store.prepare_question(
        task_id="task",
        execution_id="child-execution-1",
        question_id="question-1",
        tool_call_id="call-1",
        text="First question",
        owner_agent_id="agent-1",
    )
    store.publish_question("question-1", committed_tool_call_ids={"call-1"})
    store.prepare_question(
        task_id="task",
        execution_id="child-execution-2",
        question_id="question-2",
        tool_call_id="call-2",
        text="Second question",
        owner_agent_id="agent-2",
    )
    store.publish_question("question-2", committed_tool_call_ids={"call-2"})
    store.suspend_for_child_question("task", generation=1, execution_id="execution")

    snapshot = store.interrupt_subagent_question("question-1")

    first, second = snapshot.question("question-1"), snapshot.question("question-2")
    assert first is not None and first.state == "interrupted"
    assert second is not None and second.state == "pending"
    assert snapshot.tasks[0].state == "waiting_for_answer"


def test_foreground_child_can_ask_after_main_segment_completes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "main", generation=1)
    store.complete_task("task", "main")
    store.prepare_question(task_id="task", execution_id="child", question_id="question",
                           tool_call_id="call", text="Which directory?", owner_agent_id="agent")
    snapshot = store.publish_question("question", committed_tool_call_ids={"call"})
    question = snapshot.question("question")
    assert question is not None and question.state == "pending"
    assert snapshot.tasks[0].state == "waiting_for_answer"


@pytest.mark.parametrize("status", [SubAgentStatus.COMPLETED, SubAgentStatus.FAILED, SubAgentStatus.CANCELLED, SubAgentStatus.INTERRUPTED])
def test_persisted_terminal_result_is_discovered_once_by_its_durable_identity(
    tmp_path: Path,
    status: SubAgentStatus,
) -> None:
    task_id = str(uuid4())
    first_execution_id = str(uuid4())
    store = _store(tmp_path)
    store.begin_task(task_id, first_execution_id, generation=1)
    store.complete_task(task_id, first_execution_id)
    repository = SubAgentRecordStore(store.workspace_state, store.session_id)
    now = datetime(2026, 11, 3, tzinfo=UTC)
    record = repository.register(
        title="One-time delegated task",
        task="Return one result.",
        parent_run_id=str(uuid4()),
        source=SubAgentSource(
            kind=SubAgentSourceKind.FOREGROUND,
            restore_run_token=str(uuid4()),
            task_id=task_id,
            task_generation=1,
        ),
        creator_snapshot=SubAgentCreatorSnapshot(
            provider_id="test-provider",
            model="test-model",
            reasoning_effort="mid",
            permission_level="workspace-write",
            shell="pwsh",
            tool_names=("read_file",),
            system_prompt="You are Aide.",
        ),
    )
    running = repository.save(
        replace(
            record,
            status=SubAgentStatus.RUNNING,
            started_at=now,
            revision=record.revision + 1,
        )
    )
    terminal = repository.save(
        replace(
            running,
            status=status,
            error=SubAgentError("model_failed", "Child failed") if status in {SubAgentStatus.FAILED, SubAgentStatus.INTERRUPTED} else None,
            finished_at=now,
            conversation=({"role": "assistant", "content": "Finished."},),
            result="The durable result.",
            revision=running.revision + 1,
        )
    )

    if status in {SubAgentStatus.CANCELLED, SubAgentStatus.INTERRUPTED}:
        assert not store.reconcile([], subagents=(terminal,)).deliveries
        return

    # Attention is a nonterminal observation, even when the child has since finished.
    attention = {"role": "tool", "name": "wait_agent", "tool_call_id": "attention",
                 "content": json.dumps([{"agent_id": terminal.agent_id,
                                          "status": "waiting_for_answer"}])}
    assert store.reconcile([attention], subagents=(terminal,)).deliveries[0].state == "pending"
    stale_result = {**attention, "content": json.dumps([{
        "agent_id": terminal.agent_id, "status": status.value,
        "terminal_record_revision": terminal.revision - 1,
    }])}
    assert store.reconcile([stale_result], subagents=(terminal,)).deliveries[0].state == "pending"
    snapshot = store.reconcile([], subagents=(terminal,))
    for _ in range(99):
        snapshot = _store(tmp_path).reconcile([], subagents=(terminal,))

    assert len(snapshot.deliveries) == 1
    delivery = snapshot.deliveries[0]
    assert delivery.agent_id == terminal.agent_id
    assert delivery.terminal_record_revision == terminal.revision
    assert delivery.task_id == task_id
    assert delivery.task_generation == 1
    assert delivery.state == "pending"

    execution_id = str(uuid4())
    claimed = store.claim_delivery(
        agent_id=terminal.agent_id,
        terminal_record_revision=terminal.revision,
        task_generation=1,
        execution_id=execution_id,
    )
    assert claimed is not None and claimed.state == "claimed"
    assert claimed.claimed_execution_id == execution_id
    assert _store(tmp_path).claim_delivery(
        agent_id=terminal.agent_id,
        terminal_record_revision=terminal.revision,
        task_generation=1,
        execution_id=str(uuid4()),
    ) is None
    persisted = _store(tmp_path).load()
    assert len(persisted.deliveries) == 1
    assert persisted.deliveries[0].state == "claimed"
    committed = store.commit_delivery(
        agent_id=terminal.agent_id,
        terminal_record_revision=terminal.revision,
        task_generation=1,
        execution_id=execution_id,
        receipt_id=claimed.receipt_id,
    )
    assert committed is not None and committed.state == "committed"

    wait_result = _store(tmp_path).reconcile(
        [
            {
                "role": "tool",
                "name": "wait_agent",
                "tool_call_id": str(uuid4()),
                "content": json.dumps(
                    [
                        {
                            "agent_id": terminal.agent_id,
                            "status": status.value,
                            "terminal_record_revision": terminal.revision,
                            "result": terminal.result,
                        }
                    ]
                ),
            }
        ],
        subagents=(terminal,),
    )
    delivered = wait_result.deliveries[0]
    assert delivered.state == "committed"
    assert delivered.receipt_id is not None


@pytest.mark.parametrize("mutation", ["missing_version", "future_version", "unknown_field"])
def test_delivery_schema_rejects_damaged_or_future_state(tmp_path: Path, mutation: str) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "execution", generation=1)
    value = json.loads(store.path.read_text(encoding="utf-8"))
    if mutation == "missing_version":
        del value["schema_version"]
    elif mutation == "future_version":
        value["schema_version"] = 999
    else:
        value["unknown"] = True
    store.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ForegroundWorkError):
        store.load()

@pytest.mark.parametrize("receipts", [
    [{"generation": 2, "model_calls": 1}],
    [{"model_calls": 1}, {"model_calls": 2}],
    [{"iteration_stage": 0, "model_calls": 1}],
])
def test_uncertain_execution_receipts_never_complete_or_charge_task(
    tmp_path: Path, receipts: list[dict[str, int]],
) -> None:
    store = _store(tmp_path)
    store.begin_task("task", "execution", generation=1)
    messages = []
    for values in receipts:
        marker = {"task_id": "task", "generation": 1, "execution_id": "execution",
                  "commit_id": "task:1:execution", "iteration_stage": 1, **values}
        messages.append({"role": "assistant", "status": "completed",
                         "foreground_execution_commit": marker})
    recovered = store.reconcile(messages)
    assert recovered.tasks[0].state == "interrupted"
    assert store.remaining_iteration_budget("task", generation=1) == 50
    assert store.reconcile(messages) == recovered
