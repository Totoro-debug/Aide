"""Versioned durable coordination for foreground Main Agent questions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from aide.agent.permission import ToolPermissionLevel, validate_permission_level
from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.subagents.models import (
    SubAgentRecord,
    SubAgentSourceKind,
    SubAgentStatus,
)
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_SCHEMA_VERSION = 6
_MAX_QUESTION_LENGTH = 2_000
_MAX_ANSWER_LENGTH = 10_000
_SESSION_ID = re.compile(
    r"\d{8}-\d{6}-\d{6}_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
QuestionState = Literal[
    "preparing",
    "pending",
    "answer_received",
    "resuming",
    "executing",
    "answered",
    "interrupted",
    "invalidated",
]
AnswerKind = Literal["text", "choice"]
TaskState = Literal[
    "running",
    "waiting_for_answer",
    "completed",
    "interrupted",
    "cancelled",
    "superseded",
]
DeliveryState = Literal["pending", "claimed", "committed", "invalidated"]


class ForegroundQuestionOption(TypedDict):
    id: str
    label: str


class ForegroundWorkError(ValueError):
    """A persisted foreground-work record is malformed or unsupported."""


class ForegroundWorkConflict(ValueError):
    """An operation conflicts with the current foreground-work authority."""


class StaleForegroundWorkRevision(ForegroundWorkConflict):
    """An operation was based on an older foreground-work revision."""


@dataclass(frozen=True, slots=True)
class ForegroundWorkTask:
    task_id: str
    generation: int
    execution_id: str
    state: TaskState
    origin_anchor_id: int | None = None
    permission_level: ToolPermissionLevel = "read-only"
    iteration_stage: int = 1
    iteration_budget_limit: int = 50
    iteration_budget_used: int = 0
    charged_execution_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("iteration_stage", self.iteration_stage),
            ("iteration_budget_limit", self.iteration_budget_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.iteration_budget_used, bool)
            or not isinstance(self.iteration_budget_used, int)
            or self.iteration_budget_used < 0
        ):
            raise ValueError("iteration_budget_used must be a nonnegative integer")
        if not isinstance(self.charged_execution_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.charged_execution_ids
        ):
            raise ValueError("charged_execution_ids must contain nonempty strings")
        if len(set(self.charged_execution_ids)) != len(self.charged_execution_ids):
            raise ValueError("charged_execution_ids must be unique")


@dataclass(frozen=True, slots=True)
class ForegroundWorkQuestion:
    question_id: str
    task_id: str
    generation: int
    execution_id: str
    tool_call_id: str
    text: str
    state: QuestionState
    answer: str | None = None
    answer_request_id: str | None = None
    answer_fingerprint: str | None = None
    answer_kind: AnswerKind | None = None
    answer_option_id: str | None = None
    resume_execution_id: str | None = None
    answer_anchor_id: int | None = None
    options: tuple[ForegroundQuestionOption, ...] = ()
    owner_agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class ForegroundWorkDelivery:
    agent_id: str
    terminal_record_revision: int
    task_id: str
    task_generation: int
    state: DeliveryState
    claimed_execution_id: str | None = None
    receipt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ForegroundWorkSnapshot:
    revision: int
    tasks: tuple[ForegroundWorkTask, ...] = ()
    questions: tuple[ForegroundWorkQuestion, ...] = ()
    deliveries: tuple[ForegroundWorkDelivery, ...] = ()

    def question(self, question_id: str) -> ForegroundWorkQuestion | None:
        return next(
            (item for item in self.questions if item.question_id == question_id),
            None,
        )

    @property
    def pending_questions(self) -> tuple[ForegroundWorkQuestion, ...]:
        return tuple(item for item in self.questions if item.state == "pending")

    @property
    def pending_deliveries(self) -> tuple[ForegroundWorkDelivery, ...]:
        return tuple(item for item in self.deliveries if item.state == "pending")


class ForegroundWorkStore:
    """Atomically publish one Workspace-owned coordination file per Session."""

    def __init__(self, workspace_state: WorkspaceState, session_id: str) -> None:
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("Foreground Work Store requires WorkspaceState")
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Foreground Work Store requires a canonical foreground Session ID")
        Session._require_id(session_id, partition=SessionStoragePartition.FOREGROUND)
        self.workspace_state = workspace_state
        self.session_id = session_id
        self.path = workspace_state.foreground_work_directory / session_id / "state.json"
        self._closed = False

    def close(self) -> None:
        """Revoke writes from callbacks retaining an obsolete Session authority."""
        self._closed = True

    def load(self) -> ForegroundWorkSnapshot:
        """Read the current revision without creating storage for legacy Sessions."""
        root = self._existing_session_root()
        if root is None or not HOST_FILESYSTEM.entry_exists(self.path):
            return ForegroundWorkSnapshot(revision=0)
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
        try:
            raw: object = json.loads(_read_owned_text(self.path, within=root))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ForegroundWorkError("Foreground Work state is not valid UTF-8 JSON") from error
        return _snapshot_from_dict(raw)

    def begin_task(
        self,
        task_id: str,
        execution_id: str,
        *,
        generation: int,
        origin_anchor_id: int | None = None,
        permission_level: ToolPermissionLevel = "read-only",
        iteration_budget_limit: int = 50,
    ) -> ForegroundWorkSnapshot:
        _require_identifier(task_id, "task_id")
        _require_identifier(execution_id, "execution_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if origin_anchor_id is not None and (
            isinstance(origin_anchor_id, bool)
            or not isinstance(origin_anchor_id, int)
            or origin_anchor_id < 1
        ):
            raise ValueError("origin_anchor_id must be a positive integer")
        validate_permission_level(permission_level)
        _require_positive_integer(iteration_budget_limit, "iteration_budget_limit")
        current = self.load()
        existing = next((item for item in current.tasks if item.task_id == task_id), None)
        if existing is not None:
            if existing.execution_id == execution_id and existing.generation == generation:
                return current
            raise ForegroundWorkConflict("task_id is already bound to another execution")
        task = ForegroundWorkTask(
            task_id,
            generation,
            execution_id,
            "running",
            origin_anchor_id,
            permission_level,
            1,
            iteration_budget_limit,
        )
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, (*current.tasks, task), current.questions)
        )

    def start_execution(
        self,
        task_id: str,
        execution_id: str,
        *,
        expected_generation: int,
        explicit_user_operation: bool = True,
    ) -> ForegroundWorkSnapshot:
        """Start one user-owned execution while keeping a Task's valid generation."""
        _require_identifier(task_id, "task_id")
        _require_identifier(execution_id, "execution_id")
        if isinstance(expected_generation, bool) or not isinstance(expected_generation, int):
            raise ValueError("expected_generation must be an integer")
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None or task.generation != expected_generation:
            raise ForegroundWorkConflict("task generation is no longer current")
        if task.state in {"cancelled", "superseded"}:
            raise ForegroundWorkConflict("task is no longer active")
        if not isinstance(explicit_user_operation, bool):
            raise ValueError("explicit_user_operation must be a boolean")
        if task.execution_id == execution_id and task.state == "running":
            return current
        updated = ForegroundWorkTask(
            task.task_id,
            task.generation,
            execution_id,
            "running",
            task.origin_anchor_id,
            task.permission_level,
            task.iteration_stage + int(explicit_user_operation),
            task.iteration_budget_limit,
            0 if explicit_user_operation else task.iteration_budget_used,
            () if explicit_user_operation else task.charged_execution_ids,
        )
        tasks = (*[item for item in current.tasks if item.task_id != task_id], updated)
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, tasks, current.questions)
        )

    def claim_delivery(
        self,
        *,
        agent_id: str,
        terminal_record_revision: int,
        task_generation: int,
        execution_id: str,
        allow_reclaim: bool = False,
    ) -> ForegroundWorkDelivery | None:
        """Atomically claim one pending result for one Main execution segment."""
        _require_identifier(agent_id, "agent_id")
        _require_identifier(execution_id, "execution_id")
        if not isinstance(allow_reclaim, bool):
            raise ValueError("allow_reclaim must be a boolean")
        for name, value in (
            ("terminal_record_revision", terminal_record_revision),
            ("task_generation", task_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        current = self.load()
        key = (agent_id, terminal_record_revision, task_generation)
        delivery = next(
            (
                item
                for item in current.deliveries
                if (item.agent_id, item.terminal_record_revision, item.task_generation) == key
            ),
            None,
        )
        if delivery is None or (
            delivery.state != "pending"
            and not (allow_reclaim and delivery.state == "claimed")
        ):
            return None
        task = next((item for item in current.tasks if item.task_id == delivery.task_id), None)
        unresolved_question_states = {
            "preparing",
            "pending",
            "answer_received",
            "resuming",
            "executing",
            "interrupted",
        }
        if (
            task is None
            or task.generation != task_generation
            or task.state not in ({"running", "completed", "interrupted"} if allow_reclaim else {"running", "completed"})
            or (
                not allow_reclaim
                and task.iteration_budget_used >= task.iteration_budget_limit
            )
            or any(
                question.task_id == task.task_id
                and question.state in unresolved_question_states
                for question in current.questions
            )
        ):
            return None
        claimed = replace(
            delivery,
            state="claimed",
            claimed_execution_id=execution_id,
            receipt_id=_delivery_receipt_id(*key),
        )
        deliveries = tuple(claimed if item == delivery else item for item in current.deliveries)
        self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                current.tasks,
                current.questions,
                deliveries,
            )
        )
        return claimed

    def claim_delivery_batch(
        self,
        keys: Sequence[tuple[str, int, int]],
        *,
        execution_id: str,
    ) -> tuple[ForegroundWorkDelivery, ...] | None:
        """Atomically claim same-Task results and start their shared Main execution."""
        _require_identifier(execution_id, "execution_id")
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or not keys:
            raise ValueError("delivery batch must not be empty")
        normalized_keys: list[tuple[str, int, int]] = []
        for key in keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError("delivery batch keys must be three-item tuples")
            agent_id, terminal_record_revision, task_generation = key
            _require_identifier(agent_id, "agent_id")
            for name, value in (
                ("terminal_record_revision", terminal_record_revision),
                ("task_generation", task_generation),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            normalized_keys.append(key)
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("delivery batch keys must be unique")

        current = self.load()
        by_key = {
            (item.agent_id, item.terminal_record_revision, item.task_generation): item
            for item in current.deliveries
        }
        deliveries = tuple(by_key.get(key) for key in normalized_keys)
        if any(item is None or item.state != "pending" for item in deliveries):
            return None
        selected = cast(tuple[ForegroundWorkDelivery, ...], deliveries)
        task_id = selected[0].task_id
        task_generation = selected[0].task_generation
        if any(
            item.task_id != task_id or item.task_generation != task_generation
            for item in selected
        ):
            return None
        task = next(
            (
                item
                for item in current.tasks
                if item.task_id == task_id and item.generation == task_generation
            ),
            None,
        )
        unresolved_question_states = {
            "preparing",
            "pending",
            "answer_received",
            "resuming",
            "executing",
            "interrupted",
        }
        if (
            task is None
            or task.state not in {"running", "completed"}
            or task.iteration_budget_used >= task.iteration_budget_limit
            or any(
                question.task_id == task_id
                and question.state in unresolved_question_states
                for question in current.questions
            )
        ):
            return None

        claimed = tuple(
            replace(
                item,
                state="claimed",
                claimed_execution_id=execution_id,
                receipt_id=_delivery_receipt_id(
                    item.agent_id, item.terminal_record_revision, item.task_generation
                ),
            )
            for item in selected
        )
        claimed_by_key = {
            (item.agent_id, item.terminal_record_revision, item.task_generation): item
            for item in claimed
        }
        deliveries_after = tuple(
            claimed_by_key.get(
                (item.agent_id, item.terminal_record_revision, item.task_generation), item
            )
            for item in current.deliveries
        )
        started_task = replace(task, execution_id=execution_id, state="running")
        tasks_after = tuple(
            started_task if item.task_id == task_id and item.generation == task_generation else item
            for item in current.tasks
        )
        self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tasks_after,
                current.questions,
                deliveries_after,
            )
        )
        return claimed

    def commit_delivery(
        self,
        *,
        agent_id: str,
        terminal_record_revision: int,
        task_generation: int,
        execution_id: str,
        receipt_id: str | None,
    ) -> ForegroundWorkDelivery | None:
        """Commit a claimed delivery after its Session receipt is durable."""
        _require_identifier(agent_id, "agent_id")
        _require_identifier(execution_id, "execution_id")
        if receipt_id is not None:
            _require_identifier(receipt_id, "receipt_id")
        for name, value in (
            ("terminal_record_revision", terminal_record_revision),
            ("task_generation", task_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        current = self.load()
        key = (agent_id, terminal_record_revision, task_generation)
        delivery = next(
            (
                item
                for item in current.deliveries
                if (item.agent_id, item.terminal_record_revision, item.task_generation) == key
            ),
            None,
        )
        if delivery is None:
            return None
        stable_receipt_id = _delivery_receipt_id(*key)
        if delivery.state == "committed" and delivery.receipt_id == stable_receipt_id:
            return delivery
        if (
            delivery.state != "claimed"
            or delivery.claimed_execution_id != execution_id
            or receipt_id != stable_receipt_id
        ):
            return None
        committed = replace(delivery, state="committed", receipt_id=stable_receipt_id)
        deliveries = tuple(committed if item == delivery else item for item in current.deliveries)
        self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                current.tasks,
                current.questions,
                deliveries,
            )
        )
        return committed

    def replace_current_task(
        self,
        task_id: str,
        execution_id: str,
        *,
        origin_anchor_id: int | None = None,
        permission_level: ToolPermissionLevel = "read-only",
        iteration_budget_limit: int = 50,
    ) -> ForegroundWorkSnapshot:
        """Supersede the latest Task and atomically register its replacement."""
        _require_identifier(task_id, "task_id")
        _require_identifier(execution_id, "execution_id")
        if origin_anchor_id is not None and (
            isinstance(origin_anchor_id, bool)
            or not isinstance(origin_anchor_id, int)
            or origin_anchor_id < 1
        ):
            raise ValueError("origin_anchor_id must be a positive integer")
        validate_permission_level(permission_level)
        _require_positive_integer(iteration_budget_limit, "iteration_budget_limit")
        current = self.load()
        if any(item.task_id == task_id for item in current.tasks):
            raise ForegroundWorkConflict("replacement task ID already exists")
        tasks = list(current.tasks)
        questions = list(current.questions)
        if tasks and tasks[-1].state not in {"cancelled", "superseded"}:
            prior = tasks[-1]
            tasks[-1] = ForegroundWorkTask(
                prior.task_id,
                prior.generation + 1,
                prior.execution_id,
                "superseded",
                prior.origin_anchor_id,
                prior.permission_level,
                prior.iteration_stage,
                prior.iteration_budget_limit,
                prior.iteration_budget_used,
                prior.charged_execution_ids,
            )
            unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
            questions = [
                ForegroundWorkQuestion(
                    item.question_id,
                    item.task_id,
                    item.generation,
                    item.execution_id,
                    item.tool_call_id,
                    item.text,
                    "invalidated"
                    if item.task_id == prior.task_id and item.state in unresolved
                    else item.state,
                    item.answer,
                    item.answer_request_id,
                    item.answer_fingerprint,
                    item.answer_kind,
                    item.answer_option_id,
                    item.resume_execution_id,
                    item.answer_anchor_id,
                    item.options,
                    item.owner_agent_id,
                )
                for item in questions
            ]
        tasks.append(
            ForegroundWorkTask(
                task_id,
                1,
                execution_id,
                "running",
                origin_anchor_id,
                permission_level,
                1,
                iteration_budget_limit,
            )
        )
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, tuple(tasks), tuple(questions))
        )

    def interrupt_uncertain_tasks(self) -> ForegroundWorkSnapshot:
        """Keep ambiguous prior work visible but closed to automatic admission."""
        current = self.load()
        tasks = tuple(
            replace(task, state="interrupted")
            if task.state not in {"cancelled", "superseded"}
            else task
            for task in current.tasks
        )
        if tasks == current.tasks:
            return current
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, tasks, current.questions))

    def prepare_question(
        self,
        *,
        task_id: str,
        execution_id: str,
        question_id: str,
        tool_call_id: str,
        text: str,
        options: list[ForegroundQuestionOption] | None = None,
        owner_agent_id: str | None = None,
    ) -> ForegroundWorkSnapshot:
        for name, value in (
            ("task_id", task_id),
            ("execution_id", execution_id),
            ("question_id", question_id),
            ("tool_call_id", tool_call_id),
        ):
            _require_identifier(value, name)
        if not isinstance(text, str) or not text.strip() or len(text) > _MAX_QUESTION_LENGTH:
            raise ValueError("question must contain 1 to 2000 characters")
        normalized_options = _validated_question_options(options)
        current = self.load()
        existing = current.question(question_id)
        if existing is not None:
            if (existing.task_id, existing.execution_id, existing.tool_call_id, existing.text) == (
                task_id,
                execution_id,
                tool_call_id,
                text,
            ) and existing.options == normalized_options and existing.owner_agent_id == owner_agent_id:
                return current
            raise ForegroundWorkConflict("question_id is already bound to another question")
        if owner_agent_id is not None:
            _require_identifier(owner_agent_id, "owner_agent_id")
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        allowed_task_states = {"running"}
        if owner_agent_id is not None:
            allowed_task_states.update({"waiting_for_answer", "completed"})
        if task is None or task.state not in allowed_task_states:
            raise ForegroundWorkConflict("question belongs to an inactive execution")
        if owner_agent_id is None and task.execution_id != execution_id:
            raise ForegroundWorkConflict("question belongs to an inactive execution")
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        if owner_agent_id is None and any(
            item.task_id == task_id
            and item.owner_agent_id is None
            and item.state in unresolved
            and not (
                item.state in {"resuming", "executing"}
                and item.resume_execution_id == execution_id
                and item.answer is not None
            )
            for item in current.questions
        ):
            raise ForegroundWorkConflict("task already has an unresolved question")
        question = ForegroundWorkQuestion(
            question_id,
            task_id,
            task.generation,
            execution_id,
            tool_call_id,
            text,
            "preparing",
            options=normalized_options,
            owner_agent_id=owner_agent_id,
        )
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                current.tasks,
                (*current.questions, question),
            )
        )

    def publish_question(
        self,
        question_id: str,
        *,
        committed_tool_call_ids: set[str],
    ) -> ForegroundWorkSnapshot:
        current = self.load()
        question = current.question(question_id)
        if question is None:
            raise ForegroundWorkConflict("question does not exist")
        if question.state == "pending":
            return current
        if question.state != "preparing":
            raise ForegroundWorkConflict("question is not awaiting publication")
        committed = question.tool_call_id in committed_tool_call_ids
        next_state: QuestionState = "pending" if committed else "interrupted"
        questions = tuple(
            ForegroundWorkQuestion(
                item.question_id,
                item.task_id,
                item.generation,
                item.execution_id,
                item.tool_call_id,
                item.text,
                next_state,
                item.answer,
                item.answer_request_id,
                item.answer_fingerprint,
                item.answer_kind,
                item.answer_option_id,
                item.resume_execution_id,
                item.answer_anchor_id,
                item.options,
                item.owner_agent_id,
            )
            if item.question_id == question_id
            else item
            for item in current.questions
        )
        prior_task = next(
            (item for item in current.tasks if item.task_id == question.task_id), None
        )
        task_state: TaskState = (
            "waiting_for_answer"
            if committed and question.owner_agent_id is None
            else ("waiting_for_answer" if prior_task.state == "completed" else prior_task.state)
            if question.owner_agent_id is not None and prior_task is not None
            else "interrupted"
        )
        tasks = tuple(
            ForegroundWorkTask(
                item.task_id,
                item.generation,
                item.execution_id,
                task_state,
                item.origin_anchor_id,
                item.permission_level,
                item.iteration_stage,
                item.iteration_budget_limit,
                item.iteration_budget_used,
                item.charged_execution_ids,
            )
            if item.task_id == question.task_id
            else item
            for item in current.tasks
        )
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, tasks, questions))

    def suspend_for_child_question(
        self,
        task_id: str,
        *,
        generation: int,
        execution_id: str,
    ) -> ForegroundWorkSnapshot:
        """Keep Main nonterminal while its wait tool reports child attention."""
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if (
            task is None
            or task.generation != generation
            or task.execution_id != execution_id
            or task.state != "running"
        ):
            raise ForegroundWorkConflict("foreground execution is no longer current")
        updated = replace(task, state="waiting_for_answer")
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(updated if item.task_id == task_id else item for item in current.tasks),
                current.questions,
            )
        )

    def answer_question(
        self,
        question_id: str,
        answer: str | None = None,
        *,
        choice_id: str | None = None,
        expected_revision: int,
        request_id: str,
    ) -> ForegroundWorkSnapshot:
        _require_identifier(question_id, "question_id")
        _require_identifier(request_id, "request_id")
        if (answer is None) == (choice_id is None):
            raise ValueError("answer must be either free text or one choice ID")
        current = self.load()
        question = current.question(question_id)
        if question is None:
            raise ForegroundWorkConflict("question does not exist")
        if choice_id is not None:
            _require_identifier(choice_id, "choice_id")
            selected = next((item for item in question.options if item["id"] == choice_id), None)
            if selected is None:
                raise ValueError("choice_id does not belong to this question")
            answer_text = selected["label"]
            answer_kind: AnswerKind = "choice"
            answer_option_id: str | None = choice_id
            fingerprint = _choice_fingerprint(question_id, choice_id)
        else:
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > _MAX_ANSWER_LENGTH
            ):
                raise ValueError("answer must contain 1 to 10000 characters")
            answer_text = answer
            answer_kind = "text"
            answer_option_id = None
            fingerprint = _answer_fingerprint(question_id, answer_text)
        if question.answer_request_id == request_id:
            if question.answer_fingerprint != fingerprint:
                raise ForegroundWorkConflict("request_id was reused with a different answer")
            return current
        if (
            question.state in {"answer_received", "resuming", "executing", "answered"}
            and question.answer_fingerprint == fingerprint
        ):
            return current
        if current.revision != expected_revision:
            raise StaleForegroundWorkRevision("foreground-work revision is stale")
        if question.state != "pending":
            raise ForegroundWorkConflict("question is no longer awaiting an answer")
        return self._replace_question(
            current,
            question_id,
            state="answer_received",
            answer=answer_text,
            answer_kind=answer_kind,
            answer_option_id=answer_option_id,
            answer_request_id=request_id,
            answer_fingerprint=fingerprint,
        )

    def begin_resume(self, question_id: str, execution_id: str) -> ForegroundWorkSnapshot:
        """Claim an explicit answer submission for one new Main execution."""
        _require_identifier(question_id, "question_id")
        _require_identifier(execution_id, "execution_id")
        current = self.load()
        question = current.question(question_id)
        if question is None or question.answer is None:
            raise ForegroundWorkConflict("question has no accepted answer")
        if question.state == "resuming":
            if question.resume_execution_id == execution_id:
                return current
            raise ForegroundWorkConflict("question answer already has an active execution")
        if question.state != "answer_received":
            raise ForegroundWorkConflict("question answer is not ready to resume")
        task = next((item for item in current.tasks if item.task_id == question.task_id), None)
        if task is None:
            raise ForegroundWorkConflict("question task does not exist")
        updated_question = ForegroundWorkQuestion(
            question.question_id,
            question.task_id,
            question.generation,
            question.execution_id,
            question.tool_call_id,
            question.text,
            "resuming",
            question.answer,
            question.answer_request_id,
            question.answer_fingerprint,
            question.answer_kind,
            question.answer_option_id,
            execution_id,
            question.answer_anchor_id,
            question.options,
            question.owner_agent_id,
        )
        updated_task = ForegroundWorkTask(
            task.task_id,
            task.generation,
            execution_id,
            "running",
            task.origin_anchor_id,
            task.permission_level,
            task.iteration_stage + 1,
            task.iteration_budget_limit,
            0,
            (),
        )
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(
                    updated_task if item.task_id == task.task_id else item for item in current.tasks
                ),
                tuple(
                    updated_question if item.question_id == question_id else item
                    for item in current.questions
                ),
            )
        )

    def begin_subagent_resume(
        self,
        question_id: str,
        execution_id: str,
        *,
        answer_anchor_id: int,
    ) -> ForegroundWorkSnapshot:
        """Commit a child answer to its new execution without resuming Main."""
        _require_identifier(question_id, "question_id")
        _require_identifier(execution_id, "execution_id")
        if isinstance(answer_anchor_id, bool) or not isinstance(answer_anchor_id, int) or answer_anchor_id < 1:
            raise ValueError("answer_anchor_id must be a positive integer")
        current = self.load()
        question = current.question(question_id)
        if question is None or question.owner_agent_id is None or question.answer is None:
            raise ForegroundWorkConflict("question is not owned by a SubAgent")
        if question.state == "answered":
            if question.resume_execution_id == execution_id:
                return current
            raise ForegroundWorkConflict("question answer already has an execution")
        if question.state != "answer_received":
            raise ForegroundWorkConflict("question answer is not ready to resume")
        updated = self._replace_question(
            current,
            question_id,
            state="executing",
            resume_execution_id=execution_id,
            answer_anchor_id=answer_anchor_id,
        )
        task = next((item for item in updated.tasks if item.task_id == question.task_id), None)
        if task is None:
            raise ForegroundWorkConflict("question task does not exist")
        waiting_task = replace(task, state="waiting_for_answer")
        return self._publish(
            ForegroundWorkSnapshot(
                updated.revision + 1,
                tuple(waiting_task if item.task_id == task.task_id else item for item in updated.tasks),
                updated.questions,
            )
        )

    def finish_subagent_execution(
        self, task_id: str, *, generation: int, agent_id: str, execution_id: str
    ) -> ForegroundWorkSnapshot:
        """Release Main for an explicit continuation after child work has settled."""
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None or task.generation != generation:
            raise ForegroundWorkConflict("task generation is no longer current")
        if task.state in {"cancelled", "superseded"}:
            return current
        questions = tuple(
            replace(question, state="answered")
            if question.task_id == task_id
            and question.owner_agent_id is not None
            and question.state == "executing"
            and question.owner_agent_id == agent_id
            and question.resume_execution_id == execution_id
            else question
            for question in current.questions
        )
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        state: TaskState = task.state if task.state in {"running", "completed"} else (
            "waiting_for_answer"
            if any(
                question.task_id == task_id and question.state in unresolved
                for question in questions
            )
            else "interrupted"
        )
        if task.state == state and questions == current.questions:
            return current
        updated = replace(task, state=state)
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(updated if item.task_id == task_id else item for item in current.tasks),
                questions,
            )
        )

    def acknowledge_subagent_recovery(self, agent_id: str, execution_id: str) -> ForegroundWorkSnapshot:
        """An explicit continuation has queued an already accepted child answer."""
        current = self.load()
        questions = tuple(
            replace(question, state="answered", resume_execution_id=execution_id)
            if question.owner_agent_id == agent_id and question.state == "interrupted"
            and question.answer is not None else question for question in current.questions
        )
        if questions == current.questions:
            return current
        return self._publish(replace(current, revision=current.revision + 1, questions=questions))

    def interrupt_subagent_question(self, question_id: str) -> ForegroundWorkSnapshot:
        """Mark a child question interrupted when its saved execution resource is absent."""
        current = self.load()
        question = current.question(question_id)
        if question is None or question.owner_agent_id is None:
            raise ForegroundWorkConflict("question is not owned by a SubAgent")
        updated = self._replace_question(current, question_id, state="interrupted")
        task = next((item for item in updated.tasks if item.task_id == question.task_id), None)
        if task is None or task.state in {"cancelled", "superseded", "interrupted"}:
            return updated
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        task_state: TaskState = (
            "waiting_for_answer"
            if any(
                item.task_id == task.task_id and item.state in unresolved
                for item in updated.questions
            )
            else "interrupted"
        )
        if task.state == task_state:
            return updated
        interrupted_task = replace(task, state=task_state)
        return self._publish(
            ForegroundWorkSnapshot(
                updated.revision + 1,
                tuple(
                    interrupted_task if item.task_id == task.task_id else item
                    for item in updated.tasks
                ),
                updated.questions,
            )
        )

    def cancel_task(
        self,
        task_id: str,
        *,
        expected_revision: int,
        request_id: str,
    ) -> ForegroundWorkSnapshot:
        """Persist a generation fence and invalidate unanswered questions for one Task."""
        _require_identifier(task_id, "task_id")
        _require_identifier(request_id, "request_id")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("expected_revision must be an integer")
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None:
            raise ForegroundWorkConflict("task does not exist")
        if task.state == "cancelled":
            return current
        if task.state == "superseded":
            raise ForegroundWorkConflict("task has been superseded")
        if current.revision != expected_revision:
            raise StaleForegroundWorkRevision("foreground-work revision is stale")
        tasks = tuple(
            ForegroundWorkTask(
                item.task_id,
                item.generation + 1 if item.task_id == task_id else item.generation,
                item.execution_id,
                "cancelled" if item.task_id == task_id else item.state,
                item.origin_anchor_id,
                item.permission_level,
                item.iteration_stage,
                item.iteration_budget_limit,
                item.iteration_budget_used,
                item.charged_execution_ids,
            )
            for item in current.tasks
        )
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        questions = tuple(
            ForegroundWorkQuestion(
                item.question_id,
                item.task_id,
                item.generation,
                item.execution_id,
                item.tool_call_id,
                item.text,
                "invalidated" if item.task_id == task_id and item.state in unresolved else item.state,
                item.answer,
                item.answer_request_id,
                item.answer_fingerprint,
                item.answer_kind,
                item.answer_option_id,
                item.resume_execution_id,
                item.answer_anchor_id,
                item.options,
                item.owner_agent_id,
            )
            for item in current.questions
        )
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, tasks, questions)
        )

    def start_resume(self, question_id: str, execution_id: str) -> None:
        """Fence uncertain executions before any model or Tool side effects."""
        current = self.load()
        question = current.question(question_id)
        if (
            question is None
            or question.state != "resuming"
            or question.resume_execution_id != execution_id
        ):
            raise ForegroundWorkConflict("answer execution is no longer admitted")
        self._replace_question(current, question_id, state="executing")

    def complete_task(self, task_id: str, execution_id: str) -> ForegroundWorkSnapshot:
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None or task.execution_id != execution_id or task.state != "running":
            return current
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        state: TaskState = (
            "waiting_for_answer"
            if any(
                question.task_id == task_id and question.state in unresolved
                for question in current.questions
            )
            else "completed"
        )
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(
                    ForegroundWorkTask(
                        item.task_id,
                        item.generation,
                        item.execution_id,
                        state if item.task_id == task_id else item.state,
                        item.origin_anchor_id,
                        item.permission_level,
                        item.iteration_stage,
                        item.iteration_budget_limit,
                        item.iteration_budget_used,
                        item.charged_execution_ids,
                    )
                    for item in current.tasks
                ),
                current.questions,
            )
        )

    def record_iteration_usage(
        self,
        task_id: str,
        *,
        generation: int,
        execution_id: str,
        model_calls: int,
    ) -> ForegroundWorkSnapshot:
        """Charge one committed execution once, ignoring callbacks from revoked generations."""
        _require_identifier(task_id, "task_id")
        _require_identifier(execution_id, "execution_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
            raise ValueError("model_calls must be a nonnegative integer")
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if (
            task is None
            or task.generation != generation
            or task.execution_id != execution_id
            or task.state not in {"running", "waiting_for_answer", "completed"}
            or execution_id in task.charged_execution_ids
        ):
            return current
        updated = ForegroundWorkTask(
            task.task_id,
            task.generation,
            task.execution_id,
            task.state,
            task.origin_anchor_id,
            task.permission_level,
            task.iteration_stage,
            task.iteration_budget_limit,
            task.iteration_budget_used + model_calls,
            (*task.charged_execution_ids, execution_id),
        )
        tasks = tuple(updated if item.task_id == task_id else item for item in current.tasks)
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, tasks, current.questions))

    def remaining_iteration_budget(self, task_id: str, *, generation: int) -> int:
        """Return the nonnegative budget remaining for the current Task generation."""
        _require_identifier(task_id, "task_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None or task.generation != generation:
            raise ForegroundWorkConflict("task generation is no longer current")
        return max(0, task.iteration_budget_limit - task.iteration_budget_used)

    def is_active_generation(self, task_id: str, generation: int) -> bool:
        """Return whether a Task generation can still admit delegated work."""
        if self._closed:
            return False
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        return (
            task is not None
            and task.generation == generation
            and task.state in {"running", "waiting_for_answer", "completed"}
        )

    def is_current_execution(self, task_id: str, generation: int, execution_id: str) -> bool:
        """Return whether callbacks from this exact foreground execution remain admitted."""
        if self._closed:
            return False
        current = self.load()
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        return (
            task is not None
            and task.generation == generation
            and task.execution_id == execution_id
            and task.state == "running"
        )

    def reconcile(
        self, messages: list[dict[str, Any]], *, subagents: tuple[SubAgentRecord, ...] | None = None
    ) -> ForegroundWorkSnapshot:
        """Reconcile persisted intents with the committed Session after startup or Restore."""
        current = self.load()
        session_tool_calls = {
            message.get("foreground_question_id")
            for message in messages
            if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str)
        }
        child_records = {record.agent_id: record for record in (subagents or ())}
        persisted_receipts = {
            receipt
            for message in messages
            if message.get("role") == "assistant"
            and isinstance(message.get("foreground_delivery_receipts"), list)
            for receipt in message["foreground_delivery_receipts"]
            if isinstance(receipt, str)
        }
        execution_commits: dict[tuple[str, int, str], int] = {}
        execution_commit_positions: dict[tuple[str, int, str], int] = {}
        conflicting_commits: set[tuple[str, int, str]] = set()
        current_stages = {task.task_id: task.iteration_stage for task in current.tasks}
        latest_foreground_input_position = -1
        for position, message in enumerate(messages):
            if (
                message.get("role") == "user"
                and isinstance(message.get("foreground_task_id"), str)
            ):
                latest_foreground_input_position = position
            if message.get("role") != "assistant" or message.get("status") != "completed":
                continue
            receipt = message.get("foreground_execution_commit")
            if not isinstance(receipt, dict):
                continue
            task_id = receipt.get("task_id")
            generation = receipt.get("generation")
            execution_id = receipt.get("execution_id")
            commit_id = receipt.get("commit_id")
            model_calls = receipt.get("model_calls")
            if (
                not isinstance(task_id, str)
                or not task_id
                or type(generation) is not int
                or generation < 1
                or not isinstance(execution_id, str)
                or not execution_id
                or commit_id != f"{task_id}:{generation}:{execution_id}"
                or type(model_calls) is not int
                or model_calls < 0
                or receipt.get("iteration_stage") != current_stages.get(task_id)
            ):
                continue
            key = (task_id, generation, execution_id)
            previous_model_calls = execution_commits.get(key)
            if key in conflicting_commits:
                continue
            if previous_model_calls is not None and previous_model_calls != model_calls:
                execution_commits.pop(key, None)
                execution_commit_positions.pop(key, None)
                conflicting_commits.add(key)
                continue
            execution_commits[key] = model_calls
            execution_commit_positions[key] = max(
                position, execution_commit_positions.get(key, -1)
            )
        waited_results: set[tuple[str, int]] = set()
        for message in messages:
            if message.get("role") != "tool" or message.get("name") != "wait_agent":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                values = message.get("foreground_wait_results", json.loads(content))
            except json.JSONDecodeError:
                values = message.get("foreground_wait_results")
            if isinstance(values, list):
                waited_results.update(
                    (item["agent_id"], item["terminal_record_revision"])
                    for item in values
                    if isinstance(item, dict)
                    and isinstance(item.get("agent_id"), str)
                    and item.get("status") in {"completed", "failed"}
                    and type(item.get("terminal_record_revision")) is int
                )
        for agent_id, terminal_revision in waited_results:
            record = child_records.get(agent_id)
            if (
                record is not None
                and record.revision == terminal_revision
                and record.source.kind is SubAgentSourceKind.FOREGROUND
                and record.source.task_id is not None
                and record.source.task_generation is not None
                and record.status in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
            ):
                persisted_receipts.add(
                    _delivery_receipt_id(
                        record.agent_id, record.revision, record.source.task_generation
                    )
                )
        answer_messages = {
            message.get("foreground_question_id"): message
            for message in messages
            if message.get("role") == "user"
            and isinstance(message.get("foreground_question_id"), str)
        }
        changed = False
        tasks_by_id = {task.task_id: task for task in current.tasks}
        deliveries_by_key = {
            (item.agent_id, item.terminal_record_revision, item.task_generation): item
            for item in current.deliveries
        }
        for delivery_key, delivery in tuple(deliveries_by_key.items()):
            task = tasks_by_id.get(delivery.task_id)
            if (
                task is None
                or task.generation != delivery.task_generation
                or task.state in {"cancelled", "superseded"}
            ) and delivery.state in {"pending", "claimed"}:
                deliveries_by_key[delivery_key] = replace(delivery, state="invalidated")
                changed = True
        for record in child_records.values():
            source = record.source
            if (
                source.kind is not SubAgentSourceKind.FOREGROUND
                or source.task_id is None
                or source.task_generation is None
                or record.status not in {SubAgentStatus.COMPLETED, SubAgentStatus.FAILED}
            ):
                continue
            delivery_key = (record.agent_id, record.revision, source.task_generation)
            if delivery_key in deliveries_by_key:
                continue
            task = tasks_by_id.get(source.task_id)
            delivery_state: DeliveryState = (
                "pending"
                if task is not None
                and task.generation == source.task_generation
                and task.state not in {"cancelled", "superseded"}
                else "invalidated"
            )
            deliveries_by_key[delivery_key] = ForegroundWorkDelivery(
                agent_id=record.agent_id,
                terminal_record_revision=record.revision,
                task_id=source.task_id,
                task_generation=source.task_generation,
                state=delivery_state,
            )
            changed = True
        for delivery_key, delivery in tuple(deliveries_by_key.items()):
            receipt_id = _delivery_receipt_id(*delivery_key)
            if receipt_id in persisted_receipts:
                updated = replace(delivery, state="committed", receipt_id=receipt_id)
            elif delivery.state == "committed":
                updated = replace(delivery, state="invalidated")
            else:
                continue
            if updated != delivery:
                deliveries_by_key[delivery_key] = updated
                changed = True
        questions: list[ForegroundWorkQuestion] = []
        task_states: dict[str, tuple[str, TaskState]] = {}
        for question in current.questions:
            question_state = question.state
            answer = question.answer
            answer_kind = question.answer_kind
            answer_option_id = question.answer_option_id
            request_id = question.answer_request_id
            fingerprint = question.answer_fingerprint
            resume_id = question.resume_execution_id
            anchor = question.answer_anchor_id
            child = child_records.get(question.owner_agent_id or "")
            child_committed = (
                child is not None
                and child.source.task_id == question.task_id
                and child.source.task_generation == question.generation
            )
            if question.state == "preparing":
                committed = question.question_id in session_tool_calls
                if child_committed and child is not None:
                    committed = any(
                        message.get("role") == "tool"
                        and message.get("foreground_question_id") == question.question_id
                        and message.get("tool_call_id") == question.tool_call_id
                        for message in child.conversation
                    )
                question_state = (
                    "preparing"
                    if (child is not None and child.status.value in {"queued", "running"})
                    or (question.owner_agent_id is not None and subagents is None)
                    else "pending" if committed else "interrupted"
                )
            elif (
                question.state == "pending"
                and question.owner_agent_id is None
                and question.question_id not in session_tool_calls
            ):
                question_state = "invalidated"
            elif (
                question.state == "answer_received"
                and question.owner_agent_id is None
                and question.question_id not in session_tool_calls
            ):
                question_state = "invalidated"
                answer = request_id = fingerprint = resume_id = None
                answer_kind = answer_option_id = None
            elif question.state in {"resuming", "executing"}:
                answer_message = answer_messages.get(question.question_id)
                if answer_message is not None:
                    question_state = (
                        "executing"
                        if question.owner_agent_id is not None and question.state == "executing"
                        else "answered"
                    )
                    if (child_committed and child is not None
                            and child.execution_id == question.resume_execution_id
                            and child.status.value in {"completed", "failed", "cancelled", "interrupted"}):
                        question_state = "answered"
                    anchor_value = answer_message.get("restore_anchor_id")
                    anchor = (
                        anchor_value if type(anchor_value) is int and anchor_value > 0 else None
                    )
                else:
                    question_state = (
                        "pending"
                        if question.owner_agent_id is not None
                        else "interrupted"
                        if question.state == "executing"
                        else "answer_received"
                    )
                    if question.owner_agent_id is not None:
                        answer = request_id = fingerprint = resume_id = None
                        answer_kind = answer_option_id = None
                        anchor = None
                    resume_id = None
            elif question.state == "answered":
                answer_message = answer_messages.get(question.question_id)
                anchor_value = (
                    None if answer_message is None else answer_message.get("restore_anchor_id")
                )
                if answer_message is None or (anchor is not None and anchor_value != anchor):
                    question_state = (
                        "pending"
                        if question.owner_agent_id is not None
                        or question.question_id in session_tool_calls
                        else "invalidated"
                    )
                    answer = request_id = fingerprint = resume_id = None
                    answer_kind = answer_option_id = None
                    anchor = None
            changed |= (
                question_state != question.state
                or answer != question.answer
                or answer_kind != question.answer_kind
                or answer_option_id != question.answer_option_id
                or request_id != question.answer_request_id
                or fingerprint != question.answer_fingerprint
                or resume_id != question.resume_execution_id
                or anchor != question.answer_anchor_id
            )
            questions.append(
                ForegroundWorkQuestion(
                    question.question_id,
                    question.task_id,
                    question.generation,
                    question.execution_id,
                    question.tool_call_id,
                    question.text,
                    question_state,
                    answer,
                    request_id,
                    fingerprint,
                    answer_kind,
                    answer_option_id,
                    resume_id,
                    anchor,
                    question.options,
                    question.owner_agent_id,
                )
            )
            state_rank = {
                "interrupted": 1,
                "completed": 2,
                "running": 3,
                "waiting_for_answer": 4,
            }
            if question.owner_agent_id is not None and question_state in {"answered", "preparing"}:
                continue
            candidate = (
                question.execution_id,
                "waiting_for_answer"
                if question_state in {"pending", "answer_received", "executing"}
                else "running"
                if question_state == "resuming" and question.owner_agent_id is None
                else "completed"
                if question_state == "answered" and question.owner_agent_id is None
                else "interrupted",
            )
            previous = task_states.get(question.task_id)
            if previous is None or state_rank[candidate[1]] > state_rank[previous[1]]:
                task_states[question.task_id] = cast(tuple[str, TaskState], candidate)
        tasks: list[ForegroundWorkTask] = []
        for task in current.tasks:
            override = (
                None
                if task.state in {"cancelled", "superseded", "interrupted"}
                else task_states.get(task.task_id)
            )
            committed_execution = (
                task.task_id,
                task.generation,
                task.execution_id,
            ) in execution_commits
            committed_execution_is_latest = (
                execution_commit_positions.get(
                    (task.task_id, task.generation, task.execution_id), -1
                )
                > latest_foreground_input_position
            )
            if override is None and task.state == "running":
                override = (
                    task.execution_id,
                    "completed" if committed_execution else "interrupted",
                )
            elif (
                override is None
                and task.state == "interrupted"
                and committed_execution_is_latest
            ):
                override = (task.execution_id, "completed")
            execution_id, task_state = (
                (task.execution_id, task.state) if override is None else override
            )
            charged_execution_ids = list(task.charged_execution_ids)
            iteration_budget_used = task.iteration_budget_used
            for key, model_calls in execution_commits.items():
                committed_task_id, committed_generation, committed_execution_id = key
                if (
                    committed_task_id == task.task_id
                    and committed_generation == task.generation
                    and committed_execution_id not in charged_execution_ids
                ):
                    iteration_budget_used += model_calls
                    charged_execution_ids.append(committed_execution_id)
            changed |= (
                execution_id != task.execution_id
                or task_state != task.state
                or iteration_budget_used != task.iteration_budget_used
                or tuple(charged_execution_ids) != task.charged_execution_ids
            )
            tasks.append(
                ForegroundWorkTask(
                    task.task_id,
                    task.generation,
                    execution_id,
                    task_state,
                    task.origin_anchor_id,
                    task.permission_level,
                    task.iteration_stage,
                    task.iteration_budget_limit,
                    iteration_budget_used,
                    tuple(charged_execution_ids),
                )
            )
        if not changed:
            return current
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(tasks),
                tuple(questions),
                tuple(deliveries_by_key.values()),
            )
        )

    def _replace_question(
        self,
        current: ForegroundWorkSnapshot,
        question_id: str,
        **changes: object,
    ) -> ForegroundWorkSnapshot:
        question = current.question(question_id)
        if question is None:
            raise ForegroundWorkConflict("question does not exist")
        updated = ForegroundWorkQuestion(
            question_id=question.question_id,
            task_id=question.task_id,
            generation=question.generation,
            execution_id=question.execution_id,
            tool_call_id=question.tool_call_id,
            text=question.text,
            state=cast(QuestionState, changes.get("state", question.state)),
            answer=cast(str | None, changes.get("answer", question.answer)),
            answer_request_id=cast(
                str | None, changes.get("answer_request_id", question.answer_request_id)
            ),
            answer_fingerprint=cast(
                str | None, changes.get("answer_fingerprint", question.answer_fingerprint)
            ),
            answer_kind=cast(AnswerKind | None, changes.get("answer_kind", question.answer_kind)),
            answer_option_id=cast(
                str | None, changes.get("answer_option_id", question.answer_option_id)
            ),
            resume_execution_id=cast(
                str | None, changes.get("resume_execution_id", question.resume_execution_id)
            ),
            answer_anchor_id=cast(
                int | None, changes.get("answer_anchor_id", question.answer_anchor_id)
            ),
            options=question.options,
            owner_agent_id=question.owner_agent_id,
        )
        questions = tuple(
            updated if item.question_id == question_id else item for item in current.questions
        )
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, current.tasks, questions))

    def _publish(self, snapshot: ForegroundWorkSnapshot) -> ForegroundWorkSnapshot:
        if self._closed:
            raise ForegroundWorkConflict("foreground-work authority was revoked")
        if not snapshot.deliveries:
            existing = self.load().deliveries
            if existing:
                snapshot = replace(snapshot, deliveries=existing)
        root = self._prepare_session_root()
        value = {
            "schema_version": _SCHEMA_VERSION,
            "revision": snapshot.revision,
            "tasks": [_task_to_dict(item) for item in snapshot.tasks],
            "questions": [_question_to_dict(item) for item in snapshot.questions],
            "deliveries": [_delivery_to_dict(item) for item in snapshot.deliveries],
        }
        content = json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
        if HOST_FILESYSTEM.entry_exists(self.path):
            HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
            HOST_FILESYSTEM.atomic_replace_text(self.path, content)
        else:
            HOST_FILESYSTEM.atomic_create_text(self.path, content)
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=root)
        HOST_FILESYSTEM.restrict_private_file(self.path)
        return snapshot

    def _existing_session_root(self) -> Path | None:
        base = self.workspace_state.foreground_work_directory
        if not HOST_FILESYSTEM.entry_exists(base):
            return None
        state_root = HOST_FILESYSTEM.require_owned_directory(
            self.workspace_state.path,
            within=self.workspace_state.workspace_path,
        )
        work_root = HOST_FILESYSTEM.require_owned_directory(base, within=state_root)
        root = work_root / self.session_id
        if not HOST_FILESYSTEM.entry_exists(root):
            return None
        return HOST_FILESYSTEM.require_owned_directory(root, within=work_root)

    def _prepare_session_root(self) -> Path:
        work_root = self.workspace_state.prepare_foreground_work_directory()
        root = work_root / self.session_id
        if not HOST_FILESYSTEM.entry_exists(root):
            HOST_FILESYSTEM.path_for_io(root).mkdir(mode=0o700)
        owned = HOST_FILESYSTEM.require_owned_directory(root, within=work_root)
        HOST_FILESYSTEM.restrict_private_directory(owned)
        return owned


def _snapshot_from_dict(value: object) -> ForegroundWorkSnapshot:
    if not isinstance(value, dict):
        raise ForegroundWorkError("Foreground Work state must be an object")
    data = cast(dict[str, object], value)
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3, 4, 5, _SCHEMA_VERSION}:
        raise ForegroundWorkError("Foreground Work state has an unsupported version")
    expected_fields = {"schema_version", "revision", "tasks", "questions"}
    if schema_version >= 6:
        expected_fields.add("deliveries")
    if set(data) != expected_fields:
        raise ForegroundWorkError("Foreground Work state has an unsupported shape")
    revision, tasks_value, questions_value = (data["revision"], data["tasks"], data["questions"])
    if type(revision) is not int or revision < 1:
        raise ForegroundWorkError("Foreground Work revision is malformed")
    if not isinstance(tasks_value, list) or not isinstance(questions_value, list):
        raise ForegroundWorkError("Foreground Work state collections are malformed")
    deliveries_value = data.get("deliveries", [])
    if not isinstance(deliveries_value, list):
        raise ForegroundWorkError("Foreground Work delivery collection is malformed")
    tasks = tuple(
        _task_from_dict(item, schema_version=schema_version) for item in tasks_value
    )
    questions = tuple(
        _question_from_dict(item, schema_version=schema_version) for item in questions_value
    )
    deliveries = tuple(_delivery_from_dict(item) for item in deliveries_value)
    if len({item.task_id for item in tasks}) != len(tasks):
        raise ForegroundWorkError("Foreground Work contains duplicate task IDs")
    if len({item.question_id for item in questions}) != len(questions):
        raise ForegroundWorkError("Foreground Work contains duplicate question IDs")
    if len(
        {
            (item.agent_id, item.terminal_record_revision, item.task_generation)
            for item in deliveries
        }
    ) != len(deliveries):
        raise ForegroundWorkError("Foreground Work contains duplicate result deliveries")
    return ForegroundWorkSnapshot(revision, tasks, questions, deliveries)


def _delivery_receipt_id(agent_id: str, terminal_record_revision: int, task_generation: int) -> str:
    return f"{agent_id}:{terminal_record_revision}:{task_generation}"


def _delivery_to_dict(delivery: ForegroundWorkDelivery) -> dict[str, object]:
    return {
        "agent_id": delivery.agent_id,
        "terminal_record_revision": delivery.terminal_record_revision,
        "task_id": delivery.task_id,
        "task_generation": delivery.task_generation,
        "state": delivery.state,
        "claimed_execution_id": delivery.claimed_execution_id,
        "receipt_id": delivery.receipt_id,
    }


def _delivery_from_dict(value: object) -> ForegroundWorkDelivery:
    if not isinstance(value, dict):
        raise ForegroundWorkError("Foreground Work delivery is malformed")
    item = cast(dict[str, object], value)
    if set(item) != {
        "agent_id",
        "terminal_record_revision",
        "task_id",
        "task_generation",
        "state",
        "claimed_execution_id",
        "receipt_id",
    }:
        raise ForegroundWorkError("Foreground Work delivery has an unsupported shape")
    agent_id = _require_stored_string(item["agent_id"], "delivery.agent_id")
    task_id = _require_stored_string(item["task_id"], "delivery.task_id")
    terminal_revision = item["terminal_record_revision"]
    task_generation = item["task_generation"]
    state = item["state"]
    claimed_execution_id = item["claimed_execution_id"]
    receipt_id = item["receipt_id"]
    if (
        type(terminal_revision) is not int
        or terminal_revision < 1
        or type(task_generation) is not int
        or task_generation < 1
        or not isinstance(state, str)
        or state not in {"pending", "claimed", "committed", "invalidated"}
        or (claimed_execution_id is not None and not isinstance(claimed_execution_id, str))
        or (receipt_id is not None and not isinstance(receipt_id, str))
    ):
        raise ForegroundWorkError("Foreground Work delivery fields are malformed")
    return ForegroundWorkDelivery(
        agent_id,
        terminal_revision,
        task_id,
        task_generation,
        cast(DeliveryState, state),
        claimed_execution_id,
        receipt_id,
    )


def _task_to_dict(task: ForegroundWorkTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "generation": task.generation,
        "execution_id": task.execution_id,
        "state": task.state,
        "origin_anchor_id": task.origin_anchor_id,
        "permission_level": task.permission_level,
        "iteration_stage": task.iteration_stage,
        "iteration_budget_limit": task.iteration_budget_limit,
        "iteration_budget_used": task.iteration_budget_used,
        "charged_execution_ids": list(task.charged_execution_ids),
    }


def _task_from_dict(value: object, *, schema_version: int) -> ForegroundWorkTask:
    if not isinstance(value, dict):
        raise ForegroundWorkError("Foreground Work task is malformed")
    item = cast(dict[str, object], value)
    expected = {"task_id", "generation", "execution_id", "state", "origin_anchor_id"}
    if schema_version >= 3:
        expected.add("permission_level")
    if schema_version >= 4:
        expected.update(
            {
                "iteration_stage",
                "iteration_budget_limit",
                "iteration_budget_used",
                "charged_execution_ids",
            }
        )
    if set(item) != expected:
        raise ForegroundWorkError("Foreground Work task has an unsupported shape")
    task_id = _require_stored_string(item["task_id"], "task_id")
    execution_id = _require_stored_string(item["execution_id"], "execution_id")
    generation, state, anchor = item["generation"], item["state"], item["origin_anchor_id"]
    if type(generation) is not int or generation < 1:
        raise ForegroundWorkError("Foreground Work task generation is malformed")
    if state not in {
        "running",
        "waiting_for_answer",
        "completed",
        "interrupted",
        "cancelled",
        "superseded",
    }:
        raise ForegroundWorkError("Foreground Work task state is malformed")
    if anchor is not None and (type(anchor) is not int or anchor < 1):
        raise ForegroundWorkError("Foreground Work task anchor is malformed")
    try:
        permission_level = validate_permission_level(item.get("permission_level", "read-only"))
    except ValueError as error:
        raise ForegroundWorkError("Foreground Work task permission is malformed") from error
    iteration_stage = item.get("iteration_stage", 1)
    iteration_budget_limit = item.get("iteration_budget_limit", 50)
    iteration_budget_used = item.get("iteration_budget_used", 0)
    charged_execution_ids_value = item.get("charged_execution_ids", [])
    if (
        type(iteration_stage) is not int
        or iteration_stage < 1
        or type(iteration_budget_limit) is not int
        or iteration_budget_limit < 1
        or type(iteration_budget_used) is not int
        or iteration_budget_used < 0
        or not isinstance(charged_execution_ids_value, list)
        or any(not isinstance(value, str) or not value for value in charged_execution_ids_value)
        or len(set(charged_execution_ids_value)) != len(charged_execution_ids_value)
    ):
        raise ForegroundWorkError("Foreground Work task iteration budget is malformed")
    return ForegroundWorkTask(
        task_id,
        generation,
        execution_id,
        cast(TaskState, state),
        anchor,
        permission_level,
        iteration_stage,
        iteration_budget_limit,
        iteration_budget_used,
        tuple(charged_execution_ids_value),
    )


def _question_to_dict(question: ForegroundWorkQuestion) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "task_id": question.task_id,
        "generation": question.generation,
        "execution_id": question.execution_id,
        "tool_call_id": question.tool_call_id,
        "text": question.text,
        "state": question.state,
        "answer": question.answer,
        "answer_request_id": question.answer_request_id,
        "answer_fingerprint": question.answer_fingerprint,
        "answer_kind": question.answer_kind,
        "answer_option_id": question.answer_option_id,
        "resume_execution_id": question.resume_execution_id,
        "answer_anchor_id": question.answer_anchor_id,
        "options": [dict(option) for option in question.options],
        "owner_agent_id": question.owner_agent_id,
    }


def _question_from_dict(value: object, *, schema_version: int) -> ForegroundWorkQuestion:
    if not isinstance(value, dict):
        raise ForegroundWorkError("Foreground Work question is malformed")
    item = cast(dict[str, object], value)
    expected = {
        "question_id",
        "task_id",
        "generation",
        "execution_id",
        "tool_call_id",
        "text",
        "state",
        "answer",
        "answer_request_id",
        "answer_fingerprint",
        "resume_execution_id",
        "answer_anchor_id",
    }
    if schema_version >= 2:
        expected.update({"options", "answer_kind", "answer_option_id"})
    if schema_version >= 5:
        expected.add("owner_agent_id")
    if set(item) != expected:
        raise ForegroundWorkError("Foreground Work question has an unsupported shape")
    task_id = _require_stored_string(item["task_id"], "task_id")
    question_id = _require_stored_string(item["question_id"], "question_id")
    execution_id = _require_stored_string(item["execution_id"], "execution_id")
    tool_call_id = _require_stored_string(item["tool_call_id"], "tool_call_id")
    text = _require_stored_string(item["text"], "text")
    generation, state, anchor = item["generation"], item["state"], item["answer_anchor_id"]
    if type(generation) is not int or generation < 1:
        raise ForegroundWorkError("Foreground Work question generation is malformed")
    if state not in {
        "preparing",
        "pending",
        "answer_received",
        "resuming",
        "executing",
        "answered",
        "interrupted",
        "invalidated",
    }:
        raise ForegroundWorkError("Foreground Work question state is malformed")
    optional_names = ("answer", "answer_request_id", "answer_fingerprint", "resume_execution_id")
    for name in optional_names:
        if item[name] is not None and not isinstance(item[name], str):
            raise ForegroundWorkError(f"Foreground Work {name} is malformed")
    if anchor is not None and (type(anchor) is not int or anchor < 1):
        raise ForegroundWorkError("Foreground Work answer anchor is malformed")
    owner_agent_id = item.get("owner_agent_id")
    if schema_version < 5:
        owner_agent_id = None
    elif owner_agent_id is not None and not isinstance(owner_agent_id, str):
        raise ForegroundWorkError("Foreground Work question owner is malformed")
    if schema_version >= 2 and not isinstance(item["options"], list):
        raise ForegroundWorkError("Foreground Work question options are malformed")
    try:
        options = _validated_question_options(item.get("options") if schema_version >= 2 else None)
    except (TypeError, ValueError) as error:
        raise ForegroundWorkError("Foreground Work question options are malformed") from error
    answer = cast(str | None, item["answer"])
    if schema_version >= 2:
        answer_kind_value = item["answer_kind"]
        answer_option_id_value = item["answer_option_id"]
        if not (
            answer_kind_value is None
            or (isinstance(answer_kind_value, str) and answer_kind_value in {"text", "choice"})
        ):
            raise ForegroundWorkError("Foreground Work answer kind is malformed")
        if answer_option_id_value is not None and not isinstance(answer_option_id_value, str):
            raise ForegroundWorkError("Foreground Work answer option ID is malformed")
        answer_kind = cast(AnswerKind | None, answer_kind_value)
        answer_option_id = answer_option_id_value
    else:
        answer_kind = "text" if answer is not None else None
        answer_option_id = None
    if answer is None:
        if answer_kind is not None or answer_option_id is not None:
            raise ForegroundWorkError("Foreground Work empty answer has answer metadata")
    elif answer_kind == "choice":
        selected = next((option for option in options if option["id"] == answer_option_id), None)
        if selected is None or selected["label"] != answer:
            raise ForegroundWorkError("Foreground Work choice answer does not match an option")
    elif answer_kind != "text" or answer_option_id is not None:
        raise ForegroundWorkError("Foreground Work text answer metadata is malformed")
    return ForegroundWorkQuestion(
        question_id,
        task_id,
        generation,
        execution_id,
        tool_call_id,
        text,
        cast(QuestionState, state),
        answer,
        cast(str | None, item["answer_request_id"]),
        cast(str | None, item["answer_fingerprint"]),
        answer_kind,
        answer_option_id,
        cast(str | None, item["resume_execution_id"]),
        anchor,
        options,
        owner_agent_id,
    )


def _validated_question_options(
    value: object,
) -> tuple[ForegroundQuestionOption, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > 4:
        raise ValueError("question options must contain at most 4 items")
    normalized: list[ForegroundQuestionOption] = []
    seen_ids: set[str] = set()
    for option in value:
        if not isinstance(option, dict) or set(option) != {"id", "label"}:
            raise ValueError("each question option must contain only id and label")
        option_id = option["id"]
        label = option["label"]
        _require_identifier(option_id, "option id")
        if not isinstance(label, str) or not label.strip() or len(label) > 200:
            raise ValueError("option label must contain 1 to 200 characters")
        if option_id in seen_ids:
            raise ValueError("question option IDs must be unique")
        seen_ids.add(option_id)
        normalized.append({"id": option_id, "label": label})
    return tuple(normalized)


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty stable identifier")


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_stored_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForegroundWorkError(f"Foreground Work {name} is malformed")
    return value


def _answer_fingerprint(question_id: str, answer: str) -> str:
    encoded = json.dumps([question_id, answer], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _choice_fingerprint(question_id: str, choice_id: str) -> str:
    encoded = json.dumps(
        ["choice", question_id, choice_id], ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_owned_text(path: Path, *, within: Path) -> str:
    io_path = HOST_FILESYSTEM.path_for_io(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(io_path, flags)
    try:
        HOST_FILESYSTEM.require_opened_owned_regular_file(descriptor, path, within=within)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read().decode("utf-8")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


__all__ = [
    "ForegroundWorkConflict",
    "ForegroundWorkError",
    "ForegroundWorkQuestion",
    "ForegroundWorkSnapshot",
    "ForegroundWorkStore",
    "ForegroundWorkTask",
    "StaleForegroundWorkRevision",
]
