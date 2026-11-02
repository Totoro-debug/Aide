"""Versioned durable coordination for foreground Main Agent questions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from aide.agent.session.session import Session, SessionStoragePartition
from aide.agent.workspace_state import WorkspaceState
from aide.utils.host_filesystem import HOST_FILESYSTEM

_SCHEMA_VERSION = 1
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
TaskState = Literal["running", "waiting_for_answer", "completed", "interrupted", "cancelled"]


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
    resume_execution_id: str | None = None
    answer_anchor_id: int | None = None


@dataclass(frozen=True, slots=True)
class ForegroundWorkSnapshot:
    revision: int
    tasks: tuple[ForegroundWorkTask, ...] = ()
    questions: tuple[ForegroundWorkQuestion, ...] = ()

    def question(self, question_id: str) -> ForegroundWorkQuestion | None:
        return next(
            (item for item in self.questions if item.question_id == question_id),
            None,
        )

    @property
    def pending_questions(self) -> tuple[ForegroundWorkQuestion, ...]:
        return tuple(item for item in self.questions if item.state == "pending")


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
        current = self.load()
        existing = next((item for item in current.tasks if item.task_id == task_id), None)
        if existing is not None:
            if existing.execution_id == execution_id and existing.generation == generation:
                return current
            raise ForegroundWorkConflict("task_id is already bound to another execution")
        task = ForegroundWorkTask(task_id, generation, execution_id, "running", origin_anchor_id)
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, (*current.tasks, task), current.questions)
        )

    def prepare_question(
        self,
        *,
        task_id: str,
        execution_id: str,
        question_id: str,
        tool_call_id: str,
        text: str,
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
        current = self.load()
        existing = current.question(question_id)
        if existing is not None:
            if (existing.task_id, existing.execution_id, existing.tool_call_id, existing.text) == (
                task_id,
                execution_id,
                tool_call_id,
                text,
            ):
                return current
            raise ForegroundWorkConflict("question_id is already bound to another question")
        task = next((item for item in current.tasks if item.task_id == task_id), None)
        if task is None or task.execution_id != execution_id or task.state != "running":
            raise ForegroundWorkConflict("question belongs to an inactive execution")
        unresolved = {"preparing", "pending", "answer_received", "resuming", "executing"}
        if any(
            item.task_id == task_id
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
            question_id, task_id, task.generation, execution_id, tool_call_id, text, "preparing"
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
                item.resume_execution_id,
                item.answer_anchor_id,
            )
            if item.question_id == question_id
            else item
            for item in current.questions
        )
        task_state: TaskState = "waiting_for_answer" if committed else "interrupted"
        tasks = tuple(
            ForegroundWorkTask(
                item.task_id,
                item.generation,
                item.execution_id,
                task_state,
                item.origin_anchor_id,
            )
            if item.task_id == question.task_id
            else item
            for item in current.tasks
        )
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, tasks, questions))

    def answer_question(
        self,
        question_id: str,
        answer: str,
        *,
        expected_revision: int,
        request_id: str,
    ) -> ForegroundWorkSnapshot:
        _require_identifier(question_id, "question_id")
        _require_identifier(request_id, "request_id")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > _MAX_ANSWER_LENGTH:
            raise ValueError("answer must contain 1 to 10000 characters")
        fingerprint = _answer_fingerprint(question_id, answer)
        current = self.load()
        question = current.question(question_id)
        if question is None:
            raise ForegroundWorkConflict("question does not exist")
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
            answer=answer,
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
            execution_id,
            question.answer_anchor_id,
        )
        updated_task = ForegroundWorkTask(
            task.task_id, task.generation + 1, execution_id, "running", task.origin_anchor_id
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
        if task is None or task.execution_id != execution_id:
            return current
        return self._publish(
            ForegroundWorkSnapshot(
                current.revision + 1,
                tuple(
                    ForegroundWorkTask(
                        item.task_id,
                        item.generation,
                        item.execution_id,
                        "completed" if item.task_id == task_id else item.state,
                        item.origin_anchor_id,
                    )
                    for item in current.tasks
                ),
                current.questions,
            )
        )

    def reconcile(self, messages: list[dict[str, Any]]) -> ForegroundWorkSnapshot:
        """Reconcile persisted intents with the committed Session after startup or Restore."""
        current = self.load()
        session_tool_calls = {
            message.get("foreground_question_id")
            for message in messages
            if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str)
        }
        answer_messages = {
            message.get("foreground_question_id"): message
            for message in messages
            if message.get("role") == "user"
            and isinstance(message.get("foreground_question_id"), str)
        }
        changed = False
        questions: list[ForegroundWorkQuestion] = []
        task_states: dict[str, tuple[str, TaskState]] = {}
        for question in current.questions:
            question_state = question.state
            answer = question.answer
            request_id = question.answer_request_id
            fingerprint = question.answer_fingerprint
            resume_id = question.resume_execution_id
            anchor = question.answer_anchor_id
            if question.state == "preparing":
                question_state = (
                    "pending" if question.question_id in session_tool_calls else "interrupted"
                )
            elif question.state == "pending" and question.question_id not in session_tool_calls:
                question_state = "invalidated"
            elif (
                question.state == "answer_received"
                and question.question_id not in session_tool_calls
            ):
                question_state = "invalidated"
                answer = request_id = fingerprint = resume_id = None
            elif question.state in {"resuming", "executing"}:
                answer_message = answer_messages.get(question.question_id)
                if answer_message is not None:
                    question_state = "answered"
                    anchor_value = answer_message.get("restore_anchor_id")
                    anchor = (
                        anchor_value if type(anchor_value) is int and anchor_value > 0 else None
                    )
                else:
                    question_state = (
                        "interrupted" if question.state == "executing" else "answer_received"
                    )
                    resume_id = None
            elif question.state == "answered":
                answer_message = answer_messages.get(question.question_id)
                anchor_value = (
                    None if answer_message is None else answer_message.get("restore_anchor_id")
                )
                if answer_message is None or (anchor is not None and anchor_value != anchor):
                    question_state = (
                        "pending" if question.question_id in session_tool_calls else "invalidated"
                    )
                    answer = request_id = fingerprint = resume_id = None
                    anchor = None
            changed |= (
                question_state != question.state
                or answer != question.answer
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
                    resume_id,
                    anchor,
                )
            )
            if question_state in {"pending", "answer_received"}:
                task_states[question.task_id] = (question.execution_id, "waiting_for_answer")
            elif question_state == "resuming":
                task_states[question.task_id] = (resume_id or question.execution_id, "running")
            elif question_state == "answered":
                task_states[question.task_id] = (resume_id or question.execution_id, "completed")
            elif question_state in {"interrupted", "invalidated"}:
                task_states[question.task_id] = (question.execution_id, "interrupted")
        tasks: list[ForegroundWorkTask] = []
        for task in current.tasks:
            override = task_states.get(task.task_id)
            if override is None and task.state == "running":
                override = (task.execution_id, "interrupted")
            execution_id, task_state = (
                (task.execution_id, task.state) if override is None else override
            )
            changed |= execution_id != task.execution_id or task_state != task.state
            tasks.append(
                ForegroundWorkTask(
                    task.task_id,
                    task.generation,
                    execution_id,
                    task_state,
                    task.origin_anchor_id,
                )
            )
        if not changed:
            return current
        return self._publish(
            ForegroundWorkSnapshot(current.revision + 1, tuple(tasks), tuple(questions))
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
            resume_execution_id=cast(
                str | None, changes.get("resume_execution_id", question.resume_execution_id)
            ),
            answer_anchor_id=cast(
                int | None, changes.get("answer_anchor_id", question.answer_anchor_id)
            ),
        )
        questions = tuple(
            updated if item.question_id == question_id else item for item in current.questions
        )
        return self._publish(ForegroundWorkSnapshot(current.revision + 1, current.tasks, questions))

    def _publish(self, snapshot: ForegroundWorkSnapshot) -> ForegroundWorkSnapshot:
        if self._closed:
            raise ForegroundWorkConflict("foreground-work authority was revoked")
        root = self._prepare_session_root()
        value = {
            "schema_version": _SCHEMA_VERSION,
            "revision": snapshot.revision,
            "tasks": [_task_to_dict(item) for item in snapshot.tasks],
            "questions": [_question_to_dict(item) for item in snapshot.questions],
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
    if set(data) != {"schema_version", "revision", "tasks", "questions"}:
        raise ForegroundWorkError("Foreground Work state has an unsupported shape")
    if type(data["schema_version"]) is not int or data["schema_version"] != _SCHEMA_VERSION:
        raise ForegroundWorkError("Foreground Work state has an unsupported version")
    revision, tasks_value, questions_value = (data["revision"], data["tasks"], data["questions"])
    if type(revision) is not int or revision < 1:
        raise ForegroundWorkError("Foreground Work revision is malformed")
    if not isinstance(tasks_value, list) or not isinstance(questions_value, list):
        raise ForegroundWorkError("Foreground Work state collections are malformed")
    tasks = tuple(_task_from_dict(item) for item in tasks_value)
    questions = tuple(_question_from_dict(item) for item in questions_value)
    if len({item.task_id for item in tasks}) != len(tasks):
        raise ForegroundWorkError("Foreground Work contains duplicate task IDs")
    if len({item.question_id for item in questions}) != len(questions):
        raise ForegroundWorkError("Foreground Work contains duplicate question IDs")
    return ForegroundWorkSnapshot(revision, tasks, questions)


def _task_to_dict(task: ForegroundWorkTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "generation": task.generation,
        "execution_id": task.execution_id,
        "state": task.state,
        "origin_anchor_id": task.origin_anchor_id,
    }


def _task_from_dict(value: object) -> ForegroundWorkTask:
    if not isinstance(value, dict):
        raise ForegroundWorkError("Foreground Work task is malformed")
    item = cast(dict[str, object], value)
    if set(item) != {"task_id", "generation", "execution_id", "state", "origin_anchor_id"}:
        raise ForegroundWorkError("Foreground Work task has an unsupported shape")
    task_id = _require_stored_string(item["task_id"], "task_id")
    execution_id = _require_stored_string(item["execution_id"], "execution_id")
    generation, state, anchor = item["generation"], item["state"], item["origin_anchor_id"]
    if type(generation) is not int or generation < 1:
        raise ForegroundWorkError("Foreground Work task generation is malformed")
    if state not in {"running", "waiting_for_answer", "completed", "interrupted", "cancelled"}:
        raise ForegroundWorkError("Foreground Work task state is malformed")
    if anchor is not None and (type(anchor) is not int or anchor < 1):
        raise ForegroundWorkError("Foreground Work task anchor is malformed")
    return ForegroundWorkTask(task_id, generation, execution_id, cast(TaskState, state), anchor)


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
        "resume_execution_id": question.resume_execution_id,
        "answer_anchor_id": question.answer_anchor_id,
    }


def _question_from_dict(value: object) -> ForegroundWorkQuestion:
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
    return ForegroundWorkQuestion(
        question_id,
        task_id,
        generation,
        execution_id,
        tool_call_id,
        text,
        cast(QuestionState, state),
        cast(str | None, item["answer"]),
        cast(str | None, item["answer_request_id"]),
        cast(str | None, item["answer_fingerprint"]),
        cast(str | None, item["resume_execution_id"]),
        anchor,
    )


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty stable identifier")


def _require_stored_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForegroundWorkError(f"Foreground Work {name} is malformed")
    return value


def _answer_fingerprint(question_id: str, answer: str) -> str:
    encoded = json.dumps([question_id, answer], ensure_ascii=True, separators=(",", ":"))
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
