from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from aide.agent.permission import PermissionSnapshot
from aide.agent.session.session import Session
from aide.agent.tools.core.exec_policy import ResolvedExecShell
from aide.agent.workers.models import WorkerRecord, WorkerReportStatus
from aide.agent.workers.store import WorkerStore, WorkerStoreConflict
from aide.agent.workspace_state import WorkspaceState
from aide.provider.session_configuration import SessionModelConfiguration


def _shell() -> ResolvedExecShell:
    return ResolvedExecShell(
        selector="auto",
        platform="windows",
        family="powershell",
        executable="powershell.exe",
        flags=("-NoProfile",),
        environment=(),
        available=True,
    )


def _create_store(tmp_path: Path) -> tuple[WorkerStore, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "home")
    session = Session.create(state)
    store = WorkerStore(state, session.session_id)
    return store, session.session_id


def _create_worker(store: WorkerStore, *, task_id: str = "worker-task") -> WorkerRecord:
    return store.create(
        parent_task_id=str(uuid4()),
        parent_generation=1,
        task_id=task_id,
        task="Inspect the release notes",
        explicit_context=("Compare the latest two entries",),
        runtime_facts={"workspace": str(store.workspace_state.workspace_path)},
        permission_snapshot=PermissionSnapshot("workspace-write", _shell()),
        session_model_configuration=SessionModelConfiguration(
            provider_id="controlled", model="test-model", reasoning_effort="low"
        ),
        system_prompt="Worker system prompt",
        initial_message="Task: Inspect the release notes\nContext: Compare the latest two entries",
        restore_run_token=str(uuid4()),
    )


def test_worker_state_context_and_report_survive_store_recreation(tmp_path: Path) -> None:
    store, session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    store.checkpoint(
        worker.worker_id,
        conversation=(
            {"role": "user", "content": worker.initial_message},
            {"role": "assistant", "content": "The release notes show two changes."},
        ),
        context_state={"last_compacted": 0},
        usage={"model_calls": 1, "input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    )
    store.publish_report(
        worker.worker_id,
        state="ready",
        summary="The release notes show two changes.",
        usage={"model_calls": 1, "input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    )

    recovered = WorkerStore(store.workspace_state, session_id).get(worker.worker_id)

    assert recovered is not None
    assert recovered.state == "ready"
    assert recovered.explicit_context == ("Compare the latest two entries",)
    assert recovered.permission_level == "workspace-write"
    assert recovered.conversation == (
        {"role": "user", "content": worker.initial_message},
        {"role": "assistant", "content": "The release notes show two changes."},
    )
    assert recovered.report is not None
    assert recovered.report.summary == "The release notes show two changes."
    assert recovered.report_delivery_state == "pending"


def test_worker_status_transitions_accept_wait_and_reject_terminal_reversal(
    tmp_path: Path,
) -> None:
    store, _session_id = _create_store(tmp_path)
    worker = _create_worker(store)

    waiting = store.transition(worker.worker_id, "waiting")
    resumed = store.transition(worker.worker_id, "working")
    assert waiting.state == "waiting"
    assert resumed.state == "working"

    ready = store.publish_report(
        worker.worker_id,
        state="ready",
        summary="done",
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    assert ready.state == "ready"
    with pytest.raises(WorkerStoreConflict):
        store.transition(worker.worker_id, "working")


@pytest.mark.parametrize("terminal_state", ["ready", "blocked", "failed", "cancelled"])
def test_worker_terminal_states_publish_one_report(
    tmp_path: Path, terminal_state: WorkerReportStatus
) -> None:
    store, _session_id = _create_store(tmp_path)
    worker = _create_worker(store)

    terminal = store.publish_report(
        worker.worker_id,
        state=terminal_state,
        summary=f"{terminal_state} report",
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )

    assert terminal.state == terminal_state
    assert terminal.report is not None
    assert terminal.report.sequence == 1
    assert terminal.report_delivery_state == "pending"


def test_worker_report_delivery_uses_stable_receipt_and_is_consumed_once(
    tmp_path: Path,
) -> None:
    store, _session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    terminal = store.publish_report(
        worker.worker_id,
        state="ready",
        summary="done",
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    assert terminal.report is not None

    deliveries = [
        store.claim_report(worker.worker_id, terminal.report.report_id) for _ in range(10)
    ]
    assert all(delivery is not None for delivery in deliveries)
    first_delivery = deliveries[0]
    assert first_delivery is not None
    stable_receipt = first_delivery.receipt_id
    receipt_ids = {delivery.receipt_id for delivery in deliveries if delivery is not None}
    assert receipt_ids == {stable_receipt}
    assert store.acknowledge_report(worker.worker_id, stable_receipt) is True
    assert store.acknowledge_report(worker.worker_id, stable_receipt) is False
    assert store.claim_report(worker.worker_id, terminal.report.report_id) is None
    consumed = store.get(worker.worker_id)
    assert consumed is not None
    assert consumed.report_delivery_state == "consumed"


def test_worker_report_receipt_requires_a_successful_dispatch_message(tmp_path: Path) -> None:
    store, _session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    terminal = store.publish_report(
        worker.worker_id,
        state="ready",
        summary="done",
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    assert terminal.report is not None
    delivery = store.claim_report(worker.worker_id, terminal.report.report_id)
    assert delivery is not None
    receipt = {
        "worker_id": worker.worker_id,
        "report_id": delivery.report.report_id,
        "receipt_id": delivery.receipt_id,
    }

    failed_message = {
        "role": "tool",
        "name": "dispatch_worker",
        "status": "error",
        "worker_report_receipts": [receipt],
    }
    successful_message = {**failed_message, "status": "success"}

    assert store.reconcile_deliveries([failed_message]) == 0
    assert store.reconcile_deliveries([successful_message]) == 1


def test_worker_dispatch_identity_is_idempotent_and_parent_scope_allows_one_worker(
    tmp_path: Path,
) -> None:
    store, _session_id = _create_store(tmp_path)
    parent_task_id = str(uuid4())
    restore_run_token = str(uuid4())
    first = store.create(
        parent_task_id=parent_task_id,
        parent_generation=1,
        task_id="worker-task",
        task="Inspect the release notes",
        explicit_context=("Compare the latest two entries",),
        runtime_facts={"workspace": str(store.workspace_state.workspace_path)},
        permission_snapshot=PermissionSnapshot("workspace-write", _shell()),
        session_model_configuration=SessionModelConfiguration(
            provider_id="controlled", model="test-model", reasoning_effort="low"
        ),
        system_prompt="Worker system prompt",
        initial_message="Task: Inspect the release notes",
        restore_run_token=restore_run_token,
    )
    duplicate = store.create(
        parent_task_id=parent_task_id,
        parent_generation=1,
        task_id="worker-task",
        task="Inspect the release notes",
        explicit_context=("Compare the latest two entries",),
        runtime_facts={"workspace": str(store.workspace_state.workspace_path)},
        permission_snapshot=PermissionSnapshot("workspace-write", _shell()),
        session_model_configuration=SessionModelConfiguration(
            provider_id="controlled", model="test-model", reasoning_effort="low"
        ),
        system_prompt="Worker system prompt",
        initial_message="Task: Inspect the release notes",
        restore_run_token=restore_run_token,
    )
    assert duplicate.worker_id == first.worker_id

    with pytest.raises(WorkerStoreConflict):
        store.create(
            parent_task_id=parent_task_id,
            parent_generation=1,
            task_id="another-worker-task",
            task="Inspect the release notes",
            explicit_context=("Compare the latest two entries",),
            runtime_facts={"workspace": str(store.workspace_state.workspace_path)},
            permission_snapshot=PermissionSnapshot("workspace-write", _shell()),
            session_model_configuration=SessionModelConfiguration(
                provider_id="controlled", model="test-model", reasoning_effort="low"
            ),
            system_prompt="Worker system prompt",
            initial_message="Task: Inspect the release notes",
            restore_run_token=str(uuid4()),
        )


def test_interrupted_work_is_recovered_as_blocked_with_a_report(tmp_path: Path) -> None:
    store, session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    store.checkpoint(
        worker.worker_id,
        conversation=({"role": "user", "content": worker.initial_message},),
        context_state={"last_compacted": 0},
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )

    WorkerStore.recover_workspace(store.workspace_state)
    recovered = WorkerStore(store.workspace_state, session_id).get(worker.worker_id)

    assert recovered is not None
    assert recovered.state == "blocked"
    assert recovered.report is not None
    assert recovered.report.error_code == "service_interrupted"
    assert recovered.report_delivery_state == "pending"


def test_restart_recovers_completed_checkpoint_before_report_publication(tmp_path: Path) -> None:
    store, session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    store.checkpoint(
        worker.worker_id,
        conversation=(
            *worker.conversation,
            {
                "role": "assistant",
                "content": "Verified release 42",
                "tool_calls": [],
                "status": "completed",
                "error": None,
            },
        ),
        context_state={"last_compacted": 0},
        usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    WorkerStore.recover_workspace(store.workspace_state)
    recovered = WorkerStore(store.workspace_state, session_id).get(worker.worker_id)
    assert recovered is not None and recovered.report is not None
    assert recovered.report.status == "ready"
    assert recovered.report.summary == "Verified release 42"
    assert recovered.report_delivery_state == "pending"


@pytest.mark.asyncio
async def test_worker_executor_continues_its_reloaded_tool_checkpoint(tmp_path: Path) -> None:
    from aide.agent.tools.context import ToolRunContext
    from aide.agent.tools.core.exec_host import create_exec_host
    from aide.agent.tools.tool_gateway import ToolGateway
    from aide.agent.workers.executor import WorkerRunnerExecutor
    from aide.config.agent_home import AgentHome
    from aide.config.config import ConfigLoader
    from aide.provider.model_router import ModelRouter
    from aide.provider.models import (
        AssistantModelMessage,
        ModelCompleted,
        ModelResponse,
        ModelUsage,
    )
    from tests.configuration.test_config import MINIMAL_VALID_CONFIG
    from tests.fixtures import ScriptedFakeProvider, StreamScript

    store, session_id = _create_store(tmp_path)
    home = AgentHome(tmp_path / "model-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                (
                    ModelCompleted(
                        ModelResponse(
                            AssistantModelMessage("First line: checkpoint-only evidence"),
                            ModelUsage(1, 1, 2),
                            "stop",
                        )
                    ),
                )
            ),
        )
    )
    router = ModelRouter(configuration=configuration, provider_factory=lambda _: provider)
    worker = store.create(
        parent_task_id="parent-task",
        parent_generation=1,
        task_id="checkpoint-task",
        task="Read the report",
        explicit_context=(),
        runtime_facts={"foreground_execution_id": str(uuid4())},
        permission_snapshot=PermissionSnapshot("read-only", _shell()),
        session_model_configuration=SessionModelConfiguration("primary", "small-model", "low"),
        system_prompt="You are an Aide Worker. Use your own persisted context.",
        initial_message="Assigned task: Read the report",
        restore_run_token=None,
    )
    store.checkpoint(
        worker.worker_id,
        conversation=(
            *worker.conversation,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "saved-read",
                        "name": "read_file",
                        "arguments": '{"path":"report.txt"}',
                    }
                ],
                "status": "completed",
                "error": None,
            },
            {
                "role": "tool",
                "tool_call_id": "saved-read",
                "name": "read_file",
                "content": "checkpoint-only evidence",
                "status": "success",
            },
        ),
        context_state={"last_compacted": 0},
        usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    reloaded_store = WorkerStore(store.workspace_state, session_id)
    reloaded = reloaded_store.get(worker.worker_id)
    assert reloaded is not None
    host = create_exec_host(_shell())
    gateway = ToolGateway(
        workspace=store.workspace_state.workspace_path,
        exec_host=host,
        tool_context=ToolRunContext(workspace=store.workspace_state.workspace_path, exec_host=host),
    )
    executor = WorkerRunnerExecutor(
        workspace_state=store.workspace_state,
        store=reloaded_store,
        model_router=router,
        tool_gateway=gateway,
        exec_host=host,
        schedule_service=None,
        compact_ratio=0.9,
        max_iterations=50,
        max_tool_result_chars=65535,
        enable_tool_micro_compression=False,
        mcp_keywords={},
    )
    try:
        terminal = await executor.execute(
            reloaded,
            foreground_task_is_current=lambda *_: True,
            confirmation=None,
            cancel_requested=lambda: False,
        )
        assert terminal.report is not None
        assert terminal.report.status == "ready"
        assert terminal.report.summary == "First line: checkpoint-only evidence"
        assert len(provider.stream_requests) == 1
        assert any(
            m.get("content") == "checkpoint-only evidence"
            for m in provider.stream_requests[0].messages
        )
        assert terminal.usage["model_calls"] == 2
        assert len([m for m in terminal.conversation if m.get("role") == "tool"]) == 1
    finally:
        await router.close()


def test_receipt_write_failure_can_reconcile_once_after_store_restart(tmp_path: Path) -> None:
    store, session_id = _create_store(tmp_path)
    worker = _create_worker(store)
    terminal = store.publish_report(
        worker.worker_id,
        state="ready",
        summary="durable result",
        usage={"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    assert terminal.report is not None
    delivery = store.claim_report(worker.worker_id, terminal.report.report_id)
    assert delivery is not None

    def fail_replace(_path: Path, _content: str) -> None:
        raise OSError("injected receipt publication failure")

    failing = WorkerStore(store.workspace_state, session_id, replace_text=fail_replace)
    with pytest.raises(OSError, match="receipt publication"):
        failing.acknowledge_report(worker.worker_id, delivery.receipt_id)
    recovered = WorkerStore(store.workspace_state, session_id)
    assert recovered.claim_report(worker.worker_id, terminal.report.report_id) == delivery
    persisted_dispatch = {
        "role": "tool",
        "name": "dispatch_worker",
        "status": "success",
        "worker_report_receipts": [
            {
                "worker_id": worker.worker_id,
                "report_id": terminal.report.report_id,
                "receipt_id": delivery.receipt_id,
            }
        ],
    }
    assert recovered.reconcile_deliveries([persisted_dispatch]) == 1
    assert [recovered.reconcile_deliveries([persisted_dispatch]) for _ in range(10)] == [0] * 10
    assert recovered.claim_report(worker.worker_id, terminal.report.report_id) is None
