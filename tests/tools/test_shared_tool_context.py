from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omni.agent.permission import PermissionSnapshot
from omni.agent.tools.context import ToolRunContext
from omni.agent.tools.core.edit_file import EditFileTool
from omni.agent.tools.core.exec_host import ExecOutcome, ExecProcessSpec, resolve_exec_shell
from omni.agent.tools.core.exec_policy import ExecAssessment
from omni.agent.tools.core.write_file import WriteFileTool
from omni.agent.tools.permission import PermissionContext, ToolInvocationFacts
from omni.agent.tools.tool_gateway import BuiltInToolCatalog, ModelToolCall, ToolGateway
from omni.agent.workspace_state import WorkspaceState
from omni.schedule.service import ScheduleService


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 10, 3, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _schedule_service(workspace: Path) -> ScheduleService:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=workspace.parent / "agent-home")

    async def execute_user_job(_job: object) -> None:
        return None

    async def execute_dream() -> None:
        return None

    return ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )


def _call(name: str, arguments: dict[str, object], *, call_id: str) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


class _Recorder:
    def __init__(self) -> None:
        self.before: list[tuple[UUID, Path]] = []
        self.after: list[tuple[UUID, Path]] = []

    def begin_write(self, run_token: UUID, target: Path) -> Callable[[], None]:
        self.before.append((run_token, target))

        def complete() -> None:
            self.after.append((run_token, target))

        return complete


class _RecordingExecHost:
    def __init__(self) -> None:
        self.resolved_shell = resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=lambda name: r"C:\PowerShell\pwsh.exe" if name == "pwsh" else None,
            version_probe=lambda *_: (7, 5),
        )
        self.inspected: list[Path] = []
        self.executed: list[Path] = []

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del command
        self.inspected.append(cwd)
        return ExecAssessment(syntax_confidence="high", syntax_uncertain=False)

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        del command, timeout
        self.executed.append(cwd)
        return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        return ExecProcessSpec(
            executable="pwsh.exe",
            flags=(),
            cwd=cwd,
            environment=(),
        )


@pytest.mark.asyncio
async def test_shared_builtin_catalog_binds_each_workspace_per_call(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    catalog = BuiltInToolCatalog()
    gateway_a = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_a),
    )
    gateway_b = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_b),
    )

    assert gateway_a.catalog[0] is gateway_b.catalog[0]
    assert "_workspace" not in vars(gateway_a.catalog[0])

    first, second = await asyncio.gather(
        gateway_a.call(_call("write_file", {"path": "same.txt", "content": "A"}, call_id="a")),
        gateway_b.call(_call("write_file", {"path": "same.txt", "content": "B"}, call_id="b")),
    )

    assert (first.status, second.status) == ("success", "success")
    assert (workspace_a / "same.txt").read_text(encoding="utf-8") == "A"
    assert (workspace_b / "same.txt").read_text(encoding="utf-8") == "B"


@pytest.mark.asyncio
async def test_shared_schedule_tool_uses_the_gateway_workspace_service(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    service_a = _schedule_service(workspace_a)
    service_b = _schedule_service(workspace_b)
    catalog = BuiltInToolCatalog()
    gateway_a = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_a, schedule_service=service_a),
    )
    gateway_b = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_b, schedule_service=service_b),
    )

    results = await asyncio.gather(
        gateway_a.call(
            _call(
                "schedule",
                {"action": "add", "message": "A job", "every_seconds": 60},
                call_id="schedule-a",
            )
        ),
        gateway_b.call(
            _call(
                "schedule",
                {"action": "add", "message": "B job", "every_seconds": 60},
                call_id="schedule-b",
            )
        ),
    )

    assert [result.status for result in results] == ["success", "success"]
    jobs_a = await service_a.public_snapshot()
    jobs_b = await service_b.public_snapshot()
    assert [job.message for job in jobs_a] == ["A job"]
    assert [job.message for job in jobs_b] == ["B job"]


@pytest.mark.asyncio
async def test_contextual_file_backup_records_stay_bound_to_each_run(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    catalog = BuiltInToolCatalog()
    gateway_a = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_a),
    )
    gateway_b = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_b),
    )
    recorder_a = _Recorder()
    recorder_b = _Recorder()
    token_a = uuid4()
    token_b = uuid4()

    results = await asyncio.gather(
        gateway_a.call(
            _call("write_file", {"path": "notes.txt", "content": "A"}, call_id="backup-a"),
            file_mutation_recorder=recorder_a,
            run_token=token_a,
        ),
        gateway_b.call(
            _call("write_file", {"path": "notes.txt", "content": "B"}, call_id="backup-b"),
            file_mutation_recorder=recorder_b,
            run_token=token_b,
        ),
    )

    assert [result.status for result in results] == ["success", "success"]
    assert recorder_a.before == [(token_a, (workspace_a / "notes.txt").resolve())]
    assert recorder_b.before == [(token_b, (workspace_b / "notes.txt").resolve())]
    assert recorder_a.after == recorder_a.before
    assert recorder_b.after == recorder_b.before


@pytest.mark.asyncio
async def test_shared_catalog_keeps_permission_snapshots_isolated_per_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = BuiltInToolCatalog()
    base_gateway = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace),
        permission_context=PermissionContext(workspace_root=workspace),
    )
    read_only_gateway = base_gateway.for_run(
        exposed_names=("write_file",),
        permission_snapshot=PermissionSnapshot(
            level="read-only",
            exec_shell=resolve_exec_shell("auto"),
        ),
    )
    full_access_gateway = base_gateway.for_run(
        exposed_names=("write_file",),
        permission_snapshot=PermissionSnapshot(
            level="full-access",
            exec_shell=resolve_exec_shell("auto"),
        ),
    )

    refused, written = await asyncio.gather(
        read_only_gateway.call(
            _call("write_file", {"path": "read-only.txt", "content": "blocked"}, call_id="ro")
        ),
        full_access_gateway.call(
            _call("write_file", {"path": "full-access.txt", "content": "allowed"}, call_id="full")
        ),
    )

    assert refused.status == "refused"
    assert written.status == "success"
    assert not (workspace / "read-only.txt").exists()
    assert (workspace / "full-access.txt").read_text(encoding="utf-8") == "allowed"


@pytest.mark.asyncio
async def test_shared_exec_tool_uses_each_run_workspace_for_facts_and_execution(
    tmp_path: Path,
) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    host = _RecordingExecHost()
    catalog = BuiltInToolCatalog(exec_host=host)
    snapshot = PermissionSnapshot(level="full-access", exec_shell=host.resolved_shell)
    gateway_a = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_a, exec_host=host),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=workspace_a,
        ),
    )
    gateway_b = ToolGateway(
        catalog=catalog,
        tool_context=ToolRunContext(workspace=workspace_b, exec_host=host),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=workspace_b,
        ),
    )

    results = await asyncio.gather(
        gateway_a.call(_call("exec", {"command": "Get-Location"}, call_id="exec-a")),
        gateway_b.call(_call("exec", {"command": "Get-Location"}, call_id="exec-b")),
    )

    assert [result.status for result in results] == ["success", "success"]
    assert host.inspected == [workspace_a.resolve(), workspace_b.resolve()]
    assert host.executed == [workspace_a.resolve(), workspace_b.resolve()]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["write_file", "edit_file"])
async def test_contextual_refusal_checks_the_frozen_mutation_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    protected = tmp_path / ".omni" / "restore" / "index.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("original", encoding="utf-8")
    catalog = BuiltInToolCatalog()
    gateway = ToolGateway(catalog=catalog, tool_context=ToolRunContext(workspace=tmp_path))
    tool = next(tool for tool in catalog.tools if tool.name == tool_name)
    assert isinstance(tool, (WriteFileTool, EditFileTool))
    prepare = tool.collect_invocation_facts_for_context

    async def prepare_with_changed_path(
        arguments: dict[str, object], *, context: ToolRunContext
    ) -> ToolInvocationFacts:
        # Model a directory link changing after preparation freezes its write target.
        facts = await prepare({**arguments, "path": str(protected)}, context=context)
        return ToolInvocationFacts(
            tool_name=tool_name,
            normalized_arguments=arguments,
            file_accesses=facts.file_accesses,
        )

    monkeypatch.setattr(tool, "collect_invocation_facts_for_context", prepare_with_changed_path)
    arguments: dict[str, object] = {"path": "notes.txt"}
    if tool_name == "write_file":
        arguments["content"] = "overwritten"
    else:
        arguments.update(old_text="original", new_text="overwritten")
    recorder = _Recorder()
    result = await gateway.call(
        _call(tool_name, arguments, call_id="protected"),
        file_mutation_recorder=recorder,
        run_token=uuid4(),
    )
    assert result.status == "refused"
    assert protected.read_text(encoding="utf-8") == "original"
    assert recorder.before == []
