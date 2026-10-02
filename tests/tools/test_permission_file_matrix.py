from __future__ import annotations

import asyncio
import json
import string
import subprocess
from pathlib import Path
from typing import Any

import pytest

from myclaw.agent.tools.base import BaseTool, ToolError
from myclaw.agent.tools.core.edit_file import EditFileTool
from myclaw.agent.tools.core.exec_host import resolve_exec_shell
from myclaw.agent.tools.core.glob import GlobTool
from myclaw.agent.tools.core.grep import GrepTool
from myclaw.agent.tools.core.list_dir import ListDirTool
from myclaw.agent.tools.core.read_file import ReadFileTool
from myclaw.agent.tools.core.write_file import WriteFileTool
from myclaw.agent.tools.permission import (
    PermissionContext,
    PermissionSnapshot,
    RuntimePermissionControl,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    ToolPermissionLevel,
    ToolPermissionPolicy,
)
from myclaw.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)


def _snapshot(level: ToolPermissionLevel) -> PermissionSnapshot:
    return PermissionSnapshot(level=level, exec_shell=resolve_exec_shell("auto"))


def _call(name: str, arguments: dict[str, object], *, call_id: str = "call-1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def _gateway(
    workspace: Path,
    level: ToolPermissionLevel,
    tool: BaseTool,
    *,
    policy: ToolPermissionPolicy | None = None,
) -> ToolGateway:
    snapshot = _snapshot(level)
    context = PermissionContext.from_snapshot(
        snapshot,
        workspace_root=workspace,
    )
    return ToolGateway._for_memory(
        (tool,),
        permission_policy=policy,
        permission_context=context,
    )


def _prepare_file_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    workspace_file = workspace / "inside.txt"
    outside_file = outside / "outside.txt"
    workspace_file.write_text("needle\n", encoding="utf-8")
    outside_file.write_text("needle\n", encoding="utf-8")
    return workspace, outside, workspace_file, outside_file


def _read_tool_and_arguments(
    tool_name: str,
    *,
    workspace: Path,
    target_root: Path,
    target_file: Path,
) -> tuple[BaseTool, dict[str, object], dict[str, object]]:
    arguments: dict[str, object]
    if tool_name == "read_file":
        arguments = {"path": str(target_file)}
        return (
            ReadFileTool(workspace=workspace),
            arguments,
            {
                **arguments,
                "offset": 1,
                "limit": 2000,
            },
        )
    if tool_name == "list_dir":
        arguments = {"path": str(target_root)}
        return (
            ListDirTool(workspace=workspace),
            arguments,
            {
                **arguments,
                "recursive": False,
                "max_entries": 200,
            },
        )
    if tool_name == "glob":
        arguments = {"pattern": "*.txt", "path": str(target_root)}
        return (
            GlobTool(workspace=workspace),
            arguments,
            {
                **arguments,
                "head_limit": 200,
                "offset": 0,
                "kind": "files",
            },
        )
    if tool_name == "grep":
        arguments = {
            "pattern": "needle",
            "path": str(target_root),
            "fixed_string": True,
        }
        return (
            GrepTool(workspace=workspace),
            arguments,
            {
                **arguments,
                "glob": None,
                "type": None,
                "output_mode": "content",
                "ignore_case": False,
                "context": 0,
                "head_limit": 0,
                "offset": 0,
            },
        )
    raise AssertionError(f"unexpected read tool: {tool_name}")


def _record_execution(
    monkeypatch: pytest.MonkeyPatch,
    tool: BaseTool,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    execute = tool.execute_authorized

    async def recording_execute(
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        calls.append(arguments)
        return await execute(arguments, authorization)

    monkeypatch.setattr(tool, "execute_authorized", recording_execute)
    return calls


class _BusinessRefusalTool(BaseTool):
    name = "business_refusal"
    description = "Refuse one prepared invocation before authorization."

    def __init__(self) -> None:
        self.calls = 0

    def refusal_reason(self) -> str:
        return "business refusal"

    async def execute(self) -> str:
        self.calls += 1
        return "unexpected"


class _UnavailableCapabilityTool(BaseTool):
    name = "unavailable_capability"
    description = "Report a missing capability while collecting invocation facts."

    def __init__(self) -> None:
        self.calls = 0

    async def collect_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        del prepared_arguments
        raise ToolError("capability unavailable")

    async def execute(self) -> str:
        self.calls += 1
        return "unexpected"


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    created = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(link), str(target)),
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"directory junctions unavailable: {created.stderr.strip()}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "level", "location"),
    [
        (tool_name, level, location)
        for tool_name in ("read_file", "list_dir", "glob", "grep")
        for level in ("read-only", "workspace-write", "full-access")
        for location in ("inside", "outside")
    ],
)
async def test_foreground_file_read_matrix_requests_only_external_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    level: ToolPermissionLevel,
    location: str,
) -> None:
    workspace, outside, workspace_file, outside_file = _prepare_file_fixture(tmp_path)
    target_root = workspace if location == "inside" else outside
    target_file = workspace_file if location == "inside" else outside_file
    tool, arguments, normalized_arguments = _read_tool_and_arguments(
        tool_name,
        workspace=workspace,
        target_root=target_root,
        target_file=target_file,
    )
    executions = _record_execution(monkeypatch, tool)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = _gateway(workspace, level, tool)
    result = await gateway.call(_call(tool_name, arguments), confirmation=decline)

    expected_confirmation = location == "outside" and level != "full-access"
    assert len(requests) == int(expected_confirmation)
    assert result.status == ("refused" if expected_confirmation else "success")
    assert len(executions) == int(not expected_confirmation)
    if expected_confirmation:
        assert requests[0].tool_call_id == "call-1"
        assert requests[0].tool_name == tool_name
        assert requests[0].details == normalized_arguments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "level", "location"),
    [
        (tool_name, level, location)
        for tool_name in ("write_file", "edit_file")
        for level in ("read-only", "workspace-write", "full-access")
        for location in ("inside", "outside")
    ],
)
async def test_foreground_file_write_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    level: ToolPermissionLevel,
    location: str,
) -> None:
    workspace, _outside, workspace_file, outside_file = _prepare_file_fixture(tmp_path)
    target = workspace_file if location == "inside" else outside_file
    if tool_name == "write_file":
        arguments: dict[str, object] = {"path": str(target), "content": "written\n"}
        normalized_arguments = arguments
        tool: BaseTool = WriteFileTool(workspace=workspace)
    else:
        arguments = {
            "path": str(target),
            "old_text": "needle",
            "new_text": "edited",
        }
        normalized_arguments = {**arguments, "replace_all": False}
        tool = EditFileTool(workspace=workspace)
    original_content = target.read_text(encoding="utf-8")
    executions = _record_execution(monkeypatch, tool)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = _gateway(workspace, level, tool)
    result = await gateway.call(_call(tool_name, arguments), confirmation=decline)

    expected_confirmation = level == "read-only" or (
        level == "workspace-write" and location == "outside"
    )
    assert len(requests) == int(expected_confirmation)
    assert result.status == ("refused" if expected_confirmation else "success")
    assert len(executions) == int(not expected_confirmation)
    if expected_confirmation:
        assert requests[0].tool_call_id == "call-1"
        assert requests[0].tool_name == tool_name
        assert requests[0].details == normalized_arguments
        assert target.read_text(encoding="utf-8") == original_content
        if level == "read-only" and location == "outside":
            assert "Write access requires confirmation in read-only mode." in requests[0].reason
            assert "resolves outside the Workspace" in requests[0].reason
            assert requests[0].reason.count("requires confirmation") == 2


class _DirectAuthorizationSession:
    def initial_decision(self) -> str:
        return "direct"

    async def authorize_network_target(
        self,
        target: object,
        resolved_addresses: tuple[str, ...],
    ) -> None:
        del target, resolved_addresses


class _RecordingPolicy(ToolPermissionPolicy):
    def __init__(self) -> None:
        self.facts: ToolInvocationFacts | None = None

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        del context
        self.facts = facts
        return _DirectAuthorizationSession()  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_file_facts_use_canonical_host_paths_and_explicit_roles(tmp_path: Path) -> None:
    workspace, _outside, workspace_file, _outside_file = _prepare_file_fixture(tmp_path)
    policy = _RecordingPolicy()
    gateway = _gateway(
        workspace,
        "workspace-write",
        ReadFileTool(workspace=workspace),
        policy=policy,
    )

    result = await gateway.call(_call("read_file", {"path": "inside.txt"}))

    assert result.status == "success"
    assert policy.facts is not None
    assert len(policy.facts.file_accesses) == 1
    access = policy.facts.file_accesses[0]
    assert access.path == workspace_file.resolve()
    assert access.role == "read"
    assert access.base == workspace.resolve()


@pytest.mark.asyncio
async def test_edit_file_reports_both_read_and_write_facts(tmp_path: Path) -> None:
    workspace, _outside, workspace_file, _outside_file = _prepare_file_fixture(tmp_path)
    policy = _RecordingPolicy()
    gateway = _gateway(
        workspace,
        "workspace-write",
        EditFileTool(workspace=workspace),
        policy=policy,
    )

    result = await gateway.call(
        _call(
            "edit_file",
            {"path": "inside.txt", "old_text": "needle", "new_text": "edited"},
        )
    )

    assert result.status == "success"
    assert policy.facts is not None
    assert [(access.path, access.role) for access in policy.facts.file_accesses] == [
        (workspace_file.resolve(), "read"),
        (workspace_file.resolve(), "write"),
    ]


@pytest.mark.asyncio
async def test_missing_write_ancestor_is_canonicalized_from_existing_host_parent(
    tmp_path: Path,
) -> None:
    workspace, _outside, _workspace_file, _outside_file = _prepare_file_fixture(tmp_path)
    policy = _RecordingPolicy()
    target = workspace / "new" / "deep" / "target.txt"
    gateway = _gateway(
        workspace,
        "workspace-write",
        WriteFileTool(workspace=workspace),
        policy=policy,
    )

    result = await gateway.call(_call("write_file", {"path": str(target), "content": "created\n"}))

    assert result.status == "success"
    assert target.read_text(encoding="utf-8") == "created\n"
    assert policy.facts is not None
    assert policy.facts.file_accesses[0].path == target.resolve()


@pytest.mark.asyncio
async def test_sibling_prefix_is_external_instead_of_a_string_prefix_match(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sibling = tmp_path / "workspace-copy"
    workspace.mkdir()
    sibling.mkdir()
    target = sibling / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _gateway(
        workspace,
        "workspace-write",
        ReadFileTool(workspace=workspace),
    ).call(_call("read_file", {"path": str(target)}), confirmation=decline)

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {"path": str(target), "offset": 1, "limit": 2000}


@pytest.mark.asyncio
async def test_linked_skill_root_and_missing_write_descendant_remain_external(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    skill_root = tmp_path / "agent-home" / "skills"
    workspace.mkdir()
    skill_root.mkdir(parents=True)
    skill_file = skill_root / "SKILL.md"
    skill_file.write_text("skill", encoding="utf-8")
    linked = workspace / "linked-skill"
    _create_directory_link(linked, skill_root)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    read_result = await _gateway(
        workspace,
        "workspace-write",
        ReadFileTool(workspace=workspace, skill_root=skill_root),
    ).call(_call("read_file", {"path": str(linked / "SKILL.md")}), confirmation=decline)
    missing_target = linked / "new" / "note.txt"
    write_result = await _gateway(
        workspace,
        "workspace-write",
        WriteFileTool(workspace=workspace),
    ).call(
        _call("write_file", {"path": str(missing_target), "content": "blocked"}),
        confirmation=decline,
    )

    assert read_result.status == write_result.status == "refused"
    assert len(requests) == 2
    assert all("outside the Workspace" in request.reason for request in requests)
    assert not missing_target.exists()


@pytest.mark.asyncio
async def test_windows_path_case_uses_host_case_insensitive_containment(tmp_path: Path) -> None:
    workspace, _outside, workspace_file, _outside_file = _prepare_file_fixture(tmp_path)
    case_variant = Path(str(workspace_file).swapcase())

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        raise AssertionError(f"unexpected confirmation: {request}")

    result = await _gateway(
        workspace,
        "workspace-write",
        ReadFileTool(workspace=workspace),
    ).call(_call("read_file", {"path": str(case_variant)}), confirmation=unexpected_confirmation)

    assert (result.status, result.content) == (
        "success",
        workspace_file.read_bytes().decode("utf-8"),
    )


@pytest.mark.asyncio
async def test_windows_different_drive_is_external(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_roots = [
        Path(f"{letter}:\\")
        for letter in string.ascii_uppercase
        if letter.casefold() != workspace.drive[:1].casefold() and Path(f"{letter}:\\").exists()
    ]
    if not other_roots:
        pytest.skip("a second Windows filesystem drive is unavailable")
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    target = other_roots[0]
    result = await _gateway(
        workspace,
        "workspace-write",
        ListDirTool(workspace=workspace),
    ).call(_call("list_dir", {"path": str(target)}), confirmation=decline)

    assert result.status == "refused"
    assert len(requests) == 1
    assert "outside the Workspace" in requests[0].reason


@pytest.mark.asyncio
async def test_windows_reachable_unc_path_is_classified_by_real_host_semantics(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "inside.txt"
    target.write_text("inside", encoding="utf-8")
    resolved = target.resolve()
    drive = resolved.drive.rstrip(":")
    unc_target = Path(rf"\\localhost\{drive}$\{resolved.relative_to(resolved.anchor)}")
    if not unc_target.exists():
        pytest.skip("the local administrative UNC share is unavailable")
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _gateway(
        workspace,
        "workspace-write",
        ReadFileTool(workspace=workspace),
    ).call(_call("read_file", {"path": str(unc_target)}), confirmation=decline)

    assert result.status == "refused"
    assert len(requests) == 1
    assert "outside the Workspace" in requests[0].reason


@pytest.mark.asyncio
async def test_concurrent_run_gateways_do_not_share_permission_snapshots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = WriteFileTool(workspace=workspace)
    generation_gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(workspace_root=workspace),
    )
    read_only = generation_gateway.for_run(
        exposed_names=(tool.name,),
        permission_snapshot=_snapshot("read-only"),
    )
    workspace_write = generation_gateway.for_run(
        exposed_names=(tool.name,),
        permission_snapshot=_snapshot("workspace-write"),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        await asyncio.sleep(0)
        return "declined"

    denied_target = workspace / "denied.txt"
    allowed_target = workspace / "allowed.txt"
    denied, allowed = await asyncio.gather(
        read_only.call(
            _call(
                "write_file",
                {"path": str(denied_target), "content": "denied"},
                call_id="read-only-call",
            ),
            confirmation=decline,
        ),
        workspace_write.call(
            _call(
                "write_file",
                {"path": str(allowed_target), "content": "allowed"},
                call_id="workspace-write-call",
            ),
            confirmation=decline,
        ),
    )

    assert denied.status == "refused"
    assert allowed.status == "success"
    assert [request.tool_call_id for request in requests] == ["read-only-call"]
    assert not denied_target.exists()
    assert allowed_target.read_text(encoding="utf-8") == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
@pytest.mark.parametrize(
    "failure",
    (
        "invalid_json",
        "unknown_tool",
        "schema",
        "encoding",
        "absolute_glob",
        "io_error",
        "business_refusal",
        "missing_capability",
    ),
)
async def test_hard_errors_are_level_invariant_and_never_confirm(
    tmp_path: Path,
    level: ToolPermissionLevel,
    failure: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expected_status = "error"
    expected_content: str
    if failure == "invalid_json":
        tool: BaseTool = ReadFileTool(workspace=workspace)
        call = ModelToolCall(id="hard-error", name=tool.name, arguments="{")
        expected_content = "could not be parsed"
    elif failure == "unknown_tool":
        tool = ReadFileTool(workspace=workspace)
        call = ModelToolCall(id="hard-error", name="unknown", arguments="{}")
        expected_content = "not available"
    elif failure == "schema":
        tool = ReadFileTool(workspace=workspace)
        call = _call(tool.name, {"path": ""}, call_id="hard-error")
        expected_content = "$.path: must contain at least 1 characters"
    elif failure == "encoding":
        target = workspace / "invalid.bin"
        target.write_bytes(b"\xff")
        tool = ReadFileTool(workspace=workspace)
        call = _call(tool.name, {"path": str(target)}, call_id="hard-error")
        expected_content = "not valid UTF-8"
    elif failure == "absolute_glob":
        tool = GlobTool(workspace=workspace)
        call = _call(tool.name, {"pattern": "C:\\absolute\\*.txt"}, call_id="hard-error")
        expected_content = "relative"
    elif failure == "io_error":
        tool = ReadFileTool(workspace=workspace)
        call = _call(tool.name, {"path": "missing.txt"}, call_id="hard-error")
        expected_content = "Read File failed"
    elif failure == "business_refusal":
        tool = _BusinessRefusalTool()
        call = _call(tool.name, {}, call_id="hard-error")
        expected_status = "refused"
        expected_content = "business refusal"
    else:
        assert failure == "missing_capability"
        tool = _UnavailableCapabilityTool()
        call = _call(tool.name, {}, call_id="hard-error")
        expected_content = "capability unavailable"
    requests: list[ConfirmationRequest] = []

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _gateway(workspace, level, tool).call(
        call,
        confirmation=unexpected_confirmation,
    )

    assert result.status == expected_status
    assert expected_content in result.content
    assert requests == []
    assert getattr(tool, "calls", 0) == 0


@pytest.mark.asyncio
async def test_full_access_does_not_skip_invalid_utf8_errors(tmp_path: Path) -> None:
    workspace, _outside, _workspace_file, _outside_file = _prepare_file_fixture(tmp_path)
    invalid = workspace / "invalid.bin"
    invalid.write_bytes(b"\xff")
    gateway = _gateway(workspace, "full-access", ReadFileTool(workspace=workspace))

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        raise AssertionError(f"unexpected confirmation: {request}")

    result = await gateway.call(
        _call("read_file", {"path": str(invalid)}),
        confirmation=unexpected_confirmation,
    )

    assert result.status == "error"
    assert "not valid UTF-8" in result.content


def test_runtime_permission_control_keeps_configured_level_across_changes() -> None:
    control = RuntimePermissionControl("workspace-write")
    assert control.configured() == "workspace-write"
    assert control.current() == "workspace-write"

    control.select("read-only")

    assert control.configured() == "workspace-write"
    assert control.current() == "read-only"
    assert control.snapshot(resolve_exec_shell("auto")).level == "read-only"
