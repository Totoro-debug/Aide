from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from myclaw.tools.base import BaseTool
from myclaw.tools.files.edit_file import EditFileTool
from myclaw.tools.files.read_file import ReadFileTool
from myclaw.tools.files.write_file import WriteFileTool
from myclaw.tools.gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
)
from tests.fixtures import SingleToolGateway

type FileToolType = type[ReadFileTool] | type[WriteFileTool] | type[EditFileTool]


def _call(name: str, arguments: dict[str, object], *, call_id: str = "call_1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def _gateway(
    *tools: BaseTool,
    confirmation: ConfirmationRequester | None = None,
) -> SingleToolGateway:
    return SingleToolGateway(tools, confirmation=confirmation)


def test_gateway_exports_the_new_path_and_line_contract(workspace: Path) -> None:
    identity = workspace
    gateway = _gateway(ReadFileTool(workspace=identity))

    assert tuple(gateway.schemas) == (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": ReadFileTool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative or absolute file path.",
                            "minLength": 1,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "One-based first line.",
                            "minimum": 1,
                            "default": 1,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum lines to return.",
                            "minimum": 1,
                            "maximum": 10000,
                            "default": 2000,
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_read_file_uses_one_based_windows_and_preserves_line_endings(
    workspace: Path,
) -> None:
    target = workspace / "mixed.txt"
    target.write_bytes(b"one\r\ntwo\nthree\r")
    gateway = _gateway(ReadFileTool(workspace=workspace))

    first_window = await gateway.call(_call("read_file", {"path": "mixed.txt", "limit": 2}))
    second_line = await gateway.call(
        _call("read_file", {"path": "mixed.txt", "offset": 2, "limit": 1}, call_id="call_2")
    )
    out_of_range = await gateway.call(
        _call("read_file", {"path": "mixed.txt", "offset": 99}, call_id="call_3")
    )

    assert (first_window.status, first_window.content) == ("success", "one\r\ntwo\n")
    assert (second_line.status, second_line.content) == ("success", "two\n")
    assert (out_of_range.status, out_of_range.content) == ("success", "")


@pytest.mark.asyncio
async def test_read_file_requires_strict_utf8_but_allows_empty_and_nul_content(
    workspace: Path,
) -> None:
    identity = workspace
    empty = workspace / "empty.txt"
    empty.write_bytes(b"")
    nul = workspace / "nul.txt"
    nul.write_bytes(b"before\x00after\n")
    invalid = workspace / "invalid.txt"
    invalid.write_bytes(b"\xff")
    gateway = _gateway(ReadFileTool(workspace=identity))

    empty_result = await gateway.call(_call("read_file", {"path": "empty.txt"}))
    nul_result = await gateway.call(_call("read_file", {"path": "nul.txt"}, call_id="call_nul"))
    invalid_result = await gateway.call(
        _call("read_file", {"path": "invalid.txt"}, call_id="call_invalid")
    )

    assert (empty_result.status, empty_result.content) == ("success", "")
    assert (nul_result.status, nul_result.content) == ("success", "before\x00after\n")
    assert invalid_result.status == "error"
    assert "UTF-8" in invalid_result.content


@pytest.mark.asyncio
async def test_write_file_creates_parents_and_writes_exact_utf8_bytes(workspace: Path) -> None:
    identity = workspace
    gateway = _gateway(WriteFileTool(workspace=identity))
    content = "line one\r\n中文\n"

    result = await gateway.call(
        _call("write_file", {"path": "nested/output.txt", "content": content})
    )
    empty_result = await gateway.call(
        _call(
            "write_file",
            {"path": "nested/empty.txt", "content": ""},
            call_id="call_empty_write",
        )
    )

    assert result.status == "success"
    assert (workspace / "nested" / "output.txt").read_bytes() == content.encode("utf-8")
    assert empty_result.status == "success"
    assert (workspace / "nested" / "empty.txt").read_bytes() == b""


@pytest.mark.asyncio
async def test_edit_file_rejects_zero_and_ambiguous_single_replacements_without_writing(
    workspace: Path,
) -> None:
    identity = workspace
    target = workspace / "notes.txt"
    target.write_bytes(b"same\r\nsame\n")
    gateway = _gateway(EditFileTool(workspace=identity))

    zero = await gateway.call(
        _call(
            "edit_file",
            {"path": "notes.txt", "old_text": "missing", "new_text": "new"},
        )
    )
    ambiguous = await gateway.call(
        _call(
            "edit_file",
            {"path": "notes.txt", "old_text": "same", "new_text": "new"},
            call_id="call_ambiguous",
        )
    )

    assert zero.status == "error"
    assert ambiguous.status == "error"
    assert "match" in zero.content.lower()
    assert "ambiguous" in ambiguous.content.lower()
    assert target.read_bytes() == b"same\r\nsame\n"

    replaced = await gateway.call(
        _call(
            "edit_file",
            {
                "path": "notes.txt",
                "old_text": "same",
                "new_text": "new",
                "replace_all": True,
            },
            call_id="call_replace_all",
        )
    )

    assert replaced.status == "success"
    assert target.read_bytes() == b"new\r\nnew\n"


@pytest.mark.asyncio
async def test_workspace_state_and_absolute_internal_paths_use_host_permissions(
    workspace: Path,
) -> None:
    identity = workspace
    state_file = workspace / ".myclaw" / "sessions" / "state.txt"
    state_file.parent.mkdir(parents=True)
    state_file.write_bytes(b"workspace state")
    gateway = _gateway(ReadFileTool(workspace=identity))

    relative = await gateway.call(_call("read_file", {"path": ".myclaw/sessions/state.txt"}))
    absolute = await gateway.call(
        _call("read_file", {"path": str(state_file)}, call_id="call_absolute")
    )

    assert (relative.status, relative.content) == ("success", "workspace state")
    assert (absolute.status, absolute.content) == ("success", "workspace state")


@pytest.mark.asyncio
async def test_read_file_skill_root_requires_model_file_confirmation(
    workspace: Path,
    tmp_path: Path,
) -> None:
    identity = workspace
    skill_root = tmp_path / "agent-home" / "skills"
    skill_file = skill_root / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        ReadFileTool(workspace=identity, skill_root=skill_root),
        confirmation=approve,
    )

    result = await gateway.call(_call("read_file", {"path": str(skill_file)}))

    assert (result.status, result.content) == ("success", "---\nname: review\n---\nbody\n")
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_read_file_missing_skill_target_requires_confirmation(
    workspace: Path,
    tmp_path: Path,
) -> None:
    identity = workspace
    skill_root = tmp_path / "agent-home" / "skills"
    missing = skill_root / "review" / "SKILL.md"
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = _gateway(
        ReadFileTool(workspace=identity, skill_root=skill_root),
        confirmation=decline,
    )

    result = await gateway.call(_call("read_file", {"path": str(missing)}))

    assert result.status == "refused"
    assert len(requests) == 1
    assert not skill_root.exists()


@pytest.mark.asyncio
async def test_read_file_keeps_other_agent_home_paths_confirmed(
    workspace: Path,
    tmp_path: Path,
) -> None:
    identity = workspace
    agent_home = tmp_path / "agent-home"
    config = agent_home / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"[runtime]\nmax_tool_result_chars = 4096\n")
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        ReadFileTool(workspace=identity, skill_root=agent_home / "skills"),
        confirmation=approve,
    )

    result = await gateway.call(_call("read_file", {"path": str(config)}))

    assert (result.status, result.content) == (
        "success",
        "[runtime]\nmax_tool_result_chars = 4096\n",
    )
    assert len(requests) == 1
    assert "outside the Workspace" in requests[0].reason


@pytest.mark.asyncio
async def test_read_file_skill_root_escape_requires_confirmation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agent_home = tmp_path / "agent-home"
    external = tmp_path / "external"
    workspace.mkdir()
    skill_root = agent_home / "skills"
    skill_root.mkdir(parents=True)
    external.mkdir()
    outside = external / "SKILL.md"
    outside.write_bytes(b"outside skill\n")
    escape = skill_root / "escape"
    try:
        escape.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory links unavailable: {error}")
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        ReadFileTool(workspace=workspace, skill_root=skill_root),
        confirmation=approve,
    )

    result = await gateway.call(_call("read_file", {"path": str(escape / "SKILL.md")}))

    assert (result.status, result.content) == ("success", "outside skill\n")
    assert len(requests) == 1
    assert "outside the Workspace" in requests[0].reason


@pytest.mark.asyncio
async def test_external_targets_require_confirmation_and_bind_the_exact_call(
    workspace: Path,
    tmp_path: Path,
) -> None:
    identity = workspace
    external = tmp_path / "external.txt"
    external.write_bytes(b"outside\n")
    gateway = _gateway(ReadFileTool(workspace=identity))
    requested = str(Path("..") / external.name)
    call = _call("read_file", {"path": requested})

    missing = await gateway.call(call)

    assert missing.status == "refused"
    assert missing.confirmation is not None
    request = missing.confirmation.request
    assert "outside the Workspace" in request.reason
    assert request.details["path"] == requested

    async def approve(current: ConfirmationRequest) -> ConfirmationDecision:
        return (
            "approved"
            if current.tool_call_id == request.tool_call_id
            and current.tool_name == request.tool_name
            else "declined"
        )

    approved = await gateway.call(
        call,
        confirmation=approve,
    )

    assert (approved.status, approved.content) == ("success", "outside\n")
    assert approved.confirmation is not None
    assert approved.confirmation.decision == "approved"


@pytest.mark.asyncio
async def test_all_file_mutations_support_confirmed_external_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    identity = workspace
    read_target = external / "read.txt"
    read_target.write_text("read outside", encoding="utf-8")
    edit_target = external / "edit.txt"
    edit_target.write_text("before", encoding="utf-8")
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        ReadFileTool(workspace=identity),
        WriteFileTool(workspace=identity),
        EditFileTool(workspace=identity),
        confirmation=approve,
    )

    read_result = await gateway.call(_call("read_file", {"path": str(read_target)}))
    write_result = await gateway.call(
        _call("write_file", {"path": str(external / "written.txt"), "content": "written"})
    )
    edit_result = await gateway.call(
        _call(
            "edit_file",
            {"path": str(edit_target), "old_text": "before", "new_text": "after"},
        )
    )

    assert (read_result.status, read_result.content) == ("success", "read outside")
    assert write_result.status == "success"
    assert edit_result.status == "success"
    assert (external / "written.txt").read_bytes() == b"written"
    assert edit_target.read_text(encoding="utf-8") == "after"
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_expected_filesystem_failures_keep_operation_context_and_original_error(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = workspace
    target = workspace / "read-error.txt"
    target.write_text("content", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_read(path: Path) -> bytes:
        if path == target:
            raise OSError("raw read failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    gateway = _gateway(ReadFileTool(workspace=identity))

    result = await gateway.call(_call("read_file", {"path": "read-error.txt"}))

    assert result.status == "error"
    assert "Read File" in result.content
    assert "raw read failure" in result.content


@pytest.mark.parametrize(
    ("tool_type", "tool_name", "arguments", "operation"),
    (
        (ReadFileTool, "read_file", {"path": "resolve-error.txt"}, "Read File"),
        (
            WriteFileTool,
            "write_file",
            {"path": "resolve-error.txt", "content": "content"},
            "Write File",
        ),
        (
            EditFileTool,
            "edit_file",
            {"path": "resolve-error.txt", "old_text": "old", "new_text": "new"},
            "Edit File",
        ),
    ),
)
@pytest.mark.parametrize("failure_stage", ("target", "containment"))
@pytest.mark.asyncio
async def test_path_failures_keep_resolution_context_and_original_os_error(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_type: FileToolType,
    tool_name: str,
    arguments: dict[str, object],
    operation: str,
    failure_stage: str,
) -> None:
    target = workspace / "resolve-error.txt"
    original_resolve = Path.resolve
    workspace_resolution_count = 0
    raw_error = f"raw {failure_stage} failure"

    def fail_resolve(path: Path, strict: bool = False) -> Path:
        nonlocal workspace_resolution_count
        if path == workspace:
            workspace_resolution_count += 1
            if failure_stage == "containment" and workspace_resolution_count == 2:
                raise OSError(raw_error)
        if failure_stage == "target" and path == target:
            raise OSError(raw_error)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    gateway = _gateway(tool_type(workspace=workspace))

    result = await gateway.call(_call(tool_name, arguments))

    assert (result.status, result.content) == (
        "error",
        f"{operation} path could not be resolved: {raw_error}",
    )


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_external_confirmation_propagates(
    workspace: Path,
    tmp_path: Path,
) -> None:
    request_seen = asyncio.Event()
    release = asyncio.Event()
    external = tmp_path / "cancel.txt"
    external.write_text("cancel", encoding="utf-8")

    async def wait_for_decision(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        request_seen.set()
        await release.wait()
        return "approved"

    gateway = _gateway(
        WriteFileTool(workspace=workspace),
        confirmation=wait_for_decision,
    )
    task = asyncio.create_task(
        gateway.call(
            _call(
                "write_file",
                {"path": str(external), "content": "must not be written"},
            )
        )
    )
    await request_seen.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert external.read_text(encoding="utf-8") == "cancel"
