from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from myclaw.tools.base import BaseTool
from myclaw.tools.files.glob import GlobTool
from myclaw.tools.files.list_dir import ListDirTool
from myclaw.tools.gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
)
from tests.fixtures import SingleToolGateway


def _call(name: str, arguments: dict[str, object], *, call_id: str = "call_1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def _gateway(
    *tools: BaseTool,
    confirmation: ConfirmationRequester | None = None,
) -> SingleToolGateway:
    return SingleToolGateway(tools, confirmation=confirmation)


@pytest.mark.asyncio
async def test_list_dir_is_stable_recursive_hidden_state_aware_and_limited(
    workspace: Path,
) -> None:
    (workspace / "z.txt").write_text("z", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    (workspace / ".hidden").write_text("hidden", encoding="utf-8")
    (workspace / "nested" / "child.txt").parent.mkdir()
    (workspace / "nested" / "child.txt").write_text("child", encoding="utf-8")
    (workspace / ".myclaw" / "state.txt").parent.mkdir()
    (workspace / ".myclaw" / "state.txt").write_text("state", encoding="utf-8")
    (workspace / "node_modules" / "ignored.txt").parent.mkdir()
    (workspace / "node_modules" / "ignored.txt").write_text("ignored", encoding="utf-8")
    (workspace / ".gitignore").write_text("z.txt\n", encoding="utf-8")

    tool = ListDirTool(workspace=workspace)
    gateway = _gateway(tool)

    shallow = await gateway.call(_call("list_dir", {}))
    recursive = await gateway.call(_call("list_dir", {"recursive": True}, call_id="call_recursive"))
    limited = await gateway.call(_call("list_dir", {"max_entries": 2}, call_id="call_limited"))

    assert shallow.status == "success"
    assert shallow.content == ".gitignore\n.hidden\n.myclaw/\na.txt\nnested/\nz.txt"
    assert recursive.status == "success"
    assert recursive.content == (
        ".gitignore\n.hidden\n.myclaw/\n.myclaw/state.txt\na.txt\nnested/\nnested/child.txt\nz.txt"
    )
    assert limited.content == ".gitignore\n.hidden"


@pytest.mark.asyncio
async def test_list_dir_defaults_to_200_entries(workspace: Path) -> None:
    for index in range(201):
        (workspace / f"entry-{index:03}.txt").write_text("entry", encoding="utf-8")
    gateway = _gateway(ListDirTool(workspace=workspace))

    result = await gateway.call(_call("list_dir", {}))

    assert result.status == "success"
    assert result.content.splitlines() == [f"entry-{index:03}.txt" for index in range(200)]


@pytest.mark.asyncio
async def test_glob_supports_pattern_dialects_kinds_and_pagination(workspace: Path) -> None:
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    (workspace / "b.PY").write_text("b", encoding="utf-8")
    (workspace / "Sub" / "c.txt").parent.mkdir()
    (workspace / "Sub" / "c.txt").write_text("c", encoding="utf-8")
    (workspace / "Sub" / "Deep" / "d.txt").parent.mkdir()
    (workspace / "Sub" / "Deep" / "d.txt").write_text("d", encoding="utf-8")

    tool = GlobTool(workspace=workspace)
    gateway = _gateway(tool)

    simple = await gateway.call(_call("glob", {"pattern": "*.txt", "head_limit": 0}))
    directory = await gateway.call(
        _call("glob", {"pattern": "Sub\\*.txt", "head_limit": 0}, call_id="call_directory")
    )
    wrong_case_directory = await gateway.call(
        _call("glob", {"pattern": "sub/*.txt", "head_limit": 0}, call_id="call_case")
    )
    dirs = await gateway.call(
        _call("glob", {"pattern": "*", "kind": "dirs", "head_limit": 0}, call_id="call_dirs")
    )
    paged = await gateway.call(
        _call(
            "glob",
            {"pattern": "*.txt", "offset": 1, "head_limit": 1},
            call_id="call_paged",
        )
    )
    windows_case = await gateway.call(
        _call("glob", {"pattern": "*.TXT", "head_limit": 0}, call_id="call_host_case")
    )

    assert simple.content == "Sub/Deep/d.txt\nSub/c.txt\na.txt"
    assert directory.content == "Sub/c.txt"
    assert wrong_case_directory.content == ""
    assert dirs.content == "Sub/\nSub/Deep/"
    assert paged.content == "Sub/c.txt"
    expected_host_case = "Sub/Deep/d.txt\nSub/c.txt\na.txt" if os.name == "nt" else ""
    assert windows_case.content == expected_host_case


@pytest.mark.asyncio
async def test_glob_enforces_head_limit_and_offset_boundaries(workspace: Path) -> None:
    gateway = _gateway(GlobTool(workspace=workspace))

    maximum = await gateway.call(
        _call("glob", {"pattern": "*", "head_limit": 1000}, call_id="call_maximum")
    )
    over_maximum = await gateway.call(
        _call("glob", {"pattern": "*", "head_limit": 1001}, call_id="call_over_maximum")
    )
    negative_offset = await gateway.call(
        _call("glob", {"pattern": "*", "offset": -1}, call_id="call_negative_offset")
    )

    assert maximum.status == "success"
    assert over_maximum.status == "error"
    assert negative_offset.status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    ("/absolute/*.txt", "C:\\absolute\\*.txt", "\\\\server\\share\\*.txt"),
)
async def test_glob_rejects_absolute_drive_and_unc_patterns(
    workspace: Path,
    pattern: str,
) -> None:
    gateway = _gateway(GlobTool(workspace=workspace))

    result = await gateway.call(_call("glob", {"pattern": pattern}))

    assert result.status == "error"
    assert "relative" in result.content.lower()


@pytest.mark.asyncio
async def test_directory_tools_reject_explicit_empty_roots(workspace: Path) -> None:
    identity = workspace
    gateway = _gateway(ListDirTool(workspace=identity), GlobTool(workspace=identity))

    listing = await gateway.call(_call("list_dir", {"path": ""}, call_id="call_empty_list_root"))
    matches = await gateway.call(
        _call("glob", {"path": "", "pattern": "*"}, call_id="call_empty_glob_root")
    )

    assert listing.status == "error"
    assert matches.status == "error"


@pytest.mark.asyncio
async def test_ignored_roots_remain_ignored_without_gitignore_parsing(workspace: Path) -> None:
    ignored = workspace / "build"
    ignored.mkdir()
    (ignored / "kept.txt").write_text("kept", encoding="utf-8")
    host_case_variant = workspace / "NODE_MODULES"
    host_case_variant.mkdir()
    (host_case_variant / "case.txt").write_text("case", encoding="utf-8")
    (workspace / ".gitignore").write_text("kept.txt\n", encoding="utf-8")
    identity = workspace
    list_gateway = _gateway(ListDirTool(workspace=identity))
    glob_gateway = _gateway(GlobTool(workspace=identity))

    listing = await list_gateway.call(_call("list_dir", {}))
    ignored_listing = await list_gateway.call(
        _call("list_dir", {"path": "build"}, call_id="call_ignored_list")
    )
    matches = await glob_gateway.call(
        _call("glob", {"pattern": "*", "kind": "both", "head_limit": 0})
    )
    ignored_matches = await glob_gateway.call(
        _call(
            "glob",
            {"pattern": "*", "path": "build", "kind": "both", "head_limit": 0},
            call_id="call_ignored_glob",
        )
    )

    assert "build/" not in listing.content
    assert ignored_listing.content == ""
    assert "kept.txt" not in matches.content
    assert ignored_matches.content == ""
    assert ".gitignore" in listing.content
    if os.name == "nt":
        assert "NODE_MODULES/" not in listing.content
        assert "NODE_MODULES/" not in matches.content
    else:
        assert "NODE_MODULES/" in listing.content
        assert "NODE_MODULES/" in matches.content


@pytest.mark.asyncio
async def test_directory_links_are_reported_but_never_traversed(workspace: Path) -> None:
    target = workspace / "target"
    target.mkdir()
    (target / "inside.txt").write_text("inside", encoding="utf-8")
    link = workspace / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        if os.name != "nt":
            pytest.skip(f"directory links unavailable: {error}")
        junction = subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(link), str(target)),
            capture_output=True,
            text=True,
        )
        if junction.returncode != 0:
            pytest.skip(f"directory links unavailable: {error}")

    identity = workspace
    list_gateway = _gateway(ListDirTool(workspace=identity))
    glob_gateway = _gateway(GlobTool(workspace=identity))

    listing = await list_gateway.call(_call("list_dir", {"recursive": True}))
    matches = await glob_gateway.call(
        _call("glob", {"pattern": "*", "kind": "both", "head_limit": 0})
    )

    assert "linked/" in listing.content
    assert "linked/inside.txt" not in listing.content
    assert "linked/" in matches.content
    assert "linked/inside.txt" not in matches.content


@pytest.mark.asyncio
async def test_directory_symlink_roots_are_never_traversed(workspace: Path) -> None:
    target = workspace / "target"
    target.mkdir()
    (target / "inside.txt").write_text("inside", encoding="utf-8")
    link = workspace / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    identity = workspace
    list_gateway = _gateway(ListDirTool(workspace=identity))
    glob_gateway = _gateway(GlobTool(workspace=identity))

    listing = await list_gateway.call(_call("list_dir", {"path": "linked", "recursive": True}))
    matches = await glob_gateway.call(
        _call("glob", {"path": "linked", "pattern": "*", "head_limit": 0})
    )

    assert listing.status == "success"
    assert listing.content == ""
    assert matches.status == "success"
    assert matches.content == ""


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
async def test_directory_junction_roots_are_never_traversed(workspace: Path) -> None:
    target = workspace / "target"
    target.mkdir()
    (target / "inside.txt").write_text("inside", encoding="utf-8")
    junction = workspace / "linked"
    created = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(junction), str(target)),
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"directory junctions unavailable: {created.stderr.strip()}")

    identity = workspace
    list_gateway = _gateway(ListDirTool(workspace=identity))
    glob_gateway = _gateway(GlobTool(workspace=identity))

    listing = await list_gateway.call(_call("list_dir", {"path": "linked", "recursive": True}))
    matches = await glob_gateway.call(
        _call("glob", {"path": "linked", "pattern": "*", "head_limit": 0})
    )

    assert listing.status == "success"
    assert listing.content == ""
    assert matches.status == "success"
    assert matches.content == ""


@pytest.mark.asyncio
async def test_inaccessible_descendants_are_skipped_but_roots_fail(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant = workspace / "unreadable"
    descendant.mkdir()
    (descendant / "hidden.txt").write_text("hidden", encoding="utf-8")
    original_iterdir = Path.iterdir

    def fail_descendant(path: Path) -> Iterator[Path]:
        if path == descendant:
            raise PermissionError("descendant denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_descendant)
    gateway = _gateway(ListDirTool(workspace=workspace))

    result = await gateway.call(_call("list_dir", {"recursive": True}))

    assert result.status == "success"
    assert "unreadable/" in result.content
    assert "unreadable/hidden.txt" not in result.content

    root = workspace / "missing"
    invalid = await gateway.call(
        _call("list_dir", {"path": str(root)}, call_id="call_invalid_root")
    )
    assert invalid.status == "error"
    assert "List Dir" in invalid.content


@pytest.mark.asyncio
async def test_confirmed_external_roots_report_resolved_absolute_posix_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    (external / "outside.txt").write_text("outside", encoding="utf-8")
    identity = workspace
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        ListDirTool(workspace=identity),
        GlobTool(workspace=identity),
        confirmation=approve,
    )
    list_result = await gateway.call(
        _call("list_dir", {"path": str(external)}, call_id="call_external_list")
    )
    glob_result = await gateway.call(
        _call(
            "glob",
            {"path": str(external), "pattern": "*.txt", "head_limit": 0},
            call_id="call_external_glob",
        )
    )

    expected = (external / "outside.txt").resolve().as_posix()
    assert list_result.content == expected
    assert glob_result.content == expected
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_external_confirmation_is_bound_to_the_exact_directory_call(
    workspace: Path,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "outside.txt").write_text("outside", encoding="utf-8")
    gateway = _gateway(ListDirTool(workspace=workspace))
    call = _call("list_dir", {"path": str(external)}, call_id="call_external")

    refused = await gateway.call(call)
    assert refused.status == "refused"
    assert refused.confirmation is not None
    request = refused.confirmation.request

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
    assert approved.status == "success"
