from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from myclaw.agent.tools.base import BaseTool
from myclaw.agent.tools.core.grep import GrepTool
from myclaw.agent.tools.tool_gateway import (
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
async def test_grep_supports_regex_fixed_strings_case_insensitivity_and_invalid_patterns(
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    target.write_text("needle alpha\nNEEDLE beta\nneedle needle\nplain\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    regex = await gateway.call(_call("grep", {"pattern": r"needle \w+", "path": "notes.txt"}))
    fixed = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "path": "notes.txt", "fixed_string": True},
            call_id="call_fixed",
        )
    )
    insensitive = await gateway.call(
        _call(
            "grep",
            {
                "pattern": "needle",
                "path": "notes.txt",
                "fixed_string": True,
                "ignore_case": True,
            },
            call_id="call_insensitive",
        )
    )
    invalid = await gateway.call(
        _call("grep", {"pattern": "[", "path": "notes.txt"}, call_id="call_invalid")
    )

    assert regex.content == "notes.txt:1:needle alpha\n--\nnotes.txt:3:needle needle"
    assert fixed.content == "notes.txt:1:needle alpha\n--\nnotes.txt:3:needle needle"
    assert insensitive.content == (
        "notes.txt:1:needle alpha\n--\nnotes.txt:2:NEEDLE beta\n--\nnotes.txt:3:needle needle"
    )
    assert invalid.status == "error"
    assert "invalid" in invalid.content.lower()


@pytest.mark.asyncio
async def test_grep_fixed_string_case_insensitivity_uses_python_regex_semantics(
    workspace: Path,
) -> None:
    target = workspace / "unicode.txt"
    target.write_text("SS\n\u0131\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    sharp_s = await gateway.call(
        _call(
            "grep",
            {
                "pattern": "\u00df",
                "path": "unicode.txt",
                "fixed_string": True,
                "ignore_case": True,
            },
        )
    )
    dotless_i = await gateway.call(
        _call(
            "grep",
            {
                "pattern": "i",
                "path": "unicode.txt",
                "fixed_string": True,
                "ignore_case": True,
            },
            call_id="call_dotless_i",
        )
    )

    assert sharp_s.content == ""
    assert dotless_i.content == "unicode.txt:2:\u0131"


@pytest.mark.asyncio
async def test_grep_context_merges_windows_and_uses_context_delimiters(
    workspace: Path,
) -> None:
    target = workspace / "context.txt"
    target.write_text(
        "one\nhit two\nthree\nhit four\nfive\nsix\nhit seven\neight\n",
        encoding="utf-8",
    )
    gateway = _gateway(GrepTool(workspace=workspace))

    merged = await gateway.call(
        _call("grep", {"pattern": "hit", "path": "context.txt", "context": 1})
    )
    paged = await gateway.call(
        _call(
            "grep",
            {"pattern": "hit", "path": "context.txt", "context": 2, "head_limit": 1},
            call_id="call_paged",
        )
    )
    disjoint = await gateway.call(
        _call(
            "grep",
            {"pattern": "hit", "path": "context.txt", "context": 1, "offset": 0},
            call_id="call_disjoint",
        )
    )

    assert merged.content == (
        "context.txt-1-one\n"
        "context.txt:2:hit two\n"
        "context.txt-3-three\n"
        "context.txt:4:hit four\n"
        "context.txt-5-five\n"
        "--\n"
        "context.txt-6-six\n"
        "context.txt:7:hit seven\n"
        "context.txt-8-eight"
    )
    assert paged.content == (
        "context.txt-1-one\ncontext.txt:2:hit two\ncontext.txt-3-three\ncontext.txt-4-hit four"
    )
    assert "--" in disjoint.content


@pytest.mark.asyncio
async def test_grep_separates_disjoint_content_groups_across_files(workspace: Path) -> None:
    (workspace / "a.py").write_text("hit\n", encoding="utf-8")
    (workspace / "b.py").write_text("hit\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    result = await gateway.call(_call("grep", {"pattern": "hit", "head_limit": 0}))

    assert result.content == "a.py:1:hit\n--\nb.py:1:hit"


@pytest.mark.asyncio
async def test_grep_paginates_files_and_counts_matching_lines_not_occurrences(
    workspace: Path,
) -> None:
    (workspace / "a.py").write_text("hit hit\nplain\n", encoding="utf-8")
    (workspace / "b.py").write_text("hit\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    files = await gateway.call(
        _call(
            "grep",
            {
                "pattern": "hit",
                "output_mode": "files_with_matches",
                "head_limit": 1,
                "offset": 1,
            },
        )
    )
    counts = await gateway.call(
        _call("grep", {"pattern": "hit", "output_mode": "count"}, call_id="call_counts")
    )
    paged_counts = await gateway.call(
        _call(
            "grep",
            {"pattern": "hit", "output_mode": "count", "head_limit": 1, "offset": 1},
            call_id="call_paged_counts",
        )
    )

    assert files.content == "b.py"
    assert counts.content == "a.py:1\nb.py:1"
    assert paged_counts.content == "b.py:1"


@pytest.mark.asyncio
async def test_grep_omitted_head_limit_returns_all_matches(workspace: Path) -> None:
    target = workspace / "many.txt"
    target.write_text(
        "".join(f"hit {index}\n" for index in range(201)),
        encoding="utf-8",
    )
    gateway = _gateway(GrepTool(workspace=workspace))
    function = cast(dict[str, object], gateway.schemas[0]["function"])
    parameters = cast(dict[str, object], function["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    head_limit = cast(dict[str, object], properties["head_limit"])

    result = await gateway.call(_call("grep", {"pattern": "hit", "path": "many.txt"}))

    assert head_limit["default"] == 0
    matches = [line for line in result.content.splitlines() if line != "--"]
    assert len(matches) == 201
    assert matches[-1] == "many.txt:201:hit 200"


@pytest.mark.asyncio
async def test_grep_intersects_glob_and_type_filters_and_supports_aliases(
    workspace: Path,
) -> None:
    (workspace / "python.py").write_text("needle\n", encoding="utf-8")
    (workspace / "stub.pyi").write_text("needle\n", encoding="utf-8")
    (workspace / "script.js").write_text("needle\n", encoding="utf-8")
    (workspace / "document.md").write_text("needle\n", encoding="utf-8")
    (workspace / "custom.foo").write_text("needle\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    python = await gateway.call(
        _call("grep", {"pattern": "needle", "type": "PYTHON", "head_limit": 0})
    )
    intersected = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "type": "python", "glob": "*.py", "head_limit": 0},
            call_id="call_intersected",
        )
    )
    disabled = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "type": " ", "glob": "  ", "head_limit": 0},
            call_id="call_disabled",
        )
    )
    unknown = await gateway.call(
        _call("grep", {"pattern": "needle", "type": "foo", "head_limit": 0}, call_id="call_unknown")
    )

    assert python.content == "python.py:1:needle\n--\nstub.pyi:1:needle"
    assert intersected.content == "python.py:1:needle"
    assert disabled.content == (
        "custom.foo:1:needle\n"
        "--\n"
        "document.md:1:needle\n"
        "--\n"
        "python.py:1:needle\n"
        "--\n"
        "script.js:1:needle\n"
        "--\n"
        "stub.pyi:1:needle"
    )
    assert unknown.content == "custom.foo:1:needle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("type_name", "suffix"),
    (
        ("javascript", ".js"),
        ("typescript", ".ts"),
        ("tsx", ".tsx"),
        ("jsx", ".jsx"),
        ("json", ".json"),
        ("markdown", ".md"),
        ("go", ".go"),
        ("rust", ".rs"),
        ("java", ".java"),
        ("shell", ".sh"),
        ("yaml", ".yaml"),
        ("toml", ".toml"),
        ("sql", ".sql"),
        ("html", ".html"),
        ("css", ".css"),
    ),
)
async def test_grep_builtin_type_aliases_filter_by_their_suffix(
    workspace: Path,
    type_name: str,
    suffix: str,
) -> None:
    target = workspace / f"source{suffix}"
    target.write_text("needle\n", encoding="utf-8")
    gateway = _gateway(GrepTool(workspace=workspace))

    result = await gateway.call(
        _call("grep", {"pattern": "needle", "type": type_name, "head_limit": 0})
    )

    assert result.content == f"source{suffix}:1:needle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "glob",
    ("/absolute/*.py", "C:\\absolute\\*.py", "\\\\server\\share\\*.py"),
)
async def test_grep_reuses_glob_absolute_pattern_rejection(
    workspace: Path,
    glob: str,
) -> None:
    gateway = _gateway(GrepTool(workspace=workspace))

    result = await gateway.call(_call("grep", {"pattern": "needle", "glob": glob}))

    assert result.status == "error"
    assert "relative" in result.content.lower()


@pytest.mark.asyncio
async def test_grep_skips_ignored_and_inaccessible_descendants(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workspace / "visible.py").write_text("needle\n", encoding="utf-8")
    ignored = workspace / "build"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("needle\n", encoding="utf-8")
    inaccessible = workspace / "inaccessible"
    inaccessible.mkdir()
    (inaccessible / "hidden.py").write_text("needle\n", encoding="utf-8")
    original_iterdir = Path.iterdir

    def fail_inaccessible(path: Path) -> Iterator[Path]:
        if path == inaccessible:
            raise PermissionError("descendant denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_inaccessible)

    gateway = _gateway(GrepTool(workspace=workspace))
    result = await gateway.call(_call("grep", {"pattern": "needle", "head_limit": 0}))
    ignored_root = await gateway.call(
        _call("grep", {"pattern": "needle", "path": "build"}, call_id="call_ignored_root")
    )

    assert result.content == "visible.py:1:needle"
    assert ignored_root.content == ""


@pytest.mark.asyncio
async def test_grep_preserves_file_link_paths(workspace: Path) -> None:
    (workspace / "visible.py").write_text("needle\n", encoding="utf-8")
    link = workspace / "alias.py"
    try:
        link.symlink_to(workspace / "visible.py")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")

    gateway = _gateway(GrepTool(workspace=workspace))
    result = await gateway.call(_call("grep", {"pattern": "needle", "head_limit": 0}))

    assert result.content == "alias.py:1:needle\n--\nvisible.py:1:needle"


@pytest.mark.asyncio
async def test_grep_does_not_traverse_an_explicit_directory_link(workspace: Path) -> None:
    target = workspace / "target"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / "inside.py").write_text("needle\n", encoding="utf-8")
    link = workspace / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory links unavailable: {error}")

    gateway = _gateway(GrepTool(workspace=workspace))
    root_result = await gateway.call(_call("grep", {"pattern": "needle", "path": "linked"}))
    nested_result = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "path": "linked/nested"},
            call_id="call_nested_link",
        )
    )
    file_result = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "path": "linked/nested/inside.py"},
            call_id="call_file_below_link",
        )
    )

    assert root_result.content == ""
    assert nested_result.content == ""
    assert file_result.content == ""


@pytest.mark.asyncio
async def test_grep_skips_file_links_outside_the_approved_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    (external / "outside.py").write_text("needle\n", encoding="utf-8")
    link = workspace / "outside.py"
    try:
        link.symlink_to(external / "outside.py")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")

    gateway = _gateway(GrepTool(workspace=workspace))
    result = await gateway.call(_call("grep", {"pattern": "needle"}))

    assert result.content == ""


@pytest.mark.asyncio
async def test_grep_reports_explicit_file_links_by_their_visible_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    inside = workspace / "inside.py"
    outside = external / "outside.py"
    inside.write_text("needle\n", encoding="utf-8")
    outside.write_text("needle\n", encoding="utf-8")
    internal_link = workspace / "internal-alias.py"
    external_link = external / "external-alias.py"
    try:
        internal_link.symlink_to(outside)
        external_link.symlink_to(inside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file links unavailable: {error}")

    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(
        GrepTool(workspace=workspace),
        confirmation=approve,
    )
    internal_result = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "path": str(internal_link), "glob": "internal-*.py"},
        )
    )
    external_result = await gateway.call(
        _call(
            "grep",
            {"pattern": "needle", "path": str(external_link), "glob": "external-*.py"},
            call_id="call_external_link",
        )
    )

    assert internal_result.content == "internal-alias.py:1:needle"
    assert external_result.content == f"{external_link.absolute().as_posix()}:1:needle"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_grep_external_root_requires_confirmation_and_reports_absolute_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    (external / "outside.py").write_text("needle\n", encoding="utf-8")
    identity = workspace
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(GrepTool(workspace=identity), confirmation=approve)
    result = await gateway.call(_call("grep", {"pattern": "needle", "path": str(external)}))

    assert result.content == f"{(external / 'outside.py').resolve().as_posix()}:1:needle"
    assert len(requests) == 1
