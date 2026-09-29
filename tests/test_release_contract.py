import ast
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]


def _source_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source_class(tree: ast.AST, name: str) -> ast.ClassDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _source_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _public_method_names(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _direct_method(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _parameter_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _attribute_call_lines(
    tree: ast.AST,
    owner: str,
    attribute: str,
) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        and _attribute_owner(node.func.value) == owner
    )


def _attribute_owner(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_owner(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


def _attribute_reference_lines(
    tree: ast.AST,
    owner: str,
    attribute: str,
) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == attribute
        and _attribute_owner(node.value) == owner
    )


def _named_call_lines(tree: ast.AST, names: set[str]) -> tuple[int, ...]:
    return tuple(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
    )


def _assignment_lines(
    tree: ast.AST,
    target: str,
    value: str | None,
) -> tuple[int, ...]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(candidate, ast.Name) and candidate.id == target for candidate in node.targets
        ):
            continue
        if value is None and isinstance(node.value, ast.Constant) and node.value.value is None:
            lines.append(node.lineno)
        if value is not None and isinstance(node.value, ast.Name) and node.value.id == value:
            lines.append(node.lineno)
    return tuple(lines)


# The tracked corpus uses these simple inline/reference target forms. This is not a
# complete CommonMark parser, and deliberately does not claim to be one.
_INLINE_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]\n]*\]\((?P<target><[^>\n]+>|[^)\s\n]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\)"
)
_REFERENCE_MARKDOWN_LINK = re.compile(r"(?m)^\s*\[[^\]\n]+\]:\s*(?P<target><[^>\n]+>|\S+)")


def _tracked_markdown_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ("git", "ls-files", "-z", "--", "*.md"),
        cwd=ROOT,
    ).decode("utf-8")
    return tuple(ROOT / Path(relative) for relative in output.split("\0") if relative)


def test_tracked_markdown_inventory_preserves_missing_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b"README.md\0docs/unexpected-missing.md\0"
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: output)

    assert _tracked_markdown_paths() == (
        ROOT / "README.md",
        ROOT / "docs" / "unexpected-missing.md",
    )


def test_tracked_markdown_link_contract_rejects_missing_tracked_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected = ROOT / "docs" / "unexpected-missing.md"
    monkeypatch.setitem(globals(), "_tracked_markdown_paths", lambda: (unexpected,))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *_args, **_kwargs: b"docs/unexpected-missing.md\n",
    )

    with pytest.raises(AssertionError, match=r"unexpected-missing\.md"):
        test_tracked_markdown_local_links_resolve()


def _markdown_link_targets(content: str) -> tuple[str, ...]:
    return tuple(
        match.group("target").removeprefix("<").removesuffix(">")
        for pattern in (_INLINE_MARKDOWN_LINK, _REFERENCE_MARKDOWN_LINK)
        for match in pattern.finditer(content)
    )


def _local_markdown_path(target: str) -> str | None:
    if target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    return path or None


def _adr_status(path: Path) -> object:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0] == "---", path
    try:
        closing = lines.index("---", 1)
    except ValueError as error:
        raise AssertionError(f"{path} has no closing frontmatter delimiter") from error
    frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(frontmatter, dict), path
    return frontmatter.get("status")


def _adr_status_contract_issues(decisions: list[Path]) -> list[str]:
    numbers: dict[str, Path] = {}
    issues: list[str] = []
    for path in decisions:
        number = path.name.split("-", 1)[0]
        previous = numbers.get(number)
        if previous is not None:
            issues.append(f"duplicate ADR number {number}: {previous.name}, {path.name}")
        else:
            numbers[number] = path

    for path in decisions:
        number = path.name.split("-", 1)[0]
        status = _adr_status(path)
        if status == "accepted":
            continue
        if not isinstance(status, str):
            issues.append(f"{path.name}: missing ADR status")
            continue
        match = re.fullmatch(r"superseded by ADR-(?P<number>\d{4})", status)
        if match is None:
            issues.append(f"{path.name}: invalid ADR status {status!r}")
            continue
        superseder_number = match.group("number")
        if superseder_number == number:
            issues.append(f"{path.name}: ADR cannot supersede itself")
            continue
        if int(superseder_number) < int(number):
            issues.append(
                f"{path.name}: superseding ADR-{superseder_number} must have a later number"
            )
            continue
        superseder = numbers.get(superseder_number)
        if superseder is None:
            issues.append(f"{path.name}: superseding ADR-{superseder_number} does not exist")
            continue
        if _adr_status(superseder) != "accepted":
            issues.append(f"{path.name}: superseding ADR-{superseder_number} is not accepted")
    return issues


def test_distribution_declares_supported_loguru_release_range() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "loguru>=0.7.3,<0.8" in project["dependencies"]


def test_distribution_directly_declares_iana_timezone_database() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "tzdata>=2026.2" in project["dependencies"]


def test_distribution_directly_declares_host_timezone_discovery() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "tzlocal>=5,<6" in project["dependencies"]


def test_distribution_metadata_builds_one_host_neutral_wheel() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["myclaw"] == "myclaw.terminal.process_entry:run"
    assert "Operating System :: OS Independent" not in project["classifiers"]
    assert "Operating System :: Microsoft :: Windows" not in project["classifiers"]
    setup_path = ROOT / "setup.cfg"
    setup = setup_path.read_text(encoding="utf-8") if setup_path.exists() else ""
    assert "plat_name" not in setup


def _ignore_unclean_build_inputs(_directory: str, names: list[str]) -> set[str]:
    ignored = {".codegraph", ".git", ".pytest_cache", "build", "dist", "__pycache__"}
    return {name for name in names if name in ignored or name.endswith(".egg-info")}


def test_clean_distributions_build_and_import_cleanly(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    shutil.copytree(ROOT, source_root, ignore=_ignore_unclean_build_inputs)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(artifact_dir),
        ],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert build_result.returncode == 0, build_result.stderr

    sdists = tuple(artifact_dir.glob("myclaw-*.tar.gz"))
    wheels = tuple(artifact_dir.glob("myclaw-*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_members = {member.name.replace("\\", "/") for member in archive.getmembers()}
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_members = {member.replace("\\", "/") for member in archive.namelist()}

    assert any(member.endswith("/myclaw/__init__.py") for member in sdist_members)
    assert "myclaw/__init__.py" in wheel_members

    install_root = tmp_path / "clean-install"
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    clean_import_dir = tmp_path / "clean-import"
    clean_import_dir.mkdir()
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import myclaw\nimport myclaw.terminal.cli\n"),
        ],
        cwd=clean_import_dir,
        env={**os.environ, "PYTHONPATH": str(install_root)},
        capture_output=True,
        check=False,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_session_log_adr_publishes_the_risk_contract() -> None:
    required_contract = (
        "same-session concurrency is unsupported",
        "unbounded queue",
        "infinite drain",
        "no per-record fsync",
        "no active redaction",
        "no control escaping",
        "per-session retention",
        "legacy agent home runtime log files remain untouched",
    )

    path = ROOT / "docs" / "adr" / "0008-use-workspace-session-log.md"
    content = path.read_text(encoding="utf-8").lower()
    assert all(statement in content for statement in required_contract), path


def test_application_modules_do_not_depend_on_standard_library_logging() -> None:
    violations: list[str] = []

    for path in sorted((ROOT / "myclaw").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "logging" for alias in node.names
            ):
                violations.append(f"{path}: imports logging")
            if isinstance(node, ast.ImportFrom) and node.module == "logging":
                violations.append(f"{path}: imports from logging")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"debug", "info", "warning", "error", "critical"}
                and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "%" in node.args[0].value
            ):
                violations.append(f"{path}:{node.lineno}: percent-style logging arguments")
        if "InterceptHandler" in source:
            violations.append(f"{path}: logging interception bridge")

    assert violations == []


def test_active_support_contract_matches_host_neutral_release_evidence() -> None:
    decision_path = ROOT / "docs" / "adr" / "0007-use-host-adapters.md"
    assert decision_path.exists()
    decision = decision_path.read_text(encoding="utf-8").lower()
    assert "status: accepted" in decision
    assert "filesystem" in decision
    assert "process tree" not in decision
    assert "owned-process" not in decision
    assert "runtime log locking" not in decision

    for claim in (
        "py3-none-any",
        "windows x64",
        "currently validated",
        "macos intel",
        "apple silicon",
        "unverified",
        "no supported-platform gate",
    ):
        assert claim in decision


def test_current_adrs_have_unique_numbers_and_valid_status_contract() -> None:
    decisions = sorted((ROOT / "docs" / "adr").glob("*.md"))

    assert _adr_status_contract_issues(decisions) == []


@pytest.mark.parametrize(
    ("documents", "expected"),
    [
        (
            {"0001-first.md": "accepted", "0001-second.md": "accepted"},
            "duplicate ADR number",
        ),
        ({"0001-first.md": None}, "missing ADR status"),
        ({"0001-first.md": "draft"}, "invalid ADR status"),
        ({"0001-first.md": "superseded by ADR-0001"}, "ADR cannot supersede itself"),
        (
            {
                "0001-first.md": "accepted",
                "0002-second.md": "superseded by ADR-0001",
            },
            "must have a later number",
        ),
        (
            {
                "0001-first.md": "Superseded by ADR-0002",
                "0002-second.md": "accepted",
            },
            "invalid ADR status",
        ),
        (
            {"0001-first.md": "superseded by ADR-0099"},
            "superseding ADR-0099 does not exist",
        ),
        (
            {
                "0001-first.md": "superseded by ADR-0002",
                "0002-second.md": "superseded by ADR-0003",
            },
            "superseding ADR-0002 is not accepted",
        ),
    ],
)
def test_adr_status_contract_rejects_invalid_relationships(
    tmp_path: Path,
    documents: dict[str, str | None],
    expected: str,
) -> None:
    paths: list[Path] = []
    for filename, status in documents.items():
        status_line = "title: fixture\n" if status is None else f"status: {status}\n"
        path = tmp_path / filename
        path.write_text(f"---\n{status_line}---\n", encoding="utf-8")
        paths.append(path)

    issues = _adr_status_contract_issues(sorted(paths))

    assert expected in "\n".join(issues)


def test_mcp_transport_evidence_uses_local_fixtures() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    mcp_tests = (ROOT / "tests" / "tools" / "test_mcp.py").read_text(encoding="utf-8")
    cli_mcp_tests = (ROOT / "tests" / "test_cli_mcp_lifecycle.py").read_text(encoding="utf-8")

    assert "mcp>=2,<3" in project["dependencies"]
    test_urls = re.findall(r"https?://[^\"']+", mcp_tests)
    assert test_urls
    assert all(urlsplit(url).hostname in {"127.0.0.1", "localhost"} for url in test_urls)
    for test_name in (
        "test_stdio_transport_connects_to_a_local_real_mcp_server",
        "test_streamable_http_transport_connects_to_a_local_real_mcp_server",
    ):
        assert test_name in mcp_tests

    full_flow_test = "test_cli_real_mcp_flow_persists_result_reuses_connection_and_closes"
    assert full_flow_test in cli_mcp_tests


def test_tracked_markdown_local_links_resolve() -> None:
    tracked = _tracked_markdown_paths()

    missing: list[str] = []
    for source in tracked:
        if not source.exists():
            missing.append(f"{source.relative_to(ROOT)}: tracked Markdown source is missing")
            continue
        content = source.read_text(encoding="utf-8")
        for target in _markdown_link_targets(content):
            local_path = _local_markdown_path(target)
            if local_path is None:
                continue
            candidate = (source.parent / local_path).resolve()
            if not candidate.exists():
                missing.append(f"{source.relative_to(ROOT)}: {target} -> {candidate}")

    assert missing == []


def test_current_architecture_matches_source_ast_contracts() -> None:
    loaded_skill = _source_class(
        _source_ast(ROOT / "myclaw" / "skills" / "catalog.py"),
        "LoadedSkill",
    )
    assert {
        node.target.id
        for node in loaded_skill.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    } == {"metadata", "document", "always"}

    skill_loader = _source_class(
        _source_ast(ROOT / "myclaw" / "skills" / "catalog.py"),
        "SkillLoader",
    )
    assert _public_method_names(skill_loader) == {
        "root",
        "skills",
        "metadata",
        "get",
        "resolve_manual",
        "load",
    }

    message_bus = _source_class(
        _source_ast(ROOT / "myclaw" / "agent" / "message_bus.py"),
        "MessageBus",
    )
    assert {
        node.name
        for node in message_bus.body
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_")
    } == {
        "inbound_snapshot",
        "put_inbound",
        "get_inbound",
        "pause_inbound_delivery",
        "resume_inbound_delivery",
        "drain_inbound",
        "put_outbound",
        "get_outbound",
        "reset",
    }
    assert {
        node.name
        for node in message_bus.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    } == {"set_inbound_changed_callback", "unbind_inbound_changed_callback"}

    memory_manager = _source_class(
        _source_ast(ROOT / "myclaw" / "agent" / "memory" / "manager.py"),
        "MemoryManager",
    )
    assert _public_method_names(memory_manager) == {
        "long_term_path",
        "append_summary",
        "claim_summaries",
        "read_long_term",
        "edit_long_term",
        "memory_snapshot",
    }

    dream = _source_class(
        _source_ast(ROOT / "myclaw" / "agent" / "memory" / "dream.py"),
        "Dream",
    )
    assert _parameter_names(_source_function(dream, "__init__")) == (
        "self",
        "memory_manager",
        "model_router",
        "batch_size",
        "memory_route_status",
    )
    assert _public_method_names(dream) == {
        "run",
        "close",
        "wait_until_idle",
        "abort",
        "abort_and_wait",
    }

    schedule_tree = _source_ast(ROOT / "myclaw" / "schedule" / "service.py")
    schedule_clock = _source_class(schedule_tree, "ScheduleClock")
    assert _public_method_names(schedule_clock) == {"now", "monotonic", "sleep"}
    schedule_service = _source_class(schedule_tree, "ScheduleService")
    assert _parameter_names(_source_function(schedule_service, "__init__")) == (
        "self",
        "workspace_state",
        "clock",
        "execute_user_job",
        "execute_dream",
        "timezone_name",
    )
    assert _parameter_names(_source_function(schedule_service, "register_dream_job")) == (
        "self",
        "schedule",
    )


def test_cli_source_records_cutover_and_shutdown_order() -> None:
    cli_tree = _source_ast(ROOT / "myclaw" / "terminal" / "cli.py")
    replacement = _source_function(cli_tree, "replace_agent_loop")
    restore = _source_function(cli_tree, "rebuild_after_restore")
    handover = _source_function(cli_tree, "handover_prepared_generation")
    preflight_lines = _attribute_call_lines(replacement, "target", "preflight")
    quiesce_lines = _attribute_call_lines(handover, "terminal_app", "quiesce_for_rebind")
    pause_lines = _attribute_call_lines(handover, "schedule_service", "pause_and_drain")
    reset_lines = _attribute_call_lines(handover, "bus", "reset")
    rebind_lines = _attribute_call_lines(handover, "terminal_app", "rebind_agent_loop")
    start_lines = _attribute_call_lines(handover, "target", "start")
    activate_lines = _attribute_call_lines(handover, "runtime", "activate_mcp_generation")
    resume_lines = _attribute_call_lines(replacement, "schedule_service", "resume")
    current_none_lines = _assignment_lines(handover, "current_loop", None)
    current_target_lines = _assignment_lines(handover, "current_loop", "target")
    old_abort_lines = _named_call_lines(handover, {"abort_loop_once"})

    cutover = (
        min(quiesce_lines),
        min(pause_lines),
        min(line for line in current_none_lines if line > min(pause_lines)),
        min(old_abort_lines),
        min(reset_lines),
        min(rebind_lines),
        min(start_lines),
        min(activate_lines),
        min(current_target_lines),
    )
    assert cutover == tuple(sorted(cutover))
    for path in (replacement, restore):
        assert min(_attribute_call_lines(path, "target", "preflight")) < min(
            _named_call_lines(path, {"cancel_old_generation_confirmations"})
        ) < min(_named_call_lines(path, {"handover_prepared_generation"}))
    assert min(preflight_lines) < min(
        _named_call_lines(replacement, {"handover_prepared_generation"})
    ) < min(resume_lines)
    assert min(_named_call_lines(restore, {"handover_prepared_generation"})) < min(
        _attribute_call_lines(restore, "old_loop", "_release_replacement_barrier")
    ) < min(_attribute_call_lines(restore, "schedule_service", "resume"))

    conversation = _source_function(cli_tree, "_run_cli_conversation")
    shutdown = next(
        node for node in conversation.body if isinstance(node, ast.Try) and node.finalbody
    )
    final_tree = ast.Module(body=shutdown.finalbody, type_ignores=[])
    runtime_close_lines = _attribute_call_lines(final_tree, "runtime", "close")
    assert len(runtime_close_lines) == 1
    assert _attribute_call_lines(conversation, "runtime", "close") == runtime_close_lines
    foreground_close = _source_function(conversation, "close_foreground_loops")
    close_lines = _attribute_call_lines(foreground_close, "active_loop", "close")
    assert len(close_lines) == 1
    abort_lines = _named_call_lines(foreground_close, {"abort_loop_once"})
    assert abort_lines
    shutdown_events = (
        min(_attribute_call_lines(final_tree, "runtime", "drain_confirmation_aborts")),
        min(_attribute_call_lines(final_tree, "management", "deactivate")),
        min(runtime_close_lines),
    )
    assert shutdown_events == tuple(sorted(shutdown_events))

    runtime_tree = _source_ast(ROOT / "myclaw" / "agent" / "workspace_runtime.py")
    runtime_shutdown = _source_function(runtime_tree, "_close_owned_resources")
    resource_shutdown = (
        min(_attribute_reference_lines(runtime_shutdown, "schedule", "pause_and_drain")),
        min(_attribute_reference_lines(runtime_shutdown, "schedule", "close")),
        min(_attribute_reference_lines(runtime_shutdown, "self._mcp_manager", "close")),
        min(_attribute_reference_lines(runtime_shutdown, "self._dream", "close")),
        min(_attribute_reference_lines(runtime_shutdown, "self._router", "close")),
    )
    assert resource_shutdown == tuple(sorted(resource_shutdown))


def test_composition_and_store_signatures_match_current_contracts() -> None:
    management = _source_class(
        _source_ast(ROOT / "myclaw" / "management" / "service.py"),
        "ManagementViewService",
    )
    management_init = _direct_method(management, "__init__")
    assert _parameter_names(management_init) == (
        "self",
        "agent_home",
        "current_agent_loop",
        "workspace_state",
        "replace_agent_loop",
        "prepare_session_resume",
        "memory_manager",
        "dream",
        "schedule_status",
        "now",
        "monotonic",
        "reasoning_effort_control",
        "permission_control",
        "restore_listing",
        "restore_inspect",
        "restore_commit",
        "restore_result",
        "restore_cancel",
        "ensure_management_mutation_allowed",
    )
    assert management_init.args.defaults == []
    assert management_init.args.kw_defaults[:11] == [None] * 11
    assert all(
        isinstance(default, ast.Constant) and default.value is None
        for default in management_init.args.kw_defaults[11:]
    )

    terminal = _source_class(
        _source_ast(ROOT / "myclaw" / "terminal" / "conversation.py"),
        "TerminalConversationApp",
    )
    terminal_init = _direct_method(terminal, "__init__")
    assert _parameter_names(terminal_init) == (
        "self",
        "bus",
        "control",
        "management_dispatcher",
        "monotonic",
        "skill_metadata",
    )
    management_index = tuple(argument.arg for argument in terminal_init.args.kwonlyargs).index(
        "management_dispatcher"
    )
    assert terminal_init.args.kw_defaults[management_index] is None

    schedule_store = _source_class(
        _source_ast(ROOT / "myclaw" / "schedule" / "store.py"),
        "WorkspaceScheduleStore",
    )
    assert _parameter_names(_direct_method(schedule_store, "_publish")) == (
        "self",
        "candidate",
    )


def test_session_exposes_the_terminal_agent_run_commit() -> None:
    session = _source_class(
        _source_ast(ROOT / "myclaw" / "agent" / "session" / "session.py"),
        "Session",
    )
    assert _direct_method(session, "commit_agent_run")


def test_runtime_status_uses_the_canonical_usage_anchor_and_configured_chat_route() -> None:
    loop_tree = _source_ast(ROOT / "myclaw" / "agent" / "loop.py")
    compactor_tree = _source_ast(ROOT / "myclaw" / "agent" / "context" / "run_context.py")

    anchor_definitions = [
        node
        for node in compactor_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "latest_main_agent_usage_anchor"
    ]
    assert len(anchor_definitions) == 1

    controller = _source_class(compactor_tree, "AgentRunContextController")
    controller_init = _direct_method(controller, "__init__")
    controller_anchor_calls = [
        node
        for node in ast.walk(controller_init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "latest_main_agent_usage_anchor"
    ]
    assert len(controller_anchor_calls) == 1

    loop = _source_class(loop_tree, "AgentLoop")
    runtime_status = _direct_method(loop, "runtime_status_input")
    status_anchor_calls = [
        node
        for node in ast.walk(runtime_status)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "latest_main_agent_usage_anchor"
    ]
    configured_route_calls = [
        node
        for node in ast.walk(runtime_status)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_configured_model_route_status"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "chat"
    ]
    assert len(status_anchor_calls) == 1
    assert len(configured_route_calls) == 1


def test_agent_run_context_exports_current_request_and_terminal_contracts() -> None:
    compactor_tree = _source_ast(ROOT / "myclaw" / "agent" / "context" / "run_context.py")
    class_names = {node.name for node in compactor_tree.body if isinstance(node, ast.ClassDef)}
    expected_contracts = {"AgentRunContextRequestPreparer", "AgentRunTerminalCommitValues"}

    assert expected_contracts <= class_names

    exports_assignment = next(
        node
        for node in compactor_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    exports = ast.literal_eval(exports_assignment.value)
    assert expected_contracts <= set(exports)

    controller = _source_class(compactor_tree, "AgentRunContextController")
    assert _direct_method(controller, "terminal_commit_values")
