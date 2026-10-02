from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import yaml  # type: ignore[import-untyped]

import scripts.release_validation as release_validation
from scripts.release_validation import (
    COVERAGE_RULES,
    POSIX_CASES,
    REQUIRED_POSIX_SMOKE_NODES,
    REQUIRED_WINDOWS_ALTERNATIVE_NODES,
    RESTORE_PATH_MATRIX_NODES,
    CoverageEvidence,
    CoverageRule,
    PytestEvidence,
    ReleasePhase,
    build_acceptance_matrix,
    build_coverage_evidence,
)


def test_windows_release_phases_are_explicit() -> None:
    assert tuple(ReleasePhase) == (
        "coverage",
        "host-integration",
        "quality",
        "artifact-smoke",
        "all",
    )


def test_release_host_selector_dispatch_and_posix_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: "windows")
    assert release_validation._selectors("both") == ("powershell", "pwsh")
    assert release_validation._selectors("pwsh") == ("pwsh",)

    monkeypatch.setattr(release_validation, "_platform", lambda: "posix")
    assert release_validation._selectors("both") == ("auto",)
    with pytest.raises(ValueError, match="uses Bash"):
        release_validation._selectors("pwsh")
    with pytest.raises(SystemExit) as error:
        release_validation.main(["--phase", "host-integration", "--shell", "powershell"])
    assert error.value.code == 2


def test_posix_host_dispatch_requires_bash_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: "posix")

    async def exercise() -> dict[str, object]:
        return {"selector": "auto", "platform": "posix", "family": "bash"}

    monkeypatch.setattr(release_validation, "_exercise_bash_host", exercise)
    assert release_validation.run_host_integration(("auto",)) == [
        {"selector": "auto", "platform": "posix", "family": "bash"}
    ]
    with pytest.raises(RuntimeError, match="default Bash selector"):
        release_validation.run_host_integration(("pwsh",))


@pytest.mark.skipif(os.name != "posix", reason="requires a real POSIX release host")
def test_real_posix_release_host_covers_permission_and_identity_cases() -> None:
    results = release_validation.run_host_integration(("auto",))
    assert len(results) == 1
    assert results[0]["platform"] == "posix"
    assert results[0]["status"] == "passed"
    cases = {str(case["name"]): case for case in cast(list[dict[str, object]], results[0]["cases"])}
    assert set(cases) == release_validation.POSIX_CASES
    for name in (
        "read-outside",
        "write-outside",
        "full-access-catastrophic",
        "identity-duplicate-path",
    ):
        assert cases[name]["decision"] == "confirm"
        assert cases[name]["confirmation"] == "declined"
        assert cases[name]["status"] == "refused"
    for name in ("read-inside", "write-inside", "full-access-ordinary", "identity-single-hit"):
        assert cases[name]["decision"] == "direct"
        assert cases[name]["exit_code"] == 0
        assert cases[name]["status"] == "success"


def test_coverage_rules_are_quantified_and_fail_closed() -> None:
    assert {rule.name for rule in COVERAGE_RULES} >= {
        "shell-selection",
        "whitelist-direct-fixtures",
        "dynamic-complex",
        "identity",
        "path-edges",
        "catastrophic",
        "inspector-failures",
        "full-access-dynamic",
        "file-read",
        "file-write",
        "web",
        "schedule",
        "mcp",
        "gateway-hard-errors",
        "web-redirect-rebinding",
        "textual-modal",
        "dream-exemption",
        "catalog-stability",
        "persistence-schema",
        "restore-path-matrix",
    }
    assert all(not hasattr(rule, "multiplier") for rule in COVERAGE_RULES)
    assert all(
        pattern.startswith("^") and pattern.endswith("$")
        for rule in COVERAGE_RULES
        for pattern in rule.patterns
    )
    failing = CoverageEvidence(
        collected_nodes=("tests/example.py::test_case",),
        counts={rule.name: 0 for rule in COVERAGE_RULES},
    )
    with pytest.raises(AssertionError, match="shell-selection"):
        failing.assert_minimums()


def test_coverage_uses_distinct_explicit_pytest_nodes_and_fails_on_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pattern = release_validation._node_pattern("tests/example.py", "test_matrix")
    monkeypatch.setattr(
        release_validation,
        "COVERAGE_RULES",
        (CoverageRule("explicit-matrix", 2, (pattern,)),),
    )
    first = "tests/example.py::test_matrix[read-only]"
    second = "tests/example.py::test_matrix[full-access]"

    evidence = build_coverage_evidence(
        (first, first, second, "tests/example.py::test_matrix_renamed[full-access]")
    )

    assert evidence.counts == {"explicit-matrix": 2}
    assert evidence.details["explicit-matrix"]["matched_nodes"] == [first, second]
    with pytest.raises(AssertionError, match="observed 1; minimum is 2"):
        build_coverage_evidence((first,))


def test_windows_release_entry_is_documented_in_readme() -> None:
    readme = Path(__file__).parents[1] / "README.md"

    assert "python scripts/release_validation.py --phase all" in readme.read_text(encoding="utf-8")


def test_workflow_requires_windows_report_for_release_gate() -> None:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "release-validation.yml"
    document = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert set(document["jobs"]) == {"windows-release", "release-gate"}
    job = document["jobs"]["windows-release"]
    assert job["runs-on"] == "windows-latest"
    windows_steps = job["steps"]
    commands = "\n".join(step.get("run", "") for step in windows_steps if isinstance(step, dict))
    assert "python scripts/release_validation.py --phase all" in " ".join(commands.split())
    assert "windows-release.json" in commands
    assert 'python -m pip install -e ".[dev]" "setuptools>=77"' in commands
    uploads = [
        step for step in windows_steps if step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert len(uploads) == 1
    assert uploads[0]["if"] == "always()"
    assert "windows-release.json" in uploads[0]["with"]["path"]

    assert any(step.get("name") == "Warm Windows PowerShell 5.1" for step in windows_steps)
    assert any(
        step.get("name") == "Warm Windows PowerShell host resolution" for step in windows_steps
    )

    gate = document["jobs"]["release-gate"]
    assert gate["needs"] == ["windows-release"]
    assert gate["if"] == "always()"
    assert gate["runs-on"] == "windows-latest"
    assert gate["steps"][0]["shell"] == "pwsh"
    command = gate["steps"][0]["run"]
    assert "if ('${{ needs.windows-release.result }}' -ne 'success') { exit 1 }" in command


def test_skip_classification_is_bound_to_exact_node_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: "windows")
    accepted = {
        "nodeid": (
            "tests/tools/core/test_exec_bash_policy.py::"
            "test_real_posix_bash_inspect_policy_execute_smoke"
        ),
        "message": "requires a real POSIX production host",
    }

    assert release_validation._classify_skip(accepted) == "waived-posix-host-scope"
    assert (
        release_validation._classify_skip(
            {
                **accepted,
                "nodeid": "tests/example.py::test_real_posix_bash_inspect_policy_execute_smoke",
            }
        )
        == "unclassified"
    )
    assert (
        release_validation._classify_skip(
            {**accepted, "message": "requires a real POSIX production host for another feature"}
        )
        == "unclassified"
    )


def _host_evidence(platform: str) -> list[dict[str, object]]:
    if platform == "windows":
        return [{"selector": "powershell"}, {"selector": "pwsh"}]
    return [
        {
            "selector": "auto",
            "platform": "posix",
            "family": "bash",
            "status": "passed",
            "cases": [{"name": name} for name in POSIX_CASES],
        }
    ]


@pytest.mark.parametrize("platform", ("windows", "posix"))
def test_quality_runs_complete_sequence_and_requires_platform_evidence(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: platform)
    monkeypatch.setattr(
        release_validation,
        "_windows_path_capability_evidence",
        lambda: {
            "junction": {"available": True},
            "hardlink": {"available": True},
            "file_symlink": {"available": True},
        },
    )
    suites: list[str] = []
    commands: list[list[str]] = []

    def pytest_report(paths: object, xml_path: Path, label: str) -> PytestEvidence:
        del xml_path
        suites.append(label)
        passed = (
            REQUIRED_WINDOWS_ALTERNATIVE_NODES
            if platform == "windows"
            else REQUIRED_POSIX_SMOKE_NODES
        )
        return PytestEvidence(
            label,
            tuple(cast(tuple[str, ...], paths)),
            10,
            10,
            tuple(passed) + tuple(RESTORE_PATH_MATRIX_NODES),
            (),
        )

    def command(parts: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        arguments = list(cast(list[str], parts))
        commands.append(arguments)
        if "--outdir" in arguments:
            output = Path(arguments[arguments.index("--outdir") + 1])
            (output / "myclaw-test.whl").write_text("fixture", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(release_validation, "_run_pytest_with_report", pytest_report)
    monkeypatch.setattr(release_validation, "_run_command", command)
    report = release_validation._run_quality(_host_evidence(platform))

    assert suites == ["targeted", "full"]
    assert commands[:3] == [
        [sys.executable, "-m", "ruff", "check", "myclaw", "tests", "scripts"],
        ["git", "diff", "--check"],
        [sys.executable, "-m", "mypy", "myclaw", "tests", "scripts"],
    ]
    assert commands[3][2:4] == ["build", "--no-isolation"]
    assert report["build"] == {"artifacts": ["myclaw-test.whl"]}
    assert report["host_integration"] == _host_evidence(platform)
    assert report["static"] == {
        "ruff_lint": "passed",
        "git_diff_check": "passed",
        "mypy": "passed",
    }


@pytest.mark.parametrize("platform", ("windows", "posix"))
def test_skip_gate_fails_closed_for_missing_hosts_and_unknown_skips(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: platform)
    path_evidence = {
        "junction": {"available": True},
        "hardlink": {"available": True},
        "file_symlink": {"available": True},
    }
    passed = (
        REQUIRED_WINDOWS_ALTERNATIVE_NODES if platform == "windows" else REQUIRED_POSIX_SMOKE_NODES
    )
    passed_nodes = tuple(passed) + tuple(RESTORE_PATH_MATRIX_NODES)
    with pytest.raises(RuntimeError, match=r"host integration|host evidence"):
        release_validation._validate_skips(
            (), host_results=(), path_evidence=path_evidence, passed_nodes=passed_nodes
        )
    with pytest.raises(RuntimeError, match="unclassified pytest skip"):
        release_validation._validate_skips(
            ({"nodeid": "tests/new.py::test_new", "message": "new skip"},),
            host_results=_host_evidence(platform),
            path_evidence=path_evidence,
            passed_nodes=passed_nodes,
        )
    with pytest.raises(RuntimeError, match="path matrix nodes did not pass"):
        release_validation._validate_skips(
            (),
            host_results=_host_evidence(platform),
            path_evidence=path_evidence,
            passed_nodes=tuple(passed),
        )
    if platform == "posix":
        with pytest.raises(RuntimeError, match="POSIX smoke nodes did not pass"):
            release_validation._validate_skips(
                (),
                host_results=_host_evidence(platform),
                path_evidence=path_evidence,
                passed_nodes=(),
            )
        with pytest.raises(RuntimeError, match="POSIX Bash host evidence"):
            release_validation._validate_skips(
                (),
                host_results=[{"selector": "auto", "platform": "posix", "family": "bash"}],
                path_evidence=path_evidence,
                passed_nodes=passed_nodes,
            )
    else:
        with pytest.raises(RuntimeError, match="junction capability"):
            release_validation._validate_skips(
                (),
                host_results=_host_evidence(platform),
                path_evidence={
                    "junction": {"available": False},
                    "hardlink": {"available": True},
                    "file_symlink": {"available": True},
                },
                passed_nodes=passed_nodes,
            )
        with pytest.raises(RuntimeError, match="file symlink capability"):
            release_validation._validate_skips(
                (),
                host_results=_host_evidence(platform),
                path_evidence={
                    "junction": {"available": True},
                    "hardlink": {"available": True},
                    "file_symlink": {"available": False},
                },
                passed_nodes=passed_nodes,
            )


def test_skip_allowlists_are_platform_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    posix_smoke_skip = {
        "nodeid": "tests/tools/core/test_exec_bash_policy.py::test_real_posix_bash_inspect_policy_execute_smoke",
        "message": "requires a real POSIX production host",
    }
    windows_junction_skip = {
        "nodeid": "tests/tools/core/test_directory_tools.py::test_directory_junction_roots_are_never_traversed",
        "message": "Windows junction behavior",
    }
    native_windows_skip = {
        "nodeid": "tests/test_windows_filesystem.py::test_require_owned_regular_file_returns_normalized_owned_path",
        "message": "requires native Windows paths",
    }
    host_case_skip = {
        "nodeid": "tests/test_host_filesystem.py::test_host_path_is_within_uses_host_case_rules",
        "message": "requires native Windows paths",
    }
    posix_mode_skip = {
        "nodeid": "tests/restore/test_backup_store.py::test_restore_store_directories_are_private_on_posix",
        "message": "POSIX mode bits are not available on Windows",
    }
    powershell_path_skip = {
        "nodeid": "tests/tools/core/test_exec_powershell_policy.py::test_windows_powershell_51_canonical_workspace_read_executes_directly",
        "message": "requires native Windows PowerShell paths",
    }
    monkeypatch.setattr(release_validation, "_platform", lambda: "windows")
    assert release_validation._classify_skip(posix_smoke_skip) == "waived-posix-host-scope"
    assert release_validation._classify_skip(posix_mode_skip) == "waived-posix-mode-scope"
    assert release_validation._classify_skip(windows_junction_skip) == "unclassified"
    monkeypatch.setattr(release_validation, "_platform", lambda: "posix")
    assert release_validation._classify_skip(posix_smoke_skip) == "unclassified"
    assert release_validation._classify_skip(posix_mode_skip) == "unclassified"
    assert (
        release_validation._classify_skip(windows_junction_skip) == "waived-windows-junction-scope"
    )
    assert (
        release_validation._classify_skip(native_windows_skip) == "waived-native-windows-path-scope"
    )
    assert release_validation._classify_skip(host_case_skip) == "waived-native-windows-path-scope"
    assert (
        release_validation._classify_skip(powershell_path_skip)
        == "waived-windows-powershell-path-scope"
    )
    assert (
        release_validation._classify_skip(
            {**native_windows_skip, "nodeid": "tests/new.py::test_windows_path"}
        )
        == "unclassified"
    )


@pytest.mark.parametrize("platform", ("windows", "posix"))
def test_artifact_smoke_uses_platform_venv_paths(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(release_validation, "_platform", lambda: platform)
    commands: list[list[str]] = []

    def command(parts: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        arguments = list(cast(list[str], parts))
        commands.append(arguments)
        if "--outdir" in arguments:
            output = Path(arguments[arguments.index("--outdir") + 1])
            (output / "myclaw-test.whl").write_text("fixture", encoding="utf-8")
        elif arguments[1:3] == ["-m", "venv"]:
            folder = Path(arguments[-1]) / ("Scripts" if platform == "windows" else "bin")
            folder.mkdir(parents=True)
            (folder / ("python.exe" if platform == "windows" else "python")).touch()
            (folder / ("myclaw.exe" if platform == "windows" else "myclaw")).touch()
        help_text = "MyClaw Personal Agent runtime" if arguments[-1] == "--help" else ""
        return subprocess.CompletedProcess(arguments, 0, help_text, "")

    def smoke(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        arguments = cast(list[str], args[0])
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "marker": "ARTIFACT_CONFIG_SMOKE_OK",
                    "module_path": str(Path(arguments[0]).parents[1] / "site-packages" / "myclaw"),
                    "environment_prefix": str(Path(arguments[0]).parents[1]),
                }
            ),
            "",
        )

    monkeypatch.setattr(release_validation, "_run_command", command)
    monkeypatch.setattr(subprocess, "run", smoke)
    result = release_validation._smoke_installed_wheel(tmp_path / "fixture.whl", tmp_path)
    expected_folder = "Scripts" if platform == "windows" else "bin"
    expected_entry = "myclaw.exe" if platform == "windows" else "myclaw"
    assert Path(cast(str, result["entry_point"])).parts[-2:] == (expected_folder, expected_entry)
    assert commands[-1][-1] == "--help"
    assert Path(cast(str, result["cwd"])).name == "smoke-cwd"


def test_artifact_smoke_preserves_report_contract_without_duplicate_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = {"marker": "ARTIFACT_CONFIG_SMOKE_OK", "wheel": "fixture.whl"}
    distribution = {"direct_install": direct, "rebuilt_install": {"wheel": "rebuilt.whl"}}
    monkeypatch.setattr(release_validation, "_run_distribution_validation", lambda: distribution)
    assert release_validation._run_artifact_smoke() == {**direct, "distribution": distribution}


def test_artifact_program_rejects_source_tree_import() -> None:
    environment = dict(os.environ)
    environment["MYCLAW_SOURCE_ROOT"] = str(release_validation.ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.prefix = {str(release_validation.ROOT)!r}\n"
            + release_validation._ARTIFACT_SMOKE_PROGRAM,
        ],
        cwd=release_validation.ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_windows_skip_alternatives_are_explicit_full_suite_nodes() -> None:
    assert len(REQUIRED_WINDOWS_ALTERNATIVE_NODES) >= 10
    assert all(
        node.startswith("tests/") and ".py::test_" in node
        for node in REQUIRED_WINDOWS_ALTERNATIVE_NODES
    )


def test_command_failure_includes_sanitized_process_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=9,
            stdout="standard output",
            stderr="standard error",
        ),
    )

    with pytest.raises(RuntimeError, match=r"standard output.*standard error"):
        release_validation._run_command(("tool", "argument"))


def test_command_timeout_is_a_nonzero_gate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd=("tool",), timeout=7)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="timed out after 7s"):
        release_validation._run_command(("tool",), timeout=7)


def test_coverage_evidence_is_json_serializable() -> None:
    evidence = CoverageEvidence(
        collected_nodes=("tests/example.py::test_case",),
        counts={"shell-selection": 11},
    )
    payload = evidence.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["collected_nodes"] == ["tests/example.py::test_case"]


def test_pytest_evidence_report_preserves_passed_nodes() -> None:
    evidence = PytestEvidence(
        label="full",
        paths=("tests",),
        total=2,
        passed=1,
        passed_nodes=("tests/example.py::test_case",),
        skips=(
            {
                "suite": "full",
                "nodeid": "tests/example.py::test_skipped",
                "message": "host limitation",
            },
        ),
    )

    assert evidence.to_dict() == {
        "label": "full",
        "paths": ["tests"],
        "total": 2,
        "passed": 1,
        "passed_nodes": ["tests/example.py::test_case"],
        "skipped": 1,
        "skips": [
            {
                "suite": "full",
                "nodeid": "tests/example.py::test_skipped",
                "message": "host limitation",
            }
        ],
    }


def test_acceptance_matrix_keeps_backend_and_external_scopes_separate() -> None:
    matrix = build_acceptance_matrix(
        (
            "tests/service/test_service_concurrency.py::test_two_cli_clients_complete_distinct_sessions_through_transport",
            "tests/service/test_service_concurrency.py::test_distinct_sessions_run_in_parallel_and_cancel_is_scoped",
            "tests/service/test_service_concurrency.py::test_claim_race_denies_loser_content_over_http_events_and_reconnect",
        ),
        installed_statuses={"R02": "passed"},
    )
    assert [item["id"] for item in matrix] == [f"R{index:02d}" for index in range(1, 18)]
    r02 = matrix[1]
    assert r02["overall"] == "passed"
    scopes = cast(dict[str, dict[str, object]], r02["scopes"])
    assert scopes["backend_service"]["status"] == "passed"
    assert scopes["installed_cli_browser"]["status"] == "passed"
    assert scopes["production_browser"]["status"] == "not-run"
    assert scopes["backend_service"]["command"] == "python -m pytest -q"
    nodes = cast(list[dict[str, object]], scopes["backend_service"]["nodes"])
    assert str(nodes[0]["nodeid"]).endswith(
        "test_two_cli_clients_complete_distinct_sessions_through_transport"
    )

    parameterized = build_acceptance_matrix(
        (
            "tests/service/test_service_concurrency.py::"
            "test_event_reconnect_replays_once_and_cache_overflow_requires_snapshot",
            "tests/service/test_service_concurrency.py::"
            "test_replay_holds_live_events_until_cached_events_are_sent",
            "tests/service/test_service_concurrency.py::"
            "test_snapshot_resync_includes_selected_and_switched_away_claims",
            "tests/service/test_service_concurrency.py::"
            "test_client_expiry_keeps_claim_until_cancelled_run_cleanup_finishes",
            "tests/service/test_runtime_management.py::"
            "test_permission_resets_only_after_client_expiry_at_thirty_seconds",
            "tests/service/test_service_foundation.py::"
            "test_reacquired_claim_rejects_the_previous_version[fixture]",
        )
    )[2]
    parameterized_scopes = cast(dict[str, dict[str, object]], parameterized["scopes"])
    parameterized_node = cast(
        list[dict[str, object]], parameterized_scopes["backend_service"]["nodes"]
    )[5]
    assert parameterized_scopes["backend_service"]["status"] == "passed"
    assert parameterized_node["result"] == "passed"
    assert parameterized_node["matched_nodes"] == [
        "tests/service/test_service_foundation.py::"
        "test_reacquired_claim_rejects_the_previous_version[fixture]"
    ]


def test_release_failure_writes_partial_report_and_keeps_fail_fast_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "release.json"
    monkeypatch.setattr(release_validation, "_platform", lambda: "windows")
    monkeypatch.setattr(
        release_validation,
        "_source_identity",
        lambda: {"head": "fixture-head", "dirty": True},
    )
    monkeypatch.setattr(
        release_validation,
        "_host_capabilities",
        lambda: {"platform": "windows", "shells": {}},
    )

    def phase(
        current: ReleasePhase,
        shell_option: str,
        *,
        host_results: object = None,
    ) -> dict[str, object]:
        del shell_option, host_results
        if current is ReleasePhase.HOST_INTEGRATION:
            raise release_validation.ReleaseBlockedError("fixture link capability missing")
        if current is ReleasePhase.COVERAGE:
            return {"coverage": {"collected_nodes": ["tests/example.py::test_case"]}}
        raise AssertionError(f"unexpected phase {current}")

    monkeypatch.setattr(release_validation, "_run_named_phase", phase)

    assert (
        release_validation.main(
            ["--phase", "all", "--shell", "both", "--report", str(report_path)]
        )
        == 1
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_schema_version"] == 2
    assert report["status"] == "blocked"
    assert report["source"] == {"head": "fixture-head", "dirty": True}
    assert report["coverage"]["collected_nodes"] == ["tests/example.py::test_case"]
    assert report["execution"]["executed_phases"] == ["coverage", "host-integration"]
    assert report["execution"]["not_run_phases"] == ["quality", "artifact-smoke"]
    assert report["execution"]["remaining_gates"] == [
        "quality",
        "artifact-smoke",
        "host-capability: fixture link capability missing",
    ]
    assert report["failure"]["type"] == "ReleaseBlockedError"
    assert "fixture link capability missing" in report["failure"]["message"]


def test_reported_command_failure_retains_command_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = release_validation._ReportRecorder("quality", "both")
    token = release_validation._REPORT_CONTEXT.set(recorder)
    try:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=17,
                stdout="original stdout",
                stderr="original stderr",
            ),
        )
        with pytest.raises(RuntimeError, match=r"original stdout.*original stderr"):
            release_validation._run_command(("fixture-tool", "--case", "failure"))
    finally:
        release_validation._REPORT_CONTEXT.reset(token)

    assert recorder.commands == [
        {
            "command": ["fixture-tool", "--case", "failure"],
            "rendered": "fixture-tool --case failure",
            "cwd": str(release_validation.ROOT),
            "status": "failed",
            "exit_code": 17,
            "failure_output": "original stdoutoriginal stderr",
        }
    ]


def test_blocked_report_promotes_partial_quality_evidence() -> None:
    recorder = release_validation._ReportRecorder("all", "both")
    recorder.begin_phase("quality")
    recorder.add_phase_evidence(
        "quality",
        {
            "path_capability": {"file_symlink": {"available": False}},
            "pytest": {
                "full": {
                    "total": 3,
                    "passed": 2,
                    "failed": 1,
                    "failed_nodes": ["tests/example.py::test_failure"],
                    "skipped": 0,
                }
            },
        },
    )
    error = release_validation.ReleaseBlockedError("file symlink capability missing")
    recorder.finish_phase("quality", status="blocked", error=error)

    report = recorder.build(payload=None, error=error)
    pytest_report = cast(dict[str, object], report["pytest"])
    full_report = cast(dict[str, object], pytest_report["full"])
    path_report = cast(dict[str, object], report["path_capability"])
    file_symlink = cast(dict[str, object], path_report["file_symlink"])
    execution = cast(dict[str, object], report["execution"])
    phases = cast(dict[str, object], execution["phases"])
    quality_phase = cast(dict[str, object], phases["quality"])
    quality_evidence = cast(dict[str, object], quality_phase["evidence"])
    assert full_report["failed_nodes"] == ["tests/example.py::test_failure"]
    assert file_symlink["available"] is False
    assert quality_evidence["pytest"] == pytest_report


def test_working_tree_identity_tracks_content_deletions_and_untracked_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "scripts" / "probe.py"
    source.parent.mkdir()
    source.write_text("original", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("scripts/ignored/\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release test",
            "-c",
            "user.email=release@example.invalid",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(release_validation, "ROOT", tmp_path)
    original = release_validation._working_tree_identity()
    identity = release_validation._source_identity()
    assert len(str(identity["head_tree"])) == 40
    assert identity["working_tree"] == original
    assert release_validation._working_tree_identity() == original
    source.write_text("modified", encoding="utf-8")
    modified = release_validation._working_tree_identity()
    assert original["sha256"] != modified["sha256"]
    untracked = tmp_path / "scripts" / "new_probe.py"
    untracked.write_text("new source", encoding="utf-8")
    added = release_validation._working_tree_identity()
    assert added["sha256"] != modified["sha256"]
    source.unlink()
    deleted = release_validation._working_tree_identity()
    records = cast(list[dict[str, object]], deleted["files"])
    assert (
        next(record for record in records if record["path"] == "scripts/probe.py")["kind"]
        == "missing"
    )
    assert deleted["sha256"] != added["sha256"]
    ignored = source.parent / "ignored"
    ignored.mkdir()
    (ignored / "large-tree.txt").write_text("ignored", encoding="utf-8")
    plan = tmp_path / "docs" / "plans"
    plan.mkdir(parents=True)
    (plan / "user-plan.md").write_text("user-owned", encoding="utf-8")
    assert release_validation._working_tree_identity() == deleted


def test_reported_timeout_preserves_original_failure_and_redacts_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = release_validation._ReportRecorder("all", "both")
    recorder.begin_phase("quality")
    token = release_validation._REPORT_CONTEXT.set(recorder)

    def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired(
            cmd=("fixture", "--api-key", "command-secret"),
            timeout=7,
            output=b"provider failed with raw-env-secret; Bearer bearer-secret",
            stderr="http://localhost/#ticket=ticket-secret api_key='key-secret'",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    try:
        with pytest.raises(RuntimeError) as caught:
            release_validation._run_command(
                ("fixture", "--api-key", "command-secret"),
                timeout=7,
                env={"FIXTURE_API_KEY": "raw-env-secret"},
            )
        recorder.finish_phase("quality", status="failed", error=caught.value)
        command = recorder.commands[0]
        assert command["status"] == "failed"
        assert "exit_code" not in command
        failure = cast(dict[str, object], command["failure"])
        assert failure["type"] == "TimeoutExpired"
        assert failure["timeout_seconds"] == 7
        phase_failure = cast(dict[str, object], recorder.phases["quality"]["failure"])
        assert cast(dict[str, object], phase_failure["cause"])["type"] == "TimeoutExpired"
        encoded = json.dumps([recorder.commands, recorder.phases, str(caught.value)])
        for secret in (
            "command-secret",
            "raw-env-secret",
            "bearer-secret",
            "ticket-secret",
            "key-secret",
        ):
            assert secret not in encoded
        assert "provider failed" in encoded
        assert "secret with spaces" not in release_validation._redact_report_text(
            'api_key="secret with spaces"'
        )
        assert release_validation._safe_command(("fixture", "--api-key=secret with spaces")) == [
            "fixture",
            "--api-key=[redacted]",
        ]
        assert (
            release_validation._redact_report_text("service credential not observed")
            == "service credential not observed"
        )
    finally:
        release_validation._REPORT_CONTEXT.reset(token)


def test_exception_payload_preserves_context_and_bounds_cycles() -> None:
    try:
        raise OSError("original OS failure")
    except OSError:
        try:
            raise RuntimeError("cleanup failure")
        except RuntimeError as error:
            payload = release_validation._exception_payload(error)
    assert cast(dict[str, object], payload["context"])["type"] == "OSError"
    cyclic = RuntimeError("cycle")
    cyclic.__cause__ = cyclic
    assert release_validation._exception_payload(cyclic)["cause"] == {
        "type": "RuntimeError",
        "cycle": True,
    }


def test_exception_payload_redacts_unlabelled_secrets_without_report_context() -> None:
    error = subprocess.TimeoutExpired(
        cmd=("fixture", "--api-key", "naked-command-secret"),
        timeout=7,
        output=b"naked-command-secret naked-environment-secret",
        stderr="naked-command-secret",
    )
    payload = release_validation._exception_payload(error, secrets=("naked-environment-secret",))
    serialized = json.dumps(payload)
    assert "naked-command-secret" not in serialized
    assert "naked-environment-secret" not in serialized
    assert payload["type"] == "TimeoutExpired"
    assert payload["timeout_seconds"] == 7


def test_acceptance_matrix_requires_executed_artifact_evidence_for_r17() -> None:
    statuses = {"R17": "passed"}
    missing = build_acceptance_matrix((), browser_statuses=statuses, installed_statuses=statuses)[
        -1
    ]
    completed = build_acceptance_matrix(
        (), browser_statuses=statuses, installed_statuses=statuses, artifact_status="passed"
    )[-1]
    assert missing["id"] == completed["id"] == "R17"
    assert missing["overall"] == "partial"
    assert completed["overall"] == "passed"
