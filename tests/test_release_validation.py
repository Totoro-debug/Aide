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


def test_workflow_requires_both_platform_reports_for_release_gate() -> None:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "release-validation.yml"
    document = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert set(document["jobs"]) == {"windows-release", "posix-release", "release-gate"}
    for job_name, runner, report in (
        ("windows-release", "windows-latest", "windows-release.json"),
        ("posix-release", "ubuntu-24.04", "posix-release.json"),
    ):
        job = document["jobs"][job_name]
        assert job["runs-on"] == runner
        commands = "\n".join(step.get("run", "") for step in job["steps"] if isinstance(step, dict))
        assert "python scripts/release_validation.py --phase all" in " ".join(commands.split())
        assert report in commands
        assert 'python -m pip install -e ".[dev]" "setuptools>=77"' in commands
        uploads = [
            step for step in job["steps"] if step.get("uses") == "actions/upload-artifact@v4"
        ]
        assert len(uploads) == 1
        assert uploads[0]["if"] == "always()"
        assert report in uploads[0]["with"]["path"]

    windows_steps = document["jobs"]["windows-release"]["steps"]
    assert any(step.get("name") == "Warm Windows PowerShell 5.1" for step in windows_steps)
    assert any(
        step.get("name") == "Warm Windows PowerShell host resolution" for step in windows_steps
    )

    gate = document["jobs"]["release-gate"]
    assert gate["needs"] == ["windows-release", "posix-release"]
    assert gate["if"] == "always()"
    command = gate["steps"][0]["run"]
    assert "test '${{ needs.windows-release.result }}' = success" in command
    assert "test '${{ needs.posix-release.result }}' = success" in command


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
    assert commands[:4] == [
        [sys.executable, "-m", "ruff", "check", "myclaw", "tests", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        ["git", "diff", "--check"],
        [sys.executable, "-m", "mypy", "myclaw", "tests", "scripts"],
    ]
    assert commands[4][2:4] == ["build", "--no-isolation"]
    assert report["build"] == {"artifacts": ["myclaw-test.whl"]}
    assert report["host_integration"] == _host_evidence(platform)
    assert report["static"] == {
        "ruff_lint": "passed",
        "ruff_format": "passed",
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
    powershell_path_skip = {
        "nodeid": "tests/tools/core/test_exec_powershell_policy.py::test_windows_powershell_51_canonical_workspace_read_executes_directly",
        "message": "requires native Windows PowerShell paths",
    }
    monkeypatch.setattr(release_validation, "_platform", lambda: "windows")
    assert release_validation._classify_skip(posix_smoke_skip) == "waived-posix-host-scope"
    assert release_validation._classify_skip(windows_junction_skip) == "unclassified"
    monkeypatch.setattr(release_validation, "_platform", lambda: "posix")
    assert release_validation._classify_skip(posix_smoke_skip) == "unclassified"
    assert (
        release_validation._classify_skip(windows_junction_skip) == "waived-windows-junction-scope"
    )
    assert (
        release_validation._classify_skip(native_windows_skip) == "waived-native-windows-path-scope"
    )
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
    result = release_validation._run_artifact_smoke()
    expected_folder = "Scripts" if platform == "windows" else "bin"
    expected_entry = "myclaw.exe" if platform == "windows" else "myclaw"
    assert Path(cast(str, result["entry_point"])).parts[-2:] == (expected_folder, expected_entry)
    assert commands[-1][-1] == "--help"
    assert Path(cast(str, result["cwd"])).name == "smoke-cwd"


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
