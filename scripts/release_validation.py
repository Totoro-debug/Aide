"""Auditable Windows release validation for Session Restore and tools."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Literal, cast

from omni.agent.permission import PermissionSnapshot
from omni.agent.tools.context import ToolRunContext
from omni.agent.tools.core.exec import ExecTool
from omni.agent.tools.core.exec_host import (
    PowerShellExecHost,
    ResolvedExecShell,
    resolve_exec_shell,
)
from omni.agent.tools.core.exec_policy import ExecAssessment
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import ModelToolCall, ToolGateway
from omni.utils.platform import WINDOWS_REQUIRED_ERROR, is_windows_host

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_SECONDS: Final[int] = 1_800
POWERSHELL_FLAGS: Final[tuple[str, ...]] = (
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
)
RELEASE_SHELLS: Final[tuple[str, ...]] = ("powershell", "pwsh")


def _platform() -> Literal["windows"]:
    return "windows"


COLLECTION_PATHS: Final[tuple[str, ...]] = (
    "tests/tools/core/test_exec_host.py",
    "tests/tools/core/test_exec_powershell_policy.py",
    "tests/tools/test_permission_file_matrix.py",
    "tests/tools/core/test_web_fetch_network_authorization.py",
    "tests/tools/core/test_schedule.py",
    "tests/tools/test_permission_contract.py",
    "tests/scheduling/test_schedule_background_confirmation.py",
    "tests/scheduling/test_schedule_store.py",
    "tests/scheduling/test_schedule_model.py",
    "tests/scheduling/test_schedule_dream.py",
    "tests/agent/test_confirmation.py",
    "tests/terminal/test_conversation.py",
    "tests/restore",
    "tests/sessions/test_session_resume.py",
    "tests/test_permission_loop.py",
    "tests/tools/test_mcp.py",
    "tests/tools/test_tool_search.py",
    "tests/agent/test_fixed_catalog.py",
    "tests/agent/test_context.py",
    "tests/test_cli.py",
    "tests/memory/test_dream.py::test_dream_edit_response_is_terminal_without_a_confirmation_request",
    "tests/tools/test_fixed_tool_gateway.py::test_mcp_catalog_exposure_activation_and_search_ignore_permission_level",
    "tests/tools/test_models.py::test_normalized_tool_result_serializes_the_exact_artifact_shape",
    "tests/sessions/test_session.py::test_persist_writes_one_complete_compact_utf8_snapshot_atomically",
)

TARGETED_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_release_contract.py",
    "tests/configuration",
    "tests/tools",
    "tests/agent",
    "tests/restore",
    "tests/sessions",
    "tests/scheduling",
    "tests/management",
    "tests/terminal",
    "tests/architecture",
)


class ReleasePhase(StrEnum):
    """One independently runnable release validation phase."""

    COVERAGE = "coverage"
    HOST_INTEGRATION = "host-integration"
    QUALITY = "quality"
    ARTIFACT_SMOKE = "artifact-smoke"
    ALL = "all"


class ReleaseBlockedError(RuntimeError):
    """A release phase could not run because the host lacks a required capability."""


PHASE_ORDER: Final[tuple[str, ...]] = tuple(
    phase.value for phase in ReleasePhase if phase is not ReleasePhase.ALL
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


_SENSITIVE_NAME = r"(?:api[_-]?key|access[_-]?token|reconnect[_-]?credential|web[_-]?control[_-]?credential|credential|csrf[_-]?token|password|secret|ticket|cookie|authorization)"


def _redact_report_text(value: str, secrets: Sequence[str] = ()) -> str:
    """Retain diagnostics while removing authentication values from reports/logs."""
    recorder = _REPORT_CONTEXT.get()
    known = (*secrets, *(recorder.secret_values if recorder is not None else ()))
    for secret in sorted(known, key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"(?i)(Bearer\s+)[^\s\"'<>]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(Cookie\s*:\s*)[^\r\n]+", r"\1[redacted]", value)
    value = re.sub(
        rf"(?i)({_SENSITIVE_NAME}[\"']?\s*[:=]\s*)([\"'])(.*?)\2",
        lambda match: f"{match[1]}{match[2]}[redacted]{match[2]}",
        value,
    )
    value = re.sub(
        rf"(?i)({_SENSITIVE_NAME}[\"']?\s*[:=]\s*[\"']?)([^\s\"',;<>]+)",
        r"\1[redacted]",
        value,
    )
    return re.sub(rf"(?i)(--{_SENSITIVE_NAME}\s+)([^\s\"',;<>]+)", r"\1[redacted]", value)


def _safe_command(command: Sequence[str]) -> list[str]:
    # Redact the rendered command as well as individual options split across argv.
    parts = [str(part) for part in command]
    for index, part in enumerate(parts):
        if index and re.fullmatch(rf"(?i)--{_SENSITIVE_NAME}", parts[index - 1]):
            parts[index] = "[redacted]"
        elif re.match(rf"(?i)--{_SENSITIVE_NAME}=", part):
            parts[index] = part.split("=", 1)[0] + "=[redacted]"
        else:
            parts[index] = _redact_report_text(part)
    return parts


def _exception_payload(
    error: BaseException, *, secrets: Sequence[str] = (), _seen: set[int] | None = None
) -> dict[str, object]:
    """Keep the original failure tree inspectable without requiring a traceback parser."""
    seen = set() if _seen is None else _seen
    if id(error) in seen:
        return {"type": type(error).__name__, "cycle": True}
    seen.add(id(error))
    command_secrets: list[str] = []
    if isinstance(error, (subprocess.TimeoutExpired, subprocess.CalledProcessError)) and isinstance(
        error.cmd, (list, tuple)
    ):
        command_secrets = [
            str(raw)
            for raw, safe in zip(error.cmd, _safe_command(error.cmd), strict=True)
            if str(raw) != safe
        ]
    payload: dict[str, object] = {
        "type": type(error).__name__,
        "message": _redact_report_text(str(error) or repr(error), (*secrets, *command_secrets)),
    }
    if isinstance(error, BaseExceptionGroup):
        payload["children"] = [
            _exception_payload(child, secrets=secrets, _seen=seen) for child in error.exceptions
        ]
    if isinstance(error, (subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        command = error.cmd
        payload["command"] = (
            _safe_command(command)
            if isinstance(command, (list, tuple))
            else _redact_report_text(str(command))
        )
        for name in ("stdout", "stderr"):
            output = getattr(error, name, None)
            if output:
                text = (
                    output.decode("utf-8", errors="replace")
                    if isinstance(output, bytes)
                    else output
                )
                payload[name] = _redact_report_text(text, (*secrets, *command_secrets))[-12_000:]
        if isinstance(error, subprocess.TimeoutExpired):
            payload["timeout_seconds"] = error.timeout
        else:
            payload["exit_code"] = error.returncode
    if error.__cause__ is not None:
        payload["cause"] = _exception_payload(error.__cause__, secrets=secrets, _seen=seen)
    elif error.__context__ is not None and not error.__suppress_context__:
        payload["context"] = _exception_payload(error.__context__, secrets=secrets, _seen=seen)
    return payload


@dataclass(slots=True)
class _ReportRecorder:
    """Capture commands and partial phase state while retaining fail-fast behavior."""

    requested_phase: str
    shell: str
    phases: dict[str, dict[str, object]] = field(default_factory=dict)
    commands: list[dict[str, object]] = field(default_factory=list)
    partial_payload: dict[str, object] = field(default_factory=dict)
    failure: dict[str, object] | None = None
    secret_values: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        for name in PHASE_ORDER:
            self.phases[name] = {"status": "not-run"}

    def begin_phase(self, name: str) -> None:
        self.phases[name] = {"status": "running", "started_at": _utc_now()}

    def add_phase_evidence(self, name: str, evidence: Mapping[str, object]) -> None:
        entry = self.phases.setdefault(name, {"status": "running"})
        detached = dict(evidence)
        entry["evidence"] = detached
        self.partial_payload.update(detached)

    def finish_phase(
        self,
        name: str,
        *,
        status: Literal["passed", "failed", "blocked"],
        result: Mapping[str, object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        entry = self.phases.setdefault(name, {})
        entry["status"] = status
        entry["finished_at"] = _utc_now()
        if result is not None:
            entry["result"] = dict(result)
            self.partial_payload.update(result)
        if error is not None:
            entry["failure"] = _exception_payload(error)

    def start_command(self, command: Sequence[str], *, cwd: Path) -> int:
        self.commands.append(
            {
                "command": _safe_command(command),
                "rendered": _redact_report_text(subprocess.list2cmdline(_safe_command(command))),
                "cwd": str(cwd),
                "status": "running",
            }
        )
        return len(self.commands) - 1

    def finish_command(
        self,
        index: int,
        *,
        status: Literal["passed", "failed"],
        exit_code: int | None = None,
        failure_output: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        command = self.commands[index]
        command["status"] = status
        if exit_code is not None:
            command["exit_code"] = exit_code
        if failure_output:
            command["failure_output"] = _redact_report_text(failure_output)[-12_000:]
        if error is not None:
            command["failure"] = _exception_payload(error)

    def build(
        self, *, payload: Mapping[str, object] | None, error: BaseException | None
    ) -> dict[str, object]:
        status = "passed"
        if error is not None:
            status = "blocked" if isinstance(error, ReleaseBlockedError) else "failed"
        report: dict[str, object] = {
            "report_schema_version": 2,
            "phase": self.requested_phase,
            "status": status,
            "source": _source_identity(),
            "host_capabilities": _host_capabilities(),
            "execution": {
                "requested_phase": self.requested_phase,
                "shell": self.shell,
                "phase_order": list(PHASE_ORDER),
                "phases": self.phases,
                "executed_phases": [
                    name for name, entry in self.phases.items() if entry.get("status") != "not-run"
                ],
                "not_run_phases": [
                    name for name, entry in self.phases.items() if entry.get("status") == "not-run"
                ],
                "remaining_gates": [
                    name for name, entry in self.phases.items() if entry.get("status") == "not-run"
                ],
            },
            "commands": self.commands,
            "cleanup_errors": [],
        }
        report.update(self.partial_payload)
        if payload is not None:
            report.update(payload)
        if error is not None:
            report["failure"] = _exception_payload(error)
            if isinstance(error, BaseExceptionGroup):
                report["cleanup_errors"] = [
                    _exception_payload(child)
                    for child in error.exceptions
                    if "cleanup" in str(child).casefold()
                    or "cleanup" in type(child).__name__.casefold()
                ]
            remaining = cast(dict[str, object], report["execution"])["remaining_gates"]
            assert isinstance(remaining, list)
            if isinstance(error, ReleaseBlockedError):
                remaining.append(f"host-capability: {_redact_report_text(str(error))}")
        return report


_REPORT_CONTEXT: contextvars.ContextVar[_ReportRecorder | None] = contextvars.ContextVar(
    "omni_release_report", default=None
)


def _source_identity() -> dict[str, object]:
    """Return source and built Web identity without making identity a release gate."""
    identity: dict[str, object] = {"root": str(ROOT)}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("head_tree", ("git", "rev-parse", "HEAD^{tree}")),
        ("branch", ("git", "branch", "--show-current")),
        ("status", ("git", "status", "--porcelain")),
    ):
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            identity[name] = {"error": _exception_payload(error)}
        else:
            value = result.stdout.strip()
            identity[name] = value if name != "status" else value.splitlines()
            if result.returncode != 0:
                identity[f"{name}_error"] = result.stderr.strip()
    status = identity.get("status")
    identity["dirty"] = bool(status)
    try:
        identity["working_tree"] = _working_tree_identity()
    except (OSError, subprocess.SubprocessError) as error:
        identity["working_tree"] = {"error": _exception_payload(error)}
    manifest = ROOT / "omni" / "web_assets" / "manifest.json"
    if manifest.is_file():
        identity["web_asset_manifest"] = {
            "path": str(manifest),
            "sha256": sha256(manifest.read_bytes()).hexdigest(),
            "bytes": manifest.stat().st_size,
        }
    return identity


def _working_tree_identity() -> dict[str, object]:
    """Hash tracked content and unignored implementation inputs, including deletions."""
    paths: set[str] = set()
    for selector in ("--cached", "--others"):
        result = subprocess.run(
            ("git", "ls-files", "-z", selector, "--exclude-standard"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
        for relative in result.stdout.decode("utf-8").split("\0"):
            if relative and (
                selector == "--cached"
                or relative.split("/", 1)[0] in {"omni", "scripts", "tests", "web", ".github"}
            ):
                paths.add(relative)
    records: list[dict[str, object]] = []
    for relative in sorted(paths):
        # The user's local plan is outside the release input contract.
        if relative.startswith("docs/plans/"):
            continue
        path = ROOT / relative
        record: dict[str, object] = {"path": relative}
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8")
            record["kind"] = "symlink"
        elif path.is_file():
            content = path.read_bytes()
            record["kind"] = "file"
        else:
            content = b""
            record["kind"] = "missing"
        record["sha256"] = sha256(content).hexdigest()
        record["bytes"] = len(content)
        records.append(record)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "files": records}


def _host_capabilities() -> dict[str, object]:
    """Describe discoverable hosts; phase results separately prove execution."""
    shells: dict[str, object] = {}
    for selector in RELEASE_SHELLS:
        executable = _find_windows_shell(selector)
        shell_record: dict[str, object] = {
            "executable": executable,
            "available": executable is not None,
        }
        shells[selector] = shell_record
        if executable is not None:
            try:
                shell = _resolve_windows_shell(selector)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                shell_record["resolution_error"] = _exception_payload(error)
            else:
                shell_record["version"] = (
                    list(shell.version) if shell.version is not None else None
                )
    return {
        "platform": "windows",
        "shells": shells,
    }


@dataclass(frozen=True, slots=True)
class CoverageRule:
    """A quantified, name-based coverage contract over collected pytest nodes."""

    name: str
    minimum: int
    patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageEvidence:
    """Serializable coverage evidence and the nodes that produced each count."""

    collected_nodes: tuple[str, ...]
    counts: Mapping[str, int]
    details: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def assert_minimums(self) -> None:
        """Fail closed when a quantified category is absent or below its floor."""
        for rule in COVERAGE_RULES:
            observed = self.counts.get(rule.name, 0)
            if observed < rule.minimum:
                raise AssertionError(
                    f"coverage rule {rule.name!r} observed {observed}; minimum is {rule.minimum}"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible detached representation."""
        return {
            "collected_nodes": list(self.collected_nodes),
            "counts": dict(self.counts),
            "details": {name: dict(detail) for name, detail in self.details.items()},
        }


@dataclass(frozen=True, slots=True)
class PytestEvidence:
    """One executed pytest suite with normalized node and skip evidence."""

    label: str
    paths: tuple[str, ...]
    total: int
    passed: int
    passed_nodes: tuple[str, ...]
    skips: tuple[Mapping[str, str], ...]
    failed_nodes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "paths": list(self.paths),
            "total": self.total,
            "passed": self.passed,
            "passed_nodes": list(self.passed_nodes),
            "skipped": len(self.skips),
            "skips": [dict(skip) for skip in self.skips],
        }
        if self.failed_nodes:
            payload["failed"] = len(self.failed_nodes)
            payload["failed_nodes"] = list(self.failed_nodes)
        return payload


@dataclass(frozen=True, slots=True)
class AcceptanceScenario:
    """One R01-R17 scenario with separately auditable evidence scopes."""

    scenario_id: str
    requirement: str
    required_scopes: tuple[str, ...]
    backend_nodes: tuple[str, ...]
    backend_assertions: tuple[str, ...]
    browser_command: str
    browser_evidence: tuple[str, ...]
    installed_command: str
    installed_evidence: tuple[str, ...]


_INSTALLED_VALIDATION_COMMAND = (
    "python -m scripts.installed_web_validation --output <external-output-dir>"
)
_PRODUCTION_BROWSER_COMMAND = "npm --prefix web run test:e2e"
_STARTUP_BROWSER_COMMAND = "npm --prefix web run test:e2e:startup"
_FULL_PYTEST_COMMAND = "python -m pytest -q"


ACCEPTANCE_SCENARIOS: Final[tuple[AcceptanceScenario, ...]] = (
    AcceptanceScenario(
        "R01",
        "Two CLI starts share one service; unrelated port occupants are not killed.",
        ("backend_service", "installed_cli_browser"),
        (
            "tests/service/test_service_transport.py::test_two_processes_start_or_join_one_service",
            "tests/service/test_service_transport.py::test_startup_does_not_stop_an_unrelated_port_occupant",
            "tests/service/test_service_transport.py::test_unrelated_listener_never_receives_service_credential",
        ),
        (
            "only one service instance owns the fixed port",
            "an unrelated listener remains alive and receives no service credential",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs launches omni web and asserts the isolated service URL",
            "the production browser path does not itself start two CLI processes",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "scripts/installed_web_validation.py::_run_installed_cli_competition cold-starts two installed metadata console entries with Textual headless I/O and asserts one PID/instance/port",
            "the installed omni web executable refuses a real unrelated fixed-port listener without killing it or disclosing credentials",
        ),
    ),
    AcceptanceScenario(
        "R02",
        "Web and CLI produce in distinct Sessions; same-Session Claim loses for one client.",
        ("backend_service", "installed_cli_browser"),
        (
            "tests/service/test_service_concurrency.py::test_two_cli_clients_complete_distinct_sessions_through_transport",
            "tests/service/test_service_concurrency.py::test_distinct_sessions_run_in_parallel_and_cancel_is_scoped",
            "tests/service/test_service_concurrency.py::test_claim_race_denies_loser_content_over_http_events_and_reconnect",
        ),
        (
            "distinct Session outputs complete without cross-session cancellation",
            "the losing Claim receives session_claimed and no Session body/events",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs rejects copied-tab Claim and keeps the available Session body private",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "web/scripts/installed-web-e2e.mjs holds distinct CLI/browser provider requests together before release and asserts both completed JSONL histories",
            "web/scripts/installed_cli_probe.py rejects the browser Session Claim and its history GET without returning the private body",
        ),
    ),
    AcceptanceScenario(
        "R03",
        "Replay and snapshot resync are bounded; expiry and stale Claims fail closed.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_concurrency.py::test_event_reconnect_replays_once_and_cache_overflow_requires_snapshot",
            "tests/service/test_service_concurrency.py::test_replay_holds_live_events_until_cached_events_are_sent",
            "tests/service/test_service_concurrency.py::test_snapshot_resync_includes_selected_and_switched_away_claims",
            "tests/service/test_service_concurrency.py::test_client_expiry_keeps_claim_until_cancelled_run_cleanup_finishes",
            "tests/service/test_runtime_management.py::test_permission_resets_only_after_client_expiry_at_thirty_seconds",
            "tests/service/test_service_foundation.py::test_reacquired_claim_rejects_the_previous_version",
        ),
        (
            "reconnect replays each cached event once and overflow requires a snapshot",
            "cached events precede live events; an offline completed Session snapshot matches persisted messages exactly, and the other Run completes once after the snapshot",
            "the controlled 29/30-second expiry boundary preserves then cancels ownership",
            "a stale Claim version cannot submit a command",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs observes socket close/reconnect and asserts one accepted Run",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "the installed cross-client helper has no 29/30-second replay assertion; only backend evidence is available",
        ),
    ),
    AcceptanceScenario(
        "R04",
        "Session switching preserves the old Run, releases the unselected Claim, and drops empty drafts.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_concurrency.py::test_switch_keeps_active_claim_until_run_terminates_then_releases_it",
            "tests/service/test_service_concurrency.py::test_duplicate_command_request_id_does_not_start_a_second_run",
            "tests/service/test_service_transport.py::test_project_session_http_scope_claim_and_empty_draft_contract",
        ),
        (
            "switch waits for the active Run before releasing its Claim",
            "duplicate request_id starts no second Run",
            "empty draft release creates no JSONL history",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs releases an Empty draft and asserts no draft JSONL remains",
            "web/scripts/e2e-runner.mjs hands an available Session to the second browser client",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed browser/CLI run covers distinct Sessions but does not exercise empty-draft release",
        ),
    ),
    AcceptanceScenario(
        "R05",
        "Competing clients resolve one confirmation and execute the exact Tool at most once.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_foundation.py::test_foreground_confirmation_is_broadcast_to_workspace_clients_and_resolved_once",
            "tests/service/test_service_foundation.py::test_confirmation_resolved_cannot_overtake_requested_for_another_client",
            "tests/service/test_service_foundation.py::test_background_confirmation_broadcast_has_job_source_without_session_scope",
            "tests/service/test_runtime_management.py::test_management_revalidates_claim_after_waiting_for_client_lock",
        ),
        (
            "one confirmation resolution wins and the other token is invalid",
            "background confirmation carries job source without a foreground Session",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs records confirmation source, resolution, and persisted terminal messages",
            "web/scripts/settings-e2e.mjs keeps an existing Tool confirmation usable during config save",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "web/scripts/installed-joint-e2e.mjs races two Web Clients on one background confirmation: one acceptance, one confirmation_resolved, both dialogs close, and one persisted exact Tool result",
        ),
    ),
    AcceptanceScenario(
        "R06",
        "Unselected Jobs keep running; project removal drains activity and preserves files/Jobs.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_concurrency.py::test_due_job_runs_once_in_unselected_registered_project",
            "tests/service/test_service_concurrency.py::test_project_removal_cancels_foreground_and_schedule_runs",
            "tests/service/test_service_foundation.py::test_project_removal_closes_admission_clears_claims_and_blocks_reentry",
            "tests/service/test_service_foundation.py::test_project_removal_notifies_unattached_web_requester_without_blocking_reconnect",
        ),
        (
            "an unselected registered project dispatches one due Job",
            "removal leaves zero foreground/Schedule activity and closes admission",
            "the directory and saved Job definitions remain on disk",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs removes a registered project and asserts the removal notice plus detached registration",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed joint removal cancels unreleased CLI foreground and Schedule barriers, verifies both persisted turn_cancelled outcomes, CLI notification/Claim release, and preserved user files/Jobs",
        ),
    ),
    AcceptanceScenario(
        "R07",
        "Removed/re-registered Jobs stay paused across restart until explicit resume.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_foundation.py::test_project_schedule_stays_paused_across_service_restart_until_resumed",
            "tests/service/test_service_foundation.py::test_removed_project_re_registration_keeps_saved_jobs_paused_until_resume",
        ),
        (
            "restart does not admit saved Jobs automatically",
            "explicit resume is required before Schedule dispatch",
        ),
        _STARTUP_BROWSER_COMMAND,
        (
            "web/scripts/config-startup-e2e.mjs asserts available=false, awaiting_resume, preserved saved Jobs, and config_invalid session admission",
            "web/scripts/config-startup-e2e.mjs asserts post-repair awaiting_resume and an enabled Resume schedule control",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        ("installed lifecycle does not exercise restart-time Job admission",),
    ),
    AcceptanceScenario(
        "R08",
        "Client loss and explicit stop drain the service without starting new Jobs.",
        ("backend_service", "installed_cli_browser"),
        (
            "tests/service/test_service_foundation.py::test_last_client_grace_pauses_and_restarts_schedule",
            "tests/service/test_service_foundation.py::test_service_stop_aborts_confirmation_and_invalidates_token",
            "tests/service/test_service_concurrency.py::test_client_expiry_keeps_claim_until_cancelled_run_cleanup_finishes",
        ),
        (
            "the last-client grace pauses then restarts Schedule",
            "explicit stop invalidates pending confirmation and exits",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs uses the real service lifecycle controller and asserts reconnect/stop cleanup",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed grace reconnects the same Client at 29 seconds, then disconnects the last active Client and verifies autonomous 30-second cancellation, discovery removal, and port release without explicit stop",
            "the separate installed Web lifecycle verifies the explicit service stop entry",
        ),
    ),
    AcceptanceScenario(
        "R09",
        "Session rename persists; delete is scoped and rejects busy/unfinished Sessions.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_service_transport.py::test_workspace_session_rename_requires_claim_and_persists_metadata_version",
            "tests/service/test_service_transport.py::test_workspace_session_delete_requires_confirmation_and_cleans_only_session_data",
            "tests/service/test_service_concurrency.py::test_session_delete_rejects_active_work_and_restore_barriers",
        ),
        (
            "rename persists metadata with Claim and revision checks",
            "delete removes only target Session data after confirmation",
            "busy or Restore-barrier deletion is rejected",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs exercises filtered rename and conflict retry",
            "web/scripts/e2e-runner.mjs asserts delete confirmation, focus, retry, and preservation of another Session",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        ("installed lifecycle does not exercise rename/delete management",),
    ),
    AcceptanceScenario(
        "R10",
        "Restore overwrites only the target while its status bubble times out without new history.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_restore_management.py::test_restore_plan_and_cancel_are_bound_to_inspected_session",
            "tests/service/test_restore_management.py::test_other_client_cannot_inspect_execute_read_or_ack_restore",
            "tests/service/test_config_generation_acceptance.py::test_save_preserves_real_restore_transaction",
        ),
        (
            "Restore inspect/execute/cancel are Claim and Session scoped",
            "another client cannot read or acknowledge the Restore result",
            "config generation waits for the Restore transaction and preserves the restored Session",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs covers Restore preview, cancel, overwrite, stale response, failure acknowledgement, and refresh",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        ("installed lifecycle does not exercise Restore UI",),
    ),
    AcceptanceScenario(
        "R11",
        "Management operations have Web paths, permissions are isolated, and Job history paginates.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_schedule_jobs_transport.py::test_schedule_jobs_http_crud_validates_all_kinds_and_is_cross_client_idempotent",
            "tests/service/test_schedule_jobs_transport.py::test_schedule_job_history_groups_existing_schedule_session_and_paginates",
            "tests/service/test_schedule_jobs_transport.py::test_schedule_job_http_delete_cancels_a_running_job_and_keeps_deleted_state",
            "tests/service/test_runtime_management.py::test_typed_http_actions_require_auth_csrf_and_matching_client_identity",
        ),
        (
            "Schedule CRUD is typed, idempotent, and Claim scoped",
            "history returns grouped pages with a cursor",
            "typed management operations reject wrong identity/authentication",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs loads real Schedule history, 20 groups, cursor pagination, and retry focus",
            "web/scripts/e2e-runner.mjs exercises at/every/cron CRUD, lost-ack retry, deletion, and status polling",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed lifecycle only validates status/settings/conversation routes, not management pages",
        ),
    ),
    AcceptanceScenario(
        "R12",
        "All config fields are editable; secrets never leave the server; invalid/stale writes preserve bytes.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_config_security_acceptance.py::test_config_http_auth_csrf_client_and_unknown_fields_preserve_bytes",
            "tests/service/test_config_security_acceptance.py::test_http_invalid_complete_candidate_and_field_errors_do_not_leak_secrets",
            "tests/service/test_config_transport.py::test_config_patch_edits_models_routes_mcp_and_write_only_secrets",
            "tests/service/test_config_transport.py::test_config_patch_secret_clear_is_explicit_and_preserves_bytes_on_conflict",
        ),
        (
            "auth/CSRF/unknown-field and stale writes preserve exact config bytes",
            "existing secrets are absent from API, events, errors, and diagnostics",
            "model/route/MCP fields and write-only secret operations are typed",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/settings-e2e.mjs asserts stale model save leaves bytes unchanged",
            "web/scripts/settings-e2e.mjs records structured secret replace/keep/clear and invalid-byte evidence",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "scripts/installed_web_validation.py visits Settings and asserts the installed page/API, but does not edit secrets",
        ),
    ),
    AcceptanceScenario(
        "R13",
        "Config saves preserve startup resources and accepted work; saved settings activate after restart.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_config_generation_acceptance.py::test_http_save_preserves_foreground_schedule_confirmation_and_admission",
            "tests/service/test_config_generation_lifecycle.py::test_external_changes_preserve_resources_and_admission",
            "tests/service/test_config_generation_acceptance.py::test_save_preserves_dream_and_subsequent_dream_uses_startup_settings",
            "tests/service/test_config_generation_acceptance.py::test_save_preserves_real_restore_transaction",
            "tests/service/test_config_transport.py::test_config_patch_reports_restart_required_and_stale_conflict",
            "tests/service/test_config_transport.py::test_config_save_preserves_workspace_and_later_activation_uses_startup_settings",
        ),
        (
            "save leaves active work and WebSocket connected",
            "saved settings require restart; active resources and admission remain usable",
            "stale revision is rejected without byte mutation",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/settings-e2e.mjs keeps an active Tool confirmation through save",
            "web/scripts/settings-e2e.mjs records save/restart with real model and MCP resources",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed joint save holds real foreground/Schedule requests, verifies natural completion, saved and active revisions remain distinct, and the next CLI Run retains the startup model",
        ),
    ),
    AcceptanceScenario(
        "R14",
        "Missing, route-invalid, and malformed TOML states remain repairable without starting runtime work.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_config_startup_repair.py::test_missing_configuration_keeps_service_online_but_blocks_runtime",
            "tests/service/test_config_startup_repair.py::test_malformed_configuration_is_repaired_after_exact_private_backup",
            "tests/service/test_config_startup_repair.py::test_malformed_repair_backup_failure_leaves_original_bytes_untouched",
        ),
        (
            "missing config blocks Agent/Schedule while service remains online",
            "malformed config is backed up byte-for-byte before repair",
            "backup failure leaves original bytes untouched",
        ),
        _STARTUP_BROWSER_COMMAND,
        (
            "web/scripts/config-startup-e2e.mjs runs missing/semantic-invalid/malformed states, repair, backup bytes, and no secret exposure",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        ("installed lifecycle starts from a valid config and does not exercise repair mode",),
    ),
    AcceptanceScenario(
        "R15",
        "Origin/Host/auth/CSRF/WebSocket/path traversal attacks are rejected without side effects.",
        ("backend_service", "production_browser"),
        (
            "tests/service/test_config_security_acceptance.py::test_config_http_auth_csrf_client_and_unknown_fields_preserve_bytes",
            "tests/service/test_runtime_management.py::test_typed_http_actions_require_auth_csrf_and_matching_client_identity",
            "tests/service/test_service_transport.py::test_browser_ticket_is_one_time_cookie_auth_and_static_routes_are_bounded",
            "tests/service/test_config_security_acceptance.py::test_config_http_ws_errors_and_logs_never_expose_existing_secrets",
        ),
        (
            "invalid auth/CSRF/client identity and one-time ticket requests are rejected",
            "bounded static routes return 404 rather than the SPA document",
            "WebSocket/config errors contain no existing secrets",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs rejects a consumed browser ticket with 401",
            "web/scripts/installed-web-e2e.mjs asserts CSP, asset MIME/cache, and 404 missing resources",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        ("installed browser evidence records CSP, MIME/cache, and 404 assertions",),
    ),
    AcceptanceScenario(
        "R16",
        "Four locale/theme combinations, three viewports, and keyboard focus paths remain usable.",
        ("production_browser",),
        (),
        (
            "no horizontal overflow or sidebar/content overlap at 1440x900, 1024x768, and 768x1024",
            "keyboard dialog Escape returns focus to the trigger",
        ),
        _PRODUCTION_BROWSER_COMMAND,
        (
            "web/scripts/e2e-runner.mjs checks en/zh-CN x light/dark x three viewports and geometry",
            "web/scripts/e2e-runner.mjs checks keyboard dialog focus restoration",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "installed browser evidence validates a production route and conversation, not the full locale/theme matrix",
        ),
    ),
    AcceptanceScenario(
        "R17",
        "External direct/rebuilt wheel runs Web and CLI with no Node/npm in the installed runtime.",
        ("artifact_smoke", "installed_cli_browser"),
        ("tests/test_release_contract.py::test_clean_distributions_build_and_import_cleanly",),
        ("clean sdist and wheel build and import without source-tree dependence",),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "scripts/installed_web_validation.py runs direct and sdist-rebuilt wheels in isolated venvs",
            "web/scripts/installed-web-e2e.mjs records API, deep route, Settings, WebSocket, conversation, and 404 results",
        ),
        _INSTALLED_VALIDATION_COMMAND,
        (
            "web/scripts/installed_cli_probe.py asserts installed omni console entry, no node/npm, same service discovery, and JSONL assistant persistence",
            "direct and rebuilt reports both record stop=passed and port/discovery cleanup",
        ),
    ),
)


def build_acceptance_matrix(
    passed_nodes: Sequence[str],
    *,
    browser_statuses: Mapping[str, str] | None = None,
    installed_statuses: Mapping[str, str] | None = None,
    artifact_status: str = "not-run",
) -> list[dict[str, object]]:
    """Render R01-R17 with exact scopes instead of a single aggregate E2E claim."""
    passed = {node.replace("\\", "/") for node in passed_nodes}
    browser = browser_statuses or {}
    installed = installed_statuses or {}
    matrix: list[dict[str, object]] = []
    for scenario in ACCEPTANCE_SCENARIOS:
        backend_nodes = []
        for node in scenario.backend_nodes:
            matched_nodes = sorted(
                actual for actual in passed if actual == node or actual.startswith(f"{node}[")
            )
            backend_nodes.append(
                {
                    "nodeid": node,
                    "result": "passed" if matched_nodes else "not-run",
                    "matched_nodes": matched_nodes,
                }
            )
        backend_status = (
            "passed"
            if scenario.backend_nodes
            and all(
                any(actual == node or actual.startswith(f"{node}[") for actual in passed)
                for node in scenario.backend_nodes
            )
            else "not-run"
            if scenario.backend_nodes
            else "not-applicable"
        )
        scope_statuses = {
            "backend_service": backend_status,
            "artifact_smoke": artifact_status,
            "production_browser": browser.get(scenario.scenario_id, "not-run"),
            "installed_cli_browser": installed.get(scenario.scenario_id, "not-run"),
        }
        required_statuses = [scope_statuses[scope] for scope in scenario.required_scopes]
        overall = "passed" if all(status == "passed" for status in required_statuses) else "partial"
        if all(status in {"not-run", "not-applicable"} for status in required_statuses):
            overall = "not-run"
        matrix.append(
            {
                "id": scenario.scenario_id,
                "requirement": scenario.requirement,
                "overall": overall,
                "required_scopes": list(scenario.required_scopes),
                "scopes": {
                    "backend_service": {
                        "status": backend_status,
                        "command": _FULL_PYTEST_COMMAND,
                        "nodes": backend_nodes,
                        "assertions": list(scenario.backend_assertions),
                    },
                    "production_browser": {
                        "status": scope_statuses["production_browser"],
                        "command": scenario.browser_command,
                        "evidence": list(scenario.browser_evidence),
                    },
                    "installed_cli_browser": {
                        "status": scope_statuses["installed_cli_browser"],
                        "command": scenario.installed_command,
                        "evidence": list(scenario.installed_evidence),
                    },
                    "artifact_smoke": {
                        "status": scope_statuses["artifact_smoke"],
                        "command": "python scripts/release_validation.py --phase artifact-smoke",
                    },
                },
            }
        )
    return matrix


def _node_pattern(path: str, test_name: str) -> str:
    return rf"^{re.escape(path)}::{re.escape(test_name)}(?:\[.*\])?$"


def _node_patterns(path: str, *test_names: str) -> tuple[str, ...]:
    return tuple(_node_pattern(path, test_name) for test_name in test_names)


RESTORE_PATH_MATRIX_PATTERNS: Final[tuple[str, ...]] = _node_patterns(
    "tests/restore/test_path_matrix.py",
    "test_restore_path_matrix_restores_existing_new_and_external_targets",
    "test_restore_path_matrix_does_not_follow_a_retargeted_link",
    "test_restore_path_matrix_reports_a_file_failure_and_truncates_session",
)
RESTORE_PATH_MATRIX_NODES: Final[frozenset[str]] = frozenset(
    {
        "tests/restore/test_path_matrix.py::test_restore_path_matrix_restores_existing_new_and_external_targets[existing]",
        "tests/restore/test_path_matrix.py::test_restore_path_matrix_restores_existing_new_and_external_targets[new]",
        "tests/restore/test_path_matrix.py::test_restore_path_matrix_restores_existing_new_and_external_targets[external]",
        "tests/restore/test_path_matrix.py::test_restore_path_matrix_does_not_follow_a_retargeted_link",
        "tests/restore/test_path_matrix.py::test_restore_path_matrix_reports_a_file_failure_and_truncates_session",
    }
)


COVERAGE_RULES: Final[tuple[CoverageRule, ...]] = (
    CoverageRule(
        "shell-selection",
        7,
        _node_patterns(
            "tests/tools/core/test_exec_host.py",
            "test_resolve_exec_shell_covers_platform_and_fallback_contract",
        ),
    ),
    CoverageRule(
        "whitelist-direct-fixtures",
        30,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_every_approved_powershell_candidate_has_a_direct_fixture",
            "test_every_cross_host_git_read_form_has_a_direct_powershell_fixture",
        ),
    ),
    CoverageRule(
        "dynamic-complex",
        24,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_low_permission_powershell_dynamic_or_unknown_calls_confirm_once",
            "test_full_access_executes_parseable_noncatastrophic_powershell_dynamic_code",
            "test_powershell_inspection_boundaries_fail_closed_without_running_user_source",
        ),
    ),
    CoverageRule(
        "identity",
        9,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_noncanonical_powershell_identity_categories_confirm",
            "test_incomplete_or_inconsistent_identity_payload_is_typed_uncertainty",
            "test_powershell_inspection_returns_canonical_identity_metadata",
        ),
    ),
    CoverageRule(
        "path-edges",
        20,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_powershell_path_roles_follow_level_and_canonical_containment",
            "test_powershell_read_through_workspace_reparse_point_confirms",
        )
        + _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_windows_path_case_uses_host_case_insensitive_containment",
            "test_file_facts_use_canonical_host_paths_and_explicit_roles",
            "test_linked_skill_root_and_missing_write_descendant_remain_external",
        ),
    ),
    CoverageRule(
        "catastrophic",
        15,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_catastrophic_powershell_calls_confirm_once_at_every_permission_level",
        ),
    ),
    CoverageRule(
        "inspector-failures",
        18,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_shell_present_inspector_uncertainty_confirms_at_every_level",
        ),
    ),
    CoverageRule(
        "full-access-dynamic",
        3,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_full_access_executes_parseable_noncatastrophic_powershell_dynamic_code",
        ),
    ),
    CoverageRule(
        "file-read",
        24,
        _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_foreground_file_read_matrix_requests_only_external_reads",
        ),
    ),
    CoverageRule(
        "file-write",
        12,
        _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_foreground_file_write_matrix",
        ),
    ),
    CoverageRule(
        "web",
        18,
        _node_patterns(
            "tests/tools/core/test_web_fetch_network_authorization.py",
            "test_web_fetch_target_level_matrix",
        ),
    ),
    CoverageRule(
        "web-redirect-rebinding",
        6,
        _node_patterns(
            "tests/tools/core/test_web_fetch_network_authorization.py",
            "test_web_fetch_declining_an_unsafe_initial_target_sends_zero_request_bytes",
            "test_web_fetch_public_to_private_redirect_uses_one_popup_and_audited_addresses",
            "test_web_fetch_redirect_decline_sends_no_bytes_to_unsafe_target",
            "test_web_fetch_binds_to_the_audited_dns_answer_without_a_second_resolution",
            "test_web_fetch_rejects_an_unaudited_peer_before_sending_bytes",
            "test_aiohttp_client_ignores_system_rebinding_and_environment_proxy",
        ),
    ),
    CoverageRule(
        "schedule",
        9,
        _node_patterns(
            "tests/tools/test_permission_contract.py",
            "test_schedule_policy_maps_every_action_and_current_level",
        ),
    ),
    CoverageRule(
        "mcp",
        8,
        _node_patterns(
            "tests/tools/test_mcp.py",
            "test_foreground_mcp_permission_level_controls_each_call",
            "test_low_permission_mcp_repeated_calls_request_independent_approvals",
            "test_mcp_parse_and_lookup_errors_precede_permission_at_every_level",
            "test_mcp_schedule_context_calls_directly_without_confirmation",
        ),
    ),
    CoverageRule(
        "gateway-hard-errors",
        15,
        _node_patterns(
            "tests/tools/test_permission_contract.py",
            "test_hard_and_business_errors_do_not_open_permission_or_confirmation",
        )
        + _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_hard_errors_are_level_invariant_and_never_confirm",
        )
        + _node_patterns(
            "tests/tools/core/test_schedule.py",
            "test_schedule_hard_errors_precede_permission_at_every_level",
        )
        + _node_patterns(
            "tests/tools/test_mcp.py",
            "test_mcp_parse_and_lookup_errors_precede_permission_at_every_level",
        )
        + _node_patterns(
            "tests/tools/core/test_web_fetch_network_authorization.py",
            "test_web_fetch_connection_errors_remain_errors_at_every_level",
            "test_web_fetch_dns_failure_is_an_error_at_every_permission_level",
        ),
    ),
    CoverageRule(
        "confirmation-coordinator",
        8,
        _node_patterns(
            "tests/agent/test_confirmation.py",
            "test_active_foreground_does_not_preempt_background_and_foreground_is_prioritized",
            "test_active_background_finishes_before_queued_foreground_then_background",
            "test_each_queue_keeps_fifo_order",
            "test_async_presenter_is_stopped_and_producer_cancellation_stays_cancelled",
            "test_presenter_failures_fail_closed_and_advance_the_queue",
            "test_owner_and_generation_cancellation_raise_typed_abort",
            "test_duplicate_and_late_decisions_cannot_resolve_a_later_item",
            "test_producer_cancellation_removes_queued_item_and_active_item_dismisses",
            "test_request_has_no_runtime_timeout_and_no_presenter_fails_closed",
        ),
    ),
    CoverageRule(
        "textual-modal",
        3,
        _node_patterns(
            "tests/terminal/test_conversation.py",
            "test_coordinator_background_confirmation_uses_stable_modal_projection",
            "test_coordinator_cancellation_before_message_delivery_cannot_open_a_stale_modal",
            "test_coordinator_open_modal_is_aborted_and_drained_on_unmount",
        ),
    ),
    CoverageRule(
        "schedule-lifecycle",
        8,
        _node_patterns(
            "tests/scheduling/test_schedule_background_confirmation.py",
            "test_confirmation_abort_commits_safe_terminal_lifecycle",
            "test_delete_cancels_and_drains_the_exact_active_occurrence",
            "test_delete_persists_absence_before_aborting_pending_confirmation",
            "test_generation_abort_drain_reports_terminal_store_failure",
            "test_generation_abort_drain_waits_for_terminal_store_commit",
            "test_agent_loop_records_confirmation_abort_and_preserves_its_type",
        ),
    ),
    CoverageRule(
        "title-migration",
        5,
        _node_patterns(
            "tests/scheduling/test_schedule_store.py",
            "test_exact_old_schema_derives_title_and_rewrites_on_next_successful_mutation",
            "test_exact_old_dream_schema_uses_the_fixed_title",
            "test_public_removal_detects_title_only_changes",
        )
        + _node_patterns(
            "tests/scheduling/test_schedule_model.py",
            "test_dream_title_is_fixed_even_when_its_message_is_unstable",
            "test_schedule_job_derives_title_from_the_first_nonempty_message_line",
        ),
    ),
    CoverageRule(
        "runtime-snapshot",
        12,
        _node_patterns(
            "tests/scheduling/test_schedule_background_confirmation.py",
            "test_user_occurrence_captures_one_immutable_permission_snapshot_at_admission",
            "test_snapshot_schedule_context_applies_mcp_confirmation_policy",
            "test_snapshot_schedule_context_applies_file_policy",
            "test_snapshot_schedule_context_keeps_uncertain_exec_confirmation",
            "test_snapshot_schedule_context_applies_web_fetch_policy",
            "test_agent_loop_reuses_occurrence_snapshot_for_context_gateway_and_envelope",
        )
        + _node_patterns(
            "tests/agent/test_context.py",
            "test_foreground_runtime_context_projects_the_permission_snapshot",
            "test_schedule_runtime_context_projects_the_permission_snapshot",
        ),
    ),
    CoverageRule(
        "dream-exemption",
        1,
        _node_patterns(
            "tests/memory/test_dream.py",
            "test_dream_edit_response_is_terminal_without_a_confirmation_request",
        ),
    ),
    CoverageRule(
        "catalog-stability",
        3,
        _node_patterns(
            "tests/tools/test_fixed_tool_gateway.py",
            "test_mcp_catalog_exposure_activation_and_search_ignore_permission_level",
        ),
    ),
    CoverageRule(
        "persistence-schema",
        3,
        _node_patterns(
            "tests/tools/test_models.py",
            "test_normalized_tool_result_serializes_the_exact_artifact_shape",
        )
        + _node_patterns(
            "tests/sessions/test_session.py",
            "test_persist_writes_one_complete_compact_utf8_snapshot_atomically",
        )
        + _node_patterns(
            "tests/scheduling/test_schedule_background_confirmation.py",
            "test_user_occurrence_captures_one_immutable_permission_snapshot_at_admission",
        ),
    ),
    CoverageRule("restore-path-matrix", 5, RESTORE_PATH_MATRIX_PATTERNS),
)


def collect_pytest_nodes() -> tuple[str, ...]:
    """Collect the release-relevant pytest nodes without executing them."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *COLLECTION_PATHS]
    result = _run_command(command, timeout=300)
    nodes: list[str] = []
    for line in result.stdout.splitlines():
        candidate = line.strip().replace("\\", "/")
        if candidate.startswith("tests/") and "::test_" in candidate:
            nodes.append(candidate)
    if not nodes:
        raise RuntimeError("pytest collection produced no release-relevant test nodes")
    return tuple(dict.fromkeys(nodes))


def build_coverage_evidence(nodes: Sequence[str]) -> CoverageEvidence:
    """Map collected node IDs to every matching quantified coverage rule."""
    normalized_nodes = tuple(dict.fromkeys(node.replace("\\", "/") for node in nodes))
    counts: dict[str, int] = {}
    details: dict[str, Mapping[str, object]] = {}
    for rule in COVERAGE_RULES:
        matched = tuple(
            node
            for node in normalized_nodes
            if any(re.fullmatch(pattern, node) is not None for pattern in rule.patterns)
        )
        observed = len(matched)
        counts[rule.name] = observed
        details[rule.name] = {
            "matched_nodes": list(matched),
            "collected_count": len(matched),
            "minimum": rule.minimum,
            "observed_count": observed,
        }
    evidence = CoverageEvidence(
        collected_nodes=normalized_nodes,
        counts=counts,
        details=details,
    )
    evidence.assert_minimums()
    return evidence


def _find_windows_shell(selector: str) -> str | None:
    overrides = {
        "powershell": "OMNI_POWERSHELL_PATH",
        "pwsh": "OMNI_PWSH_PATH",
    }
    candidates: list[Path] = []
    override = os.environ.get(overrides[selector])
    if override:
        candidates.append(Path(override))
    located = shutil.which(selector)
    if located:
        candidates.append(Path(located))
    if selector == "powershell":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates.append(windir / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")
    else:
        candidates.extend(
            (
                Path(r"C:\Program Files\PowerShell\7\pwsh.exe"),
                Path.home() / "scoop" / "apps" / "pwsh" / "current" / "pwsh.exe",
            )
        )
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.name.casefold() == f"{selector}.exe":
                return str(candidate.resolve())
        except OSError:
            continue
    return None


def _resolve_windows_shell(selector: str) -> ResolvedExecShell:
    if selector not in RELEASE_SHELLS:
        raise ValueError(f"unsupported release shell: {selector}")
    executable = _find_windows_shell(selector)
    if executable is None:
        raise ReleaseBlockedError(f"{selector} executable was not found")
    typed_selector = cast(Literal["powershell", "pwsh"], selector)
    shell = resolve_exec_shell(
        typed_selector,
        platform="windows",
        which=lambda name: executable if name.casefold() == selector else None,
        environment=os.environ,
    )
    if not shell.available or shell.executable is None or shell.flags != POWERSHELL_FLAGS:
        raise ReleaseBlockedError(f"{selector} did not resolve to an available PowerShell host")
    return shell


async def _exercise_powershell_host(selector: str) -> dict[str, object]:
    shell = _resolve_windows_shell(selector)
    if shell.version is None:
        raise ReleaseBlockedError(f"{selector} did not report a version")
    if selector == "powershell" and shell.version[:2] != (5, 1):
        raise ReleaseBlockedError("powershell did not resolve to Windows PowerShell 5.1")
    if selector == "pwsh" and shell.version < (7,):
        raise ReleaseBlockedError("pwsh did not resolve to PowerShell 7 or newer")
    with tempfile.TemporaryDirectory(prefix=f"omni-{selector}-") as temporary:
        workspace = Path(temporary).resolve()
        host = PowerShellExecHost(shell)
        spec = host.process_spec(workspace)
        if (
            spec.executable != shell.executable
            or spec.flags != shell.flags
            or spec.cwd != workspace
            or spec.environment != shell.environment
        ):
            raise RuntimeError(
                f"{selector} changed process inputs between resolution and execution"
            )
        assessment: ExecAssessment = await host.inspect("Get-Location", workspace)
        if assessment.uncertain or not assessment.command_identities:
            raise RuntimeError(f"{selector} inspection was uncertain")
        outcome = await host.execute_assessed(
            "Get-Location",
            workspace,
            10,
            assessment=assessment,
        )
        if outcome.exit_code != 0 or outcome.timed_out:
            raise RuntimeError(f"{selector} canonical command did not execute successfully")
        try:
            output_lines = tuple(
                line.strip() for line in outcome.stdout.decode("utf-8").splitlines() if line.strip()
            )
            observed_cwd = Path(output_lines[-1]).resolve(strict=True)
        except (IndexError, OSError, UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(f"{selector} returned an invalid Get-Location path") from error
        if observed_cwd != workspace:
            raise RuntimeError(
                f"{selector} executed in {observed_cwd} instead of requested cwd {workspace}"
            )
        gateway = ToolGateway._for_memory(
            (ExecTool(host=host),),
            tool_context=ToolRunContext(workspace=workspace, exec_host=host),
            permission_context=PermissionContext.from_snapshot(
                PermissionSnapshot(level="full-access", exec_shell=shell),
                workspace_root=workspace,
            ),
        )
        dynamic_result = await gateway.call(
            ModelToolCall(
                id=f"release-{selector}",
                name="exec",
                arguments=json.dumps({"command": "& { Get-Location }"}),
            )
        )
        if dynamic_result.status != "success":
            raise RuntimeError(f"{selector} Full-Access dynamic command failed")
        return {
            "selector": selector,
            "platform": "windows",
            "family": shell.family,
            "version": list(shell.version),
            "executable": shell.executable,
            "flags": list(shell.flags),
            "cwd": str(workspace),
            "environment_keys": sorted(shell.env),
            "inspection": {
                "command": "Get-Location",
                "status": assessment.inspector_status,
                "syntax_uncertain": assessment.syntax_uncertain,
                "identity_count": len(assessment.command_identities),
            },
            "canonical_execution": {
                "command": "Get-Location",
                "exit_code": outcome.exit_code,
                "timed_out": outcome.timed_out,
                "observed_cwd": str(observed_cwd),
                "matches_process_spec": observed_cwd == spec.cwd,
            },
            "full_access_dynamic_command": "& { Get-Location }",
            "full_access_dynamic_status": dynamic_result.status,
        }


def run_host_integration(selectors: Sequence[str]) -> list[dict[str, object]]:
    """Run real inspection and execution on the current release host."""
    return [asyncio.run(_exercise_powershell_host(selector)) for selector in selectors]


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    recorder = _REPORT_CONTEXT.get()
    secrets = tuple(
        value
        for key, value in (os.environ if env is None else env).items()
        if value and re.search(_SENSITIVE_NAME, key, flags=re.IGNORECASE)
    )
    if recorder is not None:
        recorder.secret_values.update(secrets)
    rendered = subprocess.list2cmdline(_safe_command(command))
    command_index = recorder.start_command(command, cwd=cwd) if recorder is not None else None
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            env=None if env is None else dict(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        if recorder is not None and command_index is not None:
            recorder.finish_command(
                command_index,
                status="failed",
                failure_output=f"command timed out after {timeout}s: {rendered}",
                error=error,
            )
        raise RuntimeError(f"command timed out after {timeout}s: {rendered}") from error
    except OSError as error:
        if recorder is not None and command_index is not None:
            recorder.finish_command(command_index, status="failed", error=error)
        raise
    if result.returncode != 0:
        output = _redact_report_text((result.stdout + result.stderr).strip(), secrets)
        if len(output) > 12_000:
            output = output[-12_000:]
        if recorder is not None and command_index is not None:
            recorder.finish_command(
                command_index,
                status="failed",
                exit_code=result.returncode,
                failure_output=output,
            )
        suffix = f"\n{output}" if output else ""
        raise RuntimeError(f"command failed with exit {result.returncode}: {rendered}{suffix}")
    if recorder is not None and command_index is not None:
        recorder.finish_command(command_index, status="passed", exit_code=result.returncode)
    return result


def _junit_nodeid(case: ET.Element) -> str:
    classname = case.attrib.get("classname", "").replace(".", "/")
    name = case.attrib.get("name", "")
    suffix = ".py" if classname and not classname.endswith(".py") else ""
    return f"{classname}{suffix}::{name}"


def _parse_pytest_evidence(
    xml_path: Path,
    *,
    label: str,
    paths: Sequence[str],
) -> PytestEvidence:
    root = ET.parse(xml_path).getroot()
    skips: list[dict[str, str]] = []
    passed_nodes: list[str] = []
    failed_nodes: list[str] = []
    total = 0
    for case in root.iter("testcase"):
        total += 1
        nodeid = _junit_nodeid(case)
        skipped = case.find("skipped")
        if skipped is None:
            if case.find("failure") is not None or case.find("error") is not None:
                failed_nodes.append(nodeid)
                continue
            passed_nodes.append(nodeid)
            continue
        skips.append(
            {
                "suite": label,
                "nodeid": nodeid,
                "message": skipped.attrib.get("message", ""),
            }
        )
    return PytestEvidence(
        label=label,
        paths=tuple(paths),
        total=total,
        passed=len(passed_nodes),
        passed_nodes=tuple(passed_nodes),
        skips=tuple(skips),
        failed_nodes=tuple(failed_nodes),
    )


def _run_pytest_with_report(
    paths: Sequence[str],
    xml_path: Path,
    label: str,
) -> PytestEvidence:
    command = [sys.executable, "-m", "pytest", "-q", *paths, f"--junitxml={xml_path}"]
    _run_command(command)
    return _parse_pytest_evidence(xml_path, label=label, paths=paths)


def _windows_path_capability_evidence() -> dict[str, object]:
    """Prove the Windows reparse behavior gate and record fixture limitations."""
    with tempfile.TemporaryDirectory(prefix="omni-links-") as temporary:
        root = Path(temporary)
        target = root / "target"
        target.mkdir()
        (target / "inside.txt").write_text("inside", encoding="utf-8")
        junction = root / "junction"
        junction_result = subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(junction), str(target)),
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        junction_available = junction_result.returncode == 0 and junction.is_dir()
        if junction_available:
            junction.rmdir()

        directory_symlink = root / "directory-symlink"
        file_symlink = root / "file-symlink"
        symlink_errors: list[str] = []
        try:
            directory_symlink.symlink_to(target, target_is_directory=True)
            directory_symlink.unlink()
            directory_symlink_available = True
        except (OSError, NotImplementedError) as error:
            directory_symlink_available = False
            symlink_errors.append(f"directory: {error}")
        try:
            file_symlink.symlink_to(target / "inside.txt")
            file_symlink.unlink()
            file_symlink_available = True
        except (OSError, NotImplementedError) as error:
            file_symlink_available = False
            symlink_errors.append(f"file: {error}")

        hardlink = root / "hardlink"
        hardlink.hardlink_to(target / "inside.txt")
        hardlink_available = hardlink.is_file()
        if hardlink.exists():
            hardlink.unlink()

    return {
        "junction": {
            "available": junction_available,
            "command": "cmd /c mklink /J",
        },
        "directory_symlink": {
            "available": directory_symlink_available,
        },
        "file_symlink": {
            "available": file_symlink_available,
        },
        "hardlink": {"available": hardlink_available},
        "symlink_limitations": symlink_errors,
        "release_gate": (
            "The Session Restore path matrix requires file-symlink privilege; missing privilege "
            "fails the Windows gate. Junction, reparse, and hard-link evidence is retained too."
        ),
    }


@dataclass(frozen=True, slots=True)
class SkipRule:
    category: str
    node_patterns: tuple[str, ...]
    message_pattern: str
    platforms: tuple[str, ...] = ("windows",)


SKIP_RULES: Final[tuple[SkipRule, ...]] = (
    SkipRule(
        "covered-by-real-explicit-path-host-integration",
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_real_powershell_host_inspects_and_executes_canonical_cmdlet",
        ),
        r"^(?:powershell|pwsh) is not installed$",
    ),
    SkipRule(
        "covered-by-executed-windows-link-alternatives",
        _node_patterns(
            "tests/sessions/test_session_io_security.py",
            "test_session_load_rejects_linked_history_files",
            "test_session_persist_preserves_linked_history_files",
        )
        + _node_patterns(
            "tests/skills/test_catalog.py",
            "test_instruction_symlink_escape_is_excluded_when_links_are_available",
        )
        + _node_patterns(
            "tests/tools/core/test_directory_tools.py",
            "test_directory_symlink_roots_are_never_traversed",
        )
        + _node_patterns(
            "tests/tools/core/test_file_tools.py",
            "test_read_file_skill_root_escape_requires_confirmation",
        )
        + _node_patterns(
            "tests/tools/core/test_grep_tools.py",
            "test_grep_preserves_file_link_paths",
            "test_grep_does_not_traverse_an_explicit_directory_link",
            "test_grep_skips_file_links_outside_the_approved_root",
            "test_grep_reports_explicit_file_links_by_their_visible_paths",
        )
        + _node_patterns(
            "tests/restore/test_path_matrix.py",
            "test_restore_path_matrix_does_not_follow_a_retargeted_link",
        ),
        (
            r"^(?:file symbolic links are unavailable|file links unavailable|"
            r"directory symlinks unavailable|directory links unavailable|"
            r"symbolic links are unavailable on this host|"
            r"file symlink privilege unavailable on this host)(?::.*)?$"
        ),
    ),
)

REQUIRED_WINDOWS_ALTERNATIVE_NODES: Final[frozenset[str]] = frozenset(
    {
        "tests/sessions/test_session_io_security.py::test_session_load_rejects_linked_history_files[hardlink]",
        "tests/sessions/test_session_io_security.py::test_session_persist_preserves_linked_history_files[hardlink]",
        "tests/skills/test_catalog.py::test_skill_directory_reparse_escape_is_excluded_when_links_are_available",
        "tests/test_session_log.py::test_session_log_rejects_a_junction_logs_directory_without_stopping_work",
        "tests/test_windows_filesystem.py::test_require_owned_directory_rejects_junction_and_external_paths",
        "tests/test_windows_filesystem.py::test_require_owned_regular_file_rejects_directories_and_hard_links",
        "tests/test_workspace_state.py::test_initialization_rejects_junction_root",
        "tests/tools/core/test_directory_tools.py::test_directory_junction_roots_are_never_traversed",
        "tests/tools/core/test_exec_powershell_policy.py::test_powershell_read_through_workspace_reparse_point_confirms",
        "tests/tools/test_permission_file_matrix.py::test_linked_skill_root_and_missing_write_descendant_remain_external",
        "tests/terminal/test_conversation.py::test_coordinator_background_confirmation_uses_stable_modal_projection",
    }
)


def _classify_skip(skip: Mapping[str, str]) -> str:
    nodeid = skip.get("nodeid", "")
    message = skip.get("message", "")
    matches = tuple(
        rule.category
        for rule in SKIP_RULES
        if _platform() in rule.platforms
        and any(re.fullmatch(pattern, nodeid) is not None for pattern in rule.node_patterns)
        and re.fullmatch(rule.message_pattern, message) is not None
    )
    return matches[0] if len(matches) == 1 else "unclassified"


def _validate_skips(
    skips: Sequence[Mapping[str, str]],
    *,
    host_results: Sequence[Mapping[str, object]],
    path_evidence: Mapping[str, object],
    passed_nodes: Sequence[str],
) -> list[dict[str, str]]:
    if len(host_results) != 2 or {result.get("selector") for result in host_results} != set(RELEASE_SHELLS):
        raise ReleaseBlockedError("skip validation requires both PowerShell host integrations")
    for result in host_results:
        inspection = result.get("inspection")
        execution = result.get("canonical_execution")
        if (
            result.get("platform") != "windows"
            or result.get("family") != result.get("selector")
            or not isinstance(inspection, Mapping)
            or inspection.get("status") != "available"
            or inspection.get("syntax_uncertain") is not False
            or not isinstance(inspection.get("identity_count"), int)
            or inspection["identity_count"] < 1
            or not isinstance(execution, Mapping)
            or execution.get("exit_code") != 0
            or execution.get("timed_out") is not False
            or execution.get("matches_process_spec") is not True
            or result.get("full_access_dynamic_status") != "success"
        ):
            raise ReleaseBlockedError("skip validation requires complete PowerShell host evidence")
    junction = cast(Mapping[str, object], path_evidence["junction"])
    if junction.get("available") is not True:
        raise ReleaseBlockedError(
            "skip validation requires a working Windows junction capability"
        )
    hardlink = cast(Mapping[str, object], path_evidence["hardlink"])
    if hardlink.get("available") is not True:
        raise ReleaseBlockedError(
            "skip validation requires a working Windows hard-link capability"
        )
    file_symlink = cast(Mapping[str, object], path_evidence["file_symlink"])
    if file_symlink.get("available") is not True:
        raise ReleaseBlockedError(
            "skip validation requires a working Windows file symlink capability "
            "for the Session Restore path matrix"
        )
    missing_alternatives = sorted(REQUIRED_WINDOWS_ALTERNATIVE_NODES - set(passed_nodes))
    if missing_alternatives:
        raise RuntimeError(
            "required Windows alternative regression nodes did not pass: "
            + ", ".join(missing_alternatives)
        )
    missing_restore = sorted(RESTORE_PATH_MATRIX_NODES - set(passed_nodes))
    if missing_restore:
        raise RuntimeError(
            "required Session Restore path matrix nodes did not pass: " + ", ".join(missing_restore)
        )
    classified: list[dict[str, str]] = []
    for skip in skips:
        category = _classify_skip(skip)
        if category == "unclassified":
            raise RuntimeError(
                f"unclassified pytest skip: {skip.get('nodeid', '<unknown>')} "
                f"({skip.get('message', '')})"
            )
        classified.append(
            {
                "suite": skip.get("suite", "unknown"),
                "nodeid": skip.get("nodeid", "<unknown>"),
                "message": skip.get("message", ""),
                "classification": category,
            }
        )
    return classified


def _run_quality(host_results: Sequence[Mapping[str, object]] | None = None) -> dict[str, object]:
    actual_hosts = (
        list(host_results) if host_results is not None else run_host_integration(_selectors("both"))
    )
    path_evidence = _windows_path_capability_evidence()
    with tempfile.TemporaryDirectory(prefix="omni-release-quality-") as temporary:
        report_dir = Path(temporary)
        evidence_by_label: dict[str, PytestEvidence] = {}
        validated_skips: list[dict[str, str]] | None = None

        def run_suite(paths: Sequence[str], xml_path: Path, label: str) -> PytestEvidence:
            try:
                evidence = _run_pytest_with_report(paths, xml_path, label)
            except BaseException:
                if xml_path.is_file():
                    try:
                        evidence = _parse_pytest_evidence(xml_path, label=label, paths=paths)
                    except (OSError, ET.ParseError):
                        pass
                    else:
                        evidence_by_label[label] = evidence
                raise
            evidence_by_label[label] = evidence
            return evidence

        try:
            targeted = run_suite(TARGETED_TEST_PATHS, report_dir / "targeted.xml", "targeted")
            full = run_suite(("tests",), report_dir / "full.xml", "full")
            validated_skips = _validate_skips(
                (*targeted.skips, *full.skips),
                host_results=actual_hosts,
                path_evidence=path_evidence,
                passed_nodes=full.passed_nodes,
            )
            _run_command([sys.executable, "-m", "ruff", "check", "omni", "tests", "scripts"])
            _run_command(["git", "diff", "--check"])
            _run_command([sys.executable, "-m", "mypy", "omni", "tests", "scripts"])
            build_dir = report_dir / "build"
            build_dir.mkdir()
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    str(build_dir),
                ]
            )
            artifacts = sorted(path.name for path in build_dir.iterdir() if path.is_file())
        finally:
            recorder = _REPORT_CONTEXT.get()
            if recorder is not None:
                quality_evidence: dict[str, object] = {
                    "host_integration": actual_hosts,
                    "path_capability": path_evidence,
                    "pytest": {
                        label: evidence.to_dict() for label, evidence in evidence_by_label.items()
                    },
                }
                full_evidence = evidence_by_label.get("full")
                if full_evidence is not None:
                    quality_evidence["acceptance_matrix"] = build_acceptance_matrix(
                        full_evidence.passed_nodes
                    )
                if validated_skips is not None:
                    cast(dict[str, object], quality_evidence["pytest"])["validated_skips"] = (
                        validated_skips
                    )
                recorder.add_phase_evidence("quality", quality_evidence)
    return {
        "host_integration": actual_hosts,
        "path_capability": path_evidence,
        "pytest": {
            "targeted": targeted.to_dict(),
            "full": full.to_dict(),
            "validated_skips": validated_skips,
            "required_windows_alternatives": sorted(REQUIRED_WINDOWS_ALTERNATIVE_NODES),
            "required_restore_path_matrix": sorted(RESTORE_PATH_MATRIX_NODES),
        },
        "acceptance_matrix": build_acceptance_matrix(full.passed_nodes),
        "static": {
            "ruff_lint": "passed",
            "git_diff_check": "passed",
            "mypy": "passed",
        },
        "build": {"artifacts": artifacts},
    }


_ARTIFACT_SMOKE_PROGRAM: Final[str] = r"""
import hashlib
import json
import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import omni
import tomlkit

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigError, ConfigLoader


module_path = Path(omni.__file__).resolve()
environment_prefix = Path(sys.prefix).resolve()
source_root = Path(os.environ["OMNI_SOURCE_ROOT"]).resolve()
assert module_path.is_relative_to(environment_prefix)
assert not module_path.is_relative_to(source_root)
assert shutil.which("node") is None
assert shutil.which("npm") is None

asset_root = files("omni.web_assets")
manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["schema_version"] == 1
assert manifest["entry"] == "index.html"
for record in manifest["files"]:
    asset = (asset_root / record["path"]).read_bytes()
    assert len(asset) == record["bytes"]
    assert hashlib.sha256(asset).hexdigest() == record["sha256"]

with TemporaryDirectory(prefix="omni-wheel-config-") as temporary:
    home = Path(temporary) / "agent-home"
    loader = ConfigLoader(AgentHome(home))
    assert loader.ensure_default() is True
    source = loader.path.read_text(encoding="utf-8")

    missing = tomlkit.parse(source)
    for key in (
        "max_tool_result_chars",
        "max_iterations",
        "enable_skill_always_load",
        "compact_ratio",
        "permission_level",
        "exec_shell",
    ):
        del missing["runtime"][key]
    for key in ("batch_size", "schedule"):
        del missing["memory"][key]
    for route in missing["models"]["routes"].values():
        del route["reasoning_effort"]
    loader.path.write_text(tomlkit.dumps(missing), encoding="utf-8")
    configuration = loader.load_for_startup()
    assert configuration.runtime.max_tool_result_chars == 4096
    assert configuration.runtime.max_iterations == 50
    assert configuration.runtime.enable_skill_always_load is False
    assert configuration.runtime.compact_ratio == 0.9
    assert configuration.runtime.permission_level == "workspace-write"
    assert configuration.runtime.exec_shell == "auto"
    assert configuration.memory.batch_size == 10
    assert configuration.memory.schedule == "0 * * * *"
    assert all(route.reasoning_effort == "medium" for route in configuration.models.routes.values())
    assert loader.diagnostics == ()

    fallback = tomlkit.parse(source)
    fallback["runtime"]["permission_level"] = "wheel-secret"
    loader.path.write_text(tomlkit.dumps(fallback), encoding="utf-8")
    configuration = loader.load_for_startup()
    assert configuration.runtime.permission_level == "workspace-write"
    assert len(loader.diagnostics) == 1
    assert loader.diagnostics[0].field == "runtime.permission_level"
    assert "wheel-secret" not in loader.view().diagnostics_text()

    invalid_documents = (
        "[broken\nvalue = true\n",
        source.replace("[runtime]\n", "runtime = true\n", 1),
        source + "\n[models.providers.invalid]\nmodels = \"not-an-array\"\n",
        source.replace('provider_id = "openai-local"', "provider_id = []", 1),
    )
    for invalid in invalid_documents:
        loader.path.write_text(invalid, encoding="utf-8")
        try:
            loader.load()
        except ConfigError:
            pass
        else:
            raise AssertionError("invalid wheel configuration was accepted")

print(
    json.dumps(
        {
            "marker": "ARTIFACT_CONFIG_SMOKE_OK",
            "module_path": str(module_path),
            "environment_prefix": str(environment_prefix),
        },
        sort_keys=True,
    )
)
"""


def _artifact_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for inherited in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(inherited, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = os.pathsep.join(
        part
        for part in environment.get("PATH", "").split(os.pathsep)
        if "node" not in part.casefold() and "npm" not in part.casefold()
    )
    environment["OMNI_SOURCE_ROOT"] = str(ROOT)
    return environment


def _source_web_asset_bytes() -> dict[str, bytes]:
    asset_root = ROOT / "omni" / "web_assets"
    validator_module = cast(
        Any,
        import_module("scripts.validate_web_assets" if __package__ else "validate_web_assets"),
    )
    manifest = validator_module.validate_web_assets(asset_root)
    expected = {
        str(record["path"]): (asset_root / str(record["path"])).read_bytes()
        for record in manifest["files"]
    }
    expected["manifest.json"] = (asset_root / "manifest.json").read_bytes()
    return expected


def _assert_wheel_web_assets(wheel: Path, expected: Mapping[str, bytes]) -> None:
    prefix = "omni/web_assets/"
    with zipfile.ZipFile(wheel) as archive:
        names = {name.replace("\\", "/") for name in archive.namelist()}
        if any("/web/" in name or "node_modules/" in name for name in names):
            raise RuntimeError(f"wheel contains frontend source or node_modules: {wheel}")
        actual = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
        allowed = set(expected) | {"__init__.py"}
        if actual != allowed:
            raise RuntimeError(
                f"wheel Web asset members differ: expected {sorted(allowed)}, "
                f"found {sorted(actual)}"
            )
        for relative, contents in expected.items():
            if archive.read(prefix + relative) != contents:
                raise RuntimeError(f"wheel Web asset differs from source: {relative}")


def _assert_sdist_web_assets(sdist: Path, expected: Mapping[str, bytes]) -> None:
    marker = "/omni/web_assets/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = {member.name.replace("\\", "/"): member for member in archive.getmembers()}
        if any("/web/" in name or "node_modules/" in name for name in members):
            raise RuntimeError(f"sdist contains frontend source or node_modules: {sdist}")
        asset_members = {
            name.split(marker, 1)[1]: member
            for name, member in members.items()
            if marker in name and member.isfile()
        }
        allowed = set(expected) | {"__init__.py"}
        if set(asset_members) != allowed:
            raise RuntimeError(
                f"sdist Web asset members differ: expected {sorted(allowed)}, "
                f"found {sorted(asset_members)}"
            )
        for relative, contents in expected.items():
            extracted = archive.extractfile(asset_members[relative])
            if extracted is None or extracted.read() != contents:
                raise RuntimeError(f"sdist Web asset differs from source: {relative}")


def _extract_sdist(sdist: Path, destination: Path) -> Path:
    destination.mkdir()
    resolved_destination = destination.resolve()
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(resolved_destination):
                raise RuntimeError(f"sdist contains an unsafe path: {member.name}")
        archive.extractall(destination, filter="data")
    roots = tuple(path for path in destination.iterdir() if path.is_dir())
    if len(roots) != 1:
        raise RuntimeError(f"sdist extraction did not produce one source root: {roots}")
    return roots[0]


def _smoke_installed_wheel(wheel: Path, root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    venv_dir = root / "venv"
    _run_command([sys.executable, "-m", "venv", str(venv_dir)])
    scripts_dir = venv_dir / "Scripts"
    python = scripts_dir / "python.exe"
    entry_point = scripts_dir / "omni.exe"
    if not python.is_file():
        raise RuntimeError("wheel smoke virtual environment has no Python executable")
    environment = _artifact_environment()
    _run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            str(wheel),
        ],
        env=environment,
    )
    smoke_cwd = root / "smoke-cwd"
    smoke_cwd.mkdir()
    if not entry_point.is_file():
        raise RuntimeError("installed wheel did not create the omni console entry point")
    entry_result = _run_command(
        [str(entry_point), "--help"],
        cwd=smoke_cwd,
        env=environment,
        timeout=60,
    )
    if "Omni Personal Agent runtime" not in entry_result.stdout:
        raise RuntimeError("installed omni entry point did not start normally")
    result = subprocess.run(
        [str(python), "-c", _ARTIFACT_SMOKE_PROGRAM],
        cwd=smoke_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "installed wheel configuration smoke failed:\n" + result.stdout + result.stderr
        )
    try:
        smoke_payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "installed wheel smoke returned malformed evidence:\n" + result.stdout + result.stderr
        ) from error
    if smoke_payload.get("marker") != "ARTIFACT_CONFIG_SMOKE_OK":
        raise RuntimeError(
            "installed wheel smoke returned an unexpected marker:\n" + result.stdout + result.stderr
        )
    return {
        "wheel": wheel.name,
        "cwd": str(smoke_cwd),
        "marker": "ARTIFACT_CONFIG_SMOKE_OK",
        "module_path": smoke_payload["module_path"],
        "environment_prefix": smoke_payload["environment_prefix"],
        "entry_point": str(entry_point),
        "entry_point_help": "passed",
    }


def _run_distribution_validation() -> dict[str, object]:
    expected_assets = _source_web_asset_bytes()
    with tempfile.TemporaryDirectory(prefix="omni-release-distribution-") as temporary:
        root = Path(temporary)
        distribution_dir = root / "distribution"
        distribution_dir.mkdir()
        _run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(distribution_dir),
            ],
            env=_artifact_environment(),
        )
        wheels = tuple(distribution_dir.glob("omni-*.whl"))
        sdists = tuple(distribution_dir.glob("omni-*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(
                f"expected one direct wheel and sdist, found {len(wheels)} wheels and "
                f"{len(sdists)} sdists"
            )
        _assert_wheel_web_assets(wheels[0], expected_assets)
        _assert_sdist_web_assets(sdists[0], expected_assets)

        extracted_root = _extract_sdist(sdists[0], root / "extracted")
        rebuilt_dir = root / "rebuilt"
        rebuilt_dir.mkdir()
        _run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(rebuilt_dir),
                str(extracted_root),
            ],
            cwd=extracted_root,
            env=_artifact_environment(),
        )
        rebuilt_wheels = tuple(rebuilt_dir.glob("omni-*.whl"))
        if len(rebuilt_wheels) != 1:
            raise RuntimeError(f"expected one sdist-rebuilt wheel, found {len(rebuilt_wheels)}")
        _assert_wheel_web_assets(rebuilt_wheels[0], expected_assets)
        direct_smoke = _smoke_installed_wheel(wheels[0], root / "direct-install")
        rebuilt_smoke = _smoke_installed_wheel(rebuilt_wheels[0], root / "rebuilt-install")
        return {
            "wheel": wheels[0].name,
            "sdist": sdists[0].name,
            "rebuilt_wheel": rebuilt_wheels[0].name,
            "direct_install": direct_smoke,
            "rebuilt_install": rebuilt_smoke,
            "asset_manifest": "verified",
            "sdist_rebuild": "verified",
        }


def _run_artifact_smoke() -> dict[str, object]:
    distribution = _run_distribution_validation()
    direct_smoke = cast(dict[str, object], distribution["direct_install"])
    return {**direct_smoke, "distribution": distribution}


def _write_report(report_path: Path, payload: Mapping[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=tuple(phase.value for phase in ReleasePhase),
        default=ReleasePhase.ALL.value,
    )
    parser.add_argument(
        "--shell",
        choices=("powershell", "pwsh", "both"),
        default="both",
        help="PowerShell selector for host-integration; both is the release default.",
    )
    parser.add_argument("--report", type=Path, help="Write the JSON evidence report to this path.")
    return parser


def _selectors(value: str) -> tuple[str, ...]:
    return RELEASE_SHELLS if value == "both" else (value,)


def _run_named_phase(
    phase: ReleasePhase,
    shell_option: str,
    *,
    host_results: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if phase == ReleasePhase.COVERAGE:
        nodes = collect_pytest_nodes()
        return {"coverage": build_coverage_evidence(nodes).to_dict()}
    if phase == ReleasePhase.HOST_INTEGRATION:
        results = run_host_integration(_selectors(shell_option))
        return {"host_integration": results}
    if phase == ReleasePhase.QUALITY:
        return {"quality": _run_quality(host_results)}
    if phase == ReleasePhase.ARTIFACT_SMOKE:
        return {"artifact_smoke": _run_artifact_smoke()}
    raise AssertionError(f"unsupported named release phase: {phase}")


def _run_all_phases(
    shell_option: str,
    *,
    recorder: _ReportRecorder | None = None,
) -> dict[str, object]:
    """Run release phases in order and preserve fail-fast behavior for the caller."""
    report: dict[str, object] = {"phase": ReleasePhase.ALL.value}
    host_results: Sequence[Mapping[str, object]] | None = None
    for phase_name in PHASE_ORDER:
        phase = ReleasePhase(phase_name)
        if recorder is not None:
            recorder.begin_phase(phase_name)
        try:
            result = _run_named_phase(phase, shell_option, host_results=host_results)
        except BaseException as error:
            if recorder is not None:
                recorder.finish_phase(
                    phase_name,
                    status="blocked" if isinstance(error, ReleaseBlockedError) else "failed",
                    error=error,
                )
            raise
        if phase is ReleasePhase.HOST_INTEGRATION:
            value = result["host_integration"]
            if not isinstance(value, Sequence):
                raise RuntimeError("host integration returned an invalid result")
            host_results = cast(Sequence[Mapping[str, object]], value)
        if recorder is not None:
            recorder.finish_phase(phase_name, status="passed", result=result)
        report.update(result)
    return report


def _run_phase(phase: ReleasePhase, shell_option: str) -> dict[str, object]:
    if phase is ReleasePhase.ALL:
        return _run_all_phases(shell_option)
    return {"phase": phase.value, **_run_named_phase(phase, shell_option)}


def _run_reported_phase(
    phase: ReleasePhase,
    shell_option: str,
    recorder: _ReportRecorder,
) -> dict[str, object]:
    if phase is ReleasePhase.ALL:
        return _run_all_phases(shell_option, recorder=recorder)
    recorder.begin_phase(phase.value)
    try:
        result = _run_named_phase(phase, shell_option)
    except BaseException as error:
        recorder.finish_phase(
            phase.value,
            status="blocked" if isinstance(error, ReleaseBlockedError) else "failed",
            error=error,
        )
        raise
    recorder.finish_phase(phase.value, status="passed", result=result)
    return {"phase": phase.value, **result}


def main(argv: Sequence[str] | None = None) -> int:
    if not is_windows_host():
        print(WINDOWS_REQUIRED_ERROR, file=sys.stderr)
        return 1
    parser = _parser()
    arguments = parser.parse_args(argv)
    phase = ReleasePhase(arguments.phase)
    recorder = _ReportRecorder(arguments.phase, arguments.shell)
    context_token = _REPORT_CONTEXT.set(recorder)
    result: dict[str, object] | None = None
    failure: BaseException | None = None
    try:
        result = _run_reported_phase(phase, arguments.shell, recorder)
    except BaseException as error:
        failure = error
        print(f"release validation failed: {_redact_report_text(str(error))}", file=sys.stderr)
    report = recorder.build(payload=result, error=failure)
    report_write_error: BaseException | None = None
    if arguments.report is not None:
        try:
            _write_report(arguments.report, report)
        except BaseException as error:
            report_write_error = error
            report["report_write_error"] = _exception_payload(error)
            print(
                f"release validation report could not be written: {_redact_report_text(str(error))}",
                file=sys.stderr,
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    _REPORT_CONTEXT.reset(context_token)
    if failure is not None or report_write_error is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
