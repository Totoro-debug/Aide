"""Auditable Windows and POSIX release validation for Session Restore and tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, cast

from myclaw.agent.permission import PermissionSnapshot
from myclaw.agent.tools.core.exec import ExecTool
from myclaw.agent.tools.core.exec_host import (
    BashExecHost,
    PowerShellExecHost,
    ResolvedExecShell,
    resolve_exec_shell,
)
from myclaw.agent.tools.core.exec_policy import ExecAssessment
from myclaw.agent.tools.permission import PermissionContext
from myclaw.agent.tools.tool_gateway import ConfirmationRequest, ModelToolCall, ToolGateway

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_SECONDS: Final[int] = 1_800
POWERSHELL_FLAGS: Final[tuple[str, ...]] = (
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
)
RELEASE_SHELLS: Final[tuple[str, ...]] = ("powershell", "pwsh")
POSIX_CASES: Final[frozenset[str]] = frozenset(
    {
        "read-inside",
        "read-outside",
        "write-inside",
        "write-outside",
        "full-access-ordinary",
        "full-access-catastrophic",
        "identity-duplicate-path",
        "identity-single-hit",
    }
)


def _platform() -> Literal["windows", "posix"]:
    return "windows" if os.name == "nt" else "posix"


COLLECTION_PATHS: Final[tuple[str, ...]] = (
    "tests/tools/core/test_exec_host.py",
    "tests/tools/core/test_exec_bash_policy.py",
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

    def to_dict(self) -> dict[str, object]:
        return {
            "paths": list(self.paths),
            "total": self.total,
            "passed": self.passed,
            "skipped": len(self.skips),
            "skips": [dict(skip) for skip in self.skips],
        }


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
        57,
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_every_approved_powershell_candidate_has_a_direct_fixture",
            "test_every_cross_host_git_read_form_has_a_direct_powershell_fixture",
        )
        + _node_patterns(
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_fixed_read_candidates_run_inside_workspace",
            "test_bash_workspace_write_candidates_run_in_workspace",
            "test_bash_copy_and_move_classify_source_and_destination_roles",
            "test_bash_fixed_git_read_forms_run_when_delegation_is_safe",
            "test_bash_read_candidates_cross_real_inspection_and_policy",
            "test_bash_write_candidates_cross_real_inspection_and_policy",
        ),
    ),
    CoverageRule(
        "dynamic-complex",
        24,
        _node_patterns(
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_dynamic_construct",
            "test_bash_real_inspection_closes_grammar_and_path_bypasses",
        )
        + _node_patterns(
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
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_untrusted_identity_requires_one_confirmation",
            "test_bash_identity_rejects_plain_script_and_duplicate_path",
            "test_bash_identity_rejects_symlink",
        )
        + _node_patterns(
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
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_static_path_roles_follow_level_and_workspace_containment",
            "test_bash_copy_and_move_external_roles_require_confirmation",
            "test_bash_real_inspection_closes_grammar_and_path_bypasses",
        )
        + _node_patterns(
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
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_full_access_runs_parseable_dynamic_noncatastrophic_command",
            "test_bash_full_access_runs_parseable_noncatastrophic_dynamic_commands",
        )
        + _node_patterns(
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
        "powershell": "MYCLAW_POWERSHELL_PATH",
        "pwsh": "MYCLAW_PWSH_PATH",
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
        raise RuntimeError(f"{selector} executable was not found")
    typed_selector = cast(Literal["powershell", "pwsh"], selector)
    shell = resolve_exec_shell(
        typed_selector,
        platform="windows",
        which=lambda name: executable if name.casefold() == selector else None,
        environment=os.environ,
    )
    if not shell.available or shell.executable is None or shell.flags != POWERSHELL_FLAGS:
        raise RuntimeError(f"{selector} did not resolve to an available PowerShell host")
    return shell


async def _exercise_powershell_host(selector: str) -> dict[str, object]:
    shell = _resolve_windows_shell(selector)
    if shell.version is None:
        raise RuntimeError(f"{selector} did not report a version")
    if selector == "powershell" and shell.version[:2] != (5, 1):
        raise RuntimeError("powershell did not resolve to Windows PowerShell 5.1")
    if selector == "pwsh" and shell.version < (7,):
        raise RuntimeError("pwsh did not resolve to PowerShell 7 or newer")
    with tempfile.TemporaryDirectory(prefix=f"myclaw-{selector}-") as temporary:
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
            (ExecTool(workspace=workspace, host=host),),
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
            "family": shell.family,
            "version": list(shell.version),
            "executable": shell.executable,
            "flags": list(shell.flags),
            "cwd": str(workspace),
            "environment_keys": sorted(shell.env),
            "inspection": {
                "status": assessment.inspector_status,
                "syntax_uncertain": assessment.syntax_uncertain,
                "identity_count": len(assessment.command_identities),
            },
            "canonical_execution": {
                "exit_code": outcome.exit_code,
                "timed_out": outcome.timed_out,
                "observed_cwd": str(observed_cwd),
                "matches_process_spec": observed_cwd == spec.cwd,
            },
            "full_access_dynamic_status": dynamic_result.status,
        }


async def _exercise_bash_host() -> dict[str, object]:
    bash_path = shutil.which("bash")
    if bash_path is None:
        raise RuntimeError("bash executable was not found")
    with tempfile.TemporaryDirectory(prefix="myclaw-bash-release-") as temporary:
        root = Path(temporary).resolve()
        workspace = root / "workspace"
        workspace.mkdir()
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("cat", "touch", "rm"):
            source = shutil.which(name)
            if source is None:
                raise RuntimeError(f"{name} native executable was not found")
            shutil.copy2(Path(source).resolve(strict=True), bin_dir / name)

        environment = dict(os.environ)
        environment["PATH"] = str(bin_dir)
        shell = resolve_exec_shell(
            "auto",
            platform="posix",
            which=lambda name: bash_path if name == "bash" else None,
            environment=environment,
        )
        if not shell.available or shell.executable is None or shell.family != "bash":
            raise RuntimeError("bash did not resolve to an available POSIX host")
        host = BashExecHost(shell)
        spec = host.process_spec(workspace)
        if (
            spec.executable != shell.executable
            or spec.flags != shell.flags
            or spec.cwd != workspace
            or spec.environment != shell.environment
        ):
            raise RuntimeError("bash changed process inputs between resolution and execution")

        inside = workspace / "inside.txt"
        inside.write_text("inside-release-sentinel\n", encoding="utf-8")
        outside = root / "outside.txt"
        outside.write_text("outside-release-sentinel\n", encoding="utf-8")
        cases: list[dict[str, object]] = []

        async def check_case(
            name: str,
            level: Literal["read-only", "workspace-write", "full-access"],
            command: str,
            *,
            expected: Literal["direct", "confirm"],
            active_host: BashExecHost = host,
            identity_kind: str = "native",
            identity_hits: int = 1,
        ) -> None:
            assessment = await active_host.inspect(command, workspace)
            identities = assessment.command_identities
            if (
                assessment.uncertain
                or len(identities) != 1
                or identities[0].kind != identity_kind
                or identities[0].resolution_count != identity_hits
            ):
                raise RuntimeError(f"bash {name} executable identity did not match expectations")
            identity = identities[0]
            if identity_kind == "native" and identity.resolved != str(bin_dir / command.split()[0]):
                raise RuntimeError(f"bash {name} did not select the controlled native executable")
            gateway = ToolGateway._for_memory(
                (ExecTool(workspace=workspace, host=active_host),),
                permission_context=PermissionContext.from_snapshot(
                    PermissionSnapshot(level=level, exec_shell=active_host.resolved_shell),
                    workspace_root=workspace,
                ),
            )
            requests: list[ConfirmationRequest] = []

            async def decline(request: ConfirmationRequest) -> Literal["declined"]:
                requests.append(request)
                return "declined"

            result = await gateway.call(
                ModelToolCall(
                    id=f"release-bash-{name}",
                    name="exec",
                    arguments=json.dumps({"command": command}),
                ),
                confirmation=decline,
            )
            if expected == "direct":
                if (
                    result.status != "success"
                    or not result.content.startswith("Exit code: 0")
                    or result.confirmation is not None
                    or requests
                ):
                    raise RuntimeError(f"bash {name} did not execute directly and successfully")
            elif (
                result.status != "refused"
                or len(requests) != 1
                or result.confirmation is None
                or result.confirmation.decision != "declined"
            ):
                raise RuntimeError(f"bash {name} did not request and decline confirmation")
            if name in {"read-inside", "full-access-ordinary", "identity-single-hit"}:
                if "inside-release-sentinel" not in result.content:
                    raise RuntimeError(f"bash {name} did not return the workspace file")
            if name in {"read-outside", "write-outside"}:
                if "outside the Workspace" not in requests[0].reason:
                    raise RuntimeError(f"bash {name} was not classified as an external path")
                if "outside-release-sentinel" in result.content:
                    raise RuntimeError(f"bash {name} exposed the declined outside file")
            if (
                name == "full-access-catastrophic"
                and "catastrophic" not in requests[0].reason.lower()
            ):
                raise RuntimeError("bash catastrophic command was not classified as catastrophic")
            if name == "identity-duplicate-path" and "identity" not in requests[0].reason.lower():
                raise RuntimeError("bash repeated PATH was not classified as ambiguous identity")
            cases.append(
                {
                    "name": name,
                    "level": level,
                    "decision": expected,
                    "confirmation": "not-requested" if expected == "direct" else "declined",
                    "confirmation_reason": None if expected == "direct" else requests[0].reason,
                    "status": result.status,
                    "exit_code": 0 if expected == "direct" else None,
                    "identity": {
                        "kind": identity.kind,
                        "resolved": identity.resolved,
                        "resolution_count": identity.resolution_count,
                    },
                }
            )

        await check_case("read-inside", "read-only", "cat ./inside.txt", expected="direct")
        await check_case("read-outside", "read-only", "cat ../outside.txt", expected="confirm")
        if outside.read_text(encoding="utf-8") != "outside-release-sentinel\n":
            raise RuntimeError("declined outside read changed its sentinel")
        await check_case(
            "write-inside", "workspace-write", "touch ./created.txt", expected="direct"
        )
        if not (workspace / "created.txt").is_file():
            raise RuntimeError("direct workspace write did not create its file")
        await check_case(
            "write-outside", "workspace-write", "rm ../outside.txt", expected="confirm"
        )
        if outside.read_text(encoding="utf-8") != "outside-release-sentinel\n":
            raise RuntimeError("declined outside write changed its sentinel")
        await check_case(
            "full-access-ordinary", "full-access", "cat ./inside.txt", expected="direct"
        )
        await check_case(
            "full-access-catastrophic",
            "full-access",
            f"rm -rf {shlex.quote(str(workspace))}",
            expected="confirm",
        )
        if not inside.is_file():
            raise RuntimeError("declined catastrophic command changed the workspace")

        duplicate_environment = dict(shell.env)
        duplicate_environment["PATH"] = os.pathsep.join((str(bin_dir), str(bin_dir)))
        duplicate_shell = resolve_exec_shell(
            "auto",
            platform="posix",
            which=lambda name: bash_path if name == "bash" else None,
            environment=duplicate_environment,
        )
        await check_case(
            "identity-duplicate-path",
            "read-only",
            "cat ./inside.txt",
            expected="confirm",
            active_host=BashExecHost(duplicate_shell),
            identity_kind="ambiguous",
            identity_hits=2,
        )
        await check_case("identity-single-hit", "read-only", "cat ./inside.txt", expected="direct")
        if {case["name"] for case in cases} != POSIX_CASES:
            raise RuntimeError("bash host evidence did not cover every required case")
        return {
            "selector": "auto",
            "platform": "posix",
            "family": "bash",
            "status": "passed",
            "executable": shell.executable,
            "flags": list(shell.flags),
            "cwd": str(workspace),
            "cases": cases,
        }


def run_host_integration(selectors: Sequence[str]) -> list[dict[str, object]]:
    """Run real inspection and execution on the current release host."""
    if _platform() == "posix":
        if tuple(selectors) != ("auto",):
            raise RuntimeError("POSIX host integration requires the default Bash selector")
        return [asyncio.run(_exercise_bash_host())]
    return [asyncio.run(_exercise_powershell_host(selector)) for selector in selectors]


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.list2cmdline([str(part) for part in command])
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
        raise RuntimeError(f"command timed out after {timeout}s: {rendered}") from error
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        if len(output) > 12_000:
            output = output[-12_000:]
        suffix = f"\n{output}" if output else ""
        raise RuntimeError(f"command failed with exit {result.returncode}: {rendered}{suffix}")
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
    total = 0
    for case in root.iter("testcase"):
        total += 1
        nodeid = _junit_nodeid(case)
        skipped = case.find("skipped")
        if skipped is None:
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
    if os.name != "nt":
        raise RuntimeError("Windows path capability evidence requires a Windows runner")
    with tempfile.TemporaryDirectory(prefix="myclaw-links-") as temporary:
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

    if not junction_available or not hardlink_available:
        raise RuntimeError("Windows reparse/path capability evidence did not pass")
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
        "waived-posix-host-scope",
        _node_patterns(
            "tests/tools/core/test_exec_bash_policy.py",
            "test_real_posix_bash_inspect_policy_execute_smoke",
        ),
        r"^requires a real POSIX production host$",
    ),
    SkipRule(
        "waived-posix-release-host-scope",
        _node_patterns(
            "tests/test_release_validation.py",
            "test_real_posix_release_host_covers_permission_and_identity_cases",
        ),
        r"^requires a real POSIX release host$",
    ),
    SkipRule(
        "covered-by-headless-windows-terminal-suite",
        _node_patterns(
            "tests/test_cli.py",
            "test_installed_wheel_terminal_conversation_pseudo_terminal_smoke",
        ),
        r"^The Windows Python runtime has no termios/pty harness; use the Windows Terminal matrix\.$",
    ),
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
            "tests/tools/core/test_exec_bash_policy.py",
            "test_bash_identity_rejects_symlink",
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
    SkipRule(
        "waived-windows-junction-scope",
        _node_patterns(
            "tests/tools/core/test_directory_tools.py",
            "test_directory_junction_roots_are_never_traversed",
        ),
        r"^Windows junction behavior$",
        ("posix",),
    ),
    SkipRule(
        "waived-native-windows-path-scope",
        _node_patterns(
            "tests/test_atomic_files.py",
            "test_path_for_io_normalizes_windows_local_and_unc_paths",
            "test_path_for_io_preserves_existing_windows_extended_path",
            "test_atomic_create_and_replace_use_windows_extended_paths",
        )
        + _node_patterns(
            "tests/test_host_filesystem.py",
            "test_windows_host_filesystem_prepares_local_and_unc_io_paths",
            "test_windows_host_filesystem_accepts_an_owned_directory",
            "test_windows_host_filesystem_rejects_redirected_or_external_directory",
            "test_windows_host_filesystem_accepts_an_owned_regular_file",
            "test_windows_host_filesystem_rejects_an_open_file_with_mismatched_path",
        )
        + _node_patterns(
            "tests/test_session_log.py",
            "test_session_log_rejects_a_junction_logs_directory_without_stopping_work",
            "test_session_log_preserves_windows_acl_inheritance",
        )
        + _node_patterns(
            "tests/test_windows_filesystem.py",
            "test_require_owned_directory_returns_normalized_owned_path",
            "test_require_owned_directory_rejects_junction_and_external_paths",
            "test_require_owned_regular_file_returns_normalized_owned_path",
            "test_require_owned_regular_file_rejects_directories_and_hard_links",
        )
        + _node_patterns(
            "tests/test_workspace_state.py",
            "test_windows_drive_workspace_path_has_the_accepted_identity",
            "test_unc_workspace_path_has_the_accepted_identity",
            "test_windows_workspace_path_is_lexically_normalized",
            "test_relative_pure_windows_workspace_path_is_rejected",
            "test_initialization_rejects_case_and_junction_aliases_of_agent_home",
            "test_initialization_rejects_junction_root",
            "test_initialization_rejects_external_memory_directory_alias",
            "test_initialization_rejects_external_sessions_directory_alias",
        ),
        r"^requires native Windows paths$",
        ("posix",),
    ),
    SkipRule(
        "waived-windows-path-case-scope",
        _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_windows_path_case_uses_host_case_insensitive_containment",
        ),
        r"^Windows host path semantics$",
        ("posix",),
    ),
    SkipRule(
        "waived-windows-drive-scope",
        _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_windows_different_drive_is_external",
        ),
        r"^Windows drive semantics$",
        ("posix",),
    ),
    SkipRule(
        "waived-windows-unc-scope",
        _node_patterns(
            "tests/tools/test_permission_file_matrix.py",
            "test_windows_reachable_unc_path_is_classified_by_real_host_semantics",
        ),
        r"^Windows UNC semantics$",
        ("posix",),
    ),
    SkipRule(
        "waived-powershell-host-scope",
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_real_powershell_host_inspects_and_executes_canonical_cmdlet",
        ),
        r"^(?:powershell|pwsh) is not installed$",
        ("posix",),
    ),
    SkipRule(
        "waived-pwsh-inspection-scope",
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_real_pwsh_inspection_does_not_autoload_an_untrusted_module",
        ),
        r"^pwsh is not installed$",
        ("posix",),
    ),
    SkipRule(
        "waived-windows-powershell-path-scope",
        _node_patterns(
            "tests/tools/core/test_exec_powershell_policy.py",
            "test_windows_powershell_51_canonical_workspace_read_executes_directly",
            "test_every_approved_powershell_candidate_has_a_direct_fixture",
            "test_powershell_path_roles_follow_level_and_canonical_containment",
            "test_every_cross_host_git_read_form_has_a_direct_powershell_fixture",
        )
        + _node_patterns(
            "tests/tools/test_permission_contract.py",
            "test_exec_post_grammar_policy_preserves_shell_parity",
        ),
        r"^requires native Windows PowerShell paths$",
        ("posix",),
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
REQUIRED_POSIX_SMOKE_NODES: Final[frozenset[str]] = frozenset(
    {
        "tests/tools/core/test_exec_bash_policy.py::test_real_posix_bash_inspect_policy_execute_smoke",
        "tests/test_release_validation.py::test_real_posix_release_host_covers_permission_and_identity_cases",
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
    if _platform() == "windows":
        if {result.get("selector") for result in host_results} != set(RELEASE_SHELLS):
            raise RuntimeError("skip validation requires both PowerShell host integrations")
        junction = cast(Mapping[str, object], path_evidence["junction"])
        if junction.get("available") is not True:
            raise RuntimeError("skip validation requires a working Windows junction capability")
        hardlink = cast(Mapping[str, object], path_evidence["hardlink"])
        if hardlink.get("available") is not True:
            raise RuntimeError("skip validation requires a working Windows hard-link capability")
        file_symlink = cast(Mapping[str, object], path_evidence["file_symlink"])
        if file_symlink.get("available") is not True:
            raise RuntimeError(
                "skip validation requires a working Windows file symlink capability "
                "for the Session Restore path matrix"
            )
        missing_alternatives = sorted(REQUIRED_WINDOWS_ALTERNATIVE_NODES - set(passed_nodes))
        if missing_alternatives:
            raise RuntimeError(
                "required Windows alternative regression nodes did not pass: "
                + ", ".join(missing_alternatives)
            )
    else:
        if (
            len(host_results) != 1
            or host_results[0].get("selector") != "auto"
            or host_results[0].get("platform") != "posix"
            or host_results[0].get("family") != "bash"
            or host_results[0].get("status") != "passed"
            or {
                case.get("name")
                for case in cast(Sequence[Mapping[str, object]], host_results[0].get("cases", ()))
            }
            != POSIX_CASES
        ):
            raise RuntimeError("skip validation requires complete real POSIX Bash host evidence")
        missing_smoke = sorted(REQUIRED_POSIX_SMOKE_NODES - set(passed_nodes))
        if missing_smoke:
            raise RuntimeError(
                "required POSIX smoke nodes did not pass: " + ", ".join(missing_smoke)
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
    path_evidence = _windows_path_capability_evidence() if _platform() == "windows" else {}
    with tempfile.TemporaryDirectory(prefix="myclaw-release-quality-") as temporary:
        report_dir = Path(temporary)
        targeted = _run_pytest_with_report(
            TARGETED_TEST_PATHS,
            report_dir / "targeted.xml",
            "targeted",
        )
        full = _run_pytest_with_report(
            ("tests",),
            report_dir / "full.xml",
            "full",
        )
        skips = _validate_skips(
            (*targeted.skips, *full.skips),
            host_results=actual_hosts,
            path_evidence=path_evidence,
            passed_nodes=full.passed_nodes,
        )
        _run_command([sys.executable, "-m", "ruff", "check", "myclaw", "tests", "scripts"])
        _run_command(["git", "diff", "--check"])
        _run_command([sys.executable, "-m", "mypy", "myclaw", "tests", "scripts"])
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
    return {
        "host_integration": actual_hosts,
        "path_capability": path_evidence,
        "pytest": {
            "targeted": targeted.to_dict(),
            "full": full.to_dict(),
            "validated_skips": skips,
            "required_windows_alternatives": sorted(REQUIRED_WINDOWS_ALTERNATIVE_NODES),
            "required_posix_smoke": sorted(REQUIRED_POSIX_SMOKE_NODES),
            "required_restore_path_matrix": sorted(RESTORE_PATH_MATRIX_NODES),
        },
        "static": {
            "ruff_lint": "passed",
            "git_diff_check": "passed",
            "mypy": "passed",
        },
        "build": {"artifacts": artifacts},
    }


_ARTIFACT_SMOKE_PROGRAM: Final[str] = r"""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import myclaw
import tomlkit

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader


module_path = Path(myclaw.__file__).resolve()
environment_prefix = Path(sys.prefix).resolve()
source_root = Path(os.environ["MYCLAW_SOURCE_ROOT"]).resolve()
assert module_path.is_relative_to(environment_prefix)
assert not module_path.is_relative_to(source_root)

with TemporaryDirectory(prefix="myclaw-wheel-config-") as temporary:
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


def _run_artifact_smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="myclaw-release-artifact-") as temporary:
        root = Path(temporary)
        wheel_dir = root / "wheel"
        wheel_dir.mkdir()
        _run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(wheel_dir),
            ]
        )
        wheels = tuple(wheel_dir.glob("myclaw-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")
        venv_dir = root / "venv"
        _run_command([sys.executable, "-m", "venv", str(venv_dir)])
        scripts_dir = venv_dir / ("Scripts" if _platform() == "windows" else "bin")
        python = scripts_dir / ("python.exe" if _platform() == "windows" else "python")
        entry_point = scripts_dir / ("myclaw.exe" if _platform() == "windows" else "myclaw")
        if not python.is_file():
            raise RuntimeError("wheel smoke virtual environment has no Python executable")
        environment = dict(os.environ)
        for inherited in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            environment.pop(inherited, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["MYCLAW_SOURCE_ROOT"] = str(ROOT)
        _run_command(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                str(wheels[0]),
            ],
            env=environment,
        )
        smoke_cwd = root / "smoke-cwd"
        smoke_cwd.mkdir()
        if not entry_point.is_file():
            raise RuntimeError("installed wheel did not create the myclaw console entry point")
        entry_result = _run_command(
            [str(entry_point), "--help"],
            cwd=smoke_cwd,
            env=environment,
            timeout=60,
        )
        if "MyClaw Personal Agent runtime" not in entry_result.stdout:
            raise RuntimeError("installed myclaw entry point did not start normally")
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
                "installed wheel smoke returned malformed evidence:\n"
                + result.stdout
                + result.stderr
            ) from error
        if smoke_payload.get("marker") != "ARTIFACT_CONFIG_SMOKE_OK":
            raise RuntimeError(
                "installed wheel smoke returned an unexpected marker:\n"
                + result.stdout
                + result.stderr
            )
        return {
            "wheel": wheels[0].name,
            "cwd": str(smoke_cwd),
            "marker": "ARTIFACT_CONFIG_SMOKE_OK",
            "module_path": smoke_payload["module_path"],
            "environment_prefix": smoke_payload["environment_prefix"],
            "entry_point": str(entry_point),
            "entry_point_help": "passed",
        }


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
    if _platform() == "posix":
        if value != "both":
            raise ValueError("POSIX release validation uses Bash; --shell must remain both")
        return ("auto",)
    return RELEASE_SHELLS if value == "both" else (value,)


def _run_phase(phase: ReleasePhase, shell_option: str) -> dict[str, object]:
    if phase == ReleasePhase.COVERAGE:
        nodes = collect_pytest_nodes()
        return {"phase": phase.value, "coverage": build_coverage_evidence(nodes).to_dict()}
    if phase == ReleasePhase.HOST_INTEGRATION:
        results = run_host_integration(_selectors(shell_option))
        return {"phase": phase.value, "host_integration": results}
    if phase == ReleasePhase.QUALITY:
        return {"phase": phase.value, "quality": _run_quality()}
    if phase == ReleasePhase.ARTIFACT_SMOKE:
        return {"phase": phase.value, "artifact_smoke": _run_artifact_smoke()}

    nodes = collect_pytest_nodes()
    hosts = run_host_integration(_selectors(shell_option))
    quality = _run_quality(hosts)
    return {
        "phase": ReleasePhase.ALL.value,
        "coverage": build_coverage_evidence(nodes).to_dict(),
        "host_integration": hosts,
        "quality": quality,
        "artifact_smoke": _run_artifact_smoke(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if _platform() == "posix" and arguments.shell != "both":
        parser.error("POSIX release validation uses Bash; --shell must remain both")
    try:
        report = _run_phase(ReleasePhase(arguments.phase), arguments.shell)
    except (AssertionError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 1
    if arguments.report is not None:
        _write_report(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
